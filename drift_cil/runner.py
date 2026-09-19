import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
from importlib.metadata import version
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml
from .data import DataBundle, IndexedImages
from .model import CLIPClassifier, TinyClassifier, trainable_state, restore_trainable
from .memory import Memory, memory_loss, fisher_scores
from .optim import MatrixSteps, SpectralProposal, Transaction, accept_step

METHODS = ("adamw","muon","scalar","replay","drift")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_model(cfg,data,device):
    if cfg.get("synthetic",False):
        return TinyClassifier(len(data.classes),cfg["rank"]).to(device)
    return CLIPClassifier(cfg["model_id"],cfg.get("model_revision","main"),data.classes,
        device,cfg["rank"],cfg["alpha"],cfg["prompt"],cfg.get("gradient_checkpointing",True),
        cfg.get("model_local_files_only",False))


@torch.no_grad()
def evaluate(model,data,completed,device,batch_size,split="validation"):
    if completed == 0:
        return {"accuracy":None,"per_task":[]}
    train_mode = model.training; model.eval()
    seen = sum(data.task_classes[:completed],[])
    classes = torch.as_tensor(seen,device=device)
    test = split == "test"
    targets = data.test_targets if test else data.targets
    all_ids = np.arange(len(targets)) if test else data.val_indices
    scores = []
    correct_total = count_total = 0
    for task in range(completed):
        ids = all_ids[np.isin(targets[all_ids],data.task_classes[task])]
        correct = 0
        for start in range(0,len(ids),batch_size):
            x,y = data.batch(ids[start:start+batch_size],device,test=test)
            # Every image competes against ALL seen classes; no task-ID oracle.
            prediction = classes[model(x).float()[:,classes].argmax(-1)]
            correct += int((prediction == y).sum())
        scores.append(correct / max(1,len(ids)))
        correct_total += correct; count_total += len(ids)
    model.train(train_mode)
    return {"accuracy":correct_total/max(1,count_total),"per_task":scores,"split":split}


def schedule(step,total,warmup,base):
    if step < warmup:
        return base * (step+1)/max(1,warmup)
    progress = min(1.,(step-warmup)/max(1,total-warmup))
    return base * .5 * (1 + math.cos(math.pi*progress))


def atomic_save(obj,path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    torch.save(obj,tmp)
    os.replace(tmp,path)


def run(cfg, output, method, task_limit=None, max_updates=None, resume=None, print_every=10):
    output = Path(output)
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if (output/"manifest.json").exists() and not resume:
        raise FileExistsError(f"{output} already contains a run; choose another output or --resume")
    output.mkdir(parents=True,exist_ok=True)
    checkpoint = torch.load(resume,map_location="cpu",weights_only=False) if resume else None
    if checkpoint:
        if checkpoint["method"] != method:
            raise ValueError("Cannot resume with a different method")
        saved = checkpoint["config"]
        runtime_keys = {"device","data_root","download"}
        for key in sorted((set(cfg)|set(saved)) - runtime_keys - {"model_revision"}):
            if cfg.get(key) != saved.get(key):
                raise ValueError(f"Resume config differs at {key}; keep the original config")
        # Keep the checkpoint's resolved revision, but honor runtime location/device.
        cfg = dict(saved,**{key:cfg[key] for key in runtime_keys if key in cfg})
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg.get("cpu_threads",4))
    device_name = cfg.get("device","auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run check_env.py before this experiment")
    if device.type == "cpu" and not cfg.get("synthetic",False):
        print("WARNING: real CLIP on CPU is supported but will be slow",flush=True)
    micro, effective = cfg["micro_batch"], cfg["effective_batch"]
    if micro <= 0 or effective <= 0 or effective % micro:
        raise ValueError("effective_batch must be divisible by positive micro_batch")
    for key in ("epochs","tasks","rank","memory_micro_batch","eval_batch","fisher_samples"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if print_every <= 0 or (task_limit is not None and task_limit <= 0):
        raise ValueError("print_every and task_limit must be positive")
    if max_updates is not None and max_updates <= 0:
        raise ValueError("max_updates must be positive")
    if cfg["memory_size"] < 0 or cfg["warmup_steps"] < 0:
        raise ValueError("memory_size and warmup_steps must be nonnegative")
    if cfg["lr"] <= 0 or cfg["damping"] < 0:
        raise ValueError("lr must be positive and damping nonnegative")
    accumulation = effective // micro
    print(json.dumps({"event":"loading_data","synthetic":cfg.get("synthetic",False),
                      "data_root":cfg.get("data_root")}),flush=True)
    data = DataBundle(cfg)
    print(json.dumps({"event":"loading_model","model":cfg["model_id"],
                      "device":str(device),"debug_subset":data.debug_subset}),flush=True)
    model = make_model(cfg,data,device)
    cfg = dict(cfg,model_revision=model.revision)
    parameters = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not parameters or any(p.ndim != 2 for n,p in parameters):
        raise ValueError("Expected only trainable LoRA matrix factors")
    stepper = MatrixSteps(parameters,beta=cfg.get("beta",.95),backend=cfg.get("backend","ns5"),
                          ns_steps=cfg.get("ns_steps",5),shape_scale=cfg.get("shape_scale",True))
    adam = torch.optim.AdamW([p for n,p in parameters],lr=cfg["lr"],
             weight_decay=cfg.get("weight_decay",0.)) if method == "adamw" else None
    capacity = cfg["memory_size"] if method in ("scalar","replay","drift") else 0
    memory = Memory(capacity,cfg["seed"])
    label_rng = torch.Generator(device=device).manual_seed(cfg["seed"]+191)
    start_task,update,stage_metrics = 0,0,[]
    if checkpoint:
        restore_trainable(model,checkpoint["adapters"])
        stepper.load_state_dict(checkpoint["momentum"])
        if adam is not None: adam.load_state_dict(checkpoint["adam"])
        memory.load_state_dict(checkpoint["memory"])
        start_task,update = checkpoint["completed_tasks"],checkpoint["update"]
        stage_metrics = checkpoint["stage_metrics"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        label_rng.set_state(checkpoint["label_rng"])
    limit = min(task_limit or cfg["tasks"],cfg["tasks"])
    total_updates = sum(math.ceil(len(data.pool(t))/effective)*cfg["epochs"] for t in range(cfg["tasks"]))
    if cfg["warmup_steps"] >= total_updates:
        raise ValueError("warmup_steps must be smaller than the planned full-run update count")
    use_amp = device.type == "cuda" and cfg.get("precision","bf16") == "bf16"
    if use_amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unsupported on this GPU; set precision: fp32")
    def amp():
        return torch.autocast("cuda",dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    manifest = {"config":cfg,"method":method,"class_order":data.order,
        "task_classes":data.task_classes,"train_examples":len(data.train_indices),
        "validation_examples":len(data.val_indices),"trainable_parameters":sum(p.numel() for n,p in parameters),
        "parameter_shapes":{n:list(p.shape) for n,p in parameters},"torch":torch.__version__,
        "python":platform.python_version(),
        "packages":{name:version(name) for name in ("transformers","torchvision","numpy","scipy")},
        "code_sha256":hashlib.sha256(b"".join(p.read_bytes() for p in
            sorted(Path(__file__).parent.glob("*.py")))).hexdigest(),
        "device":str(device),"gpu":torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "planned_full_run_updates":total_updates,"task_limit":limit,"synthetic":data.synthetic,
        "debug_subset":data.debug_subset,
        "protocol_status":"research reimplementation; NOT an exact published-number reproduction",
        "memory_head":"fixed all-class frozen head; evaluation uses seen classes only",
        "ns_precision":"FP32","checkpoint_resume":"completed-task boundary only"}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    def save(completed,path):
        atomic_save({"config":cfg,"method":method,"adapters":trainable_state(model),
            "momentum":stepper.state_dict(),"adam":adam.state_dict() if adam else None,
            "memory":memory.state_dict(),"completed_tasks":completed,"update":update,
            "stage_metrics":stage_metrics,"torch_rng":torch.get_rng_state(),
            "cuda_rng":torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
            "label_rng":label_rng.get_state()},path)
    if not resume:
        save(0,output/"last.pt")
    else:
        # Resume is from a task boundary: preserve, then remove partial later-task logs.
        log_path = output/"steps.jsonl"
        if log_path.exists():
            lines = log_path.read_text().splitlines()
            completed_lines = [line for line in lines if json.loads(line)["task"] < start_task]
            (output/f"steps_before_resume_{int(time.time())}.jsonl").write_text("\n".join(lines)+"\n")
            log_path.write_text("\n".join(completed_lines)+("\n" if completed_lines else ""))
    print(json.dumps({"event":"start","method":method,"device":str(device),"trainable":manifest["trainable_parameters"],
                      "accumulation":accumulation,"start_task":start_task,"task_limit":limit}),flush=True)
    measured_step_seconds=[]; status="pilot_complete" if limit<cfg["tasks"] else "complete"
    if data.debug_subset: status="debug_"+status
    start_time=time.perf_counter()
    stop=False
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for task in range(start_task,limit):
        classes = torch.as_tensor(sum(data.task_classes[:task+1],[]),device=device)
        mapping = torch.full((len(data.classes),),-1,device=device,dtype=torch.long)
        mapping[classes] = torch.arange(len(classes),device=device)
        pool = data.pool(task)
        for epoch in range(cfg["epochs"]):
            gen = torch.Generator().manual_seed(cfg["seed"]+task*100003+epoch)
            loader = DataLoader(IndexedImages(data,pool),batch_size=micro,shuffle=True,generator=gen,
                num_workers=cfg.get("workers",0),pin_memory=device.type=="cuda",drop_last=False)
            model.train(); model.zero_grad(set_to_none=True)
            samples_in_update=0;loss_sum=0.; synchronize(device); step_start=time.perf_counter()
            for mb,(x,y) in enumerate(loader):
                x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
                with amp():
                    loss=F.cross_entropy(model(x)[:,classes].float(),mapping[y],reduction="sum")
                (loss/effective).backward()
                samples_in_update+=len(y);loss_sum+=float(loss.detach())
                last_micro=mb==len(loader)-1
                if (mb+1)%accumulation and not last_micro: continue
                if samples_in_update != effective:
                    for n,p in parameters:
                        if p.grad is not None: p.grad.mul_(effective/samples_in_update)
                old_reads={"gradient":0,"fisher":0,"gate":0,"replay":0}
                if method=="replay" and len(memory):
                    ids=memory.sample(cfg.get("replay_samples",32))
                    _,old_grad=memory_loss(model,data,ids,device,cfg["memory_micro_batch"],True,parameters,
                                           head_classes=classes.tolist())
                    for n,p in parameters:
                        if p.grad is None: p.grad=torch.zeros_like(p)
                        p.grad.add_(old_grad[n],alpha=cfg.get("replay_weight",1.))
                    old_reads["replay"]=len(ids)
                norm=torch.nn.utils.clip_grad_norm_([p for n,p in parameters],cfg.get("clip_grad",1.),error_if_nonfinite=True)
                lr=schedule(update,total_updates,cfg["warmup_steps"],cfg["lr"])
                diagnostics={"accepted_scale":1.,"accept_status":"unconstrained"}
                if adam:
                    for group in adam.param_groups: group["lr"]=lr
                    adam.step()
                else:
                    new_grad={n:p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for n,p in parameters}
                    directions=stepper.directions(lr)
                    if method in ("scalar","drift") and len(memory):
                        ids=memory.indices()
                        if method=="drift":
                            before,old_grad=memory_loss(model,data,ids,device,cfg["memory_micro_batch"],True,parameters)
                            old_reads["gradient"]=len(ids)
                            proposal=SpectralProposal(parameters,directions)
                            if proposal.size:
                                fi=memory.sample(cfg["fisher_samples"])
                                scores=fisher_scores(model,data,fi,device,proposal,label_rng)
                                old_reads["fisher"]=len(fi)
                                b,a=proposal.coordinates(new_grad),proposal.coordinates(old_grad)
                                z,diagnostics=proposal.allocate(b,a,scores,
                                    memory.anchor+cfg["delta"]-before,cfg["damping"],cfg.get("z_lower",0.),
                                    cfg.get("solver_maxiter",500))
                                directions=proposal.directions(z)
                                diagnostics["spectral_modes"]=proposal.size
                                diagnostics["negative_b_modes"]=int(np.sum(b<0))
                        else:
                            before=memory_loss(model,data,ids,device,cfg["memory_micro_batch"])
                            old_reads["gate"]+=len(ids)
                        def gate_loss():
                            old_reads["gate"]+=len(ids)
                            return memory_loss(model,data,ids,device,cfg["memory_micro_batch"])
                        transaction=Transaction(parameters,directions)
                        accepted=accept_step(transaction,gate_loss,before,memory.anchor+cfg["delta"],
                            max_backtracks=cfg.get("max_backtracks",8),atol=cfg.get("gate_tolerance",1e-6))
                        diagnostics.update(accepted)
                        diagnostics.update(memory_before=before,memory_anchor=memory.anchor,
                                           budget_ceiling=memory.anchor+cfg["delta"])
                    else:
                        Transaction(parameters,directions).apply()
                synchronize(device)
                seconds=time.perf_counter()-step_start
                measured_step_seconds.append(seconds)
                record={"task":task,"epoch":epoch,"update":update,"method":method,"lr":lr,
                    "new_loss":loss_sum/samples_in_update,"new_samples":samples_in_update,
                    "grad_norm":float(norm),"seconds":seconds,"old_reads":old_reads,
                    "memory_size":len(memory),**diagnostics}
                with (output/"steps.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps(record,allow_nan=False)+"\n")
                if update%print_every==0: print(json.dumps(record),flush=True)
                update+=1;model.zero_grad(set_to_none=True)
                samples_in_update=0;loss_sum=0.; step_start=time.perf_counter()
                if max_updates is not None and update>=max_updates:
                    # This is a profiling stop, not a completed task/benchmark.
                    stop=True;status="profile_stopped";break
            if stop: break
        if stop: break
        result=evaluate(model,data,task+1,device,cfg["eval_batch"],"validation")
        stage_metrics.append(result)
        memory.update(model,data,task,device,cfg["memory_micro_batch"])
        save(task+1,output/f"task_{task+1:02d}.pt")
        save(task+1,output/"last.pt")
        print(json.dumps({"event":"task_complete","task":task,**result}),flush=True)
    completed=len(stage_metrics)
    last=stage_metrics[-1]["accuracy"] if completed else None
    avg=float(np.mean([s["accuracy"] for s in stage_metrics])) if completed else None
    bwt=None
    if completed>1:
        bwt=float(np.mean([stage_metrics[-1]["per_task"][i]-stage_metrics[i]["per_task"][i] for i in range(completed-1)]))
    summary={"status":status,"synthetic":data.synthetic,"debug_subset":data.debug_subset,
        "method":method,"completed_tasks":completed,
        "validation_avg":avg,"validation_last":last,"validation_bwt":bwt,"stages":stage_metrics,
        "updates":update,"elapsed_seconds":time.perf_counter()-start_time,
        "median_step_seconds":float(np.median(measured_step_seconds)) if measured_step_seconds else None,
        "peak_allocated_gib":torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else None,
        "peak_reserved_gib":torch.cuda.max_memory_reserved(device)/2**30 if device.type=="cuda" else None}
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"event":"summary",**summary}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description="CLIP LoRA CIL development harness")
    parser.add_argument("--config",default="configs/cifar100.yaml")
    parser.add_argument("--method",choices=METHODS,default="drift")
    parser.add_argument("--output",required=True)
    parser.add_argument("--task-limit",type=int)
    parser.add_argument("--max-updates",type=int)
    parser.add_argument("--resume")
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--print-every",type=int,default=10)
    for name in ("micro-batch","effective-batch","seed","epochs","memory-size","fisher-samples"):
        parser.add_argument("--"+name,type=int)
    parser.add_argument("--lr",type=float)
    parser.add_argument("--delta",type=float)
    parser.add_argument("--backend",choices=("ns5","polar"))
    parser.add_argument("--device")
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for key in ("micro_batch","effective_batch","seed","epochs","memory_size","fisher_samples","lr","delta","backend","device"):
        value=getattr(args,key)
        if value is not None: cfg[key]=value
    if args.download: cfg["download"]=True
    try:
        run(cfg,args.output,args.method,args.task_limit,args.max_updates,args.resume,args.print_every)
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM: lower micro_batch and memory_micro_batch; keep effective_batch fixed. "
              "No task-boundary checkpoint is overwritten by this failure.",file=sys.stderr)
        raise


if __name__=="__main__": main()
