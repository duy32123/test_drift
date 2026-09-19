"""Two real CIFAR tasks through the SAME trainer, with a labelled tiny subset.

This checks the downloaded pretrained model, image transforms, backward,
memory/Fisher/allocator/acceptance and checkpointing. It is NOT an accuracy run.
"""
import argparse
import json
from pathlib import Path
import torch
import yaml
from drift_cil.runner import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/cifar100.yaml")
    parser.add_argument("--output",default="runs/cifar_preflight")
    parser.add_argument("--data-root")
    parser.add_argument("--model-path",help="Optional downloaded local CLIP model directory")
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--offline",action="store_true",help="Use existing CIFAR and Hugging Face caches only")
    parser.add_argument("--device",default="auto")
    parser.add_argument("--require-cuda",action="store_true")
    args=parser.parse_args()
    if args.download and args.offline:parser.error("--download and --offline are mutually exclusive")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: install a compatible PyTorch wheel first")
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("synthetic",False):
        raise SystemExit("preflight_cifar.py requires a real CIFAR config, not the synthetic fixture")
    classes_per_task=100//cfg["tasks"]
    cfg.update(debug_train_per_class=1,debug_validation_per_class=1,epochs=1,
               warmup_steps=0,micro_batch=1,effective_batch=classes_per_task,
               eval_batch=1,memory_micro_batch=1,memory_size=2*classes_per_task,
               fisher_samples=2,device=args.device,download=args.download)
    cfg["model_local_files_only"]=args.offline
    if args.data_root:cfg["data_root"]=args.data_root
    if args.model_path:cfg["model_id"]=str(Path(args.model_path).resolve())
    print(json.dumps({"event":"real_data_integration_only","warning":"Not benchmark results",
                      "tasks":2,"training_images_per_class":1,"fisher_samples":2}),flush=True)
    result=run(cfg,args.output,"drift",task_limit=2,print_every=1)
    records=[json.loads(line) for line in (Path(args.output)/"steps.jsonl").read_text().splitlines()]
    guarded=[r for r in records if r["task"]>0]
    if result["completed_tasks"]!=2 or not guarded or not all("memory_after" in r for r in guarded):
        raise SystemExit("Preflight did not complete the old-memory acceptance path")
    print(json.dumps({"event":"preflight_passed","real_data":True,"gpu_tested":
        args.device!="cpu" and torch.cuda.is_available(),"benchmark_result":False}),flush=True)


if __name__=="__main__":main()
