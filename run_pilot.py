"""Sequential, restartable pilot queue. One GPU process at a time.

Running this command again skips completed runs and resumes interrupted runs
from their latest completed task. It does not select hyperparameters on test.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parent
METHODS=("adamw","muon","scalar","replay","drift","drift_scalar")


def write_json(path,value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2),encoding="utf-8")
    temp.replace(path)


def run_queue(config,output,methods,seeds,task_limit,method_lrs=None,dry_run=False):
    cfg=yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if not 0<task_limit<=cfg["tasks"]:
        raise ValueError("task_limit must be between 1 and configured tasks")
    root=Path(output).resolve()
    jobs=[]
    for seed in seeds:
        for method in methods:
            resolved=dict(cfg,seed=seed)
            if method in (method_lrs or {}):resolved["lr"]=method_lrs[method]
            folder=root/f"{method}_s{seed}"
            jobs.append({"method":method,"seed":seed,"folder":folder,"config":resolved})
    if dry_run:
        for job in jobs:
            print(json.dumps({"run":str(job["folder"]),"method":job["method"],
                              "seed":job["seed"],"lr":job["config"]["lr"],"tasks":task_limit}))
        return 0
    root.mkdir(parents=True,exist_ok=True)
    # Validate every existing run before starting any costly training.
    for job in jobs:
        requested=job["folder"]/"requested_config.json"
        if requested.exists() and json.loads(requested.read_text())!=job["config"]:
            raise ValueError(f"Config changed for {job['folder']}; use a different output root")
        if not requested.exists() and (job["folder"]/"manifest.json").exists():
            raise ValueError(f"{job['folder']} was not created by this queue; choose a new root")
    progress={"task_limit":task_limit,"runs":[]}
    progress_path=root/"progress.json"
    for job in jobs:
        folder=job["folder"];folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/"requested_config.json",job["config"])
        config_path=folder/"resolved.yaml"
        config_path.write_text(yaml.safe_dump(job["config"],sort_keys=True),encoding="utf-8")
        row={"method":job["method"],"seed":job["seed"],"run":str(folder),"status":"pending"}
        progress["runs"].append(row)
    write_json(progress_path,progress)
    for job,row in zip(jobs,progress["runs"]):
        folder=job["folder"]
        summary_path=folder/"summary.json"
        if summary_path.exists():
            summary=json.loads(summary_path.read_text())
            if summary["completed_tasks"]>=task_limit and summary["status"] in (
                    "complete","pilot_complete","debug_complete","debug_pilot_complete"):
                row["status"]="completed_previous";write_json(progress_path,progress)
                print(json.dumps(row),flush=True);continue
        command=[sys.executable,"-u",str(ROOT/"train.py"),"--config",str(folder/"resolved.yaml"),
                 "--method",job["method"],"--output",str(folder),"--task-limit",str(task_limit),
                 "--print-every","1"]
        if (folder/"last.pt").exists():command.extend(["--resume",str(folder/"last.pt")])
        elif (folder/"manifest.json").exists():
            raise RuntimeError(f"{folder}: manifest exists but checkpoint is missing; inspect this run")
        row["status"]="running";write_json(progress_path,progress)
        print(json.dumps({"event":"queue_start",**row,"command":command}),flush=True)
        started=time.perf_counter()
        with (folder/"console.log").open("a",encoding="utf-8") as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,encoding="utf-8",errors="replace")
            try:
                for line in process.stdout:
                    print(line,end="",flush=True);log.write(line);log.flush()
                code=process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                row.update(status="interrupted",elapsed_seconds=time.perf_counter()-started)
                write_json(progress_path,progress)
                print("Interrupted. Repeat the same command to resume from a task boundary.",flush=True)
                return 130
        row.update(status="completed" if code==0 else "failed",exit_code=code,
                   elapsed_seconds=time.perf_counter()-started)
        write_json(progress_path,progress)
        if code:
            print(f"Stopped after failure: {folder}/console.log",flush=True)
            return code
    command=[sys.executable,str(ROOT/"summarize.py"),*[str(j["folder"]) for j in jobs],
             "--output",str(root/"comparison.csv")]
    return subprocess.run(command).returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/cifar100.yaml")
    parser.add_argument("--output",default="runs/pilot")
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=["muon","scalar","replay","drift"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[1993])
    parser.add_argument("--task-limit",type=int,default=2)
    parser.add_argument("--method-lr",action="append",default=[],metavar="METHOD=LR")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    if len(set(args.methods))!=len(args.methods) or len(set(args.seeds))!=len(args.seeds):
        parser.error("methods and seeds must not contain duplicates")
    rates={}
    for item in args.method_lr:
        try:
            name,value=item.split("=",1);value=float(value)
            if name not in args.methods or not 0<value<float("inf"):raise ValueError()
            rates[name]=value
        except ValueError:parser.error("Use --method-lr METHOD=positive-number for a selected method")
    raise SystemExit(run_queue(args.config,args.output,args.methods,args.seeds,args.task_limit,rates,args.dry_run))


if __name__=="__main__":main()
