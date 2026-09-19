"""Launch bounded REAL CIFAR/CLIP profiles, one fresh subprocess per micro-batch.

Only task-0/base costs are measured here. Run two-task pilots to measure the
memory/Fisher/solver overhead. This script downloads only with --download.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/cifar100.yaml")
    p.add_argument("--batches",nargs="+",type=int,default=[4,8,16])
    p.add_argument("--updates",type=int,default=10)
    p.add_argument("--output",default="runs/profiles")
    p.add_argument("--download",action="store_true")
    args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    results=[]
    for batch in args.batches:
        folder=root/f"micro_{batch}"
        command=[sys.executable,"train.py","--config",args.config,"--method","muon",
                 "--micro-batch",str(batch),"--task-limit","1","--max-updates",str(args.updates),
                 "--output",str(folder),"--print-every","1"]
        if args.download:command.append("--download")
        with (root/f"micro_{batch}.log").open("w",encoding="utf-8") as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        row={"micro_batch":batch,"exit_code":code}
        path=folder/"summary.json"
        if code==0 and path.exists():row.update(json.loads(path.read_text()))
        results.append(row);print(json.dumps(row),flush=True)
    (root/"profiles.json").write_text(json.dumps(results,indent=2))

if __name__=="__main__":main()
