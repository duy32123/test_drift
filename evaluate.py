"""Evaluate frozen task checkpoints AFTER configuration selection; no training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from drift_cil.runner import seed_all,make_model,evaluate
from drift_cil.model import restore_trainable
from drift_cil.data import DataBundle

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True,help="Directory containing task_XX.pt checkpoints")
    p.add_argument("--split",choices=("validation","test"),default="test")
    p.add_argument("--device",default="auto")
    p.add_argument("--batch-size",type=int,default=8)
    args=p.parse_args()
    folder=Path(args.run)
    files=sorted(folder.glob("task_[0-9][0-9].pt"))
    if not files:raise SystemExit("No completed-task checkpoints")
    first=torch.load(files[0],map_location="cpu",weights_only=False)
    cfg=first["config"];seed_all(cfg["seed"])
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device)
    data=DataBundle(cfg);model=make_model(cfg,data,device)
    rows=[]
    for filename in files:
        checkpoint=torch.load(filename,map_location="cpu",weights_only=False)
        restore_trainable(model,checkpoint["adapters"])
        result=evaluate(model,data,checkpoint["completed_tasks"],device,args.batch_size,args.split)
        rows.append(result)
        print(json.dumps({"checkpoint":filename.name,**result}),flush=True)
    result={"split":args.split,"synthetic":data.synthetic,"debug_subset":data.debug_subset,"stages":rows,
            "avg":float(np.mean([r["accuracy"] for r in rows])),"last":rows[-1]["accuracy"],
            "bwt":float(np.mean([rows[-1]["per_task"][i]-rows[i]["per_task"][i]
                                 for i in range(len(rows)-1)])) if len(rows)>1 else None}
    path=folder/f"{args.split}_evaluation.json"
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))

if __name__=="__main__":main()
