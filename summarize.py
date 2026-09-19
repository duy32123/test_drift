"""Summarize actually measured step costs; never extrapolate a GPU benchmark from CPU."""
import argparse
import csv
import json
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("runs",nargs="+")
    p.add_argument("--output",default="comparison.csv")
    args=p.parse_args()
    rows=[]
    for directory in args.runs:
        root=Path(directory)
        manifest=json.loads((root/"manifest.json").read_text())
        summary=json.loads((root/"summary.json").read_text())
        steps=[json.loads(s) for s in (root/"steps.jsonl").read_text().splitlines()]
        new_n=sum(s["new_samples"] for s in steps)
        old={k:sum(s["old_reads"][k] for s in steps) for k in ("gradient","fisher","gate","replay")}
        previous=[s for s in steps if s["task"]>0]
        median=statistics.median(s["seconds"] for s in previous) if previous else None
        rows.append({"run":str(root),"method":manifest["method"],"seed":manifest["config"]["seed"],
            "synthetic":summary["synthetic"],"status":summary["status"],"completed_tasks":summary["completed_tasks"],
            "debug_subset":summary.get("debug_subset",False),
            "val_avg":summary["validation_avg"],"val_last":summary["validation_last"],"val_bwt":summary["validation_bwt"],
            "median_old_task_step_s":median,"total_new_images":new_n,
            **{f"old_{k}_images":v for k,v in old.items()},
            "peak_allocated_gib":summary["peak_allocated_gib"],
            "rejected_steps":sum(s.get("accept_status")=="rejected" for s in steps)})
    with open(args.output,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2))

if __name__=="__main__":main()
