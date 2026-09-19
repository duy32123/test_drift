"""Train task A once, then fork drift and drift_scalar into fresh directories.

Each invocation requires a new output root. Checkpoints are trusted local files.
Artifacts include source snapshots, step logs, audits and checkpoint SHA-256s.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from audit_steps import audit_run
from drift_cil.runner import code_hash, file_hash, run

ROOT = Path(__file__).resolve().parent


def run_pair(config, output):
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if cfg["tasks"] < 2:
        raise ValueError("Paired run requires at least two configured tasks")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source"
    snapshot.mkdir()
    source_files = sorted((ROOT / "drift_cil").glob("*.py")) + [
        ROOT / name for name in ("run_paired.py", "audit_steps.py", "summarize.py")]
    source_hashes = {}
    for source in source_files:
        relative = source.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hashes[relative.as_posix()] = file_hash(source)
    (output / "resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    try:
        commit = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    provenance = {"git_head": commit, "code_sha256_lf": code_hash(),
                  "source_files_sha256": source_hashes,
                  "note": "Source snapshot is authoritative; git_head alone does not prove a clean tree."}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    def verify_source():
        if any(file_hash(ROOT / name) != digest for name, digest in source_hashes.items()):
            raise RuntimeError("Source changed during paired run; results must not be certified")

    common = output / "task_a"
    run(cfg, common, "drift", task_limit=1)
    checkpoint = common / "task_01.pt"
    parent_hash = file_hash(checkpoint)
    reports = []
    arms = []
    for method in ("drift", "drift_scalar"):
        verify_source()
        if file_hash(checkpoint) != parent_hash:
            raise RuntimeError("Shared checkpoint changed")
        arm = output / method
        run(cfg, arm, method, task_limit=2, fork_from=checkpoint)
        reports.append(audit_run(arm, 0.0))
        arms.append(arm)
    verify_source()
    if file_hash(checkpoint) != parent_hash:
        raise RuntimeError("Shared checkpoint changed")
    (output / "audit.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "summarize.py"), *map(str, arms),
                    "--output", str(output / "comparison.csv")], check=True)
    artifacts = {p.relative_to(output).as_posix(): file_hash(p)
                 for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "checksums.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cifar100.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_pair(args.config, args.output)
