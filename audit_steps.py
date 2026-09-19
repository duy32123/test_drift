"""Audit an EXISTING steps.jsonl. Read-only: no run, no GPU, no checkpoint touched.

Answers one question about runs already on disk: of the updates a drift run logged
as accepted, how many moved the parameters at all?

    python audit_steps.py --run runs/pilot/drift_s1993 --output runs/audit/drift_s1993

Compare arms in one call:

    python audit_steps.py --run runs/pilot/drift_s1993 runs/pilot/scalar_s1993 \
        --output runs/audit/pilot

Logs written before `z_inf` was recorded are audited from the measured memory
loss and the allocator terms instead; `undecidable` counts records that carried
neither. Read `noop_fraction_of_gated` before reading any accuracy table: a run
whose gated updates were largely no-ops was not running the method under test.
"""
import argparse
import json
from pathlib import Path

from drift_cil.audit import audit_records, read_steps, tolerance_report


def audit_run(directory, zero_atol):
    directory = Path(directory)
    steps = directory / "steps.jsonl"
    if not steps.is_file():
        raise FileNotFoundError(f"No steps.jsonl in {directory}. Experiment outputs are "
                                "gitignored; copy the directory off the training machine.")
    manifest = {}
    if (directory / "manifest.json").is_file():
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    records = read_steps(steps)
    report = audit_records(records, zero_atol)
    report["run"] = str(directory)
    report["method"] = manifest.get("method")
    report["logged_z"] = any("z_inf" in r for r in records)
    if manifest.get("config"):
        config = dict(manifest["config"])
        # Runs made after the tolerance repair record what the solver actually used.
        if manifest.get("solver_tolerance") is not None:
            config["solver_tolerance"] = manifest["solver_tolerance"]
        report["tolerance"] = tolerance_report(config)
        report["delta"] = config.get("delta")
        report["z_lower"] = config.get("z_lower")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, nargs="+", help="Run directories to audit")
    parser.add_argument("--output", help="NEW directory for audit.json; omit to print only")
    parser.add_argument("--zero-atol", type=float, default=0.,
                        help="Measured memory-loss change at or below this counts as no "
                             "movement. Keep 0.0 unless the run's own repeatability check "
                             "shows nonzero jitter.")
    args = parser.parse_args()
    if args.zero_atol < 0:
        parser.error("--zero-atol must be nonnegative")
    output = Path(args.output).resolve() if args.output else None
    if output is not None and output.exists():
        parser.error("Audit output already exists. Choose a NEW --output; keep earlier audits.")

    reports = [audit_run(directory, args.zero_atol) for directory in args.run]
    for report in reports:
        gated, noop = report["gated_updates"], report["counts"].get("noop", 0)
        print(f"\n=== {report['run']}  (method={report['method']}, "
              f"z logged={report['logged_z']}) ===")
        print(f"  records {report['records']}   gated {gated}   "
              f"effective {report['effective_updates']}")
        print(f"  outcomes: {report['counts']}")
        if report.get("solver_status"):
            print(f"  solver:   {report['solver_status']}")
        if gated:
            print(f"  NO-OPS LOGGED AS ACCEPTED: {noop}  "
                  f"({100. * noop / gated:.1f}% of gated updates)")
        tolerance = report.get("tolerance")
        if tolerance and not tolerance["matched"]:
            print(f"  TOLERANCE MISMATCH: solver {tolerance['solver_tolerance']:g} vs "
                  f"gate {tolerance['gate_tolerance']:g}  ({tolerance['ratio']:g}x)")
        if report["counts"].get("undecidable"):
            print(f"  undecidable records: {report['counts']['undecidable']} "
                  "(no z_inf, no memory_before/after, no allocator terms)")
    if output is not None:
        output.mkdir(parents=True)
        (output / "audit.json").write_text(
            json.dumps(reports if len(reports) > 1 else reports[0], indent=2, allow_nan=False),
            encoding="utf-8")
        print(f"\nwrote {output / 'audit.json'}")


if __name__ == "__main__":
    main()
