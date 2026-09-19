"""Diagnostic-only replay of scalar v0.2 from its completed-task checkpoint.

Copy beside train.py. Original runs, budgets and training source stay unchanged.
Stops at the first rejected update (or --max-updates additional updates).
Smaller trial steps are measured and ALWAYS rolled back. This is not a benchmark.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


class DiagnosticStop(Exception):
    pass


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def traced_accept(original, transaction, loss_fn, before_loss, ceiling,
                  max_backtracks=8, atol=1e-6):
    """Observe the original gate, without changing its decisions or trial grid."""
    rows = []
    original_apply = transaction.apply
    active_fraction = None

    def apply(fraction=1.):
        nonlocal active_fraction
        active_fraction = fraction
        return original_apply(fraction)

    def measured_loss():
        started = time.perf_counter()
        value = float(loss_fn())
        rows.append({"scale": active_fraction, "memory_loss": value,
                     "loss_change": value - before_loss,
                     "excess_over_ceiling": value - ceiling,
                     "finite": math.isfinite(value),
                     "seconds": time.perf_counter() - started})
        return value

    transaction.apply = apply
    try:
        outcome = original(transaction, measured_loss, before_loss, ceiling,
                           max_backtracks=max_backtracks, atol=atol)
    except BaseException:
        transaction.restore()
        raise
    finally:
        transaction.apply = original_apply
    return outcome, rows


def smaller_trials(transaction, loss_fn, before_loss, ceiling, start_exponent,
                   count, movement, atol=1e-6):
    """Probe below the original grid. Restore even on exception/interruption."""
    rows = []
    try:
        for exponent in range(start_exponent, start_exponent + count):
            fraction = 2. ** (-exponent)
            transaction.apply(fraction)
            norm = movement(transaction)
            started = time.perf_counter()
            after = float(loss_fn())
            finite = math.isfinite(after)
            feasible = finite and after <= ceiling + atol
            recovery = finite and before_loss > ceiling + atol and after < before_loss - atol
            rows.append({"scale": fraction, "memory_loss": after,
                         "loss_change": after - before_loss,
                         "excess_over_ceiling": after - ceiling,
                         "finite": finite, "within_budget": feasible,
                         "recovery": recovery, "parameter_step_l2": norm,
                         "nonzero_step": norm > 0.,
                         "seconds": time.perf_counter() - started})
            if (feasible or recovery) and norm > 0.:
                break
            if norm == 0.:
                break
    finally:
        transaction.restore()
    return rows


def classify(rows):
    if any(r["nonzero_step"] and r["within_budget"] for r in rows):
        return "smaller_nonzero_step_is_feasible"
    if any(r["nonzero_step"] and r["recovery"] for r in rows):
        return "smaller_step_only_recovers_toward_budget"
    if any(not r["nonzero_step"] for r in rows):
        return "reached_parameter_rounding_limit"
    return "no_acceptable_step_in_additional_grid"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Existing scalar_s1993 directory")
    parser.add_argument("--output", default="runs/diagnose_scalar", help="NEW diagnostic directory")
    parser.add_argument("--max-updates", type=int, default=60,
                        help="Maximum ADDITIONAL updates from checkpoint")
    parser.add_argument("--extra-backtracks", type=int, default=8)
    args = parser.parse_args()
    if args.max_updates < 1 or not 1 <= args.extra_backtracks <= 24:
        parser.error("Require max-updates >= 1 and extra-backtracks between 1 and 24")
    source = Path(args.run).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Diagnostic output already exists. Choose a NEW --output; do not delete the original run.")
    if output == source or source in output.parents:
        parser.error("Put diagnostic output outside the original run directory")
    checkpoint_path = source / "last.pt"
    if not checkpoint_path.is_file():
        parser.error("No last.pt in the scalar run. Keep existing files and send the directory listing.")

    # Heavy dependencies are imported only on an actual local training machine.
    import torch
    from drift_cil import runner

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "scalar":
        parser.error("Use a SCALAR checkpoint, not a Muon checkpoint")
    completed = int(checkpoint["completed_tasks"])
    cfg = dict(checkpoint["config"])
    if completed < 1 or completed >= cfg["tasks"]:
        parser.error("Need a scalar checkpoint with at least one completed task and another task remaining")
    if not checkpoint["memory"]["entries"]:
        parser.error("The checkpoint has no old-task memory")
    initial_update = int(checkpoint["update"])
    package = Path(runner.__file__).parent
    report = {
        "diagnostic_only": True, "status": "running", "source_run": str(source),
        "source_checkpoint": str(checkpoint_path), "completed_tasks_at_start": completed,
        "initial_update": initial_update, "max_additional_updates": args.max_updates,
        "delta": cfg["delta"], "gate_tolerance": cfg.get("gate_tolerance", 1e-6),
        "original_max_backtracks": cfg.get("max_backtracks", 8),
        "source_code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(package.glob("*.py"))},
        "note": "Replayed from a task boundary. Lost in-task weights cannot be recovered from text logs.",
    }
    output.mkdir(parents=True)
    write_json(output / "diagnostic.json", report)
    original_accept = runner.accept_step
    calls = 0
    started = time.perf_counter()

    def movement(transaction):
        with torch.no_grad():
            terms = [(p.detach() - transaction.before[n]).double().square().sum()
                     for n, p in transaction.params]
            return float(torch.stack(terms).sum().sqrt()) if terms else 0.

    def gate(transaction, loss_fn, before_loss, ceiling, max_backtracks=8, atol=1e-6):
        nonlocal calls
        calls += 1
        outcome, trials = traced_accept(original_accept, transaction, loss_fn,
                                       before_loss, ceiling, max_backtracks, atol)
        entry = {"update": initial_update + calls - 1, "before_loss": before_loss,
                 "ceiling": ceiling, "headroom": ceiling - before_loss,
                 "outcome": outcome, "trials": trials}
        with (output / "gate_trials.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean(entry), allow_nan=False) + "\n")
        if outcome["accept_status"] != "rejected":
            return outcome

        report.update(status="first_rejection_captured", first_rejection=entry)
        write_json(output / "diagnostic.json", report)
        # This is a RESEARCH SNAPSHOT, deliberately not named last.pt and not
        # compatible with the task-boundary resume interface.
        runner.atomic_save({
            "diagnostic_only": True, "not_resumable": True,
            "config": cfg, "source_checkpoint": str(checkpoint_path),
            "attempted_update": entry["update"],
            "adapters": {n: v.detach().cpu().clone() for n, v in transaction.before.items()},
            "directions": {n: v.detach().cpu().clone() for n, v in transaction.directions.items()},
            "memory": checkpoint["memory"], "before_loss": before_loss, "ceiling": ceiling,
        }, output / "diagnostic_state.pt")
        print(json.dumps({"event": "diagnostic_first_rejection", "update": entry["update"],
                          "headroom": entry["headroom"], "extra_trials_max": args.extra_backtracks}), flush=True)
        # Check repeatability before interpreting differences close to tolerance.
        repeated = float(loss_fn())
        report.update(repeated_baseline_loss=repeated,
                      repeated_baseline_abs_difference=abs(repeated - before_loss))
        extras = smaller_trials(transaction, loss_fn, before_loss, ceiling,
                                max_backtracks + 1, args.extra_backtracks, movement, atol)
        report.update(additional_trials=extras, finding=classify(extras))
        report["baseline_repeatability_warning"] = (
            not math.isfinite(repeated) or abs(repeated - before_loss) > atol)
        report["status"] = "diagnostic_complete"
        raise DiagnosticStop()

    runner.accept_step = gate
    try:
        runner.run(cfg, output, "scalar", task_limit=completed + 1,
                   max_updates=initial_update + args.max_updates,
                   resume=checkpoint_path, print_every=1)
        report.update(status="no_rejection_observed_within_limit")
    except DiagnosticStop:
        pass
    except KeyboardInterrupt:
        report.update(status="diagnostic_interrupted")
        print("Diagnostic interrupted; original run and its checkpoint are unchanged.", flush=True)
    except Exception as exc:
        report.update(status="diagnostic_failed", error=str(exc))
        raise
    finally:
        runner.accept_step = original_accept
        report.update(gated_updates_observed=calls, elapsed_seconds=time.perf_counter() - started)
        write_json(output / "diagnostic.json", report)
    print(json.dumps({"event": "diagnostic_summary", "status": report["status"],
                      "finding": report.get("finding"), "report": str(output / "diagnostic.json")}), flush=True)


if __name__ == "__main__":
    main()
