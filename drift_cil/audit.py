"""Recompute what each logged update actually DID, from steps.jsonl alone.

The runner writes `accept_status` from `accept_step`, which only ever asks
whether the MEASURED memory loss sits under the ceiling. A zero allocation
passes that test trivially: `Transaction.apply` restores the snapshot and adds
`-1.0 * 0`, so the measurement is bit-identical to the one taken before the
step and every ceiling the snapshot already met is met again. Such an update is
logged as `accept_status='accepted'`, `accepted_scale=1.0`, `backtracks=0`.

Four ways the shipped drift path reaches a zero allocation, all reproduced
against `drift_cil.allocator.solve_budget`:

  W1  rho < 0 with no first-order recovery direction. `solve_budget` returns
      `infeasible_candidate` and zeroes the candidate whenever `cost(minimum)>0`.
      If the overshoot is smaller than `gate_tolerance`, the gate still accepts.
  W2  every b_i <= 0 with `z_lower: 0.0`. The box optimum IS zero and the status
      is the reassuring `unconstrained`.
  W3  the solver's dual bisection targeted a STRICT `rho` while `accept_step`
      enforces `ceiling + gate_tolerance`, so the two ran on different budgets.
      Measured effect at realistic shapes (96 modes, 8 Fisher samples, a != 0):
      ~1.00x. It reaches 3-10x only when rho <= 1e-7 AND the step is nearly
      first-order neutral on the old task (1.a ~ 0). Worth repairing so a run
      can be reasoned about under one budget; NOT an explanation for any result.
  W4  `infeasible_candidate` does NOT imply zero: when `cost(minimum) <= 0` the
      nonzero least-harm step is kept. Read ||z||, never the status string.

This module never trusts `accept_status`. It recomputes from the raw fields and
reports the evidence behind each call, so a disagreement is inspectable rather
than a second opinion.

No torch. Runs on a laptop against logs copied off the training machine.
"""
from collections import Counter
import json
import math
from pathlib import Path

#: The gate ran and something moved.
MOVED = ("accepted", "recovery")
#: The gate ran and nothing moved.
STILL = ("noop", "rejected")
#: The gate did not run (no memory yet, or a method that does not gate).
UNGATED = ("no_gate",)
#: Raw fields were too sparse to decide.
UNKNOWN = ("undecidable",)
OUTCOMES = MOVED + STILL + UNGATED + UNKNOWN


def _get(record, key):
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def movement_evidence(record, zero_atol=0.0):
    """Collect every independent reason to believe the parameters did/did not move.

    Returns (moved, evidence) where `moved` is True, False or None (undecidable)
    and `evidence` is a list of (name, moved_bool) pairs, strongest first.

    `zero_atol` is compared against the MEASURED memory-loss change. It defaults
    to 0.0 because a genuinely zero step reproduces the measurement bit-exactly;
    raise it only if the run's own repeatability check shows nonzero jitter.
    """
    evidence = []
    scale = _get(record, "accepted_scale")
    if scale is not None and scale == 0.0:
        # accept_step exhausted the grid and called transaction.restore().
        return False, [("accepted_scale_zero", False)]

    z_inf = _get(record, "z_inf")
    if z_inf is not None:
        evidence.append(("z_inf", z_inf > 0.0))

    before, after = _get(record, "memory_before"), _get(record, "memory_after")
    if before is not None and after is not None:
        evidence.append(("memory_loss_change", abs(after - before) > zero_atol))

    objective, predicted = _get(record, "objective"), _get(record, "predicted_change")
    if objective is not None and predicted is not None:
        # b.z == 0 AND cost(z) == 0 together. With psd curvature and positive
        # damping, cost(z)==0 for z != 0 needs exact cancellation against a.z.
        evidence.append(("allocator_terms", not (objective == 0.0 and predicted == 0.0)))

    if not evidence:
        return None, []
    # A zero step makes EVERY indicator read zero. Any indicator that says
    # "moved" therefore outranks the rest; only unanimity implies a no-op.
    return any(flag for _, flag in evidence), evidence


def classify_record(record, zero_atol=0.0):
    """Return (outcome, detail) for one steps.jsonl record. Never reads `outcome`."""
    status = record.get("accept_status")
    detail = {"logged_accept_status": status,
              "solver_status": record.get("status"),
              "accepted_scale": _get(record, "accepted_scale")}
    if status in (None, "unconstrained"):
        # The runner sets this when the gate branch was skipped entirely.
        return "no_gate", detail
    moved, evidence = movement_evidence(record, zero_atol)
    detail["evidence"] = evidence
    if moved is None:
        detail["reason"] = ("no z_inf, no memory_before/after and no allocator terms; "
                            "this record cannot be audited")
        return "undecidable", detail
    detail["moved"] = moved
    if status == "rejected":
        if moved:
            detail["reason"] = "logged rejected but the raw fields show movement"
        return "rejected", detail
    if not moved:
        detail["reason"] = (f"logged {status!r} at scale "
                            f"{_get(record, 'accepted_scale')} with no movement")
        return "noop", detail
    return ("recovery" if status == "recovery" else "accepted"), detail


def audit_records(records, zero_atol=0.0):
    """Summarise a list of steps.jsonl records (already parsed)."""
    counts = Counter()
    per_task = {}
    mislabelled = []
    undecidable = []
    solver_status = Counter()
    modes = []
    negative_b = []
    for index, record in enumerate(records):
        outcome, detail = classify_record(record, zero_atol)
        counts[outcome] += 1
        task = record.get("task")
        bucket = per_task.setdefault(task, Counter())
        bucket[outcome] += 1
        if record.get("status"):
            solver_status[record["status"]] += 1
        if record.get("spectral_modes") is not None:
            modes.append(record["spectral_modes"])
        if record.get("negative_b_modes") is not None:
            negative_b.append(record["negative_b_modes"])
        if outcome == "noop":
            mislabelled.append({"line": index + 1, "update": record.get("update"),
                                "task": task, **detail})
        elif outcome == "undecidable":
            undecidable.append({"line": index + 1, "update": record.get("update")})
    gated = sum(counts[k] for k in MOVED + STILL)
    effective = sum(counts[k] for k in MOVED)
    summary = {
        "records": len(records),
        "zero_atol": zero_atol,
        "counts": dict(counts),
        "gated_updates": gated,
        "effective_updates": effective,
        "noop_fraction_of_gated": (counts["noop"] / gated) if gated else None,
        "silently_mislabelled": counts["noop"],
        "solver_status": dict(solver_status),
        "per_task": {str(k): dict(v) for k, v in sorted(
            per_task.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "undecidable_examples": undecidable[:20],
        "mislabelled_examples": mislabelled[:20],
    }
    if modes:
        summary["spectral_modes"] = {"min": min(modes), "max": max(modes),
                                     "mean": sum(modes) / len(modes)}
    if negative_b:
        summary["negative_b_modes"] = {"min": min(negative_b), "max": max(negative_b),
                                       "mean": sum(negative_b) / len(negative_b),
                                       "updates_with_any": sum(1 for v in negative_b if v > 0)}
    return summary


def read_steps(path):
    """Parse a steps.jsonl written by drift_cil.runner. Blank lines are skipped."""
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} is not valid JSON: {exc}") from exc
    return records


def tolerance_report(config):
    """Compare the gate's tolerance with the one solve_budget actually used."""
    from .allocator import solve_budget
    import inspect
    solver_default = inspect.signature(solve_budget).parameters["tolerance"].default
    gate = config.get("gate_tolerance", 1e-6)
    used = config.get("solver_tolerance", solver_default)
    return {"gate_tolerance": gate, "solver_tolerance": used,
            "solver_default": solver_default,
            "matched": bool(used == gate),
            "ratio": (gate / used) if used else None,
            "note": ("The solver confines the PREDICTED increase to rho + solver_tolerance "
                     "while the gate accepts a MEASURED increase up to ceiling + "
                     "gate_tolerance. When these differ the arm forfeits budget it is "
                     "being scored on; the effect is negligible for rho >> both and "
                     "dominant for rho at or below the larger one.")}
