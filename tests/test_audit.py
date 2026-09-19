"""The audit must reproduce the defects it claims, against the shipped code."""
import inspect
import json

import numpy as np
import pytest
import torch

from drift_cil.allocator import Curvature, solve_budget
from drift_cil.audit import (audit_records, classify_record, movement_evidence,
                             read_steps, tolerance_report)
from drift_cil.model import TinyClassifier
from drift_cil.optim import SpectralProposal, Transaction, accept_step

GATE = 1e-6
SOLVER_DEFAULT = inspect.signature(solve_budget).parameters["tolerance"].default


def tiny_proposal(seed=0, modes_scale=1e-4):
    torch.manual_seed(seed)
    model = TinyClassifier(10, rank=4)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    x, y = torch.randn(16, 3, 8, 8), torch.randint(0, 10, (16,))
    torch.nn.functional.cross_entropy(model(x), y).backward()
    grads = {n: p.grad.detach().clone() for n, p in params}
    directions = {n: g / g.norm().clamp_min(1e-9) * modes_scale for n, g in grads.items()}
    return model, params, grads, SpectralProposal(params, directions)


# ----------------------------------------------------------------- the defect

def test_zero_allocation_is_accepted_by_the_gate_and_caught_by_the_audit():
    """W1: rho below zero with no recovery direction, at the SHIPPED solver tolerance.

    The status string is not the signal. At tolerance 1e-8 this returns
    `infeasible_candidate`; at 1e-6 the same zero step is returned as `recovery`.
    Both are no-ops and both are logged as accepted. The audit reads ||z||.
    """
    model, params, grads, proposal = tiny_proposal()
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(8, proposal.size)) * 1e-3
    b = proposal.coordinates(grads)
    a = np.zeros(proposal.size)                       # no mode helps the old task
    rho = -5e-7                                       # over the ceiling, but under atol
    z, diagnostics = proposal.allocate(b, a, scores, rho, .03, 0., 500, SOLVER_DEFAULT)
    assert diagnostics["status"] == "infeasible_candidate"
    assert np.max(np.abs(z)) == 0.

    # With the tolerance matched to the gate's, this window closes: the solver
    # works to the same ceiling the gate enforces and finds a real step.
    repaired, info = proposal.allocate(b, a, scores, rho, .03, 0., 500, GATE)
    assert np.max(np.abs(repaired)) > 0.
    assert info["predicted_change"] <= rho + GATE + 1e-15

    before, ceiling = 0.2931924713484477, 0.2931924713484477 - rho
    snapshot = {n: p.detach().clone() for n, p in params}
    transaction = Transaction(params, proposal.directions(z))
    accepted = accept_step(transaction, lambda: before, before, ceiling, atol=GATE)

    # The shipped gate calls a zero step a full-scale accepted update.
    assert accepted["accept_status"] == "accepted"
    assert accepted["accepted_scale"] == 1.
    assert accepted["backtracks"] == 0
    assert all(torch.equal(p.detach(), snapshot[n]) for n, p in params)

    record = {"task": 1, "update": 369, **diagnostics, **accepted,
              "memory_before": before, "budget_ceiling": ceiling}
    assert classify_record(record)[0] == "noop"


def test_all_negative_b_makes_the_box_optimum_zero_and_status_reassuring():
    """W2: with z_lower = 0 the optimum IS zero, and the status reads 'unconstrained'."""
    curvature = Curvature(np.eye(4), np.full(4, .03))
    z, info = solve_budget(-np.ones(4), np.zeros(4), curvature, 1., tolerance=GATE)
    assert info["status"] == "unconstrained"
    assert np.max(np.abs(z)) == 0.
    record = {"accept_status": "accepted", "accepted_scale": 1., "objective": 0.,
              "predicted_change": 0., "memory_before": .3, "memory_after": .3}
    assert classify_record(record)[0] == "noop"


def test_infeasible_candidate_does_not_imply_a_zero_step():
    """W4: the least-harm step is kept when its cost is negative. Read ||z||, not status."""
    curvature = Curvature(np.eye(4), np.full(4, .03))
    z, info = solve_budget(np.ones(4), np.ones(4), curvature, -50., tolerance=GATE)
    assert info["status"] == "infeasible_candidate"
    assert np.max(np.abs(z)) > 0.
    assert info["predicted_change"] < 0.


@pytest.mark.parametrize("rho", [1e-5, 1e-7, 1e-8, 1e-9, 0.])
def test_the_solver_now_spends_the_budget_the_gate_actually_enforces(rho):
    """W3. The bisection used to target a strict rho while accept_step enforced
    ceiling + atol, so the step was held to a budget nobody was checking. It must
    now reach the optimum over the SAME set the gate accepts."""
    curvature = Curvature(np.array([[np.sqrt(2.)]]), np.zeros(1))   # cost(z) = z^2
    z, info = solve_budget(np.array([1.]), np.array([0.]), curvature, rho, tolerance=GATE)
    assert info["predicted_change"] <= rho + GATE + 1e-15
    assert z[0] == pytest.approx(np.sqrt(rho + GATE), rel=2e-2)


def test_matching_the_tolerance_changes_nothing_when_the_budget_is_large():
    """The repair must not quietly loosen a run that was never near the ceiling."""
    curvature = Curvature(np.array([[np.sqrt(2.)]]), np.zeros(1))
    for rho in (1e-3, 1e-2, 1e-1):
        _, tight = solve_budget(np.array([1.]), np.array([0.]), curvature, rho,
                                tolerance=SOLVER_DEFAULT)
        _, matched = solve_budget(np.array([1.]), np.array([0.]), curvature, rho,
                                  tolerance=GATE)
        assert matched["objective"] == pytest.approx(tight["objective"], rel=1e-3)


def test_allocate_now_passes_the_tolerance_through():
    _, _, grads, proposal = tiny_proposal()
    rng = np.random.default_rng(5)
    scores = rng.normal(size=(8, proposal.size)) * 1e-2
    b, a = proposal.coordinates(grads), np.zeros(proposal.size)
    _, tight = proposal.allocate(b, a, scores, 0., .03, 0., 500, 1e-12)
    _, loose = proposal.allocate(b, a, scores, 0., .03, 0., 500, 1e-3)
    assert loose["objective"] > tight["objective"]
    for info in (tight, loose):
        assert {"z_inf", "z_l1", "z_nonzero"} <= set(info)


# ----------------------------------------------------- the scalar restriction

def test_scalar_collapse_keeps_every_cross_mode_term():
    _, _, grads, proposal = tiny_proposal()
    rng = np.random.default_rng(11)
    scores = rng.normal(size=(8, proposal.size)) * 1e-2
    b, a = proposal.coordinates(grads), rng.normal(size=proposal.size) * 1e-3
    rho = 1e-3
    z, info = proposal.allocate_scalar(b, a, scores, rho, .03, 0., 500, GATE)

    # z really is one global amplitude
    assert np.allclose(z, z[0])
    # the 1-D cost equals the full vector cost at that z, cross terms included
    dense = proposal._curvature(scores, .03)
    full = -float(a @ z) + .5 * dense.quad(z)
    assert info["predicted_change"] == pytest.approx(full, rel=1e-9, abs=1e-14)
    assert float(b @ z) == pytest.approx(info["objective"], rel=1e-9, abs=1e-14)
    # 1.C.1 is NOT the diagonal sum: dropping cross terms would change it
    ones = np.ones(proposal.size)
    assert info["scalar_curvature"] == pytest.approx(float(ones @ dense.mv(ones)), rel=1e-12)


def test_scalar_is_nested_inside_the_vector_allocation():
    """The per-mode arm can never do worse at the same budget; that is the comparison."""
    _, _, grads, proposal = tiny_proposal()
    rng = np.random.default_rng(13)
    scores = rng.normal(size=(8, proposal.size)) * 1e-2
    b, a = np.abs(proposal.coordinates(grads)), np.zeros(proposal.size)
    for rho in (1e-6, 1e-4, 1e-2):
        _, scalar = proposal.allocate_scalar(b, a, scores, rho, .03, 0., 500, GATE)
        _, vector = proposal.allocate(b, a, scores, rho, .03, 0., 500, GATE)
        assert vector["objective"] >= scalar["objective"] - 1e-9
        assert scalar["predicted_change"] <= rho + GATE


def test_scalar_matches_a_brute_force_grid_over_the_same_budget():
    _, _, grads, proposal = tiny_proposal()
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(8, proposal.size)) * 1e-2
    b, a = proposal.coordinates(grads), rng.normal(size=proposal.size) * 1e-3
    dense = proposal._curvature(scores, .03)
    b_s, a_s = float(b.sum()), float(a.sum())
    c_s = float(np.ones(proposal.size) @ dense.mv(np.ones(proposal.size)))
    for rho in (1e-7, 1e-5, 1e-3, 1e-1):
        _, info = proposal.allocate_scalar(b, a, scores, rho, .03, 0., 500, GATE)
        grid = np.linspace(0., 1., 200_001)
        feasible = grid[-a_s * grid + .5 * c_s * grid ** 2 <= rho + GATE]
        best = float((b_s * feasible).max()) if feasible.size else -np.inf
        assert info["objective"] >= best - 1e-9 * max(1., abs(best))


# --------------------------------------------------------------- the auditor

def test_rejected_and_ungated_records_are_not_confused_with_no_ops():
    rejected = {"accept_status": "rejected", "accepted_scale": 0., "z_inf": .5,
                "memory_before": .3, "memory_after": .3}
    assert classify_record(rejected)[0] == "rejected"
    assert classify_record({"accepted_scale": 1., "accept_status": "unconstrained"})[0] == "no_gate"
    assert classify_record({})[0] == "no_gate"


def test_a_record_with_no_movement_field_is_undecidable_not_a_no_op():
    """An ABSENT z must never be read as a zero z."""
    outcome, detail = classify_record({"accept_status": "accepted", "accepted_scale": 1.})
    assert outcome == "undecidable"
    assert "cannot be audited" in detail["reason"]


def test_any_indicator_of_movement_outranks_the_others():
    # allocator terms say zero, but the measured loss moved: not a no-op
    moved, evidence = movement_evidence({"accepted_scale": 1., "objective": 0.,
                                         "predicted_change": 0., "memory_before": .3,
                                         "memory_after": .31})
    assert moved is True
    assert dict(evidence)["allocator_terms"] is False


def test_recovery_is_kept_distinct_from_acceptance():
    record = {"accept_status": "recovery", "accepted_scale": .25, "z_inf": .8,
              "memory_before": .40, "memory_after": .35}
    assert classify_record(record)[0] == "recovery"


def test_non_finite_fields_do_not_silently_become_zero():
    record = {"accept_status": "accepted", "accepted_scale": 1., "z_inf": float("nan")}
    assert classify_record(record)[0] == "undecidable"


def test_audit_recovers_the_no_op_rate_from_a_log_that_never_recorded_z():
    records = []
    for i in range(100):
        noop = i % 4 == 0
        records.append({"task": i // 50, "update": i, "method": "drift",
                        "status": "infeasible_candidate" if noop else "active",
                        "objective": 0. if noop else 1e-4,
                        "predicted_change": 0. if noop else 1e-6,
                        "accept_status": "accepted", "accepted_scale": 1.,
                        "memory_before": .3, "memory_after": .3 if noop else .3001,
                        "spectral_modes": 96, "negative_b_modes": 3})
    report = audit_records(records)
    assert report["counts"]["noop"] == 25
    assert report["counts"]["accepted"] == 75
    assert report["effective_updates"] == 75
    assert report["noop_fraction_of_gated"] == pytest.approx(.25)
    assert report["per_task"]["0"]["noop"] == 13
    assert report["negative_b_modes"]["updates_with_any"] == 100


def test_tolerance_report_flags_the_shipped_default():
    assert tolerance_report({"gate_tolerance": 1e-6})["matched"] is False
    assert tolerance_report({"gate_tolerance": 1e-6})["ratio"] == pytest.approx(100.)
    assert tolerance_report({"gate_tolerance": 1e-6, "solver_tolerance": 1e-6})["matched"]


def test_read_steps_round_trips_and_rejects_corruption(tmp_path):
    path = tmp_path / "steps.jsonl"
    path.write_text('{"a":1}\n\n{"a":2}\n', encoding="utf-8")
    assert read_steps(path) == [{"a": 1}, {"a": 2}]
    (tmp_path / "bad.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_steps(tmp_path / "bad.jsonl")


def test_end_to_end_synthetic_run_logs_z_and_an_outcome(tmp_path):
    from drift_cil import runner
    config = json.loads((__import__("pathlib").Path(__file__).parent / "smoke.json").read_text()) \
        if (__import__("pathlib").Path(__file__).parent / "smoke.json").exists() else None
    if config is None:
        import yaml
        from pathlib import Path
        config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml").read_text())
    config.update(tasks=2, epochs=1, memory_size=max(config["memory_size"], 20))
    for method in ("drift", "drift_scalar"):
        out = tmp_path / method
        runner.run(config, out, method, task_limit=2, print_every=1000)
        records = read_steps(out / "steps.jsonl")
        gated = [r for r in records if r.get("accept_status") not in (None, "unconstrained")]
        assert gated, f"{method}: no gated update was reached"
        assert all("z_inf" in r and "outcome" in r for r in gated)
        for record in gated:
            assert record["outcome"] == classify_record(record)[0]
        assert audit_records(records)["counts"].get("undecidable", 0) == 0
