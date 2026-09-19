"""CPU-only logic tests, including parity with the installed v0.2 gate.

Run: python -m unittest test_diagnose_scalar.py
No PyTorch import is needed for these logic tests. They do NOT validate CLIP/CUDA.
"""
import ast
import math
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

from diagnose_scalar import traced_accept, smaller_trials, classify


def load_original_gate():
    path = Path(__file__).resolve().parent / "drift_cil" / "optim.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "accept_step")
    namespace = {"np": SimpleNamespace(isfinite=math.isfinite)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["accept_step"]


class Transaction:
    def __init__(self, value=0., direction=1., rounding=None):
        self.before = value
        self.value = value
        self.direction = direction
        self.rounding = rounding

    def apply(self, fraction=1.):
        self.value = self.before - self.direction * fraction
        if self.rounding is not None:
            self.value = round(self.value, self.rounding)

    def restore(self):
        self.value = self.before


def movement(tx):
    return abs(tx.value - tx.before)


class DiagnosticTests(unittest.TestCase):
    def test_original_decisions_preserved_including_recovery_and_rejection(self):
        original = load_original_gate()
        rng = random.Random(410)
        statuses = set()
        for _ in range(1000):
            value, direction = rng.uniform(-2, 2), rng.uniform(-1, 1)
            linear, quadratic = rng.uniform(-1, 1), rng.uniform(0, 2)
            ceiling = 1. + rng.uniform(-.1, .1)
            t1, t2 = Transaction(value, direction), Transaction(value, direction)
            def loss(tx):
                d = tx.value - value
                return 1. + linear * d + quadratic * d * d
            expected = original(t1, lambda: loss(t1), 1., ceiling)
            actual, trace = traced_accept(original, t2, lambda: loss(t2), 1., ceiling)
            self.assertEqual(actual, expected)
            self.assertEqual(t1.value, t2.value)
            self.assertEqual(len(trace), actual["backtracks"] + (actual["accept_status"] != "rejected"))
            self.assertEqual(trace[0]["scale"], 1.)
            statuses.add(actual["accept_status"])
        self.assertEqual(statuses, {"accepted", "recovery", "rejected"})

    def test_extended_grid_can_find_feasible_nonzero_step_and_rolls_back(self):
        tx = Transaction()
        rows = smaller_trials(tx, lambda: -tx.value, 0., .001, 9, 8, movement, atol=0.)
        self.assertEqual(classify(rows), "smaller_nonzero_step_is_feasible")
        self.assertEqual(rows[-1]["scale"], 1 / 1024)
        self.assertEqual(tx.value, tx.before)

    def test_rounding_to_zero_is_not_reported_as_progress(self):
        tx = Transaction(rounding=0)
        rows = smaller_trials(tx, lambda: -tx.value, 0., 0., 9, 8, movement)
        self.assertEqual(classify(rows), "reached_parameter_rounding_limit")
        self.assertEqual(tx.value, tx.before)

    def test_impossible_positive_grid_has_no_universal_impossibility_claim(self):
        tx = Transaction()
        rows = smaller_trials(tx, lambda: -tx.value, 0., 0., 9, 8, movement, atol=0.)
        self.assertEqual(classify(rows), "no_acceptable_step_in_additional_grid")
        self.assertEqual(len(rows), 8)
        self.assertEqual(tx.value, tx.before)

    def test_probe_rolls_back_on_interrupt(self):
        tx = Transaction()
        def interrupted():
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            smaller_trials(tx, interrupted, 0., 0., 9, 8, movement)
        self.assertEqual(tx.value, tx.before)

    def test_original_gate_observer_rolls_back_on_error(self):
        tx = Transaction()
        original_apply = tx.apply
        def broken():
            raise RuntimeError("simulated forward error")
        with self.assertRaises(RuntimeError):
            traced_accept(load_original_gate(), tx, broken, 0., 0.)
        self.assertEqual(tx.value, tx.before)
        self.assertEqual(tx.apply, original_apply)

    def test_nonfinite_loss_is_not_accepted_by_probe(self):
        tx = Transaction()
        rows = smaller_trials(tx, lambda: float("nan"), 0., .1, 9, 2, movement)
        self.assertTrue(all(not r["finite"] for r in rows))
        self.assertEqual(classify(rows), "no_acceptable_step_in_additional_grid")
        self.assertEqual(tx.value, tx.before)


if __name__ == "__main__":
    unittest.main()
