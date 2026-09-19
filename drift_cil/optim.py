"""Muon base updates and their spectral allocation, in parameter coordinates."""
import math
import numpy as np
import torch
from .allocator import Curvature, solve_budget


@torch.no_grad()
def ns5(matrix, steps=5):
    """Fixed quintic (3.4445,-4.775,2.0315), FP32, Frobenius input normalization.

    FP32 is deliberate and recorded; it is not claimed bit-identical to a
    BF16 reference implementation. Every NS arm in this harness shares it.
    """
    x = matrix.float()
    if not bool(torch.count_nonzero(x)):
        return torch.zeros_like(x)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / x.norm().clamp_min(1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445*x + (-4.775*gram + 2.0315*(gram @ gram)) @ x
    return x.T if transpose else x


class MatrixSteps:
    def __init__(self, named_parameters, beta=.95, backend="ns5", ns_steps=5,
                 shape_scale=True):
        if backend not in ("ns5", "polar"):
            raise ValueError("backend must be ns5 or polar")
        if not 0 <= beta < 1 or ns_steps < 1:
            raise ValueError("Require 0 <= beta < 1 and ns_steps >= 1")
        self.params = [(n, p) for n, p in named_parameters if p.requires_grad]
        if any(p.ndim != 2 for _, p in self.params):
            raise ValueError("v0 trains only LoRA matrix factors; freeze all other parameters")
        self.beta, self.backend, self.ns_steps = beta, backend, ns_steps
        self.shape_scale = shape_scale
        self.momentum = {n: torch.zeros_like(p, dtype=torch.float32) for n,p in self.params}

    @torch.no_grad()
    def directions(self, lr):
        out = {}
        for name, p in self.params:
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            m = self.momentum[name]
            m.lerp_(g, 1. - self.beta)
            source = g.lerp(m, self.beta)  # normalized EMA + Nesterov, fixed across arms
            if not bool(torch.count_nonzero(source)):
                out[name] = torch.zeros_like(source)
                continue
            if self.backend == "ns5":
                direction = ns5(source, self.ns_steps)
            else:
                u, s, vh = torch.linalg.svd(source, full_matrices=False)
                keep = s > s[0] * 1e-6
                direction = u[:, keep] @ vh[keep]
            scale = math.sqrt(max(1., p.shape[0] / p.shape[1])) if self.shape_scale else 1.
            out[name] = direction * (lr * scale)
        return out

    def state_dict(self):
        return {n:t.cpu() for n,t in self.momentum.items()}

    def load_state_dict(self, state):
        for n,p in self.params:
            self.momentum[n].copy_(state[n].to(p.device))


class SpectralProposal:
    """Decompose the ACTUAL base step, including LR, NS amplitudes and scaling.

    With all coefficients one this reconstructs the same NS5/polar step.
    No fictional hard spectral cap is claimed for finite NS5.
    """
    @torch.no_grad()
    def __init__(self, parameters, directions):
        self.params = parameters
        self.parts = []
        offset = 0
        for name, p in parameters:
            d = directions.get(name)
            if d is None or not bool(torch.count_nonzero(d)):
                continue
            u,s,vh = torch.linalg.svd(d.float(), full_matrices=False)
            # Keep all thin modes, including tiny ones, to reconstruct the base
            # update. Their singular amplitudes stay tiny; none is promoted to 1.
            rank = len(s)
            self.parts.append((name, p, u*s, vh, slice(offset, offset+rank)))
            offset += rank
        self.size = offset

    @torch.no_grad()
    def coordinates(self, gradients):
        pieces = []
        for name,p,us,vh,sl in self.parts:
            g = gradients.get(name)
            c = torch.zeros(vh.shape[0], device=p.device) if g is None else torch.einsum(
                'mi,mn,in->i', us, g.float(), vh)
            pieces.append(c.detach().double().cpu().numpy())
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64)

    @torch.no_grad()
    def directions(self, z):
        return {name: (us * torch.as_tensor(z[sl], device=p.device, dtype=us.dtype)) @ vh
                for name,p,us,vh,sl in self.parts}

    def _curvature(self, scores, damping):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1, self.size)
        diag = np.zeros(self.size)
        for *_, sl in self.parts:
            diag[sl] = damping * max(float(np.mean(scores[:,sl] ** 2)), 1e-16)
        return Curvature(scores, diag)

    @staticmethod
    def _annotate(z, info):
        """Log the allocation itself. Without this, steps.jsonl cannot distinguish
        a zero step from a real one after the fact; see drift_cil.audit."""
        z = np.asarray(z, dtype=np.float64)
        info["z_inf"] = float(np.max(np.abs(z))) if z.size else 0.
        info["z_l1"] = float(np.sum(np.abs(z)))
        info["z_nonzero"] = int(np.count_nonzero(z))
        return z, info

    def allocate(self, b, a, scores, rho, damping=.03, lower=0., maxiter=500,
                 tolerance=1e-8):
        """`tolerance` MUST be the gate's atol. A solver held to rho+1e-8 while the
        gate enforces ceiling+1e-6 forfeits budget the arm is scored on."""
        if self.size == 0:
            return np.empty(0), {"status":"empty", "predicted_change":0., "objective":0.,
                                 "z_inf":0., "z_l1":0., "z_nonzero":0}
        return self._annotate(*solve_budget(b, a, self._curvature(scores, damping), rho,
                                            lower=lower, maxiter=maxiter, tolerance=tolerance))

    def allocate_scalar(self, b, a, scores, rho, damping=.03, lower=0., maxiter=500,
                        tolerance=1e-8):
        """The SAME budget, restricted to one global amplitude z = c * 1.

        This is the control the per-mode allocation has to beat. The restriction
        is a 1-D problem in c, NOT a re-solve with a diagonal curvature:

            max_c  c * (1.b)   s.t.  -(1.a) c + .5 c^2 (1.C.1) <= rho,  lower <= c <= 1

        1.C.1 keeps every cross-mode term and costs one matvec against the stored
        scores. Solved by the same `solve_budget`, so both arms share this
        tolerance, this status vocabulary and -- in the runner -- this gate.
        """
        if self.size == 0:
            return np.empty(0), {"status":"empty", "predicted_change":0., "objective":0.,
                                 "z_inf":0., "z_l1":0., "z_nonzero":0, "scalar_c":0.}
        curvature = self._curvature(scores, damping)
        ones = np.ones(self.size)
        b_s = float(np.sum(np.asarray(b, dtype=np.float64)))
        a_s = float(np.sum(np.asarray(a, dtype=np.float64)))
        c_s = float(ones @ curvature.mv(ones))
        collapsed = Curvature(np.array([[np.sqrt(max(c_s, 0.))]]), np.zeros(1))
        c, info = solve_budget(np.array([b_s]), np.array([a_s]), collapsed, rho,
                               lower=lower, maxiter=maxiter, tolerance=tolerance)
        z, info = self._annotate(np.full(self.size, float(c[0])), info)
        info.update(scalar_c=float(c[0]), scalar_b=b_s, scalar_a=a_s, scalar_curvature=c_s)
        return z, info


class Transaction:
    """Snapshot-based application: rejected FP32 steps restore BIT-EXACTLY."""
    def __init__(self, parameters, directions):
        self.params = parameters
        self.before = {n:p.detach().clone() for n,p in parameters}
        self.directions = directions

    @torch.no_grad()
    def apply(self, fraction=1.):
        for n,p in self.params:
            p.copy_(self.before[n])
            if n in self.directions:
                p.add_(self.directions[n].to(p.dtype), alpha=-fraction)

    @torch.no_grad()
    def restore(self):
        for n,p in self.params:
            p.copy_(self.before[n])


def accept_step(transaction, loss_fn, before_loss, ceiling, max_backtracks=8, atol=1e-6):
    """Check measured memory loss; allow labelled recovery if already over budget."""
    for attempt in range(max_backtracks+1):
        fraction = 0.5**attempt
        transaction.apply(fraction)
        after = float(loss_fn())
        feasible = np.isfinite(after) and after <= ceiling + atol
        recovery = np.isfinite(after) and before_loss > ceiling + atol and after < before_loss - atol
        if feasible or recovery:
            return {"accepted_scale":fraction, "backtracks":attempt, "memory_after":after,
                    "within_budget":feasible, "accept_status":"accepted" if feasible else "recovery"}
    transaction.restore()
    return {"accepted_scale":0., "backtracks":max_backtracks+1, "memory_after":before_loss,
            "within_budget":before_loss <= ceiling + atol, "accept_status":"rejected"}
