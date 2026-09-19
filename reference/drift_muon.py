"""
Drift-budgeted spectral allocation -- v0, in ADAPTER-FACTOR coordinates.

Every quantity below lives in the coordinates of the parameter actually updated
(each trainable LoRA factor as its own matrix), matching how the baseline applies
Muon to "every eligible matrix-valued parameter".  Nothing is reconstructed from
other coordinates.

For one matrix parameter P with Nesterov momentum M = U diag(s) V^T (thin SVD,
so at most min(P.shape) modes), the update is

    Delta P = - sum_i z_i u_i v_i^T ,      z in [z_lo, tau]^r

and, to second order in the same coordinates,

    Delta L_new  ~  - b^T z            b_i = <u_i v_i^T, grad L_new>
    Delta L_old  ~  - a^T z + ½ z^T C z    a_i = <u_i v_i^T, grad L_old>
                                           C   = old-task curvature in this basis

so the step solves

    max_z  b^T z    s.t.   - a^T z + ½ z^T C z <= rho ,   z_lo <= z <= tau

with rho the budget REMAINING from the post-old-task checkpoint:
rho = delta - (L_old(theta_t) - L_old(theta_old_ckpt)).  The budget does not
renew each step, so small degradations cannot accumulate without bound.

z = tau * 1  recovers Muon (up to the polar-vs-SVD distinction), which is what
the scalar-step-size control arm scales.  That arm is the decisive baseline:
if c * tau * 1 matches this allocation at the same realised budget, the per-mode
allocation is not earning its keep.

C is a curvature APPROXIMATION, not a certified bound, so `step` re-measures the
old-task loss after the tentative step and backtracks when the realised increase
overshoots.  Fisher is symmetric in z -> -z and therefore carries no sign
information; the linear term a is what lets a step that HELPS the old task pay
for its own curvature cost.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import torch

# --------------------------------------------------------------------------- #
#  the allocator: pure math, no torch modules, unit-testable on its own
# --------------------------------------------------------------------------- #

def _box_argmax_quadratic(lin: torch.Tensor, C: torch.Tensor, mu: float,
                          z_lo: float, tau: float, iters: int = 300) -> torch.Tensor:
    """argmax over the box of   lin^T z - (mu/2) z^T C z   (concave for mu>=0, C psd)."""
    r = lin.shape[0]
    if mu <= 0:
        return torch.where(lin > 0, torch.full_like(lin, tau), torch.full_like(lin, z_lo))
    L = mu * float(torch.linalg.matrix_norm(C, 2)) + 1e-30
    z = torch.full((r,), min(max(0.0, z_lo), tau), dtype=lin.dtype)
    for _ in range(iters):
        z = (z + (lin - mu * (C @ z)) / L).clamp(z_lo, tau)
    return z


def allocate(b: torch.Tensor, a: torch.Tensor, C: torch.Tensor, rho: float,
             tau: float = 1.0, z_lo: float = 0.0,
             iters: int = 300, bisect: int = 60) -> Tuple[torch.Tensor, Dict]:
    """
    max b^T z  s.t.  g(z) = -a^T z + 0.5 z^T C z <= rho,  z_lo <= z <= tau.

    Returns (z, info).  info['status'] is one of
      'unconstrained' : the box optimum already satisfies the budget
      'active'        : the budget constraint is tight
      'recovery'      : rho < 0 and a feasible z exists only by improving the old task
      'infeasible'    : no z in the box meets the budget -> z is the best available
                        reduction of g (the least-harm step), or z_lo if none helps
    """
    b = b.double(); a = a.double(); C = C.double()
    g = lambda z: float(-a @ z + 0.5 * z @ C @ z)

    z0 = _box_argmax_quadratic(b, C, 0.0, z_lo, tau)
    if g(z0) <= rho:
        return z0, dict(status='unconstrained', mu=0.0, g=g(z0), obj=float(b @ z0))

    # minimise g over the box, to see whether ANY feasible point exists
    zmin = _box_argmax_quadratic(a, C, 1.0, z_lo, tau, iters)   # max a^T z - ½ z^T C z
    if g(zmin) > rho:
        best = zmin if g(zmin) < 0 else torch.full_like(b, z_lo)
        return best, dict(status='infeasible', mu=float('inf'), g=g(best),
                          obj=float(b @ best), g_best=g(zmin))

    lo, hi = 0.0, 1.0
    while g(_box_argmax_quadratic(b + hi * a, C, hi, z_lo, tau, iters)) > rho and hi < 1e14:
        hi *= 4.0
    for _ in range(bisect):
        mid = (lo + hi) / 2 if lo > 0 else hi / 2
        zm = _box_argmax_quadratic(b + mid * a, C, mid, z_lo, tau, iters)
        if g(zm) > rho: lo = mid
        else: hi = mid
    z = _box_argmax_quadratic(b + hi * a, C, hi, z_lo, tau, iters)
    st = 'recovery' if rho < 0 else 'active'
    return z, dict(status=st, mu=hi, g=g(z), obj=float(b @ z))


# --------------------------------------------------------------------------- #
#  per-parameter spectral machinery
# --------------------------------------------------------------------------- #

@dataclass
class MatrixState:
    mom: torch.Tensor
    U: Optional[torch.Tensor] = None
    s: Optional[torch.Tensor] = None
    Vt: Optional[torch.Tensor] = None
    fisher_scores: List[torch.Tensor] = field(default_factory=list)


def coords(U: torch.Tensor, Vt: torch.Tensor, Gmat: torch.Tensor) -> torch.Tensor:
    """<u_i v_i^T, G> for every mode i, without forming any r x r product."""
    return torch.einsum('ai,ab,ib->i', U, Gmat.to(U.dtype), Vt)


class DriftMuon:
    """
    Usage per optimisation step:
        opt.accumulate_grad()                      # after backward on the NEW task
        opt.refresh_spectral()                     # thin SVD of the momentum
        opt.set_old_task_signals(grad_fn, score_fn)  # a and C, in the same basis
        z, info = opt.propose(rho)
        opt.apply(z, eta)                          # tentative
        ... measure old-task loss on memory ...
        opt.backtrack(factor)                      # if the realised increase overshoots
    """

    def __init__(self, params: List[Tuple[str, torch.nn.Parameter]],
                 beta: float = 0.95, nesterov: bool = True, tau: float = 1.0,
                 z_lo: float = 0.0, damping: float = 1e-2):
        self.params = [(n, p) for n, p in params if p.ndim == 2]
        self.beta, self.nesterov, self.tau, self.z_lo = beta, nesterov, tau, z_lo
        self.damping = damping
        self.state = {n: MatrixState(mom=torch.zeros_like(p)) for n, p in self.params}
        self._last = None

    @torch.no_grad()
    def accumulate_grad(self):
        for n, p in self.params:
            if p.grad is None: continue
            st = self.state[n]
            st.mom.mul_(self.beta).add_(p.grad)

    @torch.no_grad()
    def _effective_momentum(self, n: str, p) -> torch.Tensor:
        st = self.state[n]
        return st.mom * self.beta + p.grad if (self.nesterov and p.grad is not None) else st.mom

    @torch.no_grad()
    def refresh_spectral(self):
        for n, p in self.params:
            M = self._effective_momentum(n, p).double()
            U, s, Vt = torch.linalg.svd(M, full_matrices=False)
            tol = s[0] * max(M.shape) * torch.finfo(torch.float64).eps * 10
            r = max(1, int((s > tol).sum()))
            st = self.state[n]
            st.U, st.s, st.Vt = U[:, :r].contiguous(), s[:r].contiguous(), Vt[:r].contiguous()

    @torch.no_grad()
    def new_task_coords(self) -> Dict[str, torch.Tensor]:
        return {n: coords(self.state[n].U, self.state[n].Vt, p.grad.double())
                for n, p in self.params if p.grad is not None}

    @torch.no_grad()
    def push_fisher_score(self):
        """call once per sampled-label backward on OLD-task data (one score per token)"""
        for n, p in self.params:
            if p.grad is None: continue
            st = self.state[n]
            st.fisher_scores.append(coords(st.U, st.Vt, p.grad.double()))

    @torch.no_grad()
    def curvature(self, n: str) -> torch.Tensor:
        st = self.state[n]
        S = torch.stack(st.fisher_scores)
        C = (S.T @ S) / S.shape[0]
        return C + self.damping * float(torch.diagonal(C).mean()) * torch.eye(
            C.shape[0], dtype=C.dtype)

    @torch.no_grad()
    def clear_fisher(self):
        for n, _ in self.params: self.state[n].fisher_scores.clear()

    @torch.no_grad()
    def propose(self, rho: float, b: Dict[str, torch.Tensor],
                a: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """rho is the TOTAL remaining budget; it is split across matrices in
           proportion to each one's unconstrained predicted cost."""
        Cs = {n: self.curvature(n) for n, _ in self.params}
        zfree, gfree = {}, {}
        for n, _ in self.params:
            z, _ = allocate(b[n], a[n], Cs[n], float('inf'), self.tau, self.z_lo)
            zfree[n] = z
            gfree[n] = float(-a[n].double() @ z + 0.5 * z @ Cs[n] @ z)
        pos = sum(max(v, 0.0) for v in gfree.values())
        out, info = {}, {}
        for n, _ in self.params:
            share = (max(gfree[n], 0.0) / pos) if pos > 0 else 1.0 / len(self.params)
            out[n], info[n] = allocate(b[n], a[n], Cs[n], rho * share, self.tau, self.z_lo)
        self._last = out
        return out, info

    @torch.no_grad()
    def apply(self, z: Dict[str, torch.Tensor], eta: float):
        self._applied = (z, eta)
        for n, p in self.params:
            st = self.state[n]
            D = (st.U * z[n]) @ st.Vt
            p.data.add_(D.to(p.dtype), alpha=-eta)

    @torch.no_grad()
    def backtrack(self, shrink: float):
        """undo a fraction of the last step: shrink in (0,1] keeps `shrink` of it"""
        z, eta = self._applied
        for n, p in self.params:
            st = self.state[n]
            D = (st.U * z[n]) @ st.Vt
            p.data.add_(D.to(p.dtype), alpha=eta * (1.0 - shrink))
        self._applied = (z, eta * shrink)

    @torch.no_grad()
    def muon_z(self) -> Dict[str, torch.Tensor]:
        """the scalar-control baseline's direction: z = tau * 1 on the true rank"""
        return {n: torch.full((self.state[n].s.shape[0],), self.tau, dtype=torch.float64)
                for n, _ in self.params}
