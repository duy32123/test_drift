"""One GLOBAL loss-budget constraint, including cross-parameter Fisher terms.

z is dimensionless; b, a and score columns already include the applied LR,
matrix shape scaling and the singular amplitudes of the base update.
The CPU solver stores N x d scores, never a dense d x d Fisher.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass
class Curvature:
    scores: np.ndarray
    diagonal: np.ndarray

    def mv(self, z):
        return self.scores.T @ (self.scores @ z) / max(1, len(self.scores)) + self.diagonal * z

    def quad(self, z):
        return float(z @ self.mv(z))


def solve_budget(b, a, curvature, rho, lower=0.0, upper=1.0,
                 maxiter=500, bisections=40, tolerance=1e-8):
    """Maximize b.z under -a.z + .5 z.C.z <= rho and box bounds.

    L-BFGS-B solves each concave dual subproblem. The returned diagnostic
    explicitly distinguishes numerical failure from certified feasibility.
    'infeasible_candidate' is NOT a proof of infeasibility.
    """
    b, a = np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64)
    if b.shape != a.shape or b.ndim != 1 or not np.isfinite(b).all() or not np.isfinite(a).all():
        raise ValueError("a and b must be finite vectors of equal size")
    if lower > 0 or upper < 0 or lower >= upper:
        raise ValueError("The box must contain zero and have positive width")
    if len(b) == 0:
        return b.copy(), {"status": "empty", "predicted_change": 0.0, "objective": 0.0}
    if not np.isfinite(curvature.scores).all() or not np.isfinite(curvature.diagonal).all():
        raise ValueError("Non-finite curvature")
    if np.any(curvature.diagonal < 0):
        raise ValueError("Curvature damping must be nonnegative")
    def cost(z):
        return -float(a @ z) + .5 * curvature.quad(z)
    free = np.where(b > 0, upper, np.where(b < 0, lower, 0.0))
    def result(z, status, mu=0., residual=0., success=True):
        return z, {"status": status, "predicted_change": cost(z),
                   "objective": float(b @ z), "dual": float(mu),
                   "box_kkt_residual": float(residual), "inner_success": bool(success),
                   "feasible": bool(cost(z) <= rho + tolerance)}
    if cost(free) <= rho + tolerance:
        return result(free, "unconstrained")
    objective_scale = max(float(np.max(np.abs(b))), 1e-12)
    cost_scale = max(abs(cost(free)), float(np.max(np.abs(a))),
                     float(np.max(curvature.diagonal)), abs(rho), 1e-12)
    bn, an, rn = b / objective_scale, a / cost_scale, rho / cost_scale
    def cn(z):
        return curvature.mv(z) / cost_scale
    bounds = [(lower, upper)] * len(b)
    def inner(mu, start, recovery=False):
        def fun(z):
            cz = cn(z)
            g = -an @ z + .5 * z @ cz
            if recovery:
                return float(g), -an + cz
            return float(-bn @ z + mu * g), -bn + mu * (-an + cz)
        opt = minimize(fun, np.clip(start, lower, upper), jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": maxiter, "maxls": 40,
                       "ftol": 1e-14, "gtol": 1e-9, "maxcor": 15})
        z = np.clip(opt.x, lower, upper)
        grad = fun(z)[1]
        residual = np.max(np.abs(z - np.clip(z - grad, lower, upper)))
        return z, float(residual), bool(opt.success)
    minimum, _, min_success = inner(1., np.zeros_like(b), True)
    if cost(minimum) > rho + tolerance:
        # Never apply this without an actual-loss acceptance check.
        if cost(minimum) > 0:
            minimum = np.zeros_like(b)
        return result(minimum, "infeasible_candidate", success=min_success)
    lo, hi = 0., 1.
    candidate = minimum.copy()
    residual, success = 0., True
    # ONE budget everywhere below. The bisection used to target a strict `rho`
    # while the caller's feasibility check, the final guard and accept_step all
    # used `rho + tolerance`; the step returned was then held to a budget the
    # gate did not enforce, which matters once rho falls to the order of
    # `tolerance` -- exactly where a non-renewing budget spends its time.
    ceiling = rho + tolerance
    for _ in range(48):
        z, residual, success = inner(hi, candidate)
        candidate = z
        if cost(z) <= ceiling:
            break
        hi *= 4.
    else:
        return result(minimum, "dual_bracket_failure", hi, residual, False)
    feasible = candidate.copy()
    for _ in range(bisections):
        mid = (lo + hi) / 2
        z, residual, success = inner(mid, feasible)
        if cost(z) > ceiling:
            lo = mid
        else:
            hi, feasible = mid, z
    z, residual, success = inner(hi, feasible)
    if cost(z) > ceiling:
        z = feasible
    status = "recovery" if rho < 0 else "active"
    if residual > 1e-5:
        status += "_approximate"
    return result(z, status, hi, residual, success)
