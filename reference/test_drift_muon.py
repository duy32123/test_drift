"""Self-tests for drift_muon.  Run before spending GPU time."""
import sys, torch, numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-claude/2224c3a5-1968-5905-9aa3-d0dabe6dd10e/scratchpad')
from drift_muon import allocate, coords, DriftMuon
torch.manual_seed(0); torch.set_num_threads(4)
OK = [True]
def check(name, cond, extra=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")

def rand_psd(r, rank=None, g=None):
    rank = rank or r
    A = torch.randn(rank, r, generator=g, dtype=torch.float64)
    return (A.T @ A) / rank

g = torch.Generator().manual_seed(1)
gfun = lambda a, C, z: float(-a @ z + 0.5 * z @ C @ z)

print("1. the returned z always satisfies the budget when the problem is feasible")
worst = 0.0; nact = 0
for _ in range(300):
    r = int(torch.randint(3, 40, (1,), generator=g))
    b = torch.rand(r, generator=g, dtype=torch.float64)
    a = torch.randn(r, generator=g, dtype=torch.float64) * 0.1
    C = rand_psd(r, max(1, r // 3), g)
    zf, _ = allocate(b, a, C, float('inf'))
    rho = float(torch.rand(1, generator=g)) * gfun(a, C, zf)
    z, info = allocate(b, a, C, rho)
    if info['status'] in ('active', 'unconstrained'):
        nact += 1
        worst = max(worst, gfun(a, C, z) - rho)
check("budget respected", worst < 1e-6, f"worst violation {worst:.2e} over {nact} cases")

print("2. a loose budget reproduces the Muon-like box optimum")
r = 20
b = torch.rand(r, generator=g, dtype=torch.float64); a = torch.zeros(r, dtype=torch.float64)
C = rand_psd(r, r, g)
z, info = allocate(b, a, C, 1e9, tau=1.0)
check("status", info['status'] == 'unconstrained', info['status'])
check("z = tau on every positive-b mode", bool(torch.allclose(z, torch.ones(r, dtype=torch.float64))))

print("3. a tight budget makes the constraint active (equality)")
zf, _ = allocate(b, a, C, float('inf'))
rho = 0.3 * gfun(a, C, zf)
z, info = allocate(b, a, C, rho)
check("status", info['status'] == 'active', info['status'])
check("constraint tight", abs(info['g'] - rho) / abs(rho) < 1e-3,
      f"g={info['g']:.6f} rho={rho:.6f}")

print("4. KKT: z maximises the Lagrangian over the box at the returned mu")
mu = info['mu']
lin = b + mu * a
resid = 0.0
for i in range(r):
    gi = float(lin[i] - mu * (C @ z)[i])
    if z[i] > 1e-9 and z[i] < 1.0 - 1e-9: resid = max(resid, abs(gi))   # interior -> grad 0
    elif z[i] <= 1e-9: resid = max(resid, max(gi, 0.0))                 # at lower bound -> grad<=0
    else: resid = max(resid, max(-gi, 0.0))                             # at upper bound -> grad>=0
check("stationarity residual small", resid < 1e-4, f"max residual {resid:.2e}")

print("5. objective is monotone non-decreasing in the budget")
prev = -1e18; mono = True
for f in [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0]:
    z2, i2 = allocate(b, a, C, f * gfun(a, C, zf))
    if float(b @ z2) < prev - 1e-7: mono = False
    prev = float(b @ z2)
check("monotone in rho", mono)

print("6. a step that HELPS the old task can pay for its own curvature")
r = 10
b = torch.ones(r, dtype=torch.float64)
a = torch.zeros(r, dtype=torch.float64); a[0] = 1.0          # mode 0 improves the old task
C = torch.eye(r, dtype=torch.float64) * 0.5                  # curvature large enough that
z_free, _ = allocate(b, a, C, float('inf'))                  # the free step overshoots ...
check("free step is NOT feasible here", gfun(a, C, z_free) > -0.2,
      f"g(free)={gfun(a,C,z_free):.3f}")
z, info = allocate(b, a, C, -0.2)                            # ... budget already overspent
check("recovery status", info['status'] == 'recovery', info['status'])
check("recovery leans on the helpful mode", float(z[0]) > float(z[1:].max()) + 1e-9,
      f"z0={float(z[0]):.3f} max(rest)={float(z[1:].max()):.3f}")
check("budget met by recovery", info['g'] <= -0.2 + 1e-6, f"g={info['g']:.4f}")

print("7. genuinely infeasible -> flagged, and the returned step is the least-harm one")
a = torch.full((r,), -1.0, dtype=torch.float64)              # every mode hurts
z, info = allocate(b, a, torch.eye(r, dtype=torch.float64), -5.0)
check("flagged infeasible", info['status'] == 'infeasible', info['status'])
check("least-harm step is z_lo", float(z.abs().max()) < 1e-12)

print("8. sign convention against a brute-force quadratic model")
r = 6
U = torch.linalg.qr(torch.randn(12, r, generator=g, dtype=torch.float64))[0]
Vt = torch.linalg.qr(torch.randn(9, r, generator=g, dtype=torch.float64))[0].T
GA = torch.randn(12, 9, generator=g, dtype=torch.float64)
a = coords(U, Vt, GA)
z = torch.rand(r, generator=g, dtype=torch.float64)
D = (U * z) @ Vt
lin_direct = float((GA * (-D)).sum())            # <grad L_old, Delta P> with Delta P = -D
check("a^T z == -<grad_old, Delta P>", abs(lin_direct + float(a @ z)) < 1e-10,
      f"{lin_direct:.6e} vs {-float(a @ z):.6e}")

print("9. coords() matches the explicit u_i^T G v_i")
explicit = torch.tensor([float(U[:, i] @ GA @ Vt[i]) for i in range(r)], dtype=torch.float64)
check("coords correct", float((explicit - a).abs().max()) < 1e-12)

print("10. end-to-end on a tiny module: momentum, SVD rank, apply/backtrack")
lin = torch.nn.Linear(9, 12, bias=False).double()
optd = DriftMuon([('w', lin.weight)], beta=0.9, tau=1.0, damping=1e-2)
x = torch.randn(64, 9, dtype=torch.float64)
tgt = torch.randn(64, 12, dtype=torch.float64)
for _ in range(3):
    lin.zero_grad(); ((lin(x) - tgt) ** 2).mean().backward(); optd.accumulate_grad()
optd.refresh_spectral()
check("thin rank <= min(dim)", optd.state['w'].s.shape[0] <= 9, f"r={optd.state['w'].s.shape[0]}")
for _ in range(40):
    lin.zero_grad(); (lin(x).gather(1, torch.randint(0, 12, (64, 1))).mean()).backward()
    optd.push_fisher_score()
bb = optd.new_task_coords()
aa = {'w': torch.zeros_like(bb['w'])}
before = lin.weight.detach().clone()
z, info = optd.propose(1e9, bb, aa)
optd.apply(z, 1e-3)
moved = float((lin.weight.detach() - before).abs().max())
optd.backtrack(0.0)
restored = float((lin.weight.detach() - before).abs().max())
check("apply moves the weight", moved > 0)
check("backtrack(0) restores exactly", restored < 1e-12, f"residual {restored:.2e}")
Cw = optd.curvature('w')
check("curvature is psd", float(torch.linalg.eigvalsh(Cw).min()) > -1e-12)

print()
print("ALL PASS" if OK[0] else "SOME TESTS FAILED")
sys.exit(0 if OK[0] else 1)
