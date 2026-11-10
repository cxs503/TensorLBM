import sys, math, time
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import stream
from tensorlbm.herschel_bulkley_lbm import wall_aware_gradient, moving_wall_bounce_back_2d
from tensorlbm.powerlaw import collide_powerlaw_bgk_forced

torch.set_num_threads(4)


def run(h, a, nu_of_gamma, nx=4, n_steps=200000, tol=1e-9, u_init=None, nu_max=20.0, nu_min=1e-4):
    ny = h + 2
    f = torch.zeros((9, ny, nx))
    rho0 = torch.ones((ny, nx))
    zero = torch.zeros_like(rho0)
    uu = zero if u_init is None else u_init
    f[:] = equilibrium(rho0, uu.unsqueeze(1).expand(ny, nx), zero)
    lower = torch.zeros((ny, nx), dtype=torch.bool); lower[0] = True
    upper = torch.zeros((ny, nx), dtype=torch.bool); upper[-1] = True
    fluid = torch.ones((ny, nx), dtype=torch.bool); fluid[0] = False; fluid[-1] = False
    y = torch.arange(ny, dtype=torch.float64) - 0.5
    u_prev = None
    steps = 0
    for step in range(1, n_steps + 1):
        _, ux, uy = macroscopic(f)
        g = torch.abs(wall_aware_gradient(ux, fluid, 0))
        nu = torch.clamp(nu_of_gamma(g), min=nu_min, max=nu_max)
        tau = 3.0 * nu + 0.5
        f = collide_powerlaw_bgk_forced(f, a, 1.0, 1.0, tau_field=tau)
        f = stream(f)
        f = moving_wall_bounce_back_2d(f, lower, 0.0, 0.0)
        f = moving_wall_bounce_back_2d(f, upper, 0.0, 0.0)
        steps = step
        if step % 200 == 0:
            _, ux, _ = macroscopic(f)
            prof = ux.mean(dim=1)[1:-1]
            if u_prev is not None:
                d = float((prof - u_prev).abs().max())
                if d < tol * max(1e-30, float(prof.abs().max())):
                    break
            u_prev = prof
    _, ux, _ = macroscopic(f)
    return ux.mean(dim=1), y, steps


def pl_analytic(y, H, G, K, n):
    pref = (n / (n + 1.0)) * (G / K) ** (1.0 / n)
    return pref * (H ** ((n + 1) / n) - torch.abs(y) ** ((n + 1) / n))


H = 20.0
h = int(2 * H)
U = 0.02
for n in (0.7, 1.0, 1.3):
    gw = U * (n + 1) / (n * H)          # wall shear rate from u_max = n/(n+1) gw H
    nu_wall = 0.1
    K = nu_wall * gw ** (1 - n)
    G = K * gw ** n / H
    yy = torch.tensor([i - 0.5 for i in range(h + 2)], dtype=torch.float64)
    uex = pl_analytic(yy, H, G, K, n)
    uinit = torch.zeros(h + 2); uinit[1:-1] = uex[1:-1].float()
    print(f"n={n} K={K:.5g} G={G:.4g} gw={gw:.4g} umax_a={float(uex.max()):.4g} nu_wall={nu_wall}")
    t0 = time.time()
    prof, y, steps = run(h, G, lambda g, K=K, n=n: K * torch.pow(torch.clamp(g, min=1e-12), n - 1.0),
                         tol=1e-10, u_init=uinit)
    yfl = y[1:-1]
    uexf = pl_analytic(yfl, H, G, K, n)
    err = float(torch.sqrt(((prof[1:-1] - uexf) ** 2).mean()) / torch.sqrt((uexf ** 2).mean()))
    print(f"   steps={steps} L2err={err*100:.3f}%  umax={float(prof[1:-1].max()):.5f} vs {float(uexf.max()):.5f} t={time.time()-t0:.1f}s")