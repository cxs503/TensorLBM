import sys, time
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import (
    power_law, newtonian, channel_reference, power_law_velocity, l2_relative_error, run_channel,
)

torch.set_num_threads(8)
U, nu_wall = 0.02, 0.1

def powerlaw_case(h, n, U=U, nu_wall=nu_wall):
    H = h / 2
    gw = U * (n + 1) / (n * H)
    K = nu_wall * gw ** (1 - n)
    a = K * gw ** n / H
    return power_law(K, n), a, H

for (h, n) in [(16, 0.7), (16, 1.0), (16, 1.3)]:
    cl, a, H = powerlaw_case(h, n)
    y = torch.arange(h + 2, dtype=torch.float64) - 0.5 - H
    ref = channel_reference(cl, a, H, y=y)
    uinit = ref["u"].clone(); uinit[0] = 0; uinit[-1] = 0
    t0 = time.time()
    out = run_channel(h, a, cl, nx=4, n_steps=60000, tol=1e-7, check_every=500, u_init=uinit)
    rows = out["fluid_rows"]
    u_num = out["u"][rows].double()
    if n == 1.0:
        u_ref = channel_reference(newtonian(nu_wall), a, H, y=y)["u"][rows]
    else:
        u_ref = power_law_velocity(y[rows], H, a, cl.params["K"], n)
    err = l2_relative_error(u_num, u_ref)
    print(f"h={h} n={n} K={cl.params['K']:.4g} a={a:.4g} steps={out['steps']} conv={out['converged']} "
          f"resid={out['residual']:.2e} L2={err*100:.3f}% umax={out['u_max']:.5f}/{float(u_ref.max()):.5f} t={time.time()-t0:.1f}s", flush=True)