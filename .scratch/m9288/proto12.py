import sys, time, math
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import power_law, newtonian, channel_reference, l2_relative_error, run_channel
from tensorlbm.nonnewtonian_frontier.reference import power_law_velocity

torch.set_num_threads(8)
UM, nu_wall = 0.005, 0.1


def case(h, n):
    H = h / 2
    y = torch.arange(h + 2, dtype=torch.float64) - 0.5 - H
    if n == 1.0:
        cl = newtonian(nu_wall)
        a = 2 * nu_wall * UM / H ** 2
        uref = channel_reference(cl, a, H, y=y)["u"][1:-1]
    else:
        gw = UM * (n + 1) / (n * H)
        K = nu_wall * gw ** (1 - n)
        a = K * gw ** n / H
        cl = power_law(K, n)
        uref = power_law_velocity(y[1:-1], H, a, K, n)
    t0 = time.time()
    out = run_channel(h, a, cl, nx=4, n_steps=200000, tol=1e-5, check_every=500)
    e = l2_relative_error(out["u"][1:-1].double(), uref)
    return e, out["steps"], time.time() - t0, out["converged"], out["residual"]


for n in (1.0, 0.7, 1.3):
    prev = None
    print(f"--- n={n} u_max={UM}", flush=True)
    for h in (16, 32, 64):
        e, steps, dt, conv, resid = case(h, n)
        p = "" if prev is None else f" p={math.log(prev/e)/math.log(2):.2f}"
        print(f"  h={h} steps={steps} conv={conv} resid={resid:.1e} L2={e*100:.4f}%{p} t={dt:.0f}s", flush=True)
        prev = e