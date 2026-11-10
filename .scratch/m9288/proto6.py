import sys, time
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import power_law, channel_reference, l2_relative_error, run_channel
from tensorlbm.nonnewtonian_frontier.reference import power_law_velocity

torch.set_num_threads(8)
U, nu_wall, n = 0.02, 0.1, 0.7
for h in (32,):
    H = h / 2
    gw = U * (n + 1) / (n * H)
    K = nu_wall * gw ** (1 - n)
    a = K * gw ** n / H
    cl = power_law(K, n)
    y = torch.arange(h + 2, dtype=torch.float64) - 0.5 - H
    rr = channel_reference(cl, a, H, y=y)
    uinit = rr["u"].clone(); uinit[0] = 0; uinit[-1] = 0
    uref = power_law_velocity(y[1:-1], H, a, K, n)
    print(f"h={h} K={K:.5g} a={a:.4g} u_max_ref={rr['u_max']:.5f}", flush=True)
    t0 = time.time()
    for nst in (3000, 6000, 12000, 24000):
        out = run_channel(h, a, cl, nx=4, n_steps=nst, tol=1e-14, check_every=nst, u_init=uinit)
        e = l2_relative_error(out["u"][1:-1].double(), uref)
        print(f"  cumulative steps={nst} L2={e*100:.3f}% umax={out['u_max']:.5f} t={time.time()-t0:.0f}s", flush=True)
        uinit = out["u"].clone().float()