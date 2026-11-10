import sys, time
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import power_law, channel_reference, l2_relative_error, run_channel
from tensorlbm.nonnewtonian_frontier.reference import power_law_velocity

torch.set_num_threads(8)
U, nu_wall, n = 0.02, 0.1, 0.7
h = 12
H = h / 2
gw = U * (n + 1) / (n * H)
K = nu_wall * gw ** (1 - n)
a = K * gw ** n / H
cl = power_law(K, n)
y = torch.arange(h + 2, dtype=torch.float64) - 0.5 - H
uinit = channel_reference(cl, a, H, y=y)["u"].clone(); uinit[0] = 0; uinit[-1] = 0
print(f"h={h} n={n} K={K:.5g} a={a:.4g} gw={gw:.4g}", flush=True)
t0 = time.time()
for nst in (2000, 4000, 8000, 16000):
    out = run_channel(h, a, cl, nx=4, n_steps=nst, tol=1e-12, check_every=nst, u_init=uinit)
    rows = out["fluid_rows"]
    uref = power_law_velocity(y[rows], H, a, K, n)
    print(f"  steps={out['steps']} L2={l2_relative_error(out['u'][rows].double(), uref)*100:.3f}% "
          f"umax={out['u_max']:.5f}/{float(uref.max()):.5f} t={time.time()-t0:.1f}s", flush=True)
    uinit = out["u"].clone().float()