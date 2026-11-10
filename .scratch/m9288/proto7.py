import sys, time, math
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import power_law, l2_relative_error, run_channel
from tensorlbm.nonnewtonian_frontier.reference import power_law_velocity
torch.set_num_threads(8)
U, n, nu_wall = 0.02, 0.7, 0.1
prev = None
for h in (16, 32, 64):
    H = h/2
    gw = U*(n+1)/(n*H); K = nu_wall*gw**(1-n); a = K*gw**n/H
    cl = power_law(K, n)
    y = torch.arange(h+2, dtype=torch.float64)-0.5-H
    uref = power_law_velocity(y[1:-1], H, a, K, n)
    t0=time.time()
    out = run_channel(h, a, cl, nx=4, n_steps=60000, tol=1e-5, check_every=500)
    e = l2_relative_error(out["u"][1:-1].double(), uref)
    p = "" if prev is None else f" p={math.log(prev/e)/math.log(2):.2f}"
    print(f"h={h} steps={out['steps']} L2={e*100:.3f}%{p} t={time.time()-t0:.0f}s", flush=True)
    prev = e
