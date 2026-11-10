import sys, time, math
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import newtonian, channel_reference, l2_relative_error, run_channel

torch.set_num_threads(8)
nu, U = 0.1, 0.02
prev = None
for h in (16, 32, 64):
    H = h / 2
    a = 2 * nu * U / H**2
    cl = newtonian(nu)
    y = torch.arange(h + 2, dtype=torch.float64) - 0.5 - H
    uref_full = channel_reference(cl, a, H, y=y)["u"]
    uinit = uref_full.clone(); uinit[0] = 0; uinit[-1] = 0
    out = run_channel(h, a, cl, nx=4, n_steps=9000, tol=1e-9, check_every=250, u_init=uinit)
    rows = out["fluid_rows"]
    err = l2_relative_error(out["u"][rows].double(), uref_full[rows])
    du = [f"{float(out['u'][i]-uref_full[i]):+.2e}" for i in (1, 2, 3, int(h/4), int(h/2))]
    order = "" if prev is None else f" p={math.log(prev/err)/math.log(1.5):.2f}"
    print(f"h={h:3d} a={a:.4g} steps={out['steps']:6d} L2={err*100:7.3f}%{order}  wall-node diffs {du}", flush=True)
    prev = err