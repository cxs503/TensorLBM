import sys, math
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import newtonian, channel_reference, l2_relative_error, run_channel
torch.set_num_threads(8)
nu, UM = 0.1, 0.02
prev_all = prev_int = None
for h in (16, 32, 64):
    H = h / 2
    a = 2 * nu * UM / H ** 2
    cl = newtonian(nu)
    y = torch.arange(h + 2, dtype=torch.float64) - 0.5 - H
    ref = channel_reference(cl, a, H, y=y)
    out = run_channel(h, a, cl, nx=4, n_steps=200000, tol=1e-5, check_every=500)
    u = out["u"][1:-1].double(); ur = ref["u"][1:-1]
    e_all = l2_relative_error(u, ur)
    e_int = l2_relative_error(u[2:-2], ur[2:-2])
    d = (u - ur) / ur.max()
    # effective amplitude and wall-offset fits
    yy = y[1:-1]
    A = (u * (H ** 2 - yy ** 2)).sum() / ((H ** 2 - yy ** 2) ** 2).sum()
    ratio = float(u.max() / ur.max())
    print(f"h={h} steps={out['steps']} conv={out['converged']} resid={out['residual']:.1e} "
          f"L2_all={e_all*100:.4f}% L2_interior={e_int*100:.4f}% amp_ratio={ratio:.6f} "
          f"d_first={float(d[0]):+.2e} d_mid={float(d[len(d)//2]):+.2e} d_lastint={float(d[-2]):+.2e}", flush=True)
    if prev_all:
        print(f"   orders: all p={math.log(prev_all/e_all)/math.log(2):.2f} interior p={math.log(prev_int/e_int)/math.log(2):.2f}", flush=True)
    prev_all, prev_int = e_all, e_int