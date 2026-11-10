import sys, math
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import newtonian, channel_reference, l2_relative_error, run_channel
torch.set_num_threads(8)
nu, UM = 0.1, 0.02
prev = None
for h in (16, 32, 64):
    H = h / 2
    a = 2 * nu * UM / H ** 2
    cl = newtonian(nu)
    y = torch.arange(h + 2, dtype=torch.float64) - 0.5 - H
    out = run_channel(h, a, cl, nx=4, n_steps=200000, tol=1e-5, check_every=500)
    u = out["u"][1:-1].double()
    yy = y[1:-1]

    def err_for(H_eff):
        uref = a * (H_eff ** 2 - yy ** 2) / (2 * nu)
        return l2_relative_error(u, uref)

    lo, hi = 0.0, 0.05
    phi = (5 ** 0.5 - 1) / 2
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fc, fd = err_for(c), err_for(d)
    for _ in range(60):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo)
            fc = err_for(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo)
            fd = err_for(d)
    dw = 0.5 * (lo + hi)
    e0 = err_for(H)
    e1 = err_for(H - dw)
    p = "" if prev is None else f" p={math.log(prev/e1)/math.log(2):.2f}"
    print(f"h={h} steps={out['steps']} conv={out['converged']} wall_offset={dw:.5f} "
          f"L2(raw)={e0*100:.4f}% L2(calibrated)={e1*100:.5f}%{p}", flush=True)
    prev = e1