import sys
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import newtonian, channel_reference, run_channel
from tensorlbm.nonnewtonian_frontier.channel import solution_residual, shear_rate_profile

torch.set_num_threads(8)
h, UM = 32, 0.02
H = h / 2
for nu in (0.05, 0.1, 0.2):
    a = 2 * nu * UM / H ** 2
    cl = newtonian(nu)
    y = torch.arange(h + 1, dtype=torch.float64) - H
    out = run_channel(h, a, cl, nx=4, n_steps=200000, tol=1e-5, check_every=500)
    ref = channel_reference(cl, a, H, y=y)
    u = out["u"][1:-1].double(); ur = ref["u"][1:-1]
    ratio = float(u.max() / ur.max())
    res = solution_residual(out["y"], out["gamma_dot"].mean(dim=1), cl, a, H)
    g = out["gamma_dot"].mean(dim=1)[1:-1].double()
    gref = ref["gamma_dot"][1:-1]
    print(f"nu={nu} a={a:.3e} steps={out['steps']} amp_ratio={ratio:.6f} "
          f"res_max={float(res.abs().max()):.2e} res_center={float(res[len(res)//2]):.2e}", flush=True)
    print("   gamma_lbm:", [f"{float(v):.4e}" for v in g[:3]], "... ref:", [f"{float(v):.4e}" for v in gref[:3]])
    print("   gamma_lbm wallsym:", [f"{float(v):.4e}" for v in g[-3:]], "... ref:", [f"{float(v):.4e}" for v in gref[-3:]])