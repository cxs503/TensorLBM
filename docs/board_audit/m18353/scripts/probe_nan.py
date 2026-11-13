"""Audit probe: localise the NaN onset of the deposited ladder runner."""
import sys, math, torch, json
sys.path.insert(0, '/root/tp-agent/work/m18236/src')
import importlib.util
from tensorlbm import cac_lbm2d as cac
from tensorlbm import pc_ladder as P
torch.set_num_threads(1)

spec = importlib.util.spec_from_file_location("rx", "/root/tp-agent/work/m18236/TensorLBM/xval/XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT/run_xval.py")
rx = importlib.util.module_from_spec(spec)
src = open(spec.origin).read().split("def main(")[0]
exec(compile(src, spec.origin, "exec"), rx.__dict__)

def fin(sim):
    return bool(torch.isfinite(sim.f).all().item()), bool(torch.isfinite(sim.g).all().item())

out = {}
# A: pure evolution at g=0 for 800 steps (is the host solver / IC stable?)
sim = rx.build()
for n in (200, 400, 800):
    cac.run(sim, n - sim.step_count, sample_every=0)
    u, umax = P._u_rate(sim)
    out[f"A_g0_step{n}"] = dict(step=sim.step_count, finite=fin(sim), u_mean=u, u_max=umax)

# B: single tiny body force from rest, 200 steps (rung-1 situation, isolated)
sim = rx.build()
cac.run(sim, 200, sample_every=0)
out["B_before_force"] = dict(step=sim.step_count, finite=fin(sim), u_max=P._u_rate(sim)[1])
sim.gravity = (1.929012345679013e-05, 0.0)
for n in (10, 50, 200):
    cac.run(sim, n, sample_every=0)
    out[f"B_after_force_{n}"] = dict(step=sim.step_count, finite=fin(sim),
                                     u_max=P._u_rate(sim)[1],
                                     phi_min=float(sim.f.sum(0).min()), phi_max=float(sim.f.sum(0).max()))
print(json.dumps(out, indent=1, default=str))
