"""Calibration study: Krylov tolerance vs gradient accuracy for the fixed-point adjoint."""
import sys, time, math, json
sys.path.insert(0, "/srv/tensor/agent/work/m22003/src")
import torch

torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)

from tensorlbm.shape_gradient import (
    make_channel, run_to_steady, frozen_field_gradient, shape_gradient_through_flow,
)
from tensorlbm.fixed_point_adjoint import fixed_point_adjoint, fixed_point_residual

CFG = dict(nx=48, ny=32, diameter=6.0, tau=0.56, u_bulk=0.05, epsilon=0.40, wall_thickness=3)
H = CFG["ny"] - 2 * CFG["wall_thickness"]
Y_MID = (CFG["ny"] - 1) / 2.0


def solve(yc):
    g = make_channel(nx=CFG["nx"], ny=CFG["ny"], diameter=CFG["diameter"], center_y=yc,
                     tau=CFG["tau"], u_bulk=CFG["u_bulk"], epsilon=CFG["epsilon"],
                     wall_thickness=CFG["wall_thickness"])
    t0 = time.time()
    s = run_to_steady(g, max_steps=40000, tol=1e-10, min_steps=1000, check_every=100)
    return s, time.time() - t0


out = {}
yc0 = Y_MID + 0.05 * H
base, w = solve(yc0)
u_ref = base.geom.bulk_velocity()
scale = 0.5 * u_ref ** 2 * float(base.geom.diameter)
print(f"base: steps={base.steps} res={base.residual:.2e} fpres={fixed_point_residual(base):.2e} "
      f"CL={base.lift/scale:.6e} wall={w:.1f}s Re={base.geom.reynolds():.3f}", flush=True)

cl = {}
for h in (1.0, 0.5):
    t0 = time.time()
    p, _ = solve(yc0 + h)
    m, _ = solve(yc0 - h)
    cl[h] = ((p.lift - m.lift) / (2 * h)) / scale
    print(f"FD h={h}: dCL/dyc={cl[h]:.6e}  ({time.time()-t0:.1f}s)", flush=True)
g_ref = (cl[0.5] * 1.0 - cl[1.0] * 0.25) / (1.0 - 0.25)
print(f"FD Richardson: {g_ref:.6e}", flush=True)

for rtol in (1e-4, 1e-6, 1e-8):
    t0 = time.time()
    r = fixed_point_adjoint(base, rtol=rtol, restart=200, max_matvecs=3000, u_ref=u_ref)
    rel = abs(r["dJ_dy_c"] - g_ref) / abs(g_ref)
    print(f"fp rtol={rtol:g}: g={r['dJ_dy_c']:.6e} matvecs={r['krylov_iterations']} "
          f"relres={r['krylov_relative_residual']:.2e} info={r['krylov_info']} "
          f"rel_err_vs_FD={rel:.2e} wall={time.time()-t0:.1f}s", flush=True)
    out[f"rtol={rtol}"] = dict(g=r["dJ_dy_c"], matvecs=r["krylov_iterations"],
                               relres=r["krylov_relative_residual"], info=r["krylov_info"],
                               rel_err=rel, wall=time.time() - t0)

for n in (100, 200, 400):
    t0 = time.time()
    ad = shape_gradient_through_flow(base, n_unroll=n, window=min(50, n), u_ref=u_ref)
    print(f"AD n={n}: g={ad['dJ_dy_c']:.6e} rel_err_vs_FD={abs(ad['dJ_dy_c']-g_ref)/abs(g_ref):.2e} "
          f"({time.time()-t0:.1f}s)", flush=True)
    out[f"ad_n={n}"] = dict(g=ad["dJ_dy_c"], rel_err=abs(ad["dJ_dy_c"] - g_ref) / abs(g_ref))

fz = frozen_field_gradient(base, u_ref=u_ref)
print(f"frozen: g={fz['dJ_dy_c']:.6e} ratio_vs_FD={abs(fz['dJ_dy_c']/g_ref):.3f}", flush=True)

out["fd"] = {"h1": cl[1.0], "h05": cl[0.5], "richardson": g_ref}
out["config"] = CFG | {"yc0": yc0, "reynolds": base.geom.reynolds()}
print(json.dumps(out, indent=2))