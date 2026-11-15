import math, time, sys
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, gmres

sys.path.insert(0, "/root/tp-agent/work/m22003/src")
from tensorlbm.shape_gradient import (
    make_channel, run_to_steady, step, disk_fluid_weight, lift_from_probe,
    shape_gradient_through_flow, frozen_field_gradient,
)

cfg = dict(nx=48, ny=32, diameter=6.0, tau=0.56, u_bulk=0.05, epsilon=0.40, wall_thickness=3)
H = cfg["ny"] - 2 * cfg["wall_thickness"]

def build(yc):
    return make_channel(nx=cfg["nx"], ny=cfg["ny"], diameter=cfg["diameter"],
                        center_y=yc, tau=cfg["tau"], u_bulk=cfg["u_bulk"],
                        epsilon=cfg["epsilon"], wall_thickness=cfg["wall_thickness"])

yc0 = (cfg["ny"] - 1) / 2.0 + 0.05 * H
print("dtype", torch.get_default_dtype(), "H", H, "yc0", yc0)

def solve(yc, max_steps=20000):
    g = build(yc)
    t0 = time.time()
    s = run_to_steady(g, max_steps=max_steps, tol=1e-10, min_steps=500, check_every=100)
    return s, time.time() - t0

s, wall = solve(yc0)
print(f"steady: lift={s.lift:.6e} drag={s.drag:.6e} steps={s.steps} res={s.residual:.2e} wall={wall:.1f}s")
geom = s.geom
u_ref = geom.bulk_velocity()
scale = 0.5 * u_ref * u_ref * float(geom.diameter)
print("Re", geom.reynolds())

def cl(yc, max_steps=20000):
    ss, _ = solve(yc, max_steps)
    return ss.lift / scale

for h in (1.0, 0.5):
    t0 = time.time()
    d = (cl(yc0 + h) - cl(yc0 - h)) / (2 * h)
    print(f"FD h={h}: dCL/dyc = {d:.6e}  ({time.time()-t0:.0f}s)")

for nu_ in (25, 50, 100, 200):
    r = shape_gradient_through_flow(s, n_unroll=nu_, window=min(25, nu_))
    print(f"AD unroll={nu_}: {r['dJ_dy_c']:.6e}")

fr = frozen_field_gradient(s)
print("frozen:", fr["dJ_dy_c"])

cy0_t = torch.tensor(float(geom.center_y), dtype=geom.dtype)
fstar = s.f.detach().clone()

def S(f, cy):
    return step(f, geom, center_y=cy)[0]

def Jf(f, cy):
    _, probe = step(f, geom, center_y=cy)
    w = disk_fluid_weight(geom, center_y=cy)
    sw = torch.where(geom.wall_mask(), torch.zeros_like(w), 1.0 - w)
    return lift_from_probe(probe, sw)[1] / scale

with torch.no_grad():
    res = (S(fstar, cy0_t) - fstar).abs().max()
    print("fixed-point residual max|R|", float(res), " CL=", float(Jf(fstar, cy0_t)))

f_ = fstar.clone().requires_grad_(True)
J = Jf(f_, cy0_t)
Jf_grad, = torch.autograd.grad(J, f_)
print("|J_f|", float(Jf_grad.norm()))

calls = [0]
def matvec(v):
    v = torch.as_tensor(np.asarray(v).reshape(fstar.shape), dtype=geom.dtype)
    ff = fstar.clone().requires_grad_(True)
    Sv = S(ff, cy0_t)
    g, = torch.autograd.grad(Sv, ff, grad_outputs=v)
    calls[0] += 1
    return (v - g).reshape(-1).numpy()

n = int(np.prod(fstar.shape))
A = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
t0 = time.time()
lam, info = gmres(A, Jf_grad.reshape(-1).numpy(), rtol=1e-10, atol=0.0, restart=50, maxiter=500)
print(f"gmres info={info} matvecs={calls[0]} time={time.time()-t0:.1f}s, |lambda|={np.linalg.norm(lam):.3e}")
r = A.matvec(lam) - Jf_grad.reshape(-1).numpy()
print("gmres rel residual", np.linalg.norm(r) / np.linalg.norm(Jf_grad.reshape(-1).numpy()))

lam_t = torch.as_tensor(lam.reshape(fstar.shape), dtype=geom.dtype)

p = cy0_t.clone().requires_grad_(True)
Jp = torch.autograd.grad(Jf(fstar, p), p)[0]
p2 = cy0_t.clone().requires_grad_(True)
Sp = torch.autograd.grad(S(fstar, p2), p2, grad_outputs=lam_t)[0]
print("J_p", float(Jp), "lam^T S_p", float(Sp))
print("total (J_p + lam^T S_p)", float(Jp + Sp))