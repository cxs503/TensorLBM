"""Scratch probe 2: coexistence ratio and stability of 3D SC MCMP (gentle init)."""
import math, sys, time
import torch
sys.path.insert(0, "src")
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.multiphase3d import collide_sc_two_component_3d
from tensorlbm.solver3d import stream3d_cat


def profile(nz):
    z = torch.arange(nz, dtype=torch.float32)
    z1, z2, iw = nz / 4.0, 3.0 * nz / 4.0, 2.0
    p = 0.5 * (1.0 + torch.tanh((z - z1) / iw)) - 0.5 * (1.0 + torch.tanh((z - z2) / iw))
    return p.view(nz, 1, 1)


def run(G12, G11=0.0, G22=0.0, tau=1.0, steps=1500, nz=32, ny=12, nx=12,
        delta=0.05, guo=True, every=100, label=""):
    p = profile(nz).expand(nz, ny, nx).contiguous()
    r1 = 0.6 * (1 + delta) * p + 0.6 * (1 - delta) * (1 - p)
    r2 = 0.6 * (1 - delta) * p + 0.6 * (1 + delta) * (1 - p)
    z0 = torch.zeros_like(r1)
    f1, f2 = equilibrium3d(r1, z0, z0, z0), equilibrium3d(r2, z0, z0, z0)
    t0 = time.time(); umax = 0.0
    for s in range(1, steps + 1):
        f1 = stream3d_cat(f1); f2 = stream3d_cat(f2)
        f1, f2 = collide_sc_two_component_3d(f1, f2, G_12=G12, tau1=tau, tau2=tau,
                                             use_guo=guo, G_self=(G11, G22))
        if s % every == 0 or s == 1:
            rr, ux, uy, uz = macroscopic3d(f1)
            u = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max())
            umax = max(umax, u)
            if not math.isfinite(u) or float(rr.min()) <= 0:
                return dict(label=label, G12=G12, G11=G11, step=s, diverged=True,
                            umax=u, dt=round((time.time() - t0) / s, 4))
    r1f, ux, uy, uz = macroscopic3d(f1)
    r2f, *_ = macroscopic3d(f2)
    liq = slice(nz // 2 - nz // 8, nz // 2 + nz // 8)
    gas = slice(nz // 32, nz // 32 + nz // 8)
    out = dict(label=label, G12=G12, G11=G11, G22=G22, steps=steps, diverged=False,
               umax=round(umax, 5), dt=round((time.time() - t0) / steps, 4),
               r1_l=round(float(r1f[liq].mean()), 5), r1_g=round(float(r1f[gas].mean()), 6),
               r2_l=round(float(r2f[liq].mean()), 6), r2_g=round(float(r2f[gas].mean()), 5))
    tot_l = out["r1_l"] + out["r2_l"]; tot_g = out["r1_g"] + out["r2_g"]
    out["ratio_tot"] = round(tot_l / max(tot_g, 1e-9), 2)
    out["ratio_c1"] = round(out["r1_l"] / max(out["r1_g"], 1e-9), 2)
    return out


if __name__ == "__main__":
    torch.set_num_threads(8)
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "cross"
    if mode == "cross":
        for G in [-1.6, -2.0, -2.4]:
            print(run(G, steps=1500), flush=True)
    else:  # self coupling only component 1
        for G11 in [-2.0, -3.0, -3.6]:
            print(run(-1.0, G11=G11, steps=1500, label="self"), flush=True)