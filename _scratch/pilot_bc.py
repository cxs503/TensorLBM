"""Pilot: ghost-value wetting BC vs gradient-rotation (Akai/MF-LBM faithful)
wetting BC on a 2-D sessile drop — same case, same integration, short run."""
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
from tensorlbm.cg_wetting2d import OPP, geometrical_wetting_ghost, phase_field
from tensorlbm.color_gradient2d import color_gradient_two_tau_step
from tensorlbm.solver import stream
from tensorlbm.wetting.metrics import spherical_cap_inversion


def build_solid(ny, nx):
    s = torch.zeros(ny, nx, dtype=torch.bool)
    s[0, :] = True
    s[-1, :] = True
    return s


def init_drop(ny, nx, theta_deg, R, xi, dtype):
    th = math.radians(theta_deg)
    x0 = (nx - 1) / 2.0
    yc = 0.5 - R * math.cos(th)
    y = torch.arange(ny, dtype=dtype).view(-1, 1)
    x = torch.arange(nx, dtype=dtype).view(1, -1)
    r = torch.sqrt((x - x0) ** 2 + (y - yc) ** 2)
    phi = torch.tanh((R - r) / xi)
    rho_r = 0.5 * (1.0 + phi)
    rho_b = 0.5 * (1.0 - phi)
    w = torch.tensor([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
                     dtype=dtype).view(9, 1, 1)
    return w * rho_r.unsqueeze(0), w * rho_b.unsqueeze(0)


def blobs(phi, level=0.0, min_cells=20):
    """Count connected liquid clusters (4-neighbour flood fill on a boolean mask)."""
    m = (phi > level)
    m[0, :] = False
    m[-1, :] = False
    ny, nx = m.shape
    seen = np.zeros_like(m)
    sizes = []
    for i in range(ny):
        for j in range(nx):
            if m[i, j] and not seen[i, j]:
                stack = [(i, j)]
                seen[i, j] = True
                n = 0
                while stack:
                    a, b = stack.pop()
                    n += 1
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        c, d = a + da, (b + db) % nx
                        if 0 <= c < ny and m[c, d] and not seen[c, d]:
                            seen[c, d] = True
                            stack.append((c, d))
                sizes.append(n)
    return sorted([s for s in sizes if s >= min_cells], reverse=True)


def run(mode, theta, ny=64, nx=96, R=14.0, xi=2.0, steps=3000, tau=1.0, A=0.04, beta=0.70,
        every=500, verbose=True):
    torch.set_num_threads(4)
    dtype = torch.float64
    solid = build_solid(ny, nx)
    solid3 = solid.unsqueeze(0)
    f_r, f_b = init_drop(ny, nx, theta, R, xi, dtype)
    theta_map = {(1, 0): 90.0}
    t0 = time.time()
    out = []
    for it in range(1, steps + 1):
        phi = phase_field(f_r.sum(0), f_b.sum(0))
        kw = {}
        if mode == "ghost":
            kw["phi_solid"] = geometrical_wetting_ghost(
                phi, solid, theta, periodic=(True, False), theta_by_direction=theta_map)
        else:
            kw["wall_contact_angle"] = theta
            kw["wall_contact_angle_by_direction"] = theta_map
            kw["wetting_periodic"] = (True, False)
            kw["contact_angle_convention"] = "physical"
        f_r, f_b = color_gradient_two_tau_step(
            f_r, f_b, tau_r=tau, tau_b=tau, A=A, beta=beta, solid_mask=solid, **kw)
        f_r = stream(f_r)
        f_b = stream(f_b)
        f_r = torch.where(solid3, f_r[list(OPP)], f_r)
        f_b = torch.where(solid3, f_b[list(OPP)], f_b)
        if it % every == 0 or it == 1:
            phi = phase_field(f_r.sum(0), f_b.sum(0)).numpy()
            jx = ((f_r + f_b)[1] - (f_r + f_b)[3]).sum(0).numpy()
            jy = ((f_r + f_b)[2] - (f_r + f_b)[4]).sum(0).numpy()
            rho = (f_r + f_b).sum(0).numpy()
            u = np.hypot(jx / rho, jy / rho)
            u[np.isnan(u)] = 0.0
            try:
                cap = spherical_cap_inversion(phi, 0.5, level=0.0)["theta_deg"]
            except Exception:
                cap = float("nan")
            bl = blobs(phi)
            mass_r = float(f_r.sum())
            out.append(dict(step=it, theta_cap=cap, umax=float(u[1:-1].max()), blobs=bl, mass=mass_r))
            if verbose:
                print(f"  [{mode} th={theta}] {it:5d} cap={cap:7.2f} |u|max={u[1:-1].max():.2e} "
                      f"clusters={bl}", flush=True)
    print(f"  -> {mode} th={theta}: wall {time.time()-t0:.1f}s")
    return out, phase_field(f_r.sum(0), f_b.sum(0)).numpy()


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    for mode in ("rotation", "ghost"):
        for th in (60.0, 30.0):
            hist, phi = run(mode, th, steps=steps)
            np.save(f"_scratch/pilot_{mode}_{int(th)}.npy", phi)