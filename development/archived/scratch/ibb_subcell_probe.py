#!/usr/bin/env python3
"""Sub-cell wall-position probe for the 2-point interpolated bounce-back operator.

Straight 2D channel; the two no-slip planes are placed at *fractional* distances
q0, q1 from the outermost fluid rows via the sub-grid SDF.  The channel is
therefore exactly a plane Poiseuille channel of half-width

    H = (N - 1 + q0 + q1) / 2,      y_c = (-q0 + (N-1) + q1) / 2

with u(y) = g (H^2 - (y - y_c)^2) / (2 nu) -- no staircase anywhere, so the
reference carries no geometric ambiguity.
"""
import json
import sys
from pathlib import Path

import torch
torch.set_num_threads(1)

sys.path.insert(0, '/srv/tensor/agent/work/m22892')

from tensorlbm.boundary.interpolated_bounce_back import (  # noqa: E402
    interpolated_bounce_back_2d,
    alpha_bfl,
)
from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import C as C9, W as W9, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402


def two_branch(q):
    """Exact Bouzidi-Firdaouss-Lallemand 2-point two-branch alpha."""
    return torch.where(q < 0.5, torch.full_like(q, 2.0), 1.0 / q.clamp_min(1e-6))


def force_source(fx):
    out = torch.zeros((9,) + fx.shape)
    for i in range(9):
        out[i] = 3.0 * W9[i] * (C9[i, 0] * fx)
    return out


def setup(nx, n, q0, q1):
    js = torch.arange(-1, n + 1).float().view(n + 2, 1).expand(n + 2, nx)
    sdf_full = torch.minimum(js + q0, (n - 1 + q1) - js)
    solid_full = torch.zeros((n + 2, nx), dtype=torch.bool)
    solid_full[0] = True
    solid_full[-1] = True
    return sdf_full, solid_full


def run(nx, n, q0, q1, tau, g, wall, n_steps, alpha=None, init="equilibrium"):
    nu = (tau - 0.5) / 3.0
    sdf, solid = setup(nx, n, q0, q1)
    src = force_source(torch.full((n + 2, nx), g))
    rho0 = torch.ones((n + 2, nx))
    n_ = n
    yv = torch.arange(n_).float().view(n_, 1).expand(n_, nx)
    H_ = (n_ - 1 + q0 + q1) / 2.0
    yc_ = (-q0 + (n_ - 1) + q1) / 2.0
    u0 = g * (H_ ** 2 - (yv - yc_) ** 2) / (2 * nu)
    uz = torch.zeros((n_ + 2, nx))
    uz[1:n_ + 1] = u0
    f = equilibrium(rho0, uz, torch.zeros_like(uz))
    if init == "zero":
        f[:, solid] = 0.0
    elif init == "slot_eq":
        eq = equilibrium(torch.ones_like(rho0), torch.full_like(rho0, 0.01),
                         torch.zeros_like(rho0))
        f[:, solid] = eq[:, solid]

    def step(f):
        f_pre = collide_bgk(f, tau) + src
        f_st = stream(f_pre)
        if wall == "halfway":
            return bounce_back_cells(f_st, solid)
        return interpolated_bounce_back_2d(f_st, f_pre, sdf, solid, alpha=alpha)

    u_acc = torch.zeros((n, nx))
    rho_acc = torch.zeros((n, nx))
    n_acc = 0
    for it in range(n_steps):
        if False: pass
        f = step(f)
        if it >= n_steps - 200:
            rho, ux, uy = macroscopic(f)
            u_acc += ux[1:n + 1]
            rho_acc += rho[1:n + 1]
            n_acc += 1
    u = u_acc / n_acc
    rho = rho_acc / n_acc

    y = torch.arange(n).float()
    H = (n - 1 + q0 + q1) / 2.0
    yc = (-q0 + (n - 1) + q1) / 2.0
    u_ana = g * (H ** 2 - (y - yc) ** 2) / (2 * nu)
    umax = float(u_ana.max())
    mask = u_ana > 0.2 * umax
    um = u.mean(dim=1)
    err = (um[mask] - u_ana[mask]).abs() / umax
    Q = float(u.mean().item())
    Q_ana = float(u_ana.mean().item())
    return {
        "wall": wall, "init": init, "n": n, "q0": q0, "q1": q1,
        "rho_mean": float(rho.mean()),
        "u_max_err_pct": 100 * float((u[:, nx // 2].max() - umax) / umax),
        "theta_pct": 100 * float(err.max()),
        "Q_rel_pct": 100 * (Q - Q_ana) / Q_ana,
    }


if __name__ == "__main__":
    rows = []
    tau, g, nx, steps = 0.8, 1e-5, 2, 900
    for n in (8, 16):
        for q in (0.25, 0.5, 0.75):
            arms = [("halfway", None, "halfway"), ("ibb", None, "ibb_default"),
                    ("ibb", two_branch, "ibb_two_branch")]
            for wall, al, tag in arms:
                import traceback
                try:
                    r = run(nx, n, q, q, tau, g, wall, steps, alpha=al)
                except Exception:
                    print('FAIL', n, q, tag); traceback.print_exc(); raise
                r["name"] = tag
                rows.append(r)
    print(f"{'n':>3} {'q':>5} {'scheme':<16} {'theta%':>9} {'Q%':>9} {'rho':>10} {'umax%':>8}")
    for r in rows:
        print(f"{r['n']:>3} {r['q0']:>5} {r['name']:<16} {r['theta_pct']:>9.3f} "
              f"{r['Q_rel_pct']:>9.3f} {r['rho_mean']:>10.6f} {r['u_max_err_pct']:>8.3f}")
    Path('_scratch/ibb_subcell.json').write_text(json.dumps(rows, indent=1))