"""Probe: moving sub-cell (partially saturated) wall -- the untested NT branch.

Couette variant where the *moving* wall is the partial row:
row 0 solid, row 1 partial (eps_s) moving at U, top row stationary.
Analytic: u(y) = U (y_t - y) / H,  y_b = 0.5 + eps_s, y_t = ny - 1.5.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_per_step_hook_mask_blend"))
sys.path.insert(0, str(ROOT / "src"))

import run_xval as R  # noqa: E402
import tensorlbm.step_hooks as SH  # noqa: E402

torch.set_num_threads(4)
DT = torch.float64


def run_moving_partial(ny, nx, tau, eps_s, case, u_wall=0.02, steps=None):
    nu = (tau - 0.5) / 3.0
    y_b = 0.5 + float(eps_s)
    y_t = ny - 1.5
    H = y_t - y_b
    if steps is None:
        steps = R.steps_for_couette(H, nu)
    solid = torch.zeros(ny, nx, dtype=DT); solid[0, :] = 1.0
    part = torch.zeros(ny, nx, dtype=DT); part[1, :] = float(eps_s)
    top = torch.zeros(ny, nx, dtype=DT); top[-1, :] = 1.0
    f0 = R.rest_field(ny, nx)
    tables = SH.tables_for(f0)

    partial = SH.partial_cell_layer("partial_row1", part, tau, (u_wall, 0.0), reflect="post")
    layers = [SH.stationary_bounce_layer("solid_row0", solid), partial,
              SH.stationary_bounce_layer("top_wall", top)]
    op = SH.masked_step(tau=tau, layers=layers, tables=tables)
    with torch.no_grad():
        state, _ = SH.solve_with_hooks({"f": f0.clone()}, steps, operator=op)
        f = state["f"]
    rho, ux, _ = R.macroscopic(f)
    u = ux.detach().numpy()
    # fluid rows only: 2 .. ny-2
    ys = np.arange(2, ny - 1, dtype=float)
    uu = np.array([u[int(y), 0] for y in range(2, ny - 1)])
    ref = u_wall * (y_t - ys) / H
    l2 = 100.0 * np.linalg.norm(uu - ref) / np.linalg.norm(ref)
    # least squares interface from the *analytic shape* u = U (y_t - y)/H
    # slope = -U/H  =>  d_eff from a linear fit u = a + b y with wall at u=U: y_b = (a-U)/b
    A = np.polyfit(ys, uu, 1)
    y_meas = (A[1] - u_wall) / A[0]
    return dict(l2=l2, offset=y_meas - y_b, max_err=100.0 * float(np.max(np.abs(uu - ref)) / u_wall),
                finite=bool(np.isfinite(f.numpy()).all()))


def main():
    print("case       ny tau  eps   L2%     offset_dx  maxerr%")
    for case in ("couette",):
        for ny in (16, 24):
            for tau in (0.8,):
                for eps in (0.25, 0.5, 0.75, 1.0):
                    r = run_moving_partial(ny, 8, tau, eps, case)
                    print(f"{case} {ny:3d} {tau} {eps:4.2f} {r['l2']:7.4f} {r['offset']:+9.4f} {r['max_err']:8.4f}")


if __name__ == "__main__":
    main()