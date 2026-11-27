"""Party A -- TensorLBM's own D2Q9 cavity solver, driven at the shared spec.

Nothing is re-implemented here: the time step is assembled from the library
primitives that the repository's verified Re=100 case uses
(``benchmarks/verified/cavity/re100/run.py``):

    collide_mrt   tensorlbm.solver        (d'Humieres MRT, shear rate 1/tau)
    stream        tensorlbm.solver        (push/pull periodic roll)
    pre-streaming half-way bounce-back    three stationary walls (V3 fix)
    zou_he_moving_lid  tensorlbm.lid_driven_cavity   (Zou & He 1997 lid)

Wall convention (declared, see cavity_spec.py): the three stationary walls sit
half-way between the wall node and the first fluid node; the lid is the node
row ``j = n-1`` (Zou-He).  Hence, in lattice spacings,

    width  W = n - 2        (wall planes at x = 0.5 and x = n-1.5)
    height H = n - 1.5      (wall planes at y = 0.5 and y = n-1)

i.e. an aspect ratio H/W = (n-1.5)/(n-2) = 1.0039 at n = 130 -- reported with
the results, because it is a real property of the verified case and is not
zero.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cavity_spec import CavitySpec, centreline_metrics, load_ghia, write_case_output

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic  # noqa: E402
from tensorlbm.lid_driven_cavity import zou_he_moving_lid  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402


def pre_bounce(f_pre: torch.Tensor, f: torch.Tensor, wall: torch.Tensor) -> torch.Tensor:
    """Pre-streaming reflection at the stationary wall nodes (V3 formulation).

    Using the *pre-collision* state delays the reflected population by exactly
    one time step, so the wall plane ends up half-way between the wall node and
    the first fluid node (second-order half-way bounce-back).
    """
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)


def run_case(spec: CavitySpec, n_nodes: int, dtype=torch.float64) -> dict:
    tau = spec.tau
    ny = nx = n_nodes

    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = True
    wall[:, 0] = True
    wall[:, -1] = True

    rho0 = torch.ones((ny, nx), dtype=dtype)
    u0 = torch.zeros((ny, nx), dtype=dtype)
    f = equilibrium(rho0, u0, u0)

    resid_mask = torch.ones((ny, nx), dtype=torch.bool)
    resid_mask[0, :] = False
    resid_mask[:, 0] = False
    resid_mask[:, -1] = False
    resid_mask[-1, :] = False          # lid row is driven, not a residual DOF

    def step(f_pre, f):
        f = collide_mrt(f, tau=tau)
        f = pre_bounce(f_pre, f, wall)
        f = stream(f)
        return zou_he_moving_lid(f, spec.u_lid)

    # ---- time loop with a steady-state residual monitor --------------------
    _, ux_prev, uy_prev = macroscopic(f)
    ux_prev = ux_prev.clone()
    uy_prev = uy_prev.clone()
    resid = float("inf")
    t0 = time.time()
    check_every = 2000
    steps_done = 0
    hist = []
    while steps_done < spec.max_steps:
        for _ in range(check_every):
            f = step(f, f)
        steps_done += check_every
        _, ux, uy = macroscopic(f)
        resid = float(
            torch.max(
                torch.abs(ux[resid_mask] - ux_prev[resid_mask]).max(),
                torch.abs(uy[resid_mask] - uy_prev[resid_mask]).max(),
            )
        ) / spec.u_lid
        hist.append([steps_done, resid])
        ux_prev, uy_prev = ux.clone(), uy.clone()
        if resid < spec.resid_tol:
            break
    elapsed = time.time() - t0

    # ---- profiles ---------------------------------------------------------
    rho, ux, uy = macroscopic(f)
    ux_np = (ux.masked_fill(wall, 0.0) / spec.u_lid).numpy()
    uy_np = (uy.masked_fill(wall, 0.0) / spec.u_lid).numpy()

    W = nx - 2.0          # declared cavity width  (lattice spacings)
    H = ny - 1.5          # declared cavity height (lattice spacings)

    j = np.arange(0, ny)
    y_all = (j - 0.5) / H                    # fluid node j -> normalised y
    i = np.arange(0, nx)
    x_all = (i - 0.5) / W

    col = nx // 2
    row = ny // 2
    y_u = y_all[1:]                          # skip the bottom wall node
    u_cl = ux_np[1:, col]
    x_v = x_all[1:-1]
    v_cl = uy_np[row, 1:-1]

    # wall values are imposed by the BC -> prepend/append them exactly
    y_u = np.concatenate(([0.0], y_u, [1.0]))
    u_cl = np.concatenate(([0.0], u_cl, [1.0]))
    x_v = np.concatenate(([0.0], x_v, [1.0]))
    v_cl = np.concatenate(([0.0], v_cl, [0.0]))

    # vortex centre from the interior speed minimum (parabolic refinement)
    speed2 = ux_np**2 + uy_np**2
    inner = np.zeros_like(speed2, dtype=bool)
    lo_y, hi_y = int(0.30 * ny), int(0.90 * ny)
    lo_x, hi_x = int(0.30 * nx), int(0.90 * nx)
    inner[lo_y:hi_y, lo_x:hi_x] = True   # primary vortex window
    s2 = np.where(inner, speed2, np.inf)
    jy, jx = np.unravel_index(np.argmin(s2), s2.shape)

    def refine2(a, b, c):
        den = a - 2 * b + c
        if not np.isfinite(den) or abs(den) < 1e-30:
            return 0.0
        return 0.5 * (a - c) / den

    cx = (jx + refine2(s2[jy, jx - 1], s2[jy, jx], s2[jy, jx + 1]) - 0.5) / W
    cy = (jy + refine2(s2[jy - 1, jx], s2[jy, jx], s2[jy + 1, jx]) - 0.5) / H

    ghia = load_ghia(int(spec.re))
    met = centreline_metrics(y_u, u_cl, x_v, v_cl, ghia, spec.u_lid,
                             label=f"tensorlbm_n{nx}")
    met["vortex_center"] = [float(cx), float(cy)]
    met["geometry"] = {"W_lattice": W, "H_lattice": H, "aspect": H / W}
    met["wall_convention"] = (
        "3 walls half-way (pre-streaming reflection), lid at node row n-1 (Zou-He)"
    )
    met["collision"] = "MRT (s_e=1.64, s_eps=1.54, s_q=1.7), shear 1/tau"

    return {
        "party": "A",
        "tool": "TensorLBM",
        "tool_version": _git_rev(),
        "spec": spec.as_dict(),
        "n_nodes": nx,
        "steps": steps_done,
        "elapsed_s": round(elapsed, 1),
        "final_resid": resid,
        "resid_history": hist[-6:],
        "metrics": met,
        "profile": {
            "y": y_u.tolist(),
            "u": u_cl.tolist(),
            "x": x_v.tolist(),
            "v": v_cl.tolist(),
        },
    }


def _git_rev() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--n", type=int, default=130, help="nodes per side (W = n-2)")
    ap.add_argument("--u-lid", type=float, default=0.06)
    ap.add_argument("--max-steps", type=int, default=400_000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    spec = CavitySpec(re=a.re, n=a.n - 2, u_lid=a.u_lid, max_steps=a.max_steps)
    res = run_case(spec, a.n)
    write_case_output(Path(a.out), res)
    m = res["metrics"]
    print(
        f"[A n={a.n} Re={a.re:g}] steps={res['steps']} resid={res['final_resid']:.1e} "
        f"u_min={m['u_min']:+.4f} (Ghia {m['ghia_u_min']:+.4f}) "
        f"v_max={m['v_max']:+.4f} (Ghia {m['ghia_v_max']:+.4f}) "
        f"vortex=({m['vortex_center'][0]:.3f},{m['vortex_center'][1]:.3f}) "
        f"rmse_u={m['rmse_u']:.4f} max_dev={m['max_abs_dev_pct']:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()