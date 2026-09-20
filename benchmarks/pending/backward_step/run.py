#!/usr/bin/env python
"""W2-C driver: backward-facing step Re=100 (Armaly/Erturk convention).

Common entry: tensorlbm.backward_facing_step.run_backward_facing_step from the
read-only worktree /nfs/wangxi/worktrees/bm_w2 (@c0b84d96d1).  Only boundary
conditions and the measurement function are monkey-patched at module level;
no worktree file is modified.

Convention (pre-registered in NOTES.md):
  Re = U_mean * 2 * h_i / nu = 100   (Armaly 1983 / Erturk 2008, verbatim)
  U_mean = discrete mean of the imposed inlet profile, h_i = ny-1-step_h.
  ER = (ny-2)/(ny-1-step_h) = 1.9423 (Armaly geometry).
  X_r/s: subcell zero crossing of ux at first fluid row y=1, origin
  x = x_step-0.5 (halfway bounce-back step face), normalised by step_h.

The module's config.re is only a carrier for nu = u_in*step_h/re.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

import numpy as np
import torch

import tensorlbm.backward_facing_step as bfs
from tensorlbm.boundaries import zou_he_inlet_velocity, zou_he_outlet_pressure

ENV = os.environ
NY = int(ENV["BFS_NY"])
SH = int(ENV["BFS_STEP_H"])
XS = int(ENV["BFS_X_STEP"])
NX = int(ENV["BFS_NX"])
UMAX = float(ENV.get("BFS_U_MAX", "0.05"))
N_STEPS = int(ENV["BFS_N_STEPS"])
OUT_INT = int(ENV.get("BFS_OUT_INTERVAL", "10000"))
TAG = ENV["BFS_TAG"]
INLET = ENV.get("BFS_INLET", "parabolic")  # parabolic | uniform
OUTLET = ENV.get("BFS_OUTLET", "pressure")  # pressure | copy
OUT_ROOT = Path(ENV.get("BFS_OUT_ROOT", "/nfs/wangxi/runs/bm_widen_20260920/backward_step"))
RE_PHYS = float(ENV.get("BFS_RE_PHYS", "100.0"))

H_I = NY - 1 - SH  # inlet fluid rows
ER = (NY - 2) / H_I


def make_profile() -> torch.Tensor:
    y = torch.arange(NY, dtype=torch.float64)
    if INLET == "parabolic":
        # halfway-bounce-back walls: physical channel y in [SH-0.5, NY-1.5]
        yp = y - (SH - 0.5)
        prof = 4.0 * UMAX * yp * (H_I - yp) / (H_I * H_I)
        prof = torch.clamp(prof, min=0.0)
    elif INLET == "uniform":
        prof = torch.full((NY,), UMAX, dtype=torch.float64)
    else:
        raise ValueError(INLET)
    return prof


PROF = make_profile()
U_MEAN = float(PROF[SH : NY - 1].mean().item())  # discrete mean over fluid rows
NU = U_MEAN * 2.0 * H_I / RE_PHYS  # exact Re for the discrete profile
TAU = 3.0 * NU + 0.5
CFG_RE = UMAX * SH / NU  # carrier: nu = u_in*step_h/re

DEVICE = "cuda"
PROF_C = PROF.to(device=DEVICE, dtype=torch.float32)

_orig_snap = bfs._save_bfs_snapshot


def _patched_inlet(f: torch.Tensor, u_in: float, step_h: int) -> torch.Tensor:
    return zou_he_inlet_velocity(f, PROF_C, 0.0)


def _patched_outlet(f: torch.Tensor) -> torch.Tensor:
    if OUTLET == "pressure":
        return zou_he_outlet_pressure(f, 1.0)
    f2 = f.clone()
    f2[:, :, -1] = f[:, :, -2]
    return f2


def _subcell_zero_crossing(vals: np.ndarray, x_start: int) -> float | None:
    """First negative->positive crossing of vals (indexed from x_start), subcell."""
    pos = vals > 0.0
    for i in range(1, vals.shape[0]):
        if pos[i] and not pos[i - 1]:
            if vals[i - 1] < 0.0:
                frac = -vals[i - 1] / (vals[i] - vals[i - 1])
            else:  # exact zero at node i-1
                frac = 0.0
            return x_start + (i - 1) + float(frac)
    return None


def _measure_row(ux2d: np.ndarray, row: int) -> float | None:
    vals = ux2d[row, XS:]
    xc = _subcell_zero_crossing(vals, XS)
    if xc is None:
        return None
    return (xc - (XS - 0.5)) / SH


def _patched_measure(ux: torch.Tensor, x_step: int, step_h: int) -> float:
    row = ux[1, x_step:].detach().cpu().numpy()
    xc = _subcell_zero_crossing(row, x_step)
    if xc is None:
        return 0.0
    return (xc - (x_step - 0.5)) / step_h


def _patched_snap(run_dir, step, ux, solid, config) -> None:
    _orig_snap(run_dir, step, ux, solid, config)
    if step % (5 * config.output_interval) == 0 or step == config.n_steps:
        np.savez_compressed(
            Path(run_dir) / f"ux_{step:08d}.npz",
            ux=ux.detach().cpu().numpy().astype(np.float32),
            solid=solid.detach().cpu().numpy(),
        )


def main() -> None:
    print(
        f"[w2c] TAG={TAG} ny={NY} sh={SH} h_i={H_I} ER={ER:.4f} xs={XS} nx={NX}\n"
        f"[w2c] inlet={INLET} outlet={OUTLET} UMAX={UMAX} U_mean={U_MEAN:.6f}\n"
        f"[w2c] nu={NU:.6f} tau={TAU:.4f} Re_phys={U_MEAN * 2 * H_I / NU:.4f} cfg_re={CFG_RE:.4f}\n"
        f"[w2c] n_steps={N_STEPS} out_int={OUT_INT}",
        flush=True,
    )

    bfs._apply_bfs_inlet = _patched_inlet
    bfs._apply_bfs_outlet = _patched_outlet
    bfs.measure_reattachment_length = _patched_measure
    bfs._save_bfs_snapshot = _patched_snap

    cfg = bfs.BackwardFacingStepConfig(
        nx=NX,
        ny=NY,
        step_h=SH,
        x_step=XS,
        u_in=UMAX,
        re=CFG_RE,
        n_steps=N_STEPS,
        output_interval=OUT_INT,
        output_root=OUT_ROOT,
        run_name=TAG,
        device=DEVICE,
        overwrite=True,
    )
    run_dir = Path(bfs.run_backward_facing_step(cfg))

    # ---- post-processing -------------------------------------------------
    with (run_dir / "reattachment.csv").open() as fh:
        series = [(int(r[0]), float(r[1]), float(r[2])) for r in list(csv.reader(fh))[1:]]
    steps = [s for s, _, _ in series]
    xrs = [x for _, x, _ in series]
    last3 = xrs[-3:]
    last2 = xrs[-2:]
    drift_last3 = max(last3) - min(last3)
    drift_last2 = abs(last2[1] - last2[0])
    # per-10k drift over the final window
    per10k = [abs(xrs[i] - xrs[i - 1]) for i in range(1, len(xrs))][-3:]

    final_npz = run_dir / f"ux_{N_STEPS:08d}.npz"
    xr_row2 = None
    xr_row1_npz = None
    if final_npz.exists():
        ux = np.load(final_npz)["ux"]
        xr_row1_npz = _measure_row(ux, 1)
        xr_row2 = _measure_row(ux, 2)

    meta = json.loads((run_dir / "run_metadata.json").read_text())
    diag = meta["diagnostics"]
    extra = {
        "tag": TAG,
        "convention": {
            "re_definition": "U_mean*2*h_i/nu (Armaly 1983 / Erturk 2008)",
            "re_phys": U_MEAN * 2 * H_I / NU,
            "er": ER,
            "h_i": H_I,
            "step_h": SH,
            "nu": NU,
            "tau": TAU,
            "collision": "bgk (tau>=0.60)",
            "inlet": INLET,
            "outlet": OUTLET,
            "u_max": UMAX,
            "u_mean_discrete": U_MEAN,
            "upstream_cells": XS - 0.5,
            "downstream_cells": NX - XS,
            "origin": "x = x_step-0.5, normalised by step_h",
        },
        "xr_row1_final_series": xrs[-5:],
        "xr_row1_final": xrs[-1],
        "xr_row1_final_npz": xr_row1_npz,
        "xr_row2_final": xr_row2,
        "row1_row2_gap": (None if xr_row2 is None else xr_row2 - xrs[-1]),
        "drift_last3_range": drift_last3,
        "drift_last2_change": drift_last2,
        "per_output_interval_change_last3": per10k,
        "steady": bool(drift_last3 <= 0.02 and drift_last2 <= 0.01),
        "mass_drift_final": diag[-1]["mass_drift"],
        "max_speed_final": diag[-1]["max_speed"],
        "mean_rho_final": diag[-1]["mean_rho"],
        "series_steps": steps,
        "series_xr": xrs,
    }
    (run_dir / "result_extra.json").write_text(json.dumps(extra, indent=2))
    print(
        f"[w2c] DONE {TAG}: xr_row1={xrs[-1]:.5f} xr_row2={xr_row2} "
        f"drift3={drift_last3:.5f} steady={extra['steady']} "
        f"mass_drift={diag[-1]['mass_drift']:.4f} max|u|={diag[-1]['max_speed']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
