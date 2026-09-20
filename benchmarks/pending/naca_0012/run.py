#!/usr/bin/env python
"""W3-D NACA0012 alpha=5 Re=1000 Cl grid-ladder benchmark (pre-registered).

Orchestration ONLY -- every lattice kernel (collide/stream/equilibrium/
bounce-back/channel BC/momentum-exchange force) is imported from the
TensorLBM library (repo src/, bootstrap below).  Iron rule 1:
    grep -nE "def (collide|stream|equilibrium|bounce|zou_he|far_field)" run.py
must return nothing.

Pre-registration: NOTES.md (written BEFORE this run).
Reference (locked): Kurtulus 2015 Fig. 4a digitization, Cl_ref(5 deg)=0.132+-0.015.
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))  # <repo>/src

import torch

from tensorlbm.airfoil_benchmark import build_airfoil_mask  # noqa: E402
from tensorlbm.boundaries import (  # noqa: E402
    apply_simple_channel_boundaries,
    bounce_back_cells,
    compute_obstacle_forces,
    make_channel_wall_mask,
)
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.solver import collide_bgk, collide_mrt, correct_mass, stream  # noqa: E402
from tensorlbm.utils import resolve_device  # noqa: E402

CL_REF_LOCKED = 0.132  # Kurtulus 2015 Fig4a digitized (line 0.131 / interp 0.134)
CL_REF_TEXT = 0.26  # disclosure column: paper-text-slope / Di Ilio HLBM implied
ALPHA = 5.0
RE = 1000.0
U_IN = 0.06
OUT = Path(__file__).resolve().parent  # results land next to run.py


def run_level(tag, chord, nx, ny, warmup_tc=8.0, meas_tc=8.0, seed=0):
    device = resolve_device("cuda")
    torch.manual_seed(seed)
    t0 = time.time()

    mask = build_airfoil_mask(nx, ny, chord, ALPHA, 0.0, 0.0, 0.12, device=device)
    n_solid = int(mask.sum().item())
    wall_mask = make_channel_wall_mask(ny, nx, mask, device=device)

    rho0 = torch.ones(ny, nx, device=device)
    ux0 = torch.full_like(rho0, U_IN)
    uy0 = torch.zeros_like(rho0)
    ux0[mask] = 0.0
    uy0[mask] = 0.0
    f = equilibrium(rho0, ux0, uy0, device=device)

    nu = U_IN * chord / RE
    tau = 3.0 * nu + 0.5
    dyn_pressure = 0.5 * U_IN**2 * chord
    initial_mass = float(rho0.sum().item())

    tc = chord / U_IN  # convective time [steps]
    warmup = int(warmup_tc * tc)
    n_steps = warmup + int(meas_tc * tc)

    cl_trace, cd_trace, steps_seen = [], [], []
    for step in range(1, n_steps + 1):
        if tau < 0.6:
            f = collide_mrt(f, tau=tau)
        else:
            f = collide_bgk(f, tau=tau)
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, mask)
        f = apply_simple_channel_boundaries(
            f, u_in=U_IN, wall_mask=wall_mask, obstacle_mask=torch.zeros_like(mask)
        )
        f = bounce_back_cells(f, mask)
        f = correct_mass(f, initial_mass)
        if step > warmup:
            if step % 25 == 0 or step == n_steps:
                cl_trace.append(float(fy.item()) / dyn_pressure)
                cd_trace.append(float(fx.item()) / dyn_pressure)
                steps_seen.append(step)

    arr = torch.tensor(cl_trace)
    n = len(cl_trace)
    half = n // 2
    q3 = 3 * n // 4
    mean_full = float(arr.mean())
    mean_half2 = float(arr[half:].mean())
    mean_q4 = float(arr[q3:].mean())
    amp_pp = float(arr[-max(n // 4, 2) :].max() - arr[-max(n // 4, 2) :].min())
    cd_mean = float(torch.tensor(cd_trace).mean())

    res = {
        "tag": tag,
        "chord_lu": chord,
        "nx": nx,
        "ny": ny,
        "domain_x_c": nx / chord,
        "domain_y_c": ny / chord,
        "alpha_deg": ALPHA,
        "re_chord": RE,
        "u_in": U_IN,
        "mach": U_IN * math.sqrt(3.0),
        "tau": tau,
        "nu": nu,
        "collision": "mrt" if tau < 0.6 else "bgk",
        "n_steps": n_steps,
        "warmup_steps": warmup,
        "n_trace": n,
        "n_solid_cells": n_solid,
        "cl_mean": mean_full,
        "cl_mean_lasthalf": mean_half2,
        "cl_mean_lastquarter": mean_q4,
        "cl_amp_pp_lastquarter": amp_pp,
        "cd_mean": cd_mean,
        "cl_err_vs_locked_pct": abs(mean_full - CL_REF_LOCKED) / CL_REF_LOCKED * 100,
        "cl_err_vs_text_pct": abs(mean_full - CL_REF_TEXT) / CL_REF_TEXT * 100,
        "steady": bool(
            abs(mean_half2 - mean_full) / abs(mean_full) < 0.01 and amp_pp / abs(mean_full) < 0.02
        ),
        "wall_s": time.time() - t0,
        "trace_steps": steps_seen,
        "cl_trace": cl_trace,
        "cd_trace": cd_trace,
    }
    print(
        f"[{tag}] c={chord} nx={nx} ny={ny} tau={tau:.4f} steps={n_steps} "
        f"({res['wall_s']:.0f}s) Cl={mean_full:.4f} (lasthalf {mean_half2:.4f}, "
        f"amp_pp {amp_pp:.4f}) Cd={cd_mean:.4f} steady={res['steady']}"
    )
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    levels = [
        ("c60", 60, 14, 8),
        ("c90", 90, 14, 8),
        ("c120", 120, 14, 8),
        ("c150", 150, 14, 8),
        ("c90_ny12c", 90, 14, 12),  # domain-height sensitivity (disclosure)
        ("c60_legacy_ny1p33c", 60, None, None),  # library-default domain shape probe
    ]
    all_res = {}
    for tag, chord, fx, fy in levels:
        if fx is None:
            nx, ny = 200, 80  # legacy library-default shape (3.33c x 1.33c)
        else:
            nx, ny = int(fx * chord), int(fy * chord)
        r = run_level(tag, chord, nx, ny)
        all_res[tag] = r
        with open(OUT / "result_partial.json", "w") as fh:
            json.dump(all_res, fh, indent=2)

    ladder = [all_res[t] for t in ("c60", "c90", "c120", "c150")]
    cls = [r["cl_mean"] for r in ladder]
    errs = [r["cl_err_vs_locked_pct"] for r in ladder]
    mono = all((cls[i + 1] - cls[i]) * (cls[-1] - cls[0]) >= 0 for i in range(len(cls) - 1))
    # monotonic toward limit: successive differences shrink in magnitude
    diffs = [abs(cls[i + 1] - cls[i]) for i in range(len(cls) - 1)]
    mono_shrink = all(diffs[i + 1] <= diffs[i] + 1e-12 for i in range(len(diffs) - 1))
    rich = all(r["n_solid_cells"] > 0 for r in ladder)
    verdict = {
        "cl_values": cls,
        "cl_err_vs_locked_pct": errs,
        "cl_err_vs_text_pct": [r["cl_err_vs_text_pct"] for r in ladder],
        "monotonic_direction": bool(mono),
        "monotonic_shrinking_diffs": bool(mono_shrink),
        "two_level_edge_pct": [
            d / max(cls[i], 1e-9) * 100
            for i, d in enumerate([cls[i + 1] - cls[i] for i in range(len(cls) - 1)])
        ],
        "all_pass_3pct": all(e <= 3.0 for e in errs),
        "verdict": ("PASS" if (all(e <= 3.0 for e in errs) and mono and rich) else "FAIL"),
        "reference_limited": True,
        "cl_ref_locked": CL_REF_LOCKED,
        "cl_ref_disclosure_text_implied": CL_REF_TEXT,
    }
    final = {
        "pre_registration": "NOTES.md (mtime precedes all runs)",
        "reference": verdict,
        "levels": all_res,
    }
    with open(OUT / "result.json", "w") as fh:
        json.dump(final, fh, indent=2)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
