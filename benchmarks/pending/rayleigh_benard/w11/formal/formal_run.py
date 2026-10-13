#!/usr/bin/env python
"""W11-C rayleigh_benard FORMAL tier runner (prereg §5 + E1/E2/E3 errata).

- Step count: prereg §3 EXACT per E1: N = ceil(3.0*spt/|gamma_ref|) clamped to
  [4000, 6*spt]; arm 1.0 (gamma_ref=0, report-only) takes the cap 6*spt.
- fp64, amp=1e-4, eigen seed rows: npz row semantics per E3a (row 0 = lower
  ghost mirror, rows 1..d = fluid z_j=(j-0.5)/d, last row = upper ghost);
  wall rows seeded with the ghost-mirror values (overwritten by BCs anyway).
- correct_mass every step; signed A_T projection; 3-window fit (prereg §3).
- Health per E2/E3b: e_folds = |ln(a_last/a_first)|; mass drift reported
  absolute+relative (fluid-cell denominator noted in header).
- Machine-in/machine-out: reads ref_table.json + eig_ny{ny}.npz only;
  writes formal/results/f_ny{ny}_arm{frac}.json.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path("/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard")
sys.path.insert(0, str(BASE / "ref"))
sys.path.insert(0, str(BASE / "probe"))
sys.path.insert(0, "/nfs/wangxi/worktrees/bm_w11/src")

import rb_lbm as R

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import correct_mass, stream
from tensorlbm.thermal import (
    collide_bgk_force,
    pre_streaming_bounce_back,
    temperature_collision,
    temperature_equilibrium,
    temperature_stream,
)

REF = json.loads((BASE / "ref" / "ref_table.json").read_text())
WINDOWS = ((0.15, 0.90), (0.05, 0.75), (0.30, 1.00))
AMP = 1e-4
GATE_ARMS = (0.85, 0.95, 1.05, 1.15, 1.3)
ALL_ARMS = (0.85, 0.95, 1.0, 1.05, 1.15, 1.3)
OUTDIR = BASE / "formal" / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)


def steps_for(spt, gref):
    if gref == 0.0:
        return int(6 * spt)
    n = math.ceil(3.0 * spt / abs(gref))
    return int(min(max(n, 4000), 6 * spt))


def run_arm(ny, frac):
    lev = next(x for x in REF["levels"] if x["ny"] == ny)
    arm = next(x for x in lev["arms"] if abs(x["frac"] - frac) < 1e-9)
    nx, d, k, spt = lev["nx"], lev["d"], lev["k"], lev["steps_per_tstar"]
    ra, gref = arm["Ra"], arm["gamma_ref"]
    gb = ra * R.NU * R.ALPHA / d**3
    nsteps = steps_for(spt, gref)
    every = max(1, nsteps // 600)
    npz = dict(np.load(BASE / "ref" / f"eig_ny{ny}.npz"))
    tag = f"ny{ny}/arm{frac}"
    shape = {kk: npz[f"{tag}/{kk}"].tolist() for kk in ("Theta_node", "Phi_node", "Phider_node")}

    device = torch.device("cuda")
    torch.set_default_dtype(torch.float64)
    Tbar = R.base_state(ny, nx, device, "half").double()
    th_s, ux_s, uy_s = R.seed_fields(ny, nx, device, AMP, "eigen", shape, "half")
    f = equilibrium(torch.ones((ny, nx), device=device), ux_s, uy_s)
    g = temperature_equilibrium(Tbar + th_s, ux_s, uy_s)
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    m0 = float(f.sum())

    t0 = time.time()
    steps_rec, a_rec = [], []
    abort = None
    for step in range(nsteps + 1):
        if step % every == 0:
            rho, ux, uy = macroscopic(f)
            T = g.sum(dim=0)
            aT, _, _ = R.project_amplitudes(T, ux, uy, Tbar, shape, k, device)
            steps_rec.append(step)
            a_rec.append(float(aT))
            if not math.isfinite(float(aT)):
                abort = f"non-finite A_T at step {step}"
                break
        if step == nsteps:
            break
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)
        F = R.buoyancy_theta(T, gb, Tbar)
        g = temperature_collision(g, R.TAU_T, ux, uy)
        g = temperature_stream(g)
        g = R.apply_rb_temp_bc(g, "half")
        f_pre = f
        f = collide_bgk_force(f, R.TAU, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)
        f = correct_mass(f, m0)
    wall_s = time.time() - t0

    st = np.array(steps_rec, dtype=float)
    a = np.array(a_rec, dtype=float)
    slopes, gammas = [], []
    n = len(st)
    for lo, hi in WINDOWS:
        i0, i1 = int(lo * n), max(int(hi * n), int(lo * n) + 5)
        sl = float(np.polyfit(st[i0:i1], np.log(np.abs(a[i0:i1])), 1)[0])
        slopes.append(sl)
        gammas.append(sl * d * d / R.ALPHA)
    spread = (max(slopes) - min(slopes)) / abs(np.mean(slopes)) if np.mean(slopes) else 0.0
    gamma_sim = float(np.mean(gammas))
    rho, ux, uy = macroscopic(f)
    Tfin = g.sum(dim=0)
    a_first, a_last = float(a[0]), float(a[-1])
    e_folds = abs(math.log(abs(a_last / a_first))) if a_first and a_last else float("inf")
    dm = float(f.sum()) - m0
    out = {
        "ny": ny,
        "nx": nx,
        "frac": frac,
        "Ra": ra,
        "g_beta": gb,
        "amp": AMP,
        "dtype": "float64",
        "mass_correct_every_step": True,
        "steps": nsteps,
        "sample_every": every,
        "windows": [list(w) for w in WINDOWS],
        "slopes_per_step": slopes,
        "gammas_windows": gammas,
        "slope_spread_rel": spread,
        "gamma_sim": gamma_sim,
        "gamma_ref": gref,
        "rel_err": (gamma_sim / gref - 1.0) if gref else None,
        "report_only": frac == 1.0,
        "health": {
            "mass_drift_abs": dm,
            "mass_drift_rel": dm / m0,
            "u_max": float(ux.abs().max()),
            "v_max": float(uy.abs().max()),
            "theta_min": float((Tfin - Tbar).min()),
            "theta_max": float((Tfin - Tbar).max()),
            "a_first": a_first,
            "a_last": a_last,
            "e_folds": e_folds,
        },
        "wall_seconds": wall_s,
        "abort": abort,
        "seed_rows": "npz row 0 = lower ghost mirror; rows 1..d = fluid; last = upper ghost (E3a)",
    }
    out_prefix = __import__("os").environ.get("OUT_PREFIX", "f")
    (OUTDIR / f"{out_prefix}_ny{ny}_arm{frac}.json").write_text(json.dumps(out, indent=1))
    print(
        f"ny={ny} arm={frac} N={nsteps} g_sim={gamma_sim:+.6f} g_ref={gref:+.6f} "
        f"rel={(100 * out['rel_err'] if out['rel_err'] is not None else float('nan')):+.4f}% "
        f"spread={spread:.4f} efolds={e_folds:.2f} wall={wall_s:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    levels = [int(x) for x in sys.argv[1:]] or [64, 128, 256]
    for ny in levels:
        for frac in ALL_ARMS:
            run_arm(ny, frac)
    print("FORMAL_RUN_DONE", flush=True)
