"""Verification study for the two honest-FAIL items of REPORT_FS_VALIDATION.md.

R1: Guo force term double-counted the half-force velocity shift -> a_eff =
    F(1+omega/2); the closed-box hydrostatic slope was 1.42/1.54/1.83 x the
    analytic 3g instead of 1.00.  After the fix the analytic law must hold.
R2: void-ward mass flux was discarded (void sender credited as mass, void
    receiver never filled) -> dam-break mass drift 69.9% in 300 steps and a
    frozen interface.  After the fix the VoF ledger must be exact.

Every number in RESULTS is computed here (no transcription).

Writes results_r1r2.json + figures in this directory.
"""
from __future__ import annotations

import json
import time

import numpy as np

import fs_oracle as F

RESULTS: dict = {"meta": {}, "hydrostatic": {}, "zero_g": {}, "dam_break": {},
                 "equilibrium": {}, "resolution": {}}


def classify_counts(sim):
    f, i, g, o = F.classify(sim.ctype)
    return int(f.sum()), int(i.sum()), int(g.sum()), int(o.sum())


# ---------------------------------------------------------------- S1  R1
def study_hydrostatic(omegas=(0.8, 1.0, 1.5), steps=2000, g=-1.0e-4):
    """Closed box full of liquid: analytic steady state d(rho)/dy = 3g."""
    rows = {}
    for om in omegas:
        p = F.FreeSurfaceParams(nx=16, ny=32, omega=om, gravity=(0.0, g))
        sim = F.FreeSurfaceSim(p)
        sim.set_liquid_from_height(lambda ix: 32.0)  # full height -> all liquid
        for _ in range(steps):
            sim.step()
        js = np.arange(4, 28)
        prof = sim.rho[8, 4:28]
        slope = float(np.polyfit(js.astype(float), prof, 1)[0])
        ana = 3.0 * g
        rows[str(om)] = dict(
            omega=om, slope=slope, analytic=ana,
            rel_err_pct=100.0 * (slope - ana) / abs(ana),
            pred_defect=1.0 + 0.5 * om,          # a_eff multiplier of the old code
            mass_drift=sim.mass_drift(),
            u_max=float(np.hypot(sim.ux, sim.uy).max()),
            rho_top=float(prof[0]), rho_bot=float(prof[-1]),
            counts=classify_counts(sim), steps=steps,
        )
        print(f"  S1 omega={om}: slope={slope:+.6e} analytic={ana:+.6e} "
              f"err={rows[str(om)]['rel_err_pct']:+.4f}% drift={sim.mass_drift():.1e}")
    RESULTS["hydrostatic"] = rows


# ---------------------------------------------------------------- S2  zero g
def study_zero_g(steps=2000):
    p = F.FreeSurfaceParams(nx=24, ny=32, omega=1.0, gravity=(0.0, 0.0))
    sim = F.FreeSurfaceSim(p)
    sim.set_liquid_from_height(lambda ix: 21.5)
    hist = []
    for t in range(1, steps + 1):
        sim.step()
        if t in (200, 1000, steps):
            ch = sim.column_height()[1:23]
            hist.append(dict(t=t, level_ptp=float(np.ptp(ch)),
                             level_mean=float(ch.mean()),
                             mass_drift=sim.mass_drift(),
                             u_max=float(np.hypot(sim.ux, sim.uy).max())))
    RESULTS["zero_g"] = dict(hist=hist, steps=steps)
    print(f"  S2 zero-g: final ptp={hist[-1]['level_ptp']:.3e} "
          f"drift={hist[-1]['mass_drift']:.1e} umax={hist[-1]['u_max']:.1e}")


# ---------------------------------------------------------------- S3  dam break
def run_dam(nx=48, ny=48, dam_w=14, dam_h=26, g=-1e-3, omega=1.0, steps=6000,
            checkpoints=(10, 25, 50, 100, 200, 300, 600, 1500, 3000, 6000)):
    p = F.FreeSurfaceParams(nx=nx, ny=ny, omega=omega, gravity=(0.0, g))
    sim = F.FreeSurfaceSim(p)
    sim.set_dam(dam_width=dam_w, dam_height=dam_h)
    mass0 = sim.total_mass()
    ncols = nx - 2                      # interior columns (walls at ix=0, nx-1)
    exact_level = mass0 / (ncols * p.rho_l)
    out = dict(nx=nx, ny=ny, dam_w=dam_w, dam_h=dam_h, g=g, omega=omega,
               mass0=mass0, ncols=ncols, exact_level=exact_level, hist=[])
    t0 = time.time()
    for t in range(1, steps + 1):
        sim.step()
        if t in checkpoints:
            ch = sim.column_height()[1:nx - 1]
            out["hist"].append(dict(
                t=t, mass=sim.total_mass(), drift=sim.mass_drift(),
                front=sim.front_position(), level_mean=float(ch.mean()),
                level_ptp=float(np.ptp(ch)),
                u_max=float(np.hypot(sim.ux, sim.uy).max()),
                counts=classify_counts(sim),
            ))
    out["wall_s"] = time.time() - t0
    # settled hydrostatic gradient: deepest column, away from both walls
    js = np.arange(4, 4 + int(exact_level) - 3)
    prof = sim.rho[24, 4:4 + int(exact_level) - 3]
    out["settled_slope"] = float(np.polyfit(js.astype(float), prof, 1)[0])
    out["settled_slope_err_pct"] = 100.0 * (out["settled_slope"] - 3.0 * g) / abs(3.0 * g)
    out["final_level"] = float(sim.column_height()[1:nx - 1].mean())
    out["level_err_pct"] = 100.0 * (out["final_level"] - exact_level) / exact_level
    out["final_ptp"] = float(np.ptp(sim.column_height()[1:nx - 1]))
    out["final_drift"] = sim.mass_drift()
    return out


def study_dam():
    out = run_dam()
    RESULTS["dam_break"] = out
    print(f"  S3 dam 48x48: mass0={out['mass0']:.1f} drift={out['final_drift']:.2e} "
          f"level={out['final_level']:.4f} exact={out['exact_level']:.4f} "
          f"({out['level_err_pct']:+.2f}%) ptp={out['final_ptp']:.2e} "
          f"settled_slope_err={out['settled_slope_err_pct']:+.2f}% "
          f"[{out['wall_s']:.1f}s]")


def study_before_after():
    """Re-run the *pre-fix* oracle (git HEAD copy) on the same two probes, so the
    before/after table is computed, never transcribed."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fs_oracle_orig", "fs_oracle_orig.py")
    import sys
    O = importlib.util.module_from_spec(spec)
    sys.modules["fs_oracle_orig"] = O      # dataclasses need the module registered
    spec.loader.exec_module(O)
    rows = {}
    for om in (0.8, 1.0, 1.5):
        p = O.FreeSurfaceParams(nx=16, ny=32, omega=om, gravity=(0.0, -1.0e-4))
        sim = O.FreeSurfaceSim(p)
        sim.set_liquid_from_height(lambda ix: 32.0)   # original (open-side) geometry
        for _ in range(2000):
            sim.step()
        js = np.arange(4, 28).astype(float)
        sl = float(np.polyfit(js, sim.rho[8, 4:28], 1)[0])
        rows[str(om)] = dict(omega=om, slope=sl, rel_err_pct=100.0 * (sl - 3 * (-1e-4)) / 3e-4,
                             pred_1_plus_half_omega=1.0 + 0.5 * om,
                             observed_ratio=sl / (-3e-4))
    # dam break with the original ledger
    p = O.FreeSurfaceParams(nx=48, ny=48, omega=1.0, gravity=(0.0, -1e-3))
    sim = O.FreeSurfaceSim(p)
    sim.set_dam(dam_width=14, dam_height=26)
    hist = []
    for t in range(1, 301):
        sim.step()
        if t in (10, 50, 100, 200, 300):
            hist.append(dict(t=t, drift=sim.mass_drift(), front=sim.front_position(),
                             counts=classify_counts(sim)))
    RESULTS["before"] = dict(hydrostatic=rows, dam_break_hist=hist)
    print("  S4 pre-fix hydrostatic:", {k: round(v["observed_ratio"], 3) for k, v in rows.items()})
    print("  S4 pre-fix dam drift@end:", round(hist[-1]["drift"], 4),
          "front@end:", hist[-1]["front"])


def _save():
    RESULTS["meta"] = dict(
        oracle="fs_oracle.py (corrected minimal D2Q9 free-surface oracle)",
        anchor="XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS",
        r1_fix="Guo f^eq/S_i evaluated at u (not u+F/2rho): a_eff = F exactly",
        r2_fix="void-ward mass flux deposited in the void cell (re-wetting); "
               "void-sender populations no longer credited as mass",
        reference_hydrostatic="d(rho)/dy = F/cs^2 = 3g (hydrostatic, p = rho cs^2)",
        reference_volume="flat free surface at h* = M/(n_cols*rho_l)",
    )
    import os
    if os.path.exists("results_r1r2.json"):
        old = json.load(open("results_r1r2.json"))
        for k, v in old.items():
            if k not in RESULTS or not RESULTS[k]:
                RESULTS[k] = v
    with open("results_r1r2.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1, sort_keys=True)


SECTIONS = {
    # name: (callable, needed-results-keys)
    "s1": lambda: (study_hydrostatic(), study_zero_g()),
    "s3": study_dam,
    "s4": study_before_after,
}

if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or list(SECTIONS)
    for name in names:
        print(f"SECTION {name}")
        SECTIONS[name]()
        _save()
    print("wrote results_r1r2.json")
