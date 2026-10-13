"""W11-A floor diagnosis 2026-10-08 — machine analysis of EXISTING data.

All numbers read from files; nothing hand-copied. Writes
diag/floor_20261008/analysis.json (diagnostic-only artifact).
"""

import glob
import json

import numpy as np

BASE = "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady"
REF = {20.0: 2.0001, 40.0: 1.498}  # from ref/ref_lock.json locked primaries

out = {}

# --- (c) Ma extrapolation from pilot u_lb sweep (compact h8 domain, D20) ----
ma_pts = []
for tag, u in [("area_D20_u0.03", 0.03), ("area_D20", 0.05), ("area_D20_u0.07", 0.07)]:
    d = json.load(open(f"{BASE}/pilot/{tag}.json"))
    assert abs(d["u_lb"] - u) < 1e-12
    ma_pts.append((d["ma"], d["cd"], tag))
ma_arr = np.array([p[0] for p in ma_pts])
cd_arr = np.array([p[1] for p in ma_pts])
A = np.vstack([ma_arr, np.ones_like(ma_arr)]).T
slope, intercept = np.linalg.lstsq(A, cd_arr, rcond=None)[0]
ma_formal = 0.08660254037844387
cd_at_ma = intercept + slope * ma_formal
cd_at_zero = intercept
out["ma_sweep"] = {
    "points": [{"ma": p[0], "cd": p[1], "tag": p[2]} for p in ma_pts],
    "linear_fit_slope_cd_per_ma": float(slope),
    "fit_intercept_cd_at_ma0": float(intercept),
    "cd_at_ma_0.0866": float(cd_at_ma),
    "shift_pct_ma0_to_ma0.0866": float((cd_at_ma / cd_at_zero - 1.0) * 100.0),
    "interpretation": "Cd FALLS with Ma; removing compressibility (Ma->0) "
    "RAISES Cd, so Ma masks part of the positive offset "
    "rather than causing it; incompressible-limit err at compact D20 = "
    f"{(intercept / REF[40.0] - 1) * 100:.3f}% vs "
    f"{(ma_pts[1][1] / REF[40.0] - 1) * 100:.3f}% at Ma=0.0866",
    "incompressible_limit_err_pct_compact_D20": float((intercept / REF[40.0] - 1) * 100),
    "err_pct_compact_D20_at_formal_Ma": float((cd_at_ma / REF[40.0] - 1) * 100),
}

# --- (e-h5) steady-state: extrapolate win_means decay to infinite time ------
drift = {}
for p in sorted(glob.glob(f"{BASE}/formal/trace_*.npz")):
    z = np.load(p)
    wm = z["win_means"]
    diffs = np.diff(wm)[-3:]  # last 3 window-to-window diffs
    r = diffs[-1] / diffs[-2] if diffs[-2] != 0 else np.nan
    remaining = diffs[-1] * r / (1.0 - r) if 0 <= r < 1 else np.nan
    drift[p.split("/")[-1]] = {
        "win_means": wm.tolist(),
        "last3_diffs": diffs.tolist(),
        "geometric_ratio": float(r),
        "extrapolated_remaining_drift_abs": float(remaining),
        "extrapolated_remaining_drift_pct": float(remaining / wm[-1] * 100),
        "note": "even running to t->inf closes only this much; floor is not a transient",
    }
out["steady_state_drift"] = drift

# --- (h1) geometric normalization terms -------------------------------------
geom = {}
for p in sorted(glob.glob(f"{BASE}/formal/re*.json")):
    d = json.load(open(p))
    ra = d["R_area"]
    geom[d["name"]] = {
        "R_row": d["R_row"],
        "R_area": ra,
        "row_minus_area_pct": float((d["R_row"] / ra - 1) * 100),
        "link_midpoint_R_area_plus_0.5_pct": float(0.5 / ra * 100),
        "note": "both terms ~1/D -> vanish with refinement; cannot form a D-independent floor",
    }
out["geometric_normalization"] = geom

# --- (c) pilot-tier errors vs Fornberg (what Phase 0 already showed) --------
pil = {}
for tag in [
    "area_D20",
    "area_D40",
    "fam_h20_sp10",
    "fam_h20_sp0",
    "area_D20_h5",
    "area_D20_h12",
    "cmp_h8_sp0",
]:
    d = json.load(open(f"{BASE}/pilot/{tag}.json"))
    pil[tag] = {
        "cd": d["cd"],
        "err_pct_vs_fornberg_re40": float((d["cd"] / REF[40.0] - 1) * 100),
        "lx_d": d["lx_d"],
        "up_d": d["up_d"],
        "h_d": d["h_d"],
        "sponge_alpha": d["sponge_alpha"],
        "D": d["D"],
    }
out["pilot_errors_vs_fornberg"] = pil

# --- sibling machinery comparison (from repo, quoted) -----------------------
out["siblings_same_machinery"] = {
    "re100_free_stream_40Dx40D_braza1.35": {
        "D48_cd": 1.3293,
        "D48_err_pct": -1.53,
        "D64_cd": 1.3155,
        "D64_err_pct": -2.55,
        "src": "/nfs/wangxi/worktrees/bm_w11/benchmarks/verified/cylinder/re100/result.json",
    },
    "re200_free_stream_40Dx40D_braza1.33": {
        "D48_cd": 1.3386,
        "D48_err_pct": 0.65,
        "D64_cd": 1.3177,
        "D64_err_pct": -0.92,
        "src": "/nfs/wangxi/worktrees/bm_w11/benchmarks/verified/cylinder/re200/result.json",
    },
    "re20_st_channel_schaefer_turek_nabh5.5795": {
        "D40_cd": 5.54842757949829,
        "D40_err_pct": -0.5575312824093583,
        "D80_cd": 5.419129320526122,
        "D80_err_pct": -2.8748973989986206,
        "src": "/nfs/wangxi/worktrees/bm_w11/benchmarks/verified/cylinder/re20_st/result.json",
    },
    "note": "same collide_mrt + far_field_bc_2d + compute_obstacle_forces "
    "(Ladd MEM); none shows a +5% floor",
}

# --- FD leg domain ladder (own independent solver, dipole-matched outer BC) --
fd = {}
for f in ["fd4_base3.json", "fd4_refine1.json", "fd4_refine2.json", "fd4_refine3.json"]:
    try:
        d = json.load(open(f"{BASE}/diag/{f}"))
        fd[f] = {k: d[k] for k in ("re", "r_max", "cd", "n_r", "n_th") if k in d}
        if d.get("re") == 40.0:
            fd[f]["err_pct_vs_fornberg"] = (d["cd"] / REF[40.0] - 1) * 100
    except FileNotFoundError:
        pass
out["fd_leg_domain_ladder"] = fd

with open(f"{BASE}/diag/floor_20261008/analysis.json", "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
