#!/usr/bin/env python3
"""m23014 — H3 (geometry/blockage/boundary) + H4 (sampling window) discriminators.

Local, cheap: no solver run. Reads the frozen series of the DFG 2D-2 case and the
lattice masks produced by the *same* code path the solver uses.

H3 asks: is the discretised solid boundary / blockage / probe placement biased?
H4 asks: is the reported Cl amplitude a statistics artifact (transient in window,
         or too short a window)?

The amplitude estimator here is self-contained (ACF period + least-squares
fundamental fit) and is REQUIRED to reproduce the frozen m16827 re-judged values
before any H4 conclusion is drawn -- see `reproduce_check`.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.validation import cylinder_dfg as cdfg  # noqa: E402
from tensorlbm.boundary.boundaries import cylinder_mask, make_channel_wall_mask  # noqa: E402

ART = REPO / "benchmarks/verified/cylinder/re100_dfg/artifacts"
REF_CL = cdfg.REFERENCE_BANDS["2D-2"]["cl_amp"]
REF_MID = 0.5 * sum(REF_CL)


# --------------------------------------------------------------------------- #
# estimator (self-contained, documented; same convention as the frozen case)
# --------------------------------------------------------------------------- #
def acf_period(x: np.ndarray, lo: int, hi: int) -> float:
    x = x - x.mean()
    n = x.size
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac /= ac[0]
    lo = max(lo, 2)
    hi = min(hi, n - 2)
    seg = ac[lo:hi]
    # first local maximum after the first zero crossing
    zc = np.where(ac[lo:hi] < 0.0)[0]
    start = lo + int(zc[0]) if zc.size else lo
    k = start + int(np.argmax(ac[start:hi]))
    return float(k)


def fundamental_amp(cl: np.ndarray, period: float) -> float:
    n = np.arange(cl.size, dtype=float)
    w = 2.0 * np.pi / period
    A = np.column_stack([np.cos(w * n), np.sin(w * n), np.ones_like(n)])
    coef, *_ = np.linalg.lstsq(A, cl, rcond=None)
    return float(math.hypot(coef[0], coef[1]))


def per_period_peak_amps(cl: np.ndarray, period: float) -> list[float]:
    p = int(round(period))
    out = []
    for i in range(cl.size // p):
        seg = cl[i * p:(i + 1) * p]
        out.append(0.5 * (seg.max() - seg.min()))
    return out


def analyse_window(cl: np.ndarray, D: int, u_mean: float, n_periods: int) -> dict:
    """Cl amplitude over the LAST n_periods shedding periods of `cl`."""
    period = cdfg.CaseConfig(D_cells=D, u_mean_lb=u_mean, n_steps=100_000).period_steps
    n = int(round(n_periods * period))
    tail = cl[-n:] if n < cl.size else cl
    per = acf_period(tail.astype(float), lo=int(0.4 * period), hi=int(4.0 * period))
    amp = fundamental_amp(tail.astype(float), per)
    peaks = per_period_peak_amps(tail.astype(float), per)
    return {
        "n_periods": n_periods,
        "n_samples": int(tail.size),
        "period_acf": per,
        "cl_amp_fundamental": amp,
        "err_pct": 100.0 * (amp - REF_MID) / REF_MID,
        "peakpick_median": float(np.median(peaks)) if peaks else float("nan"),
        "peak_excess_pct": (100.0 * (float(np.median(peaks)) - amp) / amp) if peaks else None,
        "n_complete_periods": len(peaks),
    }


# --------------------------------------------------------------------------- #
# H3 — geometry / blockage / probe placement
# --------------------------------------------------------------------------- #
def geometry_audit(levels=(20, 28, 40, 48, 56)) -> dict:
    out = {}
    for D in levels:
        cfg = cdfg.CaseConfig(D_cells=D)
        dx = cfg.dx
        cx = round(cdfg.CYL_CX / dx)
        cy = round(cdfg.CYL_CY / dx)
        r = 0.5 * cdfg.CYL_D / dx
        solid = cylinder_mask(cfg.nx, cfg.ny, cx, cy, r, torch.device("cpu"))
        wall = make_channel_wall_mask(cfg.ny, cfg.nx, solid, torch.device("cpu"))
        nsolid = int(solid.sum())
        area_exact = math.pi * (0.5 * cdfg.CYL_D / dx) ** 2
        # half-way bounce-back puts the hydrodynamic surface half a cell outside
        # the outermost solid cell centre -> effective radius r+0.5 in lattice units
        r_eff = r + 0.5
        probe = cdfg.probe_columns(cfg)
        # x of the DFG probe positions in lattice units
        xa = cdfg.PROBE_A[0] / dx
        xb = cdfg.PROBE_B[0] / dx
        # wall-normal distance: first fluid row above y=0 wall
        first_fluid_row = 1
        while wall[first_fluid_row, cx] and first_fluid_row < cfg.ny - 1:
            first_fluid_row += 1
        out[f"D{D}"] = {
            "nx": cfg.nx, "ny": cfg.ny, "dx": dx, "tau": cfg.tau, "nu_lb": cfg.nu_lb,
            "re_check": cfg.u_mean_lb * D / cfg.nu_lb,
            "cell_centre_radius": r,
            "solid_cells": nsolid,
            "area_circle_pi_r2": area_exact,
            "area_ratio_mask_over_circle": nsolid / area_exact,
            "effective_radius_halfway": r_eff,
            "effective_diameter_over_nominal": (D + 1.0) / D,
            "blockage_nominal_D_over_H": D / cfg.ny,
            "blockspec_D_over_H": 0.1 / 0.41,
            "blockage_effective": (D + 1.0) / cfg.ny,
            "cd_bias_predicted_pct_from_effective_Da": 100.0 * ((D + 1.0) / D - 1.0),
            "probe": probe,
            "probe_x_lattice": {"a": xa, "b": xb,
                                "a_is_half_cell": abs(xa - math.floor(xa) - 0.5) < 1e-9,
                                "b_is_half_cell": abs(xb - math.floor(xb) - 0.5) < 1e-9},
            "probe_up_near_x": probe["x_up_near"] / dx,
            "probe_dn_near_x": probe["x_dn_near"] / dx,
            "probe_gap_cells_from_surface_up": (xa - (cx - r)),
            "probe_gap_cells_from_surface_dn": ((cx + r) - xb),
            "wall_first_fluid_row": first_fluid_row,
            "cylinder_to_inlet_D": (cdfg.CYL_CX - 0.5 * cdfg.CYL_D) / cdfg.CYL_D,
            "cylinder_to_outlet_D": (cdfg.LX - cdfg.CYL_CX - 0.5 * cdfg.CYL_D) / cdfg.CYL_D,
        }
    return out


# --------------------------------------------------------------------------- #
# H4 — window sensitivity on frozen series
# --------------------------------------------------------------------------- #
def _level_D(level) -> int:
    """Accept 40 or 'D40'; return 40."""
    return int(str(level).lstrip("D"))


def load_series(D, tag: str = "u0833") -> dict:
    p = ART / f"level_D{_level_D(D)}_{tag}.series.npz"
    z = np.load(p)
    return {k: z[k].astype(np.float64) for k in z.files}


def window_sweep(levels=(20, 28, 40), periods=(4, 6, 8, 10, 12)) -> dict:
    res = {}
    for D in levels:
        s = load_series(D)
        cl = s["cl"]
        u = 0.0833 if "u0833" else None
        rows = [analyse_window(cl, D, u, p) for p in periods]
        res[f"D{D}"] = {"series_len": int(cl.size), "windows": rows}
    return res


def transient_decomposition(levels=(20, 28, 40)) -> dict:
    """Split the old headline into: whole-record peak-pick vs post-transient
    peak-pick vs fundamental.  (Reproduces and cites the m16827 decomposition.)"""
    frozen = json.loads((ART / "rejudge_cl_amp.json").read_text())
    out = {}
    for lv_key in levels:
        D = _level_D(lv_key)
        s = load_series(D)
        cl = s["cl"]
        cfg = cdfg.CaseConfig(D_cells=D, u_mean_lb=0.0833, n_steps=100_000)
        period = cfg.period_steps
        n_tr = int(round(cfg.transient_periods * period))
        kept = cl[n_tr:]
        per = acf_period(kept, lo=int(0.4 * period), hi=int(4.0 * period))
        amp_new = fundamental_amp(kept, per)
        old_peak = 0.5 * (cl.max() - cl.min())
        kept_peak = float(np.median(per_period_peak_amps(kept, per)))
        fz = frozen["levels"][f"D{D}"]
        out[f"D{D}"] = {
            "old_peakpick_whole_record": float(old_peak),
            "post_transient_peakpick_median": kept_peak,
            "fundamental": amp_new,
            "transient_share_pct_of_value": 100.0 * (old_peak - kept_peak) / old_peak,
            "peak_vs_fundamental_pct": 100.0 * (kept_peak - amp_new) / amp_new,
            "err_pct_old": 100.0 * (old_peak - REF_MID) / REF_MID,
            "err_pct_new": 100.0 * (amp_new - REF_MID) / REF_MID,
            "frozen_m16827_cl_amp_new": fz["cl_amp_new"],
            "reproduce_delta_pct": 100.0 * (amp_new - fz["cl_amp_new"]) / fz["cl_amp_new"],
            "n_transient_steps": n_tr,
        }
    return out


# --------------------------------------------------------------------------- #
# H2 — tau / viscosity-model sensitivity, from frozen data (no new run)
# --------------------------------------------------------------------------- #
def tau_pair_frozen(tag_a: str = "u0600", tag_b: str = "u0833", D: int = 20) -> dict:
    """Same geometry D20, same Re=100, two U_mean -> two tau.

    Measures whether the force coefficients move when tau moves at FIXED geometry.
    If they do, the collision/relaxation (tau) model could be the error source;
    if they do not, resolution is the only remaining candidate.  Frozen data, so
    this answer costs nothing and does not depend on the cluster queue.
    """
    out = {"D_cells": D, "tags": [tag_a, tag_b], "arms": {}}
    frozen = json.loads((ART / "rejudge_cl_amp.json").read_text())
    for tag in (tag_a, tag_b):
        s = load_series(D, tag)
        u = float(f"0.{tag[1:]}") if tag.startswith("u0") else 0.0833
        cfg = cdfg.CaseConfig(D_cells=D, u_mean_lb=u, n_steps=100_000)
        per = cfg.period_steps
        n_tr = int(round(cfg.transient_periods * per))
        kept = slice(n_tr, None)
        cl = s["cl"]
        per_est = acf_period(cl[kept], lo=int(0.4 * per), hi=int(4.0 * per))
        amp = fundamental_amp(cl[kept], per_est)
        arm = {
            "tag": tag, "u_mean_lb": u, "tau": cfg.tau, "nu_lb": cfg.nu_lb,
            "period_steps": per, "n_transient": n_tr,
            "cl_amp_fundamental": amp,
            "cl_err_pct": 100.0 * (amp - REF_MID) / REF_MID,
            "cd_mean_window": float(np.mean(s["cd"][kept])),
            "dp_mean_window": float(np.mean(s["dp"][kept])),
            "st_acf": float(D / (per_est * u)),
        }
        out["arms"][tag] = arm

    # self-check: the u0833 arm must reproduce the frozen re-judged cd
    fz = frozen["levels"][f"D{D}"]["values_new"]["cd"]
    out["cd_selfcheck_frozen"] = fz
    out["cd_selfcheck_delta_pct"] = 100.0 * (out["arms"][tag_b]["cd_mean_window"] - fz) / fz

    a, b = out["arms"][tag_a], out["arms"][tag_b]
    out["deltas_pct"] = {
        "tau_span": b["tau"] - a["tau"],
        "cl_amp": 100.0 * (a["cl_amp_fundamental"] - b["cl_amp_fundamental"]) / b["cl_amp_fundamental"],
        "cd": 100.0 * (a["cd_mean_window"] - b["cd_mean_window"]) / b["cd_mean_window"],
        "dp": 100.0 * (a["dp_mean_window"] - b["dp_mean_window"]) / b["dp_mean_window"],
        "st": 100.0 * (a["st_acf"] - b["st_acf"]) / b["st_acf"],
    }
    d = abs(out["deltas_pct"]["cl_amp"])
    out["h2_verdict"] = (
        "tau_model_dominant" if d >= 5.0 else
        "tau_model_not_dominant" if d < 2.0 else "tau_model_marginal"
    )
    return out


def main() -> int:
    payload = {
        "case": "CASE-CYLINDER-DRAG-RE100",
        "mission": "m23014",
        "reference_cl_amp_band": list(REF_CL),
        "reference_mid": REF_MID,
        "H3_geometry": geometry_audit(),
        "H4_transient_decomposition": transient_decomposition(),
        "H4_window_sweep": window_sweep(),
        "H2_tau_pair_frozen": tau_pair_frozen(),
    }
    out = HERE / "h3_h4_audit.json"
    out.write_text(json.dumps(payload, indent=2))

    print("== H3 geometry ==")
    for k, v in payload["H3_geometry"].items():
        print(f"{k}: nx={v['nx']} ny={v['ny']} tau={v['tau']:.3f} Re_check={v['re_check']:.2f}")
        print(f"   solid={v['solid_cells']} / pi r^2={v['area_circle_pi_r2']:.1f} "
              f"ratio={v['area_ratio_mask_over_circle']:.4f}")
        print(f"   blockage nominal={v['blockage_nominal_D_over_H']:.5f} "
              f"spec={v['blockspec_D_over_H']:.5f} effective={v['blockage_effective']:.5f}")
        print(f"   eff Da={v['effective_diameter_over_nominal']:.5f} "
              f"-> predicted cd bias {v['cd_bias_predicted_pct_from_effective_Da']:.2f}%")
        print(f"   probe gap up/dn (cells from surface)={v['probe_gap_cells_from_surface_up']:.2f}/"
              f"{v['probe_gap_cells_from_surface_dn']:.2f} inletDist="
              f"{v['cylinder_to_inlet_D']:.2f}D outletDist={v['cylinder_to_outlet_D']:.2f}D")

    print("\n== H4 transient decomposition ==")
    for k, v in payload["H4_transient_decomposition"].items():
        print(f"{k}: old={v['old_peakpick_whole_record']:.4f} "
              f"post-tr med={v['post_transient_peakpick_median']:.4f} "
              f"fund={v['fundamental']:.4f} err_new={v['err_pct_new']:+.2f}% "
              f"repro_delta={v['reproduce_delta_pct']:+.3f}%")

    print("\n== H4 window sweep (fundamental amp / err%) ==")
    for lv, v in payload["H4_window_sweep"].items():
        row = "  ".join(f"{w['n_periods']}p:{w['cl_amp_fundamental']:.4f}({w['err_pct']:+.1f}%)"
                        for w in v["windows"])
        print(f"{lv}: {row}")
    h2 = payload["H2_tau_pair_frozen"]
    print("\n== H2 tau sensitivity (frozen D20 pair) ==")
    for t, a in h2["arms"].items():
        print(f"  {t}: tau={a['tau']:.3f} u={a['u_mean_lb']:.4f} "
              f"cl_amp={a['cl_amp_fundamental']:.4f} ({a['cl_err_pct']:+.2f}%) "
              f"cd={a['cd_mean_window']:.4f} dp={a['dp_mean_window']:.4f} st={a['st_acf']:.4f}")
    print("  deltas%:", {k: round(v, 3) for k, v in h2["deltas_pct"].items()})
    print("  cd self-check vs frozen rejudge:",
          round(h2["cd_selfcheck_frozen"], 4), "-> delta",
          round(h2["cd_selfcheck_delta_pct"], 4), "%")
    print("  verdict:", h2["h2_verdict"])

    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())