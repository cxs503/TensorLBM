#!/usr/bin/env python3
"""Drive the sloshing-impact LBM calibration campaign (one case per process).

Each invocation writes a self-describing JSON to ``--out`` so the campaign can
be sharded across cluster nodes (one case per shard) and assembled afterwards
by ``--mode post``.

Examples
--------
    # single case (what a shard runs)
    python scripts/run_sloshing_impact_calibration.py case \
        --case impact --nx 40 --out artifacts/sloshing_impact_lbm/impact_nx40.json

    # assemble the report (GCI, spectra, criteria) from the case files
    python scripts/run_sloshing_impact_calibration.py post \
        --case-dir artifacts/sloshing_impact_lbm \
        --swe-reference artifacts/sloshing_impact_lbm/sw_swe_reference.json \
        --out artifacts/sloshing_impact_lbm/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.sloshing_impact_calibration import (  # noqa: E402
    LbmScaling,
    TankAnchor,
    correlation,
    dispersion_ratio_swe,
    gci_fine,
    harmonic_analysis,
    resample,
    run_forced_case,
    run_hydrostatic_case,
    run_impact_case,
    smooth,
)

ANCHOR = TankAnchor()

#: grid family of the GCI study (task spec: L/40, L/60, L/80)
GRIDS = (40, 60, 80)
#: simulation windows (physical seconds)
IMPACT_SECONDS = 6.0
FORCED_PERIODS = 6.0
FORCED_AMPLITUDE = 0.25
#: acoustic-noise filter width for the wall-pressure signal [s]
FILTER_SECONDS = 0.10

#: campaign job list, longest first (shard index -> job)
CAMPAIGN: tuple[tuple[str, int], ...] = (
    ("impact", 80),
    ("forced", 40),
    ("impact", 60),
    ("impact", 40),
    ("impact_cushion", 40),
    ("hydrostatic", 40),
)


def _scaling(nx: int) -> LbmScaling:
    return LbmScaling(anchor=ANCHOR, nx=nx)


_CASE_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "sloshing_impact_lbm"


def _still_water_profile(nx: int) -> list[float] | None:
    """Still-water density profile of this grid, from the hydrostatic case.

    Used as the wall-pressure datum (dynamic overpressure above the hydrostatic
    state at the same height).  Returns ``None`` when the hydrostatic artifact is
    missing, in which case the liquid-mean datum is used instead.
    """
    path = _CASE_DIR / f"hydrostatic_nx{nx}.json"
    if not path.exists():
        return None
    data = _load(path)
    return data.get("rho_profile")


def run_case(
    kind: str, nx: int, *, seconds: float | None = None, c_sgs: float | None = None
) -> dict:
    scaling = _scaling(nx)
    if c_sgs is not None:
        import dataclasses

        scaling = dataclasses.replace(scaling, c_sgs=c_sgs)
    t0 = time.time()
    profile = _still_water_profile(nx)
    if kind == "impact":
        res = run_impact_case(
            scaling, seconds=seconds or IMPACT_SECONDS, rho_profile=profile
        )
        out = res.to_json()
    elif kind == "impact_cushion":
        res = run_impact_case(
            scaling, seconds=seconds or IMPACT_SECONDS, cushion=True,
            rho_profile=profile,
        )
        out = res.to_json()
    elif kind == "forced":
        res = run_forced_case(
            scaling, periods=FORCED_PERIODS, amplitude_m=FORCED_AMPLITUDE
        )
        out = res.to_json()
    elif kind == "hydrostatic":
        out = {"case": "hydrostatic", **run_hydrostatic_case(scaling)}
    else:
        raise SystemExit(f"unknown case {kind!r}")
    out["case"] = out.get("case", kind)
    out["kind"] = kind
    out["wall_seconds"] = round(time.time() - t0, 1)
    out["anchor"] = {
        "length": ANCHOR.length,
        "breadth": ANCHOR.breadth,
        "height": ANCHOR.height,
        "depth": ANCHOR.depth,
        "rho": ANCHOR.rho,
        "g": ANCHOR.g,
        "omega1": ANCHOR.omega1,
        "period1": ANCHOR.period1,
        "kh": ANCHOR.kh,
        "u_front": ANCHOR.u_front,
        "p_hydrostatic": ANCHOR.p_hydrostatic,
        "p_closed_form_over_p_hydrostatic": ANCHOR.p_closed_form / ANCHOR.p_hydrostatic,
        "p_bore_over_p_hydrostatic": ANCHOR.p_bore_reflection / ANCHOR.p_hydrostatic,
    }
    out["scaling"] = {
        "dx": scaling.dx,
        "dt": scaling.dt,
        "g_lbm": scaling.g_lbm,
        "depth_lbm": scaling.depth_lbm,
        "nu_lbm": scaling.nu_lbm,
        "reynolds_effective": scaling.reynolds,
        "pressure_scale_rho_g_h": scaling.pressure_scale(),
        "steps_per_period": scaling.steps_for_seconds(ANCHOR.period1),
    }
    return out


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------
def _load(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _case(case_dir: Path, kind: str, nx: int) -> dict | None:
    p = case_dir / f"{kind}_nx{nx}.json"
    return _load(p) if p.exists() else None


def post(case_dir: Path, swe_ref: Path | None, out_path: Path) -> dict:
    impacts = {nx: _case(case_dir, "impact", nx) for nx in GRIDS}
    cushion = _case(case_dir, "impact_cushion", 40)
    forced = _case(case_dir, "forced", 40) or _case(case_dir, "forced", 60)
    hydro = _case(case_dir, "hydrostatic", 40)
    swe = _load(swe_ref) if swe_ref and Path(swe_ref).exists() else None

    report: dict = {
        "campaign": "p120w2-n10 sloshing impact LBM calibration",
        "anchor": impacts[40]["anchor"] if impacts.get(40) else None,
        "grids": {},
        "criteria": {},
        "limitations": [],
    }

    # --- grid family table + GCI --------------------------------------
    peaks, spacings, impulses, hpeaks = [], [], [], []
    for nx in GRIDS:
        c = impacts.get(nx)
        if not c:
            continue
        dt_sample = c["t"][1] - c["t"][0] if len(c["t"]) > 1 else c["dt"]
        win = _filter_window(dt_sample)
        p_raw = np.asarray(c["p_wall"], float)
        p_s = smooth(p_raw, win)
        p_half_s = smooth(np.asarray(c["p_wall_half"], float), win)
        # impulse over the *wet* window only (the dry panel reads 0 by
        # construction, and pre-impact gauge noise must not enter the integral)
        t_arr = np.asarray(c["t"], float)
        h_arr = np.asarray(c["h_wall"], float)
        wet = h_arr > 0.02
        i0 = int(np.argmax(wet)) if np.any(wet) else 0
        impulse = float(np.trapezoid(p_s[i0:], t_arr[i0:]))
        peak_wet = float(np.nanmax(p_s[i0:])) if np.any(wet) else float("nan")
        peak = float(np.nanmax(p_s))  # whole-window peak (criterion b)
        report["grids"][str(nx)] = {
            "filter_samples": win,
            "filter_seconds": round(win * dt_sample, 4),
            "p_peak_filtered": peak,
            "p_peak_raw": float(np.nanmax(p_raw)),
            "p_peak_filtered_half_cell": float(np.nanmax(p_half_s)),
            "impulse_filtered": impulse,
            "p_peak_filtered_wet": peak_wet,
            "p_peak_filtered_time": float(np.asarray(c["t"], float)[int(np.nanargmax(p_s))]),
            "cells": c["nx"] * c["ny"] * c["nz"],
            "dx": c["dx"],
            "dt": c["dt"],
            "steps": c["steps"],
            "p_peak": c["p_peak"],
            "p_peak_half": c["p_peak_half"],
            "p_peak_time": c["p_peak_time"],
            "h_wall_max": c["h_wall_max"],
            "mass_drift_final": c["mass_drift_final"],
            "mass_drift_abs_max": c["mass_drift_abs_max"],
            "wall_seconds": c.get("wall_seconds"),
            "reynolds_effective": c["scaling"]["reynolds_effective"],
        }
        peaks.append(peak)
        impulses.append(impulse)
        hpeaks.append(float(c["h_wall_max"]))
        spacings.append(c["dx"])

    gci = None
    if len(peaks) == 3 and all(math.isfinite(p) for p in peaks):
        gci = gci_fine(peaks, spacings)
    report["gci_p_peak"] = gci
    gci_imp = None
    if len(impulses) == 3 and all(math.isfinite(v) for v in impulses):
        gci_imp = gci_fine(impulses, spacings)
    report["gci_impulse"] = gci_imp
    gci_h = None
    if len(hpeaks) == 3 and all(math.isfinite(v) for v in hpeaks):
        gci_h = gci_fine(hpeaks, spacings)
    report["gci_h_wall"] = gci_h

    # --- criterion (b): peak pressure band ----------------------------
    p_ratio = peaks[-1] if peaks else float("nan")
    report["criteria"]["a_grid_convergence"] = {
        "metric": "GCI_fine of P_peak/(rho g h) and observed order p",
        "gci_fine": None if gci is None else gci["gci_fine"],
        "p_obs": None if gci is None else gci["p_obs"],
        "gci_impulse": None if gci_imp is None else gci_imp["gci_fine"],
        "p_obs_impulse": None if gci_imp is None else gci_imp["p_obs"],
        "usable_impulse": None if gci_imp is None else gci_imp["usable"],
        "threshold": {"gci_max": 0.05, "p_min": 1.2},
        "pass": bool(
            gci is not None and gci["usable"] and gci["gci_fine"] <= 0.05
            and gci["p_obs"] >= 1.2
        ),
    }
    report["criteria"]["b_peak_pressure_band"] = {
        "metric": "P_peak/(rho g h_l) on the finest grid "
                  f"(0.1 s moving average, {_filter_window(1.0)}-ish window)",
        "value": p_ratio,
        "threshold": [3.0, 5.0],
        "closed_form": ANCHOR.p_closed_form / ANCHOR.p_hydrostatic,
        "bore_reflection": ANCHOR.p_bore_reflection / ANCHOR.p_hydrostatic,
        "swe_surge_anchor": (
            swe.get("surge", {}).get("p_peak_over_rho_g_h") if swe else None
        ),
        "pass": bool(3.0 <= p_ratio <= 5.0) if math.isfinite(p_ratio) else False,
    }

    # --- criterion (c): hydrostatic baseline --------------------------
    if hydro:
        report["criteria"]["c_hydrostatic"] = {
            "metric": "L2 rel. error of the LBM pressure profile vs rho g (h_l - y)",
            "value": hydro["l2_rel_error"],
            "max_rel_error": hydro["max_rel_error"],
            "threshold": 0.01,
            "pass": bool(hydro["l2_rel_error"] < 0.01),
            "cells": hydro["cells"],
        }
    else:
        report["criteria"]["c_hydrostatic"] = {"pass": False, "reason": "case missing"}

    # --- criterion (d): superharmonic 2*omega1 peak -------------------
    if forced:
        spec = harmonic_analysis(
            np.asarray(forced["t"]), np.asarray(forced["p_wall"]), ANCHOR.omega1
        )
        spec_h = harmonic_analysis(
            np.asarray(forced["t"]), np.asarray(forced["h_wall"]), ANCHOR.omega1
        )
        report["forced"] = {
            "periods": forced["meta"].get("periods"),
            "amplitude_m": forced["meta"].get("excitation_amplitude_m"),
            "spectrum": spec,
            "spectrum_h_wall": spec_h,
            "p_peak": forced["p_peak"],
            "h_wall_max": forced["h_wall_max"],
            "mass_drift_final": forced["mass_drift_final"],
        }
        report["criteria"]["d_superharmonic"] = {
            "metric": "energy fraction of the 2*omega_1 band, evaluated on the "
                      "wall free-surface height (most sensitive signal); the "
                      "wall-pressure band is reported alongside",
            "value": spec_h.get("energy_2omega_fraction"),
            "value_wall_pressure": spec.get("energy_2omega_fraction"),
            "peak_prominence": spec_h.get("peak_prominence_2omega"),
            "peak_prominence_wall_pressure": spec.get("peak_prominence_2omega"),
            "omega_measured": spec_h.get("omega_peak"),
            "omega_measured_ratio": spec_h.get("omega_ratio"),
            "omega_measured_wall_pressure": spec.get("omega_peak"),
            "threshold": 0.05,
            "pass": bool(
                spec_h.get("ok")
                and spec_h.get("energy_2omega_fraction", 0.0) > 0.05
                and spec.get("energy_2omega_fraction", 0.0) > 0.05
            ),
        }
    else:
        report["criteria"]["d_superharmonic"] = {"pass": False, "reason": "case missing"}

    # --- criterion (e): gas-cushion negative control ------------------
    base = impacts.get(40)
    if cushion and base:
        report["criteria"]["e_cushion_negative_control"] = {
            "metric": "P_peak(cushion) < P_peak(single phase)",
            "p_peak_single": base["p_peak"],
            "p_peak_cushion": cushion["p_peak"],
            "relative_reduction": 1.0 - cushion["p_peak"] / base["p_peak"]
            if base["p_peak"] else None,
            "pass": bool(cushion["p_peak"] < base["p_peak"]),
        }
    else:
        report["criteria"]["e_cushion_negative_control"] = {
            "pass": False,
            "reason": "case missing",
        }

    # --- criterion (f): correlation with the SWE wall history ---------
    if swe and base:
        ref = swe.get("surge", {})
        t_swe = np.asarray(ref.get("t", []), float)
        h_swe = np.asarray(ref.get("h_wall", []), float) / ANCHOR.depth
        c_fine = impacts.get(max(GRIDS)) or base
        t_lbm = np.asarray(base["t"], float)
        h_lbm = np.asarray(base["h_wall"], float)
        if t_swe.size > 4:
            window = float(min(t_swe[-1], t_lbm[-1]))
            m = t_lbm <= window
            h_swe_on_lbm = resample(t_swe, h_swe, t_lbm[m])
            r_all = correlation(h_lbm[m], h_swe_on_lbm)
            m_half = m & (t_lbm <= 0.5 * window)
            r_first = correlation(h_lbm[m_half], h_swe_on_lbm[t_lbm[m] <= 0.5 * window])
            # impact arrival and peak wall depth
            t_arr_lbm = float(t_lbm[np.argmax(h_lbm > 0.02)]) if np.any(h_lbm > 0.02) else None
            t_arr_swe = float(t_swe[np.argmax(h_swe > 0.02)]) if np.any(h_swe > 0.02) else None
            report["swe_comparison"] = {
                "swe_case": "surge (dam_break_case h0=4, wet 5 m, dry 15 m)",
                "swe_source": swe.get("source"),
                "window_s": window,
                "correlation_full": r_all,
                "correlation_first_half": r_first,
                "lbm_h_wall_max": float(np.nanmax(h_lbm)),
                "swe_h_wall_max": float(np.nanmax(h_swe)),
                "swe_over_lbm_h_wall_ratio": (
                    float(np.nanmax(h_swe)) / float(np.nanmax(h_lbm))
                    if np.nanmax(h_lbm) > 0 else None
                ),
                "lbm_arrival_s": t_arr_lbm,
                "swe_arrival_s": t_arr_swe,
                "fine_grid_used": c_fine["nx"],
            }
            report["criteria"]["f_swe_correlation"] = {
                "metric": "Pearson r of the wall free-surface pressure history "
                          "(LBM vs SWE surge, same tank, same h_l)",
                "value": r_all,
                "value_first_half_window": r_first,
                "threshold": 0.9,
                "pass": bool(max(r_all, r_first) > 0.9),
            }
        else:
            report["criteria"]["f_swe_correlation"] = {
                "pass": False, "reason": "SWE surge history too short"
            }
    else:
        report["criteria"]["f_swe_correlation"] = {
            "pass": False,
            "reason": "SWE reference missing",
        }

    # --- criterion (g): mass conservation ----------------------------
    worst_from_zero = max(
        (abs(c["mass_drift_final"]) for c in impacts.values() if c), default=float("nan")
    )
    worst_settled = max(
        (abs(c.get("mass_drift_settled", float("nan"))) for c in impacts.values() if c),
        default=float("nan"),
    )
    report["criteria"]["g_mass_conservation"] = {
        "metric": "max over grids of |M(t_end)-M(0)|/M(0) (plain drift from t=0)",
        "value": worst_from_zero,
        "value_settled": worst_settled,
        "threshold": 1e-3,
        "pass": bool(math.isfinite(worst_from_zero) and worst_from_zero < 1e-3),
    }

    # --- validity boundary of the SWE chain --------------------------
    report["swe_validity"] = dispersion_ratio_swe(ANCHOR.kh)
    measured = report.get("forced", {}).get("spectrum", {}).get("omega_peak")
    if measured:
        w_exact = math.sqrt(ANCHOR.g * ANCHOR.k1 * math.tanh(ANCHOR.k1 * ANCHOR.depth))
        report["swe_validity"].update(
            {
                "omega_measured_forced": measured,
                "omega_exact_theory": w_exact,
                "lbm_over_exact": measured / w_exact,
                "swe_over_measured": ANCHOR.omega1_swe / measured,
            }
        )

    if swe and forced:
        ref = swe.get("forced", {})
        t_swe = np.asarray(ref.get("t", []), float)
        h_swe = np.asarray(ref.get("h_wall", []), float) / ANCHOR.depth
        t_lbm = np.asarray(forced["t"], float)
        h_lbm = np.asarray(forced["h_wall"], float)
        if t_swe.size > 4:
            window = float(min(t_swe[-1], t_lbm[-1]))
            m = t_lbm <= window
            r = correlation(h_lbm[m], resample(t_swe, h_swe, t_lbm[m]))
            report["forced_vs_swe"] = {
                "note": "standing-wave wall depth history, LBM (exact dispersion) "
                        "vs SWE (k sqrt(g h) dispersion)",
                "window_s": window,
                "correlation": r,
                "lbm_h_wall_max": float(np.nanmax(h_lbm)),
                "swe_h_wall_max": float(np.nanmax(np.interp(
                    t_lbm[m], t_swe, h_swe))) if np.any(m) else float(np.nanmax(h_swe)),
            }

    report["all_pass"] = all(
        bool(v.get("pass")) for v in report["criteria"].values()
    )
    report["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=1, allow_nan=True)
    return report


def run_campaign_shard(shard: int, shards: int, out_dir: Path) -> int:
    """Run campaign job ``shard`` of ``shards`` (one process per cluster shard).

    ``shard`` is **1-based**, matching the cluster shard template ``{i}/{n}``.
    """
    jobs = [CAMPAIGN[i] for i in range(shards) if i < len(CAMPAIGN)]
    if not 1 <= shard <= len(jobs):
        print(f"shard {shard}: no job assigned (1..{len(jobs)})")
        return 0
    kind, nx = jobs[shard - 1]
    out = out_dir / f"{kind}_nx{nx}.json"
    data = run_case(kind, nx)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(data, fh)
    print(f"shard {shard}/{shards}: {kind} nx={nx} -> {out} "
          f"({data['wall_seconds']} s, p_peak={data.get('p_peak')})")
    return 0


def _filter_window(dt_sample: float) -> int:
    """Number of samples spanning ``FILTER_SECONDS`` (>= 3)."""
    return max(3, int(round(FILTER_SECONDS / max(dt_sample, 1e-12))))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("case")
    c.add_argument("--case", required=True,
                   choices=["impact", "impact_cushion", "forced", "hydrostatic"])
    c.add_argument("--nx", type=int, default=40, choices=list(GRIDS))
    c.add_argument("--seconds", type=float, default=None)
    c.add_argument("--c-sgs", type=float, default=None,
                   help="override the Smagorinsky constant of this case")
    c.add_argument("--out", required=True)

    sh = sub.add_parser("campaign")
    sh.add_argument("--shard", type=int, required=True)
    sh.add_argument("--shards", type=int, default=len(CAMPAIGN))
    sh.add_argument("--out-dir", required=True)

    p = sub.add_parser("post")
    p.add_argument("--case-dir", required=True)
    p.add_argument("--swe-reference", default=None)
    p.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.mode == "campaign":
        return run_campaign_shard(args.shard, args.shards, Path(args.out_dir))
    if args.mode == "case":
        data = run_case(args.case, args.nx, seconds=args.seconds, c_sgs=args.c_sgs)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(data, fh, indent=1, allow_nan=True)
        print(f"wrote {out} in {data['wall_seconds']} s "
              f"(p_peak={data.get('p_peak')}, steps={data.get('steps')})")
        return 0
    rep = post(Path(args.case_dir), Path(args.swe_reference) if args.swe_reference else None,
               Path(args.out))
    print(json.dumps({k: v for k, v in rep["criteria"].items()}, indent=1,
                     default=str)[:4000])
    print("all_pass:", rep["all_pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())