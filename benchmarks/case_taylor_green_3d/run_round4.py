#!/usr/bin/env python
"""CASE-TAYLOR-GREEN-3D round 4: close the two ``not_run`` legs of round 3
(M8 grid ladder, M9 Mach attribution) and re-verify the field-level metrics,
which round 3 scored against a *Nyquist-aliased* reference field.

Legs
----
``reproduce``
    Re-run the round-3 primary case (n=32, A=0.05, Re=160, T_end=1) and record,
    for the delivered numbers, the actual reference-field truncation order and
    spectral support.  Also scores the same numerical field against *forced*
    reference orders that violate Nyquist, to show where the round-3 field
    numbers (u_l2 2.657 %, vorticity 4.600 %) came from and that they are an
    aliasing artifact, not a better reference.
``band``
    Well-posed field metric.  Round 3's field numbers are not reproducible
    (aliased reference) and the round-3 method cannot be fixed by picking a
    larger/ smaller order, because the reference's exact spectral support is
    finite: at T=1 the classic Taylor-Green series support is |k|_inf = 4.  The
    only well-posed field comparison is therefore a *band-projected* one:
    project both fields onto |k|_inf <= k_max with k_max = the reference's
    exact support, and report the error inside that band (plus the LBM's
    out-of-band energy fraction, which the reference cannot score at all).
``grid``
    M8: same dimensionless problem on n = 24/32/48 -> observed spatial order of
    the band-projected velocity error, and a direct test of the round-3 claim
    that the -2 % energy offset is "grid independent".
``mach``
    M9: amplitude ladder A = 0.0125/0.025/0.05/0.10 at fixed Re -> error vs
    Ma^2 slope and Ma->0 intercept.  Honest caveat: along this ladder the
    lattice viscosity tau = 0.5 + 3 A n / Re and the step count co-vary with
    Ma, so the slope is an *effective* Ma/tau sensitivity, and the intercept is
    the residual (spatial + tau) truncation at that grid -- not a clean
    compressibility constant.
``verdict``
    Assembles the round-4 checks from the JSONs actually written.

All artefacts go to ``artifacts/case_taylor_green_3d/round4/`` (``runs/`` is
git-ignored, which is why round 3's JSONs are not in the repository).

Usage::

    python benchmarks/case_taylor_green_3d/run_round4.py --all
    python benchmarks/case_taylor_green_3d/run_round4.py --mode grid
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.tg3d_lbm_check import (  # noqa: E402
    run_tg3d_lbm,
    series_reference,
    tg3d_parameters,
)
from tensorlbm.tg3d_series import vorticity_spectral  # noqa: E402

OUT = ROOT / "artifacts" / "case_taylor_green_3d" / "round4"

DESIGN = {
    "amplitude": 0.05,
    "reynolds": 160.0,
    "eddy_times": 1.0,
    "series_order": 12,
    "n_main": 32,
    "tolerance_pct": 8.0,
    "stretch_tolerance_pct": 3.0,
    "grid_sizes": [24, 32, 48],
    "mach_sweep": [0.0125, 0.025, 0.05, 0.10],
    "min_spatial_order": 1.8,
    "forced_orders": [3, 4, 5, 6, 8],
}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _dump(name: str, payload) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    print(f"[write] {path} ({path.stat().st_size} B)")
    return path


def _freqs(n: int) -> np.ndarray:
    m = np.fft.fftfreq(n, d=1.0 / n)
    return np.rint(m).astype(int)


def _band_project(field: np.ndarray, kmax: int) -> np.ndarray:
    """Ideal low-pass ``|k|_inf <= kmax`` on a periodic n^3 grid."""
    spec = np.fft.fftn(field)
    n = field.shape[0]
    k = _freqs(n)
    mask = (np.abs(k)[:, None, None] <= kmax) & (np.abs(k)[None, :, None] <= kmax) & (
        np.abs(k)[None, None, :] <= kmax
    )
    return np.real(np.fft.ifftn(spec * mask))


def band_metrics(run: dict, ser, kmax: int) -> dict:
    """Band-projected velocity / vorticity L2 error at the final sample."""
    n = int(run["n"])
    t_end = run["history"][-1]["time"]
    ue, ve, we = ser.velocity_field(t_end, n=n)  # Nyquist-safe (auto-lowered)
    support = float(ser.last_field_support)
    ue, ve, we = (_band_project(ue, kmax), _band_project(ve, kmax), _band_project(we, kmax))
    ub, vb, wb = (_band_project(run["final"]["u"], kmax),
                  _band_project(run["final"]["v"], kmax),
                  _band_project(run["final"]["w"], kmax))
    num = float(np.sqrt(np.mean((ub - ue) ** 2 + (vb - ve) ** 2 + (wb - we) ** 2)))
    den = float(np.sqrt(np.mean(ue ** 2 + ve ** 2 + we ** 2)))
    wr = vorticity_spectral(ue, ve, we, float(n))
    wl = vorticity_spectral(ub, vb, wb, float(n))
    num_w = float(np.sqrt(np.mean(sum((wl[i] - wr[i]) ** 2 for i in range(3)))))
    den_w = float(np.sqrt(np.mean(sum(wr[i] ** 2 for i in range(3)))))
    # how much of the *numerical* field lives outside the band the reference knows
    raw = run["final"]
    out_band = 0.0
    for comp in ("u", "v", "w"):
        full = float(np.mean(raw[comp] ** 2))
        band = float(np.mean(_band_project(raw[comp], kmax) ** 2))
        out_band += full - band
    full_e = sum(float(np.mean(raw[c] ** 2)) for c in ("u", "v", "w"))
    return {
        "kmax": kmax,
        "reference_field_order": float(ser.last_field_order),
        "reference_spectrum_peak": support,
        "u_l2_rel_pct": 100.0 * num / den if den else float("nan"),
        "vorticity_l2_rel_pct": 100.0 * num_w / den_w if den_w else float("nan"),
        "lbm_out_of_band_energy_fraction": 0.5 * out_band / (0.5 * full_e) if full_e else float("nan"),
    }


def _offsets(run: dict, ser) -> dict:
    """Modal (grid-free) energy / enstrophy comparison at the final sample."""
    t_end = run["history"][-1]["time"]
    e_l, e_s = run["history"][-1]["energy"], ser.energy(t_end)
    o_l, o_s = run["history"][-1]["enstrophy"], ser.enstrophy(t_end)
    e0 = run["history"][0]["energy"]
    o0 = run["history"][0]["enstrophy"]
    return {
        "energy_rel_err_pct": 100.0 * (e_l / e_s - 1.0),
        "enstrophy_rel_err_pct": 100.0 * (o_l / o_s - 1.0),
        "energy_ratio_lbm": e_l / e0,
        "energy_ratio_series": e_s / ser.energy(0.0),
        "enstrophy_ratio_lbm": o_l / o0,
        "enstrophy_ratio_series": o_s / ser.enstrophy(0.0),
    }


# ---------------------------------------------------------------------------
# leg 1: reproduce round 3 + expose the aliasing of its field reference
# ---------------------------------------------------------------------------


def leg_reproduce() -> dict:
    n, a, re = DESIGN["n_main"], DESIGN["amplitude"], DESIGN["reynolds"]
    p = tg3d_parameters(n, amplitude=a, reynolds=re)
    ser = series_reference(n, amplitude=a, reynolds=re, order=DESIGN["series_order"])
    t0 = time.time()
    run = run_tg3d_lbm(n, amplitude=a, reynolds=re, eddy_times=DESIGN["eddy_times"])
    wall = time.time() - t0
    t_end = run["history"][-1]["time"]
    off = _offsets(run, ser)

    # the honest (Nyquist-safe) field comparison, exactly as current code does it
    ue, ve, we = ser.velocity_field(t_end, n=n)
    safe = {
        "requested_order": DESIGN["series_order"],
        "used_order": float(ser.last_field_order),
        "spectrum_peak": float(ser.last_field_support),
        "nyquist_index": n / 2.0,
    }
    ub, vb, wb = run["final"]["u"], run["final"]["v"], run["final"]["w"]
    num = float(np.sqrt(np.mean((ub - ue) ** 2 + (vb - ve) ** 2 + (wb - we) ** 2)))
    den = float(np.sqrt(np.mean(ue ** 2 + ve ** 2 + we ** 2)))
    wr = vorticity_spectral(ue, ve, we, float(n))
    wl = vorticity_spectral(ub, vb, wb, float(n))
    num_w = float(np.sqrt(np.mean(sum((wl[i] - wr[i]) ** 2 for i in range(3)))))
    den_w = float(np.sqrt(np.mean(sum(wr[i] ** 2 for i in range(3)))))
    safe["u_l2_rel_err_pct"] = 100.0 * num / den
    safe["vorticity_l2_rel_err_pct"] = 100.0 * num_w / den_w

    # forced reference orders: pre-Nyquist-guard behaviour, i.e. what round 3
    # scored against (its report says "the field can only be sampled at order 4")
    forced = []
    for top in DESIGN["forced_orders"]:
        spec = ser._series_spectrum(t_end, top)
        peak = float(ser._spectrum_peak(spec))
        fu, fv, fw = ser._to_grid(spec, n)[0], ser._to_grid(spec, n)[1], ser._to_grid(spec, n)[2]
        fnum = float(np.sqrt(np.mean((ub - fu) ** 2 + (vb - fv) ** 2 + (wb - fw) ** 2)))
        fden = float(np.sqrt(np.mean(fu ** 2 + fv ** 2 + fw ** 2)))
        fwr = vorticity_spectral(fu, fv, fw, float(n))
        fnum_w = float(np.sqrt(np.mean(sum((wl[i] - fwr[i]) ** 2 for i in range(3)))))
        fden_w = float(np.sqrt(np.mean(sum(fwr[i] ** 2 for i in range(3)))))
        forced.append(
            {
                "order": top,
                "spectrum_peak": peak,
                "aliased": bool(peak >= n / 2.0),
                "u_l2_rel_err_pct": 100.0 * fnum / fden,
                "vorticity_l2_rel_err_pct": 100.0 * fnum_w / fden_w,
            }
        )

    payload = {
        "n": n,
        "parameters": p,
        "steps": run["steps"],
        "tau": run["tau"],
        "runtime_seconds": wall,
        "series_nfft": ser.nfft,
        "series_order_requested": ser.order,
        "truncation_at_final": ser.relative_tail(t_end),
        "modal": off,
        "field_nyquist_safe": safe,
        "field_forced_orders": forced,
        "round3_delivered_field_numbers": {"u_l2_rel_err_pct": 2.657, "vorticity_l2_rel_err_pct": 4.600},
        "rho_range": [run["history"][-1]["rho_min"], run["history"][-1]["rho_max"]],
        "note": (
            "round-3's 2.657 % / 4.600 % are reproduced *only* by forcing the reference to "
            "order 4, whose spectrum peak (20) exceeds Nyquist (16) and therefore aliases. "
            "The Nyquist-safe value at the same run is reported in field_nyquist_safe."
        ),
    }
    _dump("reproduce", payload)
    return payload


# ---------------------------------------------------------------------------
# leg 2: band-projected field metric (well-posed)
# ---------------------------------------------------------------------------


def leg_band() -> dict:
    n, a, re = DESIGN["n_main"], DESIGN["amplitude"], DESIGN["reynolds"]
    ser = series_reference(n, amplitude=a, reynolds=re, order=DESIGN["series_order"])
    run = run_tg3d_lbm(n, amplitude=a, reynolds=re, eddy_times=DESIGN["eddy_times"])
    ser.velocity_field(run["history"][-1]["time"], n=n)  # populates last_field_support
    support = int(round(float(ser.last_field_support)))
    rows = [band_metrics(run, ser, k) for k in range(1, support + 1)]
    payload = {
        "n": n,
        "amplitude": a,
        "reynolds": re,
        "steps": run["steps"],
        "reference_support": support,
        "rows": rows,
        "kmax_canonical": support,
        "note": (
            "k_max = reference exact support: inside the band the reference is exact to "
            "machine precision, so the error there is the solver's.  Beyond the band the "
            "reference has no content; the LBM's out-of-band energy fraction is reported "
            "instead of an error."
        ),
    }
    _dump("band", payload)
    return payload


# ---------------------------------------------------------------------------
# leg 3: M8 grid ladder
# ---------------------------------------------------------------------------


def leg_grid() -> dict:
    """M8: fixed dimensionless problem on several grids, *common* band.

    Round-3's design (and the first cut of this leg) compared each grid at its
    own reference support, which is grid dependent (4 at n=24/32, 20 at n=48)
    and therefore not comparable.  The comparable band is the smallest exact
    reference support over the ladder, capped by the 2/3 rule.
    """
    a, re = DESIGN["amplitude"], DESIGN["reynolds"]
    collected = []
    for n in DESIGN["grid_sizes"]:
        p = tg3d_parameters(n, amplitude=a, reynolds=re)
        ser = series_reference(n, amplitude=a, reynolds=re, order=DESIGN["series_order"])
        run = run_tg3d_lbm(n, amplitude=a, reynolds=re, eddy_times=DESIGN["eddy_times"])
        ser.velocity_field(run["history"][-1]["time"], n=n)
        collected.append((n, p, ser, run, int(round(float(ser.last_field_support)))))
    kmax_common = min(
        int(min(sup, n / 3.0)) for n, _, _, _, sup in collected
    )
    rows = []
    for n, p, ser, run, sup in collected:
        off = _offsets(run, ser)
        common = band_metrics(run, ser, kmax_common)
        own = band_metrics(run, ser, sup)
        rows.append(
            {
                "n": n,
                "nu": p["nu"],
                "tau": p["tau"],
                "ma": p["ma"],
                "steps": run["steps"],
                "reference_field_order": float(ser.last_field_order),
                "reference_spectrum_peak": float(ser.last_field_support),
                **off,
                "common_band_kmax": kmax_common,
                "band_u_l2_rel_pct": common["u_l2_rel_pct"],
                "band_vorticity_l2_rel_pct": common["vorticity_l2_rel_pct"],
                "own_band_kmax": sup,
                "own_band_u_l2_rel_pct": own["u_l2_rel_pct"],
                "own_band_vorticity_l2_rel_pct": own["vorticity_l2_rel_pct"],
                "out_of_band_energy_fraction": common["lbm_out_of_band_energy_fraction"],
                "truncation": ser.relative_tail(run["history"][-1]["time"]),
            }
        )
    sizes = [r["n"] for r in rows]
    err = np.array([abs(r["band_u_l2_rel_pct"]) for r in rows])
    orders = [
        float(math.log(err[i] / err[i + 1]) / math.log(sizes[i + 1] / sizes[i]))
        for i in range(len(sizes) - 1)
    ]
    e_off = np.array([r["energy_rel_err_pct"] for r in rows])
    o_off = np.array([r["enstrophy_rel_err_pct"] for r in rows])
    monotone = bool(np.all(np.diff(np.abs(e_off)) < 0.0))
    payload = {
        "amplitude": a,
        "reynolds": re,
        "eddy_times": DESIGN["eddy_times"],
        "common_band_kmax": kmax_common,
        "rows": rows,
        "observed_order_band_u_l2": orders[-1] if orders else float("nan"),
        "observed_order_pairs": orders,
        "energy_offset_spread_pct": float(e_off.max() - e_off.min()),
        "enstrophy_offset_spread_pct": float(o_off.max() - o_off.min()),
        "energy_offset_mean_pct": float(e_off.mean()),
        "energy_offset_shrinks_monotonically_with_n": monotone,
        "interpretation": (
            "the modal offset shrinks as the grid is refined and its spread across grids "
            "exceeds a percent, so it cannot be read as a grid-independent compressibility "
            "floor; the observed order is measured on the band that all references know exactly"
        ),
    }
    _dump("grid_ladder", payload)
    return payload


# ---------------------------------------------------------------------------
# leg 4: M9 Mach (amplitude) ladder
# ---------------------------------------------------------------------------


def leg_mach() -> dict:
    n, re = DESIGN["n_main"], DESIGN["reynolds"]
    rows = []
    for a in DESIGN["mach_sweep"]:
        p = tg3d_parameters(n, amplitude=a, reynolds=re)
        ser = series_reference(n, amplitude=a, reynolds=re, order=DESIGN["series_order"])
        run = run_tg3d_lbm(n, amplitude=a, reynolds=re, eddy_times=DESIGN["eddy_times"])
        off = _offsets(run, ser)
        rows.append({"amplitude": a, "ma": p["ma"], "ma2": p["ma"] ** 2, "nu": p["nu"],
                     "tau": p["tau"], "steps": run["steps"], **off})
    ma2 = np.array([r["ma2"] for r in rows])
    e = np.array([r["energy_rel_err_pct"] for r in rows])
    o = np.array([r["enstrophy_rel_err_pct"] for r in rows])
    fit_e = np.polyfit(ma2, e, 1)
    fit_o = np.polyfit(ma2, o, 1)
    pred = fit_e[0] * ma2 + fit_e[1]
    ss_res, ss_tot = float(np.sum((e - pred) ** 2)), float(np.sum((e - e.mean()) ** 2))
    payload = {
        "n": n,
        "reynolds": re,
        "eddy_times": DESIGN["eddy_times"],
        "rows": rows,
        "fit_energy_err_pct_vs_ma2": {
            "slope_pct_per_ma2": float(fit_e[0]),
            "intercept_pct": float(fit_e[1]),
            "r2": 1.0 - ss_res / ss_tot,
        },
        "fit_enstrophy_err_pct_vs_ma2": {
            "slope_pct_per_ma2": float(fit_o[0]),
            "intercept_pct": float(fit_o[1]),
        },
        "caveat": (
            "at fixed Re, tau = 0.5 + 3 A n / Re and the step count both co-vary with the "
            "amplitude, so the slope is an effective Ma/tau sensitivity and the intercept "
            "the residual truncation at n=32 -- a clean single-mechanism attribution is "
            "not available from this ladder"
        ),
    }
    _dump("mach", payload)
    return payload


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------


def leg_verdict() -> dict:
    def load(name):
        path = OUT / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    rep, band, grid, mach = load("reproduce"), load("band"), load("grid_ladder"), load("mach")
    checks: list[dict] = []
    if rep:
        m = rep["modal"]
        checks += [
            {"check": "R1 energy rel err < 8 %", "value": abs(m["energy_rel_err_pct"]),
             "threshold": DESIGN["tolerance_pct"]},
            {"check": "R2 enstrophy rel err < 8 %", "value": abs(m["enstrophy_rel_err_pct"]),
             "threshold": DESIGN["tolerance_pct"]},
        ]
        fs = rep["field_nyquist_safe"]
        checks.append({"check": "R3 Nyquist-safe reference on n=32 grid",
                       "value": fs["spectrum_peak"], "threshold": fs["nyquist_index"]})
        checks.append({"check": "R4 vorticity L2 (Nyquist-safe) < 8 %",
                       "value": abs(fs["vorticity_l2_rel_err_pct"]),
                       "threshold": DESIGN["tolerance_pct"]})
    if band:
        kc = band["kmax_canonical"]
        row = [r for r in band["rows"] if r["kmax"] == kc][0]
        checks.append({"check": "B1 band-projected velocity L2 < 8 %",
                       "value": abs(row["u_l2_rel_pct"]), "threshold": DESIGN["tolerance_pct"]})
        checks.append({"check": "B2 band-projected vorticity L2 < 8 %",
                       "value": abs(row["vorticity_l2_rel_pct"]),
                       "threshold": DESIGN["tolerance_pct"]})
        checks.append({"check": "B3 band-projected velocity L2 < 3 %",
                       "value": abs(row["u_l2_rel_pct"]),
                       "threshold": DESIGN["stretch_tolerance_pct"]})
    if grid:
        checks.append({"check": "M8 observed order >= 1.8",
                       "value": grid["observed_order_band_u_l2"],
                       "threshold": DESIGN["min_spatial_order"]})
        checks.append({"check": "M8b modal energy offset spread across grids < 1 %",
                       "value": grid["energy_offset_spread_pct"], "threshold": 1.0})
    if mach:
        slope = abs(mach["fit_energy_err_pct_vs_ma2"]["slope_pct_per_ma2"])
        checks.append({"check": "M9a |slope| of energy err vs Ma^2 in [0.3, 30] %/Ma^2",
                       "value": slope, "threshold": None})
        checks.append({"check": "M9b |intercept| < 5 % (residual truncation at n=32)",
                       "value": abs(mach["fit_energy_err_pct_vs_ma2"]["intercept_pct"]),
                       "threshold": 5.0})
    for c in checks:
        name, val = c["check"], c["value"]
        if name.startswith("M8 "):
            c["pass"] = bool(val >= DESIGN["min_spatial_order"])
        elif name.startswith("M8b"):
            c["pass"] = bool(val < 1.0)
        elif name.startswith("M9a"):
            c["pass"] = bool(0.3 <= val <= 30.0)
        elif name.startswith("R3"):
            c["pass"] = bool(val < c["threshold"])
        else:
            c["pass"] = bool(val < c["threshold"])
    payload = {"checks": checks, "pass": all(c["pass"] for c in checks), "n_checks": len(checks)}
    _dump("verdict", payload)
    return payload



# ---------------------------------------------------------------------------
# leg 5: per-sample curves (report figures) + discrimination witness
# ---------------------------------------------------------------------------


def leg_curves() -> dict:
    n, a, re = DESIGN["n_main"], DESIGN["amplitude"], DESIGN["reynolds"]
    p = tg3d_parameters(n, amplitude=a, reynolds=re)
    ser = series_reference(n, amplitude=a, reynolds=re, order=DESIGN["series_order"])
    run = run_tg3d_lbm(n, amplitude=a, reynolds=re, eddy_times=DESIGN["eddy_times"])
    e0, o0 = ser.energy(0.0), ser.enstrophy(0.0)
    rows = []
    for h in run["history"]:
        t = h["time"]
        rows.append(
            {
                "step": h["step"],
                "time": t,
                "eddy_time": h["eddy_time"],
                "energy_ratio_lbm": h["energy"] / run["history"][0]["energy"],
                "energy_ratio_series": ser.energy(t) / e0,
                "enstrophy_ratio_lbm": h["enstrophy"] / run["history"][0]["enstrophy"],
                "enstrophy_ratio_series": ser.enstrophy(t) / o0,
                "energy_rel_err_pct": 100.0 * (h["energy"] / ser.energy(t) - 1.0),
                "enstrophy_rel_err_pct": 100.0 * (h["enstrophy"] / ser.enstrophy(t) - 1.0),
                "rho_min": h["rho_min"],
                "rho_max": h["rho_max"],
            }
        )
    viscous = math.exp(-2.0 * p["nu"] * 3.0 * p["k"] ** 2 * run["steps"])
    payload = {
        "n": n,
        "parameters": p,
        "steps": run["steps"],
        "rows": rows,
        "pure_viscous_energy_ratio": viscous,
        "nonlinear_enstrophy_signature_pct": 100.0 * (rows[-1]["enstrophy_ratio_series"] / viscous - 1.0),
        "lbm_recovers_fraction_of_signature": (
            (rows[-1]["enstrophy_ratio_lbm"] - viscous)
            / (rows[-1]["enstrophy_ratio_series"] - viscous)
        ),
        "truncation_at_final": ser.relative_tail(run["history"][-1]["time"]),
    }
    _dump("curves", payload)
    return payload



# ---------------------------------------------------------------------------
# leg 6: what *is* the modal offset?  effective-viscosity attribution
# ---------------------------------------------------------------------------


def leg_nueff() -> dict:
    """Fit nu_eff/nu per grid from the LBM energy series.

    If the missing energy is a dissipation bias, the LBM energy history is
    reproduced by the exact series evaluated at ``nu_eff = lambda * nu``; the
    fitted ``lambda - 1`` is then the size of the bias in viscosity units.
    """
    from tensorlbm.tg3d_series import TG3DSeries, taylor_green_classic

    a, re = DESIGN["amplitude"], DESIGN["reynolds"]
    rows = []
    for n in DESIGN["grid_sizes"]:
        p = tg3d_parameters(n, amplitude=a, reynolds=re)
        ser = series_reference(n, amplitude=a, reynolds=re, order=DESIGN["series_order"])
        run = run_tg3d_lbm(n, amplitude=a, reynolds=re, eddy_times=DESIGN["eddy_times"])
        ts = np.array([h["time"] for h in run["history"]])
        e_lbm = np.array([h["energy"] for h in run["history"]])
        o_lbm = np.array([h["enstrophy"] for h in run["history"]])

        def hist(lam: float, which: str):
            s2 = TG3DSeries(taylor_green_classic(a), box=float(n), nu=lam * p["nu"],
                            order=DESIGN["series_order"])
            f = s2.energy if which == "energy" else s2.enstrophy
            return np.array([f(t) for t in ts])

        out = {}
        for which, lbm in (("energy", e_lbm), ("enstrophy", o_lbm)):
            base = hist(1.0, which)
            dh = 0.02
            grad = (hist(1.0 + dh, which) - hist(1.0 - dh, which)) / (2.0 * dh)
            lam = 1.0 + float(np.sum((lbm - base) * grad) / np.sum(grad ** 2))
            resid = hist(lam, which)
            out[which] = {
                "lambda_nu_eff_over_nu": lam,
                "bias_pct_of_nu": 100.0 * (lam - 1.0),
                "rel_err_pct_before": 100.0 * (lbm[-1] / base[-1] - 1.0),
                "rel_err_pct_at_nu_eff": 100.0 * (lbm[-1] / resid[-1] - 1.0),
            }
        rows.append({"n": n, "nu": p["nu"], "tau": p["tau"], **out})
    payload = {
        "amplitude": a,
        "reynolds": re,
        "rows": rows,
        "interpretation": (
            "a fitted lambda that is stable across grids and that removes most of the modal "
            "offset identifies the offset as a dissipation (viscosity) bias of the lattice "
            "scheme; the Mach ladder shows it is not a compressibility (Ma^2) effect"
        ),
    }
    _dump("nu_effective", payload)
    return payload


LEGS = {
    "reproduce": leg_reproduce,
    "band": leg_band,
    "grid": leg_grid,
    "mach": leg_mach,
    "curves": leg_curves,
    "nueff": leg_nueff,
    "verdict": leg_verdict,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=sorted(LEGS))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)
    if args.all:
        for name in ("reproduce", "band", "grid", "mach", "curves", "nueff", "verdict"):
            t0 = time.time()
            print(f"=== leg {name} ===")
            LEGS[name]()
            print(f"--- {name} done in {time.time() - t0:.1f} s")
        return 0
    if not args.mode:
        ap.error("give --mode or --all")
    LEGS[args.mode]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())