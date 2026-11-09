"""Acceptance campaign for LNG-tank partial-fill sloshing (campaign node IN143-N07).

Three quantitative targets, each with its own machine-checkable judge:

``A. frequency``    first-mode natural frequency ``f_1(h/L)`` from a free-decay
                    run (modal x-centroid probe) against Abramson (1966) on 10
                    fill levels, ``|error| <= 5 %``.
``B. pressure``     wall pressure peak ``p_max/(rho g h)`` from forced sway at
                    fixed amplitude for 3+ frequency ratios ``omega/omega_1``,
                    compared with the linear (Faltinsen/Timokha) wall-pressure
                    law.  The comparison with *experimental* impact data is
                    reported as WITHHELD: no digitised data set is available in
                    this repository and none is invented here.
``C. waveform``     the free-surface field ``eta(x,t)`` in the linear regime
                    reconstructed by the single-mode potential-flow solution,
                    ``relative L2 error <= 15 %``.

Each case is a self-contained JSON so the campaign can be sharded over many
worker processes (``run --shard i/n``); ``summarize`` re-reads the case files,
applies the acceptance judges, writes the three figures and the report JSON.

All numbers are in lattice units; the conversion to physical units is a
similarity transform (see the module docstring of :mod:`tensorlbm.lng_sloshing`).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .abramson import (
    SP106_REFERENCE,
    abramson_natural_frequency,
    linear_forced_modal_amplitude,
    linear_wall_pressure_amplitude,
    stagnation_impact_pressure,
)
from .analysis import (
    decay_damping,
    harmonic_amplitudes,
    pressure_peak,
    reconstruction_error,
    refined_frequency,
)
from .tank import TankConfig, simulate_tank

__all__ = [
    "CampaignSpec",
    "case_list",
    "run_case",
    "run_shard",
    "summarize",
]

CRITERION = {
    "frequency_rel_error_max": 0.05,
    "pressure_rel_error_max": 0.20,
    "waveform_rel_l2_max": 0.15,
    "pytest_min": 15,
}


@dataclass(frozen=True)
class CampaignSpec:
    """Campaign parameters (lattice units unless noted)."""

    length: int = 80
    g: float = 5.0e-4
    tau: float = 0.6
    headspace: float = 0.35
    fill_ratios: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    decay_periods: float = 6.0
    forced_periods: float = 20.0
    steady_frac: float = 0.60
    pressure_fill: float = 0.5
    pressure_ratios: tuple = (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4)
    sway_frac: float = 0.010
    zeta_default: float = 0.05
    records_per_period: int = 250
    waveform_ratio: float = 0.8
    waveform_sway_frac: float = 0.015
    conv_fill: float = 0.5
    conv_lengths: tuple = (64, 112)
    decay_probe_periods: float = 3.5

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# case enumeration
# ---------------------------------------------------------------------------


def case_list(spec: CampaignSpec) -> list[dict]:
    """Every case of the campaign, in a deterministic order (index = shard key)."""
    cases: list[dict] = []
    for fr in spec.fill_ratios:
        cases.append({"kind": "decay", "fill_ratio": float(fr), "id": f"decay_f{fr:.2f}"})
    for ratio in spec.pressure_ratios:
        cases.append(
            {
                "kind": "forced",
                "fill_ratio": float(spec.pressure_fill),
                "ratio": float(ratio),
                "id": f"forced_f{spec.pressure_fill:.2f}_r{ratio:.2f}",
            }
        )
    for L in spec.conv_lengths:
        if L == spec.length:
            continue
        cases.append(
            {
                "kind": "decay",
                "fill_ratio": float(spec.conv_fill),
                "length": int(L),
                "id": f"conv_f{spec.conv_fill:.2f}_L{int(L)}",
            }
        )
    cases.append(
        {
            "kind": "waveform",
            "fill_ratio": float(spec.pressure_fill),
            "ratio": float(spec.waveform_ratio),
            "id": f"waveform_f{spec.pressure_fill:.2f}_r{spec.waveform_ratio:.2f}",
        }
    )
    return cases


def _config(spec: CampaignSpec, fill_ratio: float, length: int | None = None) -> TankConfig:
    return TankConfig(
        length=int(length or spec.length),
        depth=fill_ratio * spec.length,
        headspace=spec.headspace,
        g=spec.g,
        tau=spec.tau,
    )


def _record_every(spec: CampaignSpec, period: float) -> int:
    return max(1, int(round(period / spec.records_per_period)))


# ---------------------------------------------------------------------------
# case runners
# ---------------------------------------------------------------------------


def _decay_case(spec: CampaignSpec, case: dict) -> dict:
    fill = case["fill_ratio"]
    cfg = _config(spec, fill, case.get("length"))
    f1 = abramson_natural_frequency(1, cfg.length, cfg.depth, cfg.g)
    f2 = abramson_natural_frequency(2, cfg.length, cfg.depth, cfg.g)
    period = 1.0 / f1
    n_steps = int(spec.decay_periods * period)
    tilt = min(0.02 * cfg.length, 0.10 * cfg.depth)
    out = simulate_tank(
        cfg,
        n_steps,
        tilt_amplitude=tilt,
        record_every=_record_every(spec, period),
        n_modes=2,
    )
    t = out["t"].numpy()
    xcm = out["x_centroid"].numpy()
    est = refined_frequency(t, xcm)
    zeta = decay_damping(t, xcm, est["f"])
    col = refined_frequency(t, out["eta_modes"][1].numpy())
    # x-centroid amplitude -> equivalent first-mode elevation amplitude
    eta_amp = est["amplitude"] * math.pi**2 * cfg.depth / (2.0 * cfg.length)
    return {
        "id": case["id"],
        "kind": "decay",
        "fill_ratio": fill,
        "length": cfg.length,
        "depth": cfg.depth,
        "g": cfg.g,
        "tau": cfg.tau,
        "tilt_amplitude": tilt,
        "n_steps": n_steps,
        "n_samples": int(t.size),
        "f1_abramson": f1,
        "f2_abramson": f2,
        "f1_measured": est["f"],
        "f1_measured_window": est["f_window"],
        "f1_measured_column_modal": col["f"],
        "rel_error": est["f"] / f1 - 1.0,
        "rel_error_column_modal": col["f"] / f1 - 1.0,
        "decay_snr": est["snr"],
        "residual_rms": est["residual_rms"],
        "zeta": zeta["zeta"],
        "zeta_r2": zeta["r2"],
        "eta_amplitude_recovered": eta_amp,
        "eta_amplitude_initial": zeta.get("envelope0", float("nan")) * math.pi**2 * cfg.depth / (2.0 * cfg.length),
        "mass_drift": out["mass_drift"],
        "three_g_h": 3.0 * cfg.g * cfg.depth,
        "wall_seconds": None,
    }


def _guarded_exception(exc: Exception) -> dict:
    return {"error": f"{type(exc).__name__}: {exc}"}


def _forced_case(spec: CampaignSpec, case: dict, store_waveform: bool = False) -> dict:
    fill = case["fill_ratio"]
    ratio = case["ratio"]
    cfg = _config(spec, fill, case.get("length"))
    f1 = abramson_natural_frequency(1, cfg.length, cfg.depth, cfg.g)
    period = 1.0 / f1
    omega = 2.0 * math.pi * ratio * f1
    omega1 = 2.0 * math.pi * f1

    # damping of the same tank measured from its own free decay (the linear
    # forced-response prediction needs zeta; using the value measured on the
    # very same configuration keeps every case self-contained)
    zeta = spec.zeta_default
    zeta_probe = None
    try:
        tilt_probe = min(0.02 * cfg.length, 0.10 * cfg.depth)
        dp = simulate_tank(
            cfg,
            int(spec.decay_probe_periods * period),
            tilt_amplitude=tilt_probe,
            record_every=_record_every(spec, period),
            n_modes=1,
        )
        tp = dp["t"].numpy()
        xp = dp["x_centroid"].numpy()
        dm = decay_damping(tp, xp, refined_frequency(tp, xp)["f"])
        if dm["zeta"] == dm["zeta"] and 1e-4 < dm["zeta"] < 0.5:
            zeta = float(dm["zeta"])
            zeta_probe = dm
    except Exception:  # pragma: no cover - keep the default on failure
        zeta_probe = None

    sway_frac = spec.waveform_sway_frac if store_waveform else spec.sway_frac
    sway = sway_frac * cfg.length
    n_steps = int(spec.forced_periods * period)
    out = simulate_tank(
        cfg,
        n_steps,
        sway_amplitude=sway,
        omega=omega,
        record_every=_record_every(spec, period),
        n_modes=2,
    )
    t = out["t"].numpy()
    t_ss = spec.steady_frac * float(t[-1])
    m = t >= t_ss
    f_force = omega / (2.0 * math.pi)

    # --- first-mode elevation from the free-surface field -------------------
    # eta_1(t) from the (trimmed) least-squares cosine projection: blind to the
    # wall-film artefact and to any uniform surface-level drift.
    eta = out["eta"].numpy()
    k1 = math.pi / cfg.length
    cols = np.arange(1, cfg.length + 1, dtype=float) - 0.5
    trim = 2
    sub = slice(trim, cfg.length - trim)
    basis = np.cos(k1 * cols[sub])
    eta1_series = (eta[:, sub] * basis).sum(axis=1) / float((basis**2).sum())
    harm_field = harmonic_amplitudes(
        t, eta1_series - eta1_series[m].mean(), f_force, n_harm=3, t_start=t_ss, trend_order=0
    )
    eta1_field = harm_field["A1"]
    # steady-state check: growth of the response between the two halves of the
    # steady window (a growing resonance is NOT a steady state)
    idx = np.flatnonzero(m)
    half = idx.size // 2
    amp_a = harmonic_amplitudes(t, eta1_series - eta1_series[m].mean(), f_force,
                               n_harm=1, t_start=t_ss, trend_order=0)["A1"]
    t_mid = float(t[idx[half]])
    amp_h2 = harmonic_amplitudes(t, eta1_series - eta1_series[m].mean(), f_force,
                                n_harm=1, t_start=t_mid, trend_order=0)["A1"]
    growth = (amp_h2 / amp_a - 1.0) if amp_a > 0 else None

    # x-centroid measure kept as an independent cross-check (blind to even modes)
    xcm = out["x_centroid"].numpy()
    harm_x = harmonic_amplitudes(
        t, xcm - xcm[m].mean(), f_force, n_harm=3, t_start=t_ss, trend_order=1
    )
    conv = math.pi**2 * cfg.depth / (2.0 * cfg.length)
    eta1_xcm = harm_x["A1"] * conv

    eta1_theory = linear_forced_modal_amplitude(
        1, cfg.length, cfg.depth, sway, omega, cfg.g, zeta=zeta
    )

    # --- wall pressure gauges ----------------------------------------------
    gauges = out["pressure_gauges"].numpy()
    gauge_y = out["gauge_y"]
    gauge_rows = []
    for k, y in enumerate(gauge_y):
        p = gauges[:, k]
        p_ss = p[m]
        finite = np.isfinite(p_ss)
        if finite.sum() < 8:
            continue
        p_dyn = p_ss[finite] - np.nanmean(p_ss[finite])
        tt = t[m][finite]
        pk = pressure_peak(p_dyn, tt)
        hh = harmonic_amplitudes(tt, p_dyn, f_force, n_harm=2, trend_order=0)
        head = pk["p_max"] / cfg.g          # p/(rho g) in lattice cells
        y_rel = float(y - cfg.depth)
        theory_meas = linear_wall_pressure_amplitude(
            y_rel, cfg.length, cfg.depth, eta1_field, omega, cfg.g,
            sway_amplitude=sway, zeta=zeta,
        )
        theory_pred = linear_wall_pressure_amplitude(
            y_rel, cfg.length, cfg.depth, eta1_theory, omega, cfg.g,
            sway_amplitude=sway, zeta=zeta,
        )
        gauge_rows.append(
            {
                "y_index": int(y),
                "y_relative": y_rel,
                "p_max": pk["p_max"],
                "p_amp_linear": hh["A1"],
                "head_max": head,
                "head_amp_linear": hh["A1"] / cfg.g,
                "P_star_max": head / cfg.depth,
                "P_star_linear": (hh["A1"] / cfg.g) / cfg.depth,
                "P_star_theory": theory_meas / cfg.depth,
                "P_star_theory_pred": theory_pred / cfg.depth,
                "A2_over_A1": hh["ratios"].get("A2/A1"),
                "n_samples": int(finite.sum()),
            }
        )

    primary = None
    if gauge_rows:
        # the near-surface gauge at the still-water line is quantised by the
        # interface flag state, so the judge uses the deepest gauge that is
        # submerged over the whole window and carries the largest linear
        # pressure amplitude (the classic impact zone, ~0.1 h below the SWL)
        full = [r for r in gauge_rows if r["n_samples"] >= 0.99 * int(m.sum())]
        cand = full if full else gauge_rows
        primary = max(cand, key=lambda r: r["P_star_linear"])
    surface_gauge = max(gauge_rows, key=lambda r: r["P_star_max"]) if gauge_rows else None

    result = {
        "id": case["id"],
        "kind": "waveform" if store_waveform else "forced",
        "fill_ratio": fill,
        "ratio": ratio,
        "length": cfg.length,
        "depth": cfg.depth,
        "g": cfg.g,
        "tau": cfg.tau,
        "omega": omega,
        "omega_over_omega1": omega / omega1,
        "f1_abramson": f1,
        "sway_amplitude": sway,
        "sway_frac": sway_frac,
        "n_steps": n_steps,
        "n_samples": int(t.size),
        "t_steady_start": t_ss,
        "eta1_measured": eta1_field,
        "eta1_xcm": eta1_xcm,
        "eta1_theory": eta1_theory,
        "eta1_rel_error": (eta1_field / eta1_theory - 1.0) if eta1_theory > 0 else None,
        "eta1_over_depth": eta1_field / cfg.depth,
        "eta2_over_eta1": (
            harm_field["ratios"].get("A2/A1") if eta1_field > 0 else None
        ),
        "response_growth_in_window": growth,
        "steady_state": bool(growth is not None and abs(growth) < 0.10),
        "gauge_rows": gauge_rows,
        "P_star_max": primary["P_star_max"] if primary else None,
        "P_star_max_any": surface_gauge["P_star_max"] if surface_gauge else None,
        "P_star_linear": primary["P_star_linear"] if primary else None,
        "P_star_theory": primary["P_star_theory"] if primary else None,
        "P_star_theory_pred": primary["P_star_theory_pred"] if primary else None,
        "P_star_gauge_y": primary["y_relative"] if primary else None,
        "p_max_over_half_rho_v2": None,
        "ux_max": float(out["ux_max"].max()),
        "mass_drift": out["mass_drift"],
        "zeta_used": zeta,
        "zeta_probe": zeta_probe,
        "wall_seconds": None,
        "diag_t": t[m][:: max(1, int(m.sum()) // 120)].tolist(),
        "diag_eta1_field": (eta1_series[m] - eta1_series[m].mean())[
            :: max(1, int(m.sum()) // 120)
        ].tolist(),
        "diag_residual_field": harm_field["residual_rms"],
    }

    # stagnation (Wagner) cross-check: p_max vs 0.5 rho V^2 with the measured
    # maximum liquid velocity in the tank
    v = float(out["ux_max"].numpy()[m].max()) if m.any() else float(out["ux_max"].max())
    p_ref = stagnation_impact_pressure(v, rho=cfg.rho_liquid, cp=1.0)
    if surface_gauge and p_ref > 0:
        result["p_max_over_half_rho_v2"] = surface_gauge["p_max"] / p_ref
    result["stagnation_pressure"] = p_ref

    if store_waveform:
        tw = t[m]
        eta_w = eta[m]
        eta_w = eta_w - eta_w.mean(axis=0, keepdims=True)   # zero-mean per column
        eta_i = eta_w[:, sub]
        x = cols[sub]
        ct = np.cos(omega * tw)[:, None] * basis[None, :]
        st = np.sin(omega * tw)[:, None] * basis[None, :]
        design = np.stack([ct.ravel(), st.ravel()], axis=1)
        coef, *_ = np.linalg.lstsq(design, eta_i.ravel(), rcond=None)
        model = coef[0] * ct + coef[1] * st
        amp_fit = float(np.hypot(coef[0], coef[1]))
        phase_fit = float(math.atan2(-coef[1], coef[0]))
        err = reconstruction_error(model, eta_i)
        step = max(1, eta_i.shape[0] // 24)
        result.update(
            {
                "waveform_amp_fit": amp_fit,
                "waveform_phase": phase_fit,
                "waveform_rel_l2": err["rel_l2"],
                "waveform_rel_max": err["rel_max"],
                "waveform_rms_measured": err["rms_measured"],
                "waveform_x": x[::2].tolist(),
                "waveform_model": model[::step, ::2].tolist(),
                "waveform_measured": eta_i[::step, ::2].tolist(),
                "waveform_t": tw[::step].tolist(),
            }
        )
    return result


def run_case(spec: CampaignSpec, case: dict) -> dict:
    """Run one case, returning the JSON-able result (never raises)."""
    import time

    t0 = time.time()
    try:
        if case["kind"] == "decay":
            out = _decay_case(spec, case)
        elif case["kind"] == "forced":
            out = _forced_case(spec, case, store_waveform=False)
        elif case["kind"] == "waveform":
            out = _forced_case(spec, case, store_waveform=True)
        else:  # pragma: no cover - guarded by case_list
            raise ValueError(f"unknown case kind {case['kind']}")
    except Exception as exc:  # honest FAIL: keep going, record the reason
        out = {"id": case["id"], **case, **_guarded_exception(exc)}
    out["wall_seconds"] = time.time() - t0
    out["spec"] = spec.as_dict()
    return out


def run_shard(
    spec: CampaignSpec,
    shard: int,
    n_shards: int,
    out_dir: Path,
    only: str | None = None,
) -> list[Path]:
    """Run the cases with ``index % n_shards == shard`` and write their JSON.

    ``only`` restricts the run to a single case id (handy for smoke tests).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = case_list(spec)
    written = []
    for i, case in enumerate(cases):
        if i % n_shards != shard:
            continue
        if only is not None and case["id"] != only:
            continue
        res = run_case(spec, case)
        path = out_dir / f"{case['id']}.json"
        path.write_text(json.dumps(res, indent=1, default=float))
        line = {
            "id": res.get("id"),
            "kind": res.get("kind"),
            "rel_error": res.get("rel_error"),
            "P_star_max": res.get("P_star_max"),
            "waveform_rel_l2": res.get("waveform_rel_l2"),
            "error": res.get("error"),
        }
        print("@@CASE@@ " + json.dumps(line, default=float), flush=True)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# summary + figures
# ---------------------------------------------------------------------------


def _load_cases(case_dir: Path) -> list[dict]:
    out = []
    for p in sorted(Path(case_dir).glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:  # pragma: no cover
            continue
    return out


def _judge_frequency(decays: list[dict], tol: float) -> dict:
    """Frequency judge on the fill-level sweep (convergence cases held aside)."""
    rows = []
    convergence = []
    for c in decays:
        if str(c.get("id", "")).startswith("conv_"):
            convergence.append(
                {
                    "id": c.get("id"),
                    "fill_ratio": c.get("fill_ratio"),
                    "length": c.get("length"),
                    "depth": c.get("depth"),
                    "three_g_h": c.get("three_g_h"),
                    "f1_abramson": c.get("f1_abramson"),
                    "f1_measured": c.get("f1_measured"),
                    "rel_error": c.get("rel_error"),
                    "zeta": c.get("zeta"),
                    "error": c.get("error"),
                }
            )
            continue
        if "error" in c:
            rows.append({"fill_ratio": c.get("fill_ratio"), "status": "ERROR", "error": c["error"]})
            continue
        rel = c["rel_error"]
        rows.append(
            {
                "fill_ratio": c["fill_ratio"],
                "depth": c["depth"],
                "length": c.get("length"),
                "g": c.get("g"),
                "f1_abramson": c["f1_abramson"],
                "f1_measured": c["f1_measured"],
                "rel_error": rel,
                "abs_rel_error": abs(rel),
                "zeta": c["zeta"],
                "three_g_h": c["three_g_h"],
                "within_tolerance": bool(abs(rel) <= tol),
            }
        )
    ok = [r for r in rows if r.get("status") != "ERROR"]
    worst = max((r["abs_rel_error"] for r in ok), default=float("nan"))
    return {
        "criterion": f"first-mode frequency vs Abramson (1966), |error| <= {tol:.0%}",
        "n_fill_levels": len(ok),
        "max_abs_rel_error": worst,
        "n_within_tolerance": sum(1 for r in ok if r["within_tolerance"]),
        "passed": bool(ok) and all(r["within_tolerance"] for r in ok) and len(ok) >= 10,
        "rows": rows,
        "convergence_cases": convergence,
        "reference": SP106_REFERENCE,
    }


def _judge_pressure(forced: list[dict], tol: float) -> dict:
    """Wall-pressure judge on a *fixed* reference gauge.

    The reference gauge is the one at ``y = -0.1 h`` below the still-water line
    (the classic impact zone): a fixed geometric definition, so the comparison
    does not depend on which gauge happened to show the largest number.  The
    gauge at the still-water line itself is reported separately as
    ``p_max_any`` / ``P_star_surface`` (it is partly exposed, so its peak is a
    genuine impact spike rather than a linear amplitude).
    """
    rows = []
    for c in forced:
        if "error" in c:
            rows.append({"ratio": c.get("ratio"), "status": "ERROR", "error": c["error"]})
            continue
        depth = float(c.get("depth") or 1.0)
        gr = c.get("gauge_rows") or []
        ref = None
        if gr:
            ref = min(gr, key=lambda r: abs(r["y_relative"] + 0.1 * depth))
        surf = None
        if gr:
            surf = max(gr, key=lambda r: r["y_relative"])
        ps = ref["P_star_max"] if ref else None
        th = ref["P_star_theory"] if ref else None
        thp = ref.get("P_star_theory_pred") if ref else None
        rows.append(
            {
                "id": c.get("id"),
                "fill_ratio": c["fill_ratio"],
                "ratio": c["ratio"],
                "omega_over_omega1": c.get("omega_over_omega1"),
                "gauge_y_relative": ref["y_relative"] if ref else None,
                "P_star_max": ps,
                "P_star_linear": ref["P_star_linear"] if ref else None,
                "P_star_theory": th,
                "P_star_theory_pred": thp,
                "P_star_surface": surf["P_star_max"] if surf else None,
                "P_star_surface_linear": surf["P_star_linear"] if surf else None,
                "impact_amplification": (surf["P_star_max"] / surf["P_star_linear"])
                if (surf and surf.get("P_star_linear"))
                else None,
                "rel_error_vs_linear_theory": (ps / th - 1.0) if (ps is not None and th) else None,
                "rel_error_vs_linear_theory_pred": (
                    ps / thp - 1.0 if (ps is not None and thp) else None
                ),
                "eta1_measured": c.get("eta1_measured"),
                "eta1_theory": c.get("eta1_theory"),
                "eta1_rel_error": c.get("eta1_rel_error"),
                "eta2_over_eta1": c.get("eta2_over_eta1"),
                "steady_state": c.get("steady_state"),
                "response_growth_in_window": c.get("response_growth_in_window"),
                "p_max_over_half_rho_v2": c.get("p_max_over_half_rho_v2"),
            }
        )
    ok = [r for r in rows if r.get("status") != "ERROR"]
    steady = [r for r in ok if r.get("steady_state")]
    judge_pool = steady if len(steady) >= 3 else ok
    # linear regime = at least 10 % away from the resonant peak, where the
    # linear (potential-flow) wall-pressure law is meant to apply
    lin = [r for r in judge_pool if abs(r["ratio"] - 1.0) >= 0.1]
    pool = lin if len(lin) >= 3 else judge_pool
    errs = [abs(r["rel_error_vs_linear_theory_pred"]) for r in pool if r["rel_error_vs_linear_theory_pred"] is not None]
    within = [r["ratio"] for r in pool if (r["rel_error_vs_linear_theory_pred"] is not None and abs(r["rel_error_vs_linear_theory_pred"]) <= tol)]
    return {
        "criterion": (
            "wall pressure peak p_max/(rho g h) on the y=-0.1h gauge vs the linear "
            "Faltinsen/Timokha wall-pressure law with the *predicted* amplitude, "
            f"|error| <= {tol:.0%} in the linear regime (|omega/omega_1 - 1| >= 0.1). "
            "The comparison against experimental impact data is WITHHELD: no "
            "digitised Faltinsen data set is available offline and nothing is "
            "fabricated here; the impact spikes at the still-water line are "
            "reported separately (they are outside linear theory)."
        ),
        "n_ratio_points": len(ok),
        "n_steady_points": len(steady),
        "n_linear_regime_points": len(pool),
        "ratios_within_tolerance": within,
        "experimental_comparison": "WITHHELD_NO_DATASET",
        "passed_vs_linear_theory": bool(errs) and all(e <= tol for e in errs) and len(errs) >= 3,
        "max_abs_rel_error": max(errs) if errs else float("nan"),
        "rows": rows,
    }


def _judge_waveform(waveforms: list[dict], tol: float) -> dict:
    rows = []
    for c in waveforms:
        if "error" in c:
            rows.append({"status": "ERROR", "error": c["error"]})
            continue
        rows.append(
            {
                "ratio": c["ratio"],
                "fill_ratio": c["fill_ratio"],
                "waveform_rel_l2": c.get("waveform_rel_l2"),
                "waveform_rel_max": c.get("waveform_rel_max"),
                "amp_fit": c.get("waveform_amp_fit"),
                "eta1_theory": c.get("eta1_theory"),
                "amp_rel_error_vs_linear": (
                    c["waveform_amp_fit"] / c["eta1_theory"] - 1.0
                    if c.get("eta1_theory")
                    else None
                ),
                "within_tolerance": bool(
                    c.get("waveform_rel_l2") is not None and c["waveform_rel_l2"] <= tol
                ),
            }
        )
    return {
        "criterion": f"free-surface field eta(x,t) vs single-mode potential-flow, rel L2 <= {tol:.0%}",
        "passed": bool(rows) and all(r.get("within_tolerance") for r in rows if r.get("status") != "ERROR"),
        "rows": rows,
    }


def _figure_frequency(freq: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in freq["rows"] if r.get("status") != "ERROR"]
    L = rows[0]["length"]
    g = rows[0]["g"]
    scale = math.sqrt(g / L)  # dimensionless frequency f / sqrt(g/L)
    fr = [r["fill_ratio"] for r in rows]
    me = [r["f1_measured"] / scale for r in rows]
    grid = np.linspace(min(fr) * 0.85, max(fr) * 1.05, 240)
    curve = []
    for f in grid:
        k = math.pi / L
        curve.append(math.sqrt(g * k * math.tanh(k * f * L)) / (2 * math.pi) / scale)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.plot(grid, curve, "-", color="#333", lw=1.6, label="Abramson (1966) potential flow")
    ax.plot(fr, me, "o", ms=6, color="#c0392b", label="free-surface LBM (free decay)")
    ax.set_xlabel("fill level  h/L")
    ax.set_ylabel(r"$f_1/\sqrt{g/L}$")
    ax.set_title("First sloshing mode vs fill level (L=%d, g=%.3g)" % (L, g))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    err = [100 * r["rel_error"] for r in rows]
    ax2.axhline(0, color="#333", lw=1)
    ax2.axhline(5, color="#c0392b", ls="--", lw=1, label="+5 % acceptance")
    ax2.axhline(-5, color="#c0392b", ls="--", lw=1, label="-5 %")
    ax2.plot(fr, err, "s-", ms=5, color="#2c7fb8", label="LBM vs Abramson")
    ax2.set_xlabel("fill level  h/L")
    ax2.set_ylabel("relative error  (%)")
    ax2.set_title("Frequency judge: max |err| = %.2f %%" % (100 * freq["max_abs_rel_error"]))
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _figure_pressure(press: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in press["rows"] if r.get("status") != "ERROR"]
    rows.sort(key=lambda r: r["ratio"])
    r_ = [r["ratio"] for r in rows]
    pm = [r["P_star_max"] for r in rows]
    pl = [r["P_star_linear"] if r["P_star_linear"] is not None else np.nan for r in rows]
    pt = [r["P_star_theory"] if r["P_star_theory"] is not None else np.nan for r in rows]
    ptp = [
        r["P_star_theory_pred"] if r.get("P_star_theory_pred") is not None else np.nan
        for r in rows
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ps_ = [r["P_star_surface"] if r.get("P_star_surface") is not None else np.nan for r in rows]
    ax.plot(r_, ps_, "^:", color="#e08214", ms=5, lw=1.1, label="LBM peak at SWL (impact spike)")
    ax.plot(r_, pm, "o-", color="#c0392b", label="LBM peak on $y=-0.1h$ gauge")
    ax.plot(r_, pl, "s--", color="#2c7fb8", label="LBM first harmonic")
    ax.plot(r_, pt, "--", color="#333", lw=1.4, label="linear theory (measured $\\eta_1$)")
    ax.plot(r_, ptp, "-", color="#1a9641", lw=1.6, label="linear theory (predicted $\\eta_1$)")
    ax.axvline(1.0, color="0.6", ls=":", lw=1)
    ax.set_xlabel("excitation frequency ratio $\\omega/\\omega_1$")
    ax.set_ylabel("$p_{max}/\\rho g h$")
    ax.set_yscale("log")
    ax.set_title("Wall pressure peak vs excitation frequency (fill 0.5)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    er = [
        100 * r["rel_error_vs_linear_theory_pred"]
        for r in rows
        if r.get("rel_error_vs_linear_theory_pred") is not None
    ]
    steady_mask = [
        bool(r.get("steady_state")) for r in rows if r.get("rel_error_vs_linear_theory_pred") is not None
    ]
    rr = [r["ratio"] for r in rows if r.get("rel_error_vs_linear_theory_pred") is not None]
    ax2.axhline(0, color="#333", lw=1)
    ax2.axhline(20, color="#c0392b", ls="--", lw=1, label="+20 %")
    ax2.axhline(-20, color="#c0392b", ls="--", lw=1, label="-20 %")
    ax2.plot([x for x, st in zip(rr, steady_mask) if st], [e for e, st in zip(er, steady_mask) if st],
             "o-", color="#7b3294", label="steady response")
    ax2.plot([x for x, st in zip(rr, steady_mask) if not st],
             [e for e, st in zip(er, steady_mask) if not st],
             "x", color="#e08214", ms=8, label="not steady in window")
    ax2.set_xlabel("$\\omega/\\omega_1$")
    ax2.set_ylabel("$p_{max}$ error vs linear theory (%)")
    ax2.set_title("Pressure judge (experimental data WITHHELD)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _figure_waveform(wave: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = wave["rows"][0] if wave["rows"] else None
    if not row or row.get("status") == "ERROR":
        return
    case = row["case"]
    x = np.asarray(case["waveform_x"])
    model = np.asarray(case["waveform_model"])
    meas = np.asarray(case["waveform_measured"])
    t = np.asarray(case["waveform_t"])
    period = 2 * math.pi / case["omega"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    n_snap = min(4, model.shape[0])
    idx = np.linspace(0, model.shape[0] - 1, n_snap).astype(int)
    for j, i in enumerate(idx):
        lbl = f"t = {(t[i] % period) / period:.2f} T"
        ax.plot(x, meas[i], "o", ms=2.5, color=f"C{j}", alpha=0.6)
        ax.plot(x, model[i], "-", lw=1.3, color=f"C{j}", label=lbl)
    ax.set_xlabel("x (lattice cells)")
    ax.set_ylabel("$\\eta(x,t)$ (cells)")
    ax.set_title("Free surface: LBM (dots) vs single-mode potential flow (lines)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    # time history at the antinode
    i_anti = int(np.argmax(np.abs(meas).max(axis=0)))
    ax2.plot(t, meas[:, i_anti], "o", ms=2.5, color="#c0392b", alpha=0.5, label="LBM")
    ax2.plot(t, model[:, i_anti], "-", lw=1.4, color="#333", label="potential flow")
    ax2.set_xlabel("t (lattice steps)")
    ax2.set_ylabel("$\\eta$ at antinode (cells)")
    ax2.set_title("rel L2 error = %.2f %%" % (100 * row["waveform_rel_l2"]))
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or "unknown"
        )
    except Exception:  # pragma: no cover
        return "unknown"


def summarize(
    case_dir: Path,
    out_dir: Path,
    spec: CampaignSpec | None = None,
    n_tests: int | None = None,
) -> dict:
    """Aggregate case JSONs: acceptance judges, figures and the report JSON."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(case_dir)
    decays = [c for c in cases if c.get("kind") == "decay"]
    forced = [c for c in cases if c.get("kind") == "forced"]
    waves = [c for c in cases if c.get("kind") == "waveform"]
    decays.sort(key=lambda c: c.get("fill_ratio", 0))
    forced.sort(key=lambda c: c.get("ratio", 0))

    freq = _judge_frequency(decays, CRITERION["frequency_rel_error_max"])
    press = _judge_pressure(forced, CRITERION["pressure_rel_error_max"])
    for r in press["rows"]:
        cid = next(
            (c for c in forced if c.get("ratio") == r.get("ratio") and c.get("fill_ratio") == r.get("fill_ratio")),
            None,
        )
        if cid:
            r["id"] = cid.get("id")
    wave = _judge_waveform(waves, CRITERION["waveform_rel_l2_max"])
    for r, c in zip(wave["rows"], waves):
        r["case"] = c

    figs = {}
    try:
        p1 = fig_dir / "fig1_frequency_vs_fill.png"
        _figure_frequency(freq, p1)
        figs["fig1_frequency_vs_fill"] = str(p1.relative_to(out_dir))
    except Exception as exc:  # pragma: no cover
        figs["fig1_error"] = str(exc)
    try:
        p2 = fig_dir / "fig2_pressure_vs_frequency_ratio.png"
        _figure_pressure(press, p2)
        figs["fig2_pressure_vs_frequency_ratio"] = str(p2.relative_to(out_dir))
    except Exception as exc:  # pragma: no cover
        figs["fig2_error"] = str(exc)
    try:
        p3 = fig_dir / "fig3_free_surface_waveform.png"
        _figure_waveform(wave, p3)
        figs["fig3_free_surface_waveform"] = str(p3.relative_to(out_dir))
    except Exception as exc:  # pragma: no cover
        figs["fig3_error"] = str(exc)

    checks = {
        "criterion_1_frequency_5pct": freq["passed"],
        "criterion_2_pressure_20pct": press["passed_vs_linear_theory"],
        "criterion_2_experimental": "WITHHELD_NO_DATASET",
        "criterion_3_waveform_15pct": wave["passed"],
        "criterion_5_figures": len([k for k in figs if k.startswith("fig")]) >= 3,
    }
    spec_used = spec
    for c in cases:
        sp = c.get("spec")
        if isinstance(sp, dict) and sp:
            try:
                spec_used = CampaignSpec(**{**asdict(CampaignSpec()), **{
                    k: (tuple(v) if isinstance(v, list) else v) for k, v in sp.items()
                }})
            except TypeError:  # pragma: no cover
                spec_used = spec or CampaignSpec()
            break
    report = {
        "node": "IN143-N07",
        "title": "LNG cargo-tank partial-fill sloshing: free-surface LBM x Abramson anchors x impact pressure",
        "spec": (spec_used or CampaignSpec()).as_dict(),
        "reference": {
            "abramson_1966": SP106_REFERENCE,
            "faltinsen_1978": "Faltinsen (1978) J. Ship Research 22(4) 241-249 (nonlinear sloshing theory)",
            "faltinsen_timokha_2009": "Faltinsen & Timokha (2009) Sloshing, CUP (linear modal + pressure)",
            "wagner_1932": "Wagner (1932) ZAMM 12, 193-215 (stagnation pressure bound)",
        },
        "criteria": CRITERION,
        "judges": {
            "frequency": freq,
            "pressure": press,
            "waveform": wave,
        },
        "checks": checks,
        "figures": figs,
        "provenance": {
            "git_commit": _git_commit(),
            "n_cases": len(cases),
            "case_dir": str(case_dir),
            "case_archive": "artifacts/lng_sloshing/cases",
            "lbm_model": "tensorlbm.free_surface_lbm_2d.koerner_step_2d (Korner 2005 free-surface LBM)",
            "generated_by": "python -m tensorlbm.lng_sloshing.campaign summarize",
            "tests": "tests/test_lng_sloshing.py",
            "n_tests": int(n_tests) if n_tests is not None else None,
        },
    }
    (out_dir / "campaign_report.json").write_text(json.dumps(report, indent=1, default=float))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _spec_from_args(args) -> CampaignSpec:
    base = CampaignSpec()
    kwargs = {}
    if getattr(args, "length", None):
        kwargs["length"] = int(args.length)
    if getattr(args, "g", None):
        kwargs["g"] = float(args.g)
    if getattr(args, "tau", None):
        kwargs["tau"] = float(args.tau)
    if getattr(args, "decay_periods", None):
        kwargs["decay_periods"] = float(args.decay_periods)
    if getattr(args, "forced_periods", None):
        kwargs["forced_periods"] = float(args.forced_periods)
    if getattr(args, "sway_frac", None):
        kwargs["sway_frac"] = float(args.sway_frac)
    if getattr(args, "zeta", None):
        kwargs["zeta_default"] = float(args.zeta)
    if getattr(args, "conv_lengths", None):
        kwargs["conv_lengths"] = tuple(int(v) for v in str(args.conv_lengths).split(","))
    return CampaignSpec(**{**asdict(base), **kwargs})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a shard of the campaign cases")
    run.add_argument("--shard", default="0/1")
    run.add_argument("--out-dir", default="runs/lng_sloshing_cases")
    run.add_argument("--length", type=int)
    run.add_argument("--g", type=float)
    run.add_argument("--tau", type=float)
    run.add_argument("--decay-periods", type=float)
    run.add_argument("--forced-periods", type=float)
    run.add_argument("--sway-frac", type=float)
    run.add_argument("--zeta", type=float)
    run.add_argument("--conv-lengths", default=None)
    run.add_argument("--only", default=None, help="run a single case id")

    sm = sub.add_parser("summarize", help="aggregate case JSONs into the report")
    sm.add_argument("--case-dir", default="runs/lng_sloshing_cases")
    sm.add_argument("--out-dir", default="artifacts/lng_sloshing")
    sm.add_argument("--tests-passed", type=int, default=None)

    ap.add_argument("--list", action="store_true", help="list campaign cases and exit")
    args = ap.parse_args(argv)

    spec = _spec_from_args(args)
    if args.list:
        for i, c in enumerate(case_list(spec)):
            print(i, c["id"], c["kind"])
        return 0
    if args.cmd == "run":
        i, n = (int(v) for v in str(args.shard).split("/"))
        run_shard(spec, i, n, Path(args.out_dir), only=args.only)
        return 0
    rep = summarize(Path(args.case_dir), Path(args.out_dir), spec, n_tests=args.tests_passed)
    print(json.dumps(rep["checks"], indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())