#!/usr/bin/env python
"""p485w1-n05 — free-surface LBM discrete dispersion: closed form, phase budget, calibration.

Campaign node ``p485w1-n05-lbm-free-surface-dispersion``.  Five stages, all
machine-checkable:

``--theory``    exact lattice modes, closed-form gravity-wave budget, estimator bias
``--case NAME`` run one periodic linear-wave case and store its JSON
``--shard i/n`` run the i-th of n equal partitions of the full case list
``--analyze``   aggregate the stored cases into ``summary.json`` / ``tables.md`` / figures
``--all``       theory + shard 0/1 + analyze (single-process convenience)

Artifacts land in ``artifacts/p485w1_n05_dispersion/``.

Case naming follows ``<base>_<policy>[_sig<SIGMA>]`` where ``policy`` is the
interface-pressure policy of the phase-calibration operator (``const`` = legacy
Körner/waLBerla reconstruction, ``subcell`` = calibration operator on).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm import free_surface_dispersion as fs  # noqa: E402

ART = ROOT / "artifacts" / "p485w1_n05_dispersion"
C_TARGET = 0.06          # lattice Mach ~0.10: below the O(Ma^2) compressibility scale
TAU = 0.51               # nu = (tau-0.5)/3 = 3.33e-3
N_PERIODS = 5.0
WARMUP = 1.5
STEEPNESS = 0.04         # eps = k a
SIGMA_CAL = 1.5e-3       # first Laplace calibration coefficient (over-corrects, kept as a lever)
SIGMA_CAL2 = 5.5e-4      # refined coefficient: sigma = 2 A c_target^2 / G (G measured)
G_FIXED = C_TARGET ** 2 * (1.5 / 20.0) / math.tanh(1.5)   # gravity of the kh=1.5 reference case


BASE = {
    "policy": "const",
    "sigma_cal": 0.0,
    "n_periods": N_PERIODS,
    "warmup": WARMUP,
    "c_target": C_TARGET,
    "tau": TAU,
}


def case_list() -> list[dict]:
    """The full diagnostic sweep.

    Roles:

    ``depth_fan``      kh = 0.7 … 3.0 at fixed h: discriminates a wave-number
                       (``k dx``) error from a depth/inertia (``dx/h``) error;
    ``resolution_fan`` kh = 1.5 at h = 20, 10, 5: the ``1/h`` scaling law;
    ``tau_fan``        viscosity lever (is the deficit dissipative in origin?);
    ``amp_fan``        amplitude lever (nonlinear contamination?);
    ``mach_fan``       lattice-Mach lever (compressibility?);
    ``window_fan``     integration-window lever (start-up transient?);
    ``subcell``        the sub-cell interface-pressure operator variant;
    ``sigma_cal``      the Laplace (O(k^2)) calibration operator;
    ``decay_long``     long-horizon amplitude decay for the 100-period criterion.
    """
    cases: list[dict] = []
    for kh in (0.7, 1.0, 1.5, 2.0, 3.0):
        cases.append({**BASE, "name": f"kh{kh:.1f}".replace(".", "") + "_const",
                      "h": 20.0, "kh": kh, "amplitude": 0.25, "role": "depth_fan"})
    for h in (10.0, 5.0):
        cases.append({**BASE, "name": f"res{h:g}_const".replace(".", ""),
                      "h": h, "kh": 1.5, "amplitude": 0.25, "role": "resolution_fan"})
    for tau, tag in ((0.5025, "tau05025"), (0.53, "tau053")):
        cases.append({**BASE, "name": f"kh15_{tag}", "h": 20.0, "kh": 1.5,
                      "amplitude": 0.25, "role": "tau_fan", "tau": tau})
    for a, tag in ((0.05, "a005"), (0.5, "a050")):
        cases.append({**BASE, "name": f"kh15_{tag}", "h": 20.0, "kh": 1.5,
                      "amplitude": a, "role": "amp_fan"})
    cases.append({**BASE, "name": "kh15_mach004", "h": 20.0, "kh": 1.5,
                  "amplitude": 0.25, "role": "mach_fan", "c_target": 0.04})
    cases.append({**BASE, "name": "kh15_window12", "h": 20.0, "kh": 1.5,
                  "amplitude": 0.25, "role": "window_fan",
                  "n_periods": 12.0, "warmup": 4.0})
    for kh, h, tag in ((1.5, 20.0, "kh15_subcell"), (3.0, 20.0, "kh30_subcell"),
                       (1.5, 5.0, "res5_subcell")):
        cases.append({**BASE, "name": tag, "h": h, "kh": kh, "amplitude": 0.25,
                      "role": "subcell", "policy": "subcell"})
    for sig, tag in ((2e-3, "sig2e3"), (4e-3, "sig4e3")):
        cases.append({**BASE, "name": f"kh15_{tag}", "h": 20.0, "kh": 1.5,
                      "amplitude": 0.25, "role": "sigma_cal", "sigma_cal": sig})
    # calibrated family: one constant Laplace coefficient across the whole k range
    # (the closed form sigma = 2 A c_target^2 is k- and h-independent, see report)
    for kh in (0.7, 1.0, 1.5, 2.0, 3.0):
        cases.append({**BASE, "name": f"kh{kh:.1f}".replace(".", "") + "_sigcal",
                      "h": 20.0, "kh": kh, "amplitude": 0.25, "role": "sigma_fan",
                      "sigma_cal": SIGMA_CAL})
    # group-speed fan: *fixed* g and h, k varied (dc/dk must be taken at fixed g)
    for kh in (1.35, 1.5, 1.65):
        cases.append({**BASE, "name": f"grp_kh{str(kh).replace('.', '')}", "h": 20.0, "kh": kh,
                      "amplitude": 0.25, "role": "group_fan", "g_override": G_FIXED})
    for kh in (0.7, 1.0, 1.5, 2.0, 3.0):
        cases.append({**BASE, "name": f"kh{kh:.1f}".replace(".", "") + "_sigcal2",
                      "h": 20.0, "kh": kh, "amplitude": 0.25, "role": "sigma_fan2",
                      "sigma_cal": SIGMA_CAL2})
    for alpha, h, tag in ((0.5, 20.0, "kh15_alpha05"), (1.0, 20.0, "kh15_alpha10"),
                          (1.0, 5.0, "res5_alpha10")):
        cases.append({**BASE, "name": tag, "h": h, "kh": 1.5, "amplitude": 0.25,
                      "role": "force_cal", "interface_force_alpha": alpha})
    cases.append({**BASE, "name": "kh15_long_const", "h": 20.0, "kh": 1.5,
                  "amplitude": 0.25, "role": "decay_long",
                  "n_periods": 20.0, "warmup": 3.0})
    return cases


def run_case(spec: dict, verbose: bool = True) -> dict:
    case = fs.LinearWaveCase(
        h=spec["h"],
        kh=spec["kh"],
        amplitude=spec["amplitude"],
        c_target=spec.get("c_target", C_TARGET),
        tau=spec.get("tau", TAU),
        wavelengths=1,
        headroom=10,
        n_periods=spec.get("n_periods", N_PERIODS),
        warmup_periods=spec.get("warmup", WARMUP),
        record_per_period=32,
        interface_pressure_policy=spec["policy"],
        interface_force_alpha=spec.get("interface_force_alpha", 0.0),
        g_override=spec.get("g_override"),
        sigma_cal=spec["sigma_cal"],
    )
    t0 = time.time()
    result = fs.run_linear_wave(case, verbose=verbose)
    result["name"] = spec["name"]
    result["role"] = spec["role"]
    result["wall_s"] = float(time.time() - t0)
    out = ART / f"case_{spec['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if k != "field"}
    out.write_text(json.dumps(payload, indent=2))
    (ART / f"field_{spec['name']}.json").write_text(json.dumps(result["field"]))
    c = result["discrete"]
    print(
        f"[case] {spec['name']:>22s} k={c['k_measured']:.5f} "
        f"c={c['c_measured']:.6f} c_cont={c['c_continuum']:.6f} "
        f"err={100 * c['c_rel_error']:+.3f}%  a={result['amplitude']['final_last_period']:.4f} "
        f"({result['wall_s']:.0f}s)",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# theory stage
# ---------------------------------------------------------------------------
def stage_theory() -> dict:
    ART.mkdir(parents=True, exist_ok=True)
    modes = [
        {**fs.lattice_phase_speed_error(k, TAU), "branch": "acoustic"}
        for k in (0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8)
    ]
    biases = []
    for r in np.linspace(0.0, 0.25, 26):
        row = fs.synthetic_standing_wave_bias(0.075, float(r))
        row["closed_form"] = fs.standing_wave_estimator_bias(0.075, float(r))
        biases.append(row)
    # what the wave-tank campaign reported (benchmarks/verified/wave_tank/README.md §4)
    wave_tank = [
        {"case": "kh0.7_eps0.02", "kh": 0.7, "reported_error": 0.0845, "reported_kr": 0.1092},
        {"case": "kh0.7_eps0.08", "kh": 0.7, "reported_error": 0.0702, "reported_kr": 0.0517},
        {"case": "kh1.5_eps0.02", "kh": 1.5, "reported_error": 0.1345, "reported_kr": 0.1165},
        {"case": "kh1.5_eps0.08", "kh": 1.5, "reported_error": 0.1904, "reported_kr": 0.1387},
        {"case": "kh3.0_eps0.02", "kh": 3.0, "reported_error": 0.1695, "reported_kr": 0.1392},
        {"case": "kh3.0_eps0.08", "kh": 3.0, "reported_error": 0.1759, "reported_kr": 0.1493},
    ]
    budgets = []
    for kh in (0.7, 1.0, 1.5, 2.0, 3.0):
        h = 20.0
        k = kh / h
        g = C_TARGET**2 * k / math.tanh(kh)
        budgets.append(fs.phase_error_budget(k, h, g, TAU).as_dict())
    payload = {
        "lattice_modes": modes,
        "estimator_bias": biases,
        "wave_tank_reported": wave_tank,
        "budget": budgets,
        "meta": {"tau": TAU, "c_target": C_TARGET},
    }
    (ART / "theory.json").write_text(json.dumps(payload, indent=2))
    print("[theory] lattice acoustic mode relative error at k=0.2: "
          f"{100 * fs.lattice_phase_speed_error(0.2, TAU)['rel_error']:+.4f}%")
    print("[theory] estimator bias for r=0.10: "
          f"{100 * fs.standing_wave_estimator_bias(0.075, 0.1)['bias_min']:+.1f}% … "
          f"{100 * fs.standing_wave_estimator_bias(0.075, 0.1)['bias_max']:+.1f}%")
    return payload


# ---------------------------------------------------------------------------
# analysis stage
# ---------------------------------------------------------------------------
def _load_cases() -> dict[str, dict]:
    cases: dict[str, dict] = {}
    for path in sorted(ART.glob("case_*.json")):
        data = json.loads(path.read_text())
        cases[data["name"]] = data
    return cases


def stage_analyze() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = _load_cases()
    if not cases:
        print("[analyze] no cases found", file=sys.stderr)
        return 1
    theory = json.loads((ART / "theory.json").read_text()) if (ART / "theory.json").exists() else {}

    rows = []
    for name, data in cases.items():
        c = data["discrete"]
        cfg = data["case"]
        k_cont = cfg["k"]
        omega_cont = c["c_continuum"] * k_cont
        row = {
            "name": name,
            "role": data.get("role", ""),
            "policy": cfg["interface_pressure_policy"],
            "sigma_cal": cfg["sigma_cal"],
            "tau": cfg["tau"],
            "kh": cfg["kh"],
            "h": cfg["h"],
            "k": c["k_measured"],
            "k_true": k_cont,
            "cells_per_wavelength": cfg["cells_per_wavelength"],
            "c_measured": c["c_measured"],
            "c_continuum": c["c_continuum"],
            "c_rel_error": c["c_rel_error"],
            "amplitude": cfg["amplitude"],
            "steepness": cfg["steepness"],
            "reflection_ratio": c.get("reflection_ratio_residual", float("nan")),
            "fit_explained": c.get("fit_explained_fraction", float("nan")),
            "amp_final": data["amplitude"]["final_last_period"],
            "gamma": data["amplitude"]["gamma_measured"],
            "gamma_theory": data["amplitude"]["gamma_theory_viscous"],
            "mass_drift": data["mass"]["relative_drift_total"],
            "wall_s": data["wall_s"],
            "c_predicted": data["budget"]["c_predicted"],
        }
        # criterion (c) bookkeeping: fraction of amplitude lost over 100 periods
        omega_meas = c["omega_measured"]
        if row["gamma"] == row["gamma"] and omega_meas > 0:
            row["decay_100T"] = float(1.0 - math.exp(-row["gamma"] * 100.0 * 2.0 * math.pi
                                                     / omega_meas))
        else:
            row["decay_100T"] = float("nan")
        # the tank estimator, replayed on the same solver field
        field_path = ART / f"field_{name}.json"
        if field_path.exists():
            field = json.loads(field_path.read_text())
            eta = np.asarray(field["eta"], dtype=float)
            tt = np.asarray(field["t"], dtype=float)
            xx = np.arange(eta.shape[1], dtype=float)
            tp = fs.two_point_phase_speed(eta, tt, xx, omega_cont)
            row["two_point_rel_error_min"] = tp["rel_error_min"]
            row["two_point_rel_error_max"] = tp["rel_error_max"]
            row["two_point_rel_error_span"] = tp["rel_error_span"]
        rows.append(row)
    rows.sort(key=lambda r: (r["role"], r["name"]))

    phase_rows = [r for r in rows if r["role"] in ("depth_fan", "resolution_fan")]

    def _worst(sel):
        vals = [abs(r["c_rel_error"]) for r in sel if r["c_rel_error"] == r["c_rel_error"]]
        return max(vals) if vals else float("nan")

    worst_all = _worst(phase_rows)
    resolved = [
        r for r in phase_rows
        if r["cells_per_wavelength"] >= 40.0 and r["h"] >= 20.0
    ]
    worst_resolved = _worst(resolved)
    worst_twopoint = max(
        (max(abs(r.get("two_point_rel_error_min", 0.0)), abs(r.get("two_point_rel_error_max", 0.0)))
         for r in phase_rows),
        default=float("nan"),
    )

    mechanism = {}
    # -- group speed: fixed-g / fixed-h k triplet (dc/dk is meaningless in the
    #    depth fan, where g was tuned per case to hold c_target constant) -----
    group = []
    gfan = sorted([r for r in rows if r["role"] == "group_fan"], key=lambda r: r["k"])
    if len(gfan) >= 3:
        lo, mid, hi = gfan[0], gfan[1], gfan[2]
        dc = (hi["c_measured"] - lo["c_measured"]) / (hi["k"] - lo["k"])
        c_g = mid["c_measured"] + mid["k"] * dc
        c_g_cont = fs.continuous_group_speed(mid["k"], mid["h"], G_FIXED)
        group.append({
            "policy": "const",
            "k": mid["k"],
            "kh": mid["kh"],
            "c_group_measured": c_g,
            "c_group_continuum": c_g_cont,
            "c_phase_measured": mid["c_measured"],
            "c_phase_continuum": mid["c_continuum"],
            "rel_error": c_g / c_g_cont - 1.0,
            "g_fixed": G_FIXED,
            "k_triplet": [lo["k"], mid["k"], hi["k"]],
        })
    # analytic discrete group speed from the fitted two-term error law
    if mechanism.get("two_term_law"):
        A = mechanism["two_term_law"]["A_per_k"]
        B = mechanism["two_term_law"]["B_over_h"]
        h_ref = 20.0
        analytic = []
        for kk in (0.05, 0.075, 0.1, 0.15):
            c0 = fs.continuous_phase_speed(kk, h_ref, G_FIXED)
            c_n = c0 * (1.0 - A * kk - B / h_ref)
            cg0 = fs.continuous_group_speed(kk, h_ref, G_FIXED)
            analytic.append({"k": kk, "c_phase_continuous": c0, "c_phase_discrete": c_n,
                             "n_slope_cg_over_c": cg0 / c0})
        mechanism["analytic_group_speed_note"] = {
            "law": "c_discrete = c_cont (1 - A k - B/h)",
            "rows": analytic,
        }
    worst_group = max((abs(r["rel_error"]) for r in group), default=float("nan"))

    # -- mechanism discrimination: power law in (k dx) -----------------------
    fan = sorted([r for r in phase_rows if r["role"] == "depth_fan"], key=lambda r: r["k"])
    if len(fan) >= 3:
        kdx = np.array([r["k"] * 1.0 for r in fan])
        err = np.array([abs(r["c_rel_error"]) for r in fan])
        ok = err > 0
        if ok.sum() >= 3:
            slope, intercept = np.polyfit(np.log(kdx[ok]), np.log(err[ok]), 1)
            mechanism["power_law_exponent_in_k_dx"] = float(slope)
            mechanism["power_law_coefficient"] = float(math.exp(intercept))
    for role in ("tau_fan", "amp_fan", "mach_fan", "window_fan"):
        sel = [r for r in rows if r["role"] == role]
        if sel:
            mechanism[role] = [
                {"name": r["name"], "tau": r["tau"], "amplitude": r["amplitude"],
                 "c_rel_error": r["c_rel_error"],
                 "reflection_ratio": r["reflection_ratio"]} for r in sel
            ]
    # two-term law: err = -(A k + B / h) over the depth + resolution fans
    law_rows = [r for r in phase_rows if r["role"] in ("depth_fan", "resolution_fan")]
    if len(law_rows) >= 4:
        M = np.array([[r["k"], 1.0 / r["h"]] for r in law_rows], dtype=float)
        y = np.array([-r["c_rel_error"] for r in law_rows], dtype=float)
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        resid = y - M @ coef
        mechanism["two_term_law"] = {
            "A_per_k": float(coef[0]),
            "B_over_h": float(coef[1]),
            "rms_residual": float(np.sqrt(np.mean(resid**2))),
            "n": int(len(law_rows)),
            "form": "-err = A*k + B/h",
            "per_case": [
                {"name": r["name"], "k": r["k"], "h": r["h"],
                 "rel_error": r["c_rel_error"], "model": float(-(coef[0] * r["k"] + coef[1] / r["h"]))}
                for r in law_rows
            ],
            "sigma_cal_predicted": float(2.0 * coef[0] * (C_TARGET ** 2)),
        }
    # effective Laplace gain: measured phase shift per unit nominal shift
    base_kh15 = next((r for r in rows if r["name"] == "kh15_const"), None)
    s2 = next((r for r in rows if r["name"] == "kh15_sig2e3"), None)
    s4 = next((r for r in rows if r["name"] == "kh15_sig4e3"), None)
    if base_kh15 and s2 and s4:
        k15 = base_kh15["k"]
        nom = lambda sig: sig * k15 / (2.0 * C_TARGET ** 2)  # noqa: E731
        gain = ((s2["c_rel_error"] - base_kh15["c_rel_error"]) / nom(2e-3),
                (s4["c_rel_error"] - base_kh15["c_rel_error"]) / nom(4e-3))
        mechanism["laplace_effective_gain"] = {
            "gain_sigma_2e-3": float(gain[0]),
            "gain_sigma_4e-3": float(gain[1]),
            "gain_mean": float(0.5 * (gain[0] + gain[1])),
            "nominal_effect_sigma_2e-3": float(nom(2e-3)),
            "measured_effect_sigma_2e-3": float(s2["c_rel_error"] - base_kh15["c_rel_error"]),
        }
        if mechanism.get("two_term_law"):
            A = mechanism["two_term_law"]["A_per_k"]
            mechanism["two_term_law"]["sigma_cal_with_gain"] = float(
                2.0 * A * C_TARGET ** 2 / max(0.5 * (gain[0] + gain[1]), 1e-9)
            )
    mechanism["baseline_kh15"] = next(
        (r["c_rel_error"] for r in rows if r["name"] == "kh15_const"), None
    )

    calib_rows = [r for r in rows if r["role"] in ("sigma_cal", "subcell", "force_cal")]
    sig_fan = [r for r in rows if r["role"] == "sigma_fan"]
    sig_fan2 = [r for r in rows if r["role"] == "sigma_fan2"]
    worst_sigfan = _worst(sig_fan)
    worst_sigfan2 = _worst(sig_fan2)
    worst_calib = _worst(calib_rows)
    decay_cases = [r for r in rows if r["role"] == "decay_long" or r["role"] == "depth_fan"]
    decay_worst = max((r["decay_100T"] for r in decay_cases if r["decay_100T"] == r["decay_100T"]),
                      default=float("nan"))

    summary = {
        "node": "p485w1-n05-lbm-free-surface-dispersion",
        "repo": "TensorLBM",
        "theory": {
            "estimator_bias_r0.1_min": (
                theory["estimator_bias"][10]["bias_two_point_min"] if theory else None
            ),
            "estimator_bias_r0.1_max": (
                theory["estimator_bias"][10]["bias_two_point_max"] if theory else None
            ),
            "lattice_acoustic_rel_error": [
                {"k": m["k"], "rel_error": m["rel_error"]} for m in theory.get("lattice_modes", [])
            ],
        },
        "cases": rows,
        "group_speed": group,
        "mechanism": mechanism,
        "criteria": {
            "a_phase_error_under_3pct": {
                "worst_abs_error_all": worst_all,
                "worst_abs_error_cells_per_wavelength_ge_40": worst_resolved,
                "worst_abs_error_two_point_estimator": worst_twopoint,
                "worst_abs_error_calibration_family": worst_calib,
                "worst_abs_error_sigma_calibrated_fan": worst_sigfan,
                "pass_sigma_calibrated_fan": bool(
                    worst_sigfan == worst_sigfan and worst_sigfan < 0.03
                ),
                "worst_abs_error_sigma_calibrated_fan2": worst_sigfan2,
                "pass_sigma_calibrated_fan2": bool(
                    worst_sigfan2 == worst_sigfan2 and worst_sigfan2 < 0.03
                ),
                "threshold": 0.03,
                "pass_all": bool(worst_all == worst_all and worst_all < 0.03),
                "pass_resolved": bool(worst_resolved == worst_resolved and worst_resolved < 0.03),
            },
            "b_group_error_under_5pct": {
                "worst_abs_error": worst_group,
                "threshold": 0.05,
                "pass": bool(worst_group == worst_group and worst_group < 0.05),
            },
            "c_amplitude_decay_100T_under_1pct": {
                "worst_fraction_lost": decay_worst,
                "threshold": 0.01,
                "pass": bool(decay_worst == decay_worst and decay_worst < 0.01),
            },
        },
        "wall_s_total": float(sum(r["wall_s"] for r in rows)),
    }
    (ART / "summary.json").write_text(json.dumps(summary, indent=2))

    # -- tables -------------------------------------------------------------
    lines = [
        "# p485w1-n05 — free-surface LBM discrete dispersion (machine tables)",
        "",
        "Generated by `scripts/p485w1_n05_dispersion.py --analyze`; every number is",
        "programmatic output, no hand transcription.",
        "",
        "`c_rel_error` is the intrinsic error of the *two-wave least-squares* fit;",
        "`two-point` columns replay the wave-tank estimator on the same field.",
        "",
        "## Phase speed (all roles)",
        "",
        "| case | role | policy | kh | h | cells/lambda | tau | a | c meas | c closed form "
        "| rel. err | two-point err [min .. max] | r |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        tp = (
            f"{100 * r['two_point_rel_error_min']:+.1f} … {100 * r['two_point_rel_error_max']:+.1f} %"
            if "two_point_rel_error_min" in r else "n/a"
        )
        lines.append(
            f"| {r['name']} | {r['role']} | {r['policy']} | {r['kh']:.2f} | {r['h']:.0f} | "
            f"{r['cells_per_wavelength']:.0f} | {r['tau']:.4f} | {r['amplitude']:.2f} | "
            f"{r['c_measured']:.6f} | {r['c_continuum']:.6f} | {100 * r['c_rel_error']:+.3f} % | "
            f"{tp} | {r['reflection_ratio']:.3f} |"
        )
    lines += ["", "## Group speed", "", "| policy | k | c_g measured | c_g closed form | rel. error |",
              "|---|---|---|---|---|"]
    for r in group:
        lines.append(
            f"| {r['policy']} | {r['k']:.4f} | {r['c_group_measured']:.6f} | "
            f"{r['c_group_continuum']:.6f} | {100 * r['rel_error']:+.3f} % |"
        )
    lines += ["", "## Amplitude decay and mass ledger", "",
              "| case | a final | gamma measured | 2 nu k^2 | lost in 100 T | mass drift |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        if r["gamma"] != r["gamma"]:
            continue
        lines.append(
            f"| {r['name']} | {r['amp_final']:.4f} | {r['gamma']:.3e} | {r['gamma_theory']:.3e} | "
            f"{100 * r['decay_100T']:.2f} % | {r['mass_drift']:+.2e} |"
        )
    lines += ["", "## Mechanism discrimination", "", "```json",
              json.dumps(mechanism, indent=2), "```", "", "## Criteria", "", "```json",
              json.dumps(summary["criteria"], indent=2), "```", ""]
    (ART / "tables.md").write_text("\n".join(lines))

    _figs(plt, rows, theory, summary, group)
    print("[analyze] wrote", ART / "summary.json")
    print(json.dumps(summary["criteria"], indent=2))
    return 0


def _figs(plt, rows, theory, summary, group) -> None:
    """Four report figures, all from the machine tables (no hand plotting)."""
    # fig 1: intrinsic phase error vs cells per wavelength, with the tank estimator
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    styles = {"const": ("o", "free-surface LBM, legacy interface BC"),
              "subcell": ("s", "sub-cell interface pressure (calibration variant)")}
    for pol, (marker, label) in styles.items():
        sel = sorted([r for r in rows if r["policy"] == pol], key=lambda r: -r["cells_per_wavelength"])
        if sel:
            ax.plot([r["cells_per_wavelength"] for r in sel],
                    [100 * r["c_rel_error"] for r in sel], marker=marker, ls="-", lw=1.0,
                    label=label)
    calib = [r for r in rows if r["role"] == "sigma_cal"]
    if calib:
        ax.plot([r["cells_per_wavelength"] for r in calib],
                [100 * r["c_rel_error"] for r in calib], "d", label="with Laplace calibration")
    if theory.get("budget"):
        ks = [b["k"] for b in theory["budget"]]
        ax.plot([2 * math.pi / max(k, 1e-12) for k in ks],
                [100 * b["e_total"] for b in theory["budget"]], "k--", lw=0.9,
                label="closed-form budget $e_{lattice}+e_{visc}$")
    ax.axhline(3.0, color="r", ls="--", lw=0.8, label="+/-3 % acceptance")
    ax.axhline(-3.0, color="r", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("cells per wavelength  $2\\pi/(k\\,\\Delta x)$")
    ax.set_ylabel("phase-speed error  $c_{LBM}/c_{theory}-1$  [%]")
    ax.set_title("p485w1-n05: intrinsic dispersion of the periodic free-surface rig")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(ART / "fig1_phase_error.png", dpi=150)
    plt.close(fig)

    # fig 2: estimator bias vs residual reflection, with the measured cases
    if theory.get("estimator_bias"):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        rs = [b["reflection"] for b in theory["estimator_bias"]]
        ax.plot(rs, [100 * b["bias_two_point_max"] for b in theory["estimator_bias"]], "o-",
                ms=3, label="two-point, ds = 0.17$\\lambda$ (max over stations)")
        ax.plot(rs, [100 * b["bias_two_point_min"] for b in theory["estimator_bias"]], "o-",
                ms=3, label="two-point, ds = 0.17$\\lambda$ (min)")
        ax.plot(rs, [100 * b["closed_form"]["bias_max"] for b in theory["estimator_bias"]],
                "k--", lw=0.9, label="closed form, $ds\\to0$ (max)")
        ax.plot(rs, [100 * b["closed_form"]["bias_min"] for b in theory["estimator_bias"]],
                "k--", lw=0.9, label="closed form, $ds\\to0$ (min)")
        for wt in theory.get("wave_tank_reported", []):
            ax.plot(wt["reported_kr"], 100 * wt["reported_error"], "r^", ms=7)
        ax.plot([], [], "r^", label="wave tank reported (KR, error)")
        for r in rows:
            if r.get("two_point_rel_error_max", 0) == 0:
                continue
            ax.plot([r["reflection_ratio"]] * 2,
                    [100 * r["two_point_rel_error_min"], 100 * r["two_point_rel_error_max"]],
                    "g|", ms=6, alpha=0.8)
        ax.plot([], [], "g|", label="this work: replay of the tank estimator")
        ax.axhline(3.0, color="k", ls=":", lw=0.8)
        ax.axhline(-3.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("residual reflection $r = |B|/|A|$")
        ax.set_ylabel("apparent phase-speed error [%]")
        ax.set_title("Why 7-19 % appeared: local-slope estimator gain")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
        fig.tight_layout()
        fig.savefig(ART / "fig2_estimator_bias.png", dpi=150)
        plt.close(fig)

    # fig 3: waveform + amplitude history of the long run
    for r in [x for x in rows if x["role"] == "decay_long"]:
        field = ART / f"field_{r['name']}.json"
        if not field.exists():
            continue
        data = json.loads(field.read_text())
        eta = np.asarray(data["eta"])
        fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.0))
        axs[0].plot(eta[0], lw=1.2, label="first analysed period")
        axs[0].plot(eta[-1], lw=1.2, ls="--", label="last analysed period")
        axs[0].set_xlabel("x [cells]")
        axs[0].set_ylabel("free-surface elevation $\\eta$ [cells]")
        axs[0].set_title(f"waveform preservation: {r['name']}")
        axs[0].legend(fontsize=8)
        axs[0].grid(alpha=0.3)
        axs[1].semilogy(np.arange(eta.shape[0]), np.abs(eta).max(axis=1), lw=1.2)
        axs[1].set_xlabel("record index")
        axs[1].set_ylabel("max$|\\eta|$ [cells]")
        axs[1].set_title(f"amplitude history, gamma = {r['gamma']:.2e} /step "
                         f"(2 nu k^2 = {r['gamma_theory']:.2e})")
        axs[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ART / f"fig3_waveform_{r['name']}.png", dpi=150)
        plt.close(fig)
        break

    # fig 4: error budget breakdown + group-speed comparison
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.0))
    if theory.get("budget"):
        khs = [b["k"] * b["h"] for b in theory["budget"]]
        for key, lab in (("e_lattice", "lattice $O((k\\Delta x)^2)$"),
                         ("e_viscous", "viscous"),
                         ("e_total", "total")):
            axs[0].plot(khs, [100 * b[key] for b in theory["budget"]], "o-", label=lab)
        axs[0].set_xlabel("$kh$")
        axs[0].set_ylabel("phase-speed error [%]")
        axs[0].set_title("closed-form phase-error budget")
        axs[0].grid(alpha=0.3)
        axs[0].legend(fontsize=8)
    if group:
        ks = [g["k"] for g in group]
        axs[1].plot(ks, [g["c_group_measured"] for g in group], "o-", label="$c_g$ measured")
        axs[1].plot(ks, [g["c_group_continuum"] for g in group], "k--", label="$c_g$ closed form")
        axs[1].set_xlabel("$k\\,\\Delta x$")
        axs[1].set_ylabel("group speed [cells/step]")
        axs[1].set_title("group speed from central differences of $c_{LBM}(k)$")
        axs[1].grid(alpha=0.3)
        axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ART / "fig4_budget_group.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theory", action="store_true")
    ap.add_argument("--case", default=None, help="run one named case")
    ap.add_argument("--shard", default=None, help="i/n partition of the case list")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TL_THREADS", "4")))
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(max(args.threads, 1))

    if args.list:
        for spec in case_list():
            print(spec["name"])
        return 0
    if args.theory or args.all:
        stage_theory()
    if args.case:
        specs = {s["name"]: s for s in case_list()}
        if args.case not in specs:
            print(f"unknown case {args.case!r}", file=sys.stderr)
            return 2
        run_case(specs[args.case])
    if args.shard or args.all:
        i, n = (0, 1)
        if args.shard:
            i, n = (int(v) for v in args.shard.replace(":", "/").split("/"))
        specs = case_list()
        mine = [s for idx, s in enumerate(specs) if idx % n == i]
        print(f"[shard {i}/{n}] {len(mine)} cases", flush=True)
        for spec in mine:
            out = ART / f"case_{spec['name']}.json"
            if out.exists():
                print(f"[shard {i}/{n}] skip {spec['name']} (done)", flush=True)
                continue
            run_case(spec)
    if args.analyze or args.all:
        return stage_analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())