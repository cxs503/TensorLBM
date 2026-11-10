#!/usr/bin/env python
"""Campaign node p367seed-n08: LBM transient solve of propeller-blade ice milling.

Usage
-----
    python scripts/p367seed_n08_propeller_ice_milling.py <case>
    python scripts/p367seed_n08_propeller_ice_milling.py assemble

``<case>`` runs one simulation and stores ``artifacts/p367seed_n08/case_<name>.npz``
(raw torque/energy histories + field snapshots) plus ``case_<name>.json``
(config and per-case diagnostics).  ``assemble`` post-processes every stored case
into ``metrics.json`` and the report figures.  Splitting the work this way keeps
each invocation short enough to run inside the harness shell budget, and makes
every number in the report traceable to a stored artifact.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("TL_THREADS", "1")))

from tensorlbm.propeller_ice_milling import (  # noqa: E402
    MillingConfig, Iso19906Ice, run_milling, torque_coefficient,
    richardson_order, dominant_frequency, count_pulses, fluctuation_rms,
    mass_ledger_residual, energy_ledger_residual, iso19906_milling_torque,
    empirical_pulse_train, physical_scaling, smooth_signal,
)

OUT = Path("artifacts/p367seed_n08")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

BASE = dict(
    n_blades=2,
    blade_radius_frac=0.40,
    hub_radius_frac=0.32,
    blade_thickness_frac=0.12,
    gap_frac=0.15,
    ice_sector_width_deg=60.0,
    ice_sector_center_deg=0.0,
    wall_cells=2,
    tip_ma=0.18,
    reynolds=120.0,
)

#: case name -> (grid, overrides)
CASES = {
    "conv64": (64, dict(revolutions=0.75, idle_revolutions=0.25)),
    "conv80": (80, dict(revolutions=0.75, idle_revolutions=0.25)),
    "conv96": (96, dict(revolutions=0.75, idle_revolutions=0.25)),
    "perio64": (64, dict(revolutions=1.5, idle_revolutions=0.25)),
    "noice64": (64, dict(revolutions=1.0, idle_revolutions=0.25, with_ice=False)),
    "base64": (64, dict(revolutions=1.25, idle_revolutions=0.25, snapshots=True)),
}
GRIDS = (64, 80, 96)
N_CASE = 64


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------


def run_case(name: str) -> dict:
    n, over = CASES[name]
    cfg = MillingConfig(n=n, **{**BASE, **{k: v for k, v in over.items() if k != "snapshots"}})
    snaps = ()
    if over.get("snapshots"):
        p = int(cfg.period_steps)
        snaps = (int(0.94 * p), int(1.00 * p), int(1.06 * p))
    t0 = time.time()
    hist = run_milling(cfg, snapshot_steps=snaps)
    wall = time.time() - t0

    npz = {"step": np.asarray(hist["step"], float),
           "torque_blade": np.asarray(hist["torque_blade"], float),
           "torque_ice": np.asarray(hist["torque_ice"], float),
           "ke": np.asarray(hist["ke"], float),
           "ie": np.asarray(hist["ie"], float),
           "mass": np.asarray(hist["mass"], float),
           "diss": np.asarray(hist["diss"], float),
           "power_in": np.asarray(hist["power_in"], float),
           "mass_source": np.asarray(hist["mass_source"], float),
           "ke_added": np.asarray(hist["ke_added"], float),
           "ke_removed": np.asarray(hist["ke_removed"], float)}
    for k, s in hist["snapshots"].items():
        for fld in ("rho", "vort", "labels"):
            npz[f"snap{k}_{fld}"] = s[fld]
        npz[f"snap{k}_angle"] = np.array(s["angle"])

    payload = {
        "case": name, "n": n, "overrides": over,
        "cfg": cfg.to_dict(),
        "wall_seconds": wall,
        "ms_per_step": 1e3 * wall / max(cfg.n_steps, 1),
        "n_steps": cfg.n_steps,
        "mass_ledger": mass_ledger_residual(hist),
        "energy_ledger": energy_ledger_residual(hist, cfg),
        "torque_mean": float(np.mean(hist["torque_blade"])),
        "torque_max_raw": float(np.max(hist["torque_blade"])),
    }
    np.savez_compressed(OUT / f"case_{name}.npz", **npz)
    (OUT / f"case_{name}.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"[case {name}] n={n} steps={cfg.n_steps} {wall:.1f}s "
          f"({payload['ms_per_step']:.1f} ms/step) "
          f"mass_rel={payload['mass_ledger']['relative']:.2e} "
          f"energy_rel={payload['energy_ledger']['relative']:.2e}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------


def analyse(npz_path: Path, skip_frac: float = 0.30, smooth_frac: float = 1.0 / 16.0):
    d = np.load(npz_path)
    steps = d["step"]
    tor = d["torque_blade"]
    period = float(num_period(steps, tor))
    skip = skip_frac * period
    m = steps >= skip
    ts, tor_win = steps[m], tor[m]
    w = max(int(round(smooth_frac * period)), 1)
    sm = np.asarray(smooth_signal(tor_win, w), dtype=float)
    return {"steps": ts, "raw": tor_win, "smooth": sm, "window": w,
            "period": period, "data": d}


def num_period(steps, tor):
    """Rotation period recovered from the config stored next to the arrays."""
    # period_steps is written into the case JSON; fall back to a 1-step grid
    return _PERIOD_HINT.get(_current_case, 1.0)


_PERIOD_HINT: dict = {}


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------


def assemble() -> dict:
    metrics = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "cases": {}}

    def load(name):
        j = json.loads((OUT / f"case_{name}.json").read_text())
        d = np.load(OUT / f"case_{name}.npz")
        metrics["cases"][name] = j
        return j, d

    def window(j, d, skip_frac=0.30, smooth_frac=1.0 / 16.0):
        cfg = j["cfg"]
        period = cfg["period_steps"]
        steps = d["step"]
        m = steps >= skip_frac * period
        ts, tor = steps[m], d["torque_blade"][m]
        w = max(int(round(smooth_frac * period)), 1)
        return ts, tor, np.asarray(smooth_signal(tor, w), float), period, w

    # ---- 1. grid convergence -------------------------------------------
    conv = []
    for n in GRIDS:
        j, d = load(f"conv{n}")
        ts, tor, sm, period, w = window(j, d)
        cfg = MillingConfig(**{k: v for k, v in j["cfg"].items()
                               if k in MillingConfig.__dataclass_fields__
                               and not k.startswith(("blade_radius", "hub_radius",
                                                     "blade_thickness", "gap",
                                                     "ice_center", "ice_width",
                                                     "omega", "nu", "tau",
                                                     "n_steps"))})
        ct = np.array([torque_coefficient(v, cfg) for v in sm])
        ct_amp = torque_coefficient(float(sm.max() - sm.min()), cfg)
        ct_peak = torque_coefficient(float(sm.max()), cfg)
        rec = {
            "n": n, "R_lat": j["cfg"]["blade_radius"],
            "gap_cells": j["cfg"]["gap"],
            "thickness_cells": j["cfg"]["blade_thickness"],
            "tau": j["cfg"]["tau"], "steps_per_rev": period,
            "smooth_window": w,
            "C_T_pulse": float(ct_amp),
            "C_T_peak_abs": float(ct_peak),
            "C_T_mean": float(ct.mean()), "C_T_min": float(ct.min()),
            "T_pulse_lat": float(sm.max() - sm.min()),
            "T_peak_lat": float(sm.max()), "T_mean_lat": float(sm.mean()),
            "T_min_lat": float(sm.min()),
            "peak_lat": float(sm.max()),
            "peak_to_mean": float(sm.max() / sm.mean()),
            "pulse_contrast": float((sm.max() - sm.min()) / abs(sm.mean())),
            "smooth_window_frac_of_period": w / period,
            "mass_ledger": j["mass_ledger"],
            "energy_ledger": j["energy_ledger"],
            "wall_seconds": j["wall_seconds"],
            "ms_per_step": j["ms_per_step"],
        }
        conv.append(rec)
        print(f"[conv] n={n} R={rec['R_lat']:.1f} gap={rec['gap_cells']:.2f} "
              f"C_T_pulse={rec['C_T_pulse']:.4f} C_T_peak_abs={rec['C_T_peak_abs']:.3f}", flush=True)

    hs = [1.0 / c["R_lat"] for c in conv]
    vals = [c["C_T_pulse"] for c in conv]
    rich = richardson_order(hs, vals)
    res = [abs(vals[1] - vals[0]), abs(vals[2] - vals[1])]
    rich["residual_ratio"] = res[1] / max(res[0], 1e-30)
    rich["monotone"] = bool((vals[1] - vals[0]) * (vals[2] - vals[1]) > 0)
    rich["p_interpretable"] = bool(rich["monotone"] and np.isfinite(rich["p"])
                                   and rich["p"] > 0)
    # the *absolute* peak level is offset-sensitive (the converged quantity is the
    # pulse amplitude); report its order too so the caveat is auditable
    rich_abs = richardson_order(hs, [c["C_T_peak_abs"] for c in conv])
    refine_ratio = conv[1]["R_lat"] / conv[0]["R_lat"]
    metrics["grid_convergence"] = {
        "grids": [c["n"] for c in conv], "h": hs,
        "C_T_pulse": vals, "T_pulse_lat": [c["T_pulse_lat"] for c in conv],
        "richardson_pulse": rich,
        "richardson_absolute_peak": rich_abs,
        "refinement_ratio": float(refine_ratio),
        "order_estimate_reliable": bool(refine_ratio >= 1.5 and rich["p_interpretable"]),
        "caveat": ("the grid family refines by only 1.25 per step, so the observed "
                   "order is non-asymptotic; the absolute torque level "
                   f"({', '.join('%.2f' % c['C_T_peak_abs'] for c in conv)}) is not "
                   "converged while the pulse amplitude is"),
        "per_grid": conv,
        "criterion_p_ge_1": bool(rich["p_interpretable"] and rich["p"] >= 1.0),
    }

    # ---- 2. periodicity --------------------------------------------------
    j, d = load("perio64")
    ts, tor, sm, period, w = window(j, d, skip_frac=0.5)
    fft = dominant_frequency(sm, dt=1.0)
    cfg = j["cfg"]
    f_bpf = cfg["blade_passing_freq"]
    rel = abs(fft["f_peak"] - f_bpf) / f_bpf
    metrics["periodicity"] = {
        "n": j["n"], "steps_per_rev": period, "smooth_window": w,
        "n_samples": int(len(sm)), "df": float(fft["freqs"][1] - fft["freqs"][0]),
        "f_peak": fft["f_peak"], "f_bpf": f_bpf,
        "f_peak_over_f_bpf": fft["f_peak"] / f_bpf,
        "relative_difference": rel,
        "top_peaks_over_f_bpf": [p[0] / f_bpf for p in fft["peaks"][:4]],
        "criterion_rel_le_1pct": bool(rel <= 0.01),
        "_torque": sm, "_steps": ts, "_freqs": fft["freqs"], "_amp": fft["amplitude"],
        "_period": period, "_raw": tor, "_wall": j["wall_seconds"],
    }
    print(f"[fft] f_peak/f_bpf={fft['f_peak']/f_bpf:.5f} rel={rel:.2e}", flush=True)

    # ---- 3. negative control A: no ice ----------------------------------
    j, d = load("noice64")
    ts, tor, sm, period, w = window(j, d)
    rms = fluctuation_rms(sm)
    metrics["negative_control_no_ice"] = {
        "n": j["n"], "smooth_window": w, **rms,
        "criterion_rms_le_1pct": bool(rms["rms_over_abs_mean"] <= 0.01),
        "_torque": sm, "_steps": ts, "_period": period,
    }
    print(f"[ctrl-A] rms/|mean|={rms['rms_over_abs_mean']:.3e}", flush=True)

    # ---- 4. negative control B: baseline, one peak per pass -------------
    j, d = load("base64")
    ts, tor, sm, period, w = window(j, d)
    pulses = count_pulses(sm, prominence_frac=0.25)
    n_passes = (ts[-1] - ts[0]) / period * j["cfg"]["n_blades"]
    metrics["negative_control_static_ice"] = {
        "n": j["n"], "smooth_window": w, "n_peaks": pulses["n_peaks"],
        "expected_passes": float(n_passes),
        "peak_indices": pulses["indices"],
        "peak_heights": pulses.get("heights", []),
        "criterion_single_peak_per_pass": bool(
            abs(pulses["n_peaks"] - round(n_passes)) <= 1),
        "_torque": sm, "_steps": ts, "_period": period, "_raw": tor,
        "_snapshots": {k: d[k] for k in d.files if k.startswith("snap")},
    }
    print(f"[ctrl-B] peaks={pulses['n_peaks']} expected={n_passes:.2f}", flush=True)

    # pulse shape: duty above half-prominence and rise time, LBM vs empirical
    sm = metrics["negative_control_static_ice"]["_torque"]
    per_b = metrics["negative_control_static_ice"]["_period"]
    thr = 0.5 * (sm.max() + np.percentile(sm, 10))
    above = sm > thr
    frac_above = float(above.mean())
    idx = int(np.argmax(sm))
    j0 = idx
    while j0 > 0 and sm[j0 - 1] < thr:
        j0 -= 1
    rise_steps = float(idx - j0)
    metrics["pulse_shape"] = {
        "duty_above_half_prominence": frac_above,
        "empirical_duty": float(json.loads((OUT / "case_base64.json").read_text())["cfg"]["ice_width"]
                                / (2 * math.pi / json.loads((OUT / "case_base64.json").read_text())["cfg"]["n_blades"])),
        "rise_steps": rise_steps,
        "rise_fraction_of_period": rise_steps / per_b,
        "peak_over_mean": float(sm.max() / abs(sm.mean())),
        "trough_over_mean": float(sm.min() / abs(sm.mean())),
    }
    print(f"[pulse] duty={frac_above:.3f} rise={rise_steps:.0f} steps", flush=True)

    # ---- 5. ledgers ------------------------------------------------------
    mb = metrics["cases"]["base64"]["mass_ledger"]
    # the energy balance is evaluated on the finest grid of the family (its
    # residual is grid dependent: 26% / 2.2% / 5.7% at n=64/80/96)
    eb_fine = metrics["cases"]["conv96"]["energy_ledger"]
    eb = metrics["cases"]["base64"]["energy_ledger"]
    metrics["conservation"] = {
        "mass_ledger_baseline": mb,
        "energy_ledger_finest_grid_n96": eb_fine,
        "energy_ledger_baseline_n64": eb,
        "energy_ledger_by_grid": {k: v["energy_ledger"]["relative"]
                                  for k, v in metrics["cases"].items()},
        "mass_ledger_by_case": {k: v["mass_ledger"]["relative"]
                                for k, v in metrics["cases"].items()},
        "criterion_mass_residual_le_1e-10": bool(mb["relative"] <= 1e-10),
        "criterion_energy_residual_le_5pct": bool(eb_fine["relative"] <= 0.05),
    }
    print(f"[ledger] mass_rel={mb['relative']:.3e} energy_rel(n96)="
          f"{eb_fine['relative']:.3e} energy_rel(n64)={eb['relative']:.3e}", flush=True)

    # ---- 6. ISO 19906 comparison ---------------------------------------
    # The lattice torque is condensed into the dimensionless pulse coefficient
    # C_T = dT/(rho Omega^2 R^4 b) and evaluated at a documented operating point;
    # the ISO 19906 crushing baseline is evaluated for the same geometry.  The
    # ratio is swept over the plausible operating range (rev/s, ice strength)
    # so the answer is not a single cherry-picked point.
    cfg_b = MillingConfig(**{k: v for k, v in
                             json.loads((OUT / "case_base64.json").read_text())["cfg"].items()
                             if k in MillingConfig.__dataclass_fields__
                             and not k.startswith(("blade_radius", "hub_radius",
                                                   "blade_thickness", "gap", "ice_center",
                                                   "ice_width", "omega", "nu", "tau",
                                                   "n_steps"))})
    ice = Iso19906Ice()
    emp = iso19906_milling_torque(ice)
    c_t_pulse = conv[-1]["C_T_pulse"]
    scaling = physical_scaling(cfg_b, rev_per_s=2.0)
    t_lbm = c_t_pulse * scaling["torque_scale"]
    ratio = t_lbm / emp["torque"]
    sweep = []
    for n_rps in (1.0, 1.5, 2.0, 2.5, 3.0):
        for sig in (1.0e6, 2.0e6, 3.0e6):
            sc = physical_scaling(cfg_b, rev_per_s=n_rps)
            ic = Iso19906Ice(sigma_c=sig)
            e = iso19906_milling_torque(ic)
            sweep.append({"rev_per_s": n_rps, "sigma_c_MPa": sig / 1e6,
                          "T_lbm": c_t_pulse * sc["torque_scale"],
                          "T_emp": e["torque"],
                          "ratio": c_t_pulse * sc["torque_scale"] / e["torque"],
                          "u_tip": sc["u_tip_m_per_s"]})
    ratios = [d["ratio"] for d in sweep]
    metrics["iso19906_comparison"] = {
        "ice": ice.__dict__, "empirical": emp, "scaling": scaling,
        "C_T_pulse_lbm": c_t_pulse, "lbm_peak_grid": f"n={conv[-1]['n']}",
        "T_lbm_peak_Nm": t_lbm, "T_emp_peak_Nm": emp["torque"],
        "ratio_lbm_over_emp": ratio,
        "deviation_factor": float(max(ratio, 1.0 / ratio)),
        "criterion_within_factor_2": bool(0.5 <= ratio <= 2.0),
        "sweep": sweep,
        "sweep_ratio_min": float(min(ratios)),
        "sweep_ratio_max": float(max(ratios)),
        "sweep_fraction_within_factor_2": float(np.mean([0.5 <= r <= 2.0 for r in ratios])),
        "caveat": ("dimension analysis, not Reynolds-matched: the lattice run is at "
                   "Re=120 (viscous dominated) while a real propeller is at Re~1e6, "
                   "so C_T is not Reynolds-transferable; the agreement at the "
                   "reference point (2 rev/s) is indicative only"),
    }
    print(f"[ISO] ratio(ref 2rev/s)={ratio:.3f} sweep=[{min(ratios):.2f},{max(ratios):.2f}]",
          flush=True)

    verdict = {
        "C1_grid_convergence_p_ge_1": metrics["grid_convergence"]["criterion_p_ge_1"],
        "C2_periodicity_rel_le_1pct": metrics["periodicity"]["criterion_rel_le_1pct"],
        "C3_mass_residual_le_1e-10": metrics["conservation"]["criterion_mass_residual_le_1e-10"],
        "C4_energy_residual_le_5pct": metrics["conservation"]["criterion_energy_residual_le_5pct"],
        "C5_iso19906_within_factor_2": metrics["iso19906_comparison"]["criterion_within_factor_2"],
        "C6_no_ice_steady_rms_le_1pct": metrics["negative_control_no_ice"]["criterion_rms_le_1pct"],
        "C6_static_ice_single_peak": metrics["negative_control_static_ice"]["criterion_single_peak_per_pass"],
    }
    verdict["all_pass"] = all(verdict.values())
    metrics["verdict"] = verdict

    make_figures(metrics, conv)
    metrics_out = {k: v for k, v in metrics.items() if k != "cases"}
    for key in ("periodicity", "negative_control_no_ice", "negative_control_static_ice"):
        metrics_out[key] = {k: v for k, v in metrics[key].items() if not k.startswith("_")}
    metrics_out["cases"] = {k: {kk: vv for kk, vv in v.items() if kk != "torque"}
                            for k, v in metrics["cases"].items()}
    (OUT / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=float))
    print(json.dumps(verdict, indent=2), flush=True)
    return metrics


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(metrics, conv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = metrics["periodicity"]
    base = metrics["negative_control_static_ice"]
    nc = metrics["negative_control_no_ice"]
    gc = metrics["grid_convergence"]

    # --- fig1: torque history, baseline ------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    tot = base["_steps"] / base["_period"]
    ax.plot(tot, base["_raw"], lw=0.4, color="0.75", label="raw (lattice-scale noise)")
    ax.plot(tot, base["_torque"], lw=1.4, color="k",
            label=f"smoothed (window {base['smooth_window']} steps)")
    for c in conv:
        ax.axhline(c["T_peak_lat"], ls="--", lw=0.9,
                   label=f"smoothed peak n={c['n']}: {c['T_peak_lat']:.1f}")
    ax.set_xlabel("t / rotation period")
    ax.set_ylabel("blade torque $T$ [lattice]")
    ax.set_title("Transient blade torque in the ice-blocked passage\n"
                 "(2 blades, one 60-degree ice sector; pulse = blade passing the ice)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_torque_history.png", dpi=140)
    plt.close(fig)

    # --- fig2: FFT ---------------------------------------------------------
    fr = per["_freqs"] / per["f_bpf"]
    amp = per["_amp"] / np.max(per["_amp"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(fr, amp, lw=1.2)
    ax.axvline(1.0, color="r", ls="--",
               label=r"blade-passing frequency $N_b f_{rot}$")
    ax.set_xlim(0, 6)
    ax.set_xlabel(r"$f / f_{BPF}$")
    ax.set_ylabel("normalised amplitude")
    ax.set_title(f"Torque spectrum (n={per['n']}): peak at "
                 f"{per['f_peak_over_f_bpf']:.4f} $f_{{BPF}}$ "
                 f"(rel. diff {100*per['relative_difference']:.3f}%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_fft_bpf.png", dpi=140)
    plt.close(fig)

    # --- fig3: snapshots ---------------------------------------------------
    snaps = base.get("_snapshots", {})
    keys = sorted({int(k[4:].split("_")[0]) for k in snaps})
    if keys:
        fig, axes = plt.subplots(2, len(keys), figsize=(3.9 * len(keys), 7.4))
        if len(keys) == 1:
            axes = np.asarray(axes).reshape(2, 1)
        for jj, k in enumerate(keys):
            lab = snaps[f"snap{k}_labels"]
            vort = snaps[f"snap{k}_vort"].astype(float).copy()
            rho = snaps[f"snap{k}_rho"].astype(float).copy()
            vort[lab > 0] = np.nan
            rho[lab > 0] = np.nan
            vmax = np.nanpercentile(np.abs(vort), 99)
            im0 = axes[0, jj].imshow(vort, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                     origin="lower")
            ang = float(snaps[f"snap{k}_angle"])
            axes[0, jj].set_title(f"step {k}  theta={math.degrees(ang):.0f} deg")
            plt.colorbar(im0, ax=axes[0, jj], fraction=0.046, label=r"$\omega_z$")
            im1 = axes[1, jj].imshow(rho - 1.0, cmap="viridis", origin="lower")
            plt.colorbar(im1, ax=axes[1, jj], fraction=0.046, label=r"$\rho-1$")
            for a in (axes[0, jj], axes[1, jj]):
                a.set_xticks([])
                a.set_yticks([])
        fig.suptitle("Instantaneous fields: blade approaching / inside / leaving "
                     "the ice sector (white = solid)")
        fig.tight_layout()
        fig.savefig(FIG / "fig3_field_snapshots.png", dpi=140)
        plt.close(fig)

    # --- fig4: grid convergence -------------------------------------------
    h = np.asarray(gc["h"])
    v = np.asarray(gc["C_T_pulse"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(h, v, "o-", label=r"LBM $C_T$ pulse (peak-to-peak)")
    if gc["richardson_pulse"]["p_interpretable"]:
        p_ = gc["richardson_pulse"]
        f0 = p_["f_extrapolated"]
        a = (v[-1] - v[-2]) / (h[-1] ** p_['p'] - h[-2] ** p_['p'])
        hh = np.linspace(0, h.max() * 1.05, 60)
        ax.plot(hh, f0 + a * hh ** p_['p'], "--",
                label=f"Richardson fit p={p_['p']:.2f} (non-asymptotic)")
        ax.axhline(f0, color="g", lw=0.8, ls=":",
                   label=f"continuum estimate {f0:.4f}")
    ax.set_xlabel("grid spacing $h \\propto 1/R$  [1/lattice]")
    ax.set_ylabel(r"$C_T$ pulse $=\Delta T/(\rho\Omega^2R^4)$")
    ax.set_title("Grid convergence of the milling-load pulse amplitude")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_grid_convergence.png", dpi=140)
    plt.close(fig)

    # --- fig5: LBM vs empirical pulse model -------------------------------
    cfg_json = metrics["cases"]["base64"]["cfg"]
    duty = cfg_json["ice_width"] / (2 * math.pi / cfg_json["n_blades"])
    peak = float(np.max(base["_torque"]))
    period = base["_period"]
    t_norm = (base["_steps"] - base["_steps"][0]) / period
    emp = []
    for tn in t_norm:
        ph = (tn * cfg_json["blade_passing_freq"]) % 1.0
        emp.append(peak * math.sin(math.pi * ph / duty) if ph <= duty else 0.0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t_norm, base["_torque"], lw=1.4, color="k", label="LBM direct solve")
    ax.plot(t_norm, emp, lw=1.2, color="tab:red", alpha=0.85,
            label=f"empirical pulse model (half-sine, duty={duty:.3f})")
    ax.set_xlabel("t / rotation period")
    ax.set_ylabel("torque [lattice]")
    ax.set_title("Direct LBM solve vs the prescribed empirical pulse train "
                 "(same peak level)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_lbm_vs_empirical.png", dpi=140)
    plt.close(fig)

    # --- fig6: negative controls ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].plot(nc["_steps"] / nc["_period"], nc["_torque"], lw=1.2)
    axes[0].set_title("negative control A: no ice\n"
                      f"rms/|mean| = {100*nc['rms_over_abs_mean']:.3f}% (limit 1%)")
    axes[0].set_xlabel("t / rotation period")
    axes[0].set_ylabel("blade torque [lattice]")
    axes[1].plot(base["_steps"] / period, base["_torque"], lw=1.2)
    for idx in base["peak_indices"]:
        axes[1].plot(base["_steps"][idx] / period, base["_torque"][idx], "rx")
    axes[1].set_title("negative control B: static ice\n"
                      f"{base['n_peaks']} peaks for {base['expected_passes']:.2f} passes")
    axes[1].set_xlabel("t / rotation period")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_negative_controls.png", dpi=140)
    plt.close(fig)
    print("[figs]", sorted(p.name for p in FIG.glob("*.png")), flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "assemble":
        assemble()
    else:
        for name in sys.argv[1:]:
            run_case(name)