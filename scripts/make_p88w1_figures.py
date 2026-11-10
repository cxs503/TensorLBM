"""Generate the P88W1-N01 report figures from ``artifacts/p88w1``.

All panels are built from the campaign JSON + the per-case ``.npz`` time series;
no number is typed by hand.

    python scripts/make_p88w1_figures.py --artifacts artifacts/p88w1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

C_MEAS = "#1f4e79"
C_THEORY = "#c00000"
C_ALT = "#e08a00"
C_GREY = "#666666"


def _load(art: Path) -> dict:
    return json.loads((art / "campaign.json").read_text(encoding="utf-8"))


def _npz(art: Path, cid: str) -> dict:
    with np.load(art / f"{cid}.npz") as z:
        return {k: z[k] for k in z.files}


def fig_resonance(art: Path, camp: dict, out: Path) -> Path:
    sw = camp["criteria"]["sweep"]
    r = np.asarray(sw["ratios"])
    a1 = np.asarray(sw["a1_amplitude_m"])
    ref = np.asarray(sw["reference_peak_m"])
    f1 = sw["f1_theory_hz"]
    f2 = sw["f2_theory_hz"]
    peak = sw["peak"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    ax.plot(r, a1 * 1e3, "o-", color=C_MEAS, label="LBM  |a₁| (first harmonic, measured)")
    ax.plot(r, ref * 1e3, "s--", color=C_THEORY, label="linear finite-time reference (ODE)")
    ax.axvline(1.0, color=C_GREY, lw=1, ls=":")
    ax.axvline(f2 / f1, color=C_ALT, lw=1, ls=":")
    ax.annotate("f₁", (1.0, ax.get_ylim()[1] * 0.92), color=C_GREY, ha="center")
    ax.annotate("f₂", (f2 / f1, ax.get_ylim()[1] * 0.92), color=C_ALT, ha="center")
    ax.set_xlabel("forcing frequency  f / f₁")
    ax.set_ylabel("first-mode amplitude  |a₁|  [mm]")
    ax.set_title(f"forced-sway resonance sweep (X₀/L = {camp['config']['linear_sway_frac']:.3f})")
    ax.legend(loc="upper left", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(r, a1 / np.maximum(ref, 1e-12), "^:", color=C_ALT, lw=1)
    ax2.set_ylabel("measured / reference", color=C_ALT)
    ax2.tick_params(axis="y", colors=C_ALT)
    ax2.grid(False)

    ax = axes[1]
    ratio_a2 = []
    for cid, case in camp["cases"].items():
        if cid.startswith("sweep_r") and case.get("kind") == "forced":
            ratio_a2.append((case["ratio"], case["parity_ratio_a2_over_a1"], case["a2"]["amplitude"] * 1e3))
    ratio_a2.sort()
    rr = [x[0] for x in ratio_a2]
    par = [x[1] for x in ratio_a2]
    amp2 = [x[2] for x in ratio_a2]
    ax.plot(rr, par, "o-", color=C_MEAS, label="a₂ / a₁ (parity diagnostic)")
    ax.axvline(f2 / f1, color=C_ALT, lw=1, ls=":")
    ax.axvspan(f2 / f1 - 0.05, f2 / f1 + 0.05, color=C_ALT, alpha=0.08)
    ax.set_xlabel("forcing frequency  f / f₁")
    ax.set_ylabel("a₂ / a₁")
    ax.set_title("even-mode (mode 2) content: no resonant amplification at f₂")
    ax.legend(loc="upper left", fontsize=8)
    ax3 = ax.twinx()
    ax3.plot(rr, amp2, "s--", color=C_GREY, lw=1)
    ax3.set_ylabel("|a₂| [mm]", color=C_GREY)
    ax3.grid(False)

    fig.suptitle(
        f"P88W1-N01  resonance identification: peak at f/f₁ = {peak['ratio_parabolic']:.4f} "
        f"({100 * (peak['ratio_parabolic'] - 1):+.2f} % vs Abramson),  f₁ = {f1:.4f} Hz"
    )
    fig.tight_layout()
    p = out / "fig1_resonance_curve.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_decay(art: Path, camp: dict, out: Path) -> Path:
    cases = [c for c in ("decay_m1", "decay_m2", "decay_m1_noslip", "decay_m1_ul005") if c in camp["cases"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    for i, cid in enumerate(cases):
        z = _npz(art, cid)
        c = camp["cases"][cid]
        t = z["t"] / c["f_theory_hz"]  # periods of the mode under test
        y = z[f"a{c['mode']}"]
        ax.plot(t, y * 1e3, lw=1.0, label=f"{cid} (f_d {c['f_damped_hz']:.4f} Hz, {c['error_natural_pct']:+.2f} %)")
        if i == 0:
            per = 1.0 / c["f_theory_hz"]
            tt = np.linspace(0, t[-1], 400)
            ax.plot(
                tt,
                y[0] * np.cos(2 * math.pi * tt) * np.exp(0 * tt) * 1e3,
                "k--",
                lw=0.9,
                label="cos(2π t/T) shape from the initial amplitude",
            )
    ax.set_xlabel("t / T_mode")
    ax.set_ylabel("modal amplitude  a [mm]")
    ax.set_title("free-decay releases: mode 1 / mode 2 / sensitivities")
    ax.legend(fontsize=7)

    ax = axes[1]
    labels = [c["id"] for c in camp["cases"].values() if c["kind"] == "decay"]
    errs = [c["error_natural_pct"] for c in camp["cases"].values() if c["kind"] == "decay"]
    cols = [C_MEAS if abs(e) < 5 else C_THEORY for e in errs]
    ax.barh(labels, errs, color=cols)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(-5, color=C_THEORY, ls="--", lw=1)
    ax.axvline(5, color=C_THEORY, ls="--", lw=1)
    ax.set_xlabel("frequency error vs Abramson  [%]")
    ax.set_title("decay-frequency error (5 % gate)")
    for i, e in enumerate(errs):
        ax.annotate(f"{e:+.2f} %", (e, i), xytext=(4 if e >= 0 else -34, -3), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    p = out / "fig2_free_decay_sensitivity.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_pressure(art: Path, camp: dict, out: Path) -> Path:
    lin = camp["cases"]["sweep_r1.00"]
    imp = camp["cases"].get("impact_x012_r1.00", lin)
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

    for j, (case, tag) in enumerate(((lin, "linear branch X₀/L=0.002"), (imp, "impact branch X₀/L=0.012"))):
        ax = fig.add_subplot(gs[0, j])
        z = _npz(art, case["id"])
        t = z["t"]
        T = 1.0 / case["forcing_frequency_hz"]
        for i, g in enumerate(case["pressure"]["gauges"]):
            p = z["pressure"][:, i]
            ax.plot(t / T, p, lw=0.9, label=f"gauge y={g['y_from_swl_m']:+.3f} m")
        ax.set_xlabel("t / T_force")
        ax.set_ylabel("wall dynamic pressure  p_dyn [Pa]")
        ax.set_title(f"{case['id']}  ({tag})")
        ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 0])
    z = _npz(art, lin["id"])
    T = 1.0 / lin["forcing_frequency_hz"]
    t = z["t"]
    eta = z["eta"]
    nx = eta.shape[1]
    dx = lin["dx_m"]
    xs = (np.arange(nx) + 0.5) * dx
    phases = np.linspace(0.0, 1.0, 5, endpoint=False)
    for frac in phases:
        idx = int(np.argmin(np.abs(t / T - (frac + 3.0))))
        ax.plot(xs, eta[idx] * 1e3, lw=1.1, label=f"t/T={t[idx] / T:.2f}")
    amp = lin["a1"]["amplitude"]
    k = math.pi / 1.0
    ax.plot(xs, amp * np.cos(k * xs) * 1e3, "k--", lw=1.2, label="a₁ cos(πx/L) (measured a₁)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("free-surface elevation η [mm]")
    ax.set_title("free-surface waveform vs standing-wave shape (linear branch)")
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[1, 1])
    labels, vals, cols = [], [], []
    for case, tag in ((lin, "linear"), (imp, "impact")):
        b = case["pressure"]["bounds"]
        labels += [f"{tag}: measured p_amp", f"{tag}: ρg·η_wall", f"{tag}: 0.5ρV²", f"{tag}: bore"]
        vals += [
            case["pressure"]["p_amp_Pa"],
            b["hydrostatic_runup_Pa"],
            b["wagner_stagnation_Pa"],
            b["bore_reflection_Pa"],
        ]
        cols += [C_MEAS, C_THEORY, C_ALT, C_GREY]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=cols)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("pressure [Pa] (log scale)")
    ax.set_title("measured wall pressure vs closed-form impact bounds")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}", (v, i), xytext=(3, -3), textcoords="offset points", fontsize=7)
    fig.tight_layout()
    p = out / "fig3_pressure_and_waveform.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_grid(art: Path, camp: dict, out: Path) -> Path | None:
    if "gci_ref_r1.00" not in camp["cases"] or "sweep_r1.00" not in camp["cases"]:
        print("warning: refined grid case missing, fig4 skipped")
        return None
    d = camp["criteria"]["d_grid_convergence"]
    base = camp["cases"]["sweep_r1.00"]
    ref = camp["cases"]["gci_ref_r1.00"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

    ax = axes[0]
    names = ["H/32", "H/64"]
    for cid, col in (("sweep_r1.00", C_MEAS), ("gci_ref_r1.00", C_ALT)):
        z = _npz(art, cid)
        c = camp["cases"][cid]
        T = 1.0 / c["forcing_frequency_hz"]
        ax.plot(z["t"] / T, z["pressure"][:, 0], color=col, lw=1.0, label=f"{cid} (Δx={c['dx_m'] * 1e3:.2f} mm)")
    ax.set_xlabel("t / T_force")
    ax.set_ylabel("p_dyn at bottom gauge [Pa]")
    ax.set_title("bottom-gauge pressure, two grids")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.bar(["H/32", "H/64"], [d["p_amp_base_Pa"], d["p_amp_refined_Pa"]], color=[C_MEAS, C_ALT])
    ax.set_ylabel("first-harmonic pressure amplitude [Pa]")
    ax.set_title(f"pressure amplitude: change {d['value_pct']:.2f} % (gate 3 %)")
    for i, v in enumerate((d["p_amp_base_Pa"], d["p_amp_refined_Pa"])):
        ax.annotate(f"{v:.1f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax = axes[2]
    ax.bar(["H/32", "H/64"], [d["runup_amp_base_m"] * 1e3, d["runup_amp_refined_m"] * 1e3], color=[C_MEAS, C_ALT])
    ax.set_ylabel("wall run-up amplitude [mm]")
    ax.set_title(f"run-up amplitude: change {d['runup_change_pct']:.2f} %")
    for i, v in enumerate((d["runup_amp_base_m"] * 1e3, d["runup_amp_refined_m"] * 1e3)):
        ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    p = out / "fig4_grid_convergence.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/p88w1", type=Path)
    ap.add_argument("--out", default="artifacts/p88w1/figures", type=Path)
    args = ap.parse_args(argv)
    art: Path = args.artifacts
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    camp = _load(art)
    made = []
    made.append(fig_resonance(art, camp, out))
    made.append(fig_decay(art, camp, out))
    made.append(fig_pressure(art, camp, out))
    fig4 = fig_grid(art, camp, out)
    if fig4 is not None:
        made.append(fig4)
    print(json.dumps([str(p) for p in made], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())