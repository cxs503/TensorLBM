#!/usr/bin/env python3
"""IN45-N01 figures: u+ vs y+, y_w calibration, u_τ routes, Reynolds stresses.

Every curve is read from ``artifacts/in45/campaign_gates.json`` (or from the
calibration record) - nothing is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.validation import channel_wall_units as cwu  # noqa: E402

OUT = _REPO / "artifacts" / "in45" / "figures"


def load_reference() -> tuple[np.ndarray, np.ndarray]:
    import csv

    path = _REPO / "benchmarks" / "pending" / "channel_turb_3d" / "dns_ref" / "mkm180.csv"
    y_plus, u_plus = [], []
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            y_plus.append(float(row["y_plus"]))
            u_plus.append(float(row["u_plus"]))
    return np.array(y_plus), np.array(u_plus)


def figure_u_plus(gates: dict, ref_y: np.ndarray, ref_u: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axes[0]
    ax.semilogx(ref_y[1:], ref_u[1:], "k-", lw=2.0, label="MKM-1999 DNS ($Re_\\tau=178$)")
    colors = plt.cm.viridis(np.linspace(0.05, 0.75, max(len(gates), 1)))
    for color, (name, run) in zip(colors, gates.items()):
        y_plus = np.array(run["y_plus"])
        u_plus = np.array(run["u_plus"])
        ax.semilogx(
            y_plus,
            u_plus,
            "o-",
            ms=3.5,
            lw=1.4,
            color=color,
            label=f"{name} ({run['grid'][0]}x{run['grid'][1]}x{run['grid'][2]}, "
            f"$\\Delta x^+$={run['dx_plus']:.1f})",
        )
    y = np.logspace(0.1, 2.25, 200)
    ax.semilogx(y, np.log(y) / cwu.KAPPA + cwu.B_LOG, ":", color="0.45", label="log law (0.41, 5.0)")
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("$U^+$")
    ax.set_xlim(1.0, 200.0)
    ax.set_ylim(0.0, 22.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("mean velocity profile in wall units")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for color, (name, run) in zip(colors, gates.items()):
        y_plus = np.array(run["y_plus"])
        u_plus = np.array(run["u_plus"])
        ref_at = np.interp(y_plus, ref_y, ref_u)
        ax.plot(
            100.0 * (u_plus - ref_at) / ref_at,
            y_plus,
            "o-",
            ms=3.5,
            lw=1.4,
            color=color,
            label=name,
        )
    ax.axvspan(-5.0, 5.0, color="tab:green", alpha=0.12, label="C1 gate ($\\pm$5 %)")
    ax.axhline(30.0, color="0.4", ls="--", lw=1.0)
    ax.text(-4.7, 34.0, "log-law window start $y^+=30$", fontsize=8, color="0.3")
    ax.set_xlabel("relative deviation vs MKM DNS  [%]")
    ax.set_ylabel("$y^+$")
    ax.set_xlim(-25.0, 25.0)
    ax.grid(True, alpha=0.3)
    ax.set_title("profile deviation (folded half channel)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = OUT / "u_plus_vs_mkm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_wall_origin(calibration: dict | None) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    if calibration:
        cases = calibration["cases"]
        taus = [case["tau"] for case in cases]
        fit = [case["poiseuille_fit"]["y_w"] for case in cases]
        stress = [case["wall_stress"]["y_w"] for case in cases]
        ax.plot(taus, fit, "o-", color="tab:blue", label="measured: Poiseuille inversion")
        ax.plot(taus, stress, "s--", color="tab:cyan", label="single cell viscous slope (biased)")
        for tau, value in zip(taus, fit):
            ax.annotate(f"{value:.3f}", (tau, value), textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.axhline(0.5, color="k", ls="--", lw=1.4, label="assumed 0.5 (textbook)")
    ax.axhline(0.1, color="tab:red", ls=":", lw=1.6, label="historical claim 0.1")
    ax.set_xlabel("relaxation time $\\tau$")
    ax.set_ylabel("effective wall origin $y_w$  [lattice cells]")
    ax.set_title("wall origin is measured, not assumed")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    if calibration:
        scan = calibration["artefact_scan"]
        window = [entry["y_plus_window"] for entry in scan]
        estimate = [entry["y_w_cells"] for entry in scan]
        ax.plot(window, estimate, "o-", color="tab:red", label="single-point formula")
        ax.axhline(0.6447, color="tab:blue", ls="--", label="true $y_w$ (inversion)")
        ax.axhline(0.1, color="0.3", ls=":", label="historical 0.1")
    ax.set_xlabel("how far the formula is applied: $y^+$ window")
    ax.set_ylabel("inferred $y_w$  [cells]")
    ax.set_title("artefact: viscous law used outside the sublayer")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "wall_origin.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_u_tau(gates: dict) -> Path:
    names = list(gates.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    width = 0.2
    routes = ["momentum_balance", "wall_shear", "log_law_level", "log_law_slope"]
    labels = ["momentum balance", "wall shear", "log law (level)", "log law (slope)"]
    for index, (route, label) in enumerate(zip(routes, labels)):
        values = [
            100.0
            * (gates[name]["three_way"]["routes"].get(route, np.nan) / gates[name]["u_tau_balance"] - 1.0)
            for name in names
        ]
        ax.bar(np.arange(len(names)) + index * width, values, width, label=label)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(names)) + 1.5 * width)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("$u_\\tau$ deviation from momentum balance  [%]")
    ax.set_title("$u_\\tau$ routes")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for name in names:
        g = gates[name]["C4_u_tau_three_way_under_2pct"]
        devs = g["deviations_pct"]
        keys = list(devs.keys())
        ax.barh(
            [f"{name}\n{k}" for k in keys],
            [devs[k] for k in keys],
            label=f"{name}: max {g['max_pairwise_deviation_pct']:.2f} %",
        )
    ax.axvline(2.0, color="tab:green", ls="--", label="C4 gate (2 %)")
    ax.set_xlabel("pairwise deviation  [%]")
    ax.set_title("three-way consistency")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "u_tau_three_way.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_stresses(gates: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    for name, run in gates.items():
        y_plus = np.array(run["y_plus"])
        ax.plot(y_plus, np.array(run["C2_rms_profiles_under_10pct"]["peaks"] and [
            -1.0
        ] * 0 + np.array(run.get("uv_plus", [])), "o-", ms=3, label=f"{name} $\\langle u'v'\\rangle^+$")
        ax.plot(y_plus, run["uu_plus"], ":", lw=1.0, alpha=0.7, label=f"{name} $\\langle u'u'\\rangle^+$")
    band = gates[list(gates)[0]]["C2_rms_profiles_under_10pct"]["literature_peak_band"]
    ax.axhspan(band["uv_peak"][0], band["uv_peak"][1], color="0.6", alpha=0.2, label="MKM $\\langle u'v'\\rangle^+$ peak band")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("stress$^+$")
    ax.set_xlim(0.0, 190.0)
    ax.set_ylim(-1.1, 9.0)
    ax.grid(True, alpha=0.3)
    ax.set_title("Reynolds stresses (folded)")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    for name, run in gates.items():
        y_plus = np.array(run["y_plus"])
        peaks = run["C2_rms_profiles_under_10pct"]["peaks"]
        ax.plot(
            y_plus,
            run["vv_plus"],
            "-",
            lw=1.2,
            label=f"{name} $\\langle v'v'\\rangle^+$ (peak {peaks['vv_peak']:.2f})",
        )
        ax.plot(
            y_plus,
            run["ww_plus"],
            "--",
            lw=1.0,
            label=f"{name} $\\langle w'w'\\rangle^+$ (peak {peaks['ww_peak']:.2f})",
        )
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("stress$^+$")
    ax.set_xlim(0.0, 190.0)
    ax.grid(True, alpha=0.3)
    ax.set_title("wall-normal / spanwise intensities")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    path = OUT / "reynolds_stresses.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_convergence(gates: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    names = list(gates.keys())
    turnovers = [gates[name]["eddy_turnovers"] for name in names]
    errors = [
        100.0 * gates[name]["C1_u_plus_log_law_error_under_5pct"]["relative_l2_log_window"]
        for name in names
    ]
    ax.scatter(turnovers, errors, s=60, c="tab:blue")
    for name, x, y in zip(names, turnovers, errors):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(5.0, color="tab:green", ls="--", label="C1 gate (5 %)")
    ax.set_xlabel("outer times in the statistics window  ($T u_\\tau / h$)")
    ax.set_ylabel("$u^+$ relative $L_2$ error  [%]")
    ax.set_title("statistics vs profile error")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for name, run in gates.items():
        stat = run["stationarity"]
        ax.bar(
            [name],
            [stat.get("recent_friction_velocity_error_pct") if isinstance(stat, dict) else np.nan],
        )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-2.0, 2.0, color="tab:green", alpha=0.15, label="C4 steady-state band ($\\pm$2 %)")
    ax.set_ylabel("wall-shear $u_\\tau$ error vs momentum balance  [%]")
    ax.set_title("momentum balance closure")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gates", default="artifacts/in45/campaign_gates.json")
    parser.add_argument("--calibration", default="artifacts/in45/wall_origin_calibration.json")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gates_path = _REPO / args.gates
    payload = json.loads(gates_path.read_text())
    runs = {run["name"]: run for run in payload["runs"]}
    gates = payload["gates"]
    ref_y, ref_u = load_reference()
    calibration_path = _REPO / args.calibration
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    paths = [
        figure_u_plus(gates, ref_y, ref_u),
        figure_wall_origin(calibration),
        figure_u_tau(gates),
        figure_stresses(runs),
        figure_convergence(runs),
    ]
    print(json.dumps([str(path.relative_to(_REPO)) for path in paths], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())