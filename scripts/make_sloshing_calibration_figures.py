#!/usr/bin/env python3
"""Figures for the sloshing-impact LBM calibration campaign.

Reads the case JSONs written by ``scripts/run_sloshing_impact_calibration.py``
plus the assembled ``report.json`` and writes publication figures to
``artifacts/sloshing_impact_lbm/figures/``.

    python scripts/make_sloshing_calibration_figures.py \
        --case-dir artifacts/sloshing_impact_lbm \
        --fig-dir artifacts/sloshing_impact_lbm/figures
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.sloshing_impact_calibration import (  # noqa: E402
    TankAnchor,
    harmonic_analysis,
    smooth,
)

ANCHOR = TankAnchor()
FILTER_SECONDS = 0.10
GRIDS = (40, 60, 80)
COLORS = {40: "#1f77b4", 60: "#ff7f0e", 80: "#2ca02c"}


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _filtered(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(case["t"], float)
    p = np.asarray(case["p_wall"], float)
    dt_s = t[1] - t[0] if t.size > 1 else case["dt"]
    win = max(3, int(round(FILTER_SECONDS / dt_s)))
    return t, smooth(p, win), p


def fig_wall_pressure(cases: dict, swe: dict | None, fig_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.4), sharex=True)
    ax = axes[0]
    for nx in GRIDS:
        c = cases.get(f"impact_nx{nx}")
        if not c:
            continue
        t, pf, praw = _filtered(c)
        ax.plot(t, praw, color=COLORS[nx], lw=0.6, alpha=0.35)
        ax.plot(t, pf, color=COLORS[nx], lw=1.6,
                label=f"LBM L/{nx} ({c['nx']}x{c['ny']}x{c['nz']}), 0.1 s mean")
    ax.axhline(1.0, color="0.4", ls=":", lw=1.0)
    ax.text(0.15, 1.06, "hydrostatic $\\rho g h_l$", color="0.35", fontsize=8)
    ax.axhline(ANCHOR.p_closed_form / ANCHOR.p_hydrostatic, color="crimson",
               ls="--", lw=1.2)
    ax.text(0.15, ANCHOR.p_closed_form / ANCHOR.p_hydrostatic + 0.06,
            "closed form $4\\rho g h_l$ (Ritter stagnation)", color="crimson", fontsize=8)
    ax.axhline(ANCHOR.p_bore_reflection / ANCHOR.p_hydrostatic, color="purple",
               ls="-.", lw=1.0)
    ax.text(0.15, ANCHOR.p_bore_reflection / ANCHOR.p_hydrostatic - 0.22,
            "exact bore reflection $3.494\\rho g h_l$", color="purple", fontsize=8)
    ax.set_ylabel(r"$p_{wall}/(\rho g h_l)$  [-]")
    ax.set_title("Surge-release impact: LBM dynamic wall pressure (IN48-N07 tank)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for nx in GRIDS:
        c = cases.get(f"impact_nx{nx}")
        if not c:
            continue
        t = np.asarray(c["t"], float)
        ax.plot(t, np.asarray(c["h_wall"], float), color=COLORS[nx], lw=1.4,
                label=f"LBM L/{nx}")
    if swe and "surge" in swe:
        ts = np.asarray(swe["surge"]["t"], float)
        hs = np.asarray(swe["surge"]["h_wall"], float) / ANCHOR.depth
        ax.plot(ts, hs, "k--", lw=1.6, label="SWE (IN48-N07 chain)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$h_{wall}/h_l$  [-]")
    ax.set_title("Wall-adjacent free-surface height: LBM vs shallow-water chain")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    out = fig_dir / "fig01_impact_wall_pressure.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_fields(case: dict, fig_dir: Path) -> Path | None:
    snaps = case.get("snapshots") or []
    if not snaps:
        return None
    n = len(snaps)
    cols = min(5, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.4 * rows),
                             squeeze=False)
    for k, snap in enumerate(snaps):
        ax = axes[k // cols][k % cols]
        arr = np.asarray(snap["fill_yx"], float)
        im = ax.imshow(arr, origin="lower", cmap="Blues", vmin=0.0, vmax=1.0,
                       extent=[0, case["dx"] * arr.shape[1], 0,
                               case["dx"] * arr.shape[0]], aspect="auto")
        ax.set_title(f"t = {snap['t']:.2f} s", fontsize=9)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="Körner fill")
    fig.suptitle(
        f"Free-surface field (mid-breadth slice), LBM L/{case['nx']} "
        f"({case['nx']}x{case['ny']}x{case['nz']}, $\\tau$={case['meta']['tau']:.4f})",
        fontsize=10,
    )
    out = fig_dir / f"fig02_fields_nx{case['nx']}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_gci(report: dict, fig_dir: Path) -> Path | None:
    grids = report.get("grids") or {}
    if len(grids) < 3:
        return None
    dx = np.array([grids[k]["dx"] for k in sorted(grids, key=int)], float)
    peak = np.array([grids[k]["p_peak_filtered"] for k in sorted(grids, key=int)], float)
    imp = np.array([grids[k]["impulse_filtered"] for k in sorted(grids, key=int)], float)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, vals, label, gci_key in (
        (axes[0], peak, r"$p_{peak}/(\rho g h_l)$", "gci_p_peak"),
        (axes[1], imp, r"$\int p_{wall}\,dt$  [$\rho g h_l$ s]", "gci_impulse"),
    ):
        ax.plot(dx, vals, "o-", color="C0", label="LBM grids L/40, L/60, L/80")
        g = report.get(gci_key) or {}
        if g.get("usable") and math.isfinite(g.get("extrapolated", float("nan"))):
            ax.axhline(g["extrapolated"], color="C3", ls="--",
                       label=f"Richardson limit (p={g['p_obs']:.2f})")
            ax.text(dx.max(), g["extrapolated"], f"  GCI={100*g['gci_fine']:.2f}%",
                    color="C3", fontsize=8, va="bottom", ha="right")
        ax.set_xlabel("grid spacing dx [m]")
        ax.set_ylabel(label)
        ax.set_title(f"{label}: grid convergence")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    out = fig_dir / "fig03_gci.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_spectrum(forced: dict, fig_dir: Path) -> Path | None:
    t = np.asarray(forced["t"], float)
    p = np.asarray(forced["p_wall"], float)
    spec = harmonic_analysis(t, p, ANCHOR.omega1)
    if not spec.get("ok"):
        return None
    n = p.size
    w = np.hanning(n)
    amp = np.abs(np.fft.rfft((p - p.mean()) * w)) / (w.sum() / 2.0)
    om = 2.0 * np.pi * np.fft.rfftfreq(n, d=float(t[1] - t[0]))
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.semilogy(om, np.maximum(amp, 1e-12), "k-", lw=1.0, label="wall pressure spectrum")
    w1 = ANCHOR.omega1
    for k, style in ((1, "--"), (2, ":"), (3, "-.")):
        ax.axvline(k * w1, color="C1", ls=style, lw=1.0)
        ax.text(k * w1, ax.get_ylim()[1] * 0.5 if k == 1 else ax.get_ylim()[1] * 0.05,
                f" {k}$\\omega_1$", color="C1", fontsize=9)
    ax.axvline(spec["omega_peak"], color="C0", lw=1.0, alpha=0.8)
    ax.text(spec["omega_peak"], ax.get_ylim()[1] * 0.15,
            f" measured $\\omega$={spec['omega_peak']:.3f} rad/s", color="C0", fontsize=8)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel("|P| [arb.]")
    ax.set_title(
        "Forced sloshing (A=0.25 m at $\\omega_1$): wall-pressure spectrum, "
        f"2$\\omega_1$ energy fraction = {100*spec['energy_2omega_fraction']:.1f}%"
    )
    ax.grid(alpha=0.25, which="both")
    out = fig_dir / "fig04_forced_spectrum.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_cushion(base: dict, cushion: dict, fig_dir: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for case, label, color in (
        (base, "single phase (open ullage, ambient gas)", "C0"),
        (cushion, "sealed air cushion, $p=p_0(V_0/V)^{1.4}$", "C3"),
    ):
        t, pf, praw = _filtered(case)
        ax.plot(t, praw, color=color, lw=0.6, alpha=0.3)
        ax.plot(t, pf, color=color, lw=1.7, label=label)
    ax.axhline(1.0, color="0.4", ls=":", lw=1.0)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$p_{wall}/(\rho g h_l)$  [-]")
    ax.set_title("Negative control (criterion e): gas cushion must lower the impact peak")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    out = fig_dir / "fig05_cushion_negative_control.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_hydro(hydro: dict, fig_dir: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(
        ["L2 rel. error", "max rel. error"],
        [100.0 * hydro["l2_rel_error"], 100.0 * hydro["max_rel_error"]],
        color=["C0", "C1"], width=0.5,
    )
    ax.axhline(1.0, color="crimson", ls="--", lw=1.2)
    ax.text(0.4, 1.05, "criterion: < 1 %", color="crimson", fontsize=8)
    ax.set_ylabel("error vs $\\rho g (h_l-y)$  [%]")
    ax.set_title(
        f"Hydrostatic baseline, LBM L/{hydro['nx']} ({hydro['cells']} cells, "
        f"$u_{{max}}$={hydro['u_max_lbm']:.1e})"
    )
    ax.grid(alpha=0.25, axis="y")
    out = fig_dir / "fig06_hydrostatic.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--swe-reference", default=None)
    args = ap.parse_args(argv)
    case_dir = Path(args.case_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    swe_path = Path(args.swe_reference) if args.swe_reference else case_dir / "sw_swe_reference.json"

    report = _load(case_dir / "report.json") or {}
    swe = _load(swe_path)
    cases = {f"impact_nx{nx}": _load(case_dir / f"impact_nx{nx}.json") for nx in GRIDS}
    cushion = _load(case_dir / "impact_cushion_nx40.json")
    forced = _load(case_dir / "forced_nx40.json") or _load(case_dir / "forced_nx60.json")
    hydro = _load(case_dir / "hydrostatic_nx40.json")

    made = []
    if any(cases.values()):
        made.append(fig_wall_pressure(cases, swe, fig_dir))
    fine = cases.get(f"impact_nx{max(GRIDS)}") or cases.get("impact_nx40")
    if fine:
        made.append(fig_fields(fine, fig_dir))
    if report.get("grids"):
        made.append(fig_gci(report, fig_dir))
    if forced:
        made.append(fig_spectrum(forced, fig_dir))
    if cushion and cases.get("impact_nx40"):
        made.append(fig_cushion(cases["impact_nx40"], cushion, fig_dir))
    if hydro:
        made.append(fig_hydro(hydro, fig_dir))
    for p in made:
        if p:
            print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())