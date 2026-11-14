"""Figures for CASE-RT-INSTABILITY-ATWOOD (A = 1/2 single-mode Rayleigh-Taylor).

Run after ``run_case.py``:

    python3 make_figures.py              # figures 1,2,4 from the run artifacts
    python3 make_figures.py --snapshots  # also re-runs the reference case to dump
                                         # phi / speed fields (figure 3)

Everything is read from ``artifacts/runs/*/`` (history.csv, instant_rate.csv,
case.json, summary.json); no number is typed in by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
sys.path.insert(0, str(BENCH_DIR))

import run_case as rc  # noqa: E402

FIG_DIR = BENCH_DIR / "figures"
RUNS = BENCH_DIR / "artifacts" / "runs"


def _load(name: str) -> tuple[dict, list[dict]]:
    d = RUNS / name
    case = json.loads((d / "case.json").read_text())
    hist = list(csv.DictReader((d / "history.csv").open()))
    return case, hist


def _col(hist: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in hist], dtype=float)


def fig_amplitude(names: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for name in names:
        case, hist = _load(name)
        tau = _col(hist, "tau")
        a = _col(hist, "a_zc_over_lam")
        ok = np.isfinite(a) & (a > 0)
        ax.semilogy(tau[ok], a[ok], lw=1.4, label=f"{name} (A={case['atwood']:.2f}, λ={case['lam']})")
        # theory envelope anchored at the first sample
        if ok.sum() > 2:
            t0, a0 = tau[ok][0], a[ok][0]
            tt = np.linspace(t0, tau[ok][-1], 200)
            for sig, style, lab in ((case["sigma_viscous"], "--", r"$\sigma_{visc}$"),
                                     (case["sigma_inviscid"], ":", r"$\sigma_{inv}$")):
                ax.semilogy(tt, a0 * np.exp(sig * (tt - t0)), style, color="0.35", lw=1.0)
            ax.annotate("", xy=(0, 0), xytext=(0, 0))
    lo, hi = rc.WINDOWS[rc.PRIMARY_WINDOW]
    ax.axhspan(lo, hi, color="tab:green", alpha=0.08)
    ax.text(0.02, math.sqrt(lo * hi), " primary fit window\n a/λ ∈ [%.3f, %.3f]"
            % (lo, hi), fontsize=8, color="tab:green", va="center")
    ax.plot([], [], "--", color="0.35", label=r"theory $a_0e^{\sigma_{visc}t}$")
    ax.plot([], [], ":", color="0.35", label=r"theory $a_0e^{\sigma_{inv}t}$")
    ax.set_xlabel(r"normalised time $\tau = t\,\sqrt{Ag/\lambda}$")
    ax.set_ylabel(r"interface amplitude $a/\lambda$ (φ = 0 contour)")
    ax.set_title("A = 1/2 single-mode RT: exponential growth of the interface mode")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(5e-3, 0.3)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "fig1_amplitude_growth.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_instant_rate(names: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for name in names:
        case, _ = _load(name)
        rows = list(csv.DictReader((RUNS / name / "instant_rate.csv").open()))
        a = np.array([float(r["a_zc_over_lam"]) for r in rows])
        s = np.array([float(r["sigma_inst"]) for r in rows])
        ok = np.isfinite(s) & np.isfinite(a)
        ax.plot(a[ok], s[ok] / case["sigma_viscous"], lw=1.4,
                label=f"{name} (ν_eff/ν={case['nu_eff_over_nu']:.1f})")
    ax.axhline(1.0, color="0.2", lw=1.0, ls="--", label=r"$\sigma_{visc}$ (theory)")
    lo, hi = rc.WINDOWS[rc.PRIMARY_WINDOW]
    ax.axvspan(lo, hi, color="tab:green", alpha=0.08)
    ax.set_xscale("log")
    ax.set_xlabel(r"$a/\lambda$")
    ax.set_ylabel(r"$\sigma_{inst}/\sigma_{visc}$  ($d\ln a/dt$ on a 25-sample window)")
    ax.set_title("Instantaneous growth rate: startup transient then nonlinear decay")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / "fig2_instant_rate.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def snapshots(case_name: str = "A050_l064_g2e-4", n_frames: int = 4) -> Path:
    """Re-run the reference case and dump φ and |u| fields at n_frames instants."""
    spec = next(s for s in rc.SPECS if s.name == case_name)
    cfg = spec.cfg()
    torch.manual_seed(0)
    f, g, phi = rc.init_rt_state(cfg)
    stop_step = int(round(math.log(0.35 / cfg.eta0_frac) / rc.ref_lock.sigma_inviscid(
        spec.atwood, spec.g, spec.lam)))
    targets = [int(round(stop_step * frac)) for frac in np.linspace(0.15, 1.0, n_frames)]
    out_dir = BENCH_DIR / "artifacts" / "fields"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for step in range(1, targets[-1] + 1):
        f, g = rc.rt_step(f, g, cfg, phi=phi)
        phi = g.sum(dim=0)
        if step in targets:
            _rho, ux, uy = rc.macroscopic(f)
            speed = torch.sqrt(ux * ux + uy * uy)
            frames.append({"step": step, "tau": step * math.sqrt(spec.atwood * spec.g / spec.lam),
                           "phi": phi.numpy().copy(), "speed": speed.numpy().copy()})
    np.savez_compressed(out_dir / "fields_A050_l064_g2e-4.npz",
                        **{f"{k}_{i}": fr[k] for i, fr in enumerate(frames) for k in ("phi", "speed")},
                        steps=np.array([fr["step"] for fr in frames]),
                        taus=np.array([fr["tau"] for fr in frames]))
    return out_dir / "fields_A050_l064_g2e-4.npz"


def fig_fields(npz: Path) -> Path:
    data = np.load(npz)
    steps = data["steps"]
    taus = data["taus"]
    n = len(steps)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.0), squeeze=False)
    for i in range(n):
        phi = data[f"phi_{i}"]
        spd = data[f"speed_{i}"]
        ext = [0, phi.shape[1], 0, phi.shape[0]]
        im0 = axes[0][i].imshow(phi, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[0][i].set_title(f"φ, step {int(steps[i])}, τ={taus[i]:.2f}", fontsize=8)
        vmax = max(float(np.percentile(spd, 99.5)), 1e-6)
        im1 = axes[1][i].imshow(spd, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax)
        axes[1][i].set_title(r"$|\mathbf{u}|$", fontsize=8)
        for ax in (axes[0][i], axes[1][i]):
            ax.set_xlabel("x / lattice", fontsize=7)
            ax.set_ylabel("y / lattice", fontsize=7)
            ax.tick_params(labelsize=6)
        fig.colorbar(im0, ax=axes[0][i], fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
        fig.colorbar(im1, ax=axes[1][i], fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    fig.suptitle("A = 1/2 single-mode RT, Boussinesq D2Q9 + Allen-Cahn interface "
                 "(λ=64, g=2e-4, τ=0.515)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "fig3_fields.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_ladder(summary: dict) -> Path:
    cases = [c for c in summary["cases"] if not c["blown_up"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ladder = [c for c in cases if c["name"].startswith("A050_l0") and "_tau" not in c["name"]
              and "g1e-4" not in c["name"] and "legacy" not in c["name"]]
    ax = axes[0]
    if ladder:
        lam = [c["lam"] for c in ladder]
        err = [c["err_vs_viscous_pct"] for c in ladder]
        errl = [c["err_legacy_window_vs_viscous_pct"] for c in ladder]
        ax.plot(lam, err, "o-", label="primary window [0.02, 0.08]")
        ax.plot(lam, errl, "s--", label="legacy window [0.01, 0.05]")
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axhline(rc.GATE_PCT, color="tab:red", lw=0.8, ls=":")
    ax.axhline(-rc.GATE_PCT, color="tab:red", lw=0.8, ls=":", label=f"±{rc.GATE_PCT:.0f}% gate")
    ax.set_xlabel(r"$\lambda$ (lattice cells), constant $Ag/\lambda$")
    ax.set_ylabel("growth-rate error vs $\\sigma_{visc}$ [%]")
    ax.set_title("grid ladder (constant dimensionless problem)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    sens = [c for c in cases if c["name"] not in {x["name"] for x in ladder}]
    labels = [c["name"].replace("A050_l064_", "") for c in sens]
    vals = [c["err_vs_viscous_pct"] for c in sens]
    colors = ["tab:orange" if "legacy" in c["name"] else "tab:blue" for c in sens]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axhline(rc.GATE_PCT, color="tab:red", lw=0.8, ls=":")
    ax.axhline(-rc.GATE_PCT, color="tab:red", lw=0.8, ls=":")
    ax.set_ylabel("growth-rate error vs $\\sigma_{visc}$ [%]")
    ax.set_title("sensitivity axes (ν, g, interface scheme)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG_DIR / "fig4_ladder_sensitivity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", action="store_true")
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RUNS / "summary.json").read_text())
    names = [c["name"] for c in summary["cases"] if not c["blown_up"]]
    print(fig_amplitude(names))
    print(fig_instant_rate(names))
    print(fig_ladder(summary))
    npz = BENCH_DIR / "artifacts" / "fields" / "fields_A050_l064_g2e-4.npz"
    if args.snapshots or not npz.exists():
        npz = snapshots()
        print(npz)
    print(fig_fields(npz))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())