#!/usr/bin/env python
"""Run the pulsatile-channel (Womersley) verification case and emit artifacts.

Produces, under ``artifacts/womersley_pulsatile_<date>/``:

* ``results.json``      — every number the doc/report quotes (single source),
* ``fig_*.png``         — comparison figures (also copied to ``docs/figures/``),
* ``table_*.md``        — markdown tables injected into the doc.

Study design (each block answers a different question):

1. ``steady``          — Guo forcing + wall position calibration (u_c = aH²/8ν).
2. ``resolution``      — α = 5 at ny = 16/24/32/48: is the gate result
   resolution-independent and what is the observed order?
3. ``alpha_sweep``     — α = 1/2/5/10, Stokes layer held at ≈6.4 nodes: does the
   amplitude ratio / phase lag track the exact solution across regimes
   (quasi-steady → intermediate → boundary-layer)?
4. ``controls``        — (a) transient independence (skip periods 1 vs 3),
   (b) rest start (no analytic initial condition) must reproduce the phasor.

Usage::

    python scripts/run_womersley_pulsatile_case.py [--out DIR] [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import date
from pathlib import Path

import numpy as np

import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorlbm.verification.womersley_pulsatile import (  # noqa: E402
    CASE_KEY,
    GATE,
    ChannelPulsatileConfig,
    channel_centreline_phasor,
    run_pulsatile_channel,
    run_steady_poiseuille,
)

NU = 0.05
ALPHA_GATE = GATE["alpha"]
STOKES_NODES = 6.4  # target Stokes-layer thickness in lattice nodes for the sweep


def _sweep_resolution():
    out = []
    for ny in (16, 24, 32, 48):
        res = run_pulsatile_channel(
            ChannelPulsatileConfig(
                ny=ny, nu=NU, alpha=ALPHA_GATE, skip_periods=2, measure_periods=2,
                tag=f"resolution_ny{ny}",
            )
        )
        out.append(res)
        print(
            f"  ny={ny:3d} amp_err={res['amp_rel_pct']:.4f}% "
            f"phase={res['phase_diff_deg']:+.4f}deg "
            f"prof_max={res['profile_amp_rel_pct_max']:.3f}% "
            f"gate={res['gate']['status']}",
            flush=True,
        )
    return out


def _sweep_alpha():
    out = []
    for alpha in (1.0, 2.0, 5.0, 10.0):
        ny = max(12, int(round(STOKES_NODES * alpha / math.sqrt(2.0))))
        res = run_pulsatile_channel(
            ChannelPulsatileConfig(
                ny=ny, nu=NU, alpha=alpha, skip_periods=2, measure_periods=2,
                tag=f"alpha{alpha:g}_ny{ny}",
            )
        )
        out.append(res)
        print(
            f"  alpha={alpha:4.1f} ny={ny:3d} T={res['period_steps']:6d} "
            f"delta={res['stokes_layer_nodes']:.2f} "
            f"amp_err={res['amp_rel_pct']:.4f}% phase={res['phase_diff_deg']:+.4f}deg "
            f"prof_max={res['profile_amp_rel_pct_max']:.3f}%",
            flush=True,
        )
    return out


def _controls():
    ny = 24
    base = dict(ny=ny, nu=NU, alpha=ALPHA_GATE, measure_periods=2)
    skip1 = run_pulsatile_channel(
        ChannelPulsatileConfig(skip_periods=1, init_from_exact=True, tag="skip1", **base)
    )
    skip3 = run_pulsatile_channel(
        ChannelPulsatileConfig(skip_periods=3, init_from_exact=True, tag="skip3", **base)
    )
    rest = run_pulsatile_channel(
        ChannelPulsatileConfig(skip_periods=4, init_from_exact=False, tag="rest_start", **base)
    )
    print(
        f"  skip1 lag={skip1['centre_lag_num_deg']:.4f}  "
        f"skip3 lag={skip3['centre_lag_num_deg']:.4f}  "
        f"rest lag={rest['centre_lag_num_deg']:.4f} "
        f"(analytic {rest['centre_lag_ana_deg']:.4f})",
        flush=True,
    )
    return {"skip1": skip1, "skip3": skip3, "rest_start": rest}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _fig_profile(res, path):
    y = np.asarray(res["profile"]["yhat"])
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax[0].plot(y, res["profile"]["amp_ratio"], "o-", ms=3, label="LBM / analytic")
    ax[0].axhline(1.0, ls="--", c="k", lw=0.8)
    ax[0].set_xlabel(r"$\hat y = 2y/H$")
    ax[0].set_ylabel("amplitude ratio")
    ax[0].set_title(f"(a) amplitude ratio, α={res['alpha_effective']:.2f}, ny={res['ny']}")
    ax[0].set_ylim(0.97, 1.01)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(y, res["profile"]["lag_num_deg"], "o-", ms=3, label="LBM")
    ax[1].plot(y, res["profile"]["lag_ana_deg"], "-", label="exact")
    ax[1].set_xlabel(r"$\hat y = 2y/H$")
    ax[1].set_ylabel("phase lag vs flow rate [deg]")
    ax[1].set_title("(b) phase lag across the channel")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_waveform(res, path):
    """Centreline waveform over one period + per-row complex error."""
    ph = np.asarray(res["profile"]["phase_centers"])
    y = np.asarray(res["profile"]["yhat"])
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax[0].plot(np.degrees(ph), res["profile"]["centreline_wave_num"], "o-", ms=3, label="LBM")
    ax[0].plot(np.degrees(ph), res["profile"]["centreline_wave_ana"], "-", label="exact")
    ax[0].set_xlabel(r"phase $\omega t$ [deg]")
    ax[0].set_ylabel("$u_c$ [lattice units]")
    ax[0].set_title("(a) centreline velocity, whole-period average")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(y, res["profile"]["complex_rel_pct"], "o-", ms=3)
    ax[1].set_xlabel(r"$\hat y = 2y/H$")
    ax[1].set_ylabel("|phasor error| [%]")
    ax[1].set_title("(b) per-row complex amplitude error")
    ax[1].set_yscale("log")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_inst_profiles(res, path):
    """Instantaneous profiles at four phases: LBM markers vs exact lines."""
    y = np.asarray(res["profile"]["yhat"])
    u_num = np.asarray(res["profile"]["profile_phase_averaged"])
    ph = np.asarray(res["profile"]["phase_centers"])
    z_ana = np.asarray(res["profile"]["ana_phasor_re"]) + 1j * np.asarray(
        res["profile"]["ana_phasor_im"]
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    n = len(ph)
    for frac in (0.0, 0.25, 0.5, 0.75):
        k = int(frac * n) % n
        ang = ph[k]
        ax.plot(
            u_num[k], y, "o", ms=3,
            label=f"LBM  $\\omega t$={math.degrees(ang):.0f}°",
        )
        ax.plot(np.real(z_ana * np.exp(1j * ang)), y, "-", lw=1.0, c="k")
    ax.set_xlabel("$u_x$ [lattice units]")
    ax.set_ylabel(r"$\hat y$")
    ax.set_title("instantaneous profiles (markers LBM, lines exact)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_convergence(rows, path):
    ny = np.array([r["ny"] for r in rows], dtype=float)
    amp = np.array([r["amp_rel_pct"] for r in rows])
    pha = np.array([abs(r["phase_diff_deg"]) for r in rows])
    prof = np.array([r["profile_amp_rel_pct_max"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax[0].loglog(ny, amp, "o-", label="centreline amplitude")
    ax[0].loglog(ny, pha, "s-", label="centreline phase [deg]")
    ax[0].loglog(ny, prof, "^-", label="profile max (interior)")
    ref = amp[0] * (ny / ny[0]) ** -2.0
    ax[0].loglog(ny, ref, "k--", lw=0.8, label=r"$\propto N^{-2}$")
    ax[0].set_xlabel("channel rows $N=H$")
    ax[0].set_ylabel("error [%] / [deg]")
    ax[0].set_title("(a) grid convergence at α = 5")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")

    alpha = np.array([r["alpha_effective"] for r in rows])
    ax[1].semilogx(alpha, amp, "o-", label="amplitude error")
    ax[1].semilogx(alpha, pha, "s-", label="phase error [deg]")
    ax[1].set_xlabel(r"$\alpha$")
    ax[1].set_ylabel("error")
    ax[1].set_title("(b) same data vs α")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_alpha(sweep, path):
    alpha_num = np.array([r["alpha_effective"] for r in sweep])
    amp_num = np.array([r["centre_amp_num_over_bulk"] for r in sweep])
    lag_num = np.array([r["centre_lag_num_deg"] for r in sweep])
    lag_ana = np.array([r["centre_lag_ana_deg"] for r in sweep])
    a_fine = np.geomspace(0.5, 15.0, 200)
    lag_fine = [-math.degrees(np.angle(channel_centreline_phasor(a))) for a in a_fine]
    amp_fine = [abs(channel_centreline_phasor(a)) for a in a_fine]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax[0].semilogx(a_fine, amp_fine, "-", lw=1.0, c="k", label="exact")
    ax[0].semilogx(alpha_num, amp_num, "o", label="LBM")
    ax[0].set_xlabel(r"$\alpha$")
    ax[0].set_ylabel("centreline amplitude ratio")
    ax[0].set_title("(a) centreline amplitude")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].semilogx(a_fine, lag_fine, "-", lw=1.0, c="k", label="exact")
    ax[1].semilogx(alpha_num, lag_num, "o", label="LBM")
    ax[1].set_xlabel(r"$\alpha$")
    ax[1].set_ylabel("centreline lag vs flow rate [deg]")
    ax[1].set_title("(b) centreline phase lag")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def _tables(steady, resolution, sweep, controls):
    lines = []
    lines.append("### Resolution study at α = 5\n")
    lines.append(
        "| ny (=H) | δ [nodes] | period [steps] | amp err [%] | phase err [°] | "
        "profile max [%] | profile L2 [%] | gate |\n"
        "|---:|---:|---:|---:|---:|---:|---:|:--|\n"
    )
    for r in resolution:
        lines.append(
            f"| {r['ny']} | {r['stokes_layer_nodes']:.2f} | {r['period_steps']} | "
            f"{r['amp_rel_pct']:.4f} | {r['phase_diff_deg']:+.4f} | "
            f"{r['profile_amp_rel_pct_max']:.3f} | {r['profile_wave_l2_pct']:.3f} | "
            f"{r['gate']['status']} |\n"
        )
    lines.append("\n### Womersley-number sweep (Stokes layer ≈ 6.4 nodes)\n")
    lines.append(
        "| α | ny | period [steps] | amp ratio LBM/ana | amp err [%] | "
        "lag LBM [°] | lag exact [°] | phase err [°] | Ma |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in sweep:
        lines.append(
            f"| {r['alpha_effective']:.3f} | {r['ny']} | {r['period_steps']} | "
            f"{r['amp_ratio_num_over_ana']:.5f} | {r['amp_rel_pct']:.4f} | "
            f"{r['centre_lag_num_deg']:.3f} | {r['centre_lag_ana_deg']:.3f} | "
            f"{r['phase_diff_deg']:+.4f} | {r['Ma']:.3f} |\n"
        )
    lines.append("\n### Controls\n")
    lines.append(
        "| run | skip periods | init | centreline lag [°] | amp err [%] | "
        "phase err [°] |\n|---|---:|---|---:|---:|---:|\n"
    )
    for key, label, skip, init in (
        ("skip1", "seeded, skip 1", 1, "exact"),
        ("skip3", "seeded, skip 3", 3, "exact"),
        ("rest_start", "rest start, skip 4", 4, "rest"),
    ):
        r = controls[key]
        lines.append(
            f"| {label} | {skip} | {init} | {r['centre_lag_num_deg']:.4f} | "
            f"{r['amp_rel_pct']:.4f} | {r['phase_diff_deg']:+.4f} |\n"
        )
    lines.append("\n### Steady Poiseuille calibration\n")
    lines.append(
        f"ny={steady['ny']}, ν={steady['nu']}, a0={steady['a0']:.1e}, "
        f"steps={steady['steps']}: "
        f"$u_c$ LBM {steady['u_centre_num']:.8f} vs $aH^2/8ν$ "
        f"{steady['u_centre_ana']:.8f} → {steady['centre_rel_pct']:.4f} % "
        f"(profile max {steady['profile_rel_max_pct']:.4f} %), "
        f"density drift {steady['density_drift_pct']:.1e} %.\n"
    )
    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="skip the alpha sweep")
    args = ap.parse_args()

    out = Path(args.out) if args.out else REPO / "artifacts" / f"womersley_pulsatile_{date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("[1/4] steady Poiseuille calibration", flush=True)
    steady = run_steady_poiseuille(ny=16, nu=NU, a0=1e-4, steps=25600)
    print(
        f"  u_c rel={steady['centre_rel_pct']:.4f}% profile max="
        f"{steady['profile_rel_max_pct']:.4f}%",
        flush=True,
    )

    print("[2/4] resolution study (alpha = 5)", flush=True)
    resolution = _sweep_resolution()

    print("[3/4] alpha sweep", flush=True)
    sweep = [] if args.quick else _sweep_alpha()

    print("[4/4] controls", flush=True)
    controls = _controls()

    figs = {}
    figs["profile"] = out / "fig_profile_alpha5.png"
    figs["waveform"] = out / "fig_centreline_waveform.png"
    figs["inst"] = out / "fig_inst_profiles.png"
    figs["conv"] = out / "fig_convergence.png"
    _fig_profile(resolution[2], figs["profile"])
    _fig_waveform(resolution[2], figs["waveform"])
    _fig_inst_profiles(resolution[2], figs["inst"])
    _fig_convergence(resolution, figs["conv"])
    if sweep:
        figs["alpha"] = out / "fig_alpha_sweep.png"
        _fig_alpha(sweep, figs["alpha"])

    fig_dir = REPO / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for p in figs.values():
        shutil.copy2(p, fig_dir / f"womersley_pulsatile_{p.name[4:]}")

    tables = _tables(steady, resolution, sweep, controls)
    (out / "table_study.md").write_text(tables, encoding="utf-8")

    results = {
        "case": CASE_KEY,
        "gate": GATE,
        "nu": NU,
        "steady": steady,
        "resolution": resolution,
        "alpha_sweep": sweep,
        "controls": controls,
        "figures": [str(p.relative_to(REPO)) for p in figs.values()],
        "wall_s": round(time.time() - t_start, 1),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    gate_res = [r for r in resolution if abs(r["alpha_effective"] - ALPHA_GATE) < 5e-3]
    print(
        f"done in {results['wall_s']} s -> {out} | gate statuses: "
        f"{[r['gate']['status'] for r in gate_res]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())