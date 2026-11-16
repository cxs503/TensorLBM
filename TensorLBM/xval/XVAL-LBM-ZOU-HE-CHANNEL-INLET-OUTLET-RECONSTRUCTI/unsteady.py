#!/usr/bin/env python
"""Unsteady (Womersley) campaign for the Zou-He channel closure.

Anchor: ``XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI``.

The steady campaign (``case.py`` / ``refinement.py``) closes the converged-field
debt.  This driver closes the *transient* debt: the channel is driven entirely
through the two reconstructed planes with oscillating end densities, and the
periodic response is compared with the exact oscillatory plane-Poiseuille
(Sexl 1930) solution.

Arms
----
``alpha2``      H=16, alpha=2  -- primary arm, Stokes layer 4 cells
``alpha4``      H=16, alpha=4  -- thin Stokes layer (2 cells), hardest for
                                 half-way bounce-back
``alpha6``      H=16, alpha=6  -- stress arm: Stokes layer 1.3 cells and the
                                 amplitude peak leaves the axis (annular effect)
``alpha2_w4``   H=16, alpha=2  -- same physics, 4 averaging periods instead of
                                 2: the metric must not depend on the window
``H24`` / ``H32``  alpha=2, aspect ratio L/H=3 -- h-ladder for the observed
                                 order of accuracy of the *unsteady* error

Everything printed here is read back from the produced JSON, so the report and
figures cannot drift from the raw records.

Usage
-----
    PYTHONPATH=src python .../unsteady.py --out-dir artifacts/zou_he_unsteady
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

from tensorlbm.verification.zou_he_unsteady import (  # noqa: E402
    UnsteadyCase,
    analytic_reference,
    measured_gradient,
    nominal_gradient,
    phasor_metrics,
    run_oscillatory,
)

ARMS = [
    dict(name="alpha2", height=16, length=48, tau=0.65, womersley=2.0,
         warmup_periods=3, avg_periods=2),
    dict(name="alpha4", height=16, length=48, tau=0.65, womersley=4.0,
         warmup_periods=3, avg_periods=2),
    dict(name="alpha6", height=16, length=48, tau=0.65, womersley=6.0,
         warmup_periods=3, avg_periods=2),
    dict(name="alpha2_w4", height=16, length=48, tau=0.65, womersley=2.0,
         warmup_periods=3, avg_periods=4),
    dict(name="H24", height=24, length=72, tau=0.65, womersley=2.0,
         warmup_periods=2, avg_periods=2),
    dict(name="H32", height=32, length=96, tau=0.65, womersley=2.0,
         warmup_periods=2, avg_periods=2),
]


def _c(z: complex):
    return [float(complex(z).real), float(complex(z).imag)]


def _carr(a) -> list:
    return [_c(v) for v in np.asarray(a).ravel()]


def run_arm(spec: dict) -> dict:
    case = UnsteadyCase(**{k: v for k, v in spec.items() if k != "name"})
    res = run_oscillatory(case)
    m_fit = phasor_metrics(case, res["U"], res["g_fit"])
    m_nom = phasor_metrics(case, res["U"], res["g_nominal"])
    m_qs = phasor_metrics(case, res["U"], res["g_fit"], alpha=0.05)
    m_wrong = phasor_metrics(case, res["U"], res["g_fit"], alpha=1.5 * case.alpha)
    U_ref = analytic_reference(case, res["g_fit"])

    trace_steps = res["trace_steps"].astype(float)
    trace_u = res["trace_u"].astype(float)
    stride = max(1, trace_steps.size // 4000)
    return {
        "name": spec["name"],
        "case": {
            "height": case.height, "length": case.length, "tau": case.tau,
            "nu": case.nu, "drho": case.drho, "womersley": case.womersley,
            "warmup_periods": case.warmup_periods, "avg_periods": case.avg_periods,
            "omega": case.omega, "period": case.period, "alpha": case.alpha,
            "stokes_cells": case.stokes_cells,
            "acoustic_phase_deg": case.acoustic_phase_deg,
            "steps": res["steps"], "n_avg": res["n_avg"],
        },
        "metrics_fit": m_fit,
        "metrics_nominal": m_nom,
        "control_quasi_steady": {"amp_rel_l2": m_qs["amp_rel_l2"],
                                 "phase_rms_deg": m_qs["phase_rms_deg"]},
        "control_wrong_alpha": {"amp_rel_l2": m_wrong["amp_rel_l2"],
                                "phase_rms_deg": m_wrong["phase_rms_deg"]},
        "g_fit": _c(res["g_fit"]),
        "g_nominal": _c(nominal_gradient(case)),
        "x_spread": float(res["x_spread"]),
        "q_spread": float(res["q_spread"]),
        "prescription_residual": float(res["prescription_residual"]),
        "U": _carr(res["U"]),
        "U_ref": _carr(U_ref),
        "U_cols": _carr(res["U_cols"]),
        "fit_band_indices": list(res["fit_band_indices"]),
        "trace_steps": trace_steps[::stride].tolist(),
        "trace_u": trace_u[::stride].tolist(),
    }


def _observed_order(arms) -> dict:
    """p = log(e_coarse/e_fine)/log(h_coarse/h_fine) for the alpha=2 ladder."""
    lad = sorted([a for a in arms if a["name"] in ("alpha2", "H24", "H32")],
                 key=lambda a: a["case"]["height"])
    if len(lad) < 2:
        return {}
    out = {"levels": [], "orders": []}
    for a in lad:
        out["levels"].append({
            "H": a["case"]["height"],
            "amp_rel_l2": a["metrics_fit"]["amp_rel_l2"],
            "amp_rel_linf": a["metrics_fit"]["amp_rel_linf"],
            "phase_rms_deg": a["metrics_fit"]["phase_rms_deg"],
        })
    for lo, hi in zip(lad, lad[1:]):
        h1, h2 = lo["case"]["height"], hi["case"]["height"]
        e1, e2 = lo["metrics_fit"]["amp_rel_linf"], hi["metrics_fit"]["amp_rel_linf"]
        if e2 > 0 and e1 > 0:
            out["orders"].append({
                "H_coarse": h1, "H_fine": h2,
                "p_amp_linf": math.log(e1 / e2) / math.log(h1 / h2),
            })
    return out


def _figures(arms, path_prefix: str) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []

    # ---- fig 1: amplitude and phase profiles vs the exact solution ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for a in arms:
        if a["name"] not in ("alpha2", "alpha4", "alpha6"):
            continue
        U = np.array([complex(*v) for v in a["U"]])
        Ur = np.array([complex(*v) for v in a["U_ref"]])
        H = a["case"]["height"]
        y = (np.arange(1, H + 1) - 0.5) / H
        col = f"C{[0, 2, 3][['alpha2', 'alpha4', 'alpha6'].index(a['name'])]}"
        axes[0].plot(y, np.abs(U) / np.abs(Ur).max(), "-", color=col,
                     label=f"sim alpha={a['case']['alpha']:.0f}")
        axes[0].plot(y, np.abs(Ur) / np.abs(Ur).max(), "--", color=col,
                     alpha=0.6, label=f"Sexl alpha={a['case']['alpha']:.0f}")
        dphase = np.degrees(np.angle(U * np.conj(Ur)))
        axes[1].plot(y, dphase, "-", color=col,
                     label=f"alpha={a['case']['alpha']:.0f} "
                           f"(RMS {a['metrics_fit']['phase_rms_deg']:.2f} deg)")
    axes[0].set_xlabel("y / H"); axes[0].set_ylabel("|U| / max|U_ref|")
    axes[0].set_title("oscillatory amplitude profile")
    axes[1].set_xlabel("y / H"); axes[1].set_ylabel("phase error [deg]")
    axes[1].set_title("phase error vs exact Sexl solution")
    axes[1].axhline(0.0, color="k", lw=0.6)
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    f1 = f"{path_prefix}_profiles.png"; fig.tight_layout(); fig.savefig(f1, dpi=150)
    plt.close(fig); files.append(f1)

    # ---- fig 2: h-ladder of the unsteady error ------------------------------
    lad = sorted([a for a in arms if a["name"] in ("alpha2", "H24", "H32")],
                 key=lambda a: a["case"]["height"])
    if len(lad) >= 2:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        H = np.array([a["case"]["height"] for a in lad], dtype=float)
        e = np.array([a["metrics_fit"]["amp_rel_linf"] for a in lad])
        ax.loglog(H, e, "o-", label="L$\\infty$ amplitude error")
        ax.loglog(H, np.array([a["metrics_fit"]["amp_rel_l2"] for a in lad]),
                  "s--", label="L2 amplitude error")
        for h, ee in zip(H, e):
            ax.annotate(f"{100*ee:.2f}%", (h, ee), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
        ax.axhline(0.15, color="r", ls=":", label="15% acceptance")
        ax.axhline(0.03, color="orange", ls=":", label="3% target")
        ax.set_xlabel("H [cells]"); ax.set_ylabel("relative error")
        ax.set_title("alpha=2 h-ladder (unsteady gate)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
        f2 = f"{path_prefix}_ladder.png"; fig.tight_layout(); fig.savefig(f2, dpi=150)
        plt.close(fig); files.append(f2)

    # ---- fig 3: time trace (last two periods) + Lissajous -------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for a in arms:
        if a["name"] not in ("alpha2", "alpha4"):
            continue
        t = np.asarray(a["trace_steps"]); u = np.asarray(a["trace_u"])
        omega = a["case"]["omega"]
        keep = t > t.max() - 2.0 * a["case"]["period"]
        col = "C0" if a["name"] == "alpha2" else "C2"
        axes[0].plot(t[keep] / a["case"]["period"], u[keep], "-", color=col,
                     label=f"sim alpha={a['case']['alpha']:.0f}")
    axes[0].set_xlabel("t / period"); axes[0].set_ylabel("u_x at mid-channel")
    axes[0].set_title("periodic response (last two periods)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    for a in arms:
        if a["name"] not in ("alpha2", "alpha4"):
            continue
        t = np.asarray(a["trace_steps"]); u = np.asarray(a["trace_u"])
        omega = a["case"]["omega"]
        drho = a["case"]["drho"]
        drive = 0.5 * drho * np.cos(omega * t)
        col = "C0" if a["name"] == "alpha2" else "C2"
        axes[1].plot(drive, u, ".", ms=2, color=col,
                     label=f"alpha={a['case']['alpha']:.0f}")
    axes[1].set_xlabel("imposed end-density amplitude  drho/2 cos(wt)")
    axes[1].set_ylabel("u_x at mid-channel")
    axes[1].set_title("Lissajous: response vs driving")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    f3 = f"{path_prefix}_trace.png"; fig.tight_layout(); fig.savefig(f3, dpi=150)
    plt.close(fig); files.append(f3)

    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(REPO, "artifacts", "zou_he_unsteady"))
    ap.add_argument("--arms", default="")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if args.arms == "none":            # regenerate figures/JSON from shards only
        specs = []
    else:
        specs = [s for s in ARMS if not args.arms or s["name"] in args.arms.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    if os.path.exists(os.path.join(args.out_dir, "metrics.json")):
        with open(os.path.join(args.out_dir, "metrics.json")) as fh:
            results.update(json.load(fh).get("arms", {}))   # merge shards

    for spec in specs:
        print(f"[arm {spec['name']}] running ...", flush=True)
        rec = run_arm(spec)
        results[spec["name"]] = rec
        m = rec["metrics_fit"]
        print(f"[arm {spec['name']}] steps={rec['case']['steps']} "
              f"alpha={rec['case']['alpha']:.2f} "
              f"amp_L2={100*m['amp_rel_l2']:.3f}% amp_Linf={100*m['amp_rel_linf']:.3f}% "
              f"phase_RMS={m['phase_rms_deg']:.3f}deg "
              f"x_spread={rec['x_spread']:.1e} q_spread={rec['q_spread']:.1e}", flush=True)

    order = [s["name"] for s in ARMS]
    arms = [results[n] for n in order if n in results]
    payload = {
        "anchor": "XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI",
        "generated_by": "xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/unsteady.py",
        "arms": results,
        "ladder": _observed_order(arms),
        "acceptance": {"bound": 0.15, "target": 0.03, "phase_gate_deg": 3.0},
    }
    jf = os.path.join(args.out_dir, "metrics.json")
    with open(jf, "w") as fh:
        json.dump(payload, fh, indent=1)

    figs = _figures(arms, os.path.join(HERE, "figures", "unsteady"))
    print("wrote", jf)
    for f in figs:
        print("wrote", f)
    print(json.dumps(payload["ladder"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())