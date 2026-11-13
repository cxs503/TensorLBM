#!/usr/bin/env python3
"""Figures for CASE-CAVITY-RE5000 from the run artefacts (no hand-typed numbers).

Usage:
    python cases/cavity_re5000/make_figures.py [--result result.json] [--out figs]

Figure 1: centreline profiles u(y) at x=L/2 and v(x) at y=L/2, normalised by
          the lid velocity.  Only what the artefact contains is plotted: the
          Ghia (1982) Table III/IV overlay is drawn **only** when a
          provenance-bearing reference is supplied with --ref-json (the
          fabricated table that used to ship with this case is gone, see
          docs/audit_m15543.md).
Figure 2: convergence history (max |du| per monitor interval, log scale) with
          the density bounds, i.e. the evidence for how far the run got from a
          steady state.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1] / "src"))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(CASE_DIR / "result.json"))
    ap.add_argument("--ref-json", default=None,
                    help="provenance-bearing Ghia reference (JSON) to overlay")
    ap.add_argument("--out", default=str(CASE_DIR / "figs"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = load(args.result)
    grids = payload["grids"]
    ordered = sorted(grids.items(), key=lambda kv: int(kv[0]))

    # ---------------- Figure 1: centreline profiles ----------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for n, rec in ordered:
        if rec.get("status") != "ok":
            continue
        axes[0].plot(np.asarray(rec["u_centerline"]), rec["y_centerline"],
                     label=f"n={n}", lw=1.5)
        axes[1].plot(rec["x_centerline"], np.asarray(rec["v_centerline"]),
                     label=f"n={n}", lw=1.5)

    if args.ref_json:
        from tensorlbm.verification.lid_driven_cavity.reference_data import (
            load_reference_json,
        )
        ref = load_reference_json(args.ref_json)
        axes[0].plot(ref.u, ref.y, "ko", ms=4, mfc="none", label="Ghia (1982) ref")
        axes[1].plot(ref.x, ref.v, "ko", ms=4, mfc="none", label="Ghia (1982) ref")

    axes[0].set_xlabel("u / U_lid")
    axes[0].set_ylabel("y / L")
    axes[0].set_title("vertical centreline x = L/2")
    axes[1].set_xlabel("x / L")
    axes[1].set_ylabel("v / U_lid")
    axes[1].set_title("horizontal centreline y = L/2")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Lid-driven cavity, collision = {payload.get('collision')}", fontsize=10)
    fig.tight_layout()
    f1 = out / "centerline_profiles.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    # ---------------- Figure 2: convergence evidence ----------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for n, rec in ordered:
        hist = rec.get("resid_history") or []
        if not hist:
            continue
        ax.semilogy([h["step"] for h in hist], [h["resid"] for h in hist],
                    "o-", ms=3, lw=1.4, label=f"n={n} (Re={rec['re']:g})")
    ax.set_xlabel("step")
    ax.set_ylabel(r"max $|\Delta u|$ per monitor interval  [lattice units]")
    ax.set_title("convergence history (no steady-state claim without a plateau)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f2 = out / "convergence_history.png"
    fig.savefig(f2, dpi=140)
    plt.close(fig)

    summary = {
        "figures": [str(f1.relative_to(CASE_DIR)), str(f2.relative_to(CASE_DIR))],
        "n_grids_plotted": len(ordered),
    }
    (out / "figures.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()