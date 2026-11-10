#!/usr/bin/env python
"""Run the three-way LBM-DEM-FSI acceptance gates and build the figures.

Usage::

    python scripts/three_way_gates.py [--quick]

Writes ``runs/three_way/gates.json``, ``runs/three_way/scenario.json`` and
``docs/figs/three_way_*.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.three_way import gates  # noqa: E402
from tensorlbm.three_way.scenario import ThreeWayConfig, ThreeWaySimulation  # noqa: E402


def scenario_run(n_steps: int = 360, record_every: int = 2):
    cfg = ThreeWayConfig()
    sim = ThreeWaySimulation(cfg)
    contacts = []
    for i in range(n_steps):
        info = sim.step()
        if info["in_contact"]:
            contacts.append((i, info["contact_force"], info["y_contact"]))
        if i % record_every == 0:
            sim.record()
    return sim, contacts


def make_figures(sim, contacts, gate_out, out_dir: Path) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    figs = []
    s = sim.series
    t = s["t"]

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    ax[0].plot(t, s["stress_mid"], lw=1.2, label="mid-span (0.5 L)")
    ax[0].plot(t, s["stress_root"], lw=0.9, alpha=0.7, label="clamp (0.02 L)")
    for c in contacts:
        ax[0].axvspan(c[0] - 0.5, c[0] + 0.5, color="tab:red", alpha=0.06, lw=0)
    ax[0].set_ylabel(r"bending stress $\sigma$ [lattice]")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("whipping stress time history (red bands = ice-floe contact)")
    ax[1].plot(t, s["deflection_tip"], lw=1.1, color="tab:green")
    ax[1].set_xlabel("lattice step")
    ax[1].set_ylabel(r"tip deflection $w$ [lattice]")
    fig.tight_layout()
    p = out_dir / "three_way_whipping.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append((p, "Whipping stress at mid-span and clamp, with the DEM contact windows shaded; lower panel: beam tip deflection."))

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True)
    ax[0].plot(t, s["floe_x"], lw=1.1)
    ax[0].axhline(float(sim.cfg.beam.x_base), color="k", ls="--", lw=0.8, label="beam face")
    ax[0].set_ylabel("floe centre x [lattice]")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, s["floe_vx"], lw=1.1, color="tab:orange")
    ax[1].set_ylabel(r"surge velocity $v_x$")
    ax[1].set_xlabel("lattice step")
    fig.tight_layout()
    p = out_dir / "three_way_floe.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append((p, "DEM ice-floe trajectory and surge velocity: repeated slams against the flexible beam driven by the wave-equivalent forcing."))

    a = gate_out["gates"]["a"]
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.4))
    r = a["runs"]
    dx = [x["radius"] for x in r]
    ax[0].plot(dx, [100 * e for e in a["interface_errors"]], "o-")
    ax[0].axhline(2.0, color="r", ls="--", lw=0.9, label="2 % tolerance")
    ax[0].set_xlabel("body radius R [lattice cells]")
    ax[0].set_ylabel("interface closure error [%]")
    ax[0].legend(fontsize=8)
    ax[1].plot([x["re"] for x in r], [x["cd"] for x in r], "s-")
    ax[1].set_xlabel("Re = U D / nu")
    ax[1].set_ylabel(r"$C_D$")
    ax[1].set_xscale("log")
    ax[1].set_title("drag coefficient, grid refinement")
    fig.tight_layout()
    p = out_dir / "three_way_gate_a.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append((p, "Gate (a): interface closure error (momentum sink vs control-surface integral) and the drag coefficient across three grid resolutions."))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    c = gate_out["gates"]["c"]
    labels = ["interface work", "d(E_floe+E_beam)", "losses", "residual"]
    vals = [c["work_interface"], c["delta_energy"], c["losses"], c["residual"]]
    ax.bar(labels, vals, color=["tab:blue", "tab:green", "tab:orange", "tab:red"])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("energy [lattice]")
    ax.set_title("one-window energy ledger (floe + beam)")
    fig.tight_layout()
    p = out_dir / "three_way_ledger.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append((p, "Energy ledger of the floe + beam subsystem over one wave period: interface work versus stored energy, damping losses and the residual."))
    return figs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)

    out_root = ROOT / "runs" / "three_way"
    out_root.mkdir(parents=True, exist_ok=True)
    gates_out = gates.run_all(out_dir=out_root, quick=args.quick)

    sim, contacts = scenario_run(n_steps=240 if args.quick else 480)
    (out_root / "scenario.json").write_text(
        json.dumps(
            {
                "summary": sim.summary(),
                "contacts": contacts,
                "ledger": sim.energy_ledger(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    figs = make_figures(sim, contacts, gates_out, ROOT / "docs" / "figs")
    print(json.dumps({k: v["passed"] for k, v in gates_out["gates"].items()}, indent=1))
    print("aggregate:", "PASS" if gates_out["passed"] else "FAIL")
    print("figures:", [str(p) for p, _ in figs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())