"""TGV anchor harness for the graph-unroll BPTT selection pipeline.

This module is the *library entry* that turns the round-2 cross-validation
record (``docs/XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR.md``) into a callable,
reusable capability:

    >>> from tensorlbm import AnchorConfig, run_tgv_anchor
    >>> report = run_tgv_anchor(AnchorConfig.quick())

One call does three things on the 2-D Taylor-Green vortex (TGV):

1. **External anchor.**  Runs the plain BGK solver on a ladder of grids and
   compares every quadratic diagnostic with the exact solution
   (:func:`tensorlbm.les_graph_unroll.reference2d.tgv_analytic_ke`,
   :func:`...tgv_analytic_enstrophy`).  Without this step the LES reference is
   self-referential: a bug in the reference solver would be invisible.
2. **BPTT selection.**  Builds the coarse-grid :class:`SelectionCase` against
   the *filtered* fine-grid reference and trains the sub-grid constant
   ``theta = Cs`` (or ``Sc``) by backpropagation through the unrolled time
   loop, with a derivative-free sweep as an independent control.
3. **Held-out judgement.**  Scores the selected model on a snapshot no
   objective ever saw, and returns a ``ref``/``got``/``rel_err`` row table that
   the 15 % owner bar is applied to.

Every number is deterministic (no random seeds) and is returned as plain JSON
types, so callers (the XVAL driver, the verification suite, a platform
pipeline) never hand-copy values.

Command line entry (writes the same report to disk)::

    python -m tensorlbm.les_graph_unroll.tgv_anchor --quick --out anchor.json

The XVAL driver ``scripts/xval_r2_selection_bptt_tgv.py`` is the heavyweight
reproduction (12 arms, 2 closures, 3 objectives, 2 starts); this module is the
small, importable surface that a solver or platform calls.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import reference2d as ref2d
from .selection import (
    SelectionCase,
    bptt_select,
    holdout_metrics,
    objective_loss,
    sweep_select,
)

__all__ = [
    "AnchorConfig",
    "tau_for_grid",
    "analytic_rows",
    "grid_convergence",
    "build_case",
    "run_tgv_anchor",
    "main",
]

OWNER_BAR = 0.15  # owner 2026 precision standard: > 15 % is not acceptable


@dataclass(frozen=True)
class AnchorConfig:
    """Fixed, pre-declared workload of the TGV anchor (no tuning knobs used twice)."""

    u0: float = 0.1                     # lattice reference velocity
    re: float = 500.0                   # physical Reynolds number
    nx_les: int = 32                    # coarse (LES) grid
    nx_ref: int = 128                   # fine-grid reference
    n_steps: int = 200                  # unroll length
    train_steps: tuple = (50, 100, 150)  # snapshots the objective may see
    holdout_step: int = 200             # snapshot no objective sees
    closure: str = "smagorinsky"        # "smagorinsky" | "oss_proxy"
    objective: str = "energy"           # "field" | "energy" | "enstrophy"
    theta_init: float = 0.02
    iters: int = 20
    lr: float = 0.1
    sweep: tuple = tuple(round(0.025 * i, 4) for i in range(25))

    @classmethod
    def quick(cls) -> "AnchorConfig":
        """Small, test-sized variant (~2 s) with the same code path."""
        return cls(
            nx_les=32,
            nx_ref=64,
            n_steps=40,
            train_steps=(10, 20, 30),
            holdout_step=40,
            iters=8,
            sweep=tuple(round(0.05 * i, 4) for i in range(13)),
        )


def tau_for_grid(cfg: AnchorConfig, nx: int) -> float:
    """BGK relaxation time that realises ``Re`` at ``u0`` on an ``nx`` grid."""
    return 0.5 + 3.0 * (cfg.u0 / cfg.re) / (2.0 * math.pi / nx) ** 2


def _tgv_fields(nx: int, u0: float):
    x = np.arange(nx) * (2.0 * math.pi / nx)
    X = np.broadcast_to(x.reshape(nx, 1), (nx, nx))
    Y = np.broadcast_to(x.reshape(1, nx), (nx, nx))
    return u0 * np.sin(X) * np.cos(Y), -u0 * np.cos(X) * np.sin(Y)


def _ke(ux, uy) -> float:
    return 0.5 * float(np.mean(ux * ux + uy * uy))


def _enstrophy_lattice(ux, uy) -> float:
    """Same convention as ``selection.enstrophy`` (central differences, no 1/dx)."""
    wx = 0.5 * (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0))
    wy = 0.5 * (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1))
    w = wx - wy
    return 0.5 * float(np.mean(w * w))


def analytic_rows(cfg: AnchorConfig, nx: int, n_steps: int | None = None) -> dict:
    """Run the plain BGK solver at ``nx`` and score it against the exact TGV."""
    n_steps = int(cfg.n_steps if n_steps is None else n_steps)
    f0, _, _ = ref2d.tgv_ic(nx, cfg.u0)
    tau = tau_for_grid(cfg, nx)
    _, snaps = ref2d.run_reference(f0, n_steps, tau, snapshot_steps=(n_steps,))
    ux, uy = snaps[n_steps]
    t = float(n_steps)
    ke = _ke(ux, uy)
    ke_a = ref2d.tgv_analytic_ke(cfg.u0, cfg.u0 / cfg.re, t)
    z = _enstrophy_lattice(ux, uy)
    z_a = ref2d.tgv_analytic_enstrophy(cfg.u0, cfg.u0 / cfg.re, t, 2.0 * math.pi / nx)
    return {
        "grid": int(nx),
        "step": int(n_steps),
        "tau": tau,
        "ke": ke,
        "ke_ref_analytic": ke_a,
        "ke_rel_err": abs(ke - ke_a) / ke_a,
        "enstrophy": z,
        "enstrophy_ref_analytic": z_a,
        "enstrophy_rel_err": abs(z - z_a) / z_a,
    }


def grid_convergence(cfg: AnchorConfig, grids=(32, 64, 128, 256)) -> list:
    """No-model grid ladder vs the analytic TGV (the external-anchor evidence)."""
    rows = []
    for nx in grids:
        if nx % 2:
            continue
        rows.append(analytic_rows(cfg, int(nx)))
    return rows


def build_case(cfg: AnchorConfig) -> SelectionCase:
    """Coarse LES case whose reference is the *filtered* fine-grid solution."""
    n_steps = int(cfg.n_steps)
    steps = tuple(sorted(set(int(s) for s in cfg.train_steps) | {int(cfg.holdout_step)}))
    f0, _, _ = ref2d.tgv_ic(cfg.nx_ref, cfg.u0)
    tau_ref = tau_for_grid(cfg, cfg.nx_ref)
    _, snaps = ref2d.run_reference(f0, n_steps, tau_ref, snapshot_steps=steps)
    ratio = cfg.nx_ref // cfg.nx_les
    ref = {k: (ref2d.box_filter(ux, ratio), ref2d.box_filter(uy, ratio))
           for k, (ux, uy) in snaps.items()}

    f0n, _, _ = ref2d.tgv_ic(cfg.nx_les, cfg.u0)
    return SelectionCase(
        f0=torch.as_tensor(f0n, dtype=torch.float64),
        ref=ref,
        tau0=tau_for_grid(cfg, cfg.nx_les),
        n_steps=n_steps,
        closure=cfg.closure,
        train_steps=tuple(int(s) for s in cfg.train_steps),
        holdout_step=int(cfg.holdout_step),
    )


def _row(rid: str, ref: float, got: float, note: str = "") -> dict:
    rel = abs(float(got) - float(ref)) / max(abs(float(ref)), 1e-300)
    return {
        "id": rid,
        "ref": float(ref),
        "got": float(got),
        "abs_err": abs(float(got) - float(ref)),
        "rel_err": rel,
        "threshold": OWNER_BAR,
        "pass": bool(rel < OWNER_BAR),
        "note": note,
    }


def run_tgv_anchor(cfg: AnchorConfig | None = None, *, convergence=False) -> dict:
    """Full anchor + selection + held-out report (all JSON-serialisable)."""
    cfg = AnchorConfig() if cfg is None else cfg
    t0 = time.time()

    # 1) external anchor: fine-grid reference vs the exact TGV
    ref_row = analytic_rows(cfg, cfg.nx_ref)
    conv_rows = grid_convergence(cfg, grids=(cfg.nx_les, cfg.nx_ref)) if convergence else []

    # 2) BPTT selection against the filtered reference
    case = build_case(cfg)
    bptt = bptt_select(case, cfg.objective, cfg.theta_init, iters=cfg.iters, lr=cfg.lr)
    ctrl = sweep_select(case, cfg.objective, cfg.sweep)
    with torch.no_grad():
        loss_nomodel = float(objective_loss(case, cfg.objective, 0.0))

    # 3) held-out judgement (t = holdout_step, never trained on)
    hold = holdout_metrics(case, bptt["theta"])
    hold0 = holdout_metrics(case, 0.0)

    # analytic score of the *coarse* run (selected and no-model) at the hold-out
    analytic_les = analytic_rows(cfg, cfg.nx_les)
    ke_analytic_les_nomodel = analytic_les["ke"]
    ke_analytic_les_sel = hold["ke"]

    rows = [
        _row("anchor-ref-ke", ref_row["ke_ref_analytic"], ref_row["ke"],
             f"fine grid {cfg.nx_ref}^2 vs exact TGV K(t)"),
        _row("anchor-ref-enstrophy", ref_row["enstrophy_ref_analytic"],
             ref_row["enstrophy"], f"fine grid {cfg.nx_ref}^2 vs exact TGV Z(t)"),
        _row("selected-holdout-ke-vs-filtered-ref", hold["ke_ref"], hold["ke"],
             "BPTT-selected LES vs filtered fine-grid reference (held-out step)"),
        _row("selected-holdout-ke-vs-analytic", ref2d.tgv_analytic_ke(
            cfg.u0, cfg.u0 / cfg.re, float(cfg.holdout_step)), ke_analytic_les_sel,
            "BPTT-selected LES vs exact TGV K(t)"),
        _row("nomodel-holdout-ke-vs-filtered-ref", hold0["ke_ref"], hold0["ke"],
             "theta=0 control vs filtered reference (identifiability control)"),
        _row("selected-holdout-enstrophy-vs-filtered-ref", hold["z_ref"], hold["z"],
             "BPTT-selected LES enstrophy vs filtered reference"),
    ]
    if conv_rows:
        for r in conv_rows:
            rows.append(_row(f"convergence-{r['grid']}-ke",
                             r["ke_ref_analytic"], r["ke"],
                             f"plain BGK {r['grid']}^2 vs exact TGV K(t)"))

    report = {
        "anchor": "XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR",
        "config": {
            "u0": cfg.u0, "re": cfg.re, "ma": cfg.u0 * math.sqrt(3.0),
            "nu_phys": cfg.u0 / cfg.re, "nx_les": cfg.nx_les, "nx_ref": cfg.nx_ref,
            "n_steps": cfg.n_steps, "train_steps": list(cfg.train_steps),
            "holdout_step": cfg.holdout_step, "closure": cfg.closure,
            "objective": cfg.objective, "theta_init": cfg.theta_init,
            "iters": cfg.iters, "lr": cfg.lr, "sweep": list(cfg.sweep),
            "tau_les": tau_for_grid(cfg, cfg.nx_les), "tau_ref": tau_for_grid(cfg, cfg.nx_ref),
        },
        "analytic_anchor": {"fine_grid": ref_row, "coarse_grid": analytic_les,
                            "convergence": conv_rows},
        "selection": {
            "theta_selected": bptt["theta"],
            "loss_selected": bptt["loss"],
            "loss_at_start": bptt["loss_init"],
            "loss_no_model": loss_nomodel,
            "sweep_argmin_theta": ctrl["theta"],
            "sweep_argmin_loss": ctrl["loss"],
            # When the derivative-free argmin sits on the lower bound (theta=0,
            # i.e. "no model wins") a relative theta gap is meaningless; in that
            # case the honest agreement statistic is the relative loss gap.
            "bounds_hit": bool(ctrl["theta"] <= 1e-9),
            "theta_rel_gap": (None if ctrl["theta"] <= 1e-9 else
                              abs(bptt["theta"] - ctrl["theta"]) / abs(ctrl["theta"])),
            "loss_gap_rel": abs(bptt["loss"] - ctrl["loss"])
            / max(abs(ctrl["loss"]), 1e-300),
        },
        "holdout": {"selected": hold, "no_model": hold0},
        "benchmark": {"threshold": OWNER_BAR, "rows": rows,
                      "all_pass": all(r["pass"] for r in rows)},
        "repro_cmd": ("python -m tensorlbm.les_graph_unroll.tgv_anchor "
                      + ("--quick " if cfg == AnchorConfig.quick() else "") + "--out anchor.json"),
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "dtype": "float64"},
        "wall_seconds": time.time() - t0,
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="test-sized workload (~2 s)")
    ap.add_argument("--convergence", action="store_true", help="add the 32/64/128/256 grid ladder")
    ap.add_argument("--out", default="artifacts/xval_r2_selection_bptt_tgv/anchor.json")
    a = ap.parse_args(argv)
    cfg = AnchorConfig.quick() if a.quick else AnchorConfig()
    report = run_tgv_anchor(cfg, convergence=a.convergence)
    path = a.out
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps(report["benchmark"], indent=2))
    print("wrote", path)
    return 0 if report["benchmark"]["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())