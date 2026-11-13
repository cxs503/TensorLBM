#!/usr/bin/env python3
"""Adversarial probe of the delivered claim "planted-constant recovery 8.0% < 15%".

Question: is the 8.0% gap a *method* limit (identifiability) or an *optimiser
budget* limit (20 Adam iterations)?  Same twin problem, same code, more
iterations.  Reference optimum from the derivative-free search is the control.

PYTHONPATH=<worktree>/src python3 docs/board_audit/m17966/probe_bptt_budget.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from tensorlbm.les_graph_unroll import reference2d as ref
from tensorlbm.les_graph_unroll.calibrate import (
    CalibProblem,
    bptt_calibrate,
    field_loss,
    golden_minimize,
)
from tensorlbm.les_graph_unroll.solver2d import macroscopic, unroll

NX, U0, RE, T = 32, 0.1, 500.0, 200
SNAPS = (50, 100, 150, 200)
TRUTH = 0.17


def main() -> int:
    tau = ref.tau_for_re(NX, U0, RE)
    f0, _, _ = ref.tgv_ic(NX, u0=U0)
    f0t = torch.as_tensor(f0, dtype=torch.float64)
    with torch.no_grad():
        _, states = unroll(f0t, T, tau, TRUTH, snapshots=SNAPS)
        snaps = {}
        for k, fs in states.items():
            _, ux, uy = macroscopic(fs)
            snaps[k] = (ux.clone(), uy.clone())
    prob = CalibProblem(f0=f0t, ref_snapshots=snaps, tau0=tau, n_steps=T)

    out = {"truth": TRUTH, "tau0": tau, "n_steps": T, "runs": {}}
    for iters in (20, 40, 80):
        r = bptt_calibrate(prob, 0.02, iters=iters, lr=0.1)
        out["runs"][f"iters{iters}"] = {
            "cs": r["cs"], "loss": r["loss"], "loss_init": r["loss_init"],
            "recovery_err_pct": abs(r["cs"] - TRUTH) / TRUTH * 100.0,
        }
    g = golden_minimize(prob, 0.005, 0.6, iters=25)
    out["control_golden"] = {
        "cs": g["cs"], "loss": g["loss"],
        "recovery_err_pct": abs(g["cs"] - TRUTH) / TRUTH * 100.0,
    }
    out["loss_at_truth"] = float(field_loss(prob, TRUTH))
    out["loss_at_bptt20"] = out["runs"]["iters20"]["loss"]
    print(json.dumps(out, indent=2, default=float))
    dst = Path("artifacts/m17966/probe_bptt_budget.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2, default=float))
    print(f"[probe] wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())