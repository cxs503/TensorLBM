#!/usr/bin/env python3
"""XVAL driver: DHC square-cavity natural convection (de Vahl Davis 1983) at Ra=1e5.

Anchor: XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY (round 2)
  -> oss_inventory / ~/oss-cache hold no thermal double-distribution LBM
     natural-convection reference (see oss_survey.json), so the external
     anchor is the published benchmark lineage, not an OSS checkout:
       de Vahl Davis (1983) Int. J. Numer. Methods Fluids 3:249-264, Table 1
       (Ra=1e5, Pr=0.71: Nu=4.519, u_max=34.72, v_max=68.60)

Zero local physics: every kernel comes from `tensorlbm.thermal`
(library entry point simulate_natural_convection: D2Q9 BGK velocity +
D2Q5 BGK temperature, Guo body force with Boussinesq buoyancy).

Run one grid, write a JSON artifact that analyze.py aggregates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.thermal import simulate_natural_convection  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ra", type=float, default=1e5)
    ap.add_argument("--pr", type=float, default=0.71)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=400000)
    ap.add_argument("--report-every", type=int, default=2000)
    ap.add_argument("--stop-tol", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("LBM_THREADS", "8")))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    torch.set_grad_enabled(False)

    res = simulate_natural_convection(
        nx=args.nx,
        ra=args.ra,
        pr=args.pr,
        tau=args.tau,
        steps=args.steps,
        device="cpu",
        report_every=args.report_every,
        stop_tol_nu=args.stop_tol,
        stop_patience=args.patience,
    )

    H = res["H"]
    alpha = res["alpha"]
    u_max_nd = res["u_max"] * H / alpha
    v_max_nd = res["v_max_y"] * H / alpha

    out = {
        "case": "dhc_cavity_natural_convection",
        "anchor": "XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY",
        "library_entry": "tensorlbm.thermal.simulate_natural_convection",
        "nx": args.nx,
        "ny": args.nx,
        "ra": args.ra,
        "pr": args.pr,
        "tau": args.tau,
        "tau_T": res["tau_T"],
        "nu": res["nu"],
        "alpha": alpha,
        "H": H,
        "g_beta": res["g_beta"],
        "ma_lattice": res["u_max"] / math.sqrt(1.0 / 3.0),
        "threads": args.threads,
        "steps_requested": args.steps,
        "report_every": args.report_every,
        "stop_tol_nu": args.stop_tol,
        "patience": args.patience,
        "stopped_early": res["stopped_early"],
        "steps_done": res["steps_done"],
        "elapsed_s": res["elapsed_s"],
        "last_resid_u": res["last_resid_u"],
        "last_resid_T": res["last_resid_T"],
        "nu_grad2": res["nu_grad2"],
        "nu_grad1": res["nu_grad1"],
        "nu_halfway": res["nu_halfway"],
        "nu_left_grad2": res["nu_left_grad2"],
        "nu_right_grad2": res["nu_right_grad2"],
        "u_max_lu": res["u_max"],
        "v_max_lu": res["v_max_y"],
        "u_max_nd": u_max_nd,
        "v_max_nd": v_max_nd,
        "y_at_u_max_x": res["y_at_u_max_x"],
        "x_at_v_max_y": res["x_at_v_max_y"],
        "T_min": res["T_min"],
        "T_max": res["T_max"],
        "nu_history": res["nu_history"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"[xval ra={args.ra:g} N={args.nx}] steps={res['steps_done']} "
        f"({res['elapsed_s']}s) Nu={res['nu_grad2']:.4f} "
        f"u_max_nd={u_max_nd:.3f} v_max_nd={v_max_nd:.3f} "
        f"resid_u={res['last_resid_u']:.2e} stopper={res['stopped_early']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())