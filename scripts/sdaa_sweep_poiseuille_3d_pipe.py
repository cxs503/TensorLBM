#!/usr/bin/env python
"""SDAA sweep harness for ``benchmarks/verified/poiseuille_3d_pipe``.

Design rules (m74, round 2):

* **The case driver is not modified.**  This script imports
  ``run.py:run_case`` from the case directory and calls it with the case's
  own arguments.  All judging numbers therefore come from the case's own
  code path (zero hand-copying).
* **Device**: computation tensors live on ``'sdaa'`` (Torch-SDAA).  The
  device evidence line ``torch.sdaa.get_device_name(0)`` is captured in
  this process and persisted to the artifact JSON.
* **Criteria are the case's own (README) criteria, unchanged** — no
  threshold is relaxed.  One grid tier only (R=20, steps at the README
  lower bound); the multi-grid ladder is out of scope for this round.
* **Verdict is written immediately after the run** so a verdict exists even
  if documentation is appended later.

Usage::

    python scripts/sdaa_sweep_poiseuille_3d_pipe.py            # R=20, 20000 steps
    python scripts/sdaa_sweep_poiseuille_3d_pipe.py --smoke    # 200 steps, path check only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "benchmarks" / "verified" / "poiseuille_3d_pipe"

# --- original acceptance criteria (README), NOT tuned -----------------------
# primary: radially-averaged profile max relative error in the central region
#          vs u(r)=2*u_in*(1-(r/R_eff^Q)^2) must be <= 3%
PROFILE_MAX_PCT = 3.0
# secondary disclosed channel: domain centre-line |u_center-2u_in|/2u_in <= 3%
#          (README "名义解析直接通道" — quoted as -1.12% at R=20)
NOMINAL_UMax_PCT = 3.0
# hydraulic radius reproducibility: documented R_eff^Q - R = +0.11 (R=20:
#          +0.109).  Tolerance is a reproducibility band, not a physics门限.
REFF_Q_OFFSET = 0.11
REFF_Q_TOL = 0.05
# identity by construction: Q/Q_ana^Q == 1
Q_RATIO_TOL = 1e-6


def load_case_module():
    sys.path.insert(0, str(REPO / "benchmarks"))
    spec = importlib.util.spec_from_file_location("pipe_case_run", CASE_DIR / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def device_evidence() -> dict:
    import torch

    ev: dict = {"torch_version": torch.__version__}
    try:
        import torch_sdaa

        ev["torch_sdaa_version"] = getattr(torch_sdaa, "__version__", "unknown")
        ev["sdaa_is_available"] = bool(torch.sdaa.is_available())
        ev["sdaa_device_count"] = int(torch.sdaa.device_count())
        ev["device_name_0"] = str(torch.sdaa.get_device_name(0))
        probe = torch.zeros(4, device=torch.device("sdaa:0"))
        ev["tensor_device_type"] = probe.device.type
        ev["tensor_device_str"] = str(probe.device)
        ev["device_line"] = (
            f"torch.sdaa.get_device_name(0) = {ev['device_name_0']} | "
            f"torch_sdaa {ev['torch_sdaa_version']} | torch {torch.__version__} | "
            f"is_available={ev['sdaa_is_available']} | count={ev['sdaa_device_count']} | "
            f"probe tensor on {ev['tensor_device_str']}"
        )
    except Exception as exc:  # pragma: no cover - device absent
        ev["sdaa_is_available"] = False
        ev["error"] = f"{type(exc).__name__}: {exc}"
        ev["device_line"] = f"SDAA UNAVAILABLE: {ev['error']}"
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--mode", choices=["velocity", "pressure"], default="velocity")
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--u-in", type=float, default=0.02)
    ap.add_argument("--min-steps", type=int, default=20000)  # README lower bound
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--compile-mode", default="eager",
                    choices=["eager", "default", "max-autotune-no-cudagraphs"])
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="200-step path/device check")
    ap.add_argument("--out-json", default=str(REPO / "artifacts" / "sdaa_poiseuille_3d_pipe.json"))
    ap.add_argument("--case-out", default=str(REPO / "artifacts" / "sdaa_poiseuille_3d_pipe_case.json"))
    args = ap.parse_args()

    if args.smoke:
        args.min_steps, args.max_steps = 200, 200

    ev = device_evidence()
    print(f"[device] {ev['device_line']}", flush=True)

    import torch

    device = torch.device(args.device)
    wall0 = time.time()
    record: dict = {
        "case": "poiseuille_3d_pipe",
        "benchmark_dir": str(CASE_DIR.relative_to(REPO)),
        "round": "m74 round-2 (single tier R=20)",
        "device": ev,
        "config": {
            "R": args.R,
            "mode": args.mode,
            "tau": args.tau,
            "u_in": args.u_in,
            "min_steps": args.min_steps,
            "max_steps": args.max_steps,
            "compile_mode": args.compile_mode,
            "compile_mode_note": (
                "eager passthrough through benchmarks/compile_route.route_step; "
                "Torch-SDAA 3.1.1a3 teco_inductor cannot compile the LBM step chain "
                "(known L5 defect, documented in compile_route docstring), so "
                "route_step('default') would fall back to eager anyway — choosing "
                "'eager' explicitly only removes the failed-compile wall clock, "
                "numerics identical."
            ),
            "triton_interpreter": (
                "not needed: --compile-mode eager does not enter torch.compile/"
                "inductor, so TRITON_INTERPRET=1 is not required. No triton kernel "
                "is JIT-traced on this path."
            ),
            "device": args.device,
            "seed": args.seed,
            "case_json_reference": ["case_R20.json", "case_R40.json"],
            "steps_reference": "README lower bound 20000 (steady drift <1e-5 gate)",
        },
    }

    run_ok = False
    result: dict = {}
    try:
        mod = load_case_module()
        result = mod.run_case(
            args.R,
            args.tau,
            args.u_in,
            0.04,  # u_max (pressure-mode target; unused in velocity mode)
            args.mode,
            args.min_steps,
            args.max_steps,
            args.case_out,
            device,
            args.seed,
            compile_mode=args.compile_mode,
        )
        run_ok = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        print(f"[run] FAILED: {record['error']}", flush=True)

    wall = time.time() - wall0

    gates: list[dict] = []
    metrics: dict = {}
    if run_ok:
        g = lambda k, d=None: result.get(k, d)  # noqa: E731
        metrics = {
            "R": g("R"), "Re": g("Re"), "Ma": g("Ma"), "n_steps": g("n_steps"),
            "steady": g("steady"), "finite": g("finite"),
            "nx": g("nx"), "ny": g("ny"), "nz": g("nz"),
            "R_eff_Q": g("R_eff_Q", g("R_eff_from_Q")), "R_fit": g("R_fit"),
            "Q_ratio_Q": g("Q_ratio_Q", g("Q_ratio")), "Q": g("Q"), "Q_ana": g("Q_ana"),
            "max_rel_bin_central_Q_pct": g("max_rel_bin_central_Q_pct"),
            "l2_rel_err_Q": g("l2_rel_err_Q"),
            "max_rel_err_central_Q_pct": g("max_rel_err_central_Q_pct"),
            "u_max_err_pct": g("u_max_err_pct"),
            "u_center_pred_err_pct": g("u_center_pred_err_pct"),
            "fd_max_rel_dev_pct": g("fd_max_rel_dev_pct"),
            "mass_drift_pct": g("mass_drift_pct"),
            "u_max_dp_err_pct": g("u_max_dp_err_pct"),
            "elapsed_s": g("elapsed_s"),
            "voxels": (g("nx") or 0) * (g("ny") or 0) * (g("nz") or 0),
        }

        def add_gate(name, value, ok, crit, note=""):
            gates.append({
                "gate": name,
                "value": value,
                "criterion": crit,
                "verdict": "PASS" if ok else "FAIL",
                "note": note,
            })

        # G1 primary profile criterion (README, unchanged 3%)
        v = metrics["max_rel_bin_central_Q_pct"]
        add_gate(
            "G1 radial-averaged profile max rel err (R_eff^Q, central region)",
            v, v is not None and v <= PROFILE_MAX_PCT,
            f"<= {PROFILE_MAX_PCT}%  (README primary metric)",
            "absolute normalization U_max=2*u_in; comparison radius from measured Q",
        )
        # G2 nominal analytic direct channel (README disclosure, 3% gate)
        v = metrics["u_max_err_pct"]
        add_gate(
            "G2 centre-line u_center vs nominal analytic U_max=2*u_in",
            v, v is not None and abs(v) <= NOMINAL_UMax_PCT,
            f"|err| <= {NOMINAL_UMax_PCT}%  (README nominal direct channel)",
            "no radius inversion at all — strictest literal reading of the case",
        )
        # G3 numerical sanity: finite fields
        add_gate("G3 fields finite", metrics["finite"], bool(metrics["finite"]), "True")
        # G4 steady state per README (drift <1e-5 in a 2000-step window)
        add_gate(
            "G4 steady criterion (u_max drift <1e-5 / 2000 steps)",
            metrics["steady"], bool(metrics["steady"]),
            "steady == True at min_steps <= n_steps <= max_steps",
            f"n_steps={metrics['n_steps']} (min {args.min_steps}, max {args.max_steps})",
        )
        # G5 hydraulic-radius reproducibility (documented geometric constant)
        reff = metrics["R_eff_Q"]
        off = (reff - args.R) if isinstance(reff, (int, float)) else None
        add_gate(
            "G5 hydraulic radius R_eff^Q - R reproducibility",
            off, off is not None and abs(off - REFF_Q_OFFSET) <= REFF_Q_TOL,
            f"{REFF_Q_OFFSET:+.3f} ± {REFF_Q_TOL} (README: +0.109 @R=20)",
            "independent integral observable (flow rate), not a profile fit",
        )
        # G6 identity Q/Q_ana^Q == 1
        v = metrics["Q_ratio_Q"]
        add_gate(
            "G6 flow-rate identity Q/Q_ana^Q",
            v, v is not None and abs(v - 1.0) <= Q_RATIO_TOL,
            f"|ratio-1| <= {Q_RATIO_TOL:g} (identity by construction)",
        )
        # G7 L2 norm of radial profile (case's own secondary report)
        v = metrics["l2_rel_err_Q"]
        add_gate(
            "G7 radial profile L2 rel err (R_eff^Q)",
            v, v is not None and v <= PROFILE_MAX_PCT,
            f"<= {PROFILE_MAX_PCT}% (same 3% budget as G1)",
            "reported as-is by the case; same numeric budget as the primary metric",
        )

    passed = run_ok and all(x["verdict"] == "PASS" for x in gates)
    record.update({
        "metrics": metrics,
        "gates": gates,
        "n_gates": len(gates),
        "n_pass": sum(1 for x in gates if x["verdict"] == "PASS"),
        "verdict": ("PASS" if passed else "FAIL") if run_ok else "FAIL",
        "verdict_reason": (
            "all original criteria met on SDAA, single tier R=20"
            if passed else
            ("one or more original criteria failed — see gates" if run_ok
             else f"run did not complete: {record.get('error')}")
        ),
        "wall_clock_s": round(wall, 2),
        "throughput_steps_per_s": (
            round((metrics.get("n_steps") or 0) / wall, 1) if wall > 0 and run_ok else None
        ),
        "scope_note": (
            "Single grid tier only (R=20, README step lower bound). The README "
            "multi-grid convergence ladder (R=20 -> R=40) is NOT executed in this "
            "round, so the case-level 'verified' attribute of the full ladder is "
            "not claimed here; this artifact records the R=20 tier verdict under "
            "the unchanged 3% criterion."
        ),
    })

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False))

    print(json.dumps({
        "verdict": record["verdict"],
        "n_pass": record.get("n_pass"),
        "n_gates": record.get("n_gates"),
        "wall_clock_s": record["wall_clock_s"],
        "primary_max_rel_pct": metrics.get("max_rel_bin_central_Q_pct"),
        "u_max_err_pct": metrics.get("u_max_err_pct"),
        "n_steps": metrics.get("n_steps"),
        "steady": metrics.get("steady"),
        "device_line": ev["device_line"],
        "artifact": str(out),
    }, indent=2, ensure_ascii=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())