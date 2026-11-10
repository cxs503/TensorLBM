#!/usr/bin/env python3
"""Machine verification of node P151-N01 (D3Q27 activation) — gates (a)-(i).

Reads only JSON/artifacts written by the run scripts (plus two in-process
analytic recomputations that are cheap and float64):

  runs/tgv_convergence_d3q27.json        gate (f)
  runs/tgv_convergence_d3q19.json        gate (f) control
  runs/sphere_<lattice>_re<Re>_<extrap>.json   gates (c)(d)(e)(h)

Writes ``gate/verdict.json`` and exits non-zero iff any mandatory gate fails.
No number is transcribed by hand: every value is read from a machine artifact
or recomputed from the library.

Usage: python benchmarks/d3q27_activation/verify.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
GATES = os.path.join(HERE, "gate")

THR = {
    "a_isotropy_max": 1e-14,
    "b_mass_drift_max": 1e-12,
    "c_re100_cd_err_pct": 10.0,
    "c_d3q19_control_expected_above_pct": 15.0,
    "d_re200_cd_err_pct": 15.0,
    "e_improvement_factor_min": 1.2,
    "f_order_min": 1.8,
    "g_degenerate_max": 1e-10,
    "h_steps": 10000,
    "h_plateau_drift_pct_max": 2.0,
    "i_tests_min": 18,
    "i_negative_controls_min": 3,
}


def gate_a():
    """Recompute the D3Q27 equilibrium isotropy residuals in float64."""
    from tensorlbm.d3q27 import C as C27
    from tensorlbm.d3q27 import W_EXACT64, equilibrium27

    c = C27.to(torch.float64)
    w = W_EXACT64.to(torch.float64)
    zeros = torch.zeros((4, 4, 4), dtype=torch.float64)
    feq = equilibrium27(torch.ones_like(zeros), zeros, zeros, zeros)
    worst_m3 = 0.0
    for idx in ((0, 0, 0), (0, 1, 2), (0, 0, 1), (1, 1, 2), (2, 2, 2)):
        prod = torch.ones_like(feq)
        for axis in idx:
            prod = prod * c[:, axis].view(27, 1, 1, 1)
        worst_m3 = max(worst_m3, float((feq * prod).abs().sum()))
    delta = torch.eye(3, dtype=torch.float64)
    worst_m4 = 0.0
    for a in range(3):
        for b in range(3):
            for d in range(3):
                for e in range(3):
                    lhs = float(
                        (w * c[:, a] * c[:, b] * c[:, d] * c[:, e]).sum()
                    )
                    rhs = (
                        delta[a, b] * delta[d, e]
                        + delta[a, d] * delta[b, e]
                        + delta[a, e] * delta[b, d]
                    ) / 9.0
                    worst_m4 = max(worst_m4, abs(lhs - float(rhs)))
    m6 = float((w * c[:, 0] ** 2 * c[:, 1] ** 2 * c[:, 2] ** 2).sum())
    value = max(worst_m3, worst_m4, abs(m6 - 1.0 / 27.0))
    return {
        "gate": "a_equilibrium_isotropy",
        "value": value,
        "threshold": THR["a_isotropy_max"],
        "pass": value < THR["a_isotropy_max"],
        "detail": {"rank3_rest": worst_m3, "rank4": worst_m4, "rank6_minus_1_27": abs(m6 - 1.0 / 27.0)},
    }


def gate_b():
    """Recompute the periodic mass-conservation residual (200 D3Q27 steps)."""
    from tensorlbm.d3q27 import collide_bgk27, equilibrium27
    from tensorlbm.lbm_step_d3q27 import lbm_step_correct_27

    n = 8
    z, y, x = torch.meshgrid(*[torch.arange(n, dtype=torch.float64)] * 3, indexing="ij")
    solid = ((x - 3.5) ** 2 + (y - 3.5) ** 2 + (z - 3.5) ** 2) < 4.0
    ones = torch.ones_like(x)
    f = equilibrium27(ones, torch.zeros_like(x), 0.02 * torch.sin(x) * torch.cos(y), torch.zeros_like(x))
    m0 = float(f.sum())
    for _ in range(200):
        f = lbm_step_correct_27(f, collide_bgk27, 0.6, solid, 0.0, far_field_bc_fn=None)
    drift = abs(float(f.sum()) - m0) / m0
    return {
        "gate": "b_mass_conservation",
        "value": drift,
        "threshold": THR["b_mass_drift_max"],
        "pass": drift < THR["b_mass_drift_max"],
        "detail": {"cells": n**3, "steps": 200, "dtype": "float64"},
    }


def gate_g():
    """Recompute the D3Q19 degeneracy residual (macroscopic mapping)."""
    from tensorlbm.d3q19 import C as C19
    from tensorlbm.d3q19 import macroscopic3d
    from tensorlbm.d3q27 import C as C27
    from tensorlbm.d3q27 import macroscopic27

    m = [next(q for q in range(27) if C27[q].tolist() == C19[i].tolist()) for i in range(19)]
    f19 = torch.rand((19, 6, 5, 4), dtype=torch.float64)
    f27 = torch.zeros((27, 6, 5, 4), dtype=torch.float64)
    for i, q in enumerate(m):
        f27[q] = f19[i]
    a = macroscopic3d(f19)
    b = macroscopic27(f27)
    res = max(float((x - y).abs().max()) for x, y in zip(a, b))
    return {
        "gate": "g_d3q19_degenerate_residual",
        "value": res,
        "threshold": THR["g_degenerate_max"],
        "pass": res < THR["g_degenerate_max"],
        "detail": {"embedding": m},
    }


def load_sphere(lattice, re, extrap):
    p = os.path.join(RUNS, f"sphere_{lattice}_re{re}_{extrap}.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def gate_cdeh():
    out = []
    r100_q = load_sphere("d3q27", 100, "quadratic")
    r100_q19 = load_sphere("d3q19", 100, "quadratic")
    r200_q = load_sphere("d3q27", 200, "quadratic")
    r200_q19 = load_sphere("d3q19", 200, "quadratic")

    if r100_q is None:
        out.append({"gate": "c_re100_d3q27", "pass": False, "value": None, "detail": "missing artifact"})
    else:
        out.append(
            {
                "gate": "c_re100_d3q27_cd_error_pct",
                "value": r100_q["cd_error_pct"],
                "threshold": THR["c_re100_cd_err_pct"],
                "pass": r100_q["cd_error_pct"] < THR["c_re100_cd_err_pct"]
                and r100_q["finite"]
                and abs(r100_q["plateau_drift_pct"]) < 2.0,
                "detail": {
                    "cd": r100_q["cd_tail_mean"],
                    "cd_ref": r100_q["cd_ref_schiller_naumann"],
                    "plateau_drift_pct": r100_q["plateau_drift_pct"],
                    "mass_drift_ppm": r100_q["mass_drift_ppm"],
                    "steps": r100_q["steps_run"],
                },
            }
        )
    if r100_q19 is None:
        out.append({"gate": "c_d3q19_control", "pass": False, "value": None, "detail": "missing artifact"})
    else:
        out.append(
            {
                "gate": "c_d3q19_control_cd_error_pct",
                "value": r100_q19["cd_error_pct"],
                "expectation": "> 15% (campaign statement)",
                "pass": r100_q19["cd_error_pct"] > THR["c_d3q19_control_expected_above_pct"],
                "detail": {"cd": r100_q19["cd_tail_mean"], "cd_ref": r100_q19["cd_ref_schiller_naumann"]},
            }
        )
    if r200_q is None:
        out.append({"gate": "d_re200_d3q27", "pass": False, "value": None, "detail": "missing artifact"})
    else:
        out.append(
            {
                "gate": "d_re200_d3q27_cd_error_pct",
                "value": r200_q["cd_error_pct"],
                "threshold": THR["d_re200_cd_err_pct"],
                "pass": r200_q["cd_error_pct"] < THR["d_re200_cd_err_pct"] and r200_q["finite"],
                "detail": {
                    "cd": r200_q["cd_tail_mean"],
                    "cd_ref": r200_q["cd_ref_schiller_naumann"],
                    "plateau_drift_pct": r200_q["plateau_drift_pct"],
                    "mass_drift_ppm": r200_q["mass_drift_ppm"],
                },
            }
        )
    if r100_q is not None and r100_q19 is not None:
        ratio = r100_q19["cd_error_pct"] / r100_q["cd_error_pct"] if r100_q["cd_error_pct"] > 0 else float("inf")
        out.append(
            {
                "gate": "e_improvement_factor_d3q19_err_over_d3q27_err",
                "value": ratio,
                "threshold": THR["e_improvement_factor_min"],
                "pass": ratio > THR["e_improvement_factor_min"],
                "detail": {
                    "err_d3q19_pct": r100_q19["cd_error_pct"],
                    "err_d3q27_pct": r100_q["cd_error_pct"],
                    "grid_D": r100_q["resolution_D"],
                },
            }
        )
        # secondary diagnostic (shipped-default force path)
    out.append(
        {
            "gate": "h_mrt_stability_re200",
            "value": (r200_q or {}).get("steps_run"),
            "threshold": THR["h_steps"],
            "pass": bool(
                r200_q
                and r200_q["steps_run"] >= THR["h_steps"]
                and r200_q["finite"]
                and abs(r200_q["plateau_drift_pct"]) < THR["h_plateau_drift_pct_max"]
            ),
            "detail": {"plateau_drift_pct": (r200_q or {}).get("plateau_drift_pct")},
        }
    )
    return out


def gate_f():
    res = {}
    for lat in ("d3q27", "d3q19"):
        p = os.path.join(RUNS, f"tgv_convergence_{lat}.json")
        res[lat] = json.load(open(p)) if os.path.exists(p) else None
    d27 = res["d3q27"]
    gate = {
        "gate": "f_tgv_l2_order",
        "value": (d27 or {}).get("observed_order_primary"),
        "threshold": THR["f_order_min"],
        "pass": bool(d27 and d27["observed_order_primary"] >= THR["f_order_min"]),
        "detail": {
            "l2_errors_d3q27": (d27 or {}).get("l2_errors"),
            "observed_orders_d3q27": (d27 or {}).get("observed_orders"),
            "l2_errors_d3q19": (res["d3q19"] or {}).get("l2_errors"),
            "observed_orders_d3q19": (res["d3q19"] or {}).get("observed_orders"),
        },
    }
    return gate


def gate_i():
    """Run the activation test file and count tests / negative controls."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        os.path.join("tests", "test_general_sim_d3q27_activation.py"),
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
    ]
    proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(HERE)), capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    summary = tail[-1] if tail else ""
    n_pass = 0
    for tok in summary.replace("=", " ").replace(",", " ").split():
        if tok.isdigit():
            n_pass = max(n_pass, int(tok))
    src = open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "tests", "test_general_sim_d3q27_activation.py")).read()
    n_neg = src.count("test_negative_")
    return {
        "gate": "i_pytest_activation_suite",
        "value": {"tests_passed": n_pass, "negative_controls": n_neg, "rc": proc.returncode},
        "threshold": {"tests": THR["i_tests_min"], "negative_controls": THR["i_negative_controls_min"]},
        "pass": proc.returncode == 0 and n_pass >= THR["i_tests_min"] and n_neg >= THR["i_negative_controls_min"],
        "detail": {"summary": summary},
    }


def main() -> int:
    gates = [gate_a(), gate_b(), gate_g(), gate_f(), *gate_cdeh(), gate_i()]
    mandatory = [g for g in gates if g["gate"] != "c_d3q19_control_cd_error_pct"]
    verdict = {
        "node": "P151-N01",
        "gates": gates,
        "mandatory_all_pass": all(g["pass"] for g in mandatory),
        "failures": [g["gate"] for g in mandatory if not g["pass"]],
        "note": (
            "c_d3q19_control_cd_error_pct states the campaign *expectation* that the "
            "D3Q19 control exceeds 15%; it is informational only — if D3Q19 is also "
            "accurate the improvement factor (gate e) is the binding number."
        ),
    }
    os.makedirs(GATES, exist_ok=True)
    with open(os.path.join(GATES, "verdict.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    for g in gates:
        print(f"{'PASS' if g['pass'] else 'FAIL'}  {g['gate']}: {json.dumps(g['value'])}")
    print("mandatory_all_pass:", verdict["mandatory_all_pass"])
    return 0 if verdict["mandatory_all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())