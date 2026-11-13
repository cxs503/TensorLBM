#!/usr/bin/env python3
"""Board audit of mission m17981 (anchor XVAL-LBM-HORIZONTAL-SURVEY-EMPTY).

Independent re-derivation of every number the mission reports, from the frozen
artifacts in its work tree only (no re-run of the physics, no hand copy):

  A1  artifact presence      -- impl files / entry / runs / tests / docs / figures
  A2  trace re-fit           -- C_I recomputed from runs/*/trace.npz with the
                                predeclared window -> compare with result.json
  A3  reference re-derivation -- potential_added_mass on the identical mask,
                                plus the analytic unbounded C_a = 1 anchor
  A4  error judgment         -- |err| against the 15% gate, same-mask and analytic
  A5  registry cross-check   -- tensor_benchmark_cases row for the anchor
  A6  integration check      -- non-self callers of the new module
  A7  figure                 -- got vs reference with the 15% band

Usage:
  python tools/audit_m17981_verify.py --work /root/tp-agent/work/m17981 \
      --out artifacts/board_audit/m18089
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

GATE_PCT = 15.0
CASES = {
    # name: (trace dir, transient periods, measure periods)
    "quick": (1.0, 2.0),
    "res16": (1.5, 3.0),
    "base_m17844": (1.5, 3.0),
}


def refit_from_trace(trace_npz: str, transient: float, measure: float, nh: int = 2,
                     fallback: dict | None = None) -> dict:
    """Re-derive the first-harmonic force with an independent least-squares fit.

    Handles both trace schemas: the module arm (omega/a0/A_solid/F_mem/F_trac)
    and the pre-module arm (F1/F2/U, scalars taken from result.json numerics).
    """
    import numpy as np

    z = np.load(trace_npz)
    fb = fallback or {}
    t = np.asarray(z["t"], float)
    omega = float(z["omega"]) if "omega" in z.files else float(fb["omega"])
    a0 = float(z["a0"]) if "a0" in z.files else float(fb["a0"])
    if "A_solid" in z.files:
        A_solid = float(z["A_solid"])
    else:
        A_solid = float(fb["n_solid"])
    D_area = math.sqrt(4.0 * A_solid / math.pi)
    T = 2.0 * math.pi / omega
    t0 = float(round(transient * T))
    t1 = t0 + measure * T
    m = (t >= t0) & (t < t1)
    out = {"n_samples": int(m.sum()), "D_area_from_trace": D_area, "window": [t0, t1]}
    cols = [np.ones(int(m.sum()))]
    for k in range(1, nh + 1):
        cols += [np.cos(k * omega * t[m]), np.sin(k * omega * t[m])]
    A = np.stack(cols, axis=1)
    vol = math.pi * D_area * D_area / 4.0
    pairs = [("F_mem", "F1") if "F_mem" not in z.files else ("F_mem", "F_mem"),
             ("F_trac", "F2") if "F_trac" not in z.files else ("F_trac", "F_trac")]
    for tag, key in pairs:
        y = np.asarray(z[key], float)[m]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        fit = A @ coef
        out[tag] = {
            "C_I": float(coef[1] / (a0 * vol)),
            "C_Q": float(coef[2] / (a0 * vol)),
            "resid_rel": float(np.sqrt(np.mean((y - fit) ** 2)) / (np.std(y) + 1e-300)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/root/tp-agent/work/m17981")
    ap.add_argument("--out", default="artifacts/board_audit/m18089")
    ap.add_argument("--bench-row", default=None, help="JSON file with the tensor_benchmark_cases row")
    args = ap.parse_args()

    work = Path(args.work)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"work_tree": str(work), "gate_pct": GATE_PCT}

    # ---------------------------------------------------------------- A1 presence
    impl = {
        "module": work / "src/tensorlbm/oscillatory_cylinder_inertia.py",
        "case_runner": work / "cases/oscillatory_cylinder_inertia/run.py",
        "survey": work / "xval/horizontal_survey_empty/survey.py",
        "arm_harness": work / "xval/horizontal_survey_empty/run_arm.py",
        "analysis": work / "xval/horizontal_survey_empty/analyze.py",
        "predeclared": work / "xval/horizontal_survey_empty/predeclared.json",
    }
    docs = {
        "report": work / "docs/XVAL_LBM_HORIZONTAL_SURVEY_EMPTY.md",
        "tests": work / "tests/test_oscillatory_cylinder_inertia.py",
        "figures": [
            work / "docs/assets/xval_lbm_horizontal_survey_empty/fig1_convergence.png",
            work / "docs/assets/xval_lbm_horizontal_survey_empty/fig2_sensitivity.png",
            work / "docs/assets/xval_lbm_horizontal_survey_empty/fig3_velocity_trace.png",
        ],
    }
    rep["impl_files"] = {k: {"path": str(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0}
                         for k, p in impl.items()}
    rep["docs"] = {k: ({"path": str(v), "exists": v.exists()} if not isinstance(v, list) else
                       [{"path": str(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0} for p in v])
                   for k, v in docs.items()}
    runs = {}
    for case in CASES:
        d = work / "runs/oscillatory_cylinder_inertia" / case
        runs[case] = {f.name: {"exists": f.exists(), "bytes": f.stat().st_size}
                      for f in (d / "result.json", d / "trace.npz")}
    runs["benchmark_table.json"] = {"exists": (work / "runs/oscillatory_cylinder_inertia/benchmark_table.json").exists()}
    rep["run_artifacts"] = runs

    # ------------------------------------------------------------- A2 trace re-fit
    refits = {}
    for case, (transient, measure) in CASES.items():
        rp = work / "runs/oscillatory_cylinder_inertia" / case / "result.json"
        tp = work / "runs/oscillatory_cylinder_inertia" / case / "trace.npz"
        if not (rp.exists() and tp.exists()):
            refits[case] = {"status": "missing_artifacts"}
            continue
        res = json.loads(rp.read_text())
        fb = dict(res.get("numerics") or {})
        fb.setdefault("n_solid", (res.get("geometry") or {}).get("n_solid")
                      or (res.get("geometry") or {}).get("n_solid_cells"))
        rf = refit_from_trace(str(tp), transient, measure, fallback=fb)
        kpi = res.get("kpi", {})
        rep_ci = (kpi.get("E1_mem_area") or {}).get("C_I")
        dev = None if rep_ci in (None, 0) else 100.0 * (rf["F_mem"]["C_I"] - rep_ci) / rep_ci
        refits[case] = {
            "geometry": res.get("geometry"),
            "case_cfg": res.get("case"),
            "reported": {"E1_mem_area": rep_ci, "E2_trac_area": (kpi.get("E2_trac_area") or {}).get("C_I"),
                         "E1_mem_halfway": (kpi.get("E1_mem_halfway") or {}).get("C_I"),
                         "err_pct_in_result": res.get("err_inphase_pct_area"),
                         "reference_in_result": res.get("reference")},
            "refit_from_trace": rf,
            "reported_vs_refit_pct": dev,
            "reproduces": bool(dev is not None and abs(dev) < 1e-6),
        }
    rep["trace_refit"] = refits

    # ------------------------------------------------- A3 reference re-derivation
    import sys
    sys.path.insert(0, str(work / "src"))
    from tensorlbm.oscillatory_cylinder_inertia import potential_reference, circle_mask, geometry_readout  # noqa: E402

    refs = {}
    for case, (res_geo) in {
        "quick": (72, 12),
        "res16": (192, 16),
        "base_m17844": (288, 24),
    }.items():
        n = int(res_geo[0])
        D = float(res_geo[1])
        p = potential_reference(n, n, D)
        g = geometry_readout(circle_mask(n, n, D))
        refs[case] = {"nx": n, "D": D, "c_a_potential_raw": p["c_a"],
                      "c_a_area_normalised": p["c_a_area"], "c_a_halfway_normalised": p["c_a_halfway"],
                      "analytic_unbounded": 1.0, "mask_area": g["n_solid"], "D_area": g["D_area"],
                      "D_halfway": g["D_halfway"],
                      "ref_vs_analytic_pct": 100.0 * (p["c_a_area"] - 1.0)}
    rep["reference_rederivation"] = refs

    # ---------------------------------------------------------- A4 error judgment
    judged = {}
    for case, r in refits.items():
        if "refit_from_trace" not in r:
            continue
        got = r["refit_from_trace"]["F_mem"]["C_I"]
        got2 = r["refit_from_trace"]["F_trac"]["C_I"]
        ref = refs[case]["c_a_area_normalised"]
        err = 100.0 * (got - ref) / ref
        err_an = 100.0 * (got - 1.0)
        judged[case] = {
            "got_E1_area": got, "got_E2_area": got2,
            "estimator_spread_pct": abs(got - got2),
            "ref_same_mask_area": ref, "err_vs_same_mask_pct": err,
            "err_vs_analytic_Ca1_pct": err_an,
            "pass_same_mask": abs(err) <= GATE_PCT,
            "pass_analytic": abs(err_an) <= GATE_PCT,
        }
    # the pre-module arm used the Morison convention C_M = 2; recompute that too
    bm = work / "runs/oscillatory_cylinder_inertia/base_m17844/result.json"
    if bm.exists():
        b = json.loads(bm.read_text())
        k = b.get("kpi", {})
        hw = (k.get("E1_mem_halfway") or {}).get("C_I")
        if hw:
            judged.setdefault("base_m17844", {})
            judged["base_m17844"].update({
                "got_E1_halfway": hw,
                "ref_morison_CM2": 2.0,
                "err_vs_morison_pct": 100.0 * (hw - 2.0) / 2.0,
                "note": "pre-module arm; conclusions in the delivery are drawn from the same-mask reference, "
                        "this row shows the -21.29% number quoted by the benchmark registry",
            })
    rep["error_judgment"] = judged
    rep["gate_all_fail"] = all(not (v.get("pass_same_mask") or v.get("pass_analytic"))
                              for v in judged.values())

    # -------------------------------------------------- A5 registry cross-check
    if args.bench_row and Path(args.bench_row).exists():
        row = json.loads(Path(args.bench_row).read_text())
        got_in_metric = re.findall(r"-?\d+\.\d+", row.get("metric", ""))
        rep["benchmark_registry"] = {
            "source": args.bench_row,
            "case_name_len": len(row.get("case_name", "")),
            "tolerance": (row.get("tolerance") or "")[:120],
            "quoted_got": {"base_E1_halfway": 1.5742086608150845 if "1.5742086608150845" in row.get("metric", "") else None,
                           "err_pct": -21.29 if "-21.29" in row.get("metric", "") else None},
            "status_line": "gate NOT passed (status left pending by the registering audit)" if "NOT passed" in row.get("tolerance", "") else "unknown",
            "numbers_found": got_in_metric[:8],
            "registry_tolerance_is_15pct": "15%" in row.get("tolerance", ""),
            "row_vs_artifact_consistency": "registry quotes the pre-module (m17844) arm, ref Morison C_M=2",
        }

    # ------------------------------------------------------- A6 integration check
    try:
        grep = subprocess.run(
            ["grep", "-rn", "oscillatory_cylinder_inertia", "--include=*.py", "--include=*.md", str(work)],
            capture_output=True, text=True, timeout=180).stdout.splitlines()
    except Exception as exc:  # pragma: no cover
        grep = []
        rep["integration_grep_error"] = str(exc)
    self_files = ("src/tensorlbm/oscillatory_cylinder_inertia.py", "tests/test_oscillatory_cylinder_inertia.py",
                  "cases/oscillatory_cylinder_inertia/", "docs/XVAL_LBM_HORIZONTAL_SURVEY_EMPTY.md")
    callers = [ln for ln in grep if not any(s in ln for s in self_files)]
    rep["integration_callers"] = callers
    rep["integration_verdict"] = "none (0 non-self callers) -> L3 not met" if not callers else "callers present"

    # ------------------------------------------------------------------ A7 figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = [c for c in ("quick", "base_m17844", "res16") if c in judged]
        labels = {"quick": "quick\nD=12, box 6D", "base_m17844": "base (m17844 arm)\nD=24, box 12D",
                  "res16": "res16\nD=16, box 12D"}
        got = [judged[c]["got_E1_area"] if "got_E1_area" in judged[c] else judged[c]["got_E1_halfway"] for c in order]
        ref = [refs[c]["c_a_area_normalised"] for c in order]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        x = range(len(order))
        ax[0].bar([i - 0.2 for i in x], ref, width=0.4, label="same-mask potential $C_a$ (ref)", color="#4c72b0")
        ax[0].bar([i + 0.2 for i in x], got, width=0.4, label="measured $C_I$ (E1, m17981 artifacts)", color="#c44e52")
        ax[0].set_xticks(list(x))
        ax[0].set_xticklabels([labels[c] for c in order], fontsize=8)
        ax[0].set_ylabel("inertia coefficient [-]")
        ax[0].set_title("got vs reference (auditor re-fit from trace.npz)")
        ax[0].legend(fontsize=8)
        errs = [100.0 * (g - r) / r for g, r in zip(got, ref)]
        ax[1].axhspan(-GATE_PCT, GATE_PCT, color="#8fce8f", alpha=0.35, label="±15% gate")
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].plot(list(x), errs, "o-", color="#c44e52", label="vs same-mask $C_a$")
        an_err = [judged[c].get("err_vs_analytic_Ca1_pct") for c in order]
        ax[1].plot(list(x), [e if e is not None else float("nan") for e in an_err], "s--", color="#555555",
                   label="vs analytic $C_a=1$")
        ax[1].set_xticks(list(x))
        ax[1].set_xticklabels([labels[c] for c in order], fontsize=8)
        ax[1].set_ylabel("error [%]")
        ax[1].set_title("error judgment: every arm outside the gate")
        ax[1].legend(fontsize=8)
        fig.suptitle("AUDIT m17981  XVAL-LBM-HORIZONTAL-SURVEY-EMPTY", fontsize=10)
        fig.tight_layout()
        figp = out / "fig_audit_m17981.png"
        fig.savefig(figp, dpi=130)
        rep["figure"] = str(figp)
    except Exception as exc:  # pragma: no cover
        rep["figure_error"] = str(exc)

    (out / "audit_m17981.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({"json": str(out / "audit_m17981.json"),
                      "figure": rep.get("figure"),
                      "errors": {k: v for k, v in judged.items()},
                      "reproduces": {k: v.get("reproduces") for k, v in refits.items()},
                      "integration": rep["integration_verdict"]}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())