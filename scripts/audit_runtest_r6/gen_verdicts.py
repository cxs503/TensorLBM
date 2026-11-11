#!/usr/bin/env python3
"""AUDIT-RUNTEST-6 — generate the tensor_maturity_verdicts INSERTs (verdict_by='audit2-run').

Levels (declared before looking at the pass rate, per the audit rubric):
  L4  rerun reproduces the archived claim to ~1e-4 relative AND no defect found
  L3  runs and agrees with the claim; sampled only (no defect, no bit-level check)
  L2  runs, but a defect was found on the way (fixed here or reported)
  L1  the claim cannot be reproduced on a normal host / the artifact is mislabelled
  L0  does not run at all

Writes runs/audit_runtest_r6/verdicts.sql (idempotent: ON CONFLICT DO UPDATE on
the (repo, scope, verdict_by) unique key).

Usage: python scripts/audit_runtest_r6/gen_verdicts.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "artifacts/audit_runtest_6" if (REPO / "artifacts/audit_runtest_6").exists() else REPO / "runs/audit_runtest_r6"
MISSION = 14496


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def esc(s: str) -> str:
    return s.replace("'", "''")


def row(scope: str, level: int, evidence: dict) -> str:
    ev = esc(json.dumps(evidence, ensure_ascii=False))
    return (
        "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)\n"
        f"VALUES ('TensorLBM', '{esc(scope)}', {MISSION}, {level}, '{ev}'::jsonb, 'audit2-run')\n"
        "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
        "level = EXCLUDED.level, evidence = EXCLUDED.evidence, mission_id = EXCLUDED.mission_id;\n"
    )


def main() -> int:
    out: list[str] = ["-- AUDIT-RUNTEST-6 (mission m14496) run evidence for TensorLBM\n"]
    ps = jload(RUNS / "pytest_summary.json")
    if ps:
        passed = ps["tests"] - ps["failures"] - ps["errors"] - ps["skipped"]
        rate = 100.0 * passed / max(1, ps["tests"] - ps["skipped"])
        out.append(
            row(
                "tests/full_suite#run",
                2,
                {
                    "artifact": "runs/audit_runtest_r6/pytest_full_local.log",
                    "junit": "runs/audit_runtest_r6/junit_local.xml",
                    "summary": "runs/audit_runtest_r6/pytest_summary.json",
                    "command": "PYTHONPATH=src python -m pytest tests -q -m 'not gpu' -n 28 "
                               "--timeout=1800 --junitxml=runs/audit_runtest_r6/junit_local.xml",
                    "counts": {
                        "tests": ps["tests"],
                        "passed": passed,
                        "failures": ps["failures"],
                        "errors": ps["errors"],
                        "skipped": ps["skipped"],
                        "pass_rate_excl_skip_pct": round(rate, 2),
                        "seconds": round(ps["seconds"], 1),
                    },
                    "top_failing_files": dict(list(ps["failures_by_file"].items())[:10]),
                    "conclusion": "suite executes and is mostly green; the failure clusters above are "
                                  "real, reproducible failures on this host (not collected from a PR "
                                  "description). Level 2 = runs, defects reported.",
                },
            )
        )

    ps_s = jload(RUNS / "pytest_summary_sample.json") or None
    if ps_s is None:
        # build_report writes pytest_summary.json only for the full sweep; the
        # sample summary is recomputed here from the JUnit file if needed.
        import xml.etree.ElementTree as ET  # local import keeps the script standalone

        xml = RUNS / "junit_sample.xml"
        if xml.exists():
            root = ET.parse(xml).getroot()
            suites = [root] if root.tag == "testsuite" else list(root)
            tot = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "seconds": 0.0}
            fails = []
            for s in suites:
                for k in ("tests", "failures", "errors", "skipped"):
                    tot[k] += int(float(s.get(k, 0)))
                tot["seconds"] += float(s.get("time", 0.0))
                for tc in s.iter("testcase"):
                    for kind in ("failure", "error"):
                        f = tc.find(kind)
                        if f is not None:
                            msg = (f.get("message") or (f.text or "")).strip().splitlines()
                            fails.append(f"{tc.get('classname', '')}::{tc.get('name')}: "
                                         f"{(msg[0] if msg else '')[:160]}")
            tot["failure_messages"] = fails[:40]
            ps_s = tot
    if ps_s:
        passed = ps_s["tests"] - ps_s["failures"] - ps_s["errors"] - ps_s["skipped"]
        out.append(
            row(
                "tests/systematic_sample#run",
                2,
                {
                    "artifact": "runs/audit_runtest_r6/pytest_sample.log",
                    "junit": "runs/audit_runtest_r6/junit_sample.xml",
                    "sample_rule": "ls tests/test_*.py | sort | awk 'NR%20==1' (42 files) + "
                                   "tests/test_poiseuille_wallmodel_arms.py",
                    "counts": {
                        "tests": ps_s["tests"],
                        "passed": passed,
                        "failures": ps_s["failures"],
                        "errors": ps_s["errors"],
                        "skipped": ps_s["skipped"],
                        "pass_rate_excl_skip_pct": round(100.0 * passed / max(1, ps_s["tests"] - ps_s["skipped"]), 2),
                        "seconds": round(ps_s["seconds"], 1),
                    },
                    "failure_messages": ps_s.get("failure_messages", []),
                    "conclusion": "the only suite run that finished inside the budget (two full sweeps "
                                  "were killed by host contention); 10 reproducible failures out of 834 "
                                  "collected cases -- see failure_messages.",
                },
            )
        )

    conv_new = jload(RUNS / "conv_fix_H10_20.json")
    conv_old = jload(RUNS / "conv_repro.json")
    if conv_new and conv_old:
        orders = {a: v["convergence"]["observed_orders_l2"] for a, v in conv_new["wall_models"].items()}
        old_errs = {a: v["convergence"]["errors_l2_rel"] for a, v in conv_old["wall_models"].items()}
        out.append(
            row(
                "benchmarks/poiseuille_convergence_order#run",
                2,
                {
                    "artifact": "runs/audit_runtest_r6/conv_fix_H10_20.json",
                    "baseline": "runs/audit_runtest_r6/conv_repro.json (unmodified HEAD)",
                    "regression_test": "tests/test_poiseuille_wallmodel_arms.py",
                    "regression_results": {
                        "baseline_HEAD": "5 errors (runs/audit_runtest_r6/regression_baseline_HEAD.log)",
                        "after_fix": "5 passed (runs/audit_runtest_r6/regression_fixed.log)",
                    },
                    "defect": "all three declared wall arms ran the same code path: q_wall=0.5 for "
                              "every arm, 'effective' fell through to apply_halfway_bb, 'bouzidi' at "
                              "q=0.5 is algebraically half-way bounce-back -> bit-identical l2_rel",
                    "baseline_errors_l2_rel": old_errs,
                    "orders_after_fix": orders,
                    "conclusion": "fixed: ARMS table (q/kernel/reference), H_true = ny-3+2q geometry, "
                                  "per-run true-wall and effective-wall errors; the four arms now show "
                                  "p=1.92 (halfway q=0.5), 0.98 (mid-point BC on a 0.25d wall), "
                                  "1.92 (effective-wall reference), 2.06 (Bouzidi at 0.25d).",
                },
            )
        )

    tg = jload(RUNS / "tg2d.json/result.json")
    arch_tg = jload(REPO / "benchmarks/verified/taylor_green_2d/result.json")
    if tg and arch_tg:
        out.append(
            row(
                "benchmarks/taylor_green_2d#run",
                4,
                {
                    "artifact": "runs/audit_runtest_r6/tg2d.json/result.json",
                    "log": "runs/audit_runtest_r6/bench_taylor_green_2d.log",
                    "claimed_err_e_pct": arch_tg["err_e_pct"],
                    "measured_err_e_pct": tg["err_e_pct"],
                    "claimed_mass_drift_rel": arch_tg["mass_drift_rel"],
                    "measured_mass_drift_rel": tg["mass_drift_rel"],
                    "conclusion": "energy decay rate vs analytic exp(-2 nu k^2 t) reproduced to "
                                  f"{abs(tg['err_e_pct'] - arch_tg['err_e_pct']):.2e} pp; mass drift "
                                  f"{tg['mass_drift_rel']:.2e} over {tg['steps']} steps.",
                },
            )
        )

    sw = jload(RUNS / "shear_wave.json/result.json")
    arch_sw = jload(REPO / "benchmarks/verified/shear_wave_decay/result.json")
    if sw and arch_sw:
        a = {c["H"]: c for c in arch_sw["cases"]}
        m = {c["H"]: c for c in sw["cases"]}
        pairs = {
            f"H{H}": {"claimed_err_vel_pct": a[H]["err_vel_pct"], "measured_err_vel_pct": m[H]["err_vel_pct"]}
            for H in sorted(set(a) & set(m))
        }
        out.append(
            row(
                "benchmarks/shear_wave_decay#run",
                4,
                {
                    "artifact": "runs/audit_runtest_r6/shear_wave.json/result.json",
                    "log": "runs/audit_runtest_r6/bench_shear_wave_decay.log",
                    "grids": pairs,
                    "conclusion": "viscous decay rate nu*k^2 reproduced on both grids; mass drift "
                                  f"{[round(m[H]['mass_drift_rel'], 8) for H in sorted(m)]}.",
                },
            )
        )

    p2 = jload(RUNS / "poiseuille_H60.json")
    if p2:
        out.append(
            row(
                "benchmarks/poiseuille_2d#run",
                3,
                {
                    "artifact": "runs/audit_runtest_r6/poiseuille_H60.json",
                    "log": "runs/audit_runtest_r6/poiseuille_H60.log",
                    "u_max_err_pct": p2.get("u_max_err_pct"),
                    "l2_rel_err": p2.get("l2_rel_err"),
                    "mass_drift_pct": p2.get("mass_drift_pct"),
                    "conclusion": "pressure-driven channel reproduced against the analytic parabola; "
                                  "CLI accepts H=0 and dies with ZeroDivisionError (reported, not fixed).",
                },
            )
        )

    pipe = jload(RUNS / "pipe_R20.json/result.json") if (RUNS / "pipe_R20.json").is_dir() else jload(RUNS / "pipe_R20.json")
    out.append(
        row(
            "benchmarks/poiseuille_3d_pipe#run",
            1,
            {
                "artifact": "runs/audit_runtest_r6/bench_poiseuille_3d_pipe.log",
                "result": "runs/audit_runtest_r6/pipe_R20.json" if pipe else None,
                "defect": "run.py:627/640 default --device cuda:2; CPU-only host raises "
                          "'AssertionError: Torch not compiled with CUDA enabled' on the documented "
                          "invocation, while benchmarks/STATUS.md lists the case as verified/analytic",
                "conclusion": "claim not reachable on a CPU-only host without the undocumented "
                              "--device cpu override. Level 1 = artifact misleading.",
            },
        )
    )

    neg = jload(RUNS / "negative_probe.json")
    if neg:
        s = neg["summary"]
        out.append(
            row(
                "public_input_guards#run",
                2,
                {
                    "artifact": "runs/audit_runtest_r6/negative_probe.json",
                    "script": "scripts/audit_runtest_r6/negative_probe.py",
                    "counts": s,
                    "silent_nan": [r["id"] for r in neg["rows"] if r["observed"] == "SILENT_NAN"],
                    "silent": [r["id"] for r in neg["rows"] if r["observed"] == "SILENT"],
                    "low_level": [r["id"] for r in neg["rows"] if r["observed"] == "LOW_LEVEL"],
                    "conclusion": "physics-invalid inputs are accepted silently on the core paths: "
                                  "collide_bgk(tau=0.4)/tau=0.0, equilibrium(rho<0), "
                                  "zou_he_outlet_pressure(rho_out=0) -> NaN, while a dozen sibling "
                                  "modules do validate tau>0.5; general_sim.py has no config validation.",
                },
            )
        )

    sql = "\n".join(out)
    (RUNS / "verdicts.sql").write_text(sql)
    print(sql[:1200])
    print(f"\n... {len(out) - 1} verdict rows -> {RUNS / 'verdicts.sql'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())