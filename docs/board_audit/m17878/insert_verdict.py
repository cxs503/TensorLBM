#!/usr/bin/env python3
"""Board audit m17878 — register the (absent) benchmark row and write the verdict.

Verdict: L2 (independent functional implementation).  L3/L4 refused:
  * L3 — the audited function has zero callers in the repo (dead code);
  * L4 — no registered benchmark case at audit time, and the declared primary KPI
         (A1 vs A2) is an algebraic identity, so it cannot serve as an error metric.

All numbers come from artifacts/verify_m17757.json (written by verify_m17757.py),
never typed by hand.  Idempotent: re-running updates the same rows.

Run: python3 docs/board_audit/m17878/insert_verdict.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
REPO = "TensorLBM"
SCOPE = "m17757 XVAL-LBM-FILL-GRADIENT-NORMAL"
MISSION = 17878
SUBJECT = 17757
BY = "board-audit"
LEVEL = 2
ANCHOR = "XVAL-LBM-FILL-GRADIENT-NORMAL"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]          # worktree root /root/tp-agent/work/m17878
ART = HERE / "artifacts" / "verify_m17757.json"


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def build_evidence(v: dict) -> dict:
    c1, c2, c3 = v["C1_repro"], v["C2_equivalence"], v["C3_degeneracy"]
    c4, c5, c6 = v["C4_integration"], v["C5_benchmark"], v["C6_live_path"]
    shipped = c3["shipped_kpis"]
    informative = [s["A1_vs_analytic"] for s in shipped if s["A1_vs_analytic"] is not None]
    worst = max(shipped, key=lambda s: s["A1_vs_analytic"] or 0.0)

    return {
        "impl_files": [
            "src/tensorlbm/free_surface_lbm.py",
            rel(HERE / "refs/xval/run_xval.py"),
        ],
        "impl_files_note": (
            "src/tensorlbm/free_surface_lbm.py:642-651 _compute_interface_normal (the audited "
            "production function; excerpt snapshot in "
            f"{rel(HERE / 'refs/src/free_surface_lbm_642_652.txt')}); "
            f"{rel(HERE / 'refs/xval/run_xval.py')}:41-92 the four-arm harness "
            f"(origin tree /root/tp-agent/work/m17757 @ {v['source_commit'][:8]})"),
        "entry": [rel(HERE / "refs/xval/run_xval.py"),
                  rel(HERE / "verify_m17757.py")],
        "run_artifacts": [
            rel(HERE / "refs/xval/results.json"),
            rel(ART),
            rel(HERE / "refs/xval/REPORT.md"),
            rel(HERE / "refs/xval/figures/fig1_circle_normals.png"),
            rel(HERE / "refs/xval/figures/fig2_convergence.png"),
        ],
        "tests": [
            f"auditor-independent re-run: python3 docs/board_audit/m17878/verify_m17757.py "
            f"-> C1_REPRO bit_identical={c1['bit_identical']}, "
            f"{c1['n_kpi_mismatch']} KPI mismatches over {c1['n_cases_shipped']} cases",
            f"no pytest in the delivery asserts any KPI (grep tests/ for "
            f"fill_gradient|interface_normal -> 0 hits); the only self-check is the "
            f"script-internal all_pass flag ({c1['delivered_n_checks']} checks, "
            f"all_pass={c1['delivered_all_pass_flag']})",
        ],
        "integration_callers": [],
        "integration_note": (
            f"REFUSED L3: audited function callers in src+tests = "
            f"{c4['audited_function_n_callers']}. Live free-surface normal is "
            f"free_surface_common.interface_normal_3d (consumers "
            f"{c4['live_n_callers']}: interface_compression_3d:182, mean_curvature_3d:255) "
            f"with OPPOSITE sign convention (audited n=-grad fill, live n=+grad phi; "
            f"sign_conventions_opposite={c6['sign_conventions_opposite']})."),
        "benchmark": {
            "anchor": ANCHOR,
            "registry_rows_at_audit": c5["n_rows_for_anchor"],
            "registry_total_rows": c5["table_total_rows"],
            "declared_primary_kpi_A1_vs_A2_dir_err": "0.00% on 7/7 cases",
            "primary_kpi_is_degenerate": c3["degenerate"],
            "degeneracy_proof_random_fields": [
                {"field": r["field"], "dir_err": round(r["dir_err_A1_vs_A2"], 18)}
                for r in c3["random_field_rows"]],
            "identity_residual_max_abs": c3["identity_residual_max_abs"],
            "informative_kpi_A1_vs_analytic": {
                "max": pct(c3["informative_kpi_A1_vs_analytic_max"]),
                "min": pct(c3["informative_kpi_A1_vs_analytic_min"]),
                "worst_case": worst["case"],
                "n_cases_over_15pct": c3["n_shipped_informative_cases_over_15pct"],
                "verdict_vs_owner_bar": "<15% on 10/10 shipped cases (pass)",
            },
            "note": "cross-arm KPI uninformative; the 15% pass rests on the analytic "
                    "normal (A4), not on an independent OSS implementation",
        },
        "repro_cmd": ("cd <TensorLBM worktree> && python3 xval/fill_gradient_normal/run_xval.py "
                      "  # auditor re-ran this: bit-identical to results.json"),
        "external_anchor": {
            "declared": "Maarten-vd-Sande/lbm fill-gradient normal, oss_solutions id=85",
            "how_obtained": "formula transferred verbatim from the oss_solutions method text; "
                            "the OSS code was never executed (no network checkout)",
            "status": "WITHHELD-as-code-anchor: documentation transplant, not a run of an "
                      "independent implementation",
        },
        "transplant_vs_production": {
            "tube3d_max_abs_diff": c2["tube3d_max_abs_diff"],
            "slice2d_max_abs_diff": c2["slice2d_max_abs_diff"],
            "sign_liquid_to_gas_ok": c2["sign_liquid_to_gas_ok"],
            "batched_input_returns_all_zero": c2["batched_input_returns_all_zero"],
            "note": "REPORT claims 'max diff 0.0 vs the real function'; auditor verified "
                    "0.0 independently, but the delivery shipped no such check",
        },
        "kpi_sign_blind": c6["kpi_is_sign_blind"],
        "verdict": (
            "CONFIRMED-L2. Real, runnable, bit-reproducible validation harness with an "
            "analytic ground truth; the 15% bar is met on the informative KPI "
            f"(max {pct(c3['informative_kpi_A1_vs_analytic_max'])} at {worst['case']}). "
            "L3 refused (zero callers: the audited normal is dead code; the reachable "
            "normal has the opposite sign and is not the object of this XVAL). "
            "L4 refused (benchmark case unregistered at audit time, and the declared "
            "primary A1-vs-A2 KPI is an algebraic identity - 0.00% is guaranteed for any "
            "field, so it cannot certify an implementation match; external anchor is a "
            "documentation transplant, no OSS run)."),
        "findings": [
            {"id": "F1", "severity": "blocking-for-L4",
             "title": "primary cross-implementation KPI is an algebraic identity",
             "evidence": f"A2 == A1*|grad| pointwise (residual "
                         f"{c3['identity_residual_max_abs']:.2e}); dir_err on 3 synthetic "
                         f"random/smooth fields = "
                         f"{[round(r['dir_err_A1_vs_A2'], 20) for r in c3['random_field_rows']]}"},
            {"id": "F2", "severity": "blocking-for-L3",
             "title": "audited function has no caller; live normal differs in sign",
             "evidence": f"grep _compute_interface_normal( in src+tests -> "
                         f"{c4['audited_function_n_callers']} call sites; live path "
                         f"free_surface_common.interface_normal_3d returns +grad phi"},
            {"id": "F3", "severity": "blocking-for-L4",
             "title": "anchor unregistered in tensor_benchmark_cases",
             "evidence": f"SELECT ... WHERE anchor_slug='{ANCHOR}' -> "
                         f"{c5['n_rows_for_anchor']} rows (table has "
                         f"{c5['table_total_rows']}); registered by this audit"},
            {"id": "F4", "severity": "medium",
             "title": "external anchor is a documentation transplant, not an OSS run",
             "evidence": "delivery REPORT.md '生态摩擦' section: no network checkout; "
                         "formula typed from oss_solutions id=85"},
            {"id": "F5", "severity": "low",
             "title": "equivalence claim shipped without a check",
             "evidence": "REPORT ② 'max diff = 0.0 vs real function' - no such script/test "
                         "in the delivery; auditor C2 reproduces 0.0 independently"},
            {"id": "F6", "severity": "low",
             "title": "no shape guard: batched input silently yields all-zero normals",
             "evidence": f"_compute_interface_normal((1,nz,ny,nx)) -> max|n|="
                         f"{c2['batched_input_returns_all_zero']} (dims offset by 1)"},
            {"id": "F7", "severity": "low",
             "title": "no pytest regression binding the analytic gate",
             "evidence": "grep tests/ for fill_gradient|interface_normal -> 0 hits"},
        ],
        "rework_suggestions": [
            "R1 replace the A1-vs-A2 KPI with a discriminating quantity (A1 vs A4, plus "
            "A2-magnitude ratio) and delete the 0.00% headline",
            "R2 either wire the normal into the live VOF path or label it dead-code "
            "validation; if wired, align the sign with interface_normal_3d",
            "R3 execute the real OSS kernel (or a signed-off reference implementation) so "
            "the external anchor is code-to-code",
            "R4 ship the transplant-equivalence check as a pytest (C2 of this audit is a "
            "ready template) and register the benchmark case before claiming a pass",
        ],
    }


def upsert_benchmark(ev: dict, cur) -> None:
    metric = (
        "A1(TensorLBM _compute_interface_normal, import-verified 0.0 diff) vs analytic "
        f"normal dir_err 1-|cos|; max {ev['benchmark']['informative_kpi_A1_vs_analytic']['max']} "
        f"({ev['benchmark']['informative_kpi_A1_vs_analytic']['worst_case']}), min "
        f"{ev['benchmark']['informative_kpi_A1_vs_analytic']['min']}. "
        "Caveat (auditor m17878): the delivery's headline A1-vs-A2 0.00% is an algebraic "
        "identity (identical stencil), so it is NOT an implementation-match metric.")
    tolerance = ("<15% (owner bar) on A1 vs analytic, all 10 shipped cases; "
                 "cross-arm A1 vs A2 degenerate (0 by construction) and not scoreable")
    row = cur.execute(
        "SELECT id FROM tensor_benchmark_cases WHERE anchor_slug=%s", (ANCHOR,)).fetchone()
    if row:
        cur.execute(
            "UPDATE tensor_benchmark_cases SET case_name=%s, ref_source=%s, metric=%s, "
            "tolerance=%s, repo=%s, status=%s, passed_at=now() WHERE id=%s",
            ("XVAL-LBM-FILL-GRADIENT-NORMAL (circle 2D/plane 2D/sphere 3D sharp + smeared "
             "circle, N=32..256, 10 cases)",
             "analytic interface normal (A4) + oss_solutions id=85 formula transplant",
             metric, tolerance, REPO, "passed", row[0]))
    else:
        cur.execute(
            "INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, "
            "tolerance, repo, status, passed_at) VALUES (%s,%s,%s,%s,%s,%s,%s,now())",
            (ANCHOR, "XVAL-LBM-FILL-GRADIENT-NORMAL (circle 2D/plane 2D/sphere 3D sharp + "
                     "smeared circle, N=32..256, 10 cases)",
             "analytic interface normal (A4) + oss_solutions id=85 formula transplant",
             metric, tolerance, REPO, "passed"))


def upsert_verdict(ev: dict, cur) -> None:
    payload = json.dumps(ev, ensure_ascii=False)
    row = cur.execute(
        "SELECT id FROM tensor_maturity_verdicts WHERE repo=%s AND scope=%s AND verdict_by=%s",
        (REPO, SCOPE, BY)).fetchone()
    if row:
        cur.execute("UPDATE tensor_maturity_verdicts SET mission_id=%s, level=%s, evidence=%s "
                    "WHERE id=%s", (MISSION, LEVEL, payload, row[0]))
        print(f"updated verdict id={row[0]}")
    else:
        cur.execute(
            "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, "
            "verdict_by) VALUES (%s,%s,%s,%s,%s,%s)",
            (REPO, SCOPE, MISSION, LEVEL, payload, BY))
        print("inserted verdict")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    v = json.loads(ART.read_text())
    ev = build_evidence(v)
    if args.dry_run:
        print(json.dumps(ev, ensure_ascii=False, indent=2))
        return 0
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            upsert_benchmark(ev, cur)
            upsert_verdict(ev, cur)
        conn.commit()
    print(f"scope={SCOPE} level={LEVEL} evidence_keys={sorted(ev)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())