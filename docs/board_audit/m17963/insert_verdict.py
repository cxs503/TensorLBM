"""m17963 audit: register the benchmark judge row + maturity verdict for mission 17759.

Numbers are read from audit_numbers.json (programmatic, zero hand-copy).
Idempotent: re-running updates the benchmark row and upserts the verdict.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = json.loads((HERE / "audit_numbers.json").read_text())

ANCHOR = "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S"
SCOPE = "m17759 " + ANCHOR
MISSION_ID = 17963  # this audit mission
AUDITED = 17759

p = N["params"]
case_name = (
    "Poiseuille channel D2Q9 (NX=%d, NY=%d, nu=%g, Gx=%g, %d steps), TensorLBM arm BGK+Guo+"
    "force-corrected NEBB vs OSS haimingz/vivsim (BGK+EDM / BGK+Guo) vs closed-form"
    % (p["NX"], p["NY"], p["nu"], p["Gx"], p["steps"])
)
metric = (
    "ux(y) error vs closed form, two pre-declared-vs-reported metrics: "
    "(a) pre-declared max relative error at interior points: TLB %.2f%% (FAIL), OSS %.2f%%; "
    "(b) global-normalised max|d|/max|ref| used by the delivery report: TLB %.2f%% (PASS), OSS %.2f%%; "
    "(c) cross-arm global: %.2f%%; (d) TLB wall slip ux(y=0)=%.4g = %.2f%% of u_max (no-slip not enforced); "
    "(e) NY=20: TLB %.2f%%, OSS %.2f%%. Audit m17963 re-ran the delivery script: JSON payload bit-identical "
    "(sha256 of sorted dump 8ec29bc7ad78cb24...). Level judged L2: arm B is a hand-written numpy paraphrase, "
    "no import of tensorlbm (0 callers), so no registered judge passes under the pre-declared metric."
    % (
        100 * N["tlb_pointwise_interior_err"],
        100 * N["oss_pointwise_interior_err"],
        100 * N["tlb_global_err"],
        100 * N["oss_global_err"],
        100 * N["cross_arm_global_err"],
        N["tlb_wall_ux"],
        N["tlb_wall_slip_pct_of_umax"],
        100 * N["tlb_ny20_global_err"],
        100 * N["oss_ny20_global_err"],
    )
)
tolerance = (
    "<15%% per owner standard (pre-declared in the delivery script docstring, 'not tuned after seeing results'); "
    "ecosystem <3%%. Measured: pre-declared pointwise worst 23.50%% => GATE NOT PASSED; reported global-normalised "
    "9.40%% => would pass only after the post-hoc metric substitution (see audit m17963 F2). status=failed on the "
    "pre-declared metric on purpose, not overwritten by the mission's own PASS headline. "
    "Repro: python3 xval/xval_nebb_bodyforce_driver.py (audit copy docs/board_audit/m17963/rerun/, ~10 min 1 core)"
)


def q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def jq(obj) -> str:
    return q(json.dumps(obj, ensure_ascii=False, sort_keys=False))


evidence = {
    "impl_files": [
        "/root/tp-agent/work/m17759/xval/xval_nebb_bodyforce_driver.py:135-160 (tl_step = hand-written numpy "
        "BGK+Guo 'TensorLBM' arm; 0 imports of tensorlbm, import block :50-55 = json/pathlib/numpy only)",
        "/root/tp-agent/work/m17759/xval/xval_nebb_bodyforce_driver.py:158-250 (Arm A = numpy transcription of "
        "vivsim NEBB/EDM/Guo, source /tmp/vivsim_src; not executed against vivsim itself: jax absent, NOT_RUN)",
        "commit 08a55f8e (xval/REPORT_XVAL_NEBB_BODYFORCE.md 128 lines, xval/DIFF_OSS_VS_TENSORLBM.md)",
        "repo code claimed but NOT exercised: src/tensorlbm/gpu_core_path.py:278-343 collide_guo_into; "
        "src/tensorlbm/wave_bc_common.py:160-196 zou_he_inlet_velocity_profile_19",
    ],
    "entry": "python3 xval/xval_nebb_bodyforce_driver.py (no CLI args; writes xval/xval_nebb_bodyforce_driver.json)",
    "run_artifacts": [
        "/root/tp-agent/work/m17759/xval/xval_nebb_bodyforce_driver.json (all KPIs, 2 arms + 2 grids)",
        "/root/tp-agent/work/m17759/xval/xval_nebb_profile.png (74 KB, matplotlib, generated from the JSON only)",
        "/root/tp-agent/work/m17759/xval/xval_run.log (verdict dump; was untracked at delivery)",
        "audit rerun: /root/tp-agent/work/m17963/docs/board_audit/m17963/rerun/xval_nebb_bodyforce_driver.json "
        "(payload identical to delivery) + run.log",
    ],
    "tests": "none: grep -rn 'xval_nebb' tests/ benchmarks/ = 0 hits; no pytest, no CI hook, "
             "xval/ is not under benchmarks/ and has no benchmark harness",
    "integration_callers": "0: script never imports tensorlbm; src/ untouched by commit 08a55f8e; "
                           "no public API/module caller of the NEBB+body-force path exists in the delivery",
    "benchmark": {
        "table": "tensor_benchmark_cases",
        "anchor_slug": ANCHOR,
        "rows_at_delivery": 0,
        "note": "no registered ref/tolerance existed when m17759 self-declared PASS; row registered by this audit "
                "(status=failed on the pre-declared metric)",
        "predeclared_metric": {"name": "max relative error of ux(y) interior points vs closed form",
                               "threshold": 0.15, "got": round(N["tlb_pointwise_interior_err"], 6),
                               "pass": N["predeclared_metric_pass"]},
        "reported_metric": {"name": "global-normalised max|d|/max|ref|", "threshold": 0.15,
                            "got": round(N["tlb_global_err"], 6), "pass": N["substituted_metric_pass"]},
        "oss_arm": {"global": round(N["oss_global_err"], 6),
                    "pointwise": round(N["oss_pointwise_interior_err"], 6)},
        "cross_arm": round(N["cross_arm_global_err"], 6),
        "wall_slip": {"ux_wall": N["tlb_wall_ux"],
                      "pct_of_umax": round(N["tlb_wall_slip_pct_of_umax"], 4)},
        "grid_ny20": {"tlb": round(N["tlb_ny20_global_err"], 6),
                      "oss": round(N["oss_ny20_global_err"], 6)},
        "ecosystem_3pct_met": N["ecosystem_3pct_met"],
    },
    "repro_cmd": "cd /root/tp-agent/work/m17963/docs/board_audit/m17963/rerun && python3 "
                 "xval_nebb_bodyforce_driver.py   # ~10 min, 1 core, float64 CPU; "
                 "payload equals the delivered JSON (bit-identical)",
    "external_anchor": "haimingz/vivsim @ /tmp/vivsim_src (examples/2d/poiseuille_channel.py; "
                       "vivsim/lbm/boundary/nebb.py:23-70 shear-reconstruction NEBB; vivsim/lbm/forcing/{edm,guo}.py) "
                       "- source read and formula-checked; OSS code itself NOT executed (jax missing => NOT_RUN)",
    "audited_mission": AUDITED,
    "audited_commit": "08a55f8e",
    "level": 2,
    "level_reason": "L2 = real code input->compute->output and independently reproducible (audit re-ran, payload "
                    "identical). L3 blocked: 0 integration callers, repo Guo/NEBB code never imported. L4 blocked: "
                    "no registered benchmark row at delivery, and the pre-declared 15% metric FAILS at 23.50% "
                    "(PASS exists only after post-hoc switch to the global-normalised metric).",
    "defects": [
        "F1 (blocking): 'cross-validated TensorLBM scaffold' overstated - arm B is a hand-written numpy "
        "paraphrase; import block :50-55 has no tensorlbm; repo collide_guo_into/NEBb never executed",
        "F2 (blocking): pre-declared pointwise metric 23.50% > 15% FAIL; report headline PASS relies on "
        "switching to global-normalised 9.40% (disclosed in report section 5, but flips the verdict)",
        "F3 (high): TLB arm wall velocity ux(y=0)=0.0022 = 4.40% of u_max (no-slip not enforced); cross-arm "
        "4.400% gap equals that slip, yet report calls it a 'second-order wall effect' (unquantified)",
        "F4 (high): tensor_benchmark_cases had 0 rows for the anchor - mission self-judged PASS",
        "F5 (low): OSS arm never ran on real vivsim (jax ModuleNotFoundError) => NOT_RUN; so the 'cross' is "
        "between two transcriptions, not two independent implementations; 10 min single-core run should go via "
        "run_cluster; zero tests; xval/xval_run.log untracked at delivery",
    ],
    "checks_run": [
        "independent rerun of the delivery script in this audit worktree (payload identical, sha256 match)",
        "all error percentages recomputed from the raw ux profiles in the delivered JSON",
        "grep for tensorlbm imports / callers; repo Guo & NEBB source inspection",
        "vivsim nebb.py formula read vs the delivery DIFF table",
        "tensor_benchmark_cases query for the anchor (0 rows)",
    ],
    "checks_not_run": [
        "real vivsim execution (jax absent -> NOT_RUN)",
        "repo-level GPU/torch reproduction of collide_guo_into at the same operating point (not required for the "
        "level; would be the first step of the repair mission)",
    ],
    "figures": ["docs/board_audit/m17963/fig_m17963_profile.png",
                "docs/board_audit/m17963/fig_m17963_metrics.png"],
    "audit_report": "docs/board_audit/m17963/AUDIT_m17759.md",
    "numbers": "docs/board_audit/m17963/audit_numbers.json",
    "gaps": [
        "L3: no solver/package API caller",
        "L4: no registered judge row at delivery; pre-declared metric fails",
        "no tests / no CI for the xval script",
        "OSS arm not executed against the real OSS code (NOT_RUN)",
    ],
}

sql = f"""
BEGIN;
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
SELECT {q(ANCHOR)}, {q(case_name)},
       {q('closed-form Poiseuille u=Gx/(2nu)y(H-y) + OSS haimingz/vivsim @/tmp/vivsim_src (source-read); '
          'registered by audit m17963, absent at m17759 delivery')},
       {q(metric)}, {q(tolerance)}, 'TensorLBM', 'failed'
WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases WHERE anchor_slug = {q(ANCHOR)});
UPDATE tensor_benchmark_cases SET case_name = {q(case_name)}, metric = {q(metric)},
       tolerance = {q(tolerance)}, repo = 'TensorLBM', status = 'failed'
WHERE anchor_slug = {q(ANCHOR)};

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', {q(SCOPE)}, {MISSION_ID}, 2, {jq(evidence)}::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by)
DO UPDATE SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level,
              evidence = EXCLUDED.evidence;
COMMIT;
"""

sqlfile = HERE / "insert_verdict.sql"
sqlfile.write_text(sql)
r = subprocess.run(["psql", "-U", "ship", "-d", "tensorplatform", "-v", "ON_ERROR_STOP=1", "-f", str(sqlfile)],
                   capture_output=True, text=True)
print(r.stdout[-2000:])
print(r.stderr[-2000:])
print("rc =", r.returncode)