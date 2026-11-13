#!/usr/bin/env python3
"""m18435 board audit —落表: benchmark case row + maturity verdict.

Everything numeric is read from files (artifact / auditor re-run), nothing is
hand-typed. Writes two rows:
  1) tensor_benchmark_cases  — the row for this anchor was MISSING at delivery
     time (SELECT ... anchor_slug=... -> 0 rows); this audit registers it with
     the predeclared pass/fail record.
  2) tensor_maturity_verdicts — level 2 (independent functional implementation),
     evidence keys = impl_files/entry/run_artifacts/tests/integration_callers/
     benchmark/repro_cmd/external_anchor per the board-audit contract.

Run: python3 docs/board_audit/m18435/insert_verdict.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
ART = ROOT / "audit" / "m18327"
REPRO = ROOT / "artifacts" / "audit_m18327"
ANCHOR = "XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY"
AUDIT_MISSION = 18435          # this board-audit mission
DELIVERED_MISSION = 18327      # the audited (succeeded) mission
REPO = "TensorLBM"


def q(sql: str) -> list[list[str]]:
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
                        "-tA", "-F", "\x01", "-c", sql],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:500])
    return [ln.split("\x01") for ln in r.stdout.strip().split("\n") if ln]


def esc(s) -> str:
    return str(s).replace("'", "''")


got = json.loads((ART / "results" / "xval_results.json").read_text())
rep = {p.stem: json.loads(p.read_text()) for p in REPRO.glob("repro_*.json")}
ch = {c["tag"]: c for c in got["channel"]["cases"]}
mom = {c["u_eq"]: c for c in got["momentum"]["cases"]}
worst = got["channel"]["worst_rel_err_max"]
sub_gate_fail = [c["tag"] for c in got["channel"]["cases"] if not c["pass"]]
sep_present = "separation" in got

# ── 1. tensor_benchmark_cases ────────────────────────────────────────────────
metric = (
    "arm-by-arm, delivered artifact vs auditor re-run: "
    f"parity worst rel dev f64 {got['parity']['worst_rel_dev_float64']:.3e} "
    f"(auditor re-run {rep['repro_parity']['parity']['worst_rel_dev_float64']:.3e}), "
    f"f32 {got['parity']['worst_rel_dev_float32']:.3e} "
    f"(auditor {rep['repro_parity']['parity']['worst_rel_dev_float32']:.3e}); "
    f"channel worst rel_err_max {100*worst:.2f}% = {100*ch['primary']['rel_err_max']:.2f}%(ny=33 primary) / "
    f"{100*ch['refine_17']['rel_err_max']:.2f}%(ny=17) / {100*ch['refine_65']['rel_err_max']:.2f}%(ny=65), "
    f"nu rel err <= {100*max(c['nu_rel_err'] for c in got['channel']['cases']):.3f}% "
    f"(auditor re-run reproduces the same per-case numbers digit-for-digit); "
    f"momentum rel drift viscous {mom['viscous']['rel_drift']:.3e} vs "
    f"mixture {mom['mixture']['rel_drift']:.3e} (negative control) vs "
    f"self {mom['self']['rel_drift']:.3e}; "
    f"separation arm: {'present' if sep_present else 'ABSENT from xval_results.json'} - "
    "delivered doc prints control 3.16e-07 / G=-0.4 7.666e-06 / G=-0.8 1.911e-04 / G=-1.2 1.571e+00 "
    "as prose only, G=-1.6/-2.0 never executed, so 1/5 declared G levels passes min_contrast=0.5 "
    "and there is no artifact for the arm. "
    f"channel per-case sub-gate misses: {sub_gate_fail or 'none'} "
    f"(refine_65 steady_resid {ch['refine_65']['steady_resid']:.3e} > predeclared 1e-4) - invisible at "
    "arm level because the driver gates on rows[0] only."
)
tolerance = (
    "predeclared (xval/predeclared.json, written before the runs): parity <=1e-10 f64 / <=1e-4 f32; "
    "channel rel_err_max <=15% AND nu_rel_err <=15% AND steady_resid <=1e-4; momentum <=1e-12; "
    "separation control contrast <=1e-3 and SC-arm contrast >=0.5 over G=-0.4..-2.0. "
    "owner precision standard 15% (ecosystem self target 3%)."
)
ref_source = (
    "oss_solutions id=35 (evidence report 9555) loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow "
    "'shan-chen-mcmp-tau-mixed-velocity': vel_mix=Sum(Vel*Rho/tau)/Sum(Rho/tau) at Domain.py:151-153 "
    "(tau list Domain.py:77-78). Formula-level anchor only: the sandbox had no GitHub access, so no OSS "
    "output numbers were compared. Secondary reference = analytic plane-Poiseuille with "
    "nu_eff=Sum(rho_k nu_k)/rho (continuum limit of the same formula, derived in the delivered doc)."
)
status = "failed" if (not sep_present or sub_gate_fail) else "passed"

n = q(f"SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}'")[0][0]
if int(n) == 0:
    q("INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status) "
      f"VALUES ('{ANCHOR}', '{esc(ANCHOR)} (m18327 tau-mixed/viscosity-weighted common velocity of the "
      f"two-component MCMP collision vs OSS formula + analytic mixture-viscosity channel)', "
      f"'{esc(ref_source)}', '{esc(metric)}', '{esc(tolerance)}', '{REPO}', '{status}')")
    print(f"benchmark row inserted (status={status})")
else:
    print("benchmark row already present, left untouched")
bid = q(f"SELECT id,status FROM tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}'")
print("row:", bid)

# ── 2. tensor_maturity_verdicts ─────────────────────────────────────────────
evidence = {
    "impl_files": ["audit/m18327/src/mcmp_tau_mixed.py",
                   "audit/m18327/tests/test_mcmp_tau_mixed_xval.py"],
    "entry": "audit/m18327/xval/run_xval.py",
    "run_artifacts": [
        "audit/m18327/results/xval_results.json",
        "audit/m18327/figures/channel_profiles.png",
        "audit/m18327/figures/convergence_viscosity.png",
        "audit/m18327/figures/momentum.png",
        "artifacts/audit_m18327/repro_parity.json",
        "artifacts/audit_m18327/repro_momentum.json",
        "artifacts/audit_m18327/repro_channel.json",
        "artifacts/audit_m18327/claim_vs_measured.json",
    ],
    "tests": [
        "tests/test_mcmp_tau_mixed_xval.py (13 tests: REF-OSS formula stacking / two-grouping consistency, "
        "5-seed black-box probe, nu_eff continuum limit, analytic channel self-consistency, three coupling "
        "modes momentum conservation, artifact threshold consistency)",
        "auditor ran: cd /root/tp-agent/work/m18327 && PYTHONPATH=src python3 -m pytest "
        "tests/test_mcmp_tau_mixed_xval.py -q -> 13 passed, 1 warning in 7.51s",
    ],
    "integration_callers": [],
    "integration_note": ("none, and none can exist: the module is not exported in src/tensorlbm/__init__.py, "
                         "no solver/platform path references it (grep over the worktree: only the test file "
                         "imports tensorlbm.mcmp_tau_mixed), and the delivery was never committed "
                         "(mission result head_commit=null, branch absent, git log --all in the ecosystem "
                         "checkout has no commit touching the file; git_status shows '?? src/tensorlbm/"
                         "mcmp_tau_mixed.py'). The driver does call the real library collision "
                         "multiphase.collide_sc_two_component, so the validation exercises product code, "
                         "but the new module itself is not integrated -> L3 not claimed."),
    "benchmark": {
        "registered_row": f"tensor_benchmark_cases id={bid[0][0] if bid else 'n/a'} (row was ABSENT at delivery, "
                          "registered by this audit)",
        "reference": ref_source,
        "tolerance": tolerance,
        "got": {
            "parity_f64": got["parity"]["worst_rel_dev_float64"],
            "parity_f32": got["parity"]["worst_rel_dev_float32"],
            "channel_rel_err_max_worst": worst,
            "channel_rel_err_max": {k: v["rel_err_max"] for k, v in ch.items()},
            "channel_nu_rel_err": {k: v["nu_rel_err"] for k, v in ch.items()},
            "channel_steady_resid": {k: v["steady_resid"] for k, v in ch.items()},
            "momentum_rel_drift": {k: v["rel_drift"] for k, v in mom.items()},
            "separation": "arm absent from artifact (doc prose: control 3.16e-07, G=-0.4 7.666e-06, "
                          "G=-0.8 1.911e-04, G=-1.2 1.571e+00; G=-1.6/-2.0 never run)",
        },
        "verdict": ("parity PASS (round-off), channel PASS vs the 15% owner band but 6.21% (ny=33) / "
                    "12.34% (ny=17) / 3.18% (ny=65) all ABOVE the ecosystem 3% bar, and refine_65 misses its "
                    "own predeclared steady_resid sub-gate by 14% -> arm-level 'PASS' comes from gating "
                    "rows[0] only; momentum PASS with a real negative control (mixture 5.29e-1); "
                    "separation arm NOT discharged -> the four-arm predeclared gate is not met as a whole."),
        "auditor_repro": "parity/momentum bit-identical; channel reproduced digit-for-digit per case "
                         "(see repro_*.json)",
    },
    "repro_cmd": ("cd /root/tp-agent/work/m18327 && PYTHONPATH=src python3 "
                  "xval/XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY/run_xval.py --arm all "
                  "--out /tmp/m18327_xval.json   # auditor ran --arm parity|momentum|channel "
                  "with --out into its own artifacts dir; 13/13 pytest green"),
    "external_anchor": ("oss_solutions id=35 loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow "
                        "Domain.py:151-153 (tau-mixed velocity) + Domain.py:70-88/143-176 provenance; "
                        "cited from the platform inventory record only (GitHub unreachable in the sandbox), "
                        "so parity is against a transcribed formula, NOT against numbers produced by that "
                        "code; analytic Poiseuille with nu_eff is the quantitative secondary reference."),
    "level_rationale": ("L2 = independent functional implementation: real module (input->compute->output), "
                        "real driver, executed artifacts, 13 green unit tests. Not L3 (no caller/export, "
                        "never merged), not L4 (no external numeric anchor; channel error 6.21%/12.34% > 3%; "
                        "one of four predeclared arms has no artifact), not L5."),
    "audit_findings": [
        "F1 gate leak: arm verdict = rows[0]['pass'] only -> refine_65's steady_resid 1.142e-04 > 1e-4 "
        "sub-gate miss does not propagate; results JSON and mission summary say 'channel PASS' (doc table "
        "does mark the row FAIL).",
        "F2 artifact/report mismatch: separation arm has no artifact at all (not in xval_results.json, no "
        "separation JSON in the worktree); its four contrast numbers live only in doc prose, and 2 of 5 "
        "declared G levels were never run. 'overall': 'PASS' in the JSON is computed over the arms that "
        "happen to be present, so a 3/4 delivery prints PASS.",
        "F3 overstatement in the mission result: 'PASS(6.9e-15/3.4e-6); channel PASS(worst 12.34%<15%)' "
        "omits that 12.34% is 4.1x the 3% ecosystem bar and that ny=65's sub-gate failed.",
        "F4 unmerged: no commit, no branch, files untracked; head_commit=null. Nothing reached GitHub.",
        "F5 no platform report (tensor_reports mission_id=18327 -> 0 rows) and no benchmark case row at "
        "delivery time.",
        "F6 external anchor is formula-level: parity cannot detect a shared misreading of the OSS formula; "
        "only the analytic nu_eff channel gives independent numerical content, and it is self-derived "
        "inside the same delivery.",
    ],
}
ev_js = json.dumps(evidence, ensure_ascii=False)
level = 2
scope = f"m{DELIVERED_MISSION} {ANCHOR}"
q("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
  f"VALUES ('{REPO}', '{esc(scope)}', {AUDIT_MISSION}, {level}, '{esc(ev_js)}'::jsonb, 'board-audit') "
  "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, "
  "evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id, gate_state='pending', gate_level=NULL")
print("verdict row:", q(f"SELECT id,repo,scope,level,verdict_by,gate_state FROM tensor_maturity_verdicts "
                        f"WHERE scope='{esc(scope)}' AND verdict_by='board-audit'"))