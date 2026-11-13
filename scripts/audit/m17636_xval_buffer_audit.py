#!/usr/bin/env python3
"""Board audit m17636 - m17599 ``XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT`` (TensorLBM).

Cross-check only: no deliverable of the audited mission is modified.  Every
number below is re-derived from a primary source - the audited work tree, a
fresh re-run of the delivered entry, or the platform DB - nothing is typed in
by hand.

Checks
------
S1 static  : entry config vs recorded artifact config (repro mismatch)
S2 static  : ``mass_drift_tl = abs(x - x) / x`` self-cancelling metric
S3 static  : is the "OSS reference" the actual haimingz/vivsim code?
S4 static  : tests referencing the deliverable / tracked changes of the mission
R1 runtime : fresh run of both solvers at the *artifact* config -> error metrics
R2 runtime : fresh run vs recorded artifact numbers (fabrication probe)
D1 db      : tensor_benchmark_cases rows for the anchor (benchmark referee)
D2 db      : mission row (status / accepted)

Usage::

    python scripts/audit/m17636_xval_buffer_audit.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "artifacts" / "m17636_audit"
AUDITED = Path("/root/tp-agent/work/m17599")
DELIV = AUDITED / "benchmarks" / "xval_buffer_layer_2to1"
ARTIFACT = DELIV / "artifacts" / "xval_buffer_layer_2to1.json"
ANCHOR = "XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT"
AUDITED_MISSION = 17599
AUDIT_MISSION = 17636

sys.path.insert(0, str(DELIV))
sys.path.insert(0, str(REPO_ROOT / "src"))


def psql(sql: str) -> str:
    cmd = ["su", "postgres", "-c", f'psql -d tensorplatform -t -A -F"|" -c "{sql}"']
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def sh(cmd: str, cwd: Path) -> str:
    return subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True,
                          text=True).stdout.strip()


# --------------------------------------------------------------------------- S1-S4
def static_checks() -> dict:
    src = (DELIV / "run_xval.py").read_text()
    art = json.loads(ARTIFACT.read_text())

    # S1: literals inside main() vs the recorded artifact config
    main_body = src[src.index("def main():"):]
    entry_cfg = {k: (int(m.group(1)) if (m := re.search(rf"\b{k}\s*=\s*(\d+)", main_body)) else None)
                 for k in ("coarse", "strip", "steps")}
    art_cfg = art["config"]
    repro_mismatch = (entry_cfg["coarse"] != art_cfg["coarse"]
                      or entry_cfg["strip"] != art_cfg["strip_coarse"]
                      or entry_cfg["steps"] != art_cfg["steps"])

    # S2: self-cancelling mass-drift expression
    m = re.search(r"mass_drift_tl\s*=\s*(.+)", src)
    mass_expr = m.group(1).strip() if m else ""
    line_no = src[:m.start()].count("\n") + 1 if m else None

    # S3: is any vivsim code vendored / pinned anywhere in the audited tree?
    vendors = sh("grep -rIl vivsim --include='*.py' --include='*.md' --include='*.txt' "
                 "--include='*.toml' . 2>/dev/null | grep -v xval_buffer_layer_2to1 | head -5",
                 AUDITED)
    impl_class = sh("grep -n '^class VivsimBufferLayer2to1' run_xval.py", DELIV)
    anchor_meta = sh("grep -nE 'github|commit|sha|version|pip install' run_xval.py | head -5", DELIV)

    # S4: tests + tracked footprint
    deliv_tests = sh("ls tests 2>/dev/null; grep -rln 'xval_buffer_layer_2to1' ../../tests 2>/dev/null", DELIV)
    tracked = sh("git status --porcelain", AUDITED)
    branch = sh("git branch --show-current", AUDITED)
    ahead = sh("git log --oneline origin/main..HEAD | wc -l", AUDITED)

    return {
        "entry_config_in_main": entry_cfg,
        "artifact_config": {"coarse": art_cfg["coarse"], "strip_coarse": art_cfg["strip_coarse"],
                            "steps": art_cfg["steps"]},
        "repro_config_mismatch": repro_mismatch,
        "mass_drift_tl_expression": mass_expr,
        "mass_drift_tl_line": line_no,
        "mass_drift_tl_aliases": "abs(x - x) / x == 0 for every x",
        "vivsim_sources_vendored": vendors.splitlines() if vendors else [],
        "oss_reference_impl": impl_class,
        "oss_reference_provenance_lines": anchor_meta.splitlines() if anchor_meta else [],
        "deliverable_internal_tests": deliv_tests.splitlines() if deliv_tests else [],
        "audited_git_status": tracked.splitlines(),
        "audited_branch": branch,
        "audited_commits_ahead_of_main": int(ahead or 0),
    }


# --------------------------------------------------------------------------- R1-R2
def runtime_checks() -> dict:
    from tensorlbm.amr_cavity2d import CavityAMR2D, CavityAMR2DConfig, ghia_comparison
    from tensorlbm.lid_driven_cavity import GHIA_RE1000
    from scipy.interpolate import interp1d

    import run_xval as rx

    u_lid, coarse, strip, steps = 0.06, 32, 8, 2000

    tl = CavityAMR2D(CavityAMR2DConfig(coarse=coarse, strip_coarse=strip, re=1000.0,
                                      u_lid=u_lid, reflux=True))
    inv0_tl = tl.inventory()
    for _ in range(steps):
        tl.step()
    inv1_tl = tl.inventory()
    prof_tl = tl.centreline_profiles()

    oss = rx.VivsimBufferLayer2to1(coarse=coarse, strip_coarse=strip, re=1000.0, u_lid=u_lid)
    inv0_oss = oss.inventory()
    for _ in range(steps):
        oss.step()
    inv1_oss = oss.inventory()
    prof_oss = oss.centreline_profiles()

    # exactly the delivered comparison pipeline (run_xval.py:378-387)
    fu = interp1d(prof_oss["y"], prof_oss["u"], kind="linear", fill_value="extrapolate")
    fv = interp1d(prof_oss["x"], prof_oss["v"], kind="linear", fill_value="extrapolate")
    u_oss = fu(prof_tl["y"])
    v_oss = fv(prof_tl["x"])
    du = np.abs(prof_tl["u"] - u_oss)
    dv = np.abs(prof_tl["v"] - v_oss)
    l2_u = float(np.sqrt(np.mean((prof_tl["u"] - u_oss) ** 2)))
    l2_v = float(np.sqrt(np.mean((prof_tl["v"] - v_oss) ** 2)))

    g_tl = ghia_comparison(prof_tl, GHIA_RE1000, u_lid)
    g_oss = ghia_comparison(prof_oss, GHIA_RE1000, u_lid)

    art = json.loads(ARTIFACT.read_text())

    def delta(got, rec):
        return {"auditor_fresh": got, "artifact_recorded": rec,
                "abs_diff": (abs(got - rec) if isinstance(rec, (int, float)) else None),
                "ratio_got_over_rec": (got / rec if isinstance(rec, (int, float)) and rec else None)}

    return {
        "config": {"coarse": coarse, "strip_coarse": strip, "steps": steps, "re": 1000.0,
                   "u_lid": u_lid},
        "inventory": {
            "tensorlbm_initial": inv0_tl, "tensorlbm_final": inv1_tl,
            "tensorlbm_mass_loss_pct_of_initial": 100.0 * (1.0 - inv1_tl["total"] / inv0_tl["total"]),
            "oss_initial": inv0_oss, "oss_final": inv1_oss,
            "oss_mass_loss_pct_of_initial": 100.0 * (1.0 - inv1_oss["total"] / inv0_oss["total"]),
        },
        "fresh_vs_artifact": {
            "tensorlbm_inventory_total": delta(inv1_tl["total"], art["tensorlbm"]["inventory"]["total"]),
            "oss_inventory_total": delta(inv1_oss["total"], art["oss_reference"]["inventory"]["total"]),
            "max_du_pct_of_u_lid": delta(float(du.max() / u_lid * 100),
                                         art["comparison"]["max_du_pct_of_u_lid"]),
            "max_dv_pct_of_u_lid": delta(float(dv.max() / u_lid * 100),
                                         art["comparison"]["max_dv_pct_of_u_lid"]),
            "l2_u": delta(l2_u, art["comparison"]["l2_u"]),
            "l2_v": delta(l2_v, art["comparison"]["l2_v"]),
            "tensorlbm_ghia_max_abs_dev_u": delta(g_tl["max_abs_dev_u"],
                                                  art["tensorlbm"]["ghia"]["max_abs_dev_u"]),
            "oss_ghia_max_abs_dev_u": delta(g_oss["max_abs_dev_u"],
                                            art["oss_reference"]["ghia"]["max_abs_dev_u"]),
        },
        "fresh_ghia_vs_re1000": {
            "tensorlbm": {k: g_tl[k] for k in
                          ("max_abs_dev_u", "max_abs_dev_v", "max_abs_dev", "max_dev_pct_of_u_lid")},
            "oss_reference": {k: g_oss[k] for k in
                              ("max_abs_dev_u", "max_abs_dev_v", "max_abs_dev", "max_dev_pct_of_u_lid")},
        },
        "criterion": {
            "declared": "<15% on centreline profiles (mission goal)",
            "artifact_pass_criterion_15pct": art["comparison"]["pass_criterion_15pct"],
            "auditor_fresh_pass_15pct": bool(du.max() / u_lid * 100 < 15 and dv.max() / u_lid * 100 < 15),
            "auditor_fresh_margin_over_tolerance": float(max(du.max(), dv.max()) / u_lid * 100 / 15.0),
        },
        "field_excursion": {
            "u_lid": u_lid,
            "tensorlbm_u_range": [float(np.min(prof_tl["u"])), float(np.max(prof_tl["u"]))],
            "oss_u_range": [float(np.min(prof_oss["u"])), float(np.max(prof_oss["u"]))],
            "tensorlbm_peak_over_u_lid": float(np.max(np.abs(prof_tl["u"])) / u_lid),
            "oss_peak_over_u_lid": float(np.max(np.abs(prof_oss["u"])) / u_lid),
        },
    }


# --------------------------------------------------------------------------- D1-D2
def db_checks() -> dict:
    cases = psql(f"SELECT case_name,ref_source,metric,tolerance,status FROM tensor_benchmark_cases "
                 f"WHERE anchor_slug='{ANCHOR}';")
    total = psql("SELECT count(*) FROM tensor_benchmark_cases;")
    mission = psql(f"SELECT status||'|'||coalesce(finished_at::text,'')||'|'||"
                   f"coalesce(result->>'accepted','') FROM tensor_dev_missions WHERE id={AUDITED_MISSION};")
    return {
        "benchmark_cases_for_anchor": cases.splitlines() if cases else [],
        "benchmark_cases_total": int(total or 0),
        "mission_row": mission,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "audit_mission": AUDIT_MISSION,
        "audited_mission": AUDITED_MISSION,
        "anchor": ANCHOR,
        "repo": "TensorLBM",
        "static": static_checks(),
        "runtime": runtime_checks(),
        "db": db_checks(),
    }

    fv = result["runtime"]["fresh_vs_artifact"]
    result["findings"] = {
        "F1_no_registered_benchmark_case": not result["db"]["benchmark_cases_for_anchor"],
        "F2_own_criterion_failed": not result["runtime"]["criterion"]["auditor_fresh_pass_15pct"],
        "F3_error_metrics_understated_10x": {
            "max_du_pct": fv["max_du_pct_of_u_lid"]["ratio_got_over_rec"],
            "max_dv_pct": fv["max_dv_pct_of_u_lid"]["ratio_got_over_rec"],
            "recorded_max_du_pct": fv["max_du_pct_of_u_lid"]["artifact_recorded"],
            "fresh_max_du_pct": fv["max_du_pct_of_u_lid"]["auditor_fresh"],
        },
        "F4_l2_recorded_as_zero": {
            "l2_u_recorded": fv["l2_u"]["artifact_recorded"], "l2_u_fresh": fv["l2_u"]["auditor_fresh"],
            "l2_v_recorded": fv["l2_v"]["artifact_recorded"], "l2_v_fresh": fv["l2_v"]["auditor_fresh"],
        },
        "F5_ghia_recorded_as_zero": {
            "tensorlbm_recorded": fv["tensorlbm_ghia_max_abs_dev_u"]["artifact_recorded"],
            "tensorlbm_fresh": fv["tensorlbm_ghia_max_abs_dev_u"]["auditor_fresh"],
        },
        "F6_mass_drift_metric_self_cancelling": result["static"]["mass_drift_tl_expression"],
        "F7_tensorlbm_mass_loss_vs_report_claim": {
            "tensorlbm_mass_loss_pct": result["runtime"]["inventory"]["tensorlbm_mass_loss_pct_of_initial"],
            "oss_mass_loss_pct": result["runtime"]["inventory"]["oss_mass_loss_pct_of_initial"],
            "report_claim": "TensorLBM inventory reflux suppresses mass drift; OSS reference loses mass",
        },
        "F8_external_anchor_not_actual_oss": {
            "vendored_vivsim_files": result["static"]["vivsim_sources_vendored"],
            "reference_class": result["static"]["oss_reference_impl"],
        },
        "F9_repro_cmd_config_mismatch": result["static"]["repro_config_mismatch"],
        "F10_no_tests_no_library_change": {
            "tests_in_deliverable": result["static"]["deliverable_internal_tests"],
            "commits_ahead_of_main": result["static"]["audited_commits_ahead_of_main"],
            "tracked_changes": result["static"]["audited_git_status"],
        },
    }

    (OUT / "audit_m17636.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result["findings"], indent=2, default=str))
    print(f"\n[wrote] {OUT / 'audit_m17636.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())