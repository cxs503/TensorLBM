#!/usr/bin/env python3
"""m17853 board audit — snapshot + provenance verification for m17726.

Collects, programmatically (no hand-typed numbers):

* sha256 of every m17726 deliverable file, and whether git knows it in *any* ref
  of the TensorLBM repository (``git log --all -- <path>`` and
  ``git ls-tree -r <ref>``) -- the "is it landed in the repo" question;
* the predeclared gates recomputed from the shipped ``runs/*/result.json``;
* the finish/runtime facts (mission status, cluster job cj1656 state) read from
  the platform DB.

Writes ``artifacts/snapshot_manifest.json``.

Run: python3 docs/board_audit/m17853/verify_snapshot.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import psycopg

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # /root/tp-agent/work/m17853
REFS = HERE / "refs"
ART = HERE / "artifacts"
ORIG_WT = Path("/root/tp-agent/work/m17726")  # audited mission worktree (read-only use)
ECO = Path("/root/tensor-ecosystem/TensorLBM")
DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")

DELIVERABLES = [
    "src/tensorlbm/wetting/wall_order_parameter_2d.py",
    "tests/test_wall_order_parameter_wetting.py",
    "docs/XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER-PARAMETER.md",
    "xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py",
    "xval/LBM_WALL_ORDER_PARAMETER_WETTING/predeclared.json",
    "xval/LBM_WALL_ORDER_PARAMETER_WETTING/summary.json",
    "xval/LBM_WALL_ORDER_PARAMETER_WETTING/make_figures.py",
    "xval/LBM_WALL_ORDER_PARAMETER_WETTING/oss_runner.py",
    "xval/LBM_WALL_ORDER_PARAMETER_WETTING/oss_src/validation_ebinan92.py",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(args: list[str], cwd: Path) -> str:
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    return r.stdout.strip()


def landing_state(relpath: str) -> dict:
    """Is this path present in the repo HEAD / any ref / the audited branch?"""
    eco_head = (ECO / relpath).exists()
    in_history = git(["log", "--all", "--oneline", "--", relpath], ECO)
    on_branch = git(["ls-tree", "-r", "origin/agent/m17726" if _has_ref("origin/agent/m17726") else "agent/m17726",
                     "--name-only"], ORIG_WT)
    return {
        "path": relpath,
        "exists_in_main_clone": eco_head,
        "in_any_git_ref_history": bool(in_history),
        "in_agent_m17726_tree": relpath in on_branch.splitlines(),
        "untracked_in_m17726_worktree": bool(git(["status", "--porcelain", "--", relpath], ORIG_WT)),
    }


def _has_ref(name: str) -> bool:
    return bool(git(["rev-parse", "--verify", "--quiet", name], ORIG_WT))


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    snap = {
        "audit_mission": 17853,
        "subject_mission": 17726,
        "subject_branch_head": git(["rev-parse", "HEAD"], ORIG_WT),
        "subject_branch_uncommitted": git(["status", "--porcelain"], ORIG_WT).splitlines(),
        "deliverables": [],
        "snapshot_files": {},
    }
    for rel in DELIVERABLES:
        snap["deliverables"].append(landing_state(rel))
        orig = ORIG_WT / rel
        snap["snapshot_files"][rel] = {
            "sha256_original": sha(orig) if orig.exists() else None,
            "sha256_snapshot": sha(REFS / rel) if (REFS / rel).exists() else None,
        }
    # figures / run artefacts
    figs = sorted((REFS / "xval/figures").glob("*.png"))
    snap["figures"] = [{"name": f.name, "sha256": sha(f), "bytes": f.stat().st_size} for f in figs]
    runs = sorted((REFS / "xval/runs").glob("*/result.json"))
    snap["n_shipped_run_results"] = len(runs)

    # predeclared gates recomputed from shipped rows
    rows = []
    for p in runs:
        d = json.loads(p.read_text())
        if d.get("arm") != "port":
            continue
        rows.append({k: d.get(k) for k in
                     ("psi_wall", "interface_width", "theta_deg", "theta_analytic_deg",
                      "err_vs_analytic_pct", "converged", "iterations", "drift")})
    errs = sorted(abs(r["err_vs_analytic_pct"]) for r in rows)
    snap["shipped_gates"] = {
        "n_points": len(rows),
        "n_in_15pct": sum(e <= 15.0 for e in errs),
        "median_abs_err_pct": (errs[(len(errs) - 1) // 2] + errs[len(errs) // 2]) / 2.0,
        "max_abs_err_pct": errs[-1],
        "all_converged": all(r["converged"] for r in rows),
    }
    for r in runs:
        d = json.loads(r.read_text())
        if d.get("arm") == "oss_reference":
            snap["oss_arm"] = {"finite": d.get("finite"), "steps": d.get("steps"),
                               "psi_min": str(d.get("psi_min")), "psi_max": str(d.get("psi_max"))}
        elif d.get("arm") == "lbm":
            snap["lbm_arm"] = {k: d.get(k) for k in
                               ("steps_completed", "finite", "psi_max_final", "dissociation_ratio")}

    con = psycopg.connect(DSN)
    cur = con.cursor()
    cur.execute("select state, rc, left(cmd,90) from cluster_jobs where id=1656")
    r = cur.fetchone()
    snap["cluster_job_cj1656"] = {"state": r[0], "rc": r[1], "cmd": r[2]} if r else None
    cur.execute("select status, finished_at::text, left(result::text, 400) "
                "from tensor_dev_missions where id=17726")
    st, fin, res = cur.fetchone()
    snap["mission_17726"] = {"status": st, "finished_at": fin, "result_head": res}
    cur.execute("select count(*) from tensor_benchmark_cases where anchor_slug like %s",
                ("%WALL-ORDER%",))
    snap["benchmark_rows_for_anchor_before_audit"] = cur.fetchone()[0]

    out = ART / "snapshot_manifest.json"
    out.write_text(json.dumps(snap, indent=2))
    print(out, "written")
    print(json.dumps({k: snap[k] for k in
                      ("shipped_gates", "oss_arm", "lbm_arm", "cluster_job_cj1656",
                       "benchmark_rows_for_anchor_before_audit")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())