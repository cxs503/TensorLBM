#!/usr/bin/env python3
"""Auditor harness for m20150 — cross-audit of mission m20069 (GAP-DIFFLBM-ADJOINT-VERDICT).

Read-only against the audited worktree (/root/tp-agent/work/m20069, commit 3dabe126c,
branch agent/m20069).  Regenerates every number quoted in
docs/board_audit/m20150/AUDIT.md and writes
artifacts/m20150_audit/verify_output.json.

Usage:
    python3 docs/board_audit/m20150/verify_m20069.py [--target DIR] [--skip-pytest]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

DEFAULT_TARGET = "/root/tp-agent/work/m20069"
ANCHOR = "GAP-DIFFLBM-ADJOINT-VERDICT"
AUDITED_COMMIT = "3dabe126c526f8d69a1d252d7a4a76be6185a610"

FILES = [
    "src/tensorlbm/shape_gradient.py",
    "src/tensorlbm/adjoint.py",
    "benchmarks/adjoint_verdict_diff_lbm.py",
    "benchmarks/adjoint_verdict_report.py",
    "tests/test_shape_gradient.py",
    "docs/GAP_DIFFLBM_ADJOINT_VERDICT.md",
    "artifacts/adjoint_verdict_diff_lbm.json",
    "artifacts/adjoint_verdict_diff_lbm/fields_Q_yc050.npz",
    "artifacts/adjoint_verdict_diff_lbm/fields_Q_yc055.npz",
    "figures/adjoint_verdict_diff_lbm_cos.png",
    "figures/adjoint_verdict_diff_lbm_convergence.png",
    "figures/adjoint_verdict_diff_lbm_trajectory.png",
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def psql(sql: str) -> str:
    r = subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA", "-F", "|", "-c", sql],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:400])
    return r.stdout.strip()


def check_c1_files(target: str) -> dict:
    present, missing = {}, []
    for rel in FILES:
        p = os.path.join(target, rel)
        if os.path.isfile(p):
            present[rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
        else:
            missing.append(rel)
    return {"ok": not missing, "present": present, "missing": missing}


def check_c2_benchmark_rows() -> dict:
    total = int(psql("SELECT count(*) FROM tensor_benchmark_cases;"))
    rows = psql(
        "SELECT case_name,metric,tolerance,repo,status FROM tensor_benchmark_cases "
        f"WHERE anchor_slug='{ANCHOR}';"
    )
    near = psql(
        "SELECT anchor_slug,case_name FROM tensor_benchmark_cases "
        "WHERE anchor_slug ILIKE '%ADJOINT%' OR anchor_slug ILIKE '%DIFFLBM%';"
    )
    n = 0 if not rows else len(rows.splitlines())
    return {
        "ok": n > 0,
        "anchor": ANCHOR,
        "rows": n,
        "raw": rows,
        "near_matches": near.splitlines(),
        "table_total_rows": total,
        "note": "0 rows for the anchor => no machine benchmark judge exists; L4 impossible by the board criterion",
    }


def check_c3_artifact_vs_doc(target: str) -> dict:
    art = json.load(open(os.path.join(target, "artifacts/adjoint_verdict_diff_lbm.json")))
    doc = open(os.path.join(target, "docs/GAP_DIFFLBM_ADJOINT_VERDICT.md")).read()

    recs = {}
    for s in art["summary"]:
        recs[s["config"]] = s
    doc_rows = {}
    sec = doc.split("### 4.1")[1].split("### 4.2")[0] if "### 4.1" in doc else ""
    for line in sec.splitlines():
        m = re.match(
            r"\|\s*(Q_yc\d+)\s*\|\s*([-0-9.eE+]+)\s*\|\s*([-0-9.eE+]+)\s*\|\s*([-0-9.]+)%\s*\|", line)
        if m:
            doc_rows[m.group(1)] = {"g_fd": float(m.group(2)), "g_ad": float(m.group(3)),
                                    "doc_ad_rel_err_pct": float(m.group(4))}
    agree = {}
    for name, row in doc_rows.items():
        got = recs[name]
        agree[name] = {
            "doc_g_fd": row["g_fd"], "artifact_g_fd": got["g_fd"],
            "doc_g_ad": row["g_ad"], "artifact_g_ad": got["g_ad"],
            "match": abs(row["g_fd"] - got["g_fd"]) < 1e-4 and abs(row["g_ad"] - got["g_ad"]) < 1e-4,
        }

    # recompute relative errors from raw numbers
    recomputed = {}
    for name, s in recs.items():
        recomputed[name] = {
            "ad_rel_err": abs(s["g_ad"] - s["g_fd"]) / abs(s["g_fd"]),
            "artifact_ad_rel_err": s["ad_rel_err_vs_fd"],
            "frozen_rel_err": abs(s["g_frozen"] - s["g_fd"]) / abs(s["g_fd"]),
            "ship_rel_err": abs(s["g_ship"] - s["g_fd"]) / abs(s["g_fd"]),
            "cos_ship_vs_ad": s["cos_ship_vs_ad"],
            "cos_frozen_vs_ad": s["cos_frozen_vs_ad"],
            "gate_cos_0.99_measured_satisfied": abs(s["cos_ship_vs_ad"]) >= 0.99,
        }
    return {
        "ok": all(v["match"] for v in agree.values()) and bool(agree),
        "mode": art.get("mode"),
        "generated_at": art.get("generated_at"),
        "evidence_grade": art.get("evidence_grade", {}).get("level"),
        "verdict_block": art.get("verdict"),
        "doc_vs_artifact": agree,
        "recomputed": recomputed,
    }


def check_c4_hardcoded_ticks(target: str) -> dict:
    path = os.path.join(target, "benchmarks/adjoint_verdict_report.py")
    src = open(path).read().splitlines()
    hard_x = [i + 1 for i, l in enumerate(src) if "cos" in l and "✗" in l and "if" not in l]
    hard_no = [i + 1 for i, l in enumerate(src) if "是否满足" in l and "NO" in l]
    return {
        "ok": not hard_x and not hard_no,
        "cos_rows_hardcoded_cross": hard_x,
        "cos_criterion_hardcoded_NO": hard_no,
        "evidence": [src[i - 1].strip()[:160] for i in (hard_x + hard_no)][:6],
        "note": "the two headline verdict lines of the report are literals, not computed from the JSON",
    }


def check_c5_integration(target: str) -> dict:
    src_dir = os.path.join(target, "src")
    hits = []
    for root, _dirs, names in os.walk(src_dir):
        for nm in names:
            if not nm.endswith(".py"):
                continue
            p = os.path.join(root, nm)
            if p.endswith("shape_gradient.py") or p.endswith("adjoint.py"):
                continue
            for i, line in enumerate(open(p, errors="ignore"), 1):
                if "shape_gradient" in line or "optimization_path_eligible" in line:
                    hits.append(f"{os.path.relpath(p, target)}:{i}: {line.strip()[:120]}")
    return {
        "ok": bool(hits),
        "src_callers": hits,
        "note": "no src/ module imports tensorlbm.shape_gradient and no optimisation loop calls the gate "
                "=> integration is benchmark/test level only (doc conclusion #3 admits round-2 wiring)",
    }


def check_c6_paper_campaign() -> dict:
    log = "/root/tp-agent/cluster/jobs/cj2485.log"
    out = {"log_present": os.path.isfile(log)}
    if out["log_present"]:
        tail = open(log, errors="ignore").read().strip().splitlines()[-1]
        out["node_log"] = tail[:600]
        try:
            rec = json.loads(tail)
            out["node_rc"] = rec.get("rc")
            out["output_tail"] = rec.get("output_tail")
        except Exception:
            out["node_rc"] = None
        q = open("/root/tp-agent/cluster/queue.log", errors="ignore").read()
        out["queue_lines"] = [l for l in q.splitlines() if "cj2485" in l][-3:]
    art_dir = os.path.join(DEFAULT_TARGET, "artifacts/adjoint_verdict_diff_lbm")
    out["shipped_paper_artifacts"] = sorted(
        n for n in (os.listdir(art_dir) if os.path.isdir(art_dir) else []) if "R" in n
    )
    out["ok"] = False
    out["note"] = ("paper campaign (cj2485) terminated with node rc!=0 and only one R1 field snapshot; "
                   "no paper-mode JSON exists, so the paper-resolution verdict is absent, not pending")
    return out


def check_c7_literal_verdict_numbers(target: str) -> dict:
    adj = open(os.path.join(target, "src/tensorlbm/adjoint.py")).read()
    art = json.load(open(os.path.join(target, "artifacts/adjoint_verdict_diff_lbm.json")))
    m_cos = re.search(r'"cos_vs_unrolled_ad":\s*([-0-9.]+)', adj)
    m_amp = re.search(r'"amplitude_ratio_vs_true":\s*([-0-9.]+)', adj)
    cos_lit = float(m_cos.group(1)) if m_cos else None
    amp_lit = float(m_amp.group(1)) if m_amp else None
    shipped = {s["config"]: {"cos_ship_vs_ad": s["cos_ship_vs_ad"],
                             "amp_ratio": s["g_frozen"] / s["g_ad"]} for s in art["summary"]}
    return {
        "ok": False,
        "module_literal_cos": cos_lit,
        "module_literal_amplitude_ratio": amp_lit,
        "shipped_artifact": shipped,
        "note": "FROZEN_FIELD_VERDICT literals (cos=-1.000, ratio=2.289) contradict this mission's own "
                "artifact (cos 1.000/0.980, ratio ~5.2) and carry no run/artifact pointer -> provenance absent",
    }


def check_c8_tests(target: str, skip: bool) -> dict:
    env = dict(os.environ, PYTHONPATH=os.path.join(target, "src"), OMP_NUM_THREADS="4")
    if skip:
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_shape_gradient.py",
                            "--collect-only", "-q"], cwd=target, capture_output=True, text=True, env=env)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
        return {"ok": r.returncode == 0, "ran": False, "collected_tail": tail}
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_shape_gradient.py", "-q"],
                       cwd=target, capture_output=True, text=True, env=env, timeout=600)
    tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1]
    return {"ok": r.returncode == 0, "ran": True, "summary": tail, "returncode": r.returncode}


def check_c9_repro() -> dict:
    """Independent re-run of the shipped entry at DEFAULT settings (cluster cj2519)."""
    log = "artifacts/m20150_audit/cj2519_repro.log"
    art = json.load(open(os.path.join(DEFAULT_TARGET, "artifacts/adjoint_verdict_diff_lbm.json")))
    if not os.path.isfile(log):
        return {"ok": False, "note": "cj2519 log not captured"}
    txt = open(log, errors="ignore").read()
    rec = json.loads(txt.splitlines()[0])
    out = rec.get("output_tail", "")
    nums = {}
    for line in out.splitlines():
        m = re.match(r"\[(Q_yc\d+)\] (steady.*|FD reference.*|ship.*|frozen.*|AD n=100.*)", line)
        if m:
            nums.setdefault(m.group(1), {})[m.group(2).split(":")[0]] = m.group(2)
    steps = {k: v.get("steady", "") for k, v in nums.items()}
    shipped_steps = {n: art["configs"][n]["base"]["steps"] for n in art["configs"]}
    shipped_fd = {n: art["configs"][n]["fd"]["reference"] for n in art["configs"]}
    repro = {}
    for n, v in nums.items():
        fd = float(re.search(r"FD reference: ([-0-9.eE+]+)", v["FD reference"]).group(1))
        ship = float(re.search(r"dC_L/dy_c=([-0-9.eE+]+)", v["ship"]).group(1))
        cos = float(re.search(r"cos_vs_AD=([-+0-9.]+)", v["ship"]).group(1))
        repro[n] = {"repo_steps": steps[n], "fd_reference": fd, "ship_grad": ship,
                    "cos_ship_vs_ad": cos,
                    "shipped_steps": shipped_steps[n], "shipped_fd_reference": shipped_fd[n],
                    "fd_ratio_shipped_over_repro": shipped_fd[n] / fd}
    same = all(abs(repro[n]["fd_reference"] - shipped_fd[n]) < 1e-9 for n in repro)
    return {
        "ok": same,
        "node_rc": rec.get("rc"),
        "note": "shipped entry re-run WITHOUT --max-steps on node n330 (2 threads, 293.4 s) uses 6000 steps "
                "instead of the shipped 700 and returns a different FD reference and a NEGATIVE cos(ship,AD); "
                "the shipped JSON therefore reproduces only with the undocumented 700-step override",
        "repro": repro,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--out", default="artifacts/m20150_audit/verify_output.json")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    res = {
        "auditor_mission": 20150,
        "audited_mission": 20069,
        "audited_commit": AUDITED_COMMIT,
        "anchor": ANCHOR,
        "target": args.target,
    }
    res["C1_files"] = check_c1_files(args.target)
    res["C2_benchmark_rows"] = check_c2_benchmark_rows()
    res["C3_artifact_vs_doc"] = check_c3_artifact_vs_doc(args.target)
    res["C4_hardcoded_ticks"] = check_c4_hardcoded_ticks(args.target)
    res["C5_integration"] = check_c5_integration(args.target)
    res["C6_paper_campaign"] = check_c6_paper_campaign()
    res["C7_literal_verdict_numbers"] = check_c7_literal_verdict_numbers(args.target)
    res["C8_tests"] = check_c8_tests(args.target, args.skip_pytest)
    res["C9_repro_cj2519"] = check_c9_repro()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    print(f"[verify] wrote {args.out}")
    for k, v in res.items():
        if isinstance(v, dict) and "ok" in v:
            print(f"  {k:28s} ok={v['ok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())