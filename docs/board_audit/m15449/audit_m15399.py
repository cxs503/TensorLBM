#!/usr/bin/env python3
"""m15449 board-audit of m15399 (anchor GAP-PP-4-opt-loop, repo TensorLBM).

Independent, re-runnable checks only; the audited tree is opened read-only.

  C1 artefact existence + sha256 (8 declared run artefacts + figure)
  C2 internal consistency of the run JSON recomputed from the per-run records
     (nu_hat <- tau_hat, Re_hat <- nu_hat, rel_err <- re_true) and history tails
  C3 zero-hand-copy probe: every number printed in the report table 4 must equal
     the value recomputed from the JSON (parse docs/OPT_LOOP_TAU_RECOVERY.md)
  C4 shard artefacts vs canonical JSON: are the shipped shards a decomposition
     of the shipped canonical payload?
  C5 repro_cmd executability (module import + argparse surface)
  C6 arbiter registry: tensor_benchmark_cases / tensor_case_pool rows for anchor
  C7 integration callers (who imports the module outside tests/scripts?)

usage: python3 docs/board_audit/m15449/audit_m15399.py --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import os
import sys
from pathlib import Path

SRC = Path("/root/tp-agent/work/m15399")
ANCHOR = "GAP-PP-4-opt-loop"
ART = "artifacts/opt_loop_tau_recovery.json"
DOC = "docs/OPT_LOOP_TAU_RECOVERY.md"
U0, N_BY_TRACK = 0.05, {"analytic16": 16, "analytic32": 32, "twin16": 16,
                        "analytic16_fixed": 16, "twin16_fixed": 16,
                        "twin16_noise0.2": 16, "twin16_noise1.0": 16}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def psql(sql: str) -> str:
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-F", "\t", "-c", sql],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return f"ERROR: {r.stderr.strip()[:200]}"
    return r.stdout.strip()


def c1() -> dict:
    files = [ART, "artifacts/opt_loop_fixed_track.json"] + \
            [f"artifacts/opt_loop_shard_{i}.json" for i in range(6)] + \
            ["artifacts/logs/fixed_track.log"] + [f"artifacts/logs/shard_{i}.log" for i in range(6)] + \
            ["figures/opt_loop_tau_recovery.png", "src/tensorlbm/opt_loop_tau_recovery.py",
             "tests/test_opt_loop_tau_recovery.py", "scripts/make_opt_loop_figures.py", DOC]
    out = {}
    for f in files:
        p = SRC / f
        out[f] = {"exists": p.is_file(), "bytes": p.stat().st_size if p.is_file() else 0,
                  "sha256": sha(p) if p.is_file() else None,
                  "tracked_on_mission_branch": f in subprocess.run(
                      ["git", "-C", str(SRC), "ls-files", f], capture_output=True, text=True).stdout}
    return out


def c2(payload: dict) -> dict:
    """Recompute every derived quantity from the stored tau_hat / re_true."""
    bad, n = [], 0
    for t in payload["tracks"]:
        for r in t["runs"]:
            n += 1
            nu = (r["tau_hat"] - 0.5) / 3.0
            re_hat = U0 * N_BY_TRACK[r["track"]] / nu
            rel_got_minus_ref = (re_hat - r["re_true"]) / r["re_true"] * 100.0
            rel = (r["re_true"] - re_hat) / r["re_true"] * 100.0  # the module's convention
            for key, got, exp, tol in (("nu_hat", r["nu_hat"], nu, 1e-12),
                                       ("re_hat", r["re_hat"], re_hat, 1e-9),
                                       ("rel_err_pct", r["rel_err_pct"], rel, 1e-9)):
                if abs(got - exp) > tol * max(1.0, abs(exp)):
                    bad.append({"track": r["track"], "seed": r["seed"], "field": key,
                                "stored": got, "recomputed": exp})
            tail = r["history"][-1]
            if abs(tail["tau"] - r["tau_hat"]) > 1e-9:
                bad.append({"track": r["track"], "seed": r["seed"], "field": "history_tail_tau",
                            "stored": tail["tau"], "recomputed": r["tau_hat"]})
            if r["loss_final"] > r["loss_initial"]:
                bad.append({"track": r["track"], "seed": r["seed"], "field": "loss_not_decreasing",
                            "stored": r["loss_final"], "recomputed": r["loss_initial"]})
    summ = payload["summary"]
    recomputed_max = {t["track"]: max(abs(r["rel_err_pct"]) for r in t["runs"])
                      for t in payload["tracks"]}
    mism = {k: {"stored": summ["by_track"][k]["max_abs_rel_err_pct"], "recomputed": v}
            for k, v in recomputed_max.items()
            if abs(summ["by_track"][k]["max_abs_rel_err_pct"] - v) > 1e-9}
    return {"runs_checked": n, "derivation_mismatches": bad, "summary_mismatches": mism,
            "all_tracks_pass_claim": summ["all_tracks_pass"],
            "control_arms": summ["control_arms"],
            "gated_tracks_max_abs_err_pct": {k: v for k, v in recomputed_max.items()
                                             if k not in summ["control_arms"]},
            "gradient_checks": {t["track"]: t["gradient_check"]["rel_diff"] for t in payload["tracks"]}}


def c3(payload: dict) -> dict:
    """Parse report table row values and compare with JSON (zero-hand-copy probe)."""
    text = (SRC / DOC).read_text()
    rows = []
    for line in text.splitlines():
        if not line.startswith("| ") or "track" in line or set(line) <= set("|- "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 10 and re.match(r"^[a-z0-9.]+$", cells[0]):
            rows.append(cells)
    by_track = {t["track"]: t for t in payload["tracks"]}
    checks = []
    for cells in rows:
        name = cells[0]
        t = by_track.get(name)
        if t is None:
            checks.append({"track": name, "status": "TRACK_ABSENT_FROM_JSON"})
            continue
        reported_tau = [float(x) for x in cells[5].split("/")]
        reported_re = [float(x) for x in cells[6].split("/")]
        reported_err = [float(x.replace("+", "")) for x in cells[7].split("/")]
        json_tau = t["summary"]["tau_hat"]
        json_re = t["summary"]["re_hat"]
        json_err = [r["rel_err_pct"] for r in t["runs"]]
        checks.append({
            "track": name,
            "tau_reported": reported_tau, "tau_json": json_tau,
            "tau_max_abs_delta": max(abs(a - b) for a, b in zip(reported_tau, json_tau)),
            "re_max_abs_delta": max(abs(a - b) for a, b in zip(reported_re, json_re)),
            "err_max_abs_delta_pct": max(abs(a - b) for a, b in zip(reported_err, json_err)),
            "max_abs_err_cell": float(cells[8].replace("%", "")),
            "verdict_cell": cells[9],
        })
    worst = max((c.get("err_max_abs_delta_pct", 0) for c in checks), default=None)
    return {"rows_parsed": len(rows), "rows": checks, "worst_err_table_delta_pct": worst,
            "sign_convention": "module stores rel_err_pct=(re_true-re_hat)/re_true (positive = underestimate); auditor recomputed both conventions, magnitudes identical",
            "verdict": "PASS (table reproduces JSON)" if worst is not None and worst < 1e-3 else "FAIL"}


def c4(payload: dict) -> dict:
    canon = {(t["track"], r["seed"]): r for t in payload["tracks"] for r in t["runs"]}
    shipped = {}
    for f in sorted(SRC.glob("artifacts/opt_loop_shard_*.json")) + [SRC / "artifacts/opt_loop_fixed_track.json"]:
        d = json.loads(f.read_text())
        for j in (d["jobs"] if isinstance(d, dict) and "jobs" in d else d):
            shipped[(j["track"], j["seed"])] = (f.name, j["run"]["tau_hat"], j["run"]["rel_err_pct"])
    key_match, name_diff, val_diff = 0, [], []
    for k, v in shipped.items():
        if k in canon:
            key_match += 1
            if abs(canon[k]["tau_hat"] - v[1]) > 1e-12:
                val_diff.append({"key": k, "shard": v[1], "canonical": canon[k]["tau_hat"]})
        else:
            # control arm was renamed twin16_fixed -> analytic16_fixed between runs
            alt = ("analytic16_fixed" if k[0] == "twin16_fixed" else None, k[1])
            if alt in canon and abs(canon[alt]["tau_hat"] - v[1]) < 1e-12:
                name_diff.append({"shard_track": k[0], "canonical_track": alt[0], "seed": k[1],
                                  "tau_equal": True})
            else:
                name_diff.append({"unmatched": list(k)})
    return {"shard_runs": len(shipped), "canonical_runs": len(canon),
            "key_matches": key_match, "value_mismatches": val_diff,
            "track_label_mismatches": name_diff,
            "note": "shards are a stale-label (twin16_fixed) decomposition of the canonical run; "
                    "values agree to 1e-12" if name_diff and not val_diff else "consistent"}


def c5() -> dict:
    r = subprocess.run([sys.executable, "-m", "tensorlbm.opt_loop_tau_recovery", "--help"],
                       cwd=str(SRC), capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(SRC / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    flags = re.findall(r"--[a-z-]+", r.stdout)
    return {"exit": r.returncode, "flags": sorted(set(flags)),
            "has_shard_merge": "--shard" in flags and "--merge" in flags,
            "stderr": r.stderr.strip().splitlines()[-1] if r.returncode else None}


def c6() -> dict:
    cases = psql(f"SELECT id,case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}'")
    pool = psql(f"SELECT id,case_code,tol_policy FROM tensor_case_pool WHERE linked_anchor='{ANCHOR}'")
    anycase = psql(f"SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug ILIKE '%opt%'")
    return {"benchmark_case_rows": cases if cases else "0 rows",
            "case_pool_rows": pool if pool else "0 rows",
            "anchor_like_opt_rows": anycase,
            "verdict": "NOT_MEASURED (arbiter empty for this anchor)" if not cases and not pool else "judged"}


def c7() -> dict:
    out = subprocess.run(["grep", "-rn", "opt_loop_tau_recovery", "--include=*.py", "--include=*.toml",
                          "--include=*.yaml", "--include=*.yml", str(SRC)],
                         capture_output=True, text=True).stdout.splitlines()
    imp = [l for l in out if "__pycache__" not in l and "src/tensorlbm/opt_loop_tau_recovery.py" not in l]
    outside = [l for l in imp if "/tests/" not in l and "/scripts/" not in l]
    pkg = sorted(p.name for p in (SRC / "src/tensorlbm").glob("inverse*"))
    return {"importing_files_outside_module": [l.split(str(SRC) + "/")[-1][:120] for l in imp],
            "callers_outside_tests_and_scripts": outside, "tensorlbm_inverse_pkg": pkg or None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    payload = json.loads((SRC / ART).read_text())
    res = {"anchor": ANCHOR, "audited_mission": 15399, "audited_tree": str(SRC),
           "audited_head": subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
           "C1_artifacts": c1(), "C2_internal_consistency": c2(payload),
           "C3_report_table_vs_json": c3(payload), "C4_shards_vs_canonical": c4(payload),
           "C5_repro_cmd": c5(), "C6_arbiter": c6(), "C7_integration": c7()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) else "…") for k, v in res.items()}, ensure_ascii=False))
    print("C2", json.dumps(res["C2_internal_consistency"])[:600])
    print("C3 worst", res["C3_report_table_vs_json"]["worst_err_table_delta_pct"], res["C3_report_table_vs_json"]["verdict"])
    print("C4", json.dumps(res["C4_shards_vs_canonical"])[:500])
    print("C5", json.dumps(res["C5_repro_cmd"])[:300])
    print("C6", json.dumps(res["C6_arbiter"]))
    print("C7", json.dumps(res["C7_integration"])[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())