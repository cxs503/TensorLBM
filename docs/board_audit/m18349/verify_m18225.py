#!/usr/bin/env python3
"""m18349 交差核验（只读）：m18225 XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO

判定口径（基准判据 = tensor_benchmark_cases；报告好看不算数）：
  A 取证：m18225 本单是否留下 (a) 仓库内实现文件 (b) run_artifacts (c) 可执行 repro_cmd
  B 基准：tensor_benchmark_cases 是否登记该 anchor 的 ref/got/tolerance
  C 三件套的实际状态（m18225 工作树 + 仓库 main 双视角）
本脚本不写任何东西，只打印 JSON；结果落 docs/board_audit/m18349/verify_m18225.out。
"""
import json
import pathlib
import re
import subprocess

WT = pathlib.Path("/root/tp-agent/work/m18225")
PSQL = ["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-F", "|", "-c"]


def sql(q):
    r = subprocess.run(PSQL + [q], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return [["ERR", r.stderr.strip()[:200]]]
    return [l.split("|") for l in r.stdout.splitlines() if l.strip()]


def sh(cmd, cwd=None, timeout=600):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()




def main():
    res = {}

    # ── A1 本单是否有仓库内提交 ─────────────────────────────────────────
    rc, out, _ = sh("git rev-list --count main..HEAD", cwd=WT)
    res["branch_commits_vs_main"] = int(out or -1)
    rc, out, _ = sh("git log --oneline -1", cwd=WT)
    res["branch_head"] = out
    rc, out, _ = sh("git status --porcelain", cwd=WT)
    untracked = [l[3:] for l in out.splitlines()]
    res["worktree_dirty_files"] = untracked
    res["worktree_dirty_are_scratch_only"] = all(re.fullmatch(r"\.tmp_kb\d+\.py", f) for f in untracked)

    # ── A2 三件套是否真的在树里 ─────────────────────────────────────────
    art = WT / "benchmarks/eos_maxwell/out/results.json"
    res["run_artifacts_out_results_json_exists"] = art.exists()
    res["run_artifacts_bytes"] = art.stat().st_size if art.exists() else 0
    res["run_artifacts_git_tracked"] = sh(
        "git ls-files benchmarks/eos_maxwell/out", cwd=WT)[1].splitlines()
    res["merged_tables_present"] = {
        p: (WT / "benchmarks/eos_maxwell" / p).exists()
        for p in ("CRITERION.md", "RESULTS.md", "probe_lib.py", "run.py")
    }
    res["impl_introduced_by"] = {}
    for p in ("benchmarks/eos_maxwell/probe_lib.py", "benchmarks/eos_maxwell/run.py",
              "tests/test_eos_maxwell_stability.py"):
        res["impl_introduced_by"][p] = sh(
            "git log --diff-filter=A --format='%h %s' -1 -- " + p, cwd=WT)[1]

    # ── A3 repro_cmd 可执行性（只跑 --help，不起算） ─────────────────────
    rc, out, err = sh("python3 benchmarks/eos_maxwell/run.py --help", cwd=WT, timeout=180)
    res["repro_cli_rc"] = rc
    res["repro_cli_has_modes"] = "--mode" in out

    # ── B 基准裁判：tensor_benchmark_cases ──────────────────────────────
    for slug in ("XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO",):
        res["benchmark_cases_" + slug] = sql(
            "SELECT case_name,metric,tolerance,status FROM tensor_benchmark_cases "
            "WHERE anchor_slug='%s'" % slug)
    res["benchmark_cases_like_eos_maxwell"] = sql(
        "SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug ILIKE '%MAXWELL%' "
        "OR anchor_slug ILIKE '%EOS%'")

    # ── C 本单报告 + 原单事实 ───────────────────────────────────────────
    res["reports_for_mission_18225"] = sql(
        "SELECT id,title FROM tensor_reports WHERE mission_id=18225")
    res["mission_18225"] = sql(
        "SELECT status,fail_stage,report_owed,coalesce(report_id::text,''),result->>'head_commit',"
        "result->>'tool_calls' FROM tensor_dev_missions WHERE id=18225")
    res["cluster_cj1834"] = sql(
        "SELECT id,state,mid,left(cmd,50) FROM cluster_jobs WHERE id=1834")

    # ── D 树内既有 provenance 测试（审计侧独立复跑，非本单交付） ─────────
    rc, out, err = sh("python3 -m pytest tests/test_eos_maxwell_stability.py -q -rs "
                      "-k 'results_artefact_matches or criterion_artefact_matches or "
                      "maxwell_construction_hits_target or consumer_anchors_exist'", cwd=WT)
    res["pytest_provenance_rc"] = rc
    res["pytest_provenance_tail"] = out.splitlines()[-1] if out else err[-200:]
    res["pytest_provenance_skips"] = [l for l in out.splitlines() if l.startswith("SKIPPED")]

    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()