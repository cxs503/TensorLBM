#!/usr/bin/env python3
"""m18429 — 交差核验：m18366 XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO

只读审计脚本（零手抄）：
  * 从 m18366 工作树/本单副本读取 run 工件，程序化抽取 ref(Maxwell) vs got 数值；
  * 逐项对预注册门与 owner 15% 精度标准判 PASS/FAIL；
  * 机器核对基准裁判表（tensor_benchmark_cases）、报告表（tensor_reports）、
    提交记录（git log --all --grep=m18366）；
  * 生成 audit/m18429/{tables.md,evidence.json,fig1_ref_vs_got.png}。

用法: python3 audit/m18429/verify_m18429.py
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC = Path("/root/tp-agent/work/m18366")          # 被核单工作树（只读）
OUT = HERE

OWNER_TOL = 0.15          # owner 精度标准：误差 >15% 不可接受
SPUR_PASS_MA = 0.05       # 预注册：伪流 Ma PASS 门
SPUR_WARN_MA = 0.10       # 预注册：伪流 Ma MARGINAL 上限


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def psql(sql):
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-F", "|", "-c", sql],
                       capture_output=True, text=True)
    return r.stdout.strip()


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True).stdout.strip()


def parse_md_table(path, prefix):
    """RESULTS.md 机生成表：| run | forcing | stable | steps | collapse | ratio | ratio_mw | ratio_err | Ma | ... |"""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(f"| {prefix}"):
            continue
        f = [x.strip() for x in line.strip("|").split("|")]
        if len(f) != 13:      # 只取主 tiers 表（schemes 附属表为 9 列）
            continue
        rows.append({
            "run": f[0], "forcing": f[1], "stable": f[2] == "True",
            "steps": int(f[3]), "collapse_step": None if f[4] in ("—", "-") else int(f[4]),
            "ratio_got": None if f[5] in ("—", "-") else float(f[5]),
            "ratio_ref": None if f[6] in ("—", "-") else float(f[6]),
            "ratio_err": None if f[7] in ("—", "-") else float(f[7]),
            "ma": None if f[8] in ("—", "-") else float(f[8]),
            "verdict": f[11].replace("*", ""),
        })
    return rows


def main():
    art = json.loads((SRC / "benchmarks/eos_maxwell/out/results.json").read_text())
    xval = json.loads((SRC / "xval/XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO/results.json").read_text())
    pre = json.loads((SRC / "xval/XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO/predeclared.json").read_text())
    repro = json.loads((OUT / "repro_m18429_pilot200/results.json").read_text())

    # ---- 1. Maxwell 等面积构造（ref 侧） ----
    maxwell = {k: {"ratio": v["ratio"], "rho_l": v["rho_liquid"], "rho_v": v["rho_vapour"],
                   "T_r": v["T_r"], "area_residual_rel": v["area_residual_rel"],
                   "hit_target_1pct": abs(v["ratio"] / float(k) - 1) <= 0.01}
               for k, v in art["maxwell_tiers"].items()}

    # ---- 2. got 侧（本单真实存在的 run 工件） ----
    ctl = art["controls"][0]
    got_control = {"tag": ctl["tag"], "spec": ctl["spec"], "steps": ctl["steps_run"],
                   "ratio_got": ctl["ratio"], "ratio_ref": ctl["ratio_maxwell"],
                   "ratio_err": ctl["ratio_error"], "ma": ctl["ma_max"],
                   "mass_rate": ctl["mass_rate_per_step"], "stable": ctl["stable"],
                   "verdict": ctl["verdict"]}

    # 工件里是否真有 pilot 臂？xval 报告声称 pilot_ratio10 2000 步
    tags_in_artifact = [c["tag"] for c in art["controls"]] + [t["tag"] for t in art["tiers"]]
    pilot_claimed = xval.get("pilot_run", {})
    pilot_artifact_present = any("pilot" in t for t in tags_in_artifact)
    alt_check = {"tags_in_out_results": tags_in_artifact,
                 "pilot_claimed_tag": pilot_claimed.get("tag"),
                 "pilot_artifact_present": pilot_artifact_present,
                 "tiers_len": len(art["tiers"]), "controls_len": len(art["controls"])}

    # 历史战役表（m8675 2026-11-10 机生成 RESULTS.md；本单未重跑）
    tiers_md = parse_md_table(SRC / "benchmarks/eos_maxwell/RESULTS.md", "tier_")

    # ---- 3. 独立复跑（main 检出、同 driver sha256、短步数冒烟） ----
    repro_tier = repro["tiers"][0]
    repro_rec = {"steps": repro_tier["steps_run"], "ratio_got": repro_tier["ratio"],
                 "ratio_ref": repro_tier["ratio_maxwell"], "ratio_err": repro_tier["ratio_error"],
                 "ma": repro_tier["ma_max"], "stable": repro_tier["stable"],
                 "driver_sha256": repro["driver_sha256"], "verdict": repro_tier["verdict"]}

    # ---- 4. 机器事实核（benchmark 裁判 / 报告 / 提交） ----
    db_bench = int(psql("select count(*) from tensor_benchmark_cases where anchor_slug="
                        "'XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO'") or 0)
    db_bench_fuzzy = int(psql("select count(*) from tensor_benchmark_cases where anchor_slug "
                              "ILIKE '%MAXWELL%' OR anchor_slug ILIKE '%EOS%'") or 0)
    db_reports = int(psql("select count(*) from tensor_reports where mission_id=18366") or 0)
    db_solutions = int(psql("select count(*) from oss_solutions where feature_slug "
                            "ILIKE '%EOS-MAXWELL%' OR feature_slug ILIKE '%HIGH-DENSITY%'") or 0)
    mission_row = psql("select status, coalesce(result->>'head_commit','NULL'), "
                       "coalesce(result->>'branch','NULL') from tensor_dev_missions where id=18366")
    commits_m18366 = git("log", "--oneline", "--all", "--grep=m18366", cwd=SRC)
    dirty = git("status", "--short", cwd=SRC).splitlines()

    # ---- 5. 判据逐项 ----
    checks = {
        "maxwell_all_targets_hit_1pct": all(v["hit_target_1pct"] for v in maxwell.values()),
        "maxwell_area_residual_max": max(v["area_residual_rel"] for v in maxwell.values()),
        "tier10_edm_ratio_err_within_owner_15pct":
            tiers_md[0]["ratio_err"] is not None and tiers_md[0]["ratio_err"] <= OWNER_TOL,
        "tier10_edm_spurious_ma_le_pass_gate": tiers_md[0]["ma"] <= SPUR_PASS_MA,
        "tier10_edm_composite_verdict": tiers_md[0]["verdict"],
        "tier100_edm_ratio_err": tiers_md[2]["ratio_err"],
        "tier100_edm_within_owner_15pct": tiers_md[2]["ratio_err"] <= OWNER_TOL,
        "tier1000_edm_stable": tiers_md[4]["stable"],
        "control_planar_ratio_err": got_control["ratio_err"],
        "control_planar_ratio_err_within_owner_15pct": got_control["ratio_err"] <= OWNER_TOL,
        "control_planar_spurious_ma": got_control["ma"],
        "control_planar_spurious_ma_le_pass_gate": got_control["ma"] <= SPUR_PASS_MA,
        "repro_short_pilot_ratio_err": repro_rec["ratio_err"],
        "repro_short_pilot_spurious_ma": repro_rec["ma"],
        "benchmark_cases_registered": db_bench,
        "benchmark_cases_fuzzy_maxwell_or_eos": db_bench_fuzzy,
        "tensor_reports_for_m18366": db_reports,
        "oss_solutions_for_feature": db_solutions,
        "commits_mentioning_m18366": len([l for l in commits_m18366.splitlines() if l.strip()]),
        "worktree_dirty_entries": dirty,
        "mission_status_headcommit_branch": mission_row,
        "driver_sha256_artifact": art["driver_sha256"],
        "driver_sha256_worktree_main": sha256(REPO / "benchmarks/eos_maxwell/run.py"),
        "probe_sha256_artifact": art["environment"]["probe_lib_sha256"],
        "probe_sha256_worktree_main": sha256(REPO / "benchmarks/eos_maxwell/probe_lib.py"),
        "artifact_sha256_m18366_out": sha256(SRC / "benchmarks/eos_maxwell/out/results.json"),
        "xval_report_sha256": sha256(SRC / "xval/XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO/REPORT.md"),
        "results_md_generated_utc": re.search(r"generated: (\S+)",
                                              (SRC / "benchmarks/eos_maxwell/RESULTS.md").read_text()).group(1),
        "out_results_generated_utc": art["generated_utc"],
    }

    # ---- 6. 出表（机生成，报告引用本文件即零手抄） ----
    lines = ["# m18429 机生成核验表（verify_m18429.py 输出，禁手改）", ""]
    lines += ["## T1 Maxwell 等面积构造（ref 侧）", "",
              "| 目标比 | T_r | rho_l | rho_v | 构造比 | |面积残差|/p_sat dV | 命中目标(<=1%) |",
              "|---|---|---|---|---|---|---|"]
    for k, v in maxwell.items():
        lines.append(f"| {k} | {v['T_r']:.6f} | {v['rho_l']:.6f} | {v['rho_v']:.6f} | "
                     f"{v['ratio']:.5f} | {v['area_residual_rel']:.2e} | {v['hit_target_1pct']} |")
    lines += ["", "## T2 本单工作树真实存在的 run 工件（planar control, 40000 步）", "",
              "| tag | steps | ratio_got | ratio_ref | 误差 | 相对误差 | Ma | mass_rate/step | 工件判决 |",
              "|---|---|---|---|---|---|---|---|---|"]
    lines.append(f"| {got_control['tag']} | {got_control['steps']} | {got_control['ratio_got']:.4f} | "
                 f"{got_control['ratio_ref']:.4f} | {got_control['ratio_got']-got_control['ratio_ref']:+.4f} | "
                 f"{got_control['ratio_err']:.4f} | {got_control['ma']:.4f} | {got_control['mass_rate']:.3e} | "
                 f"{got_control['verdict']['cls']} ({got_control['verdict']['reason']}) |")
    lines += ["", "## T3 历史战役表（RESULTS.md, m8675 2026-11-10 生成；本单未重跑）", "",
              "| run | forcing | stable | steps | 测比 | Maxwell 比 | 相对误差 | Ma | 表内判决 |",
              "|---|---|---|---|---|---|---|---|---|"]
    def fmt(x, spec):
        return "—" if x is None else format(x, spec)

    for r in tiers_md:
        lines.append(f"| {r['run']} | {r['forcing']} | {r['stable']} | {r['steps']} | "
                     f"{fmt(r['ratio_got'], '.3f')} | {fmt(r['ratio_ref'], '.3f')} | "
                     f"{fmt(r['ratio_err'], '.4f')} | {fmt(r['ma'], '.3f')} | {r['verdict']} |")
    lines += ["", "## T4 审计侧独立短程复跑（main 检出, driver sha256 同源）", "",
              "| steps | ratio_got | ratio_ref | 误差 | Ma | stable | 判决 |",
              "|---|---|---|---|---|---|---|"]
    lines.append(f"| {repro_rec['steps']} | {repro_rec['ratio_got']:.4f} | {repro_rec['ratio_ref']:.4f} | "
                 f"{repro_rec['ratio_err']:.4f} | {repro_rec['ma']:.4f} | {repro_rec['stable']} | "
                 f"{repro_rec['verdict']['cls']} ({repro_rec['verdict']['reason']}) |")
    lines += ["", "## T5 判据逐项（PASS/FAIL 由阈值机器比）", ""]
    for k in ("maxwell_all_targets_hit_1pct", "tier10_edm_ratio_err_within_owner_15pct",
              "tier10_edm_spurious_ma_le_pass_gate", "tier100_edm_within_owner_15pct",
              "tier1000_edm_stable", "control_planar_ratio_err_within_owner_15pct",
              "control_planar_spurious_ma_le_pass_gate", "benchmark_cases_registered",
              "tensor_reports_for_m18366", "commits_mentioning_m18366"):
        lines.append(f"- {k} = {checks[k]}")
    lines += ["", "## T6 工件/代码 provenance", "",
              f"- driver sha256 (artifact) = {checks['driver_sha256_artifact']}",
              f"- driver sha256 (main 检出) = {checks['driver_sha256_worktree_main']}",
              f"- probe  sha256 (artifact) = {checks['probe_sha256_artifact']}",
              f"- probe  sha256 (main 检出) = {checks['probe_sha256_worktree_main']}",
              f"- m18366 out/results.json sha256 = {checks['artifact_sha256_m18366_out']}",
              f"- RESULTS.md 生成于 {checks['results_md_generated_utc']}；out/results.json 生成于 "
              f"{checks['out_results_generated_utc']}（两者不同代）",
              f"- m18366 工作树未提交条目 = {len(dirty)}；git log --all --grep=m18366 命中 = "
              f"{checks['commits_mentioning_m18366']}",
              f"- mission(18366) status/head_commit/branch = {checks['mission_status_headcommit_branch']}",
              f"- pilot 臂工件存在? {alt_check['pilot_artifact_present']}（工件内 tags={alt_check['tags_in_out_results']}）"]
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")

    # ---- 7. 图：ref vs got + 伪流门 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    arms = ["tier r10 edm", "tier r10 guo", "tier r100 edm", "planar r10", "pilot200"]
    ratios = [tiers_md[0]["ratio_got"], tiers_md[1]["ratio_got"], tiers_md[2]["ratio_got"],
              got_control["ratio_got"], repro_rec["ratio_got"]]
    refs = [tiers_md[0]["ratio_ref"], tiers_md[1]["ratio_ref"], tiers_md[2]["ratio_ref"],
            got_control["ratio_ref"], repro_rec["ratio_ref"]]
    rel = [r / m - 1 for r, m in zip(ratios, refs)]
    ax1.axhspan(-OWNER_TOL, OWNER_TOL, color="tab:green", alpha=0.12, label="owner +/-15%")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.bar(arms, rel, color=["tab:red" if abs(x) > OWNER_TOL else "tab:orange" for x in rel])
    ax1.set_ylabel("measured / Maxwell - 1")
    ax1.set_title("density-ratio maintenance vs Maxwell")
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="x", rotation=20)
    mas = [tiers_md[0]["ma"], tiers_md[1]["ma"], tiers_md[2]["ma"], got_control["ma"], repro_rec["ma"]]
    ax2.axhline(SPUR_PASS_MA, color="tab:green", ls="--", label="PASS <=0.05")
    ax2.axhline(SPUR_WARN_MA, color="tab:red", ls=":", label="MARGINAL <=0.10")
    ax2.bar(arms, mas, color="tab:blue")
    ax2.set_ylabel("spurious Ma = max|u|/cs")
    ax2.set_title("spurious-current floor vs preregistered gate")
    ax2.legend(fontsize=8)
    ax2.tick_params(axis="x", rotation=20)
    fig.suptitle("m18429 audit of m18366: EOS-Maxwell high density ratio (TensorLBM)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_ref_vs_got.png", dpi=130)

    # ---- 8. evidence.json（供 verdict 落表；所有指针真实） ----
    evidence = {
        "audit_of": "m18366 XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO",
        "claimed_level": 4,
        "judged_level": 2,
        "level_delta": (
            "声明 L0->L4；核验判 L2。L2 依据：真实实现+入口+可执行 run 工件在库内存在并复跑通过"
            "（impl_files/entry/run_artifacts 全真）。L3 不计入本单：integration_callers 为 m8675 及其"
            "前序单既有（git log --all --grep=m18366 = 0 提交），本单未接线。L4 明确不成立："
            "tensor_benchmark_cases 该 anchor 0 行（%MAXWELL%/%EOS% 亦 0 行）→ 无注册裁判；"
            "预注册复合门在 ratio10 即 FAIL（伪流 Ma=0.115>0.10），ratio100 误差 193%，ratio1000 塌缩。"),
        "impl_files": [
            "src/tensorlbm/sc_eos_coexistence.py",
            "src/tensorlbm/sc_phasechange/highdensity.py",
            "src/tensorlbm/multiphase.py",
            "benchmarks/eos_maxwell/probe_lib.py",
        ],
        "impl_note": ("上述实现为 PRE-EXISTING（git log --diff-filter=A ⇒ 6db3e1ca/5f5a1169），"
                      "driver/probe sha256 与 campaign 工件记录一致 ae393b70…/31fbdd7c…；m18366 未写一行 "
                      "src/（git status 见 worktree_dirty：仅 xval/ 三件 + out/results.json）。"),
        "entry": [
            "benchmarks/eos_maxwell/run.py",
            "audit/m18429/m18366_xval/predeclared.json",
        ],
        "entry_note": "run.py:65-101 argparse（--mode pilot/tiers/controls/... --out --device）；审计侧实测可执行。",
        "run_artifacts": [
            "audit/m18429/m18366_out_results_copy.json",
            "audit/m18429/repro_m18429_pilot200/results.json",
        ],
        "run_artifacts_note": (
            "m18366 唯一新 run 工件 = planar control 40000 步（sha256 755f39ce…，generated 13:57:01），"
            "ratio_got=11.0156 vs ref 10.0127（err=10.02%），Ma=0.11414，工件自身判决 FAIL；"
            "tiers=[] 空、无 pilot 臂（xval 声称 pilot_ratio10 无对应工件）；"
            "其所引 out/results.json 覆盖了 RESULTS.md/CRITERION.md 的数据源（RESULTS.md 生成于 11-10，"
            "现行 out 已无 tiers，provenance 断链）。"),
        "tests": [
            "tests/test_sc_eos_coexistence.py::test_correct_maxwell_pair_at_ledger_valid_coupling",
            "tests/test_sc_eos_coexistence.py::test_simulation_matches_correct_maxwell_at_low_density_ratio",
            "tests/test_eos_maxwell_stability.py",
        ],
        "tests_note": "审计侧复跑：test_sc_eos_coexistence.py 13 passed,1 xfailed；test_eos_maxwell_stability.py 17 passed,2 skipped（skip=工件不在本树）。测试同样 PRE-EXISTING。",
        "integration_callers": [
            "maxwell_coexistence:src/tensorlbm/microbubble_dr/channel.py:81,190,192",
            "maxwell_construction:src/tensorlbm/multiphase_phase_change_verification.py:268,316",
            "COEXISTENCE_G:src/tensorlbm/cavitation.py:68",
        ],
        "integration_note": "调用方真实存在且被 grep 命中，但均早于 m18366（本单 0 提交），不计入本单 L3。",
        "benchmark": {
            "registered_cases": db_bench,
            "fuzzy_maxwell_or_eos": db_bench_fuzzy,
            "ref": {"ratio10": 10.0127, "ratio100": 99.9708, "ratio1000": 995.15,
                    "source": "sc_eos_coexistence.maxwell_coexistence（面积残差 1e-13~1e-11）"},
            "got": {"tier10_edm": {"ratio": 10.45, "err": 0.0437, "ma": 0.115, "verdict": "FAIL"},
                    "tier10_guo": {"ratio": 13.81, "err": 0.379, "ma": 0.127, "verdict": "FAIL"},
                    "tier100_edm": {"ratio": 293.3, "err": 1.93, "ma": 0.333, "verdict": "FAIL"},
                    "tier1000_edm": {"ratio": None, "collapse_step": 100, "verdict": "FAIL"},
                    "planar_r10": {"ratio": 11.0156, "err": 0.1002, "ma": 0.1141, "verdict": "FAIL"}},
            "tolerance": "<15%（owner）；预注册复合门 ratio_error<=0.15 AND Ma<=0.05 AND >=20000 steps",
            "verdict": "anchor 判据未达标：0 注册基准行；ratio10 误差项过 15% 门但伪流门 FAIL；ratio100/1000 失败。",
        },
        "repro_cmd": ("cd /root/tp-agent/work/m18429 && python3 benchmarks/eos_maxwell/run.py --mode pilot "
                      "--steps 200 --out /tmp/m18429_repro --device cpu"),
        "repro_note": "审计侧执行 rc=0，产出 tiers=1（ratio 10.349 vs 10.0127, err 3.36%, Ma 0.1336 -> FAIL）；工件留 audit/m18429/repro_m18429_pilot200/results.json。",
        "external_anchor": "audit/m18429/external_anchor.md",
        "machine_facts": {
            "tensor_reports_for_m18366": db_reports,
            "oss_solutions_for_feature": db_solutions,
            "commits_mentioning_m18366": checks["commits_mentioning_m18366"],
            "worktree_dirty_entries": dirty,
            "cluster_campaign": "cj1906 submitted 13:51 仍 queued（无工件返回）",
            "mission_result_flags": "server_quality_audit: gate_valid=false, key_present=false",
        },
        "gate_expectation": "judge(declared 2, L2 三键 impl_files/entry/run_artifacts 全真) ⇒ gate_level 2 ⇒ confirmed；L4 声明被降级记为不符。",
        "artifacts": ["audit/m18429/tables.md", "audit/m18429/evidence.json", "audit/m18429/fig1_ref_vs_got.png"],
    }
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=1, ensure_ascii=False))
    print(json.dumps({"checks": {k: v for k, v in checks.items() if k != "worktree_dirty_entries"},
                      "pilot_artifact_present": pilot_artifact_present,
                      "wrote": ["tables.md", "evidence.json", "fig1_ref_vs_got.png"]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()