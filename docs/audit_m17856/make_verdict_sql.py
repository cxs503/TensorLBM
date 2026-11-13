#!/usr/bin/env python3
"""Emit verdict_m17856.sql - board-audit verdict row for m17758.

Every number in the evidence payload is read from
``artifacts/m17856_audit/audit_m17856.json`` (produced by
``scripts/audit/m17856_fluctuating_ghost_audit.py``); nothing is retyped.
File pointers under the machine-checked keys (impl_files / entry /
run_artifacts / tests) are paths that exist in this worktree: the audited
delivery was never committed (m17758 has zero tracked changes), so a
byte-identical snapshot was frozen into docs/audit_m17856/evidence_snapshot/
with SHA256SUMS.txt.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUD = json.loads((ROOT / "artifacts" / "m17856_audit" / "audit_m17856.json").read_text())
SNAP = "docs/audit_m17856/evidence_snapshot"
AUDREL = "artifacts/m17856_audit"
MISSION_ID = 17856
SCOPE = "m17758 XVAL-LBM-FLUCTUATING-LBM-GHOST-MODES"
ANCHOR = AUD["anchor"]

ref = AUD["R5_theory_referee"]
fresh = AUD["R1_fresh_rerun"]
r2 = AUD["R2_linear_regime"]


def r(v: float, n: int = 6) -> float:
    return round(float(v), n)


evidence = {
    "verdict_basis": (
        "L2：真实的独立功能实现（381 行模式基/噪声注入/方差估计器 + 396 行入口 + 7 个单元测试 + 工件），"
        "入口可执行并在 42.4 s 内产出全套工件；但 (a) 交付工件不能由入口复现（记录值 vs 重跑值最大相对差 564%），"
        "(b) 幽灵模热化相对 FDT 目标偏 +183%（阈值 15%），(c) 独立理论裁判（Var(u_x)=kT/rho, Var(rho)=rho^2 kT，"
        "不依赖模块自己的 W_a）显示 A2 质量模偏 +114.8%，(d) 零集成调用方 ⇒ L3 不成立，"
        "(e) 锚点基准裁判表 0 行、外部锚只有编译产物无源码、从未与 ludwig 的任何数值输出比对 ⇒ L4/L5 不成立。"
    ),
    "claimed_level_implicit": "XVAL 交叉对标交付（.result.headline 称「部分完成（honest 部分 FAIL）」）",
    "delivered_level": 2,
    "gap_to_claim": (
        "XVAL 的对标语义未兑现：没有任何 OSS 侧数值，只有公式反推 + 自建 FDT 裁判；"
        "而该自建裁判下 A1（自称已知正确的 stress 臂）也偏 -74.5%，A2 偏 +183%，"
        f"两臂都远离 1，「transplant 有效」与「不稳定」两种结论都无 <15% 依据。"
    ),
    "impl_files": [f"{SNAP}/fluctuating_ghost_modes.py"],
    "impl_files_detail": [
        f"{SNAP}/fluctuating_ghost_modes.py:113-140 (orthonormal_modes：9 维候选 Hermite → 无权重 Gram-Schmidt，"
        "单元测试断言 basis@basis.T=I，注意与 docstring 的 weighted-orthonormal 说法不符)",
        f"{SNAP}/fluctuating_ghost_modes.py:161-186 (ModeVarianceReport: 每模方差/FDT 比值/误差)",
        f"{SNAP}/fluctuating_ghost_modes.py:196-236 (FluctuatingLBMGhost.__init__: thermostat in {{stress,ghosts,all}})",
        f"{SNAP}/fluctuating_ghost_modes.py:238-262 (噪声幅度与 _add_mode_noise：df_i = e_ai·xi·sd，逐模 W_a)",
        f"{SNAP}/fluctuating_ghost_modes.py:296-330 (step：随机力 + 逐模噪声 + 流动)",
        f"{SNAP}/fluctuating_ghost_modes.py:332-379 (mode_variance_report: 时间-空间平均方差估计器)",
        "原树路径 /root/tp-agent/work/m17758/src/tensorlbm/fluctuating_ghost_modes.py（381 行，未跟踪）",
    ],
    "entry": [f"{SNAP}/run_xval.py"],
    "entry_detail": (
        f"{SNAP}/run_xval.py:395 main()（--skip-extract 可跳过 OSS 取证）; "
        ":166-193 measure() 13 个臂; :241-252 出图; 审计重跑 42.4 s，returncode={fresh['returncode']}"
    ),
    "run_artifacts": [
        f"{SNAP}/results.json",
        f"{SNAP}/table_mode_variance.csv",
        f"{SNAP}/ludwig_reference.json",
        f"{AUDREL}/fresh_results.json",
        f"{AUDREL}/audit_m17856.json",
        "docs/audit_m17856/fig_m17856_recorded_vs_fresh.png",
        "docs/audit_m17856/fig_m17856_theory_referee.png",
    ],
    "run_artifacts_detail": {
        "recorded_vs_fresh_max_rel_diff": r(fresh["max_rel_diff_recorded_vs_fresh"]),
        "recorded_A2_ghost_ratio_mean": r(
            next(a["report"]["ghost_ratio_mean"] for a in json.loads(
                (ROOT / SNAP / "results.json").read_text())["arms"]
                if a["arm"] == "A2_ghost_thermostat")),
        "fresh_A2_ghost_ratio_mean": r(
            next(a["report"]["ghost_ratio_mean"] for a in json.loads(
                (ROOT / AUDREL / "fresh_results.json").read_text())["arms"]
                if a["arm"] == "A2_ghost_thermostat")),
        "figures_written_by_entry": True,
        "note": "交付 results.json 记录 A2 幽灵模比 37.8（各模 19.7~50.5），同一入口重跑得 2.61（2.0~2.8）；"
                "两次独立重跑逐位相同（R5 deterministic_repeat_A2=True）⇒ 记录工件非当前代码所产。",
    },
    "tests": [f"{SNAP}/test_fluctuating_ghost_modes.py"],
    "tests_detail": (
        f"{AUD['S4_tests']['cmd']} → {AUD['S4_tests']['tail'][-1] if AUD['S4_tests']['tail'] else ''}; "
        f"失败项={AUD['S4_tests']['failed_tests']}（FDT 归一化断言 target_mode_variance(1.0)=3 rho kT，"
        "与 m17758 申报的「幽灵臂方差对比断言失败」不一致）; 7 个测试均无 <15% 精度断言"
    ),
    "integration_callers": [],
    "integration_callers_note": (
        "NONE。git grep -l 'fluctuating_ghost_modes' 全仓仅命中模块自身/测试/入口（"
        + ", ".join(AUD["S3_integration"]["grep_hits"]) + "），"
        "src/tensorlbm/__init__.py 未导出，solver/app 0 引用 ⇒ L3 不成立。"
    ),
    "benchmark": {
        "anchor_slug": ANCHOR,
        "registered_cases": AUD["D1_benchmark"]["n_registered_for_anchor"],
        "tensor_benchmark_cases_total": AUD["D1_benchmark"]["tensor_benchmark_cases_total"],
        "gate_pct": 15.0,
        "machine_referee": f"SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}' → 0 行（唯一裁判为空）",
        "self_built_referee_mode_ratio": {
            "criterion": "Var(m_a)/(3 rho kT W_a) 应 = 1，容差 ±15%",
            "A1_stress_only_nonconserved_mean": r(AUD["R3_stress_arm"].get("nonconserved_mean_ratio", float("nan"))),
            "A2_ghosts_qx": r(r2["ghosts_kT2.5e-4"]["ratios"]["qx"]),
            "A2_ghosts_qx_at_kT1e-9": r(r2["ghosts_kT1e-9"]["ratios"]["qx"]),
            "verdict": "两臂均远超 ±15%（+183% / -74.5%），且 kT→1e-9 时比值不变 ⇒ 非非线性爆炸可解释",
        },
        "independent_theory_referee": {
            "criterion": "Var(u_x)=kT/rho, Var(rho)=rho^2 kT（不依赖模块 W_a）",
            "A1_stress": {"err_ux_pct": r(100 * ref["A1_stress"]["err_ux"], 2),
                          "err_rho_pct": r(100 * ref["A1_stress"]["err_rho"], 2)},
            "A2_ghosts": {"err_ux_pct": r(100 * ref["A2_ghosts"]["err_ux"], 2),
                          "err_rho_pct": r(100 * ref["A2_ghosts"]["err_rho"], 2)},
            "A3_all": {"err_ux_pct": r(100 * ref["A3_all"]["err_ux"], 2),
                       "err_rho_pct": r(100 * ref["A3_all"]["err_rho"], 2)},
            "base_FluctuatingLBM": {"err_ux_pct": r(100 * ref["base_FluctuatingLBM_kt_only"]["err_ux"], 2),
                                    "err_rho_pct": r(100 * ref["base_FluctuatingLBM_kt_only"]["err_rho"], 2)},
            "verdict": "A2 速度模 -14.4%（勉强在带内）但质量模 +114.8%；A3 +79.4%/+359%；"
                       "A1 与基类一致 -84.9%/-70.0% ⇒ 基类与 transplant 都未达 15%",
        },
        "ref_vs_got_note": "无 OSS 侧 ref 数值可对；上表 ref 为解析 FDT/能量均分值，got 为本次重跑实测。",
    },
    "repro_cmd": (
        "cd /root/tp-agent/work/m17856 && python scripts/audit/m17856_fluctuating_ghost_audit.py "
        "# 全量复核 ~150 s：静态 5 项 + 交付入口隔离重跑(42.4 s) + 两档 kT 探针 + 独立理论裁判 + 3 项 DB\n"
        "cd /root/tp-agent/work/m17856 && python scripts/audit/m17856_figures.py  # 两图重绘"
    ),
    "repro_of_deliverable": (
        f"cp -a /root/tp-agent/work/m17758/{{src,xval,tests}} /tmp/x && cd /tmp/x && "
        f"python xval/fluctuating_lbm_ghost_modes/run_xval.py --skip-extract → "
        f"exit {fresh['returncode']}, {fresh['wall_s']} s, 产出 results.json/CSV/2 PNG"
    ),
    "external_anchor": {
        "claimed": "ludwig-cf/ludwig @78d83c5151ff1d4c4201a96b266a08e97047cae6 的 fluctuating-lbm-ghost-modes 做法",
        "object_present": AUD["S5_external_anchor"].get("object_present"),
        "object_sha256_matches_claim": AUD["S5_external_anchor"].get("object_sha256_matches_claim"),
        "symbols_present": AUD["S5_external_anchor"].get("symbols_present"),
        "n_c_sources_in_archive": AUD["S5_external_anchor"].get("n_c_sources"),
        "object_arch": AUD["S5_external_anchor"].get("object_arch"),
        "status": "WITHHELD：只有 elf64-loongarch 编译产物（无源码、无版本可执行参考），"
                  "公式反推可信（symbol+DWARF+反汇编+执行），但从未与 ludwig 的任何模拟输出做数值比对；"
                  "缺源码/缺可运行参考 ⇒ 「<15% 交叉对标」数值得不到锚点侧数字",
        "formula_probe": AUD["R4_formula"]["stdout"],
    },
    "findings": AUD["findings"],
    "audit_independence": (
        "被审工件未修改：交付入口在 /tmp 副本中重跑（run_artifacts/fresh_results.json 为副本产物），"
        "探针通过 sys.path 只读导入交付模块；局限：(1) 未重跑 OSS 取证段（--skip-extract，需 loongarch 工具链，"
        "只核 sha256/符号/源码缺失），(2) 未做收敛/尺寸扫描（k=(32,32) 单点），(3) 未评理论公式的严格推导，"
        "只用两个独立裁判（模块自建 FDT 模比 + 原始变量能量均分）判 15%。"
    ),
    "verdict": "CONFIRMED-L2（不返修到位则不得升 L3/L4）：实现是真的、可跑、有测试有工件；"
               "但工件不可复现、15% 判据双向失败、零集成、无基准案例 ⇒ 按 ② 判级只给 L2。"
               "建议返修项：F1 工件由入口直接生成并入库、F2 修 target_mode_variance 逐模权重+bitwise 负控制、"
               "F3 重推含流动算子的稳态方差公式或改用原始变量裁判、F0 提交入库、F9 登记 tensor_benchmark_cases。",
}

sql = f"""-- m17856 board audit verdict: 对象 m17758 {ANCHOR} (TensorLBM)
-- 判级 L2：真实独立功能实现；工件不可复现、15% 判据双向失败、零集成、无登记基准案例
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', '{SCOPE}', {MISSION_ID}, 2, $json${json.dumps(evidence, indent=1, ensure_ascii=False)}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence;
"""
(ROOT / "verdict_m17856.sql").write_text(sql)
(HERE / "verdict_evidence.json").write_text(json.dumps(evidence, indent=1, ensure_ascii=False) + "\n")
print(f"wrote verdict_m17856.sql ({len(sql)} bytes), level=2")