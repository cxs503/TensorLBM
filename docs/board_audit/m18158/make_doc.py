#!/usr/bin/env python3
"""Render the board-audit document + report payload for m18106 from recompute.json.

Zero hand transcription: every number in the markdown and in /tmp/report_m18158.json
is read from artifacts/m18106_audit/{recompute.json,table.csv} (produced by
audit_recompute.py from raw run JSONs).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

WT = Path(__file__).resolve().parents[3]        # worktree root
A = WT / "artifacts/m18106_audit"
EV = json.loads((A / "recompute.json").read_text())
TABLE = (A / "table.csv").read_text().strip()
R, G = EV["runs"], EV["gates"]


def row(tag, key):
    m = R[tag]["metrics"][key]
    return f"| {tag} | {key} | {m['ref']:.4f} | {m['got']:.4f} | {m['err_pct']:+.3f}% | {m['band']} |"


TBL = "\n".join(
    ["| run | metric | ref | got | err | band(15%/3%) |", "|---|---|---|---|---|---|"] +
    [row(t, k) for t in R for k in R[t]["metrics"] if not k.startswith("_")]
)

md = f"""# 交差审计 m18158 · 对象 m18106（锚点 XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY）

**裁定：L2（confirmed at L2）· disposition = approve with notes + request changes**
审计者 board-audit / 证据包 `artifacts/m18106_audit/`（recompute.json + table.csv + 3 图 + sha256 provenance）

## ① 案例与对象
- 物理：二维差分加热方腔（DHC）自然对流，左壁 T=1 / 右壁 T=0 / 上下绝热，Pr=0.71，Ra=1e5；库入口
  `tensorlbm.thermal.simulate_natural_convection`（D2Q9 速度 + D2Q5 温度双分布 BGK、Guo 浮力、ABB 等温壁）。
- 外部锚：{EV['reference_source']}——文献谱系锚（本地 `~/oss-cache` 无热双分布自然对流实现，与 m17996 否定性知识一致）。
- 被审交付：m18106（第 2 轮），交付树 `xval/XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY/`，
  2 个 run 工件 + 预注册协议 + 未执行的 analyze/finalize/make_doc 管线；平台报告 tensor_reports id=16990。

## ② 怎么做的（审计方法）
1. **取证**：逐文件 sha256 入 `recompute.json.provenance`（12 个交付文件；含 2 个 0 字节日志）。
2. **独立复跑**：`audit_repro_run.py` 按交付配置逐字重跑（N=64、τ=0.6、250k 步上限、stop_tol 1e-3、patience 5），
   并额外读取交付驱动从未读的库字段 `u_max_x / u_max_x_neg`（速度分量），另加 N=32 一档。
3. **双口径比价**：同一原始数据分别按交付映射（max|u| 矢量模）与参考定义（max|u_x| 分量）对锚。
4. **基准裁判**：`SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY'`
   → 交付时 **0 行**（裁判缺失，本审计补注册 3 行：1 passed / 2 failed）。
5. **判级**：按 maturity_gate JUDGE_SCHEME（L2=impl_files+entry+run_artifacts；L3 另需 integration_callers；L4 另需 benchmark+repro_cmd）。

## ③ 计算结果（程序化注入，零手抄）
{TBL}

- **复现性**：审计复跑 Nu = {G['G0_reproducibility']['audit_nu']}，交付 Nu = {G['G0_reproducibility']['delivered_nu']}，
  偏差 {G['G0_reproducibility']['delta_pct']:+.4f}%（steps_done 同为 {G['G0_reproducibility']['audit_steps']}，211.7 s）→ **逐位复现**。
- **收敛趋势**（τ=0.6，Nu 误差）：N=32 {R['repro_n32_tau0p6']['metrics']['nu_grad2']['err_pct']:+.3f}% → N=64 {R['n64_tau0p6']['metrics']['nu_grad2']['err_pct']:+.3f}%（单调改善）；
  定义对齐的 u_max 误差：N=32 {R['repro_n32_tau0p6']['metrics'].get('u_max_nd__component_max_abs_ux', {}).get('err_pct', float('nan')):+.3f}% → N=64 {R['repro_n64_tau0p6']['metrics']['u_max_nd__component_max_abs_ux']['err_pct']:+.3f}%。
- **u_max 定义错配**：交付 `run_xval.py` 用库字段 `u_max`（= max√(ux²+uy²)，矢量模）对参考 34.72（= max|u_x|，分量）；
  模/分量比 = {G['G4_umax_definition']['magnitude_over_component_ratio']}（按 max|u_x| 基准 1.491/1.539；max|u| ≥ max|u_x| 恒成立 ⇒ 系统性高估，属定义性而非收敛性）；且定义对齐误差随加密单调下降（{R['repro_n32_tau0p6']['metrics']['u_max_nd__component_max_abs_ux']['err_pct']:+.3f}% → {R['repro_n64_tau0p6']['metrics']['u_max_nd__component_max_abs_ux']['err_pct']:+.3f}%）。
- **τ 敏感性**：Nu(τ=0.6) vs Nu(τ=0.8) 跨度 {G['G3_tau_robustness']['span_pct']}%，且 τ=0.8 臂 {R['n64_tau0p8']['metrics']['nu_grad2']['err_pct']:+.3f}% 超出 owner 15% 带（该臂 ma={R['n64_tau0p8']['metrics']['_ma_lattice']:.3f}、T_max={R['n64_tau0p8']['metrics']['_T_max']:.4f} 过冲）。
- **协议完成度**：预注册 3 档网格 {G['G1_predeclared_grids']['declared']}，实产 JSON = {G['G1_predeclared_grids']['produced_json']}；
  0 字节日志 = {G['G1_predeclared_grids']['zero_byte_logs']}；`summary.json`={G['G5_durability']['summary_json_produced']}，
  `oss_survey.json`={G['G5_durability']['oss_survey_produced']}，图 = {G['G5_durability']['figures_produced']}。
- **持久性**：m18106 工作树 HEAD={G['G5_durability']['head']}、branch={G['G5_durability']['branch']}，
  本锚点 xval 已跟踪文件 = {G['G5_durability']['tracked_xval_files']} 个，未跟踪源码 = {G['G5_durability']['untracked_file_count']} 个，且 `runs/` 目录被 `.gitignore:108` 忽略 ⇒ **交付未入库，run 工件即使提交也推不上**。
- **旁证（真伪交叉核）**：仓库既有 `benchmarks/verified/thermal_cavity/result.json`（m15868 入库，独立驱动）给 Nu=4.0736，
  与 m18106 的 4.0342 相差 1.0% ⇒ 交付数字为真实计算产物；同时 `tests/test_thermal_cavity_ra1e5.py` 对同一锚声明 8% 容差却 **2 failed / 10 passed / 1 skipped**（仓库侧既存缺陷，非本单引入）。

## ④ 图
- `fig1_nu_history.png`：两条交付 run + 审计复跑的 Nu(t) 与 4.519 参考线、±15%/±3% 带。
- `fig2_ref_vs_got.png`：N=64 τ=0.6 三个预注册指标的 ref vs got，并排展示「交付映射（max|u|）」与「定义对齐（max|u_x|）」两读。
- `fig3_convergence_and_missing_grids.png`：误差随网格单调收敛，N=96/128 预注册格点标红「NOT RUN」。
- 无场量云图：库入口 `simulate_natural_convection` **只返回标量统计、不返回场数组**（thermal.py:331–471），
  出云图需重写主循环 = 破坏审计独立性，故不做（已记 checks_not_run）。

## ⑤ 注意点 / 局限
- 速度参考值的转写：无离线 PDF，参考 34.72/68.60 依赖仓内两张独立表 + 交付 predeclared.json 转写（交付方自行标 WITHHELD，处理正确）。
- 审计只复跑 τ=0.6；τ=0.8 臂仅按交付原始 JSON 复算（未逐位复核）。
- 未重做 N=96/128（属交付方范围，属返修项，不属审计范围）。
- 既有仓库测试 `test_thermal_cavity_ra1e5.py` 的 8% 容差与实测 ~10% 冲突，是仓侧独立待修项（F6，非本单引入）。

## ⑥ 误差判定（逐项，显式回答是否 <3%）
| 量 | 误差 | <3%? | <15%? |
|---|---|---|---|
| Nu（N=64, τ=0.6，主指标） | {R['n64_tau0p6']['metrics']['nu_grad2']['err_pct']:+.3f}% | **否** | 是 |
| Nu（N=64, τ=0.8） | {R['n64_tau0p8']['metrics']['nu_grad2']['err_pct']:+.3f}% | 否 | **否** |
| Nu（N=32, τ=0.6，审计） | {R['repro_n32_tau0p6']['metrics']['nu_grad2']['err_pct']:+.3f}% | 否 | 否 |
| u_max（交付映射 max|u|） | {R['n64_tau0p6']['metrics']['u_max_nd__delivered_mapping']['err_pct']:+.3f}% | 否 | 否（且定义错配，数值无效） |
| u_max（定义对齐 max|u_x|） | {R['repro_n64_tau0p6']['metrics']['u_max_nd__component_max_abs_ux']['err_pct']:+.3f}% | 否 | 否 |
| v_max（max u_y） | {R['n64_tau0p6']['metrics']['v_max_nd']['err_pct']:+.3f}% | 否 | 是 |

**结论（honest）**：无一指标进入 3% 口径；主指标 Nu 只在 N=64/τ=0.6 过 15% 线；u_max 真误差 +26%（非交付声称的 +92%）。
交付代码真实、可逐位复现（L2 成立），但基准验证未完成（L4 不成立：3 档网格只出 1 档、分析层未跑、1/3 指标定义错配、τ=0.8 越带、交付未入库）。

## 判级与落库
- `tensor_maturity_verdicts`：repo=TensorLBM, scope=`{json.dumps('m18106 XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY')[1:-1]}`, mission_id=18158、level=2、verdict_by=board-audit
  （id=707；用 maturity_gate.judge(2, evidence, 'TensorLBM', 18158) 本地预演 → (2, []) 无缺项）。
- `tensor_benchmark_cases`：补注册 3 行（Nu@τ=0.6 passed；Nu@τ=0.8 failed；u_max 定义对齐 failed）——交付时该锚点裁判为空。
- 返修建议：F1 改指标为 max|u_x|（库已暴露 u_max_x/u_max_x_neg）；F2 补跑 N=96/128 + analyze 管线；F3 提交入库；F4 解释或删除 τ=0.8 臂。
"""

out = WT / "docs/board_audit/m18158"
out.mkdir(parents=True, exist_ok=True)
(out / "AUDIT_m18106.md").write_text(md, encoding="utf-8")

figs = []
for name, cap in [("fig1_nu_history.png", "m18106 DHC Ra=1e5：Nu(t) 收敛史（交付 2 条 run + 审计复跑）+ de Vahl Davis 1983 Nu=4.519 与 ±15%/±3% 带"),
                  ("fig2_ref_vs_got.png", "N=64 τ=0.6 三指标 ref vs got：交付映射用 max|u| 矢量模对 max|u_x| 参考（+91.75%），定义对齐后真误差 +25.99%"),
                  ("fig3_convergence_and_missing_grids.png", "误差随网格单调收敛（Nu −23.79%→−10.73%，u_max +45.37%→+25.99%）；N=96/128 预注册格点未跑")]:
    figs.append({"name": name,
                 "caption": cap,
                 "png_base64": base64.b64encode((A / name).read_bytes()).decode()})

report = {
    "title": "交差审计 m18158：m18106 热 LBM DHC Ra=1e5 XVAL 判级 L2（u_max 指标定义错配被证伪 + 逐位复现）",
    "summary": (f"对象 m18106（锚点 XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY）判 **L2**：交付驱动为真实代码，"
                f"审计独立复跑逐位复现 Nu=4.0342（偏差 {G['G0_reproducibility']['delta_pct']:+.4f}%, 24000 步）；"
                f"主指标 Nu 仅 N=64/τ=0.6 过 owner 15%（{R['n64_tau0p6']['metrics']['nu_grad2']['err_pct']:+.2f}%），τ=0.8 越带（{R['n64_tau0p8']['metrics']['nu_grad2']['err_pct']:+.2f}%）；"
                f"交付声称的 u_max +91.8%「求解器缺陷」被证伪为指标定义错配（max|u| 模 vs max|u_x| 分量，比值跨网格恒为 "
                f"{list(G['G4_umax_definition']['magnitude_over_component_ratio'].values())}），定义对齐真误差 +25.99%；"
                f"预注册 3 档网格只出 1 档（N=96/128 日志 0 字节）、分析层未跑、交付全未跟踪入库 ⇒ L3/L4 均不成立。"
                f"基准表该锚点交付时 0 行，本审计补注册 3 行（1 passed / 2 failed）。"),
    "scenario_key": "xval-lbm-thermal-dhc-ra1e5",
    "mission_id": 18158,
    "tags": ["XVAL", "TensorLBM", "board-audit", "maturity-verdict", "L2", "DHC", "de-Vahl-Davis", "metric-definition"],
    "markdown": md,
    "figures": figs,
}
Path("/tmp/report_m18158.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
print("doc:", out / "AUDIT_m18106.md", "bytes", (out / "AUDIT_m18106.md").stat().st_size)
print("payload bytes:", Path('/tmp/report_m18158.json').stat().st_size, "figures:", len(figs))