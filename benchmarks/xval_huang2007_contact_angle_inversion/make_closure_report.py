#!/usr/bin/env python3
"""Generate the round-2 closure audit markdown from closure_result.json.

Every number in the produced document is read from the result file, so the
report cannot drift from the run (no hand-transcribed percentages).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "xval_huang2007_contact_angle_inversion"
OUT = REPO / "docs" / "AUDIT_xval_huang2007_round2_closure.md"

STAGE_TITLES = {
    "in_grid_gridA": "in-grid closure（标定网格 A 上反解 → 重跑 → 实测）",
    "transfer_gridB_with_calA": "transfer（网格 A 标定直接用在网格 B，不重标定）",
    "recal_gridB": "recalibration（网格 B 重标定后再闭环）",
}


def fmt(v, nd: int = 2) -> str:
    if isinstance(v, float):
        if not math.isfinite(v):
            return "n/a"
        return f"{v:.{nd}f}"
    return str(v)


def main(name: str = "closure_result") -> int:
    data = json.loads((ART / f"{name}.json").read_text())
    cfg = data["config"]
    lines: list[str] = []
    add = lines.append

    add("# 审计：Huang-2007 接触角反解的**闭环**验证（第 2 轮）")
    add("")
    add("**锚点**：`XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION`  ")
    add("**对手**：第 1 轮 `benchmarks/xval_huang2007_contact_angle_inversion` 的「Huang2007 反解 8/9 通过 <15%」  ")
    add(f"**本轮结论**：closed-loop verdict = **{data['verdict']['verdict']}**，"
        f"分阶段 {data['verdict']['stages']}")
    add("")
    add("---")
    add("")
    add("## 0. 为什么这一轮必须做（第 1 轮的实质缺陷）")
    add("")
    add("第 1 轮把「用标定曲线把 θ_target 反解成 f_w，再用**同一条曲线**正求回去」得到的结果当成验收：")
    add("在标定区间内，这个复合是恒等映射，误差只剩线性插值的舍入。"
        "本轮用第 1 轮自己发布的标定表复算，得到：")
    add("")
    add(f"- 区间内目标（60/75/90/105/120°）的自插值误差 = **{fmt(data['round1_tautology']['in_range_max_err_pct'], 3)} %**"
        "（即 1e-14 量级，恒等映射）。")
    add("")
    add("| θ_target | 反解 f_w | 正求回 θ | err% | 是否被端点钳制 |")
    add("|---|---|---|---|---|")
    for r in data["round1_tautology"]["rows"]:
        add(f"| {fmt(r['theta_target'],0)} | {fmt(r['fw'],4)} | {fmt(r['theta_roundtrip'],2)} | "
            f"{fmt(r['err_pct'],3)} | {'是' if r['clamped'] else '否'} |")
    add("")
    add("结论：第 1 轮报告的「8/9 通过 <15%」度量的是**插值自洽性**，不是 LBM 壁面润湿条件的物理正确性；"
        "真正被钳制到端点的行（区间外目标）反而可能碰巧落进 15% 内。"
        "仓内 `src/tensorlbm/contact_angle_inversion/validate.py` 虽然写了闭环测试，但从未落过证据文件。")
    add("")
    add("## 1. 本轮做法（把回路用求解器打断）")
    add("")
    add("| 阶段 | 做什么 | 能否证伪 |")
    add("|---|---|---|")
    add("| `in_grid_gridA` | 网格 A 上扫 f_w 标定 → 反解hold-out目标 → **重新跑 LBM** → 从新密度场实测 θ | 能：偏差会直接暴露 |")
    add("| `transfer_gridB_with_calA` | 用网格 A 的标定曲线，直接在**不同网格** B 上施加 f_w 并实测 | 能：检验标定是否可跨网格迁移 |")
    add("| `recal_gridB` | 在网格 B 上重标定，再闭环 | 区分「反解法错」与「标定是网格相关的」 |")
    add("| `convergence` | 同一算例步数翻倍 | 区分「平衡态」与「瞬态读数」 |")
    add("")
    add("两个**互不共享代码**的测量器同时读数：")
    add("- 主：`sc_surface_tension.contact_angle_from_mask`（冠部半宽 `w²+y²` 线性最小二乘圆拟合）")
    add("- 备：`contact_line.theta_fit.angle_from_density`（等密度线点云 + 非线性圆拟合）")
    add("")
    add("## 2. 参数表（lattice 单位，全部来自本次运行的 config 节）")
    add("")
    add("| 量 | 值 |")
    add("|---|---|")
    add(f"| 网格 A | {cfg['grid_A']['nx']}×{cfg['grid_A']['ny']}（wall_rows={cfg['grid_A']['wall_rows']}）, R={cfg['grid_A']['radius']:g} |")
    add(f"| 网格 B | {cfg['grid_B']['nx']}×{cfg['grid_B']['ny']}（wall_rows={cfg['grid_B']['wall_rows']}）, R={cfg['grid_B']['radius']:g} |")
    add(f"| 每点弛豫步数 | {cfg['steps']}（收敛检验 {cfg['convergence_steps']}） |")
    add(f"| 网格 A 扫 f_w | {', '.join(f'{v:g}' for v in cfg['fw_sweep_A'])} |")
    add(f"| 网格 B 扫 f_w | {', '.join(f'{v:g}' for v in cfg['fw_sweep_B'])} |")
    add(f"| hold-out 目标角 | {', '.join(f'{v:g}' for v in cfg['holdout_targets_deg'])}°（均非标定节点） |")
    add(f"| 共存密度 | ρ_l={cfg['rho_l']}, ρ_v={cfg['rho_v']} |")
    add(f"| SC 耦合 / 松弛 | G={cfg['G']}, τ={cfg['tau']}, ψ=1-e^{{-ρ}}, BGK D2Q9 |")
    add("")

    add("## 3. 标定曲线（实测 f_w → θ）")
    add("")
    for key, label in (("calibration_gridA", "网格 A"), ("calibration_gridB", "网格 B")):
        cal = data.get(key)
        if not cal:
            continue
        add(f"### {label}")
        add("")
        add("| f_w | θ_measured [deg] | 状态 | 末段漂移 [deg] |")
        add("|---|---|---|---|")
        for r in cal.get("table", []):
            add(f"| {fmt(r['fw'],2)} | {fmt(r['theta_measured'])} | {r['status']} | {fmt(r.get('drift_deg', float('nan')),3)} |")
        add("")
        add(f"标定元数据：Young 斜率 {fmt(cal.get('young_slope'),4)}、实测斜率 {fmt(cal.get('measured_slope'),4)}、"
            f"斜率比（Huang 修正倍率）**{fmt(cal.get('slope_ratio'),3)}×**、"
            f"中性壁 f_w*(Young) {fmt(cal.get('neutral_fw_young'),4)} vs f_w*(实测) {fmt(cal.get('neutral_fw_measured'),4)}。")
        add("")

    add("## 4. 闭环结果（逐行，程序化从结果文件生成）")
    add("")
    for stage, payload in data["stages"].items():
        m = payload["metrics"]
        add(f"### {STAGE_TITLES.get(stage, stage)}")
        add("")
        add(f"通过 **{m['n_passed']}/{m['n_measured']}**，最大误差 **{fmt(m['max_err_pct'])} %**（判据 <{m['criterion_pct']:g}%）")
        add("")
        add("| θ_target | f_w | θ_meas（主） | θ_meas（备） | 两器差 [deg] | err% | 判定 |")
        add("|---|---|---|---|---|---|---|")
        for i, r in enumerate(payload["rows"]):
            dis = r.get("estimator_disagreement", {})
            ok = "PASS" if m["rows"][i]["passed"] else "FAIL"
            add(f"| {fmt(r['theta_target'],0)} | {fmt(r.get('fw', float('nan')),4)} | "
                f"{fmt(r.get('theta_measured', float('nan')))} | "
                f"{fmt(r.get('theta_measured_alt', float('nan')))} | "
                f"{fmt(dis.get('delta_deg', float('nan')))} | "
                f"{fmt(m['rows'][i]['err_pct'])} | {ok} |")
        add("")

    conv = data.get("convergence")
    if conv:
        add("### 步数收敛检验")
        add("")
        add(f"目标 {fmt(conv['theta_target'],0)}°（f_w={fmt(conv['fw'],4)}）："
            f"{conv['steps_short']} 步读数 {fmt(conv['theta_short'])}° → "
            f"{conv['steps_long']} 步读数 {fmt(conv['theta_long'])}°，"
            f"差 {fmt(conv['delta_deg'])}°（{fmt(conv['time_step_effect_pct'])}%），"
            f"判定 {'已弛豫' if conv['relaxed'] else '未弛豫'}。")
        add("")

    add("## 5. 图")
    add("")
    add("![标定曲线](figures/fig1_calibration.png)  ")
    add("*图 1*：两套网格上实测 θ(f_w) 与 Young 关系（黑虚线）的对比——Young 关系系统性错，且斜率明显偏小。")
    add("")
    add("![闭环散点](figures/fig2_closure.png)  ")
    add("*图 2*：θ_target–θ_meas 散点，绿色带为 owner ±15% 判据；细竖线为两个独立测量器之间的离散度。")
    add("")
    add("![场量](figures/fig3_field.png)  ")
    add("*图 3*：一个 hold-out 算例弛豫后的密度场与等密度界面点云（测量器实际看到的输入）。")
    add("")

    add("## 6. 注意点 / 局限")
    add("")
    for note in data.get("honest_notes", []):
        add(f"- {note}")
    for note in data.get("notes", []):
        add(f"- {note}")
    add("- 测量器在冠部接近半球（θ≈90°）时最稳；接近 0°/180° 时半宽被前驱膜污染，"
        "主测量器靠 `max_half_width_frac` 剔除，仍属该方法已知弱区。")
    add("- 本轮只做静态（Ca=0、Bo=0）；动态接触角、滞后不在判据内。")
    add("")

    add("## 7. 误差判定（是否 <15%）")
    add("")
    add("| 阶段 | 最大误差 % | <15%? |")
    add("|---|---|---|")
    for stage, payload in data["stages"].items():
        m = payload["metrics"]
        add(f"| {stage} | {fmt(m['max_err_pct'])} | {'是' if m['all_passed'] else '否'} |")
    add("")
    add(f"**总结**：{data['verdict']['verdict']}。"
        "（本文件由 `make_closure_report.py` 从结果 JSON 生成，数字零手抄。）")
    add("")
    add(f"运行总时长：{data.get('runtime_total_s')} s；结果文件：`artifacts/xval_huang2007_contact_angle_inversion/{name}.json`。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "closure_result"))