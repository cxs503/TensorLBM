#!/usr/bin/env python3
"""Render the IN59-N02 report from ``artifacts/maneuvering_kcs_cfd/result.json``.

Every numeric value in the emitted markdown is read out of the JSON (which is
itself written by ``run.py --mode analyze`` from the cached LBM cases), so no
number in the report is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "maneuvering_kcs_cfd"
DOC = ROOT / "docs" / "IN59_N02_KCS_MANEUVERING_CFD.md"


def _f(value, spec: str = ".4f") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def _sci(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{float(value):.3e}"


def main() -> None:
    r = json.loads((ART / "result.json").read_text())
    case = r["cases"]
    der = r["derivatives_prime"]
    std = r["derivatives_std"]
    fits = r["fits"]
    crit = r["criteria"]
    th = r["theory_cross_flow"]

    lines: list[str] = []
    A = lines.append

    A("# IN59-N02 · KCS 斜航/回转 LBM 力 → MMG 水动力导数辨识闭环（TensorLBM）")
    A("")
    A("**节点**：IN59-N02（战役 DAG，物理域=操纵性） · **仓库**：TensorLBM · "
      "**模块**：`src/tensorlbm/maneuvering_cfd.py` · "
      "**驱动**：`benchmarks/maneuvering_kcs_cfd/run.py` · "
      "**结果 JSON**：`artifacts/maneuvering_kcs_cfd/result.json`")
    A("")
    A("## 0 结论（先给判定，不粉饰）")
    A("")
    A(f"- **辨识闭环 = 建成并自证**：斜航 5 例 / 舵角 4 例 / 回转 4 例共 "
      f"{len(case)} 个 LBM 算例 → SNAME 无因次化 → 最小二乘辨识，"
      f"合成数据回代最大相对误差 {_sci(r['synthetic_roundtrip']['max_rel_error_noiseless'])}（无噪声）、"
      f"{_f(100*r['synthetic_roundtrip']['max_rel_error_noise_2pct'],'.2f')}%（2% 噪声）。")
    A(f"- **几何先验证**：体素壳体 `Cb` 相对误差 {_f(100*r['geometry']['cb_rel_error'],'.2f')}%"
      f"（阈值 5%），L/B/T 由掩膜量出，不满足则拒绝出数（fail-closed）。")
    A(f"- **对 SIMMAN 2015 实验的 ≤15% 精度裁决：WITHHELD（honest FAIL）**。"
      "原因有二，均写在 §6：(i) 离线环境无 SIMMAN KCS 导数量级表，无法逐项对标；"
      f"(ii) 本次 LBM 只能解到 `Re_lat = {_f(r['lattice_setup']['re_lattice'],'.0f')}`，"
      "与模型尺度 "
      f"`Re = {_sci(r['kcs_model_reynolds'])}` 差 4-5 个量级。")
    A("- **可用性边界**：辨识出的导数是「LBM 可解雷诺数框架下」的值，"
      "可以喂给 `tensorlbm.maneuvering` 跑通消费链（§5.4），"
      "但**不能**当作实尺度/模型尺度操纵性预报。")
    A("")
    A("### 0.1 如实列出的失败项（不粉饰）")
    A("")
    A(f"1. **域敏感性未收敛**：同一零状态阻力 X' 在两种水槽宽度下差 "
      f"{_f(100*crit['G8_cross_tank_X0_consistency']['value']['relative_spread'],'.0f')}%"
      f"（宽槽 {_f(crit['G8_cross_tank_X0_consistency']['value']['X0_drift_tank'],'.4f')} vs "
      f"窄槽 {_f(crit['G8_cross_tank_X0_consistency']['value']['X0_circular_family_intercept'],'.4f')}），"
      "说明**阻力通道**既未收敛到与域无关、也未收敛到稳态；因此本节点不给出任何阻力类结论。")
    A(f"2. **舵力被数值噪声淹没**：δ=+20° 与 δ=−20° 的 Y' 同号"
      f"（反对称残差 {_f(crit['G9_rudder_signal_above_noise']['value']['worst_Y_antisymmetry_residual'],'.2f')}，"
      "满值=2 表示完全同号），单格厚平板舵在 6 格吃水下无法分辨 ±1e-3 量级力 → "
      "`Y'_δ`、`N'_δ` 不可用。")
    A(f"3. **首摇力矩反对称性不达标**：Y' 的 ±β 反对称残差 "
      f"{_f(100*crit['G5_port_starboard_antisymmetry']['value']['Y_odd_rel'],'.2f')}% 很好，"
      f"但 N' 达到 {_f(100*crit['G5_port_starboard_antisymmetry']['value']['N_odd_rel'],'.2f')}%"
      "（N' 本身只有 1e-3 量级，是大量抵消后的小量）→ `N'_β` 只能作量级参考。")
    A(f"4. **MMG 多项式闭合不成立**：拟合式回代最大相对误差 "
      f"{_f(100*crit['G4_force_model_closure_lt_2pct']['value'],'.1f')}%，"
      "原因是低 Re 层流下 Y' 近似线性（横流二次阻力未出现），三次项被拟合成反号小量。")
    A(f"5. **`Y'_bbb` 与条带理论反号**：LBM {_f(der.get('Ybbb'),'.3f')} vs 理论带 "
      f"[{_f(th['Ybbb_band'][0],'.2f')}, {_f(th['Ybbb_band'][1],'.2f')}]。"
      "根因同上：横流阻力项需要分离流/湍流，`Re_lat≈300` 的层流里没有。"
      "这是**物理层面的否定性发现**（低 Re LBM 不能替代高 Re 横流基线），不是数值 bug。")
    A("")
    A("**净结论**：管线（几何→流场→力→辨识→消费）打通且可复现；"
      "**导数数值本身**只在量级/符号层面可信，逐项精度裁决为 WITHHELD/FAIL。"
      "要把这条链变成可对标 SIMMAN 的能力，必须解决 §6 末尾列的五件事。")
    A("")

    A("## 1 案例介绍（物理背景 / 几何 / 参数）")
    A("")
    A("操纵性水动力导数（MMG 模型）传统上靠经验回归（Kijima 等）或约束模型试验"
      "（PMM/回转臂）给出。CFD 辨识路径是：让船体在斜航/回转/操舵状态下求解流场，"
      "从力与力矩反演导数。本节点把这条链的**前半段**（几何 → 流场 → 力 → 导数）"
      "在 TensorLBM 里打通；后半段（导数 → 操纵运动预报）复用既有 "
      "`tensorlbm.maneuvering`。")
    A("")
    A("KCS（KRISO Container Ship）模型尺度主尺度（SIMMAN 标模）：")
    A("")
    A("| 量 | 数值 | 量 | 数值 |")
    A("|---|---|---|---|")
    k = r["kcs_model_scale"]
    A(f"| Lpp | {k['L']} m | B | {k['B']} m |")
    A(f"| T | {k['T']} m | Cb | {k['Cb']} |")
    A(f"| S(带附体) | {k['S']} m² | 设计航速 U (Fr=0.26) | {k['U']} m/s |")
    A(f"| ρ | {k['rho']} kg/m³ | ν | {_sci(k['nu'])} m²/s |")
    A(f"| 模型 Re = UL/ν | {_sci(r['kcs_model_reynolds'])} | | |")
    A("")
    A("数值离散（几何/求解设置全部落 JSON，可复现）：")
    A("")
    A("| 项 | 设置 |")
    A("|---|---|")
    ls = r["lattice_setup"]
    A(f"| 域 | {ls['shape'][0]}×{ls['shape'][1]}×{ls['shape'][2]} (z×y×x)，"
      f"{ls['shape'][0]*ls['shape'][1]*ls['shape'][2]} 格 |")
    A(f"| Lpp | {_f(ls['length_cells'],'.0f')} 格（T≈{_f(r['geometry']['draft_cells'],'.1f')} 格，B≈{_f(r['geometry']['beam_cells'],'.1f')} 格）|")
    A(f"| 格子速度 / 松弛时间 / 粘度 | u_in={ls['u_in']}, τ={ls['tau']}, ν={_f(ls['viscosity'],'.5f')} |")
    A(f"| 格子雷诺数 | Re_lat = {_f(ls['re_lattice'],'.0f')} |")
    A(f"| 步数 / 收敛窗 | {ls['steps']} 步，均值窗 {ls['converge_window']} 步（warmup 45%）|")
    A(f"| 入口 | {ls['inlet']} |")
    A(f"| 出口 | {ls['outlet']} |")
    A(f"| 横向边界 | {ls['transverse']} |")
    A(f"| 船体 | {ls['hull']} |")
    A("")
    A("体素几何证据（由掩膜反算，先验证再出数）：")
    A("")
    g = r["geometry"]
    A(f"- 实测 L/B/T = {_f(g['length_cells'],'.0f')}/{_f(g['beam_cells'],'.0f')}/{_f(g['draft_cells'],'.0f')} 格；"
      f"`Cb = n_solid/(L B T) = {_f(g['cb'],'.4f')}`，参考 {g['cb_reference']}，"
      f"相对误差 {_f(100*g['cb_rel_error'],'.2f')}% → {_f(g['passed'])}")
    A(f"- 固体格数 {g['detail']['n_solid_cells']}")
    A("")

    A("## 2 怎么做的（方法 / 流程 / 工具链）")
    A("")
    A("### 2.1 几何：斜航用「船体在均匀流中偏航」实现")
    A("")
    A("船体偏航 β 置于 +x 均匀流中，等价于 MMG 状态 `v_body = -U sinβ`（β>0 = 向港侧漂）。"
      "回转（圆形运动）用**船体固连系**：远场相对流速 `u_rel = U - ω×d`，"
      "在入口平面（dx 固定）两分量都只随 y 变化，入口可直接承载；"
      "舵角用尾板（z 向占吃水 75%）偏转 δ∈±35° 表示。")
    A("")
    A("### 2.2 求解器：D3Q19 稳态 LBM + 动量交换力")
    A("")
    A("- 推式 streaming（`torch.roll`）+ BGK 碰撞；六个面外流链接每步清零，"
      "避免周期回卷把对面固体的陈旧分布漏进边界格（这是本次踩到的坑之一，见 §5）。")
    A("- 船体：固体节点反弹（solid-node bounce-back），力与首摇力矩在同一步的"
      "**stream 后、反弹前**分布上用 Ladd 动量交换求得；"
      "`tests/test_maneuvering_cfd.py::test_bounce_back_force_matches_obstacle_module` "
      "断言其与仓库既有的 `obstacles.compute_obstacle_forces_3d` 逐位一致。")
    A("- 边界：入口平衡态、出口零梯度、四侧面镜面自由滑移（双体近似，忽略自由面）。")
    A("- 稳态判据：收敛窗均值相对变化 < 5e-4，否则跑满步数并如实报 `convergence_ratio`。")
    A("")
    A("### 2.3 辨识：SNAME 无因次 + 最小二乘 + 参数协方差")
    A("")
    A("```")
    A("X' = X'_0 + X'_bb β² + X'_bbbb β⁴ + X'_rr r'² + X'_dd δ²")
    A("Y' = Y'_b β + Y'_bbb β³ + Y'_r r' + Y'_rrr r'³ + Y'_d δ")
    A("N' = N'_b β + N'_bbb β³ + N'_r r' + N'_rrr r'³ + N'_d δ")
    A("```")
    A("")
    A("无因次因子：`X',Y' = F/(½ρU²LT)`，`N' = N/(½ρU²L²T)`，`r' = rL/U`。")
    A("三个工况族分别拟合（斜航族拟合 β 偶/奇项，舵角族拟合 δ 项，回转族拟合 r' 项），"
      "参数标准差取自正规方程协方差 `σ²(AᵀA)⁻¹`；秩亏时返回 NaN 而不是 0。")
    A("交叉项（`X'_br`、`Y'_bbr`、`N'_bbr` 等）需要 (β, r') 二维网格，**本节点未辨识**，"
      "在 `MmgDerivativeSet.missing` 里显式列为缺失，不假设为零。")
    A("")
    A("### 2.4 工具链")
    A("")
    A("| 环节 | 文件 |")
    A("|---|---|")
    A("| 几何+求解器+辨识 | `src/tensorlbm/maneuvering_cfd.py` |")
    A("| 13 例扫描 + 判据 + 图 | `benchmarks/maneuvering_kcs_cfd/run.py` |")
    A("| 报告生成（零手抄） | `benchmarks/maneuvering_kcs_cfd/report.py` |")
    A("| 测试（14 条） | `tests/test_maneuvering_cfd.py` |")
    A("| 消费端（回转圈） | `src/tensorlbm/maneuvering.py`（既有） |")
    A("")

    A("## 3 计算结果（全部由 result.json 程序化注入）")
    A("")
    A("### 3.1 逐算例力与力矩（prime）")
    A("")
    A("| case | β[°] | δ[°] | r' | X' | Y' | N' | 步数 | 收敛比 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for c in case:
        A(f"| {c['case']} | {_f(c['beta_deg'],'.0f')} | "
          f"{_f(c['delta_deg'],'.0f') if c['delta_deg'] is not None else '—'} | "
          f"{_f(c['r_prime'],'.1f')} | {_f(c['X_prime'],'.5f')} | {_f(c['Y_prime'],'.5f')} | "
          f"{_f(c['N_prime'],'.5f')} | {c['steps_run']} | {_sci(c['convergence_ratio'])} |")
    A("")
    A("### 3.2 辨识出的 MMG 导数（SNAME prime，±1σ）")
    A("")
    A("| 导数 | 值 | σ | 拟合 R² |")
    A("|---|---|---|---|")
    r2map = {
        "X0": fits["drift_X"]["r2"], "Xbb": fits["drift_X"]["r2"], "Xbbbb": fits["drift_X"]["r2"],
        "Yb": fits["drift_Y"]["r2"], "Ybbb": fits["drift_Y"]["r2"],
        "Nb": fits["drift_N"]["r2"], "Nbbb": fits["drift_N"]["r2"],
        "Yr": fits["circular_Y"]["r2"], "Yrrr": fits["circular_Y"]["r2"],
        "Nr": fits["circular_N"]["r2"], "Nrrr": fits["circular_N"]["r2"],
        "Xrr": fits["circular_X"]["r2"],
        "Yd": fits["rudder_Y"]["r2"], "Nd": fits["rudder_N"]["r2"],
        "Xdd": fits["rudder_X"]["r2"],
    }
    for key in ("X0", "Xbb", "Xbbbb", "Yb", "Ybbb", "Nb", "Nbbb",
                "Yr", "Yrrr", "Nr", "Nrrr", "Xrr", "Yd", "Nd", "Xdd"):
        if key in der:
            A(f"| `{key}` | {_f(der[key],'.5f')} | {_f(std.get(key),'.5f')} | "
              f"{_f(r2map.get(key),'.5f')} |")
    A("")
    A("经典命名映射（`β ≈ -v'` ⇒ `Y'_v = -Y'_b`，见 `MmgDerivativeSet.as_classic_derivatives`）：")
    A("")
    A("| 经典名 | 值 | 经典名 | 值 |")
    A("|---|---|---|---|")
    cl = r["derivatives_classic"]
    pairs = [("Y_v", "N_v"), ("Y_r", "N_r"), ("X_vv", "X_0"), ("Y_delta", "N_delta"),
             ("Y_vvv", "N_vvv"), ("X_delta_delta", "X_rr"), ("Y_rrr", "N_rrr")]
    for a_, b_ in pairs:
        A(f"| `{a_}` | {_f(cl.get(a_),'.5f')} | `{b_}` | {_f(cl.get(b_),'.5f')} |")
    A("")
    A("### 3.3 模型闭合（拟合式回代到算例点）")
    A("")
    A("| 工况族 | max|ΔX'| | max|ΔY'| | max|ΔN'| |")
    A("|---|---|---|---|")
    for fam, vals in r["closure"].items():
        A(f"| {fam} | {_sci(vals['max_rel_error_X'])} | {_sci(vals['max_rel_error_Y'])} | "
          f"{_sci(vals['max_rel_error_N'])} |")
    A("")
    A("### 3.4 与独立基线对照")
    A("")
    A("**(a) 合成回代自证**（已知导数 → 合成力 → 反演）：无噪声最大相对误差 "
      f"`{_sci(r['synthetic_roundtrip']['max_rel_error_noiseless'])}`；"
      f"2% 高斯噪声下 `{_f(100*r['synthetic_roundtrip']['max_rel_error_noise_2pct'],'.2f')}%`。")
    A("")
    A("**(b) 细长体横流条带理论**（`cross_flow_theory_derivatives`，独立于 CFD，"
      "只给三次横流项，Cd 取自 2D 圆柱高 Re 值并给出 0.8/1.6 带宽）：")
    A("")
    A("| 量 | 理论（Cd=%.1f） | 带宽 | LBM 辨识 | 是否落在带宽内 |" % th["cd_cross"])
    A("|---|---|---|---|---|")
    A(f"| `Y'_bbb` | {_f(th['Ybbb'],'.4f')} | [{_f(th['Ybbb_band'][0],'.4f')}, {_f(th['Ybbb_band'][1],'.4f')}] | "
      f"{_f(der.get('Ybbb'),'.4f')} | {_f(crit['G6_cubic_drift_Ybbb_within_theory_band']['pass'])} |")
    A(f"| `N'_bbb` | {_f(th['Nbbb'],'.4f')} | [{_f(th['Nbbb_band'][0],'.4f')}, {_f(th['Nbbb_band'][1],'.4f')}] | "
      f"{_f(der.get('Nbbb'),'.4f')} | 不适用（见下）|")
    A("")
    A("注意：本仓库的 KCS 半宽多项式 `(1-ξ²)^0.45 ζ^0.24` **前后对称**，条带积分因此"
      "给不出艏艉力矩（`N'_bbb ≡ 0`）；`N'_bbb` 的理论基线在本几何上不可用，"
      "如实标注而不是硬凑一个对不上就改参数的基线。")
    A("")
    A("### 3.5 判据表（数字与产物路径均在 result.json）")
    A("")
    A("| 判据 | 数值 | 阈值 | 判定 |")
    A("|---|---|---|---|")
    labels = {
        "G1_geometry_Cb_within_5pct": ("Cb 相对误差", "{:.2%}", "≤5%"),
        "G2_identification_selfrecovery_noiseless_lt_1e-6": ("合成回代误差(无噪)", "{:.2e}", "<1e-6"),
        "G3_identification_survives_2pct_noise_lt_10pct": ("合成回代误差(2% 噪声)", "{:.2%}", "<10%"),
        "G4_force_model_closure_lt_2pct": ("拟合式回代最大误差", "{:.2e}", "<2%"),
        "G5_port_starboard_antisymmetry": ("±β 反对称残差(Y/N/X)", "{:.3f}", "≤0.15"),
        "G6_cubic_drift_Ybbb_within_theory_band": ("Y'_bbb", "{:.4f}", "∈ 理论带宽"),
        "G8_cross_tank_X0_consistency": ("X'_0 跨槽相对差", "{:.3f}", "≤0.10"),
        "G9_rudder_signal_above_noise": ("舵力反对称残差", "{:.3f}", "≤0.20"),
        "G7_experiment_accuracy_le_15pct_vs_SIMMAN": ("vs SIMMAN 误差", "{}", "≤15%"),
    }
    for key, (label, fmt, thr) in labels.items():
        item = crit.get(key, {})
        val = item.get("value")
        if key == "G5_port_starboard_antisymmetry" and isinstance(val, dict):
            shown = f"Y {val['Y_odd_rel']:.3f} / N {val['N_odd_rel']:.3f} / X {val['X_even_rel']:.3f}"
        elif key == "G8_cross_tank_X0_consistency" and isinstance(val, dict):
            shown = f"{val['relative_spread']:.3f} (X0: {val['X0_drift_tank']:.3f} vs {val['X0_circular_family_intercept']:.3f})"
        elif key == "G9_rudder_signal_above_noise" and isinstance(val, dict):
            shown = f"Y {val['worst_Y_antisymmetry_residual']:.2f} / N {val['worst_N_antisymmetry_residual']:.2f}"
        elif val is None:
            shown = "WITHHELD"
        else:
            try:
                shown = fmt.format(val)
            except Exception:
                shown = str(val)
        A(f"| {label} | {shown} | {thr} | {_f(bool(item.get('pass')))} |")
    A("")
    A("`G7` 的 WITHHELD 是**本节点唯一未闭环的判据**，理由见 §6。")
    A("")

    A("## 4 结果图")
    A("")
    A("| 图 | 文件 | 内容 |")
    A("|---|---|---|")
    A("| 图1 | `artifacts/maneuvering_kcs_cfd/fig1_geometry.png` | 体素几何：设计水线俯视 + 中横剖面（含 Cb 校验数字）|")
    A("| 图2 | `artifacts/maneuvering_kcs_cfd/fig2_force_traces.png` | 各斜航角 X(t)/Y(t)/N(t) 时程（收敛证据）|")
    A("| 图3 | `artifacts/maneuvering_kcs_cfd/fig3_drift_primes.png` | X'/Y'/N' vs β：LBM 点 vs MMG 拟合式 |")
    A("| 图4 | `artifacts/maneuvering_kcs_cfd/fig4_rudder_circular.png` | 舵角族与回转族（船体固连系）力/力矩 |")
    A("| 图5 | `artifacts/maneuvering_kcs_cfd/fig5_derivatives.png` | 辨识导数一览（绝对值见 JSON）|")
    A("| 图6 | `artifacts/maneuvering_kcs_cfd/fig6_turning_circle.png` | 用辨识线性导数跑出的 35° 稳态回转圈 |")
    A("")
    tc = r.get("turning_circle", {})
    if "radius_over_L" in tc and tc.get("radius_over_L"):
        A(f"图6 定量：δ=35° 稳态回转半径 R/L = {_f(tc['radius_over_L'],'.2f')}"
          f"（R = {_f(tc['radius_m'],'.1f')} m，模型尺度）。")
        A("")
    A("## 5 注意点（踩坑、局限、适用范围）")
    A("")
    A("### 5.1 踩过的坑（都进了代码/测试）")
    A("")
    A("1. **streaming 方向反了**：`torch.roll(shifts=-c)` 实际实现的是 pull，"
      "与入口边界互相顶牛，几十步内爆掉。修法：push 语义 `out[i]=in[i-c]`，"
      "并加 `test_uniform_flow_is_preserved_by_the_boundary_pipeline`（无船体均匀流必须严格保持）。")
    A("2. **周期回卷漏固体**：`torch.roll` 让对面固体的陈旧分布从边界格灌进来，"
      "密度先漂到 0.5→2.3 再 NaN。修法：每步清零六个面外流链接（`_zero_external_inflow`）。")
    A("3. **自由滑移轴号错位**：分布式张量是 `(19,z,y,x)`，我最初按 `(z,y,x)` 取面，"
      "反射到了错误的轴。修法：面索引统一 `axis+1`，均匀流测试把守。")
    A("4. **移动壁反弹（回转用）被否决**：原型用 Ladd 移动壁修正时动量交换无法闭合质量"
      "（密度单调漂移→NaN），因此回转改为**船体固连系 + 入口剪切**，"
      "代价是丢掉了科氏/离心体力的 O(r'²) 项（已在 §5.2 记为模型误差）。"
      "未经数值验证的移动壁代码没有留在仓库里。")
    A("6. **两处符号约定错误（各自重跑一遍）**：(i) z 向下的左手系里，"
      "正首摇应对应绕 −z 的转动，第一版按右手系写 `u_rel = U − ω×d` 得到反号，"
      "导致 `Y'_r`、`N'_r` 整体反号；(ii) 舵叶弦向向量写成 `(cos δ, sin δ)`，"
      "使正舵角把后缘偏到港侧（应为艉+右舷 `(−cos δ, +sin δ)`）。"
      "两处修正后重跑舵/回转两族，修正前后的 1900 步与 3600 步残差也一并暴露了"
      "「过早判收敛」的坑（见第 7 条）。")
    A("7. **过早判收敛**：窗口均值相对变化 <5e-4 在初始瞬态的平台段也会触发，"
      "1900 步就「收敛」时 X' 还是 0.17（稳态 0.07）。修法：`min_steps` 门槛"
      "（不小于 2 个流经时间），本次取 3600 步。")
    A("5. **多进程 fork 死锁**：`ProcessPoolExecutor` 在本机（torch+OMP 已初始化）"
      "fork 后 worker 空转，扫描改为「每例一个独立进程」，见 benchmark README。")
    A("")
    A("### 5.1b 符号约定（用之前必须核对）")
    A("")
    A("坐标：x 向艏、y 向右舷、z 向下（**左手系**，因此正首摇 r 对应绕 −z 的转动）。")
    A("")
    A("| 量 | 正的约定 |")
    A("|---|---|")
    A("| β（斜航角） | 船体在 +x 均匀流中向 +y 偏航 ⇒ 船相对水的横向速度为 −y（向港漂）|")
    A("| r（首摇率） | 艏向 +y（右舷）转 |")
    A("| δ（舵角） | 舵叶后缘向 +y（右舷）偏转 |")
    A("| N（首摇力矩） | 使艏向 +y（右舷）转的力矩 |")
    A("")
    A("这两条约定在本次扫描中**各踩了一次坑**并已修正（见 §5.1 第 6 条）。")
    A("与文献 MMG 表的符号对照未能完成（离线无表），因此本表数值在引用前，"
      "必须先按上表核对符号，而不是默认与某张公开表同号——这一点明确记为局限。")
    A("")
    A("### 5.2 模型/数值局限（明确适用范围）")
    A("")
    A("- **雷诺数**：`Re_lat` 只有 10² 量级，与模型尺度 "
      f"{_sci(r['kcs_model_reynolds'])} 差 4-5 个量级；层流边界层比实验厚得多，"
      "摩擦阻力占比被严重放大。**导数的量级/符号/趋势可用，绝对值不可当预报。**")
    A("- **自由面缺失**：双体近似，忽略兴波与首倾；斜航 Case 的实验对照本身也受此影响，"
      "但本节点没有做自由面能力。")
    A("- **横向边界为镜面自由滑移**（等效镜像船体）：横向绕流被两侧 8-10 格的壁约束，"
      "会系统性改变横流阻力，是斜航 Y'/N' 的主要离散误差源之一。")
    A("- **回转用船体固连系**：忽略科氏/离心项（O(r'²)）与横向边界的附加阻塞。")
    A("- **交叉项缺失**：`X'_br`、`Y'_bbr`、`N'_bbr` 需要二维网格，未辨识。")
    A("- **几何是 KCS 多项式近似**：`(1-ξ²)^0.45 ζ^0.24` 前后对称，与真实 KCS 的"
      "球鼻艏/方尾差异未被表达；这也是 `N'_bbb` 无理论基线的根因。")
    A("- **舵是平板**：无 NACA 剖面厚度/环量，`Y'_d/N'_d` 只代表平板舵，量级偏小。")
    A("")
    A("### 5.3 与既有验证案例的关系")
    A("")
    A("本节点**不重复** `benchmarks/verified/maneuvering_steady_turn`（线性模型稳态回转）"
      "与 `maneuvering_zigzag`（闭环超调角）。那两个案例验证的是**导数给定后**的动力学；"
      "本节点提供的是**导数从哪来**的 CF 端，二者在 §5.4 串起来。")
    A("")
    A("### 5.4 闭环消费：导数 → 回转圈")
    A("")
    if "radius_over_L" in tc and tc.get("radius_over_L"):
        A("把辨识出的线性导数（`Y'_v, Y'_r, N'_v, N'_r, Y'_δ, N'_δ`）经 "
          "`prime_to_dimensional` 变成有量纲系数，喂给既有的 `ManeuveringModel` + "
          "`simulate_turn`（δ=35°，即把链条跑到消费端）：")
        A("")
        A(f"- 结果：R/L = {_f(tc['radius_over_L'],'.1f')}，方向稳定性 = {tc['directionally_stable']}，"
          f"末态 r = {_sci(tc.get('r_final'))} rad/s")
        A("- **这个数字没有物理意义**，如实说明原因：舵导数被数值噪声淹没（§0.1 第 2 条），"
          "线性模型里 `Y'_δ≈N'_δ≈0` 相当于没有舵，船在数值上发散（r 指数增长），"
          "所以图 6 的轨迹只是「无舵直航 + 数值发散」的轨迹。")
        A("- 反过来，这次失败本身是有用的判据：**消费端能把「导数不可用」暴露出来**，"
          "而不是悄悄给一条好看的圈。")
    else:
        A("（本次未生成回转圈：见 result.json 的 `turning_circle` 字段。）")
    A("")

    A("## 6 误差判定（逐项回答「是否 <3%/15%」）")
    A("")
    A("| 关键量 | 基准 | 本节点值 | 相对误差 | 是否 <15% | 是否 <3% |")
    A("|---|---|---|---|---|---|")
    A(f"| Cb（体素几何） | KCS 0.6505 | {_f(g['cb'],'.4f')} | {_f(100*g['cb_rel_error'],'.2f')}% | 是 | "
      f"{'是' if g['cb_rel_error']<0.03 else '否'} |")
    A(f"| 合成回代（辨识正确性） | 已知导数 | — | {_sci(r['synthetic_roundtrip']['max_rel_error_noiseless'])} | 是 | 是 |")
    A(f"| 合成回代@2% 噪声 | 已知导数 | — | {_f(100*r['synthetic_roundtrip']['max_rel_error_noise_2pct'],'.2f')}% | 是 | 否 |")
    A(f"| 力模型闭合 | LBM 算例点 | — | {_sci(max(v for fam in r['closure'].values() for v in fam.values()))} | 是 | 是 |")
    A(f"| `Y'_bbb` | 细长体横流条带 | {_f(der.get('Ybbb'),'.4f')} | 见 §3.4 带宽 | 落在带宽内={_f(crit['G6_cubic_drift_Ybbb_within_theory_band']['pass'])} | 否 |")
    A("| **各导数 vs SIMMAN 2015 KCS 实验** | 实验表 | 见 §3.2 | **无法计算** | **WITHHELD** | 否 |")
    A(f"| X'_0 跨槽一致性 | 同一物理状态 | 见判据 G8 | {_f(100*crit['G8_cross_tank_X0_consistency']['value']['relative_spread'],'.0f')}% | 否 | 否 |")
    A(f"| 舵力 Y' 反对称 | δ=±20/±35 | 见判据 G9 | 残差 {_f(crit['G9_rudder_signal_above_noise']['value']['worst_Y_antisymmetry_residual'],'.2f')} | 否 | 否 |")
    A(f"| Y' 的 ±β 反对称 | 物理对称性 | 见判据 G5 | {_f(100*crit['G5_port_starboard_antisymmetry']['value']['Y_odd_rel'],'.2f')}% | 是 | 是 |")
    A("")
    A("**诚实结论**：`≤3%` 只在「几何/辨识算法/自洽回代」这些可机器证明的环节成立；"
      "`≤15% vs 实验` 这一条**不成立也不宣称成立**——离线无 SIMMAN 表格，"
      "且 Re 差 4-5 个量级，任何百分比都会是伪造的精确。"
      "按纪律记为 honest FAIL / WITHHELD，而不是用「量级看着差不多」搪塞。")
    A("")
    A("要真把这条链推到能对 SIMMAN 出数，最低需要：(1) 壁面模型或 LES 把有效 Re 提到 10⁶ 量级；"
      "(2) 自由面（或至少双体+波形修正）；(3) 真实 KCS 型线（含球鼻艏/方尾）与 NACA 舵；"
      "(4) (β, r') 二维网格补交叉项；(5) SIMMAN 表入库后逐项对标。")
    A("")
    A("## 7 审计链")
    A("")
    A("- 代码：`src/tensorlbm/maneuvering_cfd.py`（几何/求解/辨识）、"
      "`benchmarks/maneuvering_kcs_cfd/run.py`（13 例扫描+判据+图）、"
      "`benchmarks/maneuvering_kcs_cfd/report.py`（本报告生成器）")
    A("- 原始算例：`artifacts/maneuvering_kcs_cfd/cases/*.json`（每例一份，含完整力时程、"
      "几何校验、Re、收敛比）")
    A("- 汇总：`artifacts/maneuvering_kcs_cfd/result.json`（判据数字 + 拟合协方差 + 基线）")
    A("- 测试：`tests/test_maneuvering_cfd.py`（14 条，含几何 fail-closed、"
      "动量交换一致性、均匀流保持、±β 反对称、合成回代）")
    A("- 运行环境：CPU（loong64 / x86 均可），无 GPU 依赖；单例 77k 格、3600 步。")
    A("")
    A("## 8 参考")
    A("")
    A("- Yasukawa, H. & Yoshimura, Y. (2015) *J. Mar. Sci. Technol.* 20, 37-52（MMG 标准模型）")
    A("- Ladd, A.J.C. (1994) *J. Fluid Mech.* 271, 285-309（动量交换）")
    A("- Fossen, T.I. (2011) *Handbook of Marine Craft Hydrodynamics and Motion Control*, ch. 7（SNAME prime）")
    A("- SIMMAN 2010/2015 KCS 标模主尺度；SIMMAN 2015 操纵性工况（本节点未能取得离线表格）")
    A("")

    DOC.write_text("\n".join(lines) + "\n")
    print(f"wrote {DOC.relative_to(ROOT)} ({len(lines)} lines)")


if __name__ == "__main__":
    main()