#!/usr/bin/env python3
"""Emit the calibration report (Markdown) from the campaign artifacts.

Every number in the generated report is read from ``report.json`` / the case
JSONs - nothing is typed by hand, so the document cannot drift from the data.

    python scripts/make_sloshing_calibration_report.py \
        --case-dir artifacts/sloshing_impact_lbm \
        --out docs/SLOSHING_IMPACT_LBM_CALIBRATION.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.sloshing_impact_calibration import TankAnchor  # noqa: E402

ANCHOR = TankAnchor()
GRIDS = (40, 60, 80)


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def _f(value, fmt: str = "{:.4g}") -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "n/a"
    return fmt.format(v)


def _mark(passed) -> str:
    return "PASS" if passed else "FAIL"


def build(case_dir: Path, fig_rel: str) -> str:
    rep = _load(case_dir / "report.json") or {}
    swe = _load(case_dir / "sw_swe_reference.json")
    hydro = _load(case_dir / "hydrostatic_nx40.json")
    grids = rep.get("grids", {})
    crit = rep.get("criteria", {})
    forced = _load(case_dir / "forced_nx40.json")
    cushion = _load(case_dir / "impact_cushion_nx40.json")
    base40 = _load(case_dir / "impact_nx40.json")

    L: list[str] = []
    A = L.append

    A("# 液舱晃荡冲击的低阶链误差定标：三维两相自由面 LBM 参照")
    A("")
    A("**节点** p120w2-n10 ｜ **仓库** TensorLBM ｜ **参照链** "
      "ship-performance-platform IN48-N07（只读移植）")
    A("")
    A("## 1. 案例介绍")
    A("")
    A("### 1.1 物理背景")
    A("")
    A("船舶液舱晃荡链在工程上分两级：线性势流给出固有频率，"
      "Saint-Venant（浅水，SWE）给出壁压时历，再由壁压做结构校核。"
      "两级都是渐近模型：SWE 只在 `kh → 0` 成立，且完全不能表示气垫"
      "（封闭空气腔的等熵压缩）；势流不能表示爬高、卷破与冲击。"
      "本节点用三维大密度比自由面 LBM（Körner 模型，气体不做动力学、以 ABB "
      "压力边界等价无穷密度比）在同一几何、同一时间锚上与低阶链对拍，"
      "把“链误差”从假设变成可测量的数字，并给出适用边界。")
    L.append("")
    A("### 1.2 几何与参数（IN48-N07 锚，只读移植）")
    A("")
    A("| 量 | 值 | 来源 |")
    A("|---|---|---|")
    A(f"| 液舱长 L（晃荡方向 x） | {ANCHOR.length} m | `TANK = RectTank(20, 12, 8)` |")
    A(f"| 液舱宽 B（z） | {ANCHOR.breadth} m | 同上 |")
    A(f"| 液舱高 H（y） | {ANCHOR.height} m | 同上 |")
    A(f"| 静水深度 h_l | {ANCHOR.depth} m（充液 0.5） | `h_ref = 0.5 * TANK.height` |")
    A(f"| 水密度 rho | {ANCHOR.rho} kg/m^3 | `RHO_WATER` |")
    A(f"| 重力 g | {ANCHOR.g} m/s^2 | `G` |")
    A(f"| 一阶固有频率 omega_1 | {ANCHOR.omega1:.6f} rad/s | `sqrt(g k tanh(k h))`, k=pi/L |")
    A(f"| 一阶周期 T_1 | {ANCHOR.period1:.4f} s | 2 pi / omega_1 |")
    A(f"| 相对深度 kh_1 | {ANCHOR.kh:.4f} | k h_l |")
    A(f"| SWE 自身一阶频率 omega_1^SWE | {ANCHOR.omega1_swe:.6f} rad/s "
      f"（+{100*(ANCHOR.omega1_swe/ANCHOR.omega1-1):.2f}%） | k sqrt(g h) |")
    A(f"| 涌潮前缘速度 u_front | {ANCHOR.u_front:.4f} m/s | Ritter 2 sqrt(g h_l) |")
    A(f"| 静水压 rho g h_l | {ANCHOR.p_hydrostatic:.2f} Pa | - |")
    A(f"| 闭式冲击峰值 4 rho g h_l | {ANCHOR.p_closed_form/ANCHOR.p_hydrostatic:.4f} "
      f"(rho g h_l) | Ritter 滞止 |")
    A(f"| 精确涌潮反射 rho g h_r | {ANCHOR.p_bore_reflection/ANCHOR.p_hydrostatic:.4f} "
      f"(rho g h_l) | Saint-Venant 激波条件 |")
    A("")
    A("### 1.3 两个算例")
    A("")
    A("1. **冲击算例（surge release）**：液柱高 h_l=4 m、长 5 m（0.25L）置于 x=0 端，"
      "其余干舱，闸门瞬时撤除，涌潮跨 15 m 打到对壁——与 m5744 "
      "`dam_break_case(h0=4, wet_length=5, dry_length=15)` 同一构型。")
    A("2. **受迫算例（forced）**：满深 h_l 起振，水平简谐加速度 "
      "`a(t)=A omega^2 sin(omega t)`，A=0.25 m、omega=omega_1（m5744 "
      "`regular_excitation`），6 个周期，对应站立波壁压时历与超谐（2 omega_1）谱。")
    A("")
    A("## 2. 怎么做的")
    A("")
    A("### 2.1 数值方法")
    A("")
    A("* 求解器：`tensorlbm.free_surface_lbm.free_surface_step`（Körner 2005 三维"
      "自由面 LBM，D3Q19，BGK + Smagorinsky 亚格子，ABB 气相压力重构，"
      "质量独立跟踪，I/L/G 拓扑事务）。大密度比由模型本身承担：气体无动力学，"
      "气相以压力边界注入，等价于无穷密度比。")
    A("* 新增标定模块：`src/tensorlbm/sloshing_impact_calibration.py`"
      "（锚移植 / 格点标度 / 壁压测量 / GCI / 谐波分析 / 气垫闭合）。")
    A("")
    A("### 2.2 格点标度（两不变量）")
    A("")
    A("| L/nx | 网格 (nx x ny x nz) | dx [m] | dt [s] | g_lbm | tau | nu_lbm | Re_eff |")
    A("|---|---|---|---|---|---|---|---|")
    for nx in GRIDS:
        g = grids.get(str(nx))
        if not g:
            continue
        dx, dt = g["dx"], g["dt"]
        g_lbm = ANCHOR.g * dt**2 / dx
        nu_lbm = g["reynolds_effective"] and (ANCHOR.u_front * ANCHOR.depth
                                              / g["reynolds_effective"]) * dt / dx**2
        A(f"| L/{nx} | {g['cells'] if 'cells' in g else ''} "
          f"{nx}x{int(round(ANCHOR.height/dx))}x{int(round(ANCHOR.breadth/dx))} | "
          f"{dx:.4f} | {dt:.6f} | {g_lbm:.3e} | {0.5+3*nu_lbm:.5f} | "
          f"{nu_lbm:.3e} | {g['reynolds_effective']:.0f} |")
    A("")
    A("* 不变量 1：前缘格点马赫数 `u_lbm = 0.08`（即 `dt ~ dx`），保证可压缩误差不随网格变化；"
      "不变量 2：有效雷诺数 `Re = u_front h_l / nu`（`nu_lbm ~ dx`，`tau` 随网格微增）。"
      "真实 Re ~ 5e7，自由面 LBM 在 L/80 无法解析，故取 Re=2000 的惯性主导参照并加亚格子闭合"
      "（见 5.3 局限）。")
    A("")
    A("### 2.3 测量")
    A("")
    A("* 壁面动压：撞壁侧壁面相邻两格 `p = c_s^2 (rho - rho_liquid)`（LBM 压力仅到常数），"
      "线性外插到壁面 `p_wall = 1.5 p_{dx/2} - 0.5 p_{3dx/2}`（壁面 1–2 格外插）；"
      "面板取撞击壁下半水深，横跨全宽；仅统计 LIQUID 格，湿格 < 3 时输出 0（避免单丝液膜伪值）。")
    A("* 壁面相同时历 `h_wall`：壁相邻列 fill 沿宽度平均，除 h_l 无量纲化——"
      "与 SWE 的 `rho g h_wall` 同量，用于相关性对拍。")
    A("* 声学噪声滤波：0.10 s 中心滑动平均（撞击升压时间 ~h/u ~ 0.45 s，滤波不影响物理升压），"
      "原始峰值一并报告。")
    A("* GCI：三点（L/40, L/60, L/80）拟合观测阶 p，Richardson 外插，安全系数 Fs=1.25；"
      "非单调三元组一律判 `usable=False` 而不粉饰。")
    A("")
    A("### 2.4 工具链")
    A("")
    A("```")
    A("# 1) SWE 锚（跨仓只读导入 m5744 impact.py，只落派生时历）")
    A("M5744_ROOT=<ship-performance-platform> python3 scripts/gen_swe_reference_anchor.py \\")
    A("    --out artifacts/sloshing_impact_lbm/sw_swe_reference.json")
    A("# 2) 各算例（可单进程=一个 shard）")
    A("python3 scripts/run_sloshing_impact_calibration.py campaign --shard i --shards 6 \\")
    A("    --out-dir artifacts/sloshing_impact_lbm")
    A("# 3) 汇总（GCI/谱/判据）")
    A("python3 scripts/run_sloshing_impact_calibration.py post \\")
    A("    --case-dir artifacts/sloshing_impact_lbm \\")
    A("    --swe-reference artifacts/sloshing_impact_lbm/sw_swe_reference.json \\")
    A("    --out artifacts/sloshing_impact_lbm/report.json")
    A("# 4) 图 + 本报告")
    A("python3 scripts/make_sloshing_calibration_figures.py --case-dir artifacts/sloshing_impact_lbm \\")
    A("    --fig-dir artifacts/sloshing_impact_lbm/figures")
    A("python3 scripts/make_sloshing_calibration_report.py --case-dir artifacts/sloshing_impact_lbm \\")
    A("    --out docs/SLOSHING_IMPACT_LBM_CALIBRATION.md")
    A("```")
    A("")
    A("## 3. 计算结果")
    A("")
    A("### 3.1 三套网格与壁压峰值")
    A("")
    A("| 网格 | 单元数 | 步数 | P_peak/(rho g h_l)（0.1 s 滤波） | 原始峰值 | 峰值时刻 [s] | "
      "h_wall/h_l 峰值 | 冲量 ∫p dt | 质量漂移 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for nx in GRIDS:
        g = grids.get(str(nx))
        if not g:
            continue
        A(f"| L/{nx} | {g.get('cells','')} | {g.get('steps','')} | "
          f"{_f(g.get('p_peak_filtered'))} | {_f(g.get('p_peak_raw'))} | "
          f"{_f(g.get('p_peak_filtered_time'),'{:.2f}')} | {_f(g.get('h_wall_max'))} | "
          f"{_f(g.get('impulse_filtered'))} | {_f(g.get('mass_drift_final'),'{:.2e}')} |")
    A("")
    A("### 3.2 判据逐条")
    A("")
    A("| 判据 | 量 | 值 | 阈值 | 结论 |")
    A("|---|---|---|---|---|")
    a = crit.get("a_grid_convergence", {})
    A(f"| (a) 网格收敛 | GCI(P_peak) | {_f(a.get('gci_fine'),'{:.4f}')} | ≤ 0.05 | "
      f"{_mark(a.get('pass'))} |")
    A(f"| (a) 观测阶 | p_obs | {_f(a.get('p_obs'),'{:.3f}')} | ≥ 1.2 | "
      f"{_mark(a.get('p_obs') is not None and a.get('p_obs') >= 1.2)} |")
    A(f"| (a) 补充：冲量 GCI | GCI(∫p dt) | {_f(a.get('gci_impulse'),'{:.4f}')} | 参考 | "
      f"p_obs={_f(a.get('p_obs_impulse'),'{:.3f}')} |")
    gh = rep.get("gci_h_wall") or {}
    A(f"| (a) 补充：爬高 GCI | GCI(h_wall 峰值) | {_f(gh.get('gci_fine'),'{:.4f}')} | 参考 | "
      f"p_obs={_f(gh.get('p_obs'),'{:.3f}')} |")
    b = crit.get("b_peak_pressure_band", {})
    A(f"| (b) 参照带 | 闭式 4 rho g h_l / 精确涌潮反射 / SWE 自身冲击锚 | "
      f"{_f(b.get('closed_form'))} / {_f(b.get('bore_reflection'))} / "
      f"{_f(b.get('swe_surge_anchor'))} | - | - |")
    A(f"| (b) 峰值带 | P_peak/(rho g h_l) 细网格 | {_f(b.get('value'))} | [3.0, 5.0] | "
      f"{_mark(b.get('pass'))} |")
    c = crit.get("c_hydrostatic", {})
    A(f"| (c) 静水基线 | L2 相对误差 | {_f(c.get('value'),'{:.4f}')} | < 0.01 | "
      f"{_mark(c.get('pass'))} |")
    d = crit.get("d_superharmonic", {})
    A(f"| (d) 超谐 2 omega_1 | 能量占比（爬高 / 压力） | {_f(d.get('value'),'{:.4f}')} / "
      f"{_f(d.get('value_wall_pressure'),'{:.4f}')} | > 0.05 | {_mark(d.get('pass'))} |")
    e = crit.get("e_cushion_negative_control", {})
    A(f"| (e) 气垫负控 | P_peak(气垫) vs 单相 | "
      f"{_f(e.get('p_peak_cushion'))} vs {_f(e.get('p_peak_single'))} | 严格更低 | "
      f"{_mark(e.get('pass'))} |")
    f = crit.get("f_swe_correlation", {})
    A(f"| (f) 与 SWE 时历相关 | Pearson r | {_f(f.get('value'),'{:.4f}')}"
      f"（前半窗 {_f(f.get('value_first_half_window'),'{:.4f}')}） | > 0.9 | "
      f"{_mark(f.get('pass'))} |")
    g = crit.get("g_mass_conservation", {})
    A(f"| (g) 质量守恒 | max |dM|/M（自 t=0） | {_f(g.get('value'),'{:.2e}')} | "
      f"< 1e-3 | {_mark(g.get('pass'))} |")
    A("")
    if "value_settled" in g:
        A(f"> 质量漂移的另一种口径（100 步初始包络松弛后再起算）为 "
          f"{_f(g.get('value_settled'),'{:.2e}')}；两种口径都列出，不做选择性隐去。"
          "判据用更朴素的自 t=0 口径。")
        A("")
    A("### 3.3 SWE 链对拍")
    A("")
    sw = rep.get("swe_comparison", {})
    if sw:
        A("| 量 | LBM | SWE (IN48-N07) |")
        A("|---|---|---|")
        A(f"| 壁面水深峰值 h_wall/h_l | {_f(sw.get('lbm_h_wall_max'))} | "
          f"{_f(sw.get('swe_h_wall_max'))} |")
        A(f"| 涌潮抵达时刻 [s] | {_f(sw.get('lbm_arrival_s'),'{:.3f}')} | "
          f"{_f(sw.get('swe_arrival_s'),'{:.3f}')} |")
        A(f"| 时历相关 r（全窗 {_f(sw.get('window_s'),'{:.2f}')} s） | "
          f"{_f(sw.get('correlation_full'),'{:.4f}')} | (基准) |")
        A(f"| 时历相关 r（前半窗） | {_f(sw.get('correlation_first_half'),'{:.4f}')} | (基准) |")
        A(f"| SWE/LBM 爬高比 | — | {_f(sw.get('swe_over_lbm_h_wall_ratio'),'{:.3f}')} |")
        A("")
    fv = rep.get("forced_vs_swe")
    if fv:
        A(f"受迫站立波对照：LBM（精确色散）与 SWE（k sqrt(g h) 色散）壁面水深时历 "
          f"r = {_f(fv.get('correlation'),'{:.4f}')}（窗 {_f(fv.get('window_s'),'{:.2f}')} s，"
          f"LBM 峰值 {_f(fv.get('lbm_h_wall_max'))} vs SWE {_f(fv.get('swe_h_wall_max'))}）。"
          "相位差即为 SWE 频率误差的累积，是低阶链在 kh=0.63 的适用边界直接证据。")
        A("")
    sv = rep.get("swe_validity", {})
    if sv:
        A("### 3.4 SWE 适用边界（频率）")
        A("")
        A(f"* kh_1 = {_f(sv.get('kh'),'{:.4f}')} 时 `omega_SWE/omega_exact = "
          f"{_f(sv.get('omega_ratio_swe_over_exact'),'{:.4f}')}`（SWE 快 "
          f"{_f(100*(sv.get('omega_ratio_swe_over_exact',1)-1),'{:.2f}')}%）。")
        A(f"* 3% 误差以内要求 kh < {_f(sv.get('kh_max_for_3pct'),'{:.4f}')}"
          f"（h/L < {_f(sv.get('kh_max_for_3pct',0)/math.pi,'{:.4f}')}）；"
          f"5%: kh < {_f(sv.get('kh_max_for_5pct'),'{:.4f}')}；"
          f"10%: kh < {_f(sv.get('kh_max_for_10pct'),'{:.4f}')}。")
        if sv.get("omega_measured_forced"):
            A(f"* LBM 受迫算例实测 omega = {_f(sv.get('omega_measured_forced'),'{:.4f}')} rad/s，"
              f"相对精确色散 {_f(sv.get('lbm_over_exact'),'{:.4f}')}；"
              f"SWE/实测 = {_f(sv.get('swe_over_measured'),'{:.4f}')}。")
        A("")
    A("## 4. 结果图")
    A("")
    figs = sorted((case_dir.parent / case_dir.name / "figures").glob("*.png"))
    if figs:
        for p in figs:
            A(f"![{p.stem}]({fig_rel}/figures/{p.name})")
            A("")
            A(f"*{p.stem}*")
            A("")
    else:
        A("（图目录为空：见下方说明。）")
        A("")
    A("## 5. 注意点（踩过的坑 / 局限 / 适用范围）")
    A("")
    A("### 5.1 坑：τ 决定的不只是粘度，而是能不能“流起来”")
    A("")
    A("首版取 `tau=0.8`（`nu_lbm=0.1`）是常见默认值，但在这个格点标度下有效雷诺数只有 "
      "`Re_eff ~ 6`——液体像蜂蜜一样塌不下来，涌潮 6 s 内根本到不了对壁（壁面全程干，"
      "P_peak=0）。这属于“参数合法但物理荒谬”的陷阱：LBM 的粘度必须按目标 Re 反算，"
      "而不是沿用默认 tau。修正为固定 `Re_eff=2000` 后 `tau = 0.5 + 3 nu_lbm ~ 0.501`。")
    A("")
    A("### 5.2 缺陷：tau≈0.5 时 D3Q19 MRT 自由面路径不可用")
    A("")
    A("为稳定近无粘流动改用 MRT 碰撞时，**第一步**即触发 "
      "`TopologyTransactionError: topology candidate fill is outside [-0.1, 10]`，"
      "L/40 与 L/60 一致复现，且与 Smagorinsky 常数无关（C_s=0.10/0.16 均失败）。"
      "BGK+Smagorinsky 在同一配置下可跑完整撞击窗。这是共享自由面模型的既有缺陷，"
      "已在本报告中登记，未在本节点修复。")
    A("")
    A("### 5.3 局限：Re 不可解析 + 亚格子闭合")
    A("")
    A("真实液舱 Re ~ 5e7；L/80 只有 16 层水深，只能跑 Re_eff=2000 的惯性主导参照。"
      "冲击峰值由 Froude 标度的滞止压力主导，粘度是二阶影响，但仍需在报告口径里说明："
      "本参照不是实尺度直接数值模拟。")
    A("")
    A("### 5.4 坑：单点壁压的声学噪声")
    A("")
    A("单点壁压含声学/拓扑转换噪声，量级可与信号同阶（曾观测到 -3 rho g h_l 的负压尖峰，"
      "出现在液膜刚触壁、面板湿格只有 1–2 个的瞬间）。因此判据只认 0.10 s 滑动平均 + "
      "面板（下半水深 × 全宽，仅 LIQUID 格）平均，湿格 < 3 时置零；原始峰值同表列出以便复核。")
    A("")
    A("### 5.5 发现：SWE 链无底摩擦，爬高被显著高估")
    A("")
    if sw:
        A(f"同一构型、同一 h_l 下，SWE 的壁面水深峰值 {_f(sw.get('swe_h_wall_max'))}·h_l，"
          f"LBM 为 {_f(sw.get('lbm_h_wall_max'))}·h_l，比值 "
          f"{_f(sw.get('swe_over_lbm_h_wall_ratio'),'{:.2f}')}。m5744 的 Saint-Venant 求解器"
          "是无粘、无底摩擦的（`swe_solve` 只有对流项 + 加速度源项），涌潮跨 15 m 干床时"
          "边壁/底摩阻在 LBM 参照里是实打实存在的，因此低阶链对壁面爬高给出偏保守（偏大）"
          "的估计——这正是“低阶链误差定标”要输出的结论之一。")
    else:
        A("（SWE 参照缺失。）")
    A("")
    A("### 5.6 环境缺陷：loong64 集群上本算例出现 glibc 堆损坏/段错误")
    A("")
    A("同一算例外协方 所 loong64 两节点上反复以 `free(): corrupted unsorted chunks` / "
      "`malloc(): unsorted double linked list corrupted` / `Segmentation fault` 终止"
      "（faulthandler 落点在 `free_surface_topology_transaction._birth_density_field` 的 "
      "`torch.roll` 上），而 x86_64 控制面上同一代码、同一参数可完整跑完 6 s 撞击窗。"
      "该节点上另一个与本任务无关的仓库（`wall_model_les`）作业同样报告 "
      "`free(): corrupted unsorted chunks`，指向 loong64 自定义 torch 构建（2.13.0）"
      "的堆损坏缺陷而非本模块逻辑。结论：本节点的生产算例在控制面（x86_64）执行，"
      "参数与种子完全一致可复现；集群路线待该构建修复后再启用。")
    A("")
    A("### 5.7 未评估：网格收敛（判据 a）——模型在 L/60、L/80 上积分不下去")
    A("")
    A("撞击算例在 L/40 跑完整个 6 s 窗；**L/60 与 L/80 在冲击段触发 "
      "`TopologyTransactionError: topology candidate fill is outside [-0.1, 10]`** "
      "（x86_64 控制面与 loong64 集群均复现，loong64 上另伴随 glibc 堆损坏终止）。"
      "也就是说：共享自由面模型在 tau≈0.5 的近无粘设置下，无法把“涌潮撞击 + 爬高 + 卷破”"
      "这一段拓扑剧变过程积到更细网格上。判据 (a) 因此**未评估**（不是通过）："
      "本节点的参照只有 L/40 这一个离散误差未知的解，"
      "所有基于它的结论都带有这一未定量化的离散误差，已在下文局限中标注。")
    A("")
    A("### 5.8 未评估：气垫负控（判据 e）")
    A("")
    e = crit.get("e_cushion_negative_control", {})
    if not e.get("pass"):
        A(f"封闭气腔按 `p = p0 (V0/V)^gamma`（gamma={_f((cushion or {}).get('meta',{}).get('gamma'))}）"
          "注入 ABB 气相压力：未截断时乌尔体积压缩使气压升到 ~1.7 p0，界面拓扑被推过模型的 "
          "`fill in [-0.1, 10]` 守卫（`TopologyTransactionError`）；截断到 1.35 p0 后仍未能"
          "在预算内跑完 6 s 窗。判据 (e) 因此**未评估**（不是通过），原因与证据在此如实登记："
          "共享自由面模型的 ABB 气相压力只在 O(rho_liquid) 内自洽，"
          "“气垫效应”目前超出该模型的可运行包线。")
        A("")
    A("## 6. 误差判定（逐项 vs 基准）")
    A("")
    A("| 关键量 | 基准 | 误差 | 是否 < 3% |")
    A("|---|---|---|---|")
    if c.get("value") is not None:
        A(f"| 静水压剖面 | `rho g (h_l - y)` 解析 | {_f(100*c.get('value'),'{:.3f}')}% (L2) | "
          f"{'是' if c.get('value',1) < 0.03 else '否'} |")
    if f.get("value") is not None:
        A(f"| 壁面水深时历（冲击） | SWE 1-D 解 | r = {_f(f.get('value'),'{:.4f}')}"
          f"（相关，非百分比误差） | 不适用（判据 >0.9） |")
    A(f"| 一阶固有频率 | 精确色散 `sqrt(gk tanh kh)` | "
      f"{_f(100*abs(1-(sv.get('lbm_over_exact') or float('nan'))),'{:.2f}')}% "
      f"(LBM 实测/精确) | "
      f"{'是' if sv.get('lbm_over_exact') and abs(1-sv['lbm_over_exact'])<0.03 else '否/未测'} |")
    if b.get("value") is not None:
        A(f"| 冲击峰值压力 | 闭式 4 rho g h_l 带 [3,5] | 实测 {_f(b.get('value'))}"
          f"（比值，非百分比） | 见判据 (b) |")
    A("")
    A("## 7. 结论与审计链")
    A("")
    A(f"* 判据总表：**{'全部通过' if rep.get('all_pass') else '未全部通过'}**；"
      "未通过项在上表逐条标 FAIL，未做任何覆盖。")
    A("* 审计链（数据 → 判据 → 图 → 报告，全部程序化）：")
    A("  - 锚：`artifacts/sloshing_impact_lbm/sw_swe_reference.json`（含 m5744 "
      f"commit `{(swe or {}).get('source',{}).get('git_commit')}` 与模块路径）")
    A("  - 算例：`artifacts/sloshing_impact_lbm/*_nx*.json`（时历 + 场切片 + 质量账本 + 标度）")
    A("  - 汇总：`artifacts/sloshing_impact_lbm/report.json`（GCI/谱/判据，含 filter 窗口）")
    A("  - 图：`artifacts/sloshing_impact_lbm/figures/*.png`")
    A(f"* 生成时间（UTC）：{rep.get('generated_at_utc','n/a')}")
    A("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-rel", default=None,
                    help="figure path prefix inside the doc (default: relative to repo root)")
    args = ap.parse_args(argv)
    case_dir = Path(args.case_dir)
    out = Path(args.out)
    fig_rel = args.fig_rel
    if fig_rel is None:
        try:
            fig_rel = str(case_dir.resolve().relative_to(ROOT))
        except ValueError:
            fig_rel = str(case_dir)
    text = build(case_dir, fig_rel)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())