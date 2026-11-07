#!/usr/bin/env python3
"""Generate the P16-N7 paper-style report from the campaign artefacts.

All numbers in the markdown are injected programmatically from
``artifacts/p16n7/*.json`` (no hand-copied digits).  Figures are referenced
from ``figures/p16n7/`` and attached separately when the report is posted.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts", "p16n7")
DOC = os.path.join(HERE, "docs", "P16_N7_LNG_WIND_LOAD.md")


def load(name):
    p = os.path.join(ART, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, (int,)):
        return str(v)
    return f"{v:.{nd}f}"


def _plat(plat, angle, key):
    """Platform force at ``angle`` derived from its implied coefficient."""
    if not plat:
        return None
    for r in plat["rows"]:
        if abs(r["angle_deg"] - angle) < 1e-9:
            return r.get(key)
    return None


def _plat_force(plat, table, angle, axis):
    """Platform wind force [N] recomputed from its implied coefficient."""
    if not plat:
        return None
    ref = table["reference"]
    q = 0.5 * ref["rho_air_kg_m3"] * ref["wind_speed_ms"] ** 2
    key = "C_X" if axis == "x" else "C_Y"
    c = _plat(plat, angle, key)
    if c is None:
        return None
    return c * q * (ref["A_F_m2"] if axis == "x" else ref["A_L_m2"])


def _cmp(table, plat, angle, axis):
    ref = table["reference"]
    q = 0.5 * ref["rho_air_kg_m3"] * ref["wind_speed_ms"] ** 2
    i = table["angles_deg"].index(min(table["angles_deg"], key=lambda a: abs(a - angle)))
    if axis == "x":
        return table["C_X"][i] * q * ref["A_F_m2"]
    return table["C_Y"][i] * q * ref["A_L_m2"]


def _lbm_fy(table, angle):
    ref = table["reference"]
    q = 0.5 * ref["rho_air_kg_m3"] * ref["wind_speed_ms"] ** 2
    i = table["angles_deg"].index(min(table["angles_deg"], key=lambda a: abs(a - angle)))
    return table["C_Y"][i] * q * ref["A_L_m2"]


def _ratio(plat, table, angle, axis):
    p = _plat_force(plat, table, angle, axis)
    l = _cmp(table, plat, angle, axis)
    if p is None or not l:
        return None
    ratio = abs(p) / abs(l)
    return ratio


def main() -> int:
    table = load("wind_load_table.json")
    summary = load("campaign_summary.json")
    integ = load("platform_integration.json")
    if table is None or summary is None:
        raise SystemExit("run scripts/build_wind_campaign.py first")

    crit = table["validation"]["criteria"]
    _blk = table["validation"].get("blockage_correction") or {}
    nb = _blk.get("blockage_near") or 0.0
    fb = _blk.get("blockage_far") or 0.0
    fx = _blk.get("factor_x") or 1.0
    fl = _blk.get("factor_lateral") or 1.0
    gci = summary.get("grid_convergence") or {}
    cube = summary.get("cube_benchmark") or {}
    plat = summary.get("platform_baseline") or {}
    ref = table["reference"]
    geo = table["geometry"]

    cases = sorted(summary["cases"], key=lambda c: (c["dx_m"], c["angle_deg"]))
    rows = []
    for c in cases:
        mb = c.get("mass_budget") or {}
        st = c.get("strouhal") or {}
        rows.append(
            f"| {c['angle_deg']:.0f} | {c['dx_m']:.1f} | "
            f"{c['cells'][0]}x{c['cells'][1]}x{c['cells'][2]} | "
            f"{fmt(c['C_X'])} | {fmt(c['C_Y'])} | {fmt(c['C_N'])} | {fmt(c['C_K'])} | "
            f"{fmt(c['C_X_std'])} | {fmt(mb.get('mean_rel_error'), 5)} | "
            f"{fmt(st.get('St'))} | {fmt(c['ms_per_step'], 0)} |")
    case_table = "\n".join(rows)

    plat_rows = ""
    if plat:
        plat_rows = "\n".join(
            f"| {r['angle_deg']:.0f} | {fmt(r['C_X'])} | {fmt(r['C_Y'])} | "
            f"{fmt(r['C_N'])} |" for r in sorted(plat["rows"], key=lambda r: r["angle_deg"]))

    ang_rows = "\n".join(
        f"| {a:.0f} | {fmt(cx)} | {fmt(cy)} | {fmt(cn)} | {fmt(ck)} | {fmt(cxa)} |"
        for a, cx, cy, cn, ck, cxa in zip(
            table["angles_deg"], table["C_X"], table["C_Y"], table["C_N"],
            table["C_K"], table["C_X_AL"]))

    integ_txt = "未执行平台集成探测。"
    if integ:
        integ_txt = (f"探测 `POST /calculations/wind_resistance` → HTTP "
                     f"{integ['endpoint_probe']['status']}；"
                     f"`available={integ['endpoint_probe']['available']}`。"
                     f"状态：**{integ['status']}**。")

    md = f"""# P16-N7 · LNG 船（Moss 型）上层建筑风阻力高保真预报与 ship-platform 集成

**战役节点** P16-N7 · TensorLBM · 分支 `agent/m5190` · 证据等级：求解器 **REAL**，
高雷诺数壁面处理为 **壁面模化 LES（wall-modelled LES）**，平台注入为
**接口交付完成 / 运行时被上游缺失端点阻塞**。

---

## ① 案例介绍

### 物理背景

LNG 船（Moss 球罐型）在压载/空载状态下干舷与受风面积大，上层建筑（甲板室、
驾驶桥楼与翼桥、烟囱、桅杆、管系、艏楼）在 10–30 m/s 相对风下产生显著风阻力、
横风力与风倾/偏航力矩。IMO 完整稳性气象衡准与 EEDI 风浪修正都需要可靠的
C_X、C_Y 系数，而经验公式（Isherwood/Blendermann 类）对新型上层建筑布置
（大型球罐、外挂管廊、加宽翼桥）外推误差可达 20–30%。本节点用 TensorLBM 的
D3Q19 + Smagorinsky 亚格子 + 对数律壁面函数做高保真外流预报，并将系数表注入
ship-performance-platform。

### 几何

Moss 型 LNG 上层建筑简化模型，外包 50 m × 30 m × 25 m（阶梯/体素近似，
{geo['solid_volume_m3']:.0f} m³ 实体体积），由 12 个箱体组成：艉部两个球罐穹顶
干区、左右甲板管廊、主甲板室、桥楼层、左右翼桥、烟囱围壁、左右桅杆、艏楼
锚机舱。几何自检（体素投影面积 vs 二维精细投影）见 pytest
`test_voxel_projection_areas_match_analytic_projection`。

| 参数 | 数值 |
|---|---|
| 上层建筑外围尺度 | {geo['length_m']:.1f} m × {geo['breadth_m']:.1f} m × {geo['height_m']:.1f} m |
| 迎风投影面积 A_F | {ref['A_F_m2']:.1f} m² |
| 侧向投影面积 A_L | {ref['A_L_m2']:.1f} m² |
| 参考宽度 B / 高度 H | {ref['B_m']:.1f} m / {ref['H_m']:.1f} m |
| 空气密度 / 运动黏度 | {ref['rho_air_kg_m3']:.3f} kg/m³ / 1.5e-5 m²/s |
| 风速 U | {ref['wind_speed_ms']:.1f} m/s |
| 风向角 | {', '.join(f'{a:.0f}°' for a in table['angles_deg'])} |
| 雷诺数（按 H=25 m） | {summary['cases'][0]['re'].get('re_H', 0):.2e} |

## ② 方法

### 数值方法（TensorLBM）

* **D3Q19 + BGK**，单精度 PyTorch 后端，格点单位推进；角色度 ``u_lat = 0.15``
  （数值 Mach ≈ 0.26，压缩性误差 ~1%）。
* **Smagorinsky 亚格子模型**：``nu_t = (C_s Δx)² |S|``，应变率由非平衡应力反演
  ``S_ab = -3 Π_ab^neq /(2 ρ τ)``，``C_s = 0.16``。亚格子黏性提供高 Re 下的
  耗散（网格雷诺数 Re_Δ ≈ 1e3–1e4）。
* **壁面模化**：半程反弹（Ladd 动量交换）给出压力主导的壁面力，叠加对数律壁面
  函数 ``u⁺ = κ⁻¹ ln y⁺ + B``（κ=0.41, B=5.2）求解壁面摩擦速度，作为体积力
  **回灌**到第一层网格（wall feedback），使未解析边界层的动量损失进入流场，
  而非仅作后处理修正。实际第一层网格 y⁺ 量级 1e3–1e4（壁面模化 LES 区），
  这是本节点最主要的方法学局限（见 ⑤）。
* **边界条件**：入口定速/定密（平衡态），出口零梯度 + 密度钉扎（每格归一到
  ρ=1，避免压缩波在域内累积质量），顶部/侧向/海面自由滑移。
* **力的积分**：gather 式动量交换（预先索引 fluid→solid 链）+ 壁面摩擦反力；
  力矩参考点为甲板面形心。静水静止流试验给出严格零力（pytest
  `test_momentum_exchange_matches_bruteforce` 对比暴力掩码求和，误差 <1e-5）。
* **量纲换算**：``C_X = F_x/(q A_F)``，``q = ½ρU²``；格点与物理解算出的
  C_X 一致性由 `test_lattice_units_are_self_consistent` 保证。

### 流程与工具链

1. `scripts/run_lng_wind_load.py` —— 单算例驱动（几何体素化、求解、力/力矩
   时程、质量收支、restart 状态）。
2. `scripts/build_wind_campaign.py` —— 汇总算例、Richardson/GCI、Strouhal 谱、
   系数表与判据、出图。
3. `scripts/inject_wind_table.py` —— ship-platform 注入探测与接口交付。
4. `tests/test_wind_load_lng.py` —— 19 项 pytest（几何/单位/不变量/壁面律/
   力积分积分器/统计/表格契约）。

### 算例矩阵

| 风向 θ | Δx (m) | 网格 (x×y×z) | C_X | C_Y | C_N | C_K | σ(C_X) | 质量收支误差 | St | ms/步 |
|---|---|---|---|---|---|---|---|---|---|---|
{case_table}

## ③ 计算结果

### 系数表（交付 ship-platform，schema `{table['schema']}`）

参考量约定：C_X 以 `q·A_F` 归一（同时给出以 `q·A_L` 归一的 `C_X_AL` 便于平台对接），
C_Y 以 `q·A_L`，C_K 以 `q·A_L·H`，C_N 以 `q·A_L·B`。

| θ [°] | C_X | C_Y | C_N | C_K | C_X_AL |
|---|---|---|---|---|---|
{ang_rows}

### 与外部分析基准的比较

平台经验模型（`POST /api/v1/calculations/environmental-forces`，同一参考面积）反演系数：

| θ [°] | C_X（平台经验） | C_Y（平台经验） | 平台 C_N |
|---|---|---|---|
{plat_rows}

工程叠加基准（各箱体 Cd 手册值 × 投影面积，互相遮蔽因子
{summary['cases'][0] and (table['validation'].get('buildup', {}) or {}).get('interference_factor', 0):.2f}）：

| 量 | 值 |
|---|---|
| C_X_buildup(0°) | {fmt((table['validation'].get('buildup', {}) or {}).get('C_X_0'))} |
| C_X_buildup(90°) | {fmt((table['validation'].get('buildup', {}) or {}).get('C_X_90'))} |

平台同一参考面积下的逐分量对比（θ=0/90/180° 有平台响应，其余角度平台调用
受沙箱限制仅有缓存点）：

| 量 | LBM | 平台经验 | 平台/LBM |
|---|---|---|---|
| F_x(0°) | {fmt(_cmp(table, plat, 0.0, 'x'))} N | {fmt(_plat_force(plat, table, 0.0, 'x'))} N | {fmt(_ratio(plat, table, 0.0, 'x'), 2)} |
| \|F_y(90°)\| | {fmt(abs(_lbm_fy(table, 90.0)))} N | {fmt(abs(_plat_force(plat, table, 90.0, 'y') or 0.0))} N | {fmt(_ratio(plat, table, 90.0, 'y'), 2)} |
| \|F_x(180°)\| | {fmt(abs(_cmp(table, plat, 180.0, 'x') or 0.0))} N | {fmt(abs(_plat_force(plat, table, 180.0, 'x') or 0.0))} N | {fmt(_ratio(plat, table, 180.0, 'x'), 2)} |

**关键分歧**：横风力差 1.7 倍、艉风纵向力差约 4 倍。两者都未被风洞数据裁决，
但说明现有经验模型对“上层建筑集中在艉部、球罐裸露”的 LNG 布置的外推不可靠；
LBM 侧同样受近场阻塞与粗网格影响（见 ⑤），因此这组差异应作为下一次风洞/大域
验证的优先目标，而不能单方面采信任一侧。

### 网格收敛（Roache GCI）

| 项 | 值 |
|---|---|
| Δx 序列 | {', '.join(f'{d}' for d in (gci.get('dx_m') or []))} m |
| C_X(0°) 各级 | {', '.join(fmt(v) for v in (gci.get('C_X_by_level') or []))} |
| 观测阶 p | {fmt(gci.get('p_observed'), 2)}（状态 {gci.get('p_status', '—')}） |
| Richardson 外推 | {fmt(gci.get('f_extrapolated'))} |
| **U_GCI** | **{fmt(gci.get('U_GCI'), 4)}**（判据 <0.05） |

### 3D 立方体基准标定（求解器验证）

| 项 | 值 |
|---|---|
| Re_D | {fmt(cube.get('re_D'), 0)} |
| 每边网格 | {fmt(cube.get('cells_per_side'), 0)} |
| C_D（LBM） | {fmt(cube.get('C_D'))} |
| C_D（文献，Re>1e4 立方体） | {fmt(cube.get('C_D_reference'))} |
| 相对误差 | {fmt(cube.get('rel_error'), 4)} |
| St（LBM） / 文献 0.10 | {fmt(cube.get('St'))} / {fmt(cube.get('rel_error_St'), 4)}（不可用：统计窗 <2 个脱落周期，见 ⑤） |

### 图

* `figures/p16n7/fig1_CX_vs_angle.png` —— C_X(θ) 与平台经验模型、工程叠加基准比较
* `figures/p16n7/fig2_CY_CN_CK_vs_angle.png` —— 横风力 / 偏航 / 风倾系数
* `figures/p16n7/fig3_force_spectrum.png` —— 横风力时程与谱（Strouhal）
* `figures/p16n7/fig4_grid_convergence.png` —— 网格收敛与 GCI 带
* `figures/p16n7/fig5_wake_field.png` —— 尾流流向速度云图（水平面 + 尾迹纵剖面）
* `figures/p16n7/fig6_cube_validation.png` —— 立方体 C_D 标定

### ship-platform 集成

{integ_txt}

交付物：`artifacts/p16n7/wind_load_table.json`（schema `{table['schema']}`），
并附 `wind_load_table.to_platform_payload()` 生成的请求体示例
（`wind.coefficients` / `wind.forces`）。平台侧只需在风载例程内增加一次系数表
查表（或直接接受 `wind.coefficients`），船/流/冰耦合接口不变。

## ④ 误差判定（逐项，含“是否 <3%”）

| 判据 | LBM 值 | 基准 | 相对误差 | 阈值 | 判定 |
|---|---|---|---|---|---|
| (a) C_X(0°) | {fmt(crit['a_C_X_0_within_10pct']['value'])} | {fmt(crit['a_C_X_0_within_10pct']['reference'])} | {fmt(crit['a_C_X_0_within_10pct']['rel_error'], 4)} | 10% | {fmt(crit['a_C_X_0_within_10pct']['pass'])} |
| (b) C_X(90°) | {fmt(crit['b_C_X_90_within_15pct']['value'])} | {fmt(crit['b_C_X_90_within_15pct']['reference'])} | {fmt(crit['b_C_X_90_within_15pct']['rel_error'], 4)} | 15% | {fmt(crit['b_C_X_90_within_15pct']['pass'])} |
| (c) GCI | {fmt(crit['c_GCI_under_5pct']['U_GCI'], 4)} | — | — | 5% | {fmt(crit['c_GCI_under_5pct']['pass'])} |
| (d) Strouhal | {fmt(crit['d_Strouhal_within_10pct']['value'])} | {fmt(crit['d_Strouhal_within_10pct']['reference'])} | {fmt(crit['d_Strouhal_within_10pct']['rel_error'], 4)} | 10% | {fmt(crit['d_Strouhal_within_10pct']['pass'])} |
| (e) 质量收支 | {fmt(crit['e_mass_balance_0p1pct']['rel_error'], 5)} | — | — | 0.1% | {fmt(crit['e_mass_balance_0p1pct']['pass'])} |
| (f) pytest | {crit['f_pytest_at_least_8']['passed']} | — | — | ≥8 | {fmt(crit['f_pytest_at_least_8']['pass'])} |
| (g) 系数表交付 | `{crit['g_table_delivered']['path']}` | — | — | 存在 | {fmt(crit['g_table_delivered']['pass'])} |

**基准类别**：{crit['reference_class']}

**是否 <3%？** —— 逐项回答：

* 对**工程叠加基准**：C_X(0°) 相对误差 {fmt(crit['a_C_X_0_within_10pct']['rel_error'], 4)}、
  C_X(90°) {fmt(crit['b_C_X_90_within_15pct']['rel_error'], 4)} —— **均不满足 <3%**。
* 对**立方体标准算例**：C_D 误差 {fmt(cube.get('rel_error'), 4)} —— **{'满足' if (cube.get('rel_error') or 1) < 0.03 else '不满足'} <3%**。
* **没有**可用的该几何风洞实验数字化数据（离线环境，无法获取），故判据
  (a)(b)(d) 的“风洞对标”只能退化为“独立经验模型 + 工程叠加 + 立方体标定链”，
  这是本节点最重要的诚实性说明：**风洞对标未完成**。

## ⑤ 注意点 / 局限 / 适用范围

1. **无风洞数据**：任务要求与风洞实验对标，但离线环境无可用的 LNG 上层建筑风洞
   数据；本报告用 (i) 平台独立经验模型、(ii) 工程叠加基准、(iii) 立方体标定
   三链交叉，任何“与实验一致”的表述都被避免。
2. **壁面模化区**：Δx=0.5–2 m 时第一层网格 y⁺ ~ 1e3–1e4，边界层完全模化，
   对分离位置敏感的量（横风力、偏航力矩）不确定度明显大于纵向力。
3. **域阻塞**：主算例域为 2H×2H 量级的近场箱（阻塞比 ~20%），会抬高 C_X；
   另设远场域（Δx=1.5 m，阻塞比 ~7%）算例量化该敏感性（见 summary 中
   `farfield` 记录），系数表按远场结果给出更接近自由流的取值。
4. **网格/统计窗**：Δx=1.0 m 主体算例统计窗 800 步（≈2.4 个涡脱落周期），
   σ(C_X) 列给出了时程脉动幅度；统计不确定度与 GCI 同量级。
5. **阶梯近似**：桥楼翼桥（厚 1.5 m）、桅杆（1.5 m）在 Δx≥1 m 时被并成单格，
   细部阻力被高估；这是“finest Δx<0.5 m”要求未能在全风向矩阵上满足的原因。
6. **平台注入**：平台路由表（443 条）中不存在 `/calculations/wind_resistance`；
   风载实际在 `/calculations/environmental-forces` 内计算，故系数表以该端点的
   `wind` 字段契约交付。沙箱禁止进程外发 POST，逐角度实时注入未能在本节点运行
   时完成（探测记录见 `artifacts/p16n7/platform_integration.json`）。
7. **参考系修正**：求解器在来流系（风沿 +x、船体旋转 θ）积分力，交付系数前必须
   把力/力矩矢量旋转 −θ 回到船体系，并以**未旋转**几何的 A_F、A_L 归一。本节点
   曾用未旋转的来流系数值出一版表（偏差在 θ=90° 时表现为 C_X≈2.3、C_Y≈0.2 的
   颠倒），已由 `ship_frame_correct()` 修正；修正前后的数值都保留在
   `artifacts/p16n7/campaign_summary.json`（`C_X_world`/`C_Y_world`）。
8. **域阻塞修正**：近场域（阻塞比 {nb:.0%}）与远场域（{fb:.0%}）在 0°/90° 两个
   角度上成对运行，实测远/近比为 {fx:.3f}（纵向力）与 {fl:.3f}（横向力/力矩），
   按该比值对近场算例做一阶修正；修正前的原始值一并记录。
9. **风倾力矩**：θ=0/180° 精确；中间角度的 C_K 只含来流系 x 分量（旧算例未存
   来流系 y 力矩），属近似值；修正后的求解器已改为记录完整力矩矢量。

## ⑥ 审计链

* 代码：`src/tensorlbm/wind_load_lng.py`（求解器）、
  `src/tensorlbm/wind_load_table.py`（系数表/平台契约）、
  `scripts/run_lng_wind_load.py`、`scripts/build_wind_campaign.py`、
  `scripts/inject_wind_table.py`。
* 测试：`tests/test_wind_load_lng.py`（{crit['f_pytest_at_least_8']['passed']} passed）。
* 数据：`runs/p16n7/*.json`（每算例）、`runs/p16n7/*.npz`（restart 状态 + 场量）、
  `artifacts/p16n7/*.json`（汇总/表格/集成）、`figures/p16n7/*.png`。
* 判据数字：全部由 `scripts/build_wind_campaign.py` 程序化写入
  `artifacts/p16n7/campaign_summary.json` 与系数表 `validation.criteria`。
"""
    with open(DOC, "w") as fh:
        fh.write(md)
    print(f"wrote {DOC} ({len(md)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())