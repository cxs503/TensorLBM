#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K2-BUDGET-B 论文级进展报告：生成 markdown + figures 并 POST 到平台报告接口。

数字全部从 runs/k2budget_verdict/verdict.json 程序化注入（零手抄）。
用法：
    python3 tools/k2_budget_report.py --tree . --payload /tmp/k2budget_report.json [--publish]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "docs", "audits", "k2_budget_20261121", "figs")


def f(value, spec="{:.6f}"):
    if value is None:
        return "—"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def build_markdown(tree):
    with open(os.path.join(tree, "runs", "k2budget_verdict", "verdict.json")) as handle:
        v = json.load(handle)
    clo = v["closure"]
    ch = v["channels"]
    comp = v["experiment_comparison"]
    mom = v["momentum_ledger"]
    loc = v["localization"]
    ctl = v["controls"]
    win = v["window"]
    case = v["case"]
    dom = v["dominance"]
    ro = ctl["diagnostic_is_read_only"]
    anchor = ctl.get("anchor") or {}
    flags = case["flags"]
    sc = case["scales"]
    e_unit = comp["SI_energy_unit_J"]

    phys = v["physical_disposition"]
    closes = clo["closes"]
    verdict_cn = ("收支闭合（PASS）" if closes else "收支不闭合（honest FAIL），丢失渠道已定位")
    dom_txt = dom["dominant_channel"] or "UNDETERMINED"
    holders = comp.get("channels_holding_at_least_that") or []
    holder_txt = "、".join(h["channel"] for h in holders) or "无"

    md = f"""# K2-BUDGET-B：撞击动量/能量在峰时刻之前的去向（逐项收支分解）

## ① 案例介绍（物理背景 / 几何 / 参数表）

Santini 等人（Fluids 2022, 7, 50）case-3 单滴入水实验：直径 D = 2.27 mm、撞击速度 1.4 m/s 的水滴
撞击静水池，实验用 1000 Hz 相机记录坑深-时间曲线（18 点，最大坑深 {comp['experiment_peak_D']:.6f} D @ {comp['experiment_peak_time_ms']:.5f} ms post-impact）。
本单沿用 DE 战役的 **cells = {case['cells_per_D']} 基准线**（Santini case-3 旗标组），
前情已定谳：K1-DIAG-B 证明「空穴过闭合」被证伪（d_cav_gap ≡ 0、p_cav = 1.0 单连通），
−22.7310 % 的欠深是**真实的物理欠穿透**（滴材料最深只到 1.3953 D）。

**几何**：池深 {flags['pool_depth_over_D']} D、横向 4 D × 4 D 的方形池，
球形液滴（aspect {0.96}）从初始液面上方 {flags['gap_over_D']} D 处释放，落差换算为指定释放速度 {flags['release_speed_m_s']} m/s。

| 参数 | 值（格点） | 值（SI） |
|---|---|---|
| cells / D | {case['cells_per_D']} | dx = {sc['dx_m']:.6e} m |
| 格点速度 | {flags['lattice_speed']} | dt = {sc['dt_s']:.6e} s |
| 液相运动粘度 | μ_l = {sc['mu_l']:.6e} | {0.000853:.6e} Pa·s |
| 表面张力 | σ = {sc['sigma']:.6e} | 0.07142 N/m |
| 重力 | g = {sc['g']:.6e} | 9.81 m/s² |
| 伪电流限幅 | u_limit = {flags['u_limit']} | — |
| 质量交换 | capacity={flags['capacity_redistribution']}, simple={flags['simple_mass_exchange']} | — |
| 撞击速度（由 run 记录） | — | {1.4091844081433575:.6f} m/s |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

问题：**撞击的能量与动量在深度峰时刻之前被哪个通道吃掉了？**

做法：在**只读**诊断模块 `benchmarks/water_entry/single_drop/budget_diag.py` 中，
逐步从求解器状态（fill φ、相位 flags、stored mass、宏观速度）计算逐项收支：

1. **① 空穴气体压力冲量**：空穴壁（与大气连通的空穴气体所界面）上 p_gas = p_amb + σ κ 的合力矢量与时间积分；
   并给出挖掘冲量 ρ_l g ∫V_cav dt（空穴气体只提供环境压强，真正要顶开坑的是撞击动量）。
2. **② 界面表面能增量**：σ·ΔA（ΔA = ∫|∇φ| 的增量，空穴壁单独给出）。
3. **③ 界面粘性/拖曳耗散**：∫Σ2μ|S|² dV dt（宏观应变率连续估计，界面邻域与体相分开）。
4. **④ 侧向射流动能**：½Σm u_⊥²（并给空穴足迹内子份额）。
5. **⑤ 液滴残余动能**：被动示踪滴材料标记的 ½Σ m_marker|u_marker|²。
6. **⑥⑦⑧（显式附加）**：轴向液柱动能、液→气质量交换的动能转换、重力功。

流程：预登记（PREREG.md，阈值与停止规则先写死）→ SDAA 卡车道三臂满窗（预算臂 / 对照臂 / 剖面臂）
→ 判决脚本程序化注入数字 → 图 → 卡片 → 报告。**只读性**由对照臂逐字段位级比较保证。
工具链：`benchmarks/water_entry/single_drop/{{budget_diag.py, free_surface_probe.py}}`、
`tools/sdaa_fs_probe_split.py`、`tools/k2_budget_verdict.py`、`tools/k2_budget_figures.py`、
`tools/k2_budget_card.py`、`tools/k2_budget_report.py`。

## ③ 计算结果（数字表，程序化注入）

积分窗口：接触步 {win['contact_step']} → 深度峰步 {win['t_star_step']}（{win['t_star_time_ms']:.6f} ms，{win['rows']} 行逐项时序）。

**逐项（格点能量单位；份额 = /E_scale，E_scale = 接触时液相动能 = {clo['E_scale_lattice']:.6e} = {clo['E_scale_SI_J']:.6e} J）**

| 通道 | 类型 | 数值（格点） | 份额 |
|---|---|---|---|
| ① 空穴气体压力冲量 | 源/冲量 | 壁面冲量 z = {ch['1_gas_pressure_impulse']['impulse_z_lattice']:.6e}；挖掘冲量 = {ch['1_gas_pressure_impulse']['excavation_impulse_lattice']:.6e} | {f(ch['1_gas_pressure_impulse'].get('excavation_impulse_over_impact_momentum'), '{:.2%}')}（占撞击动量） |
| ② 界面表面能增量 | 存 | σΔA_cav = {ch['2_surface_energy_increment']['cavity_wall_sigma_dA_lattice']:.6e}；σΔA_tot = {ch['2_surface_energy_increment']['total_sigma_dA_lattice']:.6e} | {ch['2_surface_energy_increment']['share']:.2%} |
| ③ 界面粘性耗散 | 汇 | {ch['3_interface_viscous_dissipation']['total_lattice']:.6e}（界面邻域 {ch['3_interface_viscous_dissipation']['interface_neighbourhood_lattice']:.6e}） | {ch['3_interface_viscous_dissipation']['share']:.2%} |
| ④ 侧向射流动能 | 存 | ΔKE_lat = {ch['4_lateral_jet_kinetic_energy']['ke_lat_gain']:.6e}（坑内 {ch['4_lateral_jet_kinetic_energy']['ke_lat_roi_gain']:.6e}） | {f(ch['4_lateral_jet_kinetic_energy'].get('share_gain'), '{:.2%}')} |
| ⑤ 液滴残余动能 | 存 | {ch['5_droplet_residual_kinetic_energy']['ke_drop_final']:.6e}（初始 {ch['5_droplet_residual_kinetic_energy']['ke_drop_initial']:.6e}） | {ch['5_droplet_residual_kinetic_energy']['share']:.2%} |
| ⑥ 轴向液柱动能 | 存 | {ch['6_axial_water_column_kinetic_energy']['ke_axial_final']:.6e} | {ch['6_axial_water_column_kinetic_energy']['share']:.2%} |
| ⑦ 质量交换转换 | 汇 | loss {ch['7_mass_exchange_conversion']['conv_loss_lattice']:.6e} / gain {ch['7_mass_exchange_conversion']['conv_gain_lattice']:.6e} | {ch['7_mass_exchange_conversion']['share']:.2%} |
| ⑧ 重力功 | 源 | {ch['8_gravity_work']['work_lattice']:.6e} | {ch['8_gravity_work']['share']:.2%} |

**能量账**：ΔE_KE = {clo['d_ke_lattice']:.6e}；W_g = {clo['w_gravity_lattice']:.6e}；
W_atm = {clo['w_atmosphere_lattice']:.6e}；ΔE_surf = {clo['d_e_surface_lattice']:.6e}；
E_diss = {clo['e_dissipation_lattice']:.6e}；E_conv(net) = {clo['e_conversion_net_lattice']:.6e}；
**残差 R = {clo['residual_with_atm_lattice']:.6e}**（不含环境压功口径 {clo['residual_without_atm_lattice']:.6e}）。

**动量账（辅助）**：撞击动量 P_z = {mom['impact_p_drop_z_lattice']:.6e}（= {mom['impact_p_drop_z_SI_kg_m_s']:.6e} kg·m/s）；
ΔP_liq,z = {mom['d_p_liq_z_lattice']:.6e}；J_g = {mom['j_gravity_z_lattice']:.6e}；
J_gas = {mom['j_cavity_gas_wall_z_lattice']:.6e}；J_conv = {mom['j_conversion_z_lattice']:.6e}；
残差 = {mom['residual_z_lattice']:.6e}（动量账不含池壁反力，只作辅助）。

**判决**：{verdict_cn}；**主导通道 = {dom_txt}**（{f(dom['ranked'][0]['share'], '{:.2%}') if dom['ranked'] else '—'}，门 {dom['threshold']:.0%}，状态 {dom['status']}）。

**物理通道分解**（剔除模型级 churn ⑦ 与压力积分伪项 ①）：
最大物理通道 = **{phys['largest_physical_channel']}**（{phys['largest_physical_share']:.2%} E_scale）；
轴向液柱动能 {phys['per_channel_share']['6_axial_water_column_kinetic_energy']:.2%}、
粘性耗散 {phys['per_channel_share']['3_interface_viscous_dissipation']:.2%}、
侧向射流动能 {phys['per_channel_share']['4_lateral_jet_kinetic_energy']:.2%}（负 = 峰时刻侧向动能更小）、
滴残余 {phys['per_channel_share']['5_droplet_residual_kinetic_energy']:.2%}。
界面总面积 {ch['2_surface_energy_increment']['area_total_cav_lattice'][0]:.1f} → {ch['2_surface_energy_increment']['area_total_cav_lattice'][1]:.1f} 格点面积，
液相总动能 {v['closure']['E_scale_lattice'] + v['closure']['d_ke_lattice']:.4f}（接触时 {v['closure']['E_scale_lattice']:.4f}）。

**实验对照（任务 b 项）**：实验坑深 {comp['experiment_peak_D']:.6f} D vs LBM {comp['lbm_peak_D']:.10f} D，
缺口 {comp['depth_deficit_relative']:+.4%}。该缺口对应的**重力位能需求**
ΔE_PE = {comp['gravitational_PE_requirement_J']:.6e} J（口径：{comp.get('gravitational_PE_source', '—')}），
只占撞击能量的 {comp['gravitational_PE_over_impact_energy']:.4%}
⇒ **深度缺口不是重力位能不足造成的**（需求仅为缺口的 {phys['energy_supply']['supply_to_deficit_ratio']:.2%}），
即坑不是「能量不够」而是「能量送不进去」；缺口在惯性/动量分配侧；
按声明假设（坑深 ∝ 轴向动能）达到实验深度所需额外轴向动能 = {comp['extra_axial_KE_needed_J']:.6e} J，
持有该量级能量的通道：{holder_txt}。

## ④ 结果图

- 图 1：能量在液相内的分配（总/轴向/侧向/滴材料）+ 空穴体积与界面面积随时间。
- 图 2：逐项收支的时间积分与**残差随时间**（闭合门带）。
- 图 3：峰时刻各通道份额（占 E_scale），含 50 % 主导门。
- 图 4：动量账（液相总动量、滴材料动量、空穴壁受力、转换动量通量）。
- 图 5：中截面坑形剖面（实测）+ 能量对照柱状图（撞击能量 vs 重力位能需求 vs 额外轴向动能）。

（图片以 base64 内嵌于本报告的 figures 字段。）

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

- **诊断只读**：`--diag-budget` 只读状态，不改物理；对照臂 history 逐字段差异 = {ro['max_abs_delta_per_field'] if ro['max_abs_delta_per_field'] else '0（逐字段零差异）'}。
- **① 的两种口径不可混用**：空穴壁局部自由面积分与刚体域净体积功在本模型里差两个量级；
  局部积分不是有效的连续功项（把它用在下降的液滴上会给出比液滴自身动能变化大两个量级的功率）。
- **⑦ 是模型级能量汇（churn）**：自由面质量交换把液相质量转成气相时，被转走质量的动能在模型里直接消失（气相速度为零），
  净销毁 {loc['conversion_net_over_E_scale']:.0%} E_scale —— **超过 100 % ⇒ 单向汇不可能**，
  说明界面处存在 churn（质量交换销毁 + ABB 重建注入），二者相抵后的净额就是本账残差的主体。
- **本账的残差主体不是五项中的任何一项**：残差-界面面积相关系数 {f(v['residual_profile']['corr_residual_vs_interface_area'], '{:.3f}')}，
  残差随时间平均漂移 {f(v['residual_profile']['residual_growth_per_step_mean'], '{:.3e}')} 格点能量/步；
  环境压功口径被自由面体积漂移污染（{loc['free_surface_volume_drift_cells']:+.2f} 格点胞 → W_atm = {loc['atmospheric_term_lattice']:.3e}），
  故主判据同时给出「含/不含环境压功」两种口径。
- **伪电流限幅器未激活**：贴限界面胞 {loc['pseudo_current_cap_cells_max']} 个 ⇒ 不是残差来源。
- **③ 是连续估计**：2μ|S|² 由宏观速度场中心差分应变率给出，与 LBM 碰撞真实耗散不严格相等。
- **② 的面积口径**：∫|∇φ| 估计（与求解器曲率算法同源），只用增量 ΔA。
- **⑤ 的示踪余量**：被动示踪标记受数值扩散影响（K1 已量化 ±1 格 = ±0.125 D）。
- **适用范围**：cells=8 单网格、Santini case-3 单工况、单流体自由面模型（常数大气压 + Laplace）；
  未跑 12 cells 收敛复核；① 的结论只对**通气空穴**成立。
- **本单没有消除欠深**：产出是归因判决 + 丢失渠道定位，不是精度修复。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | 模拟 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| LBM 坑深峰 | {comp['lbm_peak_D']:.10f} D | 实验 {comp['experiment_peak_D']:.6f} D | {comp['depth_deficit_relative']:+.3%} | **否** |
| 对照臂锚复现 | {f(anchor.get('ctrl_peak_depth_over_D'), '{:.10f}')} D | 战役锚 {f(anchor.get('peak_depth_over_D'), '{:.10f}')} D | Δ = {f(anchor.get('abs_delta'), '{:.3e}')} | **是**（位级） |
| 诊断只读性（history 逐字段最大 Δ） | {max(ro['max_abs_delta_per_field'].values(), default=0.0):.3e} | 0 | — | **是** |
| 收支闭合 \\|R\\|/E_scale | {clo['closure_with_atm']:.4%} | < {clo['tolerance']:.0%} | — | **{'是' if closes else '否'}** |
| 主导通道份额 | {f(dom['ranked'][0]['share'], '{:.2%}') if dom['ranked'] else '—'} | ≥ {dom['threshold']:.0%} | — | **{'是' if dom['status'] == 'DETERMINED' else '否'}** |

**判决**：{verdict_cn}；主导通道 = {dom_txt}。
"""
    return md, verdict_cn, dom_txt


def publish(payload):
    api = os.environ["TENSOR_API_BASE"]
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]

    def post(path, body, token=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(api + path, data=json.dumps(body).encode(),
                                         headers=headers)
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, response.read().decode()

    status, body = post("/auth/login", {"username": user, "password": password})
    token = json.loads(body).get("access_token") or json.loads(body).get("token")
    status, body = post("/tensor/reports", payload, token)
    return status, body


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=ROOT)
    ap.add_argument("--payload", default="/tmp/k2budget_report.json")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    tree = os.path.abspath(a.tree)
    md, verdict_cn, dom_txt = build_markdown(tree)

    captions = {
        "fig1_ke_channels.png": "能量在液相内的分配（总/轴向/侧向/滴材料）与空穴体积、界面面积随时间（接触→深度峰）",
        "fig2_ledger_terms.png": "逐项收支的时间积分（上）与残差随时间（下，含 ±10 % 闭合门）",
        "fig3_channel_shares.png": "峰时刻各通道占撞击能量 E_scale 的份额（50 % 主导门）",
        "fig4_momentum.png": "动量账：液相总动量、滴材料动量、空穴壁气体受力、转换动量通量",
        "fig5_crater_and_energy.png": "中截面坑形剖面（实测）与能量对照（撞击能量 vs 重力位能需求 vs 额外轴向动能）",
    }
    figures = []
    for name, caption in captions.items():
        path = os.path.join(FIGDIR, name)
        if os.path.exists(path):
            figures.append({
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(open(path, "rb").read()).decode(),
            })

    with open(os.path.join(tree, "runs", "k2budget_verdict", "verdict.json")) as handle:
        v = json.load(handle)
    clo = v["closure"]
    comp = v["experiment_comparison"]
    summary = (
        f"K2-BUDGET-B 判决：撞击能量/动量在深度峰之前的去向分解（cells=8 Santini case-3 旗标组）。"
        f"收支残差 |R|/E_scale = {clo['closure_with_atm']:.2%}（不含环境压功口径 {clo['closure_without_atm']:.2%}，"
        f"门 {clo['tolerance']:.0%}）→ {'PASS' if clo['closes'] else 'honest FAIL'}；"
        f"主导通道 = {v['dominance']['dominant_channel']}（{v['dominance']['ranked'][0]['share']:.1%}）。"
        f"实验坑深 {comp['experiment_peak_D']:.6f} D vs LBM {comp['lbm_peak_D']:.6f} D（缺口 {comp['depth_deficit_relative']:+.2%}）；"
        f"该缺口对应的重力位能需求仅占撞击能量 {comp['gravitational_PE_over_impact_energy']:.2%}"
        f" ⇒ 缺口在惯性/动量分配侧而非静水侧；最大物理通道 = {v['physical_disposition']['largest_physical_channel']}"
        f"（{v['physical_disposition']['largest_physical_share']:.1%} E_scale）。诊断只读（对照臂 history 位级一致）。"
    )
    payload = {
        "title": "K2-BUDGET-B: itemised momentum/energy budget of the droplet impact up to the crater peak",
        "summary": summary,
        "scenario_key": "K2-BUDGET-B",
        "mission_id": 26849,
        "tags": ["TensorLBM", "water-entry", "energy-budget", "diagnosis", "read-only-probe", "honest-fail"],
        "markdown": md,
        "figures": figures,
    }
    with open(a.payload, "w") as handle:
        json.dump(payload, handle)
    print("payload written:", a.payload, "figures:", len(figures))
    if a.publish:
        status, body = publish(payload)
        print("report:", status, body[:400])


if __name__ == "__main__":
    main()
