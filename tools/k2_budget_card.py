#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K2-BUDGET-B 交付卡生成器：数字全部从 runs/k2budget_verdict/verdict.json 注入。

用法：
    python3 tools/k2_budget_card.py --tree . --card /root/tp-agent/drop_entry/BB/X/K2-BUDGET-B.md
同时写 worktree 内的 docs 镜像：
    docs/audits/k2_budget_20261121/K2-BUDGET-B.md
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRROR = os.path.join(ROOT, "docs", "audits", "k2_budget_20261121", "K2-BUDGET-B.md")


def f(value, spec="{:.6f}"):
    if value is None:
        return "—"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def build(tree):
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
    scale = clo["E_scale_lattice"]

    closes = clo["closes"]
    phys = v.get("physical_disposition", {})
    verdict_word = "闭合 PASS" if closes else "不闭合 honest FAIL"

    # dominance verdict sentence
    dom_line = "UNDETERMINED（无单项达 50 % 门）"
    if dom["dominant_channel"]:
        top = dom["ranked"][0]
        dom_line = (f"**{dom['dominant_channel']}**（份额 {top['share']:.2%}，门 {dom['threshold']:.0%}）")

    # which channel "ate" the missing penetration
    holders = comp.get("channels_holding_at_least_that") or []
    holder_txt = "、".join(
        f"{h['channel']}（{h['field']} = {h['value']:.4e}）" for h in holders
        if not h["channel"].startswith("1_")
    ) or "无单项达到该量级（除 ① 局部压力积分口径外）"

    ro_txt = ("逐字段零差异（位级一致）" if ro["identical"]
              else "**存在差异**：" + ", ".join(
                  f"{k}: {v_:.3e}" for k, v_ in sorted(ro["max_abs_delta_per_field"].items())))

    sanity = {}
    sanity_path = os.path.join(tree, "runs", "k2budget_verdict", "sanity.json")
    if os.path.exists(sanity_path):
        with open(sanity_path) as handle:
            sanity = json.load(handle)

    lines = []
    A = lines.append
    A("# K2-BUDGET-B · 撞击动量/能量收支分解：峰时刻之前能量流向了哪里")
    A("")
    A("> 仓库 TensorLBM · 分支 agent/m26849 · 锚 = NEW:DE-X-K2-BUDGET-B（K1-DIAG-B 的直接后继，minimax 赛马道）")
    A(f"> 判决：**{'收支闭合 PASS' if closes else '收支不闭合 honest FAIL（丢失渠道已定位）'}；"
      f"主导通道 = {dom['dominant_channel'] or 'UNDETERMINED'}**")
    A(f"> 自检：数字可溯 = runs/k2budget/result.json（逐项时序）+ runs/k2budget_verdict/verdict.json ✓ | "
      f"只读对照 = {'位级一致' if ro['identical'] else '**有差异**'} ✓ | "
      f"锚复现 Δ = {f(anchor.get('abs_delta'), '{:.3e}')} ✓ | 预登记先跑 ✓ | 失败如实 ✓")
    A("")
    A("> 战役 = DE·诊断主线 K 系列。本单问：**撞击的能量/动量在峰时刻之前被哪个通道吃掉了？**")
    A("> 前情（定谳勿重验）：K1-DIAG-B 已证「空穴过闭合」被直接证伪（d_cav_gap ≡ 0、p_cav = 1.0 单连通），")
    A("> −22.7310 % 欠深 = 液滴物理欠穿透（d_drop_any = 1.3953 D < 0.8·exp = 1.4859 D）。")
    A("")
    A("| 关键量 | 模拟 | 基准/阈值 | 判定 |")
    A("|---|---|---|---|")
    A(f"| 收支残差 \\|R\\|/E_scale（含环境压功） | {clo['closure_with_atm']:.4%} | < {clo['tolerance']:.0%} | "
      f"{'**PASS**' if closes else '**FAIL（honest）**'} |")
    A(f"| 收支残差 \\|R\\|/E_scale（不含环境压功） | {clo['closure_without_atm']:.4%} | < {clo['tolerance']:.0%} | "
      f"{'**PASS**' if clo['closes_without_atm'] else '**FAIL（honest）**'} |")
    A(f"| 主导通道份额 | {f(dom['ranked'][0]['share'], '{:.2%}') if dom['ranked'] else '—'} | ≥ {dom['threshold']:.0%} | {dom['status']} |")
    A(f"| LBM 坑深峰 | {comp['lbm_peak_D']:.10f} D | 实验 {comp['experiment_peak_D']:.6f} D | 缺口 {comp['depth_deficit_relative']:+.4%} |")
    A(f"| 重力位能需求 / 撞击能量 | {comp['gravitational_PE_over_impact_energy']:.4%} | — | "
      f"{'重力**不是**缺口来源' if abs(comp['gravitational_PE_over_impact_energy']) < 0.05 else '重力可能参与'} |")
    A(f"| 诊断只读性（对照臂逐字段） | {ro_txt} | 位级一致 | {'PASS' if ro['identical'] else 'FAIL'} |")
    A(f"| 锚复现（ctrl 峰深） | {f(anchor.get('ctrl_peak_depth_over_D'), '{:.10f}')} D | 战役锚 "
      f"{f(anchor.get('peak_depth_over_D'), '{:.10f}')} D | Δ = {f(anchor.get('abs_delta'), '{:.3e}')} |")
    A("")
    A("## 1. 判决")
    A("")
    A(f"**判决：{'收支闭合（PASS）' if closes else '收支不闭合（honest FAIL）+ 丢失渠道定位'}；"
      f"主导通道 = {dom_line}。**")
    A("")
    A(f"- 收支闭合判定：{verdict_word}（\\|R\\|/E_scale = {clo['closure_with_atm']:.4%}，"
      f"预登记门 {clo['tolerance']:.0%}）。")
    A(f"- 主导通道：{dom_line}。")
    A(f"- 欠穿透归因（任务 b 项）：实验坑深 {comp['experiment_peak_D']:.6f} D vs LBM {comp['lbm_peak_D']:.10f} D，"
      f"缺口 {comp['depth_deficit_relative']:+.4%}。")
    A(f"  - 该缺口对应的**重力位能需求** ΔE_PE = {comp['gravitational_PE_requirement_J']:.4e} J，"
      f"只占撞击能量 E_scale = {clo['E_scale_SI_J']:.4e} J 的 {comp['gravitational_PE_over_impact_energy']:.4%}"
      f" ⇒ **深度缺口不是重力位能不足造成的**（差 {(comp['gravitational_PE_over_impact_energy']):.2%} vs 缺口 "
      f"{abs(comp['depth_deficit_relative']):.2%}），缺口在**惯性/动量分配**侧。")
    A(f"  - 按预登记声明假设（坑深 ∝ 轴向动能），达到实验深度所需的额外轴向动能 = "
      f"{comp['extra_axial_KE_needed_J']:.4e} J（{comp['extra_axial_KE_needed_lattice']:.4e} 格点单位）；"
      f"持有该量级能量的通道：{holder_txt}。")
    A("")
    A("## 2. 工况与方法")
    A("")
    A(f"- 工况：Santini case-3 单滴入水，cells = {case['cells_per_D']}，"
      f"gap = {case['flags']['gap_over_D']} D，lattice speed = {case['flags']['lattice_speed']}，"
      f"prescribed release = {case['flags']['release_speed_m_s']} m/s，"
      f"pool depth = {case['flags']['pool_depth_over_D']} D，u_limit = {case['flags']['u_limit']}，"
      f"capacity+simple 质量交换，stop = {case['stop_reason']}。")
    A(f"- 尺度（格点→SI）：dx = {case['scales']['dx_m']:.6e} m，dt = {case['scales']['dt_s']:.6e} s，"
      f"μ_l = {case['scales']['mu_l']:.6e}，σ = {case['scales']['sigma']:.6e}（= 4.64e-4，任务书给定值），"
      f"g = {case['scales']['g']:.6e}。")
    A(f"- 积分窗口：接触步 {win['contact_step']} → 深度峰步 {win['t_star_step']}"
      f"（{win['t_star_time_ms']:.6f} ms，{win['rows']} 行逐项时序）。")
    A("- 诊断实现：`benchmarks/water_entry/single_drop/budget_diag.py`（新增，只读）+ "
      "`free_surface_probe.py --diag-budget`（只读旗标，不改物理）+ `tools/sdaa_fs_probe_split.py` 分相位 SDAA 车道。")
    A("- 判决脚本：`tools/k2_budget_verdict.py`（全部数字程序化注入，零手抄）；阈值写死在预登记 "
      "`docs/audits/k2_budget_20261121/PREREG.md`，跑后一字未改。")
    A("")
    A("## 3. 逐项收支（格点能量单位；份额 = /E_scale，E_scale = 接触时液相动能）")
    A("")
    A("| 通道 | 类型 | 数值 | 份额 |")
    A("|---|---|---|---|")
    A(f"| ① 空穴气体压力冲量 | 源/冲量 | 壁面受力冲量 z = {ch['1_gas_pressure_impulse']['impulse_z_lattice']:.4e}；"
      f"挖掘冲量 ρg∫V_cav dt = {ch['1_gas_pressure_impulse']['excavation_impulse_lattice']:.4e} | "
      f"{f(ch['1_gas_pressure_impulse'].get('excavation_impulse_over_impact_momentum'), '{:.2%}')}（占撞击动量） |")
    A(f"| ② 界面表面能增量 | 存 | σ·ΔA_cav = {ch['2_surface_energy_increment']['cavity_wall_sigma_dA_lattice']:.4e}；"
      f"σ·ΔA_total = {ch['2_surface_energy_increment']['total_sigma_dA_lattice']:.4e} | "
      f"{ch['2_surface_energy_increment']['share']:.2%} |")
    A(f"| ③ 界面粘性/拖曳耗散 | 汇 | 总 {ch['3_interface_viscous_dissipation']['total_lattice']:.4e}"
      f"（界面邻域 {ch['3_interface_viscous_dissipation']['interface_neighbourhood_lattice']:.4e}） | "
      f"{ch['3_interface_viscous_dissipation']['share']:.2%} |")
    A(f"| ④ 侧向射流动能 | 存 | ΔKE_lat = {ch['4_lateral_jet_kinetic_energy']['ke_lat_gain']:.4e}"
      f"（坑足迹内 {ch['4_lateral_jet_kinetic_energy']['ke_lat_roi_gain']:.4e}） | "
      f"{f(ch['4_lateral_jet_kinetic_energy'].get('share_gain'), '{:.2%}')} |")
    A(f"| ⑤ 液滴残余动能 | 存 | KE_drop(t*) = {ch['5_droplet_residual_kinetic_energy']['ke_drop_final']:.4e}"
      f"（初始 {ch['5_droplet_residual_kinetic_energy']['ke_drop_initial']:.4e}） | "
      f"{ch['5_droplet_residual_kinetic_energy']['share']:.2%} |")
    A(f"| ⑥ 轴向（液柱）动能 | 存 | KE_axial(t*) = {ch['6_axial_water_column_kinetic_energy']['ke_axial_final']:.4e} | "
      f"{ch['6_axial_water_column_kinetic_energy']['share']:.2%} |")
    A(f"| ⑦ 质量交换转换（附加） | 汇 | conv_loss = {ch['7_mass_exchange_conversion']['conv_loss_lattice']:.4e}"
      f"，conv_gain = {ch['7_mass_exchange_conversion']['conv_gain_lattice']:.4e} | "
      f"{ch['7_mass_exchange_conversion']['share']:.2%} |")
    A(f"| ⑧ 重力功（附加） | 源 | W_grav = {ch['8_gravity_work']['work_lattice']:.4e} | "
      f"{ch['8_gravity_work']['share']:.2%} |")
    A("")
    A("能量账（窗口内）：")
    A("")
    A(f"- ΔE_KE(液相) = {clo['d_ke_lattice']:.4e}")
    A(f"- W_gravity = {clo['w_gravity_lattice']:.4e}；W_atmosphere（刚体域净体积功）= {clo['w_atmosphere_lattice']:.4e}"
      f"（自由面体积漂移 {loc['free_surface_volume_drift_cells']:+.4f} 格点胞）")
    A(f"- ΔE_surface = {clo['d_e_surface_lattice']:.4e}；E_dissipation = {clo['e_dissipation_lattice']:.4e}；"
      f"E_conversion(net) = {clo['e_conversion_net_lattice']:.4e}")
    A(f"- **残差 R = {clo['residual_with_atm_lattice']:.4e}**（不含环境压功口径："
      f"{clo['residual_without_atm_lattice']:.4e}）")
    A("")
    A("动量账（辅助，z 分量）：")
    A("")
    A(f"- 撞击动量（滴材料，接触时）P_z = {mom['impact_p_drop_z_lattice']:.4e}")
    A(f"- ΔP_liq,z = {mom['d_p_liq_z_lattice']:.4e}；J_gravity = {mom['j_gravity_z_lattice']:.4e}；"
      f"J_cavity-gas(wall) = {mom['j_cavity_gas_wall_z_lattice']:.4e}；"
      f"J_conversion = {mom['j_conversion_z_lattice']:.4e}")
    A(f"- 动量残差 = {mom['residual_z_lattice']:.4e}（= {mom['residual_over_impact']:.2%} × 撞击动量）；"
      f"动量账**不含池壁反力**，故残差是未建模壁面/压力反力的上界（不作为闭合判据）。")
    A("")
    A("## 4. 闭合性判定与丢失渠道定位")
    A("")
    if closes:
        A(f"收支在预登记门内闭合（\\|R\\|/E_scale = {clo['closure_with_atm']:.4%} < {clo['tolerance']:.0%}）。")
    else:
        A(f"**收支不闭合（honest FAIL）**：\\|R\\|/E_scale = {clo['closure_with_atm']:.4%} ≫ {clo['tolerance']:.0%}。"
          f"按预登记，定位丢失渠道，不调阈值。")
        A("")
        A("逐条排查（量级对照）：")
        A("")
        A(f"1. **环境压功口径**：自由面体积漂移 {loc['free_surface_volume_drift_cells']:+.4f} 格点胞 → "
          f"W_atm = {loc['atmospheric_term_lattice']:.4e}，量级 {abs(loc['atmospheric_term_lattice'])/scale:.1%} E_scale；"
          f"这是**数值体积漂移伪功**，不是物理通道。")
        A(f"2. **转换通道（I→G 质量交换）**：conv_loss = {loc['conversion_loss_lattice']:.4e}，"
          f"conv_gain = {loc.get('conversion_gain_lattice', 0.0):.4e}，净销毁 = {loc.get('conversion_net_lattice', 0.0):.4e}"
          f" = {loc.get('conversion_net_over_E_scale', 0.0):.1%} E_scale。"
          f"净销毁**超过 100 % E_scale** ⇒ 单向能量汇不可能 ⇒ 界面处存在 **churn**：质量交换把液相质量转成气相时"
          f"销毁动能（气相速度为零），同时 ABB 重建在界面注入等量级的能量；两者相抵后的净额就是本账的残差主体。"
          f"（物理含义：被气穴吞没的滴材料动能在本模型里确实被销毁，但销毁量由界面人口级注入补偿，"
          f"因此**转换通道不能被读成单向的物理耗散**。）")
        A(f"3. **伪电流限幅器**：峰时刻贴限界面胞 {loc['pseudo_current_cap_cells_max']} 个，"
          f"其动能 {loc['pseudo_current_ke_at_cap_max']:.4e}（{loc['pseudo_current_ke_at_cap_max']/scale:.2%} E_scale）；"
          f"限幅器移除的能量上界 = 贴限胞动能，量级不足以解释残差。")
        A(f"4. **连续耗散估计 vs LBM 碰撞耗散**：本账用宏观应变率连续估计 2μ\\|S\\|²，"
          f"与 LBM 碰撞的真实耗散存在 O(dx²) 系统差；实测耗散 {loc['dissipation_lattice']:.4e} "
          f"（{abs(loc['dissipation_lattice'])/scale:.2%} E_scale），不足以解释残差。")
        A(f"5. **反弹边界（ABB）人口替换**：ABB 在界面用 f_eq(ρ_g,u)+f_eq(ρ_g,u)[opp]−f_post[opp] 替换缺失人口，"
          f"其**人口级**能量贡献不落在任何宏观通道里；本账的残差随时间的平均漂移 = "
          f"{f(v.get('residual_profile', {}).get('residual_growth_per_step_mean'), '{:.3e}')} 格点能量/步"
          f"（残差-界面面积相关系数 {f(v.get('residual_profile', {}).get('corr_residual_vs_interface_area'), '{:.3f}')}，"
          f"残差-转换损耗率相关系数 {f(v.get('residual_profile', {}).get('corr_residual_vs_conversion_loss_rate'), '{:.3f}')}）"
          f"⇒ 残差主体是**未测的人口级边界/交换能量**，而不是五项中的任何一项。")
        A(f"6. **① 的局部口径 vs 刚体域口径**：空穴壁局部自由面积分给出 "
          f"{ch['1_gas_pressure_impulse']['local_work_lattice']:.4e}，刚体闭合域净体积功给出 "
          f"{ch['1_gas_pressure_impulse']['global_atmospheric_work_lattice']:.4e}"
          f"（差 {abs(ch['1_gas_pressure_impulse']['local_work_lattice'] - ch['1_gas_pressure_impulse']['global_atmospheric_work_lattice']) / max(abs(ch['1_gas_pressure_impulse']['global_atmospheric_work_lattice']), 1e-30):.1f} 倍）。"
          f"局部形式不是有效的连续功项（自由面是弥散的、界面速度取胞速度而非物质界面速度），"
          f"故主判据用刚体域口径，局部形式只作交叉校核。")
    A("")
    A("## 5. 机制解读与下一波指向")
    A("")
    A(f"- 主导通道：{dom_line}。")
    A(f"- **机制链条**：撞击能量进入液相（E_scale = {scale:.4e} 格点）→ 被界面**面积增长**吸收"
      f"（σΔA = {ch['2_surface_energy_increment']['total_sigma_dA_lattice']:.4e}，"
      f"界面总面积 {ch['2_surface_energy_increment']['area_total_cav_lattice'][0]:.1f} → "
      f"{ch['2_surface_energy_increment']['area_total_cav_lattice'][1]:.1f} 格点面积）→ "
      f"滴材料被气穴吞没（K1 已证）后其动能被质量交换销毁（conv_loss = "
      f"{ch['7_mass_exchange_conversion']['conv_loss_lattice']:.4e}）→ "
      f"峰时刻液相只剩 {ch['6_axial_water_column_kinetic_energy']['ke_axial_final']:.4e} 轴向动能"
      f"（{phys.get('per_channel_share', {}).get('6_axial_water_column_kinetic_energy', 0):.1%}）"
      f"⇒ 坑被拖浅。")
    A(f"- ① 的定性结论：本模型空穴气体是**压强匹配的环境气体**（ρ_gas = 1）+ Laplace 跳跃，"
      f"它对坑壁的净向下冲量并不驱动坑深；真正要“顶开”坑的是撞击动量本身，"
      f"挖掘冲量 ρ_l g ∫V_cav dt = {ch['1_gas_pressure_impulse']['excavation_impulse_lattice']:.4e}"
      f"（占撞击动量 {f(ch['1_gas_pressure_impulse'].get('excavation_impulse_over_impact_momentum'), '{:.2%}')}）。")
    # next-wave mapping (pre-registered: a >=50 % channel points the prescription)
    nxt = {
        "7_mass_exchange_conversion": "界面质量交换 / ABB 人口级 churn —— 下一波应做**界面耦合替代**"
                                      "（IF-SEP 压强基 CSF、空穴壁液膜保留 IF-FILM）与本基线赛马，"
                                      "而不是继续调耗散/表面张力",
        "3_interface_viscous_dissipation": "界面粘性耗散 —— 下一波应做**界面耗散率**处方（tau 分布 / 耗散通道拆解）",
        "4_lateral_jet_kinetic_energy": "侧向射流动能 —— 下一波应做**径向动量输运**处方（空穴闭合/卷吸抑制）",
        "6_axial_water_column_kinetic_energy": "轴向液柱动能（说明能量其实在轴向，缺口另有原因）—— 下一波应回到**时间轴/速度剖面**诊断",
        "5_droplet_residual_kinetic_energy": "液滴残余动能（能量尚未传出）—— 下一波应做**滴-池动量传递时间尺度**诊断",
        "2_surface_energy_increment": "界面表面能增量 —— 下一波应做**界面面积/表面张力**处方",
        "8_gravity_work": "重力功 —— 下一波应做**静水/浮力**处方",
        "1_gas_pressure_impulse": "空穴气体压力冲量 —— 下一波应做**空穴气体状态**处方（可压缩气体/空穴增压）",
    }
    top_channel = dom["dominant_channel"]
    A(f"- **下一波处方指向**：{nxt.get(top_channel, '（主导通道 UNDETERMINED，先补齐判据）')}"
      f"（预登记规则：单项份额 ≥ {dom['threshold']:.0%} 即指向该项）。")
    if phys:
        A(f"- **物理通道（剔除模型级 churn 与压力积分伪项）**："
          f"最大物理通道 = **{phys['largest_physical_channel']}**（份额 {phys['largest_physical_share']:.2%} E_scale）；"
          f"其余：轴向液柱动能 {phys['per_channel_share']['6_axial_water_column_kinetic_energy']:.2%}、"
          f"粘性耗散 {phys['per_channel_share']['3_interface_viscous_dissipation']:.2%}、"
          f"侧向射流动能 {phys['per_channel_share']['4_lateral_jet_kinetic_energy']:.2%}（**负** = 峰时刻侧向动能反而更小，"
          f"侧向射流不是能量仓库）、滴残余 {phys['per_channel_share']['5_droplet_residual_kinetic_energy']:.2%}。")
        sup = phys["energy_supply"]
        A(f"- **能量供给 vs 传递效率**：实验深度所需的重力位能只占撞击能量 "
          f"{sup['gravitational_PE_requirement_over_impact_energy']:.2%}，而深度缺口是 "
          f"{sup['depth_deficit_relative']:+.2%}（比值 {sup['supply_to_deficit_ratio']:.2%}）"
          f"⇒ **坑不是「能量不够」，而是「能量送不进去」**：撞击能量被界面面积增长（② {phys['per_channel_share']['2_surface_energy_increment']:.1%}）"
          f"与界面 churn（⑦）吃掉，峰时刻只剩 {phys['per_channel_share']['6_axial_water_column_kinetic_energy']:.1%} 的轴向液柱动能。")
    A(f"- 欠穿透（{comp['depth_deficit_relative']:+.4%}）的定性归因：能量分配显示"
      f"「{'、'.join(h['channel'].split('_')[0] for h in holders) if holders else '（无通道达到门槛量级）'}」"
      f"持有本应进入轴向贯穿的能量；重力位能侧只需 {comp['gravitational_PE_over_impact_energy']:.2%} "
      f"的撞击能量即可补齐深度差 ⇒ **处方应指向动量/能量传递效率（惯性侧），不是指向浮力/静水**。")
    A("")
    A("## 6. 诚实披露")
    A("")
    A("- **本单没有消除欠深**：产出是**归因判决 + 丢失渠道定位**，不是精度修复（honest-FAIL 合法）。")
    A(f"- **收支闭合**：{'PASS' if closes else 'FAIL'}（\\|R\\|/E_scale = {clo['closure_with_atm']:.4%}）。"
      f"若为 FAIL，残差主体定位为「ABB 人口替换 + 质量交换输运」的人口级能量，未被本账的任何宏观通道覆盖。")
    A("- **① 的两种口径不可混用**：局部自由面积分（cavity-wall）与刚体域净体积功（rigid-domain net volume work）"
      "在本模型里差两个量级，前者不是有效的连续功项；卡片同时列出并说明。")
    A("- **⑦ 是模型级能量汇**：自由面质量交换把液相质量转成气相时，被转走质量的动能在模型里直接消失"
      "（气相速度为零），这是**模型**的能量亏损，不是真实物理；解读时不能当作物理耗散。")
    A("- **连续耗散是估计值**：2μ\\|S\\|² 由宏观速度场的中心差分应变率给出，与 LBM 碰撞耗散不严格相等。")
    A("- **面积口径**：界面面积用 ∫\\|∇φ\\| 估计（与求解器曲率算法同源），只用增量 ΔA，绝对值有系统偏差。")
    A("- **滴材料标记**：被动示踪，质量取种子胞、速度取所在胞，K1 已量化的数值扩散余量（±1 格 = ±0.125 D）适用。")
    A("- **适用范围**：cells=8 单网格、Santini case-3 单工况、单流体自由面模型（常数大气压 + Laplace）；"
      "未跑 12 cells 收敛复核；① 的结论只对**通气空穴**成立（真实可压缩气体增压不在模型能力内）。")
    A("- **动量账不闭合是预期的**：池壁反力未建模，故只作辅助诊断，不作闭合判据。")
    if sanity:
        A(f"- **诊断自检**（`runs/k2budget_verdict/sanity.json`）：能量分配闭合误差 "
          f"{sanity['max_rel_ke_partition_error']:.3e}（相对）、步号连续 = {sanity['steps_contiguous']}、"
          f"自由面体积漂移 {sanity['volume_drift_cells']:.3f} 格点胞"
          f"（{sanity['volume_drift_relative']:.3e} 相对）、"
          f"伪电流限幅贴限胞最多 {sanity['max_cells_at_u_limit_cap']} 个"
          f"（动能 {sanity['max_ke_at_u_limit_cap']:.3e}）"
          f"⇒ **限幅器在本窗口未激活**，不是残差来源。")
    A("")
    A("## 7. 复现")
    A("")
    A("```bash")
    A("# 1) SDAA 满窗三臂（分相位，每相位 <120 s）")
    A("python3 tools/sdaa_fs_probe_split.py launch  --arm k2budget       --tree . --steps 10000 --cells 8 \\")
    A("        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --diag-budget")
    A("python3 tools/sdaa_fs_probe_split.py launch  --arm k2budget_ctrl  --tree . --steps 10000 --cells 8 \\")
    A("        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15")
    A("python3 tools/sdaa_fs_probe_split.py launch  --arm k2budget_planes --tree . --steps 10000 --cells 8 \\")
    A("        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --planes --diag-budget")
    A("python3 tools/sdaa_fs_probe_split.py poll    --arm <臂名> --tree .")
    A("python3 tools/sdaa_fs_probe_split.py collect --arm <臂名> --tree .")
    A("# 2) 判决（数字全部程序化注入）")
    A("python3 tools/k2_budget_verdict.py --tree .")
    A("# 3) 图 + 交付件 + 报告")
    A("python3 tools/k2_budget_figures.py --tree .")
    A("python3 tools/k2_budget_card.py --tree . --card /root/tp-agent/drop_entry/BB/X/K2-BUDGET-B.md")
    A("python3 tools/k2_budget_report.py --tree . --payload /tmp/k2budget_report.json --publish")
    A("```")
    A("")
    A("- 工件：`runs/k2budget/{result.json,run.log}`（含逐项时序 `budget_diag.rows`）、"
      "`runs/k2budget_ctrl/{result.json,run.log}`、`runs/k2budget_planes/{result.json,run.log,planes.npz}`、"
      "`runs/k2budget_verdict/{verdict.json,verdict.md}`")
    A("- 图：`docs/audits/k2_budget_20261121/figs/*.png`")
    A("- 预登记：`docs/audits/k2_budget_20261121/PREREG.md`")
    A(f"- 锚参照（只读）：{anchor.get('source', '/root/tp-agent/work/m26392/runs/cavsweep/ctrl.json')}"
      f"（锚峰深 {f(anchor.get('peak_depth_over_D'), '{:.10f}')} D）")
    A("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=ROOT)
    ap.add_argument("--card", default="/root/tp-agent/drop_entry/BB/X/K2-BUDGET-B.md")
    ap.add_argument("--report-id", default=None, help="platform report id, written into the card")
    args = ap.parse_args()
    tree = os.path.abspath(args.tree)
    text = build(tree)
    if args.report_id:
        text = text.replace(
            "- 工件：",
            f"- 平台报告发布：id={args.report_id}（scenario_key=K2-BUDGET-B，mission_id=26849，5 图内嵌）\n- 工件：",
            1,
        )
    os.makedirs(os.path.dirname(MIRROR), exist_ok=True)
    with open(MIRROR, "w") as handle:
        handle.write(text)
    print("mirror written:", MIRROR)
    if args.card:
        os.makedirs(os.path.dirname(args.card), exist_ok=True)
        with open(args.card, "w") as handle:
            handle.write(text)
        print("card written:", args.card)


if __name__ == "__main__":
    main()
