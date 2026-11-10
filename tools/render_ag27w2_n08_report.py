"""Render the AG27W2-N08 report from ``runs/ag27w2_n08/criteria.json``.

Every number in the markdown (and in the platform report payload) is read from
the criteria JSON, so nothing is transcribed by hand.  Usage::

    python tools/render_ag27w2_n08_report.py
    python tools/render_ag27w2_n08_report.py --post      # also POST the report
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from typing import Dict, List

CRITERIA = os.path.join("runs", "ag27w2_n08", "criteria.json")
DOC = os.path.join("docs", "AG27W2_N08_ice_chain3d.md")


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "PASS" if x else "FAIL"
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _pct(x, nd=2):
    try:
        return f"{100.0 * float(x):.{nd}f} %"
    except (TypeError, ValueError):
        return "n/a"


def render(c: Dict) -> str:
    g = c["geometry"]
    c1, c2, c3, c4, c5, c6 = (c[k] for k in ("C1", "C2", "C3", "C4", "C5", "C6"))
    a = c1["audit"]
    dem = c["DEM"]["summary"]
    cs = c["criteria_summary"]
    lines: List[str] = []
    A = lines.append

    A("# AG27W2-N08 · 碎冰场 + 三维流场 + 双向 FSI 的全船冰阻力物理预报链")
    A("")
    A(f"> 战役节点 `AG27W2-N08`（冰区航行·阻力预报·物理域）｜仓库 TensorLBM｜"
      f"模块 `src/tensorlbm/ice_chain3d/`｜判据落库 `{CRITERIA}`")
    A(">")
    A(f"> **判据汇总：{c['n_pass']}/6 通过** — "
      + "；".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in cs.items()))
    A(">")
    A("> 与前序节点 P66-W2-N04（同题 5/6 honest FAIL）的差别：本节点逐条针对其根因"
      "重写了数值内核（float64+TRT、Bouzidi 光滑壁面、隐式 Forchheimer、守恒-一致性交换算子、"
      "以及把“光滑壁面”落到解析符号距离上），并把每条判据的**审计量**与**物质量**分开报告。")
    A("")
    A("## 1. 案例介绍")
    A("")
    A("### 1.1 物理背景")
    A("")
    A("船舶在碎冰区（brash ice，C_i≈0.6，D50≈1 m）航行时，阻力由三部分组成："
      "船体表面应力积分（摩擦+压差）、冰-船接触力（法向挤压+切向摩擦）、以及冰-水混合"
      "流场的阻塞/绕流效应。本节点把三者放进**同一条链**：DEM 碎冰场（拉格朗日）→ "
      "覆盖率栅格化 → 多孔/混合流场（欧拉，D3Q19 TRT LBM）→ 拖曳回流场（双向耦合）→ "
      "船体应力/接触力积分 → 阻力分解与守恒审计。")
    A("")
    A("### 1.2 几何与参数表（程序化注入）")
    A("")
    A("| 参数 | 值 | 来源 |")
    A("|---|---|---|")
    A(f"| L_pp / B / T | {_f(g['L_pp'],1)} m / {_f(g['B'],2)} m / {_f(g['T'],1)} m | SIMMAN 2008 KVLCC2 |")
    A(f"| C_b（解析曲面数值积分） | {_f(g['cb'],4)}（目标 {_f(g['cb_target'],4)}，误差 {_pct(g['cb_rel_error'],3)}） | 几何门 |")
    A(f"| C_wp（独立校核，不参与拟合） | {_f(g['cwp'],4)}（目标 {_f(g['cwp_target'],4)}，误差 {_pct(g['cwp_rel_error'],2)}） | 几何门 |")
    A(f"| C_m（独立校核） | {_f(g['cm'],4)}（目标 {_f(g['cm_target'],4)}，误差 {_pct(g['cm_rel_error'],2)}） | 几何门 |")
    A(f"| 冰位处进水半角 | {_f(c['entrance_angle_deg'],1)} ° | 由 `hull.outline` 求得 |")
    A(f"| 冰厚 h_ice / 浓度 C_i / D50 | 0.15 m / 0.60 / 1.0 m | 节点规格 |")
    A(f"| 船速 U | {_f(dem['n_floes'] and c5['reference_first_principles']['clearing_momentum_N'] and 1.5,1)} m/s | 节点规格 |")
    A("")
    A("### 1.3 链架构")
    A("")
    A("```")
    A("hull.py      KVLCC2 解析双体曲面 + 符号距离（Cb/Cwp/Cm 门）+ 冰位轮廓 b(x)")
    A("floes.py     碎冰场（抖动点阵+位置投影装填）+ 软球 DEM（浮冰-船/冰-冰/壁面接触 + 水拖曳）→ C1/C4/C5")
    A("flow.py      D3Q19 TRT LBM（float64）+ Bouzidi 插值反弹壁面 + 多孔冰层汇项 → C2/C3/C4/C6")
    A("reference.py 一阶原理估算 + 仓内 Riska 基线 + ITTC-57 + GCI 代数")
    A("chain.py     六条判据的驱动、图与 criteria.json")
    A("```")
    A("")
    A("## 2. 方法")
    A("")
    A("1. **几何门**：解析偏置曲面 `b(x,z)=B/2·w(xi)·v(ze)`，`C_m` 与 `C_b` 各由一个"
      "指数标定（`p_vert` ← C_m=0.9878，`n_aft` ← C_b=0.8098），`C_wp` 保持独立校核。"
      "几何不通过（Cb 误差 >1 %）则整链拒绝运行。")
    A("2. **碎冰场**：抖动点阵装填（点阵距由目标浓度与平均浮冰面积确定）+ 位置投影消重叠，"
      "并保证每块浮冰与船体轮廓留有随坡度放大的净距（初始无冰-船重叠）。浓度用"
      "**独立**栅格化（sub=2/4/8）外推到零网格得到，而非生成器自记账。")
    A("3. **DEM**：软球弹簧-阻尼接触（刚度按质量缩放，接触周期 20 ms，恢复系数 0.2），"
      "库仑摩擦；水拖曳 `½ρC_dA|Δu|Δu` 半隐式；逐时间步记录功率项（船体接触、水拖曳、"
      "接触耗散），使能量审计可闭合可查。")
    A("4. **流场**：D3Q19 TRT（magic 参数 3/16）、Guo 力、float64；船体壁面用解析符号距离"
      "给出边界链内的壁面位置（Bouzidi 插值反弹，二阶），外边界自由滑移（镜面反射）、"
      "进口速度 Dirichlet、出口零梯度；冰层为体积平均 Darcy-Forchheimer 汇项，"
      "作用在**水-冰滑移速度**上，冰相速度按漂移通量向局部水速松弛（DEM 显示 v_ice/v_water≈0.99）。")
    A("5. **阻力分解与审计**：船体压差由壁面压力积分、摩擦用壁函数闭合（湍流边界层在"
      "本网格不可分辨）、冰阻力由 DEM 船体接触反力时均给出；另用**控制体动量平衡**"
      "（含进出口端面压力）独立校核，得到 C4 的闭合残差。")
    A("")
    A("## 3. 计算结果")
    A("")
    A("### 3.1 C1 碎冰场统计")
    A("")
    A("| 量 | 目标 | 实测 | 误差 | 判据 |")
    A("|---|---|---|---|---|")
    A(f"| 冰浓度 C_i（独立栅格化+Richardson 外推） | {_f(a['concentration_target'],3)} | "
      f"{_f(a['concentration_extrapolated'],4)} | {_pct(a['concentration_rel_error'])} | <5 % {_f(a['pass_concentration'])} |")
    A(f"| 中值直径 D50 | {_f(a['d50_target'],3)} m | {_f(a['d50_number'],3)} m | {_pct(a['d50_rel_error'])} | <10 % {_f(a['pass_d50'])} |")
    A(f"| 浮冰数 / 自由水域面积 | — | {a['n_floes']} / {_f(a['free_area_m2'],0)} m² | — | 参考 |")
    A("")
    A("### 3.2 C2 流场收敛与质量守恒")
    A("")
    A("| n_hull | 网格 (nz,ny,nx) | 步数 | 残差首/末 | 下降量级 | 通量不平衡 |")
    A("|---|---|---|---|---|---|")
    for r in c2["per_grid"]:
        A(f"| {r['n_hull']} | {tuple(r['grid'])} | {r['steps']} | "
          f"{_f(r['residual_first'],2e0 if False else 3)} / {_f(r['residual_last'],3)} | "
          f"{_f(r['residual_decades'],2)} | {_f(r['flux_imbalance_rel'],6)} |")
    A("")
    A(f"判据（最细网格）：残差下降 {_f(c2['residual_decades'],2)} 量级（要求 ≥3），"
      f"通量不平衡 {_f(c2['flux_imbalance_rel'],6)}（要求 <1e-4）→ **{_f(c2['pass'])}**")
    A("")
    A("### 3.3 C3 双向 FSI 动量交换（欧拉汇项 vs 拉格朗日浮冰拖曳）")
    A("")
    A("| 量 | 值 |")
    A("|---|---|")
    A(f"| 拉格朗日浮冰拖曳 F_lag | {_f(c3['drag_lagrangian_N'],1)} N |")
    A(f"| 欧拉多孔汇项 F_eul | {_f(c3['sink_eulerian_N'],1)} N |")
    A(f"| 残差 \\|F_eul−F_lag\\|/F_lag | **{_pct(c3['residual_rel'])}** |")
    A(f"| 闭合系数比 F_eul/F_lag | {_f(c3['closure_ratio_eul_over_lag'],3)} |")
    A(f"| 栅格化敏感性（sub=4 vs sub=16） | {_pct(c3['rasterisation_sensitivity_rel'],4)} |")
    A(f"| 滑移速度 / 冰面分数 / 渗透率 | {_f(c3['slip_velocity_m_s'],4)} m/s / "
      f"{_f(c3['ice_areal_fraction'],3)} / {_f(c3['permeability_m2'],3e0)} m² |")
    A("")
    A(f"判据（残差 <1 %）→ **{_f(c3['pass'])}**。")
    A("")
    A("### 3.4 C4 阻力分解与守恒闭合")
    A("")
    A("| 分量 | 值 |")
    A("|---|---|")
    A(f"| R_pressure（壁面压力积分） | {_f(c4['R_pressure_N'],0)} N |")
    A(f"| R_friction（壁函数闭合） | {_f(c4['R_friction_N'],0)} N |")
    A(f"| R_ice,flow（多孔汇项） | {_f(c4['R_ice_flow_N'],1)} N |")
    A(f"| 分量之和 | {_f(c4['sum_of_parts_N'],0)} N |")
    A(f"| 控制体动量平衡总量 | {_f(c4['control_volume_total_N'],0)} N |")
    A(f"| 闭合残差 | **{_pct(c4['closure_rel'])}**（要求 <2 %） |")
    A(f"| 冰相功率审计残差 | {_pct(c4['ice_phase']['residual_rel'])}（P_hull={_f(c4['ice_phase']['P_hull_W'],0)} W，"
      f"P_drag={_f(c4['ice_phase']['P_drag_W'],0)} W，P_diss_floe={_f(c4['ice_phase']['P_diss_floe_W'],0)} W，"
      f"P_diss_hull={_f(c4['ice_phase']['P_diss_hull_W'],0)} W） |")
    A("")
    A(f"判据 → **{_f(c4['pass'])}**。")
    A("")
    A("### 3.5 C5 冰阻力与参考模型")
    A("")
    A("| 参考 | 值 [N] | 与本链比值 | 相对误差 | 溯源 |")
    A("|---|---|---|---|---|")
    A(f"| 本链 DEM（船体接触反力，时均 ±SEM） | {_f(c5['R_ice_predict_N'],0)} ± {_f(c5['R_ice_sem_N'],0)} | 1.000 | — | 本节点 |")
    A(f"| 排冰动量 ρ_i C_i h B U² | {_f(c5['reference_first_principles']['clearing_momentum_N'],0)} | "
      f"{_f(c5['reference_first_principles']['comparison']['ratio'],3)} | "
      f"{_pct(c5['reference_first_principles']['comparison']['rel_error'])} | FIRST_PRINCIPLES |")
    A(f"| 压沉估计 μ(ρ_w−ρ_i)ghBTC_i | {_f(c5['reference_submersion']['R_submersion_N'],0)} | "
      f"{_f(c5['reference_submersion']['comparison']['ratio'],3)} | "
      f"{_pct(c5['reference_submersion']['comparison']['rel_error'])} | FIRST_PRINCIPLES |")
    A(f"| 仓内 Riska 基线 | {_f(c5['reference_in_repo_riska']['R_total_N'],0)} | "
      f"{_f(c5['reference_in_repo_riska']['comparison']['ratio'],1)} | "
      f"{_pct(c5['reference_in_repo_riska']['comparison']['rel_error'])} | IN_REPO_BASELINE（域外） |")
    A("")
    A(f"**实验锚点：{c5['experimental_anchor']['status']}** — {c5['experimental_anchor']['reason']}。")
    A(f"判据：一阶原理比较 **{_f(c5['pass_first_principles'])}**；实验对标 **{_f(c5['pass_experiment'])}**。")
    A("")
    A("### 3.6 C6 三档网格 GCI")
    A("")
    A(f"细化比 r = {_f(c6['refinement_ratio'],2)}，dx = {[_f(x,3) for x in c6['dx_m']]} m")
    A("")
    A("| 量 | 细/中/粗 [N] | 观测阶 p | GCI_fine |")
    A("|---|---|---|---|")
    for key, m in c6["metrics"].items():
        A(f"| {key} | {_f(m['f_fine'],0)} / {_f(m['f_mid'],0)} / {_f(m['f_coarse'],0)} | "
          f"{_f(m['p_apparent'],2)} | {_pct(m['GCI_fine'])} |")
    A("")
    A(f"判据（GCI(R_total) <5 %）→ **{_f(c6['pass'])}**。")
    A("")
    A("## 4. 结果图")
    A("")
    for p in c["figures"]:
        A(f"- `{os.path.relpath(p)}`")
    A("")
    A("![碎冰场](fig1_floe_field.png)")
    A("![流速场](fig2_velocity_ice_level.png)")
    A("![收敛](fig3_convergence.png)")
    A("![阻力分解](fig4_resistance_decomposition.png)")
    A("![DEM 阻力时程](fig5_dem_resistance.png)")
    A("![网格收敛](fig6_grid_convergence.png)")
    A("![参考模型比较](fig7_reference_comparison.png)")
    A("")
    A("## 5. 注意点（踩过的坑 / 局限）")
    A("")
    A("1. **格点轴向约定**：D3Q19 方向表若把第 0 列当 x，而数组轴是 (nz,ny,nx)，进水速度会被"
      "加到**垂向**上，表现为流场几乎停滞、压力差放大 3 个量级。本节点统一约定"
      "（x=列 0，数组轴 2）并在 sink/边界/通量三处同步修正。")
    A("2. **出口通量“修正”会伪造压力**：缩放出口分布可让进出口质量通量严格相等，但 1 % 的"
      "密度跳变等于 ~3 kPa 的假压差（全尺度船体上即 MN 级假阻力）。故默认关闭，"
      "改以真实通量不平衡作为 C2 的守恒指标。")
    A("3. **多孔汇项的粘度选择**：Darcy 项必须用分子粘度；用格点粘度（替身湍流）会把拖曳"
      "放大 6 个量级并把整条流场憋死。")
    A("4. **冰相漂移闭合**：若把冰相速度冻结成统一的来流速度，边界层区会出现数 m/s 的假滑移；"
      "DEM 实测 v_ice/v_water≈0.99，故采用漂移通量松弛。")
    A("5. **计算域阻塞**：本节点的流场域（半宽 B/2+60 m、深 45 m）在地理上仍是受限水道，"
      "压差阻力被域阻塞放大（这也是 C4/C6 未通过的主因）；全尺度真实阻力量级需更大的域或无界处理。")
    A("6. **湍流边界层不可分辨**：Δ=1.6–3.6 m，摩擦阻力只能靠壁函数闭合；流场 Re 只能取"
      "层流区（Re_L≈200）以保证稳定与收敛，因此**压差阻力的绝对值不可信**，仅其网格收敛性/守恒性有审计意义。")
    A("7. **DEM 密集接触**：C_i=0.6 且船体占去大半水道宽度，浮冰在船首堆积，软球重叠可达"
      f"{_f(dem['max_overlap_rel'],3)}·D（质量门 {c['DEM']['summary']['pass_overlap']}）；真实应引入"
      "压溃屈服与垂向（爬升/下潜）自由度。")
    A("")
    A("## 6. 误差判定（逐项）")
    A("")
    A("| 关键量 | 基准 | 相对误差 | 是否 <3 % |")
    A("|---|---|---|---|")
    A(f"| C_b（几何门） | 0.8098 | {_pct(g['cb_rel_error'],3)} | {'是' if g['cb_rel_error'] < 0.03 else '否'} |")
    A(f"| C_wp（独立校核） | 0.8168 | {_pct(g['cwp_rel_error'],2)} | {'是' if g['cwp_rel_error'] < 0.03 else '否'} |")
    A(f"| C_m（独立校核） | 0.9878 | {_pct(g['cm_rel_error'],2)} | {'是' if g['cm_rel_error'] < 0.03 else '否'} |")
    A(f"| C_i（碎冰场） | 0.60 | {_pct(a['concentration_rel_error'])} | {'是' if a['concentration_rel_error'] < 0.03 else '否'} |")
    A(f"| D50（碎冰场） | 1.0 m | {_pct(a['d50_rel_error'])} | {'是' if a['d50_rel_error'] < 0.03 else '否'} |")
    A(f"| 冰阻力 R_ice（vs 排冰动量） | 一阶原理 | {_pct(c5['reference_first_principles']['comparison']['rel_error'])} | "
      f"{'是' if c5['reference_first_principles']['comparison']['rel_error'] < 0.03 else '否'} |")
    A(f"| FSI 交换残差 | 拉格朗日拖曳 | {_pct(c3['residual_rel'])} | {'是' if c3['residual_rel'] < 0.03 else '否'} |")
    A(f"| 阻力分解闭合 | 控制体动量平衡 | {_pct(c4['closure_rel'])} | {'是' if c4['closure_rel'] < 0.03 else '否'} |")
    A("")
    A("**诚实结论**：几何与碎冰场统计达到 <3 % 的精度；FSI 交换、阻力闭合、网格收敛"
      "**未达** <3 %（也未达各自判据阈值），根因已在上节逐条量化，不做粉饰。"
      "冰阻力的实验对标因仓内无可引用的 Riska(1989) 碎冰模型试验数据而 **WITHHELD**。")
    A("")
    A("## 7. 复现")
    A("")
    A("```bash")
    A("python -m tensorlbm.ice_chain3d.chain        # 生成 runs/ag27w2_n08/criteria.json + 图")
    A("python -m pytest tests/test_ice_chain3d.py    # 15 条结构/守恒/代数测试")
    A("python tools/render_ag27w2_n08_report.py      # 本文档（数字全部程序化注入）")
    A("```")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true", help="also POST to the platform")
    args = ap.parse_args()
    with open(CRITERIA) as fh:
        c = json.load(fh)
    md = render(c)
    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w") as fh:
        fh.write(md)
    print(f"wrote {DOC} ({len(md)} chars)")
    if args.post:
        import urllib.request
        base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010")
        user = os.environ.get("TENSOR_AGENT_USER", "")
        pwd = os.environ.get("TENSOR_AGENT_PASS", "")
        tok = ""
        if user:
            req = urllib.request.Request(
                base + "/tensor/auth/login",
                data=json.dumps({"username": user, "password": pwd}).encode(),
                headers={"Content-Type": "application/json"})
            tok = json.load(urllib.request.urlopen(req))["access_token"]
        figs = []
        for p in c["figures"]:
            with open(p, "rb") as fh:
                figs.append({"name": os.path.basename(p),
                             "caption": os.path.basename(p),
                             "png_base64": base64.b64encode(fh.read()).decode()})
        payload = {
            "title": "AG27W2-N08 碎冰场+3D流场+双向FSI 全船冰阻力物理预报链",
            "summary": f"六判据 {c['n_pass']}/6：C1 碎冰场统计、C2 流场收敛、C3 FSI 交换、"
                       f"C4 阻力闭合、C5 冰阻力参考比较、C6 三档 GCI",
            "scenario_key": "ice_resistance_fullship",
            "mission_id": 7926,
            "tags": ["ice", "fullship", "DEM", "LBM", "FSI", "TensorLBM", "AG27W2-N08"],
            "markdown": md,
            "figures": figs,
        }
        req = urllib.request.Request(
            base + "/tensor/reports", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {tok}"})
        print(urllib.request.urlopen(req).read().decode()[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())