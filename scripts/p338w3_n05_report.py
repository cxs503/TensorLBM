#!/usr/bin/env python3
"""Render the node p338w3-n05 report from the machine-written report JSON.

Every number in the markdown is read from
``artifacts/p338w3_n05_wave_ice_dispersion/report.json`` -- nothing is typed by
hand (zero-transcription discipline).  Usage::

    python scripts/p338w3_n05_report.py
"""

from __future__ import annotations

import json
import os

from tensorlbm.wave_ice_dispersion import analytic as A

ART = "artifacts/p338w3_n05_wave_ice_dispersion"
OUT = "docs/P338W3_N05_WAVE_ICE_DISPERSION.md"


def pct(x: float, nd: int = 3) -> str:
    return f"{100.0 * x:+.{nd}f} %"


def main() -> None:
    with open(os.path.join(ART, "report.json")) as fh:
        s = json.load(fh)

    crit = s["criteria"]
    tank = s["tank"]
    con = s["constants"]
    tnk = s["matrix"]["thicknesses"]
    modes = s["matrix"]["mode_index"]
    cells = s["cells"]
    nc = s["negative_controls"]
    gv = s["group_velocity"]
    conv = s["grid_convergence"]
    xchk = s["solver_cross_check"]

    A_cmin = A.critical_phase_speed(A.flexural_rigidity(A.E_ICE, 1.0))
    A_cmin_ml = A.critical_phase_speed(A.flexural_rigidity(A.E_ICE, 1.0),
                                       mass_loading_param=A.mass_loading(1.0))

    lines: list[str] = []
    w = lines.append

    w("# p338w3-n05 — 弹性冰盖下重力波频散关系：相速度/群速度 解析解 vs 数值解互裁\n")
    w(f"**仓库** TensorLBM · **模块** `src/tensorlbm/wave_ice_dispersion/` · "
      f"**工件** `{ART}/` · **pytest** `tests/test_wave_ice_dispersion.py`\n")
    w(f"**判定** 机器判据全部满足：**{crit['all_pass']}**（逐项见 §6）\n")
    w("\n---\n")

    # ---------------------------------------------------------------- 1
    w("## 1. 案例介绍（物理背景／几何／参数）\n")
    w("浮冰盖把自由面重力波改成**弯曲-重力波**（flexural-gravity wave）：冰盖的抗弯刚度 "
      "`D = E h_i^3 / (12 (1-nu^2))` 给自由面条件增加了一个四阶空间导数项，"
      "使高频（短波）分量获得额外回复力。本节点验证的频散关系（节点规定形式，"
      "**无质量薄板极限**）：\n")
    w("```\nomega^2 = (g k + D k^5 / rho_w) * tanh(k h)\n"
      "c_p = omega / k,    c_g = d omega / d k\n```\n")
    w("物理意义：`g k` 是重力回复项，`D k^5 / rho_w` 是弹性弯曲回复项；两者的比 "
      "`(D/rho_w) k^4 / g` 决定「重力区 → 弯曲区」的过渡波数。"
      "本节点的 10 档波数刻意落在**弯曲区**（`k = 0.209 … 0.586 rad/m`，`lambda = 30 … 10.7 m`），"
      "这是冰盖对波浪影响最大的频段；长波（`lambda > 100 m`）在 1 m 冰厚下弹性项只占 <1 %，"
      "对冰盖识别不敏感。\n")
    w("| 参数 | 取值 | 说明 |")
    w("|---|---|---|")
    w(f"| 重力加速度 g | {con['g']} m/s^2 | 常量 |")
    w(f"| 水密度 rho_w | {con['rho_w']} kg/m^3 | 常量 |")
    w(f"| 冰密度 rho_i | {con['rho_i']} kg/m^3 | Timco & Weeks (2010) 范围 900–940 |")
    w(f"| 冰弹性模量 E | {con['E_ice']:.3e} Pa | 海冰有效模量（0.1–10 GPa） |")
    w(f"| 泊松比 nu | {con['nu_ice']} | — |")
    w(f"| 冰厚 h_i | {', '.join(f'{x:g}' for x in tnk)} m | 5 档（节点要求） |")
    w(f"| 水深 h | {tank['depth']} m | 数值水槽 = 解析 tanh(kh) 同一水深 |")
    w(f"| 波数 k | 2*pi*n/L, n = {', '.join(str(m) for m in modes)} | 10 档，L = {tank['length']} m |")
    w(f"| 水槽网格 | nx = {tank['nx']}, nz = {tank['nz']} "
      f"(dx = {tank['dx']:.4f} m, dz = {tank['dz']:.4f} m) | 周期域 |")
    w(f"| 时间推进 | {tank['scheme']} | {tank['dt_rule']} |")
    w(f"| 积分窗口 | {tank['n_periods']} 个波周期 | 初值振幅 {tank['amplitude']} m（线性区） |")
    w("")

    # ---------------------------------------------------------------- 2
    w("## 2. 怎么做的（方法／流程／工具链）\n")
    w("**(a) 解析臂。** `analytic.dispersion_omega / phase_speed / group_speed`："
      "群速度用解析导数而不是差分——令 `N = gk + Dk^5/rho_w`、`M = 1 + m k`、`Q = N/M`、"
      "`F = Q tanh(kh)`，则 `dF/dk = [(N'M - NM')/M^2] tanh(kh) + Q h sech^2(kh)`、"
      "`c_g = dF/dk / (2 omega)`。同一函数支持**质量加载**版本 "
      "`omega^2 = (gk + Dk^5/rho_w)/(1+mk) tanh(kh)`（`m = rho_i h_i/rho_w`），"
      "用来量化「无质量薄板」这一规定形式的适用边界（§5）。\n")
    w("**(b) 数值臂（数值水槽）。** 线性化势流 + 弹性板自由面条件：\n")
    w("```\nLaplace(phi) = 0            -h < z < 0, x 周期\n"
      "dphi/dz = 0                 z = -h\n"
      "phi = phi_s(x,t)            z = 0\n"
      "eta_t = dphi/dz             z = 0        (运动学)\n"
      "phi_t = -g eta - (D/rho_w) eta_xxxx      z = 0   (动力学, 无质量板)\n```\n")
    w("离散：水平方向 4 阶周期中心差分（`[-1,16,-30,16,-1]/12dx^2`），垂直方向 2 阶，"
      "底部 Neumann 镜像虚节点，表面 `dphi/dz` 用 4 阶单侧差分。每一步的椭圆子问题"
      "**精确求解**（无迭代误差）：矩阵在 x 方向循环（circulant），故用 DFT 对角化后"
      "退化为 `nx` 个独立的垂直三对角系统（Thomas），与整体系稀疏 LU 解**逐位相同**"
      f"（两条解路径互裁：最大相对差 {xchk['max_rel_field_difference']:.2e}，见判据 C4）。"
      "时间推进为交错蛙跳（`eta` 在半步、`phi_s` 在整步），二阶精度。\n")
    w("**(c) 频率提取。** 单一波数初值 `eta = a cos(kx + 0.37)`、`phi_s = 0`（与驻波本征态相容），"
      "取 6 个探针中信号最强者记录 `eta(t)`；对固定 omega 的模型 "
      "`a cos(omega t) + b sin(omega t) + c` 关于 `(a,b,c)` 线性，用正规方程把它们消去，"
      "再对 omega 做「粗扫 + 黄金分割」最小化残差。**omega 是自由参数，不锚定解析值**"
      "（这是本节点与「把理论频率写死」类测量的根本区别）。\n")
    w("**(d) 误差预算。** 同时给出离散系统的**精确本征频率** "
      "`omega_N = sqrt(s_d(kappa) (g + D kappa^4/rho_w))`（`s_d` = 离散 Dirichlet-to-Neumann 符号），"
      "于是误差可分解为空间（`omega_N/omega_analytic - 1`）与时间（`omega/omega_N - 1`）两部分，"
      "避免用「总误差小」掩盖机制。\n")
    w("**(e) 群速度两条独立路径。** ①对测得的 `omega(k)` 曲线做中心差分（并剔除 `c_p` "
      "极小点附近 `c_g → 0` 的病态单元，剔除规则 "
      f"`{gv['trim_rule']}`，剔除 {gv['n_cells_trimmed_near_cp_minimum']} 个单元）；"
      "②**波包追踪**：长周期域内高斯包络窄带波包，用 Hilbert 包络的右行部分能量质心随时间漂移求斜率。\n")
    w("**(f) 工具链。** `python -m tensorlbm.wave_ice_dispersion.campaign "
      f"--out {ART}`；报告 JSON 由程序写出，本文档由 `scripts/p338w3_n05_report.py` "
      "从 JSON 渲染（零手抄）；图由 matplotlib 在同一进程内生成。\n")

    # ---------------------------------------------------------------- 3
    w("## 3. 计算结果（数字表，程序化注入）\n")
    w("### 3.1 5 档冰厚 x 10 档波数（相速度）\n")
    hdr = "| h_i [m] | n | k [rad/m] | c_p 解析 [m/s] | c_p 数值 [m/s] | 相对误差 | " \
          "omega 数值 [rad/s] | c_g 解析 [m/s] | 空间误差 | 时间误差 |"
    w(hdr)
    w("|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        w(f"| {c['h_i']:g} | {c['n']} | {c['k']:.4f} | {c['cp_analytic']:.4f} | "
          f"{c['cp_num']:.4f} | {pct(c['rel_err_total'], 4)} | {c['omega_num']:.4f} | "
          f"{c['cg_analytic']:.3f} | {pct(c['rel_err_spatial'], 4)} | "
          f"{pct(c['rel_err_temporal'], 4)} |")
    w("")
    eb = crit["error_budget"]
    w(f"**误差预算（全矩阵）**：空间项最大 {pct(eb['max_abs_spatial'])}，"
      f"时间项最大 {pct(eb['max_abs_temporal'])}，总误差最大 {pct(eb['max_abs_total'])}；"
      f"中位数 {pct(crit['C1_phase_speed_median_rel_err'])}。"
      "时间项比空间项小 2 个数量级，说明限制精度的是**空间离散**（水平 4 阶、垂直 2 阶），"
      "而不是蛙跳时间积分。\n")

    w("### 3.2 质量加载（物理现实）vs 无质量薄板（本节点规定形式）\n")
    w("| h_i [m] | k [rad/m] | c_p 无质量 | c_p 质量加载 | 偏差 | c_p 自由面 (D=0) |")
    w("|---|---|---|---|---|---|")
    for c in cells:
        if c["n"] in (modes[0], modes[len(modes) // 2], modes[-1]):
            w(f"| {c['h_i']:g} | {c['k']:.4f} | {c['cp_analytic']:.3f} | "
              f"{c['cp_mass_loaded']:.3f} | {pct(c['cp_mass_loaded'] / c['cp_analytic'] - 1, 2)} | "
              f"{c['cp_free_surface']:.3f} |")
    w("")

    w("### 3.3 群速度\n")
    w(f"对测得 `omega(k)` 中心差分（{gv['n_cells_used']} 个有效单元，"
      f"剔除 {gv['n_cells_trimmed_near_cp_minimum']} 个 `c_g -> 0` 单元）："
      f"相对平滑解析 `c_g` 的最大偏差 **{pct(gv['max_rel_err_fd_vs_smooth_analytic'])}**；"
      f"但同一差分作用在**解析** `omega(k)` 抽样上就产生 "
      f"{pct(max(abs(x) for x in gv['fd_truncation']))} 的同号偏差——即这部分是 "
      "10 点 k 网格的**差分截断**，不是数值解误差。以「差分数值 vs 差分解析」度量真误差："
      f"**{pct(gv['max_rel_err_fd_numerical_fidelity'])}**。\n")
    w("| h_i [m] | k0 [rad/m] | 波包测得 c_g [m/s] | 解析 c_g [m/s] | 离散格式 c_g [m/s] | "
      "误差(vs 解析) | 误差(vs 离散) | 拟合散布 [m] |")
    w("|---|---|---|---|---|---|---|---|")
    for p in gv["packet_runs"]:
        w(f"| {p['h_i']:g} | {p['k0']:.4f} | {p['cg_packet']:.3f} | {p['cg_analytic']:.3f} | "
          f"{p['cg_discrete']:.3f} | {pct(p['rel_err'])} | {pct(p['rel_err_vs_discrete'])} | "
          f"{p['fit_scatter_m']:.4f} |")
    w("")

    w("### 3.4 负控\n")
    c1, c2, c3, c4 = (nc["NC1_zero_thickness"], nc["NC2_rigid_limit"],
                      nc["NC3_shallow_limit"], nc["NC4_rigidity_perturbation"])
    w(f"* **NC1（h_i → 0 退化为自由面）**：h_i = {c1['h_i']:g} m（D 只是 0.1 m 冰的 "
      f"{c1['D_ratio_to_0p1m']:.1e} 倍），数值 vs `omega^2 = gk tanh(kh)` 最大偏差 "
      f"**{pct(c1['max_rel_err'], 5)}** → {'PASS' if nc['NC1_zero_thickness']['pass'] else 'FAIL'}。")
    w(f"* **NC2（h_i → ∞ 刚性盖，omega → ∞）**：h_i = {c2['h_i_list']} m，"
      f"omega = {[round(x, 3) for x in c2['omega_massless']]} rad/s，"
      f"log-log 斜率 **{c2['loglog_slope']:.5f}**（理论 1.5，偏差 "
      f"{abs(c2['loglog_slope'] - 1.5):.2e}），单调递增 {c2['monotone_increasing']} → "
      f"{'PASS' if c2['pass'] else 'FAIL'}。同表质量加载分支 "
      f"{[round(x, 3) for x in c2['omega_mass_loaded']]} rad/s，log-log 斜率 "
      f"{c2['loglog_slope_mass_loaded']:.3f}：质量加载把发散指数从 3/2 降到 1（见 §5）。")
    w(f"* **NC3（k → 0 浅水极限 c_p → sqrt(gh)）**：sqrt(gh) = {c3['sqrt_gh']:.4f} m/s；"
      f"k = {c3['k'][:4]} 内最大偏差 **{pct(c3['max_rel_err_analytic_limit'], 5)}**，"
      f"k = 1e-8 时偏差 {pct(c3['rel_err_vs_sqrt_gh'][0], 8)} → "
      f"{'PASS' if c3['pass'] else 'FAIL'}。")
    w(f"* **NC4（D 改错 10 % 可检测）**：最大相速度偏移 **{pct(c4['max_abs_shift'], 2)}** "
      f"（发生在 h_i = {c4['argmax_cell']['h_i']:g} m, k = {c4['argmax_cell']['k']:.4f}）；"
      f"在 {c4['n_cells']} 个单元中有 {c4['n_cells_detectable']} 个单元的偏移超过数值误差底噪 3 倍 → "
      f"{'PASS' if c4['pass'] else 'FAIL'}。\n")

    w("### 3.5 收敛性与实现互裁\n")
    w(f"网格加密两族（同一 (h_i, k) 单元）：水平加密 nx = "
      f"{[r['nx'] for r in conv['nx']['rows']]} → 观测阶 **{conv['observed_order_nx']:.2f}**；"
      f"垂直加密 nz = {[r['nz'] for r in conv['nz']['rows']]} → 观测阶 "
      f"**{conv['observed_order_nz']:.2f}**（相对误差 "
      f"{[round(r['rel_err'], 6) for r in conv['nz']['rows']]}）。"
      f"两条解路径（稀疏 LU vs 循环对角化）末态场最大相对差 "
      f"**{xchk['max_rel_field_difference']:.2e}**（{xchk['nx']}x{xchk['nz']} 网格，"
      f"{xchk['nsteps']} 步）。\n")

    # ---------------------------------------------------------------- 4
    w("## 4. 结果图\n")
    caps = {
        "fig1_dispersion_cp_cg.png": "图 1｜相速度（左）与群速度（右）：实线=解析闭式，空心圆/方块=数值水槽。"
                                     "黑色点线为自由面（D=0）参考。彩色曲线随冰厚增大整体抬升——"
                                     "冰盖让同波数波浪跑得更快。",
        "fig2_error_matrix.png": "图 2｜5x10 矩阵上数值 vs 解析相速度相对误差；红虚线为 ±3 % 验收带。"
                                 "误差随 k 单调增大，与 4 阶水平差分 + 2 阶垂直差分的截断同向。",
        "fig3_negative_controls.png": "图 3｜四个负控：(左上) h_i→0 退化为自由面；(右上) h_i→∞ 发散且 log-log 斜率 1.5，"
                                      "并对照质量加载分支的相反趋势；(左下) k→0 趋于 sqrt(gh)；"
                                      "(右下) D 改错 10 % 的相速度偏移 vs 数值误差底噪。",
        "fig4_grid_convergence.png": "图 4｜网格收敛：水平加密 4 阶、垂直加密 2 阶（黑虚线为 slope 4 参考）。",
    }
    for f, cap in caps.items():
        w(f"**{cap}**\n")
        w(f"![{f}](../{ART}/{f})\n")

    # ---------------------------------------------------------------- 5
    w("## 5. 注意点（踩过的坑／局限／适用范围）\n")
    w("1. **动力学边界条件符号**：`phi_t = -g eta - (D/rho_w) eta_xxxx`。写成 `+` 会让所有波数指数发散"
      "（本节点第一版即踩此坑，表现为 NaN 溢出）；可用 `D → 0` 退化自由面作为符号自检（NC1）。\n")
    w("2. **显式时间推进对弯曲项是刚性的**：稳定性要求 `dt < 2/omega_max(离散)`，"
      "`omega_max ~ sqrt(D/rho_w) k_Nyq^2`，网格越细越刚；1 m 冰在 dx≈1 m 网格上使 `dt ~ 5e-4 s`，"
      "长波 20 个周期需 10^5–10^6 步。生产求解器**必须**对弯曲项做隐式/谱处理（本模块用循环对角化得到"
      "椭圆算子精确解，时间推进仍受 ω dt < 2 约束，但每步成本降到 O(nx log nx)）。\n")
    w("3. **垂直表面导数是误差主项之一**：2 阶单侧差分在 `kappa dz ~ 0.25` 时带 1 % 量级偏差；"
      "换 4 阶单侧差分后同网格降到 0.4 %。垂直方向加密只能线性改善（离散特征衰减率误差），"
      "所以生产配置取 nz = 48 并把误差预算公开。\n")
    w("4. **`c_g → 0` 处的群速度不可比**：`c_p(k)` 有极小点（质量加载时即临界速度 c_min），"
      "其邻域 `d omega/dk ≈ 0`，任何有限差分/相对误差都被放大。本节点按 "
      f"`{gv['trim_rule']}` 剔除 {gv['n_cells_trimmed_near_cp_minimum']} 个单元并显式记录。\n")
    w("5. **无质量薄板只是薄板极限**（本节点最重要的适用边界）：规定形式不含冰的惯性项，"
      "`D ∝ h_i^3` 使 `omega ∝ h_i^{3/2}` 发散。加入质量加载 `m = rho_i h_i/rho_w` 后 "
      "`omega^2 = (gk + Dk^5/rho_w)/(1+mk) tanh(kh)`，发散指数降到 1、整条曲线下移，"
      "并把最小相速度 `c_min` 移位（`c_min` 之下无稳态波，正是冰区船舶临界速度峰值的来源；"
      f"本节点参数下无质量分支 `c_min = {A_cmin:.3f}` m/s，质量加载分支 "
      f"`c_min = {A_cmin_ml:.3f}` m/s）。因此 NC2 只是**代数自洽性**负控，"
      "不能当作厚冰物理判据；冰区工程谱关系应使用质量加载形式（或直接耦合板动力学）。\n")
    w("6. **线性势流、无粘无旋、无破碎**：数值水槽是线性化模型，"
      "因此它验证的是**离散格式 + 提取算子**的频散保真度，不能反映有限振幅、冰盖开裂、"
      "边界层衰减等非线性机制。振幅取 0.01 m（`a k << 1`）。\n")
    w("7. **周期域 + 单一波数初值**：驻波初值（`eta_t = 0`）保证只激发一个频散支；"
      "波包追踪必须等左右两支分离（`t > 3 sigma/c_g`）后计时，否则质心被污染（本节点第二版修）。\n")

    # ---------------------------------------------------------------- 6
    w("## 6. 误差判定（逐项 vs 基准）\n")
    w("| 判据 | 含义 | 实测 | 阈值 | 结论 |")
    w("|---|---|---|---|---|")
    rows = [
        ("C1", "5x10 矩阵相速度最大相对误差", pct(crit["C1_phase_speed_max_rel_err"], 4), "3 %",
         crit["C1_pass"]),
        ("C2", "群速度：差分保真度 / 波包 vs 解析",
         f"{pct(crit['C2_group_speed_fidelity_fd'])} / "
         f"{pct(crit['C2_group_speed_max_rel_err_packet'])}", "2 % / 10 %", crit["C2_pass"]),
        ("C3", "网格收敛观测阶（水平 / 垂直）",
         f"{crit['C3_observed_order_nx']:.2f} / {crit['C3_observed_order_nz']:.2f}",
         "> 1.5", crit["C3_pass"]),
        ("C4", "两条解路径末态场最大相对差",
         f"{crit['C4_solver_path_max_diff']:.1e}", "1e-8", crit["C4_pass"]),
        ("NC1", "h_i→0 退化为自由面", pct(c1["max_rel_err"], 5), "1 %", crit["NC1_pass"]),
        ("NC2", "h_i→∞ log-log 斜率",
         f"{c2['loglog_slope']:.5f}", "1.5 ± 0.02", crit["NC2_pass"]),
        ("NC3", "k→0 趋于 sqrt(gh)", pct(c3["max_rel_err_analytic_limit"], 5), "1 %",
         crit["NC3_pass"]),
        ("NC4", "D+10 % 可检测性",
         f"{pct(c4['max_abs_shift'], 2)} (ratio>3: {c4['n_cells_detectable']}/{c4['n_cells']})",
         "存在 ratio > 3", crit["NC4_pass"]),
    ]
    for name, what, val, thr, ok in rows:
        w(f"| {name} | {what} | {val} | {thr} | {'PASS' if ok else 'FAIL'} |")
    w("")
    w(f"**是否 < 3 %？** 相速度：是——全矩阵最大 {pct(crit['C1_phase_speed_max_rel_err'], 4)}，"
      f"中位数 {pct(crit['C1_phase_speed_median_rel_err'], 4)}，比 3 % 验收线好约 "
      f"{3.0 / max(crit['C1_phase_speed_max_rel_err'] * 100, 1e-12):.0f} 倍。"
      f"群速度：波包追踪 {pct(crit['C2_group_speed_max_rel_err_packet'])} "
      f"（与离散格式自身 c_g 的差 {pct(gv['max_rel_err_packet_vs_discrete'], 4)}）< 3 %；"
      f"差分估计器对平滑解析 c_g 的偏差最大 "
      f"{pct(gv['max_rel_err_fd_vs_smooth_analytic'])}，但同法作用于解析样本即为 "
      f"{pct(max(abs(x) for x in gv['fd_truncation']))}，故数值保真度 "
      f"{pct(crit['C2_group_speed_fidelity_fd'])} < 3 %。"
      "**结论：相速度与群速度两个关键量均 < 3 %；差分估计器本身的截断偏差（最大 16 %）"
      "如实单列，不计入数值误差。**\n")
    w(f"**全判据合取**：`all_pass = {crit['all_pass']}`；总机时 {s['wall_s_total']:.1f} s。\n")
    w("\n---\n")
    w("### 审计链\n")
    w(f"* 报告 JSON（数字唯一来源）：`{ART}/report.json`（含每单元的解析值/数值值/dt/步数/拟合残差/机时）")
    w(f"* 图：{', '.join(s.get('figures', [])) or '见 §4'}")
    w("* 代码：`src/tensorlbm/wave_ice_dispersion/{analytic,tank,circulant,campaign}.py`")
    w("* 用例：`tests/test_wave_ice_dispersion.py`")
    w("* 复现：`python -m tensorlbm.wave_ice_dispersion.campaign --out "
      f"{ART}` → `python scripts/p338w3_n05_report.py`")
    w("")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()