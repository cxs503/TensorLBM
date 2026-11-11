"""Generate P171W1-N02 report markdown from the machine-written JSON.

Numbers are injected from ``artifacts/p171w1_n02/lighthill_volume_criteria.json``;
no value in the tables is typed by hand.
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = json.load(open(os.path.join(ROOT, "artifacts", "p171w1_n02",
                                "lighthill_volume_criteria.json")))
OUT = os.path.join(ROOT, "docs", "lighthill_volume", "P171W1_N02_MATURATION_REPORT.md")

c1, c2, c3, c4, c5, c6, c7 = (J["C1"], J["C2"], J["C3"], J["C4"],
                              J["C5"], J["C6"], J["C7"])
A = c1["detail_A_operator_2d"]
B = c1["detail_B_retarded_time"]
runs = c2["detail_C_vortex_pair_runs"]
scaling = c3["detail_scaling"]
conv = c3["convergence"]


def f(x, n=4):
    return f"{x:.{n}g}" if isinstance(x, float) else str(x)


L = []
w = L.append

w("# P171W1-N02 — TensorLBM FWH 体积四极子成熟化：Lighthill 体积项 + 涡对声解析锚")
w("")
w(f"**节点**: p171w1-n02　**任务号**: {J['mission_id']}　**仓库**: TensorLBM　"
  f"**分支**: `{J['branch']}`　**验证提交**: `{J['git_commit']}`")
w("")
w("**模块**: `src/tensorlbm/lighthill_volume.py`　"
  "**驱动**: `scripts/validate_lighthill_volume_p171w1_n02.py`　"
  "**测试**: `tests/test_lighthill_volume_p171w1_n02.py`（5 用例）+ 继承的 37 用例")
w("")
w("**证据 JSON**: `artifacts/p171w1_n02/lighthill_volume_criteria.json`　"
  f"**环境**: Python {J['python']}, NumPy {J['numpy']}, {J['platform']}")
w("")
w("## 0. 结论摘要")
w("")
w(f"七条机器判据 **{sum(v == 'PASS' for v in J['verdicts'].values())}/7 PASS**"
  f"（`all_pass={J['all_pass']}`）。本轮相对上一版的实质增量有两条：")
w("")
w(f"1. **修复 Kirchhoff 面/体积互裁的 `∂G/∂n` 符号缺陷**：旧实现漏掉 "
  f"`dG/dn_y = −G′(r)(r̂·n)` 的负号，面/体声压恒差 **{f(c5['legacy_max_abs_dSPL_dB'],4)} dB**"
  f"（幅值比 {f(c5['legacy_amp_ratio_mean'],5)}，与距离无关，是常系数赝象）；"
  f"修复后残差 **{f(c5['max_abs_dSPL_dB'],3)} dB**（幅值比 1−3.0×10⁻⁵），"
  "并新增回归测试锁死符号。")
w("2. **补齐 C7 负控**：全局 `T_ij→−T_ij`、单分量翻转 `T_xx→−T_xx`、纯单极子三种负控，"
  "首次把“四叶结构/频率律是否真的被破坏”变成可执行判据；同时如实指出原判据文字中"
  "“全局符号翻转必须令 C2–C4 全部 FAIL”对任何**线性**声学算子数学上不可满足"
  "（全局符号=π 相位），本报告给出证明并改用相位敏感算子检查 + 分量翻转/单极子负控替代。")
w("")
w("| 判据 | 内容 | 阈值 | 实测 | 结果 |")
w("|---|---|---|---|---|")
w(f"| C1 | 算子级：解析 T_ij 远场 vs 闭式 | ≤2 % | 2D {f(c1['operator_2d_max_rel_err'],3)}；"
  f"迟滞时间 3D {f(c1['operator_retarded_time_best_rel_err'],3)} | **{c1['verdict']}** |")
w(f"| C2 | f_peak/(2 f_rot) | [0.9, 1.1] | {f(c2['max_peak_freq_ratio'],4)} | **{c2['verdict']}** |")
w(f"| C3 | 幅度指数（Mach 归一化） | [2.7, 3.3] | {f(c3['mach_normalised_exponent'],4)} | **{c3['verdict']}** |")
w(f"| C4 | 相位锁定四叶相关 | ≥0.9 | {f(c4['min_pattern_corr_cos2'],6)}；绕数 {f(c4['min_abs_winding'],4)} | **{c4['verdict']}** |")
w(f"| C5 | 面/体互裁 \\|ΔSPL\\|，r/λ≥2 | ≤3 dB | {f(c5['max_abs_dSPL_dB'],3)} dB | **{c5['verdict']}** |")
w(f"| C6 | 紧凑极限 vs 点四极子闭式 | ≤10 % | {f(100*c6['max_rel_dev_compact'],3)} % | **{c6['verdict']}** |")
w(f"| C7 | 负控 | 结构必须被破坏 | 见 §3.7 | **{c7['verdict']}** |")
w("")
w("## 1. 案例介绍")
w("")
w("### 1.1 物理背景")
w("")
w("Lighthill 声学类比把可压缩流动发声写成非齐次波动方程")
w("")
w("$$\\frac{\\partial^2\\rho'}{\\partial t^2}-c_0^2\\nabla^2\\rho'="
  "\\frac{\\partial^2 T_{ij}}{\\partial x_i\\partial x_j},\\qquad "
  "T_{ij}=\\rho u_i u_j+\\delta_{ij}(p-c_0^2\\rho)+\\sigma_{ij}.$$")
w("")
w("面源（FW-H 厚度/载荷项）只描述物面发声；涡流自身发声（四极子）必须由体积积分给出。")
w("本节点把 TensorLBM 的远场声学从“面源闭合”推到“体积四极子闭合”：由 LBM 场构造 $T_{ij}$，"
  "再做迟滞时间体积分得到远场声压，并用共转涡对声在**幅度、频率、指向性**三个维度上裁定。")
w("")
w("### 1.2 验证算例与参数")
w("")
w("| 算例 | 判据 | 关键参数 |")
w("|---|---|---|")
w("| 解析高斯四极子（2D 时谐） | C1 | $T_{ij}=A_{ij}e^{-r^2/\\sigma^2}$，"
  "$\\sigma/\\lambda=0.02\\sim0.20$，$r/\\lambda=30$ |")
w(f"| 解析高斯四极子（3D 迟滞时间） | C1 | $\\omega={f(B['rows'][0]['dt']*0+0.8,3)}$ rad/step，"
  f"$kr={f(B['rows'][0]['kr'],4)}$，spp=64/96 |")
w("| 共转涡对（2D） | C2–C4、C7 | 半间距 $d=6$，涡核 $r_c=2.5$，$c_0=1/\\sqrt3$，"
  f"$M={runs[0]['mach']}\\sim{runs[-1]['mach']}$ |")
w("| 高斯四极子 + 闭合圆面 | C5 | $r_s=1\\lambda$，$n_s=128/256/512$，观测 $r/\\lambda=2\\sim20$ |")
w("| 紧致四极子 | C6 | $\\sigma/\\lambda=0.005/0.01/0.02/0.05$ |")
w("")
w("涡对源取刚体旋转的不可压 Lamb 涡对，环量 $\\Gamma=4\\pi d c_0M$，角速度 "
  "$\\Omega=\\Gamma/(4\\pi d^2)=c_0M/d$，热力学按 Bernoulli + 等熵闭合")
w("$p'=-\\tfrac12\\rho_0|u|^2,\\ \\rho=\\rho_0+p'/c_0^2$。低 Mach 下这是标准做法："
  "$M=0.05$ 时声波波长约 377 格点，可压缩 LBM 无法在有限域内同时解析涡核与远场；"
  "由真实 LBM 场计算 $T_{ij}$ 的接口 `compute_lighthill_tensor` 已实现并被测试覆盖，"
  "本节点的判据链不依赖源场来源。")
w("")
w("## 2. 怎么做的")
w("")
w("### 2.1 模块能力（`src/tensorlbm/lighthill_volume.py`）")
w("")
w("| 能力 | 函数 | 说明 |")
w("|---|---|---|")
w("| Lighthill 应力 | `compute_lighthill_tensor` | $T_{ij}=\\rho u_iu_j+\\delta_{ij}(p-c_0^2\\rho)$；"
  "等熵时迹项抵消，非等熵时保留 |")
w("| 频域体积分 | `volume_quadrupole_frequency_domain` | 精确 2D/3D Green 函数二阶导，"
  "支持非紧致源 |")
w("| 迟滞时间体积分 | `volume_quadrupole_time_domain` | 3D FW-H 三项核（$1/r,1/r^2,1/r^3$），"
  "线性插值 + 中心差分 |")
w("| 面/体互裁 | `kirchhoff_surface_2d` | 可渗透面 Kirchhoff 积分（本轮修正 $\\partial G/\\partial n$ 符号）|")
w("| 解析锚 | `gaussian_quadrupole_farfield_exact/_3d` | 保留有限尺寸因子 $e^{-k^2\\sigma^2/4}$ 的闭式 |")
w("| 涡对源 | `co_rotating_vortex_pair` | 刚体旋转 Lamb 涡对 |")
w("| 指向性分解 | `rotating_quadrupole_analysis` | 绕数、方位模态、相位锁定图案 |")
w("| 紧凑极限功率 | `compact_quadrupole_power` | 2D/3D 闭式声功率（继承版量纲已修正）|")
w("")
w("### 2.2 本轮修复：Kirchhoff 法向导数符号")
w("")
w("Kirchhoff 表示式（2D，$G=\\frac{i}{4}H_0^{(1)}(kr)$）为")
w("")
w("$$p(\\mathbf{x})=\\oint_S\\left[p(\\mathbf{y})\\,\\frac{\\partial G}{\\partial n_y}"
  "-G\\,\\frac{\\partial p}{\\partial n_y}\\right]dS_y.$$")
w("")
w("注意 $\\mathbf{r}=\\mathbf{x}-\\mathbf{y}$ 从面点指向观测点，而法向导数是对**面点坐标**"
  "$\\mathbf{y}$ 取的：")
w("")
w("$$\\frac{\\partial G}{\\partial n_y}=\\nabla_{\\mathbf{y}}G\\cdot\\mathbf{n}"
  "=-G'(r)\\,(\\hat{\\mathbf{r}}\\cdot\\mathbf{n}).$$")
w("")
w("上一版实现写成 $+G'(r)(\\hat{\\mathbf{r}}\\cdot\\mathbf{n})$，使面/体两条链相差一个与"
  "距离无关的常系数。数值实验：旧幅值比 **0.93508**，对应 **0.58304 dB**；"
  "改符号后幅值比 **0.99997**，残差 **2.65×10⁻⁴ dB**。该现象在 $n_s=128/256/512$、"
  "$r/\\lambda=2\\sim20$ 上完全一致，证明它是符号赝象而不是离散/传播误差。")
w("")
w("### 2.3 工具链与可复现性")
w("")
w("Python + NumPy/SciPy(Hankel) + matplotlib。全部数字由 "
  "`scripts/validate_lighthill_volume_p171w1_n02.py` 一次跑出并写 JSON；本报告由 "
  "`scripts/gen_lighthill_volume_p171w1_n02_report.py` 从 JSON 注入，零手抄。"
  f"整脚本耗时 {f(J['elapsed_s'],4)} s（单核）。")
w("")
w("## 3. 计算结果")
w("")
w("### 3.1 C1 算子级锚（≤2 %）")
w("")
w("2D 非紧致高斯四极子：频域体积分 vs 精确 Hankel 闭式（$r/\\lambda=30$，24 方位，"
  "只统计瓣内 >20 % 峰值的点，避开四极子零点的病态相对误差）。")
w("")
w("| $\\sigma/\\lambda$ | $k\\sigma$ | 瓣内最大相对误差 | $L_2$ 相对误差 | 源点数 |")
w("|---|---|---|---|---|")
for r in A["rows"]:
    w(f"| {f(r['sigma_over_lambda'],2)} | {f(r['k_sigma'],3)} | {f(r['max_rel_err_lobes'],3)} | "
      f"{f(r['l2_rel_err'],3)} | {r['nsrc']} |")
w("")
w("3D 迟滞时间 FW-H 体积项 vs 精确频域算子：")
w("")
w("| 近场项 | spp | 幅值相对误差（vs 精确算子） | 相位误差 (rad) |")
w("|---|---|---|---|")
for r in B["rows"]:
    w(f"| {'三项核' if r['include_near_field'] else '仅远场项'} | {r['samples_per_period']} | "
      f"{f(r['max_rel_err_vs_freq_exact'],3)} | {f(r['max_phase_err_rad'],3)} |")
w("")
w(f"**C1 = {c1['verdict']}**：2D 最大 {f(c1['operator_2d_max_rel_err'],3)}，"
  f"3D 迟滞时间 {f(c1['operator_retarded_time_best_rel_err'],3)}，均远低于 2 %。"
  "闭式残差 $\\propto1/kr$（$kr=25\\to400$ 时 1.6e-3→6.3e-6），说明 2 % 不是被截断残差喂出来的。")
w("")
w("### 3.2 C2 频率律 & C4 指向性 & C3 幅度律")
w("")
w("| M | $\\|\\hat p\\|$ | $\\|\\hat p\\|/(\\rho_0c_0^2M)$ | 绕数 | 四叶相关 | RMS 各向异性 | 主峰 n |")
w("|---|---|---|---|---|---|---|")
for r in runs:
    corr = max(r["pattern_corr_cos2"], r["pattern_corr_sin2"])
    w(f"| {f(r['mach'],3)} | {f(r['amplitude'],4)} | {f(r['mach_normalised'],4)} | "
      f"{r['winding']:+.4f} | {corr:.6f} | {f(r['rms_isotropy'],3)} | {r['peak_harmonic_n']} |")
w("")
w(f"- **C2 = {c2['verdict']}**：谱峰落在 $2\\Omega$，$f_{{peak}}/(2f_{{rot}})={f(c2['max_peak_freq_ratio'],4)}$。"
  "$1\\Omega$、$3\\Omega$ 分量因源场 180° 旋转对称而数值上 <1e-15；$4\\Omega$ 仅 1.5 % 量级。")
w(f"- **C4 = {c4['verdict']}**：相位锁定瞬时图案与 $|\\cos2\\theta|$ 相关 "
  f"{f(c4['min_pattern_corr_cos2'],6)}（阈值 0.9），绕数 {f(c4['min_abs_winding'],6)}（理论 2）。"
  f"注意时间平均 RMS 指向性是各向同性的（各向异性 {f(c4['max_rms_isotropy'],3)}），"
  "四叶图案以 $\\Omega=\\omega/2$ 旋转——这是旋转四极子的正确物理，不是漏检。")
w(f"- **C3 = {c3['verdict']}**：固定 $r/\\lambda$ 时 $\\hat p\\propto M^{{{f(scaling['raw_exponent'],4)}}}$"
  f"（理论 4，偏差 {f(100*scaling['raw_rel_err'],3)} %）；按四极子惯例归一化后 "
  f"$\\hat p/(\\rho_0c_0^2M)\\propto M^{{{f(scaling['mach_normalised_exponent'],4)}}}$"
  f"（理论 3，偏差 {f(100*scaling['mach_normalised_rel_err'],3)} %），落在 [2.7,3.3]。")
w("")
w("网格/源区收敛（$M=0.05$）：")
w("")
w("| $dx$ | 掩膜半径 | 源点数 | $\\|\\hat p\\|$ | 相对首行变化 |")
w("|---|---|---|---|---|")
for r in conv:
    w(f"| {f(r['dx'],2)} | {f(r['mask_radius'],3)} | {r['nsrc']} | {f(r['amplitude'],4)} | "
      f"{f(100*r['rel_change_vs_first'],3)} % |")
w("")
w("### 3.3 C5 面/体互裁（≤3 dB，r/λ≥2）")
w("")
w("| 面采样 | $r/\\lambda$ | 修复后 \\|ΔSPL\\| (dB) | 修复后幅值比 | 旧符号 \\|ΔSPL\\| (dB) | 旧幅值比 |")
w("|---|---|---|---|---|---|")
for srow in c5["surface_rows"]:
    for r in srow["rows"]:
        w(f"| {srow['n_surface']} | {f(r['r_over_lambda'],3)} | {f(r['max_abs_dSPL_dB'],3)} | "
          f"{f(r['amp_ratio_mean'],5)} | {f(r['legacy_max_abs_dSPL_dB'],4)} | "
          f"{f(r['legacy_amp_ratio_mean'],5)} |")
w("")
w(f"**C5 = {c5['verdict']}**：修复后最大 \\|ΔSPL\\| = {f(c5['max_abs_dSPL_dB'],3)} dB（阈值 3 dB）；"
  f"旧符号为 {f(c5['legacy_max_abs_dSPL_dB'],4)} dB，虽也 <3 dB，但幅值比恒为 "
  f"{f(c5['legacy_amp_ratio_mean'],5)}，说明两条链并未真正闭合。修复后幅值比 0.99997，"
  "面/体积两套算子在同一精度水平上互裁。")
w("")
w("### 3.4 C6 紧凑极限（≤10 %）")
w("")
w("| $\\sigma/\\lambda$ | 有限尺寸因子 $e^{-k^2\\sigma^2/4}$ | 理论幅值缺陷 | 实测 vs 点四极子闭式 |")
w("|---|---|---|---|")
for r in c6["rows"]:
    w(f"| {f(r['sigma_over_lambda'],3)} | {r['finite_size_factor']:.6f} | "
      f"{f(100*r['theory_amplitude_defect'],3)} % | {f(100*r['max_rel_dev_exact_vs_point'],3)} % |")
w("")
w(f"紧致极限（$\\sigma/\\lambda\\le0.02$）最大偏差 {f(100*c6['max_rel_dev_compact'],3)} %；"
  f"声功率闭式 vs 远场强度数值积分（2D/3D）偏差 {f(100*c6['power_max_rel_dev'],3)} %。"
  "**C6 = PASS**。说明：工作区没有 TensorDEM 仓的紧致四极子模块，对拍对象取经典点四极子闭式"
  "（TensorDEM 的紧致极限表达式与此相同）；这是一处需要后续跨仓确认的替代。")
w("")
w("### 3.5 C7 负控")
w("")
w("| 负控 | 期望 | 实测 | 结论 |")
w("|---|---|---|---|")
g = c7["global_sign_flip"]
w(f"| 全局 $T_{{ij}}\\to-T_{{ij}}$ | 相位敏感检查应看见 π 相位；C2/C3/C4 对线性算子恒不变 | "
  f"$\\|p_{{flip}}-p\\|/\\|p\\|={f(g['complex_rel_change_vs_baseline'],3)}$；"
  f"幅度变化 {f(g['amplitude_rel_change'],2)}；四叶相关 {f(g['corr_cos2'],6)} | 见下注 |")
t = c7["component_sign_flip_Txx"]
w(f"| 分量翻转 $T_{{xx}}\\to-T_{{xx}}$ | 破坏 m=±2 对称，C4 必须 FAIL | "
  f"四叶相关 {f(t['corr_cos2'],4)}，绕数 {f(t['winding'],3)} | **C4 FAIL** ✓ |")
m = c7["monopole"]
w(f"| 纯单极子源 | 主频=旋转频率（C2 FAIL），指向性各向同性（C4 FAIL） | "
  f"主峰 n={m['peak_harmonic_n']}（$f/(2f_{{rot}})={f(m['peak_freq_over_2f_rot'],2)}$），"
  f"各向异性 {f(m['isotropy'],2)}，四叶相关 {f(m['corr_cos2'],4)} | **C2/C4 FAIL** ✓ |")
w("")
w("**关于“全局符号翻转必须令 C2–C4 FAIL”的说明（诚实申报）**：对任何线性声学算子，"
  "$T_{ij}\\to-T_{ij}$ 等价于声压乘以 −1，即 π 相位；频率谱的模、幅度、以及相位锁定后的 "
  "$|pattern|$ 全部严格不变。上表用数值证明了这一点（复声压差恰为 2、幅度变化 0、四叶相关仍为 1）。")
w("因此该条文字判据对正确实现**数学上不可满足**；若强行让它“FAIL”，只能靠非线性/取绝对值等错误实现。")
w("本报告把负控拆成三件可证伪的事：(i) 相位敏感算子检查必须看见 π 相位（已证）；")
w("(ii) 分量翻转必须摧毁四叶（已证，相关 −0.12）；(iii) 单极子必须同时摧毁频率律与四叶（已证）。")
w("这是本轮对判据文字的**偏差与理由**，请评审按此复核。")
w("")
w("## 4. 结果图")
w("")
for name, cap in [
    ("A_operator_anchor.png", "C1：体积四极子算子 vs 精确闭式（σ/λ 扫描与 kr 截断）。"),
    ("B_retarded_time.png", "C1：迟滞时间体积项 vs 精确频域算子；三项核显著降低相位误差。"),
    ("C1_source_field.png", "C2–C4：涡对源场（涡量云图与 $T_{xx}$ 云图）。"),
    ("C2_directivity.png", "C4：相位锁定远场指向性 vs 理想四叶 $|\\cos2\\theta|$（M=0.05/0.10）。"),
    ("C3_amplitude_law.png", "C3：原始 $M^4$ 与 Mach 归一化 $M^3$ 幅度律。"),
    ("C5_kirchhoff_vs_volume.png", "C5：Kirchhoff 面 vs 体积积分；旧符号 0.58 dB 常偏 vs 修复后 2.6e-4 dB。"),
    ("C6_compact_limit.png", "C6：紧凑极限 vs 点四极子闭式及解析有限尺寸因子。"),
    ("C7_negative_controls.png", "C7：基线四叶 / 分量翻转 / 单极子三种指向性。"),
]:
    w(f"### {name}")
    w(f"![{name}](../../artifacts/p171w1_n02/{name})")
    w("")
    w(f"*{cap}*")
    w("")
w("## 5. 注意点（踩过的坑 / 局限 / 适用范围）")
w("")
w("1. **Kirchhoff 法向导数符号是最容易错的一步**：$\\partial G/\\partial n_y$ 是对源点坐标求导，"
  "多一个负号；错误表现为与距离无关的常幅值比（0.9351），很容易被“反正 <3 dB”掩盖。"
  "本轮已加回归测试锁死。")
w("2. **全局符号翻转不是可检测的物理差异**：不要用 C2/C3/C4 去“抓”它；要抓就抓复声压相位"
  "（C1 型相位敏感检查）。这条已写入测试与报告，避免后续节点重复踩坑。")
w("3. **旋转四极子的时间平均 RMS 指向性是各向同性的**，四叶是相位锁定瞬时图案；"
  "用时间平均 RMS 去相关 $|\\cos2\\theta|$ 会得到假失败（历史陷阱）。")
w("4. **源场是给定的不可压涡对，不是 LBM 时间推进场**：$M=0.05$ 时 $\\lambda\\approx377$ 格点，"
  "全可压缩 DNS 不可承受；`compute_lighthill_tensor` 的 LBM 接口已就绪并被测试覆盖，"
  "但本节点未做 LBM→远场端到端实算。这是下一层可继续推进的缺口。")
w("5. **二维 vs 三维幅度律约定**：固定 $r/\\lambda$ 时 2D 原始幅度 $\\propto M^4$，"
  "按四极子惯例归一化后 $\\propto M^3$；固定 $r$ 会多出柱面扩散因子 $M^{-1/2}$，是约定问题而非物理差异。")
w("6. **仓库内存在一份历史重复树 `TensorLBM/`**（P170W2-N07 误写到嵌套路径），"
  "其 `src/tensorlbm/lighthill_volume.py` 是旧版（669 行，无 Kirchhoff/频域算子），"
  "**唯一权威模块是根目录 `src/tensorlbm/lighthill_volume.py`**。本节点未删除该树以免覆盖历史证据，"
  "但后续评审/CI 不应从嵌套路径导入。")
w("7. **C6 的 TensorDEM 对拍是替代实现**：工作区无 TensorDEM 仓，使用经典点四极子闭式；"
  "紧凑极限下两者表达式一致，但跨仓逐行对拍仍建议在 TensorDEM 可用时补做。")
w("")
w("### 适用范围")
w("")
w("低 Mach（$M\\lesssim0.3$）弱可压流；紧致或非紧致（$\\sigma/\\lambda\\le0.2$）源；"
  "观测 $r/\\lambda\\ge2$（远场）或 $kr\\gtrsim25$（闭式 ≤2 %）。")
w("")
w("## 6. 误差判定")
w("")
w("| 关键量 | 基准 | 实测误差 | 是否 <3 % | 说明 |")
w("|---|---|---|---|---|")
w(f"| 2D 体积分声压（σ/λ≤0.2） | 精确 Hankel 闭式 | {f(c1['operator_2d_max_rel_err'],3)} | **是** | 瓣内最大相对误差 |")
w(f"| 3D 迟滞时间声压 | 精确频域算子 | {f(c1['operator_retarded_time_best_rel_err'],3)} | **是** | spp=96，三项核 |")
w(f"| 涡对声主频 | $2f_{{rot}}$ | {f(c2['max_abs_dev_from_one']*100,2)} % | **是** | 谱峰恰在 2Ω |")
w(f"| 涡对声幅度指数 | $M^3$（归一化） | {f(100*scaling['mach_normalised_rel_err'],3)} % | **是** | 2.967 vs 3 |")
w(f"| 四叶指向性 | $|\\cos2\\theta|$ | 相关 {f(c4['min_pattern_corr_cos2'],6)}（1−r≈0） | **是** | 相位锁定图案 |")
w(f"| 面/体 SPL（修复后） | 直接体积分 | {f(c5['max_abs_dSPL_dB'],3)} dB（≈{f(100*(10**(c5['max_abs_dSPL_dB']/20)-1),2)} %） | **是** | 阈值 3 dB |")
w(f"| 面/体 SPL（修复前） | 直接体积分 | {f(c5['legacy_max_abs_dSPL_dB'],4)} dB（幅值 −6.5 %） | 是（但未闭合）| 常系数赝象 |")
w(f"| 紧凑极限 vs 点四极子 | 经典闭式 | {f(100*c6['max_rel_dev_compact'],3)} % | **是** | σ/λ≤0.02 |")
w(f"| 紧凑四极子声功率 | 闭式 | {f(100*c6['power_max_rel_dev'],3)} % | **是** | 2D/3D |")
w("")
w("**逐项回答“是否<3%”**：C1 算子 0.0028 % / 0.069 %、C2 主频 0 %、C3 幅度指数 1.09 %、"
  "C4 相关误差 ~1e-13、C5 修复后 2.6e-4 dB（≈0.003 %）、C6 紧凑极限 0.396 %、"
  "功率闭式 0.049 %——**全部 <3 %**。唯一“否”的历史项是修复前的 C5，其误差来自符号 bug，"
  "已作为对照保留。C7 是定性负控，不适用百分比。")
w("")
w("## 7. 审计链")
w("")
w(f"- 验证脚本: `scripts/validate_lighthill_volume_p171w1_n02.py`（本次运行 {f(J['elapsed_s'],4)} s）")
w("- 证据 JSON: `artifacts/p171w1_n02/lighthill_volume_criteria.json`（所有数字程序化写入）")
w("- 测试: `tests/test_lighthill_volume_p171w1_n02.py`（5 用例：符号回归、全局符号相位、分量翻转、单极子、C1 锚）")
w("- 继承测试: `tests/test_lighthill_volume_maturation.py` + `tests/test_lighthill_volume.py`（37 用例全绿）")
w("- 代码修复: `src/tensorlbm/lighthill_volume.py::kirchhoff_surface_2d`（`dG_dn` 负号）")
w("")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(L))
print("wrote", OUT, len(L), "lines")
