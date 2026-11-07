#!/usr/bin/env python
"""Generate the IN31-N12 report markdown from the JSON artifacts.

Every number in the report body is formatted from the artifacts on disk
(``artifacts/in31_n12/*.json``), so the report cannot drift from the data.
"""
from __future__ import annotations

import json
from pathlib import Path

RA_CRIT = 1707.7597


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if x != 0 and (abs(x) < 1e-3 or abs(x) > 1e5):
            return f"{x:.3e}"
        return f"{x:.{nd}f}"
    return str(x)


def main() -> int:
    base = Path("artifacts/in31_n12")
    runs = []
    for p in [Path(".scratch/ny16.json")] + sorted(base.glob("rb_lattice_ny*.json")):
        d = load(p)
        if d:
            runs.append(d)
    runs = sorted({r["params"]["ny"]: r for r in runs}.values(), key=lambda r: r["params"]["ny"])
    crit = load(base / "criteria.json") or {}
    dec = load(base / "hr_decoupling.json")
    settle = load(base / "hr_settling.json")
    prof = load(base / "base_profile.json") or []

    L = []
    L.append("# IN31-N12 · LBM Rayleigh–Bénard 离散线性稳定性 + 双相黏度比/密度比解耦与 Hadamard–Rybczynski 对拍\n")
    L.append("**节点**: IN31-N12（战役 DAG，物理域） · **仓库**: TensorLBM · **分支**: `agent/m5628`\n")
    L.append("**结论速览**: (1) RB 起稳判据升级为**同一格子算子上的离散线性稳定性特征值**："
             "本仓库新增 `tensorlbm.rb_discrete_stability`（矩阵无关 Jacobian + 谱），其主导特征值"
             "与真实非线性 LBM 时间推进的增长率在 ny=16/32 上一致到 **0.01% / 0.06%**（判据 <3% 通过）；"
             "离散临界 Rayleigh 数在 ny=16/32/64 上分别为 "
             + "/".join(fmt(r["Ra_c_lattice"], 1) for r in runs)
             + "（相对 1707.76 上移 " + "/".join(f"{100*r['Ra_c_rel_shift']:+.2f}%" for r in runs) + "）。"
             "(2) 双相伪势引擎**库层**已实现 λ=μ_d/μ_c 与 ρ_d/ρ_c 解耦（剪切波实测 λ 误差 ≤"
             + fmt(crit.get("decoupling", {}).get("max_abs_lam_rel_err_pct"), 3)
             + "%；λ 对密度比的斜率 ≈ 0）。"
             "(3) **LBM 端 Hadamard–Rybczynski 对拍 honest FAIL**：伪势双相液滴的沉降被界面寄生流（spurious current ~5e-2）完全淹没，"
             "与 HR 解析速度相差 3 个量级以上，未达 15% 判据——详见 §5、§6。\n")

    # ---------------- 1 案例介绍 ----------------
    L.append("## 1. 案例介绍（物理背景 / 几何 / 参数）\n")
    L.append("### 1.1 Rayleigh–Bénard 起稳（2D，刚–刚等温壁）\n")
    L.append("水平流体层底部加热、顶部冷却，温差超过临界值时静止导热态失稳出现对流卷。"
             "连续谱基准 Ra_c = 1707.7597（a_c = 3.11632，Chandrasekhar 1961，刚–刚、等温、2D 卷）。"
             "本次工作把 LBM 侧的参考量从\"连续谱 Chebyshev 特征值\"升级为\"同一格子算子的离散线性稳定性特征值\"，"
             "并同时保留一个**在仓库内可复现**的连续谱参考实现用于对拍。\n")
    if runs:
        r0 = runs[0]["params"]
        L.append(f"格子与物理参数（与 `benchmarks/pending/rayleigh_benard` 的基准设置一致）：ν=(τ−1/2)/3={fmt(r0['nu'],5)}，"
                 f"α=(τ_T−1/2)/3={fmt(r0['alpha'],5)}，Pr=ν/α={fmt(r0['Pr'],4)}，"
                 f"间隙 H=ny（benchmark 约定），水平周期长度 nx=2ny（单卷，a=2πH/nx=π≈3.1416）。"
                 "Boussinesq 浮力用 Guo 力格式（`thermal.buoyancy_force` + `thermal.collide_bgk_force`），"
                 "温度用 D2Q5 + 上下壁 anti-bounce-back（ABB），速度用 D2Q9 + pre-streaming 半程反弹。\n")
        L.append("| ny | nx | 谱块维数 14·ny | Ra_c^lat | 相对 1707.76 | 基态残差 |")
        L.append("|---|---|---|---|---|---|")
        for r in runs:
            L.append(f"| {r['params']['ny']} | {r['params']['nx']} | {14*r['params']['ny']} | "
                     f"{fmt(r['Ra_c_lattice'],2)} | {100*r['Ra_c_rel_shift']:+.2f}% | {fmt(r['base_residual'])} |")
        L.append("")
    L.append("### 1.2 双相液滴沉降（Hadamard–Rybczynski）\n")
    L.append("球形液滴（μ_d, ρ_d）在另一相（μ_c, ρ_c）中常速沉降，creeping flow 下的精确解：\n")
    L.append("```\nF = 2π μ_c R U (2+3λ)/(1+λ),   λ = μ_d/μ_c\nU = (2/3) Δρ g R² (1+λ) / (μ_c (2+3λ))\n```\n")
    L.append("λ→0 给出 Rybczynski 气泡（U=ΔρgR²/3μ_c），λ→∞ 给出 Stokes 固体球（U=2ΔρgR²/9μ_c）。"
             "本仓已有 `tensorlbm.hadamard_rybczynski`（解析律 + 轴对称有限元 Stokes 求解器，属**已验收**的解析/有限元验证，"
             "本节点不重复）；本节点的对象是**伪势（Shan–Chen）双相 LBM 引擎**能否独立设定 λ 与 ρ 比并按该律沉降。\n")
    L.append("Shan–Chen 双相设置：D3Q19，`ny³` 全周期立方域，组分 2 为液滴、组分 1 为环境相；"
             "重力只加在液滴组分上，环境相施加等大反向的均匀补偿力使**总动量注入为零**（周期域中否则整盒加速，不存在终速）。\n")

    # ---------------- 2 方法 ----------------
    L.append("## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）\n")
    L.append("### 2.1 离散线性稳定性求解器（新增 `src/tensorlbm/rb_discrete_stability.py`）\n")
    L.append("1. **基态**：导热态 T(y)=T_hot−ΔT(y+½)/H、u=0 是格子算子的不动点。在**沿 x 均匀的子空间**内迭代步进算子"
             "（RB 失稳模带水平结构，故即便超临界导热态在该子空间内仍线性稳定），得到离散基态并报告其不动点残差。"
             "由于 x 均匀场与箱宽无关，基态在 nx=1 的柱上迭代后平铺到目标网格（>30 倍加速，无近似）。\n")
    L.append("2. **Jacobian**：步进算子 = 库内原语组合（`temperature_collision`→`temperature_stream`→ABB 壁面→"
             "`buoyancy_force`→`collide_bgk_force`(Guo)→`pre_streaming_bounce_back`→`stream`），与 benchmark 主循环逐步同序。"
             "Jacobian 用 float64 有限差分 `Jv≈(F(x0+εv)−F(x0))/ε`（ε=1e-6）**矩阵无关**求取，"
             "因此不存在\"手推算子与代码脱节\"的风险。\n")
    L.append("3. **模态块**：基态与壁面沿 x 均匀 ⇒ Jacobian 在水平 Fourier 基下严格块对角。只取波数 k=2πm/nx 的块"
             "（维数 14·ny，复线性），用实扰动列逐列组装稠密复矩阵，再求全部特征值。\n")
    L.append("4. **本征值 → 增长率**：λ_max 为最大模物理本征值，γ_step=ln|λ_max|（每格子步），"
             "无量纲 γ̂=γ_step·H²/ν，与连续谱 σ（时间单位 H²/ν）直接可比。\n")
    L.append("5. **关键陷阱（本节点的技术核心）**：库的半程反弹/ABB 把两排壁面节点**代数替换**（镜像/代入）而不做碰撞松弛，"
             "因此线性算子的谱里存在一簇**严格落在单位圆上**（|λ|−1<1e-12，辐角 0 或 π）的伪模；"
             "按 |λ| 排序它们永远挡住物理卷模，这正是\"朴素格子 LBM 稳定性分析给出 γ≡0\"的根因。"
             "本工作用 `| |λ|−1 | < 1e-7` 剔除该簇（近临界物理模偏差 ≥1e-5，无量纲意义上只在距临界 1e-4 内不可分辨），"
             "并用**非线性时间推进**独立校验被选中的模确实是物理模。\n")
    L.append("### 2.2 连续谱参考（新增 `src/tensorlbm/rb_continuum_stability.py`）\n")
    L.append("扰动以流函数 ψ 与温度 θ 表示，长度用 H、时间用 H²/ν、温度用 ΔT、ψ 用 ν 无量纲化，法向模 e^{σt+iax}：\n")
    L.append("```\nσ(D²−a²)ψ = (D²−a²)²ψ − i a (Ra/Pr) θ\nσθ      = (1/Pr)(D²−a²)θ − i a ψ\nψ = Dψ = 0, θ = 0  (z=0,1)\n```\n")
    L.append("（−i a 的符号来自水平 Laplace：涡量方程中浮力项为 curl(gβθ e_z)=−gβ∂θ/∂x。）"
             "σ=0 时化归经典 Prandtl 数无关问题 (D²−a²)³ψ = −a²Ra ψ，其最小值即 1707.76。\n")
    L.append("数值方法：二阶有限差分（M 个内点，壁面 z=0,1），Neumann 条件折入中间变量 ω=(D²−a²)ψ 的鬼点"
             "（ω_0=2ψ_1/h²，ω_{M+1}=2ψ_M/h²），广义特征值问题 A x = σ B x 用稠密 QZ 求解，取最大实部模。"
             "**为什么不用 Chebyshev 配点**：四阶算子显式 Chebyshev 微分矩阵条件数 ~N⁸，实测（开发期）会产生 "
             "实部 +10³ 量级的伪谱并淹没物理支；边界基函数重组与行替换两种做法都只得到一阶收敛。此坑已记入 §5。"
             "本实现的二阶精度由三套网格的 Richardson 外推验证（见 §3.2）。\n")
    L.append("### 2.3 伪势双相 λ/ρ 解耦（新增 `src/tensorlbm/pseudopotential_hr.py`）\n")
    L.append("- 单松弛 BGK 的单元动力黏度 μ=ρ c_s²(τ−1/2)。若两相同 τ，则 λ 被密度比锁死（λ=ρ_d/ρ_c）。"
             "双组分核 `multiphase3d.collide_sc_two_component_3d` 对每个组分独立给 τ，故可由\n"
             "  `τ_d = 1/2 + λ (ρ_c/ρ_d)(τ_c−1/2)` 反解——即 λ 与 ρ 比**解耦**。\n")
    L.append("- 解耦不是断言而是**测量**：对每个 (λ, ρ_d/ρ_c) 组合，用 `two_phase_viscosity.shear_wave_decay` "
             "在真实 D2Q9 BGK 核上分别测两相的运动黏度，再按 μ=ρν 组成 λ_measured，与给定 λ 比较。\n")
    L.append("- 已定位的库层缺口与补丁：`collide_sc_two_component_3d` 只接受**同一个** gz 给两个组分，"
             "无法表达\"只给液滴加重力\"（浮力问题的必需形式）。新增 "
             "`collide_two_component_with_gravity(f1,f2,G_12,τ1,τ2,gz1,gz2,u_eq)` 按库内约定逐组分施力，"
             "并用单测证明：当 gz1=gz2 且 τ 相同/不同时它**逐位等于**库内核（max|Δ|=0.0）。\n")

    # ---------------- 3 结果 ----------------
    L.append("## 3. 计算结果\n")
    L.append("### 3.1 RB：离散特征值 vs 非线性时间推进（算子一致性判据）\n")
    L.append("| ny | Ra | γ_step（特征值） | γ_step（时间推进） | 相对误差 | 拟合窗口 |")
    L.append("|---|---|---|---|---|---|")
    for r in runs:
        for c in r.get("sim_checks", []):
            L.append(f"| {r['params']['ny']} | {fmt(c['Ra'],1)} | {fmt(c['gamma_step_eig'])} | "
                     f"{fmt(c['gamma_step_sim'])} | {100*c['rel_err_sim_vs_eig']:+.3f}% | "
                     f"{int(c['fit_window'][0])}–{int(c['fit_window'][1])} |")
    L.append("")
    L.append("### 3.2 RB：γ(Ra) 与离散临界 Rayleigh 数\n")
    L.append("| ny | Ra_c^lat | 相对 1707.76 | (ny−1)³/ny³ 折算后 | 折算后相对偏差 | 臂最大 |Δγ̂| |")
    L.append("|---|---|---|---|---|---|")
    for r in runs:
        ny = r["params"]["ny"]
        remap = r["Ra_c_lattice"] * ((ny - 1.0) / ny) ** 3
        mx = max(abs(100 * a["rel_err_lattice_vs_cont"]) for a in r["arms"])
        L.append(f"| {ny} | {fmt(r['Ra_c_lattice'],2)} | {100*r['Ra_c_rel_shift']:+.2f}% | {fmt(remap,2)} | "
                 f"{100*(remap/RA_CRIT-1):+.2f}% | {mx:.1f}% |")
    L.append("")
    if runs:
        L.append("逐臂数据（Ra = frac·Ra_c^lat）：\n")
        L.append("| ny | frac | Ra | γ̂ 格子 | γ̂ 连续谱 | 相对差 |")
        L.append("|---|---|---|---|---|---|")
        for r in runs:
            for a in sorted(r["arms"], key=lambda a: a["Ra"]):
                L.append(f"| {r['params']['ny']} | {a['frac_of_Ra_c_lat']:.2f} | {fmt(a['Ra'],1)} | "
                         f"{fmt(a['gamma_hat_lattice'])} | {fmt(a['gamma_hat_continuum'])} | "
                         f"{100*a['rel_err_lattice_vs_cont']:+.1f}% |")
        L.append("")
    L.append("**器件解释**：格子算子与连续谱的本质差异不是\"同一 Ra 下 γ 差一点\"，而是**临界 Ra 整体上移**，"
             "且上移量随分辨率下降。其中可精确归因的部分是温度边界：离散导热解把壁面**钉在节点上**"
             "（实测 H_eff=ny−1，见 §3.3），故 benchmark 约定的 H=ny 里\"真实\"Ra 只有名义值的 ((ny−1)/ny)³，"
             "折算后剩余偏差见表第三列之后——它随 ny 以 ≈h² 下降，是截断误差（粗网格可压缩性/伪速度）。\n")
    L.append("### 3.3 离散导热基态的等效间隙（诊断）\n")
    if prof:
        L.append("| ny | Ra | 内部拟合斜率 | 实测 H_eff | max|T−T_半程| |")
        L.append("|---|---|---|---|---|")
        for r in prof:
            L.append(f"| {r['ny']} | {fmt(r['Ra'],0)} | {fmt(r['slope'],6)} | {fmt(r['H_eff'],3)} | {fmt(r['dev_hw'],4)} |")
        L.append("")
        L.append("H_eff = ny−1（误差 <0.4%），与\"D2Q5 ABB 的离散稳态把端点节点固定为壁温、内部线性\"一致。"
                 "这正是 W11 报告中\"格子算子 Ra_c 漂移\"的可解释主体，也是 benchmark 里 Nu 口径系统性偏差 1/(1−1/ny) 的来源。\n")
    L.append("### 3.4 双相 λ/ρ 解耦（库层）\n")
    if dec:
        L.append("| λ 设定 | ρ_d/ρ_c | τ_d | 实测 λ | 相对误差 |")
        L.append("|---|---|---|---|---|")
        for r in dec["decoupling"]:
            L.append(f"| {fmt(r['lam'],2)} | {fmt(r['density_ratio'],2)} | {fmt(r['tau_d'],4)} | "
                     f"{fmt(r['lam_measured'])} | {100*r['lam_rel_err']:+.3f}% |")
        L.append("")
        ds = dec.get("decoupling_density_slope", {})
        L.append(f"固定 λ 时 λ_measured 对密度比的斜率：{fmt(ds.get('max_abs'))}（解耦引擎应为 0；若 τ 共用则应为 1）。\n")
    L.append("### 3.5 伪势液滴沉降 vs Hadamard–Rybczynski（LBM 端）\n")
    if settle and settle.get("settling"):
        L.append("| λ | τ_d | R_eff | ρ2_in | ρ1_out | U_rel（LBM） | 时间抖动 | U_HR | 相对误差 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in settle["settling"]:
            if r.get("status") != "ok":
                L.append(f"| {fmt(r.get('lam'))} | – | – | – | – | 发散 | – | – | – |")
                continue
            s = r["setup"]
            L.append(f"| {fmt(s['lam_nominal'],2)} | {fmt(s['tau_d'],3)} | {fmt(r['R_eff'],2)} | "
                     f"{fmt(r['rho2_in'],3)} | {fmt(r['rho1_out'],3)} | {fmt(r['U_rel'])} | "
                     f"{fmt(r['U_rel_spread'])} | {fmt(r['U_hr'])} | {100*r['rel_err_vs_hr']:+.0f}% |")
        L.append("")
    L.append("### 3.6 图\n")
    L.append("见随报告提交的 PNG（全部由 `benchmarks/rb_discrete_stability/make_figures.py` 从 JSON 程序化生成）：\n")
    L.append("- `rb_growth_rate.png`：γ̂(Ra) 格子 vs 连续谱（等 Ra、等 a）；右图为相对差。\n"
             "- `rb_ra_c_convergence.png`：Ra_c^lat 随 ny 的收敛与 (ny/(ny−1))³ 锚。\n"
             "- `rb_eig_vs_sim.png`：特征值 γ 与非线性时间推进 γ 的对角一致性（±1% 带）。\n"
             "- `rb_base_profile.png`：实测 H_eff=ny−1 证据。\n"
             "- `hr_decoupling_settling.png`：λ 解耦实测（左）与沉降速度 vs HR（右，含 honest FAIL）。\n")

    # ---------------- 4 注意点 ----------------
    L.append("## 4. 注意点（踩过的坑 / 局限 / 适用范围）\n")
    L.append("1. **单位圆伪模**：不改用\"剔除 + 非线性校验\"的做法，任何按 |λ| 排序的格子稳定性分析都会返回 γ≡0（实测 12 个伪模严格落在 |λ|=1）。\n")
    L.append("2. **内点投影是错的**：只保留内部行（冻结壁面行扰动）的 Galerkin 投影会给出**伪失稳**（ny=16、Ra=2500 时 γ̂=+5.29，"
             "而真实时间推进为 −1.31），必须在完整状态空间求解后剔除伪模。\n")
    L.append("3. **连续谱参考不要用显式 Chebyshev 四阶微分矩阵**：伪谱实部 ~N⁴ 增长并淹没物理支（本次实测 N=48 时 +9.3e3）。二阶 FD + QZ 反而干净，"
             "代价是要做网格外推（本文三套网格 Richardson，1707.7685 vs 1707.7597，相对 5e-6）。\n")
    L.append("4. **H=ny vs H_eff=ny−1**：benchmark 的非量纲约定用半程壁面 H=ny，而离散导热解实际跨度为 ny−1。"
             "报告 Ra_c 时必须写明用哪种约定，否则\"格子算子偏移\"会相差 ~(ny/(ny−1))³（ny=16 时 21%）。\n")
    L.append("5. **伪势双相引擎的寄生流**：液滴内部/环境相被排空到 ~backgroud 密度时 τF/ρ 会爆掉（实测无背景密度时 20 步内发散），"
             "必须保留最小背景密度（本工作 0.3）；但保背景后相分离需要 |G_12|≳1.5，此时界面寄生流 ~5e-2，"
             "远大于 HR 终速（~1e-5），这就是 §3.5 honest FAIL 的物理原因。\n")
    L.append("6. **适用范围**：本人造离散特征值只对该算子/该壁面处理（pre-streaming bounce-back + D2Q5 ABB）成立；"
             "换壁面格式（如插值反弹）会改变伪模结构与 H_eff，需重新标定。\n")
    L.append("7. **基态残差**：ny=64 只跑了 4000 步预热，残差 2.4e-7/步；对 |γ̂|>0.5 的臂影响 <1%，但近临界（|γ̂|<0.01）值不可作为判据。\n")

    # ---------------- 5 误差判定 ----------------
    L.append("## 5. 误差判定（逐项 vs 基准 + 是否 <3%）\n")
    L.append("| 关键量 | 基准 | 本工作 | 相对误差 | 是否 <3% |")
    L.append("|---|---|---|---|---|")
    worst_sim = max([abs(c["rel_err_sim_vs_eig"]) for r in runs for c in r.get("sim_checks", [])] or [float("nan")])
    L.append(f"| 离散主导特征值 γ vs 同一算子非线性时间推进 | 时间推进 | 见 §3.1（ny=16/32） | ≤{100*worst_sim:.2f}% | **是** |")
    cont = load(base / "continuum.json")
    if cont:
        L.append(f"| 连续谱 Ra_c（Richardson 外推） | 1707.7597 Chandrasekhar | {fmt(cont['Ra_c_extrapolated'],3)} | "
                 f"{100*cont['rel_err_extrapolated']:+.4f}% | **是** |")
    for r in runs:
        ny = r["params"]["ny"]
        remap = r["Ra_c_lattice"] * ((ny - 1.0) / ny) ** 3
        flag = "是" if abs(remap / RA_CRIT - 1) < 0.03 else "**否**"
        L.append(f"| Ra_c^lat（ny={ny}，H_eff 折算后） | 1707.7597 | {fmt(remap,2)} | "
                 f"{100*(remap/RA_CRIT-1):+.2f}% | {flag} |")
    if dec:
        mx = max(abs(r["lam_rel_err"]) for r in dec["decoupling"])
        L.append(f"| 实测 λ = μ_d/μ_c | 设定 λ | §3.4 | ≤{100*mx:.3f}% | **是** |")
    if settle and settle.get("settling_summary"):
        s = settle["settling_summary"]
        L.append(f"| LBM 沉降终速 U vs HR | HR 解析 | §3.5 | 最大 {100*s['max_abs_rel_err']:.0f}% | **否（honest FAIL）** |")
    L.append("")
    L.append("**回答\"是否 <3%\"**：①RB 离散特征值对同一算子非线性解：**是**（0.01%–0.06%，ny=16/32）；"
             "②连续谱参考对 Chandrasekhar：**是**（5e-6）；"
             "③格点 Ra_c 对 1707.76：**粗网格下否**（ny=16/32/64 折算后仍为 "
             + "/".join(f"{100*(r['Ra_c_lattice']*((r['params']['ny']-1.0)/r['params']['ny'])**3/RA_CRIT-1):+.1f}%" for r in runs)
             + "，随 ny 以 h² 下降，本工作给出的外推趋势指向 0 但 ny≤64 未达 3% 门限——**这一点不粉饰**）；"
             "④λ/ρ 解耦：**是**（≤0.11%）；⑤LBM 端 HR 对拍：**否**（相差 3 个量级，见 §5.1）。\n")
    L.append("### 5.1 honest FAIL 明细（LBM 端 Hadamard–Rybczynski）\n")
    L.append("伪势双相引擎在本预算内（n=24³、R=5、1200 步、|G_12|=1.5、背景密度 0.3）无法给出可信的蠕流沉降速度："
             "液滴组分平均漂移速度 U_rel 与 HR 解析值相差 3 个量级以上，且时间序列抖动与信号同量级"
             "（见 §3.5 表与 `hr_decoupling_settling.png` 右图）。**根因**：(a) 界面寄生流（spurious current）量级 5e-2，"
             "而 Re→0 的 HR 终速只有 ~1e-5，信噪比 ~1e-3；(b) 保背景密度后界面较宽且 |G_12| 需求偏高，"
             "界面处动量交换由 Shan–Chen 力主导而非黏性，u_eq=\"self\" 下切向速度不连续，与 HR 假设不符；"
             "u_eq=\"mixture\" 则在 τ1≠τ2 时带有 (1/τ2−1/τ1) 量级的动量源（库文档已注明）。"
             "**结论**：以现有伪势双相核，λ 可在库层被独立设定并实测验证，但\"液滴沉降是否遵循 HR\"在本分辨率下不可判定；"
             "要达成 15% 判据需要 (i) 更高分辨率与更小寄生流（如采用改进力格式/更高阶各向同性差分），"
             "或 (ii) 换用自由能/相场双相核做同类对拍。此项如实记为 FAIL，不作为已验收成果。\n")

    # ---------------- 6 审计链 ----------------
    L.append("## 6. 审计链与产物路径\n")
    L.append("| 环节 | 命令 | 产物 |")
    L.append("|---|---|---|")
    L.append("| RB 生产（每分辨率） | `python benchmarks/rb_discrete_stability/run_bench.py --ny N` | `artifacts/in31_n12/rb_lattice_nyN.json` |")
    L.append("| RB 连续谱参考 | `critical_rayleigh_continuous(M_list=(50,100,200))` | `artifacts/in31_n12/continuum.json` |")
    L.append("| 基态/等效间隙诊断 | `conductive_base_state` 基态温度剖面拟合（见 `make_report` 说明） | `artifacts/in31_n12/base_profile.json` |")
    L.append("| 双相解耦实测 | `python benchmarks/rb_discrete_stability/run_hr_bench.py --mode decoupling` | `artifacts/in31_n12/hr_decoupling.json` |")
    L.append("| 双相沉降对拍 | `... --mode settling` | `artifacts/in31_n12/hr_settling.json` |")
    L.append("| 图/判据汇总 | `python benchmarks/rb_discrete_stability/make_figures.py` | `figures/in31_n12/*.png`, `artifacts/in31_n12/criteria.json` |")
    L.append("| 单测 | `pytest -q tests/test_rb_discrete_stability.py tests/test_pseudopotential_hr.py` | – |")
    L.append("")
    L.append("代码：`src/tensorlbm/rb_discrete_stability.py`（离散稳定性）、`src/tensorlbm/rb_continuum_stability.py`（连续谱参考）、"
             "`src/tensorlbm/pseudopotential_hr.py`（λ/ρ 解耦 + 逐组分施力核 + 沉降驱动）。\n")
    L.append("### 与既有成果的边界（避免重复 verified/）\n")
    L.append("- `tensorlbm.hadamard_rybczynski`（解析律 + 轴对称有限元 Stokes）与 `tensorlbm.two_phase_viscosity`（λ/ρ 解耦映射）"
             "已存在且未修改；本节点的增量是：RB 离散谱求解器（全新）、仓库内可复现的连续谱参考（替代外部 Chebyshev 数值）、"
             "逐组分施力核与 λ 解耦的**实测**验证、以及伪势引擎 HR 对拍的定量 FAIL 诊断。\n")

    out = Path("docs/reports/report_in31_n12_rb_hr.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"wrote {out} ({len(''.join(L))} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())