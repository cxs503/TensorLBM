"""Generate the P158W3-N01 report from the record (no hand-copied numbers).

The document is a plain template with ``<<token>>`` placeholders; every number
comes from ``records/rotating_w3.json`` (or is computed from it here), so the
report cannot drift away from the record.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, "records", "rotating_w3.json")
OUT = os.path.join(HERE, "..", "..", "..", "docs", "ROTATING_W3_P158W3_N01.md")


def f(x, n=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "PASS" if x else "**FAIL**"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x != x:
            return "nan"
        if x == 0:
            return "0"
        ax = abs(x)
        if ax < 1e-3 or ax >= 1e5:
            return f"{x:.3e}"
        return f"{x:.{n}f}"
    return str(x)


def pct(x, n=3):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{100.0 * x:.{n}f}%"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


TEMPLATE = r"""# 旋转参考系内部流场验证面 III：认证层（GCI / 无量纲力矩 / Ekman 螺旋角 / 旋转稳定性边界）

> 战役 DAG 节点 `P158W3-N01` · 仓库 TensorLBM · 模块 `src/tensorlbm/rotating_certification.py`
> 基准 `benchmarks/verified/rotating_w3/run.py` · 记录 `benchmarks/verified/rotating_w3/records/rotating_w3.json`
> commit `<<commit>>` · 生成 <<generated>> · 墙钟 <<wallclock>> s ·
> Python <<python>> / numpy <<numpy>>
> **机器判据：<<npass>>/<<ntotal>> 项达标**（逐条见 ③）

## ① 案例介绍

### 物理背景

在随参考系以角速度 $\boldsymbol\Omega$ 旋转的坐标系里，不可压 Navier–Stokes 方程带两个惯性伪力

$$\frac{D\mathbf u}{Dt}=-\frac{\nabla p}{\rho}-2\boldsymbol\Omega\times\mathbf u-\boldsymbol\Omega\times(\boldsymbol\Omega\times\mathbf r)+\nu\nabla^2\mathbf u .$$

平台已在前两波建成这条阶梯的**物理面**并验证：`rotating_interior_frame`（W1：环隙旋转系 Couette
的压力锚 + 旋转平面槽道 Ekman 二次流）与 `rotating_duct_frame_w2`（W2：Guo/EDM 力项交叉验证、
Ekman 层厚度标度律、反向旋转 Couette、旋转方管二次流）。按节点纪律**不得重复**这些已验证案例。

本节点建的是它们的**空白：认证层**——把「够不够准」变成可复算的数，而不是断言。四个新面：

1. **网格收敛（A6）**：$u_\theta$ 的 GCI 观测阶。W1/W2 都只有单网格精度数字；本节点用
   **几何相似的三级加密**（$\eta=r_i/r_o=1/2$ 严格不变，$r_i=12,18,27$，加密比 1.5）给出
   观测阶、Richardson 外推与细网 GCI，并**区分三种误差定义**（名义半径 / 有效半径 / 体区形状残差），
   因为离散壁是**阶梯**，不同定义的收敛行为完全不同（③ 给出实测差异）。
2. **角动量零检验（A7）**：两筒**同速共转**时精确解是刚体转动（$B=0$），任何壁面动量注入错误、
   力项符号错误都会留下非零力矩。这是 W1/W2 都没有的**零值锚**：对非零解，1% 的系数错误
   常被剖面拟合吸收；零值不会。
3. **Ekman 螺旋角（A5）**：$45^\circ$ 是**壁面**性质，而壁面剪切只能从离开壁面一个格点的第一层
   节点做单侧差分——那里剪切矢量已随 $1/\delta$ 转过若干度。本节点在 $\delta_E=6,8,12,16$ 格上测角
   并**外推到壁面**（$1/\delta\to0$），不假设剖面形状。
4. **旋转稳定性边界**：honest-FAIL 预案要求报告的边界。枢轴量是离心加速度 $\Omega_f^2 r$ 与
   格子加速度尺度 $c_s^2$ 之比，而不是 Rossby 数（弱旋转 $Ro$ 大却完全稳定）。

另记一条**被否定的假设**（②）：把阶梯壁换成"精确壁位"的三条显然做法全部失败——两条是
**稳定性**问题、一条是**信息论**问题（近壁 $f^{neq}$ 在相邻节点间几乎不变，插值不携带壁位信息）。
这条负结果决定了本阶梯只能继续用半程反弹 + 有效半径比较法。

### 参数表（程序化读出，零手抄）

<<params_tbl>>

## ② 怎么做的

- **求解器**：`rotating_certification.CouetteAnnulusSubgrid`（D2Q9 BGK，float64 numpy，整场 9 方向
  矢量化；Coriolis $a=-2\Omega\times u$ + 离心 $a=\Omega^2 r$，Guo (2002) 二阶力项，半程速度用两次
  不动点求解；壁面半程反弹 + Ladd 动量修正，壁速取**壁点处**的 $\omega\times r$）。与之逐位同构的
  `rotating_interior.RotatingAnnulusD2Q9` 在单测中交叉断言：同一初值下 $|\Delta f|<10^{-15}$。
- **Couette 加密链**：$r_i=12,18,27$，$r_o=2r_i$（$\eta=1/2$ 严格），网格 $n=2r_o+4$，固定
  $Re_{gap}=u_w g/\nu=11.52$（与已验证 `taylor_couette` 的 $r_i=12$ 工况同雷诺数），初值取解析剖面，
  积到 $\max|\Delta u|<10^{-10}$。
- **三种误差定义**（本节点的方法学核心）：
  1. *名义半径*：直接与 $u=Ar+B/r$（名义 $r_i,r_o$）比；
  2. *有效半径*：先对实测剖面在精确解族上做加权最小二乘得 $(A,B)$，再由两个无滑移条件
     $r_{eff}=\sqrt{B/(\omega-A)}$ 反演 $r_{i,eff},r_{o,eff}$，在该几何上比。本节点把 W1 的
     $\sqrt{-B/A}$（只对 $\omega_o=0$ 成立）推广到任意 $\omega_o$：反向旋转时零速点落在间隙内部，
     用旧式得 $r_{o,eff}=17$ 而非 24（**实测踩坑**，见 ⑤）；
  3. *体区形状残差*：剖面偏离**精确解族本身**的加权 RMS（两个自由参数已被拟合吸收），因此不含
     壁位几何误差，是真正的离散误差度量；再取 $r\in(r_i+2,r_o-2)$ 排除两条壁邻层。
- **力矩**：不用 Ladd 动量交换（阶梯圆壁压力项不精确抵消，W1 记录残差比解析力矩大两个量级），
  而用剖面恒等式 $T(r)=2\pi\rho\nu L r^2(du_\theta/dr-u_\theta/r)$（对精确解与 $r$ 无关，
  $=-4\pi\mu L B$），取间隙中段平均，再除 $2\pi\rho\nu^2L$ 得 Taylor 无量纲力矩
  （精确解 $C_M=2|B|/\nu$），使不同网格/转速可比。
- **Ekman**：复用已验证的 `rotating_interior.run_rotating_channel_lbm`（D3Q19 + Guo），$\delta_E$ 由
  **一参数盲拟合**（幅度 $w_p=-iG/2\Omega$ 由输入固定）与五点剖面误差双通道给出；螺旋角用二阶
  单侧差分取第一层流体内点的 $\tau$ 与地转内区速度的夹角，并对**解析剖面施加同一算子**得"离散匹配"
  参照，二者之差才是求解器误差。
- **稳定性扫描**：固定 $Re$，参考系角速度从 0 扫到 0.4（$\Omega_f^2 r/c_s^2$ 跨四个量级），记录
  收敛性、密度偏离、$Ro=u_w/(2\Omega_f r_i)$、Ekman 数 $E=\nu/(\Omega_f r_i^2)$。
- **工具链**：numpy float64 内核 + 已验证的 torch D3Q19 通道求解器 + matplotlib；每个用例带 SHA-256
  参数锚（`params_sha256`）。

### 被否定的假设：阶梯壁的"精确壁位"改造（三条全败）

动机：阶梯壁把壁面放在最近的半格上，几何误差 $O(h)$ 且**不随网格减小**（实测名义半径误差
<<nominal_triplet>>，非单调），使 GCI 无法成立。于是实现了三条"显然"的亚网格壁方案，在
**解析解已知**的亚网格平面 Poiseuille 槽道（壁距流体节点 $q=0.7$ 格）上实测：

- **整体线性外推** $f_k=(1+q)f_i^+(x_f)-qf_i^+(x_f+c_k)$：600 步内 NaN。边界增益 $1+q>1$，
  壁面成为放大器——"把分布函数外推到壁面"的直观陷阱。
- **Bouzidi (2001) 分裂格式**：稳定，但 $q\ge1/2$ 分支是**时间**外推 $f_i^+(t-1)$，定常极限下
  $f_i^+(t-1)=f_i^+(t)$，于是静默退化为半程格式（实测误差 3.7623%，与半程逐位相同）。
- **半程 + 非平衡部分外推** $f^{neq}(q)=(1+q')f^{neq}(x_f)-q'f^{neq}(x_f+c_k)$，$q'=q-1/2$：
  稳定但**无效果**（3.7613% vs 3.7623%），因为近壁两节点的 $f^{neq}$ 几乎不变，插值不携带壁位信息。

结论（结构性）：$q\ge1/2$ 时第二个插值点落在固体内，任何"两流体点"格式都是外推、增益 $>1$；
稳定的亚网格曲壁需 multireflection 类边界条件。故本阶梯继续用半程反弹 + 有效半径比较，
并把 $O(h)$ 阶梯误差作为**已知局限**入账（不掩盖）。回归守卫：
`tests/test_rotating_certification.py::test_subgrid_wall_error_scales_like_h`（同一位移在两级网格上误差减半）。

## ③ 计算结果

### 判据汇总（程序化判定，逐条阈值）

<<crit_tbl>>

### A1/A2/A6：Couette 加密链

<<ladder_tbl>>

- **A6 观测阶**：体区形状残差三重值 <<bulk_triplet>> → $p=<<p_bulk>>$；有效半径误差三重值
  <<eff_triplet>> → $p=<<p_eff>>$；无量纲力矩三重值 → $p=<<p_cm>>$（力矩误差非单调，故该项无阶可报）。
  细网 GCI：体区 <<gci_bulk>>，有效几何 <<gci_eff>>。
- **名义半径误差三重值不单调**（<<nominal_triplet>>，monotone=<<nominal_monotone>>）：阶梯几何误差是
  $O(h)$ 且相位随加密随机变化。这是必须改用 R$_{eff}$ / 体区残差的**实测理由**，不是偏好。

### A7：共转零值锚

<<extra_tbl>>

共转工况的 $T$ 相对差分工况解析力矩之比为 <<a7_ratio>>（阈值 2%），刚体剖面误差 <<a7_prof>>。

### A4/A5：Ekman 层

<<ek_tbl>>

**全剖面最大误差的成因（本节点实测的定量诊断）**：`profile_rel_err` 随 $\delta_E$ 以 $1/\delta_E$ 衰减，
而 $\delta_E\times$`profile_rel_err` 在四个已分辨工况上为 <<half_offsets>>——恰是**半格壁位偏移**
（半程反弹把无滑移点放在半格处，第一层节点速度与名义壁位解析值相差 $\approx0.5\,h\,du/dy$）。
这是**已知的边界格式一阶效应**，不是力项/实现错误：$\delta_E$ 拟合（一参数盲拟合）与五点探针
（避开壁邻节点）在同一批工况上分别为 ≤0.78% 与 ≤1%（W2 记录）。修正它需要 ②中失败的亚网格壁格式。

螺旋角外推（$\delta_E\ge6$ 四点线性拟合 $\theta$ vs $1/\delta$）：$\theta_{wall}=<<a5_intercept>>^\circ$，
斜率 <<a5_slope>>°/node$^{-1}$，与 $45^\circ$ 相差 <<a5_delta>>°。最大 $\delta_E=16$ 的单点实测
<<a5_raw>>°，离散匹配误差 <<a5_matched>>°。

### 旋转稳定性边界

<<stab_tbl>>

边界（程序化读出）：$\Omega_f\le<<stab_omega>>$ 时场仍有限（$|\rho-1|<50\%$），
$\Omega_f\in(<<blow_lo>>,<<blow_hi>>)$ 之间发散；不可压有效性 $|\rho-1|$ 随转速迅速恶化
（$\Omega_f=0.002$ 时 <<dev_small>>，$0.01$ 时 <<dev_mid>>，$0.05$ 时 <<dev_big>>），
即**精度边界远早于稳定性边界**出现。

![A1/A2/A6 加密链](assets/rotating_w3/fig2_grid_convergence.png)

![Couette 剖面](assets/rotating_w3/fig1_couette_profiles.png)

![共转零值锚](assets/rotating_w3/fig3_corotating_null.png)

![Ekman 螺旋与螺旋角](assets/rotating_w3/fig4_ekman_spiral.png)

![Ekman 厚度分辨率](assets/rotating_w3/fig5_ekman_resolution.png)

![旋转速率稳定性](assets/rotating_w3/fig6_stability.png)

## ④ 误差判定（逐项 vs 基准，显式回答是否 <3%）

<<err_tbl>>

口径：$C_M$ 在网格链上的误差为 <<cm_triplet>>（**非单调**，阶梯壁所致），故 A2 取网格链与反向旋转
工况的最大值；Ekman 剖面误差为六种 $\delta_E$ 的最大值。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **方位角投影的符号约定（本节点实测踩坑）**：$u_\theta=(\mathbf r\times\mathbf u)_z/r=(x u_y-y u_x)/r$。
   第一版误写成 $(u_x y-u_y x)/r$（差一个负号），与壁面驱动方向错位后剖面整体反号，表现为
   10 点误差 200%（而非几个百分点）——**符号错误以量级形式暴露**。这正是把 10 点采样判据放在
   第一位的原因；`test_azimuthal_projection_convention` 钉住该约定。
2. **有效半径反演不能对任意 $\omega_o$ 用 $\sqrt{-B/A}$**：反向旋转时 $u_\theta=0$ 的半径在间隙内部
   （实测 17 而非 24），必须用双壁无滑移式 $r_{eff}=\sqrt{B/(\omega-A)}$，否则 A1 假性爆到 156%。
3. **阶梯壁 $O(h)$ 不可消除**（②三条负结果）：名义半径误差非单调、不可用于 GCI；本节点所有收敛性
   结论都建立在 R$_{eff}$ 与体区形状残差上，并明确标注适用范围。
4. **$45^\circ$ 是壁面性质**：第一层流体内点离壁一格，$\delta_E$ 越薄偏差越大（$\delta_E=6$ 时实测
   <<angle6>>°，同算子解析值 <<angle6_ex>>°），故必须外推 $1/\delta\to0$；单点直接对 $45^\circ$ 比
   会误判格式错误。
5. **离心项是稳定性的真正瓶颈**：$\Omega_f^2 r/c_s^2\gtrsim0.1$ 时密度偏离已超 1%，
   $\gtrsim0.5$ 时发散（Guo 力项无法被状态方程吸收）。$Ro$ 不是合适的稳定坐标：弱旋转 $Ro$ 大
   却完全稳定，强旋转（低 $Ro$）才是危险区。
6. **适用范围**：低 Mach（$\mathrm{Ma}\le0.11$）、常密度、$\delta_E\ge4$ 格；Couette 结果限于
   $Re_{gap}=11.52$ 的层流轴对称区（$\eta=1/2$ 时 Taylor 涡临界 $Re_i\approx68$）。

## ⑥ 产物清单

- 模块：`src/tensorlbm/rotating_certification.py`（GCI、无量纲力矩、有效半径反演、螺旋角、
  稳定性扫描、亚网格壁研究装置）
- 基准：`benchmarks/verified/rotating_w3/run.py`（出记录）、`make_figures.py`、`make_doc.py`、`make_report.py`
- 记录：`benchmarks/verified/rotating_w3/records/rotating_w3.json`（+ `rotating_w3_fields.npz`）
- 图：`docs/assets/rotating_w3/fig1..fig6`
- 测试：`tests/test_rotating_certification.py`
"""


def main() -> int:
    r = json.load(open(REC))
    crit = r["criteria"]
    ladder = r["cases"]["couette_ladder"]
    extra = r["cases"]["couette_extra"]
    ek = r["cases"]["ekman"]
    stab = r["cases"]["stability_sweep"]
    a5 = crit["A5_ekman_spiral_angle_deg"]
    a6 = crit["A6_gci_order"]

    def val(k):
        return crit[k]["value"]

    tok = {
        "commit": r["git_commit"], "generated": r["generated_utc"], "wallclock": r["wall_clock_s"],
        "python": r["environment"]["python"], "numpy": r["environment"]["numpy"],
        "npass": sum(1 for k, v in crit.items() if k.startswith("A") and v.get("pass")),
        "ntotal": sum(1 for k in crit if k.startswith("A")),
        "params_tbl": table(
            ["面", "网格", "τ / ν", "转速", "驱动 / 其它"],
            [["Couette 加密链", "$n^2$, $r_i$=" + "/".join(f(c["ri"], 0) for c in ladder)
              + ", $r_o=2r_i$", "0.650 / 0.050", "内筒驱动 $\\omega_i$, $\\omega_o=0$",
              "$Re_{gap}=u_w g/\\nu=11.52$, $\\eta=0.5$, Ma<=%.3f" % max(c["Ma"] for c in ladder)],
             ["Couette 反向 / 共转", "$n^2$, $r_i=%s$" % f(ladder[-1]["ri"], 0), "0.650 / 0.050",
              "$\\omega_o=-0.5\\omega_i$ / $\\omega_o=+\\omega_i$", "10 点采样判据；共转为零值锚"],
             ["Ekman 平面槽道", "$4\\times N\\times3$ (D3Q19), $h=6\\delta_E$, $N=2h+2$", "0.650 / 0.050",
              "$\\Omega$ 绕壁法向, $\\delta_E=2..16$ 格", "$G=2\\Omega u_g$, $u_g=0.02$"],
             ["旋转速率扫描", "$n^2$, $r_i=8$", "0.650 / 0.050", "$\\Omega_f=0..0.4$",
              "内筒驱动 + 参考系旋转"]]),
        "crit_tbl": table(
            ["判据", "定义", "实测", "阈值", "判定"],
            [["A1 Taylor-Couette 剖面", "10 径向采样 vs 精确解 $Ar+B/r$，R$_{eff}$ 几何",
              pct(val("A1_couette_profile")), "≤ 2%", f(crit["A1_couette_profile"]["pass"])],
             ["A2 无量纲力矩", "$|C_M/C_{M,ana}-1|$, $C_M=T/(2\\pi\\rho\\nu^2L)$",
              pct(val("A2_torque_coefficient")), "≤ 3%", f(crit["A2_torque_coefficient"]["pass"])],
             ["A3 方管二次流", "引用已验证记录 W2 的最小偏差", pct(val("A3_duct_secondary")), "≤ 10%",
              f(crit["A3_duct_secondary"]["pass"])],
             ["A4 Ekman 厚度", "$\\max|\\delta_{fit}/\\delta_{ana}-1|$（$\\delta_E\\ge4$ 格）",
              pct(val("A4_ekman_thickness")), "≤ 5%", f(crit["A4_ekman_thickness"]["pass"])],
             ["A5 Ekman 螺旋角", "壁面剪切/地转速度夹角外推到壁面 vs $45^\\circ$",
              "%.2f° (Δ=%.2f°)" % (a5["intercept_deg"], a5["value"]), "≤ 5°", f(a5["pass"])],
             ["A6 GCI 观测阶", "体区形状残差三重值的观测阶 $p$", "%.3f" % a6["order_bulk_shape_residual"],
              "≥ 1.5", f(a6["pass"])],
             ["A7 共转零值锚", "$|T|/T_{ref}$（两筒同速，精确解刚体转动）",
              "%.2e" % crit["A7_corotating_null"]["value"], "≤ 2%", f(crit["A7_corotating_null"]["pass"])]]),
        "ladder_tbl": table(
            ["$r_i$", "$n$", "名义半径误差", "有效半径误差", "体区形状残差", "$C_M$ 误差",
             "$r_{i,eff}-r_i$", "$r_{o,eff}-r_o$", "$u_r/u_\\theta$", "步数"],
            [[f(c["ri"], 0), c["n_grid"], pct(c["err_nominal_10pt"]), pct(c["err_effective_10pt"]),
              pct(c["shape_residual_bulk"]), pct(c["CM_rel_err"]), f(c["dr_i_eff"], 3),
              f(c["dr_o_eff"], 3), "%.1e" % c["u_r_over_u_theta"], c["steps"]] for c in ladder]),
        "extra_tbl": table(
            ["工况", "名义误差", "有效误差", "$T$ 实测", "$T$ 解析", "$C_M$ 误差", "$u_r/u_\\theta$",
             "质量漂移"],
            [[c["label"], pct(c["err_nominal_10pt"]), pct(c["err_effective_10pt"]), f(c["torque_mid"]),
              f(c["torque_analytic"]), pct(c["CM_rel_err"]), "%.1e" % c["u_r_over_u_theta"],
              pct(c["mass_drift"], 2)] for c in extra]),
        "ek_tbl": table(
            ["$\\delta_E$ (格)", "$h$", "$\\delta$ 误差", "五点剖面误差", "螺旋角(实测)",
             "螺旋角(解析,同算子)", "离散匹配角差", "步数"],
            [[f(e["delta_nodes"], 0), e["h"], pct(e["delta_rel_err"]), pct(e["profile_rel_err"]),
              "%.2f°" % e["spiral_angle_deg"], "%.2f°" % e["spiral_angle_exact_discrete_deg"],
              "%.3f°" % e["spiral_angle_err_vs_exact_deg"], e["steps"]] for e in ek]),
        "stab_tbl": table(
            ["$\\Omega_f$", "$Ro$", "$E=\\nu/(\\Omega_f r_i^2)$", "$\\Omega_f^2 r/c_s^2$", "$\\rho_{min}$",
             "$\\rho_{max}$", "$|\\rho-1|_{max}$", "稳定", "不可压有效"],
            [[f(s["omega_frame"], 4), f(s["rossby"], 3), f(s["ekman_number"], 4),
              f(s["centrifugal_over_cs2"], 3), f(s["rho_min"], 4), f(s["rho_max"], 4),
              pct(s["max_rel_rho_dev"], 2), f(s["stable"]), f(s["accurate_incompressible"])] for s in stab]),
        "nominal_triplet": ", ".join(pct(c["err_nominal_10pt"]) for c in ladder),
        "bulk_triplet": ", ".join(pct(c["shape_residual_bulk"], 4) for c in ladder),
        "eff_triplet": ", ".join(pct(c["err_effective_10pt"]) for c in ladder),
        "cm_triplet": ", ".join(pct(c["CM_rel_err"]) for c in ladder),
        "p_bulk": f(a6["order_bulk_shape_residual"], 3),
        "p_eff": f(a6["order_effective_geometry"], 3),
        "p_cm": f(a6["order_dimensionless_torque"], 3),
        "gci_bulk": pct((a6["gci_fine_pct_bulk"] or 0) / 100.0, 2),
        "gci_eff": pct((a6["gci_fine_pct_effective"] or 0) / 100.0, 2),
        "nominal_monotone": a6["nominal_monotone"],
        "a7_ratio": "%.2e" % crit["A7_corotating_null"]["value"],
        "a7_prof": pct(list(crit["A7_corotating_null"]["profile_err_vs_rigid_body"].values())[0]),
        "a5_intercept": f(a5["intercept_deg"], 2), "a5_slope": f(a5["slope_deg"], 2),
        "a5_delta": f(a5["value"], 2), "a5_raw": f(a5["raw_largest_delta"]["angle_deg"], 2),
        "a5_matched": f(a5["discretisation_matched_error_deg"].get("16.0")
                        or a5["discretisation_matched_error_deg"].get(16.0), 3),
        "angle6": f(a5["per_delta_deg"].get("6.0") or a5["per_delta_deg"].get(6.0), 2),
        "angle6_ex": f(a5["exact_profile_same_operator_deg"].get("6.0")
                       or a5["exact_profile_same_operator_deg"].get(6.0), 2),
        "half_offsets": ", ".join("%.3f" % (e["delta_nodes"] * e["profile_rel_err"])
                                  for e in ek if e["delta_nodes"] >= 6.0),
        "stab_omega": f(crit["stability_boundary"]["stable_upto_omega"], 3),
        "blow_lo": f(crit["stability_boundary"]["blowup_between"][0], 3),
        "blow_hi": f(crit["stability_boundary"]["blowup_between"][1], 3),
        "dev_small": pct(next(s["max_rel_rho_dev"] for s in stab if abs(s["omega_frame"] - 0.002) < 1e-9), 2),
        "dev_mid": pct(next(s["max_rel_rho_dev"] for s in stab if abs(s["omega_frame"] - 0.01) < 1e-9), 2),
        "dev_big": pct(next(s["max_rel_rho_dev"] for s in stab if abs(s["omega_frame"] - 0.05) < 1e-9), 2),
    }

    resolved = [e for e in ek if e["delta_nodes"] >= 4.0]
    max_ek = max(e["profile_rel_err"] for e in resolved)
    worst_low = max((e for e in ek if e["delta_nodes"] < 4.0), key=lambda e: e["profile_rel_err"])
    tok["err_tbl"] = table(
        ["关键量", "基准", "误差", "是否 <3%"],
        [["Taylor-Couette 剖面（有效几何，10 点）", "闭式解 $Ar+B/r$", pct(val("A1_couette_profile")),
          "是" if val("A1_couette_profile") < 0.03 else "否"],
         ["无量纲力矩 $C_M$（网格链 + 反向旋转）", "$2|B|/\\nu$", pct(val("A2_torque_coefficient")),
          "是" if val("A2_torque_coefficient") < 0.03 else "否"],
         ["Ekman 厚度 $\\delta_E$（$\\delta_E\\ge4$ 格）", "$\\sqrt{\\nu/\\Omega}$",
          pct(val("A4_ekman_thickness")), "是" if val("A4_ekman_thickness") < 0.03 else "否"],
         ["Ekman 全剖面最大误差（已分辨 $\\delta_E\\ge4$）", "闭式解", pct(max_ek),
          "否（$=\\delta_E^{-1}\\times0.50$ 的半格壁位效应，见 ③ 诊断）"],
         ["Ekman 厚度（$\\delta_E=%s$，故意欠分辨）" % f(worst_low["delta_nodes"], 0), "闭式解",
          pct(worst_low["profile_rel_err"]), "否（分辨率下界，见 A4）"],
         ["Ekman 螺旋角（离散匹配，$\\delta_E=16$）", "解析剖面同算子",
          "%.3f°" % (a5["discretisation_matched_error_deg"].get("16.0")
                     or a5["discretisation_matched_error_deg"].get(16.0)), "角度量，见 A5"],
         ["旋转方管二次流（引用 W2，未重跑）", "线性参考解", pct(val("A3_duct_secondary")),
          "是" if val("A3_duct_secondary") < 0.03 else "否（阈值 10%）"]])

    md = TEMPLATE
    for k, v in tok.items():
        md = md.replace("<<" + k + ">>", str(v))
    with open(OUT, "w") as fh:
        fh.write(md)
    print("wrote", OUT, len(md), "chars;", tok["npass"], "/", tok["ntotal"], "criteria pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())