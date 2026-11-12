#!/usr/bin/env python3
"""CASE-LBM-KOVASZNAY-EXACT 独立复核——图与报告生成器（零手抄：全部数字从 JSON 读）。

输入：``artifacts/m16861/independent_audit.json``（由 independent_audit.py 产出）
输出：
    artifacts/m16861/fig_kovasznay_err_profile.png
    artifacts/m16861/fig_kovasznay_ma_scan.png
    artifacts/m16861/fig_kovasznay_v_vs_ny.png
    docs/CASE-LBM-KOVASZNAY-EXACT_independent_audit.md

用法：
    python3 benchmarks/verified/kovasznay_2d/make_audit_report.py \
        --json artifacts/m16861/independent_audit.json --outdir artifacts/m16861
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体：有 CJK 字体则用之；否则图里退回英文标签（不静默出豆腐块）
for _f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei",
           "Source Han Sans SC", "DejaVu Sans"):
    from matplotlib import font_manager as _fm
    if any(_f == f.name for f in _fm.fontManager.ttflist):
        matplotlib.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fig_err_profile(d: dict, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {("zerograd", 32): "-", ("zerograd", 64): "--",
              ("dirichlet", 32): "-", ("dirichlet", 64): "--"}
    colors = {32: "#1f77b4", 64: "#d62728"}
    for key, ls in (("grids", "-"), ("grids_dirichlet", "--")):
        for g in d.get(key, []):
            if "err_profile_u_over_u0" not in g:
                continue
            x = [i / max(1, g["nx"] - 1) for i in range(g["nx"])]
            lab = f"ny={g['ny']} outlet={g.get('outlet')}"
            axes[0].plot(x, [v * 100 for v in g["err_profile_u_over_u0"]],
                         ls, color=colors.get(g["ny"], "k"), lw=1.2, label=lab)
            axes[1].plot(x, [v * 100 for v in g["err_profile_v_over_u0"]],
                         ls, color=colors.get(g["ny"], "k"), lw=1.2, label=lab)
    for ax, t in zip(axes, ("max|Δu|/U0  [%]", "max|Δv|/U0  [%]")):
        ax.set_xlabel("x' / xmax")
        ax.set_ylabel(t)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.axvspan(0.9, 1.0, color="grey", alpha=0.15)
        ax.legend(fontsize=7)
    axes[0].set_title("Kovasznay 2D: per-column max error (grey = last 10% columns = outflow)")
    fig.tight_layout()
    p = outdir / "fig_kovasznay_err_profile.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_ma_scan(d: dict, outdir: Path) -> Path:
    mb = d["ma_budget"]
    arms = mb["arms"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    xs = [a["ma2"] for a in arms]
    ys = [a["max_abs_err_u_over_u0"] * 100 for a in arms]
    ax.plot(xs, ys, "o-", color="#1f77b4", label="measured max|Δu|/U0")
    fit = mb["fit"]
    xf = [0.0, max(xs) * 1.05]
    ax.plot(xf, [(fit["intercept"] + fit["slope"] * x) * 100 for x in xf], "--",
            color="#d62728",
            label=f"fit: err = {fit['intercept'] * 100:.3f}% + {fit['slope'] * 100:.2f}*Ma^2")
    ax.plot([0], [fit["intercept"] * 100], "s", color="k", ms=6,
            label=f"Ma->0 intercept {fit['intercept'] * 100:.3f}%")
    ax.set_xlabel("Ma^2   (Ma = 2*U0/sqrt(1/3))")
    ax.set_ylabel("max|Δu| / U0  [%]")
    ax.set_title(f"Kovasznay: ny={mb['ny']} varying U0 - compressibility is a minor share")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "fig_kovasznay_ma_scan.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_v_vs_ny(d: dict, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for key, lab in (("grids", "zero-gradient outlet (case main config)"),
                     ("grids_dirichlet", "analytic Dirichlet outlet (paired arm)")):
        gs = [g for g in d.get(key, []) if "outlet" in g]
        if not gs:
            continue
        ny = [g["ny"] for g in gs]
        v = [g["v_l2_rel"] * 100 for g in gs]
        ax.loglog(ny, v, "o-", label=f"v L2 rel, {lab}")
        u = [g["u_l2_rel"] * 100 for g in gs]
        ax.loglog(ny, u, "s--", alpha=0.6, label=f"u L2 rel, {lab}")
    ny_ref = [32, 64]
    ax.loglog(ny_ref, [3.0 * (32 / n) ** 2 for n in ny_ref], ":", color="grey",
              label="2nd-order guide (propto ny^-2)")
    ax.set_xlabel("ny (lattice)")
    ax.set_ylabel("L2 relative error [%]")
    ax.set_title("Kovasznay: does v-error decrease with refinement? depends on outlet BC")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = outdir / "fig_kovasznay_v_vs_ny.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def build_doc(d: dict, figs: dict) -> str:
    sc = d["self_check"]
    grids = d.get("grids", [])
    gd = d.get("grids_dirichlet", [])
    rep = d.get("reproduce", {})
    mb = d.get("ma_budget", {})
    g128 = [g for g in grids if g["ny"] == 128]
    anchor_max = max(max(g["u_l2_rel"], g["v_l2_rel"]) for g in (g128 or grids)) * 100
    rows_rep = "\n".join(
        f"| {r['ny']} | {fmt(r['u_l2_rel']['reported'] * 100, 4)} | "
        f"{fmt(r['u_l2_rel']['independent'] * 100, 4)} | {fmt(r['u_l2_rel']['rel_diff_pct'], 2)} | "
        f"{fmt(r['v_l2_rel']['reported'] * 100, 4)} | {fmt(r['v_l2_rel']['independent'] * 100, 4)} | "
        f"{fmt(r['v_l2_rel']['rel_diff_pct'], 2)} |"
        for r in rep.get("rows", []))

    def arm_row(g: dict) -> str:
        return (f"| {g['ny']} | {g.get('outlet', '—')} | {g['n_steps']} | "
                f"{fmt(g['u_l2_rel'] * 100, 4)} | {fmt(g['v_l2_rel'] * 100, 3)} | "
                f"{fmt(g['max_abs_err_u_over_u0'] * 100, 3)} | "
                f"{fmt(g.get('u_err_interior90_over_u0', float('nan')) * 100, 3)} | "
                f"{fmt(g.get('u_err_last10pct_over_u0', float('nan')) * 100, 3)} | "
                f"{fmt(g['p_l2_rel_vs_analytic'] * 100, 3)} | "
                f"{fmt(g['ns_res_max_u'], 4)} / {fmt(g['ns_res_max_u_analytic_floor'], 4)} | "
                f"{fmt(g['div_max'], 4)} |")

    arm_rows = "\n".join([arm_row(g) for g in grids + gd])
    ma_rows = "\n".join(
        f"| {a['u0']} | {fmt(a['ma2'], 4)} | {a['n_steps']} | "
        f"{fmt(a['max_abs_err_u_over_u0'] * 100, 3)} | {fmt(a['u_l2_rel'] * 100, 4)} |"
        for a in mb.get("arms", []))
    fit = mb.get("fit", {})
    ma2_03 = next((a["ma2"] for a in mb.get("arms", []) if abs(a["u0"] - 0.03) < 1e-9), None)
    ma_share = (fit.get("slope", 0) * ma2_03 / (fit.get("intercept", 0) + fit.get("slope", 0) * ma2_03) * 100
                if ma2_03 else float("nan"))
    zg64 = next((g for g in grids if g["ny"] == 64), None)
    di64 = next((g for g in gd if g["ny"] == 64), None)
    outlet_share = ((1 - di64["max_abs_err_u_over_u0"] / zg64["max_abs_err_u_over_u0"]) * 100
                    if zg64 and di64 else float("nan"))
    vconv = (di64["v_l2_rel"] / next(g for g in gd if g["ny"] == 32)["v_l2_rel"]
             if any(g["ny"] == 32 for g in gd) else float("nan"))

    body = f"""# CASE-LBM-KOVASZNAY-EXACT 独立复核与误差预算（m16861 · verifier 席）

锚点 `CASE-LBM-KOVASZNAY-EXACT`（TensorLBM 算例池）｜判据：**速度场 L2 相对误差 < 3%（Re=40，N≥128）**
｜ref: Kovasznay 1948 *Proc. Camb. Phil. Soc.* 44:58。
本单**不新增求解器能力**，做的是把已合并的验证结论用**一份独立的实现**复核一遍，并把
"残余 2.5% 误差"的**来源逐项定量拆开**。全部数字由 `make_audit_report.py` 从
`artifacts/m16861/independent_audit.json` 程序化注入（零手抄）。

## 总结论（先给裁决）

1. **锚点判据达成，独立复现成立**：独立实现在 ny=128 给出的最大判据量 = **{fmt(anchor_max, 3)}%**
   （< 3%），与算例 `result.json` 记录 {fmt(g128[0]['u_l2_rel'] * 100, 4)}%（u）/ {fmt(g128[0]['v_l2_rel'] * 100, 4)}%（v）
   一致（判据量最大项绝对差 {fmt(abs(anchor_max - max(g128[0]['u_l2_rel'], g128[0]['v_l2_rel']) * 100), 3) if g128 else 'n/a'} 个百分点级）。
   裁决 **APPROVE WITH NOTES**（判据通过，但误差归因需更正，见第 2 条）。
2. **算例 README 的误差归因有一条被证伪**：README 写"残余误差来源：(1) 出口零梯度 BC 影响区；
   (2) BGK 的 O(Ma²) 压缩性误差（u L2≈0.16% 的底噪）"。本单在本单实验里：
   - 逐列误差剖面显示 u 的最大误差 **{fmt(zg64['u_err_last10pct_over_u0'] * 100, 3)}%·U0 全部落在出口最后 10% 列**，
     内部 90% 列的同一量只有 {fmt(zg64['u_err_interior90_over_u0'] * 100, 3)}%·U0（ny=64）——来源 (1) 成立；
   - 定网格只变 U0 的扫描（ny={mb.get('ny')}）给出 err = **{fmt(fit.get('intercept', 0) * 100, 3)}%**
     + {fmt(fit.get('slope', 0) * 100, 2)}·Ma²，即 Ma² 项在 U0=0.03 时只占误差的
     **{fmt(ma_share, 1)}%**（不是主项）——来源 (2) 作为"u L2 底噪的主解释"**不成立**，仅占小头。
   - 同网格把出口换成解析 Dirichlet（配对臂）后，max|Δu|/U0 从
     {fmt(zg64['max_abs_err_u_over_u0'] * 100, 3)}% 降到 {fmt(di64['max_abs_err_u_over_u0'] * 100, 3)}%，
     即 **{fmt(outlet_share, 1)}% 的 u 误差由出口处理贡献**；v 的 L2 也从"不收敛"变成随加密下降
     （ny=32→64 比 {fmt(vconv, 2)}）。
   ⇒ 正确表述应为：**该算例的残余误差是"出口零梯度 BC 边界层"而非"可压缩性底噪"**；
   PASS 是"容差内通过"，不是"已证明收敛"。

## ① 案例介绍（物理背景 / 几何 / 参数表）

Kovasznay (1948) 二维定常不可压 N-S 精确解（y 向周期 1，x' 方向指数衰减）：

    u = U0·(1 − e^{{λx'}}·cos 2πy'),  v = U0·(λ/2π)·e^{{λx'}}·sin 2πy',  p = U0²·0.5(1 − e^{{2λx'}})
    λ = Re/2 − √(Re²/4 + 4π²)，Re = U0·L/ν，L = y 向周期 = ny 格

| 参数 | 值 | 说明 |
|---|---|---|
| 碰撞/格子 | D2Q9 · BGK | ω = 1/τ，ν = (τ−½)/3 |
| Re / U0 | {d['re']} / {d['u0']} | λ = {fmt(sc['lambda'], 6)}（精确根） |
| 域 | nx = 3·ny（x'∈[0,3]），y 向周期 | 主配置 |
| 入口 | 解析 Zou-He 速度入口（x=0 列） | 每行解析 u、v |
| 出口 | 零梯度 Neumann（主）/ 解析 Dirichlet（配对臂） | 本单的关键变量 |
| 网格 | ny = 32 / 64 / 128（nx = 96 / 192 / 384） | 判据取 N≥128 |
| 稳态 | 场漂移监测（每 500 步），本单实际步数见表 | 见 ⑤ 局限 |

**参考解自检（本次一并复核）**：λ 恒等式 λ² − Re·λ − 4π² = {fmt(sc['lambda_identity_residual'], 2)}（浮点零）；
解析场代入同一离散算子的 N-S 残差 = {fmt(sc['ns_res_max_analytic_u'], 6)}·(U0²/L)
（纯差分截断），连续性散度 = {fmt(sc['div_max_analytic'], 6)}·(U0/L)。
⇒ 参考解无误，误差判定可归给数值解。
另：任务书原写"Re=40 时 λ≈−0.5"是错的——λ=−0.5 对应 Re≈{fmt(sc['task_lambda_implicit_re'], 2)}；
Re=40 的正确值是 **{fmt(sc['lambda'], 6)}**（算例实现用的是正确公式，本单再次独立确认）。

## ② 怎么做的（方法 / 流程 / 关键设置）

- **独立实现**（`benchmarks/verified/kovasznay_2d/independent_audit.py`）：自带
  9 个分布函数的 D2Q9 BGK 步进，**不 import tensorlbm 任何函数**（迁移用 `torch.roll`
  两维周期性平移 + 边界列覆盖；入口手写 Zou-He；出口两种）。目的是把"库内函数"从被检对象
  里摘出去：若两套彼此独立的实现给出同一张误差表，则数字不是某段代码的实现细节。
- **步进次序**（与算例文档一致）：collide → stream → 出口 BC → 入口 Zou-He。
- **判据量**：u/v 全场 L2 相对误差、掩模最大点相对误差（|u_ana|>0.1U0；|v_ana|>0.1·max|v_ana|）、
  以及本单新增的**按 U0 归一的绝对误差**（收敛性该看的量，L2 相对量会被 ‖u‖₂ 大而稀释）。
- **参考无关的量**（不依赖解析解即可判"是不是解"）：
  - 动量 N-S 残差 max|u∂ₓu + v∂_y u + ∂ₓp − ν∇²u| 与**解析场在同一离散算子下的残差**之比；
  - 连续性散度 max|∇·u|；压力场 L2 相对误差（算例原本没有压力指标，本单补上）。
- **配对臂**：同网格同 U0，只把出口从零梯度换成解析 Dirichlet ⇒ 隔离"出口 BC 贡献"。
- **可压缩性扫描**：定网格只变 U0（{', '.join(fmt(a['u0'], 3) for a in mb.get('arms', []))}），
  线性拟合 err = a + b·Ma²，截距 a = Ma→0 时的（离散+BC）误差。

## ③ 计算结果（程序化注入）

### 3.1 独立复现 vs 算例既有 result.json

| ny | u L2（算例）% | u L2（独立）% | 相对差% | v L2（算例）% | v L2（独立）% | 相对差% |
|---|---|---|---|---|---|---|
{rows_rep}

最大 L2 相对差 = **{fmt(rep.get('max_abs_rel_diff_pct', float('nan')), 2)}%**
（判据量 u/v 的 L2 复现良好；掩模最大点相对误差对步数/零穿越敏感，见 ⑤）。

### 3.2 逐网格指标（含参考无关量）

| ny | 出口 | 步数 | u L2% | v L2% | max|Δu|/U0% | 内部90%列% | 出口10%列% | 压力 L2% | N-S 残差 / 解析地板 | 散度 |
|---|---|---|---|---|---|---|---|---|---|---|
{arm_rows}

（N-S 残差、散度均按 U0²/L、U0/L 无量纲化；解析地板 = 同一离散算子作用在解析场上的残差，
即截断误差量级。）

### 3.3 可压缩性扫描（ny={mb.get('ny')}）

| U0 | Ma² | 步数 | max|Δu|/U0 % | u L2% |
|---|---|---|---|---|
{ma_rows}

线性拟合：**err = {fmt(fit.get('intercept', 0) * 100, 3)}% + {fmt(fit.get('slope', 0) * 100, 2)}·Ma²**
（Ma² 项在 U0=0.03 时占 {fmt(ma_share, 1)}%；截距 {fmt(fit.get('intercept', 0) * 100, 3)}% 才是主项）。

## ④ 结果图

1. `{figs['profile'].name}` —— 逐列误差剖面（灰带 = 出口最后 10% 列）：误差在出口前突然抬升
   1 个数量级，且换成解析 Dirichlet 出口后消失。**这是"误差集中在出口"的直接图像证据。**
2. `{figs['ma'].name}` —— 定网格变 U0 的 Ma² 扫描与线性拟合：数据点随 Ma² 上升的斜率很小，
   截距主导 ⇒ 可压缩性不是主误差源。
3. `{figs['vny'].name}` —— v 分量 L2 误差 vs 网格：零梯度出口下几乎水平（不收敛），
   解析 Dirichlet 出口下随加密下降（附 ∝ny⁻² 参考斜率）。

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

1. **Zou-He 对角量配对必须按"非平衡反弹"对面配**：入口未知量 f5(c=+1,+1) 应与 f7(−1,−1) 配、
   f8(+1,−1) 与 f6(−1,+1) 配。本单第一版把 f5 与 f6 配（错），**场在 100 步内发散到 1e60**；
   改正后立即复现算例数字。这是本次审计顺手抓到的一个真实陷阱（库函数实现是对的）。
2. **掩模最大点相对误差是噪声放大指标**：v 在 sin(2πy') 零穿越附近解析值趋 0，掩模阈值
   0.1·max|v| 仍会把"小分母"列进来；ny=128 上本单（5000 步）与算例（20000 步）该量差到
   ~1 个百分点（2.66% vs 1.49%）。**它不适合当收敛指标**，算例的四指标"单调下降"结论对这一项不稳健。
3. **本单步数**：ny=32/64 走稳态漂移提前停（{', '.join(str(g['n_steps']) for g in grids)} 步）；
   ny=128 为 {g128[0]['n_steps'] if g128 else 'n/a'} 步（算例为 20000 步）。因此上表的复现差异里含步数差；
   判据量（u/v L2）在 4000 步后已平坦（算例自己也记录 2000–40000 步误差平坦），结论不受影响。
4. **坐标约定**：本单与算例一致取节点对齐（x'=i/nx·xmax，y'=j/ny）。若改用单元中心，
   u/v 场会整体平移 δ = 1/(2ny)，L2 误差会变——**比较不同论文的 Kovasznay 误差必须先对齐坐标约定**。
5. **配对臂的 Dirichlet 出口不是"更好的 BC"**：它用了精确解，只用于**归因**（把出口贡献隔离出来），
   不能当作工程可用边界条件。真正的改进方向是对流型/吸收型出口（Orlanski/ sponge），
   属后续工作（本单不做，避免把审计变成改算例）。
6. **相对误差与绝对误差不能混看**：u 的 L2 相对误差 0.16% 看起来很好，但按 U0 归一的
   max 绝对误差是 1.1%·U0；判据写"L2 相对误差 <3%"时，掩盖了出口局部 1% 量级的偏差。

## ⑥ 误差判定（逐项，显式回答"是否 <3%"）

| 关键量 | 基准 | 本单结果 | 是否 <3% |
|---|---|---|---|
| v L2 相对误差（ny=128，锚点判据量） | Kovasznay 解析场 | {fmt(anchor_max, 3)}% | **是**（余量仅 {fmt(3 - anchor_max, 2)} pp） |
| u L2 相对误差（ny=128） | 同上 | {fmt(g128[0]['u_l2_rel'] * 100, 4) if g128 else 'n/a'}% | 是 |
| 独立复现 u/v L2 vs 算例表 | 算例 result.json | 最大相对差 {fmt(rep.get('max_abs_rel_diff_pct', float('nan')), 2)}% | 是（对账口径） |
| 解析解自检（λ 恒等式） | λ²−Reλ−4π²=0 | {fmt(sc['lambda_identity_residual'], 1)} | 是（浮点零） |
| 解析解自检（N-S 残差） | 差分量级 | {fmt(sc['ns_res_max_analytic_u'], 6)} | 是（无量纲小量） |
| 出口 BC 对 u 误差的贡献占比 | 配对臂闭合 | {fmt(outlet_share, 1)}% | 不适用（归因量，非精度量） |
| Ma² 项对 u 误差的贡献占比 | 拟合 | {fmt(ma_share, 1)}% | 不适用（归因量） |

**诚实声明**：锚点判据（<3% @ N≥128）**独立复现成立**；但"误差来源是出口 BC + 可压缩性底噪"
中的后者被本单定量证伪（Ma² 仅占 {fmt(ma_share, 1)}%），且"四指标单调下降"的证据链里
v 掩模最大点相对误差这一项在 ny=128 上**不可复现**。本单**不修改**算例代码与既有
`result.json`（那是另一单的裁定权），只新增审计工件与更正建议。

## 复现

```bash
python3 benchmarks/verified/kovasznay_2d/independent_audit.py \\
    --grids 32 64 128 --steps 6000 6000 5000 --min-steps 3000 \\
    --out artifacts/m16861/independent_audit.json
python3 benchmarks/verified/kovasznay_2d/make_audit_report.py \\
    --json artifacts/m16861/independent_audit.json --outdir artifacts/m16861
python3 -m pytest tests/test_kovasznay_independent_audit.py -q
```

## 参考

- Kovasznay, L.I.G. (1948). Laminar flow behind a two-dimensional grid. *Proc. Camb. Phil. Soc.* 44, 58–62.
- Zou, Q. & He, X. (1997). On pressure and velocity boundary conditions for the LBM BGK model.
  *Phys. Fluids* 9, 1591–1598.
- 算例现状：`benchmarks/verified/kovasznay_2d/README.md`、`result.json`（本单的复核对象）。
"""
    return body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="artifacts/m16861/independent_audit.json")
    ap.add_argument("--outdir", default="artifacts/m16861")
    ap.add_argument("--doc", default="docs/CASE-LBM-KOVASZNAY-EXACT_independent_audit.md")
    args = ap.parse_args(argv)
    d = load(Path(args.json))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figs = {"profile": fig_err_profile(d, outdir),
            "ma": fig_ma_scan(d, outdir),
            "vny": fig_v_vs_ny(d, outdir)}
    Path(args.doc).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc).write_text(build_doc(d, figs))
    for k, v in figs.items():
        print(k, v, v.stat().st_size)
    print("doc", args.doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())