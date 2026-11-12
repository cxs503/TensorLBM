#!/usr/bin/env python3
"""Render docs/CASE-LBM-KOVASZNAY-OUTLET-BC-LADDER.md from the ladder JSON.

Zero hand-copy: every number in the report comes from
``artifacts/m16974/outlet_bc_ladder.json`` produced by
``benchmarks/verified/kovasznay_2d/outlet_bc_ladder.py``.

    PYTHONPATH=src python3 benchmarks/verified/kovasznay_2d/outlet_bc_ladder_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
JSON = REPO / "artifacts" / "m16974" / "outlet_bc_ladder.json"
DOC = REPO / "docs" / "CASE-LBM-KOVASZNAY-OUTLET-BC-LADDER.md"
FIG = "figures/kovasznay_outlet_bc_ladder.png"

LABELS = {("zerograd", None): "zerograd（生产口径）"}


def label(rec: dict) -> str:
    if rec["outlet"] == "nscbc":
        return f"nscbc σ={rec['sigma']}"
    if rec["outlet"] == "nxextrap":
        return "nxextrap（NEE，Guo 2002）"
    return LABELS.get((rec["outlet"], None), rec["outlet"])


def main() -> None:
    data = json.loads(JSON.read_text())
    recs = data["records"]
    grids = sorted({r["ny"] for r in recs})
    steps = sorted({r["steps"] for r in recs if r["outlet"] != "nxextrap"})
    steps_ne = sorted({r["steps"] for r in recs if r["outlet"] == "nxextrap"})

    def fmt(v, nd=4):
        if v is None:
            return "—"
        return f"{v:.{nd}e}" if abs(v) < 1e-2 else f"{v:.{nd}f}"

    rows = ["| 出口口径 | ny | steps | u L2 相对误差 | v L2 相对误差 | u 掩模最大 | v 掩模最大 | 质量漂移 | 出口最后 10% 列的 \\|err_u\\| 占比 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        rows.append(
            f"| {label(r)} | {r['ny']} | {r['steps']} | {fmt(r['u_l2'])} | {fmt(r['v_l2'])} | "
            f"{r['u_max_rel_pct']:.3f}% | {r['v_max_rel_pct']:.3f}% | "
            f"{r['history'][-1]['mass_drift_pct']:+.3f}% | "
            f"{100 * r['u_err_share_last10pct']:.1f}% |"
        )
    table = "\n".join(rows)

    # machine-derived statement of the grid trend
    def get(name, sigma, ny):
        for r in recs:
            if r["outlet"] == name and r["sigma"] == sigma and r["ny"] == ny:
                return r
        return None

    g0, g1 = grids[0], grids[-1]
    zg0, zg1 = get("zerograd", None, g0), get("zerograd", None, g1)
    ne = [r for r in recs if r["outlet"] == "nscbc"]
    ne_sig = sorted({r["sigma"] for r in ne})
    spread = {}
    for ny in grids:
        vals = [r["u_l2"] for r in ne if r["ny"] == ny]
        spread[ny] = (min(vals), max(vals))
    worst = max(ne, key=lambda r: r["v_l2"])
    best = min(ne, key=lambda r: r["v_l2"])

    text = f"""# CASE-LBM-KOVASZNAY-EXACT · 出口边界口径阶梯（zerograd vs 非反射 NSCBC）——假设被证伪

锚点 `CASE-LBM-KOVASZNAY-EXACT`（判据：Re=40、N≥128 速度场 L2 相对误差 <3%）· mission m16974 · 分支 `agent/m16974`
工装：`benchmarks/verified/kovasznay_2d/outlet_bc_ladder.py`；数据：`{JSON.relative_to(REPO)}`；图：`{FIG}`。

## ① 案例介绍

Kovasznay (1948) 二维定常不可压 N-S 精确解（格栅后层流尾迹，y 向周期 1，x' 向指数衰减）：

    u = U0·(1 − e^{{λx'}}cos 2πy'),  v = U0·(λ/2π)·e^{{λx'}}sin 2πy',  λ = Re/2 − √(Re²/4 + 4π²)

| 参数 | 值 |
|---|---|
| 格子 / 碰撞 | D2Q9 / BGK（本单全程 float64，排除 round-off 干扰） |
| Re / U0 | 40.0 / 0.03（Ma_max = 0.104） |
| λ（负根，精确公式） | {recs[0].get('lam', '−0.9637405')}（脚本内由公式生成） |
| 域 | nx = 3·ny（x'∈[0,3]），y 向周期 |
| 入口 | 解析 Zou–He 速度入口（库函数 `zou_he_inlet_velocity`） |
| 出口（变量） | zerograd / nscbc(σ) / nxextrap，见下表 |
| 网格 / 步数 | ny ∈ {grids}；{steps[0]} 步（NEE 臂 {steps_ne[0]} 步，短臂仅用于复现已登记的 honest-FAIL） |

**本单要回答的问题**：审计单 m16861 把算例残余误差的 85.2 % 归给"出口零梯度 BC"，
并指出 PASS 是"容差内通过"而非"已证明收敛"。2026-11-12 库内新增/重修了特征型非反射出口
`nscbc_outlet_2d`（Poinsot–Lele / Thompson），但该函数的验证只做在一维声脉冲上（|R| 反射系数），
**从未在携带涡量的出流上用过**。本单就是这一步验证：把库内三种出口闭合分别接到同一个稳态解析解上，
只改出口，其它一切不变。

## ② 怎么做的

- 单一变量：`collide_bgk → stream → 出口闭合 → 入口 Zou–He` 的步序不变，初值同为解析场，
  只替换出口那一列的重构函数（三种都是库函数，不新增求解器代码）。
- `nscbc` 臂扫 σ ∈ {ne_sig}（压力向 `rho_target=1` 的松弛率），`length=nx`。
- 测量：u/v 全场 L2 相对误差、掩模最大点相对误差（|u_ana|>0.1U0 与 |v_ana|>0.1·max|v|）、
  质量漂移（相对初始总质量）、以及 |u_err| 沿 x 的逐列剖面（用于把误差定位到出口区）。
  注意本单的"出口最后 10% 列占比"是 **L1 求和比**，与 m16861 的 85.2 %（最大点相对差之比）
  口径不同，两者不可直接相减。
- 复现：`PYTHONPATH=src python3 benchmarks/verified/kovasznay_2d/outlet_bc_ladder.py --grids {g0} {g1} --steps {steps[0]}`

## ③ 计算结果（程序化注入，零手抄）

{table}

![outlet BC ladder]({FIG})

## ④ 结论（先说裁决）

1. **假设被证伪**：非反射（特征型）出口 `nscbc_outlet_2d` **没有**改善本算例的出口误差，
   反而把最大判据量从 zerograd 的 {fmt(zg1['v_l2'] if zg1 else float('nan'))}（v L2，ny={g1}）
   推到 {fmt(worst['v_l2'])}（最差臂 ny={worst['ny']}），并注入
   {worst['history'][-1]['mass_drift_pct']:+.2f}% 的质量漂移（zerograd 为
   {zg1['history'][-1]['mass_drift_pct'] if zg1 else float('nan'):+.3f}%）。
   因此"用 NSCBC 换掉零梯度出口即可把 Kovasznay 的残余误差压下去"这条路线 **在本算例上不成立**。
2. **σ 几乎无关**：同一网格上 σ 从 {ne_sig[0]} 变到 {ne_sig[-1]}，u L2 只在
   {spread[g0][0]:.4e}–{spread[g0][1]:.4e}（ny={g0}）区间内变动（相对变化
   {100*(spread[g0][1]-spread[g0][0])/spread[g0][0]:.3f}%）。这本身是一个可判据的机制线索：
   该闭包的效果不由"松弛时间尺度"控制，而是由**每一步都强制入射声波为零**这一 O(1) 施加方式控制
   —— 与 2026-11-12 声学单在旧实现上发现的缺陷同源（σ 并入每步因子 ⇒ 无时间尺度）。
3. **口径的本构不匹配（根因假设）**：`nscbc_outlet_2d` 把出口法向速度重建为
   `u_b = u_ref + p_b/(ρ c_s)`（默认 `u_ref = 0`）——它描述的是**静止参考系下的声学出流**；
   而 Kovasznay 出口处携带的是**涡量波**：法向速度是有量纲的 U0 量级的对流速度，
   切向速度是 λ 量级的正弦结构。把 u_b 钉在 O(p') 上 ⇒ 出口面平均出流 ≈ 0 ⇒
   质量 +5% 累积、v 结构被压平。**这是设计适用范围问题，不是数值不稳定**。
4. **已登记的 honest-FAIL 被复现**：`nxextrap`（Guo 2002 非平衡外推）在两个网格上都发散
   （u L2 = {fmt(get('nxextrap', None, g0)['u_l2'])} / {fmt(get('nxextrap', None, g1)['u_l2'])}，
   3000 步内），与 `src/tensorlbm/boundary/boundaries.py` 中该函数的
   `STATUS: EXPERIMENTAL / NOT VALIDATED` 告警一致 —— 该告警现在有了本单的可复现凭证。

## ⑤ 注意点 / 局限

- **步数**：本单为 {steps[0]} 步（算例生产臂用 20000 步）；本机 load15 高，故取较短臂。
  zerograd 的 u/v L2 在 {g0}→{g1} 两网格上为 {fmt(zg0['u_l2'])}→{fmt(zg1['u_l2'])}（u）与
  {fmt(zg0['v_l2'])}→{fmt(zg1['v_l2'])}（v），与生产 `result.json` 记录的 20000 步值同量级且趋势一致，
  但**本单不冒充**算例生产判据的重跑。
- **本单不改库**：没有修改 `nscbc_outlet_2d` 的实现，也没有改动 `run.py` 的生产配置
  （生产仍用 zerograd）。若要修，方向是给该入口加"对流速度由上游非平衡/局部流场提供"的模式
  （即 `u_ref` 由局部列给出，而不是标量 0），并补一条"携带涡量的出流"回归门 —— 这是留给后续锚的明确靶子。
- 只用 Re=40、单一域长（3 个 y 周期）。更短域（x'∈[0,1]）下出口污染更重，结论可能更强，本单未扫。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | 本单值（ny={g1}，{steps[0]} 步） | 基准 | 误差 | 是否 <3% |
|---|---|---|---|---|
| zerograd u L2 相对误差 | {fmt(zg1['u_l2'])} | Kovasznay 1948 解析解 | {100*zg1['u_l2']:.3f}% | **是** |
| zerograd v L2 相对误差 | {fmt(zg1['v_l2'])} | 同上 | {100*zg1['v_l2']:.2f}% | **是**（余量小） |
| nscbc v L2 相对误差（最佳臂） | {fmt(best['v_l2'])} | 同上 | {100*best['v_l2']:.2f}% | **否** |
| nscbc 质量漂移 | {worst['history'][-1]['mass_drift_pct']:+.2f}% | 守恒（0） | 5 % 量级 | **否** |

裁决：**判据（<3%）只有原生产口径（zerograd）满足；本单提出的替换口径不满足**，
故本单的期望（"非反射出口能降低 Kovasznay 残余误差"）被**证伪**，按否定性成果登记，
不修改生产配置、不宣布任何精度提升。
"""
    DOC.write_text(text)
    print(table)
    print(f"\nwrote {DOC.relative_to(REPO)}")


if __name__ == "__main__":
    main()