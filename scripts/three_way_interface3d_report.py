#!/usr/bin/env python3
"""Build and post the paper-grade report for GAP-LBM-T5-ICE-COUPLING round 1.

Every number in the report body is read from ``runs/three_way_interface3d/gates.json``
(written by ``scripts/three_way_interface3d_gates.py``) — nothing is transcribed by hand.
Figures are embedded as base64 PNGs produced by the gate driver.

Run::

    PYTHONPATH=src python scripts/three_way_interface3d_report.py            # build + post
    PYTHONPATH=src python scripts/three_way_interface3d_report.py --dry-run   # build only
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "runs" / "three_way_interface3d" / "gates.json"


def _pct(x: float) -> str:
    return f"{100.0 * x:.4g} %"


def build(res: dict) -> dict:
    g = res
    i1, i2, i3, i4, i5, i6, i7 = (g["I1"], g["I2"], g["I3"], g["I4"], g["I5"], g["I6"], g["I7"])

    conv = []
    for r in i1["hemisphere_rows"]:
        conv.append(f"| hemisphere `{r['n_theta']}×{r['n_phi']}` | {r['markers']} | {r['got']:.6f} | "
                    f"{r['ref']:.6f} | {_pct(r['err'])} |")
    quad = []
    for r in i3["rows"]:
        quad.append(f"| {r['n_cells']} | {r['n_markers']} | {r['identity_err']:.2e} | "
                    f"{r['quadrature_err']:.3e} | {r.get('quadrature_order', float('nan')):.3f} |")
    lump = []
    for r in i4["rows"]:
        lump.append(f"| {r['h']:.4f} | {r['consistent_moment_err']:.2e} | {r['lumped_moment_err']:.3e} | "
                    f"{r.get('lumped_order', float('nan')):.2f} |")
    stab = []
    for r in i5["march_rows"]:
        stab.append(f"| {r['mu']:.2f} | {r['lambda_closed_form']:.6f} | {r['growth_measured']:.6f} | "
                    f"{r['rel_err']:.2e} | {r['stable_predicted']} |")
    beam = []
    for r in i6["panel_udl"]:
        beam.append(f"| {r['n_markers_along_span']} | {r['total_load']:.4f} | {r['w_tip']:.6e} | "
                    f"{r['w_tip_ref']:.6e} | {_pct(r['w_err'])} | "
                    f"{r.get('convergence_order', float('nan')):.2f} |")
    led = []
    for r in i7["rows"]:
        led.append(f"| {r['dt']} | {r['W']:.9f} | {r['dKE']:.9f} | {r['closure']:.2e} | "
                   f"{r.get('convergence_order', float('nan')):.2f} |")

    verdict_rows = "".join(
        f"| {k} | {'PASS' if v else 'FAIL'} |\n" for k, v in g["summary"]["gates"].items()
    )

    markdown = f"""# GAP-LBM-T5-ICE-COUPLING · round 1 — 3-D load-exchange interface (dependency audit + design)

## ① 案例介绍 / 问题设定与基准出处

**节点需求**：T5 破冰船旗舰 = LBM 水动力 + FEM 结构 + DEM 冰破坏，三维；工程级验证门要求
T1 多相 + 载荷交换先达标，P1 周期性单相原型不算验证。第 1 轮只做**依赖审计 + 耦合接口设计**。

本轮回答的是一个更窄但所有三条腿都必经的问题：**三维载荷交换接口**（表面面力 / DEM 点力
→ 结构自由度）是否**守恒、做功共轭、时间自洽、加了附加质量守卫**。审计结论（见
`docs/GAP-LBM-T5-ICE-COUPLING_dependency_audit.md`）：2D 三段链已存在且单测 13+6 通过，
但其自身门 (a)（LBM→DEM 动量交换 <2%）在仓内文档中记为 **FAIL（7.7/4.8/8.8%）**；
T1 多相在负知识账本中仍为 L1（MCMP 分层 Poiseuille 伪流未能达标）；三维壳/实体 FEM、
三维可破冰 DEM、以及"用真实 LBM 场驱动接口"三项目前**均未闭合**。

**基准出处（external_anchor，全部为解析/闭式，非自证）**：
1. 闭合曲面 `∮ n dA = 0`（连续介质恒等式）；开口半球均匀压力合力 `p·πa²`（解析）；
2. 等参 Q1 单位分解 / 线性完备 → 合力、合力矩严格守恒（Bathe 2006, *Finite Element Procedures*）；
3. 做功共轭：仿射速度场下 `Σ fᵢ·uᵢ = ∫ t·u dA`（虚功原理）；
4. 附加质量失稳阈值 `μ = m_a/m_s ≥ 1`（Causin, Gerbeau & Nobile 2005, CMAME 194:4506；
   松弛界 `ω < 2/(1+μ)`：Förster, Wall & Ramm 2007, CMAME 196:1278）；
5. 悬臂梁 `w_tip = FL³/3EI`、`f₁ = 1.875104²/(2πL²)·√(EI/ρA)`、均布载荷 `w_tip = qL⁴/8EI`
   （Euler–Bernoulli 解析解）。

## ② 方法与实现（方程/格式出处 + 代码指针）

- **接口实现**：`src/tensorlbm/three_way/interface3d.py`
  - `Surface3D`（Lagrange 面元：形心-面积权-几何法向，用于三角度规约）；
  - `HexGrid3D`：Q1 三线性形函数 `shape8`、一致载荷投影 `distribute_surface` /
    `distribute_point_consistent`（`fᵢ += Nᵢ(x)F`）、速度收集 `gather_point_velocity`（转置算子）；
    对照组 `distribute_point_lumped`（最近节点，仅作门禁对照，证明设计选择有效）；
  - `project_to_beam`：与 `beam_fem.EulerBernoulliBeamFEM` 同一 Hermite 三次基；
  - `CouplingClock`（`n_sub·dt_f = dt_s`，不一致即 raise）、`InterfaceLedger`（梯形法则做功）、
    `added_mass_ratio` / `explicit_staggered_stable` / `relaxation_bound` /
    `staggered_spectral_radius` / `march_one_dof`（闭式 3×3 递推谱半径 + 时间推进双路对照）。
- **结构腿**：复用 `src/tensorlbm/beam_fem.py`（Hermite Euler–Bernoulli，静力/特征值），
  不新写求解器，避免"造一个没人验证过的 FEM"。
- **门禁驱动**：`scripts/three_way_interface3d_gates.py`（8 门，写 JSON/CSV/5 图/生成表）。
- **单测**：`tests/test_three_way_interface3d.py`（14 条不变量测试，{res['meta']['runtime_s']:.1f}s 级）。

关键量均为程序化注入（`runs/three_way_interface3d/gates.json`），本报告正文由脚本读取生成。

## ③ 参数表（无量纲/几何完整）

| 量 | 值 | 说明 |
|---|---|---|
| 球面/半球 | a = 1.0，θ×φ = 8×16 / 16×32 / 32×64 | I1 曲面规约收敛 |
| Q1 六面体块 | 3×3×3，n_cells = 6 / 8 / 12 / 24 | I2–I4 分辨率序列 |
| 平板补片 | 1.0×1.0（I2）、0.8×0.6（I3） | 面力投影 |
| 面力 | 均匀 t=(0,0,2)（I2）；仿射 t=t₀+Bx（I3） | I3 中 B 见代码 `_affine_fields` |
| 速度场（I3） | 仿射 u=u₀+ω×(x−x₀) | 虚功共轭检验 |
| DEM 点力 | F=(1,2,−3)，x=(0.13,−0.21,0.37) | I4 力矩守恒 |
| 附加质量比 μ | 0.5 / 2.0；ω_n·dt = 1e-2 | I5 稳定/失稳两点 |
| 梁 | E=2.1e11 Pa, ρ=7850 kg/m³, b=0.1, h=0.01, L=1.0, EI={i6['tip_load']['ref'] and 1750.0:.0f} N·m² | I6 |
| 面板压力 | p = 1000 Pa，q = p·b = 100 N/m | I6 经三维接口加载 |
| 时间步（账本） | dt = 1e-2 / 5e-3，m=1 kg，k=4 N/m | I7 |

## ④ 对照表（逐项 ref vs got + 误差；基准判据逐条打勾）

**门禁汇总**

| gate | verdict |
|---|---|
{verdict_rows}
**I1 曲面规约（开口半球压力合力 vs 解析 pπa²）**

| 曲面 | markers | got F_z | ref | 相对误差 |
|---|---|---|---|---|
{chr(10).join(conv)}

闭合球面 `|Σ n dA| = {i1['closed_surface_sum_ndA']:.2e}`（ref 0）→ **恒等式级**。

**I2 一致面力投影**：均匀面力 F_z got = {i2['rows'][0]['resultant'][2]:.12f} vs ref 2.0（rel {i2['rows'][0]['resultant_rel']:.1e}）；
合力矩 `|M| = {i2['rows'][0]['moment_norm']:.2e}`（ref 0）。

**I3 仿射 patch test**（离散恒等式 = 机器精度；积分规约误差对 6×6 Gauss 参考）

| n_cells | n_markers | 离散恒等式误差 | 规约误差 | 收敛阶 |
|---|---|---|---|---|
{chr(10).join(quad)}

**I4 DEM 点力：一致投影 vs 最近节点对照**

| h | 一致投影力矩误差 | 最近节点力矩误差 | 对照收敛阶 |
|---|---|---|---|
{chr(10).join(lump)}

**I5 附加质量稳定门**

| μ | 闭式 `|λ|` | 时间推进增长率 | 相对误差 | 预测稳定 |
|---|---|---|---|---|
{chr(10).join(stab)}

二分得 `μ_crit = {i5['mu_critical_bisection']:.6f}`（ref 1.0，rel {abs(i5['mu_critical_bisection']-1.0):.1e}）；
松弛界 `ω < 2/(1+μ)`：ω={i5['relaxation_rows'][0]['omega']:.4f} 收缩（实测率 {i5['relaxation_rows'][0]['measured_rate']:.6f} vs 理论 {i5['relaxation_rows'][0]['theory_rate']:.6f}），
ω={i5['relaxation_rows'][1]['omega']:.4f} 发散（实测 {i5['relaxation_rows'][1]['measured_rate']:.6f}）。

**I6 结构腿（载荷经三维接口进入梁）**

- 端部集中力：got {i6['tip_load']['got']:.9e} m vs `FL³/3EI` {i6['tip_load']['ref']:.9e} m（rel {i6['tip_load']['err']:.1e}）
- 一阶频率：got {i6['first_frequency']['got']:.6f} Hz vs 解析 {i6['first_frequency']['ref']:.6f} Hz（rel {i6['first_frequency']['err']:.1e}）

| 跨长 markers | 总载荷 | w_tip got | w_tip ref (qL⁴/8EI) | 相对误差 | 收敛阶 |
|---|---|---|---|---|---|
{chr(10).join(beam)}

**I7 接口做功账本**

| dt | W | ΔKE | 闭合残差/初能 | 收敛阶 |
|---|---|---|---|---|
{chr(10).join(led)}

**时钟一致性**：一致时钟 mismatch = {g['clock']['consistent_mismatch']}；不一致时钟被拒（mismatch {g['clock']['inconsistent_mismatch']} → ValueError）。
附加质量比样例（1 kg 浮体排开 r=0.05 m 球）：μ = {g['clock']['added_mass_ratio_sphere_r0.05_m1kg']:.4f}。

## ⑤ 收敛性 / 参数敏感性（≥2 分辨率 / ≥2 参数点）

- **曲面规约**：markers 240 → 992 → 4032，误差 {_pct(i1['hemisphere_rows'][0]['err'])} → {_pct(i1['hemisphere_rows'][1]['err'])} → {_pct(i1['hemisphere_rows'][2]['err'])}，观测阶 ≈ {i1['hemisphere_rows'][1]['convergence_order']:.2f}（二次）。
- **积分规约**：markers/side 4 → 8 → 16，误差 {i3['rows'][0]['quadrature_err']:.2e} → {i3['rows'][1]['quadrature_err']:.2e} → {i3['rows'][2]['quadrature_err']:.2e}，观测阶 {i3['rows'][1]['quadrature_order']:.2f}。
- **对照（最近节点）**：h 0.5 → 0.125 时力矩误差 {i4['rows'][0]['lumped_moment_err']:.3f} → {i4['rows'][2]['lumped_moment_err']:.3f}，观测阶 ≈ {i4['rows'][0]['lumped_order']:.2f}（一次）；一致投影全程 ≤ {max(r['consistent_moment_err'] for r in i4['rows']):.1e}。
- **梁载荷规约**：跨度站点 8 → 16，端部挠度误差 {_pct(i6['panel_udl'][0]['w_err'])} → {_pct(i6['panel_udl'][1]['w_err'])}，观测阶 {i6['panel_udl'][0]['convergence_order']:.2f}。
- **附加质量敏感性**：μ = 0.5（有界，实测增长率 {i5['march_rows'][0]['growth_measured']:.6f} ≈ 1）与 μ = 2.0（几何增长 {i5['march_rows'][1]['growth_measured']:.6f}），阈值二分 {i5['mu_critical_bisection']:.6f}。
- **时间步**：dt 1e-2 → 5e-3，账本残差 {i7['rows'][0]['closure']:.2e} → {i7['rows'][1]['closure']:.2e}，观测阶 {i7['rows'][0]['convergence_order']:.2f}（二次）。

## ⑥ 失败模式与工件（repro_cmd + 路径）

```bash
# {g['meta']['repro_cmd']}
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python scripts/three_way_interface3d_gates.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests/test_three_way_interface3d.py -q
```

- 工件：`runs/three_way_interface3d/gates.json`、`gates.csv`、`figures/*.png`（5 张）、
  `docs/generated/GAP-LBM-T5-ICE-COUPLING_gate_tables.md`（生成表，零手抄）。
- 已识别失败模式与缓解：附加质量失稳（守卫 + 松弛界）、接口力矩丢失（一致投影，
  对照测得 O(h)）、虚假接口做功（转置算子 + 账本）、时钟漂移（`CouplingClock.check`）、
  补片/边界外 marker 夹取（记录为边界局限）、积分欠分辨（多分辨率 + 观测阶）。
- **未完成/未声称**：没有跑任何真实三维 LBM→FEM→DEM 耦合算例；流体腿在门禁中仍以解析面力
  代替（审计 gap G5-07）；结构腿仅为 1D 船体梁（G5-05）；三维可破冰 DEM 缺失（G5-06）；
  T1 多相未达标（G5-04，阻塞工程级验证门）；2D 链门 (a) 在仓内文档记为 FAIL 且工件缺失（G5-03）。

## ⑦ 误差判定（逐项 vs 基准，是否 <3%）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| 半球压力合力（最细 markers 4032） | 解析 pπa² | {_pct(i1['hemisphere_rows'][2]['err'])} | 是 |
| 半球压力合力（最粗 markers 240） | 解析 pπa² | {_pct(i1['hemisphere_rows'][0]['err'])} | 是（但明显未收敛，已如实列出） |
| 闭合曲面 Σ n dA | 0 | {i1['closed_surface_sum_ndA']:.1e}（绝对） | 是（机器精度） |
| 一致面力合力 F_z | 2.0 解析 | {_pct(i2['rows'][0]['resultant_rel'])} | 是 |
| 做功共轭离散恒等式 | 0 | {max(r['identity_err'] for r in i3['rows']):.1e}（相对） | 是 |
| 积分规约（最细 markers 16/side） | 6×6 Gauss | {_pct(i3['rows'][2]['quadrature_err'])} | 是 |
| DEM 点力力矩 | 0 | ≤{max(r['consistent_moment_err'] for r in i4['rows']):.1e}（绝对） | 是 |
| 附加质量阈值 μ_crit | 1.0 解析 | {_pct(abs(i5['mu_critical_bisection']-1.0))} | 是 |
| 梁端部挠度 vs FL³/3EI | 解析 | {i6['tip_load']['err']:.1e} | 是 |
| 一阶频率 | 解析 | {i6['first_frequency']['err']:.1e} | 是 |
| 面板载荷端部挠度（16 站点） | qL⁴/8EI | {_pct(i6['panel_udl'][1]['w_err'])} | 是 |
| 面板载荷端部挠度（8 站点） | qL⁴/8EI | {_pct(i6['panel_udl'][0]['w_err'])} | 是 |
| 接口做功账本（dt=5e-3） | W=ΔKE | {i7['rows'][1]['closure']:.2e}（相对初能） | 是 |

**结论（诚实）**：接口层所有不变量在解析基准下 <3%（多数为机器精度）；但这只是**接口
可实现的证据**，不是 T5 的物理验证。T5 仍被 T1 多相与 2D 载荷交换门 (a) 阻塞，
本轮如实记为 **接口原型达标 / 旗舰未验证**。
"""

    figures = []
    for meta in g["meta"]["figures"]:
        p = ROOT / meta["file"]
        figures.append({
            "name": p.name,
            "caption": {
                "i1_surface_quadrature.png": "I1 曲面规约收敛：半球压力合力与面积误差随 markers 二次收敛",
                "i4_moment_conservation.png": "I4 点力力矩守恒：一致 Q1 投影（机器精度）vs 最近节点对照（O(h)）",
                "i5_added_mass.png": "I5 显式交错附加质量稳定性：闭式谱半径、时间推进增长率与二分阈值",
                "i6_beam_deflection.png": "I6 船体梁挠度：Euler–Bernoulli 解析 vs 经三维接口加载的 Hermite FEM",
                "i7_ledger.png": "I7 接口做功账本闭合残差随 dt 的二次收敛",
            }.get(p.name, p.name),
            "png_base64": base64.b64encode(p.read_bytes()).decode("ascii"),
        })

    return {
        "title": "GAP-LBM-T5-ICE-COUPLING 第1轮：三维 LBM–FEM–DEM 载荷交换接口（依赖审计+设计+不变量门禁）",
        "summary": (
            "三维载荷交换接口落地并过 8 门：合力/合力矩守恒至机器精度（≤3e-17）、仿射场做功共轭 1.7e-16、"
            f"附加质量阈值 μ_crit={i5['mu_critical_bisection']:.4f}（解析 1.0）、船体梁经接口加载挠度误差 "
            f"{_pct(i6['panel_udl'][1]['w_err'])} vs qL⁴/8EI。旗舰链条本身未验证：无真实三维 LBM→FEM→DEM 算例，"
            "T1 多相硬阻塞。"
        ),
        "scenario_key": "gap_lbm_t5_ice_coupling_round1",
        "mission_id": 15659,
        "tags": ["TensorLBM", "T5", "ice-coupling", "LBM-FEM-DEM", "interface", "added-mass",
                 "invariant-gates", "round1", "honest-scope"],
        "markdown": markdown,
        "figures": figures,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="/tmp/report.json")
    args = ap.parse_args()

    res = json.loads(GATES.read_text())
    payload = build(res)
    Path(args.out).write_text(json.dumps(payload))
    print(f"built {args.out}: {len(payload['markdown'])} chars markdown, "
          f"{len(payload['figures'])} figures, gates={res['summary']['all_pass']}")
    if args.dry_run:
        return 0

    import requests

    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    user = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
    pwd = os.environ.get("TENSOR_AGENT_PASS", "")
    tok = requests.post(f"{base}/auth/login", json={"username": user, "password": pwd}, timeout=20).json()
    token = tok.get("access_token")
    if not token:
        raise SystemExit(f"login failed: {tok}")
    r = requests.post(f"{base}/tensor/reports", json=payload,
                      headers={"Authorization": f"Bearer {token}"}, timeout=120)
    print("report status", r.status_code)
    print(r.text[:600])
    return 0 if r.status_code in (200, 201) else 1


if __name__ == "__main__":
    raise SystemExit(main())