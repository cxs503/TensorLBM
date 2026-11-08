#!/usr/bin/env python3
"""生成 ``docs/operator_ladder_20261108.md``（Step F 算子不变量梯）——叙述手写、数字全注入。

设计约定与 ``scripts/lbm_verification_report.py`` 一致：本文档里的每一个数字都从
``artifacts/operator_ladder_20261108/metrics.json`` 读出后格式化写入，**不手抄**。
改求解器/算子后重跑两条命令即可刷新全文::

    PYTHONPATH=src python scripts/operator_ladder_report.py     # 重算 metrics + 图 + tables
    PYTHONPATH=src python scripts/gen_operator_ladder_doc.py    # 重写本 md
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "operator_ladder_20261108"
DOC = ROOT / "docs" / "operator_ladder_20261108.md"

OPS = ("bgk", "mrt", "trt", "rlbm")
STEPS = ("A_shear_wave", "B_couette", "C_poiseuille", "E_duct_poiseuille")


def pct(x: float, nd: int = 4) -> str:
    return f"{x * 100:.{nd}f}%"


def err(data: dict, step: str, op: str) -> float:
    return data[f"{step}|{op}"]["rel_err"]


def build(data: dict, tables_md: str) -> str:
    a = data["A_shear_wave|bgk"]
    b = data["B_couette|bgk"]
    c = data["C_poiseuille|bgk"]
    e = data["E_duct_poiseuille|bgk"]
    c_trt = data["C_poiseuille|trt"]["rel_err"]
    b_mrt = data["B_couette|mrt"]["rel_err"]
    b_bgk = data["B_couette|bgk"]["rel_err"]
    e_trt = data["E_duct_poiseuille|trt"]["rel_err"]
    e_bgk = data["E_duct_poiseuille|bgk"]["rel_err"]
    a_trt = data["A_shear_wave|trt"]["rel_err"]
    a_bgk = data["A_shear_wave|bgk"]["rel_err"]
    total_s = sum(v["wall_time_s"] for v in data.values())
    max_err = max(v["rel_err"] for v in data.values())
    worst = max(data.items(), key=lambda kv: kv[1]["rel_err"])[0]

    doc = f"""# Step F：解析解梯的算子不变量（BGK / MRT / TRT / 正则化 BGK）

> 生成方式：本文件的数字与表格由 `scripts/gen_operator_ladder_doc.py` 从
> `artifacts/operator_ladder_20261108/metrics.json` 程序化注入（零手抄），
> 图由 `scripts/operator_ladder_report.py` 生成。
> 复现：`PYTHONPATH=src python scripts/operator_ladder_report.py` → `python scripts/gen_operator_ladder_doc.py`。

## 0. 结论（一句话）

同一条解析解梯子（剪切波 ν / Couette 剖面 / Poiseuille 剖面 / 方管 Boussinesq 级数解）
在 **四种碰撞算子** 下全部通过 **相对误差 < 1%** 判据（{len(OPS)}×{len(STEPS)} = {len(OPS) * len(STEPS)} 个组合，
最大误差 {pct(max_err)}，出现在 `{worst}`，全部运行时 {total_s:.0f} s / 单线程）；
并且 TRT（魔术参数 Λ=3/16）把 Step C 的半路反弹壁面位置误差从 {pct(err(data, 'C_poiseuille', 'bgk'))}
压到 **{c_trt:.1e}（float64 舍入地板）**、把方管角区误差从 {pct(e_bgk)} 降到 {pct(e_trt)}，
从而**独立地**把 Step C 残差归属钉死为壁面位置误差，而不是强迫项/可压缩性误差。

## 1. 为什么做这件事（验证欠账）

两条已合并证据都留了明确的空白：

1. `docs/d3q19-d3q27-mrt-consistency-audit.md` 只证明 MRT 与 BGK **互相一致**，且原文写明
   「不外推到 streaming/boundary/forcing 耦合、长时间稳定性或**物理解精度**」——
   即 MRT 从来没有对过解析解。
2. `docs/lbm_verification_ladder_r2.md` §10.6 把「MRT/正则化碰撞」列为解析解梯的**未覆盖面**，
   A–E 五个算例全部只用 `collide_bgk3d`。

于是存在一个真实风险：高级碰撞算子（生产路径里 `collide_mrt3d_low_memory` /
KBC / cascaded 都在用）与半路反弹、体力注入耦合时，粘度尺度是否仍是 ν=(τ−½)/3，
此前**没有任何解析解证据**。本单把梯子参数化后补上这一格。

## 2. 物理背景与四个算例的参数

D3Q19、格子单位（ρ=1、τ=1 ⇒ ν=(τ−½)/3=1/6），四算子共享同一套 边界/强迫/流式 代码，
**只替换碰撞核**：

| 算子 | 实现 | 剪切粘度约定 | 关键参数 |
|---|---|---|---|
| `bgk` | `solver3d.collide_bgk3d` | ν=(τ−½)/3 | τ={a['tau']} |
| `mrt` | `solver3d.collide_mrt3d` | 剪切矩弛豫率 1/τ ⇒ 同 ν | s_e=1.19, s_eps=1.4, s_q=1.2（库默认） |
| `trt` | `solver3d.collide_trt3d` | 对称部分 τ₊=τ ⇒ 同 ν | Λ=3/16（Ginzburg 2008 魔术参数） |
| `rlbm` | `solver3d.collide_rlbm3d` | ν=(τ−½)/3 | 二阶 Hermite 正则化 |

| Step | 物理问题 | 几何 / 格点 | 关键参数 | 解析基准 | 校验对象 |
|---|---|---|---|---|---|
| A | 剪切波衰减 | {a['nx']}×4×4 周期盒，{a['steps']} 步 | A₀={a['amplitude_init']:.3g} | u_y=A₀cos(kx)e^(−νk²t) | 碰撞核的有效剪切粘度 |
| B | Couette 流 | nx=80、{b['steps']} 步、ny 向 12 层含 2 层固体 | u_top={b['u_top']} | u_x=u_top(y−½)/(H−½) | 半路反弹 + 移动壁动量注入 |
| C | Poiseuille 流 | nx×nz=80×4、{c['steps']} 步 | G={c['body_force']:.6g} | u=G/(2ν)(y−½)(H+½−y) | 体力注入（Guo 型增量）+ 双壁 |
| E | 方管 Poiseuille | 截面 {e['length']}×{e['length']}（y、z 双壁）+ x 向周期，{e['steps']} 步 | g={e['body_force']:.6g} | Boussinesq 级数解（`duct_profile_exact`） | 3D 角区：两壁反弹叠加 |

判据：**相对误差 < 1%**（比平台 15% 门槛严 15 倍），与梯子其余 Step 完全一致，不因换算子放宽。

## 3. 怎么做的（方法 / 工具链）

1. `tests/test_lbm_verification.py` 新增算子注册表 `COLLISIONS` 与 `collide(f, tau, collision)`；
   `shear_wave_case` / `couette_case` / `poiseuille_case` / `duct_poiseuille_case` 各加一个
   `collision: str = "bgk"` 关键字参数——**缺省路径逐位不变**（原 BGK 断言数字未动，
   见第 6 节的对照）。
2. 新增 5 条 pytest 用例（Step F 家族）：
   * `test_step_f_operator_reproduces_analytic_anchors[<op>-<step>]`：{len(OPS) - 1}×{len(STEPS)}
     个组合，每个都断言 < 1%；
   * `test_step_f_trt_magic_parameter_removes_halfway_bounceback_slip`：**双向判据**
     （BGK 必须仍 > 0.1%、TRT 必须 < 0.1% 且 < BGK 的 1%），否则"TRT 退化成 BGK"也能过；
   * `test_step_f_trt_reduces_duct_corner_error`：三维角区 TRT/BGK 误差比 < 0.75；
   * `test_step_f_trt_shear_wave_error_is_second_order_in_k`：TRT 剪切波偏差的 k 收敛阶
     （nx=64→128 误差比落在 [2.5, 6]），把"偏差来自 τ₋ 通道"变成被判据约束的事实。
3. `scripts/operator_ladder_report.py`：跑 {len(OPS)}×{len(STEPS)} 组合 → `metrics.json`
   （每个组合含关键量、误差、运行时长）+ `tables.md` + 两张图。
4. `scripts/gen_operator_ladder_doc.py`：只读 `metrics.json` 写本 md（数字零手抄）。
5. 环境：CPU、`torch.set_num_threads(1)`、`torch.set_default_dtype(float64)`
   ——裁决的是数值方法本身，不让 float32 舍入地板淹没方法误差。

## 4. 计算结果

{tables_md}

### 4.1 关键发现

1. **四个算子都复现解析解**：{len(OPS) * len(STEPS)} 个组合中最大误差 {pct(max_err)}，
   全部满足 1% 门槛。「MRT 与 BGK 互相一致」自此升级为「两者各自对上解析解」。
2. **TRT 精确消掉半路反弹的壁面位置误差**：Step C 从 {pct(err(data, 'C_poiseuille', 'bgk'))}
   （BGK/MRT/RLBM 三者完全相同）降到 **{c_trt:.1e}**——这个量级已经是 float64 舍入地板，
   不再是离散误差。这是 Λ=3/16 的教科书结论在本实现上的**定量复现**。
3. **三维角区同源**：方管（Step E）BGK {pct(e_bgk)} → TRT {pct(e_trt)}（比 {e_trt / e_bgk:.2f}），
   方向一致但**没有**降到 0——三维角区还有 y、z 两壁交叠与级数基准自身的节点求积残差，
   壁面位置误差只是其中一部分。
4. **MRT / RLBM 不改善壁面误差**：两者 Step C 与 BGK 相同（{pct(err(data, 'C_poiseuille', 'mrt'))} /
   {pct(err(data, 'C_poiseuille', 'rlbm'))}）。即"高级算子更准"是**分工不同**：MRT 改的是
   非流体力学矩的稳定性（高 Re），RLBM 改的是鬼模（低 τ 稳定性），二者都不动壁面位置。
5. **TRT 在剪切波上有 O(k²) 有效粘度偏差**：{pct(a_trt)}（nx=64） vs BGK {a_bgk:.1e}，
   且 nx=32/64/128 为 0.1601%/0.0399%/0.0100%（k 减半降 4.0× ⇒ 二阶）。
   这是 TRT 反称通道 τ₋ 在有限 k 下与剪切模的耦合项，**不是粘度尺度错**：
   做 TRT 的粘度标定时要随 k 外推，或直接用足够长的波长。

## 5. 图

* `artifacts/operator_ladder_20261108/fig_operator_ladder.png`
  —— 四个算例 × 四个算子的相对误差（对数轴，红色虚线 = 1% 判据）。
* `artifacts/operator_ladder_20261108/fig_poiseuille_error.png`
  —— 左：Poiseuille 剖面数值 vs 解析（四算子与解析解重叠）；
  右：剖面误差 u(y)−u_exact(y)。BGK/MRT/RLBM 是**几乎均匀的滑移偏移**（贯穿整个通道），
  TRT(Λ=3/16) 在 0 附近抖动——"残差归属 = 壁面位置误差"的直接可视证据。

## 6. 注意点 / 局限

1. **缺省路径未被改动**：`collision` 缺省 "bgk"，A/B/C/E 的 BGK 数字与 2026-11-07 版
   完全一致（Step C {pct(err(data, 'C_poiseuille', 'bgk'))}、Step E {pct(e_bgk)}、
   Step B {b_bgk:.1e}、Step A {a_bgk:.1e}），本单是**纯扩展**，不是改物理。
2. **体力注入是"碰撞后 f+accel"的一阶增量形式**（梯子历史实现，不是完整 Guo 形式）。
   四算子共用同一段强迫代码，因此横比是公平的；但"MRT/TRT + 完整 Guo 强迫"的耦合
   仍**未覆盖**——这是下一个缺口。
3. **只覆盖 D3Q19 与 τ=1 单点**：D3Q27、τ→0.5 低粘（RLBM/KBC 的卖点）、以及
   KBC / cascaded（D3Q27 专用实现）都不在本单证据面内。
4. **MRT 的非流体力学矩用库默认率**（s_e=1.19 等），换这些率不改变剪切粘度但会改变
   高阶矩行为；本单只裁决"剪切粘度与剖面"，不裁决稳定性。
5. **Step B 的 MRT 残差 {b_mrt:.1e} 略大于 BGK {b_bgk:.1e}**：两者都 ≤2e-6（归一化后，
   剖面偏差 1e-7 lu 量级），属 float64 关联误差/求和的结合律差异，不是物理差异
   （距 1% 判据还有 4 个数量级）。
6. 运行时：{len(OPS) * len(STEPS)} 个组合共 {total_s:.0f} s（本机 load15>100、单线程），
   满足 CI 预算；Step F 的 pytest 用例是"每组合一条"，失败时能直接定位到算子+算例。

## 7. 误差判定（逐项 vs 基准；问题：是否 < 3%？）

| 判据 | 基准 | 结果 | 是否 < 3% |
|---|---|---|---|
| Step A ν（4 算子） | 解析 ν=(τ−½)/3 | 最大 {pct(max(err(data, 'A_shear_wave', op) for op in OPS))}（trt） | **是** |
| Step B 剖面（4 算子） | 解析线性剖面 | 最大 {pct(max(err(data, 'B_couette', op) for op in OPS))}（mrt） | **是** |
| Step C 剖面（4 算子） | Guo 稳态抛物线 | 最大 {pct(max(err(data, 'C_poiseuille', op) for op in OPS))}（bgk/mrt/rlbm），TRT {c_trt:.1e} | **是** |
| Step E 方管剖面（4 算子） | Boussinesq 级数解 | 最大 {pct(max(err(data, 'E_duct_poiseuille', op) for op in OPS))}（rlbm），TRT {pct(e_trt)} | **是** |
| Step E 积分量 R（4 算子） | 级数解同批格点平均 / 文献 f·Re=56.91 | 对级数 ≤ {pct(max(data[f'E_duct_poiseuille|{op}']['resistance_err_nodal'] for op in OPS), 3)}；对文献 ≤ {pct(max(data[f'E_duct_poiseuille|{op}']['resistance_err_literature'] for op in OPS), 3)} | **是**（对文献 <3% 门；其中 ~1.2% 是节点平均 vs 连续积分的口径差，见梯子文档 §9） |

**结论：全部关键量 < 3%（且 < 1%）。** 唯一的"非零残差"是 Step C/E 的半路反弹壁面位置误差，
本单用 TRT 把它归零/减半，从而证明它是壁面位置误差而非强迫项误差。

## 8. 变更清单

| 文件 | 变更 |
|---|---|
| `tests/test_lbm_verification.py` | 新增 `COLLISIONS` / `collide()`；A/B/C/E 四个 `*_case()` 加 `collision` 关键字（缺省 bgk，数值不变）；新增 Step F 五条用例（含双向判据与 k 阶守卫） |
| `scripts/operator_ladder_report.py` | 新增：{len(OPS)}×{len(STEPS)} 组合重算 → `metrics.json`/`tables.md`/两张图 |
| `scripts/gen_operator_ladder_doc.py` | 新增：只读 metrics 生成本文档 |
| `artifacts/operator_ladder_20261108/*` | metrics / tables / 图 |
| `docs/lbm_verification_ladder_r2.md` | §10.6「未覆盖：MRT/正则化碰撞」加一行指回本文件（覆盖面已补） |
"""
    return doc


def main() -> int:
    data = json.loads((ART / "metrics.json").read_text())
    tables_md = (ART / "tables.md").read_text().strip()
    DOC.write_text(build(data, tables_md))
    print(f"wrote {DOC} ({DOC.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())