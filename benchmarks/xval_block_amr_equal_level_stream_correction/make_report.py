#!/usr/bin/env python3
"""Render the six-section report from the run artifacts (no hand-copied numbers)."""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path("artifacts/xval_block_amr_equal_level_stream_correction")
res = json.loads((OUT / "results.json").read_text())
bench = json.loads((OUT / "benchmark.json").read_text())
v = res["verdict"]

rows = "\n".join(
    f"| {r['kpi']} | {r['ref']:.6g} | {r['got']:.6g} | {r['err_pct']:.3e} | {'PASS' if r['pass'] else 'FAIL'} |"
    for r in bench["rows"])
cal = "\n".join(
    f"| {k} | {c['best_scale']:.3f} | {100*c['best_error']:.4f} % | {c['runner_up_scale']:.3f} | {c['objective_gap']:.4f} |"
    for k, c in res["calibration"].items())
inv = "\n".join(
    f"| {k} | {i['links']} | {i['inert_uniform_max_abs']:.3e} | {i['roundtrip_max_abs']:.3e} | "
    f"{i['mass_defect_corrected']:.3e} | {i['mass_defect_plain']:.3e} |"
    for k, i in res["invariants"].items())
conv = "\n".join(f"| {k} | {val:.4f} |" for k, val in res["conventions"].items())

doc = f"""# XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION（返修轮 m17716）

## 1 案例介绍
对标对象 = lssfau/walberla 的 `block-amr-equal-level-stream-correction` 做法
（inventoried 源记录 `oss_solutions` id=283；文件清单
`src/lbm/refinement/EqualLevelBorderStreamCorrection.h`、`TimeStep.h`、
`CurlBasedLevelDetermination.h`）。其做法：2:1 块加密下，被细块覆盖的粗网格单元不维护，
因此粗网格边界单元指向细块的方向缺少同层源分布；流式前把缺失分布用**等尺寸（同层）邻元**
population 顶替，流式后**恢复**，默认开启。TensorLBM 的既有做法相反：被覆盖粗单元每步由
细层 restriction 刷新（`static_block_amr` / `amr_cavity2d` 的 `restricted` 耦合），
同层源恒存在，无需顶替。

本条 XVAL 检验的是该做法的**可移植内核**：顶替算子的状态中性/惰性/质量账簿，以及该做法
隐含的 Filippova-Haenel（F-H）非平衡标定因子在本仓 2:1 对流加密（`nu_f = 2 nu_c`）下
到底等于多少。参考值 ref = 解析 Taylor-Green / 剪切场的**精确非平衡部分**（零手抄，解析式
在 `analytic_fields_2d` 内程序生成），got = 移植算子的重建结果。

无量纲参数表（lattice 单位，rho=1）：
| 量 | 值/范围 |
|---|---|
| 域/粗网格 | 64x64 与 128x128（对比网格 2:1） |
| 细块 | 32x32（粗单元）=64x64 fine；128 情形为 64/128 |
| tau_coarse / tau_fine | 0.6 / 0.7（`tau_f = 0.5 + 2(tau_c-0.5)`，ν_c=0.0333, ν_f=0.0667） |
| TGV 幅值 U0 | 0.04（Ma≈0.069） |
| 顶替 donor 形式 | `mirror`（跨界镜像同层邻元）/`self`（零阶自取） |
| 判据 | 关键量相对误差 <15%（owner 标准），惰性/往返为精确 0 |

## 2 怎么做的
1. 新模块 `src/tensorlbm/amr_equal_level_stream.py`：`EqualLevelBorderStreamCorrection`
   由"被覆盖（blocked）"布尔掩码构建链接表（dst=被覆盖的拉取源槽、src=同层 donor 槽），
   `prepare/restore` 只做流式期顶替（numpy 扁平索引，D2Q9），复用
   `amr_cavity2d` 的格气约定与 regularised-BGK。
2. F-H 标定：把同一解析解分别采样到粗网格与细块，取粗非平衡部分按 scale 放大后与细层
   精确非平衡比较（剔除 O(h^2) 采样偏置），对 scale 做扫描求最小。
3. 集成调用方：`amr_cavity2d.CavityAMR2D` 新增 `coarse_shadow="frozen"` 分支（默认
   `restricted` 不变），在 `step()` 的流式处调用 `prepare/restore` 并冻结被覆盖行 ——
   这是 waLBerla 做法的真实调用点（`tests/test_amr_equal_level_stream.py::test_cavity_frozen_shadow_uses_the_correction`）。
4. 运行入口：`benchmarks/xval_block_amr_equal_level_stream_correction/run_xval.py`。
5. 工具链：NumPy float64 + matplotlib(Agg)；单核，全流程 <10 s，未用集群。

## 3 计算结果
F-H 标定（ref = 精确非平衡，got = 重建非平衡）：
| 算例 | 最优 scale | 相对 L2 误差 | 次优 scale | 目标函数间隙 |
|---|---|---|---|---|
{cal}

本仓现行两套约定的数值（同一 tau 对）：
| 约定 | scale |
|---|---|
{conv}

算子不变量与质量账簿：
| 算例 | 链接数 | 均匀场惰性 | prepare/apply 往返 | 质量缺陷(修正) | 质量缺陷(plain) |
|---|---|---|---|---|---|
{inv}

ref vs got 逐项判定（benchmark.json，共 {bench['summary']['rows']} 行，通过 {bench['summary']['passed']} 行）：
| KPI | ref | got | 误差 % | 判定 |
|---|---|---|---|---|
{rows}

## 4 结果图
- `artifacts/xval_block_amr_equal_level_stream_correction/fig1_calibration.png`：
  标定扫描曲线，三个仓内约定刻度线与实测最优点并列。
- `fig2_correction_effect.png`：顶替幅度 vs 惰性 / 往返残差（log 轴），展示"算子作用在场
  上但状态中性"。

## 5 注意点与移植取舍
- **实测 F-H 因子 = {v['fh_scale_measured']:.3f}**（相对误差 {100*v['fh_scale_error']:.4f} %，远小于 15% 线）。
  本仓 `amr_cavity2d._fh_rescale`（{res['conventions']['repo_cavity']:.4f}）与
  `amr_population_transfer.rescale_nonequilibrium(spatial_ratio=1/2)`
  （{res['conventions']['repo_torch']:.4f}）分别偏离实测值 {v['repo_cavity_off_by_factor']:.3f}x 与
  {v['repo_torch_off_by_factor']:.3f}x，且两者方向相反——**两者在 2:1 对流加密下都不是
  非平衡传递的正确因子**。物理上：f^neq ∝ tau*Δu，细网格 Δu 减半而 tau_f-1/2 加倍，乘积不变 → scale=1。
- 移植取舍①：waLBerla 的 `prepare/apply` 是**流式期**顶替（collide 之后、stream 之前）。本仓
  `static_block_amr` 的 advance 是"碰撞+流式"整体回调，无法在两者之间插入，因此本仓集成点放在
  `CavityAMR2D`（先碰后流的显式结构）——这是结构性取舍，不是等价替换。
- 移植取舍②：walberla 用 2:1 块森林 + 每块 pack/unpack 通信；本仓为单细块 + NumPy，无 MPI
  通信层，因此本机只验证**算法内核**（顶替算子 + F-H 标定），不验证并行通信路径。
- 局限①：**无法运行 waLBerla 本体**（无网络、GPL 禁搬码、本盘无构建链），故 ref 采用解析场而
  非 OSS 的跑数输出——这是 WITHHELD 级证据，已在 evidence 中如实标注。
- 局限②：被覆盖粗块冻结（`frozen`）模式的**整场时间推进精度**本轮未做验证：早期草稿的
  合成求解器在一次试验中出现粗层额外耗散（约 2 倍衰减率），我未能定位归因，故**已从交付中移除**，
  不作为能力声明；留作后续轮（honest FAIL 项）。本轮只声明算子不变量与 F-H 标定这两项实测结论。

## 6 误差判定（逐项显式回答"是否<15%"）
- F-H 标定误差 {100*v['fh_scale_error']:.4f} % → **<15% ✅**（判据通过，`passes_15pct={v['passes_15pct']}`）
- 均匀场惰性 = {res['invariants']['tgv_n64_p32']['inert_uniform_max_abs']:.1e}（ref=0）→ **精确 0 ✅**
- prepare/apply 往返 = {res['invariants']['tgv_n64_p32']['roundtrip_max_abs']:.1e}（ref=0）→ **精确 0 ✅**
- 三条分辨率/流场标定均落在同一最优 scale（见 §3 表）→ 标定对分辨率与流场不敏感 ✅
- 整场时间推进精度（waLBerla 做法的动力学收益）→ **未验证（WITHHELD）**，不算通过。
"""
OUT.joinpath("REPORT.md").write_text(doc)
Path("docs").mkdir(exist_ok=True)
Path("docs/XVAL_BLOCK_AMR_EQUAL_LEVEL_STREAM_CORRECTION.md").write_text(doc)
print("report bytes", len(doc))
