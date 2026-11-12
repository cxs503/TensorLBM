#!/usr/bin/env python3
"""Finalise the CASE-CYLINDER-DRAG-RE100 (m15776) artifacts from finished level runs.

Steps
-----
1. take every ``runs/m15776/D*_u0833.json`` (+ ``.series.npz``) that exists,
2. re-derive the Strouhal number from the archived C_L(t) with the *current*
   estimator (the level processes were started before the estimator was fixed;
   the raw series are unchanged, so this is a re-analysis, not a re-run),
3. copy them into ``benchmarks/verified/cylinder/re100_dfg/artifacts/`` and
   aggregate into ``result.json`` (the file the validation matrix judges),
4. render the report figures,
5. write the six-section mission document.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks/verified/cylinder/re100_dfg"))

import run as case_run  # noqa: E402
from tensorlbm.validation import cylinder_dfg as cdfg  # noqa: E402

RUNS = REPO / "runs/m15776"
OUT = REPO / "benchmarks/verified/cylinder/re100_dfg/artifacts"
FIGS = REPO / "benchmarks/verified/cylinder/re100_dfg/figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

levels: list[int] = []
for jf in sorted(RUNS.glob("D*_u0833.json")):
    D = int(jf.name[1:].split("_")[0])
    res = json.loads(jf.read_text())
    snpz = jf.with_suffix(".series.npz")
    if not snpz.exists():
        print(f"skip D={D}: no series")
        continue
    series = np.load(snpz)
    cl = series["cl"].astype(np.float32)
    st = cdfg.strouhal_from_series(
        torch.tensor(cl), res["period_steps_analytic"], 0.30, D, res["u_mean_lb"]
    )
    st_primary = st["st_fft"] if st["st_fft"] == st["st_fft"] else st["st_acf"]
    res["st_estimator"] = "strouhal_from_series: FFT peak + parabolic refinement on archived C_L(t)"
    res["st"] = st_primary
    res["st_acf"] = st["st_acf"]
    res["st_fft"] = st["st_fft"]
    res["st_extraction"] = st
    res["st_acf_vs_fft_pct"] = (
        abs(st["st_acf"] - st["st_fft"]) / st["st_fft"] * 100.0
        if st["st_acf"] == st["st_acf"] and st["st_fft"] == st["st_fft"]
        else float("nan")
    )
    res["st_ambiguous"] = bool(res["st_acf_vs_fft_pct"] > 3.0)
    ref = cdfg.REFERENCE_BANDS["2D-2"]
    res["bench"]["st"] = cdfg.band_error(st_primary, ref["st"])
    res["err_pct"]["st"] = res["bench"]["st"]["err_pct_vs_mid"]
    dst = OUT / f"level_D{D}_u0833.json"
    dst.write_text(json.dumps(res, indent=2, default=cdfg._json_default))
    shutil.copy(snpz, dst.with_suffix(".series.npz"))
    levels.append(D)
    print(f"D={D}: cd={res['cd']:.4f} ({res['err_pct']['cd']:+.2f}%) st={st_primary:.4f} "
          f"cl_amp={res['cl_amp']:.4f} dp*={res['dp']:.4f} window={res['avg_window']}")

result = case_run.aggregate(OUT, "u0833", tuple(sorted(levels)), 0, 0.0833, 5.0)
print("verdict:", result["verdict"], "machine_pass:", result["machine_pass"])
print(json.dumps(result["per_metric"], indent=1))
print("convergence:", result["convergence"])

# ------------------------------------------------------------------ figures
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

figdir = FIGS
colors = {20: "tab:blue", 28: "tab:green", 40: "tab:red"}

# (1) force histories
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for D in levels:
    d = np.load(OUT / f"level_D{D}_u0833.series.npz")
    axes[0].plot(np.arange(d["cl"].size) / D, d["cl"], lw=1.0, color=colors.get(D), label=f"D={D}")
    axes[1].plot(np.arange(d["cd"].size) / D, d["cd"], lw=1.0, color=colors.get(D), label=f"D={D}")
axes[0].set(xlabel="t* = tU/D (in the averaging window)", ylabel="C_L", title="Lift history")
axes[1].set(xlabel="t* = tU/D (in the averaging window)", ylabel="C_D", title="Drag history")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(figdir / "fig1_force_history.png", dpi=130)
plt.close(fig)

# (2) reference bands vs measured
ref = cdfg.REFERENCE_BANDS["2D-2"]
specs = [("cd", "C_D", 1.0), ("cl_amp", "C_L amplitude", 1.0), ("st", "St", 1.0), ("dp", "Δp raw", 0.5)]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, (key, label, scale) in zip(axes, specs):
    lo, hi = ref[key]
    ax.axhspan(lo * scale, hi * scale, color="0.75", alpha=0.7, label="S&T 1996 band")
    vals = [result["grids"][f"D{D}"][key] * scale for D in levels]
    ax.plot(levels, vals, "o-", color="tab:blue", label="measures")
    for D, v in zip(levels, vals):
        ax.annotate(f"{v:.3f}", (D, v), textcoords="offset points", xytext=(0, 7), fontsize=7,
                    ha="center")
    ax.set(xlabel="D (lattice cells)", title=label)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(figdir / "fig2_bands_vs_measured.png", dpi=130)
plt.close(fig)

# (3) centreline (y = 0.2) mean pressure
fig, ax = plt.subplots(figsize=(7, 3.6))
for D in levels:
    d = np.load(OUT / f"level_D{D}_u0833.series.npz")
    rho = d["rho_centerline_mean"].astype(float)
    dx = cdfg.CYL_D / D
    ax.plot(np.arange(rho.size) * dx, rho / 3.0, lw=1.2, color=colors.get(D), label=f"D={D}")
for xc in (cdfg.PROBE_A[0], cdfg.PROBE_B[0]):
    ax.axvline(xc, ls=":", color="k", lw=0.8)
ax.set(xlabel="x (m)", ylabel="mean p (lattice units, rho/3)", xlim=(0.05, 0.45),
       title="Centreline mean pressure (probes at x=0.15 / 0.25)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(figdir / "fig3_centerline_pressure.png", dpi=130)
plt.close(fig)
print("figures ->", sorted(p.name for p in figdir.glob("*.png")))

# ------------------------------------------------------------------ document
def f(x, n=4):
    return f"{x:.{n}f}"

rows = "\n".join(
    f"| D={D} | {result['grids'][f'D{D}']['nx']}×{result['grids'][f'D{D}']['ny']} | "
    f"{result['grids'][f'D{D}']['tau']:.4f} | {result['grids'][f'D{D}']['n_steps']} | "
    f"{f(result['grids'][f'D{D}']['cd'])} | {result['grids'][f'D{D}']['err_pct']['cd']:+.2f}% | "
    f"{f(result['grids'][f'D{D}']['cl_amp'])} | {result['grids'][f'D{D}']['err_pct']['cl_amp']:+.2f}% | "
    f"{f(result['grids'][f'D{D}']['st'])} | {result['grids'][f'D{D}']['err_pct']['st']:+.2f}% | "
    f"{f(result['grids'][f'D{D}']['dp'])} | {result['grids'][f'D{D}']['err_pct']['dp']:+.2f}% |"
    for D in levels
)
doc = REPO / "docs/CASE-CYLINDER-DRAG-RE100_m15776.md"
doc.parent.mkdir(parents=True, exist_ok=True)
doc.write_text(
    f"""# CASE-CYLINDER-DRAG-RE100 — Schäfer & Turek (1996) DFG 2D-2, TensorLBM (mission m15776)

**判决（程序化生成，勿手改）**：`{result['verdict']}` — machine_pass = `{result['machine_pass']}`，
验收带 ±{result['tolerance']['err_pct']:.0f}%（对带中点），收敛判据 ≤3%。
完成网格档：{', '.join(f'D={D}' for D in levels)}。

## ① 问题与参考出处

DFG 基准 **2D-2**：二维通道内圆柱绕流，`x∈[0,2.2]×y∈[0,0.41]`，圆柱中心 `(0.2,0.2)`、
直径 `D=0.1`（阻塞比 `D/H=0.2439`），入口 `u=4U_max y(H-y)/H²`（`U_max=1.5 → U_mean=1.0`），
出口 `p=0`，`Re=U_mean·D/ν=100`（`ν=0.001`）。

参考值（Schäfer & Turek 1996, *Flow Simulation with High-Performance Computers II*,
NNFM vol. 48, Vieweg, pp. 547–566，**case 2D-2 量值表（Table 3）**）：
`C_D=3.2200–3.2400`、`C_L amplitude=0.9900–1.0100`、`St=0.2950–0.3050`、`Δp=2.4600–2.5000`
（ρ=1、`U_mean`=1 的原始压差，等价 `Δp/(0.5ρU²)=4.92–5.00`）。
冻结台账：`benchmarks/verified/cylinder/re100_dfg/REFERENCE.md`（token + sha256 由
`benchmarks/validation_matrix/reference_audit.py` 机器复读，16/16 anchors clean）。

> **口径警告（本单最重要的坑）**：m15432 用**自由流**圆柱配置去比 S&T 锚点，得到 Cd=1.7301
> “+25.4%” 的 honest-FAIL，其实是几何/锚点错配 + D=10 数字圆柱欠解析，不是求解器缺陷
> （见 `docs/board-audit-m15432/AUDIT_m15432_CASE-CYLINDER-DRAG-RE100.md`）。
> 自由流 Re=100 的锚是 Braza 1986 `Cd=1.35±0.05`（本仓 `benchmarks/verified/cylinder/re100`），
> 通道阻塞会把 Cd 抬到 3.23——两个锚点不可混用。

## ② 方法与实现

- 实现：`src/tensorlbm/validation/cylinder_dfg.py`（库原语组装，无手写 collide/stream/equilibrium）
  - `solver.collide_mrt`（D2Q9 MRT）+ `solver.stream`
  - `boundaries.zou_he_inlet_velocity`（抛物剖面；播种期叠加反对称 uy 扰动）
  - `boundaries.zou_he_outlet_pressure(rho=1)`
  - `boundaries.bounce_back_cells`（上下壁 + 圆柱，半程反弹）
  - `boundaries.compute_obstacle_forces`（Ladd 1994 动量交换，post-stream / pre-bounce-back，
    与 `cylinder/re20_st` 已验收的错位一致）
  - **无质量重整化**（re20_st 已证其激发 ~10k 步 Cd 慢振荡）
- 入口：`benchmarks/verified/cylinder/re100_dfg/run.py`（网格序列 + 聚合 + 判决）
- 平台侧真调用：`benchmarks/validation_matrix/matrix.json` 的 `cylinder_re100_dfg` 案件
  → `runner.py check/run` 读取 `artifacts/result.json` 的 `grids.<lvl>.err_pct.<metric>`，
- 测试：`tests/test_cylinder_dfg_reference.py`（6 passed：锚点零手抄复读、探针不落固壁、St 反演、烟测）
- 压力探针：DFG 的 `x=0.15/0.25` 恰好落在数字圆柱表面（固壁格点），固壁格点反弹后
  布居是镜像、不是物理密度（D=20 试点实测使 Δp 低估 −19%）。本实现取最近**流体**列
  （`x=0.145/0.255`）并线性外推到 0.15/0.25，原始近列值同时入库作敏感性。

## ③ 参数表（无量纲量完整）

| 项 | 值 |
|---|---|
| Re | 100（`U_mean·D/ν`） |
| 阻塞比 D/H | 0.2439 |
| 格子平均速度 `u_mean` | {result['config']['u_mean_lb']} |
| 入口峰值/均值比 | 1.5（DFG 抛物剖面） |
| 压缩性 `Ma=u_max/c_s` | ≈{result['config']['u_mean_lb']*1.5*np.sqrt(3):.3f} |
| 碰撞模型 | D2Q9 MRT，τ = 0.5+3ν（逐档见表） |
| 播种 | 反对称入流 uy，2 个 St 周期，振幅 5% `u_mean` |
| 统计窗 | 末 {result['grids'][f'D{levels[-1]}']['avg_window']} 步（D={levels[-1]}） |

## ④ 对照表（ref vs got + 误差打勾叉）

| 网格 | nx×ny | τ | steps | C_D | err | C_L amp | err | St | err | Δp(原始) | err |
|---|---|---|---|---|---|---|---|---|---|---|---|
{rows}

判据：`|err| ≤ 5%`；✓/✗ 由 `per_metric` 机器给出 →
{json.dumps({m: {k: v['finest_within_tol'] for k, v in result['per_metric'].items()}}, ensure_ascii=False)}。

## ⑤ 收敛性 / 参数敏感性

- 网格：见上表（D 每档加密 1.4×），聚合收敛门 `cd_span_pct`：
  `{result['convergence'].get('cd_span_pct', float('nan')):.3f}%`，`st_span_pct`：
  `{result['convergence'].get('st_span_pct', float('nan')):.3f}%`（门限 3%）。
- 统计：C_D 均值标准误 `cd_stderr` 见表内 artifact；`drift_pct`（前半窗 vs 后半窗）逐档记录。
- St 双估计：自相关峰 `st_acf` 与 FFT 峰 `st_fft` 逐档入库；二者偏差 >3% 时 `st_ambiguous=true`
  （D=20 试点曾因原始零穿越计数得到 St=0.52 的假值，已废弃该方法）。
- Δp 探针敏感性：`dp` 与 `dp_nearest_fluid_columns` 同时入库（外推把 Δp 抬高约 3%）。

## ⑥ 失败模式与工件

工件（全部真实运行产物，非手填）：
- `benchmarks/verified/cylinder/re100_dfg/artifacts/level_D*_u0833.json` + `.series.npz`
  （C_D(t)、C_L(t)、Δp(t)、中心线平均密度，逐档）
- `benchmarks/verified/cylinder/re100_dfg/artifacts/result.json`（聚合 + 逐 metric 判决）
- 图：`figures/fig1_force_history.png`、`fig2_bands_vs_measured.png`、`fig3_centerline_pressure.png`
- 一键复现：`python benchmarks/verified/cylinder/re100_dfg/run.py --levels {' '.join(str(D) for D in levels)} --steps {result['grids'][f'D{levels[-1]}']['n_steps']} --threads 4`

失败模式与生态摩擦（如实记）：
1. **固壁探针**：DFG 探针点落在数字圆柱表面上 → 固壁布居非物理，Δp 低估 19%；改为最近流体列 + 外推。
2. **St 估计器**：原始零穿越计数在动量交换升力噪声上产生伪穿越（St=0.52 vs 真值 0.30）；
   改为自相关/FFT 峰 + 抛物线细化（有测试 `test_strouhal_extraction_recovers_known_period` 锁定）。
3. **算力摩擦**：本机无 GPU，torch CPU 每步 ~12 ms（D=20）/ ~35-48 ms（D=28/40）；线程数越多越慢
   （128 线程 183 ms/step vs 8 线程 35 ms，控制面机器被抢占），run_cluster 队列积压 362 单，
   故 grid 序列在控制面本地跑，D=40 档（60k 步 ≈ 50 min）在本轮预算内未完成 →
   **本轮只报告 {'/'.join(f'D={D}' for D in levels)} 档**，D=40 作为待续档位（不冒充数据）。
4. 阶梯（staircase）动量交换的固有网格偏差在 D=20 上把 C_D 抬高
   （{result['grids'][f'D{levels[0]}']['err_pct']['cd']:+.2f}%），随加密下降趋势需 D=40 档确认。
""",
    encoding="utf-8",
)
print("doc ->", doc)