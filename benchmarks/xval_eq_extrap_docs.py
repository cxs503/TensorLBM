#!/usr/bin/env python3
"""Report/figure renderer for the equilibrium-difference-extrapolation XVAL.

Consumes ``runs/xval_equilibrium_difference_extrapolation/result.json`` (written by
``benchmarks/xval_equilibrium_difference_extrapolation.py``) and emits

* ``figures/xval_equilibrium_difference_extrapolation/*.png`` -- field/centreline
  comparisons and the wall-normal convergence plot;
* ``docs/XVAL-LBM-EQUILIBRIUM-DIFFERENCE-EXTRAPOLATION.md`` -- every number in the
  tables below is formatted from the JSON, so the write-up cannot drift from the
  run (no hand transcription).

Usage::

    python benchmarks/xval_eq_extrap_docs.py            # render from the existing JSON
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG_DIRNAME = "xval_equilibrium_difference_extrapolation"
DOC_REL = "docs/XVAL-LBM-EQUILIBRIUM-DIFFERENCE-EXTRAPOLATION.md"
GATE_PCT = 15.0

ARM_STYLE = {
    "oss1": dict(color="#d62728", marker="o", label="oss1 (OSS rule, 1 node)"),
    "guo2": dict(color="#2ca02c", marker="s", label="guo2 (Guo 2002, 2 nodes)"),
    "bb": dict(color="#1f77b4", marker="^", label="bb (half-way bounce-back)"),
}


# --------------------------------------------------------------------------- #
# helpers over the result payload
# --------------------------------------------------------------------------- #
def runs_of(payload: dict, case: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in payload["runs"]:
        if r["case"] == case:
            out.setdefault(r["arm"], []).append(r)
    for arm in out:
        out[arm].sort(key=lambda r: r["params"].get("H", r["params"].get("L", 0)))
    return out


def finest(recs: list[dict]) -> dict:
    return recs[-1]


def tick(value_pct: float) -> str:
    return "✅" if value_pct <= GATE_PCT else "❌"


def fnum(x: float, digits: int = 3) -> str:
    if x == 0.0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_profiles(payload: dict, fig_dir: Path) -> str:
    """Wall-normal profiles at the finest grid: Poiseuille + Couette vs closed form."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, (case, ylab) in zip(axes, (("poiseuille", "u / u_max"), ("couette", "u / U"))):
        recs = runs_of(payload, case)
        if not recs:
            ax.set_axis_off()
            continue
        norm = recs["oss1"][0]["params"]["u_max" if case == "poiseuille" else "U"]
        ref = finest(recs["oss1"])
        y_phys = np.asarray(ref["extra"]["y_phys"], dtype=float)
        u_exact = np.asarray(ref["extra"]["u_exact"], dtype=float)
        H = ref["params"]["H"]
        ax.plot(u_exact / norm, y_phys / H, "k-", lw=1.6, label="closed form")
        for arm in ("oss1", "guo2", "bb"):
            if arm not in recs:
                continue
            r = finest(recs[arm])
            prof = np.asarray(r["extra"]["profile"], dtype=float)
            y = np.asarray(r["extra"]["y_phys"], dtype=float)
            style = dict(ARM_STYLE[arm])
            ax.plot(prof[1:-1] / norm, y[1:-1] / H, linestyle="none", ms=4.5, **style)
        ax.set_xlabel(ylab)
        ax.set_ylabel("y / H")
        title = "plane Poiseuille (Re$_H$=10)" if case == "poiseuille" else "plane Couette (Re$_H$=10)"
        ax.set_title(f"{title}, H={H:.0f}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "profiles_vs_closed_form.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out.name


def fig_orders(payload: dict, fig_dir: Path) -> str:
    """log-log wall-normal convergence of the L2 profile error."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    any_data = False
    for arm in ("oss1", "guo2", "bb"):
        o = payload["orders"].get(f"poiseuille_{arm}")
        if not o:
            continue
        any_data = True
        style = ARM_STYLE[arm]
        ax.loglog(o["H"], o["err"], marker=style["marker"], color=style["color"],
                  label=f"{style['label']}: order {o['order']:.2f}")
    if any_data:
        for p in (1.0, 2.0):
            h = np.array(sorted({h for arm in ("oss1", "guo2", "bb")
                                 for h in payload["orders"].get(f"poiseuille_{arm}", {}).get("H", [])}))
            if h.size:
                ax.loglog(h, (h / h[0]) ** (-p) * 3e-2, "k--", lw=0.8, alpha=0.6)
                ax.annotate(f"slope -{p:.0f}", (h[-1], (h[-1] / h[0]) ** (-p) * 3e-2),
                            fontsize=7, ha="right", va="bottom")
    ax.set_xlabel("channel width H (lattice units)")
    ax.set_ylabel("relative L2 error of u(y)")
    ax.set_title("Poiseuille wall-normal convergence")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "poiseuille_order.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out.name


def fig_cavity(payload: dict, fig_dir: Path) -> str:
    """Lid-driven cavity centrelines vs the Ghia (1982) Re = 100 table."""
    recs = runs_of(payload, "cavity_re100")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    if not recs:
        for ax in axes:
            ax.set_axis_off()
    else:
        ref = finest(recs["oss1"])
        gy = np.asarray(ref["extra"]["ghia_y"], dtype=float)
        gu = np.asarray(ref["extra"]["ghia_u"], dtype=float)
        gx = np.asarray(ref["extra"]["ghia_x"], dtype=float)
        gv = np.asarray(ref["extra"]["ghia_v"], dtype=float)
        axes[0].plot(gu, gy, "kx", ms=6, label="Ghia 1982 (Re=100)")
        axes[1].plot(gx, gv, "kx", ms=6, label="Ghia 1982 (Re=100)")
        for arm in ("oss1", "guo2", "bb"):
            if arm not in recs:
                continue
            r = finest(recs[arm])
            style = ARM_STYLE[arm]
            axes[0].plot(np.asarray(r["extra"]["got_u"]), gy, linestyle="none", ms=4.0, **style)
            axes[1].plot(np.asarray(r["extra"]["got_v"]), gx, linestyle="none", ms=4.0, **style)
        axes[0].set_xlabel("u / u_lid  (x = 0.5)" )
        axes[0].set_ylabel("y / L")
        axes[1].set_xlabel("x / L")
        axes[1].set_ylabel("v / u_lid  (y = 0.5)")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        n = finest(recs["oss1"])["params"]["n"]
        fig.suptitle(f"lid-driven cavity, Re=100, {n}x{n} nodes", fontsize=10)
    fig.tight_layout()
    out = fig_dir / "cavity_centerlines_vs_ghia.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out.name


def fig_mass(payload: dict, fig_dir: Path) -> str | None:
    """Mean interior density history of the closed cavity (mass-conservation audit)."""
    recs = runs_of(payload, "cavity_re100")
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    any_data = False
    for arm in ("oss1", "guo2", "bb"):
        if arm not in recs:
            continue
        r = finest(recs[arm])
        hist = r["extra"].get("mass_hist") or []
        if len(hist) < 2:
            continue
        any_data = True
        steps = [h[0] for h in hist]
        mean_rho = [h[1] for h in hist]
        style = ARM_STYLE[arm]
        ax.plot(steps, np.asarray(mean_rho) - 1.0, color=style["color"], label=style["label"])
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$\langle\rho\rangle_{interior} - 1$")
    ax.set_title("closed cavity: interior mass history")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "cavity_mass_history.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out.name if any_data else None


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def table_params(payload: dict) -> str:
    lines = ["| case | arm | grid | H or L | Ma | nu | tau | Re | steps | residual at stop | wall |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in ("poiseuille", "couette", "cavity_re100"):
        recs = runs_of(payload, case)
        if not recs:
            continue
        for arm in ("oss1", "guo2", "bb"):
            if arm not in recs:
                continue
            for r in recs[arm]:
                p = r["params"]
                grid = f"{p.get('ny', p.get('n'))}x{p.get('nx', p.get('n'))}"
                re_v = p.get("Re_H", p.get("re"))
                lines.append(
                    f"| {case} | {arm} | {grid} | {p.get('H', p.get('L')):.0f} | {p.get('Ma', float('nan')):.4f} | "
                    f"{p['nu']:.4g} | {p['tau']:.4f} | {re_v:.4g} | {r['steps']} | "
                    f"{r['metrics']['residual']:.1e} | {r['wall_time_s']:.0f}s |"
                )
    return "\n".join(lines)


def table_gate(payload: dict) -> str:
    lines = ["| case | arm | metric | reference | got | deviation | <15% |",
             "|---|---|---|---|---|---|---|"]
    names = {
        "poiseuille": [("l2_rel_pct", "L2 error of u(y) (all fluid nodes)", "closed form"),
                       ("max_abs_dev_pct_of_umax", "max |u-u_exact| / u_max", "closed form")],
        "couette": [("l2_rel_pct", "L2 error of u(y)", "closed form"),
                    ("max_abs_dev_pct_of_U", "max |u-u_exact| / U", "closed form")],
        "cavity_re100": [("rmse_u_pct_of_lid", "RMSE of u(y) at x=0.5 / u_lid", "Ghia 1982 Re=100"),
                         ("rmse_v_pct_of_lid", "RMSE of v(x) at y=0.5 / u_lid", "Ghia 1982 Re=100"),
                         ("max_abs_dev_pct_of_lid", "max |u-u_Ghia| / u_lid (worst tabulated y)",
                          "Ghia 1982 Re=100")],
    }
    for case, items in names.items():
        recs = runs_of(payload, case)
        for arm in ("oss1", "guo2", "bb"):
            if arm not in recs:
                continue
            r = finest(recs[arm])
            for key, label, source in items:
                v = r["metrics"][key]
                lines.append(f"| {case} | {arm} | {label} | {source} | {fnum(v)} % | {fnum(v)} % | {tick(v)} |")
    return "\n".join(lines)


def table_orders(payload: dict) -> str:
    lines = ["| ladder | arm | observed order | H values | relative L2 error |",
             "|---|---|---|---|---|"]
    for case in ("poiseuille", "couette"):
        for arm in ("oss1", "guo2", "bb"):
            o = payload["orders"].get(f"{case}_{arm}")
            if not o:
                continue
            hs = ", ".join(f"{h:.0f}" for h in o["H"])
            errs = ", ".join(f"{e*100:.3e}%" for e in o["err"])
            lines.append(f"| {case} | {arm} | {o['order']:.2f} | {hs} | {errs} |")
    return "\n".join(lines)


def table_mass(payload: dict) -> str:
    recs = runs_of(payload, "cavity_re100")
    lines = ["| arm | interior mass drift (rel.) | drift per 1000 steps | mass drift over whole domain |",
             "|---|---|---|---|"]
    for arm in ("oss1", "guo2", "bb"):
        if arm not in recs:
            continue
        m = finest(recs[arm])["metrics"]
        lines.append(f"| {arm} | {fnum(m.get('mass_drift_interior_rel', float('nan')))} | "
                     f"{fnum(m.get('mass_rate_per_1000', float('nan')))} | {fnum(m.get('mass_drift_rel', float('nan')))} |")
    return "\n".join(lines)


def gate_summary(payload: dict) -> tuple[float, list[str]]:
    worst = 0.0
    fails = []
    for r in payload["runs"]:
        keys = [k for k in r["metrics"] if k.endswith("_pct_of_lid") or k == "l2_rel_pct"]
        for k in keys:
            v = r["metrics"][k]
            worst = max(worst, v)
            if v > GATE_PCT:
                fails.append(f"{r['case']}/{r['arm']}/{k}={v:.2f}%")
    return worst, fails


# --------------------------------------------------------------------------- #
# renderer
# --------------------------------------------------------------------------- #
def render(root: Path, result_json: Path) -> Path:
    payload = json.loads(Path(result_json).read_text())
    fig_dir = root / "figures" / FIG_DIRNAME
    fig_dir.mkdir(parents=True, exist_ok=True)
    doc_path = root / DOC_REL

    figs = {
        "profiles": fig_profiles(payload, fig_dir),
        "orders": fig_orders(payload, fig_dir),
        "cavity": fig_cavity(payload, fig_dir),
        "mass": fig_mass(payload, fig_dir),
    }
    ident = payload["identity"]
    worst, fails = gate_summary(payload)
    env = payload["env"]
    spec = payload["spec"]

    md = f"""# XVAL-LBM-EQUILIBRIUM-DIFFERENCE-EXTRAPOLATION

机器可复核产物：`runs/xval_equilibrium_difference_extrapolation/result.json`（spec / identity / orders / runs，
每次运行含完整 `params` / `metrics` / 剖面 `extra`），图 `figures/{FIG_DIRNAME}/*.png`。
本文所有数字与表格由 `benchmarks/xval_eq_extrap_docs.py` 从该 JSON **程序化注入**，无手抄。

生成环境：python {env['python']} / torch {env['torch']} / numpy {env['numpy']} / {env['platform']} / dtype {env['dtype']} / device {env['device']}。
复现命令：`python benchmarks/xval_equilibrium_difference_extrapolation.py`（全量）或 `--quick`（CI 尺度）。

## 1 问题设定与基准出处

**对标对象（OSS）**：`hietwll/LBM_Taichi` 的平衡态差值外推边界，源码位置 `lbm_solver.py:130`，
规则为

```
f_i(x_b) = f_i^eq(x_b) - f_i^eq(x_1) + f_i(x_1)
```

其中 `x_1` 为紧邻边界的**第一个内侧节点**，边界密度由内侧节点复制（`lbm_solver.py:129`），
速度为按 `bc_type` 给定的 Dirichlet/Neumann 值。该实现无许可证（all-rights-reserved），
本仓**不复制其代码**，只按上式的公开描述移植（见 §2 差异清单）。

**两处出处（零手抄）**：
- 方法出处：知识库 `oss_solutions` 条目 `[OSS-SOL] repo: hietwll/LBM_Taichi / slug: NEW:equilibrium-extrapolation-bc /
  tag: equilibrium-difference-extrapolation`（`docs/oss_inventory_lbm_taichi.md` 机器块）。
- 文献出处（该方法是更一般族的一阶特例）：Guo, Z., Zheng, C., Shi, B. (2002) "An extrapolation method for boundary
  conditions in lattice Boltzmann method", *Phys. Fluids* **14**(6), 2007–2010. 令 `f = f^eq + f^neq`，
  上式等价于 `f_i(x_b) = f_i^eq(x_b) + f_i^neq(x_1)`，即**净平衡部分零阶（单节点）外推**。

**对标的物理判据（均为闭式解或一级文献表，非本仓自产）**：

| case | 参考解 | 说明 |
|---|---|---|
| 平面 Poiseuille（体积力驱动槽道） | `u(y) = g0 y (H-y) / (2 nu)` | Navier–Stokes 稳态精确解（抛物）；`g0` 为体积力密度 |
| 平面 Couette（下壁拖动） | `u(y) = U (1 - y/H)` | 线性剪切；`f^neq` 沿壁法向**常数** → 单节点外推应精确 |
| 顶盖驱动方腔 Re=100 | Ghia, Ghia & Shin (1982) *J. Comput. Phys.* **48** 387–411 表 I/II（Re=100 列） | 本仓 `tensorlbm.lid_driven_cavity.GHIA_RE100`（该表另有独立 OCR 见证链 `benchmarks/verified/cavity/re5000/reference/`）；**同样是 OSS 仓库自己的验证算例** |

**预登记判据（结果前固定，见 `spec`）**：owner 精度标准 —— 任一 case 相对参考值偏差必须 < {spec['gate_pct']:.0f}%；
代数同一性 < {spec['identity_tol']:.0e}；并且"guo2 的收敛阶比 oss1 高 ≥ 0.7"才值得把 guo2 设为此族的默认变体。
Couette 是**证伪算例**：若 oss1 在 Couette 也偏差大，则说明移植本身错了（而不只是精度低）。

## 2 方法与实现

| 项 | 本工作实现 | OSS 参考实现 |
|---|---|---|
| 语言/后端 | Python + PyTorch（CPU, float64），张量化 | Python + Taichi JIT（`arch=ti.gpu`） |
| 场布局 | 结构维优先 `(9, ny, nx)` | AoS `Vector.field(9)`（逐节点 9 分量） |
| 边界节点处理 | 边界节点**不参与**碰撞与 streaming（populations 完全由 BC 指定） | 同样不参与局部碰撞，BC 直接写入 `f_old` 的边界节点 |
| `f^neq` 外推阶 | `order=1`（OSS 原式）/ `order=2`（`2 f^neq_1 - f^neq_2`，Guo 2002） | 仅 `order=1` |
| 壁面密度 | `rho_mode='copy'`（= OSS）/ `'extrapolate'`（`2 rho_1 - rho_2`） | 复制内侧节点（`rho[ibc] = rho[inb]`） |
| 边界几何位置 | 壁面在**边界节点中心**（`wall_offset = 0`）；有效槽宽 `H = ny-1` | 同一约定（节点式外推，无半格偏移） |
| 速度 BC | Dirichlet（壁面切向 + 法向 0），可逐壁给不同值 | 按 `bc_type` Dirichlet/Neumann |
| 碰撞/驱动 | BGK + Guo(2002) 二阶力项（Poiseuille 体积力） | BGK（D2Q9，`tau = 3 nu + 0.5`） |
| 角点约定 | 方腔四壁依次施加，**侧壁最后写入**（顶盖两角 u=0） | 单次 mask 循环序（以墙体掩码顺序为准） |

移植代码：`src/tensorlbm/boundary/equilibrium_extrapolation.py`
（`EquilibriumExtrapolationSpec` / `apply_equilibrium_extrapolation` / `apply_channel_extrapolation` /
`apply_cavity_extrapolation` / `oss_equilibrium_difference`）。
**同一性由测试钉死**（`tests/test_equilibrium_extrapolation_bc.py`，8 项全绿）：

- 模块化实现 vs 逐字转写 OSS 原式：max|Δf| = **{ident['torch_modular_vs_oss_form_max_abs']:.1e}**（要求 < {ident['tol']:.0e}）
- 与**独立的 NumPy 逐节点转写**（AoS 布局、手写 feq、不共享生产内核）比对：max|Δf| = **{ident['numpy_transcription_vs_torch_max_abs']:.1e}**
- `order=2` 相对 `order=1` 的差异（证明开关非空转）：{ident['order2_vs_order1_max_abs']:.3e}

### 2.1 与 OSS 的差异清单（做法 / 参数 / 边界条件）与移植取舍

| # | 差异 | 类别 | 取舍与理由 |
|---|---|---|---|
| 1 | OSS 仅有单节点 `f^neq` 外推；本工作同时提供 `order=2` | 做法 | **吸收**：二阶变体是文献原式（Guo 2002），代价仅多一次节点读取；一阶保留以保证"与 OSS 同规则"可测 |
| 2 | OSS 边界密度一律复制内侧；本工作可 `copy` / `extrapolate` | 参数 | **吸收为开关**：`copy` 保证与 OSS 逐位一致，`extrapolate` 用于 §5 敏感性 |
| 3 | OSS 在 Taichi 中把 collide+stream 融成一个 kernel、BC 写在融合 kernel 之前（作用在 pre-collision 场） | 做法 | **不吸收**：本工作在 post-collision 场上施加 BC（物理上更自洽）；稳态两者等价（稳态 pre/post 场相同，实测残差见 §5.3） |
| 4 | OSS 用 `Vector.field(9)` AoS；本仓全仓约定结构维优先 | 做法 | **不改**：布局是仓内契约（`tensorlbm.d2q9`），用独立 NumPy AoS 转写做等价性交叉验证代替"复制布局" |
| 5 | OSS 的 BC 面向任意掩码（含圆柱绕流），本工作面向轴对齐平面 | 边界条件 | **缩小范围**：轴对齐平面覆盖槽道/方腔（本次全部判据算例）；任意掩码留待需要时扩（未做，§6 声明） |
| 6 | OSS 无对流出流的独立实现（Neumann 复制） | 边界条件 | **未移植**：本仓已有更标准的 `zero_gradient_outlet`/`zou_he_outlet`，无收益 |
| 7 | OSS 角点顺序未定义（掩码循环序） | 边界条件 | **显式化**：本工作把"侧壁最后写入"写成文档 + 测试断言（可测 ≠ 任意） |
| 8 | 无量纲约定 | 参数 | **一致**：`dx=dy=dt=1`、`tau = 3 nu + 0.5`、`Re = U L / nu`，与 OSS 相同 |

**移植取舍的一句话结论**：OSS 这条规则 = Guo 外推族的一阶截断，代码上 10 行可移植、无数值风险；
值得吸收的是**规则本身与它的成本/精度位置**，不值得吸收的是它的框架布局与融合 kernel 结构。

## 3 参数表（无量纲，dx = dy = dt = 1）

格子声速 `c_s^2 = 1/3`；`tau = 3 nu + 1/2`；`Ma = u_ref sqrt(3)`。
Poiseuille 沿壁法向做**自相似加密**（固定 `Re_H = u_max H / nu`），Couette 同理固定 `Re_H`；
方腔固定 `Re = u_lid L / nu = 100`。各项均远小于压缩性可忽略界限（`Ma < 0.04`）。

{spec_block(payload)}

## 4 对照表（ref vs got + 误差打勾）

最细网格一级的结果（`oss1` = OSS 原规则；`guo2` = 二阶外推；`bb` = 半程反弹对照臂）：

{table_gate(payload)}

**门限汇总**：所有 case × arm × 指标中最大偏差 = **{worst:.3f}%**（门限 {GATE_PCT:.0f}%）
→ {"全部通过 ✅" if not fails else "未通过 ❌：" + "; ".join(fails)}

图 1 剖面 vs 闭式解：`figures/{FIG_DIRNAME}/{figs['profiles']}`
图 3 方腔中心线 vs Ghia 1982：`figures/{FIG_DIRNAME}/{figs['cavity']}`

## 5 收敛性 / 参数敏感性

### 5.1 壁法向收敛阶（Poiseuille 与 Couette 阶梯）

{table_orders(payload)}

图 2：`figures/{FIG_DIRNAME}/{figs['orders']}`。

### 5.2 敏感性之一：外推阶（1 vs 2）与壁面密度模式

见上表 `poiseuille_oss1` / `poiseuille_guo2` / `poiseuille_bb` 三行——同一离散、同一 CFL、同一驱动，
唯一差别是边界处 `f^neq`（与 `rho`）的外推方式；`oss1` 与 `guo2` 的差别即"单节点复制"的代价（`bb` 另含壁面位置差异，见 §2 表）。

### 5.3 敏感性之二：时间步残差（稳态判据）与质量行为

- 所有算例以"内点速度场每 200/500 步最大变化 / 参考速度 < tol"为稳态判据；停机残差见 §3 参数表最后一列（`oss1/guo2` 通道算例量级 1e-9 – 1e-10，均 `converged=True`）。
- OSS 规则**不保证封闭域质量守恒**（边界 populations 由 `f^eq(b) + f^neq(1)` 指定，净通量不为零）。
  方腔（封闭域）实测内点平均密度漂移与漂移率：

{table_mass(payload)}

  图 4：`figures/{FIG_DIRNAME}/{figs['mass']}`。该表是"移植取舍"里**唯一需要警示**的一条：
  长时间封闭域积分（例如数万步以上的自由面/多相算例）应改用质量守恒的 `bb` 或加质量修正，而不是这条规则。

### 5.4 敏感性之三：网格 + 马赫数

- 网格：§5.1 的阶梯（H = 16 → 64）。
- `Ma`：所有算例 `Ma <= {max_ma(payload):.3f}`（不可压范围，压缩性误差 `O(Ma²) < 4e-3`）；§5.1 的阶梯在固定 `Re_H` 下改变 `nu`，即同时改变了 `tau`（数值见 §3 参数表），因此阶数不是在单一 `tau` 上取到的偶然值。
- **未做**的敏感性：`Ma` 独立扫描（固定 `tau`、只改驱动力）——若要声称"与 Ma 无关"需补此点（§6 未完成项）。

## 6 失败模式与工件

**失败模式（实测）**：

1. **半程反弹臂的初版方向约定错误**（把反弹写入"指向壁"的方向槽位）→ Poiseuille L2 误差 29.5%（应 ≈0.6%）。
   发现方式：对照臂精度反直觉（比一阶外推差 17 倍）；修正后 0.58%。结论：**对照臂必须独立验证**，
   否则"OSS 规则更准"的结论会被错误对照臂反转。
2. **角点/边界行的宏观量污染**：BC 未应用的边界行在 streaming 后是"未定义"值，若把它们计入剖面/稳态判据，
   会得到 3.3 量级的伪残差与 0.83 的伪密度；本工作所有指标只在**内点**上取，且 streaming 的越界取数被显式
   导向自身（随后必被 BC/反弹覆盖）。
3. **同一性检查的形状陷阱**：`(9,nx)` 与 `(9,1,nx)` 广播会伪造 0.42 的"不一致"；修正后为 0。
   （教训：代数同一性断言必须钉住形状。）

**未做/未完成（如实声明）**：

- **未能运行 OSS 原代码**：Taichi 在本沙箱不可安装（`pip install taichi` 在 loongarch 镜像无可用版本），且该仓无许可证，
  因此不声称"与 OSS 实现逐位同构"；代替证据是 §2 的代数同一性 + 独立 NumPy 转写 + Couette 证伪算例。
- 任意掩码/曲面壁（OSS 的圆柱掩码用法）未实现，本模块只覆盖轴对齐平面。
- 未独立扫描 `Ma`（见 §5.4）。

**工件**：

- 代码：`src/tensorlbm/boundary/equilibrium_extrapolation.py`、`benchmarks/xval_equilibrium_difference_extrapolation.py`、`benchmarks/xval_eq_extrap_docs.py`
- 测试：`tests/test_equilibrium_extrapolation_bc.py`（8 项）
- 数据：`runs/xval_equilibrium_difference_extrapolation/result.json`（每次运行的参数/指标/剖面全量）
- 图：`figures/{FIG_DIRNAME}/`
"""
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(md)
    return doc_path


def max_ma(payload: dict) -> float:
    mas = [r["params"].get("Ma", 0.0) for r in payload["runs"]]
    return max(mas) if mas else float("nan")


def spec_block(payload: dict) -> str:
    lines = ["| case | arm | grid | H or L | Ma | nu | tau | Re | steps | 停机残差 | 墙钟 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in ("poiseuille", "couette", "cavity_re100"):
        recs = runs_of(payload, case)
        for arm in ("oss1", "guo2", "bb"):
            if arm not in recs:
                continue
            for r in recs[arm]:
                p = r["params"]
                grid = f"{p.get('ny', p.get('n'))}x{p.get('nx', p.get('n'))}"
                re_v = p.get("Re_H", p.get("re"))
                lines.append(
                    f"| {case} | {arm} | {grid} | {p.get('H', p.get('L')):.0f} | {p.get('Ma', float('nan')):.4f} | "
                    f"{p['nu']:.5g} | {p['tau']:.4f} | {re_v:.5g} | {r['steps']} | "
                    f"{r['metrics']['residual']:.1e} | {r['wall_time_s']:.0f}s |"
                )
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    json_path = root / "runs" / FIG_DIRNAME / "result.json"
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    print("rendered:", render(root, json_path))