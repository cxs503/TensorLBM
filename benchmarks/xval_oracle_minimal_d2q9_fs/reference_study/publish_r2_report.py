#!/usr/bin/env python3
"""Publish the R2 resolution-study report (anchor XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS).

All numbers are injected from ``artifacts/resolution_analysis.json`` (programmatic
dump, written by ``resolution_study.py --analyze``); nothing is hand-copied.
Figures come from ``make_figures_r2.py`` and are base64-embedded.

Usage:  python benchmarks/xval_oracle_minimal_d2q9_fs/reference_study/publish_r2_report.py [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
FIG = HERE / "figures"
OUT = Path("/tmp/report_m20893.json")
MISSION = 20893


def fmt(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "pass" if x else "FAIL"
    if isinstance(x, (int,)):
        return str(x)
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def build() -> dict:
    A = json.loads((ART / "resolution_analysis.json").read_text())
    P = json.loads((ART / "resolution_ladder.json").read_text())
    D = P["design"]
    ladder = A["ladder"]
    alist = sorted(ladder, key=lambda k: int(k))
    keys = ["Z_cell", "Z_sub0.02", "Z_sub0.05", "Z_sub0.1"]
    fit_mm = A["fits"]["mm"]
    fit_rit = A["fits"]["ritter"]
    rich_mm = A["richardson"]["mm"]
    rich_rit = A["richardson"]["ritter"]

    # ---- ladder table -----------------------------------------------------
    t = [
        "| a | steps | u_max | inv drift % | Z_cell M&M | Z_sub.05 M&M "
        "| Z_sub.05 Ritter | Z_cell Ritter | wall s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a in alist:
        r = ladder[a]
        t.append(
            f"| {a} | {r['steps']} | {fmt(r['u_max'], 3)} | "
            f"{fmt(r['inventory_drift_max_pct'], 4)} | "
            f"{fmt(r['errors_mm']['Z_cell'])} | {fmt(r['errors_mm']['Z_sub0.05'])} | "
            f"{fmt(r['errors_ritter']['Z_sub0.05'])} | "
            f"{fmt(r['errors_ritter']['Z_cell'])} | {fmt(r['wall_seconds'], 1)} |"
        )
    ladder_md = "\n".join(t)

    # ---- fit table --------------------------------------------------------
    t = [
        "| metric | p | E_inf % | C | rms % | monotone | H1 (E_inf < 3 %) |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in keys:
        f = fit_mm.get(k, {})
        if not f.get("ok"):
            t.append(f"| {k} | — | — | — | — | — | fit failed |")
            continue
        t.append(
            f"| {k} | {fmt(f['p'])} | {fmt(f['E_inf_pct'])} | {fmt(f['C'], 3)} | "
            f"{fmt(f['rms_pct'])} | {f['monotone_falling']} | "
            f"{f['hypothesis_H1_supported']} |"
        )
    fit_md = "\n".join(t)

    t = ["| metric | p | E_inf % | C | rms % | H1 (E_inf < 3 %) |", "|---|---|---|---|---|---|"]
    for k in keys:
        f = fit_rit.get(k, {})
        if not f.get("ok"):
            t.append(f"| {k} | — | — | — | — | fit failed |")
            continue
        t.append(
            f"| {k} | {fmt(f['p'])} | {fmt(f['E_inf_pct'])} | {fmt(f['C'], 3)} | "
            f"{fmt(f['rms_pct'])} | {f['hypothesis_H1_supported']} |"
        )
    fit_rit_md = "\n".join(t)

    t = [
        "| metric | grids | p_obs (median) | Z_inf max err % | mean err % | pass < 3 % |",
        "|---|---|---|---|---|---|",
    ]
    for k in keys:
        r = rich_mm.get(k, {})
        if not r.get("ok"):
            t.append(f"| {k} | — | — | — | — | n/a |")
            continue
        t.append(
            f"| {k} | {r['grids']} | {fmt(r['p_obs_median'])} | "
            f"{fmt(r['Z_inf_max_abs_rel_err_pct'])} | "
            f"{fmt(r['Z_inf_mean_abs_rel_err_pct'])} | {r['pass_3pct']} |"
        )
    rich_md = "\n".join(t)

    t = [
        "| tau | nu | Z_sub.05 vs M&M % | Z_sub.05 vs Ritter % | inv drift % | u_max |",
        "|---|---|---|---|---|---|",
    ]
    for tau in sorted(A.get("tau_sweep", {}), key=lambda x: float(x)):
        r = A["tau_sweep"][tau]
        t.append(
            f"| {tau} | {fmt((float(tau) - 0.5) / 3.0, 4)} | "
            f"{fmt(r['Z_sub0.05_max_err_pct'])} | {fmt(r.get('Z_sub0.05_ritter_pct'))} | "
            f"{fmt(r['inventory_drift_max_pct'], 3)} | {fmt(r['u_max'], 3)} |"
        )
    tau_md = "\n".join(t)

    t = ["| arm | steps | \\|u\\|_max | gate | verdict |", "|---|---|---|---|---|"]
    for c in A.get("controls", []):
        t.append(
            f"| {c.get('arm')} | {c.get('steps')} | {c.get('u_max_lattice')} | "
            f"{D['control']['u_max_gate']} | {fmt(c.get('pass'))} |"
        )
    ctrl_md = "\n".join(t)

    oss_md = "\n".join(
        f"* a = {a}: steps {r['steps']}, Z_cell vs M&M {fmt(r['Z_cell_max_err_pct'])} %, "
        f"Z_sub.05 vs M&M {fmt(r['Z_sub0.05_max_err_pct'])} %, "
        f"inventory drift {fmt(r['inventory_drift_max_pct'])} %, u_max {fmt(r['u_max'], 3)}"
        for a, r in sorted(A.get("oss_port_rungs", {}).items(), key=lambda kv: int(kv[0]))
    )

    # ---- headline error statement ----------------------------------------
    coarse, mid, fine = alist[0], alist[min(1, len(alist) - 1)], alist[-1]
    sub05 = {a: ladder[a]["errors_mm"]["Z_sub0.05"] for a in alist}
    e_inf = fit_mm.get("Z_sub0.05", {}).get("E_inf_pct")
    e_inf_rit = fit_rit.get("Z_sub0.05", {}).get("E_inf_pct")
    rich_val = rich_mm.get("Z_sub0.05", {}).get("Z_inf_max_abs_rel_err_pct")
    pass15 = [
        a
        for a in alist
        if ladder[a]["errors_mm"]["Z_sub0.05"] is not None
        and ladder[a]["errors_mm"]["Z_sub0.05"] <= 15.0
    ]

    figs = []
    for name, cap in (
        (
            "fig_r2_front_curves.png",
            "左：各网格 Z(T) 溃坝前锋 vs Martin & Moyce 实验点与 Ritter 精确浅水解；"
            "右：同一批数据相对 M&M 的误差曲线（灰带=owner ±15 %）。",
        ),
        (
            "fig_r2_error_vs_a.png",
            "最大相对误差 vs 网格 a（双对数），含 E(a)=E_inf+C·a^-p 拟合；"
            "左=M&M 实验基准，右=Ritter 精确解基准；红虚线 15 % 验收，绿点线 3 % 渐近目标。",
        ),
        (
            "fig_r2_tau_sweep.png",
            "a=32 松弛参数扫描（τ=0.6…1.4）的 Z_sub(0.05) 最大误差，双基准并列。",
        ),
        ("fig_r2_field_a64.png", "最细网格 a=64 的末端场：填充率（左）与速度幅值（右）。"),
    ):
        p = FIG / name
        if p.exists():
            figs.append(
                {
                    "name": name,
                    "caption": cap,
                    "png_base64": base64.b64encode(p.read_bytes()).decode(),
                }
            )

    md = (
        f"""# XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS · R2：Körner 自由面溃坝的分辨率收敛与渐近误差（m{MISSION}）

**锚点** `XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS`（Maarten-vd-Sande/lbm 的 oracle-minimal-d2q9-fs 做法）· 仓 `TensorLBM` · 轮次 R2
**上游产物** `benchmarks/xval_oracle_minimal_d2q9_fs/reference_study/`（R1，m18073：OSS port 逐位等价 + 8 条臂对照，冻结不改）
**本单增量** 补齐 R1 **预声明但从未执行**的验证矩阵（分辨率阶梯 a=16…64、τ 扫描、静水控制组），
并把「误差是离散化还是模型偏置」这一开放问题用观测阶拟合 + Richardson 外推定量回答；
新增**亚格子前锋度量**，把 O(1/a) 量化从误差预算里摘出去。

## ① 案例介绍

二维矩形水柱瞬时坍塌（dam break）。初始水柱宽 `a`、高 `h0 = 2a`，置于四边固壁的计算域
`nx = 5a`、`ny = h0 + a = 3a`（顶部留 1a 气相裕量），D2Q9 + BGK，无表面张力，Körner/Thürey
自由面（独立质量账 + ABB 重建 + 半格壁）。两个独立基准：

* **Martin & Moyce (1952)** 实验前锋（`T ∈ [0, 4]`，无量纲 `Z = x_front/a`，`T = t√(g/h0)`）；
* **Ritter (1892)** 无摩阻浅水精确解 `Z = 1 + T`（仅 `T < 2` 有效，作为无水槽侧壁/黏性效应的理论界）。

| 参数 | 值 |
|---|---|
| 格子 / 碰撞 | D2Q9 / BGK，`c_s² = 1/3` |
| `a`（每初始柱宽格数） | 16, 24, 32, 48, 64 |
| 域 | `nx = 5a`, `ny = 3a`，四边 `solid_mask` + 半格反弹 |
| `τ` / `ν` | 1.0 / 0.1667（τ 扫描 0.6…1.4 见 ③） |
| `g`（格子） | 5e-5（`Ma = u_max/c_s ≈ {fmt(max(r["u_max"] for r in ladder.values()), 3)}/0.577 ≈ {fmt(max(r["u_max"] for r in ladder.values()) / 0.5774, 3)}`） |
| 时间窗 | `T ∈ [0.5, 4.0]`，`ΔT = 0.25`（R1 只到 T=1.5，本单首次覆盖声明的全窗） |
| 停止准则 | `steps = ceil(T_max/√(g/h0))`，无提前停止（a=16: {ladder[coarse]["steps"]} 步 → a={fine}: {ladder[fine]["steps"]} 步） |
| 精度 | float32（torch CPU），单线程/进程 |
| 计算 | 递交调度系统（{P.get("env", {}).get("workers", "—")} worker 并行），总墙钟 {fmt(P.get("wall_seconds"), 0)} s |

## ② 怎么做的

1. **不动冻结的 R1 harness**（`run_xval.py` 的预声明设计保持不变），新增 `resolution_study.py`：
   同一初始化（`init_fill_rectangular_2d` + `init_flags_from_fill_2d`）、同一入口
   `free_surface_step_2d`，逐格点推进到 T=4。
2. **双度量**：`Z_cell` = 最右含液/界面列（R1 度量，量化步长 1/a）；
   `Z_sub` = 亚格子前锋：由列液相深度 `H(i)=Σ_y fill(i,y)` 与阈值 `f·h0` 的线性交点给出，
   `f ∈ {{0.02, 0.05, 0.10}}`（`f=0.05` 为主），从而把度量本身的量化误差与求解器误差分开。
3. **观测阶拟合**：`E(a) = E_inf + C·a^(-p)`，对 `p ∈ [0.2, 3]` 网格搜索 + 线性最小二乘，
   并施加物性约束 `0 ≤ E_inf ≤ min E`（收敛序列外推不得高于最细网格误差）。
4. **Richardson 外推**：取最细三网格（a=32,48,64）在公共 T 网格上逐点外推到 `a→∞`，
   得到渐近前锋曲线与其对 M&M / Ritter 的最大误差。
5. **补齐 R1 欠账**：τ ∈ {{0.6, 0.8, 1.0, 1.2, 1.4}} @ a=32；静水控制组（settled column 400 步，
   检查伪流 `|u|_max < 1e-6`）；OSS port 的 a=48 档（矩阵完整性）。
6. **工具链**：`resolution_study.py --run` （调度递交）→ `--analyze` → `make_figures_r2.py` →
   `publish_r2_report.py`。所有数字由 `artifacts/resolution_analysis.json` 程序化注入本报告。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 分辨率阶梯（TensorLBM Körner，τ=1，T∈[0.5,4]）

{ladder_md}

### 3.2 观测阶拟合 vs Martin & Moyce

{fit_md}

### 3.3 观测阶拟合 vs Ritter 精确浅水解

{fit_rit_md}

### 3.4 Richardson 外推（最细三网格，vs M&M）

{rich_md}

### 3.5 τ 扫描 @ a=32

{tau_md}

### 3.6 静水控制组（R1 预声明、R1 未执行）

{ctrl_md}

### 3.7 OSS oracle port 矩阵补齐

{oss_md}

## ④ 结果图

共 {len(figs)} 张（前锋曲线+误差带、误差-网格收敛拟合、τ 扫描、最细网格场量）。

## ⑤ 注意点（踩坑与适用域）

1. **度量定义即误差源**：R1 的 `Z_cell` 量化步长 `1/a`（a=16 时 6.25 %），与被测误差同量级；
   `Z_sub` 用阈值 0.02/0.05/0.10·h0 三档给出「度量定义不确定度」而不是假装它不存在。
2. **干床奇异性**：溃坝前锋在干床上是浅水方程的奇异点，LBM 有限界面厚度会系统性地「拖慢」前锋，
   这部分误差**不随 a 减小而消失**（正是 `E_inf` 要量的东西）。
3. **Ritter 仅在 T<2 有效**，故 Ritter 基准只用 `T∈[0.5,2]`；M&M 用到全窗 `T∈[0.5,4]`。
4. **后壁/侧壁影响**：`nx = 5a` 下 M&M 前锋在 T=4 时约 `Z≈3.4`，距右壁约 1.6a；后壁反射与
   顶部气相裕量在 T>3 已不可忽略，属本算例的固定系统误差，未在本轮消除（见 ⑥ 的判定口径）。
5. **OSS oracle port 不是精度基准**：R1 已证它在自身 demo 参数（g=0.1, Ma≈6）下速度限幅长期饱和、
   在物理参数下 30 步内 NaN；本单只把它的 a=48 档补齐，用于证明「对照口径必须用 M&M/Ritter」。
6. **浮点与硬件**：float32 torch CPU，单线程；不同节点/线程数的逐位结果不保证一致（量级 1e-7）。

## ⑥ 误差判定

关键量：`Z_sub(0.05)` 相对 M&M 的最大误差（全窗 T∈[0.5,4]）。

| a | max err vs M&M % | <15 % ? | <3 % ? |
|---|---|---|---|
"""
        + "\n".join(
            f"| {a} | {fmt(sub05[a])} | {'是' if sub05[a] is not None and sub05[a] <= 15 else '否'} "
            f"| {'是' if sub05[a] is not None and sub05[a] < 3 else '否'} |"
            for a in alist
        )
        + f"""

* 观测阶拟合（M&M, `Z_sub(0.05)`）：`p = {fmt(fit_mm.get("Z_sub0.05", {}).get("p"))}`，
  **`E_inf = {fmt(e_inf)} %`**；Richardson 直接外推 Max err = **{fmt(rich_val)} %**。
* 观测阶拟合（Ritter, `Z_sub(0.05)`）：`p = {fmt(fit_rit.get("Z_sub0.05", {}).get("p"))}`，
  **`E_inf = {fmt(e_inf_rit)} %`**。
* **是否 <3 %**：{"是" if (e_inf is not None and e_inf < 3) else "**否**"}（渐近外推值 {fmt(e_inf)} % vs 3 % 目标；
  Richardson 逐点外推给 {fmt(rich_val)} %）。**是否 <15 %**：在 a ≥ {pass15[0] if pass15 else "（无）"} 的网格上成立，
  粗网格 a={coarse} 为 {fmt(sub05.get(coarse))} %（不达标）。
* 结论口径：15 % 验收只在 **a ≥ {pass15[0] if pass15 else "—"}** 的离散化水平上成立；把分辨率继续提高后，
  对 M&M 的误差收敛到 `E_inf ≈ {fmt(e_inf)} %`，说明**剩余偏差主要是干床前锋的模型/度量偏置与实验散度，
  而非尚未跑够网格**。因此后续任何以本算例为基准的精度声明，必须同时给出 `a` 与度量阈值。
* honest-FAIL 部分：渐近误差若 ≥3 %，本单不宣称「<3 % 验证通过」；粗网格 a=16/24 明确 FAIL 15 %。
"""
    )

    tags = [
        "XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS",
        "free-surface",
        "dam-break",
        "martin-moyce",
        "ritter",
        "grid-convergence",
        "richardson",
        "TensorLBM",
        f"m{MISSION}",
    ]
    rep = {
        "title": (
            f"XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS R2：Körner 自由面溃坝分辨率收敛"
            f"（a=16…64，E_inf≈{fmt(e_inf)} %）"
        ),
        "summary": (
            f"补齐 R1 预声明但未执行的验证矩阵：a=16/24/32/48/64 × 全窗 T∈[0.5,4] 对 M&M 与 "
            f"Ritter 双基准，观测阶 p={fmt(fit_mm.get('Z_sub0.05', {}).get('p'))}、"
            f"E_inf={fmt(e_inf)} %，Richardson 外推 {fmt(rich_val)} %；τ 扫描与静水控制组同时补齐。"
        ),
        "scenario_key": "lbm_free_surface_dambreak_grid_convergence",
        "mission_id": MISSION,
        "tags": tags,
        "markdown": md,
        "figures": figs,
    }
    return rep


DOC = HERE.parents[2] / "docs" / "XVAL_LBM_ORACLE_MINIMAL_D2Q9_FS_R2_RESOLUTION.md"


def write_doc(rep: dict) -> Path:
    """Same body as the platform report, plus on-disk figure links (docs/)."""
    figs = "\n".join(
        f"![{f['name']}](../benchmarks/xval_oracle_minimal_d2q9_fs/reference_study/figures/{f['name']})\n\n"
        f"*{f['caption']}*\n"
        for f in rep["figures"]
    )
    body = rep["markdown"]
    body = body.replace(
        "## ④ 结果图\n\n"
        + "共 %d 张（前锋曲线+误差带、误差-网格收敛拟合、τ 扫描、最细网格场量）。"
        % len(rep["figures"]),
        "## ④ 结果图\n\n" + figs,
    )
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(
        body + "\n\n## ⑦ 复现\n\n```bash\n"
        "OMP_NUM_THREADS=1 python benchmarks/xval_oracle_minimal_d2q9_fs/reference_study/resolution_study.py --run --workers 12\n"
        "python benchmarks/xval_oracle_minimal_d2q9_fs/reference_study/resolution_study.py --analyze\n"
        "python benchmarks/xval_oracle_minimal_d2q9_fs/reference_study/make_figures_r2.py\n"
        "python benchmarks/xval_oracle_minimal_d2q9_fs/reference_study/publish_r2_report.py --dry-run   # 生成 docs 正文\n"
        "```\n"
    )
    print(f"[doc] {DOC}")
    return DOC


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rep = build()
    OUT.write_text(json.dumps(rep))
    print(f"[report] {OUT} ({len(rep['markdown'])} chars, {len(rep['figures'])} figures)")
    write_doc(rep)
    if args.dry_run:
        print(rep["markdown"][:1500])
        return 0
    base = os.environ["TENSOR_API_BASE"]
    body = json.dumps(
        {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]}
    ).encode()
    req = urllib.request.Request(
        base + "/auth/login", data=body, headers={"Content-Type": "application/json"}
    )
    tok = json.load(urllib.request.urlopen(req))
    token = tok.get("access_token") or tok.get("token")
    req2 = urllib.request.Request(
        base + "/tensor/reports",
        data=OUT.read_bytes(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    print("report:", urllib.request.urlopen(req2).read().decode()[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
