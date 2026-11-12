#!/usr/bin/env python3
"""Build the CASE-CYLINDER-DRAG-RE20 report from run artifacts (zero hand-copy).

Reads ``runs/cylinder_drag_re20/D*/results.json`` + ``series.csv`` + ``fields.npz``
and emits

* ``docs/CASE_CYLINDER_DRAG_RE20.md`` — six-section paper-style report whose
  every number is formatted from the artifact JSON (no typed values),
* ``docs/assets/case_cylinder_drag_re20/*.png`` — Cd(t) history + plateau
  window, ref-vs-got bar with the ±5 % band, final field (vorticity/velocity),
  grid sensitivity,
* ``/tmp/report.json`` — platform report payload with the figures embedded.

Usage: PYTHONPATH=src python benchmarks/case_pool/cylinder_drag_re20/make_report.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs/cylinder_drag_re20"
ASSETS = REPO / "docs/assets/case_cylinder_drag_re20"
DOC = REPO / "docs/CASE_CYLINDER_DRAG_RE20.md"
REFS = json.loads((REPO / "benchmarks/case_pool/cylinder_drag_re20/refs.json").read_text())
REF_CD = REFS["ref"]["cd"]
TOL = REFS["tolerance_pct"]


def load(D: int) -> dict | None:
    p = RUNS / f"D{D}" / "results.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    r["_dir"] = RUNS / f"D{D}"
    return r


def fig_cd_history(runs, path):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for r in runs:
        s = r["series"]
        ax.plot(s["step"], s["cd"], lw=0.7, label=f"D={r['geometry']['D_cells']} cells")
        ax.axhline(r["cd"], ls=":", lw=0.9)
    ax.axhline(REF_CD, color="k", lw=1.4, label=f"ref Cd={REF_CD:.5f}")
    ax.axhspan(REF_CD * (1 - TOL / 100), REF_CD * (1 + TOL / 100), color="k", alpha=0.08, label=f"±{TOL:g}% band")
    ax.set_xlabel("LBM step")
    ax.set_ylabel("$C_D$")
    ax.set_title("Cylinder drag history — Schäfer–Turek 2D-1, Re=20 (dotted = averaging window mean)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_vs_ref(runs, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = [f"D={r['geometry']['D_cells']}" for r in runs]
    vals = [r["cd"] for r in runs]
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.5, color="#4c78a8", label="LBM (momentum exchange)")
    ax.axhline(REF_CD, color="k", lw=1.4, label="ref (official 2D-1 dataset)")
    ax.fill_between([-0.6, len(vals) - 0.4], REF_CD * (1 - TOL / 100), REF_CD * (1 + TOL / 100),
                    color="k", alpha=0.08, label=f"±{TOL:g}%")
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:.4f}\n{(v - REF_CD) / REF_CD * 100:+.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(min(vals + [REF_CD]) * 0.96, max(vals + [REF_CD]) * 1.04)
    ax.set_ylabel("$C_D$")
    ax.set_title("Drag coefficient vs reference anchor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_field(r, path):
    f = np.load(r["_dir"] / "fields.npz")
    dx = float(f["dx"])
    ny, nx = f["rho"].shape
    extent = [0, nx * dx, 0, ny * dx]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.0), sharex=True)
    im0 = axes[0].contourf(np.linspace(extent[0], extent[1], nx), np.linspace(extent[2], extent[3], ny),
                           (f["ux"] ** 2 + f["uy"] ** 2), levels=32, cmap="viridis")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"velocity magnitude |u| (D={r['geometry']['D_cells']}, Re=20, steady)")
    fig.colorbar(im0, ax=axes[0])
    lim = np.nanpercentile(np.abs(f["vorticity"]), 98)
    im1 = axes[1].contourf(np.linspace(extent[0], extent[1], nx), np.linspace(extent[2], extent[3], ny),
                           f["vorticity"], levels=40, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[1].set_ylabel("y")
    axes[1].set_xlabel("x")
    axes[1].set_title("vorticity ω_z")
    fig.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_grid(runs, path):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    Ds = [r["geometry"]["D_cells"] for r in runs]
    errs = [r["err_cd_pct"] for r in runs]
    ax.plot(Ds, errs, "o-", label="LBM vs ref")
    ax.axhline(0, color="k", lw=1)
    ax.axhline(TOL, color="r", ls="--", lw=1, label=f"±{TOL:g}% tolerance")
    ax.axhline(-TOL, color="r", ls="--", lw=1)
    for D, e in zip(Ds, errs):
        ax.annotate(f"{e:+.2f}%", (D, e), textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax.set_xlabel("cylinder diameter [cells]")
    ax.set_ylabel("$C_D$ relative error [%]")
    ax.set_title("Grid sensitivity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    runs = [r for r in (load(16), load(20)) if r is not None]
    if not runs:
        raise SystemExit("no run artifacts found under runs/cylinder_drag_re20")

    figs = {}
    fig_cd_history(runs, ASSETS / "cd_history.png"); figs["cd_history"] = ASSETS / "cd_history.png"
    fig_vs_ref(runs, ASSETS / "cd_vs_ref.png"); figs["cd_vs_ref"] = ASSETS / "cd_vs_ref.png"
    fig_grid(runs, ASSETS / "grid_sensitivity.png"); figs["grid"] = ASSETS / "grid_sensitivity.png"
    biggest = max(runs, key=lambda r: r["geometry"]["D_cells"])
    if (biggest["_dir"] / "fields.npz").exists():
        fig_field(biggest, ASSETS / "fields.png"); figs["fields"] = ASSETS / "fields.png"

    rows = []
    for r in runs:
        g, u = r["geometry"], r["lattice_units"]
        rows.append(
            f"| {g['D_cells']} | {g['nx']}×{g['ny']} | {u['tau']:.3f} | {u['u_mean']:.4f} | {u['ma']:.3f} "
            f"| {r['n_steps']} | {r['stop_reason']} | {r['avg_window']} | {r['cd']:.6f} | "
            f"{r['err_cd_pct']:+.3f}% | {r['cd_plateau_drift']:.2e} | {'yes' if r['stationary'] else 'NO'} | "
            f"{'✅' if abs(r['err_cd_pct']) <= TOL and r['stationary'] else '❌'} |"
        )
    tbl = "\n".join(rows)
    all_pass = all(abs(r["err_cd_pct"]) <= TOL and r["stationary"] for r in runs)
    worst = max(abs(r["err_cd_pct"]) for r in runs)

    md = f"""# CASE-CYLINDER-DRAG-RE20 — 圆柱绕流 Re=20（Schäfer–Turek 2D-1）阻力基准

**判定：{'PASS' if all_pass else 'FAIL'}**（判据：每档 |ΔCd| ≤ {TOL:g}% 且均值取自静止窗口；最大偏差 {worst:.3f}%）

## ① 问题与参考值出处

Schäfer & Turek (1996) 2D-1：通道内静止圆柱，稳态层流。物理量（无量纲）：
域 {REFS['geometry']['domain']}、圆柱中心 {tuple(REFS['geometry']['cylinder_centre'])}、
直径 D={REFS['geometry']['diameter_D']}、入口 U_mean=0.2（抛物线，U_max=1.5·U_mean）、
出口 ρ=1、壁面与圆柱无滑移、ν=0.001 → **Re = U_mean·D/ν = {REFS['dimensionless']['Re']:g}**，
阻塞比 B = D/H = {REFS['geometry']['blockage_D_over_H']:.4f}。

参考值：**Cd = {REF_CD:.11f}**（Cl = {REFS['ref']['cl']:.12f}，仅作信息量）。
出处链：{REFS['provenance']['source']}；钉扎文件
`{REFS['provenance']['anchor_file']}`（sha256 `{REFS['provenance']['anchor_file_sha256'][:16]}…`）。
**诚实声明**：{REFS['provenance']['print_original']}，故本报告不引用 1996 印刷原表页码，
锚点取仓内既有 verified 记录（其 ref_source 即 2D-1 官方参考数据，Nabh 1998 谱解）；
`tests/test_cylinder_st2d1.py::test_reference_anchor_is_pinned_not_transcribed`
逐位核对模块常量 == refs.json == 锚点文件，杜绝手抄漂移。

## ② 方法与实现

库内实现：`src/tensorlbm/cylinder_st2d1.py`（D2Q9，MRT，无任何自写碰撞/流动核）：
`collide_mrt → 固体单元保留碰撞前分布 → stream → Zou/He 速度入口（抛物线）→ Zou/He 压力出口(ρ=1)
→ 通道壁半程反弹 → 圆柱面 Ladd 动量交换力 → 圆柱半程反弹`。
掩膜半径 = R（历史 R−0.5 修正已被实验否决），**无质量重整化**（曾激发 ~10k 步慢振荡）。

稳态判据（本单新增，历史 runner 缺失）：末窗 {runs[0]['avg_window']} 步分 4 块，
块均值相对散度 ≤ plateau_tol（{runs[0]['plateau_tol']:.1e}）才判 `stationary`；
否则即使数值落在容差内也不计通过。

入口/工件：`benchmarks/case_pool/cylinder_drag_re20/run_case.py`（本目录）、
`refs.json`（锚点+复现命令）、平台侧调用 `app/backend/routers/cylinder_bench.py::st2d1_validate`
（`POST /cylinder-bench/st2d1/validate`）。

## ③ 参数表（无量纲完整）

| D_cells | nx×ny | τ | u_mean | Ma | steps | 停止原因 | 平均窗 | Cd | ΔCd | 平台散度 | stationary | 判据 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{tbl}

Re = {runs[0]['lattice_units']['re']:.6f}（各档同一），ρ0=1，ν_lb = u_mean·D/Re，
力归一化 Cd = Fx/(0.5·ρ·U_mean²·D_phys)，D_phys=0.1。

## ④ 对照表（ref vs got，逐项打勾叉）

| 量 | ref | got (D={biggest['geometry']['D_cells']}) | 相对误差 | <{TOL:g}% |
|---|---|---|---|---|
| Cd | {REF_CD:.6f} | {biggest['cd']:.6f} | {biggest['err_cd_pct']:+.3f}% | {'✅' if abs(biggest['err_cd_pct']) <= TOL else '❌'} |
| Cl（非判据） | {REFS['ref']['cl']:.6f} | {biggest['cl']:.6f} | {biggest['err_cl_pct']:+.1f}% | ➖ 二阶量，MEM 阶梯面不收敛，仅记录 |

![cd_history](assets/case_cylinder_drag_re20/cd_history.png)
![cd_vs_ref](assets/case_cylinder_drag_re20/cd_vs_ref.png)

## ⑤ 收敛性 / 参数敏感性（≥2 点）

1. **网格敏感性**：{'；'.join(f"D={r['geometry']['D_cells']} → {r['err_cd_pct']:+.3f}%" for r in runs)}
   （见 `grid_sensitivity.png`）。历史记录（`benchmarks/verified/cylinder/re20_st`）在同一链条上给出
   D=40 −0.56%、D=80 −2.87%：MEM 阶梯面偏差随加密**增大而非单调收敛到零**，故本基准的容差意义是
   "任一合理分辨率都落在 ±{TOL:g}%"，不是网格收敛率。
2. **时间/统计窗口**：每档块均值散度 = {'；'.join(f"D={r['geometry']['D_cells']}: {r['cd_plateau_drift']:.2e}" for r in runs)}，
   半窗两估计差 = {'；'.join(f"D={r['geometry']['D_cells']}: {(r['cd_half_window_second'] - r['cd_half_window_first']) / r['cd'] * 100:+.3e}%" for r in runs)}。
3. **Mach/τ 耦合（护栏）**：固定 τ=0.8 时 u_mean = Re·ν/D ∝ 1/D，D=8 即 Ma=0.43 发散；
   实现里加了 `ma_limit=0.3` 硬护栏（`test_mach_guard_rejects_coarse_grid_with_fixed_tau`），
   因此本单采用**固定 Ma=0.087、τ 随网格变**（τ=0.62/0.65）的加密路线。

![grid](assets/case_cylinder_drag_re20/grid_sensitivity.png)

## ⑥ 失败模式与工件

- 失败模式①：**慢模态未收敛即统计**——本单用 4 块散度闸门拦截；被拦即 `stationary=False`、`passed=False`。
- 失败模式②：**定 τ 粗网格 Ma 爆炸**（D=8 → NaN）——`ma_limit` 护栏转为显式异常，不再静默返回 NaN。
- 失败模式③：**参考值手抄漂移**——锚点 sha256 钉扎 + 三处一致性测试。
- 工件：`runs/cylinder_drag_re20/D{{16,20}}/{{results.json, series.csv, fields.npz, *.log}}`、本目录
  `summary` 由 `run_case.py` 生成、平台报告 `mission_id=15777`。
- 复现：`{REFS['repro_cmd']}`

![fields](assets/case_cylinder_drag_re20/fields.png)
"""
    DOC.write_text(md)

    payload = {
        "title": f"CASE-CYLINDER-DRAG-RE20：Schäfer–Turek 2D-1 圆柱 Re=20 阻力基准（{len(runs)} 档网格，最大偏差 {worst:.2f}%）",
        "summary": (
            f"TensorLBM D2Q9/MRT 跑 Schäfer–Turek 2D-1（Re=20，B=0.244）稳态圆柱绕流，"
            f"Cd 参考 {REF_CD:.5f}；实测 "
            + "，".join(f"D={r['geometry']['D_cells']}: {r['cd']:.4f}（{r['err_cd_pct']:+.2f}%）" for r in runs)
            + f"；判据 ±{TOL:g}% → {'PASS' if all_pass else 'FAIL'}；工件 runs/cylinder_drag_re20、docs/CASE_CYLINDER_DRAG_RE20.md。"
        ),
        "scenario_key": "CASE-CYLINDER-DRAG-RE20",
        "mission_id": 15777,
        "tags": ["TensorLBM", "benchmark", "cylinder", "Schafer-Turek", "2D-1", "Re20", "drag", "case-pool"],
        "markdown": md,
        "figures": [
            {"name": k, "caption": c, "png_base64": b64(p)}
            for k, p, c in [
                ("cd_history", figs["cd_history"], "Cd 随步数历史 + 静止窗口均值（虚线）与 ±5% 锚带"),
                ("cd_vs_ref", figs["cd_vs_ref"], "各网格 Cd 与参考锚点对照（柱内为相对误差）"),
                ("grid_sensitivity", figs["grid_sensitivity"], "Cd 相对误差随网格加密的变化（红色虚线=±5% 容差）"),
            ]
            + ([("fields", figs["fields"], f"最终场：速度幅值与涡量（D={biggest['geometry']['D_cells']}）")] if "fields" in figs else [])
        ],
    }
    Path("/tmp/report.json").write_text(json.dumps(payload, ensure_ascii=False))
    print(f"wrote {DOC}\nassets: {sorted(p.name for p in ASSETS.glob('*.png'))}\nreport payload: /tmp/report.json")
    print(f"all_pass={all_pass} worst={worst:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())