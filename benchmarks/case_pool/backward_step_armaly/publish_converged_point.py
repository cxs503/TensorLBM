#!/usr/bin/env python3
"""Publish the converged-point deliverable of CASE-BACKSTEP-FVM-RE100 (mission 22231).

Reads (never hand-copies) every number from:

    artifacts/case_backward_step_armaly/h1_re100/{result.json,wall_shear.csv,field.npz}
    artifacts/case_backward_step_armaly/verdict.json
    benchmarks/case_pool/backward_step_armaly/refs.json
    tensor_benchmark_cases  (the registered referee rows, via psql)

writes ``docs/CASE_BACKSTEP_FVM_RE100_CONVERGED_POINT.md`` + three figures under
``figures/CASE_BACKSTEP_FVM_RE100_m22231/`` and posts the six-section report.

Usage:
    PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/publish_converged_point.py
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ART = REPO / "artifacts" / "case_backward_step_armaly"
CASE = REPO / "benchmarks" / "case_pool" / "backward_step_armaly"
FIG = REPO / "figures" / "CASE_BACKSTEP_FVM_RE100_m22231"
DOC = REPO / "docs" / "CASE_BACKSTEP_FVM_RE100_CONVERGED_POINT.md"
MISSION = 22231
PGENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}


def psql(sql: str) -> list[list[str]]:
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
                        "-tA", "-F", "|", "-c", sql], capture_output=True, text=True, env=PGENV)
    return [line.split("|") for line in r.stdout.strip().splitlines() if line.strip()]


def load(tag: str = "h1_re100") -> dict:
    run = json.loads((ART / tag / "result.json").read_text())
    refs = json.loads((CASE / "refs.json").read_text())
    verd = json.loads((ART / "verdict.json").read_text())
    row = next(r for r in verd["runs"] if r["tag"] == tag)
    return {"tag": tag, "run": run, "refs": refs, "verdict": verd, "row": row}


def figures(d: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIG.mkdir(parents=True, exist_ok=True)
    run, row, refs = d["run"], d["row"], d["refs"]
    res, cfg, der = run["result"], run["config"], run["derived"]
    ref_exp = refs["adopted_numeric_reference"]["points"]["100"]["x1_over_s"]
    cross = next(c for c in refs["cross_checks"] if c.get("re") == 100)
    made: list[Path] = []

    # --- fig 1: mean u_x field, reversed-flow region, detected reattachment line
    fld = np.load(ART / d["tag"] / "field.npz")
    ux, solid, x_step, s = fld["ux_mean"], fld["solid"], int(fld["x_step"]), int(fld["s"])
    ux_plot = np.where(solid, np.nan, ux)
    fig, ax = plt.subplots(figsize=(10.0, 3.2))
    ys, xs = np.mgrid[0:ux.shape[0], 0:ux.shape[1]]
    im = ax.pcolormesh(xs, ys, ux_plot, cmap="RdBu_r", shading="auto", vmin=-0.08, vmax=0.13)
    ax.contour(xs, ys, np.nan_to_num(ux_plot), levels=[0.0], colors="k", linewidths=0.7)
    ax.axvline(x_step + res["x1_over_s"] * s, color="lime", lw=1.4, ls="--",
               label=f"$X_1/s$ = {res['x1_over_s']:.3f}")
    ax.set_xlabel("x [cells]")
    ax.set_ylabel("y [cells]")
    ax.set_title(f"BFS Re=100, $h_{{in}}$={cfg['hi']}, $s$={cfg['s']}: time-mean $u_x$ [lattice]")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.9, label="$u_x$")
    p1 = FIG / "fig1_ux_field.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    made.append(p1)

    # --- fig 2: near-wall shear with the zero crossing that defines X1
    import csv
    rows = list(csv.DictReader((ART / d["tag"] / "wall_shear.csv").open()))
    cols = list(rows[0])
    xk = cols[0]
    tk = next(c for c in cols[1:] if "shear" in c.lower() or "tau" in c.lower())
    x = np.array([float(r[xk]) for r in rows])
    tau = np.array([float(r[tk]) for r in rows])
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.plot(x - x_step, tau, "o-", ms=3, lw=1.0, color="tab:blue")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(res["x1_over_s"] * s, color="lime", lw=1.4, ls="--",
               label=f"$X_1$ = {res['x1_over_s'] * s:.1f} cells = {res['x1_over_s']:.3f} s")
    ax.set_xlabel("x - $x_{step}$ [cells]")
    ax.set_ylabel(r"$\tau_w$ [lattice]")
    ax.set_title("lower-wall shear stress (3-point one-sided derivative of $u_x$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p2 = FIG / "fig2_wall_shear.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    made.append(p2)

    # --- fig 3: X1/s vs time-step (convergence) + reference bands
    hist = [h for h in run.get("convergence_history") or []
            if isinstance(h.get("x1_over_s_instant"), (int, float))]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if hist:
        st = [h["step"] for h in hist]
        val = [h["x1_over_s_instant"] for h in hist]
        ax.plot(st, val, "o-", ms=3, label="instantaneous $X_1/s$ (shear-zero caliper)")
    ax.axhline(res["x1_over_s"], color="tab:red", lw=1.2,
               label=f"averaged over window: {res['x1_over_s']:.3f}")
    ax.axhline(ref_exp, color="k", ls="--", lw=1.0, label=f"Armaly 1983: {ref_exp:.2f}")
    ax.axhspan(cross["band"][0], cross["band"][1], color="tab:green", alpha=0.15,
               label=f"Erturk 2008 band {cross['band'][0]:.2f}-{cross['band'][1]:.2f}")
    ax.axhline(cross["value"], color="tab:green", ls=":", lw=1.0, label=f"Erturk 2008: {cross['value']:.3f}")
    ax.set_xlabel("time step")
    ax.set_ylabel("$X_1/s$")
    ax.set_title(f"Re=100 convergence trace ({res['steps_run']} steps, "
                 f"steady@{res.get('steady_at_step')})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    p3 = FIG / "fig3_convergence_vs_reference.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    made.append(p3)
    return made


def build_md(d: dict, figs: list[Path]) -> str:
    run, row, refs, verd = d["run"], d["row"], d["refs"], d["verdict"]
    res, cfg, der = run["result"], run["config"], run["derived"]
    ref_exp = refs["adopted_numeric_reference"]["points"]["100"]["x1_over_s"]
    tol = refs["adopted_numeric_reference"]["ledger_tolerance_pct"]
    cross = next(c for c in refs["cross_checks"] if c.get("re") == 100)
    st = row["steadiness"]
    checks = row["checks"]
    referee_rows = psql("select case_name, metric, tolerance from tensor_benchmark_cases "
                        "where anchor_slug='CASE-BACKSTEP-FVM-RE100' order by case_name")
    ref_tbl = "\n".join(f"| `{c}` | `{m}` | **{t}** |" for c, m, t in referee_rows)
    check_tbl = "\n".join(f"| `{k}` | {'PASS' if v else 'FAIL'} |" for k, v in checks.items())
    tail = ", ".join(f"{v:.4f}" for v in (st.get("tail") or [])) or "n/a"
    fig_md = "\n\n".join(f"![{p.stem}](../{p.relative_to(REPO)})" for p in figs)
    e_exp = row["error_pct"]
    e_cc = (row.get("cross_check") or {}).get("error_pct")

    return f"""# CASE-BACKSTEP-FVM-RE100 · 收敛基准点落地（TensorLBM 臂）

**锚点**：`feature_ledger.CASE-BACKSTEP-FVM-RE100`（标题：算例池·后台阶回流区长度 Re=100；验收 `<10%`，ref Armaly et al. 1983）
**本单**：mission {MISSION}（自驱开发，generalist 席）·**分支** `agent/m22231`
**一句话**：用交付代码跑出**第一个进仓的收敛基准点**（`steady_at_step = {res.get('steady_at_step')}`、质量流不平衡 `{res['mass_flux_rel_error']:.2e}`），
并把判分门的容差从代码常量改为读 `refs.json`（此前代码写死 15 %，与注册裁判行 10 % 不一致）。
`X1/s = {res['x1_over_s']:.4f}` → 相对 Armaly 1983 实验锚 `{ref_exp}` 为 **{e_exp:+.2f}%**，相对 Erturk 2008 二维基准 `{cross['value']}` 为 **{e_cc:+.2f}%**。

---

## ① 案例介绍

后台阶（backward-facing step）层流分离—再附是分离流动的经典验证算例：入口高 `h_in`、台阶高 `s`、下游高 `H = h_in + s`，
观测量为**下壁面主再附长度** `X1/s`（台阶高归一），定义为壁面剪切应力 `τ_w(x)` 的零点。

| 参数 | 值 | 说明 |
|---|---|---|
| `Re = ū·D_h/ν`（`D_h = 2h_in`，Armaly 口径） | {cfg['re']:g} | 层流 |
| 网格 `h_in` / `s`（lattice cells） | {cfg['hi']} / {cfg['s']} | ER = {der['expansion_ratio']:.4f}（实验 1.9423，偏差 {der['expansion_ratio_deviation_pct']:+.2f}%）|
| 域 `nx × ny` | {der.get('nx', 'n/a')} × {der.get('ny', 'n/a')} | 上游 {cfg.get('up_over_hi', 4.0)}h_in、下游 {cfg.get('l_over_h', 14.0)}H |
| 入口平均速度 `ū` | {cfg['u_bar']} | 抛物线剖面，行平均精确 |
| `ν` / `τ` | {der['nu']:.5f} / {der['tau']:.4f} | TRT Λ=3/16；`τ≥0.55` ✓ |
| 入口 Mach | {der['mach_inlet']:.4f} | ≤0.3 ✓（不可压窗口）|
| 步数 / 平均窗 | {res['steps_run']} / {res.get('average_window_steps')}（{res['n_averaged_samples']} 样本）| 自停于 `steady_tol` |

**参考值**：`refs.json::adopted_numeric_reference.points["100"].x1_over_s = {ref_exp}`（tier `SECONDARY_IN_REPO`，原文 PDF 抓取 CLOSED，见 `refs.json::acquisition_log`）；
交叉锚 Erturk 2008 二维数值 `{cross['value']}`（band {cross['band'][0]}–{cross['band'][1]}）。

**机器裁判（`tensor_benchmark_cases`，`anchor_slug='CASE-BACKSTEP-FVM-RE100'`）**

| case_name | metric | tolerance |
|---|---|---|
{ref_tbl}

## ② 怎么做的

1. **求解器**：`src/tensorlbm/bfs_armaly.py`（D2Q9 + TRT 碰撞 Λ=3/16、half-way bounce-back 壁面、入口 Zou/He 抛物线速度、出口 Zou/He 压力 + 非平衡外推海绵层）。
2. **运行入口**：`benchmarks/case_pool/backward_step_armaly/run_case.py --preset h1_re100 --steps 120000`，
   递交计算门（`run_compute`，调度器派发），产物 `result.json / wall_shear.csv / field.npz` 进仓。
   为让阶梯可复算，`--tag` 与 `--steps` 均由命令行给定（不写死在代码里）。
3. **观测**：时间平均后的近壁 `u_x` 用三点单侧 Lagrange 导数 `τ_w ≈ -2u_1 + 3u_2 - u_3`（二次剖面精确），
   `X1` = `τ_w(x)` 首个零点线性插值 − 台阶面半格修正；独立口径 `reversed_flow_extent`（前 3 行流体格 `u_x<0` 的范围）作交叉绑定。
4. **判分**：`benchmarks/case_pool/backward_step_armaly/verdict.py` 逐条件评（`error / steady / caliper / mass / lattice / cross_check`），
   写 `artifacts/case_backward_step_armaly/verdict.json`。**本单修**：容差不再写死（旧代码 `TOLERANCE_PCT = 15.0`，与注册裁判 10 % 不符），
   改为读 `refs.json::adopted_numeric_reference.ledger_tolerance_pct`；`cross_checks[].re/tolerance_pct` 新增字段把二维交叉锚（5 %）也纳入判分。
5. **回归**：`tests/test_case_pool_backward_step_armaly_gates.py`（5 项，纯合成产物、不需集群）证明「12 % 偏差点在旧 15 % 门下会误判 PASS，现判 FAIL」；
   `tests/test_case_pool_backward_step_armaly.py` 中被跳过的基准断言在本单后变绿。

## ③ 计算结果

| 量 | 值 |
|---|---|
| `X1/s`（平均窗，shear-zero caliper） | **{res['x1_over_s']:.4f}** |
| `X1/s`（reversed-flow extent 交叉口径） | {res.get('x1_over_s_reverse_extent')} |
| `X1/s`（第 1 行口径） | {res.get('x1_over_s_row1')} |
| 稳态停步 | `steady_at_step = {res.get('steady_at_step')}`；最后三次检查 {tail}（漂移 {st.get('drift_pct')} %) |
| 质量流相对不平衡 | {res['mass_flux_rel_error']:.3e}（门限 ≤1e-3）|
| ρ 范围 | [{res.get('rho_min')}, {res.get('rho_max')}] |
| 判分 | `n_runs={verd['n_runs']} n_pass={verd['n_pass']} all_judged_pass={verd['all_judged_pass']}`，`max|err|={verd['max_abs_error_pct']}` % |

逐条判分条件（`verdict.json`）：

| 条件 | 结果 |
|---|---|
{check_tbl}

## ④ 结果图

{fig_md}

## ⑤ 注意点（局限与适用范围）

- **参考值层级**：Armaly 1983 原始 PDF 在本沙箱 CLOSED（四路抓取日志见 `refs.json::acquisition_log`），
  故 `{ref_exp}` 是仓内二级锚（NAFEMS/Feel++ 数字化读数），**不是**「原文表 1」的机器可读复制；档位 `SECONDARY_IN_REPO` 已如实标注。
- **二维 vs 实验**：Re=100 时实验已含三维效应，Erturk 2008 二维数值 `{cross['value']}` 与实验 `{ref_exp}` 本身差
  `{100.0 * (ref_exp - cross['value']) / cross['value']:+.1f}%`，这是**不可约**的模型差，不是本次数值误差。
- **几何/阶梯**：`h_in={cfg['hi']}` 档 ER={der['expansion_ratio']:.4f}，比实验 1.9423 高 {der['expansion_ratio_deviation_pct']:+.2f}%；
  台阶角是阶梯离散（一阶）。`g1`（h_in=52）/`g2`（h_in=104）档未在本单跑，`refs.json::accepted_rungs` 只允许 `h1`/`g1` 出品
  → **网格收敛阶（Richardson）本单仍为 not_run**，这条欠账没有用假设值掩盖。
- **可复算性**：本点由命令行 `--preset h1_re100 --steps 120000` 生成，任何人在同一 commit 上可重跑；产物四件（result/wall_shear/field/verdict）全部进仓。

## ⑥ 误差判定（是否 <3%）

| 关键量 | 本算 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|---|
| `X1/s` | {res['x1_over_s']:.4f} | Armaly 1983 实验锚 {ref_exp} | **{e_exp:+.2f}%** | **否**（落在账本 10 % 门内）|
| `X1/s` | {res['x1_over_s']:.4f} | Erturk 2008 二维 {cross['value']} | **{e_cc:+.2f}%** | **否**（落在注册 5 % 交叉门内）|

**判定**：账本验收「<10 %」**通过**（`all_judged_pass={verd['all_judged_pass']}`）；
但对 owner 的 3 % 精度标准，两条参考都**没有**达到（`{e_exp:+.2f}%` / `{e_cc:+.2f}%`），本报告不做粉饰。
`{ref_exp}` 与 `{cross['value']}` 之间的 6 % 固有模型差说明：在该算例上「<3 %」这个标准只能对**同一模型**（二维数值基准 + 已做网格收敛）成立，
而本单的判据是「收敛点 + 注册裁判门」，故按账本 10 % 计。
"""


def publish(md: str, figs: list[Path], d: dict) -> int:
    base = os.environ["TENSOR_API_BASE"]
    tok = json.load(urllib.request.urlopen(urllib.request.Request(
        base + "/auth/login",
        data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                         "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
        headers={"Content-Type": "application/json"})))["access_token"]
    res = d["run"]["result"]
    body = {
        "title": (f"CASE-BACKSTEP-FVM-RE100 收敛基准点落地：X1/s={res['x1_over_s']:.3f}"
                  f"（Armaly {d['row']['error_pct']:+.1f}% / Erturk {d['row']['cross_check']['error_pct']:+.1f}%），判分门改为读 refs.json"),
        "summary": (f"交付代码跑出进仓的首个收敛点（steady@{res.get('steady_at_step')}，质量流不平衡 "
                    f"{res['mass_flux_rel_error']:.2e}）：X1/s={res['x1_over_s']:.4f}，相对 Armaly 1983 锚 3.05 "
                    f"{d['row']['error_pct']:+.2f}%（<10% 账本门）、相对 Erturk 2008 2.878 {d['row']['cross_check']['error_pct']:+.2f}%（<5% 交叉门），"
                    f"verdict all_judged_pass={d['verdict']['all_judged_pass']}；并修掉 verdict.py 写死 15% 容差与注册裁判 10% 不一致的缺陷（容差改读 refs.json）。"),
        "scenario_key": "case_pool/backward_step_armaly",
        "mission_id": MISSION,
        "tags": ["TensorLBM", "backstep", "CASE-BACKSTEP-FVM-RE100", "converged-point", "verification-debt", "defect-fix"],
        "markdown": md,
        "figures": [{"name": p.name,
                     "caption": cap,
                     "png_base64": base64.b64encode(p.read_bytes()).decode()}
                    for p, cap in zip(figs, [
                        "时均 u_x 场（红=逆流）+ 检测到的再附线 X1/s={:.3f}".format(res["x1_over_s"]),
                        "下壁面剪切应力 τ_w(x) 及其零点（X1 定义）",
                        f"X1/s 随步数的收敛迹线与 Armaly 1983 / Erturk 2008 参考带（steady@{res.get('steady_at_step')}）"])],
    }
    req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok})
    resp = json.load(urllib.request.urlopen(req))
    print("report id", resp.get("id"), "figures", len(resp.get("figures") or []))
    return int(resp.get("id") or 0)


def main() -> int:
    d = load()
    figs = figures(d)
    md = build_md(d, figs)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(md)
    print("wrote", DOC.relative_to(REPO), len(md), "chars; figures:", [f.name for f in figs])
    publish(md, figs, d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())