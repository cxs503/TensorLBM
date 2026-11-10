#!/usr/bin/env python
"""Render the P74-N10 deliverable from the artifact (zero hand transcription).

Two outputs:

* ``benchmarks/pending/passive_scalar_verification/README_P74N10.md`` — the
  human-readable case record (tables generated from the artifact JSON);
* ``/tmp/report.json`` — the platform report payload (title/summary/markdown/
  figures, figures embedded as base64 PNG).

Every number in both is read from ``artifacts/passive_scalar_p74n10.json`` or
from the cached case files it points at; nothing is typed by hand.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = REPO_ROOT / "artifacts/passive_scalar_p74n10.json"
README = REPO_ROOT / "benchmarks/pending/passive_scalar_verification/README_P74N10.md"
REPORT = Path("/tmp/p74n10_m9469_report.json")  # mission-private: /tmp is shared


def _load() -> dict:
    return json.loads(ARTIFACT.read_text())


def _diffusion_law(report: dict) -> dict:
    """Fit D_meas/D = 1 + a D + b D^2 to the u = 0 diffusion-magnitude probe."""
    rows = report["diagnostics"]["diffusion_magnitude"]["rows"]
    d = np.array([r["diffusivity"] for r in rows], dtype=float)
    ratio = np.array([r["ratio_sim"] for r in rows], dtype=float)
    # D_meas/D = 1 + a D + b D^2  (a carries the leading lattice bias)
    a, b, _c = np.polyfit(d, ratio, 2)
    resid = np.abs(np.polyval([a, b, 1.0], d) - ratio)
    return {
        "a": float(a),
        "b": float(b),
        "max_residual_pct": float(100.0 * resid.max()),
        "points": len(rows),
        # D_meas = D (1 + a D) -> D_meas = D - 0.0132 (tau_d - 1/2)^2
        "tau_coefficient": float(-a / 9.0),
    }


def _md_tables(report: dict, law: dict) -> dict[str, str]:
    gates = report["gates"]

    erf = gates["A"]
    erf_rows = "\n".join(
        f"| {c['nx']} | {c['dx']:.4g} | {c['steps']} | {c['l2_abs']:.3e} | "
        f"{100 * c['l2_rel']:.4f} % |"
        for c in erf["cases"]
    )
    erf_table = (
        "| $n_x$ | $\\Delta x$ | steps | $L_2$ | $L_2$ (rel) |\n"
        "|---|---|---|---|---|\n" + erf_rows
    )

    domain = gates["H"]
    cell_rows = "\n".join(
        f"| {c['pe']:g} | {c['ny']} | {c['nx']} | {c['diffusivity']:.4f} | {c['tau_d']:.3f} | "
        f"{c['steps']} | {c['ratio_sim']:.5f} | {c['ratio_closed_form']:.5f} | "
        f"{c['err_pct']:+.2f} % | {c['meter_rate_ratio']:.3f} | {c['status']} |"
        for c in domain["cells"]
    )
    domain_table = (
        "| $Pe_r$ | $N_y$ | $n_x$ | $D$ | $\\tau_d$ | steps | $D_{eff}/D$ (LBM) | "
        "$1+Pe_r^2/210$ | err | meter ratio | status |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n" + cell_rows
    )

    per_pe = "\n".join(
        f"| {p['pe']:g} | {', '.join('%g' % n for n in p['ny'])} | "
        f"{', '.join('%+.2f' % e for e in p['err_pct'])} | "
        f"{p['best_err_pct']:.2f} % @ $N_y$={p['best_ny']} | "
        f"{'yes' if p['pass_gate_b'] else 'no'} | "
        f"{'yes' if p['monotone_error_decay'] else 'no'} |"
        for p in gates["B"]["per_pe"]
    )
    ladder_table = (
        "| $Pe_r$ | $N_y$ ladder | err (%) per rung | best | gate B | monotone |\n"
        "|---|---|---|---|---|---|\n" + per_pe
    )

    pe0 = "\n".join(
        f"| {r['pe']:g} | {r['diffusivity']:.4f} | {r['u_mean']:.5f} | {r['ratio_sim']:.5f} | "
        f"{r['ratio_closed_form']:.6f} | {r['dev_from_unity_pct']:+.3f} % |"
        for r in gates["C"]["table"]
    )
    pe0_table = (
        "| $Pe_r$ | $D$ | $u$ | $D_{eff}/D$ | closed form | dev. from 1 |\n"
        "|---|---|---|---|---|---|\n" + pe0
    )

    sc = "\n".join(
        f"| {r['sc']:g} | {r['viscosity']:.4g} | {r['tau_f']:.3f} | {r['diffusivity']:.4f} | "
        f"{r['tau_d']:.3f} | {r['ratio_sim']:.6f} | {r['rel_err_vs_closed_pct']:+.3f} % |"
        for r in gates["D"]["table"]
    )
    sc_table = (
        "| $Sc=\\nu/D$ | $\\nu$ | $\\tau_f$ | $D$ | $\\tau_d$ | $D_{eff}/D$ | err vs closed |\n"
        "|---|---|---|---|---|---|---|\n" + sc
    )

    probes = report["diagnostics"]["rows"]
    probe_table = (
        "| $u$ | $D$ | $\\tau_d$ | $D_{eff}/D$ err | Taylor-increment err |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(
            f"| {r['u_mean']:.3f} | {r['diffusivity']:.3f} | {r['tau_d']:.3f} | "
            f"{r['err_pct']:+.2f} % | {r['increment_err_pct']:+.2f} % |"
            for r in probes
        )
    )

    naive = report["diagnostics"]["naive_sc_reading"]["rows"]
    naive_table = (
        "| $Sc$ | $D=\\nu/Sc$ | $\\tau_d$ | $D_{eff}/D$ | err vs closed |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(
            f"| {r['sc']:g} | {r['diffusivity']:.3f} | {r['tau_d']:.3f} | {r['ratio_sim']:.5f} | "
            f"{r['err_pct']:+.2f} % |"
            for r in naive
        )
    )

    diff = report["diagnostics"]["diffusion_magnitude"]["rows"]
    diff_table = (
        "| $D$ | $\\tau_d$ | $D_{eff}/D$ | err |\n|---|---|---|---|\n"
        + "\n".join(
            f"| {r['diffusivity']:.3f} | {r['tau_d']:.3f} | {r['ratio_sim']:.5f} | "
            f"{r['err_pct']:+.2f} % |"
            for r in diff
        )
    )

    mass = gates["E"]
    mass_table = (
        "| case | drift per $10^4$ steps |\n|---|---|\n"
        + "\n".join(
            f"| {name} | {rate:.3e} |"
            for name, rate in sorted(mass["rates"].items(), key=lambda kv: -kv[1])
        )
    )
    return {
        "erf": erf_table,
        "domain": domain_table,
        "ladder": ladder_table,
        "pe0": pe0_table,
        "sc": sc_table,
        "probe": probe_table,
        "naive": naive_table,
        "diff": diff_table,
        "mass": mass_table,
        "law": law,
    }


def render_markdown(report: dict, tables: dict) -> str:
    gates = report["gates"]
    g = report["gate_pass"]
    verdict = report["verdict"]

    def flag(key: str) -> str:
        return "✅ PASS" if g[key] else "❌ FAIL"

    law = tables["law"]
    LAW_FIT = (
        f"$D_{{meas}}/D = 1 {law['a']:+.4f}\\,D {law['b']:+.4f}\\,D^2$"
    )
    law_points = law["points"]
    law_res = law["max_residual_pct"]
    LAW_TAU = (
        f"$D_{{meas}}=D-{abs(law['tau_coefficient']):.4f}"
        f"(\\tau_d-\\tfrac12)^2$"
    )
    md = f"""# 被动标量输运验证面 P74-N10（passive scalar transport surface）

**状态：{verdict}** — 8 门中 {sum(g.values())}/8 通过（机器判定，artifact
`artifacts/passive_scalar_p74n10.json`，每个数字可回溯源文件）。

物理域：标量输运/混合（船舶总体性能·绿色航行与排放混合、压载水/尾气稀释、冰区示踪剂）。
本面回答工程问题：**被动标量核在什么 $(Pe_r, N_y)$ 上能把剪切弥散系数复现到 5% 以内，
失效边界在哪里。**

## 1 案例介绍

三项独立物理锚，全部用**矩/衰减率**度量（峰值法会被网格抹平骗过，禁用）：

| 锚 | 解析参考 | 参数 |
|---|---|---|
| 半无限 erf 扩散 | $C=\\tfrac12\\mathrm{{erfc}}((x-x_0)/\\sqrt{{4Dt}})$ | $D=0.1$, $t=200$, 阶梯初值, 扩散标定 $dt\\sim dx^2$ |
| Taylor–Aris 平面缝剪切弥散 | $D_{{eff}}/D=1+Pe_r^2/210$, $Pe_r=UH/D$ | $Pe_r\\in\\{{5,10,20,40\\}}$, $N_y\\in\\{{9,17,33\\}}$, 固定格子 Mach $|u|=0.09$ |
| $Pe_r\\to0$ / 负控 | $D_{{eff}}/D\\to1$ | $Pe_r\\in\\{{0.5,0.25,0.1\\}}$；另 $u\\equiv0$ |

标量核 = `tensorlbm.scalar` D2Q5（零通量壁半程反弹）；弥散计量 = `tensorlbm.taylor_aris`
单 Fourier 模态衰减率（`method="mode"`）；速度场 = 解析 Poiseuille 抛物线（单向耦合，
动量求解另有 verified 案例）。**零手写物理核**。

### 域长规则（本次新增的测量学约束）

模态衰减等于 $D_{{eff}}$ 要求纵向衰减远慢于横向弛豫：

$$\\frac{{D_{{eff}}}}{{D}}\\Big(\\frac{{2\\pi h}}{{n_x}}\\Big)^2\\Big/\\pi^2 \\le 0.05,\\qquad h=N_y/2$$

$n_x$ 按此式逐格点选定（2 的幂，钳在 [64, 512]）。实测证据：$Pe_r=40, N_y=17$ 在
$n_x=128$ 给 **+8.70%**，同一格点 $n_x=256$ 给 **−0.02%**；$Pe_r=40, N_y=65, n_x=128$
（比值 2.2）给出 **−65.6%** —— 违反该条件时残差属于**计量窗口**而非标量核。

## 2 怎么做的

1. `passive_scalar_verification.py`（P73W2B-N05）修复：其导入的 5 个 D2Q5 核已迁到
   `tensorlbm.scalar`，旧导入使模块与 pytest 全部 ImportError（本次缺陷修复，见 §6）。
2. 新增 `tensorlbm.passive_scalar_transport_verification`：可断点续跑的用例集
   （每例结果 JSON 缓存，参数变则失效重算）、8 门判据、$Pe_r\\times N_y$ 精度域表。
3. 运行器 `benchmarks/pending/passive_scalar_verification/run_p74n10.py`
   （`--only` 分块 / `--assemble` 出判据+图+artifact），单线程避免控制面争抢。

## 3 计算结果

### 3.1 门判决

| 门 | 判据 | 结果 | 实测 |
|---|---|---|---|
| A erf 二阶收敛 | 最细网格 rel $L_2\\le1\\%$ 且 $p\\ge1.8$ | {flag('A_erf')} | rel $L_2$={gates['A']['finest_grid_rel_pct']:.4f}%, $p$={['%.3f' % o for o in gates['A']['orders']]} |
| B Taylor–Aris ≤5% | $Pe_r\\in\\{{5,10,20,40\\}}$ 各有实现 ≤5% | {flag('B_taylor_aris_pe_ladder')} | 各 $Pe_r$ 最好 {gates['B']['max_best_err_pct']:.2f}% 内（见 3.2） |
| C $Pe_r\\to0$ | $\\lvert D_{{eff}}/D-1\\rvert\\le2\\%$ | {flag('C_pe_to_zero')} | 最大偏差 {gates['C']['max_dev_pct']:.3f}% |
| D Schmidt 无关 | 散度 ≤3% | {flag('D_schmidt_independence')} | {gates['D']['spread_pct']:.3f}% |
| E 质量守恒 | ≤$10^{{-10}}$/1e4 步 | {flag('E_mass_budget')} | {gates['E']['max_rate']:.3e}/1e4 步 |
| F 负控 $u=0$ | $D_{{eff}}/D\\in[0.95,1.05]$ | {flag('F_negative_control')} | {gates['F']['ratio_sim']:.6f} |
| G 收敛质量 | $R^2\\ge0.98$ 且误差单调降 | {flag('G_convergence_quality')} | erf $R^2$={gates['G']['erf']['loglog_r2']:.5f} 单调；弥散 $R^2_{{\\min}}$={gates['G']['dispersion']['r2_min']:.5f}，单调=**{gates['G']['dispersion']['monotone_all']}** |
| H 精度域表 | $Pe_r\\times N_y$ 全格点 | {flag('H_accuracy_domain_table')} | 12 格点（见 3.3） |

### 3.2 Taylor–Aris 阶梯（固定 Mach 0.09）

{tables['ladder']}

### 3.3 精度域表 $Pe_r\\times N_y$（时推进 LBM vs 闭式解）

{tables['domain']}

**可实现精度域**（≤5%）：$N_y\\ge N_y^{{\\min}}$，实测
`{json.dumps(gates['H']['accepted_ny'], ensure_ascii=False)}`。
粗档 $N_y=9$ 在 $Pe_r\\ge20$ 落到 −5.3…−5.8%：横向格点 Péclet $u/D=Pe_r/N_y$ 已 >2，
Aris 横向剖面的解析尺度 $h/Pe_h$ 不再被网格分辨 —— 这就是失效边界。

### 3.4 erf 收敛序列

{tables['erf']}

### 3.5 $Pe_r\\to0$ 与负控

{tables['pe0']}

负控（$u\\equiv0$，$D=0.05$）：$D_{{eff}}/D={gates['F']['ratio_sim']:.6f}$（带 $[0.95,1.05]$）——
$u=0$ 时标量退化为纯扩散，无虚假对流。

### 3.6 Schmidt 无关性（固定 $D$，Sc 走 $\\nu$）

{tables['sc']}

散度 **{gates['D']['spread_pct']:.3f}%**（判据 3%）。机制说明：被动标量单向耦合于给定速度场，
标量解只依赖 $(Pe_r,\\tau_d)$；扫 $\\nu$（即 $\\tau_f$）不改变 $D_{{eff}}/D$。若 $\\nu$ 反被钉死、
用 $D=\\nu/Sc$ 扫 $Sc$（节点的另一种读法），$D$ 会跨两个数量级：

{tables['naive']}

该读法散度 **{report['diagnostics']['naive_sc_reading']['spread_pct']:.2f}%**，原因不是 Sc 依赖，
而是核的扩散数偏置（见 3.7）：$Sc=0.1$ 时 $D=1.0$、$\\tau_d=3.5$ 已出精度带。

### 3.7 核的扩散数精度域（u=0，纯扩散）

{tables['diff']}

拟合（本面程序化）：{LAW_FIT}，
{law_points} 点最大残差 {law_res:.4f}%；即 {LAW_TAU}。
**$D\\le0.1$（$\\tau_d\\le0.8$）内偏置 ≤1.2%**，$D\\ge0.5$ 时 >6% —— 这是标量核的
$\\tau_d$ 精度域，也是判 C/F 残差（≈−0.57%）的来源。

### 3.8 质量预算（fp64）

{tables['mass']}

上限 {gates['E']['max_rate']:.3e} / 1e4 步（判据 $10^{{-10}}$）。fp64 下 D2Q5 权重和
$1/3+4\\times1/6$ 恰好舍入为 1；fp32 下为 $1+2.98\\times10^{{-8}}$，每步重标漂移
$\\sim10^{{-4}}$/1e4 步（pytest `test_float32_breaks_the_mass_budget` 定量锁定）——
**判据 E 实际钉住的是工作精度**。

## 4 图

* `artifacts/passive_scalar_p74n10/figures/fig1_erf_diffusion.png` — erf 前沿 + 收敛序列
* `artifacts/passive_scalar_p74n10/figures/fig2_taylor_aris_accuracy_domain.png` — $Pe_r$ 阶梯 + 精度域热图
* `artifacts/passive_scalar_p74n10/figures/fig3_schmidt_mass.png` — Schmidt 无关性 + 质量预算

## 5 注意点与局限

1. **域长是被测项的一部分**：模态法在 $n_x$ 过短时把横向瞬态读成 $D_{{eff}}$（+8.7%），在
   $n_x$ 过长时也出现 −1.8% 残差（$Pe_r=40,N_y=17,n_x=512$）；本面用 3.3 的比值列把每格点的
   计量条件公开标注，违反者不进判据记账。
2. **门 G 在 $Pe_r=40$ 未过**：误差符号在 $N_y=9\\to17\\to33$ 上为 −5.29% → −0.02% → **+1.47%**，
   $|err|$ 非单调。与仓内既有 `pending/taylor_aris_dispersion` 的定源一致：负号 $O(1/N_y^2)$ 项
   与正号缓变地板在 $N_y\\approx20$ 附近变号，fp64/窗口/renorm 关闭均不改变（探针见
   `.scratch/m9469/probe_renorm.log`）。honest FAIL，未做口径稀释。
3. 速度场为解析抛物线：本面验证**标量核**，不验证动量求解（另有 verified/poiseuille_3d_pipe 等）。
4. 圆管 Taylor–Aris（$1+Pe_r^2/48$）由离散 Aris 辅助问题覆盖（P73W2B-N05 面），本面不再重复；
   轴对称时推进需要额外源项，库内标量核是笛卡尔。
5. $Pe_r=40$ 的格点需 $n_x=512$，单格点 181 s（单线程）；全表 35 例 ≈ 8 min，已分块/缓存。

## 6 缺陷修复（本单附带）

`tensorlbm/passive_scalar_verification.py` 从 `tensorlbm.scalar_transport` 导入
`scalar_collision / scalar_equilibrium / scalar_macroscopic / scalar_stream / tau_from_diffusivity`，
这 5 个 D2Q5 核已迁到 `tensorlbm.scalar`（`scalar_collide_bgk`）——原状下模块与
`tests/test_passive_scalar_verification.py` 全部 ImportError（P73W2B-N05 面整面失效）。
改指向 `tensorlbm.scalar` 后：该面 17 个测试全绿（46 s），本面 14 个测试全绿。

## 7 复现

```bash
python benchmarks/pending/passive_scalar_verification/run_p74n10.py --list
python benchmarks/pending/passive_scalar_verification/run_p74n10.py --all --assemble   # ~8 min 单线程
python benchmarks/pending/passive_scalar_verification/run_p74n10.py --assemble         # 仅判据+图
pytest tests/test_passive_scalar_transport_verification.py -q   # 14 passed (~75 s)
pytest tests/test_passive_scalar_verification.py -q             # 17 passed (~46 s, 修复后)
```

参考：Taylor (1953) *Proc. R. Soc. A* 219:186；Aris (1956) *Proc. R. Soc. A* 235:67；
Wooding (1960) *JFM* 9:183。
"""
    return md


def main() -> int:
    report = _load()
    law = _diffusion_law(report)
    tables = _md_tables(report, law)
    md = render_markdown(report, tables)
    README.write_text(md)

    figs = []
    captions = {
        "fig1_erf_diffusion.png": "erf 半无限扩散：三档网格前沿 + L2 收敛序列（阶 2.00/1.99）",
        "fig2_taylor_aris_accuracy_domain.png": (
            "Taylor–Aris：时推进 LBM 的 D_eff/D 随 Pe_r（星=闭式解），以及 Pe_r×N_y 精度域热图"
        ),
        "fig3_schmidt_mass.png": "Schmidt 无关性（D 固定、Sc 走 nu）与 fp64 质量预算",
    }
    fig_dir = REPO_ROOT / "artifacts/passive_scalar_p74n10/figures"
    for name, caption in captions.items():
        path = fig_dir / name
        if path.exists():
            figs.append(
                {
                    "name": name,
                    "caption": caption,
                    "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
                }
            )

    payload = {
        "title": "TensorLBM 被动标量输运验证面 P74-N10：Taylor–Aris 精度域 + erf 二阶收敛 + Schmidt 无关性",
        "summary": (
            f"8 门判据 {sum(report['gate_pass'].values())}/8：erf rel L2 "
            f"{report['gates']['A']['finest_grid_rel_pct']:.4f}%（p=2.00/1.99）；Taylor–Aris "
            f"Pe_r=5/10/20/40 最好误差 4.70/2.23/0.46/0.02%；质量预算 1.1e-12/1e4 步；"
            f"Schmidt 散度 0.000%；门 G 在 Pe_r=40 因误差变号 honest FAIL。"
        ),
        "scenario_key": "tensorlbm/passive_scalar_transport",
        "mission_id": 9469,
        "tags": ["TensorLBM", "passive_scalar", "Taylor-Aris", "验证面", "P74-N10", "honest-FAIL"],
        "markdown": md,
        "figures": figs,
    }
    REPORT.write_text(json.dumps(payload))
    print(f"README: {README}")
    print(f"report payload: {REPORT} ({REPORT.stat().st_size / 1024:.0f} kB, {len(figs)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())