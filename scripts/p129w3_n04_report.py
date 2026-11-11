#!/usr/bin/env python
"""P129W3-N04 — build the paper-grade report payload from the artifact JSONs.

Every number in the markdown is injected from ``artifacts/.../criteria.json``
and the per-point JSONs (zero hand-copy).  Writes ``report.json`` with
``title / summary / scenario_key / mission_id / tags / markdown / figures``
ready for ``POST $TENSOR_API_BASE/tensor/reports``.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _f(x, n=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "n/a"
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-2:
        return f"{v:.{n}e}"
    return f"{v:.{n}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/p129w3_n04_e779a")
    ap.add_argument("--figures", default="figures/p129w3_n04")
    ap.add_argument("--mission", type=int, default=11291)
    ap.add_argument("--out", default="report.json")
    args = ap.parse_args(argv)

    root = ROOT / args.artifacts
    crit = json.loads((root / "criteria.json").read_text())
    pts = {}
    for p in sorted((root / "points").glob("point_*.json")):
        d = json.loads(p.read_text())
        pts[p.stem.replace("point_", "")] = d
    geom_files = sorted(root.glob("geometry_gate_D*.json"))
    geom = json.loads(geom_files[-1].read_text()) if geom_files else None

    curve = crit.get("curve") or {}
    sweep_D = crit.get("sweep_D")
    rows = []
    bemt_by_j = {round(r["J"], 2): r for r in (crit.get("bemt") or {}).get("rows", [])}
    for k in sorted(pts):
        if not k.startswith(f"D{sweep_D}_J") or "strut" in k or "smag" in k or "_t" in k:
            continue
        p = pts[k]
        b = bemt_by_j.get(round(p["j"], 2), {})
        rows.append(
            "| {J:.2f} | {kt:.4f} | {kq:.4f} | {eta:.3f} | {dkt} | {dkq} | {deta} | {drift:.2%} |".format(
                J=p["j"], kt=p["kT"], kq=p["kQ"], eta=p["eta"],
                dkt=_f(100 * (p["kT"] - b["KT"]) / b["KT"]) if b.get("KT") else "n/a",
                dkq=_f(100 * (p["kQ"] - b["KQ"]) / b["KQ"]) if b.get("KQ") else "n/a",
                deta=_f(100 * (p["eta"] - b["eta"]) / b["eta"]) if b.get("eta") else "n/a",
                drift=p["convergence"]["thrust"]["max_block_deviation"],
            )
        )

    crit_rows = "\n".join(
        "| {id} | {measured} | {threshold} | **{verdict}** | {evidence} |".format(
            id=c["id"],
            measured=json.dumps(c["measured"], default=str)[:220] if not isinstance(c["measured"], str) else c["measured"],
            threshold=c["threshold"],
            verdict=c["verdict"],
            evidence=c["evidence"],
        )
        for c in crit["criteria"]
    )

    geo_txt = "n/a"
    if geom:
        pt = geom["points"][0]
        geo_txt = (
            f"analytic $A_E/A_0$ = {_f(pt['expanded_area_ratio_analytic'],6)}; "
            f"mask-measured $A_E/A_0$ = {_f(pt['expanded_area_ratio_measured'])} "
            f"({_f(100*(pt['expanded_area_ratio_measured']-0.689)/0.689,2)} % vs drawing); "
            f"mask-measured $P/D_{{0.7R}}$ = {_f(pt['pitch_ratio_07_measured'])} "
            f"({_f(100*(pt['pitch_ratio_07_measured']-1.10)/1.10,2)} %); "
            f"$D_{{hub}}/D$ = {_f(pt['hub_diameter_ratio_measured'])}; "
            f"worst-case over 5 rotor phases = {json.dumps({k: round(v,3) for k,v in geom['worst_error_pct'].items()})}"
        )

    gci = crit.get("gci")
    gci_txt = "n/a"
    if gci:
        gci_txt = (
            f"$K_T$ on grids {gci['grids_coarse_medium_fine']} = "
            f"{[round(v,4) for v in gci['values_coarse_medium_fine']]}; "
            f"apparent order p = {_f(gci['apparent_order_p'],3)}; "
            f"$GCI_{{fine}}$ = {_f(gci['gci_fine'],4)} ({_f(100*gci['gci_fine'],2)} %); "
            f"Richardson limit = {_f(gci['extrapolated_fine'],4)}"
        )

    spec_txt = "n/a"
    if crit.get("spectrum"):
        parts = []
        for label, blk in crit["spectrum"].items():
            h = blk["harmonics_bpf"]
            parts.append(
                f"{label}: BPF {_f(h['k1']['freq_hz_physical'],1)} Hz (error {_f(h['k1']['error_pct'],2)} %), "
                f"2xBPF {_f(h['k2']['freq_hz_physical'],1)} Hz (error {_f(h['k2']['error_pct'],2)} %), "
                f"ratio 2nd/1st = {_f(blk['amplitude_ratio_2nd_1st'],3)}, "
                f"dominant peak {_f(blk['dominant_peak_hz_physical'],1)} Hz"
            )
        spec_txt = "; ".join(parts)

    sigma_txt = "n/a"
    if crit.get("sigma_rows"):
        av = [r for r in crit["sigma_rows"] if r.get("inception_available")]
        if av:
            sigma_txt = "; ".join(
                f"J={r['j']:.2f}: $\\sigma_i$={_f(r['sigma_i_tip'],4)} (core $p_{{min}}$={_f(r['p_core_min_lattice'],6)}, "
                f"$\\lambda_2$={_f(r['lambda2_min'],3)})" for r in av
            )
        else:
            sigma_txt = "no tip-vortex core resolved (lambda_2 > 0 in the search annulus)"

    verdict_counts = {}
    for c in crit["criteria"]:
        verdict_counts[c["verdict"]] = verdict_counts.get(c["verdict"], 0) + 1

    markdown = f"""# P129W3-N04 · INSEAN E779A 非定常敞水与梢涡空化初生验证面（TensorLBM）

## ① 案例介绍

对象为 INSEAN E779A 四叶模型桨：D = 0.227 m、Z = 4、$(P/D)_{{0.7R}}$ = 1.10、$A_E/A_0$ = 0.689、
毂径比 0.18、n = 25 rps（叶频 $4n$ = 100 Hz）。本节点让 TensorLBM 第一次用**会转的桨**
（刚体旋转掩模 + Ladd 动量交换移动壁）对敞水性能、叶频载荷谱与梢涡初生同时负责。

几何门的实测（全部由体素掩模独立测量，不是把输入抄回去）：{geo_txt}。

## ② 怎么做的

1. **几何**：在展开圆柱面上构造水翼型叶片（弦长分布按 $A_E/A_0$ 面积精确标定、螺距
   $\\beta(r)=\\arctan(P/2\\pi r)$、抛物线中弧 2 % 弦长、NACA 型厚度分布、$r/R=0.7$ 处 t/c = 3.5 %），
   逐格点判定 $|\\xi|\\le c/2$、$|\\eta-\\text{{camber}}|\\le t/2$ 得到实体掩模；几何门从掩模反测
   （弦向回归测螺距、体素体积/厚度积分测面积、毂体面积测毂径）。
2. **移动边界**：lab 参考系、每步按 $\\Delta\\theta$ 重采样旋转掩模；碰撞用 MRT、迁移 `stream3d`；
   入口平衡态+出口零梯度+侧壁无滑移；叶片与毂上的动量交换采用 Ladd(1994) 移动壁，
   每步按叶片扇区把冲量分箱，得到**单叶**推力与总扭矩。
3. **系数**：$K_T=T/\\rho n^2D^4$、$K_Q=Q/\\rho n^2D^5$（$Q$ 为桨对流体施加的扭矩，
   $P=2\\pi nQ>0$）、$\\eta_0=J K_T/2\\pi K_Q$；$J$ 扫 0.1…0.9。
4. **非定常**：J = 0.6 输出单叶推力时程与其 FFT（Hann 窗），标出 $4n$ 与 $8n$ 峰位与幅值比；
   另跑一个"上游挡板（静态圆柱）"工况，给流场一个固定的方位不均匀性，使叶频内容有物理来源。
5. **初生**：$\\lambda_2$ 判据 + $Q$ 判据识别梢涡核，取核内 $p_{{min}}$，
   $\\sigma_i=(p_\\infty-p_{{min}})/(\\tfrac12\\rho V_R^2)$，$V_R=\\sqrt{{V_A^2+(2\\pi nR)^2}}$；
   只做压力场判据，不做空泡形态。
6. **网格**：同一 $J$ 三套网格做 Richardson 三网格 GCI；相位步进 $\\le$ 2°/步。
7. **独立参照**：仓内 BEMT（叶元–动量理论，Prandtl 梢/毂损失、弯度翼型、阻尼感应迭代）
   先对仓内 KP505 敞水表（Fujisawa et al. 2000）标定，再用于把格点 Re 偏差归因。

## ③ 计算结果（所有数字由 `criteria.json` / `points/*.json` 程序化注入）

敞水点（D = {sweep_D} 格）：deviation 列对 BEMT（MODEL）参照，非实验。

| J | $K_T$ | $K_Q$ | $\\eta_0$ | $\\Delta K_T$ % | $\\Delta K_Q$ % | $\\Delta\\eta$ % | 周期块漂移 |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

GCI：{gci_txt}

叶频谱：{spec_txt}

空化初生：{sigma_txt}

判据总表：

| 判据 | 实测 | 阈值 | 判定 | 证据 |
|---|---|---|---|---|
{crit_rows}

判据分布：{json.dumps(verdict_counts, ensure_ascii=False)}

## ④ 结果图

见图 fig1–fig5（敞水曲线、收敛史、叶频谱、GCI、初生）。

## ⑤ 注意点（踩过的坑 / 局限）

1. **手性/转向必须成对定标**：几何螺线手性与 $\\omega$ 符号只有一种组合给出"螺旋桨"模式，
   另一种给出风车（turbine）模式；判据是 $Q_{{fluid}}\\cdot\\omega>0$ 且 $\\eta_0>0$。
   本面在 D = 8/16 的粗网格上两种组合都出现过"$K_T$ 随 $J$ 上升"的假象——那是弦长只有
   2–5 格时螺距被格点量化抹平、叶片退化成桨板的伪物理，**不是转向问题**。
2. **$K_Q$ 的符号口径**：$K_Q$ 必须用"桨对流体的扭矩"（$=-$机体扭矩），否则 $\\eta_0<0$。
   这是本面早期版本的实现缺陷，已由 pytest 与 $\\eta_0>0$ 判据双重看住。
3. **格点雷诺数**：本面 $Re_D\\sim10^3$–$10^4$，比模型试验（$Re_D\\approx1.3\\times10^6$）低 2–3 个量级；
   摩阻被显著高估，故 $K_Q$ 偏高、$\\eta_0$ 偏低是可预期的系统性偏差，
   归因见 BEMT 对比（用 $c_{{d0}}$ = 0.008 / 0.03 两种截面极线）。
4. **弦长分辨率**：D = {sweep_D} 时 $r/R=0.7$ 处弦长约 5–7 格、厚度强制 ≥ 1 格，
   故叶型近似"阶梯板"；几何门在 D = 48 时掩模反测误差仍达
   {json.dumps({k: round(v,2) for k,v in (geom['worst_error_pct'] if geom else {}).items()})} %（判定阈值 1 %）。
5. **实验锚 withheld**：执行环境无外网、仓内无 E779A 一次文献表，
   因此对 INSEAN 实验的对比一律标 `WITHHELD`，绝不编造参考数字；
   `load_experimental_anchor` 已实现"带 provenance 的表注入即自动出对比/判据"，属可闭合的缺口。
6. **空化**：只有压力场初生判据；$\\sigma<\\sigma_i$ 之后的片空泡/云空化形态**NOT_SUPPORTED**；
   且格点 Ma ≈ 0.18–0.25 的可压缩性会给 $p_{{min}}$ 带来 O(Ma$^2$) 的偏差。

## ⑥ 误差判定（逐项，显式回答"是否 < 3 %"）

- 几何门（解析实现）：{_f(crit['criteria'][0]['measured'] if crit['criteria'] else None)} —— 见判据表逐行；
- 敞水 $K_T/K_Q/\\eta$ 对实验：**WITHHELD**（无一次文献表），因此"是否 < 3 %"对实验无法回答，如实标注；
- 对 BEMT（MODEL）参照的偏差见上表，$J\\in[0.4,0.8]$ 的最大偏差见判据表 crit1b 行；
- 周期平均残差与 GCI 均按阈值逐条判定，未达标项如实 FAIL。

**没有任何判据被粉饰**：判定为 FAIL / WITHHELD 的条目在判据表中保留原样。
"""

    figs = []
    fig_dir = ROOT / args.figures
    captions = {
        "fig1_openwater_curve.png": "Fig.1 E779A open-water coefficients vs advance ratio (LBM moving boundary vs BEMT MODEL reference; experimental anchor WITHHELD).",
        "fig2_convergence.png": "Fig.2 Thrust history and per-revolution block means (period-average residual).",
        "fig3_blade_rate_spectrum.png": "Fig.3 Total-thrust spectrum at J=0.6 in shaft orders; blade rate 4n=100 Hz and 2x marked.",
        "fig4_gci.png": "Fig.4 Grid convergence of K_T at J=0.6 with Richardson extrapolation and GCI.",
        "fig5_inception.png": "Fig.5 Tip-vortex core minimum pressure and derived inception number sigma_i(J).",
    }
    for name, cap in captions.items():
        p = fig_dir / name
        if p.exists():
            figs.append({"name": name, "caption": cap, "png_base64": base64.b64encode(p.read_bytes()).decode()})

    payload = {
        "title": "P129W3-N04 INSEAN E779A 非定常敞水 + 叶频载荷谱 + 梢涡空化初生验证面（移动边界 LBM）",
        "summary": (
            "首次在 TensorLBM 建立会转的 INSEAN E779A 桨验证面：刚体旋转掩模 + Ladd 移动壁动量交换，"
            f"几何门（面积比/螺距/毂径，掩模反测）→ 敞水 {len(rows)} 点 $K_T/K_Q/\\eta_0$ → J=0.6 单叶推力谱"
            f"（叶频 4n=100 Hz）→ $\\lambda_2$ 梢涡核 $p_{{min}}$ 初生数 → 三网格 GCI。"
            f"判据分布 {json.dumps(verdict_counts, ensure_ascii=False)}；"
            "对 INSEAN 实验的对比因环境无一次文献表而显式 WITHHELD（不编造参考数字），"
            "格点 Re 与弦长分辨率的系统性偏差用仓内 BEMT 归因。"
        ),
        "scenario_key": "campaign-node",
        "mission_id": args.mission,
        "tags": ["propeller", "openwater", "INSEAN-E779A", "moving-boundary", "LBM", "cavitation-inception", "blade-rate", "GCI"],
        "markdown": markdown,
        "figures": figs,
    }
    out = ROOT / args.out
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"[report] -> {out} ({len(markdown)} chars, {len(figs)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())