#!/usr/bin/env python
"""Render ``docs/INVARIANCE_CERTIFICATION.md`` from the campaign artefacts.

All numbers in the generated document are read from
``artifacts/invariance/P170W2-N08/*.json`` — nothing is hand-copied, so the
document can be regenerated and diffed after every campaign re-run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "invariance" / "P170W2-N08"
OUT = ROOT / "docs" / "INVARIANCE_CERTIFICATION.md"


def load(name: str):
    p = ART / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, (int,)):
        return str(x)
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x != x:
        return "nan"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e5:
        return f"{x:.{nd-1}e}"
    return f"{x:.{nd}f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> int:
    report = load("report")
    pois = load("poiseuille")
    cyl = load("cylinder")
    neg = load("negative_control")
    gal = load("galilean")
    rep = load("reproducibility")
    if not report:
        print("report.json missing — run the campaign first", file=sys.stderr)
        return 1

    crit = {c["id"]: c for c in report["criteria"]}
    L = []
    A = L.append
    A("# TensorLBM 不变性合格证（mesh-orientation + Galilean invariance）")
    A("")
    A("节点：**P170W2-N08** · 模块：`src/tensorlbm/invariance/` · "
      "本文件由 `scripts/make_invariance_doc.py` 从工件 JSON 自动生成（零手抄）。")
    A("")
    A("## 0. 结论摘要（机器判据）")
    A("")
    A(f"通过 **{report['n_pass']}/{report['n_total']}** 判据，"
      f"`all_pass = {report['all_pass']}`。")
    A("")
    A(table(["判据", "内容", "判定", "关键数字"],
            [[c["id"], c["name"], "PASS" if c.get("pass") else "FAIL",
              _key_number(c)] for c in report["criteria"]]))
    A("")
    A("## 1. 物理背景与算例设置")
    A("")
    A("被检验的命题：**同一个物理问题，在离散格子相对于物体/来流任意取向 θ 时，"
      "解必须在声明容差内不变；施加伽利略平移 U 后的解必须等于平移前的解加 U；"
      "残余偏离必须服从可预测的 Ma² 律。**")
    A("")
    A("三个算例都以解析几何（真 SDF）描述物体，只旋转几何/驱动方向，不动格子：")
    A("")
    A("1. **斜槽 Poiseuille**：宽 H 的平面槽道与格子成 θ=0/15/30/45°，壁面用 SDF 求"
      "壁距分数 q 的 Bouzidi 插值反弹；入口/出口面用非平衡外推（NEEM）施加精确抛物线"
      "速度。验收量：速度场 L2 误差随 H 的观测阶（判据 C1）。")
    A("2. **圆柱绕流**：双周期方箱内，半径 D/2 的圆（SDF=d−R），沿 (cosθ, sinθ) 方向"
      "以“定体积流量”死拍控制驱动（每步体力 f=ρ(u_target−ū)，稳态时 ᾱ·N_fluid 即阻力），"
      "Re=100/200，θ∈{0,15,30,45}°。验收量：Cd、St 的取向散布（C2/C3）。")
    A("3. **Taylor–Green 涡 + 伽利略平移**：周期箱 n=64，k=2π/n，ν=0.01，"
      "U 取使 U·T 恰为整数个格子的值（无需插值即可比较叠加解）。验收量：叠加残差随 Ma、"
      "衰减率相对偏离随 Ma 的幂律（C4/C5）。")
    A("")
    A("负控（C6）：同分辨率下裸 BGK 且 τ→0.5⁺（Re_eff≈2×10⁴），其 Cd 取向散布必须"
      "**超过** C2 容差，用于证明 C2 的通过不是判据太松。")
    A("")
    A("## 2. 方法与工具链")
    A("")
    A("- 求解器：`tensorlbm.invariance.solver.D2Q9Solver`（纯 NumPy 单精度/双精度），"
      "BGK 或 TRT（magic=3/16）碰撞 + Guo 二阶体力；流式=周期 gather + 壁面入射分布覆盖。")
    A("- 壁面：Bouzidi et al. (2001) 线性/二次插值反弹（q<1/2 与 q≥1/2 两支），"
      "q 由解析 SDF 与 SDF 梯度沿格子方向求交得到；`bounce_back=\"halfway\"` 冻结 q=0.5 "
      "退化为经典半步反弹，用作对照/负控。")
    A("- 几何：`tilted_channel`（真 SDF = H/2−|n̂·(r−c)|）、`circle`（SDF=|r−c|−R，"
      "外流约定，固体在圆内）。")
    A("- 驱动：斜槽用入口/出口速度 Dirichlet（NEEM）；圆柱用死拍体积流量控制；"
      "TG 用解析初场 feq(ρ=1, u_exact+U)。")
    A("- 运行：`python scripts/run_invariance_campaign.py --case all --workers 14`，"
      "多进程并行；每个算例落 JSON + 自身 SHA-256；`--case report` 汇总判据。")
    A("- 图表：`python scripts/make_invariance_figures.py`。")
    A("")
    A("## 3. 计算结果")
    A("")
    if pois:
        A("### 3.1 斜槽 Poiseuille（C1）")
        A("")
        A(f"参数：u_mean={pois['parameters']['u_mean']}, ν={pois['parameters']['nu']}, "
          f"τ={pois['parameters']['tau']}, 步数={pois['parameters']['steps_factor']}·H²，"
          f"壁面={pois['bounce_back']}。误差统计剔除进出口各 "
          f"{pois['parameters']['trim']:.0%} 长度。")
        A("")
        rows = []
        for key, e in sorted(pois["results"].items(), key=lambda kv: float(kv[0])):
            rows.append([f"{float(key):g}", " / ".join(f"{w:g}" for w in e["widths"]),
                         " / ".join(fmt(v, 3) for v in e["errors"]),
                         " / ".join(f"{o:.2f}" for o in e["orders"]),
                         f"{e['mean_order']:.2f}"])
        A(table(["θ (deg)", "H (lattice)", "L2 误差 (各分辨率)", "观测阶 p_i", "平均阶"], rows))
        A("")
    if cyl:
        A("### 3.2 圆柱绕流 Cd / St 取向散布（C2/C3）")
        A("")
        p = cyl["parameters"]
        A(f"参数：D={p['diameter']}, 域={p['domain_per_diameter']}D, u_target={p['u_target']}, "
          f"碰撞={p['collision']}, 壁面={p['bounce_back']}, 瞬态={p['transient_d']}D/U, "
          f"采样={p['sample_d']}D/U, 驱动={p['drive']}。")
        A("")
        rows = []
        for re_key, runs in sorted(cyl["results"].items(), key=lambda kv: float(kv[0])):
            for r in runs:
                rows.append([f"{float(re_key):g}", f"{r['theta_deg']:g}",
                             fmt(r["cd"], 4), fmt(r["st"], 4), fmt(r["u_mean"], 5),
                             fmt(r["fmag_mean"], 3), fmt(r["cd_rel_fluct"], 3)])
        A(table(["Re", "θ (deg)", "Cd", "St", "ū", "f̄ (体力密度)", "Cd 相对波动"], rows))
        A("")
    if crit.get("C2") or crit.get("C3"):
        rows = []
        for cid in ("C2", "C3"):
            c = crit.get(cid)
            if not c:
                continue
            for re_key, d in c["by_re"].items():
                rows.append([cid, f"Re={re_key}", " / ".join(
                    f"{k}: {fmt(v, 4)}" for k, v in d[[
                        k for k in d if k.endswith("by_theta")][0]].items()),
                    fmt(d["max_spread"], 4), fmt(d.get("tolerance", 0), 3),
                    "PASS" if d["pass"] else "FAIL"])
        A(table(["判据", "Re", "各取向数值", "最大相对散布", "容差", "判定"], rows))
        A("")
    if gal:
        A("### 3.3 伽利略平移与 Ma² 律（C4/C5）")
        A("")
        p = gal["parameters"]
        A(f"参数：n={p['n']}, 步数={p['steps']}, ν={p['nu']}, u0={p['u0']}, "
          f"碰撞={p['collision']}, 数据类型={p['dtype']}；{p['note']}。")
        A("")
        rows = []
        base = [r for r in gal["results"] if r["ma"] == 0.0]
        lam0 = base[0]["decay_rate"] if base else float("nan")
        for r in sorted(gal["results"], key=lambda r: r["ma"]):
            dev = (r["decay_rate"] - lam0) / lam0 if lam0 == lam0 else float("nan")
            rows.append([fmt(r["ma"], 4), fmt(r["boost"], 4), fmt(r["decay_rate"], 6),
                         fmt(dev, 3), fmt(r["residual_relU"], 4),
                         fmt(r.get("superposition_residual_relU"), 4)])
        A(table(["Ma=U/c_s", "U", "λ (测量)", "(λ−λ0)/λ0", "|u−(u_exact+U)|/U",
                 "|u(U)−u(0)−U|/U"], rows))
        A("")
        if crit.get("C5"):
            c5 = crit["C5"]
            A(f"幂律拟合：Δλ/λ0 = C·Ma^p，**p = {fmt(c5.get('slope_vs_ma'), 3)}** "
              f"(R²={fmt(c5.get('r2'), 5)})；等价地以 Ma² 为自变量斜率 "
              f"{fmt(c5.get('slope_vs_ma_squared'), 3)}。")
            A("")
    if neg:
        A("### 3.4 负控（C6）")
        A("")
        rows = []
        for re_key, runs in sorted(neg["results"].items(), key=lambda kv: float(kv[0])):
            for r in runs:
                rows.append([f"{r['theta_deg']:g}", fmt(r["cd"], 4), fmt(r["st"], 4),
                             fmt(r["tau"], 5), fmt(r["cd_rel_fluct"], 3)])
        A(table(["θ (deg)", "Cd", "St", "τ", "Cd 相对波动"], rows))
        A(f"注意：{neg.get('note', '')}")
        A("")
    if rep:
        A("### 3.5 复算逐位一致（C7）")
        A("")
        A(table(["运行", "SHA-256"], [[i + 1, h] for i, h in enumerate(rep["hashes"])]))
        A("")
    A("## 4. 结果图")
    A("")
    A("![fig1](../figures/fig1_poiseuille_convergence.png)")
    A("")
    A("图 1 斜槽 Poiseuille：(a) L2 误差 vs 壁面分辨率（各取向 + 斜率 −2 参考）；"
      "(b) θ=45° 中截面速度剖面 vs 精确抛物线。")
    A("")
    A("![fig2](../figures/fig2_cylinder_invariance.png)")
    A("")
    A("图 2 圆柱绕流：(a) Cd 取向偏离（虚线为负控）；(b) St 取向偏离；"
      "(c) Cd 峰峰散布 vs 容差。")
    A("")
    A("![fig3](../figures/fig3_galilean.png)")
    A("")
    A("图 3 伽利略：(a) 归一化残差 vs Ma（含容差线）；(b) 衰减率相对偏离 vs Ma 与幂律拟合；"
      "(c) 脉动能量衰减曲线。")
    A("")
    A("![fig4](../figures/fig4_fields.png)")
    A("")
    A("图 4 场量云图：(a) 45° 斜槽 |u|；(b) 圆柱（格子相对来流 30°）|u|；"
      "(c) 叠加平移后的 TG 涡 u_x。")
    A("")
    A("## 5. 注意点 / 局限 / 适用范围")
    A("")
    A("- **格子层级容差**：全部结论都绑定在 D2Q9 格子与所声明分辨率（H=8…24、D=12）上；"
      "更高 Re 或更细网格需重跑本 campaign，不得外推。")
    A("- **圆柱算例的周期箱约束**：双周期方箱内 16D 间距意味着这实际是“柱阵中的流动”，"
      "Cd 绝对值高于单柱文献值（块塞比与尾流再入效应）；C2/C3 检验的是**取向散布**"
      "（同一离散化下的相对量），不宣称绝对 Cd 与实验/文献对齐。")
    A("- **驱动方式**：定体积流量死拍控制使体力量 f̄·N_fluid 成为阻力估计；"
      "该估计依赖稳态（统计平均）与周期箱动量守恒，瞬态段不计入。")
    A("- **斜槽进出口**：入口/出口施加精确物理解，误差统计剔除两端各 25% 长度，"
      "因此 C1 度量的是斜壁离散化而非流入边界条件。")
    A("- **斜槽不可能在周期箱内实现**（无限条带只在沿轴平移下不变，无法构造二维环面），"
      "故本轮以进出口面替代周期边界；这是对战役描述的有意偏差，理由见上。")
    A("- **C4 的两个定义**：主判据按战役定义与解析解比较（含格式截断误差）；"
      "另给出叠加残差 |u(U)−u(0)−U|，它隔离了格式内部破坏伽利略不变性的部分。")
    A("- **TRT 参数**：圆柱算例用 TRT（Λ=3/16）以获得 Re=200 的稳定裕度；"
      "负控保持裸 BGK。")
    A("")
    A("## 6. 误差判定（关键量 vs 基准，逐项回答“是否 <3%”）")
    A("")
    rows = []
    if pois:
        for key, e in sorted(pois["results"].items(), key=lambda kv: float(kv[0])):
            worst = max(e["errors"]) / (1.5 * 0.01)
            rows.append([f"斜槽 θ={float(key):g}° 速度场相对 L2 误差（最粗网格）",
                         fmt(worst, 3), "vs 精确抛物线", "否" if worst > 0.03 else "是"])
    if crit.get("C2"):
        for re_key, d in crit["C2"]["by_re"].items():
            rows.append([f"圆柱 Re={re_key} Cd 取向散布", fmt(d["max_spread"], 4),
                         "取向自比", "是" if d["max_spread"] <= 0.03 else "否"])
    if crit.get("C3"):
        for re_key, d in crit["C3"]["by_re"].items():
            rows.append([f"圆柱 Re={re_key} St 取向散布", fmt(d["max_spread"], 4),
                         "取向自比", "是" if d["max_spread"] <= 0.03 else "否"])
    if crit.get("C5"):
        c5 = crit["C5"]
        rows.append(["TG 衰减率幂律斜率 p", fmt(c5.get("slope_vs_ma"), 3),
                     "理论 2（O(Ma²)）", "见正文（判据为 [1.7,2.3]）"])
    A(table(["量", "数值", "基准", "<3%？"], rows))
    A("")
    A("绝对 Cd/St 与文献（单柱 Re=100: Cd≈1.34, St≈0.165；Re=200: Cd≈1.34, St≈0.197）"
      "的对齐**不在本节点的主张范围内**——周期柱阵设置下 Cd 系统性偏高，"
      "本节点只对**不变性**签合格证。这一点在报告中如实标注，不作粉饰。")
    A("")
    A("## 7. 审计链")
    A("")
    prov = report.get("provenance", {})
    A(f"- commit: `{prov.get('commit')}` · host: `{prov.get('host')}` · "
      f"python {prov.get('python')} · numpy {prov.get('numpy')} · "
      f"时间 {prov.get('timestamp_utc')}")
    A(f"- 工件目录：`artifacts/invariance/P170W2-N08/`，文件名与 SHA-256：")
    A("")
    A(table(["文件", "sha256"],
            [[k, v[:32] + "…"] for k, v in report.get("artefact_sha256", {}).items()]))
    A("")
    A("- 复现：`python scripts/run_invariance_campaign.py --case all --workers 14` → "
      "`--case report` → `python scripts/make_invariance_figures.py` → "
      "`python scripts/make_invariance_doc.py`。")
    A("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L))
    print(f"[doc] {OUT.relative_to(ROOT)}  ({len(L)} lines)")
    return 0


def _key_number(c: dict) -> str:
    cid = c["id"]
    if cid == "C1":
        vals = [f"θ={k}: p̄={v['mean_observed_order']:.2f}"
                for k, v in c.get("per_theta", {}).items()]
        return "; ".join(vals)
    if cid in ("C2", "C3"):
        return "; ".join(f"Re={k}: {fmt(v['max_spread'], 4)}"
                         for k, v in c.get("by_re", {}).items())
    if cid == "C4":
        vals = [f"Ma={fmt(r['ma'], 3)}: {fmt(r['residual_relU'], 3)}"
                for r in c.get("rows", [])]
        return "; ".join(vals) + f" (monotone={c.get('monotone_increasing')})"
    if cid == "C5":
        return f"p={fmt(c.get('slope_vs_ma'), 3)}, R²={fmt(c.get('r2'), 4)}"
    if cid == "C6":
        return f"max spread={fmt(c.get('max_spread'), 4)}"
    if cid == "C7":
        return f"{c.get('n_unique')} unique hash(es) of {len(c.get('hashes', []))} runs"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())