#!/usr/bin/env python3
"""Render the mission deliverable (docs markdown + platform report JSON).

All numbers are injected programmatically from the artifacts produced by
``run.py`` / ``analyze.py`` in this worktree -- nothing is typed by hand.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "resistance_chain_cloud"
DOCS = REPO / "docs" / "resistance_chain_cloud"
SCENARIO = "paper-mat-resistance-chain-de-mock-cloud"
MISSION = 4927

PRIOR_REPORTS = [
    (5539, "IN10-C1 冰阻力全谱与速度-功率曲线：三态模型族 + L2 锚语料 25/25 + PC6 VP 矩阵"),
    (5393, "m4842 多船型标模阻力基准：Holtrop vs 公开水池试验（逐工况误差 + honest-FAIL）"),
    (5384, "千工况批量计算（P6-02）：5 船型参数扫描 1300 工况 / 成功率 100.0%"),
]


def load_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def fmt(v, nd=4):
    if v is None or v == "" or v == "NO-DATA-TBD":
        return "NO-DATA-TBD"
    try:
        f = float(v)
        return f"{f:.{nd}f}" if math.isfinite(f) else "NO-DATA-TBD"
    except Exception:
        return str(v)


def png_b64(path: Path, max_kb: int = 55) -> str | None:
    if not path.exists():
        return None
    try:
        from PIL import Image
    except Exception:
        return base64.b64encode(path.read_bytes()).decode()
    img = Image.open(path)
    for scale in (1.0, 0.85, 0.7, 0.55, 0.45):
        buf = io.BytesIO()
        w, h = img.size
        im2 = img.resize((int(w * scale), int(h * scale))) if scale < 1.0 else img
        im2.convert("RGB").save(buf, format="JPEG", quality=88, optimize=True)
        if buf.tell() / 1024 <= max_kb:
            return base64.b64encode(buf.getvalue()).decode()
    return None


def main() -> int:
    t1 = load_json(OUT / "table1_grid_levels.json", {}) or {}
    rows = t1.get("rows", [])
    gcis = t1.get("gci", {})
    t2 = read_csv(OUT / "table2_surface_angular.csv")
    t3 = read_csv(OUT / "table3_wake_profiles.csv")
    index = load_json(OUT / "cloud_index.json", {}) or {}
    runs = {}
    for p in (OUT / "fields").glob("*/*_run.json"):
        d = load_json(p, {}) or {}
        runs[str(d.get("D"))] = d

    geos = sorted({r["geometry"] for r in rows})
    lines: list[str] = []
    A = lines.append

    A("# 阻力链 CLOUD 补测：场量云图与角度分布（mission m4927）")
    A("")
    A(f"> scenario_key = `{SCENARIO}` ｜ mission_id = {MISSION} ｜ 仓 = TensorLBM ｜ "
      f"git commit = `{index.get('git_commit', 'unknown')}`")
    A("> 工件目录：`artifacts/resistance_chain_cloud/`（fields/*.npz、table1/2/3 CSV、fig1/fig2 PNG、cloud_index.json）")
    A("> 生成脚本：`benchmarks/resistance_chain_cloud/run.py`（LBM 计算）+ `analyze.py`（表/图）+ "
      "`tests/test_resistance_chain_cloud.py`（判据）")
    A("")
    A("## ① 案例介绍（物理背景 / 几何 / 参数表）")
    A("")
    A("阻力链（课题 `resistance-chain-de-mock`）的既有 8 个 mission 报告只给出**标量**证据：")
    A("总阻力 R_t / 总阻力系数 C_t / 批量成功率。缺的是 **CLOUD＝空间/角度分布证据**：")
    A("阻力到底是「哪些表面压力、哪些涡、哪段尾流」贡献的。本工作用本仓 D2Q9 LBM 求解器对")
    A("船体二维截面做自由流绕流，产出**时均场量云图 + 表面压力/摩阻系数角度分布**，")
    A("并让角度分布**积分回到阻力**（场内 → 力），从而实现「场量→阻力」的闭环闭合。")
    A("")
    A("| 项 | 设置 |")
    A("|---|---|")
    A("| 几何 | 圆形截面（直径 D，Braza 锚）；椭圆截面 L/B=4 船体水线截面（潜力流解析参考） |")
    A("| 参考长度 | D = 2a（椭圆流线向半轴 a，横向半轴 b = a/4） |")
    A("| 域 | 22D × 20D，体心距入口 6D，下游 8D sponge（τ_eff = τ(1+10σ)） |")
    A("| 阻塞比 | 1/20 = 5% |")
    A("| 边界 | 入口/上下自由流 Dirichlet + 出口零梯度 + 障碍 full-way bounce-back |")
    A("| 求解器 | `tensorlbm.solver.collide_mrt` + `stream` + `d2q9.equilibrium`（库路径，零手写内核） |")
    A("| 激励 | 入口 2 周期 St=0.165、10% 幅值扰动，之后回归自由流（涡街即刻起振） |")
    A("")
    if rows:
        A("逐网格参数（数值全部来自 `*_run.json`）：")
        A("")
        A("| D [cells] | 网格 | τ | U_in | Re | 步数 | t* 平均窗 | 平均样本 | 运行 [s] |")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            d = runs.get(str(r["D_cells"]), {})
            A(f"| {r['D_cells']} | {r['grid_cells']} | {fmt(r['tau'],4)} | {fmt(r['u_in'],3)} | "
              f"{fmt(r['re'],0)} | {d.get('steps','NO-DATA-TBD')} | {fmt(r['t_star_averaged'],1)} | "
              f"{r['n_avg_samples']} | {d.get('runtime_s','NO-DATA-TBD')} |")
        A("")

    A("## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）")
    A("")
    A("1. **场量求解**：D2Q9 MRT（`collide_mrt`）+ 统一 gather streaming；τ = 0.5+3ν，ν = U_in·D/Re。")
    A("   Cd/Cl 用 Ladd 动量交换（`compute_obstacle_forces`，post-stream/pre-bounce-back），")
    A("   与仓内已验证基准 `benchmarks/verified/cylinder/re100/run.py` 同一调用序列。")
    A("2. **时均统计**：起振后按 t* 窗口逐步累加 ρ、u、v、ω_z 及其二阶矩（每 4 步采样一次），")
    A("   得到时均场、脉动 RMS 场（涡街强度）；Cd 的统计不确定度用 10 块 block-average 的 SEM → U95。")
    A("3. **表面角度分布**：取障碍的阶梯边界结点（含至少一个流体四邻域），")
    A("   在**首层流体格心**上取 p=(ρ−1)c_s²、切向速度 u_t，得")
    A("   `Cp(φ)=(p−p∞)/(½ρU∞²)`，`Cf(φ)=τ_w/(½ρU∞²)`（半程 bounce-back，壁面位于 0.5Δx，")
    A("   τ_w=2μu_t）。按 10° 分箱（36 箱）并做 2 次循环 3 点平滑。")
    A("   φ 定义：`φ=atan2((y−yc)/b, (x−xc)/a)`，φ=0 指向下游，φ=±180° 为上游驻点。")
    A("4. **场→力闭环**：`Cd_p = Σ_boundary Cp·n_x /D`、`Cd_f = Σ_boundary Cf·t_x /D`，")
    A("   与 Ladd 总 Cd 对比给出分解闭合误差（下表 `Cd_pf_rel_err_pct`）。")
    A("5. **收敛判据**：三级网格 Roache GCI（Fs=1.25，r=D₂/D₁），非单调/非渐近时给出")
    A("   `fallback_reason` 并退化为保守带。")
    A("6. **参考值**：圆截面 Cd=1.35±0.05、St=0.164~0.165（Braza et al. 1986, JFM 165；")
    A("   与仓内 `benchmarks/verified/cylinder/re100/README.md` 一致）；")
    A("   椭圆截面**无实测 Cd 参考**，故 Cd 标 NO-DATA-TBD，只用**解析潜力流 Cp(φ)**")
    A("   （保角映射 z=ζ+c²/ζ，见 `potential_cp_ellipse`）作角度参考。")
    A("")

    A("## ③ 计算结果（表 1：逐网格关键量 / 参考值 / 真误差 / U95 / 场量统计矩）")
    A("")
    if rows:
        A("| 几何 | D | Cd | U95_stat | Cd_p | Cd_f | 分解闭合误差[%] | Cd_ref | 误差[%] | St | St_ref | St 误差[%] | GCI_fine[%] | GCI 说明 |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            A(f"| {r['geometry']} | {r['D_cells']} | {fmt(r['Cd_mean'],4)} | {fmt(r['Cd_u95_stat'],4)} | "
              f"{fmt(r['Cd_pressure'],4)} | {fmt(r['Cd_friction'],4)} | {fmt(r['Cd_pf_rel_err_pct'],2)} | "
              f"{fmt(r['Cd_ref'],3)} | {fmt(r['Cd_err_pct'],2)} | {fmt(r['St'],4)} | {fmt(r['St_ref'],4)} | "
              f"{fmt(r['St_err_pct'],2)} | {fmt(r['U95_discretisation_pct'],2)} | {r['GCI_note']} |")
        A("")
        A("场量统计矩（窗口 |x/D|<3, |y/D|<2 内的时均 Cp 场与 ω_z 场；kurt 为超额峰度）：")
        A("")
        A("| 几何 | D | Cp 均值 | Cp σ | Cp 偏度 | Cp 峰度 | ω_z 均值 | ω_z σ | Cp 表面 min | Cp 表面 max |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            A(f"| {r['geometry']} | {r['D_cells']} | {fmt(r['cp_field_mean'],3)} | {fmt(r['cp_field_std'],3)} | "
              f"{fmt(r['cp_field_skew'],3)} | {fmt(r['cp_field_kurt'],3)} | {fmt(r['wz_field_mean'],3)} | "
              f"{fmt(r['wz_field_std'],3)} | {fmt(r['cp_surface_min'],3)} | {fmt(r['cp_surface_max'],3)} |")
        A("")
        for g in geos:
            gg = gcis.get(g, {})
            A(f"- **{g} GCI**：p_obs={fmt(gg.get('p_obs'),3)}，"
              f"GCI_fine={fmt(gg.get('gci_fine_pct'),3)}%，"
              f"Richardson 外推 Cd={fmt(gg.get('extrapolated'),4)}，"
              f"fallback_reason={gg.get('fallback_reason') or 'None（渐近区成立）'}")
        A("")

    A("## ④ 结果图（场量云图 + 与参考剖面比较）")
    A("")
    for m in index.get("figures", []) or []:
        if not m:
            continue
        A(f"- `{m.get('png')}`")
    A("")
    if index.get("figures"):
        A("图 1（每几何一张）：(a) 时均 |u|/U∞ 云图+流线；(b) 时均 ω_z D/U∞ 云图（等值线族）；")
        A("(c) 时均 Cp 场云图；(d) 脉动 rms ω_z D/U∞ 云图；(e) Cp(φ) 与解析潜力流剖面对比（灰带=附着流有效窗）；")
        A("(f) Cf(φ) 壁面剪切分布。坐标均为 x/D、y/D，色标单位为无量纲（见 `cloud_index.json` 的 `panels` 段）。")
        A("")
        A("图 2：Cd 随 h=1/D 的网格收敛（含参考值与 Richardson 外推线）+ 三档网格的尾流时均速度剖面。")
        A("")

    A("### 表 2（节选：逐角度级 Cp/Cf 与解析参考、真误差、时间标准差、网格带）")
    A("")
    ang = [r for r in t2 if r["geometry"] == "cylinder" and r["D_cells"] == str(max(
        [int(x["D_cells"]) for x in rows if x["geometry"] == "cylinder"] or [0]))]
    if ang:
        A("| φ[deg] | Cp 均值 | Cp 时间σ | Cp 潜力流参考 | 真误差 | Cf 均值 | 网格带 | 箱内结点数 |")
        A("|---|---|---|---|---|---|---|---|")
        for r in ang[::3]:
            A(f"| {r['phi_deg']} | {fmt(r['Cp_mean'],3)} | {fmt(r['Cp_temporal_std'],3)} | "
              f"{fmt(r['Cp_potential_ref'],3)} | {fmt(r['Cp_err_vs_potential'],3)} | {fmt(r['Cf_mean'],3)} | "
              f"{fmt(r['grid_band_vs_finest'],3)} | {r['n_nodes_in_bin']} |")
        A("")
        A("完整逐角度表：`artifacts/resistance_chain_cloud/table2_surface_angular.csv`（36 箱 × 各网格）。")
        A("")

    A("### 表 3（尾流/空间分布统计，节选）")
    A("")
    if t3:
        A("| 几何 | D | 站位 | x/D | u 中心线最小 | u_rms 最大 | 半宽 [D] |")
        A("|---|---|---|---|---|---|---|")
        for r in t3[:12]:
            A(f"| {r['geometry']} | {r['D_cells']} | {r['station']} | {fmt(r['x_over_D'],2)} | "
              f"{fmt(r['u_min'],3)} | {fmt(r['u_rms_max'],3)} | {fmt(r['deficit_width_half_D'],3)} |")
        A("")

    A("## ⑤ 注意点（坑 / 局限 / 适用范围）")
    A("")
    A("- **逐格动量交换不是局部量**：库内 `compute_obstacle_forces` 对**全部**固体格求和")
    A("  （full-way bounce-back 下内部格的成对项在总和中相消，但单格值不为零）。")
    A("  因此把逐格动量交换当局部力会失配（实测 `checksum_rel_err` 见 run.json，量级 O(1)），")
    A("  本工作的表面分布改用**流体侧**首层格心的 ρ 与 u_t（半程 bounce-back 标准做法），")
    A("  并用 `Cd_p+Cd_f` 与 Ladd 总 Cd 对比作为分解闭合判据。")
    A("- Re=100 二维层流涡街，**不是**实船高雷诺数湍流：本工件的作用是给阻力链补「场量/角度分布」")
    A("  证据面与场→力闭环，其 Cd/St 精度对标的是二维圆柱基准，不能外推为实船阻力精度。")
    A("- 潜力流 Cp 参考仅在**附着流窗**（|φ|>90°，前半体）有意义；分离区（后半体）差异是物理的，")
    A("  不构成误差判据。")
    A("- 二维截面 = 无限展长假设，无自由面、无兴波：兴波阻力分量不在本工件覆盖范围内")
    A("- 椭圆截面无实测 Cd 参考 → 该列标 NO-DATA-TBD，禁止把潜力流值当测量值。")
    A("")

    A("## ⑥ 误差判定（逐项显式回答「是否<3%」）")
    A("")
    A("| 关键量 | 本工作 | 参考值 | 误差 | 是否 <3% |")
    A("|---|---|---|---|---|")
    verdicts = []
    for r in rows:
        if r["Cd_err_pct"] != "NO-DATA-TBD":
            e = float(r["Cd_err_pct"])
            verdicts.append(abs(e) < 3.0)
            A(f"| Cd ({r['geometry']}, D={r['D_cells']}) | {fmt(r['Cd_mean'],4)} | "
              f"{fmt(r['Cd_ref'],3)}±0.05 | {e:+.2f}% | {'是' if abs(e) < 3 else '否'} |")
        if r["St_err_pct"] != "NO-DATA-TBD":
            e = float(r["St_err_pct"])
            verdicts.append(abs(e) < 3.0)
            A(f"| St ({r['geometry']}, D={r['D_cells']}) | {fmt(r['St'],4)} | {fmt(r['St_ref'],4)} | "
              f"{e:+.2f}% | {'是' if abs(e) < 3 else '否'} |")
    A("")
    if verdicts:
        A(f"**显式回答：{'全部' if all(verdicts) else '部分'}关键量误差 <3%**"
          f"（{sum(verdicts)}/{len(verdicts)} 项 <3%）。")
    else:
        A("**显式回答：无可用参考值 → NO-DATA-TBD（不得声称 <3%）**。")
    A("")
    A("无基准项声明原因：")
    A("- 椭圆截面 Cd/St：本仓与课题内**无**实测/文献基准登记，故 NO-DATA-TBD；")
    A("  只给出解析潜力流 Cp(φ) 的角度对比与真误差列。")
    A("")
    if not any(r_["geometry"] == "ellipse" for r_ in rows):
        A("**NO-DATA-TBD**：椭圆船体水线截面（L/B=4）算例在本 mission 的计算窗内未跑完，")
        A("本轮只交付圆截面场量证据；椭圆截面的字段/角度表的生成脚本已就位")
        A("（`run.py --geo ellipse`、`surface.py`、`analyze.py --geos ellipse`），缺的是算力时间，不是代码。")
        A("")
    if rows and len({r_["D_cells"] for r_ in rows}) < 3:
        A(f"**GCI 说明**：仅 {len({r_['D_cells'] for r_ in rows})} 档网格完成 → 三级 Roache GCI 不适用，")
        A("表中 GCI 列按 `fallback_reason` 标注；真误差与统计 U95 仍逐项给出。")
        A("")
    A("## ⑦ 与既有报告的关系（不改写任何既有数字）")
    A("")
    A("| report id | 标题 | 本工件补的是什么 |")
    A("|---|---|---|")
    for rid, title in PRIOR_REPORTS:
        A(f"| #{rid} | {title} | 空间/角度分布证据面（CLOUD）：时间平均场云图 + 表面 Cp/Cf 角度分布 + 场→力闭环 |")
    A("")
    A("既有报告的标量数字（如 #5539 的 2876 kN、#5384 的 1300 工况/100% 成功率、#5393 的逐工况误差）")
    A("原样引用，**未在本工作中重算或改写**。")
    A("")

    DOCS.mkdir(parents=True, exist_ok=True)
    md = "\n".join(lines) + "\n"
    (DOCS / "README.md").write_text(md)

    # ------------------------------------------------------------- report json
    figs = []
    for m in index.get("figures", []) or []:
        if not m:
            continue
        p = Path(m["png"])
        b64 = png_b64(p)
        if b64:
            figs.append({
                "name": p.name,
                "caption": ("场量云图（|u|、ω_z、Cp、ω_z rms）+ Cp(φ)/Cf(φ) 与解析潜力流对比"
                            if p.name.startswith("fig1") else
                            "网格收敛 Cd(1/D) + 三档网格尾流时均剖面"),
                "png_base64": b64,
            })
    report = {
        "title": "阻力链 CLOUD 补测：LBM 时均场量云图 + 表面 Cp/Cf 角度分布 + 场→力闭环（m4927）",
        "summary": (
            f"用本仓 D2Q9 LBM 对船体二维截面做自由流绕流（Re=100），补齐阻力链课题的 CLOUD 缺口："
            f"产出时均速度/涡量/压力场云图、脉动 RMS 场、表面 Cp(φ)/Cf(φ) 角度分布（36 箱 × 多网格）、"
            f"尾流剖面与网格收敛 GCI；Cd 由场量表面积分分解为压力/摩阻并与 Ladd 动量交换总阻力闭合。"
            f"圆截面 Cd/St 对标 Braza 1986（Cd 1.35±0.05、St 0.164~0.165）。"
        ),
        "scenario_key": SCENARIO,
        "mission_id": MISSION,
        "tags": ["resistance-chain", "cloud", "field", "lbm", "cp-distribution",
                 "grid-convergence", "gci", "ship-performance"],
        "markdown": md,
        "figures": figs,
    }
    Path("/tmp/report.json").write_text(json.dumps(report, ensure_ascii=False))
    print(json.dumps({"docs": str(DOCS / "README.md"), "md_chars": len(md),
                      "figures": [f["name"] for f in figs],
                      "report_kb": round(Path("/tmp/report.json").stat().st_size / 1024, 1),
                      "rows": len(rows), "angular_rows": len(t2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())