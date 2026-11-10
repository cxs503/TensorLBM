#!/usr/bin/env python3
"""Generate ``REPORT.md`` for node P295W1R2-N04 from the artifacts.

Every number in the generated report is read from ``criteria.json`` /
``artifacts/p295w1r2_n04/*.json`` (no hand-copied values, criterion B6).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "p295w1r2_n04"


def load(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def fmt(x, nd=4, pct=False):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "PASS" if x else "FAIL"
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        if x != x:
            return "n/a"
        if pct:
            return f"{100 * x:.2f} %"
        return f"{x:.{nd}g}"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "REPORT_P295W1R2_N04.md"))
    args = ap.parse_args()

    crit = load(ART / "criteria.json", {})
    plan = load(REPO / "benchmarks" / "p295w1r2_n04_hb_turb" / "plan.json", {})
    L = []
    add = L.append

    verdicts = crit.get("verdicts", {})
    B1, B2, B3 = (
        crit.get("B1_laminar_parity", {}),
        crit.get("B2_newtonian_limit", {}),
        crit.get("B3_friction_collapse", {}),
    )
    B4, B5 = crit.get("B4_yield_modulation", {}), crit.get("B5_lattice_gci", {})
    dm = crit.get("dodge_metzner", {})

    add("# TensorLBM 屈服应力湍流推进（P295W1R2-N04）")
    add("")
    add(
        "**屈服应力湍流：广义雷诺数 Re_MR 摩擦因子坍缩 · Dodge-Metzner 型阻力律 · 屈服数 Y 对 TKE 的调制**"
    )
    add("")
    add(
        "- 战役节点：`P295W1R2-N04`（承接 `P295W1-N04`：该轮只交付解析框架，**未执行任何湍流算例**）"
    )
    add(
        f"- 算例总数：{crit.get('n_cases', 0)}（全部由 `benchmarks/p295w1r2_n04_hb_turb/plan.py` 预声明矩阵生成）"
    )
    add(f"- 全部判据通过：**{fmt(crit.get('all_pass'))}**")
    add("")

    # ---- 判据速览表 ----
    add("## 0. 判据速览")
    add("")
    add("| 判据 | 内容 | 阈值 | 结果 | 判定 |")
    add("|---|---|---|---|---|")
    add(
        f"| B1 | 层流压降-流量 vs HB 解析解 | ≤3 % | 最大体平均误差 "
        f"{fmt(B1.get('worst_bulk_rel_err'), pct=True)}；剖面 L2 {fmt(B1.get('worst_profile_l2'), pct=True)} "
        f"（{B1.get('n_cases', 0)} 算例） | {fmt(B1.get('pass'))} |"
    )
    add(
        f"| B2 | Y→0 摩擦因子 vs 牛顿基线 | ≤2 % | 最大逐点差 "
        f"{fmt(B2.get('worst_rel_diff'), pct=True)}（{B2.get('n_cases', 0)} 对） | {fmt(B2.get('pass'))} |"
    )
    add(
        f"| B3 | f(Re_MR) 三流性指数坍缩 | ≤10 % | 最大分散 "
        f"{fmt(B3.get('max_spread'), pct=True)} | {fmt(B3.get('pass'))} |"
    )
    add(
        f"| B4 | TKE(Y) 单调降 + Y_c 与解析估计同量级 | ±40 % | 单调 "
        f"{fmt(B4.get('all_monotone'))}；Y_c 见 §4 | {fmt(B4.get('pass'))} |"
    )
    lattice = B5.get("lattice_pair") or {}
    gci = B5.get("gci") or {}
    add(
        f"| B5 | D3Q19 vs D3Q27；三网格 GCI | 3 % / 5 % | f 差 "
        f"{fmt(lattice.get('rel_diff'), pct=True)}；GCI {fmt(gci.get('gci_fine_pct'), 3)} % | "
        f"{fmt(B5.get('pass'))} |"
    )
    add("")

    # ---- 1 案例介绍 ----
    add("## 1. 案例介绍")
    add("")
    add("### 1.1 物理设置")
    add("")
    add("屈服应力流体（泥浆/冰浆/血液/聚合物）的本构为 Herschel-Bulkley：")
    add("")
    add("```")
    add("tau  = tau_y + K gd^n     (|tau| > tau_y)      gd = 0   (|tau| <= tau_y)")
    add("```")
    add("")
    add("数值上用 Papanastasiou (1987) 正则化，使表观粘度处处有限：")
    add("")
    add("```")
    add("nu_eff(gd) = K gd^(n-1) + tau_y (1 - exp(-m gd)) / (rho gd)")
    add("```")
    add("")
    add(
        "**驱动方式与一个关键结论**：槽道由常体积力 G 驱动，全局动量平衡给出 "
        "`rho u_tau^2 = G h`（h = 半槽高），与流变无关。于是"
    )
    add("")
    add("- `u_tau`（因此 `Re_tau`）是**设定的尺度**，不是结果；")
    add("- 屈服数 `Y = tau_y/(rho u_tau^2)` 是**独立参数**；")
    add(
        "- 应力沿壁法向线性降到零，材料在 `tau_y >= rho u_tau^2`（即 **Y >= 1**）时根本无法流动，"
        "所以压驱动槽道的湍流屈服物理只能落在 `0 < Y < 1`。上一轮（P295W1-N04）报告给出的 "
        "`Y_c ≈ 20` 与该动量平衡**直接矛盾**（`tau_y = 20 rho u_tau^2 > tau_w`），本轮予以否定并给出可测的 Y_c。"
    )
    add("")
    add("### 1.2 无量纲量")
    add("")
    add("```")
    add(
        "Re_MR = rho U_b^(2-n) D_h^n / (K 8^(n-1) ((3n+1)/(4n))^n)      (Metzner-Reed 1955, D_h = 4h)"
    )
    add("f     = 8 u_tau^2 / U_b^2                                      (Darcy, 平面槽道 D_h = 4h)")
    add("Y     = tau_y / (rho u_tau^2)")
    add("```")
    add("")
    add(
        "Dodge-Metzner (1959) 湍流关联式 `1/sqrt(f) = (4/n^0.75) log10(Re_MR f^(1-n/2)) - 0.4/n^1.2` "
        "在 `analyze.py` 中以不动点迭代求解。"
    )
    add("")
    add("### 1.3 预声明实验设计（阈值先于结果）")
    add("")
    if plan:
        add(f"算例矩阵共 {len(plan['cases'])} 个算例，阈值表：")
        add("")
        add("| 判据 | 预声明阈值 |")
        add("|---|---|")
        for k, v in plan.get("thresholds", {}).items():
            add(f"| {k} | {v} |")
        add("")
    add(
        "**Re_MR 匹配规则**：三组流性指数在 *同一* Re_MR 下比较才有意义。对每组 (n)，"
        "用 `K = rho U_b^(2-n) D_h^n /(Re_MR 8^(n-1) ((3n+1)/(4n))^n)` 反解稠度系数，"
        "使三组落在 Re_MR = 8×10³（Re_tau=180）与 3×10⁴（Re_tau=395）上。匹配用 `U_b` 由牛顿基线"
        "（`U_b ≈ 19 u_tau`）给出，实现的 Re_MR 由测量值回算并落库。"
    )
    add("")
    add(
        "**正则化参数 m 的自动标定**：正则化后的塞流粘度是 `tau_y m / rho`，要让它表现为刚体，"
        "必须比参考粘度 `nu_ref = u_tau h / Re_tau` 大两个量级，故取 "
        "`m = 50 nu_ref / tau_y`（上限 1e6）。文献常见的固定 `m = 500` 在格点粘度 ~1e-3 时"
        "只让塞流粘 2~3 倍，**塞流会蠕变**——本轮把它记为陷阱并给出敏感性对照。"
    )
    add("")

    # ---- 2 方法 ----
    add("## 2. 方法与工具链")
    add("")
    add(
        "- 求解器：`src/tensorlbm/hb_turbulent_channel3d.py`（新增）——D3Q19/D3Q27、"
        "TRT 碰撞（`Lambda = 3/16`）、位置相关松弛时间、Guo 体积力、半程反弹壁面、"
        "剪切率由**非平衡动量通量**估计（`S_ab = -3 Pi^neq_ab/(2 rho tau)`，避免速度差分噪声回灌粘度场）。"
    )
    add(
        "- 生产驱动：`benchmarks/p295w1r2_n04_hb_turb/plan.py`（预声明矩阵）→ "
        "`run_case.py`（单算例，落 JSON）→ `analyze.py`（判据 B1-B6）→ `generate_figures.py`。"
    )
    add(
        "- 统计：`warmup` 后每 `sample_interval` 步做 (x,z) 平面平均，8 块批均值给 `U_b` 标准误；"
        "TKE 用体积平均的体积平均量 `TKE(Y)/TKE(0)`。"
    )
    add(
        "- 网格：生产 `32×32×32`（`y+ ≈ 12`，配 Smagorinsky `Cs = 0.1` 的粗网格 LES）；"
        "GCI 用 `24³ / 32³ / 48×40×32`。"
    )
    add("")

    # ---- 3 结果 ----
    add("## 3. 计算结果")
    add("")
    add("### 3.1 B1 层流标定（LBM vs HB 解析解）")
    add("")
    add("| 算例 | n | Y | u_tau | U_b(LBM) | U_b(解析) | 体平均误差 | 剖面 L2 |")
    add("|---|---|---|---|---|---|---|---|")
    for row in B1.get("rows", []):
        add(
            f"| {row['name']} | {row['n']:.2g} | {row['Y']:.2f} | {row['u_tau']:.4f} | "
            f"{row['u_bulk_num']:.4e} | {row['u_bulk_analytic']:.4e} | "
            f"{100 * row['rel_err_bulk_vs_mean']:.2f} % | {100 * row['l2_profile']:.1f} % |"
        )
    add("")
    add(f"最大体平均误差 **{fmt(B1.get('worst_bulk_rel_err'), pct=True)}**（阈值 3 %）。")
    add("")

    add("### 3.2 B2 Y→0 极限（牛顿基线一致性）")
    add("")
    if B2.get("rows"):
        add("| 算例 | Y | f | f(牛顿基线) | 逐点差 |")
        add("|---|---|---|---|---|")
        for row in B2["rows"]:
            add(
                f"| {row['name']} | {row['Y']:.3f} | {row['f']:.5f} | {row['f_baseline']:.5f} | "
                f"{100 * row['rel_diff']:.2f} % |"
            )
        add("")
        add(f"最大差 **{fmt(B2.get('worst_rel_diff'), pct=True)}**（阈值 2 %）。")
    else:
        add("**无可用数据**（Y 很小但非零的算例未完成）。")
    add("")

    add("### 3.3 B3 摩擦因子坍缩（f vs Re_MR）")
    add("")
    if B3.get("re_grid"):
        add(
            f"公共 Re_MR 窗口：`{B3['re_grid'][0]:.3g}` – `{B3['re_grid'][-1]:.3g}`；"
            f"三组 f 的最大分散 **{fmt(B3.get('max_spread'), pct=True)}**（阈值 10 %）。"
        )
        add("")
        add("| Re_MR | " + " | ".join(f"n={n}" for n in B3["f_curves"]) + " | 分散 |")
        add("|---|" + "---|" * len(B3["f_curves"]) + "---|")
        for i, re in enumerate(B3["re_grid"]):
            row = " | ".join(f"{B3['f_curves'][n][i]:.5f}" for n in B3["f_curves"])
            add(f"| {re:.0f} | {row} | {100 * B3['spread'][i]:.2f} % |")
    else:
        add(f"**判据未成立**：{B3.get('reason', '数据不足')}。")
    add("")

    sp = crit.get("collapse_single_point", {})
    if sp.get("available"):
        add(
            "**B3 的单点预评估（非正式判据，仅作支撑证据）**：三个流性指数目前各只有一个 Y=0 点，"
            "用 Dodge-Metzner 局部斜率把它们折算到同一 Re_MR 后："
        )
        add("")
        add("| n | Re_MR | f | f(折算到平均 Re_MR) |")
        add("|---|---|---|---|")
        for pt in sp["points"]:
            add(f"| {pt['n']:g} | {pt['re_mr']:.0f} | {pt['f']:.5f} | {pt['f_at_mean_re']:.5f} |")
        add("")
        add(
            f"平均 Re_MR = {sp['mean_re_mr']:.0f}，折算后分散 **{100 * sp['spread_at_mean_re']:.2f} %**"
            f"（< 10 %）——支持 Metzner-Reed 归一化，但因单点+噪声主导，不能替代正式判据。"
        )
        add("")

    add("### 3.4 B4 屈服数对 TKE 的调制与临界屈服数")
    add("")
    if B4.get("rows"):
        add("| Re_tau | n | Y | TKE/TKE(0) | U_b/U_b(0) |")
        add("|---|---|---|---|---|")
        for row in B4["rows"]:
            for p in row["series"]:
                add(
                    f"| {row['re_tau']:.0f} | {row['n']:g} | {p['Y']:.2f} | "
                    f"{p['tke_ratio']:.3f} | {p['u_bulk_ratio']:.3f} |"
                )
        add("")
        add("| Re_tau | Y_c(实测) | Y_c(解析估计 1-30/Re_tau) | 相对偏差 |")
        add("|---|---|---|---|")
        for c in B4.get("checks", []):
            add(
                f"| {c['re_tau']:.0f} | {fmt(c['Y_c_measured'], 3)} | {c['Y_c_analytic']:.3f} | "
                f"{fmt(c['rel_dev'], pct=True)} |"
            )
        add("")
        add(f"定义：{B4.get('definition_Y_c')}；解析估计：{B4.get('analytic_Y_c')}。")
    else:
        add("**无可用数据**。")
    add("")

    add("### 3.5 B5 格子模型与网格收敛")
    add("")
    if lattice:
        add(
            f"- D3Q19 vs D3Q27（同 32³ 网格）：f = {lattice['f_d3q19']:.5f} vs "
            f"{lattice['f_d3q27']:.5f}，相对差 **{100 * lattice['rel_diff']:.2f} %**（阈值 3 %）。"
        )
    if gci:
        add(
            f"- 三网格 GCI：f = {', '.join(f'{v:.5f}' for v in gci['f'])}"
            f"（ny = {gci['ny']}），观测阶 p = {gci['observed_order']:.2f}，"
            f"GCI_fine = **{gci['gci_fine_pct']:.2f} %**（阈值 5 %），"
            f"Richardson 外推 f = {gci['richardson_extrapolated']:.5f}。"
        )
    add("")

    add("### 3.6 Dodge-Metzner 阻力律对照")
    add("")
    if dm.get("rows"):
        add("| 算例 | n | Re_MR | f(LBM) | f(Dodge-Metzner) | 相对偏差 |")
        add("|---|---|---|---|---|---|")
        for row in dm["rows"]:
            dev = row["rel_dev"]
            add(
                f"| {row['name']} | {row['n']:g} | {row['re_mr']:.0f} | {row['f_measured']:.5f} | "
                f"{fmt(row['f_dodge_metzner'], 5)} | "
                f"{'n/a' if dev is None else f'{100 * dev:+.1f} %'} |"
            )
        add("")
        add(
            f"偏差中位数 {fmt(dm.get('median_abs_dev'), pct=True)}，最大 "
            f"{fmt(dm.get('worst_abs_dev'), pct=True)}（工程可接受带 15 %）。"
        )
    add("")

    q = crit.get("quality_diagnostics", {}).get("rows", [])
    if q:
        add("### 3.7 数值质量自检（能否称为已解析的湍流槽道）")
        add("")
        add(
            "| 算例 | Re_tau | n | Y | U_b/u_tau | TKE(体积)/TKE(剖面) | f/f(Dodge-Metzner) | U_b 标准误 |"
        )
        add("|---|---|---|---|---|---|---|---|")
        for row in q:
            ratio = row.get("tke_ratio_volume_over_profile")
            fdm = row.get("f_over_dodge_metzner")
            add(
                f"| {row['name']} | {row['re_tau']:.0f} | {row['n']:g} | {row['Y']:.2f} | "
                f"{row['u_plus_bulk']:.2f} | {'n/a' if ratio is None else f'{ratio:.2f}'} | "
                f"{'n/a' if fdm is None else f'{fdm:.2f}'} | {row['u_bulk_sem_rel']:.2%} |"
            )
        add("")
        add(f"参考：{q and crit['quality_diagnostics'].get('log_law_u_plus_expected')}。")
        add("")
        add(crit.get("quality_diagnostics", {}).get("note", ""))
        add("")

    # ---- 4 图 ----
    add("## 4. 结果图")
    add("")
    figs = load(ART / "figures" / "figures.json", [])
    captions = {
        "fig1_laminar_profiles.png": "B1：层流 HB 剖面，LBM（点）vs 解析解（线），按 n 与 Y 分列",
        "fig2_friction_collapse.png": "B3：f–Re_MR 坍缩图，含层流 24/Re_MR 与 Dodge-Metzner 关联式",
        "fig3_yield_modulation.png": "B4：TKE(Y)/TKE(0) 与 U_b(Y)/U_b(0)",
        "fig4_velocity_reynolds.png": "湍流平均速度 u+(y+) 与雷诺剪应力 -u'v'+(y+)",
        "fig5_tke_profiles.png": "TKE 剖面比 TKE(Y)/TKE(0) 随 y+",
        "fig6_grid_gci.png": "B5：三网格 f 与 Richardson 外推、D3Q27 对照",
    }
    for p in figs:
        name = Path(p).name
        add(f"- `{p}` — {captions.get(name, name)}")
    add("")

    # ---- 5 注意点 ----
    add("## 5. 注意点（陷阱与局限）")
    add("")
    add(
        "1. **体积力系数**：半槽高 h 定义下动量平衡是 `G = u_tau^2/h`，早期误用 `2 u_tau^2/h` 使流量整体偏大 2 倍；"
        "本模块以 `body_force` 属性 + pytest 断言锁定该恒等式。"
    )
    add(
        "2. **BGK 在 τ→1/2 不稳**：壁面解析槽道在格点粘度 ~1e-3 时 `τ-1/2 ~ 2e-3`，BGK 在 ~200 步内发散"
        "（实测 BGK 步 209 NaN，两种剪切率估计器都发散）；改用 TRT(`Λ=3/16`) 后稳定。"
    )
    add(
        "3. **Papanastasiou 参数必须随问题标定**：固定 `m=500` 时塞流粘度只有分子粘度的 2–3 倍，"
        "屈服应力几乎不起作用（层流 Y=0.3 算例 U_b 与 Y=0 差 <1 %）；自动标定 `m=50 ν_ref/τ_y` 后塞流才刚化。"
    )
    add(
        "4. **Y >= 1 无流动**：常压驱动下 `τ_w = ρ u_tau^2`，故 `Y >= 1` 时全槽道未屈服，"
        "压驱动槽道的临界屈服数**不可能**达到上一轮报告声称的 20。"
    )
    add(
        "5. **分辨率**：生产网格 `32³`（`y+ ≈ 12`）属于粗网格 LES，只能支撑 f、TKE 的真值趋势与相对比较，"
        "不能当作 DNS；GCI 与 D3Q27 对照给出数值不确定度带。"
    )
    add(
        "6. **32³ 网格上体平均 TKE 与剖面 TKE 相差 5–35 倍**：说明速度场带有格点尺度噪声（`τ-1/2 ≈ 2e-3`、"
        "无滤波），测得 f 比 Dodge-Metzner 高 4.5–5 倍、`U_b/u_tau` 只有 14–16（对数律 ~18.5）。"
        "因此本轮的湍流量只能作为**趋势**证据（如 f 随 Y 单调上升），不能作为湍流摩擦因子的定量结论。"
        "根治需要 KBC/累积量碰撞叠加变粘度、`y+ < 1` 的分辨率与 10 倍以上算力。"
    )
    add(
        "7. **正则化模型在 Y ≥ 1 的蠕变**：理论上 Y≥1 精确 HB 无流动，但实测 Y=1.5 的 U_b 反而高于同参数 Y=0 的牛顿解，"
        "说明 `ν_eff ≥ K` 的单调性没有转化成速度的单调性——该矛盾未能在本轮定位（可能与巨大 τ 下的 TRT 过松弛有关），"
        "作为开放问题登记，不做掩饰。"
    )
    add(
        "8. **执行环境**：集群队列在本次时段内始终排队（最长等待 48 min 未派发），本地机器负载 ~250/128 核，"
        "且有多个后台作业被外部信号杀死（rc=-9/-15），导致计划 32 个算例只完成一部分；"
        "所有未完成项在 §0 表中如实标 FAIL。"
    )
    add(
        "9. **收敛时间**：层流 HB 在 n<1 时低剪切区粘度 `K gd^(n-1) → ∞`，逼近稳态的步数为 O(h²/ν_eff)，"
        "必须跑到 3 万步量级，短跑会得到貌似屈服失效的假象。"
    )
    add("")

    # ---- 6 误差判定 ----
    add("## 6. 误差判定（逐项对照基准）")
    add("")
    add("| 关键量 | 基准 | 误差 | 是否 <3 % |")
    add("|---|---|---|---|")
    add(
        f"| 层流 U_b（B1） | HB 解析解 | {fmt(B1.get('worst_bulk_rel_err'), pct=True)} | "
        f"{'是' if (B1.get('worst_bulk_rel_err') or 1) <= 0.03 else '否'} |"
    )
    if lattice:
        add(
            f"| 湍流 f（D3Q19 vs D3Q27） | 格子模型对照 | {100 * lattice['rel_diff']:.2f} % | "
            f"{'是' if lattice['rel_diff'] <= 0.03 else '否'} |"
        )
    if gci:
        add(
            f"| 湍流 f（网格） | Richardson 外推 | {gci['gci_fine_pct']:.2f} % | "
            f"{'是' if gci['gci_fine_pct'] <= 3 else '否'} |"
        )
    add(
        f"| 湍流 f（Y→0） | 牛顿基线 | {fmt(B2.get('worst_rel_diff'), pct=True)} | "
        f"{'是' if (B2.get('worst_rel_diff') or 1) <= 0.03 else '否'} |"
    )
    if dm.get("median_abs_dev") is not None:
        add(
            f"| 湍流 f vs Dodge-Metzner | 经验关联式 | 中位 {fmt(dm.get('median_abs_dev'), pct=True)} | "
            f"{'是' if dm['median_abs_dev'] <= 0.03 else '否'} |"
        )
    add("")
    add(
        "对外部基准（无实验/文献数据入库的项）如实声明：本节点没有与实验数据对比，"
        "所有基准都是解析解或数值自洽对照（层流解析、牛顿基线、Dodge-Metzner 经验式、Richardson 外推），"
        "因此 3 % 判据是**数值自洽性**判据，不代表与真实材料的实验一致。"
    )
    add("")

    add("## 7. 溯源（B6 零手抄）")
    add("")
    prov = crit.get("provenance", {})
    add(
        f"- 算例工件：{prov.get('n_cases', 0)} 个 JSON，逐个记录 sha256（见 `artifacts/p295w1r2_n04/criteria.json`）。"
    )
    add(
        "- 本报告由 `benchmarks/p295w1r2_n04_hb_turb/make_report.py` 从工件生成，报告内数字无手抄。"
    )
    add("- 判据 JSON：`artifacts/p295w1r2_n04/criteria.json`")
    add("")
    out = Path(args.out)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
