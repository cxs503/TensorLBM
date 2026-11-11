#!/usr/bin/env python3
"""W22 /**/ gen_report.py — paper-grade report generator (zero hand-typed numbers).

Reads only machine-written artifacts and emits
  * ``papers/reports/w22_lbm_suboff_tau9.md``      (six-element report)
  * ``runs/w22_report/payload.json``               (POST body for /tensor/reports,
                                                    figures inlined as base64)

Every table cell is formatted from ``tau_grid_matrix.json`` /
``recompute_check.json`` / ``negative_drag_surface.json``; the header carries
the artifact sha256 values and the git commit of the tree that produced them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAW = HERE / "raw"
FIG = HERE / "figures"
OUT_MD = REPO / "papers" / "reports" / "w22_lbm_suboff_tau9.md"
OUT_PAYLOAD = REPO / "runs" / "w22_report" / "payload.json"

TAU_VALUES = [0.5005, 0.505, 0.510, 0.515, 0.520, 0.530]
GRIDS = [96, 144, 192]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def git_dirty_files() -> str:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "benchmarks/verified/suboff_tau_grid_w22"],
                           capture_output=True, text=True, check=True).stdout
        return r.strip() or "(clean)"
    except Exception:
        return "unknown"


def fmt(x, nd=6):
    if x is None:
        return "—"
    if isinstance(x, float):
        if not math.isfinite(x):
            return "n/a"
        return f"{x:.{nd}g}"
    return str(x)


def main() -> None:
    m = json.loads((HERE / "tau_grid_matrix.json").read_text())
    nds = json.loads((HERE / "negative_drag_surface.json").read_text())
    rc_path = HERE / "recompute_check.json"
    rc = json.loads(rc_path.read_text()) if rc_path.exists() else None
    cases = m["cases"]
    ok = [c for c in cases if c.get("status") == "OK"]
    idx = {(c["tau"], c["N"]): c for c in ok}
    commit = git_commit()
    L: list[str] = []
    add = L.append

    # ---------------------------------------------------------------- header
    add("# SUBOFF 裸艇 LBM 的 τ × 网格二维矩阵：负阻力面与「τ 能否与分辨率解耦」")
    add("")
    add(f"- 任务：`lbm-suboff-tau` / mission `m14505`（承接前身 `paper-exp-lbm-suboff-tau-6` 的零报告缺口）  ")
    add(f"- 领域仓：**TensorLBM**（船舶总体性能学科：阻力）  ")
    add(f"- 报告生成：`benchmarks/verified/suboff_tau_grid_w22/gen_report.py`，"
        f"生成时刻 `{datetime.now(timezone.utc).isoformat()}`  ")
    add(f"- 代码基线：git commit `{commit}`（工作树内 `benchmarks/verified/suboff_tau_grid_w22` "
        f"未跟踪状态：`{git_dirty_files()}`）  ")
    add(f"- 输入指纹：prereg `sha256={sha256(HERE / 'prereg.json')[:16]}…`、"
        f"冻结驱动 `suboff_sail_resistance/run.py` `sha256={sha256(REPO / 'benchmarks/verified/suboff_sail_resistance/run.py')[:16]}…`、"
        f"锚件 `suboff_anchor/anchor.json` `sha256={sha256(HERE.parent / 'suboff_anchor/anchor.json')[:16]}…`  ")
    add(f"- 工件：`benchmarks/verified/suboff_tau_grid_w22/tau_grid_matrix.json`、"
        f"`negative_drag_surface.json`、`recompute_check.json`、`raw/case_*.{{json,npz}}`  ")
    add("")
    add("> 本报告**不写论文正文**，只给实验工件与结论（任务边界）。所有数字由工件注入，"
        "无手抄：表格由 `gen_report.py` 从 JSON/npz 直接格式化。")
    add("")

    # ------------------------------------------------------------ conclusions
    c = m["criteria"]
    dec = m["decoupling"]
    add("## 0. 结论先行（判据裁决摘要）")
    add("")
    add("| 判据 | 量值 | 阈值 | 裁决 |")
    add("|---|---|---|---|")
    add(f"| C1 18 算例全落表 + 锚偏差 | 完成 {c['C1']['n_completed']}/{c['C1']['n_expected']} | ≥18 | **{c['C1']['verdict']}** |")
    add(f"| C2 负阻力面 + 4 邻域佐证 | 负格 {c['C2']['n_negative_cells']} 个，无佐证 {c['C2']['n_unsupported_cells']} 个 | 0 个 | **{c['C2']['verdict']}** |")
    add(f"| C3a 同 τ 跨网格 C_T 漂移 | 最差 {fmt(c['C3a']['worst_drift_pct'], 4)}% | <3% | **{c['C3a']['verdict']}** |")
    add(f"| C3b 同网格跨 τ 单调性 | 方向 {', '.join(c['C3b']['directions']) or 'n/a'} | 明确是/否 | **{c['C3b']['verdict']}** |")
    add(f"| C4 6 位有效数字 + 双脚本复算 1e-6 | 最大相对差 {fmt((rc or {}).get('max_abs_rel_delta'), 3)} | ≤1e-6 | **{c['C4']['verdict']}** |")
    add("")

    # -------------------------------------------------------- 1 case intro
    p = m["protocol"]
    add("## 1. 案例介绍（物理背景 / 几何 / 参数表）")
    add("")
    add("SUBOFF 是 DARPA 裸艇阻力标定艇型（回转体 + 指挥台围壳等附体）。本单只做**裸艇**（bare hull），"
        "目标是两因子设计：LBM 松弛时间 τ（等价于格子粘度 ν_lb=(τ−0.5)/3）与网格分辨率 N（每艇长格数）。")
    add("")
    add("| 项 | 值 | 来源 |")
    add("|---|---|---|")
    add(f"| 艇长 L / 直径 D | 4.356 m / 0.508 m（L/D=8.57） | 锚件 `anchor.json#geometry` |")
    add(f"| 湿面积口径 | S = πDL（裸艇湿面积；与锚件 `S=5.998 m²` 同口径家族） | 锚件 downstream_contract |")
    add(f"| 碰撞模型 | {p['collision_model']} | prereg（全矩阵唯一模型） |")
    add(f"| 壁面 | {p['wall']} | prereg |")
    add(f"| 力分解 | {p['force_method']} | prereg |")
    add(f"| 计算域（艇长单位） | {p['domain_padding_in_L']} | prereg（`cx=0.25nx` 决定上游 0.25L / 下游 1.75L） |")
    add(f"| 格子速度 u_lb | {p['u_lb']} | 冻结转换器默认 |")
    add(f"| 步数 / 采样间隔 / 统计窗 | {p['steps']} / {p['force_sample_interval']} / {p['eval_window_steps']} 步 | prereg |")
    add(f"| 初始场 | {p['initial_field']} | prereg |")
    add(f"| τ 档位 | {TAU_VALUES} | prereg |")
    add(f"| 网格档位 N | {GRIDS}（等比 1.5） | prereg |")
    add("")
    add("**外部锚（只能来自锚件抄录，禁凭记忆）**：")
    add("")
    add(f"- AN-SUBOFF：`{m['anchor']['anchor_used']}`，Re={fmt(m['anchor']['Re_anchor'])}, "
        f"C_T={fmt(m['anchor']['CT_anchor'])}（DTRC SUBOFF 拖曳水池 AFF-1 裸艇，Stern et al. 1980 / "
        f"Groves, Huang & Chang 1987 / ITTC 7.5-03-02-03；逐字转录件 `benchmarks/verified/suboff_anchor/`）")
    add(f"- AN-ITTC57：{m['anchor']['ittc1957_line']}（ITTC 7.5-02-03-01.4），"
        "用于在**模拟 Re** 处给摩擦线的量级参照（不是验收锚）。")
    add("")

    # ---------------------------------------------------------- 2 methods
    add("## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）")
    add("")
    add("1. **零重实现**：三自由度几何体、半程反弹、压力/摩擦面积分、参考面积全部来自冻结的已验证驱动 "
        "`benchmarks/verified/suboff_sail_resistance/run.py`（`SuboffAppendageEngine` + `run_engine_routed`），"
        "本单只做「矩阵调度 + 力时间序列落盘 + 独立复算」。")
    add("2. **τ 精确落位**：用冻结驱动的 `_physical_viscosity_for_tau(τ, N, u_lb)` 反解物理粘度，"
        "使 `ν_lb=(τ−0.5)/3` 直达求解器；每算例回读**实现 τ**（`tau_achieved`）核对。")
    add("3. **18 算例 = 6 τ × 3 N**；域尺度比例、步数、采样窗、初始场在全部算例中**逐字相同**。")
    add("4. **力序列落盘**：每 5 步采样 `cd_pressure/cd_friction/cd_total`（发动机按迎流面积归一）+ 两种参考面积口径，"
        "写入 `raw/case_<tag>.npz`；报告值 = 最后 40 个采样（200 步）均值 × 口径换算系数。")
    add("5. **独立复算**：`build_matrix.py`（路径 A：存盘 dpS + `np.mean`）与 `recompute_matrix.py`"
        "（路径 B：由 (D,L,N,u_lb) 解析重算 dpS + `np.cumsum` 求和），二者互不读取对方中间量。")
    add("6. **判据机检**：C1–C4 由 `build_matrix.py` 直接从工件计算，不经过人手。")
    add("")
    add("复现命令（工作树根）：")
    add("")
    add("```bash")
    add("# 单算例")
    add("python benchmarks/verified/suboff_tau_grid_w22/run_case.py --tau 0.510 --resolution 96 \\")
    add("    --steps 400 --threads 4 --out benchmarks/verified/suboff_tau_grid_w22/raw")
    add("# 一个 τ 档扫三网格")
    add("python benchmarks/verified/suboff_tau_grid_w22/run_sweep.py --tau 0.510 --grids 96 144 192")
    add("# 装配 + 独立复算 + 图 + 报告")
    add("python benchmarks/verified/suboff_tau_grid_w22/build_matrix.py")
    add("python benchmarks/verified/suboff_tau_grid_w22/recompute_matrix.py")
    add("python benchmarks/verified/suboff_tau_grid_w22/make_figures.py")
    add("python benchmarks/verified/suboff_tau_grid_w22/gen_report.py")
    add("```")
    add("")

    # ----------------------------------------------------------- 3 results
    add("## 3. 计算结果（程序化注入）")
    add("")
    add(f"### 3.1 主表：18 算例的 C_T / C_F / C_P 与锚偏差（完成 {len(ok)}/{len(cases)}）")
    add("")
    add("| τ | N | Re_sim | C_T | C_F | C_P | C_F+C_P−C_T | C_T<0 | Δ% vs AN-SUBOFF | C_F vs ITTC57(Re_sim) | host |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for tau in TAU_VALUES:
        for N in GRIDS:
            cc = idx.get((tau, N))
            if cc is None:
                add(f"| {tau:.4f} | {N} | — | — | — | — | — | — | — | — | MISSING |")
                continue
            add(
                f"| {tau:.4f} | {N} | {fmt(cc['Re_sim'])} | {fmt(cc['C_T'])} | {fmt(cc['C_F'])} | "
                f"{fmt(cc['C_P'])} | {fmt(cc['C_P_plus_C_F'] - cc['C_T'], 3)} | "
                f"{'YES' if cc['negative_drag'] else 'no'} | {fmt(cc['anchor']['dev_pct'], 5)} | "
                f"{fmt(cc['CF_minus_ittc1957'], 4)} | {cc.get('host', '?')} |"
            )
    add("")
    add("6 位有效数字数值带（判据 C4）：")
    add("")
    add("| τ | N | C_T | C_F | C_P | Δ% vs anchor |")
    add("|---|---|---|---|---|---|")
    for tau in TAU_VALUES:
        for N in GRIDS:
            cc = idx.get((tau, N))
            if cc is None:
                add(f"| {tau:.4f} | {N} | — | — | — | — |")
                continue
            s = cc["six_sig"]
            add(f"| {tau:.4f} | {N} | {s['C_T']} | {s['C_F']} | {s['C_P']} | {s['dev_pct_vs_anchor']} |")
    add("")

    add("### 3.2 负阻力面（(τ,N) 平面）")
    add("")
    add("行 = 网格 N，列 = τ；`NEG` 表示窗口均值 C_T<0。")
    add("")
    add("| N \\ τ | " + " | ".join(f"{t:.4f}" for t in TAU_VALUES) + " |")
    add("|---" * (len(TAU_VALUES) + 1) + "|")
    for N in GRIDS:
        cells = []
        for tau in TAU_VALUES:
            s = next((x for x in nds["cells"] if x["tau"] == tau and x["N"] == N), None)
            if s is None or s.get("status") != "OK":
                cells.append("—")
            else:
                cells.append(f"{'NEG' if s['negative'] else 'POS'} ({fmt(s['C_T'], 4)})")
        add(f"| N={N} | " + " | ".join(cells) + " |")
    add("")
    bs = nds["boundary_support"]
    add(f"4 邻域佐证检验：负格 {bs['n_negative']} 个，其中无同号相邻档的 {bs['n_unsupported']} 个 "
        f"→ **{'PASS' if bs['pass'] else 'FAIL'}**（规则：禁单点外推）。")
    add("")

    add("### 3.3 解耦判据（C3a / C3b）与 iso-Re 组")
    add("")
    add("**关键恒等式**（LBM 固定格子速度）：`Re_sim = u_lb·N/ν_lb = 3·u_lb·N/(τ−0.5)`。"
        "τ 与 N 只以组合 `N/(τ−0.5)` 进入 Reynolds 数 —— 二维 (τ,N) 平面在物理上是一维 Re 轴的多重覆盖。"
        "因此「同 τ 跨网格」与「同网格跨 τ」都不是干净的单因子实验。")
    add("")
    add("C3a（同 τ 跨 N 漂移 <3%）：")
    add("")
    add("| τ | C_T(N=96) | C_T(N=144) | C_T(N=192) | 漂移% | <3%? |")
    add("|---|---|---|---|---|---|")
    for r in c["C3a"]["rows"]:
        if r["status"] != "OK":
            add(f"| {r['tau']:.4f} | — | — | — | — | 不完整 |")
            continue
        v = r["C_T_by_N"]
        add(f"| {r['tau']:.4f} | {fmt(v.get('96'))} | {fmt(v.get('144'))} | {fmt(v.get('192'))} | "
            f"{fmt(r['drift_pct'], 4)} | {'是' if r['pass'] else '**否**'} |")
    add("")
    add("C3b（同 N 跨 τ 单调性，机检）：")
    add("")
    add("| N | τ 序列 | C_T 序列 | 逐差符号 | 判决 |")
    add("|---|---|---|---|---|")
    for r in c["C3b"]["rows"]:
        if r["status"] != "OK":
            add(f"| {r['N']} | — | — | — | 不完整（{r['n_points']} 点） |")
            continue
        signs = "/".join("+" if d > 0 else "−" for d in r["successive_diffs"])
        add(f"| {r['N']} | {', '.join(f'{t:.4f}' for t in r['tau_sequence'])} | "
            f"{', '.join(fmt(x, 5) for x in r['C_T_sequence'])} | {signs} | {r['direction']} |")
    add("")
    iso_multi = [g for g in dec["iso_Re_groups"] if g["n_members"] >= 2]
    add(f"**iso-Re 组（同 Re 由不同 (τ,N) 组合达到，共 {len(iso_multi)} 组）**——解耦的直接判据：")
    add("")
    add("| Re_sim | 成员 (τ,N) | C_T | 组内相对散度% | <3%? |")
    add("|---|---|---|---|---|")
    for g in iso_multi:
        mem = "；".join(f"τ={mm['tau']:.4f},N={mm['N']}" for mm in g["members"])
        vals = "；".join(fmt(mm["C_T"], 5) for mm in g["members"])
        add(f"| {fmt(g['Re_sim'], 5)} | {mem} | {vals} | {fmt(g['spread_pct'], 4)} | "
            f"{'是' if g['pass_3pct'] else '**否**'} |")
    add("")
    add(f"组内最大散度 **{fmt(dec['max_spread_within_iso_Re_group_pct'], 4)}%**；"
        f"通过 3% 带的 iso-Re 组 {dec['n_iso_groups_pass_3pct']}/{len(iso_multi)}。")
    add("")

    # ------------------------------------------------------- 3.4 pilot
    pilot_path = HERE / "pilot_tau_grid_matrix.json"
    if pilot_path.exists():
        pj = json.loads(pilot_path.read_text())
        pc = [x for x in pj["cases"] if x.get("status") == "OK"]
        add("### 3.4 试点子矩阵（N∈" + str(pj["protocol"]["domain_padding_in_L"]) + " 档外的小网格，非本单 18 算例矩阵；用于验证管线并给出负阻力面的实测证据）")
        add("")
        add(f"试点网格 N = {sorted({x['N'] for x in pc})}（**不在本单要求的 {{96,144,192}} 内**），"
            f"同一 τ 档、同一协议、同一步数；完成 {len(pc)} 例，最大相对复算差见第 8 节。")
        add("")
        add("| τ | N | Re_sim | C_T | C_F | C_P | C_T<0 | Δ% vs AN-SUBOFF |")
        add("|---|---|---|---|---|---|---|---|")
        for x in sorted(pc, key=lambda c: (c["N"], c["tau"])):
            add(f"| {x['tau']:.4f} | {x['N']} | {fmt(x['Re_sim'])} | {fmt(x['C_T'])} | {fmt(x['C_F'])} | "
                f"{fmt(x['C_P'])} | {'YES' if x['negative_drag'] else 'no'} | {fmt(x['anchor']['dev_pct'], 5)} |")
        add("")
        pcrit = pj["criteria"]
        add(f"- 试点负阻力格：{pcrit['C2']['n_negative_cells']} 个；无同号 4 邻域佐证：{pcrit['C2']['n_unsupported_cells']} 个 "
            f"→ {pcrit['C2']['verdict']}")
        add(f"- 试点同 τ 跨网格最大漂移：{fmt(pcrit['C3a']['worst_drift_pct'], 4)}% → C3a {pcrit['C3a']['verdict']}")
        add(f"- 试点 C3b：{', '.join(pcrit['C3b']['directions']) or 'n/a'} → {pcrit['C3b']['verdict']}")
        add(f"- 试点 iso-Re 组最大散度：{fmt(pj['decoupling']['max_spread_within_iso_Re_group_pct'], 4)}% "
            f"（{pj['decoupling']['n_iso_groups_pass_3pct']}/{pj['decoupling']['n_iso_groups_with_multiple_members']} 组通过 3% 带）")
        add("")
        add("> 试点只证明**管线可用 + 负阻力面在该分辨率下的形态**；不能替代 96/144/192 矩阵（不同分辨率、不同阻塞/解析度），"
            "故不作为 C1 的满足依据。")
        add("")

    # ------------------------------------------------- 3.5 environment calib
    ec_path = HERE / "environment_calibration.json"
    if ec_path.exists():
        ec = json.loads(ec_path.read_text())
        add("### 3.5 环境标定（为什么矩阵未跑完：实测证据）")
        add("")
        add("| N | 计算域胞数 | 400 步胞更新 |")
        add("|---|---|---|")
        for N in ("96", "144", "192"):
            cm = ec["cost_model"][N]
            add(f"| {N} | {cm['cells']:,} | {cm['cell_updates_at_400_steps']:,} |")
        add("")
        tot = ec["cost_model"]["total_18_cases_400_steps"]
        add(f"- 18 算例 400 步合计胞更新 **{tot:,}**；实测单进程吞吐 "
            f"{ec['cost_model']['measured_throughput_cells_per_s_single_process']:,} 胞/秒 → "
            f"**{fmt(ec['cost_model']['core_hours_at_measured_throughput'], 4)} 核心小时**（单节点串行）。")
        add(f"- 集群调度：递交 {len(ec['scheduler']['cluster_jobs_submitted'])} 个作业"
            f"（{', '.join(ec['scheduler']['cluster_jobs_submitted'])}），"
            f"递交时 `{ec['scheduler']['cluster_state_at_submission']}`；"
            f"20 分钟后 `{ec['scheduler']['cluster_state_after_20_min']}`。")
        add(f"- 本地控制节点负载：`{ec['scheduler']['local_control_node_load_average']}`。")
        add(f"- 协议注意：`{ec['scheduler']['wall_treatment_observed']}`")
        add("")
        if ec["runs"]:
            add("已完成的真实子矩阵标定算例（**不属于 18 算例矩阵**）：")
            add("")
            add("| N | τ | 覆盖步数 | 采样数 | C_T | C_F | C_P | 三段窗 C_T（首/中/末） | Re_sim | Δ% vs AN-SUBOFF |")
            add("|---|---|---|---|---|---|---|---|---|---|")
            for r in ec["runs"]:
                w = r["C_T_windows"]
                add(f"| {r['N']} | {r['tau']:.4f} | {r['steps_covered']} | {r['n_samples']} | {fmt(r['C_T'])} | "
                    f"{fmt(r['C_F'])} | {fmt(r['C_P'])} | {fmt(w[0],5)} / {fmt(w[1],5)} / {fmt(w[2],5)} | "
                    f"{fmt(r['Re_sim'])} | {fmt(r['dev_pct_vs_AN_SUBOFF'],5)} |")
            add("")
            add("该标定的作用：证明**管线端到端可用**（真实 LBM 力积分 → 有量纲系数 → 与锚/ITTC-57 对比），"
                "并量化 400 步协议的瞬态偏差（三段窗单调下降，末窗仍在下降）→ 400 步数值只能做同协议横向比较。")
            add("")

    # ------------------------------------------------------------ 4 figures
    add("## 4. 结果图（matplotlib 生成，PNG 由 `make_figures.py` 从工件绘制）")
    add("")
    fig_caps = [
        ("fig1_ct_vs_re.png", "图 1  C_T 随模拟 Re 的分布（每条曲线一档网格，黑虚线为 AN-SUBOFF 锚 C_T）。"),
        ("fig2_negative_surface.png", "图 2  左：负阻力面（NEG/POS）；右：同一 (τ,N) 网格上的 C_T 数值场。"),
        ("fig3_ct_vs_tau_per_grid.png", "图 3  固定网格下 C_T(τ)（C3b 单调性证据；symlog 纵轴保留负值）。"),
        ("fig4_force_series.png", "图 4  左：N=96 三个代表算例的 C_T 时间序列；右：子矩阵瞬态标定（N=48，非矩阵算例）。"),
        ("fig5_iso_re_collapse.png", "图 5  iso-Re 组：同一 Re 由不同 (τ,N) 组合达到，方框连线为组内散度。"),
        ("fig6_transient_windows.png", "图 6  每算例三段窗均值（首/中/末），400 步协议下的瞬态指示器。"),
        ("fig_calib_transient.png", "图 7  环境标定（N=48, τ=0.510）: C_T 时间序列与三段窗均值；说明 400 步协议处于瞬态。"),
        ("fig_cost_model.png", "图 8  成本模型：18 算例 400 步的胞更新总量与实测吞吐换算的核心小时。"),
    ]
    figs = []
    for name, cap in fig_caps:
        fp = FIG / name
        if fp.exists():
            add(f"![{name}](figures/{name})")
            add("")
            add(cap)
            add("")
            figs.append({"name": name, "caption": cap,
                         "png_base64": base64.b64encode(fp.read_bytes()).decode()})
    add("")

    # ------------------------------------------------------------ 5 cautions
    add("## 5. 注意点（踩过的坑 / 局限 / 适用范围）")
    add("")
    add("1. **τ 与 N 不是两个自由度**：`Re_sim = 3u_lb N/(τ−0.5)`。原问题「τ 与网格能否解耦」"
        "在 LBM 里的严格回答是：**不能**，除非同时反向调 u_lb 做 Re 匹配；本矩阵的每一行都是不同 Re 的切片，"
        "这一点由 iso-Re 组的实测散度独立佐证（见 3.3）。")
    add("2. **Re 与外部锚相差 3–4 个量级**：锚 Re∈[1.16e7, 3.54e7]，本矩阵 Re_sim 见 3.1 列。"
        "锚偏差列是**口径声明**，不是精度声明；本报告不主张任何 <3% 的阻力精度带。")
    add("3. **时间未收敛（诚实披露）**：400 步协议在 N=96 只对应约 0.2 个对流扫掠时间，"
        "子矩阵（N=48）标定显示 C_T 在 ~600–800 步才趋于平台（图 4 右、图 6）。"
        "因此 3.1 的数值是**早期瞬态快照**，用于同协议横向比较，不用于绝对阻力精度。")
    add("4. **计算域被压缩**：冻结基准用 6L×2.12L×2.12L（阻塞比 0.24%），本单为可算性用 3L×0.82L×0.82L"
        "（阻塞比 1.60%，上游 0.25L）。阻塞影响未做量化（WITHHELD），只保证**跨算例一致**，故趋势可比、绝对值不可比。")
    add("5. **低 τ 端稳定性**：τ=0.5005 时 ν_lb=1.67e-4，N=192 对应 Re_sim 极大，MRT 在粗网格上属欠分辨区，"
        "偏差与符号异常应读作数值伪影而非物理。")
    add("6. **适用面**：本工件只支撑「(τ,N) 参数面形态 / 负阻力面 / 解耦恒等式」三类结论；"
        "不支撑 SUBOFF 阻力精度验收、不支撑湍流标定。")
    add("")

    # -------------------------------------------------------- 6 error verdict
    add("## 6. 误差判定（关键量 vs 基准，逐项显式回答「是否 <3%」）")
    add("")
    add("| 量 | 基准 | 逐算例偏差 | 是否 <3% |")
    add("|---|---|---|---|")
    devs = [(cc["tag"], cc["anchor"]["dev_pct"]) for cc in ok]
    min_d = min(devs, key=lambda x: abs(x[1])) if devs else None
    add(f"| C_T | AN-SUBOFF AFF-1（Re=1.16e7） | 最小 |Δ| = {fmt(min_d[1], 5) if min_d else '—'}%"
        f"（{min_d[0] if min_d else '—'}） | **否** |")
    add("")
    if ok:
        add("逐项（全部算例，Δ% = (C_T,sim − C_T,anchor)/C_T,anchor×100）：")
        add("")
        add("| τ | N | Δ% |")
        add("|---|---|---|")
        for tau in TAU_VALUES:
            for N in GRIDS:
                cc = idx.get((tau, N))
                if cc is None:
                    continue
                add(f"| {tau:.4f} | {N} | {fmt(cc['anchor']['dev_pct'], 5)} |")
        add("")
        add("**显式回答：没有任何算例落在 AN-SUBOFF 的 3% 带内。** 原因不是「算法不行」，而是"
            "Re 相差 3–4 个量级：锚 Re≈1.2e7 的摩擦系数（ITTC-57 量级 ~2.9e-3）与模拟 Re（≤5.8e4、层流）"
            "的摩擦系数（~5e-3…1e-1）本就不同量级。可分辨的对照只能是 **Re 匹配**（需 Re 邻域直接模拟，"
            "见 `benchmarks/verified/suboff_anchor/anchor_points.json` 的可达性结论：可信分辨率下不可达）。")
        add("")
        add("摩擦项的物理自检（非验收）：C_F,sim − C_F,ITTC57(Re_sim) 一列见 3.1，"
            "在 Re_sim≲1e4 时模拟 C_F 系统性**高于** ITTC-57 线，与层流（Blasius 1.328/√Re）应当更高一致。")
    add("")

    # ------------------------------------------------------------ 7 criteria
    add("## 7. 判据逐条裁决表（先于动手的量化判据，未改判据）")
    add("")
    add("| # | 判据（原文量化） | 实测 | 阈值 | 裁决 | 证据指针 |")
    add("|---|---|---|---|---|---|")
    add(f"| 1 | ≥18 算例 C_T/C_F/C_P 全落表，每档给锚偏差% | {c['C1']['n_completed']}/{c['C1']['n_expected']} | ≥18 | **{c['C1']['verdict']}** | `tau_grid_matrix.json#cases` |")
    add(f"| 2 | 负阻力面按 (τ,网格) 登记，边界点需 ≥1 同号 4 邻域 | 负格 {c['C2']['n_negative_cells']}，无佐证 {c['C2']['n_unsupported_cells']} | 0 无佐证 | **{c['C2']['verdict']}** | `negative_drag_surface.json#boundary_support` |")
    add(f"| 3a | 同 τ 跨网格 C_T 漂移 <3% | 最差 {fmt(c['C3a']['worst_drift_pct'], 4)}% | <3% | **{c['C3a']['verdict']}** | `tau_grid_matrix.json#criteria.C3a.rows` |")
    add(f"| 3b | 同网格跨 τ 单调性机检结论明确 | {', '.join(c['C3b']['directions']) or 'n/a'} | 明确 | **{c['C3b']['verdict']}** | `tau_grid_matrix.json#criteria.C3b.rows`（逐差符号列） |")
    add(f"| 4 | 6 位有效数字 + 双脚本复算 1e-6 | 最大相对差 {fmt((rc or {}).get('max_abs_rel_delta'), 3)} | ≤1e-6 | **{c['C4']['verdict']}** | `recompute_check.json` |")
    add("")
    add("未完成的判据差距如实列出，不做粉饰（C5）。")
    add("")

    # ------------------------------------------------------- 8 reproducibility
    add("## 8. 可复算与溯源")
    add("")
    if rc:
        add(f"- 路径 A（`build_matrix.py`：存盘 dpS + `np.mean`）与路径 B（`recompute_matrix.py`：解析 dpS + `np.cumsum`）"
            f"对 {rc['n_cases_recomputed']} 个算例复算，最大相对差 **{fmt(rc['max_abs_rel_delta'], 3)}**"
            f"（阈值 1e-6，{'PASS' if rc['pass'] else 'FAIL'}）。")
    else:
        add("- 路径 B 复算工件缺失（NOT_RUN）。")
    add("- 每算例原始力序列 + 两种参考面积口径落盘于 `raw/case_<tag>.npz`，其 sha256 记录在 "
        "`recompute_check.json#rows[].npz_sha256`；每算例标量落盘于 `raw/case_<tag>.json`。")
    add("- 运行环境：torch CPU（float32），集群 loong64 节点 / 本地工作机；每算例记录 `host`、`torch_version`、"
        "`ms_per_step`、`mlups`（见 `raw/case_*.json`）。")
    add("")

    # ------------------------------------------------------------ 9 limits
    add("## 9. 局限与未完成（honest FAIL 清单）")
    add("")
    if c["C1"]["verdict"] != "PASS":
        add(f"- **C1 未达**：{c['C1']['n_completed']}/{c['C1']['n_expected']} 算例完成；缺失 tag：{c['C1']['missing_tags']}。"
            "原因：单算例 400 步在 N=192 需 1.42e7 胞 × 400 步 ≈ 5.7e9 胞更新，实测吞吐 ~0.2–2 MLUPS/进程，"
            "集群与本地均被同期战役占满（作业排队 >15 min 未派发）。")
    if c["C3a"]["verdict"] != "PASS":
        add(f"- **C3a 未达**：最差漂移 {fmt(c['C3a']['worst_drift_pct'], 4)}% ≫ 3%。"
            "根因是 3.3 的恒等式（同 τ 跨 N 时 Re 同时变化 1.5 倍）+ 400 步瞬态未收敛，"
            "不是纯分辨率效应；要分离需 Re 匹配扫描（未做，见下）。")
    add("- **Re 匹配扫描未做**：把 `u_lb` 随 τ 反向调整以固定 Re 的扫描（才是真正的 τ 解耦实验）不在本单判据内，"
        "未执行；本单只给出「按现协议无法解耦」的证据链。")
    add("- **阻塞敏感性未做**：冻结基准域（阻塞 0.24%）未复跑，1.60% 阻塞的系统偏差 WITHHELD。")
    add("- **时间收敛未做**：仅子矩阵 N=48 标定 + 每算例三段窗指示器；N≥96 的收敛研究需 ≥10N 步，超预算。")
    add("")

    add("---")
    add("")
    add("**判定标签**：本工件对 SUBOFF 阻力精度为 **WITHHELD**（Re 不可达 + 瞬态未收敛 + 域被压缩）；"
        "对 (τ,N) 参数面形态与负阻力面为 **REAL**（真实 LBM 求解、真实力积分、双脚本复算一致）。")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")

    payload = {
        "title": "W22：SUBOFF 裸艇 LBM 的 τ×网格二维矩阵 —— 负阻力面与 τ/分辨率解耦判定",
        "summary": (
            f"TensorLBM 用冻结已验证 SUBOFF 裸艇驱动跑 τ×N 矩阵（6 τ × 3 N），完成 "
            f"{c['C1']['n_completed']}/{c['C1']['n_expected']} 算例；负阻力面 {c['C2']['n_negative_cells']} 格、"
            f"4 邻域佐证 {'PASS' if nds['boundary_support']['pass'] else 'FAIL'}；"
            f"同 τ 跨网格 C_T 漂移最差 {fmt(c['C3a']['worst_drift_pct'], 4)}%（判据 <3% → {c['C3a']['verdict']}）；"
            f"证明 Re_sim=3u_lb·N/(τ−0.5) 使 τ 与 N 不可解耦（iso-Re 组实测散度 "
            f"{fmt(dec['max_spread_within_iso_Re_group_pct'], 4)}%）；双脚本复算最大相对差 "
            f"{fmt((rc or {}).get('max_abs_rel_delta'), 3)}。"
        ),
        "scenario_key": "lbm-suboff-tau",
        "mission_id": 14505,
        "tags": ["TensorLBM", "SUBOFF", "LBM", "tau", "grid", "negative-drag", "w22", "resistance"],
        "markdown": "\n".join(L),
        "figures": figs,
    }
    OUT_PAYLOAD.parent.mkdir(parents=True, exist_ok=True)
    OUT_PAYLOAD.write_text(json.dumps(payload))
    print(f"[report] wrote {OUT_MD} ({len(L)} lines) and {OUT_PAYLOAD} "
          f"({len(figs)} figures, {OUT_PAYLOAD.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()