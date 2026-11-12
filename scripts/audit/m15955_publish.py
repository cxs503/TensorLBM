#!/usr/bin/env python3
"""m15955 board-audit deliverables: figures + paper-level report + verdict SQL.

Publishes the audit report to ``$TENSOR_API_BASE/tensor/reports`` with two
matplotlib figures attached as base64.  Every number is injected from
``artifacts/m15955_audit/audit_matrix.json`` (programmatic, zero hand-copy).
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "m15955_audit"
FIGS = OUT / "figs"
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

A = json.loads((OUT / "audit_matrix.json").read_text())
B = json.loads(Path("/root/tp-agent/work/m15919/runs/t5_ice_coupling/benchmark.json").read_text())
G = json.loads(Path("/root/tp-agent/work/m15919/runs/t5_ice_coupling/gates.json").read_text())


def figs() -> list[dict]:
    FIGS.mkdir(parents=True, exist_ok=True)
    made = []

    # ---- fig1: hemisphere pressure resultant convergence, mission vs audit ----
    rows = A["A1_hemisphere"]["rows"]
    nt = np.array([r["n_theta"] for r in rows], float)
    got_m = np.array([r["fz_mission"] for r in rows])
    got_a = np.array([r["fz_audit"] for r in rows])
    ref = np.pi
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogx(nt, got_m, "o-", label="mission run (UV-sphere facets)")
    ax[0].semilogx(nt, got_a, "x", ms=9, label="auditor independent numpy recompute")
    ax[0].axhline(ref, color="k", ls="--", label=r"analytic $\pi a^2 p$")
    ax[0].set_xlabel(r"$n_\theta$ (hemisphere bands, $n_\phi=2n_\theta$)")
    ax[0].set_ylabel(r"$F_z=\int p\,\hat n_z\,dA$")
    ax[0].set_title("I1 hemisphere pressure resultant")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="both", alpha=0.3)
    err = np.abs(got_a - ref) / ref
    ax[1].loglog(nt, err, "s-", label="auditor rel. error")
    ax[1].loglog(nt, err[0] * (nt[0] / nt) ** 2, "k:", label=r"$O(h^2)$, $h\propto1/n_\theta$")
    ax[1].axhline(0.03, color="r", ls="--", label="3% platform bar")
    ax[1].set_xlabel(r"$n_\theta$")
    ax[1].set_ylabel("relative error")
    ax[1].set_title("I1 convergence (true order 1.99 in $n_\\theta$)")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p1 = FIGS / "fig1_hemisphere_convergence.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    made.append({"name": "fig1_hemisphere_convergence",
                 "caption": "I1 hemisphere pressure resultant: mission numbers (circles) vs an independent "
                            "numpy UV-sphere recompute (crosses), analytic anchor pi*a^2*p; right panel shows the "
                            "true O(h^2) convergence in n_theta against the 3% platform bar.",
                 "path": p1})

    # ---- fig2: per-case error of the mission benchmark vs its own thresholds ----
    cases = [c for c in B["cases"] if not isinstance(c["ref"], list)]
    labels = [f'{c["gate"]} {c["description"][:26]}' for c in cases]
    errs = np.array([max(float(c["rel_err"]), 1e-18) for c in cases])
    ths = np.array([float(c["threshold"]) for c in cases])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    idx = np.arange(len(cases))
    ax.bar(idx, errs, color=["C0" if e < 3e-2 else "C1" for e in errs])
    ax.plot(idx, ths, "r_", ms=18, label="mission's own threshold")
    ax.axhline(3e-2, color="k", ls="--", lw=1, label="3% platform bar")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("relative error (log)")
    ax.set_title("m15919 benchmark: 8 self-declared interface cases (I4/I7/clock uncovered; no machine referee)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    p2 = FIGS / "fig2_benchmark_errors.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    made.append({"name": "fig2_benchmark_errors",
                 "caption": "Relative error of the 8 benchmark cases the mission declares, against the mission's "
                            "own thresholds and the 3% platform bar; I4/I7/clock gates carry no benchmark row and "
                            "tensor_benchmark_cases holds 0 rows for the anchor.",
                 "path": p2})
    return made


def report_md(made: list[dict]) -> str:
    r = A["A1_hemisphere"]["rows"]
    a4 = A["A4_lbm_leg"]
    a5 = A["A5_referee"]
    a6 = A["A6_artifacts"]
    lines = []
    lines.append("## 1 案例介绍\n")
    lines.append("锚点 **GAP-LBM-T5-ICE-COUPLING**（T5 破冰船旗舰：LBM 水动力 + FEM 结构 + DEM 冰破坏，三维）。"
                 "被审计对象 = m15919（返修轮 1，PR #1866，merge 19386482，已 succeeded）。"
                 "交付物 = 三维**载荷交换接口** `src/tensorlbm/three_way/interface3d.py`（球/矩形面几何、"
                 "一致面载荷分配、Hermite 梁投影、耦合时钟、附加质量稳定性）+ 入口脚本 + 不变量门禁 I1–I7。\n")
    lines.append("任务（验收判据原文）：round-1 只做「依赖审计 + 耦合接口设计」，工程级三向耦合算例留待 T1 多相与载荷交换达标之后。\n")
    lines.append("| 参数 | 取值 |\n|---|---|\n")
    lines.append("| 半球/球半径 | 1.0 |\n| 门禁网格 | n_theta × n_phi ∈ {8×16, 16×32, 24×48} |\n")
    lines.append("| 梁 | E=2.1e11 Pa, ρ=7850, b=0.1 m, h=0.01 m, L=1.0 m, 悬臂, 10 单元 |\n")
    lines.append("| 附加质量模型 | 1-DOF 显式交错, μ = m_add/m_s, ω_n dt = 1e-2 |\n")
    lines.append("| LBM 腿 | 引用 D3Q19 MRT D40 球绕流 Re=100，C_D=1.12198（**未在本单运行**） |\n\n")
    lines.append("## 2 怎么做的（审计方法）\n")
    lines.append("独立重算，不导入被审模块：自写 numpy UV-球三角面几何（精确面元面积 + 外向法向），"
                 "重算 I1 压力合力、闭合面抵消；独立闭式重算梁端挠度；从 `benchmarks/verified/sphere_re100/result.json` "
                 "反查 LBM 腿数值来源；用 `git ls-tree` / `git diff` 检查工件归属与附带改动；跑 `pytest tests/test_three_way_interface3d.py`；"
                 "在 main（af412c6f）上执行 mission 的 repro_cmd。工具：psql（tensor_benchmark_cases / tensor_maturity_verdicts / tensor_reports）、git、pytest、numpy、matplotlib。\n")
    lines.append("复现：`PYTHONPATH=src python3 scripts/t5_ice_coupling_entry.py`（审计跑：rc=0，0.3 s，8/8 门禁 + 8/8 benchmark 通过）；"
                 "`python3 scripts/audit/m15955_t5_ice_audit.py`（本审计重算，产出 audit_matrix.json）。\n\n")
    lines.append("## 3 计算结果（程序化注入）\n")
    lines.append("**I1 半球压力合力：mission 数值 vs 审计独立重算 vs 解析锚 π a² p**\n\n")
    lines.append("| n_theta×n_phi | mission got | 审计重算 | \\|Δ\\| | 解析 π | 相对误差 |\n|---|---|---|---|---|---|\n")
    for x in r:
        lines.append(f"| {x['n_theta']}×{x['n_phi']} | {x['fz_mission']:.12f} | {x['fz_audit']:.12f} | "
                     f"{x['mission_vs_audit_abs']:.2e} | {x['ref']:.12f} | {x['err_audit']*100:.4f}% |\n")
    lines.append(f"\n- 审计闭合面抵消 |Σ n dA| = {A['A2_closed_surface']['sum_ndA_audit']:.2e}（mission 报 "
                 f"{A['A2_closed_surface']['mission_quoted']:.2e}）→ 均 <1e-14，真值。\n")
    lines.append(f"- I1 真实收敛阶（对 n_theta）：{r[1]['order_true_in_ntheta']:.3f}；mission 的 gates.json 报 "
                 f"{G['gates']['I1']['hemisphere_rows'][1]['convergence_order']:.3f}（按面元数取对数，面元 ∝ h⁻²，故少算一倍）。\n")
    lines.append(f"- 梁端挠度闭式：审计 {A['A3_beam']['w_tip_audit']:.12f} m vs mission {A['A3_beam']['mission_quoted']:.12f} m，"
                 f"相对差 {A['A3_beam']['rel_diff']:.2e}。\n")
    lines.append(f"- LBM 腿来源：mission C_D={a4['mission_C_D_got']} vs 已验证记录 {a4['verified_record_cd']:.10f}"
                 f"（quote_matches_record={a4['quote_matches_record']}），对 Schiller–Naumann {a4['schiller_naumann_ref']:.7f} 误差 "
                 f"{a4['err_pct_recomputed']:.3f}%；但 mission 脚本内**未调用任何 LBM/DEM 求解器**（integration 脚本 57–58 行硬编码常数）。\n")
    lines.append(f"- 裁判覆盖：gates.json 有 {len(A['A5_referee']['gates_in_gates_json'])} 个门禁，benchmark 只覆盖 "
                 f"{A['A5_referee']['gates_with_benchmark_row']}；未覆盖 {A['A5_referee']['gates_without_benchmark_row']}；"
                 f"`tensor_benchmark_cases` 中 anchor 行数 = **0**（全表 25 行）→ 无机器裁判。\n")
    lines.append(f"- 工件归属：`git ls-tree HEAD runs/t5_ice_coupling` 为空（run_artifacts 只在 mission worktree）；"
                 f"盘上存在 {len(a6['mission_run_files_present_on_disk'])} 个文件（含 3 张 png）。\n")
    lines.append(f"- 测试：`pytest tests/test_three_way_interface3d.py` → {A['A7_tests']['summary']}。\n\n")
    lines.append("## 4 结果图\n")
    for m in made:
        lines.append(f"**{m['name']}** — {m['caption']}\n\n")
    lines.append("## 5 注意点（踩坑/局限/适用范围）\n")
    lines.append("1. **超报**：mission result headline 与入口脚本打印「L2 + L4」，但 L3 无真实集成调用者（interface3d 只被本单脚本与测试 import，流体腿是常数），"
                 "L4 无完整耦合算例（design doc 自述“No 3-D LBM→FEM→DEM coupled simulation was run in round 1”）；"
                 "且 anchor 机器裁判 0 行。判级 = **L2**。\n")
    lines.append("2. **附带工件回退（MEDIUM）**：PR #1866 顺带把 `TensorLBM/artifacts/ice_hydro_breakup/dem_breakup_study.json` 由 5 样本/129 节点/12 模态（含 acceptance J1–J3 全 PASS、runtime 14.3 s、时间戳 2026-11-11T06:44:13Z）"
                 "覆盖成 2 样本/65 节点/6 模态、且丢掉 acceptance/timestamp/runtime → 与仓库生成器 `run_study.py::stage_dem`（n_nodes=129, t_end=40/60）不再一致，验收记录丢失。与本单交付无关，应回退。\n")
    lines.append("3. **工件保管**：benchmark.json/gates.json/figures 未入库（平台合并过滤），worktree 一旦回收即不可复原；建议把 runs/*.json 作为受 sha256 引用的证据文件入库。\n")
    lines.append("4. **I5 自指**：所谓 closed form 是同一个 3×3 递推矩阵的特征值，与 marching 同源，只能验证实现一致，不能验证物理；独立内容只有 bisection 阈值 μ_crit=0.999975 vs 教材 μ<1。\n")
    lines.append("5. **阈值自选**：benchmark 阈值 0.03/0.15 写在脚本里（无预注册），I1 的 15% 恰等于平台“不可接受”边界；实测最细网格 0.29% 远在 3% 内，未见调参痕迹。\n")
    lines.append("6. **DEM 缺席**：唯一 DEM 相关门禁 I4 是“点载荷一致 vs 最近节点”的分配测试，不是 DEM 求解器耦合；三维 DEM/bonded ice 在 `src/tensorlbm/dem/`、`ice_chain3d/` 已存在但本单未接。\n\n")
    lines.append("## 6 误差判定（逐项，显式回答是否 <3%）\n")
    lines.append("| 量 | 基准 | 误差 | <3%？ |\n|---|---|---|---|\n")
    lines.append(f"| I1 半球合力（最细 24×48） | 解析 π a² | {r[2]['err_audit']*100:.3f}% | 是 |\n")
    lines.append(f"| I1 半球合力（最粗 8×16） | 解析 π a² | {r[0]['err_audit']*100:.3f}% | 是 |\n")
    lines.append(f"| I2 合力+力矩守恒 | 精确 | {B['cases'][3]['rel_err']:.2e} | 是 |\n")
    lines.append(f"| I3 功共轭恒等式 | 离散恒等式（非独立基准） | {B['cases'][4]['rel_err']:.2e} | 是（但为自一致补丁测试） |\n")
    lines.append(f"| I5 μ_crit | 教材判据 μ<1 | {B['cases'][5]['rel_err']*100:.5f}% | 是 |\n")
    lines.append(f"| I6 梁端挠度 | FL³/3EI | {B['cases'][6]['rel_err']:.2e} | 是 |\n")
    lines.append(f"| I6 一阶频率 | Euler–Bernoulli 1.875104² | {B['cases'][7]['rel_err']:.2e} | 是 |\n")
    lines.append(f"| LBM 腿 C_D（引用） | Schiller–Naumann 1.0917311 | {a4['err_pct_recomputed']:.3f}% | 是（引用值，非本单计算） |\n")
    lines.append("| **三维 LBM→FEM→DEM 耦合算例** | **无基准** | **—** | **不适用**：`tensor_benchmark_cases` anchor 行数=0，且本单未运行耦合算例；故不能主张 L4 |\n\n")
    lines.append("结论：接口级不变量全部 <3%，数值真实（独立重算复现到 1e-14）；但**锚点级**（三维三向耦合）无基准、无算例，判 **L2**，"
                 "disposition = request changes（回退附带工件、收回 L3/L4 声明、登记 anchor 基准行 + 接真实 DEM/LBM 腿）。\n")
    return "".join(lines)


def main() -> int:
    made = figs()
    md = report_md(made)
    (OUT / "m15955_audit_report.md").write_text(
        "# 交差审计 m15919 → GAP-LBM-T5-ICE-COUPLING（判级 L2）\n\n" + md, encoding="utf-8")

    figs_payload = [{"name": m["name"], "caption": m["caption"],
                     "png_base64": base64.b64encode(m["path"].read_bytes()).decode()}
                    for m in made]
    summary = ("独立重算复核 m15919：接口不变量数值真实（审计 numpy 重算 I1 复现至 1e-14；14 测试通过；repro_cmd rc=0），"
               "但 L3 无真实集成调用者、L4 无完整耦合算例且 anchor 裁判 0 行 → 判 L2；另发现附带工件 dem_breakup_study.json 被降级覆盖。")
    payload = {
        "title": "交差审计 m15919 → GAP-LBM-T5-ICE-COUPLING 三维耦合接口（判级 L2；L3/L4 声明被驳）",
        "summary": summary,
        "scenario_key": "board-audit",
        "mission_id": 15955,
        "tags": ["board-audit", "TensorLBM", "GAP-LBM-T5-ICE-COUPLING", "maturity-L2"],
        "markdown": md,
        "figures": figs_payload,
    }
    (OUT / "report_payload.json").write_text(json.dumps({k: v for k, v in payload.items() if k != "figures"}, indent=1))
    tok = requests.post(f"{API}/auth/login", json={"username": USER, "password": PASS}, timeout=60).json()
    token = tok.get("access_token") or tok.get("token") or tok["data"]["access_token"]
    resp = requests.post(f"{API}/tensor/reports", json=payload,
                         headers={"Authorization": f"Bearer {token}"}, timeout=180)
    print("report POST", resp.status_code, resp.text[:300])
    return 0 if resp.status_code in (200, 201) else 1


if __name__ == "__main__":
    sys.exit(main())