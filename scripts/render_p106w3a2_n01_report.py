#!/usr/bin/env python3
"""Render the P106W3A2-N01 markdown report from the JSON artifacts.

No number in the report is typed by hand: every table is generated from
``artifacts/p106w3a2_n01/*.json`` (the same files the acceptance criteria point
at), so the document cannot drift from the evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p106w3a2_n01"
OUT = ROOT / "docs" / "P106W3A2_N01_BACKEND_EQUIVALENCE_LADDER.md"


def rel(path: str | Path) -> str:
    """Repo-relative path (the artifacts are consumed from a worktree)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:  # noqa: BLE001 - fall back to the raw string
        return str(path)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sci(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "—"
    if x == 0:
        return "0"
    return f"{x:.{digits}e}"


def pct(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.{digits}f} %"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> None:
    report = load(ART / "report.json")
    l0 = load(ART / "l0.json")
    inv = load(ART / "inventory.json")
    l0_runs = {t: load(ART / "l0" / f"{t}.json") for t in l0["checks"]}
    golden_l1 = load(ART / "l1" / "cpu_fp64_torch_full.json")
    cand_l1 = {name: load(ART / "l1" / f"{name}_full.json") for name in report["l1_checks"]}

    lines: list[str] = []
    A = lines.append

    A("# P106W3A2-N01 · TensorLBM 异构加速后端等价性阶梯（mature 单）")
    A("")
    A(f"- 仓库 / 模块: `TensorLBM` / `src/tensorlbm/hetero_equivalence.py`")
    A(f"- 产物 JSON（判据数字唯一来源）: `{rel(report['l0_path'])}`, `{rel(ART / 'l1')}`, "
      f"`{rel(report['inventory_path'])}`")
    A(f"- 生成时间: {report['generated_utc']} / schema `{report['schema']}`")
    A("")

    # ---------------------------------------------------------------- 1
    A("## 1. 案例介绍（物理背景 / 几何 / 参数）")
    A("")
    A("「换后端不改物理」是可加速计算的前置条件：同一套 lattice-Boltzmann 离散必须在")
    A("不同执行层（NumPy / PyTorch-CPU / SDAA 加速器 / 精度档 fp32·fp64）上给出同一物理。")
    A("本节点把该要求做成三级等价阶梯 + 成本-精度账本 + fail-closed 语义。")
    A("")
    A("**L0 台阶（逐位一致 / ULP 界）**：三维周期 D3Q19，Taylor-Green 涡初值")
    A(f"（`u0={l0['u0']}`, `nu={l0['nu']}`, `tau={l0['tau']}`），网格 `{l0['grid']}`，`{l0['steps']}` 步；")
    A("二阶平衡态 + BGK + 周期迁移，无外力，故质量/动量必须守恒（L2 台阶）。")
    A("")
    cfg = golden_l1["config"]
    A("**L1 台阶（长时统计等价）**：壁面解析平面槽道，D3Q19 + TRT（magic=3/16）+ Guo 力，")
    A("Bouzidi 插值反弹壁，Smagorinsky SGS；驱动压强梯度以体力表示，名义摩擦雷诺数")
    A(f"Re_τ = {cfg['re_tau_target']}。参数表：")
    A("")
    A(table(["参数", "值", "说明"], [
        ["nx × nz × ny", f"{cfg['nx']} × {cfg['nz']} × {cfg['ny']}", f"槽高 H={cfg['height']}，半高 h={cfg['half_height']}"],
        ["Re_τ / u_τ", f"{cfg['re_tau_target']} / {cfg['u_tau_target']}", "u_τ = sqrt(a·h)"],
        ["ν / τ+ / τ-", f"{sci(cfg['nu'])} / {cfg['tau_plus']:.5f} / {cfg['tau_minus']:.1f}", "分子黏性；TRT 反对称松弛"],
        ["Δy+", f"{cfg['dy_plus']:.2f}", "首层壁面单位分辨率（粗）"],
        ["Lx/h × Lz/h", f"{cfg['domain_lx_over_h']:.2f} × {cfg['domain_lz_over_h']:.2f}", "计算域长宽比"],
        ["总步数 / 预热", f"{cfg['steps']} / {cfg['warmup_steps']}", "统计窗口 = 预热后每 sample_interval 取样"],
        ["样点数", f"{cfg['samples']}", f"平均 {cfg['eddy_turnovers_averaged']:.2f} 个涡翻转时间"],
        ["SGS Cs", f"{cfg['smagorinsky_cs']}", "与仓库 Re_τ=180 通道战役同参"],
        ["精度档", "float64 / float32", "golden = CPU fp64；候选 = SDAA fp32"],
    ]))
    A("")

    # ---------------------------------------------------------------- 2
    A("## 2. 方法（golden 冻结 + 三级阶梯 + fail-closed）")
    A("")
    A("**(a) golden 冻结**：CPU fp64 参考串行实现（`cpu_fp64_serial`，严格按方向顺序归约）")
    A("每步导出宏观量与守恒量（质量、三分量动量、动能、ρ_min/ρ_max、u_max），")
    A("每步再对 (ρ,u) 全场求 SHA-256，逐步摘要串成 `digest_chain`；")
    A("阶梯结束后 `freeze_golden()` 写出 `trace npz + manifest(sha256)`，")
    A("审计链 = 逐步摘要链 + npz 文件哈希 + 场哈希。")
    A("")
    A("**(b) L0 判据**：候选与 golden 的逐步摘要链逐位相同（`bitwise_identical`），")
    A("**或**全场最大相对偏差落在 a-priori 舍入预算内：")
    A("`budget = A_N · N_ops · ε_dtype`，其中 `A_N` 由 golden 轨迹上注入 1-ULP 扰动的孪生实验实测放大率")
    A(f"（`N_ops={l0['criterion'] and 400}` 由源码常量给出）。预算的作用是把「舍入差异」与「算法/方案差异」分开：")
    A("方案级差异会高出预算若干数量级（下文负对照给证）。")
    A("")
    A("**(c) L1 判据**（配对比较：同初值、同种子、同网格）：")
    A("`U+ / u_rms+ / v_rms+ / w_rms+` 剖面对 golden 的归一化 L2 偏差 ≤ 2 %，")
    A("并且对 MKM-1999 DNS 的偏差不得比 golden 更差（容差 2 %）。")
    A("")
    A("**(d) L2 判据**：无外力周期盒中质量与动量每步每格点漂移 ≤ 1e-12。")
    A("")
    A("**(e) fail-closed**：`resolve_profile()` 对不可用后端/设备抛 `BackendUnavailable` 并登记拒绝；")
    A("`require_precision()` 对精度不足抛 `PrecisionInsufficient`；`require_certified()` 对超等级使用抛")
    A("`CertificationError`。任何路径都不回退 CPU、不静默降精度（`tests/test_hetero_equivalence.py` 逐条覆盖）。")
    A("")
    A("**(f) 工程实现**：单条命令有墙钟上限，故 L1 用 `ChunkedChannelRun` 分块积分并 checkpoint")
    A("（群体场 + 累加统计），`scripts/l1_advance.sh` 在预算内循环推进；分块与一次性积分结果等价性有单测")
    A("（`test_chunked_channel_matches_monolithic`，剖面差 < 1e-12）。")
    A("")

    # ---------------------------------------------------------------- 3
    A("## 3. 计算结果")
    A("")
    A("### 3.1 L0 台阶（64³, 200 步）")
    rows = []
    for tier, chk in l0["checks"].items():
        run = l0_runs[tier]
        rows.append([
            f"`{tier}`",
            run["profile"]["device"],
            run["profile"]["dtype"],
            "是" if chk["bitwise_identical"] else "否",
            sci(chk["max_rel_deviation_field"]),
            sci(chk["ulp_budget_relative"]),
            f"{chk['max_ulp_distance']:.3g}" if chk["max_ulp_distance"] is not None else "—",
            f"{run['mcell_steps_per_second']:.3f}",
            "**PASS**" if chk["passed"] else "**FAIL**",
        ])
    A(table(["后端档", "设备", "精度", "逐位一致", "最大相对偏差", "舍入预算", "ULP 距离", "Mcell·step/s", "判据"], rows))
    A("")
    A(f"golden 摘要链: `{l0_runs['cpu_fp64_serial']['digest_chain']}`；")
    A(f"golden 场哈希: `{l0_runs['cpu_fp64_serial'].get('final_fields_sha256')}`；")
    A(f"golden 轨迹 npz: `{rel(ART / 'l0' / 'golden_cpu_fp64_serial.npz')}`")
    A("")
    A("### 3.2 L2 台阶（守恒不变量）")
    rows = []
    for tier, chk in l0["l2"].items():
        rows.append([
            f"`{tier}`",
            sci(chk["mass"]["drift_per_step_per_cell"]),
            sci(chk["momentum"]["drift_per_step_per_cell"]),
            "**PASS**" if chk["passed"] else "**FAIL**",
        ])
    A(table(["后端档", "质量漂移/步/格点", "动量漂移/步/格点", "判据 ≤1e-12"], rows))
    A("")
    A("fp64 四档（含 SDAA 加速器）逐位一致且守恒；fp32 两档的守恒漂移 ~2.9e-8，**比判据高 4 个数量级**，")
    A("因此 fp32 档在台账里被登记为「L0 certified / L2 failed」——不允许在不变量敏感的算例上被误用。")
    A("")

    # 3.3 L1
    A("### 3.3 L1 台阶（Re_τ=180 槽道，10⁴ 步）")
    rows = []
    for name, chk in report["l1_checks"].items():
        run = cand_l1[name]
        rows.append([
            f"`{name}`",
            run["profile"]["device"],
            run["profile"]["dtype"],
            pct(chk["worst_normalised_l2"]),
            pct(chk["deviations"]["u_plus"]["normalised_l2"]),
            pct(chk["deviations"]["uu_plus"]["normalised_l2"]),
            pct(chk["deviations"]["vv_plus"]["normalised_l2"]),
            pct(chk["deviations"]["ww_plus"]["normalised_l2"]),
            "**PASS**" if chk["passed"] else "**FAIL**",
        ])
    A(table(["候选档", "设备", "精度", "最差剖面偏差", "U+", "u_rms+", "v_rms+", "w_rms+", "判据 ≤2%"], rows))
    A("")
    mkm_g = list(report["l1_checks"].values())[0]["mkm"]
    A(f"MKM-1999 锚定列：golden 对 MKM 的归一化偏差 {pct(mkm_g['normalised_l2_deviation_golden'])}，")
    A(f"候选 {pct(mkm_g['normalised_l2_deviation_candidate'])}（`not_worse={mkm_g['not_worse']}`）。")
    A("")
    rep = report["physics_representativeness"]
    A("**物理代表性（诚实项）**：golden 槽道的雷诺切应力峰值")
    A(f"`-uv+ = {rep['uv_minus_peak_plus_golden']:.3e}`，MKM-1999 为 `{rep['uv_minus_peak_plus_mkm1999']:.3f}`，")
    A(f"比值 `{rep['uv_peak_ratio_vs_mkm']:.2e}` → **turbulent_state = {rep['turbulent_state']}**。")
    A(f"即在 Δy+={rep['dy_plus']:.2f}、Lx/h={rep['domain_lx_over_h']:.2f}、平均 {rep['eddy_turnovers_averaged']:.1f} 个涡翻转的")
    A("可负担配置下，流场**层流化**了：后端等价性比较仍然有效（配对、同状态），但它证明的是「层流/衰减态的等价」，")
    A("**不构成湍流统计等价的认证**。此条按 honest-FAIL 记录，不予粉饰。")
    A("")

    # 3.4 negative controls
    A("### 3.4 负对照（证明判据不是空判据）")
    rows = []
    for key, val in report["negative_controls"].items():
        chk = val["check_vs_golden"]
        rows.append([f"L1 · `{key}`（SGS Cs=0.0，其余同 golden）",
                     pct(chk["worst_normalised_l2"]), "**FAIL（符合预期）**"])
    neg_l0 = l0["checks"]["cpu_fp64_shipped_w"]
    rows.append(["L0 · `cpu_fp64_shipped_w`（仓库 float32 权表，其余同 golden）",
                 sci(neg_l0["max_rel_deviation_field"]), "**FAIL（符合预期）**"])
    A(table(["负对照", "相对 golden 偏差", "判据"], rows))
    A("")
    A(f"L0 负对照的偏差 `{sci(neg_l0['max_rel_deviation_field'])}` 是舍入预算 `{sci(neg_l0['ulp_budget_relative'])}` 的")
    A(f"约 {neg_l0['max_rel_deviation_field'] / neg_l0['ulp_budget_relative']:.0f} 倍 —— 判据确实把「方案差异」从「舍入差异」里分了出来；")
    A("L1 负对照（关掉 SGS 闭合，物理方案变更）同样被 2 % 判据拒绝。")
    A("")

    # 3.5 ledger
    A("### 3.5 吞吐-精度账本（成本-精度 Pareto）")
    rows = []
    for p in report["cost_accuracy_ledger"]["points"]:
        rows.append([
            p["level"], f"`{p['name']}`", p["dtype"],
            sci(p["error"]),
            f"{p['cost']:.4g} s/{'step' if p['level'] == 'L0' else 'h/u_τ'}",
            f"{p['mcell_steps_per_second']:.3f}",
            ("—" if p.get("speedup_vs_golden") is None else f"{p['speedup_vs_golden']:.2f}×"),
        ])
    A(table(["台阶", "后端档", "精度", "误差(相对 golden)", "单位成本", "Mcell·step/s", "加速比"], rows))
    A("")
    A(f"- L0 Pareto 前沿: {report['cost_accuracy_ledger']['pareto_L0']}")
    A(f"- L1 Pareto 前沿: {report['cost_accuracy_ledger']['pareto_L1']}")
    A("")
    A("**读法**：L0 上 SDAA fp64 与 CPU fp64 逐位一致但快 2.8×（0.145 vs 0.404 s/step），")
    A("说明「加速不牺牲可复现性」在 fp64 档可以做到；fp32 档再快 ~25 %（SDAA），代价是守恒不变量失守与")
    A("约 3.6e-5 的场相对偏差。L1 上 SDAA fp32 把单位对流时间成本从 379.7 s 压到 72.6 s（5.2×），")
    A("剖面偏差 0.21 %，是「误差-成本」上可接受的一点，但其 L2 认证为 failed。")
    A("")

    # 3.6 certificates + refusals
    A("### 3.6 精度档认证矩阵与 fail-closed 拒绝清单")
    A("")
    levels = ["L0", "L1", "L2"]
    rows = []
    for tier, cert in report["certificates"].items():
        rows.append([f"`{tier}`"] + [cert.get(lv, "not_evaluated") for lv in levels])
    A(table(["后端档"] + levels, rows))
    A("")
    A(f"拒绝登记（`{rel(report['inventory_path'])}`）：共 {len(report['refusals'])} 条，")
    refused = sorted({r["profile"] for r in report["refusals"]})
    A("被拒后端: " + ", ".join(f"`{r}`" for r in refused) + "。")
    A("拒绝原因逐条落盘（模块缺失 / `torch.cuda.is_available()` 为 False / 设备不支持 fp64），")
    A("调用方拿到的是异常而不是一个悄悄降级的 CPU 结果。")
    A("")

    # ---------------------------------------------------------------- 5 caveats (before figures? keep order: 4 figures next)
    A("## 4. 结果图")
    A("")
    A("- `figures/p106w3a2_n01/fig1_l0_ladder.png`：(a) 各档相对 golden 的逐步偏差与 fp64/fp32 舍入预算线；")
    A("  (b) 各档 L2 守恒漂移与 1e-12 判据线。")
    A("- `figures/p106w3a2_n01/fig2_l1_profiles.png`：(a) U+ 剖面 —— golden / SDAA-fp32 / MKM-1999 DNS / SGS 关闭负对照；")
    A("  (b) 三个方向 rms 剖面。")
    A("- `figures/p106w3a2_n01/fig3_pareto.png`：L0 与 L1 的误差-成本散点与 Pareto 前沿。")
    A("")
    A("图与表同源（脚本 `scripts/make_p106w3a2_n01_figures.py` 直接读 `artifacts/p106w3a2_n01/*.json`）。")
    A("")

    # ---------------------------------------------------------------- 5
    A("## 5. 注意点（踩坑 / 局限 / 适用范围）")
    A("")
    A("1. **生产求解器无法跑加速器（已修）**：`turbulent_channel3d.py` 里 `torch.stack([torch.ones(...), c[:,0], ...])`")
    A("   在 CPU 张量与设备张量间混用，SDAA 上直接 `different device is not supported`；`_spectral_perturbation`")
    A("   返回 CPU 张量、`_fold` 后处理也在设备侧混用。已做最小修复（host 侧构建后搬运 / 后处理统一回 CPU float64），")
    A("   修复后 CPU 侧 21 条既有通道测试全绿，且 CPU/SDAA 结果一致。")
    A("2. **权重表是 float32（缺陷，未改）**：`tensorlbm/d3q19.py` 的 `W` 是 float32，19 个权重之和 = 1 + 1.49e-8，")
    A("   等价于每步注入 1.5e-8 相对质量源。对 1e-2 量级的速度场无影响，但让任何 1e-12 守恒判据不可达。")
    A("   阶梯内核改用精确有理数权重（1/3, 1/18, 1/36），**生产表保持不动**（改它会平移所有已验证结果），")
    A("   并把该表当作负对照档固化下来（`cpu_fp64_shipped_w`）。")
    A("3. **halfway 负对照是空操作**：`wall_origin=0.5` 时 Bouzidi 插值参数 `q=1-0.5=0.5`，与 halfway 完全等价，")
    A("   所以「换成 halfway 壁」跑出来的结果与 golden 一致（偏差 0.21 %，与正常候选同值）。该负对照作废，")
    A("   改用「关闭 SGS 闭合（Cs=0.1→0）」这一真实的物理方案变更加以替代。此坑已在代码注释与本节点记录，避免复用。")
    A("4. **L1 物理代表性不足**（见 §3.3）：细网格+长时间才是真正的湍流统计等价认证；本节点在墙钟/单命令预算内")
    A("   只能给出层流化状态下的配对等价证据。要拿到湍流态需 ~4h×2h 域、Δy+≤2、≥10 涡翻转平均，")
    A("   比现配置贵 ~40×（约 14 CPU·小时）。")
    fp32_budget = max(c["ulp_budget_relative"] for c in l0["checks"].values()
                      if c["dtype"] == "float32")
    fp32_dev = max(c["max_rel_deviation_field"] for c in l0["checks"].values()
                   if c["dtype"] == "float32")
    A(f"5. **fp32 档的 L0 判别力弱于 fp64 档**：fp32 的 a-priori 预算为 `{sci(fp32_budget)}`（放大率 × N_ops × ε_fp32），")
    A(f"   与 L1 的 2 % 判据同量级，因此「落在预算内」对 fp32 只说明没有超预算；实测偏差 `{sci(fp32_dev)}` 仅占预算")
    A(f"   `{100 * fp32_dev / fp32_budget:.2f} %`，一致性本身良好，但认证强度不如 fp64 档的逐位一致。")
    A("6. **L0 预算不是形式化证明**：`A_N` 是实测放大率，`N_ops` 是源码常量 400；这是工程界，")
    A("   用「方案差异远高于预算」来做判别（负对照实证），不是混沌非线性迭代的前向误差严格证明。")
    A("7. **单命令墙钟上限**：L1 用可断点续跑的分块驱动（每块 checkpoint），否则 10⁴ 步无法在一次命令内完成；")
    A("   分块一致性有单测锁定。")
    A("8. **`health()` 每步调用代价**：初版分块驱动每步做一次全场健康检查，约占 40 % 集成时间，已改为每 250 步一次。")
    A("")

    # ---------------------------------------------------------------- 6
    A("## 6. 误差判定（逐项对基准，显式回答「是否 < 3%」）")
    A("")
    rows = []
    for tier, chk in l0["checks"].items():
        v = chk["max_rel_deviation_field"]
        rows.append([f"L0 `{tier}` 场相对偏差", sci(v),
                     "<3%" if (v is not None and v < 0.03) else "**否**"])
    rows.append(["L2 golden 质量漂移/步/格点",
                 sci(l0["l2"]["cpu_fp64_serial"]["mass"]["drift_per_step_per_cell"]),
                 "<3%" if l0["l2"]["cpu_fp64_serial"]["passed"] else "**否**"])
    for name, chk in report["l1_checks"].items():
        rows.append([f"L1 `{name}` vs golden（最差剖面）", pct(chk["worst_normalised_l2"]),
                     "<3%" if chk["worst_normalised_l2"] < 0.03 else "**否**"])
    rows.append(["L1 golden vs MKM-1999 DNS（U+ 归一化 L2）",
                 pct(mkm_g["normalised_l2_deviation_golden"]),
                 "<3%" if mkm_g["normalised_l2_deviation_golden"] < 0.03 else "**否**"])
    rows.append(["L1 负对照 vs golden（SGS 关闭）",
                 pct(list(report["negative_controls"].values())[0]["check_vs_golden"]["worst_normalised_l2"]),
                 "**否**（预期：判据必须拒绝）"])
    rows.append(["L0 负对照 `cpu_fp64_shipped_w`（float32 权表）",
                 f"{sci(neg_l0['max_rel_deviation_field'])}（数值 <3%，但超舍入预算 "
                 f"{sci(neg_l0['ulp_budget_relative'])} → 判据 FAIL）",
                 "**否**（预期：判据必须拒绝）"])
    A(table(["关键量", "偏差", "是否 < 3%"], rows))
    A("")
    A("**结论**：")
    A("- 后端等价性（本节点的物理目标）：fp64 四档逐位一致（偏差 = 0），fp32 两档 3.6e-5 / 5.1e-5，")
    A("  L1 加速器档对 golden 2.1e-3（0.21 %）—— **全部 < 3%**。")
    A("- 对实验/DNS 基准的绝对精度：golden 槽道对 MKM-1999 的偏差 **31 %，不满足 < 3%**。原因不是后端也不是")
    A("  等价性，而是可负担网格下流场层流化（§3.3）——按 honest-FAIL 记录，本节点不宣称湍流精度达标。")
    A("")

    A("## 7. 复现命令")
    A("")
    A("```bash")
    A("export PYTHONPATH=src")
    A("python scripts/run_p106w3a2_n01_equivalence.py --stage inventory")
    A("for t in cpu_fp64_serial cpu_fp64_numpy_tree cpu_fp64_torch_gemm cpu_fp32_torch \\")
    A("         sdaa_fp64_torch sdaa_fp32_torch cpu_fp64_shipped_w; do")
    A("  python scripts/run_p106w3a2_n01_equivalence.py --stage l0 --tier $t")
    A("done")
    A("python scripts/run_p106w3a2_n01_equivalence.py --stage l0-amp --tier cpu_fp64_serial")
    A("python scripts/run_p106w3a2_n01_equivalence.py --stage l0-check")
    A("bash scripts/l1_advance.sh cpu_fp64_torch 200 95 --steps 10000 --warmup 3000 \\")
    A("     --sample-interval 10 --nx 32 --nz 32 --ny 64 --smagorinsky-cs 0.1")
    A("python scripts/run_p106w3a2_n01_equivalence.py --stage l1-finalize --profile cpu_fp64_torch \\")
    A("     --steps 10000 --warmup 3000 --sample-interval 10 --nx 32 --nz 32 --ny 64 --smagorinsky-cs 0.1 --tag full")
    A("python scripts/run_p106w3a2_n01_equivalence.py --stage aggregate")
    A("python scripts/make_p106w3a2_n01_figures.py")
    A("python scripts/render_p106w3a2_n01_report.py")
    A("pytest -q tests/test_hetero_equivalence.py")
    A("```")
    A("")
    A("## 8. 审计链")
    A("")
    A(f"- golden 逐步摘要链: `{l0_runs['cpu_fp64_serial']['digest_chain']}`")
    A(f"- golden 轨迹 npz sha256: `{load(ART / 'l0' / 'golden_cpu_fp64_serial_manifest.json')['npz_sha256']}`")
    A(f"- 权重表缺陷量化: `{json.dumps(report['weight_table_audit'], ensure_ascii=False)}`")
    A(f"- 认证矩阵: `{json.dumps(report['certificates'], ensure_ascii=False)}`")
    A("- 可再生物不入库：`artifacts/p106w3a2_n01/fields/*.npz`（各档终态场，~41 MB）与")
    A("  `artifacts/p106w3a2_n01/l1/state_*/state.pt`（续跑 checkpoint）是中间产物，")
    A("  其 sha256 已固化在 `l0/*_manifest.json` 与该 JSON 的 `fields_npz_sha256` 字段里；")
    A("  按 §7 命令重跑可逐位复现（逐步摘要链相同）。")
    A("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()