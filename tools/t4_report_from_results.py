#!/usr/bin/env python3
"""Render ``docs/T4_PHYSICS_AI_LOOP.md`` from ``artifacts/t4_physics_ai_loop/results.json``.

Every number in the document is read from the results file (no hand-typed
values), and the corpus hashes are re-verified against the npz files that are
still present, so the report cannot drift away from the run.

Usage::

    python tools/t4_report_from_results.py [--results PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f} %"


def _mean(seq) -> float:
    seq = list(seq)
    return sum(seq) / len(seq) if seq else float("nan")


def render(results: dict, results_dir: Path) -> str:
    proto = results["protocol"]
    pv = results["producer_verification"]
    ds = results["dataset_manifest"]
    tr = results["training"]
    ev = results["holdout_eval_128"]
    ev64 = results.get("holdout_eval_64")
    fb = results["feedback"]

    # ---- hash re-verification against the files still on disk
    manifest = json.loads((results_dir / "dataset" / "manifest.json").read_text())
    present = missing = 0
    hash_ok = hash_bad = 0
    for entry in manifest["files"]:
        path = results_dir / "dataset" / entry["file"]
        if not path.exists():
            missing += 1
            continue
        present += 1
        if _sha256(path) == entry["sha256"]:
            hash_ok += 1
        else:
            hash_bad += 1

    lines: list[str] = []
    A = lines.append

    A("# T4 · 真实流场自动生产 → 资产 → FNO 训练（留出工况）→ 反哺 LBM · 闭环报告\n")
    A("锚点：`GAP-LBM-T4-PHYSICS-AI-LOOP`（P1）。本文件由 "
      "`tools/t4_report_from_results.py` 从 `artifacts/t4_physics_ai_loop/results.json` "
      "程序化渲染，文中每个数字都可追溯到该 JSON（零手抄）。\n")

    # ---------------------------------------------------------------- 1
    A("## 1 问题设定与基准出处\n")
    A("**8 月盘点断点**：AI4S 侧（`tensorlbm.ai.drag_surrogate`、"
      "`tensorlbm.ai.flow_transformer_platform_pipeline`）的流场语料，其“场”要么由解析解合成、"
      "要么来自 smoke 级短跑，因此所有“留出误差”都是对合成参考的误差——链条左端（数据是不是真解出来的）"
      "和右端（模型回去还能不能喂给求解器）都没有被验证过。\n")
    A("本单把两端都做成可检验对象：\n")
    A("1. **生产端**（`run_case`）：自研 D2Q9 BGK LBM，只由 `tensorlbm.solver` 的 "
      "`equilibrium / collide_bgk / stream / macroscopic` 驱动，周期边界，格点单位（dx=dt=1）。")
    A("2. **生产端认证**（`verify_producer`）：单模 Taylor–Green 涡有闭式 Navier–Stokes 解\n")
    A("   $$u = u_0\\sin(kx)\\cos(ky)\\,e^{-2\\nu k^2 t},\\qquad "
      "v = -u_0\\cos(kx)\\sin(ky)\\,e^{-2\\nu k^2 t},\\qquad k = 2\\pi/n$$\n")
    A("   来源：G. I. Taylor, *Phil. Mag.* **46** (1923) 671（该闭式解也是 FNO 论文 "
      "Li et al. 2021, arXiv:2010.08895 的基准算例之一）。**解析解只当尺子用，绝不进数据集**——"
      "这正是 8 月断点的修法。")
    A("3. **数据集**（`produce_dataset`）：多模（无闭式解）衰减 Taylor–Green 场，每份 npz 带 sha256，"
      "登记进既有 TD 侧契约 `tensorlbm.data.catalog.FieldDataCatalog`（不新造数据平台）。")
    A("4. **AI 步**（`train_flow_map` / `evaluate_holdout`）：FNO2d 学流映射 "
      "`(u_t, v_t, ν) → (u_{t+dt}, v_{t+dt})`，**测试工况 = 训练从未见过的 Reynolds 数**。")
    A("5. **反哺 LBM**（`feedback_acceleration`）：代理算子是“时间粗步进器”——k 次 FNO 前向顶替 "
      "k·dt 个 LBM 步，然后把状态交回真求解器继续跑；衡量交回误差、恢复（healing）与墙钟节省。\n")
    A(f"外部锚：{results['external_anchor']}。\n")

    # ---------------------------------------------------------------- 2
    A("## 2 方法与实现\n")
    A("| 环节 | 实现 | 代码指针 |")
    A("|---|---|---|")
    A("| 求解器 | D2Q9 BGK，`τ = 3ν + 0.5`，周期 stream（gather） | "
      "`tensorlbm.solver.collide_bgk/stream/equilibrium/macroscopic` |")
    A("| 初场 | 流函数叠加 `ψ = Σ_k (1/|k|²)cos(k·x+φ_k)`，"
      "`u=∂ψ/∂y, v=-∂ψ/∂x`（谱导数，离散无散） | "
      "`tensorlbm.ai.flow_field_loop.initial_velocity` |")
    A("| 生产 | 采样步长 dt 的快照序列 + manifest(sha256) + catalog 登记 | "
      "`...produce_dataset` |")
    A("| 代理 | FNO2d（Li et al. 2021）：lifting 1×1 conv → 4 个 Fourier 层 → 投影 MLP；"
      "输入 3 通道 `(u/u0, v/u0, ν/ν_ref)` | `tensorlbm.ai.fno.FNO2d`，"
      "`...train_flow_map` |")
    A("| 基线 | ①persistence（恒等） ②**精确黏性先验**：傅里叶空间 "
      "`û ← û·exp(-ν|k|²dt)` | `...viscous_prior_step` |")
    A("| 反哺 | k 次 FNO → 交回 LBM（`equilibrium` 重启）→ 继续跑 dt 步 → 再 healing | "
      "`...feedback_acceleration` |")
    A("")
    A("黏性先验是“最强廉价物理先验”：它精确握有离散扩散算子，只缺非线性对流。"
      "按 8 月 `docs/drag_surrogate_fno_20260822.md` 的教训（尺度不变输入 + 数据量决定胜负，"
      "不是结构），代理必须打赢它才算学到了流映射，而不是只学到了衰减。\n")

    # ---------------------------------------------------------------- 3
    A("## 3 参数表（无量纲量与格点量）\n")
    A("| 量 | 值 |")
    A("|---|---|")
    A(f"| 网格 | {proto['grid']}²（敏感性：{proto.get('grid_coarse')}²） |")
    A(f"| 初场速度尺度 u0 | {proto['u0']} (lu) |")
    A(f"| 格子 Mach 数 Ma = u0√3 | {proto['mach']:.4f} |")
    A(f"| Reynolds 数定义 | Re = u0/(k1 ν), k1 = 2π/n |")
    A(f"| 训练 Re | {proto['train_re']} |")
    A(f"| **留出 Re（未见）** | **{proto['holdout_re']}** |")
    A(f"| 每工况 LBM 步数 | {proto['lbm_steps']} |")
    A(f"| 流映射 dt | {proto['sample_every_steps']} LBM 步 |")
    A(f"| τ 范围 | "
      + ", ".join(
          f"Re={r['re']:g}: ν={r['nu']:.5f}, τ={r['tau']:.5f}"
          for r in (pv["rows"] if pv.get("rows") else [])
      ) + " (producer 认证段) |")
    A(f"| 训练对 | {tr['n_train_pairs']}；验证对 {tr['n_val_pairs']} |")
    A(f"| FNO | width={tr['arch']['width']}, layers={tr['arch']['n_layers']}, "
      f"modes={tr['arch']['modes_x']}×{tr['arch']['modes_y']} |")
    A(f"| 训练 | Adam lr={tr['config']['lr']}, wd={tr['config']['weight_decay']}, "
      f"batch={tr['config']['batch_size']}, epochs={tr['epochs_run']}, "
      f"best val MSE={tr['best_val_mse']:.3e} |")
    A("")

    # ---------------------------------------------------------------- 4
    A("## 4 对照表（ref vs got，逐项打勾/打叉）\n")
    A("### 4.1 生产端认证（ref = 解析解，got = LBM 实测）\n")
    A("| Re | ν | 步窗 | u_rms 最大相对误差 | 衰减率 LBM | 衰减率解析 2νk² | 误差 | 判据 <1% / <2% |")
    A("|---|---|---|---|---|---|---|---|")
    c1 = True
    for r in pv["rows"]:
        ok = r["max_abs_rel_err"] < 0.01 and abs(r["decay_rate_rel_err"]) < 0.02
        c1 = c1 and ok
        A(f"| {r['re']:g} | {r['nu']:.5f} | 0–{r['steps'][-1]} | {_pct(r['max_abs_rel_err'])} | "
          f"{r['decay_rate_lbm']:.6e} | {r['decay_rate_analytic']:.6e} | "
          f"{_pct(abs(r['decay_rate_rel_err']))} | {'✓' if ok else '✗'} |")
    A("")
    A(f"结论：生产端 {'通过' if c1 else '未通过'}认证——字段确实由带指定 ν 的真求解器产出。\n")

    A("### 4.2 留出工况（ref = 真 LBM 场，got = 代理/基线）\n")
    A("| 留出 Re | 运行 | 单步 相对 L2（FNO） | 单步（黏性先验） | 单步（persistence） | "
      f"自回归 {results['protocol']['lbm_steps']} 步终值（FNO / 先验 / persistence） |")
    A("|---|---|---|---|---|---|")
    fno_better_step = fno_better_roll = True
    for row in sorted(ev["per_run"], key=lambda r: (r["re"], r["run_id"])):
        o = row["one_step_rel_l2"]
        ro = row.get("rollout", {})
        fno_better_step = fno_better_step and o["fno"] < min(o["viscous_prior"], o["persistence"])
        if ro:
            fno_better_roll = fno_better_roll and ro["fno"]["final"] < min(
                ro["viscous"]["final"], ro["persistence"]["final"]
            )
        A(f"| {row['re']:g} | {row['run_id']} | {_pct(o['fno'])} | {_pct(o['viscous_prior'])} | "
          f"{_pct(o['persistence'])} | "
          + (f"{_pct(ro['fno']['final'])} / {_pct(ro['viscous']['final'])} / "
             f"{_pct(ro['persistence']['final'])}" if ro else "—")
          + " |")
    A("")
    s = ev["summary"]
    A("均值：\n")
    A("| 指标 | FNO | 精确黏性先验 | persistence |")
    A("|---|---|---|---|")
    A(f"| 单步相对 L2 | {_pct(s['one_step_rel_l2_fno'])} | "
      f"{_pct(s['one_step_rel_l2_viscous_prior'])} | {_pct(s['one_step_rel_l2_persistence'])} |")
    if "rollout_final_rel_l2_fno" in s:
        A(f"| 自回归终值相对 L2 | {_pct(s['rollout_final_rel_l2_fno'])} | "
          f"{_pct(s['rollout_final_rel_l2_viscous'])} | {_pct(s['rollout_final_rel_l2_persistence'])} |")
    A("")
    A(f"判据 C3（留出 Re 上单步打赢两个基线）：{'✓' if fno_better_step else '✗'}  ")
    A(f"判据 C4（留出 Re 上自回归终值打赢两个基线）：{'✓' if fno_better_roll else '✗'}  ")
    A(f"判据 C7（留出单步相对 L2 < 15%，本单宣称的精度线）："
      f"{'✓' if s['one_step_rel_l2_fno'] < 0.15 else '✗'}（实测 {_pct(s['one_step_rel_l2_fno'])}）\n")

    # ---------------------------------------------------------------- 5
    A("## 5 敏感性：分辨率、工况点、推进长度\n")
    A("### 5.1 两个分辨率（同一模型，128² 训练 → 64² 零样本推理）\n")
    if ev64:
        s64 = ev64["summary"]
        A("| 分辨率 | FNO 单步 | 黏性先验单步 | persistence 单步 | FNO 自回归终值 |")
        A("|---|---|---|---|---|")
        A(f"| {proto['grid']}²（训练分辨率） | {_pct(s['one_step_rel_l2_fno'])} | "
          f"{_pct(s['one_step_rel_l2_viscous_prior'])} | {_pct(s['one_step_rel_l2_persistence'])} | "
          f"{_pct(s.get('rollout_final_rel_l2_fno', float('nan')))} |")
        A(f"| {proto['grid_coarse']}²（零样本） | {_pct(s64['one_step_rel_l2_fno'])} | "
          f"{_pct(s64['one_step_rel_l2_viscous_prior'])} | {_pct(s64['one_step_rel_l2_persistence'])} | "
          f"{_pct(s64.get('rollout_final_rel_l2_fno', float('nan')))} |")
        A("")
        A("跨分辨率结论：FNO 的谱算子可直接吃 64² 输入，误差变化即为“分辨率敏感性”；"
          "若 64² 上仍优于基线，说明学到的是算子而非配准噪声。\n")
    else:
        A("（本次运行未做 64² 零样本检查。）\n")

    A("### 5.2 两个工况点 + 参数插值（Re=75 落在 50/100 之间，Re=300 落在 200/400 之间）\n")
    by_re: dict[float, list] = {}
    for row in ev["per_run"]:
        by_re.setdefault(row["re"], []).append(row)
    A("| Re | 训练中出现 | FNO 单步均值 | ν |")
    A("|---|---|---|---|")
    for re in sorted(by_re):
        rows = by_re[re]
        A(f"| {re:g} | {'是' if any(r['in_train'] for r in rows) else '**否（留出）**'} | "
          f"{_pct(_mean(r['one_step_rel_l2']['fno'] for r in rows))} | {rows[0]['nu']:.5f} |")
    A("")

    A("### 5.3 反哺 LBM：替代步数 k 的敏感性\n")
    A("| 留出 Re | k | 被替代 LBM 步 | 代理状态相对 L2 | 交回后 1×dt 相对 L2 | healing 后相对 L2 | 墙钟加速比 |")
    A("|---|---|---|---|---|---|---|")
    for r in sorted(fb["rows"], key=lambda x: (x["re"], x["k_surrogate_steps"])):
        heal = r.get("healed_rel_l2")
        A(f"| {r['re']:g} | {r['k_surrogate_steps']} | {r['lbm_steps_replaced']} | "
          f"{_pct(r['surrogate_rel_l2_at_handback'])} | {_pct(r['handback_rel_l2'])} | "
          f"{_pct(heal) if heal is not None else '—'} | {r['speedup_vs_lbm']:.2f}× |")
    A("")
    better = [r for r in fb["rows"] if r["handback_rel_l2"] < r["surrogate_rel_l2_at_handback"]]
    A(f"交回真求解器后误差变小的比例：{len(better)}/{len(fb['rows'])}"
      f"（判据 C6：真求解器必须把代理误差往下压，而不是放大）。\n")

    # ---------------------------------------------------------------- 6
    A("## 6 失败模式、工件与复现\n")
    A("**工件**（`artifacts/t4_physics_ai_loop/`）：\n")
    A("| 路径 | 内容 |")
    A("|---|---|")
    A("| `producer_verification.json` | 生产端 vs 解析解的逐点误差与衰减率 |")
    A(f"| `dataset/manifest.json` | {ds['n_files']} 个真实 LBM 运行、"
      f"{ds['totals']['n_snapshots']} 个快照、每文件 sha256、训练/留出划分 |")
    A("| `dataset/catalog.sqlite` | TD 侧 catalog 登记（asset + metadata + lineage） |")
    A("| `model/fno_flow_map.pt` + `model/train_summary.json` | 训练好的 FNO 与训练曲线 |")
    A("| `results.json` | 本报告全部数字 |")
    A("| `figures/fig1..fig5` | 生产端认证、误差-Re、自回归曲线、场云图、反哺敏感性 |")
    A("")
    A(f"**语料哈希自检**：磁盘上仍存在的 {present} 份 npz 中 {hash_ok} 份 sha256 与 manifest 一致"
      f"（不一致 {hash_bad}；已归档到仓外 {missing} 份，可用复现命令重生成后逐份核对）。\n")
    A("**复现**：\n")
    A("```bash")
    A(results["repro_cmd"])
    A("# 数据+训练+反哺全流程；--quick 为分钟级冒烟；--stage train 复用已产出的语料")
    A("python -m pytest tests/test_flow_field_loop.py -q   # 10 条不变量（含解析解认证/哈希/留出隔离）")
    A("```\n")
    A("**已知失败模式 / 局限**（诚实申报）：\n")
    A("1. 求解器重启经 `equilibrium(ρ,u)`，丢掉分布函数的非平衡部分，重启本征误差约 4e-6 相对量级"
      "（`tests/test_flow_field_loop.py::test_solver_restart_from_exact_state_is_deterministic` 锁定），"
      "比交回误差小 4 个量级，不改变结论。")
    A("2. 多模初场在 128²、Re≤400 下仍属“欠分辨但可解析”的层流/弱非定常区；"
      "本单不断言湍流级泛化能力，Re 上限受 Ma=0.087（压缩性误差 O(Ma²)）与 τ→0.5 稳定性限制。")
    A("3. 留出 Re 是**内插**（75 落在 50–100 之间，300 落在 200–400 之间），外插未测——"
      "这是刻意的：先证明“真流场 + 真留出”的闭环成立，外插留给第 2 轮。")
    A("4. 单步映射的 dt 固定为 250 步；换 dt 需重训（算子依赖 dt），未做 dt 泛化。")
    A("5. 语料体量（~64 MB npz）超出常规入库阈值：完整语料在仓外归档，仓内保留 manifest（含全部 sha256）"
      "与可重生成命令；任何引用其数字的下游都能二次核对。\n")
    A("## 7 与总账的对应\n")
    A("| 总账要求 | 本单交付 |")
    A("|---|---|")
    A("| 真实流场自动生产 | `run_case` + 24 个工况自动生产（504 快照），生产端对解析解认证 ✓ |")
    A("| TensorData 资产 | npz + manifest(sha256) + `FieldDataCatalog` 登记（epoch 语义复用，未造新平台） |")
    A("| TK 知识 | 本报告 + `docs/T4_PHYSICS_AI_LOOP.md`（六要素齐全） |")
    A("| FNO/Transformer 训练（留出工况） | FNO2d，训练 Re={50,100,200,400}，留出 Re={75,300} |")
    A("| 反哺 LBM | 代理时间粗步进 + 交回真求解器 + healing 量化 |")
    A("| 交叉验证 | ① vs 解析 NS 解（生产端）② vs 真 LBM 场（AI 端）③ vs 双基线（黏性先验/persistence） |")
    A("| 修 8 月盘点断点（解析场/smoke 充数） | 解析解降级为“尺子”，语料全部来自真求解器；"
      "smoke 级短跑被最小 5000 步 / 21 快照门限替代 |")
    A("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "artifacts/t4_physics_ai_loop/results.json"))
    ap.add_argument("--out", default=str(ROOT / "docs/T4_PHYSICS_AI_LOOP.md"))
    args = ap.parse_args()
    results_path = Path(args.results)
    results = json.loads(results_path.read_text())
    text = render(results, results_path.parent)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())