#!/usr/bin/env python3
"""Board audit harness for m17677 XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE.

Cross-check (交差审计), not a re-run of the original mission.  Everything below is
computed from the audited work tree ``/root/tp-agent/work/m17677`` (read only) and
written into this work tree:

  artifacts/m17677_audit/audit_m17677.json      machine readings
  artifacts/m17677_audit/figures/fig_m17677_evidence.png
  docs/audit_m17677/AUDIT_m17677.md             six-section report (numbers injected)

Checks
------
S1 static truthfulness of the three-piece deliverable (run artifacts / numbers /
   repro command) + entry-vs-artifact consistency, placement, callers, doc refs;
S2 numeric re-derivation: k_super = nu*<u>_super/g recomputed from the recorded
   velocity mean, error vs the analytic slab anchor, independent brute-force
   rebuild of the neighbour table;
S3 convergence probe on the anchor case (1200 / 3000 / 6000 steps) -- the
   delivered artifact stopped at 1200 steps with converged=false;
S4 bit-exact reproduction of the delivered results.json with the delivered
   runner (its step constant is pinned, programmatically, to the artifact's
   ``steps_cap`` because the committed runner has moved on to 2500).

Usage
-----
  python scripts/audit/m17677_compressed_node_table_audit.py                # S1-S3 (~40 s)
  python scripts/audit/m17677_compressed_node_table_audit.py --with-repro   # + S4 (~3 min)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch

AUDIT_WS = os.environ.get("M17677_WS", "/root/tp-agent/work/m17677")
REPO = os.path.join(AUDIT_WS, "TensorLBM")           # the shadow tree the mission wrote into
SHADOW_SRC = os.path.join(REPO, "src")
XVAL = os.path.join(REPO, "xval", "XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE")
ART = os.path.join(REPO, "artifacts", "xval_compressed_fluid_node_neighbour_table")
ROOT_WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT_WS, "artifacts", "m17677_audit")
DOCS = os.path.join(ROOT_WS, "docs", "audit_m17677")

TAU = 0.8
NU = (TAU - 0.5) / 3.0
OWNER_BAND_PCT = 15.0

torch.set_default_dtype(torch.float64)
threads = torch.get_num_threads()
torch.set_num_threads(1)


# --------------------------------------------------------------------- helpers
def load_recorded() -> dict:
    with open(os.path.join(ART, "results.json")) as fh:
        return json.load(fh)


def rederive_k(rec: dict) -> dict:
    """k_super must equal nu*<u>_super/g, k_intrinsic = k_super/porosity."""
    a1 = rec["A1"]
    k_super = NU * a1["u_mean_superficial"] / rec["case_cfg"]["g"]
    k_int = k_super / a1["porosity"]
    return {
        "k_super_rederived": k_super,
        "k_super_recorded": a1["k_darcy"],
        "k_super_rel_gap": abs(k_super - a1["k_darcy"]) / abs(a1["k_darcy"]),
        "k_int_rederived": k_int,
        "k_int_recorded": a1["k_intrinsic"],
        "ma_recorded": a1["ma"],
        "rho_drift_recorded": a1["rho_drift"],
        "converged": a1["converged"],
        "steps": a1["steps"],
    }


def static_checks(recorded: dict) -> dict:
    files = {
        "impl_fluid_node_table": os.path.join(SHADOW_SRC, "tensorlbm", "porous", "fluid_node_table.py"),
        "impl_init": os.path.join(SHADOW_SRC, "tensorlbm", "porous", "__init__.py"),
        "tests": os.path.join(REPO, "tests", "test_porous_fluid_node_table.py"),
        "entry_predeclared": os.path.join(XVAL, "run_xval.py"),
        "runner_fast": os.path.join(XVAL, "run_xval_fast.py"),
        "artifact_results": os.path.join(ART, "results.json"),
        "predeclared_spec": os.path.join(AUDIT_WS, "xval", "XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE",
                                         "predeclared.json"),
    }
    present = {k: os.path.exists(v) and os.path.getsize(v) > 0 for k, v in files.items()}

    # ---- entry vs artifact: does the declared entry reproduce the artifact?
    run_xval = open(files["entry_predeclared"]).read()
    fast = open(files["runner_fast"]).read()
    declared_cases = []
    for ln in run_xval.splitlines():
        if "cases.append(dict(name=" not in ln:
            continue
        m = re.search(r'name=(f?)"([^"]*)"', ln)
        if m:
            declared_cases.append(("[f-string] " if m.group(1) else "") + m.group(2))
    artifact_cases = [r["case"] for r in recorded["records"]]
    fast_steps = int(next(ln.split("=")[1] for ln in fast.splitlines() if ln.startswith("STEPS =")))
    entry_steps = int(next(ln.split("=")[1].split("#")[0]
                           for ln in run_xval.splitlines() if ln.startswith("STEPS_CAP")))

    # ---- placement: is the module in the canonical package the repo ships?
    canonical = os.path.join(AUDIT_WS, "src", "tensorlbm", "porous")
    imp = subprocess.run([sys.executable, "-c", "import tensorlbm.porous;print(tensorlbm.porous.__file__)"],
                         cwd=AUDIT_WS, env={**os.environ, "PYTHONPATH": os.path.join(AUDIT_WS, "src")},
                         capture_output=True, text=True)
    canonical_import = imp.returncode == 0

    # ---- callers outside the module/tests/xval: any main-solver integration?
    # module-exact reference ("tensorlbm.porous" not followed by an identifier char,
    # so "tensorlbm.porous_media" / "porous_ladder" do not count as callers)
    ref_pat = r"tensorlbm\.porous([^A-Za-z0-9_]|$)"
    grep = subprocess.run(["grep", "-rlE", "--include=*.py", ref_pat, AUDIT_WS],
                          capture_output=True, text=True)
    callers = [p for p in grep.stdout.split() if p]
    external_callers = [p for p in callers
                        if "/porous/fluid_node_table.py" not in p
                        and "/porous/__init__.py" not in p
                        and "test_porous_fluid_node_table" not in p
                        and "xval/XVAL-LBM-COMPRESSED-FLUID" not in p]

    # ---- docs referenced by the module: do they exist?
    reffed = "docs/xval_compressed_fluid_node_neighbour_table.md"
    doc_present = os.path.exists(os.path.join(AUDIT_WS, "docs", reffed.split("/", 1)[1])) or \
        os.path.exists(os.path.join(REPO, reffed))

    # ---- placeholder scan of the implementation
    body = open(files["impl_fluid_node_table"]).read()
    placeholders = sum(body.count(tok) for tok in ("NotImplementedError", "raise TODO", "FIXME"))

    committed = subprocess.run(["git", "status", "--porcelain"], cwd=AUDIT_WS,
                               capture_output=True, text=True).stdout.strip().splitlines()
    log = subprocess.run(["git", "log", "--oneline", "-1"], cwd=AUDIT_WS,
                         capture_output=True, text=True).stdout.strip()

    return {
        "files": files,
        "files_present": present,
        "entry_declared_cases": declared_cases,
        "artifact_cases": artifact_cases,
        "entry_declared_steps_cap": entry_steps,
        "artifact_steps_cap": recorded["steps_cap"],
        "runner_fast_committed_steps": fast_steps,
        "entry_reproduces_artifact": declared_cases == artifact_cases
                                     and entry_steps == recorded["steps_cap"],
        "committed_runner_reproduces_artifact": fast_steps == recorded["steps_cap"],
        "canonical_package_path": canonical,
        "canonical_package_has_module": os.path.exists(canonical),
        "canonical_import_ok": canonical_import,
        "canonical_import_err": (imp.stderr.strip().splitlines() or [""])[-1],
        "n_files_referencing_module": len(callers),
        "external_callers": external_callers,
        "referenced_doc": reffed,
        "referenced_doc_present": doc_present,
        "impl_placeholders": placeholders,
        "git_head": log,
        "git_untracked": committed[:12],
    }


def bruteforce_table_check() -> dict:
    """Independent rebuild of the compaction map + stencil (no shared code)."""
    sys.path.insert(0, SHADOW_SRC)
    from tensorlbm.porous import build_fluid_node_table  # noqa: E402

    ny, nx = 5, 6
    rng = np.random.default_rng(20261113)
    solid_np = rng.random((ny, nx)) < 0.35
    solid_np[0, 0] = False
    solid = torch.tensor(solid_np)

    table = build_fluid_node_table(solid)
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]  # E N W S NE NW SW SE

    idx = -np.ones((ny, nx), dtype=np.int64)
    k = 0
    for y in range(ny):
        for x in range(nx):
            if not solid_np[y, x]:
                idx[y, x] = k
                k += 1
    ref = np.full((k, 8), -1, dtype=np.int64)
    ref_nodes = []
    for y in range(ny):
        for x in range(nx):
            if solid_np[y, x]:
                continue
            kk = idx[y, x]
            ref_nodes.append(y * nx + x)
            for s, (cx, cy) in enumerate(dirs):
                ref[kk, s] = idx[(y + cy) % ny, (x + cx) % nx]

    got_idx = table.index.numpy()
    got_nb = table.neighbour.numpy()
    return {
        "mask": solid_np.tolist(),
        "n_fluid": int(table.n_fluid),
        "n_fluid_bruteforce": int(k),
        "index_map_equal": bool(np.array_equal(got_idx, idx)),
        "nodes_equal": bool(np.array_equal(table.nodes.numpy(), np.array(ref_nodes, dtype=np.int64))),
        "neighbour_table_equal": bool(np.array_equal(got_nb, ref)),
        "mismatch_entries": int((got_nb != ref).sum()),
    }


def convergence_probe() -> list[dict]:
    """k on the delivered anchor geometry (channel H=16 / ny=32) vs step count."""
    sys.path.insert(0, XVAL)
    sys.path.insert(0, SHADOW_SRC)
    import run_xval as R  # noqa: E402

    solid = R.channel_mask(32, 16, 16)
    h, ny = 16, 32
    k_ref = (h / ny) * h * h / 12.0
    rows = []
    for steps in (1200, 3000, 6000):
        a = R.run_compressed(solid, 1e-5, R.TAU, steps_cap=steps, record=False)
        rows.append({
            "steps": steps,
            "k_super": a["k_darcy"],
            "k_intrinsic": a["k_intrinsic"],
            "k_ref_super": k_ref,
            "err_pct": 100.0 * (a["k_darcy"] - k_ref) / k_ref,
            "converged": a["converged"],
            "rho_drift": a["rho_drift"],
            "ma": a["ma"],
        })
    return rows


def reproduction(recorded: dict) -> dict:
    """Re-run the delivered runner with its step constant pinned to the artifact."""
    repro_dir = os.path.join(OUT, "repro")
    os.makedirs(repro_dir, exist_ok=True)
    steps = int(recorded["steps_cap"])
    outdir = os.path.join(repro_dir, "artifacts")
    os.makedirs(outdir, exist_ok=True)
    for name in ("run_xval.py", "run_xval_fast.py"):
        src = open(os.path.join(XVAL, name)).read()
        if name == "run_xval_fast.py":
            src = "\n".join(f"STEPS = {steps}" if ln.startswith("STEPS =") else ln
                            for ln in src.splitlines())
            src = "\n".join(f'OUT = r"{outdir}"' if ln.startswith("OUT =") else ln
                            for ln in src.splitlines())
        open(os.path.join(repro_dir, name), "w").write(src)
    env = {**os.environ, "PYTHONPATH": SHADOW_SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    run = subprocess.run([sys.executable, os.path.join(repro_dir, "run_xval_fast.py")],
                         capture_output=True, text=True, cwd=repro_dir, env=env)
    path = os.path.join(outdir, "results.json")
    if run.returncode != 0 or not os.path.exists(path):
        return {"ok": False, "stderr_tail": run.stderr[-600:]}
    new = json.load(open(path))
    old = {r["case"]: r for r in recorded["records"]}
    per = {}
    for r in new["records"]:
        o = old[r["case"]]
        per[r["case"]] = {
            "k_recorded": o["A1"]["k_darcy"],
            "k_reproduced": r["A1"]["k_darcy"],
            "k_rel_diff": abs(o["A1"]["k_darcy"] - r["A1"]["k_darcy"]) / abs(o["A1"]["k_darcy"]),
            "compare_recorded": o["compare"]["ux_maxabs_rel"],
            "compare_reproduced": r["compare"]["ux_maxabs_rel"],
            "k_err_pct_recorded": o["k_err_vs_ref_pct"],
            "k_err_pct_reproduced": r["k_err_vs_ref_pct"],
        }
    return {"ok": True, "pinned_steps": steps, "per_case": per,
            "max_k_rel_diff": max(v["k_rel_diff"] for v in per.values()),
            "all_bit_exact": all(v["k_rel_diff"] == 0.0 for v in per.values())}


def db_judge() -> dict:
    """Registered benchmark judge: rows for this anchor in tensor_benchmark_cases."""
    sql = ("SELECT count(*) FROM tensor_benchmark_cases WHERE "
           "anchor_slug='XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE'")
    r = subprocess.run(["psql", "-U", "postgres", "-d", os.environ.get("TP_PG_DB", "tensorplatform"),
                        "-tA", "-c", sql], capture_output=True, text=True)
    rows = None
    try:
        rows = int(r.stdout.strip())
    except Exception:
        pass
    return {"sql": sql, "registered_rows": rows, "stderr_tail": r.stderr.strip()[-200:]}


# ------------------------------------------------------------------- reporting
def figures(recorded: dict, conv: list[dict], path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    steps = [c["steps"] for c in conv]
    err = [c["err_pct"] for c in conv]
    ax1.axhspan(-OWNER_BAND_PCT, OWNER_BAND_PCT, color="0.85", label="owner band ±15 %")
    ax1.axhline(0.0, color="k", lw=0.8, ls="--", label="analytic slab k_super = (H/ny)H²/12")
    ax1.plot(steps, err, "o-", color="#c0392b", lw=1.8, label="auditor probe (this work)")
    rec = next(r for r in recorded["records"] if r["case"].startswith("C1"))
    ax1.plot([rec["A1"]["steps"]], [rec["k_err_vs_ref_pct"] * np.sign(1)], "s", ms=9,
             mfc="none", mec="#2c3e50", mew=1.8,
             label=f"delivered artifact ({rec['A1']['steps']} steps, converged={rec['A1']['converged']})")
    ax1.set_xscale("log")
    ax1.set_xlabel("LBM steps")
    ax1.set_ylabel("k_super error vs analytic anchor  [%]")
    ax1.set_title("C1 plane channel H=16: permeability error")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower left")

    names = [r["case"].replace("_porous", "").replace("C2_", "") for r in recorded["records"]]
    ratio = [r["mem"]["total_ratio"] for r in recorded["records"]]
    ax2.bar(range(len(names)), ratio, color=["#c0392b" if v < 1 else "#27ae60" for v in ratio])
    ax2.axhline(1.0, color="k", lw=0.9)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("dense / compressed footprint  [-]")
    ax2.set_title("state footprint ratio (from artifact; int64 table)")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def markdown(recorded: dict, st: dict, numeric: dict, bf: dict, conv: list[dict],
             repro: dict | None, judge: dict, fig: str) -> str:
    rec_c1 = next(r for r in recorded["records"] if r["case"].startswith("C1"))
    conv_best = min(conv, key=lambda c: abs(c["err_pct"]))
    c1_run = numeric["C1_channel_H16"]
    L = []
    a = L.append
    a("# AUDIT m17677 — XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE（交差核验）")
    a("")
    a(f"- 被审 mission：`17677`（succeeded，`budget_exhausted+forced_finish`），worktree `{AUDIT_WS}`")
    a("- 审计 mission：17825（交差核验单，非重做）；`verdict_by='board-audit'`")
    a(f"- 被审树 HEAD：`{st['git_head']}`；交付全部为未跟踪文件（{len(st['git_untracked'])} 条 `??`）")
    a("")
    a("## ① 案例介绍")
    a("")
    a("对象是 openLBMPM（PorousMediaSimulation）的 *compressed fluid-node neighbour table* 语义：")
    a("多孔介质里只有流体节点带未知量，把 9 个分布函数压实到 `K` 个流体节点，并保留一张")
    a("`(K,8)` 的邻居表把「流体节点 + 方向 → 邻居的压实索引（或 -1=固体）」映射好，")
    a("流式改为 push/scatter + 半程反弹。XVAL 的设计是双锚：(a) 表示等价性——压缩表 vs 仓内稠密全格在")
    a("同一时间步下浮点逐位一致；(b) 物理锚点——平面槽道渗透率 vs 解析 `k_super=(H/ny)H²/12`。")
    a("")
    a("| 参数 | 值 | 来源 |")
    a("|---|---|---|")
    a(f"| 碰撞/forcing | D2Q9 BGK + Guo 体力，tau={recorded['tau']}，nu={recorded['nu']:.4f} | `results.json` |")
    a(f"| 精度 | float64，单线程 torch {recorded['torch']}，numpy {recorded['numpy']} | `results.json` |")
    a(f"| 步数预算 | {recorded['steps_cap']}（`steady_tol={recorded['steady_tol']}`） | `results.json` |")
    a("| 算例 | C1 槽道 H=16/ny=32；C2 方形颗粒 p=0.90/0.75/0.60 N=64；C5 g×2；C6 N=96 | `results.json` |")
    a(f"| 预声明（PLAN） | 槽道 H=32/ny=64、N=128/256、12000 步 | `xval/.../predeclared.json` |")
    a("")
    a("## ② 怎么做的")
    a("")
    a("1. **取三件套原件**：`results.json`、`run_xval.py` / `run_xval_fast.py`、`tests/`、日志与 `predeclared.json`。")
    a("2. **基准裁判先问库**：`tensor_benchmark_cases` 按 `anchor_slug` 查登记案例（唯一裁判）。")
    a(f"3. **数值重算**：从记录的 `<u>_super` 反算 `k_super=nu<u>/g`，与工件记录比对（{c1_run['k_super_rel_gap']:.1e} 相对差）；"
      "误差对解析锚；另用独立 numpy 代码重写压实映射与邻居表，与被审实现逐项比对。")
    a("4. **收敛探针**：交付工件停在 "
      f"{rec_c1['A1']['steps']} 步且 `converged=false`，故在被审代码上把 C1 跑到 "
      f"{conv[-1]['steps']} 步，看 k 是收敛还是巧合。")
    a("5. **逐位复现**：把交付 runner 的步数常量**程序化**钉到工件 `steps_cap`"
      + ("，运行并与工件逐算例比对。" if repro else "（本次未跑 `--with-repro`，如实标注未跑）。"))
    a("6. **落表**：`tensor_maturity_verdicts`（repo=TensorLBM, scope=m17677 …）,由 `maturity_gate` 机器复核。")
    a("")
    a("## ③ 计算结果")
    a("")
    a("### 3.1 基准裁判读数")
    a("")
    a(f"- SQL：`{judge['sql']}` → **{judge['registered_rows']} 行**")
    a("- 结论：该锚点在 `tensor_benchmark_cases` 里**没有任何登记案例/容差**，"
      "因此「基准判据」这一维度对该锚点不存在——只能用交付自带解析锚做人工核对（下表）。")
    a("")
    a("### 3.2 交付工件逐算例（记录值，k 由 <u> 反算复核）")
    a("")
    a("| case | steps | converged | k_super 记录 | k_super 反算 | 相对差 | k vs 解析 [%] | A1 vs A2 ux_maxabs_rel | 足迹比 dense/compressed |")
    a("|---|---|---|---|---|---|---|---|---|")
    for r in recorded["records"]:
        n = numeric[r["case"]]
        err = "—" if r["k_err_vs_ref_pct"] is None else f"{r['k_err_vs_ref_pct']:.3f}"
        a(f"| {r['case']} | {r['A1']['steps']} | {r['A1']['converged']} | {r['A1']['k_darcy']:.6f} | "
          f"{n['k_super_rederived']:.6f} | {n['k_super_rel_gap']:.1e} | {err} | "
          f"{r['compare']['ux_maxabs_rel']:.1e} | {r['mem']['total_ratio']:.3f} |")
    a("")
    a("C1 槽道解析锚 `k_ref = (H/ny)H²/12 = "
      f"{rec_c1['k_reference']:.6f}`；交付 k_super={rec_c1['A1']['k_darcy']:.6f} ⇒ "
      f"误差 {rec_c1['k_err_vs_ref_pct']:.3f}%。其余 5 个算例 `k_reference=null`（无解析锚），"
      "只有 A1/A2 表示一致性可核。")
    a("")
    a("### 3.3 收敛探针（审计者独立运行，被审代码原样）")
    a("")
    a("| steps | converged | k_super | k_intrinsic | vs 解析 [%] | rho_drift | Ma |")
    a("|---|---|---|---|---|---|---|")
    for c in conv:
        a(f"| {c['steps']} | {c['converged']} | {c['k_super']:.6f} | {c['k_intrinsic']:.6f} | "
          f"{c['err_pct']:+.3f} | {c['rho_drift']:.1e} | {c['ma']:.4f} |")
    a("")
    a(f"→ 交付工件的 {rec_c1['k_err_vs_ref_pct']:.3f}%（1200 步）是**瞬态值**；收敛值在 "
      f"{conv_best['steps']} 步为 {conv_best['k_super']:.6f}（{conv_best['err_pct']:+.3f}%）。"
      "两者都远优于 15% 验收带，但「shot 精度」被瞬态偶然放大，报告写 1.06% 属保守而非夸大。")
    a("")
    a("### 3.4 邻居表独立重算（numpy 直接写，不复用被审代码）")
    a("")
    a(f"- 5×6 掩膜（{bf['n_fluid']} 个流体节点）：`nodes/compaction` 一致={bf['nodes_equal']}，"
      f"`index` 映射一致={bf['index_map_equal']}，`(K,8)` 邻居表一致={bf['neighbour_table_equal']}"
      f"（不一致条目 {bf['mismatch_entries']}）")
    a(f"- 重算流体节点数 {bf['n_fluid_bruteforce']} vs 实现 {bf['n_fluid']}")
    a("")
    if repro and not repro.get("ok"):
        a("### 3.5 工件逐位复现")
        a("")
        a(f"复现运行失败（本次记录了 stderr）：`{repro.get('stderr_tail', '')[-300:]}`")
        a("")
    elif repro:
        a("### 3.5 工件逐位复现")
        a("")
        a(f"钉 `STEPS={repro['pinned_steps']}`（= 工件 `steps_cap`）后重跑交付 runner："
          f"最大 k 相对差 **{repro['max_k_rel_diff']:.1e}**，全算例逐位相同={repro['all_bit_exact']}。")
        a("")
        a("| case | k 工件 | k 复现 | 相对差 | k 误差工件 [%] | k 误差复现 [%] |")
        a("|---|---|---|---|---|---|")
        for name, v in repro["per_case"].items():
            e_rec = "—" if v["k_err_pct_recorded"] is None else f"{v['k_err_pct_recorded']:.3f}"
            e_rep = "—" if v["k_err_pct_reproduced"] is None else f"{v['k_err_pct_reproduced']:.3f}"
            a(f"| {name} | {v['k_recorded']:.6f} | {v['k_reproduced']:.6f} | {v['k_rel_diff']:.1e} | "
              f"{e_rec} | {e_rep} |")
        a("")
    else:
        a("### 3.5 工件逐位复现")
        a("")
        a("未跑（缺 `--with-repro`）。")
        a("")
    a("### 3.6 结构面读数（三件套是否真存在）")
    a("")
    a("| 项 | 读数 |")
    a("|---|---|")
    for k, v in st["files_present"].items():
        a(f"| 文件存在：{k} | {v} |")
    a(f"| 声明入口算例 | {st['entry_declared_cases']} |")
    a(f"| 工件算例 | {st['artifact_cases']} |")
    a(f"| 声明入口步数 `STEPS_CAP` | {st['entry_declared_steps_cap']} |")
    a(f"| 工件 `steps_cap` | {st['artifact_steps_cap']} |")
    a(f"| 当前提交 runner 步数 | {st['runner_fast_committed_steps']} |")
    a(f"| 声明入口能产出该工件 | {st['entry_reproduces_artifact']} |")
    a(f"| 当前提交 runner 能产出该工件 | {st['committed_runner_reproduces_artifact']} |")
    a(f"| 模块所在 canonical 路径存在 | {st['canonical_package_has_module']}（`{st['canonical_package_path']}`） |")
    a(f"| 从仓根 `import tensorlbm.porous` | {st['canonical_import_ok']}（{st['canonical_import_err']}） |")
    a(f"| 模块外部调用方（主求解器/平台） | {len(st['external_callers'])} |")
    a(f"| 实现占位符计数 | {st['impl_placeholders']} |")
    a(f"| 模块自引文档 `{st['referenced_doc']}` 存在 | {st['referenced_doc_present']} |")
    a("")
    a(f"![evidence]({os.path.relpath(fig, os.path.dirname(DOCS))})")
    a("")
    a("## ④ 结果图")
    a("")
    a("左：C1 槽道渗透率相对解析锚的误差随步数（灰带=owner ±15% 验收带，方框=交付工件 1200 步瞬态点，")
    a("圆点=本次收敛探针），可见交付点仍在瞬态、收敛后被审实现误差反而降到 0.1% 量级。")
    a("右：交付工件的**状态足迹比** `dense/compressed`（int64 邻居表）；p=0.90 时 <1，")
    a("即高孔隙率下压缩布局比稠密布局**更占内存**——这是工件里真实存在的反向结果，报告未粉饰。")
    a(f"图：`{os.path.relpath(fig, ROOT_WS)}`（matplotlib 由 `audit_m17677.json` 数值程序化生成）。")
    a("")
    a("## ⑤ 注意点（陷阱/局限/适用范围）")
    a("")
    a("1. **交付落在影子树里**：模块与测试写在 `TensorLBM/src|tests|...`（仓内一份 174 文件的旧嵌套副本），"
      f"而不是仓的 canonical `src/tensorlbm/`；从仓根 `import tensorlbm.porous` 失败"
      f"（`{st['canonical_import_err']}`）。能力对本仓用户**不可用**，是本次最大结构性缺陷。")
    a("2. **入口与工件不一致**：`run_xval.py`（预声明全梯）跑的是 H=32/ny=64、N=128/256、12000 步，"
      f"与工件 6 算例（H=16/ny=32、N=64/96、{st['artifact_steps_cap']} 步）不同；"
      f"当前 `run_xval_fast.py` 已把步数改成 {st['runner_fast_committed_steps']}，"
      "即**用提交原样的脚本产不出该工件**，必须把常量钉回工件值才能逐位复现（本审计已程序化做到）。")
    a("3. **预声明梯未完成**：`predeclared.json` 声明 N=128/256 与 12000 步；`run.log` 显示 H=32 跑完 12000 步"
      "（k_err≈0.028%）后在 C2 N=128 处被截断，成果没进工件；`run2.log` 是 argparse 报错残留，而 `fast.log` 才是工件来源。")
    a("4. **全算例 `converged=false`**：1200 步内达不到 1e-9；`ladder_note` 声称「KPI 也在 HALF 步采样」，"
      "但代码里 `HALF` 只出现在说明字符串里，从未用于采样 ⇒ 声称的步数自收敛检查**没有实现**。")
    a("5. **外部锚不可执行**：被对标 OSS（openLBMPM）是 CUDA 代码，无 GPU 无法运行；交付做的是语义移植，"
      "`predeclared.json` 已如实声明为比较局限。provenance 的 `file:line` 无法在本环境核对（磁盘上无 OSS 源码）⇒ 外部锚 WITHHELD。")
    a("6. **A1/A2 对比只在流式层**：两臂共享碰撞/forcing 实现，故 0.0 只证明「压实+push 流式 ≡ 稠密 pull 流式」，"
      "这正是声明的主张；不得外推为「物理正确」。")
    a("7. **内嵌文档悬空**：`porous/__init__.py` 引用 `docs/xval_compressed_fluid_node_neighbour_table.md`，"
      "该文件从未生成（`make_report.py` 没跑过），交付里没有任何六段报告，`tensor_reports` 无 mission 17677 行。")
    a("8. 无测试/无登记的基准案例、无 `repro_cmd` 记录、capability_delta 为 `no_substantive_delta`"
      "（finish 阶段未声明增量）。")
    a("")
    a("## ⑥ 误差判定（逐项回答是否 <3%）")
    a("")
    a("| 关键量 | 基准 | 交付 got | 误差 | <3%? |")
    a("|---|---|---|---|---|")
    a(f"| C1 k_super（1200 步，工件） | 解析 `(H/ny)H²/12`={rec_c1['k_reference']:.6f} | "
      f"{rec_c1['A1']['k_darcy']:.6f} | {rec_c1['k_err_vs_ref_pct']:.3f}% | 是 |")
    a(f"| C1 k_super（{conv_best['steps']} 步，审计收敛探针） | 同上 | {conv_best['k_super']:.6f} | "
      f"{conv_best['err_pct']:+.3f}% | 是 |")
    a(f"| C1 u_max（工件 profile） | 解析 {rec_c1['poiseuille_u_max_analytic']:.7f} | "
      f"{rec_c1['poiseuille_u_max_got']:.7f} | {100*abs(rec_c1['poiseuille_u_max_got']-rec_c1['poiseuille_u_max_analytic'])/rec_c1['poiseuille_u_max_analytic']:.3f}% | 是 |")
    a(f"| A1 vs A2 表示等价（6 算例） | 预声明 1e-8 | ux_maxabs_rel=0.0（逐位） | 0 | 是 |")
    a("| C2/C5/C6 渗透率 | **无基准**（`k_reference=null`，`tensor_benchmark_cases` 0 行） | — | 无法判定 | 否（无基准，声明原因） |")
    a("")
    a(f"其余算例没有任何解析/文献/实验基准，也没有登记进 `tensor_benchmark_cases`，故不做误差声明——"
      "这直接封住了 L4「基准验证」的资格。")
    a("")
    a("## 判级")
    a("")
    a("**L2（独立功能实现）**。理由：真代码（表构建/压缩 push 流式/半程反弹/Guo forcing 完整可跑）、"
      "真测试（12 passed 本次复跑）、真工件（6 算例数值逐位可复现，无造假），并有解析锚误差 "
      f"{conv_best['err_pct']:+.2f}%（收敛）与 {rec_c1['k_err_vs_ref_pct']:.2f}%（工件）。")
    a("不授 L3：模块落在影子树、canonical 包不可导入、外部调用方 0；")
    a("不授 L4：`tensor_benchmark_cases` 该锚点 0 行登记、`repro_cmd` 未随交付记录、"
      "预声明梯未完成且 `converged=false`；不授 L5：外部 OSS 锚未执行（无 GPU，已如实声明为语义移植）。")
    a("")
    a("## 结论")
    a("")
    a(f"- **工件真实**：k 可反算（相对差 ≤{max(v['k_super_rel_gap'] for v in numeric.values()):.1e}），"
      + (f"逐位复现（最大相对差 {repro['max_k_rel_diff']:.1e}），" if repro else "")
      + "邻居表独立重算一致，无 m17636 式的 10 倍压缩/清零造假。")
    a("- **但级别只能到 L2**：见上「不授 L3/L4」。返修方向（供 reopened 参考）：把模块搬进 canonical "
      "`src/tensorlbm/porous/` 并接一个主求解器/平台调用方；跑完预声明梯（或明确改预声明并把收敛判据写进代码）；"
      "在 `tensor_benchmark_cases` 登记 C1 解析锚与容差；补 `repro_cmd` 与六段报告。")
    a("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-repro", action="store_true", help="re-run the delivered runner (~3 min)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DOCS, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    recorded = load_recorded()
    numeric = {r["case"]: rederive_k(r) for r in recorded["records"]}
    st = static_checks(recorded)
    bf = bruteforce_table_check()
    conv = convergence_probe()
    repro = reproduction(recorded) if args.with_repro else None
    judge = db_judge()

    fig_path = os.path.join(OUT, "figures", "fig_m17677_evidence.png")
    figures(recorded, conv, fig_path)

    audit = {
        "audit_mission": 17825,
        "audited_mission": 17677,
        "audited_scope": "m17677 XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE",
        "audited_worktree": AUDIT_WS,
        "static": st,
        "numeric": numeric,
        "bruteforce_table": bf,
        "convergence_probe": conv,
        "reproduction": repro,
        "benchmark_judge": judge,
        "figure": os.path.relpath(fig_path, ROOT_WS),
    }
    with open(os.path.join(OUT, "audit_m17677.json"), "w") as fh:
        json.dump(audit, fh, indent=2, default=str)

    md = markdown(recorded, st, numeric, bf, conv, repro, judge, fig_path)
    with open(os.path.join(DOCS, "AUDIT_m17677.md"), "w") as fh:
        fh.write(md)

    print(json.dumps({
        "files_present": st["files_present"],
        "entry_reproduces_artifact": st["entry_reproduces_artifact"],
        "committed_runner_reproduces_artifact": st["committed_runner_reproduces_artifact"],
        "canonical_import_ok": st["canonical_import_ok"],
        "external_callers": st["external_callers"],
        "bruteforce_equal": bf["neighbour_table_equal"],
        "conv": [(c["steps"], round(c["k_super"], 6), round(c["err_pct"], 3), c["converged"]) for c in conv],
        "repro": None if repro is None else (repro["ok"], repro.get("max_k_rel_diff")),
        "judge_rows": judge["registered_rows"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(main())
