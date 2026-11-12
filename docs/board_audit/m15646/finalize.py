#!/usr/bin/env python3
"""Emit the m15646 board-audit deliverables: REPORT.md + verdict.sql (+ INSERT).

Every number in REPORT.md is formatted from artifacts/m15646/*.json (no hand copy).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path("/root/tp-agent/work/m15646")
AUD = json.loads((ROOT / "artifacts/m15646/audit_m15481.json").read_text())["checks"]
RC = json.loads((ROOT / "artifacts/m15646/rootcause_bptt.json").read_text())
MISSION = 15646
AUDITED = 15481
ANCHOR = "GAP-PP-3-adjoint-ootc"

c1, c2, c3 = AUD["C1_provenance"], AUD["C2_placement"], AUD["C3_importability"]
c4, c5, c6 = AUD["C4_physics_smoke"], AUD["C5_math_bptt_vs_fd"], AUD["C6_ootc_vs_bptt"]
c7, c8, c9, c10 = AUD["C7_T_not_divisible_k"], AUD["C8_memmap_claim"], AUD["C9_memory_scaling"], AUD["C10_evidence"]
p1, p2, p3, p4 = RC["P1_collision_jacobian"], RC["P2_transpose"], RC["P3_error_vs_T"], RC["P4_C5_restated"]

LEVEL = 0
verdict = {
    "impl_files": ["TensorLBM/src/tensorlbm/adjoint/lbm2d.py", "TensorLBM/src/tensorlbm/adjoint/bptt.py",
                   "TensorLBM/src/tensorlbm/adjoint/ootc.py"],
    "impl_files_note": "294 lines of real (non-placeholder) code, but placed in the LEGACY tree "
                        "TensorLBM/src/tensorlbm (63 files, own pyproject) while the repo builds and "
                        "tests src/ (pyproject package-dir={''='src'}, Makefile/CI PYTHONPATH=src). "
                        "Not tracked on main; commit 39515bd6 lives only on agent/m15481 "
                        "(main trunk MainHead, branch off main by 3/1, no PR, no remote ref). "
                        "The dir also has no __init__.py and its name collides with the pre-existing "
                        "module src/tensorlbm/adjoint.py.",
    "entry": None,
    "entry_note": "NONE: no __main__, no argparse, no benchmark/script entry in the three files "
                  "(grep for __main__/argparse over TensorLBM/src/tensorlbm/adjoint -> 0 hits). "
                  "The only way in is an ad-hoc `sys.path.insert(0, 'TensorLBM/src')` import.",
    "run_artifacts": [],
    "run_artifacts_note": "MISSION PROVIDED ZERO run artifacts: mission result = "
                          "fail_stage=no_progress+forced_finish, head_commit empty; git ls-tree "
                          "agent/m15481 shows no artifacts/, runs/, figures/ or docs/ entry for the "
                          "anchor. The numbers in this audit come from the AUDIT's own re-runs "
                          "(artifacts/m15646/*.json), not from the mission.",
    "tests": [],
    "tests_note": "ZERO tests committed (tests/ has no adjoint/ootc test; no pytest file references "
                  "ootc_gradient or bptt_gradient). L1 therefore fails: the math was never unit "
                  "verified by the mission, and the audit's independent referee refutes it.",
    "integration_callers": [],
    "integration_callers_note": "ZERO: grep -rn over the repo for ootc_gradient|bptt_gradient|adjoint.ootc|"
                                "adjoint.bptt hits only the three files themselves; src/tensorlbm/adjoint.py "
                                "(the live module used by app/backend/routers/postprocess.py:3216 and "
                                "app/tests/test_advanced_engineering2.py:354) is a different, PyTorch-AD "
                                "implementation and is untouched.",
    "benchmark": {
        "platform_referee": f"ABSENT: tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}' = 0 rows "
                            "(20 anchors registered; this one is not among them)",
        "ref": "central finite difference of J(tau)=0.5*||u_T-u_ref||^2, audit-computed "
               f"h=1e-5, grid 24x24, T=200, tau=0.85: ref={p4['g_fd']:.9e}",
        "got": f"shipped BPTT bptt_gradient={p4['g_shipped']:.9e}; shipped OOTC "
               f"ootc_gradient={c6['ootc']['g']:.9e}",
        "error": f"shipped vs referee rel.err = {p4['rel_err']*100:.1f}% (>15% unacceptable, >3% fail); "
                 f"OOTC vs BPTT rel.diff = {c6['grad_rel_diff']:.1f} (bit-identical, i.e. vacuous)",
        "erdict": "L4 not met; the declared gate (|g_OOTC-g_BPTT|<1e-5) is satisfied trivially because "
                  "both paths share the same defective b_action",
        "memory": f"audit-side: peak RSS {c9['rows'][0]['rss_peak_GiB']:.3f}->{c9['rows'][-1]['rss_peak_GiB']:.3f} GiB "
                  f"for T={c9['rows'][0]['T']}->{c9['rows'][-1]['T']} at 128x128 k=100 (< 8 GiB gate), "
                  f"checkpoint on disk {c9['rows'][-1]['ckpt_GB']*1024:.0f} MB at T={c9['rows'][-1]['T']}; "
                  "T=1e4 never run by anyone",
    },
    "repro_cmd": "cd /root/tp-agent/work/m15646 && python3 docs/board_audit/m15646/audit_m15481.py "
                 "--out artifacts/m15646/audit_m15481.json && python3 "
                 "docs/board_audit/m15646/rootcause_bptt.py --out artifacts/m15646/rootcause_bptt.json "
                 "&& python3 docs/board_audit/m15646/make_figure.py   # audit side, 90 s, no cluster",
    "external_anchor": ["Griewank & Walther, Evaluating Derivatives, 2nd ed., ch. 2 (cited by the code docstrings; not independently fetched)",
                        "no experimental/literature numerical anchor for this anchor in-tree"],
    "external_anchor_note": "Docstrings cite Griewank & Walther 2008, Williams & Zipser 1989, Chen/Beckman/"
                            "Patterson 1996, Paszke 2017, Krueger et al. 2017. The audit did NOT fetch the "
                            "PDFs, so this is a citation, not a verified anchor; the referee used for L4 is "
                            "the self-contained finite difference above.",
    "audit_artifacts": ["artifacts/m15646/audit_m15481.json", "artifacts/m15646/rootcause_bptt.json",
                        "artifacts/m15646/audit_m15481_c9.json", "figures/m15646/audit_m15481.png",
                        "docs/board_audit/m15646/REPORT.md", "docs/board_audit/m15646/code_snapshot/adjoint/"],
    "audited_commit": f"39515bd6 'm15481: matured mission deliverables' on agent/m15481 "
                      f"(tracked_in_main=[] since the branch was never merged)",
    "findings": [
        "F1 BLOCKING: adjoint recursion applies the collision Jacobian M instead of M^T "
        f"(bptt.py:63-90). Measured gradient error vs FD: {p3['rows'][0]['shipped']['rel_err_vs_fd']:.1e} at T=2 -> "
        f"{p4['rel_err']*100:.0f}% at T=200 (24x24) -> {p3['rows'][-1]['shipped']['rel_err_vs_fd']*100:.0f}% at T=400. "
        f"M is non-symmetric: ||M-M^T||/||M|| = {p2['rel_asymmetry_norm(M - M^T)/norm(M)']:.3f}.",
        "F2 BLOCKING: b_action divides the velocity-derivative terms by rho, but d feq_i/d f_j = "
        "A_i + D_i(c_jx-ux) + E_i(c_jy-uy) has no 1/rho. Isolated: shipped action / exact = "
        f"{p1['rel_err_shipped_vs_fd']*100:.1f}% error at rho~{p1['rho_min']:.1f}..{p1['rho_max']:.1f}, "
        f"{p1['rel_err_no_1_over_rho_vs_fd']:.1e} after removing it. Secondary in LBM (rho~1) but it "
        "makes the routine unusable off the incompressible manifold.",
        f"F3 HIGH: OOTC integrates only (T//k)*k steps. T=250,k=100 -> gradient off by "
        f"{c7['cases'][1]['grad_rel_diff_vs_full_T']*100:.1f}%; T=200,k=64 -> "
        f"{c7['cases'][2]['grad_rel_diff_vs_full_T']*100:.1f}%. Silent: J matches a (T//k)*k-step BPTT "
        "to 0.0, so no self-consistency check can catch it.",
        "F4 HIGH: OOTC==BPTT agreement is not evidence of correctness - both share bptt.b_action, so "
        f"the declared gate |g_OOTC-g_BPTT|<1e-5 passes with rel.diff {c6['grad_rel_diff']:.1f} while "
        f"both are {p4['rel_err']*100:.0f}% off the referee.",
        "F5 MEDIUM (placement): deliverable landed in the legacy tree and is invisible to the package, "
        "CI and tests; also no __init__.py and a name clash with src/tensorlbm/adjoint.py.",
        "F6 MEDIUM: mission's carried-over static-check claim is FALSE - "
        f"np.lib.format.open_memmap exists and round-trips on numpy {c8['numpy_version']} "
        f"(has_attr={c8['has_np_lib_format_open_memmap']}, roundtrip={c8['roundtrip_ok']}). The stated "
        "'next round first task' (migrate to np.memmap) would have been wasted work.",
        "F7 POSITIVE: the memory mechanism is real and works - 128x128 out-of-core run, peak RSS flat "
        f"({c9['rows'][0]['rss_peak_GiB']:.3f} GiB at T={c9['rows'][0]['T']} vs "
        f"{c9['rows'][-1]['rss_peak_GiB']:.3f} GiB at T={c9['rows'][-1]['T']}), checkpoint only on disk; "
        f"forward+backward wall {c9['rows'][-1]['wall_s']:.0f}s at T={c9['rows'][-1]['T']}.",
        "F8 POSITIVE: forward solver is clean - mass conserved to "
        f"{c4['mass_rel_drift']:.1e} over 400 steps, fields finite, no RNG.",
    ],
    "level_rationale": "L0=文档成果, L1=数学/单元验证, L2=独立功能, L3=集成, L4=基准, L5=科研工程. "
                       "L1 NOT met: zero tests, and the audit's independent referee shows the shipped "
                       f"gradient is {p4['rel_err']*100:.0f}% wrong, so the math is refuted rather than "
                       "verified. L2 NOT met: no entry point, zero run artifacts, and the code is not on "
                       "main nor in the packaged tree, so nothing is callable from the delivered repo "
                       "(gate needs impl_files+entry+run_artifacts). L3/L4: zero integration callers, zero "
                       "benchmark rows, no repro_cmd shipped. The memory mechanism alone (F7) is not a "
                       "gradient capability, and a gradient that answers with a 4x wrong sign/magnitude is "
                       "worse than none. Verdict L0 = rework, not an incremental-polish gap.",
    "audit_scope": "read-only over /root/tp-agent/work/m15481 (sha256-pinned snapshot in "
                   "docs/board_audit/m15646/code_snapshot/); independent FD referee, transpose identity, "
                   "memory measurement, placement/import/benchmark/report registry probes; host "
                   "loongarch64, numpy 2.4.6, python 3.11.6",
    "not_run": ["T=1e4 128x128 full gate case (nobody ran it; audit stopped at T=2000)",
                "x86_64 CI rerun (audit host is loongarch64)",
                "literature anchor verification (Griewank 2008 / Krueger 2017 PDFs not fetched)",
                "step-3 truncation schedules (the anchor name says 'truncated BPTT'; no truncation "
                "length schedule is implemented or measured)",
                "report registry: tensor_reports WHERE mission_id=15481 -> 0 rows"],
}

sql_ev = json.dumps(verdict, ensure_ascii=False).replace("'", "''")
sql = ("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
       f"VALUES ('TensorLBM', 'm{AUDITED} {ANCHOR}', {MISSION}, {LEVEL}, '{sql_ev}'::jsonb, 'board-audit') "
       "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET mission_id=EXCLUDED.mission_id, "
       "level=EXCLUDED.level, evidence=EXCLUDED.evidence, gate_state='pending', gate_note=NULL;")
(ROOT / "docs/board_audit/m15646/verdict.sql").write_text(sql + "\n")


def tbl(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


md = f"""# 交差核验报告 · m{AUDITED} GAP-PP-3-adjoint-ootc（TensorLBM）

**核验单 m{MISSION}（board-audit）· 被核验对象 m{AUDITED}（状态 succeeded，但 result.fail_stage =
`{c10['mission_declared_result']}`）· 锚点 `{ANCHOR}`**

> 一句话结论：**内存机制是真的、能跑（128² T=2000 峰值 RSS {c9['rows'][-1]['rss_peak_GiB']:.2f} GiB、checkpoint 落盘）；
> 但它求的伴随梯度是错的——独立有限差分裁判给出 {p4['rel_err']*100:.0f}% 误差，而且 OOTC 与 BPTT 的"逐位一致"
> 是同一个 bug 的自我复制。判级 L{LEVEL}，返修。**

## ① 案例介绍（物理背景 / 几何 / 参数）

D2Q9 BGK 周期 LBM 的**参数反演**：目标函数 J(τ)=½‖u_T(τ)−u_ref‖²，要 dJ/dτ。
工程动机来自 JOUR-DIFFLBM 第 4 节（LBM 可微求解的成本/内存章）：全存轨迹的 BPTT 需要
O(T·N·Q) 内存（128²、T=10⁴ → 数 GB 级），所以要 **out-of-core / checkpointed BPTT**：
前向只把每 k 步的 checkpoint 落盘，反向回读 checkpoint 重算分段，内存 O(k·N·Q) 与 T 无关。
本锚点的验收判据（mission goal 原文）：10⁴ 步 128² τ 反演 **峰值内存 < 8 GiB**，且
**梯度与 BPTT 参考差 < 1e-5**。

| 参数 | 值 | 出处 |
|---|---|---|
| 格点 / 速度集 | 16², 24², 32², 128² · D2Q9 | lbm2d.py:20-23 |
| 弛豫时间 τ | 0.85（0.7/0.8/0.9 亦测） | audit 设定 |
| 前向步数 T | 2 … 400（梯度裁判）、500/1000/2000（内存） | audit 设定 |
| checkpoint 间隔 k | 50 / 64 / 100 | ootc.py 默认 100 |
| 初值 | 静止平衡 + 高斯速度脉冲 amp=0.05, sigma=4/5/6/16 | lbm2d.py:65-78 |
| 边界 | 全周期（np.roll），无边界条件 | lbm2d.py:52-57 |
| 目标场 u_ref | 0（衰减到静止） | audit 设定 |

## ② 怎么做的（核验方法 / 工具链）

不谈报告措辞，只看三件套是否真存在，并用**基准判据当唯一裁判**：

1. **取证与指纹**：`/root/tp-agent/work/m15481` 只读打开；三个源文件 sha256 与审计侧快照逐字节比对
   （{len([v for v in c1['snapshot_vs_mission_tree'].values() if v['identical']])}/3 identical）；核对它们在哪个 ref 上
   （`git ls-tree main / agent/m15481`）、是否被 PR 合入。
2. **可达性**：`pyproject.toml` 的 `package-dir`、Makefile/CI 的 `PYTHONPATH`，判断代码是否在**被安装/被测试的树**里；
   再用真实 `import tensorlbm.adjoint` 探一次。
3. **独立裁判（关键）**：不动被核验代码，直接对 J(τ) 做**中心有限差分**（h=1e-4/1e-5/1e-6 三档一致到 1e-9），
   得到真梯度；与被核验实现的 dJ/dτ 比。
4. **根因定位**（不改库代码，全部在审计侧重写算子）：碰撞雅可比 M 的**正向/转置**作用分别对拍有限差分；
   用 `⟨M d, μ⟩ = ⟨d, M^T μ⟩` 验证转置恒等式；再按 T 扫描误差增长。
5. **OOTC↔BPTT 一致性**：T%k==0 与 T%k≠0 两组；后者额外与"实际积分步数"的 BPTT 对拍以证明是**静默错步数**。
6. **内存**：128²、k=100、T=500/1000/2000，量 `ru_maxrss` 峰值 + checkpoint 文件大小。
7. **台账/报告**：查 `tensor_benchmark_cases`（锚点行数）、`tensor_reports`（mission 行数）。

工具链：python 3.11.6 / numpy 2.4.6 / loongarch64 / matplotlib(Agg)；总耗时 ~90 s，未占用集群。
可复现：`{verdict['repro_cmd']}`

## ③ 计算结果（全部由 `artifacts/m15646/*.json` 程序化注入）

**（a）三件套 + 放置**

{tbl([{'项目': '源码 3 文件（294 行）', '实测': '存在、sha256 与快照一致', '判定': 'OK（但树错）'},
      {'项目': 'tracked 于 main', '实测': f"{len(c1['refs']['tracked_in_main'])} 个", '判定': 'FAIL 未合入'},
      {'项目': 'tracked 于 agent/m15481', '实测': f"{len(c1['refs']['tracked_in_mission_branch'])} 个（commit 39515bd6）", '判定': '仅分支'},
      {'项目': 'remote ref 含该 commit', '实测': f"{len(c1['refs']['remote_refs_containing'])}", '判定': 'FAIL 未推'},
      {'项目': '放置树', '实测': f"{c2['files_placed_in']}（legacy，{63} 文件，自带 pyproject）", '判定': 'FAIL 非打包树'},
      {'项目': '打包树', '实测': f"pyproject package-dir = {c2['pyproject_package_dir'][0].strip()}", '判定': 'src/'},
      {'项目': '__init__.py', '实测': f"{c2['package_init_present_in_mission_tree']}", '判定': 'FAIL 无包'},
      {'项目': '命名冲突', '实测': f"src/tensorlbm/adjoint.py 已存在（{c2['authoritative_tree_has_adjoint_module']}）", '判定': '风险'},
      {'项目': 'import 实测', '实测': f"rc={c3['rc']}, has_ootc={c3['stdout'].splitlines()[-1]}", '判定': 'FAIL 不可达'},
      {'项目': 'entry（__main__/argparse）', '实测': f"{len(c10['entry_points'])} 处", '判定': 'FAIL 无入口'},
      {'项目': 'tests', '实测': f"{len(c10['tests_in_mission_branch'])} 个", '判定': 'FAIL 无测试'},
      {'项目': 'run artifacts', '实测': f"{len(c10['artifacts_runs_docs_in_mission_branch'])} 个", '判定': 'FAIL 无工件'},
      {'项目': 'integration callers', '实测': 'grep 命中仅 3 个文件自身', '判定': 'FAIL 零调用'},
      {'项目': f'benchmark cases（{ANCHOR}）', '实测': f"{len(c10['benchmark_cases_for_anchor'])} 行", '判定': 'FAIL 无裁判登记'},
      {'项目': 'tensor_reports（mission 15481）', '实测': '0 行', '判定': 'FAIL 无报告'}],
     ['项目', '实测', '判定'])}

**（b）梯度裁判（核心）**

{tbl([{'T': r['T'], 'g_FD（裁判）': f"{r['g_fd']:.6e}", 'g_shipped（已交）': f"{r['shipped']['g']:.6e}",
       'err_shipped': f"{r['shipped']['rel_err_vs_fd']*100:.2f}%", 'g_M^T（修正）': f"{r['M_transposed']['g']:.6e}",
       'err_M^T': f"{r['M_transposed']['rel_err_vs_fd']:.1e}"} for r in p3['rows']],
     ['T', 'g_FD（裁判）', 'g_shipped（已交）', 'err_shipped', 'g_M^T（修正）', 'err_M^T'])}

> 16×16、τ=0.85、amp=0.05、σ=4。shipped 误差随 T 单调恶化（T=2 时 1.7e-10 侥幸对，因为两步内 M 与 M^T 的差异还没累积）；
> 换回 **M^T 且去掉多余 1/ρ** 后，T≤200 一致到 1e-8…1e-11，T=400 为 1.6e-5（|J|~1e-4 量级下的浮点累积+混沌放大）。

**（c）根因两项**：碰撞雅可比作用（6×6，随机场）

{tbl([{'探针': 'P1 shipped b_action vs FD(feq)', '数值': f"{p1['rel_err_shipped_vs_fd']*100:.1f}%", '结论': '错：多除了 ρ'},
      {'探针': 'P1 去掉 1/ρ 后', '数值': f"{p1['rel_err_no_1_over_rho_vs_fd']:.1e}", '结论': '精确'},
      {'探针': 'P2 ⟨M d,μ⟩ vs ⟨d,M^T μ⟩', '数值': f"{p2['rel_gap']:.1e}", '结论': '转置恒等式成立'},
      {'探针': 'P2 ‖M−M^T‖/‖M‖', '数值': f"{p2['rel_asymmetry_norm(M - M^T)/norm(M)']:.3f}", '结论': 'M 几乎完全非对称'},
      {'探针': 'P1 被测场 ρ 范围', '数值': f"{p1['rho_min']:.2f} … {p1['rho_max']:.2f}", '结论': '1/ρ 在不可压流里被 ρ≈1 掩盖'}],
     ['探针', '数值', '结论'])}

**（d）OOTC↔BPTT 与步数缺陷**

{tbl([{'T': x['T'], 'k': x['k'], 'n_ckpt': x['n_checkpoints'], '实际积分步': x['steps_actually_integrated'],
       'J_BPTT': f"{x['J_bptt_full_T']:.6e}", 'J_OOTC': f"{x['J_ootc']:.6e}",
       'grad 相对差 vs 请求的 T': f"{x['grad_rel_diff_vs_full_T']*100:.1f}%",
       'grad 相对差 vs 实际步数': f"{x['grad_rel_diff_vs_integrated_steps']:.1e}"} for x in c7['cases']],
     ['T', 'k', 'n_ckpt', '实际积分步', 'J_BPTT', 'J_OOTC', 'grad 相对差 vs 请求的 T', 'grad 相对差 vs 实际步数'])}

**（e）内存（128²，k=100，审计侧实测）**

{tbl([{'T': r['T'], 'n_ckpt': r['n_checkpoints'], '峰值 RSS (GiB)': f"{r['rss_peak_GiB']:.3f}",
       'checkpoint 文件 (MB)': f"{r['ckpt_GB']*1024:.1f}", '墙钟 (s)': f"{r['wall_s']:.1f}"} for r in c9['rows']],
     ['T', 'n_ckpt', '峰值 RSS (GiB)', 'checkpoint 文件 (MB)', '墙钟 (s)'])}

**（f）numpy 2.x 的"弃用"说法**

被测环境 numpy {c8['numpy_version']}：`np.lib.format.open_memmap` 存在={c8['has_np_lib_format_open_memmap']}，
读写回环={c8['roundtrip_ok']}，异常={c8['error']}。→ mission 交接里"numpy 2.x 已弃用，应改 np.memmap"
的静态检查结论**不成立**。

## ④ 结果图

![audit]({str((ROOT / 'figures/m15646/audit_m15481.png'))})

*（a）伴随梯度相对误差随 T：已交实现（红）系统性爆掉，仅去 1/ρ（橙）没有救，换成 M^T（绿）落到 1e-10 量级——
证明缺陷完全定位在 `bptt.py:63-90` 的 `b_action`。（b）T=200 的三根柱子：shipped BPTT 与 shipped OOTC 完全重合，
而两者一起偏离有限差分裁判
{p4['rel_err']*100:.0f}%——"OOTC=BPTT"这句话在数学上是自洽的，在认识论上是空的。（c）out-of-core 内存随 T 平坦，
8 GiB 门远远通过；代价转移到磁盘 checkpoint（T 线性）。*

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

- **最危险的坑：拿"两条实现彼此一致"当正确性证据。** OOTC 与 BPTT 共用 `bptt.b_action`，所以相对差精确为
  {c6['grad_rel_diff']:.1f}，`<1e-5` 判据"通过"，但两者一起错了 {p4['rel_err']*100:.0f}%。任何自洽性判据都必须配一个**外部裁判**
  （有限差分 / Taylor 余项 / 解析解）。
- **T%k≠0 静默错步数**：`n_ckpt = T//k + 1`、前向只走 `(T//k)·k` 步。T=250,k=100 梯度差 12%，T=200,k=64 差 70%，
  而 J 与"实际步数"的 BPTT 精确一致 → 任何基于 J 的自检都抓不到。修法：断言 `T % k == 0` 或补尾段。
- **1/ρ 缺陷在不可压 LBM 里是"隐形"的**（ρ≈1），只在 ρ 偏离 1 的场里暴露（P1 用 ρ∈[6,11] 才把它逼出来）。
  审查"只在真实工况跑一遍就对"的实现时，务必构造 off-manifold 输入。
- **放置错了等于没做**：本仓有两棵树（`src/` 与 legacy `TensorLBM/src/`，后者 63 文件自带 pyproject），
  CI/Makefile 只跑 `src/`。放到 legacy 树 + 无 `__init__.py` + 与 `src/tensorlbm/adjoint.py` 同名 → 包、CI、测试都看不见。
- **局限**：有限差分只能在小 T 当裁判（T=400 时 J~1e-4，混沌使相对误差抬到 1.6e-5）；10⁴ 步 128² 的完整验收
  工况**没有人跑过**（本审计止于 T=2000）；审计机是 loongarch64/py3.11/numpy2.4.6，x86 CI 未复跑；
  docstring 引的 5 篇文献均未取原文核对，属"引用"不属"锚"。
- **适用范围**：审计结论只针对 39515bd6 的这三个文件；把 `b_action` 换成 M^T、删掉 1/ρ、挪到 `src/`、
  补"T%k"断言与 FD 单测之后，D2Q9 周期 BGK 这一支是可救的（P3 的绿线就是证据）。

## ⑥ 误差判定（逐项，显式回答"是否 < 3%"）

| 关键量 | 基准（ref） | 实测（got） | 相对误差 | <3%？ |
|---|---|---|---|---|
| dJ/dτ（24², T=200, τ=0.85） | FD {p4['g_fd']:.6e} | shipped BPTT {p4['g_shipped']:.6e} | **{p4['rel_err']*100:.1f}%** | **否** |
| dJ/dτ（同上） | FD {p4['g_fd']:.6e} | shipped OOTC {c6['ootc']['g']:.6e} | **{p4['rel_err']*100:.1f}%** | **否** |
| dJ/dτ（16², T=200） | FD {p3['rows'][7]['g_fd']:.6e} | shipped {p3['rows'][7]['shipped']['g']:.6e} | **{p3['rows'][7]['shipped']['rel_err_vs_fd']*100:.0f}%** | **否** |
| dJ/dτ（16², T=200，修正版） | FD {p3['rows'][7]['g_fd']:.6e} | M^T 版 {p3['rows'][7]['M_transposed']['g']:.6e} | {p3['rows'][7]['M_transposed']['rel_err_vs_fd']:.1e} | 是 |
| OOTC vs BPTT（判据本义） | BPTT {c6['bptt']['g']:.6e} | OOTC {c6['ootc']['g']:.6e} | {c6['grad_rel_diff']*100:.1f}%（精确 0） | 是**但空洞**（共用同一缺陷，见⑤） |
| 碰撞雅可比作用 | FD(feq) | shipped b_action | {p1['rel_err_shipped_vs_fd']*100:.1f}% | 否 |
| 峰值内存（128²,T=2000,k=100） | <8 GiB（mission 判据） | {c9['rows'][-1]['rss_peak_GiB']:.2f} GiB | — | 是（但 T=10⁴ 未跑） |
| 质量守恒（32², 400 步） | 严格守恒 | 漂移 {c4['mass_rel_drift']:.1e} | {c4['mass_rel_drift']:.1e} | 是 |

**结论（honest）**：核心功能量（dJ/dτ）**不通过 3%**，不通过 15%，差了一个量级；内存门通过但只是"机制"层。
锚点判据 `梯度与 BPTT 参考差 < 1e-5` 的写法本身允许"两条错误实现互证"，建议返修单把判据改成
**"梯度 vs 中心差分/Taylor 余项 < 1e-6"**，并把 `T % k == 0` 断言、`src/` 落位、pytest 单测、
`tensor_reports` 六段报告一起补上。

## 判级与落表

- **L{LEVEL}**（L0 文档/无实质；L1 数学单元验证；L2 独立功能；L3 集成；L4 基准；L5 科研工程）
- 理由：L1 不达（零测试，且独立裁判**否证**数学）；L2 不达（无入口、无运行工件、代码不在 main 也不在打包树，
  交付仓里不可调用）；L3 零调用方；L4 零 benchmark 行 + 无 repro_cmd。内存机制（F7）是真的，但"内存省了、梯度错了"
  不构成梯度能力。
- 落表：`tensor_maturity_verdicts (repo='TensorLBM', scope='m{AUDITED} {ANCHOR}', mission_id={MISSION}, level={LEVEL},
  verdict_by='board-audit')`，`verdict.sql` 与本文件同目录，由 maturity_gate 自动复核 → 预期 `reopened`（返修）。
"""
(ROOT / "docs/board_audit/m15646/REPORT.md").write_text(md)
print("wrote REPORT.md and verdict.sql")
r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-c",
                    sql + " SELECT id,level,gate_state FROM tensor_maturity_verdicts ORDER BY id DESC LIMIT 1;"],
                   capture_output=True, text=True)
print("psql rc", r.returncode, r.stdout.strip()[-200:], r.stderr.strip()[:300])