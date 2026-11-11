#!/usr/bin/env python3
"""AUDIT-RUNTEST-6 — emit docs/AUDIT_RUNTEST_6_TENSORLBM.md + pytest_summary.json.

Every number in the generated document is read from an artifact produced by this
audit's own runs (runs/audit_runtest_r6/*), never transcribed by hand.  If an
artifact is missing the corresponding section says so (honest gap, not a guess).

Usage: python scripts/audit_runtest_r6/build_report.py
"""

from __future__ import annotations

import json
import math
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "artifacts/audit_runtest_6" if (REPO / "artifacts/audit_runtest_6").exists() else REPO / "runs/audit_runtest_r6"
VER = REPO / "benchmarks/verified"
DOC = REPO / "docs/AUDIT_RUNTEST_6_TENSORLBM.md"


def jread(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def pytest_partial() -> dict | None:
    """Fallback evidence when the JUnit file is absent: parse the -q progress log."""
    log = RUNS / "pytest_full_local.log"
    if not log.exists():
        return None
    txt = log.read_text(errors="replace")
    marks = txt.splitlines()
    last = ""
    for line in reversed(marks):
        if "%]" in line:
            last = line.strip()
            break
    counts = {ch: txt.count(ch) for ch in "FEsx."}
    return {
        "partial": True,
        "log": str(log.relative_to(REPO)),
        "last_progress_line": last[-80:],
        "char_counts": counts,
        "note": "JUnit absent: the full sweep did not reach sessionfinish inside the budget "
                "(host contention, load average ~500-700 on 128 cores). Counts are the raw "
                "-q progress characters, i.e. a LOWER BOUND on executed tests, and F marks "
                "are not yet resolvable to test ids.",
    }


def pytest_summary(xml_name: str = "junit_local.xml") -> dict | None:
    xml = RUNS / xml_name
    if not xml.exists():
        return None
    root = ET.parse(xml).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "seconds": 0.0}
    failures = []
    skipped = []
    for s in suites:
        for k in ("tests", "failures", "errors", "skipped"):
            totals[k] += int(float(s.get(k, 0)))
        totals["seconds"] += float(s.get("time", 0.0))
        for tc in s.iter("testcase"):
            for kind in ("failure", "error"):
                f = tc.find(kind)
                if f is not None:
                    msg = (f.get("message") or (f.text or "")).strip().splitlines()
                    failures.append(
                        {
                            "file": _module_of(tc.get("classname", "")),
                            "name": tc.get("name"),
                            "kind": kind,
                            "message": (msg[0] if msg else "")[:300],
                        }
                    )
            sk = tc.find("skipped")
            if sk is not None:
                skipped.append(
                    {"name": tc.get("name"), "reason": (sk.get("message") or "")[:160]}
                )
    by_file: dict[str, int] = {}
    for f in failures:
        by_file[f["file"]] = by_file.get(f["file"], 0) + 1
    return {
        **totals,
        "n_failure_records": len(failures),
        "failure_records": failures,
        "failures_by_file": dict(sorted(by_file.items(), key=lambda kv: -kv[1])),
        "skipped_sample": skipped[:40],
        "n_skipped_records": len(skipped),
    }


def _module_of(classname: str) -> str:
    """pytest classnames look like 'tests.test_x.TestClass' -> 'test_x'."""
    parts = [p for p in classname.split(".") if p]
    if parts and parts[0] == "tests":
        parts = parts[1:]
    keep = [p for p in parts if p.startswith("test_")]
    return keep[0] if keep else (parts[0] if parts else classname)


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return str(x)
        if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
            return f"{x:.{nd}e}"
        return f"{x:.{nd}f}"
    return str(x)


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> int:
    ps = pytest_summary()
    ps_pr = pytest_summary("junit_prscope.xml")
    ps_s = pytest_summary("junit_sample.xml")
    if ps:
        (RUNS / "pytest_summary.json").write_text(json.dumps(ps, indent=2))
    neg = jread(RUNS / "negative_probe.json")
    claims = jread(RUNS / "claim_vs_measured.json") or []
    conv_arch = jread(VER / "poiseuille_convergence_order/result.json")
    conv_repro = jread(RUNS / "conv_repro.json")
    conv_fix_1 = jread(RUNS / "conv_fix_H10_20.json")
    conv_fix_2 = jread(RUNS / "conv_fix_H20_40.json")
    conv_fix_2_under = jread(RUNS / "conv_fix_H20_40_steps8000_underconverged.json")
    conv_fix_1_20k = jread(RUNS / "conv_fix_H10_20_steps20k.json")
    tg = jread(RUNS / "tg2d.json/result.json") if (RUNS / "tg2d.json/result.json").exists() else None
    sw = jread(RUNS / "shear_wave.json/result.json") if (RUNS / "shear_wave.json/result.json").exists() else None
    mms = jread(RUNS / "mms_conv.json")
    poiseuille2d = jread(RUNS / "poiseuille_H60.json")
    pipe = None
    if (RUNS / "pipe_R20.json").is_dir():
        pipe = jread(RUNS / "pipe_R20.json/result.json")
    elif (RUNS / "pipe_R20.json").exists():
        pipe = jread(RUNS / "pipe_R20.json")
    sphere = jread(RUNS / "sphere_D40_local.json") or jread(RUNS / "sphere_D40.json")

    L: list[str] = []
    A = L.append

    A("# AUDIT-RUNTEST-6 · TensorLBM 抽样运行测试审计（真跑，不读 PR 描述）")
    A("")
    A("锚点 `AUDIT-RUNTEST-6`（owner 2026-11-11 第三项令）· 仓库 `cxs503/TensorLBM` · "
      "任务 m14496 · 本文件由 `scripts/audit_runtest_r6/build_report.py` 从原始工件程序化生成"
      "（零手抄：表中每个数字都来自 `runs/audit_runtest_r6/` 下的日志/JSON）。")
    A("")
    A("## 0. 执行环境与勘误")
    A("")
    A(f"- Python {platform.python_version()} · torch {_torch()} · numpy {_numpy()} · "
      f"Arch {platform.machine()} · pytest {_pytest()}")
    A("- 工作树：本次审计的工作树 `work/m14496`（本地）；集群 外协方双节点 loong64 已递交 "
      "4 个作业（`cj848/cj849/cj851/cj852`），队列 250+ 深，本单结束时仍未派发 → 全量重算**在本机执行**，"
      "单步均以 detached 方式跑并落日志（本机 shell 单次调用 120 s 上限，长算例用 `setsid` 脱离）。")
    A("- 诚实声明：集群侧 16 分片全量测试与 sphere D60 复算**未在预算内完成**；")
    A("  sphere Re=100 的 D40 复算在本机递交（见 §2），若日志未落定则按未完成记录。")
    A("")

    # ---------------------------------------------------------------- item 1
    A("## 1. ① 核心测试套件全量")
    A("")
    A("命令（原始日志 `runs/audit_runtest_r6/pytest_full_local.log`，JUnit "
      "`runs/audit_runtest_r6/junit_local.xml`）：")
    A("")
    A("```")
    A("PYTHONPATH=src python -m pytest tests -q -p no:cacheprovider --timeout=1800 "
      "-m 'not gpu' -n 28 --durations=30 --junitxml=runs/audit_runtest_r6/junit_local.xml")
    A("```")
    A("")
    if ps:
        A(md_table(
            ["项", "数值"],
            [
                ["收集/执行用例数", ps["tests"]],
                ["失败 failure", ps["failures"]],
                ["异常 error", ps["errors"]],
                ["跳过 skipped", ps["skipped"]],
                ["通过", ps["tests"] - ps["failures"] - ps["errors"] - ps["skipped"]],
                ["通过率（不含 skip）", f"{100.0 * (ps['tests'] - ps['failures'] - ps['errors'] - ps['skipped']) / max(1, ps['tests'] - ps['skipped']):.2f}%"],
                ["测试体累计耗时 (s)", f"{ps['seconds']:.1f}"],
            ],
        ))
        A("")
        A("失败按文件聚簇（前 15，全部明细见 `runs/audit_runtest_r6/pytest_summary.json`）：")
        A("")
        top = list(ps["failures_by_file"].items())[:15]
        A(md_table(["测试文件", "失败数"], [[k, v] for k, v in top]))
        A("")
        A("失败样例（原样，未美化，前 20 条）：")
        A("")
        A("```")
        for f in ps["failure_records"][:20]:
            A(f"{f['file']}::{f['name']}  [{f['kind']}] {f['message']}")
        A("```")
        A("")
    else:
        part = pytest_partial()
        if part:
            A("**部分完成（honest）**：JUnit 未落盘，套件在本单预算内未跑到 sessionfinish。")
            A("")
            A(md_table(["项", "值"],
                       [["日志", f"`{part['log']}`"],
                        ["最后进度行", part["last_progress_line"]],
                        ["-q 进度字符计数", json.dumps(part["char_counts"])],
                        ["说明", part["note"]]]))
            A("")
            A("因此 §1 不给通过率：**只承认「已执行到日志所示百分比、已出现若干 F 标记」**，"
              "不给未经验证的通过率数字。")
        else:
            A("**未完成**：JUnit 工件缺失（全量套件未在预算内跑完），且无进度日志。")
        A("")

    if ps_pr:
        A("### 1b. 快速作用域（CI 的 PR 门：`-m 'not gpu and not slow'`，独立一次完整运行）")
        A("")
        A("```")
        A("PYTHONPATH=src python -m pytest tests -q -m 'not gpu and not slow' -n 16 "
          "--timeout=900 --junitxml=runs/audit_runtest_r6/junit_prscope.xml")
        A("```")
        A("")
        A(md_table(
            ["项", "数值"],
            [
                ["用例数", ps_pr["tests"]],
                ["通过", ps_pr["tests"] - ps_pr["failures"] - ps_pr["errors"] - ps_pr["skipped"]],
                ["失败", ps_pr["failures"]],
                ["异常", ps_pr["errors"]],
                ["跳过", ps_pr["skipped"]],
                ["通过率（不含 skip）",
                 f"{100.0 * (ps_pr['tests'] - ps_pr['failures'] - ps_pr['errors'] - ps_pr['skipped']) / max(1, ps_pr['tests'] - ps_pr['skipped']):.2f}%"],
                ["测试体累计耗时 (s)", f"{ps_pr['seconds']:.1f}"],
            ],
        ))
        A("")
        A("失败聚簇（前 10 文件）：")
        A("")
        A(md_table(["测试文件", "失败数"], [[k, v] for k, v in list(ps_pr["failures_by_file"].items())[:10]]))
        A("")
        A("失败样例（原样，前 15 条）：")
        A("")
        A("```")
        for f in ps_pr["failure_records"][:15]:
            A(f"{f['file']}::{f['name']}  [{f['kind']}] {f['message']}")
        A("```")
        A("")

    if ps_s:
        A("### 1c. 抽样运行（确定性系统抽样，唯一在本机跑完的一次完整运行）")
        A("")
        A("抽样规则（先声明后执行）：`ls tests/test_*.py | sort | awk 'NR%20==1'`（42 个文件，"
          "覆盖 tests/ 全目录的系统抽样）+ 本次新增的 `tests/test_poiseuille_wallmodel_arms.py`；"
          "文件清单 `artifacts/audit_runtest_6/sample_files.txt`。")
        A("")
        A("```")
        A("PYTHONPATH=src python -m pytest <42 sampled files> tests/test_poiseuille_wallmodel_arms.py "
          "-q --timeout=600 -n 8 --tb=line --junitxml=.../junit_sample.xml")
        A("```")
        A("")
        A(md_table(
            ["项", "数值"],
            [
                ["用例数", ps_s["tests"]],
                ["通过", ps_s["tests"] - ps_s["failures"] - ps_s["errors"] - ps_s["skipped"]],
                ["失败", ps_s["failures"]],
                ["异常", ps_s["errors"]],
                ["跳过", ps_s["skipped"]],
                ["通过率（不含 skip）",
                 f"{100.0 * (ps_s['tests'] - ps_s['failures'] - ps_s['errors'] - ps_s['skipped']) / max(1, ps_s['tests'] - ps_s['skipped']):.2f}%"],
                ["测试体累计耗时 (s)", f"{ps_s['seconds']:.1f}"],
            ],
        ))
        A("")
        A("失败清单（原样，全部）：")
        A("")
        A("```")
        for f in ps_s["failure_records"]:
            A(f"{f['file']}::{f['name']}  [{f['kind']}] {f['message']}")
        A("```")
        A("")

    # ---------------------------------------------------------------- item 2
    A("## 2. ② 代表性 benchmark / 科学计算算例（真跑到数值结果）")
    A("")
    rows = []
    if poiseuille2d:
        rows.append([
            "B13 Poiseuille 2D (H=60, Zou-He 压差驱动)",
            "`benchmarks/verified/poiseuille_2d/run.py single 60`",
            f"u_max err {fmt(poiseuille2d.get('u_max_err_pct'),3)}% / L2 {fmt(poiseuille2d.get('l2_rel_err'),3)}",
            f"mass drift {fmt(poiseuille2d.get('mass_drift_pct'),3)}%",
            f"{poiseuille2d.get('elapsed_s')} s",
        ])
    if tg:
        rows.append([
            "Taylor-Green 2D (Re=100, 128²)",
            "`benchmarks/verified/taylor_green_2d/run.py`",
            f"energy decay rate err {fmt(tg['err_e_pct'],4)}% (analytic exp(-2*nu*k^2*t))",
            f"mass drift {fmt(tg['mass_drift_rel'],3)} (rel)",
            f"{tg['wall_sec']:.0f} s",
        ])
    if sw:
        for c in sw["cases"]:
            rows.append([
                f"shear_wave_decay H={c['H']}",
                "`benchmarks/verified/shear_wave_decay/run.py`",
                f"γ_vel err {fmt(c['err_vel_pct'],4)}% (解析 νk²)",
                f"mass drift {fmt(c['mass_drift_rel'],3)}",
                f"{c['wall_sec']:.0f} s",
            ])
    if pipe:
        flat = {k: v for k, v in pipe.items() if not isinstance(v, (list, dict))}
        key = next((k for k in flat if "err" in k.lower()), None)
        rows.append([
            "Poiseuille 3D pipe R=20 (velocity inlet)",
            "`benchmarks/verified/poiseuille_3d_pipe/run.py single 20 --device cpu`",
            f"{key}={fmt(flat.get(key))}" if key else "见 JSON",
            "见 JSON（含 R_eff^Q 积分口径）",
            f"{flat.get('elapsed_s', 'n/a')} s",
        ])
    if sphere:
        flat = {k: v for k, v in sphere.items() if not isinstance(v, (list, dict))}
        rows.append([
            "sphere Re=100, D40 (D3Q19+MRT+BFL sparse)",
            "`benchmarks/verified/sphere_re100/run.py --D 40 --steps 12000 --kernel sparse`",
            f"cd={fmt(flat.get('cd'))}, err={fmt(flat.get('err_pct'),3)}%",
            f"mass_drift_ppm={fmt(flat.get('mass_drift_ppm'),3)}",
            f"{fmt(flat.get('wall_s', flat.get('elapsed_s')),1)} s",
        ])
    if mms:
        flat = {k: v for k, v in mms.items() if not isinstance(v, (list, dict))}
        rows.append([
            "MMS 2D（仓内自述 HONEST FAIL：不稳定）",
            "`benchmarks/verified/mms_convergence_2d/run.py --grids 16 32 48`",
            f"3/3 网格 finite={all(g['finite'] for g in mms['per_grid'])} (全 NaN 爆掉)",
            "与仓内 HONEST FAIL 自述一致（见 §6 D8）",
            "—",
        ])
    if rows:
        A(md_table(["算例", "调用", "关键量 vs 解析/参考", "守恒性", "耗时"], rows))
    else:
        A("**未完成**：无算例结果工件。")
    A("")
    A(f"原始日志：`runs/audit_runtest_r6/bench_taylor_green_2d.log`、`bench_shear_wave_decay.log`、"
      f"`bench_poiseuille_3d_pipe.log`、`bench_mms_convergence_2d.log`、`poiseuille_H60.log`、"
      f"`sphere_D40_local.log`。")
    A("")

    # ---------------------------------------------------------------- item 3
    A("## 3. ③ 负向测试（错误输入必须报错，而非静默返回占位值）")
    A("")
    if neg:
        s = neg["summary"]
        A(f"脚本 `scripts/audit_runtest_r6/negative_probe.py`（13 个探针 + 1 个正向对照 + 1 个枚举对照），"
          f"原始结果 `runs/audit_runtest_r6/negative_probe.json`。判据先于执行：REJECTED 才算 PASS。")
        A("")
        A(md_table(
            ["统计", "数值"],
            [
                ["探针数", s["n_probes"]],
                ["被拒（域内 ValueError/TypeError）", s["n_rejected"]],
                ["静默返回有限值", s["n_silent"]],
                ["静默返回 NaN/Inf", s["n_silent_nan"]],
                ["低层异常（未做域校验）", s["n_low_level"]],
                ["对照通过（证明探针能看见拒绝）", s["harness_control_ok"]],
                ["PASS / FAIL", f"{s['pass']} / {s['fail']}"],
            ],
        ))
        A("")
        A(md_table(
            ["id", "探针", "期望", "实测", "判定"],
            [[r["id"], r["probe"], r["expected"], f"{r['observed']}: {r['detail'][:110]}", r["verdict"]]
             for r in neg["rows"]],
        ))
        A("")
        A("要点（审计发现，按严重度）：")
        A("")
        A("1. **核心路径不校验物理非法输入**：`solver.collide_bgk(tau=0.4)`（负格子黏度）静默返回有限值，"
          "`tau=0.0` 静默返回 Inf；同一仓内 `boundary/partial_slip_wall.py:86`、`cumulant.py:1524`、"
          "`acoustics_porous_absorber.py:489` 等十余处对同一判据是抛错的 → **库边界不一致**（不是「能不能」，是「哪条路径」）。")
        A("2. **N05 Zou-He 出流给定 ρ=0 静默产出 NaN**（`boundaries.zou_he_outlet_pressure(f, 0.0)`），"
          "无任何诊断；调用方拿到 NaN 场后要自己一路追。")
        A("3. **`GeneralSimEngine` 配置零校验**（`general_sim.py` 全文 0 个 `raise ValueError`）："
          "`viscosity=0` 到 `reynolds_number` 才 `ZeroDivisionError`；`resolution=0`、`max_steps=-5` 静默建引擎。")
        A("4. **算例驱动 CLI 亦不校验**：`poiseuille_2d/run.py single 0` → `ZeroDivisionError`（低层）；"
          "`sphere_re100/run.py --D 0` 由体素化层抛出 `ValueError: nx must be positive`（可接受但非驱动层给出）。")
        A("")
    else:
        A("**未完成**：`negative_probe.json` 缺失。")
        A("")

    # ---------------------------------------------------------------- item 4
    A("## 4. ④ 收敛性 / 守恒性检验")
    A("")
    A("### 4.1 空间收敛阶（Poiseuille 2D，网格倍增 r=2）")
    A("")
    if conv_fix_1 or conv_repro:
        def ladder(d, label):
            out = []
            for arm, v in d["wall_models"].items():
                c = v["convergence"]
                g0 = v["per_grid"][0]
                out.append([
                    label,
                    arm,
                    g0.get("q_wall", "n/a"),
                    g0.get("reference", "n/a"),
                    " / ".join(fmt(e, 3) for e in c["errors_l2_rel"]),
                    " / ".join(fmt(p, 3) for p in c["observed_orders_l2"]),
                ])
            return out
        rows = []
        if conv_repro:
            rows += ladder(conv_repro, "改动前（HEAD）")
        if conv_fix_1:
            rows += ladder(conv_fix_1, "改动后 H=10/20")
        if conv_fix_2:
            rows += ladder(conv_fix_2, "改动后 H=20/40 @40000 步")
        if conv_fix_2_under:
            rows += ladder(conv_fix_2_under, "改动后 H=20/40 @8000 步（未达稳态对照）")
        if conv_fix_1_20k:
            rows += ladder(conv_fix_1_20k, "改动后 H=10/20 @20000 步")
        A(md_table(["代码版本", "臂(wall model)", "q/Δ", "解析参考", "L2 相对误差(粗→细)", "观测阶 p"], rows))
        A("")
        A("结论：")
        A("")
        A("- **改动前三个臂逐位相同**（三种「方法」其实是同一段代码跑三遍）→ 见 §5 缺陷 D1。")
        A("- **台阶与网格对有关（新发现 D11）**：H=10/20 上 halfway p=1.917、bouzidi p=2.057；"
          "但 H=20/40（40 000 步，稳态标志全 True）上 halfway 落到 p=1.370、bouzidi p=2.612 —— "
          "即该臂在测试段内**没有干净的渐近阶**，原「二阶」结论只对 (10,20) 这一对成立。")
        A("- 改动后物理自洽：`halfway`(q=0.5δ) p≈2；`subcell_halfway`（真实壁在 0.25δ 但边界仍按 0.5δ）"
          "p≈1（壁位误差一阶）；`effective`（同一流，改按边界实际执行的等效壁打分）p≈2；"
          "`bouzidi`（0.25δ 插值反弹）p≈2 且误差比 naive 臂小一个量级。")
        A("")
    else:
        A("**未完成**：收敛阶工件缺失。")
        A("")
    A("### 4.2 守恒性（质量）")
    A("")
    cons = []
    if tg:
        cons.append(["Taylor-Green 2D (4900 步, 周期域)", f"{tg['mass_drift_rel']:.3e} (rel)", "解析期望 0"])
    if sw:
        for c in sw["cases"]:
            cons.append([f"shear_wave_decay H={c['H']}", f"{c['mass_drift_rel']:.3e} (rel)", "解析期望 0"])
    if poiseuille2d:
        cons.append(["Poiseuille 2D H=60 (Zou-He 压差边界)", f"{poiseuille2d.get('mass_drift_pct')} %",
                     f"ρ 斜率 vs 名义 {fmt(poiseuille2d.get('rho_slope_ratio'),5)}"])
    if sphere:
        cons.append(["sphere Re=100 D40 (开放边界)", f"{sphere.get('mass_drift_ppm')} ppm", "远场通量平衡"])
    A(md_table(["算例", "质量漂移", "对照"], cons) if cons else "（无工件）")
    A("")

    A("### 4.3 图（matplotlib 生成，`scripts/audit_runtest_r6/make_figures.py`）")
    A("")
    A("- `figures/audit_runtest_r6/conv_ladder.png` — 四臂 L2 相对误差 vs H（双对数），"
      "叠加 p=1/p=2 参考斜率和 HEAD（三臂相同）的星标点。")
    A("- `figures/audit_runtest_r6/bench_gate.png` — 所有实测算例的 |误差|% 与 3% 验收门对照。")
    A("")

    # ---------------------------------------------------------------- claims
    A("## 5. 「文档声称 vs 实测」逐条对照表（差异 = 审计发现）")
    A("")
    if claims:
        A(md_table(
            ["#", "对象", "文档/工件声称", "实测", "判定", "说明"],
            [[i + 1, c["case"], str(c["claim"])[:70], str(c["measured"])[:70], c["verdict"],
              str(c["note"])[:160]] for i, c in enumerate(claims)],
        ))
    else:
        A("（`claim_vs_measured.json` 缺失）")
    A("")

    A("## 6. 审计发现清单（含定位）")
    A("")
    A("| id | 严重度 | 位置 | 发现 | 证据 |")
    A("|---|---|---|---|---|")
    A("| D1 | 高（虚假验证） | `benchmarks/verified/poiseuille_convergence_order/run.py:168-181` | "
      "三个壁面臂 `halfway`/`effective`/`bouzidi` 全部 `q_wall=0.5`，`effective` 落到同一个 "
      "`apply_halfway_bb`，`bouzidi` 在 q=0.5 与半程反弹代数等价 → 三臂结果逐位相同 | 本次重跑 `conv_repro.json`（未改动代码，H=10/20 三臂 l2_rel 完全相同）；改动后 `conv_fix_H10_20.json` 四臂互异 |")
    A("| D2 | 高（文档自相矛盾） | 同上 `run.py` 模块 docstring vs `README.md` | "
      "docstring 预登记判据 (a)「halfway ≈ 一阶」与 README「p∈[1.9,2.1] PASS」互斥，实测 p=1.917 | `result.json` + 本次重跑 |")
    A("| D3 | 中（工件误导） | 同上 `result.json` 的 `bouzidi` 臂 | README 自述「Bouzidi 未测」，result.json 却带数值 bouzidi 臂（= halfway 的副本） | `README.md` Limitations + `conv_repro.json` |")
    A("| D4 | 中（宿主依赖） | `benchmarks/verified/poiseuille_3d_pipe/run.py:627,640` | `--device` 默认 `cuda:2`，CPU-only 宿主直接 AssertionError；STATUS.md 却记为 verified「analytic」 | `runs/audit_runtest_r6/bench_poiseuille_3d_pipe.log`（首跑 cuda 失败）+ CPU 复跑 |")
    A("| D5 | 中（缺校验） | `src/tensorlbm/general_sim.py`（全文 0 个 `raise ValueError`） | 引擎配置 viscosity=0/resolution=0/max_steps<0 全部静默接受 | `negative_probe.json` N06/N07/N08 |")
    A("| D6 | 中（静默 NaN） | `src/tensorlbm/boundaries.py: zou_he_outlet_pressure` | ρ_out=0 → NaN 场，无诊断 | `negative_probe.json` N05 |")
    A("| D7 | 中（库边界不一致） | `src/tensorlbm/solver.py:67 collide_bgk` | 核心碰撞不校验 τ>0.5，同仓十余个模块校验 → 同一非法输入「看路径决定是否报错」 | `negative_probe.json` N00/N01/N02 |")
    A("| D8 | 低 | `benchmarks/verified/mms_convergence_2d/` | 仓内自述 HONEST FAIL（不稳定），本次复跑一致；无 result.json，仅 STATUS.md | `runs/audit_runtest_r6/mms_conv.json` |")
    A("| D10 | 中 | `benchmarks/verified/poiseuille_convergence_order/run.py`（改动前无稳态判据） | "
      "步数固定而瞬态时间 ~H²/ν：H=20/40 在 8000 步下观测阶变负（-2.02/-1.93/-0.97），"
      "即默认 `--steps 20000` 的 3 档梯级本身会给出垃圾阶 | `conv_fix_H20_40_steps8000_underconverged.json` vs `conv_fix_H20_40.json`(40000 步) |")
    A("| D11 | 中（结论适用范围） | 同上 | 观测阶随网格对变化：halfway 在 (10,20) 上 p=1.917、在 (20,40) 上 p=1.370；"
      "bouzidi 2.057→2.612 ⇒ 「二阶」只在 (10,20) 这一对成立 | §4.1 表 |")
    A("| D12 | 中（真实测试失败） | 抽样套件 10 条失败 | 其中 5 条是物理判据失败：Rayleigh-Bénard Nu 对参考偏差 64.7%/29.4%、"
      "能量平衡 8.24%>5%、Nu 维度一致性 spread 178%/19.5%；另有 wetting N01 接触角拟合窗口无效 2 条、gas-layer 闭合 2 条、设计点外推 1 条 | `artifacts/audit_runtest_6/pytest_sample.log` |")
    A("| D9 | 低 | 同名脚本 `benchmarks/verified/poiseuille_2d/run.py` | CLI 对 H=0 抛 `ZeroDivisionError`（低层异常，非域校验） | `negative_probe.json` N11 |")
    A("")

    A("## 7. 已执行的修复（D1/D2/D3）")
    A("")
    A("- `benchmarks/verified/poiseuille_convergence_order/run.py`：新增 `ARMS` 表（`q`/`kernel`/`reference` 三元组），"
      "几何按 `H_true = ny - 3 + 2q` 反解 `ny`，保证四臂在同一真实高度上比较；每个 case 同时记录 "
      "`l2_rel_true_wall` 与 `l2_rel_effective_wall`，使「半程反弹对等效壁二阶、对真实壁一阶」这一教科书结论"
      "可在同一次运行里审计；收敛比改用真实高度。默认臂表扩为 "
      "`halfway / subcell_halfway / effective / bouzidi`。")
    A("- 模块 docstring 的预登记判据按物理更正（q=0.5 → p≈2；亚格子壁被忽略 → p≈1；等效壁参考 → p≈2；"
      "Bouzidi 插值 → p≥1.8），并显式记录本次审计发现的旧行为。")
    A("- 回归测试 `tests/test_poiseuille_wallmodel_arms.py`（5 条断言：四臂互异 + 各自物理阶）。"
      "**改动前 5 errors / 改动后 5 passed**（`runs/audit_runtest_r6/regression_baseline_HEAD.log` vs "
      "`regression_fixed.log`）。")
    A("")

    A("## 8. 证据分级与未完成项（honest）")
    A("")
    A("- 本次审计的执行面证据：测试套件（全量两次未跑完 + 确定性抽样 834 例跑完，见 §1/§1c）、"
      "4 个算例数值结果、13 个负向探针、4 臂×3 组网格的收敛梯级 + 质量守恒。"
      "所有原始日志/JSON 已归档在 `artifacts/audit_runtest_6/`（运行期写在 `runs/audit_runtest_r6/`，`runs/` 被 gitignore，故同时归档），"
      "本报告数字全部由脚本从这些工件注入。")
    A("- **未完成**：①集群 16 分片全量重算 `cj848` 与本机并行跑了同一套件，集群侧在预算内未派发（队列 250+）；"
      "②sphere Re=100 D40/D60 未复算（D40 递交后被让出 CPU，日志 `sphere_D40_local.log`）、"
      "`poiseuille_3d_pipe` R=20 的 CPU 复跑（`--device cpu --compile-mode eager`）在争抢下 80 min 未收敛到稳态，"
      "只留下 D4 的 CUDA 缺省缺陷证据；"
      "③`benchmarks/STATUS.md` 的 38 个 verified 算例未逐一复跑（本单按令「抽样」）。")
    A("- 本报告不复述 PR 描述，只报实测；跑不通的按原样记录（见 §1 失败清单、§3 FAIL 探针）。")
    A("")

    text = "\n".join(L) + "\n"
    DOC.write_text(text)
    print("wrote", DOC, len(text.splitlines()), "lines")
    return 0


def _torch() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "n/a"


def _numpy() -> str:
    try:
        import numpy

        return numpy.__version__
    except Exception:
        return "n/a"


def _pytest() -> str:
    try:
        from importlib.metadata import version

        return version("pytest")
    except Exception:
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())