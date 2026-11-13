#!/usr/bin/env python3
"""m18157 交差核验：独立复核 m17996（XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE）的三件套与数值。

立场：不读结论，只读原始数据 + 自己重算。KPI 全部由本文件自带实现重算（不调用被审脚本的
拟合/误差函数），只 import 被审库做 ①负向控制实验（gate 无参考能否 PASS）②调用方 grep 佐证。

产出 verify_output.json：evidence 指针、重算表、独立复跑一致性、负向控制、DB 实况。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

W = Path("/root/tp-agent/work/m17996")          # 被审工作树
HERE = Path(__file__).resolve().parent
OUT = HERE / "verify_output.json"
RERUN = Path("/tmp/rep17996/out")               # 本审计单独立复跑输出
ART = "artifacts/xval_negative_no_oss_reference"
TOL = 15.0
CELLS = ["main", "conv_n32", "conv_n128", "sens_u0_0.020", "sens_u0_0.100",
         "sens_tau_0.70", "sens_tau_1.00", "matched_n128"]
IMPL = ["src/tensorlbm/xval/oss_reference.py", "src/tensorlbm/xval/runners.py",
        "scripts/xval_negative_no_oss_reference.py"]
XVAL_ART = [f"{ART}/summary.json", f"{ART}/oss_reference_probe.json",
            f"{ART}/negative_knowledge_payloads.json", f"{ART}/capability_coverage.json",
            f"{ART}/manifest.json", f"{ART}/REPORT.md",
            "xval/XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE/predeclared.json",
            "docs/xval_negative_no_oss_reference_report.md"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if Path(p).is_file() else "MISSING"


def run(cmd: str, cwd: Path | None = None, env: dict | None = None) -> dict:
    p = subprocess.run(cmd, shell=True, cwd=str(cwd or HERE), capture_output=True, text=True,
                       env=env, timeout=2400)
    return {"cmd": cmd, "rc": p.returncode, "tail": (p.stdout + p.stderr).strip()[-1200:]}


def psql(sql: str) -> list[list[str]]:
    env = dict(os.environ)
    user, pw, host, db = Path("/tmp/pgcred").read_text().split("\n")[:4]
    env["PGPASSWORD"] = pw
    p = subprocess.run(["psql", "-h", host.split(":")[0], "-U", user, "-d", db, "-tA", "-F", "\x01", "-c", sql],
                       capture_output=True, text=True, env=env)
    return [ln.split("\x01") for ln in p.stdout.strip().split("\n") if ln]


def read_khist(p: Path) -> tuple[list[float], list[float], list[float]]:
    t, k, umax = [], [], []
    for i, line in enumerate(p.read_text().splitlines()):
        if i == 0:
            continue
        a = line.split(",")
        if len(a) < 3 or not a[0].strip():
            continue
        t.append(float(a[0])); k.append(float(a[1])); umax.append(float(a[2]))
    return t, k, umax


def fit_gamma(t: list[float], k: list[float], lo: float = 0.2, hi: float = 0.9) -> float:
    """本审计单自带实现：ln K 对 t 线性最小二乘（窗口 K/K0∈[lo,hi]），γ_K = -slope。"""
    k0 = k[0]
    pts = [(tt, math.log(kk)) for tt, kk in zip(t, k) if lo <= kk / k0 <= hi]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return -sxy / sxx


def read_field(p: Path) -> dict[tuple[int, int], tuple[float, float]]:
    f = {}
    for i, line in enumerate(p.read_text().splitlines()):
        if i == 0:
            continue
        a = line.split(",")
        if len(a) < 4 or not a[0].strip():
            continue
        f[(int(a[0]), int(a[1]))] = (float(a[2]), float(a[3]))
    return f


def l2_rel_pct(a: dict, b: dict) -> float:
    """本审计单自带实现：sqrt(Σ|ua-ub|²/Σ|ub|²)×100。"""
    num = sum((a[k][0] - b[k][0]) ** 2 + (a[k][1] - b[k][1]) ** 2 for k in b)
    den = sum(b[k][0] ** 2 + b[k][1] ** 2 for k in b)
    return math.sqrt(num / den) * 100.0


def l2_rel_pct_transposed(a: dict, b: dict) -> float:
    return l2_rel_pct(a, {(j, i): v for (i, j), v in b.items()})


checks: dict = {}

# ── A1 三件套存在性 + sha256 + git 入库 ────────────────────────────────────
def a1() -> dict:
    d = {"impl": [], "artifacts": [], "cells": [], "manifest_sha_check": {}}
    for rel in IMPL:
        p = W / rel
        d["impl"].append({"path": str(p), "rel": rel, "exists": p.is_file(), "sha256": sha(p),
                          "lines": len(p.read_text().splitlines()) if p.is_file() else 0})
    for rel in XVAL_ART:
        p = W / rel
        d["artifacts"].append({"path": str(p), "rel": rel, "exists": p.is_file(), "sha256": sha(p)})
    for c in CELLS:
        arms = []
        for arm in ("lbmpy", "tensorlbm"):
            dd = W / ART / "cells" / c / arm
            arms.append({"arm": arm,
                         "k_history": sorted(p.name for p in dd.glob("k_history_*.csv")),
                         "n_field_csv": len(list(dd.glob("field_*.csv"))),
                         "run_info": sorted(p.name for p in dd.glob("run_info_*.json"))})
        d["cells"].append({"cell": c,
                           "cell_result": (W / ART / "cells" / c / "cell_result.json").is_file(),
                           "arms": arms})
    g = run("git ls-files | grep -c '^artifacts/xval_negative_no_oss_reference/'", cwd=W)
    d["git_committed_artifact_files"] = int(g["tail"].split("\n")[-1] or 0)
    man = json.loads((W / ART / "manifest.json").read_text())
    for name, want in man.get("files", {}).items():
        got = sha(W / ART / name)
        d["manifest_sha_check"][name] = {"match": got == want, "recomputed": got[:16]}
    d["manifest_probe_sha256_match"] = sha(W / ART / "oss_reference_probe.json") == man.get("probe_sha256")
    d["branch_merged_into_ecosystem_main"] = run(
        "git -C /root/tensor-ecosystem/TensorLBM merge-base --is-ancestor af7b184c HEAD && echo MERGED || echo NOT_MERGED")["tail"]
    d["mission_commit"] = run("git log --format=%H%x20%s -1", cwd=W)["tail"]
    return d


# ── A2 独立重算 γ_K / u_max（原始 CSV → 自算，不信 summary.json）──────────
def a2() -> list[dict]:
    summary = json.loads((W / ART / "summary.json").read_text())
    rep = {c["cell"]: c for c in summary["cells"]}
    rows = []
    for c in CELLS:
        spec = json.loads((W / ART / "cells" / c / "cell_result.json").read_text())
        nd = spec["spec"]["nondim"]
        gamma_theory = 4.0 * nd["nu_lattice"] * nd["k"] ** 2
        g = {}
        for arm, pat in (("inhouse", "tensorlbm/k_history_tensorlbm.csv"),
                         ("oss", "lbmpy/k_history_lbmpy.csv")):
            t, k, umax = read_khist(W / ART / "cells" / c / pat)
            g[arm] = {"gamma": fit_gamma(t, k), "umax_end": umax[-1], "K_end_over_K0": k[-1] / k[0]}
        r = rep[c]
        rows.append({
            "cell": c, "n": spec["spec"]["geometry"]["n"], "gamma_theory": gamma_theory,
            "recomputed_inhouse_gamma": g["inhouse"]["gamma"],
            "recomputed_oss_gamma": g["oss"]["gamma"],
            "reported_inhouse_vs_theory_pct": r["theory"]["gamma_K_inhouse_vs_theory_pct"],
            "recomputed_inhouse_vs_theory_pct": (g["inhouse"]["gamma"] - gamma_theory) / gamma_theory * 100,
            "reported_oss_vs_theory_pct": r["theory"]["gamma_K_oss_vs_theory_pct"],
            "recomputed_oss_vs_theory_pct": (g["oss"]["gamma"] - gamma_theory) / gamma_theory * 100,
            "reported_tool_vs_tool_gamma_pct": r["kpis_tool_vs_tool"]["gamma_K_rel_pct"],
            "recomputed_tool_vs_tool_gamma_pct":
                abs(g["inhouse"]["gamma"] - g["oss"]["gamma"]) / g["oss"]["gamma"] * 100,
            "reported_tool_vs_tool_umax_pct": r["kpis_tool_vs_tool"]["umax_rel_pct"],
            "recomputed_tool_vs_tool_umax_pct":
                abs(g["inhouse"]["umax_end"] - g["oss"]["umax_end"]) / g["oss"]["umax_end"] * 100,
            "reported_field_l2_pct": r["kpis_tool_vs_tool"]["field_l2_rel_pct"],
            "K_end_over_K0": g["inhouse"]["K_end_over_K0"],
        })
    return rows


# ── A3 独立重算场 L2 + 朝向自检 ───────────────────────────────────────────
def a3() -> list[dict]:
    summary = json.loads((W / ART / "summary.json").read_text())
    rep = {c["cell"]: c for c in summary["cells"]}
    out = []
    for c in CELLS:
        d0 = W / ART / "cells" / c
        steps = sorted({int(p.stem.split("_")[-1]) for p in (d0 / "lbmpy").glob("field_lbmpy_*.csv")})
        if not steps:
            out.append({"cell": c, "skipped": "no lbmpy field csv in audited worktree"})
            continue
        last = steps[-1]
        a0 = read_field(d0 / "tensorlbm" / "field_tensorlbm_0.csv")
        b0 = read_field(d0 / "lbmpy" / "field_lbmpy_0.csv")
        a1f = read_field(d0 / "tensorlbm" / f"field_tensorlbm_{last}.csv")
        b1f = read_field(d0 / "lbmpy" / f"field_lbmpy_{last}.csv")
        out.append({"cell": c, "field_step": last,
                    "recomputed_init_direct_pct": l2_rel_pct(a0, b0),
                    "recomputed_init_transposed_pct": l2_rel_pct_transposed(a0, b0),
                    "reported_init_direct_pct": rep[c]["orientation_check"]["init_direct_pct"],
                    "reported_init_transposed_pct": rep[c]["orientation_check"]["init_transposed_pct"],
                    "recomputed_final_field_l2_pct": l2_rel_pct(a1f, b1f),
                    "recomputed_final_field_l2_transposed_pct": l2_rel_pct_transposed(a1f, b1f),
                    "reported_final_field_l2_pct": rep[c]["kpis_tool_vs_tool"]["field_l2_rel_pct"]})
    return out


# ── A4 repro_cmd 可执行 + 独立复跑与原件逐点一致性 ────────────────────────
def a4() -> dict:
    d = {"repro_cmd":
         "cd /tmp/rep17996 && PYTHONPATH=/tmp/rep17996/src python3 scripts/xval_negative_no_oss_reference.py "
         "--out /tmp/rep17996/out --mission-id 18157",
         "note": "被审工作树只读复跑：把 scripts/xval_*.py + src/ + tests/ + xval/<anchor>/ 拷到 /tmp/rep17996 后实跑"}
    d["rerun_done"] = (RERUN / "summary.json").is_file()
    if not d["rerun_done"]:
        return d
    s1 = json.loads((W / ART / "summary.json").read_text())
    s2 = json.loads((RERUN / "summary.json").read_text())
    d["gate_counts_orig"] = s1["gate_counts"]
    d["gate_counts_rerun"] = s2["gate_counts"]
    d["worst_orig"] = s1["worst_tool_vs_tool_pct"]
    d["worst_rerun"] = s2["worst_tool_vs_tool_pct"]
    d["worst_identical"] = s1["worst_tool_vs_tool_pct"] == s2["worst_tool_vs_tool_pct"]
    cmp = []
    for c in CELLS:
        for arm, fn in (("inhouse", "tensorlbm/k_history_tensorlbm.csv"),
                        ("oss", "lbmpy/k_history_lbmpy.csv")):
            p1 = W / ART / "cells" / c / fn
            p2 = RERUN / "cells" / c / fn
            if not (p1.is_file() and p2.is_file()):
                cmp.append({"cell": c, "arm": arm, "skipped": True}); continue
            t1, k1, u1 = read_khist(p1); t2, k2, u2 = read_khist(p2)
            n = min(len(k1), len(k2))
            dmax = max(max(abs(k1[i] - k2[i]) for i in range(n)),
                       max(abs(u1[i] - u2[i]) for i in range(n)))
            rel = max(abs(k1[i] - k2[i]) / max(abs(k2[i]), 1e-300) for i in range(n))
            cmp.append({"cell": c, "arm": arm, "samples": n, "max_abs_diff_k": dmax,
                        "max_rel_diff_k": rel, "bitwise_same": dmax == 0.0})
    d["khistory_compare"] = cmp
    d["n_bitwise_identical_arms"] = sum(1 for r in cmp if r.get("bitwise_same"))
    d["n_arms"] = len(cmp)
    return d


# ── A5 测试实跑 ───────────────────────────────────────────────────────────
def a5() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{W}/src"
    r = run("python3 -m pytest tests/test_xval_oss_reference.py -q -p no:cacheprovider", cwd=W, env=env)
    m = [ln for ln in r["tail"].splitlines() if "passed" in ln or "failed" in ln or "error" in ln.lower()]
    return {"rc": r["rc"], "summary_line": m[-1] if m else r["tail"][-300:], "cmd": r["cmd"]}


# ── A6 集成调用方（诚实核查，不为凑级而凑）────────────────────────────────
def a6() -> dict:
    import shlex
    pats = ["tensorlbm.xval.oss_reference", "from .oss_reference", "XvalReferenceGate",
            "probe_oss_references", "run_lbmpy"]
    hits: list[str] = []
    for pat in pats:
        hit = run("grep -rn --include='*.py' -F %s . | grep -v -E '^\\./%s|^\\./tests/test_xval_oss_reference.py|"
                  "^\\./src/tensorlbm/xval/oss_reference.py|^\\./scripts/xval_negative_no_oss_reference.py'"
                  % (shlex.quote(pat), ART), cwd=W)
        hits += [f"[{pat}] " + ln for ln in hit["tail"].splitlines() if ln.strip()]
    camp = run("grep -n 'from .runners import' src/tensorlbm/xval/campaign.py", cwd=W)
    return {"grep_pattern": pat, "hits_outside_mission_files": hits[:12],
            "n_hits": len(hits), "campaign_import_line": camp["tail"][:200],
            "verdict": "NO_INTEGRATION_CALLER" if not hits else "CHECK_HITS"}


# ── A7 负向控制：无参考不得 PASS（对抗性运行被审 gate）──────────────────────
def a7() -> dict:
    code = r'''
import json, sys
sys.path.insert(0, "%s/src")
from tensorlbm.xval.oss_reference import XvalReferenceGate, Claim
errs = {"gamma_K_rel_pct": 0.0, "umax_rel_pct": 0.0, "field_l2_rel_pct": 0.0}
def C(rid, kind="oss", e=None, cid="audit::negctl"):
    return Claim(claim_id=cid, capability="audit_negctl", reference_kind=kind,
                 reference_id=rid, errors_pct=(e or errs))
out = {}
empty = {"record_sha256": "deadbeef", "references": [{"oss_id": "lbmpy", "available": False, "error": "not installable"}]}
g = XvalReferenceGate(empty, tolerance_pct=15.0)
v1 = g.decide(C("lbmpy"))
out["no_ref_zero_error"] = {"status": v1["status"], "negative_knowledge_required": v1["negative_knowledge_required"]}
out["fake_analytic"] = {"status": g.decide(C("totally_made_up_analytic", "analytic"))["status"]}
good = {"record_sha256": "cafe", "references": [{"oss_id": "lbmpy", "available": True, "version": "2.0", "sha256": "x"}]}
g2 = XvalReferenceGate(good, tolerance_pct=15.0)
out["resolved_ref_zero_error"] = g2.decide(C("lbmpy"))["status"]
out["over_tolerance"] = g2.decide(C("lbmpy", e={"field_l2_rel_pct": 15.1}))["status"]
out["under_tolerance"] = g2.decide(C("lbmpy", e={"field_l2_rel_pct": 14.9}))["status"]
try:
    g2.decide(C(""))
    out["empty_ref_id"] = "NO_RAISE"
except Exception as e:
    out["empty_ref_id"] = type(e).__name__
print(json.dumps(out))
''' % W
    p = run(f"python3 - <<'PYEOF'\n{code}\nPYEOF")
    try:
        d = json.loads(p["tail"].splitlines()[-1])
    except Exception:
        d = {"parse_error": p["tail"][-600:], "rc": p["rc"]}
    # 静态检查：源码里有无"无参考也可 PASS"的分支
    src = (W / "src/tensorlbm/xval/oss_reference.py").read_text()
    d["static_pass_requires_resolved_ref"] = ('if not res["resolved"]' in src and
                                              src.index('if not res["resolved"]') < src.index('status="PASS"'))
    d["static_mentions_withheld"] = src.count("WITHHELD")
    return d


# ── A8 平台基准裁判（tensor_benchmark_cases）+ 负知识台账 ─────────────────
def a8() -> dict:
    rows = psql("SELECT id,case_name,metric,tolerance,status FROM tensor_benchmark_cases "
                "WHERE anchor_slug='XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE'")
    anyneg = psql("SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug ILIKE '%NEGATIVE%'")
    neg = psql("SELECT id,entry_id,left(title,60),failure_kind,registered_by FROM negative_knowledge_ledger "
               "WHERE mission_id=17996 ORDER BY id")
    rep = psql("SELECT id,length(markdown),jsonb_array_length(coalesce(figures,'[]'::jsonb)) "
               "FROM tensor_reports WHERE mission_id=17996")
    return {"registered_benchmark_cases_for_anchor": len(rows), "rows": rows,
            "negative_like_anchor_cases_in_table": anyneg[0][0] if anyneg else "?",
            "negative_knowledge_entries": neg, "n_negative_entries": len(neg),
            "tensor_reports": rep}


# ── A9 报告六段（平台报告 16912 + 仓内 md）───────────────────────────────
def a9() -> dict:
    pp = HERE / "evidence" / "payload_m17996.json"
    d = {"platform_report_id": 16912}
    if pp.is_file():
        j = json.loads(pp.read_text())
        md = j.get("markdown", "")
        d["md_len"] = len(md)
        d["sections_present"] = {k: (k in md) for k in ["①", "②", "③", "④", "⑤", "⑥"]}
        d["n_figures"] = len(j.get("figures") or [])
    rp = (W / ART / "REPORT.md").read_text()
    d["repo_report_len"] = len(rp)
    d["repo_report_sections"] = {k: (k in rp) for k in ["## ①", "## ②", "## ③", "## ④", "## ⑤", "## ⑥"]}
    d["tables"] = rp.count("|---|")
    d["claims_lt_15pct_stated"] = "<15%" in rp
    return d


# ── A10 负知识 payload 与平台台账一致性（注册是否真发生）─────────────────
def a10() -> dict:
    pl = json.loads((W / ART / "negative_knowledge_payloads.json").read_text())
    items = pl.get("payloads") or pl.get("entries") or pl if isinstance(pl, list) else pl
    if isinstance(items, dict):
        items = items.get("payloads", [])
    titles = [i.get("title") for i in items if isinstance(i, dict)]
    led = psql("SELECT left(title,60) FROM negative_knowledge_ledger WHERE mission_id=17996")
    led_titles = [r[0] for r in led]
    cap = json.loads((W / ART / "capability_coverage.json").read_text())
    without = [r["capability"] for r in cap["rows"] if r["status"] != "OSS_REFERENCE_AVAILABLE"]
    return {"n_payloads": len(items), "payload_titles": titles[:8],
            "n_ledger_rows": len(led_titles), "ledger_titles": led_titles,
            "capabilities_without_oss_reference": without,
            "coverage_counts": {"total": cap["capabilities_total"],
                                "with": cap["capabilities_with_oss_reference"],
                                "without": cap["capabilities_without_oss_reference"]}}


for name, fn in [("A1_three_piece", a1), ("A2_kpi_recompute", a2), ("A3_field_and_orientation", a3),
                 ("A4_repro_and_determinism", a4), ("A5_tests", a5), ("A6_integration_callers", a6),
                 ("A7_negative_controls", a7), ("A8_platform_judge", a8), ("A9_report_sections", a9),
                 ("A10_negative_payload_vs_ledger", a10)]:
    try:
        checks[name] = fn()
        print(f"[ok] {name}")
    except Exception as e:  # 核验失败必须显式留档，不许静默
        checks[name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[ERR] {name}: {type(e).__name__}: {e}")

OUT.write_text(json.dumps({"mission_audited": 17996, "audit_mission": 18157, "checks": checks},
                          ensure_ascii=False, indent=1))
print("wrote", OUT)