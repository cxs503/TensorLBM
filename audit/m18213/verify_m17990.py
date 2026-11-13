#!/usr/bin/env python3
"""m18213 独立核验：m17990 数值自洽性 + 交付件存在性（不 import 被审代码）。"""
import hashlib
import json
import math
from pathlib import Path

W = Path("/root/tp-agent/work/m17990")
X = W / "xval/benchmark_method_selection"
M = Path("/root/tp-agent/work/m18213")
out = {}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ---------- A1 交付件存在性（m17990 worktree） ----------
files = [
    "docs/XVAL_LBM_BENCHMARK_METHOD_SELECTION.md",
    "tests/test_xval_benchmark_method_selection.py",
    "xval/benchmark_method_selection/analyze.py",
    "xval/benchmark_method_selection/make_report.py",
    "xval/benchmark_method_selection/predeclared_v3.json",
    "xval/benchmark_method_selection/publish_report.py",
    "xval/benchmark_method_selection/run_ladder.sh",
    "xval/benchmark_method_selection/run_ladder_cluster.sh",
    "xval/benchmark_method_selection/summary.json",
    "xval/benchmark_method_selection/table.csv",
    "xval/benchmark_method_selection/oss_crosscheck.json",
    "xval/benchmark_method_selection/runs/A3_D14/fail.json",
] + ["xval/benchmark_method_selection/figures/fig%d_%s.png" % (i, n) for i, n in
     [(1, "cl_traces"), (2, "kpi_vs_A1"), (3, "resolution_ladder"), (4, "vorticity_fields"), (5, "gap_ladder")]]
out["artifacts"] = {}
for f in files:
    p = W / f
    out["artifacts"][f] = {"exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0,
                           "sha256": sha(p)[:16] if p.exists() else None}

# ---------- A2 独立重算 KPI 差值（只读 result.json） ----------
rows = {}
for arm in ["A1", "A2", "A4"]:
    rows[arm] = json.loads((X / ("runs/%s_D10/result.json" % arm)).read_text())
recalc = {}
for arm in ["A2", "A4"]:
    recalc[arm] = {}
    for k in ["cd_mean_last50", "cl_amp", "st_fft"]:
        a, b = rows["A1"][k], rows[arm][k]
        recalc[arm][k] = {"A1": a, arm: b, "delta_pct": (b - a) / a * 100.0}
out["recalc_delta_vs_A1"] = recalc
out["A1_got"] = {k: rows["A1"][k] for k in ["cd_mean_last50", "cl_amp", "st_fft", "n_samples", "tau"]}

# ---------- A3 TRT tau_minus 独立重算 ----------
trt = {}
for D, f in [(10, "runs/A3_D10/fail.json"), (14, "runs/A3_D14/fail.json")]:
    d = json.loads((X / f).read_text())
    tau_plus = d["tau"]
    lam = d["lambda_trt"]
    tau_minus_recalc = 0.5 + lam / (tau_plus - 0.5)
    trt["D%d" % D] = {"tau_plus": tau_plus, "lambda": lam, "tau_minus_artifact": d["tau_minus"],
                      "tau_minus_recalc": tau_minus_recalc, "match": abs(tau_minus_recalc - d["tau_minus"]) < 1e-9,
                      "fail_step": d["step"], "planned": d["total_steps_planned"], "status": d["status"],
                      "tau_from_nu": 0.5 + 3 * d["nu"]}
out["trt_tau_minus"] = trt

# ---------- A4 梯级完整性：D=14/20 有哪些真工件 ----------
ladder = {}
for arm in ["A1", "A2", "A3", "A4", "A5"]:
    for D in [14, 20]:
        d = X / ("runs/%s_D%d" % (arm, D))
        ladder["%s_D%d" % (arm, D)] = sorted(p.name for p in d.iterdir()) if d.exists() else None
out["ladder_present"] = ladder

# ---------- A5 OSS 内核对照数字（来自工件，独立复算比值无关项：只记录） ----------
oc = json.loads((X / "oss_crosscheck.json").read_text())
out["kernel"] = {"k1_rel_l2": oc["k1_collision"]["rel_l2_collision"],
                 "k2_rel_l2": oc["k2_streaming"]["rel_l2_streaming_interior"],
                 "src_sha_pinned": len(oc["oss_source_sha256_unmodified"]),
                 "oss_repo": oc["oss_repo"], "oss_commit": oc["oss_commit"]}

# ---------- A6 主线合并状态 ----------
import subprocess
r = subprocess.run(["git", "-C", str(M), "ls-tree", "-r", "--name-only", "HEAD",
                    "xval/benchmark_method_selection"], capture_output=True, text=True)
main_files = set(r.stdout.split())
out["main_has"] = {f: (f in main_files) for f in [
    "xval/benchmark_method_selection/predeclared_v3.json",
    "xval/benchmark_method_selection/run_ladder.sh",
    "xval/benchmark_method_selection/summary.json",
    "tests/test_xval_benchmark_method_selection.py",
    "docs/XVAL_LBM_BENCHMARK_METHOD_SELECTION.md"]}
out["main_summary_schema"] = json.loads((M / "xval/benchmark_method_selection/summary.json").read_text()).get("schema")
out["m17990_summary_schema"] = json.loads((X / "summary.json").read_text()).get("schema")

# ---------- A7 集成调用方：谁 import 这些模块 ----------
r = subprocess.run(["grep", "-rl", "--include=*.py", "-e", "run_arm", "-e", "benchmark_method_selection",
                    str(W / "tests"), str(W / "src"), str(W / "xval")], capture_output=True, text=True)
out["callers_grep"] = [x for x in r.stdout.split() if "benchmark_method_selection" in x][:20]

Path("/root/tp-agent/work/m18213/audit/m18213/verify.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1)[:5000])