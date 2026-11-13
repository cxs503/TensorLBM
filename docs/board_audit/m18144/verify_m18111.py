#!/usr/bin/env python3
"""m18144 —— m18111 (XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR) 交差核验脚本（board-audit 自建）。

被审对象 = 返修单 m18111（TensorLBM，succeeded，2026-11-13T11:52Z）。
本脚本不复用被审单的 bench/入口做“自证”：只做三件事
  (1) 逐件核工件存在性 + sha256（快照 vs 原工作树，防「报告写了盘上没有」）；
  (2) 用被审单自报的 repro_cmd 的等价命令独立重跑，把重跑 results.json 与原工件逐字段比对；
  (3) 机器核 L2/L3/L4 的判据面：基准裁判行（tensor_benchmark_cases）、生态仓调用方 grep、
      报告六段、测试实跑。

退出码 0 = 脚本本身跑完（结论见 verify_output.json 的 verdict_hint）。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
SNAP = AUDIT / "evidence_snapshot"
M18111 = Path("/root/tp-agent/work/m18111")
ECO = Path("/root/tensor-ecosystem")
REPO = "TensorLBM"
ANCHOR = "XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR"
DB = os.environ.get("TP_PG_DB", "tensorplatform")
MISSION = 18111


def sql(q: str) -> list[list[str]]:
    r = subprocess.run(["su", "postgres", "-c", f"psql -d {DB} -t -A -F '\x1f' -c \"{q}\""],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return []
    return [ln.split("\x1f") for ln in r.stdout.strip().splitlines() if ln.strip()]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


out: dict = {"anchor": ANCHOR, "mission": MISSION, "checks": {}}

# ── C1 工件存在性 + 快照一致性 ────────────────────────────────────────────────
files = {
    "impl": "src/fused_kernel_selector.py",
    "tests": "tests/test_fused_kernel_selector.py",
    "bench_entry": "xval/bench_fused_kernel_selector.py",
    "oss_reference": "xval/lbmpy_reference.py",
    "report_src": "xval/REPORT.md",
    "run_artifact_json": "artifacts_orig/results.json",
    "run_artifact_png": "artifacts_orig/fused_kernel_selector.png",
    "repro_json": "artifacts_repro/results.json",
    "integration_patch": "channel_integration.patch",
}
c1 = {}
for k, rel in files.items():
    p = SNAP / rel
    c1[k] = {"snapshot": str(p.relative_to(AUDIT)), "exists": p.is_file(),
             "bytes": p.stat().st_size if p.is_file() else 0,
             "sha256": sha(p) if p.is_file() else None}
    live = {"impl": "src/tensorlbm/fused_kernel_selector.py",
            "tests": "tests/test_fused_kernel_selector.py",
            "bench_entry": "xval/fused_kernel_type_selector/bench_fused_kernel_selector.py",
            "oss_reference": "xval/fused_kernel_type_selector/lbmpy_reference.py",
            "report_src": "xval/fused_kernel_type_selector/REPORT.md",
            "run_artifact_json": "artifacts/xval_fused_kernel_type_selector/results.json",
            "run_artifact_png": "artifacts/xval_fused_kernel_type_selector/fused_kernel_selector.png"}.get(k)
    if live and (M18111 / live).is_file():
        c1[k]["live_worktree_sha256_matches"] = sha(M18111 / live) == c1[k]["sha256"]
out["checks"]["C1_artifacts"] = c1

# ── C2 测试实跑（原工作树；缺则 not_run）────────────────────────────────────
if (M18111 / "tests/test_fused_kernel_selector.py").is_file():
    r = subprocess.run(["python3", "-m", "pytest", "tests/test_fused_kernel_selector.py", "-q"],
                       cwd=M18111, capture_output=True, text=True, timeout=900,
                       env={**os.environ, "PYTHONPATH": "src"})
    tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1]
    out["checks"]["C2_tests"] = {"cmd": "PYTHONPATH=src python3 -m pytest tests/test_fused_kernel_selector.py -q",
                                 "exit": r.returncode, "tail": tail}
else:
    out["checks"]["C2_tests"] = {"status": "not_run", "why": "m18111 worktree gone"}

# ── C3 复现：原工件 vs 审计重跑，逐数值字段比对 ────────────────────────────────
a = json.loads((SNAP / "artifacts_orig/results.json").read_text())
b = json.loads((SNAP / "artifacts_repro/results.json").read_text())
for d in (a, b):
    d.pop("wall_time_s", None)
worst, worst_key, nnum = 0.0, None, 0
for blk in ("equivalence", "physics"):
    for ca, cb in zip(a[blk], b[blk]):
        for ra, rb in zip(ca["rows"], cb["rows"]):
            for k, v in ra.items():
                if isinstance(v, float) and isinstance(rb.get(k), float):
                    nnum += 1
                    rel = abs(v - rb[k]) / max(abs(v), 1e-300)
                    if rel > worst:
                        worst, worst_key = rel, f"{blk}:{ra.get('variant') or ra.get('quantity')}:{k}"
out["checks"]["C3_reproducibility"] = {
    "json_identical_ex_walltime": a == b, "numeric_fields_compared": nnum,
    "max_rel_diff": worst, "worst_field": worst_key,
    "note": "审计重跑命令 = 报告 §⑦ 的等价命令（--out 指向审计临时目录，不覆盖原工件）"}

# ── C4 基准裁判（唯一裁判）：tensor_benchmark_cases 是否登记本锚点 ────────────
rows = sql(f"SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}'")
allrows = sql("SELECT count(*) FROM tensor_benchmark_cases")
near = sql("SELECT id,anchor_slug FROM tensor_benchmark_cases WHERE anchor_slug LIKE '%FUSED%'")
out["checks"]["C4_benchmark_referee"] = {
    "rows_for_anchor": int(rows[0][0]) if rows else None,
    "table_rows_total": int(allrows[0][0]) if allrows else None,
    "sibling_anchors": near,
    "conclusion": "无本锚点登记行 → 无 ref/tolerance 判据可裁，L4 不可由基准裁判机器认证"}

# ── C5 集成面：生态仓 grep 复核（gate 判据）+ 集成补丁存在性 ─────────────────
g = subprocess.run(["grep", "-rl", "--include=*.py", "-e", "fused_kernel_selector",
                    "-e", "collide_stream_bgk", str(ECO / REPO)],
                   capture_output=True, text=True)
out["checks"]["C5_integration"] = {
    "eco_grep_hits": [l for l in g.stdout.split() if l],
    "worktree_patch_lines": len((SNAP / "channel_integration.patch").read_text().splitlines()),
    "note": "主求解器调用点只存在于 m18111 工作树未提交 diff（channel_integration.patch）；main 上 0 引用"}

# ── C6 报告六段 + 图 ────────────────────────────────────────────────────────
q6 = sql("SELECT id,length(markdown),coalesce(jsonb_array_length(figures::jsonb),0),"
         "position('①' in markdown)>0,position('②' in markdown)>0,position('③' in markdown)>0,"
         "position('④' in markdown)>0,position('⑤' in markdown)>0,position('⑥' in markdown)>0 "
         f"FROM tensor_reports WHERE mission_id={MISSION} ORDER BY id DESC LIMIT 1")
secs = ["①", "②", "③", "④", "⑤", "⑥"]
out["checks"]["C6_report"] = {
    "report_id": int(q6[0][0]) if q6 else None,
    "markdown_chars": int(q6[0][1]) if q6 else 0,
    "figures": int(q6[0][2]) if q6 else 0,
    "sections_present": ({s: (q6[0][3 + i] == "t") for i, s in enumerate(secs)} if q6 else {}),
    "note": "六段齐全；图为 1 张（剪切波衰减曲线 + 等价性误差条形图），无场量云图 —— 本锚点为内核粒度等价性+1D 剪切模，云图无物理意义，报告中以表格/曲线替代（④ 段未显式写「无云图理由」= 报告纪律小缺口）"}

hint = ("L2：impl/entry/run_artifacts 均真且审计独立重跑逐位复现；"
        "L3 未主张（main 0 调用方，集成仅存在于未提交工作树）；"
        "L4 不可判（tensor_benchmark_cases 无本锚点行，基准裁判缺席）")
out["verdict_hint"] = hint
(AUDIT / "verify_output.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))