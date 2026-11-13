#!/usr/bin/env python3
"""m18213 负控（NC）：证明 m17990 analyze.py 的 H_gap_survives_refinement 在「单分辨率」下也返回 true。

做法（端到端，不修改被审仓）：
  1) 把 m17990 的 xval/benchmark_method_selection 复制到 /tmp/nc_m17990/one_rung；
  2) 把 D=14 的臂工件移出 runs/ —— 即「只有一个分辨率」的世界；
  3) 跑该目录里的 analyze.py（被审代码原样），读 summary.json 的 verdict；
  4) 期望（被审逻辑）: H_gap_survives_refinement == true，而 gap_ladder 里只有 rungs=[10]。
     这在语义上等价于「一个分辨率都没跑就宣称差距在加密下存活」→ 断言伪证。
  5) 修正语义（本审计主张）: 需要 >=2 个 rung 且最高 rung 仍越界才可为 true。
"""
import json
import shutil
import subprocess
from pathlib import Path

SRC = Path("/root/tp-agent/work/m17990/xval/benchmark_method_selection")
ROOT = Path("/tmp/nc_m17990/one_rung")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(SRC, ROOT)
for d in (ROOT / "runs").iterdir():
    if "_D14" in d.name or "_D20" in d.name:
        shutil.move(str(d), str(ROOT.parent / ("moved_" + d.name)))
# 同时清掉 summary.json 里可能残留的 D=14 行：analyze 会重算，无需处理
r = subprocess.run(["python3", "analyze.py"], cwd=ROOT, capture_output=True, text=True, timeout=1800)
print("analyze.py rc=", r.returncode)
s = json.loads((ROOT / "summary.json").read_text())
v = s["verdict"]
rungs = sorted({row["D"] for row in s["table"].values()})
print("published rungs in table:", rungs)
print("gap_ladder rungs:", {a: [r["D"] for r in e["rungs"]] for a, e in s["gap_ladder"].items()})
print("verdict.H_gap_survives_refinement =", v["H_gap_survives_refinement"])
print("verdict.H_conv_all_arms_converged =", v["H_conv_all_arms_converged"])
print("verdict.H_fail_persists_at_higher_D =", v["H_fail_persists_at_higher_D"])

gap = s["gap_ladder"]
corrected = bool(gap) and all(
    len(e["rungs"]) >= 2 and e["out_of_band_at_any_rung"] for e in gap.values())
print("corrected semantics (>=2 rungs required) =", corrected)

nc = {
    "nc": "single-resolution world (D=14/20 arm artifacts removed, analyze.py unmodified)",
    "published_rungs": rungs,
    "gap_ladder_rungs": {a: [r["D"] for r in e["rungs"]] for a, e in gap.items()},
    "shipped_flag_H_gap_survives_refinement": v["H_gap_survives_refinement"],
    "corrected_flag": corrected,
    "defect": "analyze.py:289-290 uses `bool(gap_ladder) and all(out_of_band_at_any_rung)`; "
              "out_of_band_at_any_rung is a single-rung property, so ONE rung already yields true "
              "=> the flag reports 'gap survives refinement' without any refinement.",
    "contrast": "H_fail_persists_at_higher_D (analyze.py:291-293) DOES require len(resolutions)>=2 "
                "=> the same file shows the correct pattern; the gap flag simply missed it.",
}
Path("/root/tp-agent/work/m18213/audit/m18213/nc_single_rung.json").write_text(
    json.dumps(nc, ensure_ascii=False, indent=1))
print(json.dumps(nc, ensure_ascii=False, indent=1))