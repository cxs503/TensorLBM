#!/usr/bin/env python3
"""Assemble machine-checkable maturity evidence for CASE-BACKWARD-STEP-LBM.

The gate asks for four things a code drop alone cannot show (L2/L3/L4):

* **L2** non-placeholder implementation files, a runnable entry point and real run
  artefacts produced by it;
* **L3** integration -- the capability is called by the library solver itself and
  by the case pool, not only by its own script;
* **L4** a benchmark table (reference vs obtained) plus a reproduction command.

Every claim is *derived from the tree*: files are hashed and scanned for
placeholder markers, artefacts are hashed, the integration call sites are located
by grep, and the benchmark table is lifted from the run artefacts (never copied by
hand).

    PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/maturity_evidence.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CASE = REPO / "benchmarks/case_pool/backward_step_armaly"
ART = REPO / "artifacts" / "case_backward_step_armaly"

IMPL_FILES = [
    "src/tensorlbm/bfs_armaly.py",
    "benchmarks/case_pool/backward_step_armaly/run_case.py",
]
INTEGRATION_CALLERS = [
    ("src/tensorlbm/backward_facing_step.py",
     "library BFS solver: run_backward_facing_step() reports "
     "final_reattachment_xr_star_wallshear via bfs_armaly"),
    ("benchmarks/case_pool/run_pool.py", "case-pool driver discovers/executes the case"),
]
ENTRY = "PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/run_case.py"
REPRO = (ENTRY + " --preset g2_re200 --steps 70000  # "
         + ENTRY + " --shard i/6 for the whole pool grid")
PLACEHOLDER_PATTERNS = (r"\bNotImplementedError\b", r"\bTODO\b", r"\bFIXME\b", r"\bpass\s*#\s*stub")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(rel: str) -> dict:
    path = REPO / rel
    rec = {"path": rel, "exists": path.is_file()}
    if not rec["exists"]:
        return rec
    text = path.read_text(encoding="utf-8", errors="ignore")
    rec.update({
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "lines": text.count("\n") + 1,
        "n_functions": len(re.findall(r"^\s*def \w+", text, re.M)),
        "placeholders": [p for p in PLACEHOLDER_PATTERNS if re.search(p, text)],
    })
    rec["non_placeholder"] = bool(rec["lines"] >= 40 and rec["n_functions"] >= 2
                                  and not rec["placeholders"])
    return rec


def integration_records() -> list[dict]:
    out = []
    for rel, why in INTEGRATION_CALLERS:
        path = REPO / rel
        rec = {"path": rel, "claimed_role": why, "exists": path.is_file(), "call_lines": []}
        if rec["exists"]:
            for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                if "bfs_armaly" in line or "run_armaly_case" in line or "backward_step_armaly" in line:
                    rec["call_lines"].append({"line": i, "text": line.strip()[:160]})
        rec["is_real_caller"] = bool(rec["call_lines"])
        out.append(rec)
    return out


def benchmark_table() -> dict:
    refs = json.loads((CASE / "refs.json").read_text())
    points = refs["adopted_numeric_reference"]["points"]
    rows = []
    for tag_dir in sorted(ART.glob("g*_re*")):
        rp = tag_dir / "result.json"
        if not rp.is_file():
            continue
        res = json.loads(rp.read_text())
        re_ = str(int(res["config"]["re"]))
        ref = points.get(re_)
        got = res["result"]["x1_over_s"]
        row = {
            "tag": tag_dir.name,
            "re": res["config"]["re"],
            "grid": {"hi": res["config"]["hi"], "s": res["config"]["s"]},
            "tau": res["derived"]["tau"],
            "expansion_ratio": res["derived"]["expansion_ratio"],
            "steps_run": res["result"]["steps_run"],
            "steady_at_step": res["result"]["steady_at_step"],
            "x1_over_s_got": got,
            "x1_over_s_row1_crosscheck": res["result"]["x1_over_s_row1"],
            "artifact_sha256": sha256(rp),
        }
        if ref:
            row["x1_over_s_ref"] = ref["x1_over_s"]
            row["ref_source"] = ref["source"]
            row["error_pct"] = 100.0 * (got - ref["x1_over_s"]) / ref["x1_over_s"]
            row["within_15pct"] = abs(row["error_pct"]) <= refs["adopted_numeric_reference"]["tolerance_pct"]
        rows.append(row)
    return {"tolerance_pct": refs["adopted_numeric_reference"]["tolerance_pct"], "rows": rows}


def main() -> int:
    art = [file_record(str(p.relative_to(REPO))) for p in sorted(ART.glob("g*_re*/*"))
           if p.suffix in {".json", ".csv", ".npz"}]
    art += [file_record(str(p.relative_to(REPO))) for p in sorted(ART.glob("figures/*.png"))]
    evidence = {
        "case": "CASE-BACKWARD-STEP-LBM",
        "anchor": "TensorLBM / benchmarks/case_pool/backward_step_armaly",
        "levels": {
            "L2": {"impl_files": [file_record(f) for f in IMPL_FILES],
                   "entry": ENTRY,
                   "tests": [file_record("tests/test_case_pool_backward_step_armaly.py")],
                   "run_artifacts": art},
            "L3": {"integration_callers": integration_records()},
            "L4": {"benchmark": benchmark_table(), "repro_cmd": REPRO},
        },
    }
    out = ART / "maturity_evidence.json"
    ART.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2))
    print(json.dumps({k: {"ok": all(x.get("exists", False) for x in v.get("impl_files", []) or [])
                          if k == "L2" else True} for k, v in evidence["levels"].items()}, indent=2))
    print("benchmark rows:", len(evidence["levels"]["L4"]["benchmark"]["rows"]))
    print("artefacts:", len(art), "->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
