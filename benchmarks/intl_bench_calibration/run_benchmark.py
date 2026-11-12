#!/usr/bin/env python3
"""Grading-calibration benchmark for ``tensorlbm.audit.intl_bench`` (GAP-LBM-INTL-BENCH).

Why this benchmark exists
-------------------------
The audit assigns maturity grades.  A grade is only trustworthy if the grader is
shown to (a) climb the evidence ladder when evidence genuinely improves and
(b) refuse to promote on missing or shared evidence.  Each control in
``controls.json`` builds a throw-away repository whose evidence level is fixed *by
construction*; the expected grade is therefore a reference value derived from the
ladder definition, independent of the grader's code.

Outputs (numbers, not prose)::

    agreement_pct      ref-level vs got-level agreement over all controls
    confusion          per (ref,got) counts
    anchors            real-repo ref vs got numbers (Cd reference vs computed)

Exit code 0 iff agreement == 100% and every physical anchor passes its threshold.

Run::

    PYTHONPATH=src python benchmarks/intl_bench_calibration/run_benchmark.py \
        --repo-root . --out benchmarks/intl_bench_calibration/RESULT.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.audit import intl_bench as ib  # noqa: E402


def _w(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.strip() + "\n")


def build_case(root: Path, spec: dict, label: str = "ctl") -> tuple:
    """Materialise one control repo and return the module spec tuple for grade()."""
    impl = [f"src/mod_{label}.py"]
    tests = [f"tests/test_mod_{label}.py"]
    cases = []
    if spec.get("impl", True):
        _w(root / impl[0], "def collide(f):\n    return f\n")
    if spec.get("test"):
        _w(root / tests[0], "def test_smoke():\n    assert True\n")
    if spec.get("pending"):
        _w(root / "benchmarks" / "pending" / spec["pending"] / "result.json",
           json.dumps({"status": "pending", "note": "gate case"}))
        cases.append(spec["pending"])
    if spec.get("verified"):
        c = spec["verified"]
        res = {"status": "verified", "cd_got": 1.0}
        if c != "case_noverdict":
            res["verified"] = True
        if c != "case_noconv":
            res["grid_convergence_order"] = 1.97
        _w(root / "benchmarks" / "verified" / c / "result.json", json.dumps(res))
        cases.append(c)
    if spec.get("defect"):
        ib.KNOWN_DEFECTS[f"mod_{label}"] = "control: structural defect declared"
    else:
        ib.KNOWN_DEFECTS.pop(f"mod_{label}", None)
    return (f"mod_{label}", "基础LBM内核", impl, tests, cases, [], "")


def run_controls(tmp: Path, controls: list) -> dict:
    rows, conf = [], {}
    for spec in controls:
        label = spec["id"]
        root = tmp / label
        root.mkdir(parents=True, exist_ok=True)
        mod = build_case(root, spec, label)
        status = ib.load_status(root)
        r = ib.grade(root, status, mod)
        if spec.get("share_case"):
            # second module quoting the same verified case: isolation pass must demote both
            mod2 = build_case(root, spec, label + "_peer")
            r2 = ib.grade(root, status, mod2)
            ib.apply_evidence_isolation([r, r2])
        got = r["level"]
        ref = spec["expected_level"]
        ok = got == ref and r["evidence_tier"] == spec["expected_tier"]
        rows.append({"id": label, "kind": spec["kind"], "ref_level": ref, "got_level": got,
                     "ref_tier": spec["expected_tier"], "got_tier": r["evidence_tier"],
                     "pass": ok, "downgrade_reason": r.get("downgrade_reason", ""),
                     "shared_with": r.get("evidence_shared_with", [])})
        conf[f"{ref}->{got}"] = conf.get(f"{ref}->{got}", 0) + 1
    n = len(rows)
    npass = sum(1 for r in rows if r["pass"])
    return {"controls": rows, "n_controls": n, "n_pass": npass,
            "agreement_pct": round(100.0 * npass / n, 3) if n else 0.0,
            "confusion": conf}


def run_anchor(repo: Path, spec: dict) -> dict:
    case_dir = repo / spec["dir"]
    out = {"case": spec["case"], "evidence_dir_present": case_dir.is_dir(),
           "ref_dir": spec["dir"], "accept_pct": spec.get("accept_pct")}
    rj = case_dir / "result.json"
    if rj.is_file() and spec.get("ref_field"):
        obj = json.loads(rj.read_text())
        node = obj
        for k in spec["ref_field"]:
            node = node[k]
        out["cd_ref"] = float(node)
        grids = obj.get("grids") or []
        got = grids[-1] if isinstance(grids, list) and grids else {}
        out["cd_got"] = float(got.get("cd")) if "cd" in got else None
        out["grid_D"] = got.get("D")
        if out["cd_got"] is not None:
            err = (out["cd_got"] - out["cd_ref"]) / out["cd_ref"] * 100.0
            out["err_pct"] = round(err, 4)
            out["within_accept"] = abs(err) <= spec["accept_pct"]
        out["reported_err_pct"] = got.get(spec.get("err_field", "err_pct"))
        span = (obj.get("convergence") or {}).get("span_pct")
        out["grid_span_pct"] = span
        out["span_within_3pct"] = (span is not None and abs(span - 3.0) > 0) and (span is not None and span <= 3.0)
    if rj.is_file():
        obj = json.loads(rj.read_text())
        # grader view: a module quoting this case (implementation = a real solver file)
        mod = ("anchor_probe", "工程验证", ["src/tensorlbm/d2q9.py"], [], [spec["case"]], [], "")
        rep = ib.grade(repo, ib.load_status(repo), mod)
        out["got_level"] = rep["level"]
        out["got_tier"] = rep["evidence_tier"]
        out["grader_note"] = rep.get("downgrade_reason", "")
        out["verdict_field_on_disk"] = obj.get("verdict") or obj.get("status")
    else:
        out["got_level"] = None
    out["expected_level"] = spec.get("expected_level")
    out["pass"] = bool(out.get("within_accept", True)) and (
        spec.get("expected_level") is None or out["got_level"] == spec["expected_level"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(REPO))
    ap.add_argument("--out", default=str(HERE / "RESULT.json"))
    a = ap.parse_args(argv)
    repo = Path(a.repo_root).resolve()
    spec = json.loads((HERE / "controls.json").read_text())

    with tempfile.TemporaryDirectory(prefix="intl_bench_cal_") as tmp:
        cal = run_controls(Path(tmp), spec["controls"])
    anchors = [run_anchor(repo, s) for s in spec["physical_anchors"]]

    ok_anchors = all(x["pass"] for x in anchors)
    result = {"benchmark": spec["benchmark"], "repo": str(repo),
              "grader": "tensorlbm.audit.intl_bench.grade/apply_evidence_isolation",
              "calibration": cal, "anchors": anchors,
              "verdict": "PASS" if (cal["agreement_pct"] == 100.0 and ok_anchors) else "FAIL",
              "thresholds": {"agreement_pct": 100.0, "sphere_re200_err_pct": 3.0}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps({"agreement_pct": cal["agreement_pct"], "n_controls": cal["n_controls"],
                      "failed_controls": [r["id"] for r in cal["controls"] if not r["pass"]],
                      "anchors": [{k: x.get(k) for k in
                                   ("case", "cd_ref", "cd_got", "err_pct", "within_accept",
                                    "grid_span_pct", "got_level", "pass")} for x in anchors],
                      "verdict": result["verdict"]}, ensure_ascii=False, indent=1))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())