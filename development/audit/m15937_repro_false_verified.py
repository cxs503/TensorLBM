#!/usr/bin/env python3
"""m15937 · audit finding F1 reproducer (GAP-LBM-INTL-BENCH / m15912).

Independent demonstration that the delivered grader
``tensorlbm.audit.intl_bench.grade`` grants ``verified`` to a benchmark case whose
own result.json reports an aggregate FAILURE (``all_gates_pass: false``) as long
as at least one nested per-gate ``"pass": true`` exists.

Real instance found in-tree: ``benchmarks/verified/compressible_shock_family``
(11/12 gates, FFS-02 = 1.458 deg > 1.0 deg, README line 47 marks it "FAIL (honest)").
Module ``compressible_shock`` is nevertheless graded ``verified`` and that row was
loaded into tensorlbm_module_maturity.

Usage:  python3 audit/m15937_repro_false_verified.py <path-to-audit-src-root>
        (e.g. /tmp/a15912/src, i.e. the m15912 branch checkout)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "src")
from tensorlbm.audit import intl_bench as ib  # noqa: E402


def build_fixture(root):
    """Minimal repo: impl file + unit test + a 'verified' case whose aggregate is FALSE."""
    os.makedirs(os.path.join(root, "src"), exist_ok=True)
    with open(os.path.join(root, "src", "fake_kernel.py"), "w") as f:
        f.write("def collide(f):\n    return [x * 0.99 for x in f]\n")
    os.makedirs(os.path.join(root, "tests"), exist_ok=True)
    with open(os.path.join(root, "tests", "test_fake.py"), "w") as f:
        f.write("def test_x():\n    assert True\n")
    case = os.path.join(root, "benchmarks", "verified", "fake_family")
    os.makedirs(case, exist_ok=True)
    json.dump({
        "suite": "fake_family",
        "all_gates_pass": False,           # <- the case itself says NOT verified
        "n_gates": 2, "n_gates_pass": 1,
        "gates": [{"id": "G1", "measured": 0.004, "threshold": 0.01, "pass": True},
                  {"id": "G2", "measured": 1.46, "threshold": 1.0, "pass": False}],
        "grids": [{"n": 64}, {"n": 96}, {"n": 128}],
    }, open(os.path.join(case, "result.json"), "w"))
    return root


def main():
    root = build_fixture(tempfile.mkdtemp(prefix="m15937_fx_"))
    mod = ("fake_kernel", "基础LBM内核", ["src/fake_kernel.py"], ["tests/test_fake.py"],
           ["fake_family"], [], "synthetic fixture")
    row = ib.grade(root, {}, mod)
    print("case_verified      :", row["case_verified"])
    print("verdict_fields     :", row["verdict_fields"])
    print("has_convergence    :", row["has_convergence_field"])
    print("evidence_tier      :", row["evidence_tier"])
    print("GRADED LEVEL       :", row["level"], "(ground truth: the case is 1/2 gates = FAIL)")
    bad = row["level"] == "verified"
    print("FALSE_VERIFIED     :", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())