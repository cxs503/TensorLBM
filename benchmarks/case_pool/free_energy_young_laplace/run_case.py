#!/usr/bin/env python3
"""Case-pool entry: free-energy LBM droplet, analytic Young-Laplace calibration.

Anchor: ``XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D``.

The physics and the grading protocol live in
``tensorlbm.phasefield.laplace_droplet_calibration``; this module is the pool
adapter that :mod:`benchmarks.case_pool.run_pool` discovers (``run_case.py`` with
a ``main(argv)``), so the case runs through the same driver as every other pool
case::

    PYTHONPATH=src python3 benchmarks/case_pool/run_pool.py --list
    PYTHONPATH=src python3 benchmarks/case_pool/run_pool.py --case free_energy_young_laplace
    PYTHONPATH=src python3 benchmarks/case_pool/run_pool.py --case free_energy_young_laplace --smoke

Artifacts are written next to this file (``artifacts/summary.json``,
``artifacts/ref_vs_got.json``, ``artifacts/fig_*.png``).  Exit code 0 = PASS,
1 = FAIL, 2 = usage error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

sys.path.insert(0, str(REPO / "src"))

from tensorlbm.phasefield.laplace_droplet_calibration import (  # noqa: E402
    LaplaceCriteria,
    default_cases,
    smoke_cases,
)

if __package__:
    from . import run_xval
else:
    import run_xval

ANCHOR = "XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D"
PREDECLARED = HERE / "predeclared.json"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    smoke = "--smoke" in argv
    out = HERE / "artifacts"
    for k, arg in enumerate(argv):
        if arg == "--out":
            if k + 1 >= len(argv):
                print("--out needs a directory", file=sys.stderr)
                return 2
            out = Path(argv[k + 1])
    out.mkdir(parents=True, exist_ok=True)

    declared = json.loads(PREDECLARED.read_text())
    criteria = LaplaceCriteria(**{
        k: v for k, v in declared["criteria"].items()
        if k in LaplaceCriteria.__dataclass_fields__
    })
    cases = smoke_cases() if smoke else default_cases()
    rows = run_xval.run_suite(cases, criteria)
    report = run_xval.grade_report(rows, criteria)
    report.update({
        "anchor": ANCHOR,
        "repo": "TensorLBM",
        "entry": "benchmarks/case_pool/free_energy_young_laplace/run_case.py",
        "smoke": smoke,
        "criteria_source": str(PREDECLARED.relative_to(REPO)),
        "protocol": {
            "measurement": "steady-window time average of the bulk pressure jump",
            "cases": [c.as_dict() for c in cases],
        },
        "rows": rows,
    })
    report["figures"] = run_xval._figures(rows, out) if not smoke else []
    (out / "summary.json").write_text(json.dumps(report, indent=1))
    bench = run_xval._benchmark_rows(report)
    (out / "ref_vs_got.json").write_text(json.dumps({
        "anchor": ANCHOR,
        "smoke": smoke,
        "verdict": report["verdict"],
        "n_rows": len(bench),
        "n_pass": sum(1 for r in bench if r["pass"]),
        "worst_abs_rel_error_sigma_capillary": report["worst_abs_rel_error_sigma_capillary"],
        "worst_abs_rel_error_vs_diffuse_exact": report["worst_abs_rel_error_vs_diffuse_exact"],
        "rows": bench,
    }, indent=1))

    print(f"[{ANCHOR}] smoke={smoke} verdict={report['verdict']} "
          f"worst|err sigma|={100*report['worst_abs_rel_error_sigma_capillary']:.2f}% "
          f"rows={len(bench)} pass={sum(1 for r in bench if r['pass'])}")
    for r in rows:
        e = r["measures"]["errors"]
        print(f"  {r['case']['case_id']}: err_sigma_cap={100*e['sigma_capillary_fit']:+.2f}% "
              f"err_sigma_int_ref={100*e['sigma_integral_vs_discrete_ref']:+.2f}% "
              f"passed={r['passed']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())