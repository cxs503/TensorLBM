#!/usr/bin/env python
"""Entry point: ref-vs-got benchmark for two-component SC layered Poiseuille.

Runs the cheap physical matrix of
:mod:`tensorlbm.verification.layered_poiseuille.benchmark` and writes

* ``benchmark.json`` -- full payload (payload metadata, environment, per-KPI
  ``ref`` / ``got`` / ``err`` / ``gate`` rows, per-case profiles);
* ``benchmark.csv``  -- the same KPI table, flat;
* ``benchmark.md``   -- paper-style markdown table with the exact repro command;
* ``benchmark.sha256`` -- sha256 of the JSON payload (bitwise artifact identity).

Exit code is 0 only when every pre-registered gate passes, so the script can be
used as a CI gate.

Usage
-----
::

    cd TensorLBM
    PYTHONPATH=src python xval/two_component_sc_layered_poiseuille/run_benchmark.py \
        --matrix core --out artifacts/xval_two_component_sc_layered_poiseuille_edm
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from tensorlbm.verification.layered_poiseuille.benchmark import (
    MATRICES,
    run_benchmark,
    write_artifacts,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matrix", default="core", choices=sorted(MATRICES),
                    help="core = 5 pre-registered cases (default); full = 6 (adds the H=48 "
                         "resolution check); light = 4 cheap cases for CI")
    ap.add_argument("--out", default="artifacts/xval_two_component_sc_layered_poiseuille_edm",
                    help="artifact directory (created if missing)")
    ap.add_argument("--cases", default=None,
                    help="explicit comma-separated case names (overrides --matrix)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    cases = [c for c in (args.cases or "").split(",") if c]
    payload = run_benchmark(args.matrix, cases=cases or None, verbose=not args.quiet)
    paths = write_artifacts(payload, out_dir)

    print(f"\nsha256(benchmark.json) = {payload['sha256']}")
    for kind, path in paths.items():
        print(f"wrote {kind:7s} {path}")
    s = payload["summary"]
    print(json.dumps({k: s[k] for k in
                      ("n_cases", "n_kpi", "n_kpi_pass", "n_kpi_fail", "all_gates_pass")},
                     indent=1))
    if not s["all_gates_pass"]:
        print("FAILED GATES:", [r for r in payload["kpi_table"] if not r["pass"]], file=sys.stderr)
    return 0 if s["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())