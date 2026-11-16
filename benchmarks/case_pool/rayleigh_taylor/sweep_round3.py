"""CASE-RAYLEIGH-TAYLOR round 3 -- resolution study of the saturated front speed.

Question raised by the round-2 record
-------------------------------------
Round 2 (m2026-11-13) reported ``u_b = 0.03664`` at ``A = 1/3, lam = 64`` and
called it a ``-0.6 %`` PASS against ``sqrt(2Ag/(Ck))`` with ``C = 1``.  Round 1
had used ``C = 3`` for the same 2-D case and FAILED by ~2x.  The constant is not
free: see ``ref/ref_lock.py::CONVENTION_CONFLICT``.  What *is* free is the
resolution, and round 2 had a single grid.  This script supplies the missing
axis under the bounded Allen-Cahn scheme:

Two families, all at ``A = 1/3, g = 2e-4``:

* ``w2``  -- fixed *lattice* interface width (``ac_width = 2``) at
  ``lam = 64, 128, 256``.  The relative interface width halves with each
  refinement, which is the fastest way to see whether the terminal velocity is
  set by the interface or by the buoyancy.
* ``rel`` -- fixed *relative* width (``ac_width/lam = 2/64``) at
  ``lam = 128 (w = 4)`` and ``lam = 256 (w = 8)``.  If the terminal velocity is
  an interface-width artefact, this family must be lam-independent; if it is a
  property of the model, it must keep overshooting the theory at every lam.

Plus one second ``(A, g)`` point (``A = 1/2, lam = 128``) so the trend is not
established on a single Atwood number.

Every number is written to ``artifacts/round3/runs/<case>/`` by
``run_case.run_case``; ``postprocess_round3.py`` re-grades them against both
constant conventions without re-simulating.

Run (one case per invocation to keep jobs short):
    python benchmarks/case_pool/rayleigh_taylor/sweep_round3.py --only R3_A033_l128_w2
    python benchmarks/case_pool/rayleigh_taylor/sweep_round3.py --aggregate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
for _p in (str(BENCH_DIR), str(BENCH_DIR / "ref"), str(BENCH_DIR.parents[2] / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_case as rc  # noqa: E402

OUT_ROOT = BENCH_DIR / "artifacts" / "round3" / "runs"
BASE = dict(atwood=1.0 / 3.0, g=2.0e-4, tau=0.515, tau_phi=0.501, tau_end=6.5,
            phase_scheme="allen_cahn", ac_sigma=3.0e-4, ac_mobility=20.0)

SPECS = (
    # fixed lattice width (relative width shrinks with refinement)
    rc.CaseSpec("R3_A033_l064_w2", lam=64, ac_width=2.0, **BASE),
    rc.CaseSpec("R3_A033_l128_w2", lam=128, ac_width=2.0, **BASE),
    rc.CaseSpec("R3_A033_l256_w2", lam=256, ac_width=2.0, **BASE),
    # fixed relative width (w/lam = 1/32)
    rc.CaseSpec("R3_A033_l128_w4", lam=128, ac_width=4.0, **BASE),
    rc.CaseSpec("R3_A033_l256_w8", lam=256, ac_width=8.0, **BASE),
    # second (A, g) point
    rc.CaseSpec("R3_A050_l128_w2", lam=128, ac_width=2.0,
                **{**BASE, "atwood": 0.5}),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args(argv)
    if args.aggregate:
        s = rc.aggregate(OUT_ROOT)
        print(json.dumps({"n_cases": s["n_cases"], "n_all_pass": s["n_all_pass"],
                          "C_2D": s["gates"]["goncharov_C_2D"],
                          "C_2D_alt": s["gates"]["goncharov_C_2D_alt"]}, indent=2))
        return 0
    specs = SPECS if not args.only else tuple(s for s in SPECS if s.name in set(args.only))
    if not specs:
        raise SystemExit(f"no spec matched {args.only}")
    s = rc.run_suite(specs, OUT_ROOT, quick=args.quick)
    print(json.dumps({"n_cases": s["n_cases"], "n_all_pass": s["n_all_pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())