#!/usr/bin/env python3
"""CASE-LBM-PRESSURE-TENSOR-DROPLET -- case-pool entry for the OSS pressure-tensor model.

Anchor: ``XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT``.
Model: free-energy binary LBM whose equilibrium carries the pressure tensor
(OSS ``sailfish`` semantics, vendored at
``xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/ref/sailfish_src``).

This is the *solver-side* entry of the transplanted model (the xval driver in
``xval/...`` is the verification harness): it advances the coupled system,
measures the mechanical pressure tensor from the populations and judges the
Laplace law against the analytic ``sigma / R`` with the owner 15 % band.

    python cases/pressure_tensor_droplet/run.py            # R = 16, 2000 steps (~20 s)
    python cases/pressure_tensor_droplet/run.py --steps 4000 --radius 24

Exit code 0 iff every requested radius is inside the 15 % band.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.phasefield.pressure_tensor_equilibrium import (  # noqa: E402
    run_coupled_droplet,
)

TOL_PCT = 15.0
A, XI = 0.1, 2.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius", type=float, nargs="+", default=[16.0])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", type=str, default=str(ROOT / "artifacts/xval_pressure_tensor_equilibrium/droplet_case.json"))
    args = ap.parse_args(argv)

    kappa = A * XI**2 / 2.0
    rows, ok = [], True
    for R in args.radius:
        r = run_coupled_droplet(A=A, kappa=kappa, radius=R, steps=args.steps,
                                tau_f=1.0, tau_phi=1.0, Gamma=0.2)
        passed = abs(r.laplace_tail_rel_error) * 100.0 < TOL_PCT
        ok = ok and passed
        rows.append({
            "radius": R, "steps": args.steps,
            "sigma_analytic": r.sigma_analytic, "sigma_measured": r.sigma_measured,
            "dp_bulk_mean": r.dp_tail_mean, "dp_laplace_ref": r.dp_laplace_ref,
            "laplace_err_pct": 100 * r.laplace_tail_rel_error,
            "max_speed": r.max_speed, "pass": passed,
        })
        print(f"R={R:5.1f}  dp={r.dp_tail_mean:.6f}  sigma/R={r.dp_laplace_ref:.6f}  "
              f"err={100 * r.laplace_tail_rel_error:+.2f}%  max|u|={r.max_speed:.2e}  "
              f"{'PASS' if passed else 'FAIL'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"criterion_pct": TOL_PCT, "rows": rows, "pass": ok}, indent=2))
    print("wrote", out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())