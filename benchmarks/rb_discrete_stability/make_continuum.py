#!/usr/bin/env python
"""Sanity artifact for the continuum reference: Ra_c(a) at the resolved wavenumber
plus the grid-extrapolated critical Rayleigh number (Chandrasekhar anchor)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.rb_continuum_stability import (  # noqa: E402
    RA_CRIT_CONTINUOUS,
    critical_rayleigh_continuous,
    growth,
    marginal_curve,
)

def main() -> int:
    out = {"node": "IN31-N12", "part": "continuum_reference"}
    out.update(critical_rayleigh_continuous(Pr=0.71, M_list=(50, 100, 200), refine=False))
    out["marginal_curve_M100"] = marginal_curve([2.6, 2.8, 3.0, 3.11632, 3.3, 3.5], M=100)
    out["growth_M200"] = [
        {"Ra": Ra, **{k: v for k, v in growth(3.11632, Ra, M=200).items() if k in ("sigma", "sigma_second_real")}}
        for Ra in (1000.0, 1400.0, 1700.0, 1707.7597, 1900.0, 2500.0, 5000.0)
    ]
    out["Ra_c_reference"] = RA_CRIT_CONTINUOUS
    dest = Path("artifacts/in31_n12/continuum.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "marginal_curve_M100"}, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
