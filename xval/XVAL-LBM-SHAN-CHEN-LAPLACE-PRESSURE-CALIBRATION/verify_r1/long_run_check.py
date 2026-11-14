"""Long-time check: is the droplet vapour still supersaturated at step 4000?

The R1 verification found that at the protocol's stopping point (step 4000)
the droplet far-field vapour pressure sits ~1.8% above the scheme's own flat
coexistence pressure, which is ~19% of the Laplace pressure itself.  In a
closed box that excess must condense onto the droplet, so either

  (i)  the state keeps creeping (R_eq grows, dp shrinks) -> the "converged"
       stopping criterion of the original protocol is too loose, or
  (ii) the state is genuinely stationary -> the vapour reference offset is a
       permanent property of the finite-box protocol.

This script simply integrates much longer (up to 40000 steps) and records the
history, so the answer is measured rather than argued.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XV_DIR = HERE.parent
REPO = XV_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(XV_DIR))

import ref_sc as REF  # noqa: E402
import run_xval as XV  # noqa: E402

OUT = HERE / "artifacts" / "long_run_check.json"


def run(forcing: str, R: float = 16.0, L: int = 64, steps: int = 40000,
        sample: int = 2000, flat_p: float | None = None):
    solver = XV.TorchSolver(forcing=forcing, threads=8)
    rho0, rr = XV.init_droplet(L, R)
    f = solver.init(rho0)
    mass0 = float(rho0.sum())
    hist = []
    for step in range(1, steps + 1):
        f = solver.step(f)
        if step % sample == 0:
            fld = solver.fields(f)
            m = XV.measure(fld, rr, R)
            rec = {k: v for k, v in m.items()}
            rec["step"] = step
            rec["mass_drift_rel"] = abs(m["mass"] - mass0) / mass0
            rec["dp_vs_flat"] = (m["p_in"] - flat_p) if flat_p is not None else None
            rec["sigma_vs_flat"] = (rec["dp_vs_flat"] * m["R_eq"]
                                    if flat_p is not None else None)
            hist.append(rec)
    return hist


def main() -> int:
    flat = json.loads((HERE / "artifacts" / "verify_r1.json").read_text())["legs"]["V1_flat"]
    res = {}
    for forcing in ("velocity_shift", "guo"):
        p_flat = flat[forcing]["p_l"]
        h = run(forcing, flat_p=p_flat)
        res[forcing] = {"p_flat": p_flat, "history": h}
        for rec in h[::5]:
            print(f"{forcing:15s} step={rec['step']:6d} R_eq={rec['R_eq']:.4f} "
                  f"dp={rec['dp']:.6f} sigma={rec['sigma_from_dp']:.5f} "
                  f"rho_out={rec['rho_out']:.5f} sigma_flat={rec['sigma_vs_flat']:.5f}")
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())