"""Post-hoc diagnostics for XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN.

These runs were *not* part of the predeclared gated arms (``predeclared.json``);
they are declared here as exploratory follow-ups that explain two gated
findings and are reported as non-gating in the mission report:

D1  the regime in which the OSS u^2 scalar-equilibrium term matters
    (``u^2`` scaling law) -- decides default-off vs default-on for the transfer;
D2  whether the TRT-vs-BGK accuracy gap survives on a *seamless* Gaussian
    (no truncation edges), i.e. whether it is a scheme property or a
    high-wavenumber/ghost-mode artefact of the reference initial data.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import xval_trt_lambda as X  # noqa: E402

D = 0.5
LAM = 1.0 / 12.0
NSTEPS = 20


def _case(ux: float, sigma: float = 3.0, nx: int = 50, seamless: bool = False) -> "X.OssCase":
    base = X.parse_oss_1d("gaussian_1d_magic12.cpp", "diag")
    return X.OssCase(
        **{**base.__dict__, "nx": nx, "x_init": 25.0, "sigma": sigma,
           "ux": ux, "nsteps": NSTEPS, "magic": LAM, "omega_hand": 0.5,
           "seamless": seamless} if False else
        {**base.__dict__, "nx": nx, "x_init": 25.0, "sigma": sigma,
         "ux": ux, "nsteps": NSTEPS, "magic": LAM, "omega_hand": 0.5}
    )


class _SmoothCase(X.OssCase):
    """Same case but with the seamless (untruncated) Gaussian of the OSS init()."""

    def ic(self, x):  # type: ignore[override]
        d = x - self.x_init
        return np.exp(-0.5 * d * d / (self.sigma**2))


def main() -> dict:
    out: dict = {"note": "post-hoc exploratory diagnostics, non-gating"}

    # ---- D1: u-regime of the u^2 equilibrium term -------------------------
    rows = {}
    for u in (0.0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5):
        case = _case(u)
        exact = np.vstack([X.exact_periodic(case, t, D) for t in range(NSTEPS + 1)])
        lin = X.run_tb(case, D, LAM, equilibrium="linear")
        sec = X.run_tb(case, D, LAM, equilibrium="oss")
        rows[str(u)] = {
            "rel_l2_linear": X.rel_l2(lin[-1], exact[-1]),
            "rel_l2_second_order": X.rel_l2(sec[-1], exact[-1]),
            "max_abs_field_diff": float(np.abs(lin - sec).max()),
            "rel_l2_diff": X.rel_l2(lin[-1], sec[-1]),
            "ratio_over_u2": X.rel_l2(lin[-1], sec[-1]) / (u * u) if u > 0 else None,
        }
    out["D1_u_regime"] = rows

    # ---- D2: seamless Gaussian, TRT vs BGK --------------------------------
    seam = {}
    for label, magic in (("TRT_lambda_1_12", LAM), ("TRT_lambda_3_16", 0.1875), ("BGK", None)):
        case = _SmoothCase(**{**_case(0.0).__dict__})
        exact = np.vstack([X.exact_periodic(case, t, D) for t in range(NSTEPS + 1)])
        h = X.run_tb(case, D, magic)
        seam[label] = {
            "rel_l2_final": X.rel_l2(h[-1], exact[-1]),
            "rel_l2_max_over_t": max(X.rel_l2(h[t], exact[t]) for t in range(1, NSTEPS + 1)),
        }
    # the same comparison on the OSS (truncated) initial data, for the record
    for label, magic in (("TRT_lambda_1_12", LAM), ("BGK", None)):
        case = _case(0.0)
        exact = np.vstack([X.exact_periodic(case, t, D) for t in range(NSTEPS + 1)])
        h = X.run_tb(case, D, magic)
        seam[label + "__truncated_ic"] = {
            "rel_l2_final": X.rel_l2(h[-1], exact[-1]),
            "rel_l2_max_over_t": max(X.rel_l2(h[t], exact[t]) for t in range(1, NSTEPS + 1)),
        }
    out["D2_seamless_vs_truncated"] = seam
    return out


if __name__ == "__main__":
    res = main()
    with open(os.path.join(HERE, "diagnostics.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))