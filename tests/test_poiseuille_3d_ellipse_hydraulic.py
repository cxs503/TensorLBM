"""Regression tests for the two-observable hydraulic inversion used by
benchmarks/pending/poiseuille_3d_ellipse/run_hydraulic.py (B14-ellipse
verification-debt closure).

The inversion determines the effective semi-axes (a_e, b_e) of the digital
staircase elliptical pipe from two independent global integral observables —
the flow rate Q and the fully-developed pressure gradient G — with the peak
velocity U_c = 2*u_in imposed by the uniform-velocity inlet.  No velocity
profile is used, so the subsequent profile comparison is a genuine test of the
solver, not a fit.

Analytic relations (steady, fully developed, no-slip, continuous ellipse):
    u(y,z) = U_c (1 - (y/a_e)^2 - (z/b_e)^2)
    Q      = pi a_e b_e U_c / 2
    G      = 2 nu U_c (a_e^2 + b_e^2) / (a_e^2 b_e^2)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "pending" / "poiseuille_3d_ellipse"))

from run_hydraulic import digital_M, invert_two_observable  # noqa: E402


def _analytic_QG(a, b, U_c, nu):
    G = 2.0 * nu * U_c * (a * a + b * b) / (a * a * b * b)
    Q = math.pi * a * b * U_c / 2.0
    return Q, G


def test_inversion_roundtrip_exact():
    """Analytic (a,b) -> (Q,G) -> invert -> (a,b) must recover (a,b) to fp noise."""
    nu = 0.1
    U_c = 0.04
    for (a, b) in [(20.0, 10.0), (40.0, 20.0), (21.18, 9.85), (30.0, 12.0)]:
        Q, G = _analytic_QG(a, b, U_c, nu)
        inv = invert_two_observable(Q, G, U_c, nu)
        assert inv["ok"]
        assert abs(inv["a_e"] - a) / a < 1e-12
        assert abs(inv["b_e"] - b) / b < 1e-12


def test_inversion_aspect_preserved():
    """The inversion must preserve the aspect ratio a_e/b_e = sqrt(S+..)/.. :
    for a true ellipse the recovered ratio equals the input ratio."""
    nu = 0.1
    U_c = 0.04
    for (a, b) in [(20.0, 10.0), (40.0, 20.0), (25.0, 10.0)]:
        Q, G = _analytic_QG(a, b, U_c, nu)
        inv = invert_two_observable(Q, G, U_c, nu)
        assert inv["ok"]
        assert abs(inv["a_e"] / inv["b_e"] - a / b) < 1e-12


def test_inversion_rejects_unphysical():
    """G too small for the given Q (disc < 0) must be flagged, not crash."""
    inv = invert_two_observable(Q=100.0, G=1e-9, U_c=0.04, nu=0.1)
    assert not inv["ok"]
    assert math.isnan(inv["a_e"])


def test_inversion_scales_with_geometry():
    """Doubling the geometry (a,b)->(2a,2b) at fixed U_c,nu: Q x4, G /4,
    and the inversion must return exactly doubled semi-axes."""
    nu = 0.1
    U_c = 0.04
    Q1, G1 = _analytic_QG(20.0, 10.0, U_c, nu)
    Q2, G2 = _analytic_QG(40.0, 20.0, U_c, nu)
    assert abs(Q2 / Q1 - 4.0) < 1e-12
    assert abs(G1 / G2 - 4.0) < 1e-12
    i1 = invert_two_observable(Q1, G1, U_c, nu)
    i2 = invert_two_observable(Q2, G2, U_c, nu)
    assert abs(i2["a_e"] - 2 * i1["a_e"]) < 1e-10
    assert abs(i2["b_e"] - 2 * i1["b_e"]) < 1e-10


def test_digital_M_continuous_limit():
    """M_d over a digital mask of the ellipse (a_e,b_e) must approach the
    continuous value pi*a_e*b_e/2 (the parabola's area integral) as the
    staircase is refined; the boundary-layer error is O(perimeter) so the
    RELATIVE error scales as 1/R."""
    def M_for(R, n):
        y = np.arange(n) - n // 2
        yy, zz = np.meshgrid(y, y, indexing="xy")
        fluid = (yy / R) ** 2 + (zz / R) ** 2 <= 1.0
        return digital_M(fluid, R, R, n // 2, n // 2)
    M1 = M_for(200.0, 2001)
    M2 = M_for(400.0, 4001)
    M1c = math.pi * 200.0 * 200.0 / 2.0
    M2c = math.pi * 400.0 * 400.0 / 2.0
    e1 = abs(M1 - M1c) / M1c
    e2 = abs(M2 - M2c) / M2c
    assert e1 < 0.02
    assert e2 < 0.02
    assert e2 < e1  # refinement reduces the relative error


def test_digital_M_positive_and_bounded():
    """For the actual a=20/b=10 digital cross-section, M_d must lie between
    0 and the cell count (each term < 1 in the fluid region)."""
    a, b = 20, 10
    ny, nz = 2 * a + 3, 2 * b + 3
    yc, zc = a + 1, b + 1
    iy = np.arange(ny)
    iz = np.arange(nz)
    YY, ZZ = np.meshgrid(iy, iz, indexing="xy")
    fluid = ((YY - yc) / a) ** 2 + ((ZZ - zc) / b) ** 2 <= 1.0
    M = digital_M(fluid, 20.2, 10.1, yc, zc)
    n_cells = int(fluid.sum())
    assert 0.0 < M < n_cells
    # continuous reference for the same (a_e,b_e): pi*a_e*b_e/2
    M_cont = math.pi * 20.2 * 10.1 / 2.0
    # digital area is within ~1% of continuous; M_d within a few % of M_cont
    assert abs(M - M_cont) / M_cont < 0.05


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"ALL {len(fns)} TESTS PASSED")
