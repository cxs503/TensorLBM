"""Verification-debt closure for GAP-LBM-T3-BENCH / poiseuille_3d_ellipse.

The pending benchmark `benchmarks/pending/poiseuille_3d_ellipse/` records that
the s^Q single-scale comparison fails at a=20 (5.11% > 3% gate) because the
digital (staircase) ellipse wall is ANISOTROPIC — a single area-equivalent scale
cannot represent its hydraulic geometry.  The promotion path named in
benchmarks/TODO.md is a TWO-OBSERVABLE (Q + fully-developed pressure gradient G)
hydraulic inversion of the effective semi-axes (a_e, b_e).

This test verifies the ANALYTIC core of that inversion (the closed-form
relations and the inversion map), which is the part that must be exact before
any simulation is trusted.  It does NOT run the heavy 3D LBM simulation; it
checks the mathematics that the simulation's Q and G observables feed into.

Analytic relations (steady, fully developed, no-slip, continuous ellipse a_e,b_e):
    u(y,z) = U_c * (1 - (y/a_e)^2 - (z/b_e)^2)
    Q      = pi * a_e * b_e * U_c / 2
    G      = 2*nu*U_c*(a_e^2 + b_e^2) / (a_e^2 * b_e^2)      [G = dp/L]
Inversion (U_c = 2*u_in imposed by the uniform-velocity inlet):
    P = (a_e*b_e)^2 = (2*Q/(pi*U_c))^2          <- from Q  (integral observable)
    S = a_e^2 + b_e^2 = G*P/(2*nu*U_c)          <- from G  (integral observable)
    a_e = sqrt((S + sqrt(S^2 - 4P))/2),  b_e = sqrt((S - sqrt(S^2 - 4P))/2)
"""

from __future__ import annotations

import math

import pytest

CS2 = 1.0 / 3.0


def invert_two_observable(Q: float, G: float, U_c: float, nu: float) -> dict:
    """Invert (a_e, b_e) from (Q, G) with U_c imposed (closed form)."""
    P = (2.0 * Q / (math.pi * U_c)) ** 2          # (a_e*b_e)^2
    S = G * P / (2.0 * nu * U_c)                  # a_e^2 + b_e^2
    disc = S * S - 4.0 * P
    if disc < 0:
        raise ValueError(f"non-physical: S^2-4P = {disc} < 0")
    a_e = math.sqrt((S + math.sqrt(disc)) / 2.0)
    b_e = math.sqrt((S - math.sqrt(disc)) / 2.0)
    return {"a_e": a_e, "b_e": b_e, "P": P, "S": S}


def forward_observables(a_e: float, b_e: float, U_c: float, nu: float) -> dict:
    """Forward map: (a_e, b_e, U_c, nu) -> (Q, G)."""
    Q = math.pi * a_e * b_e * U_c / 2.0
    G = 2.0 * nu * U_c * (a_e * a_e + b_e * b_e) / (a_e * a_e * b_e * b_e)
    return {"Q": Q, "G": G}


# ---------------------------------------------------------------------------
# Round-trip: forward -> invert must recover the exact semi-axes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a_e,b_e,U_c,nu",
    [
        (20.2252, 10.1126, 0.04, 0.1),   # a=20 s^Q-scaled nominal (anisotropic)
        (40.2041, 20.1020, 0.04, 0.1),   # a=40 s^Q-scaled nominal
        (20.0, 10.0, 0.04, 0.1),         # exact nominal 2:1
        (30.0, 15.0, 0.04, 0.1),         # exact nominal 2:1, larger
        (25.0, 12.5, 0.04, 0.1),         # exact nominal 2:1
    ],
)
def test_inversion_round_trip_recovers_semiaxes(a_e, b_e, U_c, nu):
    """Forward (Q,G) then invert must recover (a_e, b_e) to machine precision."""
    fwd = forward_observables(a_e, b_e, U_c, nu)
    inv = invert_two_observable(fwd["Q"], fwd["G"], U_c, nu)
    assert inv["a_e"] == pytest.approx(a_e, rel=1e-12)
    assert inv["b_e"] == pytest.approx(b_e, rel=1e-12)


def test_inversion_recovers_anisotropic_geometry():
    """The two-observable map must resolve an ANISOTROPIC effective geometry.

    The s^Q single-scale failure is precisely that a single scale cannot
    represent a_e != b_e * (a/b).  The two-observable inversion must recover
    the true anisotropic semi-axes from the two independent integral
    observables (Q and G), which is the property that makes it a valid
    promotion path.
    """
    # Anisotropic effective geometry: a_e = 20.3, b_e = 9.9 (a/b = 2.05, not 2)
    a_e, b_e = 20.3, 9.9
    U_c, nu = 0.04, 0.1
    fwd = forward_observables(a_e, b_e, U_c, nu)
    inv = invert_two_observable(fwd["Q"], fwd["G"], U_c, nu)
    assert inv["a_e"] == pytest.approx(a_e, rel=1e-12)
    assert inv["b_e"] == pytest.approx(b_e, rel=1e-12)
    # The single-scale s^Q would give a_e = b_e * 2 = 19.8 (wrong); the
    # two-observable map recovers the true 20.3/9.9 split.
    assert abs(inv["a_e"] - 2.0 * inv["b_e"]) > 0.1


def test_inversion_requires_two_observables():
    """Q alone is insufficient: it fixes only the product a_e*b_e (P).

    A single observable (Q) leaves the semi-axis ratio free; the pressure
    gradient G is the second observable that fixes the sum S = a_e^2 + b_e^2.
    This is the mathematical reason the s^Q single-scale path cannot close the
    anisotropic-wall verification debt.
    """
    # Two different (a_e, b_e) with the SAME product P -> same Q, different G.
    U_c, nu = 0.04, 0.1
    # (20, 10) and (25, 8): both have a_e*b_e = 200 -> same Q.
    q1 = forward_observables(20.0, 10.0, U_c, nu)["Q"]
    q2 = forward_observables(25.0, 8.0, U_c, nu)["Q"]
    assert q1 == pytest.approx(q2, rel=1e-12)          # same Q (same P)
    g1 = forward_observables(20.0, 10.0, U_c, nu)["G"]
    g2 = forward_observables(25.0, 8.0, U_c, nu)["G"]
    assert g1 != pytest.approx(g2, rel=1e-3)           # different G (different S)


def test_mass_conservation_renormalization_geometry_only():
    """U_c' = Q / M_d is a pure geometric count (no profile fit).

    The hydraulic-inversion primary metric renormalizes U_c by the geometric
    sum M_d(a_e,b_e) = sum_{digital fluid cells} (1 - (y/a_e)^2 - (z/b_e)^2)
    over the digital cross-section.  This is a geometry-only operation: it
    must equal Q / (pi*a_e*b_e/2) when the profile is exactly the parabola.
    """
    # For a continuous ellipse, M_d = pi*a_e*b_e/2 (the area under the parabola).
    a_e, b_e = 20.0, 10.0
    U_c = 0.04
    Q = forward_observables(a_e, b_e, U_c, 0.1)["Q"]
    M_d_cont = math.pi * a_e * b_e / 2.0
    U_c_prime = Q / M_d_cont
    assert U_c_prime == pytest.approx(U_c, rel=1e-12)


def test_analytic_profile_is_elliptic_paraboloid():
    """The analytic profile u = U_c*(1 - (y/a)^2 - (z/b)^2) is an exact
    elliptic paraboloid: its level sets are concentric ellipses with the
    same semi-axis ratio a/b (the property the digital staircase wall
    must reproduce to pass the <3% gate)."""
    a_e, b_e = 20.0, 10.0
    U_c = 0.04
    # Sample a few points on the same level set u = 0.5*U_c.
    # (y/a)^2 + (z/b)^2 = 0.5  ->  y = a*sqrt(0.5)*cos(t), z = b*sqrt(0.5)*sin(t)
    import numpy as np

    t = np.linspace(0, 2 * np.pi, 8)
    y = a_e * np.sqrt(0.5) * np.cos(t)
    z = b_e * np.sqrt(0.5) * np.sin(t)
    u = U_c * (1.0 - (y / a_e) ** 2 - (z / b_e) ** 2)
    assert np.allclose(u, 0.5 * U_c, rtol=1e-12)
    # The semi-axis ratio of every level set is a/b = 2 (constant).
    assert a_e / b_e == pytest.approx(2.0)
