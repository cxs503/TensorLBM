"""Sediment transport & bed-scour verification surface (mission m7329).

Closes the zero-verification gap for sediment dynamics on TensorLBM.  Four
machine-judgeable anchors:

* **C1** incipient-motion curve: numerical ``theta_cr`` (measured from the
  coupled LBM-shear + Exner/MPM bed response) vs the Soulsby-Whitehouse
  reference, 8 log-spaced ``d*`` in [1, 100], deviation <= 15%.
* **C2** Rouse suspended-load profile: numerically fitted Rouse exponent
  ``Z_num`` vs the analytic ``Z = w_s/(kappa u*)``, relative error < 5%.
* **C3** sediment mass conservation: global bed-volume drift < 1e-6 over a
  long Exner run (conservative discretisation).
* **C4** negative control: ``u* < 0.8 u*_cr``  =>  ``q_s < 1e-12``.
* **C5** single-pile local scour: equilibrium ``S/D`` in the Melville-Chiew
  band [1.0, 2.4] (2-D vertical LBM over a circular pile on a movable bed).
* **C6** end-to-end JSON contract
  ``{shields_curve, rouse_profiles, bed_elevation, scour_depth, mass_budget}``.

The physics lives in ``tensorlbm.sediment``; this file only asserts.  The
scour anchor (C5) is the expensive one and is marked ``slow`` so the fast CI
scope can skip it (``pytest -m "not slow"``); the full suite (``make
test-all``) and the artifact runner execute it.

Run:
    PYTHONPATH=src python3 -m pytest tests/test_sediment_scour.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorlbm.sediment.exner import ExnerConfig, bed_volume, exner_step, mpm_flux
from tensorlbm.sediment.morpho import MorphoConfig, measure_theta_cr
from tensorlbm.sediment.rouse import (
    fit_rouse_exponent,
    rouse_exponent,
    solve_rouse_profile,
)
from tensorlbm.sediment.shields import critical_shields, dimensionless_grain_size
from tensorlbm.sediment.verification import (
    D_STAR_POINTS,
    SedimentScenario,
    build_results,
)

# ---------------------------------------------------------------------------
# Shared scenario (SI units).  The LBM channel supplies u_star_lattice; the
# physical mapping u_scale = 10 m s^-1 per lattice unit converts to SI.
# ---------------------------------------------------------------------------
NU = 1.0e-6
S = 1.65
G = 9.81
W_S = 1.0e-2
H = 1.0
U_SCALE = 10.0
D_REF = 1.0e-4  # reference grain (d* ~ 1.2)


def _scenario(u_star_lattice: float = 0.00866, d: float = D_REF) -> SedimentScenario:
    return SedimentScenario(
        d=d, s=S, g=G, nu=NU, w_s=W_S, h=H,
        u_scale=U_SCALE, u_star_lattice=u_star_lattice,
    )


# ---------------------------------------------------------------------------
# C1: incipient-motion curve
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d_star", D_STAR_POINTS)
def test_c1_shields_curve_point(d_star: float) -> None:
    """Numerical theta_cr within 15% of Soulsby-Whitehouse at each d*."""
    scen = _scenario()
    d = d_star * NU / np.sqrt((S - 1.0) * G)
    theta_ref = critical_shields(d_star)
    mc = MorphoConfig(d=d, s=S, g=G, h_ref=H, n_steps=800)
    theta_num = measure_theta_cr(scen.u_star, mc, theta_cr=theta_ref)
    rel = abs(theta_num - theta_ref) / theta_ref
    assert rel <= 0.15, f"d*={d_star}: rel err {rel:.3f} > 15%"


def test_c1_curve_max_error() -> None:
    """The whole curve (8 points) stays within 15%."""
    scen = _scenario()
    max_rel = 0.0
    for d_star in D_STAR_POINTS:
        d = d_star * NU / np.sqrt((S - 1.0) * G)
        theta_ref = critical_shields(d_star)
        mc = MorphoConfig(d=d, s=S, g=G, h_ref=H, n_steps=800)
        theta_num = measure_theta_cr(scen.u_star, mc, theta_cr=theta_ref)
        max_rel = max(max_rel, abs(theta_num - theta_ref) / theta_ref)
    assert max_rel <= 0.15


# ---------------------------------------------------------------------------
# C2: Rouse profile
# ---------------------------------------------------------------------------
def test_c2_rouse_exponent() -> None:
    """Fitted Rouse exponent within 5% of Z = w_s/(kappa u*)."""
    scen = _scenario()
    Z_ref = rouse_exponent(scen.w_s, scen.u_star)
    y, c = solve_rouse_profile(scen.w_s, scen.u_star, scen.h)
    Z_num = fit_rouse_exponent(y, c, scen.h)
    rel = abs(Z_num - Z_ref) / Z_ref
    assert rel < 0.05, f"Z rel err {rel:.3f} >= 5%"


def test_c2_rouse_profile_shape() -> None:
    """The profile is monotonically decreasing (more sediment near the bed)."""
    scen = _scenario()
    y, c = solve_rouse_profile(scen.w_s, scen.u_star, scen.h)
    assert np.all(np.diff(c) < 0.0), "Rouse profile must decrease with height"


# ---------------------------------------------------------------------------
# C3: mass conservation
# ---------------------------------------------------------------------------
def test_c3_mass_conservation() -> None:
    """Global bed-volume drift < 1e-6 over a long Exner run."""
    scen = _scenario()
    n = 200
    dx = 0.1
    x = (np.arange(n) + 0.5) * dx
    z_b = 0.05 * scen.d * np.sin(2.0 * np.pi * x / (n * dx))
    cfg = ExnerConfig(d=scen.d, s=S, g=G, theta_cr=0.02)
    u_star = scen.u_star
    denom = (S - 1.0) * G * scen.d
    h_ref = H
    v0 = bed_volume(z_b, dx)
    for _ in range(3000):
        h = h_ref - z_b
        u_loc = u_star * (h_ref / h) ** 1.5
        theta = u_loc**2 / denom
        z_b = exner_step(z_b, mpm_flux(theta, cfg), dx, 1.0, cfg)
    v1 = bed_volume(z_b, dx)
    drift = abs(v1 - v0) / max(abs(v0), 1e-30)
    assert drift < 1e-6, f"mass drift {drift:.3e} >= 1e-6"


# ---------------------------------------------------------------------------
# C4: negative control
# ---------------------------------------------------------------------------
def test_c4_negative_control() -> None:
    """u* < 0.8 u*_cr  =>  q_s < 1e-12 (incipient-motion gate)."""
    scen = _scenario()
    cfg = ExnerConfig(d=scen.d, s=S, g=G, theta_cr=0.045)
    u_cr = np.sqrt(0.045 * (S - 1.0) * G * scen.d)
    u_test = 0.8 * u_cr
    theta = scen.theta(u_test)
    q = mpm_flux(theta, cfg)
    assert float(q) < 1e-12, f"q_s={q:.3e} >= 1e-12"


# ---------------------------------------------------------------------------
# C5: single-pile scour (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_c5_single_pile_scour() -> None:
    """Equilibrium scour depth S/D in the Melville-Chiew band [1.0, 2.4]."""
    from tensorlbm.sediment.scour import run_single_pile_scour

    result = run_single_pile_scour()
    s_over_d = result["S_over_D"]
    assert s_over_d is not None
    assert 1.0 <= s_over_d <= 2.4, f"S/D={s_over_d:.3f} outside [1.0, 2.4]"


# ---------------------------------------------------------------------------
# C6: end-to-end JSON contract
# ---------------------------------------------------------------------------
def test_c6_json_contract() -> None:
    """build_results emits the C6 keys and is json-serialisable."""
    results = build_results(u_star_lattice=0.00866)
    for key in ("shields_curve", "rouse_profiles", "bed_elevation",
                "scour_depth", "mass_budget"):
        assert key in results, f"missing key {key}"
    # must be json-serialisable
    json.dumps(results)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
