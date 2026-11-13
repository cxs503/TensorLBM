"""Tests for the automatic Chapman--Enskog derivation (:mod:`tensorlbm.chapman_enskog`).

The decisive property of an *automatic* derivation is that it is data driven: the
transport coefficients must come out of the ``(stencil, collision rates)`` input
and must react to exactly those inputs that can influence them.  The tests below
therefore mix closed-form pins with invariance probes (a rate that cannot change a
coefficient must leave it untouched), and close with one small but real solver
round trip so the test suite would fail if the derivation and the solver drifted
apart.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from tensorlbm import chapman_enskog as ce


TAUS = (0.6, 0.8, 1.2)
STENCILS = ((ce.D2Q9, 9), (ce.D3Q19, 19), (ce.D3Q27, 27))

_CACHE: dict[tuple[str, str], "ce.Derivation"] = {}


def derive(stencil, collision) -> "ce.Derivation":
    """Memoised derivation (a D3Q19/D3Q27 expansion costs 12 s / 21 s)."""
    key = (stencil.name, collision.name)
    if key not in _CACHE:
        _CACHE[key] = ce.derive_equivalent_equation(stencil, collision)
    return _CACHE[key]


# --------------------------------------------------------------------------- #
# closed-form pins
# --------------------------------------------------------------------------- #
CASES = [(ce.D2Q9, 9, tau) for tau in TAUS] + [(ce.D3Q19, 19, 0.8), (ce.D3Q27, 27, 0.8)]


@pytest.mark.parametrize("stencil,q,tau", CASES)
def test_shear_viscosity_matches_closed_form(stencil, q, tau):
    """``nu = (tau - 1/2) c_s^2`` for BGK on every stencil, from the derivation."""
    d = derive(stencil, ce.bgk(tau, q))
    cs2 = float(stencil.cs2_exact())
    assert cs2 == pytest.approx(1.0 / 3.0, rel=1e-15)
    assert d.nu_value == pytest.approx((tau - 0.5) * cs2, rel=1e-12)
    assert d.cs2_value == pytest.approx(cs2, rel=1e-15)


@pytest.mark.parametrize("stencil,q", STENCILS)
def test_bgk_bulk_viscosity_is_two_over_dim_times_nu(stencil, q):
    """The athermal BGK bulk viscosity ``zeta = 2 nu / D`` falls out of the expansion.

    This is the non-trivial literature result (Dellar 2001): the single-rate
    athermal model is *not* bulk-viscosity free, and the derivation must produce
    the dimension-dependent ratio without being told about it.
    """
    d = derive(stencil, ce.bgk(0.8, q))
    assert d.zeta_value == pytest.approx(2.0 * d.nu_value / stencil.dim, rel=1e-12)
    # ... and the observable sound-damping coefficient is consequently 2 nu in any
    # dimension, which is exactly what the discrete eigenvalues give (see docs).
    assert ce.longitudinal_viscosity(d) == pytest.approx(2.0 * d.nu_value, rel=1e-12)


def test_conservation_moments_of_first_order_population_vanish():
    """Mass and momentum of ``f^(1)`` vanish identically, else the ansatz is void."""
    for stencil, q in STENCILS:
        d = derive(stencil, ce.bgk(0.8, q))
        assert d.conservation_ok is True
        assert d.isotropy == {"second_order_isotropic": True, "fourth_order_isotropic": True}


def test_stress_equals_navier_stokes_tensor_at_zero_velocity():
    """``sigma_ab = -2 rho nu (S_ab - theta delta/D) - rho zeta theta delta`` exactly."""
    d = derive(ce.D2Q9, ce.bgk(0.8, 9))
    dim = 2
    rho = sp.Symbol("rho", positive=True)
    grad = sp.Matrix(dim, dim, lambda a, b: sp.Symbol(f"d{a}_{b}", real=True))
    theta = grad[0, 0] + grad[1, 1]
    subs = {grad[a, b]: sp.Rational((7 * a + 3 * b + 1), 11) for a in range(dim) for b in range(dim)}
    subs[rho] = sp.Rational(13, 10)
    for a in range(dim):
        for b in range(dim):
            deviatoric = grad[a, b] + grad[b, a] - (2 * theta if a == b else 0) / dim
            expected = (
                -rho * d.nu_symbolic * deviatoric
                - (rho * d.zeta_symbolic * theta if a == b else 0)
            )
            got = d.stress_linear[a, b]
            assert sp.simplify((got - expected).subs(subs)) == 0


# --------------------------------------------------------------------------- #
# invariance probes: the derivation must be driven by the input, not by a formula
# --------------------------------------------------------------------------- #
def test_cs2_is_derived_from_the_weights_not_assumed():
    """A non-standard stencil gives a non-standard ``c_s^2`` (and a flagged closure)."""
    custom = ce.Stencil(
        name="D2Q5-custom",
        c=((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)),
        w=(sp.Rational(1, 2), sp.Rational(1, 8), sp.Rational(1, 8), sp.Rational(1, 8), sp.Rational(1, 8)),
    )
    assert float(custom.cs2_exact()) == pytest.approx(0.25, rel=1e-15)
    d = derive(custom, ce.bgk(0.8, 5))
    # D2Q5 has no diagonal velocities, so its fourth moment cannot represent a
    # shear stress at all: the derivation returns nu = 0 -- and says so through the
    # closure flag instead of returning a plausible-looking number.
    assert d.isotropy["fourth_order_isotropic"] is False
    assert d.nu_value == 0.0
    # on the stencils whose fourth moment *is* isotropic the coefficient follows
    # the stencil's own cs2 (D3Q19: cs2 = 1/3 -> nu = (tau - 1/2)/3, checked above)
    diag = derive(ce.D3Q19, ce.bgk(0.8, 19))
    assert diag.nu_value == pytest.approx(0.3 * float(ce.D3Q19.cs2_exact()), rel=1e-12)


def test_trt_shear_rate_is_the_even_family_rate():
    """``nu`` follows the rate of the (f_i + f_-i) family only; ``omega_-`` is inert."""
    base = derive(ce.D2Q9, ce.trt(ce.D2Q9, 1.9, 1.25))
    assert base.nu_value == pytest.approx((1.0 / 1.9 - 0.5) / 3.0, rel=1e-12)
    shifted = derive(ce.D2Q9, ce.trt(ce.D2Q9, 1.9, 1.70))
    assert shifted.nu_value == pytest.approx(base.nu_value, rel=1e-12)
    assert shifted.zeta_value == pytest.approx(base.zeta_value, rel=1e-12)
    faster = derive(ce.D2Q9, ce.trt(ce.D2Q9, 1.5, 1.25))
    assert faster.nu_value == pytest.approx((1.0 / 1.5 - 0.5) / 3.0, rel=1e-12)
    assert faster.nu_value > base.nu_value


def test_mrt_shear_rate_sets_nu_and_energy_rate_sets_zeta():
    """In MRT only the stress rate moves ``nu``; the energy rate moves ``zeta``."""
    ref = None
    for s_e in (1.2, 1.64, 1.9):
        d = derive(ce.D2Q9, ce.d2q9_mrt(0.8, s_e=s_e, s_q=1.7))
        if ref is None:
            ref = d
        assert d.nu_value == pytest.approx((0.8 - 0.5) / 3.0, rel=1e-12)
        assert d.nu_value == pytest.approx(ref.nu_value, rel=1e-12)
    zetas = [
        derive(ce.D2Q9, ce.d2q9_mrt(0.8, s_e=s_e)).zeta_value
        for s_e in (1.2, 1.64, 1.9, 2.0)
    ]
    assert len({round(z, 9) for z in zetas}) == 4, "zeta must react to the energy-mode rate"
    assert zetas[-1] == pytest.approx(0.0, abs=1e-9)  # 1/s_e = 1/2 -> no bulk damping
    # tau still controls nu independently of s_e
    d_fast = derive(ce.D2Q9, ce.d2q9_mrt(0.6))
    assert d_fast.nu_value == pytest.approx((0.6 - 0.5) / 3.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# solver round trip (small, fast)
# --------------------------------------------------------------------------- #
def test_measured_shear_decay_matches_derived_viscosity():
    """One real D2Q9 BGK run through ``tensorlbm.solver`` against the derivation."""
    import numpy as np
    import torch

    from tensorlbm.d2q9 import equilibrium, macroscopic
    from tensorlbm.solver import collide_bgk, stream

    torch.set_default_dtype(torch.float64)
    nx, ny, tau = 32, 8, 0.8
    k = 2.0 * math.pi / nx
    profile = torch.sin(k * torch.arange(nx, dtype=torch.float64)).view(1, nx).expand(ny, nx)
    zero = torch.zeros((ny, nx), dtype=torch.float64)
    f = equilibrium(torch.ones_like(zero), zero, 1.0e-3 * profile)
    ts, amps = [], []
    for step in range(1, 601):
        f = stream(collide_bgk(f, tau))
        if step >= 300 and step % 10 == 0:
            _, _, uy = macroscopic(f)
            amps.append(float((uy * profile).mean() / (profile * profile).mean()))
            ts.append(step)
    slope = np.polyfit(np.array(ts, float), np.log(np.abs(np.array(amps))), 1)[0]
    measured = -slope / k**2
    derived = derive(ce.D2Q9, ce.bgk(tau, 9)).nu_value
    assert derived == pytest.approx(0.1, rel=1e-12)
    assert measured == pytest.approx(derived, rel=2.0e-2)
