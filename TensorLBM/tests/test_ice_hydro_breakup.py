"""Regression tests for the ice-cover hydroelastic breakup module (p338w3-n03).

Run with single-threaded BLAS (BLAS thread contention makes small dense solves
~300x slower on loaded build machines):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
        python -m pytest tests/test_ice_hydro_breakup.py -q
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

# make sure *this* worktree's tensorlbm is imported (the shared conftest may put
# another worktree on sys.path first)
_SRC = str(pathlib.Path(__file__).resolve().parents[1] / "src")
while _SRC in sys.path:
    sys.path.remove(_SRC)
sys.path.insert(0, _SRC)
for _m in [m for m in list(sys.modules) if m == "tensorlbm" or m.startswith("tensorlbm.")]:
    del sys.modules[_m]
import tensorlbm  # noqa: E402
assert tensorlbm.__file__.startswith(_SRC), tensorlbm.__file__

import numpy as np
import pytest

from tensorlbm.ice_hydro.analytical import (
    CoverProperties,
    antinode_positions,
    characteristic_length,
    crack_moment_capacity,
    critical_velocity_k2,
    flexural_gravity_omega,
    mode_wavenumbers,
    phase_speed,
    plate_rigidity,
    steady_divergence_velocity,
    u_cr_flexural_gravity,
    u_cr_k2,
)
from tensorlbm.ice_hydro.bonded_cover import BondedCover
from tensorlbm.ice_hydro.lbm_channel import poiseuille_channel


# --------------------------------------------------------------------------- analytical
def test_plate_rigidity_and_length_scales():
    D = plate_rigidity(5.0e9, 1.0, 0.3)
    assert D == pytest.approx(5.0e9 / (12.0 * (1 - 0.09)), rel=1e-12)
    # D scales with h^3
    assert plate_rigidity(5.0e9, 2.0, 0.3) == pytest.approx(8.0 * D, rel=1e-12)
    lc = characteristic_length(D, 1025.0, 9.81)
    assert lc == pytest.approx((D / (1025.0 * 9.81)) ** 0.25, rel=1e-12)


def test_k2_criterion_closed_form_matches_mode_minimum():
    props = CoverProperties(h_i=1.0)
    D = props.D
    u_cr, k_c = u_cr_k2(D, props.rho_w, props.g)
    assert u_cr == pytest.approx((4.0 * D * props.g / props.rho_w) ** 0.25, rel=1e-12)
    # the closed form is the minimum of the mode-wise expression
    k = np.linspace(0.2 * k_c, 3.0 * k_c, 4001)
    assert u_cr <= critical_velocity_k2(k, D, props.rho_w, props.g).min() + 1e-9


def test_phase_speed_equals_steady_divergence_velocity():
    """The flexural-gravity phase speed and the steady divergence velocity are the
    same first-principles quantity (see analytical module docstring)."""
    props = CoverProperties(h_i=0.7)
    k = np.array([0.01, 0.03, 0.05, 0.1, 0.3])
    c = phase_speed(k, props.D, props.h, props.rho_w, props.g)
    u = steady_divergence_velocity(k, props.D, props.h, props.rho_w, props.g)
    assert np.allclose(c, u, rtol=1e-12)


def test_deep_water_dispersion_limits():
    """omega^2 -> g k (gravity) for long waves and -> D k^5/rho_w for short waves."""
    props = CoverProperties(h_i=1.0)
    D, rho, g = props.D, props.rho_w, props.g
    h_deep = 1.0e5                     # enforce tanh(kh) -> 1 (deep water)
    k_long = 1e-3
    omega = flexural_gravity_omega(np.array([k_long]), D, h_deep, rho, g)[0]
    assert omega**2 == pytest.approx(g * k_long, rel=2e-2)
    k_short = 2.0
    omega = flexural_gravity_omega(np.array([k_short]), D, h_deep, rho, g)[0]
    assert omega**2 == pytest.approx(D * k_short**5 / rho, rel=5e-2)


def test_flexural_gravity_min_phase_speed_below_k2_criterion():
    """The two criteria are distinct: the prescribed k^2 form is a membrane-buckling
    criterion and is substantially higher than the flexural-gravity minimum."""
    props = CoverProperties(h_i=1.0)
    u_k2, _ = u_cr_k2(props.D, props.rho_w, props.g)
    u_fg, k_c = u_cr_flexural_gravity(props.D, props.h, props.rho_w, props.g)
    assert u_fg < u_k2
    assert 1.5 < u_k2 / u_fg < 10.0
    # the critical wavenumber of the flexural-gravity criterion sits near the
    # elastic-foundation wavenumber (rho g / 3D)^(1/4)
    assert k_c == pytest.approx((props.rho_w * props.g / (3.0 * props.D)) ** 0.25, rel=0.25)


def test_crack_moment_capacity_scales_with_h_squared():
    assert crack_moment_capacity(7.0e5, 1.0) == pytest.approx(7.0e5 / 6.0)
    assert crack_moment_capacity(7.0e5, 2.0) == pytest.approx(4 * 7.0e5 / 6.0)


def test_antinode_positions():
    assert antinode_positions(200.0, 1) == [100.0]
    assert antinode_positions(200.0, 3) == pytest.approx([100.0 / 3, 100.0, 500.0 / 3])
    assert mode_wavenumbers(3, 200.0)[2] == pytest.approx(3 * np.pi / 200.0)


# --------------------------------------------------------------------------- DEM lattice
def test_dem_reproduces_euler_bernoulli_simply_supported_beam():
    """Uniform load on a simply supported beam without foundation: w_max = 5qL^4/(384D).

    The bonded-particle lattice must reproduce the continuum bending stiffness.
    """
    props = dataclasses.replace(CoverProperties(h_i=1.0), rho_w=0.0)  # switch foundation off
    cover = BondedCover(props=props, n_nodes=201)
    q = 1.0e4                     # Pa, uniform pressure
    w = cover.static_deflection(np.full(cover.N, q))
    w_analytic = 5.0 * q * props.L**4 / (384.0 * props.D)
    assert np.max(np.abs(w)) == pytest.approx(w_analytic, rel=3e-2)


def test_dem_bending_moment_matches_curvature_definition():
    props = CoverProperties(h_i=1.0)
    cover = BondedCover(props=props, n_nodes=65)
    w = np.sin(np.pi * cover.x / props.L)
    M = cover.bond_moments(w)
    k = np.pi / props.L
    M_ref = -props.D * k**2 * np.sin(np.pi * cover.x / props.L)
    assert np.allclose(M[1:-1], M_ref[1:-1], rtol=1e-2, atol=1e-3 * np.abs(M_ref).max())


def test_dem_onset_momentum_matches_prescribed_criterion():
    props = CoverProperties(h_i=1.0)
    cover = BondedCover(props=props, n_nodes=129)
    u_dem = cover.onset_velocity("momentum")
    u_ref, _ = u_cr_k2(props.D, props.rho_w, props.g)
    assert abs(u_dem - u_ref) / u_ref <= 0.15


def test_dem_onset_hydro_matches_flexural_gravity_criterion():
    props = CoverProperties(h_i=1.0)
    cover = BondedCover(props=props, n_nodes=129)
    u_dem = cover.onset_velocity("hydro")
    u_ref, _ = u_cr_flexural_gravity(props.D, props.h, props.rho_w, props.g)
    assert abs(u_dem - u_ref) / u_ref <= 0.15


def test_onset_increases_with_ice_thickness():
    onsets = []
    for h_i in (0.25, 0.5, 1.0):
        props = CoverProperties(h_i=h_i)
        onsets.append(BondedCover(props=props, n_nodes=97).onset_velocity("hydro"))
    assert onsets[0] < onsets[1] < onsets[2]


def test_no_crack_below_threshold_and_crack_above():
    """J3-style check on a short dynamic run (single thickness)."""
    props = CoverProperties(h_i=1.0)
    cover = BondedCover(props=props, n_nodes=97)
    u_cr = cover.onset_velocity("hydro")
    below = BondedCover(props=props, n_nodes=97).transient(
        "hydro", U=0.8 * u_cr, t_end=20.0, seed_mode=3)
    above = BondedCover(props=props, n_nodes=97).transient(
        "hydro", U=1.15 * u_cr, t_end=40.0, seed_mode=3)
    assert below["n_cracks"] == 0
    assert above["n_cracks"] >= 1


# --------------------------------------------------------------------------- LBM
def test_lbm_poiseuille_profile_within_5_percent():
    r = poiseuille_channel(ny=20, force=1.0e-5, steps=6000)
    assert r["max_relative_deviation"] <= 0.05
    assert r["curvature_residual"] <= 0.05