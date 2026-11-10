"""Tests for :mod:`tensorlbm.ibm_openwater` (campaign ``p233w1-N07``).

Two families:

* **contract tests** (geometry, reference tables, IBM stencil/forcing
  invariants) — cheap, no long time integration;
* **verification smoke tests** — a very short cylinder and rotor run that pin
  down the sign conventions and the momentum-exact forcing.  The quantitative
  sweeps (AC1/AC2/AC4) live in ``scripts/p233w1_n07_ibm_openwater.py`` and
  their numbers are read back from JSON artefacts, not from these tests.
"""

from __future__ import annotations

import math

import os

import pytest
import torch

# The shared build box is heavily oversubscribed; torch's default of one thread
# per core makes the tiny 2-D/3-D smoke runs an order of magnitude slower.
torch.set_num_threads(int(os.environ.get("TL_THREADS", "4")))

from tensorlbm.ibm_force_transfer import build_stencil
from tensorlbm.ibm_openwater import (
    CYLINDER_CD_2D,
    KP505_10KQ,
    KP505_KT,
    CylinderConfig,
    OpenWaterConfig,
    PropellerGeometry,
    blade_markers,
    cylinder_drag_2d,
    eta_error_is_monotone,
    hub_mask,
    kp505_reference,
    momentum_balance_audit,
    open_water_efficiency,
    propeller_markers,
    rotate_markers,
    rotor_target_velocity,
    run_open_water_point,
)


# --------------------------------------------------------------------------- #
# reference data (AC2 audit chain)
# --------------------------------------------------------------------------- #


def test_kp505_thrust_decreases_with_advance_ratio():
    js = sorted(KP505_KT)
    kts = [KP505_KT[j] for j in js]
    assert all(b < a for a, b in zip(kts, kts[1:]))


def test_kp505_torque_decreases_with_advance_ratio():
    js = sorted(KP505_10KQ)
    kqs = [KP505_10KQ[j] for j in js]
    assert all(b < a for a, b in zip(kqs, kqs[1:]))


def test_reference_interpolation_reproduces_knots():
    for j, kt in KP505_KT.items():
        ref = kp505_reference(j)
        assert ref["KT"] == pytest.approx(kt, abs=1e-12)


def test_open_water_efficiency_formula():
    eta = open_water_efficiency(0.5, 0.37, 0.055)
    assert eta == pytest.approx(0.5 * 0.37 / (2.0 * math.pi * 0.055), rel=1e-12)
    assert open_water_efficiency(0.5, 0.37, 0.0) == 0.0


def test_kp505_efficiency_peaks_in_physical_range():
    js = [0.2 + 0.05 * i for i in range(19)]
    etas = [kp505_reference(j)["eta_0"] for j in js]
    j_max = js[int(max(range(len(etas)), key=lambda i: etas[i]))]
    assert 0.6 <= j_max <= 0.95
    assert 0.4 < max(etas) < 0.85


# --------------------------------------------------------------------------- #
# parametric geometry
# --------------------------------------------------------------------------- #


def test_expanded_area_ratio_is_reproduced():
    geom = PropellerGeometry(diameter=24.0, n_blades=5, blade_area_ratio=0.65)
    n = 2000
    r0, r1 = geom.hub_radius, geom.radius
    area = sum(
        geom.chord(r0 + (r1 - r0) * (i + 0.5) / n) * (r1 - r0) / n for i in range(n)
    )
    target = math.pi * geom.diameter**2 / 4.0 * geom.blade_area_ratio / geom.n_blades
    assert area == pytest.approx(target, rel=2e-3)


def test_pitch_angle_law():
    geom = PropellerGeometry(diameter=24.0, pitch_ratio_07=0.95)
    r = 0.7 * geom.radius
    assert geom.pitch_angle(r) == pytest.approx(math.atan(geom.pitch / (2 * math.pi * r)))
    assert geom.pitch_angle(0.3 * geom.radius) > geom.pitch_angle(0.9 * geom.radius)


def test_blade_markers_stay_between_hub_and_tip():
    geom = PropellerGeometry(diameter=24.0)
    center = (10.0, 12.0, 12.0)
    pts = propeller_markers(geom, 0.3, center=center)
    r = torch.sqrt((pts[:, 1] - center[1]) ** 2 + (pts[:, 2] - center[2]) ** 2)
    assert float(r.min()) >= geom.hub_radius * 0.99
    assert float(r.max()) <= geom.radius * 1.001


def test_blade_marker_chord_extent_matches_geometry():
    geom = PropellerGeometry(diameter=24.0, radial_stations=3)
    pts = blade_markers(geom, 0.0)
    r = torch.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2)
    # markers of the outermost station: chordwise extent along the chord line
    r_max = float(r.max())
    ring = pts[(r > r_max - 1e-9)]
    assert ring.shape[0] >= 2
    span = float((ring.max(dim=0).values - ring.min(dim=0).values).norm())
    chord = geom.chord(r_max)
    assert span <= chord * 1.35 + 0.5
    assert span >= chord * 0.6


def test_marker_count_follows_blade_count_and_resolution():
    g5 = PropellerGeometry(diameter=20.0, n_blades=5)
    assert propeller_markers(g5, 0.0).shape[0] == 5 * blade_markers(g5, 0.0).shape[0]
    # same expanded area ratio -> marker count grows roughly like D^2
    small = propeller_markers(PropellerGeometry(diameter=20.0, radial_stations=6), 0.0)
    large = propeller_markers(PropellerGeometry(diameter=40.0, radial_stations=12), 0.0)
    assert large.shape[0] > 2.5 * small.shape[0]


def test_rotor_target_velocity_is_rigid_rotation():
    pts = propeller_markers(PropellerGeometry(diameter=20.0), 0.4, center=(5.0, 10.0, 10.0))
    omega = 0.02
    u = rotor_target_velocity(pts, omega, center=(5.0, 10.0, 10.0))
    assert torch.allclose(u[:, 0], torch.zeros_like(u[:, 0]))
    r = torch.sqrt((pts[:, 1] - 10.0) ** 2 + (pts[:, 2] - 10.0) ** 2)
    assert torch.allclose(u.norm(dim=1), omega * r, rtol=1e-12, atol=1e-15)
    # purely tangential: u . e_r == 0
    er = torch.stack([torch.zeros_like(r), pts[:, 1] - 10.0, pts[:, 2] - 10.0], dim=1)
    er = er / er.norm(dim=1, keepdim=True)
    assert torch.allclose((u * er).sum(dim=1), torch.zeros_like(r), atol=1e-14)


def test_rotate_markers_preserves_radius_and_advances_azimuth():
    geom = PropellerGeometry(diameter=20.0)
    center = (0.0, 0.0, 0.0)
    pts = blade_markers(geom, 0.0)
    moved = rotate_markers(pts, math.pi / 3, center)
    r0 = pts[:, 1] ** 2 + pts[:, 2] ** 2
    r1 = moved[:, 1] ** 2 + moved[:, 2] ** 2
    assert torch.allclose(r0, r1, rtol=1e-12, atol=1e-14)
    assert torch.allclose(moved[:, 0], pts[:, 0])
    back = rotate_markers(moved, -math.pi / 3, center)
    assert torch.allclose(back, pts, atol=1e-12)


def test_rotate_markers_full_turn_is_identity():
    pts = blade_markers(PropellerGeometry(diameter=16.0), 0.2)
    full = rotate_markers(pts, 2.0 * math.pi, (0.0, 0.0, 0.0))
    assert torch.allclose(full, pts, atol=1e-12)


def test_hub_mask_geometry():
    geom = PropellerGeometry(diameter=24.0)
    shape = (40, 40, 80)
    center = (40.0, 20.0, 20.0)
    mask = hub_mask(shape, geom, center=center)
    assert mask.dtype == torch.bool
    assert int(mask.sum()) > 0
    nz, ny, nx = shape
    z = torch.arange(nz).view(nz, 1, 1).expand(shape)
    y = torch.arange(ny).view(1, ny, 1).expand(shape)
    x = torch.arange(nx).view(1, 1, nx).expand(shape)
    r = torch.sqrt((y - 20.0) ** 2 + (z - 20.0) ** 2)
    assert bool((r[mask] <= geom.hub_radius + 0.25 + 1e-9).all())
    assert bool((torch.abs(x[mask] - 40.0) <= 0.5 * geom.hub_length + 1e-9).all())


# --------------------------------------------------------------------------- #
# IBM machinery
# --------------------------------------------------------------------------- #


def test_stencil_partition_of_unity_for_propeller_markers():
    geom = PropellerGeometry(diameter=16.0, radial_stations=5)
    shape = (24, 24, 40)
    pts = propeller_markers(geom, 0.0, center=(20.0, 12.0, 12.0))
    st = build_stencil(pts, shape, "4pt")
    assert st.peak_normalization() == pytest.approx(1.0, abs=1e-12)


def test_direct_forcing_is_zero_for_quiescent_matching_velocity():
    from tensorlbm.ibm_force_transfer import direct_forcing

    geom = PropellerGeometry(diameter=16.0, radial_stations=4)
    shape = (24, 24, 40)
    center = (20.0, 12.0, 12.0)
    pts = propeller_markers(geom, 0.0, center=center)
    st = build_stencil(pts, shape, "4pt")
    u = torch.zeros((3,) + shape, dtype=torch.float64)
    u[0] = 0.03  # uniform stream
    target = torch.zeros_like(pts)
    target[:, 0] = 0.03
    f_m, f_grid, body_force, torque = direct_forcing(st, u, target, n_correctors=0)
    assert float(f_m.abs().max()) < 1e-12
    assert float(f_grid.abs().max()) < 1e-12
    assert float(body_force.abs().max()) < 1e-12
    assert float(torque.norm()) < 1e-12


def test_momentum_balance_is_exact_to_roundoff():
    res = momentum_balance_audit(shape=(14, 14, 24), diameter=6.0, n_steps=3, tau=0.8)
    assert res["max_abs_residual"] < 1e-8
    assert res["rel_residual"] < 1e-8


def test_eta_error_monotone_helper():
    assert eta_error_is_monotone([0.3, 0.2, 0.1])
    assert eta_error_is_monotone([0.1, 0.1, 0.05])
    assert not eta_error_is_monotone([0.1, 0.2, 0.05])
    assert not eta_error_is_monotone([0.1])


# --------------------------------------------------------------------------- #
# verification smoke tests (short integrations)
# --------------------------------------------------------------------------- #


def test_open_water_point_sign_conventions_and_finiteness():
    geom = PropellerGeometry(diameter=10.0, radial_stations=5)
    cfg = OpenWaterConfig(
        advance_ratios=(0.5,),
        tau=0.52,
        steps_per_rev=60,
        warmup_revs=0.1,
        measure_revs=0.2,
        max_steps=24,
        lateral_factor=1.2,
        upstream_factor=1.0,
        downstream_factor=1.8,
    )
    res = run_open_water_point(geom, cfg, 0.5)
    assert math.isfinite(res["KT"])
    assert math.isfinite(res["KQ"])
    assert res["KT"] > 0.0, "propulsion convention: thrust must be positive"
    assert res["KQ"] > 0.0, "propulsion convention: torque must be positive"
    assert res["numerics"]["diverged"] is False
    # the IBM momentum source is the reaction force to roundoff
    assert res["numerics"]["momentum_source_rel_residual"] < 1e-8


def test_cylinder_drag_smoke_re40():
    cfg = CylinderConfig(
        diameter=6.0, reynolds=(40,), nx_per_d=16, ny_per_d=10, n_steps=1200, transient_frac=0.5
    )
    res = cylinder_drag_2d(40, cfg)
    assert math.isfinite(res["Cd"])
    # loose band: this is a smoke test for the sign/plumbing, the quantitative
    # cylinder gate lives in artifacts/p233w1_n07_ibm_openwater/cylinder.json
    assert 1.0 < res["Cd"] < 3.0, res["Cd"]
    assert res["Cd_reference"] == pytest.approx(CYLINDER_CD_2D[40])


def test_cylinder_config_tau_gives_requested_reynolds():
    cfg = CylinderConfig(diameter=10.0, inflow=0.05)
    for re in (20, 100, 200):
        nu = (cfg.tau(re) - 0.5) / 3.0
        assert cfg.inflow * cfg.diameter / nu == pytest.approx(re, rel=1e-9)


def test_open_water_efficiency_is_consistent_with_power_balance():
    """eta = J KT/(2 pi KQ) must equal the power ratio T U /(Omega Q)."""

    j, kt, kq = 0.6, 0.33, 0.051
    eta = open_water_efficiency(j, kt, kq)
    # with rho = 1, D = 1, n = 1 -> U = J, T = KT, Q = KQ, Omega = 2 pi
    power_useful = kt * j
    power_shaft = 2.0 * math.pi * kq
    assert eta == pytest.approx(power_useful / power_shaft, rel=1e-12)