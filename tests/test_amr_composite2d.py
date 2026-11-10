"""Tests for the composite-grid 2-level AMR solver (P40-N02).

The tests cover four layers:

* lattice kernels (moments, streaming, bounce-back, momentum exchange),
* the coarse--fine transfer operators (restriction, ghost reconstruction),
* the interface flux ledger and reflux correction (mass / momentum budget),
* end-to-end consistency (full-refinement equivalence, Galilean invariance,
  wall-bounded mass conservation, verification-case plumbing).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.amr_composite2d import (
    CX,
    CY,
    CompositeAMR2D,
    PatchSpec,
    collide,
    equilibrium,
    macroscopic,
    solid_force,
    stream_periodic,
    stream_walls,
)
from tensorlbm.amr_flux_verification import (
    GHIA_U,
    GHIA_V,
    CavityConfig,
    CavityConfig as _CC,
    cavity_case,
    reflux_conservation_case,
    taylor_green_case,
    taylor_green_initial,
)


def _rest(n: int) -> np.ndarray:
    return equilibrium(np.ones((n, n)), np.zeros((n, n)), np.zeros((n, n)))


# ---------------------------------------------------------------------------
# lattice kernels
# ---------------------------------------------------------------------------


def test_equilibrium_reproduces_moments() -> None:
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.01 * rng.standard_normal((12, 12))
    ux = 0.01 * rng.standard_normal((12, 12))
    uy = 0.01 * rng.standard_normal((12, 12))
    f = equilibrium(rho, ux, uy)
    r, a, b = macroscopic(f)
    assert np.allclose(r, rho, atol=1e-14)
    assert np.allclose(a, ux, atol=1e-13)
    assert np.allclose(b, uy, atol=1e-13)


def test_periodic_streaming_conserves_mass_and_momentum() -> None:
    rng = np.random.default_rng(1)
    f = 0.1 + 0.05 * rng.random((9, 16, 16))
    fs = stream_periodic(f)
    assert fs.sum() == pytest.approx(f.sum(), rel=1e-14)
    assert np.allclose(np.tensordot(CX, fs, axes=(0, 0)).sum(),
                       np.tensordot(CX, f, axes=(0, 0)).sum(), atol=1e-12)


def test_wall_bounce_back_preserves_rest_state() -> None:
    n = 12
    solid = np.zeros((n, n), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    f = _rest(n)
    for _ in range(20):
        f, _ = stream_walls(collide(f, 0.7), ~solid)
    rho, ux, uy = macroscopic(f)
    assert np.abs(ux).max() < 1e-14
    assert np.abs(uy).max() < 1e-14
    assert np.allclose(rho[~solid], 1.0, atol=1e-13)


def test_moving_lid_drives_flow_in_lid_direction() -> None:
    n = 14
    solid = np.zeros((n, n), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    wall_ux = np.zeros((n, n))
    wall_ux[-1, :] = 0.1
    wall_uy = np.zeros((n, n))
    f = _rest(n)
    for _ in range(300):
        f, _ = stream_walls(collide(f, 0.6), ~solid, wall_ux, wall_uy)
    rho, ux, _ = macroscopic(f)
    assert ux[-2, n // 2] > 0.01  # dragged in +x by the lid
    assert rho[~solid].sum() == pytest.approx((n - 2) ** 2, rel=1e-12)


def test_solid_force_drag_is_positive() -> None:
    """A uniform stream hitting a solid block must produce a +x drag force."""
    n = 20
    fluid = np.ones((n, n), dtype=bool)
    fluid[9:11, 9:11] = False
    f = equilibrium(np.ones((n, n)), 0.05 * np.ones((n, n)), np.zeros((n, n)))
    force = solid_force(f, fluid)
    assert force[0] > 0.0
    assert abs(force[1]) < 1e-10


# ---------------------------------------------------------------------------
# composite geometry and transfer operators
# ---------------------------------------------------------------------------


def test_covered_and_active_masks_are_consistent() -> None:
    n = 24
    amr = CompositeAMR2D(_rest(n), 0.6, [PatchSpec(j0=6, i0=8, nj=8, ni=10)])
    assert amr.covered.sum() == 8 * 10
    assert amr.covered[6, 8] and amr.covered[13, 17]
    assert not amr.covered[5, 8]
    assert amr.active_c.sum() == n * n - 80
    assert np.all(amr.active_c == (~amr.covered))


def test_ghost_interpolation_reproduces_linear_field() -> None:
    """Bilinear ghost reconstruction is exact for a field linear in x and y."""
    n = 24
    yy, xx = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float),
                         indexing="ij")
    rho = 1.0 + 0.001 * xx + 0.002 * yy
    f = equilibrium(rho, np.zeros((n, n)), np.zeros((n, n)))
    spec = PatchSpec(j0=6, i0=6, nj=10, ni=10)
    amr = CompositeAMR2D(f, 0.6, [spec])
    p = amr.patches[0]
    rho_c, _, _ = macroscopic(amr.fc)
    interp = amr._interp_coarse(rho_c, p)
    nfy, nfx = p.f.shape[1], p.f.shape[2]
    a = np.arange(nfy)[:, None] * np.ones((1, nfx))
    b = np.ones((nfy, 1)) * np.arange(nfx)[None, :]
    ypos = spec.j0 + (a - 0.5) / 2.0
    xpos = spec.i0 + (b - 0.5) / 2.0
    # rho[i, j] = 1 + 0.001 j + 0.002 i; bilinear interpolation in the coarse
    # index coordinate (cell centre at index + 0.5) is exact for that field
    exact = 1.0 + 0.001 * xpos + 0.002 * ypos
    assert np.abs(interp - exact).max() < 1e-12


def test_restriction_preserves_block_mass() -> None:
    n = 24
    rng = np.random.default_rng(3)
    rho = 1.0 + 0.01 * rng.standard_normal((n, n))
    f = equilibrium(rho, np.full((n, n), 0.01), np.zeros((n, n)))
    spec = PatchSpec(j0=6, i0=6, nj=10, ni=10)
    amr = CompositeAMR2D(f, 0.62, [spec])
    p = amr.patches[0]
    avg = 0.5 * (p.f + p.f)
    restricted = amr._restrict_patch(p, avg)
    m_fine = macroscopic(p.f)[0][1:-1, 1:-1].sum() / 4.0
    m_coarse = macroscopic(restricted)[0].sum()
    assert m_coarse == pytest.approx(m_fine, rel=1e-13)


def test_composite_field_layout() -> None:
    n = 20
    amr = CompositeAMR2D(_rest(n), 0.6, [PatchSpec(j0=4, i0=4, nj=8, ni=8)])
    field = amr.composite_field()
    assert field["ux"].shape == (2 * n, 2 * n)
    assert np.allclose(field["rho"], 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# conservation / reflux
# ---------------------------------------------------------------------------


def test_uniform_rest_preserved_on_composite_grid() -> None:
    n = 24
    amr = CompositeAMR2D(_rest(n), 0.6, [PatchSpec(j0=8, i0=8, nj=8, ni=8)],
                         periodic=True)
    for _ in range(30):
        amr.step()
    field = amr.composite_field()
    assert np.abs(field["ux"]).max() < 1e-13
    assert np.allclose(field["rho"], 1.0, atol=1e-12)


def test_galilean_invariance_of_the_interface() -> None:
    """A uniform flow must cross the coarse--fine interface unchanged."""
    n = 24
    u0 = 0.02
    f = equilibrium(np.ones((n, n)), u0 * np.ones((n, n)), np.zeros((n, n)))
    amr = CompositeAMR2D(f, 0.62, [PatchSpec(j0=8, i0=8, nj=8, ni=8)],
                         periodic=True)
    for _ in range(40):
        amr.step()
    field = amr.composite_field()
    assert np.abs(field["ux"] - u0).max() < 1e-11
    assert np.abs(field["uy"]).max() < 1e-11


def test_mass_conserved_with_reflux() -> None:
    n = 32
    f0 = taylor_green_initial(n, 0.05)
    amr = CompositeAMR2D(f0, 0.58, [PatchSpec(j0=8, i0=8, nj=16, ni=16)],
                         periodic=True, reflux=True, reflux_momentum=False,
                         reflux_spread="global")
    m0 = amr.composite_mass()
    for _ in range(40):
        amr.step()
    assert abs(amr.composite_mass() - m0) / m0 < 1e-12


def test_mass_drift_without_reflux_is_larger() -> None:
    n = 32
    f0 = taylor_green_initial(n, 0.05)
    drifts = {}
    for reflux in (True, False):
        amr = CompositeAMR2D(f0, 0.58, [PatchSpec(j0=8, i0=8, nj=16, ni=16)],
                             periodic=True, reflux=reflux,
                             reflux_momentum=False, reflux_spread="global")
        m0 = amr.composite_mass()
        for _ in range(40):
            amr.step()
        drifts[reflux] = abs(amr.composite_mass() - m0) / m0
    assert drifts[True] < 1e-12
    assert drifts[False] > 1e3 * drifts[True]


def test_momentum_conserved_with_momentum_reflux() -> None:
    n = 24
    f0 = taylor_green_initial(n, 0.05)
    amr = CompositeAMR2D(f0, 0.6, [PatchSpec(j0=6, i0=6, nj=12, ni=12)],
                         periodic=True, reflux=True, reflux_momentum=True,
                         reflux_spread="interface")
    p0 = amr.composite_momentum()
    for _ in range(20):
        amr.step()
    assert np.linalg.norm(amr.composite_momentum() - p0) / max(n * n, 1) < 1e-12


def test_interface_net_flux_residual() -> None:
    n = 32
    f0 = taylor_green_initial(n, 0.05)
    amr = CompositeAMR2D(f0, 0.58, [PatchSpec(j0=8, i0=8, nj=16, ni=16)],
                         periodic=True, reflux=True, reflux_momentum=False,
                         reflux_spread="global", diag_budget=True)
    worst = 0.0
    for _ in range(30):
        d = amr.step()
        gross = abs(d["flux_in_total"]) + abs(d["flux_out_total"]) + 1e-300
        worst = max(worst, abs(d["mass_change"]) / gross)
    assert worst <= 1e-8


def test_mass_conserved_wall_bounded_case() -> None:
    n = 24
    solid = np.zeros((n, n), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    wall_ux = np.zeros((n, n))
    wall_ux[-1, :] = 0.05
    wall_uy = np.zeros((n, n))
    f = _rest(n)
    amr = CompositeAMR2D(f, 0.62, [PatchSpec(j0=1, i0=1, nj=6, ni=n - 2)],
                         solid_coarse=solid, wall_ux_coarse=wall_ux,
                         wall_uy_coarse=wall_uy, periodic=False, reflux=True,
                         reflux_momentum=False, reflux_spread="global")
    m0 = amr.composite_mass()
    for _ in range(30):
        amr.step()
    # the moving-wall bounce-back injects a small mass amount at the lid corners
    # (no link partner there); the bulk conservation is otherwise exact.
    assert abs(amr.composite_mass() - m0) / m0 < 1e-4


# ---------------------------------------------------------------------------
# end-to-end consistency
# ---------------------------------------------------------------------------


def test_fine_patch_with_walls_is_fluid_only() -> None:
    n = 20
    solid = np.zeros((n, n), dtype=bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    amr = CompositeAMR2D(_rest(n), 0.6, [PatchSpec(j0=1, i0=1, nj=6, ni=n - 2)],
                         solid_coarse=solid, periodic=False)
    p = amr.patches[0]
    # the ghost row *below* the patch sits on the solid bottom wall
    assert not p.fluid[0, 1:-1].any()
    assert p.fluid[-1, 1:-1].all()  # the top ghost row is inside the fluid
    assert p.interior_mask.sum() == (2 * 6) * (2 * (n - 2))


def test_composite_energy_normalisation_on_full_patch() -> None:
    """A patch covering everything must report the fine-grid energy exactly."""
    n = 16
    f0 = taylor_green_initial(n, 0.05)
    amr = CompositeAMR2D(f0, 0.6, [PatchSpec(j0=0, i0=0, nj=n, ni=n)],
                         periodic=False)
    p = amr.patches[0]
    _, ux, uy = macroscopic(p.f)
    e_fine = 0.5 * float(np.mean((ux * ux + uy * uy)[p.interior_mask]))
    assert amr.composite_energy() == pytest.approx(e_fine, rel=1e-12)


def test_taylor_green_case_is_accurate_at_coarse_resolution() -> None:
    res = taylor_green_case(n=32, re=100.0, u0=0.05, steps=80, record=10,
                            patch=(8, 8, 16, 16))
    assert res["l2_rel_vs_uniform"] < 0.02
    assert res["pass"] is True
    assert len(res["energy_amr"]) == len(res["energy_uniform_fine"])


def test_reflux_case_reports_criteria() -> None:
    res = reflux_conservation_case(n=24, re=100.0, u0=0.02, steps=40,
                                   patch=(8, 8, 8, 8))
    assert res["pass_mass"] is True
    assert res["pass_interface_flux"] is True
    assert res["control_needs_reflux"] is True


def test_cavity_amr_smoke() -> None:
    cfg = CavityConfig(fluid=16, strip=4, max_steps=120, tol=0.0, check_every=60)
    res_amr = cavity_case(cfg, kind="amr", record_every=60)
    res_uni = cavity_case(cfg, kind="uniform", record_every=60)
    ua = np.asarray(res_amr["profiles"]["u"])
    uu = np.asarray(res_uni["profiles"]["u"])
    assert ua.shape == uu.shape
    assert res_amr["mass_drift_rel"] < 5e-3  # lid-corner moving-wall leakage
    # both solutions must be driven in +x by the lid
    assert ua[-1] > ua[0]


def test_ghia_reference_table_is_sane() -> None:
    for table in (GHIA_U, GHIA_V):
        y = [p[0] for p in table]
        assert y == sorted(y, reverse=True)
        assert abs(table[-1][1]) < 1e-12  # no-slip wall value
    assert GHIA_U[0][1] == 1.0  # lid value
    assert min(v for _, v in GHIA_U) == pytest.approx(-0.38289, abs=1e-12)
    assert max(v for _, v in GHIA_V) == pytest.approx(0.37095, abs=1e-12)


def test_declared_tau_pair_keeps_physical_viscosity() -> None:
    n = 20
    amr = CompositeAMR2D(_rest(n), 0.58, [PatchSpec(j0=5, i0=5, nj=10, ni=10)])
    assert amr.tau_f == pytest.approx(2 * 0.58 - 0.5)
    assert amr.nu_f == pytest.approx(2 * amr.nu_c, rel=1e-14)
    assert math.isclose(amr.scale_c2f, amr.tau_f / (2 * amr.tau_c))