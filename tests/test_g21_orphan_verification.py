"""Regression pins for the G21 orphan-module literature verification.

Companion to ``tools/g21_orphan_verification.py`` /
``docs/G21_ORPHAN_VERIFICATION.md``.  These tests assert *invariants against the
literature*, not merely that code runs: every threshold below is a number
computed from the paper formula, not from the implementation.

Covered defects (see the report):
  * D1  ``ddes`` header formula contradicted the implementation (fixed in docs)
  * D2  ``ddes`` branch diagnostics mislabelled the DES/LES fractions
  * D3  ``ddes`` default constants make the DES limiter unreachable (RANS only)
  * D4  ``prism_solver`` two-branch wall law was discontinuous at y+ = 5
  * D5  ``hybrid_solver`` returned silent zero placeholders
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.ddes import (
    DDESConfig,
    ddes_branch_fractions,
    ddes_length_fields,
    ddes_length_scale,
    ddes_shielding_function,
    run_ddes_diagnostics,
)
from tensorlbm.prism_solver import PrismCorrector, build_prism, uplus_reichardt

DT = torch.float64
KAPPA = 0.41
C_DES = 0.65
C_S = 0.1


def _fd(r_d: float) -> float:
    """f_d for a target r_d = (nu + nu_t) / (|S| kappa^2 d^2) with nu=0, S=d=1."""
    return float(
        ddes_shielding_function(
            0.0,
            torch.tensor(r_d * KAPPA**2, dtype=DT),
            torch.ones((), dtype=DT),
            torch.ones((), dtype=DT),
            KAPPA,
        )
    )


def test_shielding_matches_spalart_2006():
    """D: f_d = 1 - tanh[(8 r_d)^3] (C_d1 = 8), to machine precision."""
    for r_d in (1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0):
        assert abs(_fd(r_d) - (1.0 - math.tanh((8.0 * r_d) ** 3))) < 1e-14


def test_shielding_half_switch_is_analytic():
    """f_d = 0.5 at r_d* = atanh(0.5)^(1/3) / 8."""
    r_d_star = math.atanh(0.5) ** (1.0 / 3.0) / 8.0
    assert abs(_fd(r_d_star) - 0.5) < 1e-12
    assert _fd(r_d_star * 0.99) > 0.5 > _fd(r_d_star * 1.01)  # f_d decreases in r_d


def test_length_scale_is_spalart_eq5():
    """l_DDES = l_RANS - f_d max(0, l_RANS - C_DES dx) exactly."""
    cases = [(1.0, 0.65, 0.0), (1.0, 0.65, 0.5), (1.0, 0.65, 1.0),
             (0.30, 0.65, 1.0), (0.30, 0.65, 0.25), (2.0, 0.65, 0.75)]
    for l_rans, l_les, f_d in cases:
        got = float(
            ddes_length_scale(
                torch.ones((), dtype=DT), torch.tensor(f_d, dtype=DT), l_les / C_DES,
                torch.tensor(l_rans, dtype=DT), C_DES,
            )
        )
        ref = l_rans - f_d * max(0.0, l_rans - l_les)
        assert abs(got - ref) < 1e-12, (l_rans, l_les, f_d, got, ref)


def test_length_scale_bounds_and_des_limit():
    """Bounded by [min(l_RANS,l_LES), max(...)], non-increasing in f_d, DES limit."""
    for l_rans in (0.05, 0.3, 0.65, 1.0, 3.0):
        prev = None
        for f_d in (0.0, 0.25, 0.5, 0.75, 1.0):
            got = float(
                ddes_length_scale(
                    torch.ones((), dtype=DT), torch.tensor(f_d, dtype=DT), 1.0,
                    torch.tensor(l_rans, dtype=DT), C_DES,
                )
            )
            assert min(l_rans, C_DES) - 1e-12 <= got <= max(l_rans, C_DES) + 1e-12
            if prev is not None:
                assert got <= prev + 1e-12
            prev = got
        # f_d = 0 -> pure RANS ; f_d = 1 -> DES limiter min(l_RANS, C_DES dx)
        assert abs(prev - min(l_rans, C_DES)) < 1e-12


def test_old_header_formula_would_be_wrong():
    """D1: the max(l_RANS, C_DES dx / f_d) form is NOT the implementation."""
    l_rans, l_les, f_d = 1.0, C_DES, 0.25
    old = max(l_rans, l_les / f_d)          # what the header used to claim
    got = float(
        ddes_length_scale(
            torch.ones((), dtype=DT), torch.tensor(f_d, dtype=DT), 1.0,
            torch.tensor(l_rans, dtype=DT), C_DES,
        )
    )
    assert abs(old - got) > 1.0  # the two forms differ materially


def _bl_field(ny=64, nx=8):
    y = torch.arange(ny, dtype=DT).view(ny, 1).expand(ny, nx).contiguous()
    d_wall = y + 0.5
    ux = (0.2 * torch.log1p(d_wall)).contiguous()
    return ux, torch.zeros_like(ux), d_wall


def test_branch_fractions_are_actually_measured():
    """D2: reported RANS/LES fractions must follow l_DDES, not f_d alone."""
    ux, uy, d_wall = _bl_field()
    cfg = DDESConfig()
    res = run_ddes_diagnostics(ux, uy, d_wall, cfg)
    l_rans, l_les, l_ddes, f_d = ddes_length_fields(ux, uy, d_wall, cfg)
    rans, les, blend = ddes_branch_fractions(l_rans, l_les, l_ddes)
    assert res.rans_fraction == pytest.approx(rans, abs=1e-12)
    assert res.les_fraction == pytest.approx(les, abs=1e-12)
    assert res.blend_fraction == pytest.approx(blend, abs=1e-12)
    # the legacy formula (1 - mean(f_d < 0.5)) overstates LES activity here
    legacy = float((f_d >= 0.5).float().mean())
    assert legacy - res.les_fraction > 0.9


def test_default_constants_disable_the_des_limiter():
    """D3: with C_S < C_DES the LES branch is unreachable (RANS-only model)."""
    d_wall = torch.linspace(0.5, 200.0, 200, dtype=DT).view(-1, 1).expand(-1, 4).contiguous()
    z = torch.zeros_like(d_wall)
    l_rans, l_les, l_ddes, _ = ddes_length_fields(z, z, d_wall, DDESConfig())
    assert float(l_rans.max()) < float(l_les.min())
    assert bool((l_ddes == l_rans).all())
    assert run_ddes_diagnostics(z, z, d_wall, DDESConfig()).les_limiter_active is False

    # control arm needs a sheared field (ux=0 floors the strain -> f_d=0 -> RANS)
    ux_s = (0.2 * torch.log1p(d_wall)).contiguous()
    l_rans2, l_les2, l_ddes2, _ = ddes_length_fields(ux_s, z, d_wall, DDESConfig(c_des=0.05))
    assert float((l_les2 < l_rans2).float().mean()) > 0.0
    assert bool((l_ddes2 == l_les2).any())


def test_reichardt_law_is_continuous_and_asymptotically_correct():
    """D4: composite law replaces the discontinuous two-branch switch."""
    # continuity: no jump between adjacent samples over 5 decades of y+
    yps = [10 ** (-2 + 5 * i / 4000) for i in range(4001)]
    u = [uplus_reichardt(y) for y in yps]
    assert max(abs(b - a) for a, b in zip(u, u[1:])) < 0.05
    # viscous limit u+ ~ y+
    assert abs(uplus_reichardt(1.0) - 1.0) < 0.02
    assert abs(uplus_reichardt(2.0) - 2.0) / 2.0 < 0.02
    # log-layer slope 1/kappa
    slope = (uplus_reichardt(1e3) - uplus_reichardt(1e2)) / math.log(10.0)
    assert abs(slope - 1.0 / KAPPA) / (1.0 / KAPPA) < 0.02
    assert uplus_reichardt(0.0) == 0.0
    # the defect being fixed: the legacy piecewise law jumps by >3.9 in u+
    legacy_jump = (math.log(5.0) / KAPPA + 5.0) - 5.0
    assert legacy_jump > 3.9


def test_prism_injection_uses_the_wall_law():
    """prism_solver injects u = u_tau * u+(y+) and zeroes the wall-normal part."""
    nz, ny, nx = 6, 16, 8
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    solid[:, 0, :] = True
    prism = build_prism(solid, n_layers=5, first_height=0.02, growth=1.15)
    nu, u_tau_val = 0.01, 0.02
    corr = PrismCorrector(prism, nu, torch.device("cpu"))
    u_tau = torch.full((nz, ny, nx), u_tau_val)
    z = torch.zeros(nz, ny, nx)
    ux_c, uy_c, uz_c = corr.correct_velocity(z.clone(), z.clone(), z.clone(), u_tau)

    h = prism["heights"]
    y_c = [float(h[0]) / 2]
    for k in range(1, len(h)):
        y_c.append(y_c[-1] + float(h[k - 1] + h[k]) / 2.0)
    y_iface = y_c[-1] + float(h[-1]) / 2.0
    expected = u_tau_val * uplus_reichardt(y_iface * u_tau_val / nu)
    surf = prism["surface_xyz"]
    got = float(ux_c[int(surf[0, 2]), int(surf[0, 1]), int(surf[0, 0])])
    assert abs(got - expected) / expected < 1e-5
    assert float(uy_c.abs().max()) == 0.0 and float(uz_c.abs().max()) == 0.0


def test_hybrid_placeholders_are_explicit():
    """D5: silent zero placeholders must raise instead of returning 0.0."""
    from tensorlbm.hybrid_solver import FDMBoundaryStep, HybridSolver, PrismLayerData

    n_layers, n_surf = 4, 3
    prism = PrismLayerData(
        n_surface=n_surf, n_layers=n_layers,
        layer_heights=torch.tensor([0.02 * 1.15**k for k in range(n_layers)], dtype=DT),
        surface_centers=torch.zeros(n_surf, 2, dtype=DT),
        surface_normals=torch.zeros(n_surf, 2, dtype=DT),
        layer_mask=torch.zeros(2, 4, 4, dtype=torch.bool), band_indices=(),
    )
    fdm = FDMBoundaryStep(prism, nu=2.4e-6)
    with pytest.raises(NotImplementedError):
        fdm.body_force_from_lbm(torch.zeros(2, 4, 4), torch.zeros(2, 4, 4, dtype=torch.bool))
    with pytest.raises(NotImplementedError):
        HybridSolver.drag_coefficient(object.__new__(HybridSolver), 1.0)


def test_hybrid_fdm_preserves_a_linear_profile():
    """The FDM scaffold's explicit step must leave a Couette profile unchanged."""
    from tensorlbm.hybrid_solver import FDMBoundaryStep, PrismLayerData

    n_layers, n_surf = 5, 4
    heights = torch.tensor([0.02 * 1.15**k for k in range(n_layers)], dtype=DT)
    y = torch.cumsum(torch.cat([heights[:1] / 2, (heights[:-1] + heights[1:]) / 2]), 0)
    prism = PrismLayerData(
        n_surface=n_surf, n_layers=n_layers, layer_heights=heights,
        surface_centers=torch.zeros(n_surf, 2, dtype=DT),
        surface_normals=torch.zeros(n_surf, 2, dtype=DT),
        layer_mask=torch.zeros(2, 4, 4, dtype=torch.bool), band_indices=(),
    )
    fdm = FDMBoundaryStep(prism, nu=2.4e-6, dt=0.5)
    u = (12.0 * y).view(n_layers, 1, 1).expand(n_layers, n_surf, 2).clone()
    out = fdm.step(u, torch.zeros(n_surf, 2, dtype=DT))
    rel = float((out[1:-1] - u[1:-1]).abs().max()) / float(u.abs().max())
    assert rel < 1e-3  # uniform-h_avg approximation on a graded mesh


def test_d3q27_config_arithmetic():
    """The helper is arithmetic only: Re and tau must be exactly consistent."""
    from tensorlbm.d3q27_collide import suboff_config_d3q27

    cfg = suboff_config_d3q27(nx=200, re=2e6)
    assert abs(cfg["u_in"] * cfg["hull_length"] / cfg["nu_lat"] - cfg["re"]) / cfg["re"] < 1e-12
    assert abs(cfg["tau"] - (0.5 + 3.0 * cfg["nu_lat"])) < 1e-12
    assert cfg["ny"] == int(0.4 * 200) and cfg["nz"] == int(0.4 * 200)
    assert cfg["lattice"] == "D3Q27" and cfg["smagorinsky_cs"] == 0.0