"""Unit gates for the two-conservation Zou-He reconstruction with a chemical potential.

Predeclared criteria (see ``docs/XVAL-LBM-TWO-CONSERVATION-ZOUHE-WITH-CHEMICAL-POT.md``):

* C1 ``corrected`` inlet conserves both quantities exactly: ``sum f = rho``,
  ``sum c f + F/2 = rho u`` and ``sum g = psi``, ``sum c_x g = psi u_x`` -> error < 1e-12.
* C2 ``classic`` (half-force dropped) violates the reconstruction by ``O(F/2)`` -> the
  measured velocity error must match the analytic prediction ``|F|/(2 rho)`` within 5 %.
* C3 ``impose`` (upstream deficit split) conserves ``psi`` exactly but not the advective
  ``psi``-flux -> flux residual strictly worse than ``corrected``.
* C4 outlet: ``rho`` prescribed to machine precision and zero transverse velocity.
"""

from __future__ import annotations

import torch

from tensorlbm.phasefield.zouhe_two_conservation import (
    column_invariants,
    reconstruct_inlet,
    reconstruct_outlet,
)


def _random_column(ny: int, seed: int, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    f = 0.05 + 0.1 * torch.rand((9, ny), generator=gen, dtype=dtype)
    g = -0.2 + 0.4 * torch.rand((9, ny), generator=gen, dtype=dtype)
    fx = torch.linspace(-0.05, 0.05, ny, dtype=dtype)
    fy = torch.linspace(0.02, -0.02, ny, dtype=dtype)
    u = torch.linspace(0.01, 0.06, ny, dtype=dtype)
    return f, g, fx, fy, u


def test_c1_corrected_inlet_conserves_mass_momentum_and_order_parameter():
    for ny in (16, 64):
        f, g, fx, fy, u = _random_column(ny, seed=ny)
        _, _, st = reconstruct_inlet(f, g, u, (fx, fy))
        assert float((st.mass_flux - st.rho * u).abs().max()) < 1e-12
        assert float(st.uy.abs().max()) < 1e-12
        assert float((st.psi_flux - st.psi * u).abs().max()) < 1e-12
        # both conserved densities are positive and finite
        assert bool(torch.isfinite(st.rho).all()) and bool((st.rho > 0).all())


def test_c2_classic_reconstruction_error_matches_half_force_law():
    for scale in (0.01, 0.05, 0.2):
        f, g, fx, fy, u = _random_column(32, seed=7)
        fx = fx * scale / 0.05
        _, _, st = reconstruct_inlet(f, g, u, (fx, fy), force_correction=False)
        measured = float(column_invariants(
            reconstruct_inlet(f, g, u, (fx, fy), force_correction=False)[0], g, (fx, fy)
        ).ux.__sub__(u).abs().max())
        predicted = float((fx.abs() / (2.0 * st.rho)).max())
        assert measured > 0.0
        assert abs(measured - predicted) / predicted < 0.05, (measured, predicted)


def test_c3_impose_mode_conserves_psi_but_not_its_flux():
    f, g, fx, fy, u = _random_column(32, seed=11)
    psi_target = torch.linspace(0.6, -0.6, 32, dtype=torch.float64)
    _, _, st_imp = reconstruct_inlet(
        f, g, u, (fx, fy), psi_mode="impose", psi_target=psi_target
    )
    _, _, st_flx = reconstruct_inlet(f, g, u, (fx, fy), psi_mode="flux")
    assert float((st_imp.psi - psi_target).abs().max()) < 1e-12
    err_imp = float((st_imp.psi_flux - st_imp.psi * u).abs().max())
    err_flx = float((st_flx.psi_flux - st_flx.psi * u).abs().max())
    assert err_flx < 1e-12 < err_imp


def test_c4_outlet_prescribes_density_and_zero_transverse_velocity():
    f, g, fx, fy, u = _random_column(32, seed=13)
    f_out, g_out, st = reconstruct_outlet(f, g, 1.0, (fx, fy))
    assert float((st.rho - 1.0).abs().max()) < 1e-12
    assert float(st.uy.abs().max()) < 1e-12
    assert bool(torch.isfinite(st.psi).all())