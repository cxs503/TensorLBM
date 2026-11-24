"""Regression tests for the round-2 coexistence calibration / free-energy selection.

Anchor ``XVAL-LBM-R4-COEXISTENCE-CALIBRATION-AND-FREE-ENER``.  Round 1 shipped an
honest FAIL on both arms; these tests pin the two fixes so they cannot regress:

* ``flat_interface_coexistence`` — Maxwell-seeded, library-force flat interface
  reproduces the Maxwell binodal instead of diverging (the round-1 failure mode);
* ``relax_ch`` / ``measure_planar_interface`` — the free-energy interface is only
  declared measured when the Cahn-Hilliard relaxation has actually converged, and
  the two independent surface-tension routes agree.

Sizes are deliberately small (a few thousand steps on 64x2 / 128x2 fields) so the
whole file runs in ~30 s; the production sweep is
``development/verification/mcmp-coexistence/r4/r2/run_xval_r2.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


@pytest.fixture(autouse=True)
def _float64_precision():
    """These checks resolve ~1 % deviations, so run in float64 — but restore the
    session default dtype afterwards (a module-level switch would silently change
    the numerics of every other test in the same pytest session)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

from tensorlbm import free_energy_stress as fes                                   # noqa: E402
from tensorlbm.multiphase import psi_exp                                         # noqa: E402
from tensorlbm.sc_eos_coexistence import (                                       # noqa: E402
    flat_interface_coexistence,
    maxwell_coexistence,
)

GATE = 0.15


def test_maxwell_reference_G5():
    """Maxwell binodal of the library EOS (independent of any lattice solver)."""
    m = maxwell_coexistence(G=5.0, psi_fn=psi_exp, rho_lo=1e-5, rho_hi=50.0, n=12000)
    assert m.found
    assert 1.85 < m.rho_liquid < 2.0, m.rho_liquid
    assert 0.14 < m.rho_vapour < 0.17, m.rho_vapour
    assert abs(m.area_residual) < 1e-6
    assert m.p_sat is not None and 0.03 < m.p_sat < 0.04


def test_no_coexistence_at_spinodal_coupling():
    """G = 4.0 is the critical coupling: the calibration must refuse to seed."""
    with pytest.raises(ValueError):
        flat_interface_coexistence(G=4.0, psi_fn=psi_exp, ny=32, steps=10)


def test_flat_interface_reproduces_maxwell_binodal_edm():
    """Round-1 failure mode: the flat interface must relax to the binodal, not NaN."""
    m = maxwell_coexistence(G=5.0, psi_fn=psi_exp, rho_lo=1e-5, rho_hi=50.0, n=12000)
    r = flat_interface_coexistence(
        G=5.0, psi_fn=psi_exp, ny=64, nx=2, forcing="edm", steps=6000,
        average_over=3000, u_probe_every=3000,
    )
    assert not r.diverged
    assert r.rho_liquid is not None and r.rho_vapour is not None
    assert abs(r.rho_liquid - m.rho_liquid) / m.rho_liquid < GATE
    assert abs(r.rho_vapour - m.rho_vapour) / m.rho_vapour < GATE
    # a flat interface carries no Laplace pressure, and mass is conserved
    assert r.mass_drift is not None and r.mass_drift < 1e-6
    assert r.p_liquid is not None and r.p_vapour is not None
    assert abs(r.p_liquid - r.p_vapour) / m.p_sat < 0.05
    # the spurious-current envelope is recorded (evidence the state is static)
    assert r.spurious_u_history and r.spurious_u_profile
    assert r.spurious_u_history[-1]["u_max"] == pytest.approx(r.spurious_u_max, rel=1e-6)


def test_relax_ch_reports_non_convergence():
    """A truncated relaxation must say so instead of presenting a sigma."""
    p = fes.FreeEnergyParams(A=1.0, B=1.0, kappa=2.0)
    ny = 64
    y = torch.arange(ny).view(-1, 1).double()
    phi = (torch.tanh((y - ny / 4) / 6.0) * torch.tanh((3 * ny / 4 - y) / 6.0))
    phi = phi.expand(ny, 2).contiguous().clone()
    short = fes.relax_ch(phi, p, steps=400, check_every=100)
    assert short["converged"] is False
    assert short["drift"] == short["drift"]  # not NaN (a checkpoint was taken)


def test_relax_ch_converged_sigma_passes_and_routes_agree():
    """Converged CH interface: sigma within the gate, both routes consistent."""
    p = fes.FreeEnergyParams(A=1.0, B=1.0, kappa=2.0)
    ny = 96
    y = torch.arange(ny).view(-1, 1).double()
    phi = (torch.tanh((y - ny / 4) / 6.0) * torch.tanh((3 * ny / 4 - y) / 6.0))
    phi = phi.expand(ny, 2).contiguous().clone()
    rel = fes.relax_ch(phi, p, steps=40000, check_every=500)
    assert rel["converged"] is True
    assert rel["mu_std_over_A"] < 2e-3
    m = fes.measure_planar_interface(rel["phi"], p, n_interfaces=2)
    err_sigma = abs(m["sigma_pressure"] - m["sigma_analytic"]) / m["sigma_analytic"]
    assert err_sigma < GATE
    assert m["sigma_route_spread"] < 0.05
    assert m["xi_relative_err"] < GATE
    # the relaxed interface is stress-balanced at the level the stencil allows
    assert m["max_abs_divP"] < 0.05


def test_measure_planar_interface_recovers_analytic_width():
    """xi recovery on a two-interface continuum tanh profile (geometry-only check)."""
    p = fes.FreeEnergyParams(A=1.0, B=1.0, kappa=2.0)
    ny = 128
    xi = fes.interface_width(p)
    y = torch.arange(ny).view(-1, 1).double()
    phi = (torch.tanh((y - ny / 4) / xi) * torch.tanh((3 * ny / 4 - y) / xi))
    phi = phi.expand(ny, 2).contiguous()
    m = fes.measure_planar_interface(phi, p, n_interfaces=2)
    assert abs(m["xi_lsq"] - xi) / xi < 0.02
    assert abs(m["xi_10_90"] - xi) / xi < 0.05