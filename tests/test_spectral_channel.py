"""Verification anchors for the spectral/FD channel engine.

Two kinds of tests:

* *analytic anchors* -- the forced channel must relax onto the exact Poiseuille
  parabola and the wall shear must reproduce the momentum-balance value;
* *frozen defects* -- the collocated pressure projection is known not to be the
  exact inverse of the discrete div(grad) operator, so a strongly non-solenoidal
  field leaves a residual divergence and a turbulent trajectory diverges.  Those
  behaviours are asserted here on purpose: the tests turn red when the staggered
  wall-normal velocity replaces the collocated one, which forces a re-review of
  every downstream conclusion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.turbulence.spectral_channel import (
    SpectralChannel,
    SpectralChannelConfig,
    log_law,
    poiseuille_exact,
    reichardt,
)


def _laminar_cfg(**kw):
    base = dict(nx=16, ny=33, nz=16, lx=2 * math.pi, lz=math.pi, h=1.0,
                re_tau=10.0, u_tau=1.0, dt=5.0e-3, init_profile="parabola",
                noise=0.0, stat_every=10**9)
    base.update(kw)
    return SpectralChannelConfig(**base)


def test_forced_channel_relaxes_onto_poiseuille():
    """Start OFF the solution (log-law profile) and relax onto the parabola."""
    cfg = _laminar_cfg(init_profile="log")
    solver = SpectralChannel(cfg)
    exact = poiseuille_exact(solver.y, cfg.height, cfg.accel, cfg.nu)
    solver.run(4000)          # 10 viscous times: relaxation, not just a fixed point
    rel = np.abs(solver.mean_profile() - exact).max() / exact.max()
    assert rel < 3.0e-2, rel   # ledger records 7.0e-3 for exactly this run


def test_wall_shear_matches_momentum_balance():
    cfg = _laminar_cfg()
    solver = SpectralChannel(cfg)
    solver.run(1500)
    tau = solver.wall_shear()
    expected = cfg.accel * cfg.h
    assert abs(tau - expected) / expected < 0.05, (tau, expected)


def test_discrete_divergence_is_round_off_for_smooth_field():
    solver = SpectralChannel(_laminar_cfg())
    solver.run(50)
    assert solver.divergence() < 1e-8


def test_grid_metrics_are_wall_unit_consistent():
    cfg = _laminar_cfg(ny=65, re_tau=180.0)
    m = cfg.grid_metrics()
    assert m["dy_min_plus"] < m["dy_max_plus"]
    assert m["dx_plus"] == pytest.approx(cfg.lx / cfg.nx * cfg.u_tau / cfg.nu)
    assert m["dz_plus"] == pytest.approx(cfg.lz / cfg.nz * cfg.u_tau / cfg.nu)


def test_config_rejects_inconsistent_settings():
    with pytest.raises(ValueError):
        SpectralChannelConfig(nx=15).validate()
    with pytest.raises(ValueError):
        SpectralChannelConfig(viscous_scheme="rk3").validate()
    with pytest.raises(ValueError):
        SpectralChannelConfig(dt=0.0).validate()


def test_analytic_reference_laws_behave():
    yp = np.array([1.0, 10.0, 30.0, 100.0])
    assert np.allclose(log_law(yp, 0.41, 5.2), np.log(yp) / 0.41 + 5.2)
    # Reichardt reproduces the viscous sublayer and the log layer
    assert abs(float(reichardt(1.0)) - 1.0) < 0.05
    assert abs(float(reichardt(100.0)) - float(log_law(100.0))) < 0.5


def test_profiles_and_spectra_reductions_are_consistent():
    cfg = _laminar_cfg(stat_every=10, stat_start=0)
    solver = SpectralChannel(cfg)
    solver.run(60)
    prof = solver.profiles()
    assert prof["samples"] >= 3
    assert len(prof["y_plus"]) == cfg.ny // 2 + 1
    assert prof["bulk_plus"] > 0.0
    spec = solver.spectra()
    assert len(spec["k"]) == len(spec["E_u"]) == len(spec["E_v"])
    assert max(spec["E_u"]) > 0.0


def test_FROZEN_DEFECT_collocated_projection_leaves_divergence():
    """KNOWN DEFECT: the projection is not the inverse of the discrete div(grad).

    A random non-solenoidal field must be (nearly) cleaned by an exact
    projection.  Here the residual *grows*, which is why a turbulent trajectory
    blows up.  Frozen on purpose -- see the module docstring of
    ``_solve_pressure``.
    """
    cfg = _laminar_cfg(nx=16, ny=33, nz=16)
    solver = SpectralChannel(cfg)
    rng = np.random.default_rng(7)
    solver.u = rng.standard_normal(solver.u.shape) * 0.1
    solver.u[:, 0] = 0.0
    solver.u[:, -1] = 0.0
    F = solver._fwd(solver.u)
    div0 = solver.ikex[None] * F[0] + solver._dy(F[1]) + solver.ikez[None] * F[2]
    p = solver._solve_pressure(div0 / cfg.dt)
    F2 = F.copy()
    F2[0] -= cfg.dt * solver.ikex[None] * p
    F2[1] -= cfg.dt * solver._dy(p)
    F2[2] -= cfg.dt * solver.ikez[None] * p
    div1 = solver.ikex[None] * F2[0] + solver._dy(F2[1]) + solver.ikez[None] * F2[2]
    assert np.abs(div1).max() > np.abs(div0).max(), (
        "the collocated projection suddenly removed the divergence -- the "
        "staggered wall-normal velocity landed, re-review the campaign report"
    )


def test_FROZEN_DEFECT_turbulent_trajectory_diverges():
    """KNOWN DEFECT: a turbulent trajectory of this engine does not survive."""
    cfg = _laminar_cfg(nx=16, ny=33, nz=16, noise=0.2, init_profile="log",
                       re_tau=180.0, dt=2.0e-4)
    solver = SpectralChannel(cfg)
    blew = False
    for _ in range(400):
        solver.step()
        if not np.isfinite(solver.u).all():
            blew = True
            break
    assert blew, (
        "the turbulent trajectory survived -- the projection defect is fixed, "
        "re-run the DNS campaign and re-grade the gates"
    )
