"""Unit and verification tests for :mod:`tensorlbm.free_surface`.

Design notes
------------
* The *fixtures* are deliberately domain-free: the comparison agnosticism is
  checked through the synthetic series ``metric_alpha`` (observed) and
  ``metric_beta`` (reference), so no hull / case vocabulary leaks into the
  metric contract.
* Negative controls are explicit: a dry domain must report a zero front, a lake
  at rest must not develop spurious currents, and switching the global volume
  correction *on* must not be a prerequisite for mass conservation (the
  projection alone has to conserve it).
* Everything here is cheap (<= 32 x 32 grids, <= 120 steps) so the suite stays
  inside the CPU test budget; the production-resolution runs live in
  ``benchmarks/free_surface_dambreak_wigley``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.free_surface import (
    DamBreakMoyce,
    FreeSurfaceVOF2D,
    VOFConfig,
    VariableCoeffPoisson2D,
    _superbee,
    advect_interface_convergence,
    clip_redistribute,
    fit_convergence_order,
    martin_moyce_table,
    relative_rmse_percent,
    rmse,
    vof_face_values_1d,
    wigley_halfbreadth,
)


# ---------------------------------------------------------------------------
# domain-free metric fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def metric_alpha() -> np.ndarray:
    """Observed series of a generic metric."""
    return np.array([1.00, 1.10, 1.30, 1.60])


@pytest.fixture
def metric_beta() -> np.ndarray:
    """Reference series of the same generic metric."""
    return np.array([1.00, 1.12, 1.28, 1.62])


@pytest.fixture
def small_cfg() -> VOFConfig:
    """Cheap configuration: 32 x 16 cells, short domain, no surface tension."""
    return VOFConfig(nx=32, ny=16, lx=2.0, ly=1.0, cfl=0.25,
                     max_poisson_cycles=12, poisson_tol=1.0e-6)


# ---------------------------------------------------------------------------
# 1-2: literature anchor bookkeeping
# ---------------------------------------------------------------------------

def test_reference_table_shape_and_monotonicity():
    tab = martin_moyce_table()
    assert tab["T"].size == 14 and tab["X"].size == 14
    assert np.all(np.diff(tab["T"]) > 0.0)
    assert np.all(np.diff(tab["X"]) > 0.0)
    assert tab["T"][0] == 0.0 and tab["X"][0] == 1.0
    assert tab["X"][-1] == pytest.approx(3.67, abs=1e-12)


def test_reference_axis_convention():
    """``T = t sqrt(2 g / a)`` with ``g = a = 1``: the last point is t = 2.093."""
    tab = martin_moyce_table()
    t_last = tab["T"][-1] / np.sqrt(2.0)
    assert t_last == pytest.approx(2.0929, abs=1e-3)
    # the retired axis (T = t sqrt(g/a)) would give 2.96; guard against it
    assert abs(t_last - 2.96) > 0.5


# ---------------------------------------------------------------------------
# 3: configuration guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"nx": 4, "ny": 8},
    {"rho_g": 2.0, "rho_l": 1.0},
    {"rho_g": 0.0},
    {"wall": "slip"},
    {"cfl": 0.9},
    {"nu_l": -1.0},
])
def test_config_rejects_bad_input(kwargs):
    base = {"nx": 16, "ny": 16, "lx": 1.0, "ly": 1.0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        VOFConfig(**base)


# ---------------------------------------------------------------------------
# 4-6: domain-free metric contract
# ---------------------------------------------------------------------------

def test_rmse_matches_direct_mean_square(metric_alpha, metric_beta):
    expected = np.sqrt(np.mean((metric_alpha - metric_beta) ** 2))
    assert rmse(metric_alpha, metric_beta) == pytest.approx(expected, rel=1e-12)


def test_relative_rmse_is_scale_invariant(metric_alpha, metric_beta):
    a = relative_rmse_percent(metric_alpha, metric_beta)
    b = relative_rmse_percent(10.0 * metric_alpha, 10.0 * metric_beta)
    assert a == pytest.approx(b, rel=1e-12)


def test_metric_helpers_guard_shapes(metric_alpha):
    with pytest.raises(ValueError):
        rmse(metric_alpha, metric_alpha[:-1])
    with pytest.raises(ValueError):
        relative_rmse_percent(metric_alpha, np.zeros_like(metric_alpha))


# ---------------------------------------------------------------------------
# 7-8: multigrid Poisson
# ---------------------------------------------------------------------------

def _mg_operator(nx: int, ny: int, dx: float = 1.0 / 16.0):
    bxp = np.ones((ny, nx + 1))
    byp = np.ones((ny + 1, nx))
    bxp[:, 0] = 0.0
    bxp[:, -1] = 0.0
    byp[0, :] = 0.0          # floor Neumann, top Dirichlet (p = 0 outside)
    return VariableCoeffPoisson2D(bxp, byp, dx, dx)


def test_multigrid_recovers_manufactured_field():
    nx, ny, dx = 24, 32, 1.0 / 16.0
    mg = _mg_operator(nx, ny, dx)
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(x, y)
    p_ex = np.cos(np.pi * X / (nx * dx)) * np.sin(np.pi * (ny * dx - Y) / (2 * ny * dx))
    f = mg.apply(mg.levels[0], p_ex)
    p = np.zeros((ny, nx))
    info = mg.solve(p, f, tol=1.0e-10, max_cycles=40)
    assert info["relative"] < 1.0e-6
    assert np.abs(p - p_ex).max() / np.abs(p_ex).max() < 1.0e-5


def test_multigrid_rejects_mismatched_face_shapes():
    with pytest.raises(ValueError):
        VariableCoeffPoisson2D(np.ones((8, 9)), np.ones((9, 7)), 0.1, 0.1)


# ---------------------------------------------------------------------------
# 9-12: projection, rest state and mass bookkeeping
# ---------------------------------------------------------------------------

def test_projection_is_divergence_free(small_cfg):
    sim = FreeSurfaceVOF2D(small_cfg)
    sim.set_liquid([(0.0, 0.5, 0.0, 0.5)])
    for _ in range(5):
        sim.step()
    div = (sim.u[:, 1:] - sim.u[:, :-1]) / sim.dx + (sim.v[1:, :] - sim.v[:-1, :]) / sim.dy
    scale = np.abs(sim.u).max() / sim.dx + np.abs(sim.v).max() / sim.dy
    assert np.abs(div).max() <= 1.0e-3 * max(scale, 1.0e-12)


def test_lake_at_rest_stays_at_rest():
    """Negative control: no spurious currents in a hydrostatic basin."""
    cfg = VOFConfig(nx=24, ny=16, lx=1.5, ly=1.0, cfl=0.2, max_poisson_cycles=12,
                    poisson_tol=1.0e-6)
    sim = FreeSurfaceVOF2D(cfg)
    sim.set_liquid([(0.0, 1.5, 0.0, 0.5)])
    sim.run(0.05)
    assert np.abs(sim.u).max() < 5.0e-3
    assert np.abs(sim.v).max() < 5.0e-3
    assert sim.ledger.max_drift() < 1.0e-3


def test_mass_ledger_conserves_volume_without_global_fix(small_cfg):
    sim = FreeSurfaceVOF2D(small_cfg)
    sim.set_liquid([(0.0, 0.5, 0.0, 0.6)])
    assert small_cfg.mass_correction is False
    sim.run(0.15)
    assert sim.n_steps >= 5
    assert sim.ledger.max_drift() < 1.0e-4


def test_mass_correction_toggle_is_reported(small_cfg):
    """The global volume fix is a *logged* repair, never a silent one."""
    cfg = VOFConfig(nx=32, ny=16, lx=2.0, ly=1.0, cfl=0.25, mass_correction=True,
                    max_poisson_cycles=12, poisson_tol=1.0e-6)
    sim = FreeSurfaceVOF2D(cfg)
    sim.set_liquid([(0.0, 0.5, 0.0, 0.6)])
    sim.run(0.15)
    info = sim.ledger.as_dict()
    assert "repair_clip" in info and "repair_global" in info
    assert sim.ledger.max_drift() < 1.0e-6


# ---------------------------------------------------------------------------
# 13-15: interface reconstruction primitives
# ---------------------------------------------------------------------------

def test_vof_reconstruction_is_bounded():
    rng = np.random.default_rng(7)
    f = rng.random((8, 12))
    uf = rng.normal(size=(8, 13))
    face = vof_face_values_1d(f, uf, axis=1)
    assert face.min() >= 0.0 and face.max() <= 1.0
    vf = rng.normal(size=(9, 12))
    face = vof_face_values_1d(f, vf, axis=0)
    assert face.min() >= 0.0 and face.max() <= 1.0


def test_superbee_is_tvd(metric_alpha, metric_beta):
    dl = np.array([0.0, 0.2, -0.3, 1.0, 0.5])
    dr = np.array([0.0, 0.4, -0.1, 0.2, -0.5])
    s = _superbee(dl, dr)
    for i in range(dl.size):
        if dl[i] * dr[i] <= 0.0:
            assert s[i] == 0.0
        else:
            assert abs(s[i]) <= 2.0 * min(abs(dl[i]), abs(dr[i])) + 1e-12
    # the reconstruction must not overshoot the local monotone range either
    assert np.all(np.abs(s) <= 2.0)


def test_clip_redistribute_conserves_volume():
    """Overshoot/undershoot must be repaired *without* losing volume."""
    f = np.full((8, 8), 0.3)     # room to redistribute into
    f[3, 3] = 1.2                # overshoot the transport can produce
    f[4, 5] = -0.2               # undershoot
    before = f.sum()
    repaired, leftover = clip_redistribute(f)
    assert repaired.min() >= 0.0 and repaired.max() <= 1.0
    assert leftover < 0.05
    assert repaired.sum() == pytest.approx(before, abs=1e-8)


def test_clip_redistribute_reports_unrepairable_excess():
    """A saturated neighbourhood cannot absorb the excess: it is reported."""
    f = np.ones((6, 6))          # fully saturated: no capacity anywhere
    f[2, 2] = 1.5
    repaired, leftover = clip_redistribute(f)
    assert repaired.max() <= 1.0
    assert leftover >= 0.4


# ---------------------------------------------------------------------------
# 16: order of accuracy of the interface transport
# ---------------------------------------------------------------------------

def test_advection_convergence_order_positive():
    res = advect_interface_convergence(sizes=(24, 48), t_end=0.5, n_rk=200)
    assert res["order"] > 0.8
    assert res["r2"] > 0.9
    assert all(e > 0.0 for e in res["l2"])


def test_convergence_fit_recovers_known_order():
    h = np.array([1 / 20, 1 / 40, 1 / 80])
    err = 3.0 * h ** 2
    fit = fit_convergence_order(h, err)
    assert fit["order"] == pytest.approx(2.0, abs=1e-8)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# 17: capillary anchor
# ---------------------------------------------------------------------------

def test_surface_tension_is_off_by_default():
    """The dam-break verification runs without the CSF term (Bond -> infinity).

    The CSF model itself is exercised by :func:`laplace_pressure`, which is a
    *diagnostic* and not part of the verified set: at the resolutions tested the
    static-droplet pressure jump is polluted by spurious capillary currents (see
    ``docs/free_surface_dambreak_vof_report.md``, honest-FAIL note).  What is
    asserted here is only the default-off contract, so no run silently carries
    an unverified force term.
    """
    assert VOFConfig().sigma == 0.0
    sim = FreeSurfaceVOF2D(VOFConfig(nx=16, ny=16, lx=1.0, ly=1.0))
    sim.set_droplet(0.2)
    for _ in range(3):
        sim.step()
    assert np.isfinite(sim.u).all() and np.isfinite(sim.p).all()


# ---------------------------------------------------------------------------
# 18-20: front tracking, driver and hull geometry
# ---------------------------------------------------------------------------

def test_front_position_tracks_initial_column():
    db = DamBreakMoyce(cells_per_a=16)
    assert db.sim.front_position() == pytest.approx(1.0, abs=1.5 * db.cfg.dx)
    db.sim.run(0.3)          # T = 0.42 in the experimental axis
    assert db.sim.front_position() > 1.0


def test_front_position_dry_domain_is_zero():
    """Negative control: no liquid -> no front."""
    sim = FreeSurfaceVOF2D(VOFConfig(nx=16, ny=16, lx=1.0, ly=1.0))
    assert sim.front_position() == 0.0


def test_dam_break_driver_reports_every_checkpoint():
    db = DamBreakMoyce(cells_per_a=8)
    res = db.run(t_end_mm=0.84)      # only the first two checkpoints are reached
    assert len(res["X_sim"]) == 14
    # coarse smoke-test grid: the production-resolution mass gate is measured
    # in benchmarks/free_surface_dambreak_wigley (criterion |dM|/M <= 2 %)
    assert res["mass"]["max_relative_drift"] < 5.0e-3
    assert res["rmse"] > 0.0


def test_wigley_halfbreadth_analytic_geometry():
    x = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    y = wigley_halfbreadth(x, beam=0.1)
    assert y[2] == pytest.approx(0.05, rel=1e-12)
    assert y[0] == pytest.approx(0.0, abs=1e-15)
    assert y[4] == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(y, y[::-1])
    # waterplane area of the parabolic form = (2/3) * B * L
    xs = np.linspace(-0.5, 0.5, 20001)
    area = np.trapezoid(2.0 * wigley_halfbreadth(xs, beam=0.1), xs)
    assert area == pytest.approx(2.0 / 3.0 * 0.1, rel=2e-6)