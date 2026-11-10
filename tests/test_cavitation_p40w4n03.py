"""P40W4-N03: cavitation verification-face component tests.

The campaign result (see ``docs/P40W4-N03_cavitation_verification.md``) is an
honest FAIL on the physics criteria, so these tests deliberately verify the
*instruments* rather than the physics:

* the mixed Dirichlet/Neumann pressure solve (free-boundary substrate),
* the equilibrium vapour switch,
* mass conservation of the equilibrium solver,
* the nuclei-inflow boundary condition (defect D1 of the kinetic solver),
* the cavity-metric extractor and the Strouhal estimator on synthetic fields
  with a known answer.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.cavitation_airfoil import (
    CavitationFlowConfig,
    MixtureCavitationSolver,
    airfoil_surface,
    cavity_metrics,
    dominant_frequency,
    reference_data,
)
from tensorlbm.cavitation_equilibrium import (
    EquilibriumCavityConfig,
    EquilibriumCavitySolver,
    _lap_neumann,
    equilibrium_alpha_target,
    mixed_poisson,
)

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# 1. mixed Dirichlet/Neumann Poisson (the free-boundary substrate)
# ---------------------------------------------------------------------------


def test_mixed_poisson_solves_the_masked_operator():
    torch.manual_seed(0)
    nx, ny, h = 40, 24, 1.0 / 12.0
    rhs = torch.randn(nx, ny, dtype=torch.float64)
    fixed = torch.zeros(nx, ny, dtype=torch.bool)
    fixed[:8, 6:16] = True  # a "cavity" block pinned at p_v
    p_v = -0.5
    phi, info = mixed_poisson(rhs, fixed, p_v, h, max_iter=2000, tol=1e-13)
    free = ~fixed
    assert info["mode"] == "mixed"
    assert info["iters"] > 1
    residual = (torch.where(free, _lap_neumann(phi, h), phi)
                - torch.where(free, rhs, torch.full_like(rhs, p_v))).abs()
    assert float(residual[free].max()) < 1e-10
    # the Dirichlet rows are honoured exactly
    assert float((phi[fixed] - p_v).abs().max()) < 1e-14


def test_mixed_poisson_falls_back_to_neumann_without_pinned_cells():
    nx, ny, h = 16, 12, 1.0 / 8.0
    rhs = torch.zeros(nx, ny, dtype=torch.float64)
    rhs[8, 6] = 1.0
    rhs -= rhs.mean()  # Neumann compatibility
    phi, info = mixed_poisson(rhs, torch.zeros(nx, ny, dtype=torch.bool), -0.5, h)
    assert info["mode"] == "neumann"
    # the Neumann solve reproduces the (compatibility-filtered) right-hand side
    lap = _lap_neumann(phi, h)
    assert float((lap - rhs).abs().max()) < 1e-9


def test_mixed_poisson_keeps_a_flat_dirichlet_patch_flat():
    """A cavity pinned at p_v stays at p_v while the free field solves around it."""
    nx, ny, h = 32, 20, 1.0 / 10.0
    rhs = torch.zeros(nx, ny, dtype=torch.float64)
    fixed = torch.zeros(nx, ny, dtype=torch.bool)
    fixed[6:14, 10:14] = True
    phi, _ = mixed_poisson(rhs, fixed, -0.3, h, max_iter=500, tol=1e-12)
    assert float(phi[fixed].max()) == pytest.approx(-0.3, abs=1e-12)
    assert float(phi[fixed].min()) == pytest.approx(-0.3, abs=1e-12)
    # outside the patch the field is smooth (no checkerboard): the discrete
    # Laplacian of a harmonic field is zero away from the patch
    lap = _lap_neumann(phi, h)
    far = torch.zeros_like(fixed)
    far[18:, 2:8] = True
    assert float(lap[far].abs().max()) < 1e-9


# ---------------------------------------------------------------------------
# 2. equilibrium switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p,expected", [(-0.60, 1.0), (-0.50, 0.5), (-0.40, 0.0)])
def test_equilibrium_alpha_target_saturates(p, expected):
    out = equilibrium_alpha_target(torch.tensor([p], dtype=torch.float64), -0.5, 0.02)
    assert float(out[0]) == pytest.approx(expected, abs=1e-12)


def test_equilibrium_alpha_target_is_monotone_in_pressure():
    p = torch.linspace(-0.8, -0.2, 200, dtype=torch.float64)
    a = equilibrium_alpha_target(p, -0.5, 0.02)
    assert bool((a.diff() <= 0).all())  # lower pressure -> more vapour


# ---------------------------------------------------------------------------
# 3. equilibrium solver: conservation and cold start
# ---------------------------------------------------------------------------


def _eq_solver(**kw):
    cfg = EquilibriumCavityConfig(
        cells_per_chord=12,
        domain_x_chords=6.0,
        domain_y_chords=3.0,
        le_x_chords=1.5,
        le_y_chords=1.5,
        n_steps=kw.pop("n_steps", 40),
        **kw,
    )
    return EquilibriumCavitySolver(cfg)


def test_equilibrium_solver_conserves_liquid_mass():
    s = _eq_solver(sigma=1.0)
    for _ in range(40):
        s.step()
    # divergence-free liquid and inlet/outlet flux balance; the campaign
    # acceptance level is 1e-4, the measured value at cpc=12 is O(1e-5)
    assert s.div_residual_max < 1.0e-4
    assert s.mass_balance_max < 1.0e-4
    assert torch.isfinite(s.u).all() and torch.isfinite(s.p).all()


def test_equilibrium_solver_cold_start_stays_liquid():
    s = _eq_solver(sigma=1.0, n_steps=30)
    for _ in range(30):
        s.step()
    assert int((s.alpha > 0.5).sum()) == 0
    assert s.cavity_metrics()["l_cav"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. kinetic solver: the nuclei-inflow boundary condition (defect D1)
# ---------------------------------------------------------------------------


def _kinetic(**kw):
    cfg = CavitationFlowConfig(
        cells_per_chord=12,
        domain_x_chords=6.0,
        domain_y_chords=3.0,
        le_x_chords=1.5,
        le_y_chords=1.5,
        cfl=0.2,
        **kw,
    )
    return MixtureCavitationSolver(cfg)


def test_nuclei_bc_restores_a_wiped_inflow_seed():
    """The nuclei boundary condition is a Dirichlet floor on the inflow plane.

    Without it the condensation branch (active wherever ``p > p_v``, i.e. almost
    everywhere) eats the seed population and no cavity can nucleate -- that is
    defect D1, the root cause recorded for P41-N02.
    """
    s = _kinetic(sigma=1.0, n_steps=1, nuclei_inflow=True)
    wiped = torch.zeros_like(s.alpha)
    s.alpha = wiped.clone()
    s._apply_nuclei_bc()
    cols = s.cfg.nuclei_inlet_columns
    liquid = ~s.solid[:cols, :]
    assert float(s.alpha[:cols, :][liquid].min()) >= s.cfg.alpha_nuclei
    # columns downstream of the boundary condition are untouched
    assert float(s.alpha[cols + 2:, :].abs().max()) == 0.0


def test_nuclei_bc_is_off_by_default():
    s = _kinetic(sigma=1.0, n_steps=1)
    assert s.cfg.nuclei_inflow is False
    s.alpha = torch.zeros_like(s.alpha)
    for _ in range(30):
        s.step()
    assert float(s.alpha.abs().max()) < 1e-6  # nothing is replenished


# ---------------------------------------------------------------------------
# 5. cavity metrics and Strouhal estimator against synthetic ground truth
# ---------------------------------------------------------------------------


def test_cavity_metrics_recovers_a_known_attached_cavity():
    cfg = CavitationFlowConfig(
        airfoil="NACA0015", sigma=1.0, cells_per_chord=100,
        domain_x_chords=4.0, domain_y_chords=2.0, le_x_chords=1.0, le_y_chords=1.0,
    )
    geo = airfoil_surface("NACA0015")
    h = cfg.h
    nx, ny = cfg.nx, cfg.ny
    xs = (torch.arange(nx, dtype=torch.float64) + 0.5) * h - cfg.le_x_chords
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * h - cfg.le_y_chords
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    chord = X.clamp(0.0, 1.0)
    yt = torch.zeros_like(chord)
    from tensorlbm.cavitation_airfoil import _interp1, naca4_half_thickness
    yt = _interp1(geo.xc.double(), geo.half_thickness.double(), chord)
    x0, x1, thick = 0.05, 0.30, 0.04
    alpha = torch.zeros(nx, ny, dtype=torch.float64)
    band = (X > x0) & (X < x1) & (Y > yt) & (Y < yt + thick)
    alpha[band] = 1.0
    m = cavity_metrics(alpha, cfg, threshold=0.5, geometry=geo)
    assert m["detachment_x"] == pytest.approx(x0, abs=0.02)
    assert m["l_cav"] == pytest.approx(x1 - x0, abs=0.02)
    assert m["delta_cav"] == pytest.approx(thick, abs=0.01)


@pytest.mark.parametrize("f0", [0.20, 0.16])
def test_dominant_frequency_recovers_a_known_strouhal(f0):
    dt, n = 0.02, 1200
    t = torch.arange(n, dtype=torch.float64) * dt
    cl = 0.4 + 0.1 * torch.sin(2.0 * math.pi * f0 * t)
    f, st, spec = dominant_frequency(t.tolist(), cl.tolist())
    assert f == pytest.approx(f0, rel=0.05)
    assert st is not None
    assert st[1] == pytest.approx(f0, rel=0.05)  # St = f c / U = f here
    assert len(spec["freq"]) > 10


# ---------------------------------------------------------------------------
# 6. reference table sanity (the acceptance denominators)
# ---------------------------------------------------------------------------


def test_reference_tables_are_declared_and_monotone():
    ref = reference_data()
    sheet = ref["sheet"]
    assert sheet["sigma"] == [1.0, 0.8, 0.6]
    lc = sheet["l_cav_exp"]
    assert lc[0] < lc[1] < lc[2]  # decreasing sigma -> longer cavity
    assert all(0.0 < b < 0.5 for b in (sheet["band_l_cav"], sheet["band_delta_cav"]))
    cloud = ref["cloud"]
    assert cloud["st_exp"][0] > cloud["st_exp"][1]
    assert len(cloud["sigma"]) == len(cloud["st_exp"])