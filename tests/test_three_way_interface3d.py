"""Invariant tests for the 3-D load-exchange interface (GAP-LBM-T5-ICE-COUPLING, round 1).

These are fast (~10 s) unit-level assertions of the invariants stated in
``src/tensorlbm/three_way/interface3d.py``; the quantitative convergence study
and the cross-checks against analytic references live in
``scripts/three_way_interface3d_gates.py`` (artefact:
``runs/three_way_interface3d/gates.json``).

Run::

    OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests/test_three_way_interface3d.py -q
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

_SRC = str(pathlib.Path(__file__).resolve().parents[1] / "src")
while _SRC in sys.path:
    sys.path.remove(_SRC)
sys.path.insert(0, _SRC)
for _m in [m for m in list(sys.modules) if m == "tensorlbm" or m.startswith("tensorlbm.")]:
    del sys.modules[_m]

from tensorlbm.beam_fem import BeamProperties, EulerBernoulliBeamFEM  # noqa: E402
from tensorlbm.three_way.interface3d import (  # noqa: E402
    CouplingClock,
    HexGrid3D,
    InterfaceLedger,
    added_mass_ratio,
    explicit_staggered_stable,
    march_one_dof,
    project_to_beam,
    rectangle_surface,
    relaxation_bound,
    sphere_surface,
    staggered_spectral_radius,
)

DT = torch.float64


def _grid(n: int = 6) -> HexGrid3D:
    return HexGrid3D((-1.5, -1.5, -1.5), (3.0, 3.0, 3.0), (n, n, n))


# ---------------------------------------------------------------------------
# I1 — closed-surface quadrature
# ---------------------------------------------------------------------------


def test_closed_sphere_surface_sum_is_zero():
    surf = sphere_surface((0.0, 0.0, 0.0), 1.0, n_theta=10, n_phi=20)
    s = (surf.normals * surf.dA.view(-1, 1)).sum(0)
    assert float(s.abs().max()) < 1e-14


def test_hemisphere_pressure_resultant_tends_to_p_pi_a2():
    errs = []
    for n_th, n_ph in ((8, 16), (24, 48)):
        hemi = sphere_surface((0, 0, 0), 1.0, n_theta=n_th, n_phi=n_ph, hemisphere=True)
        fz = -float(hemi.pressure_force(1.0)[2])
        errs.append(abs(fz - math.pi) / math.pi)
    assert errs[-1] < 5e-3
    assert errs[-1] < errs[0]


def test_capped_hemisphere_is_closed():
    surf = sphere_surface((0, 0, 0), 1.0, n_theta=10, n_phi=20, hemisphere=True, cap=True)
    assert float((surf.normals * surf.dA.view(-1, 1)).sum(0).abs().max()) < 1e-14


# ---------------------------------------------------------------------------
# I2/I3/I4 — load transfer invariants
# ---------------------------------------------------------------------------


def test_consistent_surface_load_conserves_resultant_and_moment():
    grid = _grid(8)
    rect = rectangle_surface((-0.4, -0.3, 0.2), (0.8, 0, 0), (0, 0.6, 0), 4, 4)
    t = torch.tensor([0.5, -1.0, 2.0], dtype=DT).view(1, 3).repeat(len(rect), 1)
    nodal = grid.distribute_surface(rect, t)
    assert torch.allclose(nodal.sum(0), torch.tensor([0.5, -1.0, 2.0], dtype=DT) * 0.48,
                          rtol=0, atol=1e-13)
    # uniform traction on a centroid-symmetric patch -> zero moment about the patch centre
    centre = rect.centroid()
    m = torch.cross(grid.nodes - centre.view(1, 3), nodal, dim=1).sum(0)
    assert float(m.abs().max()) < 1e-13


def test_discrete_work_identity_is_exact():
    """Σ_i f_i·u(x_i) = Σ_m t_m·u(x_m) dA_m for affine velocity (I3, exact identity)."""
    grid = _grid(6)
    rect = rectangle_surface((-0.5, -0.5, 0.25), (1, 0, 0), (0, 1, 0), 5, 5)
    t0 = torch.tensor([0.7, -1.3, 2.1], dtype=DT)
    B = torch.tensor([[0.3, -0.2, 0.1], [0.05, 0.4, -0.15], [-0.25, 0.2, 0.6]], dtype=DT)
    u0 = torch.tensor([0.11, -0.07, 0.23], dtype=DT)
    omega = torch.tensor([0.4, -0.9, 0.6], dtype=DT)
    x0 = torch.tensor([0.13, -0.21, 0.37], dtype=DT)

    def traction(p):
        return t0.view(1, 3) + p @ B.T

    def velocity(p):
        return u0.view(1, 3) + torch.cross(omega.view(1, 3), p - x0.view(1, 3), dim=1)

    trac = traction(rect.positions)
    nodal = grid.distribute_surface(rect, trac)
    w_nodal = float((nodal * velocity(grid.nodes)).sum())
    w_ref = float(((trac * velocity(rect.positions)).sum(1) * rect.dA).sum())
    assert abs(w_nodal - w_ref) / abs(w_ref) < 1e-12


def test_consistent_point_load_beats_nearest_node():
    grid = _grid(6)
    p = torch.tensor([0.13, -0.21, 0.37], dtype=DT)
    f = torch.tensor([1.0, 2.0, -3.0], dtype=DT)
    ref_m = torch.cross(p.view(1, 3), f.view(1, 3), dim=1)[0]
    cons = grid.distribute_point_consistent(p, f)
    lump = grid.distribute_point_lumped(p, f)
    e_cons = float(torch.linalg.norm(torch.cross(grid.nodes, cons, dim=1).sum(0) - ref_m))
    e_lump = float(torch.linalg.norm(torch.cross(grid.nodes, lump, dim=1).sum(0) - ref_m))
    assert e_cons < 1e-14
    assert e_lump > 1e-2


def test_gather_is_transpose_of_distribute():
    """Consistency: (f, N u) == (Nᵀ f, u) for the same nodal velocity field."""
    grid = _grid(6)
    p = torch.tensor([0.21, 0.07, -0.33], dtype=DT)
    f = torch.tensor([0.3, -0.4, 0.9], dtype=DT)
    u_nodes = torch.linspace(-1, 1, grid.nodes.numel(), dtype=DT).view(-1, 3)
    lhs = float((grid.distribute_point_consistent(p, f) * u_nodes).sum())
    rhs = float((grid.gather_point_velocity(p, u_nodes) * f).sum())
    assert abs(lhs - rhs) / max(abs(lhs), 1e-30) < 1e-14


# ---------------------------------------------------------------------------
# clock + added-mass guard
# ---------------------------------------------------------------------------


def test_clock_rejects_inconsistent_subcycling():
    good = CouplingClock(dt_fluid=1e-3, dt_struct=4e-3, n_sub=4, lag=1)
    good.check()
    bad = CouplingClock(dt_fluid=1e-3, dt_struct=1e-2, n_sub=4, lag=0)
    with pytest.raises(ValueError):
        bad.check()


def test_added_mass_criterion_and_threshold():
    assert explicit_staggered_stable(0.99)
    assert not explicit_staggered_stable(1.01)
    # the discrete spectral radius crosses 1 at mu = 1
    assert staggered_spectral_radius(0.9) < 1.0
    assert staggered_spectral_radius(1.1) > 1.0
    # measured growth of the time march matches the closed-form radius
    # (log-slope over a long window averages the undamped oscillator's oscillation)
    for mu, tol in ((0.5, 1e-2), (2.0, 1e-3)):
        hist = march_one_dof(mu, omega_n_dt=1e-2, n_steps=400)
        sub = hist[:, 200:]
        scale = sub.abs().amax(dim=0)
        y = torch.log(scale) + torch.log(torch.linalg.norm(sub / scale, dim=0))
        n = torch.arange(200, 401, dtype=DT)
        nm = n - n.mean()
        growth = float(torch.exp((nm * (y - y.mean())).sum() / (nm * nm).sum()))
        lam = staggered_spectral_radius(mu)
        assert abs(growth - lam) / lam < tol


def test_relaxation_bound_is_the_iteration_boundary():
    mu = 2.0
    bound = relaxation_bound(mu)
    assert abs(bound - 2.0 / 3.0) < 1e-15
    y = 1.0
    for _ in range(80):
        y = y + 0.9 * bound * (-mu * y - y)
    assert abs(y) < 1.0
    y = 1.0
    for _ in range(80):
        y = y + 1.1 * bound * (-mu * y - y)
    assert abs(y) > 1.0


def test_added_mass_ratio_sphere():
    mu = added_mass_ratio(m_s=1.0, rho_f=1000.0, displaced_volume=4.0 / 3.0 * math.pi * 0.05 ** 3)
    assert abs(mu - 1000.0 * 4.0 / 3.0 * math.pi * 0.05 ** 3) < 1e-15


# ---------------------------------------------------------------------------
# ledger + beam leg
# ---------------------------------------------------------------------------


def test_interface_ledger_closes_on_a_spring_mass():
    m, k, dt, n = 1.0, 4.0, 5e-3, 400
    x, v = 1.0, 0.0
    ledger = InterfaceLedger()
    ke0 = 0.5 * m * v * v
    f = -k * x
    for _ in range(n):
        a = f / m
        x_new = x + dt * v + 0.5 * dt * dt * a
        f_new = -k * x_new
        a_new = f_new / m
        v_new = v + 0.5 * dt * (a + a_new)
        ledger.add(torch.tensor([f], dtype=DT), torch.tensor([x], dtype=DT))
        ledger.add(torch.tensor([f_new], dtype=DT), torch.tensor([x_new], dtype=DT))
        x, v, f = x_new, v_new, f_new
    dke = 0.5 * m * v * v - ke0
    assert abs(ledger.work - dke) / 2.0 < 1e-4


def test_beam_projection_conserves_total_load_and_moment():
    L, n_el = 1.0, 8
    xs = torch.linspace(0.0, L, 101, dtype=DT)[1:-1]
    q = -12.0
    loads = torch.full_like(xs, q * (L / 100.0))  # point loads (99 interior stations)
    dof = project_to_beam(xs, loads, L, n_el)
    total = float(dof[0::2].sum())
    assert abs(total - float(loads.sum())) / abs(float(loads.sum())) < 1e-12
    # first moment of the load: Hermite basis is complete for cubic fields, so the
    # nodal work against a linear velocity field reproduces ∫ q·u dx
    def u(x):
        return 0.3 + 0.7 * x
    w_nodes = float((dof[0::2] * u(torch.linspace(0, L, n_el + 1, dtype=DT))).sum()
                    + 0.0)
    w_ref = float((loads * u(xs)).sum())
    assert abs(w_nodes - w_ref) / abs(w_ref) < 2e-2


def test_beam_static_tip_deflection_matches_euler_bernoulli():
    props = BeamProperties(youngs_modulus=2.1e11, density=7850.0, width=0.1, height=0.01,
                           length=1.0, boundary_condition="cantilever")
    beam = EulerBernoulliBeamFEM(props, n_elements=10)
    F = 100.0
    w = torch.linalg.solve(beam.K_global, beam.apply_point_load(props.length, F))
    ref = F * props.length ** 3 / (3.0 * props.flexural_rigidity)
    assert abs(float(w[-2]) - ref) / ref < 1e-9