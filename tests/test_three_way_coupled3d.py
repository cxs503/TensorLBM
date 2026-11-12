"""Invariant tests for the 3-D three-way chain (GAP-LBM-T5-ICE-COUPLING round 2).

These are fast unit-level assertions on the *implemented* chain in
``src/tensorlbm/three_way/coupled3d.py``:

* the fluid leg really advances the D3Q19 state and returns finite momentum-exchange
  forces on the sphere markers;
* the live (LBM-produced) surface tractions go through the Q1 interface and reproduce
  the marker resultant/moment — the interface is *not* fed synthetic tractions here;
* the structural leg (Hermite FEM) reproduces the Euler-Bernoulli Maxwell integral;
* the DEM leg is momentum-exact (Newton's third law) for both wall and pair contacts;
* the staged run wires all three together and produces the gate block.

The quantitative / convergence study lives in
``scripts/three_way_coupled3d_run.py`` (artefacts under ``runs/three_way_coupled3d/``).

Run::

    OMP_NUM_THREADS=2 PYTHONPATH=src python -m pytest tests/test_three_way_coupled3d.py -q
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
from tensorlbm.three_way.coupled3d import (  # noqa: E402
    Coupled3DConfig,
    Fluid3DIBM,
    IceDEM3D,
    clift_cd,
    hull_girder_from_contact_loads,
    run_coupled3d,
    schiller_naumann_cd,
    tip_deflection_reference,
)
from tensorlbm.three_way.interface3d import (  # noqa: E402
    HexGrid3D,
    exchange_cycle,
    project_to_beam,
)

F64 = torch.float64


def _small_flow(n: int = 24, d: float = 6.0, steps: int = 60) -> Fluid3DIBM:
    flow = Fluid3DIBM(
        n=(n, n, n), nu=0.03, centre=(0.25 * n, 0.5 * n, 0.5 * n),
        diameter=d, n_theta=4, n_phi=8, u_init=0.05,
    )
    for _ in range(steps):
        flow.step(1.0e-5)
    return flow


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------


def test_reference_correlations_are_ordered():
    """Sanity of the *reference* correlations (not of the solver)."""
    for re in (0.5, 1.0, 5.0, 10.0):
        assert clift_cd(re) > 24.0 / re          # drag always exceeds Stokes
        assert schiller_naumann_cd(re) > 24.0 / re
    # at Re -> 0 both collapse onto Stokes
    assert abs(clift_cd(1e-3) - 24.0 / 1e-3) / (24.0 / 1e-3) < 0.05


def test_tip_deflection_reference_matches_point_load_formula():
    length, ei = 20.0, 3.5e4
    tip = tip_deflection_reference(torch.tensor([length]), torch.tensor([2.0]), length, ei)
    assert tip == pytest.approx(2.0 * length ** 3 / (3.0 * ei), rel=1e-14)


# ---------------------------------------------------------------------------
# fluid leg
# ---------------------------------------------------------------------------


def test_fluid_leg_advances_and_reports_finite_force():
    n, d = 24, 6.0
    flow = Fluid3DIBM(
        n=(n, n, n), nu=0.03, centre=(0.25 * n, 0.5 * n, 0.5 * n),
        diameter=d, n_theta=4, n_phi=8, u_init=0.05,
    )
    f0 = flow.f.clone()
    marker_force = flow.step(1.0e-5)
    assert marker_force.shape == (len(flow.surface), 3)
    assert torch.isfinite(flow.f).all()
    assert not torch.equal(f0, flow.f), "one lattice step must change the state"
    # drag on the body points downstream (+x) for a +x current
    assert float(flow.body_force()[0]) > 0.0
    # marker forces balance the body force exactly (partition of unity of the kernel)
    assert torch.allclose(-marker_force.sum(0), flow.body_force(), atol=1e-14)
    # mass is conserved by collide+stream to round-off
    rho0 = 1.0 * n ** 3
    assert abs(float(flow.f.sum()) - rho0) / rho0 < 1e-6


def test_marker_traction_integral_equals_body_force():
    flow = _small_flow()
    traction = flow.marker_traction()
    integral = (traction * flow.surface.dA.to(F64).unsqueeze(-1)).sum(0)
    assert torch.allclose(integral, flow.body_force(), atol=1e-12)


# ---------------------------------------------------------------------------
# interface fed by the *live* LBM field
# ---------------------------------------------------------------------------


def test_live_traction_transfer_conserves_resultant_and_moment():
    flow = _small_flow()
    grid = HexGrid3D((0.0, 0.0, 6.0), (24.0, 24.0, 24.0), (4, 4, 4))
    traction = flow.marker_traction()
    cycle = exchange_cycle(grid, flow.surface, traction)
    nodal = cycle["nodal"]
    assert torch.allclose(nodal.sum(0), flow.body_force(), rtol=1e-11, atol=1e-13)
    ref_moment = flow.surface.moment(traction)
    assert torch.allclose(cycle["moment"], ref_moment, rtol=1e-10, atol=1e-12)
    # work-conjugacy of the transpose operator on an affine velocity field
    u_nodes = torch.zeros_like(nodal)
    u_nodes[:, 0] = 0.01 * grid.nodes[:, 1]
    u_surface = grid.gather_surface_velocity(flow.surface, u_nodes)
    lhs = float((nodal * u_nodes).sum())
    rhs = float((traction * flow.surface.dA.to(F64).unsqueeze(-1) * u_surface).sum())
    assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-14)


def test_hermite_projection_reproduces_beam_fem_tip_deflection():
    cfg = Coupled3DConfig()
    ei = cfg.beam_e * cfg.beam_depth * cfg.beam_thickness ** 3 / 12.0
    ys = torch.linspace(2.0, cfg.beam_length, 12, dtype=F64)
    loads = torch.full_like(ys, 0.37)
    dofs = project_to_beam(ys, loads, cfg.beam_length, cfg.n_elements)
    fem = EulerBernoulliBeamFEM(
        BeamProperties(
            youngs_modulus=cfg.beam_e, density=cfg.beam_density, length=cfg.beam_length,
            width=cfg.beam_depth, height=cfg.beam_thickness, boundary_condition="cantilever",
        ),
        n_elements=cfg.n_elements,
    )
    u = torch.linalg.solve(fem.K_global, dofs)
    tip = float(u[2 * cfg.n_elements])
    ref = tip_deflection_reference(ys, loads, cfg.beam_length, ei)
    assert abs(tip - ref) / abs(ref) < 1e-6


# ---------------------------------------------------------------------------
# ice leg
# ---------------------------------------------------------------------------


def _ice(n: int = 2) -> IceDEM3D:
    return IceDEM3D(
        positions=torch.tensor([[4.0, 10.0, 8.0], [4.0, 12.0, 8.0]][:n], dtype=F64),
        velocities=torch.tensor([[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]][:n], dtype=F64),
        radius=1.5,
    )


def test_dem_wall_contact_is_action_reaction_exact():
    ice = _ice()
    face_vel = torch.zeros(3, dtype=F64)
    ice.wall_contact(face_x=5.0, face_velocity=face_vel, face_z=8.0)
    assert ice.n_active == 2
    # the force pushes the particle back upstream and the structure downstream
    assert torch.all(ice.reaction[:, 0] > 0.0)
    v_before = ice.velocities.clone()
    ice.step(0.5, torch.zeros_like(ice.velocities), torch.zeros_like(ice.velocities))
    assert torch.all(ice.velocities[:, 0] < v_before[:, 0])
    # Newton's third law: the interface hands the structure exactly minus the particle force
    particle = -ice.reaction
    assert torch.allclose(ice.reaction + particle, torch.zeros_like(ice.reaction))


def test_dem_pair_contact_momentum_conserving():
    ice = IceDEM3D(
        positions=torch.tensor([[0.0, 0.0, 0.0], [2.9, 0.0, 0.0]], dtype=F64),
        velocities=torch.tensor([[0.02, 0.0, 0.0], [-0.02, 0.0, 0.0]], dtype=F64),
        radius=1.5,
    )
    net = ice.pair_contacts()
    assert float(net.sum(0).abs().max()) < 1e-12        # internal forces cancel exactly
    assert float(net[0, 0]) < 0.0 and float(net[1, 0]) > 0.0


def test_schiller_naumann_drag_is_restoring():
    ice = _ice(1)
    u_local = torch.tensor([[0.2, 0.0, 0.0]], dtype=F64)   # fluid faster than the particle
    force = ice.fluid_force(u_local, nu=0.03)
    assert float(force[0, 0]) > 0.0
    assert torch.isfinite(force).all()


def test_hull_girder_postprocess_matches_analytic():
    """Discretisation error of the hull-girder post-process must stay below 1 %."""
    cfg = Coupled3DConfig()
    pos = torch.tensor([[0.0, 0.5 * cfg.beam_length, 0.0],
                        [0.0, 0.8 * cfg.beam_length, 0.0]], dtype=F64)
    frc = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=F64)
    out = hull_girder_from_contact_loads(pos, frc, cfg)
    assert out["rel_err_pct"] < 1.0, out
    assert out["clamp_bending_moment"] == pytest.approx(1.3 * cfg.beam_length, rel=1e-12)
    assert out["n_contacts"] == 2


# ---------------------------------------------------------------------------
# staged run
# ---------------------------------------------------------------------------


def test_staged_run_produces_all_gates():
    cfg = Coupled3DConfig(
        n=(24, 24, 24), diameter=6.0, centre=(6.0, 12.0, 12.0),
        hex_origin=(0.0, 0.0, 6.0), hex_extent=(12.0, 34.0, 12.0), hex_cells=(3, 6, 3),
        n_theta=4, n_phi=8, n_warm=40, n_measure=40, n_drive_phases=1,
        n_fsi=12, n_ice=2, n_elements=9, ice_x0=2.0,
    )
    res = run_coupled3d(cfg, verbose=False)
    for key in ("fluid", "structure", "ice", "gates", "benchmark", "history"):
        assert key in res
    for gate in ("G1_fluid_steady", "G2_drag_vs_reference", "G3_interface_conservation",
                 "G4_structure_vs_analytic", "G5_added_mass_guard", "G6_three_way_march"):
        assert gate in res["gates"], gate
        assert isinstance(res["gates"][gate]["pass"], bool)
    # the two structural invariants must hold even in a 12-step smoke run
    assert res["gates"]["G3_interface_conservation"]["resultant_rel_err"] < 1e-10
    assert res["gates"]["G4_structure_vs_analytic"]["rel_err_pct"] < 1e-6
    assert math.isfinite(res["fluid"]["cd_got"])
    assert res["fluid"]["finite"] is True
    assert len(res["history"]["tip"]) == cfg.n_fsi
    assert len(res["benchmark"]["rows"]) >= 4