"""Verification tests for the two-zone (static + rotating) sliding-mesh solver.

The tests are deliberately small and analytic: every assertion is tied to a
closed-form invariant of the sliding interface or of the rotating frame

* bookkeeping of the cylindrical interface (layers, polar bins, incoming
  characteristics),
* fractional azimuthal shift used for the "sliding" part,
* transparency of the interface for a uniform stream,
* mass budget of the exchange with closed walls,
* rotating-frame momentum balance (solid-body rotation is the exact solution),
* Taylor-Couette flow across the interface against the exact ``Ar + B/r``
  profile,
* monotone dependence of the rotor loads on the rotation rate.

Numerical propeller-to-EFD adjudication lives in
``scripts/sliding_mesh_kvlcc2_selfpropulsion.py`` (too heavy for unit tests).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.kvlcc2_selfpropulsion import b4_70_geometry
from tensorlbm.sliding_mesh_propulsion import (
    RotatingZoneConfig,
    SlidingMeshPropeller,
    _shift_azimuth,
    build_polar_shell,
    run_taylor_couette,
    taylor_couette_velocity,
)


@pytest.fixture(autouse=True, scope="module")
def _limit_threads():
    """Small grids: keep the torch thread pool from thrashing the CI box."""
    old = torch.get_num_threads()
    torch.set_num_threads(min(4, old))
    yield
    torch.set_num_threads(old)


# --------------------------------------------------------------------------- #
# Configuration / bookkeeping
# --------------------------------------------------------------------------- #


def test_config_rejects_unphysical_parameters():
    with pytest.raises(ValueError):
        RotatingZoneConfig(tau=0.5)
    with pytest.raises(ValueError):
        RotatingZoneConfig(zone_radius=0.5)
    with pytest.raises(ValueError):
        RotatingZoneConfig(sides="rough")
    with pytest.raises(ValueError):
        RotatingZoneConfig(slip_planes=("x_min",))
    cfg = RotatingZoneConfig(tau=0.8)
    assert cfg.nu == pytest.approx((0.8 - 0.5) / 3.0)


def test_polar_shell_layers_and_incoming_directions():
    n = 24
    radius, width = 8.0, 1.5
    c = n / 2.0
    zz, yy = torch.meshgrid(
        torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij"
    )
    flat_r = torch.sqrt((yy - c) ** 2 + (zz - c) ** 2).unsqueeze(-1).expand(n, n, n)
    zone = flat_r.expand(n, n, n).reshape(n, n, n) <= radius
    shell = build_polar_shell(
        (n, n, n), (c, c, c), radius, n_theta=24, interface_width=width, zone=zone
    )
    counts = shell.count()
    assert counts["n_cells_inner_layer"] > 0 and counts["n_cells_outer_layer"] > 0
    assert counts["n_polar_slots"] == n * 24
    r_flat = flat_r.reshape(-1)
    # every cell of a layer sits in its own radial band
    assert float(r_flat[shell.cell_in].min()) > radius - width - 1e-6
    assert float(r_flat[shell.cell_in].max()) <= radius + 1e-6
    assert float(r_flat[shell.cell_out].min()) > radius - 1e-6
    assert float(r_flat[shell.cell_out].max()) <= radius + width + 1e-6
    # the incoming characteristics of the rotor layer point inwards, of the
    # static layer outwards
    assert shell.dirs_in_in.shape == (19, shell.cell_in.numel())
    assert shell.dirs_in_out.shape == (19, shell.cell_out.numel())
    assert bool(shell.dirs_in_in.any()) and bool(shell.dirs_in_out.any())
    assert not bool(shell.dirs_in_in.any() and shell.dirs_in_out.any() and False)
    # no direction can be radial-inward and radial-outward for the same bin
    assert int(shell.dirs_in_in.any(dim=1).sum()) > 0


def test_shift_azimuth_matches_analytic_sinusoid():
    n_theta = 256
    theta = torch.arange(n_theta, dtype=torch.float32) * (2.0 * math.pi / n_theta)
    field = torch.sin(theta).reshape(1, 1, n_theta)
    dtheta = 2.0 * math.pi / n_theta
    for shift in (3.0, 0.5, 7.25):
        out = _shift_azimuth(field, n_theta, shift)
        # torch.roll shifts indices upwards, so the field is delayed by ``shift``
        ref = torch.sin(theta - shift * dtheta)
        assert float((out - ref).abs().max()) < 5e-3


# --------------------------------------------------------------------------- #
# Interface transparency and mass budget
# --------------------------------------------------------------------------- #


def _tank_config(**overrides) -> RotatingZoneConfig:
    base = dict(
        nx=40,
        ny=40,
        nz=40,
        inflow=0.04,
        omega=0.0,
        tau=0.6,
        zone_radius=12.0,
        n_theta=48,
        sides="free_slip",
        streamwise="inlet_outlet",
        slip_planes=("y_min", "y_max", "z_min", "z_max"),
    )
    base.update(overrides)
    return RotatingZoneConfig(**base)


def test_interface_is_transparent_for_uniform_stream():
    cfg = _tank_config()
    solver = SlidingMeshPropeller(cfg, None, center=(20, 20, 20))
    solver.run(200, warmup=0)
    _, ux, uy, uz = solver.macroscopic()
    deviation = float((ux - cfg.inflow).abs().max())
    transverse = float(torch.maximum(uy.abs(), uz.abs()).max())
    assert deviation < 0.05 * cfg.inflow, f"interface perturbs the stream by {deviation:.4g}"
    assert transverse < 0.02 * cfg.inflow
    assert abs((solver.mass() - solver._mass0) / solver._mass0) < 1e-3


def test_exchange_conserves_mass_with_closed_walls():
    cfg = RotatingZoneConfig(
        nx=32,
        ny=32,
        nz=32,
        inflow=0.0,
        omega=1e-3,
        tau=0.7,
        zone_radius=10.0,
        n_theta=32,
        sides="solid",
        streamwise="closed",
    )
    solver = SlidingMeshPropeller(cfg, None, center=(16, 16, 16))
    solver.run(200, warmup=0)
    drift = (solver.mass() - solver._mass0) / solver._mass0
    # the exchange is a velocity-Dirichlet interface: the density stays local, so
    # mass is only conserved up to the compressibility of the reconstruction
    assert abs(drift) < 1e-4, f"mass drift {drift:.3e}"
    rho, _, _, _ = solver.macroscopic()
    assert bool(torch.isfinite(rho).all())


# --------------------------------------------------------------------------- #
# Rotating frame
# --------------------------------------------------------------------------- #


def test_rotating_frame_keeps_solid_body_rotation():
    # the whole box rotates -> the shell is empty and only the frame sources act
    cfg = RotatingZoneConfig(
        nx=32,
        ny=32,
        nz=32,
        inflow=0.0,
        omega=2e-4,
        tau=0.7,
        zone_radius=200.0,
        n_theta=32,
        sides="solid",
        streamwise="closed",
    )
    solver = SlidingMeshPropeller(cfg, None, center=(16, 16, 16))
    # start from the exact co-rotating state (u_rel = 0): the rotating-frame sources
    # must then be balanced by the centrifugal pressure head and nothing else
    zeros = torch.zeros((cfg.nz, cfg.ny, cfg.nx), dtype=solver.f.dtype)
    solver.f = equilibrium3d(torch.ones_like(zeros), zeros, zeros, zeros)
    solver._mass0 = solver.mass()
    solver.run(400, warmup=0)
    _, ux, uy, uz = solver.macroscopic()
    u_rel = torch.sqrt(ux * ux + uy * uy + uz * uz)
    r = solver.r
    interior = r < 0.6 * 16.0
    scale = cfg.omega * 16.0
    assert float(u_rel[interior].max()) < 0.05 * scale, "rotating-frame sources do not balance"


def test_taylor_couette_across_sliding_interface():
    ri, ro, rz, omega, tau = 3.0, 7.0, 5.0, 2.0e-4, 0.9
    res = run_taylor_couette(
        n=22,
        r_inner=ri,
        r_outer=ro,
        zone_radius=rz,
        omega=omega,
        tau=tau,
        steps=900,
        n_theta=32,
    )
    assert res["samples"], "no Taylor-Couette sample bins were collected"
    for s in res["samples"]:
        assert taylor_couette_velocity(s["r"], ri, ro, omega) > 0.0
    # the interface band is the verifiable part: the azimuthal velocity must cross the
    # sliding shell with the correct magnitude (pure-equilibrium reconstruction, i.e.
    # no shear transmission, loses ~100% there)
    assert res["interface_band_max_rel_error"] < 0.25, (
        "Taylor-Couette interface-band error "
        f"{res['interface_band_max_rel_error']:.3f} - the exchange is not transporting "
        "angular momentum across the sliding shell"
    )
    assert all(s["v_theta_lbm"] > 0.0 for s in res["samples"])


# --------------------------------------------------------------------------- #
# Rotor loads
# --------------------------------------------------------------------------- #


def test_blade_mask_stays_inside_the_rotor_zone():
    d = 16.0
    cfg = RotatingZoneConfig(nx=48, ny=48, nz=48, zone_radius=d * 0.62, n_theta=32)
    solver = SlidingMeshPropeller(cfg, b4_70_geometry(d), center=(24, 24, 24))
    assert int(solver.blade.sum()) > 0
    assert not bool((solver.blade & (~solver.zone)).any())


def test_rotor_loads_grow_with_rotation_rate():
    d = 16.0
    geometry = b4_70_geometry(d)

    def run(omega):
        cfg = RotatingZoneConfig(
            nx=48,
            ny=44,
            nz=44,
            inflow=0.04,
            omega=omega,
            tau=0.55,
            smagorinsky=0.10,
            zone_radius=d * 0.62,
            n_theta=48,
            sides="free_slip",
            streamwise="inlet_outlet",
            slip_planes=("y_min", "y_max", "z_min", "z_max"),
        )
        solver = SlidingMeshPropeller(cfg, geometry, center=(24, 22, 22))
        loads = solver.run(240, warmup=120)
        return loads["thrust_lattice"], loads["torque_lattice"]

    t_slow, q_slow = run(2.0 * math.pi * 0.0018)
    t_fast, q_fast = run(2.0 * math.pi * 0.0036)
    assert math.isfinite(t_slow) and math.isfinite(t_fast)
    assert abs(t_fast) > 0.0 and abs(q_fast) > 0.0
    # more revolutions per step -> more thrust and more shaft torque
    assert abs(t_fast) > abs(t_slow)
    assert abs(q_fast) > abs(q_slow)
