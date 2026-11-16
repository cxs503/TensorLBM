"""Verification suite for the D3Q19 AA-pattern fused stream-collide kernel.

Anchor: ``XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE`` (lanl/MF-LBM
``aa-pattern-fused-stream-collide``, D3Q19).

The 2D sibling (:mod:`tensorlbm.aa_fused_push_pull`) documents its own scope
limit — "D2Q9 only, nothing transfers to D3Q19/27".  These tests pin the
properties that make the D3Q19 transfer *verified* rather than merely ported:

* the operator is bit-identical to the canonical materialised collide-then-stream
  scheme on a wall-bounded, non-uniform, forced state (equality, not tolerance);
* the vacuity control is rejected: the historical destination-colliding
  ``C[S[f]]`` operator must NOT reproduce the canonical step;
* mass over the fluid nodes is conserved in a closed box (halfway bounce-back on
  the fluid side of the wall, not on the solid nodes);
* three exact solutions with the owner precision standard attached:
  plane Couette (moving-wall term), Guo-forced plane Poiseuille (weights +
  forcing, half-cell offset error), decaying shear wave (transport coefficient
  ``nu = (tau-1/2)/3``) and the 2D-extended Taylor-Green vortex (an exact 3D
  Navier-Stokes solution with a non-zero convective term).

Runtime of the whole module: ~30 s on 2 threads (kept deliberately cheap so it
runs in the ordinary test suite, not only on the scheduler).
"""

from __future__ import annotations

import math

import pytest
import torch

import tensorlbm.aa_fused_push_pull_3d as M

torch.set_grad_enabled(False)

CS2 = 1.0 / 3.0


# --------------------------------------------------------------------------- #
# lattice constants (imported from tensorlbm.d3q19 — they must not drift)
# --------------------------------------------------------------------------- #
def test_lattice_constants_are_a_consistent_d3q19():
    assert len(M.W19) == 19
    assert abs(sum(M.W19) - 1.0) < 1e-15
    for i in range(19):
        j = M.OPP[i]
        assert M.OPP[j] == i
        assert (M.CX[i], M.CY[i], M.CZ[i]) == (-M.CX[j], -M.CY[j], -M.CZ[j])
    # isotropy of the second and fourth moments (these fix c_s^2 = 1/3 and the
    # viscosity prefactor; a swapped weight table would break them)
    m2x = sum(M.W19[i] * M.CX[i] ** 2 for i in range(19))
    m4x = sum(M.W19[i] * M.CX[i] ** 4 for i in range(19))
    m22 = sum(M.W19[i] * M.CX[i] ** 2 * M.CY[i] ** 2 for i in range(19))
    assert abs(m2x - CS2) < 1e-15
    assert abs(m4x - 3 * CS2 ** 2) < 1e-15
    assert abs(m22 - CS2 ** 2) < 1e-15


def test_shape_is_validated():
    with pytest.raises(ValueError):
        M.fused_step(torch.zeros(9, 4, 4, 4, dtype=torch.float64), 0.8)
    with pytest.raises(ValueError):
        M.collide_stream_step(torch.zeros(19, 4, 4, dtype=torch.float64), 0.8)


# --------------------------------------------------------------------------- #
# operator identity — the non-vacuous equivalence property
# --------------------------------------------------------------------------- #
def _wall_force_case(nz=4, ny=6, nx=5, seed=7, tau=0.63):
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    solid[:, 3, 0:2] = True                                  # interior obstacle
    wall_u = M.wall_velocity_3d(solid, (0.05, 0.0, -0.02), row=-1)
    force = torch.zeros((3, nz, ny, nx), dtype=torch.float64)
    force[0] = 1e-4
    force[2] = -3e-5
    g = torch.Generator().manual_seed(seed)
    rho = 1.0 + 0.02 * torch.rand((nz, ny, nx), generator=g, dtype=torch.float64)
    u = [0.02 * torch.randn((nz, ny, nx), generator=g, dtype=torch.float64) for _ in range(3)]
    f = M.equilibrium(rho, u[0], u[1], u[2])
    return f, solid, wall_u, force, tau


def test_fused_is_bit_identical_to_materialised_with_walls_force():
    f, solid, wall_u, force, tau = _wall_force_case()
    fused = M.fused_step(f, tau, solid=solid, wall_u=wall_u, force=force)
    canon = M.collide_stream_step(f, tau, solid=solid, wall_u=wall_u, force=force)
    assert torch.equal(fused, canon)
    assert int((fused != canon).sum()) == 0
    assert float((fused - canon).abs().max()) == 0.0


def test_fused_is_bit_identical_periodic():
    g = torch.Generator().manual_seed(3)
    rho = 1.0 + 0.01 * torch.rand((3, 5, 4), generator=g, dtype=torch.float64)
    u = [0.03 * torch.randn((3, 5, 4), generator=g, dtype=torch.float64) for _ in range(3)]
    f = M.equilibrium(rho, u[0], u[1], u[2])
    assert torch.equal(M.aa_fused_stream_collide_3d(f, 0.7),
                       M.separate_stream_collide_3d(f, 0.7))


def test_vacuity_control_stream_then_collide_is_rejected():
    """``C[S[f]]`` (destination colliding) must not look equivalent.

    Without this test the equality test above could be passed by a suite that
    compares two copies of the same wrong operator.
    """
    f, solid, wall_u, force, tau = _wall_force_case()
    canon = M.collide_stream_step(f, tau, solid=solid, wall_u=wall_u, force=force)

    streamed = torch.empty_like(f)
    for i in range(M.ND):
        streamed[i] = M._pull(f[i], i)
    rho, ux, uy, uz = M.macroscopic(streamed)
    cs = streamed + (1.0 / tau) * (M.equilibrium(rho, ux, uy, uz) - streamed)
    rel = float((cs - canon).abs().max() / f.abs().max())
    assert rel > 1e-6, f"stream-then-collide matched the oracle ({rel:.3e}) — test is vacuous"


# --------------------------------------------------------------------------- #
# conservation
# --------------------------------------------------------------------------- #
def test_mass_conserved_on_fluid_nodes_closed_box():
    f, solid, wall_u, force, tau = _wall_force_case(seed=5)
    fluid = ~solid
    m0 = float(f[:, fluid].sum())
    cur = f
    for _ in range(50):
        cur = M.fused_step(cur, tau, solid=solid, wall_u=wall_u, force=force)
    m1 = float(cur[:, fluid].sum())
    assert abs(m1 - m0) / m0 < 1e-12


def test_solid_node_populations_do_not_leak_into_fluid():
    """Solid-node values are junk; they must be overwritten, never read back."""
    f, solid, wall_u, force, tau = _wall_force_case(seed=9)
    f = f.clone()
    f[:, solid] = 1e3                       # poison the solid nodes
    a = M.fused_step(f, tau, solid=solid, wall_u=wall_u, force=force)
    clean = f.clone()
    clean[:, solid] = 0.0
    b = M.fused_step(clean, tau, solid=solid, wall_u=wall_u, force=force)
    fluid = ~solid
    assert torch.equal(a[:, fluid], b[:, fluid])


# --------------------------------------------------------------------------- #
# exact solutions (owner standard: < 3 % is the gate, < 15 % unacceptable)
# --------------------------------------------------------------------------- #
def test_plane_couette_exact_profile():
    """Moving-wall term ``6 w rho (c . u_w)``: sign and magnitude."""
    n_fluid, u_wall, tau, steps = 8, 0.05, 0.8, 1000
    ny = n_fluid + 2
    nz, nx = 2, 2
    solid = M.channel_masks_3d(nz, ny, nx, solid_rows=1)
    wall_u = M.wall_velocity_3d(solid, (u_wall, 0.0, 0.0), row=-1)
    rho = torch.ones((nz, ny, nx), dtype=torch.float64)
    zero = torch.zeros_like(rho)
    f = M.equilibrium(rho, zero, zero, zero)
    for _ in range(steps):
        f = M.fused_step(f, tau, solid=solid, wall_u=wall_u)
    _, ux, _, _ = M.macroscopic(f)
    prof = ux[:, 1:-1, :].mean(dim=(0, 2))
    exact = M.analytic_couette(n_fluid, u_wall)
    err = M.relative_l2(prof, exact)
    assert err < 1e-4, f"Couette profile error {err:.3e} (exact solution available)"


def test_plane_couette_sign_flip_is_rejected():
    """A sign-inverted wall term must fail the same gate (control arm)."""
    n_fluid, u_wall, tau, steps = 8, 0.05, 0.8, 600
    ny = n_fluid + 2
    nz, nx = 2, 2
    solid = M.channel_masks_3d(nz, ny, nx, solid_rows=1)
    wall_u = -M.wall_velocity_3d(solid, (u_wall, 0.0, 0.0), row=-1)  # flipped
    rho = torch.ones((nz, ny, nx), dtype=torch.float64)
    zero = torch.zeros_like(rho)
    f = M.equilibrium(rho, zero, zero, zero)
    for _ in range(steps):
        f = M.fused_step(f, tau, solid=solid, wall_u=wall_u)
    _, ux, _, _ = M.macroscopic(f)
    prof = ux[:, 1:-1, :].mean(dim=(0, 2))
    exact = M.analytic_couette(n_fluid, u_wall)
    assert M.relative_l2(prof, exact) > 0.1


def test_plane_poiseuille_exact_parabola():
    """Guo forcing + D3Q19 weights vs ``u(y) = g/(2nu) (y+1/2)(H-1/2-y)``."""
    n_fluid, tau, u_max = 8, 0.8, 0.05
    nu = (tau - 0.5) / 3.0
    g = 8.0 * nu * u_max / n_fluid ** 2
    ny = n_fluid + 2
    nz, nx = 2, 2
    solid = M.channel_masks_3d(nz, ny, nx, solid_rows=1)
    force = torch.zeros((3, nz, ny, nx), dtype=torch.float64)
    force[0] = g
    rho = torch.ones((nz, ny, nx), dtype=torch.float64)
    zero = torch.zeros_like(rho)
    f = M.equilibrium(rho, zero, zero, zero)
    for _ in range(1800):
        f = M.fused_step(f, tau, solid=solid, force=force)
    _, ux, _, _ = M.macroscopic(f, force)
    prof = ux[:, 1:-1, :].mean(dim=(0, 2))
    exact, u_max_exact = M.analytic_poiseuille(n_fluid, g, nu)
    err = M.relative_l2(prof, exact)
    assert err < 0.03, f"Poiseuille profile error {err:.3%} exceeds the 3 % gate"
    assert abs(float(prof.max()) - u_max_exact) / u_max_exact < 0.05


def test_shear_wave_measures_the_right_viscosity():
    """Decay of a transverse mode -> ``nu_eff`` must match ``(tau-1/2)/3``."""
    ny, nz, nx, u0, tau, steps = 32, 2, 2, 0.02, 0.8, 200
    nu = (tau - 0.5) / 3.0
    k_lat = 2.0 * math.pi / ny
    f = M.shear_wave_initial(nz, ny, nx, u0, k_lat)
    for _ in range(steps):
        f = M.fused_step(f, tau)
    _, ux, _, _ = M.macroscopic(f)
    prof = ux[:, :, 0].mean(dim=0)
    amp = float(torch.sqrt(torch.mean(prof ** 2)))
    nu_eff = -math.log(amp / (u0 / math.sqrt(2))) / (k_lat ** 2 * steps)
    assert abs(nu_eff - nu) / nu < 0.03, f"nu_eff = {nu_eff:.5f} vs nu = {nu:.5f}"


def test_taylor_green_vortex_exact_solution():
    """2D-extended TG vortex: exact 3D Navier-Stokes solution, decay + shape."""
    n, nz, u0, tau, steps = 32, 2, 0.02, 0.8, 90
    nu = (tau - 0.5) / 3.0
    k_lat = 2.0 * math.pi / n
    f = M.taylor_green_initial(nz, n, u0, k_lat)
    for _ in range(steps):
        f = M.fused_step(f, tau)
    rho, ux, uy, uz = M.macroscopic(f)
    ex, ey, _ = M.taylor_green_exact(n, float(steps), u0, k_lat, nu)
    err = M.relative_l2(torch.stack([ux[0], uy[0]]), torch.stack([ex, ey]))
    assert err < 0.03, f"TG vortex L2 error {err:.3%} exceeds the 3 % gate"
    assert float(uz.abs().max()) < 1e-6          # z symmetry preserved
    assert abs(float(rho.mean()) - 1.0) < 1e-10  # mass preserved


def test_taylor_green_error_falls_with_refinement():
    """The gate is a discretisation error, not a fudged constant."""
    nu, u0, tau = 0.1, 0.02, 0.8
    errs = {}
    for n, steps in ((32, 90), (64, 360)):
        k_lat = 2.0 * math.pi / n
        f = M.taylor_green_initial(1, n, u0, k_lat)
        for _ in range(steps):
            f = M.fused_step(f, tau)
        _, ux, uy, _ = M.macroscopic(f)
        ex, ey, _ = M.taylor_green_exact(n, float(steps), u0, k_lat, nu)
        errs[n] = M.relative_l2(torch.stack([ux[0], uy[0]]), torch.stack([ex, ey]))
    order = math.log(errs[32] / errs[64]) / math.log(2.0)
    assert errs[64] < errs[32]
    assert order > 1.5, f"observed order {order:.2f} — expected ~2nd order"