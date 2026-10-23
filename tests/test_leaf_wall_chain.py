"""Contract tests for the leaf wall-model chain (E2, route C).

The leaf (SoA) chain must be an operator port of the grid tier's pieces,
so the octree-shell runner inherits the uniform formal tier's wall model
rather than an approximation:

* ``leaf_bfl_link_normals`` reproduces ``compute_bfl_link_normal`` on the
  same flattened mask bit for bit;
* ``leaf_wall_model_slip_velocity`` is the exact ``wall_model_slip``
  projection of the pre-stream velocity;
* ``leaf_solve_wall_law`` is ``wall_model._solve_wall_law`` itself
  (scalar ``nu``) and stays exact per level with a per-leaf ``nu``;
* ``leaf_guo_body_force`` reproduces ``guo_body_force_d3q19``/``_d3q27``
  bit for bit and its first moment is the requested force;
* ``leaf_wall_shear_traction`` applies ``-tau_w * t_hat`` per boundary
  leaf with the wall-law ``tau_w``;
* ``build_leaf_wall_chain`` places the model on the finest leaf level
  with the dual-viscosity wall ``nu``;
* the ``bfl_fn`` factory runs inside ``step_octree_shell`` end to end.
"""

from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.leaf_wall import (
    LeafWallForceProbe,
    build_leaf_wall_chain,
    leaf_bfl_link_normals,
    leaf_dual_viscosity_wall_nu,
    leaf_guo_body_force,
    leaf_solve_wall_law,
    leaf_wall_model_slip_velocity,
    leaf_wall_shear_traction,
    make_leaf_wall_chain_bfl_fn,
)
from tensorlbm.octree_boundary.stepping import build_ghost_plan, step_octree_shell
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.static_block_amr import AMRAdvanceResult
from tensorlbm.wall_chain import bfl_halfway_links_3d
from tensorlbm.wall_model import (
    _solve_wall_law,
    compute_bfl_link_normal,
    guo_body_force_d3q19,
    guo_body_force_d3q27,
    physical_wall_lattice_viscosity,
)


def _sphere_solid(shape, center, radius, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(shape[0], device=device, dtype=torch.float32),
        torch.arange(shape[1], device=device, dtype=torch.float32),
        torch.arange(shape[2], device=device, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    return r < radius


def _equilibrium_flat(n, q, u=(0.05, 0.02, -0.01), device=None, rho=1.0):
    rho_t = torch.full((n,), rho, device=device, dtype=torch.float32)
    u_t = [torch.full((n,), value, device=device, dtype=torch.float32) for value in u]
    if q == 19:
        feq = equilibrium3d(rho_t, u_t[0], u_t[1], u_t[2])
    else:
        from tensorlbm.d3q27 import equilibrium27

        feq = equilibrium27(rho_t, u_t[0], u_t[1], u_t[2])
    return feq.view(q, -1)


def test_link_normals_bitwise_parity_with_grid() -> None:
    device = torch.device("cpu")
    solid = _sphere_solid((12, 12, 12), (6.0, 6.0, 6.0), 4.5, device)
    mask, _ = bfl_halfway_links_3d(solid)
    assert bool(mask.any())
    gx, gy, gz = compute_bfl_link_normal(mask)
    lx, ly, lz = leaf_bfl_link_normals(mask.view(19, -1))
    assert torch.equal(lx.view_as(gx), gx)
    assert torch.equal(ly.view_as(gy), gy)
    assert torch.equal(lz.view_as(gz), gz)


def test_slip_velocity_projection() -> None:
    device = torch.device("cpu")
    f = _equilibrium_flat(37, 19, device=device)
    normals = (
        torch.ones(37, device=device),
        torch.zeros(37, device=device),
        torch.zeros(37, device=device),
    )
    rho, ux, uy, uz = leaf_wall_model_slip_velocity(f, normals, normal_activation=1.0)
    assert torch.allclose(ux, torch.zeros_like(ux), atol=1e-6)
    assert torch.allclose(uy, torch.full_like(uy, 0.02), atol=1e-6)
    assert torch.allclose(uz, torch.full_like(uz, -0.01), atol=1e-6)
    rho0, ux0, _, _ = leaf_wall_model_slip_velocity(f, normals, normal_activation=0.0)
    assert torch.allclose(ux0, torch.full_like(ux0, 0.05), atol=1e-6)
    assert torch.allclose(rho, rho0, atol=0.0)
    _, ux_half, _, _ = leaf_wall_model_slip_velocity(f, normals, normal_activation=0.5)
    assert torch.allclose(ux_half, torch.full_like(ux_half, 0.025), atol=1e-6)


def test_solve_wall_law_scalar_bitwise_and_grouped() -> None:
    g = torch.Generator().manual_seed(3)
    n = 64
    u_tan = 0.3 * torch.rand(n, generator=g).clamp(min=1e-3)
    near = torch.rand(n, generator=g) < 0.8
    nu, y_val = 0.02, 0.5
    # scalar nu: the grid solver itself, bit for bit (its 12-iteration
    # fixed-point accuracy is the frozen operator's own property, audited
    # with the wall_model suite — not re-checked here)
    got = leaf_solve_wall_law(u_tan, nu, y_val, "reichardt", near)
    want = _solve_wall_law(u_tan, nu, y_val, "reichardt", near)
    assert torch.equal(got, want)
    # per-leaf nu grouped by level: exact per-group scalar solves
    levels = torch.where(torch.arange(n) < n // 2, 1, 2)
    nu_leaf = torch.where(levels == 1, torch.full((n,), 0.02), torch.full((n,), 0.04))
    grouped = leaf_solve_wall_law(u_tan, nu_leaf, y_val, "reichardt", near, leaf_level=levels)
    for level, nu_level in ((1, 0.02), (2, 0.04)):
        sel = (levels == level) & near
        want = _solve_wall_law(
            u_tan[sel], nu_level, y_val, "reichardt", torch.ones(u_tan[sel].shape, dtype=torch.bool)
        )
        assert torch.equal(grouped[sel], want)
    assert bool((grouped[~near] == 0).all())


def test_guo_body_force_bitwise_parity() -> None:
    device = torch.device("cpu")
    g = torch.Generator().manual_seed(5)
    shape = (4, 5, 6)
    f = _equilibrium_flat(math.prod(shape), 19, device=device).view(19, *shape)
    f = f + 2.0e-3 * torch.randn(f.shape, generator=g)
    fx = 1e-4 * torch.rand(shape, generator=g)
    fy = 1e-4 * torch.rand(shape, generator=g)
    fz = 1e-4 * torch.rand(shape, generator=g)
    ux = 0.05 * torch.ones(shape)
    uy = 0.02 * torch.ones(shape)
    uz = -0.01 * torch.ones(shape)
    grid = guo_body_force_d3q19(f, fx, fy, fz, ux, uy, uz)
    leaf = leaf_guo_body_force(
        f.view(19, -1),
        fx.reshape(-1),
        fy.reshape(-1),
        fz.reshape(-1),
        ux.reshape(-1),
        uy.reshape(-1),
        uz.reshape(-1),
    )
    assert torch.equal(leaf.view_as(grid), grid)
    # D3Q27 parity
    from tensorlbm.d3q27 import equilibrium27

    rho = torch.ones(shape)
    f27 = equilibrium27(rho, ux, uy, uz)
    grid27 = guo_body_force_d3q27(f27, fx, fy, fz, ux, uy, uz)
    leaf27 = leaf_guo_body_force(
        f27.view(27, -1),
        fx.reshape(-1),
        fy.reshape(-1),
        fz.reshape(-1),
        ux.reshape(-1),
        uy.reshape(-1),
        uz.reshape(-1),
    )
    assert torch.equal(leaf27.view_as(grid27), grid27)


def test_guo_first_moment_equals_requested_force() -> None:
    device = torch.device("cpu")
    g = torch.Generator().manual_seed(7)
    n = 16
    f = _equilibrium_flat(n, 19, device=device).to(torch.float64)
    fx = 3e-4 * torch.rand(n, generator=g, dtype=torch.float64)
    fy = -2e-4 * torch.rand(n, generator=g, dtype=torch.float64)
    fz = 1e-4 * torch.rand(n, generator=g, dtype=torch.float64)
    ux = torch.full((n,), 0.05, dtype=torch.float64)
    uy = torch.full((n,), -0.02, dtype=torch.float64)
    uz = torch.full((n,), 0.01, dtype=torch.float64)
    out = leaf_guo_body_force(f, fx, fy, fz, ux, uy, uz)
    from tensorlbm.d3q19 import C

    moment = C.to(torch.float64).t() @ (out - f)
    assert torch.allclose(moment[0], fx, rtol=0, atol=1e-15)
    assert torch.allclose(moment[1], fy, rtol=0, atol=1e-15)
    assert torch.allclose(moment[2], fz, rtol=0, atol=1e-15)


def test_shear_traction_force_direction_and_units() -> None:
    device = torch.device("cpu")
    n = 24
    # u perpendicular to the wall normal: the tangential direction is the
    # clean x axis (u parallel to n would leave it defined only by fp32 noise)
    f = _equilibrium_flat(n, 19, u=(0.12, 0.0, 0.0), device=device)
    normals = (
        torch.zeros(n, device=device),
        torch.ones(n, device=device),
        torch.zeros(n, device=device),
    )
    near = torch.ones(n, dtype=torch.bool, device=device)
    nu, y_val = 0.02, 0.5
    f_out, diag = leaf_wall_shear_traction(f, normals, near, nu, wall_law="reichardt", y_val=y_val)
    assert bool(torch.isfinite(f_out).all())
    ut = _solve_wall_law(torch.full((n,), 0.12), nu, y_val, "reichardt", near)
    tau_w = float((ut * ut)[0])
    shear = diag["shear_force"]
    assert abs(float(shear[0]) - (-n * tau_w)) <= 1e-4 * n * tau_w
    assert abs(float(shear[1])) <= 1e-6
    assert abs(float(shear[2])) <= 1e-6
    assert diag["active_nodes"] == n


def _small_octree(device):
    host_shape = (24, 24, 24)
    octree = build_octree_shell(
        host_shape,
        center=(12.0, 12.0, 12.0),
        radius=5.0,
        bl_thickness_cells=2.0,
        d_max=2,
        transition=1,
        lattice="D3Q19",
        device=device,
    )
    plan = build_ghost_plan(octree, host_shape)
    return octree, plan, host_shape


def test_build_leaf_wall_chain_finest_only_and_dual_viscosity() -> None:
    device = torch.device("cpu")
    octree, _, _ = _small_octree(device)
    levels = octree.leaf_level
    d_max = int(levels.max().item())
    assert d_max == 2
    u_in, length_host, re = 0.06, 10.0, 1.0e5
    chain = build_leaf_wall_chain(
        octree, u_in=u_in, length_host_cells=length_host, reynolds_physical=re
    )
    assert bool((chain.near <= (levels == d_max)).all())
    assert bool(chain.near.any())
    # scalar wall nu on the single finest level: L = length_host * 2^2
    want = physical_wall_lattice_viscosity(u_in, length_host * 4.0, re)
    assert abs(chain.wall_nu - want) <= 1e-12 * max(1.0, abs(want))
    # per-level dual viscosity halves per coarsening
    nu_leaf = leaf_dual_viscosity_wall_nu(u_in, length_host, re, levels)
    nu1 = float(nu_leaf[levels == 1][0])
    nu2 = float(nu_leaf[levels == 2][0])
    assert abs(nu2 / nu1 - 2.0) <= 1e-12
    assert chain.coarser_bfl_links == int(
        octree.bfl_mask[:, octree.bfl_mask.any(dim=0) & (levels != d_max)].sum().item()
    )


def test_wall_chain_bfl_fn_inside_step_octree_shell() -> None:
    device = torch.device("cpu")
    octree, plan, host_shape = _small_octree(device)
    u_in, length_host, re = 0.06, 10.0, 1.0e5
    chain = build_leaf_wall_chain(
        octree, u_in=u_in, length_host_cells=length_host, reynolds_physical=re
    )
    step_holder = [4]  # ramp_steps=3 -> full activation
    bfl_fn, probe = make_leaf_wall_chain_bfl_fn(
        octree, chain, get_step=lambda: step_holder[0], ramp_steps=3
    )
    assert isinstance(probe, LeafWallForceProbe)

    rho = torch.ones(host_shape, device=device)
    ux = torch.full_like(rho, u_in)
    zero = torch.zeros_like(rho)
    f_host = equilibrium3d(rho, ux, zero, zero)
    octree.f_leaf = _equilibrium_flat(octree.n_leaf, 19, device=device)

    def advance(f_leaf, tau, level, substep):
        q = f_leaf.shape[0]
        post = collide_bgk3d(f_leaf.view(q, 1, 1, -1), tau).view(q, -1)
        return AMRAdvanceResult(post.view_as(f_leaf).clone(), post.view_as(f_leaf))

    collided = collide_bgk3d(f_host, 0.7)
    streamed = stream3d(collided)
    ledger = step_octree_shell(
        octree,
        advance,
        f_host,
        streamed,
        tau_coarse=0.7,
        l1_post=[collided],
        shell_level=1,
        ghost_plan=plan,
        bfl_fn=bfl_fn,
    )
    assert bool(torch.isfinite(octree.f_leaf).all())
    assert probe.link_force is not None and probe.shear_force is not None
    total = probe.total_force
    assert bool(torch.isfinite(total).all())
    assert abs(float(total[0])) > 0.0
    assert abs(float(ledger.mass_residual)) < 1e-3
