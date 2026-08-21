"""P2 octree shell stepping tests (design doc octree-boundary-design.md §4).

Covers the four P2 acceptance items:

* substep scheduling — the shell walks ``2^d`` substeps per L1 root step;
* mass conservation — sphere shell + L1 block, uniform free stream, 100
  steps: relative mass drift < 1e-6;
* reflux residual — smooth density pulse crossing the shell interface:
  per-step ``|ledger.mass_residual| < 1e-10`` and the joint mass drift is
  bounded by the accumulated residual (``dM_step = -residual.sum()``);
* plane special case — the shell degenerated to a rectangular box reproduces
  ``StaticBlockAMR3D`` link by link (per-direction link counts, per-direction
  fine transfers, reflux ledger fields and final states).
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.kinetic_flux_register import (
    apply_face_local_reflux,
    observe_kinetic_interface_transfer,
)
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.stepping import (
    build_ghost_plan,
    build_plane_shell,
    step_octree_shell,
)
from tensorlbm.refinement import BoxRegion
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    PopulationRefluxLedger,
    StaticBlockAMR3D,
    StaticBlockAMRConfig,
    convective_refined_tau,
)

TAU_C = 0.56
TAU_F = convective_refined_tau(TAU_C)
Q = 19


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _uniform_equilibrium(shape: tuple[int, int, int], rho0: float = 1.03) -> torch.Tensor:
    rho = torch.full(shape, rho0, dtype=torch.float64)
    ux = torch.full(shape, 0.031, dtype=torch.float64)
    uy = torch.full(shape, -0.012, dtype=torch.float64)
    uz = torch.full(shape, 0.007, dtype=torch.float64)
    return equilibrium3d(rho, ux, uy, uz)


def _uniform_leaf(n_leaf: int) -> torch.Tensor:
    return equilibrium3d(
        torch.full((1, 1, n_leaf), 1.03, dtype=torch.float64),
        torch.full((1, 1, n_leaf), 0.031, dtype=torch.float64),
        torch.full((1, 1, n_leaf), -0.012, dtype=torch.float64),
        torch.full((1, 1, n_leaf), 0.007, dtype=torch.float64),
    ).view(Q, n_leaf)


def _make_advance(shape: tuple[int, int, int], *, shell_collide_only: bool):
    """Advance3D callback: L1 (level 0) always collides+streams; the shell
    level (1) collides only (the stepper streams through the neighbour table)
    unless ``shell_collide_only`` is False (StaticBlockAMR3D reference path)."""

    def advance(f: torch.Tensor, tau: float, level: int, substep: int) -> AMRAdvanceResult:
        if level == 0:
            post = collide_bgk3d(f, tau)
            return AMRAdvanceResult(stream3d(post), post)
        if shell_collide_only:
            post = collide_bgk3d(f.view(Q, -1, 1, 1), tau).view_as(f)
            return AMRAdvanceResult(post.clone(), post)
        post = collide_bgk3d(f, tau)
        return AMRAdvanceResult(stream3d(post), post)

    return advance


def _sphere_shell(**kw):
    params = dict(
        shape=(32, 32, 32),
        center=(16, 16, 16),
        radius=7,
        bl_thickness_cells=3,
        d_max=1,
        device="cpu",
    )
    params.update(kw)
    return build_octree_shell(**params)


def _joint_mass(l1_f: torch.Tensor, grid) -> float:
    """Volume-integrated mass of the L1-exterior + shell system."""
    covered = grid._shell_mask
    exterior = float(l1_f[:, ~covered].sum().item())
    shell = float((grid.f_leaf.sum(dim=0) * grid.leaf_volume()).sum().item())
    return exterior + shell


def _leaf_centers64(grid) -> torch.Tensor:
    """Exact float64 leaf centres in (x, y, z) world coordinates."""
    if grid._l2_coords is not None and grid._l2_coords.numel() > 0:
        coords = torch.cat((grid._l1_coords, grid._l2_coords), dim=0)
    else:
        coords = grid._l1_coords
    return (coords.to(torch.float64) + 0.5) / (2.0 ** grid.leaf_level.to(torch.float64))[:, None]


def _pulse_equilibrium(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    pulse_center: tuple[float, float, float],
    rho0: float = 1.03,
    amp: float = 0.02,
    sigma: float = 2.5,
    ux0: float = 0.01,
) -> torch.Tensor:
    nz, ny, nx = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64),
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    r2 = (xx - pulse_center[0]) ** 2 + (yy - pulse_center[1]) ** 2 + (zz - pulse_center[2]) ** 2
    rho = rho0 + amp * torch.exp(-r2 / (2.0 * sigma * sigma))
    return equilibrium3d(
        rho,
        torch.full_like(rho, ux0),
        torch.zeros_like(rho),
        torch.zeros_like(rho),
    )


def _pulse_leaf(
    grid,
    pulse_center: tuple[float, float, float],
    rho0: float = 1.03,
    amp: float = 0.02,
    sigma: float = 2.5,
    ux0: float = 0.01,
) -> torch.Tensor:
    centers = _leaf_centers64(grid)
    r2 = (
        (centers[:, 0] - pulse_center[0]) ** 2
        + (centers[:, 1] - pulse_center[1]) ** 2
        + (centers[:, 2] - pulse_center[2]) ** 2
    )
    rho = rho0 + amp * torch.exp(-r2 / (2.0 * sigma * sigma))
    return equilibrium3d(
        rho.view(1, 1, -1),
        torch.full((1, 1, grid.n_leaf), ux0, dtype=torch.float64),
        torch.zeros((1, 1, grid.n_leaf), dtype=torch.float64),
        torch.zeros((1, 1, grid.n_leaf), dtype=torch.float64),
    ).view(Q, grid.n_leaf)


# ---------------------------------------------------------------------------
# 1. Substeps scheduling
# ---------------------------------------------------------------------------


def test_substep_scheduling_per_root_step() -> None:
    """The shell walks 2^d substeps per L1 root step with the right indices."""
    for d_max, expected in ((1, 2), (2, 4)):
        grid = _sphere_shell(radius=6, bl_thickness_cells=2, d_max=d_max)
        grid.f_leaf = _uniform_leaf(grid.n_leaf)
        l1 = _uniform_equilibrium(grid.meta["shape"])
        calls: list[tuple[int, int, float]] = []

        def advance(f, tau, level, substep):
            calls.append((level, substep, tau))
            post = collide_bgk3d(f.view(Q, -1, 1, 1), tau).view_as(f)
            return AMRAdvanceResult(post.clone(), post)

        post = collide_bgk3d(l1, TAU_C)
        l1_new = stream3d(post)
        step_octree_shell(
            grid,
            advance,
            l1.clone(),
            l1_new,
            tau_coarse=TAU_C,
            l1_post=post,
            shell_level=1,
        )
        assert [(level, s) for level, s, _ in calls] == [(1, s) for s in range(expected)]
        assert all(abs(tau - TAU_F) < 1e-12 for _, _, tau in calls)
        assert bool(torch.isfinite(grid.f_leaf).all())
        assert bool(torch.isfinite(l1_new).all())


def test_substep_scheduling_two_root_steps() -> None:
    """Two L1 root steps drive 2 * 2^d shell advances with a repeated pattern."""
    grid = _sphere_shell(radius=6, bl_thickness_cells=2, d_max=1)
    grid.f_leaf = _uniform_leaf(grid.n_leaf)
    l1 = _uniform_equilibrium(grid.meta["shape"])
    calls: list[int] = []

    def advance(f, tau, level, substep):
        calls.append(substep)
        post = collide_bgk3d(f.view(Q, -1, 1, 1), tau).view_as(f)
        return AMRAdvanceResult(post.clone(), post)

    plan = build_ghost_plan(grid, grid.meta["shape"])
    for _ in range(2):
        post = collide_bgk3d(l1, TAU_C)
        l1_new = stream3d(post)
        step_octree_shell(
            grid,
            advance,
            l1.clone(),
            l1_new,
            tau_coarse=TAU_C,
            l1_post=post,
            shell_level=1,
            ghost_plan=plan,
        )
        l1 = l1_new
    assert calls == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# 2. Mass conservation — uniform free stream, 100 steps
# ---------------------------------------------------------------------------


def test_mass_conservation_uniform_free_stream() -> None:
    """Uniform free stream across the shell interface: relative mass drift
    < 1e-6 over 100 L1 steps and reflux residual < 1e-10 every step."""
    grid = _sphere_shell()
    grid.f_leaf = _uniform_leaf(grid.n_leaf)
    l1 = _uniform_equilibrium(grid.meta["shape"])
    plan = build_ghost_plan(grid, grid.meta["shape"])
    advance = _make_advance(grid.meta["shape"], shell_collide_only=True)

    mass0 = _joint_mass(l1, grid)
    worst_residual = 0.0
    for _ in range(100):
        l1_old = l1.clone()
        post = collide_bgk3d(l1, TAU_C)
        l1_new = stream3d(post)
        ledger = step_octree_shell(
            grid,
            advance,
            l1_old,
            l1_new,
            tau_coarse=TAU_C,
            l1_post=post,
            shell_level=1,
            ghost_plan=plan,
        )
        worst_residual = max(worst_residual, abs(ledger.mass_residual))
        l1 = l1_new
        assert bool(torch.isfinite(grid.f_leaf).all())
        assert float(grid.f_leaf.min()) > 0.0

    mass1 = _joint_mass(l1, grid)
    assert worst_residual < 1e-10
    assert abs(mass1 - mass0) / mass0 < 1e-6
    assert abs(mass1 - mass0) / mass0 < 1e-10  # float64: roundoff-only drift


# ---------------------------------------------------------------------------
# 3. Reflux residual — smooth density pulse crossing the interface
# ---------------------------------------------------------------------------


def test_reflux_residual_smooth_pulse() -> None:
    """A smooth density pulse crossing the shell interface: per-step reflux
    residual < 1e-10 and the joint mass drift equals -sum(residuals)."""
    grid = _sphere_shell()
    pulse_center = (25.0, 16.0, 16.0)  # inside the shell band on the +x side
    l1 = _pulse_equilibrium(
        grid.meta["shape"],
        (16.0, 16.0, 16.0),
        pulse_center,
    )
    grid.f_leaf = _pulse_leaf(grid, pulse_center)
    plan = build_ghost_plan(grid, grid.meta["shape"])
    advance = _make_advance(grid.meta["shape"], shell_collide_only=True)

    mass0 = _joint_mass(l1, grid)
    residual_sum = 0.0
    worst = 0.0
    mass_prev = mass0
    for _ in range(20):
        l1_old = l1.clone()
        post = collide_bgk3d(l1, TAU_C)
        l1_new = stream3d(post)
        ledger = step_octree_shell(
            grid,
            advance,
            l1_old,
            l1_new,
            tau_coarse=TAU_C,
            l1_post=post,
            shell_level=1,
            ghost_plan=plan,
        )
        residual_sum += ledger.mass_residual
        worst = max(worst, abs(ledger.mass_residual))
        # mass-conservation identity: dM_step = -residual.sum() (the drift is
        # bounded by the reflux residual by construction)
        mass_step = _joint_mass(l1_new, grid)
        assert abs((mass_step - mass_prev) + ledger.mass_residual) < 1e-9
        mass_prev = mass_step
        l1 = l1_new

    mass1 = _joint_mass(l1, grid)
    assert worst < 1e-10
    assert abs(mass1 - mass0) / mass0 < 1e-6
    assert abs(mass1 - mass0 + residual_sum) < 1e-9


# ---------------------------------------------------------------------------
# 4. Plane special case — link-by-link consistency with StaticBlockAMR3D
# ---------------------------------------------------------------------------


def _run_block_step_captured(solver: StaticBlockAMR3D):
    """Replicate ``StaticBlockAMR3D.step`` capturing the interface transfers
    (the official ``step`` does not expose them)."""
    advance = _make_advance(solver.coarse_f.shape[1:], shell_collide_only=False)
    config = solver.config
    tau_c, tau_f = config.tau_coarse, config.tau_fine
    coarse_old = solver.coarse_f.clone()
    coarse_new, coarse_post = solver._unpack_advance(
        advance(solver.coarse_f, tau_c, 0, -1),
        solver.coarse_f.shape,
        require_flux_state=True,
    )
    solver.coarse_f = coarse_new
    coarse_transfer = observe_kinetic_interface_transfer(
        coarse_post,
        solver.coarse_interface_links,
    )
    fine_transfer = None
    for substep in range(config.ratio):
        alpha = substep / config.ratio
        solver._fill_ghost(
            torch.lerp(coarse_old, solver.coarse_f, alpha),
            tau_source=tau_c,
            tau_target=tau_f,
        )
        fine_new, fine_post = solver._unpack_advance(
            advance(solver.fine_f, tau_f, 1, substep),
            solver.fine_f.shape,
            require_flux_state=True,
        )
        observed = observe_kinetic_interface_transfer(
            fine_post,
            solver.fine_interface_links,
            cell_volume=1.0 / config.ratio**3,
        )
        fine_transfer = observed if fine_transfer is None else fine_transfer + observed
        solver.fine_f = fine_new
    restricted = solver._restrict_physical(tau_source=tau_f, tau_target=tau_c)
    box = config.box
    solver.coarse_f[:, box.z0 : box.z1, box.y0 : box.y1, box.x0 : box.x1] = restricted
    solver.coarse_f, report = apply_face_local_reflux(
        solver.coarse_f,
        solver.coarse_interface_links,
        coarse_transfer,
        fine_transfer,
        maximum_correction_fraction=config.maximum_reflux_correction_fraction,
        correction_stencil=config.reflux_correction_stencil,
    )
    ledger = PopulationRefluxLedger(
        report.requested_inventory_correction,
        report.applied_inventory_correction,
        report.corrected_links,
        report.residual,
        report.limited_directions,
        report.raw_kinetic_mismatch,
        0.0,
        1.0,
        0.0,
        1.0,
        report.maximum_applied_correction_fraction,
    )
    return coarse_transfer, fine_transfer, ledger


def test_plane_shell_matches_static_block_amr() -> None:
    """The plane-degenerate shell reproduces StaticBlockAMR3D link by link."""
    shape = (14, 15, 16)
    box = BoxRegion(x0=3, x1=7, y0=3, y1=6, z0=3, z1=6)
    config = StaticBlockAMRConfig(box, tau_coarse=TAU_C, ghost_interpolation="trilinear")
    coarse0 = _uniform_equilibrium(shape)

    # ---- official reference path ------------------------------------------
    ref = StaticBlockAMR3D(coarse0.clone(), config)
    ref_ledger = ref.step(_make_advance(shape, shell_collide_only=False))

    # ---- captured reference (transfers + ledger) ---------------------------
    cap = StaticBlockAMR3D(coarse0.clone(), config)
    fine_flat = cap.fine_physical.reshape(Q, -1)  # (Q, n_cells) initial values
    # (leaf_fine_flat mapping is built on the shell below)

    # ---- shell path --------------------------------------------------------
    shell = build_plane_shell(shape, box)
    assert shell.checks["symmetry"]["symmetric"] is True
    assert shell.checks["balance_21"]["balanced_21"] is True
    assert shell.checks["interface_links"]["complete"] is True
    leaf_fine = shell.meta["leaf_fine_flat"]
    shell.f_leaf = fine_flat[:, leaf_fine].clone()
    plan = build_ghost_plan(shell, shape)
    l1 = coarse0.clone()
    l1_post = collide_bgk3d(l1, TAU_C)
    l1_new = stream3d(l1_post)
    shell_ledger = step_octree_shell(
        shell,
        _make_advance(shape, shell_collide_only=True),
        l1,
        l1_new,
        tau_coarse=TAU_C,
        l1_post=l1_post,
        shell_level=1,
        ghost_plan=plan,
    )
    shell_fine_transfer = shell.meta["last_fine_transfer"]

    # ---- captured reference run (same initial fine state as the shell) -----
    # ``fine_flat`` is a (Q, n_cells) flattening of the (Q, nz, ny, nx)
    # physical fine block; copy back through the matching 4-D view.
    cap.fine_physical.copy_(fine_flat.view(cap.fine_physical.shape))
    ct_cap, ft_cap, cap_ledger = _run_block_step_captured(cap)

    # ---- per-direction link counts (the registry itself) -------------------
    for d in range(1, Q):
        n_out_shell = int((shell.interface_links[:, 1] == d).sum().item())
        n_out_ref = int(cap.fine_interface_links.outgoing_origins[d].sum().item())
        n_in_shell = int((plan.direction == d).sum().item())
        n_in_ref = int(cap.fine_interface_links.incoming_origins[d].sum().item())
        assert n_out_shell == n_out_ref, f"outgoing link count d={d}"
        assert n_in_shell == n_in_ref, f"incoming link count d={d}"

    # ---- per-direction fine transfers --------------------------------------
    torch.testing.assert_close(
        shell_fine_transfer.outgoing,
        ft_cap.outgoing,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        shell_fine_transfer.incoming,
        ft_cap.incoming,
        rtol=1e-12,
        atol=1e-12,
    )
    # coarse transfers come from identical inputs/links by construction
    torch.testing.assert_close(
        shell.meta["last_fine_transfer"].net_outgoing,
        (ft_cap.net_outgoing),
        rtol=1e-12,
        atol=1e-12,
    )

    # ---- reflux ledger, field by field -------------------------------------
    for attr in (
        "raw_kinetic_mismatch",
        "replacement_mismatch",
        "applied_shell_correction",
        "residual",
    ):
        torch.testing.assert_close(
            getattr(shell_ledger, attr),
            getattr(ref_ledger, attr),
            rtol=1e-10,
            atol=1e-12,
        )
    assert shell_ledger.shell_cells == ref_ledger.shell_cells
    assert shell_ledger.limited_directions == ref_ledger.limited_directions
    assert shell_ledger.shell_cells == cap_ledger.shell_cells
    assert abs(shell_ledger.mass_residual) < 1e-12

    # ---- final states -------------------------------------------------------
    torch.testing.assert_close(l1_new, ref.coarse_f, rtol=0.0, atol=1e-10)
    torch.testing.assert_close(
        shell.f_leaf,
        ref.fine_physical.reshape(Q, -1)[:, leaf_fine],
        rtol=0.0,
        atol=1e-10,
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
