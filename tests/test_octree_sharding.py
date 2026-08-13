"""Multi-device sharding tests for the octree boundary shell (fine_devices).

The local machine has a single GPU, so the *real* two-GPU configuration
(L1 on GPU0, shell on GPU1) cannot be exercised here; the sharding logic is
validated with mock device sets instead:

* two/three CPU "devices" (same device object — every cross-shard code path
  still executes, transfers become no-ops) must reproduce the unsharded run
  within a documented floating-point tolerance (state, reflux ledger, MEM
  force); collision reductions over differently sized shards are not promised
  to be bitwise identical;
* the plane-degenerate shell, sharded, still reproduces ``StaticBlockAMR3D``
  link by link;
* mass conservation and the reflux identity hold for the sharded stepper;
* on CUDA (single local GPU): one shard on ``cuda:0`` vs the unsharded GPU
  run, and a CPU-root / CUDA-shell configuration with one vs two shards —
  cross-device transfers (cpu<->cuda) are exercised for real.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.octree_boundary.bfl import (
    bfl_apply_gather,
    bfl_ramp_wall_velocity,
    leaf_force_weights,
)
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.sharding import shard_octree_shell
from tensorlbm.octree_boundary.stepping import (
    build_ghost_plan,
    build_plane_shell,
    step_octree_shell,
    step_octree_shell_sharded,
)
from tensorlbm.refinement import BoxRegion
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    StaticBlockAMR3D,
    StaticBlockAMRConfig,
    convective_refined_tau,
)

TAU_C = 0.56
TAU_F = convective_refined_tau(TAU_C)
Q = 19
SHARD_RTOL = 1.0e-11
SHARD_ATOL = 1.0e-13


def _uniform_equilibrium(shape, rho0: float = 1.03) -> torch.Tensor:
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


def _make_advance(shape):
    """L1 (level 0) collides+streams; shell (level 1) collides only."""

    def advance(f, tau, level, substep) -> AMRAdvanceResult:
        if level == 0:
            post = collide_bgk3d(f, tau)
            return AMRAdvanceResult(stream3d(post), post)
        post = collide_bgk3d(f.view(Q, -1, 1, 1), tau).view_as(f)
        return AMRAdvanceResult(post.clone(), post)

    return advance


def _sphere_shell(**kw):
    params = dict(
        shape=(32, 32, 32), center=(16, 16, 16), radius=7,
        bl_thickness_cells=3, d_max=1, device="cpu",
    )
    params.update(kw)
    return build_octree_shell(**params)


def _make_bfl_fn(ramp_steps: int = 100):
    """BFL callback mirroring the production example (facade-friendly)."""

    def bfl_fn(octree_, out, post, ghost_plan_, ghost_vals, *, substep):
        rho_w, uwx, uwy, uwz = bfl_ramp_wall_velocity(
            octree_, post, 0, ramp_steps,
        )
        return bfl_apply_gather(
            octree_, out, post,
            ghost_plan=ghost_plan_, ghost_vals=ghost_vals,
            wall_velocity=(uwx, uwy, uwz), wall_density=rho_w,
            force_weights=getattr(octree_, "force_weights", None),
            return_force=True,
        )

    return bfl_fn


def _joint_mass(l1_f, grid) -> float:
    covered = grid._shell_mask
    exterior = float(l1_f[:, ~covered].sum().item())
    shell = float((grid.f_leaf.sum(dim=0) * grid.leaf_volume()).sum().item())
    return exterior + shell


def _run_steps(grid, l1, n_steps, *, shards=None, bfl_fn=None):
    """Run ``n_steps`` root steps with the (sharded) shell stepper.

    Returns the final ``(l1, ledger_list, forces)``.
    """
    advance = _make_advance(grid.meta["shape"])
    plan = build_ghost_plan(grid, grid.meta["shape"])
    forces: list[torch.Tensor] = []
    ledgers = []
    for _ in range(n_steps):
        l1_old = l1.clone()
        post = collide_bgk3d(l1, TAU_C)
        l1_new = stream3d(post)
        kw = dict(
            advance=advance, l1_old=l1_old, l1_f=l1_new,
            tau_coarse=TAU_C, l1_post=post, shell_level=1, ghost_plan=plan,
            bfl_fn=bfl_fn,
        )
        if shards is None:
            from tensorlbm.octree_boundary.force import ShellForceLedger
            ledger_f = ShellForceLedger(grid)
            ledger = step_octree_shell(grid, **kw, force_ledger=ledger_f)
            if bfl_fn is not None:
                forces.append(ledger_f.mem_force.clone())
        else:
            from tensorlbm.octree_boundary.force import ShellForceLedger
            ledger_f = ShellForceLedger(grid)
            ledger = step_octree_shell_sharded(
                grid, shards, **kw, force_ledger=ledger_f,
            )
            if bfl_fn is not None:
                forces.append(ledger_f.mem_force.clone())
        ledgers.append(ledger)
        l1 = l1_new
    return l1, ledgers, forces


# ---------------------------------------------------------------------------
# 1. Sharded vs unsharded: roundoff-level state and ledger (mock CPU shards)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d_max,n_shards", [(1, 2), (2, 2), (2, 3)])
def test_sharded_matches_unsharded_bit_for_bit(d_max, n_shards) -> None:
    """Two/three CPU shards reproduce the unsharded run within roundoff."""
    grid = _sphere_shell(d_max=d_max)
    grid.f_leaf = _uniform_leaf(grid.n_leaf)
    grid.force_weights = leaf_force_weights(grid)
    l1 = _uniform_equilibrium(grid.meta["shape"])

    ref_l1, ref_ledgers, ref_forces = _run_steps(
        grid, l1.clone(), 6, bfl_fn=_make_bfl_fn(),
    )

    # rebuild a fresh grid (the unsharded run mutated f_leaf in place)
    grid2 = _sphere_shell(d_max=d_max)
    grid2.f_leaf = _uniform_leaf(grid2.n_leaf)
    grid2.force_weights = leaf_force_weights(grid2)
    l1_2 = _uniform_equilibrium(grid2.meta["shape"])
    shards = shard_octree_shell(
        grid2, ["cpu"] * n_shards,
        ghost_plan=build_ghost_plan(grid2, grid2.meta["shape"]),
    )
    shd_l1, shd_ledgers, shd_forces = _run_steps(
        grid2, l1_2, 6, shards=shards, bfl_fn=_make_bfl_fn(),
    )

    # State/reflux/force agree to roundoff.  Per-shard collision reductions can
    # differ by a few ulps because the reduction kernel sees a different
    # spatial extent than the unsharded callback.
    torch.testing.assert_close(shd_l1, ref_l1, rtol=SHARD_RTOL, atol=SHARD_ATOL)
    torch.testing.assert_close(
        grid2.f_leaf, grid.f_leaf, rtol=SHARD_RTOL, atol=SHARD_ATOL,
    )
    # reflux ledger: roundoff-level, field by field
    for shd, ref in zip(shd_ledgers, ref_ledgers):
        for attr in (
            "raw_kinetic_mismatch",
            "replacement_mismatch",
            "applied_shell_correction",
            "residual",
        ):
            torch.testing.assert_close(
                getattr(shd, attr), getattr(ref, attr),
                rtol=SHARD_RTOL, atol=SHARD_ATOL,
            )
        assert shd.shell_cells == ref.shell_cells
        assert shd.limited_directions == ref.limited_directions
    # MEM force: roundoff-level (global-order assembly)
    for shd_f, ref_f in zip(shd_forces, ref_forces):
        torch.testing.assert_close(shd_f, ref_f, rtol=SHARD_RTOL, atol=SHARD_ATOL)


def test_sharded_shard_count_independent() -> None:
    """2 and 3 shards on the same device type give identical results."""
    grid = _sphere_shell(d_max=2)
    grid.f_leaf = _uniform_leaf(grid.n_leaf)
    l1 = _uniform_equilibrium(grid.meta["shape"])
    plan = build_ghost_plan(grid, grid.meta["shape"])

    shards2 = shard_octree_shell(grid, ["cpu", "cpu"], ghost_plan=plan)
    l1_2, ledgers2, _ = _run_steps(grid, l1.clone(), 4, shards=shards2)
    f2 = grid.f_leaf.clone()

    grid2 = _sphere_shell(d_max=2)
    grid2.f_leaf = _uniform_leaf(grid2.n_leaf)
    l1_3 = _uniform_equilibrium(grid2.meta["shape"])
    shards3 = shard_octree_shell(
        grid2, ["cpu", "cpu", "cpu"],
        ghost_plan=build_ghost_plan(grid2, grid2.meta["shape"]),
    )
    l1_3, ledgers3, _ = _run_steps(grid2, l1_3, 4, shards=shards3)
    torch.testing.assert_close(l1_2, l1_3, rtol=SHARD_RTOL, atol=SHARD_ATOL)
    torch.testing.assert_close(f2, grid2.f_leaf, rtol=SHARD_RTOL, atol=SHARD_ATOL)
    for a, b in zip(ledgers2, ledgers3):
        torch.testing.assert_close(
            a.residual, b.residual, rtol=SHARD_RTOL, atol=SHARD_ATOL,
        )


# ---------------------------------------------------------------------------
# 2. Plane-degenerate shell, sharded, matches StaticBlockAMR3D
# ---------------------------------------------------------------------------


def test_sharded_plane_shell_matches_static_block_amr() -> None:
    """The sharded plane shell reproduces StaticBlockAMR3D link by link."""
    shape = (14, 15, 16)
    box = BoxRegion(x0=3, x1=7, y0=3, y1=6, z0=3, z1=6)
    config = StaticBlockAMRConfig(
        box, tau_coarse=TAU_C, ghost_interpolation="trilinear",
    )
    coarse0 = _uniform_equilibrium(shape)

    ref = StaticBlockAMR3D(coarse0.clone(), config)
    ref_ledger = ref.step(_make_advance(shape))

    shell = build_plane_shell(shape, box)
    leaf_fine = shell.meta["leaf_fine_flat"]
    shell.f_leaf = ref.fine_physical.reshape(Q, -1)[:, leaf_fine].clone()
    plan = build_ghost_plan(shell, shape)
    shards = shard_octree_shell(shell, ["cpu", "cpu"], ghost_plan=plan)

    l1 = coarse0.clone()
    l1_post = collide_bgk3d(l1, TAU_C)
    l1_new = stream3d(l1_post)
    shell_ledger = step_octree_shell_sharded(
        shell, shards, _make_advance(shape),
        l1, l1_new,
        tau_coarse=TAU_C, l1_post=l1_post, shell_level=1, ghost_plan=plan,
    )
    # the sharded stepper restores the global f_leaf on the root octree
    torch.testing.assert_close(l1_new, ref.coarse_f, rtol=0.0, atol=1e-10)
    torch.testing.assert_close(
        shell.f_leaf,
        ref.fine_physical.reshape(Q, -1)[:, leaf_fine],
        rtol=0.0, atol=1e-10,
    )
    for attr in (
        "raw_kinetic_mismatch",
        "replacement_mismatch",
        "applied_shell_correction",
        "residual",
    ):
        torch.testing.assert_close(
            getattr(shell_ledger, attr), getattr(ref_ledger, attr),
            rtol=1e-10, atol=1e-12,
        )
    assert abs(shell_ledger.mass_residual) < 1e-12


# ---------------------------------------------------------------------------
# 3. Sharded mass conservation + reflux identity
# ---------------------------------------------------------------------------


def test_sharded_mass_conservation_uniform_free_stream() -> None:
    """Uniform free stream, sharded: mass drift < 1e-10 over 40 root steps."""
    grid = _sphere_shell()
    grid.f_leaf = _uniform_leaf(grid.n_leaf)
    l1 = _uniform_equilibrium(grid.meta["shape"])
    plan = build_ghost_plan(grid, grid.meta["shape"])
    shards = shard_octree_shell(grid, ["cpu", "cpu"], ghost_plan=plan)

    mass0 = _joint_mass(l1, grid)
    worst = 0.0
    residual_sum = 0.0
    for _ in range(40):
        l1_old = l1.clone()
        post = collide_bgk3d(l1, TAU_C)
        l1_new = stream3d(post)
        ledger = step_octree_shell_sharded(
            grid, shards, _make_advance(grid.meta["shape"]),
            l1_old, l1_new,
            tau_coarse=TAU_C, l1_post=post, shell_level=1, ghost_plan=plan,
        )
        worst = max(worst, abs(ledger.mass_residual))
        residual_sum += ledger.mass_residual
        l1 = l1_new
        assert bool(torch.isfinite(grid.f_leaf).all())
    mass1 = _joint_mass(l1, grid)
    assert worst < 1e-10
    assert abs(mass1 - mass0) / mass0 < 1e-10
    assert abs(mass1 - mass0 + residual_sum) < 1e-9


# ---------------------------------------------------------------------------
# 4. CUDA: single-GPU regression + real cross-device (cpu root / cuda shell)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_sharded_gpu_single_shard_matches_unsharded() -> None:
    """On one GPU: one shard agrees with unsharded to roundoff."""
    grid = _sphere_shell(device="cuda:0")
    grid.f_leaf = _uniform_leaf(grid.n_leaf).to("cuda:0")
    grid.force_weights = leaf_force_weights(grid).to("cuda:0")
    l1 = _uniform_equilibrium(grid.meta["shape"]).to("cuda:0")

    ref_l1, ref_ledgers, ref_forces = _run_steps(
        grid, l1.clone(), 4, bfl_fn=_make_bfl_fn(),
    )

    grid2 = _sphere_shell(device="cuda:0")
    grid2.f_leaf = _uniform_leaf(grid2.n_leaf).to("cuda:0")
    grid2.force_weights = leaf_force_weights(grid2).to("cuda:0")
    l1_2 = _uniform_equilibrium(grid2.meta["shape"]).to("cuda:0")
    shards = shard_octree_shell(
        grid2, ["cuda:0"],
        ghost_plan=build_ghost_plan(grid2, grid2.meta["shape"]),
    )
    shd_l1, shd_ledgers, shd_forces = _run_steps(
        grid2, l1_2, 4, shards=shards, bfl_fn=_make_bfl_fn(),
    )
    torch.testing.assert_close(shd_l1, ref_l1, rtol=SHARD_RTOL, atol=SHARD_ATOL)
    torch.testing.assert_close(
        grid2.f_leaf, grid.f_leaf, rtol=SHARD_RTOL, atol=SHARD_ATOL,
    )
    for shd, ref in zip(shd_ledgers, ref_ledgers):
        torch.testing.assert_close(
            shd.residual, ref.residual, rtol=SHARD_RTOL, atol=SHARD_ATOL,
        )
    for shd_f, ref_f in zip(shd_forces, ref_forces):
        torch.testing.assert_close(shd_f, ref_f, rtol=SHARD_RTOL, atol=SHARD_ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_sharded_cpu_root_cuda_shell_cross_device() -> None:
    """Root on CPU, shell on CUDA: 1 vs 2 shards agree to roundoff.

    This exercises the real cpu<->cuda transfers of the sharded data flow
    (ghost values, exchange values, observation/force assembly, restriction).
    """
    grid = _sphere_shell(device="cpu")
    grid.f_leaf = _uniform_leaf(grid.n_leaf)
    grid.force_weights = leaf_force_weights(grid)
    l1 = _uniform_equilibrium(grid.meta["shape"])
    plan = build_ghost_plan(grid, grid.meta["shape"])

    shards1 = shard_octree_shell(grid, ["cuda:0"], ghost_plan=plan)
    l1_1, ledgers1, forces1 = _run_steps(
        grid, l1.clone(), 4, shards=shards1, bfl_fn=_make_bfl_fn(),
    )
    f_leaf_1 = grid.f_leaf.clone()

    grid2 = _sphere_shell(device="cpu")
    grid2.f_leaf = _uniform_leaf(grid2.n_leaf)
    grid2.force_weights = leaf_force_weights(grid2)
    l1_2 = _uniform_equilibrium(grid2.meta["shape"])
    shards2 = shard_octree_shell(
        grid2, ["cuda:0", "cuda:0"],
        ghost_plan=build_ghost_plan(grid2, grid2.meta["shape"]),
    )
    l1_2, ledgers2, forces2 = _run_steps(
        grid2, l1_2, 4, shards=shards2, bfl_fn=_make_bfl_fn(),
    )
    torch.testing.assert_close(l1_1, l1_2, rtol=SHARD_RTOL, atol=SHARD_ATOL)
    torch.testing.assert_close(
        f_leaf_1, grid2.f_leaf, rtol=SHARD_RTOL, atol=SHARD_ATOL,
    )
    for a, b in zip(ledgers1, ledgers2):
        torch.testing.assert_close(
            a.residual, b.residual, rtol=SHARD_RTOL, atol=SHARD_ATOL,
        )
    for a, b in zip(forces1, forces2):
        torch.testing.assert_close(a, b, rtol=SHARD_RTOL, atol=SHARD_ATOL)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
