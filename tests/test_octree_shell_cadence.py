"""Shell cadence contract + ledger/CV impulse closure (wall-chain fix).

Wall-chain decomposition 2026-10-11, factor F2 (coverage 0.5) and the
ledger/CV normalization audit:

* ``step_octree_shell(host_substeps=R)`` advances ``R * 2^d_max`` leaf
  substeps per call — two per finer-level substep per level — so at
  ``d_max=1`` with the hybrid root-step call cadence (``R=2``) the shell
  integrates the full host time (4 substeps) instead of half of it;
  the default ``host_substeps=1`` keeps the legacy contract bit-identical.
* With the full-coverage cadence the ledger's per-window wall impulse
  (:meth:`ShellForceLedger.wall_impulse_l1`) must equal the CV-observed
  momentum transfer over the same window (the import-minus-change residual
  is the physical check that the wall term and the streaming import agree).
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.octree_boundary.bfl import bfl_apply_gather, leaf_force_weights
from tensorlbm.octree_boundary.force import (
    ShellForceLedger,
    build_shell_control_volume,
)
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.stepping import (
    build_ghost_plan,
    step_octree_shell,
)
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.static_block_amr import AMRAdvanceResult

U_IN = 0.05
TAU = 0.6


def _shell(host_shape=(32, 32, 32), radius=4.5, bl=2.0, d_max=1):
    nz, ny, nx = host_shape
    octree = build_octree_shell(
        host_shape,
        center=(nx / 2.0, ny / 2.0, nz / 2.0),
        radius=radius,
        bl_thickness_cells=bl,
        d_max=d_max,
        transition=1,
        lattice="D3Q19",
        device="cpu",
    )
    plan = build_ghost_plan(octree, host_shape)
    return octree, plan


def _seed_leaves(octree, host) -> None:
    hc = octree.leaf_host_cell
    octree.f_leaf = host[:, hc[:, 0], hc[:, 1], hc[:, 2]].clone()


def _host_equilibrium(shape):
    return equilibrium3d(
        torch.ones(shape),
        torch.full(shape, U_IN),
        torch.zeros(shape),
        torch.zeros(shape),
    )


def test_host_substeps_runs_two_substeps_per_host_substep() -> None:
    octree, plan = _shell()
    seen: list[int] = []
    bfl_calls: list[int] = []

    def advance(f, tau, level, substep):
        seen.append(substep)
        post = collide_bgk3d(f.view(19, 1, 1, -1), tau).view(19, -1)
        return AMRAdvanceResult(post.view_as(f).clone(), post.view_as(f))

    def bfl_fn(oct, out, post, gplan, ghost_vals, *, substep):
        bfl_calls.append(substep)
        f_out, force = bfl_apply_gather(
            oct,
            out,
            post,
            ghost_plan=gplan,
            ghost_vals=ghost_vals,
            force_weights=leaf_force_weights(oct),
            return_force=True,
        )
        return f_out, force

    host = _host_equilibrium((32, 32, 32))
    _seed_leaves(octree, host)
    # legacy contract: 2 substeps at d_max=1
    step_octree_shell(
        octree,
        advance,
        host,
        host.clone(),
        tau_coarse=TAU,
        reflux=False,
        ghost_plan=plan,
        bfl_fn=bfl_fn,
    )
    assert seen == [0, 1]
    assert bfl_calls == [0, 1]

    # full-coverage contract: host_substeps=2 -> 4 substeps (2 per L1 substep)
    seen.clear()
    bfl_calls.clear()
    ledger = ShellForceLedger(octree, host_substeps=2)
    step_octree_shell(
        octree,
        advance,
        host,
        host.clone(),
        tau_coarse=TAU,
        reflux=False,
        ghost_plan=plan,
        bfl_fn=bfl_fn,
        force_ledger=ledger,
        host_substeps=2,
    )
    assert seen == [0, 1, 2, 3]
    assert bfl_calls == [0, 1, 2, 3]
    assert ledger.substep_count == 4
    assert bool(torch.isfinite(ledger.mem_force).all())


def test_host_substeps_rejects_bad_values() -> None:
    octree, plan = _shell()

    def advance(f, tau, level, substep):
        post = collide_bgk3d(f.view(19, 1, 1, -1), tau).view(19, -1)
        return AMRAdvanceResult(post.view_as(f).clone(), post.view_as(f))

    host = _host_equilibrium((32, 32, 32))
    _seed_leaves(octree, host)
    for bad in (0, -1, 1.5):
        with pytest.raises((ValueError, TypeError)):
            step_octree_shell(
                octree,
                advance,
                host,
                host.clone(),
                tau_coarse=TAU,
                reflux=False,
                ghost_plan=plan,
                host_substeps=bad,
            )


@pytest.mark.slow
def test_ledger_impulse_equals_cv_impulse_full_coverage() -> None:
    """Ledger wall impulse == CV impulse over the same window (small case).

    Mini hybrid (host lattice only): sphere radius 4.5 on a 32^3 host, BGK
    collision, octree leaf shell with gather BFL, full-coverage cadence
    (host_substeps=2).  At quasi-steady state the per-window CV impulse
    must match the ledger's wall impulse (the import-minus-change residual
    stays at the few-percent level of the historical MEM/CV acceptance).
    """
    host_shape = (32, 32, 32)
    nz, ny, nx = host_shape
    radius = 4.5
    octree, plan = _shell(host_shape=host_shape, radius=radius, bl=2.0)
    cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0
    q = 19

    solid_g = torch.zeros((nz + 2, ny + 2, nx + 2), dtype=torch.bool)
    solid_g[1:-1, 1:-1, 1:-1] = octree._solid
    solid_q = octree._solid.unsqueeze(0).expand(q, *host_shape).contiguous()
    cv_w = build_shell_control_volume(
        (nz + 2, ny + 2, nx + 2),
        (cx + 1, cy + 1, cz + 1),
        radius,
        shell_band=octree.meta["delta_mask"],
        margin=1,
        covered=octree._shell_mask,
        solid=solid_g,
        device="cpu",
    )
    cv = cv_w[1:-1, 1:-1, 1:-1]

    host = equilibrium3d(
        torch.ones(host_shape),
        torch.full(host_shape, U_IN),
        torch.zeros(host_shape),
        torch.zeros(host_shape),
    )
    host_eq = host.clone()
    s_ = slice(None)
    border = (
        (s_, 0, s_, s_),
        (s_, -1, s_, s_),
        (s_, s_, 0, s_),
        (s_, s_, -1, s_),
        (s_, s_, s_, 0),
        (s_, s_, s_, -1),
    )
    _seed_leaves(octree, host)
    leaf_weights = leaf_force_weights(octree)

    def shell_advance(f, tau, level, substep):
        post = collide_bgk3d(f.view(q, 1, 1, -1), tau).view(q, -1)
        return AMRAdvanceResult(post.view_as(f).clone(), post.view_as(f))

    def bfl_fn(oct, out, post, gplan, ghost_vals, *, substep):
        f_out, force = bfl_apply_gather(
            oct,
            out,
            post,
            ghost_plan=gplan,
            ghost_vals=ghost_vals,
            force_weights=leaf_weights,
            return_force=True,
        )
        return f_out, force

    dx_leaf = 2.0 ** (-octree.d_max)
    dt_leaf = dx_leaf
    ledger = ShellForceLedger(octree, host_substeps=2)
    rel: list[float] = []
    steps, warmup = 90, 50
    for step in range(1, steps + 1):
        l1_old = host.clone()
        l1_posts = []
        for _sub in range(2):  # two host substeps per root step
            post = torch.where(solid_q, host, collide_bgk3d(host, TAU))
            l1_posts.append(post.clone())
            host = stream3d(post)
            for sl in border:
                host[sl] = host_eq[sl]
        l1_pre_shell = host.clone()
        step_octree_shell(
            octree,
            shell_advance,
            l1_old,
            host,
            tau_coarse=TAU,
            l1_post=l1_posts,
            shell_level=1,
            ghost_plan=plan,
            bfl_fn=bfl_fn,
            force_ledger=ledger,
            host_substeps=2,
        )
        ledger.observe_cv_force(
            l1_old,
            l1_pre_shell,
            l1_posts,
            cv,
            solid=octree._solid,
            wall_mom_l1=ledger.wall_impulse_l1(dx_leaf, dt_leaf),
        )
        if step > warmup:
            imp = float(ledger.wall_impulse_l1(dx_leaf, dt_leaf)[0])
            cvf = float(ledger.cv_force[0])
            rel.append(abs(imp - cvf) / max(abs(cvf), 1e-30))
        ledger.reset()

    mean_rel = sum(rel) / len(rel)
    assert mean_rel < 0.05, f"ledger/CV impulse mismatch: {mean_rel:.4%}"
