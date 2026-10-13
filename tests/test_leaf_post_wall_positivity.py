"""Leaf post-wall positivity pass (v3.3 symmetry repair) — unit contract.

Layer-2 fix validation (adjudication 2026-10-12,
controller_adjudication_layer2.md): every BLOCK advance gets a second
(post-wall) positivity pass after its last field write of the substep;
the legacy leaf loop applied the limiter only inside ``advance``
(pre-stream), so the ``bfl_fn`` boundary-population write — an unclamped
linear Bouzidi combination minus an additive moving-wall term — could
commit negative populations after the only limiter pass.

``step_octree_shell(leaf_post_wall_positivity=True)`` closes the
asymmetry: same tool (``limit_nonequilibrium_for_positivity``), same
floor (1e-9), applied per substep after the ``bfl_fn`` block,
immediately before the ``octree.f_leaf = out`` commit.  These tests pin:

* a below-floor leaf (rho > 0, some f_i < 0) written by ``bfl_fn`` is
  clamped to the floor with rho/momentum preserved to fp roundoff, and
  only the damaged cells change;
* a rho < 0 leaf written by ``bfl_fn`` degrades to the mass-adding floor
  clamp and is disclosed by the ``rho_nonpositive_count`` diagnostic;
* the pass also runs with ``bfl_fn=None`` (blocks get the post-stream
  pass unconditionally);
* default-off is bit-identical to the frozen wt_highre_v32 @ dbb01b5e
  trajectory on a real 10-step wall-chain case (``torch.equal``);
* the diagnostics accumulator carries one entry per leaf substep.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.octree_boundary.bfl import bfl_apply_gather, leaf_force_weights
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.stepping import (
    build_ghost_plan,
    step_octree_shell,
)
from tensorlbm.solver3d import collide_bgk3d
from tensorlbm.static_block_amr import AMRAdvanceResult

U_IN = 0.05
TAU = 0.6
FLOOR = 1e-9  # limit_nonequilibrium_for_positivity default floor
# fp32 representation of the floor: float32(1e-9) is slightly below the
# double literal, and the limiter clamps in the field dtype
FLOOR_FP32 = float(torch.tensor(FLOOR, dtype=torch.float32).item())
V32_TREE = "/nfs/wangxi/wt_highre_v32"
V32_SRC = os.path.join(V32_TREE, "src")
V32_BASE_SHA = "dbb01b5e6b006a6a2af5b036fbd65529d3bd587a"  # wt_highre_v32 HEAD


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


def _host_equilibrium(shape):
    return equilibrium3d(
        torch.ones(shape),
        torch.full(shape, U_IN),
        torch.zeros(shape),
        torch.zeros(shape),
    )


def _seed_leaves(octree, host) -> None:
    hc = octree.leaf_host_cell
    octree.f_leaf = host[:, hc[:, 0], hc[:, 1], hc[:, 2]].clone()


def _plain_advance(f, tau, level, substep):
    post = collide_bgk3d(f.view(19, 1, 1, -1), tau).view(19, -1)
    return AMRAdvanceResult(post.view_as(f).clone(), post.view_as(f))


def _moments(f_cell: torch.Tensor):
    rho = float(f_cell.sum().item())
    momentum = torch.matmul(C.to(f_cell.dtype).T, f_cell.reshape(-1, 1)).reshape(-1)
    return rho, [float(v) for v in momentum.tolist()]


def test_below_floor_leaf_clamped_moments_preserved() -> None:
    """bfl_fn writes a below-floor leaf (rho>0) -> floor clamp, moments kept."""
    octree, plan = _shell()
    host = _host_equilibrium((32, 32, 32))
    _seed_leaves(octree, host)
    target = min(7, octree.n_leaf - 1)
    damaged_out: dict[str, torch.Tensor] = {}

    def bfl_fn(oct, out, post, gplan, ghost_vals, *, substep):
        f_out = out.clone()
        # synthetic unclamped Bouzidi-style write: negative components
        # injected into fluid leaves AFTER the only in-advance limiter pass
        f_out[1, target] = -0.05
        f_out[3, target] = -0.02
        damaged_out[str(substep)] = f_out.clone()
        return f_out, None

    acc: list[dict[str, float]] = []
    step_octree_shell(
        octree,
        _plain_advance,
        host,
        host.clone(),
        tau_coarse=TAU,
        reflux=False,
        ghost_plan=plan,
        bfl_fn=bfl_fn,
        leaf_post_wall_positivity=True,
        leaf_post_wall_diagnostics=acc,
    )
    final = octree.f_leaf
    assert float(final.min().item()) >= FLOOR_FP32
    # only the damaged leaf changed (all other leaves bit-identical)
    last = damaged_out[str(1)] if "1" in damaged_out else damaged_out["0"]
    others = torch.ones(octree.n_leaf, dtype=torch.bool)
    others[target] = False
    assert torch.equal(final[:, others], last[:, others])
    # rho / momentum preserved on the clamped cell to fp roundoff
    rho_before, mom_before = _moments(last[:, target])
    rho_after, mom_after = _moments(final[:, target])
    assert rho_before > 0.0
    assert rho_after == pytest.approx(rho_before, rel=1e-5, abs=1e-6)
    for before, after in zip(mom_before, mom_after):
        assert after == pytest.approx(before, rel=1e-4, abs=1e-6)
    # diagnostics: one entry per leaf substep (d_max=1, host_substeps=1 -> 2)
    assert len(acc) == 2
    assert all(entry["limited_fraction"] > 0.0 for entry in acc)
    assert all(entry["rho_nonpositive_count"] == 0 for entry in acc)


def test_rho_negative_leaf_floored_and_counted() -> None:
    """A rho<0 leaf degrades to the mass-adding floor clamp, counter fires."""
    octree, plan = _shell()
    host = _host_equilibrium((32, 32, 32))
    _seed_leaves(octree, host)
    target = min(11, octree.n_leaf - 1)

    def bfl_fn(oct, out, post, gplan, ghost_vals, *, substep):
        f_out = out.clone()
        f_out[:, target] = -0.01  # every component negative -> rho = -0.19
        return f_out, None

    acc: list[dict[str, float]] = []
    step_octree_shell(
        octree,
        _plain_advance,
        host,
        host.clone(),
        tau_coarse=TAU,
        reflux=False,
        ghost_plan=plan,
        bfl_fn=bfl_fn,
        leaf_post_wall_positivity=True,
        leaf_post_wall_diagnostics=acc,
    )
    final = octree.f_leaf
    assert float(final.min().item()) >= FLOOR_FP32
    rho_after = float(final[:, target].sum().item())
    assert rho_after == pytest.approx(19.0 * FLOOR, rel=1e-3)
    assert rho_after > 0.0  # mass was forced in (disclosed degradation)
    assert len(acc) == 2
    assert all(entry["rho_nonpositive_count"] == 1 for entry in acc)
    assert all(entry["limited_fraction"] > 0.0 for entry in acc)


def test_pass_applies_with_bfl_fn_none() -> None:
    """Blocks get the post-stream pass unconditionally — so must the leaf."""
    for enabled in (False, True):
        octree, plan = _shell()
        host = _host_equilibrium((32, 32, 32))
        _seed_leaves(octree, host)

        def advance(f, tau, level, substep):
            post = collide_bgk3d(f.view(19, 1, 1, -1), tau).view(19, -1)
            if substep == 1:
                # damaged populations stream into the gathered leaf field
                # (equilibrium component ~0.064 at u=0.05, so -0.2 drives it
                # well below the floor)
                post = post.clone()
                post[1, :] = post[1, :] - 0.2
            return AMRAdvanceResult(post.view_as(f).clone(), post.view_as(f))

        step_octree_shell(
            octree,
            advance,
            host,
            host.clone(),
            tau_coarse=TAU,
            reflux=False,
            ghost_plan=plan,
            bfl_fn=None,
            leaf_post_wall_positivity=enabled,
        )
        if enabled:
            assert float(octree.f_leaf.min().item()) >= FLOOR_FP32
        else:
            assert float(octree.f_leaf.min().item()) < FLOOR


def _trajectory(out_path: str, steps: int = 10) -> None:
    """Deterministic 10-root-step wall-chain trajectory; saves final f_leaf.

    Runs entirely on the importing tree's code (no v3.3-only symbols), so
    the same builder executes under PYTHONPATH=wt_highre_v32/src.
    """
    torch.manual_seed(0)
    octree, plan = _shell()
    host = _host_equilibrium((32, 32, 32))
    _seed_leaves(octree, host)
    weights = leaf_force_weights(octree)

    def advance(f, tau, level, substep):
        return _plain_advance(f, tau, level, substep)

    def bfl_fn(oct, out, post, gplan, ghost_vals, *, substep):
        return bfl_apply_gather(
            oct,
            out,
            post,
            ghost_plan=gplan,
            ghost_vals=ghost_vals,
            force_weights=weights,
            return_force=True,
        )

    l1 = host.clone()
    for _ in range(steps):
        old = l1.clone()
        step_octree_shell(
            octree,
            advance,
            old,
            l1,
            tau_coarse=TAU,
            reflux=False,
            ghost_plan=plan,
            bfl_fn=bfl_fn,
            host_substeps=2,
        )
    torch.save(octree.f_leaf, out_path)


def _v32_base_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", V32_TREE, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def test_default_off_bit_identical_to_frozen_v32_base() -> None:
    """Flag-off (default) trajectory == wt_highre_v32 @ dbb01b5e trajectory."""
    base_sha = _v32_base_sha()
    if not (os.path.isdir(V32_SRC) and base_sha == V32_BASE_SHA):
        pytest.skip(
            f"frozen reference tree {V32_TREE} @ {V32_BASE_SHA} not available "
            f"(found sha {base_sha!r}); bit-identity vs dbb01b5e requires it"
        )
    tmp_dir = os.environ.get("TMPDIR", "/tmp")
    out_local = os.path.join(tmp_dir, f"v33_foff_local_{os.getpid()}.pt")
    out_v32 = os.path.join(tmp_dir, f"v33_foff_v32_{os.getpid()}.pt")
    # in-process (v3.3 tree, flag defaults to off — no argument passed)
    _trajectory(out_local)
    # frozen v3.2 base tree via subprocess on the same interpreter
    env = dict(os.environ)
    env["PYTHONPATH"] = V32_SRC
    subprocess.run(
        [sys.executable, os.path.abspath(__file__), out_v32],
        check=True,
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        timeout=600,
    )
    f_local = torch.load(out_local, weights_only=True)
    f_v32 = torch.load(out_v32, weights_only=True)
    assert f_local.shape == f_v32.shape
    assert f_local.dtype == f_v32.dtype
    assert torch.equal(f_local, f_v32)
    os.remove(out_local)
    os.remove(out_v32)


if __name__ == "__main__":  # subprocess driver for the frozen-tree arm
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_leaf_post_wall_positivity.py OUT_PT")
    _trajectory(sys.argv[1])
