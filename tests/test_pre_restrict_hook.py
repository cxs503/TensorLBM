"""Contract tests for the ``pre_restrict_hook`` hybrid-stage insertion point.

The hook is where a hybrid octree-shell stage (route C) replaces the
covered region of the finest static level with the leaf restriction and
reflux-corrects its exterior — see ``StaticBlockAMR3D.step`` /
``NestedStaticBlockAMR3D.step``.  These tests pin:

* the single-block hook fires once per step with ``(fine_old, fine_new)``
  where ``fine_old`` is the pre-step fine state;
* an in-place mutation inside the hook reaches the restriction (the
  coarse box sees the mutated values);
* the nested hook fires at the innermost interface once per parent
  substep burst (twice per root step for two 2:1 interfaces), with the
  burst-start state as ``fine_old``;
* ``hook=None`` leaves the trajectory bit-identical (regression).
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.refinement import BoxRegion
from tensorlbm.solver3d import collide_mrt3d, stream3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    NestedStaticBlockAMR3D,
    StaticBlockAMR3D,
    StaticBlockAMRConfig,
)

GHOST = 1


def _equilibrium(shape, rho=1.01, ux=0.025, uy=-0.006, uz=0.003):
    return equilibrium3d(
        torch.full(shape, rho, dtype=torch.float64),
        torch.full(shape, ux, dtype=torch.float64),
        torch.full(shape, uy, dtype=torch.float64),
        torch.full(shape, uz, dtype=torch.float64),
    )


def _advance_mrt(tau_fine):
    calls = {"n": 0}

    def advance(f, tau, level, substep):
        calls["n"] += 1
        if tau_fine is not None and level >= 1:
            tau = tau_fine
        post = collide_mrt3d(f, tau)
        return AMRAdvanceResult(stream3d(post), post)

    return advance, calls


def test_single_block_hook_timing_and_tensors() -> None:
    config = StaticBlockAMRConfig(BoxRegion(x0=3, x1=9, y0=3, y1=9, z0=3, z1=8), tau_coarse=0.56)
    solver = StaticBlockAMR3D(_equilibrium((12, 12, 14)), config)
    fine_before = solver.fine_f.clone()
    seen = []

    def hook(fine_old, fine_new):
        seen.append((fine_old.clone(), fine_new.clone()))

    advance, _ = _advance_mrt(None)
    solver.step(advance, pre_restrict_hook=hook)
    assert len(seen) == 1
    fine_old, fine_new = seen[0]
    assert fine_old.shape == solver.fine_f.shape
    # fine_old is the pre-step state (ghost ring included, pre-refill)
    assert torch.equal(fine_old, fine_before)
    assert not torch.equal(fine_new, fine_before)


def test_single_block_hook_mutation_reaches_restriction() -> None:
    config = StaticBlockAMRConfig(BoxRegion(x0=3, x1=9, y0=3, y1=9, z0=3, z1=8), tau_coarse=0.56)
    clean = StaticBlockAMR3D(_equilibrium((12, 12, 14)), config)
    hooked = StaticBlockAMR3D(_equilibrium((12, 12, 14)), config)
    advance, _ = _advance_mrt(None)
    clean.step(_advance_mrt(None)[0])

    def hook(fine_old, fine_new):
        # wipe the physical interior to a constant equilibrium field
        g = GHOST
        const = fine_new[:, g : g + 1, g : g + 1, g : g + 1].expand_as(
            fine_new[:, g:-g, g:-g, g:-g],
        )
        fine_new[:, g:-g, g:-g, g:-g] = const.clone()

    hooked.step(_advance_mrt(None)[0], pre_restrict_hook=hook)
    b = config.box
    # the coarse box under the hook differs from the clean run
    diff = (
        clean.coarse_f[:, b.z0 : b.z1, b.y0 : b.y1, b.x0 : b.x1]
        - hooked.coarse_f[:, b.z0 : b.z1, b.y0 : b.y1, b.x0 : b.x1]
    )
    assert float(diff.abs().max()) > 0.0
    # and the mutated fine state is what the object retains after the step
    g = GHOST
    interior = hooked.fine_f[:, g:-g, g:-g, g:-g]
    spatial_spread = interior.amax(dim=(1, 2, 3)) - interior.amin(dim=(1, 2, 3))
    assert float(spatial_spread.max()) == pytest.approx(0.0, abs=1e-15)


def test_nested_hook_fires_per_parent_burst() -> None:
    outer = StaticBlockAMRConfig(
        BoxRegion(x0=3, x1=11, y0=3, y1=9, z0=3, z1=9),
        tau_coarse=0.56,
    )
    inner = StaticBlockAMRConfig(
        BoxRegion(x0=4, x1=14, y0=4, y1=10, z0=4, z1=10),
        tau_coarse=outer.tau_fine,
    )
    hierarchy = NestedStaticBlockAMR3D(_equilibrium((12, 12, 14)), (outer, inner))
    seen = []

    def hook(fine_old, fine_new):
        seen.append((fine_old.clone(), fine_new.clone()))

    advance, _ = _advance_mrt(None)
    finest_before = hierarchy.finest_f.clone()
    hierarchy.step(advance, pre_restrict_hook=hook)
    # two parent substeps per root step -> two bursts of the finest level
    assert len(seen) == 2
    # first burst's fine_old is the pre-step finest state
    assert torch.equal(seen[0][0], finest_before)
    # second burst starts from the state the first burst ended on
    assert torch.equal(seen[1][0], seen[0][1])
    assert seen[0][0].shape == hierarchy.finest_f.shape


def test_hook_none_is_bitwise_unchanged() -> None:
    outer, inner = (
        StaticBlockAMRConfig(BoxRegion(x0=3, x1=11, y0=3, y1=9, z0=3, z1=9), tau_coarse=0.56),
        StaticBlockAMRConfig(
            BoxRegion(x0=4, x1=14, y0=4, y1=10, z0=4, z1=10),
            tau_coarse=0.62,
        ),
    )
    a = NestedStaticBlockAMR3D(_equilibrium((12, 12, 14)), (outer, inner))
    b = NestedStaticBlockAMR3D(_equilibrium((12, 12, 14)), (outer, inner))
    advance, _ = _advance_mrt(None)
    for _ in range(3):
        a.step(_advance_mrt(None)[0])
        b.step(_advance_mrt(None)[0], pre_restrict_hook=None)
    for fa, fb in zip(a.level_populations, b.level_populations, strict=True):
        assert torch.equal(fa, fb)
