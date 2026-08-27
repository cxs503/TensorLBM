"""Tests for tensorlbm.boundary_registry (XLB-style integer-id BC registry).

Coverage:
  * registry id assignment / conflicts / unregister / reset (0 reserved);
  * build_bc_mask (id field, unregistered rejection, empty rejection);
  * check_bc_overlaps (strict raise vs ordered warn);
  * missing-direction masks: "stream the boolean field once" derivation
    vs an independent brute-force per-direction geometric reference, for
    D3Q19 and D3Q27 over random solids and every periodicity combination;
  * consistency with the hand-written direction sets in boundaries3d
    (inlet/outlet/lid planes) — the historical hand-transcription trap;
  * application equivalence: registry dispatch (with and without the
    ``bc_mask == id`` field) reproduces the direct boundaries3d calls.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from tensorlbm.boundaries3d import (
    _D3Q19_INLET_DIRS,
    _D3Q19_OUTLET_DIRS,
    bounce_back_cells_3d,
    zou_he_inlet_velocity_3d,
)
from tensorlbm.boundary_registry import (
    BC_ID_NONE,
    BCKind,
    BCPhase,
    BoundaryCondition,
    BoundaryConditionRegistry,
    apply_boundary_conditions,
    boundary_condition_registry,
    build_bc_mask,
    check_bc_consistency,
    check_bc_overlaps,
    derive_missing_mask,
    derive_missing_mask_reference,
    face_cells,
)
from tensorlbm.d3q19 import OPPOSITE, C, equilibrium3d


@pytest.fixture()
def registry():
    reg = BoundaryConditionRegistry()
    yield reg
    reg.reset()


def _mask(value: float | None = None, nz=5, ny=6, nx=7):
    if value is None:
        return torch.zeros((nz, ny, nx), dtype=torch.bool)
    return torch.rand((nz, ny, nx)) < value


class TestRegistryIds:
    def test_ids_start_at_one_and_increment(self, registry):
        ids = [
            registry.register(BoundaryCondition(BCKind.BOUNCE_BACK, mask=_mask(0.2)))
            for _ in range(3)
        ]
        assert ids == [1, 2, 3]
        assert all(i != BC_ID_NONE for i in ids)

    def test_re_register_same_instance_rejected(self, registry):
        bc = BoundaryCondition(BCKind.BOUNCE_BACK, mask=_mask(0.2))
        registry.register(bc)
        with pytest.raises(ValueError, match="already has id"):
            registry.register(bc)

    def test_register_wrong_type_rejected(self, registry):
        with pytest.raises(TypeError):
            registry.register("not a bc")

    def test_unregister_retires_id(self, registry):
        bc = BoundaryCondition(BCKind.BOUNCE_BACK, mask=_mask(0.2))
        first = registry.register(bc)
        registry.unregister(bc)
        assert bc.id is None
        with pytest.raises(KeyError):
            registry.bc_of(first)
        second = registry.register(BoundaryCondition(BCKind.BOUNCE_BACK, mask=_mask(0.2)))
        assert second == first + 1  # retired ids are never reused

    def test_unregister_unknown_rejected(self, registry):
        bc = BoundaryCondition(BCKind.BOUNCE_BACK, mask=_mask(0.2))
        with pytest.raises(KeyError):
            registry.unregister(bc)

    def test_id_zero_reserved(self, registry):
        bc = BoundaryCondition(BCKind.BOUNCE_BACK, mask=_mask(0.2))
        registry.register(bc)
        with pytest.raises(ValueError, match="reserved"):
            registry.bc_of(0)

    def test_id_of_unregistered_raises(self, registry):
        bc = BoundaryCondition(BCKind.BOUNCE_BACK, mask=_mask(0.2))
        with pytest.raises(KeyError):
            registry.id_of(bc)

    def test_reset_clears_and_restarts(self, registry):
        bc = BoundaryCondition(BCKind.BOUNCE_BACK, mask=_mask(0.2))
        registry.register(bc)
        registry.reset()
        assert len(registry) == 0 and bc.id is None
        assert registry.next_id == 1

    def test_global_singleton_usable(self):
        bc = BoundaryCondition(BCKind.BOUNCE_BACK, mask=_mask(0.2))
        boundary_condition_registry.register(bc)
        try:
            assert boundary_condition_registry.id_of(bc) >= 1
        finally:
            boundary_condition_registry.unregister(bc)


class TestBCMasks:
    def test_build_bc_mask_assigns_ids(self, registry):
        solid = _mask(0.0)
        solid[:, 0, :] = True  # y=0 plane (disjoint from the x=0 inlet face)
        a = BoundaryCondition(BCKind.BOUNCE_BACK, mask=solid, name="a")
        b = BoundaryCondition(
            BCKind.ZOU_HE_INLET_VELOCITY, face="x-", params={"u_in": 0.05}, name="b"
        )
        registry.register(a)
        registry.register(b)
        bc_mask = build_bc_mask((5, 6, 7), [a, b])
        assert bc_mask.dtype == torch.int64
        inlet = torch.zeros((5, 6, 7), dtype=torch.bool)
        inlet[:, :, 0] = True
        # Disjoint interiors carry their own id; on the shared edge line
        # (x=0 ∧ y=0) the LAST BC in the list wins (documented semantics).
        assert (bc_mask[solid & ~inlet] == a.id).all()
        assert (bc_mask[inlet] == b.id).all()
        assert (bc_mask[~(solid | inlet)] == 0).all()

    def test_build_bc_mask_per_phase(self, registry):
        """Per-phase masks keep cross-phase shared cells selectable for the
        earlier phase (a combined last-wins field would hide them)."""
        wall = _mask(0.0)
        wall[:, :, 0] = True  # x=0 plane (crosses the lid plane)
        pre = BoundaryCondition(BCKind.BOUNCE_BACK, phase="pre_streaming", mask=wall, name="walls")
        lid = BoundaryCondition(BCKind.MOVING_LID, face="y+", params={"u_lid": 0.05}, name="lid")
        registry.register(pre)
        registry.register(lid)
        combined = build_bc_mask((5, 6, 7), [pre, lid])
        pre_mask = build_bc_mask((5, 6, 7), [pre, lid], phase="pre_streaming")
        post_mask = build_bc_mask((5, 6, 7), [pre, lid], phase="post_streaming")
        edge = (slice(None), -1, 0)  # lid plane ∩ wall plane
        # combined: the later lid claims the shared edge cells
        assert (combined[edge] == lid.id).all()
        # per-phase: the PRE selection still covers every wall cell …
        assert (pre_mask[wall] == pre.id).all()
        assert (pre_mask[~wall] == 0).all()
        # … and the POST mask carries the lid on its whole plane
        assert (post_mask[:, -1, :] == lid.id).all()
        assert (post_mask[:, :-1, :] == 0).all()

    def test_build_bc_mask_requires_registration(self):
        bc = BoundaryCondition(BCKind.BOUNCE_BACK, mask=_mask(0.3))
        with pytest.raises(ValueError, match="not registered"):
            build_bc_mask((5, 6, 7), [bc])

    def test_build_bc_mask_rejects_empty_cells(self, registry):
        bc = BoundaryCondition(BCKind.BOUNCE_BACK, mask=_mask(0.0), name="empty")
        registry.register(bc)
        with pytest.raises(ValueError, match="selects no cells"):
            build_bc_mask((5, 6, 7), [bc])

    def test_face_cells_planes(self):
        shape = (4, 5, 6)
        z0 = face_cells("z-", shape, torch.device("cpu"))
        assert z0[0].all() and not z0[1:].any()
        yplus = face_cells("y+", shape, torch.device("cpu"))
        assert yplus[:, -1].all() and not yplus[:, :-1].any()
        x0 = face_cells("x-", shape, torch.device("cpu"))
        assert x0[..., 0].all() and not x0[..., 1:].any()

    def test_bad_construction_rejected(self):
        with pytest.raises(ValueError):
            BoundaryCondition(BCKind.BOUNCE_BACK)  # mask required
        with pytest.raises(ValueError):
            BoundaryCondition(BCKind.BOUNCE_BACK, mask=_mask(0.3), face="x-")
        with pytest.raises(ValueError):
            BoundaryCondition(BCKind.MOVING_LID, mask=_mask(0.3))  # plane BC takes face
        with pytest.raises(ValueError):
            BoundaryCondition(BCKind.MOVING_LID, face="q-")  # invalid label


class TestOverlapChecks:
    def _two_bcs(self, overlap: bool):
        m1 = _mask(0.0)
        m1[:, :, 0] = True
        m2 = _mask(0.0)
        m2[:, :, -1] = True
        if overlap:  # both claim a shared plane
            m2[:, 0, :] = True
            m1[:, 0, :] = True
        return (
            BoundaryCondition(BCKind.BOUNCE_BACK, mask=m1, name="bc1"),
            BoundaryCondition(BCKind.BOUNCE_BACK, mask=m2, name="bc2"),
        )

    def test_disjoint_ok(self, registry):
        a, b = self._two_bcs(False)
        registry.register(a)
        registry.register(b)
        check_bc_overlaps([a, b], (5, 6, 7))

    def test_overlap_strict_raises(self, registry):
        a, b = self._two_bcs(True)
        registry.register(a)
        registry.register(b)
        with pytest.raises(ValueError, match="overlap"):
            check_bc_overlaps([a, b], (5, 6, 7))

    def test_overlap_ordered_warns(self, registry):
        a, b = self._two_bcs(True)
        registry.register(a)
        registry.register(b)
        with pytest.warns(UserWarning, match="overlap"):
            check_bc_overlaps([a, b], (5, 6, 7), strict=False)

    def test_cross_phase_overlap_allowed(self, registry):
        """Cavity geometry: a PRE wall plane meeting a POST lid plane on
        its edge lines is not an error — the two-phase pipeline applies
        them in a fixed order (pre before post)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning either
            wall = _mask(0.0)
            wall[:, :, 0] = True
            wall[:, 0, :] = True
            pre = BoundaryCondition(
                BCKind.BOUNCE_BACK, phase="pre_streaming", mask=wall, name="walls"
            )
            lid = BoundaryCondition(
                BCKind.MOVING_LID, face="y+", params={"u_lid": 0.05}, name="lid"
            )
            registry.register(pre)
            registry.register(lid)
            check_bc_overlaps([pre, lid], (5, 6, 7))  # strict: must not raise
            check_bc_consistency([pre, lid], (5, 6, 7))

    def test_consistency_requires_ids(self):
        a, _ = self._two_bcs(False)
        with pytest.raises(ValueError, match="no valid registry id"):
            check_bc_consistency([a], (5, 6, 7))

    def test_plane_bc_must_cover_its_face(self, registry):
        bc = BoundaryCondition(BCKind.ZOU_HE_INLET_VELOCITY, face="x-", params={"u_in": 0.05})
        registry.register(bc)
        check_bc_consistency([bc], (5, 6, 7))  # face-derived cells match


class TestMissingMask:
    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    @pytest.mark.parametrize(
        "periodic",
        [
            False,
            True,
            (True, False, False),
            (False, True, False),
            (False, False, True),
            {"x": True, "y": False, "z": True},
        ],
    )
    def test_stream_method_matches_bruteforce(self, lattice, periodic):
        torch.manual_seed(0)
        nz, ny, nx = 5, 6, 7
        solid = torch.rand((nz, ny, nx)) < 0.2
        fast = derive_missing_mask(
            (nz, ny, nx), solid_mask=solid, periodic=periodic, lattice=lattice
        )
        ref = derive_missing_mask_reference(
            (nz, ny, nx), solid_mask=solid, periodic=periodic, lattice=lattice
        )
        assert fast.shape == ref.shape
        assert fast.dtype == torch.bool
        assert torch.equal(fast, ref)

    def test_rest_direction_never_missing(self):
        torch.manual_seed(3)
        solid = torch.rand((4, 5, 6)) < 0.4
        mask = derive_missing_mask((4, 5, 6), solid_mask=solid)
        # q=0 pulls from the cell itself: missing only where the cell is
        # itself solid, never at fluid cells.
        assert not mask[0][~solid].any()
        assert mask[0][solid].all()

    def test_interior_fluid_cell_without_solid_neighbours(self):
        mask = derive_missing_mask((6, 6, 6))  # all blocked padding, no solid
        assert not mask[:, 2, 2, 2].any()  # deep interior misses nothing

    def test_matches_boundaries3d_inlet_directions(self):
        """Auto-derived missing set at the x=0 plane interior must equal
        the hand-written ``_D3Q19_INLET_DIRS`` used by
        ``zou_he_inlet_velocity_3d``.  (Edge/corner cells of the plane
        additionally miss the other non-periodic faces' directions — the
        classic LBM corner caveat — so the interior is compared.)"""
        nz, ny, nx = 6, 7, 8
        mask = derive_missing_mask((nz, ny, nx))
        inner = (slice(1, nz - 1), slice(1, ny - 1), 0)
        missing_at_inlet = {q for q in range(19) if mask[q][inner].any()}
        assert missing_at_inlet == set(_D3Q19_INLET_DIRS)
        inner_out = (slice(1, nz - 1), slice(1, ny - 1), nx - 1)
        missing_at_outlet = {q for q in range(19) if mask[q][inner_out].any()}
        assert missing_at_outlet == set(_D3Q19_OUTLET_DIRS)

    def test_matches_moving_lid_unknown_directions(self):
        """Lid plane interior y=ny-1 misses exactly the cy<0 directions
        {4, 8, 9, 16, 18} reconstructed by ``zou_he_moving_lid_3d``."""
        nz, ny, nx = 6, 7, 8
        mask = derive_missing_mask((nz, ny, nx))
        inner = (slice(1, nz - 1), ny - 1, slice(1, nx - 1))
        missing_at_lid = {q for q in range(19) if mask[q][inner].any()}
        c_y = C[:, 1]
        expected = {q for q in range(19) if int(c_y[q]) < 0}
        assert missing_at_lid == expected == {4, 8, 9, 16, 18}

    def test_periodic_face_has_no_missing_directions(self):
        nz, ny, nx = 6, 7, 8
        mask = derive_missing_mask((nz, ny, nx), periodic={"z": True})
        # z-periodic faces pull from the wrap; interior cells of those
        # faces miss nothing.
        inner = (0, slice(1, ny - 1), slice(1, nx - 1))
        assert not mask[(slice(None),) + inner].any()
        inner_top = (nz - 1, slice(1, ny - 1), slice(1, nx - 1))
        assert not mask[(slice(None),) + inner_top].any()
        # non-periodic x=0 still misses the inflowing directions.
        inlet = (slice(1, nz - 1), slice(1, ny - 1), 0)
        assert {q for q in range(19) if mask[q][inlet].any()} == set(_D3Q19_INLET_DIRS)

    def test_near_wall_cell_misses_directions_into_solid(self):
        nz, ny, nx = 6, 7, 8
        solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
        solid[:, :, 0] = True  # wall plane at x=0
        mask = derive_missing_mask((nz, ny, nx), solid_mask=solid)
        # At the adjacent fluid plane x=1 the pull source x - c_x lands on
        # the wall exactly for the cx>0 directions (the same set the
        # Zou-He inlet reconstructs at a -x boundary); the bounce-back
        # reflection is what supplies them.
        from_wall = {q for q in range(19) if int(C[q, 0]) > 0}
        inner = (slice(1, nz - 1), slice(1, ny - 1), 1)
        assert {q for q in range(19) if mask[q][inner].any()} == from_wall

    def test_invalid_solid_mask_rejected(self):
        with pytest.raises(ValueError):
            derive_missing_mask((4, 5, 6), solid_mask=torch.zeros(3, 3, 3, dtype=torch.bool))


def _random_f(nz=5, ny=6, nx=7, seed=1):
    torch.manual_seed(seed)
    rho = 1.0 + 0.01 * torch.randn((nz, ny, nx))
    u = 0.02 * torch.randn((3, nz, ny, nx))
    return equilibrium3d(rho, u[0], u[1], u[2])


class TestApplicationEquivalence:
    def test_pre_bounce_back_matches_verified_cavity_helper(self):
        """Registry PRE_STREAMING bounce-back (f_pre-aware, bc_mask path)
        reproduces the verified cavity's ``stationary_pre_bounce3d``."""
        wall = _mask(0.0)
        wall[:, :, 0] = True
        wall[:, 0, :] = True
        f_pre = _random_f(seed=2)
        f_collided = _random_f(seed=3)
        opp = OPPOSITE
        expected = torch.where(wall.unsqueeze(0), f_pre[opp], f_collided)

        reg = BoundaryConditionRegistry()
        bc = BoundaryCondition(
            BCKind.BOUNCE_BACK, phase=BCPhase.PRE_STREAMING, mask=wall, name="walls"
        )
        reg.register(bc)
        bc_mask = build_bc_mask((5, 6, 7), [bc])
        out = apply_boundary_conditions(
            f_collided, [bc], phase="pre_streaming", bc_mask=bc_mask, f_pre=f_pre
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        reg.reset()

    def test_maskless_dispatch_matches_bc_mask_dispatch(self):
        wall = _mask(0.3)
        f = _random_f(seed=4)
        reg = BoundaryConditionRegistry()
        bc = BoundaryCondition(BCKind.BOUNCE_BACK, mask=wall, name="bb")
        reg.register(bc)
        via_mask = apply_boundary_conditions(
            f, [bc], phase="post_streaming", bc_mask=build_bc_mask((5, 6, 7), [bc])
        )
        via_cells = apply_boundary_conditions(f, [bc], phase="post_streaming")
        torch.testing.assert_close(via_mask, via_cells, rtol=0, atol=0)
        torch.testing.assert_close(via_mask, bounce_back_cells_3d(f, wall), rtol=0, atol=0)
        reg.reset()

    def test_pre_bounce_requires_f_pre(self):
        wall = _mask(0.3)
        reg = BoundaryConditionRegistry()
        bc = BoundaryCondition(BCKind.BOUNCE_BACK, phase=BCPhase.PRE_STREAMING, mask=wall)
        reg.register(bc)
        with pytest.raises(ValueError, match="f_pre"):
            apply_boundary_conditions(_random_f(), [bc], phase="pre_streaming")
        reg.reset()

    def test_zou_he_inlet_dispatch_matches_direct_call(self):
        f = _random_f(seed=5)
        reg = BoundaryConditionRegistry()
        bc = BoundaryCondition(BCKind.ZOU_HE_INLET_VELOCITY, face="x-", params={"u_in": 0.05})
        reg.register(bc)
        out = apply_boundary_conditions(f, [bc], phase="post_streaming")
        torch.testing.assert_close(out, zou_he_inlet_velocity_3d(f, 0.05), rtol=0, atol=0)
        reg.reset()

    def test_periodic_is_noop(self):
        f = _random_f(seed=6)
        reg = BoundaryConditionRegistry()
        bc = BoundaryCondition(BCKind.PERIODIC, face="z-")
        reg.register(bc)
        out = apply_boundary_conditions(f, [bc], phase="post_streaming")
        assert out is f
        reg.reset()

    def test_phase_filtering(self):
        wall = _mask(0.3)
        f = _random_f(seed=7)
        reg = BoundaryConditionRegistry()
        pre_bc = BoundaryCondition(
            BCKind.BOUNCE_BACK, phase=BCPhase.PRE_STREAMING, mask=wall, name="pre"
        )
        post_bc = BoundaryCondition(
            BCKind.BOUNCE_BACK, phase=BCPhase.POST_STREAMING, mask=wall, name="post"
        )
        reg.register(pre_bc)
        reg.register(post_bc)
        only_pre = apply_boundary_conditions(
            f, [pre_bc, post_bc], phase="pre_streaming", f_pre=f.clone()
        )
        only_post = apply_boundary_conditions(f, [pre_bc, post_bc], phase="post_streaming")
        # PRE with f_pre == f is the plain bounce-back swap.
        torch.testing.assert_close(
            only_pre,
            torch.where(wall.unsqueeze(0), f[OPPOSITE], f),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(only_post, bounce_back_cells_3d(f, wall), rtol=0, atol=0)
        reg.reset()
