"""far_field_bc_3d face-label → tensor-axis contract tests.

``far_field_bc_3d`` accepts face labels ("y-", "y+", "z-", "z+") in its
``bc_config``.  The tensor layout is ``(Q, nz, ny, nx)``: dim 1 is the
z-axis and dim 2 is the y-axis, so a label must slice the axis it names.
Historically the y and z branches were swapped relative to the axes —
unobservable while every caller wrote one uniform far-field value on all
four lateral planes, but a trap for any differentiated per-face logic
(the multi-GPU z-slab runner had to compensate with deliberately swapped
labels).  These tests pin the corrected mapping face-by-face with
sentinel tensors in which every plane is distinguishable.
"""

import pytest
import torch

from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d

# Ground-truth label -> (axis dim, index) for the (Q, nz, ny, nx) layout.
FACE_TO_SLICE = {
    "y-": (2, 0),
    "y+": (2, -1),
    "z-": (1, 0),
    "z+": (1, -1),
}

NZ, NY, NX = 5, 7, 9  # distinct sizes so axis mix-ups cannot cancel
U_IN = 0.07


def _sentinel() -> torch.Tensor:
    """(19, nz, ny, nx) tensor with a unique value in every cell.

    Values are far away from any equilibrium population (O(1e3) vs
    O(1e-2)), so "plane equals feq" and "plane kept its sentinel" are
    unambiguous bit-level distinctions.
    """
    base = torch.arange(19 * NZ * NY * NX, dtype=torch.float32)
    return 1000.0 + base.reshape(19, NZ, NY, NX).clone()


def _feq_vec() -> torch.Tensor:
    rho1 = torch.ones((1, 1, 1))
    feq = equilibrium3d(
        rho1, torch.full_like(rho1, U_IN), torch.zeros_like(rho1), torch.zeros_like(rho1)
    )
    return feq[:, 0, 0, 0]  # (19,)


def _view4(*assignments: tuple[int, object]) -> tuple[object, ...]:
    """Build a 4-D index tuple; unassigned dims are ``slice(None)``."""
    sl: list[object] = [slice(None)] * 4
    for dim, idx in assignments:
        sl[dim] = idx
    return tuple(sl)


class TestFaceLabelWritesNamedAxis:
    """Each face label must write the plane of the axis it names — only it."""

    @pytest.mark.parametrize("face", list(FACE_TO_SLICE))
    def test_named_plane_gets_equilibrium(self, face: str) -> None:
        dim, idx = FACE_TO_SLICE[face]
        f0 = _sentinel()
        out = far_field_bc_3d(
            f0, u_in=U_IN, bc_config={"far_field_faces": [face], "periodic_faces": []}
        )
        written = out[_view4((dim, idx))]
        expected = _feq_vec().view(19, *([1] * (written.ndim - 1))).expand_as(written)
        assert torch.equal(written, expected)

    @pytest.mark.parametrize("face", list(FACE_TO_SLICE))
    def test_other_lateral_planes_untouched(self, face: str) -> None:
        """No other lateral plane may change, apart from the shared edge."""
        dim, _idx = FACE_TO_SLICE[face]
        f0 = _sentinel()
        out = far_field_bc_3d(
            f0, u_in=U_IN, bc_config={"far_field_faces": [face], "periodic_faces": []}
        )
        for other, (odim, oidx) in FACE_TO_SLICE.items():
            if other == face:
                continue
            # Drop the shared corner edge with the written plane, the
            # always-written inlet column (x=0) and the outlet column
            # (x=nx-1, zero-gradient copy) — none of those are lateral writes.
            view = _view4((odim, oidx), (dim, slice(1, -1)), (3, slice(1, -1)))
            assert torch.equal(out[view], f0[view]), (
                f"face {face!r} leaked a write into the {other!r} plane"
            )

    @pytest.mark.parametrize("face", list(FACE_TO_SLICE))
    def test_swap_partner_plane_not_written(self, face: str) -> None:
        """Explicit guard: the label must NOT hit the swapped-axis plane.

        This is the regression that motivated the fix — "y-" used to write
        ``f[:, 0, :, :]`` (the z-axis low plane) and vice versa.
        """
        dim, idx = FACE_TO_SLICE[face]
        swapped_dim = 3 - dim  # 1 <-> 2
        f0 = _sentinel()
        out = far_field_bc_3d(
            f0, u_in=U_IN, bc_config={"far_field_faces": [face], "periodic_faces": []}
        )
        for sidx in (0, -1):
            view = _view4((swapped_dim, sidx), (dim, slice(1, -1)), (3, slice(1, -1)))
            assert torch.equal(out[view], f0[view]), (
                f"face {face!r} wrote the axis-swapped plane dim{swapped_dim}[{sidx}]"
            )

    @pytest.mark.parametrize("face", list(FACE_TO_SLICE))
    def test_interior_untouched_and_outlet_zero_gradient(self, face: str) -> None:
        dim, _idx = FACE_TO_SLICE[face]
        f0 = _sentinel()
        out = far_field_bc_3d(
            f0, u_in=U_IN, bc_config={"far_field_faces": [face], "periodic_faces": []}
        )
        interior = _view4((1, slice(1, -1)), (2, slice(1, -1)), (3, slice(1, -1)))
        assert torch.equal(out[interior], f0[interior])
        # Outlet is a zero-gradient copy of x=nx-2, away from the lateral
        # planes whose corner cells the lateral feq write overwrites.
        view_out = _view4((3, -1), (dim, slice(1, -1)))
        view_src = _view4((3, -2), (dim, slice(1, -1)))
        assert torch.equal(out[view_out], f0[view_src])


class TestInletOutlet:
    def test_inlet_equilibrium_outlet_zero_gradient(self) -> None:
        f0 = _sentinel()
        out = far_field_bc_3d(
            f0,
            u_in=U_IN,
            bc_config={"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []},
        )
        feq = _feq_vec().view(19, 1, 1, 1).expand(19, NZ, NY, NX)
        assert torch.equal(out[:, :, :, 0], feq[:, :, :, 0])
        # Compare the outlet on the dims-1/2 interior: the lateral feq write
        # overwrites the outlet plane's corner cells after the copy.
        view_out = _view4((3, -1), (1, slice(1, -1)), (2, slice(1, -1)))
        view_src = _view4((3, -2), (1, slice(1, -1)), (2, slice(1, -1)))
        assert torch.equal(out[view_out], f0[view_src])

    def test_periodic_inlet_and_outlet_left_untouched(self) -> None:
        f0 = _sentinel()
        out = far_field_bc_3d(
            f0,
            u_in=U_IN,
            bc_config={
                "far_field_faces": ["y-", "y+", "z-", "z+"],
                "periodic_faces": ["x-", "x+"],
            },
        )
        # Inlet/outlet planes keep their sentinel except the rows/columns
        # the lateral far-field writes legitimately touch.
        for xidx in (0, -1):
            view = _view4((3, xidx), (1, slice(1, -1)), (2, slice(1, -1)))
            assert torch.equal(out[view], f0[view])


class TestLegacyAndDefaults:
    def test_legacy_none_equals_explicit_all_four(self) -> None:
        """bc_config=None must stay bit-identical to listing all 4 faces.

        This is why the historical label swap never surfaced: the legacy
        path writes one uniform far-field value on every lateral plane.
        """
        f0 = _sentinel()
        legacy = far_field_bc_3d(f0, u_in=U_IN)
        explicit = far_field_bc_3d(
            f0,
            u_in=U_IN,
            bc_config={"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []},
        )
        assert torch.equal(legacy, explicit)

    def test_face_in_neither_list_left_untouched(self) -> None:
        """A face absent from both lists gets no far-field write.

        (thermal_common's x-only config relies on this; the docstring used
        to claim such faces "default to far-field", which the code never
        did — the doc now matches this pinned behaviour.)
        """
        f0 = _sentinel()
        out = far_field_bc_3d(
            f0,
            u_in=U_IN,
            bc_config={"far_field_faces": ["x-", "x+"], "periodic_faces": ["z-", "z+"]},
        )
        for dim, idx in FACE_TO_SLICE.values():
            view = _view4((dim, idx), (3, slice(1, -1)))
            assert torch.equal(out[view], f0[view])

    def test_periodic_wins_over_far_field_list(self) -> None:
        f0 = _sentinel()
        out = far_field_bc_3d(
            f0,
            u_in=U_IN,
            bc_config={"far_field_faces": ["y-"], "periodic_faces": ["y-"]},
        )
        view = _view4((2, 0), (3, slice(1, -1)))
        assert torch.equal(out[view], f0[view])


class TestCrossImplementationConsistency:
    @pytest.mark.parametrize("face", list(FACE_TO_SLICE))
    def test_non_equilibrium_open_boundary_same_plane(self, face: str) -> None:
        """external_open_boundary (always axis-correct) must agree on planes.

        non_equilibrium_far_field_bc_3d maps "y-" to the dim-2 plane and
        "z-" to the dim-1 plane; far_field_bc_3d must touch exactly the
        same plane for the same label.
        """
        ext = pytest.importorskip("tensorlbm.external_open_boundary")
        dim, idx = FACE_TO_SLICE[face]
        f0 = _sentinel()
        out = ext.non_equilibrium_far_field_bc_3d(f0, u_in=U_IN, faces=(face,))
        changed = out != f0
        inside = _view4((dim, idx))
        assert changed[inside].any(), f"face {face!r}: nothing written in dim{dim}[{idx}]"
        mask = torch.ones_like(changed)
        mask[inside] = False
        assert not (changed & mask).any(), (
            f"non_equilibrium face {face!r} changed cells outside dim{dim}[{idx}]"
        )


class TestDistributedRankConfig:
    """suboff_torch_distributed's per-rank config under the true labels."""

    def test_build_rank_far_field_bc_config(self) -> None:
        mod = pytest.importorskip("tensorlbm.suboff_torch_distributed")
        cfg = mod.build_rank_far_field_bc_config(0, 4)
        assert cfg == {"far_field_faces": ["y-", "y+", "z-"], "periodic_faces": ["z+"]}
        cfg = mod.build_rank_far_field_bc_config(2, 4)
        assert cfg == {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}
        cfg = mod.build_rank_far_field_bc_config(3, 4)
        assert cfg == {"far_field_faces": ["y-", "y+", "z+"], "periodic_faces": ["z-"]}
        cfg = mod.build_rank_far_field_bc_config(0, 1)
        assert cfg == {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}

    @pytest.mark.parametrize("rank,world", [(0, 1), (0, 4), (2, 4), (3, 4)])
    def test_rank_config_writes_expected_planes(self, rank: int, world: int) -> None:
        """y faces on every rank; z faces only on global-boundary ranks.

        On a z-slab decomposition (dim 1), an interior rank's dim-1 planes
        are halo neighbours, not domain boundaries — writing far-field
        there would corrupt the interior, which is exactly what the old
        swapped-label compensation existed to avoid.  The plane set must
        therefore be identical to the pre-fix compensated behaviour.
        """
        mod = pytest.importorskip("tensorlbm.suboff_torch_distributed")
        cfg = mod.build_rank_far_field_bc_config(rank, world)
        f0 = _sentinel()
        out = far_field_bc_3d(f0, u_in=U_IN, bc_config=cfg)

        feq3 = _feq_vec().view(19, 1, 1)  # plane views are (Q, n, n)
        # y-axis (dim 2) faces: far-field on every rank.
        for idx in (0, -1):
            view = _view4((2, idx))
            assert torch.equal(out[view], feq3.expand(out[view].shape))
        # z-axis (dim 1) faces: far-field only on the owning boundary rank.
        owns_low = rank == 0 or world == 1
        owns_high = rank == world - 1 or world == 1
        for idx, owns in ((0, owns_low), (-1, owns_high)):
            view = _view4((1, idx))
            if owns:
                assert torch.equal(out[view], feq3.expand(out[view].shape))
            else:
                # Exclude the y-face rows (always far-field on every
                # rank) and the inlet/outlet columns.
                view_inner = _view4((1, idx), (2, slice(1, -1)), (3, slice(1, -1)))
                assert torch.equal(out[view_inner], f0[view_inner])
        interior = _view4((1, slice(1, -1)), (2, slice(1, -1)), (3, slice(1, -1)))
        assert torch.equal(out[interior], f0[interior])
