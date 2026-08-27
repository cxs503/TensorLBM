"""Tests for the B4 v3 physics-geometry conditioning (``tensorlbm.ai.drag_cond``).

Pins the encoding guarantees the v3 protocol relies on:

- determinism and bit-level bare-hull scale invariance / appendage no-op;
- sail/fin fractions monotone in their own scale (production grid);
- scale=1 features agree exactly with ``build_suboff_mask`` (counts, stats
  anchors, and bit-identical mask composition);
- :class:`QuotaSampler` conserves per-dataset quotas exactly;
- :func:`force_tail_bins` bin arithmetic and the C_D tail-window convention;
- :class:`CondFNODrag` forward shapes and seed-comparable init with/without
  the auxiliary head.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.ai.drag_cond import (
    COND_V3_CHANNEL_NAMES,
    COND_V5_CHANNEL_NAMES,
    GEOMETRY_CHANNEL_NAMES,
    PRODUCTION_GRID,
    CondFNODrag,
    QuotaSampler,
    condition_v3,
    condition_v5,
    force_tail_bins,
    geometry_channels,
    sail_axial_channel,
    suboff_geometry_features,
)
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask

HULLS = ("bare_hull", "with_sail", "full")

# Pre-scale regression anchors (tests/test_suboff_appendage_scale.py, base 2f75646).
BASE_SOLID_CELLS = {"bare_hull": 4093, "with_sail": 4121, "full": 4157}


def feats(hull: str, sail: float = 1.0, fin: float = 1.0):
    return suboff_geometry_features(hull, sail, fin)


class TestGeometryEncoding:
    def test_deterministic(self):
        a = feats("full", 1.7, 2.3)
        b = suboff_geometry_features("full", 1.7, 2.3)
        assert a == b
        assert np.array_equal(geometry_channels(a), geometry_channels(b))

    def test_bare_hull_scale_invariance_bitwise(self):
        ref = feats("bare_hull", 1.0, 1.0)
        for s, f in ((0.4, 3.0), (2.5, 0.7), (1.9, 1.9)):
            other = feats("bare_hull", s, f)
            # geometry is bit-identical: same counts -> identical float channels
            assert np.array_equal(geometry_channels(other), geometry_channels(ref))
            assert (other.v_bare, other.v_sail, other.v_fin, other.v_solid, other.aproj) == (
                ref.v_bare,
                ref.v_sail,
                ref.v_fin,
                ref.v_solid,
                ref.aproj,
            )

    def test_appendage_no_op_when_absent(self):
        ws = feats("with_sail", 1.6, 1.0)
        # fin_scale is a no-op on with_sail: identical geometry channels
        assert np.array_equal(
            geometry_channels(feats("with_sail", 1.6, 2.9)), geometry_channels(ws)
        )
        assert ws.v_fin == 0 and ws.fin_frac == 0.0
        bare = feats("bare_hull", 1.0, 1.0)
        assert ws.v_bare == bare.v_bare and ws.aproj_bare == bare.aproj_bare

    def test_fractions_monotone_in_own_scale(self):
        scales = (0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0)
        sail = [feats("with_sail", s, 1.0).sail_frac for s in scales]
        fin = [feats("full", 1.0, f).fin_frac for f in scales]
        assert all(b >= a for a, b in zip(sail, sail[1:]))
        assert all(b >= a for a, b in zip(fin, fin[1:]))
        assert sail[-1] > 0.05 and fin[-1] > 0.05  # non-degenerate dynamic range

    def test_scale1_matches_build_suboff_mask_exactly(self):
        for hull in HULLS:
            f = feats(hull, 1.0, 1.0)
            mask, stats = build_suboff_mask(
                hull_type=hull,
                nx=PRODUCTION_GRID.nx,
                ny=PRODUCTION_GRID.ny,
                nz=PRODUCTION_GRID.nz,
                cx=PRODUCTION_GRID.cx,
                cy=PRODUCTION_GRID.cy,
                cz=PRODUCTION_GRID.cz,
                length=PRODUCTION_GRID.length,
                device="cpu",
            )
            assert f.v_solid == stats["solid_cells"] == BASE_SOLID_CELLS[hull]
            assert f.v_bare == stats["bare_hull_solid_cells"] == BASE_SOLID_CELLS["bare_hull"]
            assert f.v_sail + f.v_fin == stats["appendage_solid_cells"]
            assert f.aproj == int((mask.max(dim=2).values > 0).sum().item())
            assert f.solid_frac == pytest.approx(1.0 + f.sail_frac + f.fin_frac)
            assert f.log_aproj_ratio == pytest.approx(math.log10(f.aproj / f.aproj_bare))

    def test_disjoint_decomposition_and_composition_bitwise(self):
        """hull | sail | fin == build_suboff_mask, and counts partition it."""
        from tensorlbm.suboff_cad import (
            SuboffConfig,
            suboff_fins_contain_points,
            suboff_hull_mask,
            suboff_sail_contains_points,
        )

        g = PRODUCTION_GRID
        zz, yy, xx = torch.meshgrid(
            torch.arange(g.nz, dtype=torch.float32),
            torch.arange(g.ny, dtype=torch.float32),
            torch.arange(g.nx, dtype=torch.float32),
            indexing="ij",
        )
        for hull, s, f in (("full", 1.0, 1.0), ("full", 0.4, 2.7), ("with_sail", 2.2, 1.0)):
            ft = feats(hull, s, f)
            mask, _ = build_suboff_mask(
                hull_type=hull,
                nx=g.nx,
                ny=g.ny,
                nz=g.nz,
                cx=g.cx,
                cy=g.cy,
                cz=g.cz,
                length=g.length,
                device="cpu",
                config=SuboffConfig(sail_scale=s, fin_scale=f),
            )
            hull_m = suboff_hull_mask(
                g.nx, g.ny, g.nz, g.cx, g.cy, g.cz, g.length, 0.0, torch.device("cpu"), None
            )
            comp = hull_m.clone()
            if hull in ("with_sail", "full"):
                comp |= suboff_sail_contains_points(
                    xx, yy, zz, center=(g.cx, g.cy, g.cz), length=g.length, scale=s
                )
            if hull == "full":
                comp |= suboff_fins_contain_points(
                    xx, yy, zz, center=(g.cx, g.cy, g.cz), length=g.length, scale=f
                )
            assert torch.equal(comp, mask)  # predicates == voxel builder, bitwise
            assert ft.v_bare + ft.v_sail + ft.v_fin == ft.v_solid == int(mask.sum().item())

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            suboff_geometry_features("full", sail_scale=0.0)
        with pytest.raises(ValueError):
            suboff_geometry_features("full", fin_scale=float("nan"))
        with pytest.raises(ValueError):
            suboff_geometry_features("winged_hull")


class TestConditionV3:
    def test_assembly_and_geometry_block(self):
        n = 4
        re = np.array([100.0, 200.0, 400.0, 800.0])
        uin = np.array([0.1, 0.12, 0.08, 0.1])
        sail = np.array([1.0, 1.0, 0.4, 2.0])
        fin = np.array([1.0, 3.0, 1.0, 0.5])
        geo = np.stack(
            [geometry_channels(feats(h, s, f)) for h, s, f in zip(("full",) * n, sail, fin)]
        )
        cond = condition_v3(re, uin, sail, fin, geo)
        assert cond.shape == (n, 8)
        assert list(COND_V3_CHANNEL_NAMES) == [
            "log10_re",
            "log10_u_in",
            "log10_sail_scale",
            "log10_fin_scale",
        ] + list(GEOMETRY_CHANNEL_NAMES)
        np.testing.assert_allclose(cond[:, 0], np.log10(re))
        np.testing.assert_allclose(cond[:, 4:], geo)
        # no identity/one-hot column: every column takes >2 distinct values
        assert all(len(np.unique(c)) > 2 for c in cond.T)

    def test_geometry_shape_validation(self):
        with pytest.raises(ValueError, match="geometry block"):
            condition_v3(np.ones(3), np.ones(3), np.ones(3), np.ones(3), np.ones((3, 3)))


class TestConditionV5:
    """Sail axial-position channel (B4 v5, 2026-08-27 sail_x campaign fix).

    Pins: (1) the v4-default path is untouched — first 8 columns are
    condition_v3 bit-identically, (2) the v5 channel breaks the sail
    translation invariance that the mask-derived geo block provably cannot
    (P1: max |diff| = 0.0 across the sweep), (3) mother encodes as exactly
    0.0, (4) the channel is strictly monotone in the multiplier.
    """

    def test_names_and_prefix_bit_identity(self):
        assert COND_V5_CHANNEL_NAMES == COND_V3_CHANNEL_NAMES + ("log10_sail_x_mult",)
        re = np.array([100.0, 200.0, 400.0])
        uin = np.full(3, 0.1)
        geo = np.stack([geometry_channels(feats("with_sail", 1.0, 1.0))] * 3)
        v3 = condition_v3(re, uin, np.ones(3), np.ones(3), geo)
        v5 = condition_v5(re, uin, np.ones(3), np.ones(3), geo, [0.7, 1.0, 1.4])
        assert v5.shape == (3, 9)
        np.testing.assert_array_equal(v5[:, :8], v3)  # bit-identical prefix

    def test_channel_breaks_translation_invariance(self):
        """Same design, different sail_x: geo block equal, cond vectors differ.

        The root-cause pin: the sail translation leaves the mask-derived
        counts bit-identical (identical geo rows) while the v5 vector —
        through its design-parameter channel — separates the points the v3
        vector could not.
        """
        re = np.array([200.0, 200.0])
        geo = np.stack([geometry_channels(feats("with_sail", 1.0, 1.0))] * 2)
        mults = [0.7, 1.4]
        v3 = condition_v3(re, np.ones(2) * 0.1, np.ones(2), np.ones(2), geo)
        v5 = condition_v5(re, np.ones(2) * 0.1, np.ones(2), np.ones(2), geo, mults)
        np.testing.assert_array_equal(v3[0], v3[1])  # v3: flat (the failure)
        assert not np.array_equal(v5[0], v5[1])  # v5: axis is visible
        np.testing.assert_allclose(v5[:, 8], np.log10(mults))

    def test_mother_is_exactly_zero_and_monotone(self):
        for m in (1.0, 1):
            assert float(sail_axial_channel(m)) == 0.0
        grid = [0.7, 0.85, 1.0, 1.15, 1.3, 1.4]
        col = sail_axial_channel(grid)
        assert np.all(np.diff(col) > 0)  # strictly increasing
        np.testing.assert_allclose(col, np.log10(grid))

    def test_scalar_broadcast_and_validation(self):
        re = np.full(4, 100.0)
        geo = np.stack([geometry_channels(feats("with_sail", 1.0, 1.0))] * 4)
        v5 = condition_v5(re, np.ones(4) * 0.1, np.ones(4), np.ones(4), geo, 1.3)
        np.testing.assert_allclose(v5[:, 8], np.log10(1.3))
        with pytest.raises(ValueError, match="scalar or length-4"):
            condition_v5(re, np.ones(4) * 0.1, np.ones(4), np.ones(4), geo, [1.0, 1.1])
        with pytest.raises(ValueError, match="finite and positive"):
            sail_axial_channel(0.0)
        with pytest.raises(ValueError, match="finite and positive"):
            sail_axial_channel(np.nan)
        with pytest.raises(ValueError, match="1-D"):
            sail_axial_channel(np.ones((2, 2)))


class TestQuotaSampler:
    def test_quota_conservation_and_membership(self):
        labels = np.array([0] * 50 + [1] * 8 + [2] * 20 + [3] * 8)
        fit = np.arange(86)
        sampler = QuotaSampler(labels, fit)
        assert sampler.quota == 50
        assert sampler.per_dataset_fit_counts == {0: 50, 1: 8, 2: 20, 3: 8}
        rng = np.random.default_rng(0)
        epoch = sampler.epoch_indices(rng)
        counts = np.bincount(labels[epoch], minlength=4)
        assert counts.tolist() == [50, 50, 50, 50]  # exact quota, every epoch
        assert set(epoch.tolist()) <= set(fit.tolist())
        assert len(epoch) == 4 * 50

    def test_deterministic_and_shuffled(self):
        labels = np.array([0] * 10 + [1] * 4)
        fit = np.arange(14)
        e1 = QuotaSampler(labels, fit).epoch_indices(np.random.default_rng(7))
        e2 = QuotaSampler(labels, fit).epoch_indices(np.random.default_rng(7))
        assert np.array_equal(e1, e2)
        assert not np.array_equal(e1, np.arange(len(e1)))  # actually shuffled

    def test_single_dataset_reduces_to_plain(self):
        labels = np.zeros(12, dtype=int)
        sampler = QuotaSampler(labels, np.arange(12))
        epoch = sampler.epoch_indices(np.random.default_rng(0))
        assert sorted(epoch.tolist()) == list(range(12))

    def test_validation(self):
        with pytest.raises(ValueError, match="non-empty"):
            QuotaSampler(np.zeros(4), [])
        with pytest.raises(ValueError):
            QuotaSampler(np.zeros(2), [0, 1, 2])  # labels shorter than indices


class TestForceTailBins:
    def test_bins_match_direct_computation(self):
        force = np.linspace(1.0, 2.0, 160) ** 2
        out = force_tail_bins(force, tail_frac=0.25, n_bins=8)
        tail = force[int(160 * 0.75) :]
        expected = np.log10(np.array([b.mean() for b in np.array_split(tail, 8)]))
        np.testing.assert_allclose(out, expected)
        # tail window == the C_D label convention
        assert len(tail) == 40

    def test_uneven_tail_splits(self):
        out = force_tail_bins(np.full(101, 3.0) + np.arange(101) * 1e-6, n_bins=8)
        assert out.shape == (8,)
        assert np.isfinite(out).all()

    def test_rejects_bad_series(self):
        with pytest.raises(ValueError, match="n_bins"):
            force_tail_bins(np.ones(10), n_bins=16)
        with pytest.raises(ValueError, match="positive"):
            force_tail_bins(np.linspace(1.0, -1.0, 160), n_bins=4)  # tail is negative
        with pytest.raises(ValueError, match="tail_frac"):
            force_tail_bins(np.ones(160), tail_frac=0.0)


class TestCondFNODrag:
    def _model(self, **kw):
        torch.manual_seed(0)
        return CondFNODrag(modes=(8, 16), **kw)

    def test_forward_shapes(self):
        model = self._model(cond_dim=8, aux_dim=8)
        x = torch.randn(3, 5, 16, 32)
        p = torch.randn(3, 8)
        assert model(x, p).shape == (3,)
        y, aux = model(x, p, return_aux=True)
        assert y.shape == (3,) and aux.shape == (3, 8)

    def test_seed_comparable_init_with_aux(self):
        """Aux construction must not shift the shared modules' init draws."""
        m1 = self._model(cond_dim=8, aux_dim=0)
        m2 = self._model(cond_dim=8, aux_dim=8)
        assert m2.aux_head is not None and m1.aux_head is None
        for k, v in m1.state_dict().items():
            assert torch.equal(v, m2.state_dict()[k]), k
        x = torch.randn(2, 5, 16, 32)
        p = torch.randn(2, 8)
        torch.testing.assert_close(m1(x, p), m2(x, p))

    def test_aux_requires_aux_dim(self):
        model = self._model(cond_dim=8, aux_dim=0)
        with pytest.raises(RuntimeError, match="aux_dim"):
            model(torch.randn(2, 5, 16, 32), torch.randn(2, 8), return_aux=True)

    def test_training_smoke_cpu(self):
        model = self._model(cond_dim=8, aux_dim=4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(8, 5, 16, 32)
        p = torch.randn(8, 8)
        y_t = torch.randn(8)
        a_t = torch.randn(8, 4)
        for _ in range(3):
            opt.zero_grad()
            y, aux = model(x, p, return_aux=True)
            loss = torch.nn.functional.mse_loss(y, y_t) + 0.1 * torch.nn.functional.mse_loss(
                aux, a_t
            )
            loss.backward()
            opt.step()
        assert torch.isfinite(loss)


class TestSubStarBareEquivalence:
    """B4 v4 data-closure premise: the sail-disappearance scale s*.

    The sail scales down by similarity about the deck plane; below the
    largest scale that still captures a voxel centre (measured s* = 0.133
    at 0.001 resolution, threshold in (0.1333, 0.1335) on the production
    grid) the with_sail design is bit-identical to the bare hull - the
    physical basis for the G2b sub-s* bare-equivalent anchors
    (docs/drag_surrogate_fno_g2_20260824.md, v4 section).
    """

    LADDER = {
        0.02: 0,
        0.05: 0,
        0.13: 0,
        0.133: 0,
        0.134: 1,
        0.15: 1,
        0.29: 1,
        0.30: 2,
        0.35: 3,
        0.40: 5,
    }

    def test_voxel_ladder_pinned(self):
        for s, v_sail in self.LADDER.items():
            assert feats("with_sail", s, 1.0).v_sail == v_sail, (
                f"sail_scale={s}: net sail voxels drifted from {v_sail}"
            )

    def test_s_star_boundary(self):
        # 0.133 is bare-equivalent, the next 0.001 step up is not
        assert feats("with_sail", 0.133, 1.0).v_sail == 0
        assert feats("with_sail", 0.134, 1.0).v_sail > 0

    def test_below_substar_is_bit_bare(self):
        bare_ch = geometry_channels(feats("bare_hull", 1.0, 1.0))
        bare_mask, _ = build_suboff_mask(
            hull_type="bare_hull",
            nx=PRODUCTION_GRID.nx,
            ny=PRODUCTION_GRID.ny,
            nz=PRODUCTION_GRID.nz,
            cx=PRODUCTION_GRID.cx,
            cy=PRODUCTION_GRID.cy,
            cz=PRODUCTION_GRID.cz,
            length=PRODUCTION_GRID.length,
            device="cpu",
        )
        for s in (0.13, 0.05, 0.02):
            f = feats("with_sail", s, 1.0)
            assert np.array_equal(geometry_channels(f), bare_ch)
            assert f.v_solid == f.v_bare and f.aproj == f.aproj_bare
            mask, _ = build_suboff_mask(
                hull_type="with_sail",
                nx=PRODUCTION_GRID.nx,
                ny=PRODUCTION_GRID.ny,
                nz=PRODUCTION_GRID.nz,
                cx=PRODUCTION_GRID.cx,
                cy=PRODUCTION_GRID.cy,
                cz=PRODUCTION_GRID.cz,
                length=PRODUCTION_GRID.length,
                device="cpu",
                config=SuboffConfig(sail_scale=s),
            )
            assert torch.equal(mask, bare_mask)
