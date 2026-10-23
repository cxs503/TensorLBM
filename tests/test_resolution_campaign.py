"""Tests for the B4-g4 resolution axis: scan plumbing + conditioning.

Two groups:

* **Resolution plumbing** — the ``suboff_n128`` case and the scan chain
  accept non-production integer ``resolution`` values (the campaign runs
  the *same* registered case at n=64 / n=160 with grid-relative hull
  placement), while the default (resolution=128) path is unchanged: same
  production grid, same ``tau`` formula, integer coercion through
  ``coerce_case_params`` as before.
* **Resolution conditioning** — :func:`resolution_channel` /
  :func:`condition_v4` pure functions: production n maps to exactly 0.0,
  the first 8 columns of v4 are bit-identical to v3, and per-point /
  scalar / invalid inputs behave.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorlbm.ai.drag_cond import (
    COND_V3_CHANNEL_NAMES,
    COND_V4_CHANNEL_NAMES,
    PRODUCTION_GRID,
    RESOLUTION_CHANNEL_NAME,
    SuboffGrid,
    condition_v3,
    condition_v4,
    geometry_channels,
    resolution_channel,
    suboff_geometry_features,
)
from tensorlbm.cases.suboff import SuboffChannelCase
from tensorlbm.scan_runner import ScanPlan, ScanPoint, ScanVariable, coerce_case_params
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask

TIERS = (64, 128, 160)
U_IN = 0.10


def _case(n: int, re: float = 100.0) -> SuboffChannelCase:
    return SuboffChannelCase(resolution=n, re=re, u_in=U_IN, device="cpu")


class TestResolutionPlumbing:
    """Grid / tau / geometry behaviour of the case at each tier + default."""

    def test_grid_shapes_track_resolution(self):
        for n in TIERS:
            case = _case(n)
            assert tuple(case.resolution) == (n // 2, n // 2, n)
            assert case.hull_length == pytest.approx(0.6 * n)

    def test_default_construction_is_production_grid(self):
        # no resolution kwarg: unchanged production placement
        case = SuboffChannelCase(re=100.0, u_in=U_IN, device="cpu")
        assert tuple(case.resolution) == (
            PRODUCTION_GRID.nz,
            PRODUCTION_GRID.ny,
            PRODUCTION_GRID.nx,
        )
        assert case.hull_length == pytest.approx(76.8)

    def test_tau_formula_scales_with_grid(self):
        # tau = 0.5 + 3 * u_in * (0.6 n) / re for every tier (the
        # CaseUnits.from_reference contract, evaluated on each grid)
        for n in TIERS:
            for re in (60.0, 420.0, 700.0):
                case = _case(n, re)
                assert case.units.tau == pytest.approx(0.5 + 3.0 * U_IN * 0.6 * n / re)

    def test_solid_mask_voxelises_on_every_tier(self):
        for n in TIERS:
            mask = np.asarray(_case(n).solid_mask().cpu()) > 0
            assert mask.shape == (n // 2, n // 2, n)
            assert mask.sum() > 0
            # hull centred at cx = 0.35 nx: solid confined to x in [0.05, 0.65] nx
            xs = mask.any(axis=(0, 1)).nonzero()[0]
            assert 0.0 <= xs.min() / n and xs.max() / n <= 0.65 + 1e-9

    def test_coerce_case_params_rounds_resolution(self):
        out = coerce_case_params("suboff_n128", {"resolution": 64.0, "re": 123.4})
        assert out["resolution"] == 64 and isinstance(out["resolution"], int)
        assert isinstance(out["re"], float)
        assert coerce_case_params("suboff_n128", {"resolution": 160})["resolution"] == 160

    def test_scan_plan_round_trip_keeps_resolution_fixed_param(self):
        plan = ScanPlan(
            scan_id="scan-suboff-resolution-n64-test",
            case="suboff_n128",
            variables=(ScanVariable(name="re", low=60.0, high=700.0),),
            method="latin_hypercube",
            n_points=1,
            seed=0,
            steps=100,
            snapshot_every=50,
            code_sha="0" * 40,
            fixed_params={"resolution": 64, "collision": "cumulant", "u_in": U_IN},
            points=(
                ScanPoint(
                    index=0,
                    point_id="p0000",
                    run_id="scan-suboff-resolution-n64-test-p0000",
                    params={"hull_type": "with_sail", "re": 100.0},
                ),
            ),
        )
        again = ScanPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        assert again.fixed_params["resolution"] == 64
        assert isinstance(again.fixed_params["resolution"], int)
        assert again.plan_digest() == plan.plan_digest()

    def test_geometry_features_equal_mask_truth_on_tier_grids(self):
        # the CAD-predicate feature counts must equal the mask the scan
        # chain voxelises with, away from the production grid too
        for n in TIERS:
            grid = SuboffGrid.from_resolution(n)
            feat = suboff_geometry_features("with_sail", 1.0, 1.0, grid=grid)
            mask, _ = build_suboff_mask(
                hull_type="with_sail",
                nx=grid.nx,
                ny=grid.ny,
                nz=grid.nz,
                cx=grid.cx,
                cy=grid.cy,
                cz=grid.cz,
                length=grid.length,
                config=SuboffConfig(),
                device="cpu",
            )
            mask = np.asarray(mask.cpu()) > 0
            assert feat.v_solid == int(mask.sum())
            assert feat.aproj == int((mask.max(axis=2) > 0).sum())
            if n >= 64:
                assert feat.v_sail > 0, "sail must survive voxelisation on campaign tiers"

    def test_bare_channels_scale_invariant_on_tier_grids(self):
        for n in TIERS:
            ch = geometry_channels(
                suboff_geometry_features("bare_hull", 1.0, 1.0, grid=SuboffGrid.from_resolution(n))
            )
            assert ch[0] == 0.0 and ch[1] == 0.0 and ch[2] == 0.0
            assert ch[3] == 1.0


class TestResolutionChannel:
    def test_production_maps_to_exact_zero(self):
        assert resolution_channel(128).item() == 0.0
        assert resolution_channel(128.0).item() == 0.0

    def test_tier_values(self):
        got = resolution_channel(np.array([64, 128, 160]))
        assert got.shape == (3,)
        assert got[0] == pytest.approx(math.log10(0.5))
        assert got[1] == 0.0
        assert got[2] == pytest.approx(math.log10(1.25))

    def test_invalid_resolution_raises(self):
        for bad in (0, -64, math.nan, math.inf):
            with pytest.raises(ValueError):
                resolution_channel(bad)
        with pytest.raises(ValueError):
            resolution_channel(np.zeros((2, 2)))

    def test_cond_v4_names_extend_v3(self):
        assert COND_V4_CHANNEL_NAMES == COND_V3_CHANNEL_NAMES + (RESOLUTION_CHANNEL_NAME,)
        assert len(COND_V4_CHANNEL_NAMES) == 9


class TestConditionV4:
    N = 5

    def _inputs(self):
        rng = np.random.default_rng(7)
        re = 10.0 ** rng.uniform(1.7, 2.9, self.N)
        u_in = np.full(self.N, 0.1)
        sail = rng.uniform(0.4, 2.0, self.N)
        fin = rng.uniform(0.5, 2.0, self.N)
        geometry = rng.normal(size=(self.N, 4))
        return re, u_in, sail, fin, geometry

    def test_first_eight_columns_bit_identical_to_v3(self):
        re, u_in, sail, fin, geometry = self._inputs()
        v3 = condition_v3(re, u_in, sail, fin, geometry)
        v4 = condition_v4(re, u_in, sail, fin, geometry, 128)
        assert v4.shape == (self.N, 9)
        assert np.array_equal(v4[:, :8], v3)
        assert (v4[:, 8] == 0.0).all()

    def test_scalar_broadcast_and_per_point(self):
        re, u_in, sail, fin, geometry = self._inputs()
        broad = condition_v4(re, u_in, sail, fin, geometry, 64)
        per = condition_v4(re, u_in, sail, fin, geometry, np.full(self.N, 64.0))
        assert np.array_equal(broad, per)
        assert (broad[:, 8] == math.log10(0.5)).all()
        mixed = condition_v4(re, u_in, sail, fin, geometry, np.array([64, 128, 160, 64, 128]))
        assert mixed[2, 8] == pytest.approx(math.log10(1.25))

    def test_bad_resolution_shape_raises(self):
        re, u_in, sail, fin, geometry = self._inputs()
        with pytest.raises(ValueError):
            condition_v4(re, u_in, sail, fin, geometry, np.ones(self.N + 1))
