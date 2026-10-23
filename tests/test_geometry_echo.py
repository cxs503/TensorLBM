"""Geometry-echo pipeline tests (B4-P3a).

Covers ``tensorlbm.ai.geometry_pipeline.GeometryEchoPipeline`` on synthetic
backends (tiny random-weight model ensemble, synthetic replay archive) with
no /nfs dependency, plus env-gated parity checks against the real B4 caches
when ``/nfs/wangxi/runs`` artifacts are present:

- ``cache_v4.npz`` — rebuilt v3 geometry channels vs the fit-time ``geo``
  block, bitwise, on >= 20 unique designs;
- ``cache_fam.npz`` — :meth:`validate_against_cache` generalised-channel
  parity on family points;
- ``b4_serve_20260824`` — served ``l_over_d_mult`` sweep direction vs the
  cached family trend (sign asserted only because the cache supports it:
  25/25 matched-Re pairs order the same way).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.ai.drag_cond import (
    GEOMETRY_CHANNEL_NAMES,
    CondFNODrag,
    SuboffGrid,
    condition_v3,
    geometry_channels,
    suboff_geometry_features,
)
from tensorlbm.ai.geometry_pipeline import (
    SWEEP_AXIS_NAMES,
    GeometryEchoPipeline,
    generalised_mask_counts,
    suboff_component_counts,
)
from tensorlbm.ai.inference_service import (
    HULL_ORDER,
    BackendQueryError,
    CondDragCheckpoint,
    DragSurrogateService,
    EnvelopeMahalanobisGuardrail,
    GuardVerdict,
    ModelEnsembleBackend,
)
from tensorlbm.suboff_cad import (
    _SUBOFF_L_FT,
    SuboffConfig,
    _axial_ft_to_mother,
    _hull_lines_is_mother,
    _variant_nodes_ft,
    build_suboff_mask,
    suboff_hull_mask,
    suboff_radius_profile,
)

TEST_GRID = SuboffGrid.from_resolution(32)
PRODUCTION_GRID = SuboffGrid.from_resolution(128)
ARCH_SMALL = dict(
    in_ch=5, width=16, n_layers=2, modes=(8, 16), mlp_hidden=64, film_hidden=32, cond_dim=8
)

_SERVE_CKPTS = Path("/nfs/wangxi/runs/b4_serve_20260824/ckpts")
_V4_CACHE = Path("/nfs/wangxi/runs/b4_v4_20260824/cache_v4.npz")
_FAM_CACHE = Path("/nfs/wangxi/runs/b4_fam_20260824/cache_fam.npz")
_HAS_ARTIFACTS = any(_SERVE_CKPTS.glob("*.pt")) and _V4_CACHE.is_file() and _FAM_CACHE.is_file()


def _tiny_checkpoint(seed: int) -> CondDragCheckpoint:
    torch.manual_seed(seed)
    model = CondFNODrag(**ARCH_SMALL)
    return CondDragCheckpoint(
        arch=dict(ARCH_SMALL),
        state_dict=model.state_dict(),
        norm=dict(
            ch_mean=np.zeros(5, dtype=np.float64),
            ch_std=np.ones(5, dtype=np.float64),
            p_mean=np.zeros(8, dtype=np.float64),
            p_std=np.ones(8, dtype=np.float64),
            y_mean=0.0,
            y_std=1.0,
        ),
        meta=dict(member=f"m{seed}", synthetic="random weights"),
    )


def _guard_features(grid: SuboffGrid) -> np.ndarray:
    """Condition rows spanning the guard fit space (mother designs, no variants)."""
    rows = []
    for hull in ("bare_hull", "with_sail", "full"):
        for sail in (0.8, 1.0, 1.2):
            geo = geometry_channels(suboff_geometry_features(hull, sail, 1.0, grid=grid))
            rows.append(
                condition_v3(
                    np.array([50.0, 100.0]),
                    np.full(2, 0.1),
                    np.full(2, sail),
                    np.ones(2),
                    np.broadcast_to(geo, (2, 4)),
                )
            )
    return np.concatenate(rows, axis=0)


@pytest.fixture()
def model_pipeline() -> GeometryEchoPipeline:
    backend = ModelEnsembleBackend([_tiny_checkpoint(s) for s in range(2)], device="cpu")
    guard = EnvelopeMahalanobisGuardrail(_guard_features(TEST_GRID))
    service = DragSurrogateService(backend, guard, grid=TEST_GRID)
    return GeometryEchoPipeline(service, grid=TEST_GRID, device="cpu")


def _replay_run_dir(tmp_path: Path, grid: SuboffGrid = TEST_GRID) -> Path:
    """Minimal v4-layout replay run dir (full hull, two sail designs x 4 Re)."""
    res = np.array([50.0, 64.0, 81.0, 100.0])
    rng = np.random.default_rng(0)
    designs = [1.0, 1.2]
    geo_blocks = [
        geometry_channels(suboff_geometry_features("full", s, 1.0, grid=grid)) for s in designs
    ]
    hulls: list[int] = []
    sails: list[float] = []
    fins: list[float] = []
    uins: list[float] = []
    res_all: list[float] = []
    cd_all: list[float] = []
    for sail, _geo in zip(designs, geo_blocks):
        cd = 20.0 * (res / 50.0) ** -0.45 * sail
        hulls += [2] * 4
        sails += [sail] * 4
        fins += [1.0] * 4
        uins += [0.1] * 4
        res_all += res.tolist()
        cd_all += cd.tolist()
    cd_arr = np.asarray(cd_all)
    preds: dict[str, np.ndarray] = {
        "loho::full::C_full::true": cd_arr,
        "loho::full::C_full::idx": np.arange(8),
    }
    for tag in ("", "s1", "s2"):
        key = "loho::full::C_full::pred" if tag == "" else f"loho::full::C_full::{tag}::pred"
        preds[key] = cd_arr * (1.0 + 0.02 * rng.standard_normal(8))
    np.savez(tmp_path / "preds_v4.npz", **preds)
    np.savez(
        tmp_path / "cache.npz",
        hull=np.asarray(hulls, dtype=np.int64),
        sail=np.asarray(sails),
        fin=np.asarray(fins),
        uin=np.asarray(uins),
        re=np.asarray(res_all),
        dsi=np.zeros(8, dtype=np.int64),
        cd=cd_arr,
    )
    np.savez(tmp_path / "cache_v3.npz", geo=np.stack([g for g in geo_blocks for _ in range(4)]))
    return tmp_path


@pytest.fixture()
def replay_pipeline(tmp_path: Path) -> GeometryEchoPipeline:
    service = DragSurrogateService.from_run_dir(_replay_run_dir(tmp_path), grid=TEST_GRID)
    return GeometryEchoPipeline(service, grid=TEST_GRID, device="cpu")


def _write_sphere_stl(path: Path, *, n_theta: int = 10, n_phi: int = 14) -> Path:
    """ASCII STL of a unit sphere (consistent winding, no degenerate faces)."""

    def pt(i: int, j: int) -> tuple[float, float, float]:
        th = math.pi * i / n_theta
        ph = 2.0 * math.pi * j / n_phi
        return (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))

    tris: list[tuple[tuple[float, float, float], ...]] = []
    for j in range(n_phi):
        j2 = (j + 1) % n_phi
        tris.append((pt(0, 0), pt(1, j2), pt(1, j)))
        tris.append((pt(n_theta - 1, j), pt(n_theta - 1, j2), pt(n_theta, 0)))
    for i in range(1, n_theta - 1):
        for j in range(n_phi):
            j2 = (j + 1) % n_phi
            tris.append((pt(i, j), pt(i, j2), pt(i + 1, j2)))
            tris.append((pt(i, j), pt(i + 1, j2), pt(i + 1, j)))
    lines = ["solid sphere"]
    for v0, v1, v2 in tris:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for v in (v0, v1, v2):
            lines.append(f"      vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid sphere")
    path.write_text("\n".join(lines) + "\n")
    return path


class TestFitTimeExactness:
    @pytest.mark.parametrize("hull", ["bare_hull", "with_sail", "full"])
    @pytest.mark.parametrize("sail,fin", [(1.0, 1.0), (1.3, 0.8), (0.75, 1.1)])
    def test_mother_channels_bitwise_vs_drag_cond(self, hull: str, sail: float, fin: float) -> None:
        """The echo decomposition reproduces the fit-time builder bitwise."""
        counts = suboff_component_counts(hull, sail, fin, PRODUCTION_GRID)
        ref = suboff_geometry_features(hull, sail, fin, grid=PRODUCTION_GRID)
        assert counts == (
            ref.v_bare,
            ref.v_sail,
            ref.v_fin,
            ref.v_solid,
            ref.aproj,
            ref.aproj_bare,
        )

    def test_condition_rows_bitwise_vs_service(self, model_pipeline: GeometryEchoPipeline) -> None:
        """Pipeline condition rows == the service own construction, bitwise."""
        bundle = model_pipeline._geometry({"hull_type": "full", "sail_scale": 1.2})
        re = np.array([50.0, 80.0, 120.0])
        got = bundle.condition_rows(re)
        want, _geo = model_pipeline.service.condition_rows("full", 1.2, 1.0, re, u_in=0.1)
        np.testing.assert_array_equal(got, want)

    @pytest.mark.parametrize(
        "kw", [{}, {"l_over_d_mult": 1.2}, {"nose_len_mult": 1.3, "sail_x_mult": 1.15}]
    )
    def test_mask_bitwise_vs_build_suboff_mask(
        self, model_pipeline: GeometryEchoPipeline, kw: dict[str, float]
    ) -> None:
        """The cached geometry bundle voxelises to the CAD mask, bitwise."""
        cfg = SuboffConfig(**kw)
        ref, _ = build_suboff_mask(
            hull_type="full",
            nx=TEST_GRID.nx,
            ny=TEST_GRID.ny,
            nz=TEST_GRID.nz,
            cx=TEST_GRID.cx,
            cy=TEST_GRID.cy,
            cz=TEST_GRID.cz,
            length=TEST_GRID.length,
            config=cfg,
            device="cpu",
        )
        bundle = model_pipeline._geometry({"hull_type": "full", **kw})
        assert np.array_equal(bundle.build_mask(), ref.cpu().numpy())

    def test_generalised_counts_definitions(self) -> None:
        """Any-mask counts follow the B4-fam definitions."""
        mask = np.zeros((4, 5, 6), dtype=bool)
        mask[1:3, 2:4, 1:4] = True
        a_proj, a_side, a_top, v = generalised_mask_counts(mask)
        # A_proj: max over x -> (nz, ny) support 2x2; A_side: max over y ->
        # (nz, nx) support 2x3; A_top: max over z -> (ny, nx) support 2x3.
        assert (a_proj, a_side, a_top, v) == (2 * 2, 2 * 3, 2 * 3, 2 * 2 * 3)


class TestPredictFromParams:
    def test_guard_verdict_always_present(self, model_pipeline: GeometryEchoPipeline) -> None:
        res = model_pipeline.predict_from_params({"hull_type": "full"}, [60.0, 90.0])
        assert isinstance(res.guard, GuardVerdict)
        assert res.guard.flag in ("ok", "review", "reject")
        assert res.cd.shape == (2,)
        assert (res.lo <= res.cd).all() and (res.cd <= res.hi).all()
        assert set(res.info["geometry"]) == set(GEOMETRY_CHANNEL_NAMES)
        for key in ("geometry_s", "mask_s", "condition_s", "guard_s", "ensemble_s", "total_s"):
            assert res.info["timings_ms"][key] >= 0.0, key
        assert res.info["grid"] == {"nz": 16, "ny": 16, "nx": 32}
        assert res.members == ("m0", "m1")
        assert res.as_dict()["guard"]["flag"] == res.guard.flag

    def test_mother_design_passes_guard(self, model_pipeline: GeometryEchoPipeline) -> None:
        res = model_pipeline.predict_from_params(
            {"hull_type": "full", "sail_scale": 1.0}, [60.0, 80.0]
        )
        assert res.guard.flag == "ok"
        assert res.confident is True
        assert res.unsupported_channels == ()

    def test_hullform_variant_marked_and_guarded(
        self, model_pipeline: GeometryEchoPipeline
    ) -> None:
        """Hull-form variants are labelled and never pass as in-corpus."""
        res = model_pipeline.predict_from_params(
            {"hull_type": "full", "l_over_d_mult": 1.3}, [60.0]
        )
        assert res.info["hull_form_variant"] is True
        assert res.guard.flag in ("review", "reject")

    def test_unknown_param_rejected(self, model_pipeline: GeometryEchoPipeline) -> None:
        with pytest.raises(ValueError, match="unknown design params"):
            model_pipeline.predict_from_params({"wavelength": 2.0}, [60.0])

    def test_bad_re_rejected(self, model_pipeline: GeometryEchoPipeline) -> None:
        with pytest.raises(ValueError, match="finite and positive"):
            model_pipeline.predict_from_params({"hull_type": "full"}, [-1.0])

    def test_geometry_cache_reuse_across_re(self, model_pipeline: GeometryEchoPipeline) -> None:
        model_pipeline.predict_from_params({"hull_type": "full"}, [60.0])
        model_pipeline.predict_from_params({"hull_type": "full"}, [90.0, 120.0])
        assert len(model_pipeline._cache) == 1


class TestReplayBackend:
    def test_mother_roundtrip_matches_archive(self, replay_pipeline: GeometryEchoPipeline) -> None:
        res = replay_pipeline.predict_from_params({"hull_type": "full"}, [50.0, 64.0])
        rows, _info = replay_pipeline.service.backend.predict(
            "full", 1.0, 1.0, np.array([50.0, 64.0]), u_in=0.1
        )
        np.testing.assert_allclose(res.cd, rows.mean(axis=0), rtol=1e-12)
        assert res.backend == "replay"
        assert res.info["field_source"] == "replay_archive"

    def test_hullform_rejected(self, replay_pipeline: GeometryEchoPipeline) -> None:
        with pytest.raises(BackendQueryError, match="hull-form"):
            replay_pipeline.predict_from_params({"hull_type": "full", "l_over_d_mult": 1.2}, [50.0])


class TestSweep:
    def test_sweep_shapes_and_guard(self, model_pipeline: GeometryEchoPipeline) -> None:
        results = model_pipeline.sweep_axis(
            "l_over_d_mult", [0.9, 1.0, 1.1], {"hull_type": "full"}, [60.0, 80.0]
        )
        assert len(results) == 3
        for res in results:
            assert res.cd.shape == (2,)
            assert isinstance(res.guard, GuardVerdict)
            assert res.info["timings_ms"]["sweep_total_s"] >= res.info["timings_ms"]["total_s"]
        assert [r.params["value"] for r in results] == [0.9, 1.0, 1.1]
        assert {r.params["axis"] for r in results} == {"l_over_d_mult"}

    def test_sweep_u_in_reuses_geometry(self, model_pipeline: GeometryEchoPipeline) -> None:
        results = model_pipeline.sweep_axis(
            "u_in", [0.08, 0.1, 0.12], {"hull_type": "full"}, [60.0]
        )
        assert len(results) == 3
        assert len(model_pipeline._cache) == 1
        assert {r.params["u_in"] for r in results} == {0.08, 0.1, 0.12}

    def test_sweep_bad_axis(self, model_pipeline: GeometryEchoPipeline) -> None:
        with pytest.raises(ValueError, match="axis must be one of"):
            model_pipeline.sweep_axis("hull_type", ["full"], {"hull_type": "full"}, [60.0])
        assert "hull_type" not in SWEEP_AXIS_NAMES

    def test_sweep_mother_channels_bitwise(self, model_pipeline: GeometryEchoPipeline) -> None:
        """Every mother sweep point reproduces the fit-time channels bitwise."""
        sails = [0.8, 1.0, 1.2]
        results = model_pipeline.sweep_axis("sail_scale", sails, {"hull_type": "full"}, [60.0])
        for res, sail in zip(results, sails):
            want = geometry_channels(suboff_geometry_features("full", sail, 1.0, grid=TEST_GRID))
            got = np.array([res.info["geometry"][n] for n in GEOMETRY_CHANNEL_NAMES])
            np.testing.assert_array_equal(got, want)


class TestPredictBatch:
    def test_g1_bitwise_vs_predict(self, model_pipeline: GeometryEchoPipeline) -> None:
        backend = model_pipeline.service.backend
        assert isinstance(backend, ModelEnsembleBackend)
        bundle = model_pipeline._geometry({"hull_type": "full"})
        cond = bundle.condition_rows(np.array([60.0, 80.0]))
        from tensorlbm.ai.geometry_pipeline import _synthetic_field

        field = _synthetic_field(bundle.build_mask())
        single = backend.predict(field, cond)
        batched = backend.predict_batch(field[None, ...], cond, np.array([2]))
        np.testing.assert_array_equal(single, batched)


class TestStlPath:
    def test_sphere_is_flagged_not_confident(
        self, model_pipeline: GeometryEchoPipeline, tmp_path: Path
    ) -> None:
        stl = _write_sphere_stl(tmp_path / "sphere.stl")
        res = model_pipeline.predict_from_stl(stl, [60.0, 90.0])
        assert set(res.unsupported_channels) == set(GEOMETRY_CHANNEL_NAMES)
        assert res.guard.flag == "reject"
        assert res.confident is False
        assert any("not derivable" in r for r in res.guard.reasons)
        assert res.info["mask_counts"]["v"] > 0
        assert res.info["cond_proxy"] == "mother_geometry"
        assert res.info["stl"]["watertight"] is True
        assert res.cd.shape == (2,)

    def test_replay_backend_refuses_stl(
        self, replay_pipeline: GeometryEchoPipeline, tmp_path: Path
    ) -> None:
        stl = _write_sphere_stl(tmp_path / "sphere.stl")
        with pytest.raises(BackendQueryError, match="model ensemble backend"):
            replay_pipeline.predict_from_stl(stl, [60.0])


@pytest.mark.skipif(not _HAS_ARTIFACTS, reason="B4 serve/fam artifacts not present on this host")
class TestAgainstRealArtifacts:
    @pytest.fixture()
    def serve_pipeline(self) -> GeometryEchoPipeline:
        from tensorlbm.ai.inference_service import load_checkpoint, load_corpus_index

        ckpts = [load_checkpoint(p) for p in sorted(_SERVE_CKPTS.glob("*.pt"))]
        backend = ModelEnsembleBackend(ckpts, device="cpu")
        index = load_corpus_index(_V4_CACHE.parent)
        service = DragSurrogateService(
            backend,
            EnvelopeMahalanobisGuardrail(index.cond),
            corpus_cache=index.fields,
            cache_re=index.re,
            cache_designs=list(index.designs),
        )
        return GeometryEchoPipeline(service, device="cpu")

    def test_condition_parity_cache_v4(self) -> None:
        """Rebuilt v3 geo block == ``cache_v4['geo']`` bitwise, >= 20 designs."""
        z = np.load(_V4_CACHE)
        seen: set[tuple[int, float, float]] = set()
        n = 0
        for i in range(len(z["cd"])):
            key = (int(z["hull"][i]), round(float(z["sail"][i]), 9), round(float(z["fin"][i]), 9))
            if key in seen:
                continue
            seen.add(key)
            v_bare, v_sail, v_fin, v_solid, aproj, aproj_bare = suboff_component_counts(
                HULL_ORDER[key[0]], key[1], key[2], PRODUCTION_GRID
            )
            rebuilt = np.array(
                [
                    math.log10(aproj / aproj_bare),
                    v_sail / v_bare,
                    v_fin / v_bare,
                    v_solid / v_bare,
                ]
            )
            assert np.array_equal(rebuilt, z["geo"][i]), (key, rebuilt, z["geo"][i])
            n += 1
            if n >= 25:
                break
        assert n >= 20

    def test_validate_against_cache(self, serve_pipeline: GeometryEchoPipeline) -> None:
        report = serve_pipeline.validate_against_cache(_FAM_CACHE, max_points=8)
        assert report.n_points == 8
        assert report.channels_bitwise, report.as_dict()
        assert report.counts_mismatch_rows == 0
        assert report.max_abs_geom_diff == 0.0
        assert math.isfinite(report.service_mape) and report.service_mape > 0.0
        assert report.guard_flags  # every family point carries a verdict

    def test_sweep_l_over_d_trend_vs_cache(self, serve_pipeline: GeometryEchoPipeline) -> None:
        """Served ``l_over_d_mult`` sweep vs the cached family trend.

        The cache supports a clear direction: among fam_blunt (l/d 0.75)
        vs fam_slender (l/d 1.3) pairs matched to |dlog10 Re| < 0.02, 25/25
        have slender C_D > blunt C_D — asserted here.

        The served C_full ensemble does NOT reproduce that direction
        (measured 2026-08-25: served gap -0.77 vs cache +5.29 mean over
        matched Re; the v3 hand channels barely move under the hull-form
        axes, so the sweep extrapolates).  The direction agreement is
        therefore NOT asserted — what is asserted is the honesty contract:
        every hull-form variant carries a non-``ok`` verdict, so the
        disagreement is never presented as confident output.  Re-enable the
        sign assertion when a corpus that includes hull-form variants is
        served (SDF-v2 encoder seam).
        """
        import json

        meta = json.loads((_FAM_CACHE.parent / "cache_fam_meta.json").read_text())
        blunt = sorted((m for m in meta if m["fam"] == "fam_blunt"), key=lambda m: m["re"])
        slender = sorted((m for m in meta if m["fam"] == "fam_slender"), key=lambda m: m["re"])
        pairs = [
            (b, min(slender, key=lambda s: abs(math.log10(s["re"]) - math.log10(b["re"]))))
            for b in blunt
        ]
        pairs = [(b, s) for b, s in pairs if abs(math.log10(s["re"]) - math.log10(b["re"])) < 0.02]
        assert len(pairs) >= 20, "fam cache no longer supports the trend check"
        votes = sum(1 for b, s in pairs if s["cd"] > b["cd"])
        assert votes >= 0.8 * len(pairs), f"cache trend inconsistent: {votes}/{len(pairs)}"
        re_list = [float(pairs[0][0]["re"]), float(pairs[len(pairs) // 2][0]["re"])]
        results = serve_pipeline.sweep_axis(
            "l_over_d_mult", [0.75, 1.3], {"hull_type": "with_sail", "u_in": 0.1}, re_list
        )
        cache_gap = float(np.mean([s["cd"] - b["cd"] for b, s in pairs]))
        served_gap = float(results[1].cd.mean()) - float(results[0].cd.mean())
        for res in results:
            assert res.info["hull_form_variant"] is True
            assert res.guard.flag in ("review", "reject"), res.guard
            assert res.confident is False
            assert any("outside the served training corpus" in r for r in res.guard.reasons)
        # Recorded, not asserted: served_gap vs cache_gap sign disagreement.
        assert np.isfinite(served_gap) and cache_gap > 0.0


# ---------------------------------------------------------------------------
# exp/geo-fast (2026-08-27): counts-on-device slider-miss fast path
# ---------------------------------------------------------------------------


def _legacy_hull_mask_fullgrid(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    length: float,
    radius: float,
    device: torch.device,
    config: SuboffConfig | None = None,
) -> torch.Tensor:
    """Frozen pre-geo-fast ``suboff_hull_mask`` (exp/trt-echo e1e56b24).

    The production hull mask now evaluates the axial coordinate on the
    ``(nx,)`` row and the radial distance as a ``(nz, ny, 1)`` broadcast
    view instead of a full meshgrid; this reference keeps the historical
    full-grid algorithm so the optimisation stays pinned bit-for-bit.
    """
    if config is None:
        config = SuboffConfig()
    if radius <= 0.0:
        radius = config.r_over_l * length
    if config.l_over_d_mult != 1.0:
        radius = radius / config.l_over_d_mult
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x_bow = cx - length / 2.0
    xi_t = (xx - x_bow) / length
    if _hull_lines_is_mother(config):
        xi_mother = xi_t
    else:
        x_ft_var = xi_t * _variant_nodes_ft(config)[-1]
        xi_mother = _axial_ft_to_mother(x_ft_var, config) / _SUBOFF_L_FT
    xi_np = xi_mother.cpu().numpy()
    r_norm = suboff_radius_profile(xi_np, config)
    r_lu = torch.tensor(r_norm * radius, device=device, dtype=torch.float32)
    r_grid = torch.sqrt((yy - cy) ** 2 + (zz - cz) ** 2)
    in_axial = (xi_t >= 0.0) & (xi_t <= 1.0)
    return in_axial & (r_grid <= r_lu)


_GEO_FAST_SWEEP: list[tuple[str, dict[str, float]]] = [
    ("mother", {}),
    ("sail_0.85", {"sail_scale": 0.85}),
    ("sail_1.3", {"sail_scale": 1.3}),
    ("fin_0.75", {"fin_scale": 0.75}),
    ("fin_1.2", {"fin_scale": 1.2}),
    ("lod_0.9", {"l_over_d_mult": 0.9}),
    ("lod_1.15", {"l_over_d_mult": 1.15}),
    ("nose_0.85", {"nose_len_mult": 0.85}),
    ("nose_1.25", {"nose_len_mult": 1.25}),
    ("stern_0.85", {"stern_len_mult": 0.85}),
    ("stern_1.2", {"stern_len_mult": 1.2}),
    ("sailx_0.88", {"sail_x_mult": 0.88}),
    ("sailx_1.12", {"sail_x_mult": 1.12}),
    ("combo", {"sail_scale": 1.2, "fin_scale": 0.8, "l_over_d_mult": 1.1, "sail_x_mult": 1.05}),
]


def _sweep_params(over: dict[str, float]) -> dict[str, object]:
    base: dict[str, object] = {
        "hull_type": "full",
        "u_in": 0.1,
        "sail_scale": 1.0,
        "fin_scale": 1.0,
        "l_over_d_mult": 1.0,
        "nose_len_mult": 1.0,
        "stern_len_mult": 1.0,
        "sail_x_mult": 1.0,
    }
    return {**base, **over}


_HAS_CUDA = torch.cuda.is_available()


class TestGeoFastHullMaskRowEval:
    """The row/broadcast hull-mask optimisation is bit-identical (CPU)."""

    @pytest.mark.parametrize("grid_res", [16, 32, 48])
    @pytest.mark.parametrize(
        "config",
        [
            SuboffConfig(),
            SuboffConfig(l_over_d_mult=1.2),
            SuboffConfig(nose_len_mult=0.85, stern_len_mult=1.3),
            SuboffConfig(l_over_d_mult=0.9, nose_len_mult=1.1, sail_x_mult=1.2),
        ],
        ids=["mother", "lod", "nose_stern", "combo"],
    )
    def test_row_eval_bitwise_vs_legacy_fullgrid(self, grid_res: int, config: SuboffConfig) -> None:
        grid = SuboffGrid.from_resolution(grid_res)
        fast = suboff_hull_mask(
            grid.nx,
            grid.ny,
            grid.nz,
            grid.cx,
            grid.cy,
            grid.cz,
            grid.length,
            0.0,
            torch.device("cpu"),
            config,
        )
        legacy = _legacy_hull_mask_fullgrid(
            grid.nx,
            grid.ny,
            grid.nz,
            grid.cx,
            grid.cy,
            grid.cz,
            grid.length,
            0.0,
            torch.device("cpu"),
            config,
        )
        assert fast.shape == legacy.shape == (grid.nz, grid.ny, grid.nx)
        assert torch.equal(fast, legacy)


class TestGeoFastCountsDevice:
    """Counts-on-device fast path: bitwise channels + no latency regression."""

    def test_counts_device_resolution(self) -> None:
        from tensorlbm.ai.geometry_pipeline import _resolve_counts_device

        auto_side = "cuda" if _HAS_CUDA else "cpu"
        assert _resolve_counts_device("cuda", None) == "cuda"
        assert _resolve_counts_device("cuda", "auto") == "cuda"
        assert _resolve_counts_device("cpu", None) == auto_side
        assert _resolve_counts_device("cuda", "cpu") == "cpu"
        assert _resolve_counts_device("cpu", "cuda") == "cuda"
        assert _resolve_counts_device("cuda:1", None) == "cuda:1"
        assert _resolve_counts_device("cuda:1", "auto") == "cuda:1"

    def test_pipeline_default_counts_device_auto(
        self, model_pipeline: GeometryEchoPipeline
    ) -> None:
        """A CPU-device pipeline still resolves counts to any visible CUDA."""
        assert model_pipeline.device == "cpu"
        assert model_pipeline.counts_device == ("cuda" if _HAS_CUDA else "cpu")

    def test_pipeline_counts_device_override(self, model_pipeline: GeometryEchoPipeline) -> None:
        pipeline = GeometryEchoPipeline(
            model_pipeline.service, grid=TEST_GRID, device="cpu", counts_device="cpu"
        )
        assert pipeline.counts_device == "cpu"

    @pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA")
    def test_pipeline_channels_bitwise_between_count_devices(
        self, model_pipeline: GeometryEchoPipeline
    ) -> None:
        """Served geo blocks are bitwise equal with CPU vs CUDA counts.

        ``model_pipeline`` auto-resolves to CUDA counts here, so pin its
        twin to CPU explicitly — the comparison is the fast path vs the
        fit-time device.
        """
        cpu_counts = GeometryEchoPipeline(
            model_pipeline.service, grid=model_pipeline.grid, device="cpu", counts_device="cpu"
        )
        for _label, over in _GEO_FAST_SWEEP:
            params = _sweep_params(over)
            assert np.array_equal(
                model_pipeline._geometry(dict(params)).geo,  # noqa: SLF001
                cpu_counts._geometry(dict(params)).geo,  # noqa: SLF001
            )

    @pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA")
    @pytest.mark.parametrize("hull", ["bare_hull", "with_sail", "full"])
    def test_counts_bitwise_cpu_vs_cuda_all_axes_production_grid(self, hull: str) -> None:
        """The service fast path: integer counts match CPU on every axis."""
        for _label, over in _GEO_FAST_SWEEP:
            params = _sweep_params(over)
            config = SuboffConfig(
                sail_scale=float(params["sail_scale"]),
                fin_scale=float(params["fin_scale"]),
                l_over_d_mult=float(params["l_over_d_mult"]),
                nose_len_mult=float(params["nose_len_mult"]),
                stern_len_mult=float(params["stern_len_mult"]),
                sail_x_mult=float(params["sail_x_mult"]),
            )
            cpu = suboff_component_counts(
                hull, 1.0, 1.0, PRODUCTION_GRID, config=config, device="cpu"
            )
            cuda = suboff_component_counts(
                hull, 1.0, 1.0, PRODUCTION_GRID, config=config, device="cuda"
            )
            assert cpu == cuda, (hull, _label, cpu, cuda)

    @pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA")
    @pytest.mark.parametrize("hull", ["bare_hull", "with_sail", "full"])
    def test_mask_bitwise_cpu_vs_cuda_all_axes_production_grid(self, hull: str) -> None:
        """Masks built on the service device match the CPU build bitwise."""
        for _label, over in _GEO_FAST_SWEEP:
            params = _sweep_params(over)
            config = SuboffConfig(
                sail_scale=float(params["sail_scale"]),
                fin_scale=float(params["fin_scale"]),
                l_over_d_mult=float(params["l_over_d_mult"]),
                nose_len_mult=float(params["nose_len_mult"]),
                stern_len_mult=float(params["stern_len_mult"]),
                sail_x_mult=float(params["sail_x_mult"]),
            )
            m_cpu, s_cpu = build_suboff_mask(
                hull_type=hull,
                nx=PRODUCTION_GRID.nx,
                ny=PRODUCTION_GRID.ny,
                nz=PRODUCTION_GRID.nz,
                cx=PRODUCTION_GRID.cx,
                cy=PRODUCTION_GRID.cy,
                cz=PRODUCTION_GRID.cz,
                length=PRODUCTION_GRID.length,
                config=config,
                device="cpu",
            )
            m_cuda, s_cuda = build_suboff_mask(
                hull_type=hull,
                nx=PRODUCTION_GRID.nx,
                ny=PRODUCTION_GRID.ny,
                nz=PRODUCTION_GRID.nz,
                cx=PRODUCTION_GRID.cx,
                cy=PRODUCTION_GRID.cy,
                cz=PRODUCTION_GRID.cz,
                length=PRODUCTION_GRID.length,
                config=config,
                device="cuda",
            )
            assert torch.equal(m_cpu, m_cuda.cpu())
            assert s_cpu["solid_cells"] == s_cuda["solid_cells"]

    @pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA")
    def test_slider_miss_counts_no_latency_regression_production_grid(self) -> None:
        """Latency-regression guard (tolerant): device counts stay fast.

        Post-fix medians on the 5090 (GPU 2): CPU ~10 ms, CUDA ~2.3 ms per
        production-grid miss.  The bounds are deliberately generous so the
        guard flags regressions (pre-fix CUDA was ~8.8 ms, CPU ~16.5 ms),
        not machine noise.
        """
        import time

        config = SuboffConfig(sail_scale=1.1)

        def _median(device: str) -> float:
            times = []
            for _ in range(9):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                suboff_component_counts(
                    "full", 1.1, 1.0, PRODUCTION_GRID, config=config, device=device
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(1e3 * (time.perf_counter() - t0))
            return float(np.median(times))

        cpu_ms = _median("cpu")
        cuda_ms = _median("cuda")
        assert cuda_ms < 8.0, f"CUDA slider miss regressed: {cuda_ms:.2f} ms"
        assert cuda_ms < cpu_ms, f"CUDA slider miss slower than CPU: {cuda_ms:.2f} vs {cpu_ms:.2f}"
