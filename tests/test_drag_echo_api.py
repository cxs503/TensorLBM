"""API tests for the drag-echo router (``/api/drag/echo``).

Exercises the FastAPI layer over
``tensorlbm.ai.geometry_pipeline.GeometryEchoPipeline`` with synthetic
backends (no /nfs dependency, no GPU): a tiny random-weight model ensemble
for the params/sweep/STL paths and a synthetic replay run dir for the
replay-specific behaviour (hull-form 404, STL refusal).  Mirrors
``tests/test_drag_surrogate_api.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# Make the ``backend`` package (under ``app/``) importable — mirrors
# tests/test_data_catalog_api.py.  ``src/`` is on sys.path via pythonpath=src.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _REPO_ROOT / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from backend.routers import drag_echo as echo_router  # noqa: E402

from tensorlbm.ai.drag_cond import (  # noqa: E402
    GEOMETRY_CHANNEL_NAMES,
    CondFNODrag,
    SuboffGrid,
    condition_v3,
    geometry_channels,
    suboff_geometry_features,
)
from tensorlbm.ai.geometry_pipeline import GeometryEchoPipeline  # noqa: E402
from tensorlbm.ai.inference_service import (  # noqa: E402
    CondDragCheckpoint,
    DragSurrogateService,
    EnvelopeMahalanobisGuardrail,
    ModelEnsembleBackend,
)

TEST_GRID = SuboffGrid.from_resolution(32)
ARCH_SMALL = dict(
    in_ch=5, width=16, n_layers=2, modes=(8, 16), mlp_hidden=64, film_hidden=32, cond_dim=8
)


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


def _guard_features() -> np.ndarray:
    rows = []
    for hull in ("bare_hull", "with_sail", "full"):
        for sail in (0.8, 1.0, 1.2):
            geo = geometry_channels(suboff_geometry_features(hull, sail, 1.0, grid=TEST_GRID))
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


def _model_pipeline() -> GeometryEchoPipeline:
    backend = ModelEnsembleBackend([_tiny_checkpoint(s) for s in range(2)], device="cpu")
    service = DragSurrogateService(
        backend, EnvelopeMahalanobisGuardrail(_guard_features()), grid=TEST_GRID
    )
    return GeometryEchoPipeline(service, grid=TEST_GRID, device="cpu")


def _replay_run_dir(tmp_path: Path) -> Path:
    """Minimal v4-layout replay run dir (full hull, two sail designs x 4 Re)."""
    res = np.array([50.0, 64.0, 81.0, 100.0])
    rng = np.random.default_rng(0)
    designs = [1.0, 1.2]
    geo_blocks = [
        geometry_channels(suboff_geometry_features("full", s, 1.0, grid=TEST_GRID)) for s in designs
    ]
    hulls: list[int] = []
    sails: list[float] = []
    cd_all: list[float] = []
    res_all: list[float] = []
    for sail in designs:
        cd = 20.0 * (res / 50.0) ** -0.45 * sail
        hulls += [2] * 4
        sails += [sail] * 4
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
        fin=np.ones(8),
        uin=np.full(8, 0.1),
        re=np.asarray(res_all),
        dsi=np.zeros(8, dtype=np.int64),
        cd=cd_arr,
    )
    np.savez(tmp_path / "cache_v3.npz", geo=np.stack([g for g in geo_blocks for _ in range(4)]))
    return tmp_path


def _replay_pipeline(tmp_path: Path) -> GeometryEchoPipeline:
    service = DragSurrogateService.from_run_dir(_replay_run_dir(tmp_path), grid=TEST_GRID)
    return GeometryEchoPipeline(service, grid=TEST_GRID, device="cpu")


def _client(pipeline: GeometryEchoPipeline) -> TestClient:
    def override() -> Iterator[GeometryEchoPipeline]:
        yield pipeline

    app = FastAPI()
    app.dependency_overrides[echo_router.get_echo_service] = override
    app.include_router(echo_router.router, prefix="/api/drag/echo")
    return TestClient(app)


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with _client(_model_pipeline()) as c:
        yield c


@pytest.fixture()
def replay_client(tmp_path: Path) -> Iterator[TestClient]:
    with _client(_replay_pipeline(tmp_path)) as c:
        yield c


def _sphere_stl_bytes() -> bytes:
    import math

    n_theta, n_phi = 10, 14

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
    return ("\n".join(lines) + "\n").encode()


class TestEchoHealth:
    def test_health(self, client: TestClient) -> None:
        r = client.get("/api/drag/echo/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["backend"] == "model"
        assert body["members"] == ["m0", "m1"]
        assert "log10_re" in body["guard_features"]
        assert body["guard_n_fit"] == 18
        assert body["grid"] == {"nz": 16, "ny": 16, "nx": 32}
        assert body["device"] == "cpu"
        # counts device follows the geo-fast auto rule (echo device when
        # CUDA, else any visible CUDA, else CPU)
        assert body["counts_device"] == ("cuda" if torch.cuda.is_available() else "cpu")
        assert isinstance(body["cache_entries"], int)


class TestEchoParams:
    def test_params_roundtrip(self, client: TestClient) -> None:
        r = client.post(
            "/api/drag/echo/params",
            json={
                "params": {
                    "hull_type": "full",
                    "sail_scale": 1.0,
                    "fin_scale": 1.0,
                    "u_in": 0.1,
                },
                "re_list": [60.0, 80.0],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["re"] == [60.0, 80.0]
        assert len(body["cd"]) == 2
        assert body["uq"]["lo"] <= body["uq"]["hi"]
        assert body["guard"]["flag"] == "ok"
        assert body["confident"] is True
        assert body["backend"] == "model"
        assert body["members"] == ["m0", "m1"]
        assert body["unsupported_channels"] == []
        assert body["info"]["timings_ms"]["total_s"] > 0.0
        assert body["info"]["geometry"]["sail_frac"] >= 0.0

    def test_hullform_variant_served_with_verdict(self, client: TestClient) -> None:
        r = client.post(
            "/api/drag/echo/params",
            json={"params": {"l_over_d_mult": 1.3, "sail_scale": 1.1}, "re_list": [70.0]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["info"]["hull_form_variant"] is True
        assert body["guard"]["flag"] in ("review", "reject")
        assert body["params"]["l_over_d_mult"] == 1.3

    def test_params_defaults_are_mother(self, client: TestClient) -> None:
        r = client.post("/api/drag/echo/params", json={"re_list": [70.0]})
        assert r.status_code == 200
        body = r.json()
        assert body["params"]["hull_type"] == "full"
        assert body["params"]["u_in"] == pytest.approx(0.1)
        assert body["info"]["hull_form_variant"] is False

    def test_validation_errors(self, client: TestClient) -> None:
        bad_hull = client.post(
            "/api/drag/echo/params",
            json={"params": {"hull_type": "triangle"}, "re_list": [60.0]},
        )
        assert bad_hull.status_code == 422
        bad_re = client.post("/api/drag/echo/params", json={"params": {}, "re_list": [-1.0]})
        assert bad_re.status_code == 422
        empty_re = client.post("/api/drag/echo/params", json={"params": {}, "re_list": []})
        assert empty_re.status_code == 422
        bad_scale = client.post(
            "/api/drag/echo/params", json={"params": {"sail_scale": 0.0}, "re_list": [60.0]}
        )
        assert bad_scale.status_code == 422


class TestEchoSweep:
    def test_sweep_roundtrip(self, client: TestClient) -> None:
        r = client.post(
            "/api/drag/echo/sweep",
            json={
                "axis": "sail_scale",
                "values": [0.8, 1.0, 1.2],
                "base_params": {"hull_type": "full"},
                "re_list": [60.0, 80.0],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["axis"] == "sail_scale"
        assert body["values"] == [0.8, 1.0, 1.2]
        assert len(body["results"]) == 3
        for res, sail in zip(body["results"], (0.8, 1.0, 1.2)):
            assert len(res["cd"]) == 2
            assert res["params"]["sail_scale"] == sail
            assert res["guard"]["flag"] in ("ok", "review", "reject")
            assert res["info"]["timings_ms"]["sweep_total_s"] > 0.0

    def test_sweep_u_in_axis(self, client: TestClient) -> None:
        r = client.post(
            "/api/drag/echo/sweep",
            json={
                "axis": "u_in",
                "values": [0.08, 0.12],
                "base_params": {"hull_type": "full"},
                "re_list": [60.0],
            },
        )
        assert r.status_code == 200
        results = r.json()["results"]
        assert [x["params"]["u_in"] for x in results] == [0.08, 0.12]

    def test_sweep_validation_errors(self, client: TestClient) -> None:
        bad_axis = client.post(
            "/api/drag/echo/sweep",
            json={"axis": "hull_type", "values": [1.0], "re_list": [60.0]},
        )
        assert bad_axis.status_code == 422
        bad_values = client.post(
            "/api/drag/echo/sweep",
            json={"axis": "sail_scale", "values": [0.0], "re_list": [60.0]},
        )
        assert bad_values.status_code == 422
        too_many = client.post(
            "/api/drag/echo/sweep",
            json={"axis": "sail_scale", "values": [1.0] * 257, "re_list": [60.0]},
        )
        assert too_many.status_code == 422


class TestEchoReplayBackend:
    def test_replay_params_ok(self, replay_client: TestClient) -> None:
        r = replay_client.post(
            "/api/drag/echo/params",
            json={"params": {"hull_type": "full"}, "re_list": [50.0, 64.0]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["backend"] == "replay"
        assert body["guard"]["flag"] == "ok"

    def test_replay_hullform_404(self, replay_client: TestClient) -> None:
        r = replay_client.post(
            "/api/drag/echo/params",
            json={"params": {"l_over_d_mult": 1.2}, "re_list": [50.0]},
        )
        assert r.status_code == 404
        assert "hull-form" in r.json()["detail"]

    def test_replay_stl_404(self, replay_client: TestClient) -> None:
        r = replay_client.post(
            "/api/drag/echo/stl",
            files={"file": ("sphere.stl", _sphere_stl_bytes(), "model/stl")},
            data={"re_list": "[60.0]", "u_in": "0.1", "hull_type": "full"},
        )
        assert r.status_code == 404
        assert "model ensemble backend" in r.json()["detail"]


class TestEchoStl:
    def test_stl_upload_flagged_not_confident(self, client: TestClient) -> None:
        r = client.post(
            "/api/drag/echo/stl",
            files={"file": ("sphere.stl", _sphere_stl_bytes(), "model/stl")},
            data={"re_list": "[60.0, 90.0]", "u_in": "0.1", "hull_type": "full"},
        )
        assert r.status_code == 200
        body = r.json()
        assert set(body["unsupported_channels"]) == set(GEOMETRY_CHANNEL_NAMES)
        assert body["guard"]["flag"] == "reject"
        assert body["confident"] is False
        assert any("not derivable" in reason for reason in body["guard"]["reasons"])
        assert body["info"]["cond_proxy"] == "mother_geometry"
        assert body["info"]["mask_counts"]["v"] > 0
        assert len(body["cd"]) == 2

    def test_stl_bad_re_list_422(self, client: TestClient) -> None:
        r = client.post(
            "/api/drag/echo/stl",
            files={"file": ("sphere.stl", _sphere_stl_bytes(), "model/stl")},
            data={"re_list": "not json"},
        )
        assert r.status_code == 422


class TestServiceUnavailable:
    def test_503_when_service_cannot_build(self) -> None:
        app = FastAPI()
        app.include_router(echo_router.router, prefix="/api/drag/echo")
        previous = (echo_router._pipeline, echo_router._pipeline_error)
        echo_router.set_echo_service(None, error="no artifacts on this host")
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.get("/api/drag/echo/health")
                assert r.status_code == 503
                assert "no artifacts" in r.json()["detail"]
        finally:
            echo_router.set_echo_service(*previous)


class TestRouterRegistration:
    def test_router_mounted_on_platform_app(self) -> None:
        """The platform application registers /api/drag/echo (skipped when the
        heavy platform import is unavailable in this environment).

        Asserted through ``_router_registry`` (what ``app.include_router``
        consumes) rather than ``app.routes`` — see
        ``tests/test_drag_surrogate_api.py`` for the starlette caveat.
        """
        try:
            import backend.main as main_mod  # noqa: F401
        except Exception as exc:  # pragma: no cover — depends on app deps
            pytest.skip(f"backend.main not importable here: {exc}")
        registered = {
            (getattr(mod, "__name__", ""), prefix)
            for mod, prefix, _tag in getattr(main_mod, "_router_registry", [])
        }
        assert ("backend.routers.drag_echo", "/api/drag/echo") in registered
        assert main_mod.drag_echo is not None
        paths = {getattr(r, "path", "") for r in main_mod.drag_echo.router.routes}
        assert paths == {"/health", "/params", "/sweep", "/stl"}
