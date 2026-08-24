"""API tests for the drag-surrogate router (``/api/drag``).

Exercises the FastAPI layer over ``tensorlbm.ai.inference_service`` with a
synthetic replay run directory (no /nfs dependency), plus a registration
check against the real platform application in ``backend.main`` — the same
shape as ``tests/test_data_catalog_api.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# Make the ``backend`` package (under ``app/``) importable — mirrors
# tests/test_data_catalog_api.py.  ``src/`` is on sys.path via pythonpath=src.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _REPO_ROOT / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from backend.routers import drag_surrogate as drag_router  # noqa: E402

from tensorlbm.ai.drag_cond import (  # noqa: E402
    SuboffGrid,
    geometry_channels,
    suboff_geometry_features,
)
from tensorlbm.ai.inference_service import DragSurrogateService  # noqa: E402

TEST_GRID = SuboffGrid.from_resolution(32)


def _syn_run_dir(tmp_path: Path) -> Path:
    """Minimal v4-layout replay run directory (full hull, Re sweep)."""
    res = [50.0, 64.0, 81.0, 100.0]
    geo = geometry_channels(suboff_geometry_features("full", 1.0, 1.0, grid=TEST_GRID))
    rng = np.random.default_rng(0)
    cd = 20.0 * (np.array(res) / 50.0) ** -0.45
    preds: dict[str, np.ndarray] = {
        "loho::full::C_full::true": cd,
        "loho::full::C_full::idx": np.arange(4),
    }
    for tag in ("", "s1", "s2"):
        key = "loho::full::C_full::pred" if tag == "" else f"loho::full::C_full::{tag}::pred"
        preds[key] = cd * (1.0 + 0.02 * rng.standard_normal(4))
    np.savez(tmp_path / "preds_v4.npz", **preds)
    np.savez(
        tmp_path / "cache.npz",
        hull=np.array([2, 2, 2, 2], dtype=np.int64),
        sail=np.ones(4),
        fin=np.ones(4),
        uin=np.full(4, 0.1),
        re=np.array(res),
        dsi=np.zeros(4, dtype=np.int64),
        cd=cd,
    )
    np.savez(
        tmp_path / "cache_v3.npz",
        geo=np.stack([geo] * 4),
    )
    return tmp_path


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    run_dir = _syn_run_dir(tmp_path)
    service = DragSurrogateService.from_run_dir(run_dir, grid=TEST_GRID)

    def override() -> Iterator[DragSurrogateService]:
        yield service

    app = FastAPI()
    app.dependency_overrides[drag_router.get_drag_service] = override
    app.include_router(drag_router.router, prefix="/api/drag")
    with TestClient(app) as c:
        yield c


class TestDragEndpoints:
    def test_health(self, client: TestClient) -> None:
        r = client.get("/api/drag/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["backend"] == "replay"
        assert body["members"] == ["s0", "s1", "s2"]
        assert "log10_re" in body["guard_features"]
        assert body["guard_n_fit"] == 4

    def test_drag_curve_roundtrip(self, client: TestClient) -> None:
        r = client.post(
            "/api/drag",
            json={
                "hull_type": "full",
                "sail_scale": 1.0,
                "fin_scale": 1.0,
                "re_grid": [50.0, 70.0, 100.0],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["re"] == [50.0, 70.0, 100.0]
        assert len(body["cd"]) == 3
        assert body["uq"]["lo"] <= body["uq"]["hi"]
        assert body["guard"]["flag"] == "ok"
        assert body["backend"] == "replay"
        assert body["info"]["mode"] == "log_re_interp"
        assert body["info"]["n_exact"] == 2

    def test_extrapolated_re_is_flagged_but_served(self, client: TestClient) -> None:
        r = client.post(
            "/api/drag",
            json={
                "hull_type": "full",
                "sail_scale": 1.0,
                "fin_scale": 1.0,
                "re_grid": [5000.0],
            },
        )
        assert r.status_code == 200  # flagged, not refused
        assert r.json()["guard"]["flag"] == "reject"
        assert any("log10_re" in reason for reason in r.json()["guard"]["reasons"])

    def test_unknown_design_404(self, client: TestClient) -> None:
        r = client.post(
            "/api/drag",
            json={
                "hull_type": "full",
                "sail_scale": 9.9,
                "fin_scale": 1.0,
                "re_grid": [64.0],
            },
        )
        assert r.status_code == 404
        assert "not present" in r.json()["detail"]

    def test_validation_errors(self, client: TestClient) -> None:
        bad_hull = client.post(
            "/api/drag",
            json={"hull_type": "triangle", "sail_scale": 1.0, "fin_scale": 1.0, "re_grid": [64.0]},
        )
        assert bad_hull.status_code == 422
        bad_re = client.post(
            "/api/drag",
            json={"hull_type": "full", "sail_scale": 1.0, "fin_scale": 1.0, "re_grid": [-1.0]},
        )
        assert bad_re.status_code == 422
        empty = client.post(
            "/api/drag",
            json={"hull_type": "full", "sail_scale": 1.0, "fin_scale": 1.0, "re_grid": []},
        )
        assert empty.status_code == 422


class TestServiceUnavailable:
    def test_503_when_service_cannot_build(self) -> None:
        app = FastAPI()
        app.include_router(drag_router.router, prefix="/api/drag")
        previous = (drag_router._service, drag_router._service_error)
        drag_router.set_drag_service(None, error="no artifacts on this host")
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.get("/api/drag/health")
                assert r.status_code == 503
                assert "no artifacts" in r.json()["detail"]
        finally:
            drag_router.set_drag_service(*previous)


class TestRouterRegistration:
    def test_router_mounted_on_platform_app(self) -> None:
        """The platform application registers /api/drag (skipped when the
        heavy platform import is unavailable in this environment).

        Asserted through ``_router_registry`` (what ``app.include_router``
        consumes) rather than ``app.routes`` — newer starlette versions
        expose included routers as ``_IncludedRouter`` objects without a
        ``.path`` attribute (the pre-existing failure mode of
        ``test_data_catalog_api.py::test_router_registered_in_main_app``
        in the tensorlbm venv).
        """
        try:
            import backend.main as main_mod  # noqa: F401
        except Exception as exc:  # pragma: no cover — depends on app deps
            pytest.skip(f"backend.main not importable here: {exc}")
        registered = {
            (getattr(mod, "__name__", ""), prefix)
            for mod, prefix, _tag in getattr(main_mod, "_router_registry", [])
        }
        assert ("backend.routers.drag_surrogate", "/api/drag") in registered
        assert main_mod.drag_surrogate is not None
        # Endpoint paths on the router itself (prefix applied at include time).
        paths = {getattr(r, "path", "") for r in main_mod.drag_surrogate.router.routes}
        assert paths == {"", "/health"}
