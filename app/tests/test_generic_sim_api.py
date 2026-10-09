"""Tests for the generic simulation API (``/api/sim/generic``).

Covers the acceptance criteria of the generic-run fusion
(``PLATFORM_ANALYSIS.md`` §4.2 + the PR #180 benchmark compile-route
standard):

1.  Registry discovery (``GET /api/sim/generic/cases``).
2.  Small real cases run through the endpoint and return diagnostics,
    the compile-route audit trail and the common-module manifest.
3.  Invalid input is 4xx: unknown case, unknown params, negative steps,
    bad collision, and cudagraph-class compile modes rejected with the
    shared structural reason.
4.  All three compile modes route as in the benchmark suite (``eager``
    passthrough, ``default``, ``max-autotune-no-cudagraphs``).
5.  Parity: the cavity case reproduces the verified benchmark
    ``benchmarks/verified/cavity/re100/run.py`` bit-for-bit on the same
    small grid (identical common-module chain).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Registry discovery
# ---------------------------------------------------------------------------


def test_list_cases(client):
    r = client.get("/api/sim/generic/cases")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 5
    for case in ("cavity", "poiseuille", "couette", "shear_wave", "cylinder"):
        assert case in body["cases"]
        entry = body["cases"][case]
        assert entry["grid"] and entry["physics"]
        assert entry["collision_default"] in ("bgk", "mrt")
        assert entry["default_steps"] > 0
    assert "compile_route" in body and "compile_utils" in body["compile_route"]


# ---------------------------------------------------------------------------
# 2. Small real cases through the endpoint
# ---------------------------------------------------------------------------


def test_cavity_small_run(client, waiter):
    r = client.post(
        "/api/sim/generic",
        json={
            "case": "cavity",
            "grid": {"nx": 32},
            "physics": {"Re": 100.0, "u_lid": 0.06},
            "steps": 300,
            "compile_mode": "eager",
        },
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    final = waiter(job_id)
    assert final["status"] == "completed", final.get("error")

    result = client.get(f"/api/sim/generic/{job_id}/result").json()
    assert result["case"] == "cavity"
    assert result["finite"] is True
    assert result["steps"] == 300
    # Physics convention identical to the benchmark: tau = 3*u_lid*nx/Re + 0.5
    assert result["physics"]["tau"] == pytest.approx(3 * 0.06 * 32 / 100 + 0.5)
    # Metrics returned with diagnostics
    assert -0.5 < result["metrics"]["u_mid"] < 0.5
    assert "ghia" in result["metrics"]  # Re=100 reference present
    # Audit trail: common modules + compile routing
    assert any("tensorlbm.solver.collide" in m for m in result["modules_used"])
    assert any("zou_he_moving_lid" in m for m in result["modules_used"])
    assert result["compile"]["routed"] == "eager (compile_step passthrough)"
    assert result["compile"]["route"] == "benchmarks/compile_route.py"

    # Status endpoint exposes the latest progress diagnostic
    status = client.get(f"/api/sim/generic/{job_id}/status").json()
    assert status["job_id"] == job_id
    assert status["progress"]["step"] == 300


def test_poiseuille_small_run(client, waiter):
    r = client.post(
        "/api/sim/generic",
        json={
            "case": "poiseuille",
            "grid": {"H": 10},
            "physics": {"tau": 0.8, "u_max": 0.04},
            "steps": 2500,
            "compile_mode": "eager",
        },
    )
    assert r.status_code == 200
    final = waiter(r.json()["job_id"])
    assert final["status"] == "completed", final.get("error")

    result = client.get(f"/api/sim/generic/{final['job_id']}/result").json()
    metrics = result["metrics"]
    # Small grid, short run: the profile must still track the analytic
    # parabola within a generous engineering tolerance.
    assert metrics["l2_rel_err"] < 0.15, metrics
    assert metrics["u_max_num"] > 0.5 * metrics["u_max_ana"]
    assert result["finite"] is True


def test_cylinder_tiny_run(client, waiter):
    r = client.post(
        "/api/sim/generic",
        json={
            "case": "cylinder",
            "grid": {"D": 8, "domain_D": 10, "cyl_x_D": 3, "sponge_D": 3},
            "physics": {"Re": 100.0, "u_in": 0.05},
            "steps": 250,
            "compile_mode": "default",
        },
    )
    assert r.status_code == 200
    final = waiter(r.json()["job_id"])
    assert final["status"] == "completed", final.get("error")

    result = client.get(f"/api/sim/generic/{final['job_id']}/result").json()
    assert result["finite"] is True
    assert result["compile"]["canonical_mode"] == "default"
    assert result["compile"]["routed"] == "torch.compile(mode='default')"
    # Force path through the common momentum-exchange module
    assert any("compute_obstacle_forces" in m for m in result["modules_used"])
    assert "cd_mean" in result["metrics"]


# ---------------------------------------------------------------------------
# 3. Invalid input -> 4xx
# ---------------------------------------------------------------------------


def test_unknown_case_is_422(client):
    r = client.post("/api/sim/generic", json={"case": " nonexistent "})
    assert r.status_code == 422
    assert "cavity" in r.json()["detail"] and "Unknown case" in r.json()["detail"]


def test_unknown_grid_param_is_422(client):
    r = client.post("/api/sim/generic", json={"case": "cavity", "grid": {"nxx": 32}})
    assert r.status_code == 422
    assert "nxx" in r.json()["detail"]


def test_grid_below_minimum_is_422(client):
    r = client.post("/api/sim/generic", json={"case": "cavity", "grid": {"nx": 2}})
    assert r.status_code == 422
    assert "minimum" in r.json()["detail"]


def test_unknown_physics_param_is_422(client):
    r = client.post("/api/sim/generic", json={"case": "shear_wave", "physics": {"Re": 100}})
    assert r.status_code == 422
    assert "Re" in r.json()["detail"]


def test_negative_steps_is_422(client):
    r = client.post("/api/sim/generic", json={"case": "cavity", "steps": -1})
    assert r.status_code == 422  # pydantic ge=0


def test_bad_collision_is_422(client):
    r = client.post("/api/sim/generic", json={"case": "cavity", "collision": "trt"})
    assert r.status_code == 422
    assert "collision" in r.json()["detail"]


def test_cudagraph_compile_mode_is_422_with_structural_reason(client):
    r = client.post(
        "/api/sim/generic", json={"case": "cavity", "compile_mode": "reduce-overhead"}
    )
    assert r.status_code == 422
    assert "cudagraph" in r.json()["detail"]


def test_unknown_compile_mode_is_422(client):
    r = client.post("/api/sim/generic", json={"case": "cavity", "compile_mode": "turbo"})
    assert r.status_code == 422
    assert "compile_mode" in r.json()["detail"]


def test_invalid_device_is_422(client):
    r = client.post("/api/sim/generic", json={"case": "cavity", "device": "not-a-device"})
    assert r.status_code == 422


def test_status_result_unknown_job_404(client):
    assert client.get("/api/sim/generic/deadbeef/status").status_code == 404
    assert client.get("/api/sim/generic/deadbeef/result").status_code == 404


# ---------------------------------------------------------------------------
# 4. The three compile modes route like the benchmark suite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, canonical, routed",
    [
        ("eager", None, "eager (compile_step passthrough)"),
        ("default", "default", "torch.compile(mode='default')"),
        ("max-autotune-no-cudagraphs", "max-autotune-no-cudagraphs",
         "torch.compile(mode='max-autotune-no-cudagraphs')"),
    ],
)
def test_compile_mode_routing(client, waiter, mode, canonical, routed):
    """One tiny periodic case per mode: eager, default, max-autotune."""
    r = client.post(
        "/api/sim/generic",
        json={
            "case": "shear_wave",
            "grid": {"n": 12},
            "physics": {"tau": 0.8, "u0": 0.05},
            "steps": 5,
            "compile_mode": mode,
        },
    )
    assert r.status_code == 200
    final = waiter(r.json()["job_id"], timeout=300.0)
    assert final["status"] == "completed", final.get("error")

    result = client.get(f"/api/sim/generic/{final['job_id']}/result").json()
    assert result["compile"]["canonical_mode"] == canonical
    assert result["compile"]["routed"] == routed
    assert result["compile"]["requested_mode"] == mode
    assert result["finite"] is True


def test_compile_default_matches_eager_numerics(client, waiter):
    """Same case through both compiled and eager routing must agree."""
    payloads = []
    for mode in ("eager", "default"):
        r = client.post(
            "/api/sim/generic",
            json={
                "case": "shear_wave",
                "grid": {"n": 16},
                "physics": {"tau": 0.8, "u0": 0.05},
                "steps": 40,
                "compile_mode": mode,
            },
        )
        final = waiter(r.json()["job_id"])
        assert final["status"] == "completed", final.get("error")
        payloads.append(
            client.get(f"/api/sim/generic/{final['job_id']}/result").json()
        )
    a, b = (p["metrics"]["energy_final"] for p in payloads)
    assert a == pytest.approx(b, rel=1e-4, abs=1e-8)


def test_diverging_config_fails_job_cleanly(client, waiter):
    """An unstable configuration fails the job instead of returning NaNs.

    cavity at nx=24 gives tau = 0.5432, which diverges around step 110 on
    the benchmark's own path too; the generic-run finite guard surfaces
    that as a failed job with a clear error message.
    """
    r = client.post(
        "/api/sim/generic",
        json={
            "case": "cavity",
            "grid": {"nx": 24},
            "physics": {"Re": 100.0, "u_lid": 0.06},
            "steps": 200,
            "compile_mode": "eager",
        },
    )
    assert r.status_code == 200
    final = waiter(r.json()["job_id"])
    assert final["status"] == "failed"
    assert "non-finite" in (final.get("error") or "")


# ---------------------------------------------------------------------------
# 5. Parity with the verified benchmark suite
# ---------------------------------------------------------------------------


def _load_benchmark_cavity():
    """Import ``benchmarks/verified/cavity/re100/run.py`` as a module."""
    path = _REPO_ROOT / "benchmarks" / "verified" / "cavity" / "re100" / "run.py"
    spec = importlib.util.spec_from_file_location("bench_cavity_re100", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bench_cavity_re100", mod)
    spec.loader.exec_module(mod)
    return mod


def test_parity_with_benchmark_cavity(client, waiter):
    """The generic cavity case reproduces the benchmark run_case exactly.

    Both sides compose the identical common-module chain (collide_mrt →
    pre-streaming half-way bounce-back → stream → zou_he_moving_lid), so
    on the same grid/steps/mode the trajectory is bit-identical.
    """
    import torch

    bench = _load_benchmark_cavity()
    bench_result = bench.run_case(
        nx=32, re=100.0, u_lid=0.06, steps=300,
        device=torch.device("cpu"), compile_mode="eager",
    )

    r = client.post(
        "/api/sim/generic",
        json={
            "case": "cavity",
            "grid": {"nx": 32},
            "physics": {"Re": 100.0, "u_lid": 0.06},
            "steps": 300,
            "compile_mode": "eager",
        },
    )
    assert r.status_code == 200
    final = waiter(r.json()["job_id"])
    assert final["status"] == "completed", final.get("error")
    metrics = client.get(f"/api/sim/generic/{final['job_id']}/result").json()["metrics"]

    assert metrics["u_mid"] == pytest.approx(bench_result["u_mid"], abs=1e-12)
    assert metrics["u_bot"] == pytest.approx(bench_result["u_bot"], abs=1e-12)
    assert metrics["v_mid"] == pytest.approx(bench_result["v_mid"], abs=1e-12)


def test_parity_with_benchmark_cavity_compiled(client, waiter):
    """Parity also holds on the compiled (default) routing path.

    Grid nx=32 keeps tau = 0.5576 in the stable range (nx=24 gives
    tau = 0.5432, which diverges on *both* sides around step 110 — the
    generic-run finite guard turns that into a failed job, mirroring the
    benchmark's NaN diagnostics).
    """
    import torch

    bench = _load_benchmark_cavity()
    bench_result = bench.run_case(
        nx=32, re=100.0, u_lid=0.06, steps=120,
        device=torch.device("cpu"), compile_mode="default",
    )

    r = client.post(
        "/api/sim/generic",
        json={
            "case": "cavity",
            "grid": {"nx": 32},
            "physics": {"Re": 100.0, "u_lid": 0.06},
            "steps": 120,
            "compile_mode": "default",
        },
    )
    final = waiter(r.json()["job_id"], timeout=300.0)
    assert final["status"] == "completed", final.get("error")
    metrics = client.get(f"/api/sim/generic/{final['job_id']}/result").json()["metrics"]

    assert metrics["u_mid"] == pytest.approx(bench_result["u_mid"], abs=1e-9)
    assert metrics["v_mid"] == pytest.approx(bench_result["v_mid"], abs=1e-9)
