"""Acceptance tests for the case-pool case CASE-COUETTE-2D (TensorLBM).

Three layers:

1. ``test_analytic_reference_is_self_consistent`` — the reference generator is a
   closed form; this pins it (linear profile, exact momentum integral, exact
   wall shear) so a silent change of the target is caught.
2. ``test_couette_2d_meets_pool_tolerance`` — actually runs the case through the
   two repo capabilities (small grid, short run) and asserts the <2% pool
   tolerance on every predeclared quantity.
3. ``test_committed_artifact_acceptance`` — the committed ``summary.json``
   artifact must carry a passing verdict, the predeclared thresholds and the
   per-quantity ref/got numbers (guards against "artifact without evidence").

Run: PYTHONPATH=src:benchmarks pytest -q tests/test_case_pool_couette_2d.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "benchmarks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from case_pool.couette_2d import run_case as cp  # noqa: E402

ART = REPO / "benchmarks" / "case_pool" / "couette_2d" / "artifacts"


def test_analytic_reference_is_self_consistent() -> None:
    U0, H = 0.05, 32.0
    y = np.arange(1, int(H) + 1) - 0.5
    u = cp.analytic_profile(U0, H, y)
    # linear in y through (0, 0) and (H, U0)
    assert np.allclose(u, U0 * y / H)
    assert np.isclose(u[0] + u[-1], U0)  # symmetric about the mid-gap value
    # exact momentum per unit width of the closed form
    ref = cp.analytic_momentum_integral(U0, H)
    assert np.isclose(ref, 0.5 * U0 * H)
    nodes_u = np.concatenate(([0.0], U0 * y / H, [U0]))
    nodes_y = np.concatenate(([0.0], y, [H]))
    assert np.isclose(np.trapezoid(nodes_u, nodes_y), ref, rtol=1e-12)
    # wall shear of the linear solution: mu du/dy = nu U0 / H (rho = 1)
    nu = (0.8 - 0.5) / 3.0
    assert np.isclose(cp.analytic_wall_shear(nu, U0, H), nu * U0 / H)
    # the pool tolerance is <2% for this case
    assert cp.TOL_PCT == 2.0
    assert all(v is not None for v in cp.THRESHOLDS.values())


@pytest.mark.parametrize("H,tau,U0", [(16, 0.8, 0.05)])
def test_couette_2d_meets_pool_tolerance(tmp_path: Path, H: int, tau: float, U0: float) -> None:
    r = cp.run_route_a(
        H=H,
        tau=tau,
        U0=U0,
        collision="bgk",
        out_dir=tmp_path,
        min_steps=2000,
        max_steps=8000,
        compile_mode="eager",
    )
    err = r["errors_pct"]
    assert err["profile_l2_rel_pct"] < cp.TOL_PCT, err
    assert err["profile_max_rel_pct"] < cp.TOL_PCT, err
    assert err["wall_shear_rel_pct"] < cp.TOL_PCT, err
    assert err["top_row_u_rel_pct"] < cp.TOL_PCT, err
    assert err["momentum_integral_rel_pct"] < cp.TOL_PCT, err
    assert err["mass_drift_pct"] < cp.THRESHOLDS["mass_drift_pct"], err
    assert all(r["pass"].values()), r["pass"]
    # ref and got must both be present (no vacuous comparison)
    for key in ("u_top_row", "wall_shear", "momentum_integral"):
        assert np.isfinite(r["ref_vs_got"][key]["ref"])
        assert np.isfinite(r["ref_vs_got"][key]["got"])


def test_route_b_force_meets_pool_tolerance(tmp_path: Path) -> None:
    r = cp.run_route_b(ny=18, wall_speed=0.05, tau=0.8, fmt="halfway", out_dir=tmp_path, steps=2000)
    for key, ok in r["pass"].items():
        assert ok, (key, r["errors_pct"][key], r["thresholds_pct"][key])


def test_mass_drift_is_an_arithmetic_precision_effect(tmp_path: Path) -> None:
    """Root cause of the round-1 conservation failure: float32 rounding, not physics.

    Same physics, same scheme, same step count, only the population dtype differs:
    float32 leaves a one-signed sub-epsilon residue per moving-wall injection, so a
    *cumulative* 1e-3 % bound is unreachable; float64 conserves to round-off.
    """
    out = {}
    for dtype in ("float32", "float64"):
        out_dir = tmp_path / f"prec_{dtype}"
        out_dir.mkdir(parents=True, exist_ok=True)
        r = cp.run_route_a(
            H=16, tau=0.8, U0=0.05, collision="bgk",
            out_dir=out_dir,
            min_steps=2000, max_steps=6000, compile_mode="eager", force=True, dtype=dtype,
        )
        out[dtype] = r["errors_pct"]["mass_drift_pct"]
        assert r["dtype"] == dtype
    assert out["float64"] < cp.THRESHOLDS["mass_drift_pct"], out
    assert out["float64"] < out["float32"], out          # precision genuinely helps
    assert out["float32"] > 1e-6, out                    # ... and float32 is not already clean


def test_profile_is_stored_losslessly(tmp_path: Path) -> None:
    """`round(u, 8)` used to quantise the profile (1.3e-3 % at H=64 > the guard floor)."""
    # Only the serialisation is under test, so a very short run is enough (the
    # quantisation error is a property of the writer, not of the convergence state).
    out_dir = tmp_path / "lossless"
    out_dir.mkdir(parents=True, exist_ok=True)
    r = cp.run_route_a(
        H=64, tau=0.8, U0=0.05, collision="bgk",
        out_dir=out_dir,
        min_steps=200, max_steps=400, compile_mode="eager", force=True, dtype="float64",
    )
    u = np.array([p["u_num"] for p in r["profile"]], dtype=np.float64)
    # quantisation to 1e-8 would make every |u| a multiple of 1e-8; the true profile is not
    assert not np.allclose(u, np.round(u, 8), rtol=0.0, atol=0.0), "profile still quantised"
    # the analytic target must be stored losslessly as well (it drives the ref/got table)
    ur = np.array([p["u_ref"] for p in r["profile"]], dtype=np.float64)
    assert not np.allclose(ur, np.round(ur, 8), rtol=0.0, atol=0.0), "analytic ref quantised"


def test_committed_artifact_acceptance() -> None:
    summary_path = ART / "summary.json"
    if not summary_path.exists():  # pragma: no cover - artifact not committed yet
        pytest.skip("summary.json artifact not present in this checkout")
    summary = json.loads(summary_path.read_text())

    assert summary["case_code"] == "COUETTE-2D"
    assert summary["tol_policy_pct"] == 2.0
    assert summary["predeclared_thresholds_pct"]["profile_l2_rel_pct"] == 2.0
    # the case's acceptance criterion is the <2% pool tolerance
    assert summary["verdict"]["pool_tolerance_pass"] is True, summary["verdict"]["tolerance_failing"]
    # guard v2 (absolute accuracy floor at every resolution) is the deciding guard
    floor = summary["convergence_floor_guard"]
    assert floor and all(g["pass"] for g in floor), floor
    assert all(g["l2_pct"] <= g["floor_pct"] for g in floor)
    # guard v1 (refinement slope) is retained verbatim as a recorded failure
    v1 = summary["guard_v1_convergence_order"]
    assert v1["outcome"].startswith("FAIL"), v1
    assert v1["failing"], v1
    # the precision comparison proves the float32/float64 root cause
    prec = summary["precision_comparison"]
    assert prec, "precision_comparison missing"
    for row in prec:
        assert row["profile_l2_rel_pct_float64"] < row["profile_l2_rel_pct_float32"], row
        assert row["mass_drift_pct_float64"] < row["mass_drift_pct_float32"], row
        assert row["mass_drift_pct_float64"] < row["guard_pct"], row
    # float32 must break the cumulative guard somewhere (a lucky point passing on
    # its own does not make float32 usable), float64 must pass everywhere
    assert max(r["mass_drift_pct_float32"] for r in prec) > prec[0]["guard_pct"], prec
    # every committed route-A point must carry its dtype (no unlabelled precision)
    assert all(r.get("dtype") for r in summary["route_a"])
    # self-imposed guards that are NOT met must be disclosed, never silently passed
    assert summary["verdict"]["guard_or_diagnostic_failing"] == summary["verdict"]["guard_or_diagnostic_failing"]
    assert summary["verdict"]["pass"] == (not summary["verdict"]["failing"])

    # primary reference-grid point must carry the full ref/got table
    ref_pt = next(
        r
        for r in summary["route_a"]
        if r["H"] == 32 and r["tau"] == 0.8 and r["U0"] == 0.05 and r["collision"] == "bgk"
    )
    for key in ("profile_l2_rel_pct", "profile_max_rel_pct", "wall_shear_rel_pct",
                "momentum_integral_rel_pct", "mass_drift_pct"):
        assert ref_pt["errors_pct"][key] <= summary["predeclared_thresholds_pct"][key]
    assert ref_pt["ref_vs_got"]["wall_shear"]["got"] > 0
    assert len(ref_pt["profile"]) == 32

    # an independent force route must be present and pass
    assert summary["route_b"], "route B evidence missing"
    assert all(all(b["pass"].values()) for b in summary["route_b"])
    # reproducibility metadata
    env = summary["environment"]
    assert env["git_commit"] and env["torch"]
    # the committed arithmetic must be declared and must match the reference point
    # (round 2 moved the conservation-critical path to float64; see 5.3 of README)
    assert env["dtype"] in ("float32", "float64")
    assert ref_pt["dtype"] == env["dtype"], (ref_pt["dtype"], env["dtype"])
    assert "case_pool/couette_2d/run_case.py" in summary["repro_cmd"]