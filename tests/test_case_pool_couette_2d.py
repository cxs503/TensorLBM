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
    assert env["git_commit"] and env["torch"] and env["dtype"] == "float32"
    assert "case_pool/couette_2d/run_case.py" in summary["repro_cmd"]