"""Benchmark suite for the MCMP verification/calibration tooling (FEAT-LBM-VERIFY-BENCH).

Three deliverables are pinned here:

* the Young-Laplace calibrator (:func:`tensorlbm.mcmp_verification.young_laplace_calibration`)
  including the *guardrails* that make it falsifiable (an unmeasurable state must
  return a code, not a number - the defect recorded for inventory entry 21);
* the spurious-current baseline, i.e. that a droplet at rest really is measured
  and that the level improves with resolution;
* the fast contact-angle / interface-thickness checks, calibrated against
  analytic fields with known answers.

The report (`docs/MCMP_VERIFY_BENCH.md`) is rendered from
`artifacts/mcmp_verify_bench/results.json`; `test_report_is_rendered_from_json`
re-renders it and fails if the two disagree, which is the machine check behind
the "zero hand-copied numbers" requirement.

Runtime: the analytic tests are milliseconds. The solver-backed ones are marked
`slow` (they relax real droplets); `pytest -m "not slow"` runs the fast set.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest
import torch

import tensorlbm.mcmp_verification as mv
from tensorlbm.mcmp_verify_report import render_report

REPO_ROOT = Path(__file__).resolve().parents[2]

# This box has 128 cores and is shared: torch's default thread pool oversubscribes
# it and the small (<=128^2) tensors of these runs spend more time in thread
# hand-off than in arithmetic (measured 57 ms/step with 8 threads vs 3.8 ms/step
# with one). Pin to a single thread for the solver-backed tests.
torch.set_num_threads(1)
ARTIFACTS = REPO_ROOT / "artifacts" / "mcmp_verify_bench"
DOC = REPO_ROOT / "docs" / "MCMP_VERIFY_BENCH.md"


# ---------------------------------------------------------------------------
# Unit: conventions
# ---------------------------------------------------------------------------


def test_pressure_pair_sum_matches_gradient_form():
    """The pair-sum pressure (inventory entry 20) and the gradient derivation agree.

    ``grad p = F_1 + F_2`` for this library's discrete force
    ``F_1 = +G_12 cs^2 psi1 grad psi2`` integrates to
    ``p = cs^2 (rho1 + rho2) + G_12 cs^2 psi1 psi2``, which must equal the
    upstream pair sum ``sum rho_s/3 + 2 g psi1 psi2/6`` with ``g = G_12``
    (the same sign: the ``cs^2`` factor that this kernel carries inside ``G``
    must not be double counted by a sign flip - see the module docstring).
    """
    torch.manual_seed(0)
    rho1 = torch.rand(16, 16) + 0.1
    rho2 = torch.rand(16, 16) + 0.1
    G = -2.5
    p = mv.sc_mcmp_pressure(rho1, rho2, G)
    pair_sum = (rho1 + rho2) / 3.0 + 2.0 * G * (rho1 * rho2) / 6.0
    assert torch.allclose(p, pair_sum, atol=1e-6)


def test_pressure_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        mv.sc_mcmp_pressure(torch.zeros(4, 5), torch.zeros(4, 6), -2.5)


def test_barycentric_velocity_is_mass_weighted():
    rho1 = torch.full((8, 8), 1.0)
    rho2 = torch.full((8, 8), 3.0)
    z = torch.zeros_like(rho1)
    from tensorlbm.d2q9 import equilibrium

    f1 = equilibrium(rho1, torch.full_like(rho1, 0.04), z)
    f2 = equilibrium(rho2, z, z)
    ux, uy, rho_tot = mv.barycentric_velocity(f1, f2)
    # (1*0.04 + 3*0)/4 = 0.01
    assert float(ux.mean()) == pytest.approx(0.01, abs=1e-6)
    assert float(uy.abs().max()) < 1e-6
    assert torch.allclose(rho_tot, torch.full((8, 8), 4.0), atol=1e-5)


# ---------------------------------------------------------------------------
# Unit: guardrails (the falsifiability requirement)
# ---------------------------------------------------------------------------


def test_droplet_radius_flags_missing_crossing():
    """A uniform field has no interface: the radius must be ``None`` + a code."""
    out = mv.droplet_radius(torch.full((32, 32), 0.5), rho_av=0.5)
    assert out["radius"] is None
    codes = {g["code"] for g in out["guardrails"]}
    assert codes & {"centre_not_liquid", "no_crossing"}


def test_droplet_radius_flags_periodic_image_risk():
    """A droplet wider than 0.35 n interacts with its images and must be flagged."""
    n = 32
    yy, xx = mv._grid(n, torch.device("cpu"))
    r = torch.sqrt((xx - 15.5) ** 2 + (yy - 15.5) ** 2)
    rho = 0.14 + 0.78 * 0.5 * (1.0 - torch.tanh((r - 14.0) / 2.0))
    out = mv.droplet_radius(rho, rho_av=0.53)
    assert out["radius"] is not None
    assert "periodic_image_risk" in {g["code"] for g in out["guardrails"]}


def test_droplet_radius_recovers_known_radius():
    n = 96
    yy, xx = mv._grid(n, torch.device("cpu"))
    r = torch.sqrt((xx - 47.5) ** 2 + (yy - 47.5) ** 2)
    rho = 0.14 + 0.78 * 0.5 * (1.0 - torch.tanh((r - 20.0) / 2.0))
    out = mv.droplet_radius(rho, rho_av=0.14 + 0.39)
    assert out["radius"] == pytest.approx(20.0, abs=0.35)
    assert out["crossings"] == 4
    assert not out["guardrails"]


def test_measure_laplace_returns_guardrail_on_non_finite_field():
    nan = torch.full((16, 16), float("nan"))
    out = mv.measure_laplace(
        {
            "rho1": nan,
            "rho2": nan.clone(),
            "G_12": -2.5,
            "radius_init": 5.0,
            "steps_run": 10,
            "converged": False,
            "u_max": float("nan"),
        }
    )
    assert out["sigma"] is None and out["dp"] is None
    assert {g["code"] for g in out["guardrails"]} == {"non_finite_field"}


def test_simulate_rejects_bad_grid_size():
    with pytest.raises(ValueError):
        mv.simulate_static_droplet(G_12=-2.5, n=-2.2, radius_init=5.0, steps=1)


# ---------------------------------------------------------------------------
# Analytic calibration: interface thickness and contact angle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("w", [1.0, 1.5, 2.0, 3.0])
def test_interface_thickness_recovers_known_width(w):
    n = 128
    yy, xx = mv._grid(n, torch.device("cpu"))
    r = torch.sqrt((xx - 63.5) ** 2 + (yy - 63.5) ** 2)
    rho = 0.14 + 0.78 * 0.5 * (1.0 - torch.tanh((r - 20.0) / w))
    out = mv.interface_thickness(rho)
    assert out["w_tanh"] == pytest.approx(w, rel=0.02)
    assert out["width_10_90"] == pytest.approx(mv.interface_width_10_90(w), rel=0.02)
    assert out["R_fit"] == pytest.approx(20.0, abs=0.1)
    assert out["r2"] > 0.99


def test_interface_thickness_flags_uniform_field():
    out = mv.interface_thickness(torch.full((64, 64), 0.7))
    assert out["w_tanh"] is None
    assert out["guardrails"]


@pytest.mark.parametrize("theta", [30.0, 45.0, 60.0, 90.0, 120.0, 135.0])
def test_contact_angle_three_estimators(theta):
    """Analytic cap with prescribed contact angle: three independent estimators.

    Tolerances are the calibrated accuracy of each route (see the report):
    circle fit <= 0.5 deg, cap reconstruction <= 3 deg, near-wall gradient
    projection <= 6 deg (it carries a ~-3.5 deg systematic offset from the
    wall extrapolation).
    """
    rho, solid = mv.synthetic_cap_field(n=96, radius=25.0, theta_deg=theta, wall_rows=6)
    geo = mv.geometric_contact_angle(rho, solid, order=2, first_layer=2)
    circ = mv.circle_fit_contact_angle(rho, solid)
    cap = mv.cap_geometry_contact_angle(rho, solid)
    assert circ["theta_deg"] == pytest.approx(theta, abs=0.5)
    assert cap["theta_deg"] == pytest.approx(theta, abs=3.0)
    assert geo["theta_deg"] == pytest.approx(theta, abs=6.0)


def test_contact_angle_convention_is_physical():
    """A wetting cap (theta < 90) must have its centre *below* the wall.

    Regression against the sign error made while building this tool: with
    ``height = +R cos(theta)`` every measured angle came back as
    ``180 - theta`` and the calibration looked perfect while measuring the
    inverse of the physical convention.
    """
    rho, solid = mv.synthetic_cap_field(n=96, radius=25.0, theta_deg=60.0, wall_rows=6)
    out = mv.circle_fit_contact_angle(rho, solid)
    assert out["theta_deg"] < 90.0
    assert out["height_above_wall"] < 0.0
    assert out["height_above_wall"] == pytest.approx(25.0 * math.cos(math.radians(120.0)), abs=0.5)


def test_contact_angle_rejects_unsupported_wall_geometry():
    rho, solid = mv.synthetic_cap_field(n=64, radius=15.0, theta_deg=90.0, wall_rows=4)
    blobby = solid.clone()
    blobby[30:34, 30:34] = True  # a pillar: no longer a single flat strip
    for fn in (
        mv.geometric_contact_angle,
        mv.circle_fit_contact_angle,
        mv.cap_geometry_contact_angle,
    ):
        out = fn(rho, blobby)
        assert out["theta_deg"] is None
        assert {g["code"] for g in out["guardrails"]} == {"unsupported_wall_geometry"}


# ---------------------------------------------------------------------------
# Solver-backed checks (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.physics
def test_flat_interface_has_no_pressure_jump():
    """A planar interface must satisfy ``p_liq = p_gas``: the pressure<->force check.

    This is the strongest single check of the pressure convention: get the sign
    or the 1/6 factor wrong and a flat interface produces a spurious jump of the
    same order as the Laplace pressure of a droplet.
    """
    # n=96 (the production size) gives 1.9e-6; n=64 gives 6.6e-4.  n=48 is
    # deliberately not used: at that size the initial composition leaves the box
    # with more component 1 than its coexistence pair holds, the liquid ends up
    # compressed (rho1_liq = 0.910 vs 0.882) and the "flat interface jump" grows
    # to 1.2e-2 - which is exactly what this check is meant to expose.
    out = mv.estimate_coexistence(G_12=-2.5, n=64, geometry="planar", steps=8000)
    assert out["converged"]
    assert abs(out["flat_interface_dp"]) < 2e-3
    assert out["rho1_liq"] > 4.0 * out["rho1_gas"]


@pytest.mark.slow
@pytest.mark.physics
def test_spurious_current_decreases_with_resolution():
    """At fixed physical radius, refining the grid must not increase |u|max."""
    runs = []
    for n in (48, 64):
        run = mv.simulate_static_droplet(G_12=-2.5, n=n, radius_init=10.0, steps=6000)
        case = mv.measure_laplace(run)
        assert case["converged"]
        assert case["u_max"] == pytest.approx(case["u_max"])  # finite
        assert case["u_max"] < 0.05 * math.sqrt(mv.CS2) * 3.0  # ~5% of cs ceiling
        assert case["sigma"] is not None and case["sigma"] > 0
        runs.append((n, case))
    coarse, fine = runs[0][1]["u_max"], runs[1][1]["u_max"]
    assert fine <= coarse * 1.02, f"|u|max grew with resolution: {coarse:.3e} -> {fine:.3e}"


@pytest.mark.slow
@pytest.mark.physics
@pytest.mark.benchmark
def test_young_laplace_calibration_passes_on_smoke_protocol():
    """End-to-end Laplace calibration on a reduced protocol (acceptance < 5%)."""
    result = mv.young_laplace_calibration(
        [-2.5], radii=[14.0, 19.0], n=64, steps=8000, coexist_steps=8000
    )
    level = result["levels"][0]
    assert level["sigma_origin"] is not None and level["sigma_origin"] > 0
    hard = {
        "no_crossing",
        "not_converged",
        "negative_sigma",
        "centre_not_liquid",
        "non_finite_field",
    }
    assert not (hard & {g["code"] for g in level["guardrails"]})
    assert level["max_rel_err"] < 0.05, level
    assert level["r2"] > 0.99
    assert result["pass"] is True


# ---------------------------------------------------------------------------
# Artifact / report integrity ("zero hand-copied numbers")
# ---------------------------------------------------------------------------


def test_report_is_rendered_from_json(tmp_path):
    """The committed report must be exactly what the renderer produces."""
    assert (ARTIFACTS / "results.json").exists(), "run benchmarks/mcmp_verify_bench.py first"
    assert DOC.exists(), "docs/MCMP_VERIFY_BENCH.md missing"
    for name in ("results.json", "oss_consumption_chain.json"):
        shutil.copy(ARTIFACTS / name, tmp_path / name)
    hero = ARTIFACTS / "hero_fields.npz"
    if hero.exists():
        shutil.copy(hero, tmp_path / "hero_fields.npz")
    rendered = render_report(tmp_path, tmp_path / "report.md")
    assert rendered == DOC.read_text(), (
        "docs/MCMP_VERIFY_BENCH.md is not the rendered output of results.json - a number was "
        "edited by hand; re-run benchmarks/mcmp_verify_bench.py --report"
    )


def test_acceptance_numbers_in_results_json():
    """The committed (full-protocol) results must satisfy the mission criteria."""
    res = json.loads((ARTIFACTS / "results.json").read_text())
    assert res["protocol"]["quick"] is False, "acceptance numbers require a full run"
    lap = res["laplace"]
    assert len(lap["levels"]) >= 3, "at least three surface-tension levels are required"
    assert len(lap["protocol"]["radii"]) >= 3
    for level in lap["levels"]:
        assert level["sigma_origin"] is not None
        assert level["max_rel_err"] < 0.05, (level["G_12"], level["max_rel_err"])
        hard = {
            "no_crossing",
            "not_converged",
            "negative_sigma",
            "centre_not_liquid",
            "non_finite_field",
        }
        assert not (hard & {g["code"] for g in level["guardrails"]})
    sigmas = [level["sigma_origin"] for level in lap["levels"]]
    assert sigmas == sorted(sigmas), "surface tension must grow with |G_12|"
    sp = res["spurious"]
    assert len(sp["rows"]) >= 6
    assert len({row["n"] for row in sp["rows"]}) >= 2
    assert len({row["G_12"] for row in sp["rows"]}) >= 2
    for row in sp["rows"]:
        assert row["u_max"] is not None and row["u_max"] >= 0.0
        assert row["ca_spurious"] is None or row["ca_spurious"] >= 0.0


def test_consumption_chain_manifest_is_actionable():
    """Every OSS entry cited in the report must resolve to code in this repo."""
    chain = json.loads((ARTIFACTS / "oss_consumption_chain.json").read_text())
    entries = chain["entries"]
    assert len(entries) >= 5, "the mission requires >= 5 consumed OSS entries"
    assert len({e["id"] for e in entries}) >= 4, "at least four distinct inventory entries"
    known = {"re-derived", "adapted", "defect-fix", "design-reference", "inventory-only"}
    for entry in entries:
        assert entry["status"] in known, entry["status"]
        assert entry["verified_source"], entry["id"]
        symbol = entry["symbol"]
        if entry["status"] in {"design-reference"} or "not consumed" in symbol:
            continue
        for name in symbol.replace("+", ",").split(","):
            name = name.strip().split(" ")[0]
            if not name or not name.isidentifier():
                continue
            assert hasattr(mv, name), f"entry {entry['id']} cites missing symbol {name}"
    # the report must carry the same ids, so the mapping cannot drift from the code
    doc = DOC.read_text()
    for entry in entries:
        assert f"`{entry['id']}`" in doc


# ---------------------------------------------------------------------------
# Laplace factor: the fields are 2D, so Delta p = sigma / R (not 2 sigma / R)
# ---------------------------------------------------------------------------


def _laplace_case(*, radius, dp, G=-2.5):
    return {
        "G_12": G,
        "radius": radius,
        "radius_init": radius,
        "dp": dp,
        "sigma": dp * radius,  # 2D measurement: sigma = dp R / (dims - 1), dims = 2
        "sigma_origin": None,
        "converged": True,
        "u_max": 1e-3,
        "guardrails": [],
    }


def test_level_fit_uses_the_2d_laplace_factor():
    """``sigma_origin`` must equal the through-origin fit with ``x = 1/R``.

    Regression for the defect fixed on 2026-11-12: these fields are a 2D circular
    droplet (unit depth), whose law is ``Delta p = sigma / R``; until then the
    calibrator used the 3D sphere factor ``x = 2/R``, so every reported sigma was
    **half** of its value and every ``Ca_sp = mu |u|/sigma`` twice its value.  The
    law residual is invariant under a rescaling of ``x`` (see the next test), so
    the <5% acceptance verdict never exposed the bug - only the absolute sigma did.
    """
    cases = [
        _laplace_case(radius=14.0, dp=0.0138),
        _laplace_case(radius=19.0, dp=0.0102),
        _laplace_case(radius=24.0, dp=0.0082),
    ]
    level = mv._fit_laplace_level(-2.5, cases, 0.05, dims=2)
    xs = [1.0 / c["radius"] for c in cases]
    ys = [c["dp"] for c in cases]
    independent = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    assert level["sigma_origin"] == pytest.approx(independent, rel=1e-12)
    # and the physical magnitude: sigma ~ 0.19, not the halved 0.095
    assert level["sigma_origin"] == pytest.approx(0.19, abs=0.02)
    assert level["laplace_dims"] == 2


def test_factor_scaling_leaves_the_acceptance_metric_invariant():
    """The <5% residual cannot detect the factor; the absolute sigma must.

    ``sigma_0`` scales linearly with the factor and ``x`` scales with it too, so
    ``sigma_0 x`` is unchanged and the per-case residual is identical.  This is
    why the factor defect survived a passing acceptance test, and why the
    regression has to assert the *value* of sigma, not just the verdict.
    """
    cases = [
        _laplace_case(radius=14.0, dp=0.0138),
        _laplace_case(radius=19.0, dp=0.0102),
        _laplace_case(radius=24.0, dp=0.0082),
    ]
    two_d = mv._fit_laplace_level(-2.5, cases, 0.05, dims=2)
    three_d = mv._fit_laplace_level(-2.5, cases, 0.05, dims=3)
    assert three_d["sigma_origin"] == pytest.approx(0.5 * two_d["sigma_origin"], rel=1e-12)
    assert three_d["max_rel_err"] == pytest.approx(two_d["max_rel_err"], rel=1e-12)


def test_committed_sigmas_carry_the_2d_factor():
    """Every committed case sigma must equal ``dp * R`` (2D), never ``dp R/2``."""
    res = json.loads((ARTIFACTS / "results.json").read_text())
    lap = res["laplace"]
    assert lap["protocol"]["laplace_dims"] == 2, "the protocol must pin the dimension"
    for case in lap["cases"]:
        if case["sigma"] is None:
            continue
        assert case["laplace_dims"] == 2
        assert case["sigma"] == pytest.approx(case["dp"] * case["radius"], rel=1e-6)
    for level in lap["levels"]:
        assert level["laplace_dims"] == 2
    # the spurious-current comparison level is derived from sigma, so it must use
    # the same factor: Ca_sp = mu |u|/sigma with sigma = dp R
    for row in res["spurious"]["rows"]:
        if row["ca_spurious"] is None:
            continue
        assert row["ca_spurious"] == pytest.approx(
            row["mu_lattice"] * row["u_max"] / (row["dp"] * row["radius"]), rel=1e-9
        )
