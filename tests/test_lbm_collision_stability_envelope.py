"""Tests for the D2Q9 collision-operator stability envelope (node P482-N05).

The tests lock the four campaign legs at small scale:

* the MRT model operator matches the pre-existing ``stability.Collision`` MRT
  operator when the rates coincide (no silent redefinition of the collision),
* the von Neumann verdict reproduces the known D2Q9 BGK behaviour (stable for
  ``tau > 0.5`` at moderate ``Ma``, unstable in the low-``tau``/high-``Ma``
  corner of the declared window),
* the measured modal growth agrees with the prediction at the *same*
  box-representable wave vector (the exchange law) to better than 0.5% in the
  amplification factor,
* the declared defective operators are rejected by both routes,
* the envelope is deterministic (digest reproducible) and the
  accuracy-recommendation gate is fail-closed outside it.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tensorlbm.accuracy_recommendation import (
    recommend_with_stability_envelope,
)
from tensorlbm.stability import collision_stability as cs
from tensorlbm.stability import envelope as ev


# ---------------------------------------------------------------------------
# Collision-operator definitions
# ---------------------------------------------------------------------------


def test_mrt_operator_matches_legacy_mrt_collision() -> None:
    """Same rates -> same collision matrix as the pre-existing MRT model."""
    from tensorlbm.stability.von_neumann import Collision

    tau = 0.7
    mine = cs.mrt_model(tau, s_e=1.64, s_eps=1.54, s_qx=1.7, s_qy=1.7)
    legacy = Collision(kind="mrt", tau=tau, s_e=1.64, s_eps=1.54, s_q=1.7)
    assert np.allclose(mine.operator_matrix(), legacy.operator_matrix(), atol=1e-13)


def test_bgk_rates_and_shear_tie() -> None:
    m = cs.bgk_model(0.62)
    assert np.allclose(m.rates, np.full(9, 1.0 / 0.62))
    assert m.genus == "bgk"
    mrt = cs.mrt_model(0.62)
    named = dict(zip(cs.MOMENT_SLOTS, mrt.rates))
    assert named["pxx"] == pytest.approx(1.0 / 0.62)
    assert named["pxy"] == pytest.approx(1.0 / 0.62)
    assert named["rho"] == 0.0 and named["jx"] == 0.0


def test_orthogonal_and_mixed_basis_defect() -> None:
    assert cs.basis_orthogonality_defect(cs.orthogonal_basis()) == pytest.approx(0.0, abs=1e-12)
    mixed = cs.mixed_basis("jx", "qx", 0.8)
    assert cs.basis_orthogonality_defect(mixed) > 1.0
    # the mixed basis stays invertible: the defect must be a *physics* defect,
    # not a singular transform
    assert np.all(np.isfinite(np.linalg.inv(mixed)))


def test_rates_vector_slots() -> None:
    rates = cs.rates_vector(0.6, s_e=1.1, s_eps=1.2, s_qx=1.3, s_qy=1.4, s_pxx=1.5, s_pxy=1.6)
    named = dict(zip(cs.MOMENT_SLOTS, rates))
    assert named == {
        "rho": 0.0, "e": 1.1, "eps": 1.2, "jx": 0.0, "qx": 1.3, "jy": 0.0,
        "qy": 1.4, "pxx": 1.5, "pxy": 1.6,
    }


# ---------------------------------------------------------------------------
# M1 prediction side
# ---------------------------------------------------------------------------


def test_bgk_stable_at_reference_point_and_unstable_in_corner() -> None:
    stable = cs.spectral_radius(cs.bgk_model(0.7), 0.2, n_theta=33)
    assert stable["lambda_max"] <= 1.0 + cs.STABILITY_TOL
    assert stable["stable"] is True

    corner = cs.spectral_radius(cs.bgk_model(0.505), 0.5, n_theta=65)
    assert corner["lambda_max"] > 1.02
    assert corner["stable"] is False
    # the unstable branch sits at a finite, non-zero wave vector (not the mean mode)
    kx, ky = corner["k_argmax"]
    assert math.hypot(kx, ky) > 0.5


def test_stability_is_monotone_in_tau_at_fixed_ma() -> None:
    lams = [cs.spectral_radius(cs.bgk_model(t), 0.5, n_theta=33)["lambda_max"]
            for t in (0.505, 0.52, 0.56, 0.7)]
    assert lams[0] > lams[1] > lams[2] - 1e-12
    assert lams[-1] <= 1.0 + cs.STABILITY_TOL


def test_neutral_checkerboard_branch_is_exactly_marginal() -> None:
    """D2Q9 has a |mu| = 1 branch at (pi, 0); it must not be counted as growth."""
    for tau in (0.55, 0.8):
        model = cs.bgk_model(tau)
        eigs = cs.amplification_eigenvalues(
            np.stack([model.map_matrix(0.0)]), np.array([[math.pi, 0.0]])
        )[0, 0]
        assert abs(np.abs(eigs).max() - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# M1 measurement side (the exchange law)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau,ma", [(0.505, 0.5), (0.51, 0.45)])
def test_measured_growth_matches_prediction_at_same_wave_vector(tau: float, ma: float) -> None:
    model = cs.bgk_model(tau)
    pred = cs.predicted_box_radius(model, ma, n_box=32)
    meas = cs.run_mode_box(model, ma, k=pred["k_argmax"], n=32, n_steps=300)
    assert meas["fit_r2"] > 0.95
    assert meas["lambda_meas"] == pytest.approx(pred["lambda_max"], rel=5e-3)
    assert pred["stable"] is False


def test_wave_vector_must_be_box_representable() -> None:
    model = cs.bgk_model(0.6)
    with pytest.raises(ValueError, match="not representable"):
        cs.run_mode_box(model, 0.2, k=(0.34, 1.87), n=32, n_steps=5)


def test_stable_point_does_not_grow() -> None:
    model = cs.bgk_model(0.8)
    pred = cs.predicted_box_radius(model, 0.1, n_box=32)
    meas = cs.run_mode_box(model, 0.1, k=pred["k_argmax"], n=32, n_steps=200)
    assert meas["total_growth"] < 1.0


def test_predicted_and_measured_boundary_agree() -> None:
    factory = cs.tier_factory("bgk")
    fine = cs.predicted_critical_tau(factory, 0.5, iters=6)
    box = cs.box_predicted_critical_tau(factory, 0.5, iters=6)
    meas = cs.measured_critical_tau(factory, 0.5, iters=5)
    assert fine["interior"] and box["interior"] and meas["interior"]
    assert meas["tau_crit"] == pytest.approx(box["tau_crit"], rel=0.02)
    assert meas["tau_crit"] == pytest.approx(fine["tau_crit"], rel=0.03)


# ---------------------------------------------------------------------------
# M2 tiers
# ---------------------------------------------------------------------------


def test_tiers_are_distinct_and_share_the_shear_tie() -> None:
    tau = 0.6
    bgk = cs.tier_model_at("bgk", tau, 0.3)
    a = cs.tier_model_at("mrt_a", tau, 0.3)
    b = cs.tier_model_at("mrt_b", tau, 0.3, variant="oracle")
    assert bgk.genus == "bgk" and a.genus == "mrt" and b.genus == "mrt"
    for model in (a, b):
        named = dict(zip(cs.MOMENT_SLOTS, model.rates))
        assert named["pxx"] == pytest.approx(1.0 / tau)
        assert named["pxy"] == pytest.approx(1.0 / tau)
    design = cs.design_rates("mrt_b", tau)
    assert design["s_qx"] in cs.PREDECLARED["rate_candidates"]["s_qx"]


def test_plane_scan_case_count_and_ratio() -> None:
    plane = ev.scan_plane("bgk", "canonical", n_tau=7, n_ma=6, n_k=9)
    assert plane["n_cases"] == 42
    assert 0.0 <= plane["stable_area_ratio"] <= 1.0
    assert len(plane["lambda_max"]) == 7 and len(plane["lambda_max"][0]) == 6
    assert plane["lambda_grid_sha256"] == ev.scan_plane("bgk", "canonical", n_tau=7, n_ma=6, n_k=9)["lambda_grid_sha256"]


# ---------------------------------------------------------------------------
# M3 falsifiability and negative controls
# ---------------------------------------------------------------------------


def test_poor_mrt_is_less_stable_than_bgk() -> None:
    tau, ma = 0.51, 0.45
    bgk = cs.spectral_radius(cs.bgk_model(tau), ma, n_theta=33)["lambda_max"]
    poor = cs.spectral_radius(cs.poor_mrt_model(tau), ma, n_theta=33)["lambda_max"]
    assert poor > bgk + 1e-3


@pytest.mark.parametrize("family", ["over_relax_shear", "over_relax_energy"])
def test_negative_controls_rejected_by_both_routes(family: str) -> None:
    model = cs.negative_control(family, 0.6)
    pred = cs.spectral_radius(model, 0.2, n_theta=33)
    assert pred["lambda_max"] > 1.0 + cs.STABILITY_TOL
    k = cs.predicted_box_radius(model, 0.2, n_box=16)["k_argmax"]
    meas = cs.run_mode_box(model, 0.2, k=k, n=16, n_steps=150)
    assert meas["lambda_meas"] > 1.0 + cs.MEASURED_TOL


def test_mixed_basis_probe_is_not_a_stability_defect() -> None:
    """Specificity: the basis defect alone must not be flagged (documented limit)."""
    model = cs.negative_control("mixed_basis", 0.6)
    assert model.describe()["basis_orthogonality_defect"] > 1.0
    assert cs.spectral_radius(model, 0.2, n_theta=33)["lambda_max"] <= 1.0 + cs.STABILITY_TOL


def test_mixed_basis_with_out_of_range_rate_is_rejected() -> None:
    model = cs.negative_control("mixed_basis_over_relax", 0.6)
    assert model.max_rate > 2.0
    assert cs.spectral_radius(model, 0.2, n_theta=33)["lambda_max"] > 1.0 + cs.STABILITY_TOL


# ---------------------------------------------------------------------------
# M4 envelope, digest and the recommendation gate
# ---------------------------------------------------------------------------


def _toy_envelope() -> dict:
    planes = {
        "bgk:canonical": ev.scan_plane("bgk", "canonical", n_tau=5, n_ma=4, n_k=7),
        "mrt_a:canonical": ev.scan_plane("mrt_a", "canonical", n_tau=5, n_ma=4, n_k=7),
    }
    foil = ev.scan_plane("mrt_a", "oracle", n_tau=5, n_ma=4, n_k=7)
    return ev.assemble_envelope(planes, {"mismatched": foil}, None, runtime_seconds=0.0)


def test_envelope_digest_is_reproducible() -> None:
    first = _toy_envelope()
    second = _toy_envelope()
    assert first["digest"] == second["digest"]
    assert first["digest"] == cs.envelope_digest(first)
    assert len(first["digest"]) == 64


def test_gate_admits_interior_and_rejects_outside() -> None:
    envelope = _toy_envelope()
    inside = ev.point_in_envelope(envelope, 0.8, 0.1)
    assert inside["inside"] is True
    outside = ev.point_in_envelope(envelope, 0.99, 0.1)
    assert outside["inside"] is False and outside["reason"] == "OUTSIDE_DECLARED_WINDOW"
    unknown = ev.point_in_envelope(envelope, 0.8, 0.1, tier="not_a_tier")
    assert unknown["inside"] is False and unknown["reason"] == "UNKNOWN_TIER"
    nonfinite = ev.point_in_envelope(envelope, float("nan"), 0.1)
    assert nonfinite["inside"] is False


def test_gate_rejects_cell_next_to_unstable_region() -> None:
    envelope = {
        "operating_window": {
            "tau": [0.505, 0.9],
            "ma": [0.02, 0.5],
            "tau_axis": [0.505, 0.6, 0.7],
            "ma_axis": [0.02, 0.25, 0.5],
        },
        "tiers": {"bgk:canonical": {"stable_mask": [[0, 0, 0], [1, 1, 1], [1, 1, 1]]}},
    }
    assert ev.point_in_envelope(envelope, 0.7, 0.5)["inside"] is True
    blocked = ev.point_in_envelope(envelope, 0.6, 0.5)
    assert blocked["inside"] is False
    assert blocked["reason"] == "INSIDE_OR_NEAR_UNSTABLE_REGION"


def test_accuracy_recommendation_gate_is_fail_closed() -> None:
    envelope = {
        "operating_window": {
            "tau": [0.505, 0.9],
            "ma": [0.02, 0.5],
            "tau_axis": [0.505, 0.7, 0.9],
            "ma_axis": [0.02, 0.25, 0.5],
        },
        "tiers": {"bgk:canonical": {"stable_mask": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]}},
    }
    withheld = recommend_with_stability_envelope([], envelope, 0.95, 0.1)
    assert withheld.status == "WITHHELD_OUTSIDE_STABILITY_ENVELOPE"
    assert withheld.recommended_candidate_id is None
    # inside the envelope the ordinary (evidence-driven) path is taken
    admitted = recommend_with_stability_envelope([], envelope, 0.7, 0.1)
    assert admitted.status == "WITHHELD_NO_PHYSICAL_ACCURACY_EVIDENCE"


def test_canonical_json_rounds_and_rejects_non_finite() -> None:
    text = cs.canonical_json({"a": 0.1234567890123456, "b": [1, 2]})
    assert json.loads(text)["a"] == pytest.approx(0.123456789012, abs=0)
    with pytest.raises(ValueError):
        cs.canonical_json({"a": float("nan")})


@pytest.mark.parametrize("path", [
    "TensorLBM/artifacts/p482_n05/stability_envelope.json",
])
def test_campaign_artefact_verifies_if_present(path: str) -> None:
    p = pathlib.Path(__file__).resolve().parents[1] / path
    if not p.exists():  # pragma: no cover - the campaign run may not have happened yet
        pytest.skip("campaign envelope artefact not present")
    result = ev.verify_envelope(p)
    assert result["intact"] is True
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["m2_planes"]["n_cases_per_tier"] >= 900
    assert data["tiers"]["bgk:canonical"]["stable_mask"]
