"""Convergence-order evidence for D2Q9: MMS + Guo forcing + wall-position variants.

These tests are the CI face of ``src/tensorlbm/convergence_mms.py`` and of the
campaign node ``p121w1-n01-lbm-convergence`` (mission m7059).  They are split in
two layers:

* **fast physics layer** -- small-grid refinement studies that must reproduce the
  *qualitative* order of each configuration (second order with the manufactured
  force and with interpolated bounce-back; first order for the two negative
  controls).  Runtime is a few seconds per study.
* **artifact layer** -- the committed evidence record
  ``artifacts/lbm_convergence_m7059/convergence_evidence.json`` is validated
  field by field, its numbers are recomputed by an independent implementation,
  and the declared criteria are re-evaluated from the JSON alone (no re-run).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.convergence_mms import (
    CS2,
    OPPOSITE,
    SCHEMA_VERSION,
    C,
    equilibrium,
    fit_effective_wall,
    guo_source,
    macroscopic,
    mms_fields,
    mms_force_field,
    observed_order,
    observed_order_independent,
    poiseuille_exact,
    rel_l2,
    richardson_order_nested,
    run_mms_2d,
    run_poiseuille_2d,
    study_mms,
    study_poiseuille,
    validate_evidence,
)

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "lbm_convergence_m7059"
    / "convergence_evidence.json"
)


# --------------------------------------------------------------------------------------
# layer 0: the building blocks agree with the production solver
# --------------------------------------------------------------------------------------
def test_equilibrium_and_moments_match_reference_solver():
    """NumPy kernels are bit-compatible with the torch reference solver."""
    torch = pytest.importorskip("torch")
    from tensorlbm.solver import macroscopic as ref_macroscopic

    rng = np.random.default_rng(7059)
    rho = 1.0 + 1e-3 * rng.standard_normal((8, 8))
    ux = 0.01 * rng.standard_normal((8, 8))
    uy = 0.01 * rng.standard_normal((8, 8))
    f_np = equilibrium(rho, ux, uy)
    t_rho = torch.from_numpy(rho)
    t_ux = torch.from_numpy(ux)
    t_uy = torch.from_numpy(uy)
    from tensorlbm.solver import equilibrium as ref_equilibrium

    f_t = ref_equilibrium(t_rho, t_ux, t_uy).numpy()
    assert np.max(np.abs(f_np - f_t)) == 0.0

    r_np, x_np, y_np = macroscopic(f_np)
    r_t, x_t, y_t = ref_macroscopic(torch.from_numpy(f_np))
    assert np.allclose(r_np, r_t.numpy(), rtol=0, atol=1e-15)
    assert np.allclose(x_np, x_t.numpy(), rtol=0, atol=1e-15)
    assert np.allclose(y_np, y_t.numpy(), rtol=0, atol=1e-15)


def test_guo_source_conserves_mass_and_delivers_the_force():
    """Guo's discrete force term has zero mass and the prescribed momentum rate."""
    rng = np.random.default_rng(11)
    n = 6
    rho = np.ones((n, n))
    f = equilibrium(rho, 0.01 * rng.standard_normal((n, n)), 0.01 * rng.standard_normal((n, n)))
    fx = 0.3 * rng.standard_normal((n, n))
    fy = 0.3 * rng.standard_normal((n, n))
    tau = 0.8
    s, rho_out, ux, uy = guo_source(f, fx, fy, tau)
    assert np.max(np.abs(s.sum(axis=0))) < 1e-15  # mass source vanishes identically
    mx = (s * C[:, 0].astype(float).reshape(9, 1, 1)).sum(axis=0)
    my = (s * C[:, 1].astype(float).reshape(9, 1, 1)).sum(axis=0)
    # sum_i c_i S_i = (1 - 1/(2 tau)) F
    assert np.allclose(mx, (1.0 - 0.5 / tau) * fx, rtol=0, atol=1e-16)
    assert np.allclose(my, (1.0 - 0.5 / tau) * fy, rtol=0, atol=1e-16)
    # half-force macroscopic velocity
    rho_ref, ux_raw, uy_raw = macroscopic(f)
    assert np.allclose(ux, ux_raw + 0.5 * fx / rho_ref, rtol=0, atol=1e-18)
    assert np.allclose(uy, uy_raw + 0.5 * fy / rho_ref, rtol=0, atol=1e-18)


def test_opposite_table_is_an_involution():
    assert all(int(OPPOSITE[int(OPPOSITE[i])]) == i for i in range(9))


def test_order_estimators_are_mutually_consistent():
    """Two independent implementations must agree to round-off on synthetic data."""
    h = [1.0 / 24, 1.0 / 48, 1.0 / 96]
    e = [3.1e-3, 7.8e-4, 1.9e-4]
    a = observed_order(h, e)
    b = observed_order_independent(h, e)
    assert abs(a["p"] - b["p"]) < 1e-12
    # exact second-order data must be recovered exactly by both (and by Richardson)
    e2 = [4.0, 1.0, 0.25]
    assert abs(observed_order([4.0, 2.0, 1.0], e2)["p"] - 2.0) < 1e-12
    assert abs(observed_order_independent([4.0, 2.0, 1.0], e2)["p"] - 2.0) < 1e-12
    assert abs(richardson_order_nested(e2, ratio=2.0) - 2.0) < 1e-12


# --------------------------------------------------------------------------------------
# layer 1: fast physics -- each configuration reproduces its own order
# --------------------------------------------------------------------------------------
def test_mms_manufactured_force_is_second_order():
    """MMS + Guo forcing: observed order 2 within +/-0.1 on three grids."""
    study = study_mms((16, 32, 64), u0=0.003, tau=1.0, force=True, tol=1e-11)
    p = study["order_fit"]["p"]
    assert 1.9 <= p <= 2.1, (p, [lv["l2_rel"] for lv in study["levels"]])
    assert abs(p - 2.0) <= 0.1
    # the two fits must agree to round-off, and errors must decrease monotonically
    assert abs(p - study["order_fit_independent"]["p"]) < 1e-12
    errs = [lv["l2_rel"] for lv in study["levels"]]
    assert errs[0] > errs[1] > errs[2] > 1e3 * np.finfo(np.float64).eps
    for lv in study["levels"]:
        assert lv["converged"], lv
        assert lv["div_l2"] < 1e-5  # incompressibility of the discrete steady state


def test_mms_negative_control_without_force_does_not_converge():
    """Gate-can-fail witness: with the manufactured source off, order collapses."""
    study = study_mms((16, 32, 64), u0=0.003, tau=1.0, force=False, tol=1e-11)
    p = study["order_fit"]["p"]
    assert p <= 1.2, p
    # the field decays to rest, so the error saturates at O(1)
    for lv in study["levels"]:
        assert lv["l2_rel"] > 0.5


def test_mms_reaches_the_same_steady_state_from_rest():
    """The measured steady state is a fixed point, not the initial condition."""
    n = 24
    a = run_mms_2d(n, u0=0.01, force=True, init="exact", tol=1e-11)
    b = run_mms_2d(n, u0=0.01, force=True, init="rest", tol=1e-11)
    assert b["steps"] > a["steps"]
    d = np.max(np.abs(a["fields"]["ux"] - b["fields"]["ux"]))
    assert d / np.max(np.abs(a["fields"]["ux_exact"])) < 1e-4
    # both routes are stopped by the same residual tolerance, so they differ only
    # by the transient lag (residual / decay rate), not by a different fixed point
    assert abs(a["l2_rel"] - b["l2_rel"]) / a["l2_rel"] < 1e-4


def test_poiseuille_bouzidi_is_second_order():
    """Quadratic interpolated bounce-back at the true fractional wall: order 2."""
    study = study_poiseuille(
        (12, 24, 48), q=0.25, wall_model="bouzidi", reference="true_wall", u_max=0.01, tol=1e-9
    )
    p = study["order_fit"]["p"]
    assert p >= 1.9, (p, [lv["l2_rel"] for lv in study["levels"]])
    assert study["levels"][-1]["l2_rel"] < 1e-2  # < 1 % at the finest grid
    assert abs(p - study["order_fit_independent"]["p"]) < 1e-12


def test_poiseuille_halfway_wall_is_first_order_and_the_cause_is_the_offset():
    """Half-way bounce-back fixes y_w = 0.5 Delta; measured against the true wall
    geometry that is a first-order wall-position error -- and the profile itself
    shows the wall sitting at -(0.5 - q) cell."""
    study = study_poiseuille(
        (12, 24, 48), q=0.25, wall_model="halfway", reference="true_wall", u_max=0.01, tol=1e-9
    )
    p = study["order_fit"]["p"]
    assert p <= 1.2, (p, [lv["l2_rel"] for lv in study["levels"]])
    finest = study["levels"][-1]
    assert finest["wall_offset_left"] == pytest.approx(-(0.5 - 0.25), abs=0.02)
    assert finest["wall_offset_right"] == pytest.approx(+(0.5 - 0.25), abs=0.02)


def test_poiseuille_halfway_with_effective_wall_reference_recovers_second_order():
    """Same data, reference shifted to y_w = 0.5 Delta: the O(Delta) part disappears."""
    study = study_poiseuille(
        (12, 24, 48),
        q=0.25,
        wall_model="halfway",
        reference="effective_wall",
        u_max=0.01,
        tol=1e-11,
    )
    assert study["order_fit"]["p"] >= 1.8, study["order_fit"]
    true_wall = study_poiseuille(
        (12, 24, 48), q=0.25, wall_model="halfway", reference="true_wall", u_max=0.01, tol=1e-11
    )
    # the offset correction removes >90 % of the error at every level
    for eff, tru in zip(study["levels"], true_wall["levels"]):
        assert eff["l2_rel"] < 0.1 * tru["l2_rel"]


def test_poiseuille_profile_and_effective_wall_fit():
    """A single run: parabolic shape, mobility, and the wall recovered by fitting."""
    nu = CS2 * (1.0 - 0.5)
    n_f = 24
    q = 0.25
    height = (n_f - 1) + 2 * q
    g = 8.0 * nu * 0.01 / height**2
    r = run_poiseuille_2d(n_f, q=q, wall_model="bouzidi", g=g, init="exact", tol=1e-10)
    exact = poiseuille_exact(r["y"], g, nu, r["y_left_true"], r["y_right_true"])
    assert rel_l2(r["ux"], np.zeros_like(r["ux"]), exact, np.zeros_like(exact)) < 1e-2
    # analytic maximum of the parabola
    u_max_exact = g * height**2 / (8.0 * nu)
    assert np.max(r["ux"]) == pytest.approx(u_max_exact, rel=5e-3)
    fit = fit_effective_wall(r["y"], r["ux"], g, nu)
    assert fit["quadratic_coeff"] == pytest.approx(-g / (2.0 * nu), rel=1e-3)
    # the windowed parabola fit recovers the wall to a few hundredths of a cell;
    # the *difference* between wall models (see the negative-control test) is exact
    assert fit["wall_left_fit"] == pytest.approx(0.0, abs=0.05)
    assert fit["wall_right_fit"] == pytest.approx(height, abs=0.05)
    assert abs(r["uy_max"]) < 1e-10


# --------------------------------------------------------------------------------------
# layer 2: the committed evidence artifact
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def evidence() -> dict:
    if not ARTIFACT.exists():
        pytest.skip(f"evidence artifact not generated: {ARTIFACT}")
    return json.loads(ARTIFACT.read_text())


def test_artifact_declares_schema_and_passes_its_own_gate(evidence):
    assert evidence["schema_version"] == SCHEMA_VERSION
    check = validate_evidence(evidence)
    assert check["ok"], check["problems"]
    assert check["max_order_disagreement"] < 1e-12
    assert evidence["overall_pass"] is True


def test_artifact_criteria_are_all_satisfied_and_cover_the_node_contract(evidence):
    ids = {c["id"] for c in evidence["criteria"]}
    assert {"a", "b", "c1", "c2", "d1", "d2", "e", "f", "floor", "g"} <= ids
    failing = [c for c in evidence["criteria"] if not c["pass"]]
    assert failing == []
    by_id = {c["id"]: c for c in evidence["criteria"]}
    assert 1.9 <= by_id["a"]["value"] <= 2.1
    assert by_id["c1"]["value"] <= 1.2
    assert by_id["c2"]["value"] <= 1.2
    assert by_id["d1"]["value"] < 1.0  # percent
    assert by_id["d2"]["value"] >= 1.9
    assert by_id["e"]["value"] < 1e-12
    assert by_id["f"]["value"] <= 1e-12
    assert by_id["g"]["value"] >= 0  # regression pass count delta
    reg = evidence["regression"]
    assert reg["after"]["passed"] >= reg["baseline"]["passed"]
    assert reg["after"]["failed"] <= reg["baseline"]["failed"]


def test_artifact_orders_are_recomputable_from_the_stored_errors(evidence):
    """Independent recomputation (criterion e/f): every recorded order is re-derived
    from the stored grid/error pairs to < 1e-12."""
    for name, case in evidence["cases"].items():
        levels = case["levels"]
        if name.startswith("mms"):
            h = [1.0 / lv["n"] for lv in levels]
        else:
            h = [1.0 / lv["n_f"] for lv in levels]
        e = [lv["l2_rel"] for lv in levels]
        rec = observed_order_independent(h, e)
        assert abs(rec["p"] - case["order_fit"]["p"]) < 1e-12, name
        assert abs(rec["p"] - case["order_fit_independent"]["p"]) < 1e-12, name


def test_artifact_negative_controls_really_fail(evidence):
    p_neg = evidence["cases"]["mms_no_force"]["order_fit"]["p"]
    p_hbb = evidence["cases"]["poiseuille_halfway_true_wall"]["order_fit"]["p"]
    assert p_neg <= 1.2 and p_hbb <= 1.2
    # and the corresponding positive configurations are genuinely second order
    assert evidence["cases"]["mms_force"]["order_fit"]["p"] >= 1.9
    assert evidence["cases"]["poiseuille_bouzidi_true_wall"]["order_fit"]["p"] >= 1.9


def test_validate_evidence_is_fail_closed(evidence):
    import copy

    broken = copy.deepcopy(evidence)
    broken["criteria"][0]["value"] += 1.0  # tamper -> hash mismatch
    bad = validate_evidence(broken)
    assert not bad["ok"]
    assert any("sha256" in p for p in bad["problems"])

    broken = copy.deepcopy(evidence)
    broken["cases"]["mms_force"]["order_fit"]["p"] += 1e-6  # tamper -> hash mismatch
    assert not validate_evidence(broken)["ok"]

    broken = copy.deepcopy(evidence)
    broken["extra_key"] = 1
    bad = validate_evidence(broken)
    assert not bad["ok"]
    assert any("unexpected top-level keys" in p for p in bad["problems"])

    broken = copy.deepcopy(evidence)
    broken["cases"]["mms_force"]["levels"][0].pop("l2_rel")
    bad = validate_evidence(broken)
    assert not bad["ok"]
    assert any("missing level fields" in p for p in bad["problems"])


def test_artifact_protocol_is_predeclared_and_documented(evidence):
    proto = evidence["protocol"]
    assert proto["criteria_predeclared"] is True
    assert len(proto["mms_grids"]) == 3 and proto["mms_grids"] == [32, 64, 128]
    assert proto["poiseuille_wall_fraction_q"] == pytest.approx(0.25)
    assert proto["nu"] == pytest.approx(CS2 * (proto["tau"] - 0.5))
    assert "m5740" in evidence["notes"]["criterion_e"]  # honest provenance note
    assert evidence["audit"]["payload_sha256"]
    assert math.isfinite(evidence["generator"]["wall_clock_s"])


def test_mms_fields_use_lattice_wavenumber():
    """k must be 2*pi*waves/n: h = 1 lattice unit (a fixed k=2*pi would be one cell)."""
    for n in (16, 32, 64):
        _x, _y, ux, uy, h, k = mms_fields(n, 0.01, waves=1)
        assert h == 1.0
        assert k == pytest.approx(2.0 * math.pi / n)
        # |u|^2 = U0^2 (cos^2 sin^2 + sin^2 cos^2) is not constant on the lattice
        speed2 = ux * ux + uy * uy
        assert np.max(np.sqrt(speed2)) <= 0.01 * (1 + 1e-12)
        assert np.max(np.sqrt(speed2)) > 0.9 * 0.01


def test_manufactured_force_balances_viscosity_for_the_analytic_field():
    """F/rho = 2 nu k^2 u is exactly the force that makes the field a steady
    solution of the incompressible momentum equation."""
    n = 32
    _x, _y, ux, uy, _h, k = mms_fields(n, 0.01)
    nu = 1.0 / 6.0
    rho = np.ones_like(ux)
    fx, fy = mms_force_field(rho, ux, uy, nu, k)
    # spectral Laplacian of the analytic field equals -2 k^2 u (the continuous
    # identity the derivation rests on; finite differences would only be O(k^2 h^2))
    ky = np.fft.fftfreq(n, d=1.0) * 2.0 * math.pi
    kx = np.fft.rfftfreq(n, d=1.0) * 2.0 * math.pi
    kk = kx[None, :] ** 2 + ky[:, None] ** 2
    lap_x = np.fft.irfft2(-kk * np.fft.rfft2(ux), s=ux.shape)
    assert np.allclose(lap_x, -2.0 * k * k * ux, rtol=0, atol=1e-12)
    assert np.allclose(fx, -rho * nu * lap_x, rtol=0, atol=1e-14)
