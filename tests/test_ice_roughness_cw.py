"""Tests for the ice-keel roughness -> ice/water drag closure (P103W2B-N02).

Coverage:
* geometry   : ridge statistics -> k_s / lambda, rib pitch = lambda, sand control
* regime law : analytic limits (smooth / fully rough), k0 inversion & fit,
                monotonicity of Cf/Cf_smooth, Nikuradse calibration,
                smooth-channel DNS reference values
* drift      : free-drift balance residual, angle monotonicity in C_w,
                amplification back-fill fixed point, Nansen reference
* campaign   : case table sanity **and the log-window design constraint**
                (3 k_s^+ < 0.35 Re_tau), which the first campaign violated
* engine     : laminar Poiseuille anchor for the native D3Q19 back-end -- the
                regression test for the TRT forcing bookkeeping defect and for
                the -march=native miscompilation (see the module docstrings)
"""

from __future__ import annotations

import math

import pytest

from tensorlbm.ice_roughness import freedrift as fd
from tensorlbm.ice_roughness import geometry as geom
from tensorlbm.ice_roughness import regime_law as rl
from tensorlbm.ice_roughness import sweep


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def test_keel_roughness_rms_and_spacing():
    depths = [2.0, 3.0, 1.8, 4.0]
    spac = [12.0, 30.0, 20.0, 40.0]
    kr = geom.ice_keel_roughness(depths, spac, level_ice_draft_m=1.5)
    heights = [d - 1.5 for d in depths]
    assert kr.ks_m == pytest.approx(math.sqrt(sum(h * h for h in heights) / len(heights)))
    assert kr.lambda_m == pytest.approx(sum(spac) / len(spac))
    assert kr.lambda_over_ks == pytest.approx(kr.lambda_m / kr.ks_m)
    assert kr.keel_draft_m == pytest.approx(sum(depths) / len(depths))


def test_keel_roughness_rejects_bad_input():
    with pytest.raises(ValueError):
        geom.ice_keel_roughness([], [10.0])
    with pytest.raises(ValueError):
        geom.ice_keel_roughness([1.0], [10.0], level_ice_draft_m=1.5)


def test_keel_uncertainty_brackets_rms():
    u = geom.keel_roughness_uncertainty([2.0, 3.0, 6.0], [10.0, 30.0], level_ice_draft_m=1.5)
    assert u["ks_mean_m"] < u["ks_rms_m"] < u["ks_max_m"]
    assert u["lambda_over_ks_rms"] == pytest.approx(20.0 / u["ks_rms_m"])


def test_rib_geometry_pitch_equals_lambda_over_ks():
    h, nz = 24, 50
    for lam in geom.LAMBDA_OVER_KS_LEVELS:
        el = geom.rib_elements_for_case(h, h, nz, ks_lattice=3, lambda_over_ks=lam)
        p = dict(el.params)
        assert p["pitch_over_ks"] == pytest.approx(lam)
        # pitch is lambda = lam * ks lattice units
        assert p["pitch"] == pytest.approx(lam * el.ks_nominal)
        assert el.mask[0].all() is False or el.mask[0].sum() == 0  # nothing on the wall layer
        assert el.mask[1].any()  # ribs stand on the first fluid layer


def test_rib_geometry_rejects_unresolved_ks():
    with pytest.raises(ValueError):
        geom.rib_elements_for_case(24, 24, 50, ks_lattice=0.4, lambda_over_ks=5.0)


# --------------------------------------------------------------------------
# regime law
# --------------------------------------------------------------------------

def test_delta_u_plus_three_regimes():
    k0 = 4.0
    # hydraulically smooth: dU+ grows linearly with ks+ and vanishes
    assert rl.delta_u_plus(0.0, k0) == pytest.approx(0.0)
    assert rl.delta_u_plus(0.01, k0) < 0.01
    # transitional: monotone, derivative bounded by 1/kappa
    lo, hi = rl.delta_u_plus(2.0, k0), rl.delta_u_plus(8.0, k0)
    assert 0.0 < lo < hi
    deriv = (rl.delta_u_plus(4.01 * k0, k0) - rl.delta_u_plus(4.0 * k0, k0)) / math.log(1.0025)
    assert 0.0 < deriv < 1.0 / rl.KAPPA
    # fully rough: slope 1/kappa
    a, b = 1000.0 * k0, 1100.0 * k0
    slope = (rl.delta_u_plus(b, k0) - rl.delta_u_plus(a, k0)) / math.log(b / a)
    assert slope == pytest.approx(1.0 / rl.KAPPA, rel=1e-3)


def test_nikuradse_calibration():
    # k0 is fixed by requiring the fully rough line to be Nikuradse's
    assert rl.K0_NIKURADSE == pytest.approx(math.exp(rl.KAPPA * (8.5 - 5.2)))
    assert rl.K0_NIKURADSE == pytest.approx(3.87, abs=0.01)
    errors = []
    for ks in (100.0, 1000.0, 10000.0):
        # Nikuradse: dU+ = (1/kappa) ln(ks+) + B - 8.5 (the asymptote of the
        # ln(1 + ks+/k0) family; the finite form converges to it from above)
        expected = math.log(ks) / rl.KAPPA + rl.B_SMOOTH - rl.B_FULLY_ROUGH
        got = rl.nikuradse_delta_u_plus(ks)
        errors.append(got - expected)
        assert got >= expected
    assert errors[0] == pytest.approx(0.0925, abs=5e-3)          # 1.2% at ks+=100
    assert errors[2] < 5e-3                                       # <0.05% at ks+=1e4
    assert rl.equivalent_sand_grain_ratio(rl.K0_NIKURADSE) == pytest.approx(1.0)


def test_k0_inversion_and_fit_roundtrip():
    k0_true = 6.3
    ks = [20.0, 60.0, 150.0, 400.0, 900.0]
    du = [float(rl.delta_u_plus(k, k0_true)) for k in ks]
    for k, d in zip(ks, du):
        assert rl.k0_from_delta_u_plus(k, d) == pytest.approx(k0_true, rel=1e-9)
    fit = rl.fit_k0(ks, du)
    assert fit["k0"] == pytest.approx(k0_true, rel=1e-9)
    assert fit["n_points"] == len(ks)
    assert fit["rms_log_residual"] < 1e-9


def test_fit_k0_drops_uninformative_smooth_points():
    # a hydraulically smooth point carries no k0 information and must be skipped
    fit = rl.fit_k0([1.0, 300.0], [0.05, float(rl.delta_u_plus(300.0, 5.0))])
    assert fit["n_points"] == 1
    assert fit["k0"] == pytest.approx(5.0, rel=1e-6)


def test_zone_boundaries_scale_with_k0():
    b_nik = rl.ks_plus_boundaries(rl.K0_NIKURADSE)
    assert b_nik["ks_plus_smooth_upper"] == pytest.approx(5.0)
    assert b_nik["ks_plus_fully_rough_lower"] == pytest.approx(70.0)
    b2 = rl.ks_plus_boundaries(2.0 * rl.K0_NIKURADSE)
    assert b2["ks_plus_fully_rough_lower"] == pytest.approx(140.0)
    assert rl.regime_of(1.0, rl.K0_NIKURADSE) == "hydraulically_smooth"
    assert rl.regime_of(20.0, rl.K0_NIKURADSE) == "transitional"
    assert rl.regime_of(200.0, rl.K0_NIKURADSE) == "fully_rough"


def test_cf_ratio_is_monotone_and_increasing():
    law = rl.ThreeZoneLaw(lambda_over_ks=5.0, k0=4.0)
    cf_smooth = rl.cf_channel_smooth_reference(500.0)
    ks = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    ratios = [law.cf_ratio(k, cf_smooth) for k in ks]
    assert all(r >= 1.0 for r in ratios)
    assert all(b > a for a, b in zip(ratios, ratios[1:]))
    # strongly rough wall cannot reduce the friction
    assert ratios[-1] > 2.0


def test_smooth_channel_reference_matches_dns():
    # Moser, Kim & Mansour (1999) / Lee & Moser (2015) channel DNS
    assert rl.cf_channel_smooth_reference(180.0) == pytest.approx(0.00597, rel=0.01)
    assert rl.cf_channel_smooth_reference(590.0) == pytest.approx(0.00445, rel=0.01)
    with pytest.raises(ValueError):
        rl.cf_channel_smooth_reference(0.5)


def test_interpolate_law_is_monotone_in_log_space():
    laws = {
        2.0: rl.ThreeZoneLaw(2.0, 2.0),
        5.0: rl.ThreeZoneLaw(5.0, 6.0),
        10.0: rl.ThreeZoneLaw(10.0, 12.0),
    }
    prev = 0.0
    for lam in (2.5, 3.5, 6.0, 8.0, 9.5):
        k0 = rl.interpolate_law(lam, laws).k0
        assert k0 > prev
        prev = k0
    assert rl.interpolate_law(1.0, laws).k0 == pytest.approx(2.0)      # clamped low
    assert rl.interpolate_law(50.0, laws).k0 == pytest.approx(12.0)    # clamped high


def test_solve_friction_velocity_is_self_consistent():
    law = rl.ThreeZoneLaw(lambda_over_ks=5.0, k0=6.0)
    sol = rl.solve_friction_velocity(0.1, 0.35, law)
    u_tau = sol["u_tau"]
    assert sol["ks_plus"] == pytest.approx(0.35 * u_tau / rl.NU_WATER, rel=1e-12)
    assert sol["C_w"] == pytest.approx((u_tau / 0.1) ** 2, rel=1e-12)
    # the log law it came from
    rhs = math.log(rl.DEFAULT_REFERENCE_HEIGHT_M * u_tau / rl.NU_WATER) + \
        rl.KAPPA * (rl.B_SMOOTH - law.delta_u_plus(sol["ks_plus"]))
    assert u_tau == pytest.approx(rl.KAPPA * 0.1 / rhs, rel=1e-8)


# --------------------------------------------------------------------------
# free drift / back-fill
# --------------------------------------------------------------------------

def test_free_drift_balance_and_monotonicity():
    env = fd.DriftEnvironment()
    lo = fd.drift_angle(env, 3e-3)
    hi = fd.drift_angle(env, 8e-3)
    # more drag -> slower but straighter drift (the wind stress is imposed):
    # U = sqrt(T /(rho_w C_w sqrt(1+B^2))) and tan(alpha) = rho_i h f /(rho_w C_w U)
    assert lo["drift_angle_deg"] > hi["drift_angle_deg"]
    assert lo["drift_speed_m_s"] > hi["drift_speed_m_s"]
    for row in (lo, hi):
        assert abs(row["balance_residual_Pa"]) < 1e-12
        assert 0.0 < row["drift_angle_deg"] < 90.0


def test_constant_cw_reproduces_documented_drift_angle():
    # C_w = 5.5e-3, h_i = 2 m, f = 1.46e-4, U_a = 5 m/s -> ~32 deg,
    # inside the 20-40 deg band quoted for Nansen drift
    env = fd.DriftEnvironment()
    row = fd.drift_angle(env, fd.CW_SMOOTH_REFERENCE)
    assert 25.0 < row["drift_angle_deg"] < 40.0
    assert fd.NANSEN_ANGLE_BAND_DEG[0] <= row["drift_angle_deg"] <= fd.NANSEN_ANGLE_BAND_DEG[1]


def test_amplification_backfill_fixed_point():
    law = rl.ThreeZoneLaw(lambda_over_ks=5.0, k0=6.0)
    env = fd.DriftEnvironment()
    sol = fd.drift_with_roughness_amplification(env, 0.35, law)
    assert sol["amplification"] >= 1.0
    # the amplification is the log-law ratio at the matched reference height
    z_plus = sol["environment"]["z_ref_m"] * math.sqrt(sol["C_w"]) * sol["drift_speed_m_s"] / \
        sol["environment"]["nu_water"]
    assert sol["amplification"] == pytest.approx(law.cw_amplification(sol["ks_plus"], z_plus), rel=1e-6)
    assert sol["C_w"] == pytest.approx(fd.CW_SMOOTH_REFERENCE * sol["amplification"], rel=1e-6)
    # k_s^+ is consistent with the converged drag and drift speed
    assert sol["ks_plus"] == pytest.approx(
        0.35 * math.sqrt(sol["C_w"]) * sol["drift_speed_m_s"] / env.nu_water, rel=1e-6)
    assert sol["angle_error_deg"] == pytest.approx(
        abs(sol["drift_angle_deg"] - fd.NANSEN_DRIFT_ANGLE_DEG))
    # rougher underside (smaller k0 = closer to sand) drifts straighter
    rough = fd.drift_with_roughness_amplification(env, 0.35, rl.ThreeZoneLaw(5.0, 1.0))
    assert rough["drift_angle_deg"] < sol["drift_angle_deg"]


# --------------------------------------------------------------------------
# campaign table
# --------------------------------------------------------------------------

def test_campaign_table_shape():
    cases = sweep.campaign_cases("campaign")
    rough = [c for c in cases if c.kind == "rough"]
    sand = [c for c in rough if c.morphology == "sand"]
    ribs = [c for c in rough if c.morphology == "ribs"]
    assert len(ribs) == 15          # 5 ks+ x 3 lambda/ks
    assert len(sand) == 5           # Nikuradse control group
    assert len([c for c in cases if c.kind == "smooth"]) == 5
    assert {c.lambda_over_ks for c in ribs} == set(geom.LAMBDA_OVER_KS_LEVELS)
    # one fixed geometry per group: k_s in lattice nodes does not vary within a group
    assert {c.ks_nodes for c in ribs} == {5.0}


def test_campaign_cases_have_a_log_layer():
    """Design constraint learnt the hard way: 3 k_s^+ must be below 0.35 Re_tau."""
    for case in sweep.campaign_cases("campaign"):
        if case.kind != "rough":
            continue
        ks_plus = case.ks_nodes * case.re_tau / case.h_eff_estimate
        lo = 3.0 * ks_plus
        hi = 0.35 * case.re_tau
        assert lo < hi, f"{case.case_id}: 3ks+={lo:.1f} >= 0.35 Re_tau={hi:.1f} (no log layer)"


def test_case_config_is_numerically_admissible():
    for case in sweep.campaign_cases("campaign"):
        cfg = case.config()
        assert 0.5 < cfg.tau < 1.2 if hasattr(cfg, "tau") else True
        assert cfg.steps > cfg.warmup > 0
        # bulk Mach number of a forced channel stays in the incompressible range
        assert cfg.u_tau * 25.0 * math.sqrt(1.0 / 3.0) < 0.3


def test_run_case_records_are_aggregatable(tmp_path):
    """The aggregation must work on records alone (no live run needed)."""
    import json

    def fake(case, cf, ks_plus, d_profile, log_valid=True, b=4.0):
        return {
            "case": case.to_json(), "cf": cf, "ks_plus": ks_plus, "re_tau": case.re_tau,
            "u_plus_bulk": math.sqrt(2.0 / cf), "u_tau": 0.012, "h_eff": case.h_eff_estimate,
            "dy_plus": case.re_tau / case.h_eff_estimate, "ks_nodes": case.ks_nodes,
            "log_fit": {"b": b} if log_valid else None,
            "log_window": {"n_points": 5 if log_valid else 0},
            "wall_seconds": 1.0, "backend": "fake",
        }

    cases = sweep.campaign_cases("campaign")
    for case in cases:
        if case.kind == "smooth":
            rec = fake(case, rl.cf_channel_smooth_reference(case.re_tau), 0.0, 0.0, log_valid=False, b=5.2)
        else:
            k0 = 4.0 if case.morphology == "ribs" else rl.K0_NIKURADSE
            ks_plus = case.ks_nodes * case.re_tau / case.h_eff_estimate
            d = float(rl.delta_u_plus(ks_plus, k0))
            smooth_cf = rl.cf_channel_smooth_reference(case.re_tau)
            cf = smooth_cf * (math.sqrt(2.0 / smooth_cf) / (math.sqrt(2.0 / smooth_cf) - d)) ** 2
            rec = fake(case, cf, ks_plus, d, b=5.2 - d)
        (tmp_path / f"{case.case_id}.json").write_text(json.dumps(rec))

    summary = sweep.aggregate(tmp_path)
    assert summary["n_rough"] == 20 and summary["n_smooth"] == 5
    assert summary["C1_smooth_wall_cf"]["passed"] is True
    assert summary["C2_fully_rough_vs_nikuradse"]["passed"] is True
    assert summary["C3_transition_monotone"]["passed"] is True
    assert "C4_nansen_drift_angle" in summary
    # the synthetic rib law must be recovered from the records
    k0_rib = {g["group"]: g["fit"]["k0"] for g in summary["groups"]}["5"]
    assert k0_rib == pytest.approx(4.0, rel=0.15)


# --------------------------------------------------------------------------
# engine anchors (native back-end)
# --------------------------------------------------------------------------

def _native_ready() -> bool:
    from tensorlbm.rough_channel3d_native import native_available

    return native_available()


@pytest.mark.skipif(not _native_ready(), reason="no C compiler for the native D3Q19 back-end")
def test_native_laminar_poiseuille_anchor():
    """Forced-channel momentum balance: U_b^+ -> Re_tau/3 in the laminar limit.

    This is the regression test for the TRT forcing bookkeeping defect: with the
    force weighted by ``(1 - s_plus/2)`` instead of ``(1 - s_minus/2)`` the
    channel is driven by ``F [1 + (s_minus - s_plus)/2]`` and U_b^+ came out at
    ~0.5 Re_tau/3 (measured: 1.67 vs 3.33 at Re_tau = 10).  It also fails
    outright when the native code is compiled with ``-march=native`` on hosts
    whose GCC vectorises the loops differently (all cases NaN).
    """
    from tensorlbm.rough_channel3d import RoughChannelConfig
    from tensorlbm.rough_channel3d_native import run_rough_channel_native

    # h=10 with ~3 viscous diffusion times (h^2/nu) so that the *laminar*
    # steady state is reached: with fewer steps the test measures the startup
    # transient (4 eddy turnovers gave U_b^+ = 3.59 instead of 3.33).
    h, u_tau, re_tau = 10, 0.02, 10.0
    steps = 17000
    cfg = RoughChannelConfig(
        nx=h, ny=h, nz=2 * h + 2, re_tau=re_tau, u_tau=u_tau,
        steps=steps, warmup=steps // 3, sample_interval=10, threads=2,
        use_compile=False, init_scale=0.5, noise_amplitude=0.0, smagorinsky_cs=0.0,
    )
    rec = run_rough_channel_native(cfg, None, smooth_b_reference=5.2)
    # laminar plane-Poiseuille: U_b^+ = Re_tau/3 (u_tau^2 = g h_eff); the
    # bounce-back wall sits half a cell outside the last fluid node, which is
    # already folded into h_eff.  4% covers the h_eff discretisation at h=12.
    assert rec["u_plus_bulk"] == pytest.approx(re_tau / 3.0, rel=0.04)
    assert rec["cf"] == pytest.approx(2.0 / (re_tau / 3.0) ** 2, rel=0.10)