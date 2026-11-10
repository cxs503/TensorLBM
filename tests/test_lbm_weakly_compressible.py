"""Machine-checkable tests for the weak-compressibility campaign surface.

Two layers:

* **unit / synthetic** (fast, no time marching): the pressure-Poisson solver,
  the differential-Ma diagnostic, the Mach-independence / applicability
  boundary logic, the gate assembly and the artifact writers.  Synthetic
  records are used so that a *known* violation must trip the corresponding
  gate -- a gate that cannot fail is not a gate.
* **artifact integration** (runs only when the campaign artifacts exist):
  re-reads ``benchmarks/verified/lbm_weakly_compressible/{result,gates}.json``
  and asserts that the recorded numbers satisfy the criteria they claim and
  that every criterion C1..C8 is present with a status.

The physics campaign itself is produced by
``python -m tensorlbm.verification.weakly_compressible`` (or
``scripts/run_lbm_weakly_compressible_campaign.py``); these tests never
time-march a cylinder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.verification import weakly_compressible as wv

ARTIFACTS = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "lbm_weakly_compressible"


# ---------------------------------------------------------------------------
# C6: pressure Poisson reference
# ---------------------------------------------------------------------------
def test_poisson_reference_reproduces_the_linear_channel_solution():
    nx, ny = 40, 11
    p, info = wv.poisson_reference_pressure(nx, ny, p_in=3.0, p_out=1.0)
    expect = np.linspace(3.0, 1.0, nx)
    assert np.max(np.abs(p - expect[None, :])) < 1e-12
    assert np.max(np.abs(p[0, :] - p[-1, :])) < 1e-12  # dp/dy = 0 on the walls
    assert info["converged"]


def test_poisson_solver_converges_from_a_perturbed_guess():
    """The SOR sweep must actually iterate (not just return the initial guess)."""
    nx, ny = 30, 9
    p, info = wv.poisson_reference_pressure(nx, ny, p_in=1.0, p_out=0.0, max_iter=4000)
    # a genuine 2-D solve that ignored the equations would keep the perturbation
    assert info["iterations"] > 0
    assert info["residual"] < 1e-10
    assert np.max(np.abs(p - np.linspace(1.0, 0.0, nx)[None, :])) < 1e-10


def test_pressure_poisson_criterion_flags_a_bad_pressure_field():
    nx, ny, accel = 48, 17, 4.0e-3
    ref = wv.CS2 * 1.0 + accel * np.arange(nx - 1, -1, -1, dtype=float)
    good = {
        "p_lbm_x": ref.tolist(),
        "ny": ny,
        "accel": accel,
        "mach": 0.05,
        "mach_measured": 0.05,
        "nu": 0.05,
        "tau": 0.65,
        "u_max_probe": 0.0289,
        "nx": nx,
    }
    crit_good = wv.pressure_poisson_criterion(good)
    assert crit_good["r2"] > 0.999
    assert crit_good["max_abs_dev"] < 1e-12
    bad = dict(good, p_lbm_x=(ref + 0.02 * np.sin(np.linspace(0, 6.28, nx))).tolist())
    crit_bad = wv.pressure_poisson_criterion(bad)
    assert crit_bad["r2"] < crit_good["r2"]
    assert crit_bad["max_abs_dev"] > 1e-3


def test_pressure_poisson_sweep_recovers_ma2_deviation_order():
    """Synthetic sweep with dev = 1e-1 * Ma^2 must be fitted back as order 2."""
    machs = [0.01, 0.05, 0.1, 0.2, 0.3]
    runs = []
    for ma in machs:
        nx, ny, accel = 48, 17, 4.0e-3
        ref = wv.CS2 + accel * np.arange(nx - 1, -1, -1, dtype=float)
        runs.append(
            {
                "p_lbm_x": (ref + 1e-1 * ma**2).tolist(),
                "ny": ny,
                "accel": accel,
                "mach": ma,
                "mach_measured": ma,
                "nu": 0.05,
                "tau": 0.65,
                "u_max_probe": ma * 0.577,
            }
        )
    sweep = wv.pressure_poisson_sweep({"runs": runs})
    assert abs(sweep["fit_dev_vs_ma"]["exponent_plain"] - 2.0) < 0.05
    assert all(conv for conv in sweep["poisson_converged"])


# ---------------------------------------------------------------------------
# C1: differential Ma scaling
# ---------------------------------------------------------------------------
def _synthetic_channel_sweep(machs, exponent=2.0, amplitude=1.0):
    """Profiles whose difference from the lowest-Ma run is ``A * Ma^p``."""
    y = np.linspace(0.0, 1.0, 31)
    base = 1.0 - (y - 0.5) ** 2 * 4.0
    base = base / base.max()
    runs = []
    for ma in machs:
        prof = base + amplitude * ma**exponent * (base - base.mean())
        nx, ny, accel = 48, 17, 4.0e-3
        p_ref = wv.CS2 + accel * np.arange(nx - 1, -1, -1, dtype=float)
        runs.append(
            {
                "mach": ma,
                "mach_measured": ma,
                "u_num": prof.tolist(),
                "err_inf": 0.07 + 1e-3 * ma,  # floor-dominated, Ma-independent
                "drho_over_rho0": 3.0 * ma**2,
                "nu": 0.05,
                "tau": 0.65,
                "ny": ny,
                "accel": accel,
                "u_max_probe": ma * 0.577,
                "p_lbm_x": (p_ref + 5e-1 * ma**2).tolist(),
            }
        )
    return {
        "case": "synthetic_channel_sweep",
        "runs": runs,
        "machs": [r["mach_measured"] for r in runs],
        "err_inf": [r["err_inf"] for r in runs],
        "err_l2": [r["err_inf"] for r in runs],
        "drho_over_rho0": [r["drho_over_rho0"] for r in runs],
        "nu": [r["nu"] for r in runs],
        "fit_error_raw": {"exponent_plain": 0.4},
        "fit_error_excess": {"exponent_excess": 2.0, "r2_excess": 0.99},
        "fit_drho": {"exponent_plain": 2.0},
    }


def test_differential_error_recovers_the_ma2_order():
    sweep = _synthetic_channel_sweep([0.01, 0.05, 0.1, 0.2, 0.3], exponent=2.0)
    out = wv.profile_error_vs_lowest_mach(sweep)
    assert abs(out["exponent_diff"] - 2.0) < 0.05
    assert out["diff_err_inf"][0] == pytest.approx(0.0, abs=1e-12)


def test_differential_error_distinguishes_order_one_from_order_two():
    s2 = wv.profile_error_vs_lowest_mach(_synthetic_channel_sweep([0.01, 0.05, 0.1, 0.2, 0.3], exponent=2.0))
    s1 = wv.profile_error_vs_lowest_mach(_synthetic_channel_sweep([0.01, 0.05, 0.1, 0.2, 0.3], exponent=1.0))
    assert abs(s2["exponent_diff"] - 2.0) < 0.05
    # the first-order control is biased slightly high (the reference run's own
    # Ma_ref term is subtracted, adding a small negative offset); it must still
    # be clearly separated from the second-order case
    assert abs(s1["exponent_diff"] - 1.0) < 0.1


# ---------------------------------------------------------------------------
# C2 / C5 / C7: Mach independence and the applicability boundary
# ---------------------------------------------------------------------------
def _cyl_run(ma, st, cd, cl, converged=True, blowup=-1):
    return {
        "mach": ma,
        "mach_measured": ma,
        "st": st,
        "cd_mean": cd,
        "cl_amplitude": cl,
        "converged": converged,
        "blowup_step": blowup,
        "steps": 1000,
        "nu": 1e-3,
        "tau": 0.503,
    }


def test_mach_independent_hydrodynamics_passes_c2_and_c5():
    runs = [_cyl_run(ma, 0.1650, 1.360, 0.330) for ma in (0.05, 0.1, 0.2)]
    out = wv.hydrodynamic_mach_independence({"runs": runs})
    assert out["max_rel_st_low_ma"] == pytest.approx(0.0, abs=1e-12)
    assert out["max_rel_cd_low_ma"] == pytest.approx(0.0, abs=1e-12)
    assert out["pass_c2_lowest_usable_reference"] is True
    assert out["pass_c5_lowest_usable_reference"] is True
    assert out["c7_triggered"] is False
    assert out["ma_max"] == pytest.approx(0.2)


def test_mach_dependent_strouhal_fails_c2_and_caps_ma_max():
    runs = [
        _cyl_run(0.05, 0.1650, 1.360, 0.330),
        _cyl_run(0.1, 0.1680, 1.361, 0.330),
        _cyl_run(0.2, 0.1765, 1.370, 0.331),
    ]
    out = wv.hydrodynamic_mach_independence({"runs": runs})
    assert out["max_rel_st_low_ma"] > 0.01
    assert out["pass_c2_lowest_usable_reference"] is False
    assert out["c7_triggered"] is True
    assert out["ma_max"] < 0.2


def test_diverged_run_is_reported_as_unusable_not_as_a_result():
    runs = [
        _cyl_run(0.01, float("nan"), float("nan"), float("nan"), converged=False, blowup=300),
        _cyl_run(0.05, 0.1650, 1.360, 0.330),
        _cyl_run(0.1, 0.1650, 1.360, 0.330),
    ]
    out = wv.hydrodynamic_mach_independence({"runs": runs})
    assert out["unusable_runs"] and out["unusable_runs"][0]["blowup_step"] == 300
    assert out["reference_mach"] == pytest.approx(0.05)
    assert all(np.isfinite(np.asarray(out["st"])))


# ---------------------------------------------------------------------------
# gates + artifacts
# ---------------------------------------------------------------------------
def _synthetic_record(no_filter=True, st_ma2=0.0):
    machs = [0.01, 0.05, 0.1, 0.2, 0.3]
    poiseuille = _synthetic_channel_sweep(machs, exponent=0.0, amplitude=1e-6)
    poiseuille["fit_error_raw"] = {"exponent_plain": 0.4}
    poiseuille["fit_error_excess"] = {"exponent_excess": 0.4, "r2_excess": 0.1}
    pressure = _synthetic_channel_sweep(machs, exponent=2.0)
    pressure["fit_error_raw"] = {"exponent_plain": 0.4}
    pressure["fit_error_excess"] = {"exponent_excess": 2.0, "r2_excess": 0.99}
    cyl_runs = [
        _cyl_run(0.01, float("nan"), float("nan"), float("nan"), converged=False, blowup=300),
        _cyl_run(0.05, 0.1650, 1.360, 0.330),
        _cyl_run(0.1, 0.1650 * (1 + st_ma2), 1.360, 0.330),
        _cyl_run(0.2, 0.1650 * (1 + 4 * st_ma2), 1.360, 0.330),
    ]
    cylinder = {
        "runs": cyl_runs,
        "machs": [0.05, 0.1, 0.2],
        "st": [0.1650, 0.1650 * (1 + st_ma2), 0.1650 * (1 + 4 * st_ma2)],
        "cd_mean": [1.36, 1.36, 1.36],
        "cl_amplitude": [0.33, 0.33, 0.33],
        "drho_over_rho0": [3.0 * m**2 for m in (0.05, 0.1, 0.2)],
        "st_ref": 0.1650,
        "cd_ref": 1.36,
        "cl_ref": 0.33,
        "rel_st": [0.0, st_ma2, 4 * st_ma2],
        "rel_cd": [0.0, 0.0, 0.0],
        "rel_cl": [0.0, 0.0, 0.0],
        "max_rel_st_low_ma": st_ma2,
        "max_rel_cd_low_ma": 0.0,
        "max_rel_cl_low_ma": 0.0,
    }
    acoustic = None
    if not no_filter:
        acoustic = {
            "implemented": "test filter",
            "preconditioning_available": True,
            "pass_acoustic_band": True,
            "pass_literal_total_drho": True,
            "target_filtered_drho": 1e-5,
            "standard": {"drho_acoustic_amplitude": 1e-3},
            "filtered": {"drho_acoustic_amplitude": 1e-6},
            "suppression_factor": 1000.0,
        }
    record = wc_record(poiseuille, pressure, cylinder, acoustic)
    poisson = wv.pressure_poisson_sweep(pressure)
    independence = wv.hydrodynamic_mach_independence(cylinder)
    independence["c2_literal"] = {
        "status": wv.NOT_MEASURABLE, "mach": 0.01, "st": None, "blowup_step": 300, "steps": 300, "tau": 0.503
    }
    record["pressure_poisson_c6"] = poisson
    record["mach_independence"] = independence
    record["c1_differential"] = {
        "pressure_channel": wv.profile_error_vs_lowest_mach(pressure),
        "body_force_control": wv.profile_error_vs_lowest_mach(poiseuille),
    }
    return record, poisson, independence


def wc_record(poiseuille, pressure, cylinder, acoustic):
    from tensorlbm.weak_compressibility import build_verification_record

    return build_verification_record(
        poiseuille=poiseuille, pressure_channel=pressure, cylinder=cylinder, acoustic_filter=acoustic
    )


def test_gates_contain_every_criterion_and_flag_missing_preconditioner():
    record, poisson, independence = _synthetic_record(no_filter=True)
    gates = wv.build_gates(
        record, poisson=poisson, independence=independence, differential=record["c1_differential"]
    )
    for key in (
        "c1_poiseuille_ma2_scaling",
        "c2_strouhal_mach_independence",
        "c3_density_fluctuation_order",
        "c4_acoustic_filter",
        "c5_hydrodynamic_coefficient_independence",
        "c6_pressure_poisson",
        "c7_applicability_boundary",
        "c8_honest_gate",
    ):
        assert key in gates, key
    assert gates["c1_poiseuille_ma2_scaling"]["status"] == wv.PASS
    assert abs(gates["c1_poiseuille_ma2_scaling"]["exponent"] - 2.0) < 0.05
    assert gates["c4_acoustic_filter"]["status"] == wv.NOT_SUPPORTED
    assert "c2_literal_ma_0p01" in gates["c8_honest_gate"]["requirements"]["unmeasurable_criteria_declared"]


def _valid_instrument():
    """A validated instrument record, so the FAIL branches of C2/C5/C7 are reachable."""
    return {
        "validated": True,
        "verdict": wv.PASS,
        "reference": wv.CYLINDER_REFERENCE,
        "runs": [],
        "failed": [],
        "reason": "test fixture",
    }


def test_gate_fails_when_the_strouhal_number_depends_on_mach():
    record, poisson, independence = _synthetic_record(st_ma2=5e-3)
    gates = wv.build_gates(
        record,
        poisson=poisson,
        independence=independence,
        differential=record["c1_differential"],
        instrument=_valid_instrument(),
    )
    assert gates["c2_strouhal_mach_independence"]["status"] == wv.FAIL
    assert gates["c7_applicability_boundary"]["triggered"] is True


def test_implemented_filter_is_graded_instead_of_not_supported():
    record, poisson, independence = _synthetic_record(no_filter=False)
    gates = wv.build_gates(
        record,
        poisson=poisson,
        independence=independence,
        differential=record["c1_differential"],
        instrument=_valid_instrument(),
    )
    assert gates["c4_acoustic_filter"]["status"] == wv.PASS
    assert gates["c4_acoustic_filter"]["target_filtered_drho"] == 1e-5


def test_write_artifacts_round_trips_and_csv_has_header(tmp_path):
    record, poisson, independence = _synthetic_record()
    gates = wv.build_gates(
        record, poisson=poisson, independence=independence, differential=record["c1_differential"]
    )
    paths = wv.write_artifacts(tmp_path, record, gates, provenance={"git_commit": "test"})
    for key in ("result", "gates", "csv"):
        assert Path(paths[key]).exists()
    res = json.loads(Path(paths["result"]).read_text())
    assert res["provenance"]["git_commit"] == "test"
    lines = Path(paths["csv"]).read_text().strip().splitlines()
    assert lines[0].startswith("case,mach_requested,mach_measured")
    assert len(lines) > 10
    gates_back = json.loads(Path(paths["gates"]).read_text())
    assert gates_back["gates"]["c8_honest_gate"]["status"] == wv.PASS


def test_task_specs_hold_the_viscosity_fixed():
    from tensorlbm.weak_compressibility import CS

    specs = wv.channel_task_specs([0.01, 0.1, 0.3], "pressure_channel")
    for spec in specs:
        nu = spec["mach"] * CS * (spec["ny"] - 1) / spec["re"]
        assert nu == pytest.approx(0.05, rel=1e-12)


# ---------------------------------------------------------------------------
# artifact integration (skipped when the campaign has not been run)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (ARTIFACTS / "gates.json").exists(), reason="campaign artifacts not present")
def test_recorded_gates_are_consistent_with_recorded_numbers():
    """The artifact must be *self-consistent*, whatever the verdicts are.

    A verification artifact whose gate says PASS while its numbers say otherwise
    (or vice versa) is worse than a FAIL: this test re-derives every verdict from
    the recorded numbers instead of asserting that everything passed.
    """
    gates = json.loads((ARTIFACTS / "gates.json").read_text())["gates"]
    result = json.loads((ARTIFACTS / "result.json").read_text())
    assert set(gates) >= {
        "c0_cylinder_instrument_validation",
        "c1_poiseuille_ma2_scaling",
        "c2_strouhal_mach_independence",
        "c3_density_fluctuation_order",
        "c4_acoustic_filter",
        "c5_hydrodynamic_coefficient_independence",
        "c6_pressure_poisson",
        "c7_applicability_boundary",
        "c8_honest_gate",
    }
    for name, gate in gates.items():
        assert gate["status"] in (wv.PASS, wv.FAIL, wv.NOT_SUPPORTED, wv.NOT_MEASURABLE), name

    c1 = gates["c1_poiseuille_ma2_scaling"]
    expected = wv.PASS if abs(c1["exponent"] - 2.0) <= 0.2 else wv.FAIL
    assert c1["status"] == expected, (c1["status"], c1["exponent"])
    # the negative control must be separated from the pressure-driven channel by
    # at least 5x for the C1 diagnostic to mean anything
    assert c1["control_body_force_channel"]["control_separated"] is True

    c6 = gates["c6_pressure_poisson"]
    ok = c6["min_r2"] >= 0.99 and abs(c6["dev_order"] - 2.0) <= 0.4
    assert c6["status"] == (wv.PASS if ok else wv.FAIL)
    assert len(c6["r2_pass"]) == len(c6["r2"])

    c4 = gates["c4_acoustic_filter"]
    if c4["status"] == wv.PASS:
        assert c4["filtered"]["drho_acoustic_amplitude"] <= c4["target_filtered_drho"]
        assert c4["standard"]["drho_acoustic_amplitude"] > c4["target_filtered_drho"]
        assert c4["suppression_factor"] > 1.0

    # an unvalidated instrument must not yield a PASS on the Mach-independence gates
    if gates["c0_cylinder_instrument_validation"]["status"] != wv.PASS:
        for name in (
            "c2_strouhal_mach_independence",
            "c5_hydrodynamic_coefficient_independence",
            "c7_applicability_boundary",
        ):
            assert gates[name]["status"] in (wv.NOT_SUPPORTED, wv.NOT_MEASURABLE)

    assert "provenance" in result and result["provenance"]["git_commit"]
    assert gates["c8_honest_gate"]["status"] == wv.PASS


@pytest.mark.skipif(not (ARTIFACTS / "ma_sweep.csv").exists(), reason="campaign artifacts not present")
def test_cylinder_strouhal_and_drag_are_reported_and_validated():
    """The recorded cylinder numbers must be finite and the validation gate honest.

    The literature band is *not* asserted as a pass condition here: when the
    instrument does not reproduce the reference drag, the correct outcome is a
    validation gate that says so (and C2/C5/C7 ``NOT_SUPPORTED``), not a widened
    band.  This test checks exactly that consistency.
    """
    import csv

    rows = list(csv.DictReader((ARTIFACTS / "ma_sweep.csv").read_text().splitlines()))
    cyl = [r for r in rows if r["case"] == "cylinder_re100"]
    st = [float(r["value"]) for r in cyl if r["quantity"] == "st" and np.isfinite(float(r["value"]))]
    cd = [float(r["value"]) for r in cyl if r["quantity"] == "cd_mean" and np.isfinite(float(r["value"]))]
    assert st and cd, "no converged cylinder coefficients in the artifact"
    assert all(np.isfinite(v) for v in st + cd), (st, cd)
    # the artifact records *invalid* points too (e.g. St=9.86 at Ma=0.01, where no
    # vortex street developed); the validation gate -- not a widened band -- is
    # what has to flag them, so the two must agree
    gates_pre = json.loads((ARTIFACTS / "gates.json").read_text())["gates"]
    if gates_pre["c0_cylinder_instrument_validation"]["status"] != wv.PASS:
        assert any(not (0.14 <= v <= 0.20) for v in st), st
    gates = json.loads((ARTIFACTS / "gates.json").read_text())["gates"]
    instrument = gates["c0_cylinder_instrument_validation"]
    cd_ok = all(abs(v - wv.CYLINDER_REFERENCE["cd"]) / wv.CYLINDER_REFERENCE["cd"] <= 0.15 for v in cd)
    assert instrument["validated"] is bool(cd_ok and all(
        abs(s - wv.CYLINDER_REFERENCE["st"]) <= wv.CYLINDER_REFERENCE["st_tol"] for s in st
    ))