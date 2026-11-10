"""Tests for :mod:`tensorlbm.ice_roughness.forcing_audit` (mission m9872).

The audit answers one question with a machine-checkable verdict: did the TRT
forcing-weight defect ``(1 - s_plus/2)`` contaminate the coefficients the
ice-roughness programme publishes?  The tests pin down

* the *defect signature* (closed form, machine precision) and that the three
  engine variants differ only in the forcing weight (plus, for the authentic
  defect-window build, the Smagorinsky coefficient),
* that the laminar Poiseuille anchor of the real C back-end separates the two
  engines,
* the verdict arithmetic of :func:`forcing_audit.contamination` -- an audit
  that cannot flag a 20 % change has no discriminating power,
* that the quantity pipeline recovers a known ``k0`` from records alone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.ice_roughness import forcing_audit as fa
from tensorlbm.ice_roughness import regime_law as rl
from tensorlbm.ice_roughness import sweep as sw


# --------------------------------------------------------------------------
# engine provenance
# --------------------------------------------------------------------------

def test_variants_differ_only_where_intended():
    head = fa.native_source("fixed")
    assert head.count(fa.BLOCKING_EXPR_FIXED) == 1
    assert fa.BLOCKING_EXPR_DEFECT not in head

    single = fa.native_source("defect_single_factor")
    assert single.count(fa.BLOCKING_EXPR_DEFECT) == 1
    # exactly one token block differs from HEAD
    diff = sum(1 for a, b in zip(single.splitlines(), head.splitlines()) if a != b)
    assert diff == 1
    assert len(single.splitlines()) == len(head.splitlines())

    window = fa.native_source("defect_window")
    assert window.count(fa.BLOCKING_EXPR_DEFECT) == 1
    assert fa.BLOCKING_EXPR_FIXED not in window
    # the authentic defect-window engine also predates the Smagorinsky fix
    assert "54.0 * smag2" not in window

    shas = {v: fa.engine_sha256(v) for v in fa.VARIANTS}
    assert len(set(shas.values())) == 3


# --------------------------------------------------------------------------
# closed-form defect signature
# --------------------------------------------------------------------------

def test_drive_factor_limits_and_monotonicity():
    tau = np.linspace(0.5002, 40.0, 400)
    d = np.array([fa.drive_factor(float(t)) for t in tau])
    assert np.all(np.diff(d) > 0.0), "the defective drive factor is monotone in tau"
    assert fa.drive_factor(0.5002) < 0.02, "tau -> 0.5 collapses the drive to ~0"
    assert 1.8 < fa.drive_factor(40.0) <= 2.0, "viscous limit over-drives by ~2x"
    # campaign Reynolds numbers sit next to the stability limit => the defect is
    # not a constant factor 2 but a tau-dependent rescaling
    assert fa.drive_factor(0.502) < 0.06
    assert fa.drive_factor(0.554) == pytest.approx(0.223, abs=5e-3)


def test_drive_factor_half_is_machine_precise():
    """The seeding specification quotes the signature as ``F/2``: that is *one*
    tau, and the closed form hits it to machine precision."""
    tau = fa.tau_for_drive_factor(0.5)
    assert 0.5 < tau < 1.5
    assert abs(fa.drive_factor(tau) - 0.5) < 1e-12
    with pytest.raises(ValueError):
        fa.drive_factor(0.5)


# --------------------------------------------------------------------------
# engine-level anchor
# --------------------------------------------------------------------------

def _compiler_ready() -> bool:
    from tensorlbm.rough_channel3d_native import native_available

    return native_available()


@pytest.mark.skipif(not _compiler_ready(), reason="no C compiler for the native back-end")
def test_laminar_anchor_separates_fixed_from_defect_engine():
    """``U_b^+ = Re_tau/3`` for the fixed engine; strictly below for the defect.

    The two variants are compiled from the same source with one expression
    swapped, so this is a single-factor controlled probe of the engine, not a
    comparison of two different codes.
    """
    kw = dict(re_tau=8.0, h=8, u_tau=0.02, steps=8000, threads=2)
    fixed = fa.laminar_anchor("fixed", **kw)
    defect = fa.laminar_anchor("defect_single_factor", **kw)

    assert fixed["u_plus_bulk_analytic"] == pytest.approx(8.0 / 3.0)
    assert fixed["ratio_to_analytic"] == pytest.approx(1.0, abs=0.08)
    assert defect["ratio_to_analytic"] < 0.75 * fixed["ratio_to_analytic"]
    # measured drop is of the size the closed form predicts (within the
    # discretisation error of an h=8 channel)
    assert defect["ratio_to_analytic"] == pytest.approx(
        fixed["ratio_to_analytic"] * defect["drive_factor_closed_form"], rel=0.30
    )


# --------------------------------------------------------------------------
# verdict arithmetic
# --------------------------------------------------------------------------

def _synthetic_engine(k0: float, ks_values=(20.0, 45.0, 90.0), seeds=(1, 2)) -> dict:
    """Engine table shaped like ``forcing_audit.quantities`` output."""
    per_seed = []
    for seed in seeds:
        points = {}
        for ks in ks_values:
            d_u = float(rl.delta_u_plus(ks, k0))
            cf_smooth = rl.cf_channel_smooth_reference(200.0)
            u_smooth = math.sqrt(2.0 / cf_smooth)
            points[f"{ks:g}"] = {
                "ks_plus_target": float(ks),
                "cf": cf_smooth * (u_smooth / (u_smooth - d_u)) ** 2,
                "cf_smooth": cf_smooth,
                "dU_plus": d_u,
                "cf_ratio": (u_smooth / (u_smooth - d_u)) ** 2,
                "cf_ratio_lbm": (u_smooth / (u_smooth - d_u)) ** 2,
            }
        per_seed.append({"seed": seed, "k0": float(k0), "points": points})
    downstream = [{"seed": s, "C_w": 6.0e-3, "ice_water_drag_Pa": 0.5,
                   "drift_speed_m_s": 0.1, "drift_angle_deg": 25.0} for s in seeds]
    return {"per_seed": per_seed, "downstream": downstream}


def test_contamination_accepts_one_percent_and_flags_five_percent():
    base = _synthetic_engine(11.8)
    close = _synthetic_engine(11.8 * 1.005)
    away = _synthetic_engine(11.8 * 1.20)

    ok = fa.contamination(base, close)
    assert ok["turbulent_calibration_clean"] is True
    assert ok["contaminated"] is False
    assert ok["quantities"]["k0"]["abs_mean"] < fa.DELTA_ACCEPT

    bad = fa.contamination(base, away)
    assert bad["contaminated"] is True
    assert bad["turbulent_calibration_clean"] is False
    assert "k0" in bad["exceeding_quantities"]
    assert bad["quantities"]["k0"]["abs_mean"] > fa.DELTA_ACCEPT


def test_contamination_needs_both_engines():
    base = _synthetic_engine(11.8)
    empty = {"per_seed": [], "downstream": []}
    verdict = fa.contamination(base, empty)
    # no paired evidence must never read as "clean"
    assert verdict["contaminated"] is True
    assert verdict["turbulent_calibration_clean"] is False
    assert verdict["evaluable_quantities"] == 0


# --------------------------------------------------------------------------
# quantity pipeline on records alone
# --------------------------------------------------------------------------

def _synthetic_records(tmp_path, *, k0: float, variant: str, seed: int = 7):
    """Write the record files a real run would produce (aggregation must not
    need a live engine)."""
    import json

    ks_values = (20.0, 45.0, 90.0)
    written = {}
    for ks in ks_values:
        case = sw.Case(case_id=sw._case_id(ks, 5.0), ks_plus_target=ks,
                       lambda_over_ks=5.0, profile="smoke", seed=seed)
        d_u = float(rl.delta_u_plus(ks, k0))
        cf_smooth = rl.cf_channel_smooth_reference(case.re_tau)
        u_smooth = math.sqrt(2.0 / cf_smooth)
        rec = {
            "case": case.to_json(), "re_tau": case.re_tau, "cf_smooth": cf_smooth,
            "cf": cf_smooth * (u_smooth / (u_smooth - d_u)) ** 2,
            "u_plus_bulk": u_smooth - d_u, "ks_plus": ks,
            "log_fit": {"b": float(rl.B_SMOOTH) - d_u}, "log_window": {"n_points": 5},
        }
        stem = fa.case_record_id(case, variant)
        (tmp_path / f"{stem}.json").write_text(json.dumps(rec))
        written[stem] = rec
    # one smooth baseline per Re_tau
    for ks in ks_values:
        rough = sw.Case(case_id="", ks_plus_target=ks, lambda_over_ks=5.0, profile="smoke", seed=seed)
        rt = round(rough.re_tau, 3)
        case = sw.Case(case_id=f"smooth_re{rt:g}", ks_plus_target=rt, lambda_over_ks=1.0,
                       profile="smoke", kind="smooth", seed=seed)
        cf = rl.cf_channel_smooth_reference(rt)
        rec = {"case": case.to_json(), "re_tau": rt, "cf": cf,
               "u_plus_bulk": math.sqrt(2.0 / cf), "ks_plus": 0.0,
               "log_fit": {"b": float(rl.B_SMOOTH)}, "log_window": {"n_points": 0}}
        stem = fa.case_record_id(case, variant)
        (tmp_path / f"{stem}.json").write_text(json.dumps(rec))
        written[stem] = rec
    return written


def test_quantities_recovers_k0_and_runs_downstream(tmp_path):
    records = _synthetic_records(tmp_path, k0=11.8, variant="fixed")
    loaded = fa.load_records(tmp_path, "fixed")
    assert len(loaded) == len(records)

    table = fa.quantities(loaded, lambda_over_ks=5.0, ks_targets=(20.0, 45.0, 90.0))
    assert len(table["per_seed"]) == 1
    assert table["per_seed"][0]["k0"] == pytest.approx(11.8, rel=0.02)
    row = table["per_seed"][0]["points"]["90"]
    assert row["cf_ratio"] == pytest.approx(row["cf_ratio_lbm"], rel=1e-6)
    down = table["downstream"][0]
    assert down["C_w"] > 0.0
    assert 0.0 < down["drift_speed_m_s"] < 1.0
    assert 0.0 < down["drift_angle_deg"] < 90.0