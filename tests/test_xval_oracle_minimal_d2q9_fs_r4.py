"""R4 tests for anchor XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS.

Fast (<5 s, no solver): they pin the three claims R4 makes about the analysis
code itself --

1. the ladder at fixed lattice tau is a Reynolds sweep (Re ~ a**1.5) and
   ``tau_for_re`` inverts it exactly (the fixed-Re trio is well posed);
2. a run whose series has no point inside the declared window must not crash
   the analysis (the cj3126 regression: ``NoneType.__format__`` at
   resolution_study.py:708 killed a finished 8-run job);
3. the domain-width control is a separate family: it carries a width suffix in
   its label and is excluded from the ladder rungs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
XV = HERE.parent / "benchmarks" / "xval_oracle_minimal_d2q9_fs" / "reference_study"
sys.path.insert(0, str(XV))
sys.path.insert(0, str(HERE.parent / "src"))

G = 5e-5


def test_reynolds_scales_as_a_to_the_three_halves():
    import resolution_study as R

    re16 = R.reynolds(16, 1.0, G)
    re64 = R.reynolds(64, 1.0, G)
    # h0 = 2a and U0 = sqrt(g h0) => Re ~ a**1.5
    assert re64 / re16 == pytest.approx(8.0, rel=1e-12)
    assert re16 == pytest.approx(7.68, rel=1e-6)
    assert R.reynolds(32, 1.0, G) == pytest.approx(21.72232031805074, rel=1e-12)


def test_tau_for_re_roundtrip_and_fixed_re_trio_is_well_posed():
    import resolution_study as R

    re0 = R.re_reference(G)
    assert re0 == pytest.approx(R.reynolds(R.RE_REF_A, 1.0, G), rel=1e-15)
    for a in (16, 24, 48, 64):
        tau = R.tau_for_re(a, re0, G)
        assert R.reynolds(a, tau, G) == pytest.approx(re0, rel=1e-12)
        assert tau > 0.5  # a stable BGK relaxation time


def test_campaign_re_contains_fixed_re_trio_tau_sweep_and_width_control():
    import resolution_study as R

    tasks = R.plan("re")
    arms = [t for t in tasks if t["kind"] == "arm"]
    re0 = R.re_reference()
    # the campaign stores tau rounded to 1e-6, so match Re to 1e-3 relative
    trio_a = sorted(t["a"] for t in arms
                    if abs(R.reynolds(t["a"], t["tau"]) - re0) <= 1e-3 * re0)
    assert trio_a == [24, 48]  # a = 32 / tau = 1 is already in the R2 ladder
    sweep = sorted(t["tau"] for t in arms if t["a"] == 32 and t["tau"] != 1.0)
    assert sweep == [0.6, 0.8, 1.2, 1.4]
    wide = [t for t in arms if t.get("width_in_a") == 8.0]
    assert len(wide) == 1 and wide[0]["a"] == 64


def test_width_suffix_in_label_and_ladder_exclusion():
    import re_confound as RC
    import resolution_study as R

    assert R.result_label({"arm": "tensorlbm_koerner", "a": 64, "tau": 1.0,
                           "width_in_a": 8.0}) == "tensorlbm_koerner_a64_tau1_w8"
    assert R.result_label({"arm": "tensorlbm_koerner", "a": 32, "tau": 1.0}) == \
        "tensorlbm_koerner_a32_tau1"
    wide = {"arm": "tensorlbm_koerner", "a": 64, "tau": 1.0, "width_in_a": 8.0,
            "status": "ok", "nx": 512, "u_max": 0.25,
            "series": [{"T": 1.0, "Z_cell": 2.0, "Z_sub0.05": 2.0},
                       {"T": 2.0, "Z_cell": 3.0, "Z_sub0.05": 3.0}]}
    an = RC.analyze([wide], [])
    assert an["ladder_tau1_w5"] == []
    assert [m["a"] for m in an["width_control"]] == [64]


def test_metrics_tolerate_run_without_in_window_series():
    """Regression for cj3126: a run with no in-window point must not raise."""
    import re_confound as RC

    empty = {"arm": "tensorlbm_koerner", "a": 48, "tau": 1.0, "status": "ok",
             "nx": 240, "u_max": 0.2,
             "series": [{"T": 0.1, "Z_cell": 1.0, "Z_sub0.05": 1.0}]}
    m = RC.metrics(empty)
    assert "max_abs_err_pct" not in m  # nothing in [0.5, 4.0]
    assert m["points"] == []
    an = RC.analyze([empty], [])
    assert an["ladder_tau1_w5"] == [] or all(
        "max_abs_err_pct" not in x for x in an["ladder_tau1_w5"])
    RC.render_md(an)  # must not raise on an evaluated-but-empty run


def test_sign_convention_reports_lagging_front_as_negative():
    """A rung that lags the experiment must come out with a negative signed error."""
    import re_confound as RC
    import resolution_study as R

    lo, hi = R.DESIGN["window_T"]
    series = []
    T = lo
    while T <= hi + 1e-9:
        zref = R.mm_interp(T)
        if zref is not None:
            series.append({"T": T, "Z_cell": 0.7 * zref, "Z_sub0.05": 0.7 * zref})
        T += 0.25
    m = RC.metrics({"arm": "tensorlbm_koerner", "a": 16, "tau": 1.0, "status": "ok",
                    "nx": 80, "u_max": 0.2, "series": series})
    assert m["points"]
    assert m["signed_err_at_max_pct"] == pytest.approx(-30.0, abs=0.5)
    assert m["pass_15pct"] is False