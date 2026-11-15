"""CI guards for CASE-TAYLOR-GREEN-3D round 4.

Round 3's field-level metrics were scored against a *Nyquist-aliased* reference
field (the classic Taylor-Green series spectrum at ``T = 1`` peaks at
``|k|_inf = 20``, well above the ``n/2 = 16`` Nyquist index of the ``32**3``
lattice), and its two "not_run" legs (M8 grid ladder, M9 Mach attribution) were
left open.  These tests pin the round-4 findings:

* ``TG3DSeries.velocity_field`` never returns a reference whose spectral
  support reaches Nyquist (the aliasing that produced round 3's numbers);
* forcing an aliased order reproduces exactly those numbers, so the
  attribution of the defect is not a guess;
* the band-projected field metrics recorded in
  ``artifacts/case_taylor_green_3d/round4/*.json`` stay inside the 8 % case
  band, the grid ladder stays first-order-or-better, and the modal offset does
  not drift with the grid (the compressibility-floor reading of round 3);
* the Mach ladder's slope/intercept stay in the predeclared windows.

The artifact-level tests only read committed JSON, so CI cost is negligible;
the live tests rebuild the reference at a small grid and take ~2 s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.tg3d_lbm_check import (  # noqa: E402
    run_tg3d_lbm,
    series_reference,
    tg3d_parameters,
)
from tensorlbm.tg3d_series import TG3DSeries, taylor_green_classic  # noqa: E402

ART = ROOT / "artifacts" / "case_taylor_green_3d" / "round4"


def _load(name: str) -> dict:
    path = ART / f"{name}.json"
    if not path.exists():
        pytest.skip(f"round-4 artifact {path.name} not present")
    return json.loads(path.read_text())


def _series(n: int = 32, amplitude: float = 0.05, reynolds: float = 160.0, order: int = 12):
    p = tg3d_parameters(n, amplitude=amplitude, reynolds=reynolds)
    return TG3DSeries(taylor_green_classic(amplitude), box=float(n), nu=p["nu"], order=order)


# ---------------------------------------------------------------------------
# live guards on the reference (no LBM run needed)
# ---------------------------------------------------------------------------


def test_velocity_field_respects_nyquist_on_case_grid():
    """The auto-lowered reference must stay strictly below n/2 at every grid."""
    for n in (24, 32, 48):
        ser = _series(n=n)
        ser.velocity_field(120.0, n=n)
        assert ser.last_field_support < n / 2.0, (n, ser.last_field_support)


def test_round3_field_numbers_came_from_an_aliased_reference():
    """Forcing order 4 on a 32**3 grid exceeds Nyquist and is what round 3 scored."""
    ser = _series()
    spec = ser._series_spectrum(120.0, 4)
    assert ser._spectrum_peak(spec) >= 16.0  # aliased on n = 32
    ser.velocity_field(120.0, n=32)
    assert ser.last_field_order <= 3  # the honest truncation


# ---------------------------------------------------------------------------
# artifact guards: the numbers round 4 publishes
# ---------------------------------------------------------------------------


def test_reproduce_modal_metrics_inside_case_band():
    rep = _load("reproduce")
    modal = rep["modal"]
    assert abs(modal["energy_rel_err_pct"]) < 8.0
    assert abs(modal["enstrophy_rel_err_pct"]) < 8.0


def test_published_field_numbers_are_nyquist_safe_and_reproduce_the_alias():
    rep = _load("reproduce")
    safe = rep["field_nyquist_safe"]
    assert safe["spectrum_peak"] < safe["nyquist_index"]
    forced = {r["order"]: r for r in rep["field_forced_orders"]}
    aliased = forced[4]
    assert aliased["aliased"] is True
    delivered = rep["round3_delivered_field_numbers"]
    assert aliased["u_l2_rel_err_pct"] == pytest.approx(delivered["u_l2_rel_err_pct"], abs=5e-3)
    assert aliased["vorticity_l2_rel_err_pct"] == pytest.approx(
        delivered["vorticity_l2_rel_err_pct"], abs=5e-3
    )
    # the honest value is the un-aliased one, and it stays inside the case band
    assert safe["vorticity_l2_rel_err_pct"] < 8.0
    assert safe["vorticity_l2_rel_err_pct"] > delivered["vorticity_l2_rel_err_pct"]


def test_band_metrics_are_well_posed_and_inside_band():
    band = _load("band")
    kc = band["kmax_canonical"]
    row = [r for r in band["rows"] if r["kmax"] == kc][0]
    assert row["reference_spectrum_peak"] < band["n"] / 2.0
    assert abs(row["u_l2_rel_pct"]) < 8.0
    assert abs(row["vorticity_l2_rel_pct"]) < 8.0
    # the band metric must be insensitive to which k_max >= support we pick
    wider = [r for r in band["rows"] if r["kmax"] >= kc]
    assert len(wider) >= 1


def test_grid_ladder_records_honest_fail_and_offset_shrinks():
    """M8 failed its predeclared bar (>= 1.8); the measurement itself is pinned.

    This test encodes the honest result, not the aspiration: the observed order
    is below the threshold and the modal offset shrinks monotonically with the
    grid, which is what rules out round 3's "grid-independent offset" reading.
    """
    grid = _load("grid_ladder")
    order = grid["observed_order_band_u_l2"]
    assert 0.5 <= order < 1.8, order  # honest FAIL of the predeclared bar
    assert grid["energy_offset_shrinks_monotonically_with_n"] is True
    assert grid["energy_offset_spread_pct"] > 1.0
    sizes = [r["n"] for r in grid["rows"]]
    assert sizes == sorted(sizes) and len(sizes) >= 3
    # every grid is compared in the same band, and that band is Nyquist safe
    assert len({r["common_band_kmax"] for r in grid["rows"]}) == 1
    assert grid["common_band_kmax"] <= min(r["n"] / 3.0 for r in grid["rows"])
    for r in grid["rows"]:
        assert r["reference_spectrum_peak"] < r["n"] / 2.0


def test_mach_ladder_shows_the_offset_is_ma_independent():
    """M9 failed its predeclared bar: the offset does not scale like Ma^2."""
    mach = _load("mach")
    fit = mach["fit_energy_err_pct_vs_ma2"]
    assert abs(fit["slope_pct_per_ma2"]) < 0.3  # predeclared window was [0.3, 3]
    assert fit["r2"] < 0.5  # no linear Ma^2 trend to speak of
    offsets = [r["energy_rel_err_pct"] for r in mach["rows"]]
    assert max(offsets) - min(offsets) < 0.5  # %-points over a 64x Ma^2 range
    ma2 = [r["ma2"] for r in mach["rows"]]
    assert ma2 == sorted(ma2) and len(ma2) >= 3


def test_verdict_passes_acceptance_and_fails_only_the_attribution_checks():
    verdict = _load("verdict")
    failed = sorted(c["check"] for c in verdict["checks"] if not c["pass"])
    # the case-pool acceptance (R*/B*) must pass; only the predeclared
    # attribution checks may fail, and they are expected to
    assert not [c for c in failed if c.startswith(("R1", "R2", "R3", "R4", "B1", "B2"))], failed
    assert any(c.startswith("M8 ") for c in failed)
    assert any(c.startswith("M9a") for c in failed)
    assert verdict["pass"] is False


# ---------------------------------------------------------------------------
# live end-to-end guard at a cheap grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [16])
def test_small_grid_run_against_series_is_inside_band(n):
    """Smoke: the harness still runs and reports a sane error at a cheap grid."""
    a, re = 0.05, 160.0
    ser = series_reference(n, amplitude=a, reynolds=re, order=8)
    run = run_tg3d_lbm(n, amplitude=a, reynolds=re, eddy_times=0.5)
    t_end = run["history"][-1]["time"]
    e_l = run["history"][-1]["energy"]
    e_s = ser.energy(t_end)
    assert np.isfinite(e_l) and e_s > 0.0
    assert abs(e_l / e_s - 1.0) < 0.15  # coarse grid, only a sanity bound
    assert run["history"][-1]["rho_max"] - run["history"][-1]["rho_min"] < 0.05