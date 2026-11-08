"""Literature anchors for :mod:`tensorlbm.roughness_laws`.

``roughness_laws`` is the adjudication basis of the roughness campaign
(``tools/run_roughness_sweep.py``): a measured roughness function ``dU+`` is
turned into a friction increment through it, and the ship-scale allowance is
compared against it.  The module shipped without any test, so these are the
independent anchors that make the published constants falsifiable.

Every expected value below is rebuilt from an **independent statement** of the
classic law (velocity laws, exact algebra, published companion table), never
from the module's own formula.  Companion artefacts (machine numbers, figures)
are produced by ``tools/verify_roughness_laws_literature.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm import roughness_laws as rl

#: Independent constants, re-typed from the literature statements.
KAPPA_LIT = 0.41
B_SMOOTH_LIT = 5.0          # Schlichting: smooth log-law intercept
B_ROUGH_LIT = 8.5           # Nikuradse (1933): fully rough intercept

#: ITTC-57 companion table, 4 significant figures (ITTC 7.5-02-03-01.4).
ITTC57_PUBLISHED = {
    1.0e5: 8.333e-3,
    1.0e6: 4.688e-3,
    1.0e7: 3.000e-3,
    1.0e8: 2.083e-3,
    1.0e9: 1.531e-3,
}


def _delta_u_plus_from_laws(ks_plus: float) -> float:
    """``dU+`` as the difference of the two classical velocity laws at ``y = ks``."""
    u_smooth = (1.0 / KAPPA_LIT) * math.log(ks_plus) + B_SMOOTH_LIT
    u_rough = B_ROUGH_LIT  # y/ks = 1 at the crest
    return u_smooth - u_rough


# --------------------------------------------------------------------------
# A. ITTC-57 correlation line
# --------------------------------------------------------------------------
@pytest.mark.parametrize("re,published", sorted(ITTC57_PUBLISHED.items()))
def test_ittc57_matches_published_companion_table(re: float, published: float) -> None:
    """Constants/exponent of the ITTC-57 line are complete to the tabulated precision."""
    assert float(rl.ittc57_cf(re)) == pytest.approx(published, rel=1.0e-3)


def test_ittc57_positive_and_monotone_decreasing() -> None:
    res = np.array([1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9])
    cf = np.array([float(rl.ittc57_cf(r)) for r in res])
    assert np.all(cf > 0.0)
    assert np.all(np.diff(cf) < 0.0)


def test_ittc57_rejects_out_of_domain_reynolds() -> None:
    with pytest.raises(ValueError):
        rl.ittc57_cf(100.0)
    with pytest.raises(ValueError):
        rl.ittc57_cf(float("nan"))


# --------------------------------------------------------------------------
# B. Nikuradse fully rough asymptote
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ks_plus", [5.0, 10.0, 20.0, 50.0, 90.0, 100.0, 300.0, 1000.0])
def test_nikuradse_equals_independent_law_difference(ks_plus: float) -> None:
    """``dU+`` must equal ``U+_smooth(ks+) - U+_rough`` from the classic laws."""
    got = float(rl.nikuradse_delta_u_plus(ks_plus))
    assert got == pytest.approx(_delta_u_plus_from_laws(ks_plus), abs=1.0e-12)


def test_nikuradse_intercept_is_smooth_minus_rough_intercept() -> None:
    """``NIKURADSE_C`` is not free: it must equal ``B_smooth - 8.5``."""
    assert rl.NIKURADSE_C == pytest.approx(rl.B_SMOOTH - B_ROUGH_LIT, abs=1.0e-12)
    assert rl.KAPPA == pytest.approx(KAPPA_LIT, abs=1.0e-12)


def test_nikuradse_clip_threshold_is_exp_of_kappa_times_intercept() -> None:
    """The asymptote is clipped at ``exp(-C*kappa) = 4.1996``, *not* at 2.25.

    ``regime()`` calls ``ks+ < 2.25`` smooth, but the clipped asymptote only
    becomes positive above ``exp(3.5*0.41) = 4.1996``; between the two the
    function returns exactly zero while the regime label already reads
    "transitional".  Pinned here so the documented behaviour cannot drift.
    """
    clip = rl.NIKURADSE_CLIP_KS_PLUS
    assert clip == pytest.approx(math.exp(-rl.NIKURADSE_C * rl.KAPPA), rel=1.0e-15)
    assert clip == pytest.approx(4.199645008787923, rel=1.0e-12)
    assert float(rl.nikuradse_delta_u_plus(clip * 0.999)) == 0.0
    assert float(rl.nikuradse_delta_u_plus(clip * 1.001)) > 0.0
    assert rl.regime(clip * 0.999) == "transitional"
    assert rl.SMOOTH_REGIME_KS_PLUS < clip


@pytest.mark.parametrize("ks_plus", [5.0, 50.0, 90.0, 1000.0])
def test_cebeci_matches_published_transition_closure(ks_plus: float) -> None:
    assert float(rl.cebeci_delta_u_plus(ks_plus)) == pytest.approx(
        (1.0 / KAPPA_LIT) * math.log(1.0 + 0.3 * ks_plus), rel=1.0e-12
    )


def test_cebeci_fully_rough_intercept_offset() -> None:
    """Fully rough limit: Nikuradse minus Cebeci = ``-ln(0.3)/kappa - 3.5``."""
    pred = -math.log(0.3) / KAPPA_LIT - 3.5
    num = float(rl.nikuradse_delta_u_plus(1.0e8)) - float(rl.cebeci_delta_u_plus(1.0e8))
    assert num == pytest.approx(pred, abs=1.0e-6)
    assert num == pytest.approx(-0.563481, abs=1.0e-5)


def test_literature_band_brackets_both_closures() -> None:
    ks = np.array([5.0, 50.0, 200.0, 1000.0])
    band = rl.literature_band(ks, tol_u=0.5)
    ceb = rl.cebeci_delta_u_plus(ks)
    nik = rl.nikuradse_delta_u_plus(ks)
    assert np.all(band["lo"] <= np.minimum(ceb, nik) + 1.0e-12)
    assert np.all(band["hi"] >= np.maximum(ceb, nik) - 1.0e-12)
    # width is exactly the closure spread widened by 2*tol (no hidden fudge)
    assert np.allclose(band["hi"] - band["lo"], np.abs(ceb - nik) + 1.0, rtol=1e-12)


def test_equivalent_ks_plus_round_trips() -> None:
    for d_u_plus in (1.0, 3.0, 5.0, 8.0):
        ks = rl.equivalent_ks_plus(d_u_plus)
        assert float(rl.nikuradse_delta_u_plus(ks)) == pytest.approx(d_u_plus, abs=1.0e-12)


# --------------------------------------------------------------------------
# C. ITTC-78 roughness allowance
# --------------------------------------------------------------------------
def test_ittc78_zero_crossing_is_exact_algebra() -> None:
    """``105 (ks/L)^(1/3) - 0.64 = 0`` => ``ks/L = (0.64/105)^3 = 2.2645e-7``."""
    zero = (0.64 / 105.0) ** 3
    assert zero == pytest.approx(2.2644984342943528e-07, rel=1.0e-12)
    assert float(rl.ittc78_allowance(zero)) == pytest.approx(0.0, abs=1.0e-18)
    assert float(rl.ittc78_allowance(zero * 0.5)) == 0.0
    # the raw polynomial is strictly positive at the value the docstring used
    raw = (105.0 * ((5.3e-7) ** (1.0 / 3.0)) - 0.64) * 1.0e-3
    assert raw > 0.0
    assert raw == pytest.approx(2.0959e-4, rel=1.0e-3)
    assert float(rl.ittc78_allowance(5.3e-7)) == pytest.approx(raw, rel=1.0e-12)


def test_ittc78_worked_value_200m_hull() -> None:
    """Hand arithmetic: L=200 m, ks=150 um -> ks/L=7.5e-7 -> dCf=3.140e-4."""
    ks_over_l = 150.0e-6 / 200.0
    assert ks_over_l == pytest.approx(7.5e-7, rel=1.0e-15)
    # (7.5e-7)^(1/3) = 0.908560e-2 -> 105 * that - 0.64 = 0.313988
    hand = (105.0 * 0.9085603e-2 - 0.64) * 1.0e-3
    assert float(rl.ittc78_allowance(ks_over_l)) == pytest.approx(hand, rel=1.0e-6)
    assert float(rl.ittc78_allowance(ks_over_l)) == pytest.approx(3.139883e-4, rel=1.0e-6)


def test_ittc78_monotone_and_domain_checks() -> None:
    xs = np.logspace(-8, -3, 200)
    vals = rl.ittc78_allowance(xs)
    assert np.all(np.diff(vals) >= 0.0)
    assert np.all(vals >= 0.0)  # never a drag reduction
    with pytest.raises(ValueError):
        rl.ittc78_allowance(-1.0e-7)


# --------------------------------------------------------------------------
# D. Momentum convention and the dU+ -> dCf conversion
# --------------------------------------------------------------------------
@pytest.mark.parametrize("u_plus", [12.0, 18.0, 25.0, 36.0])
def test_momentum_cf_equals_symbol_derivation(u_plus: float) -> None:
    """``Cf = tau_w/(1/2 rho U^2)`` with ``tau_w = rho u_tau^2`` gives ``2/U+^2``."""
    assert float(rl.cf_from_u_plus(u_plus)) == pytest.approx(2.0 * (1.0 / u_plus) ** 2, rel=1.0e-15)


@pytest.mark.parametrize("u_plus", [18.0, 25.0, 36.0])
def test_delta_cf_matches_definition(u_plus: float) -> None:
    """``dCf = Cf(U+ - dU+) - Cf(U+)`` and it is positive for ``dU+ > 0``."""
    d_u_plus = 3.5
    got = float(rl.delta_cf_from_delta_u_plus(u_plus, d_u_plus))
    want = 2.0 / (u_plus - d_u_plus) ** 2 - 2.0 / u_plus**2
    assert got == pytest.approx(want, rel=1.0e-14)
    assert got > 0.0


def test_u_plus_and_cf_round_trip() -> None:
    for cf in (2.0e-3, 1.5e-3, 4.7e-3):
        u_plus = math.sqrt(2.0 / cf)
        assert float(rl.cf_from_u_plus(u_plus)) == pytest.approx(cf, rel=1.0e-14)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        rl.cf_from_u_plus(0.0)
    with pytest.raises(ValueError):
        rl.u_plus_from_velocity(1.0, 0.0)
    with pytest.raises(ValueError):
        rl.nikuradse_delta_u_plus(-1.0)
    with pytest.raises(ValueError):
        rl.cebeci_delta_u_plus(-0.5)
    with pytest.raises(ValueError):
        rl.delta_cf_from_delta_u_plus(5.0, 5.0)  # shift would make U+ <= 0
    with pytest.raises(ValueError):
        rl.regime(-0.1)


def test_regime_boundaries() -> None:
    assert rl.regime(2.24) == "smooth"
    assert rl.regime(2.25) == "transitional"
    assert rl.regime(90.0) == "transitional"
    assert rl.regime(90.01) == "fully_rough"


# --------------------------------------------------------------------------
# E. log-law fit (used by rough_channel3d to measure dU+ from a profile)
# --------------------------------------------------------------------------
def test_fit_log_law_recovers_generating_constants() -> None:
    y_plus = np.logspace(1.3, 2.6, 25)
    kappa_true, b_true = 0.40, 5.10
    u_plus = (1.0 / kappa_true) * np.log(y_plus) + b_true
    fit = rl.fit_log_law(y_plus, u_plus, fit_kappa=True)
    assert fit["kappa"] == pytest.approx(kappa_true, rel=1.0e-9)
    assert fit["b"] == pytest.approx(b_true, abs=1.0e-9)
    assert fit["rms"] < 1.0e-12


def test_fit_log_law_imposed_kappa_returns_roughness_intercept() -> None:
    y_plus = np.logspace(1.0, 2.6, 40)
    b_true = 5.0 - 3.0  # a wall that is shifted down by dU+ = 3
    u_plus = (1.0 / rl.KAPPA) * np.log(y_plus) + b_true
    fit = rl.fit_log_law(y_plus, u_plus, fit_kappa=False)
    assert fit["kappa"] == pytest.approx(rl.KAPPA, rel=1.0e-15)
    assert fit["b"] == pytest.approx(b_true, abs=1.0e-9)
    assert rl.B_SMOOTH - fit["b"] == pytest.approx(3.0, abs=1.0e-9)


def test_fit_log_law_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        rl.fit_log_law([10.0], [12.0])
    with pytest.raises(ValueError):
        rl.fit_log_law([10.0, 20.0], [12.0])
    with pytest.raises(ValueError):
        rl.fit_log_law([-1.0, 0.0], [12.0, 13.0])

# --------------------------------------------------------------------------
# F. campaign measurement chain (synthetic round trip, no solver needed)
# --------------------------------------------------------------------------
def test_measurement_chain_round_trip() -> None:
    """profile -> fit_log_law -> dU+ -> ks_eff -> dCf round-trips exactly.

    This is the exact path ``tools/run_roughness_sweep.py`` takes after a run
    (``b`` from ``fit_log_law(fit_kappa=False)``, ``dU+ = B_SMOOTH - b``,
    ``ks_eff = equivalent_ks_plus(dU+)``, ``dCf = delta_cf_from_delta_u_plus``),
    exercised here on a synthetic profile so the chain is tested independently
    of the solver.
    """
    ks_plus = 120.0
    d_u_plus = float(rl.nikuradse_delta_u_plus(ks_plus))
    y_plus = np.logspace(1.0, 2.4, 30)
    u_smooth = (1.0 / rl.KAPPA) * np.log(y_plus) + rl.B_SMOOTH
    fit = rl.fit_log_law(y_plus, u_smooth - d_u_plus, fit_kappa=False)

    recovered = rl.B_SMOOTH - fit["b"]
    assert recovered == pytest.approx(d_u_plus, abs=1.0e-9)
    assert rl.equivalent_ks_plus(recovered) == pytest.approx(ks_plus, rel=1.0e-9)

    u_plus = math.sqrt(2.0 / float(rl.ittc57_cf(1.0e9)))
    d_cf = float(rl.delta_cf_from_delta_u_plus(u_plus, recovered))
    assert d_cf > 0.0
    assert d_cf == pytest.approx(2.0 / (u_plus - recovered) ** 2 - 2.0 / u_plus**2, rel=1.0e-13)
