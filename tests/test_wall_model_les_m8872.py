"""Regression pins for mission m8872 (wall-model LES maturation pass).

These tests pin the *defects* found and fixed in this pass, so they cannot come
back silently:

1. the DNS loader hard-coded ``Re_tau = 180``; the shipped tables stop at the
   last cell centre (``y+ = 178.12``), so the inferred value is *not* ``h+`` and
   a registry entry has to win;
2. the shipped ``Re_tau = 395`` table is not a channel profile (non-monotone
   ``u+``, centreline ``u+ = 15.96`` against a log-law estimate of 19.78, stored
   correlation reaching 6.0) and must be rejected by the integrity gate;
3. the a-priori stress-peak block reported the *truncation boundary* of a profile
   capped at the last DNS sample; the closure must be solved over the whole
   channel half-height for the peak to be an interior maximum;
4. an eddy-viscosity closure cannot reproduce the Reynolds-number-invariant
   buffer-layer stress peak -- the drift is the signature, and it is pinned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les.budget import (  # noqa: E402
    DNS_REFERENCES,
    load_dns_channel_profile,
    peak_location,
)
from tensorlbm.wall_model_les.closure_channel import (  # noqa: E402
    log_layer_fit,
    solve_channel_closure,
)
from tensorlbm.wall_model_les.reference_gate import (  # noqa: E402
    validate_channel_dns_table,
)

MKM180 = DNS_REFERENCES["mkm180"]["path"]
MKM395 = DNS_REFERENCES["mkm395"]["path"]


def test_dns_loader_takes_re_tau_from_the_registry_not_the_table_end():
    profile = load_dns_channel_profile(MKM180)
    assert profile.re_tau == pytest.approx(180.0)
    # the table stops at the last cell centre; using that value as h+ mis-scales
    # every derived quantity by 1 %
    assert profile.y_plus.max() == pytest.approx(178.12, abs=0.01)
    assert profile.uv_peak_y_plus == pytest.approx(31.98, abs=0.5)


def test_reference_gate_accepts_the_real_channel_and_rejects_the_bad_table():
    good = validate_channel_dns_table(MKM180, re_tau=180.0)
    assert good.passed, good.reasons

    bad = validate_channel_dns_table(MKM395, re_tau=395.0)
    assert not bad.passed
    text = " ".join(bad.reasons)
    assert "not monotone" in text
    assert "centreline" in text
    assert "wall stress" in text
    assert bad.metrics["uv_plus_max"] > 1.0


def test_closure_channel_satisfies_the_exact_stress_balance():
    solution = solve_channel_closure(180.0, n_layers=2000)
    assert solution.u_plus[0] == 0.0
    assert solution.stress_identity_max_abs_error < 1.0e-10
    assert np.all(np.diff(solution.u_plus) > 0.0)


def test_closure_stress_peak_is_an_interior_maximum():
    solution = solve_channel_closure(180.0, n_layers=4000)
    peak = solution.uv_peak_y_plus
    assert 15.0 < peak < 60.0
    assert peak < float(solution.y_plus.max())
    value = float(np.max(solution.uv_plus[solution.y_plus >= 1.0]))
    assert 0.6 < value <= 1.0


def test_closure_recovers_its_own_log_layer_asymptote_at_high_re():
    solution = solve_channel_closure(10000.0, n_layers=4000)
    fit = log_layer_fit(solution.y_plus, solution.u_plus, band=(30.0, 300.0), h_plus=10000.0,
                        wall_fraction=1.0)
    # the mixing length is built from kappa = 0.41, so the fitted slope has to
    # come back close to it (the van Driest damping leaves a few percent)
    assert fit["kappa"] == pytest.approx(0.41, rel=0.10)
    assert fit["max_rel_error"] < 0.05


def test_stress_peak_drifts_with_reynolds_number_under_an_eddy_viscosity_closure():
    """The structural finding: the modelled peak abscissa is not Re-invariant."""

    low = solve_channel_closure(180.0, n_layers=2000).uv_peak_y_plus
    high = solve_channel_closure(1000.0, n_layers=2000).uv_peak_y_plus
    assert high > 1.10 * low


def test_peak_location_uses_a_parabolic_vertex_fit():
    grid = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
    values = np.array([0.1, 0.3, 0.5, 0.45, 0.2])
    assert peak_location(grid, values) == pytest.approx(10.2, abs=0.6)


def test_kvlcc2_offline_chain_does_not_need_the_ship_api():
    sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "wall_model_les"))
    kvlcc2 = pytest.importorskip("kvlcc2")

    from tensorlbm.wall_model_les.resistance import kvlcc2_geometry

    geometry = kvlcc2_geometry(scale=kvlcc2._model_length_for_re(  # noqa: SLF001
        kvlcc2.RE_TARGET, kvlcc2.FROUDE, kvlcc2.NU_SEA
    ) / 320.0)
    speed = kvlcc2.FROUDE * (9.80665 * (geometry["Lpp"])) ** 0.5
    chain = kvlcc2._offline_chain(geometry=geometry, speed=speed)  # noqa: SLF001
    criterion = kvlcc2._criterion_d(chain)  # noqa: SLF001

    assert criterion["ct_rel_error"] < criterion["threshold"]
    assert criterion["verdict"] == "PASS"
    assert criterion["experimental_limb"] == "WITHHELD"
    # the wall-law plate model is slightly more conservative than ITTC-1957 at
    # Re = 1.3e7; a sign flip would mean the wake term is broken again
    assert chain["cf_wall_model"] > chain["cf_ittc_1957"]