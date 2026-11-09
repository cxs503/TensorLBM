"""IN65-N04 -- machine gate for the DG convection/diffusion verification frontier.

Every assertion is a *property* that the previous node's verification face could
not state, in the order of the module documentation:

* HLLC is exact on a contact and near-exact on a shock, while Rusanov is not
  (why the Sod runs needed a contact-resolving flux);
* the Zhang-Shu limiter is conservative to round-off and only scales cells that
  actually violate positivity;
* the DG advection operator conserves mass and converges at ``p + 1`` on smooth
  data, while kinked data is capped at the ``H^{3/2}`` rate -- the mechanism
  behind the rotating-cone order gate;
* a shock tube needs transmissive ends: the periodic tube of length 1 is
  contaminated by the wrap-around Riemann problem, and the pressure-based shock
  locator is the correct one (the density-based one finds the contact);
* a limited square wave creates no new extrema and its dissipation/dispersion
  are finite and quantified.

Sizes are deliberately small: the fidelity runs live in
``scripts/run_dg_frontier_in65.py`` (artefacts in ``artifacts/IN65_N04``), the
tests only guard the mechanisms (whole file < 60 s).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.dg_verification.benchmarks import exact_sod  # noqa: E402
from tensorlbm.dg_verification.frontier import (  # noqa: E402
    EulerDG1D_HLLC,
    ScalarDG1D,
    bench_euler_mass_ledger,
    bench_kink_order,
    bench_mms_varcoeff,
    bench_rotating,
    bench_sod_hllc,
    bench_square_wave_metrics,
    contact_position,
    total_variation,
)
from tensorlbm.dg_verification.hllc import (  # noqa: E402
    AdmissibleError,
    hllc_flux,
    rusanov_flux,
    zhang_shu_limit,
)

GAMMA = 1.4


def _state(rho, u, p, gamma=GAMMA):
    return np.array([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u * u]).reshape(3, 1)


def _phys_flux(rho, u, p, gamma=GAMMA):
    return np.array(
        [rho * u, rho * u * u + p, u * (p / (gamma - 1.0) + 0.5 * rho * u * u + p)]
    ).reshape(3, 1)


# ---------------------------------------------------------------------------
# HLLC flux properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wave_speed", ["toro", "davis", "einfeldt"])
def test_rusanov_vs_hllc_contact(wave_speed):
    """Across a pure contact HLLC is exact; Rusanov is not (its mass flux blurs)."""
    UL = _state(1.0, 1.0, 1.0)
    UR = _state(0.125, 1.0, 1.0)
    exact = _phys_flux(1.0, 1.0, 1.0)
    F = hllc_flux(UL, UR, gamma=GAMMA, wave_speed=wave_speed)
    assert np.allclose(F, exact, rtol=1e-12, atol=1e-12), (F.ravel(), exact.ravel())
    Fr = rusanov_flux(UL, UR, gamma=GAMMA)
    # Rusanov's mass flux is off by more than 100 % here -- the documented
    # reason the Sod contact was unusable in the previous node.
    assert abs(Fr[0, 0] - exact[0, 0]) / abs(exact[0, 0]) > 1.0


def test_hllc_shock_rankine_hugoniot():
    """HLLC reproduces the flux of an exact shock (post-shock vs pre-shock state)."""
    UL = _state(0.26557, 0.92745, 0.30313)
    UR = _state(0.125, 0.0, 0.1)
    exact = _phys_flux(0.26557, 0.92745, 0.30313)
    F = hllc_flux(UL, UR, gamma=GAMMA)
    assert np.allclose(F, exact, rtol=8e-3, atol=8e-3)


def test_hllc_consistent_on_equal_states():
    """A numerical flux must reduce to the physical flux for UL == UR."""
    for rho, u, p in ((1.0, 0.3, 1.0), (0.2, -0.7, 0.4)):
        U = _state(rho, u, p)
        F = hllc_flux(U, U, gamma=GAMMA)
        assert np.allclose(F, _phys_flux(rho, u, p), rtol=1e-6, atol=1e-6)


def test_hllc_isothermal_consistency():
    """Isothermal closure: no entropy wave exists, so check consistency only.

    The isothermal Euler system has just three characteristics (u, u +- c) and *no*
    contact, so a density jump at equal velocity is an acoustic Riemann problem,
    not a contact -- the closure this DG solver shares with the D2Q9 lattice.
    """
    cs2 = 1.0 / 3.0
    U = np.array([[1.0], [0.3]])
    assert np.allclose(hllc_flux(U, U, gamma=None, cs2=cs2), np.array([[0.3], [0.3 ** 2 + cs2]]))
    # small density jump at rest: the flux must stay O(jump) away from the mean flux
    UL = np.array([[1.0], [0.0]])
    UR = np.array([[1.01], [0.0]])
    F = hllc_flux(UL, UR, gamma=None, cs2=cs2)
    assert abs(F[0, 0]) < 0.02                       # no spurious mass flux
    assert abs(F[1, 0] - cs2 * 1.005) < 0.01 * cs2   # momentum flux == p to 1 %


# ---------------------------------------------------------------------------
# positivity-preserving limiter
# ---------------------------------------------------------------------------


def test_zhang_shu_conservative_and_admissible():
    rng = np.random.default_rng(7)
    U = np.zeros((3, 6, 3))
    U[0] = 1.0 + 0.05 * rng.normal(size=(6, 3))
    U[1] = 0.02 * rng.normal(size=(6, 3))
    U[2] = U[0] * 2.5
    # make one cell genuinely inadmissible at an interior node: with
    # m = 3 and E = 1 the pressure (gamma-1)(E - m^2/2rho) is negative
    U[1, 2, 1] = 3.0
    U[2, 2, 1] = 1.0
    means = U.mean(axis=-1)
    out, theta = zhang_shu_limit(U, gamma=GAMMA)
    assert np.allclose(out.mean(axis=-1), means, rtol=0, atol=1e-14)  # conservative
    p = (GAMMA - 1.0) * (out[2] - 0.5 * out[1] ** 2 / out[0])
    assert np.all(out[0] > 0.0)
    assert np.all(p > 0.0)
    assert theta[2] < 1.0 and theta[0] == pytest.approx(1.0)


def test_zhang_shu_rejects_inadmissible_average():
    U = np.zeros((3, 2, 3))
    U[0] = -1.0  # negative density *average*: the update itself was unstable
    U[2] = -1.0
    with pytest.raises(AdmissibleError):
        zhang_shu_limit(U, gamma=GAMMA)


# ---------------------------------------------------------------------------
# scalar DG: conservation, smooth order, kink cap
# ---------------------------------------------------------------------------


def test_scalar_dg_mass_conservation():
    sol = ScalarDG1D(degree=2, n_cells=64, velocity=1.0)
    x = sol.x
    sol.set_state(np.exp(-0.5 * ((x - 0.4) / 0.1) ** 2))
    m0 = float(sol.elem.integral(sol.u, sol.dx).sum())
    sol.advance(1.0, cfl=0.3)
    m1 = float(sol.elem.integral(sol.u, sol.dx).sum())
    assert abs(m1 - m0) / abs(m0) < 1e-12


def test_kink_order_is_capped_below_smooth_order():
    """Kinked data (H^{3/2-eps}) cannot reach the design (p+1) rate."""
    kink = bench_kink_order(n_list=(32, 64, 128), degrees=(2,), initial="tent")
    smooth = bench_kink_order(n_list=(32, 64, 128), degrees=(2,), initial="gaussian")
    o_kink = kink["orders"]["P2"]["L2"]["order_fit"]
    o_smooth = smooth["orders"]["P2"]["L2"]["order_fit"]
    assert o_smooth > 2.5, o_smooth            # P2 design order on smooth data
    assert o_kink < 1.75, o_kink               # capped near 1.5 by the kink
    assert o_smooth - o_kink > 1.0


def test_rotating_cone_error_is_kink_dominated():
    """The kink band (2 cells wide) carries most of the squared error."""
    res = bench_rotating(n_list=(16, 32), degrees=(1,), initial="cone")
    rows = res["rows"]
    assert all(r["frac_err_band"] > 0.5 for r in rows), [r["frac_err_band"] for r in rows]
    assert rows[-1]["L2"] < rows[0]["L2"]       # converges
    assert rows[-1]["mass_drift"] < 1e-12


# ---------------------------------------------------------------------------
# manufactured solution (variable coefficient + LDG gradient)
# ---------------------------------------------------------------------------


def test_mms_varcoeff_order():
    res = bench_mms_varcoeff(n_list=(8, 16), degrees=(1, 2), nu=0.05)
    for key, theory in (("P1", 2.0), ("P2", 3.0)):
        got = res["orders"][key]["L2_u"]["order_last_pair"]
        assert abs(got - theory) < 0.5, (key, got)
    assert res["orders"]["P1"]["L2_q"]["order_last_pair"] > 1.5


# ---------------------------------------------------------------------------
# Sod shock tube
# ---------------------------------------------------------------------------


def test_sod_outflow_shock_position():
    """With transmissive ends the Sod gate is met by a P1 HLLC run."""
    res = bench_sod_hllc(n_list=(200,), degrees=(1,), fluxes=("hllc",), t_end=0.2, cfl=0.3)
    row = res["rows"][0]
    assert row["shock_pos_err_pct"] < 2.0, row["shock_pos_err_pct"]
    assert row["shock_strength_err_pct"] < 20.0, row["shock_strength_err_pct"]
    assert row["min_rho"] > 0.0
    assert row["contact_pos_err_cells"] < 4.0, row["contact_pos_err_cells"]


def test_periodic_tube_is_contaminated_reproduction():
    """Regression on the *setup* bug behind IN70's 25 % ideal-gas Sod "FAIL".

    A length-1 periodic tube sends a second Riemann problem from the wrap-around
    into the domain: at t = 0.2 the waves from x = 0 and x = 1 overlap and the
    Sod plateau at x ~ 0.78 is destroyed (pressure ~0.78 instead of 0.303).
    The shock-tube benchmark therefore *must* use transmissive ends.
    """
    n = 200
    sol = EulerDG1D_HLLC(degree=1, n_cells=n, gamma=1.4, flux="hllc",
                         use_limiter=True, bc="periodic")
    xc0 = (np.arange(n) + 0.5) / n
    side = (xc0 < 0.5)[:, None]
    sol.set_state(
        sol.from_primitive(
            np.where(side, 1.0, 0.125) * np.ones_like(sol.x),
            np.zeros_like(sol.x),
            np.where(side, 1.0, 0.1) * np.ones_like(sol.x),
        )
    )
    sol.advance(0.2, cfl=0.3)
    xc = sol.cell_centers()
    rho, u, p = sol.primitive(sol.U)
    p_plateau = float(np.interp(0.78, xc, p.mean(axis=-1)))
    assert p_plateau > 2.0 * 0.30313, p_plateau


def test_sod_exact_wave_positions():
    from tensorlbm.dg_verification.frontier import _sod_exact_waves

    ex = _sod_exact_waves(0.2)
    assert ex["shock_x"] == pytest.approx(0.8504, abs=2e-3)
    assert ex["contact_x"] == pytest.approx(0.6855, abs=2e-3)
    assert ex["rarefaction_head_x"] == pytest.approx(0.2634, abs=2e-3)
    assert ex["p_star"] == pytest.approx(0.30313, abs=1e-4)


def test_euler_conservation_ledger():
    res = bench_euler_mass_ledger(n_cells=60, degrees=(1,), fluxes=("hllc", "rusanov"))
    for row in res["rows"]:
        assert row["mass_drift"] < 1e-12, row
        assert row["energy_drift"] < 1e-12, row


# ---------------------------------------------------------------------------
# square wave: TVD + quantified dissipation/dispersion
# ---------------------------------------------------------------------------


def test_square_wave_tvd_and_dissipation():
    res = bench_square_wave_metrics(n_list=(128,), degrees=(1,), periods=(1,), limiter_states=(True, False))
    lim = [r for r in res["square_rows"] if r["limiter"]][0]
    unl = [r for r in res["square_rows"] if not r["limiter"]][0]
    assert lim["overshoot_pct"] == 0.0 and lim["undershoot_pct"] == 0.0
    assert unl["overshoot_pct"] > 1.0  # unlimited DG oscillates at the jump
    # a limiter can only lose total variation (dissipation), never create it
    assert lim["TV_ratio"] <= 1.0 + 1e-9
    # single-mode: dissipation (amplitude < 1) and a finite, small phase error
    mode = [r for r in res["mode_rows"] if r["limiter"]][0]
    assert 0.9 < mode["amplitude_ratio"] < 1.0001
    assert abs(mode["phase_err_rad"]) < 0.2


def test_total_variation_helper():
    assert total_variation(np.array([0.0, 1.0, 0.0])) == pytest.approx(2.0)


def test_contact_position_on_exact_sod():
    """Locator sanity: on the *exact* Sod profile it must find contact and shock."""
    n = 400
    xc = (np.arange(n) + 0.5) / n
    ref = exact_sod(xc, 0.2)
    assert contact_position(xc, ref["rho"], 0.60, 0.80) == pytest.approx(0.6855, abs=5e-3)
    assert contact_position(xc, ref["p"], 0.60, 1.00) == pytest.approx(0.8504, abs=5e-3)
    assert math.isfinite(contact_position(xc, ref["rho"], 0.0, 1.0))

def test_torus_velocity_inscribed_disk_validity():
    """``rotated IC == exact solution`` only holds inside the inscribed disk.

    The test velocity ``a = (0.5 - y, x - 0.5)`` is a rigid rotation about the box
    centre that is *discontinuous across the box wall* (``a_x`` jumps from 0.5 to
    -0.5 at ``y = 0/1``).  Data carried outside the circle of radius 0.5 is
    advected by the wrong velocity after the periodic wrap, so its L2 error
    against the rotated initial field plateaus no matter how fine the grid is.
    IN70's sigma = 0.1 companion (support to r = 0.55) is such a case; the
    sigma = 0.05 companion (support to r = 0.40) is not.
    """
    wide = bench_rotating(n_list=(16, 32), degrees=(2,), initial="gaussian")
    narrow = bench_rotating(n_list=(16, 32), degrees=(2,), initial="smooth")
    w = [r["L2"] for r in wide["rows"]]
    n = [r["L2"] for r in narrow["rows"]]
    assert n[0] / n[1] > 4.0, (n, "narrow Gaussian must converge")
    assert w[0] / w[1] < 2.0, (w, "wide Gaussian must plateau (benchmark artefact)")
