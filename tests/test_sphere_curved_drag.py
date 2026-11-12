"""Invariant tests for the curved-boundary sphere drag ladder (m16085).

These assert *properties*, not merely that code runs:

* the Hasimoto array factor is the documented series (and reduces to the
  familiar ``1.7601 phi^(1/3)`` leading term);
* ``compute_q_sphere`` returns the exact analytic fractional distance from the
  boundary node to the sphere surface (this is the geometric content of the
  curved wall — if it is wrong, every BFL force is wrong);
* the Guo source of the driver injects exactly ``(1 - 1/(2 tau)) g`` momentum
  per unit volume per step -- the ``9 (c.u)(c.F)`` term has *zero* first moment
  by lattice symmetry (``sum_i w_i c_ix^3 = 0``), so the momentum-balance drag
  ``F = (1 - 1/(2 tau)) g V`` carries no velocity-dependent bias.  This was
  checked numerically rather than assumed (a ``(1 + 3 u_x)`` correction was
  hypothesised first and falsified here);
* the reset rest state used for the solid interior carries zero momentum;
* the driver's feed-forward controller is scale invariant;
* a short run of the production driver returns finite, complete records.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "sphere_curved_drag"))

from tensorlbm.interpolated_bc import compute_q_sphere  # noqa: E402

import driver  # noqa: E402


def test_hasimoto_matches_volume_fraction_leading_term():
    for lam in (1 / 6, 1 / 8, 1 / 10, 1 / 20):
        phi = 4.0 * math.pi * lam**3 / 3.0
        leading = 1.0 / (1.0 - 1.7601 * phi ** (1.0 / 3.0))
        # the two series agree to O(phi), i.e. within a few % for lam <= 1/6
        assert abs(driver.hasimoto_K(lam) - leading) / leading < 0.06


def test_hasimoto_monotone_and_above_one():
    prev = 1.0
    for lam in (0.02, 0.05, 0.08, 0.1, 0.12, 1 / 6):
        k = driver.hasimoto_K(lam)
        assert k > prev
        prev = k
    with pytest.raises(ValueError):
        driver.hasimoto_K(0.4)


def test_q_sphere_reconstructs_the_analytic_sphere():
    """The BFL wall reconstructed from (node, q, direction) must land on |x-x_c| = R.

    This is the geometric content of the curved boundary: for every active
    boundary link, ``x_node + q * c_d`` must be a point of the analytic sphere.
    If q were measured from the wrong side, or not normalised by |c_d|, the
    reconstructed radius would be off by up to a lattice unit.
    """
    L, R, c = 24, 6.0, 12.0
    mask, q = compute_q_sphere(L, L, L, c, c, c, R, torch.device("cpu"))
    cvec = driver.C.float()
    checked = 0
    for d in range(19):
        cx, cy, cz = (float(v) for v in cvec[d])
        if cx == 0 and cy == 0 and cz == 0:
            continue
        for z, y, x in mask[d].nonzero()[:40].tolist():
            s = float(q[d, z, y, x])
            assert 0.0 < s <= 1.0 + 1e-9, (d, x, y, z, s)
            px, py, pz = x - c + s * cx, y - c + s * cy, z - c + s * cz
            r = math.sqrt(px * px + py * py + pz * pz)
            assert abs(r - R) < 1e-4, (d, x, y, z, s, r)
            checked += 1
    assert checked > 100


def test_guo_source_momentum_injection_factor():
    """Sum of the source's first moment = (1 - 1/(2 tau)) g V exactly.

    The ``9 (c.u)(c.F)`` part contributes nothing: its first moment is
    ``9 g u0 sum_i w_i c_ix^3 = 0``.  Asserting this keeps the momentum-balance
    drag estimator honest against future edits of the source term.
    """
    L, tau, g, u0 = 8, 0.68, 1.0e-4, 0.05
    dev = torch.device("cpu")
    cvec = driver.C.to(dev).float().view(19, 3, 1, 1, 1)
    wvec = driver.W.to(dev).float().view(19, 1, 1, 1)
    rho = torch.ones((L, L, L))
    ux = torch.full((L, L, L), u0)
    cu = cvec[:, 0] * ux
    f = wvec * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * ux * ux)
    f_new, *_ = driver._fused_bgk_guo(f, tau, g, cvec, wvec)
    d_mom = float(((f_new - f) * cvec[:, 0]).sum())
    expected = (1.0 - 0.5 / tau) * g * L**3
    assert abs(d_mom - expected) / abs(expected) < 1e-3, (d_mom, expected)


def test_rest_state_has_no_momentum():
    mom = torch.einsum("i,ij->j", driver.W.float(), driver.C.float())
    assert float(mom.abs().max()) < 1e-6


def test_controller_is_scale_invariant_and_clamped():
    g = driver._controller_force(1e-4, 1e-4, 0.05, 0.05)
    assert abs(g - 1e-4) < 1e-12
    # doubling the measured velocity halves the force only through the sqrt
    g2 = driver._controller_force(1e-4, 1e-4, 0.05, 0.10)
    assert abs(g2 / 1e-4 - math.sqrt(0.5)) < 1e-9
    # clamp keeps the controller from running away
    assert driver._controller_force(1e-4, 1e-4, 5.0, 1e-6) <= 2.0 * 1e-4
    # no feed-forward information -> hold the previous force
    assert driver._controller_force(7e-5, 0.0, 0.05, 0.05) == 7e-5


def test_short_run_produces_complete_record():
    rec = driver.run_case(
        {"tag": "unit_short", "L": 12, "R": 3.0, "Re": 10.0, "steps": 25,
         "sample_every": 5, "threads": 1, "tail": 2}
    )
    for key in ("F_momentum_balance", "cd_mb_fluid", "bfl_rel_diff_pct", "steady", "hasimoto_K"):
        assert key in rec
    assert math.isfinite(rec["F_momentum_balance"])
    assert math.isfinite(rec["cd_mb_fluid"])
    assert rec["n_bfl_links"] > 0
    assert rec["phi_solid"] == pytest.approx(rec["n_solid_cells"] / 12**3)
    assert rec["steady"] is False  # 25 steps cannot pass the drift gate

# ---------------------------------------------------------------------------
# GAP-LBM-SPHERE v2: the m16085 drag deficit was a *kinetic* (large-tau)
# artifact, not a curved-wall defect.  These tests pin the two facts that
# decision rests on, so neither can regress silently.
# ---------------------------------------------------------------------------
def test_re_measured_uses_dynamic_viscosity_not_nu_over_3():
    """``Re = rho u D / mu`` with ``rho = 1`` ⇒ ``mu = nu``.

    m16085 divided by ``nu/3`` and reported ``Re_measured`` three times too
    large (0.2944 instead of 0.0981 on the Re=0.1 case).
    """
    rec = driver.run_case(
        {"tag": "unit_re", "L": 12, "R": 3.0, "Re": 10.0, "steps": 12,
         "sample_every": 5, "threads": 1, "tail": 2}
    )
    re_expected = rec["u_measured"] * rec["D"] / rec["nu"]
    assert rec["Re_measured"] == pytest.approx(re_expected, rel=1e-12)
    assert rec["Re_superficial"] < rec["Re_measured"]  # superficial < interstitial


def test_mean_free_path_gate_flags_the_large_tau_protocol():
    """``c_s tau / R`` must be recorded and must flag tau = 18.5 as non-hydrodynamic.

    At tau = 18.5 the kinetic mean free path (``tau/sqrt(3)`` ~ 10.7 cells) is
    larger than the sphere radius (6 cells): the BGK viscosity the flow sees is
    far below ``(tau - 1/2)/3``, so a Stokes comparison is not interpretable.
    """
    rec = driver.run_case(
        {"tag": "unit_mfp_bad", "L": 16, "R": 5.0, "Re": 0.1, "u": 0.05,
         "steps": 6, "sample_every": 1, "threads": 1, "tail": 2}
    )
    assert rec["tau"] == pytest.approx(0.5 + 3.0 * rec["nu"])
    assert rec["tau"] > 10.0                      # the m16085 regime
    assert rec["mfp_over_R"] == pytest.approx(
        (rec["tau"] / math.sqrt(3.0)) / rec["R"]
    )
    assert rec["mfp_over_R"] > 1.0
    assert rec["hydrodynamic_regime"] is False

    rec2 = driver.run_case(
        {"tag": "unit_mfp_ok", "L": 16, "R": 2.0, "Re": 0.1, "u": 0.0025,
         "steps": 6, "sample_every": 1, "threads": 1, "tail": 2}
    )
    assert rec2["mfp_over_R"] < 0.5
    assert rec2["hydrodynamic_regime"] is True


def test_effective_viscosity_is_hydrodynamic_at_small_tau_and_collapses_at_large_tau():
    """Steady single-mode forcing: ``nu_eff(k)`` vs ``(tau-1/2)/3``.

    This is the instrument behind the root-cause claim.  At ``tau = 1.4`` the
    relaxation viscosity at ``k = 2*pi/24`` is the hydrodynamic value (within a
    few %); at ``tau = 18.5`` it collapses by more than a factor of three, which
    is the origin of the m16085 drag deficit.
    """
    sys.path.insert(0, str(ROOT / "benchmarks" / "sphere_curved_drag"))
    import diag_visc_k  # noqa: PLC0415

    good = diag_visc_k.nu_eff_mode(1.4, 24)
    assert abs(good["rel_err_pct"]) < 5.0

    bad = diag_visc_k.nu_eff_mode(18.5, 24)
    assert bad["nu_eff"] < 0.5 * bad["nu_theory"]
