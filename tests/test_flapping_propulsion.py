"""Tests for the flapping-foil propulsion verification face (P162W1-N07).

Fast unit tests: the analytic Theodorsen/Garrick anchors, the GCI helper and a
tiny LBM smoke test (chord 8, a few cycles) that checks sign conventions,
bit-wise reproducibility and the IBM-off control.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm import flapping_lbm_2d as M  # noqa: E402
from tensorlbm.flapping_propulsion import (  # noqa: E402
    EnergyLedger,
    FlappingKinematics,
    garrick_coefficients,
    gci,
    k_from_strouhal,
    reduced_frequency,
    strouhal,
    theodorsen_C,
    theodorsen_C_bessel,
    theodorsen_response,
    theodorsen_table,
    thrust_reversal_heave,
)


# ---------------------------------------------------------------------------
# Theodorsen function
# ---------------------------------------------------------------------------
def test_theodorsen_exact_limits():
    assert abs(theodorsen_C(0.0) - 1.0) < 1e-15
    # k -> infinity: circulation cannot follow the motion
    assert abs(theodorsen_C(1e4).real - 0.5) < 1e-3
    assert abs(theodorsen_C(1e4).imag) < 1e-3


def test_theodorsen_sign_and_smoothness():
    ks = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    F = [theodorsen_C(k).real for k in ks]
    G = [theodorsen_C(k).imag for k in ks]
    assert all(f > 0.0 for f in F)          # F stays positive
    assert all(g < 0.0 for g in G)          # G strictly negative (phase lag)
    assert F == sorted(F, reverse=True)     # F decreases with k
    assert all(abs(g) < abs(f) for f, g in zip(F, G))


def test_theodorsen_two_independent_routes_agree():
    for k in (0.05, 0.2, 0.7, 1.5, 3.0, 8.0):
        a = theodorsen_C(k)
        b = theodorsen_C_bessel(k)
        assert abs(a - b) < 1e-12, (k, a, b)


def test_theodorsen_table_schema():
    rows = theodorsen_table([0.1, 1.0])
    assert set(rows[0]) == {
        "k", "F", "G", "absC", "absC2", "bessel_re", "bessel_im", "hankel_minus_bessel"
    }
    for r in rows:
        assert abs(r["absC"] ** 2 - r["absC2"]) < 1e-15


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------
def test_strouhal_reduced_frequency_roundtrip():
    kin = FlappingKinematics(U=1.0, c=1.0, h0=0.05)
    for k in (0.2, 0.5, 1.0, 2.0):
        st = kin.strouhal(k)
        assert abs(strouhal(f=1.0 / kin.period(k), A=2 * kin.h0, U=1.0) - st) < 1e-14
        assert abs(k_from_strouhal(st, 0.05) - k) < 1e-12
        assert abs(reduced_frequency(1.0 / kin.period(k), 1.0, 1.0) - k) < 1e-12


def test_kinematics_time_domain_matches_phasors():
    kin = FlappingKinematics(U=1.0, c=1.0, h0=0.02, alpha0=0.05, phi=0.3)
    k = 0.7
    w = kin.omega(k)
    for t in (0.0, 0.37, 1.9):
        assert abs(kin.heave(t, k) - 0.02 * math.sin(w * t)) < 1e-14
        assert abs(kin.heave_rate(t, k) - 0.02 * w * math.cos(w * t)) < 1e-13
        assert abs(kin.pitch(t, k) - 0.05 * math.sin(w * t + 0.3)) < 1e-14


# ---------------------------------------------------------------------------
# Garrick closed forms
# ---------------------------------------------------------------------------
def test_garrick_pure_heave_matches_textbook_closed_form():
    """C_T = 4*pi*k^2*(h0/c)^2*|C|^2, C_P = 4*pi*k^2*(h0/c)^2*F."""
    for h0c in (0.02, 0.1):
        for k in (0.5, 1.0, 2.0):
            kin = FlappingKinematics(U=1.0, c=1.0, h0=h0c, alpha0=0.0, a=0.0)
            g = garrick_coefficients(kin, k)
            C = theodorsen_C(k)
            assert abs(g.C_T - 4 * math.pi * k**2 * h0c**2 * abs(C) ** 2) < 1e-14
            assert abs(g.C_P - 4 * math.pi * k**2 * h0c**2 * C.real) < 1e-14
            assert abs(g.eta - abs(C) ** 2 / C.real) < 1e-12


def test_garrick_phasor_equals_quadrature():
    for k in (0.3, 1.0, 2.5):
        kin = FlappingKinematics(U=1.0, c=1.0, h0=0.05, alpha0=0.1,
                                 phi=math.radians(70), a=-0.5)
        a = garrick_coefficients(kin, k)
        b = garrick_coefficients(kin, k, n_samples=8192)
        assert abs(a.C_T - b.C_T) < 1e-12 + 1e-9 * abs(a.C_T)
        assert abs(a.C_P - b.C_P) < 1e-12 + 1e-9 * abs(a.C_P)


def test_efficiency_is_physical():
    """0 <= eta <= 1 for every tested harmonic motion (energy conservation)."""
    for k in (0.2, 0.5, 1.0, 2.0, 4.0):
        for h0c, a0, phi in ((0.02, 2.0, 90.0), (0.1, 5.0, 45.0),
                             (0.15, 10.0, 90.0), (0.05, 0.0, 0.0)):
            kin = FlappingKinematics(U=1.0, c=1.0, h0=h0c,
                                     alpha0=math.radians(a0),
                                     phi=math.radians(phi), a=-0.5)
            g = garrick_coefficients(kin, k)
            if g.C_T > 0:
                assert 0.0 < g.eta <= 1.0 + 1e-9, (k, h0c, a0, phi, g.eta)
            assert g.C_P_loss >= -1e-12  # thrust power never exceeds input power


def test_quasi_steady_pitch_gives_no_thrust():
    """A slowly pitching flat plate produces no net thrust (k -> 0)."""
    kin = FlappingKinematics(U=1.0, c=1.0, h0=0.0, alpha0=math.radians(5), a=-0.5)
    g = garrick_coefficients(kin, 1e-5)
    scale = math.pi * kin.alpha0**2  # natural O(alpha^2) scale of the suction
    assert abs(g.C_T) < 0.02 * scale


def test_pure_heave_k_to_zero_limit():
    """C_T -> 2*pi*b*mean(hdot^2)/q in the quasi-steady limit."""
    kin = FlappingKinematics(U=1.0, c=1.0, h0=0.02, alpha0=0.0, a=0.0)
    k = 1e-3
    g = garrick_coefficients(kin, k)
    w = kin.omega(k)
    ref = 2 * math.pi * 0.5 * (0.02 * w) ** 2 / 2 / 0.5
    assert abs(g.C_T - ref) / ref < 0.02


def test_theodorsen_response_amplitude_pure_heave():
    """Phasor lift of a heaving plate: L = i*pi*b^2*omega^2*h0 + 2*pi*U*b*C*omega*h0.

    (independent algebraic statement of the added-mass + circulatory parts)
    """
    kin = FlappingKinematics(U=1.0, c=1.0, h0=0.02, alpha0=0.0, a=0.0)
    for k in (0.5, 1.0, 2.0):
        r = theodorsen_response(kin, k)
        w_om = kin.omega(k)
        b = 0.5
        ref = 1j * math.pi * b**2 * w_om**2 * kin.h0 + 2 * math.pi * 1.0 * b * r.C * w_om * kin.h0
        assert abs(r.L_hat - ref) / abs(ref) < 1e-12
        # the added-mass part is O(omega^2) and becomes comparable at k=2
        circ = abs(2 * math.pi * b * r.C * w_om * kin.h0)
        assert r.lift_amplitude > 0.5 * circ


# ---------------------------------------------------------------------------
# thrust reversal
# ---------------------------------------------------------------------------
def test_thrust_reversal_bisection():
    res = thrust_reversal_heave(k=1.0, alpha0=0.0, phi=0.0, a=-0.5, cd0=0.01)
    kin = FlappingKinematics(U=1.0, c=1.0, h0=res["h0_over_c_crit"], alpha0=0.0, a=-0.5)
    assert abs(garrick_coefficients(kin, 1.0).C_T - 0.01) < 1e-9
    assert res["St_crit"] > 0.0
    # no reversal when the drag level is unreachable
    far = thrust_reversal_heave(k=1.0, alpha0=0.0, phi=0.0, a=-0.5, cd0=1e6)
    assert far["exists"] is False


# ---------------------------------------------------------------------------
# GCI
# ---------------------------------------------------------------------------
def test_gci_second_order():
    h = np.array([4.0, 2.0, 1.0])
    exact = 3.14159265358979
    f = exact + 0.5 * h**2            # pure second-order error
    g = gci(f_fine=f[2], f_medium=f[1], f_coarse=f[0], r=2.0)
    assert abs(g.p_observed - 2.0) < 1e-10
    assert abs(g.error_ratio - 4.0) < 1e-10
    assert abs(g.f_extrapolated - exact) < 1e-10


def test_gci_first_order():
    h = np.array([4.0, 2.0, 1.0])
    f = 1.0 + 0.3 * h
    g = gci(f_fine=f[2], f_medium=f[1], f_coarse=f[0], r=2.0)
    assert abs(g.p_observed - 1.0) < 1e-10
    assert abs(g.error_ratio - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# energy ledger
# ---------------------------------------------------------------------------
def test_energy_ledger_closure():
    led = EnergyLedger()
    led.add(W=1.0, dKE=0.1, Phi=0.85, flux=0.05)
    led.add(W=2.0, dKE=0.0, Phi=1.9, flux=0.1)
    s = led.residual()
    assert s["relative_residual"] < 1e-15


# ---------------------------------------------------------------------------
# LBM smoke tests (chord 8, a couple of cycles: fast)
# ---------------------------------------------------------------------------
def _tiny_cfg(**kw):
    base = dict(chord=8, re=200.0, U=0.15, h0_over_c=0.05, alpha0_deg=2.0,
                k=2.0, phi_deg=90.0, n_cycles=2, warmup_cycles=1,
                n_samples_per_cycle=16)
    base.update(kw)
    return M.LBMConfig(**base)


def test_lbm_uniform_flow_is_stationary():
    cfg = _tiny_cfg(ibm=False, n_cycles=1, warmup_cycles=0)
    s = M.FlappingFoil2D(cfg)
    ke0 = None
    for _ in range(60):
        d = s.step()
        if ke0 is None:
            ke0 = d["ke"]
    assert abs(d["ke"] - ke0) / ke0 < 1e-10
    rho, ux, uy = s.macros()
    assert np.abs(uy).max() < 1e-12
    assert abs(ux.max() - cfg.U) < 1e-12 and abs(ux.min() - cfg.U) < 1e-12


def test_lbm_locked_plate_produces_drag():
    cfg = _tiny_cfg(locked=True, n_cycles=4, warmup_cycles=1)
    s = M.FlappingFoil2D(cfg)
    res = s.run()
    avg = M.cycle_average(res, cfg.U, cfg.chord)
    assert math.isfinite(avg["C_T"])
    assert avg["C_T"] < 0.0            # drag, i.e. negative thrust
    assert avg["C_P"] == 0.0           # a motionless body injects no work


def test_lbm_ibm_off_gives_zero_force():
    cfg = _tiny_cfg(ibm=False, n_cycles=1, warmup_cycles=0)
    s = M.FlappingFoil2D(cfg)
    for _ in range(40):
        d = s.step()
        assert d["Fx_fluid"] == 0.0 and d["Fy_fluid"] == 0.0
        assert d["body_work"] == 0.0


def test_lbm_marker_kinematics():
    cfg = _tiny_cfg(chord=8, h0_over_c=0.1, alpha0_deg=0.0, phi_deg=0.0, k=1.0)
    kin = cfg.kinematic
    T = cfg.period()
    pos0, vel0 = M.build_markers(cfg, 0.0)
    pos_q, vel_q = M.build_markers(cfg, T / 4.0)
    # heave is positive downward: at t=T/4 the foil sits lower in the lattice
    assert pos_q[1].mean() < pos0[1].mean()
    assert abs(vel0[1].mean() - (-kin.h0 * kin.omega(cfg.k))) < 1e-10
    # a quarter period later the vertical velocity is zero
    assert abs(vel_q[1].mean()) < 1e-10


def test_lbm_bitwise_reproducibility(tmp_path):
    shas = []
    for tag in ("a", "b"):
        cfg = _tiny_cfg(n_cycles=2, warmup_cycles=1)
        s = M.FlappingFoil2D(cfg)
        csv_path = str(tmp_path / f"run_{tag}.csv")
        s.run(csv_path=csv_path)
        shas.append(M.file_sha256(csv_path))
    assert shas[0] == shas[1]


def test_lbm_energy_budget_terms_are_finite():
    cfg = _tiny_cfg(n_cycles=2, warmup_cycles=1)
    s = M.FlappingFoil2D(cfg)
    res = s.run()
    rows = res["rows"]
    W = sum(r["W"] for r in rows)
    dKE = rows[-1]["ke"] - rows[0]["ke"]
    phi = sum(r["phi"] for r in rows)
    flux = sum(r["flux"] for r in rows)
    assert math.isfinite(W) and math.isfinite(phi) and math.isfinite(flux)
    assert phi > 0.0                     # dissipation is positive
    rel = abs(W - dKE - phi - flux) / max(abs(W), 1e-30)
    # The open-domain budget is *not* closed: the velocity inlet and the sponge
    # outlet are not energy-conserving, so the boundary pressure work dominates.
    # What is asserted here is that the four terms are finite, that dissipation
    # is positive and that the residual is a finite, reported quantity.
    assert math.isfinite(rel)
    phi_neq = sum(r.get("phi_neq", 0.0) for r in rows)
    assert phi_neq > 0.0


def test_peskin_kernel_normalisation():
    r = np.linspace(-2.0, 2.0, 20001)
    w = M.peskin4(r)
    assert abs(np.trapezoid(w, r) - 1.0) < 1e-6
    assert M.peskin4(np.array([2.5]))[0] == 0.0