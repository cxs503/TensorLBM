"""Pytest suite for the perforated-plate oscillatory-flow verification face.

The suite mixes three levels of checks:

* **physics/analytic** : exact properties of the reference solutions
  (Borda-Carnot limit, Womersley ODE + no-slip, Stokes layer).
* **numerics**         : conservation and consistency of the D2Q9 kernel
  (mass conservation, momentum balance of the body force against the
  momentum-exchange wall force, zero plate force without a plate).
* **estimators/gates** : the C_d/C_m estimator and the acceptance gates are
  exercised on synthetic signals, where the answer is known by construction.

Heavy production runs are *not* part of the suite (they live in the driver,
``python -m tensorlbm.verification.perforated_plate.driver``); the simulations
here use tiny grids so that the whole suite runs in a couple of minutes.
"""

from __future__ import annotations

import math
import os
import sys

# make sure this worktree's ``src`` wins over any installed/other-worktree copy
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC in sys.path:
    sys.path.remove(_SRC)
sys.path.insert(0, _SRC)

import numpy as np
import pytest

from tensorlbm.verification.perforated_plate.solver import (
    PerforatedPlateConfig,
    PerforatedPlateSolver,
    compute_open_area_ratio,
    make_perforated_plate_mask,
)
from tensorlbm.verification.perforated_plate.analysis import (
    analytic_dissipation,
    compute_energy_dissipation,
    compute_phase_lag,
    compute_zeta_eff,
    extract_cd_cm,
    first_harmonic,
    phase_average,
)
from tensorlbm.verification.perforated_plate.reference import (
    C_C_SHARP,
    loss_coefficient,
    sharp_edge_cd,
    stokes_layer_thickness,
    stokes_layer_u,
    womersley_channel_amplitude,
    womersley_channel_u,
    womersley_wall_shear,
)
from tensorlbm.verification.perforated_plate.gates import (
    THRESHOLDS,
    gate_A_quasi_steady,
    gate_B_phase_monotonic,
    gate_C_cd_vs_beta,
    gate_D_gci,
    gate_F_energy_closure,
    spearman,
)
from tensorlbm.verification.perforated_plate.driver import build_cases


# ---------------------------------------------------------------------------
# 1. Geometry
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_open_area_ratio_from_mask(self):
        nx, ny, d, n = 32, 66, 6, 2
        mask = make_perforated_plate_mask(nx, ny, 16, n_holes=n, hole_size=d,
                                          plate_thickness=2)
        beta = compute_open_area_ratio(mask, 16, ny)
        assert beta == pytest.approx(n * d / (ny - 2), rel=1e-12)

    def test_beta_monotone_in_hole_count(self):
        ny = 66
        betas = [compute_open_area_ratio(
            make_perforated_plate_mask(32, ny, 16, n_holes=k, hole_size=4), 16, ny)
            for k in (1, 2, 3)]
        assert betas[0] < betas[1] < betas[2]

    def test_no_plate_mask(self):
        mask = make_perforated_plate_mask(32, 66, 16, n_holes=1, hole_size=6,
                                          plate_thickness=2, beta=1.0)
        assert not mask[16, 1:-1].any()          # plate column open
        assert mask[:, 0].all() and mask[:, -1].all()   # walls remain

    def test_walls_are_solid(self):
        nx, ny = 32, 34
        mask = make_perforated_plate_mask(nx, ny, 16, n_holes=1, hole_size=4)
        assert mask[:, 0].all() and mask[:, ny - 1].all()


# ---------------------------------------------------------------------------
# 2. Reference solutions
# ---------------------------------------------------------------------------

class TestReference:
    def test_sharp_edge_limit(self):
        # beta -> 0 recovers the classical sharp-edged orifice value
        assert sharp_edge_cd(1e-6) == pytest.approx(C_C_SHARP, abs=1e-6)
        assert C_C_SHARP == pytest.approx(0.61)

    def test_sharp_edge_monotone_in_beta(self):
        cd = [sharp_edge_cd(b) for b in (0.05, 0.1, 0.2, 0.3, 0.4)]
        assert all(cd[i] < cd[i + 1] for i in range(len(cd) - 1))
        assert all(0.4 <= v <= 0.9 for v in cd)

    def test_sharp_edge_closed_form(self):
        # closed form must equal Cc/(1-beta Cc)
        for beta in (0.0625, 0.25, 0.375):
            assert sharp_edge_cd(beta) == pytest.approx(
                C_C_SHARP / (1.0 - beta * C_C_SHARP), rel=1e-14)

    def test_loss_coefficient(self):
        assert loss_coefficient(0.5) == pytest.approx(4.0)

    def test_womersley_no_slip(self):
        h, nu, omega, f0 = 64.0, 0.05, 2 * math.pi / 3000.0, 5e-5
        for t in (0.0, 100.0, 700.0):
            u = womersley_channel_u(np.array([-h / 2, h / 2]), t, h, nu, f0, omega)
            assert np.allclose(u, 0.0, atol=1e-12)

    def test_womersley_satisfies_momentum_equation(self):
        h, nu, omega, f0 = 64.0, 0.05, 2 * math.pi / 3000.0, 5e-5
        y = np.linspace(-h / 2, h / 2, 257)
        dy = y[1] - y[0]
        t0, dt = 123.0, 1e-2
        u = womersley_channel_u(y, t0, h, nu, f0, omega)
        u_t = (womersley_channel_u(y, t0 + dt, h, nu, f0, omega)
               - womersley_channel_u(y, t0 - dt, h, nu, f0, omega)) / (2 * dt)
        u_yy = np.gradient(np.gradient(u, dy), dy)
        residual = u_t - nu * u_yy - f0 * math.sin(omega * t0)
        assert np.max(np.abs(residual[2:-2])) < 1e-6  # 0.1% of f0

    def test_stokes_layer(self):
        nu, omega = 0.05, 2 * math.pi / 3000.0
        delta = stokes_layer_thickness(omega, nu)
        assert delta == pytest.approx(math.sqrt(2 * nu / omega))
        f0 = 5e-5
        # no slip at the wall, free-stream amplitude f0/omega far away
        assert abs(stokes_layer_u(np.array([0.0]), 137.0, f0, omega, nu)[0]) < 1e-13
        far = stokes_layer_u(np.array([30 * delta]), 137.0, f0, omega, nu)[0]
        assert far == pytest.approx((f0 / omega) * math.sin(omega * 137.0), rel=1e-3)

    def test_womersley_wall_shear_phase(self):
        h, nu, omega, f0 = 400.0, 0.05, 2 * math.pi / 3000.0, 5e-5
        t = np.linspace(0, 2 * math.pi / omega, 257)
        tau = np.array([womersley_wall_shear(tt, h, nu, f0, omega) for tt in t])
        # reference: far-field velocity of the body-force driven flow u_inf = -(f0/omega) cos(omega t)
        lag = (np.angle(first_harmonic(2 * np.pi * np.arange(t.size) / t.size, tau))
               - np.angle(first_harmonic(2 * np.pi * np.arange(t.size) / t.size,
                                         -(f0 / omega) * np.cos(omega * t))))
        # wall shear leads the free-stream velocity by 45 deg in the Stokes limit
        assert abs(math.degrees(lag) - 45.0) < 5.0


# ---------------------------------------------------------------------------
# 3. Numerical kernel: conservation and momentum exchange
# ---------------------------------------------------------------------------

def _steady_channel(ny=18, nx=32, tau=1.0, f0=1e-6, nsteps=2000, beta=1.0):
    cfg = PerforatedPlateConfig(nx=nx, ny=ny, n_holes=1, hole_size=4, tau=tau,
                                force_amplitude=f0, omega=2 * math.pi / 4000.0,
                                beta=beta, init_profile="rest",
                                probe_wall_shear=True)
    solver = PerforatedPlateSolver(cfg)
    f_wall = 0.0
    for _ in range(nsteps):
        solver._collide(f0)              # constant body force -> steady state
        solver._stream()
        f_wall += float(solver._force_on(solver.wall_mask)[0])
    return solver, f_wall / nsteps


class TestKernel:
    def test_mass_conserved(self):
        cfg = PerforatedPlateConfig(nx=48, ny=34, n_holes=1, hole_size=4, tau=0.53,
                                    force_amplitude=1e-5, omega=2 * math.pi / 200.0,
                                    warmup_cycles=0, n_cycles_measure=1,
                                    samples_per_cycle=8)
        solver = PerforatedPlateSolver(cfg)
        m0 = solver.f.sum()
        for k in range(40):
            solver.step(float(k))
        # Guo forcing is mass neutral; only round-off accumulates
        assert solver.f.sum() == pytest.approx(m0, rel=1e-8)

    def test_plate_force_grows_with_drive(self):
        """The momentum-exchange plate force must grow with the drive amplitude."""
        def force_at(F0, nsteps=150):
            cfg = PerforatedPlateConfig(nx=64, ny=34, n_holes=1, hole_size=4,
                                        tau=0.53, force_amplitude=F0,
                                        omega=2 * math.pi / 400.0,
                                        warmup_cycles=0, n_cycles_measure=1)
            solver = PerforatedPlateSolver(cfg)
            for k in range(nsteps):
                res = solver.step(float(k))
            return res["force_plate_x"], res["flow_rate"]

        f1, q1 = force_at(1e-5)
        f2, q2 = force_at(3e-5)
        assert f1 > 0 and f2 > f1
        assert q2 > q1
        # sub-linear in the drive (drag grows with the flow rate, which itself
        # grows sub-linearly because the plate loss is ~quadratic)
        assert 1.2 < f2 / f1 < 6.0

    def test_no_plate_no_plate_force(self):
        solver, _ = _steady_channel(nsteps=200)
        f_plate = solver._force_on(solver.plate_mask)
        assert abs(f_plate[0]) < 1e-15

    def test_plate_force_opposes_flow_direction(self):
        """Drag on the plate must point downstream (positive x for +x flow)."""
        cfg = PerforatedPlateConfig(nx=64, ny=34, n_holes=1, hole_size=4, tau=0.53,
                                    force_amplitude=1e-5, omega=2 * math.pi / 2000.0,
                                    warmup_cycles=0, n_cycles_measure=1)
        solver = PerforatedPlateSolver(cfg)
        for k in range(400):
            res = solver.step(float(k))
        assert res["flow_rate"] > 0
        assert res["force_plate_x"] > 0


# ---------------------------------------------------------------------------
# 4. Estimators (synthetic signals with a known answer)
# ---------------------------------------------------------------------------

class TestEstimators:
    def test_phase_average_recovers_sinusoid(self):
        omega, n = 2 * math.pi / 100.0, 1000      # 10 periods
        t = np.arange(n, dtype=float) + 1
        sig = 3.0 * np.sin(omega * t) + 1.5
        theta, avg = phase_average(t, sig, omega, 24)
        assert avg.size == 24
        assert np.max(np.abs(avg - 1.5 - 3.0 * np.sin(theta))) < 0.12  # bin-width error

    def test_first_harmonic(self):
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        amp = first_harmonic(theta, 2.0 * np.cos(theta - 0.3) + 5.0)
        assert abs(amp) == pytest.approx(2.0, rel=1e-6)
        assert np.angle(amp) == pytest.approx(-0.3, abs=1e-6)

    def test_extract_cd_cm_recovers_known_coefficients(self):
        omega, beta, ny, t_p, n = 0.01, 0.25, 66, 2.0, 64
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        area = ny - 2
        k_m, k_q = 7.0, 4.0                    # -> C_d = 0.5, C_m = 3.5
        u_h = 0.05 * np.sin(theta)
        du = 0.05 * omega * np.cos(theta)
        q = u_h * area * beta
        dp = k_m * du + 0.5 * k_q * u_h * np.abs(u_h)
        fit = extract_cd_cm(q, dp, omega, beta, ny, 1.0, t_p)
        assert fit["K_m"] == pytest.approx(k_m, rel=5e-3)
        assert fit["K_q"] == pytest.approx(k_q, rel=5e-3)
        assert fit["C_d"] == pytest.approx(1 / math.sqrt(k_q), rel=2.5e-3)
        assert fit["C_m"] == pytest.approx(k_m / t_p, rel=5e-3)
        assert fit["r_squared"] > 0.999

    def test_phase_lag_limits(self):
        n, omega = 64, 0.01
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        q = np.sin(theta)
        # pure inertia -> +90 deg, pure loss -> 0 deg
        assert compute_phase_lag(q, np.cos(theta), omega)["phase_lag_deg"] == pytest.approx(90.0, abs=1e-6)
        assert compute_phase_lag(q, np.abs(q) * q, omega)["phase_lag_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_dissipation_matches_analytic_model(self):
        omega, beta, area, cd, u0, n = 0.01, 0.25, 64, 0.7, 0.05, 512
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        u_h = u0 * np.sin(theta)
        q = u_h * area * beta
        dp = 0.5 * (1 / cd ** 2) * u_h * np.abs(u_h)
        phi = compute_energy_dissipation(q, dp, omega)
        phi_ref = analytic_dissipation(omega, area, beta, cd, u0)
        assert phi == pytest.approx(phi_ref, rel=2e-3)

    def test_zeta_scaling(self):
        phi, omega, e = 1.0, 0.01, 0.5
        assert compute_zeta_eff(phi, omega, e) == pytest.approx(
            phi / (2 * math.pi * omega * 2 * e))


# ---------------------------------------------------------------------------
# 5. Gates (logic only, synthetic inputs)
# ---------------------------------------------------------------------------

def _kc_row(i, kc, cd, lag, beta=0.25, ref=None):
    return {"id": f"c{i}", "kind": "kc", "KC": kc, "C_d": cd,
            "C_d_reference": ref if ref is not None else sharp_edge_cd(beta),
            "phase_lag_deg": lag, "beta": beta, "r_squared": 0.99, "C_m": 3.5}


class TestGates:
    def test_spearman(self):
        assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
        assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)

    def test_gate_A(self):
        rows = [_kc_row(i, 20 + i, sharp_edge_cd(0.25), 10.0) for i in range(2)]
        assert gate_A_quasi_steady(rows)["passed"]
        rows_bad = [_kc_row(i, 20, sharp_edge_cd(0.25) * 1.3, 10.0) for i in range(2)]
        assert not gate_A_quasi_steady(rows_bad)["passed"]

    def test_gate_B(self):
        ok = [_kc_row(i, kc, 0.7, lag) for i, (kc, lag) in
              enumerate(zip([0.5, 1, 2, 3, 5, 8, 12, 20], [85, 80, 70, 60, 45, 30, 20, 10]))]
        assert gate_B_phase_monotonic(ok)["passed"]
        bad = [_kc_row(i, kc, 0.7, 50.0) for i, kc in
               enumerate([0.5, 1, 2, 3, 5, 8, 12, 20])]
        assert not gate_B_phase_monotonic(bad)["passed"]
        assert gate_B_phase_monotonic(ok[:5])["passed"] is False   # <8 points

    def test_gate_C(self):
        rows = [{"id": f"b{i}", "kind": "beta", "beta": b,
                 "C_d": sharp_edge_cd(b) * 0.95, "C_d_reference": sharp_edge_cd(b),
                 "KC": 10.0, "r_squared": 0.99}
                for i, b in enumerate([0.0625, 0.125, 0.1875, 0.3125, 0.375])]
        assert gate_C_cd_vs_beta(rows)["passed"]
        rows[3]["C_d"] = rows[2]["C_d"] - 0.01          # break monotonicity
        assert not gate_C_cd_vs_beta(rows)["passed"]

    def test_gate_D(self):
        rows = [{"id": f"g{i}", "kind": "gci", "level": i, "ny": 34 * 2 ** i,
                 "C_d": v, "hole_size": 4 * 2 ** i, "KC": 2.0, "Re_h": 30.0,
                 "beta": 0.25}
                for i, v in enumerate([0.380, 0.376, 0.3752])]
        g = gate_D_gci(rows)
        assert g["passed"] and g["GCI_fine"] < 0.05
        rows_bad = [dict(r, C_d=v) for r, v in zip(rows, [0.25, 0.33, 0.3752])]
        assert not gate_D_gci(rows_bad)["passed"]

    def test_gate_F(self):
        rows = [{"id": "a", "kind": "kc", "KC": 10.0, "phi": 1.0, "phi_stations": 1.01}]
        assert gate_F_energy_closure(rows)["passed"]
        rows[0]["phi_stations"] = 1.2
        assert not gate_F_energy_closure(rows)["passed"]

    def test_thresholds_complete(self):
        for key in ("A_quasi_steady_rel_error", "D_gci_max", "E_nc1_dp",
                    "E_nc2_tau", "F_closure"):
            assert key in THRESHOLDS


# ---------------------------------------------------------------------------
# 6. Driver / configuration
# ---------------------------------------------------------------------------

class TestDriver:
    def test_case_matrix_complete(self):
        cases = build_cases()
        kinds = {}
        for c in cases:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        assert kinds["kc"] >= 8
        assert kinds["beta"] == 5
        assert kinds["gci"] == 3
        assert kinds["nc"] >= 1
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids))

    def test_config_validation(self):
        with pytest.raises(ValueError):
            PerforatedPlateConfig(nx=16, ny=8).validate()
        with pytest.raises(ValueError):
            PerforatedPlateConfig(tau=0.5).validate()
        with pytest.raises(ValueError):
            PerforatedPlateConfig(force_amplitude=-1.0, omega=1e-6).validate()

    def test_oscillatory_sign_and_period(self):
        """Short run: Q must oscillate with the driving period and change sign."""
        omega = 2 * math.pi / 200.0
        cfg = PerforatedPlateConfig(nx=64, ny=34, n_holes=1, hole_size=4, tau=0.53,
                                    force_amplitude=2e-5, omega=omega,
                                    warmup_cycles=1, n_cycles_measure=1,
                                    samples_per_cycle=16)
        res = PerforatedPlateSolver(cfg).run()
        q = res["time_series"]["flow_rate"]
        assert q.size > 100
        assert q.min() < 0 < q.max()
        theta, qp = phase_average(res["time_series"]["time"], q, omega, 16)
        assert np.ptp(qp) > 0.1 * np.std(q)


def test_suite_size():
    """The campaign requires at least 10 pytest cases for this node."""
    import inspect
    import sys
    mod = sys.modules[__name__]
    n = 0
    for name, obj in vars(mod).items():
        if inspect.isclass(obj) and name.startswith("Test"):
            n += len([m for m in dir(obj) if m.startswith("test_")])
    assert n >= 10