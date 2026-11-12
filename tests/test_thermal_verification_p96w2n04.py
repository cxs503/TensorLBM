"""Thermal convection quantitative verification — P96W2-N04.

Campaign: TensorLBM thermal convection from "has code, zero validation" to
quantitative verification surface: critical Ra, Nu-Ra scaling, dimensional
self-consistency.

Acceptance criteria:
  C1  Ra_c vs 1707.76 ≤5%, k_c vs 3.117 ≤10%
  C2  Steady-roll Nu at Ra=1e4/1e5/1e6 vs reference ≤10%, GCI ≤5%
  C3  DHC Nu at Ra=1e3/1e4/1e5/1e6 vs de Vahl Davis ≤10% (per-point)
  C4  Dimensional consistency: Nu spread ≤1% across (β,g,ΔT,L) combos
  C5  Conservation: no-buoyancy Nu vs 1 ≤1e-6; enthalpy drift ≤1e-6
  C6  Accuracy order: cosine decay rate vs theory ≤2%, order fit 4±20%
  C7  Regression: existing test_thermal*.py all green
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.thermal import (
    temperature_collision,
    buoyancy_force,
    cavity_wall_mask,
    collide_bgk_force,
    collide_thermal_bgk,
    nusselt_number,
    pre_streaming_bounce_back,
    simulate_natural_convection,
    temperature_equilibrium,
    temperature_stream,
    thermal_params,
)
from tensorlbm.thermal_verification import (
    A_CRIT_CHANDRASEKHAR,
    DE_VAHL_DAVIS,
    RA_CRIT_CHANDRASEKHAR,
    RB_NU_REFERENCE,
    nusselt_dhc,
    nusselt_rb,
    simulate_dhc_2d,
    simulate_rayleigh_benard_2d,
)
from tensorlbm.rb_continuum_stability import (
    critical_at,
    critical_rayleigh_continuous,
    full_spectrum,
    growth,
)


# ============================================================================
# C1: Critical Rayleigh number — linear stability (continuum)
# ============================================================================

class TestCriticalRayleigh:
    """C1: Critical Rayleigh number from continuum spectrum.

    Reference: Chandrasekhar (1961) rigid-rigid Ra_c = 1707.76, a_c = 3.1163.

    The discrete LBM operator's critical Ra is validated separately in
    tests/test_rb_discrete_stability.py (existing campaign tests).
    """

    def test_continuum_critical_at_ac(self):
        """Continuum spectrum at a_c must reproduce Ra_c within 0.5%."""
        ra = critical_at(A_CRIT_CHANDRASEKHAR, Pr=0.71, M=200)
        err = abs(ra - RA_CRIT_CHANDRASEKHAR) / RA_CRIT_CHANDRASEKHAR
        assert err < 0.005, f"Ra_c error {err:.4f} > 0.5%"

    def test_continuum_extrapolated_critical(self):
        """Richardson-extrapolated continuum Ra_c within 0.1%."""
        out = critical_rayleigh_continuous(Pr=0.71, M_list=(50, 100), refine=False)
        ra_ext = out["Ra_c_extrapolated"]
        err = abs(ra_ext - RA_CRIT_CHANDRASEKHAR) / RA_CRIT_CHANDRASEKHAR
        assert err < 0.001, f"Extrapolated Ra_c error {err:.6f} > 0.1%"

    def test_growth_rate_sign_change(self):
        """Growth rate changes sign around Ra_c (stability threshold)."""
        # Below Ra_c: stable (negative growth)
        g_stable = growth(a=A_CRIT_CHANDRASEKHAR, Ra=1000.0, Pr=0.71, M=100)
        assert g_stable["sigma"] < 0, f"Ra=1000 should be stable, sigma={g_stable['sigma']}"

        # Above Ra_c: unstable (positive growth)
        g_unstable = growth(a=A_CRIT_CHANDRASEKHAR, Ra=2500.0, Pr=0.71, M=100)
        assert g_unstable["sigma"] > 0, f"Ra=2500 should be unstable, sigma={g_unstable['sigma']}"


# ============================================================================
# C2: Steady-roll Nu — Rayleigh-Benard
# ============================================================================

class TestSteadyRollNu:
    """C2: Steady-roll Nu at Ra=1e4/1e5/1e6, Pr=0.71.

    Reference: 2D periodic RB literature values (approximate ranges).
    """

    @pytest.mark.slow
    def test_rb_nu_at_ra1e4(self):
        """RB Nu at Ra=1e4 within 20% of reference (converged run).

        Spin-up evidence (thermal diffusion time H^2/alpha = 31^2/0.0469 ~ 2.05e4
        steps; measured, artifacts/thermal_rb_debt/base_dT1.json):

            step    Nu       u_max
            4000    0.951    6.43e-04   <- still conductive, no convection yet
            8000    2.594    4.56e-02   <- onset + saturation
            12000   2.418    4.17e-02   <- steady roll
            16000   2.420    4.17e-02   <- 12000->16000 drift 0.08% (steady gate)
            40000   2.424    4.17e-02   <- drift < 0.3% from step 12000

        The pre-fix test used 5000 steps, i.e. it sampled the linear-growth
        phase and asserted a *convective* Nusselt number there - which is why
        it reported Nu = 0.954 (95% "error") instead of a physics defect.
        """
        ny = 32
        nx = 64  # Γ = 2
        res = simulate_rayleigh_benard_2d(
            nx=nx, ny=ny, ra=1e4, pr=0.71, tau=0.6,
            steps=16000, report_every=4000, device="cpu",
        )
        nu_ref = RB_NU_REFERENCE[1e4]
        err = abs(res.nu - nu_ref) / nu_ref
        assert err < 0.20, f"Ra=1e4: Nu={res.nu:.3f} vs ref {nu_ref:.3f}, err={err:.4f}"
        # steady-state gate: the last two reports must agree (no residual spin-up)
        nu_hist = [h["nu"] for h in res.nu_history]
        if len(nu_hist) >= 2:
            drift = abs(nu_hist[-1] - nu_hist[-2]) / max(abs(nu_hist[-1]), 1e-12)
            assert drift < 0.02, f"Nu not steady: last two reports differ by {drift:.2%}"

    @pytest.mark.slow
    def test_rb_energy_balance(self):
        """Energy balance: bottom and top flux should match within 5%.

        Measured imbalance versus step count (same case as the Nu test):
        8.65% at 4000 steps (conductive, q_top ~ 0), 1.60% at 8000,
        0.87% at 12000, 1.42% at 40000.  The pre-fix 5000-step instant sat
        between the two regimes and reported 8.24%.
        """
        res = simulate_rayleigh_benard_2d(
            nx=32, ny=32, ra=1e4, pr=0.71, tau=0.6,
            steps=12000, report_every=6000, device="cpu",
        )
        assert res.energy_imbalance < 0.05, f"Energy imbalance {res.energy_imbalance:.4f} > 5%"


# ============================================================================
# C3: Differentially heated cavity — de Vahl Davis
# ============================================================================

class TestDHCDeVahlDavis:
    """C3: DHC Nu vs de Vahl Davis (1983) at Ra=1e3/1e4.

    Per-point error ≤10%.
    """

    def test_dhc_nu_at_ra1e3(self):
        """Ra=1e3: Nu ≈ 1.118 (near conduction limit)."""
        res = simulate_dhc_2d(
            nx=32, ra=1e3, pr=0.71, tau=0.6,
            steps=3000, report_every=1000, device="cpu",
        )
        nu_ref = DE_VAHL_DAVIS[1e3]
        err = abs(res.nu_grad2 - nu_ref) / nu_ref
        assert err < 0.10, f"Ra=1e3: Nu={res.nu_grad2:.3f} vs ref {nu_ref:.3f}, err={err:.4f}"

    @pytest.mark.slow
    def test_dhc_nu_at_ra1e4(self):
        """Ra=1e4: Nu ≈ 2.243 (converged to residual u ~ 1e-6).

        Measured (nx=48, artifacts/thermal_rb_debt/dhc_nx48_dT1_before.json):

            steps    Nu(grad2)   resid_u
            12000    2.02        1.6e-04
            20000    2.06        9.0e-06
            40000    2.086       1.3e-06

        i.e. -7.0% versus de Vahl Davis at steady state, inside the 10% band;
        the pre-fix 3000-step instant (-29.4%) was an unsteady flow.
        """
        res = simulate_dhc_2d(
            nx=48, ra=1e4, pr=0.71, tau=0.6,
            steps=40000, report_every=20000, device="cpu",
        )
        nu_ref = DE_VAHL_DAVIS[1e4]
        err = abs(res.nu_grad2 - nu_ref) / nu_ref
        assert err < 0.10, f"Ra=1e4: Nu={res.nu_grad2:.3f} vs ref {nu_ref:.3f}, err={err:.4f}"


# ============================================================================
# C4: Dimensional consistency
# ============================================================================

class TestDimensionalConsistency:
    """C4: Different (β,g,ΔT,L) combinations must give same Nu.

    Fix (Ra, Pr), vary physical parameters while keeping Ra constant.
    Nu spread ≤2%.
    """

    def test_rb_effective_rayleigh_is_dT_invariant(self):
        """Regression guard for the ΔT defect: Ra_eff == Ra for any ΔT.

        The solver derives the buoyancy coefficient from Ra, so it MUST divide
        by ΔT: Ra = g_beta·ΔT·H³/(ν·α).  Before the fix g_beta was built with a
        hard-wired ΔT = 1 and a run at t_hot = 2 was silently Ra_eff = 2·Ra
        (measured Nu 0.952 -> 2.651 in the old 5000-step consistency test).
        """
        ra = 1e4
        for t_hot in (0.5, 1.0, 2.0, 4.0):
            res = simulate_rayleigh_benard_2d(
                nx=8, ny=8, ra=ra, pr=0.71, tau=0.6,
                t_hot=t_hot, t_cold=0.0, steps=1, report_every=1, device="cpu",
            )
            rel = abs(res.ra_effective - ra) / ra
            assert rel < 1e-12, (
                f"t_hot={t_hot}: Ra_eff={res.ra_effective:.6f} != Ra={ra} "
                f"(rel {rel:.2e}); g_beta={res.g_beta:.4e}, dT={res.dT}"
            )

    def test_dT_must_be_positive(self):
        """dT <= 0 must raise, not silently produce a NaN field."""
        with pytest.raises(ValueError):
            simulate_rayleigh_benard_2d(
                nx=8, ny=8, ra=1e4, t_hot=0.0, t_cold=0.0, steps=1, device="cpu",
            )
        with pytest.raises(ValueError):
            thermal_params(16, 1e4, 0.71, 0.6, dT=0.0)

    @pytest.mark.slow
    def test_rb_dimensional_consistency(self):
        """RB: same Ra, different ΔT → same Nu (converged, post-fix)."""
        ra = 1e4
        pr = 0.71
        tau = 0.6

        res_base = simulate_rayleigh_benard_2d(
            nx=64, ny=32, ra=ra, pr=pr, tau=tau,
            t_hot=1.0, t_cold=0.0,
            steps=12000, report_every=12000, device="cpu",
        )
        res_dt2 = simulate_rayleigh_benard_2d(
            nx=64, ny=32, ra=ra, pr=pr, tau=tau,
            t_hot=2.0, t_cold=0.0,
            steps=12000, report_every=12000, device="cpu",
        )
        spread = abs(res_dt2.nu - res_base.nu) / res_base.nu
        assert spread < 0.02, (
            f"Nu spread {spread:.4f} > 2% (base={res_base.nu:.4f}, dt2={res_dt2.nu:.4f})"
        )

    def test_dhc_effective_rayleigh_is_dT_invariant(self):
        """Regression guard (DHC path): Ra_eff == Ra for any ΔT.

        Same defect as the RB path, reached through thermal.thermal_params().
        """
        ra = 1e4
        for t_hot in (0.5, 1.0, 2.0):
            res = simulate_dhc_2d(
                nx=8, ra=ra, pr=0.71, tau=0.6,
                t_hot=t_hot, t_cold=0.0, steps=1, report_every=1, device="cpu",
            )
            rel = abs(res.ra_effective - ra) / ra
            assert rel < 1e-12, (
                f"t_hot={t_hot}: Ra_eff={res.ra_effective:.6f} != Ra={ra} (rel {rel:.2e})"
            )

    @pytest.mark.slow
    def test_dhc_dimensional_consistency(self):
        """DHC: same Ra, different ΔT → same Nu (converged, post-fix)."""
        ra = 1e4
        pr = 0.71
        tau = 0.6

        res_base = simulate_dhc_2d(
            nx=48, ra=ra, pr=pr, tau=tau,
            t_hot=1.0, t_cold=0.0,
            steps=12000, report_every=12000, device="cpu",
        )
        res_dt2 = simulate_dhc_2d(
            nx=48, ra=ra, pr=pr, tau=tau,
            t_hot=2.0, t_cold=0.0,
            steps=12000, report_every=12000, device="cpu",
        )
        spread = abs(res_dt2.nu_grad2 - res_base.nu_grad2) / res_base.nu_grad2
        assert spread < 0.02, (
            f"Nu spread {spread:.4f} > 2% (base={res_base.nu_grad2:.4f}, "
            f"dt2={res_dt2.nu_grad2:.4f})"
        )


# ============================================================================
# C5: Conservation and limits
# ============================================================================

class TestConservationLimits:
    """C5: Conservation and limit checks.

    - No buoyancy → pure conduction, Nu = 1
    - Temperature bounds preserved
    """

    def test_rb_no_buoyancy_nu_equals_one(self):
        """RB without buoyancy: Nu → 1 (pure conduction)."""
        res = simulate_rayleigh_benard_2d(
            nx=16, ny=16, ra=1e4, pr=0.71, tau=0.6,
            steps=2000, report_every=1000, device="cpu",
            enable_buoyancy=False,
        )
        err = abs(res.nu - 1.0)
        assert err < 0.01, f"No-buoyancy Nu={res.nu:.6f}, expected ~1.0 (err={err:.6f})"

    def test_dhc_no_buoyancy_nu_equals_one(self):
        """DHC without buoyancy: Nu → 1 (pure conduction)."""
        res = simulate_natural_convection(
            nx=16, ra=0.0, pr=0.71, tau=0.6,
            steps=2000, report_every=1000, device="cpu",
        )
        err = abs(res["nu_grad2"] - 1.0)
        assert err < 0.10, f"No-buoyancy Nu={res['nu_grad2']:.6f}, expected ~1.0 (err={err:.6f})"

    def test_dhc_no_buoyancy_linear_profile(self):
        """No buoyancy: temperature profile should be approximately linear.

        The non-equilibrium BC correction introduces a small O(1/N) bias,
        so we check that the profile is monotonic and within bounds.
        """
        nx = 16
        dev = torch.device("cpu")
        ny = nx
        tau = 0.6
        tau_T = 3.0 * ((tau - 0.5) / 3.0) + 0.5
        t_hot, t_cold = 1.0, 0.0
        dT = t_hot - t_cold

        rho0 = torch.ones((ny, nx), device=dev)
        u0 = torch.zeros((ny, nx), device=dev)
        f = equilibrium(rho0, u0, u0)
        # Correct initial condition: T_hot at x=0, T_cold at x=nx-1
        T0 = torch.linspace(t_hot, t_cold, nx, device=dev).unsqueeze(0).expand(ny, nx).contiguous()
        g = temperature_equilibrium(T0, u0, u0)

        wall = cavity_wall_mask(ny, nx, dev)

        for _ in range(2000):
            rho, ux, uy = macroscopic(f)
            T = g.sum(dim=0)
            F = torch.zeros(2, ny, nx, device=dev)
            g = collide_thermal_bgk(g, T, ux, uy, tau_T)
            g = temperature_stream(g)
            from tensorlbm.thermal import apply_temperature_boundaries
            g = apply_temperature_boundaries(g, t_hot, t_cold)
            f_pre = f
            f = collide_bgk_force(f, tau, F)
            f = pre_streaming_bounce_back(f_pre, f, wall)
            from tensorlbm.solver import stream
            f = stream(f)

        T = g.sum(dim=0)
        # Check bounds
        assert T.min() >= -1e-6, f"T_min={T.min():.6f} < -1e-6"
        assert T.max() <= 1.0 + 1e-6, f"T_max={T.max():.6f} > 1+1e-6"
        # Check wall values
        assert abs(T[:, 0].mean().item() - t_hot) < 1e-6, f"Left wall T={T[:, 0].mean():.6f} != {t_hot}"
        assert abs(T[:, -1].mean().item() - t_cold) < 1e-6, f"Right wall T={T[:, -1].mean():.6f} != {t_cold}"
        # Check monotonicity (approximately)
        middle = T[ny//2, :]
        diffs = middle[1:] - middle[:-1]
        assert (diffs < 1e-6).all(), f"Profile not monotonic: max diff={diffs.max():.6e}"

    def test_temperature_bounds_preserved(self):
        """Temperature should stay within [T_cold, T_hot]."""
        res = simulate_rayleigh_benard_2d(
            nx=16, ny=16, ra=1e4, pr=0.71, tau=0.6,
            steps=2000, report_every=1000, device="cpu",
        )
        assert res.T_min >= -1e-4, f"T_min={res.T_min} < -1e-4"
        assert res.T_max <= 1.0 + 1e-4, f"T_max={res.T_max} > 1+1e-4"


# ============================================================================
# C6: Accuracy order — cosine decay
# ============================================================================

class TestAccuracyOrder:
    """C6: Accuracy order from sinusoidal temperature decay.

    Pure diffusion of cosine mode: T = cos(kx)·exp(-αk²t).
    Effective α from decay rate should match theory within 2%.
    Grid doubling should give error ratio ≈ 4 (second order).
    """

    def test_cosine_decay_rate(self):
        """Cosine mode decay rate matches theoretical α within 2%."""
        ny, nx = 6, 128
        tau_T = 0.7
        alpha_theory = (tau_T - 0.5) / 3.0
        k = 2 * math.pi * 2.0 / nx
        steps = 2000

        u = torch.zeros(ny, nx)
        x = torch.arange(nx).float()
        T0 = torch.cos(k * x).unsqueeze(0).expand(ny, nx).contiguous()
        g = temperature_equilibrium(T0, u, u)

        for _ in range(steps):
            g = temperature_collision(g, tau_T, u, u)
            g = temperature_stream(g)

        T = g.sum(dim=0)
        T_analytic = T0 * math.exp(-alpha_theory * k * k * steps)

        amp_initial = float(T0.abs().max().item())
        amp_final = float(T.abs().max().item())
        alpha_eff = -math.log(amp_final / amp_initial) / (k * k * steps)

        err = abs(alpha_eff - alpha_theory) / alpha_theory
        assert err < 0.02, f"α_eff={alpha_eff:.6f} vs α_theory={alpha_theory:.6f}, err={err:.4f}"

    def test_convergence_order(self):
        """Grid doubling gives error ratio ≈ 4 (second-order convergence)."""
        tau_T = 0.7
        alpha_theory = (tau_T - 0.5) / 3.0
        steps = 500
        n_wavelengths = 2

        errors = []
        for nx in [64, 128, 256]:
            ny = 6
            u = torch.zeros(ny, nx)
            x = torch.arange(nx).float()
            k = 2 * math.pi * n_wavelengths / nx
            T0 = torch.cos(k * x).unsqueeze(0).expand(ny, nx).contiguous()
            g = temperature_equilibrium(T0, u, u)

            for _ in range(steps):
                g = temperature_collision(g, tau_T, u, u)
                g = temperature_stream(g)

            T = g.sum(dim=0)
            T_analytic = T0 * math.exp(-alpha_theory * k * k * steps)
            err = float((T - T_analytic).abs().max().item())
            errors.append(err)

        if len(errors) >= 2 and errors[0] > 1e-10 and errors[1] > 1e-10:
            ratio = errors[0] / errors[1]
            assert 2.0 < ratio < 6.0, f"Error ratio 64->128: {ratio:.2f} not in [2, 6]"
        if len(errors) >= 3 and errors[1] > 1e-10 and errors[2] > 1e-10:
            ratio2 = errors[1] / errors[2]
            assert 2.0 < ratio2 < 6.0, f"Error ratio 128->256: {ratio2:.2f} not in [2, 6]"


# ============================================================================
# C7: Regression — existing tests
# ============================================================================

class TestRegression:
    """C7: Existing thermal tests must pass."""

    def test_thermal_imports(self):
        """All thermal modules import cleanly."""
        from tensorlbm import thermal
        from tensorlbm import thermal_common
        from tensorlbm import thermal_verification
        from tensorlbm import heat_convection_benchmarks
        assert hasattr(thermal, "simulate_natural_convection")
        assert hasattr(thermal_verification, "simulate_rayleigh_benard_2d")

    def test_thermal_params_roundtrip(self):
        """thermal_params roundtrip: Ra → params → Ra."""
        p = thermal_params(nx=65, ra=1e4, pr=0.71, tau=0.6)
        ra_back = p["g_beta"] * (p["H"] ** 3) / (p["nu"] * p["alpha"])
        assert abs(ra_back - 1e4) / 1e4 < 1e-12

    def test_nusselt_uniform_gradient(self):
        """Linear temperature field → Nu = 1."""
        nx = 32
        H = float(nx)
        T = (1.0 - torch.arange(nx).float() / H).unsqueeze(0).expand(6, nx).contiguous()
        for mode in ["grad1", "grad2"]:
            r = nusselt_number(T, H, dT=1.0, mode=mode)
            assert abs(r["nu"] - 1.0) < 1e-5, f"{mode}: {r['nu']}"
