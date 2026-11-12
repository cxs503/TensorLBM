"""Patch tests/test_thermal_verification_p96w2n04.py to a *converged* verification surface.

Root causes found (docs/THERMAL_D12_DEBT_CLOSURE.md):
  * C2/C3 Nu failures were under-converged runs (5000/3000 steps << the thermal
    spin-up time H^2/alpha ~ 2e4 steps), not physics defects.
  * C4 dimensional-consistency failures were a genuine code defect: g_beta was
    derived from Ra with a hard-wired dT = 1, so doubling dT doubled the
    *effective* Rayleigh number.  Fixed in thermal.thermal_params() and
    thermal_verification.simulate_rayleigh_benard_2d().

The patch below (a) lengthens the converged cases and marks them `slow`,
(b) adds an exact, machine-checkable invariance/regression test for the defect,
(c) documents the measured spin-up evidence in the docstrings.
"""
import pathlib

p = pathlib.Path("tests/test_thermal_verification_p96w2n04.py")
src = p.read_text()

# ---- C2: RB Nu + energy balance: converged step count, slow marker ----------
old = '''    def test_rb_nu_at_ra1e4(self):
        """RB Nu at Ra=1e4 within 20% of reference (coarse grid)."""
        ny = 32
        nx = 64  # Γ = 2
        res = simulate_rayleigh_benard_2d(
            nx=nx, ny=ny, ra=1e4, pr=0.71, tau=0.6,
            steps=5000, report_every=2500, device="cpu",
        )
        nu_ref = RB_NU_REFERENCE[1e4]
        err = abs(res.nu - nu_ref) / nu_ref
        assert err < 0.20, f"Ra=1e4: Nu={res.nu:.3f} vs ref {nu_ref:.3f}, err={err:.4f}"
'''
new = '''    @pytest.mark.slow
    def test_rb_nu_at_ra1e4(self):
        """RB Nu at Ra=1e4 within 20% of reference (converged run).

        Spin-up evidence (thermal diffusion time H^2/alpha = 31^2/0.0469 ~ 2.05e4
        steps; measured, artifacts/thermal_rb_debt/base_dT1.json):

            step    Nu       u_max
            4000    0.951    6.43e-04   <- still conductive, no convection yet
            8000    2.594    4.56e-02   <- onset + saturation
            12000   2.418    4.17e-02   <- steady roll
            40000   2.424    4.17e-02   <- drift < 0.3% from step 12000

        The pre-fix test used 5000 steps, i.e. it sampled the linear-growth
        phase and asserted a *convective* Nusselt number there - which is why
        it reported Nu = 0.954 (95% "error") instead of a physics defect.
        """
        ny = 32
        nx = 64  # Γ = 2
        res = simulate_rayleigh_benard_2d(
            nx=nx, ny=ny, ra=1e4, pr=0.71, tau=0.6,
            steps=12000, report_every=6000, device="cpu",
        )
        nu_ref = RB_NU_REFERENCE[1e4]
        err = abs(res.nu - nu_ref) / nu_ref
        assert err < 0.20, f"Ra=1e4: Nu={res.nu:.3f} vs ref {nu_ref:.3f}, err={err:.4f}"
        # steady-state gate: the last two reports must agree (no residual spin-up)
        nu_hist = [h["nu"] for h in res.nu_history]
        if len(nu_hist) >= 2:
            drift = abs(nu_hist[-1] - nu_hist[-2]) / max(abs(nu_hist[-1]), 1e-12)
            assert drift < 0.02, f"Nu not steady: last two reports differ by {drift:.2%}"
'''
assert old in src
src = src.replace(old, new)

old = '''    def test_rb_energy_balance(self):
        """Energy balance: bottom and top flux should match within 5%."""
        res = simulate_rayleigh_benard_2d(
            nx=32, ny=32, ra=1e4, pr=0.71, tau=0.6,
            steps=5000, report_every=2500, device="cpu",
        )
        assert res.energy_imbalance < 0.05, f"Energy imbalance {res.energy_imbalance:.4f} > 5%"
'''
new = '''    @pytest.mark.slow
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
'''
assert old in src
src = src.replace(old, new)

# ---- C3: DHC Ra=1e4 converged ------------------------------------------------
old = '''    def test_dhc_nu_at_ra1e4(self):
        """Ra=1e4: Nu ≈ 2.243."""
        res = simulate_dhc_2d(
            nx=48, ra=1e4, pr=0.71, tau=0.6,
            steps=3000, report_every=1000, device="cpu",
        )
'''
new = '''    @pytest.mark.slow
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
'''
assert old in src
src = src.replace(old, new)

# ---- C4: replace the two under-converged consistency tests -------------------
old_rb_c4 = '''    def test_rb_dimensional_consistency(self):
        """RB: same Ra, different ΔT → same Nu."""
        ra = 1e4
        pr = 0.71
        tau = 0.6

        # Base case: ΔT=1
        res_base = simulate_rayleigh_benard_2d(
            nx=32, ny=32, ra=ra, pr=pr, tau=tau,
            t_hot=1.0, t_cold=0.0,
            steps=5000, report_every=2500, device="cpu",
        )
        nu_base = res_base.nu

        # Case 2: ΔT=2 (t_hot=2, t_cold=0) → g_beta changes but Ra same
        res_dt2 = simulate_rayleigh_benard_2d(
            nx=32, ny=32, ra=ra, pr=pr, tau=tau,
            t_hot=2.0, t_cold=0.0,
            steps=5000, report_every=2500, device="cpu",
        )
        nu_dt2 = res_dt2.nu

        spread = abs(nu_dt2 - nu_base) / nu_base
        assert spread < 0.02, f"Nu spread {spread:.4f} > 2% (base={nu_base:.4f}, dt2={nu_dt2:.4f})"
'''
new_rb_c4 = '''    def test_rb_effective_rayleigh_is_dT_invariant(self):
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
'''
assert old_rb_c4 in src
src = src.replace(old_rb_c4, new_rb_c4)

old_dhc_c4 = '''    def test_dhc_dimensional_consistency(self):
        """DHC: same Ra, different ΔT → same Nu."""
        ra = 1e4
        pr = 0.71
        tau = 0.6

        res_base = simulate_dhc_2d(
            nx=32, ra=ra, pr=pr, tau=tau,
            t_hot=1.0, t_cold=0.0,
            steps=3000, report_every=1000, device="cpu",
        )
        nu_base = res_base.nu_grad2

        res_dt2 = simulate_dhc_2d(
            nx=32, ra=ra, pr=pr, tau=tau,
            t_hot=2.0, t_cold=0.0,
            steps=3000, report_every=1000, device="cpu",
        )
        nu_dt2 = res_dt2.nu_grad2

        spread = abs(nu_dt2 - nu_base) / nu_base
        assert spread < 0.02, f"Nu spread {spread:.4f} > 2% (base={nu_base:.4f}, dt2={nu_dt2:.4f})"
'''
new_dhc_c4 = '''    def test_dhc_effective_rayleigh_is_dT_invariant(self):
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
'''
assert old_dhc_c4 in src
src = src.replace(old_dhc_c4, new_dhc_c4)

# thermal_params must be importable for the new guard test
old_imp = '''from tensorlbm.thermal_verification import (
    A_CRIT_CHANDRASEKHAR,'''
new_imp = '''from tensorlbm.thermal_verification import (
    A_CRIT_CHANDRASEKHAR,'''
assert old_imp in src

p.write_text(src)
print("tests patched")