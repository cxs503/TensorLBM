"""Verification suite for the porous-absorber / impedance-boundary module.

Acceptance mapping (campaign node P87W2-N06)
--------------------------------------------
(a) TMM algebra vs closed forms / published exponents -> TestDelanyBazley, TestTransferMatrix
(b) absorption-curve shape and peak law                -> TestAbsorptionPhysics
(c) multilayer consistency                             -> TestTransferMatrix, TestMultilayerLining
(d) insertion loss                                     -> TestInsertionLoss
(e) smooth alpha(f)                                    -> TestAbsorptionPhysics
(f) negative controls (rigid wall / lossless / air gap) -> TestLimits
(g) energy bookkeeping                                 -> TestTransferMatrix, TestLBMImpedanceBoundary
(h) >= 10 pytest items                                 -> this file

Anchors used (no digitised experimental curve ships with the repository):
exact analytical relations, the published correlation constants and their
log-log exponents, and a 1-D lattice-Boltzmann experiment whose only shared
ingredient with the TMM is the target reflectance.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.acoustics_porous_absorber import (
    DB_COEFFICIENTS,
    DEFAULT_C0,
    DEFAULT_RHO0,
    DuctLBM,
    ImpedanceWall,
    Layer,
    Lining,
    absorption_coefficient,
    band_statistics,
    db_validity_ratio,
    delany_bazley,
    insertion_loss,
    layer_transfer_matrix,
    measure_reflection,
    optimize_lining,
    reflection_coefficient,
    surface_impedance,
)

RHO0, C0 = DEFAULT_RHO0, DEFAULT_C0
Z0 = RHO0 * C0


# ---------------------------------------------------------------------------
# (a) Delany-Bazley correlation
# ---------------------------------------------------------------------------
class TestDelanyBazley:
    def test_published_coefficients_at_x_equals_one(self):
        """At X = rho0 f / sigma = 1 the correlation reduces to the published
        constants: Z/rho c = 1.0571 + 0.0870 i, k/k0 = 1.0978 + 0.189 i."""
        sigma = 20000.0
        f = sigma / RHO0           # X = 1 exactly
        Zc, kc = delany_bazley(f, sigma)
        assert np.isclose(Zc, Z0 * (1.0571 + 0.0870j), rtol=0, atol=1e-9)
        k0 = 2.0 * np.pi * f / C0
        assert np.isclose(kc, k0 * (1.0978 + 0.1890j), rtol=0, atol=1e-9)
        assert np.isclose(float(db_validity_ratio(f, sigma)), 1.0)

    def test_published_exponents_are_recovered(self):
        """Log-log slopes reproduce the published exponents (-0.754 / -0.732 for
        the impedance, -0.700 / -0.595 for the wavenumber); this catches any
        exponent or sign typo in the constant table."""
        X = np.geomspace(0.02, 1.0, 60)
        sigma = 20000.0
        f = X * sigma / RHO0
        Zc, kc = delany_bazley(f, sigma)
        z0 = Z0

        def slope(y):
            return np.polyfit(np.log(X), np.log(y), 1)[0]

        assert np.isclose(slope(Zc.real - z0), DB_COEFFICIENTS["zr_exp"], atol=1e-6)
        assert np.isclose(slope(Zc.imag), DB_COEFFICIENTS["zi_exp"], atol=1e-6)
        krel = kc / (2.0 * np.pi * f / C0)
        assert np.isclose(slope(krel.real - 1.0), DB_COEFFICIENTS["kr_exp"], atol=1e-6)
        assert np.isclose(slope(krel.imag), DB_COEFFICIENTS["ki_exp"], atol=1e-6)

    def test_physical_branches(self):
        """Attenuation along +x needs Im(k) > 0 and a mass-like reactance needs
        Im(Z) > 0 in the exp(-i omega t) convention used throughout."""
        Zc, kc = delany_bazley(np.geomspace(100.0, 5000.0, 40), 20000.0)
        assert np.all(kc.imag > 0.0)
        assert np.all(Zc.imag > 0.0)
        assert np.all(Zc.real > Z0)      # resistance above the free field

    def test_conjugate_convention_gives_same_absorption(self):
        """The tabulated ``-j`` form belongs to the opposite time convention.
        Conjugating Z_c and k_c *and* flipping the ``i`` of the layer matrix
        must give the same physical absorption."""
        f = np.geomspace(100.0, 5000.0, 50)
        d, sigma = 0.05, 20000.0
        Zc, kc = delany_bazley(f, sigma)
        z_std = surface_impedance([Layer(d, sigma)], f)
        Zcc, kcc = np.conj(Zc), np.conj(kc)
        kd = kcc * d
        T = np.empty(kd.shape + (2, 2), dtype=complex)
        T[..., 0, 0] = np.cos(kd)
        T[..., 0, 1] = 1j * Zcc * np.sin(kd)
        T[..., 1, 0] = 1j * np.sin(kd) / Zcc
        T[..., 1, 1] = np.cos(kd)
        z_conj = T[..., 0, 0] / T[..., 1, 0]
        a1 = absorption_coefficient(reflection_coefficient(z_std))
        a2 = absorption_coefficient(reflection_coefficient(z_conj))
        assert np.allclose(a1, a2, atol=1e-12)
        zc_tab, _ = delany_bazley(f, sigma, conjugate=True)
        assert np.allclose(zc_tab, np.conj(Zc), atol=1e-12)

    def test_validity_ratio_window(self):
        """The correlation is fitted for 0.01 < X < 1; the helper reports X so
        designs outside the window can be flagged."""
        assert np.isclose(float(db_validity_ratio(20000.0 * 0.5 / RHO0, 20000.0)), 0.5)
        assert float(db_validity_ratio(50.0, 20000.0)) < 0.01


# ---------------------------------------------------------------------------
# (a), (c), (g) transfer-matrix algebra
# ---------------------------------------------------------------------------
class TestTransferMatrix:
    def test_single_layer_closed_form(self):
        """Hard-backed single layer: TMM surface impedance equals the analytic
        ``i Z_c cot(k_c d)`` to machine precision."""
        f = np.geomspace(80.0, 6000.0, 200)
        d, sigma = 0.05, 20000.0
        z_tmm = surface_impedance([Layer(d, sigma)], f)
        Zc, kc = delany_bazley(f, sigma)
        z_ref = 1j * Zc / np.tan(kc * d)
        assert np.max(np.abs((z_tmm - z_ref) / z_ref)) < 1e-12

    def test_piston_matrix_structure(self):
        """Each layer matrix has unit determinant and A = D (reciprocity)."""
        f = np.array([200.0, 1000.0, 4000.0])
        T = layer_transfer_matrix(Layer(0.04, 12000.0), f)
        assert np.allclose(np.linalg.det(T), 1.0, atol=1e-12)
        assert np.allclose(T[..., 0, 0], T[..., 1, 1], atol=1e-12)

    def test_layer_splitting_invariance(self):
        """Splitting a homogeneous layer into two sub-layers must not change the
        surface impedance (exact multiplicative property of the TMM)."""
        f = np.geomspace(100.0, 5000.0, 60)
        single = surface_impedance([Layer(0.05, 20000.0)], f)
        split = surface_impedance([Layer(0.02, 20000.0), Layer(0.03, 20000.0)], f)
        assert np.max(np.abs((single - split) / single)) < 1e-12

    def test_anechoic_backing_matches_analytic_single_layer(self):
        """Matched backing: ``Z = Zc (Zb - i Zc tan)/(Zc - i Zb tan)`` is an
        independent closed form of the same transfer matrix."""
        f = np.geomspace(100.0, 5000.0, 80)
        d, sigma = 0.05, 20000.0
        Zc, kc = delany_bazley(f, sigma)
        tan = np.tan(kc * d)
        z_ref = Zc * (Z0 - 1j * Zc * tan) / (Zc - 1j * Z0 * tan)
        z_tmm = surface_impedance([Layer(d, sigma)], f, backing="anechoic")
        assert np.max(np.abs((z_tmm - z_ref) / z_ref)) < 1e-10

    def test_explicit_complex_backing(self):
        """An arbitrary complex backing impedance must be honoured."""
        f = np.array([500.0])
        z_back = 3.0 * Z0
        z_tmm = surface_impedance([Layer(0.05, 20000.0)], f, backing=z_back)
        T = layer_transfer_matrix(Layer(0.05, 20000.0), f)
        z_ref = (T[0, 0, 0] * z_back + T[0, 0, 1]) / (T[0, 1, 0] * z_back + T[0, 1, 1])
        assert np.isclose(z_tmm[0], z_ref)

    def test_energy_conservation(self):
        """Passive wall: alpha + |R|^2 = 1 exactly (to round-off)."""
        f = np.geomspace(100.0, 5000.0, 100)
        for sigma in (5000.0, 20000.0, 50000.0):
            R = Lining([Layer(0.05, sigma)]).reflection(f)
            alpha = absorption_coefficient(R)
            assert np.max(np.abs(alpha + np.abs(R) ** 2 - 1.0)) < 1e-12


# ---------------------------------------------------------------------------
# (b), (e) absorption physics
# ---------------------------------------------------------------------------
class TestAbsorptionPhysics:
    def test_passivity_bounds(self):
        """|R| <= 1 and 0 <= alpha <= 1 over the design space.

        The Delany-Bazley pair (Z_c, k_c) is not a thermodynamic pair, so the
        correlation can create up to ~0.5 % of energy at the low-frequency edge
        of its validity window; the bound below documents (and pins) that.
        """
        f = np.geomspace(50.0, 6000.0, 120)
        worst = 0.0
        for d in (0.02, 0.04, 0.08, 0.1):
            for sigma in np.geomspace(5000.0, 50000.0, 6):
                R = Lining([Layer(d, sigma)]).reflection(f)
                alpha = absorption_coefficient(R)
                worst = max(worst, float(np.abs(R).max()))
                assert np.all(alpha >= -1e-15) and np.all(alpha <= 1.0)
        assert worst < 1.01

    def test_correlation_passivity_violation_is_small_and_bounded(self):
        """Honest bookkeeping: the DB correlation is not strictly passive.
        Quantify the worst-case energy creation and pin it under 2 %."""
        f = np.geomspace(20.0, 6000.0, 400)
        R = Lining([Layer(0.05, 20000.0)]).reflection(f)
        excess = float(np.abs(R).max() ** 2 - 1.0)
        assert 0.0 < excess < 0.02      # measured 1.1 % at 35 Hz (see report)
        # ... and it lives at the low-frequency edge, outside the fitted window
        assert float(db_validity_ratio(f[int(np.argmax(np.abs(R)))], 20000.0)) < 0.01

    def test_absorption_curve_is_smooth(self):
        """No numerical ripple: the discrete second difference of alpha on a
        10 Hz grid stays tiny, and adjacent samples never jump."""
        f = np.arange(100.0, 5000.0 + 1.0, 10.0)
        alpha = Lining([Layer(0.05, 20000.0)]).absorption(f)
        assert np.max(np.abs(np.diff(alpha))) < 0.02
        assert np.max(np.abs(np.diff(alpha, 2))) < 1e-3
        # a single hard-backed layer has at most a handful of extrema
        n_extrema = int(np.sum(np.diff(np.sign(np.diff(alpha))) != 0))
        assert n_extrema <= 8

    def test_first_resonance_scales_with_thickness(self):
        """First absorption maximum obeys the quarter-wave scaling: f_first * d
        is constant (within 10 %) for layers of different thickness."""
        products = []
        for d in (0.04, 0.06, 0.08):
            f = np.arange(100.0, 3000.0, 2.0)
            alpha = Lining([Layer(d, 20000.0)]).absorption(f)
            da = np.diff(alpha)
            idx = np.where((da[:-1] > 0) & (da[1:] < 0))[0]
            assert idx.size >= 1, "no interior absorption peak found"
            f_first = f[idx[0] + 1]
            products.append(f_first * d)
            # the resonance is a genuine interior maximum
            assert alpha[idx[0] + 1] > 0.8
        products = np.asarray(products)
        assert np.max(np.abs(products - products.mean())) / products.mean() < 0.10

    def test_peak_frequency_decreases_with_thickness(self):
        peaks = []
        for d in (0.04, 0.06, 0.08):
            f = np.arange(100.0, 3000.0, 2.0)
            alpha = Lining([Layer(d, 20000.0)]).absorption(f)
            da = np.diff(alpha)
            idx = np.where((da[:-1] > 0) & (da[1:] < 0))[0]
            peaks.append(f[idx[0] + 1])
        assert peaks[0] > peaks[1] > peaks[2]

    def test_band_statistics(self):
        f = np.arange(100.0, 5000.0, 10.0)
        alpha = Lining([Layer(0.05, 20000.0)]).absorption(f)
        stats = band_statistics(f, alpha, (250.0, 2000.0))
        assert stats["f_lo"] == 250.0 and stats["f_hi"] == 2000.0
        assert 0.0 < stats["alpha_mean"] < 1.0
        assert stats["alpha_min"] <= stats["alpha_peak"] <= stats["alpha_max"]


# ---------------------------------------------------------------------------
# (f) limits / negative controls
# ---------------------------------------------------------------------------
class TestLimits:
    def test_rigid_wall_limit(self):
        """sigma -> infinity is a rigid wall: alpha -> 0."""
        f = np.array([200.0, 1000.0, 3000.0])
        alpha = Lining([Layer(0.05, 1.0e8)]).absorption(f)
        assert np.all(alpha < 0.02)

    def test_lossless_limit(self):
        """sigma -> 0 makes the medium lossless and impedance-matched
        (Zc -> rho0 c0, kc -> k0): the hard-backed layer is a pure reactance and
        alpha -> 0."""
        f = np.array([200.0, 1000.0, 3000.0])
        alpha = Lining([Layer(0.05, 1.0)]).absorption(f)
        assert np.all(alpha < 0.02)
        Zc, kc = delany_bazley(f, 1.0)
        assert np.all(np.abs(Zc / Z0 - 1.0) < 5e-3)

    def test_air_layer_has_no_absorption(self):
        """A lossless cavity in front of a rigid wall stores energy only."""
        f = np.array([200.0, 1000.0, 3000.0])
        alpha = Lining([Layer(0.10, model="air")]).absorption(f)
        assert np.all(alpha < 1e-12)

    def test_thin_layer_low_frequency_limit(self):
        """d -> 0 with a rigid backing is transparent at low frequency."""
        alpha = Lining([Layer(1.0e-4, 20000.0)]).absorption(np.array([100.0]))
        assert alpha[0] < 1e-3


# ---------------------------------------------------------------------------
# (d) insertion loss
# ---------------------------------------------------------------------------
class TestInsertionLoss:
    def test_insertion_loss_formula(self):
        alpha = np.array([0.0, 0.5, 0.9, 0.99])
        il = insertion_loss(alpha)
        assert np.allclose(il, -10.0 * np.log10(1.0 - alpha))
        assert np.allclose(il[1], 10.0 * np.log10(2.0))
        assert np.allclose(il[3], 20.0)

    def test_band_averaged_insertion_loss_grows_with_thickness(self):
        """Band-averaged IL is monotone in thickness (pointwise IL is not, since
        interference peaks move)."""
        f = np.geomspace(250.0, 2000.0, 60)
        ils = [insertion_loss(Lining([Layer(d, 20000.0)]).absorption(f)).mean() for d in (0.02, 0.05, 0.1)]
        assert ils[0] < ils[1] < ils[2]
        assert np.all(np.isfinite(ils))


# ---------------------------------------------------------------------------
# (c) multilayer lining
# ---------------------------------------------------------------------------
class TestMultilayerLining:
    def test_two_layer_surface_impedance_recursion(self):
        """Two-layer stack vs the impedance-transformation recursion."""
        f = np.geomspace(100.0, 5000.0, 40)
        l1, l2 = Layer(0.03, 10000.0), Layer(0.02, 40000.0)
        z_tmm = surface_impedance([l1, l2], f)
        Zc1, kc1 = delany_bazley(f, l1.sigma)
        Zc2, kc2 = delany_bazley(f, l2.sigma)
        z2 = 1j * Zc2 / np.tan(kc2 * l2.thickness)     # layer 2 on a rigid wall
        t1 = np.tan(kc1 * l1.thickness)
        z_ref = Zc1 * (z2 - 1j * Zc1 * t1) / (Zc1 - 1j * z2 * t1)
        assert np.max(np.abs((z_tmm - z_ref) / z_ref)) < 1e-10

    def test_cavity_resonance_adds_structure(self):
        """Porous facing + air cavity + porous base shows cavity resonances, so
        the alpha curve has more extrema than a single homogeneous layer."""
        f = np.arange(100.0, 3000.0, 2.0)
        single = Lining([Layer(0.05, 20000.0)]).absorption(f)
        stacked = Lining(
            [Layer(0.02, 8000.0), Layer(0.05, model="air"), Layer(0.02, 30000.0)]
        ).absorption(f)

        def n_extrema(a):
            da = np.diff(a)
            return int(np.sum((da[:-1] > 0) & (da[1:] < 0)))

        assert n_extrema(stacked) > n_extrema(single)
        assert np.all(stacked <= 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# engineering optimisation under a mass constraint
# ---------------------------------------------------------------------------
class TestOptimisation:
    def test_mass_constraint_respected(self):
        report = optimize_lining(
            (250.0, 2000.0), d_max=0.10, mass_limit=1.0, n_d=9, n_sigma=7, n_freq=15
        )
        assert report["best"]["areal_mass"] <= 1.0 + 1e-12
        assert 0.0 < report["best"]["alpha_mean"] <= 1.0
        assert report["thinnest_near_optimal"]["areal_mass"] <= 1.0 + 1e-12
        assert (
            report["thinnest_near_optimal"]["alpha_mean"]
            <= report["best"]["alpha_mean"] + 1e-12
        )

    def test_optimum_is_interior_in_sigma(self):
        """Too low and too high a flow resistivity both hurt."""
        report = optimize_lining((250.0, 2000.0), d_max=0.10, n_d=7, n_sigma=9, n_freq=15)
        sigmas = np.asarray(report["sigma_grid"])
        assert sigmas[0] < report["best"]["sigma"] < sigmas[-1]


# ---------------------------------------------------------------------------
# (g) LBM impedance boundary: mechanism, energy budget, TMM agreement
# ---------------------------------------------------------------------------
class TestLBMImpedanceBoundary:
    def setup_method(self):
        self.duct_kwargs = dict(n=300, length=0.6, settle_periods=8.0)

    @pytest.mark.physics
    @pytest.mark.integration
    def test_duct_is_stable_and_quiet(self):
        duct = DuctLBM(n=200, length=0.4)
        wall = ImpedanceWall(0.5 + 0.3j, 500.0, duct.dt)
        for step in range(2000):
            duct.add_soft_source(10, 1e-4 * np.sin(2 * np.pi * 500.0 * step * duct.dt))
            duct.step(wall)
        assert np.all(np.isfinite(duct.f))
        assert np.max(np.abs(duct.pressure())) < 1e-2

    @pytest.mark.physics
    @pytest.mark.integration
    def test_prescribed_reflectance_is_reproduced(self):
        """The lattice-Boltzmann wall must realise the prescribed complex R."""
        for R_in, freq in ((1.0 + 0.0j, 300.0), (0.5 + 0.0j, 500.0), (0.6j, 500.0)):
            res = measure_reflection(R_in, freq, **self.duct_kwargs)
            assert abs(res["alpha_lbm"] - (1.0 - abs(R_in) ** 2)) < 0.01
            assert abs(res["R_lbm"] - R_in) < 0.06

    @pytest.mark.physics
    @pytest.mark.integration
    def test_energy_budget_at_the_wall(self):
        """p'u' measured at the wall node must equal the absorbed intensity."""
        res = measure_reflection(0.6 + 0.0j, 500.0, **self.duct_kwargs)
        assert abs(res["alpha_energy"] - res["alpha_lbm"]) < 0.02
        assert res["alpha_energy"] > 0.0

    @pytest.mark.physics
    @pytest.mark.integration
    def test_lbm_agrees_with_tmm_single_layer(self):
        """Independent numerical path (time-domain LBM) vs TMM for a DB lining."""
        lining = Lining([Layer(0.05, 20000.0)])
        for freq in (250.0, 800.0, 2000.0):
            res = measure_reflection(lining, freq, **self.duct_kwargs)
            assert abs(res["alpha_lbm"] - res["alpha_tmm"]) < 0.03
            assert abs(res["R_lbm"] - res["R_tmm"]) < 0.06

    @pytest.mark.physics
    @pytest.mark.integration
    def test_lbm_agrees_with_tmm_two_layer(self):
        lining = Lining([Layer(0.02, 8000.0), Layer(0.03, 30000.0)])
        for freq in (400.0, 1200.0):
            res = measure_reflection(lining, freq, **self.duct_kwargs)
            assert abs(res["alpha_lbm"] - res["alpha_tmm"]) < 0.03

    @pytest.mark.physics
    def test_wall_requires_passive_reflectance(self):
        with pytest.raises(ValueError):
            ImpedanceWall(1.5 + 0.0j, 500.0, 1e-5)

    @pytest.mark.physics
    @pytest.mark.integration
    def test_reflection_free_wall_is_anechoic(self):
        """R = 0 wall absorbs everything."""
        res = measure_reflection(0.0 + 0.0j, 1000.0, **self.duct_kwargs)
        assert abs(res["R_lbm"]) < 0.02
        assert res["alpha_lbm"] > 0.99