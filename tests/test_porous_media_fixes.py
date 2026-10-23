"""Discriminating tests for porous_media mechanical defect fixes (W4-D).

Each test targets one defect from the W4-B verification campaign defect
list (``runs/bm_widen_w4_20260920/washburn_twophase/NOTES.md`` §2 and
amendment #2 §9.2).  Fixes are mechanical only; the deep-physics items are
represented by ``xfail(strict=True)`` reproducers at the bottom so the
suite stays green while preserving the evidence and alerting on any future
change of behaviour.

No physics kernels are defined here: collide/stream/equilibrium/
bounce-back calls all come from the library; tests add configuration,
geometry, initialisation and measurement only.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tensorlbm.porous_media import (
    CapillaryInvasionConfig,
    LaplaceTestConfig,
    TwoPhasePoiseuilleConfig,
    _capillary_mask,
    _estimate_washburn_exponent,
    _full_swap_fields,
    _two_phase_poiseuille_analytical,
    run_capillary_invasion,
    run_laplace_test,
    run_two_phase_poiseuille,
)

CS2 = 1.0 / 3.0


def _synthetic_series(beta: float, c: float = 7.0) -> list[tuple[int, float, float]]:
    """Series (step, front, sqrt_t) with front = c * t**beta, front > 2."""
    steps = [200, 400, 800, 1600, 3200, 6400]
    return [(t, c * t**beta, math.sqrt(t)) for t in steps]


# ---------------------------------------------------------------------------
# Fix 2: _estimate_washburn_exponent must return β (slope vs log t), not 2β
# ---------------------------------------------------------------------------


class TestWashburnExponent:
    def test_beta_half_synthetic(self) -> None:
        assert _estimate_washburn_exponent(_synthetic_series(0.5)) == pytest.approx(0.5, abs=1e-6)

    def test_beta_sixty_synthetic(self) -> None:
        assert _estimate_washburn_exponent(_synthetic_series(0.6)) == pytest.approx(0.6, abs=1e-6)

    def test_returns_beta_not_two_beta(self) -> None:
        """L ∝ t^0.25 must give 0.25; the legacy log(√t) regression gave 0.5."""
        assert _estimate_washburn_exponent(_synthetic_series(0.25)) == pytest.approx(0.25, abs=1e-6)

    def test_too_few_valid_points_is_nan(self) -> None:
        series = [(1, 0.0, 1.0), (2, 1.0, 1.4), (3, 0.5, 1.7)]
        assert math.isnan(_estimate_washburn_exponent(series))


# ---------------------------------------------------------------------------
# Fix 1: G_12 sign convention — negative separates, positive mixes
# ---------------------------------------------------------------------------


class TestG12SignContract:
    def test_segregation_coupling_accepted(self) -> None:
        LaplaceTestConfig(G_12=-2.5).validate()  # must not raise

    def test_mixing_coupling_accepted(self) -> None:
        LaplaceTestConfig(G_12=0.9).validate()  # must not raise

    def test_zero_coupling_rejected(self) -> None:
        with pytest.raises(ValueError, match="G_12"):
            LaplaceTestConfig(G_12=0.0).validate()


# ---------------------------------------------------------------------------
# Fix 5 (unit level): full-swap initialisation has uniform total density
# ---------------------------------------------------------------------------


class TestFullSwapInit:
    def test_total_density_uniform(self) -> None:
        alpha = torch.tensor([[0.0, 0.2, 0.8, 1.0]])
        rho1, rho2 = _full_swap_fields(alpha, 0.7, 0.3)
        tot = rho1 + rho2
        assert torch.allclose(tot, torch.full_like(tot, 1.0))
        assert rho1.shape == alpha.shape

    def test_phase_compositions(self) -> None:
        alpha = torch.tensor([[0.0, 1.0]])
        rho1, rho2 = _full_swap_fields(alpha, 0.7, 0.3)
        # alpha=0: component 1 majority (0.7, 0.3); alpha=1: swapped.
        assert rho1[0, 0] == pytest.approx(0.7)
        assert rho2[0, 0] == pytest.approx(0.3)
        assert rho1[0, 1] == pytest.approx(0.3)
        assert rho2[0, 1] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Fix 3: dp_inlet is load-bearing (constant-pressure reservoir at x=0)
# ---------------------------------------------------------------------------


class TestDpInlet:
    def _run(self, tmp_path: Path, dp: float) -> dict[str, object]:
        cfg = CapillaryInvasionConfig(
            nx=100,
            ny=14,
            tube_width=8,
            G_12=-2.5,
            G_ads_water=0.0,
            dp_inlet=dp,
            n_steps=200,
            output_interval=200,
            output_root=tmp_path,
            overwrite=True,
        )
        result = run_capillary_invasion(cfg)
        fs = result["final_state"]
        assert fs["finite"] is True
        return result

    def test_inlet_pressure_tracks_dp_inlet(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, 2e-3)
        measured = result["final_state"]["p_inlet_minus_coexistence_ref"]
        # imposed ΔP = cs²·ρ_tot·dρ = dp_inlet exactly (dρ from the quadratic)
        assert measured == pytest.approx(2e-3, abs=1e-4)
        # dρ = (sqrt(1 + 4·dp/(cs²·ρ₀)) − 1)/2 with ρ₀ = 1.0, cs² = 1/3
        assert result["drho_inlet"] == pytest.approx((math.sqrt(1.024) - 1.0) / 2.0, abs=1e-6)

    def test_zero_dp_inlet_gives_reference_pressure(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, 0.0)
        measured = result["final_state"]["p_inlet_minus_coexistence_ref"]
        assert abs(measured) < 1e-6


# ---------------------------------------------------------------------------
# Fix 6: solid seam buries the periodic streaming seam
# ---------------------------------------------------------------------------


class TestSolidSeam:
    def test_seam_makes_last_column_solid(self) -> None:
        cfg = CapillaryInvasionConfig(solid_seam=True)
        solid = _capillary_mask(cfg, torch.device("cpu"))
        assert solid[:, -1].all()

    def test_legacy_flag_keeps_last_column_fluid_in_tube(self) -> None:
        cfg = CapillaryInvasionConfig(solid_seam=False)
        solid = _capillary_mask(cfg, torch.device("cpu"))
        # the tube interior of the last column stays fluid (legacy geometry)
        assert not solid[cfg.ny // 2, -1]

    def test_default_is_solid_seam(self) -> None:
        assert CapillaryInvasionConfig().solid_seam is True

    def test_run_name_flags_legacy_geometry(self) -> None:
        assert "noseam" not in CapillaryInvasionConfig().resolved_run_name()
        assert "noseam" in CapillaryInvasionConfig(solid_seam=False).resolved_run_name()


# ---------------------------------------------------------------------------
# Fix 5: default configurations survive ≥10k steps finite (mechanical gate)
# ---------------------------------------------------------------------------


class TestDefaultConfigsLongRunFinite:
    """Default physics parameters, 10k steps: finite, min f > 0, bounded.

    The legacy 5 %-minority step initialisation NaN'd these configurations
    within ~20 steps (laplace step 18, poiseuille step 6, invasion step 10).
    Interface *survival* under the default G_12 = +0.9 (a mixing point in
    this library's sign convention) is asserted separately at the
    segregation coupling in TestInterfaceSurvival.
    """

    @pytest.mark.slow
    def test_laplace_default_10k_finite(self, tmp_path: Path) -> None:
        cfg = LaplaceTestConfig(
            n_steps=10000, output_interval=1000, output_root=tmp_path, overwrite=True
        )
        result = run_laplace_test(cfg)
        fs = result["final_state"]
        assert result["nonfinite_step"] is None
        assert fs["finite"] is True
        assert fs["min_f_water"] > 0.0
        assert fs["min_f_gas"] > 0.0
        assert 0.1 < fs["rho_total_min"] <= fs["rho_total_max"] < 3.0
        assert math.isfinite(result["final_delta_p"])

    @pytest.mark.slow
    def test_two_phase_poiseuille_default_10k_finite(self, tmp_path: Path) -> None:
        cfg = TwoPhasePoiseuilleConfig(
            n_steps=10000, output_interval=1000, output_root=tmp_path, overwrite=True
        )
        result = run_two_phase_poiseuille(cfg)
        fs = result["final_state"]
        assert result["nonfinite_step"] is None
        assert fs["finite"] is True
        assert fs["min_f_water"] > 0.0
        assert fs["min_f_gas"] > 0.0
        assert 0.1 < fs["rho_total_min"] <= fs["rho_total_max"] < 3.0
        assert math.isfinite(result["l2_error_rel"])

    @pytest.mark.slow
    def test_capillary_invasion_default_10k_finite(self, tmp_path: Path) -> None:
        cfg = CapillaryInvasionConfig(
            n_steps=10000, output_interval=1000, output_root=tmp_path, overwrite=True
        )
        result = run_capillary_invasion(cfg)
        fs = result["final_state"]
        assert result["nonfinite_step"] is None
        assert fs["finite"] is True
        assert fs["min_f_water"] > 0.0
        assert fs["min_f_gas"] > 0.0
        assert 0.1 < fs["rho_total_min"] <= fs["rho_total_max"] < 3.0


class TestInterfaceSurvival:
    """At the segregation coupling G_12 = −2.5 the interface must survive.

    (Under the default G_12 = +0.5…+0.9 the coupling is attractive/mixing in
    this library's sign convention, so interfaces legitimately diffuse away;
    survival is asserted at the verified segregation point instead.)
    """

    def test_laplace_bubble_survives_at_segregation(self, tmp_path: Path) -> None:
        cfg = LaplaceTestConfig(
            nx=64,
            ny=64,
            bubble_radius=14.0,
            G_12=-2.5,
            n_steps=3000,
            output_interval=1000,
            output_root=tmp_path,
            overwrite=True,
        )
        result = run_laplace_test(cfg)
        fs = result["final_state"]
        assert fs["finite"] is True
        assert fs["min_f_water"] > 0.0
        # |contrast| ≈ 0.71 measured; sign depends on which phase ends up
        # inside under the symmetric full-swap initialisation (registered
        # observation, see W4-D NOTES) — only the magnitude is load-bearing.
        assert abs(fs["phi_contrast_inside_minus_outside"]) > 0.5

    def test_poiseuille_layers_survive_at_segregation(self, tmp_path: Path) -> None:
        cfg = TwoPhasePoiseuilleConfig(
            G_12=-2.5, n_steps=3000, output_interval=1000, output_root=tmp_path, overwrite=True
        )
        result = run_two_phase_poiseuille(cfg)
        fs = result["final_state"]
        assert fs["finite"] is True
        assert fs["min_f_water"] > 0.0
        assert abs(fs["phi_contrast_upper_minus_lower"]) > 0.5


# ---------------------------------------------------------------------------
# Fix 4: mu_ratio is load-bearing in the analytical Poiseuille profile
# ---------------------------------------------------------------------------


def _legacy_analytical_profile(ny: int, half: int, g_x: float, nu_w: float, nu_g: float):
    """Pre-fix implementation (stress row built from ν_σ, mu_ratio unused).

    Kept verbatim here as the regression oracle: the fixed function with
    mu_ratio = ν_w/ν_g must reproduce it exactly (equal-density layers).
    """
    big_h, h = float(ny - 1), float(half)
    aw = -g_x / (2.0 * nu_w)
    ag = -g_x / (2.0 * nu_g)
    a00, a01, a10, a11 = nu_w, -nu_g, h, big_h - h
    b0 = nu_g * 2.0 * ag * h - nu_w * 2.0 * aw * h
    b1 = ag * (h - big_h) * (h + big_h) - aw * h * h
    det = a00 * a11 - a01 * a10
    bw = (b0 * a11 - b1 * a01) / det
    bg = (a00 * b1 - a10 * b0) / det
    cg = -ag * big_h * big_h - bg * big_h
    return [
        max((aw * y * y + bw * y) if j <= half else (ag * y * y + bg * y + cg), 0.0)
        for j, y in enumerate(range(ny))
    ]


class TestMuRatioAnalytical:
    NY, HALF, GX = 30, 15, 5e-5

    def test_equivalent_to_legacy_for_kinematic_ratio(self) -> None:
        nu_w, nu_g = 1 / 6, 1 / 12
        new = _two_phase_poiseuille_analytical(
            self.NY, self.HALF, self.GX, nu_w, nu_g, mu_ratio=nu_w / nu_g
        )
        legacy = _legacy_analytical_profile(self.NY, self.HALF, self.GX, nu_w, nu_g)
        assert new == pytest.approx(legacy, rel=1e-12, abs=1e-15)

    def test_mu_ratio_changes_profile(self) -> None:
        nu_w, nu_g = 1 / 6, 1 / 12
        base = _two_phase_poiseuille_analytical(
            self.NY, self.HALF, self.GX, nu_w, nu_g, mu_ratio=nu_w / nu_g
        )
        stiff_gas = _two_phase_poiseuille_analytical(
            self.NY, self.HALF, self.GX, nu_w, nu_g, mu_ratio=4.0 * nu_w / nu_g
        )
        assert max(abs(a - b) for a, b in zip(base, stiff_gas, strict=True)) > 1e-9

    def test_stress_continuity_holds(self) -> None:
        nu_w, nu_g = 1 / 6, 1 / 12
        mu_ratio = nu_w / nu_g
        p = _two_phase_poiseuille_analytical(self.NY, self.HALF, self.GX, nu_w, nu_g, mu_ratio)
        h = self.HALF
        # second-order one-sided derivatives of a quadratic (exact)
        du_w = (3 * p[h] - 4 * p[h - 1] + p[h - 2]) / 2.0
        du_g = (-3 * p[h] + 4 * p[h + 1] - p[h + 2]) / 2.0
        assert mu_ratio * du_w == pytest.approx(du_g, rel=1e-9)

    def test_no_slip_walls(self) -> None:
        p = _two_phase_poiseuille_analytical(
            self.NY, self.HALF, self.GX, 1 / 6, 1 / 12, mu_ratio=2.0
        )
        assert p[0] == pytest.approx(0.0, abs=1e-14)
        assert p[-1] == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
# Fix 6 (behaviour): seam stabilises the segregation point
# ---------------------------------------------------------------------------


class TestSolidSeamStability:
    _CFG_KW = dict(
        nx=120,
        ny=14,
        tube_width=8,
        G_12=-2.5,
        G_ads_water=0.0,
        n_steps=6000,
        output_interval=2000,
        overwrite=True,
    )

    def test_segregation_point_stays_bounded_with_seam(self, tmp_path: Path) -> None:
        """With the seam solid, G_12 = −2.5 stays finite/bounded (6k steps)."""
        cfg = CapillaryInvasionConfig(solid_seam=True, output_root=tmp_path, **self._CFG_KW)
        result = run_capillary_invasion(cfg)
        fs = result["final_state"]
        assert result["nonfinite_step"] is None
        assert fs["finite"] is True
        assert fs["min_f_water"] > 1e-4
        assert fs["min_f_gas"] > 1e-4
        assert fs["rho_total_max"] < 1.8

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Defect 9 reproducer (legacy geometry, kept reachable via "
            "solid_seam=False): the periodic seam makes column 0 adjacent to "
            "column nx−1, so the inlet gas reservoir faces a virtual "
            "gas-water interface across the seam — densities diverge "
            "monotonically and the run goes non-finite (measured: ρ_tot max "
            "3.4 at 6k steps, NaN ≤ 6k; W4-B amendment #2 §9.2).  Expected "
            "to xpass only if the seam interaction is fixed at the physics "
            "level, which would obsolete the seam workaround."
        ),
    )
    def test_legacy_geometry_survives_segregation_point(self, tmp_path: Path) -> None:
        cfg = CapillaryInvasionConfig(solid_seam=False, output_root=tmp_path, **self._CFG_KW)
        result = run_capillary_invasion(cfg)
        assert result["nonfinite_step"] is None
        assert result["final_state"]["finite"] is True


# ---------------------------------------------------------------------------
# Deep-physics items (registered, NOT fixed): xfail(strict) reproducers
# ---------------------------------------------------------------------------


def _closed_box_phi_contrast(g_ads_water: float, steps: int = 2000) -> float:
    """Closed-box wettability protocol drift measurement (library kernels).

    Replicates the invasion/drainage wall protocol: adsorption
    pseudo-densities at solid nodes rewritten into equilibrium every step.
    Returns the interior gas-fraction contrast (upper − lower halves).
    """
    from tensorlbm.boundaries import bounce_back_cells
    from tensorlbm.d2q9 import equilibrium, macroscopic
    from tensorlbm.multiphase import collide_sc_two_component
    from tensorlbm.porous_media import apply_wall_wettability_sc
    from tensorlbm.solver import stream

    ny, nx = 24, 40
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    ys = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, nx)
    alpha = 0.5 * (1.0 - torch.tanh((ys - ny / 2.0) / 3.0))  # water below
    rho_w, rho_g = _full_swap_fields(alpha, 0.7, 0.3)
    zero = torch.zeros_like(rho_w)
    f_w = equilibrium(rho_w, zero, zero)
    f_g = equilibrium(rho_g, zero, zero)
    for _ in range(steps):
        rho_w, _, _ = macroscopic(f_w)
        rho_g, _, _ = macroscopic(f_g)
        rho_w, rho_g = apply_wall_wettability_sc(
            rho_w, rho_g, solid, G_ads1=g_ads_water, G_ads2=0.0
        )
        zero2 = torch.zeros_like(rho_w)
        f_w = torch.where(solid.unsqueeze(0), equilibrium(rho_w, zero2, zero2), f_w)
        f_g = torch.where(solid.unsqueeze(0), equilibrium(rho_g, zero2, zero2), f_g)
        f_w, f_g = collide_sc_two_component(
            f_w, f_g, G_12=-2.5, tau1=1.0, tau2=1.0, solid_mask=solid
        )
        f_w = stream(f_w)
        f_g = stream(f_g)
        f_w = bounce_back_cells(f_w, solid)
        f_g = bounce_back_cells(f_g, solid)
    rho_w, _, _ = macroscopic(f_w)
    rho_g, _, _ = macroscopic(f_g)
    phi = rho_g / (rho_w + rho_g + 1e-12)
    return float(phi[ny // 2 + 2 : -2, 2:-2].mean() - phi[2 : ny // 2 - 2, 2:-2].mean())


class TestDeepDefectReproducers:
    """Evidence-only locks for the registered (unfixed) deep-physics items."""

    def test_neutral_wall_keeps_interface_control(self) -> None:
        """Control: G_ads = 0 (neutral wall) keeps the interface (|c|>0.4)."""
        assert abs(_closed_box_phi_contrast(0.0)) > 0.4

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Deep defect (registered, W4-B §2 item 4): the adsorption wall "
            "protocol pumps components — a closed box homogenises within "
            "~2k steps (contrast ≈ 6e-8 vs ≈ 0.57 for the neutral control). "
            "Expected to xpass once a non-pumping wettability BC lands."
        ),
    )
    def test_adsorbing_wall_holds_interface(self) -> None:
        assert abs(_closed_box_phi_contrast(0.3)) > 0.4

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Deep defect (registered, W4-B §2 item 2): collide_sc_two_component "
            "relaxes each component towards its own velocity equilibrium "
            "instead of the mixture centre-of-mass velocity, biasing the "
            "interfacial momentum coupling.  Measured profile-vs-analytical "
            "L2 at the segregation coupling ≈ 0.43 (W4-B Part 1: 26–33%). "
            "Expected to xpass once the kernel is re-centred."
        ),
    )
    def test_poiseuille_profile_matches_analytical_at_segregation(self, tmp_path: Path) -> None:
        cfg = TwoPhasePoiseuilleConfig(
            G_12=-2.5, n_steps=8000, output_interval=2000, output_root=tmp_path, overwrite=True
        )
        result = run_two_phase_poiseuille(cfg)
        assert math.isfinite(result["l2_error_rel"])
        assert result["l2_error_rel"] < 0.15
