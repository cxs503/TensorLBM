"""Regression gates for XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY.

The point of these tests is that the *model distinction* is asserted, not merely
executed: an MCMP cross-force must not leak into a two-phase-only configuration,
the shared interaction kernel must be bit-identical when the partner is a
duplicated copy, and the two coupling signs must mean opposite things.

Run with ``pytest tests/test_xval_no_mcmp_two_phase.py -q`` (≈20 s, CPU).
"""

from __future__ import annotations

import torch
import pytest

from tensorlbm.multiphase import (
    collide_sc_single_component,
    collide_sc_two_component,
    equilibrium,
    psi_exp,
    psi_linear,
    sc_single_component_force,
    sc_two_component_force,
)
from tensorlbm.solver import stream
from tensorlbm.xval import no_mcmp_two_phase as gates


class TestKernelIdentity:
    """G1: one interaction kernel, two physical readings."""

    @pytest.mark.parametrize("psi_name", ["exp", "linear"])
    @pytest.mark.parametrize("G", [5.0, 1.0, -1.0])
    def test_duplicated_partner_reproduces_self_force(self, psi_name: str, G: float) -> None:
        psi = gates.PSI_CHOICES[psi_name]
        rho = gates.random_smooth_field(32, 40, seed=7)
        fx_sc, fy_sc = sc_single_component_force(rho, G=G, psi_fn=psi)
        fx1, fy1, fx2, fy2 = sc_two_component_force(rho, rho.clone(), G_12=G, psi=psi)

        scale = float(torch.sqrt(fx_sc**2 + fy_sc**2).max())
        assert scale > 1e-3, "degenerate test field (no force to compare)"
        assert float((fx1 - fx_sc).abs().max()) <= 1e-12 * scale
        assert float((fy1 - fy_sc).abs().max()) <= 1e-12 * scale
        # exchange symmetry: swapping the components swaps the force fields
        assert float((fx2 - fx_sc).abs().max()) <= 1e-12 * scale

    def test_identity_holds_in_float32(self) -> None:
        """The two code paths must share the kernel, not just agree in float64."""
        rho = gates.random_smooth_field(24, 32, seed=9, dtype=torch.float32)
        fx_sc, fy_sc = sc_single_component_force(rho, G=5.0, psi_fn=psi_exp)
        fx1, fy1, _, _ = sc_two_component_force(rho, rho.clone(), G_12=5.0, psi=psi_exp)
        assert float(torch.maximum((fx1 - fx_sc).abs(), (fy1 - fy_sc).abs()).max()) == 0.0

    def test_identity_is_not_trivial(self) -> None:
        """A field whose force vanishes everywhere would make the gate vacuous."""
        rho = gates.random_smooth_field(32, 40, seed=3)
        fx, fy = sc_single_component_force(rho, G=5.0, psi_fn=psi_exp)
        assert float(fx.abs().max()) > 1e-3
        assert float(fy.abs().max()) > 1e-3


class TestNullCouplingPurity:
    """G2: no MCMP term survives a two-phase-only configuration."""

    def test_zero_coupling_gives_zero_force(self) -> None:
        rho_a = gates.random_smooth_field(32, 32, seed=21)
        rho_b = gates.random_smooth_field(32, 32, seed=22)
        out = sc_two_component_force(rho_a, rho_b, G_12=0.0, psi=psi_exp)
        assert all(float(f.abs().max()) == 0.0 for f in out)

    def test_uniform_partner_exerts_no_force(self) -> None:
        rho_a = gates.random_smooth_field(32, 32, seed=23)
        for const in (0.1, 1.0, 5.0):
            fx, fy, _, _ = sc_two_component_force(
                rho_a, torch.full_like(rho_a, const), G_12=-2.0, psi=psi_exp
            )
            assert float(fx.abs().max()) == 0.0
            assert float(fy.abs().max()) == 0.0

    def test_nonzero_coupling_has_teeth(self) -> None:
        result = gates.gate_null_coupling_purity()
        assert result["pass"]
        assert result["nontrivial_max"] > 1e-3


class TestDynamicalReduction:
    """G3: the duplicated-partner MCMP integration *is* the SCMP integration."""

    def test_trajectories_coincide(self) -> None:
        result = gates.gate_duplicated_partner_dynamics(G=5.0, steps=60, sample_every=20)
        assert result["pass"], result["final_rel_diff"]
        assert result["final_max_abs_diff"] <= 1e-12
        assert result["partner_tracks_component"] <= 1e-12
        # and the run is not trivial: the two-phase model actually separates
        assert result["phase_separated"]

    def test_single_step_equivalence_of_collision_operators(self) -> None:
        rho = gates.droplet_field(32, 32, radius=8.0)
        zeros = torch.zeros_like(rho)
        f = equilibrium(rho, zeros, zeros)
        f_sc = collide_sc_single_component(f, G=5.0, tau=1.0, psi_fn=psi_exp)
        f1, f2 = collide_sc_two_component(
            f, f, G_12=5.0, tau1=1.0, tau2=1.0, psi=psi_exp, force_scheme="shift"
        )
        assert float((f_sc - f1).abs().max()) == 0.0
        assert float((f1 - f2).abs().max()) == 0.0


class TestNoSelfInteraction:
    """G4: an MCMP lattice model with an inert partner is *not* a two-phase model."""

    def test_single_phase_with_inert_partner_only_diffuses(self) -> None:
        result = gates.gate_no_self_interaction(G_12=-1.0, steps=120)
        assert result["pass"]
        # contrast may only decay (viscous diffusion), never grow into an interface
        assert result["contrast_ratio_inert"] <= 1.0
        history = result["contrast_history_inert"]
        assert history == sorted(history, reverse=True), history
        # control: the same coupling does segregate when the partner is complementary
        assert result["spatial_correlation_final"] < 0.0


class TestCouplingSign:
    """G5: sign semantics of the shared kernel, measured."""

    def test_alignment_is_odd_and_tracer_follows_sign(self) -> None:
        result = gates.gate_coupling_sign(steps=200)
        assert result["pass"]
        assert result["alignment_oddness_rel"] is not None
        assert result["alignment_oddness_rel"] <= 1e-12
        for row in result["rows"]:
            assert row["tracer_sign"] == row["alignment_sign"]
            if row["G_12"] != 0:
                # 200 steps is enough for the frozen-partner tracer to transport
                # the single species onto its preferred region (|corr| -> 1 slowly)
                assert abs(row["tracer_correlation"]) > 0.95

    def test_negative_coupling_repels_positive_attracts(self) -> None:
        result = gates.gate_coupling_sign(couplings=(-1.0, 1.0), steps=80)
        by_sign = {row["G_12"]: row for row in result["rows"]}
        assert by_sign[-1.0]["tracer_correlation"] < 0.0
        assert by_sign[1.0]["tracer_correlation"] > 0.0


class TestTwoPhaseCoexistence:
    """G6: the two-phase-only path against the analytic Maxwell construction."""

    def test_flat_interface_matches_maxwell_within_acceptance(self) -> None:
        result = gates.gate_two_phase_coexistence((5.0,), steps=2500)
        row = result["rows"][0]
        assert row["pass"], row
        assert row["err_liquid"] <= 0.15 and row["err_vapour"] <= 0.15
        # engineering target for this configuration
        assert row["err_vapour"] <= 0.03, row["err_vapour"]


class TestFieldHelpers:
    """Helpers used by the gates must stay well behaved."""

    def test_random_smooth_field_is_periodic_and_bounded(self) -> None:
        field = gates.random_smooth_field(16, 24, seed=1, rho_min=0.3, rho_max=1.1)
        assert field.shape == (16, 24)
        assert float(field.min()) >= 0.3 - 1e-12
        assert float(field.max()) <= 1.1 + 1e-12
        # periodic: no step discontinuity across the wrap-around seam
        assert float((field[:, 0] - field[:, -1]).abs().max()) < 0.5

    def test_droplet_field_profile(self) -> None:
        field = gates.droplet_field(32, 32, radius=8.0, rho_in=1.0, rho_out=0.2)
        assert float(field[16, 16]) > 0.95
        assert float(field[0, 0]) < 0.25