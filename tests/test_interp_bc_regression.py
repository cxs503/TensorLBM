"""Regression equivalence & bug-identification tests for interpolated BC.

Tests three dimensions:
1. Bug identification — does the original bouzidi_bounce_back_3d carry known bugs?
2. Equivalence — original D3Q19 vs common-module D3Q19 (bit-identical);
   D3Q27 (common-only) physical reasonableness.
3. Combination — interpolated BC + BGK collision end-to-end.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import (
    C as C3D,
    OPPOSITE as OPP3D,
    equilibrium3d,
    macroscopic3d,
)
from tensorlbm.d3q27 import (
    OPPOSITE as OPP27,
    equilibrium27,
    macroscopic27,
)
from tensorlbm.interpolated_bc import (
    bouzidi_bounce_back_3d,
    compute_q_sphere,
)
from tensorlbm.interpolated_bc_common import (
    bouzidi_bounce_back_3d_common,
    compute_q_sphere_27,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_f_pair_3d(
    nz: int = 6, ny: int = 6, nx: int = 6, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (f, f_prev, fluid_nodes) for D3Q19 with perturbed distributions."""
    torch.manual_seed(seed)
    rho = torch.ones(nz, ny, nx)
    f = equilibrium3d(
        rho,
        torch.full_like(rho, 0.05),
        torch.zeros_like(rho),
        torch.zeros_like(rho),
    )
    f = f + 0.01 * torch.randn_like(f)
    f_prev = equilibrium3d(
        rho,
        torch.full_like(rho, -0.03),
        torch.full_like(rho, 0.02),
        torch.zeros_like(rho),
    )
    f_prev = f_prev + 0.01 * torch.randn_like(f_prev)
    fluid_nodes = torch.zeros(nz, ny, nx, dtype=torch.bool)
    fluid_nodes[2:4, 2:4, 2:4] = True
    return f, f_prev, fluid_nodes


def _make_f_pair_27(
    nz: int = 6, ny: int = 6, nx: int = 6, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (f, f_prev, fluid_nodes) for D3Q27."""
    torch.manual_seed(seed)
    rho = torch.ones(nz, ny, nx)
    f = equilibrium27(
        rho,
        torch.full_like(rho, 0.05),
        torch.zeros_like(rho),
        torch.zeros_like(rho),
    )
    f = f + 0.01 * torch.randn_like(f)
    f_prev = equilibrium27(
        rho,
        torch.full_like(rho, -0.03),
        torch.full_like(rho, 0.02),
        torch.zeros_like(rho),
    )
    f_prev = f_prev + 0.01 * torch.randn_like(f_prev)
    fluid_nodes = torch.zeros(nz, ny, nx, dtype=torch.bool)
    fluid_nodes[2:4, 2:4, 2:4] = True
    return f, f_prev, fluid_nodes


# ===========================================================================
# 1. BUG IDENTIFICATION
# ===========================================================================

class TestBouzidiBounceBack3DBugs:
    """Identify pre-existing bugs in the original bouzidi_bounce_back_3d."""

    def test_quadratic_uses_fp_opp_not_fp_d(self) -> None:
        """BUG: quadratic branch (q >= 0.5) uses fp_opp instead of fp_d.

        Standard BFL quadratic (q >= 0.5):
            f_bc = f_opp / (2q) + (2q - 1) / (2q) * f_prev[direction]
        Code uses:
            f_bc = f_opp / (2q) + (2q - 1) / (2q) * f_prev[opp]
                                     ^^^^^^^^^^^^^^^^  should be f_prev[direction]

        This bug is faithfully reproduced in the common module.
        """
        f, f_prev, fluid_nodes = _make_f_pair_3d()
        direction = 1
        opp = int(OPP3D[direction].item())
        q_val = 0.75
        q = torch.full(f.shape[1:], q_val)

        f_out = bouzidi_bounce_back_3d(f, f_prev, fluid_nodes, q, direction=direction)

        # What the code actually computes (using fp_opp):
        f_opp = f[opp][fluid_nodes]
        fp_opp = f_prev[opp][fluid_nodes]
        fp_d = f_prev[direction][fluid_nodes]
        code_result = f_opp / (2 * q_val) + (2 * q_val - 1) / (2 * q_val) * fp_opp

        # What standard BFL would compute (using fp_d):
        standard_result = f_opp / (2 * q_val) + (2 * q_val - 1) / (2 * q_val) * fp_d

        # The code matches the fp_opp version (buggy), not the fp_d version (standard)
        assert torch.allclose(f_out[direction][fluid_nodes], code_result, atol=1e-6)
        # The bug means code != standard when fp_opp != fp_d
        if not torch.allclose(fp_opp, fp_d, atol=1e-8):
            assert not torch.allclose(code_result, standard_result, atol=1e-6), (
                "Expected quadratic formula to differ from standard BFL due to fp_opp bug"
            )

    def test_quadratic_bug_reproduced_in_common(self) -> None:
        """The same quadratic bug is reproduced in the common module."""
        f, f_prev, fluid_nodes = _make_f_pair_3d()
        direction = 1
        opp = int(OPP3D[direction].item())
        q_val = 0.75
        q = torch.full(f.shape[1:], q_val)

        f_out_common = bouzidi_bounce_back_3d_common(
            f, f_prev, fluid_nodes, q, direction=direction, lattice="D3Q19"
        )

        f_opp = f[opp][fluid_nodes]
        fp_opp = f_prev[opp][fluid_nodes]
        code_result = f_opp / (2 * q_val) + (2 * q_val - 1) / (2 * q_val) * fp_opp

        assert torch.allclose(f_out_common[direction][fluid_nodes], code_result, atol=1e-6)

    def test_linear_branch_uses_correct_fp_d(self) -> None:
        """Linear branch (q < 0.5) correctly uses fp_d (pre-stream same direction).

        Standard BFL linear (q < 0.5):
            f_bc = 2q * f_opp + (1 - 2q) * f_prev[direction]
        This is correct in both original and common.
        """
        f, f_prev, fluid_nodes = _make_f_pair_3d()
        direction = 1
        opp = int(OPP3D[direction].item())
        q_val = 0.25
        q = torch.full(f.shape[1:], q_val)

        f_out = bouzidi_bounce_back_3d(f, f_prev, fluid_nodes, q, direction=direction)

        f_opp = f[opp][fluid_nodes]
        fp_d = f_prev[direction][fluid_nodes]
        expected = 2.0 * q_val * f_opp + (1.0 - 2.0 * q_val) * fp_d

        assert torch.allclose(f_out[direction][fluid_nodes], expected, atol=1e-6)

    def test_2d_dead_code_line65(self) -> None:
        """BUG (cosmetic): 2D bouzidi_bounce_back has a dead no-op expression.

        Line 65 of interpolated_bc.py: ``f[direction][fluid_nodes]`` is a bare
        indexing expression with no assignment — it computes a value and
        discards it.  This is dead code that does not affect output but
        indicates a possible leftover from an earlier formula.
        """
        import inspect

        from tensorlbm.interpolated_bc import bouzidi_bounce_back

        source = inspect.getsource(bouzidi_bounce_back)
        # The dead line exists as a standalone expression
        assert "f[direction][fluid_nodes]" in source
        # Verify it's NOT an assignment (no = after it on the same logical line)
        lines = source.split("\n")
        dead_lines = [
            ln for ln in lines
            if "f[direction][fluid_nodes]" in ln and "=" not in ln.split("f[direction]")[0]
        ]
        # At least one dead expression line exists
        assert len(dead_lines) >= 1, "Expected dead no-op expression f[direction][fluid_nodes]"


# ===========================================================================
# 2. EQUIVALENCE: original D3Q19 vs common D3Q19
# ===========================================================================

class TestBouzidiEquivalenceD3Q19:
    """Verify original bouzidi_bounce_back_3d == common D3Q19, bit-identical."""

    @pytest.mark.parametrize("q_val", [0.01, 0.1, 0.25, 0.49, 0.5, 0.51, 0.75, 0.99])
    @pytest.mark.parametrize("direction", [0, 1, 5, 9, 13, 18])
    def test_bit_identical_all_q_directions(self, q_val: float, direction: int) -> None:
        """Original and common must produce bit-identical results for D3Q19."""
        f, f_prev, fluid_nodes = _make_f_pair_3d()
        q = torch.full(f.shape[1:], q_val)

        f_orig = bouzidi_bounce_back_3d(
            f.clone(), f_prev.clone(), fluid_nodes, q, direction=direction
        )
        f_comm = bouzidi_bounce_back_3d_common(
            f.clone(), f_prev.clone(), fluid_nodes, q, direction=direction, lattice="D3Q19"
        )
        assert torch.equal(f_orig, f_comm), (
            f"Mismatch at direction={direction}, q={q_val}: "
            f"max_diff={ (f_orig - f_comm).abs().max().item()}"
        )

    def test_all_19_directions_sweep(self) -> None:
        """Exhaustive: all 19 directions × multiple q values."""
        f, f_prev, fluid_nodes = _make_f_pair_3d()
        q_vals = [0.1, 0.3, 0.5, 0.7, 0.9]

        for direction in range(19):
            for q_val in q_vals:
                q = torch.full(f.shape[1:], q_val)
                f_orig = bouzidi_bounce_back_3d(
                    f.clone(), f_prev.clone(), fluid_nodes, q, direction=direction
                )
                f_comm = bouzidi_bounce_back_3d_common(
                    f.clone(), f_prev.clone(), fluid_nodes, q, direction=direction, lattice="D3Q19"
                )
                assert torch.equal(f_orig, f_comm)

    def test_non_fluid_nodes_unchanged(self) -> None:
        """Populations at non-fluid nodes must be identical in both versions."""
        f, f_prev, fluid_nodes = _make_f_pair_3d()
        q = torch.full(f.shape[1:], 0.3)
        direction = 5

        f_orig = bouzidi_bounce_back_3d(f, f_prev, fluid_nodes, q, direction=direction)
        f_comm = bouzidi_bounce_back_3d_common(
            f, f_prev, fluid_nodes, q, direction=direction, lattice="D3Q19"
        )
        mask_other = ~fluid_nodes
        assert torch.equal(f_orig[direction][mask_other], f_comm[direction][mask_other])

    def test_q_half_gives_bounce_back_both(self) -> None:
        """At q=0.5, both versions reproduce standard bounce-back: f_bc = f_opp."""
        f, f_prev, fluid_nodes = _make_f_pair_3d()
        direction = 7
        opp = int(OPP3D[direction].item())
        q = torch.full(f.shape[1:], 0.5)

        f_orig = bouzidi_bounce_back_3d(f, f_prev, fluid_nodes, q, direction=direction)
        f_comm = bouzidi_bounce_back_3d_common(
            f, f_prev, fluid_nodes, q, direction=direction, lattice="D3Q19"
        )
        # Both should equal f[opp] at fluid nodes
        assert torch.allclose(f_orig[direction][fluid_nodes], f[opp][fluid_nodes], atol=1e-6)
        assert torch.allclose(f_comm[direction][fluid_nodes], f[opp][fluid_nodes], atol=1e-6)


# ===========================================================================
# 2b. D3Q27 physical reasonableness (common-only, no original to compare)
# ===========================================================================

class TestBouzidiD3Q27PhysicalReasonableness:
    """D3Q27 is common-module-only; verify physical reasonableness."""

    def test_q_half_bounce_back_all_directions(self) -> None:
        """q=0.5 must give standard bounce-back for all 27 directions."""
        f, f_prev, fluid_nodes = _make_f_pair_27()
        q = torch.full(f.shape[1:], 0.5)

        for direction in range(27):
            opp = int(OPP27[direction].item())
            f_out = bouzidi_bounce_back_3d_common(
                f.clone(), f_prev.clone(), fluid_nodes, q,
                direction=direction, lattice="D3Q27",
            )
            assert torch.allclose(
                f_out[direction][fluid_nodes], f[opp][fluid_nodes], atol=1e-5
            ), f"q=0.5 bounce-back failed for direction {direction}"

    def test_output_finite_all_directions(self) -> None:
        """All outputs must be finite for all 27 directions."""
        f, f_prev, fluid_nodes = _make_f_pair_27()
        for direction in range(27):
            for q_val in [0.01, 0.25, 0.5, 0.75, 0.99]:
                q = torch.full(f.shape[1:], q_val)
                f_out = bouzidi_bounce_back_3d_common(
                    f.clone(), f_prev.clone(), fluid_nodes, q,
                    direction=direction, lattice="D3Q27",
                )
                assert torch.isfinite(f_out).all(), f"Non-finite at dir={direction}, q={q_val}"

    def test_non_fluid_nodes_unchanged(self) -> None:
        """Non-fluid-node populations must not change."""
        f, f_prev, fluid_nodes = _make_f_pair_27()
        q = torch.full(f.shape[1:], 0.3)
        direction = 5
        f_out = bouzidi_bounce_back_3d_common(
            f, f_prev, fluid_nodes, q, direction=direction, lattice="D3Q27"
        )
        mask_other = ~fluid_nodes
        assert torch.equal(f_out[direction][mask_other], f[direction][mask_other])

    def test_compute_q_sphere_27_shapes_and_range(self) -> None:
        """compute_q_sphere_27 returns correct shapes and valid q range."""
        nz, ny, nx = 16, 16, 16
        mask, q = compute_q_sphere_27(nx, ny, nz, 8.0, 8.0, 8.0, 4.0, torch.device("cpu"))
        assert mask.shape == (27, nz, ny, nx)
        assert q.shape == (27, nz, ny, nx)
        assert mask.any(), "No boundary nodes detected"
        boundary_q = q[mask]
        assert float(boundary_q.min().item()) > 0.0
        assert float(boundary_q.max().item()) <= 1.0 + 1e-5
        # Non-boundary entries are 0.5
        assert torch.allclose(q[~mask], torch.full_like(q[~mask], 0.5))

    def test_d3q27_more_boundary_nodes_than_d3q19(self) -> None:
        """D3Q27 has more directions, so should detect >= boundary nodes vs D3Q19."""
        nz, ny, nx = 16, 16, 16
        mask19, _ = compute_q_sphere(nx, ny, nz, 8.0, 8.0, 8.0, 4.0, torch.device("cpu"))
        mask27, _ = compute_q_sphere_27(nx, ny, nz, 8.0, 8.0, 8.0, 4.0, torch.device("cpu"))
        # D3Q27 has 26 non-rest directions vs D3Q19's 18, so >= boundary nodes
        assert mask27.sum() >= mask19.sum()

    def test_d3q27_shared_directions_match_d3q19(self) -> None:
        """D3Q19 directions are a subset of D3Q27; q-fields should agree on shared dirs."""
        nz, ny, nx = 16, 16, 16
        _, q19 = compute_q_sphere(nx, ny, nz, 8.0, 8.0, 8.0, 4.0, torch.device("cpu"))
        _, q27 = compute_q_sphere_27(nx, ny, nz, 8.0, 8.0, 8.0, 4.0, torch.device("cpu"))

        # D3Q19 and D3Q27 share the same first 19 direction vectors
        # (both include all face and edge directions)
        c19 = C3D  # (19, 3)
        from tensorlbm.d3q27 import C as C27
        c27 = C27  # (27, 3)

        # Find matching directions
        matched = 0
        for i19 in range(19):
            for i27 in range(27):
                if torch.equal(c19[i19], c27[i27]):
                    # Compare q-fields for this direction
                    if torch.allclose(q19[i19], q27[i27], atol=1e-4):
                        matched += 1
                    break
        # At least 15 of 19 shared directions should have matching q-fields
        assert matched >= 15, f"Only {matched}/19 shared directions have matching q-fields"


# ===========================================================================
# 3. COMBINATION: interpolated BC + collision
# ===========================================================================

class TestInterpolatedBCWithCollision:
    """Combination test: interpolated BC applied after BGK collision."""

    def _bgk_collision_3d(self, f: torch.Tensor, tau: float, lattice: str) -> torch.Tensor:
        """Simple BGK collision for D3Q19 or D3Q27."""
        if lattice == "D3Q19":
            rho, ux, uy, uz = macroscopic3d(f)
            feq = equilibrium3d(rho, ux, uy, uz)
        else:
            rho, ux, uy, uz = macroscopic27(f)
            feq = equilibrium27(rho, ux, uy, uz)
        return f - (f - feq) / tau

    def test_bouzidi_after_collision_d3q19_finite(self) -> None:
        """Apply bouzidi BC after BGK collision; result must be finite."""
        nz, ny, nx = 6, 6, 6
        rho = torch.ones(nz, ny, nx)
        f = equilibrium3d(
            rho,
            torch.full_like(rho, 0.05),
            torch.zeros_like(rho),
            torch.zeros_like(rho),
        )
        f_prev = f.clone()
        fluid_nodes = torch.zeros(nz, ny, nx, dtype=torch.bool)
        fluid_nodes[2:4, 2:4, 2:4] = True
        tau = 1.0

        # Collision step
        f_post = self._bgk_collision_3d(f, tau, "D3Q19")
        assert torch.isfinite(f_post).all()

        # Apply BC for several directions with a sphere q-field
        mask, q_field = compute_q_sphere(nx, ny, nz, 3.0, 3.0, 3.0, 1.5, torch.device("cpu"))
        for d in range(19):
            if mask[d].any():
                q_d = q_field[d].clone()
                f_post = bouzidi_bounce_back_3d(
                    f_post, f_prev, mask[d], q_d,
                    direction=d,
                )
        assert torch.isfinite(f_post).all()

    def test_bouzidi_common_after_collision_d3q19(self) -> None:
        """Common-module BC after collision == original BC after collision."""
        f, f_prev, fluid_nodes = _make_f_pair_3d()
        tau = 1.0

        f_post = self._bgk_collision_3d(f.clone(), tau, "D3Q19")
        f_post_common = self._bgk_collision_3d(f.clone(), tau, "D3Q19")
        # Collision is deterministic, so post-collision states are identical
        assert torch.equal(f_post, f_post_common)

        direction = 5
        q = torch.full(f.shape[1:], 0.35)
        f_orig = bouzidi_bounce_back_3d(f_post.clone(), f_prev, fluid_nodes, q, direction=direction)
        f_comm = bouzidi_bounce_back_3d_common(
            f_post_common.clone(), f_prev, fluid_nodes, q, direction=direction, lattice="D3Q19"
        )
        assert torch.equal(f_orig, f_comm)

    def test_bouzidi_common_after_collision_d3q27_finite(self) -> None:
        """D3Q27 common BC after collision; result must be finite."""
        f, f_prev, fluid_nodes = _make_f_pair_27()
        tau = 1.0

        f_post = self._bgk_collision_3d(f, tau, "D3Q27")
        assert torch.isfinite(f_post).all()

        for direction in [1, 5, 10, 15, 20, 25]:
            q = torch.full(f.shape[1:], 0.4)
            f_post = bouzidi_bounce_back_3d_common(
                f_post, f_prev, fluid_nodes, q,
                direction=direction, lattice="D3Q27",
            )
        assert torch.isfinite(f_post).all()

    def test_multi_step_collision_bc_stability(self) -> None:
        """Run several collision+BC steps; verify no NaN/Inf divergence."""
        f, f_prev, fluid_nodes = _make_f_pair_3d()
        tau = 1.5

        for step in range(10):
            f = self._bgk_collision_3d(f, tau, "D3Q19")
            for direction in [1, 5, 10]:
                q = torch.full(f.shape[1:], 0.3 + 0.05 * step)
                q = q.clamp(0.01, 0.99)
                f = bouzidi_bounce_back_3d_common(
                    f, f_prev, fluid_nodes, q,
                    direction=direction, lattice="D3Q19",
                )
            assert torch.isfinite(f).all(), f"Non-finite at step {step}"
            # Mass should not explode
            mass = f.sum().item()
            assert mass < 1e6, f"Mass explosion at step {step}: {mass}"
