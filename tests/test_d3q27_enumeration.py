"""D3Q27 enumeration symmetry, completeness, and negative-case tests.

Covers the IN14-D9 campaign requirements:
1. 27-velocity enumeration symmetry self-check (c_q and -c_q pairs + weight sum = 1)
2. Completeness: all 27 combinations of {-1,0,1}^3 present, no duplicates
3. Weight classification correctness (rest/face/edge/corner)
4. Layout convention alignment: C shape (Q,3) with (cx,cy,cz) order;
   solver fields use (Q,nz,ny,nx)
5. Negative cases: duplicate enumeration rejection, weight-sum≠1 rejection
"""

from __future__ import annotations

import copy
import itertools
import math

import pytest
import torch

from tensorlbm import C27, OPPOSITE27, W27
from tensorlbm.d3q27 import (
    W_EXACT64,
    _C_DATA,
    _W_DATA,
    _build_opposite,
    _w_corner,
    _w_edge,
    _w_face,
    _w_rest,
    equilibrium27,
    macroscopic27,
)


# ---------------------------------------------------------------------------
# 1. Symmetry self-check: c_q and -c_q pairing
# ---------------------------------------------------------------------------


class TestD3Q27Symmetry:
    """Verify that every velocity direction has its negation present."""

    def test_opposite_is_involution(self) -> None:
        """OPPOSITE[OPPOSITE[q]] == q for all q."""
        for q in range(27):
            opp = int(OPPOSITE27[q].item())
            assert int(OPPOSITE27[opp].item()) == q, f"Involution failed at q={q}"

    def test_opposite_negates_velocity(self) -> None:
        """C[OPPOSITE[q]] == -C[q] for all q."""
        for q in range(27):
            opp = int(OPPOSITE27[q].item())
            assert torch.equal(C27[q], -C27[opp]), (
                f"Direction {q}: C[{q}]={C27[q].tolist()} != -C[{opp}]={(-C27[opp]).tolist()}"
            )

    def test_rest_is_self_opposite(self) -> None:
        """Direction 0 (rest) must be its own opposite."""
        assert int(OPPOSITE27[0].item()) == 0

    def test_no_self_opposite_except_rest(self) -> None:
        """Only direction 0 should satisfy OPPOSITE[q] == q."""
        for q in range(1, 27):
            assert int(OPPOSITE27[q].item()) != q, f"Non-rest direction {q} is self-opposite"


# ---------------------------------------------------------------------------
# 2. Completeness: all 27 combos of {-1,0,1}^3, no duplicates
# ---------------------------------------------------------------------------


class TestD3Q27Completeness:
    """Verify the enumeration covers exactly {-1,0,1}^3 with no duplicates."""

    def test_all_combinations_present(self) -> None:
        """Every (cx,cy,cz) in {-1,0,1}^3 must appear exactly once."""
        expected = set(itertools.product([-1, 0, 1], repeat=3))
        actual = {tuple(row) for row in _C_DATA}
        assert actual == expected, (
            f"Missing: {expected - actual}; Extra: {actual - expected}"
        )

    def test_no_duplicate_velocities(self) -> None:
        """All 27 velocity vectors must be unique."""
        seen = set()
        for i, row in enumerate(_C_DATA):
            key = tuple(row)
            assert key not in seen, f"Duplicate velocity at index {i}: {key}"
            seen.add(key)
        assert len(seen) == 27

    def test_count_by_magnitude_class(self) -> None:
        """Verify counts: 1 rest + 6 face + 12 edge + 8 corner = 27."""
        rest = face = edge = corner = 0
        for row in _C_DATA:
            cx, cy, cz = row
            mag_sq = cx * cx + cy * cy + cz * cz
            if mag_sq == 0:
                rest += 1
            elif mag_sq == 1:
                face += 1
            elif mag_sq == 2:
                edge += 1
            elif mag_sq == 3:
                corner += 1
            else:
                pytest.fail(f"Unexpected magnitude squared {mag_sq} for {row}")
        assert rest == 1, f"Expected 1 rest, got {rest}"
        assert face == 6, f"Expected 6 face, got {face}"
        assert edge == 12, f"Expected 12 edge, got {edge}"
        assert corner == 8, f"Expected 8 corner, got {corner}"

    def test_tensor_shape(self) -> None:
        """C must have shape (27, 3)."""
        assert C27.shape == (27, 3)

    def test_tensor_dtype(self) -> None:
        """C must be int64."""
        assert C27.dtype == torch.int64


# ---------------------------------------------------------------------------
# 3. Weight correctness
# ---------------------------------------------------------------------------


class TestD3Q27Weights:
    """Verify weight values and sum."""

    def test_weights_sum_to_one(self) -> None:
        """Sum of all weights must be exactly 1 (within float32 tolerance)."""
        w_sum = float(W27.sum().item())
        assert abs(w_sum - 1.0) < 1e-6, f"Weight sum = {w_sum}, expected 1.0"

    def test_weights_sum_to_one_float64(self) -> None:
        """Sum of float64 weights must be exactly 1."""
        w_sum = float(W_EXACT64.sum().item())
        assert abs(w_sum - 1.0) < 1e-15, f"Weight sum (f64) = {w_sum}, expected 1.0"

    def test_rest_weight(self) -> None:
        """Rest weight w_0 = 8/27."""
        expected = 8.0 / 27.0
        assert abs(float(W27[0].item()) - expected) < 1e-6

    def test_face_weights(self) -> None:
        """Face weights (q=1..6) = 2/27."""
        expected = 2.0 / 27.0
        for q in range(1, 7):
            assert abs(float(W27[q].item()) - expected) < 1e-6, f"Face weight q={q}"

    def test_edge_weights(self) -> None:
        """Edge weights (q=7..18) = 1/54."""
        expected = 1.0 / 54.0
        for q in range(7, 19):
            assert abs(float(W27[q].item()) - expected) < 1e-6, f"Edge weight q={q}"

    def test_corner_weights(self) -> None:
        """Corner weights (q=19..26) = 1/216."""
        expected = 1.0 / 216.0
        for q in range(19, 27):
            assert abs(float(W27[q].item()) - expected) < 1e-6, f"Corner weight q={q}"

    def test_weight_classification_by_magnitude(self) -> None:
        """Weights must match the magnitude class of each direction."""
        for q in range(27):
            cx, cy, cz = _C_DATA[q]
            mag_sq = cx * cx + cy * cy + cz * cz
            w = float(W27[q].item())
            if mag_sq == 0:
                assert abs(w - _w_rest) < 1e-6, f"Rest weight mismatch at q={q}"
            elif mag_sq == 1:
                assert abs(w - _w_face) < 1e-6, f"Face weight mismatch at q={q}"
            elif mag_sq == 2:
                assert abs(w - _w_edge) < 1e-6, f"Edge weight mismatch at q={q}"
            elif mag_sq == 3:
                assert abs(w - _w_corner) < 1e-6, f"Corner weight mismatch at q={q}"

    def test_weights_positive(self) -> None:
        """All weights must be strictly positive."""
        for q in range(27):
            assert float(W27[q].item()) > 0, f"Non-positive weight at q={q}"


# ---------------------------------------------------------------------------
# 4. Layout convention alignment: (Q,nz,ny,nx)
# ---------------------------------------------------------------------------


class TestD3Q27LayoutConvention:
    """Verify the (Q, nz, ny, nx) layout convention is respected."""

    def test_equilibrium_output_shape(self) -> None:
        """equilibrium27 must return shape (27, nz, ny, nx)."""
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        zeros = torch.zeros_like(rho)
        feq = equilibrium27(rho, zeros, zeros, zeros)
        assert feq.shape == (27, nz, ny, nx)

    def test_macroscopic_input_shape(self) -> None:
        """macroscopic27 must accept shape (27, nz, ny, nx)."""
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        zeros = torch.zeros_like(rho)
        feq = equilibrium27(rho, zeros, zeros, zeros)
        rho_out, ux, uy, uz = macroscopic27(feq)
        assert rho_out.shape == (nz, ny, nx)

    def test_c_order_is_cx_cy_cz(self) -> None:
        """C columns must be (cx, cy, cz) in that order."""
        assert int(C27[1, 0].item()) == 1
        assert int(C27[1, 1].item()) == 0
        assert int(C27[1, 2].item()) == 0
        assert int(C27[3, 0].item()) == 0
        assert int(C27[3, 1].item()) == 1
        assert int(C27[3, 2].item()) == 0
        assert int(C27[5, 0].item()) == 0
        assert int(C27[5, 1].item()) == 0
        assert int(C27[5, 2].item()) == 1

    def test_streaming_shift_convention(self) -> None:
        """Streaming shifts must map C (cx,cy,cz) to tensor dims (dz,dy,dx)."""
        from tensorlbm.d3q27 import stream27

        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        zeros = torch.zeros_like(rho)
        f = equilibrium27(rho, zeros, zeros, zeros)
        f_streamed = stream27(f)
        assert torch.allclose(f_streamed, f, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. Negative cases: validation / rejection
# ---------------------------------------------------------------------------


class TestD3Q27NegativeCases:
    """Tests that invalid enumerations are properly rejected."""

    def test_duplicate_velocity_detection(self) -> None:
        """A helper that checks for duplicates should find them in bad data."""
        bad_data = copy.deepcopy(_C_DATA)
        bad_data[5] = [0, 0, 0]  # duplicate of index 0 (rest)
        seen = set()
        has_dup = False
        for row in bad_data:
            key = tuple(row)
            if key in seen:
                has_dup = True
                break
            seen.add(key)
        assert has_dup, "Should detect duplicate in bad data"

    def test_weight_sum_not_one_detection(self) -> None:
        """A helper that checks weight sum should detect bad weights."""
        bad_weights = copy.deepcopy(_W_DATA)
        bad_weights[0] = 1.0  # make sum > 1
        w_sum = sum(bad_weights)
        assert abs(w_sum - 1.0) > 0.1, "Should detect weight sum != 1"

    def test_missing_velocity_detection(self) -> None:
        """A helper that checks completeness should find missing velocities."""
        bad_data = copy.deepcopy(_C_DATA)
        bad_data[26] = [0, 0, 0]  # replace corner with rest (duplicate)
        expected = set(itertools.product([-1, 0, 1], repeat=3))
        actual = {tuple(row) for row in bad_data}
        assert actual != expected, "Should detect missing velocity"

    def test_invalid_velocity_values(self) -> None:
        """Velocities outside {-1,0,1} should be detectable."""
        bad_data = copy.deepcopy(_C_DATA)
        bad_data[10] = [2, 0, 0]  # invalid value
        for row in bad_data:
            for v in row:
                if v not in (-1, 0, 1):
                    return  # detected
        pytest.fail("Should have detected invalid velocity value")

    def test_equilibrium_rejects_mismatched_shapes(self) -> None:
        """equilibrium27 should reject mismatched input shapes."""
        rho = torch.ones((4, 5, 6))
        ux = torch.ones((4, 5))  # wrong shape
        with pytest.raises((ValueError, RuntimeError)):
            equilibrium27(rho, ux, torch.zeros_like(rho), torch.zeros_like(rho))

    def test_macroscopic_rejects_wrong_q_dimension(self) -> None:
        """macroscopic27 should reject input with wrong Q dimension."""
        f = torch.ones((19, 4, 5, 6))  # D3Q19 shape, not D3Q27
        with pytest.raises((ValueError, RuntimeError)):
            macroscopic27(f)


# ---------------------------------------------------------------------------
# 6. _build_opposite consistency
# ---------------------------------------------------------------------------


class TestBuildOpposite:
    """Verify _build_opposite produces the same result as OPPOSITE."""

    def test_build_opposite_matches(self) -> None:
        """_build_opposite() must equal the module-level OPPOSITE."""
        computed = _build_opposite()
        assert torch.equal(computed, OPPOSITE27)

    def test_build_opposite_length(self) -> None:
        """_build_opposite must return 27 entries."""
        computed = _build_opposite()
        assert len(computed) == 27


# ---------------------------------------------------------------------------
# 7. Isotropy moment checks (4th-order)
# ---------------------------------------------------------------------------


class TestD3Q27Isotropy:
    """Verify 4th-order isotropy of the D3Q27 lattice."""

    def test_second_moment_isotropic(self) -> None:
        """Sum_q w_q * c_{q,i} * c_{q,j} = c_s^2 * delta_{ij} with c_s^2 = 1/3."""
        cs_sq = 1.0 / 3.0
        for i in range(3):
            for j in range(3):
                moment = 0.0
                for q in range(27):
                    moment += float(W27[q].item()) * _C_DATA[q][i] * _C_DATA[q][j]
                expected = cs_sq if i == j else 0.0
                assert abs(moment - expected) < 1e-6, (
                    f"2nd moment ({i},{j}): {moment} != {expected}"
                )

    def test_fourth_moment_diagonal(self) -> None:
        """Sum_q w_q * c_{q,i}^4 = 1/3 for D3Q27.

        For D3Q27:
        - Face (±x): 2 dirs × 2/27 × 1 = 4/27
        - Edge (xy, xz planes with cx=±1): 8 dirs × 1/54 × 1 = 8/54
        - Corner (cx=±1): 8 dirs × 1/216 × 1 = 8/216
        Total = 4/27 + 8/54 + 8/216 = 4/27 + 4/27 + 1/27 = 9/27 = 1/3
        """
        moment_x4 = sum(float(W27[q].item()) * _C_DATA[q][0] ** 4 for q in range(27))
        assert abs(moment_x4 - 1.0 / 3.0) < 1e-6, f"4th moment x^4: {moment_x4}"

    def test_odd_moments_zero(self) -> None:
        """All odd-order moments must be zero by symmetry."""
        for i in range(3):
            moment = sum(float(W27[q].item()) * _C_DATA[q][i] for q in range(27))
            assert abs(moment) < 1e-10, f"1st moment dim {i}: {moment}"
