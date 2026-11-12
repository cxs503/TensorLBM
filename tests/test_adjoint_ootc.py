"""Tests for the adjoint / OOTC gradient module.

Verifies:
1. BPTT gradient is non-trivial (non-zero, finite)
2. OOTC gradient matches BPTT reference within 1e-5 relative error
3. OOTC objective matches BPTT objective
4. Memory scaling: OOTC peak memory << BPTT peak memory
5. Determinism: repeated OOTC runs give identical results
6. Edge cases: k=T (single segment), k=1 (every step checkpointed)
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TensorLBM", "src"))

from tensorlbm.adjoint.lbm2d import initial_pulse, step, macro
from tensorlbm.adjoint.bptt import bptt_gradient
from tensorlbm.adjoint.ootc import ootc_gradient


def _make_problem(N=32, T=100, tau=1.0, amp=0.05):
    """Create a small test problem with a perturbed reference."""
    f0 = initial_pulse(N, N, amp=amp, sigma=float(N) / 8.0)
    # Run forward to get terminal state
    f = f0.copy()
    for _ in range(T):
        f = step(f, tau)
    rho_t, ux_t, uy_t = macro(f)
    ux_ref = ux_t * 0.9
    uy_ref = uy_t * 0.9
    return tau, T, f0, ux_ref, uy_ref


class TestBPTTGradient:
    """BPTT reference gradient tests."""

    def test_gradient_nonzero(self):
        """BPTT gradient is non-zero for a non-trivial problem."""
        tau, T, f0, ux_ref, uy_ref = _make_problem(N=32, T=100)
        dJ, J, _, _ = bptt_gradient(tau, T, f0, ux_ref, uy_ref)
        assert abs(dJ) > 1e-15, "Gradient should be non-zero"
        assert np.isfinite(dJ), "Gradient should be finite"
        assert J >= 0, "Objective should be non-negative"

    def test_gradient_finite_large_T(self):
        """BPTT gradient remains finite for larger T."""
        tau, T, f0, ux_ref, uy_ref = _make_problem(N=16, T=500)
        dJ, J, _, _ = bptt_gradient(tau, T, f0, ux_ref, uy_ref)
        assert np.isfinite(dJ)
        assert np.isfinite(J)

    def test_zero_reference_gives_zero_gradient(self):
        """If reference equals terminal, gradient should be ~0."""
        tau, T, f0, _, _ = _make_problem(N=16, T=50)
        # Run forward to get exact terminal
        f = f0.copy()
        for _ in range(T):
            f = step(f, tau)
        _, ux_t, uy_t = macro(f)
        dJ, J, _, _ = bptt_gradient(tau, T, f0, ux_t, uy_t)
        assert abs(dJ) < 1e-12, "Gradient should be ~0 when reference=terminal"
        assert J < 1e-12, "Objective should be ~0"


class TestOOTCvsBPTT:
    """OOTC gradient must match BPTT reference."""

    @pytest.mark.parametrize("k", [10, 25, 50, 100])
    def test_ootc_matches_bptt(self, k):
        """OOTC gradient matches BPTT within 1e-5 relative error."""
        tau, T, f0, ux_ref, uy_ref = _make_problem(N=32, T=100)
        dJ_bptt, J_bptt, _, _ = bptt_gradient(tau, T, f0, ux_ref, uy_ref)
        dJ_ootc, J_ootc, _, _, _ = ootc_gradient(tau, T, f0, ux_ref, uy_ref, k=k)

        rel_err = abs(dJ_bptt - dJ_ootc) / max(abs(dJ_bptt), 1e-30)
        assert rel_err < 1e-5, f"rel_err={rel_err:.2e} >= 1e-5 for k={k}"
        assert abs(J_bptt - J_ootc) < 1e-12, "Objective should match exactly"

    def test_ootc_k_equals_T(self):
        """k=T (single segment) should still match BPTT."""
        tau, T, f0, ux_ref, uy_ref = _make_problem(N=16, T=50)
        dJ_bptt, J_bptt, _, _ = bptt_gradient(tau, T, f0, ux_ref, uy_ref)
        dJ_ootc, J_ootc, _, _, _ = ootc_gradient(tau, T, f0, ux_ref, uy_ref, k=T)

        rel_err = abs(dJ_bptt - dJ_ootc) / max(abs(dJ_bptt), 1e-30)
        assert rel_err < 1e-5

    def test_ootc_k_equals_1(self):
        """k=1 (checkpoint every step) should match BPTT."""
        tau, T, f0, ux_ref, uy_ref = _make_problem(N=16, T=30)
        dJ_bptt, J_bptt, _, _ = bptt_gradient(tau, T, f0, ux_ref, uy_ref)
        dJ_ootc, J_ootc, _, _, _ = ootc_gradient(tau, T, f0, ux_ref, uy_ref, k=1)

        rel_err = abs(dJ_bptt - dJ_ootc) / max(abs(dJ_bptt), 1e-30)
        assert rel_err < 1e-5


class TestOOTCMemory:
    """OOTC memory scaling tests."""

    def test_ootc_uses_less_memory_than_bptt(self):
        """OOTC peak memory should be significantly less than BPTT."""
        tau, T, f0, ux_ref, uy_ref = _make_problem(N=32, T=200)

        import tracemalloc
        tracemalloc.start()
        bptt_gradient(tau, T, f0, ux_ref, uy_ref)
        _, peak_bptt = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        ootc_gradient(tau, T, f0, ux_ref, uy_ref, k=20)
        _, peak_ootc = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # OOTC should use at most 20% of BPTT memory for k=T/10
        assert peak_ootc < 0.25 * peak_bptt, (
            f"OOTC memory {peak_oott/1024**2:.1f} MiB should be < 25% of "
            f"BPTT {peak_bptt/1024**2:.1f} MiB"
        )


class TestOOTCDeterminism:
    """OOTC should be deterministic."""

    def test_ootc_deterministic(self):
        """Two OOTC runs give identical results."""
        tau, T, f0, ux_ref, uy_ref = _make_problem(N=32, T=100)
        dJ1, J1, _, _, _ = ootc_gradient(tau, T, f0, ux_ref, uy_ref, k=25)
        dJ2, J2, _, _, _ = ootc_gradient(tau, T, f0, ux_ref, uy_ref, k=25)

        assert dJ1 == dJ2, "OOTC gradient should be deterministic"
        assert J1 == J2, "OOTC objective should be deterministic"


class TestOOTCCheckpointFile:
    """Checkpoint file handling."""

    def test_ootc_custom_ckpt_path(self):
        """OOTC writes checkpoint to specified path."""
        tau, T, f0, ux_ref, uy_ref = _make_problem(N=16, T=50)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test.ckpt.npy")
            dJ, J, _, n_ckpt, ret_path = ootc_gradient(
                tau, T, f0, ux_ref, uy_ref, k=10, ckpt_path=ckpt_path
            )
            assert os.path.exists(ckpt_path), "Checkpoint file should exist"
            assert ret_path == ckpt_path
            ckpt = np.load(ckpt_path)
            assert ckpt.shape[0] == n_ckpt

    def test_ootc_temp_ckpt_cleaned(self):
        """Temp checkpoint file is created when no path given."""
        tau, T, f0, ux_ref, uy_ref = _make_problem(N=16, T=50)
        dJ, J, _, n_ckpt, ckpt_path = ootc_gradient(
            tau, T, f0, ux_ref, uy_ref, k=10
        )
        assert os.path.exists(ckpt_path), "Temp checkpoint should exist"
        assert ckpt_path.endswith(".ckpt.npy")
        # Clean up
        os.unlink(ckpt_path)
