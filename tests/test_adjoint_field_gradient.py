"""Tests for the adjoint field dJ/df^0 returned by ``return_field=True``.

The scalar dJ/dtau agreement between OOTC and BPTT is *not* a sufficient
check (both share the same formulas: a wrong formula is wrong in both,
which is how defects D1/D2 survived).  These tests validate the full
costate field at n = 0 against an independent reference:

  * a brute-force column-by-column central-difference Jacobian of
    J(f0) (small N, so the O(N*Q) columns are affordable), and
  * a directional derivative <dJ/df0, v> along a random direction,
    which also confirms that the *streaming* of the field matters.

They also pin the backwards-compatible default tuple shapes.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TensorLBM", "src"))

from tensorlbm.adjoint.lbm2d import initial_pulse, step, macro
from tensorlbm.adjoint.bptt import bptt_gradient
from tensorlbm.adjoint.ootc import ootc_gradient


def _problem(N=6, T=4, tau=1.0):
    f0 = initial_pulse(N, N, amp=0.05, sigma=float(N) / 8.0)
    f = f0.copy()
    for _ in range(T):
        f = step(f, tau)
    rho_t, ux_t, uy_t = macro(f)
    return tau, T, f0, ux_t * 0.9, uy_t * 0.9


def _J(f0, T, tau, ux_ref, uy_ref):
    f = f0.copy()
    for _ in range(T):
        f = step(f, tau)
    _, ux, uy = macro(f)
    return 0.5 * float(np.sum((ux - ux_ref) ** 2 + (uy - uy_ref) ** 2))


class TestFieldAgainstBruteForce:
    def test_bptt_field_matches_bruteforce_jacobian(self):
        """dJ/df0 from BPTT equals the column-by-column FD Jacobian."""
        tau, T, f0, ux_ref, uy_ref = _problem(N=6, T=4)
        _, _, _, _, g = bptt_gradient(tau, T, f0, ux_ref, uy_ref,
                                      return_field=True)
        h = 1e-6
        g_ref = np.zeros_like(f0)
        for idx in np.ndindex(*f0.shape):
            e = np.zeros_like(f0)
            e[idx] = h
            g_ref[idx] = (_J(f0 + e, T, tau, ux_ref, uy_ref)
                          - _J(f0 - e, T, tau, ux_ref, uy_ref)) / (2 * h)
        assert np.abs(g - g_ref).max() < 1e-9, (
            f"max err {np.abs(g - g_ref).max():.3e}")

    def test_ootc_field_matches_bptt(self):
        """OOTC returns the same dJ/df0 as the BPTT reference."""
        tau, T, f0, ux_ref, uy_ref = _problem(N=8, T=11)
        _, _, _, _, g_bptt = bptt_gradient(tau, T, f0, ux_ref, uy_ref,
                                           return_field=True)
        _, _, _, _, _, g_ootc = ootc_gradient(tau, T, f0, ux_ref, uy_ref,
                                              k=4, return_field=True)
        scale = np.abs(g_bptt).max()
        assert np.abs(g_ootc - g_bptt).max() <= 1e-12 * max(scale, 1e-30)

    @pytest.mark.parametrize("h", [1e-5, 1e-6])
    def test_directional_derivative(self, h):
        """<dJ/df0, v> matches a central difference along v."""
        tau, T, f0, ux_ref, uy_ref = _problem(N=8, T=8)
        rng = np.random.default_rng(7)
        v = rng.standard_normal(f0.shape)
        v *= np.sqrt((f0 * f0).sum()) / np.sqrt((v * v).sum())
        _, _, _, _, g = bptt_gradient(tau, T, f0, ux_ref, uy_ref,
                                      return_field=True)
        adj = float((g * v).sum())
        fd = (_J(f0 + h * v, T, tau, ux_ref, uy_ref)
              - _J(f0 - h * v, T, tau, ux_ref, uy_ref)) / (2 * h)
        rel = abs(adj - fd) / abs(fd)
        assert rel < 1e-8, f"rel={rel:.3e} (h={h})"


class TestReturnFieldCompatibility:
    def test_default_tuples_unchanged(self):
        """Default calls keep the historical tuple lengths."""
        tau, T, f0, ux_ref, uy_ref = _problem(N=6, T=5)
        assert len(bptt_gradient(tau, T, f0, ux_ref, uy_ref)) == 4
        assert len(ootc_gradient(tau, T, f0, ux_ref, uy_ref, k=2)) == 5

    def test_field_does_not_change_scalar_gradient(self):
        """Asking for the field must not perturb dJ/dtau / J."""
        tau, T, f0, ux_ref, uy_ref = _problem(N=8, T=13)
        d1, J1, _, _ = bptt_gradient(tau, T, f0, ux_ref, uy_ref)
        d2, J2, _, _, g = bptt_gradient(tau, T, f0, ux_ref, uy_ref,
                                        return_field=True)
        assert d1 == d2 and J1 == J2
        assert g.shape == f0.shape
        assert np.isfinite(g).all()

class TestRollShiftPortability:
    """The streaming/anti-streaming rolls must not rely on numpy accepting
    float shift values (numpy>=2 on some builds raises TypeError, which
    broke every adjoint run on the cluster nodes -- defect D5)."""

    def test_forward_and_adjoint_streaming_are_integer_shifts(self):
        from tensorlbm.adjoint.lbm2d import EX, EY, stream
        from tensorlbm.adjoint.bptt import anti_stream
        # every lattice velocity must be integral, so int() loses nothing
        assert np.array_equal(EX, np.round(EX))
        assert np.array_equal(EY, np.round(EY))
        rng = np.random.default_rng(3)
        g = rng.standard_normal((9, 5, 7))
        ref = np.empty_like(g)
        for i in range(9):
            ref[i] = np.roll(g[i], shift=(int(EX[i]), int(EY[i])), axis=(0, 1))
        assert np.array_equal(stream(g), ref)
        # anti_stream must be the exact inverse shift of the forward one
        back = np.empty_like(g)
        for i in range(9):
            back[i] = np.roll(stream(g)[i], shift=(-int(EX[i]), -int(EY[i])),
                              axis=(0, 1))
        assert np.array_equal(anti_stream(stream(g)), back)
