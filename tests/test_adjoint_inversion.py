"""Tests for the long-horizon tau inversion driver (adjoint_ootc.inversion).

Context: GAP-PP-3-adjoint-ootc's gradient gate was met, but its inversion
driver (hand-tuned steepest descent) stalled at |d tau|/tau = 20 % and was
recorded as UNFINISHED.  These tests pin the replacement driver:

1. it is importable from the *installed* package (the kernels used to live
   in a nested ``TensorLBM/src`` shadow tree that the installed package
   could not reach);
2. it recovers tau to well below the 3 % accuracy bar;
3. it converges from several starting points, including from the
   non-convex (J concave) far field;
4. it reports honestly when the root is not bracketed instead of
   returning an unconverged iterate as success;
5. the OOTC gradient it is driven by agrees with a forward-only central
   difference;
6. memory stays O(k*N*Q) on a long horizon.
"""

from __future__ import annotations

import os
import tracemalloc

import numpy as np
import pytest

import tensorlbm.adjoint_ootc as adj
from tensorlbm.adjoint_ootc import (
    fd_gradient,
    initial_pulse,
    invert_tau,
    ootc_gradient,
    reference_from_tau,
    terminal_objective,
)

TAU_TRUE = 0.8


def _problem(N=16, T=150, tau_true=TAU_TRUE):
    f0 = initial_pulse(N, N, amp=0.05, sigma=float(N) / 8.0)
    ux_ref, uy_ref = reference_from_tau(tau_true, T, f0)
    return f0, ux_ref, uy_ref


class TestImportability:
    """The kernels must ship in the installable tree."""

    def test_package_is_the_src_tree(self):
        path = os.path.abspath(adj.__file__)
        assert os.path.join("src", "tensorlbm", "adjoint_ootc") in path, path
        # not the nested shadow copy
        assert "/TensorLBM/src/" not in path, path

    def test_public_api(self):
        for name in (
            "ootc_gradient",
            "bptt_gradient",
            "invert_tau",
            "terminal_objective",
            "reference_from_tau",
            "fd_gradient",
        ):
            assert hasattr(adj, name), name


class TestGradientConsistency:
    """The driver's gradient source is the OOTC adjoint -- check it."""

    @pytest.mark.parametrize("tau", [0.7, 1.0, 1.3])
    def test_ootc_gradient_matches_central_difference(self, tau):
        f0, ux_ref, uy_ref = _problem(N=16, T=100)
        dJ = ootc_gradient(tau, 100, f0, ux_ref, uy_ref, k=25)[0]
        g_fd = fd_gradient(tau, 100, f0, ux_ref, uy_ref, h=1e-5)
        rel = abs(dJ - g_fd) / max(abs(g_fd), 1e-30)
        assert rel < 1e-5, f"tau={tau}: OOTC {dJ:.6e} vs FD {g_fd:.6e} (rel {rel:.2e})"


class TestInversionConvergence:
    """tau must be recovered far below the 3 % accuracy bar."""

    def test_recovers_tau_from_above(self):
        f0, ux_ref, uy_ref = _problem()
        res = invert_tau(1.2, 150, f0, ux_ref, uy_ref, k=25)
        assert res["converged"], res["reason"]
        rel = abs(res["tau"] - TAU_TRUE) / TAU_TRUE
        assert rel < 1e-3, f"rel err {rel:.2e} (tau={res['tau']:.12f})"
        assert res["J"] < 1e-12

    def test_recovers_tau_from_below(self):
        f0, ux_ref, uy_ref = _problem()
        res = invert_tau(0.6, 150, f0, ux_ref, uy_ref, k=25)
        assert res["converged"], res["reason"]
        assert abs(res["tau"] - TAU_TRUE) / TAU_TRUE < 1e-3

    def test_converges_through_concave_region(self):
        """Starting at tau=1.35 J is concave locally; the bracket must hold."""
        f0, ux_ref, uy_ref = _problem()
        res = invert_tau(1.35, 150, f0, ux_ref, uy_ref, k=25)
        assert res["converged"], res["reason"]
        assert abs(res["tau"] - TAU_TRUE) / TAU_TRUE < 1e-3
        # the driver must have stayed inside the declared physical bounds
        lo, hi = res["bracket"]
        assert res["bounds_ok"] if "bounds_ok" in res else lo <= res["tau"] <= hi

    def test_gradient_vanishes_at_solution(self):
        f0, ux_ref, uy_ref = _problem()
        res = invert_tau(1.2, 150, f0, ux_ref, uy_ref, k=25)
        scale = max(abs(res["history"][0]["dJ_dtau"]), 1e-300)
        assert abs(res["dJ_dtau"]) / scale < 1e-6


class TestHonestFailure:
    """A driver that cannot converge must say so."""

    def test_unbracketed_bounds_are_reported(self):
        f0, ux_ref, uy_ref = _problem()
        # g > 0 everywhere here (both ends above the minimiser): no root
        res = invert_tau(1.2, 150, f0, ux_ref, uy_ref, k=25, bounds=(0.9, 1.4))
        assert res["converged"] is False
        assert "not bracketed" in res["reason"]

    def test_tau0_outside_bounds_rejected(self):
        f0, ux_ref, uy_ref = _problem()
        with pytest.raises(ValueError):
            invert_tau(1.5, 150, f0, ux_ref, uy_ref, bounds=(0.6, 1.4))


class TestDriverLogic:
    """Driver mechanics, exercised with a synthetic (cheap) gradient."""

    def test_injected_analytic_gradient(self):
        """g(tau) = 2 (tau - 0.9)  =>  solution tau = 0.9."""
        calls = []

        def grad_fn(tau, T, f0, ux_ref, uy_ref, k):
            calls.append(tau)
            return 2.0 * (tau - 0.9), (tau - 0.9) ** 2

        f0, ux_ref, uy_ref = _problem()
        res = invert_tau(
            1.3, 150, f0, ux_ref, uy_ref, grad_fn=grad_fn, tol_rel_grad=1e-10, tol_dtau=1e-12
        )
        assert res["converged"], res["reason"]
        assert abs(res["tau"] - 0.9) < 1e-10
        assert res["grad_source"] == "custom"
        assert len(calls) == res["nfev"]

    def test_negative_curvature_cannot_push_iterate_out_of_bounds(self):
        """g with dg/dtau < 0 at the start: a bare Newton step exits bounds."""

        def grad_fn(tau, T, f0, ux_ref, uy_ref, k):
            g = (tau - 1.0) * (1.5 - tau)  # root at 1.0, g' < 0 for tau > 1.25
            return g, 0.5 * g * g

        f0, ux_ref, uy_ref = _problem()
        res = invert_tau(
            1.3, 150, f0, ux_ref, uy_ref, grad_fn=grad_fn, bounds=(0.9, 1.45), tol_rel_grad=1e-10
        )
        assert res["converged"], res["reason"]
        assert abs(res["tau"] - 1.0) < 1e-8
        assert all(0.9 <= h["tau"] <= 1.45 for h in res["history"])

    def test_iteration_budget_exhaustion_is_not_success(self):
        """A one-step budget must report converged=False, not a silent pass."""
        f0, ux_ref, uy_ref = _problem()
        res = invert_tau(1.2, 150, f0, ux_ref, uy_ref, k=25, max_iter=1)
        assert res["converged"] is False
        assert res["reason"] == "iterations exhausted"


class TestMemory:
    """The inversion inherits the OOTC memory bound (independent of T)."""

    def test_peak_memory_independent_of_horizon(self):
        peaks = {}
        for T in (100, 400):
            f0, ux_ref, uy_ref = _problem(N=16, T=T)
            tracemalloc.start()
            ootc_gradient(1.1, T, f0, ux_ref, uy_ref, k=25)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks[T] = peak
        # 4x the horizon must not cost 4x the memory (segment-bounded)
        assert peaks[400] < 1.6 * peaks[100], peaks

    def test_objective_matches_oootc(self):
        f0, ux_ref, uy_ref = _problem(N=16, T=100)
        J_ootc = ootc_gradient(1.1, 100, f0, ux_ref, uy_ref, k=25)[1]
        J_fwd = terminal_objective(1.1, 100, f0, ux_ref, uy_ref)
        assert np.isclose(J_ootc, J_fwd, rtol=0, atol=1e-16)
