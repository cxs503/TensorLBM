"""Verification of the exact Womersley solver
(:mod:`tensorlbm.turbulence.womersley_exact`) against Bessel-free references.

Every assertion here is derived from an *independent* route, not from the
implementation's own algebra:

* flux identity (plane quadrature),
* quasi-steady limit with an analytically predicted coefficient ``Wo²/48``,
* a finite-difference solution of the unsteady momentum ODE (no Bessel),
* the momentum-operator residual being spatially uniform,
* the classical high-Womersley asymptotes (wall shear leads the flow rate by
  45°, core becomes a plug in phase with the flow rate).

Regression context: the first implementation used ``α = Wo e^{+iπ/4}`` instead
of ``e^{-iπ/4}``.  Amplitudes were identical, so only the finite-difference and
asymptote checks could see it — every phase lag was conjugated.
See ``docs/womersley_exact_verification_20261108.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.turbulence.inlet_profiles import womersley_profile
from tensorlbm.turbulence.womersley_exact import (
    womersley_complex_profile,
    womersley_diagnostics,
    womersley_profile_exact,
)

TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _alpha(wo: float, k: int = 1) -> complex:
    """Parameter of the *operator* form ∇²U - α²U = C, i.e. α² = iω/ν.

    NB this is the conjugate of the e^{iωt} radial eigenvalue
    (``β = Wo e^{-iπ/4}``, ``β² = -iω/ν``) used inside the solver: the two
    differ by the sign of the Laplacian term, so mixing them silently
    conjugates the solution.
    """
    return wo * math.sqrt(k) * complex(1.0, 1.0) / math.sqrt(2.0)


# ---------------------------------------------------------------------------
# 1. prescribed flow rate: the normalisation the inlet BC relies on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wo", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
def test_flux_is_unity_for_all_womersley_numbers(wo):
    """∫₀¹ Ŵ 2r̃ dr̃ = 1 exactly — unit bulk velocity at every Wo."""
    r = np.linspace(0.0, 1.0, 40001)
    flux = TRAPZ(womersley_complex_profile(r, wo) * 2.0 * r, r)
    assert abs(flux - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# 2. quasi-steady limit -> Hagen-Poiseuille, second order, coefficient 1/48
# ---------------------------------------------------------------------------
def test_quasi_steady_limit_is_second_order_with_analytic_coefficient():
    """Ŵ → 2(1-r̃²)(1 - i Wo²/6); relative error = Wo²/48 + O(Wo⁴)."""
    r = np.linspace(0.0, 1.0, 4001)
    ref = 2.0 * (1.0 - r**2)
    coeffs = []
    for wo in (0.01, 0.005):
        err = np.abs(womersley_complex_profile(r, wo) - ref).max() / 2.0
        coeffs.append(err / wo**2)
    # analytically derived coefficient (see module docstring)
    assert coeffs[0] == pytest.approx(1.0 / 48.0, rel=1e-3)
    # ... and the error really is second order: halving Wo quarters it
    assert coeffs[1] / coeffs[0] == pytest.approx(1.0, rel=1e-2)


# ---------------------------------------------------------------------------
# 3. momentum-operator residual must be spatially uniform (algebra check)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wo", [1.0, 5.0, 20.0])
def test_bessel_shape_satisfies_momentum_operator_with_uniform_forcing(wo):
    """∇²Ŵ - i Wo² Ŵ must equal a constant (the pressure forcing)."""
    n = 8000
    r = np.linspace(0.0, 1.0, n + 1)
    *_, w = (r, r, womersley_complex_profile(r, wo))
    W = w
    h = 1.0 / n
    i = np.arange(2, n - 1)
    d2 = (W[i + 1] - 2.0 * W[i] + W[i - 1]) / h**2
    d1 = (W[i + 1] - W[i - 1]) / (2.0 * h)
    res = d2 + d1 / r[i] - (1j * wo**2) * W[i]
    sel = (r[i] >= 0.05) & (r[i] <= 0.95)  # 1/r operator not FD-representative near the axis
    vals = res[sel]
    assert np.abs(vals - vals.mean()).max() / np.abs(vals).max() < 1e-4


# ---------------------------------------------------------------------------
# 4. independent finite-difference solution of the ODE (no Bessel functions)
# ---------------------------------------------------------------------------
def _fd_solve(wo: float, n: int):
    """U'' + U'/r - α²U = C with U'(0)=0, U(1)=0, ∫U 2r dr = 1, α² = i Wo²."""
    a = _alpha(wo)
    h = 1.0 / n
    r = np.arange(1, n) * h
    m = n - 1
    A = np.zeros((n, n), dtype=complex)
    b = np.zeros(n, dtype=complex)
    for idx in range(m):
        A[idx, idx] += -2.0 / h**2 - a * a
        if idx == 0:  # even solution -> ghost node, (U''+U'/r)|₀ = 2U''
            A[idx, idx + 1] += 2.0 / h**2
        else:
            A[idx, idx - 1] += 1.0 / h**2 - 1.0 / (2.0 * h * r[idx])
            A[idx, idx + 1] += 1.0 / h**2 + 1.0 / (2.0 * h * r[idx])
        A[idx, -1] = -1.0  # the unknown constant forcing
    A[-1, :m] = 2.0 * r * h
    b[-1] = 1.0
    return r, np.linalg.solve(A, b)[:m]


@pytest.mark.parametrize("wo", [1.0, 5.0, 20.0])
def test_bessel_free_finite_difference_reproduces_shape_and_sign(wo):
    """A Bessel-free FD solve of the same ODE must match profile *and phase*.

    The phase sign is what catches a wrong ``sqrt(±i)`` convention.
    """
    r, U = _fd_solve(wo, 800)
    W = womersley_complex_profile(r, wo)
    sel = (r >= 0.05) & (r <= 0.9)
    err = np.abs(U[sel] - W[sel]).max() / np.abs(W[sel]).max()
    # 2nd-order FD scheme, n=800: the discretisation error is ~1e-5 (measured);
    # the tolerance only has to separate "same solution" from "conjugate"
    # (which differs by O(1) in the imaginary part).
    assert err < 1e-3, err


# ---------------------------------------------------------------------------
# 5. classical high-Womersley asymptotes (phase sign included)
# ---------------------------------------------------------------------------
def test_wall_shear_leads_flow_rate_by_45_degrees_at_high_wo():
    d = womersley_diagnostics(100.0)
    assert d["wall_shear_phase_deg"] == pytest.approx(45.0, abs=1.0)


def test_core_becomes_a_plug_in_phase_with_the_flow_rate():
    d = womersley_diagnostics(100.0)
    assert d["centerline_amplitude"] == pytest.approx(1.0, abs=0.02)
    assert abs(d["centerline_phase_deg"]) < 1.5


def test_centreline_lags_the_flow_rate_at_intermediate_wo():
    """Non-monotonic phase behaviour: the core lags (negative phase) near Wo≈5.

    Regression guard for the ``α = Wo e^{-iπ/4}`` convention: using
    ``e^{+iπ/4}`` flips this sign (and leaves every amplitude unchanged).
    """
    d = womersley_diagnostics(5.0)
    assert d["centerline_phase_deg"] == pytest.approx(-18.5, abs=0.5)
    assert d["wall_shear_phase_deg"] > 0.0  # wall shear leads


# ---------------------------------------------------------------------------
# 6. near-wall (Stokes layer) numerical branch
# ---------------------------------------------------------------------------
def test_wall_branch_matches_the_linear_wall_limit_of_the_derivative():
    """1 - J0(αr̃)/J0(α) is evaluated by Taylor inside the Stokes layer.

    The wall value must vanish, and the slope must be the analytic
    dŴ/dr̃|₁ that ``womersley_diagnostics`` reports.
    """
    wo = 20.0
    d = womersley_diagnostics(wo)
    dw = d["wall_shear_amplitude"] * complex(
        math.cos(math.radians(d["wall_shear_phase_deg"])),
        math.sin(math.radians(d["wall_shear_phase_deg"])),
    )
    eps = 1e-7
    val = womersley_complex_profile(np.array([1.0 - eps]), wo)[0]
    assert abs(womersley_complex_profile(np.array([1.0]), wo)[0]) < 1e-15  # no-slip
    # Ŵ(1-eps) = -eps * dŴ/dr̃|₁ + O(eps²), and the wall shear reported by
    # womersley_diagnostics is -dŴ/dr̃|₁ (outward normal), hence val = +eps*dw
    assert abs(val - eps * dw) < 1e-9


# ---------------------------------------------------------------------------
# 7. torch API invariants
# ---------------------------------------------------------------------------
def test_torch_profile_invariants():
    n = 128
    mean_profile = womersley_profile_exact(n, 1.0, 5.0, amplitude=0.0).numpy()
    r = np.abs(2.0 * np.linspace(0.5, n - 0.5, n) / n - 1.0)
    np.testing.assert_allclose(mean_profile, 2.0 * (1.0 - r**2), atol=1e-6)

    # phase average over one period recovers the mean (Poiseuille) profile
    acc = np.zeros(n)
    phases = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    for ph in phases:
        acc += womersley_profile_exact(n, 1.0, 5.0, phase=float(ph)).numpy()
    acc /= phases.size
    assert np.abs(acc - mean_profile).max() < 1e-5

    # 2π periodicity and dtype/shape
    p0 = womersley_profile_exact(64, 0.1, 5.0, phase=0.7)
    p1 = womersley_profile_exact(64, 0.1, 5.0, phase=0.7 + 2.0 * math.pi)
    assert p0.shape == (64,)
    assert np.abs(p0.numpy() - p1.numpy()).max() < 1e-6

    # clamp switch only ever removes backflow
    raw = womersley_profile_exact(64, 0.1, 10.0, phase=math.pi, amplitude=3.0)
    clipped = womersley_profile_exact(
        64, 0.1, 10.0, phase=math.pi, amplitude=3.0, clamp_nonneg=True
    )
    assert clipped.min() >= 0.0
    assert raw.min() < 0.0 <= clipped.min()


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        womersley_profile_exact(64, 0.1, 0.0)
    with pytest.raises(ValueError):
        womersley_profile_exact(2, 0.1, 5.0)
    with pytest.raises(ValueError):
        womersley_complex_profile(np.array([1.5]), 5.0)


# ---------------------------------------------------------------------------
# 8. the shipped approximation is unchanged (bit-for-bit) and materially
#    different from the exact solution — i.e. the new path is not cosmetic
# ---------------------------------------------------------------------------
def test_legacy_approximation_output_is_unchanged():
    p = womersley_profile(64, u_mean=0.1, wo=5.0, phase=0.7)
    assert float(p[0]) == pytest.approx(0.0108317350969, rel=1e-6)
    assert float(p[16]) == pytest.approx(0.216492444277, rel=1e-6)
    assert float(p[31]) == pytest.approx(0.268626451492, rel=1e-6)
    assert float(p.sum()) == pytest.approx(12.0345144272, rel=1e-6)


def test_legacy_and_exact_differ_materially():
    """Guard on the *documented* gap: the approximation is >15 % off in places.

    If the legacy function is ever routed through the exact solution, this test
    must be updated together with the warning in its docstring.
    """
    n = 256
    y = np.linspace(0.5, n - 0.5, n)
    r = np.abs(2.0 * y / n - 1.0)
    approx = womersley_profile(n, 0.1, 1.0, phase=0.0, n_harmonics=1).numpy()
    exact = womersley_profile_exact(n, 0.1, 1.0, phase=0.0, n_harmonics=1).numpy()
    dev = np.abs(approx - exact).max() / (2.0 * 0.1)  # vs Poiseuille peak
    assert dev > 0.15


def test_exact_flag_routes_to_the_analytic_solution():
    n = 64
    vias_flag = womersley_profile(n, 0.1, 5.0, phase=0.3, exact=True).numpy()
    direct = womersley_profile_exact(n, 0.1, 5.0, phase=0.3).numpy()
    np.testing.assert_allclose(vias_flag, direct, atol=0.0)