"""P269W2-N04 - linear stability and transition sensitivity of the D2Q9 channel operator.

Acceptance criteria exercised here (see ``docs/P269W2_N04_lattice_shear_stability.md``):

* ``A1`` uniform-flow Stokes damping vs ``-nu k^2`` (relative error <= 1e-3);
* ``A2`` sign of the growth rate vs the Orr-Sommerfeld benchmark;
* ``A4`` ``A -> 0`` linear limit of the perturbation amplitude;
* ``A5`` this file (>= 10 tests);
* plus the audit chain that makes the numbers trustworthy: numpy collision ==
  library collision, analytic operator == finite-difference Jacobian of the real
  grid step, time march == eigenvalue, OS solver == Orszag (1971).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pytest

from tensorlbm.stability.channel_shear import (
    OPPOSITE,
    ChannelFlow,
    channel_mode_operator,
    collide_np,
    grid_channel_step,
    lattice_mode_spectrum,
    left_eigenvector,
    local_jacobian,
    mode_project,
    richardson_extrapolate,
    stokes_damping_benchmark,
    time_march_growth_rate,
)
from tensorlbm.stability.orr_sommerfeld import (
    ORSZAG_RE10000_ALPHA1,
    cheb,
    cheb_interp,
    orszag_benchmark_check,
    os_spectrum,
)
from tensorlbm.stability.von_neumann import CX, CY, Collision, equilibrium, spectrum


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _spectrum(ny: int, Re: float, alpha_key: tuple[int, int], kind: str):
    flow = ChannelFlow(ny=ny, Re=Re, u_max=0.05)
    m, k = alpha_key
    alpha = 2.0 * math.pi * m / (k * ny)
    return flow, alpha, lattice_mode_spectrum(flow, alpha, kind, n_cheb=64)


def _real_base(flow: ChannelFlow, nx: int) -> np.ndarray:
    rho = np.full((flow.ny, nx), 1.0)
    ux = np.tile(flow.base_velocity.reshape(-1, 1), (1, nx))
    uy = np.zeros((flow.ny, nx))
    return equilibrium(rho, ux, uy)


# ---------------------------------------------------------------------------
# OS reference
# ---------------------------------------------------------------------------


def test_chebyshev_differentiation_is_exact_for_polynomials():
    D, x = cheb(8)
    assert D.shape == (x.size, x.size), "Chebyshev operator must be square"
    for power in (1, 2, 3):
        exact = power * x ** (power - 1)
        assert np.max(np.abs(D @ x**power - exact)) < 1e-10


def test_orszag_benchmark_eigenvalue_reproduced():
    res = orszag_benchmark_check(64)
    assert res["rel_error_complex"] < 1e-6, res
    assert abs(res["c_computed"][1]) > 0.0


def test_os_critical_reynolds_sign_change():
    low = os_spectrum(1.02, 5000.0, 64)
    high = os_spectrum(1.02, 6500.0, 64)
    assert low.c_leading.imag < 0.0, "plane Poiseuille must be linearly stable below Re_c"
    assert high.c_leading.imag > 0.0, "Tollmien-Schlichting wave must grow above Re_c"


def test_cheb_interp_reproduces_smooth_function():
    D, x = cheb(24)
    f = np.exp(0.5 * x) * (1 - x**2)
    y = np.linspace(-0.95, 0.95, 21)
    assert np.max(np.abs(cheb_interp(f, x, y) - np.exp(0.5 * y) * (1 - y**2))) < 1e-10


# ---------------------------------------------------------------------------
# library consistency: numpy collision vs tensorlbm.solver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["bgk", "mrt", "rlbm"])
def test_numpy_collision_matches_library_collision(kind):
    torch = pytest.importorskip("torch")
    from tensorlbm.solver import collide_bgk, collide_mrt, collide_rlbm

    rng = np.random.default_rng(4)
    f = rng.random((9, 5, 7)) + 0.05
    f /= f.sum(axis=0)
    ref = {"bgk": collide_bgk, "mrt": collide_mrt, "rlbm": collide_rlbm}[kind]
    expected = ref(torch.tensor(f, dtype=torch.float64), 0.7).numpy()
    # 1e-8 tolerance: ``tensorlbm.d2q9.W`` stores the weights in float32, which
    # is the only difference between the two paths (BGK/MRT reach 1e-17).
    assert np.max(np.abs(collide_np(f, kind, 0.7) - expected)) < 1e-8


# ---------------------------------------------------------------------------
# analytic linear operator vs finite-difference Jacobian of the grid step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["bgk", "mrt", "rlbm"])
def test_analytic_operator_matches_fd_jacobian(kind):
    ny, nx, mode = 8, 24, 1
    alpha = 2.0 * math.pi * mode / nx
    flow = ChannelFlow(ny=ny, Re=1000.0, u_max=0.05)
    L = channel_mode_operator(flow, alpha, kind)
    f0 = _real_base(flow, nx)
    f0p = grid_channel_step(f0, kind, flow.tau)
    phase = np.exp(1j * alpha * np.arange(nx))
    eps = 1e-8
    worst = 0.0
    for j in range(9 * ny):
        g = np.zeros((9, ny), dtype=complex)
        g[j // ny, j % ny] = 1.0
        delta = 2.0 * np.real(g.reshape(9, ny, 1) * phase.reshape(1, 1, nx))
        fp = grid_channel_step(f0 + eps * delta, kind, flow.tau)
        fd = mode_project(fp - f0p, mode) / eps
        worst = max(worst, float(np.max(np.abs(fd.ravel() - L @ g.ravel()))))
    assert worst < 1e-7, f"|J_fd - L| = {worst:.2e}"


def test_operator_wall_rows_are_link_bounce_back():
    """Outgoing directions at the walls are replaced by the reflected population."""
    ny, nx, mode = 8, 24, 1
    alpha = 2.0 * math.pi * mode / nx
    flow = ChannelFlow(ny=ny, Re=1000.0, u_max=0.05)
    L = channel_mode_operator(flow, alpha, "bgk")
    jac = local_jacobian("bgk", flow.base_velocity[0], flow.tau)
    checked = 0
    for i in range(9):
        if CY[i] > 0:  # population arriving from the bottom wall at node y = 0
            row = i * ny
            expected = np.zeros(9 * ny, dtype=complex)
            expected[np.arange(9) * ny] = jac[OPPOSITE[i], :]
            assert np.allclose(L[row], expected), f"direction {i}"
            checked += 1
    assert checked == 3, "three directions (5, 2, 6) reach the bottom wall"


def test_channel_flow_reynolds_definition_is_consistent():
    flow = ChannelFlow(ny=64, Re=4000.0, u_max=0.05)
    assert flow.h == 32.0
    assert abs(flow.nu - 0.05 * 32.0 / 4000.0) < 1e-15
    assert abs(flow.tau - (0.5 + 3.0 * flow.nu)) < 1e-15
    assert flow.tau > 0.5
    assert abs(flow.base_velocity.max() - 0.05 * (1.0 - 1.0 / 64**2)) < 1e-6
    assert flow.base_velocity[0] == flow.base_velocity[-1]


# ---------------------------------------------------------------------------
# A1 - uniform flow Stokes damping
# ---------------------------------------------------------------------------


def test_a1_stokes_damping_von_neumann_vs_continuum():
    # resolved shear wave (n = 64 -> k = pi/32): the criterion is a k -> 0 statement
    res = stokes_damping_benchmark(n=64, mode=1, tau=0.8, n_steps=400)
    assert res["rel_error_von_neumann"] < 1e-3, res


def test_a1_stokes_damping_measured_vs_continuum():
    res = stokes_damping_benchmark(n=64, mode=1, tau=0.8, n_steps=2000)
    assert res["sigma_measured"] < 0.0
    assert res["rel_error_measured"] < 1e-3, res


def test_stokes_damping_deviation_scales_as_k_squared():
    """The residual of A1 is lattice dispersion of the shear mode (k^2/12)."""
    coarse = stokes_damping_benchmark(n=32, mode=1, tau=0.8, n_steps=400)
    fine = stokes_damping_benchmark(n=64, mode=1, tau=0.8, n_steps=400)
    ratio = coarse["rel_error_von_neumann"] / fine["rel_error_von_neumann"]
    assert 3.0 < ratio < 5.0, ratio  # k doubling -> factor 4 in k^2


# ---------------------------------------------------------------------------
# lattice spectrum vs Orr-Sommerfeld
# ---------------------------------------------------------------------------


def test_lattice_spectrum_selects_orr_sommerfeld_branch():
    flow, alpha, spec = _spectrum(32, 1000.0, (1, 3), "bgk")
    assert spec.overlap > 0.99, "OS branch must be identified by eigenfunction overlap"
    assert spec.sigma_nondim() < 0.0, "Re=1000 is subcritical: sigma must be negative"
    assert spec.sigma_nondim() == pytest.approx(spec.os_reference.sigma_leading, rel=0.2)


def test_least_damped_mode_is_acoustic_not_orr_sommerfeld():
    """Negative control for the mode-identification rule of the analysis."""
    flow, alpha, spec = _spectrum(32, 1000.0, (1, 3), "bgk")
    assert spec.index_least_damped != spec.selected
    assert spec.sigma_nondim(spec.index_least_damped) > spec.sigma_nondim()


def test_ny_convergence_towards_orr_sommerfeld():
    devs = []
    for ny in (32, 64):
        flow, alpha, spec = _spectrum(ny, 3000.0, (1, 3), "bgk")
        devs.append(abs(spec.sigma_nondim() - spec.os_reference.sigma_leading)
                    / abs(spec.os_reference.sigma_leading))
    assert devs[1] < devs[0], f"refinement must reduce the OS deviation, got {devs}"
    assert devs[1] < 0.25, devs


def test_supercritical_growth_rate_is_positive():
    flow, alpha, spec = _spectrum(32, 8000.0, (1, 3), "bgk")
    assert spec.sigma_nondim() > 0.0, "Re=8000, alpha=pi/3 must be unstable (TS wave)"
    assert spec.os_reference.sigma_leading > 0.0


# ---------------------------------------------------------------------------
# A4 - A -> 0 linear limit; cross audit of the time march
# ---------------------------------------------------------------------------


def test_time_march_matches_eigenvalue_and_linear_limit():
    ny = 16
    flow = ChannelFlow(ny=ny, Re=2000.0, u_max=0.05)
    alpha = 2.0 * math.pi / (6 * ny)
    spec = lattice_mode_spectrum(flow, alpha, "rlbm", n_cheb=64)
    L = channel_mode_operator(flow, alpha, "rlbm")
    w = left_eigenvector(L, spec.lam[spec.selected]).reshape(9, ny)
    sigmas = []
    # amplitudes stay well inside the linear window: the mode-selective readout
    # has a contamination floor set by the accuracy of the left eigenvector, which
    # is why the march is kept short (5000 steps, ~1.5 e-folds) in the test.
    for amp in (1e-4, 5e-5):
        res = time_march_growth_rate(
            flow, alpha, "rlbm", k_box=6, mode=1, n_steps=5000,
            amplitude=amp, projector=w,
        )
        assert res["diverged_at"] is None
        sigmas.append(res["sigma_nondim"])
        assert abs(res["sigma_nondim"] - spec.sigma_nondim()) / abs(spec.sigma_nondim()) < 0.05
    # A -> 0 : the growth rate must not depend on the injection amplitude
    assert abs(sigmas[0] - sigmas[1]) / abs(sigmas[1]) < 0.05, sigmas


# ---------------------------------------------------------------------------
# A3 machinery
# ---------------------------------------------------------------------------


def test_richardson_extrapolation_recovers_known_order():
    exact = 0.37
    levels = (64, 128, 256)
    vals = [exact + 0.4 / lv for lv in levels]  # first order convergence
    res = richardson_extrapolate(vals, levels)
    assert res["observed_order"] == pytest.approx(1.0, abs=0.05)
    assert res["limit"] == pytest.approx(exact, abs=1e-9)
    assert res["residual_finest_rel"] == pytest.approx(0.4 / 256 / abs(exact), rel=1e-6)


def test_stokes_limit_is_consistent_with_von_neumann_shear_branch():
    coll = Collision(kind="bgk", tau=1.2)
    spec = spectrum(0.2, 0.0, coll)
    shear = spec.mu[spec.index_of("shear")]
    sigma = np.log(abs(shear))
    assert abs(sigma + coll.viscosity * 0.2**2) / (coll.viscosity * 0.2**2) < 2e-2