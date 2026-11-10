"""Analytic anchors for the forced-sway sloshing verification surface (P88W1-N01).

Two anchors are used by the campaign, both closed form and both *independent of
the lattice solver*:

1. **Standing-wave dispersion** (re-used from
   :mod:`tensorlbm.sloshing_resonance.theory`, Abramson 1966 NASA SP-106 §2 /
   Faltinsen & Timokha 2009 eq. 4.13)::

       k_n = n pi / L,   omega_n^2 = g k_n tanh(k_n h),   f_n = omega_n / 2 pi

2. **Linear forced-sway response (RAO)** — derived here, because the node asks
   for a *forced* frequency sweep and no existing module carries the transfer
   function.

   With the tank accelerating as ``X(t) = X0 sin(omega t)`` (tank frame), the
   linearised free-surface problem for the modal coefficients
   ``eta(x,t) = sum_n a_n(t) cos(k_n x)`` gives, after projecting the dynamic
   free-surface condition ``dphi/dt + g eta = -Xddot x`` on the Neumann modes
   ``cos(k_n x)`` (the only non-vanishing projection comes from the boundary
   term, ``int_0^L x cos(k_n x) dx = ((1)^n - 1) / k_n^2``)::

       addot_n + omega_n^2 a_n = F_n,
       F_n = -(2 Xddot / L) * ((-1)^n - 1) * tanh(k_n h) / k_n

   so **horizontal translation excites odd modes only** (``(-1)^n - 1`` vanishes
   for every even ``n``) — the parity rule that this module also *measures*.
   For odd ``n`` and ``Xddot = -X0 omega^2 sin(omega t)``::

       |F_n| = 4 X0 omega^2 tanh(k_n h) / (L k_n)

   and, with the modal damping ratio ``zeta_n`` measured from a free-decay run::

       a_n(omega) = F_n / (omega_n^2 - omega^2 + 2 i zeta_n omega_n omega)

   **Static check (exact):** for ``omega -> 0`` the odd-mode sum reproduces the
   rigid-body tilt ``eta(x) = -Xddot (x - L/2) / g`` of a slowly accelerated
   tank (``tests/test_sloshing_forced_p88w1.py`` asserts this to 1e-12), which
   is the reason the parity/coefficient bookkeeping above can be trusted.

3. **Wall pressure and impact bounds** — re-used from
   :mod:`tensorlbm.sloshing_resonance.theory`: the linear dynamic wall pressure
   ``p(y) = rho g a cosh(k(y+h)) / cosh(k h)`` (Faltinsen & Timokha 2009 eq.
   4.14), the Wagner stagnation bound ``p = 0.5 rho V^2`` and the bore-reflection
   / hydrostatic run-up bounds.

All functions are pure (no state, no I/O) so they can be exercised directly by
the pytest suite.
"""

from __future__ import annotations

import cmath
import math

from ..sloshing_resonance.theory import (  # re-exported, single source of truth
    abramson_angular_frequency,
    abramson_brief_formula,
    abramson_frequency,
    abramson_period,
    abramson_wave_number,
    bore_reflection_pressure,
    hydrostatic_runup_pressure,
    linear_wall_pressure_amplitude,
    linear_wall_pressure_bottom_ratio,
    stagnation_impact_pressure,
)

__all__ = [
    "abramson_angular_frequency",
    "abramson_brief_formula",
    "abramson_frequency",
    "abramson_period",
    "abramson_wave_number",
    "bore_reflection_pressure",
    "hydrostatic_runup_pressure",
    "linear_wall_pressure_amplitude",
    "linear_wall_pressure_bottom_ratio",
    "stagnation_impact_pressure",
    "mode_excitation_coefficient",
    "modal_forcing_amplitude",
    "steady_modal_amplitude",
    "steady_wall_amplitude",
    "steady_pressure_amplitude",
    "resonance_ratio",
    "peak_frequency",
    "static_tilt_elevation",
    "odd_mode_fraction",
    "gauge_height_from_swl",
]

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# excitation parity / modal forcing
# ---------------------------------------------------------------------------


def mode_excitation_coefficient(mode: int) -> float:
    """``(-1)^n - 1``: 0 for even modes (not excited), -2 for odd modes.

    This is the whole parity content of horizontal translation excitation; the
    campaign measures the predicted null at the second natural frequency.
    """
    if mode < 1:
        raise ValueError("mode must be >= 1")
    return float(((-1) ** mode) - 1)


def modal_forcing_amplitude(
    mode: int, sway_amplitude: float, omega: float, depth: float, length: float
) -> float:
    """``|F_n| = 4 X0 omega^2 tanh(k_n h) / (L k_n)`` for odd ``n``, else 0.0."""
    k = abramson_wave_number(mode, length)
    coef = abs(mode_excitation_coefficient(mode))
    if coef == 0.0:
        return 0.0
    # F_n = -(2 Xddot / L) ((-1)^n - 1) tanh(k_n h) / k_n  ->  |F_n| = 2 |coef| X0 w^2 tanh/(L k)
    return 2.0 * coef * sway_amplitude * omega * omega * math.tanh(k * depth) / (length * k)


def steady_modal_amplitude(
    mode: int,
    sway_amplitude: float,
    omega: float,
    zeta: float,
    depth: float,
    length: float,
    g: float = 9.81,
) -> complex:
    """Complex steady-state modal amplitude ``a_n`` (metres).

    ``zeta`` is the modal damping ratio of mode ``n`` (from free decay).  The
    returned complex value carries the phase lag behind the excitation
    ``Xddot(t) = -X0 omega^2 sin(omega t)``.
    """
    if zeta < 0.0:
        raise ValueError("zeta must be non-negative")
    force = 2.0 * abs(mode_excitation_coefficient(mode)) * sway_amplitude * omega * omega
    if force == 0.0:
        return 0.0 + 0.0j
    k = abramson_wave_number(mode, length)
    fn = force * math.tanh(k * depth) / (length * k)
    wn = abramson_angular_frequency(mode, depth, length, g)
    denom = complex(wn * wn - omega * omega, 2.0 * zeta * wn * omega)
    if abs(denom) == 0.0:
        raise ValueError(
            "undamped resonance: the steady-state amplitude is unbounded "
            "(use finite_time_modal_response for a finite-time reference)"
        )
    return fn / denom


def steady_wall_amplitude(
    sway_amplitude: float,
    omega: float,
    zeta: float,
    depth: float,
    length: float,
    g: float = 9.81,
    modes: int = 5,
) -> complex:
    """Complex amplitude of ``eta`` at the left wall (``x = 0``), modes 1..``modes``.

    At ``x = 0`` every ``cos(k_n x)`` equals 1, so the wall elevation is the plain
    sum of the modal amplitudes (odd modes only, even ones have zero forcing).
    """
    total = 0.0 + 0.0j
    for n in range(1, int(modes) + 1):
        total += steady_modal_amplitude(n, sway_amplitude, omega, zeta, depth, length, g)
    return total


def steady_pressure_amplitude(
    sway_amplitude: float,
    omega: float,
    zeta: float,
    y_from_swl: float,
    depth: float,
    length: float,
    g: float = 9.81,
    rho: float = 1000.0,
    modes: int = 5,
) -> float:
    """Linear dynamic wall-pressure amplitude (Pa) at height ``y_from_swl``.

    Sum of the modal contributions ``rho g a_n cosh(k_n(y+h))/cosh(k_n h)``; the
    closed form of anchor 3 evaluated with the *forced-response* modal
    amplitudes of anchor 2 (rather than a decay amplitude).
    """
    total = 0.0 + 0.0j
    for n in range(1, int(modes) + 1):
        a = steady_modal_amplitude(n, sway_amplitude, omega, zeta, depth, length, g)
        if a == 0.0:
            continue
        total += a * linear_wall_pressure_amplitude(
            n, 1.0, y_from_swl, depth, length, g, rho
        )
    return abs(total)


# ---------------------------------------------------------------------------
# resonance curve helpers
# ---------------------------------------------------------------------------


def resonance_ratio(omega: float, omega_n: float, zeta: float) -> float:
    """Dynamic amplification ``1 / sqrt((1-r^2)^2 + (2 zeta r)^2)``, ``r=omega/omega_n``."""
    r = omega / omega_n
    return 1.0 / math.sqrt((1.0 - r * r) ** 2 + (2.0 * zeta * r) ** 2)


def peak_frequency(omega_n: float, zeta: float) -> float:
    """Frequency of the maximum of the damped-oscillator transfer function.

    ``omega_peak = omega_n sqrt(1 - 2 zeta^2)`` (zero if ``zeta >= 1/sqrt(2)``,
    where the response has no interior maximum).
    """
    if zeta < 0.0:
        raise ValueError("zeta must be non-negative")
    val = 1.0 - 2.0 * zeta * zeta
    return omega_n * math.sqrt(val) if val > 0.0 else 0.0


def static_tilt_elevation(
    x: float, xddot: float, g: float, length: float
) -> float:
    """Rigid-body tilt of a slowly accelerated tank: ``-Xddot (x - L/2) / g``."""
    return -xddot * (x - 0.5 * length) / g


def odd_mode_fraction(modal: dict, modes: int = 6) -> float:
    """Share of the free-surface energy carried by odd modes (parity diagnostic)."""
    num = 0.0
    den = 0.0
    for n in range(1, int(modes) + 1):
        val = float(modal.get(n, 0.0))
        if n % 2 == 1:
            num += val * val
        den += val * val
    return num / den if den > 0.0 else float("nan")


def gauge_height_from_swl(row: int, depth_cells: float, dx: float) -> float:
    """Height of the *cell centre* of ``row`` above the still-water line (m).

    Rows are counted from the tank bottom: cell row ``j`` covers ``[j-1, j]``
    cells above the bottom, so its centre sits at ``(j - 0.5)`` cells.
    """
    return (row - 0.5 - depth_cells) * dx


def modal_phase(a: complex) -> float:
    """Phase (radians) of a complex modal amplitude — small utility for tests."""
    return cmath.phase(a)

# ---------------------------------------------------------------------------
# finite-time linear reference (transient included)
# ---------------------------------------------------------------------------


def finite_time_modal_response(
    mode: int,
    sway_amplitude: float,
    omega: float,
    zeta: float,
    depth: float,
    length: float,
    g: float = 9.81,
    n_periods: float = 4.0,
    ramp_periods: float = 0.5,
    samples_per_period: int = 400,
) -> tuple:
    """Integrate the damped modal ODE with the campaign's ramp and forcing.

    The reference is the *same* second-order ODE that :func:`steady_modal_amplitude`
    solves in closed form, but integrated in time so the **start-up transient** of
    a short forced run (the campaign uses a few forcing periods only, because the
    Körner model's numerical damping is small) is reproduced exactly::

        addot + 2 zeta omega_n adot + omega_n^2 a = F(t),
        F(t) = -4 X0 omega^2 tanh(k h) / (L k) * ramp(t) * sin(omega t)

    ``ramp(t)`` is linear over ``ramp_periods`` mode-1 natural periods, matching
    ``tank2d.run_sway``.  Returns ``(t, a)`` as numpy float64 arrays.
    """
    import numpy as np

    if n_periods <= 0.0:
        raise ValueError("n_periods must be positive")
    wn = abramson_angular_frequency(mode, depth, length, g)
    k = abramson_wave_number(mode, length)
    t1 = 2.0 * math.pi / abramson_angular_frequency(1, depth, length, g)
    t_end = n_periods * 2.0 * math.pi / omega
    n_steps = max(int(samples_per_period * n_periods), 32)
    t = np.linspace(0.0, t_end, n_steps + 1)
    dt = t[1] - t[0]
    t_ramp = ramp_periods * t1
    coef = 2.0 * abs(mode_excitation_coefficient(mode)) * sway_amplitude * omega * omega * math.tanh(
        k * depth
    ) / (length * k)

    def force(tt: float) -> float:
        ramp = min(1.0, tt / t_ramp) if t_ramp > 0.0 else 1.0
        return -coef * ramp * math.sin(omega * tt)

    a = np.zeros_like(t)
    v = np.zeros_like(t)

    def rhs(state, tt):
        x, u = state
        return u, force(tt) - 2.0 * zeta * wn * u - wn * wn * x

    for i in range(n_steps):
        tt = t[i]
        st = (a[i], v[i])
        k1 = rhs(st, tt)
        k2 = rhs((st[0] + 0.5 * dt * k1[0], st[1] + 0.5 * dt * k1[1]), tt + 0.5 * dt)
        k3 = rhs((st[0] + 0.5 * dt * k2[0], st[1] + 0.5 * dt * k2[1]), tt + 0.5 * dt)
        k4 = rhs((st[0] + dt * k3[0], st[1] + dt * k3[1]), tt + dt)
        a[i + 1] = st[0] + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v[i + 1] = st[1] + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return t, a


def finite_time_reference_amplitude(
    mode: int,
    sway_amplitude: float,
    omega: float,
    zeta: float,
    depth: float,
    length: float,
    g: float = 9.81,
    n_periods: float = 4.0,
    ramp_periods: float = 0.5,
) -> dict:
    """Scalar summary of :func:`finite_time_modal_response` in its final period."""
    t, a = finite_time_modal_response(
        mode, sway_amplitude, omega, zeta, depth, length, g, n_periods, ramp_periods
    )
    import numpy as np

    period = 2.0 * math.pi / omega
    win = t >= t[-1] - period * (n_periods * 0.5)
    half = t >= t[-1] - 0.5 * (t[-1] - t[0])
    return {
        "t_end": float(t[-1]),
        "a_end": float(a[-1]),
        "a_abs_max_last_period": float(np.max(np.abs(a[t >= t[-1] - period]))),
        "a_peak_half": float(np.max(np.abs(a[half]))),
        "a_peak_overall": float(np.max(np.abs(a))),
        "n_samples": int(t.size),
        "period_s": float(period),
        "window_points": int(win.sum()),
    }
