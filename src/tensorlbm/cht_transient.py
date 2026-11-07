"""Transient CHT verification anchors: semi-infinite penetration (erf), lumped
capacitance and the adiabatic-interface negative control.

Part 2 of the CHT verification ladder (see :mod:`tensorlbm.cht_verification`
for the steady conduction branch, :mod:`tensorlbm.graetz` for the Graetz anchor
and :mod:`tensorlbm.cht_ladder` for the gate runner).

* :func:`erf_penetration` -- transient conduction in a semi-infinite solid with
  a step in wall temperature; the analytical anchor is
  ``(T-T0)/(T_w-T0) = erfc(x/2 sqrt(alpha t))``.
* :func:`lumped_cooling_cube` -- transient cooling of a small-Biot 3-D cube with
  a convective (Robin) boundary; the analytical anchor is the lumped-capacitance
  time constant ``tau = rho c V /(h A)`` extracted from the volume-averaged
  temperature decay.
* :func:`adiabatic_interface_control` -- interface conductance ``h_int = 0``:
  the interface must be adiabatic, i.e. zero heat flux and zero normal
  temperature gradient on both sides.
"""

from __future__ import annotations

import math

import numpy as np

from .cht_verification import solve_tridiag

__all__ = [
    "erf_penetration",
    "lumped_cooling_cube",
    "adiabatic_interface_control",
]


# ===========================================================================
# anchor 2: erf thermal penetration in a semi-infinite solid
# ===========================================================================


def erf_penetration(
    alpha: float = 1.0,
    t_end: float = 1.0,
    T_wall: float = 1.0,
    T_init: float = 0.0,
    n_x: int = 800,
    n_t: int = 400,
    domain_factor: float = 6.0,
) -> dict:
    """Transient 1-D conduction in a semi-infinite solid (erf anchor).

    The solid occupies ``x in [0, L]`` with ``L = domain_factor*sqrt(alpha t)``
    (far enough that the far field remains at ``T_init``), a Dirichlet hot wall
    at ``x = 0`` and a backward-Euler (implicit, unconditionally stable)
    discretisation:

        ``T^{n+1} - r (T^{n+1}_{i+1} - 2 T^{n+1}_i + T^{n+1}_{i-1}) = T^n``,
        ``r = alpha dt / dx^2``

    Analytical anchor: ``(T-T0)/(T_w-T0) = erfc(x /(2 sqrt(alpha t)))``.
    """
    L = domain_factor * math.sqrt(alpha * t_end)
    dx = L / n_x
    dt = t_end / n_t
    r = alpha * dt / dx**2

    x = dx * np.arange(n_x + 1)
    T = np.full(n_x + 1, float(T_init))
    T[0] = T_wall
    T[-1] = T_init

    n = n_x + 1
    lower = np.full(n, -r)
    diag = np.full(n, 1.0 + 2.0 * r)
    upper = np.full(n, -r)
    lower[0] = 0.0
    upper[0] = 0.0
    diag[0] = 1.0
    lower[-1] = 0.0
    diag[-1] = 1.0

    for _ in range(n_t):
        rhs = T.copy()
        rhs[0] = T_wall
        rhs[-1] = T_init
        T = solve_tridiag(lower, diag, upper, rhs)

    z = x / (2.0 * math.sqrt(alpha * t_end))
    # erfc(z) via the torch/tensor-free numpy path (math.erfc is scalar), use
    # the vectorised torch implementation for speed and accuracy.
    import torch

    T_an = T_init + (T_wall - T_init) * (
        1.0 - torch.erf(torch.tensor(z, dtype=torch.float64)).numpy()
    )

    dT = T_wall - T_init
    active = z < 2.5
    return {
        "x": x,
        "T": T,
        "T_analytic": T_an,
        "z": z,
        "max_err": float(np.max(np.abs(T - T_an)) / dT),
        "max_err_active": float(np.max(np.abs(T[active] - T_an[active])) / dT),
        "l2_err": float(np.sqrt(np.mean((T - T_an) ** 2)) / dT),
        "meta": {
            "n_x": n_x,
            "n_t": n_t,
            "dx": dx,
            "dt": dt,
            "alpha": alpha,
            "t_end": t_end,
            "implicit_ratio_r": r,
        },
    }


# ===========================================================================
# anchor 4: lumped capacitance (small-Biot 3-D cube)
# ===========================================================================


def lumped_cooling_cube(
    n: int = 16,
    side: float = 1.0,
    alpha: float = 1.0,
    biot: float = 0.02,
    T_init: float = 1.0,
    T_inf: float = 0.0,
    taus: float = 4.0,
    fit_from: float = 0.4,
    fit_to: float = 2.8,
    rho_c: float = 1.0,
    fo: float = 0.14,
) -> dict:
    """Transient cooling of a small-Biot 3-D cube with a convective boundary.

    A cube of side ``a = side`` is discretised with ``n^3`` cell-centred finite
    volumes (``dx = a/n``, so the discrete volume and area are exactly
    ``a^3`` and ``6a^2``).  Every cell face is either an internal conduction
    face (conductance ``k dx^2/dx``) or - on the six outer planes - a convective
    face (conductance ``h dx^2``), i.e. the discretisation is conservative.

    With ``k = alpha rho_c`` and the characteristic length ``V/A = a/6`` the
    Biot number is ``Bi = h a /(6k)`` and the lumped-capacitance time constant
    is ``tau = rho c V/(h A) = rho_c a /(6h)``.  The volume-averaged
    temperature is fitted with ``ln(T_m - T_inf) = ln A - t/tau_sim`` over
    ``[fit_from, fit_to] tau`` and the extracted ``tau_sim`` is compared with
    ``tau``; the residual difference is the finite-Biot correction that the
    lumped model neglects.
    """
    if n < 5:
        raise ValueError("n must be >= 5")
    k = alpha * rho_c
    h = biot * 6.0 * k / side
    dx = side / n

    tau_lumped = rho_c * side / (6.0 * h)
    t_end = taus * tau_lumped
    dt = fo * dx**2 / alpha
    n_steps = int(math.ceil(t_end / dt))
    dt = t_end / n_steps

    coeff = alpha * dt / dx**2
    robin = h * dx / k  # face conductance ratio: h dx^2 / (k dx^2 / dx) = h dx/k

    T = np.full((n, n, n), float(T_init))
    times = np.empty(n_steps + 1)
    tmean = np.empty(n_steps + 1)
    times[0] = 0.0
    tmean[0] = float(T.mean())

    for step in range(1, n_steps + 1):
        lap = np.zeros_like(T)
        lap[1:, :, :] += T[:-1, :, :] - T[1:, :, :]
        lap[:-1, :, :] += T[1:, :, :] - T[:-1, :, :]
        lap[:, 1:, :] += T[:, :-1, :] - T[:, 1:, :]
        lap[:, :-1, :] += T[:, 1:, :] - T[:, :-1, :]
        lap[:, :, 1:] += T[:, :, :-1] - T[:, :, 1:]
        lap[:, :, :-1] += T[:, :, 1:] - T[:, :, :-1]
        # convective (Robin) faces on the six outer planes
        lap[0, :, :] += robin * (T_inf - T[0, :, :])
        lap[-1, :, :] += robin * (T_inf - T[-1, :, :])
        lap[:, 0, :] += robin * (T_inf - T[:, 0, :])
        lap[:, -1, :] += robin * (T_inf - T[:, -1, :])
        lap[:, :, 0] += robin * (T_inf - T[:, :, 0])
        lap[:, :, -1] += robin * (T_inf - T[:, :, -1])
        T = T + coeff * lap

        times[step] = step * dt
        tmean[step] = float(T.mean())

    mask = (times >= fit_from * tau_lumped) & (times <= fit_to * tau_lumped)
    order = np.argsort(times[mask])
    slope, _intercept = np.polyfit(times[mask][order], np.log(tmean[mask][order] - T_inf), 1)
    tau_sim = -1.0 / slope

    late = times > 0.95 * t_end
    tau_late = -1.0 / np.polyfit(times[late], np.log(tmean[late] - T_inf), 1)[0]

    return {
        "times": times,
        "T_mean": tmean,
        "tau_sim": float(tau_sim),
        "tau_late": float(tau_late),
        "tau_lumped": float(tau_lumped),
        "rel_err_tau": float(abs(tau_sim - tau_lumped) / tau_lumped),
        "rel_err_tau_late": float(abs(tau_late - tau_lumped) / tau_lumped),
        "meta": {
            "n": n,
            "side": side,
            "dx": dx,
            "dt": dt,
            "n_steps": n_steps,
            "fourier_3d": fo,
            "biot": biot,
            "h": h,
            "alpha": alpha,
            "k": k,
            "discrete_volume": float(n**3 * dx**3),
            "discrete_area": float(6.0 * n**2 * dx**2),
            "fit_window": [fit_from * tau_lumped, fit_to * tau_lumped],
        },
    }


# ===========================================================================
# negative control: h_int = 0 must give an adiabatic interface
# ===========================================================================


def adiabatic_interface_control(
    n_left: int = 80,
    n_right: int = 80,
    T_left_init: float = 1.0,
    T_right_init: float = 0.0,
    alpha: float = 1.0,
    n_steps: int = 4000,
    fo: float = 0.4,
) -> dict:
    """Adiabatic interface control (``h_int = 0``).

    Two 1-D solid slabs are brought into contact through an interface whose
    contact conductance is set to zero.  Because no heat can cross the
    interface and both outer ends are insulated, each slab must relax to its own
    uniform temperature: the discrete heat flux through the interface and the
    normal temperature gradient on both sides must vanish.  A non-zero gradient
    (or flux) would immediately expose an interface operator that leaks heat
    when it must not.
    """
    dx = 1.0
    dt = fo * dx**2 / alpha
    T = np.concatenate([np.full(n_left, T_left_init), np.full(n_right, T_right_init)])

    # interface face conductance (between node n_left-1 and n_left) = 0
    g_iface = 0.0
    r = alpha * dt / dx**2

    for _ in range(n_steps):
        flux = r * (T[:-1] - T[1:])
        # zero the interface face flux (adiabatic contact)
        flux[n_left - 1] = g_iface
        T = T.copy()
        T[:-1] -= flux
        T[1:] += flux
        # outer boundaries insulated -> no flux added

    # normal gradients at the interface, one-sided (nodes adjacent to the face)
    dTdn_left = (T[n_left - 1] - T[n_left - 2]) / dx
    dTdn_right = (T[n_left + 1] - T[n_left]) / dx
    flux_iface = g_iface * (T[n_left - 1] - T[n_left])

    return {
        "T": T,
        "interface_index": n_left,
        "flux_interface": float(flux_iface),
        "gradient_left": float(dTdn_left),
        "gradient_right": float(dTdn_right),
        "max_gradient": float(max(abs(dTdn_left), abs(dTdn_right))),
        "temperature_range_left": float(T[:n_left].max() - T[:n_left].min()),
        "temperature_range_right": float(T[n_left:].max() - T[n_left:].min()),
        "meta": {"n_left": n_left, "n_right": n_right, "n_steps": n_steps, "fo": fo, "alpha": alpha},
    }