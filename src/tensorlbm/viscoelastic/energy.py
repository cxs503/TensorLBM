"""Energy budget of a viscoelastic channel flow (pump work vs dissipation).

For an incompressible Oldroyd-B/FENE-P fluid the mechanical energy balance of the
fully developed channel reads

    P_in = Phi_s + Phi_p + dE_el/dt + dKE/dt

with

* ``P_in  = G Q``                      pump (pressure-gradient) power per unit
                                       depth and unit streamwise length,
* ``Phi_s = int eta_s gamma_dot^2 dy`` solvent viscous dissipation,
* ``Phi_p = int (eta_p / 2 lam^2) f (f tr C - 3 a) dy``  polymer (conformational)
                                       dissipation, ``>= 0``, zero at equilibrium,
* ``E_el  = int (eta_p / 2 lam) L2 ln[(L2-3)/(L2-trC)] dy``  elastic energy
                                       (Oldroyd-B limit ``eta_p(trC-3)/2lam``).

Discretisation of the integrals
-------------------------------
The solver stores fields on the *fluid* nodes only, which span
``y in [-H + dy/2, H - dy/2]`` for a half-way bounce-back wall at ``y = +-H``.
Integrating over the fluid nodes alone would miss the two near-wall half cells,
where ``gamma_dot^2`` and ``tr C - 3`` still carry ~17% of their total weight
(``gamma_dot ~ y`` near the wall), so the budget would close only to O(10%) --
not a physical effect but a quadrature error.  ``energy_budget`` therefore
extends every profile to the walls:

* velocity by the no-slip condition ``u(+-H) = 0``,
* ``gamma_dot`` and ``C`` by linear extrapolation in ``y`` (exact for the steady
  Poiseuille solution, where both are linear in ``y``).
"""

from __future__ import annotations

import numpy as np

from . import conformation as conf

__all__ = [
    "trace_quad",
    "strain_rate_tensor_1d",
    "polymer_work",
    "pad_profiles_to_walls",
    "energy_budget",
]


def trace_quad(field, y, method: str = "trapezoid"):
    """Integral of a wall-normal profile over the channel.

    ``method="trapezoid"`` (default) is the quadrature that matches the finite
    volume of the lattice cells; ``method="simpson"`` is offered for
    quadrature-convergence studies on refined/analytic profiles, where the
    integrands are polynomial in ``y`` and Simpson's rule is exact.
    """
    f = np.asarray(field, dtype=float)
    yy = np.asarray(y, float)
    if method == "simpson":
        from scipy.integrate import simpson

        return float(simpson(f, x=yy))
    return float(np.trapezoid(f, yy))


def strain_rate_tensor_1d(gamma_dot):
    """Symmetric strain-rate tensor for a 1-D shear with ``gamma_dot = du/dy``."""
    g = np.asarray(gamma_dot, dtype=float)
    out = np.zeros(g.shape + (3, 3))
    out[..., 0, 1] = out[..., 1, 0] = 0.5 * g
    return out


def polymer_work(tau_p, gamma_dot):
    """``tau_p : grad u`` density for a 1-D shear.

    With ``L = gamma_dot e_x (x) e_y`` only ``L_xy`` is non-zero, so
    ``tau : grad u = sum_ij tau_ij L_ji = tau_yx gamma_dot = tau_xy gamma_dot``
    (no factor 2 -- the factor 2 lives in ``2 S_xy = gamma_dot``).
    """
    return np.asarray(tau_p, float)[..., 0, 1] * np.asarray(gamma_dot, float)


def pad_profiles_to_walls(y, u, gamma_dot, C=None):
    """Extend fluid-node profiles to the walls at ``y = +-H``.

    ``H`` is inferred from the grid (half-way bounce-back: the first fluid node
    sits half a cell away from the wall).  Velocity uses the no-slip condition;
    ``gamma_dot`` and ``C`` use linear extrapolation (exact for the exact
    Poiseuille solution).
    """
    y = np.asarray(y, float)
    dy = y[1] - y[0]
    H = -y[0] + 0.5 * dy
    y_pad = np.concatenate(([-H], y, [H]))
    u = np.asarray(u, float)
    u_pad = np.concatenate(([0.0], u, [0.0]))
    g = np.asarray(gamma_dot, float)
    g_pad = np.concatenate(([2.0 * g[0] - g[1]], g, [2.0 * g[-1] - g[-2]]))
    out = {"y": y_pad, "u": u_pad, "gamma_dot": g_pad, "H": H}
    if C is not None:
        C = np.asarray(C, float)
        C_pad = np.concatenate(([2.0 * C[0] - C[1]], C, [2.0 * C[-1] - C[-2]]), axis=0)
        out["C"] = C_pad
    return out


def energy_budget(
    u,
    gamma_dot,
    C,
    model,
    eta_p: float,
    nu_s: float,
    G: float,
    y,
    C_prev=None,
    elapsed: float | None = None,
    u_prev=None,
    pad_to_walls: bool = True,
):
    """Evaluate the mechanical energy budget on a channel solution.

    Parameters
    ----------
    u, gamma_dot, y : array_like
        Wall-normal profiles of streamwise velocity, ``du/dy`` and the wall-normal
        coordinate (lattice units, *fluid nodes only*).
    C : ndarray (ny, 3, 3)
        Conformation tensor profile on the same nodes.
    model : OldroydB | FeneP
    eta_p, nu_s, G : float
        Polymer viscosity, solvent viscosity, body force (density is 1).
    C_prev, u_prev, elapsed : optional
        Previous state and the time between snapshots, used to measure the
        ``dE_el/dt`` and ``dKE/dt`` terms of a transient run.
    pad_to_walls : bool
        Extend the integrals over the two near-wall half cells (default; see the
        module docstring for why this matters).

    Returns
    -------
    dict with ``P_in``, ``Phi_s``, ``Phi_p``, ``dE_dt``, ``dKE_dt``, ``E_el``,
    ``polymer_work``, ``residual``, ``residual_rel``, ``identity_residual_rel``,
    ``tau_p_xy``.
    """
    u = np.asarray(u, float)
    g = np.asarray(gamma_dot, float)
    C = np.asarray(C, float)
    y = np.asarray(y, float)

    if pad_to_walls and y.size >= 3:
        padded = pad_profiles_to_walls(y, u, g, C)
        y_i, u_i, g_i, C_i = padded["y"], padded["u"], padded["gamma_dot"], padded["C"]
        if u_prev is not None:
            u_prev = np.asarray(u_prev, float)
            u_prev = np.concatenate(([0.0], u_prev, [0.0]))
        if C_prev is not None:
            C_prev = np.asarray(C_prev, float)
            C_prev = np.concatenate(
                ([2.0 * C_prev[0] - C_prev[1]], C_prev, [2.0 * C_prev[-1] - C_prev[-2]]),
                axis=0,
            )
    else:
        y_i, u_i, g_i, C_i = y, u, g, C

    tr_c = conf.trace3(C_i)

    Q = trace_quad(u_i, y_i)
    P_in = G * Q
    Phi_s = trace_quad(nu_s * g_i ** 2, y_i)

    tau_p = model.stress(C_i, eta_p)
    work = polymer_work(tau_p, g_i)
    W_p = trace_quad(work, y_i)

    e_dens = model.elastic_energy_density(tr_c, eta_p)
    E_el = trace_quad(e_dens, y_i)

    phi_p_dens = model.peterlin_dissipation(tr_c, eta_p)
    Phi_p = trace_quad(phi_p_dens, y_i)

    if C_prev is not None and elapsed:
        tr_prev = conf.trace3(C_prev)
        e_prev = model.elastic_energy_density(tr_prev, eta_p)
        dE_dt = (E_el - trace_quad(e_prev, y_i)) / float(elapsed)
    else:
        # steady state: dE/dt = tau_p:grad u - Phi_p
        dE_dt = W_p - Phi_p

    if u_prev is not None and elapsed:
        dKE_dt = (trace_quad(0.5 * u_i ** 2, y_i)
                  - trace_quad(0.5 * np.asarray(u_prev, float) ** 2, y_i)) / float(elapsed)
    else:
        dKE_dt = 0.0

    total_out = Phi_s + Phi_p + dE_dt + dKE_dt
    residual = P_in - total_out
    scale = abs(P_in) if abs(P_in) > 0 else 1.0
    identity = W_p - (dE_dt + Phi_p)
    id_scale = abs(W_p) if abs(W_p) > 0 else 1.0
    return {
        "Q": Q,
        "P_in": P_in,
        "Phi_s": Phi_s,
        "Phi_p": Phi_p,
        "E_el": E_el,
        "dE_dt": dE_dt,
        "dKE_dt": dKE_dt,
        "polymer_work": W_p,
        "residual": residual,
        "residual_rel": abs(residual) / scale,
        "identity_residual_rel": abs(identity) / id_scale,
        "tau_p_xy": tau_p[..., 0, 1],
    }