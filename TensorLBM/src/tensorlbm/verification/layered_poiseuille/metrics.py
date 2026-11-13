"""Integral error metrics for the layered Poiseuille cross-validation case.

The metric set is a faithful port of TCLB's ``updateMyGlobals`` / ``updateTrackers``
(``models/multiphase/d3q27_pf_velocity/Dynamics.c.Rt:721-800``), which is how the
reference implementation measures the error of a ``developedFlow`` run:

``updateMyGlobals(real_t pf)`` builds, per node,
  * ``rho  = Density_l + (pf - PhaseField_l)*(Density_h - Density_l)/(PhaseField_h - PhaseField_l)``
  * ``AddToTotalDensity(rho)`` and ``AddToKineticEnergy(rho*u2mag)`` (note: no 1/2)
  * for ``pf < 0.5`` (gas) ``(1-pf)*|u|``, ``(1-pf)*U``, ``(1-pf)*V``, ``(1-pf)*W``,
    ``AddToGasTotalPhase(1-pf)`` and ``AddToXLocation((1-pf)*X)``
  * for ``pf >= 0.5`` (liquid) ``pf*|u|``, ``pf*U``, ``pf*V``, ``pf*W``, ``AddToLiqTotalPhase(pf)``
  * for ``NODE_flux_nodes`` the arithmetic flux samples

``updateTrackers`` locates an interface crossing by linear interpolation of the
phase field at ``C = 0.5``: ``location = Y - (C-0.5)/(C - PhaseF(0,-1,0))``.

Because the layered channel used here has a *prescribed, stationary* interface the
phase-field rows above are computed from a static marker field; that makes
"interface position" an identity rather than a measurement, which is reported as
such.  The dynamic quantity that the piecewise solution really predicts is the
location of the velocity maximum ``y_peak`` (it depends on the viscosity ratio and
the interface position), so it is added as a genuine interface-sensitive KPI.

Reductions kept from TCLB: ``<U> = SumLiquidVelocityX / SumLiquidPhase`` style
phase-averaged velocities, i.e. our ``q_layer`` is the layer-integrated flux per
unit width in lattice units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass
class Globals:
    """TCLB ``updateMyGlobals`` reduction of one snapshot."""

    total_density: float
    kinetic_energy: float          # TCLB convention: sum(rho*|u|^2), no 1/2
    gas_total_velocity: float
    liq_total_velocity: float
    gas_total_velocity_x: float
    liq_total_velocity_x: float
    gas_total_phase: float
    liq_total_phase: float
    flux_node_count: int
    flux_x: float
    flux_y: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "total_density": self.total_density,
            "kinetic_energy": self.kinetic_energy,
            "gas_total_velocity": self.gas_total_velocity,
            "liq_total_velocity": self.liq_total_velocity,
            "gas_total_velocity_x": self.gas_total_velocity_x,
            "liq_total_velocity_x": self.liq_total_velocity_x,
            "gas_total_phase": self.gas_total_phase,
            "liq_total_phase": self.liq_total_phase,
            "flux_node_count": float(self.flux_node_count),
            "flux_x": self.flux_x,
            "flux_y": self.flux_y,
        }


def update_my_globals(
    ux: np.ndarray,
    uy: np.ndarray,
    pf: np.ndarray,
    density_l: float = 1.0,
    density_h: float = 1.0,
    phasefield_l: float = 0.0,
    phasefield_h: float = 1.0,
    flux_mask: np.ndarray | None = None,
) -> Globals:
    """Port of TCLB ``updateMyGlobals`` on a dense 2-D slice.

    ``pf`` is the phase marker (>=0.5 = "liquid", TCLB's convention), ``ux/uy`` the
    macroscopic velocity.  Node-type masks collapse to a boolean ``flux_mask``.
    """
    pf = np.asarray(pf, dtype=float)
    rho = density_l + (pf - phasefield_l) * (density_h - density_l) / (phasefield_h - phasefield_l)
    u2mag = ux * ux + uy * uy
    umag = np.sqrt(u2mag)
    tmp_pf = 1.0 - pf

    gas = pf < 0.5
    liq = ~gas
    total_density = float(rho.sum())
    kinetic = float((rho * u2mag).sum())
    gas_tv = float((tmp_pf[gas] * umag[gas]).sum())
    liq_tv = float((pf[liq] * umag[liq]).sum())
    gas_tvx = float((tmp_pf[gas] * ux[gas]).sum())
    liq_tvx = float((pf[liq] * ux[liq]).sum())
    gas_tp = float(tmp_pf[gas].sum())
    liq_tp = float(pf[liq].sum())

    if flux_mask is None:
        flux_mask = np.zeros_like(pf, dtype=bool)
    nflux = int(flux_mask.sum())
    flux_x = float(ux[flux_mask].sum()) if nflux else 0.0
    flux_y = float(uy[flux_mask].sum()) if nflux else 0.0

    return Globals(total_density, kinetic, gas_tv, liq_tv, gas_tvx, liq_tvx,
                   gas_tp, liq_tp, nflux, flux_x, flux_y)


def track_interface_y(pf_column: np.ndarray, y_coords: np.ndarray) -> float | None:
    """Port of TCLB ``updateTrackers`` interface tracking along Y.

    TCLB: ``if C < 0.5 and PhaseF(0,-1,0) > 0.5: location = Y - (C-0.5)/(C-PhaseF(0,-1,0))``
    -- a linear interpolation of the 0.5 crossing between two neighbouring nodes.
    Returns ``None`` if no crossing is found.
    """
    pf_column = np.asarray(pf_column, dtype=float)
    y = np.asarray(y_coords, dtype=float)
    for j in range(1, len(pf_column)):
        c, below = pf_column[j], pf_column[j - 1]
        if c < 0.5 <= below:
            return float(y[j] - (c - 0.5) / (c - below))
    return None


# --------------------------------------------------------------------------- #
# Error metrics against the exact solution
# --------------------------------------------------------------------------- #
def profile_errors(u_num: np.ndarray, u_exact: np.ndarray) -> Dict[str, float]:
    """Error norms of the cross-section profile, normalised by the exact peak."""
    u_num = np.asarray(u_num, dtype=float)
    u_exact = np.asarray(u_exact, dtype=float)
    denom = max(float(np.max(np.abs(u_exact))), 1e-30)
    diff = u_num - u_exact
    return {
        "eps_inf": float(np.max(np.abs(diff)) / denom),
        "eps_l2_rel": float(np.sqrt(np.mean(diff ** 2)) / np.sqrt(np.mean(u_exact ** 2))),
        "eps_mean": float(np.mean(diff) / denom),
    }


def _edge_value(u: np.ndarray, y_wall: np.ndarray, x: float) -> float:
    """Value of a sampled profile at a layer edge.

    Walls are no-slip (``u = 0`` at the half-way wall); interior layer edges are
    linearly interpolated from the two bracketing samples.  The rule is applied
    identically to the numerical and the exact profile so that the *same*
    discrete functional is compared -- the discretisation bias of the reduction
    then cancels out of the error and only the physical difference remains.
    """
    if x <= y_wall[0] or x >= y_wall[-1]:
        return 0.0
    k = int(np.searchsorted(y_wall, x))
    y0, y1 = y_wall[k - 1], y_wall[k]
    t = (x - y0) / (y1 - y0)
    return float(u[k - 1] + t * (u[k] - u[k - 1]))


def layer_fluxes(u: np.ndarray, y_wall: np.ndarray,
                 interfaces: Sequence[float]) -> list[float]:
    """Flux per unit width through every layer: integral of u dy over the layer.

    Same reduction as TCLB's ``SumLiqTotalVelocityX / SumLiqTotalPhase`` (phase
    average), written in continuum form so that it is comparable entry by entry
    to the exact layer fluxes.  Compare the numerical and the exact profile with
    *this* functional (see :func:`_edge_value`) rather than against a
    continuum integral, otherwise the O(dy^2) trapezoid bias shows up as a fake
    physical error.
    """
    u = np.asarray(u, dtype=float)
    y_wall = np.asarray(y_wall, dtype=float)
    edges = [0.0, *[float(i) for i in interfaces], float(y_wall[-1] + 0.5)]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (y_wall > lo) & (y_wall < hi)
        y = np.concatenate(([lo], y_wall[sel], [hi]))
        vals = np.concatenate(([_edge_value(u, y_wall, lo)], u[sel],
                               [_edge_value(u, y_wall, hi)]))
        out.append(float(np.trapezoid(vals, y) if hasattr(np, "trapezoid") else np.trapz(vals, y)))
    return out


def peak_metrics(u: np.ndarray, y_wall: np.ndarray) -> Dict[str, float]:
    """Location and value of the profile maximum (interface-position sensitive)."""
    k = int(np.argmax(u))
    return {"u_peak_num": float(u[k]), "y_peak_num": float(y_wall[k])}


def rel_err(got: float, ref: float) -> float:
    return float(abs(got - ref) / max(abs(ref), 1e-30))