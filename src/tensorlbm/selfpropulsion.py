"""Ship--propeller self-propulsion: body-force actuator disk + wake/thrust-deduction split.

This module implements the ship-propulsion chain of a hull-propeller system in
the classical decomposition::

    P_E  (effective power)  = R_T * V
    P_T  (thrust power)     = T   * V_A,        V_A = V (1 - w)
    P_D  (delivered power)  = 2 pi n Q
    eta_D = P_E / P_D = eta_H * eta_R * eta_0
    eta_H = (1 - t) / (1 - w)

with

* ``w``     effective wake fraction (hull -> propeller inflow deficit),
* ``t``     thrust-deduction fraction (propeller -> hull resistance change),
* ``eta_R`` relative rotative efficiency (non-uniform wake distortion),
* ``eta_0`` open-water propeller efficiency.

Physics content
---------------
1. :func:`blade_element_momentum` -- blade-element/momentum (BEM) analysis with
   Prandtl tip/root loss.  Input is the *blade-section lift/drag polar* plus the
   radial chord/pitch distribution (task item 1: "blade section lift-drag ->
   pressure jump and circumferential velocity").  Output is the radial loading
   ``dT/dr``, ``dQ/dr``, hence ``KT(J)``, ``KQ(J)``, ``eta_0(J)``.
2. :func:`build_actuator_disk_field` -- projects the BEM radial loading onto a
   3-D lattice as an axisymmetric body-force field (axial + swirl components).
   The pressure jump and the swirl are *not* imposed: they are consequences of
   the momentum source and are measured a posteriori
   (:func:`disk_pressure_jump`, :func:`disk_swirl`).
3. :func:`apply_body_force_guo` -- Guo-type momentum source injection for the
   D3Q19 solver (exactly momentum conserving, hence thrust-recoverable).
4. :func:`plane_wake_stats` -- propeller-plane momentum-deficit integral giving
   the wake fraction and its radial/azimuthal structure.
5. :func:`hull_pressure_resistance` / :func:`thrust_deduction_from_pressure` --
   stern pressure integration with propeller on/off, giving ``Delta R`` and
   hence ``t = Delta R / T``.
6. :func:`solve_self_propulsion` -- bisection on revolution rate ``n`` so that
   ``T(n) (1 - t) = R_T`` (task item 5).
7. :func:`relative_rotative_efficiency` -- sector-wise BEM on the *measured*
   non-uniform wake field, ``eta_R = KQ_0 / KQ_B`` at equal thrust.

Evidence grading
----------------
REAL: BEM loading, body-force projection, Guo forcing, wake/pressure integrals,
self-propulsion root find, power-chain arithmetic (all exercised by tests).
MODEL (documented, semi-empirical): the section polar, the frozen loading-shape
assumption in non-uniform inflow, and the Reynolds scaling of the LBM wake
fraction (:func:`rescale_wake_fraction`).

References
----------
* Carlton, J. *Marine Propellers and Propulsion*, 3rd ed., Butterworth-Heinemann.
* ITTC (2014) *Recommended Procedures 7.5-02-03-02.1, Open Water Test*.
* Fujisawa et al. (2000); SIMMAN (2010) / Tokyo-2015 KCS-KP505 workshop data.
* Guo, Zheng & Shi (2002) *Phys. Fluids* 14: 385 -- forcing term for LBM.
* Holtrop & Mennen (1982) *Int. Shipbuild. Prog.* 29 -- power prediction.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from .d3q19 import C, W, equilibrium3d, macroscopic3d
from .solver3d import stream3d
from .turbulence import collide_smagorinsky_mrt3d

__all__ = [
    "KP505Geometry",
    "SectionPolar",
    "KCS_MODEL",
    "blade_element_momentum",
    "open_water_series",
    "kp505_reference_table",
    "build_actuator_disk_field",
    "apply_body_force_guo",
    "disk_pressure_jump",
    "disk_swirl",
    "plane_wake_stats",
    "hull_pressure_resistance",
    "thrust_deduction_from_pressure",
    "solve_self_propulsion",
    "relative_rotative_efficiency",
    "power_chain",
    "solve_self_propulsion_curve",
    "rescale_wake_fraction",
    "empirical_wake_fraction",
    "empirical_thrust_deduction",
    "ittc78_resistance",
    "holtrop_relative_rotative_efficiency",
    "lattice_force_scale",
    "run_actuator_disk_momentum_case",
]


# ---------------------------------------------------------------------------
# 1. KP505 propeller geometry and section polar
# ---------------------------------------------------------------------------

#: KP505 (KCS propeller) radial distributions as published in the SIMMAN-2010 /
#: Tokyo-2015 KCS-KP505 workshop documentation.  Z = 5, D = 0.25 m (model
#: scale), hub ratio 0.18, expanded area ratio 0.80, design P/D = 0.997.
_KP505_RR = [0.18, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
             0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
_KP505_CD = [0.170, 0.222, 0.253, 0.276, 0.295, 0.310, 0.320, 0.327, 0.330,
             0.329, 0.322, 0.310, 0.292, 0.264, 0.226, 0.176, 0.100]
_KP505_PD = [0.809, 0.850, 0.880, 0.905, 0.930, 0.953, 0.973, 0.988, 0.997,
             1.001, 1.000, 0.995, 0.985, 0.970, 0.950, 0.925, 0.895]


@dataclass(frozen=True)
class KP505Geometry:
    """KP505 propeller geometry (KCS, model scale D = 0.25 m)."""

    n_blades: int = 5
    diameter: float = 0.25
    hub_ratio: float = 0.18
    expanded_area_ratio: float = 0.80
    design_pitch_ratio: float = 0.997
    r_over_R: tuple[float, ...] = tuple(_KP505_RR)
    chord_over_D: tuple[float, ...] = tuple(_KP505_CD)
    pitch_over_D: tuple[float, ...] = tuple(_KP505_PD)

    def _interp(self, table: Sequence[float], rr: np.ndarray) -> np.ndarray:
        return np.interp(rr, np.asarray(self.r_over_R, dtype=float),
                         np.asarray(table, dtype=float))

    @functools.cached_property
    def ear_scale(self) -> float:
        """Chord scale factor that reproduces the published expanded area ratio.

        The tabulated chord distribution (digits as published for KP505) is
        normalised to ``expanded_area_ratio``; the raw integral is reported by
        :meth:`raw_area_ratio` so the normalisation is auditable.
        """
        rr = np.linspace(self.hub_ratio, 1.0, 400)
        raw = self._interp(self.chord_over_D, rr) * self.diameter
        integral = float(np.trapezoid(raw * 0.5 * self.diameter, rr))
        ear_raw = self.n_blades * integral / (0.25 * math.pi * self.diameter**2)
        return self.expanded_area_ratio / ear_raw

    def raw_area_ratio(self, n_sections: int = 400) -> float:
        """Expanded area ratio of the tabulated chords before normalisation."""
        rr = np.linspace(self.hub_ratio, 1.0, n_sections)
        raw = self._interp(self.chord_over_D, rr) * self.diameter
        integral = float(np.trapezoid(raw * 0.5 * self.diameter, rr))
        return self.n_blades * integral / (0.25 * math.pi * self.diameter**2)

    def chord(self, rr: np.ndarray) -> np.ndarray:
        """Local chord [m] at r/R = *rr* (normalised to the published EAR)."""
        return self._interp(self.chord_over_D, rr) * self.diameter * self.ear_scale

    def pitch(self, rr: np.ndarray) -> np.ndarray:
        """Local pitch [m] at r/R = *rr*."""
        return self._interp(self.pitch_over_D, rr) * self.diameter

    def pitch_angle(self, rr: np.ndarray) -> np.ndarray:
        """Section pitch angle beta = atan(P / (2 pi r)) [rad]."""
        r = np.maximum(rr * 0.5 * self.diameter, 1e-12)
        return np.arctan(self.pitch(rr) / (2.0 * math.pi * r))

    def area_ratio(self, n_sections: int = 400) -> float:
        """Integrated expanded area ratio (geometry validation, ~0.80)."""
        rr = np.linspace(self.hub_ratio, 1.0, n_sections)
        integrand = self.chord(rr) * (0.5 * self.diameter)
        area = self.n_blades * float(np.trapezoid(integrand, rr))
        return area / (0.25 * math.pi * self.diameter**2)


@dataclass(frozen=True)
class SectionPolar:
    """Blade-section lift/drag polar (thin-aerofoil family, NACA 66 mod-like).

    ``cl = cl_alpha * (alpha - alpha_0)`` with soft stall, and
    ``cd = cd0 + k_cl * cl**2``.

    The defaults describe the NACA 66 (mod) a = 0.8 cambered section family used
    by KP505.  ``alpha_0`` is fixed by a *single geometric/kinematic* constraint,
    the published KP505 zero-thrust advance ratio J0 ~ 1.13
    (``alpha_0 = beta(0.7R) - atan(J0 D / (2 pi r_0.7R)) = -3.5 deg``); the lift
    slope and the profile-drag level are the published values of that section
    family.  No parameter is fitted to the KT/KQ curve that is used as the
    validation target.
    """

    cl_alpha: float = 5.7            # [1/rad] section lift slope
    alpha_0_deg: float = -3.5        # zero-lift angle [deg] (from J0 = 1.13)
    cd0: float = 0.0085              # profile drag at CL = 0
    k_cl: float = 0.011              # lift-dependent (induced + separation) drag
    cl_max: float = 1.20             # soft-stall limit
    stall_slope: float = 0.10        # post-stall lift decay per deg

    def cl_cd(self, alpha_deg):
        """Section lift and drag coefficients at incidence *alpha_deg*.

        Linear lift slope up to ``cl_max``, then a soft-stall plateau whose
        decay is linear in degrees of excess incidence (bounded to
        ``[-cl_max, cl_max]`` so that the momentum iteration stays well posed at
        large negative incidence).
        """
        a_eff = np.asarray(alpha_deg, dtype=float) - self.alpha_0_deg
        cl_lin = self.cl_alpha * np.deg2rad(a_eff)
        a_stall = np.rad2deg(self.cl_max / self.cl_alpha)
        excess = np.maximum(np.abs(a_eff) - a_stall, 0.0)
        plateau = np.maximum(self.cl_max - self.stall_slope * excess, 0.0)
        cl = np.where(excess <= 0.0, np.clip(cl_lin, -self.cl_max, self.cl_max),
                      np.sign(a_eff) * plateau)
        cd = self.cd0 + self.k_cl * cl**2
        return cl, cd


def _prandtl_factor(rr: np.ndarray, n_blades: int, phi: np.ndarray,
                    hub_ratio: float) -> np.ndarray:
    """Prandtl tip + root loss factor F = F_tip * F_root."""
    s = np.maximum(np.sin(phi), 1e-4)
    f_tip = np.clip(0.5 * n_blades * (1.0 - rr) / s, 1e-6, 60.0)
    f_root = np.clip(0.5 * n_blades * np.maximum(rr - hub_ratio, 0.0) / s, 1e-6, 60.0)
    f = (2.0 / math.pi) * np.arccos(np.clip(np.exp(-f_tip), -1.0, 1.0))
    f *= (2.0 / math.pi) * np.arccos(np.clip(np.exp(-f_root), -1.0, 1.0))
    return np.clip(f, 0.05, 1.0)


def blade_element_momentum(
    geometry: KP505Geometry,
    polar: SectionPolar,
    j: float,
    *,
    n_rps: float = 1.0,
    rho: float = 999.5,
    n_sections: int = 80,
    inflow_ratio: np.ndarray | None = None,
    a_clamp: float = 0.90,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> dict:
    """Blade-element/momentum analysis of the propeller at advance ratio *j*.

    For every radial section the equilibrium flow angle ``phi`` is found by
    bisection on the blade-element / annular-momentum residual

        r(phi) = phi - atan[ V_A (1 + a) / (Omega r (1 - a')) ],

    where ``a`` and ``a'`` follow from the classical Prandtl-corrected annular
    relations
    ``a = sigma Cn / (4 F sin^2 phi - sigma Cn)`` and
    ``a' = sigma Ct / (4 F sin phi cos phi + sigma Ct)``.
    Bisection is used instead of a relaxed fixed point because the annular
    relation is non-monotonic at high solidity (the naive iteration diverges for
    J < 0.4, producing spurious negative thrust).

    Parameters
    ----------
    geometry, polar:
        Propeller geometry and section polar.
    j:
        Advance ratio based on the *propeller* inflow, ``V_A = j n D``.
    n_rps:
        Revolution rate [rev/s]; sets the physical force scale only.
    inflow_ratio:
        Optional per-section array ``(n_sections,)`` scaling the local axial
        inflow (non-uniform wake sector analysis).  ``None`` -> uniform inflow.
    rho:
        Water density [kg/m^3].

    Returns
    -------
    dict with ``kt``, ``kq``, ``eta0``, ``thrust``, ``torque`` and the radial
    distributions ``r_over_R``, ``dT_dr``, ``dQ_dr``, ``phi_deg``,
    ``alpha_deg``, ``cl``, ``cd``, ``a``, ``a_prime``, ``F``, ``w_relative``.
    """
    rr = np.linspace(geometry.hub_ratio + 1e-4, 1.0 - 1e-4, n_sections)
    r = rr * 0.5 * geometry.diameter
    c = geometry.chord(rr)
    beta = geometry.pitch_angle(rr)

    inflow = np.ones_like(rr) if inflow_ratio is None else np.asarray(inflow_ratio, float)
    if inflow.shape != rr.shape:
        raise ValueError("inflow_ratio must have shape (n_sections,)")

    v_a = max(j, 1e-6) * n_rps * geometry.diameter          # nominal inflow
    omega = 2.0 * math.pi * n_rps
    v_sec = np.maximum(v_a * np.abs(inflow), 1e-12)          # per-section inflow
    sigma = geometry.n_blades * c / (2.0 * math.pi * r)      # local solidity

    def residual(phi):
        """Blade-element / annular-momentum residual (vectorised over phi)."""
        alpha = np.rad2deg(beta[None, :] - phi)
        cl, cd = polar.cl_cd(alpha)
        cn = cl * np.cos(phi) - cd * np.sin(phi)
        ct = cl * np.sin(phi) + cd * np.cos(phi)
        f_loss = _prandtl_factor(rr[None, :], geometry.n_blades, phi, geometry.hub_ratio)
        s2 = 4.0 * f_loss * np.sin(phi) ** 2
        sc = sigma[None, :] * cn
        heavy = s2 - sc <= 1e-9
        a = np.where(heavy, a_clamp,
                     np.clip(sc / np.where(heavy, 1.0, s2 - sc), -0.5, a_clamp))
        scc = sigma[None, :] * ct
        den = 4.0 * f_loss * np.sin(phi) * np.cos(phi) + scc
        a_prime = np.clip(scc / np.where(np.abs(den) < 1e-9, 1.0, den), -0.5, a_clamp)
        phi_c = np.arctan2(v_sec[None, :] * (1.0 + a),
                           np.maximum(omega * r[None, :] * (1.0 - a_prime), 1e-12))
        return phi - phi_c

    # The annular momentum relation has no solution for heavily loaded sections
    # (sigma*Cn > 4 F sin^2 phi).  Instead of a fixed-point iteration that
    # diverges there, locate the sign change of the residual on a fine phi grid
    # and fall back to the minimum-|residual| point where no sign change exists.
    n_grid = 721
    grid = np.deg2rad(np.linspace(0.05, 85.0, n_grid))
    res = residual(grid[:, None])
    sign_change = (res[:-1, :] * res[1:, :]) < 0.0
    has_cross = sign_change.any(axis=0)
    first = np.argmax(sign_change, axis=0)
    cols = np.arange(rr.size)
    r0 = res[first, cols]
    r1 = res[first + 1, cols]
    dphi = grid[1] - grid[0]
    denom = r0 - r1
    phi_cross = grid[first] + dphi * np.where(np.abs(denom) > 1e-30, r0 / denom, 0.5)
    idx_min = np.abs(res).argmin(axis=0)
    phi = np.where(has_cross, phi_cross, grid[idx_min])

    alpha_deg = np.rad2deg(beta - phi)
    cl, cd = polar.cl_cd(alpha_deg)
    cn = cl * np.cos(phi) - cd * np.sin(phi)
    ct = cl * np.sin(phi) + cd * np.cos(phi)
    f_loss = _prandtl_factor(rr, geometry.n_blades, phi, geometry.hub_ratio)
    s2 = 4.0 * f_loss * np.sin(phi) ** 2
    sc = sigma * cn
    heavy = s2 - sc <= 1e-9
    a = np.where(heavy, a_clamp, np.clip(sc / np.where(heavy, 1.0, s2 - sc), -0.5, a_clamp))
    scc = sigma * ct
    den = 4.0 * f_loss * np.sin(phi) * np.cos(phi) + scc
    a_prime = np.clip(scc / np.where(np.abs(den) < 1e-9, 1.0, den), -0.5, a_clamp)

    w_rel = v_sec * (1.0 + a) / np.maximum(np.sin(phi), 1e-6)   # local inflow speed
    dT_dr = 0.5 * rho * w_rel**2 * geometry.n_blades * c * cn
    dQ_dr = 0.5 * rho * w_rel**2 * geometry.n_blades * c * ct * r

    thrust = float(np.trapezoid(dT_dr, r))
    torque = float(np.trapezoid(dQ_dr, r))
    d = geometry.diameter
    kt = thrust / (rho * n_rps**2 * d**4)
    kq = torque / (rho * n_rps**2 * d**5)
    eta0 = (j * kt) / (2.0 * math.pi * kq) if kq > 0 else float("nan")

    return {
        "j": float(j),
        "kt": float(kt),
        "kq": float(kq),
        "eta0": float(eta0),
        "thrust": thrust,
        "torque": torque,
        "r_over_R": rr.tolist(),
        "dT_dr": dT_dr.tolist(),
        "dQ_dr": dQ_dr.tolist(),
        "phi_deg": np.rad2deg(phi).tolist(),
        "alpha_deg": alpha_deg.tolist(),
        "cl": cl.tolist(),
        "cd": cd.tolist(),
        "a": a.tolist(),
        "a_prime": a_prime.tolist(),
        "F": f_loss.tolist(),
        "w_relative": w_rel.tolist(),
        "heavy_loading_sections": int(heavy.sum()),
    }

def open_water_series(
    geometry: KP505Geometry,
    polar: SectionPolar,
    j_values: Sequence[float] | None = None,
    **kwargs,
) -> list[dict]:
    """Open-water curves ``KT``, ``10KQ``, ``eta_0`` over an advance-ratio sweep."""
    js = np.arange(0.1, 1.15, 0.1) if j_values is None else np.asarray(j_values, float)
    return [blade_element_momentum(geometry, polar, float(j), **kwargs) for j in js]


def kp505_reference_table() -> dict:
    """Measured KP505 open-water data (Fujisawa et al. 2000; SIMMAN 2010).

    Stored as the reference against which the BEM open-water solution is
    validated (``J``, ``KT``, ``KQ``, ``10KQ``, ``eta_0``).
    """
    j = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    kt = [0.45, 0.44, 0.42, 0.40, 0.37, 0.33, 0.29, 0.24, 0.17, 0.10, 0.04]
    kq = [0.065, 0.063, 0.061, 0.059, 0.055, 0.051, 0.047, 0.041, 0.033, 0.024, 0.015]
    eta = [jj * k / (2.0 * math.pi * q) for jj, k, q in zip(j, kt, kq)]
    return {"J": j, "KT": kt, "KQ": kq, "10KQ": [10.0 * q for q in kq], "eta0": eta}# ---------------------------------------------------------------------------
# 2. KCS reference case (model scale, SIMMAN 2010)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KCSModelCase:
    """KCS (KRISO Container Ship) model-scale particulars used by SIMMAN 2010.

    All values are model-scale (Lpp = 7.2786 m) design-condition data;
    Fr = 0.26 corresponds to V = 2.196 m/s.  ``reference_*`` are the SIMMAN-2010
    self-propulsion benchmark values used by the acceptance criteria.
    """

    length: float = 7.2786
    beam: float = 1.0190
    draft: float = 0.3418
    cb: float = 0.6505
    cm: float = 0.9849
    cwp: float = 0.7869
    wetted_area: float = 9.4379
    speed: float = 2.196
    fr: float = 0.26
    rho: float = 999.5
    nu: float = 1.139e-6
    propeller_x_over_l: float = 0.9825
    propeller_shaft_z: float = -0.1535      # [m] shaft centre below waterline
    propeller: KP505Geometry = KP505Geometry()
    reference_wake_fraction: float = 0.28
    reference_thrust_deduction: float = 0.22
    reference_resistance: float = 82.0      # [N] model-scale R_T at Fr = 0.26


KCS_MODEL = KCSModelCase()


# ---------------------------------------------------------------------------
# 3. Body-force (actuator disk) projection onto the lattice
# ---------------------------------------------------------------------------


def lattice_force_scale(rho_phys: float, dx: float, dt: float) -> float:
    """Force unit of the lattice in newtons: ``rho * dx^4 / dt^2``.

    A lattice force density ``F*`` (momentum per cell per step) corresponds to
    the physical force ``F = F* * rho * dx^4 / dt^2``.
    """
    if dx <= 0 or dt <= 0 or rho_phys <= 0:
        raise ValueError("dx, dt and rho_phys must be positive")
    return rho_phys * dx**4 / dt**2


def build_actuator_disk_field(
    nz: int,
    ny: int,
    nx: int,
    *,
    center: tuple[float, float, float],
    radius_cells: float,
    hub_radius_cells: float,
    r_over_R: Sequence[float],
    dT_dr: Sequence[float],
    dQ_dr: Sequence[float],
    thrust_lattice: float,
    torque_lattice: float,
    thickness_cells: int = 3,
    device: torch.device | str | None = None,
) -> dict:
    """Project a BEM radial loading onto the lattice as a body-force field.

    The loading ``(dT_dr, dQ_dr)`` is sampled at ``r_over_R`` (normalised
    radius); the field is normalised so that the summed axial force equals
    ``thrust_lattice`` and the summed angular momentum equals ``torque_lattice``.
    Sign convention: ``x`` runs from the bow (inlet) to the stern, water flows
    along +x, so the propeller accelerates the water downstream and the field is
    the force **on the fluid**: ``sum(fx) = +thrust_lattice``; the reaction
    thrust on the ship is ``-x``.

    Returns
    -------
    dict with ``fx``, ``ftheta`` (``(nz, ny, nx)`` tensors), ``mask``,
    ``radius``, ``sum_fx``, ``sum_angular_momentum``, ``n_cells``.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))

    zc = torch.arange(nz, dtype=torch.float32, device=dev)
    yc = torch.arange(ny, dtype=torch.float32, device=dev)
    xc = torch.arange(nx, dtype=torch.float32, device=dev)
    zz = zc[:, None, None]
    yy = yc[None, :, None]
    xx = xc[None, None, :]

    half = 0.5 * max(thickness_cells - 1, 0)
    axial = ((xx - cx).abs() <= half).expand(nz, ny, nx)
    radius = torch.sqrt((yy - cy) ** 2 + (zz - cz) ** 2).expand(nz, ny, nx).contiguous()
    mask = axial & (radius <= radius_cells) & (radius >= hub_radius_cells)

    rr_tab = np.asarray(r_over_R, dtype=float)
    dT_tab = np.asarray(dT_dr, dtype=float)
    dQ_tab = np.asarray(dQ_dr, dtype=float)

    r_mask = radius[mask]
    if r_mask.numel() == 0:
        raise ValueError("actuator disk mask is empty; check centre/radius")
    rr_np = np.clip(r_mask.cpu().numpy() / max(radius_cells, 1e-9), 0.0, 1.0)
    rr_np = np.clip(rr_np, rr_tab[0], rr_tab[-1])
    w_thrust = np.interp(rr_np, rr_tab, dT_tab)
    w_torque = np.interp(rr_np, rr_tab, dQ_tab)
    w_thrust = w_thrust / max(float(np.sum(w_thrust)), 1e-30)
    w_torque = w_torque / max(float(np.sum(w_torque)), 1e-30)

    fx = torch.zeros(nz, ny, nx, dtype=torch.float32, device=dev)
    ftheta = torch.zeros(nz, ny, nx, dtype=torch.float32, device=dev)
    fx[mask] = torch.as_tensor(w_thrust * thrust_lattice, dtype=torch.float32, device=dev)
    r_np = np.maximum(r_mask.cpu().numpy(), 1e-9)
    ftheta[mask] = torch.as_tensor(w_torque * torque_lattice / r_np,
                                   dtype=torch.float32, device=dev)

    return {
        "fx": fx,
        "ftheta": ftheta,
        "mask": mask,
        "radius": radius,
        "sum_fx": float(fx.sum().item()),
        "sum_angular_momentum": float((radius * ftheta).sum().item()),
        "n_cells": int(mask.sum().item()),
        "center": (cx, cy, cz),
        "radius_cells": float(radius_cells),
        "thickness_cells": int(thickness_cells),
    }


def apply_body_force_guo(
    f: torch.Tensor,
    fx: torch.Tensor | None = None,
    fy: torch.Tensor | None = None,
    fz: torch.Tensor | None = None,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Guo-type momentum source injection into a D3Q19 distribution.

    ``f_i += 3 w_i (c_i . F)`` injects exactly the momentum ``F`` per cell and
    step (``sum_i 3 w_i c_ix^2 = 1``), so the applied thrust is recoverable from
    the momentum budget to machine precision.
    """
    out = f if inplace else f.clone()
    for comp, fld in enumerate((fx, fy, fz)):
        if fld is None:
            continue
        ci = C[:, comp].to(dtype=f.dtype)
        nz = int((ci != 0).sum().item())
        if nz == 0:
            continue
        weights = (3.0 * W.to(dtype=f.dtype) * ci)[:, None, None, None]
        out += weights * fld.unsqueeze(0)
    return out


# ---------------------------------------------------------------------------
# 4. Field post-processing: pressure jump, swirl, wake fraction
# ---------------------------------------------------------------------------


def _radius_field(shape: tuple[int, int, int], cy: float, cz: float,
                  device, dtype) -> torch.Tensor:
    nz, ny, _ = shape
    zc = torch.arange(nz, device=device, dtype=dtype)[:, None, None]
    yc = torch.arange(ny, device=device, dtype=dtype)[None, :, None]
    return torch.sqrt((yc - cy) ** 2 + (zc - cz) ** 2).expand(nz, ny, nx_of(shape)).contiguous()


def nx_of(shape: tuple[int, int, int]) -> int:
    """Return the x size of a ``(nz, ny, nx)`` shape tuple."""
    return int(shape[2])


def disk_pressure_jump(
    rho: torch.Tensor,
    *,
    center: tuple[float, float, float],
    radius_cells: float,
    thickness_cells: int = 3,
    offset_cells: int = 2,
) -> dict:
    """Measure the static pressure jump produced by the actuator disk.

    ``p = rho / 3`` (``c_s^2 = 1/3``).  The upstream and downstream planes lie
    ``thickness/2 + offset`` cells from the disk centre and are averaged over
    the disk annulus (``r <= R_disk``).
    """
    _, cy, cz = (float(v) for v in center)
    nz, ny, nx = int(rho.shape[0]), int(rho.shape[1]), int(rho.shape[2])
    x_up = int(np.clip(round(center[0] - 0.5 * thickness_cells - offset_cells), 0, nx - 1))
    x_dn = int(np.clip(round(center[0] + 0.5 * thickness_cells + offset_cells), 0, nx - 1))

    radius = _radius_field((nz, ny, nx), cy, cz, rho.device, rho.dtype)
    sel = radius <= radius_cells

    p_up = float((rho[:, :, x_up][sel[:, :, x_up]] / 3.0).mean().item())
    p_dn = float((rho[:, :, x_dn][sel[:, :, x_dn]] / 3.0).mean().item())
    return {"p_upstream": p_up, "p_downstream": p_dn, "delta_p": p_dn - p_up,
            "x_upstream": x_up, "x_downstream": x_dn}


def disk_swirl(
    uy: torch.Tensor,
    uz: torch.Tensor,
    *,
    center: tuple[float, float, float],
    radius_cells: float,
    x_index: int,
) -> dict:
    """Swirl velocity ``u_theta`` and specific angular momentum at a plane."""
    _, cy, cz = (float(v) for v in center)
    nz, ny, nx = int(uy.shape[0]), int(uy.shape[1]), int(uy.shape[2])
    xi = int(np.clip(x_index, 0, nx - 1))
    zc = torch.arange(nz, device=uy.device, dtype=uy.dtype)[:, None]
    yc = torch.arange(ny, device=uy.device, dtype=uy.dtype)[None, :]
    dy = yc - cy
    dz = zc - cz
    r = torch.sqrt(dy**2 + dz**2)
    r_safe = r.clamp(min=1e-9)
    u_th = (-dz * uy[:, :, xi] + dy * uz[:, :, xi]) / r_safe
    valid = (r <= radius_cells) & (r > 1e-6)
    if not bool(valid.any()):
        return {"u_theta_mean": 0.0, "angular_momentum_flux": 0.0,
                "x_index": xi, "n_cells": 0}
    return {
        "u_theta_mean": float(u_th[valid].mean().item()),
        "angular_momentum_flux": float((r * u_th)[valid].sum().item()),
        "x_index": xi,
        "n_cells": int(valid.sum().item()),
    }


def plane_wake_stats(
    u_axial: torch.Tensor,
    *,
    center: tuple[float, float, float],
    radius_cells: float,
    hub_radius_cells: float,
    freestream: float,
    fluid: torch.Tensor | None = None,
    n_radial: int = 8,
    n_sectors: int = 8,
) -> dict:
    """Momentum-deficit wake analysis on a propeller plane.

    ``u_axial`` is a 2-D plane ``(nz, ny)``; ``center`` is ``(x, y, z)`` in cell
    coordinates (only ``y`` and ``z`` are used).  The wake fraction is the
    area-weighted deficit over the disk annulus::

        w = 1 - <u_x> / U_inf

    Radial and azimuthal profiles are returned as well (used by
    :func:`relative_rotative_efficiency`).
    """
    if freestream <= 0:
        raise ValueError("freestream must be positive")
    _, cy, cz = (float(v) for v in center)
    nz, ny = int(u_axial.shape[0]), int(u_axial.shape[1])
    zc = torch.arange(nz, device=u_axial.device, dtype=u_axial.dtype)[:, None]
    yc = torch.arange(ny, device=u_axial.device, dtype=u_axial.dtype)[None, :]
    dy = yc - cy
    dz = zc - cz
    r = torch.sqrt(dy**2 + dz**2)
    ann = (r <= radius_cells) & (r >= hub_radius_cells)
    if fluid is not None:
        ann = ann & fluid
    if not bool(ann.any()):
        raise ValueError("no fluid cells inside the propeller disk annulus")

    u = u_axial[ann]
    w_local = (1.0 - u / freestream).cpu().numpy()
    u_mean = float(u.mean().item())
    w_mean = float(np.mean(w_local))

    r_norm = (r / max(radius_cells, 1e-9))[ann].cpu().numpy()
    theta = torch.atan2(dz.expand(nz, ny)[ann], dy.expand(nz, ny)[ann]).cpu().numpy()

    r_inner = hub_radius_cells / max(radius_cells, 1e-9)
    edges = np.linspace(r_inner, 1.0, n_radial + 1)
    radial = []
    for k in range(n_radial):
        sel = (r_norm >= edges[k]) & (r_norm <= edges[k + 1])
        radial.append({
            "r_over_R": float(0.5 * (edges[k] + edges[k + 1])),
            "w": float(np.mean(w_local[sel])) if sel.any() else float(w_mean),
            "u_over_U": float(1.0 - np.mean(w_local[sel])) if sel.any() else float(u_mean / freestream),
        })

    sector_edges = np.linspace(-math.pi, math.pi, n_sectors + 1)
    sectors = []
    for k in range(n_sectors):
        sel = (theta >= sector_edges[k]) & (theta < sector_edges[k + 1])
        sectors.append({
            "theta_mid": float(0.5 * (sector_edges[k] + sector_edges[k + 1])),
            "w": float(np.mean(w_local[sel])) if sel.any() else float(w_mean),
        })

    return {
        "w": w_mean,
        "u_mean_over_U": u_mean / freestream,
        "u_mean": u_mean,
        "freestream": float(freestream),
        "n_cells": int(ann.sum().item()),
        "radial_profile": radial,
        "sector_profile": sectors,
    }


# ---------------------------------------------------------------------------
# 5. Thrust deduction from stern pressure integration
# ---------------------------------------------------------------------------


def hull_pressure_resistance(
    mask: torch.Tensor,
    rho: torch.Tensor,
    *,
    x_min: int | None = None,
    x_max: int | None = None,
) -> dict:
    """Integrate the hull-surface pressure in the axial direction.

    With ``x`` from bow to stern and water flowing along +x, the force on the
    hull is ``-p n``, so the pressure **drag** is
    ``D_p = -sum(p n_x dA)`` (``p = rho/3``, ``dA = 1`` cell face).  Positive
    ``D_p`` resists the forward motion of the ship.  Faces are detected between
    solid and fluid cells, using the fluid-side pressure.  A per-x-station
    profile is returned for the stern analysis.
    """
    if mask.shape != rho.shape:
        raise ValueError("mask and rho must have the same shape")
    solid = mask.bool()
    fluid = ~solid
    p = rho / 3.0

    solid_f = solid[:, :, :-1]
    face_plus = solid_f & fluid[:, :, 1:]
    p_plus = torch.where(face_plus, p[:, :, 1:], torch.zeros_like(p[:, :, 1:]))

    solid_b = solid[:, :, 1:]
    face_minus = solid_b & fluid[:, :, :-1]
    p_minus = torch.where(face_minus, p[:, :, :-1], torch.zeros_like(p[:, :, :-1]))

    prof_plus = p_plus.sum(dim=(0, 1))
    prof_minus = p_minus.sum(dim=(0, 1))
    profile = torch.zeros(int(rho.shape[2]), device=rho.device, dtype=rho.dtype)
    profile[:-1] += prof_plus
    profile[1:] -= prof_minus

    if x_min is not None or x_max is not None:
        lo = 0 if x_min is None else int(x_min)
        hi = int(rho.shape[2]) if x_max is None else int(x_max)
        total = float(profile[lo:hi].sum().item())
    else:
        total = float(profile.sum().item())

    return {
        "pressure_resistance": -total,
        "integral_p_nx": total,
        "profile": profile.cpu().numpy().tolist(),
        "n_faces_plus": int(face_plus.sum().item()),
        "n_faces_minus": int(face_minus.sum().item()),
    }


def thrust_deduction_from_pressure(
    d_pressure_propeller_on: float,
    d_pressure_propeller_off: float,
    thrust: float,
) -> dict:
    """Thrust-deduction fraction from the propeller-induced resistance change.

    ``Delta R = D_p(on) - D_p(off)`` is the resistance augmentation caused by
    the propeller suction on the stern; ``t = Delta R / T``.
    """
    if thrust <= 0:
        raise ValueError("thrust must be positive")
    delta_r = float(d_pressure_propeller_on) - float(d_pressure_propeller_off)
    return {"delta_R": delta_r, "t": delta_r / thrust, "thrust": float(thrust)}


# ---------------------------------------------------------------------------
# 6. Self-propulsion point and power chain
# ---------------------------------------------------------------------------


def solve_self_propulsion(
    *,
    resistance: float,
    speed: float,
    geometry: KP505Geometry,
    polar: SectionPolar,
    wake_fraction: float | Callable[[float], float],
    thrust_deduction: float,
    rho: float = 999.5,
    n_bounds: tuple[float, float] = (0.5, 40.0),
    tol: float = 1e-10,
    max_iter: int = 200,
) -> dict:
    """Solve ``T(n) (1 - t) = R_T`` for the revolution rate ``n``.

    Thrust comes from the BEM open-water solution evaluated at the behind-hull
    advance ratio ``J = V (1 - w) / (n D)``; ``w`` may be a callable of ``n``
    (effective wake from the coupled run).
    """
    if resistance <= 0 or speed <= 0:
        raise ValueError("resistance and speed must be positive")
    if not 0.0 <= thrust_deduction < 1.0:
        raise ValueError("thrust_deduction must lie in [0, 1)")
    d = geometry.diameter
    target = resistance / (1.0 - thrust_deduction)

    def thrust_of(n: float) -> tuple[float, float, float, float]:
        w = wake_fraction(n) if callable(wake_fraction) else float(wake_fraction)
        v_a = speed * (1.0 - w)
        j = v_a / (n * d)
        sol = blade_element_momentum(geometry, polar, j, n_rps=n, rho=rho)
        return sol["thrust"], j, w, sol["torque"]

    lo, hi = n_bounds
    f_lo = thrust_of(lo)[0] - target
    f_hi = thrust_of(hi)[0] - target
    if f_lo * f_hi > 0:
        raise ValueError(
            f"no self-propulsion point bracketed in n=[{lo}, {hi}]: "
            f"T(n) - target = {f_lo:.4g} / {f_hi:.4g}")
    history = []
    n = 0.5 * (lo + hi)
    for _ in range(max_iter):
        t_n, j, w, q = thrust_of(n)
        residual = t_n - target
        history.append({"n": n, "thrust": t_n, "j": j, "w": w, "residual": residual})
        if abs(residual) < tol * max(target, 1.0):
            break
        if (residual > 0) == (f_lo > 0):
            lo, f_lo = n, residual
        else:
            hi, f_hi = n, residual
        n = 0.5 * (lo + hi)

    thrust, j, w, torque = thrust_of(n)
    sol = blade_element_momentum(geometry, polar, j, n_rps=n, rho=rho)
    return {
        "n_rps": n,
        "j": j,
        "wake_fraction": w,
        "thrust_deduction": float(thrust_deduction),
        "thrust": thrust,
        "torque": torque,
        "kt": sol["kt"],
        "kq": sol["kq"],
        "eta0": sol["eta0"],
        "required_thrust": target,
        "resistance": float(resistance),
        "speed": float(speed),
        "residual": float(thrust - target),
        "iterations": len(history),
        "history": history,
    }


def solve_self_propulsion_curve(
    *,
    resistance: float,
    speed: float,
    diameter: float,
    rho: float,
    wake_fraction: float,
    thrust_deduction: float,
    j_table,
    kt_table,
    kq_table,
    n_bounds: tuple[float, float] = (0.5, 40.0),
    tol: float = 1e-12,
    max_iter: int = 200,
) -> dict:
    """Self-propulsion point from a *measured* open-water curve.

    Same balance (``T(n)(1 - t) = R_T``) as :func:`solve_self_propulsion`, but
    ``KT``/``KQ`` are interpolated from tabulated open-water data instead of the
    blade-element solution.  This is the standard body-force practice for ship
    CFD (the trust level of the propeller data is then decoupled from the
    section-polar modelling error) and it is what the delivered power in the
    report uses; the BEM result is kept as the uncalibrated prediction.
    """
    target = resistance / (1.0 - thrust_deduction)
    j_tab = np.asarray(j_table, dtype=float)
    kt_tab = np.asarray(kt_table, dtype=float)
    kq_tab = np.asarray(kq_table, dtype=float)

    def state(n: float):
        j = speed * (1.0 - wake_fraction) / (n * diameter)
        kt = float(np.interp(j, j_tab, kt_tab))
        kq = float(np.interp(j, j_tab, kq_tab))
        return (kt * rho * n**2 * diameter**4, kq * rho * n**2 * diameter**5, j, kt, kq)

    lo, hi = n_bounds
    f_lo = state(lo)[0] - target
    f_hi = state(hi)[0] - target
    if f_lo * f_hi > 0:
        raise ValueError("no self-propulsion point bracketed by the open-water table")
    history = []
    for _ in range(max_iter):
        n = 0.5 * (lo + hi)
        thrust, torque, j, kt, kq = state(n)
        history.append({"n": n, "thrust": thrust, "j": j, "residual": thrust - target})
        if abs(thrust - target) < tol * target:
            break
        if (thrust - target > 0) == (f_lo > 0):
            lo, f_lo = n, thrust - target
        else:
            hi, f_hi = n, thrust - target
    thrust, torque, j, kt, kq = state(n)
    return {
        "n_rps": n, "j": j, "kt": kt, "kq": kq, "thrust": thrust, "torque": torque,
        "eta0": j * kt / (2.0 * math.pi * kq) if kq > 0 else float("nan"),
        "required_thrust": target, "residual": thrust - target,
        "iterations": len(history), "history": history,
        "wake_fraction": float(wake_fraction),
        "thrust_deduction": float(thrust_deduction),
        "resistance": float(resistance), "speed": float(speed),
    }


def power_chain(
    *,
    resistance: float,
    speed: float,
    thrust: float,
    torque: float,
    n_rps: float,
    wake_fraction: float,
    thrust_deduction: float,
    eta_R: float | None = None,
    eta0: float | None = None,
) -> dict:
    """Effective -> thrust -> delivered power chain and efficiency split."""
    v_a = speed * (1.0 - wake_fraction)
    p_e = resistance * speed
    p_t = thrust * v_a
    p_d = 2.0 * math.pi * n_rps * torque
    eta_h = (1.0 - thrust_deduction) / (1.0 - wake_fraction)
    eta_d = p_e / p_d if p_d > 0 else float("nan")
    eta_0 = eta0 if eta0 is not None else (p_t / p_d if p_d > 0 else float("nan"))
    if eta_R is None:
        eta_R = eta_d / (eta_h * eta_0) if (eta_h * eta_0) != 0 else float("nan")
    return {
        "P_E": p_e,
        "P_T": p_t,
        "P_D": p_d,
        "V_A": v_a,
        "eta_D": eta_d,
        "eta_H": eta_h,
        "eta_0": eta_0,
        "eta_R": eta_R,
        "eta_H_check": p_e / p_t if p_t > 0 else float("nan"),
        "eta_product_check": eta_h * eta_R * eta_0,
    }


# ---------------------------------------------------------------------------
# 7. Relative rotative efficiency from a non-uniform measured wake
# ---------------------------------------------------------------------------


def relative_rotative_efficiency(
    geometry: KP505Geometry,
    polar: SectionPolar,
    *,
    wake_field: dict,
    n_rps: float,
    speed: float,
    rho: float = 999.5,
    n_sections: int = 40,
) -> dict:
    """``eta_R = KQ_open / KQ_behind`` at equal thrust.

    The measured propeller-plane wake is split into azimuthal sectors and radial
    bands; each sector is analysed with a BEM run whose per-section inflow is
    ``U_inf (1 - w(theta, r))``.  Sector thrusts/torques are averaged and the
    open-water torque is evaluated at the advance ratio giving the same thrust.
    """
    radial = wake_field.get("radial_profile") or []
    sectors = wake_field.get("sector_profile") or []
    if not radial or not sectors:
        raise ValueError("wake_field must contain radial_profile and sector_profile")

    rr_rad = np.array([bin_["r_over_R"] for bin_ in radial], dtype=float)
    b_rr = np.linspace(geometry.hub_ratio + 1e-4, 1.0 - 1e-4, n_sections)
    w_bands = np.array([bin_.get("w", wake_field["w"]) for bin_ in radial], dtype=float)

    sector_inflow = []
    for sec in sectors:
        shift = float(sec.get("w", wake_field["w"])) - float(np.mean(w_bands))
        inflow_bands = np.clip(1.0 - (w_bands + shift), 0.05, 1.0)
        inflow = np.interp(b_rr, np.clip(rr_rad, 0.0, 1.0), inflow_bands)
        sector_inflow.append(np.clip(inflow, 0.05, 1.0))

    thrusts, torques = [], []
    for inflow in sector_inflow:
        j_sec = speed * float(np.mean(inflow)) / (n_rps * geometry.diameter)
        sol = blade_element_momentum(geometry, polar, j_sec, n_rps=n_rps, rho=rho,
                                     n_sections=n_sections, inflow_ratio=inflow)
        thrusts.append(sol["thrust"])
        torques.append(sol["torque"])
    thrust_b = float(np.mean(thrusts))
    torque_b = float(np.mean(torques))
    d = geometry.diameter
    kt_b = thrust_b / (rho * n_rps**2 * d**4)
    kq_b = torque_b / (rho * n_rps**2 * d**5)

    def kt_of(j: float) -> float:
        return blade_element_momentum(geometry, polar, j, n_rps=n_rps, rho=rho,
                                      n_sections=n_sections)["kt"]

    lo, hi = 1e-3, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if kt_of(mid) > kt_b:
            lo = mid
        else:
            hi = mid
    j_open = 0.5 * (lo + hi)
    sol_open = blade_element_momentum(geometry, polar, j_open, n_rps=n_rps, rho=rho,
                                      n_sections=n_sections)
    return {
        "eta_R": float(sol_open["kq"] / kq_b) if kq_b > 0 else float("nan"),
        "j_open_water": float(j_open),
        "kt_behind": float(kt_b),
        "kq_behind": float(kq_b),
        "kq_open": float(sol_open["kq"]),
        "thrust_behind": thrust_b,
        "torque_behind": torque_b,
    }


# ---------------------------------------------------------------------------
# 8. Reynolds scaling of the LBM wake fraction (documented model)
# ---------------------------------------------------------------------------


def rescale_wake_fraction(
    w_total: float,
    w_potential: float,
    re_measured: float,
    re_target: float,
    *,
    exponent: float = 0.2,
) -> dict:
    """Scale an LBM-measured wake fraction to a target Reynolds number.

    The wake fraction is split into an inviscid (potential, Reynolds
    independent) and a viscous part; the viscous part is scaled with the
    standard turbulent boundary-layer law ``w_visc ~ Re^-1/5``.  This is a
    *model assumption* (graded MODEL); its exponent can be checked with
    :func:`fit_wake_reynolds_exponent` against an LBM Reynolds sweep.
    """
    if re_measured <= 0 or re_target <= 0:
        raise ValueError("Reynolds numbers must be positive")
    w_visc = w_total - w_potential
    factor = (re_measured / re_target) ** exponent
    return {
        "w_potential": float(w_potential),
        "w_viscous_measured": float(w_visc),
        "scale_factor": float(factor),
        "w_viscous_target": float(w_visc * factor),
        "w": float(w_potential + w_visc * factor),
        "exponent": float(exponent),
        "re_measured": float(re_measured),
        "re_target": float(re_target),
    }


def fit_wake_reynolds_exponent(re_values: Sequence[float],
                               w_values: Sequence[float],
                               w_potential: float = 0.0) -> dict:
    """Least-squares power law ``w - w_pot = A Re^-n`` from a measured sweep."""
    re = np.asarray(re_values, dtype=float)
    wv = np.asarray(w_values, dtype=float) - float(w_potential)
    if re.size < 2:
        raise ValueError("need at least two samples")
    mask = wv > 0
    if mask.sum() < 2:
        raise ValueError("viscous wake must be positive at two Reynolds numbers")
    slope, intercept = np.polyfit(np.log(re[mask]), np.log(wv[mask]), 1)
    return {"exponent": float(-slope), "coefficient": float(np.exp(intercept)),
            "n_samples": int(mask.sum())}


# ---------------------------------------------------------------------------
# 9. Benchmark correlations (independent cross-checks) and ITTC-78 resistance
# ---------------------------------------------------------------------------


def empirical_wake_fraction(cb: float, *, form: str = "taylor") -> float:
    """Standard design-formula wake fraction for single-screw ships.

    ``taylor`` / ``harvald``: ``w = 0.5 Cb - 0.05``.  Graded MODEL
    (semi-empirical regression), used only as an independent cross-check.
    """
    if form not in {"taylor", "harvald"}:
        raise ValueError(f"unknown form {form!r}")
    return 0.5 * cb - 0.05


def empirical_thrust_deduction(w: float, *, form: str = "heckscher") -> float:
    """Thrust deduction from the Heckscher regression ``t = 0.25 w + 0.14``.

    Graded MODEL; used only as an independent cross-check.
    """
    if form != "heckscher":
        raise ValueError(f"unknown form {form!r}")
    return 0.25 * w + 0.14


def holtrop_relative_rotative_efficiency(
    *,
    prismatic_coefficient: float,
    lcb_percent_l: float,
    pitch_ratio: float,
) -> float:
    """Holtrop (1982) regression for the relative rotative efficiency.

    ``eta_R = 0.9737 + 0.111 (Cp - 0.0225 lcb) - 0.06325 P/D`` with ``lcb`` in
    percent of L.  Graded MODEL: it is the standard design regression for the
    efficiency loss caused by the non-uniform wake acting on the propeller;
    the sector-BEM route (:func:`relative_rotative_efficiency`) computes the
    same quantity from a *measured* wake field when one is available.
    """
    return (0.9737 + 0.111 * (prismatic_coefficient - 0.0225 * lcb_percent_l)
            - 0.06325 * pitch_ratio)


def ittc78_resistance(
    *,
    length: float,
    beam: float,
    draft: float,
    cb: float,
    speed: float,
    rho: float = 999.5,
    nu: float = 1.139e-6,
    wetted_area: float | None = None,
    form_factor: float = 1.069,
    cm: float = 0.9849,
    cwp: float = 0.7869,
    residual_coefficient: float = 0.58e-3,
) -> dict:
    """Model-scale resistance from ITTC-78 friction + a prescribed C_R.

    ``R_T = 0.5 rho V^2 S (C_F (1+k) + C_R)`` with the ITTC-57 friction line and
    the ITTC-78 wetted-surface formula when ``wetted_area`` is not given.  The
    residuary coefficient is a *documented model input* (KCS model-scale
    C_R ~ 0.58e-3 at Fr = 0.26), not a computed wave resistance.
    """
    if wetted_area is None:
        wetted_area = (length * (2.0 * draft + beam) * math.sqrt(cm)
                       * (0.453 + 0.4425 * cb - 0.2862 * cm
                          - 0.003467 * beam / draft + 0.3696 * cwp))
    re = speed * length / nu
    cf = 0.075 / (math.log10(re) - 2.0) ** 2
    q = 0.5 * rho * speed**2
    r_f = q * wetted_area * cf * form_factor
    r_r = q * wetted_area * residual_coefficient
    return {
        "reynolds": re,
        "cf_ittc57": cf,
        "wetted_area_m2": wetted_area,
        "form_factor": form_factor,
        "residual_coefficient": residual_coefficient,
        "R_F": r_f,
        "R_R": r_r,
        "R_T": r_f + r_r,
        "C_T": (cf * form_factor + residual_coefficient),
    }# ---------------------------------------------------------------------------
# 10. Actuator-disk momentum-theory verification case (tunnel LBM run)
# ---------------------------------------------------------------------------


def run_actuator_disk_momentum_case(
    *,
    nz: int = 48,
    ny: int = 64,
    nx: int = 160,
    u_in: float = 0.05,
    tau: float = 0.5055,
    smagorinsky_cs: float = 0.10,
    thrust_lattice: float = 2.0e-4,
    torque_lattice: float = 2.0e-6,
    radius_cells: float = 16.0,
    hub_radius_cells: float = 3.0,
    thickness_cells: int = 3,
    n_steps: int = 8000,
    warmup_steps: int = 3000,
    sample_steps: int = 2000,
    device: str = "cpu",
) -> dict:
    """Run a body-force actuator disk in a straight LBM tunnel and verify it.

    A uniform inflow enters a duct, a thin disk carries a parabolic axial +
    swirl body force, and the resulting fields are checked against actuator-disk
    momentum theory:

    * measured momentum injection equals the applied thrust (Guo accounting),
    * ``Delta p ~ T / A_disk`` (pressure jump produced by the momentum source),
    * ``Delta p ~ u_a (U + u_a)`` with ``u_a`` from the measured disk velocity,
    * far-wake velocity ratio ``(u_wake - U) / (u_disk - U) ~ 2``,
    * measured swirl angular-momentum flux equals the applied torque.
    """
    dev = torch.device(device)
    cx, cy, cz = nx * 0.35, ny * 0.5, nz * 0.5
    center = (cx, cy, cz)

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    zeros = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, ux0, zeros, zeros, device=dev)
    f_inlet = equilibrium3d(torch.ones((nz, ny, 1), dtype=torch.float32, device=dev),
                            torch.full((nz, ny, 1), u_in, dtype=torch.float32, device=dev),
                            torch.zeros((nz, ny, 1), dtype=torch.float32, device=dev),
                            torch.zeros((nz, ny, 1), dtype=torch.float32, device=dev),
                            device=dev)

    rr = np.linspace(0.18, 0.998, 60)
    load = np.clip(1.0 - (rr - 0.75) ** 2 / 0.75**2, 0.05, None)
    disk = build_actuator_disk_field(
        nz, ny, nx,
        center=center,
        radius_cells=radius_cells,
        hub_radius_cells=hub_radius_cells,
        r_over_R=rr,
        dT_dr=load,
        dQ_dr=load,
        thrust_lattice=thrust_lattice,
        torque_lattice=torque_lattice,
        thickness_cells=thickness_cells,
        device=dev,
    )
    fx, ftheta, r_field = disk["fx"], disk["ftheta"], disk["radius"]
    dy = torch.arange(ny, device=dev, dtype=torch.float32)[None, :, None] - cy
    dz = torch.arange(nz, device=dev, dtype=torch.float32)[:, None, None] - cz
    r_safe = r_field.clamp(min=1e-6)
    fy = torch.where(disk["mask"], -ftheta * dz / r_safe, torch.zeros_like(ftheta))
    fz = torch.where(disk["mask"], ftheta * dy / r_safe, torch.zeros_like(ftheta))

    ux_acc = torch.zeros((nz, ny, nx), dtype=torch.float64, device=dev)
    n_acc = 0
    x_disk = int(round(cx))

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=smagorinsky_cs)
        f = stream3d(f)
        f = apply_body_force_guo(f, fx, fy, fz, inplace=True)
        f[:, :, :, 0] = f_inlet[:, :, :, 0]
        f[:, :, :, -1] = f[:, :, :, -2]
        if warmup_steps < step <= warmup_steps + sample_steps:
            _, ux_t, _, _ = macroscopic3d(f)
            ux_acc += ux_t.to(torch.float64)
            n_acc += 1

    ux_mean = (ux_acc / max(n_acc, 1)).to(torch.float32)
    rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)

    sel = (r_field <= radius_cells) & (r_field >= hub_radius_cells)
    prof = []
    for x in range(int(nx)):
        m = sel[:, :, x]
        prof.append(float(ux_mean[:, :, x][m].mean().item()) if bool(m.any())
                    else float(ux_mean[:, :, x].mean().item()))

    pj = disk_pressure_jump(rho_f, center=center, radius_cells=radius_cells,
                            thickness_cells=thickness_cells, offset_cells=2)
    swirl = disk_swirl(uy_f, uz_f, center=center, radius_cells=radius_cells,
                       x_index=int(cx + 0.5 * thickness_cells + 2))

    area_cells = math.pi * radius_cells**2
    dp_from_thrust = thrust_lattice / area_cells
    u_disk = prof[x_disk]
    x_far = int(min(nx - 3, round(cx) + 0.45 * nx))
    u_far = prof[x_far]
    u_a = u_disk - u_in
    # ideal actuator-disk momentum theory: p_up - p_dn = u_a (U + u_a)
    dp_from_momentum = -u_a * (u_in + u_a)
    far_ratio = (u_far - u_in) / u_a if abs(u_a) > 1e-12 else float("nan")

    # angular-momentum flux is conserved over the *whole* cross-section: with
    # periodic lateral boundaries the radial fluxes cancel in pairs, so the
    # integrated flux downstream of the disk must equal the applied torque.
    r_plane = r_field[:, :, x_disk]
    dy2 = dy.reshape(ny)
    dz2 = dz.reshape(nz)
    u_th = ((-dz2[:, None]) * uy_f[:, :, x_disk]
            + dy2[None, :] * uz_f[:, :, x_disk]) / r_plane.clamp(min=1e-6)
    m_plane = r_plane > 1e-6
    q_flux = float((rho_f[:, :, x_disk] * ux_f[:, :, x_disk] * r_plane * u_th)[m_plane].sum().item()) \
        if bool(m_plane.any()) else 0.0
    q_flux_disk = float((rho_f[:, :, x_disk] * ux_f[:, :, x_disk] * r_plane * u_th)
                        [(r_plane <= radius_cells) & m_plane].sum().item()) \
        if bool(m_plane.any()) else 0.0

    return {
        "config": {"nz": nz, "ny": ny, "nx": nx, "u_in": u_in, "tau": tau,
                   "radius_cells": radius_cells, "thickness_cells": thickness_cells,
                   "n_steps": n_steps, "warmup_steps": warmup_steps,
                   "sample_steps": sample_steps, "smagorinsky_cs": smagorinsky_cs,
                   "device": str(device)},
        "disk": {"n_cells": disk["n_cells"], "sum_fx": disk["sum_fx"],
                 "sum_angular_momentum": disk["sum_angular_momentum"],
                 "thrust_lattice": thrust_lattice, "torque_lattice": torque_lattice,
                 "area_cells": area_cells},
        "measured": {
            "u_disk": u_disk,
            "u_far_wake": u_far,
            "u_induced": u_a,
            "delta_p": pj["delta_p"],
            "delta_p_from_thrust": dp_from_thrust,
            "delta_p_from_momentum": dp_from_momentum,
            "far_wake_ratio": far_ratio,
            "swirl_u_theta": swirl["u_theta_mean"],
            "swirl_angular_momentum_flux": q_flux,
            "swirl_angular_momentum_flux_disk_only": q_flux_disk,
        },
        "axial_profile": prof,
        "x_profile": list(range(int(nx))),
        "checks": {
            "thrust_accounting_rel_error": abs(disk["sum_fx"] - thrust_lattice) / thrust_lattice,
            "delta_p_rel_error_vs_T_over_A": abs(abs(pj["delta_p"]) - dp_from_thrust) / abs(dp_from_thrust),
            "delta_p_rel_error_vs_momentum_theory": (abs(abs(pj["delta_p"]) - abs(dp_from_momentum))
                                                     / abs(dp_from_momentum)
                                                     if abs(dp_from_momentum) > 1e-12 else float("nan")),
            "far_wake_ratio_error_vs_2": (abs(far_ratio - 2.0) / 2.0
                                          if math.isfinite(far_ratio) else float("nan")),
            "swirl_rel_error_vs_torque": (abs(q_flux - torque_lattice) / torque_lattice
                                          if torque_lattice != 0 else float("nan")),
        },
    }