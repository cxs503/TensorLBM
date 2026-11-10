"""KVLCC2 self-propulsion with a sliding-mesh (rotating-zone) propeller.

This module is the *ship* half of the two-zone sliding-mesh capability added in
:mod:`tensorlbm.sliding_mesh_propulsion`:

* a **static-zone hull run** (KVLCC2 double body, D3Q19 + Smagorinsky) that
  gives the hull resistance and the *nominal* wake field at the propeller plane;
* a **rotor-zone propeller run** (Wageningen B4-70 geometry, sliding interface)
  that gives the open-water ``K_T(J)``/``K_Q(J)`` and the behind-hull loads when
  the measured wake field is used as a non-uniform inflow;
* the classical decomposition, computed only from those two runs plus the
  momentum-theory identity:

      thrust identity:  K_T^B = K_T^0(J_T)  ->  w_T = 1 - J_T n D / V
      thrust deduction: t = (R_T^SP - R_T^tow) / T
      self-propulsion:  T (1 - t) = R_T,       n = n_sp
      power chain:      eta_D = eta_H eta_R eta_0,  eta_H = (1-t)/(1-w)

Evidence grading (declared per number in the report JSON)
--------------------------------------------------------
REAL: sliding-mesh propeller loads, hull momentum-exchange resistance, thrust
      identity, self-propulsion root find, power-chain arithmetic.
MODEL: the ITTC-78 resistance split (``C_R`` is a prescribed input), the
      Reynolds scaling used to lift the lattice wake fraction to model scale,
      and the coarse resolution of the hull zone (see ``docs/``).
WITHHELD: an EFD-grade adjudication of ``w``/``t`` needs the KRISO model-test
      table, which is not in this repository; literature bands are used and
      flagged as such (see :data:`KVLCC2_REFERENCE_BANDS`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .propeller_cad import PropellerGeometryConfig
from .selfpropulsion import (
    empirical_thrust_deduction,
    empirical_wake_fraction,
    ittc78_resistance,
    power_chain,
)
from .ship_cad import build_hull_mask
from .sliding_mesh_propulsion import RotatingZoneConfig, SlidingMeshPropeller

__all__ = [
    "KVLCC2Model",
    "KVLCC2_MODEL",
    "KVLCC2_REFERENCE_BANDS",
    "b4_70_geometry",
    "run_hull_zone",
    "disk_plane_wake",
    "radial_loading",
    "thrust_identity_wake",
    "thrust_deduction",
    "solve_self_propulsion",
    "reynolds_wake_scaling",
]


# --------------------------------------------------------------------------- #
# Case particulars
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KVLCC2Model:
    """KVLCC2 model-scale particulars and the propeller used in this case.

    The hull is the KRISO KVLCC2 tanker (``Cb = 0.8098``) at model scale
    ``lambda = 58`` (``Lpp = 5.5172 m``).  The propulsor is the
    **mission-specified Wageningen B4-70** (``Z = 4``, ``A_E/A_0 = 0.70``), not
    the ``A_E/A_0 = 0.431`` propeller of the KRISO campaign -- the blade-area
    substitution is declared here because it changes ``K_T`` at fixed ``J``.
    """

    length: float = 5.5172  # [m] model Lpp, lambda = 58
    beam: float = 1.0  # [m]
    draft: float = 0.3586  # [m]
    cb: float = 0.8098
    cm: float = 0.9950
    cwp: float = 0.9153
    wetted_area: float = 8.083  # [m^2]
    speed: float = 1.0468  # [m/s]  Fr = 0.1423 (design speed, 15.5 kn full scale)
    fr: float = 0.1423
    rho: float = 999.5  # [kg/m^3]
    nu: float = 1.139e-6  # [m^2/s]
    form_factor: float = 1.35  # (1+k) KVLCC2 tanker
    residual_coefficient: float = 0.30e-3  # [MODEL] prescribed C_R at Fr = 0.142
    propeller_diameter: float = 0.17  # [m] D/Lpp = 0.0308
    propeller_blades: int = 4
    propeller_ae_a0: float = 0.70  # B4-70
    propeller_pitch_ratio: float = 0.721  # P/D at 0.7R
    propeller_hub_ratio: float = 0.155
    propeller_x_over_l: float = 0.9825
    shaft_z_over_t: float = 0.29  # shaft centre above keel / draft

    @property
    def shaft_depth(self) -> float:
        """Shaft immersion below the waterline [m]."""
        return self.draft * (1.0 - self.shaft_z_over_t)

    @property
    def reynolds(self) -> float:
        return self.speed * self.length / self.nu

    @property
    def foude(self) -> float:
        return self.speed / math.sqrt(9.81 * self.length)


KVLCC2_MODEL = KVLCC2Model()

#: Literature reference bands, used as the adjudication target.
#: Provenance: KVLCC2 self-propulsion model tests reported in the Tokyo-2015 /
#: SIMMAN workshop series.  The exact tabulated digits are *not* available in
#: this repository, so a +/-10% band around the widely quoted values
#: (``w_T = 0.35``, ``t = 0.20``) is used and the adjudication is graded
#: ``WITHHELD-EFD`` rather than REAL.
KVLCC2_REFERENCE_BANDS: dict[str, dict] = {
    "wake_fraction": {
        "value": 0.35,
        "band": (0.315, 0.385),
        "source": "KVLCC2 self-propulsion model tests (Tokyo-2015 workshop series)",
        "grade": "WITHHELD-EFD",
    },
    "thrust_deduction": {
        "value": 0.20,
        "band": (0.17, 0.23),
        "source": "KVLCC2 self-propulsion model tests (Tokyo-2015 workshop series)",
        "grade": "WITHHELD-EFD",
    },
}


def b4_70_geometry(diameter: float) -> PropellerGeometryConfig:
    """Wageningen B4-70 blade geometry at a given lattice diameter."""
    return PropellerGeometryConfig(
        n_blades=4,
        diameter=float(diameter),
        hub_diameter_ratio=0.155,
        hub_length_ratio=0.40,
        pitch_ratio_07=0.721,
        blade_area_ratio=0.70,
        skew_deg=0.0,
        rake_ratio=0.0,
        max_thickness_ratio=0.05,
    )


# --------------------------------------------------------------------------- #
# Hull zone (static zone of the two-zone solver)
# --------------------------------------------------------------------------- #


def _hull_layout(nx: int, ny: int, nz: int, *, fill: float = 0.55):
    """Return ``(length, beam, draft, cx, cy, cz_keel)`` in lattice units.

    The hull is placed so that the aft body (where the propeller lives) is well
    upstream of the outlet and the waterline is the top of the domain.
    """
    length = nx * fill
    beam = length * (1.0 / 5.5172)  # model B/L
    draft = length * (0.3586 / 5.5172)
    cx = nx * 0.5
    cy = ny * 0.5
    cz_keel = 0.5 * (nz - draft)
    return length, beam, draft, cx, cy, cz_keel


def run_hull_zone(
    *,
    nx: int = 140,
    ny: int = 56,
    nz: int = 40,
    inflow: float = 0.05,
    tau: float = 0.502,
    smagorinsky: float = 0.12,
    steps: int = 6000,
    warmup: int = 3000,
    disk: dict | None = None,
    device: str = "cpu",
) -> dict:
    """KVLCC2 double-body run in the static zone.

    Parameters
    ----------
    disk:
        Optional actuator-disk loading used as the *propeller-on* configuration:
        ``{"x_over_l": float, "diameter_over_l": float, "thrust": float,
        "torque": float}`` in lattice units.  The loading is distributed over a
        3-cell-thick disk with a radial shape taken from a B4-70 circulation
        estimate (see :func:`disk_body_force`), momentum-conserving via Guo
        forcing, so that ``R_T^SP - R_T^tow`` measures the thrust deduction.
    """
    length, beam, draft, cx, cy, cz_keel = _hull_layout(nx, ny, nz)
    mask, stats = build_hull_mask(
        "kvlcc2",
        nx,
        ny,
        nz,
        cx=cx,
        cy=cy,
        cz_keel=cz_keel,
        length=length,
        beam=beam,
        draft=draft,
        device=device,
    )
    cfg = RotatingZoneConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        inflow=inflow,
        omega=0.0,
        tau=tau,
        smagorinsky=smagorinsky,
        zone_radius=1.0,  # no rotor zone in the hull-only run
        n_theta=8,
        sides="free_slip",
        streamwise="inlet_outlet",
        slip_planes=("z_min",),
    )
    solver = SlidingMeshPropeller(
        cfg, None, center=(cx, cy, nz * 0.5), outer_solid=mask, device=device
    )
    if disk is not None:
        fx, fy, fz, region = disk_body_force(solver, **disk)
        solver.set_body_force(region, fx, fy, fz)

    res = solver.run(steps, warmup=warmup, measure_every=5)
    rho, ux, uy, uz = solver.macroscopic()
    x_disk = int(round(nx * (disk["x_over_l"] if disk else 0.9825)))
    x_disk = min(max(x_disk, 2), nx - 2)
    wake = disk_plane_wake(solver, x_disk)

    # axial force on the hull = -(momentum exchange along +x)
    resistance = -res["thrust_lattice"]
    dens = float(rho.mean().item())
    return {
        "resistance_lattice": resistance,
        "density_mean": dens,
        "hull_stats": stats,
        "geometry_ok": abs(stats["Cb_numerical"] - 0.8098) < 0.02,
        "cb_numerical": stats["Cb_numerical"],
        "solid_cells": stats["solid_cells"],
        "wake": wake,
        "samples": res["samples"],
        "steps": steps,
        "warmup": warmup,
        "u_mean": float(ux.mean().item()),
        "mass_drift": (solver.mass() - solver._mass0) / solver._mass0,
        "x_disk": x_disk,
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "lattice": {
            "length": length,
            "beam": beam,
            "draft": draft,
            "re_lattice": inflow * length / cfg.nu,
        },
    }


def disk_body_force(
    solver: SlidingMeshPropeller,
    *,
    x_over_l: float,
    diameter_over_l: float,
    thrust: float,
    torque: float,
    n_rps: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Momentum-conserving actuator-disk loading for the hull zone.

    The axial loading follows the classical B4-70 radial circulation shape
    ``dK_T/d(r/R) ~ (r/R) sqrt(1 - r/R)`` normalised so that the integral equals
    the requested thrust; the swirl follows ``dQ/dr ~ (r/R)^2`` normalised to
    the requested shaft torque.  Both are smoothed over three lattice cells in
    the axial direction so that the lattice can carry the pressure jump.
    """
    cfg = solver.cfg
    x0 = x_over_l * cfg.nx
    radius = 0.5 * diameter_over_l
    dy, dz = solver.dy, solver.dz
    r = solver.r
    eta = torch.clamp(r / radius, 0.0, 1.0)
    axial_shape = eta * torch.sqrt(torch.clamp(1.0 - eta, min=0.0))
    swirl_shape = eta * eta
    region = (
        (torch.abs(torch.arange(cfg.nx, dtype=torch.float32) - x0).view(1, 1, cfg.nx) <= 1.5)
        & (r <= radius)
        & (r > 0.155 * radius)
    )
    region = region & (~solver.solid)

    norm_a = float((axial_shape * region).sum().item())
    norm_s = float((swirl_shape * region * r).sum().item())
    f_axial = torch.zeros_like(r)
    f_swirl = torch.zeros_like(r)
    if norm_a > 0:
        f_axial = axial_shape * (thrust / norm_a)
    if norm_s > 0:
        f_swirl = swirl_shape * (-torque / norm_s)
    # swirl: tangential direction (0, -dz, dy) / r
    r_safe = r.clamp(min=1e-6)
    fx = torch.where(region, f_axial, torch.zeros_like(f_axial))
    fy = torch.where(region, f_swirl * (-dz / r_safe), torch.zeros_like(f_axial))
    fz = torch.where(region, f_swirl * (dy / r_safe), torch.zeros_like(f_axial))
    return fx, fy, fz, region


def disk_plane_wake(solver: SlidingMeshPropeller, x_index: int, nbins: int = 16) -> dict:
    """Radial distribution of the axial velocity at a cross plane."""
    rho, ux, uy, uz = solver.macroscopic()
    plane = ux[:, :, x_index]
    r_plane = solver.r[:, :, x_index]
    rmax = float(r_plane.max().item())
    edges = torch.linspace(0.0, rmax, nbins + 1)
    profile = []
    for b in range(nbins):
        m = (r_plane >= edges[b]) & (r_plane < edges[b + 1])
        if bool(m.any()):
            profile.append(
                {
                    "r": float((0.5 * (edges[b] + edges[b + 1])).item()),
                    "u_axial": float(plane[m].mean().item()),
                    "samples": int(m.sum().item()),
                }
            )
    field = plane.detach().cpu().numpy().tolist()
    return {"x_index": int(x_index), "profile": profile, "field": field}


# --------------------------------------------------------------------------- #
# Decomposition helpers
# --------------------------------------------------------------------------- #


def radial_loading(
    solver: SlidingMeshPropeller,
    *,
    hub_radius: float | None = None,
    radius: float | None = None,
    nbins: int = 10,
) -> list[dict]:
    """Radial ``dT/dr`` and ``dQ/dr`` of the resolved propeller loads."""
    force = solver._bounce_back_field(solver.f)
    r = solver.r
    rr = 0.5 * radius if radius is not None else 0.5 * solver.cfg.zone_radius
    edges = torch.linspace(0.0, rr, nbins + 1)
    out = []
    for b in range(nbins):
        m = (r >= edges[b]) & (r < edges[b + 1])
        n = int(m.sum().item())
        if n == 0:
            continue
        fx = -float(force[0][m].sum().item())
        tq = -float((solver.dy * force[2] - solver.dz * force[1])[m].sum().item())
        dr = float((edges[b + 1] - edges[b]).item())
        out.append(
            {
                "r": float((0.5 * (edges[b] + edges[b + 1])).item()),
                "dT_dr": fx / dr,
                "dQ_dr": tq / dr,
                "cells": n,
            }
        )
    return out


def thrust_identity_wake(
    *,
    thrust: float,
    n_rps: float,
    diameter: float,
    speed: float,
    rho: float,
    j_table: list[float],
    kt_table: list[float],
) -> dict:
    """``w_T`` and ``J_T`` from the thrust identity (ITTC open-water procedure)."""
    kt_b = thrust / (rho * n_rps**2 * diameter**4)
    jt = float(np.interp(kt_b, kt_table[::-1], j_table[::-1]))
    v_a = jt * n_rps * diameter
    w_t = 1.0 - v_a / speed
    return {
        "kt_behind_hull": kt_b,
        "j_thrust_identity": jt,
        "advance_velocity": v_a,
        "wake_fraction_identity": w_t,
    }


def thrust_deduction(
    *,
    resistance_towed: float,
    resistance_propelled: float,
    thrust: float,
) -> dict:
    """``t = (R_T^SP - R_T^tow) / T`` with the sign and budget made explicit."""
    delta = resistance_propelled - resistance_towed
    return {
        "resistance_towed": resistance_towed,
        "resistance_propelled": resistance_propelled,
        "delta_resistance": delta,
        "thrust": thrust,
        "thrust_deduction": delta / thrust if thrust else float("nan"),
    }


def reynolds_wake_scaling(
    *,
    re_lattice: float,
    re_model: float,
    exponent: float = -0.2,
) -> dict:
    """[MODEL] Reynolds lifting of a wake fraction computed at the lattice Re.

    The boundary-layer displacement thickness scales as ``delta* ~ Re^-0.2``
    (flat plate, turbulent).  With ``w ~ delta*/D`` at a fixed aft-body shape

        ``w_model / w_lattice = (Re_model / Re_lattice)^-0.2``

    so that ``w_model = w_lattice (Re_lattice / Re_model)^0.2``.  This is a
    documented similarity assumption, not a computed correction.
    """
    factor = (re_model / re_lattice) ** exponent
    return {
        "exponent": exponent,
        "re_lattice": re_lattice,
        "re_model": re_model,
        "factor": factor,
    }


def solve_self_propulsion(
    *,
    resistance: float,
    speed: float,
    diameter: float,
    rho: float,
    wake_fraction: float,
    thrust_deduction_value: float,
    j_table: list[float],
    kt_table: list[float],
    kq_table: list[float],
    eta_R: float | None = None,
    n_bounds: tuple[float, float] = (0.5, 60.0),
    tol: float = 1e-12,
    max_iter: int = 200,
) -> dict:
    """Bisection on ``n`` for ``T(n) (1 - t) = R_T`` using measured ``K_T(J)``."""
    target = resistance / (1.0 - thrust_deduction_value)
    j_arr = np.asarray(j_table, dtype=float)
    kt_arr = np.asarray(kt_table, dtype=float)
    kq_arr = np.asarray(kq_table, dtype=float)

    def state(n: float):
        j = speed * (1.0 - wake_fraction) / (n * diameter)
        kt = float(np.interp(j, j_arr, kt_arr))
        kq = float(np.interp(j, j_arr, kq_arr))
        return kt * rho * n * n * diameter**4, kq * rho * n * n * diameter**5, j, kt, kq

    lo, hi = n_bounds
    f_lo = state(lo)[0] - target
    f_hi = state(hi)[0] - target
    if f_lo * f_hi > 0:
        raise ValueError(
            f"no self-propulsion point bracketed: T(lo)-target={f_lo:.4g}, T(hi)-target={f_hi:.4g}"
        )
    history = []
    n = 0.5 * (lo + hi)
    for _ in range(max_iter):
        thrust, torque, j, kt, kq = state(n)
        residual = thrust - target
        history.append({"n": n, "thrust": thrust, "j": j, "residual": residual})
        if abs(residual) < tol * max(target, 1.0):
            break
        if (residual > 0) == (f_lo > 0):
            lo, f_lo = n, residual
        else:
            hi, f_hi = n, residual
        n = 0.5 * (lo + hi)
    thrust, torque, j, kt, kq = state(n)
    eta0 = j * kt / (2.0 * math.pi * kq) if kq > 0 else float("nan")
    chain = power_chain(
        resistance=resistance,
        speed=speed,
        thrust=thrust,
        torque=torque,
        n_rps=n,
        wake_fraction=wake_fraction,
        thrust_deduction=thrust_deduction_value,
        eta_R=eta_R,
        eta0=eta0,
    )
    return {
        "n_rps": n,
        "j": j,
        "kt": kt,
        "kq": kq,
        "eta0": eta0,
        "thrust": thrust,
        "torque": torque,
        "required_thrust": target,
        "residual": thrust - target,
        "residual_rel": (thrust - target) / target,
        "iterations": len(history),
        "history": history,
        "wake_fraction": wake_fraction,
        "thrust_deduction": thrust_deduction_value,
        "resistance": resistance,
        "speed": speed,
        "power_chain": chain,
    }


def empirical_cross_check(cb: float, wake_fraction: float) -> dict:
    """Design-formula cross-check (MODEL grade, independent of the LBM runs)."""
    w_taylor = empirical_wake_fraction(cb, form="taylor")
    t_heckscher = empirical_thrust_deduction(wake_fraction, form="heckscher")
    return {
        "wake_fraction_taylor": w_taylor,
        "thrust_deduction_heckscher": t_heckscher,
        "eta_H_empirical": (1.0 - t_heckscher) / (1.0 - w_taylor),
    }


def model_scale_resistance(model: KVLCC2Model = KVLCC2_MODEL) -> dict:
    """ITTC-78 model-scale resistance (MODEL: ``C_R`` is a prescribed input)."""
    return ittc78_resistance(
        length=model.length,
        beam=model.beam,
        draft=model.draft,
        cb=model.cb,
        speed=model.speed,
        rho=model.rho,
        nu=model.nu,
        wetted_area=model.wetted_area,
        form_factor=model.form_factor,
        cm=model.cm,
        cwp=model.cwp,
        residual_coefficient=model.residual_coefficient,
    )
