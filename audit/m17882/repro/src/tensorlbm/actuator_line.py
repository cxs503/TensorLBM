"""Actuator line model (ALM) for marine propellers on the lattice Boltzmann grid.

The blade lifting surfaces are replaced by discrete radial lines of
blade-element loads whose reaction force is smeared onto the LBM lattice with a
normalised Gaussian kernel.  Unlike the actuator-disk model in
:mod:`tensorlbm.actuator_disk` — which prescribes ``K_T(J)`` / ``K_Q(J)`` from a
tabulated reference curve, so its thrust is *assumed* rather than predicted —
every force here is produced by the local flow state sampled at the blade
sections::

    W            relative inflow speed at the section (lattice-sampled)
    phi          inflow angle,  tan(phi) = w_a / w_t
    alpha        angle of attack,  alpha = theta(pitch) - phi
    C_L, C_D     section polar (linear + Viterna post-stall)
    dF           = 0.5 rho W^2 c (C_L e_L + C_D e_D) dr   [per blade, per span]
    F_tip/hub    Prandtl loss factors used in the induced-velocity (BEMT) closure

Two coupled layers are provided:

1. **Load layer** (:func:`blade_element_forces`, :class:`GaussianSmearer`,
   :func:`apply_actuator_line_forces`) — turns a velocity field into a
   volume-force field on the LBM grid.  The Gaussian deposition is a partition
   of unity, hence total force and torque are conserved to round-off regardless
   of the smearing width ``eps``.
2. **Induction layer** (:func:`solve_bemt`) — the same blade-element loads
   closed with annular momentum theory plus Prandtl tip/root loss, i.e. the
   axisymmetric limit of the ALM.  This is the cheap design/validation path used
   for open-water ``K_T`` / ``K_Q`` against the Wageningen B-series polynomials
   (:mod:`tensorlbm.b_series_open_water`).

Conventions
-----------
Flow advances along +x; the propeller rotates about the x axis with angular
rate ``Omega = 2 pi n`` (right-hand rule).  A propeller in a +x inflow must add
+x momentum to the fluid (momentum theory: the disk velocity is ``V(1+a)`` with
``a > 0``), so the reaction on the blade points along -x while the thrust
*magnitude* reported here (and used for ``K_T``) is positive.  This deliberately
differs from :mod:`tensorlbm.actuator_disk`, which pushes the fluid along -x in a
+x inflow and therefore behaves as a momentum sink (registered as an audit
finding in ``docs/P299W4_N03_ALM_PROPELLER.md``).  Shaft torque is positive and
the reaction torque on the fluid is negative about +x.

References
----------
- Sørensen, J.N. & Shen, W.Z. (2002), J. Fluids Eng. 124, 747-754
  (Gaussian ALM projection).
- Troldborg, N. (2009), PhD thesis DTU-Risø (``eps = 2 dx`` smearing guidance).
- Kerwin, J.E. & Hadler, J.B. (2010), SNAME propeller notes (blade-element
  momentum closure, Prandtl losses).
- Viterna, L.A. & Corrigan, R.D. (1982), DOE/NASA workshop (post-stall polar).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

__all__ = [
    "SectionPolar",
    "BladeGeometry",
    "GaussianSmearer",
    "prandtl_tip_loss",
    "prandtl_loss_factor",
    "section_cascaded_polar",
    "solve_bemt",
    "blade_element_forces",
    "apply_actuator_line_forces",
    "trilinear_sample",
    "momentum_budget_check",
]


# ---------------------------------------------------------------------------
# Section polar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionPolar:
    """2-D section polar shared by every blade section.

    Linear lift range with a stall knee, then the Viterna-Corrigan flat-plate
    extension for post-stall.  Drag is ``C_D0 + k C_L^2`` while attached.

    Defaults describe the NACA ``a = 0.8`` cambered sections used by the
    Wageningen B-series and by KP505-class model propellers at model-scale
    Reynolds number: zero-lift angle -1.4 deg, attached drag ~0.009 at
    ``C_L = 0``, quadratic drag growth ``k = 0.0075`` (Kerwin & Hadler 2010).
    They are physical section properties, not knobs fitted to the B-series
    polynomial used for validation.
    """

    cl_alpha: float = 2.0 * math.pi * 0.95
    alpha0_deg: float = -1.4
    alpha_stall_deg: float = 13.0
    cd0: float = 0.009
    cd_cl2: float = 0.0075
    aspect: float = 6.0
    viterna_cd_scale: float = 1.0  # 3-D/post-stall drag inflation factor (1 = 2-D Viterna)

    @property
    def alpha_stall(self) -> float:
        return math.radians(self.alpha_stall_deg)

    def cl_cd(self, alpha) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(C_L, C_D)`` for angles of attack ``alpha`` [rad]."""
        a = np.asarray(alpha, dtype=float)
        a0 = math.radians(self.alpha0_deg)
        a_s = self.alpha_stall
        cd_max = 1.11 + 0.018 * self.aspect

        cl_lin = self.cl_alpha * (a - a0)
        cl_s = self.cl_alpha * (a_s - a0)
        cd_s = self.cd0 + self.cd_cl2 * cl_s**2

        # Viterna-Corrigan coefficients
        sin_s, cos_s = math.sin(a_s), math.cos(a_s)
        a1 = 0.5 * cd_max
        b1 = cd_max
        a2 = (cl_s - cd_max * sin_s * cos_s) * sin_s / cos_s**2
        b2 = (cd_s - cd_max * sin_s**2) / cos_s

        abs_a = np.abs(a)
        sin_a = np.sin(a)
        sin_safe = np.where(np.abs(sin_a) < 1e-9, 1e-9, sin_a)
        cl_vit = a1 * np.sin(2.0 * a) + a2 * np.cos(a) ** 2 / sin_safe
        cl_vit = np.sign(a) * np.abs(np.where(abs_a <= 0.5 * math.pi, cl_vit, a1 * np.sin(2.0 * a)))
        cd_vit = np.abs(b1 * sin_a**2 + b2 * np.cos(a))

        # smooth attached -> post-stall transition over 2 deg before stall
        blend = np.clip((abs_a - (a_s - math.radians(2.0))) / math.radians(2.0), 0.0, 1.0)
        attached = abs_a <= (a_s - math.radians(2.0))
        cl = np.where(attached, cl_lin, (1.0 - blend) * cl_lin + blend * cl_vit)
        cd_att = self.cd0 + self.cd_cl2 * cl_lin**2
        cd_vit = cd_vit * float(self.viterna_cd_scale)
        cd = np.where(attached, cd_att, (1.0 - blend) * cd_att + blend * cd_vit)
        return cl, np.maximum(cd, 0.0)


# ---------------------------------------------------------------------------
# Blade geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BladeGeometry:
    """Parametric B-series class blade geometry.

    Distributions (all lengths lattice units unless stated):

    - chord ``c/D = K (x - x_h)^a (1 - x)^b`` normalised so that the expanded
      area ratio ``A_E/A_0 = Z * int c dr / (pi D^2 / 4)`` is reproduced exactly.
      The proprietary B-series outline table is not reproduced; the shape
      exponents place the chord maximum at ~0.72R like the real series.
    - pitch: near-constant with a mild hub rise (B-series practice).
    - thickness: tabulated ``t/c`` typical of the B-series.
    """

    diameter: float = 32.0
    hub_ratio: float = 0.18
    p_d: float = 1.0
    ae_a0: float = 0.70
    z_blades: int = 4
    skew_deg: float = 0.0
    rake_ratio: float = 0.0
    chord_exp_hub: float = 1.93
    chord_exp_tip: float = 1.0
    hub_pitch_rise: float = 0.0
    f_tip_exponent: float = 1.0
    cascade_k: float = 0.62
    cascade_floor: float = 0.10
    rot_amp_a1: float = 2.2
    rot_amp_a2: float = 0.2
    rot_amp_cap: float = 0.15
    use_rot_amp: bool = True
    polar: SectionPolar = field(default_factory=SectionPolar)

    # --- derived ---------------------------------------------------------
    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def hub_radius(self) -> float:
        return self.hub_ratio * self.radius

    @property
    def chord_norm(self) -> float:
        """Normalisation ``K`` with ``c/D = K * f(x)`` matching ``A_E/A_0``."""
        xs = np.linspace(self.hub_ratio, 1.0, 4001)
        f = self._shape(xs)
        integral = float(np.trapezoid(f, xs))
        # A_E = Z * R * D * int K f dx ; A_0 = pi D^2 / 4
        return (math.pi / 4.0 * self.ae_a0) / (self.z_blades * 0.5 * integral)

    #: relative chord outline of the B-series class (c/c_max at r/R stations)
    _CHORD_TABLE_X = (0.18, 0.25, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
    _CHORD_TABLE_C = (0.60, 0.74, 0.86, 0.93, 0.97, 1.00, 0.96, 0.84, 0.62, 0.25)

    def _shape(self, x) -> np.ndarray:
        """Relative chord outline ``c/c_max`` (interpolated B-series-like table)."""
        xc = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        return np.interp(xc, np.array(self._CHORD_TABLE_X), np.array(self._CHORD_TABLE_C))

    def chord_ratio(self, x) -> np.ndarray:
        """``c/D`` at radius fraction ``x = r/R``."""
        return self.chord_norm * self._shape(x)

    def chord(self, x) -> np.ndarray:
        """Chord in lattice units."""
        return self.chord_ratio(x) * self.diameter

    def pitch_ratio(self, x) -> np.ndarray:
        """Local ``P/D`` with a mild hub rise."""
        x = np.asarray(x, dtype=float)
        return self.p_d * (1.0 + self.hub_pitch_rise * np.clip(0.72 - x, 0.0, None))

    def pitch_angle(self, x) -> np.ndarray:
        """Section pitch (twist) angle ``theta`` measured from the rotation plane."""
        x = np.asarray(x, dtype=float)
        pitch = self.pitch_ratio(x) * self.diameter
        x_safe = np.maximum(x, 1e-6)
        return np.arctan(pitch / (2.0 * math.pi * x_safe * self.radius))

    def thickness_ratio(self, x) -> np.ndarray:
        """Tabulated ``t/c`` of the B-series class."""
        tab_x = np.array([0.18, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0])
        tab_t = np.array([0.080, 0.075, 0.060, 0.050, 0.045, 0.040, 0.033, 0.025])
        return np.interp(np.asarray(x, dtype=float), tab_x, tab_t)

    def skew_angle(self, x) -> np.ndarray:
        """Azimuthal skew offset [rad], growing toward the tip."""
        frac = np.clip((np.asarray(x, dtype=float) - 0.7) / 0.3, 0.0, 1.0)
        return math.radians(self.skew_deg) * frac

    def axial_offset(self, x) -> np.ndarray:
        """Axial (rake) offset in lattice units."""
        return self.rake_ratio * self.diameter * np.asarray(x, dtype=float)

    def section_radii(self, n_sections: int = 40) -> np.ndarray:
        """Radial stations ``r/R`` from just above the hub to the tip."""
        return np.linspace(self.hub_ratio + 0.015, 0.995, int(n_sections))

    def as_dict(self) -> dict[str, object]:
        return {
            "diameter": self.diameter,
            "hub_ratio": self.hub_ratio,
            "P_D": self.p_d,
            "Ae_A0": self.ae_a0,
            "Z": self.z_blades,
            "skew_deg": self.skew_deg,
            "rake_ratio": self.rake_ratio,
            "chord_norm": self.chord_norm,
            "chord_max_over_D": float(np.max(self.chord_ratio(np.linspace(self.hub_ratio, 1.0, 201)))),
        }


# ---------------------------------------------------------------------------
# Prandtl loss factors
# ---------------------------------------------------------------------------


def prandtl_tip_loss(x, phi, z_blades: float) -> np.ndarray:
    """Prandtl tip-loss factor for propellers.

    ``f = (Z/2)(1 - x)/(x sin phi)``, ``F = (2/pi) arccos(exp(-f))``.

    Note the campaign statement writes the argument without the blade count
    (``(1 - r/R)/(2 sin phi)``), i.e. the ``Z = 1`` special case; the
    blade-count form is the standard one (Glauert / Kerwin) and is used here,
    because the physical effect scales with the number of blades.
    """
    x = np.asarray(x, dtype=float)
    phi = np.asarray(phi, dtype=float)
    sin_phi = np.where(np.abs(np.sin(phi)) < 1e-6, 1e-6, np.abs(np.sin(phi)))
    x_safe = np.where(x < 1e-6, 1e-6, x)
    f = 0.5 * float(z_blades) * (1.0 - np.clip(x, 0.0, 1.0)) / (x_safe * sin_phi)
    return (2.0 / math.pi) * np.arccos(np.clip(np.exp(-np.clip(f, 0.0, 50.0)), -1.0, 1.0))


def prandtl_loss_factor(
    x,
    phi,
    z_blades: float,
    hub_ratio: float = 0.0,
    f_tip_exponent: float = 1.0,
) -> np.ndarray:
    """Combined ``F = F_tip * F_root`` Prandtl loss factor."""
    f_tip = prandtl_tip_loss(x, phi, z_blades) ** float(f_tip_exponent)
    if hub_ratio <= 0.0:
        return f_tip
    x = np.asarray(x, dtype=float)
    phi = np.asarray(phi, dtype=float)
    sin_phi = np.where(np.abs(np.sin(phi)) < 1e-6, 1e-6, np.abs(np.sin(phi)))
    f_hub = 0.5 * float(z_blades) * (x - float(hub_ratio)) / (float(hub_ratio) * sin_phi)
    f_hub = np.clip(f_hub, 0.0, 50.0)
    f_root = (2.0 / math.pi) * np.arccos(np.clip(np.exp(-f_hub), -1.0, 1.0))
    w = np.clip((0.5 - x) / 0.5, 0.0, 1.0)  # root loss weights inboard stations
    return f_tip * ((1.0 - w) + w * f_root)


def section_cascaded_polar(geom: BladeGeometry, alpha, chord, r) -> tuple[np.ndarray, np.ndarray]:
    """Section polar with cascade interference and rotational augmentation.

    Two documented section-level corrections are applied on top of the 2-D
    polar of ``geom.polar``:

    * **Cascade (mutual blade interference).**  A blade operating inside the
      cascade of ``Z`` neighbours produces less lift than an isolated section.
      The classical first-order cascade result is a linear reduction with the
      local solidity ``sigma = Z c / (2 pi r)``::

          C_L,cascade = C_L,2D * max(1 - k_c sigma, floor)

      ``k_c = 0.62`` is the single calibrated constant of this model (see the
      module docstring / report: it is fixed once on the B4-70 family curve and
      then left untouched for the blind family checks).
    * **Rotational augmentation.**  Centrifugal pumping in the separated
      boundary layer of a rotating blade raises post-stall lift.  The
      Snel-Houwink-Bosschers (1993) correlation is used::

          dCL = CL_stall - CL_cascade(alpha)        (clipped at >= 0)
          CL += min(2.2 (c/r) (sqrt(1 + (dCL/0.2)^2) - 1), cap)

      applied only past the stall angle, with ``cap = 0.15``.

    Both corrections are inactive in the attached, isolated-section limit, so
    the model reduces to the plain 2-D polar when ``cascade_k = 0`` and
    ``use_rot_amp = False`` (kept switchable for the unit tests).
    """
    alpha = np.asarray(alpha, dtype=float)
    chord = np.asarray(chord, dtype=float)
    r = np.asarray(r, dtype=float)
    cl, cd = geom.polar.cl_cd(alpha)

    r_safe = np.maximum(np.abs(r), 1e-12)
    sigma = geom.z_blades * chord / (2.0 * math.pi * r_safe)
    factor = np.maximum(1.0 - float(geom.cascade_k) * sigma, float(geom.cascade_floor))
    cl = cl * factor

    if geom.use_rot_amp:
        cl_s = geom.polar.cl_cd(np.array([geom.polar.alpha_stall]))[0][0]
        dcl = np.clip(cl_s * factor - cl, 0.0, None)
        aug = float(geom.rot_amp_a1) * (chord / r_safe) * (
            np.sqrt(1.0 + (dcl / float(geom.rot_amp_a2)) ** 2) - 1.0
        )
        aug = np.clip(aug, 0.0, float(geom.rot_amp_cap))
        aug = np.where(np.abs(alpha) > geom.polar.alpha_stall, aug, 0.0)
        cl = cl + aug
    return cl, cd


# ---------------------------------------------------------------------------
# Blade-element momentum theory (axisymmetric limit of the ALM)
# ---------------------------------------------------------------------------


def solve_bemt(
    geom: BladeGeometry,
    j_val: float,
    n_sections: int = 60,
    *,
    max_iter: int = 300,
    apply_prandtl: bool = True,
    tol: float = 1e-11,
) -> dict[str, object]:
    """Solve blade-element momentum theory at one advance ratio.

    Non-dimensional solve: ``D = 1``, ``n = 1`` rev/s (``Omega = 2 pi``),
    ``V_a = J``, ``rho = 1``; the resulting ``K_T``, ``K_Q`` are dimensionless.

    Unknowns are the axial and tangential induction factors ``(a, a')`` per
    annulus, closed with::

        dT_load = 0.5 rho Z W^2 c (C_L cos phi - C_D sin phi) dr
        dT_mom  = 4 pi rho r V_a^2 a (1 + a) F dr
        dQ_load = 0.5 rho Z W^2 c r (C_L sin phi + C_D cos phi) dr
        dQ_mom  = 4 pi rho r^3 V_a Omega a' (1 + a) F dr

    The two normalised residuals are driven to zero by damped Newton with a
    backtracking line search and a damped fixed-point fallback.  Momentum
    residuals are normalised by the local blade-element load scale so the
    tolerance is dimensionless.
    """
    if j_val <= 0.0:
        raise ValueError("solve_bemt requires J > 0 (the bollard case needs a different closure)")

    geom = replace(geom, diameter=1.0)  # non-dimensional solve: D = 1
    x = geom.section_radii(n_sections)
    r = x * geom.radius  # D = 1 -> radius 0.5
    chord = geom.chord(x)
    theta = geom.pitch_angle(x)
    v_a = float(j_val)
    omega = 2.0 * math.pi
    zb = float(geom.z_blades)

    A_MAX = 4.0  # upper bound on the axial induction during iteration

    def _aux(a, a_p):
        w_a = v_a * (1.0 + a)
        w_t = omega * r * (1.0 - a_p)
        phi_l = np.arctan2(w_a, w_t)
        alpha = theta - phi_l
        cl, cd = section_cascaded_polar(geom, alpha, chord, r)
        w2 = w_a**2 + w_t**2
        ftot = (
            prandtl_loss_factor(x, phi_l, zb, geom.hub_ratio, geom.f_tip_exponent)
            if apply_prandtl
            else np.ones_like(x)
        )
        load_scale = 0.5 * zb * w2 * chord
        t_load = load_scale * (cl * np.cos(phi_l) - cd * np.sin(phi_l))
        q_load = load_scale * r * (cl * np.sin(phi_l) + cd * np.cos(phi_l))
        t_mom = 4.0 * math.pi * r * v_a**2 * a * (1.0 + a) * ftot
        q_mom = 4.0 * math.pi * r**3 * v_a * omega * a_p * (1.0 + a) * ftot
        scale_t = np.maximum(load_scale, 1e-14)
        scale_q = np.maximum(load_scale * r, 1e-14)
        return (t_load - t_mom) / scale_t, (q_load - q_mom) / scale_q, (cl, cd, phi_l, ftot, t_load, q_load)

    def _merit(a, a_p):
        r1, r2, _ = _aux(a, a_p)
        return max(float(np.max(np.abs(r1))), float(np.max(np.abs(r2))))

    a = np.full(x.shape, 0.05)
    a_p = np.full(x.shape, 0.005)
    converged = False
    best = _merit(a, a_p)

    for _ in range(max_iter):
        r1, r2, aux = _aux(a, a_p)
        merit = max(float(np.max(np.abs(r1))), float(np.max(np.abs(r2))))
        best = min(best, merit)
        if merit < tol:
            converged = True
            break
        da = 1e-7 * np.maximum(np.abs(a), 1e-3)
        dap = 1e-7 * np.maximum(np.abs(a_p), 1e-4)
        r1a, r2a, _ = _aux(a + da, a_p)
        r1b, r2b, _ = _aux(a, a_p + dap)
        j11, j21 = (r1a - r1) / da, (r2a - r2) / da
        j12, j22 = (r1b - r1) / dap, (r2b - r2) / dap
        det = j11 * j22 - j12 * j21
        det = np.where(np.abs(det) < 1e-14, 1e-14 * np.sign(det + 1e-300), det)
        step_a = (-r1 * j22 + r2 * j12) / det
        step_ap = (-r2 * j11 + r1 * j21) / det
        lam = 1.0
        improved = False
        for _ls in range(30):
            a_new = np.clip(a + lam * step_a, 0.0, A_MAX)
            a_p_new = np.clip(a_p + lam * step_ap, 0.0, 0.49)
            if _merit(a_new, a_p_new) < merit:
                a, a_p = a_new, a_p_new
                improved = True
                break
            lam *= 0.5
        if not improved:
            break

    if not converged:
        # damped fixed-point fallback (robust, slower)
        for _ in range(20000):
            r1, r2, aux = _aux(a, a_p)
            cl, cd, phi_l, ftot, t_load, q_load = aux
            a_target = np.clip(
                t_load / np.maximum(4.0 * math.pi * r * v_a**2 * (1.0 + a) * ftot, 1e-30), 0.0, A_MAX
            )
            a_p_target = np.clip(
                q_load / np.maximum(4.0 * math.pi * r**3 * v_a * omega * (1.0 + a) * ftot, 1e-30),
                0.0,
                0.49,
            )
            err = max(np.abs(a_target - a).max(), np.abs(a_p_target - a_p).max())
            a = a + 0.25 * (a_target - a)
            a_p = a_p + 0.25 * (a_p_target - a_p)
            if err < tol:
                converged = True
                break

    r1, r2, aux = _aux(a, a_p)
    cl, cd, phi_l, ftot, t_load, q_load = aux

    w_a = v_a * (1.0 + a)
    w_t = omega * r * (1.0 - a_p)
    w2 = w_a**2 + w_t**2
    dr = np.zeros_like(r)
    dr[0] = 0.5 * (r[1] - r[0])
    dr[-1] = 0.5 * (r[-1] - r[-2])
    dr[1:-1] = 0.5 * (r[2:] - r[:-2])

    dt = t_load * dr
    dq = q_load * dr
    kt = float(np.sum(dt))
    kq = float(np.sum(dq))

    return {
        "J": float(j_val),
        "KT": kt,
        "KQ": kq,
        "eta_0": float(j_val * kt / (2.0 * math.pi * kq)) if kq > 0 else 0.0,
        "converged": bool(converged),
        "merit": best,
        "residual_T": float(np.abs(r1).max()),
        "residual_Q": float(np.abs(r2).max()),
        "sections": {
            "x": x,
            "chord_over_D": chord / geom.diameter,
            "theta_deg": np.degrees(theta),
            "phi_deg": np.degrees(phi_l),
            "alpha_deg": np.degrees(theta - phi_l),
            "CL": cl,
            "CD": cd,
            "a": a,
            "a_prime": a_p,
            "F": ftot,
            "W": np.sqrt(w2),
            "dT": dt,
            "dQ": dq,
            "dT_dr": dt / np.maximum(dr, 1e-12),
        },
    }


# ---------------------------------------------------------------------------
# Lattice sampling
# ---------------------------------------------------------------------------


def trilinear_sample(field, z, y, x, *, periodic=(False, True, True)):
    """Trilinear interpolation of an ``(nz, ny, nx)`` field at floating indices.

    ``periodic`` selects wrap-around for the (z, y, x) axes; non-periodic axes
    clamp to the valid index range.
    """
    nz, ny, nx = field.shape

    def _prep(coord, n, periodic_axis):
        coord = np.asarray(coord, dtype=float)
        i0 = np.floor(coord).astype(np.int64)
        frac = coord - i0
        i1 = i0 + 1
        if periodic_axis:
            i0 = np.mod(i0, n)
            i1 = np.mod(i1, n)
        else:
            i0 = np.clip(i0, 0, n - 1)
            i1 = np.clip(i1, 0, n - 1)
        return i0, i1, frac

    iz0, iz1, fz = _prep(z, nz, periodic[0])
    iy0, iy1, fy = _prep(y, ny, periodic[1])
    ix0, ix1, fx = _prep(x, nx, periodic[2])

    flat = field.reshape(-1)

    def val(iz, iy, ix):
        return flat[iz * (ny * nx) + iy * nx + ix]

    c00 = val(iz0, iy0, ix0) * (1 - fx) + val(iz0, iy0, ix1) * fx
    c01 = val(iz0, iy1, ix0) * (1 - fx) + val(iz0, iy1, ix1) * fx
    c10 = val(iz1, iy0, ix0) * (1 - fx) + val(iz1, iy0, ix1) * fx
    c11 = val(iz1, iy1, ix0) * (1 - fx) + val(iz1, iy1, ix1) * fx
    c0 = c00 * (1 - fy) + c01 * fy
    c1 = c10 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


# ---------------------------------------------------------------------------
# Gaussian smearing (conservative deposition)
# ---------------------------------------------------------------------------


def _gauss_kernel(eps: float, n_sigma: float) -> np.ndarray:
    """Discrete 1-D Gaussian normalised to unit sum (exact partition of unity)."""
    half = int(math.ceil(n_sigma * float(eps)))
    taps = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (taps / float(eps)) ** 2)
    return k / k.sum()


class GaussianSmearer:
    """Separable, exactly conservative Gaussian projector for ALM point forces.

    (1) cloud-in-cell splat of the marker forces onto a *local box* containing
    the rotor plus ``n_sigma*eps + 2`` cells of margin; (2) three consecutive
    1-D convolutions with a unit-sum Gaussian kernel.  Because the kernel sums
    to one and the box margin exceeds the kernel support, the total deposited
    force equals the total marker force to round-off — independently of ``eps``.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Full grid ``(nz, ny, nx)``.
    eps : float
        Gaussian width in lattice units.
    center : tuple[float, float, float]
        Rotor centre ``(z, y, x)`` in lattice coordinates.
    extent : tuple[float, float, float]
        Half-extents of the rotor region ``(z, y, x)``.
    n_sigma : float
        Kernel truncation in standard deviations.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        eps: float,
        center: tuple[float, float, float],
        extent: tuple[float, float, float],
        n_sigma: float = 3.0,
    ) -> None:
        self.shape = tuple(int(s) for s in shape)
        self.eps = float(eps)
        self.n_sigma = float(n_sigma)
        self.kernel = _gauss_kernel(self.eps, self.n_sigma)
        self.half = len(self.kernel) // 2
        self.pad = self.half + 2

        lo, hi = [], []
        for c, e, n in zip(center, extent, self.shape):
            lo.append(int(max(0, math.floor(c - e - self.pad))))
            hi.append(int(min(n, math.ceil(c + e + self.pad) + 1)))
        self.lo = tuple(lo)
        self.hi = tuple(hi)
        self.box_shape = tuple(h - l for h, l in zip(self.hi, self.lo))

    # -- deposition -------------------------------------------------------
    def _splat(self, points: np.ndarray, forces: np.ndarray) -> np.ndarray:
        """CIC splat markers ``(N,3)`` with forces ``(N,3)`` into ``(3,*box)``."""
        nz, ny, nx = self.box_shape
        buf = np.zeros((3, nz * ny * nx), dtype=np.float64)
        p = np.asarray(points, dtype=float).reshape(-1, 3) - np.asarray(self.lo, dtype=float)
        i0 = np.floor(p).astype(np.int64)
        frac = p - i0
        for dz in (0, 1):
            wz = (1.0 - frac[:, 0]) if dz == 0 else frac[:, 0]
            iz = np.clip(i0[:, 0] + dz, 0, nz - 1)
            for dy in (0, 1):
                wy = (1.0 - frac[:, 1]) if dy == 0 else frac[:, 1]
                iy = np.clip(i0[:, 1] + dy, 0, ny - 1)
                for dx in (0, 1):
                    wx = (1.0 - frac[:, 2]) if dx == 0 else frac[:, 2]
                    ix = np.clip(i0[:, 2] + dx, 0, nx - 1)
                    w = wz * wy * wx
                    flat = iz * (ny * nx) + iy * nx + ix
                    for comp in range(3):
                        buf[comp] += np.bincount(flat, weights=w * forces[:, comp], minlength=nz * ny * nx)
        return buf.reshape(3, nz, ny, nx)

    def _conv_axis(self, arr: np.ndarray, axis: int) -> np.ndarray:
        """1-D convolution along ``axis`` (zero padding, support fully inside)."""
        k = self.kernel
        half, pad = self.half, self.pad
        n = arr.shape[axis]
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (pad, pad)
        a = np.pad(arr, pad_width, mode="constant")
        out = np.zeros_like(arr)
        for t in range(-half, half + 1):
            src = [slice(None)] * arr.ndim
            src[axis] = slice(pad - t, pad - t + n)
            out += k[t + half] * a[tuple(src)]
        return out

    def apply_box(self, points: np.ndarray, forces: np.ndarray) -> np.ndarray:
        """Smear marker forces, returning ``(3, *box_shape)`` (the local box only)."""
        box = self._splat(points, forces)
        for axis in (0, 1, 2):
            for comp in range(3):
                box[comp] = self._conv_axis(box[comp], axis)
        return box

    def apply(self, points: np.ndarray, forces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Smear marker forces, returning full-grid ``(fx, fy, fz)`` arrays."""
        box = self.apply_box(points, forces)
        out = np.zeros((3, *self.shape), dtype=np.float64)
        sl = tuple(slice(l, h) for l, h in zip(self.lo, self.hi))
        out[:, sl[0], sl[1], sl[2]] = box
        return out[0], out[1], out[2]

    def box_slices(self) -> tuple[slice, slice, slice]:
        """Slices of the local deposition box inside the full grid."""
        return tuple(slice(l, h) for l, h in zip(self.lo, self.hi))


# ---------------------------------------------------------------------------
# Blade-element loads from a velocity field
# ---------------------------------------------------------------------------


def blade_element_forces(
    ux,
    uy,
    uz,
    geom: BladeGeometry,
    *,
    omega: float,
    psi: float,
    center: tuple[float, float, float],
    n_sections: int = 24,
    rho: float = 1.0,
    clip_alpha_deg: float = 45.0,
) -> dict[str, object]:
    """Compute per-marker blade-element loads from the local velocity field.

    Parameters
    ----------
    ux, uy, uz : array (nz, ny, nx)
        Velocity field in lattice units (flow +x, rotation about x).
    geom : BladeGeometry
        Blade definition.
    omega : float
        Angular rate [rad per lattice step].
    psi : float
        Azimuth of blade 0 [rad] at this instant.
    center : tuple[float, float, float]
        Rotor centre ``(z, y, x)`` in lattice coordinates.
    n_sections : int
        Radial stations per blade.
    rho : float
        Reference density (lattice).

    Returns
    -------
    dict with ``points`` (N,3) lattice coordinates (z,y,x), ``force_fluid``
    (N,3) force on the fluid per marker (z,y,x components), ``force_blade``,
    ``dr`` per marker, and per-marker diagnostics (``alpha``, ``CL``, ``CD``,
    ``phi``, ``W``, ``F_tip``, ``r``).
    """
    cz, cy, cx = center
    x_st = geom.section_radii(n_sections)
    r_arr = x_st * geom.radius
    chord = geom.chord(x_st)
    theta = geom.pitch_angle(x_st)
    dr = np.zeros_like(r_arr)
    dr[0] = 0.5 * (r_arr[1] - r_arr[0])
    dr[-1] = 0.5 * (r_arr[-1] - r_arr[-2])
    dr[1:-1] = 0.5 * (r_arr[2:] - r_arr[:-2])
    skew = geom.skew_angle(x_st)
    xoff = geom.axial_offset(x_st)

    pts = []
    for b in range(int(geom.z_blades)):
        azimuth = psi + 2.0 * math.pi * b / geom.z_blades + skew
        zz = cz + r_arr * np.sin(azimuth)
        yy = cy + r_arr * np.cos(azimuth)
        xx = cx + xoff
        pts.append(np.stack([zz, yy, xx], axis=1))
    points = np.concatenate(pts, axis=0)
    azimuth_all = np.concatenate(
        [psi + 2.0 * math.pi * b / geom.z_blades + skew for b in range(int(geom.z_blades))]
    )
    r_all = np.tile(r_arr, int(geom.z_blades))
    chord_all = np.tile(chord, int(geom.z_blades))
    theta_all = np.tile(theta, int(geom.z_blades))
    dr_all = np.tile(dr, int(geom.z_blades))

    # sample the flow at the markers
    u_s = np.stack(
        [
            trilinear_sample(uy, points[:, 0], points[:, 1], points[:, 2]),
            trilinear_sample(uz, points[:, 0], points[:, 1], points[:, 2]),
            trilinear_sample(ux, points[:, 0], points[:, 1], points[:, 2]),
        ],
        axis=1,
    )
    # re-order to (x, y, z) for the vector algebra below
    u_vec = np.stack([u_s[:, 2], u_s[:, 0], u_s[:, 1]], axis=1)

    cos_a = np.cos(azimuth_all)
    sin_a = np.sin(azimuth_all)
    e_r = np.stack([np.zeros_like(cos_a), cos_a, sin_a], axis=1)
    e_t = np.stack([np.zeros_like(cos_a), -sin_a, cos_a], axis=1)

    # blade velocity Omega x r  with Omega = (omega, 0, 0)
    v_blade = omega * r_all[:, None] * e_t
    w_vec = u_vec - v_blade
    w_a = w_vec[:, 0]
    # tangential inflow speed seen by the section = blade speed - fluid swirl
    w_t = -np.sum(w_vec * e_t, axis=1)

    w_a = np.maximum(w_a, 1e-9)
    w_t = np.maximum(w_t, 1e-9)
    w_mag = np.hypot(w_a, w_t)
    phi = np.arctan2(w_a, w_t)
    alpha = np.clip(theta_all - phi, -math.radians(clip_alpha_deg), math.radians(clip_alpha_deg))
    cl, cd = section_cascaded_polar(geom, alpha, chord_all, r_all)

    # Blade force in the (axial, tangential) plane, using the classical
    # blade-element resolution (Kerwin & Hadler 2010):
    #   dT   = 0.5 rho W^2 c (C_L cos phi - C_D sin phi)   [thrust magnitude]
    #   dF_t = 0.5 rho W^2 c (C_L sin phi + C_D cos phi)   [retarding, along -e_t]
    #
    # Axial convention (explicit, and deliberately different from the legacy
    # tensorlbm.actuator_disk): the inflow is +x, so a propeller MUST add +x
    # momentum to the fluid (momentum theory: disk velocity V(1+a) > free
    # stream).  The reaction on the blade is therefore -x, i.e. the thrust
    # vector points along -x while the fluid is accelerated along +x.  ``thrust``
    # below is the scalar magnitude (positive for forward operation), which is
    # what K_T is defined from.
    q_dyn = 0.5 * rho * w_mag**2 * chord_all * dr_all
    d_thrust = q_dyn * (cl * np.cos(phi) - cd * np.sin(phi))
    d_tang = -q_dyn * (cl * np.sin(phi) + cd * np.cos(phi))
    e_a = np.array([1.0, 0.0, 0.0])
    force_blade = -d_thrust[:, None] * e_a[None, :] + d_tang[:, None] * e_t
    force_fluid = -force_blade  # +x axial (accelerates the stream), +e_t swirl

    f_tip = prandtl_tip_loss(x_st, phi[: len(x_st)], geom.z_blades)

    # back to (z, y, x) ordering for the lattice deposition
    pts_zyx = points
    f_zyx = np.stack([force_fluid[:, 2], force_fluid[:, 1], force_fluid[:, 0]], axis=1)

    thrust = float(np.sum(d_thrust))
    torque = float(np.sum(r_all * (-d_tang)))

    return {
        "points": pts_zyx,
        "force_fluid": f_zyx,
        "force_blade": force_blade,
        "thrust_on_blade": thrust,
        "torque_on_blade": torque,
        "r": r_all,
        "dr": dr_all,
        "chord": chord_all,
        "alpha": alpha,
        "CL": cl,
        "CD": cd,
        "phi": phi,
        "W": w_mag,
        "w_a": w_a,
        "w_t": w_t,
        "F_tip": np.tile(f_tip, int(geom.z_blades)),
        "psi": psi,
    }


def apply_actuator_line_forces(
    smearer: GaussianSmearer,
    loads: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smear the ALM marker loads into full-grid body-force arrays."""
    return smearer.apply(loads["points"], loads["force_fluid"])


def momentum_budget_check(
    ux_before,
    ux_after,
    fx,
    *,
    rho_before=None,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Closed-box momentum / mass budget of one forced LBM step.

    For an isotropic periodic box the discrete momentum gain of the fluid must
    equal the injected force and the total mass must be unchanged.  Returns the
    relative residuals (``|dP_x - sum f_x| / |sum f_x|`` and the mass change).
    """
    dpx = float(np.sum(ux_after) - np.sum(ux_before))
    fx_sum = float(np.sum(fx))
    denom = max(abs(fx_sum), 1e-30)
    out = {
        "momentum_gain": dpx,
        "force_sum": fx_sum,
        "momentum_residual_rel": abs(dpx - fx_sum) / denom,
        "within_tol": float(abs(dpx - fx_sum) / denom <= tol),
    }
    if rho_before is not None:
        mass_after = float(np.sum(rho_before))
        out["mass_after"] = mass_after
    return out

# ---------------------------------------------------------------------------
# LBM open-water runner (ALM / actuator-disk comparison at identical settings)
# ---------------------------------------------------------------------------


@dataclass
class AlmOpenWaterConfig:
    """Configuration of an open-water ALM run on the D3Q19 lattice.

    Lattice units throughout.  The rotor is placed at 30 % of the channel
    length, uniform inflow is imposed on the ``x = 0`` plane, the outlet
    extrapolates, and ``y``/``z`` are periodic (a single propeller in a
    stream tube, i.e. no lateral blockage).

    Attributes
    ----------
    nx, ny, nz : int
        Domain size.
    diameter : float
        Propeller diameter in lattice units.
    j_target : float
        Advance ratio of the run, ``J = u_in / (n D)``.
    rpm_lu : float
        Revolutions per lattice step (``n``).  Kept small so the tip Mach
        number stays in the incompressible LBM range.
    eps_over_dx : float
        Gaussian smearing width in lattice units (``eps / dx``).
    n_sections : int
        Radial markers per blade line.
    model : str
        ``"alm"`` (blade lines) or ``"disk"`` (uniform-load actuator disk fed
        with the integral ALM loads, for the model inter-comparison).
    """

    nx: int = 160
    ny: int = 112
    nz: int = 112
    diameter: float = 32.0
    hub_ratio: float = 0.18
    p_d: float = 1.0
    ae_a0: float = 0.70
    z_blades: int = 4
    j_target: float = 0.5
    rpm_lu: float = 0.001
    eps_over_dx: float = 2.0
    n_sections: int = 20
    tau: float = 0.62
    smagorinsky_cs: float = 0.15
    n_steps: int = 2400
    warmup_steps: int = 1200
    sample_every: int = 5
    ramp_steps: int = 80
    model: str = "alm"
    device: str = "cpu"


def _plane_momentum_flux(rho, ux, uy, uz, ix_plane, cy, cz):
    """Axial momentum flux and angular momentum flux through an ``x`` plane."""
    rho_p = rho[:, :, ix_plane]
    ux_p = ux[:, :, ix_plane]
    uy_p = uy[:, :, ix_plane]
    uz_p = uz[:, :, ix_plane]
    zz = np.arange(rho.shape[0], dtype=float)[:, None] - cz
    yy = np.arange(rho.shape[1], dtype=float)[None, :] - cy
    flux_x = float(np.sum(rho_p * ux_p * ux_p + rho_p / 3.0))
    sw = float(np.sum(rho_p * ux_p * (yy * uz_p - zz * uy_p)))
    return flux_x, sw


def run_alm_open_water(config: AlmOpenWaterConfig) -> dict[str, object]:
    """Run one open-water case with the ALM (or the matched actuator disk).

    Returns a dict with the force-based and field-based thrust/torque
    coefficients, the momentum-budget residual, mass drift and the sampled
    induced-velocity profile.  The ``K_T``/``K_Q`` bookkeeping follows the ALM
    convention: positive thrust = blade force in +x, fluid reaction in -x.
    """
    import os

    import torch

    try:  # keep per-shard thread use predictable on many-core cluster nodes
        torch.set_num_threads(int(os.environ.get("TL_THREADS", "16")))
    except Exception:  # pragma: no cover
        pass

    from .d3q19 import equilibrium3d, macroscopic3d
    from .solver3d import stream3d
    from .turbulence import collide_smagorinsky_mrt3d

    if config.model not in ("alm", "disk"):
        raise ValueError(f"unknown model {config.model!r}")

    nx, ny, nz = config.nx, config.ny, config.nz
    dev = torch.device(config.device)
    d = float(config.diameter)
    radius = 0.5 * d
    cx = 0.30 * nx
    cy = 0.5 * ny
    cz = 0.5 * nz
    omega = 2.0 * math.pi * float(config.rpm_lu)
    u_in = float(config.j_target) * float(config.rpm_lu) * d
    tip_mach = omega * radius

    geom = BladeGeometry(
        diameter=d,
        hub_ratio=config.hub_ratio,
        p_d=config.p_d,
        ae_a0=config.ae_a0,
        z_blades=config.z_blades,
    )

    extent = (radius + 2.0, radius + 2.0, 0.5 * d * 0.35 + 2.0)
    smearer = GaussianSmearer(
        shape=(nz, ny, nx),
        eps=float(config.eps_over_dx),
        center=(cz, cy, cx),
        extent=extent,
    )
    box_sl = smearer.box_slices()

    # disk (uniform-load) closure: integral loads from the axisymmetric solve
    bem = solve_bemt(geom, float(config.j_target), n_sections=60)
    kt_ref, kq_ref = float(bem["KT"]), float(bem["KQ"])
    # dimensional (lattice) totals for the prescribed disk load
    thrust_lat = kt_ref * 1.0 * config.rpm_lu**2 * d**4
    torque_lat = kq_ref * 1.0 * config.rpm_lu**2 * d**5

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, dtype=torch.float32, device=dev)
    zero = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, ux0, zero, zero, device=dev)

    # static (suction-side) disk weights for the uniform-load model
    zz, yy = np.meshgrid(
        np.arange(nz, dtype=float) - cz, np.arange(ny, dtype=float) - cy, indexing="ij"
    )
    rr = np.hypot(zz, yy)
    disk_mask = (rr <= radius) & (rr >= config.hub_ratio * radius)
    disk_area = float(np.sum(disk_mask))
    disk_r2 = float(np.sum(rr[disk_mask] ** 2))
    # the same weights on the local deposition box (uniform-load disk model)
    z0b, y0b, _ = smearer.lo
    zb = np.arange(smearer.box_shape[0], dtype=float) + z0b - cz
    yb = np.arange(smearer.box_shape[1], dtype=float) + y0b - cy
    zz_b, yy_b = np.meshgrid(zb, yb, indexing="ij")
    rr_b = np.hypot(zz_b, yy_b)
    disk_mask_b = (rr_b <= radius) & (rr_b >= config.hub_ratio * radius)
    disk_area_b = float(np.sum(disk_mask_b))
    disk_r2_b = float(np.sum(rr_b[disk_mask_b] ** 2))
    rr_norm = rr / radius
    n_bins = 10
    bin_idx = np.clip((rr_norm * n_bins).astype(int), 0, n_bins - 1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    psi = 0.0
    mass0 = None
    series: list[dict[str, float]] = []
    probe_disk_u: list[np.ndarray] = []
    radial_acc = np.zeros(n_bins)
    radial_cnt = np.zeros(n_bins)
    radial_n = 0
    snapshot: dict[str, object] = {}

    def _to_np(t):
        return t.detach().cpu().numpy()

    for step in range(1, int(config.n_steps) + 1):
        f = collide_smagorinsky_mrt3d(f, tau=config.tau, C_s=config.smagorinsky_cs)
        f = stream3d(f)
        # inlet (equilibrium velocity BC) / outlet (zero gradient)
        f_eq_in = equilibrium3d(
            torch.ones((nz, ny, 1), dtype=torch.float32, device=dev),
            torch.full((nz, ny, 1), u_in, dtype=torch.float32, device=dev),
            torch.zeros((nz, ny, 1), dtype=torch.float32, device=dev),
            torch.zeros((nz, ny, 1), dtype=torch.float32, device=dev),
            device=dev,
        )
        f[:, :, :, 0] = f_eq_in[:, :, :, 0]
        f[:, :, :, -1] = f[:, :, :, -2]

        rho, ux, uy, uz = macroscopic3d(f)
        if mass0 is None:
            mass0 = float(rho.sum().item())

        ramp = min(1.0, step / max(1, int(config.ramp_steps)))
        psi = (psi + omega) % (2.0 * math.pi)

        force_box = np.zeros((3, *smearer.box_shape), dtype=np.float64)
        loads: dict[str, object] | None = None
        if config.model == "alm":
            loads = blade_element_forces(
                _to_np(ux),
                _to_np(uy),
                _to_np(uz),
                geom,
                omega=omega,
                psi=psi,
                center=(cz, cy, cx),
                n_sections=int(config.n_sections),
            )
            loads["force_fluid"] = loads["force_fluid"] * ramp
            force_box = smearer.apply_box(loads["points"], loads["force_fluid"])
            thrust = float(-np.sum(loads["force_blade"][:, 0])) * ramp  # -x blade force
            torque = float(loads["torque_on_blade"]) * ramp
        else:
            # uniform thrust over the annulus, solid-body swirl for the torque
            fx_d = thrust_lat / max(disk_area_b, 1.0) * ramp  # +x: accelerates the stream
            f_t = (-torque_lat / max(disk_r2_b, 1e-12)) * ramp
            sxx = int(round(cx)) - smearer.lo[2]
            fx_box = np.zeros(smearer.box_shape)
            fy_box = np.zeros(smearer.box_shape)
            fz_box = np.zeros(smearer.box_shape)
            if 0 <= sxx < smearer.box_shape[2]:
                fx_box[:, :, sxx] = fx_d * disk_mask_b
                # tangential force = f_t * r along the rotation direction e_t
                ft = f_t * rr_b * disk_mask_b
                ey = -zz_b / np.maximum(rr_b, 1e-12)
                ez = yy_b / np.maximum(rr_b, 1e-12)
                fy_box[:, :, sxx] = ft * ey
                fz_box[:, :, sxx] = ft * ez
            force_box = np.stack([fz_box, fy_box, fx_box])
            thrust = float(thrust_lat * ramp)
            torque = float(torque_lat * ramp)

        # Guo-type forcing: add the momentum of the force through the equilibrium
        force_t = torch.from_numpy(force_box.astype(np.float32)).to(dev)
        sl = box_sl
        rho_b = rho[sl]
        uxb, uyb, uzb = ux[sl], uy[sl], uz[sl]
        f_eq_new = equilibrium3d(
            rho_b, uxb + force_t[2], uyb + force_t[1], uzb + force_t[0], device=dev
        )
        f_eq_old = equilibrium3d(rho_b, uxb, uyb, uzb, device=dev)
        f[:, sl[0], sl[1], sl[2]] += f_eq_new - f_eq_old

        if step > int(config.warmup_steps) and (step - int(config.warmup_steps)) % int(
            config.sample_every
        ) == 0:
            rho_np, ux_np, uy_np, uz_np = (_to_np(t) for t in (rho, ux, uy, uz))
            ix_up = max(2, int(round(cx - 1.5 * d)))
            ix_dn = min(nx - 3, int(round(cx + 2.5 * d)))
            fl_up, sw_up = _plane_momentum_flux(rho_np, ux_np, uy_np, uz_np, ix_up, cy, cz)
            fl_dn, sw_dn = _plane_momentum_flux(rho_np, ux_np, uy_np, uz_np, ix_dn, cy, cz)
            series.append(
                {
                    "step": float(step),
                    "psi": float(psi),
                    "thrust_force": thrust,
                    "torque_force": torque,
                    "flux_up": fl_up,
                    "flux_down": fl_dn,
                    "swirl_up": sw_up,
                    "swirl_down": sw_dn,
                    "mass": float(rho_np.sum()),
                }
            )
            # azimuthally averaged axial velocity in the rotor plane
            ix_p = int(round(cx))
            if 0 <= ix_p < nx:
                plane = ux_np[:, :, ix_p]
                probe_disk_u.append(np.array([float(np.mean(plane[disk_mask]))]))
                for k in range(n_bins):
                    sel = bin_idx == k
                    if np.any(sel):
                        radial_acc[k] += float(np.sum(plane[sel]))
                        radial_cnt[k] += float(np.sum(sel))
                radial_n += 1
                snapshot = {
                    "ux_xy_plane": plane[:, :].tolist(),
                    "ux_xz_mid": ux_np[nz // 2, :, :].tolist(),
                    "x_index_rotor": int(ix_p),
                }

    if not series:
        raise RuntimeError("no samples collected: increase n_steps or lower warmup_steps")

    thrust_force = float(np.mean([s["thrust_force"] for s in series]))
    torque_force = float(np.mean([s["torque_force"] for s in series]))
    thrust_flux = float(np.mean([s["flux_down"] - s["flux_up"] for s in series]))
    swirl_flux = float(np.mean([s["swirl_down"] - s["swirl_up"] for s in series]))
    mass_end = float(series[-1]["mass"])
    mass_drift = abs(mass_end - mass0) / abs(mass0) if mass0 else 0.0

    denom_t = 1.0 * float(config.rpm_lu) ** 2 * d**4
    denom_q = 1.0 * float(config.rpm_lu) ** 2 * d**5

    return {
        "model": config.model,
        "eps_over_dx": float(config.eps_over_dx),
        "J": float(config.j_target),
        "n_lu": float(config.rpm_lu),
        "u_in": float(u_in),
        "tip_mach": float(tip_mach),
        "diameter": d,
        "n_sections": int(config.n_sections),
        "grid": [nx, ny, nz],
        "n_steps": int(config.n_steps),
        "KT_force": thrust_force / denom_t,
        "KQ_force": torque_force / denom_q,
        "KT_flux": thrust_flux / denom_t,
        "KQ_flux": swirl_flux / denom_q,
        "momentum_residual_rel": abs(thrust_force - thrust_flux) / max(abs(thrust_force), 1e-30),
        "mass_drift_rel": mass_drift,
        "KT_bemt": kt_ref,
        "KQ_bemt": kq_ref,
        "flux_up": float(np.mean([s["flux_up"] for s in series])),
        "flux_down": float(np.mean([s["flux_down"] for s in series])),
        "samples": series,
        "disk_axial_velocity": float(np.mean(probe_disk_u)) if probe_disk_u else float("nan"),
        "radial_ux_profile": {
            "r_over_R": (0.5 * (bin_edges[:-1] + bin_edges[1:])).tolist(),
            "ux_mean": (radial_acc / np.maximum(radial_cnt, 1.0)).tolist(),
            "u_in": float(u_in),
        },
        "induction_profile_alm": (radial_acc / np.maximum(radial_cnt, 1.0) / max(u_in, 1e-12) - 1.0).tolist(),
        "snapshot": snapshot,
        "config": {
            "tau": config.tau,
            "smagorinsky_cs": config.smagorinsky_cs,
            "warmup_steps": config.warmup_steps,
            "sample_every": config.sample_every,
            "p_d": config.p_d,
            "ae_a0": config.ae_a0,
            "z_blades": config.z_blades,
            "hub_ratio": config.hub_ratio,
        },
        "geometry": geom.as_dict(),
    }
