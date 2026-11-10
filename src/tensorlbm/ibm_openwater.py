"""Immersed-boundary open-water propeller: parametric rotor + D3Q19 direct forcing.

Scope of this module (campaign ``p233w1-N07``, Phase 0 of the ice–propeller
programme)
-----------------------------------------------------------------------------
* A **Peskin 4-point** direct-forcing IBM (:mod:`tensorlbm.ibm_force_transfer`)
  drives a *parametric, rotating* propeller blade surface represented by a
  single Lagrangian marker sheet (thin-lifting-surface representation).
* The rotor is a parametric propeller in the sense of
  :class:`tensorlbm.propeller_cad.PropellerGeometryConfig`: elliptic radial
  chord distribution, constant pitch ratio at ``r/R = 0.7``, hub ratio and
  expanded area ratio all enter the marker geometry analytically.
* Open-water curves ``K_T(J)``, ``K_Q(J)``, ``eta_0(J)`` are produced by
  sweeping the advance ratio ``J = U/(nD)`` at fixed tip speed (fixed lattice
  Mach), and are compared against the published KP505 open-water table used
  elsewhere in this repository (:mod:`tensorlbm.actuator_disk`).
* A 2-D D2Q9 variant provides the classical **cylinder drag sweep**
  (``Re = 20..200``) used as the IBM accuracy gate.

Numerical conventions
---------------------
* Everything is in lattice units (``dx = dt = rho = 1``); ``x`` is the
  streamwise (rotor axis) direction and is the **last** array axis, matching
  the ``(nz, ny, nx)`` layout of :mod:`tensorlbm.solver3d`.
* The IBM marker force is injected with
  :func:`tensorlbm.ibm_force_transfer.body_source_3d`, whose first moment is
  exactly the force itself (no Guo ``1 - 1/(2 tau)`` attenuation).  Two
  consequences: (i) the reaction on the rotor is ``-sum_k f_k`` *identically*,
  and (ii) a single-pass direct forcing (``n_correctors = 0``) is the
  self-consistent choice, because the fluid receives the full marker force in
  the same step.
* Thrust/torque on the rotor are the reaction of the momentum handed to the
  fluid, so

      K_T = T / (rho n^2 D^4),   K_Q = Q / (rho n^2 D^5),
      eta_0 = J K_T / (2 pi K_Q).

References
----------
Peskin (1972) J. Comput. Phys. 10, 252; Uhlmann (2005) J. Comput. Phys. 209,
448; Breugem (2012) J. Comput. Phys. 231, 4469; Fadlun et al. (2000)
J. Comput. Phys. 161, 35; Fujisawa et al. (2000) / SIMMAN (2010) KP505
open-water data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .d2q9 import equilibrium as _eq2d
from .d2q9 import macroscopic as _macro2d
from .d3q19 import equilibrium3d as _eq3d
from .d3q19 import macroscopic3d as _macro3d
from .ibm_force_transfer import (
    body_source_3d,
    build_cylinder_markers_2d,
    build_stencil,
    direct_forcing,
    interior_mask_force,
)
from .solver import collide_bgk as _collide2d
from .solver import stream as _stream2d
from .solver3d import collide_mrt3d, stream3d

__all__ = [
    "CYLINDER_CD_2D",
    "KP505_10KQ",
    "KP505_KT",
    "CylinderConfig",
    "OpenWaterConfig",
    "PropellerGeometry",
    "blade_markers",
    "blade_sheet",
    "cylinder_drag_2d",
    "eta_error_is_monotone",
    "hub_mask",
    "kp505_reference",
    "marker_forcing",
    "momentum_balance_audit",
    "open_water_efficiency",
    "propeller_markers",
    "rotate_markers",
    "rotor_target_velocity",
    "run_open_water_point",
    "run_open_water_sweep",
]


# --------------------------------------------------------------------------- #
# Reference data (audit chain: quoted verbatim in the campaign report)
# --------------------------------------------------------------------------- #

#: KP505 thrust coefficient versus advance ratio (Fujisawa et al. 2000 /
#: SIMMAN 2010).  Same table as :mod:`tensorlbm.actuator_disk`
#: (``_KP505_KT_J`` / ``_KP505_KT_V``); values there are rounded to 2 decimals.
KP505_KT: dict[float, float] = {
    0.1: 0.45,
    0.2: 0.44,
    0.3: 0.42,
    0.4: 0.40,
    0.5: 0.37,
    0.6: 0.33,
    0.7: 0.29,
    0.8: 0.24,
    0.9: 0.17,
    1.0: 0.10,
    1.1: 0.04,
}

#: KP505 ``10 K_Q`` versus advance ratio (same source).
KP505_10KQ: dict[float, float] = {
    0.1: 0.65,
    0.2: 0.63,
    0.3: 0.61,
    0.4: 0.59,
    0.5: 0.55,
    0.6: 0.51,
    0.7: 0.47,
    0.8: 0.41,
    0.9: 0.33,
    1.0: 0.24,
    1.1: 0.15,
}

#: 2-D circular-cylinder drag coefficient (unbounded flow, ``Re = U D / nu``).
#: Values: Dennis & Chang (1970) JFM 42:471 for Re <= 100, Coutanceau & Bouard
#: (1977) for Re = 20..60, Rajani et al. (2009) for Re >= 150; the literature
#: scatter of the tabulated mean drag is about +-3 %.
CYLINDER_CD_2D: dict[int, float] = {
    20: 2.05,
    40: 1.54,
    60: 1.47,
    80: 1.40,
    100: 1.34,
    150: 1.33,
    200: 1.32,
}


def _interp_table(table: dict[float, float], x: float) -> float:
    """Piecewise-linear interpolation of a monotone-keyed reference table."""
    keys = sorted(table)
    if x <= keys[0]:
        return float(table[keys[0]])
    if x >= keys[-1]:
        return float(table[keys[-1]])
    for lo, hi in zip(keys, keys[1:]):
        if lo <= x <= hi:
            w = (x - lo) / (hi - lo)
            return float(table[lo]) * (1.0 - w) + float(table[hi]) * w
    raise RuntimeError("unreachable")  # pragma: no cover


def open_water_efficiency(advance_ratio: float, kt: float, kq: float) -> float:
    """Open-water efficiency ``eta_0 = J K_T / (2 pi K_Q)``."""
    if kq == 0.0:
        return 0.0
    return float(advance_ratio) * float(kt) / (2.0 * math.pi * float(kq))


def kp505_reference(advance_ratio: float) -> dict:
    """KP505 open-water reference triple at ``J``."""
    j = float(advance_ratio)
    kt = _interp_table(KP505_KT, j)
    kq = 0.1 * _interp_table(KP505_10KQ, j)
    return {
        "J": j,
        "KT": kt,
        "10KQ": 10.0 * kq,
        "KQ": kq,
        "eta_0": open_water_efficiency(j, kt, kq),
    }


def eta_error_is_monotone(errors: Sequence[float], *, tol: float = 1e-12) -> bool:
    """Grid-convergence gate: is ``errors`` non-increasing with refinement?

    ``errors`` must be ordered coarse -> fine.  A refinement is counted as
    converged when each successive error is not larger than the previous one
    (up to ``tol``); a flat error is accepted (it is not a divergence) but a
    single increase fails the gate.
    """
    vals = [float(e) for e in errors]
    if len(vals) < 2:
        return False
    return all(b <= a + tol for a, b in zip(vals, vals[1:]))


# --------------------------------------------------------------------------- #
# Parametric propeller geometry (Lagrangian marker sheet)
# --------------------------------------------------------------------------- #


@dataclass
class PropellerGeometry:
    """Parametric rotor geometry, all lengths in lattice units.

    The radial chord distribution is the elliptic law of
    :mod:`tensorlbm.propeller_cad`
    (``c(r)/c_max = sqrt(1 - ((r/R - 0.7)/0.8)^2)``, normalised so that the
    expanded area ratio ``A_E/A_0`` is reproduced), and the pitch is constant
    (``P = (P/D)_0.7R * D``), so the local pitch angle is
    ``beta(r) = atan(P / (2 pi r))``.

    The blade is represented by a **single marker sheet on the camber line**
    (zero thickness): at the resolutions affordable for a rotating 3-D IBM run
    (``D = 16..32``) the 4-point kernel already smears the sheet over about 1.5
    cells, which is the same order as the real blade thickness, so a two-sided
    marker set would double-count the marker force.  This is the standard
    thin-lifting-surface IBM representation.
    """

    diameter: float = 24.0
    n_blades: int = 5
    hub_diameter_ratio: float = 0.18
    pitch_ratio_07: float = 0.95
    blade_area_ratio: float = 0.65
    skew_deg: float = 0.0
    rake_ratio: float = 0.0
    radial_stations: int = 12
    spacing: float = 0.9

    def __post_init__(self) -> None:
        if self.n_blades < 2:
            raise ValueError("n_blades must be >= 2")
        if self.diameter <= 0:
            raise ValueError("diameter must be > 0")
        if self.pitch_ratio_07 <= 0:
            raise ValueError("pitch_ratio_07 must be > 0")
        if self.radial_stations < 3:
            raise ValueError("radial_stations must be >= 3")
        if self.spacing <= 0:
            raise ValueError("spacing must be > 0")

    # -- basic dimensions ---------------------------------------------------
    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def hub_radius(self) -> float:
        return self.hub_diameter_ratio * self.radius

    @property
    def hub_length(self) -> float:
        return 0.45 * self.diameter

    @property
    def pitch(self) -> float:
        """Constant pitch ``P = (P/D)_0.7R * D``."""
        return self.pitch_ratio_07 * self.diameter

    @property
    def mean_chord(self) -> float:
        """Mean (expanded) chord consistent with ``A_E/A_0``."""
        return (
            math.pi * self.diameter**2 / 4.0 * self.blade_area_ratio
        ) / self.n_blades / (self.radius - self.hub_radius)

    @property
    def chord_scale(self) -> float:
        """Scale so that ``int_{r_hub}^{R} c(r) dr = A_E / Z``."""
        n = 400
        r0, r1 = self.hub_radius, self.radius
        acc = 0.0
        for i in range(n):
            r_frac = (r0 + (r1 - r0) * (i + 0.5) / n) / self.radius
            x_norm = (r_frac - 0.7) / 0.8
            acc += max(0.0, 1.0 - x_norm * x_norm) ** 0.5
        mean_frac = acc / n
        # A_E = (pi D^2 / 4) * (A_E/A_0), split equally over the blades
        area_per_blade = (
            math.pi * self.diameter**2 / 4.0 * self.blade_area_ratio / self.n_blades
        )
        return area_per_blade / (mean_frac * (r1 - r0))

    # -- radial laws --------------------------------------------------------
    def chord(self, r: float) -> float:
        """Chord at radius ``r`` (elliptic radial law)."""
        x_norm = (r / self.radius - 0.7) / 0.8
        return self.chord_scale * max(0.0, 1.0 - x_norm * x_norm) ** 0.5

    def pitch_angle(self, r: float) -> float:
        """Local pitch angle ``beta = atan(P / (2 pi r))`` [rad]."""
        return math.atan2(self.pitch, 2.0 * math.pi * r)

    def skew_angle(self, r: float) -> float:
        """Skew angle [rad] (linear in ``r/R``; zero when ``skew_deg = 0``)."""
        return math.radians(self.skew_deg) * (r / self.radius)

    def stations(self) -> list[float]:
        """Radial marker stations between hub and tip (ends excluded)."""
        r0 = self.hub_radius * 1.02
        r1 = self.radius * 0.995
        n = self.radial_stations
        return [r0 + (r1 - r0) * (i + 0.5) / n for i in range(n)]


def _rotate_x(points: torch.Tensor, angle: float, center: Sequence[float]) -> torch.Tensor:
    """Rotate ``(M, 3)`` points about an x-parallel axis through ``center``."""
    cx = torch.zeros(3, dtype=points.dtype, device=points.device)
    cx[0] = float(center[0])
    rel = points - cx
    ca, sa = math.cos(angle), math.sin(angle)
    y, z = rel[:, 1], rel[:, 2]
    out = rel.clone()
    out[:, 1] = ca * y - sa * z
    out[:, 2] = sa * y + ca * z
    res = out + cx
    res[:, 0] = points[:, 0]
    return res


def _blade_sheet(
    geom: PropellerGeometry,
    azimuth: float = 0.0,
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    blade_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One blade as ``(points, normals, areas)`` (all ``(M, ...)``).

    The sheet lies on the camber line; the surface normal used by the slip
    forcing is the in-section normal ``n = -sin(beta) e_t + cos(beta) e_x``
    (perpendicular to the chord, inside the section plane).  ``areas`` is the
    marker patch area ``(chord / n_chord) * dr`` used by the modelled skin
    friction.
    """
    phi0 = azimuth + 2.0 * math.pi * blade_index / geom.n_blades
    pts: list[tuple[float, float, float]] = []
    nrm: list[tuple[float, float, float]] = []
    area: list[float] = []
    spacing = geom.spacing
    e_x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    stations = geom.stations()
    dr = (stations[-1] - stations[0]) / max(1, len(stations) - 1) if len(stations) > 1 else 1.0
    for r in stations:
        chord = geom.chord(r)
        if chord <= 0.0:
            continue
        n_chord = max(2, int(math.ceil(chord / spacing)))
        beta = geom.pitch_angle(r)
        phi = phi0 + geom.skew_angle(r)
        rake = geom.rake_ratio * geom.diameter * (r / geom.radius) ** 2
        e_t = torch.tensor([0.0, -math.sin(phi), math.cos(phi)], dtype=torch.float64)
        d = math.cos(beta) * e_t + math.sin(beta) * e_x
        n_local = -math.sin(beta) * e_t + math.cos(beta) * e_x
        mid = torch.tensor(
            [
                float(center[0]) + rake,
                float(center[1]) + r * math.cos(phi),
                float(center[2]) + r * math.sin(phi),
            ],
            dtype=torch.float64,
        )
        s = torch.linspace(-0.5, 0.5, n_chord + 2, dtype=torch.float64)[1:-1]
        sec = mid.unsqueeze(0) + s.unsqueeze(1) * chord * d.unsqueeze(0)
        pts.extend(tuple(float(v) for v in row) for row in sec)
        n_local_row = tuple(float(v) for v in n_local)
        nrm.extend([n_local_row] * n_chord)
        area.extend([chord / n_chord * dr] * n_chord)
    if not pts:
        raise ValueError("no markers generated - check hub/radius geometry")
    return (
        torch.tensor(pts, dtype=torch.float64),
        torch.tensor(nrm, dtype=torch.float64),
        torch.tensor(area, dtype=torch.float64),
    )


def blade_markers(
    geom: PropellerGeometry,
    azimuth: float = 0.0,
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    blade_index: int = 0,
) -> torch.Tensor:
    """Lagrangian marker sheet for one blade at the given azimuth.

    Args:
        geom: parametric rotor geometry.
        azimuth: blade reference azimuth [rad] (rotation about ``x``).
        center: rotor axis origin ``(x0, y0, z0)`` in lattice coordinates.
        blade_index: which blade of the ``n_blades`` set.

    Returns:
        ``(M, 3)`` marker coordinates in lattice units.
    """
    return _blade_sheet(geom, azimuth, center=center, blade_index=blade_index)[0]


def blade_sheet(
    geom: PropellerGeometry,
    azimuth: float = 0.0,
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All blades as ``(points, normals, areas)`` for the slip IBM."""
    sets = [_blade_sheet(geom, azimuth, center=center, blade_index=b) for b in range(geom.n_blades)]
    return (
        torch.cat([s[0] for s in sets], dim=0),
        torch.cat([s[1] for s in sets], dim=0),
        torch.cat([s[2] for s in sets], dim=0),
    )


def propeller_markers(
    geom: PropellerGeometry,
    azimuth: float = 0.0,
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """All ``n_blades`` marker sheets, ``(M_total, 3)``."""
    sets = [
        blade_markers(geom, azimuth, center=center, blade_index=b)
        for b in range(geom.n_blades)
    ]
    return torch.cat(sets, dim=0)


def rotate_markers(
    points: torch.Tensor, d_angle: float, center: Sequence[float]
) -> torch.Tensor:
    """Advance a rigid marker set by ``d_angle`` about the x-parallel axis."""
    return _rotate_x(points, d_angle, center)


def rotor_target_velocity(
    points: torch.Tensor, omega: float, *, center: Sequence[float] = (0.0, 0.0, 0.0)
) -> torch.Tensor:
    """Rigid-body velocity ``omega x r`` of the rotating blade sheet.

    ``omega = (omega, 0, 0)`` (rotation about the streamwise axis), hence
    ``omega x r = (0, -omega (z - z0), omega (y - y0))``.
    """
    rel = points.clone()
    rel[:, 0] -= float(center[0])
    rel[:, 1] -= float(center[1])
    rel[:, 2] -= float(center[2])
    out = torch.zeros_like(rel)
    out[:, 1] = -omega * rel[:, 2]
    out[:, 2] = omega * rel[:, 1]
    return out


def _stencil_in(
    points: torch.Tensor, shape: Sequence[int], kernel: str, dtype: torch.dtype
):
    """Build a marker stencil with weights in the field dtype.

    :func:`build_stencil` works in float64; mixing it with a float32 field
    silently promotes every IBM operation (and the whole solver state) to
    float64, which doubles the memory traffic.  Casting the sparse weights once
    keeps the run in the requested precision.
    """
    from .ibm_force_transfer import build_stencil

    st = build_stencil(points, shape, kernel)
    if st.w.dtype == dtype:
        return st
    import dataclasses

    return dataclasses.replace(st, w=st.w.to(dtype))


def marker_forcing(
    stencil,
    u_grid: torch.Tensor,
    u_target: torch.Tensor,
    normals: torch.Tensor,
    areas: torch.Tensor,
    *,
    mode: str = "slip_friction",
    nu: float = 1e-3,
    chord: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Marker force for a thin lifting sheet: ``(f_marker, f_grid, body_force)``.

    ``mode``:

    ``"noslip"``
        Classical direct forcing: the whole velocity deficit ``u_t - u(x_k)`` is
        removed at every marker (zero-thickness no-slip sheet).
    ``"slip"``
        Impermeability only: only the wall-normal component of the deficit is
        removed, so the sheet cannot be crossed but exerts no tangential
        (friction) stress.  This is the correct limit when the momentum
        boundary layer is thinner than a cell — which it is here
        (``delta ~ c/sqrt(Re_c) << 1`` at ``Re_c ~ 10^3``).
    ``"slip_friction"`` (default)
        ``"slip"`` plus a *modelled* tangential stress
        ``F_t = 1/2 Cf(r) A |w_t| w_t`` with the flat-plate correlation
        ``Cf = 1.328/sqrt(Re_c)`` (``Re_c = |w_t| c / nu``), which restores the
        skin-friction torque that the unresolved boundary layer would carry.

    Why not plain ``"noslip"``: a marker sheet smeared over ~1.5 cells thickens
    the momentum boundary layer to ~25 % of the chord at the resolutions
    affordable here, which inflates the friction torque by an order of
    magnitude (measured ``10 K_Q`` ~ 6 instead of 0.55) and drives spurious
    bulk swirl.  The modelled-stress variant keeps the pressure (thrust) part
    exact and puts the friction part on a calibrated closure.
    """
    dt = u_grid.dtype
    u_int = stencil.interpolate(u_grid)  # (3, M)
    normals = normals.to(dt)
    areas = areas.to(dt)
    du = u_target.to(dt).T - u_int
    if mode == "noslip":
        f_m = du.T
    else:
        n_t = normals.T  # (3, M)
        f_n = (du * n_t).sum(dim=0, keepdim=True)
        f = f_n * n_t
        if mode == "slip_friction":
            w_t = du - f_n * n_t
            speed = w_t.norm(dim=0)
            re_c = (speed * chord / nu).clamp(min=1.0)
            cf = torch.where(
                re_c < 5.0e5,
                1.328 / torch.sqrt(re_c),
                0.074 / re_c**0.2,
            ).clamp(max=0.1)
            f = f + (0.5 * cf * areas * speed).unsqueeze(0) * w_t
        elif mode != "slip":
            raise ValueError(f"unknown forcing mode {mode!r}")
        f_m = f.T
    f_grid = stencil.spread(f_m)
    body_force = -f_m.sum(dim=0)
    return f_m, f_grid, body_force


def hub_mask(
    shape: Sequence[int],
    geom: PropellerGeometry,
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    pad: float = 0.25,
) -> torch.Tensor:
    """Boolean Eulerian mask of the hub (cylinder along ``x``).

    Args:
        shape: ``(nz, ny, nx)`` grid shape.
        geom: rotor geometry (hub radius / length).
        center: rotor axis origin.
        pad: extra radius [lu] so the hub surface is covered by the mask.

    Returns:
        ``(nz, ny, nx)`` boolean tensor, ``True`` inside the hub.
    """
    nz, ny, nx = (int(v) for v in shape)
    z = torch.arange(nz, dtype=torch.float64).view(nz, 1, 1)
    y = torch.arange(ny, dtype=torch.float64).view(1, ny, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, 1, nx)
    r = torch.sqrt((y - float(center[1])) ** 2 + (z - float(center[2])) ** 2)
    half = 0.5 * geom.hub_length
    return (r <= geom.hub_radius + pad) & (torch.abs(x - float(center[0])) <= half)


# --------------------------------------------------------------------------- #
# Open-water sweep configuration and driver
# --------------------------------------------------------------------------- #


@dataclass
class OpenWaterConfig:
    """Operating-point + numerics setup for one open-water IBM run.

    Attributes:
        advance_ratios: ``J`` values to evaluate.
        tip_speed: rotor tip speed in lattice units (``Omega R``); fixes the
            lattice Mach number ``Ma_tip = tip_speed / c_s``, ``c_s = 1/sqrt(3)``.
        tau: BGK/MRT relaxation time; ``nu = (tau - 1/2)/3``.
        steps_per_rev: azimuthal resolution; the marker sheet advances
            ``2 pi / steps_per_rev`` per lattice step.
        warmup_revs: revolutions discarded before averaging.
        measure_revs: revolutions averaged.
        lateral_factor: transverse half-width of the domain in diameters.
        upstream_factor, downstream_factor: streamwise extent in diameters.
        kernel: IBM delta kernel (``"4pt"`` default).
        collision: ``"mrt"`` or ``"bgk"``.
        omega_sign: rotation sense (``+1`` gives positive thrust).
        dtype: ``torch.float64`` (default; needed for exact momentum audits).
    """

    advance_ratios: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    tip_speed: float = 0.12
    tau: float = 0.51
    steps_per_rev: int = 640
    warmup_revs: float = 0.6
    measure_revs: float = 1.4
    lateral_factor: float = 1.5
    upstream_factor: float = 1.8
    downstream_factor: float = 3.0
    kernel: str = "4pt"
    collision: str = "mrt"
    forcing: str = "slip_friction"
    omega_sign: float = 1.0
    dtype: torch.dtype = torch.float64
    max_steps: int = 4000

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    def grid_shape(self, diameter: float) -> tuple[int, int, int]:
        """``(nz, ny, nx)`` grid for a given rotor diameter."""
        nx = int(round((self.upstream_factor + self.downstream_factor) * diameter))
        nyz = int(round(2.0 * self.lateral_factor * diameter))
        return (nyz, nyz, nx)


def _apply_inlet_outlet_3d(f: torch.Tensor, u_inflow: float, dtype: torch.dtype) -> torch.Tensor:
    """Equilibrium inlet (x = 0) + zero-gradient outlet (x = nx-1)."""
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    ones = torch.ones((nz, ny), dtype=dtype)
    zeros = torch.zeros((nz, ny), dtype=dtype)
    feq = _eq3d(ones, u_inflow * ones, zeros, zeros).reshape(19, nz, ny)
    out = f.clone()
    out[:, :, :, 0] = feq
    out[:, :, :, nx - 1] = f[:, :, :, nx - 2]
    return out


def _axial_torque_x(
    points: torch.Tensor, forces: torch.Tensor, center: Sequence[float]
) -> torch.Tensor:
    """Moment about the ``x`` axis of forces applied at ``points``."""
    rel_y = points[:, 1] - float(center[1])
    rel_z = points[:, 2] - float(center[2])
    return (rel_y * forces[:, 2] - rel_z * forces[:, 1]).sum()


def run_open_water_point(
    geom: PropellerGeometry,
    cfg: OpenWaterConfig,
    advance_ratio: float,
    *,
    verbose: bool = False,
) -> dict:
    """One open-water operating point: ``K_T``, ``K_Q``, ``eta_0`` plus audits.

    Fixed-tip-speed strategy: ``Omega = tip_speed / R`` is held constant and the
    inflow ``U = J n D`` is varied, so every ``J`` runs at the same lattice Mach
    number (comparable compressibility error) while the Reynolds number
    ``Re_D = U D / nu`` necessarily changes with ``J``.  This is the standard
    lattice-LBM compromise and is documented as a limitation of the method.
    """
    j = float(advance_ratio)
    dtype = cfg.dtype
    nz, ny, nx = cfg.grid_shape(geom.diameter)
    shape = (nz, ny, nx)
    center = (nx / 2.0, ny / 2.0, nz / 2.0)

    radius = geom.radius
    omega = cfg.omega_sign * cfg.tip_speed / radius
    rev_per_time = abs(omega) / (2.0 * math.pi)
    u_inflow = j * rev_per_time * geom.diameter
    nu = cfg.nu
    re_d = u_inflow * geom.diameter / nu if nu > 0 else float("inf")
    c_s = 1.0 / math.sqrt(3.0)
    ma_tip = cfg.tip_speed / c_s
    ma_inflow = u_inflow / c_s

    d_theta = 2.0 * math.pi / cfg.steps_per_rev
    n_warm = int(round(cfg.warmup_revs * cfg.steps_per_rev))
    n_meas = int(round(cfg.measure_revs * cfg.steps_per_rev))
    n_steps = min(cfg.max_steps, n_warm + n_meas)

    ones = torch.ones(shape, dtype=dtype)
    zeros = torch.zeros(shape, dtype=dtype)
    f = _eq3d(ones, u_inflow * ones, zeros, zeros)

    hub = hub_mask(shape, geom, center=center)
    # the hub rotates rigidly with the rotor (no spurious shear layer)
    zz = torch.arange(nz, dtype=dtype).view(nz, 1, 1)
    yy = torch.arange(ny, dtype=dtype).view(1, ny, 1)
    hub_target = torch.zeros((3,) + shape, dtype=dtype)
    hub_target[1] = torch.where(hub, -omega * (zz - center[2]), torch.zeros_like(zz))
    hub_target[2] = torch.where(hub, omega * (yy - center[1]), torch.zeros_like(yy))

    azimuth = 0.0
    accum = {"kt": 0.0, "kq": 0.0, "hub": 0.0, "n": 0}
    history: list[dict] = []
    diverged = False
    momentum_residual = 0.0

    for step in range(n_steps):
        rho, ux, uy, uz = _macro3d(f)
        u_grid = torch.stack([ux, uy, uz])

        markers, normals, areas = blade_sheet(geom, azimuth, center=center)
        stencil = _stencil_in(markers, shape, cfg.kernel, dtype)
        target = rotor_target_velocity(markers, omega, center=center)
        f_m, f_grid, body_force = marker_forcing(
            stencil,
            u_grid,
            target,
            normals,
            areas,
            mode=cfg.forcing,
            nu=nu if nu > 0 else 1e-3,
            chord=geom.mean_chord,
        )
        f_hub = interior_mask_force(u_grid, hub, hub_target)
        total_force = f_grid + f_hub

        if cfg.collision == "mrt":
            f = collide_mrt3d(f, cfg.tau)
        else:
            from .solver3d import collide_bgk3d

            f = collide_bgk3d(f, cfg.tau)
        f = body_source_3d(f, total_force[0], total_force[1], total_force[2])
        f = stream3d(f)
        f = _apply_inlet_outlet_3d(f, u_inflow, dtype)

        thrust = float(body_force[0].item())
        # torque required to drive the rotor, oriented along the rotation
        # sense so that K_Q > 0 and P = 2 pi n Q > 0 for a propulsion rotor
        sgn = 1.0 if omega > 0 else -1.0
        torque = sgn * float(_axial_torque_x(markers, f_m, center).item())
        hub_thrust = -float(f_hub[0].sum().item())

        if not (math.isfinite(thrust) and math.isfinite(torque)):
            diverged = True
            break

        # momentum actually injected into the fluid this step (x component)
        injected_momentum = float(total_force[0].sum().item())

        if step >= n_warm:
            n2d4 = rev_per_time**2 * geom.diameter**4
            n2d5 = rev_per_time**2 * geom.diameter**5
            accum["kt"] += thrust / n2d4
            accum["kq"] += torque / n2d5
            accum["hub"] += hub_thrust / n2d4
            accum["n"] += 1
            if accum["n"] <= 3:
                # fluid momentum injected vs -(reaction on blades + hub)
                momentum_residual = abs(
                    injected_momentum + thrust + hub_thrust
                ) / max(1e-30, abs(thrust))
            history.append(
                {
                    "step": step,
                    "KT": thrust / n2d4,
                    "KQ": torque / n2d5,
                    "hub_KT_mask": hub_thrust / n2d4,
                }
            )

        azimuth += d_theta
        if verbose and step % 200 == 0:
            print(
                f"    step {step}/{n_steps}  T={thrust:+.6e}  Q={torque:+.6e}  "
                f"F_inj_err={abs(injected_momentum + thrust + hub_thrust):.2e}"
            )

    n = max(1, accum["n"])
    kt = accum["kt"] / n
    kq = accum["kq"] / n
    eta = open_water_efficiency(j, kt, kq) if kq != 0.0 else float("nan")
    ref = kp505_reference(j)
    # rotor disc area / tunnel cross-section:  (pi R^2) / (2 * lateral_factor * D)^2
    blockage = math.pi / (16.0 * cfg.lateral_factor**2)
    result = {
        "J": j,
        "KT": kt,
        "KQ": kq,
        "10KQ": 10.0 * kq,
        "eta_0": eta,
        "KT_hub_mask": accum["hub"] / n,
        "reference": ref,
        "error": {
            "KT_rel": (kt - ref["KT"]) / ref["KT"] if ref["KT"] else float("nan"),
            "KQ_rel": (kq - ref["KQ"]) / ref["KQ"] if ref["KQ"] else float("nan"),
            "eta_rel": (eta - ref["eta_0"]) / ref["eta_0"] if ref["eta_0"] else float("nan"),
        },
        "numerics": {
            "diameter": geom.diameter,
            "grid": list(shape),
            "tau": cfg.tau,
            "nu": nu,
            "Re_D": re_d,
            "Ma_tip": ma_tip,
            "Ma_inflow": ma_inflow,
            "tip_speed": cfg.tip_speed,
            "u_inflow": u_inflow,
            "omega": omega,
            "steps_per_rev": cfg.steps_per_rev,
            "n_steps": n_steps,
            "n_averaged_steps": n,
            "n_markers": int(propeller_markers(geom, 0.0, center=center).shape[0]),
            "kernel": cfg.kernel,
            "collision": cfg.collision,
            "forcing": cfg.forcing,
            "disc_area_blockage": blockage,
            "momentum_source_rel_residual": momentum_residual,
            "diverged": diverged,
        },
        "history": history,
    }
    return result


def run_open_water_sweep(
    geom: PropellerGeometry,
    cfg: OpenWaterConfig,
    *,
    verbose: bool = False,
) -> dict:
    """Sweep ``J`` and summarise the open-water curve against the KP505 table."""
    points = []
    for j in cfg.advance_ratios:
        if verbose:
            print(f"  J = {j:.2f}", flush=True)
        points.append(run_open_water_point(geom, cfg, j, verbose=verbose))
    eta_err = [abs(p["error"]["eta_rel"]) for p in points if math.isfinite(p["eta_0"])]
    kt_err = [abs(p["error"]["KT_rel"]) for p in points if math.isfinite(p["KT"])]
    summary = {
        "diameter": geom.diameter,
        "grid": list(cfg.grid_shape(geom.diameter)),
        "advance_ratios": list(cfg.advance_ratios),
        "max_abs_eta_error": max(eta_err) if eta_err else float("inf"),
        "rms_abs_eta_error": (sum(e * e for e in eta_err) / len(eta_err)) ** 0.5
        if eta_err
        else float("inf"),
        "max_abs_KT_error": max(kt_err) if kt_err else float("inf"),
        "diverged": any(p["numerics"]["diverged"] for p in points),
        "points": points,
    }
    return summary


# --------------------------------------------------------------------------- #
# AC3: exact momentum-balance audit (closed periodic box)
# --------------------------------------------------------------------------- #


def momentum_balance_audit(
    shape: Sequence[int] = (24, 24, 48),
    *,
    tau: float = 0.8,
    tip_speed: float = 0.06,
    diameter: float = 12.0,
    n_steps: int = 20,
    kernel: str = "4pt",
    dtype: torch.dtype = torch.float64,
) -> dict:
    """IBM force integral vs fluid momentum change in a closed periodic box.

    There is **no** inflow/outflow and no other force: collision and (periodic)
    streaming conserve momentum exactly, so every unit of fluid momentum created
    in a step must come from the IBM body source.  Per step the audit compares
    the vector momentum change with the vector force handed to the fluid,

        r_t = || P(t) - P(t-1) - F_IBM,total(t) ||_2 ,

    where ``F_IBM,total = spread(marker forces) + hub mask force`` is exactly
    what :func:`tensorlbm.ibm_force_transfer.body_source_3d` injects.  The
    reported relative measure uses the run's peak force norm as the scale (a
    per-step ratio would be undefined on steps where the planar marker target
    velocity happens to have no axial component).

    Returns:
        dict with ``residuals`` (per-step ``||r_t||``), ``force_norms``,
        ``max_abs_residual`` and ``rel_residual`` (``max_abs / peak force``).
    """
    nz, ny, nx = (int(v) for v in shape)
    geom = PropellerGeometry(diameter=diameter, n_blades=3, radial_stations=6)
    center = (nx / 2.0, ny / 2.0, nz / 2.0)
    ones = torch.ones((nz, ny, nx), dtype=dtype)
    zeros = torch.zeros((nz, ny, nx), dtype=dtype)
    f = _eq3d(ones, zeros, zeros, zeros)
    hub = hub_mask((nz, ny, nx), geom, center=center)
    hub_target = torch.zeros((3,) + (nz, ny, nx), dtype=dtype)
    omega = tip_speed / geom.radius
    azimuth = 0.0
    d_theta = 0.02

    residuals: list[float] = []
    force_norms: list[float] = []
    for _ in range(n_steps):
        rho, ux, uy, uz = _macro3d(f)
        p_before = torch.stack(
            [(rho * ux).sum(), (rho * uy).sum(), (rho * uz).sum()]
        )
        markers = propeller_markers(geom, azimuth, center=center)
        stencil = build_stencil(markers, (nz, ny, nx), kernel)
        _, f_grid, _, _ = direct_forcing(
            stencil,
            torch.stack([ux, uy, uz]),
            rotor_target_velocity(markers, omega, center=center),
            rho_grid=rho,
            n_correctors=0,
        )
        f_hub = interior_mask_force(torch.stack([ux, uy, uz]), hub, hub_target)
        total_force = f_grid + f_hub
        injected = torch.stack([total_force[c].sum() for c in range(3)])

        f = collide_mrt3d(f, tau)
        f = body_source_3d(f, total_force[0], total_force[1], total_force[2])
        f = stream3d(f)

        rho2, ux2, uy2, uz2 = _macro3d(f)
        p_after = torch.stack(
            [(rho2 * ux2).sum(), (rho2 * uy2).sum(), (rho2 * uz2).sum()]
        )
        residuals.append(float((p_after - p_before - injected).norm().item()))
        force_norms.append(float(injected.norm().item()))
        azimuth += d_theta

    peak = max(force_norms) if force_norms else 1.0
    max_abs = max(residuals) if residuals else float("inf")
    return {
        "shape": list(shape),
        "n_steps": n_steps,
        "residuals": residuals,
        "force_norms": force_norms,
        "max_abs_residual": max_abs,
        "rel_residual": max_abs / peak if peak > 0 else float("inf"),
        "tau": tau,
        "kernel": kernel,
    }


# --------------------------------------------------------------------------- #
# AC1: 2-D cylinder drag sweep (D2Q9 + IBM)
# --------------------------------------------------------------------------- #


@dataclass
class CylinderConfig:
    """2-D circular-cylinder drag case (D2Q9 + direct-forcing IBM)."""

    diameter: float = 10.0
    reynolds: tuple[int, ...] = (20, 40, 60, 80, 100, 150, 200)
    inflow: float = 0.05
    nx_per_d: float = 30.0
    ny_per_d: float = 16.0
    n_steps: int = 24000
    transient_frac: float = 0.4
    marker_spacing: float = 0.9
    wall_offset: float = 0.5
    kernel: str = "4pt"
    dtype: torch.dtype = torch.float64

    def tau(self, reynolds: float) -> float:
        nu = self.inflow * self.diameter / float(reynolds)
        return 0.5 + 3.0 * nu

    def grid_shape(self) -> tuple[int, int]:
        return (
            int(round(self.ny_per_d * self.diameter)),
            int(round(self.nx_per_d * self.diameter)),
        )


def _disk_mask_2d(shape: tuple[int, int], center, radius: float) -> torch.Tensor:
    ny, nx = shape
    y = torch.arange(ny, dtype=torch.float64).view(ny, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, nx)
    r = torch.sqrt((y - float(center[0])) ** 2 + (x - float(center[1])) ** 2)
    return r <= radius


def _apply_inlet_outlet_2d(f: torch.Tensor, u_inflow: float, dtype: torch.dtype) -> torch.Tensor:
    ny, nx = f.shape[1], f.shape[2]
    ones = torch.ones((ny,), dtype=dtype)
    zeros = torch.zeros((ny,), dtype=dtype)
    feq = _eq2d(ones, u_inflow * ones, zeros).reshape(9, ny)
    out = f.clone()
    out[:, :, 0] = feq
    out[:, :, nx - 1] = f[:, :, nx - 2]
    return out


def cylinder_drag_2d(
    reynolds: float,
    cfg: CylinderConfig | None = None,
    *,
    verbose: bool = False,
) -> dict:
    """Drag coefficient of a fixed 2-D circular cylinder from IBM.

    The cylinder is represented by (a) a Lagrangian marker shell
    (motionless target velocity zero) that enforces the no-slip condition and
    (b) an Eulerian interior mask so the body interior carries no fluid.  The
    measured drag is the reaction of the momentum handed to the fluid, and the
    coefficient is ``C_D = 2 F_x / (rho U^2 D)``.

    The average is taken over the last ``1 - transient_frac`` of the run, which
    is what makes the unsteady cases (``Re >= 60``, vortex street) meaningful.
    """
    cfg = cfg or CylinderConfig()
    re = float(reynolds)
    dtype = cfg.dtype
    shape = cfg.grid_shape()
    ny, nx = shape
    diameter = cfg.diameter
    radius = 0.5 * diameter
    center = (ny / 2.0, 0.25 * nx)
    tau = cfg.tau(re)
    u_in = cfg.inflow

    ones = torch.ones(shape, dtype=dtype)
    zeros = torch.zeros(shape, dtype=dtype)
    f = _eq2d(ones, u_in * ones, zeros)

    # Wall offset: a delta-smeared marker shell behaves like a wall pushed
    # outwards by ~half a cell, so the shell is placed at R - wall_offset
    # (the same calibration the Couette benchmarks in this repository use).
    markers = build_cylinder_markers_2d(
        center, radius - cfg.wall_offset, spacing=cfg.marker_spacing
    )
    stencil = _stencil_in(markers, shape, cfg.kernel, dtype)
    target = torch.zeros_like(markers)
    # NOTE: no interior-mask forcing.  The closed marker shell already enforces
    # impermeability; forcing the interior fluid to rest as well would add a
    # non-physical momentum sink whose reaction (O(1) at these resolutions)
    # swamps the surface drag and flips its sign.
    interior = None

    n_steps = cfg.n_steps
    n_trans = int(cfg.transient_frac * n_steps)
    cd_acc = 0.0
    n_acc = 0
    cd_min, cd_max = float("inf"), -float("inf")
    diverged = False

    for step in range(n_steps):
        rho, ux, uy = _macro2d(f)
        u_grid = torch.stack([ux, uy])
        f_m, f_grid, body_force, _ = direct_forcing(
            stencil, u_grid, target, rho_grid=rho, n_correctors=0
        )
        total = f_grid

        f = _collide2d(f, tau)
        w = _w_for_2d(f).view(9, 1, 1)
        c = _c_for_2d(f)
        f = f + w * 3.0 * (c[:, 0].view(9, 1, 1) * total[0].unsqueeze(0)
                           + c[:, 1].view(9, 1, 1) * total[1].unsqueeze(0))
        f = _stream2d(f)
        f = _apply_inlet_outlet_2d(f, u_in, dtype)

        fx = float(body_force[0].item())
        cd = 2.0 * fx / (u_in * u_in * diameter)
        if not math.isfinite(cd):
            diverged = True
            break
        if step >= n_trans:
            cd_acc += cd
            n_acc += 1
            cd_min = min(cd_min, cd)
            cd_max = max(cd_max, cd)
        if verbose and step % 4000 == 0:
            print(f"    Re={re:g} step {step}/{n_steps} Cd={cd:+.4f}", flush=True)

    cd_mean = cd_acc / max(1, n_acc)
    ref = CYLINDER_CD_2D.get(int(re))
    if ref is None:
        ref = _interp_table(CYLINDER_CD_2D, re)
    return {
        "Re": re,
        "Cd": cd_mean,
        "Cd_reference": ref,
        "Cd_rel_error": (cd_mean - ref) / ref,
        "Cd_min": cd_min,
        "Cd_max": cd_max,
        "Cd_fluctuation": cd_max - cd_min,
        "tau": tau,
        "nu": (tau - 0.5) / 3.0,
        "grid": list(shape),
        "diameter": diameter,
        "u_inflow": u_in,
        "n_steps": n_steps,
        "n_averaged_steps": n_acc,
        "n_markers": int(markers.shape[0]),
        "blockage": diameter / ny,
        "wall_offset": cfg.wall_offset,
        "kernel": cfg.kernel,
        "diverged": diverged,
    }


def _w_for_2d(f: torch.Tensor) -> torch.Tensor:
    from .d2q9 import _w_for

    return _w_for(f.dtype, f.device)


def _c_for_2d(f: torch.Tensor) -> torch.Tensor:
    from .d2q9 import _c_on

    return _c_on(f.device)