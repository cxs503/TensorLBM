"""INSEAN E779A propeller — unsteady open water, blade-rate spectrum and
tip-vortex cavitation-inception validation face (moving-boundary LBM).

What this module is
-------------------
A *validation face* (not a new solver) that drives the repository's D3Q19
lattice-Boltzmann core with a **rigidly rotating, cambered, pitched blade
geometry** for the INSEAN E779A four-bladed model propeller, and exposes four
measurement channels that are all anchored on machine-checkable quantities:

1. **geometry gate** — expanded area ratio ``A_E/A_0``, pitch ratio ``P/D`` at
   ``r/R = 0.7`` and hub ratio ``D_hub/D`` measured *from the voxelised mask*
   (not from the input numbers) against the E779A drawing values;
2. **open-water loads** — ``K_T(J)``, ``K_Q(J)``, ``eta_0(J)`` from the Ladd
   (1994) moving-wall momentum exchange, with a per-revolution block-mean
   convergence residual (the "period-average" criterion);
3. **blade-rate content** — per-blade thrust time histories and their
   amplitude spectra (shaft harmonics ``k n`` and blade rate ``Z n``);
4. **tip-vortex cavitation inception** — ``Q`` / ``lambda_2`` vortex-core
   identification, core minimum pressure and the resulting inception number
   ``sigma_i = (p_inf - p_min) / (0.5 rho V_R^2)``.

Evidence grading (see ``docs/P129W3_N04_INSEAN_E779A_OPENWATER.md``)
--------------------------------------------------------------------
REAL: geometry-gate measurements, momentum-exchange bookkeeping (machine
precision), kinematic/scaling laws, negative controls, spectrum peak
positions, grid-convergence machinery.
WITHHELD: absolute comparison against the INSEAN open-water experiment and
the published cavitation-inception curve — the execution environment is
air-gapped and ships no primary-source E779A table, so any experimental
numbers would have to be invented.  The comparison machinery is implemented
and is fed from :func:`load_experimental_anchor` (provenance-tagged JSON); it
reports ``WITHHELD`` until a table with a verifiable source is injected.
NOT_SUPPORTED: sheet/cloud cavitation *morphology* once ``sigma < sigma_i`` —
this face only converts a pressure-field minimum into an inception number.

Units
-----
Lattice units inside the solver (``dx = dt = rho = 1``); coefficients are
dimensionless by construction, so the lattice run is directly comparable with
model-scale coefficients.  Physical parameters (``D = 0.227 m``, ``n = 25``
rps, water at 20 degC) are used for the reported Reynolds/advance ratios and
for the inception number.

References
----------
* Ladd, A.J.C. (1994) *J. Fluid Mech.* 271, 285 — moving-wall momentum exchange.
* Guo, Z., Zheng, C., Shi, B. (2002) *Phys. Fluids* 14, 385 — forcing scheme.
* Roache, P.J. (1994) *J. Fluids Eng.* 116, 405 — GCI.
* Jeong, J., Hussain, F. (1995) *J. Fluid Mech.* 285, 69 — lambda_2 criterion.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .d3q19 import C, W, equilibrium3d
from .solver3d import collide_mrt3d_fused, stream3d

__all__ = [
    "E779ASpec",
    "E779A_SPEC",
    "OpenWaterConfig",
    "advance_ratio",
    "bemt_reference",
    "blade_thrust_spectrum",
    "build_e779a_mask",
    "gci_roache",
    "geometry_gate",
    "geometry_metrics",
    "load_experimental_anchor",
    "openwater_curve_criteria",
    "run_openwater_point",
    "tip_vortex_inception",
    "vortex_criteria_fields",
]

CS = 1.0 / math.sqrt(3.0)  # lattice speed of sound


# ---------------------------------------------------------------------------
# 1.  Specimen description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class E779ASpec:
    """INSEAN E779A propeller description.

    The first block is the *drawing-given* specification used by the
    acceptance gates.  The second block is the set of **declared
    approximations** this repository uses because the detailed E779A drawings
    (radial chord / pitch / skew / rake tables, NACA-66-mod section ordinates)
    are not available offline: chord and thickness laws are analytic and
    calibrated so that the three drawing-given integral quantities are matched
    exactly.
    """

    # --- drawing-given (gate) ---------------------------------------------
    diameter_m: float = 0.227
    n_blades: int = 4
    pitch_ratio_07: float = 1.10
    expanded_area_ratio: float = 0.689
    hub_diameter_ratio: float = 0.18
    # --- operational point ------------------------------------------------
    n_rps: float = 25.0
    rho: float = 1000.0
    nu: float = 1.0e-6
    p_vapour: float = 2330.0
    p_ref: float = 101325.0
    # --- declared approximations (NOT drawing data) -----------------------
    hub_length_ratio: float = 0.36
    chord_peak_rfrac: float = 0.70
    chord_root_frac: float = 0.72
    chord_tip_frac: float = 0.30
    thickness_ratio_07: float = 0.035
    camber_ratio: float = 0.020
    skew_deg: float = 0.0
    rake_ratio: float = 0.0
    handedness: float = -1.0   # sign of the helicoid: x = handedness * tan(beta) * s

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m

    @property
    def hub_radius_ratio(self) -> float:
        return 0.5 * self.hub_diameter_ratio

    def advance_speed(self, j: float) -> float:
        """Inflow speed [m/s] for advance ratio ``J``."""
        return j * self.n_rps * self.diameter_m

    def tip_speed(self) -> float:
        """Tip speed [m/s]."""
        return math.pi * self.n_rps * self.diameter_m

    def reynolds_diameter(self, j: float) -> float:
        """``Re_D = V_A D / nu`` (advance-speed based, the usual open-water form)."""
        return self.advance_speed(j) * self.diameter_m / self.nu

    def reynolds_tip(self) -> float:
        """``Re = V_R c_0.7 / nu`` with ``V_R`` the resultant tip inflow speed."""
        v_r = math.hypot(self.advance_speed(0.0), self.tip_speed())
        return v_r * 0.3 * self.diameter_m / self.nu

    def cavitation_number(self, j: float) -> float:
        """``sigma_n = (p_ref - p_v) / (0.5 rho (n D)^2)``."""
        return (self.p_ref - self.p_vapour) / (0.5 * self.rho * (self.n_rps * self.diameter_m) ** 2)

    def resultant_inflow(self, j: float, r_frac: float = 0.7) -> float:
        """``V_R = sqrt(V_A^2 + (2 pi n r)^2)`` at ``r_frac``."""
        v_t = 2.0 * math.pi * self.n_rps * r_frac * self.radius_m
        return math.hypot(self.advance_speed(j), v_t)


E779A_SPEC = E779ASpec()


# ---------------------------------------------------------------------------
# 2.  Radial geometry (declared analytic laws, integral-matched to the drawing)
# ---------------------------------------------------------------------------


def _chord_shape(s: np.ndarray, spec: E779ASpec) -> np.ndarray:
    """Normalised chord shape ``c(s)/c_peak`` for ``s in [0, 1]`` (hub -> tip)."""
    s_peak = (spec.chord_peak_rfrac - spec.hub_radius_ratio) / (1.0 - spec.hub_radius_ratio)
    s_peak = float(np.clip(s_peak, 0.2, 0.9))
    up = spec.chord_root_frac + (1.0 - spec.chord_root_frac) * (s / s_peak)
    down = 1.0 + (spec.chord_tip_frac - 1.0) * ((s - s_peak) / (1.0 - s_peak))
    return np.where(s <= s_peak, up, down)


def _chord_scale(diameter_lu: float, spec: E779ASpec, n_radial: int = 20001) -> float:
    """Scale factor making the expanded area ratio exactly ``spec.expanded_area_ratio``.

    ``A_E = Z * int_{R_hub}^{R} c(r) dr`` and ``A_0 = pi D^2 / 4``.
    """
    s = np.linspace(0.0, 1.0, n_radial)
    r = spec.hub_radius_ratio * diameter_lu + s * (0.5 * diameter_lu - spec.hub_radius_ratio * diameter_lu)
    dr = np.gradient(r)
    integral = float(np.sum(_chord_shape(s, spec) * dr))
    target_area = spec.expanded_area_ratio * math.pi * diameter_lu**2 / 4.0
    return target_area / (spec.n_blades * integral)


def _thickness_ratio(r_frac: np.ndarray, spec: E779ASpec) -> np.ndarray:
    """Local thickness ratio ``t/c``: thick root, thin tip (declared law)."""
    ratio = spec.thickness_ratio_07 * (1.0 + 0.9 * (spec.chord_peak_rfrac - r_frac))
    return np.clip(ratio, 0.4 * spec.thickness_ratio_07, 1.8 * spec.thickness_ratio_07)


def _naca4_half_thickness(xi: np.ndarray) -> np.ndarray:
    x = np.clip(xi, 0.0, 1.0)
    return 5.0 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)


def _camber_line(xi: np.ndarray, spec: E779ASpec) -> np.ndarray:
    """Parabolic camber (declared stand-in for the NACA a=0.8 mean line)."""
    return 4.0 * spec.camber_ratio * xi * (1.0 - xi)


# ---------------------------------------------------------------------------
# 3.  Voxelisation
# ---------------------------------------------------------------------------


def _box_indices(n: int, centre: float, half_width: float) -> tuple[int, int]:
    lo = int(max(0, math.floor(centre - half_width)))
    hi = int(min(n, math.ceil(centre + half_width) + 1))
    return lo, hi


def build_e779a_mask(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    *,
    diameter_lu: float,
    spec: E779ASpec = E779A_SPEC,
    angle_deg: float = 0.0,
    min_thickness_lu: float = 0.9,
    chord_pad_lu: float = 0.0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Voxelise the E779A propeller (hub + ``spec.n_blades`` cambered blades).

    The blade surface is built in the *developed cylinder* plane: at radius
    ``r`` the section chord line makes the helix pitch angle
    ``beta(r) = atan(P / (2 pi r))`` with the tangential direction, so that

    .. math:: \\xi = s\\cos\\beta - x\\sin\\beta,\\quad
              \\eta = s\\sin\\beta + x\\cos\\beta

    are the chordwise / chord-normal coordinates of a cell located at
    tangential offset ``s = r (theta - theta_k)`` and axial offset ``x`` from
    the blade mid-chord.  A cell is solid when
    ``|xi| <= c/2`` and ``|eta - camber(xi)| <= t(xi)/2``.

    Returns a dict with ``mask`` (``(nz, ny, nx)`` bool), ``blade_index``
    (``(nz, ny, nx)`` int8, ``-1`` for hub / fluid) and the geometry audit.
    """
    dev = torch.device(device)
    R = 0.5 * diameter_lu
    R_hub = R * spec.hub_diameter_ratio
    hub_len = diameter_lu * spec.hub_length_ratio
    pitch = spec.pitch_ratio_07 * diameter_lu
    c_scale = _chord_scale(diameter_lu, spec)

    mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    blade_index = torch.full((nz, ny, nx), -1, dtype=torch.int8, device=dev)

    # -- hub (rotates rigidly, excluded from blade load bookkeeping) --------
    yh0, yh1 = _box_indices(ny, cy, R_hub + 2.0)
    zh0, zh1 = _box_indices(nz, cz, R_hub + 2.0)
    xh0, xh1 = _box_indices(nx, cx, 0.5 * hub_len + 2.0)
    yy_h = torch.arange(yh0, yh1, device=dev, dtype=torch.float32) - cy
    zz_h = torch.arange(zh0, zh1, device=dev, dtype=torch.float32) - cz
    xx_h = torch.arange(xh0, xh1, device=dev, dtype=torch.float32) - cx
    r_hub = torch.sqrt(yy_h[None, :, None] ** 2 + zz_h[:, None, None] ** 2)
    hub = (r_hub <= R_hub) & (xx_h[None, None, :].abs() <= 0.5 * hub_len)
    mask[zh0:zh1, yh0:yh1, xh0:xh1] |= hub.bool()

    # -- blades -------------------------------------------------------------
    y0, y1 = _box_indices(ny, cy, R + 2.5)
    z0, z1 = _box_indices(nz, cz, R + 2.5)
    x0, x1 = _box_indices(nx, cx, 0.5 * pitch + 3.0)
    yy = torch.arange(y0, y1, device=dev, dtype=torch.float32) - cy
    zz = torch.arange(z0, z1, device=dev, dtype=torch.float32) - cz
    xx = torch.arange(x0, x1, device=dev, dtype=torch.float32) - cx
    rr = torch.sqrt(zz[:, None, None] ** 2 + yy[None, :, None] ** 2)
    rr = rr.expand(z1 - z0, y1 - y0, x1 - x0).contiguous()
    th = torch.atan2(zz[:, None, None], yy[None, :, None]).expand_as(rr).contiguous()
    xa = xx[None, None, :].expand_as(rr).contiguous()
    r_frac = rr / R

    annulus = (r_frac >= spec.hub_radius_ratio * 0.985) & (r_frac <= 1.0)
    r_ann = rr[annulus]
    th_ann = th[annulus]
    x_ann = xa[annulus]

    s_frac = (r_ann - R * spec.hub_radius_ratio) / (R * (1.0 - spec.hub_radius_ratio))
    shape = torch.as_tensor(_chord_shape(s_frac.numpy(), spec), device=dev)
    chord = shape * c_scale
    beta = torch.atan(torch.as_tensor(pitch, device=dev) / (2.0 * math.pi * r_ann))
    cos_b, sin_b = torch.cos(beta), torch.sin(beta)
    t_ratio = torch.as_tensor(_thickness_ratio((r_ann / R).numpy(), spec), device=dev)

    mask_ann = torch.zeros_like(r_ann, dtype=torch.bool)
    idx_ann = torch.full_like(r_ann, -1, dtype=torch.int8)
    for k in range(spec.n_blades):
        phase = math.radians(angle_deg) + 2.0 * math.pi * k / spec.n_blades
        d_theta = torch.atan2(torch.sin(th_ann - phase), torch.cos(th_ann - phase))
        s_tan = r_ann * d_theta
        # right-handed helicoid: the blade surface advances +x with +theta,
        # i.e. x = tan(beta) * s (see module docstring / handedness note).
        xi = s_tan * cos_b - x_ann * sin_b
        eta = s_tan * sin_b + x_ann * cos_b
        xc = (xi / chord.clamp(min=1e-6) + 0.5).clamp(0.0, 1.0)
        camber = torch.as_tensor(_camber_line(xc.numpy(), spec), device=dev) * chord
        half_t = 0.5 * torch.as_tensor(_naca4_half_thickness(xc.numpy()), device=dev) * t_ratio * chord
        half_t = torch.clamp(half_t, min=0.5 * min_thickness_lu)
        inside = (xi.abs() <= 0.5 * chord + chord_pad_lu) & ((eta - camber).abs() <= half_t)
        mask_ann |= inside
        idx_ann = torch.where(inside & (idx_ann < 0), torch.full_like(idx_ann, k), idx_ann)

    blade_sub = torch.zeros_like(rr, dtype=torch.bool)
    blade_sub[annulus] = mask_ann
    idx_sub = torch.full_like(rr, -1, dtype=torch.int8)
    idx_sub[annulus] = idx_ann

    mask[z0:z1, y0:y1, x0:x1] |= blade_sub
    blade_index[z0:z1, y0:y1, x0:x1] = torch.where(
        blade_sub, idx_sub, blade_index[z0:z1, y0:y1, x0:x1]
    )
    # hub cells never carry a blade index
    sub_idx = blade_index[z0:z1, y0:y1, x0:x1]
    hub_over = torch.zeros_like(sub_idx, dtype=torch.bool)
    oy0, oy1 = max(yh0, y0), min(yh1, y1)
    oz0, oz1 = max(zh0, z0), min(zh1, z1)
    ox0, ox1 = max(xh0, x0), min(xh1, x1)
    if oy0 < oy1 and oz0 < oz1 and ox0 < ox1:
        hub_over[oz0 - z0:oz1 - z0, oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] = hub[
            oz0 - zh0:oz1 - zh0, oy0 - yh0:oy1 - yh0, ox0 - xh0:ox1 - xh0
        ].bool()
    blade_index[z0:z1, y0:y1, x0:x1] = torch.where(
        hub_over, torch.full_like(sub_idx, -1), sub_idx
    )
    return {"mask": mask, "blade_index": blade_index}


# ---------------------------------------------------------------------------
# 4.  Geometry gate (measurements taken from the voxel mask)
# ---------------------------------------------------------------------------


def _mask_cells(mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zz, yy, xx = torch.nonzero(mask, as_tuple=True)
    return zz.numpy(), yy.numpy(), xx.numpy()


def geometry_metrics(
    mask: torch.Tensor,
    blade_index: torch.Tensor,
    *,
    cx: float,
    cy: float,
    cz: float,
    diameter_lu: float,
    spec: E779ASpec = E779A_SPEC,
) -> dict[str, Any]:
    """Measure ``A_E/A_0``, ``P/D`` at ``0.7R`` and ``D_hub/D`` from a voxel mask.

    The pitch ratio is measured *independently of the input* by regressing the
    axial coordinate on the developed tangential coordinate of the blade cells
    inside a narrow radial band around ``r/R = 0.7``::

        x = x0 - tan(beta) * s   =>   P/D = 2 pi (r/R) tan(beta)

    The expanded area is estimated from the voxel volume: every blade cell
    contributes ``1 / t_cells`` of developed area, where ``t_cells`` is the
    local section thickness in cells.
    """
    R = 0.5 * diameter_lu
    zz, yy, xx = _mask_cells(mask)
    r = np.sqrt((yy - cy) ** 2 + (zz - cz) ** 2)
    r_frac = r / R
    bidx = blade_index.numpy()[zz, yy, xx]

    x_ax = xx - cx
    theta = np.arctan2(zz - cz, yy - cy)

    out: dict[str, Any] = {}

    # --- hub ratio: largest solid radius of the hub body -------------------
    hub_sel = bidx < 0
    blade_sel = bidx >= 0
    hub_r = r[hub_sel]
    hub_len_lu = float(spec.hub_length_ratio * diameter_lu)
    hub_area = float(hub_sel.sum()) / max(hub_len_lu, 1e-9)
    out["hub_diameter_ratio_measured"] = float(
        2.0 * math.sqrt(hub_area / math.pi) / diameter_lu
    ) if hub_sel.any() else 0.0
    out["hub_radius_max_cells"] = float(hub_r.max()) if hub_r.size else 0.0
    if not np.any(blade_sel):
        raise ValueError("mask contains no blade cells")

    rb = r_frac[blade_sel]
    xb = x_ax[blade_sel]
    thb = theta[blade_sel]
    m_cells = rb.size

    # --- pitch ratio at r/R = 0.7 from an independent mask regression ------
    band = (np.abs(rb - 0.7) < 0.03) & (bidx[blade_sel] == 0)
    if int(band.sum()) < 20:
        band = (np.abs(rb - 0.7) < 0.06) & (bidx[blade_sel] == 0)
    r_band = float(np.mean(rb[band] * R))
    th0 = float(np.mean(thb[band]))
    dth = (thb[band] - th0 + math.pi) % (2.0 * math.pi) - math.pi
    s_loc = rb[band] * R * dth
    slope = float(np.polyfit(s_loc, xb[band], 1)[0])
    beta_meas = math.atan(abs(slope))
    out["pitch_ratio_07_measured"] = float(
        2.0 * math.pi * (r_band / diameter_lu) * math.tan(beta_meas)
    )
    out["pitch_angle_07_deg_measured"] = math.degrees(beta_meas)
    out["pitch_band_r_frac"] = r_band / R
    out["pitch_band_cells"] = int(band.sum())

    # --- expanded area from voxel volume ----------------------------------
    chord_scale = _chord_scale(diameter_lu, spec)
    s_frac = (rb - spec.hub_radius_ratio) / (1.0 - spec.hub_radius_ratio)
    chord = _chord_shape(s_frac, spec) * chord_scale
    t_ratio = _thickness_ratio(rb, spec)
    t_cells = np.maximum(t_ratio * chord, 0.9)          # matches min_thickness_lu
    area_cells = float(np.sum(1.0 / t_cells))
    disk_area = math.pi * R**2
    out["expanded_area_ratio_measured"] = float(area_cells / disk_area)
    out["expanded_area_ratio_analytic"] = float(_mask_integral_area(diameter_lu, spec) / disk_area)

    # --- discretisation diagnostics ---------------------------------------
    out["blade_cells"] = int(m_cells)
    out["hub_cells"] = int(hub_sel.sum())
    out["mean_chord_lu_07"] = float(np.mean(chord[band])) if np.any(band) else 0.0
    out["min_thickness_lu"] = float(np.min(t_cells))

    # --- gate --------------------------------------------------------------
    def rel(meas: float, ref: float) -> float:
        return abs(meas - ref) / ref * 100.0

    out["gate"] = {
        "expanded_area_ratio": {
            "spec": spec.expanded_area_ratio,
            "measured": out["expanded_area_ratio_measured"],
            "error_pct": rel(out["expanded_area_ratio_measured"], spec.expanded_area_ratio),
            "tol_pct": 1.0,
        },
        "pitch_ratio_07": {
            "spec": spec.pitch_ratio_07,
            "measured": out["pitch_ratio_07_measured"],
            "error_pct": rel(out["pitch_ratio_07_measured"], spec.pitch_ratio_07),
            "tol_pct": 1.0,
        },
        "hub_diameter_ratio": {
            "spec": spec.hub_diameter_ratio,
            "measured": out["hub_diameter_ratio_measured"],
            "error_pct": rel(out["hub_diameter_ratio_measured"], spec.hub_diameter_ratio),
            "tol_pct": 1.0,
        },
    }
    out["gate"]["passed"] = all(v["error_pct"] <= v["tol_pct"] for v in out["gate"].values() if isinstance(v, dict))
    return out


def _mask_integral_area(diameter_lu: float, spec: E779ASpec, n: int = 20001) -> float:
    s = np.linspace(0.0, 1.0, n)
    R, R_hub = 0.5 * diameter_lu, 0.5 * diameter_lu * spec.hub_diameter_ratio
    r = R_hub + s * (R - R_hub)
    chord = _chord_shape(s, spec) * _chord_scale(diameter_lu, spec)
    return float(spec.n_blades * np.sum(chord * np.gradient(r)))


def geometry_gate(
    *,
    diameter_lu: float,
    spec: E779ASpec = E779A_SPEC,
    angles_deg: Sequence[float] = (0.0, 13.0, 27.0),
    nz: int | None = None,
    ny: int | None = None,
    nx: int | None = None,
) -> dict[str, Any]:
    """Run :func:`geometry_metrics` at several rotor phases and aggregate."""
    nx = nx or int(round(4.0 * diameter_lu))
    ny = ny or nz or int(round(3.0 * diameter_lu))
    nz = nz or ny
    cx, cy, cz = 0.35 * nx, 0.5 * ny, 0.5 * nz
    rows = []
    for ang in angles_deg:
        built = build_e779a_mask(
            nx, ny, nz, cx, cy, cz, diameter_lu=diameter_lu, spec=spec, angle_deg=ang
        )
        m = geometry_metrics(
            built["mask"], built["blade_index"], cx=cx, cy=cy, cz=cz,
            diameter_lu=diameter_lu, spec=spec,
        )
        m["angle_deg"] = ang
        rows.append(m)
    worst = {
        key: float(np.median([row["gate"][key]["error_pct"] for row in rows]))
        for key in ("expanded_area_ratio", "pitch_ratio_07", "hub_diameter_ratio")
    }
    worst_max = {
        key: max(row["gate"][key]["error_pct"] for row in rows)
        for key in ("expanded_area_ratio", "pitch_ratio_07", "hub_diameter_ratio")
    }
    return {
        "spec": {
            "diameter_lu": diameter_lu,
            "expanded_area_ratio": spec.expanded_area_ratio,
            "pitch_ratio_07": spec.pitch_ratio_07,
            "hub_diameter_ratio": spec.hub_diameter_ratio,
        },
        "points": rows,
        "worst_error_pct": worst,
        "worst_error_pct_max_over_phases": worst_max,
        "passed": all(v <= 1.0 for v in worst_max.values()),
    }

# ---------------------------------------------------------------------------
# 5.  Moving-boundary solver (lab frame, rigidly rotating mask)
# ---------------------------------------------------------------------------

_C19 = C.to(torch.float32)
_W19 = W.to(torch.float32)
_OPP = torch.tensor(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=torch.long
)
_CU = C.to(torch.float64)


@dataclass
class OpenWaterConfig:
    """Discretisation/run control for one E779A open-water point.

    The rotor advance rate is prescribed through the **tip Mach number**, which
    fixes the number of lattice steps per revolution
    ``steps_per_rev = pi D / (Ma_tip c_s)``; e.g. ``D = 32``, ``Ma_tip = 0.18``
    gives 967 steps/rev, i.e. a rotor phase increment of 0.37 deg/step (the
    campaign contract asks for ``<= 2`` deg/step).
    """

    diameter_lu: int = 32
    axial_diameters: float = 4.0
    lateral_diameters: float = 3.0
    tip_mach: float = 0.18
    tau: float = 0.5008
    smagorinsky: float = 0.0
    warmup_revs: float = 4.0
    avg_revs: float = 2.0
    min_thickness_lu: float = 0.9
    steps_per_rev_max: int = 4000
    save_fields: bool = False       # write the (rho, u) snapshot to .npz
    inception: bool = True          # run the lambda_2 tip-vortex inception analysis in-process
    strut_diameter_ratio: float = 0.0   # >0: static wake screen upstream (blade-rate excitation)
    strut_offset_r: float = 0.72        # radial position of the strut axis [R]
    strut_gap_d: float = 1.0            # upstream distance from rotor plane [D]
    rotation_sign: float = 1.0          # +1: omega about +x (right-handed rotor)

    def grid(self) -> tuple[int, int, int]:
        D = self.diameter_lu
        return (
            int(round(self.axial_diameters * D)),
            int(round(self.lateral_diameters * D)),
            int(round(self.lateral_diameters * D)),
        )

    def lattice_rate(self) -> float:
        """Rotation rate [rev/step] set by the tip Mach number."""
        return self.tip_mach * CS / (math.pi * self.diameter_lu)

    def steps_per_rev(self) -> float:
        return min(1.0 / self.lattice_rate(), float(self.steps_per_rev_max))

    def effective_rate(self) -> float:
        """Rotation rate actually used [rev/step] (after the step cap)."""
        return 1.0 / self.steps_per_rev()

    def reynolds_diameter(self, j: float) -> float:
        """Lattice ``Re_D = (n D) D / nu`` at this discretisation."""
        nu = (self.tau - 0.5) / 3.0
        return self.lattice_rate() * self.diameter_lu**2 / nu


def _macro(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = f.sum(dim=0)
    inv = 1.0 / rho.clamp(min=1e-12)
    ux = (f * _C19[:, 0].view(19, 1, 1, 1)).sum(dim=0) * inv
    uy = (f * _C19[:, 1].view(19, 1, 1, 1)).sum(dim=0) * inv
    uz = (f * _C19[:, 2].view(19, 1, 1, 1)).sum(dim=0) * inv
    return rho, ux, uy, uz


def _equilibrium_plane(rho: torch.Tensor, ux, uy, uz) -> torch.Tensor:
    cu = (
        _C19[:, 0].view(19, 1, 1) * ux.unsqueeze(0)
        + _C19[:, 1].view(19, 1, 1) * uy.unsqueeze(0)
        + _C19[:, 2].view(19, 1, 1) * uz.unsqueeze(0)
    )
    usq = ux * ux + uy * uy + uz * uz
    return _W19.view(19, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq.unsqueeze(0)
    )


def _masked_moving_wall(
    f: torch.Tensor,
    mask: torch.Tensor,
    ux_w: torch.Tensor,
    uy_w: torch.Tensor,
    uz_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ladd moving-wall update restricted to solid cells, returning impulses.

    Applies exactly ``f_q(x) <- f_qbar(x) + 2 w_q rho (c_q . u_w) / cs^2`` on
    ``mask`` (identical to
    :func:`tensorlbm.propeller_benchmark.moving_wall_bounce_back_3d_with_reaction`)
    but only touches the masked columns, and returns the per-cell fluid-impulse
    matrix ``(N, 3)`` and the ``(N, 3)`` cell index array.
    """
    idx = mask.nonzero(as_tuple=False)
    cols = f[:, mask]
    rho = cols.sum(dim=0)
    uw = torch.stack([ux_w[mask], uy_w[mask], uz_w[mask]], dim=1)   # (N, 3)
    cu = _C19.to(cols) @ uw.transpose(0, 1)                          # (19, N)
    corr = 2.0 * _W19.to(cols).view(19, 1) * rho.unsqueeze(0) * cu * 3.0
    new = cols[_OPP.to(cols.device)] + corr
    f[:, mask] = new
    impulse = (new - cols).transpose(0, 1) @ _C19.to(cols)            # (N, 3)
    return impulse, idx, cols


def _open_water_bc(
    f: torch.Tensor,
    u_in: float,
    wall_mask: torch.Tensor,
    profile_deg: float | None = None,
    wake_deficit: float = 0.0,
    wake_width_deg: float = 25.0,
) -> None:
    """Inflow / outflow / lateral-wall treatment for the open-water box.

    Inlet (``x = 0``): equilibrium with the prescribed axial profile.
    Outlet (``x = nx-1``): zero axial gradient of the populations.
    Lateral faces: no-slip (moving-wall operator with ``u_w = 0``).
    """
    nz, ny, nx = f.shape[1:]
    ux_plane = torch.full((nz, ny), float(u_in), dtype=f.dtype, device=f.device)
    if profile_deg is not None and wake_deficit > 0.0:
        zz, yy = torch.meshgrid(
            torch.arange(nz, dtype=f.dtype, device=f.device),
            torch.arange(ny, dtype=f.dtype, device=f.device),
            indexing="ij",
        )
        cy, cz = 0.5 * (ny - 1), 0.5 * (nz - 1)
        theta = torch.atan2(zz - cz, yy - cy)
        d = torch.atan2(
            torch.sin(theta - math.radians(profile_deg)),
            torch.cos(theta - math.radians(profile_deg)),
        )
        width = math.radians(wake_width_deg)
        ux_plane = ux_plane * (1.0 - wake_deficit * torch.exp(-(d / width) ** 2))
    rho_plane = torch.ones_like(ux_plane)
    zero = torch.zeros_like(ux_plane)
    f[:, :, :, 0] = _equilibrium_plane(rho_plane, ux_plane, zero, zero)
    f[:, :, :, -1] = f[:, :, :, -2]
    zero_full = torch.zeros_like(wall_mask, dtype=f.dtype)
    _masked_moving_wall(f, wall_mask, zero_full, zero_full, zero_full)


def _wall_mask(nz: int, ny: int, nx: int, device) -> torch.Tensor:
    w = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    w[:, 0, :] = True
    w[:, -1, :] = True
    w[0, :, :] = True
    w[-1, :, :] = True
    return w


def _strut_mask(
    nx: int, ny: int, nz: int, cx: float, cy: float, cz: float,
    diameter_lu: float, cfg: OpenWaterConfig,
) -> torch.Tensor:
    """Static cylindrical wake screen upstream of the rotor (optional)."""
    m = torch.zeros((nz, ny, nx), dtype=torch.bool)
    if cfg.strut_diameter_ratio <= 0.0:
        return m
    R = 0.5 * diameter_lu
    d = cfg.strut_diameter_ratio * diameter_lu
    # strut axis at (y,z) offset, radius r = strut_offset_r * R, azimuth 90 deg
    xs = cx - cfg.strut_gap_d * diameter_lu
    ys = cy + cfg.strut_offset_r * R
    x0, x1 = max(0, int(xs - 2)), min(nx, int(xs + 3))
    y0, y1 = max(0, int(ys - d)), min(ny, int(ys + d) + 2)
    z0, z1 = 0, nz
    yy = torch.arange(y0, y1, dtype=torch.float32).view(1, -1, 1)
    zz = torch.arange(z0, z1, dtype=torch.float32).view(-1, 1, 1)
    rad = torch.sqrt((yy - ys) ** 2 + (zz - cz) ** 2)
    m[z0:z1, y0:y1, x0:x1] = (rad <= 0.5 * d).expand(z1 - z0, y1 - y0, x1 - x0)
    return m


def _block_mean_residual(history: np.ndarray, steps_per_rev: float, avg_revs: float) -> dict[str, float]:
    """Relative drift of successive per-revolution block means over the window."""
    n = history.size
    n_avg = int(round(avg_revs * steps_per_rev))
    if n_avg <= 0 or n_avg > n:
        n_avg = n
    window = history[-n_avg:]
    n_rev = max(int(round(steps_per_rev)), 1)
    blocks = [window[i:i + n_rev] for i in range(0, window.size - n_rev + 1, n_rev)]
    means = np.array([float(np.mean(b)) for b in blocks]) if blocks else np.array([float(np.mean(window))])
    ref = float(np.mean(window))
    drift = float(np.max(np.abs(means - ref)) / max(abs(ref), 1e-30)) if means.size else 0.0
    step_drift = 0.0
    if means.size >= 2:
        step_drift = float(abs(means[-1] - means[-2]) / max(abs(means[-1]), 1e-30))
    return {
        "n_blocks": int(means.size),
        "block_means": means.tolist(),
        "mean": ref,
        "max_block_deviation": drift,
        "last_two_block_drift": step_drift,
    }


def run_openwater_point(
    j: float,
    cfg: OpenWaterConfig = OpenWaterConfig(),
    spec: E779ASpec = E779A_SPEC,
    *,
    seed_mean_thrust: float = 0.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one open-water point ``J`` and return coefficients + diagnostics.

    Load bookkeeping (Ladd momentum exchange on the *blade* cells only; the hub
    and the optional wake screen are reported separately) gives

    .. math:: K_T = \\frac{T}{\\rho n^2 D^4},\\qquad
              K_Q = \\frac{Q}{\\rho n^2 D^5},\\qquad
              \\eta_0 = \\frac{J K_T}{2 \\pi K_Q}

    with lattice ``rho = 1`` and ``n`` in rev/step.
    """
    D = float(cfg.diameter_lu)
    nx, ny, nz = cfg.grid()
    cx, cy, cz = 0.35 * nx, 0.5 * (ny - 1), 0.5 * (nz - 1)
    rate = cfg.effective_rate()                 # rev/step
    omega = 2.0 * math.pi * rate * cfg.rotation_sign   # rad/step
    steps_per_rev = cfg.steps_per_rev()
    u_in = j * rate * D
    dtheta_deg = 360.0 * rate

    mask_info = build_e779a_mask(
        nx, ny, nz, cx, cy, cz, diameter_lu=D, spec=spec, angle_deg=0.0,
        min_thickness_lu=cfg.min_thickness_lu,
    )
    blade_index0 = mask_info["blade_index"]
    strut = _strut_mask(nx, ny, nz, cx, cy, cz, D, cfg)
    wall = _wall_mask(nz, ny, nx, torch.device("cpu"))

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32)
    f = equilibrium3d(
        rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0), torch.zeros_like(rho0)
    )

    n_warm = int(round(cfg.warmup_revs * steps_per_rev))
    n_avg = int(round(cfg.avg_revs * steps_per_rev))
    n_total = n_warm + n_avg

    hist_mx = np.zeros(n_total, dtype=np.float64)
    hist_tq = np.zeros(n_total, dtype=np.float64)
    hist_blade = np.zeros((n_total, spec.n_blades), dtype=np.float64)
    hist_hub = np.zeros(n_total, dtype=np.float64)
    hist_strut = np.zeros(n_total, dtype=np.float64)

    for step in range(n_total):
        ang = math.degrees((step * omega) % (2.0 * math.pi))
        built = build_e779a_mask(
            nx, ny, nz, cx, cy, cz, diameter_lu=D, spec=spec, angle_deg=ang,
            min_thickness_lu=cfg.min_thickness_lu,
        )
        mask = built["mask"] | strut
        bidx = built["blade_index"]
        if cfg.smagorinsky > 0.0:
            from .turbulence import collide_smagorinsky_mrt3d

            f = collide_smagorinsky_mrt3d(f, tau=cfg.tau, C_s=cfg.smagorinsky)
        else:
            f = collide_mrt3d_fused(f, tau=cfg.tau)
        f = stream3d(f)
        _open_water_bc(f, u_in, wall)

        ux_w, uy_w, uz_w = _wall_velocity(bidx, cx, cy, cz, omega, mask)
        impulse, idx, _ = _masked_moving_wall(f, mask, ux_w, uy_w, uz_w)

        cell_b = bidx[idx[:, 0], idx[:, 1], idx[:, 2]].to(torch.int64)
        imp64 = impulse.to(torch.float64)
        fsum = imp64.sum(dim=0)
        dy = (idx[:, 1].to(torch.float64) - cy)
        dz = (idx[:, 2].to(torch.float64) - cz)
        cell_tq = dy * imp64[:, 2] - dz * imp64[:, 1]
        is_blade = cell_b >= 0
        blade_sel = [is_blade & (cell_b == k) for k in range(spec.n_blades)]
        # body force = -fluid impulse; report the reaction (force on the body)
        # body reaction (force/torque on the body) = - fluid impulse; the
        # *propeller* torque is the torque the rotor exerts on the fluid,
        # Q_fluid = -Q_body, which is the quantity that enters K_Q and the
        # shaft power P = 2 pi n Q_fluid.
        hist_mx[step] = -float(fsum[0])
        hist_tq[step] = float(cell_tq.sum())
        for k in range(spec.n_blades):
            hist_blade[step, k] = -float(imp64[blade_sel[k], 0].sum())
        hub_sel = (~is_blade) & (~strut[idx[:, 0], idx[:, 1], idx[:, 2]])
        hist_hub[step] = -float(imp64[hub_sel, 0].sum()) if hub_sel.any() else 0.0
        if cfg.strut_diameter_ratio > 0.0:
            st_sel = strut[idx[:, 0], idx[:, 1], idx[:, 2]]
            hist_strut[step] = -float(imp64[st_sel, 0].sum()) if st_sel.any() else 0.0
        if verbose and step % max(int(steps_per_rev), 1) == 0:
            print(f"step {step}/{n_total} rev={step * rate:.2f} Fx={hist_mx[step]:+.5f}", flush=True)

    win = slice(n_warm, n_total)
    n2d4 = rate**2 * D**4
    n2d5 = rate**2 * D**5
    blade_mean = hist_blade[win].mean(axis=0)
    thrust_body = float(hist_mx[win].mean())          # reaction on body (blades+hub+strut)
    thrust_blades = float(blade_mean.sum())
    torque = float(hist_tq[win].mean())   # Q_fluid (rotor -> fluid)

    k_t = thrust_blades / n2d4
    k_t_total = thrust_body / n2d4
    k_q = torque / n2d5
    eta = j * k_t / (2.0 * math.pi * k_q) if abs(k_q) > 1e-30 else float("nan")

    resid = _block_mean_residual(hist_mx[win], steps_per_rev, cfg.avg_revs)
    resid_tq = _block_mean_residual(hist_tq[win], steps_per_rev, cfg.avg_revs)

    win_blade = hist_blade[win]
    try:
        spec_total = blade_thrust_spectrum(win_blade.sum(axis=1), rate_per_step=rate, n_blades=spec.n_blades)
        spec_blade = blade_thrust_spectrum(win_blade[:, 0], rate_per_step=rate, n_blades=spec.n_blades)
        spectrum_block = {
            "total": {k: spec_total[k] for k in (
                "harmonics_bpf", "harmonics_shaft", "amplitude_ratio_2nd_1st",
                "dominant_peak_hz_physical", "bpf_hz_physical", "bpf_cycles_per_step", "df_cycles_per_step")},
            "per_blade": {k: spec_blade[k] for k in (
                "harmonics_bpf", "harmonics_shaft", "amplitude_ratio_2nd_1st",
                "dominant_peak_hz_physical")},
        }
    except Exception as exc:  # pragma: no cover - diagnostics must never kill a run
        spectrum_block = {"error": str(exc)}

    out: dict[str, Any] = {
        "j": float(j),
        "spectrum": spectrum_block,
        "reynolds_diameter_lattice": cfg.reynolds_diameter(j),
        "reynolds_diameter_physical": spec.reynolds_diameter(j),
        "u_in_lattice": u_in,
        "rate_per_step": rate,
        "steps_per_rev": steps_per_rev,
        "phase_increment_deg": dtheta_deg,
        "tip_mach": cfg.tip_mach,
        "tau": cfg.tau,
        "nu_lattice": (cfg.tau - 0.5) / 3.0,
        "grid": [nx, ny, nz],
        "diameter_lu": D,
        "warmup_revs": cfg.warmup_revs,
        "avg_revs": cfg.avg_revs,
        "kT": k_t,
        "kQ": k_q,
        "eta": eta,
        "kT_total_body": k_t_total,
        "thrust_blades": thrust_blades,
        "thrust_hub": float(hist_hub[win].mean()),
        "thrust_strut": float(hist_strut[win].mean()),
        "blade_thrust_mean": blade_mean.tolist(),
        "blade_thrust_spread_pct": float(
            (blade_mean.max() - blade_mean.min()) / max(abs(blade_mean.mean()), 1e-30) * 100.0
        ),
        "torque_mean": torque,
        "convergence": {"thrust": resid, "torque": resid_tq},
        "history": {
            "thrust_body": hist_mx.tolist(),
            "torque": hist_tq.tolist(),
            "blade_thrust": hist_blade.tolist(),
        },
        "config": {
            "diameter_lu": cfg.diameter_lu,
            "axial_diameters": cfg.axial_diameters,
            "lateral_diameters": cfg.lateral_diameters,
            "tip_mach": cfg.tip_mach,
            "tau": cfg.tau,
            "smagorinsky": cfg.smagorinsky,
            "strut_diameter_ratio": cfg.strut_diameter_ratio,
        },
    }
    if cfg.save_fields or cfg.inception:
        rho, ux, uy, uz = _macro(f)
        fld = {
            "rho": rho.numpy().astype(np.float32),
            "ux": ux.numpy().astype(np.float32),
            "uy": uy.numpy().astype(np.float32),
            "uz": uz.numpy().astype(np.float32),
            "origin": [cx, cy, cz],
            "diameter_lu": D,
        }
        if cfg.inception:
            out["inception"] = tip_vortex_inception(fld, j=j, spec=spec, cfg=cfg)
        if cfg.save_fields:
            out["fields"] = fld
    out["_mask"] = built["mask"]
    return out


def _wall_velocity(blade_index, cx, cy, cz, omega, mask):
    nz, ny, nx = mask.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    ux_w = torch.zeros_like(xx)
    uy_w = -omega * (zz - cz)
    uz_w = omega * (yy - cy)
    return ux_w, uy_w, uz_w


# ---------------------------------------------------------------------------
# 6.  Signal analysis: blade-rate spectrum
# ---------------------------------------------------------------------------


def blade_thrust_spectrum(
    series: Sequence[float] | np.ndarray,
    *,
    rate_per_step: float,
    n_blades: int = 4,
    steps_per_rev: float | None = None,
    n_harmonics: int = 8,
    window: str = "hann",
) -> dict[str, Any]:
    """Amplitude spectrum of a (per-blade) thrust history.

    ``rate_per_step`` is the shaft rate in rev/step, so the shaft frequency is
    ``rate_per_step`` cycles/step and the blade-passing frequency (BPF) is
    ``n_blades * rate_per_step``.  The spectrum is reported both in
    cycles/step and in Hz, the latter using the *physical* shaft rate
    (``n = 25`` rps), which is the campaign's ``4 n = 100`` Hz blade rate.

    Returns the spectrum, the harmonic amplitudes at ``k * BPF`` (``k = 1, 2``),
    their ratio and the peak-position errors against the exact multiples.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim == 2:
        x = x.sum(axis=1)
    n = x.size
    x = x - x.mean()
    if window == "hann":
        w = np.hanning(n)
        x = x * w
        norm = np.sum(w) / 2.0
    else:
        norm = n / 2.0
    spec = np.abs(np.fft.rfft(x)) / max(norm, 1e-30)
    freqs = np.fft.rfftfreq(n, d=1.0)             # cycles/step
    df = freqs[1] - freqs[0] if freqs.size > 1 else 1.0

    shaft = rate_per_step
    bpf = n_blades * shaft

    def _peak_near(f0: float, tol_bins: int = 3) -> dict[str, float]:
        if f0 <= 0 or f0 / df > freqs.size - 1:
            return {"freq": float("nan"), "amp": float("nan"), "error_pct": float("nan")}
        i0 = int(round(f0 / df))
        lo, hi = max(0, i0 - tol_bins), min(freqs.size, i0 + tol_bins + 1)
        i = lo + int(np.argmax(spec[lo:hi]))
        return {
            "freq_cycles_per_step": float(freqs[i]),
            "freq_hz_physical": float(freqs[i] / rate_per_step * 25.0),
            "amp": float(spec[i]),
            "error_pct": float(abs(freqs[i] - f0) / f0 * 100.0),
        }

    harmonics = {f"k{k}": _peak_near(k * bpf, tol_bins=3) for k in (1, 2)}
    # shaft harmonics for reference
    shaft_harmonics = {f"k{k}": _peak_near(k * shaft, tol_bins=2) for k in range(1, min(n_harmonics, 8) + 1)}

    a1 = harmonics["k1"]["amp"]
    a2 = harmonics["k2"]["amp"]
    ratio = float(a2 / a1) if a1 and np.isfinite(a1) and a1 > 0 else float("nan")

    # dominant spectral peak anywhere (diagnostic: where does the energy sit?)
    i_dom = int(np.argmax(spec[1:])) + 1 if spec.size > 1 else 0

    return {
        "n_samples": int(n),
        "rate_per_step": float(rate_per_step),
        "bpf_cycles_per_step": float(bpf),
        "bpf_hz_physical": float(n_blades * 25.0),
        "df_cycles_per_step": float(df),
        "freqs": freqs.tolist(),
        "amplitude": spec.tolist(),
        "harmonics_bpf": harmonics,
        "harmonics_shaft": shaft_harmonics,
        "amplitude_ratio_2nd_1st": ratio,
        "dominant_peak_hz_physical": float(freqs[i_dom] / rate_per_step * 25.0),
        "dominant_peak_cycles_per_step": float(freqs[i_dom]),
    }


# ---------------------------------------------------------------------------
# 7.  Vortex identification and cavitation inception
# ---------------------------------------------------------------------------


def vortex_criteria_fields(
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, h: float = 1.0
) -> dict[str, np.ndarray]:
    """``Q`` and ``lambda_2`` fields from a velocity snapshot.

    ``Q = 0.5 (||Omega||^2 - ||S||^2)`` and ``lambda_2`` is the second largest
    eigenvalue of ``S^2 + Omega^2`` (vortex where ``lambda_2 < 0``).
    Gradients are central differences with spacing ``h`` (``np.gradient``);
    one-cell boundary layers are therefore invalid and are masked out by the
    caller.
    """
    dux = np.gradient(ux, h, axis=(0, 1, 2))
    duy = np.gradient(uy, h, axis=(0, 1, 2))
    duz = np.gradient(uz, h, axis=(0, 1, 2))
    # index order of the snapshot is (z, y, x) -> derivatives (d/dz, d/dy, d/dx)
    ux_z, ux_y, ux_x = dux
    uy_z, uy_y, uy_x = duy
    uz_z, uz_y, uz_x = duz

    S = np.empty(ux.shape + (3, 3), dtype=np.float32)
    O = np.empty(ux.shape + (3, 3), dtype=np.float32)
    # velocity components (u, v, w) = (ux, uy, uz); coordinates (x, y, z)
    grad = np.stack(
        [
            np.stack([ux_x, ux_y, ux_z], axis=-1),
            np.stack([uy_x, uy_y, uy_z], axis=-1),
            np.stack([uz_x, uz_y, uz_z], axis=-1),
        ],
        axis=-2,
    )  # grad[i, j] = du_i / dx_j
    S = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    O = 0.5 * (grad - np.swapaxes(grad, -1, -2))
    Q = 0.5 * (np.sum(O * O, axis=(-1, -2)) - np.sum(S * S, axis=(-1, -2)))

    M = S @ S + O @ O
    lam = np.linalg.eigvalsh(M.astype(np.float64))
    lambda2 = lam[..., 1].astype(np.float32)
    return {"Q": Q.astype(np.float32), "lambda2": lambda2, "S": S, "Omega": O}


def tip_vortex_inception(
    fields: dict[str, Any],
    *,
    j: float,
    spec: E779ASpec = E779A_SPEC,
    cfg: OpenWaterConfig | None = None,
    r_band: tuple[float, float] = (0.85, 1.30),
    x_band: tuple[float, float] = (0.08, 1.20),
    min_lambda2_cells: int = 4,
) -> dict[str, Any]:
    """Tip-vortex core pressure and the resulting inception number.

    The tip vortex is located by the ``lambda_2 < 0`` criterion inside the
    annulus ``r/R in r_band`` downstream of the rotor (``x/D in x_band``); the
    core pressure is the *minimum* pressure inside that vortex region, and the
    inception number is

    .. math:: \\sigma_i = \\frac{p_\\infty - p_{min}}{\\frac12 \\rho V_R^2},
        \\qquad V_R = \\sqrt{V_A^2 + (2 \\pi n R)^2}

    evaluated in lattice units (the ratio is scale invariant).  ``p_inf`` is
    taken from the upstream reference plane ``x/D < -0.8``.

    This is a **pressure-field inception criterion only** — no bubble dynamics
    and no sheet/cloud cavity morphology (explicitly NOT_SUPPORTED).
    """
    rho = np.asarray(fields["rho"], dtype=np.float32)
    ux = np.asarray(fields["ux"], dtype=np.float32)
    uy = np.asarray(fields["uy"], dtype=np.float32)
    uz = np.asarray(fields["uz"], dtype=np.float32)
    cx, cy, cz = fields["origin"]
    D = float(fields["diameter_lu"])
    R = 0.5 * D
    nz, ny, nx = rho.shape
    p = rho / 3.0  # lattice equation of state

    zz, yy, xx = np.meshgrid(
        np.arange(nz) - cz, np.arange(ny) - cy, np.arange(nx) - cx, indexing="ij"
    )
    r = np.sqrt(yy**2 + zz**2)

    inner = 2
    slab = (slice(inner, nz - inner), slice(inner, ny - inner), slice(inner, nx - inner))
    vc = vortex_criteria_fields(
        ux[slab], uy[slab], uz[slab]
    )
    lam2 = vc["lambda2"]
    Qv = vc["Q"]
    p_in = p[slab]
    r_in = r[slab]
    x_in = xx[slab]

    ref_plane = x_in < -0.8 * D
    p_ref = float(np.median(p_in[ref_plane])) if np.any(ref_plane) else float(np.median(p_in))
    u_ref = float(np.median(ux[slab][ref_plane])) if np.any(ref_plane) else 0.0

    region = (
        (r_in > r_band[0] * R)
        & (r_in < r_band[1] * R)
        & (x_in > x_band[0] * D)
        & (x_in < x_band[1] * D)
    )
    core = region & (lam2 < 0.0)
    n_core = int(core.sum())
    out: dict[str, Any] = {
        "j": float(j),
        "n_core_cells": n_core,
        "p_ref_lattice": p_ref,
        "u_ref_lattice": u_ref,
        "inception_available": n_core >= min_lambda2_cells,
        "criterion": "lambda2<0 tip-vortex core minimum pressure (no bubble dynamics)",
    }
    if n_core >= min_lambda2_cells:
        # keep the strongest vortex cells (most negative lambda_2) for the core
        thr = np.quantile(lam2[core], 0.10)
        strongest = core & (lam2 <= thr)
        p_core_min = float(p_in[strongest].min())
        # V_R at the tip (and at 0.7R for reference), lattice units
        rate = (cfg.effective_rate() if cfg else 0.0)
        omega = 2.0 * math.pi * rate
        v_r_tip = float(np.hypot(u_ref, omega * R))
        v_r_07 = float(np.hypot(u_ref, omega * 0.7 * R))
        dp = p_ref - p_core_min
        out.update(
            {
                "p_core_min_lattice": p_core_min,
                "dp_lattice": dp,
                "v_r_tip_lattice": v_r_tip,
                "v_r_07_lattice": v_r_07,
                "sigma_i_tip": float(dp / max(0.5 * v_r_tip**2, 1e-30)),
                "sigma_i_07": float(dp / max(0.5 * v_r_07**2, 1e-30)),
                "core_radius_cells": float(
                    np.sqrt(n_core / math.pi / max(1, int(np.unique(np.round(x_in[core] / 2)).size)))
                ),
                "min_pressure_location": [
                    float(x_in[strongest][np.argmin(p_in[strongest])]),
                    float(r_in[strongest][np.argmin(p_in[strongest])]) / R,
                ],
                "q_max_core": float(Qv[strongest].max()),
                "lambda2_min": float(lam2[strongest].min()),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 8.  Grid convergence (Roache GCI)
# ---------------------------------------------------------------------------


def gci_roache(
    values: Sequence[float],
    grids: Sequence[float],
    *,
    safety: float = 1.25,
    observed_order_floor: float = 0.5,
) -> dict[str, Any]:
    """Roache grid-convergence index for three successive grids.

    ``values`` and ``grids`` are ordered **coarse -> medium -> fine**; ``grids``
    may hold the cell size or the resolution (only the ratio matters).  With
    ``eps_cm = f_m - f_c`` and ``eps_mf = f_f - f_m`` the apparent order is

    .. math:: p = \ln|\epsilon_{cm}/\epsilon_{mf}| / \ln r,\quad
              r = h_m/h_f,\qquad
              GCI_{fine} = \frac{F_s\,|\epsilon_{mf}/f_f|}{r^p - 1}

    (Celik et al. 2008, equal refinement ratios).  ``p`` is floored at
    ``observed_order_floor`` before evaluating the GCI so that a non-monotone
    triple still returns a finite, conservative number instead of dividing by
    zero.
    """
    if len(values) != 3 or len(grids) != 3:
        raise ValueError("gci_roache needs exactly three grids")
    f_c, f_m, f_f = (float(v) for v in values)
    h_c, h_m, h_f = (float(g) for g in grids)
    r = h_m / h_f
    eps_cm = f_m - f_c
    eps_mf = f_f - f_m
    monotone = bool(abs(eps_cm) > 0 and abs(eps_mf) > 0 and (eps_cm * eps_mf) > 0)
    p = float("nan")
    if abs(eps_mf) > 1e-30 and r > 1.0:
        ratio = abs(eps_cm / eps_mf)
        if ratio > 0 and np.isfinite(ratio):
            p = math.log(ratio) / math.log(r)
    p_used = max(p, observed_order_floor) if np.isfinite(p) else observed_order_floor
    denom = max(r**p_used - 1.0, 1e-12)
    gci_fine = safety * abs(eps_mf / f_f) / denom if abs(f_f) > 1e-30 else float("inf")
    r_cm = h_c / h_m
    gci_medium = (
        safety * abs(eps_cm / f_m) / max(r_cm**p_used - 1.0, 1e-12) if abs(f_m) > 1e-30 else float("inf")
    )
    return {
        "values_coarse_medium_fine": [f_c, f_m, f_f],
        "grids_coarse_medium_fine": [h_c, h_m, h_f],
        "refinement_ratio": r,
        "apparent_order_p": float(p) if np.isfinite(p) else None,
        "apparent_order_used": float(p_used),
        "extrapolated_fine": float(f_f + eps_mf / max(r**p_used - 1.0, 1e-12)),
        "gci_fine": float(gci_fine),
        "gci_medium": float(gci_medium),
        "monotone_convergence": monotone,
    }


# ---------------------------------------------------------------------------
# 9.  Independent (MODEL-level) blade-element momentum reference
# ---------------------------------------------------------------------------


def _prandtl_loss(x_frac: np.ndarray, hub_frac: float, n_blades: int, phi: np.ndarray) -> np.ndarray:
    """Prandtl tip + hub loss factor for a propeller annulus."""
    def _loss(x_edge: np.ndarray) -> np.ndarray:
        arg = np.clip(
            0.5 * n_blades * (x_edge - x_frac) / np.maximum(x_frac * np.abs(np.sin(phi)), 1e-6),
            -20.0, 20.0,
        )
        return 2.0 / math.pi * np.arccos(np.clip(np.exp(-np.abs(arg)), 0.0, 1.0))

    tip = _loss(np.ones_like(x_frac))
    hub = _loss(np.full_like(x_frac, hub_frac))
    return np.clip(tip * hub, 0.05, 1.0)


def bemt_reference(
    spec: E779ASpec = E779A_SPEC,
    js: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    *,
    cd0: float = 0.008,
    cl_slope: float = 2.0 * math.pi,
    drag_quadratic: float = 0.010,
    cl_max: float = 1.45,
    stall_alpha_deg: float = 12.0,
    n_sections: int = 80,
    n_iter: int = 600,
    relax: float = 0.15,
) -> dict[str, Any]:
    """Blade-element / momentum-theory open-water reference (MODEL level).

    Uses the *same* radial chord and pitch laws as the voxel geometry, a
    cambered-section lift curve ``C_l = cl_slope (alpha + 2 f/c)`` with a
    stall plateau, a quadratic drag polar ``C_d = cd0 + k C_l^2`` and Prandtl
    tip/hub losses.  Axial/tangential induction factors are solved by damped
    fixed-point iteration

    .. math:: a = \frac{\sigma C_n}{4 F \sin^2\phi},\qquad
              a' = \frac{\sigma C_t}{4 F \sin\phi\cos\phi}

    This is an **independent method** (never an experiment): it is calibrated
    against the KP505 open-water table already in the repository
    (Fujisawa et al. 2000) and is then used to attribute the lattice results'
    Reynolds-number offset, ``cd0`` / ``cl_slope`` being the knobs that
    represent high-Re versus lattice-Re section behaviour.
    """
    D = spec.diameter_m
    R = 0.5 * D
    r_hub = spec.hub_radius_ratio * D
    chord_scale = _chord_scale(D, spec)
    xs = np.linspace(r_hub + 0.02 * (R - r_hub), 0.995 * R, n_sections)
    x_frac = xs / R
    s_frac = (xs - r_hub) / (R - r_hub)
    chord = _chord_shape(s_frac, spec) * chord_scale
    beta = np.arctan(spec.pitch_ratio_07 * D / (2.0 * math.pi * xs))
    camber_offset = 2.0 * spec.camber_ratio
    stall = math.radians(stall_alpha_deg)

    rows = []
    for j in js:
        v_a = spec.advance_speed(j)
        omega = 2.0 * math.pi * spec.n_rps
        a = np.full_like(xs, 0.10)
        a_p = np.full_like(xs, 0.02)
        for _ in range(n_iter):
            phi = np.arctan2(v_a * (1.0 + a), omega * xs * (1.0 - a_p))
            alpha = beta - phi
            cl = cl_slope * (alpha + camber_offset)
            cl = np.clip(cl, -cl_max, cl_max)
            # smooth stall: cap the incidence used in the lift law
            alpha_eff = np.clip(alpha + camber_offset, -stall, stall) - camber_offset
            cl = np.where(np.abs(alpha) > stall, cl_slope * (np.sign(alpha) * stall + camber_offset), cl)
            cd = cd0 + drag_quadratic * cl**2
            F = _prandtl_loss(x_frac, spec.hub_radius_ratio, spec.n_blades, phi)
            sigma = spec.n_blades * chord / (2.0 * math.pi * xs)
            cn = cl * np.cos(phi) - cd * np.sin(phi)
            ct = cl * np.sin(phi) - cd * np.cos(phi)
            denom_a = 4.0 * F * np.sin(phi) ** 2
            denom_ap = 4.0 * F * np.sin(phi) * np.cos(phi)
            with np.errstate(divide="ignore", invalid="ignore"):
                a_new = np.where(denom_a > 1e-8, sigma * cn / np.maximum(denom_a, 1e-8), a)
                ap_new = np.where(np.abs(denom_ap) > 1e-8, sigma * ct / np.where(
                    np.abs(denom_ap) > 1e-8, denom_ap, 1.0), a_p)
            a_new = np.clip(a_new, -0.5, 0.6)
            ap_new = np.clip(ap_new, -0.2, 0.3)
            a = a + relax * (a_new - a)
            a_p = a_p + relax * (ap_new - a_p)
        phi = np.arctan2(v_a * (1.0 + a), omega * xs * (1.0 - a_p))
        alpha = beta - phi
        cl = cl_slope * (alpha + camber_offset)
        cl = np.where(np.abs(alpha) > stall, cl_slope * (np.sign(alpha) * stall + camber_offset), cl)
        cl = np.clip(cl, -cl_max, cl_max)
        cd = cd0 + drag_quadratic * cl**2
        F = _prandtl_loss(x_frac, spec.hub_radius_ratio, spec.n_blades, phi)
        sigma = spec.n_blades * chord / (2.0 * math.pi * xs)
        cn = cl * np.cos(phi) - cd * np.sin(phi)
        ct = cl * np.sin(phi) - cd * np.cos(phi)
        w = v_a * (1.0 + a) / np.maximum(np.sin(phi), 1e-6)
        dT = 0.5 * spec.rho * w**2 * spec.n_blades * chord * cn * F
        dQ = 0.5 * spec.rho * w**2 * spec.n_blades * chord * ct * F * xs
        T = float(np.trapezoid(dT, xs))
        Q = float(np.trapezoid(dQ, xs))
        kt = T / (spec.rho * spec.n_rps**2 * D**4)
        kq = Q / (spec.rho * spec.n_rps**2 * D**5)
        rows.append(
            {
                "J": float(j),
                "KT": float(kt),
                "KQ": float(kq),
                "eta": float(j * kt / (2.0 * math.pi * kq)) if abs(kq) > 1e-12 else float("nan"),
            }
        )
    return {
        "model": "blade-element momentum theory (Prandtl tip/hub loss, cambered section, damped induction)",
        "parameters": {
            "cd0": cd0,
            "cl_slope": cl_slope,
            "drag_quadratic": drag_quadratic,
            "cl_max": cl_max,
            "stall_alpha_deg": stall_alpha_deg,
            "n_sections": n_sections,
            "n_iter": n_iter,
            "relax": relax,
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 10.  Experimental anchor contract
# ---------------------------------------------------------------------------


ANCHOR_SCHEMA = {
    "propeller": "INSEAN E779A",
    "quantity": "open_water (KT, KQ vs J)",
    "rows": [{"J": 0.0, "KT": None, "KQ": None}],
    "source": "primary-source citation (test report / journal table)",
    "retrieval": "how the numbers were obtained (file, page, date)",
    "verification": "REAL | WITHHELD",
}


def load_experimental_anchor(path: str | Path | None) -> dict[str, Any]:
    """Load a provenance-tagged experimental anchor table.

    The face refuses to *invent* reference numbers: an anchor file must carry
    ``source``, ``retrieval`` and numeric rows.  Anything else yields
    ``status = WITHHELD`` with a machine-readable reason, and every downstream
    comparison is then reported as ``WITHHELD`` instead of silently passing.
    """
    if path is None:
        return {
            "status": "WITHHELD",
            "reason": "no anchor file supplied (air-gapped run environment ships no "
                      "primary-source INSEAN E779A table)",
            "schema": ANCHOR_SCHEMA,
            "rows": [],
        }
    p = Path(path)
    if not p.exists():
        return {
            "status": "WITHHELD",
            "reason": f"anchor file not found: {p}",
            "schema": ANCHOR_SCHEMA,
            "rows": [],
        }
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = [r for r in data.get("rows", []) if r.get("KT") is not None and r.get("KQ") is not None]
    if not rows or not data.get("source") or not data.get("retrieval"):
        return {
            "status": "WITHHELD",
            "reason": "anchor file incomplete (needs numeric KT/KQ rows + source + retrieval)",
            "schema": ANCHOR_SCHEMA,
            "rows": rows,
        }
    return {"status": "REAL", "source": data["source"], "retrieval": data["retrieval"], "rows": rows}


def openwater_curve_criteria(
    points: Sequence[dict[str, Any]],
    *,
    anchor: dict[str, Any] | None = None,
    bemt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the open-water acceptance criteria for a set of J points.

    Criterion 1 (``J in [0.4, 0.8]``: ``|dKT| <= 8 %``, ``|dKQ| <= 10 %``,
    ``|d eta| <= 5 %``) and criterion 2 (end points ``J = 0.1/0.9`` at 15 %)
    are evaluated against the experimental anchor when available, otherwise
    against the BEMT reference (explicitly labelled MODEL) or reported
    ``WITHHELD``.
    """
    pts = sorted(points, key=lambda p: p["j"])
    js = np.array([p["j"] for p in pts])
    kt = np.array([p["kT"] for p in pts])
    kq = np.array([p["kQ"] for p in pts])
    eta = np.array([p["eta"] for p in pts])

    out: dict[str, Any] = {
        "n_points": int(len(pts)),
        "j": js.tolist(),
        "kT": kt.tolist(),
        "kQ": kq.tolist(),
        "eta": eta.tolist(),
    }

    # --- in-house physics checks (no external data required) ---------------
    dkt = np.diff(kt)
    out["kt_monotone_decreasing"] = bool(np.all(dkt < 0))
    zero_cross = None
    for i in range(len(js) - 1):
        if kt[i] * kt[i + 1] < 0:
            zero_cross = float(
                js[i] - kt[i] * (js[i + 1] - js[i]) / (kt[i + 1] - kt[i])
            )
            break
    out["jt_zero_thrust"] = zero_cross
    out["kq_positive"] = bool(np.all(kq > 0))
    out["eta_positive"] = bool(np.all(eta > 0))

    # --- convergence residuals --------------------------------------------
    resid = [
        p.get("convergence", {}).get("thrust", {}).get("max_block_deviation", float("nan"))
        for p in pts
    ]
    out["period_average_residual_max"] = float(np.nanmax(resid)) if resid else float("nan")
    out["period_average_residual_lt_1pct"] = bool(np.nanmax(resid) < 0.01) if resid else False

    # --- reference comparison ---------------------------------------------
    def _compare(ref_rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        by_j = {round(float(r["J"]), 3): r for r in ref_rows}
        rows = []
        for p in pts:
            key = round(float(p["j"]), 3)
            r = by_j.get(key)
            if r is None:
                continue
            row = {"J": float(p["j"])}
            for q in ("KT", "KQ", "eta"):
                ref = r.get(q)
                if ref is None or abs(float(ref)) < 1e-12:
                    row[f"{q}_err_pct"] = float("nan")
                else:
                    row[f"{q}_err_pct"] = float((p["k" + ("T" if q == "KT" else "Q") if q != "eta" else "eta"] - ref) / ref * 100.0) if q != "eta" else float((p["eta"] - ref) / ref * 100.0)
            rows.append(row)
        inside = [r for r in rows if 0.4 - 1e-9 <= r["J"] <= 0.8 + 1e-9]
        ends = [r for r in rows if abs(r["J"] - 0.1) < 1e-9 or abs(r["J"] - 0.9) < 1e-9]
        def _mx(rs, q):
            vals = [abs(r[q]) for r in rs if np.isfinite(r[q])]
            return float(max(vals)) if vals else float("nan")
        return {
            "reference": label,
            "rows": rows,
            "max_err_pct_J04_08": {
                "KT": _mx(inside, "KT_err_pct"),
                "KQ": _mx(inside, "KQ_err_pct"),
                "eta": _mx(inside, "eta_err_pct"),
            },
            "max_err_pct_endpoints": {
                "KT": _mx(ends, "KT_err_pct"),
                "KQ": _mx(ends, "KQ_err_pct"),
                "eta": _mx(ends, "eta_err_pct"),
            },
            "crit1_pass": bool(
                np.isfinite(_mx(inside, "KT_err_pct"))
                and _mx(inside, "KT_err_pct") <= 8.0
                and _mx(inside, "KQ_err_pct") <= 10.0
                and _mx(inside, "eta_err_pct") <= 5.0
            ),
            "crit2_pass": bool(
                np.isfinite(_mx(ends, "KT_err_pct"))
                and _mx(ends, "KT_err_pct") <= 15.0
                and _mx(ends, "KQ_err_pct") <= 15.0
                and _mx(ends, "eta_err_pct") <= 15.0
            ),
        }

    if anchor and anchor.get("status") == "REAL" and anchor.get("rows"):
        out["vs_experiment"] = _compare(anchor["rows"], "INSEAN E779A experiment (injected anchor)")
    else:
        out["vs_experiment"] = {
            "status": "WITHHELD",
            "reason": (anchor or {}).get("reason", "no experimental anchor"),
            "crit1_pass": None,
            "crit2_pass": None,
        }
    if bemt and bemt.get("rows"):
        out["vs_bemt"] = _compare(bemt["rows"], "BEMT (MODEL, independent method)")
    return out
