"""Porous-media verification ladder for medium/low-porosity seepage (D3Q19 LBM).

This module is the *verification surface* of campaign node
``p339w2-n06-lbm-porous-media``.  It is organised as a falsifiable staircase in
which every rung has a pre-declared numeric gate and in which the
*resolved-geometry* route (port A) and the *volume-averaged* route (port B)
cross-adjudicate each other.

Rungs
-----
G1  Kozeny-Carman permeability      K = phi^3 d_p^2 / [180 (1 - phi)^2]
    measured on resolved disordered sphere packs, 3 porosity tiers,
    gate |K_lbm / K_kc - 1| < 5 %.
G2  Ergun inertial deviation
        -dp/dx = 150 mu (1-phi)^2 u / (phi^3 d_p^2) + 1.75 rho (1-phi) u^2 / (phi^3 d_p)
    measured at 2 pore-Reynolds tiers, gate < 10 %.
G3  Beavers-Joseph interface slip   du/dy|_int = alpha/sqrt(K) * (u_int - u_D)
    alpha extracted independently from a pressure-driven and a shear-driven
    configuration of the same ordered bed; gate on the cross-mechanism
    consistency |alpha_i/alpha_ii - 1| < 15 %.
G4  Brinkman screening length       delta = sqrt(K/phi)
    compared with the e-folding length lambda of the REV-averaged tangential
    velocity inside the bed (shear-driven), gate < 10 %.
G5  Two-port overlap: the volume-averaged Brinkman/Darcy closure calibrated on
    the shear-driven port blind-predicts the pressure-driven port; deviation
    of the slab-averaged velocity in the free flow + interface band < 5 %.

Conventions (declared before any run)
-------------------------------------
* Lattice units: dx = dt = 1, rho0 = 1.
* The body force is a per-fluid-node acceleration ``a`` (momentum input
  ``rho*a`` per fluid node per step).  At steady state the momentum balance is
  exact, so the force per *total* volume is ``G = rho*a*phi`` and Darcy's law
  reads ``K = nu * <u_x>_fluid / a`` (verified analytically on plane
  Poiseuille, where it returns ``h^2/12``).
* ``phi`` is the *void* fraction, ``d_p`` the Sauter (equal-volume sphere)
  diameter.  Nominal (continuous-geometry) values are verdict-bearing;
  digitised-mask values are reported as diagnostics.
* ``K`` is always the Darcy permeability in lattice units; comparisons are made
  on ``K/d_p^2``.

References
----------
Kozeny (1927); Carman (1937); Ergun (1952) Chem. Eng. Prog. 48, 89;
Beavers & Joseph (1967) JFM 30, 197; Saffman (1971) Stud. Appl. Math. 50, 93;
Brinkman (1947); Ginzburg (2008) TRT; Guo, Zheng & Shi (2002) PRE 65, 046308.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from .d3q19 import C as C3D, W as W3D, equilibrium3d, macroscopic3d
from .solver3d import collide_trt3d, stream3d
from .boundaries3d import bounce_back_cells_3d

__all__ = [
    "kozeny_carman_permeability",
    "kozeny_constant",
    "ergun_pressure_gradient",
    "brinkman_screening_length",
    "pore_reynolds",
    "SpherePack",
    "generate_sphere_pack",
    "bcc_bed",
    "rasterize_spheres",
    "pack_statistics",
    "specific_surface",
    "plane_average",
    "effective_geometry",
    "plane_fluid_fraction",
    "PorousLBM3D",
    "darcy_permeability",
    "free_region_quadratic_fit",
    "composite_brinkman_profile",
    "beavers_joseph_alpha",
    "fit_decay_length",
]


# ======================================================================
# 1. Analytic closures
# ======================================================================
def kozeny_carman_permeability(phi: float, d_p: float, constant: float = 180.0) -> float:
    """Kozeny-Carman permeability ``K = phi^3 d_p^2 / [C (1-phi)^2]``.

    ``constant=180`` is the classical form used by this ladder; ``150`` is the
    Ergun-viscous constant, kept switchable so the empirical band of the
    constant can be reported without touching the verdict value.
    """
    if not 0.0 < phi < 1.0:
        raise ValueError(f"porosity must be in (0,1), got {phi}")
    return phi**3 * d_p**2 / (constant * (1.0 - phi) ** 2)


def kozeny_constant(phi: float, d_p: float, K: float) -> float:
    """Effective Kozeny-Carman constant implied by a measured ``K``."""
    return phi**3 * d_p**2 / ((1.0 - phi) ** 2 * K)


def ergun_pressure_gradient(
    phi: float,
    d_p: float,
    u_super: float,
    rho: float = 1.0,
    mu: float = 1.0 / 6.0,
    viscous_constant: float = 150.0,
    inertial_constant: float = 1.75,
) -> dict:
    """Ergun (1952) pressure gradient and its two additive parts."""
    u = abs(u_super)
    visc = viscous_constant * mu * (1.0 - phi) ** 2 * u / (phi**3 * d_p**2)
    inert = inertial_constant * rho * (1.0 - phi) * u**2 / (phi**3 * d_p)
    return {
        "total": visc + inert,
        "viscous": visc,
        "inertial": inert,
        "inertial_fraction": inert / (visc + inert) if (visc + inert) > 0 else 0.0,
    }


def brinkman_screening_length(K: float, phi: float) -> float:
    """Node-specified Brinkman screening length ``delta = sqrt(K/phi)``."""
    return math.sqrt(K / phi)


def pore_reynolds(u_super: float, d_p: float, nu: float) -> float:
    """Pore Reynolds number ``Re_p = |u_s| d_p / nu``."""
    return abs(u_super) * d_p / nu


# ======================================================================
# 2. Geometry
# ======================================================================
@dataclass
class SpherePack:
    """A sphere pack in lattice units (periodic axes configurable)."""

    centers: np.ndarray  # (N, 3) in node coordinates, order (z, y, x)
    radius: float
    shape: tuple[int, int, int]  # (nz, ny, nx)
    periodic: tuple[bool, bool, bool] = (True, True, True)
    phi_nominal: float = float("nan")
    phi_mask: float = float("nan")
    mask: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def as_mask(self) -> np.ndarray:
        if self.mask is None:
            r_rast = self.radius + float(self.meta.get("raster_offset", 0.0))
            self.mask = rasterize_spheres(
                self.centers, r_rast, self.shape, periodic=self.periodic
            )
        return self.mask


def _min_image(centers: np.ndarray, shape: Sequence[float], periodic: Sequence[bool]) -> np.ndarray:
    """Pairwise minimum-image displacement matrix, shape (N, N, 3)."""
    L = np.asarray(shape, dtype=float)
    d = centers[:, None, :] - centers[None, :, :]
    for ax, per in enumerate(periodic):
        if per:
            d[..., ax] -= L[ax] * np.round(d[..., ax] / L[ax])
    return d


def _relax_overlaps(
    centers: np.ndarray,
    shape: Sequence[int],
    d_min: float,
    periodic: Sequence[bool],
    iters: int = 200,
    step: float = 0.35,
) -> tuple[np.ndarray, float]:
    """Jacobi relaxation pushing overlapping sphere centres apart (periodic)."""
    L = np.asarray(shape, dtype=float)
    c = centers.copy()
    rng = np.random.default_rng(12345)
    worst = 0.0
    for _ in range(iters):
        d = _min_image(c, shape, periodic)
        dist = np.sqrt((d**2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        overlap = dist < d_min
        worst = float((d_min - dist[overlap]).max()) if overlap.any() else 0.0
        if not overlap.any():
            break
        w = np.where(overlap, (d_min - dist) / d_min, 0.0)
        disp = (d * w[..., None]).sum(axis=1) * step
        disp += rng.normal(scale=1e-3 * d_min, size=disp.shape)  # break dead-locks
        c = c + disp
        for ax, per in enumerate(periodic):
            if per:
                c[:, ax] = np.mod(c[:, ax], L[ax])
    return c, worst


RASTER_OFFSET = 0.0  # raster at the nominal radius; the node-centre wall shrinks
#                      the effective body by half a cell, which is accounted for
#                      analytically in ``effective_geometry`` below (validated on
#                      the simple-cubic / Zick-Homsy anchor to 1.3 %).


def generate_sphere_pack(
    shape: Sequence[int],
    d_p: float,
    phi_target: float,
    seed: int = 0,
    *,
    grow_steps: int = 24,
    relax_iters: int = 160,
    raster_offset: float = RASTER_OFFSET,
) -> SpherePack:
    """Disordered monodisperse sphere pack by growth + overlap relaxation.

    Random-sequential-adsorption saturates near 0.62 solid fraction, so a fixed
    number of spheres ``N`` (set by the *target* porosity) is grown from a
    small diameter up to ``d_p`` while overlaps are relaxed away.  The final
    porosity is **not** tuned afterwards: ``phi_nominal`` is whatever the
    relaxed continuous geometry gives and is reported as such.
    """
    rng = np.random.default_rng(seed)
    L = np.asarray(shape, dtype=float)
    vol = float(np.prod(L))
    r = 0.5 * float(d_p)
    n = max(1, int(round((1.0 - phi_target) * vol / (4.0 / 3.0 * math.pi * r**3))))
    centers = rng.uniform(0.0, L, size=(n, 3))
    periodic = (True, True, True)
    d_min_target = 2.0 * r
    worst = 0.0
    for k in range(1, grow_steps + 1):
        d_min = d_min_target * k / grow_steps
        centers, worst = _relax_overlaps(centers, shape, d_min, periodic, iters=relax_iters)
    centers, worst = _relax_overlaps(centers, shape, d_min_target * 1.0005, periodic, iters=relax_iters)
    pack = SpherePack(
        centers=centers,
        radius=r,
        shape=tuple(int(v) for v in shape),
        periodic=periodic,
        phi_nominal=1.0 - n * (4.0 / 3.0 * math.pi * r**3) / vol,
        meta={
            "raster_offset": float(raster_offset),
            "raster_radius": float(r + raster_offset),
            "n_spheres": n,
            "d_p": float(d_p),
            "phi_target": float(phi_target),
            "seed": int(seed),
            "max_residual_overlap": worst,
            "packer": "growth+relaxation",
        },
    )
    return pack


def bcc_bed(
    n_layers: int,
    cell: float,
    radius: float,
    *,
    free_height: int,
    floor_layers: int = 1,
) -> SpherePack:
    """BCC sphere bed resting on a solid floor, slab-periodic in x and z.

    The bed occupies ``y in [0, y_bed_top]``; the top sphere layer is a
    complete square lattice so that the fluid/bed interface plane is flat
    (required for an unambiguous Beavers-Joseph interface condition).
    ``free_height`` lattice rows of pure fluid are added above the bed, plus one
    solid row for the moving/no-slip top wall.
    """
    a = float(cell)
    nx = int(round(a))
    nz = int(round(a))
    layer_dy = a / 2.0
    # sphere centres (z, y, x); layer k shifted by (a/2, a/2) on odd k
    centers = []
    for k in range(n_layers):
        # one sphere per layer per unit cell: x/z periodicity rebuilds the
        # square lattice, alternating a/2 offsets give the BCC stacking
        y = radius + k * layer_dy  # first layer touches the floor row
        shift = (a / 2.0) if (k % 2) else 0.0
        centers.append([shift, y, shift])
    centers = np.asarray(centers, dtype=float)
    y_top = max(c[1] for c in centers) + radius
    ny = int(math.ceil(y_top)) + int(free_height) + 1  # +1 top wall row
    pack = SpherePack(
        centers=centers,
        radius=radius,
        shape=(nz, ny, nx),
        periodic=(True, False, True),
        phi_nominal=float("nan"),
        meta={
            "raster_offset": RASTER_OFFSET,
            "raster_radius": float(radius + RASTER_OFFSET),
            "bed": "bcc",
            "cell": a,
            "n_layers": int(n_layers),
            "layer_dy": layer_dy,
            "y_bed_top": float(y_top),
            "free_height": int(free_height),
            "floor_layers": int(floor_layers),
        },
    )
    bed_vol = nx * nz * y_top
    pack.phi_nominal = float(1.0 - len(centers) * (4.0 / 3.0 * math.pi * radius**3) / bed_vol)
    pack.meta["phi_bed_nominal"] = pack.phi_nominal
    return pack


def rasterize_spheres(
    centers: np.ndarray,
    radius: float,
    shape: Sequence[int],
    periodic: Sequence[bool] = (True, True, True),
) -> np.ndarray:
    """Boolean solid mask ``(nz, ny, nx)``; nodes whose centre lies inside a sphere."""
    nz, ny, nx = (int(v) for v in shape)
    ZZ = np.arange(nz, dtype=float)[:, None, None]
    YY = np.arange(ny, dtype=float)[None, :, None]
    XX = np.arange(nx, dtype=float)[None, None, :]
    d2 = np.full((nz, ny, nx), np.inf)
    r2 = float(radius) ** 2
    for cz, cy, cx in np.asarray(centers, dtype=float):
        dz = np.abs(ZZ - cz)
        if periodic[0]:
            dz = np.minimum(dz, nz - dz)
        dy = np.abs(YY - cy)
        if periodic[1]:
            dy = np.minimum(dy, ny - dy)
        dx = np.abs(XX - cx)
        if periodic[2]:
            dx = np.minimum(dx, nx - dx)
        d2 = np.minimum(d2, dz * dz + dy * dy + dx * dx)
        del dz, dy, dx
    return d2 < r2


def pack_statistics(mask: np.ndarray, d_p: float) -> dict:
    """Digitised-geometry audit: porosity, specific surface, Sauter diameter.

    The staircase (voxel) surface *over*-counts area, so ``d_sauter_mask`` is
    systematically smaller than the analytic ``6(1-phi)/d_p`` (by ~25 % at
    ``d_p`` ~ 10 lattice units, less at higher resolution).  Verdicts therefore
    use the nominal geometric ``d_p``; this function only audits the digitiser.
    """
    solid = np.asarray(mask, dtype=bool)
    n_tot = solid.size
    n_sol = int(solid.sum())
    phi = 1.0 - n_sol / n_tot
    # internal solid/fluid faces -> surface area (dx = 1)
    faces = 0
    for ax in range(3):
        a = np.swapaxes(solid, 0, ax)
        diff = np.diff(a.astype(np.int8), axis=0)
        faces += int(np.abs(diff).sum())
        # wrap-around faces are *not* counted: periodic images are interior
    # specific surface per total volume
    Sv = faces / n_tot
    d_sauter = 6.0 * (1.0 - phi) / Sv if Sv > 0 else float("nan")
    return {
        "phi_mask": phi,
        "phi_solid": n_sol / n_tot,
        "n_solid": n_sol,
        "solid_faces": faces,
        "Sv": Sv,
        "d_sauter_mask": d_sauter,
        "d_sauter_over_dp": d_sauter / d_p if d_p else float("nan"),
    }


def specific_surface(mask: np.ndarray) -> float:
    """Specific surface area per total volume (staircase-counted)."""
    return pack_statistics(mask, d_p=0.0)["Sv"]


def effective_geometry(phi_mask: float, d_p: float, d_sauter_mask: float | None = None,
                       phi_nominal: float | None = None) -> dict:
    """Effective (flow-seeing) geometry of a voxelised pack.

    The library bounce-back places the no-slip wall at the **solid node
    centres**, so the fluid region is larger than the mask complement by a
    half-cell shell::

        (1 - phi_eff) = (1 - phi_mask) - 0.5 * S_v(mask)
        d_eff         = d_p - 1          (0.5 cell of radius lost per side)

    Both statements are calibrated against the simple-cubic array of Zick &
    Homsy (1982) (see ``benchmarks/pending/porous_ladder/cases/anchor_zick_homsy.json``):
    the measured permeability there matches the tabulated drag at the
    *effective* porosity, not at the mask porosity.
    """
    solid_mask = 1.0 - phi_mask
    if phi_nominal is not None:
        # analytic specific surface of the *nominal* geometry (the staircase-
        # counted surface is inflated ~10-20 % and would bias the correction)
        Sv = 6.0 * (1.0 - phi_nominal) / d_p
    elif d_sauter_mask:
        Sv = 6.0 * solid_mask / d_sauter_mask
    else:
        Sv = float("nan")
    solid_eff = solid_mask - 0.5 * Sv
    return {
        "phi_eff": 1.0 - solid_eff,
        "d_eff": d_p - 1.0,
        "Sv_mask": Sv,
        "solid_mask": solid_mask,
        "solid_eff": solid_eff,
    }


def plane_average(field: torch.Tensor, fluid: torch.Tensor, axis: int = 1) -> np.ndarray:
    """Superficial (per-total-area) plane average of ``field`` over fluid nodes.

    ``axis=1`` averages the ``y`` planes of a ``(nz, ny, nx)`` field, i.e.
    returns ``sum_fluid(field) / (nz*nx)`` per plane (void fraction included).
    """
    f = field.detach().cpu().to(torch.float64)
    m = fluid.detach().cpu().to(torch.float64)
    nz, ny, nx = m.shape
    if axis != 1:
        raise ValueError("plane_average only supports axis=1 (wall-normal y)")
    return ((f * m).sum(dim=(0, 2)) / (nz * nx)).numpy()


def plane_fluid_fraction(fluid: torch.Tensor, axis: int = 1) -> np.ndarray:
    m = fluid.detach().cpu().to(torch.float64)
    nz, ny, nx = m.shape
    return m.sum(dim=(0, 2)).numpy() / (nz * nx)


# ======================================================================
# 3. LBM driver (D3Q19, TRT, half-way bounce-back, Guo-type body force)
# ======================================================================
class PorousLBM3D:
    """Minimal, auditable D3Q19 porous-media driver.

    Everything except the step loop is a library kernel
    (:func:`tensorlbm.solver3d.collide_trt3d`, :func:`~stream3d`,
    :func:`tensorlbm.boundaries3d.bounce_back_cells_3d`).
    """

    def __init__(
        self,
        solid: torch.Tensor | np.ndarray,
        tau: float = 1.0,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        lambda_trt: float = 3.0 / 16.0,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        if isinstance(solid, np.ndarray):
            solid = torch.from_numpy(solid)
        self.solid = solid.to(device=self.device, dtype=torch.bool)
        self.fluid = ~self.solid
        self.tau = float(tau)
        self.nu = (self.tau - 0.5) / 3.0
        self.lambda_trt = float(lambda_trt)
        nz, ny, nx = self.solid.shape
        zeros = torch.zeros((nz, ny, nx), dtype=dtype, device=self.device)
        ones = torch.ones((nz, ny, nx), dtype=dtype, device=self.device)
        self.f = equilibrium3d(ones, zeros, zeros, zeros, device=self.device)
        self._w = W3D.to(device=self.device, dtype=dtype).view(-1, 1, 1, 1)
        self.last_accel = 0.0
        self._cx = C3D[:, 0].to(device=self.device, dtype=dtype).view(-1, 1, 1, 1)

    # -- helpers -------------------------------------------------------
    def _force_term(self, accel: float, accel_mask: torch.Tensor | None) -> torch.Tensor:
        """Simple (first-order) forcing ``f_q += 3 w_q (c_q . a) rho``.

        The force is **masked to fluid nodes** by default: injecting it into
        solid nodes would leak a "phantom" momentum back into the fluid through
        the streaming step.
        """
        mask = self.fluid if accel_mask is None else accel_mask
        term = 3.0 * self._w * (self._cx * float(accel))
        return term * mask.to(dtype=self.dtype).unsqueeze(0)

    def macroscopic(self):
        """Raw lattice moments ``(rho, ux, uy, uz)`` (no force correction)."""
        return macroscopic3d(self.f)

    def velocity(self):
        """Hydrodynamic velocity: raw moment + ``a/2`` forcing correction.

        For the simple forcing used here the physical velocity is
        ``u = sum_q c_q f_q / rho + F/(2 rho)`` (Guo, Zheng & Shi 2002).  The
        offset is *not* negligible for Darcy permeability: it is an O(1)
        fraction of ``u`` (``a/2 / u = nu / (2 phi K)``), so every reported
        velocity and every permeability uses this corrected field.
        """
        rho, ux, uy, uz = macroscopic3d(self.f)
        c = 0.5 * float(self.last_accel)
        return rho, ux + c, uy, uz

    def step(
        self,
        accel: float = 0.0,
        accel_mask: torch.Tensor | None = None,
        top_wall_speed: float | None = None,
    ) -> None:
        f = collide_trt3d(self.f, self.tau, self.lambda_trt)
        f = stream3d(f)
        f = bounce_back_cells_3d(f, self.solid)
        self.last_accel = float(accel)
        if accel != 0.0:
            f = f + self._force_term(accel, accel_mask)
        if top_wall_speed is not None:
            nz, ny, nx = self.solid.shape
            wall = torch.zeros_like(self.solid)
            wall[:, -1, :] = True
            rho_w = torch.ones((nz, 1, nx), dtype=self.dtype, device=self.device)
            uz0 = torch.zeros_like(rho_w)
            feq_w = equilibrium3d(
                rho_w, rho_w * float(top_wall_speed), uz0, uz0,
                device=self.device,
            )
            f = torch.where(wall.unsqueeze(0), feq_w, f)
        self.f = f

    def run(
        self,
        steps: int,
        *,
        accel: float = 0.0,
        accel_mask: torch.Tensor | None = None,
        top_wall_speed: float | None = None,
        check_every: int = 200,
        rtol: float = 1e-7,
        min_steps: int = 400,
        max_wall_s: float = 1e9,
    ) -> dict:
        """March to steady state (velocity drift over a window < ``rtol``)."""
        import time

        t0 = time.time()
        hist: list[float] = []
        u_ref = None
        for n in range(1, steps + 1):
            self.step(accel, accel_mask, top_wall_speed)
            if n % check_every == 0:
                _, ux, _, _ = self.velocity()
                u = float((ux * self.fluid).sum() / self.fluid.sum())
                if u_ref is None:
                    u_ref = u
                hist.append(u)
                if os.environ.get("TL_PROGRESS"):
                    print(f"    step {n:6d}  u={u:.6e}  wall={time.time()-t0:7.1f}s", flush=True)
                if n >= min_steps and len(hist) >= 3:
                    a, b = hist[-3], hist[-1]
                    if abs(b - a) <= rtol * max(abs(b), 1e-30):
                        break
                if time.time() - t0 > max_wall_s:
                    break
        return {
            "steps": n,
            "wall_s": time.time() - t0,
            "steady": bool(len(hist) >= 3 and abs(hist[-1] - hist[-3]) <= rtol * max(abs(hist[-1]), 1e-30)),
            "u_history_tail": [float(v) for v in hist[-4:]],
        }


def darcy_permeability(nu: float, u_mean_fluid: float, accel: float) -> float:
    """``K = nu * <u_x>_fluid / a`` (exact steady-state momentum balance)."""
    return nu * u_mean_fluid / accel


# ======================================================================
# 4. Interface closure: Beavers-Joseph / Brinkman
# ======================================================================
def free_region_quadratic_fit(
    y: np.ndarray, u: np.ndarray, y_int: float, drop: int = 1
) -> dict:
    """Least-squares quadratic fit of the free-flow profile, evaluated at y_int.

    Returns ``u_int`` (value at the interface plane), ``slope`` (du/dy at the
    interface) and the curvature (``-a/nu`` for a pressure-driven port).
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    sel = np.isfinite(u)
    if drop:
        # drop the rows adjacent to the walls: staircase/BB rows are not part
        # of the continuum free-flow profile
        idx = np.where(sel)[0]
        sel[idx[:drop]] = False
        sel[idx[-drop:]] = False
    coeffs = np.polyfit(y[sel], u[sel], 2)
    c2, c1, c0 = coeffs
    return {
        "u_int": float(np.polyval(coeffs, y_int)),
        "slope": float(2 * c2 * y_int + c1),
        "curvature": float(2 * c2),
        "coeffs": [float(c) for c in coeffs],
        "n_points": int(sel.sum()),
        "rms_residual": float(np.sqrt(np.mean((np.polyval(coeffs, y[sel]) - u[sel]) ** 2))),
    }


def beavers_joseph_alpha(K: float, slope: float, u_int: float, u_darcy: float) -> float:
    """``alpha = sqrt(K) * (du/dy) / (u_int - u_D)``."""
    denom = u_int - u_darcy
    if abs(denom) < 1e-300:
        return float("nan")
    return math.sqrt(K) * slope / denom


def composite_brinkman_profile(
    *,
    K: float,
    phi: float,
    alpha: float,
    y_int: float,
    free_top: float,
    nu: float,
    accel_free: float = 0.0,
    accel_bed: float = 0.0,
    wall_speed: float = 0.0,
    rho: float = 1.0,
    mu_eff: float | None = None,
    n_bed: int = 400,
    n_free: int = 400,
) -> dict:
    """Volume-averaged (port B) composite Brinkman/Darcy solution.

    Free region ``y in [y_int, free_top]``: ``u'' = -accel_free/nu``.
    Bed ``y in [0, y_int]``: ``mu_eff U'' - (mu/K) U = -G_bed`` with
    ``G_bed = rho*accel_bed*phi`` and screening length ``lam = sqrt(mu_eff K/mu)``.
    Matching: no-slip at the floor, velocity continuity ``u_free(y_int) =
    U(y_int)/phi``, the Beavers-Joseph condition at ``y_int`` and the wall
    velocity at ``free_top``.
    """
    mu = rho * nu
    mu_eff = mu if mu_eff is None else float(mu_eff)
    lam = math.sqrt(mu_eff * K / mu)
    G_bed = rho * accel_bed * phi
    u_D = K * G_bed / mu  # Darcy (superficial) velocity deep in the bed
    A2 = -rho * accel_free / (2.0 * mu)

    eB = math.exp(-y_int / lam)
    # unknowns: [A1, A0, P, Q]
    M = np.zeros((4, 4))
    rhs = np.zeros(4)
    # (1) U(0) = 0
    M[0, 2] = eB
    M[0, 3] = 1.0 / eB
    rhs[0] = -u_D
    # (2) u_free(y_int) = U(y_int)/phi
    M[1, 1] = 1.0
    M[1, 2] = -1.0 / phi
    M[1, 3] = -1.0 / phi
    rhs[1] = u_D / phi
    # (3) u_free(free_top) = wall_speed
    H = free_top - y_int
    M[2, 0] = H
    M[2, 1] = 1.0
    rhs[2] = wall_speed - A2 * H * H
    # (4) B-J: du/dy|_{y_int} = alpha/sqrt(K) * (u_free(y_int) - u_D)
    bk = alpha / math.sqrt(K)
    M[3, 0] = 1.0
    M[3, 1] = -bk
    rhs[3] = -bk * u_D

    sol = np.linalg.solve(M, rhs)
    A1, A0, P, Q = sol
    y_bed = np.linspace(0.0, y_int, n_bed)
    dy = y_bed - y_int
    U_super = u_D + P * np.exp(dy / lam) + Q * np.exp(-dy / lam)
    y_fr = np.linspace(y_int, free_top, n_free)
    u_free = A2 * (y_fr - y_int) ** 2 + A1 * (y_fr - y_int) + A0
    y_all = np.concatenate([y_bed, y_fr[1:]])
    u_all = np.concatenate([U_super, u_free[1:]])
    return {
        "y": y_all,
        "u_super": u_all,
        "u_free": u_free,
        "U_bed": U_super,
        "u_D": u_D,
        "u_int": float(A0),
        "slope": float(A1),
        "lam": lam,
        "alpha_used": float(alpha),
        "K": float(K),
    }


def fit_decay_length(
    y: np.ndarray,
    u: np.ndarray,
    *,
    y_hi: float,
    y_lo: float,
    floor: float = 0.02,
) -> dict:
    """Fit ``u ~ exp(y/lam)`` on a band ``y in [y_lo, y_hi]`` of the bed profile."""
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    sel = (y >= y_lo) & (y <= y_hi) & (u > floor * np.nanmax(u)) & np.isfinite(u)
    if sel.sum() < 4:
        return {"lam": float("nan"), "n_points": int(sel.sum())}
    slope, intercept = np.polyfit(y[sel], np.log(u[sel]), 1)
    lam = 1.0 / slope
    pred = np.exp(intercept + slope * y[sel])
    r2 = 1.0 - np.sum((u[sel] - pred) ** 2) / np.sum((u[sel] - u[sel].mean()) ** 2)
    return {
        "lam": float(lam),
        "n_points": int(sel.sum()),
        "decay_r2": float(r2),
        "band": [float(y_lo), float(y_hi)],
    }