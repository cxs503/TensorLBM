"""Micro-textured super-hydrophobic walls for the SCMP lattice-Boltzmann kernel.

Extends the *smooth-wall* wettability capability (``wetting_n01`` /
``benchmarks/in32_l01_wetting_ladder``) to **micro-structured textures**: a
periodic array of pillars of width ``D``, pitch ``P`` and height ``H`` whose
solid surface carries an *intrinsic* contact angle ``theta_Y``.

Geometry (closed form vs voxel audit)
-------------------------------------
For 2-D pillar strips (what a D2Q9 field can represent)

    f_s = D / P                       (projected solid fraction)
    r   = 1 + 2 H / P                 (wetted area / projected area)

(the Wenzel state wets pillar tops *and* side walls *and* the groove floor:
``D + 2H + (P - D) = P + 2H`` per pitch).  For a 3-D square array of circular
pillars (analytic reference only)

    f_s = pi D^2 / (4 P^2),   r = 1 + pi D H / P^2 .

:func:`measure_voxel_geometry` re-measures ``f_s`` and ``r`` from the voxelised
mask (columns reaching the pillar-top row; exposed solid-fluid faces per unit
projected length) so the closed form is *checked* against the geometry the
solver actually sees.

State closures
--------------
Cassie-Baxter (composite interface, gas trapped under the drop)::

    cos theta_CB = f_s (cos theta_Y + 1) - 1

Wenzel (liquid impregnates the grooves; saturated at |cos| = 1, i.e. complete
wetting when ``r |cos theta_Y| >= 1``)::

    cos theta_W  = r cos theta_Y

Comparing the two surface energies per projected area (Young's relation
``sigma_sg - sigma_sl = sigma cos theta_Y``) gives

    E_C - E_W = (1 - f_s) sigma + (r - f_s) sigma cos theta_Y ,

so the Cassie state is the thermodynamic minimum iff

    cos theta_Y < cos theta_c = (f_s - 1) / (r - f_s).

The kinetic barrier between the states is the Laplace pressure required to push
the suspended meniscus through the groove opening ``w = P - D``::

    p_c = -2 sigma cos theta_Y / (P - D)                    (2-D gap, used here)

with the widely quoted 3-D pillar force-balance variant

    p_c = -4 sigma cos theta_Y f_s / (D (1 - f_s)) = 4 sigma |cos theta_Y| / (P - D)

(exactly twice the 2-D gap value for this geometry).  Both are returned.

All quantities are in lattice units of the ``G = 5`` / ``psi_exp`` / ``tau = 1``
SCMP scheme (``rho_l = 1.957``, ``rho_v = 0.1596``, ``sigma = 0.05611`` — the
calibration of ``benchmarks/verified/laplace_droplet``).

References
----------
Cassie & Baxter (1944) Trans. Faraday Soc. 40, 546.
Wenzel (1936) Ind. Eng. Chem. 28, 988.
Lafuma & Quere (2003) Nature Mater. 2, 457 (state diagram / crossover).
Bico, Marzolin & Quere (1999) Europhys. Lett. 47, 220 (impalement pressure).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..d2q9 import OPPOSITE, equilibrium, macroscopic
from ..multiphase import collide_sc_single_component, psi_exp
from ..solver import stream
from .engine import RHO_L_DEFAULT, RHO_V_DEFAULT, SIGMA_DEFAULT, tanh_profile
from .theta_fit import angle_estimate, circle_fit, interface_points, level_fraction

__all__ = [
    "PillarTexture",
    "PressurisedStepper",
    "TexturedChannel",
    "apparent_angle",
    "apparent_angle_n01",
    "cap_radius",
    "cap_radius",
    "breakthrough_pressure",
    "cassie_wenzel_crossover_deg",
    "cos_theta_cassie_baxter",
    "cos_theta_wenzel",
    "groove_fill_fraction",
    "interface_width_cells",
    "measure_voxel_geometry",
    "predicted_wetting_state",
    "run_impalement",
    "run_static_angle",
    "textured_drop_state",
    "theta_deg_from_cos",
]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PillarTexture:
    """Periodic pillar array on the bottom wall.

    Args:
        pitch:         Centre-to-centre pillar spacing ``P`` (lattice cells).
        pillar_width:  Pillar width ``D`` (cells); ``1 <= D < P``.
        height:        Pillar height ``H`` (cells above the base slab).
        theta_y_deg:   Nominal intrinsic contact angle of the pillar material.
        wall_fraction: Optional override of the SCMP wall-wetting fraction
                       (``wall_psi / psi(rho_l)``); defaults to the calibrated
                       :func:`tensorlbm.wetting_n01.angle_to_wall_fraction` map.
        layout:        ``"strip2d"`` (pillar strips — what the solver sees) or
                       ``"pillar3d"`` (circular-pillar analytic reference).
    """

    pitch: int
    pillar_width: int
    height: int
    theta_y_deg: float
    wall_fraction: float | None = None
    layout: str = "strip2d"

    def __post_init__(self) -> None:
        if self.pitch < 4:
            raise ValueError("pitch must be >= 4 cells")
        if not (1 <= self.pillar_width < self.pitch):
            raise ValueError("pillar_width must satisfy 1 <= D < P")
        if self.height < 0:
            raise ValueError("height must be >= 0")
        if self.layout not in ("strip2d", "pillar3d"):
            raise ValueError("layout must be 'strip2d' or 'pillar3d'")

    @property
    def solid_fraction(self) -> float:
        """Projected solid fraction ``f_s`` (closed form)."""
        if self.layout == "strip2d":
            return self.pillar_width / self.pitch
        return math.pi * self.pillar_width**2 / (4.0 * self.pitch**2)

    @property
    def roughness(self) -> float:
        """Roughness factor ``r`` = wetted / projected area (closed form)."""
        if self.layout == "strip2d":
            return 1.0 + 2.0 * self.height / self.pitch
        return 1.0 + math.pi * self.pillar_width * self.height / self.pitch**2

    @property
    def gap(self) -> float:
        """Groove opening width ``w = P - D`` (cells)."""
        return self.pitch - self.pillar_width

    @property
    def cos_theta_y(self) -> float:
        return math.cos(math.radians(self.theta_y_deg))

    @property
    def wetting_fraction(self) -> float:
        """SCMP wall-wetting fraction that produces ``theta_Y`` on a flat wall."""
        if self.wall_fraction is not None:
            return float(self.wall_fraction)
        from ..wetting_n01 import angle_to_wall_fraction

        return angle_to_wall_fraction(self.theta_y_deg)

    def to_dict(self) -> dict:
        return {
            "pitch": self.pitch,
            "pillar_width": self.pillar_width,
            "height": self.height,
            "gap": self.gap,
            "theta_y_deg": self.theta_y_deg,
            "wetting_fraction": self.wetting_fraction,
            "layout": self.layout,
            "solid_fraction_closed": self.solid_fraction,
            "roughness_closed": self.roughness,
        }


def theta_deg_from_cos(cos_theta: float) -> float:
    """Angle in degrees from a clipped cosine."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(cos_theta)))))


def cos_theta_cassie_baxter(cos_theta_y: float, solid_fraction: float) -> float:
    """``cos theta_CB = f_s (cos theta_Y + 1) - 1`` (Cassie & Baxter 1944)."""
    return solid_fraction * (float(cos_theta_y) + 1.0) - 1.0


def cos_theta_wenzel(cos_theta_y: float, roughness: float) -> float:
    """``cos theta_W = r cos theta_Y`` (Wenzel 1936); clip to [-1, 1] at use."""
    return roughness * float(cos_theta_y)


def cassie_wenzel_crossover_deg(solid_fraction: float, roughness: float) -> float:
    """Intrinsic angle separating the Wenzel- from the Cassie-favoured regime.

    ``cos theta_c = (f_s - 1) / (r - f_s)``; Cassie is favoured for
    ``theta_Y > theta_c`` (energy comparison in the module docstring).
    """
    if roughness <= solid_fraction:
        raise ValueError("roughness must exceed the solid fraction")
    return theta_deg_from_cos((solid_fraction - 1.0) / (roughness - solid_fraction))


def predicted_wetting_state(texture: PillarTexture) -> dict:
    """Cassie/Wenzel predictions for a texture (closed forms only)."""
    f_s = texture.solid_fraction
    r = texture.roughness
    c_y = texture.cos_theta_y
    cos_cb = f_s * (c_y + 1.0) - 1.0
    cos_w = r * c_y
    theta_c = cassie_wenzel_crossover_deg(f_s, r)
    state = "cassie" if texture.theta_y_deg > theta_c else "wenzel"
    return {
        "solid_fraction": f_s,
        "roughness": r,
        "cos_theta_y": c_y,
        "theta_y_deg": texture.theta_y_deg,
        "cos_theta_cb": cos_cb,
        "theta_cb_deg": theta_deg_from_cos(cos_cb),
        "cos_theta_w": cos_w,
        "theta_w_deg": theta_deg_from_cos(cos_w),
        "wenzel_saturated": bool(abs(cos_w) >= 1.0),
        "theta_crossover_deg": theta_c,
        "state_predicted": state,
        "theta_predicted_deg": (
            theta_deg_from_cos(cos_cb) if state == "cassie" else theta_deg_from_cos(cos_w)
        ),
    }


def breakthrough_pressure(texture: PillarTexture, sigma: float = SIGMA_DEFAULT) -> dict:
    """Laplace break-through (impalement) pressure of the suspended meniscus.

    Both closed forms quoted in the module docstring, in lattice pressure units
    (``sigma / length``).
    """
    c_y = texture.cos_theta_y
    w = float(texture.gap)
    p_2d = -2.0 * sigma * c_y / w
    p_3d = -4.0 * sigma * c_y * texture.solid_fraction / (
        texture.pillar_width * (1.0 - texture.solid_fraction)
    )
    return {
        "gap": w,
        "p_c_laplace_2d": p_2d,
        "p_c_force_balance_3d": p_3d,
        "p_c_abs_2d": abs(p_2d),
        "p_c_abs_3d": abs(p_3d),
    }


# ---------------------------------------------------------------------------
# Textured channel (duck-types the Channel contract used by WallStepper)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TexturedChannel:
    """Channel with a pillar array on the bottom wall and a flat top wall.

    Duck-types :class:`tensorlbm.contact_line.engine.Channel`: exposes
    ``shape``, ``nx``, ``ny``, ``device``, ``grid()``, ``solid_mask()`` and
    ``y_wall()`` (the *composite* plane, i.e. the pillar tops), so the unmodified
    :class:`tensorlbm.contact_line.engine.WallStepper` can drive it.
    """

    nx: int
    ny: int
    wall_rows: int
    texture: PillarTexture
    device: str = "cpu"

    def __post_init__(self) -> None:
        tex = self.texture
        if self.nx % tex.pitch != 0:
            raise ValueError(
                f"nx={self.nx} must be a multiple of the pitch {tex.pitch} "
                "(periodic pillar array)"
            )
        if self.wall_rows < 1:
            raise ValueError("wall_rows must be >= 1")
        if 2 * self.wall_rows + tex.height >= self.ny - 4:
            raise ValueError("texture + walls leave no fluid region")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def n_pillars(self) -> int:
        return self.nx // self.texture.pitch

    @property
    def n_fluid_y(self) -> int:
        return self.ny - 2 * self.wall_rows - self.texture.height

    def y_wall(self) -> float:
        """Height of the composite (pillar-top) plane."""
        return self.wall_rows + self.texture.height - 0.5

    def y_base(self) -> float:
        """Height of the flat base-wall plane between pillars."""
        return self.wall_rows - 0.5

    def y_top(self) -> float:
        """Height of the top wall plane."""
        return self.ny - self.wall_rows - 0.5

    def pillar_columns(self) -> list[tuple[int, int]]:
        """``(x0, x1)`` inclusive column spans of the pillars."""
        tex = self.texture
        off = (tex.pitch - tex.pillar_width) // 2
        return [
            (k * tex.pitch + off, k * tex.pitch + off + tex.pillar_width - 1)
            for k in range(self.n_pillars)
        ]

    def solid_mask(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Boolean ``(ny, nx)`` mask: base slab + pillars + flat top wall."""
        dev = torch.device(device if device is not None else self.device)
        tex = self.texture
        solid = torch.zeros(self.shape, dtype=torch.bool, device=dev)
        solid[: self.wall_rows, :] = True
        if tex.height > 0:
            for x0, x1 in self.pillar_columns():
                solid[self.wall_rows : self.wall_rows + tex.height, x0 : x1 + 1] = True
        solid[self.ny - self.wall_rows :, :] = True
        return solid

    def groove_mask(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Mask of the (empty) groove volume between pillars."""
        dev = torch.device(device if device is not None else self.device)
        tex = self.texture
        solid = self.solid_mask(dev)
        groove = torch.zeros(self.shape, dtype=torch.bool, device=dev)
        groove[self.wall_rows : self.wall_rows + tex.height, :] = True
        return groove & ~solid

    def flat_mask(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Solid mask of the same channel without pillars (flat reference wall)."""
        dev = torch.device(device if device is not None else self.device)
        solid = torch.zeros(self.shape, dtype=torch.bool, device=dev)
        solid[: self.wall_rows, :] = True
        solid[self.ny - self.wall_rows :, :] = True
        return solid

    def grid(self, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        dev = torch.device(device if device is not None else self.device)
        yy, xx = torch.meshgrid(
            torch.arange(self.ny, dtype=torch.float32, device=dev),
            torch.arange(self.nx, dtype=torch.float32, device=dev),
            indexing="ij",
        )
        return xx, yy


def measure_voxel_geometry(channel: TexturedChannel) -> dict:
    """Re-measure ``f_s`` and ``r`` from the *voxelised* mask (closed-form audit).

    ``f_s``: fraction of columns whose solid reaches the pillar-top row.
    ``r``:   exposed solid-fluid 4-neighbour faces in the texture band divided by
             the projected length ``nx`` (the domain is periodic in ``x``, so the
             wrap face is included).  The voxel surface is piece-wise axis
             aligned, hence the face count is the exact staircase wetted
             perimeter and agrees with the closed form to ``O(1/nx)``.
    """
    tex = channel.texture
    solid = channel.solid_mask()
    top_row = channel.wall_rows + tex.height - 1
    nx = channel.nx

    f_s_vox = float(solid[top_row, :].sum()) / nx

    band = solid[: top_row + 1, :]
    fv = float((band[:, :-1] != band[:, 1:]).sum() + (band[:, -1] != band[:, 0]).sum())
    above = torch.zeros_like(band[:1, :])
    ext = torch.cat([band, above], dim=0)
    fh = float((ext[:-1, :] != ext[1:, :]).sum())
    r_vox = (fv + fh) / nx

    return {
        "f_s_voxel": f_s_vox,
        "f_s_closed": tex.solid_fraction,
        "f_s_rel_error": abs(f_s_vox - tex.solid_fraction) / max(tex.solid_fraction, 1e-12),
        "r_voxel": r_vox,
        "r_closed": tex.roughness,
        "r_rel_error": abs(r_vox - tex.roughness) / max(tex.roughness, 1e-12),
        "gap_cells": tex.gap,
        "pillar_height_cells": tex.height,
        "n_pillars": channel.n_pillars,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def interface_width_cells(
    rho: torch.Tensor,
    solid: torch.Tensor,
    *,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    column: int | None = None,
) -> float:
    """10-90 % diffuse-interface thickness along a vertical cut (cells).

    Taken through the apex of the drop (the column with the highest liquid
    density), which is where the interface is normal to the cut and therefore
    where the profile is least biased by curvature.
    """
    fluid = ~solid
    col = rho.clone()
    col[~fluid] = rho_v
    if column is None:
        column = int(torch.argmax(col.max(dim=0).values))
    prof = col[:, column]
    hi = rho_v + 0.9 * (rho_l - rho_v)
    lo = rho_v + 0.1 * (rho_l - rho_v)
    above_hi = (prof > hi).nonzero().flatten()
    above_lo = (prof > lo).nonzero().flatten()
    if above_hi.numel() == 0 or above_lo.numel() == 0:
        return float("nan")
    return float(above_lo.max() - above_hi.max())


def groove_fill_fraction(
    rho: torch.Tensor,
    channel: TexturedChannel,
    *,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    x_range: tuple[int, int] | None = None,
) -> float:
    """Mean liquid fraction ``(rho - rho_v)/(rho_l - rho_v)`` inside the grooves.

    ``x_range`` restricts the measurement to a column window (the drop
    footprint).  ``NaN`` when the window contains no groove cells.
    """
    groove = channel.groove_mask(rho.device)
    if x_range is not None:
        window = torch.zeros_like(groove)
        window[:, int(x_range[0]) : int(x_range[1])] = True
        groove = groove & window
    if not bool(groove.any()):
        return float("nan")
    phi = (rho[groove] - rho_v) / (rho_l - rho_v)
    return float(torch.clamp(phi, 0.0, 1.0).mean())


def apparent_angle(
    rho: torch.Tensor,
    solid: torch.Tensor,
    y_plane: float,
    *,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    offset: float = 2.0,
    bootstrap: int = 32,
    seed: int = 0,
) -> dict:
    """Apparent contact angle of a drop on the composite plane (physical convention).

    Convention (important — ``theta_fit.contact_angle_from_circle`` returns the
    *supplement* of this one, see the note below): ``theta`` is measured **through
    the liquid**, between the wall plane and the interface tangent at the
    contact line, i.e. ``theta -> 180 deg`` for a complete non-wetting bead and
    ``theta -> 0`` for a spreading film.

    For a spherical cap of radius ``R_s`` with contact angle ``theta`` the centre
    sits at ``y_c = y_plane - R_s cos theta``, so the cap half-width at the plane
    is ``a = R_s sin theta`` and

        theta = 180 deg - atan2(a, y_c - y_plane) ,

    which reduces to ``theta = 2 atan(h/a)`` for the cap height ``h``.  A
    synthetic-cap regression in
    ``tests/test_microtexture_wetting.py::test_apparent_angle_convention`` pins
    this convention to 0.2 deg over ``theta in [30, 160]``.

    The meniscus inside the grooves is removed by keeping only interface points
    at least ``offset`` cells above the composite plane.

    Returns:
        dict with ``theta_deg`` (circle fit), ``theta_shape_deg``
        (``2 atan(h/a)`` cross-check), ``unc_deg`` (bootstrap standard deviation
        of the circle fit), ``theta_bias_deg`` (``|circle - shape|``) and the
        fit diagnostics.
    """
    level = level_fraction(rho_l, rho_v)
    pts = interface_points(rho, level, solid)
    empty = {
        "theta_deg": float("nan"),
        "theta_shape_deg": float("nan"),
        "theta_bias_deg": float("nan"),
        "unc_deg": float("nan"),
        "n_points": 0,
        "rms": float("nan"),
        "radius_fit": float("nan"),
        "cap_height": float("nan"),
        "base_half_width": float("nan"),
    }
    if pts.numel() == 0:
        return dict(empty)
    pts = pts[pts[:, 1] >= y_plane + float(offset)]
    if pts.shape[0] < 8:
        empty["n_points"] = int(pts.shape[0])
        return dict(empty)

    def _circle_angle(points: torch.Tensor) -> tuple[float, float, float]:
        fit = circle_fit(points)
        hc = fit.yc - float(y_plane)
        disc = fit.r * fit.r - hc * hc
        if disc <= 0.0 or not math.isfinite(fit.r):
            return float("nan"), fit.r, fit.rms
        a = math.sqrt(disc)
        return 180.0 - math.degrees(math.atan2(a, hc)), fit.r, fit.rms

    theta_c, r_fit, rms = _circle_angle(pts)

    y = pts[:, 1]
    x = pts[:, 0]
    h_cap = float(y.max()) - float(y_plane)
    base = pts[y <= y_plane + float(offset) + 2.0]
    a_base = 0.5 * float(base[:, 0].max() - base[:, 0].min()) if base.numel() else float("nan")
    theta_shape = (
        2.0 * math.degrees(math.atan2(h_cap, a_base))
        if math.isfinite(a_base) and a_base > 0.0
        else float("nan")
    )

    unc = float("nan")
    if bootstrap > 0:
        g = torch.Generator().manual_seed(int(seed))
        vals = []
        n = pts.shape[0]
        for _ in range(int(bootstrap)):
            idx = torch.randint(0, n, (n,), generator=g)
            th, _, _ = _circle_angle(pts[idx])
            if math.isfinite(th):
                vals.append(th)
        if len(vals) > 2:
            t = torch.tensor(vals)
            unc = float(t.std())

    return {
        "theta_deg": theta_c,
        "theta_shape_deg": theta_shape,
        "theta_bias_deg": abs(theta_c - theta_shape),
        "unc_deg": unc,
        "n_points": int(pts.shape[0]),
        "rms": rms,
        "radius_fit": r_fit,
        "cap_height": h_cap,
        "base_half_width": a_base,
    }


def apparent_angle_n01(
    rho: torch.Tensor,
    solid: torch.Tensor,
    y_plane: float,
    *,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    windows: tuple[float, ...] = (8.0, 10.0, 12.0, 14.0),
    offset: float = 2.0,
    bootstrap: int = 32,
) -> dict:
    """The smooth-wall pipeline estimate (``theta_fit.angle_estimate``), unchanged.

    Kept verbatim so the textured module can reproduce the ``wetting_n01``
    baseline exactly when the texture degenerates to a flat wall (regression
    criterion of the campaign).  Note this estimator inherits the supplement
    convention of :func:`tensorlbm.contact_line.theta_fit.contact_angle_from_circle`.
    """
    level = level_fraction(rho_l, rho_v)
    pts = interface_points(rho, level, solid)
    if pts.numel() == 0:
        return {"theta_deg": float("nan"), "unc_deg": float("nan"), "n_points": 0}
    est = angle_estimate(
        pts, y_plane, side="cap", windows=windows, offset=offset, bootstrap=bootstrap
    )
    return {
        "theta_deg": est.theta_deg,
        "unc_deg": est.unc_deg,
        "n_points": est.n_points,
        "rms": est.rms,
        "radius_fit": est.radius_fit,
    }


def cap_radius(volume_radius: float, theta_deg: float) -> float:
    """Sphere radius whose cap above a plane has the volume of a drop of ``R0``.

    The cap of height ``h = R_s (1 - cos theta)`` has volume
    ``V = (pi h^2/3)(3 R_s - h)``; setting it equal to ``(4/3) pi R0^3`` gives

        R_s = R0 [ 4 / ((1 - cos theta)^2 (2 + cos theta)) ]^(1/3) .
    """
    c = math.cos(math.radians(float(theta_deg)))
    return float(volume_radius) * (4.0 / ((1.0 - c) ** 2 * (2.0 + c))) ** (1.0 / 3.0)


def textured_drop_state(
    channel: TexturedChannel,
    radius: float,
    *,
    center_x: float | None = None,
    height: float | None = None,
    fill_grooves: bool = False,
    init_angle_deg: float | None = None,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    width: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sessile drop tangent to the composite plane, optionally with wetted grooves.

    ``fill_grooves=True`` starts from the *Wenzel* configuration (grooves
    prefilled with liquid), so a hydrophilic texture does not have to drain
    vapour for thousands of steps before equilibrating.

    ``init_angle_deg`` replaces the tangent circle by the volume-conserving
    spherical cap of that contact angle (:func:`cap_radius`).  Starting from the
    predicted angle removes the very slow bulk spreading/contraction transient
    (tens of capillary times for ``R = 40``), so a few thousand steps suffice for
    the interface profile to equilibrate; the initialisation is recorded in the
    result and the residual angle drift is reported so the choice is auditable.
    """
    dev = torch.device(channel.device)
    solid = channel.solid_mask(dev)
    xx, yy = channel.grid(dev)
    cx = channel.nx / 2.0 if center_x is None else float(center_x)
    if init_angle_deg is not None:
        rs = cap_radius(radius, float(init_angle_deg))
        cy = channel.y_wall() + rs * math.cos(math.radians(float(init_angle_deg)))
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - rs
    else:
        cy = channel.y_wall() + (radius if height is None else float(height))
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - float(radius)
    rho = tanh_profile(channel.shape, dist, rho_l, rho_v, width)
    if init_angle_deg is not None:
        # only the part of the sphere above the composite plane is liquid
        above = (yy > channel.y_wall()).to(rho.dtype)
        rho = rho_v + (rho - rho_v) * above
    if fill_grooves:
        groove = channel.groove_mask(dev)
        rho = torch.where(groove, torch.full_like(rho, rho_l), rho)
    rho = torch.where(solid, torch.full_like(rho, rho_l), rho)
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero), rho


# ---------------------------------------------------------------------------
# Static apparent-angle case
# ---------------------------------------------------------------------------


def run_static_angle(
    channel: TexturedChannel,
    *,
    drop_radius: float,
    relax_steps: int = 6000,
    fill_grooves: bool = False,
    init_angle_deg: float | None = None,
    G: float = 5.0,
    tau: float = 1.0,
    forcing: str = "velocity_shift",
    scheme: str = "standard",
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    theta_y_deg: float | None = None,
    checkpoint=None,
    every: int = 2000,
    field_path: str | None = None,
) -> dict:
    """Relax a drop on the texture, then measure apparent angle and state.

    ``checkpoint(step, f)`` is called every ``every`` steps (used by the
    convergence study to record the angle history).
    """
    from .engine import WallStepper

    tex = channel.texture
    theta_y = tex.theta_y_deg if theta_y_deg is None else float(theta_y_deg)
    solid = channel.solid_mask()
    stepper = WallStepper(
        channel,
        G=G,
        tau=tau,
        rho_l=rho_l,
        rho_v=rho_v,
        wall_fraction=tex.wetting_fraction,
        forcing=forcing,
        scheme=scheme,
    )
    f, _ = textured_drop_state(
        channel,
        drop_radius,
        fill_grooves=fill_grooves,
        init_angle_deg=init_angle_deg,
        rho_l=rho_l,
        rho_v=rho_v,
    )
    f = stepper.run(f, int(relax_steps), callback=checkpoint, every=every)

    rho = stepper.density(f)
    if field_path is not None:
        import numpy as _np

        _np.savez_compressed(
            field_path,
            rho=rho.detach().cpu().numpy().astype("float16"),
            solid=channel.solid_mask().detach().cpu().numpy(),
            y_wall=_np.array([channel.y_wall()]),
            rho_l=_np.array([rho_l]),
            rho_v=_np.array([rho_v]),
        )
    cx = channel.nx / 2.0
    x0 = max(0, int(cx - drop_radius - 6))
    x1 = min(channel.nx, int(cx + drop_radius + 6))
    fill = groove_fill_fraction(rho, channel, rho_l=rho_l, rho_v=rho_v, x_range=(x0, x1))
    ang = apparent_angle(rho, solid, channel.y_wall(), rho_l=rho_l, rho_v=rho_v)

    if math.isnan(fill):
        state_obs = "flat"
    elif fill < 0.25:
        state_obs = "cassie"
    elif fill > 0.75:
        state_obs = "wenzel"
    else:
        state_obs = "mixed"

    pred = predicted_wetting_state(
        PillarTexture(
            pitch=tex.pitch,
            pillar_width=tex.pillar_width,
            height=tex.height,
            theta_y_deg=theta_y,
            wall_fraction=tex.wetting_fraction,
            layout=tex.layout,
        )
    )
    return {
        "theta_app_deg": ang["theta_deg"],
        "interface_width_cells": interface_width_cells(rho, solid, rho_l=rho_l, rho_v=rho_v),
        "theta_unc_deg": ang["unc_deg"],
        "theta_n_points": ang["n_points"],
        "theta_shape_deg": ang.get("theta_shape_deg", float("nan")),
        "theta_bias_deg": ang.get("theta_bias_deg", float("nan")),
        "theta_rms": ang.get("rms", float("nan")),
        "radius_fit": ang.get("radius_fit", float("nan")),
        "groove_fill": fill,
        "state_observed": state_obs,
        "state_predicted": pred["state_predicted"],
        "state_match": state_obs == pred["state_predicted"],
        "theta_cb_deg": pred["theta_cb_deg"],
        "theta_w_deg": pred["theta_w_deg"],
        "theta_predicted_deg": pred["theta_predicted_deg"],
        "theta_crossover_deg": pred["theta_crossover_deg"],
        "theta_dev_from_pred_deg": abs(ang["theta_deg"] - pred["theta_predicted_deg"]),
        "drop_radius": float(drop_radius),
        "relax_steps": int(relax_steps),
        "fill_grooves_init": bool(fill_grooves),
        "init_angle_deg": None if init_angle_deg is None else float(init_angle_deg),
        "forcing": forcing,
        "scheme": scheme,
        "texture": {
            "pitch": tex.pitch,
            "pillar_width": tex.pillar_width,
            "height": tex.height,
            "theta_y_nominal_deg": tex.theta_y_deg,
            "theta_y_used_deg": theta_y,
            "wetting_fraction": tex.wetting_fraction,
            "solid_fraction": tex.solid_fraction,
            "roughness": tex.roughness,
        },
    }


# ---------------------------------------------------------------------------
# Quasi-static impalement (Cassie -> Wenzel transition)
# ---------------------------------------------------------------------------


class PressurisedStepper:
    """SCMP stepper with a *liquid-only* downward body force (hydrostatic head).

    The body force is ``rho * g * w(rho)`` with
    ``w = clamp((rho - rho_m)/(rho_l - rho_m), 0, 1)`` so vapour is never pushed:
    a liquid column of height ``h`` exerts the Laplace-scale pressure
    ``p = rho_l g h`` on the suspended meniscus.  Collision, forcing and
    bounce-back are the library kernel.
    """

    def __init__(
        self,
        channel: TexturedChannel,
        *,
        wall_fraction: float,
        G: float = 5.0,
        tau: float = 1.0,
        rho_l: float = RHO_L_DEFAULT,
        rho_v: float = RHO_V_DEFAULT,
        forcing: str = "guo",
        scheme: str = "standard",
    ) -> None:
        from .engine import WallStepper

        self.channel = channel
        self.device = torch.device(channel.device)
        self.G = float(G)
        self.tau = float(tau)
        self.rho_l = float(rho_l)
        self.rho_v = float(rho_v)
        self.rho_mid = 0.5 * (self.rho_l + self.rho_v)
        self.forcing = forcing
        self.scheme = scheme
        self.solid = channel.solid_mask(self.device)
        ref = WallStepper(
            channel,
            G=G,
            tau=tau,
            rho_l=rho_l,
            rho_v=rho_v,
            wall_fraction=wall_fraction,
            forcing=forcing,
            scheme=scheme,
        )
        self.wall_psi = ref.wall_psi

    def liquid_weight(self, rho: torch.Tensor) -> torch.Tensor:
        w = (rho - self.rho_mid) / (self.rho_l - self.rho_mid)
        return torch.clamp(w, 0.0, 1.0)

    def step(self, f: torch.Tensor, g_accel: float) -> torch.Tensor:
        rho = macroscopic(f)[0]
        gy = -float(g_accel) * self.liquid_weight(rho)
        f = collide_sc_single_component(
            f,
            self.G,
            tau=self.tau,
            psi_fn=psi_exp,
            gx=0.0,
            gy=gy,
            solid_mask=self.solid,
            wall_psi=self.wall_psi,
            forcing=self.forcing,
            scheme=self.scheme,
        )
        f = stream(f)
        return torch.where(self.solid.unsqueeze(0), f[OPPOSITE], f)

    def run(self, f: torch.Tensor, n_steps: int, g_accel: float) -> torch.Tensor:
        for _ in range(int(n_steps)):
            f = self.step(f, g_accel)
        return f

    def density(self, f: torch.Tensor) -> torch.Tensor:
        return macroscopic(f)[0]


def column_state(
    channel: TexturedChannel,
    column_height: float,
    *,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    width: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat liquid column standing on the composite plane (grooves start dry)."""
    dev = torch.device(channel.device)
    solid = channel.solid_mask(dev)
    yy = torch.arange(channel.ny, dtype=torch.float32, device=dev).view(-1, 1)
    dist = (channel.y_wall() - float(column_height) - yy).expand(channel.ny, channel.nx)
    rho = tanh_profile(channel.shape, dist, rho_l, rho_v, width)
    # grooves start DRY: only cells above the composite plane belong to the column
    yy_e = torch.arange(channel.ny, dtype=torch.float32, device=dev).view(-1, 1)
    dry = (yy_e < channel.y_wall()).expand(channel.ny, channel.nx)
    rho = torch.where(dry, torch.full_like(rho, rho_v), rho)
    rho = torch.where(solid, torch.full_like(rho, rho_l), rho)
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero), rho


def _liquid_head(rho: torch.Tensor, channel: TexturedChannel, rho_l: float, rho_v: float) -> float:
    """Mean liquid-column height above the composite plane (lattice units)."""
    solid = channel.solid_mask(rho.device)
    fluid = ~solid
    phi = torch.clamp((rho - rho_v) / (rho_l - rho_v), 0.0, 1.0) * fluid
    mass = float(phi.sum())
    if mass <= 0.0:
        return 0.0
    yy = torch.arange(channel.ny, dtype=torch.float32, device=rho.device).view(-1, 1)
    ybar = float((phi * yy).sum()) / mass
    return max(ybar - channel.y_wall(), 0.0)


def run_impalement(
    channel: TexturedChannel,
    *,
    column_height: float,
    g_values: tuple[float, ...],
    steps_per_g: int = 3000,
    wall_fraction: float | None = None,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    forcing: str = "guo",
    scheme: str = "standard",
    fill_threshold: float = 0.5,
    refine: int = 2,
) -> dict:
    """Measure the Cassie -> Wenzel break-through pressure by hydrostatic loading.

    For every body-force amplitude ``g`` the column is independently relaxed from
    the same dry-groove initial state; the groove fill fraction is recorded.  The
    smallest ``g`` whose fill crosses ``fill_threshold`` is the transition point,
    which is then refined by bisection (``refine`` extra runs).  The measured
    pressure is ``p = rho_l g h`` with ``h`` the liquid head at the last stable
    state.
    """
    tex = channel.texture
    wf = tex.wetting_fraction if wall_fraction is None else float(wall_fraction)
    stepper = PressurisedStepper(
        channel,
        wall_fraction=wf,
        rho_l=rho_l,
        rho_v=rho_v,
        forcing=forcing,
        scheme=scheme,
    )
    solid = channel.solid_mask()
    groove = channel.groove_mask()

    def one(g: float) -> dict:
        f, _ = column_state(channel, column_height, rho_l=rho_l, rho_v=rho_v)
        f = stepper.run(f, steps_per_g, g)
        rho = stepper.density(f)
        fill = float(
            torch.clamp((rho[groove] - rho_v) / (rho_l - rho_v), 0.0, 1.0).mean()
        )
        head = _liquid_head(rho, channel, rho_l, rho_v)
        # deepest penetration: highest groove row that stays liquid
        col = groove.nonzero()
        deep = 0.0
        if col.numel():
            phi = torch.clamp((rho[groove] - rho_v) / (rho_l - rho_v), 0.0, 1.0)
            rows = col[:, 0]
            wetted = phi > 0.5
            if bool(wetted.any()):
                deep = float(channel.y_wall() - rows[wetted].min())
        return {
            "g": float(g),
            "groove_fill": fill,
            "liquid_head": head,
            "p_head": rho_l * float(g) * head,
            "penetration_depth": deep,
            "impaled": fill > fill_threshold,
        }

    history = [one(g) for g in g_values]
    stable = [h for h in history if not h["impaled"]]
    unstable = [h for h in history if h["impaled"]]
    transition: dict | None = None
    if stable and unstable:
        g_lo = max(h["g"] for h in stable)
        g_hi = min(h["g"] for h in unstable)
        h_lo = max(stable, key=lambda h: h["g"])
        for _ in range(int(refine)):
            g_mid = 0.5 * (g_lo + g_hi)
            rec = one(g_mid)
            history.append(rec)
            if rec["impaled"]:
                g_hi = g_mid
            else:
                g_lo = g_mid
                h_lo = rec
        transition = {
            "g_stable": g_lo,
            "g_unstable": g_hi,
            "g_threshold": 0.5 * (g_lo + g_hi),
            "liquid_head": h_lo["liquid_head"],
            "p_c_measured": rho_l * 0.5 * (g_lo + g_hi) * h_lo["liquid_head"],
        }

    closed = breakthrough_pressure(tex)
    out = {
        "texture": tex.to_dict(),
        "column_height": float(column_height),
        "steps_per_g": int(steps_per_g),
        "fill_threshold": float(fill_threshold),
        "history": history,
        "transition": transition,
        "closed_form": closed,
        "forcing": forcing,
        "scheme": scheme,
    }
    if transition is not None:
        ratio = transition["p_c_measured"] / closed["p_c_abs_2d"]
        out["ratio_to_laplace_2d"] = ratio
        out["ratio_to_force_balance_3d"] = transition["p_c_measured"] / closed["p_c_abs_3d"]
        out["ratio_in_band"] = bool(0.5 <= ratio <= 2.0)
    return out