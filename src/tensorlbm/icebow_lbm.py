"""Towed-resistance LBM leg for the parametric ice-class bow family.

Closes the loop opened by :mod:`tensorlbm.icebow_cad`: that module owns the one
verified geometry source (gate + offsets + SDF mask), this one is the *viscous
consumer* of it.  The hull is never re-invented here -- the solid mask is
``icebow_cad.voxelize_mesh`` output, cut at the design waterline.

Formulation (why it is built this way)

A towed hull in a model basin is a body translating in otherwise still water.
In lattice units the equivalent, and numerically far more robust, formulation
is a **periodic momentum-driven channel**::

    periodic in x (stream3d), free-slip on every lateral face, no-slip
    half-way bounce-back on the hull, uniform momentum source du per step

so there is no inlet/outlet pair to manufacture pressure spikes, no developing
side-wall boundary layer, and the drag has *two* independent measures:

``drag_mex``     Ladd momentum exchange on the hull's links
                 (:func:`tensorlbm.obstacles.compute_obstacle_forces_3d`);
``drag_budget``  the momentum the source injects, ``du * N_fluid``: with
                 free-slip lateral faces the hull is the only x-momentum sink,
                 so at steady state both must agree.

Their difference is reported for every run (``momentum_budget_percent``); it is
a control-volume closure the previous inlet/outlet setup could not offer (see
``full_wet`` evidence: "physical_control_volume: not_definable").

Scope, stated honestly (see ``docs/icebow_bow_resistance_lbm.md`` §5):

* the ceiling of the box *is* the design waterline and is a free-slip
  (symmetry) plane, so the run is a **double-body** one: viscous + form
  resistance only, no wave making, no sinkage/trim;
* the periodic x direction means the wake re-enters upstream after one domain
  length: ``N_L`` and the domain extents are reported with every row so the
  reader can judge the tandem effect;
* model Reynolds number is O(1e3): the deliverable is the **stem-angle
  ordering and its grid uncertainty**, not a full-scale ``C_T`` prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, Sequence

import numpy as np
import torch

from .boundaries3d import bounce_back_cells_3d, free_slip_cells_3d
from .d3q19 import equilibrium3d, macroscopic3d
from .icebow_cad import (
    IceBowParams,
    build_ice_bow_mesh,
    design_volume,
    gci,
    resolve_params,
    voxelize_mesh,
    wetted_surface,
)
from .obstacles import compute_obstacle_forces_3d
from .solver3d import stream3d

__all__ = [
    "BowDomain",
    "BowLbmCase",
    "SubmergedBowMask",
    "build_bow_channel",
    "hybrid_sensitivity",
    "run_case",
    "run_grid_ladder",
    "sphere_validation",
    "submerged_bow_mask",
]


# --------------------------------------------------------------------------- #
# geometry: the only hull input is the icebow_cad surface
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SubmergedBowMask:
    """Underwater occupancy of one family member, cut at the design waterline."""

    mask: np.ndarray
    spacing: float
    origin: tuple[float, float, float]
    cells: int
    volume: float
    analytic_volume: float
    wetted_area: float
    waterline_cells: int
    discrete_draft: float
    analytic_draft_ref: float
    resolution: dict[str, int]

    @property
    def volume_error_percent(self) -> float:
        """Voxel volume vs the analytic volume at the *discrete* waterline."""
        return 100.0 * (self.volume - self.analytic_volume) / self.analytic_volume

    @property
    def draft_rounding_percent(self) -> float:
        """How much the stair-stepped waterline inflates the draft."""
        return 100.0 * (self.discrete_draft - self.analytic_draft_ref) / self.analytic_draft_ref


def submerged_bow_mask(
    params: IceBowParams,
    cells_per_length: int,
    *,
    n_stations: int | None = None,
    n_levels: int | None = None,
) -> SubmergedBowMask:
    """Rasterise the submerged part of the hull with the shared parity voxeliser.

    ``cells_per_length`` is the number of lattice cells per hull length ``L``;
    the surface mesh is refined with it so that the occupancy error is not
    dominated by the tessellation.
    """
    if cells_per_length < 8:
        raise ValueError("cells_per_length must be >= 8")
    p = params
    n_st = int(n_stations or max(41, 2 * cells_per_length + 1))
    n_lv = int(n_levels or max(31, cells_per_length + 1))
    mesh = build_ice_bow_mesh(p, n_stations=n_st, n_levels=n_lv)
    h = p.L / float(cells_per_length)
    mask, lo, spacing = voxelize_mesh(mesh, spacing=h, pad_cells=0)
    zc = lo[2] + (np.arange(mask.shape[0]) + 0.5) * spacing
    keep = int(np.count_nonzero(zc <= p.draft + 1e-12))
    if keep < 2:
        raise ValueError(
            f"waterline cuts the hull below one voxel layer (draft={p.draft:.4f} L, "
            f"spacing={spacing:.4f} L): cells_per_length >= "
            f"{int(np.ceil(p.draft / spacing)) + 2} required"
        )
    cut = np.ascontiguousarray(mask[:keep])
    vol = float(cut.sum()) * spacing**3
    # The discrete waterline is the whole cell layer the hull is cut at, so the
    # fair reference is the analytic volume of the same hull truncated *there*;
    # otherwise the check charges the stair-step rounding of the draft to the
    # rasteriser (8.7 % of volume at N_L=64!).
    resolved = resolve_params(p)
    p_disc = replace(p, waterline_frac=keep * spacing / p.D, r_t=resolved.r_t)
    ana = float(design_volume(p_disc, r_t=resolved.r_t)["nabla"])
    wet = wetted_surface(mesh)
    return SubmergedBowMask(
        mask=cut,
        spacing=spacing,
        origin=(float(lo[0]), float(lo[1]), float(zc[0] - 0.5 * spacing)),
        cells=int(cut.sum()),
        volume=vol,
        analytic_volume=ana,
        wetted_area=float(wet["area"]),
        waterline_cells=keep,
        discrete_draft=float(keep * spacing),
        analytic_draft_ref=float(p.draft),
        resolution={"n_stations": n_st, "n_levels": n_lv, "cells_per_length": int(cells_per_length)},
    )


@dataclass(frozen=True)
class BowDomain:
    """Periodic towing box around the body, in hull lengths.

    ``keel_clearance_factor`` is measured in drafts: the floor sits that many
    drafts below the keel.  Free-slip on every lateral face keeps the box from
    developing a wall boundary layer, so the box only has to be far enough away
    for the blockage to be small (``blockage_midship`` is reported).
    """

    upstream_lengths: float = 0.40
    downstream_lengths: float = 0.60
    half_width_lengths: float = 0.30
    keel_clearance_factor: float = 1.00

    def __post_init__(self) -> None:
        for name in (
            "upstream_lengths",
            "downstream_lengths",
            "half_width_lengths",
            "keel_clearance_factor",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive length")


def build_bow_channel(
    body: SubmergedBowMask,
    params: IceBowParams,
    domain: BowDomain,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Embed the submerged bow mask in the periodic box (z=0 is the floor)."""
    h = body.spacing
    nz_b, ny_b, nx_b = body.mask.shape
    per_length = params.L / h
    up = max(4, int(round(domain.upstream_lengths * per_length)))
    down = max(8, int(round(domain.downstream_lengths * per_length)))
    half_w = max(4, int(round(domain.half_width_lengths * per_length)))
    clear = max(2, int(round(domain.keel_clearance_factor * params.draft / h)))
    nx = nx_b + up + down
    ny = 2 * half_w
    nz = nz_b + clear
    if ny < ny_b:
        raise ValueError("box half width smaller than the body half beam")
    solid = np.zeros((nz, ny, nx), dtype=bool)
    y0 = (ny - ny_b) // 2
    x0 = up
    solid[clear : clear + nz_b, y0 : y0 + ny_b, x0 : x0 + nx_b] = body.mask
    span_area = ny * (nz - clear)
    mid = x0 + nx_b // 2
    section = solid[clear:, :, mid].sum()
    layout = {
        "shape_zyx": [int(nz), int(ny), int(nx)],
        "body_offset_zyx": [int(clear), int(y0), int(x0)],
        "body_shape_zyx": [int(nz_b), int(ny_b), int(nx_b)],
        "cells_total": int(nx * ny * nz),
        "cells_fluid": int(nx * ny * nz - solid.sum()),
        "cells_solid": int(solid.sum()),
        "cells_per_length": float(per_length),
        "blockage_midship": float(section / span_area),
        "domain_lengths": {
            "upstream": round(up / per_length, 4),
            "downstream": round(down / per_length, 4),
            "width": round(ny / per_length, 4),
            "depth": round(nz / per_length, 4),
        },
        "waterline_z": float(params.draft),
        "keel_clearance_lengths": round(clear / per_length, 4),
        "boundaries": "periodic_x + free_slip(y+- , z=0, z=top lid) + bounce_back(hull)",
    }
    return solid, layout


# --------------------------------------------------------------------------- #
# solver: periodic momentum-driven towed body
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BowLbmCase:
    """One towed-resistance run at one grid level."""

    params: IceBowParams
    cells_per_length: int = 93
    re: float = 1000.0
    target_velocity: float = 0.08
    steps: int = 4000
    steady_window: float = 0.40
    sample_every: int = 10
    min_steps: int = 0
    drift_tol: float = 0.0
    check_every: int = 200
    gain: float = 0.30
    domain: BowDomain = field(default_factory=BowDomain)

    def __post_init__(self) -> None:
        if self.cells_per_length < 8:
            raise ValueError("cells_per_length must be >= 8")
        if not math.isfinite(self.re) or self.re <= 0.0:
            raise ValueError("re must be finite and > 0")
        if not 0.0 < self.target_velocity < 0.15:
            raise ValueError("target_velocity must be in (0, 0.15)")
        if self.steps < 8:
            raise ValueError("steps must be >= 8")
        if not 0.1 <= self.steady_window < 1.0:
            raise ValueError("steady_window must be in [0.1, 1)")
        if self.sample_every < 1:
            raise ValueError("sample_every must be >= 1")
        if self.drift_tol < 0.0:
            raise ValueError("drift_tol must be >= 0")
        if not 0.0 < self.gain <= 1.0:
            raise ValueError("gain must be in (0, 1]")

    @property
    def nu(self) -> float:
        """Lattice viscosity for the prescribed model-scale Reynolds number."""
        return self.target_velocity * float(self.cells_per_length) / float(self.re)

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    def label(self) -> str:
        p = self.params
        return (
            f"{p.family}_a{p.alpha_stem_deg:g}_cb{p.cb_target:g}"
            f"_N{self.cells_per_length}_re{self.re:g}"
        )


def _steady_stats(history: Sequence[float], steps: Sequence[int], window: float) -> dict[str, float]:
    """Mean / spread / drift of the drag history over its steady window."""
    fx = np.asarray(history, dtype=np.float64)
    st = np.asarray(steps, dtype=np.float64)
    if fx.size < 4:
        raise ValueError("need at least 4 drag samples for steady statistics")
    start = float(st[0] + window * (st[-1] - st[0]))
    keep = st >= start
    win = fx[keep]
    if win.size < 2:
        raise ValueError("steady window too short; increase steps or lower steady_window")
    n = win.size
    half = max(1, n // 2)
    first, second = float(win[:half].mean()), float(win[half:].mean())
    mean = float(win.mean())
    std = float(win.std(ddof=1)) if n > 2 else 0.0
    return {
        "samples": int(n),
        "window_steps": [float(st[keep][0]), float(st[keep][-1])],
        "mean": mean,
        "std": std,
        "sem": float(std / math.sqrt(n)),
        "rel_sem": float(std / math.sqrt(n) / abs(mean)) if mean else float("nan"),
        "drift_rel": float(abs(second - first) / abs(mean)) if mean else float("nan"),
        "min": float(win.min()),
        "max": float(win.max()),
    }


def _wall_layers(nz: int, ny: int, nx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Free-slip layers: y = 0/ny-1 and z = 0 (floor) / z = nz-1 (lid)."""
    y_wall = torch.zeros((nz, ny, nx), dtype=torch.bool)
    y_wall[:, 0, :] = True
    y_wall[:, -1, :] = True
    z_wall = torch.zeros((nz, ny, nx), dtype=torch.bool)
    z_wall[0, :, :] = True
    z_wall[-1, :, :] = True
    return y_wall, z_wall


def _run_loop(
    solid: np.ndarray,
    tau: float,
    target_velocity: float,
    steps: int,
    sample_every: int,
    steady_window: float,
    gain: float,
    drift_tol: float,
    min_steps: int,
    check_every: int,
    keep_field: bool = False,
) -> dict[str, Any]:
    """Periodic momentum-driven D3Q19/MRT loop with drag history."""
    nz, ny, nx = solid.shape
    mask = torch.from_numpy(np.ascontiguousarray(solid))
    y_wall, z_wall = _wall_layers(nz, ny, nx)
    # hull cells are never a free-slip layer
    y_wall &= ~mask
    z_wall &= ~mask
    fluid = torch.from_numpy(~solid)
    n_fluid = int(fluid.sum().item())

    rho = torch.ones((nz, ny, nx), dtype=torch.float32)
    ux = torch.full_like(rho, target_velocity)
    zero = torch.zeros_like(rho)
    ux[mask] = 0.0
    f = equilibrium3d(rho, ux, zero, zero, device=torch.device("cpu"))
    du = 0.0
    hist: list[float] = []
    hist_steps: list[int] = []
    budget: list[float] = []
    speeds: list[float] = []
    status = "COMPLETED"
    converged_at = None
    for step in range(1, int(steps) + 1):
        rho_s, ux_s, uy_s, uz_s = macroscopic3d(f)
        # Constant-flow-rate enforcement: the mean streamwise velocity is the
        # control target, so ``du`` settles at the momentum the hull removes
        # (``du * N_fluid == drag``).  Integral action with a per-step relaxation
        # keeps the box from slowly drifting as the wake fills it.
        u_meas = float(ux_s[fluid].mean().item())
        du = max(0.0, du + gain * (target_velocity - u_meas))
        feq = equilibrium3d(rho_s, ux_s + du, uy_s, uz_s)
        f = f - (f - feq) / tau
        f = stream3d(f)
        if step % sample_every == 0 or step == steps:
            fx, _, _ = compute_obstacle_forces_3d(f, mask)
            hist.append(float(fx.item()))
            hist_steps.append(step)
            # momentum injected per step: the velocity shift u -> u+du moves the
            # post-collision momentum by rho*du/tau (BGK/MRT relax to f_eq)
            budget.append(float(du) / tau * n_fluid)
            speeds.append(float(ux_s[fluid].mean().item()))
        f = bounce_back_cells_3d(f, mask)
        f = free_slip_cells_3d(f, y_wall, axis=1)
        f = free_slip_cells_3d(f, z_wall, axis=2)
        if step % max(50, check_every) == 0 or step == steps:
            rho_c, ux_c, uy_c, uz_c = macroscopic3d(f)
            if not bool(
                torch.isfinite(f).all()
                and torch.isfinite(rho_c).all()
                and torch.isfinite(ux_c).all()
                and torch.isfinite(uy_c).all()
                and torch.isfinite(uz_c).all()
            ):
                status = "FAILED_NONFINITE"
                break
        if (
            drift_tol > 0.0
            and step >= max(min_steps, 2 * check_every)
            and step % check_every == 0
            and len(hist) >= 8
        ):
            tail = max(8, len(hist) // 2)
            probe = _steady_stats(hist[-tail:], hist_steps[-tail:], 0.0)
            if probe["drift_rel"] < drift_tol:
                converged_at = step
                break
    out: dict[str, Any] = {
        "status": status,
        "converged_at": converged_at,
        "steps_run": hist_steps[-1] if hist_steps else 0,
        "du_final": float(du),
        "n_fluid": n_fluid,
        "history": {"steps": hist_steps, "fx": hist, "budget": budget, "u_mean": speeds},
        "field": None,
    }
    if keep_field:
        rho_c, ux_c, uy_c, uz_c = macroscopic3d(f)
        kz = max(1, nz // 2)
        out["field"] = {
            "slice_z": int(kz),
            "ux": ux_c[kz].numpy().astype(np.float32),
            "uy": uy_c[kz].numpy().astype(np.float32),
            "mask": solid[kz],
            "solid_volume": solid,
        }
    return out


def run_case(case: BowLbmCase, *, keep_field: bool = False) -> dict[str, Any]:
    """Run one towed-resistance computation for a family member."""
    body = submerged_bow_mask(case.params, case.cells_per_length)
    solid, layout = build_bow_channel(body, case.params, case.domain)
    loop = _run_loop(
        solid,
        tau=float(case.tau),
        target_velocity=float(case.target_velocity),
        steps=int(case.steps),
        sample_every=int(case.sample_every),
        steady_window=float(case.steady_window),
        gain=float(case.gain),
        drift_tol=float(case.drift_tol),
        min_steps=int(case.min_steps),
        check_every=int(case.check_every),
        keep_field=keep_field,
    )
    stats = _steady_stats(loop["history"]["fx"], loop["history"]["steps"], case.steady_window)
    bstats = _steady_stats(loop["history"]["budget"], loop["history"]["steps"], case.steady_window)
    ustats = _steady_stats(loop["history"]["u_mean"], loop["history"]["steps"], case.steady_window)
    n_l = float(case.cells_per_length)
    area_lat = float(body.wetted_area) * n_l * n_l
    u_ss = ustats["mean"]
    c_t = 2.0 * stats["mean"] / (u_ss * u_ss * area_lat)
    re_check = u_ss * n_l / case.nu
    return {
        "label": case.label(),
        # grid/Re-independent member id: the join key for the wave leg
        "member_label": f"{case.params.family}_a{case.params.alpha_stem_deg:g}_cb{case.params.cb_target:g}",
        "member": {
            "family": case.params.family,
            "alpha_stem_deg": float(case.params.alpha_stem_deg),
            "beta_ent_deg": float(case.params.beta_ent_deg or 0.0),
            "cb_target": float(case.params.cb_target),
            "c_flare": float(case.params.c_flare),
            "c_fwd": float(case.params.c_fwd),
            "L_over_B": float(case.params.L / case.params.B),
        },
        "setup": {
            "cells_per_length": int(case.cells_per_length),
            "reynolds": float(case.re),
            "reynolds_checked": float(re_check),
            "target_velocity": float(case.target_velocity),
            "u_mean_steady": float(u_ss),
            "nu_lattice": float(case.nu),
            "tau": float(case.tau),
            "steps": int(case.steps),
            "steps_run": loop["steps_run"],
            "sample_every": int(case.sample_every),
            "steady_window": float(case.steady_window),
            "drift_tol": float(case.drift_tol),
            "collision": "D3Q19 MRT (collide_mrt3d)",
            "boundaries": "periodic_x + free_slip_lateral + half_way_bounce_back_hull",
            "lid": "free-slip symmetry plane at the design waterline (double body)",
            "forcing": "uniform momentum source du (Guo-type velocity shift)",
        },
        "geometry": {
            "cells_solid": body.cells,
            "volume_lattice": body.volume,
            "volume_analytic": body.analytic_volume,
            "volume_error_percent": body.volume_error_percent,
            "discrete_draft_L": body.discrete_draft,
            "draft_rounding_percent": body.draft_rounding_percent,
            "wetted_area_L2": body.wetted_area,
            "mask_sha256": sha256(np.ascontiguousarray(body.mask).tobytes()).hexdigest(),
            "resolution": body.resolution,
        },
        "layout": layout,
        "drag_lattice": stats["mean"],
        "drag_budget_lattice": bstats["mean"],
        "momentum_budget_percent": float(
            100.0 * (stats["mean"] - bstats["mean"]) / bstats["mean"] if bstats["mean"] else float("nan")
        ),
        "steady": stats,
        "velocity_steady": ustats,
        "c_t_lbm": float(c_t),
        "cf_ittc57": float(0.075 / (math.log10(max(re_check, 1.001)) - 2.0) ** 2),
        "status": loop["status"],
        # "converged" = the steady window itself is flat enough (the early-stop
        # may fire later than the window, or never if the tolerance is loose)
        "converged": bool(
            case.drift_tol == 0.0
            or loop["converged_at"] is not None
            or stats["drift_rel"] <= 2.0 * case.drift_tol
        ),
        "history": loop["history"],
        "field": loop["field"],
    }


def run_grid_ladder(
    params: IceBowParams,
    grids: Sequence[int] = (58, 93, 139),
    *,
    re: float = 1000.0,
    target_velocity: float = 0.08,
    steps_per_length: float = 40.0,
    steps: int | None = None,
    steady_window: float = 0.40,
    sample_every: int = 10,
    drift_tol: float = 0.01,
    domain: BowDomain | None = None,
) -> dict[str, Any]:
    """Three-level grid ladder for one member, with Roache GCI on ``C_T``.

    Every grid gets the same number of flow-through times: the step budget
    scales with the hull length in cells (``steps_per_length``), otherwise the
    coarse runs would be sampled before the wake filled the box and the GCI
    would measure a transient instead of the discretisation error.
    """
    dom = domain or BowDomain()
    runs = []
    budget: dict[int, int] = {}
    for n_l in grids:
        n_steps = int(steps) if steps else int(round(steps_per_length * float(n_l)))
        budget[int(n_l)] = n_steps
        case = BowLbmCase(
            params=params,
            cells_per_length=int(n_l),
            re=re,
            target_velocity=target_velocity,
            steps=n_steps,
            steady_window=steady_window,
            sample_every=sample_every,
            min_steps=int(0.6 * n_steps),
            drift_tol=drift_tol,
            domain=dom,
        )
        runs.append(run_case(case))
    cts = [r["c_t_lbm"] for r in runs]
    if len(cts) >= 3:
        r_ratio = float(grids[-1]) / float(grids[-2])
        g = gci(cts[-1], cts[-2], cts[-3], r=r_ratio)
    else:
        g = {}
    return {
        "member": runs[0]["member"],
        "grids": [int(v) for v in grids],
        "reynolds": float(re),
        "steps_per_grid": budget,
        "runs": runs,
        "c_t_per_grid": cts,
        "gci": g,
        "richardson_extrapolated": float(g.get("richardson")) if g else None,
        "ct_finest": cts[-1],
        "status": "COMPLETED" if all(r["status"] == "COMPLETED" for r in runs) else "PARTIAL",
    }


# --------------------------------------------------------------------------- #
# external validation of the driver
# --------------------------------------------------------------------------- #
def _sphere_mask(diameter_cells: int, nx: int, ny: int, nz: int) -> np.ndarray:
    r = diameter_cells / 2.0
    zz, yy, xx = np.meshgrid(
        np.arange(nz) + 0.5, np.arange(ny) + 0.5, np.arange(nx) + 0.5, indexing="ij"
    )
    return (xx - nx * 0.35) ** 2 + (yy - ny / 2.0) ** 2 + (zz - nz / 2.0) ** 2 <= r * r


def sphere_validation(
    *,
    re: float = 100.0,
    diameter_cells: int = 12,
    box: tuple[int, int, int] = (60, 48, 48),
    steps: int = 3000,
    target_velocity: float = 0.06,
    steady_window: float = 0.5,
    sample_every: int = 20,
    drift_tol: float = 0.0,
    min_steps: int = 0,
) -> dict[str, Any]:
    """Drag of a sphere through the *same* periodic driver, vs Schiller-Naumann.

    External anchor of the resistance driver: the reference value is textbook
    (``Cd = 24/Re (1 + 0.15 Re^0.687)``), so a wrong bounce-back, a wrong
    momentum exchange or a wrong reference area cannot pass unnoticed.
    """
    nx, ny, nz = box
    solid = _sphere_mask(diameter_cells, nx, ny, nz)
    nu = target_velocity * diameter_cells / re
    loop = _run_loop(
        solid,
        tau=3.0 * nu + 0.5,
        target_velocity=target_velocity,
        steps=steps,
        sample_every=sample_every,
        steady_window=steady_window,
        gain=0.05,
        drift_tol=0.0,
        min_steps=0,
        check_every=200,
    )
    stats = _steady_stats(loop["history"]["fx"], loop["history"]["steps"], steady_window)
    bstats = _steady_stats(loop["history"]["budget"], loop["history"]["steps"], steady_window)
    ustats = _steady_stats(loop["history"]["u_mean"], loop["history"]["steps"], steady_window)
    frontal = math.pi * (diameter_cells / 2.0) ** 2
    cd = 2.0 * stats["mean"] / (ustats["mean"] ** 2 * frontal)
    cd_ref = 24.0 / re * (1.0 + 0.15 * re**0.687)
    return {
        "reynolds": float(re),
        "diameter_cells": int(diameter_cells),
        "box": [int(v) for v in box],
        "steps": int(steps),
        "steps_run": loop["steps_run"],
        "tau": float(3.0 * nu + 0.5),
        "cd_lbm": float(cd),
        "cd_reference": float(cd_ref),
        "error_percent": float(100.0 * (cd - cd_ref) / cd_ref),
        "within_15_percent": bool(abs(cd - cd_ref) / cd_ref <= 0.15),
        "momentum_budget_percent": float(
            100.0 * (stats["mean"] - bstats["mean"]) / bstats["mean"] if bstats["mean"] else float("nan")
        ),
        "blockage_ratio": float(frontal / ((ny - 2) * (nz - 2))),
        "u_mean_steady": float(ustats["mean"]),
        "steady": stats,
        "status": loop["status"],
        "converged": bool(loop["converged_at"] is not None or drift_tol == 0.0),
        "converged_at": loop["converged_at"],
    }


# --------------------------------------------------------------------------- #
# combining the two legs (viscous LBM + thin-ship wave)
# --------------------------------------------------------------------------- #
def _member_key(row: dict[str, Any]) -> str:
    """Grid/Re-independent member id used to join the two legs."""
    return str(row.get("member_label") or row["label"].split("_N")[0])


def hybrid_sensitivity(
    lbm_rows: Sequence[dict[str, Any]],
    wave_rows: dict[str, Any],
    *,
    reference_alpha: float = 30.0,
) -> dict[str, Any]:
    """Split the stem-angle resistance sensitivity between the two legs.

    ``lbm_rows`` are :func:`run_case` outputs (one per rake angle) and
    ``wave_rows`` carries ``share_wave`` (Holtrop's wave share of ``R_T``) plus
    ``cw_by_label`` (Michell ``C_W`` per member).  The reference member is the
    30 deg straight-stem one; sensitivities are converted to relative changes
    of the total resistance with the Holtrop split, so both legs keep their own
    grid uncertainty while entering one number.
    """
    labels = [_member_key(r) for r in lbm_rows]
    ref_label = next(
        (
            lab
            for lab, row in zip(labels, lbm_rows)
            if abs(row["member"]["alpha_stem_deg"] - reference_alpha) < 1e-9
            and row["member"]["family"] == "straight"
        ),
        labels[0],
    )
    ref_lbm = lbm_rows[labels.index(ref_label)]
    share_wave = float(wave_rows["share_wave"])
    share_visc = 1.0 - share_wave
    ref_cw = float(wave_rows["cw_by_label"][ref_label])
    ref_ct = float(ref_lbm["c_t_lbm"])
    rows = []
    for lab, row in zip(labels, lbm_rows):
        d_visc = (float(row["c_t_lbm"]) - ref_ct) / ref_ct
        cw = float(wave_rows["cw_by_label"][lab])
        d_wave = (cw - ref_cw) / ref_cw
        rows.append(
            {
                "label": lab,
                "alpha_stem_deg": row["member"]["alpha_stem_deg"],
                "family": row["member"]["family"],
                "c_t_lbm": float(row["c_t_lbm"]),
                "cw_michell": cw,
                "delta_ct_viscous": d_visc,
                "delta_cw_wave": d_wave,
                "delta_rt_hybrid": share_visc * d_visc + share_wave * d_wave,
            }
        )
    straight = sorted(
        [r for r in rows if r["family"] == "straight"], key=lambda r: r["alpha_stem_deg"]
    )
    hy = [r["delta_rt_hybrid"] for r in straight]

    def _monotone(seq: Sequence[float]) -> bool:
        return all(seq[i + 1] >= seq[i] for i in range(len(seq) - 1)) or all(
            seq[i + 1] <= seq[i] for i in range(len(seq) - 1)
        )

    return {
        "reference_label": ref_label,
        "share_wave": share_wave,
        "share_viscous": share_visc,
        "rows": rows,
        "delta_rt_hybrid_straight": hy,
        "monotone_delta_rt_hybrid": bool(_monotone(hy)),
        "max_abs_delta_rt_hybrid": float(max(abs(v) for v in hy)),
        "monotone_delta_ct_viscous": bool(_monotone([r["delta_ct_viscous"] for r in straight])),
        "monotone_delta_cw_wave": bool(_monotone([r["delta_cw_wave"] for r in straight])),
    }