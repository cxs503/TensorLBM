"""Körner free-surface LBM driver for the dam-break verification face.

One driver, two geometries:

* **planar (2-D) dam break** -- a thin slab ``nz`` cells wide with *free-slip*
  spanwise walls, so the span is a symmetry direction and the run represents the
  planar dam break.  The repository's D2Q9 Körner module
  (:mod:`tensorlbm.free_surface_lbm_2d`) is not used here: it is documented as
  unstable/not mass conserving for this case
  (``benchmarks/pending/dam_break_sc/REPORT_SCMP_vs_FS2D.md`` records NaN with a
  physical gas density and a 41% mass-ledger drift plus a 232% inventory drift
  at ``rho_gas = 1``), so the *production* D3Q19 Körner model
  (:mod:`tensorlbm.free_surface_lbm`, the implementation guarded by the free-
  surface test suite) is used instead.
* **3-D dam break** -- a square column ``a x a x h0`` collapsing in a box; the
  same driver with ``nz = span_cells`` walls no-slip.

Everything recorded here is a *measurement*: the driver does not tune
parameters to hit a gate.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# Small grids are dominated by thread-launch overhead on a many-core host; a
# modest pool is measurably faster.  Override with TL_THREADS.
torch.set_num_threads(int(os.environ.get("TL_THREADS", "8")))

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    free_surface_step,
    init_fill_rectangular,
    init_flags_from_fill,
    init_mass_from_fill,
    total_liquid_inventory,
)


@dataclass
class FSConfig:
    """Geometry and physics of one free-surface dam-break run (lattice units)."""

    h0: int
    """Water depth of the initial column [cells]."""

    c0: float = 0.08
    """Target gravity-wave celerity ``sqrt(g h0)`` (Mach 0.14 at ``c_s=1/sqrt(3)``)."""

    tau: float = 0.52
    """BGK relaxation time; ``nu = (tau - 0.5)/3`` (ignored when ``reynolds_target`` is set)."""

    reynolds_target: float | None = None
    """When set, ``tau`` is derived per grid so ``Re = c0 h0 / nu`` is grid independent.

    A refinement study must hold the Reynolds number fixed; fixing ``tau``
    instead would make ``Re`` grow with the grid and mix discretisation error
    with a physical (viscosity) change.
    """

    column_width: int | None = None
    """Streamwise width of the column (default ``h0``)."""

    span_cells: int = 4
    """Spanwise extent (``nz``); 4 + free-slip walls = planar (2-D) run."""

    span_free_slip: bool = True
    """Free-slip (specular) on the spanwise walls (planar symmetry)."""

    domain_x: float = 4.0
    """Streamwise domain length in units of ``h0``."""

    domain_y: float = 2.0
    """Vertical domain height in units of ``h0``."""

    span_ratio: float = 1.0
    """Spanwise extent in units of ``h0`` when ``span_free_slip`` is False."""

    rho_gas: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def g(self) -> float:
        return self.c0**2 / self.h0

    @property
    def nu(self) -> float:
        if self.reynolds_target:
            return self.c0 * self.h0 / self.reynolds_target
        return (self.tau - 0.5) / 3.0

    @property
    def tau_effective(self) -> float:
        """Relaxation time actually used (from ``reynolds_target`` when given)."""
        return 3.0 * self.nu + 0.5

    @property
    def reynolds(self) -> float:
        """``Re = c0 h0 / nu`` (the dam-break Reynolds number of the run)."""
        return self.c0 * self.h0 / self.nu

    def grid(self) -> tuple[int, int, int]:
        nx = int(round(self.domain_x * self.h0))
        ny = int(round(self.domain_y * self.h0))
        nz = self.span_cells if self.span_free_slip else int(round(self.span_ratio * self.h0))
        return max(nz, 1), max(ny, 3), max(nx, 3)


def _spanwise_free_slip_mask(solid: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    """Wall cells on the two spanwise (``z``) faces, for specular reflection."""
    mask = torch.zeros_like(solid)
    mask[0, :, :] = solid[0, :, :]
    if solid.shape[0] > 1:
        mask[-1, :, :] = solid[-1, :, :]
    return mask


def run(config: FSConfig, *, checkpoints: tuple[float, ...], device: str = "cpu") -> dict[str, Any]:
    """Integrate the free-surface dam break and record the audit quantities.

    Returns a dict with the grid, the step count, the mass/inventory drift time
    series (relative), the depth profiles at the requested ``t*`` checkpoints
    and the final water-surface maps.
    """
    nz, ny, nx = config.grid()
    column_width = config.column_width if config.column_width is not None else config.h0
    fill, solid = init_fill_rectangular(nz, ny, nx, column_width, config.h0, device)
    flags = init_flags_from_fill(fill, solid)
    rho0 = torch.where(flags == LIQUID, torch.ones_like(fill), torch.zeros_like(fill))
    f = equilibrium3d(rho0, torch.zeros_like(fill), torch.zeros_like(fill), torch.zeros_like(fill))
    mass = init_mass_from_fill(fill, flags, 1.0)

    slip_mask = _spanwise_free_slip_mask(solid, flags) if config.span_free_slip else None

    mass0 = float(mass.sum())
    inv0 = float(total_liquid_inventory(f, fill, flags, 1.0))
    drift: list[float] = [0.0]
    inv_drift: list[float] = [0.0]

    t_star_scale = config.c0 / config.h0  # t* = step * c0 / h0
    n_steps = int(round(max(checkpoints) / t_star_scale))
    wanted = {round(cp / t_star_scale): cp for cp in checkpoints}
    profiles: dict[str, dict[str, Any]] = {}
    nan_step: int | None = None
    snapshots: dict[str, Any] = {}

    for step in range(1, n_steps + 1):
        f, fill, flags, mass, _ = free_surface_step(
            f,
            fill,
            flags,
            solid,
            mass=mass,
            tau=config.tau_effective,
            gy=-config.g,
            rho_liquid=1.0,
            rho_gas=config.rho_gas,
            free_slip_mask=slip_mask,
        )
        if not math.isfinite(float(mass.sum())):
            nan_step = step
            break
        drift.append((float(mass.sum()) - mass0) / mass0)
        if step % 5 == 0 or step in wanted:
            inv_drift.append((float(total_liquid_inventory(f, fill, flags, 1.0)) - inv0) / inv0)
        if step in wanted:
            depth_x = _depth_profile_x(fill, nz)
            profiles[f"{wanted[step]:.3f}"] = {
                "t_star": wanted[step],
                "step": step,
                "depth_x": depth_x.tolist(),
                "depth_xz": _depth_column_map(fill).tolist(),
            }
        if step == n_steps:
            snapshots["depth_xz"] = _depth_column_map(fill).tolist()

    x = np.arange(nx) + 0.5
    return {
        "grid": {"nz": nz, "ny": ny, "nx": nx},
        "h0": config.h0,
        "column_width": column_width,
        "c0": config.c0,
        "g": config.g,
        "tau": config.tau_effective,
        "nu": config.nu,
        "reynolds": config.reynolds,
        "span_free_slip": config.span_free_slip,
        "rho_gas": config.rho_gas,
        "n_steps": n_steps,
        "t_star_end": n_steps * t_star_scale,
        "x": x.tolist(),
        "mass_drift_series": drift,
        "mass_drift_max_abs": float(np.max(np.abs(drift))),
        "inventory_drift_series": inv_drift,
        "inventory_drift_max_abs": float(np.max(np.abs(inv_drift))),
        "profiles": profiles,
        "snapshots": snapshots,
        "nan_step": nan_step,
        "interface_cells": int((flags == INTERFACE).sum()),
        "gas_cells": int((flags == GAS).sum()),
    }


def _depth_profile_x(fill: torch.Tensor, nz: int) -> np.ndarray:
    """Streamwise depth: vertical fill sum, averaged over the span."""
    return (fill.clamp(0.0, 1.0).sum(dim=(0, 1)) / nz).cpu().numpy()


def _depth_column_map(fill: torch.Tensor) -> np.ndarray:
    """Span-averaged water column height on the ``(z, x)`` plane."""
    return (fill.clamp(0.0, 1.0).sum(dim=1)).cpu().numpy()
