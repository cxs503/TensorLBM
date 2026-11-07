"""Wall-treatment mesh budget: the negative control of the validation ladder.

The claim that wall functions are *necessary* at high Reynolds number is a
quantitative statement about cell counts, not an opinion.  Given a Reynolds
number, a reference length and a target first-cell ``y+``, the wall-normal cell
count follows directly from the friction velocity:

    ``δ+ = δ u_tau / ν``,   ``u_tau = U sqrt(C_f/2)``,
    ``N_y = δ+ / dy+``      (uniform, wall-resolved)
    ``N_y = ln(1 + δ+ (q-1)/dy+_1) / ln(q)``  (geometric stretching, q = growth)

Combined with ``Δx+``/``Δz+`` targets this gives the total cell count for a
wall-resolved mesh (``dy+ ≈ 1``) and for a wall-modelled mesh
(``dy+_1 ∈ [30, 300]``).  Their ratio is the negative-control metric of the
ladder: it quantifies how many orders of magnitude of resolution the wall
function buys, and therefore which of the two treatments is feasible.

All inputs are physical; there is no tuning constant except the standard
turbulent boundary-layer growth ``δ/L = 0.37 Re_L^{-1/5}`` and the geometric
stretching ratio.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from .wall_law import ittc57_cf

__all__ = [
    "MeshBudget",
    "comparison_report",
    "plate_mesh_budget",
    "total_cells",
]

#: Streamwise / spanwise resolutions used by the two treatments.
RESOLVED_RESOLUTION = {"dx_plus": 15.0, "dz_plus": 8.0, "dy_plus": 1.0}
WALL_MODELED_RESOLUTION = {"dx_plus": 100.0, "dz_plus": 50.0, "dy_plus": 50.0}


@dataclass(frozen=True)
class MeshBudget:
    """Cell budget of a boundary-layer block at a given wall resolution."""

    reynolds: float
    length: float
    cf: float
    u_tau_over_reference: float
    boundary_layer_thickness_over_length: float
    delta_plus: float
    dy_plus_first_cell: float
    dx_plus: float
    dz_plus: float
    cells_wall_normal: float
    cells_streamwise: float
    cells_spanwise: float
    growth_ratio: float | None
    treatment: str

    @property
    def total_cells(self) -> float:
        return self.cells_wall_normal * self.cells_streamwise * self.cells_spanwise

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_cells"] = self.total_cells
        return payload


def total_cells(budget: MeshBudget) -> float:
    """Total cell count of the boundary-layer block (convenience wrapper)."""
    return budget.total_cells


def _boundary_layer_growth(reynolds: float, growth: float, exponent: float) -> float:
    return growth * reynolds ** (-exponent)


def plate_mesh_budget(
    reynolds: float,
    length: float,
    *,
    treatment: str,
    cf_model: Callable[[float], float] = ittc57_cf,
    dy_plus_first_cell: float = 1.0,
    dx_plus: float = 15.0,
    dz_plus: float = 8.0,
    spanwise_extent_fraction: float = 0.5,
    boundary_layer_growth: float = 0.37,
    boundary_layer_exponent: float = 0.2,
    growth_ratio: float | None = None,
) -> MeshBudget:
    """Cell budget for a flat-plate-like boundary layer.

    ``treatment`` is free text (``"wall_resolved"`` / ``"wall_modeled"``) and is
    echoed in the report.  ``growth_ratio`` > 1 activates geometric stretching
    of the wall-normal spacing, which is what a wall-modelled mesh actually
    uses; with ``None`` the spacing is uniform.
    """
    if reynolds <= 0.0 or length <= 0.0:
        raise ValueError("reynolds and length must be positive")
    if dy_plus_first_cell <= 0.0 or dx_plus <= 0.0 or dz_plus <= 0.0:
        raise ValueError("wall resolutions must be positive")
    cf = cf_model(reynolds)
    u_tau_over_u = math.sqrt(cf / 2.0)
    delta_over_length = _boundary_layer_growth(
        reynolds, boundary_layer_growth, boundary_layer_exponent
    )
    delta_plus = delta_over_length * reynolds * u_tau_over_u
    cells_streamwise = reynolds * u_tau_over_u / dx_plus
    cells_spanwise = (
        spanwise_extent_fraction * delta_plus / dz_plus
    )
    if growth_ratio is not None:
        if growth_ratio <= 1.0:
            raise ValueError("growth_ratio must exceed 1 for a stretched mesh")
        cells_wall_normal = math.log(
            1.0 + delta_plus * (growth_ratio - 1.0) / dy_plus_first_cell
        ) / math.log(growth_ratio)
    else:
        cells_wall_normal = delta_plus / dy_plus_first_cell
    return MeshBudget(
        reynolds=reynolds,
        length=length,
        cf=cf,
        u_tau_over_reference=u_tau_over_u,
        boundary_layer_thickness_over_length=delta_over_length,
        delta_plus=delta_plus,
        dy_plus_first_cell=dy_plus_first_cell,
        dx_plus=dx_plus,
        dz_plus=dz_plus,
        cells_wall_normal=cells_wall_normal,
        cells_streamwise=cells_streamwise,
        cells_spanwise=cells_spanwise,
        growth_ratio=growth_ratio,
        treatment=treatment,
    )


def comparison_report(
    reynolds_values: Sequence[float],
    length: float,
    *,
    spanwise_extent_fraction: float = 0.5,
    cf_model: Callable[[float], float] = ittc57_cf,
    resolved_resolution: dict[str, float] | None = None,
    wall_modeled_resolution: dict[str, float] | None = None,
    wall_modeled_growth_ratio: float = 1.2,
) -> dict[str, Any]:
    """Negative control: wall-resolved vs wall-modelled cell budget.

    Returns per-Reynolds budgets plus the cell-count ratio, which is the number
    that quantifies *why* wall functions are mandatory at high Reynolds number.
    """
    resolved_resolution = dict(resolved_resolution or RESOLVED_RESOLUTION)
    wall_modeled_resolution = dict(wall_modeled_resolution or WALL_MODELED_RESOLUTION)
    rows = []
    for reynolds in reynolds_values:
        resolved = plate_mesh_budget(
            reynolds,
            length,
            treatment="wall_resolved",
            cf_model=cf_model,
            dy_plus_first_cell=resolved_resolution["dy_plus"],
            dx_plus=resolved_resolution["dx_plus"],
            dz_plus=resolved_resolution["dz_plus"],
            spanwise_extent_fraction=spanwise_extent_fraction,
        )
        modeled = plate_mesh_budget(
            reynolds,
            length,
            treatment="wall_modeled",
            cf_model=cf_model,
            dy_plus_first_cell=wall_modeled_resolution["dy_plus"],
            dx_plus=wall_modeled_resolution["dx_plus"],
            dz_plus=wall_modeled_resolution["dz_plus"],
            spanwise_extent_fraction=spanwise_extent_fraction,
            growth_ratio=wall_modeled_growth_ratio,
        )
        rows.append(
            {
                "reynolds": reynolds,
                "wall_resolved": resolved.to_dict(),
                "wall_modeled": modeled.to_dict(),
                "cell_ratio": resolved.total_cells / max(modeled.total_cells, 1.0),
                "wall_normal_ratio": resolved.cells_wall_normal
                / max(modeled.cells_wall_normal, 1.0),
            }
        )
    return {
        "length": length,
        "spanwise_extent_fraction": spanwise_extent_fraction,
        "resolved_resolution": resolved_resolution,
        "wall_modeled_resolution": wall_modeled_resolution,
        "wall_modeled_growth_ratio": wall_modeled_growth_ratio,
        "rows": rows,
        "maximum_cell_ratio": max(row["cell_ratio"] for row in rows),
        "maximum_total_cells_resolved": max(
            row["wall_resolved"]["total_cells"] for row in rows
        ),
        "maximum_total_cells_modeled": max(
            row["wall_modeled"]["total_cells"] for row in rows
        ),
    }