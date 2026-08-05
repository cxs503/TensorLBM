"""Cold cell-level audit of free-surface conversion representation changes.

This module only reads already captured pre/post conversion evidence.  It does
not rebuild populations, adjust mass/fill, execute a solver step, or assert a
physical closure.  Missing evidence is explicitly withheld rather than turned
into a synthetic zero observation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE = "DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE"
WITHHELD_MISSING_CONVERSION_EVIDENCE = "WITHHELD_MISSING_CONVERSION_EVIDENCE"


@dataclass(frozen=True)
class ConversionDensityCell:
    """One immutable, actual-state conversion representation observation."""

    cell: tuple[int, int, int]
    classification: str
    flag_before: int
    flag_after: int
    fill_rho_before: float
    fill_rho_after: float
    independent_mass_before: float
    independent_mass_after: float
    population_density_before: float
    population_density_after: float
    production_inventory_before: float
    production_inventory_after: float
    production_inventory_delta: float
    representation_switch_delta: float
    population_nominal_density_gap: float | None


@dataclass(frozen=True)
class ConversionDensityAudit:
    """Immutable, observer-only result; never a physical closure verdict."""

    status: str
    withheld_reason: str | None
    cells: tuple[ConversionDensityCell, ...]
    conversion_inventory_delta: float | None
    sum_cell_production_inventory_delta: float | None
    observed_conversion_inventory_delta: float | None
    cell_sum_inventory_residual: float | None
    representation_switch_delta: float | None
    cell_sum_matches_conversion_inventory_delta: bool | None
    i_to_l_count: int
    i_to_g_count: int
    l_to_g_count: int
    g_to_i_count: int
    other_count: int
    i_to_l_population_nominal_density_gap: float | None
    i_to_l_population_nominal_density_gap_cells: int


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"conversion evidence {field} must be numeric")
    return float(value)


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"conversion evidence {field} must be an integer")
    return int(value)


def _cell(value: object) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("conversion evidence cell must be a three-index sequence")
    return tuple(_integer(item, "cell index") for item in value)  # type: ignore[return-value]


def _classification(flag_before: int, flag_after: int) -> str:
    names = {0: "G", 1: "L", 2: "I"}
    return f"{names.get(flag_before, 'OTHER')}_TO_{names.get(flag_after, 'OTHER')}"


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    """Accept a frozen tuple-of-pairs report as well as live mapping evidence."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pairs = tuple(value)
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in pairs
        ):
            return {key: item for key, item in pairs}  # type: ignore[misc]
    raise ValueError(f"{field} must be a mapping")


def _production_inventory(flag: int, fill_rho: float, population: float) -> float:
    # This is exactly the existing production convention: population density
    # for LIQUID; fill*rho for INTERFACE; zero for GAS/other phases.
    if flag == 1:
        return population
    if flag == 2:
        return fill_rho
    return 0.0


def build_conversion_density_audit(
    conversion_evidence: Mapping[str, object] | None, *, rho_liquid: float,
    observed_conversion_inventory_delta: float | None = None,
) -> ConversionDensityAudit:
    """Read actual sparse conversion evidence and report representation deltas.

    ``conversion_evidence`` must be the plan's pre/post conversion snapshot.
    The function is deliberately pure and returns a detached immutable report.
    """
    if conversion_evidence is None:
        return ConversionDensityAudit(
            status=DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE,
            withheld_reason=WITHHELD_MISSING_CONVERSION_EVIDENCE,
            cells=(), conversion_inventory_delta=None,
            sum_cell_production_inventory_delta=None, observed_conversion_inventory_delta=None,
            cell_sum_inventory_residual=None, representation_switch_delta=None,
            cell_sum_matches_conversion_inventory_delta=None,
            i_to_l_count=0, i_to_g_count=0, l_to_g_count=0, g_to_i_count=0, other_count=0,
            i_to_l_population_nominal_density_gap=None,
            i_to_l_population_nominal_density_gap_cells=0,
        )
    if isinstance(rho_liquid, bool) or not isinstance(rho_liquid, (int, float)):
        raise ValueError("rho_liquid must be numeric")
    if observed_conversion_inventory_delta is not None:
        observed_conversion_inventory_delta = _number(
            observed_conversion_inventory_delta, "observed_conversion_inventory_delta",
        )
    raw_cells = conversion_evidence.get("conversion_cells")
    if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes)):
        raise ValueError("conversion evidence must contain conversion_cells")

    cells: list[ConversionDensityCell] = []
    for raw in raw_cells:
        raw = _as_mapping(raw, "each conversion evidence cell")
        flag_before = _integer(raw.get("flag_before"), "flag_before")
        flag_after = _integer(raw.get("flag_after"), "flag_after")
        fill_before = _number(raw.get("fill_before"), "fill_before")
        fill_after = _number(raw.get("fill_after"), "fill_after")
        mass_before = _number(raw.get("mass_before"), "mass_before")
        mass_after = _number(raw.get("mass_after"), "mass_after")
        population_before = _number(raw.get("population_before"), "population_before")
        population_after = _number(raw.get("population_after"), "population_after")
        fill_rho_before, fill_rho_after = fill_before * float(rho_liquid), fill_after * float(rho_liquid)
        production_before = _production_inventory(flag_before, fill_rho_before, population_before)
        production_after = _production_inventory(flag_after, fill_rho_after, population_after)
        classification = _classification(flag_before, flag_after)
        population_gap = population_after - float(rho_liquid) if classification == "I_TO_L" else None
        cells.append(ConversionDensityCell(
            cell=_cell(raw.get("cell")), classification=classification,
            flag_before=flag_before, flag_after=flag_after,
            fill_rho_before=fill_rho_before, fill_rho_after=fill_rho_after,
            independent_mass_before=mass_before, independent_mass_after=mass_after,
            population_density_before=population_before, population_density_after=population_after,
            production_inventory_before=production_before, production_inventory_after=production_after,
            production_inventory_delta=production_after - production_before,
            representation_switch_delta=production_after - fill_rho_before,
            population_nominal_density_gap=population_gap,
        ))

    frozen_cells = tuple(cells)
    conversion_delta = sum(cell.production_inventory_delta for cell in frozen_cells)
    switch_delta = sum(cell.representation_switch_delta for cell in frozen_cells)
    cell_sum_residual = (
        None if observed_conversion_inventory_delta is None
        else conversion_delta - observed_conversion_inventory_delta
    )
    groups = {name: tuple(cell for cell in frozen_cells if cell.classification == name)
              for name in ("I_TO_L", "I_TO_G", "L_TO_G", "G_TO_I")}
    i_to_l_gaps = tuple(
        cell.population_nominal_density_gap for cell in groups["I_TO_L"]
        if cell.population_nominal_density_gap is not None
    )
    return ConversionDensityAudit(
        status=DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE, withheld_reason=None,
        cells=frozen_cells, conversion_inventory_delta=conversion_delta,
        sum_cell_production_inventory_delta=conversion_delta,
        observed_conversion_inventory_delta=observed_conversion_inventory_delta,
        cell_sum_inventory_residual=cell_sum_residual,
        representation_switch_delta=switch_delta,
        cell_sum_matches_conversion_inventory_delta=(
            True if cell_sum_residual is None else cell_sum_residual == 0.0
        ),
        i_to_l_count=len(groups["I_TO_L"]), i_to_g_count=len(groups["I_TO_G"]),
        l_to_g_count=len(groups["L_TO_G"]), g_to_i_count=len(groups["G_TO_I"]),
        other_count=len(frozen_cells) - sum(len(group) for group in groups.values()),
        i_to_l_population_nominal_density_gap=(sum(i_to_l_gaps) if i_to_l_gaps else 0.0),
        i_to_l_population_nominal_density_gap_cells=sum(gap != 0.0 for gap in i_to_l_gaps),
    )
