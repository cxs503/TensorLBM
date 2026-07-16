"""Minimal R1 software runtime from a static SUBOFF wall-link asset to ``Ct``.

This deliberately is not a CFD solver: it creates deterministic synthetic
D3Q19 populations, then measures stationary bounce-back momentum exchange on
every compiled solid-to-fluid wall link.  It does not invoke collision,
boundary, obstacle, or full-wet solver paths.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Any

import torch

from .d3q19 import C, OPPOSITE, equilibrium3d
from .force_observation import ForceObservation
from .marine_geometry import GeometryAsset, compile_d3q19_wall_links
from .marine_resistance_contract import build_resistance_force_contract
from .suboff_case_definition import SuboffCaseDefinition


@dataclass(frozen=True, slots=True)
class SuboffFullWetRuntimeConfig:
    """Explicit synthetic-population and coefficient normalisation inputs."""

    samples: int = 1
    density_lattice: float = 1.0
    velocity: tuple[float, float, float] = (0.03, 0.0, 0.0)
    rho: float = 1000.0
    U: float = 1.0
    reference_area: float = 1.0
    length: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples <= 0:
            raise ValueError("samples must be a positive integer")
        for name in ("density_lattice", "rho", "U", "reference_area", "length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if not isinstance(self.velocity, tuple) or len(self.velocity) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value))
            for value in self.velocity
        ):
            raise ValueError("velocity must be a finite (x, y, z) tuple")
        if not any(self.velocity):
            raise ValueError("velocity must be non-zero for a force candidate")


def _hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _observation_dict(observation: ForceObservation) -> dict[str, Any]:
    return {
        "method": observation.method,
        "lattice_id": observation.lattice_id,
        "sample_phase": observation.sample_phase,
        "force_on": observation.force_on,
        "origin": observation.origin,
        "status": observation.status,
        "force": observation.force,
        "link_ownership": observation.link_ownership,
    }


def run_suboff_full_wet_runtime(
    asset: GeometryAsset,
    *,
    case: SuboffCaseDefinition | None = None,
    config: SuboffFullWetRuntimeConfig | None = None,
) -> dict[str, Any]:
    """Measure all compiled links and classify the result as a software candidate.

    Each link uses the incident population at its explicitly compiled fluid
    neighbour.  For stationary bounce-back the reflected population is equal
    to that incident population, so the body force is ``-2 f_incident c_q``.
    This is evaluated link-by-link and only summed afterwards.
    """
    if not isinstance(asset, GeometryAsset):
        raise TypeError("asset must be a GeometryAsset")
    resolved_case = case or SuboffCaseDefinition()
    if not isinstance(resolved_case, SuboffCaseDefinition):
        raise TypeError("case must be a SuboffCaseDefinition")
    resolved_config = config or SuboffFullWetRuntimeConfig()
    if not isinstance(resolved_config, SuboffFullWetRuntimeConfig):
        raise TypeError("config must be a SuboffFullWetRuntimeConfig")

    links = compile_d3q19_wall_links(asset)
    if links.count <= 0:
        raise ValueError("runtime requires at least one compiled wall link")
    shape = asset.solid_mask.shape
    rho_field = torch.full(shape, float(resolved_config.density_lattice), dtype=torch.float64)
    velocities = [torch.full(shape, float(component), dtype=torch.float64) for component in resolved_config.velocity]
    populations = equilibrium3d(rho_field, velocities[0], velocities[1], velocities[2])
    c = C.to(dtype=populations.dtype)
    opposite = OPPOSITE.to(dtype=torch.int64)

    series: list[dict[str, Any]] = []
    total_force = torch.zeros(3, dtype=torch.float64)
    for sample_index in range(resolved_config.samples):
        exchanges: list[dict[str, Any]] = []
        sample_force = torch.zeros(3, dtype=torch.float64)
        for index in range(links.count):
            q = int(links.direction[index].item())
            neighbor = links.neighbor_zyx[index]
            z, y, x = (int(value.item()) for value in neighbor)
            incident_q = int(opposite[q].item())
            incident_population = float(populations[incident_q, z, y, x].item())
            force = -2.0 * incident_population * c[q]
            sample_force += force
            exchanges.append({
                "link_index": index,
                "owner_zyx": tuple(int(value.item()) for value in links.owner_zyx[index]),
                "neighbor_zyx": (z, y, x),
                "outward_direction_q": q,
                "incident_direction_q": incident_q,
                "incident_population": incident_population,
                "reflected_population": incident_population,
                "force_on_body": tuple(float(value.item()) for value in force),
            })
        total_force += sample_force
        series.append({
            "sample": sample_index,
            "sample_phase": "synthetic_population_post_stream_pre_bounce_back",
            "link_count": links.count,
            "per_link_momentum_exchange": exchanges,
            "force_on_body": tuple(float(value.item()) for value in sample_force),
        })
    mean_force = (
        float(total_force[0].item()) / resolved_config.samples,
        float(total_force[1].item()) / resolved_config.samples,
        float(total_force[2].item()) / resolved_config.samples,
    )
    method = "d3q19_linkwise_momentum_exchange"
    sample_phase = "synthetic_population_post_stream_pre_bounce_back"
    observation = ForceObservation(
        method=method, lattice_id="D3Q19", sample_phase=sample_phase,
        force_on="body", origin=asset.origin, status="measured", force=mean_force,
        link_ownership=links.has_link_ownership,
    )
    ownership = {"status": "complete", "owner": asset.body_id, "owned_links": links.count,
                 "geometry_source_hash": asset.source_hash}
    contract = build_resistance_force_contract(
        reference_area=resolved_config.reference_area, length=resolved_config.length,
        rho=resolved_config.rho, U=resolved_config.U, direction=resolved_config.velocity,
        method=method, sample_phase=sample_phase, link_ownership=ownership, force=mean_force,
    )
    contract_dict = asdict(contract)
    case_record = {
        "application": resolved_case.application, "configuration": resolved_case.configuration,
        "medium": resolved_case.medium, "schema": resolved_case.schema,
        "reference_sha256": resolved_case.reference.sha256,
    }
    config_record = asdict(resolved_config)
    return {
        "artifact_kind": "software_runtime", "schema": "suboff-full-wet-runtime-r1",
        "case": case_record, "case_hash": _hash(case_record),
        "geometry": {"body_id": asset.body_id, "source_hash": asset.source_hash, "shape_zyx": tuple(shape)},
        "runtime_config_hash": _hash(config_record), "method": method, "sample_phase": sample_phase,
        "links": {"count": links.count, "ownership": ownership}, "force_series": series,
        "force": mean_force, "Ct": contract.Ct, "status": contract.status,
        "force_observation": _observation_dict(observation), "contract": contract_dict,
        "reference": {"source_status": resolved_case.reference.source_status, "physical": False},
        "physical_validation": False,
        "prohibitions": ["no_collision", "no_boundary_update", "no_population_reset"],
    }


__all__ = ["SuboffFullWetRuntimeConfig", "run_suboff_full_wet_runtime"]
