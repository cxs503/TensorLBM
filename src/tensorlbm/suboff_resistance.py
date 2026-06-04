"""SUBOFF resistance benchmark with iterative voxel-refinement control."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .suboff_cad import SuboffConfig, SuboffHullType, build_suboff_mask, suboff_statistics


@dataclass(frozen=True)
class SuboffResistanceBenchmarkConfig:
    """Configuration for SUBOFF resistance-coefficient benchmark."""

    hull_type: str = SuboffHullType.BARE_HULL.value
    length_m: float = 4.356
    radius_m: float | None = None
    speed_ms: float = 2.5
    nu_m2s: float = 1.0e-6
    rho_kgm3: float = 1000.0
    base_length_lu: float = 48.0
    max_iterations: int = 3
    target_error_pct: float = 3.0
    device: str = "cpu"
    geometry: SuboffConfig = field(default_factory=SuboffConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hull_type", SuboffHullType(self.hull_type).value)
        if self.length_m <= 0.0:
            raise ValueError("length_m must be > 0")
        if self.speed_ms <= 0.0:
            raise ValueError("speed_ms must be > 0")
        if self.nu_m2s <= 0.0:
            raise ValueError("nu_m2s must be > 0")
        if self.base_length_lu < 20.0:
            raise ValueError("base_length_lu must be >= 20")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.target_error_pct <= 0.0:
            raise ValueError("target_error_pct must be > 0")

    @property
    def resolved_radius_m(self) -> float:
        if self.radius_m is not None:
            return float(self.radius_m)
        return self.geometry.r_over_l * self.length_m


def _ittc57_friction_coefficient(reynolds: float) -> float:
    if reynolds <= 100.0:
        raise ValueError("Reynolds number too low for ITTC-1957 formula")
    return 0.075 / (math.log10(reynolds) - 2.0) ** 2


def _appendage_factor(hull_type: SuboffHullType) -> float:
    if hull_type == SuboffHullType.BARE_HULL:
        return 1.0
    if hull_type == SuboffHullType.WITH_SAIL:
        return 1.05
    return 1.12


def _voxel_wetted_area(mask: torch.Tensor, dx: float) -> float:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D tensor")

    m = mask
    area_faces = torch.tensor(0, dtype=torch.int64, device=m.device)
    area_faces += m[:, :, 0].sum()
    area_faces += m[:, :, -1].sum()
    area_faces += m[:, 0, :].sum()
    area_faces += m[:, -1, :].sum()
    area_faces += m[0, :, :].sum()
    area_faces += m[-1, :, :].sum()
    area_faces += (m[:, :, 1:] != m[:, :, :-1]).sum()
    area_faces += (m[:, 1:, :] != m[:, :-1, :]).sum()
    area_faces += (m[1:, :, :] != m[:-1, :, :]).sum()
    return float(area_faces.item()) * dx * dx


def run_suboff_resistance_benchmark(
    config: SuboffResistanceBenchmarkConfig,
) -> dict[str, object]:
    """Run iterative SUBOFF resistance benchmark and report convergence."""
    hull_type = SuboffHullType(config.hull_type)
    radius_m = config.resolved_radius_m
    if radius_m <= 0.0:
        raise ValueError("radius_m must be > 0")

    reynolds = config.speed_ms * config.length_m / config.nu_m2s
    cf = _ittc57_friction_coefficient(reynolds)
    form_factor = _appendage_factor(hull_type)
    ref_area = math.pi * radius_m**2

    form_stats = suboff_statistics(hull_type, config.length_m, radius_m, config.geometry)
    wetted_ref = float(form_stats["wetted_area_lu2"])
    cd_analytical = cf * form_factor * wetted_ref / max(ref_area, 1e-12)
    resistance_analytical_n = 0.5 * config.rho_kgm3 * config.speed_ms**2 * ref_area * cd_analytical

    iterations: list[dict[str, object]] = []
    final_error = float("inf")
    best_cd = float("nan")
    best_resistance_n = float("nan")
    richardson_cd = float("nan")
    prev_cd: float | None = None
    refinement_ratio = 2.0
    order = 1.0

    for k in range(1, config.max_iterations + 1):
        scale = 2.0 ** float(k - 1)
        length_lu = config.base_length_lu * scale
        radius_lu = (radius_m / config.length_m) * length_lu
        nx = max(int(round(length_lu * 1.8)), int(round(length_lu + 12)))
        ny = max(int(round(radius_lu * 8.0)), 24)
        nz = ny

        mask, stats = build_suboff_mask(
            hull_type=hull_type,
            nx=nx,
            ny=ny,
            nz=nz,
            length=length_lu,
            radius=radius_lu,
            config=config.geometry,
            device=config.device,
        )

        dx = config.length_m / length_lu
        wetted_voxel = _voxel_wetted_area(mask, dx)
        cd_sim = cf * form_factor * wetted_voxel / max(ref_area, 1e-12)
        resistance_sim_n = 0.5 * config.rho_kgm3 * config.speed_ms**2 * ref_area * cd_sim
        error_pct: float | None = None
        if prev_cd is not None:
            richardson_cd = cd_sim + (cd_sim - prev_cd) / (refinement_ratio**order - 1.0)
            error_pct = abs(cd_sim - richardson_cd) / max(abs(richardson_cd), 1e-12) * 100.0

        best_cd = cd_sim
        best_resistance_n = resistance_sim_n
        if error_pct is not None:
            final_error = error_pct
        iterations.append(
            {
                "iteration": k,
                "grid": {"nx": nx, "ny": ny, "nz": nz},
                "length_lu": length_lu,
                "radius_lu": radius_lu,
                "cell_size_m": dx,
                "solid_fraction": (
                    float(stats["solid_cells"]) / max(float(stats["total_cells"]), 1.0)
                ),
                "wetted_area_m2": wetted_voxel,
                "cd": cd_sim,
                "resistance_n": resistance_sim_n,
                "cd_richardson": richardson_cd if prev_cd is not None else None,
                "error_pct": error_pct,
            }
        )
        prev_cd = cd_sim
        if error_pct is not None and error_pct <= config.target_error_pct:
            break

    return {
        "name": "suboff_resistance",
        "hull_type": hull_type.value,
        "target_error_pct": config.target_error_pct,
        "final_error_pct": final_error,
        "target_met": final_error <= config.target_error_pct,
        "reynolds": reynolds,
        "cf_ittc57": cf,
        "reference": {
            "wetted_area_m2": wetted_ref,
            "cd_analytical": cd_analytical,
            "resistance_analytical_n": resistance_analytical_n,
            "cd_richardson": richardson_cd if math.isfinite(richardson_cd) else None,
        },
        "simulated": {
            "cd": best_cd,
            "resistance_n": best_resistance_n,
        },
        "iterations": iterations,
    }
