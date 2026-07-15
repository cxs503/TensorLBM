"""Propeller open-water benchmark using 3D LBM with moving-wall bounce-back.

Models a rotating propeller in uniform inflow to compute thrust and torque
coefficients (KT, KQ) as functions of advance ratio J.

Uses a fixed-RPM variable-inflow strategy: inflow velocity is varied while
RPM is held constant to maintain tip-speed stability (tip Ma < 0.005).

Stability: the Ladd (1994) moving-wall BC is stable for tip Ma < 0.004.
Default rpm=1e-5 with D=32 gives tip Ma=0.002, well within limits.

Reference data
--------------
- KP505 open-water: Fujisawa et al. (2000), SIMMAN 2008/2014.
- ITTC (2014) "Recommended Procedures: Open Water Test", 7.5-02-03-02.1.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .boundaries3d import (
    apply_zou_he_channel_boundaries_3d,
    bounce_back_cells_3d,
    make_channel_wall_mask_3d,
)
from .d3q19 import C, W, equilibrium3d
from .obstacles import compute_obstacle_forces_3d, compute_obstacle_moments_3d
from .propeller_cad import (
    KP505_PRESET,
    PropellerGeometryConfig,
    build_propeller_mask,
    propeller_statistics,
)
from .solver3d import stream3d
from .turbulence import collide_smagorinsky_mrt3d
from .utils import (
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
)


# The moving-wall implementation is qualified only below this value.  Equal
# Mach numbers above it are not low-Mach comparison evidence.
LOW_MACH_TIP_GATE = 0.004
_SENSITIVITY_MATCH_TOL = 1.0e-12


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropellerBenchmarkConfig:
    """Configuration for the propeller open-water benchmark.

    Uses a fixed-RPM variable-inflow strategy where the advance ratio J
    is varied by changing the inflow velocity at constant RPM.
    """

    geometry: PropellerGeometryConfig = field(default_factory=lambda: KP505_PRESET)
    inflow_velocities: tuple[float, ...] = (0.005, 0.010, 0.015)
    rpm: float = 0.000005
    nx: int = 200
    ny: int = 100
    nz: int = 100
    tau: float = 0.8
    smagorinsky_cs: float = 0.0
    n_revolutions: int = 3
    sampling_steps: int | None = None
    warmup_steps: int = 200
    sample_window_steps: int = 200
    window_convergence_rel_tol: float = 0.02
    device: str = "cpu"
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    overwrite: bool = False

    # Physical model-scale parameters (for scaled KT/KQ output)
    model_diameter_m: float = 0.25
    model_speed_ms: float = 2.5
    model_rho_kgm3: float = 1000.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        if self.nx < 40 or self.ny < 20 or self.nz < 20:
            raise ValueError("nx, ny, nz must be at least 40, 20, 20")
        if self.rpm <= 0:
            raise ValueError("rpm must be > 0")
        if self.tau <= 0.5:
            raise ValueError("tau must be > 0.5")
        if self.n_revolutions < 1:
            raise ValueError("n_revolutions must be >= 1")
        if self.sampling_steps is not None and self.sampling_steps < 1:
            raise ValueError("sampling_steps must be >= 1 when specified")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.sample_window_steps < 1:
            raise ValueError("sample_window_steps must be >= 1")
        if not math.isfinite(self.window_convergence_rel_tol) or self.window_convergence_rel_tol <= 0:
            raise ValueError("window_convergence_rel_tol must be finite and > 0")
        if not self.inflow_velocities:
            raise ValueError("inflow_velocities must not be empty")

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * self.rpm

    @property
    def steps_per_revolution(self) -> int:
        """Nearest integer update count; exactness is checked for sensitivity.

        The solver advances azimuth by ``2*pi*rpm`` per update. Consequently,
        this rounded diagnostic describes a full revolution only when ``rpm``
        is exactly its reciprocal; the sensitivity contract verifies that.
        """
        return max(1, round(1.0 / max(self.rpm, 1e-10)))

    @property
    def angular_increment_degrees(self) -> float:
        """Physical azimuth increment actually used by each solver update."""
        return 360.0 * self.rpm

    @property
    def tip_ma(self) -> float:
        return self.omega * self.geometry.radius / 0.577

    @property
    def re_d(self) -> float:
        return self.rpm * self.geometry.diameter**2 / self.nu

    @property
    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        D = int(self.geometry.diameter)
        tau_str = f"tau{self.tau:.3f}".replace(".", "p")
        rpm_str = f"rpm{self.rpm:.2g}".replace("+", "p").replace("-", "m")
        return f"propeller_n{self.geometry.n_blades}_D{D}_nx{self.nx}_{tau_str}_{rpm_str}"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        d = asdict(self)
        d["output_root"] = str(d["output_root"])
        d["geometry"] = asdict(self.geometry)
        path.write_text(f"{json.dumps(d, indent=2, sort_keys=True)}\n", encoding="utf-8")
        return path


# ============================================================================
# 3-D moving-wall bounce-back (Ladd 1994, extended to D3Q19)
# ============================================================================

@dataclass(frozen=True)
class MovingWallReaction3D:
    """Reaction recorded from the exact moving-wall update applied to ``f``.

    ``fluid_impulse`` is the D3Q19 distribution-momentum increment caused by
    the complete bounce-back *and* moving-wall correction.  The body reaction
    is its negative. This is distinct from legacy static obstacle ME.
    """

    fluid_impulse: torch.Tensor
    fluid_torque_impulse: torch.Tensor
    body_reaction: torch.Tensor
    body_reaction_torque: torch.Tensor
    action_reaction_signed_residual_norm: float
    action_reaction_absolute_residual_norm: float
    action_reaction_relative_residual: float
    torque_action_reaction_signed_residual_norm: float
    torque_action_reaction_absolute_residual_norm: float
    torque_action_reaction_relative_residual: float


def rotating_wall_velocity_3d(
    obstacle_mask: torch.Tensor,
    cx: float,
    cy: float,
    cz: float,
    omega: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rigid-body rotation velocity field about the x-axis.

    u_w = omega x r, where omega = (omega, 0, 0) and r = (x-cx, y-cy, z-cz).
    Returns (ux_w, uy_w, uz_w) each of shape (nz, ny, nx).
    """
    device = obstacle_mask.device
    nz, ny, nx = obstacle_mask.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ux_w = torch.zeros_like(xx)
    uy_w = -omega * (zz - cz)
    uz_w = omega * (yy - cy)
    return ux_w, uy_w, uz_w


def moving_wall_bounce_back_3d(
    f: torch.Tensor,
    mask: torch.Tensor,
    ux_w: torch.Tensor,
    uy_w: torch.Tensor,
    uz_w: torch.Tensor,
) -> torch.Tensor:
    """Ladd (1994) moving-wall bounce-back for D3Q19.

    f_i(x) = f_i(x) - 2*w_i*rho*(c_i.u_w)/cs^2

    where rho is the local density and u_w is the prescribed wall velocity.
    Stable for tip Ma < 0.004.
    """
    device = f.device
    c = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
         [0, 0, 1], [0, 0, -1],
         [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
         [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
         [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]],
        dtype=f.dtype, device=device,
    )
    w = W.to(device).to(f.dtype)
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)
    w_view = w.view(19, 1, 1, 1)
    rho = f.sum(dim=0)
    f_bb = bounce_back_cells_3d(f, mask)
    cu_w = cx * ux_w.unsqueeze(0) + cy * uy_w.unsqueeze(0) + cz * uz_w.unsqueeze(0)
    correction = 2.0 * w_view * rho.unsqueeze(0) * cu_w * 3.0  # 1/cs^2 = 3
    return torch.where(mask.unsqueeze(0), f_bb + correction, f_bb)


def moving_wall_bounce_back_3d_with_reaction(
    f: torch.Tensor, mask: torch.Tensor,
    ux_w: torch.Tensor, uy_w: torch.Tensor, uz_w: torch.Tensor,
    *, origin: tuple[float, float, float],
) -> tuple[torch.Tensor, MovingWallReaction3D]:
    """Apply the moving-wall operator and record its same-operator reaction.

    The fluid impulse is measured from the exact masked population delta, so
    it includes both reflection and the Ladd moving-wall correction.  Its
    negative is the body reaction comparable to the CV wall contribution.
    """
    after = moving_wall_bounce_back_3d(f, mask, ux_w, uy_w, uz_w)
    c = C.to(device=f.device, dtype=f.dtype)
    masked_delta = (after - f)[:, mask]
    fluid_impulse = c.T @ masked_delta.sum(dim=1)

    nz, ny, nx = mask.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=f.device, dtype=f.dtype),
        torch.arange(ny, device=f.device, dtype=f.dtype),
        torch.arange(nx, device=f.device, dtype=f.dtype), indexing="ij",
    )
    positions = torch.stack((xx[mask] - origin[0], yy[mask] - origin[1], zz[mask] - origin[2]), dim=1)
    cell_impulses = masked_delta.T @ c
    fluid_torque = torch.cross(positions, cell_impulses, dim=1).sum(dim=0)
    body_reaction = -fluid_impulse
    body_torque = -fluid_torque
    residual = fluid_impulse + body_reaction
    signed_norm = float(torch.linalg.vector_norm(residual).item())
    absolute_norm = float(torch.linalg.vector_norm(residual.abs()).item())
    relative = absolute_norm / max(float(torch.linalg.vector_norm(fluid_impulse).item()), 1e-30)
    torque_residual = fluid_torque + body_torque
    torque_signed_norm = float(torch.linalg.vector_norm(torque_residual).item())
    torque_absolute_norm = float(torch.linalg.vector_norm(torque_residual.abs()).item())
    torque_relative = torque_absolute_norm / max(float(torch.linalg.vector_norm(fluid_torque).item()), 1e-30)
    return after, MovingWallReaction3D(
        fluid_impulse=fluid_impulse,
        fluid_torque_impulse=fluid_torque,
        body_reaction=body_reaction,
        body_reaction_torque=body_torque,
        action_reaction_signed_residual_norm=signed_norm,
        action_reaction_absolute_residual_norm=absolute_norm,
        action_reaction_relative_residual=relative,
        torque_action_reaction_signed_residual_norm=torque_signed_norm,
        torque_action_reaction_absolute_residual_norm=torque_absolute_norm,
        torque_action_reaction_relative_residual=torque_relative,
    )


# ============================================================================
# KT/KQ computation with physical-scale conversion
# ============================================================================

def _compute_kt_kq(
    fx: float, mx: float, u_in: float, rpm: float, diameter: float,
    rho_ref: float = 1.0,
) -> tuple[float, float, float, float]:
    """Compute lattice-scaled thrust and torque coefficients."""
    n = rpm
    n2_d4 = n * n * (diameter**4)
    n2_d5 = n * n * (diameter**5)
    kt = fx / (rho_ref * n2_d4) if n2_d4 != 0 else float("nan")
    kq = mx / (rho_ref * n2_d5) if n2_d5 != 0 else float("nan")
    j_val = u_in / (n * diameter) if (n * diameter) != 0 else float("nan")
    eta = (j_val / (2.0 * math.pi)) * (kt / kq) if (kq != 0 and not math.isnan(kq)) else 0.0
    return kt, kq, j_val, eta


def _convert_to_physical_kt_kq(
    kt_lu: float, kq_lu: float, j_val: float,
    rpm_lu: float, d_lu: float, u_lu: float,
    d_phys: float, u_phys: float, rho_phys: float,
) -> tuple[float, float, float]:
    """Convert lattice-scaled KT/KQ to physical-scaled values.

    Uses velocity-based scaling: n_phys/n_lu = (d_lu/d_phys)*(u_phys/u_lu).
    Then KT_phys = KT_lu * (n_lu/n_phys)^2.
    """
    n_scale = (d_lu / d_phys) * (u_phys / u_lu)
    n_phys_rps = rpm_lu * n_scale
    inv_n2 = 1.0 / max(n_scale**2, 1e-20)
    return kt_lu * inv_n2, kq_lu * inv_n2, n_phys_rps


def _relative_change(previous: float, current: float) -> float:
    """Return a bounded denominator relative change for window diagnostics."""
    return abs(current - previous) / max(abs(current), 1e-30)


def _summarize_windows(
    samples: list[dict[str, float | int]], *, window_steps: int,
    transient_discard_steps: int, convergence_rel_tol: float,
) -> dict[str, object]:
    """Summarize complete post-discard windows against a strict KT/KQ criterion."""
    retained = [sample for sample in samples if int(sample["step"]) > transient_discard_steps]
    complete_count = len(retained) // window_steps
    windows: list[dict[str, float | int]] = []
    for index in range(complete_count):
        chunk = retained[index * window_steps:(index + 1) * window_steps]
        windows.append({
            "index": index,
            "step_start": int(chunk[0]["step"]),
            "step_end": int(chunk[-1]["step"]),
            "n_samples": len(chunk),
            "j_mean": sum(float(s["j"]) for s in chunk) / len(chunk),
            "kt_mean": sum(float(s["kt"]) for s in chunk) / len(chunk),
            "kq_mean": sum(float(s["kq"]) for s in chunk) / len(chunk),
        })
    if len(windows) < 2:
        convergence: dict[str, object] = {
            "available": False,
            "window_converged": False,
            "reason": "fewer_than_two_complete_windows",
            "kt_rel_tol": convergence_rel_tol,
            "kq_rel_tol": convergence_rel_tol,
        }
    else:
        previous, current = windows[-2], windows[-1]
        kt_delta = _relative_change(float(previous["kt_mean"]), float(current["kt_mean"]))
        kq_delta = _relative_change(float(previous["kq_mean"]), float(current["kq_mean"]))
        convergence = {
            "available": True,
            "window_converged": kt_delta < convergence_rel_tol and kq_delta < convergence_rel_tol,
            "kt_last_window_rel_change": kt_delta,
            "kq_last_window_rel_change": kq_delta,
            "kt_rel_tol": convergence_rel_tol,
            "kq_rel_tol": convergence_rel_tol,
        }
    return {
        "window_steps": window_steps,
        "transient_discard_steps": transient_discard_steps,
        "discarded_transient_samples": len(samples) - len(retained),
        "dropped_incomplete_tail_samples": len(retained) - complete_count * window_steps,
        "windows": windows,
        "convergence": convergence,
    }


def _is_exact_integer(value: float) -> bool:
    return math.isfinite(value) and abs(value - round(value)) <= _SENSITIVITY_MATCH_TOL


def _spatial_level_contract(config: PropellerBenchmarkConfig) -> dict[str, object]:
    """Describe one actual voxel level without calling it convergence evidence."""
    config.validate()
    steps = config.steps_per_revolution
    if not math.isclose(config.rpm * steps, 1.0, rel_tol=0.0, abs_tol=_SENSITIVITY_MATCH_TOL):
        raise ValueError(
            "actual angular increment "
            f"{config.angular_increment_degrees:.17g} deg/update disagrees with claimed "
            f"{360.0 / steps:.17g} deg/update"
        )
    if config.sampling_steps is not None:
        raise ValueError("spatial refinement requires sampling_steps=None")
    sampled = config.n_revolutions * steps
    for label, rotations in (("warmup_steps", config.warmup_steps * config.rpm), ("sample_window_steps", config.sample_window_steps * config.rpm), ("total sampling steps", sampled * config.rpm)):
        if not _is_exact_integer(rotations):
            raise ValueError(f"{label} must contain an exact whole number of rotations")
    if config.tip_ma >= LOW_MACH_TIP_GATE:
        raise ValueError("low-Mach gate")
    d_lu = config.geometry.diameter
    extents = [config.nx / d_lu, config.ny / d_lu, config.nz / d_lu]
    return {
        "diameter_lu": d_lu, "cell_size_m": config.model_diameter_m / d_lu,
        "domain_cells": [config.nx, config.ny, config.nz], "domain_per_diameter": extents,
        "domain_physical_m": [extent * config.model_diameter_m for extent in extents],
        "steps_per_revolution": steps, "angular_increment_degrees": config.angular_increment_degrees,
        "warmup_steps": config.warmup_steps, "sample_window_steps": config.sample_window_steps,
        "sampled_steps": sampled, "complete_rotations": sampled * config.rpm,
        "complete_windows": sampled // config.sample_window_steps, "tip_ma": config.tip_ma,
        "re_d": config.re_d, "advance_ratios": [u / (config.rpm * d_lu) for u in config.inflow_velocities],
    }


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=_SENSITIVITY_MATCH_TOL)


def _persist_spatial_evidence(config: PropellerBenchmarkConfig, evidence: dict[str, object]) -> None:
    path = config.output_root / "propeller_owt" / "resolution_sensitivity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence["artifact"] = str(path)
    path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def run_propeller_resolution_sensitivity(configs: tuple[PropellerBenchmarkConfig, ...], *, level_names: tuple[str, ...] | None = None) -> dict[str, object]:
    """Assess true voxel refinement and fail closed before incompatible runs.

    Direct voxelisation means increasing ``diameter_lu`` is spatial refinement
    only with a fixed physical propeller and tank.  This moving-boundary solver
    also needs identical update/angle histories; if low-Mach/Re/J scaling
    conflicts with those histories, no coefficient is computed or reported.
    """
    if len(configs) < 3:
        for config in configs:
            _spatial_level_contract(config)
        raise ValueError("at least 3 valid spatial resolution levels are required")
    names = level_names or tuple(f"level_{i}" for i in range(len(configs)))
    if len(names) != len(configs) or len(set(names)) != len(names):
        raise ValueError("level_names must be unique and match configs")
    contracts = [_spatial_level_contract(c) for c in configs]
    diameters = [float(c["diameter_lu"]) for c in contracts]
    if any(b <= a for a, b in zip(diameters, diameters[1:])):
        raise ValueError("spatial resolution must strictly refine: diameter_lu must increase")
    reference, ref_contract = configs[0], contracts[0]
    geometry_fields = ("n_blades", "hub_diameter_ratio", "hub_length_ratio", "pitch_ratio_07", "blade_area_ratio", "skew_deg", "rake_ratio", "max_thickness_ratio")
    geometry_matched = all(getattr(c.geometry, field) == getattr(reference.geometry, field) for c in configs[1:] for field in geometry_fields) and all(_same(c.model_diameter_m, reference.model_diameter_m) for c in configs[1:])
    domain_matched = all(
        all(_same(float(value), float(ref_value)) for value, ref_value in zip(
            list(contract["domain_per_diameter"]), list(ref_contract["domain_per_diameter"]),
        ))
        for contract in contracts[1:]
    )
    j_matched = all(list(contract["advance_ratios"]) == list(ref_contract["advance_ratios"]) for contract in contracts[1:])
    re_matched = all(_same(float(contract["re_d"]), float(ref_contract["re_d"])) for contract in contracts[1:])
    mach_matched = all(_same(float(contract["tip_ma"]), float(ref_contract["tip_ma"])) for contract in contracts[1:])
    temporal_matched = all(contract["steps_per_revolution"] == ref_contract["steps_per_revolution"] and _same(float(contract["angular_increment_degrees"]), float(ref_contract["angular_increment_degrees"])) and contract["warmup_steps"] == ref_contract["warmup_steps"] and contract["sample_window_steps"] == ref_contract["sample_window_steps"] and contract["sampled_steps"] == ref_contract["sampled_steps"] for contract in contracts[1:])
    comparable = geometry_matched and domain_matched and j_matched and re_matched and mach_matched and temporal_matched
    levels = [{"name": name, **contract, "campaign_status": "not_run", "same_operator_force_torque_check": {"status": "withheld", "reason": "campaign_not_executed"}} for name, contract in zip(names, contracts)]
    evidence: dict[str, object] = {
        "name": "propeller_open_water_spatial_refinement", "status": "withheld",
        "reason": "eligible_for_execution" if comparable else "incomparable_voxel_refinement_contract",
        "comparison_basis": {"kind": "same_physical_geometry_true_spatial_refinement", "geometry_matched": geometry_matched, "physical_domain_matched": domain_matched, "advance_ratios_matched": j_matched, "re_d_matched": re_matched, "low_mach_matched": mach_matched, "exact_rotation_time_sampling_matched": temporal_matched, "levels": contracts},
        "levels": levels,
        "metric_convergence": {"status": "withheld", "reason": "incomparable_or_unexecuted", "kt_kq_eta_differences": None},
    }
    _persist_spatial_evidence(configs[0], evidence)
    return evidence


def _d3q19_momentum_x(
    distributions: torch.Tensor, region: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return CV x momentum as ``Σ_i C[i, 0] f_i`` over all 19 directions."""
    if distributions.shape[0] != 19:
        raise ValueError("D3Q19 distributions must have 19 population directions")
    cx = C[:, 0].to(device=distributions.device, dtype=distributions.dtype)
    density = (cx.view(19, 1, 1, 1) * distributions).sum(dim=0)
    return density.sum() if region is None else density[region].sum()


def _summarize_control_volume_cross_check(samples: list[dict[str, float | int]]) -> dict[str, object]:
    """Summarize sampled discrete streamwise CV momentum terms.

    CV contributions use positive-x as fluid momentum gained. ``wall_reaction_x``
    is the body reaction emitted by the same moving-wall operator that supplied
    ``wall_momentum_contribution_x``; their sum is an action/reaction residual.
    Legacy ``wall_me_load_x`` remains explicitly non-comparable because it is
    sampled before the moving-wall operator by a static estimator.
    """
    if not samples:
        return {"available": False, "status": "withheld", "reason": "no_post_warmup_samples"}
    if not all(bool(sample.get("open_faces_available", False)) for sample in samples):
        return {
            "available": False,
            "status": "withheld",
            "reason": "open_face_momentum_flux_unavailable",
            "sample_count": len(samples),
        }

    def mean(name: str) -> float:
        return sum(float(sample[name]) for sample in samples) / len(samples)

    wall_cv = mean("wall_momentum_contribution_x")
    wall_reaction = mean("wall_reaction_x")
    wall_me = mean("wall_me_load_x")
    ar_residuals = [
        float(sample["wall_momentum_contribution_x"]) + float(sample["wall_reaction_x"])
        for sample in samples
    ]
    ar_absolute = [abs(value) for value in ar_residuals]
    ar_relative = [
        absolute / max(abs(float(sample["wall_momentum_contribution_x"])), 1e-30)
        for absolute, sample in zip(ar_absolute, samples)
    ]
    torque_fields = (
        "wall_fluid_torque_impulse_x",
        "wall_reaction_torque_x",
    )
    missing_torque_fields = sorted({
        field for sample in samples for field in torque_fields if field not in sample
    })
    if missing_torque_fields:
        # Historical force-only samples predate same-operator torque output.
        # Do not infer torque from force or substitute zero: that would turn an
        # unavailable diagnostic into a fabricated action/reaction result.
        torque_action_reaction: dict[str, object] = {
            "status": "withheld",
            "reason": "missing_same_operator_torque_fields",
            "sample_count": len(samples),
            "missing_fields": missing_torque_fields,
            "required_fields": list(torque_fields),
        }
    else:
        torque_ar_residuals = [
            float(sample["wall_fluid_torque_impulse_x"]) + float(sample["wall_reaction_torque_x"])
            for sample in samples
        ]
        torque_ar_absolute = [abs(value) for value in torque_ar_residuals]
        torque_ar_relative = [
            absolute / max(abs(float(sample["wall_fluid_torque_impulse_x"])), 1e-30)
            for absolute, sample in zip(torque_ar_absolute, samples)
        ]
        torque_action_reaction = {
            "status": "comparable",
            "sample_count": len(samples),
            "fluid_torque_impulse_x_mean": mean("wall_fluid_torque_impulse_x"),
            "body_reaction_torque_x_mean": mean("wall_reaction_torque_x"),
            "signed_residual_x_mean": sum(torque_ar_residuals) / len(torque_ar_residuals),
            "absolute_residual_x_mean": sum(torque_ar_absolute) / len(torque_ar_absolute),
            "absolute_residual_x_max": max(torque_ar_absolute),
            "relative_residual_x_max": max(torque_ar_relative),
            "definition": (
                "fluid torque impulse and body reaction torque are emitted by "
                "the same moving_wall_bounce_back_3d update about the propeller axis"
            ),
        }
    return {
        "available": True,
        "status": "comparable",
        "method": "discrete_full_control_volume_momentum_budget",
        "sample_count": len(samples),
        "fluid_momentum_delta_x_mean": mean("fluid_momentum_delta_x"),
        "collision_momentum_contribution_x_mean": mean("collision_momentum_contribution_x"),
        "streaming_momentum_contribution_x_mean": mean("streaming_momentum_contribution_x"),
        "open_face_momentum_flux_x_mean": mean("open_face_momentum_flux_x"),
        "fixed_channel_wall_momentum_contribution_x_mean": mean("fixed_channel_wall_momentum_contribution_x"),
        "moving_mask_reset_momentum_contribution_x_mean": mean("moving_mask_reset_momentum_contribution_x"),
        "wall_momentum_contribution_x_mean": wall_cv,
        "wall_reaction_x_mean": wall_reaction,
        "wall_me_load_x_mean": wall_me,
        "same_operator_action_reaction_status": "comparable",
        "same_operator_action_reaction_residual_x_mean": sum(ar_residuals) / len(ar_residuals),
        "same_operator_action_reaction_abs_residual_x_mean": sum(ar_absolute) / len(ar_absolute),
        "same_operator_action_reaction_abs_residual_x_max": max(ar_absolute),
        "same_operator_action_reaction_relative_residual_max": max(ar_relative),
        "same_operator_torque_action_reaction": torque_action_reaction,
        "budget_residual_x_mean": mean("budget_residual_x"),
        "me_vs_cv_comparison_status": "noncomparable",
        "me_vs_cv_comparison_reason": (
            "wall_me_load_x is a legacy static estimator sampled before "
            "moving_wall_bounce_back_3d; it is not the same discrete operator "
            "as wall_reaction_x and remains non-comparable"
        ),
        "me_vs_cv_wall_nonclosure_x_mean": wall_cv + wall_me,
        "me_vs_cv_wall_nonclosure_abs_x_mean": abs(wall_cv + wall_me),
        # Normalize against the magnitude of the reported ME body load.  This
        # is a size diagnostic only, not a criterion, because the operators
        # above are deliberately marked non-comparable.
        "me_vs_cv_wall_nonclosure_rel": abs(wall_cv + wall_me) / max(abs(wall_me), 1e-30),
        "sign_convention": {
            "positive_x": "positive streamwise (+x) momentum",
            "cv_terms": "positive values add momentum to the fluid distributions in the CV",
            "wall_momentum_contribution_x": "positive values are moving-wall-operator momentum gained by fluid",
            "wall_reaction_x": "negative same-operator body reaction to the fluid wall impulse",
            "wall_me_load_x": "legacy static ME body load; non-comparable to moving-wall CV",
            "residual": "fluid_delta - sum(all sampled CV contributions)",
        },
    }


def _max_sample_field(
    results: list[dict[str, object]], field: str,
) -> float:
    """Return a finite maximum across actual post-warmup campaign samples."""
    values = [
        abs(float(sample[field]))
        for result in results
        for sample in result["samples"]  # type: ignore[index]
    ]
    return max(values) if values else float("nan")


def _resolution_level_summary(name: str, summary: dict[str, object]) -> dict[str, object]:
    """Keep repeatability evidence tied to actual executed per-J samples."""
    results = summary["results"]
    assert isinstance(results, list)
    campaign = summary["campaign"]
    assert isinstance(campaign, dict)
    config = summary["config"]
    assert isinstance(config, dict)
    return {
        "name": name,
        "n_j_cases": campaign["n_j_cases"],
        "campaign_status": campaign["status"],
        "per_j_window_status": campaign["per_j_window_status"],
        "tip_ma": config["tip_ma"],
        "re_d": config["re_d"],
        "temporal_angular_resolution": config["temporal_angular_resolution"],
        "resolution": {
            "diameter_lu": config["diameter_lu"],
            "nx": config["nx"], "ny": config["ny"], "nz": config["nz"],
        },
        "action_reaction_residual": {
            "force_absolute_max": _max_sample_field(
                results, "wall_action_reaction_absolute_residual_norm",
            ),
            "torque_absolute_max": _max_sample_field(
                results, "wall_torque_action_reaction_absolute_residual_norm",
            ),
            "force_relative_max": _max_sample_field(
                results, "wall_action_reaction_relative_residual",
            ),
            "torque_relative_max": _max_sample_field(
                results, "wall_torque_action_reaction_relative_residual",
            ),
        },
    }


def _require_resolution_sensitivity_contract(
    configs: tuple[PropellerBenchmarkConfig, ...],
) -> dict[str, object]:
    """Reject non-grid changes before executing an expensive comparison."""
    def exact_steps_per_revolution(config: PropellerBenchmarkConfig) -> int:
        """Return a true integer period, never a rounded angular surrogate."""
        steps = config.steps_per_revolution
        actual_increment = config.angular_increment_degrees
        claimed_increment = 360.0 / steps
        if not math.isclose(
            config.rpm, 1.0 / steps,
            rel_tol=_SENSITIVITY_MATCH_TOL, abs_tol=_SENSITIVITY_MATCH_TOL,
        ):
            raise ValueError(
                "resolution sensitivity rpm must be the reciprocal of an integer "
                "steps_per_revolution: actual angular increment "
                f"{actual_increment:.17g} deg/update disagrees with claimed "
                f"{claimed_increment:.17g} deg/update (rpm={config.rpm:.17g}, "
                f"nearest steps_per_revolution={steps})"
            )
        return steps

    def require_complete_rotation_schedule(
        config: PropellerBenchmarkConfig, steps: int,
    ) -> None:
        if config.sampling_steps is not None:
            raise ValueError("resolution sensitivity requires sampling_steps=None; derive samples from complete revolutions")
        # This campaign policy uses whole rotations for each temporal interval.
        if config.sample_window_steps % steps:
            raise ValueError(
                "resolution sensitivity requires sample_window_steps to contain "
                "an exact whole number of rotations"
            )
        if config.warmup_steps % steps:
            raise ValueError(
                "resolution sensitivity requires warmup_steps to contain an "
                "exact whole number of rotations"
            )
        total_sampling_steps = config.n_revolutions * steps
        if total_sampling_steps % config.sample_window_steps:
            raise ValueError(
                "resolution sensitivity requires total sampling steps "
                "(n_revolutions * steps_per_revolution) to divide exactly "
                "into complete sample windows"
            )

    if len(configs) not in (2, 3):
        raise ValueError("resolution sensitivity requires a two- or three-level coarse-to-fine sequence")
    baseline = configs[0]
    baseline.validate()
    baseline_steps = exact_steps_per_revolution(baseline)
    require_complete_rotation_schedule(baseline, baseline_steps)
    base_j = tuple(v / (baseline.rpm * baseline.geometry.diameter) for v in baseline.inflow_velocities)
    base_window_revs = baseline.sample_window_steps / baseline_steps
    base_warmup_revs = baseline.warmup_steps / baseline_steps
    levels: list[dict[str, object]] = []
    for index, config in enumerate(configs):
        config.validate()
        steps = exact_steps_per_revolution(config)
        require_complete_rotation_schedule(config, steps)
        j_values = tuple(v / (config.rpm * config.geometry.diameter) for v in config.inflow_velocities)
        same = lambda a, b: math.isclose(a, b, rel_tol=_SENSITIVITY_MATCH_TOL, abs_tol=_SENSITIVITY_MATCH_TOL)
        if len(j_values) != len(base_j) or not all(same(a, b) for a, b in zip(j_values, base_j)):
            raise ValueError("resolution sensitivity requires matched advance ratios J")
        if not same(config.re_d, baseline.re_d):
            raise ValueError("resolution sensitivity requires matched Re_D; adjust nu/tau safely before comparing grids")
        if config.tip_ma > LOW_MACH_TIP_GATE:
            raise ValueError(f"resolution sensitivity withheld: tip Mach exceeds low-Mach gate {LOW_MACH_TIP_GATE}")
        if not same(config.tip_ma, baseline.tip_ma):
            raise ValueError("resolution sensitivity requires matched tip Mach")
        # The moving geometry advances once per lattice update.  Dimensionless
        # flow matches alone cannot make two different rotor update histories
        # comparable, so enforce both explicit angular/temporal invariants.
        if steps != baseline_steps:
            raise ValueError(
                "resolution sensitivity requires matched steps_per_revolution "
                "and angular_increment_degrees"
            )
        angular_increment = config.angular_increment_degrees
        baseline_angular_increment = baseline.angular_increment_degrees
        if not same(angular_increment, baseline_angular_increment):
            raise ValueError(
                "resolution sensitivity requires matched angular_increment_degrees"
            )
        if config.n_revolutions != baseline.n_revolutions:
            raise ValueError("resolution sensitivity requires the same sampled number of complete revolutions")
        window_revs = config.sample_window_steps / steps
        warmup_revs = config.warmup_steps / steps
        if not same(window_revs, base_window_revs):
            raise ValueError("resolution sensitivity requires matched complete-window duration in revolutions")
        if not same(warmup_revs, base_warmup_revs):
            raise ValueError("resolution sensitivity requires matched warmup duration in revolutions")
        levels.append({
            "level_index": index, "advance_ratios": list(j_values), "tip_ma": config.tip_ma,
            "re_d": config.re_d, "steps_per_revolution": steps,
            "angular_increment_degrees": config.angular_increment_degrees,
            "sampled_revolutions": config.n_revolutions, "warmup_revolutions": warmup_revs,
            "window_revolutions": window_revs,
            "complete_windows": config.n_revolutions / window_revs,
        })
        if index:
            previous = configs[index - 1]
            if not (config.nx > previous.nx and config.ny > previous.ny and config.nz > previous.nz):
                raise ValueError(
                    "resolution sensitivity levels must be strictly coarse-to-fine in nx, ny, and nz"
                )
    return {"low_mach_gate": LOW_MACH_TIP_GATE, "levels": levels}


def _legacy_run_propeller_resolution_sensitivity(
    configs: tuple[PropellerBenchmarkConfig, ...], *, level_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Legacy two-domain comparator retained privately; not convergence evidence.

    Each level is independently run by :func:`run_propeller_benchmark`; no
    coefficient is interpolated or synthesized. The evidence reports final-
    window status and same-operator force/torque action-reaction residuals for
    every level, plus changes at matching J relative to the first level.
    """
    if len(configs) not in (2, 3):
        raise ValueError("resolution sensitivity requires a two- or three-level coarse-to-fine sequence")
    if level_names is None:
        level_names = tuple(f"level_{index}" for index in range(len(configs)))
    if len(level_names) != len(configs) or len(set(level_names)) != len(level_names):
        raise ValueError("level_names must be unique and match configs")
    contract = _require_resolution_sensitivity_contract(configs)

    summaries = [run_propeller_benchmark(config) for config in configs]
    levels = [
        _resolution_level_summary(name, summary)
        for name, summary in zip(level_names, summaries)
    ]
    baseline_results = summaries[0]["results"]
    assert isinstance(baseline_results, list)
    changes_from_baseline: list[dict[str, object]] = []
    matched_j = True
    for level_name, summary in zip(level_names[1:], summaries[1:]):
        results = summary["results"]
        assert isinstance(results, list)
        if len(results) != len(baseline_results):
            matched_j = False
            continue
        for baseline, current in zip(baseline_results, results):
            baseline_j = float(baseline["j_actual"])
            current_j = float(current["j_actual"])
            if not math.isclose(current_j, baseline_j, rel_tol=1e-12, abs_tol=1e-12):
                matched_j = False
                continue
            changes_from_baseline.append({
                "level": level_name,
                "j_actual": current_j,
                "kt_relative_change": _relative_change(float(baseline["kt"]), float(current["kt"])),
                "kq_relative_change": _relative_change(float(baseline["kq"]), float(current["kq"])),
                "eta_o_relative_change": _relative_change(float(baseline["eta_o"]), float(current["eta_o"])),
            })
    tip_machs = [float(level["tip_ma"]) for level in levels]
    low_mach_matched = all(tip_ma <= LOW_MACH_TIP_GATE for tip_ma in tip_machs) and all(
        math.isclose(tip_ma, tip_machs[0], rel_tol=1e-12, abs_tol=1e-12)
        for tip_ma in tip_machs[1:]
    )
    all_window_converged = all(
        bool(status["convergence"].get("window_converged", False))
        for level in levels
        for status in level["per_j_window_status"]  # type: ignore[index]
    )
    evidence = {
        "name": "propeller_open_water_resolution_sensitivity",
        "status": "converged" if all_window_converged else "not_converged",
        "comparison_basis": {
            "baseline_level": level_names[0],
            "advance_ratios_matched": matched_j,
            "low_mach_matched": low_mach_matched,
            "low_mach_gate": LOW_MACH_TIP_GATE,
            "tip_ma_by_level": dict(zip(level_names, tip_machs)),
            "re_d_matched": True,
            "temporal_angular_contract": contract,
            "coefficient_change_definition": "abs(current - baseline) / max(abs(current), 1e-30)",
        },
        "levels": levels,
        "changes_from_baseline": changes_from_baseline,
    }
    output_root = configs[0].output_root
    evidence_path = output_root / "propeller_owt" / "resolution_sensitivity.json"
    evidence["artifact"] = str(evidence_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


# ============================================================================
# Single-speed simulation
# ============================================================================

def _run_single_speed(
    *, config: PropellerBenchmarkConfig, u_in: float,
) -> dict[str, object]:
    """Run a single inflow-velocity simulation and return results."""
    device = resolve_device(config.device)
    geo = config.geometry
    D = geo.diameter

    nx, ny, nz = config.nx, config.ny, config.nz
    cx = int(nx * 0.35)
    cy = ny // 2
    cz = nz // 2

    # Re-voxelize at each physical azimuth.  The solver advances these masks;
    # this is deliberately not a static-mask surrogate campaign.
    previous_mask = build_propeller_mask(
        nx=nx, ny=ny, nz=nz, cx=cx, cy=cy, cz=cz,
        angle_deg=0.0, config=geo, device=str(device),
    )

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux0 = torch.full_like(rho0, u_in)
    ux0[previous_mask] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)

    steps_per_rev = config.steps_per_revolution
    n_sampling = config.sampling_steps if config.sampling_steps is not None else config.n_revolutions * steps_per_rev
    n_total = config.warmup_steps + n_sampling

    fx_samples: list[float] = []
    mx_samples: list[float] = []
    me_samples: list[dict[str, float | int]] = []
    campaign_samples: list[dict[str, float | int]] = []
    t_start = time.perf_counter()

    for step in range(1, n_total + 1):
        azimuth_deg = math.degrees((step * config.omega) % (2.0 * math.pi))
        mask = build_propeller_mask(
            nx=nx, ny=ny, nz=nz, cx=cx, cy=cy, cz=cz,
            angle_deg=azimuth_deg, config=geo, device=str(device),
        )
        wall_mask = make_channel_wall_mask_3d(nz, ny, nx, mask, device=device)
        ux_w, uy_w, uz_w = rotating_wall_velocity_3d(mask, cx, cy, cz, config.omega)
        # Account for every operator in the actual order used by this campaign.
        # Positive terms below mean +x momentum added to the fluid CV.
        momentum_start = _d3q19_momentum_x(f)
        f = collide_smagorinsky_mrt3d(f, tau=config.tau, C_s=config.smagorinsky_cs)
        momentum_after_collision = _d3q19_momentum_x(f)
        f = stream3d(f)
        momentum_after_streaming = _d3q19_momentum_x(f)
        fx, _, _ = compute_obstacle_forces_3d(f, mask)
        mx, _, _ = compute_obstacle_moments_3d(f, mask, cx, cy, cz)

        # The channel routine combines x-open-face reconstruction and the four
        # fixed transverse bounce-back walls.  Sample the disjoint interior of
        # the x faces; retain the complementary update as its own fixed-wall
        # contribution, including edge/corner treatment.
        f_before_boundary = f.clone()
        open_face_mask = torch.zeros_like(mask)
        open_face_mask[1:-1, 1:-1, 0] = True
        open_face_mask[1:-1, 1:-1, -1] = True
        f = apply_zou_he_channel_boundaries_3d(
            f, u_in=u_in, wall_mask=wall_mask,
            obstacle_mask=torch.zeros_like(mask),
        )
        momentum_after_boundary = _d3q19_momentum_x(f)
        open_face_delta = (
            _d3q19_momentum_x(f, open_face_mask)
            - _d3q19_momentum_x(f_before_boundary, open_face_mask)
        )
        fixed_channel_wall_delta = (
            momentum_after_boundary - momentum_after_streaming - open_face_delta
        )

        momentum_before_wall = momentum_after_boundary
        f, wall_reaction = moving_wall_bounce_back_3d_with_reaction(
            f, mask, ux_w, uy_w, uz_w, origin=(float(cx), float(cy), float(cz)),
        )
        momentum_after_wall = _d3q19_momentum_x(f)

        # Cells released by the moving solid receive a finite equilibrium state;
        # its momentum change is an explicit CV term, not an omitted remainder.
        released = previous_mask & ~mask
        momentum_before_reset = momentum_after_wall
        if bool(released.any()):
            equilibrium = equilibrium3d(
                torch.ones_like(rho0), torch.full_like(rho0, u_in),
                torch.zeros_like(rho0), torch.zeros_like(rho0), device=device,
            )
            f[:, released] = equilibrium[:, released]
        momentum_after_reset = _d3q19_momentum_x(f)
        previous_mask = mask

        if step > config.warmup_steps:
            fx_value = float(fx.item())
            mx_value = float(mx.item())
            kt_sample, kq_sample, j_sample, _ = _compute_kt_kq(
                fx_value, mx_value, u_in=u_in, rpm=config.rpm, diameter=D,
            )
            fx_samples.append(fx_value)
            mx_samples.append(mx_value)
            me_samples.append({
                "step": step,
                "fx_me_lu": fx_value,
                "mx_me_lu": mx_value,
            })
            collision_delta = momentum_after_collision - momentum_start
            streaming_delta = momentum_after_streaming - momentum_after_collision
            # Use the link-wise population-delta accumulator emitted by the same
            # operator as the CV wall term. This fixes its reduction order and
            # makes the recorded body reaction exactly comparable; the complete
            # CV budget still exposes any independent global-reduction roundoff.
            moving_wall_delta = wall_reaction.fluid_impulse[0]
            wall_cv_reaction_residual = moving_wall_delta + wall_reaction.body_reaction[0]
            wall_cv_reaction_abs = wall_cv_reaction_residual.abs()
            wall_cv_reaction_relative = wall_cv_reaction_abs / wall_reaction.fluid_impulse[0].abs().clamp_min(1e-30)
            reset_delta = momentum_after_reset - momentum_before_reset
            fluid_delta = momentum_after_reset - momentum_start
            budget_sum = collision_delta + streaming_delta + open_face_delta + fixed_channel_wall_delta + moving_wall_delta + reset_delta
            budget_residual = fluid_delta - budget_sum
            campaign_samples.append({
                "step": step,
                "azimuth_deg": azimuth_deg,
                "j": j_sample,
                "kt": kt_sample,
                "kq": kq_sample,
                "fx_me_lu": fx_value,
                "mx_me_lu": mx_value,
                "fluid_momentum_delta_x": float(fluid_delta.item()),
                "collision_momentum_contribution_x": float(collision_delta.item()),
                "streaming_momentum_contribution_x": float(streaming_delta.item()),
                "open_face_momentum_flux_x": float(open_face_delta.item()),
                "fixed_channel_wall_momentum_contribution_x": float(fixed_channel_wall_delta.item()),
                "moving_mask_reset_momentum_contribution_x": float(reset_delta.item()),
                "wall_momentum_contribution_x": float(moving_wall_delta.item()),
                "wall_reaction_x": float(wall_reaction.body_reaction[0].item()),
                "wall_fluid_impulse_x": float(wall_reaction.fluid_impulse[0].item()),
                "wall_fluid_torque_impulse_x": float(wall_reaction.fluid_torque_impulse[0].item()),
                "wall_reaction_torque_x": float(wall_reaction.body_reaction_torque[0].item()),
                "wall_action_reaction_signed_residual_norm": float(wall_cv_reaction_residual.item()),
                "wall_action_reaction_absolute_residual_norm": float(wall_cv_reaction_abs.item()),
                "wall_action_reaction_relative_residual": float(wall_cv_reaction_relative.item()),
                "wall_torque_action_reaction_signed_residual_norm": wall_reaction.torque_action_reaction_signed_residual_norm,
                "wall_torque_action_reaction_absolute_residual_norm": wall_reaction.torque_action_reaction_absolute_residual_norm,
                "wall_torque_action_reaction_relative_residual": wall_reaction.torque_action_reaction_relative_residual,
                "wall_me_load_x": fx_value,
                "budget_residual_x": float(budget_residual.item()),
                "open_faces_available": True,
            })

        if step % 2000 == 0 or step == n_total:
            elapsed = time.perf_counter() - t_start
            pct = 100 * step / n_total
            print(f"  u_in={u_in:.3f}  step {step}/{n_total} ({pct:.0f}%)  "
                  f"elapsed={elapsed:.1f}s")

    fx_mean = sum(fx_samples) / max(len(fx_samples), 1)
    mx_mean = sum(mx_samples) / max(len(mx_samples), 1)
    kt, kq, j_actual, eta = _compute_kt_kq(fx_mean, mx_mean, u_in=u_in, rpm=config.rpm, diameter=D)

    # Physical-scale conversion
    kt_phys, kq_phys, n_phys_rps = _convert_to_physical_kt_kq(
        kt, kq, j_actual, config.rpm, D, u_in,
        d_phys=config.model_diameter_m, u_phys=config.model_speed_ms,
        rho_phys=config.model_rho_kgm3,
    )

    geo_stats = propeller_statistics(geo, previous_mask)
    window_report = _summarize_windows(
        campaign_samples,
        window_steps=config.sample_window_steps,
        transient_discard_steps=0,
        convergence_rel_tol=config.window_convergence_rel_tol,
    )
    control_volume_cross_check = _summarize_control_volume_cross_check(campaign_samples)
    re_d = config.rpm * D * D / config.nu

    return {
        "u_in": u_in, "j_actual": j_actual,
        "fx_mean_lu": fx_mean, "mx_mean_lu": mx_mean,
        "kt": kt, "kq": kq, "eta_o": eta,
        "kt_over_j2": kt / max(j_actual**2, 1e-10),
        "kq_over_j2": kq / max(j_actual**2, 1e-10),
        "kt_phys": kt_phys, "kq_phys": kq_phys, "n_phys_rps": n_phys_rps,
        "re_d": re_d, "steps": n_total,
        "sampling_steps": len(fx_samples),
        "transient_discard_steps": config.warmup_steps,
        "dynamic_geometry": True,
        "samples": campaign_samples,
        "me_samples": me_samples,
        "window_report": window_report,
        "control_volume_cross_check": control_volume_cross_check,
        "geometry": geo_stats,
        "runtime_s": time.perf_counter() - t_start,
    }


# ============================================================================
# Main benchmark runner
# ============================================================================

def run_propeller_benchmark(config: PropellerBenchmarkConfig) -> dict[str, object]:
    """Run propeller open-water benchmark over multiple inflow velocities."""
    config.validate()
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    tip_ma = config.tip_ma
    print(f"Propeller Open-Water Benchmark (fixed-RPM variable-inflow)")
    print(f"  Device:     {device}     Blades: {config.geometry.n_blades}")
    print(f"  Diameter:   {config.geometry.diameter} lu   "
          f"P/D(0.7R): {config.geometry.pitch_ratio_07:.3f}")
    print(f"  Domain:     {config.nx}x{config.ny}x{config.nz}   "
          f"tau: {config.tau:.3f}  Cs: {config.smagorinsky_cs:.2f}")
    print(f"  RPM:        {config.rpm:.2e}  omega={config.omega:.2e}  "
          f"tip Ma={tip_ma:.4f}")
    j_vals = [v / (config.rpm * config.geometry.diameter) for v in config.inflow_velocities]
    print(f"  J range:    {[f'{j:.1f}' for j in j_vals]}")
    print()

    run_dir = prepare_run_dir(
        config.output_root, "propeller_owt",
        config.resolved_run_name, config.overwrite,
    )
    config.save(run_dir / "config.json")
    print(f"Run directory: {run_dir}\n")

    results: list[dict[str, object]] = []
    for u_in in config.inflow_velocities:
        j_est = u_in / (config.rpm * config.geometry.diameter)
        print(f"{'='*60}")
        print(f"  u_in = {u_in:.3f} (J approx {j_est:.1f})")
        print(f"{'='*60}")
        result = _run_single_speed(config=config, u_in=u_in)
        results.append(result)
        kt_p = result.get("kt_phys", float("nan"))
        kq_p = result.get("kq_phys", float("nan"))
        n_p = result.get("n_phys_rps", 0.0)
        print(f"  -> KT_lu={float(result['kt']):.0f}  KT_phys={float(kt_p):.4f}  "
              f"10KQ_phys={10 * float(kq_p):.4f}  "
              f"n={float(n_p):.2f}rps  eta={float(result['eta_o']):.4f}\n")

    # Write CSV
    csv_path = run_dir / "open_water.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["J", "KT_lu", "KT_phys", "10KQ_phys", "eta_o", "n_phys_rps", "Re_D"])
        for r in results:
            writer.writerow([
                f"{float(r['j_actual']):.4f}", f"{float(r['kt']):.1f}",
                f"{float(r.get('kt_phys', 0)):.6f}",
                f"{10 * float(r.get('kq_phys', 0)):.6f}",
                f"{float(r['eta_o']):.4f}",
                f"{float(r.get('n_phys_rps', 0)):.3f}",
                f"{float(r['re_d']):.1f}",
            ])

    # Summary
    kt_p_vals = [float(r.get("kt_phys", 0)) for r in results]  # type: ignore[arg-type]
    j_vals = [float(r["j_actual"]) for r in results]  # type: ignore[arg-type]
    eta_vals = [float(r["eta_o"]) for r in results]  # type: ignore[arg-type]
    per_j_window_status: list[dict[str, object]] = []
    for result in results:
        window_report = result["window_report"]
        assert isinstance(window_report, dict)
        convergence = window_report["convergence"]
        assert isinstance(convergence, dict)
        per_j_window_status.append({
            "j_actual": float(result["j_actual"]),
            "complete_window_count": len(window_report["windows"]),
            "convergence": convergence,
        })
    campaign_converged = all(
        bool(status["convergence"].get("window_converged", False))
        for status in per_j_window_status
        if isinstance(status["convergence"], dict)
    ) and len(per_j_window_status) == len(results)

    summary = {
        "name": "propeller_open_water",
        "config": {
            "n_blades": config.geometry.n_blades,
            "diameter_lu": config.geometry.diameter,
            "pitch_ratio_07": config.geometry.pitch_ratio_07,
            "ae_a0": config.geometry.blade_area_ratio,
            "tip_ma": tip_ma,
            "nx": config.nx, "ny": config.ny, "nz": config.nz,
            "tau": config.tau, "cs": config.smagorinsky_cs,
            "rpm": config.rpm, "nu_lattice": config.nu,
            "re_d": config.re_d,
            "n_revolutions": config.n_revolutions,
            "temporal_angular_resolution": {
                "steps_per_revolution": config.steps_per_revolution,
                "angular_increment_degrees": config.angular_increment_degrees,
                "sampling_steps_requested": config.sampling_steps,
                "sampling_steps_executed": (
                    config.sampling_steps if config.sampling_steps is not None
                    else config.n_revolutions * config.steps_per_revolution
                ),
                "sampled_revolutions_executed": (
                    (config.sampling_steps * config.rpm)
                    if config.sampling_steps is not None else float(config.n_revolutions)
                ),
                "warmup_revolutions": config.warmup_steps * config.rpm,
                "window_revolutions": config.sample_window_steps * config.rpm,
            },
            "sample_window_steps": config.sample_window_steps,
            "window_convergence_rel_tol": config.window_convergence_rel_tol,
            "model_diameter_m": config.model_diameter_m,
            "model_speed_ms": config.model_speed_ms,
        },
        "results": results,
        "campaign": {
            "n_j_cases": len(results),
            "status": "converged" if campaign_converged else "not_converged",
            "per_j_window_status": per_j_window_status,
        },
        "summary": {
            "j_range": [min(j_vals), max(j_vals)],
            "kt_phys_range": [min(kt_p_vals), max(kt_p_vals)],
            "eta_max": max(eta_vals),
            "eta_max_j": j_vals[eta_vals.index(max(eta_vals))],
        },
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
    }

    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    # Print summary
    print(f"\n{'='*70}")
    print("  Open-Water Results")
    print(f"{'='*70}")
    print(f"  {'J':>6s}  {'KT_phys':>10s}  {'10KQ_phys':>10s}  "
          f"{'eta':>8s}  {'n(rps)':>8s}  {'Re_D':>8s}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {float(r['j_actual']):6.2f}  "
              f"{float(r.get('kt_phys', 0)):10.6f}  "
              f"{10 * float(r.get('kq_phys', 0)):10.6f}  "
              f"{float(r['eta_o']):8.4f}  "
              f"{float(r.get('n_phys_rps', 0)):8.3f}  "
              f"{float(r['re_d']):8.0f}")
    print(f"  {'='*60}")
    print(f"  max eta = {max(eta_vals):.4f} at J = {j_vals[eta_vals.index(max(eta_vals))]:.2f}")
    print(f"\n  CSV:  {csv_path}")
    print(f"  JSON: {metadata_path}")

    return summary


__all__ = [
    "PropellerBenchmarkConfig",
    "MovingWallReaction3D",
    "rotating_wall_velocity_3d",
    "moving_wall_bounce_back_3d",
    "moving_wall_bounce_back_3d_with_reaction",
    "run_propeller_resolution_sensitivity",
    "run_propeller_benchmark",
]
