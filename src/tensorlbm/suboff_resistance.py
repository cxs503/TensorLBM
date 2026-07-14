"""SUBOFF resistance benchmark with iterative voxel-refinement control."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .adaptive_refinement import (
    AdaptationSchedule,
    AdaptiveSolver3D,
    nonequilibrium_indicator_3d,
)
from .boundaries3d import apply_zou_he_channel_boundaries_3d, make_channel_wall_mask_3d
from .d3q19 import equilibrium3d, macroscopic3d
from .obstacles import compute_obstacle_forces_3d
from .solver3d import stream3d
from .suboff_cad import SuboffConfig, SuboffHullType, build_suboff_mask, suboff_statistics
from .turbulence import collide_smagorinsky_mrt3d
from .utils import resolve_device
from .rans_ke import KESolver
from .wall_model import apply_wall_model_bounce_back


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
    lbm_u_in: float = 0.06
    lbm_tau: float = 0.58
    lbm_steps: int = 60
    lbm_warmup_steps: int = 20
    lbm_sample_interval: int = 5
    conservation_max_relative_mass_drift: float = 1.0e-3
    conservation_max_relative_momentum_drift: float = 1.0e-3
    smagorinsky_cs: float = 0.1
    max_length_lu: float = 80.0
    use_adaptive_mesh: bool = False
    use_wall_model: bool = False
    use_rans_ke: bool = False
    adaptive_l1_pad: int = 4
    adaptive_l2_margin: int = 1
    adaptive_l2_pad: int = 2
    adaptive_interval: int = 5
    adaptive_refine_threshold: float = 1.0e-4
    adaptive_coarsen_threshold: float = 1.0e-6
    adaptive_max_patches: int = 8
    geometry: SuboffConfig = field(default_factory=SuboffConfig)
    # --- snapshot export (for ML training data) ---
    save_snapshots: bool = False
    snapshot_dir: str = "./suboff_snapshots"
    snapshot_start_step: int = 0
    snapshot_end_step: int = 0
    snapshot_interval: int = 1
    snapshot_crop_size: int = 100

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
        if not (0.0 < self.lbm_u_in < 0.15):
            raise ValueError("lbm_u_in must be in (0, 0.15)")
        if self.lbm_tau <= 0.5:
            raise ValueError("lbm_tau must be > 0.5")
        if self.lbm_steps < 10:
            raise ValueError("lbm_steps must be >= 10")
        if self.lbm_warmup_steps < 0:
            raise ValueError("lbm_warmup_steps must be >= 0")
        if self.lbm_sample_interval < 1:
            raise ValueError("lbm_sample_interval must be >= 1")
        if (not math.isfinite(self.conservation_max_relative_mass_drift)
                or self.conservation_max_relative_mass_drift < 0.0):
            raise ValueError("conservation_max_relative_mass_drift must be finite and >= 0")
        if (not math.isfinite(self.conservation_max_relative_momentum_drift)
                or self.conservation_max_relative_momentum_drift < 0.0):
            raise ValueError("conservation_max_relative_momentum_drift must be finite and >= 0")
        if self.max_length_lu < 20.0:
            raise ValueError("max_length_lu must be >= 20")
        if self.adaptive_l1_pad < 0:
            raise ValueError("adaptive_l1_pad must be >= 0")
        if self.adaptive_l2_margin < 0:
            raise ValueError("adaptive_l2_margin must be >= 0")
        if self.adaptive_l2_pad < 0:
            raise ValueError("adaptive_l2_pad must be >= 0")
        if self.adaptive_interval < 1:
            raise ValueError("adaptive_interval must be >= 1")
        if self.adaptive_refine_threshold <= 0.0:
            raise ValueError("adaptive_refine_threshold must be > 0")
        if self.adaptive_coarsen_threshold <= 0.0:
            raise ValueError("adaptive_coarsen_threshold must be > 0")
        if self.adaptive_coarsen_threshold >= self.adaptive_refine_threshold:
            raise ValueError(
                "adaptive_coarsen_threshold must be < adaptive_refine_threshold"
            )
        if self.adaptive_max_patches < 1:
            raise ValueError("adaptive_max_patches must be >= 1")

    @property
    def resolved_radius_m(self) -> float:
        if self.radius_m is not None:
            return float(self.radius_m)
        return self.geometry.r_over_l * self.length_m


def _ittc57_friction_coefficient(reynolds: float) -> float:
    if reynolds <= 100.0:
        raise ValueError("Reynolds number too low for ITTC-1957 formula")
    return 0.075 / (math.log10(reynolds) - 2.0) ** 2


def _laminar_friction_coefficient(reynolds: float) -> float:
    """Blasius laminar flat-plate Cf.

    Cf = 1.328 / sqrt(Re)

    Valid for Re < 5e5 (laminar boundary layer).
    """
    if reynolds <= 0:
        raise ValueError("Reynolds number must be > 0")
    return 1.328 / math.sqrt(max(reynolds, 1.0))


def _scale_cd_to_physical(
    cd_sim: float,
    re_lbm: float,
    re_phys: float,
) -> float:
    """Scale simulated Cd to physical Reynolds number using form-factor method.

    Assumes the form factor (1+k) = Cd/Cf is geometry-dependent and
    approximately Re-independent.  Then:

      Cf_sim   = 1.328 / sqrt(Re_lbm)      (laminar flat-plate)
      ff       = Cd_sim / Cf_sim            (form factor)
      Cf_phys  = ITTC-57 at Re_phys         (turbulent)
      Cd_pred  = Cf_phys * ff               (predicted physical Cd)

    Returns predicted Cd_phys (wetted-area reference).
    """
    cf_sim = _laminar_friction_coefficient(re_lbm)
    ff = cd_sim / max(cf_sim, 1e-10)
    cf_phys = _ittc57_friction_coefficient(re_phys)
    return cf_phys * ff


def _force_scale_factor(
    rho_phys: float, u_phys: float, l_phys: float,
    rho_lu: float, u_lu: float, l_lu: float,
) -> float:
    """Scale lattice force to physical force.

    F_phys = F_lu * (rho_phys/rho_lu) * (u_phys/u_lu)^2 * (l_phys/l_lu)^2
    """
    return rho_phys / rho_lu * (u_phys / u_lu) ** 2 * (l_phys / l_lu) ** 2


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


def voxel_wetted_area_x_slab(
    mask: torch.Tensor,
    dx: float,
    *,
    has_left_neighbor: bool,
    has_right_neighbor: bool,
) -> float:
    """Return wetted area of an x-decomposed physical slab.

    Interior rank cuts are communication interfaces, not exposed solid faces.
    Only a slab touching a physical x-domain boundary contributes that end
    face, making the summed area partition invariant.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.ndim != 3:
        raise ValueError("mask must be a 3D tensor")

    m = mask
    area_faces = torch.tensor(0, dtype=torch.int64, device=m.device)
    if not has_left_neighbor:
        area_faces += m[:, :, 0].sum()
    if not has_right_neighbor:
        area_faces += m[:, :, -1].sum()
    area_faces += m[:, 0, :].sum()
    area_faces += m[:, -1, :].sum()
    area_faces += m[0, :, :].sum()
    area_faces += m[-1, :, :].sum()
    area_faces += (m[:, :, 1:] != m[:, :, :-1]).sum()
    area_faces += (m[:, 1:, :] != m[:, :-1, :]).sum()
    area_faces += (m[1:, :, :] != m[:-1, :, :]).sum()
    return float(area_faces.item()) * dx * dx


def _crop_central_region(tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Crop the central [crop_size]*3 region of a [nz, ny, nx] tensor."""
    _, ny, nx = tensor.shape[-3:]
    if crop_size >= min(ny, nx):
        return tensor
    z = tensor.shape[0] if tensor.ndim == 3 else 1
    sy = (ny - crop_size) // 2
    sx = (nx - crop_size) // 2
    sz = (max(z, crop_size) - crop_size) // 2
    if tensor.ndim == 3:
        return tensor[sz:sz + crop_size, sy:sy + crop_size, sx:sx + crop_size]
    return tensor[:, sy:sy + crop_size, sx:sx + crop_size]


def _export_snapshot(
    f: torch.Tensor,
    step: int,
    config: SuboffResistanceBenchmarkConfig,
) -> None:
    """Export a single flow-field snapshot as 4 NPY files (p, ux, uy, uz)."""
    import os
    import numpy as np
    base = config.snapshot_dir
    dirs = (f"{base}/p", f"{base}/ux", f"{base}/uy", f"{base}/uz")
    rho, ux, uy, uz = macroscopic3d(f)
    rho_c = _crop_central_region(rho, config.snapshot_crop_size)
    ux_c = _crop_central_region(ux, config.snapshot_crop_size)
    uy_c = _crop_central_region(uy, config.snapshot_crop_size)
    uz_c = _crop_central_region(uz, config.snapshot_crop_size)
    idx = (step - config.snapshot_start_step) // config.snapshot_interval
    for arr, d in [(rho_c, dirs[0]), (ux_c, dirs[1]),
                    (uy_c, dirs[2]), (uz_c, dirs[3])]:
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"{idx}.npy"), arr.cpu().numpy().astype(np.float32))


def _run_suboff_lbm_drag(
    *,
    config: SuboffResistanceBenchmarkConfig,
    hull_type: SuboffHullType,
    nx: int,
    ny: int,
    nz: int,
    length_lu: float,
    radius_lu: float,
) -> tuple[float, float, dict[str, object]]:
    device = resolve_device(config.device)
    mask, _stats = build_suboff_mask(
        hull_type=hull_type,
        nx=nx,
        ny=ny,
        nz=nz,
        length=length_lu,
        radius=radius_lu,
        config=config.geometry,
        device=str(device),
    )
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, mask, device=device)

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux0 = torch.full_like(rho0, config.lbm_u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[mask] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    # k-ε RANS solver initialization
    ke_solver: KESolver | None = None
    if config.use_rans_ke:
        nu_laminar = (config.lbm_tau - 0.5) / 3.0
        ke_solver = KESolver(nu=nu_laminar)
        _, ux_init, uy_init, uz_init = macroscopic3d(f)
        ke_solver.initialize(ux_init, uy_init, uz_init)

    form_stats_lu = suboff_statistics(hull_type, length_lu, radius_lu, config.geometry)
    ref_area_lu = float(form_stats_lu["wetted_area_lu2"])
    dyn_pressure_lu = 0.5 * config.lbm_u_in**2 * max(ref_area_lu, 1e-12)
    drag_samples: list[float] = []
    # Runtime numerical evidence is measured inside the actual solver loop.
    completed_steps = 0
    finite_population_checks = 0
    finite_density_checks = 0
    all_populations_finite = True
    all_densities_finite = True
    density_min = float("inf")
    density_max = float("-inf")
    # Conservation is sampled from the actual population state, including the
    # initialized lattice state before any update.  This is observational data,
    # not an inference from a finite drag coefficient.
    initial_mass = float(f.sum().item())
    initial_rho, initial_ux, initial_uy, initial_uz = macroscopic3d(f)
    initial_momentum = torch.stack(tuple(
        (initial_rho * velocity).sum()
        for velocity in (initial_ux, initial_uy, initial_uz)
    ))
    initial_momentum_norm = float(torch.linalg.vector_norm(initial_momentum).item())
    final_mass = initial_mass
    final_momentum_norm = initial_momentum_norm
    max_abs_mass_drift = 0.0
    max_relative_mass_drift = 0.0
    max_abs_momentum_drift = 0.0
    max_relative_momentum_drift = 0.0
    mass_sample_count = 1
    coarse_cells = int(nx * ny * nz)

    adaptive_solver: AdaptiveSolver3D | None = None
    adaptive_cells: list[float] = []
    if config.use_adaptive_mesh:
        adaptive_solver = AdaptiveSolver3D(
            f,
            schedule=AdaptationSchedule(
                interval=config.adaptive_interval,
                warmup=config.lbm_warmup_steps,
                max_patches=config.adaptive_max_patches,
                refine_threshold=config.adaptive_refine_threshold,
                coarsen_threshold=config.adaptive_coarsen_threshold,
            ),
            mask=mask,
        )

    for step in range(1, config.lbm_steps + 1):
        if config.use_rans_ke and ke_solver is not None:
            _, ux, uy, uz = macroscopic3d(f)
            nu_t = ke_solver.step(ux, uy, uz, mask)
            nu_eff = (config.lbm_tau - 0.5) / 3.0 + nu_t.mean().item()
            tau_eff = min(max(3.0 * nu_eff + 0.5, 0.501), 2.0)
            f = collide_smagorinsky_mrt3d(f, tau=tau_eff, C_s=0.0)
        else:
            f = collide_smagorinsky_mrt3d(f, tau=config.lbm_tau, C_s=config.smagorinsky_cs)
        f = stream3d(f)
        fx, _, _ = compute_obstacle_forces_3d(f, mask)
        if config.use_wall_model:
            _, ux, uy, uz = macroscopic3d(f)
            nu = (config.lbm_tau - 0.5) / 3.0
            f = apply_wall_model_bounce_back(f, mask, ux, uy, uz, nu)
            f = apply_zou_he_channel_boundaries_3d(f, u_in=config.lbm_u_in, wall_mask=wall_mask, obstacle_mask=torch.zeros_like(mask))
        else:
            f = apply_zou_he_channel_boundaries_3d(f, u_in=config.lbm_u_in, wall_mask=wall_mask, obstacle_mask=mask)
        # Record direct per-step state observations. A finite final drag alone
        # cannot stand in for this numerical evidence.
        completed_steps += 1
        populations_finite = bool(torch.isfinite(f).all().item())
        finite_population_checks += 1
        all_populations_finite = all_populations_finite and populations_finite
        rho_step, ux_step, uy_step, uz_step = macroscopic3d(f)
        densities_finite = bool(torch.isfinite(rho_step).all().item())
        finite_density_checks += 1
        all_densities_finite = all_densities_finite and densities_finite
        if densities_finite:
            density_min = min(density_min, float(rho_step.min().item()))
            density_max = max(density_max, float(rho_step.max().item()))
        mass = float(f.sum().item())
        momentum = torch.stack(tuple(
            (rho_step * velocity).sum() for velocity in (ux_step, uy_step, uz_step)
        ))
        momentum_norm = float(torch.linalg.vector_norm(momentum).item())
        final_mass = mass
        final_momentum_norm = momentum_norm
        abs_mass_drift = abs(mass - initial_mass)
        abs_momentum_drift = abs(momentum_norm - initial_momentum_norm)
        max_abs_mass_drift = max(max_abs_mass_drift, abs_mass_drift)
        max_relative_mass_drift = max(
            max_relative_mass_drift, abs_mass_drift / max(abs(initial_mass), 1.0e-30),
        )
        max_abs_momentum_drift = max(max_abs_momentum_drift, abs_momentum_drift)
        max_relative_momentum_drift = max(
            max_relative_momentum_drift,
            abs_momentum_drift / max(abs(initial_momentum_norm), 1.0e-30),
        )
        mass_sample_count += 1
        # --- snapshot export ---
        if config.save_snapshots and step >= config.snapshot_start_step and step <= config.snapshot_end_step and (step - config.snapshot_start_step) % config.snapshot_interval == 0:
            _export_snapshot(f, step, config)
        if adaptive_solver is not None:
            adaptive_solver.coarse_f = f
            if adaptive_solver.should_adapt(step):
                rho, ux, uy, uz = macroscopic3d(f)
                indicator = nonequilibrium_indicator_3d(f, rho, ux, uy, uz)
                adaptive_solver.adapt(indicator)
            adaptive_cells.append(float(adaptive_solver.total_cells))
        if step > config.lbm_warmup_steps and (
            step % config.lbm_sample_interval == 0 or step == config.lbm_steps
        ):
            drag_samples.append(float(fx.item()))

    if adaptive_cells:
        active_cells = int(round(sum(adaptive_cells) / len(adaptive_cells)))
        finest_uniform_cells = int(coarse_cells * 9)
    else:
        active_cells = coarse_cells
        finest_uniform_cells = coarse_cells

    if not drag_samples:
        drag_samples.append(0.0)
    fx_lu = float(sum(drag_samples) / len(drag_samples))
    cd = abs(fx_lu) / dyn_pressure_lu
    mesh_stats = {
        "adaptive": bool(config.use_adaptive_mesh),
        "coarse_cells": coarse_cells,
        "active_cells": active_cells,
        "finest_uniform_cells": finest_uniform_cells,
        "cell_saving_pct": (
            (1.0 - float(active_cells) / max(float(finest_uniform_cells), 1.0)) * 100.0
        ),
        "runtime_evidence": {
            "requested_steps": config.lbm_steps,
            "completed_steps": completed_steps,
            "finite_population_checks": finite_population_checks,
            "finite_density_checks": finite_density_checks,
            "all_populations_finite": all_populations_finite,
            "all_densities_finite": all_densities_finite,
            "density_min": density_min if math.isfinite(density_min) else None,
            "density_max": density_max if math.isfinite(density_max) else None,
            "initial_lattice_mass": initial_mass,
            "final_lattice_mass": final_mass,
            "max_abs_mass_drift": max_abs_mass_drift,
            "max_relative_mass_drift": max_relative_mass_drift,
            "initial_lattice_momentum_norm": initial_momentum_norm,
            "final_lattice_momentum_norm": final_momentum_norm,
            "max_abs_momentum_drift": max_abs_momentum_drift,
            "max_relative_momentum_drift": max_relative_momentum_drift,
            "mass_sample_count": mass_sample_count,
            "sampled_step_count": completed_steps,
        },
    }
    return cd, fx_lu, mesh_stats


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
    ref_area = float(form_stats["wetted_area_lu2"])
    cd_analytical = cf * form_factor
    resistance_analytical_n = 0.5 * config.rho_kgm3 * config.speed_ms**2 * ref_area * cd_analytical

    iterations: list[dict[str, object]] = []
    final_error = float("inf")
    best_cd = float("nan")
    best_resistance_n = float("nan")
    best_cd_lbm_re_analytical = float("nan")
    richardson_cd = float("nan")
    prev_cd: float | None = None
    refinement_ratio = 2.0
    order = 1.0

    for k in range(1, config.max_iterations + 1):
        scale = 2.0 ** float(k - 1)
        length_lu = min(config.base_length_lu * scale, config.max_length_lu)
        radius_lu = (radius_m / config.length_m) * length_lu
        nx = max(int(round(length_lu * 1.8)), int(round(length_lu + 12)))
        ny = max(int(round(radius_lu * 16.0)), 32)
        nz = ny

        # Lattice Reynolds number for this resolution
        nu_lu = (config.lbm_tau - 0.5) / 3.0
        re_lbm = config.lbm_u_in * length_lu / nu_lu
        # Analytical Cd at the lattice Re (laminar flat-plate friction only)
        cf_lbm = _laminar_friction_coefficient(re_lbm)
        cd_lbm_analytical = cf_lbm * form_factor

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
        cd_sim, fx_lu, mesh_stats = _run_suboff_lbm_drag(
            config=config,
            hull_type=hull_type,
            nx=nx,
            ny=ny,
            nz=nz,
            length_lu=length_lu,
            radius_lu=radius_lu,
        )
        # Scale lattice force to physical force
        f_scale = _force_scale_factor(
            config.rho_kgm3, config.speed_ms, config.length_m,
            1.0, config.lbm_u_in, length_lu,
        )
        resistance_sim_n = fx_lu * f_scale
        error_pct: float | None = None
        if prev_cd is not None:
            richardson_cd = cd_sim + (cd_sim - prev_cd) / (refinement_ratio**order - 1.0)
            error_pct = abs(cd_sim - richardson_cd) / max(abs(richardson_cd), 1e-12) * 100.0

        best_cd = cd_sim
        best_resistance_n = resistance_sim_n
        best_cd_lbm_re_analytical = cd_lbm_analytical
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
                "re_lbm": re_lbm,
                "cd": cd_sim,
                "cd_lbm_analytical": cd_lbm_analytical,
                "resistance_n": resistance_sim_n,
                "drag_lu": fx_lu,
                "lbm": {
                    "u_in": config.lbm_u_in,
                    "tau": config.lbm_tau,
                    "steps": config.lbm_steps,
                },
                "mesh": mesh_stats,
                "runtime_evidence": mesh_stats["runtime_evidence"],
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
            "wetted_area_m2": ref_area,
            "cd_analytical": cd_analytical,
            "resistance_analytical_n": resistance_analytical_n,
            "cd_richardson": richardson_cd if math.isfinite(richardson_cd) else None,
            "re_lbm_analytical_cd": best_cd_lbm_re_analytical if math.isfinite(best_cd_lbm_re_analytical) else None,
        },
        "simulated": {
            "cd": best_cd,
            "resistance_n": best_resistance_n,
        },
        "adaptive_mesh": {
            "enabled": bool(config.use_adaptive_mesh),
            "active_cells_mean": (
                float(sum(float(i["mesh"]["active_cells"]) for i in iterations))
                / max(float(len(iterations)), 1.0)
            ),
            "finest_uniform_cells_mean": (
                float(sum(float(i["mesh"]["finest_uniform_cells"]) for i in iterations))
                / max(float(len(iterations)), 1.0)
            ),
            "cell_saving_pct_mean": (
                float(sum(float(i["mesh"]["cell_saving_pct"]) for i in iterations))
                / max(float(len(iterations)), 1.0)
            ),
        },
        "iterations": iterations,
    }


def run_suboff_resistance_runtime(
    config: SuboffResistanceBenchmarkConfig,
) -> dict[str, object]:
    """Execute the local SUBOFF runner and retain only directly observed facts.

    A normal return or a finite drag is not validation evidence.  Downstream
    artifact/gate code must keep unobserved preflight, numerical, conservation,
    and physics assertions withheld.
    """
    result = run_suboff_resistance_benchmark(config)
    simulated = result.get("simulated")
    iterations = result.get("iterations")
    coefficient = simulated.get("cd") if isinstance(simulated, dict) else None
    final_iteration = iterations[-1] if isinstance(iterations, list) and iterations and isinstance(iterations[-1], dict) else None
    runtime_evidence = final_iteration.get("runtime_evidence") if isinstance(final_iteration, dict) else None
    if not isinstance(runtime_evidence, dict):
        raise RuntimeError("SUBOFF runner returned no runtime numerical evidence")
    if (not isinstance(coefficient, (int, float)) or isinstance(coefficient, bool)
            or not math.isfinite(float(coefficient))):
        raise RuntimeError("SUBOFF runner returned no finite measured resistance coefficient")
    requested_steps = runtime_evidence.get("requested_steps")
    completed_steps = runtime_evidence.get("completed_steps")
    population_checks = runtime_evidence.get("finite_population_checks")
    density_checks = runtime_evidence.get("finite_density_checks")
    density_min = runtime_evidence.get("density_min")
    density_max = runtime_evidence.get("density_max")
    initial_mass = runtime_evidence.get("initial_lattice_mass")
    final_mass = runtime_evidence.get("final_lattice_mass")
    max_abs_mass_drift = runtime_evidence.get("max_abs_mass_drift")
    max_relative_mass_drift = runtime_evidence.get("max_relative_mass_drift")
    initial_momentum = runtime_evidence.get("initial_lattice_momentum_norm")
    final_momentum = runtime_evidence.get("final_lattice_momentum_norm")
    max_abs_momentum_drift = runtime_evidence.get("max_abs_momentum_drift")
    max_relative_momentum_drift = runtime_evidence.get("max_relative_momentum_drift")
    mass_sample_count = runtime_evidence.get("mass_sample_count")
    sampled_step_count = runtime_evidence.get("sampled_step_count")
    grid = final_iteration.get("grid")
    numerical_fields = (requested_steps, completed_steps, population_checks, density_checks, density_min, density_max)
    evidence_is_numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                              and math.isfinite(float(value)) for value in numerical_fields)
    numerics_pass = (evidence_is_numeric and requested_steps == config.lbm_steps
                     and completed_steps == requested_steps and population_checks == completed_steps
                     and density_checks == completed_steps
                     and runtime_evidence.get("all_populations_finite") is True
                     and runtime_evidence.get("all_densities_finite") is True
                     and float(density_min) > 0.0 and float(density_min) <= float(density_max))
    domain_pass = (isinstance(grid, dict) and all(isinstance(grid.get(axis), int) and grid[axis] > 0
                  for axis in ("nx", "ny", "nz")))
    lattice_mach = config.lbm_u_in / math.sqrt(1.0 / 3.0)
    mach_pass = math.isfinite(lattice_mach) and lattice_mach < 0.25
    preflight_checks = {
        "config": {"pass": True, "lbm_tau": config.lbm_tau, "lbm_u_in": config.lbm_u_in},
        "domain": {"pass": domain_pass, "grid": grid},
        "mach": {"pass": mach_pass, "lattice_mach": lattice_mach, "limit": 0.25},
    }
    if not numerics_pass:
        raise RuntimeError("SUBOFF runner returned invalid runtime numerical evidence")
    if not domain_pass or not mach_pass:
        raise RuntimeError("SUBOFF runner failed runtime preflight")
    conservation_fields = (
        initial_mass, final_mass, max_abs_mass_drift, max_relative_mass_drift,
        initial_momentum, final_momentum, max_abs_momentum_drift,
        max_relative_momentum_drift,
    )
    conservation_is_numeric = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in conservation_fields
    )
    conservation_sampled = (
        isinstance(mass_sample_count, int) and not isinstance(mass_sample_count, bool)
        and isinstance(sampled_step_count, int) and not isinstance(sampled_step_count, bool)
        and mass_sample_count == completed_steps + 1 and sampled_step_count == completed_steps
    )
    conservation_pass = (
        conservation_is_numeric and conservation_sampled and float(initial_mass) > 0.0
        and float(final_mass) > 0.0 and float(max_abs_mass_drift) >= 0.0
        and float(max_relative_mass_drift) >= 0.0 and float(max_abs_momentum_drift) >= 0.0
        and float(max_relative_momentum_drift) >= 0.0
        and float(max_relative_mass_drift) <= config.conservation_max_relative_mass_drift
        and float(max_relative_momentum_drift) <= config.conservation_max_relative_momentum_drift
    )
    return {
        "schema": "suboff-resistance-runtime-observation-v1",
        "case": "suboff_runtime",
        "runner": "tensorlbm.suboff_resistance.run_suboff_resistance_benchmark",
        "completion": {
            "state": "COMPLETED",
            "requested_steps": int(requested_steps),
            "completed_steps": int(completed_steps),
            "evidence": "per_step_runtime_observation",
        },
        "resistance": {
            "coefficient": float(coefficient),
            "basis": "runner_simulated_cd",
            "status": "measured",
        },
        "preflight": {"status": "measured", "pass": all(check["pass"] for check in preflight_checks.values()),
                      "checks": preflight_checks},
        "numerics": {"status": "measured", "pass": numerics_pass,
                     "requested_steps": int(requested_steps), "completed_steps": int(completed_steps),
                     "finite_population_checks": int(population_checks), "finite_density_checks": int(density_checks),
                     "all_populations_finite": runtime_evidence["all_populations_finite"],
                     "all_densities_finite": runtime_evidence["all_densities_finite"],
                     "density_min": float(density_min), "density_max": float(density_max)},
        "conservation": {
            "status": "measured",
            "pass": conservation_pass,
            "initial_lattice_mass": float(initial_mass) if conservation_is_numeric else None,
            "final_lattice_mass": float(final_mass) if conservation_is_numeric else None,
            "max_abs_mass_drift": float(max_abs_mass_drift) if conservation_is_numeric else None,
            "max_relative_mass_drift": float(max_relative_mass_drift) if conservation_is_numeric else None,
            "initial_lattice_momentum_norm": float(initial_momentum) if conservation_is_numeric else None,
            "final_lattice_momentum_norm": float(final_momentum) if conservation_is_numeric else None,
            "max_abs_momentum_drift": float(max_abs_momentum_drift) if conservation_is_numeric else None,
            "max_relative_momentum_drift": float(max_relative_momentum_drift) if conservation_is_numeric else None,
            "mass_sample_count": mass_sample_count,
            "sampled_step_count": sampled_step_count,
            "max_relative_mass_drift_limit": config.conservation_max_relative_mass_drift,
            "max_relative_momentum_drift_limit": config.conservation_max_relative_momentum_drift,
        },
        "physics": {"status": "withheld", "pass": False,
                    "reason": "no_independent_physics_validation"},
    }
