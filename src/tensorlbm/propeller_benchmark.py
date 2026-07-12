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
from .d3q19 import W, equilibrium3d
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
    warmup_steps: int = 200
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
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not self.inflow_velocities:
            raise ValueError("inflow_velocities must not be empty")

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * self.rpm

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

    mask = build_propeller_mask(
        nx=nx, ny=ny, nz=nz, cx=cx, cy=cy, cz=cz,
        angle_deg=0.0, config=geo, device=str(device),
    )
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, mask, device=device)
    ux_w, uy_w, uz_w = rotating_wall_velocity_3d(mask, cx, cy, cz, config.omega)

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux0 = torch.full_like(rho0, u_in)
    ux0[mask] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)

    steps_per_rev = max(1, int(1.0 / max(config.rpm, 1e-10)))
    n_sampling = config.n_revolutions * steps_per_rev
    n_total = config.warmup_steps + n_sampling

    fx_samples: list[float] = []
    mx_samples: list[float] = []
    me_samples: list[dict[str, float | int]] = []
    t_start = time.perf_counter()

    for step in range(1, n_total + 1):
        f = collide_smagorinsky_mrt3d(f, tau=config.tau, C_s=config.smagorinsky_cs)
        f = stream3d(f)
        fx, _, _ = compute_obstacle_forces_3d(f, mask)
        mx, _, _ = compute_obstacle_moments_3d(f, mask, cx, cy, cz)
        f = apply_zou_he_channel_boundaries_3d(
            f, u_in=u_in, wall_mask=wall_mask,
            obstacle_mask=torch.zeros_like(mask),
        )
        f = moving_wall_bounce_back_3d(f, mask, ux_w, uy_w, uz_w)

        if step > config.warmup_steps:
            fx_value = float(fx.item())
            mx_value = float(mx.item())
            fx_samples.append(fx_value)
            mx_samples.append(mx_value)
            me_samples.append({
                "step": step,
                "fx_me_lu": fx_value,
                "mx_me_lu": mx_value,
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

    geo_stats = propeller_statistics(geo, mask)
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
        "me_samples": me_samples,
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

    tip_ma = config.omega * config.geometry.radius / 0.577
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
            "n_revolutions": config.n_revolutions,
            "model_diameter_m": config.model_diameter_m,
            "model_speed_ms": config.model_speed_ms,
        },
        "results": results,
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
    "rotating_wall_velocity_3d",
    "moving_wall_bounce_back_3d",
    "run_propeller_benchmark",
]
