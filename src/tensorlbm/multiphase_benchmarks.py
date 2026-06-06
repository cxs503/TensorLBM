"""Multiphase LBM model benchmark suite for D2Q9 and D3Q19.

Provides four canonical benchmarks that cover all four multiphase models:

1. **Static Droplet / Laplace Pressure Test** (:class:`StaticDropletConfig` /
   :func:`run_static_droplet`):
   A circular droplet of one phase sits in a periodic domain filled with a
   second phase.  At steady state the pressure jump across the interface
   satisfies the Young-Laplace equation ``ΔP = σ/R``.  Running over multiple
   radii allows a linear fit ``ΔP = σ/R + const`` to extract the effective
   surface tension ``σ_eff`` for the SC two-component (SCMC) and
   Color-Gradient (CG) models.  In addition, the maximum spurious current
   (velocity) inside the static droplet is recorded as a quality metric.

2. **Spinodal Decomposition** (:class:`SpinodaleConfig` /
   :func:`run_spinodal_decomposition`):
   A uniform density field with small random noise spontaneously separates into
   liquid and gas regions under the Shan-Chen single-component (SCMP)
   attractive pseudopotential.  The benchmark tracks the evolving density
   extrema and reports the steady-state coexistence densities
   ``(ρ_liquid, ρ_gas)`` and density ratio.

3. **Free-Energy Droplet Relaxation** (:class:`FreeEnergyDropletConfig` /
   :func:`run_free_energy_droplet`):
   A phase-field droplet is initialized in a periodic domain and evolved with
   the free-energy binary-fluid model.  The benchmark tracks conserved
   order-parameter mass drift, equivalent droplet-radius drift, phase-field
   boundedness, and the maximum spurious current during relaxation.

4. **Two-Phase Poiseuille Comparison** (:class:`TwoPhaseChannelCompareConfig` /
   :func:`run_two_phase_channel_compare`):
   Two immiscible fluids fill the lower (phase 1) and upper (phase 2) halves of
   a 2-D channel driven by a body force.  The benchmark runs both SCMC (with
   distinct relaxation times for each component, allowing viscosity contrast)
   and CG (with a shared relaxation time), and compares the steady-state
   velocity profiles against the analytical piecewise-parabolic solution.

Suite runner
------------
:func:`run_multiphase_benchmark_suite` executes all four benchmarks in
sequence and returns a structured comparison report including summary
statistics and a brief analysis.

References
----------
Young (1805) Phil. Trans. R. Soc. 95 65
Laplace (1806) Mécanique Céleste, Supplément
Shan & Chen (1993) Phys. Rev. E 47 1815
Latva-Kokko & Rothman (2005) Phys. Rev. E 71 056702
Pan, Hilpert & Miller (2004) Phys. Rev. E 70 026702
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib
import torch

from .boundaries import bounce_back_cells
from .d2q9 import equilibrium, macroscopic
from .d3q19 import equilibrium3d, macroscopic3d
from .multiphase import (
    collide_sc_single_component,
    collide_sc_two_component,
    color_gradient_step,
    free_energy_step,
    init_free_energy_g,
    psi_exp,
)
from .multiphase3d import (
    collide_sc_single_component_3d,
    collide_sc_two_component_3d,
    color_gradient_step_3d,
)
from .solver import stream
from .solver3d import stream3d
from .utils import prepare_run_dir, resolve_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CS2 = 1.0 / 3.0

MultiphaseModel = Literal["scmc", "cg"]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _circular_mask(ny: int, nx: int, r: float, device: torch.device) -> torch.Tensor:
    """Boolean mask: *True* inside a centred circle of radius *r*."""
    ys = torch.arange(ny, dtype=torch.float32, device=device)
    xs = torch.arange(nx, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cy, cx = ny / 2.0, nx / 2.0
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2


def _spherical_mask(
    nz: int, ny: int, nx: int, r: float, device: torch.device,
) -> torch.Tensor:
    """Boolean mask: *True* inside a centred sphere of radius *r*."""
    zs = torch.arange(nz, dtype=torch.float32, device=device)
    ys = torch.arange(ny, dtype=torch.float32, device=device)
    xs = torch.arange(nx, dtype=torch.float32, device=device)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    cz, cy, cx = nz / 2.0, ny / 2.0, nx / 2.0
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= r ** 2


def _measure_pressure_jump(
    f_total: torch.Tensor,
    r: float,
) -> tuple[float, float, float]:
    """Return ``(p_inside, p_outside, delta_p)`` from the total density.

    The lattice pressure is ``p = cs² ρ = ρ / 3``.
    Inside and outside averages exclude a transition band of 0.5 R around
    the nominal radius to avoid contamination by the diffuse interface.
    """
    rho = f_total.sum(dim=0)  # (ny, nx)
    ny, nx = rho.shape
    ys = torch.arange(ny, dtype=torch.float32, device=rho.device)
    xs = torch.arange(nx, dtype=torch.float32, device=rho.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cy, cx = ny / 2.0, nx / 2.0
    r_field = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inside = r_field <= r * 0.5
    outside = r_field >= r * 1.5
    p_in = float((_CS2 * rho[inside]).mean().item()) if inside.any() else float("nan")
    p_out = float((_CS2 * rho[outside]).mean().item()) if outside.any() else float("nan")
    dp = p_in - p_out
    return p_in, p_out, dp


def _measure_pressure_jump_3d(
    f_total: torch.Tensor,
    r: float,
) -> tuple[float, float, float]:
    """Return ``(p_inside, p_outside, delta_p)`` from total density in 3-D."""
    rho = f_total.sum(dim=0)  # (nz, ny, nx)
    nz, ny, nx = rho.shape
    zs = torch.arange(nz, dtype=torch.float32, device=rho.device)
    ys = torch.arange(ny, dtype=torch.float32, device=rho.device)
    xs = torch.arange(nx, dtype=torch.float32, device=rho.device)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    cz, cy, cx = nz / 2.0, ny / 2.0, nx / 2.0
    r_field = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    inside = r_field <= r * 0.5
    outside = r_field >= r * 1.5
    p_in = float((_CS2 * rho[inside]).mean().item()) if inside.any() else float("nan")
    p_out = float((_CS2 * rho[outside]).mean().item()) if outside.any() else float("nan")
    dp = p_in - p_out
    return p_in, p_out, dp


def _max_velocity(f_total: torch.Tensor) -> float:
    """Return the maximum fluid velocity ``|u|`` for any lattice node."""
    rho = f_total.sum(dim=0).clamp(min=1e-12)  # (ny, nx)
    c_dev = f_total.new_zeros((9, 2))
    from .d2q9 import C  # noqa: PLC0415
    c_dev = C.to(f_total.device).float()
    cx = c_dev[:, 0].view(9, 1, 1)
    cy = c_dev[:, 1].view(9, 1, 1)
    ux = (f_total * cx).sum(0) / rho
    uy = (f_total * cy).sum(0) / rho
    return float(torch.sqrt(ux ** 2 + uy ** 2).max().item())


def _max_velocity_3d(f_total: torch.Tensor) -> float:
    """Return the maximum fluid velocity ``|u|`` for any 3-D lattice node."""
    rho = f_total.sum(dim=0).clamp(min=1e-12)  # (nz, ny, nx)
    from .d3q19 import C as C3  # noqa: PLC0415
    c_dev = C3.to(f_total.device).float()
    cx = c_dev[:, 0].view(19, 1, 1, 1)
    cy = c_dev[:, 1].view(19, 1, 1, 1)
    cz = c_dev[:, 2].view(19, 1, 1, 1)
    ux = (f_total * cx).sum(0) / rho
    uy = (f_total * cy).sum(0) / rho
    uz = (f_total * cz).sum(0) / rho
    return float(torch.sqrt(ux ** 2 + uy ** 2 + uz ** 2).max().item())


# ---------------------------------------------------------------------------
# Benchmark 1 – Static Droplet: Laplace pressure + spurious currents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticDropletConfig:
    """Configuration for the static-droplet (Laplace + spurious currents) benchmark.

    A circular droplet of *heavy* phase with radius ``R`` is placed at the
    centre of a periodic domain filled with the *light* phase.  The simulation
    is run until steady state and both the Young-Laplace pressure jump
    ``ΔP = σ / R`` and the maximum spurious velocity are measured.

    Multiple radii can be supplied via *radii*; the benchmark fits a linear
    relationship ``ΔP = σ / R`` to extract the effective surface tension
    ``σ_eff`` (slope of the ΔP vs 1/R curve).

    The benchmark runs the **SCMC** model (Shan-Chen two-component, two
    separate distribution functions) and the **CG** model (Color-Gradient,
    also two distribution functions but with a shared relaxation time).

    Attributes
    ----------
    nx, ny:        Domain size (square domain recommended).
    radii:         Tuple of bubble radii (lattice units) to test.
    n_steps:       Time steps per radius to reach steady state.
    output_interval: Diagnostic sampling interval.
    scmc_G12:      SC coupling constant (> 0 for phase separation).
    scmc_tau:      Relaxation time for both SCMC components.
    scmc_rho_heavy: Density of the heavy (droplet) component.
    scmc_rho_light: Density of the light (surrounding) component.
    cg_A:          CG surface-tension amplitude (larger → stronger tension).
    cg_beta:       CG recoloring parameter ∈ (0, 1].
    cg_tau:        CG relaxation time (shared for both components).
    cg_rho_heavy:  Initial heavy-phase density inside the droplet.
    cg_rho_light:  Initial light-phase density inside the droplet
                   (and vice-versa outside).
    output_root:   Root directory for output files.
    run_name:      Optional run identifier; auto-generated if ``None``.
    device:        PyTorch device string.
    overwrite:     Overwrite existing run directory if *True*.
    """

    nx: int = 100
    ny: int = 100
    radii: tuple[float, ...] = (10.0, 15.0, 20.0)
    n_steps: int = 4000
    output_interval: int = 1000
    # SCMC parameters
    scmc_G12: float = 0.9
    scmc_tau: float = 1.0
    scmc_rho_heavy: float = 0.7
    scmc_rho_light: float = 0.3
    # CG parameters
    cg_A: float = 0.04
    cg_beta: float = 0.7
    cg_tau: float = 1.0
    cg_rho_heavy: float = 0.65
    cg_rho_light: float = 0.05
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        """Raise ValueError if configuration is invalid."""
        if self.nx < 30 or self.ny < 30:
            msg = "nx and ny must be ≥ 30"
            raise ValueError(msg)
        if not self.radii:
            msg = "radii must be non-empty"
            raise ValueError(msg)
        if any(r <= 0 or r >= min(self.nx, self.ny) / 2 - 5 for r in self.radii):
            msg = "each radius must be positive and well inside the domain"
            raise ValueError(msg)
        if self.scmc_tau <= 0.5 or self.cg_tau <= 0.5:
            msg = "tau must be > 0.5"
            raise ValueError(msg)
        if self.scmc_rho_heavy <= self.scmc_rho_light:
            msg = "scmc_rho_heavy must exceed scmc_rho_light"
            raise ValueError(msg)
        if self.cg_rho_heavy <= self.cg_rho_light:
            msg = "cg_rho_heavy must exceed cg_rho_light"
            raise ValueError(msg)
        if self.scmc_G12 <= 0:
            msg = "scmc_G12 must be > 0 for SC phase separation"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return f"static_droplet_nx{self.nx}_steps{self.n_steps}"


def _run_scmc_droplet(
    r: float,
    config: StaticDropletConfig,
    device: torch.device,
) -> dict[str, object]:
    """Run one SCMC static-droplet simulation and return metrics."""
    ny, nx = config.ny, config.nx
    inside = _circular_mask(ny, nx, r, device)
    zero = torch.zeros((ny, nx), device=device)
    rho1 = torch.where(
        inside, torch.full_like(zero, config.scmc_rho_heavy), zero + config.scmc_rho_light
    )
    rho2 = torch.where(
        inside, torch.full_like(zero, config.scmc_rho_light), zero + config.scmc_rho_heavy
    )
    f1 = equilibrium(rho1, zero, zero)
    f2 = equilibrium(rho2, zero, zero)

    for _ in range(config.n_steps):
        f1, f2 = collide_sc_two_component(
            f1, f2, G_12=config.scmc_G12,
            tau1=config.scmc_tau, tau2=config.scmc_tau,
        )
        f1 = stream(f1)
        f2 = stream(f2)

    f_total = f1 + f2
    p_in, p_out, dp = _measure_pressure_jump(f_total, r)
    max_u = _max_velocity(f_total)
    sigma_eff = dp * r

    return {
        "r": r,
        "p_inside": round(p_in, 8),
        "p_outside": round(p_out, 8),
        "delta_p": round(dp, 8),
        "sigma_eff": round(sigma_eff, 6),
        "max_spurious_u": round(max_u, 8),
    }


def _run_cg_droplet(
    r: float,
    config: StaticDropletConfig,
    device: torch.device,
) -> dict[str, object]:
    """Run one CG static-droplet simulation and return metrics."""
    ny, nx = config.ny, config.nx
    inside = _circular_mask(ny, nx, r, device)
    zero = torch.zeros((ny, nx), device=device)
    # Red = heavy phase inside, blue = light phase inside (and vice-versa outside)
    rho_r = torch.where(
        inside, torch.full_like(zero, config.cg_rho_heavy), zero + config.cg_rho_light
    )
    rho_b = torch.where(
        inside, torch.full_like(zero, config.cg_rho_light), zero + config.cg_rho_heavy
    )
    f_r = equilibrium(rho_r, zero, zero)
    f_b = equilibrium(rho_b, zero, zero)

    for _ in range(config.n_steps):
        f_r, f_b = color_gradient_step(
            f_r, f_b, tau=config.cg_tau, A=config.cg_A, beta=config.cg_beta,
        )
        f_r = stream(f_r)
        f_b = stream(f_b)

    f_total = f_r + f_b
    p_in, p_out, dp = _measure_pressure_jump(f_total, r)
    max_u = _max_velocity(f_total)
    sigma_eff = dp * r

    return {
        "r": r,
        "p_inside": round(p_in, 8),
        "p_outside": round(p_out, 8),
        "delta_p": round(dp, 8),
        "sigma_eff": round(sigma_eff, 6),
        "max_spurious_u": round(max_u, 8),
    }


def _fit_surface_tension(radii: list[float], delta_ps: list[float]) -> float:
    """Fit σ_eff = slope of ΔP vs 1/R via least-squares through the origin."""
    if len(radii) < 2:
        return delta_ps[0] * radii[0] if radii else float("nan")
    inv_r = [1.0 / r for r in radii]
    num = sum(x * y for x, y in zip(inv_r, delta_ps, strict=True))
    den = sum(x * x for x in inv_r)
    return num / den if abs(den) > 1e-20 else float("nan")


def run_static_droplet(config: StaticDropletConfig) -> dict[str, object]:
    """Run the static-droplet benchmark for SCMC and CG models.

    For each radius in *config.radii* and each model (SCMC, CG), a circular
    droplet of heavy phase is placed in a periodic domain and run to steady
    state.  The Young-Laplace pressure jump ``ΔP = σ / R`` is measured from
    the total lattice pressure, and the effective surface tension ``σ_eff`` is
    extracted by a linear fit over all radii.

    In addition, the maximum spurious velocity in the static droplet is
    recorded as a quality metric (smaller is better; physically it should be
    identically zero for a perfectly isotropic lattice).

    Args:
        config: Benchmark configuration.

    Returns:
        Dictionary with keys ``scmc`` and ``cg``, each containing:
        * ``per_radius``: list of per-radius result dicts.
        * ``sigma_eff_fit``: effective surface tension from linear fit (ΔP·R).
        * ``mean_max_spurious_u``: mean spurious current over all radii.
    """
    config.validate()
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "static_droplet", config.resolved_run_name(), config.overwrite
    )

    results: dict[str, object] = {}

    for model_name in ("scmc", "cg"):
        print(f"\n--- Static Droplet: {model_name.upper()} ---")
        per_r: list[dict[str, object]] = []
        for r in config.radii:
            print(f"  R={r:.0f}  steps={config.n_steps}")
            if model_name == "scmc":
                row = _run_scmc_droplet(r, config, device)
            else:
                row = _run_cg_droplet(r, config, device)
            print(
                f"    ΔP={row['delta_p']:.6f}  σ_eff={row['sigma_eff']:.4f}"
                f"  max_u={row['max_spurious_u']:.3e}"
            )
            per_r.append(row)

        sigma_fit = _fit_surface_tension(
            [float(d["r"]) for d in per_r],  # type: ignore[arg-type]
            [float(d["delta_p"]) for d in per_r],  # type: ignore[arg-type]
        )
        mean_u = sum(float(d["max_spurious_u"]) for d in per_r) / len(per_r)  # type: ignore[arg-type, misc]
        print(f"  σ_eff (fit) = {sigma_fit:.6f}   mean max_u = {mean_u:.3e}")
        results[model_name] = {
            "per_radius": per_r,
            "sigma_eff_fit": round(sigma_fit, 8),
            "mean_max_spurious_u": round(mean_u, 8),
        }

    # Save results
    metadata: dict[str, object] = {
        "benchmark": "static_droplet",
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "results": results,
        "note": (
            "delta_p = P_inside - P_outside (lattice pressure cs2*rho). "
            "sigma_eff_fit: slope of ΔP vs 1/R (Young-Laplace: ΔP = σ/R). "
            "max_spurious_u: max velocity in static droplet (quality metric)."
        ),
    }
    (run_dir / "static_droplet.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nSaved → {run_dir / 'static_droplet.json'}")

    # Plot ΔP vs 1/R for both models
    _plot_laplace(results, run_dir)

    return metadata


def _plot_laplace(results: dict[str, object], run_dir: Path) -> None:
    """Save a ΔP-vs-1/R plot for SCMC and CG."""
    fig, ax = plt.subplots()
    colours = {"scmc": "C0", "cg": "C1"}
    for model in ("scmc", "cg"):
        if model not in results:
            continue
        per_r: list[dict[str, object]] = results[model]["per_radius"]  # type: ignore[index]
        inv_r = [1.0 / float(d["r"]) for d in per_r]  # type: ignore[arg-type]
        dp = [float(d["delta_p"]) for d in per_r]  # type: ignore[arg-type]
        sigma = float(results[model]["sigma_eff_fit"])  # type: ignore[index]
        inv_r_line = [min(inv_r) * 0.8, max(inv_r) * 1.2]
        ax.scatter(inv_r, dp, color=colours[model], label=model.upper(), zorder=3)
        ax.plot(inv_r_line, [sigma * x for x in inv_r_line], "--",
                color=colours[model], alpha=0.7)

    ax.set_xlabel("1/R (lattice units⁻¹)")
    ax.set_ylabel("ΔP (lattice units)")
    ax.set_title("Laplace pressure test: ΔP vs 1/R")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(run_dir / "laplace_pressure.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Benchmark 2 – Spinodal Decomposition (SCMP)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpinodaleConfig:
    """Configuration for the spinodal decomposition (SCMP) benchmark.

    The Shan-Chen single-component model with a self-attractive pseudopotential
    (G < 0) is initialised from a uniform density field with small random
    noise.  Under the attractive SC interaction the field spontaneously
    separates into a *liquid* phase (high density) and a *gas* phase (low
    density).

    The benchmark tracks the evolving density extrema and reports the
    steady-state coexistence densities and their ratio, which can be compared
    with the Maxwell equal-area construction for the SC equation of state.

    Attributes
    ----------
    nx, ny:          Domain size (periodic, no walls).
    G:               SC self-coupling constant (< 0 for phase separation).
    tau:             Relaxation time (ν = cs²(τ − ½)).
    rho0:            Mean initial density.
    noise_amp:       Amplitude of the initial uniform-random density noise.
    n_steps:         Total simulation steps.
    output_interval: Diagnostic sampling interval.
    seed:            Random seed for the initial noise.
    output_root:     Root directory for output files.
    run_name:        Optional run identifier.
    device:          PyTorch device string.
    overwrite:       Overwrite existing run directory if *True*.
    """

    nx: int = 64
    ny: int = 64
    G: float = -4.0
    tau: float = 1.0
    rho0: float = 0.7
    noise_amp: float = 0.05
    n_steps: int = 3000
    output_interval: int = 500
    seed: int = 42
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        """Raise ValueError if configuration is invalid."""
        if self.nx < 10 or self.ny < 10:
            msg = "nx and ny must be ≥ 10"
            raise ValueError(msg)
        if self.G >= 0:
            msg = "G must be < 0 for SCMP phase separation (attractive interaction)"
            raise ValueError(msg)
        if self.tau <= 0.5:
            msg = "tau must be > 0.5"
            raise ValueError(msg)
        if self.rho0 <= 0:
            msg = "rho0 must be positive"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return f"spinodal_G{self.G:.1f}_rho{self.rho0:.2f}_nx{self.nx}_steps{self.n_steps}"


def run_spinodal_decomposition(config: SpinodaleConfig) -> dict[str, object]:
    """Run the spinodal decomposition (SCMP) benchmark.

    Initialises a uniform density field with small random noise and evolves
    it under the Shan-Chen single-component model.  The attractive
    self-interaction drives spontaneous phase separation.  The benchmark
    tracks the maximum and minimum densities over time and reports the
    final coexistence densities.

    The effective density ratio ``ρ_liquid / ρ_gas`` should exceed 1.0
    (phase separation achieved) and can be compared with the Maxwell
    equal-area construction for the SC equation of state.

    Args:
        config: Benchmark configuration.

    Returns:
        Dictionary with ``rho_liquid``, ``rho_gas``, ``density_ratio``,
        ``diagnostics``, and ``phase_separated`` (bool).
    """
    config.validate()
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "spinodal", config.resolved_run_name(), config.overwrite
    )

    ny, nx = config.ny, config.nx
    torch.manual_seed(config.seed)
    rho0 = torch.full((ny, nx), config.rho0, device=device)
    noise = (torch.rand((ny, nx), device=device) - 0.5) * 2.0 * config.noise_amp
    rho = (rho0 + noise).clamp(min=0.01)

    zero = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho, zero, zero)

    print(
        f"Spinodal decomposition  NX={nx}  NY={ny}  G={config.G}  "
        f"τ={config.tau}  ρ₀={config.rho0}  steps={config.n_steps}"
    )

    diagnostics: list[dict[str, object]] = []

    for step in range(1, config.n_steps + 1):
        f = collide_sc_single_component(f, G=config.G, tau=config.tau, psi_fn=psi_exp)
        f = stream(f)

        if step % config.output_interval == 0 or step == config.n_steps:
            rho_cur, _, _ = macroscopic(f)
            rho_max = float(rho_cur.max().item())
            rho_min = float(rho_cur.min().item())
            rho_std = float(rho_cur.std().item())
            diag: dict[str, object] = {
                "step": step,
                "rho_max": round(rho_max, 6),
                "rho_min": round(rho_min, 6),
                "rho_std": round(rho_std, 6),
                "density_ratio": round(rho_max / max(rho_min, 1e-12), 4),
            }
            diagnostics.append(diag)
            print(
                f"  step={step:5d}  ρ_max={rho_max:.4f}  ρ_min={rho_min:.4f}"
                f"  ρ_std={rho_std:.4f}  ratio={rho_max / max(rho_min, 1e-12):.2f}"
            )

    # Final state
    rho_final, _, _ = macroscopic(f)
    rho_liquid = float(rho_final.max().item())
    rho_gas = float(rho_final.min().item())
    density_ratio = rho_liquid / max(rho_gas, 1e-12)
    phase_separated = density_ratio > 2.0  # heuristic: ratio > 2 → phase separation

    metadata: dict[str, object] = {
        "benchmark": "spinodal_decomposition",
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "rho_liquid": round(rho_liquid, 6),
        "rho_gas": round(rho_gas, 6),
        "density_ratio": round(density_ratio, 4),
        "phase_separated": phase_separated,
        "diagnostics": diagnostics,
        "note": (
            "SCMP (Shan-Chen single-component) spinodal decomposition. "
            "rho_liquid / rho_gas → Maxwell construction coexistence densities at steady state."
        ),
    }
    (run_dir / "spinodal.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"\nSaved → {run_dir / 'spinodal.json'}"
        f"  (ρ_l={rho_liquid:.4f}, ρ_g={rho_gas:.4f}, ratio={density_ratio:.2f})"
    )

    # Plot density evolution
    _plot_spinodal(diagnostics, run_dir)

    return metadata


def _plot_spinodal(
    diagnostics: list[dict[str, object]],
    run_dir: Path,
) -> None:
    """Save a density-extrema-vs-time plot for the spinodal benchmark."""
    steps = [int(d["step"]) for d in diagnostics]  # type: ignore[call-overload]
    rho_max = [float(d["rho_max"]) for d in diagnostics]  # type: ignore[arg-type]
    rho_min = [float(d["rho_min"]) for d in diagnostics]  # type: ignore[arg-type]

    fig, ax = plt.subplots()
    ax.plot(steps, rho_max, label="ρ_liquid (max)")
    ax.plot(steps, rho_min, label="ρ_gas (min)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Density (lattice units)")
    ax.set_title("SCMP spinodal decomposition: coexistence densities")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(run_dir / "spinodal_densities.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Benchmark 3 – Free-Energy droplet relaxation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeEnergyDropletConfig:
    """Configuration for the free-energy phase-field droplet benchmark."""

    nx: int = 60
    ny: int = 60
    radius: float = 12.0
    interface_width: float = 2.0
    n_steps: int = 1500
    output_interval: int = 300
    tau_f: float = 1.0
    tau_g: float = 0.8
    A: float = 0.04
    B: float = 0.04
    kappa: float = 0.03
    Gamma: float = 0.5
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        if self.nx < 24 or self.ny < 24:
            msg = "nx and ny must be ≥ 24"
            raise ValueError(msg)
        if self.radius <= 0 or self.radius >= min(self.nx, self.ny) / 2 - 4:
            msg = "radius must be positive and well inside the domain"
            raise ValueError(msg)
        if self.interface_width <= 0:
            msg = "interface_width must be > 0"
            raise ValueError(msg)
        if self.n_steps < 1:
            msg = "n_steps must be ≥ 1"
            raise ValueError(msg)
        if self.output_interval < 1:
            msg = "output_interval must be ≥ 1"
            raise ValueError(msg)
        if self.tau_f <= 0.5 or self.tau_g <= 0.5:
            msg = "tau_f and tau_g must be > 0.5"
            raise ValueError(msg)
        if self.B <= 0 or self.kappa <= 0 or self.Gamma <= 0:
            msg = "B, kappa and Gamma must be > 0"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return f"free_energy_droplet_n{self.nx}x{self.ny}_steps{self.n_steps}"


def _equivalent_radius_from_phi(phi: torch.Tensor) -> float:
    """Return the equivalent circular radius inferred from the phase field."""
    alpha = ((phi.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
    area = float(alpha.sum().item())
    return math.sqrt(area / math.pi)


def run_free_energy_droplet(config: FreeEnergyDropletConfig) -> dict[str, object]:
    """Run the free-energy droplet relaxation benchmark."""
    config.validate()
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root,
        "free_energy_droplet",
        config.resolved_run_name(),
        config.overwrite,
    )

    ny, nx = config.ny, config.nx
    ys = torch.arange(ny, dtype=torch.float32, device=device)
    xs = torch.arange(nx, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cy, cx = ny / 2.0, nx / 2.0
    distance = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    phi0 = torch.tanh((config.radius - distance) / config.interface_width)
    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=device)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    g = init_free_energy_g(phi0, zero, zero)

    initial_phase_mass = float(phi0.sum().item())
    initial_equiv_radius = _equivalent_radius_from_phi(phi0)

    diagnostics: list[dict[str, object]] = []
    for step in range(1, config.n_steps + 1):
        f, g = free_energy_step(
            f,
            g,
            tau_f=config.tau_f,
            tau_g=config.tau_g,
            A=config.A,
            B=config.B,
            kappa=config.kappa,
            Gamma=config.Gamma,
        )
        f = stream(f)
        g = stream(g)

        if step % config.output_interval == 0 or step == config.n_steps:
            phi = g.sum(dim=0)
            _rho, ux, uy = macroscopic(f)
            phase_mass = float(phi.sum().item())
            equiv_radius = _equivalent_radius_from_phi(phi)
            rel_mass_drift = abs(phase_mass - initial_phase_mass) / max(
                abs(initial_phase_mass), 1e-12,
            )
            rel_radius_drift = abs(equiv_radius - initial_equiv_radius) / max(
                initial_equiv_radius, 1e-12,
            )
            max_velocity = float(torch.sqrt(ux ** 2 + uy ** 2).max().item())
            phi_min = float(phi.min().item())
            phi_max = float(phi.max().item())
            max_overshoot = max(phi_max - 1.0, -1.0 - phi_min, 0.0)
            diag = {
                "step": step,
                "phase_mass": round(phase_mass, 6),
                "relative_mass_drift": round(rel_mass_drift, 8),
                "equivalent_radius": round(equiv_radius, 6),
                "relative_radius_drift": round(rel_radius_drift, 8),
                "max_velocity": round(max_velocity, 8),
                "phi_min": round(phi_min, 6),
                "phi_max": round(phi_max, 6),
                "max_phase_overshoot": round(max_overshoot, 8),
            }
            diagnostics.append(diag)
            print(
                f"  step={step:5d}  mass_drift={rel_mass_drift:.2e}"
                f"  radius_drift={rel_radius_drift:.2e}  max_u={max_velocity:.2e}"
                f"  overshoot={max_overshoot:.2e}"
            )

    final_phi = g.sum(dim=0)
    _rho, ux, uy = macroscopic(f)
    final_phase_mass = float(final_phi.sum().item())
    final_equiv_radius = _equivalent_radius_from_phi(final_phi)
    relative_phase_mass_drift = abs(final_phase_mass - initial_phase_mass) / max(
        abs(initial_phase_mass), 1e-12,
    )
    relative_radius_drift = abs(final_equiv_radius - initial_equiv_radius) / max(
        initial_equiv_radius, 1e-12,
    )
    max_spurious_u = float(torch.sqrt(ux ** 2 + uy ** 2).max().item())
    final_phi_min = float(final_phi.min().item())
    final_phi_max = float(final_phi.max().item())
    max_phase_overshoot = max(final_phi_max - 1.0, -1.0 - final_phi_min, 0.0)

    metadata: dict[str, object] = {
        "benchmark": "free_energy_droplet",
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "initial_phase_mass": round(initial_phase_mass, 6),
        "final_phase_mass": round(final_phase_mass, 6),
        "relative_phase_mass_drift": round(relative_phase_mass_drift, 8),
        "initial_equivalent_radius": round(initial_equiv_radius, 6),
        "final_equivalent_radius": round(final_equiv_radius, 6),
        "relative_radius_drift": round(relative_radius_drift, 8),
        "max_spurious_u": round(max_spurious_u, 8),
        "phi_min": round(final_phi_min, 6),
        "phi_max": round(final_phi_max, 6),
        "max_phase_overshoot": round(max_phase_overshoot, 8),
        "bounded_phase_field": max_phase_overshoot <= 5e-2,
        "diagnostics": diagnostics,
        "note": (
            "Free-energy / phase-field droplet relaxation benchmark. "
            "Tracks conserved order-parameter drift, equivalent-radius drift, "
            "phase-field boundedness, and spurious currents."
        ),
    }
    (run_dir / "free_energy_droplet.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"\nSaved → {run_dir / 'free_energy_droplet.json'}"
        f"  (mass_drift={relative_phase_mass_drift:.2e},"
        f" radius_drift={relative_radius_drift:.2e}, max_u={max_spurious_u:.2e})"
    )

    _plot_free_energy_droplet(diagnostics, run_dir)
    return metadata


def _plot_free_energy_droplet(
    diagnostics: list[dict[str, object]],
    run_dir: Path,
) -> None:
    """Save phase-field conservation diagnostics for the FE droplet benchmark."""
    steps = [int(d["step"]) for d in diagnostics]  # type: ignore[call-overload]
    mass_drift = [float(d["relative_mass_drift"]) for d in diagnostics]  # type: ignore[arg-type]
    radius_drift = [float(d["relative_radius_drift"]) for d in diagnostics]  # type: ignore[arg-type]
    max_velocity = [float(d["max_velocity"]) for d in diagnostics]  # type: ignore[arg-type]

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    axes[0].plot(steps, mass_drift, marker="o")
    axes[0].set_ylabel("Rel. mass drift")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, radius_drift, marker="o", color="tab:orange")
    axes[1].set_ylabel("Rel. radius drift")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, max_velocity, marker="o", color="tab:green")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Max |u|")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Free-energy droplet relaxation diagnostics")
    fig.savefig(run_dir / "free_energy_diagnostics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Benchmark 4 – 3D multiphase extensions (D3Q19)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticDroplet3DConfig:
    """Configuration for 3D static-droplet (Laplace) benchmark."""

    nx: int = 40
    ny: int = 40
    nz: int = 40
    radii: tuple[float, ...] = (8.0, 12.0)
    n_steps: int = 1200
    output_interval: int = 400
    # SCMC
    scmc_G12: float = 0.9
    scmc_tau: float = 1.0
    scmc_rho_heavy: float = 0.7
    scmc_rho_light: float = 0.3
    # CG
    cg_A: float = 0.04
    cg_beta: float = 0.7
    cg_tau: float = 1.0
    cg_rho_heavy: float = 0.65
    cg_rho_light: float = 0.05
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        if self.nx < 20 or self.ny < 20 or self.nz < 20:
            msg = "nx, ny and nz must be ≥ 20"
            raise ValueError(msg)
        if not self.radii:
            msg = "radii must be non-empty"
            raise ValueError(msg)
        limit = min(self.nx, self.ny, self.nz) / 2 - 4
        if any(r <= 0 or r >= limit for r in self.radii):
            msg = "each radius must be positive and well inside the 3D domain"
            raise ValueError(msg)
        if self.scmc_tau <= 0.5 or self.cg_tau <= 0.5:
            msg = "tau must be > 0.5"
            raise ValueError(msg)
        if self.scmc_rho_heavy <= self.scmc_rho_light:
            msg = "scmc_rho_heavy must exceed scmc_rho_light"
            raise ValueError(msg)
        if self.cg_rho_heavy <= self.cg_rho_light:
            msg = "cg_rho_heavy must exceed cg_rho_light"
            raise ValueError(msg)
        if self.scmc_G12 <= 0:
            msg = "scmc_G12 must be > 0 for SC phase separation"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (
            f"static_droplet_3d_n{self.nx}x{self.ny}x{self.nz}"
            f"_steps{self.n_steps}"
        )


def _run_scmc_droplet_3d(
    r: float,
    config: StaticDroplet3DConfig,
    device: torch.device,
) -> dict[str, object]:
    nz, ny, nx = config.nz, config.ny, config.nx
    inside = _spherical_mask(nz, ny, nx, r, device)
    zero = torch.zeros((nz, ny, nx), device=device)
    rho1 = torch.where(
        inside, torch.full_like(zero, config.scmc_rho_heavy), zero + config.scmc_rho_light
    )
    rho2 = torch.where(
        inside, torch.full_like(zero, config.scmc_rho_light), zero + config.scmc_rho_heavy
    )
    f1 = equilibrium3d(rho1, zero, zero, zero)
    f2 = equilibrium3d(rho2, zero, zero, zero)

    for _ in range(config.n_steps):
        f1, f2 = collide_sc_two_component_3d(
            f1, f2, G_12=config.scmc_G12, tau1=config.scmc_tau, tau2=config.scmc_tau,
        )
        f1 = stream3d(f1)
        f2 = stream3d(f2)

    f_total = f1 + f2
    p_in, p_out, dp = _measure_pressure_jump_3d(f_total, r)
    max_u = _max_velocity_3d(f_total)
    sigma_eff = dp * r
    return {
        "r": r,
        "p_inside": round(p_in, 8),
        "p_outside": round(p_out, 8),
        "delta_p": round(dp, 8),
        "sigma_eff": round(sigma_eff, 6),
        "max_spurious_u": round(max_u, 8),
    }


def _run_cg_droplet_3d(
    r: float,
    config: StaticDroplet3DConfig,
    device: torch.device,
) -> dict[str, object]:
    nz, ny, nx = config.nz, config.ny, config.nx
    inside = _spherical_mask(nz, ny, nx, r, device)
    zero = torch.zeros((nz, ny, nx), device=device)
    rho_r = torch.where(
        inside, torch.full_like(zero, config.cg_rho_heavy), zero + config.cg_rho_light
    )
    rho_b = torch.where(
        inside, torch.full_like(zero, config.cg_rho_light), zero + config.cg_rho_heavy
    )
    f_r = equilibrium3d(rho_r, zero, zero, zero)
    f_b = equilibrium3d(rho_b, zero, zero, zero)

    for _ in range(config.n_steps):
        f_r, f_b = color_gradient_step_3d(
            f_r, f_b, tau=config.cg_tau, A=config.cg_A, beta=config.cg_beta,
        )
        f_r = stream3d(f_r)
        f_b = stream3d(f_b)

    f_total = f_r + f_b
    p_in, p_out, dp = _measure_pressure_jump_3d(f_total, r)
    max_u = _max_velocity_3d(f_total)
    sigma_eff = dp * r
    return {
        "r": r,
        "p_inside": round(p_in, 8),
        "p_outside": round(p_out, 8),
        "delta_p": round(dp, 8),
        "sigma_eff": round(sigma_eff, 6),
        "max_spurious_u": round(max_u, 8),
    }


def run_static_droplet_3d(config: StaticDroplet3DConfig) -> dict[str, object]:
    """Run 3-D static-droplet benchmark for SCMC and CG models."""
    config.validate()
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "static_droplet_3d", config.resolved_run_name(), config.overwrite,
    )
    results: dict[str, object] = {}

    for model_name in ("scmc", "cg"):
        print(f"\n--- 3D Static Droplet: {model_name.upper()} ---")
        per_r: list[dict[str, object]] = []
        for r in config.radii:
            print(f"  R={r:.0f}  steps={config.n_steps}")
            if model_name == "scmc":
                row = _run_scmc_droplet_3d(r, config, device)
            else:
                row = _run_cg_droplet_3d(r, config, device)
            print(
                f"    ΔP={row['delta_p']:.6f}  σ_eff={row['sigma_eff']:.4f}"
                f"  max_u={row['max_spurious_u']:.3e}"
            )
            per_r.append(row)

        sigma_fit = _fit_surface_tension(
            [float(d["r"]) for d in per_r],  # type: ignore[arg-type]
            [float(d["delta_p"]) for d in per_r],  # type: ignore[arg-type]
        )
        mean_u = sum(float(d["max_spurious_u"]) for d in per_r) / len(per_r)  # type: ignore[arg-type, misc]
        results[model_name] = {
            "per_radius": per_r,
            "sigma_eff_fit": round(sigma_fit, 8),
            "mean_max_spurious_u": round(mean_u, 8),
        }

    metadata: dict[str, object] = {
        "benchmark": "static_droplet_3d",
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "results": results,
        "note": "3D spherical Laplace benchmark on D3Q19 for SCMC/CG.",
    }
    (run_dir / "static_droplet_3d.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nSaved → {run_dir / 'static_droplet_3d.json'}")
    return metadata


@dataclass(frozen=True)
class Spinodal3DConfig:
    """Configuration for 3D spinodal decomposition benchmark (SCMP)."""

    nx: int = 40
    ny: int = 40
    nz: int = 40
    G: float = -4.0
    tau: float = 1.0
    rho0: float = 0.7
    noise_amp: float = 0.05
    n_steps: int = 1200
    output_interval: int = 300
    seed: int = 42
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        if self.nx < 12 or self.ny < 12 or self.nz < 12:
            msg = "nx, ny and nz must be ≥ 12"
            raise ValueError(msg)
        if self.G >= 0:
            msg = "G must be < 0 for SCMP phase separation (attractive interaction)"
            raise ValueError(msg)
        if self.tau <= 0.5:
            msg = "tau must be > 0.5"
            raise ValueError(msg)
        if self.rho0 <= 0:
            msg = "rho0 must be positive"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (
            f"spinodal_3d_G{self.G:.1f}_rho{self.rho0:.2f}"
            f"_n{self.nx}x{self.ny}x{self.nz}_steps{self.n_steps}"
        )


def run_spinodal_decomposition_3d(config: Spinodal3DConfig) -> dict[str, object]:
    """Run 3-D spinodal decomposition (SCMP) benchmark on D3Q19."""
    config.validate()
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "spinodal_3d", config.resolved_run_name(), config.overwrite,
    )

    torch.manual_seed(config.seed)
    rho0 = torch.full((config.nz, config.ny, config.nx), config.rho0, device=device)
    noise = (
        torch.rand((config.nz, config.ny, config.nx), device=device) - 0.5
    ) * 2.0 * config.noise_amp
    rho = (rho0 + noise).clamp(min=0.01)
    zero = torch.zeros((config.nz, config.ny, config.nx), device=device)
    f = equilibrium3d(rho, zero, zero, zero)
    diagnostics: list[dict[str, object]] = []

    for step in range(1, config.n_steps + 1):
        f = collide_sc_single_component_3d(
            f, G=config.G, tau=config.tau, psi_fn=psi_exp,
        )
        f = stream3d(f)
        if step % config.output_interval == 0 or step == config.n_steps:
            rho_cur, _, _, _ = macroscopic3d(f)
            rho_max = float(rho_cur.max().item())
            rho_min = float(rho_cur.min().item())
            rho_std = float(rho_cur.std().item())
            diagnostics.append(
                {
                    "step": step,
                    "rho_max": round(rho_max, 6),
                    "rho_min": round(rho_min, 6),
                    "rho_std": round(rho_std, 6),
                    "density_ratio": round(rho_max / max(rho_min, 1e-12), 4),
                }
            )

    rho_final, _, _, _ = macroscopic3d(f)
    rho_liquid = float(rho_final.max().item())
    rho_gas = float(rho_final.min().item())
    density_ratio = rho_liquid / max(rho_gas, 1e-12)
    phase_separated = density_ratio > 2.0

    metadata: dict[str, object] = {
        "benchmark": "spinodal_decomposition_3d",
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "rho_liquid": round(rho_liquid, 6),
        "rho_gas": round(rho_gas, 6),
        "density_ratio": round(density_ratio, 4),
        "phase_separated": phase_separated,
        "diagnostics": diagnostics,
        "note": "3D SCMP spinodal decomposition benchmark on D3Q19.",
    }
    (run_dir / "spinodal_3d.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"\nSaved → {run_dir / 'spinodal_3d.json'}"
        f"  (ρ_l={rho_liquid:.4f}, ρ_g={rho_gas:.4f}, ratio={density_ratio:.2f})"
    )
    return metadata


# ---------------------------------------------------------------------------
# Benchmark 4 – Two-Phase Poiseuille: SCMC vs CG
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoPhaseChannelCompareConfig:
    """Configuration for the two-phase Poiseuille comparison benchmark.

    Two immiscible fluids occupy the lower half (phase 1 / heavy) and upper
    half (phase 2 / light) of a 2-D channel driven by a body force ``G_x``.
    The benchmark runs both the SCMC and CG models and compares their
    steady-state velocity profiles against the analytical piecewise-parabolic
    solution.

    **SCMC** supports distinct relaxation times (viscosities) for each
    component, yielding a piecewise-parabolic profile with a kink at the
    interface.

    **CG** uses a single shared relaxation time (same viscosity for both
    components), yielding a simple parabolic profile.

    Attributes
    ----------
    nx, ny:            Domain size (channel: periodic in x, no-slip walls at
                       y=0 and y=ny-1).
    G_x:               Body-force acceleration in x (lattice units).
    n_steps:           Number of time steps to reach steady state.
    output_interval:   Diagnostic sampling interval.
    scmc_G12:          SC coupling constant.
    scmc_tau_heavy:    Relaxation time for the heavy (lower) component.
    scmc_tau_light:    Relaxation time for the light (upper) component.
    scmc_rho_heavy:    Density of the heavy component.
    scmc_rho_light:    Density of the light component.
    cg_A:              CG surface-tension amplitude.
    cg_beta:           CG recoloring parameter.
    cg_tau:            CG relaxation time (shared for both components).
    cg_rho_heavy:      Initial heavy-phase density in the lower half.
    cg_rho_light:      Corresponding minority fraction in the other half.
    output_root:       Root directory for output files.
    run_name:          Optional run identifier.
    device:            PyTorch device string.
    overwrite:         Overwrite existing run directory if *True*.
    """

    nx: int = 4
    ny: int = 40
    G_x: float = 5e-5
    n_steps: int = 8000
    output_interval: int = 2000
    # SCMC parameters
    scmc_G12: float = 0.9
    scmc_tau_heavy: float = 1.0
    scmc_tau_light: float = 0.7
    scmc_rho_heavy: float = 0.7
    scmc_rho_light: float = 0.3
    # CG parameters
    cg_A: float = 0.04
    cg_beta: float = 0.7
    cg_tau: float = 1.0
    cg_rho_heavy: float = 0.65
    cg_rho_light: float = 0.05
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        """Raise ValueError if configuration is invalid."""
        if self.ny < 10:
            msg = "ny must be ≥ 10"
            raise ValueError(msg)
        if self.scmc_tau_heavy <= 0.5 or self.scmc_tau_light <= 0.5:
            msg = "SCMC tau must be > 0.5"
            raise ValueError(msg)
        if self.cg_tau <= 0.5:
            msg = "CG tau must be > 0.5"
            raise ValueError(msg)
        if self.scmc_rho_heavy <= 0 or self.scmc_rho_light <= 0:
            msg = "SCMC densities must be positive"
            raise ValueError(msg)
        if self.cg_rho_heavy <= self.cg_rho_light:
            msg = "cg_rho_heavy must exceed cg_rho_light"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (
            f"two_phase_poiseuille_compare_ny{self.ny}"
            f"_tauw{self.scmc_tau_heavy:.2f}_taug{self.scmc_tau_light:.2f}"
        )

    def scmc_nu_heavy(self) -> float:
        """Kinematic viscosity of the heavy (lower) SCMC component."""
        return _CS2 * (self.scmc_tau_heavy - 0.5)

    def scmc_nu_light(self) -> float:
        """Kinematic viscosity of the light (upper) SCMC component."""
        return _CS2 * (self.scmc_tau_light - 0.5)

    def cg_nu(self) -> float:
        """Kinematic viscosity of the CG model (same for both components)."""
        return _CS2 * (self.cg_tau - 0.5)


def _poiseuille_analytical_two_phase(
    ny: int,
    half: int,
    G_x: float,
    nu_lower: float,
    nu_upper: float,
) -> list[float]:
    """Analytical piecewise-parabolic Poiseuille profile for two-fluid channel.

    The analytical solution accounts for the viscosity jump at y = half:
    * ``u''(y) = -G_x / ν``  in each layer,
    * continuity of velocity and shear stress at the interface.
    """
    H = float(ny - 1)
    h = float(half)
    a_lo = -G_x / (2.0 * nu_lower) if nu_lower > 1e-20 else 0.0
    a_hi = -G_x / (2.0 * nu_upper) if nu_upper > 1e-20 else 0.0

    # BCs: u(0)=0, u(H)=0, u continuity & shear continuity at y=h
    # Solve 2×2 for b_lo, b_hi (linear coefficients):
    A00, A01 = nu_lower, -nu_upper
    A10, A11 = h, H - h
    b0 = nu_upper * 2.0 * a_hi * h - nu_lower * 2.0 * a_lo * h
    b1 = a_hi * (h - H) * (h + H) - a_lo * h * h
    det = A00 * A11 - A01 * A10
    if abs(det) < 1e-20:
        return [0.0] * ny
    b_lo = (b0 * A11 - b1 * A01) / det
    b_hi = (A00 * b1 - A10 * b0) / det
    c_hi = -a_hi * H * H - b_hi * H

    profile = []
    for j in range(ny):
        y = float(j)
        u = a_lo * y * y + b_lo * y if j <= half else a_hi * y * y + b_hi * y + c_hi
        profile.append(max(u, 0.0))
    return profile


def _run_scmc_poiseuille(
    config: TwoPhaseChannelCompareConfig,
    device: torch.device,
) -> dict[str, object]:
    """Run SCMC two-phase Poiseuille and return velocity profile + error."""
    ny, nx = config.ny, config.nx
    half = ny // 2
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True

    # Use a larger minority fraction (0.15) to prevent early instability in the
    # SC two-component model: the heavy-phase density in the light region must be
    # large enough that the SC force does not produce extreme velocity shifts in
    # the first few timesteps.
    frac = 0.15
    zero = torch.zeros((ny, nx), device=device)
    rho1 = torch.zeros((ny, nx), device=device)
    rho2 = torch.zeros((ny, nx), device=device)
    rho1[:half, :] = config.scmc_rho_heavy
    rho1[half:, :] = config.scmc_rho_heavy * frac
    rho2[:half, :] = config.scmc_rho_light * frac
    rho2[half:, :] = config.scmc_rho_light

    f1 = equilibrium(rho1, zero, zero)
    f2 = equilibrium(rho2, zero, zero)

    for _ in range(config.n_steps):
        f1, f2 = collide_sc_two_component(
            f1, f2,
            G_12=config.scmc_G12,
            tau1=config.scmc_tau_heavy,
            tau2=config.scmc_tau_light,
            gx=config.G_x,
            solid_mask=wall,
        )
        f1 = stream(f1)
        f2 = stream(f2)
        f1 = bounce_back_cells(f1, wall)
        f2 = bounce_back_cells(f2, wall)

    # Mixture velocity
    rho1_f, ux1, _ = macroscopic(f1)
    rho2_f, ux2, _ = macroscopic(f2)
    rho_tot = (rho1_f + rho2_f).clamp(min=1e-12)
    ux_mix = (rho1_f * ux1 + rho2_f * ux2) / rho_tot
    ux_profile = ux_mix[:, nx // 2].tolist()

    analytical = _poiseuille_analytical_two_phase(
        ny, half, config.G_x, config.scmc_nu_heavy(), config.scmc_nu_light()
    )
    sim_t = torch.tensor(ux_profile[1:-1])
    ana_t = torch.tensor(analytical[1:-1])
    l2_err = float((sim_t - ana_t).norm() / (ana_t.norm().clamp(min=1e-15)))

    return {
        "velocity_profile": [round(v, 8) for v in ux_profile],
        "analytical_profile": [round(v, 8) for v in analytical],
        "l2_error_rel": round(l2_err, 6),
        "nu_heavy": round(config.scmc_nu_heavy(), 6),
        "nu_light": round(config.scmc_nu_light(), 6),
        "viscosity_ratio": round(config.scmc_nu_heavy() / max(config.scmc_nu_light(), 1e-12), 4),
    }


def _run_cg_poiseuille(
    config: TwoPhaseChannelCompareConfig,
    device: torch.device,
) -> dict[str, object]:
    """Run CG two-phase Poiseuille and return velocity profile + error."""
    ny, nx = config.ny, config.nx
    half = ny // 2
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True

    zero = torch.zeros((ny, nx), device=device)
    rho_r = torch.zeros((ny, nx), device=device)
    rho_b = torch.zeros((ny, nx), device=device)
    rho_r[:half, :] = config.cg_rho_heavy
    rho_r[half:, :] = config.cg_rho_light
    rho_b[:half, :] = config.cg_rho_light
    rho_b[half:, :] = config.cg_rho_heavy

    f_r = equilibrium(rho_r, zero, zero)
    f_b = equilibrium(rho_b, zero, zero)

    for _ in range(config.n_steps):
        f_r, f_b = color_gradient_step(
            f_r, f_b,
            tau=config.cg_tau,
            A=config.cg_A,
            beta=config.cg_beta,
            gx=config.G_x,
            solid_mask=wall,
        )
        f_r = stream(f_r)
        f_b = stream(f_b)
        f_r = bounce_back_cells(f_r, wall)
        f_b = bounce_back_cells(f_b, wall)

    # Total velocity (CG: both components share the same velocity field)
    f_total = f_r + f_b
    rho_tot, ux_tot, _ = macroscopic(f_total)
    ux_profile = ux_tot[:, nx // 2].tolist()

    # CG uses same τ for both → analytical = single-fluid parabola
    nu_cg = config.cg_nu()
    analytical = _poiseuille_analytical_two_phase(ny, half, config.G_x, nu_cg, nu_cg)
    sim_t = torch.tensor(ux_profile[1:-1])
    ana_t = torch.tensor(analytical[1:-1])
    l2_err = float((sim_t - ana_t).norm() / (ana_t.norm().clamp(min=1e-15)))

    return {
        "velocity_profile": [round(v, 8) for v in ux_profile],
        "analytical_profile": [round(v, 8) for v in analytical],
        "l2_error_rel": round(l2_err, 6),
        "nu": round(nu_cg, 6),
        "viscosity_ratio": 1.0,
    }


def run_two_phase_channel_compare(
    config: TwoPhaseChannelCompareConfig,
) -> dict[str, object]:
    """Run the two-phase Poiseuille comparison benchmark.

    Runs both SCMC and CG models in a two-fluid channel and compares the
    steady-state velocity profiles against the analytical solution.  SCMC
    supports a viscosity contrast between the phases; CG uses a shared
    relaxation time (viscosity ratio = 1).

    Args:
        config: Benchmark configuration.

    Returns:
        Dictionary with keys ``scmc`` and ``cg``, each containing
        ``velocity_profile``, ``analytical_profile``, ``l2_error_rel``, and
        viscosity parameters.
    """
    config.validate()
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "two_phase_poiseuille",
        config.resolved_run_name(), config.overwrite,
    )

    visc_ratio = config.scmc_nu_heavy() / max(config.scmc_nu_light(), 1e-12)
    print(f"\n--- Two-Phase Poiseuille: SCMC (νₕ/νₗ = {visc_ratio:.2f}) ---")
    scmc_result = _run_scmc_poiseuille(config, device)
    print(f"  SCMC L2 error = {scmc_result['l2_error_rel']:.4f}")

    print(f"\n--- Two-Phase Poiseuille: CG (ν = {config.cg_nu():.4f}, ratio = 1) ---")
    cg_result = _run_cg_poiseuille(config, device)
    print(f"  CG L2 error = {cg_result['l2_error_rel']:.4f}")

    results: dict[str, object] = {"scmc": scmc_result, "cg": cg_result}

    metadata: dict[str, object] = {
        "benchmark": "two_phase_poiseuille_compare",
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "results": results,
        "note": (
            "l2_error_rel = ||sim - analytical|| / ||analytical|| (excluding walls). "
            "SCMC supports viscosity contrast; CG uses a shared relaxation time."
        ),
    }
    (run_dir / "two_phase_poiseuille_compare.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nSaved → {run_dir / 'two_phase_poiseuille_compare.json'}")

    # Plot velocity profiles
    _plot_poiseuille(results, run_dir)

    return metadata


def _plot_poiseuille(results: dict[str, object], run_dir: Path) -> None:
    """Save a velocity-profile comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, model in zip(axes, ("scmc", "cg"), strict=True):
        if model not in results:
            continue
        data = results[model]
        sim = data["velocity_profile"]  # type: ignore[index]
        ana = data["analytical_profile"]  # type: ignore[index]
        y = list(range(len(sim)))
        ax.plot(ana, y, "k--", label="Analytical")
        ax.plot(sim, y, label=model.upper())
        ax.set_xlabel("u_x (lattice units)")
        ax.set_title(f"{model.upper()}  L2 err = {data['l2_error_rel']:.4f}")  # type: ignore[index]
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("y (lattice nodes)")
    fig.suptitle("Two-phase Poiseuille: velocity profiles")
    fig.tight_layout()
    fig.savefig(run_dir / "poiseuille_profiles.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------


def _generate_analysis(
    droplet: dict[str, object],
    spinodal: dict[str, object],
    free_energy: dict[str, object],
    poiseuille: dict[str, object],
    droplet3d: dict[str, object] | None = None,
    spinodal3d: dict[str, object] | None = None,
) -> dict[str, object]:
    """Generate a concise quantitative analysis of all benchmark results."""
    from typing import cast  # noqa: PLC0415

    analysis: dict[str, object] = {}

    # Surface tension comparison
    scmc_res = cast("dict[str, dict[str, object]]", droplet.get("results", {}))
    sigma_scmc = float(scmc_res.get("scmc", {}).get("sigma_eff_fit", float("nan")))  # type: ignore[arg-type]
    sigma_cg = float(scmc_res.get("cg", {}).get("sigma_eff_fit", float("nan")))  # type: ignore[arg-type]
    u_scmc = float(scmc_res.get("scmc", {}).get("mean_max_spurious_u", float("nan")))  # type: ignore[arg-type]
    u_cg = float(scmc_res.get("cg", {}).get("mean_max_spurious_u", float("nan")))  # type: ignore[arg-type]
    analysis["surface_tension"] = {
        "scmc_sigma_eff": round(sigma_scmc, 6),
        "cg_sigma_eff": round(sigma_cg, 6),
        "scmc_mean_spurious_u": round(u_scmc, 8),
        "cg_mean_spurious_u": round(u_cg, 8),
        "lower_spurious_currents": "cg" if u_cg < u_scmc else "scmc",
    }

    # Spinodal
    rho_l = float(spinodal.get("rho_liquid", 0.0))  # type: ignore[arg-type]
    rho_g = float(spinodal.get("rho_gas", 0.0))  # type: ignore[arg-type]
    ratio = float(spinodal.get("density_ratio", 0.0))  # type: ignore[arg-type]
    analysis["spinodal"] = {
        "rho_liquid": round(rho_l, 4),
        "rho_gas": round(rho_g, 4),
        "density_ratio": round(ratio, 3),
        "phase_separated": bool(spinodal.get("phase_separated", False)),
    }

    analysis["free_energy"] = {
        "relative_phase_mass_drift": round(
            float(free_energy.get("relative_phase_mass_drift", float("nan"))), 8,  # type: ignore[arg-type]
        ),
        "relative_radius_drift": round(
            float(free_energy.get("relative_radius_drift", float("nan"))), 8,  # type: ignore[arg-type]
        ),
        "max_spurious_u": round(
            float(free_energy.get("max_spurious_u", float("nan"))), 8,  # type: ignore[arg-type]
        ),
        "max_phase_overshoot": round(
            float(free_energy.get("max_phase_overshoot", float("nan"))), 8,  # type: ignore[arg-type]
        ),
        "bounded_phase_field": bool(free_energy.get("bounded_phase_field", False)),
    }

    # Poiseuille
    pois_res = cast("dict[str, dict[str, object]]", poiseuille.get("results", {}))
    scmc_err = float(pois_res.get("scmc", {}).get("l2_error_rel", float("nan")))  # type: ignore[arg-type]
    cg_err = float(pois_res.get("cg", {}).get("l2_error_rel", float("nan")))  # type: ignore[arg-type]
    scmc_visc_ratio = float(pois_res.get("scmc", {}).get("viscosity_ratio", 1.0))  # type: ignore[arg-type]
    analysis["poiseuille"] = {
        "scmc_l2_error": round(scmc_err, 6),
        "cg_l2_error": round(cg_err, 6),
        "scmc_viscosity_ratio": round(scmc_visc_ratio, 4),
        "better_accuracy": "cg" if cg_err < scmc_err else "scmc",
    }

    if droplet3d is not None and spinodal3d is not None:
        d3_res = cast("dict[str, dict[str, object]]", droplet3d.get("results", {}))
        sigma3d_scmc = float(d3_res.get("scmc", {}).get("sigma_eff_fit", float("nan")))  # type: ignore[arg-type]
        sigma3d_cg = float(d3_res.get("cg", {}).get("sigma_eff_fit", float("nan")))  # type: ignore[arg-type]
        ratio3d = float(spinodal3d.get("density_ratio", 0.0))  # type: ignore[arg-type]
        analysis["three_d"] = {
            "scmc_sigma_eff": round(sigma3d_scmc, 6),
            "cg_sigma_eff": round(sigma3d_cg, 6),
            "spinodal_density_ratio": round(ratio3d, 3),
            "spinodal_phase_separated": bool(spinodal3d.get("phase_separated", False)),
        }

    # Spinodal G value for the summary
    spinodal_cfg = cast("dict[str, object]", spinodal.get("config", {}))
    g_val = spinodal_cfg.get("G", "N/A")
    fe_mass_drift = float(free_energy.get("relative_phase_mass_drift", float("nan")))  # type: ignore[arg-type]
    fe_radius_drift = float(free_energy.get("relative_radius_drift", float("nan")))  # type: ignore[arg-type]
    fe_max_u = float(free_energy.get("max_spurious_u", float("nan")))  # type: ignore[arg-type]
    fe_bounded = bool(free_energy.get("bounded_phase_field", False))

    # Narrative summary
    summary_lines = [
        "=== Multiphase LBM Model Comparison ===",
        "",
        "1. Laplace Pressure Test (static circular droplet, periodic domain):",
        f"   SCMC (Shan-Chen two-component): σ_eff = {sigma_scmc:.4f} lu",
        f"   CG  (Color-Gradient):           σ_eff = {sigma_cg:.4f} lu",
        "   → SCMC and CG both satisfy ΔP ∝ 1/R (Young-Laplace).",
        f"   Spurious currents — SCMC: {u_scmc:.2e}, CG: {u_cg:.2e}",
        "",
        "2. Spinodal Decomposition (SCMP, Shan-Chen single-component):",
        f"   G = {g_val}  (attractive self-interaction)",
        f"   Coexistence: ρ_liquid = {rho_l:.4f}, ρ_gas = {rho_g:.4f}",
        f"   Density ratio ρ_l/ρ_g = {ratio:.2f}",
        f"   Phase separation achieved: {bool(spinodal.get('phase_separated', False))}",
        "",
        "3. Free-Energy droplet relaxation (phase-field benchmark):",
        f"   Relative phase-mass drift: {fe_mass_drift:.2e}",
        f"   Relative radius drift:     {fe_radius_drift:.2e}",
        f"   Max spurious current:      {fe_max_u:.2e}",
        f"   Phase field bounded:       {fe_bounded}",
        "",
        "4. Two-Phase Poiseuille (channel flow with body force):",
        f"   SCMC viscosity ratio ν_heavy/ν_light = {scmc_visc_ratio:.2f}",
        f"   SCMC L2 error vs analytical: {scmc_err:.4f}",
        "   CG  viscosity ratio = 1 (shared τ)",
        f"   CG  L2 error vs analytical: {cg_err:.4f}",
        "",
        "Key model characteristics:",
        "  SCMC: density-contrast phases, adjustable viscosity ratio, diffuse interface.",
        "  CG  : sharp interface via recoloring, single shared viscosity.",
        "  SCMP: single-component phase separation (liquid/gas EOS), density ratio from G.",
        "  FE  : thermodynamically consistent, conserved order parameter, tunable σ and ξ.",
    ]
    if droplet3d is not None and spinodal3d is not None:
        summary_lines.extend(
            [
                "",
                "5. 3D multiphase benchmarks (D3Q19):",
                f"   3D Laplace σ_eff — SCMC: {sigma3d_scmc:.4f}, CG: {sigma3d_cg:.4f}",
                f"   3D spinodal density ratio ρ_l/ρ_g = {ratio3d:.2f}",
            ]
        )
    analysis["summary"] = "\n".join(summary_lines)
    return analysis


@dataclass
class MultiphaseBenchmarkSuiteConfig:
    """Top-level configuration for the full multiphase benchmark suite.

    Bundles the individual benchmark configs and shared I/O settings.

    Attributes
    ----------
    droplet:    Config for the static-droplet / Laplace benchmark.
    spinodal:   Config for the spinodal decomposition benchmark.
    free_energy: Config for the phase-field droplet benchmark.
    poiseuille: Config for the two-phase Poiseuille comparison.
    output_root: Root directory for all output files.
    device:     PyTorch device string.
    overwrite:  Overwrite existing run directories if *True*.
    """

    droplet: StaticDropletConfig = field(default_factory=StaticDropletConfig)
    spinodal: SpinodaleConfig = field(default_factory=SpinodaleConfig)
    free_energy: FreeEnergyDropletConfig = field(default_factory=FreeEnergyDropletConfig)
    poiseuille: TwoPhaseChannelCompareConfig = field(default_factory=TwoPhaseChannelCompareConfig)
    droplet_3d: StaticDroplet3DConfig | None = None
    spinodal_3d: Spinodal3DConfig | None = None
    output_root: Path = Path("outputs")
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)


def run_multiphase_benchmark_suite(
    config: MultiphaseBenchmarkSuiteConfig | None = None,
) -> dict[str, object]:
    """Run the full multiphase LBM benchmark suite.

    Executes the canonical 2-D benchmarks (static droplet, spinodal
    decomposition, free-energy droplet, two-phase Poiseuille) in sequence and
    assembles a comprehensive comparison report with quantitative metrics and a
    brief narrative analysis.

    The report is saved as ``multiphase_suite_report.json`` under *output_root*.

    Args:
        config: Suite configuration.  If ``None``, sensible defaults are used.

    Returns:
        Nested dictionary containing the raw results of all benchmark cases
        and a top-level ``analysis`` section with summary statistics.
    """
    if config is None:
        config = MultiphaseBenchmarkSuiteConfig()

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Propagate shared settings to individual configs
    def _patch(cfg: object, **kwargs: object) -> object:
        """Return a new dataclass instance with updated fields."""
        import dataclasses  # noqa: PLC0415
        return dataclasses.replace(cfg, **kwargs)  # type: ignore[type-var]

    droplet_cfg: StaticDropletConfig = _patch(  # type: ignore[assignment]
        config.droplet,
        output_root=output_root,
        device=config.device,
        overwrite=config.overwrite,
    )
    spinodal_cfg: SpinodaleConfig = _patch(  # type: ignore[assignment]
        config.spinodal,
        output_root=output_root,
        device=config.device,
        overwrite=config.overwrite,
    )
    free_energy_cfg: FreeEnergyDropletConfig = _patch(  # type: ignore[assignment]
        config.free_energy,
        output_root=output_root,
        device=config.device,
        overwrite=config.overwrite,
    )
    poiseuille_cfg: TwoPhaseChannelCompareConfig = _patch(  # type: ignore[assignment]
        config.poiseuille,
        output_root=output_root,
        device=config.device,
        overwrite=config.overwrite,
    )
    droplet_3d_cfg: StaticDroplet3DConfig | None = None
    if config.droplet_3d is not None:
        droplet_3d_cfg = _patch(  # type: ignore[assignment]
            config.droplet_3d,
            output_root=output_root,
            device=config.device,
            overwrite=config.overwrite,
        )
    spinodal_3d_cfg: Spinodal3DConfig | None = None
    if config.spinodal_3d is not None:
        spinodal_3d_cfg = _patch(  # type: ignore[assignment]
            config.spinodal_3d,
            output_root=output_root,
            device=config.device,
            overwrite=config.overwrite,
        )

    print("=" * 60)
    print("TensorLBM Multiphase Model Benchmark Suite")
    print("=" * 60)

    print("\n[1/4] Static Droplet (Laplace pressure + spurious currents)")
    droplet_result = run_static_droplet(droplet_cfg)

    print("\n[2/4] Spinodal Decomposition (SCMP)")
    spinodal_result = run_spinodal_decomposition(spinodal_cfg)

    print("\n[3/4] Free-Energy droplet relaxation")
    free_energy_result = run_free_energy_droplet(free_energy_cfg)

    print("\n[4/4] Two-Phase Poiseuille (SCMC vs CG)")
    poiseuille_result = run_two_phase_channel_compare(poiseuille_cfg)

    droplet_3d_result: dict[str, object] | None = None
    spinodal_3d_result: dict[str, object] | None = None
    if droplet_3d_cfg is not None:
        print("\n[4/5] 3D Static Droplet (Laplace pressure + spurious currents)")
        droplet_3d_result = run_static_droplet_3d(droplet_3d_cfg)
    if spinodal_3d_cfg is not None:
        print("\n[5/5] 3D Spinodal Decomposition (SCMP)")
        spinodal_3d_result = run_spinodal_decomposition_3d(spinodal_3d_cfg)

    # Analysis
    analysis = _generate_analysis(
        droplet_result,
        spinodal_result,
        free_energy_result,
        poiseuille_result,
        droplet_3d_result,
        spinodal_3d_result,
    )
    print("\n" + str(analysis["summary"]))

    report: dict[str, object] = {
        "benchmarks": {
            "static_droplet": droplet_result,
            "spinodal_decomposition": spinodal_result,
            "free_energy_droplet": free_energy_result,
            "two_phase_poiseuille": poiseuille_result,
        },
        "analysis": analysis,
    }
    if droplet_3d_result is not None:
        report["benchmarks"]["static_droplet_3d"] = droplet_3d_result  # type: ignore[index]
    if spinodal_3d_result is not None:
        report["benchmarks"]["spinodal_decomposition_3d"] = spinodal_3d_result  # type: ignore[index]

    report_path = output_root / "multiphase_suite_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\n[Done] Full report saved → {report_path}")
    return report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Benchmark 1: static droplet
    "StaticDropletConfig",
    "run_static_droplet",
    # Benchmark 2: spinodal decomposition
    "SpinodaleConfig",
    "run_spinodal_decomposition",
    # Benchmark 3: free-energy droplet relaxation
    "FreeEnergyDropletConfig",
    "run_free_energy_droplet",
    # Benchmark 4: 3D static droplet + spinodal
    "StaticDroplet3DConfig",
    "run_static_droplet_3d",
    "Spinodal3DConfig",
    "run_spinodal_decomposition_3d",
    # Benchmark 5: two-phase Poiseuille comparison
    "TwoPhaseChannelCompareConfig",
    "run_two_phase_channel_compare",
    # Suite runner
    "MultiphaseBenchmarkSuiteConfig",
    "run_multiphase_benchmark_suite",
]
