"""Passive scalar transport via lattice Boltzmann method (D2Q5).

Implements an advection–diffusion LBM solver for a passive scalar field
(concentration, pollutant, CO2, temperature-as-tracer) coupled to an
existing D2Q9 momentum solver.  This is analogous to the species/passive-
scalar transport available in XFlow and PowerFlow.

Physics
-------
The scalar field C (concentration) obeys:

    ∂C/∂t + u · ∇C = D ∇²C  +  S(x, t)

where D is the diffusivity and S is an optional source term.

LBM implementation uses the D2Q5 lattice (same as the thermal solver) with
BGK collision:

    g_i* = g_i − (g_i − g_i^eq) / τ_D
    g_i^eq = w_i C (1 + 3 (c_{ix} ux + c_{iy} uy))
    τ_D = D / c_s² + 0.5 = 3D + 0.5

where c_s² = 1/3 for D2Q5.

References
----------
Shi, B., & Guo, Z. (2009). Lattice Boltzmann model for nonlinear
    convection-diffusion equations. *Phys. Rev. E* 79, 016701.
He, X., Chen, S., & Doolen, G. D. (1998). A novel thermal model for the
    lattice Boltzmann method in incompressible limit.
    *J. Comput. Phys.* 146(1), 282–300.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

__all__ = [
    "PassiveScalarConfig",
    "equilibrium_scalar",
    "collide_scalar_bgk",
    "stream_scalar",
    "macroscopic_scalar",
    "apply_scalar_dirichlet_bc",
    "run_passive_scalar_transport",
]

# D2Q5 lattice
_CX5 = torch.tensor([0.0,  1.0,  0.0, -1.0,  0.0])
_CY5 = torch.tensor([0.0,  0.0,  1.0,  0.0, -1.0])
_W5 = torch.tensor([2.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0])
_CS2_D2Q5 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Core LBM operators (D2Q5 scalar)
# ---------------------------------------------------------------------------

def equilibrium_scalar(
    c: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> torch.Tensor:
    """D2Q5 scalar equilibrium distribution.

    g_i^eq = w_i C (1 + (c_{ix} ux + c_{iy} uy) / c_s²)

    Args:
        c:  Scalar concentration field, shape ``(ny, nx)``.
        ux: x-velocity, shape ``(ny, nx)``.
        uy: y-velocity, shape ``(ny, nx)``.

    Returns:
        Equilibrium DFs, shape ``(5, ny, nx)``.
    """
    device = c.device
    cx = _CX5.to(device).view(5, 1, 1)
    cy = _CY5.to(device).view(5, 1, 1)
    w = _W5.to(device).view(5, 1, 1)

    cu = cx * ux + cy * uy  # (5, ny, nx)
    return w * c * (1.0 + cu / _CS2_D2Q5)


def collide_scalar_bgk(
    g: torch.Tensor,
    c: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    tau_d: float,
    source: torch.Tensor | None = None,
) -> torch.Tensor:
    """BGK collision step for the scalar distribution.

    Args:
        g:      Scalar DF, shape ``(5, ny, nx)``.
        c:      Scalar concentration, shape ``(ny, nx)``.
        ux, uy: Velocity fields, shape ``(ny, nx)``.
        tau_d:  Scalar relaxation time (τ_D = 3D + 0.5).
        source: Optional scalar source term, shape ``(ny, nx)``.

    Returns:
        Post-collision scalar DF, shape ``(5, ny, nx)``.
    """
    g_eq = equilibrium_scalar(c, ux, uy)
    g_out = g - (g - g_eq) / tau_d
    if source is not None:
        device = g.device
        w = _W5.to(device).view(5, 1, 1)
        g_out = g_out + w * source
    return g_out


def stream_scalar(g: torch.Tensor) -> torch.Tensor:
    """Streaming step for D2Q5 scalar DFs.

    Args:
        g: Scalar DF, shape ``(5, ny, nx)``.

    Returns:
        Streamed scalar DF, shape ``(5, ny, nx)``.
    """
    g_new = torch.empty_like(g)
    g_new[0] = g[0]
    g_new[1] = torch.roll(g[1], shifts=1, dims=1)   # +x
    g_new[2] = torch.roll(g[2], shifts=1, dims=0)   # +y
    g_new[3] = torch.roll(g[3], shifts=-1, dims=1)  # -x
    g_new[4] = torch.roll(g[4], shifts=-1, dims=0)  # -y
    return g_new


def macroscopic_scalar(g: torch.Tensor) -> torch.Tensor:
    """Recover scalar concentration from D2Q5 DFs.

    Args:
        g: Scalar DF, shape ``(5, ny, nx)``.

    Returns:
        Scalar concentration C, shape ``(ny, nx)``.
    """
    return g.sum(dim=0)


def apply_scalar_dirichlet_bc(
    g: torch.Tensor,
    c_value: float,
    mask: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> torch.Tensor:
    """Enforce Dirichlet scalar BC on masked cells.

    Resets DFs to equilibrium at the prescribed concentration.

    Args:
        g:       Scalar DF, shape ``(5, ny, nx)``.
        c_value: Prescribed scalar value.
        mask:    Boolean mask of Dirichlet cells, shape ``(ny, nx)``.
        ux, uy:  Velocity fields, shape ``(ny, nx)``.

    Returns:
        Updated scalar DF.
    """
    c_bc = torch.full_like(ux, c_value)
    g_eq = equilibrium_scalar(c_bc, ux, uy)
    mask_exp = mask.unsqueeze(0).expand(5, -1, -1)
    return torch.where(mask_exp, g_eq, g)


# ---------------------------------------------------------------------------
# Configuration and runner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PassiveScalarConfig:
    """Configuration for a 2-D passive scalar transport simulation."""

    nx: int = 256
    ny: int = 128
    u_in: float = 0.05          # inlet velocity (l.u.)
    re: float = 100.0           # flow Reynolds number
    diffusivity: float = 0.01   # scalar diffusivity D (l.u.)
    # Scalar source: circular patch at (src_cx, src_cy) with radius src_r
    src_cx: float = 0.15        # source centre x (fraction of nx)
    src_cy: float = 0.5         # source centre y (fraction of ny)
    src_radius: float = 0.04    # source radius (fraction of nx)
    src_strength: float = 0.1   # source emission rate (l.u. per step)
    # Obstacle: circular cylinder at (cx, cy)
    cyl_cx: float = 0.25
    cyl_cy: float = 0.5
    cyl_radius: float = 0.06    # fraction of ny
    n_steps: int = 3000
    output_interval: int = 500
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    @property
    def nu(self) -> float:
        L = max(self.cyl_radius * self.ny, 1.0)
        return self.u_in * L / self.re

    @property
    def tau_f(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def tau_d(self) -> float:
        return 3.0 * self.diffusivity + 0.5


def run_passive_scalar_transport(
    cfg: PassiveScalarConfig | None = None,
    **kwargs: object,
) -> Path:
    """Run a 2-D passive scalar transport simulation.

    Couples a D2Q9 BGK flow solver with a D2Q5 scalar advection–diffusion
    solver to model pollutant/tracer dispersion past a cylinder.

    Args:
        cfg:     Configuration object.
        **kwargs: Override any :class:`PassiveScalarConfig` field.

    Returns:
        Path to the run output directory.
    """
    from .boundaries import (  # noqa: PLC0415
        apply_simple_channel_boundaries,
        bounce_back_cells,
        cylinder_mask,
    )
    from .config_io import save_config_json  # noqa: PLC0415
    from .d2q9 import equilibrium as f_equilibrium  # noqa: PLC0415
    from .d2q9 import macroscopic
    from .logging_config import configure_logging  # noqa: PLC0415
    from .logging_config import logger as _logger
    from .solver import collide_bgk, stream  # noqa: PLC0415
    from .utils import (  # noqa: PLC0415
        get_reproducibility_metadata,
        prepare_run_dir,
        resolve_device,
    )

    if cfg is None:
        valid = set(PassiveScalarConfig.__dataclass_fields__)
        cfg = PassiveScalarConfig(**{k: v for k, v in kwargs.items() if k in valid})

    device = resolve_device(cfg.device)
    run_dir = prepare_run_dir(cfg.output_root, cfg.run_name or "passive_scalar", cfg.overwrite)
    configure_logging(run_dir)
    save_config_json(asdict(cfg), run_dir / "config.json")

    _logger.info("Passive scalar: nx=%d ny=%d Re=%.1f D=%.4f device=%s",
                 cfg.nx, cfg.ny, cfg.re, cfg.diffusivity, cfg.device)

    nx, ny = cfg.nx, cfg.ny

    # Solid mask
    R_cyl = cfg.cyl_radius * ny
    cx_pix = int(cfg.cyl_cx * nx)
    cy_pix = int(cfg.cyl_cy * ny)
    solid = cylinder_mask(ny, nx, cy_pix, cx_pix, R_cyl, device)

    # Source mask (Dirichlet scalar = 1 at source patch)
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )
    src_x = cfg.src_cx * nx
    src_y = cfg.src_cy * ny
    src_r = cfg.src_radius * nx
    source_mask = ((xx - src_x) ** 2 + (yy - src_y) ** 2) < src_r ** 2

    # Init flow
    rho = torch.ones(ny, nx, device=device)
    ux0 = torch.full((ny, nx), cfg.u_in, device=device)
    uy0 = torch.zeros(ny, nx, device=device)
    f = f_equilibrium(rho, ux0, uy0)

    # Init scalar
    c0 = torch.zeros(ny, nx, device=device)
    ux_init = ux0.clone()
    uy_init = uy0.clone()
    g = equilibrium_scalar(c0, ux_init, uy_init)

    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    steps_out: list[int] = []
    c_mean_out: list[float] = []

    for step in range(cfg.n_steps):
        # --- Flow solver ---
        rho, ux, uy = macroscopic(f)
        f = collide_bgk(f, rho, ux, uy, cfg.tau_f)
        f = bounce_back_cells(f, solid)
        f = apply_simple_channel_boundaries(f, cfg.u_in)
        f = stream(f)

        # --- Scalar solver ---
        c = macroscopic_scalar(g)
        g = collide_scalar_bgk(g, c, ux, uy, cfg.tau_d)
        # Dirichlet BC at source
        g = apply_scalar_dirichlet_bc(g, 1.0, source_mask, ux, uy)
        # Zero flux at solid (bounce-back)
        g = bounce_back_cells(g, solid)
        # Inlet: zero scalar
        g[:, :, 0] = equilibrium_scalar(
            torch.zeros(ny, device=device), ux[:, 0], uy[:, 0]
        )
        g = stream_scalar(g)

        if (step + 1) % cfg.output_interval == 0 or step == cfg.n_steps - 1:
            c_pp = macroscopic_scalar(g)
            rho_pp, ux_pp, uy_pp = macroscopic(f)
            speed = torch.sqrt(ux_pp ** 2 + uy_pp ** 2)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(speed.cpu().numpy(), origin="lower", cmap="RdBu_r")
            axes[0].set_title(f"Velocity – step {step + 1}")
            im = axes[1].imshow(c_pp.cpu().numpy(), origin="lower", cmap="hot", vmin=0, vmax=1)
            axes[1].set_title("Scalar C")
            plt.colorbar(im, ax=axes[1])
            plt.tight_layout()
            fig.savefig(run_dir / f"step_{step + 1:06d}.png", dpi=100)
            plt.close(fig)

            c_mean = float(c_pp.mean().item())
            steps_out.append(step + 1)
            c_mean_out.append(c_mean)
            _logger.info("step=%d  <C>=%.4f", step + 1, c_mean)

    with (run_dir / "scalar_stats.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "c_mean"])
        w.writerows(zip(steps_out, c_mean_out, strict=True))

    meta = {
        **get_reproducibility_metadata(),
        "config": asdict(cfg),
        "steps": steps_out,
        "c_mean": c_mean_out,
    }
    with (run_dir / "run_metadata.json").open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    _logger.info("Passive scalar transport complete → %s", run_dir)
    return run_dir
