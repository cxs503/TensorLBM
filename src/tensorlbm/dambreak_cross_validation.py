"""Dam-break cross-validation runner: D3Q19/D3Q27 × BGK/MRT × SGS.

Runs a small-grid dam-break simulation across all combinations of:
  - Lattice:   D3Q19, D3Q27
  - Collision: BGK, MRT
  - SGS model: none, Smagorinsky, WALE, Vreman

and produces a machine-readable artifact (JSON) with the front-position
comparison matrix.

This is a **diagnostic-only** tool.  ``status='diagnostic_only'`` and
``physical_validation=False`` are always set in the output.  The small
grid (default 24×12×12, 50 steps) is insufficient for physical
validation; the purpose is to verify that all solver combinations run
to completion and to compare their front-position signatures.

The runner uses the existing free-surface steps
(:func:`free_surface_lbm.free_surface_step` for D3Q19 and
:func:`free_surface_lbm_27.free_surface_step_27` for D3Q27) without
modifying their hot paths.  SGS models are selected via the ``sgs_model``
parameter (Smagorinsky/WALE/Vreman) and the ``C_s`` constant, which is
interpreted as the model-appropriate constant (C_s, C_w, or C_V).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from .d3q19 import equilibrium3d
from .d3q27 import equilibrium27
from .free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    free_surface_step,
    init_fill_rectangular,
    init_flags_from_fill,
    init_mass_from_fill,
)
from .free_surface_lbm_27 import (
    free_surface_step_27,
    init_fill_rectangular_27,
    init_flags_from_fill_27,
    init_mass_from_fill_27,
)
from .utils import resolve_device

Lattice = Literal["d3q19", "d3q27"]
Collision = Literal["bgk", "mrt"]
SgsModel = Literal["none", "smagorinsky", "wale", "vreman"]

# SGS model constants (defaults from the turbulence module)
_SGS_CONSTANTS = {
    "smagorinsky": 0.1,    # C_s
    "wale": 0.5,            # C_w
    "vreman": 0.025,        # C_V
}


@dataclass(frozen=True)
class CrossValidationConfig:
    """Configuration for the dam-break cross-validation runner.

    Default grid is 24×12×12 with 50 steps — small enough for fast
    cross-validation, large enough to observe front advancement.
    """

    # Grid
    nx: int = 24
    ny: int = 12
    nz: int = 12
    # Dam-break geometry
    dam_width: int = 8
    fill_height: int = 8
    # Time-stepping
    n_steps: int = 50
    # Physics (lattice units)
    tau: float = 0.8
    gravity: float = 1e-4
    rho_liquid: float = 1.0
    rho_gas: float = 0.1
    # SGS constants (used when sgs_model != 'none')
    C_smag: float = 0.1
    C_wale: float = 0.5
    C_vreman: float = 0.025
    # Runtime
    device: str = "cpu"
    output_path: str = "dambreak_cross_validation.json"

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        if self.nx < 8 or self.ny < 4 or self.nz < 4:
            raise ValueError("grid too small (need nx>=8, ny>=4, nz>=4)")
        if self.dam_width <= 0 or self.dam_width >= self.nx:
            raise ValueError("dam_width must be in (0, nx)")
        if self.fill_height <= 0 or self.fill_height > self.ny - 1:
            raise ValueError("fill_height must be in (0, ny-1)")
        if self.tau <= 0.5:
            raise ValueError(f"tau={self.tau} <= 0.5")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")


def _sgs_constant(sgs_model: SgsModel, config: CrossValidationConfig) -> float:
    """Return the model-appropriate SGS constant, or 0 for 'none'."""
    if sgs_model == "none":
        return 0.0
    if sgs_model == "smagorinsky":
        return config.C_smag
    if sgs_model == "wale":
        return config.C_wale
    if sgs_model == "vreman":
        return config.C_vreman
    raise ValueError(f"unknown sgs_model: {sgs_model}")


def _find_front_position(flags: torch.Tensor) -> float:
    """Find the rightmost x-column containing LIQUID or INTERFACE cells.

    Uses the flag field directly (same approach as
    :func:`dam_break_3d._find_front_x_3d` with ``flags`` argument).
    """
    liquid_or_iface = (flags == LIQUID) | (flags == INTERFACE)
    # flags shape is (nz, ny, nx); collapse z and y to get per-x-column
    water_cols = liquid_or_iface.any(dim=1).any(dim=0).int()
    front = water_cols.nonzero(as_tuple=True)[0]
    return float(front.max().item()) if front.numel() > 0 else 0.0


def _run_d3q19(
    config: CrossValidationConfig,
    collision: Collision,
    sgs_model: SgsModel,
) -> dict:
    """Run a single D3Q19 dam-break simulation."""
    device = resolve_device(config.device)
    nx, ny, nz = config.nx, config.ny, config.nz

    # Initialize geometry
    fill, solid = init_fill_rectangular(
        nz, ny, nx,
        column_width=float(config.dam_width),
        column_height=float(config.fill_height),
        device=device,
    )
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags, config.rho_liquid)

    # Initialize distribution: equilibrium for liquid/interface, zero for gas
    active = (flags == LIQUID) | (flags == INTERFACE)
    zero_f = torch.zeros((nz, ny, nx), device=device)
    rho_init = torch.where(active, torch.ones((nz, ny, nx), device=device), zero_f)
    f = equilibrium3d(rho_init, zero_f, zero_f, zero_f)

    initial_mass = float(mass.sum().item())
    C_s = _sgs_constant(sgs_model, config)
    gy = -config.gravity

    # Use 'smagorinsky' as the sgs_model name for the solver (it's the
    # default); the actual model is selected by the sgs_model parameter.
    # When C_s=0, no SGS is applied regardless of sgs_model name.
    sgs_name = sgs_model if sgs_model != "none" else "smagorinsky"

    for _step in range(config.n_steps):
        f, fill, flags, mass, _df = free_surface_step(
            f, fill, flags, solid,
            mass=mass,
            tau=config.tau,
            gy=gy,
            rho_liquid=config.rho_liquid,
            rho_gas=config.rho_gas,
            C_s=C_s,
            sgs_model=sgs_name,
            collision=collision,
        )

    front = _find_front_position(flags)
    mass_drift = float(mass.sum().item()) - initial_mass
    finite = bool(
        torch.isfinite(f).all()
        and torch.isfinite(fill).all()
        and torch.isfinite(mass).all()
    )

    return {
        "lattice": "d3q19",
        "collision": collision,
        "sgs_model": sgs_model,
        "front_position": front,
        "mass_drift": mass_drift,
        "finite": finite,
    }


def _run_d3q27(
    config: CrossValidationConfig,
    collision: Collision,
    sgs_model: SgsModel,
) -> dict:
    """Run a single D3Q27 dam-break simulation."""
    device = resolve_device(config.device)
    nx, ny, nz = config.nx, config.ny, config.nz

    # Initialize geometry
    fill, solid = init_fill_rectangular_27(
        nz, ny, nx,
        column_width=int(config.dam_width),
        column_height=int(config.fill_height),
        device=device,
    )
    flags = init_flags_from_fill_27(fill, solid)
    mass = init_mass_from_fill_27(fill, flags, config.rho_liquid)

    # Initialize distribution: equilibrium for liquid/interface, zero for gas
    active = (flags == LIQUID) | (flags == INTERFACE)
    zero_f = torch.zeros((nz, ny, nx), device=device)
    rho_init = torch.where(active, torch.ones((nz, ny, nx), device=device), zero_f)
    f = equilibrium27(rho_init, zero_f, zero_f, zero_f)

    initial_mass = float(mass.sum().item())
    C_s = _sgs_constant(sgs_model, config)
    gy = -config.gravity

    sgs_name = sgs_model if sgs_model != "none" else "smagorinsky"

    for _step in range(config.n_steps):
        f, fill, flags, mass, _df = free_surface_step_27(
            f, fill, flags, solid,
            mass=mass,
            tau=config.tau,
            gy=gy,
            rho_liquid=config.rho_liquid,
            rho_gas=config.rho_gas,
            C_s=C_s,
            sgs_model=sgs_name,
            collision=collision,
        )

    front = _find_front_position(flags)
    mass_drift = float(mass.sum().item()) - initial_mass
    finite = bool(
        torch.isfinite(f).all()
        and torch.isfinite(fill).all()
        and torch.isfinite(mass).all()
    )

    return {
        "lattice": "d3q27",
        "collision": collision,
        "sgs_model": sgs_model,
        "front_position": front,
        "mass_drift": mass_drift,
        "finite": finite,
    }


def run_single_dambreak(
    lattice: Lattice,
    collision: Collision,
    sgs_model: SgsModel,
    config: CrossValidationConfig,
) -> dict:
    """Run a single dam-break simulation and return the result dict.

    Args:
        lattice: ``'d3q19'`` or ``'d3q27'``.
        collision: ``'bgk'`` or ``'mrt'``.
        sgs_model: ``'none'``, ``'smagorinsky'``, ``'wale'``, or ``'vreman'``.
        config: Cross-validation configuration.

    Returns:
        Dict with keys: ``lattice``, ``collision``, ``sgs_model``,
        ``front_position``, ``mass_drift``, ``finite``.
    """
    config.validate()

    if lattice == "d3q19":
        return _run_d3q19(config, collision, sgs_model)
    elif lattice == "d3q27":
        return _run_d3q27(config, collision, sgs_model)
    else:
        raise ValueError(f"unknown lattice: {lattice!r}")


def run_dambreak_cross_validation(
    config: CrossValidationConfig | None = None,
    lattices: list[Lattice] | None = None,
    collisions: list[Collision] | None = None,
    sgs_models: list[SgsModel] | None = None,
) -> dict:
    """Run all combinations and return the comparison matrix.

    Produces a machine-readable artifact (JSON) at ``config.output_path``
    containing the full matrix of results.

    Args:
        config: Cross-validation configuration.  If ``None``, uses defaults.
        lattices: List of lattices to test (default: both).
        collisions: List of collision models to test (default: both).
        sgs_models: List of SGS models to test (default: all four).

    Returns:
        Dict with keys: ``status``, ``physical_validation``, ``config``,
        ``matrix``.  The ``matrix`` is a list of result dicts, one per
        combination.
    """
    if config is None:
        config = CrossValidationConfig()
    config.validate()

    if lattices is None:
        lattices = ["d3q19", "d3q27"]
    if collisions is None:
        collisions = ["bgk", "mrt"]
    if sgs_models is None:
        sgs_models = ["none", "smagorinsky", "wale", "vreman"]

    matrix: list[dict] = []
    for lattice in lattices:
        for collision in collisions:
            for sgs_model in sgs_models:
                result = run_single_dambreak(lattice, collision, sgs_model, config)
                matrix.append(result)

    artifact = {
        "status": "diagnostic_only",
        "physical_validation": False,
        "config": {
            "nx": config.nx,
            "ny": config.ny,
            "nz": config.nz,
            "dam_width": config.dam_width,
            "fill_height": config.fill_height,
            "n_steps": config.n_steps,
            "tau": config.tau,
            "gravity": config.gravity,
            "rho_liquid": config.rho_liquid,
            "rho_gas": config.rho_gas,
            "C_smag": config.C_smag,
            "C_wale": config.C_wale,
            "C_vreman": config.C_vreman,
            "device": config.device,
        },
        "matrix": matrix,
    }

    # Write machine-readable artifact
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return artifact
