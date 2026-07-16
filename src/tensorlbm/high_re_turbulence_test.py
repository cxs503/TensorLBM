"""High-Re cylinder flow turbulence model comparison.

Runs 2D cylinder flow at Re=1000 and Re=5000 with different SGS turbulence
models (none/Smagorinsky/WALE/Vreman/DynSmag) and collision operators
(BGK/MRT) to diagnose the effect of sub-grid-scale modeling on drag
coefficient (Cd) and Strouhal number at high Reynolds numbers.

**status=diagnostic_only** — this module produces machine-readable artifacts
for diagnostic comparison, not pass/fail validation against literature values.
The purpose is to identify which SGS models stabilise high-Re flows and
how they affect the integral force coefficients.

D2Q9 collision + turbulence dispatch table
-------------------------------------------
WALE, Vreman, and Dynamic-Smagorinsky only have BGK variants for D2Q9.
Smagorinsky has both BGK and MRT.  Plain BGK and MRT serve as the "none"
baseline.

================  ===========  ===========  ===========  ===========  ===========
collision         none         Smagorinsky  WALE         Vreman       DynSmag
================  ===========  ===========  ===========  ===========  ===========
BGK               ✓            ✓             ✓            ✓            ✓
MRT               ✓            ✓             —            —            —
================  ===========  ===========  ===========  ===========  ===========
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .boundaries import (
    apply_simple_channel_boundaries,
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
)
from .d2q9 import equilibrium
from .solver import collide_bgk, collide_mrt, stream
from .turbulence import (
    collide_dynamic_smagorinsky_bgk,
    collide_smagorinsky_bgk,
    collide_smagorinsky_mrt,
    collide_vreman_bgk,
    collide_wale_bgk,
)

# ---------------------------------------------------------------------------
# Dispatch table: (collision, turbulence_model) -> collision function
# ---------------------------------------------------------------------------

COLLISION_DISPATCH: dict[tuple[str, str], Callable[..., torch.Tensor]] = {
    ("bgk", "none"): collide_bgk,
    ("bgk", "smagorinsky"): collide_smagorinsky_bgk,
    ("bgk", "wale"): collide_wale_bgk,
    ("bgk", "vreman"): collide_vreman_bgk,
    ("bgk", "dynsmag"): collide_dynamic_smagorinsky_bgk,
    ("mrt", "none"): collide_mrt,
    ("mrt", "smagorinsky"): collide_smagorinsky_mrt,
}

REYNOLDS_LIST: tuple[int, ...] = (1000, 5000)
COLLISION_LIST: tuple[str, ...] = ("bgk", "mrt")
TURBULENCE_LIST: tuple[str, ...] = ("none", "smagorinsky", "wale", "vreman", "dynsmag")


# ---------------------------------------------------------------------------
# Strouhal number estimation
# ---------------------------------------------------------------------------

def _compute_strouhal(
    cl_series: list[float],
    diameter: float,
    u_in: float,
) -> float:
    """Estimate Strouhal number from lift-coefficient time series via FFT.

    The vortex-shedding frequency is identified as the dominant non-DC peak
    in the one-sided amplitude spectrum of the Cl signal.  The Strouhal
    number is then ``St = f * D / U``.

    If the signal contains NaN/Inf (diverged simulation) the result is NaN.
    If no clear peak is found the result is 0.0.
    """
    cl = np.asarray(cl_series, dtype=np.float64)
    n = len(cl)
    if n < 4:
        return 0.0
    if not np.all(np.isfinite(cl)):
        return float("nan")
    # Remove mean (DC component)
    cl_centered = cl - cl.mean()
    # One-sided FFT
    spectrum = np.fft.rfft(cl_centered)
    power = np.abs(spectrum) ** 2
    power[0] = 0.0  # exclude DC
    if power.max() < 1e-30:
        return 0.0
    k_peak = int(np.argmax(power))
    freq = k_peak / n  # cycles per lattice time-step
    return float(freq * diameter / u_in)


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

def run_high_re_cylinder(
    re: float,
    collision: str = "bgk",
    turbulence_model: str = "none",
    nx: int = 200,
    ny: int = 100,
    steps: int = 500,
    radius: float = 6.0,
    u_in: float = 0.06,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a single high-Re cylinder-flow simulation.

    Args:
        re: Reynolds number.
        collision: Collision operator (``"bgk"`` or ``"mrt"``).
        turbulence_model: SGS model name (``"none"``, ``"smagorinsky"``,
            ``"wale"``, ``"vreman"``, ``"dynsmag"``).
        nx: Grid size in x.
        ny: Grid size in y.
        steps: Number of LBM time steps.
        radius: Cylinder radius (lattice units).
        u_in: Inlet velocity (lattice units).
        device: Torch device string.

    Returns:
        Dict with keys: ``Re``, ``collision``, ``turbulence_model``,
        ``Cd``, ``Strouhal``, ``finite``.
    """
    key = (collision, turbulence_model)
    if key not in COLLISION_DISPATCH:
        msg = (
            f"Unsupported (collision, turbulence_model) combination: {key}. "
            f"Available: {sorted(COLLISION_DISPATCH.keys())}"
        )
        raise ValueError(msg)
    collide_fn = COLLISION_DISPATCH[key]

    diameter = 2.0 * radius
    nu = u_in * diameter / re
    tau = 3.0 * nu + 0.5

    dev = torch.device(device)
    mask = cylinder_mask(nx, ny, nx // 3, ny // 2, radius, device=dev)
    wall_mask = make_channel_wall_mask(ny, nx, mask, device=dev)

    rho0 = torch.ones(ny, nx, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0), device=dev)

    dyn_pressure = 0.5 * u_in ** 2 * diameter

    cd_series: list[float] = []
    cl_series: list[float] = []

    for _step in range(1, steps + 1):
        f = collide_fn(f, tau)
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, mask)
        f = apply_simple_channel_boundaries(
            f,
            u_in=u_in,
            wall_mask=wall_mask,
            obstacle_mask=torch.zeros_like(mask),
        )
        f = bounce_back_cells(f, mask)

        cd_series.append(float(fx.item()) / dyn_pressure)
        cl_series.append(float(fy.item()) / dyn_pressure)

    # Time-averaged Cd from the last 2/3 of the simulation (skip transient)
    avg_start = steps // 3
    cd_tail = cd_series[avg_start:]
    cd_mean = sum(cd_tail) / max(len(cd_tail), 1)

    # Strouhal from lift-coefficient time series
    st = _compute_strouhal(cl_series[avg_start:], diameter, u_in)

    finite = math.isfinite(cd_mean) and math.isfinite(st)

    return {
        "Re": int(re) if float(re).is_integer() else re,
        "collision": collision,
        "turbulence_model": turbulence_model,
        "Cd": float(cd_mean),
        "Strouhal": float(st),
        "finite": bool(finite),
    }


# ---------------------------------------------------------------------------
# Full matrix
# ---------------------------------------------------------------------------

def _sanitize_for_json(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace NaN/Inf floats with None for strict JSON (RFC 8259) compliance."""
    clean: list[dict[str, Any]] = []
    for r in results:
        row = dict(r)
        for k in ("Cd", "Strouhal"):
            v = row.get(k)
            if isinstance(v, float) and not math.isfinite(v):
                row[k] = None
        clean.append(row)
    return clean


def run_high_re_turbulence_matrix(
    nx: int = 200,
    ny: int = 100,
    steps: int = 500,
    radius: float = 6.0,
    u_in: float = 0.06,
    device: str = "cpu",
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run the full high-Re turbulence-model comparison matrix.

    Iterates over Re ∈ {1000, 5000} × collision ∈ {bgk, mrt} ×
    turbulence_model ∈ {none, smagorinsky, wale, vreman, dynsmag},
    skipping combinations not in :data:`COLLISION_DISPATCH`.

    Total: 7 combinations × 2 Reynolds = 14 simulations.

    Args:
        nx: Grid size in x.
        ny: Grid size in y.
        steps: Number of LBM time steps.
        radius: Cylinder radius.
        u_in: Inlet velocity.
        device: Torch device string.
        output_path: Optional path to save a JSON artifact.

    Returns:
        List of result dicts (see :func:`run_high_re_cylinder`).
    """
    results: list[dict[str, Any]] = []

    for re in REYNOLDS_LIST:
        for collision in COLLISION_LIST:
            for turbulence in TURBULENCE_LIST:
                key = (collision, turbulence)
                if key not in COLLISION_DISPATCH:
                    continue
                result = run_high_re_cylinder(
                    re=re,
                    collision=collision,
                    turbulence_model=turbulence,
                    nx=nx,
                    ny=ny,
                    steps=steps,
                    radius=radius,
                    u_in=u_in,
                    device=device,
                )
                results.append(result)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Replace NaN/Inf with null for strict JSON compliance (RFC 8259).
        # The ``finite`` flag already records whether values are usable.
        clean = _sanitize_for_json(results)
        with open(p, "w") as fh:
            json.dump(clean, fh, indent=2, allow_nan=False)

    return results


__all__ = [
    "COLLISION_DISPATCH",
    "REYNOLDS_LIST",
    "COLLISION_LIST",
    "TURBULENCE_LIST",
    "run_high_re_cylinder",
    "run_high_re_turbulence_matrix",
]
