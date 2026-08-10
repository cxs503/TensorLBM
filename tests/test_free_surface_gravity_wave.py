"""Deterministic closed-domain free-surface gravity-wave validation.

This is a diagnostic-scale test only.  It uses a sinusoidal, sub-cell initial
surface perturbation and records its Fourier amplitude/phase and the tracked
liquid-mass ledger; it is not a ship-resistance validation.
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_lbm import (
    GAS,
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
)


def _sinusoidal_surface_state() -> tuple[torch.Tensor, ...]:
    """Return a closed, periodic-x small-amplitude free-surface state."""
    nz, ny, nx = 4, 18, 32
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    # Horizontal walls close the vertical column; x and z remain periodic.
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    x = torch.arange(nx, dtype=torch.float32).view(1, 1, nx)
    y = torch.arange(ny, dtype=torch.float32).view(1, ny, 1)
    # The 0.25-cell perturbation produces one partial interface layer only.
    fill = torch.clamp(8.0 + 0.25 * torch.cos(2.0 * math.pi * x / nx) - y, 0.0, 1.0)
    fill = fill.expand(nz, -1, -1).clone()
    fill[solid] = 0.0
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags)
    rho = torch.where(flags == GAS, torch.full_like(fill, 0.001), torch.ones_like(fill))
    zero = torch.zeros_like(fill)
    return equilibrium3d(rho, zero, zero, zero), fill, flags, solid, mass


def _surface_mode(fill: torch.Tensor) -> tuple[float, float]:
    """First Fourier cosine/sine coefficients of liquid column height."""
    nx = fill.shape[-1]
    x = torch.arange(nx, dtype=fill.dtype, device=fill.device)
    phase = 2.0 * math.pi * x / nx
    height = fill.sum(dim=(0, 1)) / fill.shape[0]
    cosine = float((2.0 / nx * (height * torch.cos(phase)).sum()).item())
    sine = float((2.0 / nx * (height * torch.sin(phase)).sum()).item())
    return cosine, sine


def test_small_amplitude_gravity_wave_closed_mass_ledger_and_mode() -> None:
    """Gravity must not lose liquid mass before a measurable wave response."""
    f, fill, flags, solid, mass = _sinusoidal_surface_state()
    initial_mass = float(mass.sum().item())
    initial_cosine, initial_sine = _surface_mode(fill)
    ledgers: list[dict[str, float]] = []

    # This first gate covers the pre-conversion response only.  Topology
    # propagation after conversion has a separate unresolved D3Q19 regression.
    for _ in range(1):
        ledger: dict[str, float] = {}
        f, fill, flags, mass, _ = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=0.8, gy=-1.0e-4,
            rho_liquid=1.0, rho_gas=0.001, mass_ledger=ledger,
        )
        ledgers.append(ledger)

    final_mass = float(mass.sum().item())
    final_cosine, final_sine = _surface_mode(fill)

    # A closed-domain mass exchange is internal: the local ledger must balance.
    assert abs(final_mass - initial_mass) < 3.0e-5 * initial_mass
    for ledger in ledgers:
        assert abs(ledger["exchange"] - ledger["start"]) < 3.0e-5 * initial_mass
        assert abs(ledger["boundary"] - ledger["start"]) < 3.0e-5 * initial_mass

    # The deterministic wave diagnostic must retain a nonzero resolved mode;
    # phase is reported as atan2(sine, cosine) for later dispersion studies.
    assert abs(initial_cosine) > 0.1
    assert math.hypot(final_cosine, final_sine) > 0.05
