"""Periodic, tensor-only static-droplet setup and guarded diagnostics.

This module is a diagnostic helper, not a pressure reconstruction or a Laplace
validation.  It uses the shared Cahn--Hilliard free-energy and operator path.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import torch

from .diagnostics import phase_volume_smoothed, phase_volume_threshold
from .free_energy import DoubleWellFreeEnergy, force_minus_phi_grad_mu


@dataclass(frozen=True)
class DropletGeometryDiagnostic:
    """Candidate geometry inferred from the ``phi > 0`` droplet convention."""

    center_zyx: tuple[float, float, float]
    threshold_volume: float
    smoothed_volume: float
    equivalent_radius: float
    candidate_mean_curvature: float


@dataclass(frozen=True)
class KortewegForceDiagnostic:
    """Inventory of ``-phi grad(mu)``, explicitly not a pressure measurement."""

    net_force: tuple[float, float, float]
    l2_norm: float
    max_norm: float
    chemical_potential_min: float
    chemical_potential_max: float


@dataclass(frozen=True)
class LaplaceStyleDiagnostic:
    """Pressure-jump comparison state; unavailable without a pressure field."""

    status: str
    observed_pressure_jump: float | None
    expected_pressure_jump: float | None
    reason: str


@dataclass(frozen=True)
class StaticDropletDiagnosticResult:
    """Diagnostic-only result that deliberately cannot grant physical acceptance."""

    status: str
    physical_acceptance: bool
    geometry: DropletGeometryDiagnostic
    force: KortewegForceDiagnostic
    laplace: LaplaceStyleDiagnostic


def _validate_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) != 3 or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape):
        raise ValueError("shape must contain three positive integer (z, y, x) sizes")
    return tuple(shape)  # type: ignore[return-value]


def _validate_phi(phi: torch.Tensor) -> None:
    if phi.ndim != 3:
        raise ValueError("static-droplet diagnostics require a 3-D scalar tensor shaped (z, y, x)")
    if not (phi.is_floating_point() or phi.is_complex()):
        raise ValueError("phi must have a floating-point dtype")
    if phi.is_complex():
        raise ValueError("phi must be real-valued")
    if not bool(torch.isfinite(phi).all().item()):
        raise ValueError("phi must contain only finite values")


def _periodic_delta(coordinate: torch.Tensor, center: float, size: int) -> torch.Tensor:
    """Shortest signed displacement on one periodic lattice axis."""
    return torch.remainder(coordinate - center + 0.5 * size, size) - 0.5 * size


def initialize_static_droplet(
    shape: Sequence[int],
    *,
    radius: float,
    interface_width: float | None = None,
    width: float | None = None,
    center: Sequence[float] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a periodic spherical tanh profile, shaped ``(z, y, x)``.

    The droplet is the ``phi > 0`` phase: ``phi=tanh((R-r)/(sqrt(2)*width))``.
    Distances are minimum-image distances, so centers and interfaces may cross a
    periodic domain face.  This function creates only a scalar tensor; it does
    not initialize D3Q19 populations or alter production operators.
    """
    shape_zyx = _validate_shape(shape)
    if interface_width is None:
        interface_width = width
    elif width is not None and width != interface_width:
        raise ValueError("width and interface_width must agree when both are supplied")
    if isinstance(radius, bool) or not isinstance(radius, Real) or not torch.isfinite(torch.tensor(radius)) or radius <= 0.0:
        raise ValueError("radius must be a finite positive number")
    if (
        isinstance(interface_width, bool)
        or not isinstance(interface_width, Real)
        or not torch.isfinite(torch.tensor(interface_width))
        or interface_width <= 0.0
    ):
        raise ValueError("interface_width must be a finite positive number")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("dtype must be a real floating-point dtype")
    if center is None:
        center_zyx = tuple((size - 1) * 0.5 for size in shape_zyx)
    else:
        if len(center) != 3 or any(isinstance(value, bool) or not isinstance(value, Real) or not torch.isfinite(torch.tensor(value)) for value in center):
            raise ValueError("center must contain three finite (z, y, x) coordinates")
        center_zyx = tuple(float(value) for value in center)

    axes = [torch.arange(size, dtype=dtype, device=device) for size in shape_zyx]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    distance_squared = (
        _periodic_delta(z, center_zyx[0], shape_zyx[0]) ** 2
        + _periodic_delta(y, center_zyx[1], shape_zyx[1]) ** 2
        + _periodic_delta(x, center_zyx[2], shape_zyx[2]) ** 2
    )
    return torch.tanh((float(radius) - torch.sqrt(distance_squared)) / (2.0**0.5 * float(interface_width)))


def estimate_droplet_radius(phi: torch.Tensor) -> torch.Tensor:
    """Return the threshold-volume equivalent radius for the ``phi > 0`` phase."""
    _validate_phi(phi)
    volume = phase_volume_threshold(phi)
    return torch.pow(3.0 * volume / (4.0 * torch.pi), 1.0 / 3.0)


def _periodic_center(phi: torch.Tensor) -> tuple[float, float, float]:
    mask = (phi > 0).to(dtype=phi.dtype)
    mass = mask.sum()
    if float(mass.item()) == 0.0:
        return (float("nan"), float("nan"), float("nan"))
    result: list[float] = []
    for axis, size in enumerate(phi.shape):
        weights = mask.sum(dim=tuple(index for index in range(3) if index != axis))
        coordinates = torch.arange(size, dtype=phi.dtype, device=phi.device)
        angles = 2.0 * torch.pi * coordinates / size
        angle = torch.atan2((weights * torch.sin(angles)).sum(), (weights * torch.cos(angles)).sum())
        result.append(float(torch.remainder(angle * size / (2.0 * torch.pi), size).item()))
    return tuple(result)  # type: ignore[return-value]


def periodic_chemical_potential_and_korteweg_force(
    phi: torch.Tensor, model: DoubleWellFreeEnergy
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Use shared periodic operators for ``mu`` and the ``-phi grad(mu)`` force."""
    _validate_phi(phi)
    mu = model.chemical_potential(phi, boundary="periodic")
    return mu, force_minus_phi_grad_mu(phi, mu, boundary="periodic")


def diagnose_static_droplet(phi: torch.Tensor, model: DoubleWellFreeEnergy) -> StaticDropletDiagnosticResult:
    """Report geometry and force inventory while withholding Laplace acceptance.

    No thermodynamic pressure field is available in this scalar CH helper.
    Chemical potential and Korteweg force are therefore never re-labelled as
    pressure, and no observed/expected pressure jump or Laplace PASS is made.
    """
    _validate_phi(phi)
    mu, force = periodic_chemical_potential_and_korteweg_force(phi, model)
    force_x, force_y, force_z = force
    radius = estimate_droplet_radius(phi)
    radius_value = float(radius.item())
    force_norm = torch.sqrt(force_x**2 + force_y**2 + force_z**2)
    geometry = DropletGeometryDiagnostic(
        center_zyx=_periodic_center(phi),
        threshold_volume=float(phase_volume_threshold(phi).item()),
        smoothed_volume=float(phase_volume_smoothed(phi).item()),
        equivalent_radius=radius_value,
        candidate_mean_curvature=2.0 / radius_value if radius_value > 0.0 else float("nan"),
    )
    force_diagnostic = KortewegForceDiagnostic(
        net_force=(float(force_x.sum().item()), float(force_y.sum().item()), float(force_z.sum().item())),
        l2_norm=float(torch.linalg.vector_norm(force_norm).item()),
        max_norm=float(force_norm.max().item()),
        chemical_potential_min=float(mu.min().item()),
        chemical_potential_max=float(mu.max().item()),
    )
    laplace = LaplaceStyleDiagnostic(
        status="withheld",
        observed_pressure_jump=None,
        expected_pressure_jump=None,
        reason=(
            "WITHHELD: no true thermodynamic pressure field was supplied or reconstructed; "
            "chemical potential (mu) and Korteweg force are not pressure and must not be used "
            "as an observed pressure jump or a Laplace PASS."
        ),
    )
    return StaticDropletDiagnosticResult(
        status="diagnostic_only", physical_acceptance=False, geometry=geometry, force=force_diagnostic, laplace=laplace
    )
