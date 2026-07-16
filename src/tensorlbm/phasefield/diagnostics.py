"""Phase-volume diagnostics with deliberately distinct interface conventions."""

from __future__ import annotations

import torch


def _validate_phi(phi: torch.Tensor) -> None:
    if phi.ndim != 3:
        raise ValueError("phase volume diagnostics require a 3-D scalar tensor shaped (z, y, x)")


def phase_volume_smoothed(phi: torch.Tensor) -> torch.Tensor:
    """Return the smooth ``phi=+1`` phase volume, ``sum((1 + phi) / 2)``."""
    _validate_phi(phi)
    return ((1.0 + phi) * 0.5).sum()


def phase_volume_threshold(phi: torch.Tensor, *, threshold: float = 0.0) -> torch.Tensor:
    """Return the cell-count volume using the explicit ``phi > threshold`` rule."""
    _validate_phi(phi)
    return (phi > threshold).to(dtype=phi.dtype).sum()
