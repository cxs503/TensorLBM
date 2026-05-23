"""Scientific I/O helpers for TensorLBM simulation data.

Provides:
- :func:`save_vtk`  – Legacy ASCII VTK rectilinear grid for ParaView / VisIt.
- :func:`save_hdf5` – HDF5 file via h5py for large-scale post-processing.

VTK export does not require any additional package beyond the standard library;
HDF5 export requires ``h5py`` (``pip install h5py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def save_vtk(
    path: str | Path,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor | None = None,
    rho: torch.Tensor | None = None,
    vorticity: torch.Tensor | None = None,
) -> Path:
    """Write a legacy ASCII VTK structured-points file.

    Supports 2-D (ny × nx) and 3-D (nz × ny × nx) fields. All tensors must
    share the same spatial shape. The file can be opened directly in
    ParaView, VisIt, or any VTK-aware tool.

    Args:
        path: Output file path (should end with ``.vtk``).
        ux: x-velocity field.
        uy: y-velocity field.
        uz: z-velocity field (3-D only; *None* for 2-D).
        rho: Density field (optional).
        vorticity: Vorticity scalar field (optional; e.g. z-component).

    Returns:
        Resolved output path.
    """
    path = Path(path)
    is_3d = ux.ndim == 3

    if is_3d:
        nz, ny, nx = ux.shape
    else:
        ny, nx = ux.shape
        nz = 1

    n_points = nx * ny * nz

    def _to_flat(t: torch.Tensor) -> list[float]:
        return t.detach().cpu().float().reshape(-1).tolist()

    lines: list[str] = [
        "# vtk DataFile Version 3.0",
        "TensorLBM output",
        "ASCII",
        "DATASET STRUCTURED_POINTS",
        f"DIMENSIONS {nx} {ny} {nz}",
        "ORIGIN 0 0 0",
        "SPACING 1 1 1",
        f"POINT_DATA {n_points}",
    ]

    if uz is not None:
        vel_u = _to_flat(ux)
        vel_v = _to_flat(uy)
        vel_w = _to_flat(uz)
        lines.append("VECTORS velocity float")
        for u, v, w in zip(vel_u, vel_v, vel_w, strict=False):
            lines.append(f"{u:.6g} {v:.6g} {w:.6g}")
    else:
        ux_f = _to_flat(ux)
        uy_f = _to_flat(uy)
        lines.append("VECTORS velocity float")
        for u, v in zip(ux_f, uy_f, strict=False):
            lines.append(f"{u:.6g} {v:.6g} 0.0")

    if rho is not None:
        lines.append("SCALARS density float 1")
        lines.append("LOOKUP_TABLE default")
        for val in _to_flat(rho):
            lines.append(f"{val:.6g}")

    if vorticity is not None:
        lines.append("SCALARS vorticity float 1")
        lines.append("LOOKUP_TABLE default")
        for val in _to_flat(vorticity):
            lines.append(f"{val:.6g}")

    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def save_hdf5(
    path: str | Path,
    step: int,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor | None = None,
    rho: torch.Tensor | None = None,
) -> Path:
    """Write simulation fields to an HDF5 file.

    Requires ``h5py`` (``pip install h5py``). If h5py is not installed a
    clear :class:`ImportError` is raised.

    Each call writes (or overwrites) a group ``/step_{step:06d}`` containing
    datasets ``ux``, ``uy`` (and optionally ``uz``, ``rho``).

    Args:
        path: Output ``.h5`` file path.
        step: Simulation step number (used as group name).
        ux: x-velocity field.
        uy: y-velocity field.
        uz: z-velocity field (optional).
        rho: Density field (optional).

    Returns:
        Resolved output path.
    """
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 output: pip install h5py"
        ) from exc

    path = Path(path)

    def _np(t: torch.Tensor) -> object:
        return t.detach().cpu().float().numpy()

    with h5py.File(path, "a") as fh:
        grp_name = f"step_{step:06d}"
        if grp_name in fh:
            del fh[grp_name]
        grp = fh.create_group(grp_name)
        grp.create_dataset("ux", data=_np(ux), compression="gzip")
        grp.create_dataset("uy", data=_np(uy), compression="gzip")
        if uz is not None:
            grp.create_dataset("uz", data=_np(uz), compression="gzip")
        if rho is not None:
            grp.create_dataset("rho", data=_np(rho), compression="gzip")
        grp.attrs["step"] = step

    return path


__all__ = ["save_vtk", "save_hdf5"]
