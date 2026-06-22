"""VTK export utilities for TensorLBM simulation results.

Exports LBM field data to VTK ImageData (`.vti`) format, which can be
loaded directly into ParaView, VisIt, or any VTK-capable post-processor.

Both 2-D (D2Q9) and 3-D (D3Q19/D3Q27) checkpoints are supported.

Supported field outputs
-----------------------
2-D fields (``ny × nx`` grids):
    * ``velocity_magnitude`` – |u|
    * ``vorticity``          – ∂u_y/∂x − ∂u_x/∂y
    * ``density``            – ρ
    * ``pressure``           – (ρ − 1) / 3  (lattice units)
    * ``ux``, ``uy``        – velocity components

3-D fields (``nz × ny × nx`` grids):
    * ``velocity_magnitude`` – |u|
    * ``density``            – ρ
    * ``pressure``           – (ρ − 1) / 3
    * ``ux``, ``uy``, ``uz`` – velocity components
    * ``q_criterion``        – second invariant of ∇u (vortex identification)

Output format
-------------
ASCII VTK Legacy format (``.vti``) — compatible with all VTK/ParaView
versions without requiring the ``vtk`` Python package.

References
----------
VTK File Formats: https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
Kitware (2006) "The VTK User's Guide", 11th ed.
"""
from __future__ import annotations

from pathlib import Path

import torch

__all__ = [
    "export_vtk_2d",
    "export_vtk_3d",
    "export_checkpoint_vtk",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pressure_from_rho(rho: torch.Tensor) -> torch.Tensor:
    """Lattice pressure p = (ρ − 1) / 3 (incompressible LBM, cs² = 1/3)."""
    return (rho - 1.0) / 3.0


def _vorticity_2d(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """2-D vorticity ω_z = ∂u_y/∂x − ∂u_x/∂y (second-order central diff)."""
    duy_dx = torch.zeros_like(uy)
    dux_dy = torch.zeros_like(ux)

    duy_dx[:, 1:-1] = (uy[:, 2:] - uy[:, :-2]) * 0.5
    duy_dx[:, 0] = uy[:, 1] - uy[:, 0]
    duy_dx[:, -1] = uy[:, -1] - uy[:, -2]

    dux_dy[1:-1, :] = (ux[2:, :] - ux[:-2, :]) * 0.5
    dux_dy[0, :] = ux[1, :] - ux[0, :]
    dux_dy[-1, :] = ux[-1, :] - ux[-2, :]

    return duy_dx - dux_dy


def _q_criterion_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Q-criterion Q = ½(‖Ω‖² − ‖S‖²) for vortex-core identification.

    Args:
        ux, uy, uz: Velocity components, each of shape ``(nz, ny, nx)``.

    Returns:
        Q field of shape ``(nz, ny, nx)``.
    """
    def _grad(v: torch.Tensor, dim: int) -> torch.Tensor:
        g = torch.zeros_like(v)
        slc_f = [slice(None)] * v.ndim
        slc_b = [slice(None)] * v.ndim
        slc_c_f = [slice(None)] * v.ndim
        slc_c_b = [slice(None)] * v.ndim
        n = v.shape[dim]

        slc_c_f[dim] = slice(1, n - 1)
        slc_c_b[dim] = slice(1, n - 1)
        slc_f[dim] = slice(2, n)
        slc_b[dim] = slice(0, n - 2)
        g[tuple(slc_c_f)] = (v[tuple(slc_f)] - v[tuple(slc_b)]) * 0.5

        # Forward diff at start
        sf = [slice(None)] * v.ndim
        sb = [slice(None)] * v.ndim
        sf[dim] = 0
        sb[dim] = 1
        g[tuple(sf)] = v[tuple(sb)] - v[tuple(sf)]

        # Backward diff at end
        sf2 = [slice(None)] * v.ndim
        sb2 = [slice(None)] * v.ndim
        sf2[dim] = -1
        sb2[dim] = -2
        g[tuple(sf2)] = v[tuple(sf2)] - v[tuple(sb2)]

        return g

    dux_dx = _grad(ux, 2)
    dux_dy = _grad(ux, 1)
    dux_dz = _grad(ux, 0)
    duy_dx = _grad(uy, 2)
    duy_dy = _grad(uy, 1)
    duy_dz = _grad(uy, 0)
    duz_dx = _grad(uz, 2)
    duz_dy = _grad(uz, 1)
    duz_dz = _grad(uz, 0)

    # Symmetric strain S and antisymmetric rotation Ω
    S11, S22, S33 = dux_dx, duy_dy, duz_dz
    S12 = 0.5 * (dux_dy + duy_dx)
    S13 = 0.5 * (dux_dz + duz_dx)
    S23 = 0.5 * (duy_dz + duz_dy)

    O12 = 0.5 * (dux_dy - duy_dx)
    O13 = 0.5 * (dux_dz - duz_dx)
    O23 = 0.5 * (duy_dz - duz_dy)

    S_sq = S11**2 + S22**2 + S33**2 + 2.0 * (S12**2 + S13**2 + S23**2)
    O_sq = 2.0 * (O12**2 + O13**2 + O23**2)

    return 0.5 * (O_sq - S_sq)


# ---------------------------------------------------------------------------
# ASCII VTK Legacy helpers
# ---------------------------------------------------------------------------

def _write_vtk_header_2d(
    nx: int,
    ny: int,
    spacing: float,
) -> list[str]:
    """Return header lines for a 2-D VTK structured-points file."""
    lines = [
        "# vtk DataFile Version 3.0",
        "TensorLBM 2D field export",
        "ASCII",
        "DATASET STRUCTURED_POINTS",
        f"DIMENSIONS {nx} {ny} 1",
        "ORIGIN 0.0 0.0 0.0",
        f"SPACING {spacing} {spacing} {spacing}",
        f"POINT_DATA {nx * ny}",
    ]
    return lines


def _write_vtk_header_3d(
    nx: int,
    ny: int,
    nz: int,
    spacing: float,
) -> list[str]:
    """Return header lines for a 3-D VTK structured-points file."""
    lines = [
        "# vtk DataFile Version 3.0",
        "TensorLBM 3D field export",
        "ASCII",
        "DATASET STRUCTURED_POINTS",
        f"DIMENSIONS {nx} {ny} {nz}",
        "ORIGIN 0.0 0.0 0.0",
        f"SPACING {spacing} {spacing} {spacing}",
        f"POINT_DATA {nx * ny * nz}",
    ]
    return lines


def _scalar_section(name: str, data: torch.Tensor) -> list[str]:
    """Flatten *data* to a scalar section in VTK ASCII format."""
    lines = [f"SCALARS {name} float 1", "LOOKUP_TABLE default"]
    flat = data.cpu().float().flatten().tolist()
    lines += [f"{v:.6e}" for v in flat]
    return lines


def _vector_section(
    name: str,
    vx: torch.Tensor,
    vy: torch.Tensor,
    vz: torch.Tensor | None = None,
) -> list[str]:
    """Interleave three components into a VTK VECTORS section."""
    lines = [f"VECTORS {name} float"]
    vx_f = vx.cpu().float().flatten().tolist()
    vy_f = vy.cpu().float().flatten().tolist()
    if vz is not None:
        vz_f = vz.cpu().float().flatten().tolist()
        lines += [f"{x:.6e} {y:.6e} {z:.6e}" for x, y, z in zip(vx_f, vy_f, vz_f, strict=True)]
    else:
        lines += [f"{x:.6e} {y:.6e} 0.0" for x, y in zip(vx_f, vy_f, strict=True)]
    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_vtk_2d(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    output_path: str | Path,
    spacing: float = 1.0,
    fields: list[str] | None = None,
) -> Path:
    """Export a 2-D LBM macroscopic field to a VTK legacy file.

    Args:
        rho:  Density field, shape ``(ny, nx)``.
        ux:   x-velocity field, shape ``(ny, nx)``.
        uy:   y-velocity field, shape ``(ny, nx)``.
        output_path: Destination ``.vtk`` file path (created/overwritten).
        spacing: Physical grid spacing (default 1.0, lattice units).
        fields: List of field names to export.  If ``None`` all fields are
            written.  Choices: ``density``, ``pressure``, ``velocity_magnitude``,
            ``vorticity``, ``velocity``.

    Returns:
        Resolved ``Path`` of the written file.
    """
    ny, nx = rho.shape
    _all = {"density", "pressure", "velocity_magnitude", "vorticity", "velocity"}
    selected = _all if fields is None else set(fields) & _all

    lines = _write_vtk_header_2d(nx, ny, spacing)

    if "density" in selected:
        lines += _scalar_section("density", rho)
    if "pressure" in selected:
        lines += _scalar_section("pressure", _pressure_from_rho(rho))
    if "velocity_magnitude" in selected:
        vel_mag = torch.sqrt(ux * ux + uy * uy)
        lines += _scalar_section("velocity_magnitude", vel_mag)
    if "vorticity" in selected:
        vort = _vorticity_2d(ux, uy)
        lines += _scalar_section("vorticity", vort)
    if "velocity" in selected:
        lines += _vector_section("velocity", ux, uy)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out.resolve()


def export_vtk_3d(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    output_path: str | Path,
    spacing: float = 1.0,
    fields: list[str] | None = None,
) -> Path:
    """Export a 3-D LBM macroscopic field to a VTK legacy file.

    Args:
        rho:  Density field, shape ``(nz, ny, nx)``.
        ux:   x-velocity, shape ``(nz, ny, nx)``.
        uy:   y-velocity, shape ``(nz, ny, nx)``.
        uz:   z-velocity, shape ``(nz, ny, nx)``.
        output_path: Destination ``.vtk`` file path.
        spacing: Physical grid spacing (default 1.0, lattice units).
        fields: Fields to export.  If ``None``, all are written.  Choices:
            ``density``, ``pressure``, ``velocity_magnitude``, ``q_criterion``,
            ``velocity``.

    Returns:
        Resolved ``Path`` of the written file.
    """
    nz, ny, nx = rho.shape
    _all = {"density", "pressure", "velocity_magnitude", "q_criterion", "velocity"}
    selected = _all if fields is None else set(fields) & _all

    lines = _write_vtk_header_3d(nx, ny, nz, spacing)

    if "density" in selected:
        lines += _scalar_section("density", rho)
    if "pressure" in selected:
        lines += _scalar_section("pressure", _pressure_from_rho(rho))
    if "velocity_magnitude" in selected:
        vel_mag = torch.sqrt(ux * ux + uy * uy + uz * uz)
        lines += _scalar_section("velocity_magnitude", vel_mag)
    if "q_criterion" in selected:
        q = _q_criterion_3d(ux, uy, uz)
        lines += _scalar_section("q_criterion", q)
    if "velocity" in selected:
        lines += _vector_section("velocity", ux, uy, uz)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out.resolve()


def export_checkpoint_vtk(
    checkpoint_dir: str | Path,
    output_path: str | Path | None = None,
    spacing: float = 1.0,
    fields: list[str] | None = None,
) -> Path:
    """Load a TensorLBM checkpoint and export it to VTK format.

    Detects whether the checkpoint is 2-D (D2Q9, shape ``(9, ny, nx)``) or
    3-D (D3Q19/D3Q27, shape ``(19|27, nz, ny, nx)``) and calls the
    appropriate export function.

    Args:
        checkpoint_dir: Directory containing ``checkpoint_f.pt``.
        output_path: Output VTK path.  Defaults to
            ``<checkpoint_dir>/field.vtk``.
        spacing: Physical grid spacing.
        fields: Subset of fields to include; see :func:`export_vtk_2d` /
            :func:`export_vtk_3d` for valid names.

    Returns:
        Path to the written VTK file.
    """
    from .checkpoint import load_checkpoint

    ckpt_dir = Path(checkpoint_dir)
    f, step, _meta = load_checkpoint(ckpt_dir)

    out = ckpt_dir / f"field_step{step:06d}.vtk" if output_path is None else Path(output_path)

    if f.ndim == 3:
        # 2-D: f shape (Q, ny, nx)
        from .d2q9 import macroscopic
        rho, ux, uy = macroscopic(f)
        return export_vtk_2d(rho, ux, uy, out, spacing=spacing, fields=fields)

    if f.ndim == 4:
        # 3-D: f shape (Q, nz, ny, nx)
        q = f.shape[0]
        if q == 19:
            from .d3q19 import macroscopic3d
            rho, ux, uy, uz = macroscopic3d(f)
        elif q == 27:
            from .d3q27 import macroscopic27
            rho, ux, uy, uz = macroscopic27(f)
        else:
            raise ValueError(f"Unsupported velocity set size Q={q}")
        return export_vtk_3d(rho, ux, uy, uz, out, spacing=spacing, fields=fields)

    raise ValueError(
        f"Unexpected distribution tensor shape {tuple(f.shape)}; "
        "expected (Q, ny, nx) for 2-D or (Q, nz, ny, nx) for 3-D."
    )
