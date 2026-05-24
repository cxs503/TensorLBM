"""Scientific I/O helpers for TensorLBM simulation data.

Provides:
- :func:`save_vtk`        – Legacy ASCII VTK structured-points file.
- :func:`save_vtk_binary` – Binary little-endian VTK structured-points file.
- :func:`save_hdf5`       – HDF5 file via h5py for large-scale post-processing.

ASCII VTK export does not require any additional package beyond the standard
library; binary VTK uses the :mod:`struct` module (also stdlib).
HDF5 export requires ``h5py`` (``pip install h5py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _validate_field_shapes(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor | None = None,
    rho: torch.Tensor | None = None,
    vorticity: torch.Tensor | None = None,
) -> None:
    """Validate that all exported fields share a supported spatial shape."""
    if ux.ndim not in (2, 3):
        raise ValueError(f"ux must be a 2-D or 3-D tensor, got shape {tuple(ux.shape)}")
    if uy.shape != ux.shape:
        raise ValueError(f"ux and uy shapes must match: {tuple(ux.shape)} != {tuple(uy.shape)}")
    if uz is not None and uz.shape != ux.shape:
        raise ValueError(f"uz shape must match ux: {tuple(uz.shape)} != {tuple(ux.shape)}")
    if rho is not None and rho.shape != ux.shape:
        raise ValueError(f"rho shape must match ux: {tuple(rho.shape)} != {tuple(ux.shape)}")
    if vorticity is not None and vorticity.shape != ux.shape:
        raise ValueError(
            f"vorticity shape must match ux: {tuple(vorticity.shape)} != {tuple(ux.shape)}"
        )


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
    _validate_field_shapes(ux, uy, uz, rho, vorticity)
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
        for u, v, w in zip(vel_u, vel_v, vel_w, strict=True):
            lines.append(f"{u:.6g} {v:.6g} {w:.6g}")
    else:
        ux_f = _to_flat(ux)
        uy_f = _to_flat(uy)
        lines.append("VECTORS velocity float")
        for u, v in zip(ux_f, uy_f, strict=True):
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
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 output: pip install h5py"
        ) from exc

    path = Path(path)
    _validate_field_shapes(ux, uy, uz, rho)

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


def save_vtk_binary(
    path: str | Path,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor | None = None,
    rho: torch.Tensor | None = None,
    vorticity: torch.Tensor | None = None,
) -> Path:
    """Write a binary (little-endian) legacy VTK structured-points file.

    Produces smaller files and faster I/O than the ASCII :func:`save_vtk`
    variant, suitable for large 3-D grids.  All fields are written as
    ``float`` (32-bit IEEE 754, big-endian as required by the VTK legacy
    format).

    Args:
        path: Output file path (should end with ``.vtk``).
        ux: x-velocity field.
        uy: y-velocity field.
        uz: z-velocity field (3-D only; *None* for 2-D).
        rho: Density field (optional).
        vorticity: Vorticity scalar field (optional).

    Returns:
        Resolved output path.
    """
    import numpy as np

    path = Path(path)
    _validate_field_shapes(ux, uy, uz, rho, vorticity)
    is_3d = ux.ndim == 3

    if is_3d:
        nz, ny, nx = ux.shape
    else:
        ny, nx = ux.shape
        nz = 1

    n_points = nx * ny * nz

    def _to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().float().numpy().flatten()

    # VTK legacy format requires big-endian binary data
    header_lines = [
        "# vtk DataFile Version 3.0\n",
        "TensorLBM binary output\n",
        "BINARY\n",
        "DATASET STRUCTURED_POINTS\n",
        f"DIMENSIONS {nx} {ny} {nz}\n",
        "ORIGIN 0 0 0\n",
        "SPACING 1 1 1\n",
        f"POINT_DATA {n_points}\n",
    ]

    with path.open("wb") as fh:
        for line in header_lines:
            fh.write(line.encode("ascii"))

        if uz is not None:
            fh.write(b"VECTORS velocity float\n")
            vel_data = np.column_stack(
                [_to_np(ux), _to_np(uy), _to_np(uz)]
            ).flatten().astype(">f4")
        else:
            fh.write(b"VECTORS velocity float\n")
            zeros = np.zeros(n_points, dtype=np.float32)
            vel_data = np.column_stack(
                [_to_np(ux), _to_np(uy), zeros]
            ).flatten().astype(">f4")
        fh.write(vel_data.tobytes())
        fh.write(b"\n")

        if rho is not None:
            fh.write(b"SCALARS density float 1\n")
            fh.write(b"LOOKUP_TABLE default\n")
            fh.write(_to_np(rho).astype(">f4").tobytes())
            fh.write(b"\n")

        if vorticity is not None:
            fh.write(b"SCALARS vorticity float 1\n")
            fh.write(b"LOOKUP_TABLE default\n")
            fh.write(_to_np(vorticity).astype(">f4").tobytes())
            fh.write(b"\n")

    return path


def save_xdmf(
    h5_path: str | Path,
    xdmf_path: str | Path,
    step: int,
    ux_shape: tuple[int, ...],
    has_uz: bool = False,
    has_rho: bool = False,
) -> Path:
    """Write an XDMF metadata file for an existing TensorLBM HDF5 output.

    Args:
        h5_path: Path to the HDF5 file.
        xdmf_path: Output XDMF path.
        step: Simulation step number.
        ux_shape: Spatial shape of ``ux``.
        has_uz: Whether a ``uz`` dataset exists.
        has_rho: Whether a ``rho`` dataset exists.

    Returns:
        Resolved output XDMF path.
    """
    h5_path = Path(h5_path)
    xdmf_path = Path(xdmf_path)
    group = f"step_{step:06d}"
    h5_name = h5_path.name

    if len(ux_shape) == 3:
        nz, ny, nx = ux_shape
        topology_type = "3DCoRectMesh"
        dimensions = f"{nz} {ny} {nx}"
        geometry_type = "ORIGIN_DXDYDZ"
        geom_dims = 3
        vector_dim = 3 if has_uz else 2
        join_args = "$0, $1, $2" if has_uz else "$0, $1"
        velocity_items = [
            (
                f"          <DataItem Dimensions=\"{dimensions}\" NumberType=\"Float\" "
                f"Precision=\"4\" Format=\"HDF\">\n"
                f"            {h5_name}:/{group}/ux\n"
                f"          </DataItem>"
            ),
            (
                f"          <DataItem Dimensions=\"{dimensions}\" NumberType=\"Float\" "
                f"Precision=\"4\" Format=\"HDF\">\n"
                f"            {h5_name}:/{group}/uy\n"
                f"          </DataItem>"
            ),
        ]
        if has_uz:
            velocity_items.append(

                    f"          <DataItem Dimensions=\"{dimensions}\" NumberType=\"Float\" "
                    f"Precision=\"4\" Format=\"HDF\">\n"
                    f"            {h5_name}:/{group}/uz\n"
                    f"          </DataItem>"

            )
    elif len(ux_shape) == 2:
        ny, nx = ux_shape
        topology_type = "2DCoRectMesh"
        dimensions = f"{ny} {nx}"
        geometry_type = "ORIGIN_DXDY"
        geom_dims = 2
        vector_dim = 2
        join_args = "$0, $1"
        velocity_items = [
            (
                f"          <DataItem Dimensions=\"{dimensions}\" NumberType=\"Float\" "
                f"Precision=\"4\" Format=\"HDF\">\n"
                f"            {h5_name}:/{group}/ux\n"
                f"          </DataItem>"
            ),
            (
                f"          <DataItem Dimensions=\"{dimensions}\" NumberType=\"Float\" "
                f"Precision=\"4\" Format=\"HDF\">\n"
                f"            {h5_name}:/{group}/uy\n"
                f"          </DataItem>"
            ),
        ]
    else:
        raise ValueError(f"ux_shape must have length 2 or 3, got {ux_shape}")

    rho_block = ""
    if has_rho:
        rho_block = f'''
      <Attribute Name="rho" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{dimensions}" NumberType="Float" Precision="4" Format="HDF">
          {h5_name}:/{group}/rho
        </DataItem>
      </Attribute>'''

    zeros = " ".join(["0.0"] * geom_dims)
    ones = " ".join(["1.0"] * geom_dims)
    xml = f'''<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="{group}" GridType="Uniform">
      <Topology TopologyType="{topology_type}" Dimensions="{dimensions}"/>
      <Geometry GeometryType="{geometry_type}">
        <DataItem Format="XML" Dimensions="{geom_dims}">{zeros}</DataItem>
        <DataItem Format="XML" Dimensions="{geom_dims}">{ones}</DataItem>
      </Geometry>
      <Attribute Name="velocity" AttributeType="Vector" Center="Node">
        <DataItem
            ItemType="Function"
            Dimensions="{dimensions} {vector_dim}"
            Function="JOIN({join_args})"
        >
{chr(10).join(velocity_items)}
        </DataItem>
      </Attribute>{rho_block}
    </Grid>
  </Domain>
</Xdmf>
'''
    xdmf_path.write_text(xml, encoding="utf-8")
    return xdmf_path.resolve()


__all__ = ["save_vtk", "save_vtk_binary", "save_hdf5", "save_xdmf"]
