"""3-D ship CAD mesh utilities for platform CAD workflow.

MVP scope:
- Parametric hull mesh generation (Wigley / Series60 / KCS)
- STL import/export
- glTF export for web visualization
- Optional STEP import/export when CadQuery is available
"""
from __future__ import annotations

import base64
import json
import struct
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from .ship_cad import ShipHullType


class CADGeometryEngine(StrEnum):
    """Geometry backends exposed by the CAD service."""

    NATIVE = "native"
    CADQUERY = "cadquery"


@dataclass(slots=True)
class TriangleMesh:
    """Simple triangle mesh container."""

    vertices: np.ndarray
    faces: np.ndarray
    units: str = "lu"
    name: str = "hull"
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate shape/dtype constraints."""
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")
        if self.faces.size and int(self.faces.max()) >= int(self.vertices.shape[0]):
            raise ValueError("face indices out of range")

    def stats(self) -> dict[str, object]:
        """Return mesh statistics useful for CAD/API payloads."""
        self.validate()
        mins = self.vertices.min(axis=0)
        maxs = self.vertices.max(axis=0)
        size = maxs - mins
        return {
            "vertex_count": int(self.vertices.shape[0]),
            "face_count": int(self.faces.shape[0]),
            "bounds_min": [float(v) for v in mins],
            "bounds_max": [float(v) for v in maxs],
            "size": [float(v) for v in size],
            "units": self.units,
            "name": self.name,
        }


def _half_beam(hull_type: ShipHullType, xi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    in_hull = (np.abs(xi) <= 1.0) & (zeta >= 0.0) & (zeta <= 1.0)
    xi_c = np.clip(xi, -1.0, 1.0)
    zeta_c = np.clip(zeta, 0.0, 1.0)
    if hull_type == ShipHullType.WIGLEY:
        hb = (1.0 - xi_c**2) * (1.0 - (1.0 - zeta_c) ** 2)
    elif hull_type == ShipHullType.SERIES60:
        hb = (1.0 - xi_c**2) ** 0.51 * zeta_c**0.30
    else:
        hb = (1.0 - xi_c**2) ** 0.45 * zeta_c**0.24
    return np.where(in_hull, np.clip(hb, 0.0, 1.0), 0.0)


def create_parametric_hull_mesh(
    hull_type: ShipHullType | str,
    *,
    length: float,
    beam: float,
    draft: float,
    n_long: int = 80,
    n_vert: int = 40,
    units: str = "lu",
    name: str | None = None,
) -> TriangleMesh:
    """Create a watertight-ish hull mesh from the internal parametric profile."""
    if isinstance(hull_type, str):
        hull_type = ShipHullType(hull_type)

    n_long = max(int(n_long), 4)
    n_vert = max(int(n_vert), 4)

    xi_arr = np.linspace(-1.0, 1.0, n_long)
    z_arr = np.linspace(0.0, 1.0, n_vert)
    xi_grid, zeta_grid = np.meshgrid(xi_arr, z_arr, indexing="ij")
    hb = _half_beam(hull_type, xi_grid, zeta_grid)

    x = ((xi_grid + 1.0) * 0.5 * length).astype(np.float32)
    y = (hb * (beam * 0.5)).astype(np.float32)
    z = (zeta_grid * draft).astype(np.float32)

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def _add_vertex(v: tuple[float, float, float]) -> int:
        verts.append(v)
        return len(verts) - 1

    idx_sb = np.empty((n_long, n_vert), dtype=np.int32)
    idx_pt = np.empty((n_long, n_vert), dtype=np.int32)

    for i in range(n_long):
        for k in range(n_vert):
            idx_sb[i, k] = _add_vertex((float(x[i, k]), float(y[i, k]), float(z[i, k])))
            idx_pt[i, k] = _add_vertex((float(x[i, k]), float(-y[i, k]), float(z[i, k])))

    for i in range(n_long - 1):
        for k in range(n_vert - 1):
            a = int(idx_sb[i, k])
            b = int(idx_sb[i + 1, k])
            c = int(idx_sb[i + 1, k + 1])
            d = int(idx_sb[i, k + 1])
            faces.append((a, b, c))
            faces.append((a, c, d))

            a2 = int(idx_pt[i, k])
            b2 = int(idx_pt[i + 1, k])
            c2 = int(idx_pt[i + 1, k + 1])
            d2 = int(idx_pt[i, k + 1])
            faces.append((a2, c2, b2))
            faces.append((a2, d2, c2))

    # deck closure at z=draft
    for i in range(n_long - 1):
        p0 = int(idx_sb[i, n_vert - 1])
        p1 = int(idx_sb[i + 1, n_vert - 1])
        p2 = int(idx_pt[i + 1, n_vert - 1])
        p3 = int(idx_pt[i, n_vert - 1])
        faces.append((p0, p1, p2))
        faces.append((p0, p2, p3))

    # bow/transom closure
    for j in (0, n_long - 1):
        for k in range(n_vert - 1):
            s0 = int(idx_sb[j, k])
            s1 = int(idx_sb[j, k + 1])
            p1 = int(idx_pt[j, k + 1])
            p0 = int(idx_pt[j, k])
            if j == 0:
                faces.append((s0, p1, s1))
                faces.append((s0, p0, p1))
            else:
                faces.append((s0, s1, p1))
                faces.append((s0, p1, p0))

    mesh = TriangleMesh(
        vertices=np.asarray(verts, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        units=units,
        name=name or f"{hull_type.value}_hull",
        metadata={"hull_type": hull_type.value, "engine": CADGeometryEngine.NATIVE.value},
    )
    mesh.validate()
    return mesh


def _tri_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    n = np.cross(v1 - v0, v2 - v0)
    mag = float(np.linalg.norm(n))
    if mag < 1e-12:
        return np.zeros(3, dtype=np.float32)
    return (n / mag).astype(np.float32)


def export_mesh_stl_ascii(mesh: TriangleMesh, output_path: str | Path) -> Path:
    """Export mesh as ASCII STL."""
    mesh.validate()
    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [f"solid {mesh.name}"]
    for f0, f1, f2 in mesh.faces:
        v0 = mesh.vertices[int(f0)]
        v1 = mesh.vertices[int(f1)]
        v2 = mesh.vertices[int(f2)]
        n = _tri_normal(v0, v1, v2)
        lines.append(f"  facet normal {n[0]:.7e} {n[1]:.7e} {n[2]:.7e}")
        lines.append("    outer loop")
        lines.append(f"      vertex {v0[0]:.7e} {v0[1]:.7e} {v0[2]:.7e}")
        lines.append(f"      vertex {v1[0]:.7e} {v1[1]:.7e} {v1[2]:.7e}")
        lines.append(f"      vertex {v2[0]:.7e} {v2[1]:.7e} {v2[2]:.7e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {mesh.name}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def export_mesh_gltf(mesh: TriangleMesh, output_path: str | Path) -> Path:
    """Export mesh as minimal embedded glTF (.gltf JSON + base64 buffer)."""
    mesh.validate()
    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    verts = mesh.vertices.astype(np.float32)
    idx = mesh.faces.astype(np.uint32).ravel()
    bin_blob = verts.tobytes(order="C") + idx.tobytes(order="C")
    uri = "data:application/octet-stream;base64," + base64.b64encode(bin_blob).decode("ascii")

    verts_len = verts.nbytes
    idx_len = idx.nbytes
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)

    gltf: dict[str, object] = {
        "asset": {"version": "2.0", "generator": "tensorlbm.ship_cad3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": mesh.name}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(bin_blob), "uri": uri}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": verts_len, "target": 34962},
            {
                "buffer": 0,
                "byteOffset": verts_len,
                "byteLength": idx_len,
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": int(verts.shape[0]),
                "type": "VEC3",
                "min": [float(v) for v in mins],
                "max": [float(v) for v in maxs],
            },
            {
                "bufferView": 1,
                "componentType": 5125,
                "count": int(idx.size),
                "type": "SCALAR",
                "min": [int(idx.min()) if idx.size else 0],
                "max": [int(idx.max()) if idx.size else 0],
            },
        ],
    }
    p.write_text(json.dumps(gltf, indent=2), encoding="utf-8")
    return p


def import_mesh_stl(stl_path: str | Path, *, units: str = "lu") -> TriangleMesh:
    """Import ASCII/Binary STL into TriangleMesh."""
    p = Path(stl_path).resolve()
    blob = p.read_bytes()

    # Binary STL heuristic
    if len(blob) >= 84:
        tri_count = struct.unpack("<I", blob[80:84])[0]
        expected = 84 + tri_count * 50
        if expected == len(blob):
            verts: list[tuple[float, float, float]] = []
            faces: list[tuple[int, int, int]] = []
            for i in range(tri_count):
                base = 84 + i * 50
                coords = struct.unpack("<12f", blob[base : base + 48])[3:]
                tri_idx: list[int] = []
                for j in range(0, 9, 3):
                    verts.append((coords[j], coords[j + 1], coords[j + 2]))
                    tri_idx.append(len(verts) - 1)
                faces.append((tri_idx[0], tri_idx[1], tri_idx[2]))
            mesh = TriangleMesh(
                vertices=np.asarray(verts, dtype=np.float32),
                faces=np.asarray(faces, dtype=np.int32),
                units=units,
                name=p.stem,
                metadata={"source": "stl", "engine": CADGeometryEngine.NATIVE.value},
            )
            mesh.validate()
            return mesh

    # Fallback ASCII parser
    text = blob.decode("utf-8", errors="ignore")
    verts = []
    faces = []
    for line in text.splitlines():
        s = line.strip().lower()
        if s.startswith("vertex"):
            parts = line.strip().split()
            if len(parts) >= 4:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(verts) % 3 == 0:
                    n = len(verts)
                    faces.append((n - 3, n - 2, n - 1))
    if not verts or not faces:
        raise ValueError("failed to parse STL mesh")
    mesh = TriangleMesh(
        vertices=np.asarray(verts, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        units=units,
        name=p.stem,
        metadata={"source": "stl", "engine": CADGeometryEngine.NATIVE.value},
    )
    mesh.validate()
    return mesh


def export_mesh_step(mesh: TriangleMesh, output_path: str | Path) -> Path:
    """Export mesh to STEP via CadQuery when available."""
    try:
        import cadquery as cq
        from cadquery import exporters
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("STEP export requires cadquery/openCascade") from exc

    mesh.validate()
    pts = [tuple(float(v) for v in row) for row in mesh.vertices]
    faces = [tuple(int(i) for i in row) for row in mesh.faces]
    face_objs = []
    for f in faces:
        poly = cq.Wire.makePolygon([pts[i] for i in f] + [pts[f[0]]])
        face_objs.append(cq.Face.makeFromWires(poly))
    shell = cq.Shell.makeShell(face_objs)
    solid = cq.Solid.makeSolid(shell)
    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    exporters.export(solid, str(p), exportType="STEP")
    return p


def import_mesh_step(step_path: str | Path, *, units: str = "lu") -> TriangleMesh:
    """Import STEP via CadQuery tessellation when available."""
    try:
        import cadquery as cq
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("STEP import requires cadquery/openCascade") from exc

    shape = cq.importers.importStep(str(step_path)).val()
    verts_data, tris = shape.tessellate(1e-2)
    verts = np.asarray([(float(v.x), float(v.y), float(v.z)) for v in verts_data], dtype=np.float32)
    faces = np.asarray([(int(t[0]), int(t[1]), int(t[2])) for t in tris], dtype=np.int32)
    mesh = TriangleMesh(
        vertices=verts,
        faces=faces,
        units=units,
        name=Path(step_path).stem,
        metadata={"source": "step", "engine": CADGeometryEngine.CADQUERY.value},
    )
    mesh.validate()
    return mesh


def make_model_id() -> str:
    """Generate stable CAD model id."""
    return f"cad3d-{uuid.uuid4().hex[:12]}"
