from __future__ import annotations

import json

from tensorlbm.ship_cad3d import (
    create_parametric_hull_mesh,
    export_mesh_gltf,
    export_mesh_stl_ascii,
    import_mesh_stl,
)


def test_parametric_mesh_stats() -> None:
    mesh = create_parametric_hull_mesh(
        "series60",
        length=100.0,
        beam=16.0,
        draft=8.0,
        n_long=24,
        n_vert=16,
    )
    stats = mesh.stats()
    assert stats["vertex_count"] > 0
    assert stats["face_count"] > 0
    assert stats["units"] == "lu"


def test_export_import_stl_roundtrip(tmp_path) -> None:
    mesh = create_parametric_hull_mesh(
        "wigley",
        length=60.0,
        beam=10.0,
        draft=5.0,
        n_long=12,
        n_vert=8,
    )
    stl = export_mesh_stl_ascii(mesh, tmp_path / "mesh.stl")
    loaded = import_mesh_stl(stl)
    assert loaded.vertices.shape[1] == 3
    assert loaded.faces.shape[1] == 3
    assert loaded.faces.shape[0] > 0


def test_export_gltf_json(tmp_path) -> None:
    mesh = create_parametric_hull_mesh(
        "kcs",
        length=80.0,
        beam=12.0,
        draft=6.0,
        n_long=12,
        n_vert=8,
    )
    gltf = export_mesh_gltf(mesh, tmp_path / "mesh.gltf")
    data = json.loads(gltf.read_text(encoding="utf-8"))
    assert data["asset"]["version"] == "2.0"
    assert data["meshes"]
