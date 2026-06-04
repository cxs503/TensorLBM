"""3-D CAD service layer for platform endpoints."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tensorlbm.ship_cad import ShipHullType, build_hull_mask
from tensorlbm.ship_cad3d import (
    TriangleMesh,
    create_parametric_hull_mesh,
    export_mesh_gltf,
    export_mesh_step,
    export_mesh_stl_ascii,
    import_mesh_step,
    import_mesh_stl,
    make_model_id,
)


@dataclass(slots=True)
class CAD3DVersion:
    version: int
    source_type: Literal["parametric", "stl", "step"]
    payload: dict[str, object]


@dataclass(slots=True)
class CAD3DModel:
    model_id: str
    source_type: Literal["parametric", "stl", "step"]
    payload: dict[str, object]
    units: str = "lu"
    versions: list[CAD3DVersion] = field(default_factory=list)


class CAD3DService:
    """In-memory model registry + geometry operations."""

    def __init__(self) -> None:
        self._models: dict[str, CAD3DModel] = {}

    def create_model(
        self,
        *,
        source_type: Literal["parametric", "stl", "step"],
        payload: dict[str, object],
        units: str = "lu",
    ) -> CAD3DModel:
        model_id = make_model_id()
        model = CAD3DModel(model_id=model_id, source_type=source_type, payload=payload.copy(), units=units)
        model.versions.append(CAD3DVersion(version=1, source_type=source_type, payload=payload.copy()))
        self._models[model_id] = model
        return model

    def get_model(self, model_id: str) -> CAD3DModel:
        model = self._models.get(model_id)
        if model is None:
            raise KeyError(f"model not found: {model_id}")
        return model

    def update_model(self, model_id: str, payload: dict[str, object]) -> CAD3DModel:
        model = self.get_model(model_id)
        model.payload = payload.copy()
        model.versions.append(
            CAD3DVersion(
                version=len(model.versions) + 1,
                source_type=model.source_type,
                payload=payload.copy(),
            )
        )
        return model

    def restore_version(self, model_id: str, version: int) -> CAD3DModel:
        model = self.get_model(model_id)
        hit = next((v for v in model.versions if v.version == version), None)
        if hit is None:
            raise KeyError(f"version not found: {version}")
        model.payload = hit.payload.copy()
        model.versions.append(
            CAD3DVersion(
                version=len(model.versions) + 1,
                source_type=hit.source_type,
                payload=hit.payload.copy(),
            )
        )
        return model

    def _mesh_from_model(self, model: CAD3DModel) -> TriangleMesh:
        if model.source_type == "parametric":
            return create_parametric_hull_mesh(
                str(model.payload.get("hull_type", ShipHullType.SERIES60.value)),
                length=float(model.payload.get("length", 100.0)),
                beam=float(model.payload.get("beam", 16.0)),
                draft=float(model.payload.get("draft", 8.0)),
                n_long=int(model.payload.get("n_long", 80)),
                n_vert=int(model.payload.get("n_vert", 40)),
                units=model.units,
            )

        file_path = str(model.payload.get("file_path", ""))
        if not file_path:
            raise ValueError("missing file_path for imported model")
        if model.source_type == "stl":
            return import_mesh_stl(file_path, units=model.units)
        return import_mesh_step(file_path, units=model.units)

    def model_mesh(self, model_id: str) -> TriangleMesh:
        return self._mesh_from_model(self.get_model(model_id))

    def model_stats(self, model_id: str) -> dict[str, object]:
        model = self.get_model(model_id)
        mesh = self._mesh_from_model(model)
        return {
            "model_id": model_id,
            "source_type": model.source_type,
            "versions": len(model.versions),
            "mesh": mesh.stats(),
        }

    def export_model(self, model_id: str, fmt: Literal["stl", "gltf", "step"]) -> tuple[Path, str]:
        model = self.get_model(model_id)
        mesh = self._mesh_from_model(model)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            if fmt == "stl":
                out = export_mesh_stl_ascii(mesh, root / f"{mesh.name}.stl")
                mime = "model/stl"
            elif fmt == "gltf":
                out = export_mesh_gltf(mesh, root / f"{mesh.name}.gltf")
                mime = "model/gltf+json"
            else:
                out = export_mesh_step(mesh, root / f"{mesh.name}.step")
                mime = "application/step"
            content = out.read_bytes()

        final = Path(tempfile.gettempdir()) / "tensorlbm_platform" / "cad3d_exports"
        final.mkdir(parents=True, exist_ok=True)
        target = final / out.name
        target.write_bytes(content)
        return target, mime

    def build_lbm_mask(self, model_id: str, *, nx: int, ny: int, nz: int, device: str = "cpu") -> dict[str, object]:
        model = self.get_model(model_id)
        if model.source_type != "parametric":
            raise ValueError("LBM mask bridge currently supports parametric hull models only")

        hull_type = str(model.payload.get("hull_type", ShipHullType.SERIES60.value))
        length = float(model.payload.get("length", nx * 0.5))
        beam = float(model.payload.get("beam", ny * 0.25))
        draft = float(model.payload.get("draft", nz * 0.3))

        mask, stats = build_hull_mask(
            hull_type=hull_type,
            nx=nx,
            ny=ny,
            nz=nz,
            length=length,
            beam=beam,
            draft=draft,
            device=device,
        )
        return {
            "model_id": model_id,
            "solid_cells": int(mask.sum().item()),
            "stats": stats,
        }


cad3d_service = CAD3DService()
