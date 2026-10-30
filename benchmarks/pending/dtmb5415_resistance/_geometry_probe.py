#!/usr/bin/env python3
"""Engine-level voxel probe: what body does the solver actually integrate?

``GeneralSimEngine.setup()`` is cheap (voxelisation only, no time stepping), so
one probe per (STL, resolution) gives the geometry table used by
``artifacts/sdaa_dtmb5415_diagnosis.json``:

    solid cells -> voxel displacement -> Cb error vs 0.507*L*B*T
    staircase wall faces vs smooth wetted area (the normalisation bias)

The mask is exactly the one ``run.py`` will integrate, so nothing here is
reconstructed after the fact.

Usage
-----
  PYTHONPATH=src python _geometry_probe.py --out runs/.../geometry_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from run import (  # noqa: E402
    BEAM,
    DRAFT,
    LPP,
    U_PHYS,
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
    geometry_resolution_report,
    stl_wetted_area_breakdown,
)


def probe(stl: Path, l_cells: int) -> dict:
    config = GeneralSimConfig(
        name=f"probe_L{l_cells}",
        geometry=GeometryConfig(source=GeometrySource.STL_FILE, stl_path=str(stl), stl_units="m"),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=U_PHYS * LPP / 1000.0,
            inlet_velocity=U_PHYS,
            reference_length=LPP,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT,
            resolution=l_cells,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=0,
            device="cpu",
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            mass_correction=False,
        ),
        output=OutputConfig(directory="/tmp/m454/probe", formats=[], save_forces=False),
    )
    engine = GeneralSimEngine(config)
    setup = engine.setup()
    s = engine.solid.cpu().numpy().astype(np.int8)
    stair = int(
        np.abs(np.diff(s, axis=2)).sum()
        + np.abs(np.diff(s, axis=1)).sum()
        + np.abs(np.diff(s, axis=0)).sum()
    )
    bd = stl_wetted_area_breakdown(stl, l_cells, DRAFT)
    rep = geometry_resolution_report(
        n_solid_cells=int(s.sum()),
        dx_m=LPP / l_cells,
        stair_wall_faces=stair,
        smooth_wetted_lb=bd["wetted_area_lb"],
        near_wall_cells=setup["near_wall_cells"],
    )
    rep["stl"] = stl.name
    rep["L_cells"] = l_cells
    rep["wetted_area_m2"] = bd["wetted_area_m2"]
    rep["cap_area_m2"] = bd["cap_area_m2"]
    rep["domain_lu"] = setup["domain_lu"]
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE.parents[2] / "runs" / "sdaa_dtmb5415_m452"
                                        / "geometry_probe.json"))
    ap.add_argument("--resolutions", type=int, nargs="*", default=[24, 32, 48, 64])
    args = ap.parse_args()

    stls = [_HERE / "geometry" / "dtmb5415_underwater.stl",
            _HERE / "geometry" / "dtmb5415_underwater_closed.stl"]
    rows = []
    for stl in stls:
        if not stl.exists():
            continue
        for L in args.resolutions:
            r = probe(stl, L)
            rows.append(r)
            print(
                f"{stl.name:38s} L={L:3d} T/dx={r['T_over_dx']:5.2f} "
                f"solid={r['n_solid_cells']:6d} Cb={r['cb_voxel']:7.3f} "
                f"({r['cb_error_pct']:+8.1f}%) A_stair/A_smooth="
                f"{r['stair_over_smooth_wetted_area']:6.2f} -> {r['verdict']}",
                flush=True,
            )
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                json.dumps(
                    {
                        "reference": {
                            "Lpp_m": LPP,
                            "beam_m": BEAM,
                            "draft_m": DRAFT,
                            "cb": 0.507,
                            "displacement_m3": 0.507 * LPP * BEAM * DRAFT,
                            "wetted_area_m2": 3021.7,
                        },
                        "probe": "GeneralSimEngine.setup() solid mask -> stair faces",
                        "rows": rows,
                    },
                    indent=2,
                )
            )
    print("probe ->", args.out, flush=True)


if __name__ == "__main__":
    main()