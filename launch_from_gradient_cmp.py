#!/usr/bin/env python3
"""Launch from_gradient comparison runs on STL-voxelized hulls.

Runs the same STL hull geometry but with from_gradient normals (instead
of STL normals) to enable a fair comparison of normal methods on the
same geometry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stl_ship_worker import run_ship_benchmark

STL_DIR = Path(
    "/root/ship-performance-platform-incoming/ship-performance-platform/"
    "backend/data/geometry/ships"
)

if __name__ == "__main__":
    ship = sys.argv[1] if len(sys.argv) > 1 else "kvlcc2"
    did = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    out = sys.argv[3] if len(sys.argv) > 3 else None

    configs = {
        "kvlcc2": dict(
            stl_path=STL_DIR / "KVLCC2_Hull.stl", ship_name="KVLCC2",
            nx=200, ny=80, nz=80, Re=1e5, u_in=0.06,
            n_steps=5000, warmup=1000, cs_smag=0.05,
        ),
        "dtmb5415": dict(
            stl_path=STL_DIR / "DTMB5415_Hull.stl", ship_name="DTMB5415",
            nx=200, ny=80, nz=80, Re=1e5, u_in=0.06,
            n_steps=5000, warmup=1000, cs_smag=0.05,
        ),
        "kcs": dict(
            stl_path=STL_DIR / "KCS_Hull.stl", ship_name="KCS",
            nx=200, ny=80, nz=80, Re=1000, u_in=0.06,
            n_steps=5000, warmup=500, cs_smag=0.05,
        ),
    }

    cfg = configs[ship]
    run_ship_benchmark(
        test_id=0, device_id=did,
        normal_method="from_gradient",
        dA_method="none",  # from_gradient uses dA=1.0
        output_path=out,
        **cfg,
    )
