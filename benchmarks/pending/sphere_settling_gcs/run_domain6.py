#!/usr/bin/env python3
"""Extra study: confinement at R=6 (λ→0 on a resolved body).

`study_domain()` runs the λ ladder at R=4; the R=4 voxel sphere turns out to be
under-resolved for creep flow (see the report §5), so the λ→0 extrapolation is
repeated on the resolved body: R=6 with L = 48, 64, 72 (λ = 0.125 … 0.083).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[2] / "src"))

import torch  # noqa: E402

from run import run_case  # noqa: E402


def main() -> int:
    torch.set_num_threads(8)
    runs = [run_case(R=6.0, L=L, re=0.05) for L in (48, 64, 72)]
    out = {"case": "sphere_settling_gcs", "study": "domain6", "runs": runs}
    (_HERE / "out_domain6.json").write_text(json.dumps(out, indent=1))
    print("wrote out_domain6.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())