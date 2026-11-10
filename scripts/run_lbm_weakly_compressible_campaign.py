#!/usr/bin/env python3
"""Thin CLI for the weak-compressibility verification campaign (P102W3-N05).

All the work lives in :mod:`tensorlbm.verification.weakly_compressible`; this
wrapper only exists so the campaign can be launched the same way as the other
verification campaigns in ``scripts/``.

Examples
--------
Full campaign, one process per case (default 5 workers)::

    python scripts/run_lbm_weakly_compressible_campaign.py --workers 8

Re-run only the channel sweeps (the cylinder raw records are reused)::

    python scripts/run_lbm_weakly_compressible_campaign.py \
        --only poiseuille,pressure_channel

Serial (small) run for a quick end-to-end check::

    python scripts/run_lbm_weakly_compressible_campaign.py serial --machs 0.05,0.1,0.2
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Pin BLAS to one thread before numpy is imported: the D2Q9 kernels here are
# tiny-matrix reductions and OpenBLAS thread contention costs an order of
# magnitude on a 128-core node.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.verification.weakly_compressible import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))