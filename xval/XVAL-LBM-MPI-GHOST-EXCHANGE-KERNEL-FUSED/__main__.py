#!/usr/bin/env python3
"""Entry point for XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xval.XVAL_LBM_MPI_GHOST_EXCHANGE_KERNEL_FUSED.run_xval import main

if __name__ == "__main__":
    sys.exit(main())
