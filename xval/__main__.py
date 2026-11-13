#!/usr/bin/env python3
"""Entry point for XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S (Round 2).

Usage:
    python -m tensorlbm.xval_nebb_bodyforce_r2
    python xval/xval_nebb_bodyforce_driver_r2.py

Produces: xval/xval_nebb_bodyforce_driver_r2.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo src is on path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from xval.xval_nebb_bodyforce_driver_r2 import main

if __name__ == "__main__":
    main()
