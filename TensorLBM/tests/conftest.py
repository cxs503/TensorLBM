"""Pytest configuration for TensorLBM tests."""

import sys
from pathlib import Path

# Add the m10138 path where the ASBL modules are installed
_m10138_path = Path("/root/tp-agent/work/m10138/src")
if str(_m10138_path) not in sys.path:
    sys.path.insert(0, str(_m10138_path))

# Also add the worktree source for local modules
_worktree_src = Path(__file__).parent.parent / "src"
if str(_worktree_src) not in sys.path:
    sys.path.insert(0, str(_worktree_src))
