"""Pytest configuration for TensorLBM tests."""

import sys
from pathlib import Path

# Add the installed tensorlbm package path so tests can find the new modules
# The full package is installed from /root/tp-agent/work/m9250
_installed_path = Path("/root/tp-agent/work/m9250/src")
if str(_installed_path) not in sys.path:
    sys.path.insert(0, str(_installed_path))

# Also add the worktree source for local modules
_worktree_src = Path(__file__).parent.parent / "src"
if str(_worktree_src) not in sys.path:
    sys.path.insert(0, str(_worktree_src))
