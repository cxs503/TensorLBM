"""Pytest configuration for TensorLBM tests."""

import sys
from pathlib import Path

# Add the installed tensorlbm package path so tests can find the new modules
# The worktree's TensorLBM/src only has airfoil/lighthill; the full package
# is installed from /root/tp-agent/work/m9187
_installed_path = Path("/root/tp-agent/work/m9187/src")
if str(_installed_path) not in sys.path:
    sys.path.insert(0, str(_installed_path))
