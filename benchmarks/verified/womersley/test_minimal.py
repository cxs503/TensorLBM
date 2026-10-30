#!/usr/bin/env python
"""Minimal test to reproduce the Womersley H=29 issue."""
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import numpy as np

print(f"Python: {sys.executable}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

DEVICE = torch.device("cpu")
print(f"DEVICE: {DEVICE}")

# Try to create a tensor
try:
    wall = torch.zeros((31, 16), dtype=torch.bool, device=DEVICE)
    print(f"✓ Created wall tensor: {wall.shape} on {wall.device}")
except Exception as e:
    print(f"✗ Failed to create wall tensor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now try importing tensorlbm modules
try:
    from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic
    from tensorlbm.solver import collide_bgk, stream
    from tensorlbm.turbulent_channel import _apply_body_force_2d
    print("✓ Imported tensorlbm modules")
except Exception as e:
    print(f"✗ Failed to import tensorlbm: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try creating tensor again after imports
try:
    wall2 = torch.zeros((31, 16), dtype=torch.bool, device=DEVICE)
    print(f"✓ Created wall2 tensor after imports: {wall2.shape} on {wall2.device}")
except Exception as e:
    print(f"✗ Failed to create wall2 tensor after imports: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
