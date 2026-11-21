"""Capture the phi field at the death step of the prescribed n=8 run (diagnostic figure)."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

import numpy as np
import torch

import benchmarks.water_entry.single_drop.run as R

CAPTURE = {}


def spy(sim):
    out = R._macro_fp64_orig(sim)
    spy.calls += 1
    if spy.calls == 27:  # 0-based step 26 = the volume_drift death step
        CAPTURE["phi"] = out[0].numpy()
        CAPTURE["step"] = 26
    return out


R._macro_fp64_orig = R._macro_fp64
R._macro_fp64 = spy
spy.calls = 0

result = R.run(8, 0.03, 26, 6.0, 0.4, False)
print("stop:", result["stop_reason"], "contact:", result["contact_step"])
assert "phi" in CAPTURE, "capture missed"
np.save("/root/tp-agent/work/m26277/runs/t1_phi_n8_step26.npy", CAPTURE["phi"])
print("saved phi", CAPTURE["phi"].shape)
