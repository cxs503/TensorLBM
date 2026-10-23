#!/usr/bin/env python3
"""Matrix runner: test receive-gate variants at g=0 (static column) and g=1e-2."""
from __future__ import annotations
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import probe  # noqa: E402

OUT = Path("/root/TensorLBM_feat2/runs/fs_aprime_20260929")

ARMS = {
    "baseline":        {},                                     # open recv (F1)
    "recv_closed":     {"TL_FS_ABL_RECVCLOSED": "1"},          # strict fill>1e-3
    "graded_n3":       {"TL_FS_RECV_GRADED": "1", "TL_FS_RECV_NMIN": "3"},
    "graded_n4":       {"TL_FS_RECV_GRADED": "1", "TL_FS_RECV_NMIN": "4"},
    "graded_n5":       {"TL_FS_RECV_GRADED": "1", "TL_FS_RECV_NMIN": "5"},
    "graded_n6":       {"TL_FS_RECV_GRADED": "1", "TL_FS_RECV_NMIN": "6"},
    "graded_n3_m0":    {"TL_FS_RECV_GRADED": "1", "TL_FS_RECV_NMIN": "3", "TL_FS_RECV_MASS_EPS": "0.0"},
}

def apply(env):
    keys = [k for k in os.environ if k.startswith("TL_FS_")]
    for k in keys:
        del os.environ[k]
    os.environ.update(env)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "g0"
    if mode == "g0":
        a, g, steps, interval, rg = 8, 0.0, 200, 100, 1.0
    else:
        a, g, steps, interval, rg = 8, 1e-2, 60, 30, 1.0
    summary = []
    for name, env in ARMS.items():
        apply(env)
        print(f"\n########## {name}  env={env}", flush=True)
        s = probe.run(a, g, steps, interval, 0.8, rg, "cpu", OUT,
                      f"mat_{mode}_{name}", False, False)
        summary.append((name, s[-1]["front"], s[-1]["X"], s[-1]["H"],
                        s[-1]["iface_cells"], s[-1]["mass_drift_rel"]))
    print(f"\n=== SUMMARY mode={mode} a={a} g={g} steps={steps} ===")
    for name, front, X, H, iface, drift in summary:
        print(f"  {name:16s} front={front:3d} X={X:6.3f} H={H:6.3f} iface={iface:5d} drift={drift:+.4%}")