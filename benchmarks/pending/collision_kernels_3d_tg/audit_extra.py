#!/usr/bin/env python
"""W5-C follow-up diagnostics: (a) library's own audit module on entropic_kbc,
(b) shear-wave length dependence for mrt27 vs cumulant (k-coupling vs tau-coupling)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

WORKTREE_SRC = "/nfs/wangxi/worktrees/bm_w5/src"
sys.path.insert(0, WORKTREE_SRC)

import tensorlbm  # noqa: E402

assert tensorlbm.__file__.startswith(WORKTREE_SRC)

from tensorlbm.collision_viscosity_audit import (  # noqa: E402
    CollisionViscosityAuditConfig,
    run_collision_viscosity_audit,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_viscosity import shear_wave_case  # noqa: E402


def main() -> None:
    torch.set_num_threads(16)
    out = {}

    # (a) library's own audit on entropic_kbc (tau=0.8 default and 0.9)
    for tau in (0.8, 0.9):
        cfg = CollisionViscosityAuditConfig(
            collision_model="entropic_kbc",
            tau=tau,
            dtype="float64",
        )
        try:
            rep = run_collision_viscosity_audit(cfg)
            d = rep.__dict__ if hasattr(rep, "__dict__") else dict(rep)
            for k, v in list(d.items()):
                if hasattr(v, "__dict__") or isinstance(v, tuple):
                    try:
                        d[k] = v._asdict() if hasattr(v, "_asdict") else str(v)
                    except Exception:
                        d[k] = str(v)
            out[f"lib_audit_entropic_kbc_tau{tau}"] = d
            print(f"lib audit entropic_kbc tau={tau}: {json.dumps(d)[:600]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            out[f"lib_audit_entropic_kbc_tau{tau}"] = {"error": repr(exc)}
            print(f"lib audit entropic_kbc tau={tau} FAILED: {exc!r}", flush=True)

    # (b) length dependence
    rows = []
    for kernel in ("mrt27", "cumulant_d3q27", "cascaded_d3q27", "trt27", "rlbm27"):
        for tau in (0.9, 1.3):
            for length in (32, 64):
                steps = 200 if length == 32 else 500
                r = shear_wave_case(kernel, tau, 1.0e-3, length=length, steps=steps)
                rows.append(r)
                print(
                    f"{kernel:16s} tau={tau:.2f} L={length}  nu_eff={r['nu_eff']:.8f} "
                    f"err={r['err_pct']:+.4f}%",
                    flush=True,
                )
    out["length_scan"] = rows

    path = Path(__file__).resolve().parent / "audit_extra.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
