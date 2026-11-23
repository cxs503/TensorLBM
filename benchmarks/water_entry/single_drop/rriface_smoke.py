"""DE-X-RELX3-RRIFACE Gate0 smoke: 160-step divergence check.

Usage:
  python3 rriface_smoke.py arm bgk|rr_iface   # run one arm, dump f history
  python3 rriface_smoke.py check              # compare the two dumps

The arm is only admissible if the two trajectories diverge
(max|f_rr_iface - f_bgk| > 1e-6) and stay finite; otherwise the arm
aborts before any full-window run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import benchmarks.water_entry.single_drop.free_surface_probe as P

STEPS = 160
DIVERGE_FLOOR = 1e-6
DUMP = "/tmp/rriface_smoke_{arm}.pt"


def run_arm(rr_iface: bool) -> None:
    torch.set_num_threads(7)
    torch.manual_seed(0)
    captured: list[torch.Tensor] = []
    orig = P.free_surface_step

    def spy(f, fill, flags, solid, **kw):
        captured.append(f.clone())
        return orig(f, fill, flags, solid, **kw)

    P.free_surface_step = spy
    try:
        P.probe(
            8,
            0.06,
            STEPS,
            0.18,
            capacity=True,
            nearest=False,
            pilot=False,
            span=4.0,
            pool_depth=3.0,
            simple=True,
            release=1.53,
            planes_path=None,
            silhouette_path=None,
            ledger=False,
            diag_path=None,
            u_limit=0.15,
            cav_clos_alpha=0.0,
            rr_tau=0.0,
            rr_iface=rr_iface,
            cav_series_path=None,
        )
    finally:
        P.free_surface_step = orig
    assert len(captured) == STEPS, len(captured)
    torch.save(captured, DUMP.format(arm="bgk" if not rr_iface else "rr_iface"))
    print(f"arm={'bgk' if not rr_iface else 'rr_iface'} steps={len(captured)} saved")


def check() -> int:
    bgk = torch.load(DUMP.format(arm="bgk"))
    rr = torch.load(DUMP.format(arm="rr_iface"))
    assert len(bgk) == len(rr) == STEPS, (len(bgk), len(rr))
    diffs = [(a - b).abs().max().item() for a, b in zip(rr, bgk)]
    peak = max(diffs)
    finite = all(torch.isfinite(a).all().item() for a in rr)
    print(f"steps={len(diffs)} max_abs_diff={peak:.6e} finite={finite}")
    print(f"diff@step0={diffs[0]:.3e} diff@step{STEPS}={diffs[-1]:.6e}")
    ok = peak > DIVERGE_FLOOR and finite
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "check":
        sys.exit(check())
    run_arm(sys.argv[1] == "rr_iface")
