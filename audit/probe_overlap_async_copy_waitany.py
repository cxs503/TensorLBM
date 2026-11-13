"""Independent audit probe for m17454 (XVAL-LBM-ASYNC-COPY-WAITANY-OVERLAP).

Not part of the mission deliverable: re-checks the bit-exactness claim outside the
ranges the mission's own script/test use (widths 9..24, worlds 2..5, a NONLINEAR
pointwise collision so exactness cannot be an artefact of the affine fixture).
"""
import sys

import torch

sys.path.insert(0, "/tmp/m17586_repro/src")
from tensorlbm.comm_overlap.virtual_case import run_virtual_case  # noqa: E402


def nl_collide(f):
    # pointwise, strictly non-affine, cell-local => border/interior split still legal
    return torch.tanh(f) * 1.25 + 0.1 * f * f + 1e-3


rows = []
bad = []
for width in (9, 10, 11, 12, 15, 16, 19, 20, 23, 24):
    for world in (2, 3, 4, 5):
        if width // world < 3:
            continue
        torch.manual_seed(1000 * width + world)
        full = torch.randn(19, 2, 3, width, dtype=torch.float64)
        for mode in ("blocking", "waitall", "waitany"):
            r = run_virtual_case(full, mode=mode, world_size=world, n_steps=3,
                                 collide_fn=nl_collide)
            rows.append((width, world, mode, r["mismatch"], r["max_abs_error"],
                         r["stall_us"], r["step_us"]))
            if r["mismatch"] != 0 or r["max_abs_error"] != 0.0:
                bad.append((width, world, mode, r["mismatch"], r["max_abs_error"]))
            if r["stall_us"] < -1e-9 or r["step_us"] <= 0:
                bad.append((width, world, mode, "negative-stall", r["stall_us"]))

print(f"cases run = {len(rows)}; nonzero-mismatch cases = {len(bad)}")
for b in bad[:10]:
    print("  BAD", b)
# stall ordering sanity: waitany <= waitall <= blocking (same model) must hold or be reported
viol = 0
for width in (9, 12, 16, 20, 24):
    for world in (2, 3, 4):
        if width // world < 3:
            continue
        torch.manual_seed(1000 * width + world)
        full = torch.randn(19, 2, 3, width, dtype=torch.float64)
        s = {m: run_virtual_case(full, mode=m, world_size=world, n_steps=3,
                                 collide_fn=nl_collide)["stall_us"]
             for m in ("blocking", "waitall", "waitany")}
        ok = s["waitany"] <= s["waitall"] + 1e-9 and s["waitall"] <= s["blocking"] + 1e-9
        if not ok:
            viol += 1
            print(f"  ORDER-VIOLATION width={width} world={world} {s}")
print(f"stall-order violations (waitany<=waitall<=blocking) = {viol}")
print("sample:", rows[:3])