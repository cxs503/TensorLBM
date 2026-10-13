"""Prereg §3 single extension (frozen lines 61-63): spread>2% arms re-run ONCE
at N x 1.5 with the frozen serial runner; N is the ONLY change (steps_for
monkeypatch; seed/protocol/windows/mass-correction code path identical)."""

import sys

sys.path.insert(0, "/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/formal")
import formal_run as F

EXT = {0.95: 3013173, 1.05: 3086496}  # ceil(1.5 * base 2008782 / 2057664)

for frac, n in EXT.items():
    F.steps_for = lambda spt, gref, _n=n: _n
    F.run_arm(256, frac)
    print(f"EXT_DONE arm{frac} N={n}", flush=True)
print("EXT_ALL_DONE", flush=True)
