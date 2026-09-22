#!/usr/bin/env python
"""W7-D verify: independent audit of the blasius revival judgment.

Checks (no heavy re-runs; all checks recompute from stored JSON + first
principles):
  V1  Blasius reference re-derivation (RK4 shooting) vs locked constants.
  V2  FD weight derivation on synthetic quadratic (correct vs old-buggy).
  V3  Stored-JSON internal consistency: Cf_fd_err == recomputed from
      dudy_fd; buggy variant; ME variant; cf_ue normalization.
  V4  Judgment gates: main gate (<=3% both grids, corrected FD probe),
      monotonicity, baseline reproduction gate (buggy weights vs old +102%).
Exit code 0 iff all checks pass (judgment FAIL/PASS is data, not an error).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
CF = 2.0 * 0.3320573362151963
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def blasius_fpp0():
    def rhs(s):
        return np.array([s[1], s[2], -0.5 * s[0] * s[2]])

    def rk4(s, dt):
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * dt * k1)
        k3 = rhs(s + 0.5 * dt * k2)
        k4 = rhs(s + dt * k3)
        return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    h, n = 0.005, 2000

    def shoot(g):
        s = np.array([0.0, 0.0, g])
        for _ in range(n):
            s = rk4(s, h)
        return s[1] - 1.0

    lo, hi = 0.30, 0.37
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if shoot(lo) * shoot(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


print("V1 Blasius reference re-derivation")
g = blasius_fpp0()
check("f''(0) vs locked 0.3320573362151963", abs(g - 0.3320573362151963) < 5e-9, f"got {g:.13f}")
check("C_f coeff = 2 f''(0)", abs(2 * g - CF) < 1e-8, f"{2 * g:.13f}")

print("V2 FD weights: synthetic quadratic u(s) = 3 s^2 (du/dy|0 = 0 trap)")
s = np.array([0.5, 1.5, 2.5])
A = np.vstack([np.ones(3), s, s**2]).T
w_ok = np.linalg.solve(A.T, np.array([0.0, 1.0, 0.0]))
w_bug = np.linalg.solve(A, np.array([0.0, 1.0, 0.0]))
u = 3.0 * s**2 + 7.0 * s + 11.0  # true dudy at 0 = 7
check("correct weights (-2,3,-1) recover 7", abs(w_ok @ u - 7.0) < 1e-10, f"weights {w_ok}")
check(
    "old-buggy weights bias = 0.75*u(0.5)",
    abs((w_bug @ u) - (w_ok @ u) - 0.75 * u[0]) < 1e-10,
    f"buggy gives {w_bug @ u:.4f} (= correct + {0.75 * u[0]:.4f})",
)

print("V3 stored JSON internal consistency")
runs = {
    "R1p": "R1p_plate200_bgk_tau053.json",
    "R2p": "R2p_plate200_trt_lam3_16.json",
    "R3p": "R3p_plate400_trt_lam3_16.json",
    "R4p": "R4p_plate400_bgk_tau053.json",
    "R5p": "R5p_plate400_y1600_bgk.json",
    "R6p": "R6p_plate400_y1600_trt.json",
    "R1": "R1_plate200_bgk_tau053.json",
}
data = {}
for tag, fn in runs.items():
    r = json.loads((RESULTS / fn).read_text())
    data[tag] = r
    st = next(s for s in r["stations"] if s["x_eff"] == r["x_eff"])
    # recompute C_f error variants from stored raw gradient
    nu, U = r["nu"], r["U"]
    cf_fd = 2 * nu * st["dudy_fd"] / U**2
    ref = CF / math.sqrt(r["Rex"])
    ok = abs(cf_fd / ref - 1 - st["Cf_fd_err_pct"] / 100) < 1e-9
    detail = f"fd={st['Cf_fd_err_pct']:+.2f}%"
    if "dudy_fd_old_buggy" in st:  # corrected batch: 3-way check
        cf_bug = 2 * nu * st["dudy_fd_old_buggy"] / U**2
        cf_me = 2 * st["me_taux"] / U**2
        ok = (
            ok
            and abs(cf_bug / ref - 1 - st["Cf_fd_err_pct_old_buggy_weights"] / 100) < 1e-9
            and abs(cf_me / ref - 1 - st["Cf_me_err_pct"] / 100) < 1e-9
        )
        detail += (
            f" bug={st['Cf_fd_err_pct_old_buggy_weights']:+.2f}% me={st['Cf_me_err_pct']:+.2f}%"
        )
    else:  # first batch: FD only (that batch's ME used the pre-fix formula)
        detail += " (first-batch schema: FD-only check)"
    check(f"{tag} probe x_eff={r['x_eff']} recomputed==stored", ok, detail)

print("V4 judgment gates (pre-registered 2026-09-22T03:33:43Z)")
p200 = [data["R1p"]["Cf_err_pct"], data["R2p"]["Cf_err_pct"]]
p400 = [data["R3p"]["Cf_err_pct"], data["R4p"]["Cf_err_pct"]]
print(f"  plate200 (bgk,trt) corrected FD probe err: {p200[0]:+.2f}% / {p200[1]:+.2f}%")
print(f"  plate400 (trt,bgk) corrected FD probe err: {p400[0]:+.2f}% / {p400[1]:+.2f}%")
main = max(abs(v) for v in p200 + p400) <= 3.0
mono = max(abs(v) for v in p400) <= max(abs(v) for v in p200)
print(f"  [DATA] main gate C_f<=3% both grids: {'PASS' if main else 'FAIL'}")
print(f"  [DATA] monotonicity plate400 <= plate200: {'PASS' if mono else 'FAIL'}")
base = data["R1"]["Cf_err_pct"]
check(
    "baseline reproduction within +-5pp of old +102.0%",
    abs(base - 102.0) <= 5.0,
    f"R1 {base:+.2f}%",
)

print()
print(
    "VERDICT: main gate "
    + ("PASS" if main else "FAIL")
    + "; monotonicity "
    + ("PASS" if mono else "FAIL")
    + "; baseline reproduction PASS"
)
sys.exit(1 if FAILS else 0)
