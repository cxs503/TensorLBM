"""Allen-Cahn mobility ladder on the LBM itself (verifier probe, no model assumptions).

The accepted case measures ``sigma_sim = 2.7725e-3`` for the reference parameter
point and compares it with the **sharp-interface** closed form
``sigma_visc = 3.0855e-3`` (-10.1 %).  The report attributes the residual to a
"solver-side buoyancy/interface coupling bias" without a mechanism test.

This probe runs the *same* solver, same grid, same everything, and only changes
the Allen-Cahn mobility ``M`` (the rate at which the phase field relaxes to its
double-well equilibrium).  ``M -> 0`` removes the phase-field damping channel
while leaving the flow solver (BGK + Boussinesq force) untouched.  Therefore:

* ``sigma_sim(M) -> sigma_visc``  =>  the deficit is produced by the interface
  model (a *model-form / parameter* effect), not by the flow solver;
* ``sigma_sim(M)`` flat          =>  the deficit lives in the flow discretisation
  or in the interface *thickness*, and the mobility is not the mechanism.

Protocol identical to ``run_case.py`` (same RTConfig fields, same estimator
``mode_amplitude_zc``, same primary window ``a/lam in [0.02, 0.08]``), so the
``M = 20`` arm is a reproduction check of the accepted number.

Usage
-----
    python3 verify/probe_mobility_ladder.py --mobility 20
    python3 verify/probe_mobility_ladder.py --all         # the whole ladder
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("RT_THREADS", "4")))

HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parent
REPO_ROOT = CASE_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.d2q9 import macroscopic  # noqa: E402
from tensorlbm.rayleigh_taylor import (  # noqa: E402
    RTConfig, init_rt_state, mode_amplitude_zc, rt_step,
)

ART = HERE / "artifacts" / "lbm_ladder"
WINDOW = (0.020, 0.080)
GATE_PCT = 15.0

#: reference point of the accepted case (run_case.SPECS[0])
BASE = dict(lam=64, layers=8.0, g=2.0e-4, atwood=0.5, tau=0.515, tau_phi=0.501,
            ac_sigma=3.0e-4, ac_width=2.0, phase_scheme="allen_cahn",
            seed_eigenmode=True, eta0_frac=0.01)
M_STEPS = 5200
SAMPLE = 6
A_STOP = 0.15


def sigma_inviscid(A, g, lam):
    return math.sqrt(A * g * 2.0 * math.pi / lam)


def sigma_viscous(A, g, lam, nu):
    k = 2.0 * math.pi / lam
    x = nu * k * k
    return math.sqrt(A * g * k + x * x) - x


def run(mobility: float, n_steps: int = M_STEPS) -> dict:
    cfg = RTConfig(**BASE, ac_mobility=mobility, n_steps=n_steps, sample_every=SAMPLE,
                   a_lo_frac=WINDOW[0], a_hi_frac=WINDOW[1], stop_a_frac=10.0)
    cfg.validate()
    torch.manual_seed(0)
    f, g, phi = init_rt_state(cfg)
    ts, amps, ma = [], [], 0.0
    t0 = time.time()
    for step in range(1, cfg.n_steps + 1):
        f, g = rt_step(f, g, cfg, phi=phi)
        phi = g.sum(dim=0)
        if step % cfg.sample_every == 0:
            a = mode_amplitude_zc(phi, cfg.lam) / cfg.lam
            if not math.isfinite(a):
                break
            ts.append(float(step))
            amps.append(a)
            _r, ux, uy = macroscopic(f)
            ma = max(ma, float(torch.sqrt(ux * ux + uy * uy).max()) * math.sqrt(3.0))
            if a > A_STOP:
                break
    wall = time.time() - t0

    pts = [(t, a) for t, a in zip(ts, amps) if WINDOW[0] <= a <= WINDOW[1]]
    sig = float("nan")
    r2 = float("nan")
    if len(pts) >= 4:
        tt = [p[0] for p in pts]
        ll = [math.log(p[1]) for p in pts]
        n = len(tt)
        mt, ml = sum(tt) / n, sum(ll) / n
        sxx = sum((x - mt) ** 2 for x in tt)
        sxy = sum((x - mt) * (y - ml) for x, y in zip(tt, ll))
        sig = sxy / sxx
        resid = [y - (ml + sig * (x - mt)) for x, y in zip(tt, ll)]
        ssr = sum(r * r for r in resid)
        sst = sum((y - ml) ** 2 for y in ll) or 1.0
        r2 = 1.0 - ssr / sst

    nu = cfg.nu
    s_ref = sigma_viscous(BASE["atwood"], BASE["g"], BASE["lam"], nu)
    return {
        "ac_mobility": mobility,
        "sigma_sim": sig,
        "r2": r2,
        "n_points": len(pts),
        "window": list(WINDOW),
        "nu": nu,
        "sigma_visc": s_ref,
        "sigma_inviscid": sigma_inviscid(BASE["atwood"], BASE["g"], BASE["lam"]),
        "err_vs_visc_pct": 100.0 * (sig / s_ref - 1.0) if math.isfinite(sig) else float("nan"),
        "err_vs_inv_pct": 100.0 * (sig / sigma_inviscid(BASE["atwood"], BASE["g"], BASE["lam"]) - 1.0)
        if math.isfinite(sig) else float("nan"),
        "gate_pct": GATE_PCT,
        "pass": bool(math.isfinite(sig) and abs(sig / s_ref - 1.0) * 100.0 <= GATE_PCT),
        "max_mach": ma,
        "n_steps_run": len(ts) * cfg.sample_every,
        "wall_s": wall,
        "history": {"t": ts, "a_over_lam": amps},
        "base": dict(BASE),
        "m_beta": mobility * 1.5 * BASE["ac_sigma"] / BASE["ac_width"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mobility", type=float, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--late", action="store_true", help="run to the case's full time")
    a = ap.parse_args(argv)
    ART.mkdir(parents=True, exist_ok=True)
    todo = [20.0, 10.0, 5.0, 2.0, 0.5] if a.all else [a.mobility if a.mobility else 20.0]
    out = []
    for m in todo:
        r = run(m, M_STEPS if not a.late else 5200)
        tag = f"M{m:g}"
        hist = r.pop("history")
        (ART / f"{tag}.json").write_text(json.dumps(r, indent=2) + "\n")
        with open(ART / f"{tag}_history.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["step", "a_over_lam"])
            for t, amp in zip(hist["t"], hist["a_over_lam"]):
                wr.writerow([t, f"{amp:.8e}"])
        out.append(r)
        print(f"[{tag:>6}] sigma={r['sigma_sim']:.6e} err_vs_visc={r['err_vs_visc_pct']:+7.3f}% "
              f"r2={r['r2']:.5f} n={r['n_points']:3d} Ma={r['max_mach']:.3f} "
              f"wall={r['wall_s']:.1f}s pass={r['pass']}", flush=True)
    if len(out) > 1:
        (ART / "ladder.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())