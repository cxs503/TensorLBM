"""Measure the interface thickness the LBM actually carries (verifier probe).

The accepted case's parameter table states ``delta = xi = 2.0 cells`` and
``k delta = 0.196``, i.e. it treats the Allen-Cahn parameter ``ac_width`` as the
interface thickness that the linear growth rate sees.  That is only true if the
*discretised* equilibrium profile is resolved.  This probe measures the profile
instead of assuming it:

1. ``relax``: start from the case's initial condition, switch buoyancy off
   (``g = 0``) so only the Allen-Cahn operator acts, and follow the x-averaged
   profile ``phi(y)``.  A tanh fit gives the *discretisation* equilibrium
   thickness (``xi_meas``) -- compare with ``ac_width = 2.0``;
2. ``flow``: the same measurement during the growth window of the reference run
   (``a/lam in [0.02, 0.08]``), i.e. the thickness the linear mode actually sees.

If ``xi_meas`` is much larger than ``ac_width``, the case's ``k delta`` -- and
therefore its "finite interface thickness cannot explain the residual" argument,
which varied ``ac_width`` rather than the measured profile -- is not what the
solver integrates.

Usage
-----
    python3 verify/probe_interface_width.py            # both arms, ~25 s
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
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

ART = HERE / "artifacts"
BASE = dict(lam=64, layers=8.0, g=2.0e-4, atwood=0.5, tau=0.515, tau_phi=0.501,
            ac_sigma=3.0e-4, ac_width=2.0, phase_scheme="allen_cahn",
            ac_mobility=20.0, seed_eigenmode=True, eta0_frac=0.01)


def profile(y: np.ndarray, phi_col: np.ndarray) -> dict:
    """Thickness of ``phi = tanh((y - y0)/xi)`` by a *linear* fit of ``atanh(phi)``.

    ``atanh(tanh(u)) = u`` exactly, so regressing ``atanh(phi)`` on ``y`` over the
    transition band returns ``1/xi`` without the finite-stencil bias a central
    difference at the crossing carries (that bias inflated xi by ~30 % for a
    2-cell profile, which is why the first version of this probe reported 2.68
    for a nominal 2.0).
    """
    s = np.sign(phi_col[int(np.argmax(y))])
    p = s * phi_col
    m = np.abs(p) < 0.98                       # keep away from the atanh poles
    if m.sum() < 6:
        return {"xi": float("nan"), "y0": float("nan"), "rms": float("nan"), "n": int(m.sum())}
    yy, uu = y[m], np.arctanh(p[m])
    A = np.vstack([yy, np.ones_like(yy)]).T
    (slope, icpt), *_ = np.linalg.lstsq(A, uu, rcond=None)
    xi = 1.0 / slope if slope != 0 else float("nan")
    pred = np.tanh((yy - (-icpt / slope)) / xi) if slope != 0 else np.zeros_like(yy)
    return {"xi": float(abs(xi)), "y0": float(-icpt / slope),
            "rms": float(np.sqrt(np.mean((p[m] - pred) ** 2))), "n": int(m.sum())}


def _col(phi: torch.Tensor, x: int) -> np.ndarray:
    return phi[:, x].detach().cpu().numpy().astype(float)


def measure(mode: str) -> dict:
    cfg = RTConfig(**BASE, n_steps=4000, sample_every=5, a_lo_frac=0.02,
                   a_hi_frac=0.08, stop_a_frac=10.0)
    g_eff = 0.0 if mode == "relax" else BASE["g"]
    cfg = RTConfig(**{**BASE, "g": g_eff}, n_steps=cfg.n_steps,
                   sample_every=cfg.sample_every, a_lo_frac=0.02, a_hi_frac=0.08,
                   stop_a_frac=10.0)
    torch.manual_seed(0)
    f, g, phi = init_rt_state(cfg)
    ny = phi.shape[0]
    y = np.arange(ny, dtype=float)
    rows = []
    for step in range(1, 1601 if mode == "relax" else 2600):
        f, g = rt_step(f, g, cfg, phi=phi)
        phi = g.sum(dim=0)
        if step % 25 == 0:
            a = mode_amplitude_zc(phi, cfg.lam) / cfg.lam
            prof = phi.detach().cpu().numpy().astype(float).mean(axis=1)
            r = profile(y, prof)
            rows.append({"step": step, "a_over_lam": a, **r})
            if mode == "flow" and a > 0.08:
                break
    # finite differences of the profile for a figure
    prof = phi.detach().cpu().numpy().astype(float).mean(axis=1)
    return {"mode": mode, "ac_width_param": BASE["ac_width"], "rows": rows,
            "final_profile": {"y": y.tolist(), "phi": prof.tolist()},
            "xi_initial": profile(y, 0.5 * (_col(init_rt_state(cfg)[2], 0)
                                            + _col(init_rt_state(cfg)[2], cfg.lam // 2)))["xi"]}


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    out = {}
    for mode in ("relax", "flow"):
        r = measure(mode)
        out[mode] = r
        xi_final = r["rows"][-1]["xi"]
        print(f"[{mode}] ac_width(param)={r['ac_width_param']:.2f}  "
              f"xi_initial={r.get('xi_initial', float('nan')):.3f}  "
              f"xi_measured(final)={xi_final:.3f}  "
              f"k*xi={2*math.pi/64*xi_final:.3f}  "
              f"(k*ac_width={2*math.pi/64*r['ac_width_param']:.3f})", flush=True)
        for row in r["rows"][::max(1, len(r["rows"]) // 6)]:
            print(f"    step={row['step']:5d} a/lam={row['a_over_lam']:.4f} "
                  f"xi={row['xi']:.3f} rms={row['rms']:.4f}", flush=True)
    (ART / "interface_width.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[written] {ART/'interface_width.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())