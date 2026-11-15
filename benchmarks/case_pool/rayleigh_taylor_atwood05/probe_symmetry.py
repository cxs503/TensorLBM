"""Diagnostic probe -- is the spike/bubble asymmetry of this case real?

``REPORT.md`` states that the Boussinesq equations solved here are invariant
under ``(phi, y) -> (-phi, -y)``, so the spike and bubble fronts *must* be
mirror images and the finite-Atwood two-fluid asymmetry cannot be reproduced.
The late-time run (:mod:`run_late`) nevertheless shows ``h_up`` and ``h_down``
diverging by a factor ~2.  Two explanations are possible and this probe
separates them:

H1 **physical/methodological**: the discrete problem really is asymmetric
   (initial interface placed at ``y = ny/2 = 256``, whereas the discrete box is
   symmetric about ``y = (ny-1)/2 = 255.5`` -> a half-cell offset; walls in y);
H2 **measurement**: the front estimator is biased.  ``interface_zero_crossing``
   returns the *lowest* crossing of each column, and both ``h_up`` and
   ``h_down`` are taken relative to the *instantaneous* column mean (which drifts
   when the interface rolls up / becomes multi-valued) instead of a fixed
   reference.

Measured here, per sample:

* ``sym_res``  = max|phi(x, y) + phi(x + lam/2, 2*yc - y)| / max|phi|
  (mirror-antisymmetry residual about the discrete centre ``yc = (ny-1)/2``),
  on the rows where both members exist;
* ``multi``   = number of columns whose phi profile has more than one
  ``phi = 0`` crossing (a single-valued contour is a precondition of the
  ``zc``-based front estimator);
* ``mean_dev`` = ``mean(zc) - y0`` in wavelengths -- the bias H2 predicts;
* fronts from *both* estimators:
  ``zc``-based (lowest crossing, instantaneous mean = the existing definition)
  and "tip"-based (fixed reference ``y0 = ny/2``, bubble tip = max of the
  *topmost* crossing, spike tip = min of the *lowest* crossing).

Usage: ``python3 probe_symmetry.py [--steps N] [--lam 64]``
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("RT_THREADS", "4")))

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT), str(BENCH_DIR), str(BENCH_DIR / "ref")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from run_case import CaseSpec  # noqa: E402

from tensorlbm.rayleigh_taylor import init_rt_state, rt_step  # noqa: E402


def crossings(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(lowest, topmost, count) crossing heights per column, NaN when none."""
    ny, nx = phi.shape
    cols = torch.arange(nx, device=phi.device)
    neg = (phi < 0.0)[:-1] & ~(phi < 0.0)[1:]
    low = torch.full((nx,), float("nan"), dtype=torch.float64)
    top = torch.full((nx,), float("nan"), dtype=torch.float64)
    cnt = neg.sum(dim=0).to(torch.float64)
    for j in range(nx):
        idx = torch.nonzero(neg[:, j], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        def interp(i: int) -> float:
            c0 = float(phi[i, j])
            c1 = float(phi[i + 1, j])
            return i + (-c0) / (c1 - c0)
        low[j] = interp(int(idx[0]))
        top[j] = interp(int(idx[-1]))
    del ny, cols
    return low, top, cnt


def symmetry_residual(phi: torch.Tensor, lam: int) -> float:
    """max|phi(x,y) + phi(x+lam/2, 2*yc-y)| / max|phi| about ``yc=(ny-1)/2``."""
    ny, nx = phi.shape
    yc = 0.5 * (ny - 1)
    shifted = torch.roll(phi, shifts=-(lam // 2), dims=1)
    # y -> 2*yc - y maps row 0 -> 2*yc (out of range) and row ny-1 -> 0;
    # compare only rows with an in-range mirror partner (0 .. 2*yc-1)
    n = int(2 * yc)                      # rows 0..n-1 map into 0..n-1 reversed
    a = phi[:n]
    b = torch.flip(shifted[:n], dims=(0,))
    res = float((a + b).abs().max())
    scale = float(phi.abs().max())
    return res / scale if scale > 0.0 else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2600)
    ap.add_argument("--lam", type=int, default=64)
    ap.add_argument("--g", type=float, default=2.0e-4)
    ap.add_argument("--every", type=int, default=200)
    args = ap.parse_args(argv)

    spec = CaseSpec(f"probe_l{args.lam}", lam=args.lam, g=args.g)
    cfg = spec.cfg()
    torch.manual_seed(0)
    f, g, phi = init_rt_state(cfg)
    ny, nx = phi.shape
    y0 = 0.5 * ny                              # initial mean interface height
    yc = 0.5 * (ny - 1)                        # discrete mirror plane
    tau_unit = math.sqrt(spec.atwood * spec.g / spec.lam)
    print(f"lam={args.lam} ny={ny} nx={nx} y0={y0} yc={yc} (half-cell offset "
          f"{y0 - yc:+.2f}) steps={args.steps}")
    print(f"{'tau':>6} {'sym_res':>10} {'max|phi|':>9} {'multi':>6} "
          f"{'mean_dev/lam':>13} {'h_up_zc':>9} {'h_dn_zc':>9} "
          f"{'h_bub_tip':>10} {'h_spk_tip':>10}")
    for step in range(1, args.steps + 1):
        f, g = rt_step(f, g, cfg, phi=phi)
        phi = g.sum(dim=0)
        if step % args.every == 0:
            low, top, cnt = crossings(phi)
            mean = float(torch.nan_to_num(low, nan=0.0).sum()
                         / max(1, int((~torch.isnan(low)).sum())))
            lo_max = float(torch.nan_to_num(low, nan=-1e30).max())
            lo_min = float(torch.nan_to_num(low, nan=1e30).min())
            top_max = float(torch.nan_to_num(top, nan=-1e30).max())
            h_up_zc = (lo_max - mean) / args.lam
            h_dn_zc = (mean - lo_min) / args.lam
            h_bub = (top_max - y0) / args.lam
            h_spk = (y0 - lo_min) / args.lam
            print(f"{step * tau_unit:6.3f} {symmetry_residual(phi, args.lam):10.3e} "
                  f"{float(phi.abs().max()):9.4f} {int((cnt > 1).sum()):6d} "
                  f"{(mean - y0) / args.lam:+13.4f} {h_up_zc:9.4f} {h_dn_zc:9.4f} "
                  f"{h_bub:10.4f} {h_spk:10.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())