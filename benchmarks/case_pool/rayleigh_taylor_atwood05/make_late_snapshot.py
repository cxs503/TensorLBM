"""Field snapshots + cloud maps for the late-time phase of
``CASE-RT-INSTABILITY-ATWOOD``.

One short run of the reference configuration (``lam = 64``, ``g = 2e-4``,
``A = 1/2``) saves the phase field ``phi``, the vertical velocity ``u_y`` and the
``phi = 0`` interface (with the two tip markers used by ``run_late.py``) at three
instants inside/around the terminal-velocity window, writes them to
``artifacts/late/snapshots/`` and draws

    fig_late3_snapshots.png   phi and u_y at tau = 2.0 / 3.5 / 5.0, with the
                              bubble/spike tips marked and the Layzer velocity
                              as a reference vector.

Purpose: a field-level picture that shows *why* the extremal-crossing estimator
has to be split into "first" and "last" light->heavy transition -- the interface
rolls up and becomes multi-valued at exactly the time the terminal velocity sets
in.

Usage: ``python3 make_late_snapshot.py [--steps N]``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

matplotlib = __import__("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

torch.set_num_threads(int(os.environ.get("RT_THREADS", "4")))

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT), str(BENCH_DIR), str(BENCH_DIR / "ref")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import late_analysis as la  # noqa: E402
import run_late  # noqa: E402
from run_case import CaseSpec  # noqa: E402

from tensorlbm.d2q9 import macroscopic  # noqa: E402
from tensorlbm.rayleigh_taylor import (  # noqa: E402
    init_rt_state,
    interface_zero_crossing,
    rt_step,
)

SNAP_TAU = (2.0, 3.5, 5.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BENCH_DIR / "artifacts" / "late" / "snapshots"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spec = CaseSpec("late_l064_g2e-4", lam=64, g=2.0e-4)
    cfg = spec.cfg()
    torch.manual_seed(0)
    f, g, phi = init_rt_state(cfg)
    ny = phi.shape[0]
    y0 = 0.5 * ny
    tau_unit = math.sqrt(spec.atwood * spec.g / spec.lam)
    targets = [t / tau_unit for t in SNAP_TAU]     # in steps
    snaps = []

    for step in range(1, cfg.n_steps + 1):
        f, g = rt_step(f, g, cfg, phi=phi)
        phi = g.sum(dim=0)
        if targets and step >= targets[0]:
            tau = step * tau_unit
            _rho, ux, uy = macroscopic(f)
            low = interface_zero_crossing(phi)
            top = run_late.topmost_crossing(phi)
            snaps.append({
                "tau": tau,
                "step": step,
                "phi": phi.cpu().numpy().astype(np.float32),
                "uy": uy.cpu().numpy().astype(np.float32),
                "ux": ux.cpu().numpy().astype(np.float32),
                "zc_low": low.cpu().numpy(),
                "zc_top": top.cpu().numpy(),
                "h_bub": (float(torch.nan_to_num(top, nan=-1e30).max()) - y0) / spec.lam,
                "h_spk": (y0 - float(torch.nan_to_num(low, nan=1e30).min())) / spec.lam,
                "n_multi": int(((phi[:-1] < 0) & ~(phi[1:] < 0)).sum(dim=0).gt(1).sum()),
            })
            print(f"snapshot tau={tau:.2f} step={step} h_bub={snaps[-1]['h_bub']:.3f} "
                  f"h_spk={snaps[-1]['h_spk']:.3f} multi={snaps[-1]['n_multi']}", flush=True)
            targets.pop(0)

    np.savez_compressed(out / "snapshots.npz",
                        **{f"{k}_{i}": s[k] for i, s in enumerate(snaps)
                           for k in ("phi", "uy", "ux", "zc_low", "zc_top")},
                        meta=json.dumps([{k: v for k, v in s.items()
                                          if k not in ("phi", "uy", "ux", "zc_low", "zc_top")}
                                         for s in snaps]))

    fr_lz = la.layzer_froude(spec.atwood, spec.g, spec.lam, la.C_2D)
    u_lz = la.layzer_velocity(spec.atwood, spec.g, spec.lam, la.C_2D)
    fig, axes = plt.subplots(len(snaps), 2, figsize=(9.0, 3.4 * len(snaps)), squeeze=False)
    for row, s in zip(axes, snaps):
        phi_np, uy_np = s["phi"], s["uy"]
        ny, nx = phi_np.shape
        ext = (0.0, nx / spec.lam, y0 / spec.lam - ny / (2 * spec.lam),
               y0 / spec.lam + ny / (2 * spec.lam))
        im0 = row[0].imshow(phi_np, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, extent=ext,
                            aspect="auto")
        row[0].contour(np.linspace(ext[0], ext[1], nx), np.linspace(ext[2], ext[3], ny),
                       phi_np, levels=[0.0], colors="k", linewidths=0.6)
        x = np.linspace(ext[0], ext[1], nx)
        row[0].plot(x, s["zc_low"] / spec.lam, ".", ms=1.4, color="lime",
                    label="first crossing")
        row[0].plot(x, s["zc_top"] / spec.lam, ".", ms=1.4, color="magenta",
                    label="last crossing")
        row[0].set_title(fr"$\phi$ at $\tau={s['tau']:.2f}$: $h_b/\lambda={s['h_bub']:.2f}$, "
                         fr"$h_s/\lambda={s['h_spk']:.2f}$, {s['n_multi']} multi-valued cols",
                         fontsize=8)
        row[0].set_ylabel(r"$y/\lambda$ (from initial interface)")
        row[0].legend(fontsize=6, loc="lower left", ncol=2)
        plt.colorbar(im0, ax=row[0], fraction=0.03, pad=0.01)
        im1 = row[1].imshow(uy_np, origin="lower", cmap="PuOr_r", extent=ext, aspect="auto",
                            vmin=-2 * u_lz, vmax=2 * u_lz)
        row[1].set_title(rf"$u_y$ (colour scale $\pm2u_{{LZ}}$, $u_{{LZ}}={u_lz:.4f}$)", fontsize=8)
        row[1].set_ylabel(r"$y/\lambda$")
        plt.colorbar(im1, ax=row[1], fraction=0.03, pad=0.01)
        for ax in row:
            ax.set_xlabel(r"$x/\lambda$")
            ax.set_xlim(ext[0], ext[1])
    fig.suptitle(f"single-mode RT, $A=1/2$, $\\lambda=64$, $g=2\\times10^{{-4}}$; "
                 f"Layzer $Fr={fr_lz:.3f}$", fontsize=9)
    fig.tight_layout()
    png = BENCH_DIR / "figures" / "fig_late3_snapshots.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"wrote {png.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())