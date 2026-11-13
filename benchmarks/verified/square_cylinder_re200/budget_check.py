#!/usr/bin/env python3
"""Independent Cd cross-check for the square-cylinder case: control-volume
momentum balance vs. the Ladd momentum-exchange force sum.

The case driver (``run.py``) reports Cd from ``compute_obstacle_forces`` only.
This instrument accumulates the time mean of the momentum-flux components
``rho*u_x*u_x``, ``rho*u_x*u_y`` and of the pressure ``p = rho/3`` over whole
shedding periods and re-derives the drag from the CV balance
(``tensorlbm.drag_budget.momentum_balance_drag``).  The two estimators share
the collide/stream/boundary chain but *not* the force bookkeeping, so their
agreement is a genuine check of the force instrumentation.

Usage:
    PYTHONPATH=src python benchmarks/verified/square_cylinder_re200/budget_check.py \
        --D 16 --re 200 --steps 24000 --blockage 20 --threads 8 \
        --out runs/m17121_re200/B20_D16
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.boundaries import (  # noqa: E402
    compute_obstacle_forces,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.drag_budget import momentum_balance_drag  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402
from tensorlbm.square_cylinder import (  # noqa: E402
    blockage_ratio,
    drag_statistics,
    force_coefficients,
    rect_mask_2d,
    strouhal_from_series,
)

CASE = "CASE-LBM-SQUARE-CYL-NORBERG"


def run(D: int, re: float, u_in: float, steps: int, *, blockage: float = 20.0,
        len_up_d: float = 8.0, len_down_d: float = 18.0, sponge_d: float = 6.0,
        sponge_alpha: float = 10.0, st_seed: float = 0.16, seed_periods: float = 2.0,
        seed_ampl: float = 0.10, warmup_frac: float = 0.5,
        box_up_d: float = 2.0, box_down_d: float = 5.0,
        sample_every: int = 2, device: str = "cpu") -> dict:
    dev = torch.device(device)
    nx = int(round((len_up_d + len_down_d) * D))
    ny = int(round(blockage * D))
    nu = u_in * D / re
    tau = 0.5 + 3.0 * nu

    mask = rect_mask_2d(nx, ny, len_up_d * D, ny / 2.0, D, device=dev)
    sigma = make_sponge_strength(ny, nx, int(nx - sponge_d * D), int(sponge_d * D), power=2.0, device=dev)
    tau_field = tau * (1.0 + sponge_alpha * sigma)

    rho0 = torch.ones((ny, nx), device=dev)
    f = equilibrium(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0))

    omega_seed = 2.0 * math.pi * st_seed * u_in / D
    seed_steps = int(round(seed_periods * D / (st_seed * u_in)))
    uy_amp = seed_ampl * u_in

    fx = np.zeros(steps)
    fy = np.zeros(steps)
    w0 = int(warmup_frac * steps)
    acc_uux = np.zeros((ny, nx))
    acc_uxuy = np.zeros((ny, nx))
    acc_p = np.zeros((ny, nx))
    n_samp = 0
    t0 = time.time()
    for step in range(1, steps + 1):
        f = stream(collide_mrt(f, tau, tau_field=tau_field))
        fx_k, fy_k = compute_obstacle_forces(f, mask)  # post-stream, pre bounce-back
        f = far_field_bc_2d(f, u_in, mask)
        if step <= seed_steps:
            uy_col = torch.full((ny,), uy_amp * math.sin(omega_seed * step), device=dev, dtype=f.dtype)
            f[:, :, 0] = equilibrium(
                torch.ones((ny, 1), device=dev, dtype=f.dtype),
                torch.full((ny, 1), u_in, device=dev, dtype=f.dtype),
                uy_col.view(ny, 1),
            )[:, :, 0]
        fx[step - 1] = float(fx_k)
        fy[step - 1] = float(fy_k)
        if step > w0 and (step - w0) % sample_every == 0:
            rho, ux, uy = macroscopic(f)
            rho_n = rho.cpu().numpy(); ux_n = ux.cpu().numpy(); uy_n = uy.cpu().numpy()
            acc_uux += rho_n * ux_n * ux_n
            acc_uxuy += rho_n * ux_n * uy_n
            acc_p += rho_n / 3.0
            n_samp += 1
    elapsed = time.time() - t0

    cd, cl = force_coefficients(fx, fy, u_in, D)
    drag = drag_statistics(cd, w0)
    st = strouhal_from_series(cl[w0:], u_in, D)
    rho, ux, uy = macroscopic(f)
    vort = (torch.gradient(uy, dim=1)[0] - torch.gradient(ux, dim=0)[0]).cpu().numpy()

    # control volume: everything upstream of the sponge and short of the outlet
    i_body0 = int(len_up_d * D)
    i0 = max(2, i_body0 - int(box_up_d * D))
    i1 = min(int(nx - sponge_d * D) - 1, i_body0 + D + int(box_down_d * D))
    j0, j1 = 0, ny - 1
    m_uux = acc_uux[j0:j1 + 1, i0:i1 + 1] / n_samp
    m_uxuy = acc_uxuy[j0:j1 + 1, i0:i1 + 1] / n_samp
    m_p = acc_p[j0:j1 + 1, i0:i1 + 1] / n_samp
    budget = momentum_balance_drag(m_uux, m_uxuy, m_p, u_in, float(D))
    cd_mom = budget["cd"]
    rel_pct = abs(cd_mom - drag["mean"]) / abs(drag["mean"]) * 100.0 if drag["mean"] else float("nan")

    return {
        "case": CASE,
        "D": D, "re": re, "u_in": u_in, "steps": steps, "analyze_from": w0,
        "n_samples": n_samp, "sample_every": sample_every,
        "blockage_measured": blockage_ratio(mask),
        "grid": {"nx": nx, "ny": ny},
        "cv_box": {"i0": i0, "i1": i1, "j0": j0, "j1": j1,
                   "upstream_D": (i_body0 - i0) / D, "downstream_D": (i1 - i_body0 - D) / D},
        "cd_ladd": drag["mean"], "cd_rms": drag["rms"],
        "cd_cv": cd_mom, "cv_faces": budget["faces"],
        "rel_diff_pct": rel_pct,
        "st_ladd": st["st"], "st_fft": st["fft"], "st_crossing": st["crossing"],
        "elapsed_s": round(elapsed, 1), "threads": torch.get_num_threads(),
        "_series": {"cd": cd, "cl": cl, "ux_mean": acc_uux / n_samp, "vorticity": vort},
    }


def make_figure(rec: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = rec["_series"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(s["cd"], lw=0.7, label="Cd(t)")
    ax[0].plot(s["cl"], lw=0.7, label="Cl(t)")
    ax[0].axvline(rec["analyze_from"], color="k", ls="--", lw=1, label="averaging window start")
    ax[0].set_xlabel("lattice step"); ax[0].set_ylabel("coefficient")
    ax[0].set_title(f"D={rec['D']} Re={rec['re']:g}: forces (Cd_ladd={rec['cd_ladd']:.4f}, "
                    f"Cd_CV={rec['cd_cv']:.4f})")
    ax[0].legend(fontsize=7)

    im = ax[1].imshow(s["vorticity"], cmap="RdBu_r", origin="lower",
                      vmin=-np.abs(s["vorticity"]).max() * 0.3, vmax=np.abs(s["vorticity"]).max() * 0.3)
    b = rec["cv_box"]
    ax[1].add_patch(plt.Rectangle((b["i0"], b["j0"]), b["i1"] - b["i0"], b["j1"] - b["j0"],
                                  fill=False, ec="k", lw=1.2, ls="--"))
    ax[1].set_title("vorticity + control volume")
    fig.colorbar(im, ax=ax[1], fraction=0.03)
    fig.tight_layout()
    png = out / "budget_fig.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--re", type=float, default=200.0)
    ap.add_argument("--u-in", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--periods", type=float, default=12.0)
    ap.add_argument("--blockage", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    steps = args.steps or int(round(args.periods * args.D / (0.16 * args.u_in)))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rec = run(args.D, args.re, args.u_in, steps, blockage=args.blockage)
    if not np.isfinite(rec["cd_ladd"]):
        raise SystemExit("non-finite Cd")
    series = rec.pop("_series")
    np.savez_compressed(out / "budget_series.npz", **series)
    png = make_figure({**rec, "_series": series}, out)
    rec["figure"] = str(png)
    (out / "budget.json").write_text(json.dumps(rec, indent=2))
    print(f"[D={rec['D']}] Cd_ladd={rec['cd_ladd']:.4f} Cd_CV={rec['cd_cv']:.4f} "
          f"(rel {rec['rel_diff_pct']:.2f}%) St={rec['st_ladd']:.4f} "
          f"B={rec['blockage_measured']:.4f} t={rec['elapsed_s']:.0f}s -> {out/'budget.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
