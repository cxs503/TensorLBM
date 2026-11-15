#!/usr/bin/env python3
"""GAP-DIFFLBM-ADJOINT-VERDICT: horizon-free anchor for the lifting gradient (study scale).

Small, fast, fully reproducible study of the *verification debt* left by the
first verdict run (``benchmarks/adjoint_verdict_diff_lbm.py``):

1. **FD step-size study.**  The first verdict used central differences with
   ``h = 1.0, 0.5`` lattice units on a *soft* (sdf-smeared) disk.  Moving the
   disk by half a lattice unit already moves the smeared interface by a sixth
   of a radius, so the h-ladder is queried down to ``h = 1/16`` to show the
   finite-difference reference itself converging -- the large-``h`` values are
   biased and cannot be used as an anchor.
2. **Horizon ladder of the unrolled AD** against that converged reference.
3. **Fixed-point adjoint** (:func:`tensorlbm.fixed_point_adjoint.fixed_point_adjoint`):
   implicit-function theorem + matrix-free GMRES, no time horizon, no tape.
   It must land on the same number as the converged FD and the long-horizon AD.
4. **Frozen-field derivative** (:func:`tensorlbm.shape_gradient.frozen_field_gradient`):
   must stay anti-parallel / outside the optimisation path, and its explicit
   term must equal ``dJ/dp|_f`` reported separately by the fixed-point adjoint.

Run::

    PYTHONPATH=src python benchmarks/adjoint_verdict_fixed_point_study.py \
        --out artifacts/adjoint_verdict_fixed_point_study.json --figures

Scale: H = 10 lu, D = 3 lu, eps = 0.3, tau = 0.6, Re = 7.5 -- a *study* grid
(5 cells across the radius).  The algorithms, the gate and the numbers'
consistency are resolution independent; the absolute C_L is not.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorlbm.fixed_point_adjoint import fixed_point_adjoint  # noqa: E402
from tensorlbm.shape_gradient import (  # noqa: E402
    disk_fluid_weight,
    frozen_field_gradient,
    make_channel,
    run_to_steady,
    shape_gradient_through_flow,
)

COS_GATE = 0.99
GEOM = dict(nx=24, ny=16, diameter=3.0, tau=0.6, u_bulk=0.05, epsilon=0.3, wall_thickness=3)
H = GEOM["ny"] - 2 * GEOM["wall_thickness"]
Y_MID = (GEOM["ny"] - 1) / 2.0
OFFSET_FRAC = 0.04
FD_LADDER = (0.5, 0.25, 0.125, 0.0625)
UNROLL = (100, 200, 400)
SOLVE = dict(max_steps=6000, tol=1e-10, min_steps=400, check_every=100)


def build(yc: float):
    return make_channel(
        nx=GEOM["nx"], ny=GEOM["ny"], diameter=GEOM["diameter"], center_y=yc,
        tau=GEOM["tau"], u_bulk=GEOM["u_bulk"], epsilon=GEOM["epsilon"],
        wall_thickness=GEOM["wall_thickness"],
    )


def solve(yc: float):
    t0 = time.time()
    st = run_to_steady(build(yc), **SOLVE)
    st.wall = time.time() - t0  # type: ignore[attr-defined]
    return st


def cosine(a, b) -> float:
    a = list(a) if isinstance(a, (list, tuple)) else [a]
    b = list(b) if isinstance(b, (list, tuple)) else [b]
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else sum(x * y for x, y in zip(a, b)) / (na * nb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/adjoint_verdict_fixed_point_study.json")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                    help="reuse the cached base state / FD ladder / AD ladder of an earlier run")
    ap.add_argument("--fp-rtol", type=float, default=1e-6)
    ap.add_argument("--fp-max-matvecs", type=int, default=800)
    args = ap.parse_args()

    cache_dir = os.path.splitext(args.out)[0] + "_cache"
    os.makedirs(cache_dir, exist_ok=True)
    fd_cache = os.path.join(cache_dir, "fd.json")
    ad_cache = os.path.join(cache_dir, "ad.json")
    state_cache = os.path.join(cache_dir, "base.pt")
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)

    yc0 = Y_MID + OFFSET_FRAC * H
    t_all = time.time()

    # ---- 1. base + FD step-size ladder ----------------------------------
    from tensorlbm.shape_gradient import SteadySolution

    if args.resume and os.path.exists(state_cache):
        blob = torch.load(state_cache, weights_only=False)
        geom = build(float(blob["y_c"]))
        base = SteadySolution(f=blob["f"], geom=geom, lift=float(blob["lift"]),
                              drag=float(blob["drag"]), steps=int(blob["steps"]),
                              residual=float(blob["residual"]), history=[])
        base.wall = 0.0  # type: ignore[attr-defined]
        print(f"base: loaded from cache (C_L={base.lift / (0.5 * geom.bulk_velocity() ** 2 * float(geom.diameter)):.6e})", flush=True)
    else:
        base = solve(yc0)
    u_ref = base.geom.bulk_velocity()
    scale = 0.5 * u_ref * u_ref * float(base.geom.diameter)
    cl0 = base.lift / scale
    print(f"base: steps={base.steps} res={base.residual:.1e} C_L={cl0:.6e} "
          f"Re={base.geom.reynolds():.2f} ({base.wall:.1f}s)", flush=True)

    fd = {}
    if args.resume and os.path.exists(fd_cache):
        fd = json.load(open(fd_cache))
        print(f"FD ladder: loaded {len(fd)} steps from cache", flush=True)
    for h in ([] if fd else FD_LADDER):
        sp = solve(yc0 + h)
        sm = solve(yc0 - h)
        g = ((sp.lift - sm.lift) / (2 * h)) / scale
        fd[f"h={h}"] = {
            "dJ_dy_c": g,
            "C_L_plus": sp.lift / scale,
            "C_L_minus": sm.lift / scale,
            "steps_plus": sp.steps,
            "steps_minus": sm.steps,
        }
        print(f"FD h={h}: {g:.6e}", flush=True)
    if not os.path.exists(fd_cache):
        json.dump(fd, open(fd_cache, "w"), indent=2)
        torch.save({"f": base.f, "y_c": float(base.geom.center_y), "lift": base.lift,
                    "drag": base.drag, "steps": base.steps, "residual": base.residual},
                   state_cache)
    h1, h2 = FD_LADDER[-2], FD_LADDER[-1]
    g1, g2 = fd[f"h={h1}"]["dJ_dy_c"], fd[f"h={h2}"]["dJ_dy_c"]
    g_ref = (4 * g2 - g1) / 3.0  # Richardson in h^2 on the two smallest steps
    print(f"FD reference (Richardson on h={h2},{h1}): {g_ref:.6e}", flush=True)

    # ---- 2/3/4. AD ladder, fixed-point adjoint, frozen proxy -------------
    ad = {}
    if args.resume and os.path.exists(ad_cache):
        ad = json.load(open(ad_cache))
        print(f"AD ladder: loaded {len(ad)} horizons from cache", flush=True)
    for n in ([] if ad else UNROLL):
        t0 = time.time()
        r = shape_gradient_through_flow(base, n_unroll=n, window=min(50, n), u_ref=u_ref)
        ad[f"n={n}"] = {
            "dJ_dy_c": r["dJ_dy_c"],
            "rel_error_vs_fd": abs(r["dJ_dy_c"] - g_ref) / abs(g_ref),
            "wall_seconds": round(time.time() - t0, 1),
        }
        print(f"AD n={n}: {r['dJ_dy_c']:.6e} (rel err {ad[f'n={n}']['rel_error_vs_fd']:.2e})", flush=True)

    if not os.path.exists(ad_cache):
        json.dump(ad, open(ad_cache, "w"), indent=2)

    t0 = time.time()
    fp = fixed_point_adjoint(base, u_ref=u_ref, rtol=args.fp_rtol,
                             restart=200, max_matvecs=args.fp_max_matvecs)
    fp["wall_seconds"] = round(time.time() - t0, 1)
    fp["rel_error_vs_fd"] = abs(fp["dJ_dy_c"] - g_ref) / abs(g_ref)
    fp["cos_vs_fd"] = cosine(fp["dJ_dy_c"], g_ref)
    fp["cos_vs_ad"] = cosine(fp["dJ_dy_c"], ad[f"n={UNROLL[-1]}"]["dJ_dy_c"])
    fp["rel_diff_vs_ad"] = abs(fp["dJ_dy_c"] - ad[f"n={UNROLL[-1]}"]["dJ_dy_c"]) / abs(fp["dJ_dy_c"])
    print(f"FP: {fp['dJ_dy_c']:.6e} (rel err vs FD {fp['rel_error_vs_fd']:.2e}, "
          f"krylov {fp['krylov_iterations']} matvecs, relres {fp['krylov_relative_residual']:.1e}, "
          f"cos vs AD {fp['cos_vs_ad']:+.6f})", flush=True)

    frozen = frozen_field_gradient(base, u_ref=u_ref)
    frozen["cos_vs_fp"] = cosine(frozen["dJ_dy_c"], fp["dJ_dy_c"])
    frozen["amplitude_ratio_vs_fp"] = abs(frozen["dJ_dy_c"]) / abs(fp["dJ_dy_c"])
    frozen["rel_error_vs_fd"] = abs(frozen["dJ_dy_c"] - g_ref) / abs(g_ref)
    print(f"frozen: {frozen['dJ_dy_c']:.6e} cos_vs_fp={frozen['cos_vs_fp']:+.4f} "
          f"ratio={frozen['amplitude_ratio_vs_fp']:.3f}", flush=True)

    # ---- field snapshot for the report ----------------------------------
    rho, ux, uy = base.fields()
    w = disk_fluid_weight(base.geom)
    import numpy as np

    out_dir = os.path.splitext(args.out)[0]
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "fields.npz"),
        ux=ux.numpy(), uy=uy.numpy(), rho=rho.numpy(),
        solid=(1.0 - w).numpy(), wall=base.geom.wall_mask().numpy(),
        y_c=float(base.geom.center_y), diameter=float(base.geom.diameter),
    )

    payload = {
        "benchmark": "adjoint_verdict_fixed_point_study",
        "anchor": "GAP-DIFFLBM-ADJOINT-VERDICT",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "hostname": platform.node(), "threads": args.threads,
        },
        "case": {
            "geom": GEOM, "H": H, "y_c": yc0, "offset_over_H": OFFSET_FRAC,
            "reynolds": base.geom.reynolds(), "u_bulk": u_ref,
            "nu": base.geom.kinematic_viscosity(), "cells_per_radius": (GEOM["diameter"] / 2) / GEOM["epsilon"],
            "objective": "lift C_L", "parameter": "y_c", "shape": "smooth (sdf) disk, eps = %.2f lu" % GEOM["epsilon"],
            "solve": SOLVE,
        },
        "base": {"C_L": cl0, "C_D": base.drag / scale, "steps": base.steps,
                 "residual": base.residual, "wall_seconds": round(base.wall, 1)},
        "fd": {"ladder": fd, "reference": g_ref,
               "reference_formula": "Richardson in h^2 on the two smallest steps"},
        "ad": ad,
        "fp": fp,
        "frozen": frozen,
        "decomposition": {
            "explicit_term": fp["explicit_term"]["y_c"],
            "frozen_field_derivative": frozen["dJ_dy_c"],
            "flow_response_term": fp["flow_response_term"]["y_c"],
            "total": fp["dJ_dy_c"],
            "explicit_vs_frozen_rel_diff": abs(fp["explicit_term"]["y_c"] - frozen["dJ_dy_c"])
            / abs(frozen["dJ_dy_c"]),
        },
        "verdict": {
            "cos_gate": COS_GATE,
            "fp_rtol": args.fp_rtol,
            "fp_matches_fd": bool(fp["rel_error_vs_fd"] <= 0.02),
            "fp_agrees_with_ad": bool(fp["cos_vs_ad"] >= COS_GATE),
            "fp_krylov_converged": bool(fp["krylov_relative_residual"] < 1e-6),
            "fp_eligible": bool(fp["optimization_path_eligible"]),
            "frozen_fails_gate": bool(frozen["cos_vs_fp"] < COS_GATE),
            "fd_large_h_bias_rel": abs(fd[f"h={FD_LADDER[0]}"]["dJ_dy_c"] - g_ref) / abs(g_ref),
            "fd_convergence_monotone": bool(
                abs(fd[f"h={FD_LADDER[-1]}"]["dJ_dy_c"] - g_ref)
                < abs(fd[f"h={FD_LADDER[0]}"]["dJ_dy_c"] - g_ref)
            ),
        },
    }
    payload["wall_seconds_total"] = round(time.time() - t_all, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out} ({payload['wall_seconds_total']:.0f}s)", flush=True)

    if args.figures:
        figs = make_figures(payload, out_dir, "figures")
        payload["figures"] = figs
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print("figures:", figs, flush=True)
    return 0


def make_figures(payload: dict, data_dir: str, fig_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(fig_dir, exist_ok=True)
    written = []
    g_ref = payload["fd"]["reference"]
    g_fp = payload["fp"]["dJ_dy_c"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    hs = [float(k.split("=")[1]) for k in payload["fd"]["ladder"]]
    gs = [payload["fd"]["ladder"][k]["dJ_dy_c"] for k in payload["fd"]["ladder"]]
    order = np.argsort(hs)
    hs = [hs[i] for i in order]
    gs = [gs[i] for i in order]
    ax.semilogx(hs, gs, "o-", label="central FD of converged $C_L$")
    ax.axhline(g_ref, color="C1", ls="--", label="Richardson (two smallest $h$)")
    ax.axhline(g_fp, color="C2", ls=":", label="fixed-point adjoint")
    ax.set_xlabel("FD step $h$ [lu]")
    ax.set_ylabel("$dC_L/dy_c$")
    ax.set_title("the FD reference itself needs $h\\to0$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ns = [int(k.split("=")[1]) for k in payload["ad"]]
    err = [payload["ad"][k]["rel_error_vs_fd"] for k in payload["ad"]]
    ax.loglog(ns, err, "o-", label="unrolled AD (truncated BPTT)")
    ax.axhline(payload["fp"]["rel_error_vs_fd"], color="C2", ls=":",
               label="fixed-point adjoint (horizon-free)")
    ax.set_xlabel("unroll horizon $n$")
    ax.set_ylabel("relative error vs FD reference")
    ax.set_title("horizon truncation of the unrolled route")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    names = ["fixed-point\nadjoint", "unrolled AD\n$n=%d$" % ns[-1], "frozen-field\nproxy"]
    vals = [g_fp, payload["ad"]["n=%d" % ns[-1]]["dJ_dy_c"], payload["frozen"]["dJ_dy_c"]]
    colors = ["C2", "C0", "C3"]
    ax.bar(names, vals, color=colors)
    ax.axhline(g_ref, color="k", ls="--", lw=1, label="FD reference")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel("$dC_L/dy_c$")
    ax.set_title("flow-response vs frozen-field (sign matters)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(fig_dir, "adjoint_verdict_fixed_point_study_gradients.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    d = np.load(os.path.join(data_dir, "fields.npz"))
    ux, uy, solid, wall = d["ux"], d["uy"], d["solid"], d["wall"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.0), sharex=True)
    for ax, fld, ttl in ((axes[0], np.hypot(ux, uy), "$|u|$"), (axes[1], uy, "$u_y$")):
        im = ax.imshow(fld, cmap="viridis", aspect="equal", origin="lower")
        ax.contour(solid, levels=[0.5], colors="w", linewidths=0.8)
        ax.contour(wall.astype(float), levels=[0.5], colors="0.8", linewidths=0.6)
        ax.set_title(f"study case ($y_c$={float(d['y_c']):.2f} lu, D={float(d['diameter']):.1f} lu) {ttl}")
        fig.colorbar(im, ax=ax, fraction=0.03)
    axes[1].set_xlabel("x [lu]")
    fig.tight_layout()
    p = os.path.join(fig_dir, "adjoint_verdict_fixed_point_study_field.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)
    return written


if __name__ == "__main__":
    raise SystemExit(main())