#!/usr/bin/env python3
"""GAP-DIFFLBM-ADJOINT-VERDICT: second, horizon-free anchor for the flow-response gradient.

The verdict case is fixed: objective = lift ``C_L``, parameter = disk centre
``y_c``.  Three gradient sources are compared against a finite-difference (FD)
reference of the *converged* solver:

``fd``
    central difference of the converged ``C_L(y_c)``, two step sizes
    ``h = 1.0, 0.5`` plus a Richardson extrapolation in ``h^2``.
``ad``
    :func:`tensorlbm.shape_gradient.shape_gradient_through_flow` -- unrolled /
    truncated-BPTT discrete adjoint, sweep over ``n_unroll``.
``fp``
    :func:`tensorlbm.fixed_point_adjoint.fixed_point_adjoint` -- implicit
    function theorem, ``(I - S_f)^T lambda = (dJ/df)^T`` solved matrix-free by
    GMRES.  No time horizon, no stored tape.
``frozen`` / ``ship``
    the frozen-field derivatives: the theoretical ceiling of any frozen-field
    method, and the shipped ``tensorlbm.adjoint`` surrogate folded over the
    boundary nodes.  Both must stay *outside* the optimisation path.

What the run delivers:

1. ``fp`` vs ``fd`` relative error and cosine, plus the ``ad``-vs-``fp`` cosine
   (two structurally different flow-response algorithms agreeing).
2. the ``n_unroll`` ladder of ``ad`` against ``fd`` and against ``fp`` -- the
   horizon truncation error of the unrolled route is *measured*, not assumed.
3. mirror symmetry ``C_L(y_c) = -C_L(H - y_c)``,
   ``dC_L/dy_c(y_c) = dC_L/dy_c(H - y_c)`` -- an analytic invariant, so it can
   be checked without any reference value.
4. a one-step optimisation: from ``y_c0`` every source steps ``|dy_c| = 2`` lu
   in the direction its gradient sign implies (lift maximisation), and the
   converged ``C_L`` is re-measured.  A sign error shows up as a trajectory
   that moves the wrong way.
5. the decomposition ``dJ/dp = dJ/dp|_f + lambda^T dS/dp``: the fixed-point
   adjoint reports its explicit and flow-response terms separately, and the
   explicit term must coincide with the frozen-field derivative.

Run::

    PYTHONPATH=src python benchmarks/adjoint_verdict_fixed_point.py \
        --mode quick --out artifacts/adjoint_verdict_fixed_point.json --figures
    PYTHONPATH=src python benchmarks/adjoint_verdict_fixed_point.py --mode paper ...

Artifacts: the JSON above, ``artifacts/adjoint_verdict_fixed_point/fields_*.npz``
and ``figures/adjoint_verdict_fixed_point_*.png`` (with ``--figures``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorlbm.adjoint import (  # noqa: E402
    FROZEN_FIELD_VERDICT,
    adjoint_sensitivity,
    optimization_path_eligible,
)
from tensorlbm.fixed_point_adjoint import fixed_point_adjoint, fixed_point_residual  # noqa: E402
from tensorlbm.shape_gradient import (  # noqa: E402
    ChannelDisk,
    disk_fluid_weight,
    frozen_field_gradient,
    make_channel,
    run_to_steady,
    shape_gradient_through_flow,
)

COS_GATE = 0.99


@dataclass
class Config:
    name: str
    nx: int
    ny: int
    diameter: float
    tau: float = 0.56
    u_bulk: float = 0.05
    epsilon: float = 0.5
    wall_thickness: int = 3
    yc_frac_offsets: tuple[float, ...] = (0.05,)
    unroll_sweep: tuple[int, ...] = (100, 200, 400)
    opt_step: float = 2.0
    # small steps: at h = 0.5 lu the *soft* disk moves by 1/6 of a radius and the
    # central difference of the converged lift is biased by ~1.5 %, which was
    # masked in the h = 1.0/0.5 pair of the first verdict run
    fd_steps: tuple[float, ...] = (0.25, 0.125)

    def height(self) -> float:
        return float(self.ny - 2 * self.wall_thickness)

    def y_mid(self) -> float:
        return (self.ny - 1) / 2.0


# Smoke scale (runs in ~1 minute of CPU, 5 cells across the disk radius): the
# algorithms, the gate and the symmetry invariant are resolution independent,
# the absolute C_L is not -- paper numbers come from PAPER_CONFIGS.
QUICK_CONFIGS = [
    Config("Q", nx=24, ny=16, diameter=3.0, epsilon=0.30, unroll_sweep=(100, 200, 400)),
]

PAPER_CONFIGS = [
    Config("R1", nx=96, ny=64, diameter=12.0, epsilon=0.6, unroll_sweep=(100, 200, 400)),
    Config("R2", nx=128, ny=96, diameter=16.0, epsilon=0.6, unroll_sweep=(100, 200, 400)),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def cosine(a, b) -> float:
    a = list(a) if isinstance(a, (list, tuple)) else [a]
    b = list(b) if isinstance(b, (list, tuple)) else [b]
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def build(cfg: Config, yc: float, xc: float | None = None) -> ChannelDisk:
    geom = make_channel(
        nx=cfg.nx,
        ny=cfg.ny,
        diameter=cfg.diameter,
        center_y=yc,
        tau=cfg.tau,
        u_bulk=cfg.u_bulk,
        epsilon=cfg.epsilon,
        wall_thickness=cfg.wall_thickness,
        center_x=xc,
    )
    return geom


def solve(cfg: Config, yc: float, args, xc: float | None = None):
    geom = build(cfg, yc, xc)
    t0 = time.time()
    st = run_to_steady(
        geom, max_steps=args.max_steps, tol=args.tol, min_steps=args.min_steps, check_every=100
    )
    st.wall = time.time() - t0  # type: ignore[attr-defined]
    return st


def cl_of(sol, u_ref: float) -> float:
    scale = 0.5 * u_ref * u_ref * float(sol.geom.diameter)
    return sol.lift / scale


def save_fields(sol, path: str) -> None:
    import numpy as np

    rho, ux, uy = sol.fields()
    w = disk_fluid_weight(sol.geom)
    np.savez_compressed(
        path,
        rho=rho.numpy(),
        ux=ux.numpy(),
        uy=uy.numpy(),
        solid=(1.0 - w).numpy(),
        wall=sol.geom.wall_mask().numpy(),
        y_c=float(sol.geom.center_y),
        diameter=float(sol.geom.diameter),
    )


def shipped_surrogate(sol, u_ref: float) -> dict:
    geom = sol.geom
    scale = 0.5 * u_ref * u_ref * float(geom.diameter)
    rho, ux, uy = sol.fields()
    out = adjoint_sensitivity(
        rho.detach(), ux.detach(), uy.detach(), geom.hard_mask(), objective="lift"
    )
    g = float(sum(out["sensitivity_y"])) / scale
    return {
        "dJ_dy_c": g,
        "grad": {"y_c": g},
        "grad_vector": [g],
        "shape_derivative_only": bool(out["shape_derivative_only"]),
        "includes_flow_response": bool(out["includes_flow_response"]),
        "optimization_path_eligible": optimization_path_eligible(out),
        "method": "adjoint.adjoint_sensitivity (frozen field, folded over boundary nodes)",
    }


# ---------------------------------------------------------------------------
# one configuration
# ---------------------------------------------------------------------------


def run_config(cfg: Config, args, out_dir: str) -> dict:
    H = cfg.height()
    y_mid = cfg.y_mid()
    delta = cfg.yc_frac_offsets[0] * H
    yc0 = y_mid + delta
    ref = build(cfg, yc0)
    u_ref = ref.bulk_velocity()

    rec: dict = {
        "name": cfg.name,
        "units": "lattice (D2Q9 BGK, Guo forcing, partially saturated bounce-back)",
        "config": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "diameter": cfg.diameter,
            "y_c": yc0,
            "H": H,
            "offset_over_H": delta / H,
            "tau": cfg.tau,
            "nu": ref.kinematic_viscosity(),
            "u_bulk": u_ref,
            "reynolds": ref.reynolds(),
            "epsilon": cfg.epsilon,
            "wall_thickness": cfg.wall_thickness,
            "blockage_D_over_H": cfg.diameter / H,
        },
        "sources": {},
        "mirror": {},
        "trajectory": {},
    }

    # ---- base converged solution ----------------------------------------
    base = solve(cfg, yc0, args)
    cl0 = cl_of(base, u_ref)
    rec["base"] = {
        "C_L": cl0,
        "C_D": base.drag / (0.5 * u_ref * u_ref * float(cfg.diameter)),
        "steps": base.steps,
        "residual": base.residual,
        "wall_seconds": round(getattr(base, "wall", 0.0), 1),
        "fixed_point_residual": fixed_point_residual(base),
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_fields(base, os.path.join(out_dir, f"fields_{cfg.name}.npz"))
    print(f"[{cfg.name}] steady {base.steps} steps C_L={cl0:.6e} (Re={ref.reynolds():.2f})", flush=True)

    # ---- FD reference ----------------------------------------------------
    fd = {}
    for h in cfg.fd_steps:
        cl_p = cl_of(solve(cfg, yc0 + h, args), u_ref)
        cl_m = cl_of(solve(cfg, yc0 - h, args), u_ref)
        fd[f"h={h}"] = {"C_L_plus": cl_p, "C_L_minus": cl_m, "dJ_dy_c": (cl_p - cl_m) / (2 * h)}
        print(f"[{cfg.name}] FD h={h}: {fd[f'h={h}']['dJ_dy_c']:.6e}", flush=True)
    hs = cfg.fd_steps
    if len(hs) >= 2:
        g1, g2 = fd[f"h={hs[0]}"]["dJ_dy_c"], fd[f"h={hs[1]}"]["dJ_dy_c"]
        f1, f2 = hs[0] ** 2, hs[1] ** 2
        g_ref = (g2 * f1 - g1 * f2) / (f1 - f2)
    else:
        g_ref = fd[f"h={hs[0]}"]["dJ_dy_c"]
    rec["fd"] = {"steps": fd, "reference": g_ref, "reference_vector": [g_ref]}
    print(f"[{cfg.name}] FD reference (Richardson h^2): {g_ref:.6e}", flush=True)

    # ---- unrolled AD ladder ---------------------------------------------
    lad = {}
    for n in cfg.unroll_sweep:
        ad = shape_gradient_through_flow(base, n_unroll=n, window=min(50, n), u_ref=u_ref)
        lad[f"n={n}"] = {
            "dJ_dy_c": ad["dJ_dy_c"],
            "rel_error_vs_fd": abs(ad["dJ_dy_c"] - g_ref) / abs(g_ref),
            "shape_derivative_only": ad["shape_derivative_only"],
            "optimization_path_eligible": ad["optimization_path_eligible"],
        }
        print(f"[{cfg.name}] AD n={n}: {ad['dJ_dy_c']:.6e} "
              f"(rel err vs FD {lad[f'n={n}']['rel_error_vs_fd']:.2e})", flush=True)
    rec["sources"]["ad"] = {
        "ladder": lad,
        "dJ_dy_c": lad[f"n={cfg.unroll_sweep[-1]}"]["dJ_dy_c"],
        "rel_error_vs_fd": lad[f"n={cfg.unroll_sweep[-1]}"]["rel_error_vs_fd"],
        "grad_vector": [lad[f"n={cfg.unroll_sweep[-1]}"]["dJ_dy_c"]],
        "optimization_path_eligible": True,
        "method": "unrolled AD (truncated BPTT, n_unroll=%d)" % cfg.unroll_sweep[-1],
    }

    # ---- fixed-point adjoint --------------------------------------------
    t0 = time.time()
    fp = fixed_point_adjoint(base, u_ref=u_ref)
    fp_wall = time.time() - t0
    fp["rel_error_vs_fd"] = abs(fp["dJ_dy_c"] - g_ref) / abs(g_ref)
    fp["cos_vs_fd"] = cosine(fp["dJ_dy_c"], g_ref)
    fp["cos_vs_ad"] = cosine(fp["dJ_dy_c"], rec["sources"]["ad"]["dJ_dy_c"])
    fp["rel_diff_vs_ad"] = abs(fp["dJ_dy_c"] - rec["sources"]["ad"]["dJ_dy_c"]) / abs(fp["dJ_dy_c"])
    fp["wall_seconds"] = round(fp_wall, 1)
    rec["sources"]["fp"] = fp
    print(f"[{cfg.name}] FP adjoint: {fp['dJ_dy_c']:.6e} (rel err vs FD {fp['rel_error_vs_fd']:.2e}, "
          f"cos vs AD {fp['cos_vs_ad']:+.5f}, krylov {fp['krylov_iterations']} matvecs, "
          f"rel res {fp['krylov_relative_residual']:.1e})", flush=True)

    # ---- frozen-field derivatives ---------------------------------------
    frozen = frozen_field_gradient(base, u_ref=u_ref)
    frozen["rel_error_vs_fd"] = abs(frozen["dJ_dy_c"] - g_ref) / abs(g_ref)
    frozen["cos_vs_fd"] = cosine(frozen["dJ_dy_c"], g_ref)
    frozen["cos_vs_fp"] = cosine(frozen["dJ_dy_c"], fp["dJ_dy_c"])
    frozen["amplitude_ratio_vs_fp"] = abs(frozen["dJ_dy_c"]) / abs(fp["dJ_dy_c"])
    rec["sources"]["frozen"] = frozen
    ship = shipped_surrogate(base, u_ref)
    ship["cos_vs_fp"] = cosine(ship["dJ_dy_c"], fp["dJ_dy_c"])
    ship["amplitude_ratio_vs_fp"] = abs(ship["dJ_dy_c"]) / abs(fp["dJ_dy_c"])
    ship["rel_error_vs_fd"] = abs(ship["dJ_dy_c"] - g_ref) / abs(g_ref)
    rec["sources"]["ship"] = ship
    print(f"[{cfg.name}] frozen: {frozen['dJ_dy_c']:.6e} cos_vs_fp={frozen['cos_vs_fp']:+.4f}; "
          f"ship: {ship['dJ_dy_c']:.6e} cos_vs_fp={ship['cos_vs_fp']:+.4f}", flush=True)

    # ---- decomposition: explicit term must equal the frozen derivative ---
    rec["decomposition"] = {
        "explicit_term": fp["explicit_term"]["y_c"],
        "frozen_field_derivative": frozen["dJ_dy_c"],
        "flow_response_term": fp["flow_response_term"]["y_c"],
        "total": fp["dJ_dy_c"],
        "flow_response_fraction": fp["flow_response_term"]["y_c"] / fp["dJ_dy_c"],
        "explicit_vs_frozen_rel_diff": abs(fp["explicit_term"]["y_c"] - frozen["dJ_dy_c"])
        / abs(frozen["dJ_dy_c"]),
    }

    # ---- mirror symmetry -------------------------------------------------
    mirror = solve(cfg, y_mid - delta, args)
    cl_mirror = cl_of(mirror, u_ref)
    fp_mirror = fixed_point_adjoint(mirror, u_ref=u_ref)
    rec["mirror"] = {
        "y_c_up": yc0,
        "y_c_dn": y_mid - delta,
        "C_L_up": cl0,
        "C_L_dn": cl_mirror,
        "C_L_oddness_rel": abs(cl0 + cl_mirror) / max(abs(cl0), abs(cl_mirror)),
        "g_up": fp["dJ_dy_c"],
        "g_dn": fp_mirror["dJ_dy_c"],
        "g_evenness_rel": abs(fp["dJ_dy_c"] - fp_mirror["dJ_dy_c"]) / abs(fp["dJ_dy_c"]),
    }
    print(f"[{cfg.name}] mirror: C_L oddness {rec['mirror']['C_L_oddness_rel']:.2e}, "
          f"g evenness {rec['mirror']['g_evenness_rel']:.2e}", flush=True)

    # ---- one-step optimisation (lift maximisation) -----------------------
    for key in ("fp", "ad", "frozen", "ship"):
        src = rec["sources"][key]
        g = src["dJ_dy_c"]
        direction = 1.0 if g >= 0.0 else -1.0
        yc_new = yc0 + direction * cfg.opt_step
        st = solve(cfg, yc_new, args)
        cl_new = cl_of(st, u_ref)
        rec["trajectory"][key] = {
            "y_c_from": yc0,
            "y_c_to": yc_new,
            "C_L_from": cl0,
            "C_L_to": cl_new,
            "delta_C_L": cl_new - cl0,
            "improved": bool(cl_new > cl0),
        }
        print(f"[{cfg.name}] traj[{key}] y_c {yc0:.2f}->{yc_new:.2f} "
              f"dC_L={cl_new - cl0:+.3e} improved={cl_new > cl0}", flush=True)

    # ---- judgement -------------------------------------------------------
    rec["verdict"] = {
        "cos_gate": COS_GATE,
        "fp_rel_error_vs_fd": fp["rel_error_vs_fd"],
        "fp_cos_vs_fd": fp["cos_vs_fd"],
        "fp_matches_fd": bool(fp["rel_error_vs_fd"] <= args.tol_grad),
        "fp_agrees_with_ad": bool(fp["cos_vs_ad"] >= COS_GATE),
        "fp_krylov_converged": bool(fp["krylov_relative_residual"] < 1e-8),
        "fp_eligible": bool(fp["optimization_path_eligible"]),
        "frozen_fails_gate": bool(rec["sources"]["frozen"]["cos_vs_fp"] < COS_GATE),
        "ship_fails_gate": bool(rec["sources"]["ship"]["cos_vs_fp"] < COS_GATE),
        "frozen_trajectory_wrong_way": bool(not rec["trajectory"]["frozen"]["improved"]),
        "fp_trajectory_improves": bool(rec["trajectory"]["fp"]["improved"]),
        "mirror_g_evenness_rel": rec["mirror"]["g_evenness_rel"],
        "mirror_ok": bool(rec["mirror"]["g_evenness_rel"] < 5e-2 and rec["mirror"]["C_L_oddness_rel"] < 5e-2),
        "declared_surrogate_verdict": FROZEN_FIELD_VERDICT,
    }
    return rec


# ---------------------------------------------------------------------------
# summary + figures
# ---------------------------------------------------------------------------


def summarise(payload: dict, tol_grad: float) -> dict:
    rows = []
    for name, rec in payload["configs"].items():
        rows.append(
            {
                "config": name,
                "reynolds": rec["config"]["reynolds"],
                "C_L": rec["base"]["C_L"],
                "g_fd": rec["fd"]["reference"],
                "g_ad": rec["sources"]["ad"]["dJ_dy_c"],
                "g_fp": rec["sources"]["fp"]["dJ_dy_c"],
                "g_frozen": rec["sources"]["frozen"]["dJ_dy_c"],
                "g_ship": rec["sources"]["ship"]["dJ_dy_c"],
                "fp_rel_err_vs_fd": rec["verdict"]["fp_rel_error_vs_fd"],
                "fp_cos_vs_fd": rec["verdict"]["fp_cos_vs_fd"],
                "fp_cos_vs_ad": rec["sources"]["fp"]["cos_vs_ad"],
                "frozen_cos_vs_fp": rec["sources"]["frozen"]["cos_vs_fp"],
                "ship_cos_vs_fp": rec["sources"]["ship"]["cos_vs_fp"],
                "frozen_amp_ratio": rec["sources"]["frozen"]["amplitude_ratio_vs_fp"],
                "traj_fp_improved": rec["trajectory"]["fp"]["improved"],
                "trajectory_frozen_wrong_way": rec["verdict"]["frozen_trajectory_wrong_way"],
                "mirror_g_evenness_rel": rec["verdict"]["mirror_g_evenness_rel"],
            }
        )
    ok = all(
        r["fp_rel_err_vs_fd"] <= tol_grad
        and r["fp_cos_vs_ad"] >= COS_GATE
        and r["frozen_cos_vs_fp"] < COS_GATE
        and r["trajectory_fp_improved"]
        and r["trajectory_frozen_wrong_way"]
        for r in rows
    )
    payload["summary"] = {
        "rows": rows,
        "tol_grad": tol_grad,
        "verdict": "PASS" if ok else "FAIL",
        "claim": (
            "the horizon-free fixed-point adjoint reproduces the converged finite-difference "
            "lifting gradient; the frozen-field surrogate stays anti-parallel and its "
            "optimisation step still moves the wrong way"
        ),
    }
    return payload


def make_figures(payload: dict, fig_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(fig_dir, exist_ok=True)
    written = []
    names = list(payload["configs"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    width = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        rec = payload["configs"][name]
        srcs = rec["sources"]
        vals = [srcs["fp"]["dJ_dy_c"], srcs["ad"]["dJ_dy_c"],
                srcs["frozen"]["dJ_dy_c"], srcs["ship"]["dJ_dy_c"]]
        x = np.arange(4) + i * width
        ax.bar(x, vals, width=width, label=name)
        ax.axhline(rec["fd"]["reference"], color="k", ls="--", lw=0.8,
                   label="FD reference" if i == 0 else None)
    ax.set_xticks(np.arange(4) + width * (len(names) - 1) / 2)
    ax.set_xticklabels(["fixed-point", "unrolled AD", "frozen", "shipped"])
    ax.set_ylabel(r"$dC_L/dy_c$")
    ax.set_title("gradient by source vs converged FD")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for name in names:
        rec = payload["configs"][name]
        lad = rec["sources"]["ad"]["ladder"]
        ns = [int(k.split("=")[1]) for k in lad]
        errs = [lad[k]["rel_error_vs_fd"] for k in lad]
        ax.loglog(ns, errs, "o-", label=f"{name} unrolled AD")
        ax.axhline(rec["sources"]["fp"]["rel_error_vs_fd"], ls=":",
                   label=f"{name} fixed-point (horizon-free)")
    ax.set_xlabel("unroll horizon $n$")
    ax.set_ylabel("relative error vs FD")
    ax.set_title("horizon truncation of the unrolled route")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    labels, vals, colors = [], [], []
    for name in names:
        rec = payload["configs"][name]
        labels += [f"{name} fp-vs-fd", f"{name} fp-vs-ad", f"{name} frozen-vs-fp"]
        vals += [rec["verdict"]["fp_cos_vs_fd"], rec["sources"]["fp"]["cos_vs_ad"],
                 rec["sources"]["frozen"]["cos_vs_fp"]]
        colors += ["tab:green", "tab:blue", "tab:red"]
    ax.bar(np.arange(len(vals)), vals, color=colors)
    ax.axhline(COS_GATE, color="k", ls="--", lw=1, label="gate cos = 0.99")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("cosine")
    ax.set_title("admission gate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(fig_dir, "adjoint_verdict_fixed_point_gradients.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # trajectory + mirror symmetry
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for name in names:
        rec = payload["configs"][name]
        for key, marker in (("fp", "o"), ("frozen", "x"), ("ship", "s")):
            t = rec["trajectory"][key]
            ax.plot([t["y_c_from"], t["y_c_to"]], [t["C_L_from"], t["C_L_to"]],
                    marker=marker, ls="-", label=f"{name} {key}")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("$y_c$ [lu]")
    ax.set_ylabel("$C_L$")
    ax.set_title("one optimisation step of size 2 lu (lift maximisation)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name in names:
        rec = payload["configs"][name]
        m = rec["mirror"]
        ax.plot([m["y_c_up"], m["y_c_dn"]], [m["C_L_up"], m["C_L_dn"]], "o-", label=f"{name} $C_L$")
        ax.plot([m["y_c_up"], m["y_c_dn"]], [m["g_up"], m["g_dn"]], "s--",
                label=f"{name} $dC_L/dy_c$")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("$y_c$ [lu]")
    ax.set_title("mirror symmetry: $C_L$ odd, $dC_L/dy_c$ even")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, "adjoint_verdict_fixed_point_trajectory.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # flow field of the base solution
    for name in names:
        npz = os.path.join(payload.get("field_dir", ""), f"fields_{name}.npz")
        if not os.path.exists(npz):
            continue
        d = np.load(npz)
        ux, uy = d["ux"], d["uy"]
        solid, wall = d["solid"], d["wall"]
        y0 = max(int(np.argmax(solid > 0.5) // ux.shape[1]) - 3, 0) if (solid > 0.5).any() else 0
        y1 = uy.shape[0]
        fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
        speed = np.hypot(ux, uy)
        for ax, fld, ttl in ((axes[0], speed, "|u|"), (axes[1], uy, "$u_y$")):
            im = ax.imshow(fld, cmap="viridis", aspect="equal", origin="lower")
            ax.contour(solid, levels=[0.5], colors="w", linewidths=0.8)
            ax.contour(wall.astype(float), levels=[0.5], colors="0.7", linewidths=0.6)
            ax.set_title(f"{name} {ttl}  (y_c={float(d['y_c']):.2f} lu)")
            fig.colorbar(im, ax=ax, fraction=0.03)
        axes[1].set_xlabel("x [lu]")
        fig.tight_layout()
        path = os.path.join(fig_dir, f"adjoint_verdict_fixed_point_field_{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("quick", "paper"), default="quick")
    ap.add_argument("--out", default="artifacts/adjoint_verdict_fixed_point.json")
    ap.add_argument("--only", default=None, help="comma separated config names")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--min-steps", type=int, default=None)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--tol-grad", type=float, default=0.05, help="max rel. error of fp vs FD")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)

    if args.max_steps is None:
        args.max_steps = 6000 if args.mode == "quick" else 40000
    if args.min_steps is None:
        args.min_steps = 400 if args.mode == "quick" else 1000
    configs = QUICK_CONFIGS if args.mode == "quick" else PAPER_CONFIGS
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        configs = [c for c in configs if c.name in keep]
    field_dir = os.path.splitext(args.out)[0]
    payload = {
        "benchmark": "adjoint_verdict_fixed_point",
        "anchor": "GAP-DIFFLBM-ADJOINT-VERDICT",
        "mode": args.mode,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "hostname": platform.node(),
            "threads": args.threads,
            "dtype": str(torch.get_default_dtype()),
        },
        "configs": {},
        "field_dir": field_dir,
    }
    t_all = time.time()
    for cfg in configs:
        print(f"=== {cfg.name} ===", flush=True)
        t0 = time.time()
        payload["configs"][cfg.name] = run_config(cfg, args, field_dir)
        payload["configs"][cfg.name]["wall_seconds"] = round(time.time() - t0, 1)
    payload = summarise(payload, args.tol_grad)
    payload["wall_seconds_total"] = round(time.time() - t_all, 1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}  verdict={payload['summary']['verdict']}", flush=True)

    if args.figures:
        figs = make_figures(payload, "figures")
        payload["figures"] = figs
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print("figures:", figs, flush=True)

    return 0 if payload["summary"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())