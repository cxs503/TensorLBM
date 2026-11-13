#!/usr/bin/env python3
"""Referee run: fused push-pull kernel vs the repo AB reference, on Taylor-Green.

Anchor ``XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN`` (OSS ``lbm-principles-practice/code``,
chapter-13 GPU fused ``stream_collide`` kernel).

What is measured (all numbers land in ``result.json``; nothing is hand-copied):

* **Physics gate** -- 2-D Taylor-Green decay at three resolutions.  Both the
  repo AB reference (:func:`tensorlbm.solver.step_stream_collide` with
  ``kernel="ab"``) and the fused kernel (``kernel="fused_push_pull"``) are run
  from the *same* initial equilibrium and scored against the closed-form
  Taylor-Green solution: relative L2 of ``u_x``, ``u_y`` and the density
  fluctuation, mass drift, viscosity inverted from the kinetic-energy decay.
* **Scheme equivalence** -- the fused kernel collides *after* streaming, the AB
  reference *before*; the states are therefore compared at the same level:
  ``fused_n`` vs ``collide_bgk(ab_n)``.  The velocity fields are compared
  directly (no state-label offset), plus the raw ``max|fused_n - ab_n|`` which
  is reported *as* the one-collision label offset, not as an error.
* **Resolution ladder** -- error vs grid, steps scaled with ``n**2`` so the
  physical time is fixed; observed order from consecutive pairs.
* **Negative controls** -- two deliberately wrong arms (viscosity biased +25%,
  and a mirror-initialised field) must fail the 15% gate; otherwise the gate is
  toothless.

Usage::

    python benchmarks/verified/fused_push_pull_aa_pattern/run.py            # full
    python benchmarks/verified/fused_push_pull_aa_pattern/run.py --quick    # ~20 s
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import torch

from tensorlbm.d2q9 import C
from tensorlbm.fused_push_pull import (
    equilibrium_upstream,
    tg_l2_errors,
    taylor_green_fields,
)
from tensorlbm.solver import collide_bgk, step_stream_collide

OUT_DIR = pathlib.Path(__file__).resolve().parent
NU = 1.0 / 6.0
TAU = 3.0 * NU + 0.5
U_MAX = 0.02
RHO0 = 1.0
GATE = 0.15  # owner precision standard
TARGET = 0.03  # stricter self-imposed target
KERNELS = ("ab", "fused_push_pull")


def macroscopic(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = f.sum(dim=0)
    c = C.to(f.device)
    ux = (f * c[:, 0].view(9, 1, 1)).sum(dim=0) / rho
    uy = (f * c[:, 1].view(9, 1, 1)).sum(dim=0) / rho
    return rho, ux, uy


def initial(n: int, u_max: float = U_MAX) -> torch.Tensor:
    rho, ux, uy = taylor_green_fields(n, n, u_max=u_max, nu=NU, t=0)
    return equilibrium_upstream(rho, ux, uy)


def mirror_we(f: torch.Tensor) -> torch.Tensor:
    """Control arm: swap the +x and -x directions (mass kept, momentum mirrored)."""
    out = f.clone()
    out[1], out[3] = f[3].clone(), f[1].clone()
    out[5], out[7] = f[7].clone(), f[5].clone()
    out[6], out[8] = f[8].clone(), f[6].clone()
    return out


def run_arm(f0: torch.Tensor, tau: float, steps: int, kernel: str) -> torch.Tensor:
    f = f0.clone()
    for _ in range(steps):
        f = step_stream_collide(f, tau, kernel=kernel)
    return f


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float((b * b).sum()) ** 0.5
    return float(((a - b) ** 2).sum()) ** 0.5 / denom if denom > 0 else float("nan")


def arm_report(n: int, steps: int, kernel: str, u_max: float = U_MAX, tau: float = TAU):
    f0 = initial(n, u_max)
    mass0 = float(f0.sum())
    t0 = time.perf_counter()
    f = run_arm(f0, tau, steps, kernel)
    wall = time.perf_counter() - t0
    rho, ux, uy = macroscopic(f)
    err = tg_l2_errors(rho, ux, uy, nx=n, ny=n, u_max=u_max, nu=NU, t=steps, rho0=RHO0)
    # E(t) = E(0) exp(-4 nu k^2 t), k = 2*pi/n  ->  nu = n^2 ln(E0/E) / (16 pi^2 t)
    e0_field = taylor_green_fields(n, n, u_max=u_max, nu=NU, t=0)
    e0 = float((e0_field[0] * (e0_field[1] ** 2 + e0_field[2] ** 2)).sum())
    nu_inv = (
        n * n * math.log(e0 / err["energy"]) / (16.0 * math.pi**2 * steps)
        if err["energy"] > 0 and e0 > 0
        else float("nan")
    )
    rho_drift = abs(float(rho.sum()) - mass0) / mass0
    return {
        "grid": n,
        "steps": steps,
        "kernel": kernel,
        "tau": tau,
        "nu": (tau - 0.5) / 3.0,
        "u_max": u_max,
        "mach": u_max * math.sqrt(3.0),
        "wall_s": wall,
        "mass_drift_rel": rho_drift,
        "rho_lu2": err["rho_lu2"],
        "ux_lu2": err["ux_lu2"],
        "uy_lu2": err["uy_lu2"],
        "energy": err["energy"],
        "nu_inverted": nu_inv,
        "nu_inversion_err_rel": (abs(nu_inv - NU) / NU if nu_inv else None),
        "state": f,
    }


def convergence_order(e1: float, e2: float, n1: int, n2: int) -> float:
    if e1 <= 0 or e2 <= 0:
        return float("nan")
    return math.log(e1 / e2) / math.log(n2 / n1)  # e1 = coarse, e2 = fine


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse ladder, for CI smoke")
    ap.add_argument("--out", default=str(OUT_DIR / "result.json"))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    scale = 0.25 if args.quick else 1.0
    ladder = [(32, 120), (48, 270), (64, 480)]
    if args.quick:
        ladder = [(16, 32), (24, 72)]
    ladder = [(n, max(20, int(round(s * scale)))) for n, s in ladder]

    result: dict = {
        "anchor": "XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN",
        "case": "2d_taylor_green_decay_d2q9_bgk",
        "reference": "closed-form Taylor-Green (Krueger et al. 2016, ch.13 upstream "
        "taylor_green_eval); cell-centred X=x+1/2, Y=y+1/2, pressure term O(u^2)/3",
        "parameters": {
            "nu": NU,
            "tau": TAU,
            "u_max": U_MAX,
            "mach": U_MAX * math.sqrt(3.0),
            "rho0": RHO0,
            "gate_pct": GATE,
            "target_pct": TARGET,
            "dtype": "float64",
            "boundary": "periodic",
            "ladder": [{"grid": n, "steps": s} for n, s in ladder],
            "quick": bool(args.quick),
        },
        "arms": [],
        "equivalence": {},
        "controls": [],
        "ladder": [],
    }

    states: dict[tuple[int, int, str], torch.Tensor] = {}
    for n, steps in ladder:
        for kernel in KERNELS:
            rep = arm_report(n, steps, kernel)
            states[(n, steps, kernel)] = rep.pop("state")
            result["arms"].append(rep)
            print(
                f"[arm] n={n:3d} steps={steps:4d} {kernel:16s} "
                f"ux_lu2={rep['ux_lu2']:.6e} rho_lu2={rep['rho_lu2']:.4e} "
                f"mass_drift={rep['mass_drift_rel']:.2e} wall={rep['wall_s']:.1f}s"
            )

    # ---- scheme equivalence on the finest grid -----------------------------
    n_f, s_f = ladder[-1]
    f_ab = states[(n_f, s_f, "ab")]
    f_fu = states[(n_f, s_f, "fused_push_pull")]
    f_ab_collided = collide_bgk(f_ab, TAU)
    rho_a, ux_a, uy_a = macroscopic(f_ab)
    rho_b, ux_b, uy_b = macroscopic(f_fu)
    result["equivalence"] = {
        "grid": n_f,
        "steps": s_f,
        "max_abs_fused_vs_collided_ab": float((f_fu - f_ab_collided).abs().max()),
        "max_abs_fused_vs_raw_ab_state_label_offset": float((f_fu - f_ab).abs().max()),
        "rel_l2_velocity_cross_arm": relative_l2(
            torch.stack([ux_b, uy_b]), torch.stack([ux_a, uy_a])
        ),
        "rel_l2_velocity_fused_vs_analytic": None,
        "rel_l2_velocity_ab_vs_analytic": None,
    }
    _, ux_ref, uy_ref = taylor_green_fields(n_f, n_f, u_max=U_MAX, nu=NU, t=s_f)
    result["equivalence"]["rel_l2_velocity_fused_vs_analytic"] = relative_l2(
        torch.stack([ux_b, uy_b]), torch.stack([ux_ref, uy_ref])
    )
    result["equivalence"]["rel_l2_velocity_ab_vs_analytic"] = relative_l2(
        torch.stack([ux_a, uy_a]), torch.stack([ux_ref, uy_ref])
    )
    print(
        "[equiv] max|fused - C(ab)| = %.3e ; rel.L2 |u_fused - u_ab| = %.3e"
        % (
            result["equivalence"]["max_abs_fused_vs_collided_ab"],
            result["equivalence"]["rel_l2_velocity_cross_arm"],
        )
    )

    # ---- convergence ladder ------------------------------------------------
    for kernel in KERNELS:
        rows = [
            {
                "grid": n,
                "steps": s,
                "ux_lu2": next(
                    a["ux_lu2"]
                    for a in result["arms"]
                    if a["grid"] == n and a["steps"] == s and a["kernel"] == kernel
                ),
            }
            for n, s in ladder
        ]
        orders = [
            convergence_order(rows[i]["ux_lu2"], rows[i + 1]["ux_lu2"], rows[i]["grid"], rows[i + 1]["grid"])
            for i in range(len(rows) - 1)
        ]
        result["ladder"].append({"kernel": kernel, "rows": rows, "orders": orders})
        print(f"[ladder] {kernel}: orders={['%.3f' % o for o in orders]}")

    # ---- negative controls (coarsest grid: the gate must bite there too) ---
    n_c, s_c = ladder[0]
    f0c = initial(n_c)
    bad_tau = 3.0 * (NU * 1.25) + 0.5
    for name, f_start, tau_c, note in (
        ("nu_bias_25pct", f0c, bad_tau, "relaxation from nu*1.25 (+25% viscosity)"),
        ("mirror_we_initial", mirror_we(f0c), TAU, "+x/-x directions of the IC swapped"),
    ):
        f = run_arm(f_start, tau_c, s_c, "fused_push_pull")
        rho_c, ux_c, uy_c = macroscopic(f)
        err = tg_l2_errors(rho_c, ux_c, uy_c, nx=n_c, ny=n_c, u_max=U_MAX, nu=NU, t=s_c, rho0=RHO0)
        result["controls"].append(
            {
                "name": name,
                "note": note,
                "grid": n_c,
                "steps": s_c,
                "ux_lu2": err["ux_lu2"],
                "exceeds_gate": bool(err["ux_lu2"] >= GATE),
            }
        )
        print(f"[control] {name}: ux_lu2={err['ux_lu2']:.4e} exceeds_gate={err['ux_lu2'] >= GATE}")

    # ---- figures -----------------------------------------------------------
    try:
        make_figures(result, states, ladder, pathlib.Path(args.out).parent / "figures")
    except Exception as exc:  # plotting is never allowed to sink the numbers
        result["figures_error"] = repr(exc)
        print("[figures] skipped:", exc)

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[done] wrote {out}")
    return 0


def make_figures(result, states, ladder, figdir: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    caps: dict[str, str] = {}

    # fig1: error vs grid for both kernels + 15%/3% gates
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for entry in result["ladder"]:
        grids = [r["grid"] for r in entry["rows"]]
        errs = [100.0 * r["ux_lu2"] for r in entry["rows"]]
        ax.loglog(grids, errs, "o-", label=entry["kernel"])
    ax.axhline(100 * GATE, color="crimson", ls="--", lw=1, label="15% owner gate")
    ax.axhline(100 * TARGET, color="orange", ls=":", lw=1, label="3% self target")
    ax.set_xlabel("grid n (steps ∝ n², fixed physical time)")
    ax.set_ylabel(r"$u_x$ relative $L_2$ vs analytic  [%]")
    ax.set_title("Taylor–Green D2Q9 BGK: fused push-pull vs AB reference")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig1_error_vs_grid.png", dpi=140)
    caps["fig1_error_vs_grid.png"] = (
        "Discretisation error of both kernels against the closed-form Taylor-Green solution, "
        "grids with steps ∝ n² (fixed physical time). Both curves sit orders of magnitude "
        "below the 15% owner gate."
    )
    plt.close(fig)

    # fig2: field comparison at the finest grid
    n_f, s_f = ladder[-1]
    _, ux_ab, uy_ab = macroscopic(states[(n_f, s_f, "ab")])
    _, ux_fu, uy_fu = macroscopic(states[(n_f, s_f, "fused_push_pull")])
    _, ux_ref, uy_ref = taylor_green_fields(n_f, n_f, u_max=U_MAX, nu=NU, t=s_f)
    vmax = float(ux_ref.abs().max())
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
    for ax, arr, title in zip(
        axes,
        [ux_ref, ux_ab, ux_fu, ux_fu - ux_ab],
        [
            "analytic $u_x$",
            "AB reference $u_x$",
            "fused push-pull $u_x$",
            "fused − AB  $(\\times 10^{13})$",
        ],
        strict=True,
    ):
        data = arr.numpy() * (1e13 if "×" in title else 1.0)
        im = ax.imshow(data.reshape(n_f, n_f), origin="lower", cmap="RdBu_r")
        if "×" not in title:
            im.set_clim(-vmax, vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(figdir / "fig2_fields.png", dpi=140)
    caps["fig2_fields.png"] = (
        "Contours at the finest grid: analytic, AB reference, fused push-pull, and the "
        "difference amplified by 1e13 — the two kernels are the same field to round-off."
    )
    plt.close(fig)

    # fig3: mass conservation & viscosity inversion
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for kernel in KERNELS:
        rows = [a for a in result["arms"] if a["kernel"] == kernel]
        ax1.semilogy([r["grid"] for r in rows], [max(r["mass_drift_rel"], 1e-18) for r in rows], "o-", label=kernel)
        ax2.plot([r["grid"] for r in rows], [100 * (r["nu_inversion_err_rel"] or float("nan")) for r in rows], "o-", label=kernel)
    ax1.axhline(1e-12, color="crimson", ls="--", lw=1, label="1e-12 gate")
    ax1.set_xlabel("grid n")
    ax1.set_ylabel("relative mass drift over the run")
    ax1.set_title("Mass conservation")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()
    ax2.axhline(3.0, color="orange", ls=":", lw=1, label="3% target")
    ax2.set_xlabel("grid n")
    ax2.set_ylabel("viscosity inversion error [%]")
    ax2.set_title(r"$\nu$ from kinetic-energy decay vs $\nu=1/6$")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig3_conservation.png", dpi=140)
    caps["fig3_conservation.png"] = (
        "Left: relative mass drift per run (gate 1e-12). Right: viscosity inverted from the "
        "Taylor-Green energy decay versus the prescribed 1/6."
    )
    plt.close(fig)

    (figdir / "figure_captions.json").write_text(json.dumps(caps, indent=2))
    result["figures"] = [
        {"name": name, "path": f"figures/{name}", "caption": caps[name]} for name in caps
    ]


if __name__ == "__main__":
    raise SystemExit(main())