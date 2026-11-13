"""XVAL-LBM-PORTER-EXPLICIT-FORCING-4TH-ORDER driver (round 1, open-source cross-benchmark).

Benchmarks
  B1  single-component forced channel, from rest   -> analytic Poiseuille       (forcing bookkeeping)
  B2  two-component layered channel, started on the analytic two-layer Stokes
      profile (``fixed_point`` mode)                -> analytic two-layer Stokes (viscosity-weighted u_eq)
  B2t two-component layered channel from rest      -> transient / convergence sensitivity
  B3  force-stencil isotropy + spurious tangential force on an analytic interface (operator level;
      the reference implementation ships the stencils but has no such measurement)

Arms
  doc_literal / doc_seq / guo    tensorlbm.porter_forcing (shipped API)
  REF_doc_literal / REF_doc_seq  independent literal transcription (ref/porter_ref.py)
  legacy                         control: pre-Porter Shan-Chen recipe (force folded into u_eq_i)

Why the fixed-point start: the momentum diffusion time of the light layer is
``H^2/nu_light`` = 32^2 * 60 ~ 61000 steps at M=10, i.e. an order of magnitude
beyond a from-rest budget; the fixed-point start measures the quantity the
decision is about (does the discrete scheme preserve the analytic solution)
without that confound, and the from-rest run is kept separately (B2t).

Outputs: artifacts/result.json, artifacts/table.csv
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import got_driver as G  # noqa: E402

from tensorlbm.porter_forcing import (  # noqa: E402
    CS2,
    gradient_field,
    isotropic_gradient_weights,
    isotropy_moments,
)

TAU_A = 1.0  # bulk (component 0) relaxation time -> nu = 1/6
G0 = 5e-6  # body acceleration, lattice units
FP_STEPS = 2000  # fixed-point budget (any bookkeeping error shows in << 1000 steps)


def nu_of(tau: float) -> float:
    return CS2 * (tau - 0.5)


def taperr(u_lbm, u_exact, y, yi, band=2.0):
    y = np.asarray(y, float)
    scale = float(np.max(np.abs(u_exact)))
    err = float(np.max(np.abs(u_lbm - u_exact))) / scale
    keep = np.abs(y - yi) > band
    err_excl = float(np.max(np.abs(u_lbm[keep] - u_exact[keep]))) / float(np.max(np.abs(u_exact[keep])))
    return err, err_excl


def layer_viscosities(y, u, yi, g, half=6):
    """Effective kinematic viscosity of each layer from the LBM profile curvature."""
    out = {}
    for name, sel in (("below", y < yi - 2.5), ("above", y > yi + 2.5)):
        yy = y[sel]
        uu = u[sel]
        # local second derivative by central differences
        d2 = (uu[2:] - 2 * uu[1:-1] + uu[:-2]) / ((yy[2:] - yy[1:-1]) ** 2)
        nu_eff = float(np.median(-g / d2)) if np.all(d2 < 0) else float("nan")
        out[name] = nu_eff
    return out


def run_single(ny, arm, steps=None):
    tau = TAU_A
    steps = steps or G.interior_steps(ny)
    t0 = time.time()
    if arm == "legacy":
        out = G.run_channel(ny, [tau], [np.ones((ny, 1))], [G0], steps, scheme="legacy")
    else:
        out = G.run_channel(ny, [tau], [np.ones((ny, 1))], [G0], steps, scheme="porter", variant=arm)
    h = ny - 2
    y = np.arange(1, ny - 1) - 0.5
    u_exact = G.layered_stokes(y, h, h / 2.0, nu_of(tau), nu_of(tau), G0)
    err, err_excl = taperr(out["ux"][1:-1], u_exact, y, h / 2.0)
    return {
        "bench": "B1_single_channel", "arm": arm, "ny": ny, "M": 1.0,
        "err": err, "err_excl": err_excl, "u_max_exact": float(np.max(np.abs(u_exact))),
        "steps": out["steps"], "mass_drift": out["mass_drift"], "unstable": out["unstable"],
        "wall_s": round(time.time() - t0, 1), "mode": "from_rest",
    }


def run_layered(ny, arm, M, mode="fixed_point", steps=None):
    nu_a = nu_of(TAU_A)
    nu_b = nu_a / M
    tau_b = 0.5 + nu_b / CS2
    rho0, h, yi = G.layers(ny, 1)
    y = np.arange(1, ny - 1) - 0.5
    u_exact = G.layered_stokes(y, h, yi, nu_a, nu_b, G0)
    ux0 = None
    if mode == "fixed_point":
        ux0 = np.zeros(ny)
        ux0[1:-1] = u_exact
        steps = steps or FP_STEPS
    else:
        steps = steps or G.interior_steps(ny)
    t0 = time.time()
    if arm.startswith("REF_"):
        import ref.porter_ref as REF

        ux, uy, rho_tot, n_done, hist = REF.run(
            ny, 1, [TAU_A, tau_b], rho0, [G0, G0], [0, 0], steps,
            check_every=500, tol=1e-30, variant=arm[4:], ux0=ux0,
        )
        unstable = not np.isfinite(ux).all()
        ux_1d = np.asarray(ux, float).reshape(ny, 1)[:, 0]
        ux_full = ux_1d if not unstable else np.full(ny, np.nan)
        mass = float("nan")
        uymax = float(np.max(np.abs(uy))) if not unstable else float("nan")
    else:
        scheme = "legacy" if arm == "legacy" else "porter"
        out = G.run_channel(
            ny, [TAU_A, tau_b], rho0, [G0, G0], steps,
            scheme=scheme, variant=arm, atol=1e-30, ux0=ux0,
        )
        unstable = out["unstable"]
        ux_full = out["ux"]
        mass = out["mass_drift"]
        uymax = out["uy_max"]
        n_done = out["steps"]
    if unstable:
        err = err_excl = float("nan")
    else:
        err, err_excl = taperr(ux_full[1:-1], u_exact, y, yi)
    row = {
        "bench": "B2_layered_channel", "arm": arm, "ny": ny, "M": M, "mode": mode,
        "tau": [TAU_A, tau_b], "err": err, "err_excl": err_excl,
        "u_max_exact": float(np.max(np.abs(u_exact))), "steps": n_done,
        "mass_drift": mass, "unstable": unstable, "wall_s": round(time.time() - t0, 1),
        "uy_max": uymax,
    }
    if not unstable:
        row["nu_eff_measured"] = layer_viscosities(y, ux_full[1:-1], yi, G0)
        row["nu_exact"] = {"below": nu_a, "above": nu_b}
        row["ux"] = ux_full[1:-1].tolist()
        row["u_exact"] = u_exact.tolist()
        row["y"] = y.tolist()
    return row


def b3_stencils(n=128, radius=24.0, xi=1.5):
    """Operator-level isotropy + spurious-force metric on analytic fields.

    A flat interface has, in the continuum, a purely normal force whose
    resultant vanishes; a circular droplet has a purely radial force.  Any
    tangential component is discretisation error and is exactly what feeds
    spurious currents.
    """
    out = {}
    for order in (2, 4, 6):
        offsets, weights = isotropic_gradient_weights(order)
        mom = isotropy_moments(offsets, weights)
        y = (np.arange(n) - n // 2)[:, None] * np.ones((1, n))
        psi_flat = np.tanh((y - 0.5) / xi)
        gx, gy = gradient_field(psi_flat, offsets, weights)
        yy, xx = np.meshgrid(np.arange(n) - n / 2, np.arange(n) - n / 2, indexing="ij")
        r = np.hypot(xx, yy) + 1e-12
        psi_drop = np.tanh((radius - r) / xi)
        gx2, gy2 = gradient_field(psi_drop, offsets, weights)
        radial = (gx2 * xx + gy2 * yy) / r
        tang = (-gx2 * yy + gy2 * xx) / r
        shell = np.abs(r - radius) < 2.0
        out[str(order)] = {
            "isotropy": mom,
            "flat_max_tangential": float(np.max(np.abs(gx))),
            "flat_max_normal": float(np.max(np.abs(gy))),
            "drop_max_radial": float(np.max(np.abs(radial))),
            "drop_max_tangential": float(np.max(np.abs(tang))),
            "drop_tangential_over_radial_shell": float(
                np.max(np.abs(tang[shell])) / np.max(np.abs(radial[shell]))
            ),
        }
    return out


def main():
    os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
    results = {"config": {"TAU_A": TAU_A, "G0": G0, "nu_A": nu_of(TAU_A), "fp_steps": FP_STEPS}, "runs": []}

    def record(row):
        results["runs"].append(row)
        with open(os.path.join(HERE, "artifacts", "result.json"), "w") as f:
            json.dump(results, f, indent=1)
        print(
            f"{row['bench'][:12]:12s} {row['arm']:16s} ny={row['ny']:3d} M={row['M']:<5} {row['mode'][:11]:11s} "
            f"err={row['err']:.3e} excl={row['err_excl']:.3e} mass={row.get('mass_drift', float('nan')):.1e} "
            f"t={row.get('wall_s')}s",
            flush=True,
        )

    # ---- B1: single component, grid sweep, all arms, from rest
    for ny in (18, 34, 66):
        for arm in ("doc_literal", "doc_seq", "guo", "legacy"):
            record(run_single(ny, arm))

    # ---- B2: layered, fixed-point start, grid + viscosity-ratio sweep
    for M in (1.0, 10.0):
        for ny in (18, 34, 66):
            for arm in ("doc_literal", "legacy"):
                record(run_layered(ny, arm, M))
    for M in (5.0, 100.0):
        for arm in ("doc_literal", "legacy"):
            record(run_layered(34, arm, M))
    record(run_layered(34, "doc_seq", 1.0))  # bookkeeping check of the second reading

    # ---- reference arm (independent transcription), same fixed-point start
    for variant in ("doc_literal", "doc_seq"):
        for M in (1.0, 10.0):
            record(run_layered(34, "REF_" + variant, M))

    # ---- B2t: transient from rest (convergence sensitivity)
    for arm in ("doc_literal", "legacy"):
        record(run_layered(34, arm, 1.0, mode="from_rest"))

    # ---- B3: operator level
    results["b3"] = b3_stencils()
    with open(os.path.join(HERE, "artifacts", "result.json"), "w") as f:
        json.dump(results, f, indent=1)
    for k, v in results["b3"].items():
        print("stencil order", k, json.dumps(v), flush=True)


if __name__ == "__main__":
    main()