#!/usr/bin/env python
"""Ekman layer in a rotating reference frame (D3Q19) — analytic-spiral benchmark.

Physics
-------
A homogeneous fluid sits above a plate at ``z = 0`` that is at rest in the
frame rotating with angular velocity ``Omega = (0, 0, Omega_z)``.  A constant
specific body force ``F = (0, f U_g, 0)`` (``f = 2 Omega_z``: Coriolis
parameter) plays the role of an imposed pressure gradient; in the rotating
frame its geostrophic balance is a uniform flow ``u = (U_g, 0, 0)``, i.e.
``-2 Omega x u_g + F = 0``.  The no-slip plate then generates the classic
Ekman spiral (Pedlosky, *GFD*, §3; Greenspan 1968):

    zeta = z / delta_E,        delta_E = sqrt(2 nu / f)

    u_par(zeta) = U_g [1 - e^{-zeta} cos zeta]      (geostrophic / azimuthal)
    u_perp(zeta) = U_g  e^{-zeta} sin zeta          (cross-isobar / radial)

``u_par`` is the component along the free-stream (in rotating-disk language the
azimuthal/geostrophic direction), ``u_perp`` the cross-isobar component that
transports fluid radially inside the layer and vanishes in the interior.  The
hodograph of ``(u_par, u_perp)`` is the exponential spiral: it turns by one
full radian per ``delta_E`` of depth and reaches the geostrophic vector
exponentially.

Setup (real simulation, no extrapolation, no correction factors)
---------------------------------------------------------------
* Domain ``(nz, ny, nx) = (round(H_delta*delta_E), 4, 4)``; periodic in x and
  y (the flow is horizontally homogeneous, so 4x4 columns are exact);
  ``z = 0`` is a no-slip half-way bounce-back plate, ``z = nz-1`` a no-slip
  lid.  ``H = 9 delta_E`` puts the lid ``9`` decay lengths above the plate, so
  its own Ekman layer is down by ``e^{-9} = 1.2e-4`` at the measurement
  heights (``zeta <= 2.5``).
* ``nu = 0.1`` (``tau = 0.8``); ``f = 2 nu / delta_E^2`` in lattice units, so
  the analytic ``delta_E`` is exactly ``--delta-nodes`` cells by construction
  and the criterion tests the *solver*, not a hand-tuned thickness.
* Start from rest; the inertial oscillation excited by the switch-on decays
  with the Ekman spin-up time ``H/(2 delta_E Omega)``.  Integration stops when
  ``max|u(t) - u(t-500)| < 2e-7`` (the residual is written to the JSON).
* One ``(z)`` profile is taken from the column mean of ``u_x, u_y``.

Acceptance (fixed a priori)
---------------------------
* ``delta_E``: least-squares fit of the whole spiral to the analytic family
  vs the analytic ``sqrt(2 nu/f)``: ``|err| <= 5 %``.
* velocity profile at the five prescribed heights
  ``zeta = 0.5, 1.0, 1.5, 2.0, 2.5``: ``max |u_num - u_ana| / U_g <= 3 %``
  over both components.

Usage
-----
    run.py single out.json [--delta-nodes 16] [--height-deltas 9]
                           [--nu 0.1] [--Ug 0.02] [--tol 2e-7]
    run.py scan out_dir [--delta-nodes 12 16 24]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from tensorlbm.rotating_frame import (
    RotatingFrameLBM,
    ekman_delta,
    ekman_profile,
    fit_ekman_thickness,
    lattice_scaling,
)

PROBE_ZETA = (0.5, 1.0, 1.5, 2.0, 2.5)


def _provenance() -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # pragma: no cover - not a git checkout
        sha = "unknown"
    return {
        "git_sha": sha,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_case(
    *,
    delta_nodes: float = 16.0,
    height_deltas: float = 9.0,
    nu: float = 0.1,
    U_g: float = 0.02,
    tol: float = 1e-4,
    max_steps: int = 400_000,
    check_every: int = 500,
    threads: int = 2,
    figure: Path | None = None,
    init: str = "rest",
) -> dict:
    """Run one Ekman-layer case and return the full result record."""
    torch.set_num_threads(threads)
    tau = 3.0 * nu + 0.5
    f_lat = 2.0 * nu / (delta_nodes * delta_nodes)
    delta_ana = ekman_delta(nu, f_lat)
    nz = max(8, int(round(height_deltas * delta_nodes)))
    ny = nx = 4

    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[0] = True  # no-slip plate (the Ekman layer under test)
    solid[nz - 1] = True  # lid, 9 decay lengths above the plate
    slip = torch.zeros_like(solid)
    slip[nz - 1] = True  # free-slip (no-stress) lid: it cannot launch its own
    #                       Ekman layer, so the measured profile stays clean

    sim = RotatingFrameLBM(
        (nz, ny, nx),
        tau=tau,
        omega=(0.0, 0.0, 0.5 * f_lat),
        body_force=(0.0, f_lat * U_g, 0.0),
        solid=solid,
        slip_solid=slip,
        dtype=torch.float32,
    )

    if init == "analytic":
        # start from the analytic spiral: a discretisation-consistent solver
        # must keep this state (fixed-point / stability evidence)
        zz = torch.arange(nz, dtype=torch.float64)
        a_par, a_perp = ekman_profile(zz / delta_ana, U_g)
        ux0 = a_par.to(torch.float32).view(nz, 1, 1).expand(nz, ny, nx).contiguous()
        uy0 = a_perp.to(torch.float32).view(nz, 1, 1).expand(nz, ny, nx).contiguous()
        sim.initialize_from_velocity(ux0, uy0, torch.zeros_like(ux0))
    elif init != "rest":
        raise SystemExit(f"unknown init mode {init!r}")

    # one inertial period (2 pi / Omega): the natural averaging window of a
    # rotating flow, used to strip the inertial oscillation from the residual
    inertial_period = int(round(2.0 * math.pi / (0.5 * f_lat)))

    t0 = time.time()
    conv = sim.run_until_steady(
        tol=tol,
        max_steps=max_steps,
        check_every=check_every,
        period=inertial_period,
    )
    wall = time.time() - t0

    rho, ux, uy, uz = sim.fields()
    fluid = ~solid
    # horizontal homogeneity: average over the 4x4 columns (fluid cells only).
    # Rows that are fully solid (the plate and the lid) have zero fluid cells,
    # so the column count is clamped: dividing by the raw count yields 0/0 = NaN
    # and silently poisons the whole profile (trap recorded in the README).
    cnt = fluid.sum(dim=(1, 2)).clamp(min=1)
    u_par = (ux * fluid).sum(dim=(1, 2)) / cnt
    u_perp = (uy * fluid).sum(dim=(1, 2)) / cnt
    w_max = float(uz[fluid].abs().max())
    rho_min, rho_max = float(rho[fluid].min()), float(rho[fluid].max())

    z = torch.arange(nz, dtype=torch.float64)  # depth measured from the plate
    # the fit only uses the fluid part of the boundary-layer profile
    zf = z[1:-1]
    fit = fit_ekman_thickness(zf, u_par[1:-1], u_perp[1:-1])

    # five prescribed heights, both components vs the analytic spiral
    probes = []
    for zeta in PROBE_ZETA:
        zi = int(round(zeta * delta_ana))
        zi = min(max(zi, 1), nz - 2)
        a_par, a_perp = ekman_profile(torch.tensor([zeta], dtype=torch.float64), U_g)
        a_par, a_perp = float(a_par[0]), float(a_perp[0])
        n_par, n_perp = float(u_par[zi]), float(u_perp[zi])
        e_par = abs(n_par - a_par) / U_g
        e_perp = abs(n_perp - a_perp) / U_g
        probes.append(
            {
                "zeta": zeta,
                "z_index": zi,
                "u_par_num": n_par,
                "u_par_ana": a_par,
                "u_perp_num": n_perp,
                "u_perp_ana": a_perp,
                "err_par": e_par,
                "err_perp": e_perp,
                "err_max": max(e_par, e_perp),
            }
        )

    # full-profile error over the measured part of the layer
    zmask = (z >= 1.0) & (z <= 2.5 * delta_ana)
    a_all = ekman_profile(z[zmask] / delta_ana, U_g)
    prof_err = max(
        float((u_par[zmask] - a_all[0]).abs().max()) / U_g,
        float((u_perp[zmask] - a_all[1]).abs().max()) / U_g,
    )

    d_err = abs(fit["delta"] - delta_ana) / delta_ana
    p_err = max(p["err_max"] for p in probes)
    scaling = lattice_scaling(0.01, 0.01, delta_nodes, nu)

    result = {
        "case": "ekman_layer",
        "config": {
            "delta_nodes": delta_nodes,
            "height_deltas": height_deltas,
            "nu_lattice": nu,
            "tau": tau,
            "f_lattice": f_lat,
            "omega_lattice": 0.5 * f_lat,
            "U_g": U_g,
            "Mach": U_g * math.sqrt(3.0),
            "shape": [nz, ny, nx],
            "Reynolds_delta": U_g * delta_ana / nu,
            "probe_zeta": list(PROBE_ZETA),
            "boundary": "z=0 no-slip plate, z=nz-1 no-slip lid, x/y periodic",
            "tol": tol,
            "inertial_period_steps": inertial_period,
            "lid": "free-slip",
            "wall": "no-slip",
        },
        "convergence": {
            "steps": conv["steps"],
            "residual": conv["residual"],
            "period_steps": conv["period"],
            "converged": conv["converged"],
            "wall_seconds": wall,
            "history": conv["history"][:: max(1, len(conv["history"]) // 40)],
        },
        "profiles": {
            "z": [float(v) for v in z],
            "u_par": [float(v) for v in u_par],
            "u_perp": [float(v) for v in u_perp],
        },
        "fit": {
            "delta_fit": fit["delta"],
            "delta_ana": delta_ana,
            "U_g_fit": fit["U_g"],
            "U_g_ana": U_g,
            "residual_rms": fit["residual_rms"],
            "delta_rel_err": d_err,
        },
        "probes": probes,
        "checks": {
            "delta_err": d_err,
            "profile_max_err": p_err,
            "profile_error_over_layer": prof_err,
            "delta_pass_5pct": bool(d_err <= 0.05),
            "profile_pass_3pct": bool(p_err <= 0.03),
            "w_max_over_Ug": w_max / U_g,
            "rho_range": [rho_min, rho_max],
            "density_variation": (rho_max - rho_min) / 1.0,
        },
        "physical_mapping": scaling,
        "provenance": _provenance(),
    }
    result["verdict"] = bool(
        result["checks"]["delta_pass_5pct"] and result["checks"]["profile_pass_3pct"]
    )
    if figure is not None:
        result["figure"] = str(plot_case(result, Path(figure)))
    return result


def plot_case(res: dict, path: Path) -> Path:
    """Profile + hodograph comparison figure (matplotlib, Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    z = np.asarray(res["profiles"]["z"])
    up = np.asarray(res["profiles"]["u_par"])
    uq = np.asarray(res["profiles"]["u_perp"])
    U_g = res["config"]["U_g"]
    d_ana = res["fit"]["delta_ana"]
    zeta = z / d_ana
    a_par = U_g * (1.0 - np.exp(-zeta) * np.cos(zeta))
    a_perp = U_g * np.exp(-zeta) * np.sin(zeta)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    m = zeta <= 4.0
    ax[0].plot(up[m] / U_g, zeta[m], "o", ms=4, label="LBM $u_\\parallel$")
    ax[0].plot(a_par[m] / U_g, zeta[m], "-", label="analytic")
    ax[0].plot(uq[m] / U_g, zeta[m], "s", ms=3.5, label="LBM $u_\\perp$")
    ax[0].plot(a_perp[m] / U_g, zeta[m], "--", label="analytic")
    ax[0].set_xlabel("$u/U_g$")
    ax[0].set_ylabel("$\\zeta = z/\\delta_E$")
    ax[0].set_title("Ekman profiles")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(up[m] / U_g, uq[m] / U_g, "o-", ms=3, label="LBM hodograph")
    ax[1].plot(a_par[m] / U_g, a_perp[m] / U_g, "-", label="analytic spiral")
    ax[1].set_xlabel("$u_\\parallel/U_g$")
    ax[1].set_ylabel("$u_\\perp/U_g$")
    ax[1].set_title("Ekman hodograph")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[1].set_aspect("equal", adjustable="datalim")

    errs = [
        [p["err_par"] for p in res["probes"]] + [p["err_perp"] for p in res["probes"]],
    ]
    ax[2].bar(np.arange(1, len(errs[0]) + 1), 100 * np.asarray(errs[0]), color="tab:blue")
    ax[2].axhline(3.0, color="r", ls="--", label="3 % criterion")
    ax[2].set_xlabel("probe (5 heights x 2 components)")
    ax[2].set_ylabel("rel. error [% of $U_g$]")
    ax[2].set_title(
        f"$\\delta_E$ err {100 * res['fit']['delta_rel_err']:.2f} %, "
        f"profile max {100 * res['checks']['profile_max_err']:.2f} %"
    )
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.suptitle(
        "Rotating-frame LBM: Ekman layer, "
        f"$\\delta_E={res['config']['delta_nodes']:.0f}$ nodes, "
        f"$f={res['config']['f_lattice']:.2e}$",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    sp = sub.add_parser("single")
    sp.add_argument("out", type=Path)
    sp.add_argument("--delta-nodes", type=float, default=16.0)
    sp.add_argument("--height-deltas", type=float, default=9.0)
    sp.add_argument("--nu", type=float, default=0.1)
    sp.add_argument("--Ug", type=float, default=0.02)
    sp.add_argument("--tol", type=float, default=1e-4)
    sp.add_argument("--max-steps", type=int, default=400_000)
    sp.add_argument("--threads", type=int, default=2)
    sp.add_argument("--figure", type=Path, default=None)
    sp.add_argument("--init", choices=["rest", "analytic"], default="rest")

    sc = sub.add_parser("scan")
    sc.add_argument("outdir", type=Path)
    sc.add_argument("--delta-nodes", type=float, nargs="+", default=[12.0, 16.0, 24.0])
    sc.add_argument("--height-deltas", type=float, default=9.0)
    sc.add_argument("--nu", type=float, default=0.1)
    sc.add_argument("--Ug", type=float, default=0.02)
    sc.add_argument("--tol", type=float, default=1e-4)
    sc.add_argument("--threads", type=int, default=2)

    args = ap.parse_args(argv)
    if args.mode == "single":
        res = run_case(
            delta_nodes=args.delta_nodes,
            height_deltas=args.height_deltas,
            nu=args.nu,
            U_g=args.Ug,
            tol=args.tol,
            max_steps=args.max_steps,
            threads=args.threads,
            figure=args.figure,
            init=args.init,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "verdict": res["verdict"],
                    "delta_rel_err": res["checks"]["delta_err"],
                    "profile_max_err": res["checks"]["profile_max_err"],
                    "steps": res["convergence"]["steps"],
                },
                indent=2,
            )
        )
        return 0 if res["verdict"] else 3

    args.outdir.mkdir(parents=True, exist_ok=True)
    runs = []
    for dn in args.delta_nodes:
        res = run_case(
            delta_nodes=dn,
            height_deltas=args.height_deltas,
            nu=args.nu,
            U_g=args.Ug,
            tol=args.tol,
            threads=args.threads,
            figure=args.outdir / f"ekman_delta{dn:.0f}.png",
        )
        (args.outdir / f"ekman_delta{dn:.0f}.json").write_text(json.dumps(res, indent=2))
        runs.append(
            {
                "delta_nodes": dn,
                "delta_rel_err": res["checks"]["delta_err"],
                "profile_max_err": res["checks"]["profile_max_err"],
                "steps": res["convergence"]["steps"],
                "verdict": res["verdict"],
            }
        )
        print(json.dumps(runs[-1]))
    summary = {
        "kind": "ekman_scan",
        "runs": runs,
        "all_pass": all(r["verdict"] for r in runs),
        "provenance": _provenance(),
    }
    (args.outdir / "scan_summary.json").write_text(json.dumps(summary, indent=2))
    return 0 if summary["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
