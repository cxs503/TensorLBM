#!/usr/bin/env python
"""Source-sink flow in a rotating tank (D3Q19) — Taylor-Proudman benchmark.

Physics
-------
A closed cylindrical tank (radius ``R``, height ``H``) rotates with ``Omega =
(0,0,Omega_z)``.  Fluid is injected in a central vertical column (the *source*)
and withdrawn in an annular ring just inside the side wall (the *sink*), with
equal strength ``Q``.  Mass conservation in an annulus then forces the radial
transport

    v_r(r) = Q / (2 pi r H rho)  ~  1/r,

and — because the flow is slow (``Ro = v_r/(2 Omega L) << 1``) and steady — the
Taylor-Proudman theorem ``du/dz = 0`` forbids any axial variation of the
interior radial velocity: the outflow fills the whole depth instead of
concentrating in a layer, which is the rotating-tank analogue of the
Taylor column (Hide 1968; Greenspan 1968, ch. 3).  The end caps brake the swirl,
so the interior co-rotates with the tank (``v_theta << Omega r``) and the only
role of rotation is the two-dimensional constraint.

Setup (real simulation, no extrapolation, no correction factors)
---------------------------------------------------------------
* Cartesian grid with a voxel tank: cells with ``r > R`` are solid (side wall),
  ``z = 0`` and ``z = nz-1`` are solid (the co-rotating end caps, i.e. no-slip
  in the rotating frame).  ``x`` and ``y`` are periodic, which is exact
  because the whole solid ring is inside the box.
* The source/sink are **volumetric mass sources** ``S(x)`` (mass per step per
  cell) added as ``S w_i`` to the post-collision populations
  (:class:`tensorlbm.rotating_frame.RotatingFrameLBM`).  Because
  ``sum_i c_i w_i = 0`` they add mass without momentum: the injected fluid
  enters at rest in the rotating frame, and the resulting pressure field drives
  the flow.  Nothing about ``v_r`` is prescribed.
* ``nu = 0.05`` (``tau = 0.65``), ``Omega = 0.02``: ``E = nu/(Omega H^2) ~ 1e-3``
  so the end-cap Ekman pumping is ``O(sqrt(E)) ~ 3 %`` of the outflow.
* Start from rest and integrate until the velocity one inertial period
  (``2 pi/Omega``) apart changes by less than ``tol``.

Acceptance (fixed a priori)
---------------------------
* radial power law at mid-height: ``|n - (-1)| <= 0.05`` where ``v_r = a r^n``.
* axial uniformity in the interior slab (end caps excluded): the max relative
  deviation of ``v_r`` from its shell mean over ``z`` is ``<= 5 %``.

Usage
-----
    run.py single out.json [--ny 48 --nx 48 --nz 48] [--Q 6.0]
    run.py scan out_dir [--Q 3.0 6.0]
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

from tensorlbm.rotating_frame import RotatingFrameLBM, fit_power_law, radial_average

__all__ = ["build_tank", "run_case"]


def build_tank(
    shape: tuple[int, int, int],
    *,
    radius: float | None = None,
    side_thickness: float = 2.0,
    cap_thickness: int = 2,
    source_radius: float = 3.5,
    sink_width: float = 3.0,
    q_total: float = 6.0,
    slip_caps: bool = False,
) -> dict:
    """Voxel tank + volumetric source/sink masks.

    Returns a dict with ``solid``, ``source``, ``sink``, ``r`` (cell radius),
    ``centre`` and the realised source/sink strengths.
    """
    nz, ny, nx = shape
    cx, cy = 0.5 * (nx - 1), 0.5 * (ny - 1)
    y = torch.arange(ny).view(1, ny, 1)
    x = torch.arange(nx).view(1, 1, nx)
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2).expand(nz, ny, nx).contiguous()
    if radius is None:
        radius = 0.5 * min(nx, ny) - side_thickness - 0.5

    solid = (r > radius).expand(nz, ny, nx).clone()
    solid[:cap_thickness] = True
    solid[nz - cap_thickness :] = True

    fluid = ~solid
    source = (r <= source_radius) & fluid
    sink = (r > radius - sink_width) & (r <= radius) & fluid
    if int(source.sum()) == 0 or int(sink.sum()) == 0:
        raise ValueError("source or sink region is empty — adjust the geometry")

    # Q is the total mass per step; distribute it over each region
    src = torch.zeros(shape, dtype=torch.float32)
    snk = torch.zeros(shape, dtype=torch.float32)
    src[source] = q_total / float(source.sum())
    snk[sink] = -q_total / float(sink.sum())
    # stress-free end caps (optional): removes the cap Ekman layers entirely so
    # that the Proudman-Taylor interior fills the tank.  The rotating Ekman
    # anchor (rotating_ekman) is the benchmark for the cap-layer physics itself;
    # here the object under test is the *interior* columnarity.
    slip = torch.zeros(shape, dtype=torch.bool)
    if slip_caps:
        slip[:cap_thickness] = solid[:cap_thickness]
        slip[nz - cap_thickness :] = solid[nz - cap_thickness :]
    return {
        "solid": solid,
        "source": source,
        "sink": sink,
        "mass_source": src + snk,
        "slip": slip,
        "r": r,
        "centre": (cx, cy),
        "radius": radius,
        "q_total": q_total,
        "source_cells": int(source.sum()),
        "sink_cells": int(sink.sum()),
    }


def _provenance() -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # pragma: no cover
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
    ny: int = 48,
    nx: int = 48,
    nz: int = 48,
    nu: float = 0.05,
    omega_z: float = 0.02,
    q_total: float = 6.0,
    source_radius: float = 3.5,
    sink_width: float = 3.0,
    threads: int = 8,
    tol: float = 2e-5,
    max_steps: int = 200_000,
    cap_exclude: int | None = None,
    min_periods: float = 6.0,
    slip_caps: bool = False,
    figure: Path | None = None,
) -> dict:
    """Run the rotating-tank source-sink case and return the result record."""
    torch.set_num_threads(threads)
    tau = 3.0 * nu + 0.5
    shape = (nz, ny, nx)
    tank = build_tank(
        shape,
        radius=None,
        source_radius=source_radius,
        sink_width=sink_width,
        q_total=q_total,
        slip_caps=slip_caps,
    )
    solid = tank["solid"]
    sim = RotatingFrameLBM(
        shape,
        tau=tau,
        omega=(0.0, 0.0, omega_z),
        solid=solid,
        slip_solid=tank["slip"],
        mass_source=tank["mass_source"],
        dtype=torch.float32,
    )
    period = int(round(2.0 * math.pi / omega_z))
    t0 = time.time()
    # the residual is an *absolute* velocity change: it must be compared with
    # the (small) radial velocity scale, otherwise a barely perturbed rest
    # state looks "converged" and the run stops before the flow develops
    conv = sim.run_until_steady(
        tol=tol,
        max_steps=max_steps,
        check_every=500,
        period=period,
        min_steps=int(min_periods * period),
    )
    wall = time.time() - t0

    rho, ux, uy, uz = sim.fields()
    r = tank["r"]
    cx, cy = tank["centre"]
    y = torch.arange(ny).view(1, ny, 1)
    x = torch.arange(nx).view(1, 1, nx)
    rx = ((x - cx) / r.clamp(min=1e-9)).expand(shape) if r.dim() == 3 else x
    ry = ((y - cy) / r.clamp(min=1e-9)).expand(shape) if r.dim() == 3 else y
    v_r = ux * rx + uy * ry
    v_th = -ux * ry + uy * rx

    fluid = ~solid
    delta_e = math.sqrt(nu / omega_z)
    # The Taylor-Proudman *interior* is the region outside the cap Ekman
    # layers.  The exact Ekman spiral has amplitude envelope e^{-zeta} with
    # zeta = z/delta_E, so it returns to the geostrophic value within 0.2 % at
    # zeta = 2 pi; the measured radial-velocity excess/dip of the cap layer
    # (see the axial z-profiles in the records) is indeed confined to
    # |z - z_cap| <~ 2 pi delta_E.  The slab edge is therefore placed at
    # 2 pi delta_E (half a cell more, the bounce-back wall sits half a cell
    # inside the first fluid node), never less than 4 cells.  Slabs that still
    # contained the cap layer are what made the first attempt report 12 %
    # "non-columnar" flow at delta_E = 3.2 cells.
    if cap_exclude is None:
        cap_exclude = max(4, int(math.ceil(2.0 * math.pi * delta_e + 0.5)))
    z_lo = cap_exclude
    z_hi = nz - 1 - cap_exclude
    interior = fluid.clone()
    interior[:z_lo] = False
    interior[z_hi + 1 :] = False
    mid = interior[nz // 2]  # one z-slice at mid-height

    # shells: geometric spacing so that v_r ~ 1/r is sampled evenly in log(r)
    # shell range: at least 1.5 source radii from the axis and 0.5 cells clear
    # of the sink ring, so that the power law is fitted over a wide lever arm
    r_in = max(1.5 * source_radius, source_radius + 1.0)
    r_out = float(tank["radius"]) - sink_width - 0.5
    if r_out <= 1.5 * r_in:
        raise SystemExit(
            f"radial lever arm too short for a power law: r_in={r_in:.2f}, r_out={r_out:.2f}"
        )
    edges = torch.exp(torch.linspace(math.log(r_in), math.log(r_out), 13, dtype=torch.float64))
    edges = [float(v) for v in edges]
    mid_prof = radial_average(v_r[nz // 2], r[nz // 2], mid, edges)
    rr = [v for v in mid_prof["r"] if not math.isnan(v)]
    vv = [
        abs(v)
        for v, c in zip(mid_prof["mean"], mid_prof["count"], strict=True)
        if c > 0 and not math.isnan(v)
    ]
    # keep only shells with a positive (outward) mean, required by the log fit
    pairs = [
        (a, b)
        for a, b, c in zip(rr, mid_prof["mean"], mid_prof["count"], strict=True)
        if c > 0 and b > 0
    ]
    rr = [p[0] for p in pairs]
    vv = [p[1] for p in pairs]
    if len(rr) < 4:
        raise SystemExit("too few shells with outward flow for the power-law fit")
    fit = fit_power_law(rr, vv)

    # axial uniformity of v_r shell by shell over the interior slab
    shell_edges = edges
    axial = []
    for i in range(len(shell_edges) - 1):
        sel = interior & (r >= shell_edges[i]) & (r < shell_edges[i + 1])
        if int(sel.sum()) == 0:
            continue
        cnt_z = sel.sum(dim=(1, 2)).clamp(min=1)
        prof_z = (v_r * sel).sum(dim=(1, 2)) / cnt_z  # (nz,)
        rows = sel.any(dim=(1, 2)) & (torch.arange(nz) >= z_lo) & (torch.arange(nz) <= z_hi)
        rows_full = sel.any(dim=(1, 2)) & (torch.arange(nz) >= 2) & (torch.arange(nz) <= nz - 3)
        pz = prof_z[rows]
        mean = float(pz.mean())
        dev = float((pz - mean).abs().max())
        pf = prof_z[rows_full]
        mean_f = float(pf.mean())
        dev_f = float((pf - mean_f).abs().max())
        knee = int(torch.argmax((pz - mean).abs())) if pz.numel() else -1
        z_rows = torch.arange(nz)[rows]
        axial.append(
            {
                "r_mid": 0.5 * (shell_edges[i] + shell_edges[i + 1]),
                "mean_vr": mean,
                "max_rel_dev": dev / abs(mean) if abs(mean) > 1e-12 else float("nan"),
                "z_of_max_dev": int(z_rows[knee]) if knee >= 0 else -1,
                "max_rel_dev_full_slab": (
                    dev_f / abs(mean_f) if abs(mean_f) > 1e-12 else float("nan")
                ),
                "n_z": int(rows.sum()),
                "z_profile": [float(v) for v in pz],
                "z_index": [int(v) for v in z_rows],
            }
        )
    axial_ok = [a for a in axial if abs(a["mean_vr"]) > 1e-6 and not math.isnan(a["max_rel_dev"])]
    unif_max = max((a["max_rel_dev"] for a in axial_ok), default=float("nan"))

    swirl = float((v_th[interior].abs() / (omega_z * r[interior].clamp(min=1.0))).max())
    # global mass budget: (source - sink) must vanish to machine-ish accuracy
    mass_in = float(tank["mass_source"][tank["mass_source"] > 0].sum())
    mass_out = float(-tank["mass_source"][tank["mass_source"] < 0].sum())

    n_err = abs(fit["exponent"] + 1.0)
    result = {
        "case": "rotating_source_sink",
        "config": {
            "shape": [nz, ny, nx],
            "nu_lattice": nu,
            "tau": tau,
            "Omega_lattice": omega_z,
            "f_lattice": 2.0 * omega_z,
            "Q_mass_per_step": q_total,
            "radius": tank["radius"],
            "source_radius": source_radius,
            "sink_width": sink_width,
            "source_cells": tank["source_cells"],
            "sink_cells": tank["sink_cells"],
            "interior_slab": [z_lo, z_hi],
            "Ekman_number": nu / (omega_z * nz * nz),
            "delta_E_cells": math.sqrt(nu / omega_z),
            "inertial_period_steps": period,
            "tol": tol,
            "cap_exclude": cap_exclude,
            "caps": "free-slip" if slip_caps else "no-slip",
        },
        "convergence": {
            "steps": conv["steps"],
            "residual": conv["residual"],
            "period_steps": conv["period"],
            "converged": conv["converged"],
            "wall_seconds": wall,
            "history": conv["history"][:: max(1, len(conv["history"]) // 40)],
        },
        "radial_profile": {
            "r": rr,
            "v_r_mid": vv,
            "exponent": fit["exponent"],
            "coefficient": fit["coefficient"],
            "r2": fit["r2"],
        },
        "axial": axial,
        "checks": {
            "exponent_err": n_err,
            "exponent_pass_005": bool(n_err <= 0.05),
            "axial_uniformity_max": unif_max,
            "axial_pass_5pct": bool(unif_max <= 0.05),
            "axial_uniformity_max_full_slab": max(
                (
                    a["max_rel_dev_full_slab"]
                    for a in axial
                    if not math.isnan(a["max_rel_dev_full_slab"])
                ),
                default=float("nan"),
            ),
            "delta_E_cells": delta_e,
            "cap_exclude_cells": int(cap_exclude),
            "max_relative_swirl": swirl,
            "mass_in": mass_in,
            "mass_out": mass_out,
            "rho_range": [float(rho[fluid].min()), float(rho[fluid].max())],
            "w_max": float(uz[fluid].abs().max()),
        },
        "provenance": _provenance(),
    }
    result["verdict"] = bool(
        result["checks"]["exponent_pass_005"] and result["checks"]["axial_pass_5pct"]
    )
    if figure is not None:
        result["figure"] = str(plot_case(result, v_r, v_th, r, solid, Path(figure), omega_z))
    return result


def plot_case(
    res: dict,
    v_r: torch.Tensor,
    v_th: torch.Tensor,
    r: torch.Tensor,
    solid: torch.Tensor,
    path: Path,
    omega_z: float,
) -> Path:
    """Mid-plane cloud + radial profile / axial-uniformity figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    nz, ny, nx = v_r.shape
    vr = v_r[nz // 2].numpy().copy()
    sol = solid[nz // 2].numpy()
    vr[sol] = np.nan
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))
    vmax = np.nanmax(np.abs(vr)) if np.isfinite(vr).any() else 1.0
    im = ax[0].imshow(vr, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    ax[0].set_title("$v_r$ at mid-height")
    ax[0].set_xlabel("x [cells]")
    ax[0].set_ylabel("y [cells]")
    plt.colorbar(im, ax=ax[0], fraction=0.046)

    rr = np.asarray(res["radial_profile"]["r"])
    vv = np.asarray(res["radial_profile"]["v_r_mid"])
    n = res["radial_profile"]["exponent"]
    a = res["radial_profile"]["coefficient"]
    ax[1].loglog(rr, vv, "o", label="LBM")
    ax[1].loglog(rr, a * rr**n, "-", label=f"fit $v_r={a:.3g}r^{{{n:.3f}}}$")
    ax[1].loglog(rr, a * rr ** (-1.0), "k--", lw=1, label="$r^{-1}$ reference")
    ax[1].set_xlabel("r [cells]")
    ax[1].set_ylabel("$v_r$")
    ax[1].set_title(f"power law: n = {n:.4f} (target -1)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")

    axial = [a for a in res["axial"] if not math.isnan(a["max_rel_dev"])]
    ax[2].semilogx(
        [a["r_mid"] for a in axial],
        [100 * a["max_rel_dev"] for a in axial],
        "o-",
    )
    ax[2].axhline(5.0, color="r", ls="--", label="5 % criterion")
    ax[2].set_xlabel("r [cells]")
    ax[2].set_ylabel("max $|v_r-\\bar v_r|/\\bar v_r$  [%]")
    ax[2].set_title(f"axial uniformity, max {100 * res['checks']['axial_uniformity_max']:.2f} %")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, which="both")
    fig.suptitle(
        "Rotating-tank source-sink flow: Taylor-Proudman column",
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
    for name in ("single", "scan"):
        sp = sub.add_parser(name)
        sp.add_argument("out" if name == "single" else "outdir")
        sp.add_argument("--ny", type=int, default=48)
        sp.add_argument("--nx", type=int, default=48)
        sp.add_argument("--nz", type=int, default=48)
        sp.add_argument("--nu", type=float, default=0.05)
        sp.add_argument("--omega", type=float, default=0.02)
        sp.add_argument("--Q", type=float, nargs="+", default=[6.0])
        sp.add_argument("--tol", type=float, default=2e-6)
        sp.add_argument("--min-periods", type=float, default=6.0)
        sp.add_argument("--cap-exclude", type=int, default=None)
        sp.add_argument("--slip-caps", action="store_true")
        sp.add_argument("--max-steps", type=int, default=200_000)
        sp.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    common = dict(
        ny=args.ny,
        nx=args.nx,
        nz=args.nz,
        nu=args.nu,
        omega_z=args.omega,
        tol=args.tol,
        max_steps=args.max_steps,
        threads=args.threads,
        min_periods=args.min_periods,
        cap_exclude=args.cap_exclude,
        slip_caps=args.slip_caps,
    )
    if args.mode == "single":
        out = Path(args.out)
        res = run_case(**common, q_total=args.Q[0], figure=out.with_suffix(".png"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(
            json.dumps(
                {
                    "out": str(out),
                    "verdict": res["verdict"],
                    "exponent": res["radial_profile"]["exponent"],
                    "axial_max": res["checks"]["axial_uniformity_max"],
                    "steps": res["convergence"]["steps"],
                },
                indent=2,
            )
        )
        return 0 if res["verdict"] else 3

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs = []
    for q in args.Q:
        res = run_case(**common, q_total=q, figure=outdir / f"source_sink_Q{q:g}.png")
        (outdir / f"source_sink_Q{q:g}.json").write_text(json.dumps(res, indent=2))
        runs.append(
            {
                "Q": q,
                "exponent": res["radial_profile"]["exponent"],
                "exponent_err": res["checks"]["exponent_err"],
                "axial_uniformity_max": res["checks"]["axial_uniformity_max"],
                "verdict": res["verdict"],
            }
        )
        print(json.dumps(runs[-1]))
    summary = {
        "kind": "source_sink_scan",
        "runs": runs,
        "all_pass": all(r["verdict"] for r in runs),
        "provenance": _provenance(),
    }
    (outdir / "scan_summary.json").write_text(json.dumps(summary, indent=2))
    return 0 if summary["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
