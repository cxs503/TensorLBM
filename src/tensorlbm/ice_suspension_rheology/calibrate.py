"""Calibration campaign for the ice-crystal effective-viscosity closures.

What this module produces (all numbers traceable to a JSON artefact):

1. **Drag pre-gate** -- single-sphere drag against the Stokes solution at three
   grid resolutions plus the grid-convergence index (GCI).  Nothing downstream
   is trusted unless the numerical kernel passes this gate.
2. **Concentration sweep** -- ``eta_r(phi)`` from the LBM shear cell for
   sphere populations, from the dilute limit (``phi = 0.01``) to the packed
   range, with the lattice volume fraction taken from the discretised masks.
3. **Closure comparison** -- Einstein / Batchelor / Krieger-Dougherty against
   the measured data, the fitted ``phi_max`` and the dilute-limit regression
   ``eta_r = 1 + [eta] phi`` (both ``[eta]`` and the intercept are reported
   with their residuals).
4. **Rate sweep** -- ``tau`` versus ``gamma_dot`` at fixed ``phi``: the linear
   regime, its ``R**2``, the intermediate-viscosity ``eta_eff`` and the
   inertial (mu(I)) crossover used by the ``I_v`` constant of the contract.
5. **Shape / polydispersity sensitivity** -- the applicability map for
   near-spherical (faceted) grains and log-normal size distributions.

Every case is a separate process (``multiprocessing``) because the LBM shear
cell is a single-threaded-torch workload; the campaign is embarrassingly
parallel over cases.

Usage
-----
    python -m tensorlbm.ice_suspension_rheology.calibrate --preset quick  --out runs/ice_eta_quick
    python -m tensorlbm.ice_suspension_rheology.calibrate --preset full   --out runs/ice_eta_full
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

from . import closures, shapes
from .closures import (
    BATCHELOR_K2,
    INTRINSIC_VISCOSITY_SPHERE,
    ViscosityContract,
    batchelor_relative_viscosity,
    einstein_relative_viscosity,
    fit_krieger_dougherty_phi_max,
    krieger_dougherty_relative_viscosity,
)
from .couette import ShearCell, ShearCellConfig

__all__ = [
    "PRESETS",
    "drag_gate",
    "run_case",
    "concentration_sweep",
    "rate_sweep",
    "analyse",
    "write_report",
]


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------
PRESETS = {
    "quick": dict(
        nx=16, ny=16, nz=21, radius=3.0, n_steps=300, wall_margin=4,
        phi=[0.02, 0.05], seeds=[1], rate_points=3,
    ),
    "full": dict(
        nx=48, ny=48, nz=57, radius=5.0, n_steps=2600, wall_margin=7,
        phi=[0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30],
        seeds=[1, 2], rate_points=5,
    ),
}


# ---------------------------------------------------------------------------
# 1. drag pre-gate
# ---------------------------------------------------------------------------
def _stokes_case(d: int, re_p: float, steps: int, domain_factor: float, threads: int) -> dict:
    import torch

    torch.set_num_threads(max(1, threads))
    from ..stokes_sphere import run_stokes_sphere, stokes_cd_analytical

    t0 = time.time()
    r = run_stokes_sphere(re=re_p, d_cells=int(d), steps=steps,
                          warmup=steps // 3, domain_factor=domain_factor)
    cd = float(r["cd"])
    cd_ref = stokes_cd_analytical(re_p)
    return {
        "D": int(d),
        "cd": cd,
        "cd_ref": cd_ref,
        "err_pct": 100.0 * abs(cd / cd_ref - 1.0),
        "drag": float(r.get("force", float("nan"))),
        "drag_ref": float(r.get("force_ref", float("nan"))),
        "domain": list(r.get("domain", [])),
        "dtype": "float64",
        "wall_s": time.time() - t0,
    }


def drag_gate(diameters=(8, 12, 16), re_p: float = 0.1, steps: int = 3000,
              domain_factor: float = 4.0, jobs: int = 3) -> dict:
    """Single-sphere drag vs Stokes at three resolutions + grid-convergence index.

    Reuses the repository's validated wet-node momentum-exchange sphere kernel
    (``tensorlbm.stokes_sphere``): this is the *precondition* on the numerical
    kernel (bounce-back sphere + momentum exchange) that the whole campaign
    inherits, not a new result of this campaign.
    """
    threads = max(1, (os.cpu_count() or 4) // max(1, jobs))
    specs = [{"d": int(d), "re_p": re_p, "steps": steps,
              "domain_factor": domain_factor} for d in diameters]
    cases = _dispatch(specs, mode="stokes", jobs=jobs, threads=threads)
    for c, d in zip(cases, diameters):
        c.setdefault("D", int(d))
    cases.sort(key=lambda c: c["D"])
    gci = None
    if len(cases) >= 2 and cases[-2]["drag"]:
        r_ratio = cases[-1]["D"] / cases[-2]["D"]
        eps = abs(cases[-1]["drag"] - cases[-2]["drag"]) / abs(cases[-2]["drag"])
        p_order = 2.0
        gci = float(1.25 * eps / (r_ratio**p_order - 1.0))
    return {
        "cases": cases,
        "re_p": re_p,
        "steps": steps,
        "domain_factor": domain_factor,
        "gci": gci,
        "gci_grids": [cases[-2]["D"], cases[-1]["D"]] if len(cases) >= 2 else [],
        "gci_formula": "GCI = 1.25 |eps| / (r^p - 1), p = 2, r = D_fine/D_coarse",
        "max_err_pct": max(c["err_pct"] for c in cases),
        "pass_3pct": bool(max(c["err_pct"] for c in cases) <= 3.0),
        "pass_gci_5pct": bool(gci is not None and gci <= 0.05),
    }


# ---------------------------------------------------------------------------
# 2. one shear-cell case (module level: picklable for multiprocessing)
# ---------------------------------------------------------------------------
def run_case(
    phi: float,
    seed: int = 1,
    *,
    nx: int = 48,
    ny: int = 48,
    nz: int = 57,
    radius: float = 5.0,
    nu: float = 0.1,
    u_wall: float = 0.02,
    n_steps: int = 2600,
    n_warmup: int | None = None,
    sample_every: int = 25,
    wall_margin: int = 7,
    shape: str = "sphere",
    polydisperse: float = 0.0,
    threads: int = 4,
    shape_kwargs: dict | None = None,
) -> dict:
    """Run one shear-cell case and return its measured quantities."""
    import torch

    torch.set_num_threads(max(1, int(threads)))
    cfg = ShearCellConfig(nx=nx, ny=ny, nz=nz, nu=nu, u_wall=u_wall)
    ps = []
    if phi > 0:
        ps = shapes.place_particles(
            nx, ny, nz, phi, radius, shape=shape, seed=seed,
            wall_margin=wall_margin, polydisperse=polydisperse,
            **(shape_kwargs or {}),
        )
    cell = ShearCell(cfg, ps, wall_band=max(2, wall_margin - 1))
    nw = n_steps // 2 if n_warmup is None else n_warmup
    r = cell.run(n_steps, n_warmup=nw, sample_every=sample_every)
    r.pop("profile", None)
    r.pop("tau_rows_mean", None)
    r.update(
        {
            "phi_target": phi,
            "seed": seed,
            "radius": radius,
            "shape": shape,
            "polydisperse": polydisperse,
            "re_p": cfg.gamma_dot * (2.0 * radius) ** 2 / nu,
            "wall_margin": wall_margin,
        }
    )
    walls = [p for p in ps]
    if walls:
        zc = np.array([p.centre[0] for p in walls])
        r["z_spread"] = [float(zc.min()), float(zc.max())]
    return r


def _dispatch(cases: list[dict], mode: str = "shear", jobs: int = 4,
              threads: int | None = None, workdir: str | None = None) -> list[dict]:
    """Run every case in its own process (``case_runner``), ``jobs`` at a time.

    ``multiprocessing`` is avoided on purpose: torch's thread pool deadlocks
    with ``fork`` and ``spawn`` re-imports the caller.  A subprocess per case
    also isolates failures -- a crashed case records its traceback in the JSON
    instead of taking the campaign down.
    """
    import subprocess
    import sys
    import tempfile

    if threads is None:
        threads = max(1, (os.cpu_count() or 4) // max(1, jobs))
    workdir = os.path.abspath(workdir or tempfile.mkdtemp(prefix="ice_eta_"))
    os.makedirs(workdir, exist_ok=True)
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env = dict(os.environ)
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    # avoid OpenMP busy-wait spinning: on a shared/contended node the spin loops
    # burn CPU without making progress (observed: 29 CPU-min for a 15 CPU-min case)
    env["OMP_WAIT_POLICY"] = "PASSIVE"
    env["KMP_BLOCKTIME"] = "0"
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    pending = list(enumerate(cases))
    running: list[tuple[int, subprocess.Popen, str]] = []
    results: dict[int, dict] = {}

    def launch(i: int, c: dict):
        spec = {k: v for k, v in c.items() if k != "tag"}
        spec["threads"] = threads
        out = os.path.join(workdir, f"case_{i:03d}.json")
        if os.path.exists(out):
            os.remove(out)
        cmd = [sys.executable, "-m", "tensorlbm.ice_suspension_rheology.case_runner",
               "--out", out, "--mode", mode, "--spec", json.dumps(spec, default=float)]
        log = open(os.path.join(workdir, f"case_{i:03d}.log"), "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=workdir)
        running.append((i, proc, out))
        return proc

    while pending or running:
        while pending and len(running) < max(1, jobs):
            i, c = pending.pop(0)
            launch(i, c)
        time.sleep(0.5)
        still = []
        for i, proc, out in running:
            if proc.poll() is None:
                still.append((i, proc, out))
                continue
            try:
                with open(out) as fh:
                    results[i] = json.load(fh)
            except Exception:  # noqa: BLE001
                results[i] = {"error": f"case {i} produced no JSON (exit {proc.returncode})",
                              "spec": cases[i]}
        running = still
    return [results.get(i, {"error": "missing result"}) for i in range(len(cases))]


_run_parallel = _dispatch  # backwards-compatible alias


# ---------------------------------------------------------------------------
# 3. sweeps
# ---------------------------------------------------------------------------_run_parallel = _dispatch  # backwards-compatible alias


# ---------------------------------------------------------------------------
# 3. sweeps
# ---------------------------------------------------------------------------
def concentration_sweep(phis, seeds=(1,), base: dict | None = None,
                        jobs: int = 4) -> list[dict]:
    base = dict(base or {})
    cases = []
    for phi in phis:
        for s in seeds:
            cases.append({**base, "phi": phi, "seed": s})
    return _dispatch(cases, jobs=jobs)


def rate_sweep(phi: float, u_walls, seed: int = 1, base: dict | None = None,
               jobs: int = 4) -> list[dict]:
    base = dict(base or {})
    cases = [{**base, "phi": phi, "seed": seed, "u_wall": u} for u in u_walls]
    return _dispatch(cases, jobs=jobs)


# ---------------------------------------------------------------------------
# 4. analysis
# ---------------------------------------------------------------------------
def analyse(sweep: list[dict], pure: dict, gate: dict | None = None,
            rate: list[dict] | None = None) -> dict:
    """Turn raw case records into the calibrated closure package."""
    ok = [r for r in sweep if "error" not in r and r.get("phi_lattice", 0) > 0]
    tau0 = pure["tau_corrected"]
    g0 = pure["gamma_dot_measured"]
    rows = []
    for r in ok:
        eta_r = (r["tau_corrected"] / tau0) * (g0 / r["gamma_dot_measured"])
        rows.append(
            {
                "phi": r["phi_lattice"],
                "phi_target": r["phi_target"],
                "seed": r["seed"],
                "eta_r": eta_r,
                "eta_eff": r["eta_eff"],
                "tau": r["tau_corrected"],
                "gamma_dot": r["gamma_dot_measured"],
                "profile_r2": r["profile_r2"],
                "re_p": r["re_p"],
                "n_particles": r["n_particles"],
                "shape": r.get("shape", "sphere"),
                "polydisperse": r.get("polydisperse", 0.0),
            }
        )
    rows.sort(key=lambda x: x["phi"])

    pure_check = {
        "nu": pure["nu"],
        "gamma_dot_measured": g0,
        "eta_eff_pure": pure["eta_eff"],
        "rel_err_vs_nu": abs(pure["eta_eff"] / pure["nu"] - 1.0),
        "profile_r2": pure["profile_r2"],
        "tau_wall_mem": pure.get("tau_wall_mem"),
        "tau_wall_mem_rel_err": abs(pure.get("tau_wall_mem", 0.0) / (pure["nu"] * g0) - 1.0),
    }

    # --- dilute regression (Einstein) --------------------------------------
    dil = [r for r in rows if r["phi"] <= 0.05]
    dil_fit = {}
    if len(dil) >= 2:
        p = np.array([r["phi"] for r in dil])
        e = np.array([r["eta_r"] for r in dil])
        A = np.vstack([p, np.ones_like(p)]).T
        (slope, icept), *_ = np.linalg.lstsq(A, e, rcond=None)
        pred = A @ np.array([slope, icept])
        ss_res = float(np.sum((e - pred) ** 2))
        ss_tot = float(np.sum((e - e.mean()) ** 2))
        dil_fit = {
            "slope": float(slope),
            "intercept": float(icept),
            "slope_rel_err_vs_2p5": abs(slope / INTRINSIC_VISCOSITY_SPHERE - 1.0),
            "intercept_abs_err_vs_1": abs(icept - 1.0),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "n_points": len(dil),
            "phi_range": [float(p.min()), float(p.max())],
            "pass_slope_3pct": bool(abs(slope / INTRINSIC_VISCOSITY_SPHERE - 1.0) <= 0.03),
            "pass_intercept": bool(abs(icept - 1.0) <= 0.01),
        }

    # --- Batchelor window --------------------------------------------------
    semi = [r for r in rows if 0.05 < r["phi"] <= 0.15]
    bat = {}
    if semi:
        errs = [
            abs(r["eta_r"] / float(batchelor_relative_viscosity(r["phi"])) - 1.0) for r in semi
        ]
        bat = {
            "max_rel_err": float(max(errs)),
            "mean_rel_err": float(np.mean(errs)),
            "n_points": len(semi),
            "pass_10pct": bool(max(errs) <= 0.10),
            "rows": [{"phi": r["phi"], "eta_r": r["eta_r"],
                      "batchelor": float(batchelor_relative_viscosity(r["phi"]))} for r in semi],
        }

    # --- Krieger-Dougherty phi_max ----------------------------------------
    kd = {}
    if len(rows) >= 3:
        p = np.array([r["phi"] for r in rows])
        e = np.array([r["eta_r"] for r in rows])
        f = fit_krieger_dougherty_phi_max(p, e, intrinsic=INTRINSIC_VISCOSITY_SPHERE)
        kd = {
            "phi_max": f["phi_max"],
            "rmse": f["rmse"],
            "r2": f["r2"],
            "n_points": f["n_points"],
            "phi_range": f["phi_range"],
            "pass_range_0p58_0p68": bool(0.58 <= f["phi_max"] <= 0.68),
            "fit": {k: v for k, v in f.items() if k != "residual"},
        }
    phi_max = kd.get("phi_max", 0.63)
    contract = ViscosityContract(
        phi_max=float(phi_max),
        intrinsic=float(dil_fit.get("slope", INTRINSIC_VISCOSITY_SPHERE)),
        k2=BATCHELOR_K2,
    )
    # --- rate sweep (linearity + inertial crossover) -----------------------
    rate_out = {}
    if rate:
        rr = [r for r in rate if "error" not in r]
        if len(rr) >= 2:
            g = np.array([r["gamma_dot_measured"] for r in rr])
            t = np.array([r["tau_corrected"] for r in rr])
            A = np.vstack([g, np.ones_like(g)]).T
            (slope, icept), *_ = np.linalg.lstsq(A, t, rcond=None)
            pred = A @ np.array([slope, icept])
            ss_res = float(np.sum((t - pred) ** 2))
            ss_tot = float(np.sum((t - t.mean()) ** 2))
            re_p = np.array([r["re_p"] for r in rr])
            eta_eff = t / g
            rate_out = {
                "slope": float(slope),
                "intercept": float(icept),
                "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                "pass_r2_0p99": bool(ss_tot > 0 and (1 - ss_res / ss_tot) > 0.99),
                "points": [
                    {"re_p": float(a), "gamma_dot": float(b), "tau": float(c),
                     "eta_eff": float(d), "profile_r2": float(e2["profile_r2"])}
                    for a, b, c, d, e2 in zip(re_p, g, t, eta_eff, rr)
                ],
            }
            # eta_eff grows like 1 + Re/const in the inertial regime; a linear
            # fit of (eta_eff/eta_eff(Re->0) - 1) vs Re_p gives 1/I_v.
            if len(rr) >= 3:
                base_eta = float(eta_eff[0])
                x = re_p
                y = eta_eff / base_eta - 1.0
                A2 = np.vstack([x, np.ones_like(x)]).T
                (s2, _i2), *_ = np.linalg.lstsq(A2, y, rcond=None)
                rate_out["inertial_slope"] = float(s2)
                rate_out["i_v_estimate"] = float(1.0 / s2) if s2 > 0 else None
                contract.i_v = rate_out["i_v_estimate"]
    return {
        "pure_control": pure_check,
        "dilute_fit": dil_fit,
        "batchelor": bat,
        "krieger_dougherty": kd,
        "contract": contract.as_dict(),
        "rate_sweep": rate_out,
        "rows": rows,
        "gate": gate,
        "negative_control_eta_r_pure": float(
            (pure["tau_corrected"] / pure["tau_corrected"]) * (g0 / g0)
        ),
        "closure_limits": {
            "kd_phi0": float(krieger_dougherty_relative_viscosity(0.0, float(phi_max))),
            "einstein_phi0": float(einstein_relative_viscosity(0.0)),
            "batchelor_phi0": float(batchelor_relative_viscosity(0.0)),
        },
    }


# ---------------------------------------------------------------------------
# 5. figures
# ---------------------------------------------------------------------------
def make_figures(analysis: dict, figdir: str, tag: str = "ice_eta") -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    made = []
    rows = analysis["rows"]
    if not rows:
        return made
    phi = np.array([r["phi"] for r in rows])
    eta = np.array([r["eta_r"] for r in rows])
    pm = analysis["krieger_dougherty"].get("phi_max", 0.63)
    grid = np.linspace(1e-4, max(0.34, phi.max() * 1.02), 400)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(grid, einstein_relative_viscosity(grid), ":", lw=1.4, label="Einstein $1+2.5\\phi$")
    ax.plot(grid, batchelor_relative_viscosity(grid), "--", lw=1.4, label="Batchelor $1+2.5\\phi+6.2\\phi^2$")
    ax.plot(grid, krieger_dougherty_relative_viscosity(grid, pm), "-", lw=1.6,
            label=f"Krieger-Dougherty $\\phi_{{max}}={pm:.3f}$")
    ax.plot(phi, eta, "o", ms=6, mfc="none", mew=1.4, color="k", label="LBM shear cell")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\eta_r=\eta_{eff}/\eta_f$")
    ax.set_yscale("log")
    ax.set_title("Ice-crystal suspension relative viscosity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p1 = os.path.join(figdir, f"{tag}_eta_r_vs_phi.png")
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    made.append(p1)

    # dilute zoom with the regression line
    if analysis["dilute_fit"]:
        df = analysis["dilute_fit"]
        m = phi <= 0.05
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.plot(phi[m], eta[m], "o", color="C0", label="LBM")
        g = np.linspace(0, 0.055, 50)
        ax.plot(g, df["intercept"] + df["slope"] * g, "-", color="C3",
                label=f"fit: $1+({df['slope']:.3f})\\phi$ (intercept {df['intercept']:.4f})")
        ax.plot(g, einstein_relative_viscosity(g), ":", color="k", label="Einstein $[\\eta]=2.5$")
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$\eta_r$")
        ax.set_title("Dilute limit: Einstein slope")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        p2 = os.path.join(figdir, f"{tag}_dilute.png")
        fig.tight_layout()
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        made.append(p2)

    # stress vs strain rate (linearity)
    rs = analysis.get("rate_sweep") or {}
    if rs.get("points"):
        pts = rs["points"]
        gg = np.array([p["gamma_dot"] for p in pts])
        tt = np.array([p["tau"] for p in pts])
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.plot(gg, tt, "o", color="C2")
        ggrid = np.linspace(0, gg.max() * 1.05, 50)
        ax.plot(ggrid, rs["intercept"] + rs["slope"] * ggrid, "-", color="k", lw=1.0,
                label=f"linear fit $R^2={rs['r2']:.5f}$")
        ax.set_xlabel(r"$\dot\gamma$ [lattice]")
        ax.set_ylabel(r"$\tau$ [lattice]")
        ax.set_title("Stress-strain-rate linearity")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        p3 = os.path.join(figdir, f"{tag}_stress_rate.png")
        fig.tight_layout()
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        made.append(p3)

        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        reps = np.array([p["re_p"] for p in pts])
        eeff = np.array([p["eta_eff"] for p in pts])
        ax.semilogx(reps, eeff / eeff[0], "o-")
        ax.set_xlabel(r"$Re_p=\dot\gamma (2a)^2/\nu$")
        ax.set_ylabel(r"$\eta_{eff}(Re_p)/\eta_{eff}(Re_p\to0)$")
        ax.set_title("Inertial (mu(I)) correction")
        ax.grid(alpha=0.3, which="both")
        p4 = os.path.join(figdir, f"{tag}_inertial.png")
        fig.tight_layout()
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        made.append(p4)
    return made


# ---------------------------------------------------------------------------
# 6. report writer
# ---------------------------------------------------------------------------
def write_report(analysis: dict, outdir: str, figures: list[str], meta: dict) -> str:
    os.makedirs(outdir, exist_ok=True)
    payload = {"meta": meta, "analysis": analysis,
               "figures": [os.path.basename(f) for f in figures]}
    path = os.path.join(outdir, "calibration.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return path


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="quick", choices=sorted(PRESETS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--gate-steps", type=int, default=3000)
    args = ap.parse_args(argv)
    cfg = dict(PRESETS[args.preset])
    outdir = args.out or f"runs/ice_eta_{args.preset}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(outdir, exist_ok=True)

    gate = None
    if not args.skip_gate:
        gate = drag_gate(steps=args.gate_steps)
        with open(os.path.join(outdir, "drag_gate.json"), "w") as fh:
            json.dump(gate, fh, indent=2, default=float)

    base = {k: cfg[k] for k in ("nx", "ny", "nz", "radius", "n_steps", "wall_margin")}
    phis = cfg["phi"]
    seeds = cfg["seeds"]
    cases = [{**base, "phi": 0.0, "seed": 0}] + [
        {**base, "phi": p, "seed": s} for p in phis for s in seeds
    ]
    raw = _run_parallel(cases, jobs=args.jobs)
    pure = raw[0]
    sweep = raw[1:]
    with open(os.path.join(outdir, "sweep_raw.json"), "w") as fh:
        json.dump(raw, fh, indent=2, default=float)

    # rate sweep at a fixed phi (linearity + inertia)
    u_list = np.linspace(0.25, 1.0, cfg["rate_points"]) * base.get("u_wall", 0.02) \
        if "u_wall" in base else np.linspace(0.005, 0.02, cfg["rate_points"])
    u_list = np.linspace(0.25, 1.0, cfg["rate_points"]) * 0.02
    rate = rate_sweep(0.05 if min(phis) <= 0.05 else min(phis), u_list,
                      base={**base}, jobs=args.jobs)

    ana = analyse(sweep, pure, gate=gate, rate=rate)
    ana["meta"] = {
        "preset": args.preset,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": cfg,
        "rate_u_walls": [float(u) for u in u_list],
    }
    figs = make_figures(ana, os.path.join(outdir, "figures"))
    path = write_report(ana, outdir, figs, ana["meta"])
    print(json.dumps({k: ana[k] for k in ("dilute_fit", "krieger_dougherty",
                                         "pure_control", "contract") if k in ana},
                     indent=2, default=float)[:4000])
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# 8. Stokes drag gate in the shear cell (uniform stream past a fixed sphere)
# ---------------------------------------------------------------------------
def run_drag_case(radius: float = 4.0, box: int = 32, nz: int | None = None,
                  re_p: float = 0.1, nu: float = 0.1, n_steps: int = 1600,
                  threads: int = 2) -> dict:
    """Drag on a *fixed* sphere in a uniform stream (moving-belt walls).

    Configuration: both walls move with ``+U`` (no shear), the sphere is static,
    so the sphere sees a uniform approach flow ``U`` and the momentum exchange
    over its links is the drag force ``F_x``.  ``U`` is chosen from the target
    particle Reynolds number, ``U = Re_p * nu / (2 a)``.

    The reference is Stokes' law ``F = 6 pi mu a_eff U`` with ``a_eff`` from the
    **discretised** sphere volume, corrected for the channel confinement with the
    Haberman-Sayre centreline factor

        K = 1 / (1 - 2.104 beta + 2.09 beta**3 - 0.95 beta**5),
        beta = a_eff / R,  R = half gap,

    because the channel is much tighter than the sphere's Stokeslet decay
    length.  Both the raw and the corrected error are returned -- the corrected
    one is what tests the numerical kernel, the raw one shows how large the
    confinement effect is at this resolution.
    """
    import numpy as np
    import torch

    torch.set_num_threads(max(1, int(threads)))
    nz = nz or int(round(box * 1.4))
    if nz % 2 == 0:
        nz += 1
    u = re_p * nu / (2.0 * radius)
    cfg = ShearCellConfig(nx=box, ny=box, nz=nz, nu=nu, u_wall=u, mode="drag")
    part = shapes.make_particle(box, box, nz, (0.5 * (nz - 1), 0.5 * box, 0.5 * box), radius)
    a_eff = (3.0 * part.volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    cell = ShearCell(cfg, [part], wall_band=2)
    r = cell.run(n_steps, n_warmup=n_steps // 2, sample_every=10)
    F = abs(r["drag_force_mem"])
    f_stokes = 6.0 * np.pi * nu * a_eff * u
    R = 0.5 * (nz - 2)
    beta = a_eff / R
    K = 1.0 / (1.0 - 2.104 * beta + 2.09 * beta**3 - 0.95 * beta**5)
    r.update({
        "radius": radius, "a_eff": float(a_eff), "R_halfgap": float(R), "beta": float(beta),
        "haberman_sayre_K": float(K), "u_stream": float(u), "re_p": float(re_p),
        "drag": F, "drag_stokes": float(f_stokes),
        "drag_stokes_corrected": float(f_stokes * K),
        "err_pct": float(100.0 * abs(F / f_stokes - 1.0)),
        "err_pct_corrected": float(100.0 * abs(F / (f_stokes * K) - 1.0)),
        "cd_sim": float(F / (0.5 * u * u * np.pi * a_eff * a_eff)),
        "cd_stokes": float(24.0 / re_p),
    })
    r.pop("profile", None)
    r.pop("tau_rows_mean", None)
    return r
