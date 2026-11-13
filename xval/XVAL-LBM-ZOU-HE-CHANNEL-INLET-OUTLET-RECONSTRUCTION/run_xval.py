#!/usr/bin/env python3
"""XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION — round-2 driver.

Arms
----
A0  analytic steady plane Poiseuille solution (Navier–Stokes, no fitting)
A1  TensorLBM repo code: ``zou_he_inlet_velocity`` / ``zou_he_outlet_pressure``
    (and, when present, ``zou_he_inlet_pressure``) from
    ``src/tensorlbm/boundary/boundaries.py``, driven by the repo D2Q9 kernels
    ``tensorlbm.solver.collide_bgk`` / ``tensorlbm.solver.stream`` and
    ``tensorlbm.d2q9.equilibrium``.  Walls: ``bounce_back_cells`` on the two
    solid rows (full-way bounce-back ⇒ no-slip plane at the wall node centre).

Metrics are the predeclared ones in ``PREREG.md`` (primary: ``dp_max``).

Usage
-----
    python run_xval.py --out artifacts            # full suite
    python run_xval.py --out artifacts --quick    # coarse cases only
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

import tensorlbm.boundary.boundaries as BND  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

zou_he_inlet_velocity = BND.zou_he_inlet_velocity
zou_he_outlet_pressure = BND.zou_he_outlet_pressure
bounce_back_cells = BND.bounce_back_cells
zou_he_inlet_pressure = getattr(BND, "zou_he_inlet_pressure", None)

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# analytic reference (A0)
# ---------------------------------------------------------------------------
def u_poiseuille(y: torch.Tensor, h: float, u_max: float) -> torch.Tensor:
    """Plane Poiseuille profile, no-slip planes at ``y = 0`` and ``y = h``."""
    s = y / h
    return u_max * 4.0 * s * (1.0 - s)


# ---------------------------------------------------------------------------
# A1: channel driver built from repo kernels + repo boundary functions
# ---------------------------------------------------------------------------
def run_channel(
    *,
    nx: int,
    ny: int,
    tau: float,
    steps: int,
    mode: str,
    u_in: float = 0.05,
    u_in_profile: torch.Tensor | None = None,
    rho_in: float = 1.0,
    rho_out: float = 1.0,
    measure_cols: tuple[int, ...] = (),
    sample_every: int = 500,
    drift_tol: float = 1e-7,
) -> dict:
    """March a 2-D channel to steady state; return profiles and diagnostics.

    ``mode="velocity"``: Zou/He velocity inlet (uniform or profile) + Zou/He
    pressure outlet.  ``mode="pressure"``: Zou/He pressure inlet + pressure
    outlet (requires ``zou_he_inlet_pressure``).
    """
    if mode == "pressure" and zou_he_inlet_pressure is None:
        msg = "zou_he_inlet_pressure not implemented in this revision"
        raise NotImplementedError(msg)

    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = True
    wall[-1, :] = True
    obstacle = torch.zeros_like(wall)

    rho0 = torch.ones((ny, nx), dtype=DTYPE)
    ux0 = torch.zeros((ny, nx), dtype=DTYPE)
    if mode == "velocity":
        ux0[1:-1, :] = u_in
        # walls stay at rest
    uy0 = torch.zeros_like(ux0)
    f = equilibrium(rho0, ux0, uy0)

    prof_in = u_in_profile
    history: list[dict] = []
    prev_ux: torch.Tensor | None = None
    drift = float("inf")
    used_steps = 0
    for n in range(1, steps + 1):
        f = collide_bgk(f, tau)
        f = stream(f)
        if mode == "velocity":
            f = zou_he_inlet_velocity(f, prof_in if prof_in is not None else u_in)
            f = zou_he_outlet_pressure(f, rho_out)
        else:
            f = zou_he_inlet_pressure(f, rho_in)
            f = zou_he_outlet_pressure(f, rho_out)
        f = bounce_back_cells(f, wall)
        f = bounce_back_cells(f, obstacle)
        used_steps = n
        if n % sample_every == 0:
            rho, ux, uy = macroscopic(f)
            mid_col = measure_cols[0] if measure_cols else nx // 2
            cur = ux[1:-1, mid_col]
            drift = (
                float("inf")
                if prev_ux is None
                else float((cur - prev_ux).abs().max() / max(float(cur.abs().max()), 1e-30))
            )
            hist = {"step": n, "umax": float(cur.max()), "drift": drift}
            if mode == "pressure":
                hist["dxy_in"] = float(rho[1:-1, 0].mean())
                hist["dxy_out"] = float(rho[1:-1, -1].mean())
            history.append(hist)
            prev_ux = cur
            if drift < drift_tol and n > 4 * sample_every:
                break

    rho, ux, uy = macroscopic(f)
    fluid = slice(1, -1)
    out: dict = {
        "nx": nx,
        "ny": ny,
        "h": ny - 1,  # full-way BB: no-slip planes at the wall node centres
        "tau": tau,
        "nu": (tau - 0.5) / 3.0,
        "mode": mode,
        "steps": used_steps,
        "drift": drift,
        "history": history,
    }
    cols = [c for c in (measure_cols or (nx // 2,)) if 0 < c < nx]
    out["profiles"] = {
        str(c): {
            "ux": ux[fluid, c].tolist(),
            "rho": rho[fluid, c].tolist(),
            "Q": float((rho[fluid, c] * ux[fluid, c]).sum()),
        }
        for c in cols
    }
    out["rho_in_col"] = float(rho[fluid, 0].mean())
    out["rho_out_col"] = float(rho[fluid, -1].mean())
    out["Q_in"] = float((rho[fluid, 0] * ux[fluid, 0]).sum())
    out["Q_out"] = float((rho[fluid, -1] * ux[fluid, -1]).sum())
    # cross-section flux uniformity (mass conservation over all interior columns)
    q_all = (rho[fluid, 1:-1] * ux[fluid, 1:-1]).sum(dim=0)
    out["Q_interior_min"] = float(q_all.min())
    out["Q_interior_max"] = float(q_all.max())
    out["uy_max"] = float(uy[fluid, :].abs().max())
    return out


# ---------------------------------------------------------------------------
# A0/A1 comparison metrics (predeclared in PREREG.md)
# ---------------------------------------------------------------------------
def compare_profile(
    run: dict,
    *,
    col: int,
    u_max_ana: float | None = None,
    fit_amplitude: bool = False,
) -> dict:
    """Compare the LBM profile at ``col`` against the analytic parabola."""
    prof = run["profiles"][str(col)]
    uv = torch.tensor(prof["ux"], dtype=DTYPE)
    y = torch.arange(1, run["ny"] - 1, dtype=DTYPE)
    h = float(run["h"])

    if fit_amplitude:
        # amplitude from the discrete flux (shape test, amplitude not predicted)
        slope = float(u_poiseuille(y, h, 1.0).sum())
        q = float(prof["Q"])
        rho_bar = float(torch.tensor(prof["rho"], dtype=DTYPE).mean())
        u_max = q / (rho_bar * slope)
        amplitude_source = "flux-matched"
    else:
        u_max = float(u_max_ana)
        amplitude_source = "a-priori (G H^2 / 8nu)"

    ana = u_poiseuille(y, h, u_max)
    ref = float(ana.abs().max())
    diff = (uv - ana).abs()
    dp_max = float(diff.max() / ref)
    dp_l2 = float(torch.sqrt((diff**2).mean()) / ref)
    # wall-offset diagnostic: linear extrapolation of the first two fluid rows
    u1, u2 = float(uv[0]), float(uv[1])
    y_wall = 1.0 - u1 / (u2 - u1) if abs(u2 - u1) > 1e-14 else float("nan")
    return {
        "col": col,
        "h": h,
        "u_max_ana": u_max,
        "u_max_got": float(uv.max()),
        "amplitude_source": amplitude_source,
        "amplitude_err": (float(uv.max()) - u_max) / u_max,
        "dp_max": dp_max,
        "dp_l2": dp_l2,
        "ux_from_lbm": uv.tolist(),
        "ux_analytic": ana.tolist(),
        "slip_row1": u1,
        "wall_offset": y_wall,
        "Q": prof["Q"],
    }


def flux_error(run: dict) -> float:
    qi, qo = run["Q_in"], run["Q_out"]
    return abs(qo - qi) / abs(qi)


# ---------------------------------------------------------------------------
# M1: reconstruction closure (exact algebra, non-equilibrium input field)
# ---------------------------------------------------------------------------
def closure_test(ny: int = 18, nx: int = 12) -> dict:
    torch.manual_seed(19970317)
    f = torch.rand((9, ny, nx), dtype=DTYPE) + 0.1  # arbitrary, non-equilibrium
    u_in = torch.linspace(0.02, 0.09, ny, dtype=DTYPE)
    uy_in = torch.linspace(-0.02, 0.02, ny, dtype=DTYPE)

    # --- existing arm: velocity inlet (west) ---
    g = zou_he_inlet_velocity(f, u_in, uy_in)
    rho_g, ux_g, uy_g = macroscopic(g)
    f0, f2, f4 = f[0, :, 0], f[2, :, 0], f[4, :, 0]
    f3, f6, f7 = f[3, :, 0], f[6, :, 0], f[7, :, 0]
    rho_target = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u_in)
    inlet = {
        "rho_rel_err": float(((rho_g[:, 0] - rho_target).abs() / rho_target).max()),
        "ux_abs_err": float((ux_g[:, 0] - u_in).abs().max()),
        "uy_abs_err": float((uy_g[:, 0] - uy_in).abs().max()),
        # untouched populations must survive bit-exactly
        "known_unchanged": bool(
            torch.equal(g[3, :, 0], f3)
            and torch.equal(g[6, :, 0], f6)
            and torch.equal(g[7, :, 0], f7)
            and torch.equal(g[2, :, 0], f2)
            and torch.equal(g[4, :, 0], f4)
        ),
    }

    # --- existing arm: pressure outlet (east) ---
    rho_out = 1.003
    g2 = zou_he_outlet_pressure(f, rho_out)
    rho_g2, ux_g2, uy_g2 = macroscopic(g2)
    outlet = {
        "rho_rel_err": float(((rho_g2[:, -1] - rho_out).abs() / rho_out).max()),
        "uy_abs_err": float(uy_g2[:, -1].abs().max()),
        "known_unchanged": bool(
            torch.equal(g2[1, :, -1], f[1, :, -1])
            and torch.equal(g2[5, :, -1], f[5, :, -1])
            and torch.equal(g2[8, :, -1], f[8, :, -1])
        ),
    }
    # consistency of the reconstructed outlet momentum with the mass constraint
    f1, f5, f8 = f[1, :, -1], f[5, :, -1], f[8, :, -1]
    ux_expect = -1.0 + (f[0, :, -1] + f[2, :, -1] + f[4, :, -1] + 2.0 * (f1 + f5 + f8)) / rho_out
    outlet["ux_vs_mass_constraint"] = float((ux_g2[:, -1] - ux_expect).abs().max())

    result = {
        "inlet_velocity": inlet,
        "outlet_pressure": outlet,
        "closure_err": max(
            inlet["rho_rel_err"],
            inlet["ux_abs_err"],
            inlet["uy_abs_err"],
            outlet["rho_rel_err"],
            outlet["uy_abs_err"],
        ),
    }

    # --- extension arm: pressure inlet (west), if implemented ---
    if zou_he_inlet_pressure is not None:
        uy_in2 = uy_in
        g3 = zou_he_inlet_pressure(f, 1.004, uy_in2)
        rho_g3, ux_g3, uy_g3 = macroscopic(g3)
        inlet_p = {
            "rho_rel_err": float(((rho_g3[:, 0] - 1.004).abs() / 1.004).max()),
            "uy_abs_err": float((uy_g3[:, 0] - uy_in2).abs().max()),
        }
        ux_expect3 = (
            1.0
            - (f[0, :, 0] + f[2, :, 0] + f[4, :, 0] + 2.0 * (f3 + f6 + f7)) / 1.004
        )
        inlet_p["ux_vs_mass_constraint"] = float((ux_g3[:, 0] - ux_expect3).abs().max())
        result["inlet_pressure"] = inlet_p
        result["closure_err"] = max(
            result["closure_err"],
            inlet_p["rho_rel_err"],
            inlet_p["uy_abs_err"],
            inlet_p["ux_vs_mass_constraint"],
        )
    return result


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument(
        "--cases",
        default="all",
        choices=("all", "coarse", "fine"),
        help="coarse: C1+C3+P profile+C5 sweep; fine: C2+C4 (for chunked local runs)",
    )
    ap.add_argument("--merge", nargs="*", default=None, help="merge part files into the final JSON")
    args = ap.parse_args()

    torch.set_num_threads(4)
    t0 = time.time()
    if args.merge:
        parts = [json.loads(Path(m).read_text()) for m in args.merge]
        merged = dict(parts[0])
        res_all: dict = {}
        for p_ in parts:
            for k, v in p_["results"].items():
                # never let a "not_run (chunk)" placeholder overwrite real data
                if isinstance(v, str) and isinstance(res_all.get(k), list):
                    continue
                res_all[k] = v
        merged["results"] = res_all
        merged["wall_time_s"] = sum(p_.get("wall_time_s", 0.0) for p_ in parts)
        merged["chunked_local_run"] = [str(m) for m in args.merge]
        gating = {
            k: v["dp_max"] for k, v in res_all.items() if isinstance(v, dict) and "dp_max" in v
        }
        merged["verdict"] = {
            "primary_metric": "dp_max",
            "threshold": 0.15,
            "stretch": 0.03,
            "dp_max_all": gating,
            "worst": max(gating.values()) if gating else None,
            "PASS": bool(gating) and max(gating.values()) < 0.15,
            "STRETCH": bool(gating) and max(gating.values()) < 0.03,
            "closure_ok": merged["closure"]["closure_err"] < 1e-12,
        }
        outpath = HERE / args.out / "zou_he_channel_results.json"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(json.dumps(merged, indent=1))
        print(json.dumps(merged["verdict"], indent=1))
        print("merged ->", outpath)
        return 0
    outdir = HERE / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "anchor": "XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION",
        "env": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dtype": "float64",
            "threads": torch.get_num_threads(),
        },
        "closure": closure_test(),
    }

    cases = []
    results = {}

    # ---- velocity-inlet arm (pre-existing repo BC pair) -------------------
    for tag, nx, ny in (("C1_coarse", 120, 18), ("C2_fine", 220, 34)):
        if args.quick and tag != "C1_coarse":
            continue
        if args.cases == "coarse" and tag != "C1_coarse":
            continue
        if args.cases == "fine" and tag != "C2_fine":
            continue
        u_in = 0.05
        run = run_channel(
            nx=nx,
            ny=ny,
            tau=0.8,
            steps=args.steps,
            mode="velocity",
            u_in=u_in,
            rho_out=1.0,
            measure_cols=(nx // 2, nx - 4),
        )
        for col in (nx // 2, nx - 4):
            cmp_ = compare_profile(run, col=col, fit_amplitude=True)
            results[f"{tag}_col{col}"] = {
                "case": tag,
                "mode": "velocity",
                "nx": nx,
                "ny": ny,
                "u_in": u_in,
                "steps": run["steps"],
                "drift": run["drift"],
                "flux_err": flux_error(run),
                "uy_max": run["uy_max"],
                **{k: v for k, v in cmp_.items()},
            }
        results[f"{tag}_run"] = {k: v for k, v in run.items() if k != "profiles"}

    # ---- parabolic inlet profile (repo profile support) -------------------
    if args.cases == "fine":
        nx, ny = None, None
    else:
        nx, ny = (120, 18)
    h = (ny - 1) if ny else 17
    y = torch.arange(1, (ny or 18) - 1, dtype=DTYPE)
    # repo BC takes the full column profile (ny,); wall rows are re-imposed by
    # bounce-back afterwards, so their entries are irrelevant -> set to 0.
    prof = torch.zeros(ny or 18, dtype=DTYPE)
    prof[1:-1] = u_poiseuille(y, h, 0.06)
    if args.cases != "fine":
      run = run_channel(
        nx=nx,
        ny=ny,
        tau=0.8,
        steps=args.steps,
        mode="velocity",
        u_in_profile=prof,
        rho_out=1.0,
        measure_cols=(nx // 2, nx - 4),
    )
      cmp_ = compare_profile(run, col=nx // 2, u_max_ana=0.06)
      results["P_inlet_profile"] = {
        "case": "P_inlet_profile",
        "mode": "velocity-profile",
        "nx": nx,
        "ny": ny,
        "u_max_in": 0.06,
        "steps": run["steps"],
        "drift": run["drift"],
        "flux_err": flux_error(run),
        **cmp_,
      }

    # ---- pressure-driven arm (Zou/He pressure inlet + outlet) ------------
    if zou_he_inlet_pressure is not None:
        for tag, nx, ny in (("C3_coarse", 120, 18), ("C4_fine", 220, 34)):
            if args.quick and tag != "C3_coarse":
                continue
            if args.cases == "coarse" and tag != "C3_coarse":
                continue
            if args.cases == "fine" and tag != "C4_fine":
                continue
            d_rho = 0.004
            rho_in, rho_out = 1.0 + d_rho / 2, 1.0 - d_rho / 2
            run = run_channel(
                nx=nx,
                ny=ny,
                tau=0.8,
                steps=args.steps,
                mode="pressure",
                rho_in=rho_in,
                rho_out=rho_out,
                measure_cols=(nx // 2, nx - 4),
            )
            h = ny - 1
            nu = (0.8 - 0.5) / 3.0
            L = nx - 1
            G = d_rho / 3.0 / L  # dp/dx, p = rho/3
            u_max = G * h * h / (8.0 * nu)
            for col in (nx // 2, nx - 4):
                cmp_ = compare_profile(run, col=col, u_max_ana=u_max)
                results[f"{tag}_col{col}"] = {
                    "case": tag,
                    "mode": "pressure",
                    "nx": nx,
                    "ny": ny,
                    "d_rho": d_rho,
                    "G": G,
                    "steps": run["steps"],
                    "drift": run["drift"],
                    "flux_err": flux_error(run),
                    "uy_max": run["uy_max"],
                    "Re": u_max * h / nu,
                    "Ma": u_max * math.sqrt(3.0),
                    **cmp_,
                }
            results[f"{tag}_run"] = {k: v for k, v in run.items() if k != "profiles"}

        # ---- C5: driving-level sweep (parameter sensitivity) -------------
        sweep = []
        if args.cases == "fine":
            results["C5_drho_sweep"] = "not_run (fine chunk)"
            sweep = None
        for d_rho in ((0.001, 0.002, 0.004) if sweep is not None else ()):
            nx, ny = 120, 18
            rho_in, rho_out = 1.0 + d_rho / 2, 1.0 - d_rho / 2
            run = run_channel(
                nx=nx,
                ny=ny,
                tau=0.8,
                steps=args.steps,
                mode="pressure",
                rho_in=rho_in,
                rho_out=rho_out,
                measure_cols=(nx // 2,),
            )
            h, nu, L = ny - 1, 0.1, nx - 1
            G = d_rho / 3.0 / L
            u_max = G * h * h / (8.0 * nu)
            cmp_ = compare_profile(run, col=nx // 2, u_max_ana=u_max)
            sweep.append(
                {
                    "d_rho": d_rho,
                    "u_max_ana": u_max,
                    "u_max_got": cmp_["u_max_got"],
                    "dp_max": cmp_["dp_max"],
                    "dp_l2": cmp_["dp_l2"],
                    "amplitude_err": cmp_["amplitude_err"],
                    "flux_err": flux_error(run),
                    "steps": run["steps"],
                }
            )
        if sweep is not None:
            results["C5_drho_sweep"] = sweep
    else:
        results["C5_drho_sweep"] = "not_run: zou_he_inlet_pressure absent"

    report["results"] = results
    report["wall_time_s"] = time.time() - t0

    # ---- verdict (predeclared metric dp_max) -----------------------------
    gating = {
        k: v["dp_max"]
        for k, v in results.items()
        if isinstance(v, dict) and "dp_max" in v
    }
    report["verdict"] = {
        "primary_metric": "dp_max",
        "threshold": 0.15,
        "stretch": 0.03,
        "dp_max_all": gating,
        "worst": max(gating.values()) if gating else None,
        "PASS": bool(gating) and max(gating.values()) < 0.15,
        "STRETCH": bool(gating) and max(gating.values()) < 0.03,
        "closure_ok": report["closure"]["closure_err"] < 1e-12,
    }

    path = outdir / "zou_he_channel_results.json"
    path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["verdict"], indent=1))
    print(f"wrote {path}  ({report['wall_time_s']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())