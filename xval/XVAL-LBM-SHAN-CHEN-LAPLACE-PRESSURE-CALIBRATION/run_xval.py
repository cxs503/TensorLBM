#!/usr/bin/env python3
"""XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION — cross-validation driver.

Anchor: ``XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION``
Reference practice: ``lbm-principles-practice/code`` ``shan-chen`` Laplace
pressure calibration (SCMP pseudopotential droplet, ``dp = sigma / R`` fit).

What is cross-validated (three independent legs)
------------------------------------------------
L1  *Implementation leg.*  The TensorLBM library path
    (``tensorlbm.multiphase.collide_sc_single_component`` + ``tensorlbm.solver.stream``)
    against an independent from-scratch NumPy SCMP solver (``ref_sc.py``) that
    imports nothing from ``tensorlbm``.  Same lattice ordering only, so that
    distribution functions can be compared cell-by-cell (step 0 equivalence is
    enforced in ``tests/test_xval_sc_laplace.py``).
L2  *Thermodynamic leg.*  The simulated bulk densities are compared with the
    Maxwell equal-area construction on the Shan-Chen equation of state
    ``p(rho) = cs^2 rho + (G_eff cs^2 / 2) psi^2`` — an analytic reference that
    does not use the lattice solver at all.
L3  *Protocol leg.*  Sensitivity of the calibrated surface tension to choices
    the earlier ``benchmarks/verified/laplace_droplet`` acceptance did not
    probe: pressure-band radius, periodic-domain extent (``L = 4R`` vs ``6R``
    vs ``8R``) and forcing scheme (velocity-shift vs Guo).

Physics
-------
A stationary 2-D liquid drop in a periodic box obeys Young-Laplace
``dp = p_in - p_out = sigma / R`` (cylinder).  Fitting ``dp`` against ``1/R``
over several radii yields ``sigma``; linearity (R^2) and a vanishing intercept
are the consistency checks.  All quantities are lattice units,
``psi(rho) = 1 - exp(-rho)``, ``G_lib = +5`` (= textbook ``G_eff = -5``),
``tau = 1`` (``nu = 1/6``), D2Q9.

Usage
-----
    python run_xval.py            # full study  (artifacts/result.json + figures)
    python run_xval.py --quick    # reduced study (CI / smoke)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))  # canonical TensorLBM package
sys.path.insert(0, str(HERE))

import ref_sc as REF  # noqa: E402  independent reference implementation

# ---------------------------------------------------------------------------
# Fixed model / protocol parameters (lattice units)
# ---------------------------------------------------------------------------
CFG = {
    "model": "SCMP single-component Shan-Chen (Shan & Chen 1993/1994)",
    "psi": "psi(rho) = 1 - exp(-rho)  (rho0 = 1)",
    "G_lib": 5.0,  # library argument
    "G_eff": -5.0,  # textbook convention (attraction), = -G_lib
    "tau": 1.0,  # nu = 1/6
    "lattice": "D2Q9",
    "forcing_default": "velocity_shift",
    "rho_l_init": 1.957,
    "rho_v_init": 0.1596,
    "w_int": 4.0,
    "domain_rule": "L = 4R (same as benchmarks/verified/laplace_droplet)",
    "bands": (0.5, 1.5),  # r <= 0.5 R_eq (liquid) / r >= 1.5 R_eq (vapour)
    "min_steps": 4000,
    "max_steps": 16000,
    "sample_every": 500,
    "conv_dp": 2e-3,
    "conv_r": 2e-3,
    "sigma_locked_doc": 0.0561,  # value locked in tensorlbm.multiphase docs
    "criteria_pct": 15.0,  # owner accuracy standard
}


# ---------------------------------------------------------------------------
# Solvers (identical model, two independent code paths)
# ---------------------------------------------------------------------------
class TorchSolver:
    """TensorLBM library path."""

    name = "tensorlbm"

    def __init__(self, forcing: str = "velocity_shift", threads: int = 8):
        import torch

        torch.set_num_threads(threads)
        self.torch = torch
        from tensorlbm.d2q9 import equilibrium, macroscopic
        from tensorlbm.multiphase import collide_sc_single_component, psi_exp
        from tensorlbm.solver import stream

        self._eq = equilibrium
        self._mac = macroscopic
        self._collide = collide_sc_single_component
        self._psi = psi_exp
        self._stream = stream
        self.forcing = forcing

    def init(self, rho: np.ndarray):
        t = self.torch
        r = t.tensor(np.ascontiguousarray(rho), dtype=t.float32)
        z = t.zeros_like(r)
        return self._eq(r, z, z)

    def step(self, f):
        return self._stream(
            self._collide(f, G=CFG["G_lib"], tau=CFG["tau"], psi_fn=self._psi, forcing=self.forcing)
        )

    def fields(self, f) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho, ux, uy = self._mac(f)
        return rho.numpy(), ux.numpy(), uy.numpy()

    def force(self, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = self.torch
        r = t.tensor(np.ascontiguousarray(rho), dtype=t.float32)
        from tensorlbm.multiphase import sc_single_component_force

        fx, fy = sc_single_component_force(r, G=CFG["G_lib"], psi_fn=self._psi)
        return fx.numpy(), fy.numpy()


class NumpySolver:
    """Independent from-scratch reference implementation (``ref_sc.py``)."""

    name = "ref_numpy"

    def __init__(self, forcing: str = "velocity_shift"):
        self.forcing = forcing

    def init(self, rho: np.ndarray):
        z = np.zeros_like(rho)
        return REF.equilibrium(rho, z, z)

    def step(self, f):
        return REF.stream(REF.collide_sc(f, G=CFG["G_lib"], tau=CFG["tau"], forcing=self.forcing))

    def fields(self, f):
        return REF.macroscopic(f)

    def force(self, rho: np.ndarray):
        return REF.sc_force_standard(rho, CFG["G_lib"])


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
def init_droplet(L: int, R: float) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.meshgrid(np.arange(L) + 0.5, np.arange(L) + 0.5, indexing="ij")
    rr = np.sqrt((xs - L / 2.0) ** 2 + (ys - L / 2.0) ** 2)
    rho = CFG["rho_v_init"] + 0.5 * (CFG["rho_l_init"] - CFG["rho_v_init"]) * (
        1.0 + np.tanh((R - rr) / CFG["w_int"])
    )
    return np.clip(rho, 1e-3, None), rr


def eos_pressure(rho: np.ndarray) -> np.ndarray:
    return REF.eos_pressure(rho, CFG["G_eff"])


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure(fields, rr: np.ndarray, R_guess: float, bands=(0.5, 1.5)) -> dict:
    """Young-Laplace pressure measurement (self-consistent equivalent radius)."""
    rho, ux, uy = fields
    p = eos_pressure(rho)
    umag = np.sqrt(ux * ux + uy * uy)
    r_eq = float(R_guess)
    for _ in range(6):
        inside = rr <= r_eq * 0.5
        outside = rr >= r_eq * 1.5
        rho_in = float(rho[inside].mean())
        rho_out = float(rho[outside].mean())
        mid = 0.5 * (rho_in + rho_out)
        n_liq = int((rho > mid).sum())
        r_eq_new = math.sqrt(n_liq / math.pi)
        if abs(r_eq_new - r_eq) < 1e-5:
            r_eq = r_eq_new
            break
        r_eq = r_eq_new
    out = {"R_eq": r_eq, "u_max": float(umag.max()), "u_mean": float(umag.mean())}
    a, b = bands
    inside = rr <= a * r_eq
    outside = rr >= b * r_eq
    out["p_in"] = float(p[inside].mean())
    out["p_out"] = float(p[outside].mean())
    out["dp"] = out["p_in"] - out["p_out"]
    out["rho_in"] = float(rho[inside].mean())
    out["rho_out"] = float(rho[outside].mean())
    out["mass"] = float(rho.sum())
    return out


def band_sensitivity(fields, rr: np.ndarray, R_eq: float) -> dict:
    """dp for several (inner, outer) band pairs — protocol fragility probe."""
    rho = fields[0]
    p = eos_pressure(rho)
    out = {}
    for a, b in ((0.4, 1.6), (0.5, 1.5), (0.6, 1.4), (0.7, 1.3)):
        ins = rr <= a * R_eq
        outs = rr >= b * R_eq
        out[f"{a}-{b}"] = float(p[ins].mean() - p[outs].mean())
    return out


# ---------------------------------------------------------------------------
# One droplet case
# ---------------------------------------------------------------------------
def run_case(
    solver,
    R: float,
    L: int,
    verbose: bool = True,
    min_steps: int | None = None,
    max_steps: int | None = None,
) -> dict:
    min_steps = CFG["min_steps"] if min_steps is None else min_steps
    max_steps = CFG["max_steps"] if max_steps is None else max_steps
    rho0, rr = init_droplet(L, R)
    f = solver.init(rho0)
    mass0 = float(rho0.sum())
    t0 = time.perf_counter()
    hist: list[dict] = []
    step = 0
    converged = False
    for step in range(1, max_steps + 1):
        f = solver.step(f)
        if step % CFG["sample_every"] == 0:
            m = measure(solver.fields(f), rr, R)
            m["step"] = step
            if not np.isfinite(m["dp"]) or not np.isfinite(m["R_eq"]):
                raise RuntimeError(f"non-finite state: {solver.name} R={R} L={L} step={step}")
            hist.append(m)
            if len(hist) >= 2 and step >= min_steps:
                d_dp = abs(hist[-1]["dp"] - hist[-2]["dp"]) / max(abs(hist[-1]["dp"]), 1e-12)
                d_r = abs(hist[-1]["R_eq"] - hist[-2]["R_eq"]) / max(hist[-1]["R_eq"], 1e-12)
                if d_dp < CFG["conv_dp"] and d_r < CFG["conv_r"]:
                    converged = True
                    break
    fields = solver.fields(f)
    final = measure(fields, rr, R)
    final.update(
        {
            "solver": solver.name,
            "forcing": solver.forcing,
            "R_init": R,
            "L": L,
            "step": step,
            "converged": converged,
            "mass_drift_rel": abs(final["mass"] - mass0) / mass0,
            "band_sensitivity": band_sensitivity(fields, rr, final["R_eq"]),
            "sigma_i": final["dp"] * final["R_eq"],  # 2-D cylinder: dp = sigma / R
            "wall_s": time.perf_counter() - t0,
        }
    )
    # radial profiles (for figures / band diagnostics)
    r = np.sqrt(rr)
    nb = 60
    edges = np.linspace(0.0, L / 2.0, nb + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nb - 1)
    prof_rho = np.bincount(idx, weights=fields[0].ravel(), minlength=nb) / np.maximum(
        np.bincount(idx, minlength=nb), 1
    )
    prof_p = np.bincount(idx, weights=eos_pressure(fields[0]).ravel(), minlength=nb) / np.maximum(
        np.bincount(idx, minlength=nb), 1
    )
    final["profile"] = {
        "r": (0.5 * (edges[:-1] + edges[1:])).round(6).tolist(),
        "rho": np.round(prof_rho, 6).tolist(),
        "p": np.round(prof_p, 8).tolist(),
    }
    if verbose:
        print(
            f"  [{solver.name:9s} L={L:3d} R={R:4.1f}] steps={step:5d} conv={converged} "
            f"R_eq={final['R_eq']:7.3f} dp={final['dp']:.6f} sigma_i={final['sigma_i']:.5f} "
            f"u_max={final['u_max']:.3f} rho_in={final['rho_in']:.4f} rho_out={final['rho_out']:.4f} "
            f"({final['wall_s']:.0f}s)",
            flush=True,
        )
    return final


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
def laplace_fit(rows: list[dict]) -> dict:
    """Fit dp = sigma/R (+ intercept) and report two-dimensional residuals."""
    R = np.array([r["R_eq"] for r in rows], dtype=float)
    dp = np.array([r["dp"] for r in rows], dtype=float)
    x = 1.0 / R
    sigma_o = float((dp * x).sum() / (x * x).sum())  # through origin
    # free-intercept least squares  dp = sigma x + b
    A = np.vstack([x, np.ones_like(x)]).T
    (sigma_f, b), *_ = np.linalg.lstsq(A, dp, rcond=None)
    pred = A @ np.array([sigma_f, b])
    ss_res = float(((dp - pred) ** 2).sum())
    ss_tot = float(((dp - dp.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sigma_i = dp * R
    dev_o = (sigma_i - sigma_o) / sigma_o * 100.0
    dev_f = (sigma_i - sigma_f) / sigma_f * 100.0
    return {
        "sigma_through_origin": sigma_o,
        "sigma_free_intercept": float(sigma_f),
        "intercept": float(b),
        "intercept_frac_pct": float(abs(b) / (abs(sigma_f) * x.max()) * 100.0),
        "R2": r2,
        "max_abs_dev_pct_through_origin": float(np.abs(dev_o).max()),
        "max_abs_dev_pct_free_intercept": float(np.abs(dev_f).max()),
        "per_radius": [
            {
                "R_eq": float(R[i]),
                "dp": float(dp[i]),
                "sigma_i": float(sigma_i[i]),
                "dev_pct_through_origin": float(dev_o[i]),
                "dev_pct_free_intercept": float(dev_f[i]),
            }
            for i in range(len(R))
        ],
        "n_points": int(len(R)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced study")
    ap.add_argument("--out", default=str(HERE / "artifacts"))
    ap.add_argument("--stage", default="all", help="all|main|sensitivity")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    res: dict = {"anchor": "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION", "config": CFG}
    res["config"]["bands"] = list(CFG["bands"])

    # ---- L2: analytic Maxwell coexistence on the SC EOS -------------------
    rho_v_mx, rho_l_mx, p_c = REF.maxwell_coexistence(CFG["G_eff"])
    res["maxwell"] = {"rho_v": rho_v_mx, "rho_l": rho_l_mx, "p_c": p_c}
    print(f"Maxwell (analytic EOS): rho_v={rho_v_mx:.5f} rho_l={rho_l_mx:.5f} p_c={p_c:.6f}")

    if args.quick:
        radii = [8.0, 12.0]
        dom_study = []
        forcing_study = []
        solvers = [TorchSolver()]
        min_steps, max_steps = 1500, 4000
    else:
        radii = [12.0, 16.0, 20.0, 24.0]
        dom_study = [4.0, 6.0, 8.0] if args.stage in ("all", "sensitivity") else []
        forcing_study = ["velocity_shift", "guo"] if args.stage in ("all", "sensitivity") else []
        solvers = [TorchSolver(), NumpySolver()]
        min_steps, max_steps = None, None

    def _span(R, f):
        return int(round(f * R))

    # ---- L1/L3 main scan ---------------------------------------------------
    if args.stage in ("all", "main"):
        runs = []
        for solver in solvers:
            print(f"== droplet Laplace scan, solver={solver.name} ==")
            for R in radii:
                runs.append(
                    run_case(solver, R, _span(R, 4.0), min_steps=min_steps, max_steps=max_steps)
                )
        by_solver: dict[str, list] = {}
        for r in runs:
            by_solver.setdefault(r["solver"], []).append(r)
        res["laplace"] = {}
        for name, rows in by_solver.items():
            fit = laplace_fit(rows)
            fit["rows"] = rows
            res["laplace"][name] = fit
            print(
                f"  FIT {name:9s} sigma_origin={fit['sigma_through_origin']:.5f} "
                f"sigma_free={fit['sigma_free_intercept']:.5f} R2={fit['R2']:.8f} "
                f"maxdev(origin)={fit['max_abs_dev_pct_through_origin']:.2f}% "
                f"intercept={fit['intercept']:+.2e} ({fit['intercept_frac_pct']:.2f}%)"
            )
        # cross-implementation deviation, same radii
        d = None
        if len(by_solver) == 2:
            so = {r["R_init"]: r for r in by_solver["tensorlbm"]}
            sr = {r["R_init"]: r for r in by_solver["ref_numpy"]}
            common = sorted(set(so) & set(sr))
            d = {
                "radii": common,
                "sigma_origin_rel_pct": abs(
                    res["laplace"]["tensorlbm"]["sigma_through_origin"]
                    - res["laplace"]["ref_numpy"]["sigma_through_origin"]
                )
                / res["laplace"]["ref_numpy"]["sigma_through_origin"]
                * 100.0,
                "dp_rel_pct": [
                    abs(so[R]["dp"] - sr[R]["dp"]) / abs(sr[R]["dp"]) * 100.0 for R in common
                ],
                "rho_in_rel_pct": [
                    abs(so[R]["rho_in"] - sr[R]["rho_in"]) / sr[R]["rho_in"] * 100.0 for R in common
                ],
                "rho_out_rel_pct": [
                    abs(so[R]["rho_out"] - sr[R]["rho_out"]) / sr[R]["rho_out"] * 100.0 for R in common
                ],
            }
            d["max_dp_rel_pct"] = float(np.max(d["dp_rel_pct"]))
            d["max_rho_in_rel_pct"] = float(np.max(d["rho_in_rel_pct"]))
            d["max_rho_out_rel_pct"] = float(np.max(d["rho_out_rel_pct"]))
            res["impl_vs_impl"] = d
            print(
                f"  IMPL-IMPL: max|dp| dev={d['max_dp_rel_pct']:.3f}% "
                f"sigma dev={d['sigma_origin_rel_pct']:.3f}% "
                f"rho_in={d['max_rho_in_rel_pct']:.3f}% rho_out={d['max_rho_out_rel_pct']:.3f}%"
            )

    # ---- L3 sensitivity: domain extent ------------------------------------
    if dom_study:
        print("== domain-extent sensitivity (solver=tensorlbm, velocity_shift) ==")
        sol = TorchSolver()
        rows = [run_case(sol, 16.0, _span(16.0, f)) for f in dom_study]
        res["domain_study"] = {
            "R": 16.0,
            "rows": rows,
            "dp_spread_pct": float(
                (max(r["dp"] for r in rows) - min(r["dp"] for r in rows))
                / np.mean([r["dp"] for r in rows])
                * 100.0
            ),
        }
        print(f"  dp spread over L/R in {dom_study}: {res['domain_study']['dp_spread_pct']:.2f}%")

    # ---- L3 sensitivity: forcing scheme -----------------------------------
    if forcing_study:
        print("== forcing-scheme sensitivity (R=16, L=6R) ==")
        rows = [run_case(TorchSolver(forcing=fs), 16.0, 96) for fs in forcing_study]
        res["forcing_study"] = {
            "R": 16.0,
            "L": 96,
            "rows": rows,
            "dp_spread_pct": float(
                (max(r["dp"] for r in rows) - min(r["dp"] for r in rows))
                / np.mean([r["dp"] for r in rows])
                * 100.0
            ),
        }
        print(f"  dp spread over forcings: {res['forcing_study']['dp_spread_pct']:.2f}%")

    # ---- verdict -----------------------------------------------------------
    if "laplace" in res:
        crit = CFG["criteria_pct"]
        verdict = {}
        for name, fit in res["laplace"].items():
            verdict[name] = {
                "sigma": fit["sigma_through_origin"],
                "dev_vs_locked_doc_pct": abs(
                    fit["sigma_through_origin"] - CFG["sigma_locked_doc"]
                )
                / CFG["sigma_locked_doc"]
                * 100.0,
                "laplace_linear_R2": fit["R2"],
                "max_dev_pct": fit["max_abs_dev_pct_through_origin"],
                "pass_15pct": bool(abs(fit["sigma_through_origin"] - CFG["sigma_locked_doc"])
                                   / CFG["sigma_locked_doc"] * 100.0 < crit),
                "pass_3pct": bool(abs(fit["sigma_through_origin"] - CFG["sigma_locked_doc"])
                                  / CFG["sigma_locked_doc"] * 100.0 < 3.0),
            }
        res["verdict"] = verdict
        res["pass_all_15pct"] = all(v["pass_15pct"] for v in verdict.values())

    path = out / f"result{'_quick' if args.quick else ''}.json"
    if path.exists():  # merge stages (main / sensitivity) into one artifact
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            old.update(res)
            res = old
        except json.JSONDecodeError:
            pass
    path.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nSaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())