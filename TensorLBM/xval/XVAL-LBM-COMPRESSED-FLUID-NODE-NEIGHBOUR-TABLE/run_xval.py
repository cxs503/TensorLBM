#!/usr/bin/env python3
"""XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE runner.

Cross-validates the compressed fluid-node neighbour table (openLBMPM transplant,
see ``TensorLBM/src/tensorlbm/porous/fluid_node_table.py``) against

  A2  the dense full-lattice path of the same repo (representation control), and
  A3  the analytic plane-Poiseuille permeability k = H^2/12 (physical anchor),

collects every KPI into ``results.json`` / ``kpi.csv`` and writes nothing that
is not derived from the raw per-arm records printed by this script.

Design (thresholds, stopping rule, confounders) is predeclared in
``predeclared.json`` next to this file and was written before the runs.

Usage
-----
    python run_xval.py                # full predeclared ladder
    python run_xval.py --quick        # smoke subset (not for the report KPI table)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tensorlbm.porous import (  # noqa: E402
    build_fluid_node_table,
    intrinsic_permeability,
    compressed_step,
    dense_step,
    macroscopic_with_forcing,
    permeability,
)
from tensorlbm.porous.fluid_node_table import FluidNodeTable  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

TAU = 0.8
NU = (TAU - 0.5) / 3.0  # 0.1 lattice units
CS = math.sqrt(1.0 / 3.0)

STEADY_TOL = 1e-9          # predeclared: max|u(n)-u(n-200)| / max|u(n)|
STEADY_WINDOW = 200
STEPS_CAP = 12000
MA_LIMIT = 0.1
RHO_BAND = (0.95, 1.05)


# ------------------------------------------------------------------ geometries
def channel_mask(ny: int, nx: int, fluid_rows: int) -> torch.Tensor:
    """Plane channel: solid slabs top and bottom, periodic along x."""
    solid = torch.ones((ny, nx), dtype=torch.bool)
    y0 = (ny - fluid_rows) // 2
    solid[y0 : y0 + fluid_rows, :] = False
    return solid


def square_grain_mask(n: int, porosity: float) -> torch.Tensor:
    """Periodic 2x2 array of square grains with the requested porosity."""
    solid = torch.zeros((n, n), dtype=torch.bool)
    side = int(round(n * math.sqrt((1.0 - porosity) / 4.0)))
    if side <= 0:
        raise ValueError("porosity too close to 1 for this lattice")
    off = [n // 4, 3 * n // 4]
    for y0 in off:
        for x0 in off:
            solid[y0 - side // 2 : y0 - side // 2 + side, x0 - side // 2 : x0 - side // 2 + side] = True
    return solid


# --------------------------------------------------------------------- drivers
def run_compressed(solid, g, tau, steps_cap=STEPS_CAP, check_every=100, record=False):
    table = build_fluid_node_table(solid)
    k_nodes = table.n_fluid
    f = torch.zeros((9, k_nodes))
    rho0 = torch.full((k_nodes,), 1.0)
    f[0] = 4.0 / 9.0
    for d in (1, 2, 3, 4):
        f[d] = (1.0 / 9.0)
    for d in (5, 6, 7, 8):
        f[d] = (1.0 / 36.0)
    u_hist = []
    steps = 0
    t0 = time.perf_counter()
    converged = False
    while steps < steps_cap:
        f = compressed_step(f, table, tau, g, 0.0)
        steps += 1
        if steps % check_every == 0:
            rho, ux, uy = macroscopic_with_forcing(f, g, 0.0)
            u_hist.append(ux.clone())
            if not torch.isfinite(ux).all():
                raise RuntimeError("non-finite velocity in compressed arm")
            if len(u_hist) > STEADY_WINDOW // check_every:
                prev = u_hist[-1 - STEADY_WINDOW // check_every]
                cur = u_hist[-1]
                den = float(cur.abs().max())
                if den > 0 and float((cur - prev).abs().max()) / den < STEADY_TOL:
                    converged = True
                    break
    elapsed = time.perf_counter() - t0
    rho, ux, uy = macroscopic_with_forcing(f, g, 0.0)
    out = {
        "arm": "A1_compressed",
        "steps": steps,
        "converged": converged,
        "elapsed_s": elapsed,
        "mlups": table.n_total * steps / elapsed / 1e6,
        "porosity": table.porosity,
        "n_fluid": table.n_fluid,
        "n_total": table.n_total,
        "ux_fluid": ux.numpy(),
        "node_index": table.nodes.numpy(),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "rho_drift": float((rho.sum() - rho0.sum()).abs() / rho0.sum()),
        "mem": table.state_bytes(),
        "u_mean_superficial": float(ux.sum() / table.n_total),
        "table": table if record else None,
        "ny": int(solid.shape[0]),
        "nx": int(solid.shape[1]),
    }
    out["k_darcy"] = permeability(out["u_mean_superficial"], NU, g)
    out["k_intrinsic"] = intrinsic_permeability(out["k_darcy"], table.porosity)
    out["ma"] = float(ux.abs().max()) / CS
    return out


def run_dense(solid, g, tau, steps_cap=STEPS_CAP, check_every=100):
    ny, nx = int(solid.shape[0]), int(solid.shape[1])
    fluid = ~solid
    f = torch.zeros((9, ny, nx))
    f[0] = 4.0 / 9.0
    f[1] = f[2] = f[3] = f[4] = 1.0 / 9.0
    f[5] = f[6] = f[7] = f[8] = 1.0 / 36.0
    u_hist = []
    steps = 0
    t0 = time.perf_counter()
    converged = False
    while steps < steps_cap:
        f = dense_step(f, solid, tau, g, 0.0)
        steps += 1
        if steps % check_every == 0:
            rho, ux, uy = macroscopic_with_forcing(f, g, 0.0)
            u_hist.append(ux.clone())
            if not torch.isfinite(ux).all():
                raise RuntimeError("non-finite velocity in dense arm")
            if len(u_hist) > STEADY_WINDOW // check_every:
                prev = u_hist[-1 - STEADY_WINDOW // check_every]
                cur = u_hist[-1]
                den = float(cur.abs().max())
                if den > 0 and float((cur - prev).abs().max()) / den < STEADY_TOL:
                    converged = True
                    break
    elapsed = time.perf_counter() - t0
    rho, ux, uy = macroscopic_with_forcing(f, g, 0.0)
    out = {
        "arm": "A2_dense",
        "steps": steps,
        "converged": converged,
        "elapsed_s": elapsed,
        "mlups": ny * nx * steps / elapsed / 1e6,
        "porosity": float(fluid.sum()) / (ny * nx),
        "n_fluid": int(fluid.sum()),
        "n_total": ny * nx,
        "ux_fluid": ux[fluid].numpy(),
        "node_index": (fluid.nonzero(as_tuple=True)[0] * nx + fluid.nonzero(as_tuple=True)[1]).numpy(),
        "rho_min": float(rho[fluid].min()),
        "rho_max": float(rho[fluid].max()),
        "rho_drift": float((rho[fluid].sum() - fluid.sum().double()).abs() / fluid.sum()),
        "mem": None,
        "u_mean_superficial": float(ux[fluid].sum() / (ny * nx)),
        "ny": ny,
        "nx": nx,
    }
    out["k_darcy"] = permeability(out["u_mean_superficial"], NU, g)
    out["ma"] = float(ux[fluid].abs().max()) / CS
    return out


def compare(a1, a2):
    """Field-level representation deviation, matched through the compaction map."""
    if not np.array_equal(a1["node_index"], a2["node_index"]):
        raise RuntimeError("compaction order differs between arms")
    u1, u2 = a1["ux_fluid"], a2["ux_fluid"]
    den = np.abs(u2).max()
    return {
        "ux_maxabs_rel": float(np.abs(u1 - u2).max() / den),
        "ux_l2_rel": float(np.linalg.norm(u1 - u2) / np.linalg.norm(u2)),
        "k_rel_diff": float(abs(a1["k_darcy"] - a2["k_darcy"]) / abs(a2["k_darcy"])),
    }


def strip(rec):
    out = {k: v for k, v in rec.items() if k not in ("ux_fluid", "node_index", "table")}
    return out


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "artifacts",
                                                  "xval_compressed_fluid_node_neighbour_table"))
    args = ap.parse_args()
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    cases = []
    # C1: plane channel, analytic anchor k = H^2/12 (H = 32 fluid rows)
    cases.append(dict(name="C1_channel_H32", kind="channel", ny=64, nx=32, H=32, g=1e-5))
    # C2..C4: periodic square-grain porous medium, porosity sweep at N=128
    for p in (0.90, 0.75, 0.60):
        cases.append(dict(name=f"C2_porous_p{int(p*100)}_N128", kind="porous", n=128, porosity=p, g=1e-5))
    # C5: forcing linearity at fixed geometry
    cases.append(dict(name="C5_porous_p75_N128_g2e5", kind="porous", n=128, porosity=0.75, g=2e-5))
    # C6: resolution sensitivity
    cases.append(dict(name="C6_porous_p75_N256", kind="porous", n=256, porosity=0.75, g=1e-5))
    if args.quick:
        cases = [c for c in cases if c["name"] in ("C1_channel_H32", "C2_porous_p75_N128")]

    records = []
    for case in cases:
        if case["kind"] == "channel":
            solid = channel_mask(case["ny"], case["nx"], case["H"])
            k_ref = (case["H"] / case["ny"]) * case["H"] ** 2 / 12.0
            ref_kind = "analytic_slab: k_super = (H/ny)*H^2/12"
        else:
            solid = square_grain_mask(case["n"], case["porosity"])
            k_ref, ref_kind = None, "none"
        print(f"\n=== {case['name']}  solid_fraction={float(solid.double().mean()):.4f}", flush=True)
        a1 = run_compressed(solid, case["g"], TAU, record=True)
        a2 = run_dense(solid, case["g"], TAU)
        cmp = compare(a1, a2)
        rec = {
            "case": case["name"],
            "case_cfg": case,
            "k_reference": k_ref,
            "reference_kind": ref_kind,
            "A1": strip(a1),
            "A2": strip(a2),
            "compare": cmp,
            "k_err_vs_ref_pct": (None if k_ref is None else 100.0 * abs(a1["k_darcy"] - k_ref) / k_ref),
            "k_err_vs_ref_pct_A2": (None if k_ref is None else 100.0 * abs(a2["k_darcy"] - k_ref) / k_ref),
            "mem": a1["mem"],
        }
        # analytic Poiseuille profile check for the channel case
        if case["kind"] == "channel":
            ux = a1["ux_fluid"]
            ny, nx = a1["ny"], a1["nx"]
            idx = a1["node_index"]
            ys = idx // nx
            prof = np.zeros(ny)
            cnt = np.zeros(ny)
            for yy, uu in zip(ys, ux):
                prof[yy] += uu
                cnt[yy] += 1
            valid = cnt > 0
            prof[valid] /= cnt[valid]
            y0 = int(valid.nonzero()[0][0])
            rows = np.arange(y0, y0 + case["H"])
            yc = (rows - y0 + 0.5)  # distance from the lower wall
            H = float(case["H"])
            u_an = case["g"] / (2 * NU) * yc * (H - yc)
            rel = float(np.abs(prof[rows] - u_an).max() / u_an.max())
            rec["poiseuille_profile_maxrel"] = rel
            rec["poiseuille_u_max_analytic"] = float(u_an.max())
            rec["poiseuille_u_max_got"] = float(prof[rows].max())
        records.append(rec)
        print(json.dumps({k: rec[k] for k in ("case", "compare", "k_err_vs_ref_pct")}, default=str), flush=True)
        print(f"    A1 k_super={a1['k_darcy']:.6g} k_int={a1['k_intrinsic']:.6g} steps={a1['steps']} conv={a1['converged']} "
              f"mlups={a1['mlups']:.1f} | A2 k={a2['k_darcy']:.6g} steps={a2['steps']} "
              f"conv={a2['converged']} mlups={a2['mlups']:.1f}", flush=True)

    # ------------------------------------------------------- performance probe
    perf = {}
    for n, p in ((128, 0.75), (256, 0.75)):
        solid = square_grain_mask(n, p)
        table = build_fluid_node_table(solid)
        f = torch.full((9, table.n_fluid), 1.0)
        for _ in range(20):
            f = compressed_step(f, table, TAU, 1e-5, 0.0)
        t0 = time.perf_counter()
        for _ in range(100):
            f = compressed_step(f, table, TAU, 1e-5, 0.0)
        t_c = (time.perf_counter() - t0) / 100
        fd = torch.full((9, n, n), 1.0)
        for _ in range(20):
            fd = dense_step(fd, solid, TAU, 1e-5, 0.0)
        t0 = time.perf_counter()
        for _ in range(100):
            fd = dense_step(fd, solid, TAU, 1e-5, 0.0)
        t_d = (time.perf_counter() - t0) / 100
        perf[f"N{n}_p{int(p*100)}"] = {
            "s_per_step_compressed": t_c,
            "s_per_step_dense": t_d,
            "mlups_compressed": n * n / t_c / 1e6,
            "mlups_dense": n * n / t_d / 1e6,
            "speedup_dense_over_compressed": t_d / t_c,
            **table.state_bytes(),
        }
        print(f"perf {n}: compressed {t_c*1e3:.3f} ms/step, dense {t_d*1e3:.3f} ms/step, "
              f"ratio {t_d/t_c:.2f}x", flush=True)

    summary = {
        "tau": TAU,
        "nu": NU,
        "steps_cap": STEPS_CAP,
        "steady_tol": STEADY_TOL,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "records": records,
        "performance": perf,
    }
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # KPI table (one row per case x arm)
    rows = ["case,arm,porosity,k_darcy_super,k_intrinsic,k_ref_super,k_err_pct,ux_maxabs_rel_vs_A2,"
            "steps,converged,mlups,ma,rho_drift"]
    for rec in records:
        for arm in ("A1", "A2"):
            a = rec[arm]
            k_ref = "" if rec["k_reference"] is None else f"{rec['k_reference']:.8g}"
            k_err = "" if rec["k_err_vs_ref_pct"] is None else f"{rec['k_err_vs_ref_pct']:.4f}"
            rows.append(
                f"{rec['case']},{arm},{a['porosity']:.4f},{a['k_darcy']:.8g},{a['k_intrinsic']:.8g},"
                f"{k_ref},{k_err},"
                f"{rec['compare']['ux_maxabs_rel']:.3e},{a['steps']},{a['converged']},"
                f"{a['mlups']:.2f},{a['ma']:.4f},{a['rho_drift']:.2e}"
            )
    with open(os.path.join(outdir, "kpi.csv"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"\nwrote {outdir}/results.json and kpi.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())