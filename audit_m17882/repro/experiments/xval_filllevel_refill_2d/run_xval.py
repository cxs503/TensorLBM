#!/usr/bin/env python3
"""XVAL-LBM-FREE-SURFACE-FILLLEVEL-RECONSTRUCT-REFILL — 2D fill-level reconstruct/refill XVAL.

Cross-validation of TensorLBM's D2Q9 Körner fill-level free-surface
*reconstruction / refilling* practice against the OSS reference
``lssfau/walberla`` ``free_surface`` module (``PdfReconstructionModel.h``,
``PdfRefillingModel.h``, ``ExcessMassDistributionModel.h``,
``functionality/ReconstructInterfaceCellABB.h``,
``dynamics/PdfRefillingSweep.impl.h``).

Design (pre-declared, see DESIGN below — written before any result was read):

  decision   : which refill density policy should be the TensorLBM default for
               newly converted GAS->INTERFACE cells?
  alternatives: repo_average (historical), walberla_equilibrium (OSS
               EquilibriumRefilling: exclude same-step G->I neighbours,
               fallback rho=1,u=0), fixed_rho (= rho_liquid everywhere).
  metric      : front position Z=x/a at the 14 Martin & Moyce (1952) points on
               the T = t*sqrt(2 g/a) axis; relative RMSE and max |rel err|.
  threshold   : rel-RMSE <= 15 % (owner precision standard) and max |rel err|
               <= 15 % — both pre-declared.
  control     : static hydrostatic column (zero forcing) -> spurious |u|max;
               conservation -> relative liquid-inventory drift.

Usage
  python experiments/xval_filllevel_refill_2d/run_xval.py --matrix
  python experiments/xval_filllevel_refill_2d/run_xval.py --variant walberla_equilibrium --a 32
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import os
import torch
torch.set_num_threads(int(os.environ.get("XVAL_THREADS", "1")))

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.free_surface_lbm_2d import (  # noqa: E402
    GAS,
    INTERFACE,
    LIQUID,
    free_surface_step_2d,
    init_fill_rectangular_2d,
    init_flags_from_fill_2d,
    total_liquid_inventory_2d,
)

ART = HERE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# --- pre-declared design (frozen before inspecting results) ----------------
DESIGN = {
    "decision": "default refill policy for G->I converted fill-level cells",
    "alternatives": ["repo_average", "walberla_equilibrium", "fixed_rho"],
    "reference": "Martin & Moyce (1952) digitised collapse front, H/a=2, "
    "T = t*sqrt(2 g/a), X = x/a, valid T<=2.96 (Lethe dam-break-2d table)",
    "metrics": ["rel_rmse_pct", "max_rel_err_pct", "mass_drift_pct", "hydro_u_max"],
    "thresholds": {"rel_rmse_pct_max": 15.0, "max_rel_err_pct_max": 15.0},
    "stopping_rule": "one run per (policy, a); no threshold tuning after looking",
    "unit_of_analysis": "one dam-break run (policy x resolution)",
    "confounders": [
        "resolution a (lattice cells over column width)",
        "lattice viscosity tau (nu=(tau-1/2)/3)",
        "gravity/Mach number g=1e-4 fixed for all runs (u~0.10 lattice)",
    ],
}

REF_T = [0.00, 0.41, 0.84, 1.19, 1.43, 1.63, 1.82, 1.97, 2.20, 2.32, 2.50, 2.64, 2.82, 2.96]
REF_Z = [1.00, 1.11, 1.23, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3.00, 3.22, 3.44, 3.67]

RHO_L, RHO_G = 1.0, 1.0
G_ABS = 1.0e-4
T_MAX = 2.96


def _front_x(fill: torch.Tensor) -> int:
    """Rightmost column whose liquid column height is at least half a cell."""
    h = fill.sum(dim=0)  # liquid height per column (fill-weighted)
    idx = (h >= 0.5).nonzero(as_tuple=False)
    return 0 if idx.numel() == 0 else int(idx[-1].item())


def _interp(t, t_vals, v_vals):
    if t <= t_vals[0]:
        return v_vals[0]
    if t >= t_vals[-1]:
        return v_vals[-1]
    for i in range(len(t_vals) - 1):
        if t_vals[i] <= t <= t_vals[i + 1]:
            frac = (t - t_vals[i]) / (t_vals[i + 1] - t_vals[i])
            return v_vals[i] + frac * (v_vals[i + 1] - v_vals[i])
    return v_vals[-1]


def run_dambreak(a: int = 32, tau: float = 0.6, policy: str = "repo_average", steps=None,
                 tall: float = 2.0) -> dict:
    """Lock-release dam break, M&M aspect ratio H/a (default 2), walls on all sides."""
    ny = int(tall * a) + 2
    nx = 8 * a + 2
    fill, solid = init_fill_rectangular_2d(ny, nx, float(a), float(tall * a), "cpu")
    flags = init_flags_from_fill_2d(fill, solid)
    rho0 = torch.where(flags == LIQUID, torch.full_like(fill, RHO_L), torch.full_like(fill, RHO_G))
    f = equilibrium(rho0, torch.zeros_like(fill), torch.zeros_like(fill))
    mass = fill * f.sum(dim=0).clamp(min=1e-6)

    sqrt_2ga = math.sqrt(2.0 * G_ABS / a)
    n_steps = steps if steps is not None else int(math.ceil(T_MAX / sqrt_2ga))
    inv0 = float(total_liquid_inventory_2d(f, mass, flags, RHO_L))
    f0 = _front_x(fill) / a
    t_vals, z_vals = [0.0], [f0]
    t0 = time.time()
    nan_at = None
    for k in range(1, n_steps + 1):
        f, mass, flags = free_surface_step_2d(
            f, fill, flags, solid, tau=tau, gy=-G_ABS, rho_liquid=RHO_L, rho_gas=RHO_G,
            mass=mass, return_mass=True, refill_policy=policy,
        )
        fill = torch.where(flags == LIQUID, torch.ones_like(fill), torch.zeros_like(fill))
        fill = torch.where(flags == INTERFACE, (mass / f.sum(0).clamp(min=1e-6)), fill)
        if not torch.isfinite(f).all():
            nan_at = k
            break
        t = k * sqrt_2ga
        t_vals.append(t)
        z_vals.append(_front_x(fill) / a)
    dt = time.time() - t0
    inv1 = float(total_liquid_inventory_2d(f, mass, flags, RHO_L))
    mass_drift_pct = 100.0 * (inv1 - inv0) / inv0

    cmp_t, cmp_sim, cmp_ref, cmp_err = [], [], [], []
    for T, Z in zip(REF_T, REF_Z):
        if T > t_vals[-1] + 1e-12:
            continue
        zs = _interp(T, t_vals, z_vals)
        cmp_t.append(T)
        cmp_sim.append(zs)
        cmp_ref.append(Z)
        cmp_err.append(100.0 * (zs - Z) / Z)
    if cmp_err:
        rel_rmse = math.sqrt(sum(e * e for e in cmp_err) / len(cmp_err))
        max_abs = max(abs(e) for e in cmp_err)
    else:
        rel_rmse = max_abs = float("nan")
    return {
        "case": "dambreak_martin_moyce",
        "policy": policy,
        "a": a,
        "tau": tau,
        "nu": (tau - 0.5) / 3.0,
        "g": -G_ABS,
        "nx": nx,
        "ny": ny,
        "aspect": tall,
        "steps": n_steps,
        "sqrt_2g_a": sqrt_2ga,
        "T_max": t_vals[-1],
        "Z0": f0,
        "Z_end": z_vals[-1],
        "wall_seconds": round(dt, 2),
        "mass_drift_pct": mass_drift_pct,
        "nan_at_step": nan_at,
        "n_cmp": len(cmp_err),
        "rel_rmse_pct": rel_rmse,
        "max_rel_err_pct": max_abs,
        "compare": {"T": cmp_t, "Z_sim": cmp_sim, "Z_ref": cmp_ref, "rel_err_pct": cmp_err},
        "front_series": {"T": t_vals[::25], "Z": z_vals[::25]},
        "pass_rmse": bool(cmp_err and rel_rmse <= DESIGN["thresholds"]["rel_rmse_pct_max"]),
        "pass_max": bool(cmp_err and max_abs <= DESIGN["thresholds"]["max_rel_err_pct_max"]),
    }


def run_hydrostatic(policy: str = "repo_average", tau: float = 0.6, steps: int = 300,
                    ny: int = 48, nx: int = 48) -> dict:
    """Static column at rest: measures spurious interface velocity (control)."""
    fill, solid = init_fill_rectangular_2d(ny, nx, float(nx - 2), float((ny - 2) / 2.0), "cpu")
    flags = init_flags_from_fill_2d(fill, solid)
    rho0 = torch.where(flags == LIQUID, torch.full_like(fill, RHO_L), torch.full_like(fill, RHO_G))
    f = equilibrium(rho0, torch.zeros_like(fill), torch.zeros_like(fill))
    mass = fill * f.sum(dim=0).clamp(min=1e-6)
    u_max = 0.0
    for _ in range(steps):
        f, mass, flags = free_surface_step_2d(
            f, fill, flags, solid, tau=tau, gy=-G_ABS, rho_liquid=RHO_L, rho_gas=RHO_G,
            mass=mass, return_mass=True, refill_policy=policy,
        )
        fill = torch.where(flags == LIQUID, torch.ones_like(fill), torch.zeros_like(fill))
        fill = torch.where(flags == INTERFACE, mass / f.sum(0).clamp(min=1e-6), fill)
        rho = f.sum(0).clamp(min=1e-6)
        ux = f[[1, 5, 8]].sum(0) / rho
        uy = f[[2, 6, 7]].sum(0) / rho
        fluid = (flags == LIQUID) | (flags == INTERFACE)
        u = (ux**2 + uy**2).sqrt().masked_fill(~fluid, 0.0)
        u_max = max(u_max, float(u.max()))
    return {"case": "hydrostatic_control", "policy": policy, "tau": tau,
            "u_max_lattice": u_max, "steps": steps}


def plot(results: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(REF_T, REF_Z, "k--o", ms=3.5, label="Martin & Moyce 1952 (digitised)")
    styles = {"repo_average": "C0-", "walberla_equilibrium": "C3-", "fixed_rho": "C2-"}
    for r in results:
        if r["case"] != "dambreak_martin_moyce" or r["a"] != 32 or r["tau"] != 0.6:
            continue
        ax.plot(r["compare"]["T"], r["compare"]["Z_sim"], styles.get(r["policy"], "k-"),
                lw=1.6, label=f"{r['policy']} (a=32)")
    ax.set_xlabel("T = t sqrt(2g/a)")
    ax.set_ylabel("Z = x_front / a")
    ax.set_title("2D fill-level dam break vs M&M(1952)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for r in results:
        if r["case"] != "dambreak_martin_moyce" or r["a"] != 32 or r["tau"] != 0.6:
            continue
        ax.plot(r["compare"]["T"], r["compare"]["rel_err_pct"],
                styles.get(r["policy"], "k-"), marker="s", ms=3,
                label=f"{r['policy']}: RMSE {r['rel_rmse_pct']:.1f}%")
    ax.axhline(15.0, color="r", ls=":", lw=1.0, label="pre-declared 15% gate")
    ax.axhline(-15.0, color="r", ls=":", lw=1.0)
    ax.set_xlabel("T = t sqrt(2g/a)")
    ax.set_ylabel("rel. error of Z [%]")
    ax.set_title("Front-position error (a=32, tau=0.6)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("figure ->", out_png)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--variant", default="repo_average")
    ap.add_argument("--a", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--tall", type=float, default=2.0)
    ap.add_argument("--no-hydro", action="store_true")
    ap.add_argument("--tag", default="xval_results.json")
    ap.add_argument("--rho-g", type=float, default=RHO_G)
    args = ap.parse_args()

    results: list[dict] = []
    if args.matrix:
        for a in (16, 32):
            for policy in DESIGN["alternatives"]:
                print(f"[dambreak] a={a} tau=0.6 policy={policy}", flush=True)
                r = run_dambreak(a=a, tau=0.6, policy=policy)
                print(f"   rel_rmse={r['rel_rmse_pct']:.2f}%  max={r['max_rel_err_pct']:.2f}%"
                      f"  drift={r['mass_drift_pct']:.3e}%  T_max={r['T_max']:.2f}"
                      f"  {r['wall_seconds']}s", flush=True)
                results.append(r)
        if not args.no_hydro:
            for policy in DESIGN["alternatives"]:
                r = run_hydrostatic(policy=policy)
                print(f"[hydro] {policy}: u_max={r['u_max_lattice']:.3e}", flush=True)
                results.append(r)
    else:
        r = run_dambreak(a=args.a, tau=args.tau, policy=args.variant, tall=args.tall)
        print(json.dumps({k: v for k, v in r.items() if k != "front_series"}, indent=1)[:1200])
        results.append(r)

    payload = {"design": DESIGN, "results": results,
               "env": {"torch": torch.__version__, "device": "cpu"}}
    (ART / args.tag).write_text(json.dumps(payload, indent=1))
    csv = ["policy,a,tau,tau_nu,rel_rmse_pct,max_rel_err_pct,mass_drift_pct,n_cmp,T_max,pass_rmse,pass_max"]
    for r in results:
        if r["case"] != "dambreak_martin_moyce":
            continue
        csv.append(f"{r['policy']},{r['a']},{r['tau']},{r['nu']:.5f},{r['rel_rmse_pct']:.3f},"
                   f"{r['max_rel_err_pct']:.3f},{r['mass_drift_pct']:.3e},{r['n_cmp']},"
                   f"{r['T_max']:.3f},{int(r['pass_rmse'])},{int(r['pass_max'])}")
    (ART / "xval_summary.csv").write_text("\n".join(csv) + "\n")
    try:
        plot(results, ART / "xval_front_vs_mm.png")
    except Exception as exc:  # pragma: no cover
        print("plot failed:", exc)
    print("\n".join(csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())