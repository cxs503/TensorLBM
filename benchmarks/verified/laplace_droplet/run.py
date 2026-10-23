#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B23 static-droplet Laplace benchmark — SCMP (Shan-Chen single-component).

Model: SC94 (Shan-Chen 1993) pseudopotential psi(rho) = 1 - exp(-rho),
       physical coupling G_eff = -5.0, tau = 1.0.

Library path (common module, zero hand-written collide/stream):
  tensorlbm.multiphase.collide_sc_single_component (D2Q9)
  tensorlbm.multiphase3d.collide_sc_single_component_3d (D3Q19)
  + solver.stream / solver3d.stream3d (periodic)

IMPORTANT sign-flip workaround (verified 2026-08-18, see README + gap file):
  the library's `_sc_neighbor_weighted_sum` uses the BACKWARD gather
  psi(x - c) instead of the standard forward psi(x + c), which flips the
  SC force sign:  F_lib(G) = -F_standard(G)  exactly.  Hence the library
  call with G = +5.0 realises the standard SC94 attractive model with
  G_eff = -5.0 (coexistence exists, droplet stable).  Calling with the
  nominal G = -5.0 is REPULSIVE (no stable droplet) - documented gap.
  EOS pressure for measurement uses the standard-convention coupling:
  p(rho) = rho/3 + G_eff * psi(rho)^2 / 6,  G_eff = -5.0.

Discrete coexistence (measured from this scheme, L=100 R=25 droplet run):
  rho_l = 1.957, rho_v = 0.1596  (ratio ~12.3; continuum-Maxwell values
  1.7505/0.0493 differ due to lattice discreteness - init only).

Physics: liquid droplet in vapour.  Young-Laplace: 2D DeltaP = sigma/R,
         3D DeltaP = 2*sigma/R.  sigma_eff from linear fit of DeltaP vs
         1/R_eq.  Pass criterion (Laplace law): per-radius sigma within
         3% of the fitted sigma_eff, fit R^2 >= 0.999, |intercept| <= 3%
         of slope/Rmax.  Real simulation only - no extrapolation.

Usage: python run.py [--dims 2,3] [--device2d cpu] [--device3d cuda:0]
                     [--radii 15,25,40] [--max-steps N] [--min-steps N]
                     [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

import numpy as np
import torch
from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

torch.set_num_threads(32)

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, psi_exp  # noqa: E402
from tensorlbm.multiphase3d import collide_sc_single_component_3d  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402
from tensorlbm.solver3d import stream3d  # noqa: E402

CS2 = 1.0 / 3.0
G_LIB = 5.0  # library argument (sign-flipped convention)
G_EFF = -5.0  # physical standard-convention SC94 coupling used in the EOS
TAU = 1.0
RHO_L, RHO_V = 1.957, 0.1596  # discrete coexistence (measured)
W_INT = 4.0


def eos_pressure(rho: torch.Tensor, g: float = G_EFF) -> torch.Tensor:
    """SC EOS pressure (standard convention, consistent with applied force)."""
    psi = 1.0 - torch.exp(-rho)
    return rho / 3.0 + g * psi * psi / 6.0


def init_rho(dim: int, L: int, R: float, device: torch.device) -> torch.Tensor:
    """tanh droplet (liquid inside, vapour outside) on a periodic domain."""
    if dim == 2:
        ys = torch.arange(L, dtype=torch.float32, device=device)
        xs = torch.arange(L, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    else:
        zs = torch.arange(L, dtype=torch.float32, device=device)
        ys = torch.arange(L, dtype=torch.float32, device=device)
        xs = torch.arange(L, dtype=torch.float32, device=device)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
        r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2 + (zz - L / 2.0) ** 2)
    rho = RHO_V + 0.5 * (RHO_L - RHO_V) * (1.0 + torch.tanh((R - r) / W_INT))
    return rho.clamp(min=1e-3)


def _radius_field(dim: int, L: int, device: torch.device) -> torch.Tensor:
    if dim == 2:
        ys = torch.arange(L, dtype=torch.float32, device=device)
        xs = torch.arange(L, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    zs = torch.arange(L, dtype=torch.float32, device=device)
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2 + (zz - L / 2.0) ** 2)


def measure(
    dim: int, f: torch.Tensor, R_guess: float, rr: torch.Tensor, device: torch.device
) -> dict:
    """Self-consistent measurement: bands -> bulk densities -> mid threshold -> R_eq."""
    if dim == 2:
        rho, ux, uy = macroscopic(f)
        umag = torch.sqrt(ux**2 + uy**2)
    else:
        rho, ux, uy, uz = macroscopic3d(f)
        umag = torch.sqrt(ux**2 + uy**2 + uz**2)
    p = eos_pressure(rho)
    r_eq = R_guess
    for _ in range(3):
        inside = rr <= r_eq * 0.5
        outside = rr >= r_eq * 1.5
        rho_in = float(rho[inside].mean().item()) if inside.any() else float("nan")
        rho_out = float(rho[outside].mean().item()) if outside.any() else float("nan")
        mid = 0.5 * (rho_in + rho_out)
        n_liq = int((rho > mid).sum().item())
        if dim == 2:
            r_eq_new = math.sqrt(n_liq / math.pi)
        else:
            r_eq_new = (3.0 * n_liq / (4.0 * math.pi)) ** (1.0 / 3.0)
        if abs(r_eq_new - r_eq) < 1e-4:
            r_eq = r_eq_new
            break
        r_eq = r_eq_new
    inside = rr <= r_eq * 0.5
    outside = rr >= r_eq * 1.5
    p_in = float(p[inside].mean().item()) if inside.any() else float("nan")
    p_out = float(p[outside].mean().item()) if outside.any() else float("nan")
    rho_in = float(rho[inside].mean().item()) if inside.any() else float("nan")
    rho_out = float(rho[outside].mean().item()) if outside.any() else float("nan")
    return {
        "p_in": p_in,
        "p_out": p_out,
        "dp": p_in - p_out,
        "rho_in": rho_in,
        "rho_out": rho_out,
        "R_eq": r_eq,
        "max_u": float(umag.max().item()),
        "mass": float(rho.sum().item()),
    }


def run_droplet(
    dim: int,
    R: float,
    L: int,
    device: torch.device,
    max_steps: int,
    min_steps: int,
    sample_interval: int,
    conv_dp: float = 2e-4,
    conv_r: float = 2e-3,
    compile_mode: str | None = "default",
) -> tuple[dict, list[dict]]:
    t0 = time.perf_counter()
    rho0 = init_rho(dim, L, R, device)
    mass0 = float(rho0.sum().item())
    zero = torch.zeros_like(rho0)
    if dim == 2:
        f = equilibrium(rho0, zero, zero)
    else:
        f = equilibrium3d(rho0, zero, zero, zero)
    rr = _radius_field(dim, L, device)

    # ---- 整步步进函数（共性 compile 路径；dim 分支静态，NaN 守卫/测量留在编译域外）----
    def _step(f):
        if dim == 2:
            return stream(collide_sc_single_component(f, G=G_LIB, tau=TAU, psi_fn=psi_exp))
        return stream3d(collide_sc_single_component_3d(f, G=G_LIB, tau=TAU, psi_fn=psi_exp))

    step_fn = route_step(_step, compile_mode, name=f"laplace_droplet[{dim}d]")

    hist: list[dict] = []
    step = 0
    converged = False
    for step in range(1, max_steps + 1):
        f = step_fn(f)
        if step % sample_interval == 0:
            rho_cur = f.sum(dim=0)
            if float(rho_cur.min().item()) < 0.0 or not torch.isfinite(rho_cur).all():
                raise RuntimeError(f"dim={dim} R={R}: NaN/negative rho at step {step}")
            m = measure(dim, f, R, rr, device)
            m["step"] = step
            hist.append(m)
            if len(hist) >= 2:
                d_dp = abs(hist[-1]["dp"] - hist[-2]["dp"]) / max(abs(hist[-1]["dp"]), 1e-12)
                d_r = abs(hist[-1]["R_eq"] - hist[-2]["R_eq"]) / max(hist[-1]["R_eq"], 1e-12)
                if d_dp < conv_dp and d_r < conv_r and step >= min_steps:
                    converged = True
                    break
    tail = hist[-3:]
    final = {
        "step": int(step),
        "compile_mode": compile_mode,
        "converged": converged,
        "p_in": float(np.mean([s["p_in"] for s in tail])),
        "p_out": float(np.mean([s["p_out"] for s in tail])),
        "dp": float(np.mean([s["dp"] for s in tail])),
        "dp_std": float(np.std([s["dp"] for s in tail])),
        "rho_in": float(np.mean([s["rho_in"] for s in tail])),
        "rho_out": float(np.mean([s["rho_out"] for s in tail])),
        "R_eq": float(np.mean([s["R_eq"] for s in tail])),
        "R_eq_std": float(np.std([s["R_eq"] for s in tail])),
        "max_u": float(np.mean([s["max_u"] for s in tail])),
        "mass_drift": abs(float(tail[-1]["mass"]) - mass0) / mass0,
        "dt_s": time.perf_counter() - t0,
    }
    return final, hist


def fit_laplace(rows: list[dict], dim: int) -> dict:
    """Fit sigma from DeltaP vs 1/R_eq. 2D: dp=sigma/R; 3D: dp=2 sigma/R."""
    inv_r = np.array([1.0 / r["R_eq"] for r in rows])
    y = np.array([r["dp"] for r in rows])
    if dim == 3:
        y = y / 2.0
    sigma_origin = float(np.sum(inv_r * y) / np.sum(inv_r * inv_r))
    A = np.vstack([inv_r, np.ones_like(inv_r)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = a * inv_r + b
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    per_r = []
    for r in rows:
        sigma_i = r["dp"] * r["R_eq"] if dim == 2 else r["dp"] * r["R_eq"] / 2.0
        dev = abs(sigma_i - sigma_origin) / sigma_origin
        per_r.append(
            {
                "R_init": r["R_init"],
                "R_eq": r["R_eq"],
                "p_in": r["p_in"],
                "p_out": r["p_out"],
                "dp": r["dp"],
                "rho_in": r["rho_in"],
                "rho_out": r["rho_out"],
                "max_u": r["max_u"],
                "sigma_i": sigma_i,
                "dev_sigma_pct": dev * 100.0,
                "step": r["step"],
                "mass_drift": r["mass_drift"],
                "converged": r["converged"],
            }
        )
    max_dev = max(p["dev_sigma_pct"] for p in per_r)
    intercept_frac = abs(b) / (abs(a) * max(inv_r)) if abs(a) > 0 else float("nan")
    return {
        "sigma_eff_fit": sigma_origin,
        "slope_free": float(a),
        "intercept_free": float(b),
        "R2": r2,
        "max_dev_sigma_pct": max_dev,
        "intercept_frac_pct": intercept_frac * 100.0,
        "pass_laplace": bool(max_dev <= 3.0 and r2 >= 0.999 and intercept_frac <= 0.03),
        "per_radius": per_r,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="2,3")
    ap.add_argument("--device2d", default="cpu")
    ap.add_argument("--device3d", default="cuda:0")
    ap.add_argument("--radii", default="15,25,40")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--min-steps", type=int, default=6000)
    ap.add_argument("--sample-interval", type=int, default=1000)
    ap.add_argument("--out", default=str(script_dir))
    add_compile_mode_arg(ap)
    args = ap.parse_args()
    compile_mode = compile_mode_from_args(args)

    radii = [float(x) for x in args.radii.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "benchmark": "laplace_droplet",
        "model": (
            "SCMP SC94 psi_exp (psi=1-exp(-rho)), physical G_eff=-5.0, tau=1.0; "
            "library called with G=+5.0 (library SC force sign flipped - backward "
            "gather; see README)"
        ),
        "eos": {
            "p(rho)": "rho/3 + G_eff*psi^2/6  (G_eff=-5.0, standard convention)",
            "discrete_coexistence": {"rho_l": RHO_L, "rho_v": RHO_V},
        },
        "criteria": {
            "laplace": "2D dp=sigma/R, 3D dp=2*sigma/R; per-radius sigma within "
            "3% of fit; R2>=0.999; |intercept|<=3% of slope/Rmax",
        },
        "dims": {},
    }
    for dim_s in args.dims.split(","):
        dim = int(dim_s)
        dev_s = args.device2d if dim == 2 else args.device3d
        device = torch.device(dev_s)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"WARN: {dev_s} unavailable, falling back to cpu")
            device = torch.device("cpu")
        print(f"\n===== dim={dim} device={device} =====")
        rows = []
        for R in radii:
            L = int(4 * R)
            print(f"  R={R:.0f}  L={L}", flush=True)
            try:
                final, hist = run_droplet(
                    dim,
                    R,
                    L,
                    device,
                    args.max_steps,
                    args.min_steps,
                    args.sample_interval,
                    compile_mode=compile_mode,
                )
            except RuntimeError as e:
                print(f"  FAILED: {e}")
                rows.append(
                    {
                        "R_init": R,
                        "R_eq": float("nan"),
                        "dp": float("nan"),
                        "p_in": float("nan"),
                        "p_out": float("nan"),
                        "rho_in": float("nan"),
                        "rho_out": float("nan"),
                        "max_u": float("nan"),
                        "step": 0,
                        "mass_drift": float("nan"),
                        "converged": False,
                        "error": str(e),
                    }
                )
                continue
            final["R_init"] = R
            rows.append(final)
            print(
                f"    steps={final['step']} conv={final['converged']} "
                f"R_eq={final['R_eq']:.3f} dp={final['dp']:.6f} "
                f"rho_in={final['rho_in']:.4f} rho_out={final['rho_out']:.4f} "
                f"max_u={final['max_u']:.2e} md={final['mass_drift']:.1e} "
                f"({final['dt_s']:.0f}s)"
            )
            hist_out = out / f"hist_dim{dim}_R{int(R)}.json"
            hist_out.write_text(
                json.dumps(
                    [
                        {k: (round(v, 8) if isinstance(v, float) else v) for k, v in h.items()}
                        for h in hist
                    ],
                    indent=1,
                ),
                encoding="utf-8",
            )
        fit = fit_laplace(rows, dim)
        fit["dim"] = dim
        results["dims"][f"{dim}d"] = fit
        print(
            f"  FIT dim={dim}: sigma_eff={fit['sigma_eff_fit']:.6f} "
            f"R2={fit['R2']:.6f} max_dev={fit['max_dev_sigma_pct']:.3f}% "
            f"intercept_frac={fit['intercept_frac_pct']:.3f}% "
            f"PASS={fit['pass_laplace']}"
        )

    overall = all(results["dims"][k]["pass_laplace"] for k in results["dims"])
    results["verified"] = bool(overall)
    (out / "result.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nSaved -> {out / 'result.json'}  verified={overall}")


if __name__ == "__main__":
    main()
