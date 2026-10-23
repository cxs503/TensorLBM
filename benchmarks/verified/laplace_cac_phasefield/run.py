#!/usr/bin/env python3
"""B1/B3 static-droplet Laplace benchmark — CAC conservative Allen-Cahn.

Model: two-phase D3Q19 CAC/HCH lattice Boltzmann (tensorlbm.cac_lbm),
       fp64, harmonic mu(phi) law, interface width W=4, periodic cube.

Physics: liquid droplet in gas, Young-Laplace 3D DeltaP = 2*sigma/R.
  sigma_rec = DeltaP * R / 2 measured per run on a fixed nominal R
  (bands inside phi>0.9 / outside phi<0.1 avoid the diffuse interface).

Route-B sigma calibration (frozen, see README):
  the CAC scheme realises sigma_eff(W) != sigma_input; a W=4 scan
  (R in {20,28,40}, 128^3) gives bar_eff(W=4) = 0.9238958795895513,
  monotone in W and R-independent, so the model input that realises
  sigma_target = 0.01 is  sigma_model = 0.01 / bar_eff(W=4).

B1 arms: R in {20, 32, 48}, 128^3, rho ratio 100 (R-transfer ladder).
B3 arms: rho ratio in {10, 100}, 96^3, R=20 (density-ratio transfer).
Both reuse the same Route-B constant — no per-arm retuning.

Gates: per-arm |sigma_rec/0.01 - 1| <= 1.5%, B1 spread <= 1%,
       u_max <= 2e-4, phi drift at 10k steps <= 1e-6, NaN-free.

Usage: python run.py [--arms b1_R20,b1_R32,b1_R48,b3_rho10,b3_rho100]
                     [--device auto] [--sigma auto] [--steps 12000]
                     [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))  # own repo src

import torch  # noqa: E402

from tensorlbm import cac_lbm as m  # noqa: E402

SIGMA_TARGET = 0.01
BAR_EFF_W4 = 0.9238958795895513  # measured, R-independent (README)
SIGMA_ROUTE_B = SIGMA_TARGET / BAR_EFF_W4
# frozen sigma_eff(W) calibration scan (R in {20,28,40}, 128^3, fp64, GPU),
# monotone in W, R-spread <= 0.4% per W; W=2 unstable (whole-field NaN)
SIGMA_EFF_LADDER = {
    3: 0.8779650815057204,
    5: 0.9479940754517359,
    6: 0.9647459873384855,
    8: 0.9813195231472474,
}
W_STAR_ON_VALID = 8.0
LINEARITY_REL_DIFF = 0.0005702514186345375  # sigma x0.02 -> ratio x0.0202
N_STEPS = 12_000
MOB = 0.05
NU = 0.1  # both phases (frozen)
W_INT = 4.0
DEFAULT_ARMS = ("b1_R20", "b1_R32", "b1_R48", "b3_rho10", "b3_rho100")

GATES = {
    "per_arm_err_pct": 1.5,
    "b1_spread_pct": 1.0,
    "u_max": 2.0e-4,
    "drift_10k": 1.0e-6,
}


def run_arm(
    kind: str, arg: float, sigma: float, device: torch.device, dtype: torch.dtype, n_steps: int
) -> dict:
    if kind == "b1":
        shape = (128, 128, 128)
        radius = float(arg)
        rho_l, rho_g = 1.0, 0.01
    else:
        shape = (96, 96, 96)
        radius = 20.0
        rho_l, rho_g = 1.0, 1.0 / float(arg)
    mu_l, mu_g = NU * rho_l, NU * rho_g

    sim = m.CACSim(
        shape,
        rho_l=rho_l,
        rho_g=rho_g,
        mu_l=mu_l,
        mu_g=mu_g,
        sigma=sigma,
        width=W_INT,
        mobility=MOB,
        p0=0.01,
        axes="ppp",
        mu_law="harmonic",
        dtype=dtype,
        device=device,
        gravity=(0.0, 0.0, 0.0),
        gravity_mode="rho",
    )
    center = tuple(s / 2.0 for s in shape)
    sim.initialize(
        m.init_phi_droplet(shape, radius=radius, width=W_INT, center=center, dtype=dtype)
    )

    t0 = time.time()
    series = []
    phi0_int = None
    for it in range(1, n_steps + 1):
        sim.step()
        if it == 1 or it % 500 == 0:
            phi, rho, grad, mu_phi, F, u, p, u_star = m._macro_fp64(sim)
            if phi0_int is None:
                phi0_int = phi.sum().item()
            inside = phi > 0.9
            outside = phi < 0.1
            p_in = p[inside].mean().item() if inside.any() else float("nan")
            p_out = p[outside].mean().item() if outside.any() else float("nan")
            umax = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt().max().item()
            series.append(
                {
                    "step": it,
                    "phi_integral": phi.sum().item(),
                    "phi_drift_rel": abs(phi.sum().item() - phi0_int) / phi0_int,
                    "p_in": p_in,
                    "p_out": p_out,
                    "dP": p_in - p_out,
                    "sigma_rec": (p_in - p_out) * radius / 2.0,
                    "u_max": umax,
                    "rho_min": rho.min().item(),
                    "rho_max": rho.max().item(),
                    "nan_f": int(torch.isnan(sim.f).sum().item()),
                    "nan_g": int(torch.isnan(sim.g).sum().item()),
                }
            )
            if it % 2000 == 0:
                s = series[-1]
                print(
                    f"[{kind}/{arg:g}] step {it}  sig_rec {s['sigma_rec']:.5f}  "
                    f"u_max {s['u_max']:.2e}  drift {s['phi_drift_rel']:.2e}",
                    flush=True,
                )

    win = [s for s in series if s["step"] > n_steps - 2000]
    return {
        "kind": kind,
        "arg": arg,
        "params": dict(
            shape=list(shape),
            radius=radius,
            rho_l=rho_l,
            rho_g=rho_g,
            mu_l=mu_l,
            mu_g=mu_g,
            nu=NU,
            sigma=sigma,
            W=W_INT,
            M_mob=MOB,
            p0=0.01,
            n_steps=n_steps,
            axes="ppp",
            mu_law="harmonic",
            dtype=str(dtype),
            device=str(device),
            window="last 2000 steps, samples every 500",
        ),
        "sigma_target": SIGMA_TARGET,
        "sigma_model_input": sigma,
        "sigma_rec_window_mean": sum(s["sigma_rec"] for s in win) / len(win),
        "sigma_rec_rel_err": abs(sum(s["sigma_rec"] for s in win) / len(win) - SIGMA_TARGET)
        / SIGMA_TARGET,
        "dP_window_mean": sum(s["dP"] for s in win) / len(win),
        "u_max_window": max(s["u_max"] for s in win),
        "B4_drift_rel_at_10k": next(s["phi_drift_rel"] for s in series if s["step"] == 10_000),
        "nan_free": all(s["nan_f"] == 0 and s["nan_g"] == 0 for s in series),
        "series_len": len(series),
        "tail_sigma_rec": [s["sigma_rec"] for s in win],
        "wall_time_s": time.time() - t0,
    }


def summarize(d: dict) -> dict:
    """Fixed projection of one arm run (or archived campaign run) for result.json."""
    pm = d["params"]
    out = {
        k: d[k]
        for k in (
            "kind",
            "arg",
            "sigma_target",
            "sigma_model_input",
            "sigma_rec_window_mean",
            "sigma_rec_rel_err",
            "dP_window_mean",
            "u_max_window",
            "B4_drift_rel_at_10k",
            "nan_free",
            "series_len",
            "tail_sigma_rec",
        )
    }
    out["err_pct"] = d["sigma_rec_rel_err"] * 100.0
    out["params"] = {
        k: pm[k]
        for k in (
            "shape",
            "radius",
            "rho_l",
            "rho_g",
            "mu_l",
            "mu_g",
            "nu",
            "sigma",
            "W",
            "M_mob",
            "n_steps",
            "mu_law",
            "dtype",
            "device",
            "window",
        )
    }
    return out


def score(arms: dict) -> dict:
    b1 = [a for a in arms.values() if a["kind"] == "b1"]
    b3 = [a for a in arms.values() if a["kind"] == "b3"]
    if b1:
        sig = [a["sigma_rec_window_mean"] for a in b1]
        spread = (max(sig) - min(sig)) / (sum(sig) / len(sig)) * 100.0
    else:
        spread = None
    gates = {
        "per_arm_err_max_pct": max(a["err_pct"] for a in arms.values()),
        "b1_spread_pct": spread,
        "u_max_max": max(a["u_max_window"] for a in arms.values()),
        "drift_10k_max": max(a["B4_drift_rel_at_10k"] for a in arms.values()),
        "nan_free_all": all(a["nan_free"] for a in arms.values()),
        "b3_arms_present": sorted(a["arg"] for a in b3),
        "b1_arms_present": sorted(a["arg"] for a in b1),
    }
    gates["pass"] = bool(
        gates["per_arm_err_max_pct"] <= GATES["per_arm_err_pct"]
        and spread is not None
        and spread <= GATES["b1_spread_pct"]
        and gates["u_max_max"] <= GATES["u_max"]
        and gates["drift_10k_max"] <= GATES["drift_10k"]
        and gates["nan_free_all"]
        and len(b1) == 3
        and len(b3) == 2
    )
    return gates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--sigma", default="auto", help="model sigma; default = Route-B 0.01/bar_eff(W=4)"
    )
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    sigma = SIGMA_ROUTE_B if args.sigma == "auto" else float(args.sigma)
    dtype = torch.float64

    # iron rule: zero hand-written collide/stream kernels in this file
    src = open(os.path.abspath(__file__)).read()
    pats = [r"def\s+bounce", r"def\s+stream_3d", r"torch\.roll\(\s*f\b"]
    grep = {p: len(re.findall(p, src)) for p in pats}
    assert all(v == 0 for v in grep.values()), grep

    arms: dict = {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        kind = "b1" if arm.startswith("b1") else "b3"
        arg = float(arm.split("_R")[1] if kind == "b1" else arm.split("_rho")[1])
        print(f"===== {arm} on {device} (sigma={sigma:.15g}) =====", flush=True)
        arms[arm] = summarize(run_arm(kind, arg, sigma, device, dtype, args.steps))

    results = {
        "benchmark": "laplace_cac_phasefield",
        "model": (
            "CAC conservative Allen-Cahn two-phase LBM, D3Q19, fp64, "
            "harmonic mu(phi), W=4, tensorlbm.cac_lbm (Li et al PRE 97,033309)"
        ),
        "calibration": {
            "route": "B",
            "sigma_target": SIGMA_TARGET,
            "bar_eff_W4": BAR_EFF_W4,
            "sigma_model_input": SIGMA_ROUTE_B,
            "sigma_eff_ladder": {str(k): v for k, v in sorted(SIGMA_EFF_LADDER.items())},
            "W_star_on_valid": W_STAR_ON_VALID,
            "linearity_P7_rel_diff": LINEARITY_REL_DIFF,
            "note": "sigma_eff(W) measured once (R-independent, monotone in W); "
            "model input = target/bar_eff(W=4); same constant for all "
            "B1 radii and B3 density ratios",
        },
        "criteria": {
            "per_arm_err_pct": GATES["per_arm_err_pct"],
            "b1_spread_pct": GATES["b1_spread_pct"],
            "u_max": GATES["u_max"],
            "drift_10k": GATES["drift_10k"],
            "nan_free": True,
        },
        "arms": arms,
        "gates": score(arms),
    }
    results["verified"] = results["gates"]["pass"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    g = results["gates"]
    print(
        f"\narms={len(arms)} err_max={g['per_arm_err_max_pct']:.4f}% "
        f"spread={g['b1_spread_pct']:.4f}% u_max={g['u_max_max']:.2e} "
        f"drift={g['drift_10k_max']:.2e} PASS={g['pass']}"
    )
    print(f"Saved -> {out / 'result.json'}")


if __name__ == "__main__":
    main()
