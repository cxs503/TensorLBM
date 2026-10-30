#!/usr/bin/env python
"""SDAA validation sweep for laplace_pr_eos benchmark.

Runs the PR-EOS Laplace droplet case on SDAA device, compares key quantities
against the baseline result.json, and writes artifacts.

This is an *independent* runner: it imports the same tensorlbm modules but
does NOT modify any business code.  It re-implements the minimal simulation
loop (copied from benchmarks/verified/laplace_pr_eos/run.py) to avoid
depending on the original runner's output structure.

Usage:
    python scripts/sdaa_sweep_laplace_pr_eos.py --device sdaa:0 --out artifacts/sdaa_laplace_pr_eos.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Ensure repo src is on path (same pattern as the original runner)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import tensorlbm  # noqa: E402
from tensorlbm.d2q9 import C, equilibrium  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, make_psi_eos  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

_REPO_SRC = str(_REPO_ROOT / "src")
assert tensorlbm.__file__.startswith(_REPO_SRC), tensorlbm.__file__

# ---------------------------------------------------------------------------
# Constants (identical to benchmarks/verified/laplace_pr_eos/run.py)
# ---------------------------------------------------------------------------
CS2 = 1.0 / 3.0
TR_MAIN = 0.55
EOS_MAIN = "pr"
BASE_R = [12, 16, 20, 24]
L_GRID = [128, 192, 256]
RESID_GATE = 0.03
MASS_RATE_GATE = 5e-8

MAXWELL_LOCK = {
    "pr": {
        "critical": {"T_c": 0.07291903717301021},
        "table": [
            {
                "T_r": 0.55,
                "T": 0.04010547044515562,
                "rho_l": 7.997835999814006,
                "rho_v": 0.0649052793822428,
                "ratio": 123.22319657100351,
                "p_sat": 0.002449394170218779,
                "rho_cross": 9.708178500398901,
            }
        ],
    },
}

# Baseline sigma values from result.json (primary, gated)
BASELINE_SIGMA = {128: 0.32022612409321427, 192: 0.32649180156335816, 256: 0.32942224369200646}
BASELINE_DP_POINTS = {
    128: [
        {"r_eff": 13.575397872924805, "dp": 0.024087546009104698},
        {"r_eff": 17.218267822265624, "dp": 0.019128882512450218},
        {"r_eff": 20.976315689086913, "dp": 0.01576374479336664},
        {"r_eff": 24.801242446899415, "dp": 0.013418000971432776},
    ],
    192: [
        {"r_eff": 20.080131149291994, "dp": 0.016464889387134463},
        {"r_eff": 25.59018440246582, "dp": 0.012990262720268219},
        {"r_eff": 31.2648193359375, "dp": 0.010630529653280974},
        {"r_eff": 37.03185119628906, "dp": 0.009036388189997524},
    ],
    256: [
        {"r_eff": 26.661817932128905, "dp": 0.01247483678162098},
        {"r_eff": 34.02446517944336, "dp": 0.009813363663852215},
        {"r_eff": 41.60483627319336, "dp": 0.008044925157446415},
        {"r_eff": 49.306514739990234, "dp": 0.006799036799930036},
    ],
}


def maxwell_row(mw: dict, T_r: float) -> dict:
    rows = [r for r in mw["table"] if abs(r["T_r"] - T_r) < 1e-12]
    assert len(rows) == 1, (T_r, len(rows))
    return rows[0]


def make_step(forcing: str, G: float, tau: float, psi_fn):
    def step(f):
        return stream(collide_sc_single_component(f, G=G, tau=tau, psi_fn=psi_fn, forcing=forcing))
    return step


def init_droplet(L: int, R: float, rho_l: float, rho_v: float, device, width: float = 4.0):
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    rho = rho_v + 0.5 * (rho_l - rho_v) * (1.0 + torch.tanh((R - r) / width))
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero)


def run_droplet(
    T, G, tau, psi_fn, L, R, rho_l0, rho_v0, device,
    max_steps=40000, min_steps=4000, sample=1000, conv_tol=1e-5,
):
    step = make_step("edm", G, tau, psi_fn)
    f = init_droplet(L, R, rho_l0, rho_v0, device)
    mass0 = float(f.sum(dim=(0, 1, 2)).item())
    ys = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    core0 = r <= 0.25 * L / 2
    far0 = r >= 0.85 * L / 2
    far = r >= 0.80 * L / 2

    hist, stable, err_msg, step_i = [], True, None, 0
    for step_i in range(1, max_steps + 1):
        f = step(f)
        if step_i % sample:
            continue
        rho = f.sum(dim=0)
        if not torch.isfinite(rho).all() or float(rho.min().item()) < 0.0:
            stable, err_msg = False, f"nonfinite/negative rho at step {step_i}"
            break
        rho_l = float(rho[core0].mean().item())
        rho_v = float(rho[far0].mean().item())
        for _ in range(2):
            r_eff = float(
                torch.sqrt(
                    torch.clamp((rho - rho_v).sum(), min=0.0)
                    / (3.141592653589793 * max(rho_l - rho_v, 1e-12))
                ).item()
            )
            if r_eff <= 0:
                break
            core = r <= 0.5 * r_eff
            rho_l = float(rho[core].mean().item())
            rho_v = float(rho[far].mean().item())
        if r_eff <= 0:
            stable, err_msg = False, f"degenerate droplet at step {step_i}"
            break
        psi = psi_fn(rho)
        p_mech = rho * CS2 - 0.5 * G * CS2 * psi * psi
        dp = float(p_mech[core].mean().item()) - float(p_mech[far].mean().item())
        mom_x = (f * C.to(device)[:, 0].view(9, 1, 1)).sum(dim=0)
        mom_y = (f * C.to(device)[:, 1].view(9, 1, 1)).sum(dim=0)
        rho_c = torch.clamp(rho, min=1e-12)
        umax = float(torch.sqrt((mom_x / rho_c) ** 2 + (mom_y / rho_c) ** 2).max().item())
        mass = float(rho.sum().item())
        hist.append(
            dict(
                step=step_i, rho_l=rho_l, rho_v=rho_v, dp=dp, r_eff=r_eff,
                u_max=umax, mass=mass, rho_min=float(rho.min().item()),
                rho_max=float(rho.max().item()),
                mass_drift=abs(mass - mass0) / mass0,
            )
        )
        if len(hist) >= 4 and step_i >= min_steps:
            rec = hist[-3:]
            if all(
                abs(h["rho_l"] - hp["rho_l"]) / max(abs(h["rho_l"]), 1e-12) <= conv_tol
                and abs(h["rho_v"] - hp["rho_v"]) / max(abs(h["rho_v"]), 1e-12) <= conv_tol
                for h, hp in zip(rec[1:], rec[:-1])
            ):
                break

    out = dict(
        forcing="edm", T=T, G=G, tau=tau, L=L, R=R,
        rho_l0=rho_l0, rho_v0=rho_v0, max_steps=max_steps,
        steps_run=int(step_i), stable=stable, error=err_msg,
        converged=bool(len(hist) >= 4 and step_i < max_steps and stable),
    )
    tail = hist[-5:] if hist else []
    if tail:
        for k in ("rho_l", "rho_v", "dp", "r_eff"):
            m = sum(t[k] for t in tail) / len(tail)
            out[k] = m
            out[k + "_std"] = (sum((t[k] - m) ** 2 for t in tail) / len(tail)) ** 0.5
        out["ratio"] = out["rho_l"] / out["rho_v"]
        out["u_max"] = max(t["u_max"] for t in tail)
        out["rho_max_field"] = max(t["rho_max"] for t in tail)
        out["rho_min_field"] = min(t["rho_min"] for t in tail)
        out["mass_drift"] = max(t["mass_drift"] for t in tail)
        out["mass_rate_per_step"] = out["mass_drift"] / out["steps_run"]
        out["mass_rate_gate_pass"] = out["mass_rate_per_step"] <= MASS_RATE_GATE
    return out


def fit_sigma(runs):
    xs = [1.0 / r["r_eff"] for r in runs if r.get("r_eff")]
    ys = [r["dp"] for r in runs if r.get("r_eff")]
    n = len(xs)
    if n < 2:
        return {"n": n, "sigma": None, "intercept": None, "rms_residual": None, "dp_max": None, "resid_gate": False, "points": []}
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(e * e for e in resid) / n)
    return dict(
        n=n, sigma=slope, intercept=intercept, rms_residual=rms,
        dp_max=max(ys), resid_gate=rms <= RESID_GATE * max(ys),
        points=[dict(r_eff=r["r_eff"], inv_r_eff=1.0 / r["r_eff"], dp=r["dp"], tag=r["tag"]) for r in runs if r.get("r_eff")],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--out", default="artifacts/sdaa_laplace_pr_eos.json")
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--l-grid", nargs="+", type=int, default=None,
                    help="L values to run (default: all three tiers)")
    args = ap.parse_args()

    device = torch.device(args.device)
    l_grid = args.l_grid if args.l_grid else L_GRID
    mw = MAXWELL_LOCK
    row = maxwell_row(mw["pr"], TR_MAIN)
    T = TR_MAIN * mw["pr"]["critical"]["T_c"]
    psi_fn = make_psi_eos("pr", T)

    print(f"SDAA sweep: device={device}, L_grid={l_grid}, max_steps={args.max_steps}")
    print(f"  rho_l={row['rho_l']}, rho_v={row['rho_v']}, T={T}")

    all_runs = []
    sigma_by_L = {}
    fit_results = {}

    for L in l_grid:
        runs_for_L = []
        for base in BASE_R:
            R = base * L / 128.0  # scaled_R (primary/gated)
            tag = f"b_pr_Tr0p55_scaled_R_L{L}_R{R:g}"
            t0 = time.time()
            print(f"[{tag}] start", flush=True)
            r = run_droplet(
                T, +1.0, 1.0, psi_fn, L, R,
                row["rho_l"], row["rho_v"], device,
                max_steps=args.max_steps,
            )
            r["tag"] = tag
            r["wall_s"] = round(time.time() - t0, 1)
            if r.get("rho_l") is not None:
                r["clamp_exposure"] = bool(r["rho_max_field"] > row["rho_cross"])
            runs_for_L.append(r)
            all_runs.append(r)
            status = "OK" if r["stable"] else "FAIL"
            print(
                f"[{tag}] {status} conv={r['converged']} steps={r['steps_run']} "
                f"dp={r.get('dp', float('nan')):.5f} r_eff={r.get('r_eff', float('nan')):.2f} "
                f"ratio={r.get('ratio', float('nan')):.1f} wall={r['wall_s']}s",
                flush=True,
            )

        fit = fit_sigma(runs_for_L)
        fit_results[L] = fit
        if fit["sigma"] is not None:
            sigma_by_L[L] = fit["sigma"]
            print(f"  L={L}: sigma={fit['sigma']:.6f}, rms_resid={fit['rms_residual']:.2e}, "
                  f"resid_gate={'PASS' if fit['resid_gate'] else 'FAIL'}")

    # ---- Comparison against baseline ----
    comparison = {}
    for L in l_grid:
        bl_sigma = BASELINE_SIGMA.get(L)
        sd_sigma = sigma_by_L.get(L)
        if bl_sigma is not None and sd_sigma is not None:
            err_pct = abs(sd_sigma - bl_sigma) / bl_sigma * 100
            comparison[L] = {
                "baseline_sigma": bl_sigma,
                "sdaa_sigma": sd_sigma,
                "error_pct": err_pct,
                "pass_3pct": err_pct < 3.0,
            }
        elif bl_sigma is not None:
            comparison[L] = {"baseline_sigma": bl_sigma, "sdaa_sigma": None, "error_pct": None, "pass_3pct": False, "note": "SDAA run did not produce sigma"}
        else:
            comparison[L] = {"baseline_sigma": None, "sdaa_sigma": sd_sigma, "error_pct": None, "pass_3pct": False, "note": "no baseline for this L"}

    # DP point comparison
    dp_comparison = {}
    for L in l_grid:
        bl_points = BASELINE_DP_POINTS.get(L, [])
        sd_runs = [r for r in all_runs if r.get("L") == L]
        if bl_points and sd_runs:
            dp_comp = []
            for bl_p, sd_r in zip(bl_points, sorted(sd_runs, key=lambda x: x["R"])):
                dp_err = abs(sd_r["dp"] - bl_p["dp"]) / bl_p["dp"] * 100 if bl_p["dp"] != 0 else 0
                re_err = abs(sd_r["r_eff"] - bl_p["r_eff"]) / bl_p["r_eff"] * 100 if bl_p["r_eff"] != 0 else 0
                dp_comp.append({
                    "tag": sd_r["tag"],
                    "baseline_dp": bl_p["dp"],
                    "sdaa_dp": sd_r["dp"],
                    "dp_error_pct": dp_err,
                    "baseline_r_eff": bl_p["r_eff"],
                    "sdaa_r_eff": sd_r["r_eff"],
                    "r_eff_error_pct": re_err,
                })
            dp_comparison[L] = dp_comp

    # Overall verdict
    all_pass = all(c.get("pass_3pct", False) for c in comparison.values() if c.get("error_pct") is not None)
    clause_i_pass = all(f.get("resid_gate", False) for f in fit_results.values())
    sigma_vals = [sigma_by_L[L] for L in sorted(sigma_by_L.keys())]
    abs_diffs = [abs(sigma_vals[i+1] - sigma_vals[i]) for i in range(len(sigma_vals)-1)]
    clause_ii_pass = all(abs_diffs[i] >= abs_diffs[i+1] for i in range(len(abs_diffs)-1)) if len(abs_diffs) >= 2 else True

    verdict = {
        "clause_i_resid_gate": clause_i_pass,
        "clause_ii_monotone": clause_ii_pass,
        "sigma_3pct_gate": all_pass,
        "overall_pass": clause_i_pass and clause_ii_pass and all_pass,
        "abs_sigma_diffs": abs_diffs,
    }

    # ---- Build output ----
    out = {
        "benchmark": "laplace_pr_eos",
        "sdaa_sweep": True,
        "device": str(device),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "eos": "pr",
            "T_r": TR_MAIN,
            "T": T,
            "G": 1.0,
            "tau": 1.0,
            "forcing": "edm",
            "l_grid": l_grid,
            "base_R": BASE_R,
            "max_steps": args.max_steps,
        },
        "baseline_source": "benchmarks/verified/laplace_pr_eos/result.json",
        "sigma_by_L": sigma_by_L,
        "fit_results": {str(k): v for k, v in fit_results.items()},
        "comparison": {str(k): v for k, v in comparison.items()},
        "dp_comparison": {str(k): v for k, v in dp_comparison.items()},
        "verdict": verdict,
        "runs": all_runs,
    }

    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults written to {outpath}")
    print(f"Verdict: {verdict}")

    return 0 if verdict["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
