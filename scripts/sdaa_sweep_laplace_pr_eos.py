#!/usr/bin/env python
"""SDAA sweep: laplace_pr_eos benchmark (TensorLBM benchmarks/verified/laplace_pr_eos).

Runs the PR-EOS high-density-ratio droplet Laplace gate on SDAA device.
Criteria are identical to the original benchmark (prereg §5(b) frozen):
  clause_i  : per-L LSQ residual RMS <= 3% * max(Delta P)
  clause_ii : |sigma(L_hi) - sigma(L_lo)| non-increasing across three L tiers
  stability : finite, non-negative density
  mass_cons : mass drift rate <= 5e-8 / step
  psi_domain: rho_max < rho_cross (clamp not triggered)
  sim_ratio : steady-state rho_l/rho_v >= 100 (disclosure only)

Configuration: D2Q9 SCMP, PR EOS, T_r=0.55, EDM forcing, tau=1, G=+1, float32, periodic.
Primary (gated): R/L preserved (R scales with L).
Grid: L = {128, 192, 256}, R = {12, 16, 20, 24} scaled by L/128.
Max steps: 15000 (with early convergence at min 2000, sample 500).

No modification to business code. All computation on 'sdaa' device.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── TRITON_INTERPRET bypass ──────────────────────────────────────────
# If any import path touches triton kernels, force the interpreter fallback.
os.environ.setdefault("TRITON_INTERPRET", "1")
TRITON_BYPASS = os.environ.get("TRITON_INTERPRET") == "1"

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tensorlbm  # noqa: E402
from tensorlbm.d2q9 import C, equilibrium  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, make_psi_eos  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

_REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")
assert tensorlbm.__file__.startswith(_REPO_SRC), tensorlbm.__file__

CS2 = 1.0 / 3.0

# ── Maxwell lock values (identical to original run.py) ───────────────
MAXWELL_LOCK_PR = {
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
}

BASE_R = [12, 16, 20, 24]
L_GRID = [128, 192, 256]
RESID_GATE = 0.03
MASS_RATE_GATE = 5e-8


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
    max_steps=15000, min_steps=2000, sample=500, conv_tol=1e-5,
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
        hist.append(dict(
            step=step_i, rho_l=rho_l, rho_v=rho_v, dp=dp, r_eff=r_eff,
            u_max=umax, mass=mass,
            rho_min=float(rho.min().item()), rho_max=float(rho.max().item()),
            mass_drift=abs(mass - mass0) / mass0,
        ))
        if len(hist) >= 4 and step_i >= min_steps:
            rec = hist[-3:]
            if all(
                abs(h["rho_l"] - hp["rho_l"]) / max(abs(h["rho_l"]), 1e-12) <= conv_tol
                and abs(h["rho_v"] - hp["rho_v"]) / max(abs(h["rho_v"]), 1e-12) <= conv_tol
                for h, hp in zip(rec[1:], rec[:-1])
            ):
                break

    out = dict(
        eos="pr", T=T, G=G, tau=tau, L=L, R=R,
        rho_l0=rho_l0, rho_v0=rho_v0,
        max_steps=max_steps, steps_run=int(step_i),
        stable=stable, error=err_msg,
        converged=bool(len(hist) >= 4 and step_i < max_steps and stable),
        hist=hist,
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
        return dict(n=n, sigma=None, intercept=None, rms_residual=None,
                    dp_max=max(ys) if ys else None, resid_gate=False, points=[])
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(e * e for e in resid) / n)
    return dict(
        n=n, sigma=slope, intercept=intercept, rms_residual=rms,
        dp_max=max(ys), resid_gate=rms <= RESID_GATE * max(ys),
        points=[dict(r_eff=r["r_eff"], inv_r_eff=1.0 / r["r_eff"],
                     dp=r["dp"], tag=r["tag"]) for r in runs if r.get("r_eff")],
    )


def main():
    t_global_start = time.time()

    # ── Device evidence ──────────────────────────────────────────────
    device = torch.device("sdaa")
    try:
        device_name = torch.sdaa.get_device_name(0)
    except Exception as e:
        device_name = f"ERROR: {e}"
    device_evidence = {
        "device_type": "sdaa",
        "device_name": device_name,
        "torch_version": torch.__version__,
        "sdaa_device_count": torch.sdaa.device_count(),
    }
    print(f"[SDAA] device={device_name}", flush=True)

    # ── Maxwell lock ─────────────────────────────────────────────────
    mw = MAXWELL_LOCK_PR
    row = [r for r in mw["table"] if abs(r["T_r"] - 0.55) < 1e-12][0]
    T = 0.55 * mw["critical"]["T_c"]
    psi_fn = make_psi_eos("pr", T)
    rho_l0, rho_v0 = row["rho_l"], row["rho_v"]
    rho_cross = row["rho_cross"]

    # ── Primary (gated): R/L preserved ───────────────────────────────
    primary_runs = []
    for L in L_GRID:
        for base in BASE_R:
            R = base * L / 128.0
            tag = f"primary_L{L}_R{R:g}"
            print(f"[{tag}] start", flush=True)
            t0 = time.time()
            r = run_droplet(T, 1.0, 1.0, psi_fn, L, R, rho_l0, rho_v0, device)
            r.update(tag=tag, T_r=0.55, L=L, R_requested=R,
                     rho_l_maxwell=rho_l0, rho_v_maxwell=rho_v0,
                     ratio_maxwell=row["ratio"], rho_cross=rho_cross)
            r["wall_s"] = round(time.time() - t0, 1)
            if r.get("rho_l") is not None:
                r["clamp_exposure"] = bool(r["rho_max_field"] > rho_cross)
            primary_runs.append(r)
            print(
                f"[{tag}] stable={r['stable']} conv={r['converged']} "
                f"steps={r['steps_run']} dp={r.get('dp', float('nan')):.5f} "
                f"r_eff={r.get('r_eff', float('nan')):.2f} "
                f"ratio={r.get('ratio', float('nan')):.1f} "
                f"wall={r['wall_s']}s",
                flush=True,
            )

    # ── Per-L sigma fits (primary) ───────────────────────────────────
    per_L = {}
    sigma_by_L = {}
    for L in L_GRID:
        l_runs = [r for r in primary_runs if r["L"] == L and r.get("r_eff")]
        fit = fit_sigma(l_runs)
        per_L[str(L)] = fit
        sigma_by_L[str(L)] = fit["sigma"]

    # ── Clause evaluation ────────────────────────────────────────────
    clause_i_pass = all(per_L[str(L)]["resid_gate"] for L in L_GRID)

    sigma_vals = [sigma_by_L[str(L)] for L in L_GRID]
    abs_sigma_diffs = [abs(sigma_vals[i + 1] - sigma_vals[i]) for i in range(len(sigma_vals) - 1)]
    clause_ii_pass = all(abs_sigma_diffs[i] >= abs_sigma_diffs[i + 1]
                         for i in range(len(abs_sigma_diffs) - 1))

    stability_pass = all(r["stable"] for r in primary_runs)
    mass_cons_pass = all(r.get("mass_rate_gate_pass", False) for r in primary_runs)
    psi_domain_pass = all(not r.get("clamp_exposure", True) for r in primary_runs)
    sim_ratio_disclosure = all(r.get("ratio", 0) >= 100 for r in primary_runs if r.get("ratio"))

    # ── Build verdict ────────────────────────────────────────────────
    verdict = {
        "primary": {
            "gated": True,
            "per_L": per_L,
            "sigma_by_L": sigma_by_L,
            "abs_sigma_diffs": abs_sigma_diffs,
            "clause_i_resid": clause_i_pass,
            "clause_ii_monotone_nonincreasing": clause_ii_pass,
            "pass_both": clause_i_pass and clause_ii_pass,
        },
        "stability": {"all_stable": stability_pass},
        "mass_conservation": {"all_pass": mass_cons_pass},
        "psi_domain": {"no_clamp_trigger": psi_domain_pass},
        "sim_ratio_disclosure": {"all_above_100": sim_ratio_disclosure},
    }

    # ── Per-run gate details ─────────────────────────────────────────
    run_summaries = []
    for r in primary_runs:
        s = {
            "tag": r["tag"], "L": r["L"], "R": r["R_requested"],
            "stable": r["stable"], "converged": r["converged"],
            "steps_run": r["steps_run"], "wall_s": r["wall_s"],
        }
        if r.get("r_eff") is not None:
            s.update({
                "rho_l": r["rho_l"], "rho_v": r["rho_v"],
                "dp": r["dp"], "r_eff": r["r_eff"],
                "ratio": r["ratio"],
                "mass_rate_per_step": r["mass_rate_per_step"],
                "mass_rate_gate_pass": r["mass_rate_gate_pass"],
                "clamp_exposure": r.get("clamp_exposure"),
            })
        else:
            s["error"] = r.get("error", "no r_eff")
        run_summaries.append(s)

    # ── Assemble output ──────────────────────────────────────────────
    wall_total = round(time.time() - t_global_start, 1)

    output = {
        "benchmark": "laplace_pr_eos",
        "prereg": "prereg_v2.md §5(b), frozen 2026-10-01",
        "device_evidence": device_evidence,
        "triton_bypass": TRITON_BYPASS,
        "triton_bypass_note": (
            "TRITON_INTERPRET=1 set as precaution; this benchmark does not "
            "use triton kernels directly (verified by source inspection of "
            "make_psi_eos, collide_sc_single_component, stream)."
        ),
        "configuration": {
            "eos": "pr", "T_r": 0.55, "T": T,
            "G": 1.0, "tau": 1.0, "forcing": "edm",
            "L_grid": L_GRID, "base_R": BASE_R,
            "geometry": "R/L_preserved",
            "max_steps": 15000, "min_steps": 2000, "sample": 500,
            "conv_tol": 1e-5, "dtype": "float32",
        },
        "verdict": verdict,
        "run_summaries": run_summaries,
        "wall_total_s": wall_total,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # ── Write artifacts ──────────────────────────────────────────────
    artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "sdaa_laplace_pr_eos.json"
    with open(out_path, "w") as fp:
        json.dump(output, fp, indent=2, default=str)
    print(f"\n[ARTIFACT] {out_path}", flush=True)

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("SDAA LAPLACE PR-EOS SWEEP SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Device: {device_name}", flush=True)
    print(f"Wall clock: {wall_total}s", flush=True)
    print(f"Triton bypass: {TRITON_BYPASS}", flush=True)
    print()
    print("Per-L sigma fits (primary, R/L preserved):", flush=True)
    for L in L_GRID:
        fit = per_L[str(L)]
        print(f"  L={L}: sigma={fit['sigma']:.6f}  "
              f"resid_rms={fit['rms_residual']:.2e}  "
              f"dp_max={fit['dp_max']:.5f}  "
              f"resid_gate={'PASS' if fit['resid_gate'] else 'FAIL'}",
              flush=True)
    print()
    print(f"Sigma diffs: {[f'{d:.6e}' for d in abs_sigma_diffs]}", flush=True)
    print()
    print("Gate results:", flush=True)
    print(f"  clause_i  (residual RMS <= 3% max DP):  {'PASS' if clause_i_pass else 'FAIL'}", flush=True)
    print(f"  clause_ii (|Δσ| non-increasing):         {'PASS' if clause_ii_pass else 'FAIL'}", flush=True)
    print(f"  stability (finite, non-negative ρ):      {'PASS' if stability_pass else 'FAIL'}", flush=True)
    print(f"  mass_cons  (drift ≤ 5e-8/step):          {'PASS' if mass_cons_pass else 'FAIL'}", flush=True)
    print(f"  psi_domain (no clamp trigger):            {'PASS' if psi_domain_pass else 'FAIL'}", flush=True)
    print(f"  sim_ratio  (ρ_l/ρ_v ≥ 100, disclosure):  {'PASS' if sim_ratio_disclosure else 'FAIL'}", flush=True)
    print()
    overall = clause_i_pass and clause_ii_pass and stability_pass and mass_cons_pass and psi_domain_pass
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
