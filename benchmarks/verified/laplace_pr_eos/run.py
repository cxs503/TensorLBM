#!/usr/bin/env python
"""M4 (b): PR-EOS high-density-ratio droplet Laplace gate, sigma self-test.

Protocol: prereg_v2.md section 5(b) (frozen 2026-10-01).  PR EOS at
T_r = 0.55 (Maxwell coexistence ratio 123.2), circular droplet in a
periodic L x L domain, EDM forcing with tau = 1.0, G = +1, corrected-sign
EOS pseudopotential.  R ladder {12, 16, 20, 24}, L in {128, 192, 256}.
Measurement: steady-state Delta P = <p_mech>_core - <p_mech>_far with
p_mech = rho*cs2 - (G/2)*cs2*psi^2; sigma = LSQ slope of Delta P vs
1/R_eff; R_eff = equ-area radius sqrt(Sum(rho-rho_v)/(pi*(rho_l-rho_v))).
Gates: (i) per-L RMS residual <= 3% * max(Delta P); (ii) |sigma(N_hi) -
sigma(N_lo)| non-increasing across the three L tiers (or two-tier
disclosed).  PASS iff both.  Fallback per prereg: if PR droplet tiers die
-> CS @ T_r=0.60 same geometry, reason disclosed.  sim ratio and
clamp_exposure are disclosure columns only.

AMBIGUITY DECLARED BEFORE ANY RESULT: the frozen parenthetical "(R/L
preserved -> same R ladder per tier)" admits two readings.  Both are run,
declared here:
  primary  (gated)   : R/L preserved -- R scales with L (base 12/16/20/24
                       at L=128 -> 18/24/30/36 at 192, 24/32/40/48 at 256);
                       genuine grid refinement of the interface.
  secondary(disclosure): fixed absolute R ladder {12,16,20,24} at L in
                       {192,256} (L=128 coincides with primary), i.e. the
                       same droplet in a larger domain: a domain-size
                       independence table, not gated.
No gate selection happens after numbers are seen; both tables are emitted.

Reproduction caliber: imports resolve against THIS repository tree (repo
src/ is prepended to sys.path and asserted).  The Maxwell coexistence
inputs are the equal-area construction lock values (machine-copied from
the campaign archive pr_maxwell.json / cs_maxwell.json, generator
code_sha256 embedded there; area residuals ~1e-17).  The repo functions
used here are expression-identical to the campaign runner: make_psi_eos
is the corrected-sign psi closure, and collide_sc_single_component with
forcing="edm" is the exact-difference collision (same expressions as the
staged EDM wrapper; the "sc" default branch is bit-identical to the
pre-patch implementation, locked by tests/test_multiphase_psi_eos.py).

Run: python run.py --device cpu --out out   (or --device cuda:0)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import tensorlbm  # noqa: E402
from tensorlbm.d2q9 import C, equilibrium  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, make_psi_eos  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

_REPO_SRC = str(Path(__file__).resolve().parents[3] / "src")
assert tensorlbm.__file__.startswith(_REPO_SRC), tensorlbm.__file__

CS2 = 1.0 / 3.0

TR_MAIN = 0.55
EOS_MAIN = "pr"
BASE_R = [12, 16, 20, 24]
L_GRID = [128, 192, 256]
RESID_GATE = 0.03
MASS_RATE_GATE = 5e-8

# Equal-area Maxwell construction lock values (campaign archive
# phase0/out/{cs,pr}_maxwell.json; only the rows this runner consumes).
MAXWELL_LOCK = {
    "cs": {
        "critical": {"T_c": 0.09432870314611914},
        "table": [
            {
                "T_r": 0.60,
                "T": 0.05659722188767148,
                "rho_l": 0.4062041397340105,
                "rho_v": 0.0030814585698829495,
                "ratio": 131.82203509211558,
                "p_sat": 0.00016707291001994627,
                "rho_cross": 0.5218018103044764,
            }
        ],
    },
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
    eos,
    T,
    G,
    tau,
    psi_fn,
    L,
    R,
    rho_l0,
    rho_v0,
    device,
    max_steps=40000,
    min_steps=4000,
    sample=1000,
    conv_tol=1e-5,
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
        # equ-area radius, then refined band definitions (two passes, stable)
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
                step=step_i,
                rho_l=rho_l,
                rho_v=rho_v,
                dp=dp,
                r_eff=r_eff,
                u_max=umax,
                mass=mass,
                rho_min=float(rho.min().item()),
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
        forcing="edm",
        eos=eos,
        T=T,
        G=G,
        tau=tau,
        L=L,
        R=R,
        rho_l0=rho_l0,
        rho_v0=rho_v0,
        max_steps=max_steps,
        steps_run=int(step_i),
        stable=stable,
        error=err_msg,
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
    import math

    xs = [1.0 / r["r_eff"] for r in runs if r.get("r_eff")]
    ys = [r["dp"] for r in runs if r.get("r_eff")]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(e * e for e in resid) / n)
    return dict(
        n=n,
        sigma=slope,
        intercept=intercept,
        rms_residual=rms,
        dp_max=max(ys),
        resid_gate=rms <= RESID_GATE * max(ys),
        points=[
            dict(r_eff=r["r_eff"], inv_r_eff=1.0 / r["r_eff"], dp=r["dp"], tag=r["tag"])
            for r in runs
            if r.get("r_eff")
        ],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-steps", type=int, default=40000)
    args = ap.parse_args()
    outdir = Path(args.out)
    device = torch.device(args.device)
    maxw = MAXWELL_LOCK

    def ladder(eos, T_r, geom):
        mw = maxw[eos]
        row = maxwell_row(mw, T_r)
        T = T_r * mw["critical"]["T_c"]
        psi_fn = make_psi_eos(eos, T)
        runs = []
        for L in L_GRID:
            for base in BASE_R:
                R = base if geom == "fixed_R" else base * L / 128.0
                tag = f"b_{eos}_Tr{str(T_r).replace('.', 'p')}_{geom}_L{L}_R{R:g}"
                print(f"[{tag}] start", flush=True)
                t0 = time.time()
                r = run_droplet(
                    eos,
                    T,
                    +1.0,
                    1.0,
                    psi_fn,
                    L,
                    R,
                    row["rho_l"],
                    row["rho_v"],
                    device,
                    max_steps=args.max_steps,
                )
                r.update(
                    tag=tag,
                    T_r=T_r,
                    L=L,
                    R_requested=R,
                    rho_l_maxwell=row["rho_l"],
                    rho_v_maxwell=row["rho_v"],
                    ratio_maxwell=row["ratio"],
                    rho_cross=row["rho_cross"],
                )
                r["wall_s"] = round(time.time() - t0, 1)
                if r.get("rho_l") is not None:
                    r["clamp_exposure"] = bool(r["rho_max_field"] > row["rho_cross"])
                runs.append(r)
                print(
                    f"[{tag}] stable={r['stable']} conv={r['converged']} steps={r['steps_run']} "
                    f"dp={r.get('dp', float('nan')):.5f} r_eff={r.get('r_eff', float('nan')):.2f} "
                    f"ratio={r.get('ratio', float('nan')):.1f} u_max={r.get('u_max', float('nan')):.2e} "
                    f"mass_rate={r['mass_rate_per_step']:.2e}"
                    if r.get("dp") is not None
                    else f"[{tag}] stable={r['stable']} err={r['error']}",
                    flush=True,
                )
                (outdir / "b_laplace_partial.json").write_text(
                    json.dumps(runs, indent=1) + "\n", encoding="utf-8"
                )
        return runs

    primary = ladder(EOS_MAIN, TR_MAIN, "scaled_R")
    dead = [r for r in primary if not r["stable"]]
    result_extra = {}
    if dead:
        result_extra["fallback"] = dict(
            triggered=True,
            reason=f"PR droplet unstable tiers: {[r['tag'] for r in dead]}",
            rule="prereg §5(b): fallback EOS CS @ T_r=0.60, reason disclosed",
        )
        primary_fb = ladder("cs", 0.60, "scaled_R")
    else:
        result_extra["fallback"] = dict(triggered=False)

    secondary = ladder(EOS_MAIN, TR_MAIN, "fixed_R")
    # secondary L=128 coincides with primary L=128; reuse those rows
    secondary_all = [r for r in primary if r["L"] == 128] + secondary

    def verdict_block(runs, gated):
        per_L = {}
        for L in L_GRID:
            rows = [r for r in runs if r["L"] == L]
            if len(rows) != len(BASE_R) or not all(r.get("dp") is not None for r in rows):
                per_L[str(L)] = dict(complete=False, n=len(rows))
                continue
            per_L[str(L)] = dict(complete=True, **fit_sigma(rows))
        sig = [per_L.get(str(L), {}).get("sigma") for L in L_GRID]
        diffs = [abs(b - a) for a, b in zip(sig, sig[1:]) if a is not None and b is not None]
        mono = (
            all(d1 >= d2 - 1e-12 for d1, d2 in zip(diffs, diffs[1:])) if len(diffs) >= 2 else None
        )
        clause_i = all(v.get("resid_gate") for v in per_L.values() if v.get("complete"))
        return dict(
            gated=gated,
            per_L=per_L,
            sigma_by_L=dict(zip(map(str, L_GRID), sig)),
            abs_sigma_diffs=diffs,
            clause_i_resid=clause_i,
            clause_ii_monotone_nonincreasing=mono,
            clause_ii_note="three-tier monotone required, or two-tier disclosed",
            pass_both=bool(clause_i and mono),
        )

    v_primary = verdict_block(primary, True)
    v_secondary = verdict_block(secondary_all, False)

    result = dict(
        benchmark="m4_formal_b_laplace_sigma_selftest",
        prereg="prereg_v2.md §5(b) + v2.1/v2.2, frozen 2026-10-01",
        forcing="edm",
        tau=1.0,
        G=+1.0,
        psi="make_psi_eos",
        eos_main=EOS_MAIN,
        T_r_main=TR_MAIN,
        base_R=BASE_R,
        L_grid=L_GRID,
        resid_gate_frac=RESID_GATE,
        ambiguity_declaration=(
            "frozen text '(R/L 保持 -> 每档同 R 阶梯)' ambiguous; BOTH readings run "
            "with declaration before any result: primary gated = R/L preserved "
            "(R scales with L, grid refinement); secondary disclosure = fixed "
            "absolute R ladder (domain-size independence); no post-hoc selection"
        ),
        verdict=dict(primary=v_primary, secondary=v_secondary, **result_extra),
        runs=primary + secondary + (primary_fb if dead else []),
        generated_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
    )
    path = outdir / "b_laplace.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    for name, v in (("PRIMARY(gated)", v_primary), ("SECONDARY(disclosure)", v_secondary)):
        print(
            name,
            "sigma_by_L=",
            v["sigma_by_L"],
            "resid=",
            v["clause_i_resid"],
            "mono=",
            v["clause_ii_monotone_nonincreasing"],
            "PASS=",
            v["pass_both"],
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
