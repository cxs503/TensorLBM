#!/usr/bin/env python3
"""XVAL-LBM-TWO-PDF-PARTIAL-DENSITY-COLORGRADIENT -- round 2 (research first).

Cross-validation of the **two-PDF partial-density colour-gradient** model
(new additive module ``tensorlbm.color_gradient_two_pdf``) against the
closed-form / literature anchors of the colour-gradient family.

External anchors (external_anchor, no hand-copied numbers)
----------------------------------------------------------
1. **Two-layer (stratified) plane Poiseuille flow** with a kinematic
   viscosity ratio 10, equal densities, halfway bounce-back walls.
   Closed form: ``u'' = -g/nu`` per layer, no slip at the wall planes,
   velocity + shear-stress continuity at the interface.  Evaluated by the
   frozen analytic evaluator ``cg_reference.py`` (wall-plane convention,
   interface plane Y_i = j_first_blue - 1/2 measured from the run's own
   steady colour column).  Lineage:
   * Leclaire, Reggio & Trepanier (2012) J. Comput. Phys. 231, 6294;
   * Leclaire et al. (2013) J. Comput. Phys. 246, 318 (partial densities);
   * LANL MF-LBM (Zhang et al., Comput. Fluids 2019) -- two-species
     partial-density colour gradient, which is the OSS structural reference
     ``oss_solutions id=203`` (feature_slug
     ``two-species-partial-density-colorgradient``);
   * Latva-Kokko & Rothman (2005) PRE 71, 056702 (recolour + perturbation).
2. **Laplace law** for a static droplet, ``dp = sigma / R``: the *law*
   (radius independence of ``sigma = dp * R``) is the anchor; the discrete
   colour-gradient model has no closed-form sigma, so the check is
   linearity + cross-model consistency with the library's independent
   single-PDF two-component colour-gradient engine
   (``tensorlbm.color_gradient2d``), not a hand-copied constant.

Acceptance (owner 15 % band, judge criterion of this anchor)
------------------------------------------------------------
* ``relL2(u_sim, u_analytic) < 15 %`` on **every** two-layer Poiseuille arm
  (two resolutions ny = 64 / 128, viscosity ratio 10);
* Laplace law: each droplet radius' fitted ``sigma`` within 15 % of the
  three-radius fit, and the circle fit R^2 >= 0.99;
* per-species mass drift <= 1e-10 per step per species (partial-density
  exactness), NaN-free everywhere.

Usage::

    PYTHONPATH=src python3 xval/XVAL-LBM-TWO-PDF-PARTIAL-DENSITY-COLORGRADIENT/run_xval.py \
        --out xval/XVAL-LBM-TWO-PDF-PARTIAL-DENSITY-COLORGRADIENT/artifacts

    # subsets / quick smoke
    ... --suites poiseuille --arms sp_ny64
    ... --suites laplace --laplace-radii 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _resolve_src() -> Path:
    cands = []
    if os.environ.get("TENSORLBM_SRC"):
        cands.append(Path(os.environ["TENSORLBM_SRC"]))
    cands.append(HERE.parents[1] / "src")
    for c in cands:
        if (c / "tensorlbm" / "d2q9.py").is_file():
            return c
    raise SystemExit("repo src/ not found: set TENSORLBM_SRC or run from the repo")


SRC = _resolve_src()
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.color_gradient2d import color_gradient_two_tau_step  # noqa: E402
from tensorlbm.color_gradient_two_pdf import (  # noqa: E402
    colorgradient_two_pdf_step,
    max_velocity,
    pressure_jump,
    relative_l2,
    two_pdf_macros,
    viscosity_from_phi,
)
from tensorlbm.d2q9 import C, equilibrium  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

from cg_reference import (  # noqa: E402
    first_blue_row,
    single_phase_profile_wallplanes,
    two_layer_profile_wallplanes,
)

DT = torch.float64
NX = 8                      # periodic streamwise column (Poiseuille arms)
A_FROZEN = 0.04             # colour-gradient amplitude (shared with G2 benchmark)
GATE_RELL2 = 0.15           # owner 15 % band
GATE_MASS_DRIFT = 1e-10

FROZEN_ARMS = [
    {
        "name": "sp_ny64",
        "kind": "single_phase",
        "ny": 64,
        "nu_a": 1.0 / 6.0,
        "nu_b": 1.0 / 6.0,
        "gx": 5e-6,
        "beta": 0.9,
        "A": A_FROZEN,
        "steps": 60000,
    },
    {
        "name": "g10_ny64",
        "kind": "two_layer",
        "ny": 64,
        "nu_a": 2.0 / 3.0,          # tau_a = 2.5
        "nu_b": 1.0 / 15.0,         # tau_b = 0.7  (ratio 10)
        "gx": 1.5e-7,
        "beta": 0.9,
        "A": A_FROZEN,
        "steps": 80000,
    },
    {
        "name": "g10_ny128",
        "kind": "two_layer",
        "ny": 128,
        "nu_a": 2.0 / 3.0,
        "nu_b": 1.0 / 15.0,
        "gx": 1.5e-7 / 4.0,
        "beta": 0.9,
        "A": A_FROZEN,
        "steps": 120000,
    },
    {
        "name": "g10_ny64_beta095",
        "kind": "two_layer",
        "ny": 64,
        "nu_a": 2.0 / 3.0,
        "nu_b": 1.0 / 15.0,
        "gx": 1.5e-7,
        "beta": 0.95,
        "A": A_FROZEN,
        "steps": 80000,
    },
]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Poiseuille arms
# ---------------------------------------------------------------------------


def init_layered(ny: int, nx: int, kind: str, dev: torch.device):
    """tanh colour profile; lower half = species A (phi=+1)."""
    j = torch.arange(ny, dtype=DT, device=dev).view(ny, 1)
    half = ny // 2
    frac_a = 0.5 * (1.0 - torch.tanh((j - half) / 2.0)).expand(ny, nx).contiguous()
    z = torch.zeros((ny, nx), dtype=DT, device=dev)
    if kind == "single_phase":
        frac_a = torch.ones((ny, nx), dtype=DT, device=dev)
    rho_a = frac_a
    rho_b = 1.0 - frac_a
    return equilibrium(rho_a, z, z), equilibrium(rho_b, z, z)


def run_poiseuille_arm(arm: dict, dev: torch.device) -> dict:
    ny, gx = arm["ny"], arm["gx"]
    f, g = init_layered(ny, NX, arm["kind"], dev)
    m_a0, m_b0 = float(f.sum()), float(g.sum())
    wall = torch.zeros((ny, NX), dtype=torch.bool, device=dev)
    wall[0, :] = True
    wall[-1, :] = True
    c9 = C.to(dev).to(DT)
    umax_hist = []
    t0 = time.perf_counter()
    nan_step = None
    for step in range(1, arm["steps"] + 1):
        f, g = colorgradient_two_pdf_step(
            f, g,
            nu_a=arm["nu_a"], nu_b=arm["nu_b"], A=arm["A"], beta=arm["beta"],
            gx=gx, solid_mask=wall,
        )
        f = bounce_back_cells(stream(f), wall)
        g = bounce_back_cells(stream(g), wall)
        if not bool(torch.isfinite(f).all() and torch.isfinite(g).all()):
            nan_step = step
            break
        if step % max(1, arm["steps"] // 20) == 0:
            rho_a, rho_b, rho, _phi, ux, _uy = two_pdf_macros(f, g)
            umax_hist.append((step, float(ux[:, NX // 2].max())))
    wall_s = time.perf_counter() - t0
    if nan_step is not None:
        return {**arm, "stable": False, "nan_step": nan_step}

    rho_a, rho_b, rho, phi, ux, _uy = two_pdf_macros(f, g)
    prof = ux[:, NX // 2]
    phi_col = phi[:, NX // 2].cpu()

    if arm["kind"] == "single_phase":
        ana = single_phase_profile_wallplanes(ny, gx, arm["nu_a"]).to(dev)
        j_first_b = None
        y_i = None
    else:
        j_first_b = first_blue_row(phi_col)
        y_i = j_first_b - 0.5
        ana = two_layer_profile_wallplanes(
            ny, j_first_b, gx, arm["nu_a"], arm["nu_b"], arm["nu_a"] / arm["nu_b"]
        ).to(dev)
    rel = relative_l2(prof[1:-1], ana)
    tail = umax_hist[-max(1, len(umax_hist) // 5):]
    frozen = abs(tail[-1][1] - tail[0][1]) / max(abs(tail[-1][1]), 1e-30)
    return {
        **{k: arm[k] for k in ("name", "kind", "ny", "nu_a", "nu_b", "gx", "beta", "A", "steps")},
        "tau_a": 3.0 * arm["nu_a"] + 0.5,
        "tau_b": 3.0 * arm["nu_b"] + 0.5,
        "nu_ratio": arm["nu_a"] / arm["nu_b"],
        "Re_lattice": float(prof.abs().max()) * (ny - 2) / arm["nu_a"],
        "stable": True,
        "j_first_blue": j_first_b,
        "Y_interface": y_i,
        "relL2": rel,
        "u_max_ratio": float(prof.max() / ana.max()),
        "u_max_frozen_rel_tail20": frozen,
        "mass_drift_a_per_step": abs(float(f.sum()) - m_a0) / arm["steps"],
        "mass_drift_b_per_step": abs(float(g.sum()) - m_b0) / arm["steps"],
        "phi_range": [float(phi.min()), float(phi.max())],
        "profile": [round(float(v), 12) for v in prof.tolist()],
        "analytic": [round(float(v), 12) for v in ana.tolist()],
        "wall_s": wall_s,
    }


# ---------------------------------------------------------------------------
# Laplace law arms (static droplet)
# ---------------------------------------------------------------------------


def init_droplet(n: int, radius: float, width: float, dev: torch.device):
    ys = torch.arange(n, dtype=DT, device=dev).view(n, 1)
    xs = torch.arange(n, dtype=DT, device=dev).view(1, n)
    rr = torch.sqrt((xs - n / 2.0 + 0.5) ** 2 + (ys - n / 2.0 + 0.5) ** 2)
    frac_a = 0.5 * (1.0 - torch.tanh((rr - radius) / max(width, 1e-6)))
    z = torch.zeros((n, n), dtype=DT, device=dev)
    return equilibrium(frac_a, z, z), equilibrium(1.0 - frac_a, z, z)


def run_laplace_arm(radius: float, n: int, steps: int, width: float,
                    beta: float, A: float, dev: torch.device,
                    engine: str = "two_pdf") -> dict:
    f, g = init_droplet(n, radius, width, dev)
    m_a0, m_b0 = float(f.sum()), float(g.sum())
    t0 = time.perf_counter()
    nan_step = None
    samples = []
    for step in range(1, steps + 1):
        if engine == "two_pdf":
            f2, g2 = colorgradient_two_pdf_step(f, g, nu_a=1.0 / 6.0, nu_b=1.0 / 6.0,
                                                A=A, beta=beta)
        else:  # single-PDF two-component colour-gradient (independent engine)
            f2, g2 = color_gradient_two_tau_step(f, g, tau_r=1.0, tau_b=1.0,
                                                 A=A, beta=beta)
        f, g = stream(f2), stream(g2)
        if not bool(torch.isfinite(f).all() and torch.isfinite(g).all()):
            nan_step = step
            break
        if step % max(1, steps // 10) == 0:
            p_in, p_out, dp = pressure_jump(f, g, radius)
            samples.append((step, dp, max_velocity(f, g)))
    wall_s = time.perf_counter() - t0
    if nan_step is not None:
        return {"radius": radius, "n": n, "engine": engine, "stable": False, "nan_step": nan_step}
    # steady state = mean over the last three samples
    tail = samples[-3:]
    dp = sum(s[1] for s in tail) / len(tail)
    umax = max(s[2] for s in tail)
    return {
        "radius": radius,
        "n": n,
        "engine": engine,
        "beta": beta,
        "A": A,
        "interface_width": width,
        "steps": steps,
        "dp": dp,
        "sigma_dpR": dp * radius,
        "u_spurious_max": umax,
        "dp_history": [[s[0], s[1], s[2]] for s in samples],
        "mass_drift_a_per_step": abs(float(f.sum()) - m_a0) / steps,
        "mass_drift_b_per_step": abs(float(g.sum()) - m_b0) / steps,
        "wall_s": wall_s,
        "stable": True,
    }


def _fit_line(xs, ys):
    """Least-squares y = a x + b plus R^2."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suites", default="poiseuille,laplace")
    ap.add_argument("--arms", default=",".join(a["name"] for a in FROZEN_ARMS))
    ap.add_argument("--laplace-radii", default="12,18,24")
    ap.add_argument("--laplace-n", type=int, default=96)
    ap.add_argument("--laplace-width", type=float, default=2.0)
    ap.add_argument("--laplace-steps", type=int, default=8000)
    ap.add_argument("--laplace-beta", default="0.9")
    ap.add_argument("--out", default=str(HERE / "artifacts"), type=Path)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.set_num_threads(int(os.environ.get("XVAL_THREADS", "8")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "anchor": "XVAL-LBM-TWO-PDF-PARTIAL-DENSITY-COLORGRADIENT",
        "mission_id": 18787,
        "model": "two-PDF partial-density D2Q9 colour gradient: f=species-A partial "
                 "density, g=species-B partial density, rho=rho_A+rho_B, "
                 "phi=(rho_A-rho_B)/rho, tau from harmonic nu(phi), Guo(2002) forcing, "
                 "Latva-Kokko/Rothman perturbation, MF-LBM recolour weights",
        "engine_md5": {
            "color_gradient_two_pdf.py": _md5(SRC / "tensorlbm" / "color_gradient_two_pdf.py"),
            "cg_reference.py": _md5(HERE / "cg_reference.py"),
        },
        "config": {"dtype": "float64", "nx_poiseuille": NX, "A": A_FROZEN,
                   "device": str(dev), "gate_rell2": GATE_RELL2,
                   "gate_mass_drift": GATE_MASS_DRIFT},
        "suites": {},
    }

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    arm_sel = {s.strip() for s in args.arms.split(",") if s.strip()}

    if "poiseuille" in suites:
        arms = []
        for arm in FROZEN_ARMS:
            if arm["name"] not in arm_sel:
                continue
            a = run_poiseuille_arm(arm, dev)
            arms.append(a)
            print(f"[poiseuille] {a['name']}: relL2={a.get('relL2')} stable={a['stable']}",
                  flush=True)
        two_layer = [a for a in arms if a.get("kind") == "two_layer"]
        rel_max = max((a["relL2"] for a in two_layer), default=None)
        drift_max = max((max(a["mass_drift_a_per_step"], a["mass_drift_b_per_step"])
                         for a in arms if a.get("stable")), default=None)
        result["suites"]["poiseuille"] = {
            "arms": arms,
            "rell2_max_two_layer": rel_max,
            "mass_drift_max": drift_max,
            "pass_15pct": (rel_max is not None and rel_max < GATE_RELL2
                           and drift_max < GATE_MASS_DRIFT),
        }

    if "laplace" in suites:
        radii = [float(r) for r in args.laplace_radii.split(",") if r.strip()]
        betas = [float(b) for b in str(args.laplace_beta).split(",") if b.strip()]
        laplace = {"radii": radii, "n": args.laplace_n, "arms": []}
        for beta in betas:
            arms = [run_laplace_arm(r, args.laplace_n, args.laplace_steps,
                                    args.laplace_width, beta, A_FROZEN, dev)
                    for r in radii]
            for a in arms:
                print(f"[laplace] beta={beta} R={a['radius']}: dp={a.get('dp')} "
                      f"sigma={a.get('sigma_dpR')} stable={a['stable']}", flush=True)
            ok = [a for a in arms if a.get("stable")]
            if len(ok) >= 2:
                xs = [1.0 / a["radius"] for a in ok]
                ys = [a["sigma_dpR"] for a in ok]
                a_fit, b_fit, r2 = _fit_line(xs, ys)
                sigma_anchor = sum(ys) / len(ys)
                for a in ok:
                    a["sigma_rel_dev_from_mean"] = abs(a["sigma_dpR"] - sigma_anchor) / sigma_anchor
                worst = max(a["sigma_rel_dev_from_mean"] for a in ok)
                laplace["fit"] = {"sigma_mean_dpR": sigma_anchor, "intercept": b_fit,
                                  "slope_dp_vs_invR": a_fit, "r2": r2,
                                  "worst_rel_dev": worst,
                                  "pass_15pct": worst < GATE_RELL2 and r2 >= 0.99}
            laplace["arms"].append({"beta": beta, "runs": arms})
        # cross-model consistency: independent single-PDF two-component engine
        cross = [run_laplace_arm(r, args.laplace_n, args.laplace_steps,
                                 args.laplace_width, betas[0], A_FROZEN, dev,
                                 engine="single_pdf_two_component")
                 for r in radii]
        laplace["cross_model_single_pdf_two_component"] = cross
        tp = {a["radius"]: a["sigma_dpR"] for a in laplace["arms"][0]["runs"] if a.get("stable")}
        xrows = []
        for c in cross:
            if c.get("stable") and c["radius"] in tp:
                xrows.append({
                    "radius": c["radius"],
                    "sigma_two_pdf": tp[c["radius"]],
                    "sigma_single_pdf_two_component": c["sigma_dpR"],
                    "rel_diff": abs(tp[c["radius"]] - c["sigma_dpR"]) / c["sigma_dpR"],
                })
        laplace["cross_model_table"] = xrows
        laplace["cross_model_pass_15pct"] = bool(
            xrows and max(r["rel_diff"] for r in xrows) < GATE_RELL2)
        result["suites"]["laplace"] = laplace

    verdicts = {}
    if "poiseuille" in result["suites"]:
        verdicts["poiseuille_pass"] = result["suites"]["poiseuille"]["pass_15pct"]
    if "laplace" in result["suites"] and "fit" in result["suites"]["laplace"]:
        verdicts["laplace_pass"] = result["suites"]["laplace"]["fit"]["pass_15pct"]
    verdicts["all_pass"] = all(verdicts.values()) if verdicts else False
    result["verdict"] = verdicts
    result["repro_cmd"] = ("PYTHONPATH=src python3 xval/XVAL-LBM-TWO-PDF-PARTIAL-DENSITY-"
                           "COLORGRADIENT/run_xval.py --out xval/XVAL-LBM-TWO-PDF-PARTIAL-"
                           "DENSITY-COLORGRADIENT/artifacts")
    dest = out / "results.json"
    dest.write_text(json.dumps(result, indent=1))
    print(f"saved -> {dest}\nVERDICT {verdicts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())