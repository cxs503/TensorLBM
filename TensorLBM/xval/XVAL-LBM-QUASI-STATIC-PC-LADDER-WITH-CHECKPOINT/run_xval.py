#!/usr/bin/env python3
"""XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT -- runner.

Cross-validates this repository's implementation of the OSS practice
``je-santos/MPLBM-UT :: quasi-static-pc-ladder-with-checkpoint``
(``src/tensorlbm/pc_ladder.py``) against analytic anchors:

E0  measurement chain: static droplet curvature (kappa = 1/R) and Laplace pressure
E1  quasi-static ladder on a tapered channel with per-rung checkpoint
    -> rung-wise Young-Laplace residual sigma(kappa_lead - kappa_trail) vs drho*g*L
E2  checkpoint completeness: resume from rung k reproduces the continuous ladder
E3  control: single-shot (whole range in ONE rung) vs the ladder
E4  sensitivity: rung count and convergence tolerance

Writes ``results.json`` and two figures; every number in the report is read back
from ``results.json`` (no hand copy).
"""

from __future__ import annotations

import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorlbm import cac_lbm2d as cac
from tensorlbm import pc_ladder as P

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "artifacts")
os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, "ckpt")
os.makedirs(CKPT, exist_ok=True)

CFG = dict(
    rho_l=1.0,
    rho_g=0.1,
    mu_l=0.2,
    mu_g=0.02,
    sigma=0.05,
    width=3.0,
    mobility=0.05,
    mu_law="harmonic",
    p0=0.01,
    gravity=(0.0, 0.0),
    gravity_mode="rho",
    axes="pp",
)
SIGMA = CFG["sigma"]
DRHO = CFG["rho_l"] - CFG["rho_g"]
THETA = 60.0
COS_T = math.cos(math.radians(THETA))
NY, NX = 44, 128
R_WIDE, R_MIN = 20.0, 8.0
THROAT_C, THROAT_A = 78.0, 16.0
PLUG = (26.0, 56.0)
LP = PLUG[1] - PLUG[0]

res: dict = {
    "anchor": "XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT",
    "oss": "je-santos/MPLBM-UT",
    "tag": "quasi-static-pc-ladder-with-checkpoint",
    "config": {**CFG, "theta_deg": THETA, "grid": [NY, NX], "r_wide": R_WIDE,
               "r_min": R_MIN, "throat": [THROAT_C, THROAT_A], "plug": list(PLUG)},
    "cases": {},
}


def build(zero_net=True):
    shape = (NY, NX)
    mask, _ = P.tapered_channel(
        NY, NX, wall_t=2, r_wide=R_WIDE, r_min=R_MIN,
        throat_center=THROAT_C, throat_half_length=THROAT_A,
    )
    sim = cac.CAC2D(
        shape, wall_mask=mask, wall_phi_policy="contact",
        contact_angle_deg=THETA, wall_contact_zero_net=zero_net, **CFG
    )
    sim.initialize(P.plug_phi(shape, width=CFG["width"], x_lo=PLUG[0], x_hi=PLUG[1]))
    return sim


# --------------------------------------------------------------------- E0
def e0_measurement():
    shape = (64, 64)
    R0 = 20.0
    sim = cac.CAC2D(shape, **CFG)
    sim.initialize(cac.init_phi_droplet(shape, R0, CFG["width"], center=(31.5, 31.5)))
    t0 = time.time()
    cac.run(sim, 400, sample_every=0)
    m = cac.macro(sim)
    phi, p = m["phi"], m["p"]
    kap = P.curvature_field(phi)
    band = (phi > 0.3) & (phi < 0.7)
    kappa = float(kap[band].mean())
    Req = math.sqrt(float((phi > 0.5).sum().item()) / math.pi)
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    rr = ((yy - 31.5) ** 2 + (xx - 31.5) ** 2).sqrt()
    dp = float(p[rr < Req - 4].mean() - p[rr > Req + 8].mean())
    rec = {
        "sigma_input": SIGMA,
        "radius_init": R0,
        "radius_equiv": Req,
        "kappa_band": kappa,
        "kappa_from_Req": 1.0 / Req,
        "kappa_rel_err_vs_Req": abs(kappa - 1.0 / Req) / (1.0 / Req),
        "dp_field": dp,
        "sigma_kappa": SIGMA * kappa,
        "sigma_effective_from_dp": dp / kappa,
        "laplace_relerr_field": abs(dp - SIGMA * kappa) / (SIGMA * kappa),
        "steps": 400,
        "wall_time_s": time.time() - t0,
    }
    res["cases"]["E0_measurement"] = rec
    print("[E0]", json.dumps(rec), flush=True)
    return rec


# --------------------------------------------------------------------- E1/E2
def e1_ladder(num_pc_steps=6, conv=1e-4, it_conv=50, it_max=200, tag="main"):
    pressures = P.pc_stair_schedule(
        sigma=SIGMA, cos_theta=COS_T, r_min=R_MIN, num_pc_steps=num_pc_steps, dim=2, span="full"
    )
    params = P.LadderParams(conv=conv, it_conv=it_conv, it_max=it_max, min_iters=it_conv * 2)
    ck = os.path.join(CKPT, tag)
    logs: list[str] = []
    t0 = time.time()
    recs = P.run_ladder(
        lambda: build(),
        pressures=pressures,
        delta_rho=DRHO,
        plug_length=LP,
        params=params,
        checkpoint_dir=ck,
        log=logs.append,
    )
    rows = []
    for r in recs:
        rl = P.young_laplace_residual(
            sim_sigma=SIGMA,
            kappa_lead=r.meas.get("kappa_lead", float("nan")),
            kappa_trail=r.meas.get("kappa_trail", float("nan")),
            delta_rho=DRHO,
            g=r.g,
            plug_length=LP,
        )
        row = r.as_dict()
        row["laplace"] = rl
        rows.append(row)
    out = {
        "num_pc_steps": num_pc_steps,
        "pressures": pressures,
        "conv": conv,
        "it_conv": it_conv,
        "it_max": it_max,
        "rows": rows,
        "log": logs,
        "wall_time_s": time.time() - t0,
        "checkpoint_dir": ck,
    }
    print(f"[E1/{tag}] " + " | ".join(
        f"r{r['rung']} p={r['pressure']:.2e} steps={r['steps']} conv={r['converged']} "
        f"xl={r['meas'].get('x_lead')} relerr={r['laplace']['relerr']:.3f}" for r in rows), flush=True)
    return out


def e2_resume(base: dict):
    """Resume the ladder from rung 2 and compare with the continuous rung-N state."""
    rows = base["rows"]
    last = max(r["rung"] for r in rows)
    if last < 4:
        return {"skipped": "ladder too short"}
    k_resume = 2
    ck = os.path.join(base["checkpoint_dir"], f"rung_{k_resume:03d}")
    # continuous end state: load the last rung checkpoint
    simA = build()
    P.load_ladder_state(os.path.join(base["checkpoint_dir"], f"rung_{last:03d}"), simA)
    phiA = simA.f.sum(dim=0).clone()
    # resumed path
    params = P.LadderParams(conv=base["conv"], it_conv=base["it_conv"], it_max=base["it_max"],
                            min_iters=base["it_conv"] * 2)
    recs = P.run_ladder(
        build,
        pressures=base["pressures"],
        delta_rho=DRHO,
        plug_length=LP,
        params=params,
        checkpoint_dir=None,
        resume_from=ck,
        log=None,
    )
    simB = build()
    for r in recs:
        pass
    # re-run deterministically from the same resume point to capture the field
    simB = build()
    P.load_ladder_state(ck, simB)
    for k in range(k_resume + 1, last + 1):
        g = base["pressures"][k - 1] / (DRHO * LP)
        P.settle(simB, g=g, params=params)
    phiB = simB.f.sum(dim=0).clone()
    diff = (phiA - phiB).abs()
    out = {
        "resume_rung": k_resume,
        "final_rung": last,
        "max_abs_dphi": float(diff.max().item()),
        "mean_abs_dphi": float(diff.mean().item()),
        "rungs_resumed": [r["rung"] for r in recs],
        "resume_reproduced": bool(float(diff.max().item()) < 1e-6),
    }
    print("[E2]", json.dumps(out), flush=True)
    return out


# --------------------------------------------------------------------- E3
def e3_single_shot(num_rungs: int):
    """Same total pressure in ONE rung (no ladder, no per-rung settling)."""
    pressures = P.pc_stair_schedule(sigma=SIGMA, cos_theta=COS_T, r_min=R_MIN,
                                    num_pc_steps=num_rungs, dim=2, span="full")
    total = pressures[-1]
    params = P.LadderParams(conv=1e-4, it_conv=50, it_max=50 * num_rungs, min_iters=100)
    recs = P.run_ladder(build, pressures=[total], delta_rho=DRHO, plug_length=LP,
                        params=params, checkpoint_dir=None)
    last = recs[-1]
    rl = P.young_laplace_residual(
        sim_sigma=SIGMA, kappa_lead=last.meas.get("kappa_lead", float("nan")),
        kappa_trail=last.meas.get("kappa_trail", float("nan")),
        delta_rho=DRHO, g=last.g, plug_length=LP)
    out = {"total_pressure": total, "steps_budget": 50 * num_rungs,
           "rung": last.as_dict(), "laplace": rl}
    print("[E3]", json.dumps({k: out[k] for k in ("total_pressure", "laplace")}), flush=True)
    return out


def figures(main, e0):
    rows = main["rows"]
    rungs = [r["rung"] for r in rows]
    xl = [r["meas"].get("x_lead", float("nan")) for r in rows]
    kap_l = [r["meas"].get("kappa_lead", float("nan")) for r in rows]
    kap_t = [r["meas"].get("kappa_trail", float("nan")) for r in rows]
    lifts = [r["laplace"]["sigma_dkappa"] for r in rows]
    drives = [r["laplace"]["drho_g_L"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(rungs, xl, "o-")
    ax[0].set_xlabel("rung"); ax[0].set_ylabel("leading meniscus x_lead [lu]")
    ax[0].set_title("quasi-static front position")
    ax[1].plot(rungs, kap_l, "s-", label=r"$\kappa_{lead}$")
    ax[1].plot(rungs, kap_t, "^-", label=r"$\kappa_{trail}$")
    ax[1].plot(rungs, [abs(d) / SIGMA for d in drives], "k--", label=r"$\Delta\rho g L/\sigma$")
    ax[1].set_xlabel("rung"); ax[1].set_ylabel("curvature [1/lu]"); ax[1].legend(fontsize=7)
    ax[1].set_title("measured curvature vs imposed")
    ax[2].bar(rungs, [r["laplace"]["relerr"] for r in rows])
    ax[2].axhline(0.15, color="r", ls="--", label="15% gate")
    ax[2].set_xlabel("rung"); ax[2].set_ylabel("Young-Laplace rel. err"); ax[2].legend(fontsize=7)
    ax[2].set_title("rung-wise residual")
    fig.tight_layout()
    f1 = os.path.join(OUT, "fig_ladder.png")
    fig.savefig(f1, dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([e0["kappa_from_Req"]], [e0["kappa_band"]], "o")
    lo = 0.9 * e0["kappa_from_Req"]
    ax.plot([lo, e0["kappa_from_Req"] * 1.1], [lo, e0["kappa_from_Req"] * 1.1], "k--", lw=0.8)
    ax.set_xlabel(r"$1/R_{eq}$"); ax.set_ylabel(r"measured $\kappa$")
    ax.set_title("E0 curvature estimator")
    fig.tight_layout()
    f2 = os.path.join(OUT, "fig_e0.png")
    fig.savefig(f2, dpi=130)
    plt.close(fig)
    return [f1, f2]


def main():
    t0 = time.time()
    res["cases"]["E0_measurement"] = e0_measurement()
    main_ladder = e1_ladder(num_pc_steps=6, conv=1e-4, it_conv=50, it_max=200, tag="main")
    res["cases"]["E1_ladder"] = main_ladder
    try:
        res["cases"]["E2_resume"] = e2_resume(main_ladder)
    except Exception as exc:  # honest partial failure
        res["cases"]["E2_resume"] = {"error": f"{type(exc).__name__}: {exc}"}
        print("[E2] FAILED", exc, flush=True)
    try:
        res["cases"]["E3_single_shot"] = e3_single_shot(6)
    except Exception as exc:
        res["cases"]["E3_single_shot"] = {"error": f"{type(exc).__name__}: {exc}"}
        print("[E3] FAILED", exc, flush=True)
    # E4: two sensitivity axes (rung count, convergence tolerance)
    sens = {}
    for tag, kw in (("rungs4", dict(num_pc_steps=4)), ("conv1e-2", dict(num_pc_steps=6, conv=1e-2))):
        try:
            sens[tag] = e1_ladder(**kw, it_conv=50, it_max=200, tag=tag)
        except Exception as exc:
            sens[tag] = {"error": f"{type(exc).__name__}: {exc}"}
    res["cases"]["E4_sensitivity"] = sens
    figs = figures(main_ladder, res["cases"]["E0_measurement"])
    res["figures"] = figs
    res["wall_time_s"] = time.time() - t0
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("[done]", res["wall_time_s"], "s ->", os.path.join(OUT, "results.json"), flush=True)


if __name__ == "__main__":
    main()