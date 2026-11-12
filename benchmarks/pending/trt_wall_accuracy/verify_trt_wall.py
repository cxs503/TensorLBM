#!/usr/bin/env python3
"""TRT (D2Q9) wall-placement verification: body-force plane Poiseuille, exact reference.

Claim under test (module docstring of ``tensorlbm.solver.collide_trt``):
"Λ = 3/16 eliminates wall placement errors in Poiseuille flow (Ginzburg 2008)".

Reference (exact, White §3-2): u(y) = g/(2ν)(y−1/2)(H+1/2−y), Q = gH³/(12ν).
Observables: δ_eff (effective wall displacement from the flux, no profile fit) and
the profile L2 error against the nominal H.  Both schemes share one driver, one
IC (the nominal analytic profile, transient-free), one measurement operator.

Outputs: result.json, tables.md, figures/trt_wall_accuracy.png.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from tensorlbm.d2q9 import C, OPPOSITE, W, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, collide_trt, stream  # noqa: E402

G = 2.0e-6
NX = 4
STEPS = 3000
AVG = 200
TAUS = (0.55, 0.60, 0.70, 0.80, 0.933, 1.20, 1.90)
GRIDS = (8, 16, 32)
LAM = 3.0 / 16.0
FCOEF = (3.0 * G * W.double() * C[:, 0].double()).view(9, 1, 1)


def run(H: int, tau: float, scheme: str, lam: float = LAM, steps: int = STEPS) -> dict:
    ny = H + 2
    nu = (tau - 0.5) / 3.0
    y = torch.arange(1, H + 1, dtype=torch.float64)
    u0 = G / (2.0 * nu) * (y - 0.5) * (H + 0.5 - y)
    uu = torch.zeros(ny, NX, dtype=torch.float64)
    uu[1:H + 1, :] = u0.view(H, 1)
    f = equilibrium(torch.ones(ny, NX, dtype=torch.float64), uu, torch.zeros(ny, NX, dtype=torch.float64))
    t0 = time.time()
    for _ in range(steps):
        f = collide_bgk(f, tau) if scheme == "bgk" else collide_trt(f, tau, lam)
        f = f + FCOEF
        f[:, 0, :] = f[OPPOSITE, 1, :]
        f[:, H + 1, :] = f[OPPOSITE, H, :]
        f = stream(f)
    prof = torch.zeros(H, dtype=torch.float64)
    for _ in range(AVG):
        f = collide_bgk(f, tau) if scheme == "bgk" else collide_trt(f, tau, lam)
        f = f + FCOEF
        f[:, 0, :] = f[OPPOSITE, 1, :]
        f[:, H + 1, :] = f[OPPOSITE, H, :]
        f = stream(f)
        prof += macroscopic(f)[1][1:H + 1, :].mean(dim=1)
    prof /= AVG
    wall_s = time.time() - t0
    una = u0.numpy()
    unum = prof.numpy()
    Q = float(unum.sum())
    Qa = float(G * H ** 3 / (12.0 * nu))
    H_eff = (12.0 * nu * Q / G) ** (1.0 / 3.0)
    ic = int(np.argmax(una))
    return {
        "scheme": scheme, "H": H, "tau": tau, "nu": nu,
        "lambda_trt": lam if scheme == "trt" else None, "steps": steps, "wall_s": round(wall_s, 2),
        "Q": Q, "Q_ana": Qa, "Q_rel_err_pct": round((Q / Qa - 1.0) * 100.0, 4),
        "H_eff": round(H_eff, 5), "delta_eff": round(H_eff - H, 5),
        "delta_eff_pct_of_H": round((H_eff - H) / H * 100.0, 4),
        "u_center_num": round(float(unum[ic]), 9), "u_center_ana": round(float(una[ic]), 9),
        "u_center_rel_err_pct": round((unum[ic] / una[ic] - 1.0) * 100.0, 4),
        "profile_L2_rel_pct": round(float(np.linalg.norm(unum - una) / np.linalg.norm(una)) * 100.0, 4),
    }


def order(rows, key):
    out = []
    for a, b in zip(rows[:-1], rows[1:]):
        if a[key] > 0 and b[key] > 0:
            out.append({"H_from": a["H"], "H_to": b["H"],
                        "p": round(math.log(a[key] / b[key]) / math.log(b["H"] / a["H"]), 3)})
    return out


def main() -> int:
    out = Path(__file__).resolve().parent
    sweep = []
    for tau in TAUS:
        for scheme in ("bgk", "trt"):
            sweep.append(run(32, tau, scheme))
    conv = {s: [run(H, 0.80, s) for H in GRIDS] for s in ("bgk", "trt")}
    orders = {s: order(conv[s], "profile_L2_rel_pct") for s in conv}

    res = {
        "case": "trt_wall_accuracy",
        "question": "does TRT (Lambda=3/16) suppress the tau-dependent half-way bounce-back "
                    "wall displacement that BGK shows in plane Poiseuille?",
        "reference": {"u": "u(y)=g/(2nu)(y-1/2)(H+1/2-y)", "Q": "g H^3/(12 nu)",
                      "source": "White, Viscous Fluid Flow 3rd ed. §3-2; "
                                "Ginzburg 2008 Commun. Comput. Phys. 3(2) 427 (magic parameter Lambda)"},
        "setup": {"g": G, "nx_periodic": NX, "steps": STEPS, "avg_steps": AVG,
                  "ic": "analytic profile (transient-free init); steady check: Q plateau <1e-7 over 200 steps",
                  "wall": "pre-streaming half-way bounce-back", "dtype": "float64",
                  "note": "delta_eff>0 == effective wall outside the nominal wall plane"},
        "tau_sweep_H32": sweep,
        "grid_convergence_tau080": conv,
        "observed_order_profile_L2": orders,
        "extrap": "none",
    }
    (out / "result.json").write_text(json.dumps(res, indent=1))

    L = ["### τ sweep (H = 32, body-force plane Poiseuille)", "",
         "| τ | scheme | δ_eff (cell) | Q rel err % | u_c rel err % | profile L2 % |",
         "|---|---|---|---|---|---|"]
    for r in sweep:
        L.append(f"| {r['tau']:.3f} | {r['scheme']} | {r['delta_eff']:+.5f} | {r['Q_rel_err_pct']:+.4f} "
                 f"| {r['u_center_rel_err_pct']:+.4f} | {r['profile_L2_rel_pct']:.4f} |")
    L += ["", "### grid convergence (τ = 0.80)", "",
          "| H | scheme | δ_eff (cell) | profile L2 % | u_c rel err % |", "|---|---|---|---|---|"]
    for s in ("trt", "bgk"):
        for r in conv[s]:
            L.append(f"| {r['H']} | {r['scheme']} | {r['delta_eff']:+.5f} | {r['profile_L2_rel_pct']:.4f} "
                     f"| {r['u_center_rel_err_pct']:+.4f} |")
    L += ["", f"observed order (profile L2): TRT {orders['trt']} · BGK {orders['bgk']}"]
    (out / "tables.md").write_text("\n".join(L) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for s, mk in (("bgk", "o-"), ("trt", "s-")):
            xs = [r["tau"] for r in sweep if r["scheme"] == s]
            ys = [r["delta_eff"] for r in sweep if r["scheme"] == s]
            ax[0].plot(xs, ys, mk, label=s.upper())
        ax[0].axvline(0.5 + math.sqrt(3) / 4, ls=":", c="k", label="τ=1/2+√3/4")
        ax[0].set_xlabel("τ₊"); ax[0].set_ylabel("δ_eff (cell)")
        ax[0].set_title("wall displacement vs τ (H=32)"); ax[0].legend(); ax[0].grid(alpha=.3)
        for s, mk in (("bgk", "o-"), ("trt", "s-")):
            xs = [r["H"] for r in conv[s]]
            ys = [r["profile_L2_rel_pct"] for r in conv[s]]
            ax[1].loglog(xs, ys, mk, label=s.upper())
        ax[1].set_xlabel("H (fluid rows)"); ax[1].set_ylabel("profile L2 rel err (%)")
        ax[1].set_title("grid convergence (τ=0.80)"); ax[1].legend(); ax[1].grid(alpha=.3, which="both")
        fig.tight_layout()
        (out / "figures").mkdir(exist_ok=True)
        fig.savefig(out / "figures" / "trt_wall_accuracy.png", dpi=140)
    except Exception as exc:  # pragma: no cover
        print("figure skipped:", exc)
    print("wrote", out / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())