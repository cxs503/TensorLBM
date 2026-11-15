"""R2 audit: is the Shan-Chen Laplace calibration conditioned, and does a
radius-free (Korteweg) readout confirm the same surface tension?

Motivation (why this round exists)
----------------------------------
Three earlier rounds calibrated ``Delta p * R = sigma`` for the single-component
Shan-Chen model on a circular drop (``benchmarks/verified/laplace_droplet``,
``xval/.../run_xval.py``, ``xval/.../verify_r1``).  They agree on the *fit*
(per-radius spread < 1 %) but the reported surface tension differs between
rounds by a factor ~2.7 when the coupling is moved from ``G_eff = -5.0`` to
``-4.7``.  Two things were never quantified:

1. **Conditioning.**  The SC mechanical pressure is a difference of two large
   terms,
       ``p = rho/3 + G_eff psi^2 / 6``,  ``psi = 1 - exp(-rho)``,
   so ``Delta p`` across the interface is a small residual of two O(0.6) numbers
   (``kappa = (|A|+|B|)/|Delta p|`` is several hundred).  A relative error
   ``eps`` in the measured bulk densities is amplified into ``~kappa*eps`` in
   ``sigma``.  Any cross-round comparison of ``sigma`` without that number is
   uninterpretable.
2. **Independence.**  ``p_eos`` *is* the model assumption, so ``Delta p * R``
   cannot test the capillary stress by itself.  The Korteweg readout
   ``sigma_K = |K| * Integral (d psi / dn)^2 dn`` (module
   :mod:`tensorlbm.sc_surface_tension`, eq. 6) contains no radius, so it tests
   the curvature relation on an independent footing.  An earlier round declared
   it FAIL (18-60 % high) using fields that were not converged (init not at
   coexistence, 4000 steps).

This script therefore measures, on the *same* converged protocol as the
verified benchmark (velocity-shift forcing, mass-conserving coexistence init):

* closure ``Delta p``, ``R_eq``, ``sigma = Delta p * R_eq`` per radius;
* the Korteweg ``sigma_K`` from the *converged* density field, with **both**
  prefactors that appear in the module (code: ``|G_eff| cs^2 kappa3``;
  docstring eq. 6: ``|G_eff| kappa3``) so the ambiguity is resolved by data;
* the condition number and the analytic sensitivity of ``sigma`` to the bulk
  densities and to ``G_eff``;
* one off-benchmark coupling (``G_lib = 4.7``, i.e. ``G_eff = -4.7``) to show
  what conditioning does to the same measurement.

Everything is written to JSON; no number is retyped by hand.

Run:  python xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION/audit_r2/run_audit.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, psi_exp  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

CS2 = 1.0 / 3.0
TAU = 1.0
RHO_L, RHO_V = 1.957, 0.1596      # discrete coexistence used by the verified benchmark
W_INT = 4.0
KAPPA3 = 1.0 / 6.0                # a3/a1 of the D2Q9 shell stencil (module's kappa3)
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# closure / conditioning
# ---------------------------------------------------------------------------
def psi(rho: float) -> float:
    return 1.0 - math.exp(-rho)


def p_eos(rho: float, g_eff: float) -> float:
    """SC closure pressure (verified against the flat-interface gate)."""
    ps = psi(rho)
    return rho * CS2 + 0.5 * g_eff * CS2 * ps * ps


def conditioning(rho_in: float, rho_out: float, g_eff: float) -> dict:
    """Decomposition ``Delta p = A + B`` and its condition number.

    ``A = cs^2 (rho_in - rho_out)``, ``B = (g_eff cs^2/2)(psi_in^2 - psi_out^2)``
    (``B < 0`` for attraction), ``Delta p = A + B``.  Because the two bulk phases
    are close in ``p`` (``Delta p / <p>`` is a few percent), a relative error
    ``eps`` in a bulk density becomes ``eps * kappa`` in ``Delta p``.
    """
    a = CS2 * (rho_in - rho_out)
    b = 0.5 * g_eff * CS2 * (psi(rho_in) ** 2 - psi(rho_out) ** 2)
    dp = a + b
    # d(Delta p)/d(rho_in) and d/d(rho_out): d psi^2/d rho = 2 psi exp(-rho)
    ddp_drho_in = CS2 + g_eff * CS2 * psi(rho_in) * math.exp(-rho_in)
    ddp_drho_out = -CS2 - g_eff * CS2 * psi(rho_out) * math.exp(-rho_out)
    pbar = 0.5 * (p_eos(rho_in, g_eff) + p_eos(rho_out, g_eff))
    return {
        "A_density_term": a,
        "B_interaction_term": b,
        "dp_from_closure": dp,
        "condition_number_kappa": (abs(a) + abs(b)) / abs(dp) if dp else float("inf"),
        "dp_over_p_mean": abs(dp) / pbar if pbar else float("inf"),
        "ddp_drho_in": ddp_drho_in,
        "ddp_drho_out": ddp_drho_out,
        # sigma = dp * R  ->  relative sigma error for a 1e-3 absolute density error
        "dsigma_rel_per_1e-3_rho_in": abs(ddp_drho_in) * 1e-3 / abs(dp),
        "dsigma_rel_per_1e-3_rho_out": abs(ddp_drho_out) * 1e-3 / abs(dp),
    }


# ---------------------------------------------------------------------------
# droplet protocol (identical to benchmarks/verified/laplace_droplet)
# ---------------------------------------------------------------------------
def init_rho(L: int, R: float, device) -> torch.Tensor:
    ys = torch.arange(L, dtype=DTYPE, device=device)
    xs = torch.arange(L, dtype=DTYPE, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    rho = RHO_V + 0.5 * (RHO_L - RHO_V) * (1.0 + torch.tanh((R - r) / W_INT))
    return rho.clamp(min=1e-3), r


def measure(f: torch.Tensor, R_guess: float, rr: torch.Tensor, g_lib: float,
            g_eff: float) -> dict:
    rho, ux, uy = macroscopic(f)
    ps = psi_exp(rho)
    p = rho * CS2 + 0.5 * g_eff * CS2 * ps * ps
    r_eq = R_guess
    for _ in range(4):
        inside = rr <= r_eq * 0.5
        outside = rr >= r_eq * 1.5
        rho_in = float(rho[inside].mean())
        rho_out = float(rho[outside].mean())
        mid = 0.5 * (rho_in + rho_out)
        r_eq_new = math.sqrt(float((rho > mid).sum()) / math.pi)
        if abs(r_eq_new - r_eq) < 1e-5:
            r_eq = r_eq_new
            break
        r_eq = r_eq_new
    inside = rr <= r_eq * 0.5
    outside = rr >= r_eq * 1.5
    p_in = float(p[inside].mean())
    p_out = float(p[outside].mean())
    rho_in = float(rho[inside].mean())
    rho_out = float(rho[outside].mean())
    return {
        "R_eq": r_eq,
        "p_in": p_in,
        "p_out": p_out,
        "dp": p_in - p_out,
        "rho_in": rho_in,
        "rho_out": rho_out,
        "max_u": float(torch.sqrt(ux * ux + uy * uy).max()),
        "mass": float(rho.sum()),
        "mass_drift_rel": abs(float(rho.sum()) - float(_MASS0[0])) / float(_MASS0[0]),
    }


_MASS0 = [1.0]


def run_case(R: float, L: int, g_lib: float, max_steps: int, min_steps: int,
             sample_every: int, conv_dp: float = 2e-3, conv_r: float = 2e-3) -> dict:
    device = torch.device("cpu")
    g_eff = -g_lib
    rho0, rr = init_rho(L, R, device)
    _MASS0[0] = float(rho0.sum())
    f = equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    t0 = time.perf_counter()
    hist: list[dict] = []
    converged = False
    step = 0
    for step in range(1, max_steps + 1):
        f = stream(collide_sc_single_component(f, G=g_lib, tau=TAU, psi_fn=psi_exp))
        if step % sample_every == 0:
            m = measure(f, R, rr, g_lib, g_eff)
            m["step"] = step
            hist.append(m)
            if len(hist) >= 2 and step >= min_steps:
                d_dp = abs(hist[-1]["dp"] - hist[-2]["dp"]) / abs(hist[-1]["dp"])
                d_r = abs(hist[-1]["R_eq"] - hist[-2]["R_eq"]) / hist[-1]["R_eq"]
                if d_dp < conv_dp and d_r < conv_r:
                    converged = True
                    break

    tail = hist[-3:]
    out = {k: sum(t[k] for t in tail) / len(tail)
           for k in ("R_eq", "p_in", "p_out", "dp", "rho_in", "rho_out", "max_u")}
    out["step"] = int(step)
    out["L"] = L
    out["R_init"] = R
    out["G_lib"] = g_lib
    out["G_eff"] = g_eff
    out["converged"] = converged
    out["dp_std"] = float(max(t["dp"] for t in tail) - min(t["dp"] for t in tail))
    out["mass_drift_rel"] = tail[-1]["mass_drift_rel"]
    out["sigma_laplace"] = out["dp"] * out["R_eq"]
    out["wallclock_s"] = time.perf_counter() - t0
    out["conditioning"] = conditioning(out["rho_in"], out["rho_out"], g_eff)

    # ---- radius-free Korteweg readout on the converged field -------------
    rho_f = f.sum(dim=0)
    ps = psi_exp(rho_f)
    # (a) radial-mean profile integral, fine bins
    n_bins = 400
    rmax = 0.5 * L
    idx = (rr / rmax * n_bins).long().clamp(0, n_bins - 1)
    cnt = torch.bincount(idx.ravel(), minlength=n_bins).to(DTYPE)
    ssum = torch.bincount(idx.ravel(), weights=ps.ravel().to(DTYPE),
                          minlength=n_bins).to(DTYPE)
    ok = cnt > 4
    rc = (torch.arange(n_bins, dtype=DTYPE) + 0.5) * (rmax / n_bins)
    prof = (ssum / cnt.clamp(min=1))
    rc, prof = rc[ok], prof[ok]
    lo, hi = 0.4 * out["R_eq"], 2.0 * out["R_eq"]
    sel = (rc >= lo) & (rc <= hi)
    rc_s, pr_s = rc[sel], prof[sel]
    integ = float(((pr_s[1:] - pr_s[:-1]) / (rc_s[1:] - rc_s[:-1])).pow(2).mul(
        rc_s[1:] - rc_s[:-1]).sum())
    # (b) 2-D box integral of |grad psi|^2 divided by the interface length
    gx = 0.5 * (torch.roll(ps, -1, dims=1) - torch.roll(ps, 1, dims=1))
    gy = 0.5 * (torch.roll(ps, -1, dims=0) - torch.roll(ps, 1, dims=0))
    box = float((gx * gx + gy * gy).sum()) / (2.0 * math.pi * out["R_eq"])
    # 10-90 interface width from the same profile
    lvl_hi = float(prof.max()) - 0.1 * (float(prof.max()) - float(prof.min()))
    lvl_lo = float(prof.min()) + 0.1 * (float(prof.max()) - float(prof.min()))

    def crossing(level: float) -> float:
        s = pr_s - level
        for k in range(len(s) - 1):
            if float(s[k]) * float(s[k + 1]) <= 0.0:
                t = float(s[k]) / float(s[k] - s[k + 1])
                return float(rc_s[k] + t * (rc_s[k + 1] - rc_s[k]))
        return float("nan")

    w90 = abs(crossing(lvl_lo) - crossing(lvl_hi))

    k_code = abs(g_eff * CS2 * KAPPA3)
    k_doc = abs(g_eff * KAPPA3)
    out["korteweg"] = {
        "integral_dpsi_dr_sq": integ,
        "integral_box_per_length": box,
        "K_code_absG_cs2_kappa3": k_code,
        "K_doc_absG_kappa3": k_doc,
        "sigma_K_code": k_code * integ,
        "sigma_K_doc": k_doc * integ,
        "sigma_K_code_box": k_code * box,
        "sigma_K_doc_box": k_doc * box,
        "interface_width_10_90": w90,
        "ratio_sigmaK_code_over_laplace": (k_code * integ) / out["sigma_laplace"],
        "ratio_sigmaK_doc_over_laplace": (k_doc * integ) / out["sigma_laplace"],
    }
    out["profile"] = {"r": rc_s.tolist(), "psi": pr_s.tolist()}
    s_la = out["sigma_laplace"]
    print(f"[case] G={g_lib:+.1f} R_init={R:4.0f} L={L:3d} steps={step:5d} "
          f"conv={converged} dp={out['dp']:.4e} R_eq={out['R_eq']:.3f} "
          f"sigma={s_la:.5f} sigma_K(cs2)={k_code * integ:.5f} "
          f"({100 * (k_code * integ / s_la - 1):+.1f}%) "
          f"kappa={out['conditioning']['condition_number_kappa']:.0f}", flush=True)
    return out


def fit_sigma(rows: list[dict]) -> dict:
    inv = [1.0 / r["R_eq"] for r in rows]
    dp = [r["dp"] for r in rows]
    num = sum(a * b for a, b in zip(inv, dp))
    den = sum(a * a for a in inv)
    sigma = num / den
    rel = [abs(d - sigma * i) / abs(sigma * i) for d, i in zip(dp, inv)]
    return {"sigma_fit": sigma, "rel_dev_max": max(rel),
            "rel_dev_per_radius": rel, "n_radii": len(rows)}


def figure(cases: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    main = [c for c in cases if c["G_lib"] == 5.0]
    inv = [1.0 / c["R_eq"] for c in main]
    sig = [c["sigma_laplace"] for c in main]
    ax[0].plot(inv, sig, "o-", label=r"$\Delta p\,R_{eq}$ (closure)")
    if len(main) >= 2:
        fit = fit_sigma(main)
        xs = [0.0, max(inv) * 1.08]
        ax[0].plot(xs, [fit["sigma_fit"] * x for x in xs], "k--",
                   label=rf"fit $\sigma$={fit['sigma_fit']:.5f}")
    kk = [c["korteweg"]["sigma_K_code"] for c in main]
    ax[0].axhline(sum(kk) / len(kk), color="tab:red", ls=":",
                  label=r"$\sigma_K$ (Korteweg, conv. field)")
    ax[0].set_xlabel(r"$1/R_{eq}$ [$1/\mathrm{lu}$]")
    ax[0].set_ylabel(r"$\Delta p\,R_{eq}$ [lu]")
    ax[0].set_title("Young-Laplace readouts (G=-5)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for c in main:
        r = c["profile"]["r"]
        ax[1].plot(r, c["profile"]["psi"], label=rf"$R_0$={c['R_init']:.0f}")
    ax[1].set_xlabel("r [lu]")
    ax[1].set_ylabel(r"$\psi(\rho)$")
    ax[1].set_title(r"converged $\psi$ profile (integrand of $\sigma_K$)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"[fig] {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=16000)
    ap.add_argument("--min-steps", type=int, default=3000)
    ap.add_argument("--sample-every", type=int, default=250)
    ap.add_argument("--radii", type=float, nargs="+", default=[12.0, 20.0, 30.0])
    ap.add_argument("--smoke", action="store_true", help="short non-converged probe")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if args.smoke:
        args.max_steps, args.min_steps, args.radii = 900, 900, [12.0]

    cases = []
    for R in args.radii:
        L = int(round(4 * R))
        L += L % 2
        cases.append(run_case(R, L, 5.0, args.max_steps, args.min_steps,
                              args.sample_every))
    # off-benchmark coupling: what does -4.7 do to the same measurement?
    cases.append(run_case(16.0, 64, 4.7, args.max_steps, args.min_steps,
                          args.sample_every))

    main_rows = [c for c in cases if c["G_lib"] == 5.0]
    off = [c for c in cases if c["G_lib"] != 5.0]
    fit = fit_sigma(main_rows) if len(main_rows) >= 2 else {}
    ref = 0.0561  # benchmarks/verified/laplace_droplet (2-D), documented value
    result = {
        "audit": "laplace_conditioning_and_korteweg_r2",
        "protocol": {
            "model": "SCMP, D2Q9, psi=1-exp(-rho)",
            "forcing": "velocity_shift (same as benchmarks/verified/laplace_droplet)",
            "tau": TAU, "dtype": "float64", "domain_rule": "L = 4 R_init",
            "init": "tanh drop at discrete coexistence 1.957/0.1596, width 4",
            "conv": "|d dp|/dp < 2e-3 and |d R|/R < 2e-3 over one sample window",
            "kappa3": KAPPA3,
        },
        "cases": cases,
        "laplace_fit": fit,
        "sigma_reference_verified_benchmark": ref,
        "fit_vs_reference_pct": 100 * (fit["sigma_fit"] / ref - 1) if fit else None,
        "korteweg_vs_laplace_pct": [
            {"R_init": c["R_init"],
             "code_prefactor_pct": 100 * (c["korteweg"]["ratio_sigmaK_code_over_laplace"] - 1),
             "doc_prefactor_pct": 100 * (c["korteweg"]["ratio_sigmaK_doc_over_laplace"] - 1)}
            for c in cases
        ],
        "off_benchmark_G": [
            {"G_lib": c["G_lib"], "R_eq": c["R_eq"], "dp": c["dp"],
             "sigma_laplace": c["sigma_laplace"],
             "kappa": c["conditioning"]["condition_number_kappa"],
             "sigma_K_code": c["korteweg"]["sigma_K_code"]}
            for c in off
        ],
    }
    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_r2.json").write_text(json.dumps(result, indent=1))
    figure(cases, out_dir / "fig_audit_r2.png")
    print(json.dumps({k: result[k] for k in
                      ("laplace_fit", "fit_vs_reference_pct", "korteweg_vs_laplace_pct",
                       "off_benchmark_G")}, indent=1))


if __name__ == "__main__":
    main()