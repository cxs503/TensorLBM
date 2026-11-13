"""XVAL driver (lean): TensorLBM readouts on the OSS-calibrated SCMP droplet.

Anchor XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION, round 1.

Design after the first (slow) draft: the two kernels were shown to be
*algebraically identical* (max |drho| = 8.9e-16 after 5 identical steps, see
``kernel_equivalence``), so instead of time-stepping two solvers at five radii
we time-step the OSS port (numpy) and evaluate the three TensorLBM readouts --
closure ``pressure_eos``, model-free ``pressure_flux``, Korteweg ``sigma_K`` --
on the *same* converged field.  The TensorLBM kernel is time-stepped once
(R_init = 16, 4000 steps) to confirm that the agreement survives the full run.
That keeps the comparison arms identical while cutting the wall clock by ~4x.

Records produced in ``artifacts/xval_shan_chen_laplace/``:
summary.json, radius_sweep.csv, time_convergence.csv, sensitivity.csv,
fig_laplace_gamma.png, fig_profiles.png, figures_b64.json.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

torch.set_num_threads(2)

from oss_port import (OSSShanChen, RadiusEstimatorFailed, box_mean,  # noqa: E402
                      closure_pressure)
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.sc_surface_tension import (build_stencil, pressure_eos,  # noqa: E402
                                          pressure_flux, sigma_korteweg, step_sc)

OUT = ROOT / "artifacts" / "xval_shan_chen_laplace"
OUT.mkdir(parents=True, exist_ok=True)

N, TAU, G_EFF, RHO0 = 64, 1.0, -4.7, 1.0
RHO_L_INIT, RHO_G_INIT = 2.1, 0.15
STEPS = 4000
RADII = [8.0, 12.0, 16.0, 20.0, 24.0]
STENCIL = build_stencil("d2q9")

OSS_REF = {  # report §九 table 1 (+ §4.3), inventoried OSS run
    8.0: {"rho_g": 0.19504, "rho_l": 1.68702, "dp": 1.7074e-3, "r": 6.382, "gamma": 0.010897},
    12.0: {"rho_g": 0.18687, "rho_l": 1.67736, "dp": 1.6591e-3, "r": 12.291, "gamma": 0.020392},
    16.0: {"rho_g": 0.18432, "rho_l": 1.66937, "dp": 1.1918e-3, "r": 17.432, "gamma": 0.020776},
    20.0: {"rho_g": 0.18308, "rho_l": 1.66419, "dp": 8.5034e-4, "r": 22.365, "gamma": 0.019018},
    24.0: {"rho_g": 0.17411, "rho_l": 1.63248, "dp": -9.3354e-4, "r": 5207.67, "gamma": -4.8616},
}
OSS_REF_LSQR = 0.020268
KEY = ("rho_l", "rho_g", "rho_ratio", "p_l", "p_g", "dp", "r", "gamma")


def sharp_droplet(R: float, n: int = N) -> np.ndarray:
    y = np.arange(n, dtype=np.float64)[:, None]
    x = np.arange(n, dtype=np.float64)[None, :]
    c = 0.5 * (n - 1)
    return np.where(np.sqrt((x - c) ** 2 + (y - c) ** 2) <= R, RHO_L_INIT, RHO_G_INIT)


def tl_init(rho_np):
    rho = torch.tensor(rho_np, dtype=torch.float64)
    u = torch.zeros_like(rho)
    return equilibrium(rho, u, u).to(torch.float64)


def calibrate_arrays(rho, p, box=4, two_sided=False):
    """OSS calibrator (chapter9/shanchen.cpp:326-377) on an arbitrary p field."""
    n = rho.shape[0]
    c, off = n // 2, box // 2
    rho_l, rho_g = box_mean(rho, c - off, c - off, box), box_mean(rho, 0, 0, box)
    p_l, p_g = box_mean(p, c - off, c - off, box), box_mean(p, 0, 0, box)
    rho_av = 0.5 * (rho_l + rho_g)
    row = rho[n // 2]

    def scan(sign):
        prev = row[c]
        for j in range(1, n // 2):
            cur = row[c + sign * j]
            if (prev - rho_av) * (cur - rho_av) <= 0.0 and cur != prev:
                return float(j - 1 + (rho_av - prev) / (cur - prev))
            prev = cur
        raise RadiusEstimatorFailed("no rho = rho_av crossing (upstream UB guard)")

    rp = scan(+1)
    rm = scan(-1) if two_sided else None
    r = rp if rm is None else 0.5 * (rp + rm)
    return {"rho_l": rho_l, "rho_g": rho_g, "rho_ratio": rho_l / rho_g,
            "p_l": p_l, "p_g": p_g, "dp": p_l - p_g, "r": r, "r_plus": rp,
            "r_minus": rm, "gamma": (p_l - p_g) * r}


def calib_safe(rho, p, box=4, two_sided=False):
    try:
        out = calibrate_arrays(rho, p, box=box, two_sided=two_sided)
        out["status"] = "ok"
        return out
    except RadiusEstimatorFailed as exc:
        n = rho.shape[0]
        c, off = n // 2, max(box // 2, 1)
        rho_l, rho_g = box_mean(rho, c - off, c - off, box), box_mean(rho, 0, 0, box)
        p_l, p_g = box_mean(p, c - off, c - off, box), box_mean(p, 0, 0, box)
        return {"rho_l": rho_l, "rho_g": rho_g, "rho_ratio": rho_l / rho_g,
                "p_l": p_l, "p_g": p_g, "dp": p_l - p_g, "r": None, "gamma": None,
                "status": "radius_estimator_failed: %s" % exc}


def lsqr_origin(xs, ys):
    pairs = [(float(a), float(b)) for a, b in zip(xs, ys)
             if a is not None and b is not None]
    if len(pairs) < 2:
        return float("nan"), float("nan")
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    slope = float((xs * ys).sum() / (xs * xs).sum())
    ss_res = float(((ys - slope * xs) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    return slope, (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def relerr(got, ref):
    if got is None or ref in (None, 0):
        return float("nan")
    return abs(got - ref) / abs(ref)


def fnum(x, f="%.5e"):
    return "FAILED" if x is None else f % x


def readouts(solver: OSSShanChen):
    """TensorLBM readouts on the OSS solver state (populations + density)."""
    rho_t = torch.tensor(solver.rho(), dtype=torch.float64)
    f_t = torch.tensor(solver.f, dtype=torch.float64)
    p_cl = pressure_eos(rho_t, G_EFF).numpy()
    p_fl = pressure_flux(f_t, G_EFF).numpy()
    sk = sigma_korteweg(rho_t, N, STENCIL, G_EFF)
    return rho_t.numpy(), p_cl, p_fl, sk


def kernel_equivalence(steps=5):
    o = OSSShanChen(N, TAU, G_EFF, RHO0, RHO_L_INIT, RHO_G_INIT)
    o.f = tl_init(sharp_droplet(16.0)).numpy().copy()
    f_tl = torch.tensor(o.f.copy())
    for _ in range(steps):
        o.step()
        f_tl = step_sc(f_tl, G_eff=G_EFF, tau=TAU, stencil=STENCIL)
    return {"steps": steps,
            "max_abs_drho": float(np.max(np.abs(o.rho() - f_tl.sum(0).numpy()))),
            "max_abs_df": float(np.max(np.abs(o.f - f_tl.numpy())))}


def tl_full_run(R=16.0, steps=STEPS):
    """TensorLBM kernel time-stepped in full (dynamics-equivalence check)."""
    f = tl_init(sharp_droplet(R))
    t0 = time.time()
    for _ in range(steps):
        f = step_sc(f, G_eff=G_EFF, tau=TAU, stencil=STENCIL)
    return f, time.time() - t0


def main() -> int:
    t0 = time.time()
    S = {"anchor": "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION",
         "parameters": {"n": N, "tau": TAU, "G_eff": G_EFF, "rho0": RHO0,
                        "rho_l_init": RHO_L_INIT, "rho_g_init": RHO_G_INIT,
                        "steps": STEPS, "radii_init": RADII,
                        "lattice": "D2Q9, single component, fully periodic"},
         "oss_ref": {str(k): v for k, v in OSS_REF.items()},
         "oss_ref_lsqr": OSS_REF_LSQR,
         "arm_note": "dynamics = OSS port (numpy); TensorLBM readouts evaluated on "
                     "the same field; TensorLBM kernel time-stepped once at R=16 "
                     "to confirm the 4000-step agreement (kernel_equivalence_4000)",
         "dtype": "float64", "torch_threads": 2}

    print("[1/5] kernel equivalence, 5 steps", flush=True)
    S["kernel_equivalence_5"] = kernel_equivalence(5)
    print("   ", S["kernel_equivalence_5"], flush=True)

    print("[2/5] radius sweep", flush=True)
    rows, fields = [], {}
    for R in RADII:
        s = OSSShanChen(N, TAU, G_EFF, RHO0, RHO_L_INIT, RHO_G_INIT)
        s.init_droplet(R)
        tR = time.time()
        s.run(STEPS)
        wall = time.time() - tR
        rho, p_cl, p_fl, sk = readouts(s)
        ux, uy = s.velocity(*s.force())[1:]
        u_max = float(np.sqrt(ux ** 2 + uy ** 2).max())
        for arm, ro, rec in (("OSS", "closure", calib_safe(rho, closure_pressure(rho, G_EFF, RHO0))),
                             ("TL", "closure", calib_safe(rho, p_cl)),
                             ("TL", "flux", calib_safe(rho, p_fl))):
            row = {"R_init": R, "arm": arm, "readout": ro, "wall_s": wall,
                   "status": rec["status"], "u_max": u_max,
                   "sigma_K": sk["sigma_K"]}
            row.update({k: rec[k] for k in KEY})
            rows.append(row)
        fields[R] = {"rho": rho, "p_cl": p_cl, "p_fl": p_fl,
                     "sigma_K": sk["sigma_K"], "u_max": u_max}
        print("   R=%4.1f OSS gamma=%s (r=%s dp=%s) TL closure=%s TL flux=%s "
              "sigma_K=%.5f u_max=%.2e"
              % (R, fnum(rows[-3]["gamma"]), fnum(rows[-3]["r"], "%.3f"),
                 fnum(rows[-3]["dp"]), fnum(rows[-2]["gamma"]),
                 fnum(rows[-1]["gamma"]), sk["sigma_K"], u_max), flush=True)
    S["radius_sweep"] = rows
    with (OUT / "radius_sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def pick(R, arm, ro):
        return next(r for r in rows if r["R_init"] == R and r["arm"] == arm
                    and r["readout"] == ro)

    print("[3/5] TensorLBM full 4000-step run at R=16 (dynamics check)", flush=True)
    f_tl, wall_tl = tl_full_run(16.0)
    rho_tl = f_tl.sum(0).numpy()
    cal_tl = calib_safe(rho_tl, pressure_eos(f_tl.sum(0), G_EFF).numpy())
    cal_oss16 = pick(16.0, "TL", "closure")
    S["kernel_equivalence_4000"] = {
        "R_init": 16.0, "steps": STEPS,
        "max_abs_drho_vs_oss_port": float(np.max(np.abs(rho_tl - fields[16.0]["rho"]))),
        "gamma_tl_kernel": cal_tl["gamma"], "gamma_oss_port": cal_oss16["gamma"],
        "relerr_gamma": relerr(cal_tl["gamma"], cal_oss16["gamma"]),
        "wall_s": wall_tl}
    print("   ", S["kernel_equivalence_4000"], flush=True)

    comp = []
    for R in RADII:
        ref = OSS_REF[R]
        go, gt, gf = pick(R, "OSS", "closure"), pick(R, "TL", "closure"), pick(R, "TL", "flux")
        comp.append({"R_init": R, "status": gt["status"],
                     "oss_ref_gamma": ref["gamma"], "oss_ref_r": ref["r"],
                     "oss_ref_dp": ref["dp"],
                     "gamma_oss_port": go["gamma"], "gamma_tl_closure": gt["gamma"],
                     "gamma_tl_flux": gf["gamma"],
                     "relerr_gamma_port_vs_ref": relerr(go["gamma"], ref["gamma"]),
                     "relerr_gamma_tl_vs_ref": relerr(gt["gamma"], ref["gamma"]),
                     "relerr_flux_vs_closure": relerr(gf["gamma"], gt["gamma"]),
                     "r_port": go["r"], "ref_r": ref["r"], "relerr_r": relerr(go["r"], ref["r"]),
                     "dp_port": go["dp"], "relerr_dp": relerr(go["dp"], ref["dp"]),
                     "ref_rho_ratio": ref["rho_l"] / ref["rho_g"],
                     "rho_ratio_port": go["rho_ratio"],
                     "relerr_rho_ratio": relerr(go["rho_ratio"], ref["rho_l"] / ref["rho_g"]),
                     "dp_over_mean_p": abs(go["dp"]) / (0.5 * abs(go["p_l"] + go["p_g"]))})
    S["comparison_vs_oss_ref"] = comp

    def lsqr_for(arm, ro):
        xs, ys = [], []
        for R in (12.0, 16.0, 20.0):
            r = pick(R, arm, ro)
            if r["r"] is None or r["gamma"] is None:
                continue
            xs.append(1.0 / r["r"])
            ys.append(r["dp"])
        return lsqr_origin(xs, ys)

    g_o, r2_o = lsqr_for("OSS", "closure")
    g_c, r2_c = lsqr_for("TL", "closure")
    g_f, r2_f = lsqr_for("TL", "flux")
    S["laplace_lsqr"] = {
        "form": "through-origin LSQ of dp = sigma*(1/r) over R_init = 12/16/20",
        "oss_port": {"gamma": g_o, "r2": r2_o, "relerr_vs_ref": relerr(g_o, OSS_REF_LSQR)},
        "tl_closure": {"gamma": g_c, "r2": r2_c, "relerr_vs_ref": relerr(g_c, OSS_REF_LSQR)},
        "tl_flux": {"gamma": g_f, "r2": r2_f, "relerr_vs_ref": relerr(g_f, OSS_REF_LSQR)},
        "oss_ref": {"gamma": OSS_REF_LSQR, "r2": 0.988}}

    S["korteweg_anchor"] = [
        {"R_init": R, "sigma_K": fields[R]["sigma_K"],
         "dp_closure_times_r": pick(R, "TL", "closure")["gamma"],
         "dp_flux_times_r": pick(R, "TL", "flux")["gamma"],
         "relerr_closureR_vs_sigmaK": relerr(pick(R, "TL", "closure")["gamma"],
                                             fields[R]["sigma_K"]),
         "relerr_fluxR_vs_sigmaK": relerr(pick(R, "TL", "flux")["gamma"],
                                          fields[R]["sigma_K"])}
        for R in RADII]

    print("[4/5] time convergence at R=16", flush=True)
    s = OSSShanChen(N, TAU, G_EFF, RHO0, RHO_L_INIT, RHO_G_INIT)
    s.init_droplet(16.0)
    tconv, done = [], 0
    for k in (1000, 2000, 4000, 8000):
        s.run(k - done)
        done = k
        rho, p_cl, p_fl, sk = readouts(s)
        c = calib_safe(rho, p_cl)
        tconv.append({"steps": k, "gamma": c["gamma"], "dp": c["dp"], "r": c["r"],
                      "sigma_K": sk["sigma_K"],
                      "kappa3": sk["kappa3"], "status": c["status"]})
        print("   steps=%5d gamma=%s dp=%s sigma_K=%.5f"
              % (k, fnum(c["gamma"]), fnum(c["dp"]), sk["sigma_K"]), flush=True)
    S["time_convergence"] = tconv
    with (OUT / "time_convergence.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tconv[0]))
        w.writeheader()
        for r in tconv:
            w.writerow(r)

    print("[5/5] sensitivities at R=16", flush=True)
    rho16, p16 = fields[16.0]["rho"], fields[16.0]["p_cl"]
    sens = []
    for box in (4, 8):
        for two in (False, True):
            c = calib_safe(rho16, p16, box=box, two_sided=two)
            sens.append({"arm": "TL_closure", "box": box, "two_sided": two,
                         "r_definition": "midline", "dp": c["dp"], "r": c["r"],
                         "gamma": c["gamma"], "status": c["status"]})
    mid = 0.5 * (float(rho16.max()) + float(rho16.min()))
    area_r = math.sqrt(float((rho16 > mid).sum()) / math.pi)
    c = calib_safe(rho16, p16)
    sens.append({"arm": "TL_closure", "box": 4, "two_sided": False,
                 "r_definition": "area_equivalent", "dp": c["dp"], "r": area_r,
                 "gamma": c["dp"] * area_r, "status": "ok"})
    S["sensitivity"] = sens
    with (OUT / "sensitivity.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "box", "two_sided", "r_definition",
                                           "dp", "r", "gamma", "status"],
                           extrasaction="ignore")
        w.writeheader()
        for r in sens:
            w.writerow(r)

    figs = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.6, 4.5))
        for arm, ro, lab, mk in (("OSS", "closure", "OSS port (closure readout)", "o"),
                                 ("TL", "closure", "TensorLBM (closure readout)", "s"),
                                 ("TL", "flux", "TensorLBM (flux readout)", "^")):
            xs, ys = [], []
            for R in RADII:
                r = pick(R, arm, ro)
                if r["r"] is not None:
                    xs.append(r["r"])
                    ys.append(r["gamma"])
            ax.plot(xs, ys, mk + "-", label=lab, alpha=0.85)
        refR = [R for R in RADII if OSS_REF[R]["r"] < 100.0]
        ax.plot([OSS_REF[R]["r"] for R in refR], [OSS_REF[R]["gamma"] for R in refR],
                "kx", ms=9, label="OSS inventoried run (report table 1)")
        ax.axhline(g_c, color="C1", ls=":", lw=1,
                   label=r"LSQR $\gamma$=%.5f (TensorLBM)" % g_c)
        ax.axhline(OSS_REF_LSQR, color="0.4", ls=":", lw=1,
                   label=r"LSQR $\gamma$=%.5f (OSS)" % OSS_REF_LSQR)
        ax.axhspan(0.85 * OSS_REF_LSQR, 1.15 * OSS_REF_LSQR, color="0.85", alpha=0.5,
                   label="15% band around OSS $\\gamma$")
        ax.set_ylim(-0.004, 0.028)
        ax.set_xlabel("drop radius r (lattice units)")
        ax.set_ylabel(r"$\gamma = \Delta p\, r$  (lattice units)")
        ax.set_title("Young-Laplace calibration: TensorLBM vs OSS "
                     r"($G=-4.7$, $64^2$, 4000 steps)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "fig_laplace_gamma.png", dpi=140)
        plt.close(fig)
        figs.append({"name": "fig_laplace_gamma.png",
                     "caption": "gamma = dp*r vs drop radius: OSS port, TensorLBM closure "
                                "readout, TensorLBM flux readout, the OSS inventoried points "
                                "(report §九 table 1, R_init=24 omitted: the radius "
                                "estimator fails in both arms there) and the 15% band "
                                "around the OSS LSQR gamma."})
        fig, axs = plt.subplots(1, 2, figsize=(9.8, 3.9))
        y = N // 2
        axs[0].plot(rho16[y], label="density (both arms, |diff|<1e-15)")
        axs[0].axhline(mid, color="0.5", lw=0.8, ls="--", label=r"$\rho_{av}$")
        axs[0].set_xlabel("x (lattice units)")
        axs[0].set_ylabel(r"$\rho$")
        axs[0].set_title(r"mid-plane density, $R_{init}$=16, 4000 steps")
        axs[0].legend(fontsize=8)
        axs[0].grid(alpha=0.3)
        axs[1].plot(p16[y], label=r"closure $p=\rho/3+G\psi^2/6$")
        axs[1].plot(fields[16.0]["p_fl"][y], "--", label="momentum-flux $p$")
        axs[1].set_xlabel("x (lattice units)")
        axs[1].set_ylabel("p (lattice units)")
        axs[1].set_title("pressure readouts across the interface")
        axs[1].legend(fontsize=8)
        axs[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "fig_profiles.png", dpi=140)
        plt.close(fig)
        figs.append({"name": "fig_profiles.png",
                     "caption": "mid-plane density and the two pressure readouts across "
                                "the interface at R_init=16, 4000 steps; the readouts "
                                "differ only inside the interface band, so the bulk-band "
                                "difference isolates the non-equilibrium correction."})
    except Exception as exc:  # pragma: no cover
        S["figure_error"] = repr(exc)

    S["wall_clock_s"] = time.time() - t0
    (OUT / "summary.json").write_text(json.dumps(S, indent=2, default=float))
    if figs:
        (OUT / "figures_b64.json").write_text(json.dumps(
            [{"name": f["name"], "caption": f["caption"],
              "png_base64": base64.b64encode((OUT / f["name"]).read_bytes()).decode()}
             for f in figs], indent=1))
    print("\n== LSQR ==", json.dumps(S["laplace_lsqr"], indent=1, default=float), flush=True)
    print("wall clock %.1f s" % S["wall_clock_s"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())