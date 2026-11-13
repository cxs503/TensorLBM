"""XVAL: TensorLBM free-energy body force (-phi grad mu) vs ludwig
FE_FORCE_METHOD_STRESS_DIVERGENCE (F = -div P from the free-energy pressure
tensor).  Run: python experiments/xval_ludwig_fe_stress/run_xval.py"""
from __future__ import annotations
import json, math, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from tensorlbm.free_energy_stress import (
    FreeEnergyParams, grad, laplacian, pressure_tensor, stress_divergence_force,
    grad_mu_force, surface_tension_analytic, interface_width, planar_profile,
    droplet_profile,
)
torch.set_default_dtype(torch.float64)
OUT = pathlib.Path(__file__).resolve().parent
res = {"metric_defs": {
    "sigma_rel_err": "|sum_window kappa|grad phi|^2 - sigma_analytic| / sigma_analytic (flat interface window)",
    "force_diff_rel": "max|F_stress - F_gradmu| / max|F_gradmu| over the interface window",
    "residual_stress_over_Fscale": "max|F_stress| / (sigma/xi^2) over the window (0 at equilibrium)",
    "window": "+-8 xi around the interface centre; periodic-boundary artefacts excluded",
}, "cases": {}}

planar = {}
for kappa in (2.0, 8.0, 32.0):                     # xi = 2, 4, 8
    p = FreeEnergyParams(1.0, 1.0, kappa); L = 256; xi = interface_width(p)
    x0 = L // 2; w = int(8 * xi)
    sl = slice(x0 - w, x0 + w)
    for scheme in ("5pt", "iso9"):
        phi = planar_profile(L, L, p)
        gx, gy = grad(phi, scheme)
        sig_d = float(p.kappa * (gx[0, sl] ** 2).sum())
        sig_a = surface_tension_analytic(p)
        Fs, Fg = stress_divergence_force(phi, p, scheme), grad_mu_force(phi, p, scheme)
        mS = max(float(f[:, sl].abs().max()) for f in Fs)
        mG = max(float(f[:, sl].abs().max()) for f in Fg)
        dS = max(float((a[:, sl] - b[:, sl]).abs().max()) for a, b in zip(Fs, Fg))
        Fs_ = sig_a / xi ** 2
        planar[f"xi{xi:.1f}_{scheme}"] = {
            "kappa": kappa, "xi": xi, "scheme": scheme, "sigma_analytic": sig_a,
            "sigma_discrete": sig_d, "sigma_rel_err": abs(sig_d - sig_a) / sig_a,
            "residual_stress_over_Fscale": mS / Fs_, "residual_gradmu_over_Fscale": mG / Fs_,
            "force_diff_rel": dS / mG,
        }
res["cases"]["planar"] = planar

# ---- curved interface (droplet): Korteweg curvature balance ------------------
curved = {}
for L, R, kap in ((128, 32, 8.0), (256, 64, 32.0)):
    p = FreeEnergyParams(1.0, 1.0, kap); xi = interface_width(p); sig_a = surface_tension_analytic(p)
    phi = droplet_profile(L, L, p, radius=R); c = L // 2
    sl = slice(c + 1, min(L - 2, c + int(R + 6 * xi) + 1))   # positive half only: one interface
    gx, gy = grad(phi, "iso9")
    Pxx, Pyy, Pxy = pressure_tensor(phi, p, "iso9")
    d = Pxx - p.kappa * gx ** 2
    Pnn = d[c, :] + p.kappa * gx[c, :] ** 2
    Ptt = d[c, :]
    sig_curv = float((Pnn - Ptt)[sl].sum())                     # sigma from the curved interface
    r = torch.arange(L, dtype=torch.float64) - c
    kern = ((Pnn - Ptt) / r.abs().clamp_min(xi))[sl]            # dP_rr/dr=(P_tt-P_rr)/r -> int = sigma/R
    balance = float(kern.sum())                                  # should approach sigma/R
    # an independent sigma from a flat interface at the same kappa
    gfx = grad(planar_profile(L, L, p), "iso9")[0]; slf = slice(L // 2 - int(6 * xi), L // 2 + int(6 * xi))
    sig_flat = float(p.kappa * (gfx[0, slf] ** 2).sum())
    curved[f"L{L}_R{R}_xi{xi:.1f}"] = {
        "L": L, "R": R, "xi": xi, "R_over_xi": R / xi, "sigma_analytic": sig_a,
        "sigma_from_curved_tensor": sig_curv, "sigma_curved_rel_err": abs(sig_curv - sig_a) / sig_a,
        "sigma_flat_discrete": sig_flat,
        "radial_balance_lhs": balance, "sigma_flat_over_R": sig_flat / R,
        "radial_balance_abs_rel_err": abs(abs(balance) - sig_flat / R) / (sig_flat / R),
        "Pnn_centre": float(Pnn[c]), "Pnn_far": float(Pnn[L - 8]),
        "note": "symmetric free energy: both bulk phases share p0, so the classical "
                "dP=sigma/R jump lives in the total (density-coupled) LBM pressure, not in the tensor",
    }
res["cases"]["curved_interface"] = curved

res["summary"] = {
    "sigma_rel_err": {k: v["sigma_rel_err"] for k, v in planar.items()},
    "force_diff_rel": {k: v["force_diff_rel"] for k, v in planar.items()},
    "worst_sigma_rel_err": max(v["sigma_rel_err"] for v in planar.values()),
    "worst_force_diff_rel": max(v["force_diff_rel"] for v in planar.values()),
    "worst_curved_sigma_rel_err": max(v["sigma_curved_rel_err"] for v in curved.values()),
    "pass_15pct": bool(max(v["sigma_rel_err"] for v in planar.values()) < 0.15
                       and max(v["sigma_curved_rel_err"] for v in curved.values()) < 0.15),
}
(OUT / "results.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res["summary"], indent=2))
for k, v in planar.items():
    print(f"{k:12s} sig_err={v['sigma_rel_err']:.3e} resS={v['residual_stress_over_Fscale']:.3e} "
          f"resG={v['residual_gradmu_over_Fscale']:.3e} fdiff={v['force_diff_rel']:.3e}")
for k, v in curved.items():
    print(f"{k:16s} sigc_err={v['sigma_curved_rel_err']:.3e} bal_lhs={v['radial_balance_lhs']:.4f} "
          f"sigflat/R={v['sigma_flat_over_R']:.4f} bal_err={v['radial_balance_abs_rel_err']:.3e}")
