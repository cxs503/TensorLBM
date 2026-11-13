"""Aggregate the xval arm artefacts into results.json + figures (zero hand-typed numbers)."""
from __future__ import annotations
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ART = os.path.join(ROOT, "artifacts", "xval_free_energy_binary_equilibrium")
FIG = os.path.join(ROOT, "docs", "figures")
PRE = json.load(open(os.path.join(ROOT, "xval", "XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE", "predeclared.json")))
TH = PRE["predeclared_thresholds"]

def load(name):
    p = os.path.join(ART, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None

arms = {n: load(n) for n in ("flat", "dropR16", "dropR24", "dropR36")}
os.makedirs(FIG, exist_ok=True)

flat = arms["flat"]
rows = []
checks = []
if flat:
    checks += [
        ("closure_rel <= 1e-12", flat["closure_final"]["rel"] <= TH["closure_rel_max"], flat["closure_final"]["rel"], TH["closure_rel_max"]),
        ("sigma rel err <= 15%", flat["sigma_rel_error"] <= TH["sigma_rel_error_max"], flat["sigma_rel_error"], TH["sigma_rel_error_max"]),
        ("Pnn uniformity <= 15%", flat["Pnn_uniformity_rel"] <= TH["Pnn_uniformity_rel_max"], flat["Pnn_uniformity_rel"], TH["Pnn_uniformity_rel_max"]),
    ]
for name in ("dropR16", "dropR24", "dropR36"):
    a = arms[name]
    if not a:
        continue
    rows.append(dict(arm=name, R=a["R"], xi=a["config"]["xi"],
                     sigma_th=a["config"]["sigma_analytic"],
                     dp=a["dp"], dp_th=a["dp_analytic"],
                     ratio=a["laplace_ratio_analytic"], err=a["laplace_rel_error_analytic"],
                     sigma_ray=a["sigma_ray"]["mean"], ray_rel=a["sigma_ray"]["ray_spread_rel"],
                     ratio_ray=a["laplace_ratio_ray"], err_ray=a["laplace_rel_error_ray"],
                     u_max=a["spurious_velocity"]["max"], steps=a["steps_run"],
                     closure=a["closure_final"]["rel"]))
    checks += [
        (f"{name} laplace <= 15%", a["laplace_rel_error_analytic"] <= TH["laplace_rel_error_max"], a["laplace_rel_error_analytic"], TH["laplace_rel_error_max"]),
        (f"{name} ray-sigma laplace <= 15%", a["laplace_rel_error_ray"] <= TH["laplace_rel_error_max"], a["laplace_rel_error_ray"], TH["laplace_rel_error_max"]),
        (f"{name} spurious |u| <= 1e-2", a["spurious_velocity"]["max"] <= TH["spurious_velocity_max"], a["spurious_velocity"]["max"], TH["spurious_velocity_max"]),
        (f"{name} closure <= 1e-12", a["closure_final"]["rel"] <= TH["closure_rel_max"], a["closure_final"]["rel"], TH["closure_rel_max"]),
    ]

verdict = all(c[1] for c in checks) if checks else False
out = dict(anchor=PRE["anchor"], checks=[dict(name=c[0], passed=bool(c[1]), value=c[2], threshold=c[3]) for c in checks],
           verdict="PASS" if verdict else "FAIL", laplace_rows=rows,
           flat=({k: flat[k] for k in ("sigma_measured", "sigma_analytic", "sigma_rel_error", "Pnn_uniformity_rel", "spurious_velocity", "rho_range", "closure_final", "config")} if flat else None))
json.dump(out, open(os.path.join(ART, "results.json"), "w"), indent=1)

# figures
if flat:
    pr = flat["profiles"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(pr["x"], pr["phi"], "o", ms=3, label="simulated phi")
    ax[0].plot(pr["x"], pr["phi_analytic"], "-", label="tanh analytic")
    ax[0].set_xlabel("x [lattice]"); ax[0].set_ylabel("phi"); ax[0].legend(); ax[0].set_title("flat band: order parameter")
    ax[1].plot(pr["x"], pr["Pxx_measured"], "o", ms=3, label="P_NN measured (2nd moment)")
    ax[1].plot(pr["x"], pr["Pxx_analytic"], "-", label="P_NN analytic tensor")
    ax[1].set_xlabel("x [lattice]"); ax[1].set_ylabel("P_NN"); ax[1].legend(); ax[1].set_title("mechanical equilibrium")
    ax[2].plot(pr["x"], pr["Pyy_measured"], label="P_TT measured")
    ax[2].plot(pr["x"], pr["Pxx_measured"], label="P_NN measured")
    ax[2].set_xlabel("x [lattice]"); ax[2].set_ylabel("pressure"); ax[2].legend(); ax[2].set_title("sigma = integral(P_NN-P_TT)/2")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "xval_fe_pressure_flat_band.png"), dpi=130); plt.close(fig)

for name in ("dropR16", "dropR24", "dropR36"):
    a = arms[name]
    if not a:
        continue
    r = a["radial"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(r["r"], r["phi"], "o", ms=3, label="simulated phi")
    ax[0].plot(r["r"], r["phi_analytic"], "-", label="tanh analytic")
    ax[0].set_xlabel("r"); ax[0].set_ylabel("phi"); ax[0].legend(); ax[0].set_title(f"{name}: phi(r), R={a['R']}, xi={a['config']['xi']}")
    ax[1].plot(r["r"], r["p"], label="p=(Pxx+Pyy)/2")
    ax[1].axhline(a["p_in"], ls=":", label="p_in"); ax[1].axhline(a["p_out"], ls="--", label="p_out")
    ax[1].set_xlabel("r"); ax[1].set_ylabel("p"); ax[1].legend(); ax[1].set_title(f"dp={a['dp']:.4e} analytic={a['dp_analytic']:.4e}")
    ax[2].plot(r["r"], r["u_mag"], label="|u|")
    ax[2].set_yscale("log"); ax[2].set_xlabel("r"); ax[2].set_ylabel("|u|"); ax[2].legend(); ax[2].set_title("spurious current")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"xval_fe_pressure_{name}.png"), dpi=130); plt.close(fig)

if rows:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(1.0, color="k", ls="--", label="dp = sigma/R")
    Rs = [r["R"] for r in rows]; rr = [r["ratio"] for r in rows]
    ax.errorbar(Rs, rr, yerr=[[abs(v)*0.15 for v in rr], [abs(v)*0.15 for v in rr]], fmt="o", label="measured")
    ax.set_xlabel("droplet radius R"); ax.set_ylabel("dp R / sigma"); ax.set_title("Laplace law, 15% band"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "xval_fe_pressure_laplace.png"), dpi=130); plt.close(fig)

print(json.dumps(out["checks"], indent=1))
print("VERDICT", out["verdict"])
for r in rows:
    print(f"R={r['R']} xi={r['xi']} dp={r['dp']:.5e} th={r['dp_th']:.5e} ratio={r['ratio']:.4f} err={r['err']*100:.2f}% sigma_ray={r['sigma_ray']:.5f} ratio_ray={r['ratio_ray']:.4f} err_ray={r['err_ray']*100:.2f}% umax={r['u_max']:.2e}")
