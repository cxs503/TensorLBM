"""One-shot delivery for XVAL-LBM-R4-COEXISTENCE (fast sizes; artifacts + figure + report)."""
from __future__ import annotations
import importlib.util, json, math, os, sys, time, traceback
import torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
spec = importlib.util.spec_from_file_location("rx", os.path.join(HERE, "run_xval.py"))
rx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rx)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorlbm import free_energy_stress as fes
from tensorlbm.sc_eos_coexistence import maxwell_coexistence

OUT = os.path.join(ROOT, "artifacts", "xval_lbm_r4_coexistence")
os.makedirs(OUT, exist_ok=True)
GATE = 0.15
t0 = time.time()
res = {"gate": GATE, "wall_start": time.strftime("%Y-%m-%dT%H:%M:%S"), "route_M": [], "route_S": [], "route_C": []}

psi_fn = lambda r: rx.sc_psi(r, 1.0)   # noqa: E731
for G in (4.5, 5.0):
    m = maxwell_coexistence(G=G, psi_fn=psi_fn, rho_lo=1e-5, rho_hi=50.0, n=6000, n_psat=80, n_quad=200)
    d = m.as_dict() if hasattr(m, "as_dict") else dict(m)
    res["route_M"].append({"G": G, **{k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in d.items()}})
    print("M", G, json.dumps(res["route_M"][-1])[:300], flush=True)

for G in (4.5, 5.0):
    for ny in (48, 64):
        r = rx.sc_flat_interface(G=G, ny=ny, nx=2, steps=6000, tau=0.8)
        res["route_S"].append({k: v for k, v in r.items() if k != "profile_mid_column"})
        res.setdefault("profiles_SC", {})[f"{G}_{ny}"] = r["profile_mid_column"]
        print("S", G, ny, r["rho_l"], r["rho_v"], round(time.time() - t0, 1), flush=True)

for kappa in (2.0, 0.5):
    for ny in (96, 128):
        p, phi = rx.ch_relax(A=1.0, B=1.0, kappa=kappa, ny=ny, nx=4, steps=6000)
        m = rx.measure_interface(phi, p, ny)
        res["route_C"].append({"kappa": kappa, "A": 1.0, "B": 1.0, "ny": ny, **m})
        if ny == 128:
            res.setdefault("profiles_CH", {})[str(kappa)] = [float(v) for v in phi[:, 0]]
        print("C", kappa, ny, m["xi_lsq"], m["sigma_tensor"], round(time.time() - t0, 1), flush=True)

# ---- comparisons
rows = []
m_by_G = {round(r["G"], 3): r for r in res["route_M"]}
key_l = lambda d: next((k for k in d if "rho_l" in k.lower() or "liquid" in k.lower()), None)   # noqa: E731
key_v = lambda d: next((k for k in d if "rho_v" in k.lower() or "vap" in k.lower()), None)       # noqa: E731
for s in res["route_S"]:
    mm = m_by_G.get(round(s["G"], 3))
    if not mm or not (key_l(mm) and key_v(mm)):
        continue
    rl, rv = float(mm[key_l(mm)]), float(mm[key_v(mm)])
    el = abs(s["rho_l"] - rl) / abs(rl) if rl else float("nan")
    ev = abs(s["rho_v"] - rv) / abs(rv) if rv else float("nan")
    rows.append({"kind": "coexistence(SC)", "G": s["G"], "ny": s["ny"],
                 "ref_rho_l": rl, "got_rho_l": s["rho_l"], "err_rho_l": el, "pass_rho_l": bool(el < GATE),
                 "ref_rho_v": rv, "got_rho_v": s["rho_v"], "err_rho_v": ev, "pass_rho_v": bool(ev < GATE),
                 "spurious_u_max": s["spurious_u_max"]})
for c in res["route_C"]:
    e1 = abs(c["xi_lsq"] - c["xi_analytic"]) / c["xi_analytic"]
    e2 = abs(c["xi_10_90"] - c["xi_analytic"]) / c["xi_analytic"]
    e3 = abs(c["sigma_tensor"] - c["sigma_analytic"]) / c["sigma_analytic"]
    rows.append({"kind": "free-energy(CH)", "kappa": c["kappa"], "ny": c["ny"],
                 "ref_xi": c["xi_analytic"], "got_xi": c["xi_lsq"], "err_xi": e1,
                 "got_xi_1090": c["xi_10_90"], "err_xi_1090": e2,
                 "ref_sigma": c["sigma_analytic"], "got_sigma": c["sigma_tensor"], "err_sigma": e3,
                 "pass_xi": bool(max(e1, e2) < GATE), "pass_sigma": bool(e3 < GATE),
                 "max_abs_divP": c["max_abs_divP"], "mu_std_over_A": c["mu_std_over_A"]})
res["comparison"] = rows
flags = [v for r in rows for k, v in r.items() if k.startswith("pass_")]
res["gate_summary"] = {"n_checks": len(flags), "n_pass": int(sum(bool(v) for v in flags)),
                       "all_pass": bool(flags and all(bool(v) for v in flags)),
                       "max_err": max([v for r in rows for k, v in r.items() if k.startswith("err_") and isinstance(v, float)] or [0.0])}
res["wall_s"] = round(time.time() - t0, 1)
with open(os.path.join(OUT, "xval_r4_coexistence.json"), "w") as fh:
    json.dump(res, fh, indent=1)
print("SUMMARY", json.dumps(res["gate_summary"]), flush=True)

# ---- figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for k, prof in res.get("profiles_SC", {}).items():
    ax[0].plot(prof, label=f"SC G={k.split('_')[0]} ny={k.split('_')[1]}")
ax[0].set_xlabel("y (lattice)"); ax[0].set_ylabel(r"$\rho$"); ax[0].set_title("Shan-Chen flat interface (measured)")
ax[0].legend(fontsize=7)
for k, prof in res.get("profiles_CH", {}).items():
    p = fes.FreeEnergyParams(A=1.0, B=1.0, kappa=float(k))
    yy = torch.arange(len(prof)).double()
    ref = fes.planar_profile(1, len(prof), p)[0]
    ax[1].plot(prof, label=f"relaxed kappa={k}")
    ax[1].plot(0.5 * ref, "k--", lw=0.8, label="analytic tanh (shifted)" if k == "2.0" else None)
ax[1].set_xlabel("y"); ax[1].set_ylabel(r"$\phi$"); ax[1].set_title("Free-energy interface: relaxed vs analytic")
ax[1].legend(fontsize=7)
lbl = [f"{r['kind'].split('(')[0]}\nG={r.get('G', r.get('kappa'))} ny={r['ny']}" for r in rows]
ax[2].bar(range(len(rows)), [max(r.get(k, 0) for k in r if k.startswith("err_")) * 100 for r in rows])
ax[2].axhline(GATE * 100, color="r", ls="--", label="gate 15%")
ax[2].set_yscale("log"); ax[2].set_xticks(range(len(rows))); ax[2].set_xticklabels(lbl, fontsize=6, rotation=90)
ax[2].set_ylabel("max relative error [%]"); ax[2].set_title("vs reference (Maxwell / analytic)"); ax[2].legend()
fig.tight_layout()
png = os.path.join(OUT, "xval_r4_coexistence.png")
fig.savefig(png, dpi=130)
print("figure", png, flush=True)
json.dump({"rows": rows, "summary": res["gate_summary"], "png": png},
          open(os.path.join(OUT, "_rows.json"), "w"), indent=1)