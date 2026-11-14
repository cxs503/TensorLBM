"""Compact driver: run the round-2 XVAL anchor study on a local CPU budget."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path('/srv/tensor/agent/work/m21044')
XV = HERE / 'xval' / 'XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACT'
sys.path.insert(0, str(HERE / 'src'))
sys.path.insert(0, str(XV))

import xval_linear_theory as X

ART = XV / 'artifacts'
ART.mkdir(exist_ok=True)
k = X.K_MODE
N, L = 601, 400.0
W = X.W_FRONT
LOG = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


log(f"== XVAL round-2 anchor study (compact local run) ==")
log(f"   k = {k:.6e}  k^2 = {k*k:.6e}  w = {W}  grid N={N} L={L}")

v1 = X.exact_linear_sigma(k, 0.0, W, N=N, L=L)
v1_rel = abs(v1["sigma"].real + k * k) / (k * k)
log(f"V1 R=0: sigma={v1['sigma'].real:+.6e} vs -k^2={-k*k:+.6e} rel.dev={100*v1_rel:.3f}%")

v2 = {"rows": []}
for R in (-2.0, -1.0, -0.5):
    st = X.step_analytic_sigma(R, k)
    for w in (8.0, 4.0, 2.0):
        s = X.exact_linear_sigma(k, R, w, N=N, L=L)
        rel = abs(s["sigma"].real - st["sigma"].real) / abs(st["sigma"].real)
        v2["rows"].append({"R": R, "w": w, "k_w": k * w, "sigma_num": s["sigma"].real,
                           "sigma_step": st["sigma"].real, "rel_dev": rel})
        log(f"V2 R={R:<5} w={w:<4} sigma_num={s['sigma'].real:+.6e} "
            f"sigma_step={st['sigma'].real:+.6e} rel.dev={100*rel:.2f}%")
    v2[f"step_R{R}"] = {"kappa": str(st["kappa"]), "sigma": st["sigma"]}
rr = [r for r in v2["rows"] if r["R"] == -1.0]
v2["monotone"] = all(a["rel_dev"] > b["rel_dev"] for a, b in zip(rr, rr[1:]))
log(f"   w->0 monotone: {v2['monotone']}")

sens = []
for (n2, l2) in ((401, 400.0), (801, 400.0), (601, 800.0)):
    s = X.exact_linear_sigma(k, -1.0, W, N=n2, L=l2)
    sens.append({"N": n2, "L": l2, "h": s["h"], "sigma": s["sigma"].real})
    log(f"   sensitivity N={n2:<4} L={l2:<6} h={s['h']:.3f} sigma={s['sigma'].real:+.6e}")

mod = [X.module_growth_rate(R) for R in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)]
for g in mod:
    log(f"M  module R={g['R']:<5} sigma={g['sigma']:+.6e}")

rows = []
theory = {}
for g in mod:
    R = g["R"]
    th = X.exact_linear_sigma(k, R, W, N=N, L=L)["sigma"].real
    theory[R] = th
    st = X.step_analytic_sigma(R, k)
    sts = st["sigma"].real if st["admissible"] else float("nan")
    rel = abs(g["sigma"] - th) / abs(th) if abs(th) > 1e-6 else float("nan")
    rows.append({"R": R, "sigma_module": g["sigma"], "sigma_theory": th,
                 "sigma_step_analytic": sts, "k2_diffusive": k * k,
                 "abs_err_module_vs_theory": abs(g["sigma"] - th),
                 "rel_err_module_vs_theory": rel,
                 "sign_agree": bool(np.sign(g["sigma"]) == np.sign(th)),
                 "pass_15pct": bool(rel < 0.15)})
    log(f"X  R={R:<5} sigma_module={g['sigma']:+.6e} sigma_theory={th:+.6e} "
        f"sigma_step={sts:+.6e} sign_agree={rows[-1]['sign_agree']} rel.dev={100*rel:.1f}%")

mirror = []
for g in mod:
    R = g["R"]
    if R == 0.0:
        continue
    th = theory[-R]
    rel = abs(g["sigma"] - th) / abs(th)
    mirror.append({"R": R, "sigma_module": g["sigma"], "sigma_theory_minusR": th,
                   "rel_dev": rel, "pass_15pct": bool(rel < 0.15)})
    log(f"Y  R={R:<5} sigma_module={g['sigma']:+.6e} sigma_theory(-R)={th:+.6e} "
        f"rel.dev={100*rel:.1f}%")

figs = X.make_figures(k, rows, mirror, v2, mod)
res = {"k": k, "k2": k * k, "mode": X.MODE, "pe": X.PE, "w_front": W, "dt": X.DT,
       "ny": X.NY, "nx": X.NX, "t_fit": list(X.T_FIT), "anchor_grid": {"N": N, "L": L},
       "V1_sigma_R0": v1["sigma"].real, "V1_rel_dev_vs_minus_k2": v1_rel,
       "V2_sharp_step": v2, "anchor_sensitivity": sens,
       "module_runs": [{kk: vv for kk, vv in g.items() if kk not in ("ts", "amps")} for g in mod],
       "comparison": rows, "mirror_comparison": mirror, "figures": figs, "log": LOG}
(ART / "linear_theory.json").write_text(json.dumps(res, indent=1, default=str))
(ART / "xval_r2_run.log").write_text("\n".join(LOG) + "\n")
log(f"wrote {ART/'linear_theory.json'} + {len(figs)} figures")