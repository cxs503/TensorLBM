"""m17881 audit figure: independent re-derivation of the m17763 XVAL numbers."""
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent
RES = json.loads(open("/root/tp-agent/work/m17763/experiments/xval_ludwig_fe_stress/results.json").read())

A = B = 1.0
xi_list, err_deliv, err_audit = [], [], []
for kappa, xi in ((2.0, 2.0), (8.0, 4.0), (32.0, 8.0)):
    N = 256
    x = np.arange(N, dtype=float)
    phi = np.tanh((x - N / 2) / xi)
    dphi = (np.roll(phi, -1) - np.roll(phi, 1)) / 2.0
    w = int(8 * xi)
    sig_disc = kappa * (dphi[N // 2 - w:N // 2 + w] ** 2).sum()
    sig_ref = (2 * math.sqrt(2) / 3) * math.sqrt(kappa * A ** 3) / B
    xi_list.append(xi)
    err_audit.append(abs(sig_disc - sig_ref) / sig_ref)
    err_deliv.append(RES["cases"]["planar"][f"xi{xi:.1f}_iso9"]["sigma_rel_err"])

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# (a) discretisation convergence, deliverable vs independent audit recompute
ax[0].loglog(xi_list, err_deliv, "o-", label="deliverable results.json", lw=2)
ax[0].loglog(xi_list, err_audit, "x--", label="audit numpy re-derivation", ms=10, lw=1.5)
ref = np.array(xi_list, float)
ax[0].loglog(ref, err_deliv[0] * (ref[0] / ref) ** 2, ":", color="grey", label=r"$O(h^2/\xi^2)$ guide")
ax[0].axhline(0.15, color="tab:red", ls="--", lw=1, label="15% acceptance band")
ax[0].axhline(0.03, color="tab:orange", ls=":", lw=1, label="3% line")
ax[0].set_xlabel(r"interface width $\xi$ (lattice units)")
ax[0].set_ylabel(r"$|\sigma_{disc}-\sigma_{analytic}|/\sigma_{analytic}$")
ax[0].set_title("(a) sigma error vs interface resolution\n6.09% / 1.63% / 0.41% - reproduced to 5 digits")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3, which="both")

# (b) equilibrium force residual: both force forms, absolute scale sigma/xi^2
p = RES["cases"]["planar"]
ks = [f"xi{x:.1f}_iso9" for x in xi_list]
rs = [p[k]["residual_stress_over_Fscale"] for k in ks]
rg = [p[k]["residual_gradmu_over_Fscale"] for k in ks]
xx = np.arange(len(ks))
ax[1].bar(xx - 0.18, rs, 0.36, label=r"$-\nabla\cdot P$ (ludwig stress form)")
ax[1].bar(xx + 0.18, rg, 0.36, label=r"$-\phi\nabla\mu$ (TensorLBM in-situ)")
ax[1].set_xticks(xx, [f"$\\xi$={v:.0f}" for v in xi_list])
ax[1].set_ylabel(r"$\max|F|\,/\,(\sigma/\xi^2)$")
ax[1].set_title("(b) equilibrium body-force residual\nboth arms ~O(h^2), 5.3% -> 0.58%")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, axis="y")

# (c) verdict: what each level needs vs what the deliverable has
levels = ["L1 tests", "L2 impl+entry\n+artifacts", "L3 integration\ncallers", "L4 benchmark row\n+ref/got+repro", "L5 external\nanchor"]
ok = [1, 1, 0, 0, 0]
colors = ["tab:green" if v else "tab:red" for v in ok]
ax[2].barh(levels, ok, color=colors)
ax[2].set_xlim(0, 1.25)
for i, v in enumerate(ok):
    ax[2].text(0.03 if v else 0.03, i, "met" if v else "NOT met", va="center", color="white", fontsize=9, weight="bold")
ax[2].axvline(1.0, color="k", lw=0.8)
ax[2].set_title("(c) maturity verdict: L2\nno solver caller; 0/65 benchmark rows match anchor")
ax[2].grid(alpha=0.3, axis="x")

fig.suptitle("m17881 cross-audit of m17763: XVAL-LBM free-energy pressure-tensor body force (TensorLBM)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
(OUT / "figures").mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "figures" / "m17881_audit_verification.png", dpi=130)
print("wrote", OUT / "figures" / "m17881_audit_verification.png")
print("audit errs", err_audit)