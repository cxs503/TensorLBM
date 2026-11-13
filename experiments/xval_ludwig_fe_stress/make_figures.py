import json, math, pathlib, sys, base64
import torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from tensorlbm.free_energy_stress import *
torch.set_default_dtype(torch.float64)
OUT = pathlib.Path(__file__).resolve().parent; (OUT / "figures").mkdir(exist_ok=True)
r = json.loads((OUT / "results.json").read_text())
p = FreeEnergyParams(1., 1., 8.); L = 256; x = torch.arange(L) - L // 2
phi = planar_profile(L, L, p)
Fs = stress_divergence_force(phi, p, "iso9"); Fg = grad_mu_force(phi, p, "iso9")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(x.numpy(), Fs[0][0].numpy(), label=r"ludwig stress divergence $-\nabla\cdot P$")
ax[0].plot(x.numpy(), Fg[0][0].numpy(), "--", label=r"TensorLBM $-\phi\nabla\mu$")
ax[0].set_xlim(-16, 16); ax[0].set_xlabel("x"); ax[0].set_ylabel(r"$F_x$")
ax[0].set_title(r"body force, $\xi=4$ (equilibrium profile)"); ax[0].legend(fontsize=8)
xi = [v["xi"] for k, v in r["cases"]["planar"].items() if k.endswith("iso9")]
se = [v["sigma_rel_err"] for k, v in r["cases"]["planar"].items() if k.endswith("iso9")]
rs = [v["residual_stress_over_Fscale"] for k, v in r["cases"]["planar"].items() if k.endswith("iso9")]
rg = [v["residual_gradmu_over_Fscale"] for k, v in r["cases"]["planar"].items() if k.endswith("iso9")]
ax[1].loglog(xi, se, "o-", label=r"$\sigma$ rel. error vs analytic")
ax[1].loglog(xi, rs, "s--", label="stress residual / ($\\sigma/\\xi^2$)")
ax[1].loglog(xi, rg, "^:", label="grad$\\mu$ residual / ($\\sigma/\\xi^2$)")
ax[1].axhline(0.15, color="r", ls="-.", lw=1, label="15% criterion")
ax[1].set_xlabel(r"interface width $\xi$ [lattice]"); ax[1].set_ylabel("relative error")
ax[1].set_title("convergence w.r.t. interface resolution"); ax[1].legend(fontsize=7)
L2, R = 128, 32; p2 = FreeEnergyParams(1., 1., 8.)
ph = droplet_profile(L2, L2, p2, radius=R); c = L2 // 2
Pxx, Pyy, Pxy = pressure_tensor(ph, p2, "iso9"); gx, gy = grad(ph, "iso9")
d = Pxx - p2.kappa * gx ** 2; Pnn = d[c] + p2.kappa * gx[c] ** 2; Ptt = d[c]
ax[2].plot(torch.arange(L2).numpy() - c, Pnn.numpy(), label=r"$P_{NN}$")
ax[2].plot(torch.arange(L2).numpy() - c, Ptt.numpy(), "--", label=r"$P_{TT}$")
ax[2].set_ylim(0.0, 0.42); ax[2].set_xlabel("r"); ax[2].set_ylabel("pressure tensor")
ax[2].set_title(r"curved interface, $R=32,\ \xi=4$"); ax[2].legend(fontsize=8)
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); f1 = OUT / "figures" / "xval_fe_stress.png"; fig.savefig(f1, dpi=130)
b64 = base64.b64encode(f1.read_bytes()).decode()
pathlib.Path("/tmp/fig_b64.txt").write_text(b64)
json.dump(r["summary"], open("/tmp/xval_summary.json", "w"), indent=2)
print("figure", f1, "bytes", f1.stat().st_size, "b64", len(b64))
print(json.dumps({k: v for k, v in r["cases"]["curved_interface"].items()}, indent=1))
