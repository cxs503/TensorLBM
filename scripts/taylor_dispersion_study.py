"""Taylor-Aris shear-dispersion study: Pe sweep vs analytic reference + figures.

Reproduces docs/taylor_aris_dispersion_20261109.md.  Figures and results.json are
generated programmatically (no hand-copied numbers).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(1)

from tensorlbm.taylor_dispersion import (
    ShearDispersionConfig,
    periodic_shear_taylor_diffusivity,
    run_periodic_shear_dispersion,
    run_uniform_flow_control,
)

OUT = Path("docs/assets/taylor_dispersion")
OUT.mkdir(parents=True, exist_ok=True)
D = 0.1
PES = [0.001, 2.0, 5.0, 8.0, 12.0]

rows = []
for pe in PES:
    cfg = ShearDispersionConfig(pe=pe)
    res = run_periodic_shear_dispersion(cfg)
    rows.append(
        {
            "pe": pe,
            "u_amp": cfg.u_amp,
            "k": cfg.k,
            "d_eff_numeric": res.d_eff_numeric,
            "d_eff_theory": res.d_eff_theory,
            "rel_error": res.rel_error,
            "rate_fit": res.rate_fit,
            "transverse_rate": res.transverse_rate,
            "axial_rate_theory": res.axial_rate_theory,
            "ny": cfg.ny,
            "nx": cfg.nx,
            "n_steps": cfg.n_steps,
            "fit_start_time": res.fit_start_time,
        }
    )
    print(f"Pe={pe:6.3f}  D_eff num={res.d_eff_numeric:.5f} th={res.d_eff_theory:.5f} err={res.rel_error*100:+.2f}%")

ctl = run_uniform_flow_control(ShearDispersionConfig(pe=8.0))
print(f"uniform control: num={ctl.d_eff_numeric:.5f} vs D={D} err={ctl.rel_error*100:+.2f}%")

sweep = [r for r in rows if r["pe"] >= 2.0]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
pe_line = [2.0 + 0.05 * i for i in range(int((12.0 - 2.0) / 0.05) + 1)]
ax.plot(pe_line, [1.0 + p**2 / (8 * math.pi**2) for p in pe_line], "-",
        color="k", lw=1.6, label=r"Taylor–Aris: $D_{eff}/D=1+Pe^2/8\pi^2$")
ax.plot([r["pe"] for r in sweep], [r["d_eff_numeric"] / D for r in sweep], "o",
        ms=7, color="C3", label="LBM D2Q5 (modal decay fit)")
ax.plot([8.0], [ctl.d_eff_numeric / D], "s", ms=7, mfc="none", color="C0",
        label="negative control (uniform flow)")
ax.set_xlabel(r"transverse Péclet  $Pe=U H/D$")
ax.set_ylabel(r"$D_{eff}/D$")
ax.set_title("Taylor–Aris dispersion: LBM vs analytic (D=0.1, H=16, L=128)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "dispersion_vs_pe.png", dpi=140)
plt.close(fig)

cfg = ShearDispersionConfig(pe=8.0)
res = run_periodic_shear_dispersion(cfg)
t = torch.tensor(res.times, dtype=torch.float64)
a = torch.tensor(res.amplitudes, dtype=torch.float64)
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.semilogy(t, a, "o", ms=2.5, color="C0", label="cross-section mean mode |A(t)|")
fit = t >= res.fit_start_time
ax.semilogy(t[fit], torch.exp(torch.log(a[fit][0]) - res.rate_fit * (t[fit] - t[fit][0])),
            "-", color="k", lw=1.2, label=rf"fit: $\exp(-k^2D_{{eff}}t)$ → $D_{{eff}}$={res.d_eff_numeric:.4f}")
ax.axvline(res.fit_start_time, color="C3", ls="--", lw=1, label="fit window start")
ax.set_xlabel("time step")
ax.set_ylabel(r"mode amplitude $|A|$")
ax.set_title(f"Taylor mode decay (Pe=8): theory $D_{{eff}}$={res.d_eff_theory:.4f} ({res.rel_error*100:+.2f}%)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "mode_decay.png", dpi=140)
plt.close(fig)

payload = {
    "config": {"D": D, "lattice": "D2Q5", "dtype": "float64", "ny": cfg.ny, "nx": cfg.nx,
               "n_steps": cfg.n_steps, "measure_every": cfg.measure_every, "D_keys": list(cfg.__dict__)},
    "sweep": rows,
    "uniform_control": {"pe": ctl.pe, "d_eff_numeric": ctl.d_eff_numeric, "rel_error": ctl.rel_error},
    "analytic": {"periodic_shear": periodic_shear_taylor_diffusivity(D, 0.05, 16.0)},
}
(OUT / "results.json").write_text(json.dumps(payload, indent=2))
print("wrote", OUT / "results.json")
