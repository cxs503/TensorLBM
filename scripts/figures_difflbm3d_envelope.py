#!/usr/bin/env python3
"""Figures for the GAP-DIFFLBM-3D-COST envelope report (numbers from artifacts)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "difflbm3d_envelope"
OUT = ROOT / "figures" / "difflbm3d_envelope"
OUT.mkdir(parents=True, exist_ok=True)


def load(p: Path) -> dict:
    return json.loads(p.read_text())


# ---------------------------------------------------------------- memory model
pts = [(32, 0.675, 0.00232), (64, 1.267, 0.01855), (96, 2.231, 0.06262), (128, 3.958, 0.14844)]
for extra in ("n96_fp32_env_segmented", "n128_fp32_env_segmented"):
    p = ART / f"{extra}.json"
    if p.exists():
        row = load(p)["envelope"]["rows"][0]
        pts = [q for q in pts if q[0] != row["n"]] + [(row["n"], row["peak_rss_gib"], row["field_gib"])]
pts.sort()
n_arr = np.array([p[0] for p in pts], float)
mem = np.array([p[1] for p in pts])
fld = np.array([p[2] for p in pts])
b, a = np.polyfit(fld, mem, 1)  # mem = a + b * field
f256 = 19 * 256**3 * 4 / 2**30

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(fld, mem, "o-", label="measured (VmHWM, fp32, seg_len=1)")
xs = np.linspace(0, f256, 50)
ax[0].plot(xs, a + b * xs, "--", label=f"fit: {a:.2f} + {b:.1f}·field GiB")
ax[0].axhline(8.0, color="crimson", label="criterion: 8 GB")
ax[0].axvline(f256, color="grey", ls=":", label="256³ field = 1.28 GiB")
ax[0].set_xlabel("one distribution function (GiB)")
ax[0].set_ylabel("peak RSS of fwd+bwd (GiB)")
ax[0].set_title("Memory envelope: linear in N³, slope ≈ %.1f fields" % b)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

# ---------------------------------------------------------------- time model
tstep = np.array([0.0191, 0.1575, 0.4757, 1.1133])  # s/step, 6 threads (32,64,96,128)
ax[1].loglog(n_arr, tstep, "s-", label="measured fwd s/step (6 threads)")
ax[1].loglog(n_arr, tstep[0] * (n_arr / n_arr[0]) ** 3, ":", label="N³ scaling")
n256 = 256
pred6 = tstep[-1] * (n256 / 128) ** 3
ax[1].loglog([128, 256], [tstep[-1], pred6], "x--", label=f"extrapolated 256³ = {pred6:.1f} s/step")
for T, lbl in ((50, "T=50 inversion (3× fwd eq.)"),):
    ax[1].axhline(pred6 * T * 3, color="crimson", ls="--", alpha=0.6)
    ax[1].text(34, pred6 * T * 3 * 1.25, f"{lbl}: {pred6*T*3/60:.0f} min", fontsize=8, color="crimson")
ax[1].set_xlabel("lattice resolution N")
ax[1].set_ylabel("wall clock (s)")
ax[1].set_title("Time envelope (16-core run projects ~2× faster than 6 threads)")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT / "envelope_memory_time.png", dpi=150)
print("->", OUT / "envelope_memory_time.png")

# ---------------------------------------------------------------- physics anchor
phys = load(ART / "physics_n32_n64.json")["physics"]["cases"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for case, key, ref, title, ylab in (
    ("taylor_green_2d_in_3d", "E", "E_analytic", "2D Taylor-Green in 3D: E(t)", "kinetic energy E"),
    ("shear_wave_z", "A", "A_analytic", "z shear wave: mode amplitude", "u_x amplitude"),
):
    for c in [c for c in phys if c["case"] == case]:
        t = [s["t"] for s in c["series"]]
        ax[0 if case.startswith("taylor") else 1].plot(t, [s[key] for s in c["series"]], "-", label=f"LBM n={c['n']}")
        ax[0 if case.startswith("taylor") else 1].plot(
            t, [s[ref] for s in c["series"]], "k--", lw=1, label="analytic" if c["n"] == 32 else None
        )
        ax[0 if case.startswith("taylor") else 1].set_title(f"{title} (rel err {100*c['rel_err']:.2f}%)")
        ax[0 if case.startswith("taylor") else 1].set_xlabel("step t")
        ax[0 if case.startswith("taylor") else 1].set_ylabel(ylab)
        ax[0 if case.startswith("taylor") else 1].legend(fontsize=8)
        ax[0 if case.startswith("taylor") else 1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "physics_anchor.png", dpi=150)
print("->", OUT / "physics_anchor.png")