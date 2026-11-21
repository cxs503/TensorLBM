"""Generate T5 solo report figures (depth comparison + mid-plane snapshot)."""
import base64
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MINE = "/tmp/cj5662x/cj5662/repo/runs/t5_solo/t5_combined8.json"
EXP_CSV = "benchmarks/water_entry/single_drop/fig5_experiment.csv"
PLANES = "/tmp/hist/case3_planes.npz"

# ---- load my run history ----
d = json.load(open(MINE))
h = d["history"]
ts = np.array([r["time_ms"] for r in h])
ds = np.array(
    [r["depth_over_D"] if r.get("depth_over_D") is not None else np.nan for r in h]
)

# ---- load experiment points ----
exp_t, exp_d = [], []
for line in open(EXP_CSV):
    line = line.strip()
    if not line or line.startswith("#") or "x_px" in line:
        continue
    parts = line.split(",")
    if len(parts) >= 4:
        try:
            exp_t.append(float(parts[2]))
            exp_d.append(float(parts[3]))
        except ValueError:
            pass
exp_t = np.array(exp_t)
exp_d = np.array(exp_d)

# ---- Figure 1: depth time series ----
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
valid = ~np.isnan(ds)
ax.plot(ts[valid], ds[valid], "-", color="tab:blue", lw=1.6, label="sim (combined, 8 cells/D)")
ax.plot(exp_t, exp_d, "ko", ms=5, label="experiment (Santini case-3)")
ax.axvline(10.39446, color="gray", ls=":", lw=1)
ax.text(10.45, 0.1, "peak 10.394 ms", rotation=90, fontsize=8, va="bottom", color="gray")
ax.axhline(1.857434, color="red", ls="--", lw=0.8)
ax.text(0.5, 1.87, "exp peak 1.8574 D", color="red", fontsize=8)
ax.set_xlabel("time (ms)")
ax.set_ylabel("cavity depth / D")
ax.set_title("Single-drop deep-water entry: cavity depth (T5 solo, cells=8)")
ax.set_xlim(0, 18)
ax.set_ylim(0, 2.1)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png")
fig1_b64 = base64.b64encode(buf.getvalue()).decode()

# ---- Figure 2: mid-plane fill at 10.394 ms ----
p = np.load(PLANES)
idx = int(np.argmin(np.abs(p["time_ms"] - 10.39446)))
fill = p["fill"][idx]
x = p["x_over_D"]
z = p["z_over_D"]
fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
X, Z = np.meshgrid(x, z)
cs = ax.contourf(X, Z, fill, levels=20, cmap="Blues")
ax.contour(X, Z, fill, levels=[0.5], colors="k", linewidths=1.2)
ax.set_xlabel("x / D")
ax.set_ylabel("z / D (up)")
ax.set_title(f"mid-plane fill @ {p['time_ms'][idx]:.3f} ms (archived combined 8-cell)")
fig.colorbar(cs, ax=ax, label="fill (0=gas, 1=liquid)")
fig.tight_layout()
buf2 = io.BytesIO()
fig.savefig(buf2, format="png")
fig2_b64 = base64.b64encode(buf2.getvalue()).decode()

json.dump(
    {"fig1": fig1_b64, "fig2": fig2_b64, "plane_time_ms": float(p["time_ms"][idx])},
    open("/tmp/t5_figures.json", "w"),
)
print("figures written; plane time", p["time_ms"][idx])
print("exp points:", len(exp_t))
