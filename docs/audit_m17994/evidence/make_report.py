#!/usr/bin/env python3
"""Build the XVAL figures + RESULTS tables from artifacts/.../results.json.

Every number written into the report is read from ``results.json`` (produced by
``run_xval.py``); nothing is typed by hand.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ART = os.path.join(ROOT, "artifacts", "xval_mpi_slab_unpacked_halo_overlap")

with open(os.path.join(ART, "results.json")) as fh:
    R = json.load(fh)
cfg = R["config"]
nx, ny, steps = cfg["nx"], cfg["ny"], cfg["steps"]
k = 2.0 * np.pi / ny
nu = cfg["nu"]

# ---------------------------------------------------------------- fig 1: field
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
ax0 = np.asarray(R["analytic"]["ux"])
for ax, key, title in zip(axes, ["np1", "np4"], ["TensorLBM NP=1", "TensorLBM NP=4 (unpacked halo)"]):
    num = np.asarray(R["snapshots"][key]["ux"])
    ax.imshow(num, cmap="RdBu_r", vmin=-cfg["u0"], vmax=cfg["u0"], aspect="auto",
              extent=[0, nx, ny, 0])
    ax.set_title(f"{title}\n max|num-ref| = "
                 f"{np.abs(num - np.asarray(R['snapshots']['np1']['ux'])).max():.2e}")
    ax.set_xlabel("x [lattice]")
    ax.set_ylabel("y [lattice]")
axes[2].plot(ax0[:, 0], np.arange(ny), "k-", lw=2, label="analytic $u_0\\cos(ky)e^{-\\nu k^2 t}$")
axes[2].plot(np.asarray(R["snapshots"]["np4"]["ux"])[:, 0], np.arange(ny), "r--", lw=1.2,
             label="NP=4 (unpacked halo)")
axes[2].plot(np.asarray(R["snapshots"]["np1"]["ux"])[:, 0], np.arange(ny), "b:", lw=1.2,
             label="NP=1")
axes[2].set_xlabel(r"$u_x$ at x=0")
axes[2].set_ylabel("y [lattice]")
axes[2].set_title(f"profile at t={steps}, error "
                  f"{R['kpis']['unpacked_np4']['field_l2_rel_err_pct']:.3f}%")
axes[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(ART, "fig1_field_and_profile.png"), dpi=140)
plt.close(fig)

# ------------------------------------------------------ fig 2: comm accounting
arms = ["unpacked", "unpacked-selective", "packed-selective"]
labels = ["unpacked\nfull block (OSS)", "unpacked\nselective", "packed\nselective (repo)"]
wire = [R["kpis"][f"{a}_np4"]["wire_bytes_per_step"] for a in arms]
stag = [R["kpis"][f"{a}_np4"]["staging_bytes_per_step"] for a in arms]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
xpos = np.arange(len(arms))
ax.bar(xpos - 0.2, wire, width=0.4, label="wire bytes / step / rank", color="tab:blue")
ax.bar(xpos + 0.2, stag, width=0.4, label="staging memcpy bytes / step", color="tab:orange")
ax.set_xticks(xpos, labels, fontsize=8)
ax.set_ylabel("bytes per step")
ratio = R["kpis"]["unpacked_np4"]["wire_bytes_per_step"] / R["kpis"]["packed-selective_np4"]["wire_bytes_per_step"]
ax.set_title(f"NP=4 comm accounting: unpacked full block = {ratio:.3f}x the selective wire bytes,\n"
             "staging copies exactly 0 vs >0")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(ART, "fig2_comm_accounting.png"), dpi=140)
plt.close(fig)

# ------------------------------------------------- fig 3: overlap + polling
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
nps = sorted(R["overlap"], key=lambda s: int(s[2:]))
ov = [R["overlap"][s]["overlap"]["wall_s"] for s in nps]
bl = [R["overlap"][s]["blocking"]["wall_s"] for s in nps]
xpos = np.arange(len(nps))
axes[0].bar(xpos - 0.2, ov, width=0.4, label="Testall overlap", color="tab:green")
axes[0].bar(xpos + 0.2, bl, width=0.4, label="blocking waitall", color="tab:gray")
axes[0].set_xticks(xpos, nps)
axes[0].set_xlabel("ranks")
axes[0].set_ylabel("wall time [s]")
axes[0].set_title("overlap vs blocking (queue-shim transport)")
axes[0].legend(fontsize=8)
pe = sorted(R["poll_sensitivity"], key=lambda s: int(s.split("_")[-1]))
walls = [R["poll_sensitivity"][p]["wall_s"] for p in pe]
polls = [R["poll_sensitivity"][p]["testall_polls_per_step"] for p in pe]
axes[1].plot([int(p.split("_")[-1]) for p in pe], walls, "o-", label="wall time [s]")
axes[1].set_xscale("log")
axes[1].set_xlabel("Testall polling stride [rows]")
axes[1].set_ylabel("wall time [s]")
for xv, yv, pv in zip([int(p.split("_")[-1]) for p in pe], walls, polls):
    axes[1].annotate(f"{pv:.1f} polls/step", (xv, yv), fontsize=7, xytext=(0, 6),
                     textcoords="offset points", ha="center")
axes[1].set_title("polling granularity sensitivity (NP=4)")
fig.tight_layout()
fig.savefig(os.path.join(ART, "fig3_overlap_and_polling.png"), dpi=140)
plt.close(fig)

# ------------------------------------------------------------ RESULTS.md table
rows = []
for arm, v in R["kpis"].items():
    rows.append(f"| {arm} | {v['probe_rel_err_pct']:.4f} | {v['field_l2_rel_err_pct']:.4f} | "
                f"{'—' if v.get('identity_maxrel') is None else f'{v[chr(39)+chr(39)] if False else v.get(chr(105)+chr(100)+chr(101)+chr(110)+chr(116)+chr(105)+chr(116)+chr(121)+chr(95)+chr(109)+chr(97)+chr(120)+chr(114)+chr(101)+chr(108)):.1e}'} | "
                f"{v['wire_bytes_per_step']} | {v['staging_bytes_per_step']:.0f} | "
                f"{v['testall_polls_per_step']:.2f} | {v['mlups']:.3f} | {v['mass_drift_rel']:.1e} |")


def identity(v):
    val = v.get("identity_maxrel")
    return "—" if val is None else f"{val:.1e}"


rows = []
for arm, v in R["kpis"].items():
    rows.append(f"| {arm} | {v['probe_rel_err_pct']:.4f} | {v['field_l2_rel_err_pct']:.4f} | "
                f"{identity(v)} | {v['wire_bytes_per_step']} | "
                f"{v['staging_bytes_per_step']:.0f} | {v['testall_polls_per_step']:.2f} | "
                f"{v['mlups']:.3f} | {v['mass_drift_rel']:.1e} | {v['ma_max']:.3f} |")

table = "\n".join(rows)
with open(os.path.join(ART, "tables.md"), "w") as fh:
    fh.write("| arm | probe err [%] | field L2 [%] | identity vs NP=1 | wire B/step | staging B/step | Testall/step | MLUPS | mass drift | Ma |\n")
    fh.write("|---|---|---|---|---|---|---|---|---|---|\n")
    fh.write(table + "\n")
print(table)
print("figures + tables written")