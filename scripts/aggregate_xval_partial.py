#!/usr/bin/env python3
"""Rebuild results.json (and the report figures) from saved campaign payloads."""
from __future__ import annotations
import glob, json, os
import torch
from tensorlbm.xval_mpi_ghost_fused.campaign import ARMS, write_artifacts
from tensorlbm.xval_mpi_ghost_fused.case import TGV, tgv_error

OUT = os.path.join("xval", "XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED")
results = []
for path in sorted(glob.glob(os.path.join(OUT, "decomposed_*.pt"))):
    payload = torch.load(path, weights_only=False)
    cfg = payload["cfg"]
    fp = os.path.join(OUT, f"fields_{cfg['tag']}.pt")
    if not os.path.exists(fp):
        continue
    fields = torch.load(fp, weights_only=False)
    case = TGV(n=cfg["n"], u0=cfg["u0"], tau=cfg["tau"],
               dtype=torch.float64 if cfg["dtype"] == "float64" else torch.float32)
    serial_field = fields["serial_field"]
    arms = {}
    for name, _, _ in ARMS:
        rec = dict(payload["arms"][name])
        field = fields[f"field_{name}"]
        arms[name] = {
            "step_ms": rec["step_ms"], "exchange_ms": rec["exchange_ms"],
            "pack_ms": rec["pack_ms"], "unpack_ms": rec["unpack_ms"],
            "mlups": rec["mlups"], "bytes_per_step": rec["bytes_per_step"],
            "bytes_per_step_per_rank": rec["bytes_per_step_per_rank"],
            "series": rec["series"], "byte_accounting": rec["byte_accounting"],
            **tgv_error(field, case, cfg["steps"]),
            "identity_max_abs_vs_serial": float((field - serial_field).abs().max()),
        }
    for name in arms:
        arms[name]["identity_max_abs_vs_selective"] = float(
            (fields[f"field_{name}"] - fields["field_selective_persistent"]).abs().max())
    results.append({"tag": cfg["tag"], "cfg": cfg,
                    "serial": {"step_ms": None, "mlups": None, "series": {},
                               **tgv_error(serial_field, case, cfg["steps"])},
                    "arms": arms})
write_artifacts(results, OUT)
print("tags:", [r["tag"] for r in results])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
figs = os.path.join(OUT, "figs"); os.makedirs(figs, exist_ok=True)
labels = [n for n, _, _ in ARMS]
fig, ax = plt.subplots(figsize=(7, 4))
for off, r in enumerate(results):
    ax.bar([i + 0.18 * (off - (len(results) - 1) / 2) for i in range(4)],
           [100 * r["arms"][n]["tgv_rel_l2"] for n in labels], width=0.17, label=r["tag"])
for lvl in (15.0, 5.0, 3.0):
    ax.axhline(lvl, linestyle="--", linewidth=1.0, label=f"{lvl:g}%")
ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("TGV rel-L2 error [%]"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(figs, "fig2_error_vs_analytic.png"), dpi=140); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, key, title, sc in ((axes[0], "bytes_per_step", "bytes/step (2 faces)", 1/1024),
                             (axes[1], "exchange_ms", "exchange ms/step", 1.0)):
    for off, r in enumerate(results):
        axis.bar([i + 0.19 * (off - (len(results) - 1) / 2) for i in range(4)],
                 [r["arms"][n][key] * sc for n in labels], width=0.18, label=r["tag"])
    axis.set_yscale("log"); axis.set_xticks(range(4)); axis.set_xticklabels(labels, rotation=15)
    axis.set_title(title); axis.legend(fontsize=7); axis.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(figs, "fig3_transport_cost.png"), dpi=140); plt.close(fig)

if "n64_r2" in [r["tag"] for r in results]:
    f = torch.load(os.path.join(OUT, "fields_n64_r2.pt"), weights_only=False)
    from tensorlbm.xval_mpi_ghost_fused.case import macroscopics
    _, uxs, _, _ = macroscopics(f["serial_field"], torch.float32)
    _, uxd, _, _ = macroscopics(f["field_selective_persistent"], torch.float32)
    d = (uxd - uxs).abs(); mid = uxs.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im = axes[0].imshow(uxs[mid].numpy(), origin="lower", cmap="RdBu_r")
    axes[0].set_title(r"$u_x$ mid-$z$, monolithic, t=400"); fig.colorbar(im, ax=axes[0], shrink=.85)
    im2 = axes[1].imshow(d[mid].numpy(), origin="lower", cmap="magma", vmin=0, vmax=1e-8)
    axes[1].set_title(f"|selective - monolithic| max={float(d.max()):.1e}")
    fig.colorbar(im2, ax=axes[1], shrink=.85)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "fig4_field_identity.png"), dpi=140); plt.close(fig)
json.dump(sorted(glob.glob(os.path.join(figs, "*.png"))), open(os.path.join(OUT, "figures.json"), "w"))
print("figures done")
