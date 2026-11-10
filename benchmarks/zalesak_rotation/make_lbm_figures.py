#!/usr/bin/env python
"""Figures for the P190-N01 LBM free-surface Zalesak campaign.

Reads ``artifacts/zalesak_p190_n01/zalesak_lbm_all.json`` and
``fields_all.npz`` (both written by ``run_zalesak_lbm_campaign.py``) and writes

* ``fig01_fields.png``           initial / proposed / VOF final fields + error
* ``fig02_convergence.png``      L1 error, overlap and interface thickness vs N
* ``fig03_mass_and_tau.png``     mass drift history and the relaxation scan
* ``fig04_3d_cross_section.png`` D3Q19 extruded cross-section vs D2Q9

Every plotted number comes from the JSON, so the figures cannot drift from the
reported metrics.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load(out_dir):
    path = os.path.join(out_dir, "zalesak_lbm_all.json")
    if not os.path.exists(path):
        alt = [f for f in os.listdir(out_dir) if f.endswith(".json")]
        if not alt:
            raise SystemExit(f"no campaign JSON in {out_dir}")
        path = os.path.join(out_dir, sorted(alt)[-1])
    with open(path) as fh:
        res = json.load(fh)
    npz_path = os.path.join(out_dir, "fields_all.npz")
    fields = dict(np.load(npz_path)) if os.path.exists(npz_path) else {}
    return res, fields, path


def _field(fields, kind, label):
    key = f"{kind}__{label}"
    if key not in fields:
        raise SystemExit(f"missing field {key}; keys={sorted(fields)}")
    return fields[key]


def _imshow(ax, arr, title, *, vmin=0.0, vmax=1.0, cmap="viridis"):
    im = ax.imshow(arr, origin="lower", extent=(0, 1, 0, 1), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=7)
    return im


def fig_fields(res, fields, out_dir):
    cmp_ = res["compare"]["runs"]
    lbm = next(r for r in cmp_ if r["scheme"] == "lbm")
    vof = next(r for r in cmp_ if r["scheme"] == "vof_muscl")
    lab_lbm, lab_vof = lbm["label"], vof["label"]
    phi0 = _field(fields, "phi_initial", lab_lbm)
    phi_lbm = _field(fields, "phi_final", lab_lbm)
    phi_vof = _field(fields, "phi_final", lab_vof)
    n = phi0.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.6))
    ax = axes.ravel()
    im = _imshow(ax[0], phi0, f"(a) initial fill fraction, N={n}")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    im = _imshow(
        ax[1], phi_lbm,
        f"(b) LBM tau=1/2+{lbm['tau'] - 0.5:.3g} + ledger, N={n}\n"
        f"L1={lbm['l1_shape_error']:.3e}, overlap={lbm['shape_overlap']:.3f}",
    )
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    im = _imshow(
        ax[2], phi_vof,
        f"(c) VOF MUSCL+compression (reference), N={n}\n"
        f"L1={vof['l1_shape_error']:.3e}, overlap={vof['shape_overlap']:.3f}",
    )
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    err = np.abs(phi_lbm - phi0)
    im = _imshow(ax[3], err, f"(d) |phi(LBM) - phi0| after 2*pi, max={err.max():.3f}",
                 vmax=max(err.max(), 1e-6), cmap="magma")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    fig.suptitle("Zalesak slotted cylinder, one revolution, D2Q9 fill transport", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig01_fields.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_convergence(res, out_dir):
    g = res["gci"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    for seq, lab, style in ((g["lbm"], "LBM fill transport", "o-"), (g["vof"], "VOF MUSCL+compression", "s--")):
        ns = [r["n"] for r in seq]
        l1 = [r["l1_shape_error"] for r in seq]
        ov = [r["shape_overlap"] for r in seq]
        th = [r["interface_thickness_mean_cells"] for r in seq]
        axes[0].loglog(ns, l1, style, label=lab, markersize=4)
        axes[1].semilogx(ns, ov, style, label=lab, markersize=4)
        axes[2].loglog(ns, th, style, label=lab, markersize=4)
    o = g["orders_lbm"]
    axes[0].set_xlabel("N (cells per axis)")
    axes[0].set_ylabel("L1 shape error")
    axes[0].set_title(
        f"L1 error (observed order {o['order_finest_pair']:.2f})", fontsize=9
    )
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("Jaccard overlap with initial shape")
    axes[1].set_title("shape fidelity", fontsize=9)
    axes[2].set_xlabel("N")
    axes[2].set_ylabel("10-90% interface thickness [cells]")
    axes[2].set_title("interface sharpness", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig02_convergence.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_mass_and_tau(res, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
    g = res["gci"]
    for seq, lab, style in ((g["lbm"], "LBM fill transport", "o-"), (g["vof"], "VOF reference", "s--")):
        for r in seq:
            hist = r.get("history") or {}
            if not hist.get("step"):
                continue
            axes[0].semilogy(
                hist["step"],
                [abs(x) + 1e-18 for x in hist["mass_rel_drift"]],
                style, lw=1.0, label=f"{lab} N={r['n']}",
            )
    axes[0].set_xlabel("lattice step")
    axes[0].set_ylabel("|relative volume drift|")
    axes[0].set_title("mass ledger", fontsize=9)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=6)
    axes[0].tick_params(labelsize=7)

    scan = res["compare"].get("tau_scan", [])
    if scan:
        taus = [r["tau"] for r in scan]
        axes[1].plot(taus, [r["l1_shape_error"] for r in scan], "o-", label="L1 error")
        ax2 = axes[1].twinx()
        ax2.plot(taus, [r["shape_overlap"] for r in scan], "s--", color="tab:red",
                 label="overlap")
        ax2.set_ylabel("overlap", fontsize=8)
        axes[1].set_xlabel("BGK relaxation rate tau")
        axes[1].set_ylabel("L1 shape error")
        axes[1].set_title("relaxation-rate scan, N=64 (nu = c_s^2 (tau-1/2))", fontsize=9)
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=7, loc="upper left")
        ax2.legend(fontsize=7, loc="lower right")
        axes[1].tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig03_mass_and_tau.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_3d(res, fields, out_dir):
    s = res["stage3d"]
    two, three = s["two_d"], s["three_d_extruded"]
    dev = s.get("cross_section_deviation", {})
    phi2 = _field(fields, "phi_final", two["label"])
    phi3 = _field(fields, "phi_final", three["label"])
    cs = phi3[:, phi3.shape[1] // 2, :]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    im = _imshow(axes[0], phi2, f"(a) D2Q9 final, N={two['n']}\nL1={two['l1_shape_error']:.3e}")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    im = _imshow(
        axes[1], cs,
        f"(b) D3Q19 extruded cross-section y=0.5, N={three['n']}\n"
        f"L1={three['l1_shape_error']:.3e}",
    )
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    d = np.abs(cs - phi2)
    im = _imshow(axes[2], d, f"(c) |D3Q19 - D2Q9|, rel.L1={dev.get('relative_l1', float('nan')):.3f}",
                 vmax=max(float(d.max()), 1e-6), cmap="magma")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle("D2Q9 vs D3Q19 fill transport on the same (y-invariant) body", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig04_3d_cross_section.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/zalesak_p190_n01")
    args = ap.parse_args(argv)
    res, fields, path = _load(args.out)
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    made = []
    if "compare" in res and fields:
        made.append(fig_fields(res, fields, fig_dir))
    if "gci" in res:
        made.append(fig_convergence(res, fig_dir))
        made.append(fig_mass_and_tau(res, fig_dir))
    if "stage3d" in res and fields and "cross_section_deviation" in res["stage3d"]:
        made.append(fig_3d(res, fields, fig_dir))
    print("source:", path)
    for m in made:
        print("wrote", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())