#!/usr/bin/env python3
"""Figures for P189-N06 (Zalesak rotation benchmark) from the campaign JSON.

Produces:

1. ``fig1_initial_condition.png`` -- initial fill fraction + prescribed rotation field
2. ``fig2_shape_overlay.png``    -- phi = 0.5 contours, initial vs after one revolution, + error map
3. ``fig3_arm_comparison.png``   -- operator arms at N = 128: sharpness vs shape fidelity
4. ``fig4_grid_convergence.png`` -- L1 shape error vs N with observed orders + GCI
5. ``fig5_conservation_history.png`` -- volume error and interface thickness vs revolution
6. ``fig6_three_d.png``          -- D3Q19 cross-section vs D2Q9 and 3-D slice profile

All numbers come from ``artifacts/zalesak_p189_n06/zalesak_results.json`` or from
a fresh (deterministic) run of the same library functions; nothing is typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tensorlbm.free_surface_zalesak import (  # noqa: E402
    cross_section,
    cross_section_index,
    rigid_rotation_field,
    run_zalesak_case,
    slotted_cylinder_phi_2d,
    slotted_sphere_phi_3d,
)

ART = ROOT / "artifacts" / "zalesak_p189_n06"
FIG = ART / "figures"


def _load() -> dict:
    return json.loads((ART / "zalesak_results.json").read_text())


def fig1_initial_condition() -> Path:
    n = 128
    phi = slotted_cylinder_phi_2d(n).numpy()
    u = rigid_rotation_field(n, dim=2, omega=1.0)
    step = 8
    ys = (np.arange(0, n, step) + 0.5) / n
    xs = (np.arange(0, n, step) + 0.5) / n
    U = u[1].numpy()[::step, ::step]
    V = u[0].numpy()[::step, ::step]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(phi, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=0, vmax=1)
    ax.quiver(xs, ys, U, V, color="crimson", scale=25, width=0.004)
    ax.plot(0.5, 0.5, "k+", ms=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Initial liquid fraction phi (N={n}) and rotation field")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="phi")
    out = FIG / "fig1_initial_condition.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig2_shape_overlay(payload: dict) -> Path:
    """Contours after one revolution for the selected arm at N = 128."""
    arm = payload["selected_arm"]
    row = payload["stages"]["arms"]["arms"][arm]["metrics"]
    res = run_zalesak_case(
        128, dim=2, u_max_lat=payload["stages"]["arms"]["u_max_lat"],
        c_comp=row["c_comp"], scheme=row["scheme"], limiter=row["limiter"],
        sample_every=100_000,
    )
    p0 = res["phi_initial"].numpy()
    p1 = res["phi_final"].numpy()
    x = (np.arange(128) + 0.5) / 128
    yy, xx = np.meshgrid(x, x, indexing="ij")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    ax = axes[0]
    ax.contour(xx, yy, p0, levels=[0.1, 0.5, 0.9], colors="k", linestyles=["--", "-", "--"])
    ax.contour(xx, yy, p1, levels=[0.5], colors="crimson")
    ax.set_title("phi = 0.5 contour after 1 revolution\n(black = initial, red = final, N=128)")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax = axes[1]
    im = ax.imshow(p1 - p0, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_title(
        f"phi_final - phi_initial\nov(recall)={row['overlap_recall']:.4f}, "
        f"IoU={row['overlap_iou']:.4f}"
    )
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[2]
    ax.imshow(p1, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"final phi, 10-90 thickness = {row['thickness_mean']:.2f} cells\n"
                 f"|dV|/V = {abs(row['volume_rel_change']):.1e}")
    ax.set_aspect("equal")
    out = FIG / "fig2_shape_overlay.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig3_arm_comparison(payload: dict) -> Path:
    arms = payload["stages"]["arms"]["arms"]
    names = list(arms)
    labels = [n.split("_", 1)[0] + "\n" + n.split("_", 1)[1].replace("_", " ") for n in names]
    thick = [arms[k]["metrics"]["thickness_mean"] for k in names]
    iou = [arms[k]["metrics"]["overlap_iou"] for k in names]
    recall = [arms[k]["metrics"]["overlap_recall"] for k in names]
    l1 = [arms[k]["metrics"]["l1_error"] for k in names]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    idx = np.arange(len(names))
    ax = axes[0]
    ax.bar(idx, thick, color="steelblue")
    ax.axhline(3.0, color="crimson", ls="--", label="criterion (b): < 3 cells")
    ax.set_yscale("log")
    ax.set_xticks(idx, labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("10-90 interface thickness [cells]")
    ax.legend(fontsize=8)
    ax.set_title("(b) interface sharpness at N = 128")
    ax = axes[1]
    ax.bar(idx - 0.2, recall, width=0.4, label="overlap (intersection / initial)")
    ax.bar(idx + 0.2, iou, width=0.4, label="Jaccard IoU")
    ax.axhline(0.95, color="crimson", ls="--", label="criterion (c): > 0.95")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(idx, labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("overlap after one revolution")
    ax.legend(fontsize=7)
    ax.set_title("(c) shape fidelity at N = 128")
    ax = axes[2]
    ax.bar(idx, l1, color="darkorange")
    ax.set_yscale("log")
    ax.set_xticks(idx, labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("L1 shape error")
    ax.set_title("L1 error |phi_T - phi_0| at N = 128")
    out = FIG / "fig3_arm_comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig4_grid_convergence(payload: dict) -> Path:
    grid = payload["stages"]["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    colors = {"A5_muscl_superbee_c100": "tab:blue", "A2_muscl_vanleer_c050": "tab:green"}
    for i, (arm, blk) in enumerate(grid.items()):
        ns = sorted(int(k) for k in blk["runs"])
        e = [blk["runs"][str(k)]["metrics"]["l1_error"] for k in ns]
        ax.loglog(ns, e, "o-", color=colors.get(arm, f"C{i}"), label=arm)
        for k0, k1, order in zip(ns[:-1], ns[1:],
                                 [v["p_l1"] for v in blk["convergence"].values()]):
            ax.annotate(f"p={order:.2f}", ((k0 * k1) ** 0.5, (e[ns.index(k0)] * e[ns.index(k1)]) ** 0.5),
                        textcoords="offset points", xytext=(6, 6), fontsize=8,
                        color=colors.get(arm, f"C{i}"))
    ns_ref = np.array(sorted(int(k) for k in grid[payload["selected_arm"]]["runs"]), dtype=float)
    ax.loglog(ns_ref, 0.9 / ns_ref, "k:", lw=1.0, label="slope -1 reference")
    ax.axhline(1e-3, color="crimson", ls="--", lw=0.8)
    ax.set_xlabel("grid N per axis")
    ax.set_ylabel("L1 shape error after one revolution")
    ax.set_title("(d) grid convergence of the shape error")
    ax.legend(fontsize=8)
    ax = axes[1]
    blk = grid[payload["selected_arm"]]
    ns = sorted(int(k) for k in blk["runs"])
    thick = [blk["runs"][str(k)]["metrics"]["thickness_mean"] for k in ns]
    recall = [blk["runs"][str(k)]["metrics"]["overlap_recall"] for k in ns]
    iou = [blk["runs"][str(k)]["metrics"]["overlap_iou"] for k in ns]
    ax.plot(ns, recall, "o-", label="overlap (intersection / initial)")
    ax.plot(ns, iou, "s-", label="Jaccard IoU")
    ax.axhline(0.95, color="crimson", ls="--", label="criterion (c): 0.95")
    ax2 = ax.twinx()
    ax2.plot(ns, thick, "^--", color="gray", label="thickness [cells]")
    ax2.set_ylabel("10-90 interface thickness [cells]")
    ax.set_xlabel("grid N per axis")
    ax.set_ylabel("overlap after one revolution")
    ax.set_xscale("log")
    ax.set_title(f"shape fidelity and sharpness vs N ({payload['selected_arm']})")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
    out = FIG / "fig4_grid_convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig5_conservation_history(payload: dict) -> Path:
    arms = payload["stages"]["arms"]["arms"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    ax = axes[0]
    for name, blk in arms.items():
        h = blk["history"]
        t = [r["time_frac"] for r in h]
        v = [abs(r["volume_rel_err"]) + 1e-20 for r in h]
        ax.semilogy(t, v, label=name.split("_", 1)[0], lw=1.2)
    ax.axhline(5e-3, color="crimson", ls="--", label="criterion (a): 0.5 %")
    ax.set_xlabel("revolution fraction t/T")
    ax.set_ylabel("relative volume error |dV|/V")
    ax.set_title("(a) mass conservation during one revolution (N = 128)")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    for name, blk in arms.items():
        h = blk["history"]
        t = [r["time_frac"] for r in h]
        th = [r["thickness_mean"] for r in h]
        ax.plot(t, th, label=name.split("_", 1)[0], lw=1.2)
    ax.axhline(3.0, color="crimson", ls="--", label="criterion (b): 3 cells")
    ax.set_yscale("log")
    ax.set_xlabel("revolution fraction t/T")
    ax.set_ylabel("10-90 interface thickness [cells]")
    ax.set_title("interface sharpness history (N = 128)")
    ax.legend(fontsize=7, ncol=2)
    out = FIG / "fig5_conservation_history.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig6_three_d(payload: dict) -> Path:
    # Keep the figure run cheap: 48^3 is enough to show the slice structure.
    n = min(48, payload["stages"]["three_d"]["n"])
    arm = payload["stages"]["three_d"]["arm"]
    arms = payload["stages"]["arms"]["arms"][arm]["metrics"]
    phi3 = slotted_sphere_phi_3d(n)
    kw = dict(u_max_lat=payload["stages"]["three_d"]["u_max_lat"],
              c_comp=arms["c_comp"], scheme=arms["scheme"], limiter=arms["limiter"],
              sample_every=100_000)
    res3 = run_zalesak_case(n, dim=3, phi_init=phi3, **kw)
    res2 = run_zalesak_case(n, dim=2, phi_init=cross_section(phi3), **kw)
    sec = cross_section(res3["phi_final"]).numpy()
    ref = res2["phi_final"].numpy()
    x = (np.arange(n) + 0.5) / n
    yy, xx = np.meshgrid(x, x, indexing="ij")
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    ax = axes[0]
    ax.contour(xx, yy, ref, levels=[0.5], colors="k", linestyles="-")
    ax.contour(xx, yy, sec, levels=[0.5], colors="crimson", linestyles="--")
    ax.set_title(f"D2Q9 vs D3Q19 cross-section at y = 0.5\n(N={n}, black = 2-D, red = 3-D sphere slice)")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax = axes[1]
    areas = [float(phi3[:, k].mean()) for k in range(n)]
    ax.plot((np.arange(n) + 0.5) / n, areas, "o-", ms=3)
    ax.axvline((cross_section_index(n) + 0.5) / n, color="crimson", ls="--", label="compared slice")
    ax.set_xlabel("y")
    ax.set_ylabel("slice liquid area")
    ax.set_title("3-D content: liquid area of each y slice\n(the set-up is not y-invariant)")
    ax.legend(fontsize=8)
    ax = axes[2]
    im = ax.imshow(sec - ref, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.set_title(f"3-D slice - 2-D field (mean |dphi| = {np.abs(sec - ref).mean():.4f})")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8)
    out = FIG / "fig6_three_d.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    torch.set_num_threads(1)
    FIG.mkdir(parents=True, exist_ok=True)
    payload = _load()
    made = [fig1_initial_condition(), fig2_shape_overlay(payload), fig3_arm_comparison(payload),
            fig4_grid_convergence(payload), fig5_conservation_history(payload), fig6_three_d(payload)]
    payload.setdefault("figures", [])
    payload["figures"] = [str(p.relative_to(ROOT)) for p in made]
    (ART / "zalesak_results.json").write_text(json.dumps(payload, indent=2))
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()