"""Figures for the P158W3-N01 record (run.py writes the record first)."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, "records", "rotating_w3.json")


def _load_fields(rec):
    """Attach the per-case curve data: from the record when present, else from
    the archived npz written by ``run.py``."""
    npz_path = os.path.join(HERE, "records", "rotating_w3_fields.npz")
    npz = np.load(npz_path) if os.path.exists(npz_path) else None

    def get(case_key, case):
        if "_field_light" in case:
            return {k: np.asarray(v, dtype=float) for k, v in case["_field_light"].items()
                    if isinstance(v, list)}
        pre = case_key + "__"
        return {k[len(pre):]: npz[k] for k in npz.files if k.startswith(pre)}

    for i, c in enumerate(rec["cases"]["couette_ladder"]):
        c["_field_light"] = get(f"couette_{i}", c)
    for c in rec["cases"]["couette_extra"]:
        c["_field_light"] = get(f"couette_{c['label']}", c)
    for e in rec["cases"]["ekman"]:
        e["_field_light"] = get(f"ekman_{e['delta_nodes']}", e)
    return rec


def main() -> int:
    rec = _load_fields(json.load(open(REC)))
    fig_dir = os.path.join(HERE, "..", "..", "..", "docs", "assets", "rotating_w3")
    os.makedirs(fig_dir, exist_ok=True)
    ladder = rec["cases"]["couette_ladder"]
    extra = rec["cases"]["couette_extra"]
    ek = rec["cases"]["ekman"]
    crit = rec["criteria"]

    # ---- fig 1: Couette profiles ---------------------------------------- #
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for c in ladder:
        f = c["_field_light"]
        ax[0].plot(f["r"], f["u_theta"], ".", ms=3, label=f"ri={c['ri']:.0f} (n={c['n_grid']})")
    f0 = ladder[0]["_field_light"]
    ax[0].plot(f0["r"], f0["analytic"], "-", color="k", lw=1.2, label="exact $Ar+B/r$")
    ax[0].set_xlabel("r (lattice)")
    ax[0].set_ylabel(r"$u_\theta$")
    ax[0].set_title("Taylor-Couette: shell-averaged profile")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for c, mk in zip(extra, ("s", "o")):
        f = c["_field_light"]
        ax[1].plot(f["r"], f["u_theta"], mk, ms=3, label=f"{c['label']}")
        ax[1].plot(f["r"], f["analytic"], "-", lw=1.0)
    ax[1].set_xlabel("r (lattice)")
    ax[1].set_title("differential cases (points) vs exact (lines)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_couette_profiles.png"), dpi=150)
    plt.close(fig)

    # ---- fig 2: grid convergence ---------------------------------------- #
    r = np.array([c["ri"] for c in ladder])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].loglog(r, [c["err_nominal_10pt"] for c in ladder], "o-", label="nominal radii (staircase)")
    ax[0].loglog(r, [c["err_effective_10pt"] for c in ladder], "s-", label="effective radii (R$_{eff}$)")
    ax[0].loglog(r, [c["shape_residual_bulk"] for c in ladder], "^-", label="bulk shape residual")
    ax[0].loglog(r, [c["CM_rel_err"] for c in ladder], "d-", label="C$_M$ error")
    ax[0].axhline(0.02, color="r", ls=":", lw=1, label="A1 threshold 2%")
    ax[0].set_xlabel("gap refinement ri (eta = 1/2)")
    ax[0].set_ylabel("relative error")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")
    ax[0].set_title("A1/A2/A6 convergence ladder")
    for c in ladder:
        ax[1].semilogx(c["ri"], c["dr_i_eff"], "o", color="C0")
        ax[1].semilogx(c["ri"], c["dr_o_eff"], "s", color="C1")
    ax[1].set_xlabel("gap refinement")
    ax[1].set_ylabel(r"$r_{eff}-r_{nom}$ (lattice)")
    ax[1].set_title("effective-wall offset (staircase, half link)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_grid_convergence.png"), dpi=150)
    plt.close(fig)

    # ---- fig 3: co-rotation null test ----------------------------------- #
    co = [c for c in extra if c["label"].startswith("co_rotating")][0]
    diff = [c for c in extra if c["label"] == "counter_rotating"][0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    f = co["_field_light"]
    ax[0].plot(f["r"], f["u_theta"], "o", ms=3, label="co-rotating (LBM)")
    om = co["omega_i"]
    ax[0].plot(f["r"], om * f["r"], "-", lw=1.2, label=r"rigid body $\Omega r$")
    ax[0].set_xlabel("r")
    ax[0].set_ylabel(r"$u_\theta$")
    ax[0].set_title(f"A7 co-rotating null test (err {co['err_nominal_10pt']*100:.2f}%)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    labels = ["counter-rotating", "co-rotating"]
    vals = [abs(diff["torque_mid"]), abs(co["torque_mid"])]
    ax[1].bar(labels, vals, color=["C0", "C3"])
    ax[1].axhline(abs(diff["torque_analytic"]), color="k", ls="--", lw=1,
                  label="analytic |T| (differential)")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("|T| from the profile identity")
    ax[1].set_title(f"A7: ratio co/differential = {vals[1]/vals[0]:.2e}")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_corotating_null.png"), dpi=150)
    plt.close(fig)

    # ---- fig 4: Ekman spiral hodographs + 45 deg extrapolation ----------- #
    big = [e for e in ek if e["delta_nodes"] >= 6.0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for e in big:
        f = e["_field_light"]
        uz = np.array(f["uz"]) / max(abs(np.array(f["uz"])).max(), 1e-30)
        ux = np.array(f["ux"]) / max(abs(np.array(f["uz"])).max(), 1e-30)
        ax[0].plot(ux, uz, ".-", ms=3, lw=1, label=f"$\\delta$={e['delta_nodes']:.0f}")
        uza = np.array(f["uz_analytic"]) / max(abs(np.array(f["uz_analytic"])).max(), 1e-30)
        uxa = np.array(f["ux_analytic"]) / max(abs(np.array(f["ux_analytic"])).max(), 1e-30)
        ax[0].plot(uxa, uza, "k-", lw=0.6, alpha=0.6)
    ax[0].set_xlabel(r"$u_x/u_{z,geo}$")
    ax[0].set_ylabel(r"$u_z/u_{z,geo}$")
    ax[0].set_title("Ekman hodographs (points) vs exact (black)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    th = np.array([e["spiral_angle_deg"] for e in big])
    inv = np.array([1.0 / e["delta_nodes"] for e in big])
    ax[1].plot(inv, th, "o-", label="measured wall-shear angle")
    ax[1].plot(inv, [e["spiral_angle_exact_discrete_deg"] for e in big], "x--",
               label="exact profile, same operator")
    xf = np.linspace(0.0, inv.max() * 1.05, 20)
    ax[1].plot(xf, crit["A5_ekman_spiral_angle_deg"]["intercept_deg"]
               + crit["A5_ekman_spiral_angle_deg"]["slope_deg"] * xf, ":",
               label=f"extrapolation -> {crit['A5_ekman_spiral_angle_deg']['intercept_deg']:.2f} deg")
    ax[1].axhline(45.0, color="r", ls="--", lw=1, label="theory 45 deg")
    ax[1].set_xlabel(r"$1/\delta$ (nodes$^{-1}$)")
    ax[1].set_ylabel("spiral angle (deg)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    ax[1].set_title("A5: spiral angle vs layer resolution")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_ekman_spiral.png"), dpi=150)
    plt.close(fig)

    # ---- fig 5: Ekman delta at the resolution limit ---------------------- #
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    dn = np.array([e["delta_nodes"] for e in ek])
    err = np.array([e["delta_rel_err"] for e in ek])
    ax.semilogy(dn, err, "o-")
    ax.axhline(0.05, color="r", ls="--", lw=1, label="A4 threshold 5%")
    ax.axhline(0.01, color="gray", ls=":", lw=1, label="1% (verified range)")
    ax.set_xlabel(r"$\delta_E$ (lattice nodes)")
    ax.set_ylabel(r"$|\delta_{fit}/\delta_{analytic}-1|$")
    ax.set_title("A4: Ekman thickness vs layer resolution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig5_ekman_resolution.png"), dpi=150)
    plt.close(fig)

    # ---- fig 6: rotation-rate stability boundary ------------------------- #
    st = rec["cases"]["stability_sweep"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    om = np.array([s["omega_frame"] for s in st])
    ok = np.array([s["stable"] for s in st])
    ax[0].semilogy(om[ok], [(s["residual"] if s["residual"] else np.nan) for s in st if s["stable"]],
                   "o-", label="stable residual")
    ax[0].semilogy(om[~ok], [s["centrifugal_over_cs2"] for s in st if not s["stable"]], "rx",
                   label="unstable (pseudo-force/ cs$^2$)")
    ax[0].set_xlabel(r"$\Omega_f$ (lattice)")
    ax[0].set_ylabel("residual / force scale")
    ax[0].set_title("frame-rate sweep")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    xs = [(s["ekman_number"] if s["ekman_number"] is not None else np.nan) for s in st]
    ax[1].plot(xs, [s["centrifugal_over_cs2"] for s in st], "o-")
    for s, xv in zip(st, xs):
        if xv is None or not np.isfinite(xv):
            continue
        ax[1].annotate(f"{s['omega_frame']:.3g}", (xv, s["centrifugal_over_cs2"]),
                       fontsize=7, textcoords="offset points", xytext=(3, 3))
    ax[1].axhline(1.0, color="r", ls="--", lw=1, label=r"$\Omega_f^2 r = c_s^2$")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Ekman number $\\nu/(\\Omega_f r^2)$")
    ax[1].set_ylabel(r"$\Omega_f^2 r/c_s^2$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig6_stability.png"), dpi=150)
    plt.close(fig)

    print("figures written to", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
