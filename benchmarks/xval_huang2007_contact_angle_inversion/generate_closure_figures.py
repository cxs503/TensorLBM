#!/usr/bin/env python3
"""Figures for the round-2 closure audit of the Huang-2007 contact-angle inversion.

Reads ``artifacts/xval_huang2007_contact_angle_inversion/closure_result.json``
(written by ``run_closure.py``) and produces:

``fig1_calibration.png``  measured theta(f_w) on both grids vs the Young relation
``fig2_closure.png``      target vs measured theta for every closure stage, +/-15% band
``fig3_field.png``        one relaxed droplet field with the fitted interface circle

Everything numeric is read from the result file (or recomputed by the same
solver call the driver uses), so no value is transcribed by hand.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "xval_huang2007_contact_angle_inversion"))

from tensorlbm.contact_line.engine import (  # noqa: E402
    RHO_L_DEFAULT,
    RHO_V_DEFAULT,
    Channel,
    WallStepper,
    sessile_drop_state,
)
from tensorlbm.contact_line.theta_fit import interface_points  # noqa: E402
from tensorlbm.sc_surface_tension import psi_exp  # noqa: E402

ART = REPO / "artifacts" / "xval_huang2007_contact_angle_inversion"
FIG = REPO / "benchmarks" / "xval_huang2007_contact_angle_inversion" / "figures"


def young_curve(fw: np.ndarray, psi_l: float, psi_v: float) -> np.ndarray:
    """Young relation theta(f_w) = acos((psi_w - psi_v)/(psi_l - psi_v))."""
    cos_t = np.clip((fw * psi_l - psi_v) / (psi_l - psi_v), -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


def main(name: str = "closure_result") -> int:
    path = ART / f"{name}.json"
    data = json.loads(path.read_text())
    FIG.mkdir(parents=True, exist_ok=True)

    cfg = data["config"]
    psi_fn_l = float(psi_exp(torch.tensor(RHO_L_DEFAULT)))
    psi_fn_v = float(psi_exp(torch.tensor(RHO_V_DEFAULT)))

    # ---------------- fig 1: calibration curves ---------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fw_dense = np.linspace(0.28, 0.82, 200)
    ax.plot(fw_dense, young_curve(fw_dense, psi_fn_l, psi_fn_v), "k--", lw=1.5,
            label=r"Young relation $\cos\theta=(\psi_w-\psi_v)/(\psi_l-\psi_v)$")
    for key, style, colour in (
        ("calibration_gridA", "o-", "tab:blue"),
        ("calibration_gridB", "s-", "tab:red"),
    ):
        cal = data.get(key)
        if not cal:
            continue
        tbl = [r for r in cal.get("table", []) if r.get("status") == "ok"]
        if not tbl:
            continue
        fw = [r["fw"] for r in tbl]
        th = [r["theta_measured"] for r in tbl]
        g = cfg["grid_A"] if key.endswith("gridA") else cfg["grid_B"]
        ax.plot(fw, th, style, color=colour, mfc="white", ms=7,
                label=f"measured ({key[-5:]}: {g['nx']}x{g['ny']}, R={g['radius']:g})")
    ax.axhline(90.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"wall pseudopotential fraction $f_w=\psi_w/\psi_l$")
    ax.set_ylabel(r"contact angle $\theta$ [deg]")
    ax.set_title("Huang-2007 calibration: measured vs Young-predicted $\\theta(f_w)$")
    ax.set_xlim(0.25, 0.85)
    ax.set_ylim(0, 185)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_calibration.png", dpi=150)
    plt.close(fig)

    # ---------------- fig 2: closure scatter ------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    lim = [40, 150]
    ax.plot(lim, lim, "k-", lw=1.0, label=r"$\theta_{\rm meas}=\theta_{\rm target}$")
    tg = np.array(lim)
    ax.fill_between(tg, tg * 0.85, tg * 1.15, color="tab:green", alpha=0.10,
                    label=r"$\pm15\%$ (owner criterion)")
    styles = {
        "in_grid_gridA": ("o", "tab:blue", "in-grid closure (cal + run on grid A)"),
        "transfer_gridB_with_calA": ("^", "tab:orange", "grid-A calibration applied on grid B"),
        "recal_gridB": ("s", "tab:red", "recalibrated + closed on grid B"),
    }
    for stage, rows in data["stages"].items():
        marker, colour, label = styles.get(stage, ("x", "gray", stage))
        xs = [r["theta_target"] for r in rows["rows"] if math.isfinite(r.get("theta_measured", float("nan")))]
        ys = [r["theta_measured"] for r in rows["rows"] if math.isfinite(r.get("theta_measured", float("nan")))]
        ys_alt = [r.get("theta_measured_alt", float("nan")) for r in rows["rows"]]
        if xs:
            ax.plot(xs, ys, marker, color=colour, ms=9, label=label)
        for x, y, ya in zip(xs, ys, [v for v in ys_alt if isinstance(v, (int, float))]):
            if isinstance(ya, (int, float)) and math.isfinite(ya):
                ax.plot([x, x], [y, ya], color=colour, lw=1.2, alpha=0.55, zorder=1)
            ax.annotate(f"{stage.split('_')[0]}\n{(y - x) / x * 100:+.0f}%", (x, y),
                        textcoords="offset points", xytext=(6, -12), fontsize=7, color=colour)
    ax.set_xlabel(r"$\theta_{\rm target}$ [deg]")
    ax.set_ylabel(r"$\theta_{\rm measured}$ [deg]")
    ax.set_title("Closure test: prescribe $\\theta$ via Huang-2007 inversion, then measure it\n"
                 "(thin bars = spread between the two independent estimators)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_closure.png", dpi=150)
    plt.close(fig)

    # ---------------- fig 3: one relaxed field ----------------------------
    row = next((r for r in data["stages"]["in_grid_gridA"]["rows"]
                if math.isfinite(r.get("theta_measured", float("nan")))), None)
    if row is not None:
        g = cfg["grid_A"]
        steps = int(cfg["steps"])
        torch.set_num_threads(4)
        # one relaxation with the same solver configuration the driver used
        channel = Channel(nx=g["nx"], ny=g["ny"], wall_rows=g["wall_rows"], device="cpu")
        solid = channel.solid_mask()
        f, _ = sessile_drop_state(channel, radius=g["radius"], height=0.0)
        st = WallStepper(channel, G=5.0, tau=1.0, rho_l=RHO_L_DEFAULT, rho_v=RHO_V_DEFAULT,
                         psi_fn=psi_exp, wall_fraction=row["fw"])
        for _ in range(steps):
            f = st.step(f)
        rho = st.density(f).numpy()
        solid_np = solid.numpy()
        mid = 0.5 * (RHO_L_DEFAULT + RHO_V_DEFAULT)
        pts = interface_points(torch.as_tensor(rho), mid, solid).numpy()

        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        masked = np.where(solid_np, np.nan, rho)
        im = ax.imshow(masked, origin="lower", cmap="viridis", aspect="equal")
        ax.contour(np.linspace(0, rho.shape[1] - 1, rho.shape[1]),
                   np.linspace(0, rho.shape[0] - 1, rho.shape[0]),
                   np.where(solid_np, np.nan, rho), levels=[mid], colors="w", linewidths=1.2)
        if pts.size:
            ax.plot(pts[:, 0], pts[:, 1], ".", color="tab:red", ms=1.2, alpha=0.7,
                    label=f"interface (mid-density contour, {len(pts)} pts)")
        ax.set_title(f"relaxed sessile drop, grid A, $f_w$={row['fw']:.4f} "
                     f"$\\rightarrow$ $\\theta$={row['theta_measured']:.1f}$\\deg$ "
                     f"(target {row['theta_target']:.0f}$\\deg$)")
        ax.set_xlabel("x [lattice]")
        ax.set_ylabel("y [lattice]")
        ax.legend(fontsize=8, loc="upper right")
        fig.colorbar(im, ax=ax, label=r"$\rho$")
        fig.tight_layout()
        fig.savefig(FIG / "fig3_field.png", dpi=150)
        plt.close(fig)

    print(f"figures written to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "closure_result"))