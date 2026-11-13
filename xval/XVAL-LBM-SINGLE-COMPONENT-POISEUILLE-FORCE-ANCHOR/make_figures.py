#!/usr/bin/env python3
"""Figures + embedded base64 payload for the anchor report (results.json -> PNG)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"


def _load():
    return json.loads((HERE / "results.json").read_text())


def main() -> int:
    FIG.mkdir(exist_ok=True)
    data = _load()
    cases = data["cases"]
    by_id = {c["case_id"]: c for c in cases}
    payload = {"figures": []}

    def save(fig, name, caption):
        path = FIG / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        payload["figures"].append(
            {"name": name, "caption": caption,
             "png_base64": base64.b64encode(path.read_bytes()).decode()}
        )

    # ---- fig1: base-case profiles -----------------------------------------
    base = {c["arm"]: c for c in cases if c["tag"] == "factorial_base"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm, style in (("AO_OS", "o"), ("AR_OS", "s"), ("AR_RS", "^")):
        if arm in base:
            p = base[arm]["profile"]
            ax.plot(np.asarray(p["u_phys"]), p["y"], style + "-", ms=3, lw=1,
                    label=f"{arm}  err={base[arm]['metrics']['rel_err_u_max_pct']:.3f}%")
    ref = base.get("AO_OS") or base.get("AR_RS")
    if ref:
        ny, nu, F = ref["ny"], (ref["tau"] - 0.5) / 3.0, ref["force"]
        y = np.arange(1, ny - 1, dtype=float)
        ue = F / (2 * nu) * (y - 0.5) * ((ny - 1.5) - y)
        ax.plot(ue, y, "k--", lw=1, label="analytic (halfway BB walls)")
    ax.set_xlabel("$u_x$ (lattice units)")
    ax.set_ylabel("y (lattice units)")
    ax.set_title("Single-component force-driven Poiseuille, ny=32, tau=0.8")
    ax.legend(fontsize=7)
    save(fig, "fig1_profile_base.png",
         "Base-case mean profile: OSS-transplant (AO_OS), repo path with ported forcing "
         "(AR_OS) and the repository's own convention (AR_RS) against the closed-form "
         "parabola; all three coincide within 0.15% of u_max.")

    # ---- fig2: grid convergence -------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for arm, style in (("AO_OS", "o-"), ("AR_RS", "^-"), ("AR_OS", "s-")):
        pts = sorted(
            (c["ny"], c["metrics"]["rel_err_u_max_pct"])
            for c in cases
            if c["arm"] == arm and c["tag"] in ("grid_sweep", "factorial_base")
        )
        if len(pts) >= 2:
            h = np.array([p[0] - 2 for p in pts], float)
            e = np.array([p[1] for p in pts], float)
            ax.loglog(h, e, style, ms=4, label=arm)
    ax.set_xlabel("channel height H = ny-2 (lattice units)")
    ax.set_ylabel("|u_max error| (%)")
    ax.set_title("Grid convergence of the force anchor")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "fig2_convergence.png",
         "u_max error versus channel height (log-log) for the three arms; second-order "
         "decay means the anchor error is a discretisation error, not a model error.")

    # ---- fig3: tau sensitivity + forcing-convention difference ------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for arm, style in (("AO_OS", "o-"), ("AR_RS", "^-"), ("AR_OS", "s-")):
        pts = sorted(
            (c["tau"], c["metrics"]["rel_err_u_max_pct"])
            for c in cases
            if c["arm"] == arm and c["tag"] in ("tau_sweep", "factorial_base")
        )
        if len(pts) >= 2:
            t = [1.0 / p[0] for p in pts]
            e = [p[1] for p in pts]
            order = np.argsort(t)
            axes[0].plot(np.array(t)[order], np.array(e)[order], style, ms=4, label=arm)
    axes[0].set_xlabel(r"relaxation rate  $\omega = 1/\tau$")
    axes[0].set_ylabel("|u_max error| (%)")
    axes[0].set_title("Relaxation-rate sensitivity")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    ut = sorted({c["u_target"] for c in cases if c["tag"] in ("force_sweep", "factorial_base")})
    for arm, style in (("AO_OS", "o-"), ("AR_RS", "^-"), ("AR_OS", "s-")):
        e = [
            c["metrics"]["rel_err_u_max_pct"]
            for u in ut
            for c in cases
            if c["arm"] == arm and abs(c["u_target"] - u) < 1e-15
            and c["tag"] in ("force_sweep", "factorial_base")
        ]
        if len(e) == len(ut):
            axes[1].plot(np.array(ut) ** 2, e, style, ms=4, label=arm)
    axes[1].set_xlabel(r"$u_{target}^2$  (force magnitude, $\propto F^2$)")
    axes[1].set_ylabel("|u_max error| (%)")
    axes[1].set_title("Force/Mach sensitivity")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    save(fig, "fig3_sensitivity.png",
         "Left: u_max error versus BGK relaxation rate (omega = 1.818 ... 0.667); "
         "right: error versus force magnitude squared, showing the forcing-convention "
         "effect is a second-order (Ma^2) effect.")

    # ---- fig4: profile residual / shape invariant -------------------------
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for arm, style in (("AO_OS", "o-"), ("AR_RS", "^-"), ("AR_OS", "s-")):
        if arm in base:
            m = base[arm]["metrics"]
            ax.plot([m["u_max_exact_nodes"] * 1e4], [m["umax_over_umean"]], style, ms=6,
                    label=f"{arm}: $u_{{max}}/u_{{mean}}$={m['umax_over_umean']:.5f}")
    ax.axhline(1.5, color="k", ls="--", lw=1, label="exact parabola 1.5")
    ax.set_xlabel("$u_{max}$ (1e-4 lattice units)")
    ax.set_ylabel("$u_{max}/u_{mean}$")
    ax.set_title("Shape invariant (insensitive to force/viscosity errors)")
    ax.legend(fontsize=8)
    save(fig, "fig4_shape_invariant.png",
         "u_max/u_mean versus u_max: the exact parabola gives 1.5, so this isolates a "
         "profile-shape error (wall modelling) from an amplitude error (viscosity).")

    (HERE / "figures_b64.json").write_text(json.dumps(payload))
    print("wrote", len(payload["figures"]), "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())