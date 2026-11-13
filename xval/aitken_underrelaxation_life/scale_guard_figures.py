"""Figures for the Aitken guard scaling audit (machine-read from
``out/scale_guard.json`` -- no hand-copied numbers)."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def main() -> int:
    data = json.load(open(os.path.join(OUT, "scale_guard.json")))
    rows = data["driver_ladder"]
    arms = [
        ("shipped_after_patch", "default, omega_min=0 (scale free)", "o", "-"),
        ("raw_no_clip", "clip=False (LIFE raw)", "s", "--"),
        ("pre_patch_floor_1e-4", "m17727 floor 1e-4", "^", "-."),
        ("pre_patch_floor_5e-2", "original floor 5e-2", "v", ":"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax = axes[0]
    for key, label, marker, ls in arms:
        sel = [r for r in rows if r["arm"] == key]
        beta = np.array([r["beta_eff"] for r in sel])
        it = np.array([r["iters_mean"] if r["converged"] else np.nan for r in sel])
        ax.loglog(beta, it, marker=marker, ls=ls, ms=4, lw=1.4, label=label)
        bad = [r for r in sel if not r["converged"]]
        if bad:
            ax.loglog([r["beta_eff"] for r in bad], [3.0] * len(bad), "x",
                      color="crimson", ms=12, mew=2.2, ls="none",
                      label="diverged / not converged" if key == "pre_patch_floor_1e-4" else None)
    ax.axvline(9999.0, color="grey", ls=":", lw=1.0)
    ax.annotate(r"$\beta=1/\omega_{min}-1=9999$" + "\n(m17727 floor 1e-4)",
                xy=(9999.0, 2.0), xytext=(300.0, 12.0), fontsize=8, color="grey")
    ax.axvline(39.0, color="grey", ls=":", lw=1.0)
    ax.annotate(r"$\beta=39$ (floor 5e-2)", xy=(39.0, 2.0), xytext=(4.0, 6.0),
                fontsize=8, color="grey")
    ax.set_xlabel(r"added-mass ratio $\beta_{eff}=m_a/m_{eff}$")
    ax.set_ylabel("sub-iterations per time step")
    ax.set_title("(a) driver: iterations & divergence thresholds")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper center")

    ax = axes[1]
    beta = np.unique([r["beta_eff"] for r in data["module_factor"]])
    ax.loglog(beta, 1.0 / (1.0 + beta), "k-", lw=1.6, label=r"analytic $\omega^*=1/(1+\beta)$")
    styles = {"partitioned": ("o", "fsi_partitioned (this fix)"),
              "added_mass": ("s", "fsi_added_mass (this fix)"),
              "stability": ("^", "fsi_partitioned_stability (this fix)")}
    for kind, (marker, label) in styles.items():
        sel = [r for r in data["module_factor"] if r["kind"] == kind]
        xs = np.array([r["beta_eff"] for r in sel], dtype=float)
        ys = np.array([r["omega_final"] if r["omega_final"] is not None else np.nan for r in sel],
                      dtype=float)
        ys = np.where([r["converged"] for r in sel], ys, np.nan)
        ax.loglog(xs, ys, marker=marker, ls="none", ms=5, label=label, alpha=0.85)
        bad = [r["beta_eff"] for r in sel if not r["converged"]]
        if bad:
            ax.loglog(bad, [1e-14] * len(bad), "x", color="crimson", ms=11, mew=2.2)
    ax.set_xlabel(r"added-mass ratio $\beta_{eff}$")
    ax.set_ylabel(r"recovered relaxation factor $\omega$")
    ax.set_title("(b) all three copies reproduce the analytic factor")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("Aitken under-relaxation guard: scale-free positivity floor vs absolute floors", fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT, "fig5_scale_guard.png")
    fig.savefig(path, dpi=160)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())