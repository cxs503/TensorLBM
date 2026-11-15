"""Figure for the m21930 audit of the host-staged-halo D3b reference.

Produces ``docs/figures/xval_host_staged_halo_m21930.png``:

* left  — transverse sector of the linearised D3Q19 BGK step operator: the two
          slow hydrodynamic shear modes vs the four ghost modes, against the
          ``nu*khat^2`` closed form (log axes), plus the note that the m21387
          row-pairing yields an *empty* candidate set;
* right — measured shear decay rate of the decomposed solver vs the exact
          eigenvalue and vs the closed form (16^3 / 24^3, nu = 0.02).

    PYTHONPATH=src python tools/xval_host_staged_halo/audit_reference_figure_m21930.py
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.host_staged_halo import discrete_transverse_rate, shear_khat2, transverse_modes

HERE = Path(__file__).resolve()
OUT = HERE.parents[2] / "docs" / "figures" / "xval_host_staged_halo_m21930.png"

NU = 0.02


def load_driver():
    path = HERE.parent / "run_xval_host_staged_halo.py"
    spec = importlib.util.spec_from_file_location("xval_driver_fig", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # ---- left: transverse spectrum vs closed form ------------------------
    ns = np.arange(12, 65)
    slow, ghost, closed = [], [], []
    for n in ns:
        k = 2 * math.pi / n
        modes = transverse_modes((k, 0.0, 0.0), NU)
        slow.append(discrete_transverse_rate((k, 0.0, 0.0), NU))
        ghost.append(modes[2]["rate"])
        closed.append(NU * shear_khat2((k, 0.0, 0.0)))
    slow, closed = np.array(slow), np.array(closed)
    ax1.loglog(ns, ghost, color="tab:red", lw=1.2, label="4 ghost modes (fast, unphysical)")
    ax1.loglog(ns, slow, "o", ms=3.5, color="tab:blue", label="shear pair: exact eigenvalue")
    ax1.loglog(ns, closed, "-", color="0.2", lw=1.2, label=r"closed form $\nu\hat{k}^{2}$")
    ax1.set_xlabel("grid size n  (k = 2$\\pi$/n)")
    ax1.set_ylabel("decay exponent per step")
    ax1.set_title(f"transverse spectrum, linearised D3Q19 BGK, $\\nu$={NU}")
    ax12 = ax1.twinx()
    ax12.semilogx(ns, 100 * (slow - closed) / closed, "--", color="tab:green", lw=1.0,
                  label=r"eigenvalue vs $\nu\hat{k}^{2}$  [%]")
    ax12.set_ylabel("eigenvalue vs closed form [%]")
    ax1.set_zorder(ax12.get_zorder() + 1)
    ax1.patch.set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax12.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
    ax1.text(0.03, 0.97, "m21387 row-pairing: 0 candidates -> RuntimeError",
             transform=ax1.transAxes, va="top", fontsize=7.5, color="tab:red")

    # ---- right: measured vs references ----------------------------------
    driver = load_driver()
    rows = driver.table_shear_discrete(grids=(16, 24), nus=(NU,),
                                       device_grids=((1, 1, 1), (2, 2, 2)))
    grids = sorted({r["grid"] for r in rows}, key=lambda g: int(g.split("^")[0]))
    x = np.arange(len(grids))
    meas = []
    for g in grids:
        vals = [r["gamma_LBM"] for r in rows if r["grid"] == g]
        assert max(vals) - min(vals) < 1e-9 * max(vals), vals  # decomposition must not move it
        meas.append(float(np.mean(vals)))
    eigen = [next(r["gamma_discrete"] for r in rows if r["grid"] == g) for g in grids]
    cont = [next(r["gamma_continuum"] for r in rows if r["grid"] == g) for g in grids]
    ax2.plot(x, meas, "o", ms=6, color="tab:blue", label="measured (1x1x1 and 2x2x2)")
    ax2.plot(x, eigen, "_", ms=18, mew=2, color="tab:orange", label="exact eigenvalue")
    ax2.plot(x, cont, "s", ms=5, mfc="none", color="0.3", label=r"closed form $\nu\hat{k}^{2}$")
    for xi, m, e, c in zip(x, meas, eigen, cont):
        ax2.annotate(f"{100 * (m - e) / e:+.3f}%", (xi, m), textcoords="offset points",
                     xytext=(6, 6), fontsize=7)
        ax2.annotate(f"{100 * (m - c) / c:+.2f}%", (xi, c), textcoords="offset points",
                     xytext=(6, -12), fontsize=7, color="0.3")
    ax2.set_xticks(x)
    ax2.set_xticklabels([g.replace("^3", "$^3$") for g in grids])
    ax2.set_ylabel("shear decay exponent per step")
    ax2.set_xlabel("grid (D3b arm, t = 20 -> 220)")
    ax2.set_title("measured vs reference (labels: vs eigen / vs closed form)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    print({g: (m, e, c) for g, m, e, c in zip(grids, meas, eigen, cont)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())