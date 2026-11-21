"""Restore the two free-surface validation figures from the committed run data.

Background
----------
The original m21376 deliverable (platform report 19326) shipped two figures,
``figures/fs_validation_summary.png`` and ``figures/fs_dambreak_rows.png``,
referenced by ``REPORT_FS_VALIDATION.md``.  When the increment was merged
(#2458) the binary files were dropped, leaving the report's image links
dangling.  ``results_fs.json`` (the machine-readable output of the *same*
deterministic run) survived, so the figures can be rebuilt exactly from it
without re-running any simulation.

This script is the reproducible path for that restore: it reads
``results_fs.json`` and re-renders the two PNGs with the identical layout,
colours and annotations used by ``publish_fs_report.py``.  It does **not**
re-run any simulation and does **not** touch ``results_fs.json``.

Note on which state the figures show: they are the *historical* m21376
figures (Guo hydrostatic -54.0%, dam-break mass drift 69.9%), consistent with
the tables in ``REPORT_FS_VALIDATION.md``.  The post-fix state (R1/R2 closed,
R3 open) is documented in ``REPORT_R1R2.md`` / ``results_r1r2.json``.

Usage::

    python restore_figures.py
"""

from __future__ import annotations

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(HERE, "results_fs.json")) as fh:
    RES = json.load(fh)

g = RES["guo"]["g"]
analytic = 3.0 * g  # d rho / d y for hydrostatic equilibrium, cs^2 = 1/3

# ---------------------------------------------------------------------------
# figure 1: fs_validation_summary.png  (3 panels)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

ax = axes[0]
js = np.array(RES["guo"]["js"])
ax.plot(RES["guo"]["rho"], js, "o-", ms=3, label="Guo forcing (measured)")
ax.plot(RES["raw"]["rho"], np.array(RES["raw"]["js"]), "s-", ms=3,
        label="skeleton 'raw' force term (measured)")
y = np.linspace(js.min(), js.max(), 50)
ax.plot(1.0 + analytic * (y - js.min()), y, "k--", lw=1,
        label=r"analytic $\mathrm{d}\rho/\mathrm{d}y = 3g$")
ax.plot(1.0 + RES["defect_slope"] * (y - js.min()), y, "r:", lw=1,
        label=r"defect theory $3(-g/3)=-g$")
ax.set_xlabel(r"$\rho$ (lattice)")
ax.set_ylabel("y (lattice)")
ax.set_title("(a) hydrostatic gradient, closed box")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[1]
t = np.array(RES["skeleton"]["times"])
m = np.array(RES["skeleton"]["mass"])
ax.semilogy(t, np.abs(m), "o-", ms=3, label="Maarten port: |liquid mass|")
ax.semilogy(t, np.full_like(t, abs(m[0])), "k--", lw=1, label="initial mass")
ax.set_xlabel("step")
ax.set_ylabel(r"$|\sum m|$")
ax.set_title("(b) skeleton free-surface port diverges")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
# the original rendered the x-extent of the liquid mask per step; results_fs.json
# stores exactly that extent, so the panel is reproduced without the masks.
for k, colour in ((0, "C0"), (50, "C1"), (150, "C2"), (300, "C3")):
    x0, x1 = RES["dam"]["extent"][str(k)]
    ax.plot([x0, x1], [k, k], color=colour, lw=2)
ax.set_xlabel("x (lattice)")
ax.set_ylabel("step (offset)")
ax.set_title(f"(c) corrected dam break: liquid extent, mass drift {RES['dam']['drift']:.1%}")
ax.grid(alpha=0.3)

fig.tight_layout()
p1 = os.path.join(OUT, "fs_validation_summary.png")
fig.savefig(p1, dpi=130)
plt.close(fig)

# ---------------------------------------------------------------------------
# figure 2: fs_dambreak_rows.png
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.6))
for k, colour in ((0, "C0"), (50, "C1"), (150, "C2"), (300, "C3")):
    ax.plot(RES["dam"]["row_counts"][str(k)], label=f"step {k}")
ax.set_xlabel("y (lattice)")
ax.set_ylabel("liquid-cell count")
ax.set_title("corrected dam break: liquid volume per row")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
p2 = os.path.join(OUT, "fs_dambreak_rows.png")
fig.savefig(p2, dpi=130)
plt.close(fig)

print("restored figures:")
for p in (p1, p2):
    print("  ", os.path.relpath(p, HERE), os.path.getsize(p), "bytes")
