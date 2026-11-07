#!/usr/bin/env python3
"""Build the three composite figures of the W18-N08 rotating-frame report.

Reads the JSON records produced by
``benchmarks/verified/rotating_{ekman,taylor_column,source_sink}/run.py`` and
writes compact PNGs into this directory (kept small on purpose: the report
payload is base64-embedded).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / "benchmarks" / "verified"
OUT = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fig_ekman(rec: dict) -> Path:
    prof = rec["profiles"]
    z = np.asarray(prof["z"])
    up = np.asarray(prof["u_par"])
    uq = np.asarray(prof["u_perp"])
    d = rec["fit"]["delta_fit"]
    ug = rec["fit"]["U_g_fit"]
    zz = np.linspace(0.0, z.max(), 400)
    zp = zz / d
    up_a = ug * (1.0 - np.exp(-zp) * np.cos(zp))
    uq_a = ug * (np.exp(-zp) * np.sin(zp))

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax[0].plot(z / d, up / ug, "o", ms=3, label="LBM $u_\\parallel/U_g$")
    ax[0].plot(zz / d, up_a / ug, "-", lw=1, label="analytic")
    ax[0].plot(z / d, uq / ug, "s", ms=3, label="LBM $u_\\perp/U_g$")
    ax[0].plot(zz / d, uq_a / ug, "--", lw=1, label="analytic")
    ax[0].set_xlabel(r"$z/\delta_E$")
    ax[0].set_ylabel(r"$u/U_g$")
    ax[0].set_title(
        f"Ekman spiral: $\\delta_E$ fit {rec['fit']['delta_fit']:.2f} vs "
        f"{rec['fit']['delta_ana']:.0f} ({100 * rec['checks']['delta_err']:.2f} %)"
    )
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    probes = rec["probes"]
    zs = [p["zeta"] for p in probes]
    ep = [-100 * p["err_par"] for p in probes]
    eq = [-100 * p["err_perp"] for p in probes]
    ax[1].bar(np.asarray(zs) - 0.08, ep, width=0.16, label="relative error $u_\\parallel$")
    ax[1].bar(np.asarray(zs) + 0.08, eq, width=0.16, label="relative error $u_\\perp$")
    ax[1].axhline(-3.0, color="r", ls=":", label="3 % criterion")
    ax[1].set_xlabel(r"probe height $z/\delta_E$")
    ax[1].set_ylabel("error [%] (signed, negative = LBM low)")
    ax[1].set_title(
        f"5-probe max error {100 * rec['checks']['profile_max_err']:.2f} %"
    )
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "fig1_ekman.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def fig_taylor(rec: dict, ctrl: dict | None) -> Path:
    prof = rec["column"]["z_profile"]
    z = [p["z"] for p in prof]
    w = [100 * p["w_max_rel"] for p in prof]
    u = [100 * p["u_mean_rel"] for p in prof]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax[0].plot(w, z, "o-", ms=4, label=r"max $|w|/U_\infty$")
    ax[0].plot(u, z, "s--", ms=3, label=r"mean $|u_h|/U_\infty$")
    ax[0].axvline(5.0, color="r", ls=":", label="5 % criterion")
    if ctrl is not None:
        cw = [100 * p["w_max_rel"] for p in ctrl["column"]["z_profile"]]
        ax[0].plot(cw, [p["z"] for p in ctrl["column"]["z_profile"]], "^-",
                   color="0.5", ms=3, label="no-rotation control")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("velocity in the column / $U_\\infty$  [%]")
    ax[0].set_ylabel("z [cells]")
    ax[0].set_title(
        f"Taylor column, $E_L$={rec['config']['nu_lattice'] / (rec['config']['Omega_lattice'] * rec['config']['radius'] ** 2):.4f}"
        f", Ro={rec['config']['Ro']:.3f}"
    )
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")

    names = ["$w_{max}$", "$w_{max}$\ninterior", "axial\nextent err", "$\\\\partial_z u$\ncolumnarity"]
    vals = [
        100 * rec["column"]["w_max_rel"],
        100 * rec["column"]["w_max_rel_interior"],
        100 * rec["column"]["length_rel_err"],
        100 * rec["column"]["columnarity_dev"],
    ]
    cols = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    ax[1].bar(names, vals, color=cols)
    ax[1].axhline(5.0, color="r", ls=":", label="5 % stagnation criterion")
    ax[1].axhline(10.0, color="orange", ls="--", label="10 % extent criterion")
    ax[1].set_ylabel("[%] of $U_\\infty$ / relative error")
    ax[1].set_title("acceptance metrics")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT / "fig2_taylor.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def fig_source_sink(rec: dict) -> Path:
    rp = rec["radial_profile"]
    r = np.asarray(rp["r"])
    v = np.asarray(rp["v_r_mid"])
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax[0].loglog(r, v, "o", label="LBM $v_r$ (mid-height)")
    ax[0].loglog(r, rp["coefficient"] * r ** rp["exponent"], "-",
                 label=f"fit $r^{{{rp['exponent']:.3f}}}$")
    ax[0].loglog(r, v[0] * (r / r[0]) ** -1.0, "k--", lw=1, label="$r^{-1}$")
    ax[0].set_xlabel("r [cells]")
    ax[0].set_ylabel("$v_r$ [lattice]")
    ax[0].set_title(
        f"source-sink: exponent {rp['exponent']:.4f} (target -1, err "
        f"{abs(rp['exponent'] + 1):.4f})"
    )
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")

    axial = [a for a in rec["axial"] if not math.isnan(a["max_rel_dev"])]
    ax[1].plot([a["r_mid"] for a in axial], [100 * a["max_rel_dev"] for a in axial],
               "o-", label="interior slab")
    ax[1].plot([a["r_mid"] for a in axial],
               [100 * a["max_rel_dev_full_slab"] for a in axial],
               "s--", ms=3, color="0.5", label="full slab (incl. cap layers)")
    ax[1].axhline(5.0, color="r", ls=":", label="5 % criterion")
    ax[1].set_xlabel("r [cells]")
    ax[1].set_ylabel(r"max $|v_r-\bar v_r|/\bar v_r$  [%]")
    ax[1].set_title(
        f"axial uniformity, interior max {100 * rec['checks']['axial_uniformity_max']:.2f} %"
    )
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "fig3_source_sink.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def main() -> int:
    ek = load(next((REC / "rotating_ekman" / "records").glob("ekman_delta16*.json")))
    tc_path = Path(__file__).resolve().parent / "chosen_taylor.txt"
    tc = load(REC / "rotating_taylor_column" / "records" / tc_path.read_text().strip())
    ctrl_path = REC / "rotating_taylor_column" / "records" / "taylor_norot_control.json"
    ctrl = load(ctrl_path) if ctrl_path.exists() else None
    ss = load(next((REC / "rotating_source_sink" / "records").glob("source_sink_*.json")))
    print(fig_ekman(ek))
    print(fig_taylor(tc, ctrl))
    print(fig_source_sink(ss))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
