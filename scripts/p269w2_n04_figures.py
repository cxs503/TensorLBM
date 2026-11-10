"""P269W2-N04 figures: lattice linear stability of plane Poiseuille flow.

Reads ``artifacts/p269w2_n04/acceptance.json`` (+ the per-task point records)
and writes PNG figures into ``artifacts/p269w2_n04/figures/``.  Every number in
the figures comes from those JSON records - nothing is typed by hand.

    python scripts/p269w2_n04_figures.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("P269_ART", ROOT / "artifacts" / "p269w2_n04"))
FIG = ART / "figures"
RE_C = 5772.2218  # Orszag (1971) critical Reynolds number
OSPEC = (0.23752648882034, 0.0037396706229)  # Orszag Re=10000, alpha=1

COLORS = {"bgk": "#1f77b4", "mrt": "#d62728", "rlbm": "#2ca02c"}
ALPHA_LABEL = {"pi/6": 0.5235987755982988, "pi/4": 0.7853981633974483,
               "pi/3": 1.0471975511965979, "2pi/5": 1.2566370614359172}


def load() -> dict:
    return json.loads((ART / "acceptance.json").read_text())


def _points(acc: dict, **kw) -> list[dict]:
    out = []
    for p in acc["points"]:
        if all(abs(p.get(k, float("nan")) - v) < 1e-9 if isinstance(v, float) else p.get(k) == v
               for k, v in kw.items()):
            out.append(p)
    return out


def fig_spectrum(acc: dict) -> Path | None:
    """Full eigenvalue spectrum: lattice operator vs Orr-Sommerfeld continuum."""
    cand = [p for p in acc["points"] if p["collision"] == "bgk" and p["ny"] == 64]
    if not cand:
        return None
    cand.sort(key=lambda p: (abs(p["Re"] - 10000.0), abs(p["alpha_nd"] - 1.0472)))
    p = cand[0]
    eig = np.array(p["eig_omega"])
    os_ref = np.array([[p["sigma_os"], -p["alpha_nd"] * p["c_os"][0]]])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(eig[:, 1], eig[:, 0], "o", ms=5, mfc="none", color="#1f77b4",
               label=f"lattice BGK spectrum (Ny={p['ny']})")
    ax[0].plot([p["c_os"][0]], [p["c_os"][1]], "k*", ms=13,
               label="OS leading eigenvalue")
    ax[0].axhline(0.0, color="0.6", lw=0.8)
    ax[0].set_xlabel(r"$c_r$")
    ax[0].set_ylabel(r"$c_i$  (growth per $h/U_{max}$ / $\alpha$)")
    ax[0].set_title(f"Re={p['Re']:.0f}, $\\alpha$={p['alpha_nd']:.3f}, Ny={p['ny']}")
    ax[0].legend(loc="best", fontsize=8)
    ax[0].set_xlim(-0.1, 1.15)
    # zoom on the physical (least damped) part
    m = (eig[:, 0] > -0.1) & (eig[:, 0] < 1.1) & (eig[:, 1] > -1.0) & (eig[:, 1] < 0.1)
    ax[1].plot(eig[m, 1], eig[m, 0], "o", ms=6, mfc="none", color="#1f77b4",
               label="lattice")
    ax[1].plot([p["c_os"][0]], [p["c_os"][1]], "k*", ms=14, label="Orr-Sommerfeld")
    ax[1].annotate(f"$\\sigma_{{LBM}}$={p['sigma_nondim']:.4f}\n$\\sigma_{{OS}}$={p['sigma_os']:.4f}",
                   xy=(p["c_os"][0], p["c_os"][1]), xytext=(10, -30),
                   textcoords="offset points", fontsize=8)
    ax[1].set_xlabel(r"$c_r$")
    ax[1].set_ylabel(r"$c_i$")
    ax[1].set_title("physical branch (zoom)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Fig.1  Lattice D2Q9 spectrum vs Orr-Sommerfeld spectrum (channel)")
    fig.tight_layout()
    out = FIG / "fig1_spectrum.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_growth_vs_re(acc: dict) -> Path | None:
    """sigma(Re) at fixed alpha for the three collision models, Ny=64."""
    akey = "pi/3"
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    any_data = False
    for kind in ("bgk", "mrt", "rlbm"):
        ps = sorted([p for p in acc["points"]
                     if p["alpha_key"] == akey and p["collision"] == kind
                     and p["ny"] == 64], key=lambda r: r["Re"])
        if not ps:
            continue
        any_data = True
        re = [p["Re"] for p in ps]
        ax.plot(re, [p["sigma_nondim"] for p in ps], "o-", color=COLORS[kind],
                label=f"LBM {kind.upper()}", ms=5)
        if kind == "bgk":
            ax.plot(re, [p["sigma_os"] for p in ps], "k--", lw=1.4,
                    label="Orr-Sommerfeld (Chebyshev)")
    if not any_data:
        plt.close(fig)
        return None
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axvline(RE_C, color="0.7", ls=":", lw=1.2)
    ax.text(RE_C * 1.01, ax.get_ylim()[0] * 0.9, " $Re_c=5772$", fontsize=8, color="0.35")
    ax.set_xlabel("Re = $U_{max}h/\\nu$")
    ax.set_ylabel(r"$\sigma\ h/U_{max}$")
    ax.set_title(f"Fig.2  growth rate vs Re, $\\alpha$={ALPHA_LABEL[akey]:.3f}, Ny=64")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig2_growth_vs_re.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_deviation(acc: dict) -> Path | None:
    """Relative deviation from the OS benchmark vs Re, with the 15 % line."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    any_data = False
    for ax, akey in zip(axes, ("pi/3", "pi/4")):
        for kind in ("bgk", "mrt", "rlbm"):
            ps = sorted([p for p in acc["points"] if p["alpha_key"] == akey
                         and p["collision"] == kind and p["ny"] == 64],
                        key=lambda r: r["Re"])
            if not ps:
                continue
            any_data = True
            ax.semilogy([p["Re"] for p in ps], [max(p["rel_dev"], 1e-6) for p in ps],
                        "o-", color=COLORS[kind], label=kind.upper(), ms=4)
        ax.axhline(0.15, color="k", ls="--", lw=1.2)
        ax.text(1050, 0.16, "15 % acceptance line", fontsize=8)
        ax.set_xlabel("Re")
        ax.set_title(f"$\\alpha$ = {ALPHA_LABEL[akey]:.3f}, Ny=64")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$|\sigma_{LBM}-\sigma_{OS}|/|\sigma_{OS}|$")
    axes[0].legend(fontsize=8)
    if not any_data:
        plt.close(fig)
        return None
    fig.suptitle("Fig.3  deviation of the lattice growth rate from the OS benchmark")
    fig.tight_layout()
    out = FIG / "fig3_deviation.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_convergence(acc: dict) -> Path | None:
    """sigma vs 1/Ny with Richardson extrapolation (A3)."""
    rows = acc.get("criteria", {}).get("A3", {}).get("rows", [])
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    marks = {"bgk": "o", "mrt": "s", "rlbm": "^"}
    for r in rows:
        akey = r["alpha_key"]
        ps = sorted([p for p in acc["points"] if abs(p["Re"] - r["Re"]) < 1e-9
                     and p["alpha_key"] == akey and p["collision"] == r["collision"]],
                    key=lambda p: p["ny"])
        if not ps:
            continue
        ax.plot([1.0 / p["ny"] for p in ps], [p["sigma_nondim"] for p in ps],
                marks.get(r["collision"], "o") + "-", ms=5,
                label=f"Re={r['Re']:.0f} {r['collision'].upper()}")
        ax.plot([0.0], [r["limit"]], "x", color="k", ms=6)
    ax.axhline(rows[0]["sigma_os"], color="k", ls="--", lw=1.2,
               label=f"OS (alpha={ALPHA_LABEL[rows[0]['alpha_key']]:.3f})")
    ax.set_xlabel("1/Ny")
    ax.set_ylabel(r"$\sigma\ h/U_{max}$")
    ax.set_title("Fig.4  wall-normal refinement (x = Richardson limit)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig4_convergence.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_mode_shape(acc: dict) -> Path | None:
    """Mode shape (wall-normal velocity) and the corresponding field map."""
    cand = [p for p in acc["points"] if p["collision"] == "bgk" and p["ny"] in (64, 128)
            and p["alpha_key"] == "pi/3"]
    cand.sort(key=lambda p: (abs(p["Re"] - 6000.0), -p["ny"]))
    if not cand:
        return None
    p = cand[0]
    y = np.array(p["profile_y"])
    v = np.array(p["profile_v"])
    v = v[:, 0] + 1j * v[:, 1]
    v = v / np.max(np.abs(v))
    sys.path.insert(0, str(ROOT / "src"))
    from tensorlbm.stability.orr_sommerfeld import cheb_interp, os_spectrum  # noqa: PLC0415

    os_spec = os_spectrum(p["alpha_nd"], p["Re"], 80)
    v_os = cheb_interp(os_spec.v_leading, os_spec.y, y)
    v_os = v_os / np.max(np.abs(v_os))
    if abs(np.vdot(v_os, v)) > 0:
        ph = np.vdot(v_os, v) / abs(np.vdot(v_os, v))
        v = v / ph  # align the arbitrary phase before comparing

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(np.real(v_os), y, "k--", lw=1.5, label="Orr-Sommerfeld")
    ax[0].plot(np.real(v), y, "-", color="#1f77b4", lw=2,
               label=f"lattice BGK (Ny={p['ny']})")
    ax[0].plot(np.imag(v_os), y, "k:", lw=1.0, label="OS (imag)")
    ax[0].plot(np.imag(v), y, ":", color="#1f77b4", lw=1.0)
    ax[0].set_xlabel(r"$\hat v(y)$")
    ax[0].set_ylabel("y / h")
    ax[0].set_title(f"mode shape, Re={p['Re']:.0f}, $\\alpha$={p['alpha_nd']:.3f}")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    # field map: u_y(x, y) over one streamwise wavelength
    xs = np.linspace(0, 2 * math.pi / p["alpha_nd"], 160)
    field = np.real(np.outer(v, np.exp(1j * p["alpha_nd"] * xs)))
    im = ax[1].pcolormesh(xs, y, field, cmap="RdBu_r", shading="gouraud",
                          vmin=-np.abs(field).max(), vmax=np.abs(field).max())
    fig.colorbar(im, ax=ax[1], label=r"$u_y / A$")
    ax[1].set_xlabel(r"$x/h$")
    ax[1].set_ylabel("y / h")
    ax[1].set_title("reconstructed disturbance field of the selected eigenmode")
    fig.suptitle("Fig.5  eigenfunction comparison and disturbance field")
    fig.tight_layout()
    out = FIG / "fig5_mode_shape.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_stokes(acc: dict) -> Path | None:
    """A1: measured Stokes damping vs the analytical -nu k^2."""
    a1 = acc.get("a1_stokes")
    if not a1:
        return None
    rows = a1["rows"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    cont = [r["sigma_continuum"] for r in rows]
    meas = [r["sigma_measured"] for r in rows]
    vn = [r["sigma_von_neumann"] for r in rows]
    lim = [min(cont) * 1.1, max(cont) * 0.5]
    ax.plot(cont, meas, "o", ms=7, color="#1f77b4", label="measured (nonlinear box)")
    ax.plot(cont, vn, "x", ms=7, color="#d62728", label="von Neumann (linear)")
    ax.plot(lim, lim, "k--", lw=1.0, label=r"$\sigma=-\nu k^2$")
    for r in rows:
        ax.annotate(f"n={r['n']},$\\tau$={r['tau']}", (r["sigma_continuum"], r["sigma_measured"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(r"$\sigma_{continuum}=-\nu k^2$")
    ax.set_ylabel(r"$\sigma$ per timestep")
    ax.set_title("Fig.6  A1: uniform-flow Stokes damping")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig6_stokes.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    acc = load()
    made = []
    for fn in (fig_spectrum, fig_growth_vs_re, fig_deviation, fig_convergence,
               fig_mode_shape, fig_stokes):
        try:
            out = fn(acc)
        except Exception as exc:  # noqa: BLE001 - a missing panel must not stop the rest
            print(f"[warn] {fn.__name__}: {exc}")
            out = None
        if out is not None:
            made.append(str(out))
            print("wrote", out)
    (ART / "figures.json").write_text(json.dumps(made, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())