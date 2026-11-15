"""Figures for XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY (reads summary.json + reference JSON)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def fig_repro(summary: dict) -> Path | None:
    rep = [r for r in summary["rep"] if "C_I_E1_area" in r]
    if not rep:
        return None
    D = [r["D"] for r in rep]
    x = np.arange(len(D))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - 0.27, [r["m21062_C_I"] for r in rep], 0.25, label="m21062 report_payload (claimed)")
    ax.bar(x, [r["C_I_E1_area"] for r in rep], 0.25, label="re-run, library as-is (E1)")
    ax.bar(x + 0.27, [r["c_a_pot_area"] for r in rep], 0.25, label="same-mask potential reference")
    ax.set_xticks(x, [f"D={d}" for d in D])
    ax.set_ylabel("$C_I$ (area normalised)")
    ax.set_title(r"m21062 ladder does not reproduce ($\beta$=50, box=4D, KC=0.2)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG / "fig2_reproduction_gap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_protocol(summary: dict) -> Path | None:
    rows = [r for r in summary["protocol"] if "C_I_E1" in r]
    if not rows:
        return None
    labels = [f"t={r['transient_periods']:.1f}\nm={r['measure_periods']:.1f}" for r in rows]
    x = np.arange(len(rows))
    pot = rows[0]["c_a_pot_area"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].plot(x, [r["C_I_E1"] for r in rows], "o-", label="E1 $C_I$")
    ax[0].plot(x, [r["C_I_E2"] for r in rows], "s--", label="E2 $C_I$")
    ax[0].axhline(pot, color="k", ls=":", label=f"same-mask potential {pot:.3f}")
    ax[0].axhline(0.0, color="gray", lw=0.8)
    ax[0].set_xticks(x, labels, fontsize=8)
    ax[0].set_ylabel("$C_I$")
    ax[0].set_title(r"spin-up sensitivity (D=16, $\beta$=50, box=4D): $C_I$ changes sign")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(x, [r["resid_E1"] for r in rows], "o-", label="residual E1")
    ax[1].plot(x, [r["resid_E2"] for r in rows], "s--", label="residual E2")
    ax[1].axhline(0.05, color="r", ls=":", label="fit-quality gate 0.05")
    ax[1].set_xticks(x, labels, fontsize=8)
    ax[1].set_ylabel("relative fit residual")
    ax[1].set_title("first-harmonic fit quality (m21062 protocol: 34 %)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig3_protocol_convergence.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_potential_grid() -> Path | None:
    p = HERE / "reference" / "potential_grid.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    D = [r["D"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax[0].semilogx(D, [r["c_a_raw"] for r in rows], "o-", label="$c_a$ (raw, $\\rho\\pi R^2$)")
    ax[0].semilogx(D, [r["c_a_area"] for r in rows], "s--", label="$c_a$ (actual mask area)")
    ax[0].axhline(1.0, color="k", ls=":", label="unbounded analytic $C_a=1$")
    ax[0].set_xlabel("resolution D (box = 4D, nx = 4D)")
    ax[0].set_ylabel("potential added mass")
    ax[0].set_title("same-mask potential reference is NOT converged at nx=64")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].semilogx(D, [100 * (r["area_over_analytic"] - 1) for r in rows], "o-")
    ax[1].axhline(0.0, color="k", ls=":")
    ax[1].set_xlabel("resolution D")
    ax[1].set_ylabel("staircase mask area excess [%]")
    ax[1].set_title("why the two normalisations differ")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = FIG / "fig4_potential_ref_grid.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_pressure(summary: dict) -> Path | None:
    rows = [r for r in summary.get("e3", []) if "p_dipole_rows" in r]
    if not rows:
        return None
    return None


def fig_field(D: int = 16, beta: float = 50.0) -> Path:
    """One field snapshot (u_x, vorticity) after 8 periods -- short local run."""
    import math

    import torch

    from tensorlbm.boundary.boundaries import bounce_back_cells
    from tensorlbm.boussinesq import apply_guo_forcing
    from tensorlbm.d2q9 import equilibrium, macroscopic
    from tensorlbm.oscillatory_cylinder_inertia import circle_mask
    from tensorlbm.solver import stream

    torch.set_num_threads(2)
    nx = ny = 4 * D
    kc, u0 = 0.2, 0.04
    T = kc * D / u0
    omega = 2 * math.pi / T
    a0 = omega * u0
    nu = D * D / (beta * T)
    tau = 3.0 * nu + 0.5
    mask = circle_mask(nx, ny, D)
    fluid = ~mask
    rho0 = torch.ones((ny, nx), dtype=torch.float64)
    z = torch.zeros_like(rho0)
    f = bounce_back_cells(equilibrium(rho0, z, z), mask)
    Fx, Fy = torch.zeros_like(rho0), torch.zeros_like(rho0)
    n_steps = int(8 * T)
    for step in range(1, n_steps + 1):
        Fx[fluid] = a0 * math.cos(omega * float(step))
        old = f[:, mask].clone()
        f = apply_guo_forcing(f, torch.stack([Fx, Fy]), tau)
        f[:, mask] = old
        f = stream(f)
        rho, ux, uy = macroscopic(f)
        f = bounce_back_cells(f, mask)
    ux_np, uy_np = ux.numpy(), uy.numpy()
    vort = np.zeros_like(ux_np)
    vort[1:-1, 1:-1] = (uy_np[2:, 1:-1] - uy_np[:-2, 1:-1]) / 2.0 - (
        ux_np[1:-1, 2:] - ux_np[1:-1, :-2]) / 2.0
    solid = mask.numpy()
    vort[solid] = np.nan
    ux_plot = ux_np.copy()
    ux_plot[solid] = np.nan
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    im0 = ax[0].imshow(ux_plot, origin="lower", cmap="RdBu_r")
    ax[0].set_title(r"$u_x$ at peak drive (D=16, $\beta$=50, box=4D)")
    fig.colorbar(im0, ax=ax[0], shrink=0.85)
    vmax = np.nanpercentile(np.abs(vort), 99)
    im1 = ax[1].imshow(vort, origin="lower", cmap="inferno", vmin=-vmax, vmax=vmax)
    ax[1].set_title(r"vorticity $\omega_z$ (Stokes layer $\delta\approx D/\sqrt{\beta}$)")
    fig.colorbar(im1, ax=ax[1], shrink=0.85)
    for a in ax:
        a.set_xlabel("x [cells]")
        a.set_ylabel("y [cells]")
    fig.tight_layout()
    out = FIG / "fig1_field_snapshot.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    summary = json.loads((HERE / "summary.json").read_text())
    made = [f for f in (fig_field(), fig_repro(summary), fig_protocol(summary),
                        fig_potential_grid()) if f]
    for p in made:
        print(p.name, p.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())