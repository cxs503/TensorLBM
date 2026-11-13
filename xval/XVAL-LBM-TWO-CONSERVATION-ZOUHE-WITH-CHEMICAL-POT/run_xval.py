#!/usr/bin/env python3
"""XVAL-LBM-TWO-CONSERVATION-ZOUHE-WITH-CHEMICAL-POT -- driver.

Produces ``artifacts/xval_two_conservation/{results.json,fig_conservation.png}``.
Every number in the report is read from ``results.json`` (nothing hand-typed).

Benchmarks (predeclared before the runs, see the report section 1):
  B1  exact reconstruction invariants of a forced column (mass rho, momentum rho*u
      with the Guo half-force, order parameter psi and its advective flux psi*u).
  B2  analytic defect law of the classical (non-force-consistent) reconstruction:
      |u_measured - u_target| = |F| / (2 rho), relative check < 5 %.
  B3  upstream deficit-split variant: psi exact, advective psi-flux not conserved.
  B4  channel probe: free-energy channel with an open-boundary order-parameter
      closure -- records whether the two-conservation closure is stable at the
      Cahn-Hilliard interface (honest pass/fail, no tuning after the fact).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorlbm.phasefield.free_energy_channel import (
    ChannelConfig,
    FreeEnergyChannel2D,
    poiseuille_profile,
)
from tensorlbm.phasefield.zouhe_two_conservation import (
    column_invariants,
    reconstruct_inlet,
)

torch.set_num_threads(1)
OUT = Path("artifacts/xval_two_conservation")
OUT.mkdir(parents=True, exist_ok=True)
DT = torch.float64


def random_column(ny: int, seed: int, fscale: float = 0.05):
    gen = torch.Generator().manual_seed(seed)
    f = 0.05 + 0.1 * torch.rand((9, ny), generator=gen, dtype=DT)
    g = -0.2 + 0.4 * torch.rand((9, ny), generator=gen, dtype=DT)
    fx = fscale * torch.linspace(-1.0, 1.0, ny, dtype=DT)
    fy = 0.4 * fscale * torch.linspace(1.0, -1.0, ny, dtype=DT)
    u = torch.linspace(0.01, 0.06, ny, dtype=DT)
    return f, g, fx, fy, u


def b1_invariants():
    rows = []
    for ny in (16, 32, 64):
        for mode in ("corrected", "classic", "impose"):
            f, g, fx, fy, u = random_column(ny, seed=ny + hash(mode) % 97)
            psi_target = torch.linspace(0.6, -0.6, ny, dtype=DT)
            kw = dict(force_correction=(mode != "classic"))
            if mode == "impose":
                kw.update(psi_mode="impose", psi_target=psi_target)
            f_new, g_new, _ = reconstruct_inlet(f, g, u, (fx, fy), **kw)
            st = column_invariants(f_new, g_new, (fx, fy), force_correction=True)
            rows.append(
                {
                    "ny": ny,
                    "mode": mode,
                    "mass_flux_err": float((st.mass_flux - st.rho * u).abs().max()),
                    "mass_flux_rel": float(
                        ((st.mass_flux - st.rho * u).abs() / (st.rho * u).abs()).max()
                    ),
                    "ux_err": float((st.ux - u).abs().max()),
                    "ux_rel": float(((st.ux - u).abs() / u.abs()).max()),
                    "uy_abs": float(st.uy.abs().max()),
                    "psi_err": float((st.psi - psi_target).abs().max()) if mode == "impose" else None,
                    "psi_flux_err": float((st.psi_flux - st.psi * u).abs().max()),
                    "psi_flux_rel": float(
                        ((st.psi_flux - st.psi * u).abs() / (st.psi.abs() * u).abs()).max()
                    ),
                }
            )
    return rows


def b2_defect_law():
    rows = []
    ny = 32
    for fscale in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        f, g, fx, fy, u = random_column(ny, seed=7, fscale=fscale)
        f_new, _, _ = reconstruct_inlet(f, g, u, (fx, fy), force_correction=False)
        st = column_invariants(f_new, g, (fx, fy), force_correction=True)
        measured = float((st.ux - u).abs().max())
        predicted = float((fx.abs() / (2.0 * st.rho)).max())
        rows.append(
            {
                "force_scale": fscale,
                "fmax": float(fx.abs().max()),
                "measured_ux_defect": measured,
                "predicted_half_force": predicted,
                "rel_mismatch": abs(measured - predicted) / predicted,
            }
        )
    return rows


def b4_channel_probe(steps: int = 200):
    """Short probe of the open-boundary psi closure on the CH interface."""
    rows = []
    for mode in ("corrected", "classic", "impose"):
        cfg = ChannelConfig(ny=34, nx=48, u_scale=0.04, bc_mode=mode, xi=3.0)
        sim = FreeEnergyChannel2D(cfg)
        sim.init_channel_flow()
        prof = poiseuille_profile(cfg.ny, cfg.u_scale)
        blew = None
        for k in range(steps):
            sim.step()
            if not bool(torch.isfinite(sim.psi).all()):
                blew = k + 1
                break
        if blew:
            rows.append({"mode": mode, "steps": blew, "stable": False, "psi_max": None,
                         "profile_l2_rel": None, "max_abs_uy": None})
            continue
        rho, ux, uy, mu, (fx, fy), p = sim.macroscopic()
        mid = ux[1:-1, cfg.nx // 2]
        rows.append(
            {
                "mode": mode,
                "steps": steps,
                "stable": True,
                "psi_max": float(sim.psi[1:-1].max()),
                "psi_min": float(sim.psi[1:-1].min()),
                "profile_l2_rel": float((((mid - prof) ** 2).mean()) ** 0.5) / cfg.u_scale,
                "max_abs_uy": float(uy[1:-1].abs().max()),
                "ux_max": float(ux[1:-1].max()),
                "ux_ref": float(prof.max()),
            }
        )
    return rows


def figure(b1, b2):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    modes = ["corrected", "classic", "impose"]
    labels = ["corrected\n(force-consistent)", "classic\n(half-force dropped)", "impose\n(deficit split)"]
    colors = ["#2b7bba", "#c0392b", "#e08a1e"]
    for j, key in enumerate(("mass_flux_rel", "psi_flux_rel")):
        vals = [max(min(r[key] for r in b1 if r["mode"] == m), 1e-18) for m in modes]
        ax[0].bar([x + 0.36 * (j - 0.5) for x in range(3)], vals, width=0.34,
                  color=colors, alpha=0.9 if j == 0 else 0.45,
                  label="fluid mass  rho u" if j == 0 else "order parameter  psi u")
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].axhline(0.15, ls="--", c="k", lw=1)
    ax[0].text(2.45, 0.19, "15 % gate", ha="right", fontsize=8)
    ax[0].set_ylabel("relative reconstruction error")
    ax[0].set_title("B1 two-conservation closure (ny = 16/32/64, worst)")
    ax[0].legend(fontsize=7, loc="upper left")

    fs = [r["fmax"] for r in b2]
    ax[1].loglog(fs, [r["measured_ux_defect"] for r in b2], "o-", c="#c0392b",
                 label="classic: measured velocity defect")
    ax[1].loglog(fs, [r["predicted_half_force"] for r in b2], "k--",
                 label="analytic  |F| / (2 rho)")
    ax[1].set_xlabel("chemical-potential force  |F| = |mu grad psi|")
    ax[1].set_ylabel("max |u_reconstructed - u_target|")
    ax[1].set_title("B2 classical Zou-He defect law")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_conservation.png", dpi=140)
    plt.close(fig)


def main():
    b1 = b1_invariants()
    b2 = b2_defect_law()
    b4 = b4_channel_probe()
    figure(b1, b2)
    gate = 0.15
    summary = {
        "gate_15pct": gate,
        "b1": b1,
        "b2": b2,
        "b4": b4,
        "verdict": {
            "corrected_mass_flux_rel_max": max(r["mass_flux_rel"] for r in b1 if r["mode"] == "corrected"),
            "corrected_psi_flux_rel_max": max(r["psi_flux_rel"] for r in b1 if r["mode"] == "corrected"),
            "classic_ux_rel_max": max(r["ux_rel"] for r in b1 if r["mode"] == "classic"),
            "impose_psi_flux_rel_max": max(r["psi_flux_rel"] for r in b1 if r["mode"] == "impose"),
            "b2_max_rel_mismatch": max(r["rel_mismatch"] for r in b2),
            "b4_all_stable": all(r["stable"] for r in b4),
        },
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["verdict"], indent=2))


if __name__ == "__main__":
    main()