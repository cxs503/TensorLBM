#!/usr/bin/env python3
"""Audit figures for m17514 — every number is read from a machine artefact or
recomputed here; nothing is typed by hand.

fig1_gammaK: kinetic-energy decay history (CSV artefacts) vs the closed-form
             Taylor-Green curve, AA vs AB, two resolutions.
fig2_field:  independent 400-step AA run (n=48) -> ux field, closed form, and the
             difference field; prints the recomputed relative L2 to stdout.
fig3_hazard: parity-hazard / parity-blind magnitudes from result.json against the
             15 % owner gate.

Run: PYTHONPATH=<m17514>/src python3 docs/audit_m17514/gen_figs_audit.py
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

SRC_TREE = Path("/root/tp-agent/work/m17514")
sys.path.insert(0, str(SRC_TREE / "src"))
from tensorlbm.aa_pattern import AAPatternStepper, equilibrium_from_velocity  # noqa: E402

ART = SRC_TREE / "artifacts/xval_aa_pattern_even_odd"
OUT = Path(__file__).resolve().parent
RES = json.loads((ART / "result.json").read_text())
SPECS = {s["name"]: s for s in RES["specs"]}


def load_csv(name: str):
    rows = [l.split(",") for l in (ART / name).read_text().strip().splitlines()]
    head, body = rows[0], rows[1:]
    cols = {h: np.array([float(r[i]) for r in body]) for i, h in enumerate(head) if h != "phase"}
    return cols


def fig_gammaK():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rec = {}
    for ax, name in zip(axes, ("tg_n32_re100", "tg_n48_re100")):
        spec = SPECS[name]
        k = spec["k"]
        nu = spec["nu"]
        k0 = 0.25 * spec["u0"] ** 2  # K(0) = (U0^2)/4, closed form
        exact = np.array([k0 * math.exp(-4.0 * nu * k * k * t) for t in np.arange(401)])
        for arm, style in (("aa", "-"), ("ab", "--")):
            c = load_csv(f"k_history_{arm}_{name}.csv")
            ax.plot(c["step"], c["kinetic_energy"], style, lw=1.6, label=f"{arm.upper()} (measured)")
            rec[f"{name}_{arm}_K_end"] = float(c["kinetic_energy"][-1])
        ax.plot(np.arange(401), exact, "k:", lw=1.2, label="closed form (Green-Taylor)")
        ax.set_yscale("log")
        ax.set_xlabel("lattice step t")
        ax.set_ylabel("K(t) = 0.5<u^2+v^2>")
        ax.set_title(f"{name}  N={spec['n']}  Re={spec['re']:.0f}  tau={spec['tau']:.3f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("m17514 audit: AA in-place vs AB two-field vs closed-form TGV decay", fontsize=11)
    fig.tight_layout()
    p = OUT / "fig1_gammaK_aa_vs_ab_vs_exact.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, rec


def fig_field():
    spec = SPECS["tg_n48_re100"]
    n, nu, k, tau, steps = spec["n"], spec["nu"], spec["k"], spec["tau"], spec["steps"]
    x = np.arange(n) + 0.5
    ux = -spec["u0"] * np.cos(k * x)[None, :] * np.sin(k * x)[:, None]
    uy = spec["u0"] * np.sin(k * x)[None, :] * np.cos(k * x)[:, None]
    f0 = equilibrium_from_velocity(
        torch.ones((n, n), dtype=torch.float64),
        torch.tensor(ux, dtype=torch.float64),
        torch.tensor(uy, dtype=torch.float64),
    )
    st = AAPatternStepper(f0, tau)
    for _ in range(steps):
        st.step()
    _, uxa, uya = st.macroscopic()
    uxa = uxa.numpy()
    uya = uya.numpy()
    decay = math.exp(-2.0 * nu * k * k * steps)
    ex, ey = ux * decay, uy * decay
    l2 = 100.0 * math.sqrt(np.sum((uxa - ex) ** 2 + (uya - ey) ** 2)) / math.sqrt(np.sum(ex**2 + ey**2))
    redo_gamma = None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (data, title) in zip(axes, ((uxa, "AA ux"), (ex, "closed form ux"), (uxa - ex, "AA - closed form"))):
        im = ax.imshow(data, origin="lower", cmap="RdBu_r" if "diff" in title or "-" in title else "viridis")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"m17514 audit, independent re-run: N=48 t={steps}, relative L2 = {l2:.4f} %", fontsize=11)
    fig.tight_layout()
    p = OUT / "fig2_field_aa_vs_exact_independent.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, {"field_l2_pct_recomputed": l2, "shipped": RES["criteria"][2]["value"], "decay": decay}


def fig_hazard():
    hp = RES["hazard_probe"]
    crit = {c["name"]: c["value"] for c in RES["criteria"]}
    labels = ["half-window read\nsmooth field", "half-window read\nwhite noise",
              "parity-blind\nfield L2 (n48)", "parity-blind\ngamma_K err (n48)"]
    vals = [hp["smooth_tg"]["half_vs_complete_vel_l2_pct"], hp["white_noise"]["half_vs_complete_vel_l2_pct"],
            crit["parity_blind_arm_error_pct"],
            abs(RES["cases"][1]["metrics"]["aa_parity_blind"]["gamma_K_err_pct"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, vals, color=["tab:green", "tab:red", "tab:red", "tab:orange"])
    ax.axhline(15.0, color="k", ls="--", lw=1.2, label="owner 15 % gate")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.2f}%", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("error [%] (log)")
    ax.set_title("m17514 audit: is the even/odd tracker load bearing?")
    ax.legend()
    fig.tight_layout()
    p = OUT / "fig3_parity_hazard.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, {"hazard_smooth_pct": vals[0], "hazard_noise_pct": vals[1],
               "parity_blind_field_pct": vals[2], "parity_blind_gammaK_pct": vals[3]}


if __name__ == "__main__":
    out = {}
    for fn in (fig_gammaK, fig_field, fig_hazard):
        p, rec = fn()
        out[p.name] = rec
        print(p.name, json.dumps(rec, ensure_ascii=False))
    (OUT / "figures.json").write_text(json.dumps(out, indent=1))