"""Round-2 campaign for anchor ``XVAL-LBM-PHASEFIELD-MCMP-POISEUILLE``.

Produces the numbers and figures of
``docs/XVAL-LBM-PHASEFIELD-MCMP-POISEUILLE_R2.md``.  Everything printed here is
written to ``runs/phasefield_mcmp_poiseuille_r2/results.json`` so the report can
quote machine-generated values instead of transcribed ones.

Arms
----
A. free-energy parameterisation (surface tension / interface width) and the
   rest-state invariants: spurious current decay, mass conservation;
B. uniform-viscosity channel ladder against the exact parabolic solution;
C. viscosity-contrast ladder ``nu_low : nu_up = 3 : 1`` and ``1 : 3`` — the arm
   that a psi-blind (passive scalar) code cannot pass, plus the analytic
   separation between the contrast profile and the single-phase profile;
D. round-1 binding: the round-1 kernel as shipped is re-run to reproduce its
   published error, and the round-1 *wall schedule* is re-run inside the round-2
   kernel to isolate its contribution;
E. curved-interface (Laplace) attempt with a windowed bulk sampler.

Run: ``python3 scripts/xval_phasefield_mcmp_poiseuille_r2.py``
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorlbm.phasefield.soy_two_phase import (
    CS2,
    LayeredPoiseuille,
    SOYConfig,
    SOYTwoPhase,
    kappa_from_sigma_xi,
    surface_tension_from_kappa,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "phasefield_mcmp_poiseuille_r2"
OUT.mkdir(parents=True, exist_ok=True)

SIGMA, XI = 0.01, 4.0
GX = 1e-5
GATE = 0.15                      # owner precision standard
FIGURES: list[dict] = []


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return float(o)


def save_fig(fig, name: str, caption: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    FIGURES.append({"name": name, "caption": caption,
                    "png_base64": base64.b64encode(buf.getvalue()).decode()})
    plt.close(fig)


# ---------------------------------------------------------------------------
# A. parameterisation + rest-state invariants
# ---------------------------------------------------------------------------

def arm_a() -> dict:
    kappa = kappa_from_sigma_xi(SIGMA, XI)
    out = {
        "kappa": kappa,
        "sigma_roundtrip": surface_tension_from_kappa(kappa, XI),
        "xi_init": XI,
    }

    sim = SOYTwoPhase(SOYConfig(ny=34, nx=2, sigma=SIGMA, xi=XI, tau_a=1.0, tau_g=0.8))
    sim.set_planar(interface_y=1.0 + 0.5 * 32)
    out["interface_width_fit"] = sim.interface_width()
    m0 = sim.psi_mass()
    hist = []
    for n in range(1, 3001):
        sim.step()
        if n % 250 == 0:
            hist.append({"step": n, "max_velocity": sim.max_velocity(),
                         "mass_rel_drift": (sim.psi_mass() - m0) / abs(m0)})
    out["rest_history"] = hist
    out["rest_mass_rel_drift_final"] = hist[-1]["mass_rel_drift"]
    out["rest_max_velocity_final"] = hist[-1]["max_velocity"]

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    st = [h["step"] for h in hist]
    ax[0].semilogy(st, [max(h["max_velocity"], 1e-18) for h in hist], "o-")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel(r"$\max|\mathbf{u}|$ (lattice)")
    ax[0].set_title("rest state: spurious current decay")
    ax[0].grid(alpha=0.3)
    ax[1].plot(st, [h["mass_rel_drift"] for h in hist], "s-")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel(r"$\Delta M_\psi / |M_\psi|$")
    ax[1].set_title("order-parameter mass")
    ax[1].grid(alpha=0.3)
    save_fig(fig, "fig1_rest_invariants.png",
             "Rest state: parasitic-current decay and conservation of the order-parameter mass")

    # interface profile vs the tanh the parameterisation claims
    prof = sim.psi[1:-1, 0].cpu().numpy()
    y = np.arange(1, sim.ny - 1) - 0.5
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(y, prof, "o", ms=3, label="SOY kernel")
    ax.plot(y, np.tanh((y - (17.0 - 0.5)) / XI), "-", lw=1, label=r"$\tanh((y-y_i)/\xi)$")
    ax.set_xlabel("y (lattice, wall at 0)")
    ax.set_ylabel(r"$\psi$")
    ax.set_title(f"equilibrium interface, fitted width {out['interface_width_fit']:.3f} (xi={XI})")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "fig2_interface_profile.png",
             "Equilibrium phase-field profile against tanh((y-y_i)/xi); the fitted width recovers xi")
    return out


# ---------------------------------------------------------------------------
# B/C. channel ladders
# ---------------------------------------------------------------------------

def run_channel(ny: int, tau_a: float, tau_b: float | None, steps: int) -> dict:
    cfg = SOYConfig(ny=ny, nx=2, sigma=SIGMA, xi=XI, tau_a=tau_a, tau_b=tau_b,
                    tau_g=0.8, Gx=GX)
    sim = SOYTwoPhase(cfg)
    sim.set_planar(interface_y=1.0 + 0.5 * (ny - 2))
    mass0 = sim.psi_mass()
    t0 = time.time()
    for _ in range(steps):
        sim.step()
    ref = sim.layered_reference()
    res = sim.poiseuille_error(ref)
    res.update({
        "ny": ny, "H": ny - 2, "tau_a": tau_a, "tau_b": tau_b,
        "nu_low": ref.nu_low, "nu_up": ref.nu_up,
        "steps": steps, "wall_seconds": time.time() - t0,
        "mass_rel_drift": (sim.psi_mass() - mass0) / abs(mass0),
        "Ma": res["u_max_sim"] / math.sqrt(CS2),
        "Re": res["u_max_sim"] * (ny - 2) / ref.nu_up,
        "Cn": XI / (ny - 2),
    })
    return res


LADDERS = {
    "uniform": (1.0, None),
    "contrast_3to1": (0.8, 1.4),
    "contrast_1to3": (0.8, 0.6),
}


def arm_bc(subset=None) -> dict:
    ladder = [34, 50, 66]
    steps = 6000
    subset = subset or list(LADDERS)
    out: dict = {k: [] for k in subset}
    for ny in ladder:
        for key in subset:
            tau_a, tau_b = LADDERS[key]
            out[key].append(run_channel(ny, tau_a, tau_b, steps))

    # what a psi-blind (uniform-relaxation) solver would score against the
    # contrast reference -- analytic, no solver involved
    for H in (32.0, 48.0, 64.0):
        y = torch.arange(0.5, H, dtype=torch.float64)
        contrast = LayeredPoiseuille(H=H, h=H / 2, G=GX, nu_low=0.3, nu_up=0.1).profile(y)
        single = LayeredPoiseuille(H=H, h=H / 2, G=GX, nu_low=0.1, nu_up=0.1).profile(y)
        out.setdefault("analytic_separation", []).append({
            "H": H,
            "l2_single_vs_contrast": float((single - contrast).norm() / contrast.norm()),
        })

    # profiles
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(
            axes,
            ("uniform", "contrast_3to1"),
            ("uniform viscosity ($\\nu_{low}=\\nu_{up}$)",
             "viscosity contrast ($\\nu_{low}=3\\nu_{up}$)")):
        if not out.get(key):
            continue
        rec = max(out[key], key=lambda r: r["ny"])
        y = np.arange(1, rec["ny"] - 1) - 0.5
        ax.plot(rec["profile_sim"], y, "o", ms=3, label="SOY kernel (R2)")
        ax.plot(rec["profile_ana"], y, "-", lw=1.2, label="exact layered Poiseuille")
        ax.axhline(rec["H"] / 2, color="0.6", ls="--", lw=0.8, label="interface")
        ax.set_xlabel("$u_x$ (lattice)")
        ax.set_ylabel("y (lattice, wall at 0)")
        ax.set_title(f"{title}\n$H={rec['H']}$, L2={100 * rec['l2_rel']:.2f}%")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    save_fig(fig, "fig3_velocity_profiles.png",
             "Steady channel profiles against the exact layered Poiseuille solution (uniform and 3:1 viscosity contrast)")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    marks = {"uniform": ("uniform viscosity", "o-"),
             "contrast_3to1": (r"$\nu_{low}=3\nu_{up}$", "s-"),
             "contrast_1to3": (r"$\nu_{low}=\nu_{up}/3$", "^-")}
    for key, recs in out.items():
        if key not in marks or not isinstance(recs, list) or not recs:
            continue
        lab, mk = marks[key]
        ax.semilogy([r["H"] for r in recs], [max(r["l2_rel"], 1e-6) for r in recs], mk, label=lab)
    ax.axhline(GATE, color="r", ls="--", lw=1, label="owner gate 15%")
    ax.set_xlabel("channel height H (lattice)")
    ax.set_ylabel("relative L2 error of $u_x(y)$")
    ax.set_title("channel ladder: error vs exact solution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    save_fig(fig, "fig4_error_ladder.png",
             "Relative L2 error against the exact layered solution versus channel height, both viscosity arms, with the 15% acceptance gate")
    return out


# ---------------------------------------------------------------------------
# D. round-1 binding
# ---------------------------------------------------------------------------

def _load_round1():
    spec = importlib.util.spec_from_file_location(
        "r1_phasefield", ROOT / "xval" / "xval_phasefield_mcmp_poiseuille.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses need a registered module
    spec.loader.exec_module(mod)
    return mod


def arm_d() -> dict:
    out: dict = {}
    r1 = _load_round1()
    ny, nx, steps, h_frac = 48, 4, 5000, 0.5
    H, h = ny - 2, h_frac * (ny - 2)
    nu = (1.0 - 0.5) / 3.0
    sim = r1.SOYFreeEnergyLBM(ny, nx, sigma=SIGMA, xi=XI, tau_f=1.0, tau_g=0.8,
                              gamma=(0.8 - 0.5) / 3.0, dtype=torch.float64)
    sim._psi_wall = None
    sim.init_poiseuille(h_frac=h_frac, psi_wall=0.0, Gx=GX)
    for _ in range(steps):
        sim.step()
    u = np.array(sim.measure_poiseuille()["u_profile"])
    ana = np.array(r1.analytical_stratified_poiseuille(ny, float(H), h, GX, nu, nu)["profile"][1:-1])
    out["r1_kernel_as_shipped"] = {
        "ny": ny, "steps": steps,
        "l2_rel": float(np.linalg.norm(u - ana) / np.linalg.norm(ana)),
        "u_max": float(np.max(np.abs(u))),
        "u_max_ana": float(np.max(np.abs(ana))),
    }

    # same grid, round-2 kernel, round-1 wall schedule vs corrected walls
    for legacy in (True, False):
        cfg = SOYConfig(ny=ny, nx=2, sigma=SIGMA, xi=XI, tau_a=1.0, tau_g=0.8,
                        Gx=GX, legacy_walls=legacy)
        s2 = SOYTwoPhase(cfg)
        s2.set_planar(interface_y=1.0 + h)
        for _ in range(4000):
            s2.step()
        res = s2.poiseuille_error(s2.layered_reference())
        out["r2_kernel_round1_walls" if legacy else "r2_kernel_corrected_walls"] = {
            "ny": ny, "steps": 4000, "l2_rel": res["l2_rel"], "linf_rel": res["linf_rel"],
            "u_max_sim": res["u_max_sim"], "u_max_ana": res["u_max_ana"],
        }

    fig, ax = plt.subplots(figsize=(6, 3.4))
    keys = ["r1_kernel_as_shipped", "r2_kernel_round1_walls", "r2_kernel_corrected_walls"]
    labels = ["R1 kernel\n(as shipped)", "R2 kernel,\nR1 wall schedule", "R2 kernel,\ncorrected walls"]
    vals = [100 * out[k]["l2_rel"] for k in keys]
    bars = ax.bar(range(3), vals, color=["0.55", "tab:orange", "tab:green"])
    ax.axhline(100 * GATE, color="r", ls="--", lw=1, label="owner gate 15%")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.2f}%", ha="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("relative L2 error")
    ax.set_title("same physics, three wall treatments (H=46)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    save_fig(fig, "fig5_wall_defect_binding.png",
             "Error at H=46 for the round-1 kernel, the round-1 wall schedule inside the round-2 kernel, and corrected half-way bounce-back")
    return out


# ---------------------------------------------------------------------------
# E. curved interface (Laplace)
# ---------------------------------------------------------------------------

def arm_e() -> dict:
    out = {"radii": []}
    for R in (6.0, 9.0, 12.0):
        ny = int(2 * 1.9 * R) + 8
        cfg = SOYConfig(ny=ny, nx=ny, sigma=SIGMA, xi=XI, tau_a=1.0, tau_g=0.8)
        sim = SOYTwoPhase(cfg)
        sim.set_droplet(R)
        dist = sim._dist
        inner = dist < 0.5 * R
        outer = (dist > 1.3 * R) & (dist < 1.55 * R)
        t0 = time.time()
        for _ in range(4000):
            sim.step()
        rho = sim.macroscopic()["rho"]
        dp = CS2 * (float(rho[inner].mean()) - float(rho[outer].mean()))
        out["radii"].append({
            "R": R, "ny": ny, "x_extent": 1.9, "steps": 4000,
            "wall_seconds": time.time() - t0,
            "dP_measured": dp,
            "dP_laplace_sigma_over_R": SIGMA / R,
            "ratio": dp / (SIGMA / R),
            "max_velocity": sim.max_velocity(),
        })

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    inv = [1.0 / r["R"] for r in out["radii"]]
    ax.plot(inv, [r["dP_measured"] for r in out["radii"]], "o-", label="measured $\\Delta p$")
    ax.plot(inv, [SIGMA * i for i in inv], "k--", lw=1, label=r"$\sigma/R$")
    ax.set_xlabel("1/R (1/lattice)")
    ax.set_ylabel(r"$\Delta p$")
    ax.set_title("curved interface: pressure jump unreadable at the parasitic-current floor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_fig(fig, "fig6_laplace_attempt.png",
             "Attempted Laplace-law read-out: the density jump is not resolvable above the parasitic-current floor of the uniform-density SOY scheme")
    return out


# ---------------------------------------------------------------------------

ARMS = [
    ("A_parameterisation_and_rest", lambda: arm_a()),
    ("BC_channel_ladders", lambda: arm_bc(
        [k for k in (__import__("os").environ.get("ARM_BC_SUBSET") or ",".join(LADDERS)).split(",") if k])),
    ("D_round1_binding", lambda: arm_d()),
    ("E_laplace", lambda: arm_e()),
]


def main(argv=None) -> int:
    import os
    argv = sys.argv[1:] if argv is None else argv
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    t0 = time.time()
    results = {
        "anchor": "XVAL-LBM-PHASEFIELD-MCMP-POISEUILLE",
        "round": 2,
        "config": {"sigma": SIGMA, "xi": XI, "Gx": GX, "gate": GATE},
        "oss_counterpart": "ebinan92/Fingering_dynamics (phasefield-mcmp-poiseuille)",
        "arms": {},
    }
    if not argv and (OUT / "results.json").exists():
        results = json.loads((OUT / "results.json").read_text())
    for name, fn in ARMS:
        if argv and name not in argv:
            continue
        print(f"[{name}] ...", flush=True)
        ta = time.time()
        prev = results.get("arms", {}).get(name, {})
        fresh = fn()
        if name == "BC_channel_ladders":
            fresh = {**prev, **fresh}
        results.setdefault("arms", {})[name] = fresh
        results["arms"][name]["_seconds"] = time.time() - ta
        print(f"[{name}] done in {time.time() - ta:.0f}s", flush=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=2, default=_jsonable))
        (OUT / "figures_b64.json").write_text(json.dumps(FIGURES))

    results["total_seconds"] = time.time() - t0
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=_jsonable))
    (OUT / "figures_b64.json").write_text(json.dumps(FIGURES))
    print("wrote", OUT / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())