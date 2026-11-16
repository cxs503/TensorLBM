"""XVAL round 2 -- wall/boundary node types for the pressure-evolution + Allen-Cahn
phase-field model (:mod:`tensorlbm.phasefield.pf_pressure_evolution`).

Round 1 (mission m18194) ported the periodic kernel of the CFD-GO/TCLB model
``models/multiphase/d2q9_pf_pressureEvolution`` and verified it against the 2D
Young-Laplace law, Taylor-Green decay and phase-mass conservation, but left the
reference model's ``Solid``/``Wall``/``EVelocity`` node types unported, so no
wall-bounded case existed.  This driver closes that gap and verifies the new
boundary treatment against *analytic* references that wall conditions make
available:

  C1  moving-wall plane Couette            u_x = U y / H            (EVelocity)
  C2  single-phase plane Poiseuille        u_x = F y (H-y) / (2 mu) (Solid, no-slip)
  C3  two-layer (layered) Poiseuille       piecewise parabolic, exact
  C4  transport matrix: density-only / viscosity-only / both, to identify which
      shear-continuity law the discrete model reproduces
  C5  wall phase condition: phase mass over fluid nodes, interface flatness
      (neutral wetting, d_n phi = 0)

Two references are used for the layered case:

* ``sharp`` -- the textbook sharp-interface solution with continuity of
  ``mu du/dy`` (``mu = rho nu``) or of ``nu du/dy`` (per-phase acceleration).
* ``diffuse`` -- a one-dimensional finite-difference solve of
  ``d/dy(mu(y) du/dy) = -F`` on the *measured* phi(y) profile with the model's own
  interpolation (``rho`` linear in phi, ``omega = 1/(3 nu)`` linear in phi).
  This isolates solver/boundary error from the finite interface width ``W``.

Run: python scripts/xval_pf_pressure_walls_r2.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.phasefield import pf_pressure_evolution as M  # noqa: E402

OUT = pathlib.Path("outputs/xval_pf_walls_r2")
DEV = "cpu"
DT = torch.float64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


WALL_T = 2   # wall band thickness: the mixed (2-cell) stencil needs two wall rows


def _channel(ny: int, nx: int, thick: int = WALL_T) -> torch.Tensor:
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[:thick, :] = True
    solid[ny - thick:, :] = True
    return solid


def _inner(ny: int, thick: int = WALL_T) -> slice:
    return slice(thick, ny - thick)


def _height(ny: int, thick: int = WALL_T) -> float:
    return float(ny - 2 * thick)


def _y_eff(ny: int, thick: int = WALL_T) -> torch.Tensor:
    """Analysis coordinate of the interior nodes: halfway-BB walls at 0 and H."""
    return torch.arange(thick, ny - thick, dtype=DT) - (thick - 0.5)


def _tanh_layer(ny: int, nx: int, y_int: float, width: float) -> torch.Tensor:
    """``phi = 1`` (heavy) below ``y_int``, ``0`` above; ``y_int`` in node units."""
    y = torch.arange(ny, dtype=DT).view(-1, 1)
    return (0.5 * (1.0 - torch.tanh(2.0 * (y - y_int) / width))).expand(ny, nx).contiguous()


def _solve_diffuse_reference(phi_col: np.ndarray, rho_h, rho_l, nu_h, nu_l, force):
    """FD solution of ``d/dy(mu du/dy) = -F`` on the measured phi profile.

    ``phi_col`` are the interior-node values of phi (length ``ny-2``); walls sit
    half a cell beyond the first/last node, at ``y = 0`` and ``y = H``.
    """
    frac = np.clip(phi_col, 0.0, 1.0)
    rho = rho_l + (rho_h - rho_l) * frac
    omega = (1.0 / (3.0 * nu_l)) + ((1.0 / (3.0 * nu_h)) - (1.0 / (3.0 * nu_l))) * frac
    mu = rho / (3.0 * omega)
    n = phi_col.size
    a_mat = np.zeros((n, n))
    b = np.full(n, -force)
    for j in range(n):
        lo_mu = mu[j] if j == 0 else 0.5 * (mu[j] + mu[j - 1])
        hi_mu = mu[j] if j == n - 1 else 0.5 * (mu[j] + mu[j + 1])
        d_lo = 0.5 if j == 0 else 1.0
        d_hi = 0.5 if j == n - 1 else 1.0
        a_mat[j, j] = -(lo_mu / d_lo + hi_mu / d_hi)
        if j > 0:
            a_mat[j, j - 1] = lo_mu / d_lo
        if j < n - 1:
            a_mat[j, j + 1] = hi_mu / d_hi
    return np.linalg.solve(a_mat, b)


def _rel_metrics(u: np.ndarray, ref: np.ndarray, band: np.ndarray | None = None) -> dict:
    scale = float(np.abs(ref).max())
    err = u - ref
    out = {
        "l2_rel": float(np.linalg.norm(err) / np.linalg.norm(ref)),
        "max_rel": float(np.abs(err).max() / scale),
    }
    if band is not None:
        out["max_rel_out_band"] = float(np.abs(err[band]).max() / scale)
    return out


# ---------------------------------------------------------------------------
# case runners
# ---------------------------------------------------------------------------


def run_couette(ny=32, nx=4, nu=0.1, wall_speed=0.01, steps=6000) -> dict:
    sol = M.PFPressureEvolution(
        shape=(ny, nx), rho_h=1.0, rho_l=1.0, phi_h=0.0, phi_l=0.0,
        sigma=0.05, width=4.0, viscosity_h=nu, viscosity_l=nu, dtype=DT, device=DEV,
    )
    solid = _channel(ny, nx)
    uw = (torch.zeros((ny, nx), dtype=DT), torch.zeros((ny, nx), dtype=DT))
    uw[0][-WALL_T:, :] = wall_speed     # top wall moves in +x
    sol.set_walls(solid, uw)
    y = _y_eff(ny)
    height = _height(ny)
    sl = _inner(ny)
    u_init = torch.zeros((ny, nx), dtype=DT)
    u_init[sl, :] = (wall_speed * y / height).view(-1, 1).expand(y.numel(), nx)
    sol.initialize(torch.zeros((ny, nx), dtype=DT), u0=(u_init, torch.zeros_like(u_init)))
    sol.run(steps)
    u = sol.velocity()
    prof = u[0][sl, :].mean(1).numpy()
    ref = np.array(M.couette_profile(y, height, wall_speed).tolist())
    return {
        "case": "C1_couette_moving_wall",
        "ny": ny, "nx": nx, "nu": nu, "wall_speed": wall_speed, "steps": steps,
        "u_profile": prof.tolist(), "u_ref": ref.tolist(),
        "uy_max_interior": float(u[1][sl].abs().max()),
        "metrics": _rel_metrics(prof, ref),
        "near_wall_u_at_y_eff_H-0.5": float(prof[-1]),
        "ref_u_at_y_eff_H-0.5": float(ref[-1]),
    }


def run_single_phase_poiseuille(ny=32, nx=4, nu=0.1, force=1e-5, steps=4000) -> dict:
    sol = M.PFPressureEvolution(
        shape=(ny, nx), rho_h=1.0, rho_l=1.0, phi_h=0.0, phi_l=0.0,
        sigma=0.05, width=4.0, viscosity_h=nu, viscosity_l=nu, dtype=DT, device=DEV,
    )
    solid = _channel(ny, nx)
    sol.set_walls(solid)
    y = _y_eff(ny)
    height = _height(ny)
    sl = _inner(ny)
    body = (torch.full((ny, nx), force, dtype=DT), torch.zeros((ny, nx), dtype=DT))
    ui = torch.zeros((ny, nx), dtype=DT)
    ui[sl, :] = M.poiseuille_profile(y, height, force, nu).view(-1, 1).expand(y.numel(), nx)
    sol.initialize(torch.zeros((ny, nx), dtype=DT), u0=(ui, torch.zeros_like(ui)), body=body)
    sol.run(steps, body=body)
    prof = sol.velocity()[0][sl, :].mean(1).numpy()
    ref = np.array(M.poiseuille_profile(y, height, force, nu).tolist())
    return {
        "case": "C2_single_phase_poiseuille",
        "ny": ny, "nx": nx, "nu": nu, "force": force, "steps": steps,
        "u_profile": prof.tolist(), "u_ref": ref.tolist(),
        "u_max_got": float(prof.max()), "u_max_ref": float(ref.max()),
        "err_umax_pct": float((prof.max() - ref.max()) / ref.max() * 100.0),
        "metrics": _rel_metrics(prof, ref),
    }


def run_layered(
    ny=32, nx=4, width=4.0, rho_h=1.5, rho_l=1.0, nu_h=0.05, nu_l=0.2,
    force=1e-5, sigma=0.05, mobility=0.05, steps=3500, label="",
) -> dict:
    sol = M.PFPressureEvolution(
        shape=(ny, nx), rho_h=rho_h, rho_l=rho_l, sigma=sigma, width=width,
        mobility=mobility, viscosity_h=nu_h, viscosity_l=nu_l, dtype=DT, device=DEV,
    )
    solid = _channel(ny, nx)
    sol.set_walls(solid)
    sl = _inner(ny)
    height = _height(ny)
    y_int_eff = height / 2.0
    phi0 = _tanh_layer(ny, nx, y_int_eff + WALL_T - 0.5, width)
    body = (torch.full((ny, nx), force, dtype=DT), torch.zeros((ny, nx), dtype=DT))

    y = _y_eff(ny)
    ref_dyn = np.array(M.layered_poiseuille_profile(
        y, height, y_int_eff, rho_h * nu_h, rho_l * nu_l, force, force).tolist())
    ref_kin = np.array(M.layered_poiseuille_profile(
        y, height, y_int_eff, nu_h, nu_l, force / rho_h, force / rho_l).tolist())

    ui = torch.zeros((ny, nx), dtype=DT)
    ui[sl, :] = torch.as_tensor(ref_dyn, dtype=DT).view(-1, 1).expand(y.numel(), nx)
    sol.initialize(phi0, u0=(ui, torch.zeros_like(ui)), body=body)
    t0 = time.time()
    sol.run(steps, body=body)
    wall = time.time() - t0

    u = sol.velocity()
    prof = u[0][sl, :].mean(1).numpy()
    phi_col = sol.phi[sl, :].mean(1).numpy()
    ref_dif = _solve_diffuse_reference(phi_col, rho_h, rho_l, nu_h, nu_l, force)

    band = np.abs(y.numpy() - y_int_eff) > width      # outside the diffuse interface
    # interface position drift: mean y of the phi = 0.5 crossing per column
    phi_full = sol.phi.mean(1).numpy()
    yc = np.arange(ny)
    above = np.where(phi_full > 0.5)[0]
    y_half = float(above.max()) if above.size else float("nan")

    return {
        "case": f"C3_layered_poiseuille{label}",
        "ny": ny, "nx": nx, "width": width, "steps": steps,
        "rho_h": rho_h, "rho_l": rho_l, "nu_h": nu_h, "nu_l": nu_l,
        "force": force, "sigma": sigma, "mobility": mobility,
        "wallclock_s": round(wall, 1),
        "u_profile": prof.tolist(),
        "ref_sharp_dyn": ref_dyn.tolist(),
        "ref_sharp_kin": ref_kin.tolist(),
        "ref_diffuse": ref_dif.tolist(),
        "phi_profile": phi_col.tolist(),
        "metrics_vs_sharp_dyn": _rel_metrics(prof, ref_dyn, band),
        "metrics_vs_sharp_kin": _rel_metrics(prof, ref_kin, band),
        "metrics_vs_diffuse": _rel_metrics(prof, ref_dif, band),
        "umax": float(prof.max()),
        "umax_sharp_dyn": float(ref_dyn.max()),
        "err_umax_pct": float((prof.max() - ref_dyn.max()) / ref_dyn.max() * 100.0),
        "uy_max_interior": float(u[1][sl].abs().max()),
        "mass_err": float(sol.mass_error()),
        "phi_min": float(sol.phi.min()), "phi_max": float(sol.phi.max()),
        "interface_y_half": y_half,
        "interface_y_half_init": float(y_int_eff + WALL_T - 0.5),
    }


def run_wall_phase_control(ny=64, nx=8, width=4.0, steps=4000) -> dict:
    """Static layered phi between walls: mass conservation + interface flatness."""
    sol = M.PFPressureEvolution(
        shape=(ny, nx), rho_h=1.5, rho_l=1.0, sigma=0.05, width=width,
        mobility=0.05, viscosity_h=0.05, viscosity_l=0.2, dtype=DT, device=DEV,
    )
    sol.set_walls(_channel(ny, nx))
    y0 = _height(ny) / 2.0 + WALL_T - 0.5
    sol.initialize(_tanh_layer(ny, nx, y0, width))
    hist = []
    for k in range(steps):
        sol.step()
        if k % 500 == 0:
            phi = sol.phi.mean(1).numpy()
            above = np.where(phi > 0.5)[0]
            hist.append({
                "step": k,
                "mass_err": sol.mass_error(),
                "interface_y_half": float(above.max()) if above.size else None,
                "uy_max": float(sol.velocity()[1][_inner(ny)].abs().max()),
            })
    phi = sol.phi.mean(1).numpy()
    above = np.where(phi > 0.5)[0]
    return {
        "case": "C5_wall_phase_control",
        "ny": ny, "nx": nx, "width": width, "steps": steps,
        "history": hist,
        "mass_err_final": float(sol.mass_error()),
        "interface_y_half_final": float(above.max()) if above.size else None,
        "interface_y_half_init": float(y0),
        "phi_min": float(sol.phi.min()), "phi_max": float(sol.phi.max()),
        "uy_max": float(sol.velocity()[1][_inner(ny)].abs().max()),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--local", action="store_true",
                    help="one small case per family, <10 min single-core budget")
    ap.add_argument("--subset", action="store_true",
                    help="3 discriminating cases only (local <10 min budget)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))
    OUT.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []

    if args.local:
        print("[C1-local] moving-wall Couette", flush=True)
        cases.append(run_couette(ny=24, nx=2, steps=2500))
        print("   L2=%.4f" % cases[-1]["metrics"]["l2_rel"], flush=True)
        print("[C2-local] single-phase Poiseuille", flush=True)
        cases.append(run_single_phase_poiseuille(ny=24, nx=2, steps=2500))
        print("   err(u_max)=%+.2f%%" % cases[-1]["err_umax_pct"], flush=True)
        print("[C3-local] layered Poiseuille (both)", flush=True)
        cases.append(run_layered(ny=32, nx=2, steps=3000, rho_h=1.5, rho_l=1.0,
                                 nu_h=0.05, nu_l=0.2, label="_local_both"))
        c = cases[-1]
        print("   dev_dyn=%.4f dev_kin=%.4f dev_dif=%.4f uy=%.2e mass=%.1e"
              % (c["metrics_vs_sharp_dyn"]["max_rel"], c["metrics_vs_sharp_kin"]["max_rel"],
                 c["metrics_vs_diffuse"]["max_rel"], c["uy_max_interior"], c["mass_err"]), flush=True)
        print("[C5-local] wall phase control", flush=True)
        cases.append(run_wall_phase_control(ny=32, nx=2, steps=500))
        print("   mass=%.1e interface=%s" % (cases[-1]["mass_err_final"],
                                             cases[-1]["interface_y_half_final"]), flush=True)
        _write_summary(cases, args)
        _figures(cases)
        return 0

    if args.subset:
        # the three cases that separate the two candidate shear-continuity laws
        for name, kw in (
            ("density_only", dict(rho_h=2.0, rho_l=1.0, nu_h=0.1, nu_l=0.1)),
            ("viscosity_only", dict(rho_h=1.0, rho_l=1.0, nu_h=0.05, nu_l=0.2)),
            ("both", dict(rho_h=1.5, rho_l=1.0, nu_h=0.05, nu_l=0.2)),
        ):
            print(f"[C4-subset] {name}", flush=True)
            cases.append(run_layered(ny=32, nx=2, steps=4000, label=f"_{name}", **kw))
            c = cases[-1]
            print("   dev_dyn=%.4f dev_kin=%.4f dev_dif=%.4f uy=%.2e"
                  % (c["metrics_vs_sharp_dyn"]["max_rel"],
                     c["metrics_vs_sharp_kin"]["max_rel"],
                     c["metrics_vs_diffuse"]["max_rel"], c["uy_max_interior"]), flush=True)
        _write_summary(cases, args)
        _figures(cases)
        return 0

    print("[C1] moving-wall Couette", flush=True)
    cases.append(run_couette(steps=4000 if args.quick else 6000))
    print(f"     L2={cases[-1]['metrics']['l2_rel']:.4f}", flush=True)

    print("[C2] single-phase Poiseuille", flush=True)
    cases.append(run_single_phase_poiseuille(steps=3000 if args.quick else 4000))
    print(f"     err(u_max)={cases[-1]['err_umax_pct']:+.2f}%", flush=True)

    print("[C4] transport matrix (ny=32, W=4)", flush=True)
    matrix = [
        ("uniform", dict(rho_h=1.5, rho_l=1.5, nu_h=0.1, nu_l=0.1)),
        ("density_only", dict(rho_h=2.0, rho_l=1.0, nu_h=0.1, nu_l=0.1)),
        ("viscosity_only", dict(rho_h=1.0, rho_l=1.0, nu_h=0.05, nu_l=0.2)),
        ("both", dict(rho_h=1.5, rho_l=1.0, nu_h=0.05, nu_l=0.2)),
    ]
    for name, kw in matrix:
        print(f"  - {name}", flush=True)
        cases.append(run_layered(
            steps=2500 if args.quick else 4000, label=f"_{name}", **kw))

    print("[C3] grid scan (W=4, both)", flush=True)
    for ny in (24, 32, 48):
        print(f"  - ny={ny}", flush=True)
        cases.append(run_layered(ny=ny, rho_h=1.5, rho_l=1.0, nu_h=0.05, nu_l=0.2,
                                 steps=2500 if args.quick else (3000 if ny < 32 else 4000),
                                 label=f"_grid_ny{ny}"))

    print("[C3b] interface-width scan (ny=32, both)", flush=True)
    for w in (2.0, 3.0):
        print(f"  - W={w}", flush=True)
        cases.append(run_layered(width=w, rho_h=1.5, rho_l=1.0, nu_h=0.05, nu_l=0.2,
                                 steps=2500 if args.quick else 4000,
                                 label=f"_width_{w:g}"))

    print("[C5] wall phase control", flush=True)
    cases.append(run_wall_phase_control(ny=48, nx=8, steps=1200 if args.quick else 2500))

    _write_summary(cases, args)
    _figures(cases)
    return 0


def _write_summary(cases: list[dict], args) -> None:
    summary = {
        "mission": 23233,
        "anchor": "XVAL-LBM-PRESSURE-EVOLUTION-ALLENCAHN-PHASEFIELD",
        "round": 2,
        "scope": "wall/boundary node types (Solid no-slip, EVelocity moving wall, "
                 "neutral-wetting phi) for the pressure-evolution + Allen-Cahn port",
        "torch": torch.__version__,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": cases,
        "quick": bool(args.quick),
        "subset": bool(getattr(args, "subset", False)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {OUT/'summary.json'}", flush=True)


def _figures(cases: list[dict]) -> None:
    """Profile figures (numeric vs analytic) + a metrics bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - figures are optional
        print(f"[figures] skipped: {exc}", flush=True)
        return

    lay = [c for c in cases if c["case"].startswith("C3_layered")]
    others = [c for c in cases if c["case"].startswith(("C1_", "C2_"))]

    # fig 1: layered profiles vs the two analytic references
    if lay:
        fig, axes = plt.subplots(1, min(3, len(lay)), figsize=(4.2 * min(3, len(lay)), 4.2),
                                 squeeze=False)
        for ax, c in zip(axes[0], lay):
            ny = c["ny"]
            y = np.arange(WALL_T, ny - WALL_T) - (WALL_T - 0.5)
            ax.plot(c["u_profile"], y, "o", ms=3, label="LBM (pressure evolution)")
            ax.plot(c["ref_sharp_dyn"], y, "-", lw=1.2, label=r"sharp analytic, $\mu\,\partial_y u$ cont.")
            ax.plot(c["ref_sharp_kin"], y, "--", lw=1.1, label=r"sharp analytic, $\nu\,\partial_y u$ cont.")
            ax.plot(c["ref_diffuse"], y, ":", lw=1.6, label="diffuse-interface FD reference")
            ax.set_title(c["case"].replace("C3_layered_poiseuille_", ""), fontsize=9)
            ax.set_xlabel(r"$u_x$ (lattice units)")
            ax.set_ylabel(r"$y$ (lattice units)")
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "fig1_layered_profiles.png", dpi=150)
        plt.close(fig)

    # fig 2: wall-BC validations (Couette + single-phase Poiseuille)
    if others:
        fig, axes = plt.subplots(1, len(others), figsize=(4.2 * len(others), 4.2), squeeze=False)
        for ax, c in zip(axes[0], others):
            ny = c["ny"]
            y = np.arange(WALL_T, ny - WALL_T) - (WALL_T - 0.5)
            ax.plot(c["u_profile"], y, "o", ms=3, label="LBM")
            ax.plot(c["u_ref"], y, "-", lw=1.2, label="analytic")
            ax.set_title(c["case"], fontsize=9)
            ax.set_xlabel(r"$u_x$")
            ax.set_ylabel(r"$y$")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "fig2_wall_bc_validations.png", dpi=150)
        plt.close(fig)

    # fig 3: relative deviation of every layered case against the three references
    if lay:
        labels = [c["case"].replace("C3_layered_poiseuille_", "") for c in lay]
        xs = np.arange(len(lay))
        fig, ax = plt.subplots(figsize=(max(6.0, 0.85 * len(lay)), 3.6))
        for key, lab, mk in (
            ("metrics_vs_sharp_dyn", "vs sharp analytic, dynamic-viscosity continuity", "o"),
            ("metrics_vs_sharp_kin", "vs sharp analytic, kinematic-viscosity continuity", "s"),
            ("metrics_vs_diffuse", "vs diffuse-interface FD reference", "^"),
        ):
            ax.plot(xs, [100.0 * c[key]["max_rel"] for c in lay], mk + "-", label=lab)
        ax.axhline(15.0, color="r", ls="--", lw=1.0, label="15% acceptance")
        ax.axhline(3.0, color="orange", ls=":", lw=1.0, label="3% target")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(r"$\max|u-u_{ref}|/\max u_{ref}$  (%)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "fig3_error_summary.png", dpi=150)
        plt.close(fig)
    print(f"[figures] written to {OUT}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())