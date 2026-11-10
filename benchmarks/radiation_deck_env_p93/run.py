"""P93W3-N02 benchmark: sphere-plate / cylinder-plate view factors, Gauss-Seidel
radiosity, and the ship-deck temperature *field* under a solar/sky/sea boundary.

Campaign node P93W3-N02 pushes the TensorLBM thermal-radiation surface from
"panel-level load with a direct linear solve" to "REAL field capability":

(a) the five-configuration closed-form view-factor family (parallel rectangles,
    perpendicular rectangles, coaxial disks, **sphere-plate**, **cylinder-plate**)
    is checked against independent estimators (multi-dimensional Gauss-Legendre
    quadrature of the double-area integral, definitional Monte-Carlo ray tracing)
    and catalogue values, worst relative error <= 1 %;
(b) the gray-diffuse enclosure radiosity solved by **Gauss-Seidel sweeps**
    reaches an iteration residual <= 1e-6 and reproduces the dense LU solution;
(c) an **isothermal** closed cavity exchanges no net power: the enclosure energy
    residual stays <= 0.1 % for every emissivity;
(d) the steady **deck temperature field** from a D2Q5 thermal lattice-Boltzmann
    solver agrees with two independent analytic solutions to better than 3 %:
    through-thickness (nonlinear surface balance by bisection + linear interior)
    and a 2-D shadow edge (exact cosine-series solution of the linearised
    problem).

Outputs: ``result.json`` here, ``artifacts/P93W3_N02_radiation_deck_env.json``,
``figures/p93w3_n02/*.png``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.deck_thermal_field import (  # noqa: E402
    DeckMaterial,
    analytic_steady_2d_fourier,
    analytic_steady_field_1d,
    solar_flux_components,
    solve_deck_field,
    surface_net_flux,
)
from tensorlbm.deck_thermal_load import (  # noqa: E402
    DeckPanel,
    SkySeaEnvironment,
    SunState,
    absorbed_solar_flux,
)
from tensorlbm.gray_enclosure_radiosity import (  # noqa: E402
    STEFAN_BOLTZMANN,
    GrayEnclosure,
    GraySurface,
    concentric_spheres_enclosure,
    cube_enclosure,
)
from tensorlbm.radiative_view_factors import (  # noqa: E402
    monte_carlo_sphere_to_infinite_plane,
    monte_carlo_view_factor_rectangles,
    numerical_element_to_infinite_cylinder,
    numerical_element_to_sphere,
    numerical_parallel_coaxial_disks,
    numerical_parallel_equal_rectangles,
    vf_element_to_infinite_cylinder,
    vf_element_to_sphere,
    vf_infinite_cylinder_to_parallel_plane,
    vf_parallel_coaxial_disks,
    vf_parallel_equal_rectangles,
    vf_perpendicular_rectangles_common_edge,
    vf_sphere_to_infinite_plane,
)
from tensorlbm.radiosity_gauss_seidel import (  # noqa: E402
    solve_radiosity_gauss_seidel,
    spectral_radius,
)

TOL_A = 0.01  # 1 % view-factor gate
TOL_B = 1e-6  # iteration residual gate
TOL_C = 1e-3  # 0.1 % isothermal-cavity gate
TOL_D = 0.03  # 3 % deck-field gate


# ---------------------------------------------------------------------------
# environment / geometry shared by the deck cases
# ---------------------------------------------------------------------------


def deck_case() -> tuple[DeckPanel, SkySeaEnvironment, DeckMaterial]:
    panel = DeckPanel(tilt_deg=0.0, emissivity=0.9, solar_absorptance=0.6, h_conv=12.0)
    sun = SunState(
        beam_irradiance=800.0, elevation_deg=60.0, azimuth_deg=180.0, diffuse_horizontal=120.0
    )
    env = SkySeaEnvironment(T_air=303.0, T_dewpoint=288.0, T_sea=293.0, sun=sun)
    return panel, env, DeckMaterial(k=45.0, rho=7800.0, cp=500.0, thickness=0.025)


def cylinder_to_plane_mc(R: float, h: float, n_rays: int = 400_000, seed: int = 7) -> dict:
    """Definitional Monte Carlo for infinite-cylinder -> parallel plane (1/2).

    Points uniform on the cylinder, cosine-weighted directions about the outward
    normal; every ray with ``d_z < 0`` reaches the plane (convex body, no
    self-shadowing).  Purely geometric, so it certifies the convex-body theorem
    independently of the module's algebra.
    """
    rng = np.random.default_rng(seed)
    phi = 2.0 * math.pi * rng.random(n_rays)
    nvec = np.stack([np.sin(phi), np.zeros_like(phi), np.cos(phi)], axis=1)
    r_a = rng.random(n_rays)
    r_b = rng.random(n_rays)
    ct = np.sqrt(1.0 - r_a)
    st = np.sqrt(r_a)
    ph = 2.0 * math.pi * r_b
    e1 = np.stack([np.cos(phi), np.zeros_like(phi), -np.sin(phi)], axis=1)
    e2 = np.stack([np.zeros_like(phi), np.ones_like(phi), np.zeros_like(phi)], axis=1)
    d = (st * np.cos(ph))[:, None] * e1 + (st * np.sin(ph))[:, None] * e2 + ct[:, None] * nvec
    F = float(np.mean(d[:, 2] < 0.0))
    return {"F": F, "stderr": float(math.sqrt(F * (1.0 - F) / n_rays)), "n_rays": int(n_rays)}


# ---------------------------------------------------------------------------
# criterion (a): five-configuration closed-form family
# ---------------------------------------------------------------------------


def criterion_a() -> dict:
    rows: list[dict] = []

    def add(family, params, F_cf, F_ref, method, stderr=None):
        denom = abs(F_ref) if abs(F_ref) > 1e-300 else 1.0
        rows.append(
            {
                "family": family,
                "params": params,
                "F_closed_form": float(F_cf),
                "F_reference": float(F_ref),
                "rel_error": float(abs(F_cf - F_ref) / denom),
                "reference_method": method,
                **({} if stderr is None else {"reference_stderr": float(stderr)}),
            }
        )

    # 1) parallel rectangles
    for X, Y in [(0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (3.0, 4.0)]:
        add(
            "parallel_rect",
            {"X": X, "Y": Y},
            vf_parallel_equal_rectangles(X, Y),
            numerical_parallel_equal_rectangles(X, Y, n=60),
            "4-D Gauss-Legendre (n=60) of the double-area integral",
        )

    # 2) perpendicular rectangles sharing an edge
    for H, W in [(0.5, 0.5), (1.0, 1.0), (2.0, 1.0)]:
        s1 = ((0.0, 0.0, 0.0), (H, 0.0, 0.0), (0.0, 1.0, 0.0))
        s2 = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, W))
        mc = monte_carlo_view_factor_rectangles(s1, s2, n_rays=600_000, seed=11)
        add(
            "perp_rect",
            {"H": H, "W": W},
            vf_perpendicular_rectangles_common_edge(H, W),
            mc["F"],
            "Monte-Carlo ray tracing (6e5 rays)",
            mc["stderr"],
        )

    # 3) coaxial parallel disks
    for R1, R2, S in [(1.0, 1.0, 1.0), (0.5, 1.0, 1.0), (2.0, 1.0, 3.0)]:
        add(
            "coax_disks",
            {"R1": R1, "R2": R2, "S": S},
            vf_parallel_coaxial_disks(R1, R2, S),
            numerical_parallel_coaxial_disks(R1, R2, S, n=60),
            "polar Gauss-Legendre quadrature (n=60)",
        )

    # 4) sphere-plate (new)
    for R, d in [(1.0, 2.0), (1.0, 5.0), (2.0, 2.5)]:
        add(
            "sphere_plate",
            {"R": R, "d": d, "mode": "element->sphere"},
            vf_element_to_sphere(R, d),
            numerical_element_to_sphere(R, d),
            "3-D surface quadrature of the double-area integral",
        )
    mc = monte_carlo_sphere_to_infinite_plane(1.0, 1.5, n_rays=400_000, seed=3)
    add(
        "sphere_plate",
        {"R": 1.0, "h": 1.5, "mode": "sphere->infinite plane"},
        vf_sphere_to_infinite_plane(1.0, 1.5),
        mc["F"],
        "Monte-Carlo ray tracing (4e5 rays)",
        mc["stderr"],
    )

    # 5) cylinder-plate (new)
    for R, h in [(1.0, 2.0), (0.4, 2.0), (0.9, 1.0)]:
        add(
            "cylinder_plate",
            {"R": R, "h": h, "mode": "element->infinite cylinder"},
            vf_element_to_infinite_cylinder(R, h),
            numerical_element_to_infinite_cylinder(R, h, n_phi=200, n_psi=200),
            "3-D surface quadrature with y = sqrt(D) tan(psi) substitution",
        )
    mc = cylinder_to_plane_mc(0.8, 2.0, n_rays=400_000)
    add(
        "cylinder_plate",
        {"R": 0.8, "h": 2.0, "mode": "infinite cylinder->infinite plane"},
        vf_infinite_cylinder_to_parallel_plane(0.8, 2.0),
        mc["F"],
        "Monte-Carlo ray tracing (4e5 rays)",
        mc["stderr"],
    )

    families: dict[str, dict] = {}
    for row in rows:
        fam = families.setdefault(row["family"], {"n": 0, "max_rel_error": 0.0})
        fam["n"] += 1
        fam["max_rel_error"] = max(fam["max_rel_error"], row["rel_error"])
    worst = max(r["rel_error"] for r in rows)
    return {
        "tolerance": TOL_A,
        "pass": bool(worst <= TOL_A),
        "worst_rel_error": float(worst),
        "n_configurations": len(rows),
        "family_summary": families,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# criterion (b): Gauss-Seidel radiosity residual
# ---------------------------------------------------------------------------


def enclosure_set() -> dict[str, GrayEnclosure]:
    A, A_p = 24.0, 1000.0 * 24.0
    F_deck = np.array(
        [
            [0.0, 0.75, 0.25],
            [A * 0.75 / A_p, 1.0 - A * 0.75 / A_p, 0.0],
            [A * 0.25 / A_p, 0.0, 1.0 - A * 0.25 / A_p],
        ]
    )
    A3 = np.array([1.0, 2.0, 3.0])
    F3 = np.array(
        [
            [0.0, 0.4, 0.6],
            [0.2, 0.0, 0.8],
            [0.2, 1.6 / 3.0, 1.0 - 0.2 - 1.6 / 3.0],
        ]
    )
    return {
        "cube_two_eps": cube_enclosure(1.0, 0.9, 0.2, 700.0, 400.0),
        "cube_low_eps": cube_enclosure(1.0, 0.1, 0.05, 900.0, 350.0),
        "concentric_spheres": concentric_spheres_enclosure(1.0, 4.0, 0.8, 0.4, 600.0, 300.0),
        "three_surface_reradiating": GrayEnclosure(
            [
                GraySurface(A3[0], 0.9, 800.0, name="hot"),
                GraySurface(A3[1], 0.1, 400.0, name="cold"),
                GraySurface(A3[2], 0.0, None, name="reradiating"),
            ],
            F3,
        ),
        "deck_sky_sea_open": GrayEnclosure(
            [
                GraySurface(A, 0.9, 320.0, name="deck"),
                GraySurface(A_p, 1.0, 280.0, name="sky"),
                GraySurface(A_p, 0.96, 293.0, name="sea"),
            ],
            F_deck,
            summation_tol=1e-6,
        ),
    }


def criterion_b() -> dict:
    rows = []
    for name, enc in enclosure_set().items():
        lu = enc.solve()
        gs = solve_radiosity_gauss_seidel(enc, tol=TOL_B, record_history=True)
        ref = np.max(np.abs(lu.radiosity))
        rows.append(
            {
                "enclosure": name,
                "n_surfaces": int(len(enc.surfaces)),
                "iterations": int(gs.iterations),
                "final_residual": float(gs.final_residual),
                "algebraic_residual": float(gs.algebraic_residual),
                "energy_residual_vs_lu": float(gs.energy_residual),
                "spectral_radius": float(gs.spectral_radius),
                "max_rel_diff_vs_lu": float(np.max(np.abs(gs.radiosity - lu.radiosity)) / ref),
                "lu_energy_residual": float(lu.energy_residual),
                "pass": bool(gs.final_residual <= TOL_B),
            }
        )
    worst = max(r["final_residual"] for r in rows)
    return {
        "tolerance": TOL_B,
        "pass": bool(all(r["pass"] for r in rows) and worst <= TOL_B),
        "worst_residual": float(worst),
        "worst_rel_diff_vs_lu": float(max(r["max_rel_diff_vs_lu"] for r in rows)),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# criterion (c): isothermal cavity energy closure
# ---------------------------------------------------------------------------


def criterion_c() -> dict:
    rows = []
    for T in (300.0, 450.0, 700.0):
        for eps in (0.05, 0.3, 0.7, 1.0):
            enc = cube_enclosure(2.0, eps_opposite=eps, eps_adjacent=eps, T_opposite=T, T_adjacent=T)
            gs = solve_radiosity_gauss_seidel(enc, tol=TOL_B)
            emitted = sum(s.area * s.emissivity for s in enc.surfaces) * STEFAN_BOLTZMANN * T**4
            rows.append(
                {
                    "cavity": "cube",
                    "T_K": T,
                    "emissivity": eps,
                    "net_power_W": float(np.sum([s.area for s in enc.surfaces] * gs.net_flux)),
                    "emitted_power_W": float(emitted),
                    "rel_net_power": float(
                        abs(np.sum([s.area for s in enc.surfaces] * gs.net_flux)) / emitted
                    ),
                    "max_surface_flux_rel": float(
                        np.max(np.abs(gs.net_flux)) / (STEFAN_BOLTZMANN * T**4)
                    ),
                }
            )
    ens = concentric_spheres_enclosure(1.0, 4.0, 0.8, 0.4, 500.0, 500.0)
    gs = solve_radiosity_gauss_seidel(ens, tol=TOL_B)
    emitted = sum(s.area * s.emissivity for s in ens.surfaces) * STEFAN_BOLTZMANN * 500.0**4
    rows.append(
        {
            "cavity": "concentric_spheres",
            "T_K": 500.0,
            "emissivity": 0.8,
            "net_power_W": float(np.sum([s.area for s in ens.surfaces] * gs.net_flux)),
            "emitted_power_W": float(emitted),
            "rel_net_power": float(abs(np.sum([s.area for s in ens.surfaces] * gs.net_flux)) / emitted),
            "max_surface_flux_rel": float(np.max(np.abs(gs.net_flux)) / (STEFAN_BOLTZMANN * 500.0**4)),
        }
    )
    worst = max(r["rel_net_power"] for r in rows)
    worst_surface = max(r["max_surface_flux_rel"] for r in rows)
    return {
        "tolerance": TOL_C,
        "pass": bool(worst <= TOL_C and worst_surface <= TOL_C),
        "worst_rel_net_power": float(worst),
        "worst_surface_flux_rel": float(worst_surface),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# criterion (d): deck temperature field vs analytic
# ---------------------------------------------------------------------------


def criterion_d() -> dict:
    panel, env, mat = deck_case()
    comp = solar_flux_components(panel, env.sun)
    q_full, q_shade = comp["total"], comp["diffuse"]

    # ---- (d1) through-thickness, full nonlinear coupling -------------------
    ana = analytic_steady_field_1d(panel, env, mat, T_interior=300.0, h_interior=100.0,
                                   q_solar_abs=q_full)
    d1_rows = []
    d1_fields = {}
    for ny in (6, 8):
        sol = solve_deck_field(
            panel, env, mat, 300.0, 100.0, solar_profile=q_full, nx=1, ny=ny,
            tau=1.0, tol=1e-7, check_every=200, max_steps=400_000,
        )
        T_ana = np.interp(sol.z, np.array(ana["z"]), np.array(ana["T_profile"]))
        err = float(np.max(np.abs(sol.T[:, 0] - T_ana)) / np.max(T_ana))
        d1_rows.append(
            {
                "ny": ny,
                "steps": int(sol.steps),
                "converged": bool(sol.converged),
                "T_top_K": float(sol.T[-1, 0]),
                "T_bottom_K": float(sol.T[0, 0]),
                "T_top_analytic_K": float(ana["T_surface"]),
                "q_top_W_m2": float(sol.q_top[0]),
                "q_analytic_W_m2": float(ana["q"]),
                "q_rel_error": float(abs(sol.q_top[0] - ana["q"]) / abs(ana["q"])),
                "max_rel_error_field": err,
                "energy_residual": float(sol.energy_residual),
                "pass": bool(err <= TOL_D),
            }
        )
        d1_fields[ny] = sol

    # ---- (d2) 2-D shadow edge, frozen linearisation ------------------------
    width, nx, ny2 = 0.2, 32, 4
    x = (np.arange(nx) + 0.5) * width / nx
    profile = np.where(x < width / 2.0, q_shade, q_full)
    Tm = float(ana["T_surface"])
    fourier = analytic_steady_2d_fourier(
        panel, env, mat, 300.0, 100.0, [0.0, width / 2.0, width], [q_shade, q_full], width, Tm,
        n_modes=600,
    )
    sol2 = solve_deck_field(
        panel, env, mat, 300.0, 100.0, solar_profile=profile, nx=nx, ny=ny2,
        tau=1.0, linearise_at=Tm, tol=1e-7, check_every=200, max_steps=400_000,
    )
    T_ana2 = fourier["evaluate"](sol2.x, sol2.z)
    err2 = float(np.max(np.abs(sol2.T - T_ana2)) / np.max(T_ana2))
    d2 = {
        "width_m": width,
        "nx": nx,
        "ny": ny2,
        "steps": int(sol2.steps),
        "converged": bool(sol2.converged),
        "T_min_K": float(sol2.T.min()),
        "T_max_K": float(sol2.T.max()),
        "max_rel_error_field": err2,
        "energy_residual": float(sol2.energy_residual),
        "q_top_min_W_m2": float(sol2.q_top.min()),
        "q_top_max_W_m2": float(sol2.q_top.max()),
        "T_linearisation_K": Tm,
        "pass": bool(err2 <= TOL_D),
    }

    worst = max([r["max_rel_error_field"] for r in d1_rows] + [err2])
    return {
        "tolerance": TOL_D,
        "pass": bool(worst <= TOL_D and all(r["pass"] for r in d1_rows) and d2["pass"]),
        "worst_rel_error": float(worst),
        "solar_components_W_m2": {k: float(v) for k, v in comp.items() if k not in ("F_sky", "F_sea")},
        "d1_through_thickness": d1_rows,
        "d2_shadow_edge": d2,
        "analytic_1d": {
            "T_surface_K": float(ana["T_surface"]),
            "q_W_m2": float(ana["q"]),
            "residual_W_m2": float(ana["residual"]),
            "resistance_K_m2_W": float(ana["resistance"]),
        },
        "_fields": {"d1": d1_fields, "d2": (sol2, fourier)},
    }


# ---------------------------------------------------------------------------
# solar boundary table (three components)
# ---------------------------------------------------------------------------


def solar_table() -> dict:
    rows = []
    for tilt in (0.0, 30.0, 60.0, 90.0):
        panel = DeckPanel(
            tilt_deg=tilt, azimuth_deg=180.0, emissivity=0.9, solar_absorptance=0.6, h_conv=12.0
        )
        sun = SunState(beam_irradiance=800.0, elevation_deg=60.0, azimuth_deg=180.0,
                       diffuse_horizontal=120.0)
        comp = solar_flux_components(panel, sun)
        rows.append(
            {
                "tilt_deg": tilt,
                "F_sky": comp["F_sky"],
                "F_sea": comp["F_sea"],
                "direct_W_m2": comp["direct"],
                "diffuse_W_m2": comp["diffuse"],
                "ground_reflected_W_m2": comp["ground_reflected"],
                "total_W_m2": comp["total"],
                "aggregate_check_rel": float(
                    abs(comp["total"] - absorbed_solar_flux(panel, sun))
                    / max(abs(comp["total"]), 1e-12)
                ),
            }
        )
    return {"rows": rows, "worst_aggregate_rel": float(max(r["aggregate_check_rel"] for r in rows))}


# ---------------------------------------------------------------------------
# numerical studies: relaxation-time sensitivity and observed grid order
# ---------------------------------------------------------------------------


def numerical_studies(d: dict) -> dict:
    """Quantify two numerical properties of the D2Q5 ADE field solver.

    1. **tau sensitivity.**  The time step of the ADE follows the relaxation
       time, ``dt = cs^2 (tau - 1/2) dx^2 / alpha``.  Large ``tau`` means big
       steps and a fast march, but the *steady* discrete solution is not
       exactly tau-invariant (the naive ``w_i S`` source term leaves an
       O(tau) consistency error), so the gate is only met for moderate ``tau``.
       This is reported, not hidden: it is why the solver runs at ``tau = 1``.
    2. **observed grid order** of the boundary-flux treatment, from two
       resolutions.
    """
    panel, env, mat = deck_case()
    q_full = solar_flux_components(panel, env.sun)["total"]
    ana = analytic_steady_field_1d(panel, env, mat, 300.0, 100.0, q_solar_abs=q_full)
    rows = []
    for tau in (1.0, 2.0, 5.0, 20.0):
        sol = solve_deck_field(
            panel, env, mat, 300.0, 100.0, solar_profile=q_full, nx=1, ny=6,
            tau=tau, tol=1e-7, check_every=200, max_steps=400_000,
        )
        T_ana = np.interp(sol.z, np.array(ana["z"]), np.array(ana["T_profile"]))
        rows.append(
            {
                "tau": tau,
                "dt_s": float(sol.dt),
                "steps": int(sol.steps),
                "converged": bool(sol.converged),
                "T_top_K": float(sol.T[-1, 0]),
                "max_rel_error_field": float(
                    np.max(np.abs(sol.T[:, 0] - T_ana)) / np.max(T_ana)
                ),
            }
        )
    d1 = d["d1_through_thickness"]
    order = None
    if len(d1) >= 2:
        (ny1, e1), (ny2, e2) = (d1[0]["ny"], d1[0]["max_rel_error_field"]), (
            d1[1]["ny"],
            d1[1]["max_rel_error_field"],
        )
        order = float(math.log(e1 / e2) / math.log(ny2 / ny1))
    return {"tau_sensitivity": rows, "observed_grid_order": order}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(a: dict, b: dict, c: dict, d: dict, sol_table: dict, outdir: Path) -> list[dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    figs: list[dict] = []

    # 1) closed-form family vs independent references
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0))
    fams = ["parallel_rect", "perp_rect", "coax_disks", "sphere_plate", "cylinder_plate"]
    for ax, fam in zip(axes.ravel(), fams + [None]):
        if fam is None:
            ax.axis("off")
            continue
        rows = [r for r in a["rows"] if r["family"] == fam]
        idx = np.arange(len(rows))
        ax.plot(idx, [r["F_closed_form"] for r in rows], "o", ms=8, label="closed form")
        ax.plot(idx, [r["F_reference"] for r in rows], "x", ms=10, label="independent reference")
        ax.set_title(f"{fam}\nmax rel. err = {max(r['rel_error'] for r in rows):.2e}")
        ax.set_xticks(idx)
        def _lab(params: dict) -> str:
            items = [
                (k, f"{v:g}" if isinstance(v, (int, float)) else str(v).replace("->", "->\n"))
                for k, v in params.items()
            ]
            return "\n".join(f"{k}={v}" for k, v in items if k != "mode")

        ax.set_xticklabels([_lab(r["params"]) for r in rows], rotation=0, fontsize=6)
        ax.set_ylabel("view factor F")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("(a) closed-form view factors vs independent estimators (gate 1 %)", fontsize=12)
    fig.tight_layout()
    path = outdir / "p93_fig_a_view_factors.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figs.append({"name": "view_factor_family", "path": str(path),
                 "caption": "Five-configuration closed-form view-factor family (parallel rectangles, "
                            "perpendicular rectangles, coaxial disks, sphere-plate, cylinder-plate) "
                            "against independent quadrature / Monte-Carlo estimators."})

    # 2) Gauss-Seidel convergence histories
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for name, enc in enclosure_set().items():
        gs = solve_radiosity_gauss_seidel(enc, tol=1e-13, max_iter=5000, record_history=True)
        hist = np.array(gs.residual_history)
        ax.semilogy(np.arange(1, hist.size + 1), hist, label=f"{name} (rho={spectral_radius(enc):.3f})")
    ax.axhline(TOL_B, color="k", ls="--", lw=1, label="gate 1e-6")
    ax.set_xlabel("Gauss-Seidel sweep")
    ax.set_ylabel("iteration residual  max|dJ|/max|J|")
    ax.set_title("(b) radiosity fixed-point convergence")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "p93_fig_b_radiosity_convergence.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figs.append({"name": "radiosity_convergence", "path": str(path),
                 "caption": "Gauss-Seidel radiosity residual history for five enclosures; the "
                            "annotated spectral radius rho(diag(1-eps)F) sets the asymptotic rate."})

    # 3) isothermal cavity closure
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    rows = [r for r in c["rows"] if r["cavity"] == "cube"]
    for T in sorted({r["T_K"] for r in rows}):
        sub = [r for r in rows if r["T_K"] == T]
        ax.semilogx([r["emissivity"] for r in sub], [r["rel_net_power"] for r in sub], "o-",
                    label=f"T = {T:g} K")
    ax.axhline(TOL_C, color="k", ls="--", lw=1, label="gate 0.1 %")
    ax.set_xlabel("cavity emissivity")
    ax.set_ylabel("|sum A q| / emitted power")
    ax.set_title("(c) isothermal cavity power closure")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "p93_fig_c_isothermal_cavity.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figs.append({"name": "isothermal_cavity", "path": str(path),
                 "caption": "Isothermal closed cavity: net radiative power normalised by the emitted "
                            "power, over three temperatures and four emissivities (gate 0.1 %)."})

    # 4) through-thickness deck profile, LBM vs analytic
    fields = d["_fields"]["d1"]
    ana_prof = None
    panel, env, mat = deck_case()
    ana = analytic_steady_field_1d(panel, env, mat, 300.0, 100.0,
                                   q_solar_abs=solar_flux_components(panel, env.sun)["total"],
                                   n_points=41)
    ana_prof = (np.array(ana["z"]), np.array(ana["T_profile"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax1.plot(ana_prof[1], ana_prof[0] * 1e3, "k-", lw=2, label="analytic (bisection + linear)")
    for ny, sol in sorted(fields.items()):
        ax1.plot(sol.T[:, 0], sol.z * 1e3, "o--", ms=5, label=f"LBM ny={ny}")
    ax1.set_xlabel("temperature [K]")
    ax1.set_ylabel("height through deck [mm]")
    ax1.set_title("(d1) through-thickness steady profile")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    # surface-energy balance curve: root of q_net(T_s)
    q_s = solar_flux_components(panel, env.sun)["total"]
    Temps = np.linspace(295.0, 320.0, 200)
    q_bal = np.array([float(surface_net_flux(T, panel, env, q_s)[0]) for T in Temps])
    ax2.plot(Temps, q_bal, label="surface net load q_net(T_s)")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.plot([ana["T_surface"]], [0.0], "k*", ms=14, label=f"equilibrium {ana['T_surface']:.3f} K")
    ax2.set_xlabel("deck surface temperature [K]")
    ax2.set_ylabel("flux into deck [W/m2]")
    ax2.set_title("(d1) nonlinear surface balance")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "p93_fig_d_deck_1d_field.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figs.append({"name": "deck_1d_field", "path": str(path),
                 "caption": "Through-thickness deck field: LBM (two resolutions) against the analytic "
                            "bisection+linear solution, and the nonlinear surface balance whose root "
                            "defines the surface temperature."})

    # 5) 2-D shadow-edge field
    sol2, fourier = d["_fields"]["d2"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    im = ax1.pcolormesh(sol2.x * 1e3, sol2.z * 1e3, sol2.T, shading="auto", cmap="inferno")
    fig.colorbar(im, ax=ax1, label="T [K]")
    ax1.set_xlabel("x along deck [mm]")
    ax1.set_ylabel("height [mm]")
    ax1.set_title("(d2) LBM deck field, shadow edge at 100 mm")
    ax2.plot(sol2.x * 1e3, sol2.T[-1, :], "o", ms=4, label="LBM top row")
    T_ana2 = fourier["evaluate"](sol2.x, sol2.z)
    ax2.plot(sol2.x * 1e3, T_ana2[-1, :], "k-", lw=1.5, label="analytic cosine series")
    ax2.plot(sol2.x * 1e3, sol2.T[0, :], "s", ms=3, label="LBM bottom row")
    ax2.plot(sol2.x * 1e3, T_ana2[0, :], "k--", lw=1.0, label="analytic bottom")
    ax2.set_xlabel("x along deck [mm]")
    ax2.set_ylabel("temperature [K]")
    ax2.set_title(f"(d2) shadow-edge temperature, max rel. err = {d['d2_shadow_edge']['max_rel_error_field']:.2e}")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "p93_fig_d2_shadow_edge.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figs.append({"name": "deck_2d_shadow", "path": str(path),
                 "caption": "Two-dimensional deck field across a shadow edge: LBM pcolormesh and "
                            "top/bottom-row comparison with the exact cosine-series solution."})

    # 6) solar components vs tilt
    rows = sol_table["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    tilts = [r["tilt_deg"] for r in rows]
    ax.bar(tilts, [r["direct_W_m2"] for r in rows], width=8, label="direct beam")
    ax.bar(tilts, [r["diffuse_W_m2"] for r in rows], width=8,
           bottom=[r["direct_W_m2"] for r in rows], label="sky diffuse")
    ax.bar(tilts, [r["ground_reflected_W_m2"] for r in rows], width=8,
           bottom=[r["direct_W_m2"] + r["diffuse_W_m2"] for r in rows], label="sea/ground reflected")
    ax.set_xlabel("panel tilt from horizontal [deg]")
    ax.set_ylabel("absorbed solar flux [W/m2]")
    ax.set_title("solar boundary: three explicit components")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "p93_fig_solar_components.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figs.append({"name": "solar_components", "path": str(path),
                 "caption": "Absorbed solar flux split into direct beam, isotropic sky diffuse and "
                            "ground(sea)-reflected components versus panel tilt."})
    return figs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> dict:
    here = Path(__file__).resolve().parent
    print("[1/5] criterion (a): five-configuration closed-form view factors ...")
    a = criterion_a()
    print("      pass:", a["pass"], "worst:", f"{a['worst_rel_error']:.3e}")

    print("[2/5] criterion (b): Gauss-Seidel radiosity residual ...")
    b = criterion_b()
    print("      pass:", b["pass"], "worst:", f"{b['worst_residual']:.3e}")

    print("[3/5] criterion (c): isothermal cavity power closure ...")
    c = criterion_c()
    print("      pass:", c["pass"], "worst:", f"{c['worst_rel_net_power']:.3e}")

    print("[4/5] criterion (d): deck temperature field vs analytic ...")
    d = criterion_d()
    print("      pass:", d["pass"], "worst:", f"{d['worst_rel_error']:.3e}")

    print("[5/5] solar component table + numerical studies + figures ...")
    sol_table = solar_table()
    studies = numerical_studies(d)
    figs = make_figures(a, b, c, d, sol_table, ROOT / "figures" / "p93w3_n02")

    detail_d = {k: v for k, v in d.items() if not k.startswith("_")}
    result = {
        "mission": "P93W3-N02",
        "node": "TensorLBM thermal radiation: view-factor family + Gauss-Seidel radiosity + deck field",
        "criteria": {
            "a_view_factor_closed_form": {
                "requirement": "5 configurations (parallel rect, perpendicular rect, coaxial disks, "
                               "sphere-plate, cylinder-plate) <= 1 % vs closed form / independent reference",
                "tolerance": TOL_A,
                "pass": bool(a["pass"]),
                "worst": a["worst_rel_error"],
            },
            "b_radiosity_iteration_residual": {
                "requirement": "gray enclosure radiosity Gauss-Seidel iteration residual <= 1e-6",
                "tolerance": TOL_B,
                "pass": bool(b["pass"]),
                "worst": b["worst_residual"],
            },
            "c_isothermal_cavity_energy": {
                "requirement": "isothermal closed cavity net radiative flux <= 0.1 %",
                "tolerance": TOL_C,
                "pass": bool(c["pass"]),
                "worst": c["worst_rel_net_power"],
            },
            "d_deck_field_vs_analytic": {
                "requirement": "deck temperature field vs analytic steady solution <= 3 %",
                "tolerance": TOL_D,
                "pass": bool(d["pass"]),
                "worst": d["worst_rel_error"],
            },
        },
        "all_pass": bool(a["pass"] and b["pass"] and c["pass"] and d["pass"]),
        "detail": {
            "a": a,
            "b": b,
            "c": c,
            "d": detail_d,
            "solar_boundary": sol_table,
            "numerical_studies": studies,
        },
        "figures": [
            {"name": f["name"], "caption": f["caption"],
             "path": str(Path(f["path"]).relative_to(ROOT))}
            for f in figs
        ],
    }
    (here / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    art = ROOT / "artifacts" / "P93W3_N02_radiation_deck_env.json"
    art.parent.mkdir(exist_ok=True)
    art.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("      wrote", here / "result.json", "and", art)
    print("ALL PASS:", result["all_pass"])
    return result


if __name__ == "__main__":
    main()
