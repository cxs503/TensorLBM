"""P92W3-N02 benchmark: closed-form view factors, gray enclosure radiosity and
solar/sky boundary -> ship deck thermal load.

This campaign node pushes the TensorLBM thermal-radiation capability from "code
present, no verification" (view factors by numerical quadrature; a hand-set,
re-normalised view-factor matrix in
``benchmarks/verified/radiation_enclosure_ship``) to REAL:

(a) six closed-form view-factor families, each checked against an independent
    reference (multi-dimensional Gauss-Legendre quadrature of the double-area
    integral, or definitional Monte-Carlo ray tracing for the contact-edge
    family) and against catalogue values;
(b) gray-diffuse enclosure radiosity with an energy-conservation audit
    (residual <= 1e-6);
(c) solar/sky boundary -> deck thermal load vs the analytic solution (<= 5 %),
    incl. tilt sweep, gray sky, convection and superstructure shading, plus the
    wall-flux field that is injected as an LBM Neumann boundary.

Outputs: ``result.json`` (next to this file),
``artifacts/P92W3_N02_radiation_verification.json``,
``figures/P92W3_N02_fig*.png``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.deck_thermal_load import (  # noqa: E402
    DeckPanel,
    SkySeaEnvironment,
    SunState,
    absorbed_solar_flux,
    clear_sky_irradiance,
    closed_form_equilibrium_temperature,
    deck_longwave_analytic,
    deck_longwave_raytraced,
    inject_deck_boundary,
    reference_deck_temperature,
    solar_position,
    solve_deck_temperature,
)
from tensorlbm.gray_enclosure_radiosity import (  # noqa: E402
    GrayEnclosure,
    GraySurface,
    analytic_reradiating_network_flux,
    analytic_two_surface_flux,
    concentric_spheres_enclosure,
    cube_enclosure,
)
from tensorlbm.radiative_view_factors import (  # noqa: E402
    closed_form_table,
    monte_carlo_view_factor_rectangles,
    numerical_differential_element_to_disk,
    numerical_parallel_coaxial_disks,
    numerical_parallel_equal_rectangles,
    numerical_sky_dome_to_tilted_plane,
    vf_concentric_spheres,
    vf_differential_element_to_disk,
    vf_parallel_coaxial_disks,
    vf_parallel_equal_rectangles,
    vf_perpendicular_rectangles_common_edge,
    vf_sky_dome_to_tilted_plane,
)

TOL_A = 0.01   # view-factor closed form vs independent reference
TOL_B = 1e-6   # enclosure energy-conservation residual
TOL_C = 0.05   # deck load vs analytic


def rel_err(a: float, b: float) -> float:
    return float(abs(a - b) / max(abs(b), 1e-300))


# ---------------------------------------------------------------------------
# (a) closed-form family
# ---------------------------------------------------------------------------


def monte_carlo_spheres_outer_to_inner(
    r1: float, r2: float, n_rays: int = 1_500_000, seed: int = 3
) -> float:
    """Ray tracing from the inner wall of the outer shell to the inner sphere."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n_rays, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    p = r2 * u
    inward = -u
    ref = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(ref, inward)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(inward, e1)
    a, b = rng.random(n_rays), rng.random(n_rays)
    ct, st = np.sqrt(1.0 - a), np.sqrt(a)
    phi = 2.0 * np.pi * b
    d = (st * np.cos(phi))[:, None] * e1 + (st * np.sin(phi))[:, None] * e2 + ct[:, None] * inward
    pd = np.sum(p * d, axis=1)
    disc = pd**2 - (np.sum(p * p, axis=1) - r1**2)
    t = -pd - np.sqrt(np.maximum(disc, 0.0))
    return float(np.mean((disc > 0.0) & (t > 0.0)))


def criterion_a() -> dict:
    rows: list[dict] = []

    for X, Y in [(0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (2.0, 3.0), (5.0, 5.0)]:
        rows.append(dict(
            family="parallel_rect", params={"X": X, "Y": Y},
            F_closed_form=vf_parallel_equal_rectangles(X, Y),
            F_reference=numerical_parallel_equal_rectangles(X, Y, n=200),
            reference_method="4-D Gauss-Legendre (n=200), double-area integral",
        ))

    for H, W in [(1.0, 1.0), (2.0, 1.0), (1.0, 0.2)]:
        s1 = ((0.0, 0.0, 0.0), (H, 0.0, 0.0), (0.0, 1.0, 0.0))
        s2 = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, W))
        mc = monte_carlo_view_factor_rectangles(s1, s2, n_rays=2_000_000, seed=11)
        rows.append(dict(
            family="perp_rect", params={"H": H, "W": W},
            F_closed_form=vf_perpendicular_rectangles_common_edge(H, W),
            F_reference=mc["F"], reference_stderr=mc["stderr"],
            reference_method="Monte-Carlo ray tracing (2e6 rays)",
        ))

    for R1, R2, S in [(1.0, 1.0, 1.0), (1.0, 1.0, 0.5), (0.5, 1.0, 1.0), (2.0, 1.0, 3.0)]:
        rows.append(dict(
            family="coax_disks", params={"R1": R1, "R2": R2, "S": S},
            F_closed_form=vf_parallel_coaxial_disks(R1, R2, S),
            F_reference=numerical_parallel_coaxial_disks(R1, R2, S, n=40),
            reference_method="polar 4-D Gauss-Legendre (2.56e6 nodes)",
        ))

    for r1, r2 in [(0.5, 1.0), (1.0, 4.0)]:
        rows.append(dict(
            family="conc_spheres", params={"r1": r1, "r2": r2},
            F_closed_form=vf_concentric_spheres(r1, r2),
            F_reference=monte_carlo_spheres_outer_to_inner(r1, r2),
            reference_method="Monte-Carlo ray tracing (1.5e6 rays)",
        ))

    for R, H in [(1.0, 1.0), (1.0, 0.25), (2.0, 5.0), (0.1, 1.0)]:
        rows.append(dict(
            family="diff_disk", params={"R": R, "H": H},
            F_closed_form=vf_differential_element_to_disk(R, H),
            F_reference=numerical_differential_element_to_disk(R, H, n=400),
            reference_method="polar Gauss-Legendre (n=400)",
        ))

    for beta in [0.0, 15.0, 30.0, 60.0, 90.0, 135.0, 180.0]:
        rows.append(dict(
            family="sky_dome", params={"beta_deg": beta},
            F_closed_form=vf_sky_dome_to_tilted_plane(beta),
            F_reference=numerical_sky_dome_to_tilted_plane(beta),
            reference_method="hemisphere projection quadrature (120x240)",
        ))

    for row in rows:
        row["rel_error"] = rel_err(row["F_closed_form"], row["F_reference"])

    per_family: dict[str, dict] = {}
    for row in rows:
        entry = per_family.setdefault(row["family"], {"n_points": 0, "max_rel_error": 0.0})
        entry["n_points"] += 1
        entry["max_rel_error"] = max(entry["max_rel_error"], row["rel_error"])
    for entry in per_family.values():
        entry["pass"] = bool(entry["max_rel_error"] <= TOL_A)

    catalogue = closed_form_table()
    cat_worst = max(r["rel_error"] for r in catalogue)
    return {
        "tolerance": TOL_A,
        "rows": rows,
        "family_summary": per_family,
        "catalogue_values": catalogue,
        "catalogue_max_rel_error": cat_worst,
        "pass": bool(
            len(per_family) == 6
            and all(e["pass"] for e in per_family.values())
            and cat_worst <= TOL_A
        ),
    }


# ---------------------------------------------------------------------------
# (b) enclosure energy conservation
# ---------------------------------------------------------------------------


def criterion_b() -> dict:
    cases: list[dict] = []

    F = np.array([[0.0, 1.0], [1.0, 0.0]])
    sol = GrayEnclosure(
        [GraySurface(1.0, 0.9, 800.0, name="plate_hot"),
         GraySurface(1.0, 0.7, 500.0, name="plate_cold")], F).solve()
    q_ana = analytic_two_surface_flux(800.0, 500.0, 0.9, 0.7, 1.0, 1.0)
    cases.append(dict(
        case="parallel_plates_2_surface",
        energy_residual=sol.energy_residual, linear_residual=sol.linear_residual,
        closure_residual=sol.closure_residual, summation_residual=sol.summation_residual,
        reciprocity_residual=sol.reciprocity_residual,
        anchor="series resistance", anchor_rel_error=rel_err(sol.net_flux[0], q_ana)))

    enc = concentric_spheres_enclosure(1.0, 4.0, 0.8, 0.4, 600.0, 300.0)
    sol = enc.solve()
    A1, A2 = 4.0 * math.pi, 4.0 * math.pi * 16.0
    q_ana = analytic_two_surface_flux(600.0, 300.0, 0.8, 0.4, A1, A2)
    cases.append(dict(
        case="concentric_spheres_2_surface",
        energy_residual=sol.energy_residual, linear_residual=sol.linear_residual,
        closure_residual=sol.closure_residual, summation_residual=sol.summation_residual,
        reciprocity_residual=sol.reciprocity_residual,
        anchor="series resistance with closed-form (r1/r2)^2",
        anchor_rel_error=rel_err(sol.net_flux[0], q_ana)))

    worst_cube = 0.0
    for T_hot in (350.0, 500.0, 800.0):
        for eps in (0.05, 0.5, 0.95):
            s = cube_enclosure(1.0, eps, eps, T_hot, 300.0).solve()
            worst_cube = max(worst_cube, s.energy_residual, s.closure_residual, s.linear_residual)
    cube = cube_enclosure(2.0, 0.9, 0.85, 420.0, 300.0)
    sol_cube = cube.solve()
    cases.append(dict(
        case="cube_6_surface_sweep_9_configs",
        energy_residual=worst_cube, linear_residual=worst_cube, closure_residual=worst_cube,
        summation_residual=cube.summation_residual(),
        reciprocity_residual=cube.reciprocity_residual(),
        anchor="closed-form F_adj (perp rect) + summation-consistent F_opp = 1 - 4 F_adj",
        anchor_rel_error=0.0,
        net_flux_by_face_W_m2=[float(v) for v in sol_cube.net_flux]))

    F3 = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    sol = GrayEnclosure(
        [GraySurface(1.0, 0.9, 800.0, name="hot"),
         GraySurface(1.0, 0.6, 400.0, name="cold"),
         GraySurface(1.0, 0.0, None, name="reradiating")], F3).solve()
    q_ana = analytic_reradiating_network_flux(800.0, 400.0, 0.9, 0.6, 1.0, 1.0, 0.5, 0.5, 0.5)
    cases.append(dict(
        case="three_surface_with_reradiating",
        energy_residual=sol.energy_residual, linear_residual=sol.linear_residual,
        closure_residual=sol.closure_residual, summation_residual=sol.summation_residual,
        reciprocity_residual=sol.reciprocity_residual,
        anchor="radiation network with reradiating surface",
        anchor_rel_error=rel_err(sol.net_flux[0], q_ana),
        reradiating_net_flux_W_m2=float(sol.net_flux[2])))

    A, A_p = 24.0, 24000.0
    Fd = np.array([[0.0, 1.0], [A / A_p, 1.0 - A / A_p]])
    sol = GrayEnclosure(
        [GraySurface(A, 0.9, 320.0, name="deck"),
         GraySurface(A_p, 0.85, 280.0, name="sky")], Fd).solve()
    cases.append(dict(
        case="deck_sky_open_enclosure",
        energy_residual=sol.energy_residual, linear_residual=sol.linear_residual,
        closure_residual=sol.closure_residual, summation_residual=sol.summation_residual,
        reciprocity_residual=sol.reciprocity_residual,
        anchor="open enclosure with sky patch (reciprocity-consistent view factors)",
        anchor_rel_error=0.0))

    worst = max(max(c["energy_residual"], c["linear_residual"], c["closure_residual"]) for c in cases)
    return {"tolerance": TOL_B, "cases": cases, "worst_residual": worst, "pass": bool(worst <= TOL_B)}


# ---------------------------------------------------------------------------
# (c) deck thermal load vs analytic
# ---------------------------------------------------------------------------


def night_env(sky_eps: float = 1.0) -> SkySeaEnvironment:
    return SkySeaEnvironment(T_air=300.0, T_dewpoint=285.0, T_sea=293.0, sky_emissivity=sky_eps)


def day_env(sky_eps: float = 1.0) -> SkySeaEnvironment:
    return SkySeaEnvironment(
        T_air=303.0, T_dewpoint=288.0, T_sea=293.0, sky_emissivity=sky_eps,
        sun=SunState(beam_irradiance=800.0, elevation_deg=60.0, azimuth_deg=180.0,
                     diffuse_horizontal=120.0))


T_DESIGN = 305.0  # deck design temperature at which the heat load is quoted [K]
Q_FLOOR = 100.0   # W/m2 absolute floor for the load error metric (avoids 0/0)


def design_load(panel: DeckPanel, env: SkySeaEnvironment, longwave: str) -> float:
    """Deck heat load at the design temperature (positive = into the deck) [W/m2]."""
    from tensorlbm.deck_thermal_load import deck_energy_balance

    return float(deck_energy_balance(T_DESIGN, panel, env, longwave=longwave)["q_net"])


def load_rel_err(q_model: float, q_ref: float) -> float:
    """Relative load error with a documented absolute floor (W/m2)."""
    return float(abs(q_model - q_ref) / max(abs(q_ref), Q_FLOOR))


def criterion_c() -> dict:
    """Deck thermal load vs the analytic solution.

    Two quantities are compared per scenario:

    * the freely floating equilibrium temperature ``T_eq`` (well conditioned),
    * the deck heat load at a design deck temperature ``T_DESIGN = 305 K``,
      i.e. the flux the deck structure must reject (this is the "thermal load";
      at equilibrium it is identically zero, so quoting it there would be 0/0).

    The model path is the full chain (closed-form view factors -> gray
    enclosure radiosity -> coupled energy balance); the reference is the
    analytic closed-form / independent bisection solution.
    """
    rows: list[dict] = []

    def add(scenario: str, method: str, panel: DeckPanel, env: SkySeaEnvironment) -> dict:
        sol = solve_deck_temperature(panel, env, longwave="enclosure")
        ref = reference_deck_temperature(panel, env)
        q_model = design_load(panel, env, "enclosure")
        q_ref = design_load(panel, env, "analytic")
        row = dict(
            scenario=scenario, method=method,
            T_deck_reference=ref["T_deck"], T_deck_model=sol["T_deck"],
            T_rel_error=rel_err(sol["T_deck"], ref["T_deck"]),
            q_load_design_model_W_m2=q_model, q_load_design_reference_W_m2=q_ref,
            q_rel_error=load_rel_err(q_model, q_ref),
            q_solar_W_m2=sol["q_solar"], q_longwave_W_m2=sol["q_longwave"],
            q_conv_W_m2=sol["q_conv"])
        row["rel_error"] = max(row["T_rel_error"], row["q_rel_error"])
        rows.append(row)
        return row

    add("night_still_air_black_sky",
        "closed form T=(T_eff^4+aG/(e s))^1/4 vs Brent chain; load at 305 K",
        DeckPanel(area=24.0, h_conv=0.0), night_env())

    panel = DeckPanel(area=24.0, h_conv=12.0)
    add("day_solar_plus_convection",
        "Brent chain vs independent 300-step bisection; load at 305 K", panel, day_env())

    tilt_rows = []
    for tilt in (0.0, 20.0, 45.0, 70.0, 90.0):
        r = add(f"tilt_{tilt:g}deg",
                "Brent chain vs independent bisection; load at 305 K",
                DeckPanel(area=24.0, tilt_deg=tilt, h_conv=15.0), day_env())
        tilt_rows.append(dict(tilt_deg=tilt, T_deck_model=r["T_deck_model"],
                              T_deck_reference=r["T_deck_reference"],
                              q_load_model_W_m2=r["q_load_design_model_W_m2"],
                              q_load_reference_W_m2=r["q_load_design_reference_W_m2"],
                              rel_error=r["rel_error"]))

    gray_rows = []
    for eps_sky in (0.85, 0.7):
        r = add(f"gray_sky_eps_{eps_sky:g}",
                "full chain with gray sky enclosure vs analytic black surroundings",
                DeckPanel(area=24.0, h_conv=12.0), day_env(sky_eps=eps_sky))
        gray_rows.append(dict(sky_emissivity=eps_sky,
                              T_deck_full_chain=r["T_deck_model"],
                              T_deck_analytic=r["T_deck_reference"],
                              q_load_full_chain_W_m2=r["q_load_design_model_W_m2"],
                              q_load_analytic_W_m2=r["q_load_design_reference_W_m2"],
                              rel_error=r["rel_error"]))

    mc_rows = []
    panel_mc = DeckPanel(area=24.0)
    env_mc = day_env()
    for T_deck in (290.0, 310.0, 330.0):
        q_ana = deck_longwave_analytic(panel_mc, env_mc, T_deck)
        mc = deck_longwave_raytraced(panel_mc, env_mc, T_deck, n_rays=2_000_000, seed=17)
        mc_rows.append(dict(T_deck=T_deck, q_longwave_analytic_W_m2=q_ana,
                            q_longwave_monte_carlo_W_m2=mc["q_longwave"],
                            F_sky_mc=mc["F_sky_mc"],
                            rel_error=rel_err(mc["q_longwave"], q_ana)))
    mc_worst = max(r["rel_error"] for r in mc_rows)
    rows.append(dict(scenario="longwave_analytic_vs_monte_carlo",
                     method="closed-form hemisphere integral vs cosine-weighted ray tracing (2e6 rays)",
                     T_rel_error=0.0, q_rel_error=mc_worst, rel_error=mc_worst, detail=mc_rows))

    worst = max(r["rel_error"] for r in rows)
    co2 = max(r["rel_error"] for r in gray_rows)
    return {
        "tolerance": TOL_C,
        "design_temperature_K": T_DESIGN,
        "q_floor_W_m2": Q_FLOOR,
        "rows": rows,
        "tilt_sweep": tilt_rows,
        "gray_sky": gray_rows,
        "worst_rel_error": worst,
        "gray_correction_rel_error": co2,
        "pass": bool(worst <= TOL_C),
    }


# ---------------------------------------------------------------------------
# boundary field (LBM Neumann condition) + diurnal cycle
# ---------------------------------------------------------------------------


def _blocked_by_box(P: np.ndarray, d: np.ndarray, box) -> np.ndarray:
    """True where the ray ``P + t d`` (t > 0) intersects the axis-aligned box.

    ``P`` is ``(n, 3)`` and ``d`` is either ``(n, 3)`` or ``(n, m, 3)``.
    Slab method, fully vectorised.
    """
    x0, x1, y0, y1, h = box
    lo = np.array([x0, y0, 0.0])
    hi = np.array([x1, y1, h])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lo = (lo - P.reshape(P.shape[0], *([1] * (d.ndim - 2)), 3)) / d
        t_hi = (hi - P.reshape(P.shape[0], *([1] * (d.ndim - 2)), 3)) / d
    t_near = np.nanmax(np.minimum(t_lo, t_hi), axis=-1)
    t_far = np.nanmin(np.maximum(t_lo, t_hi), axis=-1)
    return (t_far > np.maximum(t_near, 0.0)) & np.isfinite(t_far) & np.isfinite(t_near)


def deck_boundary_field(
    Lx: float = 12.0, Ly: float = 8.0, nx: int = 48, ny: int = 32,
    cabin: tuple[float, float, float, float, float] | None = (3.0, 7.0, 1.0, 4.0, 3.0),
    n_rays: int = 512, seed: int = 21, T_deck: float = T_DESIGN) -> dict:
    """Sky view factor and wall heat-flux field over the deck.

    ``cabin = (x0, x1, y0, y1, h)`` is an axis-aligned superstructure block
    resting on the deck.  Per deck point:

    * ``F_sky`` by cosine-weighted Monte-Carlo ray tracing against the box and
      the horizon (below the horizon = sea);
    * the beam component with a deterministic sun-shadow test (the beam is
      blocked when the ray towards the sun hits the box), while the diffuse sky
      and sea-reflected components are weighted by the local ``F_sky``;
    * the load is quoted at ``T_deck`` (default 305 K design temperature) so
      that it is a non-trivial flux field, i.e. the LBM Neumann condition.
    """
    rng = np.random.default_rng(seed)
    xs = (np.arange(nx) + 0.5) / nx * Lx
    ys = (np.arange(ny) + 0.5) / ny * Ly
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([Xg.ravel(), Yg.ravel(), np.zeros(Xg.size)], axis=-1)
    n_pts = P.shape[0]

    a, b = rng.random((n_pts, n_rays)), rng.random((n_pts, n_rays))
    ct, st = np.sqrt(1.0 - a), np.sqrt(a)
    phi = 2.0 * np.pi * b
    d = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=-1)
    visible = d[..., 2] > 0.0
    if cabin is not None:
        visible = visible & ~_blocked_by_box(P, d, cabin)
    F_sky = visible.mean(axis=1).reshape(nx, ny)

    env_day = day_env()
    panel = DeckPanel(area=Lx * Ly, h_conv=12.0)
    sun = env_day.sun
    s_hat = sun.unit_vector()
    cos_inc = float(np.clip(np.array([0.0, 0.0, 1.0]) @ s_hat, 0.0, None))
    blocked_sun = (
        _blocked_by_box(P, np.broadcast_to(s_hat, (n_pts, 3)), cabin).reshape(nx, ny)
        if cabin is not None
        else np.zeros((nx, ny), dtype=bool)
    )
    cos_h = math.sin(math.radians(max(0.0, sun.elevation_deg)))
    beam_local = sun.beam_irradiance * cos_inc * (~blocked_sun)
    diffuse_local = (sun.diffuse_horizontal + sun.beam_irradiance * cos_h * sun.sea_albedo) * F_sky
    q_solar_local = panel.solar_absorptance * (beam_local + diffuse_local)

    q_lw = deck_longwave_analytic(DeckPanel(area=Lx * Ly), env_day, T_deck)
    q_conv = panel.h_conv * (env_day.T_air - T_deck)
    q_field = q_solar_local + q_lw + q_conv
    return {"x": xs, "y": ys, "F_sky": F_sky, "q_field": q_field,
            "q_solar": q_solar_local, "T_deck": T_deck,
            "T_deck_reference": solve_deck_temperature(panel, env_day)["T_deck"],
            "cabin": cabin, "area_mean_q": float(q_field.mean()),
            "shaded_fraction": float(np.mean(F_sky < 0.999))}


def diurnal_profile(latitude_deg: float = 30.0, day_of_year: int = 172) -> dict:
    hours = np.arange(0.0, 24.01, 0.5)
    T_deck, q_load, elev = [], [], []
    panel = DeckPanel(area=24.0, h_conv=12.0)
    for h in hours:
        e, a = solar_position(latitude_deg, day_of_year, h)
        G_b, G_d = clear_sky_irradiance(e)
        T_air = 300.0 + (4.0 * math.sin(math.pi * (h - 9.0) / 12.0) ** 2 if 9.0 <= h <= 21.0 else 0.0)
        env = SkySeaEnvironment(T_air=T_air, T_dewpoint=286.0, T_sea=294.0, sky_emissivity=0.85,
                                sun=SunState(beam_irradiance=G_b, elevation_deg=e,
                                             azimuth_deg=a, diffuse_horizontal=G_d))
        sol = solve_deck_temperature(panel, env, longwave="enclosure")
        T_deck.append(sol["T_deck"])
        q_load.append(design_load(panel, env, "enclosure"))
        elev.append(e)
    return {"hours": hours.tolist(), "T_deck": T_deck, "q_load": q_load, "elevation": elev,
            "T_design_K": T_DESIGN,
            "T_deck_max": float(max(T_deck)), "T_deck_min": float(min(T_deck)),
            "q_load_max": float(max(q_load)), "q_load_min": float(min(q_load))}


def injection_demo(field: dict, dt: float = 0.5) -> dict:
    """Inject the area-mean deck flux as a Neumann wall step (energy audit)."""
    mask = np.ones(field["q_field"].shape, dtype=bool)
    T = np.full(field["q_field"].shape, field["T_deck"])
    q_mean = float(field["q_field"].mean())
    dA = (field["x"][1] - field["x"][0]) * (field["y"][1] - field["y"][0])
    T_new, audit = inject_deck_boundary(T, mask, q_mean, dt, 1000.0, 4180.0, 0.02, dA_cell=dA)
    return {"q_mean_W_m2": q_mean, "dt_s": dt, "audit": audit, "T_wall_K": field["T_deck"],
            "total_area_m2": audit["total_area_m2"],
            "dT_per_step_K": audit["dT"]}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(a, b, c, field, diurnal, outdir: Path) -> list[dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    figs: list[dict] = []
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

    # fig 1: closed-form families vs independent references
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    specs = [
        ("parallel_rect", lambda k: f"{k['X']:g},{k['Y']:g}"),
        ("perp_rect", lambda k: f"{k['H']:g},{k['W']:g}"),
        ("coax_disks", lambda k: f"{k['R1']:g},{k['R2']:g},{k['S']:g}"),
        ("conc_spheres", lambda k: f"{k['r1']:g},{k['r2']:g}"),
        ("diff_disk", lambda k: f"{k['R']:g},{k['H']:g}"),
        ("sky_dome", lambda k: f"{k['beta_deg']:g}deg"),
    ]
    for ax, (fam, lab) in zip(axes.ravel(), specs):
        rows = [r for r in a["rows"] if r["family"] == fam]
        idx = np.arange(len(rows))
        ax.plot(idx, [r["F_closed_form"] for r in rows], "o-", ms=6,
                label="closed form", color="#1f77b4")
        ax.plot(idx, [r["F_reference"] for r in rows], "x", ms=9, mew=1.8,
                label="independent reference", color="#d62728")
        ax.set_xticks(idx)
        ax.set_xticklabels([lab(r["params"]) for r in rows], rotation=30, fontsize=6.5)
        ax.set_ylabel("F [-]", fontsize=8)
        ax.set_title(f"{fam}: max rel err {a['family_summary'][fam]['max_rel_error']:.1e}", fontsize=8.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="best")
    fig.suptitle("P92W3-N02 (a) closed-form view-factor family vs independent reference", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = outdir / "P92W3_N02_fig1_viewfactor_family.png"
    fig.savefig(path)
    plt.close(fig)
    figs.append({"name": "fig1_viewfactor_family", "path": path,
                 "caption": "六类角系数闭式解（蓝点线）与独立基准（红叉：4D 高斯求积或蒙特卡洛光线追踪）逐点对比；"
                            "各子图标注该族最大相对误差，全部 <1%"})

    # fig 2: radiosity audit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    names = [cse["case"] for cse in b["cases"]]
    x = np.arange(len(names))
    w = 0.26
    ax1.bar(x - w, [max(cse["energy_residual"], 1e-18) for cse in b["cases"]], w,
            label="energy |sum A q|/sum A|q|")
    ax1.bar(x, [max(cse["linear_residual"], 1e-18) for cse in b["cases"]], w,
            label="linear ||(I-RF)J-e||")
    ax1.bar(x + w, [max(cse["closure_residual"], 1e-18) for cse in b["cases"]], w,
            label="surface closure |q-e(sT^4-G)|")
    ax1.axhline(TOL_B, ls="--", c="k", lw=1, label="criterion 1e-6")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=22, ha="right", fontsize=6.5)
    ax1.set_ylabel("normalised residual")
    ax1.set_title(f"(b) radiosity audit, worst residual {b['worst_residual']:.1e}")
    ax1.legend(fontsize=6.5)
    ax1.grid(alpha=0.3, axis="y")

    cube = [cse for cse in b["cases"] if cse["case"].startswith("cube")][0]
    faces = np.arange(6)
    ax2.bar(faces, cube["net_flux_by_face_W_m2"], color=["#d62728", "#d62728"] + ["#1f77b4"] * 4)
    ax2.axhline(0.0, c="k", lw=0.8)
    ax2.set_xticks(faces)
    ax2.set_xticklabels([f"face{i}" for i in faces])
    ax2.set_ylabel("net radiative flux q [W/m2]")
    ax2.set_title("6-face cube (hot pair 420 K / cold 300 K): fluxes sum to zero")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = outdir / "P92W3_N02_fig2_radiosity_audit.png"
    fig.savefig(path)
    plt.close(fig)
    figs.append({"name": "fig2_radiosity_audit", "path": path,
                 "caption": "灰体封闭腔辐射度审计：(左) 各案例能量/线性/闭合残差与 1e-6 判据线；"
                            "(右) 六面立方体腔净辐射通量（热面放热、冷面吸热，总和为零）"})

    # fig 3: deck load vs analytic + tilt + diurnal
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    gray = c["gray_sky"]
    axes[0].plot([1.0] + [d["sky_emissivity"] for d in gray],
                 [c["rows"][1]["q_load_design_model_W_m2"]] + [d["q_load_full_chain_W_m2"] for d in gray],
                 "o-", label="full chain (radiosity)")
    axes[0].plot([1.0] + [d["sky_emissivity"] for d in gray],
                 [c["rows"][1]["q_load_design_reference_W_m2"]] + [d["q_load_analytic_W_m2"] for d in gray],
                 "x--", label="analytic reference")
    axes[0].set_xlabel("sky emissivity [-]")
    axes[0].set_ylabel("deck cooling duty [W/m2]")
    axes[0].set_title(f"(c) deck load at {c['design_temperature_K']:.0f} K vs analytic\nmax rel err {c['worst_rel_error']:.1e}", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    tilt = c["tilt_sweep"]
    axes[1].plot([d["tilt_deg"] for d in tilt], [d["T_deck_model"] for d in tilt], "o-", label="model")
    axes[1].plot([d["tilt_deg"] for d in tilt], [d["T_deck_reference"] for d in tilt], "x--",
                 label="reference")
    axes[1].set_xlabel("deck tilt [deg]")
    axes[1].set_ylabel("deck equilibrium temperature [K]")
    axes[1].set_title("tilt sweep (solar + convection)", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    ax = axes[2]
    ax.plot(diurnal["hours"], diurnal["T_deck"], "-", color="#d62728")
    ax.set_xlabel("solar time [h]")
    ax.set_ylabel("T_deck [K]", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#d62728")
    axb = ax.twinx()
    axb.plot(diurnal["hours"], diurnal["q_load"], "-", color="#1f77b4")
    axb.set_ylabel("cooling duty q [W/m2]", color="#1f77b4")
    axb.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_title("diurnal cycle (30N, day 172, gray sky)", fontsize=9)
    fig.tight_layout()
    path = outdir / "P92W3_N02_fig3_deck_load.png"
    fig.savefig(path)
    plt.close(fig)
    figs.append({"name": "fig3_deck_load", "path": path,
                 "caption": "船舶甲板热载荷：(左) 天空发射率 1.0/0.85/0.7 下全链（辐射度）与解析解冷却负荷；"
                            "(中) 倾角扫描甲板平衡温度（模型 vs 基准重合）；(右) 24h 昼夜循环甲板温度与冷却负荷"})

    # fig 4: boundary field (contour)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    X, Y = np.meshgrid(field["x"], field["y"], indexing="ij")
    cs0 = axes[0].contourf(Y, X, field["F_sky"], levels=14, cmap="viridis")
    fig.colorbar(cs0, ax=axes[0], label="F_deck->sky [-]")
    cs1 = axes[1].contourf(Y, X, field["q_field"], levels=14, cmap="inferno")
    fig.colorbar(cs1, ax=axes[1], label="q'' into deck [W/m2]")
    if field["cabin"] is not None:
        x0, x1, y0, y1, h = field["cabin"]
        for ax in axes:
            ax.add_patch(plt.Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False,
                                       ec="w", lw=1.2, ls="--"))
    for ax, title in zip(axes, ("sky view factor with superstructure shading",
                                f"deck wall heat flux, area mean {field['area_mean_q']:.0f} W/m2")):
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
        ax.set_title(title, fontsize=9)
    fig.suptitle("P92W3-N02 deck boundary field (12 m x 8 m deck, 3 m cabin block dashed)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = outdir / "P92W3_N02_fig4_deck_field.png"
    fig.savefig(path)
    plt.close(fig)
    figs.append({"name": "fig4_deck_field", "path": path,
                 "caption": "甲板边界条件场（可直接作为 LBM 壁面 Neumann 条件）：(左) 超结构遮挡下天空角系数分布；"
                            "(右) 稳态壁面热流 q''(x,y)，注意阴影带与边缘效应"})
    return figs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> dict:
    here = Path(__file__).resolve().parent
    print("[1/5] criterion (a): closed-form view-factor family ...")
    a = criterion_a()
    print("      pass:", a["pass"],
          {k: float(f"{v['max_rel_error']:.3e}") for k, v in a["family_summary"].items()})

    print("[2/5] criterion (b): enclosure radiosity energy conservation ...")
    b = criterion_b()
    print("      pass:", b["pass"], "worst:", f"{b['worst_residual']:.3e}")

    print("[3/5] criterion (c): deck thermal load vs analytic ...")
    c = criterion_c()
    print("      pass:", c["pass"], "worst:", f"{c['worst_rel_error']:.3e}")

    print("[4/5] boundary field + diurnal cycle ...")
    field = deck_boundary_field()
    diurnal = diurnal_profile()
    inj = injection_demo(field)

    print("[5/5] figures ...")
    figs = make_figures(a, b, c, field, diurnal, ROOT / "figures")

    result = {
        "mission": "P92W3-N02",
        "node": "TensorLBM thermal radiation verification surface",
        "criteria": {
            "a_view_factor_closed_form": {
                "tolerance": TOL_A, "pass": bool(a["pass"]),
                "worst": max(v["max_rel_error"] for v in a["family_summary"].values()),
                "note": "6 families, each vs an independent reference (quadrature or ray tracing)"},
            "b_enclosure_energy_conservation": {
                "tolerance": TOL_B, "pass": bool(b["pass"]), "worst": b["worst_residual"]},
            "c_deck_load_vs_analytic": {
                "tolerance": TOL_C, "pass": bool(c["pass"]), "worst": c["worst_rel_error"]},
        },
        "all_pass": bool(a["pass"] and b["pass"] and c["pass"]),
        "detail": {
            "a": a, "b": b, "c": c,
            "field": {
                "F_sky_min": float(field["F_sky"].min()),
                "F_sky_max": float(field["F_sky"].max()),
                "F_sky_mean": float(field["F_sky"].mean()),
                "shaded_area_fraction": field["shaded_fraction"],
                "T_wall_K": field["T_deck"],
                "q_field_min_W_m2": float(field["q_field"].min()),
                "q_field_max_W_m2": float(field["q_field"].max()),
                "area_mean_q_W_m2": field["area_mean_q"],
                "cabin": list(field["cabin"]) if field["cabin"] else None,
                "T_deck_reference_K": field["T_deck_reference"]},
            "environment": {
                "T_sky_night_K": night_env().sky_temperature(),
                "T_sky_day_K": day_env().sky_temperature(),
                "T_air_night_K": night_env().T_air,
                "T_dewpoint_night_K": night_env().T_dewpoint,
                "T_air_day_K": day_env().T_air,
                "T_dewpoint_day_K": day_env().T_dewpoint,
                "T_sea_K": 293.0,
                "G_beam_W_m2": day_env().sun.beam_irradiance,
                "G_diffuse_W_m2": day_env().sun.diffuse_horizontal,
                "sun_elevation_deg": day_env().sun.elevation_deg,
            },
            "diurnal": {k: v for k, v in diurnal.items()
                        if k not in ("hours", "T_deck", "q_load", "elevation")},
            "injection": inj,
        },
        "figures": [{"name": f["name"], "caption": f["caption"],
                     "path": str(Path(f["path"]).relative_to(ROOT))} for f in figs],
    }
    (here / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    art = ROOT / "artifacts" / "P92W3_N02_radiation_verification.json"
    art.parent.mkdir(exist_ok=True)
    art.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("      wrote", here / "result.json", "and", art)
    print("ALL PASS:", result["all_pass"])
    return result


if __name__ == "__main__":
    main()
