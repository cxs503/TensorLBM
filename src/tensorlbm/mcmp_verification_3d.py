"""3D MCMP verification and calibration tooling (Young-Laplace / spurious currents).

Purpose
-------
Three-dimensional analogue of :mod:`tensorlbm.mcmp_verification`: Young-Laplace
calibration of the two-component Shan-Chen model on D3Q19, with the same
guardrail discipline (no uninitialised numbers, explicit validity flags).

The 3D Laplace law is ``Delta p = 2 sigma / R`` (sphere), identical in form to
the 2D case (cylinder: ``Delta p = sigma / R``) but with a factor of 2 instead
of 1.  This module uses the **2** factor consistently.

Key differences from the 2D module:
* D3Q19 stencil (19 velocities) instead of D2Q9
* ``(nz, ny, nx)`` tensor ordering instead of ``(ny, nx)``
* 6-ray radius measurement (+/-x, +/-y, +/-z) instead of 4
* ``collide_sc_two_component_3d`` and ``stream3d`` instead of 2D equivalents
* ``sc_mcmp_pressure`` is reused (tensor-shape agnostic)
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .d3q19 import C, W, equilibrium3d, macroscopic3d
from .multiphase3d import collide_sc_two_component_3d
from .mcmp_verification import CS2, Guardrail, sc_mcmp_pressure
from .solver3d import stream3d

__all__ = [
    "CS2",
    "Guardrail",
    "init_static_droplet_3d",
    "init_planar_slab_3d",
    "simulate_planar_slab_3d",
    "barycentric_velocity_3d",
    "droplet_radius_3d",
    "measure_laplace_3d",
    "estimate_coexistence_3d",
    "young_laplace_calibration_3d",
    "simulate_static_droplet_3d",
]


# ---------------------------------------------------------------------------
# 3D grid helper
# ---------------------------------------------------------------------------


def _grid_3d(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.arange(n, dtype=torch.float32, device=device)
    return torch.meshgrid(idx, idx, idx, indexing="ij")


# ---------------------------------------------------------------------------
# Barycentric velocity (3D)
# ---------------------------------------------------------------------------


def barycentric_velocity_3d(
    f1: torch.Tensor,
    f2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mass-averaged velocity of the two components in 3D.

    Returns ``(ux, uy, uz, rho_tot)``.
    """
    rho1, ux1, uy1, uz1 = macroscopic3d(f1)
    rho2, ux2, uy2, uz2 = macroscopic3d(f2)
    rho_tot = rho1 + rho2
    safe = torch.clamp(rho_tot, min=1e-12)
    return (
        (rho1 * ux1 + rho2 * ux2) / safe,
        (rho1 * uy1 + rho2 * uy2) / safe,
        (rho1 * uz1 + rho2 * uz2) / safe,
        rho_tot,
    )


# ---------------------------------------------------------------------------
# Static droplet: initialisation, running, measurement (3D)
# ---------------------------------------------------------------------------


def init_static_droplet_3d(
    n: int,
    radius: float,
    rho1_in: float,
    rho2_in: float,
    rho1_out: float,
    rho2_out: float,
    *,
    interface_width: float = 2.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tanh-profiled spherical droplet of component 1 in a component-2 bath (3D).

    The profile is ``s = 0.5 (1 - tanh((r - R0)/w))`` and
    ``rho_s = rho_s^gas + s (rho_s^liq - rho_s^gas)``; populations are set to the
    local equilibrium at rest.
    """
    if not isinstance(n, int) or n < 4:
        raise ValueError(f"n must be an integer grid size >= 4, got {n!r}")
    dev = torch.device(device)
    zz, yy, xx = _grid_3d(n, dev)
    centre = (n - 1) / 2.0
    r = torch.sqrt((xx - centre) ** 2 + (yy - centre) ** 2 + (zz - centre) ** 2)
    s = 0.5 * (1.0 - torch.tanh((r - radius) / max(interface_width, 1e-6)))
    rho1 = rho1_out + s * (rho1_in - rho1_out)
    rho2 = rho2_out + s * (rho2_in - rho2_out)
    zero = torch.zeros_like(rho1)
    return equilibrium3d(rho1, zero, zero, zero), equilibrium3d(rho2, zero, zero, zero)


def simulate_static_droplet_3d(
    *,
    G_12: float,
    n: int,
    radius_init: float,
    rho1_in: float = 0.92,
    rho2_in: float = 0.08,
    rho1_out: float = 0.14,
    rho2_out: float = 0.86,
    steps: int = 8000,
    tau1: float = 1.0,
    tau2: float = 1.0,
    u_eq: str = "self",
    interface_width: float = 2.0,
    stop_tol: float = 1e-4,
    check_every: int = 250,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Relax a static two-component droplet in 3D and return fields plus diagnostics.

    Convergence is declared when the relative change of ``|u|_max`` over one
    ``check_every`` window drops below ``stop_tol`` for two consecutive windows.
    """
    dev = torch.device(device)
    f1, f2 = init_static_droplet_3d(
        n, radius_init, rho1_in, rho2_in, rho1_out, rho2_out,
        interface_width=interface_width, device=dev,
    )
    history: list[list[float]] = []
    u_prev = float("inf")
    stable_windows = 0
    converged = False
    steps_run = 0
    for step in range(1, steps + 1):
        f1, f2 = collide_sc_two_component_3d(f1, f2, G_12=G_12, tau1=tau1, tau2=tau2, u_eq=u_eq)
        f1, f2 = stream3d(f1), stream3d(f2)
        steps_run = step
        if step % check_every == 0:
            ux, uy, uz, _ = barycentric_velocity_3d(f1, f2)
            u_max = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max())
            if not math.isfinite(u_max):
                history.append([float(step), float("nan")])
                break
            history.append([float(step), u_max])
            if math.isfinite(u_prev) and abs(u_max - u_prev) <= stop_tol * max(u_max, 1e-12):
                stable_windows += 1
                if stable_windows >= 2:
                    converged = True
                    break
            else:
                stable_windows = 0
            u_prev = u_max
    rho1, _, _, _ = macroscopic3d(f1)
    rho2, _, _, _ = macroscopic3d(f2)
    ux, uy, uz, _ = barycentric_velocity_3d(f1, f2)
    return {
        "G_12": G_12,
        "n": n,
        "radius_init": radius_init,
        "tau1": tau1,
        "tau2": tau2,
        "u_eq": u_eq,
        "steps_requested": steps,
        "steps_run": steps_run,
        "converged": converged,
        "u_max": float(torch.sqrt(ux * ux + uy * uy + uz * uz).max()),
        "history": history,
        "rho1": rho1,
        "rho2": rho2,
        "ux": ux,
        "uy": uy,
        "uz": uz,
    }


# ---------------------------------------------------------------------------
# 3D box mean
# ---------------------------------------------------------------------------


def _box_mean_3d(field: torch.Tensor, z0: int, y0: int, x0: int, size: int) -> float:
    return float(field[z0 : z0 + size, y0 : y0 + size, x0 : x0 + size].mean())


# ---------------------------------------------------------------------------
# 3D droplet radius measurement (6 rays)
# ---------------------------------------------------------------------------


def droplet_radius_3d(
    rho1: torch.Tensor,
    rho_av: float,
    *,
    n_rays: int = 6,
) -> dict[str, Any]:
    """Radius of a centred 3D droplet from the ``rho_1 = rho_av`` crossing.

    Scans along 6 rays (+/-x, +/-y, +/-z) from the centre, interpolating
    linearly at the interface crossing.  Returns mean radius, spread, and
    guardrails.
    """
    guardrails: list[Guardrail] = []
    nz, ny, nx = rho1.shape
    if nz != ny or ny != nx:
        raise ValueError("droplet_radius_3d expects a cubic field")
    n = nz
    c = n // 2

    # Build 6 rays from centre
    scans = [
        [float(v) for v in rho1[c, c, c:]],                          # +x
        [float(v) for v in torch.flip(rho1[c, c, :c + 1], dims=(0,))],  # -x
        [float(v) for v in rho1[c, c:, c]],                          # +y
        [float(v) for v in torch.flip(rho1[c, :c + 1, c], dims=(0,))],  # -y
        [float(v) for v in rho1[c:, c, c]],                          # +z
        [float(v) for v in torch.flip(rho1[:c + 1, c, c], dims=(0,))],  # -z
    ]
    crossings = 0
    radii: list[float] = []
    for ray in scans[:n_rays]:
        if not all(math.isfinite(v) for v in ray):
            guardrails.append(
                Guardrail("non_finite_field", "the density ray contains NaN/Inf: the run diverged")
            )
            return {"radius": None, "radius_spread": None, "crossings": 0, "guardrails": guardrails}
        if ray[0] <= rho_av:
            guardrails.append(
                Guardrail(
                    "centre_not_liquid",
                    f"rho1 at the centre is {ray[0]:.4f} <= rho_av {rho_av:.4f}: "
                    "no droplet to measure",
                )
            )
            break
        inside = [i for i, v in enumerate(ray) if v > rho_av]
        if len(inside) == len(ray):
            continue
        i_in = inside[-1]
        r_in, r_out = ray[i_in], ray[i_in + 1]
        denom = r_in - r_out
        frac = 0.0 if abs(denom) < 1e-12 else (r_in - rho_av) / denom
        radii.append(i_in + min(max(frac, 0.0), 1.0))
        crossings += 1
    if crossings < n_rays:
        guardrails.append(
            Guardrail(
                "no_crossing",
                f"only {crossings}/{n_rays} rays crossed rho_av={rho_av:.4f}; droplet "
                "likely dissolved or merged with its periodic images",
            )
        )
    if not radii:
        return {"radius": None, "radius_spread": None, "crossings": crossings, "guardrails": guardrails}
    radius = sum(radii) / len(radii)
    spread = max(radii) - min(radii)
    if radius > 0.35 * n:
        guardrails.append(
            Guardrail(
                "periodic_image_risk",
                f"R={radius:.2f} exceeds 0.35*n={0.35 * n:.2f}; the droplet interacts "
                "with its periodic images and the Laplace pressure is inflated",
            )
        )
    if spread > max(1.5, 0.05 * radius):
        guardrails.append(
            Guardrail(
                "anisotropic",
                f"ray radii spread {spread:.2f} > max(1.5, 5% of R={radius:.2f}): the "
                "interface is not spherical (not relaxed, or a wall is present)",
            )
        )
    return {"radius": radius, "radius_spread": spread, "crossings": crossings, "guardrails": guardrails}


# ---------------------------------------------------------------------------
# 3D Laplace measurement
# ---------------------------------------------------------------------------


def measure_laplace_3d(run: dict[str, Any], *, box: int = 4) -> dict[str, Any]:
    """Measure ``Delta p``, ``R`` and ``sigma = Delta p R / 2`` for one 3D run.

    Liquid pressure is averaged in a ``box x box x box`` window at the centre,
    gas pressure in a window at a corner, both with :func:`sc_mcmp_pressure`.
    """
    rho1, rho2 = run["rho1"], run["rho2"]
    nz, ny, nx = rho1.shape
    if not (bool(torch.isfinite(rho1).all()) and bool(torch.isfinite(rho2).all())):
        return {
            "G_12": run["G_12"],
            "n": nz,
            "radius_init": run["radius_init"],
            "steps_run": run["steps_run"],
            "converged": False,
            "u_max": run["u_max"],
            "p_liq": None, "p_gas": None, "dp": None,
            "radius": None, "radius_spread": None, "sigma": None,
            "rho1_liq": None, "rho1_gas": None,
            "rho2_liq": None, "rho2_gas": None,
            "density_ratio": None,
            "guardrails": [dict(Guardrail("non_finite_field",
                f"density field is not finite after {run['steps_run']} steps: the run diverged"))],
        }
    p = sc_mcmp_pressure(rho1, rho2, run["G_12"])
    c = nz // 2
    lo = c - box // 2
    p_liq = _box_mean_3d(p, lo, lo, lo, box)
    p_gas = _box_mean_3d(p, 0, 0, 0, box)
    rho1_liq = _box_mean_3d(rho1, lo, lo, lo, box)
    rho1_gas = _box_mean_3d(rho1, 0, 0, 0, box)
    rho2_liq = _box_mean_3d(rho2, lo, lo, lo, box)
    rho2_gas = _box_mean_3d(rho2, 0, 0, 0, box)
    rho_av = 0.5 * (rho1_liq + rho1_gas)
    measured = droplet_radius_3d(rho1, rho_av)
    guardrails: list[Guardrail] = list(measured["guardrails"])
    if not run["converged"]:
        guardrails.append(
            Guardrail("not_converged",
                f"stopped at step {run['steps_run']} without reaching a |u| plateau")
        )
    dp = p_liq - p_gas
    radius = measured["radius"]
    sigma = None if radius is None else 0.5 * dp * radius  # sphere: dp = 2*sigma/R => sigma = dp*R/2
    if sigma is not None and sigma <= 0:
        guardrails.append(
            Guardrail("negative_sigma",
                f"Delta p = {dp:.6f} gives sigma = {sigma:.6f} <= 0")
        )
    return {
        "G_12": run["G_12"], "n": nz, "radius_init": run["radius_init"],
        "steps_run": run["steps_run"], "converged": run["converged"],
        "u_max": run["u_max"],
        "p_liq": p_liq, "p_gas": p_gas, "dp": dp,
        "radius": radius, "radius_spread": measured["radius_spread"],
        "sigma": sigma,
        "rho1_liq": rho1_liq, "rho1_gas": rho1_gas,
        "rho2_liq": rho2_liq, "rho2_gas": rho2_gas,
        "density_ratio": rho1_liq / max(rho1_gas, 1e-12),
        "guardrails": [dict(g) for g in guardrails],
    }


# ---------------------------------------------------------------------------
# 3D planar slab (coexistence)
# ---------------------------------------------------------------------------


def init_planar_slab_3d(
    n: int,
    rho1_in: float,
    rho2_in: float,
    rho1_out: float,
    rho2_out: float,
    *,
    half_thickness: float = 20.0,
    interface_width: float = 2.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Planar slab initialisation for curvature-free coexistence (3D)."""
    dev = torch.device(device)
    zz, yy, xx = _grid_3d(n, dev)
    centre = (n - 1) / 2.0
    # Slab in z-direction: component 1 fills |z - centre| < half_thickness
    s = 0.5 * (
        1.0 - torch.tanh((torch.abs(zz - centre) - half_thickness) / max(interface_width, 1e-6))
    )
    rho1 = rho1_out + s * (rho1_in - rho1_out)
    rho2 = rho2_out + s * (rho2_in - rho2_out)
    zero = torch.zeros_like(rho1)
    return equilibrium3d(rho1, zero, zero, zero), equilibrium3d(rho2, zero, zero, zero)


def simulate_planar_slab_3d(
    *,
    G_12: float,
    n: int,
    rho1_in: float = 0.92,
    rho2_in: float = 0.08,
    rho1_out: float = 0.14,
    rho2_out: float = 0.86,
    half_thickness: float = 20.0,
    steps: int = 8000,
    tau1: float = 1.0,
    tau2: float = 1.0,
    u_eq: str = "self",
    interface_width: float = 2.0,
    stop_tol: float = 1e-4,
    check_every: int = 250,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Relax a planar two-phase slab in 3D."""
    dev = torch.device(device)
    f1, f2 = init_planar_slab_3d(
        n, rho1_in, rho2_in, rho1_out, rho2_out,
        half_thickness=half_thickness, interface_width=interface_width, device=dev,
    )
    history: list[list[float]] = []
    u_prev = float("inf")
    stable = 0
    converged = False
    steps_run = 0
    for step in range(1, steps + 1):
        f1, f2 = collide_sc_two_component_3d(f1, f2, G_12=G_12, tau1=tau1, tau2=tau2, u_eq=u_eq)
        f1, f2 = stream3d(f1), stream3d(f2)
        steps_run = step
        if step % check_every == 0:
            ux, uy, uz, _ = barycentric_velocity_3d(f1, f2)
            u_max = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max())
            if not math.isfinite(u_max):
                history.append([float(step), float("nan")])
                break
            history.append([float(step), u_max])
            if math.isfinite(u_prev) and abs(u_max - u_prev) <= stop_tol * max(u_max, 1e-12):
                stable += 1
                if stable >= 2:
                    converged = True
                    break
            else:
                stable = 0
            u_prev = u_max
    rho1, _, _, _ = macroscopic3d(f1)
    rho2, _, _, _ = macroscopic3d(f2)
    ux, uy, uz, _ = barycentric_velocity_3d(f1, f2)
    return {
        "G_12": G_12, "n": n, "half_thickness": half_thickness,
        "steps_requested": steps, "steps_run": steps_run,
        "converged": converged,
        "u_max": float(torch.sqrt(ux * ux + uy * uy + uz * uz).max()),
        "history": history,
        "rho1": rho1, "rho2": rho2, "ux": ux, "uy": uy, "uz": uz,
    }


def estimate_coexistence_3d(
    *,
    G_12: float,
    n: int = 96,
    geometry: str = "planar",
    steps: int = 8000,
    device: torch.device | str = "cpu",
    **kwargs: Any,
) -> dict[str, Any]:
    """Coexistence densities from a flat slab (3D)."""
    if geometry == "planar":
        run = simulate_planar_slab_3d(
            G_12=G_12, n=n,
            half_thickness=kwargs.pop("half_thickness", 0.23 * n),
            steps=steps, device=device, **kwargs,
        )
        rho1, rho2 = run["rho1"], run["rho2"]
        c = n // 2
        box = 8
        p = sc_mcmp_pressure(rho1, rho2, G_12)
        out = {
            "G_12": G_12, "geometry": "planar", "n": n,
            "converged": run["converged"], "steps_run": run["steps_run"],
            "u_max": run["u_max"],
            "rho1_liq": _box_mean_3d(rho1, c - box // 2, c - box // 2, c - box // 2, box),
            "rho2_liq": _box_mean_3d(rho2, c - box // 2, c - box // 2, c - box // 2, box),
            "rho1_gas": _box_mean_3d(rho1, 2, 2, 2, box),
            "rho2_gas": _box_mean_3d(rho2, 2, 2, 2, box),
            "flat_interface_dp": (
                _box_mean_3d(p, c - box // 2, c - box // 2, c - box // 2, box)
                - _box_mean_3d(p, 2, 2, 2, box)
            ),
        }
        return out
    raise ValueError(f"geometry={geometry!r} not supported for 3D coexistence")


# ---------------------------------------------------------------------------
# 3D Young-Laplace calibration
# ---------------------------------------------------------------------------


def _fit_laplace_level_3d(
    G_12: float, level_cases: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    """One-parameter Young-Laplace test for 3D: Delta p = 2 sigma / R.

    Fits sigma through the origin: sigma_0 = sum(x_i y_i) / sum(x_i^2)
    where x_i = 2/R_i, y_i = Delta p_i.
    Also fits a free two-parameter line as diagnostic.
    """
    usable = [c for c in level_cases if c["sigma"] is not None and c["radius"]]
    guardrails: list[Guardrail] = []
    for c in level_cases:
        for g in c["guardrails"]:
            guardrails.append(Guardrail(g["code"], f"R_init={c['radius_init']}: {g['detail']}"))
    if len(usable) < 2:
        return {
            "G_12": G_12, "sigma_origin": None, "sigma_slope": None,
            "sigma_cases": [c["sigma"] for c in usable],
            "slope": None, "intercept": None, "r2": None,
            "max_rel_err": None, "pass": False,
            "guardrails": [dict(g) for g in guardrails],
            "reason": "fewer than two usable radii; no fit",
        }
    if len(usable) < 3:
        guardrails.append(
            Guardrail("too_few_radii", f"only {len(usable)} usable radii (law test wants >= 3)")
        )
    xs = [2.0 / c["radius"] for c in usable]
    ys = [c["dp"] for c in usable]
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    sigma_origin = sxy / sxx if sxx > 1e-30 else float("nan")
    errs = [abs(y - sigma_origin * x) / abs(sigma_origin * x) for x, y in zip(xs, ys)]
    max_err = max(errs)
    for c, e in zip(usable, errs):
        c["sigma_origin_level"] = sigma_origin
        c["rel_err_vs_law"] = e
    # free-line diagnostic
    m = len(xs)
    xs_inv = [1.0 / c["radius"] for c in usable]
    sx, sy = sum(xs_inv), sum(ys)
    sx2 = sum(x * x for x in xs_inv)
    sxi = sum(x * y for x, y in zip(xs_inv, ys))
    denom = m * sx2 - sx * sx
    slope = (m * sxi - sx * sy) / denom if abs(denom) > 1e-30 else float("nan")
    intercept = (sy - slope * sx) / m
    ss_tot = sum((y - sy / m) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs_inv, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")
    sigma_slope = 0.5 * slope
    if sigma_slope and abs(sigma_slope - sigma_origin) / abs(sigma_origin) > threshold:
        guardrails.append(
            Guardrail(
                "curvature_trend",
                f"free-line sigma {sigma_slope:.5f} differs from the through-origin "
                f"{sigma_origin:.5f} by {abs(sigma_slope - sigma_origin) / abs(sigma_origin):.1%}",
            )
        )
    return {
        "G_12": G_12,
        "sigma_origin": sigma_origin,
        "sigma_slope": sigma_slope,
        "sigma_cases": [c["sigma"] for c in usable],
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "max_rel_err": max_err,
        "pass": max_err <= threshold,
        "guardrails": [dict(g) for g in guardrails],
    }


def young_laplace_calibration_3d(
    *,
    G_values: list[float],
    radii: list[float],
    n: int = 96,
    steps: int = 12000,
    coexist_steps: int = 12000,
    threshold: float = 0.05,
    geometry: str = "planar",
    device: torch.device | str = "cpu",
    **kwargs: Any,
) -> dict[str, Any]:
    """Full Young-Laplace calibration protocol for 3D MCMP.

    For each coupling level G_12:
    1. Estimate coexistence densities from a planar slab
    2. Run droplets at each radius with coexistence boundary conditions
    3. Measure Delta p, R, sigma for each droplet
    4. Fit sigma through the origin and check per-case relative error

    Acceptance: max relative error <= threshold (default 5%), >= 3 radii per level.
    """
    cases: list[dict[str, Any]] = []
    levels: list[dict[str, Any]] = []
    for G in G_values:
        coex = estimate_coexistence_3d(
            G_12=G, n=max(n, 96), geometry=geometry, steps=coexist_steps, device=device, **kwargs,
        )
        level_cases: list[dict[str, Any]] = []
        for radius in radii:
            run = simulate_static_droplet_3d(
                G_12=G, n=n, radius_init=radius,
                rho1_in=coex["rho1_liq"], rho2_in=coex["rho2_liq"],
                rho1_out=coex["rho1_gas"], rho2_out=coex["rho2_gas"],
                steps=steps, device=device, **kwargs,
            )
            case = measure_laplace_3d(run)
            case["coexistence"] = {k: v for k, v in coex.items()}
            level_cases.append(case)
            cases.append(case)
        levels.append(_fit_laplace_level_3d(G, level_cases, threshold))
    ok = bool(levels) and all(level["pass"] for level in levels)
    return {
        "protocol": {
            "n": n, "steps": steps, "coexist_steps": coexist_steps,
            "radii": list(radii), "G_values": list(G_values),
            "threshold": threshold, "coexistence_geometry": geometry,
            "stress": "dP vs 2/R across radii, per G_12 level; sigma from the slope",
            "kwargs": {k: v for k, v in kwargs.items() if k != "device"},
        },
        "levels": levels,
        "cases": cases,
        "pass": ok,
    }
