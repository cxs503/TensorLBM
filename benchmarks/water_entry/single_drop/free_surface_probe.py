"""Independent constant-atmosphere D3Q19 check of the water-side interface evolution."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from benchmarks.water_entry.single_drop.run import (
    DropTracker,
    equivalent_radius,
    load_case,
    open_depth,
    scales,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    SOLID,
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
    total_liquid_inventory,
)
from tensorlbm.free_surface_topology_transaction import TopologyTransactionError


def _summarize_ledger(ledger):
    """Reduce the per-step runtime mass ledger to a compact operator attribution.

    Returns a JSON-safe dict: net/gross tracked-mass delta per operator, the
    dominant-operator histogram, and the cumulative unexplained residual.
    """
    net = defaultdict(float)
    gross = defaultdict(float)
    dominant = defaultdict(int)
    unexplained = 0.0
    for rec in ledger.get("steps", []):
        for ev in rec["operator_attribution"]["events"]:
            if ev["tracked_mass"]:
                net[ev["operator"]] += ev["net_delta"]
                gross[ev["operator"]] += ev["gross_magnitude"]
        dominant[rec["operator_attribution"]["dominant_operator"]] += 1
        unexplained += rec["unexplained"]
    return {
        "operator_net_tracked_mass": {k: v for k, v in net.items()},
        "operator_gross_tracked_mass": {k: v for k, v in gross.items()},
        "dominant_operator_counts": dict(dominant),
        "cumulative_unexplained_residual": unexplained,
        "steps_recorded": len(ledger.get("steps", [])),
    }


def _top_mach_cells(dens, ux, uy, uz, flags, fill, k=5):
    """Top-k active cells by lattice speed (diagnostic; off by default)."""
    active = (flags != GAS) & (flags != SOLID)
    u2 = ((ux**2 + uy**2 + uz**2) * active.to(ux.dtype)).flatten()
    k = min(k, int(active.sum()))
    if k <= 0:
        return []
    vals, idx = torch.topk(u2, k)
    shape = (dens.shape[0], dens.shape[1], dens.shape[2])
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        z, y, x = (int(t) for t in torch.unravel_index(torch.tensor(i), shape))
        out.append(
            {
                "z": z,
                "y": y,
                "x": x,
                "flag": int(flags[z, y, x]),
                "rho": float(dens[z, y, x]),
                "fill": float(fill[z, y, x]),
                "u": [float(ux[z, y, x]), float(uy[z, y, x]), float(uz[z, y, x])],
                "u_abs": float(v**0.5),
            }
        )
    return out



def _cavgas_diagnostics(field, dens, level, radius, n, sc):
    """Per-step cavity gas pressure diagnostics (pure observation).

    ``field`` is the interior fill slice (``fill[1:-1]``); ``level`` is the
    initial free surface in interior coordinates (``level - 1`` of the full
    grid, the same convention as :func:`open_depth`).  Pure observation:
    reads ``field``/``dens`` only, never mutates solver state.

    * ``cav_volume``: volume (in cells) of gas (fill < 0.5) below the initial
      surface inside the ROI r=1.5D (the air pocket the drop displaces).
    * ``rho_g_field``: mean macroscopic density of the gas cells in the
      cavity (lattice units; the model holds rho_gas=1.0 constant).
    * ``pressure_ratio_adiabatic``: (V0/V)^gamma with gamma=1.4 (Rayleigh
      1917 adiabatic cavity closure); None until V0 is established.
    * ``pressure_ratio_isothermal``: V0/V (isothermal p=p0*V0/V).
    """
    out = {}
    gas = field < 0.5
    z_int = np.arange(field.shape[0])
    ny, nx = field.shape[1:]
    yy, xx = np.indices((ny, nx))
    roi = (yy - (ny - 1) / 2) ** 2 + (xx - (nx - 1) / 2) ** 2 <= radius**2
    below = z_int[:, None, None] < level
    cav_gas = gas & below & roi[None]
    cav_volume = float(cav_gas.sum())
    out["cav_volume"] = cav_volume
    if cav_gas.any():
        out["rho_g_field"] = float(dens[1:-1][cav_gas].mean())
    else:
        out["rho_g_field"] = None
    return out


def _find_cavity_center(field, level, radius):
    """Deepest headspace-connected gas-to-liquid interface point (cavity bottom).

    Ported from the unmerged agent/m26730 branch (DE-X-DIAG1-CAV_PRESSURE card
    section 2).  ``field`` is the interior fill slice (``fill[1:-1]``); the
    headspace is seeded at the top interior slice (same convention as
    :func:`open_depth` in run.py) and propagated through the gas mask.
    Returns (z_idx, y_idx, x_idx) in field coordinates, or None.
    """
    gas = field < 0.5
    seeds = np.zeros_like(gas)
    seeds[-1] = gas[-1]
    atmosphere = ndimage.binary_propagation(seeds, mask=gas)
    y, x = np.indices(gas.shape[1:])
    roi = (y - (gas.shape[1] - 1) / 2) ** 2 + (x - (gas.shape[2] - 1) / 2) ** 2 <= radius**2
    k, j, i = np.where(atmosphere[1:] & ~gas[:-1] & roi[None])
    if not len(k):
        return None
    min_z_idx = k.argmin()
    return (int(k[min_z_idx]), int(j[min_z_idx]), int(i[min_z_idx]))


def _rayleigh_plesset_pressure(t, R0, p_inf, p_v, rho, sigma, mu, kappa=1.4, p_g0=0.0):
    """Rayleigh (1917)/Plesset (1949) closed-bubble pressure reference line.

    p_b = p_v + p_g0*(R0/R)^(3*kappa) - 2*sigma/R - 4*mu*R'/R with the
    analytical empty-cavity Rayleigh trajectory R(t) = R0*sqrt(1-(t/T_c)^2),
    T_c = 0.9147*R0*sqrt(rho/dp).  Reference line only: the K1-DIAG-B
    adjudication shows the cavity is an open ventilated pit, not a closed
    bubble, so this line is a deviation reference, not a prediction.
    """
    dp = max(abs(p_inf - p_v), 1e-30)
    T_c = 0.9147 * R0 * np.sqrt(rho / dp)
    t_clamped = min(t, T_c * 0.99)
    R = R0 * np.sqrt(max(0.0, 1 - (t_clamped / T_c) ** 2))
    if R < 1e-12:
        R = 1e-12
    Rdot = -R0**2 * t_clamped / (2 * R * T_c**2)
    p_b = p_v - 2.0 * sigma / R
    if p_g0:
        p_b += p_g0 * (R0 / R) ** (3.0 * kappa)
    p_b -= 4.0 * mu * Rdot / R
    return p_b, R, Rdot, T_c


def rayleigh_collapse_time_lattice(R0, dp, rho):
    """Rayleigh (1917) collapse time in lattice units."""
    return 0.9147 * R0 * np.sqrt(rho / max(dp, 1e-30))


def probe(
    n,
    speed,
    limit,
    gap_fraction,
    capacity=False,
    nearest=False,
    pilot=False,
    span=4.0,
    pool_depth=3.0,
    simple=False,
    release=None,
    planes_path=None,
    silhouette_path=None,
    ledger=False,
    diag_path=None,
    u_limit=None,
    cav_clos_alpha=0.0,
    rr_tau=0.0,
    C_s=0.0,
    diag_cavgas=False,
    cav_probe=False,
    cav_probe_out=None,
):
    # DE-X-COL2-REG: regularized-collision arm.  rr_tau > 0 switches the
    # free-surface collision from BGK to the regularized (RR) operator and
    # overrides the physical relaxation time with rr_tau (stability-window
    # scan).  rr_tau == 0 keeps the legacy BGK path bit-identical to ctrl.
    rr_tau = float(rr_tau or 0.0)
    if rr_tau < 0:
        raise ValueError(f"rr_tau must be >= 0, got {rr_tau!r}")
    rr_active = rr_tau > 0.0
    # Smagorinsky SGS constant (DE-X-TURB1-SMAG): 0 = legacy bit-identical path.
    C_s = float(C_s or 0.0)
    # Cell volume quadrature for the initial spherical volume fraction.
    case = load_case(pilot)
    sc = scales(case, n, speed)
    if u_limit is None:
        u_limit = 2.5 * speed
    u_limit = float(u_limit)
    nz, side = round((pool_depth + 2) * n) + 2, round(span * n)
    level = round(pool_depth * n) + 0.5
    aspect = case["aspect_ratio"]
    gap = gap_fraction * n
    mid = (side - 1) / 2
    z, y, x = (
        v.to(torch.float64)
        for v in torch.meshgrid(
            torch.arange(nz), torch.arange(side), torch.arange(side), indexing="ij"
        )
    )
    center = level + gap + 0.5 * n * aspect ** (2 / 3)
    drop = torch.zeros_like(z)
    offsets = (-0.375, -0.125, 0.125, 0.375)
    for dz in offsets:
        for dy in offsets:
            for dx in offsets:
                drop += (
                    equivalent_radius(z + dz - center, y + dy - mid, x + dx - mid, aspect) < 0.5 * n
                ) / 64
    fill = torch.maximum(drop, (level - z + 0.5).clamp(0, 1))
    solid = torch.zeros_like(fill, dtype=torch.bool)
    solid[[0, -1]] = True
    fill[solid] = 0
    flags = init_flags_from_fill(fill, solid)
    active = (flags != GAS) & (flags != SOLID)
    moving = z > level + 0.5 * gap
    if nearest:
        # For a sub-cell air gap the empty pool envelope shares a horizontal
        # plane with the drop. Assign velocity by closest physical surface.
        dist_drop = equivalent_radius(z - center, y - mid, x - mid, aspect) - 0.5 * n
        moving = dist_drop < z - level
    release_lattice = (
        math.sqrt(speed**2 - 2 * sc["g"] * gap)
        if release is None
        else release * sc["dt_s"] / sc["dx_m"]
    )
    velocity = -release_lattice * (moving & active)
    # Ambient gas pressure is represented as a reference liquid density of one.
    rho = 1 + 3 * (sc["g"] * (level - z).clamp_min(0) + (4 * sc["sigma"] / n) * moving)
    f = equilibrium3d(rho * active, torch.zeros_like(z), torch.zeros_like(z), velocity)
    mass = init_mass_from_fill(fill, flags) * rho
    mass0 = total_liquid_inventory(f, fill, flags).item()
    geometric_volume0 = float(fill.sum())
    stored_mass0 = float(mass.sum())
    history = []
    plane_times, plane_fill, plane_uz = [], [], []
    sample_targets_ms = (1.39643, 6.4, 10.39446, 13.4)

    # Cavitation pressure probe (DE-X-GAS2-CAVPROBE-R1; read-only, off by default)
    cav_probe_data = [] if cav_probe else None
    cav_R0 = 0.5 * n * sc["dx_m"]  # Initial cavity radius ~ half drop radius
    cav_p_inf = 1.0 + 3 * sc["g"] * level  # Ambient pressure at surface (lattice units)
    cav_p_v = 1.0  # Vapor pressure ~ atmospheric (lattice units)
    cav_rho = 1.0  # Liquid density (lattice units)
    cav_sigma = sc["sigma"]
    cav_mu = sc["mu_l"]
    cav_contact_step = None
    cav_collapse_time = None
    cav_T_rayleigh = None

    silhouette_samples = []
    tracker = DropTracker()
    last_speed = None
    impact_speed = None
    reason = "step_limit"
    runtime_ledger = {"steps": []} if ledger else None
    diag_rows = [] if diag_path is not None else None
    cavgas_rows = [] if diag_cavgas else None
    cav_volume0 = None
    for step in range(limit + 1):
        dens, ux, uy, uz = macroscopic3d(f)
        active = (flags != GAS) & (flags != SOLID)
        mach = float((3 * (ux**2 + uy**2 + uz**2))[active].max().sqrt())
        drift = abs(total_liquid_inventory(f, fill, flags).item() / mass0 - 1)
        row = {
            "step": step,
            "time_ms": 1000 * step * sc["dt_s"],
            "volume_drift": drift,
            "geometric_volume_drift": abs(float(fill.sum()) / geometric_volume0 - 1),
            "represented_mass_drift": abs(
                float(dens[flags == LIQUID].sum() + mass[flags == INTERFACE].sum()) / stored_mass0
                - 1
            ),
            "mach_max": mach,
            "rho_min": float(dens[active].min()),
        }
        history.append(row)
        if diag_rows is not None:
            diag_rows.append(
                {
                    "step": step,
                    "mach_max": mach,
                    "n_iface_fast": int(((ux**2 + uy**2 + uz**2) > 0.01).logical_and(active).sum()),
                    "top": _top_mach_cells(dens, ux, uy, uz, flags, fill, 5),
                }
            )
        if not all(math.isfinite(v) for v in row.values()):
            reason = "nonfinite"
            break
        if (
            mach > case["gates"]["maximum_lattice_mach"]
            or row["rho_min"] <= 0
            or drift > case["gates"]["maximum_liquid_volume_relative_drift"]
        ):
            reason = "health_limit"
            break
        field = fill[1:-1].numpy()
        try:
            detached = tracker.update(field, step, level - 1)
        except ValueError as error:
            reason = str(error)
            break
        if detached is not None:
            if silhouette_path is not None:
                from benchmarks.water_entry.single_drop.silhouette import centre_z

                centre = centre_z(field, detached)
                if centre is not None:
                    silhouette_samples.append(
                        dict(
                            time_ms=row["time_ms"],
                            centre_height_mm=(centre - level + 1) * sc["dx_m"] * 1000,
                        )
                    )
            last_speed = float(-uz[1:-1][detached].mean()) * sc["dx_m"] / sc["dt_s"]
        elif tracker.contact_step == step:
            impact_speed = last_speed
        if step and step % 50 == 0:
            print(
                f"step {step}, t={row['time_ms']:.4f} ms, contact={tracker.contact_step}",
                flush=True,
            )

        row["depth_over_D"] = open_depth(field, level - 1, 1.5 * n) / n
        if cav_probe and tracker.contact_step is not None:
            cavity_center = _find_cavity_center(field, level - 1, 1.5 * n)
            if cavity_center is not None:
                cz, cy, cx = cavity_center
                p_cav_lattice = float(dens[1:-1][cz, cy, cx])
                p_cav_pa = p_cav_lattice * sc["rho_g"] * sc["dx_m"] / sc["dt_s"] ** 2 * 1e3
                if cav_contact_step is None:
                    cav_contact_step = step
                    cav_T_rayleigh = rayleigh_collapse_time_lattice(cav_R0, cav_p_inf - cav_p_v, cav_rho)
                t_since_contact = (step - cav_contact_step) * sc["dt_s"]
                p_rp, R_t, Rdot_t, T_c = _rayleigh_plesset_pressure(
                    t_since_contact, cav_R0, cav_p_inf, cav_p_v, cav_rho, cav_sigma, cav_mu
                )
                p_rp_pa = p_rp * sc["rho_g"] * sc["dx_m"] / sc["dt_s"] ** 2 * 1e3
                cav_probe_data.append(
                    {
                        "step": step,
                        "time_ms": row["time_ms"],
                        "time_since_contact_ms": t_since_contact * 1000,
                        "p_cav_lattice": p_cav_lattice,
                        "p_cav_pa": p_cav_pa,
                        "p_ref_lattice": cav_p_v,
                        "p_rp_lattice": p_rp,
                        "p_rp_pa": p_rp_pa,
                        "p_rp_pa_minus_p_atm_pa": p_rp_pa - 101325.0,
                        "R_t": R_t,
                        "Rdot_t": Rdot_t,
                        "T_collapse_s": T_c,
                        "cavity_center_z": cz,
                        "cavity_center_y": cy,
                        "cavity_center_x": cx,
                        "depth_over_D": row["depth_over_D"],
                    }
                )
                if cav_collapse_time is None and cav_T_rayleigh is not None:
                    cav_collapse_time = cav_T_rayleigh
        if diag_cavgas and tracker.contact_step is not None:
            diag = _cavgas_diagnostics(field, dens, level - 1, 1.5 * n, n, sc)
            if cav_volume0 is None:
                cav_volume0 = diag["cav_volume"]
            if cav_volume0 is not None and cav_volume0 > 0 and diag["cav_volume"] > 0:
                diag["pressure_ratio_adiabatic"] = (cav_volume0 / diag["cav_volume"]) ** 1.4
                diag["pressure_ratio_isothermal"] = cav_volume0 / diag["cav_volume"]
            else:
                diag["pressure_ratio_adiabatic"] = None
                diag["pressure_ratio_isothermal"] = None
            row.update(diag)
            cavgas_rows.append(diag)
        if (
            planes_path is not None
            and tracker.contact_step is not None
            and len(plane_times) < len(sample_targets_ms)
        ):
            target_step = round(sample_targets_ms[len(plane_times)] / (1000 * sc["dt_s"]))
            if step - tracker.contact_step == target_step:
                plane_times.append(target_step * 1000 * sc["dt_s"])
                plane_fill.append(field[:, side // 2, :].copy())
                plane_uz.append((uz[1:-1, side // 2, :] * (sc["dx_m"] / sc["dt_s"])).numpy().copy())

        if tracker.contact_step is not None and (step - tracker.contact_step) * sc["dt_s"] >= 0.024:
            reason = "window_complete"
            break
        if step < limit:
            try:
                f, fill, flags, mass, _ = free_surface_step(
                    f,
                    fill,
                    flags,
                    solid,
                    mass=mass,
                    tau=rr_tau if rr_active else 0.5 + 3 * sc["mu_l"],
                    gz=-sc["g"],
                    surface_tension=sc["sigma"],
                    rho_gas=1.0,
                    collision="rr" if rr_active else "bgk",
                    capacity_redistribution=capacity,
                    simple_mass_exchange=simple,
                    runtime_ledger=runtime_ledger,
                    u_limit=u_limit,
                    cav_clos_alpha=cav_clos_alpha,
                    C_s=C_s,
                )
            except TopologyTransactionError as error:
                reason = f"topology_step_{step + 1}: {error}"
                break

    silhouette_summary = None
    if silhouette_path is not None:
        from benchmarks.water_entry.single_drop.silhouette import summarize

        silhouette_path.parent.mkdir(exist_ok=True, parents=True)
        silhouette_summary = summarize(silhouette_samples)
        silhouette_path.write_text(json.dumps(silhouette_summary, indent=2, allow_nan=False) + "\n")
    if planes_path is not None and plane_times:
        planes_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(
            planes_path,
            time_ms=plane_times,
            fill=plane_fill,
            uz_m_s=plane_uz,
            x_over_D=(np.arange(side) - mid) / n,
            z_over_D=(np.arange(1, nz - 1) - level) / n,
            y_over_D=(side // 2 - mid) / n,
        )
    if diag_rows is not None:
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag_path.write_text(json.dumps({"rows": diag_rows}, indent=1, allow_nan=False) + "\n")
    if cav_probe and cav_probe_data:
        cav_result = {
            "probe": "cavitation_pressure",
            "cavity_center_radius_m": cav_R0,
            "ambient_pressure_lattice": cav_p_inf,
            "vapor_pressure_lattice": cav_p_v,
            "liquid_density_lattice": cav_rho,
            "surface_tension_lattice": cav_sigma,
            "viscosity_lattice": cav_mu,
            "contact_step": cav_contact_step,
            "T_rayleigh_s": cav_T_rayleigh if cav_T_rayleigh else None,
            "data": cav_probe_data,
        }
        if cav_probe_out is not None:
            cav_probe_out.parent.mkdir(parents=True, exist_ok=True)
            cav_probe_out.write_text(json.dumps(cav_result, indent=2, allow_nan=False) + "\n")
        result_cav_probe = cav_result
    else:
        result_cav_probe = None
    result = {
        "solver": "free_surface_step/D3Q19, constant atmospheric pressure",
        "cells_per_D": n,
        "case_id": "pilot" if pilot else "santini_case3",
        "span_over_D": span,
        "pool_depth_over_D": pool_depth,
        "lattice_speed": speed,
        "scales": sc,
        "stop_reason": reason,
        "contact_step": tracker.contact_step,
        "gap_over_D": gap_fraction,
        "capacity_redistribution": capacity,
        "nearest_surface_velocity": nearest,
        "simple_mass_exchange": simple,
        "u_limit": u_limit,
        "rr_tau": rr_tau if rr_active else None,
        "C_s": C_s,
        "release_speed_m_s": release_lattice * sc["dx_m"] / sc["dt_s"],
        "prescribed_release_speed_m_s": release,
        "impact_speed_m_s": impact_speed,
        "side_view_measurement": silhouette_summary,
        "history": history,
    }
    if ledger:
        result["mass_ledger_summary"] = _summarize_ledger(runtime_ledger)
    if cav_probe:
        result["cavitation_probe"] = {
            "contact_step": cav_contact_step,
            "T_rayleigh_s": cav_T_rayleigh,
            "n_samples": len(cav_probe_data) if cav_probe_data else 0,
        }
        if result_cav_probe is not None:
            result["cav_probe"] = result_cav_probe
    if diag_cavgas and cavgas_rows:
        result["cavgas_diag"] = {
            "definitions": {
                "cav_volume": "volume (cells) of gas (fill<0.5) below initial surface inside ROI r=1.5D",
                "rho_g_field": "mean macroscopic density of gas cells in cavity (lattice units)",
                "pressure_ratio_adiabatic": "(V0/V)^1.4 (Rayleigh 1917 adiabatic)",
                "pressure_ratio_isothermal": "V0/V (isothermal)",
            },
            "cav_volume0": cav_volume0,
            "rows": cavgas_rows,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planes",
        type=Path,
        help="save mid-plane fill and vertical velocity at registered post-contact times",
    )
    parser.add_argument("--silhouette", type=Path, help="save detached side-view centre trajectory")
    parser.add_argument("--capacity", action="store_true")
    parser.add_argument("--simple-exchange", action="store_true")
    parser.add_argument(
        "--release-speed",
        type=float,
        help="separate SI release speed in m/s, fixed independently of the crater curve",
    )
    parser.add_argument("--nearest-surface", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--ledger",
        action="store_true",
        help="record the per-step runtime tracked-mass ledger and write a compact "
        "operator-attribution summary into the output JSON (diagnostic; adds overhead)",
    )
    parser.add_argument("--span", type=float, default=4.0)
    parser.add_argument("--pool-depth", type=float, default=3.0)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--speed", type=float, default=0.04)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--diag-out",
        type=Path,
        default=None,
        help="write per-step top-5 lattice-speed cells (diagnostic; off by default)",
    )
    parser.add_argument(
        "--u-limit",
        type=float,
        default=None,
        help="interface pseudo-current speed cap in lattice units (default 2.5*speed; 0 disables)",
    )
    parser.add_argument("--gap", type=float, default=0.4)
    parser.add_argument("--cav-clos-alpha", type=float, default=0.0, help="cavity closure delay factor: threshold = kappa*(1+alpha)")
    parser.add_argument(
        "--diag-cavgas",
        action="store_true",
        help="record per-step cavity volume and gas pressure ratio (diagnostic; off by default, zero behavior change)",
    )
    parser.add_argument(
        "--cav-probe",
        action="store_true",
        help="enable cavitation pressure probe at cavity bottom (read-only; off by default, zero behavior change)",
    )
    parser.add_argument(
        "--cav-probe-out",
        type=Path,
        default=None,
        help="output path for cavitation pressure data (JSON)",
    )
    args = parser.parse_args()
    if (
        args.span < 3
        or args.pool_depth < 2
        or args.cells < 8
        or not 0 < args.speed < 0.15
        or not 0 < args.gap <= 0.8
        or (
            args.release_speed is not None
            and not 0 < args.release_speed < 2 * load_case()["velocity_m_s"]
        )
        or args.steps < 0
    ):
        parser.error("invalid resolution, lattice speed, gap or step count")
    torch.set_num_threads(1)
    result = probe(
        args.cells,
        args.speed,
        args.steps,
        args.gap,
        args.capacity,
        args.nearest_surface,
        args.pilot,
        args.span,
        args.pool_depth,
        args.simple_exchange,
        args.release_speed,
        args.planes,
        args.silhouette,
        args.ledger,
        args.diag_out,
        args.u_limit,
        args.cav_clos_alpha,
        0.0,  # rr_tau: not exposed on the CLI; legacy BGK path
        0.0,  # C_s: not exposed on the CLI; legacy path
        args.diag_cavgas,
        args.cav_probe,
        args.cav_probe_out,
    )
    for row in result["history"]:
        for key, value in row.items():
            if value is None or not math.isfinite(value):
                row[key] = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["stop_reason"], result["history"][-1])
    tol = load_case(args.pilot)["gates"]["maximum_liquid_volume_relative_drift"]
    volume_pass = all(
        row["geometric_volume_drift"] is not None and row["geometric_volume_drift"] <= tol
        for row in result["history"]
    )
    raise SystemExit(0 if result["stop_reason"] == "window_complete" and volume_pass else 1)
