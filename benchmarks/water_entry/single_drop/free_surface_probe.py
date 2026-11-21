"""Independent constant-atmosphere D3Q19 check of the water-side interface evolution."""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from benchmarks.water_entry.single_drop.budget_diag import budget_snapshot
from benchmarks.water_entry.single_drop.cavity_diag import (
    DropMaterialMarkers,
    cavity_diagnostics,
    closure_point,
    headspace_mask,
)
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


# Tree-local arm switch for the shared SDAA wrapper.  That wrapper fixes its
# environment list, so the cavity-diagnosis switch travels inside the tree that
# is transferred with the sources.  The file is deliberately not tracked: the
# diagnosis stays off by default for every other user.
DIAG_CAVITY_MARKER = Path(__file__).with_name("DIAG_CAVITY_ON")

# Tree-local arm switch for the shared SDAA wrapper (same pattern as the
# cavity-diagnosis marker): the file content is the momentum-carrying
# conversion mode ("inherit" / "redistribute").  The file is not tracked, so
# every other user keeps the legacy (bit-identical) behaviour.
MOMCONV_MARKER = Path(__file__).with_name("MOMCONV_MODE")
MOMCONV_MODES = ("off", "inherit", "redistribute", "redistribute_liq")


def diag_cavity_enabled(request):
    """Resolve the cavity-diagnosis switch.

    ``None`` means "auto": the diagnosis runs when the tree-local marker file
    exists.  ``True``/``False`` force it on/off.
    """
    if request is None:
        return DIAG_CAVITY_MARKER.exists()
    return bool(request)


def momconv_mode(request):
    """Resolve the momentum-carrying conversion mode.

    ``None`` means "auto": the mode is read from the ``TL_FS_MOMCONV``
    environment variable, else from the tree-local ``MOMCONV_MODE`` marker
    file, else the legacy mode "off" (bit-identical to the baseline).
    """
    if request is None:
        request = os.environ.get("TL_FS_MOMCONV") or None
        if request is None and MOMCONV_MARKER.exists():
            request = MOMCONV_MARKER.read_text().strip()
    mode = str(request or "off").strip().lower()
    if mode in ("", "none", "off", "0", "false"):
        return "off"
    if mode in MOMCONV_MODES:
        return mode
    raise ValueError(f"momconv must be one of {MOMCONV_MODES}, got {request!r}")


def _diag_summary(rows, history):
    """Reduce the diagnostic time series to peaks, minima and the depth-peak row."""
    def extreme(key, mode):
        best = None
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            if best is None or (value > best[0] if mode == "max" else value < best[0]):
                best = (value, row["time_ms"])
        if best is None:
            return None
        return {"value": best[0], "time_ms": best[1]}

    peak_step = None
    peak_value = None
    for entry in history:
        value = entry.get("depth_over_D")
        if value is None:
            continue
        if peak_value is None or value > peak_value:
            peak_value = value
            peak_step = entry["step"]
    at_peak = None
    for row in rows:
        if row["step"] == peak_step:
            at_peak = row
            break
    return {
        "depth_peak": {"value": peak_value, "step": peak_step, "time_ms": None},
        "at_depth_peak": at_peak,
        "peak_d_cav_open": extreme("d_cav_open", "max"),
        "peak_d_cav_all": extreme("d_cav_all", "max"),
        "peak_d_cav_gap": extreme("d_cav_gap", "max"),
        "peak_d_drop": extreme("d_drop", "max"),
        "peak_d_drop_any": extreme("d_drop_any", "max"),
        "peak_d_ceil": extreme("d_ceil", "max"),
        "min_p_cav": extreme("p_cav", "min"),
        "peak_u_back": extreme("u_back", "max"),
        "max_cav_pockets": extreme("cav_pockets", "max"),
        "rows": len(rows),
    }


def _closure_summary(rows, history, contact_step, sc):
    """Reduce the closure-point trajectory to the closure time and velocity.

    The closure time is the first step after the depth peak where the closure
    point (deepest open gas-liquid floor) starts moving up (z increasing, since
    axis 0 points up and a smaller index is deeper).  The closure velocity is
    the vertical speed of the closure point at that step.
    """
    if not rows:
        return {"t_close": None, "U_close": None, "rows": 0}
    # find the depth peak (deepest closure point)
    peak_step = None
    peak_depth = None
    for row in rows:
        d = row.get("depth")
        if d is None:
            continue
        if peak_depth is None or d > peak_depth:
            peak_depth = d
            peak_step = row["step"]
    # find the first step after the peak where z starts increasing (moving up)
    t_close = None
    U_close = None
    prev_z = None
    for row in rows:
        if row["step"] <= peak_step:
            prev_z = row.get("z")
            continue
        z = row.get("z")
        if z is None or prev_z is None:
            prev_z = z
            continue
        # axis 0 points up: smaller index = deeper, so moving up = z increasing
        if z > prev_z:
            t_close = row["time_ms"]
            # velocity: (z - prev_z) * dx / dt in m/s (upward = positive)
            dz = (z - prev_z) * sc["dx_m"]
            dt = (row["time_ms"] - (row["step"] - 1) * 1000 * sc["dt_s"]) / 1000
            U_close = dz / dt if dt > 0 else None
            break
        prev_z = z
    # t_close in ms after contact
    t_close_after = None
    if t_close is not None and contact_step is not None:
        t_contact = contact_step * 1000 * sc["dt_s"]
        t_close_after = t_close - t_contact
    return {
        "t_close": t_close,
        "t_close_after_contact_ms": t_close_after,
        "U_close": U_close,
        "peak_step": peak_step,
        "peak_depth": peak_depth,
        "rows": len(rows),
    }


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
    diag_cavity=None,
    gas_nu_ratio=1.0,
    diag_budget=False,
    momconv=None,
    cav_closure=False,
    gas_flow=False,
):
    diag_cavity = diag_cavity_enabled(diag_cavity)
    diag_budget = bool(diag_budget)
    momconv = momconv_mode(momconv)
    # The inherited gas velocity has to survive between steps.
    momconv_state = {}
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
    if gas_flow:
        # Potential flow of the exterior of a translating sphere, decaying as r^-3
        # (mirrors run.py --gas-flow so the SDAA-lane probe arm matches the CACSim arm).
        r = ((x - mid) ** 2 + (y - mid) ** 2 + (z - center) ** 2).sqrt().clamp_min(n / 2)
        pool_mask = (level - z + 0.5).clamp(0, 1)
        scale = -0.5 * release_lattice * (n / (2 * r)) ** 3 * (1 - drop) * (1 - pool_mask)
        ux = scale * 3 * (x - mid) * (z - center) / r**2
        uy = scale * 3 * (y - mid) * (z - center) / r**2
        uz = velocity + scale * (3 * ((z - center) / r) ** 2 - 1)
    else:
        ux = torch.zeros_like(z)
        uy = torch.zeros_like(z)
        uz = velocity
    f = equilibrium3d(rho * active, ux, uy, uz)
    mass = init_mass_from_fill(fill, flags) * rho
    mass0 = total_liquid_inventory(f, fill, flags).item()
    geometric_volume0 = float(fill.sum())
    stored_mass0 = float(mass.sum())
    history = []
    plane_times, plane_fill, plane_uz = [], [], []
    sample_targets_ms = (1.39643, 6.4, 10.39446, 13.4)

    silhouette_samples = []
    markers = None
    marker_state = None
    last_drop_mask = None
    cavity_rows = []
    closure_rows = []
    budget_rows = []
    prev_budget = None
    # the anti-bounce-back reconstruction carries the ambient gas pressure
    # rho_gas = 1.0 (pressure matched), so p_amb = cs^2 * rho_gas = 1/3
    p_amb = 1.0 / 3.0
    tracker = DropTracker()
    last_speed = None
    impact_speed = None
    reason = "step_limit"
    runtime_ledger = {"steps": []} if ledger else None
    diag_rows = [] if diag_path is not None else None
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
            if diag_cavity or diag_budget:
                last_drop_mask = detached
        elif tracker.contact_step == step:
            impact_speed = last_speed
            if (diag_cavity or diag_budget) and markers is None and last_drop_mask is not None:
                markers = DropMaterialMarkers.from_mask(last_drop_mask)
                marker_state = {
                    "mass": np.asarray(
                        mass[1:-1].numpy()[last_drop_mask], dtype=np.float64
                    )
                }
        if step and step % 50 == 0:
            print(
                f"step {step}, t={row['time_ms']:.4f} ms, contact={tracker.contact_step}",
                flush=True,
            )

        row["depth_over_D"] = open_depth(field, level - 1, 1.5 * n) / n
        if (diag_cavity or diag_budget or cav_closure) and tracker.contact_step is not None:
            uxi = ux[1:-1].numpy()
            uyi = uy[1:-1].numpy()
            uzi = uz[1:-1].numpy()
            if markers is not None:
                markers.advect(uxi, uyi, uzi, field, surf=level - 1, n=n)
            if diag_cavity:
                diag_row = cavity_diagnostics(
                    field,
                    level - 1,
                    n,
                    ux=uxi,
                    uy=uyi,
                    uz=uzi,
                    velocity_scale=sc["dx_m"] / sc["dt_s"],
                    markers=markers,
                )
                diag_row["step"] = step
                diag_row["time_ms"] = row["time_ms"]
                cavity_rows.append(diag_row)
            if cav_closure:
                gas = field < 0.5
                headspace = headspace_mask(gas)
                side = field.shape[1]
                center = (side - 1) / 2
                yy, xx = np.indices(field.shape[1:])
                roi = (yy - center) ** 2 + (xx - center) ** 2 <= (1.5 * n) ** 2
                cp = closure_point(field, gas, headspace, roi, level - 1, n)
                if cp is not None:
                    cp["step"] = step
                    cp["time_ms"] = row["time_ms"]
                    closure_rows.append(cp)
            if diag_budget:
                mass_np = mass[1:-1].numpy()
                budget_row = budget_snapshot(
                    field,
                    flags[1:-1].numpy(),
                    mass_np,
                    uxi,
                    uyi,
                    uzi,
                    level - 1,
                    n,
                    sigma=sc["sigma"],
                    mu=sc["mu_l"],
                    g=sc["g"],
                    p_amb=p_amb,
                    u_limit=u_limit,
                    markers=markers,
                    marker_state=marker_state,
                    prev=prev_budget,
                )
                budget_row["step"] = step
                budget_row["time_ms"] = row["time_ms"]
                budget_rows.append(budget_row)
                prev_budget = {
                    "mass": np.asarray(mass_np, dtype=np.float64),
                    "u2": uxi**2 + uyi**2 + uzi**2,
                }
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
                    tau=0.5 + 3 * sc["mu_l"],
                    gz=-sc["g"],
                    surface_tension=sc["sigma"],
                    rho_gas=1.0,
                    collision="bgk",
                    capacity_redistribution=capacity,
                    simple_mass_exchange=simple,
                    runtime_ledger=runtime_ledger,
                    u_limit=u_limit,
                    gas_nu_ratio=gas_nu_ratio,
                    momconv=momconv,
                    momconv_state=momconv_state,
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
        "gas_nu_ratio": gas_nu_ratio,
        "momconv": momconv,
        "release_speed_m_s": release_lattice * sc["dx_m"] / sc["dt_s"],
        "prescribed_release_speed_m_s": release,
        "impact_speed_m_s": impact_speed,
        "side_view_measurement": silhouette_summary,
        "history": history,
    }
    if ledger:
        result["mass_ledger_summary"] = _summarize_ledger(runtime_ledger)
    if diag_budget:
        result["budget_diag"] = {
            "enabled": True,
            "units": "lattice (dx=dt=1, rho_liquid=1); convert with rho_l*dx^3 (energy)",
            "definitions": {
                "ke_liq": "1/2 sum m|u|^2 over liquid+interface cells",
                "ke_axial": "1/2 sum m u_z^2 (axial / water-column motion)",
                "ke_lat": "1/2 sum m (u_x^2+u_y^2) (lateral / jet motion)",
                "ke_rad": "radial part of ke_lat (u_r = e_r . u in the x-y plane)",
                "ke_lat_roi": "ke_lat restricted to the crater footprint (r<=1.5 D, below the initial surface)",
                "ke_drop": "1/2 sum m_marker|u_marker|^2 of the tracked drop material",
                "area_total": "sum |grad phi| (smeared-interface area, lattice area units)",
                "area_cav": "the cavity-wall part of area_total",
                "cav_volume": "cavity gas volume (lattice volume units)",
                "p_lap_mean": "mean Laplace pressure sigma*kappa on the cavity wall",
                "gas_force_*": "force of the cavity gas on the liquid (item 1, momentum)",
                "gas_power": "work rate of the cavity gas on the liquid (item 1, energy)",
                "grav_power": "work rate of gravity on the liquid",
                "diss_rate": "continuum viscous dissipation rate 2 mu |S|^2 (item 3)",
                "diss_iface": "the interface-neighbourhood part of diss_rate",
                "diss_bulk": "the bulk part of diss_rate",
                "conv_loss": "kinetic energy carried away by liquid mass converted to gas",
                "conv_gain": "kinetic energy created by gas mass converted back to liquid",
                "n_at_cap": "number of interface cells sitting on the pseudo-current cap (u_limit)",
                "ke_at_cap": "kinetic energy held by the cells sitting on the cap",
                "p_liq_*": "total liquid momentum",
                "p_drop_*": "momentum carried by the tracked drop material",
            },
            "rows": budget_rows,
        }
    if diag_cavity:
        result["cavity_diag"] = {
            "enabled": True,
            "definitions": {
                "d_cav_open": "deepest headspace-connected gas-to-liquid floor in r=1.5 D (D)",
                "d_cav_all": "deepest gas-to-liquid floor in r=1.5 D ignoring connectivity (D)",
                "d_cav_gap": "d_cav_all - d_cav_open, depth lost to closure (D)",
                "d_drop": "deepest depth reached by drop-material markers still in liquid (D)",
                "d_drop_any": "deepest depth reached by any drop-material marker (D)",
                "d_ceil": "deepest liquid cell sitting directly above cavity gas (D)",
                "p_cav": "d_cav_open / d_cav_all, vented depth fraction",
                "cav_vent_ratio": "fraction of cavity gas cells connected to the headspace",
                "cav_pockets": "number of connected cavity-gas pockets below the initial surface",
                "u_back": "maximum upward liquid velocity below the initial surface (m/s)",
            },
            "summary": _diag_summary(cavity_rows, history),
            "rows": cavity_rows,
        }
    if cav_closure:
        result["cav_closure"] = {
            "enabled": True,
            "definitions": {
                "z": "sub-cell z position of the deepest open gas-liquid floor (axis 0 = up)",
                "x": "sub-cell x position of the deepest open gas-liquid floor",
                "y": "sub-cell y position of the deepest open gas-liquid floor",
                "depth": "depth of the closure point below the initial surface (D)",
                "t_close": "closure time (ms after contact): first step after the depth peak where the closure point starts moving up",
                "U_close": "closure velocity (m/s): vertical speed of the closure point at t_close",
            },
            "summary": _closure_summary(closure_rows, history, tracker.contact_step, sc),
            "rows": closure_rows,
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
    parser.add_argument(
        "--gas-nu-ratio",
        type=float,
        default=1.0,
        help="gas/liquid kinematic-viscosity ratio for GAS-cell collision "
        "(tau_g = 0.5 + ratio*(tau_l - 0.5); default 1.0 = legacy single tau)",
    )
    parser.add_argument(
        "--diag-cavity",
        action="store_true",
        help="record the cavity-closure / drop-penetration diagnosis "
        "(read only; does not change the trajectory)",
    )
    parser.add_argument(
        "--cav-closure",
        action="store_true",
        help="record the crater-closure point trajectory (deepest open "
        "gas-liquid floor position and velocity; read only, does not "
        "change the trajectory)",
    )
    parser.add_argument(
        "--diag-budget",
        action="store_true",
        help="record the momentum/energy budget ledger of the impact "
        "(read only; does not change the trajectory)",
    )
    parser.add_argument("--gap", type=float, default=0.4)
    parser.add_argument(
        "--momconv",
        default=None,
        choices=list(MOMCONV_MODES),
        help="momentum-carrying I->G conversion (DE-X-K3-MOMCONV): "
        "off (default, legacy bit-identical), inherit or redistribute",
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
        args.diag_cavity,
        args.gas_nu_ratio,
        args.diag_budget,
        args.momconv,
        args.cav_closure,
    )
    for row in result["history"]:
        for key, value in row.items():
            if not math.isfinite(value):
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
