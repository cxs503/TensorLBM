"""Independent constant-atmosphere D3Q19 check of the water-side interface evolution."""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

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


def _read_iface_tau_dose(base_dir=None):
    """Dose channel for the shared SDAA lane (wrapper passes no new env).

    Priority: env WE_IFACE_DOSE > phase_tau_dose.json next to this module
    > 0.0 (off).  The JSON file is tarred with the tree by the lane wrapper,
    so per-arm doses are set by writing the file before each arm run.
    """
    env = os.environ.get("WE_IFACE_DOSE")
    if env not in (None, ""):
        return float(env)
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    cfg = base / "phase_tau_dose.json"
    if cfg.exists():
        return float(json.loads(cfg.read_text()).get("iface_tau_dose", 0.0))
    return 0.0


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
    gas_nu_ratio=1.0,
    iface_tau_dose=None,
):
    if iface_tau_dose is None:
        iface_tau_dose = _read_iface_tau_dose()
    iface_tau_dose = float(iface_tau_dose)
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

    silhouette_samples = []
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
        elif tracker.contact_step == step:
            impact_speed = last_speed
        if step and step % 50 == 0:
            print(
                f"step {step}, t={row['time_ms']:.4f} ms, contact={tracker.contact_step}",
                flush=True,
            )

        row["depth_over_D"] = open_depth(field, level - 1, 1.5 * n) / n
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
                    iface_tau_dose=iface_tau_dose,
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
        "iface_tau_dose": iface_tau_dose,
        "release_speed_m_s": release_lattice * sc["dx_m"] / sc["dt_s"],
        "prescribed_release_speed_m_s": release,
        "impact_speed_m_s": impact_speed,
        "side_view_measurement": silhouette_summary,
        "history": history,
    }
    if ledger:
        result["mass_ledger_summary"] = _summarize_ledger(runtime_ledger)
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
        "--iface-tau-dose",
        type=float,
        default=None,
        help="fill-weighted interface relaxation dose: INTERFACE cells relax "
        "with tau*(1+dose*(1-fill)); default None = read WE_IFACE_DOSE env or "
        "phase_tau_dose.json, else 0.0 (off)",
    )
    parser.add_argument("--gap", type=float, default=0.4)
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
        args.gas_nu_ratio,
        args.iface_tau_dose,
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
