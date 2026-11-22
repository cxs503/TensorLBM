"""CH4 water-entry probe (DE-X-PF1-CH4, wave W1).

Runs the Cahn-Hilliard fourth-order phase-field LBE (He-Shan-Doolen
proportional-control single distribution + D3Q19 fluid with the
chemical-potential force) on the **same grid** as the in-repo free-surface
baseline (``free_surface_probe``), so the peak depth can be compared directly
against the ctrl / experiment anchors.

The probe has the same ``probe(...)`` signature as ``free_surface_probe.probe``
so the shared SDAA wrapper can drive it unchanged (via ``--probe-module``).
"""

from __future__ import annotations

import json
import math
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
from tensorlbm.ch4_phasefield import (
    _c_on,
    ch4_coupled_step,
    ch4_init,
)


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
    # CH4-specific parameters (defaults tuned for the Santini case 3 at n=8)
    ch4_w: float = 3.0,
    ch4_A: float = 1.63e-4,
    ch4_kappa: float | None = None,
    ch4_M: float = 0.1,
    ch4_tau_g: float = 0.8,
    ch4_force_scale: float = 0.1,
    ch4_rho_g: float = 1.0,
    ch4_rho_l: float = 1000.0,
    ch4_rho0: float = 1.0,
):
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
    # CH order parameter: +1 liquid, -1 gas
    phi = (2.0 * fill - 1.0).to(torch.float32)
    moving = z > level + 0.5 * gap
    release_lattice = (
        math.sqrt(speed**2 - 2 * sc["g"] * gap)
        if release is None
        else release * sc["dt_s"] / sc["dx_m"]
    )
    velocity = -release_lattice * (moving & (fill > 0.5))
    if ch4_kappa is None:
        ch4_kappa = ch4_A * ch4_w * ch4_w
    f, g = ch4_init(
        phi,
        ch4_A,
        ch4_kappa,
        ch4_M,
        ch4_rho_g,
        ch4_rho_l,
        ch4_rho0,
        ux=torch.zeros_like(phi),
        uy=torch.zeros_like(phi),
        uz=velocity.to(torch.float32),
    )
    vol0 = float(((phi + 1) / 2).sum())
    tau_f = 0.5 + 3 * sc["mu_l"]
    gz = -sc["g"]
    c = _c_on(phi.device)
    tracker = DropTracker()
    last_speed = None
    impact_speed = None
    reason = "step_limit"
    history = []
    for step in range(limit + 1):
        rho_f = f.sum(0)
        mom_x = (f * c[:, 0].view(19, 1, 1, 1)).sum(0)
        mom_y = (f * c[:, 1].view(19, 1, 1, 1)).sum(0)
        mom_z = (f * c[:, 2].view(19, 1, 1, 1)).sum(0)
        ux = mom_x / rho_f.clamp(min=1e-12)
        uy = mom_y / rho_f.clamp(min=1e-12)
        uz = mom_z / rho_f.clamp(min=1e-12)
        mach = float((3 * (ux**2 + uy**2 + uz**2)).sqrt().max())
        drift = abs(float(((phi + 1) / 2).sum()) / vol0 - 1)
        row = {
            "step": step,
            "time_ms": 1000 * step * sc["dt_s"],
            "volume_drift": drift,
            "mach_max": mach,
        }
        history.append(row)
        if not all(math.isfinite(v) for v in row.values()):
            reason = "nonfinite"
            break
        if mach > case["gates"]["maximum_lattice_mach"] or drift > case["gates"]["maximum_liquid_volume_relative_drift"]:
            reason = "health_limit"
            break
        field = ((phi + 1) / 2)[1:-1].numpy()
        try:
            detached = tracker.update(field, step, level - 1)
        except ValueError as error:
            reason = str(error)
            break
        if detached is not None:
            last_speed = float(-uz[1:-1][detached].mean()) * sc["dx_m"] / sc["dt_s"]
        elif tracker.contact_step == step:
            impact_speed = last_speed
        if step and step % 50 == 0:
            print(
                f"step {step}, t={row['time_ms']:.4f} ms, contact={tracker.contact_step}",
                flush=True,
            )
        row["depth_over_D"] = open_depth(field, level - 1, 1.5 * n) / n
        if tracker.contact_step is not None and (step - tracker.contact_step) * sc["dt_s"] >= 0.024:
            reason = "window_complete"
            break
        if step < limit:
            f, g, phi = ch4_coupled_step(
                f,
                g,
                phi,
                ch4_A,
                ch4_kappa,
                ch4_M,
                ch4_tau_g,
                tau_f,
                gz,
                ch4_force_scale,
                ch4_rho_g,
                ch4_rho_l,
                ch4_rho0,
            )
    result = {
        "solver": "ch4_phasefield/D3Q19, Cahn-Hilliard fourth-order, chemical-potential force",
        "cells_per_D": n,
        "case_id": "pilot" if pilot else "santini_case3",
        "span_over_D": span,
        "pool_depth_over_D": pool_depth,
        "lattice_speed": speed,
        "scales": sc,
        "stop_reason": reason,
        "contact_step": tracker.contact_step,
        "gap_over_D": gap_fraction,
        "u_limit": u_limit,
        "release_speed_m_s": release_lattice * sc["dx_m"] / sc["dt_s"],
        "prescribed_release_speed_m_s": release,
        "impact_speed_m_s": impact_speed,
        "ch4": {
            "w": ch4_w,
            "A": ch4_A,
            "kappa": ch4_kappa,
            "M": ch4_M,
            "tau_g": ch4_tau_g,
            "force_scale": ch4_force_scale,
            "rho_g": ch4_rho_g,
            "rho_l": ch4_rho_l,
            "rho0": ch4_rho0,
        },
        "history": history,
    }
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2100)
    parser.add_argument("--speed", type=float, default=0.06)
    parser.add_argument("--gap", type=float, default=0.18)
    parser.add_argument("--pd", type=float, default=3.0)
    parser.add_argument("--rs", type=float, default=1.53)
    parser.add_argument("--ul", type=float, default=0.15)
    parser.add_argument("--force-scale", type=float, default=0.1)
    parser.add_argument("--rho-l", type=float, default=1000.0)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("ch4_we_result.json"))
    a = parser.parse_args()
    res = probe(
        n=a.n,
        speed=a.speed,
        limit=a.steps,
        gap_fraction=a.gap,
        capacity=True,
        nearest=False,
        pilot=False,
        span=4.0,
        pool_depth=a.pd,
        simple=True,
        release=a.rs,
        planes_path=None,
        silhouette_path=None,
        ledger=False,
        diag_path=None,
        u_limit=a.ul,
        ch4_force_scale=a.force_scale,
        ch4_rho_l=a.rho_l,
    )
    a.out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    h = res.get("history") or []
    pk = max(h, key=lambda r_: r_.get("depth_over_D", -1)) if h else {}
    print(
        "CH4_WE stop=%s peak=%.4fD@%sms contact=%s"
        % (res.get("stop_reason"), pk.get("depth_over_D", -1), pk.get("time_ms", -1), res.get("contact_step"))
    )
