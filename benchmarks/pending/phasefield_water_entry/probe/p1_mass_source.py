"""P1 probe — attribution of the wall/gravity phase-field mass drift (blocker B2).

Context
-------
P0 (``p0_hydrostatic.py``) found a *one-signed* drift in a closed walled tank:
the interface creeps upward and the phase-field integral grows by ~1.5e-3 in
6000 steps.  That is a first-order error for any "free-surface height" readout,
so S2 (wedge entry) cannot be gated until it is attributed or bounded.

The drift can only come from three places, and this probe switches them
independently:

======================  ==========================================
lever                   what it isolates
======================  ==========================================
gravity on / off        interface motion itself (no motion => no drift
                        if the source is a *moving* ghost)
walls / periodic        solid-boundary phase flux at all
volume x2 (same wall)   whether the drift is a wall flux (absolute,
                        constant in mass) or a bulk effect (relative)
interface height        symmetry / offset dependence
wall_phi_policy         ``freeze`` (default, ghost pinned to the value
                        seeded at t=0) vs ``neumann`` (ghost refreshed to
                        the fluid-neighbour mean every step = zero normal
                        gradient, neutral wetting)
======================  ==========================================

All arms use the verified Laplace parameter set (nu=0.1, sigma=0.01, W=4,
M=0.05) and the same tank as P0 so the numbers are directly comparable.

This is a prerequisite probe, not a benchmark and not a validation claim.

Run
---
    PYTHONPATH=src python benchmarks/pending/phasefield_water_entry/probe/p1_mass_source.py \
        --out /tmp/p1_mass_source.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorlbm import cac_lbm as m


def wall_mask(shape: tuple[int, int, int]) -> torch.Tensor:
    w = torch.zeros(shape, dtype=torch.bool)
    w[0, :, :] = w[-1, :, :] = True
    w[:, 0, :] = w[:, -1, :] = True
    w[:, :, 0] = w[:, :, -1] = True
    return w


def interface_height(phi_row: torch.Tensor) -> float:
    for y in range(phi_row.numel() - 1):
        a, b = float(phi_row[y]), float(phi_row[y + 1])
        if (a - 0.5) * (b - 0.5) <= 0.0 and a != b:
            return y + (0.5 - a) / (b - a)
    return float("nan")


def run_arm(
    *,
    label: str,
    shape: tuple[int, int, int],
    walls: bool,
    axes: str,
    y_walls: tuple[int, ...] | None = None,
    g: float,
    y0: float,
    width: float,
    policy: str,
    n_steps: int,
    sample_every: int,
) -> dict:
    if walls:
        wall = wall_mask(shape)
    elif y_walls is not None:
        wall = torch.zeros(shape, dtype=torch.bool)
        for idx in y_walls:
            wall[:, idx, :] = True
    else:
        wall = None
    sim = m.CACSim(
        shape,
        rho_l=1.0,
        rho_g=0.01,
        mu_l=0.1,
        mu_g=0.001,
        sigma=0.01,
        width=width,
        mobility=0.05,
        p0=0.01,
        gravity=(0.0, -g, 0.0),
        gravity_mode="rho",
        axes=axes,
        wall_mask=wall,
        wall_phi_policy=policy,
    )
    sim.initialize(m.init_phi_layers(shape, width=width, y0=y0))

    inner = ~wall if wall is not None else torch.ones(shape, dtype=torch.bool)
    phi0 = float(sim.f.sum(dim=0)[inner].double().sum())
    h0 = interface_height(sim.f.sum(dim=0)[shape[0] // 2, :, shape[2] // 2])
    t0 = time.time()
    samples = []
    for step in range(1, n_steps + 1):
        sim.step()
        if step % sample_every == 0 or step == n_steps:
            d = m.diagnostics(sim)
            phi = sim.f.sum(dim=0)
            h = interface_height(phi[shape[0] // 2, :, shape[2] // 2])
            samples.append(
                {
                    "step": step,
                    "phi_integral_drift_rel": (d["phi_integral_fluid"] - phi0) / phi0,
                    "interface_drift": h - h0,
                    "u_max": d["u_max_fluid"],
                    "phi_min": d["phi_min_fluid"],
                    "phi_max": d["phi_max_fluid"],
                }
            )
    drift = samples[-1]["phi_integral_drift_rel"]
    return {
        "label": label,
        "shape": list(shape),
        "walls": walls,
        "y_walls": list(y_walls) if y_walls is not None else None,
        "axes": axes,
        "g": g,
        "y0": y0,
        "width": width,
        "wall_phi_policy": policy,
        "n_steps": n_steps,
        "cells": int(torch.tensor(shape).prod().item()),
        "phi_integral_drift_rel": drift,
        "phi_drift_per_1k_steps": drift / (n_steps / 1000.0),
        "interface_drift": samples[-1]["interface_drift"],
        "u_max": samples[-1]["u_max"],
        "nan": bool(samples[-1]["phi_min"] != samples[-1]["phi_min"]),
        "wall_time_s": time.time() - t0,
        "samples": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/p1_mass_source.json"))
    ap.add_argument("--n-steps", type=int, default=6000)
    ap.add_argument("--sample-every", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    base = {}
    arms = [
        # --- attribution levers (all 6000 steps) -------------------------
        dict(
            label="A_walls_g1e-4",
            shape=(8, 96, 8),
            walls=True,
            axes="www",
            g=1e-4,
            y0=48.0,
            width=4.0,
            policy="freeze",
            n_steps=args.n_steps,
            sample_every=args.sample_every,
        ),
        dict(
            label="B_walls_g0",
            n_steps=args.n_steps,
            sample_every=args.sample_every,
            shape=(8, 96, 8),
            walls=True,
            axes="www",
            g=0.0,
            y0=48.0,
            width=4.0,
            policy="freeze",
        ),
        dict(
            label="C_sidewalls_off_g1e-4",
            n_steps=args.n_steps,
            sample_every=args.sample_every,
            shape=(8, 96, 8),
            walls=False,
            axes="pwp",
            g=1e-4,
            y0=48.0,
            width=4.0,
            policy="freeze",
            y_walls=(0, -1),
        ),
        dict(
            label="D_walls_double_height",
            n_steps=args.n_steps,
            sample_every=args.sample_every,
            shape=(8, 192, 8),
            walls=True,
            axes="www",
            g=1e-4,
            y0=96.0,
            width=4.0,
            policy="freeze",
        ),
        dict(
            label="E_walls_low_interface",
            n_steps=args.n_steps,
            sample_every=args.sample_every,
            shape=(8, 96, 8),
            walls=True,
            axes="www",
            g=1e-4,
            y0=32.0,
            width=4.0,
            policy="freeze",
        ),
        dict(
            label="F_walls_W6",
            n_steps=args.n_steps,
            sample_every=args.sample_every,
            shape=(8, 96, 8),
            walls=True,
            axes="www",
            g=1e-4,
            y0=48.0,
            width=6.0,
            policy="freeze",
        ),
        # --- candidate fix ----------------------------------------------
        dict(
            label="G_walls_g1e-4_neumann",
            n_steps=args.n_steps,
            sample_every=args.sample_every,
            shape=(8, 96, 8),
            walls=True,
            axes="www",
            g=1e-4,
            y0=48.0,
            width=4.0,
            policy="neumann",
        ),
        dict(
            label="H_walls_g0_neumann",
            n_steps=args.n_steps,
            sample_every=args.sample_every,
            shape=(8, 96, 8),
            walls=True,
            axes="www",
            g=0.0,
            y0=48.0,
            width=4.0,
            policy="neumann",
        ),
        # --- saturation test (transient vs steady source), last -----------
        dict(
            label="I_long_freeze",
            shape=(8, 96, 8),
            walls=True,
            axes="www",
            g=1e-4,
            y0=48.0,
            width=4.0,
            policy="freeze",
            n_steps=24000,
            sample_every=3000,
        ),
        dict(
            label="J_long_double_height",
            shape=(8, 192, 8),
            walls=True,
            axes="www",
            g=1e-4,
            y0=96.0,
            width=4.0,
            policy="freeze",
            n_steps=24000,
            sample_every=3000,
        ),
    ]

    out = {"probe": "p1_mass_source", "model": "cac_lbm.CACSim", "arms": []}
    for arm in arms:
        label = arm.pop("label")
        print(f"[p1] {label} ...", flush=True)
        res = run_arm(label=label, **base, **arm)
        print(
            f"[p1]   phi {res['phi_integral_drift_rel']:+.3e} "
            f"({res['phi_drift_per_1k_steps']:+.2e}/1k) | iface {res['interface_drift']:+.3f} cell | "
            f"max|u| {res['u_max']:.2e} | {res['wall_time_s']:.0f}s",
            flush=True,
        )
        out["arms"].append(res)

    # verdict helpers (NaN-safe: a diverged arm reports nan, never a verdict)
    def get(label: str) -> dict:
        return next(a for a in out["arms"] if a["label"] == label)

    def finite(*vals: float) -> bool:
        return all(v == v and abs(v) != float("inf") for v in vals)

    a, b, c = get("A_walls_g1e-4"), get("B_walls_g0"), get("C_sidewalls_off_g1e-4")
    d = get("D_walls_double_height")
    g_arm = get("G_walls_g1e-4_neumann")
    i_long, j_long = get("I_long_freeze"), get("J_long_double_height")

    def ratio(x: float, y: float) -> float | None:
        if not finite(x, y) or y == 0.0:
            return None
        return x / y

    out["attribution"] = {
        "A_walls_g1e-4_per_1k": a["phi_drift_per_1k_steps"],
        "B_walls_g0_per_1k": b["phi_drift_per_1k_steps"],
        "gravity_removal_changes_drift_by_factor": ratio(
            b["phi_drift_per_1k_steps"], a["phi_drift_per_1k_steps"]
        ),
        "C_side_walls_off_per_1k": c["phi_drift_per_1k_steps"],
        "D_double_height_per_1k": d["phi_drift_per_1k_steps"],
        "double_height_reduction_factor": ratio(
            a["phi_drift_per_1k_steps"], d["phi_drift_per_1k_steps"]
        ),
        "G_neumann_per_1k": g_arm["phi_drift_per_1k_steps"],
        "neumann_reduction_factor": ratio(
            a["phi_drift_per_1k_steps"], g_arm["phi_drift_per_1k_steps"]
        ),
        "saturation": {
            "freeze_last_1k": i_long["phi_drift_per_1k_steps"],
            "double_height_last_1k": j_long["phi_drift_per_1k_steps"],
            "freeze_first_half_rate": None,
            "freeze_second_half_rate": None,
        },
    }
    # saturation test: compare the drift accrued in the first vs the last half
    for key, arm in (("freeze", i_long), ("double_height", j_long)):
        samples = arm["samples"]
        mid = len(samples) // 2
        if len(samples) >= 4:
            half_early = (
                samples[mid]["phi_integral_drift_rel"] - samples[0]["phi_integral_drift_rel"]
            )
            half_late = (
                samples[-1]["phi_integral_drift_rel"] - samples[mid]["phi_integral_drift_rel"]
            )
            out["attribution"]["saturation"][f"{key}_rate_shrinks_by"] = ratio(
                half_early, half_late
            )
            out["attribution"]["saturation"][f"{key}_total"] = samples[-1]["phi_integral_drift_rel"]
    out["notes"] = (
        "Prerequisite probe: attributes and (optionally) bounds the wall/gravity phase-field "
        "mass drift. Not a benchmark; no external reference is compared."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[p1] wrote {args.out}")
    print(json.dumps(out["attribution"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
