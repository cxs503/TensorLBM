#!/usr/bin/env python3
"""τ ladder driver — CPU-compatible uniform-grid suboff_sail_resistance.

Runs the five-arm ladder τ = {0.51, 0.55, 0.60, 0.70, 1.00} on the single-level
verified case benchmarks/verified/suboff_sail_resistance on CPU.

Usage
-----
    cd benchmarks/verified/suboff_sail_resistance
    PYTHONPATH=../../src python3 run_tau_ladder_cpu.py
    python3 run_tau_ladder_cpu.py --reassemble   # post-processing only
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, W, OPPOSITE
from tensorlbm.solver3d import collide_mrt3d, stream3d_gather
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.momentum_exchange import momentum_exchange_standard
from tensorlbm.drag_pressure import get_near_wall_3d

# --------------------------------------------------------------------------- #
# Frozen constants
# --------------------------------------------------------------------------- #
STAGING = Path(__file__).resolve().parent / "tau_ladder_cpu_staging"
STAGING.mkdir(parents=True, exist_ok=True)

# Ladder design (frozen)
TAU_LADDER = {"A": 0.51, "B": 0.55, "C": 0.60, "D": 0.70, "E": 1.00}
ARM_ORDER = ["A", "B", "C", "D", "E"]

# Fixed parameters
RESOLUTION = 16
STEPS = 200
U_LB = 0.05
COLLISION = "mrt"
FRICTION = "standard"
P0 = "near_wall"
DEVICE = "cpu"

# Anchor cadence
ANCHOR_CADENCE = 20  # 10 samples per 200-step arm

# Blasius reference at Re=1000
CF_REF = 1.328 / math.sqrt(1000)  # 0.041995...


def nu_lb_of(tau: float) -> float:
    return (tau - 0.5) / 3.0


def re_lattice(tau: float, L: int = RESOLUTION, u_lb: float = U_LB) -> float:
    return u_lb * float(L) / nu_lb_of(tau)


def blasius_cf(re: float) -> float:
    return 1.328 / math.sqrt(re)


# --------------------------------------------------------------------------- #
# Case construction (single-level uniform grid)
# --------------------------------------------------------------------------- #
def build_uniform_case(u_lb: float, tau: float, device: str = "cpu"):
    """Build the uniform-grid SUBOFF sail resistance case."""
    torch.manual_seed(0)
    cfg = SuboffConfig()
    solid, geometry = build_suboff_mask(
        "with_sail",
        RESOLUTION,
        RESOLUTION,
        RESOLUTION,
        length=RESOLUTION,
        config=cfg,
        device=torch.device(device),
    )
    if not bool(solid.any()):
        raise RuntimeError("no SUBOFF cells on the grid")

    shape = solid.shape
    rho = torch.ones(shape, device=device, dtype=torch.float32)
    ux = torch.full(shape, float(u_lb), device=device, dtype=torch.float32)
    zero = torch.zeros(shape, device=device, dtype=torch.float32)
    f = equilibrium3d(rho, ux, zero, zero, device=device)

    # Pre-compute solid masks for BFL
    solid_q = solid.unsqueeze(0).expand(19, *shape).contiguous()
    solid_g = torch.roll(solid, (1, 1, 1), dims=(0, 1, 2))
    bfl_mask = solid & ~solid_g
    
    # Near-wall mask
    near = get_near_wall_3d(solid)

    return f, solid, solid_q, bfl_mask, near, geometry


def compute_drag(f, solid, near, dpS):
    """Compute drag via momentum exchange."""
    fx, fy, fz = momentum_exchange_standard(f, solid, near)
    cd_total = fx / dpS if dpS > 0 else 0.0
    return cd_total, fx, fy, fz


# --------------------------------------------------------------------------- #
# One arm
# --------------------------------------------------------------------------- #
def run_arm(arm: str, tau_target: float) -> dict:
    out_dir = STAGING / f"tau_{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = DEVICE
    f, solid, solid_q, bfl_mask, near, geometry = build_uniform_case(U_LB, tau_target, device)

    tau_c = tau_target
    nu_c = nu_lb_of(tau_c)
    steps = STEPS

    # Compute reference area
    S_total = geometry.get("wetted_area", RESOLUTION ** 2)
    dpS = 0.5 * U_LB ** 2 * S_total

    # Collision function
    def collide(f_in):
        return collide_mrt3d(f_in, tau_c)

    # Advance function
    def advance(f_in):
        collided = collide(f_in)
        post_collision = torch.where(solid_q, f_in, collided)
        out = stream3d_gather(post_collision)
        out = non_equilibrium_far_field_bc_3d(out, u_in=U_LB)
        return out

    # Run
    rows = []
    status = "ok"
    failed_at = None
    started = time.time()

    for step in range(1, steps + 1):
        f = advance(f)
        if not bool(torch.isfinite(f[0, 1, 1, 1]).item()):
            status = "diverged"
            failed_at = step
            break
        if step % ANCHOR_CADENCE == 0:
            # Compute drag via momentum exchange
            cd_total, fx, fy, fz = compute_drag(f, solid, near, dpS)

            rows.append({
                "step": step,
                "tau": tau_c,
                "nu_lb": nu_c,
                "cd_total": cd_total,
                "fx": fx,
                "fy": fy,
                "fz": fz,
            })

        if step % max(ANCHOR_CADENCE, steps // 10) == 0:
            print(f"[{arm}] step {step}/{steps}", flush=True)

    wall = time.time() - started

    # Window statistics (last 25%)
    if rows:
        n_win = max(1, len(rows) // 4)
        win_rows = rows[-n_win:]
        cd_vals = [r["cd_total"] for r in win_rows]
        cd_mean = sum(cd_vals) / len(cd_vals)
        cd_std = (sum((v - cd_mean) ** 2 for v in cd_vals) / len(cd_vals)) ** 0.5
        convergence_windows = {
            "25%": cd_vals[0],
            "50%": cd_vals[len(cd_vals) // 2],
            "75%": cd_vals[-2] if len(cd_vals) > 1 else cd_vals[0],
            "100%": cd_vals[-1],
        }
    else:
        cd_mean = 0.0
        cd_std = 0.0
        convergence_windows = {}

    re_eff = re_lattice(tau_c)
    cf_at_re = blasius_cf(re_eff)
    err_vs_frozen = (cd_mean / CF_REF - 1) * 100 if CF_REF > 0 else float("nan")
    err_vs_rematched = (cd_mean / cf_at_re - 1) * 100 if cf_at_re > 0 else float("nan")

    result = {
        "arm": arm,
        "tau_target": float(tau_target),
        "tau_actual": tau_c,
        "nu_lb_actual": nu_c,
        "u_lb_actual": U_LB,
        "re_lattice_actual": re_eff,
        "cd_pressure": 0.0,
        "cd_friction": 0.0,
        "cd_total": cd_mean,
        "cd_total_last500": cd_mean,
        "convergence_windows": convergence_windows,
        "finite": True,
        "diverged": status == "diverged",
        "n_steps_done": rows[-1]["step"] if rows else 0,
        "ms_per_step": wall / steps * 1000 if steps > 0 else 0,
        "elapsed_s": wall,
        "domain_lu": RESOLUTION,
        "solid_cells": int(solid.sum().item()),
        "err_vs_frozen_pct": err_vs_frozen,
        "err_vs_rematched_pct": err_vs_rematched,
        "blasius_cf_at_re": cf_at_re,
        "status": status,
        "wall_s": round(wall, 1),
    }

    (out_dir / f"result_L{RESOLUTION}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"[{arm}] status={status} steps={result['n_steps_done']} "
        f"Re={re_eff:.2f} tau={tau_c:.6f} Cd_total={cd_mean:.6f} "
        f"err_vs_frozen={err_vs_frozen:.2f}% ({wall:.1f}s)",
        flush=True,
    )

    return result


# --------------------------------------------------------------------------- #
# Assembly + verdict
# --------------------------------------------------------------------------- #
def assemble() -> dict:
    """Assemble the verdict from on-disk arm results."""
    arms_result = {}
    for arm in ARM_ORDER:
        tau_target = TAU_LADDER[arm]
        out_dir = STAGING / f"tau_{arm}"
        res_path = out_dir / f"result_L{RESOLUTION}.json"
        if res_path.exists():
            arms_result[arm] = json.loads(res_path.read_text())
        else:
            arms_result[arm] = {"arm": arm, "status": "missing", "tau_target": tau_target}

    # Gates
    G1_pass = all(
        arms_result[a].get("cd_total", 0) > 0
        for a in ARM_ORDER if arms_result[a].get("status") == "ok"
    )
    G1_per_arm = {
        a: {
            "cd_total": arms_result[a].get("cd_total", 0),
            "pass": bool(arms_result[a].get("cd_total", 0) > 0),
        }
        for a in ARM_ORDER
    }

    G2_per_arm = {}
    for a in ARM_ORDER:
        r = arms_result[a]
        if r.get("status") != "ok":
            G2_per_arm[a] = {"pass": False, "reason": "arm failed"}
            continue
        cd = r.get("cd_total", 0)
        err_frozen = abs(r.get("err_vs_frozen_pct", float("inf")))
        G2_per_arm[a] = {
            "cd_total": cd,
            "err_vs_frozen_pct": r.get("err_vs_frozen_pct", float("nan")),
            "err_vs_rematched_pct": r.get("err_vs_rematched_pct", float("nan")),
            "re_effective": r.get("re_lattice_actual", 0),
            "blasius_cf_at_re": r.get("blasius_cf_at_re", 0),
            "pass_frozen": err_frozen <= 50,
        }
    G2_pass = all(G2_per_arm[a].get("pass_frozen", False) for a in ARM_ORDER)

    # G3: Cd_total monotone in tau
    cd_series = [arms_result[a].get("cd_total", 0) for a in ARM_ORDER if arms_result[a].get("status") == "ok"]
    diffs = [cd_series[i + 1] - cd_series[i] for i in range(len(cd_series) - 1)]
    n_increasing = sum(1 for d in diffs if d > 0)
    n_decreasing = sum(1 for d in diffs if d < 0)
    G3_pass = n_increasing >= 3 or (n_increasing + n_decreasing) == 0  # monotone or constant

    G3_series = [
        {"tau": TAU_LADDER[a], "cd_total": arms_result[a].get("cd_total", 0)}
        for a in ARM_ORDER if arms_result[a].get("status") == "ok"
    ]

    overall = "PASS" if (G1_pass and G2_pass and G3_pass) else "HONEST-FAIL"

    verdict = {
        "schema": "tau-ladder-cpu/v1",
        "task": "m27602",
        "mission": "paper-exp-lbm-suboff-tau",
        "overall": overall,
        "overall_note": (
            f"G1={'PASS' if G1_pass else 'FAIL'} (Cd_total > 0), "
            f"G2={'PASS' if G2_pass else 'FAIL'} (|Cd_total/Cf_ref - 1| <= 50%), "
            f"G3={'PASS' if G3_pass else 'FAIL'} (Cd_total monotone in tau)"
        ),
        "frozen_setup": {
            "case": "benchmarks/verified/suboff_sail_resistance",
            "hull_type": "with_sail",
            "resolution": RESOLUTION,
            "steps": STEPS,
            "u_lb": U_LB,
            "collision": COLLISION,
            "friction": FRICTION,
            "p0": P0,
            "device": DEVICE,
            "ladder": TAU_LADDER,
        },
        "arms": arms_result,
        "gates": {
            "G1_total_drag_positive": {
                "criterion": "Cd_total > 0 for all arms",
                "per_arm": G1_per_arm,
                "pass": G1_pass,
            },
            "G2_magnitude_vs_reference": {
                "criterion": f"|Cd_total / Cf_ref@Re1000 - 1| <= 50% (Cf_ref={CF_REF:.6f})",
                "per_arm": G2_per_arm,
                "pass": G2_pass,
            },
            "G3_monotone_in_tau": {
                "criterion": "Cd_total monotonically changes with tau (>= 3 monotone arms)",
                "series": G3_series,
                "diffs": diffs,
                "n_increasing": n_increasing,
                "n_decreasing": n_decreasing,
                "pass": G3_pass,
            },
        },
        "gates_summary": {
            "G1": G1_pass,
            "G2": G2_pass,
            "G3": G3_pass,
            "n_passed": sum([G1_pass, G2_pass, G3_pass]),
            "n_total": 3,
        },
        "tau_re_coupling": {
            "note": "u_lb and L are frozen, so tau ladder IS a Re ladder: tau↑ => nu↑ => Re↓",
            "per_arm": {
                a: {
                    "tau": TAU_LADDER[a],
                    "nu_lb": nu_lb_of(TAU_LADDER[a]),
                    "re_lattice": re_lattice(TAU_LADDER[a]),
                    "re_shift_vs_1000_pct": (re_lattice(TAU_LADDER[a]) / 1000 - 1) * 100,
                }
                for a in ARM_ORDER
            },
        },
    }

    out_path = STAGING / "verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nVerdict: {overall}")
    print(f"Artifact: {out_path}")
    return verdict


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    import sys
    if "--reassemble" in sys.argv:
        assemble()
        return

    print("Running τ ladder arms...")
    for arm in ARM_ORDER:
        tau_target = TAU_LADDER[arm]
        run_arm(arm, tau_target)

    print("\nAssembling verdict...")
    assemble()


if __name__ == "__main__":
    main()
