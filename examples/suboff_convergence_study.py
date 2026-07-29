"""Long-time drag convergence study — 100k steps, full convergence curve.

Runs D3Q27 CUMULANT+Smagorinsky on SUBOFF bare_hull 160³ for
100,000 time steps, recording every drag value for post-hoc
convergence analysis.  Produces a per-step CSV and a summary
JSON with windowed statistics.

Usage:
    PYTHONPATH=src python examples/suboff_convergence_study.py [--device sdaa:0] [--steps 100000]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.d3q27 import equilibrium27, correct_mass27, stream27
from tensorlbm.cumulant_smag import collide_cumulant_smag_d3q27
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.wall_model import wall_function_d3q27
from tensorlbm.boundaries_d3q27 import far_field_bc_27


def run(
    nx: int = 160,
    ny: int = 64,
    nz: int = 64,
    hull_length: float = 64.0,
    u_in: float = 0.06,
    re: float = 2e6,
    C_s: float = 0.05,
    n_steps: int = 100_000,
    warmup: int = 5_000,
    device_str: str = "sdaa:0",
    output_dir: str = "/tmp/suboff_100k",
) -> int:
    """Run convergence study, return exit code."""
    device = torch.device(device_str)
    torch.sdaa.set_device(device)

    nu = u_in * hull_length / re
    tau = 3.0 * nu + 0.5
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "drag_series.csv"
    summary_path = out / "summary.json"
    ckpt_path = out / "checkpoint.pt"

    print(f"D3Q27 CUMULANT+Smag(Cs={C_s}) bare_hull {nx}³ {n_steps} steps")
    print(f"tau={tau:.6f} nu={nu:.2e} warmup={warmup}")
    print(f"Output: {out}")
    print()

    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    solid, _ = build_suboff_mask(
        hull_type=SuboffHullType.BARE_HULL,
        nx=nx, ny=ny, nz=nz, cx=cx, cy=cy, cz=cz,
        length=hull_length, device=device,
    )
    S = _voxel_wetted_area(solid, 1.0)
    dpS = 0.5 * 1.0 * u_in ** 2 * S

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0
    f = equilibrium27(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    initial_mass = float(rho0.sum().item())

    # Accumulators for running statistics
    drag_fric: list[float] = []
    drag_pres: list[float] = []
    start_step = 1

    # Resume from checkpoint if exists
    if ckpt_path.exists():
        print("Resuming from checkpoint...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        f = ckpt["f"].to(device)
        start_step = ckpt["step"] + 1
        drag_fric = ckpt.get("drag_fric", [])
        drag_pres = ckpt.get("drag_pres", [])
        print(f"  resumed at step {start_step}, {len(drag_fric)} drag records")

    # CSV header
    if start_step == 1:
        csv_path.write_text("step,Ct_fric,Ct_pres,Ct_total\n")

    t0 = time.time()
    report_interval = max(1000, n_steps // 20)  # ~20 reports
    checkpoint_interval = 5000

    print(f"{'step':>7} {'Ct_fric':>9} {'Ct_pres':>9} {'Ct_avg':>9} {'Ct_std':>9} {'Δ5%':>9} {'Conv?':<6}")
    print("-" * 75)

    with open(csv_path, "a") as csv_file:
        for step in range(start_step, n_steps + 1):
            f = collide_cumulant_smag_d3q27(f, tau, C_s=C_s)
            f = stream27(f)
            f, df, dp = wall_function_d3q27(f, solid, nu, y_val=0.5)
            f = far_field_bc_27(f, u_in=u_in)
            if step % 100 == 0:
                f = correct_mass27(f, initial_mass)

            if step > warmup and math.isfinite(df):
                drag_fric.append(df)
                drag_pres.append(dp)
                cf = df / dpS
                cp = dp / dpS
                csv_file.write(f"{step},{cf:.8f},{cp:.8f},{cf+cp:.8f}\n")

            if not torch.isfinite(f).all():
                print(f"\nDIVERGED at step {step}")
                break

            # Periodic report
            if step % report_interval == 0 and len(drag_fric) > 10:
                n = len(drag_fric)
                window = max(500, n // 5)
                early_f = drag_fric[:-window]
                early_p = drag_pres[:-window]
                late_f = drag_fric[-window:]
                late_p = drag_pres[-window:]

                avg_f = sum(drag_fric) / n / dpS
                avg_p = sum(drag_pres) / n / dpS
                avg_t = avg_f + avg_p

                early_t = (sum(early_f) + sum(early_p)) / max(len(early_f), 1) / dpS
                late_t = (sum(late_f) + sum(late_p)) / max(len(late_f), 1) / dpS
                change = (late_t - early_t) / max(abs(early_t), 1e-12)

                # Std of last window
                vals = [(drag_fric[i] + drag_pres[i]) / dpS for i in range(-window, 0)]
                std_t = (sum((v - late_t) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5

                elapsed = time.time() - t0
                rate = step / max(elapsed, 1)
                eta = (n_steps - step) / max(rate, 1e-6)
                conv = "✓" if abs(change) < 0.01 and std_t < 0.001 else ""

                print(f"{step:7d} {avg_f:9.5f} {avg_p:9.5f} {avg_t:9.5f} {std_t:9.5f} {change:9.5f} {conv:<6} "
                      f"({elapsed:.0f}s ETA {eta:.0f}s)")

            # Checkpoint
            if step % checkpoint_interval == 0:
                torch.save({
                    "step": step,
                    "f": f.cpu(),
                    "drag_fric": drag_fric,
                    "drag_pres": drag_pres,
                }, ckpt_path.with_suffix(".tmp"))
                ckpt_path.with_suffix(".tmp").rename(ckpt_path)
                print(f"  [checkpoint saved at step {step}]")

    # Final summary
    n = len(drag_fric)
    if n > 0:
        win = max(1, n // 5)
        avg_f = sum(drag_fric) / n
        avg_p = sum(drag_pres) / n
        late_f = sum(drag_fric[-win:]) / win
        late_p = sum(drag_pres[-win:]) / win
        late_t = (late_f + late_p) / dpS
        vals = [(drag_fric[i] + drag_pres[i]) / dpS for i in range(-win, 0)]
        std_t = (sum((v - late_t) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5

        summary = {
            "grid": f"{nx}x{ny}x{nz}",
            "lattice": "D3Q27",
            "collision": "CUMULANT+Smag",
            "C_s": C_s,
            "Re": re,
            "tau": tau,
            "nu": nu,
            "steps_requested": n_steps,
            "steps_completed": step,
            "warmup": warmup,
            "drag_samples": n,
            "Ct_fric_avg": avg_f / dpS,
            "Ct_pres_avg": avg_p / dpS,
            "Ct_total_avg": (avg_f + avg_p) / dpS,
            "Ct_total_last_window": late_t,
            "Ct_total_last_std": std_t,
            "convergence_5pct_change": (late_t - (sum(drag_fric[:-win]) + sum(drag_pres[:-win])) / max(n - win, 1) / dpS),
            "elapsed_s": time.time() - t0,
            "diverged": not bool(torch.isfinite(f).all().item()),
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary: {summary_path}")

    return 0 if torch.isfinite(f).all() else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="sdaa:0")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--nx", type=int, default=160)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--hull-length", type=float, default=64.0)
    p.add_argument("--cs", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--output", default="/tmp/suboff_100k")
    args = p.parse_args()
    sys.exit(run(
        nx=args.nx, ny=args.ny, nz=args.nz,
        hull_length=args.hull_length,
        C_s=args.cs, n_steps=args.steps, warmup=args.warmup,
        device_str=args.device, output_dir=args.output,
    ))
