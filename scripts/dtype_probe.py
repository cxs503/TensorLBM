#!/usr/bin/env python
"""M-T2·P1 dtype 策略探测：Couette 2D + Taylor-Green 2D 上测 fp32/tf32/bf16 三档。

对照解析解，产出误差-加速对比表 + 曲线图。
判据：仅当对解析解误差 <= 0.1% 且相对 fp32 加速 >= 1.3x 的 dtype 才进推荐清单。

用法:
    python scripts/dtype_probe.py [--device sdaa:0] [--n-tg 64] [--H-couette 40]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── repo bootstrap ──────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

from tensorlbm.d2q9 import C, OPPOSITE, W, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

# ── constants ───────────────────────────────────────────────────────────
CS2 = 1.0 / 3.0

# ── dtype helpers ───────────────────────────────────────────────────────
DTYPE_MAP = {
    "fp32": torch.float32,
    "tf32": torch.float32,  # tf32 uses fp32 storage but enables TF32 matmul
    "bf16": torch.bfloat16,
}


def set_tf32(enabled: bool) -> None:
    """Enable/disable TF32 for matmul/conv on the current backend."""
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = enabled
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = enabled
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if enabled else "highest")


# ── Couette 2D ──────────────────────────────────────────────────────────
def run_couette(
    H: int,
    tau: float,
    U0: float,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    max_steps: int = 8000,
    measure_steps: int = 200,
) -> dict:
    """Run 2D Couette flow and return error + timing."""
    torch.manual_seed(42)
    ny = H + 2
    nx = H
    nu = (tau - 0.5) / 3.0
    H_eff = float(ny - 2)

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    u_wall = torch.zeros((ny, nx), device=device, dtype=dtype)
    u_wall[-1, :] = U0

    yy = torch.arange(ny, device=device, dtype=dtype)
    u0 = torch.clamp(U0 * (yy - 0.5) / H_eff, min=0.0, max=U0).view(ny, 1).expand(ny, nx).clone()
    u0[0, :] = 0.0
    u0[-1, :] = 0.0
    rho = torch.ones((ny, nx), device=device, dtype=dtype)
    f = equilibrium(rho, u0, torch.zeros((ny, nx), device=device, dtype=dtype))

    opp = OPPOSITE.to(device)
    c = C.to(device)
    w = W.to(device)

    def _step(f):
        f_pre = f.clone()
        f = collide_bgk(f, tau)
        f_new = torch.where(wall.unsqueeze(0), f_pre[opp], f)
        rho_w = torch.clamp(f_pre.sum(dim=0), min=1e-12)
        cu = c[:, 0].view(9, 1, 1) * u_wall.unsqueeze(0)
        injection = (2.0 * w.view(9, 1, 1) * rho_w.unsqueeze(0) * cu) / CS2
        f_new = f_new + injection * wall.unsqueeze(0)
        return stream(f_new)

    # Warmup 200 steps
    for _ in range(200):
        f = _step(f)

    # Timed run
    t0 = time.perf_counter()
    for step in range(max_steps):
        f = _step(f)
    elapsed = time.perf_counter() - t0

    # Average profile over last measure_steps
    col = nx // 2
    prof_acc = torch.zeros(ny, device=device, dtype=dtype)
    for _ in range(measure_steps):
        f = _step(f)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, col]
    prof_acc /= measure_steps

    u_num = prof_acc[1 : ny - 1].cpu().to(torch.float64).numpy()
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    u_ana = U0 * (y_phys / H_eff)

    l2_rel = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    max_abs_err = float(np.max(np.abs(u_num - u_ana)))
    mask = u_ana > 0.2 * U0
    max_rel = float(np.max(np.abs(u_num[mask] - u_ana[mask]) / u_ana[mask]) * 100.0) if mask.any() else 0.0

    return {
        "l2_rel_err": l2_rel,
        "max_abs_err": max_abs_err,
        "max_rel_err_pct": max_rel,
        "n_steps": max_steps + measure_steps,
        "elapsed_s": round(elapsed, 3),
        "u_profile": u_num.tolist(),
        "u_analytic": u_ana.tolist(),
    }


# ── Taylor-Green 2D ─────────────────────────────────────────────────────
def run_taylor_green(
    n: int,
    re: float,
    u0: float,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    steps: int = 0,
    record_every: int = 50,
) -> dict:
    """Run 2D Taylor-Green vortex decay and return error + timing."""
    L = float(n)
    k = 2.0 * math.pi / L
    nu = u0 * L / re
    tau = 0.5 + 3.0 * nu
    gamma_e_theory = 4.0 * nu * k * k
    gamma_vel_theory = 2.0 * nu * k * k

    if steps <= 0:
        # Target E/E0 = e^-2 for reliable fit
        steps = max(500, int(math.ceil(2.0 / gamma_e_theory / record_every) * record_every))
        steps = min(steps, 50000)

    y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=dtype),
        torch.arange(n, device=device, dtype=dtype),
        indexing="ij",
    )
    ux0 = -u0 * torch.cos(k * x) * torch.sin(k * y)
    uy0 = +u0 * torch.sin(k * x) * torch.cos(k * y)
    rho = torch.ones_like(ux0)
    f = equilibrium(rho, ux0, uy0)

    def _step(f):
        return collide_bgk(stream(f), tau)

    # Warmup 100 steps
    for _ in range(100):
        f = _step(f)

    # Timed run
    times: list[int] = []
    energies: list[float] = []
    umaxs: list[float] = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        f = _step(f)
        if step % record_every == 0:
            _, uxm, uym = macroscopic(f)
            e = float((0.5 * (uxm * uxm + uym * uym)).mean().item())
            um = float((uxm * uxm + uym * uym).sqrt().max().item())
            times.append(step)
            energies.append(e)
            umaxs.append(um)
    elapsed = time.perf_counter() - t0

    # Fit decay rates
    t = np.asarray(times, dtype=np.float64)
    lnE = np.log(np.asarray(energies, dtype=np.float64))
    a_e, b_e = np.polyfit(t, lnE, 1)
    gamma_e_sim = -a_e
    resid = lnE - (b_e + a_e * t)
    r2 = 1.0 - float(np.sum(resid**2) / np.sum((lnE - lnE.mean()) ** 2))

    lnU = np.log(np.asarray(umaxs, dtype=np.float64))
    a_u, _ = np.polyfit(t, lnU, 1)
    gamma_vel_sim = -a_u

    err_e_pct = (gamma_e_sim - gamma_e_theory) / gamma_e_theory * 100.0
    err_vel_pct = (gamma_vel_sim - gamma_vel_theory) / gamma_vel_theory * 100.0

    return {
        "gamma_e_theory": gamma_e_theory,
        "gamma_e_sim": gamma_e_sim,
        "err_e_pct": err_e_pct,
        "gamma_vel_theory": gamma_vel_theory,
        "gamma_vel_sim": gamma_vel_sim,
        "err_vel_pct": err_vel_pct,
        "r2": r2,
        "n_steps": steps,
        "elapsed_s": round(elapsed, 3),
        "times": times,
        "energies": energies,
        "umaxs": umaxs,
    }


# ── Main probe ──────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="M-T2·P1 dtype strategy probe")
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--n-tg", type=int, default=64)
    ap.add_argument("--H-couette", type=int, default=40)
    ap.add_argument("--re-tg", type=float, default=100)
    ap.add_argument("--u0-tg", type=float, default=0.05)
    ap.add_argument("--u0-couette", type=float, default=0.05)
    ap.add_argument("--tau-couette", type=float, default=0.8)
    ap.add_argument("--out", default=str(REPO / "artifacts" / "dtype_probe.json"))
    ap.add_argument("--plot-dir", default=str(REPO / "artifacts"))
    args = ap.parse_args()

    device = torch.device(args.device)
    dtypes_to_test = ["fp32", "tf32", "bf16"]

    results: dict = {
        "mission": "M-T2·P1",
        "description": "dtype strategy probe: Couette 2D + Taylor-Green 2D vs analytical solution",
        "device": str(device),
        "device_name": torch.sdaa.get_device_name(0) if hasattr(torch, "sdaa") else "unknown",
        "n_tg": args.n_tg,
        "H_couette": args.H_couette,
        "re_tg": args.re_tg,
        "u0_tg": args.u0_tg,
        "u0_couette": args.u0_couette,
        "tau_couette": args.tau_couette,
        "criteria": {
            "max_err_pct": 0.1,
            "min_speedup_vs_fp32": 1.3,
        },
        "couette": {},
        "taylor_green": {},
        "summary": {},
        "recommendations": [],
    }

    # ── Run Couette 2D ──────────────────────────────────────────────────
    print("=" * 60)
    print("Couette 2D dtype probe")
    print("=" * 60)

    fp32_couette_time = None
    for dname in dtypes_to_test:
        dt = DTYPE_MAP[dname]
        if dname == "tf32":
            set_tf32(True)
        else:
            set_tf32(False)

        print(f"\n--- Couette 2D: {dname} ---")
        try:
            res = run_couette(
                H=args.H_couette,
                tau=args.tau_couette,
                U0=args.u0_couette,
                device=device,
                dtype=dt,
                dtype_name=dname,
            )
            results["couette"][dname] = res
            print(f"  L2 rel err: {res['l2_rel_err']:.2e}")
            print(f"  Max rel err: {res['max_rel_err_pct']:.4f}%")
            print(f"  Time: {res['elapsed_s']:.1f}s")

            if dname == "fp32":
                fp32_couette_time = res["elapsed_s"]
        except Exception as e:
            print(f"  FAILED: {e}")
            results["couette"][dname] = {"error": str(e)}

    # ── Run Taylor-Green 2D ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Taylor-Green 2D dtype probe")
    print("=" * 60)

    fp32_tg_time = None
    for dname in dtypes_to_test:
        dt = DTYPE_MAP[dname]
        if dname == "tf32":
            set_tf32(True)
        else:
            set_tf32(False)

        print(f"\n--- Taylor-Green 2D: {dname} ---")
        try:
            res = run_taylor_green(
                n=args.n_tg,
                re=args.re_tg,
                u0=args.u0_tg,
                device=device,
                dtype=dt,
                dtype_name=dname,
            )
            results["taylor_green"][dname] = res
            print(f"  Energy err: {res['err_e_pct']:+.4f}%")
            print(f"  Velocity err: {res['err_vel_pct']:+.4f}%")
            print(f"  R2: {res['r2']:.6f}")
            print(f"  Time: {res['elapsed_s']:.1f}s")

            if dname == "fp32":
                fp32_tg_time = res["elapsed_s"]
        except Exception as e:
            print(f"  FAILED: {e}")
            results["taylor_green"][dname] = {"error": str(e)}

    # ── Compute speedups & recommendations ──────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    recommendations = []
    comparison_table = []

    for dname in dtypes_to_test:
        row = {"dtype": dname}

        # Couette error
        c_res = results["couette"].get(dname, {})
        if "error" not in c_res:
            couette_err_pct = c_res["max_rel_err_pct"]
            couette_time = c_res["elapsed_s"]
            row["couette_err_pct"] = round(couette_err_pct, 4)
            row["couette_time_s"] = couette_time
            if fp32_couette_time and fp32_couette_time > 0:
                row["couette_speedup"] = round(fp32_couette_time / couette_time, 3)
            else:
                row["couette_speedup"] = 1.0
        else:
            row["couette_err_pct"] = None
            row["couette_time_s"] = None
            row["couette_speedup"] = None

        # TG error
        tg_res = results["taylor_green"].get(dname, {})
        if "error" not in tg_res:
            tg_err_pct = abs(tg_res["err_e_pct"])
            tg_time = tg_res["elapsed_s"]
            row["tg_err_pct"] = round(tg_err_pct, 4)
            row["tg_time_s"] = tg_time
            if fp32_tg_time and fp32_tg_time > 0:
                row["tg_speedup"] = round(fp32_tg_time / tg_time, 3)
            else:
                row["tg_speedup"] = 1.0
        else:
            row["tg_err_pct"] = None
            row["tg_time_s"] = None
            row["tg_speedup"] = None

        # Max error across both cases
        errs = [e for e in [row.get("couette_err_pct"), row.get("tg_err_pct")] if e is not None]
        row["max_err_pct"] = max(errs) if errs else None

        # Min speedup across both cases
        speedups = [s for s in [row.get("couette_speedup"), row.get("tg_speedup")] if s is not None]
        row["min_speedup"] = min(speedups) if speedups else None

        # Check criteria
        if row["max_err_pct"] is not None and row["min_speedup"] is not None:
            passes_err = row["max_err_pct"] <= 0.1
            passes_speed = row["min_speedup"] >= 1.3
            row["passes_err_0.1pct"] = passes_err
            row["passes_speedup_1.3x"] = passes_speed
            row["recommended"] = passes_err and passes_speed
            if row["recommended"]:
                recommendations.append(dname)
        else:
            row["recommended"] = False

        comparison_table.append(row)
        print(f"  {dname}: err={row['max_err_pct']}%, speedup={row['min_speedup']}x, recommended={row['recommended']}")

    results["comparison_table"] = comparison_table
    results["recommendations"] = recommendations

    # ── Write JSON ──────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip large arrays for JSON
    json_out = {
        "mission": results["mission"],
        "description": results["description"],
        "device": results["device"],
        "device_name": results["device_name"],
        "n_tg": results["n_tg"],
        "H_couette": results["H_couette"],
        "criteria": results["criteria"],
        "couette": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("u_profile", "u_analytic")}
            for k, v in results["couette"].items()
        },
        "taylor_green": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("times", "energies", "umaxs")}
            for k, v in results["taylor_green"].items()
        },
        "comparison_table": results["comparison_table"],
        "recommendations": results["recommendations"],
    }
    out_path.write_text(json.dumps(json_out, indent=2, ensure_ascii=False))
    print(f"\nJSON written to {out_path}")

    # ── Generate plots ──────────────────────────────────────────────────
    _generate_plots(results, args.plot_dir)

    return results


def _generate_plots(results: dict, plot_dir: str) -> None:
    """Generate error-speedup comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    table = results["comparison_table"]
    dtypes = [r["dtype"] for r in table]

    # ── Plot 1: Error comparison bar chart ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    couette_errs = [r.get("couette_err_pct") or 0 for r in table]
    tg_errs = [r.get("tg_err_pct") or 0 for r in table]

    x = np.arange(len(dtypes))
    width = 0.35
    bars1 = ax.bar(x - width / 2, couette_errs, width, label="Couette 2D", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, tg_errs, width, label="Taylor-Green 2D", color="#55A868")

    ax.axhline(y=0.1, color="red", linestyle="--", linewidth=1, label="0.1% threshold")
    ax.set_xlabel("dtype")
    ax.set_ylabel("Max relative error (%)")
    ax.set_title("dtype Probe: Error vs Analytical Solution")
    ax.set_xticks(x)
    ax.set_xticklabels(dtypes)
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(plot_dir / "dtype_error_comparison.png", dpi=150)
    plt.close()

    # ── Plot 2: Speedup comparison bar chart ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    couette_sp = [r.get("couette_speedup") or 1.0 for r in table]
    tg_sp = [r.get("tg_speedup") or 1.0 for r in table]

    bars1 = ax.bar(x - width / 2, couette_sp, width, label="Couette 2D", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, tg_sp, width, label="Taylor-Green 2D", color="#55A868")

    ax.axhline(y=1.3, color="red", linestyle="--", linewidth=1, label="1.3x threshold")
    ax.set_xlabel("dtype")
    ax.set_ylabel("Speedup vs fp32")
    ax.set_title("dtype Probe: Speedup vs fp32")
    ax.set_xticks(x)
    ax.set_xticklabels(dtypes)
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "dtype_speedup_comparison.png", dpi=150)
    plt.close()

    # ── Plot 3: Couette profile comparison ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    couette = results["couette"]
    for dname in ["fp32", "tf32", "bf16"]:
        if dname in couette and "u_profile" in couette[dname]:
            u_num = couette[dname]["u_profile"]
            u_ana = couette[dname]["u_analytic"]
            H = len(u_num)
            y = np.arange(1, H + 1, dtype=np.float64) - 0.5
            ax.plot(y, u_num, "o-", label=f"{dname} (sim)", markersize=3)
            if dname == "fp32":
                ax.plot(y, u_ana, "k--", label="Analytical", linewidth=2)

    ax.set_xlabel("y (lattice units)")
    ax.set_ylabel("u_x")
    ax.set_title("Couette 2D: Velocity Profile vs Analytical")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "couette_profile_comparison.png", dpi=150)
    plt.close()

    # ── Plot 4: TG energy decay ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    tg = results["taylor_green"]
    for dname in ["fp32", "tf32", "bf16"]:
        if dname in tg and "times" in tg[dname]:
            t = np.array(tg[dname]["times"])
            e = np.array(tg[dname]["energies"])
            ax.semilogy(t, e, "o-", label=f"{dname}", markersize=3)

    # Analytical decay
    if "fp32" in tg and "error" not in tg["fp32"]:
        gamma = tg["fp32"]["gamma_e_theory"]
        e0 = tg["fp32"]["energies"][0]
        t_full = np.array(tg["fp32"]["times"])
        ax.semilogy(t_full, e0 * np.exp(-gamma * t_full), "k--", label="Analytical", linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Kinetic Energy")
    ax.set_title("Taylor-Green 2D: Energy Decay vs Analytical")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "tg_energy_decay.png", dpi=150)
    plt.close()

    print(f"Plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
