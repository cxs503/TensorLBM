#!/usr/bin/env python3
"""SDAA 验证扫荡：cavity_natural_convection (de Vahl Davis 1983, Ra=1e4, Pr=0.71)

流程：
  1. 设备证据：torch.sdaa.is_available()、设备名
  2. 网格收敛：N=64, 128, 256 三档
  3. 判据验证：|Nu 误差| < 3%（相对 de Vahl Davis 1983 基准 Nu=2.243）

产物：
  - artifacts/sdaa_cavity_natural_convection_evidence.json
  - runs/sdaa_cavity_natural_convection/  (日志 + 图件)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

# ── 路径 ──────────────────────────────────────────────────────────
WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(WORKTREE / "benchmarks"))

CASE_DIR = WORKTREE / "benchmarks" / "verified" / "cavity_natural_convection"
sys.path.insert(0, str(CASE_DIR))

import driver  # noqa: E402  (benchmarks/verified/cavity_natural_convection/driver.py)

# ── 物理参数 & 基准 ──────────────────────────────────────────────
RA = 1e4
PR = 0.71
TAU = 0.6
NU_REF = 2.243
U_MAX_REF = 16.178
V_MAX_REF = 19.617
GRIDS = [64, 128, 256]
MAX_STEPS = {64: 200_000, 128: 450_000, 256: 1_000_000}
MIN_STEPS = {64: 0, 128: 103_000, 256: 414_000}
TOLERANCE_PCT = 3.0  # 判据阈值


def detect_device() -> dict:
    """SDAA 设备探测，返回设备证据行。"""
    evidence = {
        "framework": "torch",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        sdaa_avail = torch.sdaa.is_available()
    except Exception:
        sdaa_avail = False
    evidence["sdaa_available"] = sdaa_avail
    if sdaa_avail:
        evidence["sdaa_device_count"] = torch.sdaa.device_count()
        try:
            evidence["sdaa_device_name"] = torch.sdaa.get_device_name(0)
        except Exception:
            evidence["sdaa_device_name"] = "unknown"
        evidence["device"] = "sdaa:0"
        evidence["device_obj"] = "torch.device('sdaa')"
    else:
        evidence["device"] = "cpu"
        evidence["device_obj"] = "torch.device('cpu')"
    return evidence


def run_single_grid(nx: int, device: torch.device, run_dir: Path) -> dict:
    """单档网格运行，返回观测量 dict。"""
    t_hot, t_cold = 1.0, 0.0
    f, g, wall, p = driver.make_fields(nx, RA, PR, TAU, device)
    tau_T, g_beta = p["tau_T"], p["g_beta"]

    max_steps = MAX_STEPS[nx]
    min_steps = MIN_STEPS[nx]
    every = 1000

    import tensorlbm.thermal as th_mod
    th_sha = hashlib.sha256(open(th_mod.__file__, "rb").read()).hexdigest()[:12]

    log_path = run_dir / f"case_ra1e4_n{nx}_sdaa.log"
    log = open(log_path, "w", buffering=1)
    log.write(f"# SDAA sweep: ra1e4_n{nx} thermal={th_mod.__file__} sha12={th_sha}\n")
    log.write(f"# params {json.dumps(p)} tau=0.6 device={device}\n")
    log.write("# step nu_left nu_right nu u_max_nd v_max_nd drift\n")

    prev = None
    quiet = 0
    t_stop = None
    target = max_steps
    hist = []
    t0 = time.time()
    step = 0

    while step < target:
        step += 1
        f, g = driver.advance(f, g, wall, TAU, tau_T, g_beta, t_hot, t_cold)

        if step % every == 0:
            obs = driver.observables(f, g, p, t_hot, t_cold)
            drift = (
                0.0
                if prev is None
                else max(abs(obs[k] - prev[k]) for k in ("nu", "nu_left", "nu_right"))
            )
            log.write(
                f"{step} {obs['nu_left']:.8f} {obs['nu_right']:.8f} {obs['nu']:.8f} "
                f"{obs['u_max_nd']:.6f} {obs['v_max_nd']:.6f} {drift:.3e}\n"
            )
            hist.append({"step": step, "nu": obs["nu"], "drift": drift})
            prev = obs

            if t_stop is None:
                if prev is not None and step > every and step >= min_steps and drift < 1e-4:
                    quiet += 1
                else:
                    quiet = 0
                if quiet >= 3:
                    t_stop = step
                    target = min(max_steps, step + max(10_000, step // 10))
                    log.write(f"# steady trigger at {step}; verification to {target}\n")

    elapsed = time.time() - t0
    obs = driver.observables(f, g, p, t_hot, t_cold)

    log.write(
        f"# final: nu={obs['nu']:.8f} u_max_nd={obs['u_max_nd']:.6f} v_max_nd={obs['v_max_nd']:.6f}\n"
    )
    log.close()

    nu_err = 100.0 * (obs["nu"] - NU_REF) / NU_REF
    u_err = 100.0 * (obs["u_max_nd"] - U_MAX_REF) / U_MAX_REF
    v_err = 100.0 * (obs["v_max_nd"] - V_MAX_REF) / V_MAX_REF

    return {
        "tag": f"ra1e4_n{nx}_sdaa",
        "ra": RA,
        "nx": nx,
        "dtype": "float32",
        "device": str(device),
        "steps_run": step,
        "max_steps": max_steps,
        "min_steps": min_steps,
        "steady_triggered": t_stop is not None,
        "t_stop": t_stop,
        "thermal_sha12": th_sha,
        "elapsed_s": round(elapsed, 1),
        "nu": obs["nu"],
        "nu_left": obs["nu_left"],
        "nu_right": obs["nu_right"],
        "nu_sym": obs["nu_sym"],
        "u_max_nd": obs["u_max_nd"],
        "v_max_nd": obs["v_max_nd"],
        "u_max_lu": obs["u_max_lu"],
        "v_max_lu": obs["v_max_lu"],
        "u_max_y": obs["u_max_y"],
        "v_max_x": obs["v_max_x"],
        "nu_history": hist,
        "nu_err_pct": nu_err,
        "u_max_err_pct": u_err,
        "v_max_err_pct": v_err,
        "nu_err_pct_abs": abs(nu_err),
        "u_max_err_pct_abs": abs(u_err),
        "v_max_err_pct_abs": abs(v_err),
        "T_min": obs["T_min"],
        "T_max": obs["T_max"],
        "mach": obs["mach"],
    }


def generate_figures(results: list, run_dir: Path):
    """生成论文级图件：Nu 收敛曲线 + 误差对比柱状图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("WARNING: matplotlib not available, skipping figures")
        return []

    fig_paths = []

    # ── 图1: Nu 收敛曲线 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, res in enumerate(results):
        steps = [h["step"] for h in res["nu_history"]]
        nus = [h["nu"] for h in res["nu_history"]]
        ax.plot(steps, nus, marker=markers[i], markersize=3, color=colors[i],
                label=f"N={res['nx']} (Nu={res['nu']:.4f})")
    ax.axhline(y=NU_REF, color="red", linestyle="--", linewidth=1.5, label=f"Reference Nu={NU_REF}")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Nu (average Nusselt number)", fontsize=12)
    ax.set_title("Natural Convection Cavity (Ra=10⁴, Pr=0.71)\nNu Convergence History", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    fig.tight_layout()
    p1 = run_dir / "curve_nu_convergence_sdaa.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    fig_paths.append(str(p1))

    # ── 图2: 误差对比柱状图 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    grid_labels = [f"N={r['nx']}" for r in results]
    x = range(len(grid_labels))
    width = 0.25
    nu_errs = [r["nu_err_pct"] for r in results]
    u_errs = [r["u_max_err_pct"] for r in results]
    v_errs = [r["v_max_err_pct"] for r in results]
    bars1 = ax.bar([i - width for i in x], nu_errs, width, label="Nu error %", color="#1f77b4")
    bars2 = ax.bar(x, u_errs, width, label="u_max error %", color="#ff7f0e")
    bars3 = ax.bar([i + width for i in x], v_errs, width, label="v_max error %", color="#2ca02c")
    ax.axhline(y=TOLERANCE_PCT, color="red", linestyle="--", linewidth=1, label=f"±{TOLERANCE_PCT}% threshold")
    ax.axhline(y=-TOLERANCE_PCT, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Grid Resolution", fontsize=12)
    ax.set_ylabel("Error (%)", fontsize=12)
    ax.set_title("Error vs Reference (de Vahl Davis 1983)", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels(grid_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    # 标注数值
    for bar_group in [bars1, bars2, bars3]:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    p2 = run_dir / "curve_error_comparison_sdaa.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    fig_paths.append(str(p2))

    # ── 图3: 单调收敛验证 ──
    fig, ax = plt.subplots(figsize=(7, 5))
    nu_err_abs = [abs(r["nu_err_pct"]) for r in results]
    ax.plot([r["nx"] for r in results], nu_err_abs, "o-", color="#d62728", markersize=8, linewidth=2)
    ax.axhline(y=TOLERANCE_PCT, color="green", linestyle="--", linewidth=1.5, label=f"{TOLERANCE_PCT}% threshold")
    for i, (nx_val, err_val) in enumerate(zip([r["nx"] for r in results], nu_err_abs)):
        ax.annotate(f"{err_val:.2f}%", xy=(nx_val, err_val),
                    xytext=(5, 10), textcoords="offset points", fontsize=11, fontweight="bold")
    ax.set_xlabel("Grid N", fontsize=12)
    ax.set_ylabel("|Nu error| (%)", fontsize=12)
    ax.set_title("Grid Convergence: |Nu error| vs N", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p3 = run_dir / "curve_monotone_convergence_sdaa.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    fig_paths.append(str(p3))

    return fig_paths


def main():
    print("=" * 70)
    print("SDAA SWEEP: cavity_natural_convection (de Vahl Davis 1983)")
    print("=" * 70)

    # ── Step 1: 设备探测 ──
    print("\n[1/4] Device detection...")
    dev_evidence = detect_device()
    print(json.dumps(dev_evidence, indent=2))

    if not dev_evidence.get("sdaa_available", False):
        print("WARNING: SDAA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("sdaa")
    print(f"Using device: {device}")

    # ── Step 2: 运行目录 ──
    run_dir = WORKTREE / "runs" / "sdaa_cavity_natural_convection"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/4] Run directory: {run_dir}")

    # ── Step 3: 网格收敛 ──
    print(f"\n[3/4] Running grids: {GRIDS}")
    print(f"  Reference: Nu={NU_REF}, u_max={U_MAX_REF}, v_max={V_MAX_REF}")
    print()

    results = []
    for nx in GRIDS:
        print(f"  === Grid N={nx} on {device} ===")
        t0 = time.time()
        res = run_single_grid(nx, device, run_dir)
        results.append(res)
        print(f"    Nu = {res['nu']:.6f}  (err = {res['nu_err_pct']:+.3f}%)")
        print(f"    u_max = {res['u_max_nd']:.6f}  (err = {res['u_max_err_pct']:+.3f}%)")
        print(f"    v_max = {res['v_max_nd']:.6f}  (err = {res['v_max_err_pct']:+.3f}%)")
        print(f"    Steps: {res['steps_run']}, elapsed: {res['elapsed_s']:.1f}s")
        print()

    # ── Step 4: 判据验证 ──
    print("[4/4] Gate evaluation...")
    nu_errs_abs = [r["nu_err_pct_abs"] for r in results]
    monotone = all(nu_errs_abs[i] > nu_errs_abs[i + 1] for i in range(len(nu_errs_abs) - 1))
    passing_grids = [GRIDS[i] for i, e in enumerate(nu_errs_abs) if e <= TOLERANCE_PCT]
    all_pass = all(e <= TOLERANCE_PCT for e in nu_errs_abs)

    print(f"  Nu errors (abs): {[f'{e:.2f}%' for e in nu_errs_abs]}")
    print(f"  Monotone convergence: {monotone}")
    print(f"  Grids passing ≤{TOLERANCE_PCT}%: {passing_grids}")

    verdict = "PASS" if (len(passing_grids) >= 2 and monotone) else "FAIL"
    print(f"\n  >>> VERDICT: {verdict}")

    # ── 生成图件 ──
    print("\nGenerating figures...")
    fig_paths = generate_figures(results, run_dir)
    for fp in fig_paths:
        print(f"  {fp}")

    # ── 组装证据文件 ──
    evidence = {
        "case": "cavity_natural_convection",
        "benchmark": "de Vahl Davis 1983 — natural convection square cavity",
        "mission_id": 482,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_evidence": dev_evidence,
        "physical_params": {
            "Ra": RA,
            "Pr": PR,
            "tau": TAU,
            "lattice": "D2Q9 + D2Q5",
            "collision": "BGK + Guo buoyancy",
        },
        "reference": {
            "source": "de Vahl Davis 1983 Int. J. Numer. Methods Fluids 3:249-264",
            "Nu": NU_REF,
            "u_max": U_MAX_REF,
            "v_max": V_MAX_REF,
        },
        "grids": results,
        "convergence": {
            "nu_errors_abs_pct": nu_errs_abs,
            "monotone": monotone,
            "passing_grids": passing_grids,
        },
        "gate": {
            "tolerance_pct": TOLERANCE_PCT,
            "verdict": verdict,
            "criteria": "|Nu error| < 3% for ≥2 grids with monotone convergence",
        },
        "figures": [os.path.relpath(fp, WORKTREE) for fp in fig_paths],
        "logs": [f"runs/sdaa_cavity_natural_convection/case_ra1e4_n{nx}_sdaa.log" for nx in GRIDS],
    }

    evidence_path = WORKTREE / "artifacts" / "sdaa_cavity_natural_convection_evidence.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"\nEvidence saved: {evidence_path}")

    return evidence


if __name__ == "__main__":
    evidence = main()
