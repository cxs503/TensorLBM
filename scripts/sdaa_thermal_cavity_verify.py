#!/usr/bin/env python3
"""简化验证脚本：thermal_cavity 方腔自然对流。

由于 CPU 运行慢，本脚本：
1. 运行少量步骤验证代码通路
2. 使用已有 result.json 中的完整结果进行误差分析
3. 生成对比图和报告

用法：
  cd /root/tp-agent/work/m325
  PYTHONPATH=src:. python3 scripts/sdaa_thermal_cavity_verify.py --steps 5000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks" / "verified" / "thermal_cavity"
RESULT_JSON = BENCH_DIR / "result.json"

# 基准参考值
REFERENCES = {
    1000.0: {"nu": 1.118, "u_max": 3.649, "v_max": 3.697, "u_max_y": 0.813, "v_max_x": 0.178,
             "nu_max": 1.505, "nu_min": 0.692},
    10000.0: {"nu": 2.243, "u_max": 16.178, "v_max": 19.617, "u_max_y": 0.823, "v_max_x": 0.119,
              "nu_max": 3.528, "nu_min": 0.586},
}


def run_short_simulation(ra: float, nx: int, steps: int, dtype: str = "float32") -> dict:
    """运行少量步骤验证代码通路。"""
    sys.path.insert(0, str(BENCH_DIR))
    import driver

    device = torch.device("cpu")
    dt = torch.float32 if dtype == "float32" else torch.float64

    t0 = time.time()
    f, g, wall, p = driver.make_fields(nx, ra, 0.71, 0.6, device, dtype=dt)
    tau_T, g_beta = p["tau_T"], p["g_beta"]

    # 运行指定步数
    for step in range(1, steps + 1):
        f, g = driver.advance(f, g, wall, 0.6, tau_T, g_beta)
        if step % 1000 == 0:
            obs = driver.observables(f, g, p, 1.0, 0.0)
            print(f"  step={step}: nu={obs['nu']:.6f}, u_max={obs['u_max_nd']:.4f}, v_max={obs['v_max_nd']:.4f}")

    obs = driver.observables(f, g, p, 1.0, 0.0)
    elapsed = time.time() - t0

    return {
        "ra": ra,
        "nx": nx,
        "dtype": dtype,
        "steps": steps,
        "elapsed_s": round(elapsed, 1),
        "obs": obs,
    }


def load_existing_results() -> list[dict]:
    """加载已有 result.json 中的完整结果。"""
    with open(RESULT_JSON) as f:
        data = json.load(f)
    return data.get("cases", [])


def compute_errors(cases: list[dict]) -> list[dict]:
    """计算每个工况的误差。"""
    results = []
    for case in cases:
        ra = case["ra"]
        ref = REFERENCES.get(ra, {})
        errors = {}
        gates = {}

        for key, ref_key in [("nu", "nu"), ("u_max_nd", "u_max"), ("v_max_nd", "v_max")]:
            val = case.get(key)
            ref_val = ref.get(ref_key)
            if val is not None and ref_val is not None:
                err_pct = (val - ref_val) / ref_val * 100 if ref_val != 0 else float("inf")
                errors[key] = {"value": val, "ref": ref_val, "err_pct": round(err_pct, 4)}
                gates[key] = abs(err_pct) <= 3.0

        results.append({
            "tag": case["tag"],
            "ra": ra,
            "nx": case["nx"],
            "dtype": case["dtype"],
            "steps_run": case["steps_run"],
            "steady_triggered": case["steady_triggered"],
            "errors": errors,
            "gates": gates,
            "pass": all(gates.values()) if gates else None,
            "full_case": case,
        })

    return results


def generate_comparison_plot(results: list[dict], output_dir: Path):
    """生成误差对比图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 按 Ra 分组
    ra_groups = {}
    for r in results:
        ra = r["ra"]
        if ra not in ra_groups:
            ra_groups[ra] = {"nx": [], "nu_err": [], "u_err": [], "v_err": [], "pass": []}
        ra_groups[ra]["nx"].append(r["nx"])
        ra_groups[ra]["nu_err"].append(r["errors"].get("nu", {}).get("err_pct", None))
        ra_groups[ra]["u_err"].append(r["errors"].get("u_max_nd", {}).get("err_pct", None))
        ra_groups[ra]["v_err"].append(r["errors"].get("v_max_nd", {}).get("err_pct", None))
        ra_groups[ra]["pass"].append(r["pass"])

    colors = {1000.0: "steelblue", 10000.0: "coral"}
    labels = {1000.0: "Ra=1e3", 10000.0: "Ra=1e4"}

    for idx, (metric, ylabel) in enumerate([("nu_err", "Nu error (%)"), ("u_err", "u_max error (%)"), ("v_err", "v_max error (%)")]):
        ax = axes[idx]
        for ra, data in ra_groups.items():
            valid = [(n, e) for n, e in zip(data["nx"], data[metric]) if e is not None]
            if valid:
                nx_vals, err_vals = zip(*valid)
                ax.plot(nx_vals, err_vals, "o-", color=colors[ra], label=labels[ra], linewidth=2, markersize=8)
                # 标注 PASS/FAIL
                for n, e, p in zip(data["nx"], data[metric], data["pass"]):
                    if e is not None:
                        ax.annotate("✓" if p else "✗", (n, e), textcoords="offset points",
                                   xytext=(0, 10), ha="center", fontsize=10,
                                   color="green" if p else "red")

        ax.axhline(y=3, color="red", linestyle="--", alpha=0.5, label="±3% gate")
        ax.axhline(y=-3, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Grid size N")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([64, 128, 256])

    plt.tight_layout()
    png_path = output_dir / "error_comparison.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存误差对比图: {png_path}")
    return png_path


def generate_field_plot(output_dir: Path):
    """生成示意图（由于无完整场数据，生成示意云图）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 5))

    # 生成示意温度场（基于解析解近似）
    nx, ny = 64, 64
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # 简化温度场：线性导热 + 对流扰动
    T = (1 - X) + 0.1 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    T = np.clip(T, 0, 1)

    im = ax.contourf(X, Y, T, levels=20, cmap="RdYlBu_r")
    ax.set_xlabel("x/H")
    ax.set_ylabel("y/H")
    ax.set_title("Temperature field (schematic, Ra=1e3, N=64)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="T")

    png_path = output_dir / "temperature_field_schematic.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存温度场示意图: {png_path}")
    return png_path


def generate_convergence_plot(results: list[dict], output_dir: Path):
    """生成收敛曲线图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, ra in enumerate([1000.0, 10000.0]):
        ax = axes[idx]
        ra_results = [r for r in results if r["ra"] == ra]
        ra_results.sort(key=lambda r: r["nx"])

        nx_vals = [r["nx"] for r in ra_results]
        nu_vals = [r["full_case"]["nu"] for r in ra_results]
        ref = REFERENCES[ra]["nu"]

        ax.plot(nx_vals, nu_vals, "o-", color="steelblue", linewidth=2, markersize=8, label="LBM")
        ax.axhline(y=ref, color="red", linestyle="--", alpha=0.7, label=f"de Vahl Davis ({ref})")

        ax.set_xlabel("Grid size N")
        ax.set_ylabel("Nu")
        ax.set_title(f"Nu convergence (Ra={ra:.0e})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([64, 128, 256])

    plt.tight_layout()
    png_path = output_dir / "nu_convergence.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存收敛曲线图: {png_path}")
    return png_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000, help="验证运行步数")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "runs" / f"thermal_verify_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Thermal Cavity Verification ===")
    print(f"时间: {ts}")
    print(f"输出: {output_dir}")

    # 1. 运行少量步骤验证代码通路
    print("\n[1] 运行少量步骤验证代码通路...")
    short_results = []
    for ra, nx in [(1000.0, 64), (10000.0, 64)]:
        print(f"  Ra={ra:.0e}, N={nx}, steps={args.steps}")
        result = run_short_simulation(ra, nx, args.steps)
        short_results.append(result)
        print(f"  完成: {result['elapsed_s']:.1f}s, nu={result['obs']['nu']:.6f}")

    # 保存短期运行结果
    with open(output_dir / "short_run.json", "w") as f:
        json.dump(short_results, f, indent=2)

    # 2. 加载已有完整结果
    print("\n[2] 加载已有 result.json 完整结果...")
    existing_cases = load_existing_results()
    print(f"  加载 {len(existing_cases)} 个工况")

    # 3. 计算误差
    print("\n[3] 计算误差...")
    results = compute_errors(existing_cases)

    # 保存误差分析
    error_report = []
    for r in results:
        report = {
            "tag": r["tag"],
            "ra": r["ra"],
            "nx": r["nx"],
            "dtype": r["dtype"],
            "steps_run": r["steps_run"],
            "steady_triggered": r["steady_triggered"],
            "nu": {"value": r["errors"].get("nu", {}).get("value"),
                   "ref": r["errors"].get("nu", {}).get("ref"),
                   "err_pct": r["errors"].get("nu", {}).get("err_pct"),
                   "gate_le_3pct": r["gates"].get("nu")},
            "u_max": {"value": r["errors"].get("u_max_nd", {}).get("value"),
                      "ref": r["errors"].get("u_max_nd", {}).get("ref"),
                      "err_pct": r["errors"].get("u_max_nd", {}).get("err_pct"),
                      "gate_le_3pct": r["gates"].get("u_max_nd")},
            "v_max": {"value": r["errors"].get("v_max_nd", {}).get("value"),
                      "ref": r["errors"].get("v_max_nd", {}).get("ref"),
                      "err_pct": r["errors"].get("v_max_nd", {}).get("err_pct"),
                      "gate_le_3pct": r["gates"].get("v_max_nd")},
            "pass": r["pass"],
        }
        error_report.append(report)

    with open(output_dir / "error_analysis.json", "w") as f:
        json.dump(error_report, f, indent=2)

    # 打印误差表
    print("\n  误差表:")
    print(f"  {'Case':<15} {'Nu err%':>10} {'u_max err%':>12} {'v_max err%':>12} {'Pass':>6}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*12} {'-'*6}")
    for r in error_report:
        nu_err = f"{r['nu']['err_pct']:.2f}" if r['nu']['err_pct'] is not None else "N/A"
        u_err = f"{r['u_max']['err_pct']:.2f}" if r['u_max']['err_pct'] is not None else "N/A"
        v_err = f"{r['v_max']['err_pct']:.2f}" if r['v_max']['err_pct'] is not None else "N/A"
        status = "✅" if r["pass"] else "❌"
        print(f"  {r['tag']:<15} {nu_err:>10} {u_err:>12} {v_err:>12} {status:>6}")

    # 4. 生成图表
    print("\n[4] 生成图表...")
    generate_comparison_plot(results, output_dir)
    generate_convergence_plot(results, output_dir)
    generate_field_plot(output_dir)

    # 5. 生成判定
    print("\n[5] 判定...")
    n_pass = sum(1 for r in results if r["pass"])
    n_total = len(results)
    verdict = "PARTIAL" if 0 < n_pass < n_total else ("PASS" if n_pass == n_total else "FAIL")
    print(f"  通过: {n_pass}/{n_total}")
    print(f"  判定: {verdict}")

    # 保存判定
    verdict_data = {
        "verdict": verdict,
        "n_pass": n_pass,
        "n_total": n_total,
        "details": error_report,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "verdict.json", "w") as f:
        json.dump(verdict_data, f, indent=2)

    print(f"\n完成。结果目录: {output_dir}")
    return verdict_data


if __name__ == "__main__":
    main()
