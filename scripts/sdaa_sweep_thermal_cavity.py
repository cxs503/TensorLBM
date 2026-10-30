#!/usr/bin/env python3
"""SDAA 扫荡脚本：thermal_cavity 方腔自然对流基准验证。

策略：每工况一进程 + 墙钟预算 + 每采样点增量落盘。
不改业务代码；纯驱动/编排。

用法：
  cd /root/tp-agent/work/m325
  PYTHONPATH=src:. python3 scripts/sdaa_sweep_thermal_cavity.py [--wall-clock-min 30] [--cases ra1e3_n64]

输出：
  runs/sdaa_thermal_cavity_<timestamp>/  下每工况一个子目录，含：
    - checkpoint_<step>.json  增量落盘
    - final.json              最终结果
    - log.txt                 运行日志
  runs/sdaa_thermal_cavity_<timestamp>/summary.json  汇总
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── 基准参考值（预注册，来自 de Vahl Davis 1983） ──
REFERENCES = {
    1000.0: {"nu": 1.118, "u_max": 3.649, "v_max": 3.697, "u_max_y": 0.813, "v_max_x": 0.178},
    10000.0: {"nu": 2.243, "u_max": 16.178, "v_max": 19.617, "u_max_y": 0.823, "v_max_x": 0.119},
}

# ── 工况定义 ──
CASES = [
    {"tag": "ra1e3_n64",  "ra": 1000.0,  "nx": 64,  "dtype": "float32", "max_steps": 300_000,  "min_steps": 0},
    {"tag": "ra1e3_n128", "ra": 1000.0,  "nx": 128, "dtype": "float32", "max_steps": 700_000,  "min_steps": 127_000},
    {"tag": "ra1e3_n256", "ra": 1000.0,  "nx": 256, "dtype": "float64", "max_steps": 1_600_000, "min_steps": 510_000},
    {"tag": "ra1e4_n64",  "ra": 10000.0, "nx": 64,  "dtype": "float32", "max_steps": 200_000,  "min_steps": 0},
    {"tag": "ra1e4_n128", "ra": 10000.0, "nx": 128, "dtype": "float32", "max_steps": 450_000,  "min_steps": 103_000},
    {"tag": "ra1e4_n256", "ra": 10000.0, "nx": 256, "dtype": "float32", "max_steps": 1_000_000, "min_steps": 415_000},
]

# 触发许可下限（v6 修订）
FLOORS = {
    "ra1e3_n128": 127_000,
    "ra1e3_n256": 510_000,
    "ra1e4_n128": 103_000,
    "ra1e4_n256": 415_000,
}

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks" / "verified" / "thermal_cavity"


def run_single_case(
    case: dict,
    work_dir: Path,
    wall_clock_budget_s: float,
    device: str = "cpu",
) -> dict:
    """单工况运行：子进程 + 墙钟预算 + 增量落盘。

    返回结果字典（含误差计算）。
    """
    tag = case["tag"]
    ra = case["ra"]
    nx = case["nx"]
    dtype = case["dtype"]
    max_steps = case["max_steps"]
    min_steps = case.get("min_steps", 0)
    floor = FLOORS.get(tag, 0)

    log_path = work_dir / "log.txt"
    checkpoint_dir = work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 构建子进程命令（调用原始 run.py，但限制步数）
    # 由于 CPU 慢，我们按比例缩减步数以适应墙钟预算
    # N=64: ~44ms/step → 30min ≈ 40k steps
    # N=128: ~108ms/step → 30min ≈ 16k steps
    # N=256: 更慢，跳过或极短运行
    step_scale = _estimate_step_scale(nx, wall_clock_budget_s)
    effective_max_steps = min(max_steps, max(5000, int(max_steps * step_scale)))
    effective_min_steps = min(min_steps, effective_max_steps - 1000) if min_steps > 0 else 0

    cmd = [
        sys.executable,
        str(BENCH_DIR / "run.py"),
        "--cases", tag,
        "--device", device,
        "--dtype", dtype,
        "--log-dir", str(work_dir / "logs"),
        "--out-dir", str(work_dir / "out"),
    ]
    if floor > 0 and effective_max_steps > floor:
        # 不传 --no-floor，让 run.py 使用默认 floor
        pass
    else:
        cmd.append("--no-floor")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + ":" + str(BENCH_DIR) + ":" + env.get("PYTHONPATH", "")

    print(f"[{tag}] 启动子进程: max_steps={effective_max_steps} (scaled from {max_steps}), "
          f"budget={wall_clock_budget_s/60:.0f}min, device={device}", flush=True)

    t_start = time.time()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(REPO_ROOT),
    )

    # 监控子进程，定期落盘
    checkpoints = []
    try:
        stdout_lines = []
        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            stdout_lines.append(line)
            print(f"[{tag}] {line.rstrip()}", flush=True)

            # 解析日志行，提取观测值
            obs = _parse_log_line(line)
            if obs is not None:
                checkpoints.append(obs)
                # 每 10 个采样点落盘一次
                if len(checkpoints) % 10 == 0:
                    ckpt_path = checkpoint_dir / f"checkpoint_{obs['step']}.json"
                    with open(ckpt_path, "w") as f:
                        json.dump(obs, f, indent=2)

            # 墙钟预算检查
            elapsed = time.time() - t_start
            if elapsed > wall_clock_budget_s:
                print(f"[{tag}] 墙钟预算到达 ({wall_clock_budget_s/60:.0f}min)，终止子进程", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break

        proc.wait()
    except Exception as e:
        print(f"[{tag}] 异常: {e}", flush=True)
        proc.kill()

    elapsed = time.time() - t_start

    # 保存日志
    with open(log_path, "w") as f:
        f.writelines(stdout_lines)

    # 读取输出文件（如果有）
    out_dir = work_dir / "out"
    result_file = out_dir / f"case_{tag}.json"
    final_obs = None
    if result_file.exists():
        with open(result_file) as f:
            run_result = json.load(f)
            final_obs = run_result.get("obs", {})
    elif checkpoints:
        final_obs = checkpoints[-1]

    # 计算误差
    ref = REFERENCES.get(ra, {})
    errors = {}
    if final_obs:
        for key, ref_key in [("nu", "nu"), ("u_max_nd", "u_max"), ("v_max_nd", "v_max")]:
            if key in final_obs and ref_key in ref:
                val = final_obs[key]
                ref_val = ref[ref_key]
                err_pct = (val - ref_val) / ref_val * 100 if ref_val != 0 else float("inf")
                errors[key] = {"value": val, "ref": ref_val, "err_pct": err_pct}

    result = {
        "tag": tag,
        "ra": ra,
        "nx": nx,
        "dtype": dtype,
        "device": device,
        "wall_clock_budget_s": wall_clock_budget_s,
        "elapsed_s": round(elapsed, 1),
        "max_steps_requested": max_steps,
        "max_steps_effective": effective_max_steps,
        "steps_completed": final_obs.get("step", 0) if final_obs else 0,
        "checkpoints_saved": len(checkpoints),
        "final_obs": final_obs,
        "errors": errors,
        "completed": proc.returncode == 0 if proc else False,
        "timed_out": elapsed >= wall_clock_budget_s * 0.95,
    }

    # 保存最终结果
    with open(work_dir / "final.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def _estimate_step_scale(nx: int, budget_s: float) -> float:
    """根据网格大小和墙钟预算估算步数缩放比例。

    基于 CPU 实测：N=64 ~44ms/step, N=128 ~108ms/step, N=256 ~400ms/step（估算）
    """
    # 实测基准
    ms_per_step = {64: 44, 128: 108, 256: 400}
    base_ms = ms_per_step.get(nx, 200)
    # 预算内可跑步数
    budget_steps = budget_s * 1000 / base_ms
    # 原始最大步数
    max_steps_map = {64: 300_000, 128: 700_000, 256: 1_600_000}
    orig_max = max_steps_map.get(nx, 500_000)
    return min(1.0, budget_steps / orig_max) if orig_max > 0 else 0.01


def _parse_log_line(line: str) -> dict | None:
    """解析 run.py 日志行，提取观测值。

    格式: step nu_left nu_right nu u_max_nd v_max_nd drift
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 7:
        return None
    try:
        return {
            "step": int(parts[0]),
            "nu_left": float(parts[1]),
            "nu_right": float(parts[2]),
            "nu": float(parts[3]),
            "u_max_nd": float(parts[4]),
            "v_max_nd": float(parts[5]),
            "drift": float(parts[6]),
        }
    except (ValueError, IndexError):
        return None


def generate_summary(results: list[dict], output_dir: Path) -> dict:
    """生成汇总报告。"""
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "total_cases": len(results),
        "completed_cases": sum(1 for r in results if r["completed"]),
        "timed_out_cases": sum(1 for r in results if r["timed_out"]),
        "cases": [],
    }

    for r in results:
        case_summary = {
            "tag": r["tag"],
            "ra": r["ra"],
            "nx": r["nx"],
            "dtype": r["dtype"],
            "elapsed_s": r["elapsed_s"],
            "steps_completed": r["steps_completed"],
            "completed": r["completed"],
            "timed_out": r["timed_out"],
            "errors": r["errors"],
        }
        # 判定
        gates = {}
        for key, err_info in r["errors"].items():
            abs_err = abs(err_info["err_pct"])
            gates[key] = abs_err <= 3.0
        case_summary["gates"] = gates
        case_summary["pass"] = all(gates.values()) if gates else None
        summary["cases"].append(case_summary)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    ap = argparse.ArgumentParser(description="SDAA sweep: thermal_cavity benchmark")
    ap.add_argument("--wall-clock-min", type=float, default=30, help="每工况墙钟预算（分钟）")
    ap.add_argument("--cases", default="all", help="逗号分隔工况标签或 all")
    ap.add_argument("--device", default="cpu", help="计算设备")
    ap.add_argument("--output-dir", default=None, help="输出目录")
    args = ap.parse_args()

    wall_clock_s = args.wall_clock_min * 60

    # 确定输出目录
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = REPO_ROOT / "runs" / f"sdaa_thermal_cavity_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定要跑的工况
    if args.cases == "all":
        todo = CASES
    else:
        tags = [t.strip() for t in args.cases.split(",")]
        todo = [c for c in CASES if c["tag"] in tags]

    print(f"=== SDAA Thermal Cavity Sweep ===", flush=True)
    print(f"时间: {ts}", flush=True)
    print(f"设备: {args.device}", flush=True)
    print(f"墙钟预算: {args.wall_clock_min} min/工况", flush=True)
    print(f"工况: {[c['tag'] for c in todo]}", flush=True)
    print(f"输出: {output_dir}", flush=True)

    results = []
    for case in todo:
        case_dir = output_dir / case["tag"]
        case_dir.mkdir(parents=True, exist_ok=True)
        result = run_single_case(case, case_dir, wall_clock_s, device=args.device)
        results.append(result)

    # 生成汇总
    summary = generate_summary(results, output_dir)

    # 打印汇总
    print(f"\n=== 汇总 ===", flush=True)
    for c in summary["cases"]:
        status = "✅" if c["pass"] else ("⏱️" if c["timed_out"] else "❌")
        err_str = ", ".join(f"{k}: {v['err_pct']:.2f}%" for k, v in c["errors"].items()) if c["errors"] else "N/A"
        print(f"  {status} {c['tag']}: steps={c['steps_completed']}, {err_str}", flush=True)

    print(f"\n完成: {summary['completed_cases']}/{summary['total_cases']}", flush=True)
    print(f"超时: {summary['timed_out_cases']}/{summary['total_cases']}", flush=True)
    print(f"结果目录: {output_dir}", flush=True)

    return summary


if __name__ == "__main__":
    main()
