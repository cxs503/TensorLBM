#!/usr/bin/env python3
"""
mem_profile.py — 单卡显存/功耗画像脚本
========================================
目标: 测量不同网格规模下 matmul 的显存峰值(MB)及功耗/降频信息。
规模阶梯: 512², 1024², 2048², 4096² (float32 matmul)

设备优先级: torch_sdaa (MLU) > torch.cuda > CPU fallback
功耗接口: cnmon / sysfs devfreq / torch_sdaa power API
         若均不可用则显式声明不可测。

产出:
  - artifacts/mem_table.json  (规模 → 峰值显存 MB)
  - artifacts/mem_profile_curve.png (matplotlib 曲线)
"""

import json
import os
import sys
import time
import gc
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# 0. 路径 & 常量
# ---------------------------------------------------------------------------
WORK_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS = WORK_DIR / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

GRID_SIZES = [512, 1024, 2048, 4096]
DTYPE = "float32"
NUM_WARMUP = 2
NUM_REPEAT = 5

# ---------------------------------------------------------------------------
# 1. 设备检测
# ---------------------------------------------------------------------------
def detect_device():
    """返回 (device_type, device_name, device_obj_or_None)"""
    # 尝试 torch_sdaa
    try:
        import torch_sdaa
        import torch
        # torch_sdaa 通过 privateuse1 注册
        if hasattr(torch, 'sdaa') and torch.sdaa.is_available():
            name = torch.sdaa.get_device_name(0)
            return ("sdaa", name, torch.device("sdaa:0"))
    except Exception:
        pass

    # 尝试 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return ("cuda", name, torch.device("cuda:0"))
    except Exception:
        pass

    # CPU fallback
    import torch
    return ("cpu", "CPU (no accelerator)", torch.device("cpu"))


# ---------------------------------------------------------------------------
# 2. 内存测量
# ---------------------------------------------------------------------------
def measure_memory_torch(device_type):
    """通过 torch 接口获取已分配显存 (MB)"""
    import torch
    if device_type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024**2)
    elif device_type == "sdaa":
        try:
            return torch.sdaa.max_memory_allocated() / (1024**2)
        except Exception:
            pass
    # CPU / fallback: 用 psutil 进程 RSS
    return None


def measure_memory_psutil():
    """通过 psutil 获取进程 RSS (MB)"""
    import psutil
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024**2)


def reset_memory_stats(device_type):
    import torch
    gc.collect()
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    elif device_type == "sdaa":
        try:
            torch.sdaa.reset_peak_memory_stats()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 3. 功耗 / 降频探测
# ---------------------------------------------------------------------------
def probe_power_capabilities():
    """探测可用的功耗/频率接口，返回 dict 描述"""
    result = {
        "cnmon": False,
        "sysfs_devfreq": False,
        "torch_sdaa_power": False,
        "sysfs_hwmon": False,
        "attempts": []
    }

    # cnmon
    try:
        r = subprocess.run(["cnmon", "-d", "0"], capture_output=True, text=True, timeout=5)
        result["cnmon"] = (r.returncode == 0)
        result["attempts"].append(f"cnmon: returncode={r.returncode}")
    except FileNotFoundError:
        result["attempts"].append("cnmon: not found in PATH")
    except Exception as e:
        result["attempts"].append(f"cnmon: {e}")

    # sysfs devfreq
    devfreq_path = Path("/sys/class/devfreq")
    if devfreq_path.exists():
        entries = list(devfreq_path.iterdir())
        result["sysfs_devfreq"] = len(entries) > 0
        result["attempts"].append(f"sysfs_devfreq: {len(entries)} entries")
    else:
        result["attempts"].append("sysfs_devfreq: /sys/class/devfreq not found")

    # torch_sdaa power API
    try:
        import torch_sdaa
        import torch
        if hasattr(torch, 'sdaa'):
            if hasattr(torch.sdaa, 'power'):
                result["torch_sdaa_power"] = True
                result["attempts"].append("torch_sdaa.power: available")
            else:
                result["attempts"].append("torch_sdaa.power: attribute not found")
        else:
            result["attempts"].append("torch.sdaa: not available (no hardware)")
    except Exception as e:
        result["attempts"].append(f"torch_sdaa power: {e}")

    # hwmon
    hwmon_path = Path("/sys/class/hwmon")
    if hwmon_path.exists():
        result["sysfs_hwmon"] = True
        result["attempts"].append(f"sysfs_hwmon: {hwmon_path} exists")
    else:
        result["attempts"].append("sysfs_hwmon: not found")

    return result


# ---------------------------------------------------------------------------
# 4. 核心测量: matmul 阶梯
# ---------------------------------------------------------------------------
def run_matmul_profile(N, device, device_type):
    """对 N×N matmul 做 warmup+repeat, 返回峰值显存 MB 和耗时 ms"""
    import torch

    reset_memory_stats(device_type)

    # 分配
    A = torch.randn(N, N, dtype=torch.float32, device=device)
    B = torch.randn(N, N, dtype=torch.float32, device=device)

    # warmup
    for _ in range(NUM_WARMUP):
        C = torch.matmul(A, B)
        del C

    # 同步 & 测内存
    if device_type in ("cuda", "sdaa"):
        try:
            torch.sdaa.synchronize() if device_type == "sdaa" else torch.cuda.synchronize()
        except Exception:
            pass

    mem_before = measure_memory_psutil()
    peak_times = []

    for _ in range(NUM_REPEAT):
        t0 = time.perf_counter()
        C = torch.matmul(A, B)
        if device_type in ("cuda", "sdaa"):
            try:
                if device_type == "sdaa":
                    torch.sdaa.synchronize()
                else:
                    torch.cuda.synchronize()
            except Exception:
                pass
        t1 = time.perf_counter()
        peak_times.append((t1 - t0) * 1000)  # ms
        del C

    # 峰值显存
    torch_peak = measure_memory_torch(device_type)
    rss_peak = measure_memory_psutil()

    # 理论显存: 3 matrices (A, B, C) × N² × 4 bytes
    theoretical_mb = 3 * N * N * 4 / (1024**2)

    # 清理
    del A, B
    gc.collect()
    if device_type in ("cuda", "sdaa"):
        try:
            if device_type == "sdaa":
                torch.sdaa.empty_cache()
            else:
                torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "N": N,
        "theoretical_mb": round(theoretical_mb, 2),
        "peak_mem_mb": round(torch_peak if torch_peak is not None else rss_peak, 2),
        "mem_source": "torch_device" if torch_peak is not None else "psutil_rss",
        "avg_time_ms": round(sum(peak_times) / len(peak_times), 3),
        "min_time_ms": round(min(peak_times), 3),
        "max_time_ms": round(max(peak_times), 3),
    }


# ---------------------------------------------------------------------------
# 5. 绘图
# ---------------------------------------------------------------------------
def plot_curve(results, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    Ns = [r["N"] for r in results]
    peaks = [r["peak_mem_mb"] for r in results]
    theoretical = [r["theoretical_mb"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(Ns))
    width = 0.35

    bars1 = ax.bar(x - width/2, peaks, width, label="Measured Peak (MB)", color="#2196F3")
    bars2 = ax.bar(x + width/2, theoretical, width, label="Theoretical 3×N²×4B (MB)", color="#FF9800")

    ax.set_xlabel("Grid Size N")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Profile: N×N matmul Peak Memory vs Grid Size")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 在柱上标注数值
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.0f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.0f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Plot saved: {output_path}")


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  TensorLBM 单卡显存/功耗画像  ")
    print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # 设备检测
    device_type, device_name, device = detect_device()
    print(f"\n[Device] type={device_type}, name={device_name}")

    # 功耗探测
    power_info = probe_power_capabilities()
    print(f"\n[Power Probe]")
    for a in power_info["attempts"]:
        print(f"  - {a}")
    power_available = any([power_info["cnmon"], power_info["sysfs_devfreq"],
                           power_info["torch_sdaa_power"]])
    if not power_available:
        print("  ⚠ 功耗/降频: 所有厂商接口均不可用，显式声明不可测")

    # 显存测量
    print(f"\n[Memory Profiling] sizes={GRID_SIZES}, dtype={DTYPE}")
    results = []
    for N in GRID_SIZES:
        print(f"  N={N} ... ", end="", flush=True)
        try:
            r = run_matmul_profile(N, device, device_type)
            r["device_type"] = device_type
            r["device_name"] = device_name
            results.append(r)
            print(f"peak={r['peak_mem_mb']:.1f} MB, avg={r['avg_time_ms']:.1f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()
            results.append({
                "N": N,
                "peak_mem_mb": None,
                "error": str(e),
                "device_type": device_type,
                "device_name": device_name,
            })

    # 写 JSON
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_type": device_type,
        "device_name": device_name,
        "dtype": DTYPE,
        "num_warmup": NUM_WARMUP,
        "num_repeat": NUM_REPEAT,
        "power_probe": power_info,
        "power_available": power_available,
        "power_note": "功耗/降频不可测: cnmon 未安装, /sys/class/devfreq 无条目, torch_sdaa 无 power 接口 (无硬件). 尝试了 cnmon CLI, sysfs devfreq, torch_sdaa power API, /sys/class/hwmon 四种途径." if not power_available else "",
        "measurements": results,
    }

    json_path = ARTIFACTS / "mem_table.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] JSON saved: {json_path}")

    # 绘图
    valid_results = [r for r in results if r.get("peak_mem_mb") is not None]
    if valid_results:
        plot_path = ARTIFACTS / "mem_profile_curve.png"
        plot_curve(valid_results, plot_path)
    else:
        print("[WARN] No valid measurements, skipping plot")

    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
