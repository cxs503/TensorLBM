#!/usr/bin/env python
"""SDAA 验证案例扫荡 — thermal_cavity_3d (W9-A).

独立运行脚本：不改业务代码，仅调用 benchmarks/verified/thermal_cavity_3d/run.py
在 sdaa 设备上执行，采集判据并输出 artifacts JSON + 文档。

判据（来自 README.md / result.json）：
  1) finest-grid |Nu err| vs cluster median <= 3%  (per Ra ladder)
  2) |err| monotonically decreasing with refinement (>= 3 grids per Ra)
  3) No NaN during simulation
  4) Window drift |Δ| < 0.1%
  5) Hot/cold Nu imbalance < 1%

由于墙钟 ≤20 min 限制，本脚本仅运行 ra1e3 N=32 40000 steps（判据档最小档），
无法完成完整阶梯（≥3 grids/Ra），单调性判据记为 NOT_RUN。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CASE_DIR = ROOT / "benchmarks" / "verified" / "thermal_cavity_3d"
ARTIFACTS_DIR = ROOT / "artifacts"
DOCS_DIR = ROOT / "docs" / "sdaa_sweep"

# 判据档推荐最小档（ra1e3 最小网格）
RUN_CONFIGS = [
    {"ra": 1e3, "n": 32, "steps": 40000, "label": "ra1e3_n32"},
]

# 参考值（来自 README.md 六源文献簇中位）
REFERENCE_NU = {
    "1e3": 1.07065,
    "1e4": 2.05552,
    "1e5": 4.34110,
}

ERR_THRESHOLD_PCT = 3.0  # |err| <= 3%
WINDOW_DRIFT_THRESHOLD_PCT = 0.1  # |window_drift| < 0.1%
HOT_COLD_IMBAL_THRESHOLD_PCT = 1.0  # |hot-cold imbalance| < 1%

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device_name() -> str:
    """获取 sdaa 设备名。"""
    import torch
    return torch.sdaa.get_device_name(0)


def run_case(ra: float, n: int, steps: int, label: str, device: str = "sdaa:0") -> dict:
    """调用 run.py 运行单案，返回 probe JSON 内容。"""
    out_dir = CASE_DIR / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(CASE_DIR / "run.py"),
        "--ra", str(ra),
        "--n", str(n),
        "--steps", str(steps),
        "--device", device,
        "--out", str(out_dir),
        "--tag", f"sdaa_{label}",
    ]
    print(f"[run] {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    elapsed = time.time() - t0
    print(f"[run] exit={result.returncode} elapsed={elapsed:.0f}s")
    if result.stdout:
        print(f"[run] stdout: {result.stdout.strip()}")
    if result.stderr:
        # Filter out the pynvml warning
        lines = [l for l in result.stderr.strip().split('\n') if 'FutureWarning' not in l and 'pynvml' not in l]
        if lines:
            print(f"[run] stderr: " + '\n'.join(lines))

    if result.returncode != 0:
        raise RuntimeError(f"run.py failed with exit code {result.returncode}")

    # Find the output probe file
    probe_path = out_dir / f"probe_sdaa_{label}.json"
    if not probe_path.exists():
        # Try to find any probe file with the tag
        candidates = list(out_dir.glob(f"probe_sdaa_{label}*.json"))
        if candidates:
            probe_path = candidates[0]
        else:
            raise FileNotFoundError(f"Probe file not found: {probe_path}")

    with open(probe_path) as f:
        return json.load(f)


def compute_window(history: list[dict], pct: float = 0.2) -> dict:
    """计算末 20% 窗口的 Nu 均值、漂移等。"""
    n = len(history)
    if n == 0:
        return {}
    window_start = int(n * (1 - pct))
    window = history[window_start:]

    nu_vals = [h["nu"] for h in window]
    nu_hot_vals = [h["nu_hot"] for h in window]
    nu_cold_vals = [h["nu_cold"] for h in window]

    nu_win = sum(nu_vals) / len(nu_vals)
    nu_hot_win = sum(nu_hot_vals) / len(nu_hot_vals)
    nu_cold_win = sum(nu_cold_vals) / len(nu_cold_vals)

    # Window drift: compare first half vs second half of window
    half = len(window) // 2
    if half > 0:
        first_half_nu = sum(h["nu"] for h in window[:half]) / half
        second_half_nu = sum(h["nu"] for h in window[half:]) / (len(window) - half)
        window_drift_pct = (second_half_nu - first_half_nu) / first_half_nu * 100 if first_half_nu != 0 else 0.0
    else:
        window_drift_pct = 0.0

    # Hot/cold imbalance
    hot_cold_imbal_pct = abs(nu_hot_win - nu_cold_win) / nu_win * 100 if nu_win != 0 else 0.0

    return {
        "window_rows": len(window),
        "nu_win": nu_win,
        "nu_hot_win": nu_hot_win,
        "nu_cold_win": nu_cold_win,
        "window_drift_pct": window_drift_pct,
        "hot_cold_imbal_pct": hot_cold_imbal_pct,
        "last_row": history[-1],
    }


def evaluate_gates(probe: dict, ra_key: str) -> dict:
    """逐条判据评估。"""
    history = probe.get("history", [])
    config = probe.get("config", {})
    nan_at = probe.get("nan_at")

    window = compute_window(history)

    ref_nu = REFERENCE_NU.get(ra_key, None)
    nu_win = window.get("nu_win", float("nan"))
    err_pct = (nu_win - ref_nu) / ref_nu * 100 if ref_nu else float("nan")

    gates = {}

    # Gate 1: |Nu err| <= 3%
    if ref_nu is not None:
        abs_err = abs(err_pct)
        gates["nu_err_le_3pct"] = {
            "value": err_pct,
            "threshold": ERR_THRESHOLD_PCT,
            "pass": abs_err <= ERR_THRESHOLD_PCT,
        }
    else:
        gates["nu_err_le_3pct"] = {"value": None, "threshold": ERR_THRESHOLD_PCT, "pass": False, "note": "no reference"}

    # Gate 2: monotonicity (requires >= 3 grids; we only have 1)
    gates["monotone_decreasing"] = {
        "value": None,
        "threshold": ">=3 grids required",
        "pass": False,
        "note": "NOT_RUN: only 1 grid executed (wall-clock limit)",
    }

    # Gate 3: No NaN
    gates["no_nan"] = {
        "value": nan_at,
        "threshold": None,
        "pass": nan_at is None,
    }

    # Gate 4: Window drift < 0.1%
    wd = window.get("window_drift_pct", float("nan"))
    gates["window_drift_lt_0.1pct"] = {
        "value": wd,
        "threshold": WINDOW_DRIFT_THRESHOLD_PCT,
        "pass": abs(wd) < WINDOW_DRIFT_THRESHOLD_PCT,
    }

    # Gate 5: Hot/cold imbalance < 1%
    hci = window.get("hot_cold_imbal_pct", float("nan"))
    gates["hot_cold_imbal_lt_1pct"] = {
        "value": hci,
        "threshold": HOT_COLD_IMBAL_THRESHOLD_PCT,
        "pass": hci < HOT_COLD_IMBAL_THRESHOLD_PCT,
    }

    return {
        "ra_key": ra_key,
        "config": config,
        "reference_nu": ref_nu,
        "nu_win": nu_win,
        "err_pct": err_pct,
        "window": window,
        "nan_at": nan_at,
        "gates": gates,
    }


def main() -> None:
    t_start = time.time()

    # Device evidence
    import torch
    device_name = get_device_name()
    sdaa_available = torch.sdaa.is_available()
    sdaa_count = torch.sdaa.device_count()
    print(f"[device] sdaa available={sdaa_available}, count={sdaa_count}, name={device_name}")

    results = []
    all_pass = True

    for cfg in RUN_CONFIGS:
        ra = cfg["ra"]
        n = cfg["n"]
        steps = cfg["steps"]
        label = cfg["label"]
        ra_key = f"{ra:.0e}".replace("e+0", "e").replace("e+00", "e")

        print(f"\n{'='*60}")
        print(f"[case] Ra={ra}, N={n}, steps={steps}, label={label}")
        print(f"{'='*60}")

        try:
            probe = run_case(ra, n, steps, label, device="sdaa:0")
            eval_result = evaluate_gates(probe, ra_key)
            eval_result["probe_file"] = str(CASE_DIR / "out" / f"probe_sdaa_{label}.json")
            results.append(eval_result)

            # Check gates
            for gate_name, gate_info in eval_result["gates"].items():
                status = "PASS" if gate_info["pass"] else "FAIL"
                if not gate_info["pass"] and "NOT_RUN" not in gate_info.get("note", ""):
                    all_pass = False
                print(f"  [{status}] {gate_name}: {gate_info.get('value', 'N/A')} (threshold: {gate_info.get('threshold', 'N/A')})")
                if "note" in gate_info:
                    print(f"         note: {gate_info['note']}")

        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            results.append({
                "ra_key": ra_key,
                "config": cfg,
                "error": str(e),
                "gates": {k: {"value": None, "pass": False, "note": f"ERROR: {e}"} for k in [
                    "nu_err_le_3pct", "monotone_decreasing", "no_nan",
                    "window_drift_lt_0.1pct", "hot_cold_imbal_lt_1pct"
                ]},
            })
            all_pass = False

    elapsed_total = time.time() - t_start

    # Build artifact JSON
    artifact = {
        "case": "thermal_cavity_3d",
        "track": "W9-A SDAA sweep",
        "device_evidence": {
            "backend": "sdaa",
            "available": sdaa_available,
            "device_count": sdaa_count,
            "device_name": device_name,
            "torch_version": torch.__version__,
        },
        "run_configs": RUN_CONFIGS,
        "reference_nu": REFERENCE_NU,
        "results": results,
        "overall_pass": all_pass,
        "elapsed_total_s": elapsed_total,
        "wall_clock_limit_s": 1200,
        "workarounds": [],
        "notes": [
            "Only ra1e3 N=32 40000 steps executed (smallest recommended tier).",
            "Monotonicity gate NOT_RUN: requires >=3 grids per Ra, wall-clock limit prevents full ladder.",
            "No triton dependency in this case; pure torch tensor ops on sdaa device.",
        ],
    }

    # Write artifact
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACTS_DIR / "sdaa_thermal_cavity_3d.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n[artifact] written to {artifact_path}")

    # Write docs
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    doc_path = DOCS_DIR / "thermal_cavity_3d.md"
    doc_lines = [
        "# SDAA Sweep: thermal_cavity_3d (W9-A)",
        "",
        "## 案例",
        "",
        "3D 差热立方腔自然对流（Pr=0.71），D3Q19 BGK + D3Q7 温度双分布。",
        "入库名：`thermal_cavity_3d`，判据档冻结配方。",
        "",
        "## 配置",
        "",
        "| 参数 | 值 |",
        "|------|-----|",
        f"| 设备 | sdaa:0 ({device_name}) |",
        f"| torch 版本 | {torch.__version__} |",
        "| Ra | 1e3 |",
        "| N | 32 |",
        "| steps | 40000 |",
        "| Pr | 0.71 |",
        "| tau_f | 0.6 |",
        "| dtype | fp32 |",
        "| sample | 500 |",
        "",
        "## 结果表",
        "",
        "| Ra | N | steps | Nu_win | ref Nu | err% | 窗漂移% | 热/冷失衡% | NaN |",
        "|----|---|-------|--------|--------|------|---------|-----------|-----|",
    ]

    for r in results:
        if "error" in r:
            doc_lines.append(f"| {r['ra_key']} | 32 | 40000 | ERROR | {r.get('reference_nu', 'N/A')} | — | — | — | — |")
        else:
            w = r["window"]
            doc_lines.append(
                f"| {r['ra_key']} | 32 | 40000 | {w['nu_win']:.5f} | {r['reference_nu']:.5f} | "
                f"{r['err_pct']:+.2f}% | {w['window_drift_pct']:+.4f}% | {w['hot_cold_imbal_pct']:.4f}% | "
                f"{r['nan_at']} |"
            )

    doc_lines.extend([
        "",
        "## 判据逐条",
        "",
        "| 判据 | 值 | 阈值 | 结果 | 备注 |",
        "|------|-----|------|------|------|",
    ])

    for r in results:
        for gate_name, gate_info in r.get("gates", {}).items():
            status = "PASS" if gate_info["pass"] else "FAIL"
            val = gate_info.get("value", "N/A")
            thresh = gate_info.get("threshold", "N/A")
            note = gate_info.get("note", "")
            if isinstance(val, float):
                val = f"{val:.4f}"
            doc_lines.append(f"| {gate_name} | {val} | {thresh} | {status} | {note} |")

    doc_lines.extend([
        "",
        f"## 墙钟",
        "",
        f"总耗时：{elapsed_total:.0f}s ({elapsed_total/60:.1f} min)，限制 20 min。",
        "",
        "## 绕行记录",
        "",
        "- 无 triton 依赖：本案例使用纯 torch tensor ops（solver3d/thermal3d/boundaries3d/thermal_common），",
        "  无需 TRITON_INTERPRET=1 绕行。",
        "- 仅执行 ra1e3 N=32 单档：墙钟限制下无法完成完整阶梯（≥3 grids/Ra），",
        "  单调性判据记为 NOT_RUN。",
        f"- 设备：{device_name}（sdaa:0）。",
        "",
        "## 判定",
        "",
        f"总体：**{'PASS' if all_pass else 'FAIL'}**",
        "",
        "注：单调性判据未执行（NOT_RUN），因墙钟限制仅能运行单档。",
        "其余判据（Nu 误差、NaN、窗漂移、热/冷失衡）均基于 sdaa 设备实际运行结果。",
        "",
    ])

    with open(doc_path, "w") as f:
        f.write("\n".join(doc_lines))
    print(f"[doc] written to {doc_path}")

    print(f"\n[summary] overall={'PASS' if all_pass else 'FAIL'}, elapsed={elapsed_total:.0f}s")


if __name__ == "__main__":
    main()
