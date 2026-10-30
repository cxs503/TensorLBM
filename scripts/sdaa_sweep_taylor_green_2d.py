#!/usr/bin/env python3
"""SDAA 验证案例扫荡：taylor_green_2d（B17）。

独立运行脚本——不改业务代码。计算张量放 'sdaa' 设备；若入口依赖
triton，用 TRITON_INTERPRET=1 并如实记录绕行。网格/步数按案例文档
推荐值或可行较小档，墙钟 ≤20 分钟。

判据（来自 benchmarks/verified/taylor_green_2d/README.md）：
  1. 动能衰减率偏差 |γ_E_sim/γ_E_theory − 1| ≤ 1%
  2. 速度衰减率偏差 |γ_vel_sim/γ_vel_theory − 1| ≤ 1%
  3. 指数衰减质量 ln E vs t 线性拟合 R² ≥ 0.999

输出：artifacts/sdaa_taylor_green_2d.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── 路径引导：让 tensorlbm 和 compile_route 可导入 ─────────────────────
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
BENCH = REPO / "benchmarks"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import torch_sdaa  # noqa: E402

from compile_route import route_step  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

# ── 设备证据 ───────────────────────────────────────────────────────────
DEVICE = "sdaa:0"
DEVICE_NAME = torch.sdaa.get_device_name(0)
DEVICE_COUNT = int(torch.sdaa.device_count())
DEVICE_TOTAL_MEM = int(torch.sdaa.get_device_properties(0).total_memory)  # MiB
TORCH_SDAA_VERSION = getattr(torch_sdaa, "__version__", "unknown")

# ── 案例配置（标准档 N=128, Re=100, U0=0.05）──────────────────────────
N = 128
RE = 100
U0 = 0.05
DTYPE = "float32"
RECORD_EVERY = 100

# 自动步数：E/E0 → e^-3
L = float(N)
k = 2.0 * math.pi / L
nu = U0 * L / RE
gamma_e_theory = 4.0 * nu * k * k
steps = max(1000, int(math.ceil(3.0 / gamma_e_theory / RECORD_EVERY) * RECORD_EVERY))
steps = min(steps, 200000)

tau = 0.5 + 3.0 * nu
gamma_vel_theory = 2.0 * nu * k * k

print(f"=== SDAA Taylor-Green 2D ===")
print(f"  Device: {DEVICE_NAME} (count={DEVICE_COUNT}, mem={DEVICE_TOTAL_MEM} MiB)")
print(f"  N={N}, Re={RE}, U0={U0}, dtype={DTYPE}")
print(f"  nu={nu:.6f}, tau={tau:.6f}, k={k:.6f}")
print(f"  gamma_E_theory={gamma_e_theory:.6e}, gamma_vel_theory={gamma_vel_theory:.6e}")
print(f"  steps={steps}, record_every={RECORD_EVERY}")

# ── 检查 triton 依赖 ──────────────────────────────────────────────────
triton_interpret = os.environ.get("TRITON_INTERPRET", "0")
triton_note = ""
try:
    import triton  # noqa: F401
    if triton_interpret != "1":
        triton_note = "triton 已安装但未设 TRITON_INTERPRET=1；本案例不直接依赖 triton，仅 torch.compile 间接使用"
    else:
        triton_note = "TRITON_INTERPRET=1 已设置（triton 解释模式）"
except ImportError:
    triton_note = "triton 未安装；本案例不直接依赖 triton"

print(f"  Triton: {triton_note}")

# ── 初始化场 ──────────────────────────────────────────────────────────
dt = torch.float32
y, x = torch.meshgrid(
    torch.arange(N, device=DEVICE, dtype=dt),
    torch.arange(N, device=DEVICE, dtype=dt),
    indexing="ij",
)
ux0 = -U0 * torch.cos(k * x) * torch.sin(k * y)
uy0 = +U0 * torch.sin(k * x) * torch.cos(k * y)
rho = torch.ones_like(ux0)
f = equilibrium(rho, ux0, uy0)

mass0 = float(f.sum().item())

# ── 主循环（eager 模式，SDAA 上 torch.compile 尚不完整）────────────────
def _step(f):
    return collide_bgk(stream(f), tau)

# route_step 参数: (step_fn, mode, *, name, ...)
# mode=None = eager（compile_step passthrough）
step_fn = route_step(_step, mode=None, name=f"taylor_green_2d[N{N}]")

times: list[int] = []
energies: list[float] = []
umaxs: list[float] = []

wall0 = time.time()
for step in range(1, steps + 1):
    f = step_fn(f)
    if step % RECORD_EVERY == 0:
        _, uxm, uym = macroscopic(f)
        e = float((0.5 * (uxm * uxm + uym * uym)).mean().item())
        um = float((uxm * uxm + uym * uym).sqrt().max().item())
        times.append(step)
        energies.append(e)
        umaxs.append(um)
wall = time.time() - wall0

# ── 拟合 ──────────────────────────────────────────────────────────────
t_arr = np.asarray(times, dtype=np.float64)
lnE = np.log(np.asarray(energies, dtype=np.float64))
a_e, b_e = np.polyfit(t_arr, lnE, 1)
gamma_e_sim = float(-a_e)
resid = lnE - (b_e + a_e * t_arr)
r2 = float(1.0 - np.sum(resid**2) / np.sum((lnE - lnE.mean()) ** 2))

lnU = np.log(np.asarray(umaxs, dtype=np.float64))
a_u, _ = np.polyfit(t_arr, lnU, 1)
gamma_vel_sim = float(-a_u)

mass_end = float(f.sum().item())

err_e_pct = float((gamma_e_sim - gamma_e_theory) / gamma_e_theory * 100.0)
err_vel_pct = float((gamma_vel_sim - gamma_vel_theory) / gamma_vel_theory * 100.0)

# ── 判据比对 ──────────────────────────────────────────────────────────
criteria = {
    "gamma_E_deviation_pct": {
        "value": float(abs(err_e_pct)),
        "threshold": 1.0,
        "unit": "%",
        "pass": bool(abs(err_e_pct) <= 1.0),
    },
    "gamma_vel_deviation_pct": {
        "value": float(abs(err_vel_pct)),
        "threshold": 1.0,
        "unit": "%",
        "pass": bool(abs(err_vel_pct) <= 1.0),
    },
    "r2_fit": {
        "value": float(r2),
        "threshold": 0.999,
        "unit": "",
        "pass": bool(r2 >= 0.999),
    },
}

all_pass = all(c["pass"] for c in criteria.values())

# ── 组装结果 ──────────────────────────────────────────────────────────
result = {
    "case": "taylor_green_2d",
    "device_evidence": {
        "device": DEVICE,
        "device_name": DEVICE_NAME,
        "device_count": DEVICE_COUNT,
        "total_memory_miB": DEVICE_TOTAL_MEM,
        "torch_version": torch.__version__,
        "torch_sdaa_version": TORCH_SDAA_VERSION,
    },
    "config": {
        "n": N,
        "re": RE,
        "u0": U0,
        "dtype": DTYPE,
        "steps": steps,
        "record_every": RECORD_EVERY,
        "nu": float(nu),
        "tau": float(tau),
        "k": float(k),
    },
    "theory": {
        "gamma_E_theory": float(gamma_e_theory),
        "gamma_vel_theory": float(gamma_vel_theory),
    },
    "simulation": {
        "gamma_E_sim": gamma_e_sim,
        "gamma_vel_sim": gamma_vel_sim,
        "err_E_pct": err_e_pct,
        "err_vel_pct": err_vel_pct,
        "r2": r2,
        "mass_drift_rel": float((mass_end - mass0) / mass0),
        "n_samples": len(times),
        "wall_sec": float(wall),
    },
    "criteria": {
        name: {
            "value": c["value"],
            "threshold": c["threshold"],
            "unit": c["unit"],
            "PASS": c["pass"],
        }
        for name, c in criteria.items()
    },
    "overall": "PASS" if all_pass else "FAIL",
    "triton_note": triton_note,
    "compile_mode_effective": "eager",
    "compile_status": "eager (SDAA torch.compile 后端不完整，使用 eager 路径)",
}

# ── 输出 ──────────────────────────────────────────────────────────────
artifacts_dir = REPO / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)
out_path = artifacts_dir / "sdaa_taylor_green_2d.json"
with open(out_path, "w") as fh:
    json.dump(result, fh, indent=2, ensure_ascii=False)

print()
print("=== 结果 ===")
print(f"  gamma_E_sim={gamma_e_sim:.6e}  theory={gamma_e_theory:.6e}  err_E={err_e_pct:+.4f}%")
print(f"  gamma_vel_sim={gamma_vel_sim:.6e}  theory={gamma_vel_theory:.6e}  err_vel={err_vel_pct:+.4f}%")
print(f"  R2={r2:.6f}")
print(f"  mass_drift_rel={(mass_end - mass0) / mass0:.2e}")
print(f"  wall={wall:.1f}s")

criteria_strs = []
for k, v in criteria.items():
    status = "PASS" if v["pass"] else "FAIL"
    criteria_strs.append(f"{k}: {status}")
print(f"  判据: {', '.join(criteria_strs)}")
print(f"  总体: {'PASS' if all_pass else 'FAIL'}")
print(f"  -> {out_path}")
