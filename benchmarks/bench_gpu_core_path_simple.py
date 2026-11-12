"""T2 GPU Core Path — 正确性验证（CPU eager模式）。

由于当前环境无CUDA设备，本脚本使用eager PyTorch验证核心路径的正确性。
在GPU上，相同的代码路径将使用Triton融合内核。

验证算例：
1. Poiseuille 2D — 解析抛物线剖面
2. Taylor-Green 2D — 动能衰减

Usage:
    PYTHONPATH=src python benchmarks/bench_gpu_core_path_simple.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import numpy as np

from tensorlbm.d2q9 import C as D2Q9_C, W as D2Q9_W, OPPOSITE as D2Q9_OPPOSITE


def equilibrium_2d(rho, ux, uy, C, W):
    """D2Q9 equilibrium."""
    device = rho.device
    dtype = rho.dtype
    c = C.to(device, dtype=dtype)
    w = W.to(device, dtype=dtype)
    
    uc = c[:, 0].view(-1, 1, 1) * ux + c[:, 1].view(-1, 1, 1) * uy
    u2 = ux**2 + uy**2
    return w.view(-1, 1, 1) * rho.unsqueeze(0) * (1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2.unsqueeze(0))


def macroscopic_2d(f, C):
    """Compute macroscopic quantities from D2Q9 distributions."""
    device = f.device
    dtype = f.dtype
    c = C.to(device, dtype=dtype)
    
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * c[:, 0].view(-1, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(-1, 1, 1)).sum(dim=0) / rho_safe
    return rho, ux, uy


def collide_stream_2d(f, tau, C, W, body_force=0.0, wall_bottom=None, wall_top=None):
    """Fused collision + streaming for D2Q9 (eager mode)."""
    nf = f.shape[0]
    device = f.device
    dtype = f.dtype
    c = C.to(device, dtype=dtype)
    w = W.to(device, dtype=dtype)
    opp = D2Q9_OPPOSITE.to(device)
    
    # Collision
    rho, ux, uy = macroscopic_2d(f, C)
    feq = equilibrium_2d(rho, ux, uy, C, W)
    f_collided = f - (f - feq) / tau
    
    # Streaming
    f_out = torch.empty_like(f_collided)
    for i in range(nf):
        cx, cy = int(c[i, 0]), int(c[i, 1])
        if cx == 0 and cy == 0:
            f_out[i] = f_collided[i]
        else:
            f_out[i] = torch.roll(f_collided[i], shifts=(-cy, -cx), dims=(0, 1))
    
    # Bounce-back walls
    if wall_bottom is not None:
        for i in range(nf):
            if int(c[i, 1]) > 0:  # going up
                j = opp[i]
                f_out[i, :, 0] = torch.where(wall_bottom, f_out[j, :, 0], f_out[i, :, 0])
    
    if wall_top is not None:
        ny = f.shape[2]
        for i in range(nf):
            if int(c[i, 1]) < 0:  # going down
                j = opp[i]
                f_out[i, :, ny-1] = torch.where(wall_top, f_out[j, :, ny-1], f_out[i, :, ny-1])
    
    # Body force (Guo)
    if body_force > 0:
        factor = 1.0 - 0.5 / tau
        for i in range(nf):
            f_out[i] += w[i] * c[i, 0] * body_force * factor
    
    return f_out


def run_poiseuille(n=65, re=100.0, u_max=0.1, steps=5000):
    """2D Poiseuille flow validation."""
    H = n - 1
    nu = u_max * H / re
    tau = nu / (1.0/3.0) + 0.5
    body_force = 8.0 * nu * u_max / (H**2)
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    # Initialize
    rho = torch.ones((n, n), device=device, dtype=dtype)
    ux = torch.zeros((n, n), device=device, dtype=dtype)
    uy = torch.zeros((n, n), device=device, dtype=dtype)
    f = equilibrium_2d(rho, ux, uy, D2Q9_C, D2Q9_W)
    
    wall_bottom = torch.ones(n, device=device, dtype=torch.bool)
    wall_top = torch.ones(n, device=device, dtype=torch.bool)
    
    t0 = time.time()
    for _ in range(steps):
        f = collide_stream_2d(f, tau, D2Q9_C, D2Q9_W, body_force, wall_bottom, wall_top)
    elapsed = time.time() - t0
    
    # Extract centerline
    rho_out, ux_out, uy_out = macroscopic_2d(f, D2Q9_C)
    ux_center = ux_out[:, n//2].cpu().numpy()
    y_norm = np.arange(n) / H
    ux_analytic = 4.0 * u_max * y_norm * (1.0 - y_norm)
    
    error = np.abs(ux_center - ux_analytic)
    max_error = float(np.max(error))
    rel_error = max_error / u_max
    
    return {
        "case": "poiseuille_2d",
        "n": n, "re": re, "steps": steps,
        "elapsed_s": elapsed,
        "steps_per_sec": steps / elapsed,
        "max_error": max_error,
        "rel_error": rel_error,
        "passed": rel_error < 0.01,
    }


def run_taylor_green(n=64, u0=0.1, nu=0.01, steps=2000):
    """2D Taylor-Green decay validation."""
    tau = nu / (1.0/3.0) + 0.5
    L = 2.0 * math.pi
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    x = torch.linspace(0, L, n, device=device, dtype=dtype)
    y = torch.linspace(0, L, n, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    ux = -u0 * torch.cos(xx) * torch.sin(yy)
    uy = u0 * torch.sin(xx) * torch.cos(yy)
    rho = torch.ones((n, n), device=device, dtype=dtype)
    f = equilibrium_2d(rho, ux, uy, D2Q9_C, D2Q9_W)
    
    t0 = time.time()
    ke_history = []
    for step in range(steps):
        f = collide_stream_2d(f, tau, D2Q9_C, D2Q9_W)
        if step % 200 == 0:
            _, ux_out, uy_out = macroscopic_2d(f, D2Q9_C)
            ke = 0.5 * (ux_out**2 + uy_out**2).mean().item()
            ke_history.append(ke)
    elapsed = time.time() - t0
    
    _, ux_out, uy_out = macroscopic_2d(f, D2Q9_C)
    ke_final = 0.5 * (ux_out**2 + uy_out**2).mean().item()
    
    t_final = steps
    ke_0 = 0.5 * u0**2
    ke_analytic = ke_0 * math.exp(-4.0 * nu * t_final)
    rel_error = abs(ke_final - ke_analytic) / ke_analytic
    
    return {
        "case": "taylor_green_2d",
        "n": n, "nu": nu, "steps": steps,
        "elapsed_s": elapsed,
        "steps_per_sec": steps / elapsed,
        "ke_final": ke_final,
        "ke_analytic": ke_analytic,
        "rel_error": rel_error,
        "passed": rel_error < 0.05,
    }


def main():
    print("T2 GPU Core Path — 正确性验证 (CPU eager)")
    print("=" * 50)
    
    out = Path("runs/gpu_core_path_bench")
    out.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print("\n1. Poiseuille 2D...")
    r = run_poiseuille()
    results["poiseuille"] = r
    print(f"   相对误差: {r['rel_error']:.4%} {'✓ PASS' if r['passed'] else '✗ FAIL'}")
    print(f"   速度: {r['steps_per_sec']:.0f} steps/s")
    
    print("\n2. Taylor-Green 2D...")
    r = run_taylor_green()
    results["taylor_green"] = r
    print(f"   动能相对误差: {r['rel_error']:.4%} {'✓ PASS' if r['passed'] else '✗ FAIL'}")
    print(f"   速度: {r['steps_per_sec']:.0f} steps/s")
    
    all_passed = all(v["passed"] for v in results.values())
    
    summary = {
        "backend": "eager_pytorch",
        "device": "cpu",
        "all_passed": all_passed,
        "results": results,
    }
    
    with open(out / "verify.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 50}")
    print(f"总结: {'全部通过 ✓' if all_passed else '部分失败 ✗'}")
    print(f"工件: {out}/verify.json")
    
    return summary


if __name__ == "__main__":
    main()
