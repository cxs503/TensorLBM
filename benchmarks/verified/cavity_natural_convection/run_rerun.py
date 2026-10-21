#!/usr/bin/env python3
"""自然对流方腔重跑（Ra=1e4, Pr=0.71）— 对齐 thermal_cavity 使用修复后库。

本脚本使用修复后的库（apply_temperature_boundaries v3b + mask_wall_u=True），
与 verified/thermal_cavity 相同的方法。预注册参考：de Vahl Davis 1983, Nu=2.243。

用法：
  CUDA_VISIBLE_DEVICES=5 /nfs/wangxi/venvs/tensorlbm/bin/python run_rerun.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

# 设置路径：src 和 benchmarks 目录（必须指向本 worktree，非主仓库）
this_dir = Path(__file__).resolve().parent
worktree_root = this_dir.parents[2]  # /nfs/wangxi/tensor_eco/agent_work/m76/TensorLBM
sys.path.insert(0, str(worktree_root / "src"))
sys.path.insert(0, str(worktree_root / "benchmarks"))

# 使用与 thermal_cavity 相同的 driver
sys.path.insert(0, str(this_dir))
import driver

# 预注册参考（与 thermal_cavity 同源：de Vahl Davis 1983）
RA = 1e4
PR = 0.71
TAU = 0.6
NU_REF = 2.243
U_MAX_REF = 16.178
V_MAX_REF = 19.617

# 网格档位
GRIDS = [64, 128, 256]

# 最大步数（与 thermal_cavity 一致）
MAX_STEPS = {64: 200_000, 128: 450_000, 256: 1_000_000}

# 稳态触发许可下限（N>=128 防止假触发）
MIN_STEPS = {64: 0, 128: 103_000, 256: 414_000}


def run_case(nx: int, device: torch.device, out_dir: str) -> dict:
    """运行单档网格，返回观测量 dict。"""
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    t_hot, t_cold = 1.0, 0.0
    f, g, wall, p = driver.make_fields(nx, RA, PR, TAU, device)
    tau_T, g_beta = p["tau_T"], p["g_beta"]
    
    max_steps = MAX_STEPS[nx]
    min_steps = MIN_STEPS[nx]
    every = 1000
    
    # 记录 thermal 库 hash
    import tensorlbm.thermal as th_mod
    th_sha = hashlib.sha256(open(th_mod.__file__, "rb").read()).hexdigest()[:12]
    
    log_path = f"{out_dir}/case_ra1e4_n{nx}.log"
    log = open(log_path, "w", buffering=1)
    log.write(f"# case ra1e4_n{nx} thermal={th_mod.__file__} sha12={th_sha}\n")
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
                0.0 if prev is None
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
                    log.write(f"# steady trigger at {step}; verification segment to {target}\n")
    
    elapsed = time.time() - t0
    
    # 最终观测量
    obs = driver.observables(f, g, p, t_hot, t_cold)
    nu_at_stop = next((h["nu"] for h in reversed(hist) if h["step"] <= t_stop), None) if t_stop else None
    
    result = {
        "tag": f"ra1e4_n{nx}",
        "ra": RA,
        "nx": nx,
        "dtype": "float32",
        "steps_run": step,
        "max_steps": max_steps,
        "min_steps": min_steps,
        "steady_triggered": t_stop is not None,
        "t_stop": t_stop,
        "verification_steps": target - (t_stop or 0),
        "thermal_sha12": th_sha,
        "elapsed_s": round(elapsed, 1),
        **obs,
        "nu_history": hist,
    }
    
    log.write(f"# final: nu={obs['nu']:.8f} u_max_nd={obs['u_max_nd']:.6f} v_max_nd={obs['v_max_nd']:.6f}\n")
    log.close()
    
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = str(Path(__file__).resolve().parent)
    
    print(f"Running cavity_natural_convection Ra=1e4 on {device}")
    print(f"Grids: {GRIDS}")
    print(f"Reference: Nu={NU_REF}, u_max={U_MAX_REF}, v_max={V_MAX_REF}")
    print()
    
    results = []
    for nx in GRIDS:
        print(f"=== Grid N={nx} ===")
        res = run_case(nx, device, out_dir)
        results.append(res)
        
        nu_err = 100 * (res["nu"] - NU_REF) / NU_REF
        u_err = 100 * (res["u_max_nd"] - U_MAX_REF) / U_MAX_REF
        v_err = 100 * (res["v_max_nd"] - V_MAX_REF) / V_MAX_REF
        
        print(f"  Nu = {res['nu']:.6f} (err = {nu_err:+.3f}%)")
        print(f"  u_max = {res['u_max_nd']:.6f} (err = {u_err:+.3f}%)")
        print(f"  v_max = {res['v_max_nd']:.6f} (err = {v_err:+.3f}%)")
        print(f"  Steps: {res['steps_run']} (steady at {res['t_stop']})")
        print()
    
    # 保存结果
    output = {
        "case": "cavity_natural_convection_ra1e4_rerun",
        "title": "de Vahl Davis natural convection square cavity (Ra=1e4, Pr=0.71) — rerun with fixed library",
        "lattice": "D2Q9 (velocity) + D2Q5 (temperature)",
        "collision": "bgk + Guo buoyancy",
        "pr": PR,
        "tau": TAU,
        "extrap": "none",
        "driver_scheme": "wall-node u=0 mask before temperature_collision (mask_wall_u=True, aligned with thermal_cavity)",
        "reference": {
            "source": "de Vahl Davis 1983 Int. J. Numer. Methods Fluids 3:249-264",
            "Nu": NU_REF,
            "u_max": U_MAX_REF,
            "v_max": V_MAX_REF,
        },
        "grids": results,
    }
    
    out_path = Path(out_dir) / "result_rerun.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {out_path}")
    
    # 收敛性检查
    nu_errs = [abs(100 * (r["nu"] - NU_REF) / NU_REF) for r in results]
    monotone = all(nu_errs[i] > nu_errs[i+1] for i in range(len(nu_errs)-1))
    passing = [i for i, e in enumerate(nu_errs) if e <= 3.0]
    
    print("\n=== Summary ===")
    print(f"Nu errors: {[f'{e:.2f}%' for e in nu_errs]}")
    print(f"Monotone convergence: {monotone}")
    print(f"Passing grids (<=3%): {[GRIDS[i] for i in passing]}")
    
    if len(passing) >= 2 and monotone:
        print("\n✓ PASS: ≥2 grids pass with monotone convergence")
    else:
        print("\n✗ FAIL: does not meet closure criteria")


if __name__ == "__main__":
    main()
