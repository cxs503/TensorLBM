#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B3D: 3D 后向台阶 Re=100 (U_max·H/ν) 展向周期再附着长度验证 (D3Q19)。

参考: Armaly, Durst, Pereira & Schoenung, JFM 127:473-496 (1983):
      Re=100 (基于最大入口速度 U_max 与台阶高 h), 膨胀比 ER≈1.94,
      再附着长度 X_r/h ≈ 3.0 (容差 ±3%)。

几何 (OpenLB bstep3d 同款思路, 展向周期):
    x: 流动方向;  y: 竖直方向 (上/下壁);  z: 展向 (周期, stream 天然周期)。
    台阶块: (x < x_step) & (y < step_h), 全 z 层。y=0 (x≥x_step) 与 y=ny-1 为壁面。
    ER = (ny-2)/(ny-1-step_h)  (2D 模块同款格定义)。
    两档网格: H=40 -> (nx,ny,nz)=(400,82,80),  ER=1.9512;
              H=60 -> (600,123,120),            ER=1.9516  (ER 恒定, 几何相似)。

入口: 充分发展抛物线 u(y)=4·U_max·yp·(H_up-yp)/H_up², yp=y-(step_h-0.5),
      H_up=ny-1-step_h; 经 3D Zou/He 速度 BC 逐点施加 (公式逐点镜像库函数
      boundaries3d.zou_he_inlet_velocity_3d, 仅将标量 u_in 场化为 (nz,ny) 剖面;
      库函数不支持剖面 -> 共性模块缺口, 见 /tmp/bstep3d_gap.md)。
出口: Zou/He 压力出口 rho=1 (公式同库 zou_he_outlet_pressure_3d)。
壁面/台阶: 库 bounce_back_cells_3d。流场: 库 stream3d_roll (周期)。
碰撞: 库 collide_bgk3d (默认) 或 collide_mrt3d (--mrt), torch.compile 加速。

性能说明 (H=40, RTX 3090): torch.compile(max-autotune) 编译 collide 与
inlet/outlet BC (消除 CPU launch 开销), stream3d_roll 替代 gather 版 stream3d
(避免 4×19×N 索引缓存), d3q19.C 常驻 fp32 CUDA (int64 广播乘法带宽减半)。
约 12 ms/步 (H=40) / 40 ms/步 (H=60)。

Re = U_max·step_h/ν = 100 (Armaly 定义)。ν = U_max·H/Re, τ = 3ν + 0.5。

再附着点: 台阶后壁面剪切 τ_w ∝ ux(y=1) 过零点 (线性亚格插值), 距台阶下游立面
          x = x_step-0.5 归一化: X_r/h。主指标 = 展向平均场测量值; 另报告
          各 z 平面 xr(z) 的 std (展向均匀性)。

判定: 真实模拟 (无外推), |X_r/h - 3.0|/3.0 ≤ 3% 且两档网格 (H=40/60) 收敛
      (细网格 X_r/h 与粗网格差 < 0.15 且细网格误差不劣于粗网格)。

用法:
    python run.py --h 40 [--steps 300000] [--quick N] [--device cuda:0] [--mrt]
    python run.py --scan [--steps ...] [--device0 cuda:0] [--device1 cuda:1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

import numpy as np
import torch

import tensorlbm.d3q19 as _d3q19
import tensorlbm.boundaries3d as _b3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d_roll, collide_bgk3d, collide_mrt3d
from tensorlbm.boundaries3d import bounce_back_cells_3d

# ---------------------------------------------------------------------------
# 全局性能 patch (基准脚本内, 不改库): C/OPPOSITE 常驻 fp32 CUDA
# ---------------------------------------------------------------------------
def _patch_lattice_constants(device: torch.device) -> None:
    """d3q19.C 转 fp32 常驻 CUDA (int64 广播乘法带宽减半, 速度分量 ±1/0 精确)。

    OPPOSITE 常驻 CUDA 使库 bounce_back_cells_3d 的 .to(device) 变为 no-op
    (消除每步 H2D 拷贝)。仅在 benchmark 进程内生效。
    """
    _d3q19.C = _d3q19.C.to(device=device, dtype=torch.float32)
    _d3q19.OPPOSITE = _d3q19.OPPOSITE.to(device)
    _b3d.OPPOSITE = _d3q19.OPPOSITE


# D3Q19 入口 (cx>0 未知) / 出口 (cx<0 未知) 方向 (与库 boundaries3d 一致)
_INLET_DIRS = [1, 7, 9, 11, 13]
_INLET_OPP = [2, 8, 10, 12, 14]
_OUTLET_DIRS = [2, 8, 10, 12, 14]
_OUTLET_OPP = [1, 7, 9, 11, 13]

CS2 = 1.0 / 3.0
X_R_REF = 3.0          # Armaly Re=100 X_r/h 参考值
ER_REF = 1.94          # Armaly 实验膨胀比


# ---------------------------------------------------------------------------
# 几何
# ---------------------------------------------------------------------------
def grid_dims(h: int) -> tuple[int, int, int, int]:
    """两档几何相似网格: H=40 -> (400,82,80), H=60 -> (600,123,120)。

    ny = round(2.05*H) 使 ER=(ny-2)/(ny-1-H) 恒定 (~1.951, 接近 Armaly 1.94);
    nx = 10H (上游 x_step=2H, 下游 8H); nz = 2H (展向周期宽度)。
    """
    ny = int(round(2.05 * h))
    nx = 10 * h
    nz = 2 * h
    x_step = 2 * h
    return nx, ny, nz, x_step


def make_bfs_solid_mask_3d(
    nz: int, ny: int, nx: int, step_h: int, x_step: int, device: torch.device,
) -> torch.Tensor:
    """3D 后向台阶固体掩码 (nz, ny, nx): 顶壁 + 台阶后底壁 + 台阶块。z 向无壁 (周期)。"""
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, -1, :] = True                              # top wall (y=ny-1)
    solid[:, 0, x_step:] = True                         # bottom wall after step
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device),
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij",
    )
    solid |= (xx < x_step) & (yy < step_h)              # step solid block
    return solid


def make_parabolic_profile(ny: int, step_h: int, u_max: float) -> np.ndarray:
    """充分发展抛物线剖面 (上游通道壁面在 y=step_h-0.5 与 y=ny-0.5, 高 H_up=ny-1-step_h)。

    峰值精确 = u_max (4·yp·(H-yp)/H² 在 yp=H/2 处 = 1)。"""
    H = float(ny - 1 - step_h)
    yp = np.arange(ny, dtype=np.float64) - (step_h - 0.5)
    u = np.zeros(ny, dtype=np.float64)
    inside = (yp >= 0.0) & (yp <= H)
    u[inside] = 4.0 * u_max * yp[inside] * (H - yp[inside]) / (H * H)
    return u


# ---------------------------------------------------------------------------
# 入口/出口 BC: 3D Zou/He, 逐点剖面版 (公式镜像库函数, 无外推)
# ---------------------------------------------------------------------------
def inlet_bc(f: torch.Tensor, ux_field: torch.Tensor) -> torch.Tensor:
    """Zou/He 速度入口 BC at x=0, 逐点 ux = ux_field (nz, ny)。

    逐点镜像库 boundaries3d.zou_he_inlet_velocity_3d (Zou & He 1997 非平衡
    反弹重构): rho = (sum_cz0 + 2·sum_cx_neg)/(1-ux), 未知方向
    f[d] = feq[d] - feq[opp] + f[opp]。逐方向 setitem 便于 torch.compile。
    """
    sum_cx0 = (
        f[0, :, :, 0] + f[3, :, :, 0] + f[4, :, :, 0]
        + f[5, :, :, 0] + f[6, :, :, 0]
        + f[15, :, :, 0] + f[16, :, :, 0] + f[17, :, :, 0] + f[18, :, :, 0]
    )
    sum_cx_neg = (
        f[2, :, :, 0] + f[8, :, :, 0] + f[10, :, :, 0]
        + f[12, :, :, 0] + f[14, :, :, 0]
    )
    rho = (sum_cx0 + 2.0 * sum_cx_neg) / (1.0 - ux_field)   # (nz, ny)
    feq = equilibrium3d(
        rho.unsqueeze(-1), ux_field.unsqueeze(-1),
        torch.zeros_like(rho).unsqueeze(-1), torch.zeros_like(rho).unsqueeze(-1),
        device=f.device,
    )  # (19, nz, ny, 1)
    for d, od in zip(_INLET_DIRS, _INLET_OPP):
        f[d, :, :, 0] = feq[d, :, :, 0] - feq[od, :, :, 0] + f[od, :, :, 0]
    return f


def outlet_pressure_bc(f: torch.Tensor, rho_out: float = 1.0) -> torch.Tensor:
    """Zou/He 压力出口 BC at x=nx-1 (rho=rho_out), 公式同库
    boundaries3d.zou_he_outlet_pressure_3d: ux_out = -1 + (sum_cx0+2·sum_cx_pos)/rho_out。
    """
    sum_cx0 = (
        f[0, :, :, -1] + f[3, :, :, -1] + f[4, :, :, -1]
        + f[5, :, :, -1] + f[6, :, :, -1]
        + f[15, :, :, -1] + f[16, :, :, -1] + f[17, :, :, -1] + f[18, :, :, -1]
    )
    sum_cx_pos = (
        f[1, :, :, -1] + f[7, :, :, -1] + f[9, :, :, -1]
        + f[11, :, :, -1] + f[13, :, :, -1]
    )
    ux_out = -1.0 + (sum_cx0 + 2.0 * sum_cx_pos) / rho_out   # (nz, ny)
    rho_field = torch.full_like(ux_out, rho_out)
    feq = equilibrium3d(
        rho_field.unsqueeze(-1), ux_out.unsqueeze(-1),
        torch.zeros_like(rho_field).unsqueeze(-1), torch.zeros_like(rho_field).unsqueeze(-1),
        device=f.device,
    )
    for d, od in zip(_OUTLET_DIRS, _OUTLET_OPP):
        f[d, :, :, -1] = feq[d, :, :, 0] - feq[od, :, :, 0] + f[od, :, :, -1]
    return f


# ---------------------------------------------------------------------------
# 再附着长度测量 (壁面剪切过零亚格插值, 展向平均 + 展向散布)
# ---------------------------------------------------------------------------
def _zero_crossing_col(row: np.ndarray, x_step: int) -> float | None:
    """row = ux[1, x_step:]; 返回再附着点连续列坐标 (负→正过零点, 线性插值)。"""
    pos = int(np.argmax(row > 0.0))
    if pos == 0 or row[pos] <= 0.0:
        return None
    a, b = row[pos - 1], row[pos]
    if b - a == 0.0:
        return None
    return x_step + (pos - 1) + (0.0 - a) / (b - a)


def measure_reattach_3d(
    ux: torch.Tensor, x_step: int, step_h: int,
) -> dict[str, float]:
    """X_r/h: 距台阶下游立面 x=x_step-0.5 的归一化再附着长度。

    - xr_h: 展向平均场 (mean over z) 的过零插值 —— 等价 2D 观测量;
    - xr_planes_*: 各 z 平面 xr(z) 的均值/标准差 (展向均匀性)。
    """
    nz = ux.shape[0]
    ux_np = ux.detach().cpu().numpy()
    xr_planes: list[float] = []
    for z in range(nz):
        zc = _zero_crossing_col(ux_np[z, 1, x_step:], x_step)
        xr_planes.append(0.0 if zc is None else float((zc - (x_step - 0.5)) / step_h))
    xr_arr = np.array(xr_planes)
    row_mean = ux_np[:, 1, x_step:].mean(axis=0)
    zc_mean = _zero_crossing_col(row_mean, x_step)
    xr_main = 0.0 if zc_mean is None else float((zc_mean - (x_step - 0.5)) / step_h)
    return {
        "xr_h": xr_main,
        "xr_planes_mean": float(xr_arr.mean()),
        "xr_planes_std": float(xr_arr.std()),
        "xr_planes_min": float(xr_arr.min()),
        "xr_planes_max": float(xr_arr.max()),
    }


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------
def inlet_profile_check(ux: torch.Tensor, step_h: int, u_max: float) -> dict[str, float]:
    """核对 x=0 列实际施加剖面 (展向平均) 与目标抛物线的偏差。"""
    ny = ux.shape[1]
    target = make_parabolic_profile(ny, step_h, u_max)
    actual = ux[:, :, 0].mean(dim=0).detach().cpu().numpy()   # mean over z
    fluid = np.arange(step_h, ny - 1)
    dev = np.abs(actual[fluid] - target[fluid]).max() / u_max
    q_num = float(actual[fluid].sum())
    q_ana = (2.0 / 3.0) * u_max * (ny - 1 - step_h)
    return {"max_abs_dev_over_umax": float(dev), "flux_ratio": q_num / q_ana}


def spanwise_uniformity(ux: torch.Tensor, solid: torch.Tensor,
                        u_max: float) -> dict[str, float]:
    """展向均匀性: 下游区 (x 从 0.6·nx 到 nx-2) 流体场 ux 的 z 向相对散布。"""
    u = ux.detach().cpu().numpy()
    s = solid.detach().cpu().numpy()
    x0, x1 = int(0.6 * u.shape[2]), u.shape[2] - 2
    seg = u[:, 1:-1, x0:x1].copy()
    seg[s[:, 1:-1, x0:x1]] = np.nan
    mean_z = np.nanmean(seg, axis=0)
    rel_dev = np.nanmax(np.abs(seg - mean_z) / max(abs(u_max), 1e-12))
    return {"max_z_rel_dev_over_umax": float(rel_dev)}


def separation_bubble_diag(ux: torch.Tensor, x_step: int, step_h: int,
                           u_max: float) -> dict[str, float]:
    """分离泡诊断: 台阶下游最大回流强度及其位置 (展向平均场, 距台阶立面归一化)。"""
    u = ux.detach().cpu().numpy().mean(axis=0)   # mean over z
    bubble = u[1:, x_step:]
    min_ux = float(bubble.min())
    ys, xs = np.where(bubble == bubble.min())
    return {
        "max_backflow_over_umax": min_ux / u_max,
        "backflow_x_rel_h": float(xs[0]) / step_h,
    }


# ---------------------------------------------------------------------------
# 单档模拟
# ---------------------------------------------------------------------------
def run_case(h: int, steps: int, out_interval: int, device: torch.device,
             u_max: float, re: float, use_mrt: bool, out_dir: Path,
             quick: int = 0, do_compile: bool = True) -> dict:
    nx, ny, nz, x_step = grid_dims(h)
    er = (ny - 2) / (ny - 1 - h)
    nu = u_max * h / re
    tau = 3.0 * nu + 0.5

    torch.manual_seed(0)
    _patch_lattice_constants(device)
    solid = make_bfs_solid_mask_3d(nz, ny, nx, h, x_step, device)

    # 初始化: rho=1 + 抛物线剖面 (全 x), uy=uz=0
    prof = make_parabolic_profile(ny, h, u_max)
    ux0 = torch.tensor(prof, dtype=torch.float32, device=device)
    ux0 = ux0.view(1, ny, 1).expand(nz, ny, nx).contiguous()
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0 = ux0.masked_fill(solid, 0.0)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(f.sum().item())

    # 入口剖面场 (nz, ny) 张量 (展向均匀, y 依赖)
    u_in_field = torch.tensor(
        np.broadcast_to(prof.reshape(1, ny), (nz, ny)).astype(np.float32),
        device=device,
    )

    if use_mrt:
        collide = collide_mrt3d
    else:
        collide = collide_bgk3d

    if do_compile:
        collide = torch.compile(collide, mode="max-autotune")
        inlet_c = torch.compile(inlet_bc)
        outlet_c = torch.compile(outlet_pressure_bc)
        bounce_c = torch.compile(bounce_back_cells_3d)
    else:
        inlet_c, outlet_c, bounce_c = inlet_bc, outlet_pressure_bc, bounce_back_cells_3d

    def step(f: torch.Tensor) -> torch.Tensor:
        f = collide(f, tau=tau)
        f = stream3d_roll(f)
        f = inlet_c(f, u_in_field)
        f = outlet_c(f, 1.0)
        f = bounce_c(f, solid)
        return f

    t0 = time.time()
    diagnostics: list[dict] = []
    for s in range(1, steps + 1):
        f = step(f)
        if s % out_interval == 0 or s == steps:
            rho, ux, uy, uz = macroscopic3d(f)
            ux = ux.masked_fill(solid, 0.0)
            uy = uy.masked_fill(solid, 0.0)
            uz = uz.masked_fill(solid, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy + uz * uz)
            meas = measure_reattach_3d(ux, x_step, h)
            diag = {
                "step": s,
                "mass_drift": float(rho.sum().item()) - initial_mass,
                "max_speed": float(speed.max().item()),
                "xr_h": meas["xr_h"],
                "xr_z_std": meas["xr_planes_std"],
            }
            diagnostics.append(diag)
            print(
                f"  step={s:6d} mass_drift={diag['mass_drift']:+.3e} "
                f"max|u|={diag['max_speed']:.5f} xr/h={meas['xr_h']:.4f} "
                f"(z-std {meas['xr_planes_std']:.4f})",
                flush=True,
            )
            if quick and s >= quick:
                break

    wall_t = time.time() - t0
    rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
    ux_f = ux_f.masked_fill(solid, 0.0)

    xr_series = [d["xr_h"] for d in diagnostics]
    steps_arr = [d["step"] for d in diagnostics]
    last3 = xr_series[-3:]
    converged = (
        len(last3) >= 3 and (max(last3) - min(last3)) <= 0.02
        and abs(last3[-1] - last3[-2]) <= 0.01
    )

    inlet_diag = inlet_profile_check(ux_f, h, u_max)
    span_diag = spanwise_uniformity(ux_f, solid, u_max)
    bubble_diag = separation_bubble_diag(ux_f, x_step, h, u_max)
    meas_final = measure_reattach_3d(ux_f, x_step, h)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"final_ux_H{h}.npz",
        ux=ux_f.detach().cpu().numpy(),
        solid=solid.detach().cpu().numpy(),
    )

    result = {
        "case": "bstep_3d",
        "name": f"backward_facing_step_3d_re100_H{h}",
        "reference": {
            "source": "Armaly, Durst, Pereira & Schoenung, JFM 127:473-496 (1983)",
            "re_definition": "Re = U_max * step_h / nu (最大入口速度, 台阶高)",
            "xr_h_ref": X_R_REF,
            "er_ref": ER_REF,
        },
        "geometry": {
            "nx": nx, "ny": ny, "nz": nz, "step_h": h, "x_step": x_step,
            "expansion_ratio": er,
            "er_dev_pct": (er - ER_REF) / ER_REF * 100.0,
            "spanwise": f"periodic, Lz={nz} = {nz / h:.1f}H",
            "inlet": f"fully_developed_parabolic (3D Zou/He profile), U_max={u_max}",
            "outlet": "Zou/He pressure (rho=1)",
        },
        "physics": {
            "re": re, "nu": nu, "tau": tau,
            "collision": "mrt" if use_mrt else "bgk",
            "lattice": "D3Q19",
        },
        "result": {
            "xr_h": meas_final["xr_h"],
            "xr_h_err_pct": (meas_final["xr_h"] - X_R_REF) / X_R_REF * 100.0,
            "xr_planes_mean": meas_final["xr_planes_mean"],
            "xr_planes_std": meas_final["xr_planes_std"],
            "xr_planes_min": meas_final["xr_planes_min"],
            "xr_planes_max": meas_final["xr_planes_max"],
            "xr_series": xr_series,
            "steps": steps_arr,
            "converged": bool(converged),
            "n_steps_run": diagnostics[-1]["step"] if diagnostics else 0,
            "wall_time_s": round(wall_t, 1),
            "device": str(device),
            "inlet_profile_check": inlet_diag,
            "spanwise_uniformity": span_diag,
            "separation_bubble": bubble_diag,
            "mass_drift_final": diagnostics[-1]["mass_drift"] if diagnostics else 0.0,
            "finite": bool(torch.isfinite(f).all().item()),
        },
    }
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="3D backward-facing step benchmark")
    ap.add_argument("--h", type=int, default=40, help="step height in cells (40/60)")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("B3D_STEPS", 300000)))
    ap.add_argument("--interval", type=int, default=10000)
    ap.add_argument("--quick", type=int, default=0, help="quick sanity run (steps)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mrt", action="store_true", help="use MRT collision instead of BGK")
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    ap.add_argument("--u-max", type=float, default=0.05)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--out", default="/tmp/bstep3d_runs")
    ap.add_argument("--verified", default="/home/wxsc/cxs/TensorLBM/benchmarks/verified/bstep_3d")
    ap.add_argument("--scan", action="store_true", help="run H=40 and H=60, write convergence summary")
    ap.add_argument("--device0", default="cuda:0")
    ap.add_argument("--device1", default="cuda:1")
    args = ap.parse_args()

    err_tol_pct = 3.0

    if args.scan:
        jobs = [(40, torch.device(args.device0)), (60, torch.device(args.device1))]
    else:
        jobs = [(args.h, torch.device(args.device))]

    cases = []
    for h, device in jobs:
        steps = args.quick if args.quick else args.steps
        print(f"=== B3D BFS Re={args.re} H={h} steps={steps} device={device} "
              f"collision={'mrt' if args.mrt else 'bgk'} "
              f"compile={'on' if not args.no_compile else 'off'} ===", flush=True)
        r = run_case(h, steps, args.interval, device, args.u_max, args.re,
                     args.mrt, Path(args.out), quick=args.quick,
                     do_compile=not args.no_compile)
        cases.append(r)
        print(f"H={h}: X_r/h = {r['result']['xr_h']:.4f} "
              f"err {r['result']['xr_h_err_pct']:+.2f}% "
              f"converged={r['result']['converged']} wall={r['result']['wall_time_s']:.0f}s",
              flush=True)

    vdir = Path(args.verified)
    vdir.mkdir(parents=True, exist_ok=True)
    for c in cases:
        h = c["geometry"]["step_h"]
        with (vdir / f"result_H{h}.json").open("w", encoding="utf-8") as fh:
            json.dump(c, fh, indent=2, ensure_ascii=False, default=float)

    if args.scan and len(cases) >= 2:
        errs = [c["result"]["xr_h_err_pct"] for c in cases]
        xrs = [c["result"]["xr_h"] for c in cases]
        grid_converged = abs(xrs[1] - xrs[0]) < 0.15 and errs[1] <= errs[0] + 1e-12
        per_case_ok = all(c["result"]["converged"] for c in cases)
        passed = all(abs(e) <= err_tol_pct for e in errs) and grid_converged and per_case_ok
        summary = {
            "case": "bstep_3d_convergence",
            "name": "backward_facing_step_3d_re100_spanwise_periodic",
            "reference_xr_h": X_R_REF,
            "err_tol_pct": err_tol_pct,
            "grids": [c["geometry"] for c in cases],
            "xr_h": xrs,
            "err_pct": errs,
            "grid_converged": bool(grid_converged),
            "per_case_converged": [bool(c["result"]["converged"]) for c in cases],
            "spanwise_z_std": [c["result"]["xr_planes_std"] for c in cases],
            "verified": bool(passed),
            "notes": (
                "判定: 两档网格 |err|<=3% 且细网格 X_r/h 与粗网格差 <0.15 且"
                "细网格误差不劣于粗网格, 且各档稳态收敛。"
            ),
        }
        with (vdir / "result.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False, default=float)
        print(f"=== scan summary: xr_h = {[round(v, 4) for v in xrs]}, "
              f"err% = {[round(v, 2) for v in errs]}, "
              f"grid_converged={grid_converged}, verified={passed} ===", flush=True)
        print(f"-> {vdir / 'result.json'}", flush=True)
    else:
        h = cases[0]["geometry"]["step_h"]
        print(f"-> {vdir / ('result_H%d.json' % h)}", flush=True)


if __name__ == "__main__":
    main()
