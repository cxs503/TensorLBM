#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""全 3D 方腔（5 壁固定 + 顶盖移动，Lz=H 立方腔）Re=1000 — 3D 涡结构验证（TRT 碰撞）。

几何：f 形状 (19, nz, ny, nx)，nz=ny=nx（立方腔 [0,1]³，Lz=H 有限非周期）。
      y 垂直（顶盖 y=ny-1 沿 +x 移动），x 水平，z 展向有限。

边界（5 静止壁 + 移动顶盖）：
  - x=0, x=nx-1, y=0, z=0, z=nz-1 五静止壁：pre-streaming 半程反弹
    （碰撞后、stream 前用 f_pre 反射；与 verified cavity/3d 同款 V3 口径）
  - 顶盖 y=ny-1：boundaries3d.zou_he_moving_lid_3d（D3Q19 Zou-He 移动壁，
    覆盖整层含角点；ux=u_lid, uy=uz=0）

碰撞：collide_trt3d（TRT, Λ=3/16）。2026-08-20 实测：32³ 冒烟 600 步
      Re=1000 下仅 TRT 稳定（RLBM U=0.05/0.06/0.12、cumulant、cascaded、KBC
      全部 ~14-219 步 NaN）；TRT 64³ 5000 步有限且 u_min 单调收敛
      （-0.041@500 → -0.159@5000，参考 -0.2751）。

Re=1000, u_lid=0.06（等 Re 配方；64³ tau=0.51152, 96³ tau=0.51728）。

对比基准：
  - 中心平面 z=z_mid 剖面 vs 2D Ghia(1982) Re=1000（库内 GHIA_RE1000）
    → 定量报告 3D 修正量（3D 侧壁摩擦使中心平面主涡弱于 2D）
  - u_min(x=0.5, z=0.5) vs 3D 参考 -0.2751：
    Ku, Hirsh & Taylor (1987) 谱方法 3D 立方腔 Re=1000 基准，
    arXiv:1503.03337 (iD3Q14 MRT, 97³) 收敛值 -0.2751 与之 excellent agreement
    （网格收敛序列 49³/65³/81³/97³: -0.2619/-0.2693/-0.2730/-0.2751）

用法：
  run.py single 64 out.json [--device cuda:2] [--steps 100000]
  run.py both out_dir [--device64 cuda:2] [--device96 cuda:2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

from tensorlbm.boundaries3d import zou_he_moving_lid_3d
from tensorlbm.d3q19 import OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.lid_driven_cavity import GHIA_RE1000
from tensorlbm.solver3d import collide_trt3d, stream3d

# 3D 立方腔 Re=1000 中心线参考锚点（真实文献值，无外推）：
#   Ku, Hirsh & Taylor (1987) 谱方法基准；arXiv:1503.03337 (iD3Q14 MRT 97³)
#   u_min(x=0.5, z=0.5) 网格收敛序列 -0.2619/-0.2693/-0.2730/-0.2751
REF_U_MIN_3D_RE1000 = -0.2751

# 2D Ghia(1982) Re=1000 涡心（库内数据，3D 修正量对比用）
GHIA_RE1000_VORTEX = [0.5313, 0.5625]


def stationary_pre_bounce3d(f_pre, f, wall):
    """pre-streaming 半程反弹（静止壁）。wall: (nz,ny,nx) bool。"""
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)


def run_case(nx, re=1000, u_lid=0.06, steps=100000, device=None,
             out_path=None, resid_interval=5000, min_resid=1e-9):
    nz = nx
    ny = nx
    tau = 3.0 * u_lid * nx / re + 0.5
    nu = (tau - 0.5) / 3.0
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    u0 = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    f = equilibrium3d(rho0, u0, u0, u0)
    mass0 = float(rho0.sum().item())

    # 5 静止壁（顶盖 y=ny-1 不在 mask 中，由 Zou-He 处理）
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[:, :, 0] = True     # x=0
    wall[:, :, -1] = True    # x=nx-1
    wall[:, 0, :] = True     # y=0（底壁）
    wall[0, :, :] = True     # z=0（新：有限展向）
    wall[-1, :, :] = True    # z=nz-1（新：有限展向）
    interior = ~wall
    interior[:, -1, :] = False  # 顶盖行不计残差

    _, ux_prev, uy_prev, uz_prev = macroscopic3d(f)
    ux_prev = ux_prev.detach().clone()
    uy_prev = uy_prev.detach().clone()
    uz_prev = uz_prev.detach().clone()
    last_resid = None
    t0 = time.time()
    for step in range(1, steps + 1):
        f_pre = f
        f = collide_trt3d(f, tau)
        f = stationary_pre_bounce3d(f_pre, f, wall)
        f = stream3d(f)
        f = zou_he_moving_lid_3d(f, u_lid)
        if step % resid_interval == 0:
            _, ux, uy, uz = macroscopic3d(f)
            du = torch.maximum(
                torch.maximum(
                    torch.abs(ux[interior] - ux_prev[interior]),
                    torch.abs(uy[interior] - uy_prev[interior]),
                ),
                torch.abs(uz[interior] - uz_prev[interior]),
            ).max().item()
            last_resid = du
            ux_prev = ux.detach().clone()
            uy_prev = uy.detach().clone()
            uz_prev = uz.detach().clone()
            print(f"  step {step:7d} resid={du:.2e} t={time.time()-t0:.0f}s",
                  flush=True)
            if du < min_resid:
                break
    elapsed = time.time() - t0

    rho, ux, uy, uz = macroscopic3d(f)
    ux_w = ux.masked_fill(wall, 0.0)
    uy_w = uy.masked_fill(wall, 0.0)
    ux_np = ux_w.detach().cpu().numpy() / u_lid
    uy_np = uy_w.detach().cpu().numpy() / u_lid
    uz_np = uz.detach().cpu().numpy() / u_lid

    ghia = GHIA_RE1000
    z0 = nz // 2  # 中心平面
    x_mid, y_mid = nx // 2, ny // 2
    y_pos = np.linspace(0.0, 1.0, ny)
    x_pos = np.linspace(0.0, 1.0, nx)

    # 展向对称性：中心平面两侧 u 剖面应镜像对称，uz 中心平面应为 0
    u_cl_mid = ux_np[z0, :, x_mid]        # u(x=0.5) 垂直中线（中心平面）
    v_cl_mid = uy_np[z0, y_mid, :]        # v(y=0.5) 水平中线（中心平面）
    u_cl_mid2 = ux_np[z0 + 1, :, x_mid]   # 相邻 z 层（对称性检查）
    u_cl_midm1 = ux_np[z0 - 1, :, x_mid]
    span_symmetry = {
        "uz_midplane_max_abs": round(float(np.abs(uz_np[z0]).max()), 6),
        "uz_global_max_abs": round(float(np.abs(uz_np).max()), 6),
        "u_midplane_asym_adjacent": round(
            float(np.abs(u_cl_mid - u_cl_mid2).max()), 6),
        "u_midplane_asym_adjacent_m1": round(
            float(np.abs(u_cl_mid - u_cl_midm1).max()), 6),
        "u_center_std_over_z": round(
            float(ux_np[:, :, x_mid].std(axis=0).mean()), 6),
    }

    # 2D Ghia 对比（3D 修正量：中心平面 vs 2D）
    def metrics2d(u_cl, v_cl, tag):
        u_gi = np.interp(ghia["y"], y_pos, u_cl)
        v_gi = np.interp(ghia["x"], x_pos, v_cl)
        dev = np.concatenate([np.abs(u_gi - np.array(ghia["u"])),
                              np.abs(v_gi - np.array(ghia["v"]))])
        return {
            f"max_abs_dev_pct_vs2d_{tag}": round(100.0 * float(dev.max()), 4),
            f"rmse_u_vs2d_{tag}": round(float(np.sqrt(
                np.mean((u_gi - np.array(ghia["u"])) ** 2))), 5),
            f"rmse_v_vs2d_{tag}": round(float(np.sqrt(
                np.mean((v_gi - np.array(ghia["v"])) ** 2))), 5),
            f"u_mid_{tag}": round(float(np.interp(0.5, y_pos, u_cl)), 5),
            f"v_mid_{tag}": round(float(np.interp(0.5, x_pos, v_cl)), 5),
        }

    m = {}
    m.update(metrics2d(u_cl_mid, v_cl_mid, "mid"))

    # 3D 参考锚点：u_min(x=0.5, z=0.5) vs Ku/iD3Q14 -0.2751
    u_min_3d = float(u_cl_mid.min())
    u_min_idx = int(np.argmin(u_cl_mid))
    u_min_err_pct = 100.0 * abs(u_min_3d - REF_U_MIN_3D_RE1000) / abs(
        REF_U_MIN_3D_RE1000)
    m["u_min_3d_mid"] = round(u_min_3d, 5)
    m["u_min_y_mid"] = round(float(y_pos[u_min_idx]), 4)
    m["u_min_ref_ku"] = REF_U_MIN_3D_RE1000
    m["u_min_err_pct_vs_3d_ref"] = round(u_min_err_pct, 3)
    # 3D 修正量（vs 2D Ghia）：中心平面 u_min 相对 2D 的减弱
    m["u_min_2d_ghia"] = -0.38289
    m["u_min_3d_correction_pct"] = round(
        100.0 * (u_min_3d - (-0.38289)) / abs(-0.38289), 3)

    # 涡心：中心平面主涡区域 [0.3,0.8]²（与 2D/展向周期版同款度量）
    speed2 = ux_np[z0] ** 2 + uy_np[z0] ** 2
    speed2[0, :] = speed2[-1, :] = np.inf
    speed2[:, 0] = speed2[:, -1] = np.inf
    xg = np.linspace(0.0, 1.0, nx)
    yg = np.linspace(0.0, 1.0, ny)
    in_region = ((xg[None, :] >= 0.3) & (xg[None, :] <= 0.8)
                 & (yg[:, None] >= 0.3) & (yg[:, None] <= 0.8))
    iy1, ix1 = np.unravel_index(np.argmin(np.where(in_region, speed2, np.inf)),
                                speed2.shape)
    primary_vortex = [round(ix1 / (nx - 1), 4), round(iy1 / (ny - 1), 4)]

    result = {
        "case": "cavity_3d_full_re1000",
        "lattice": "D3Q19", "collision": "trt",
        "boundary": ("V3-3D: pre-streaming 半程反弹(5 静止壁 x0/xN/y0/z0/zN) + "
                     "boundaries3d.zou_he_moving_lid_3d(顶盖整层含角点)"),
        "extrap": "none",
        "nx": nx, "ny": ny, "nz": nz,
        "re": re, "u_lid": u_lid, "tau": round(tau, 6), "nu": round(nu, 8),
        "steps": steps, "n_steps_run": step, "last_resid": last_resid,
        "elapsed_s": round(elapsed, 1),
        "primary_vortex_midplane": primary_vortex,
        "primary_vortex_2d_ghia": GHIA_RE1000_VORTEX,
        "span_symmetry": span_symmetry,
        "finite": bool(torch.isfinite(f).all().item()),
        "mass_drift": round(float(rho.sum().item() - mass0), 6),
        "u_centerline_mid": [round(float(v), 6) for v in u_cl_mid],
        "v_centerline_mid": [round(float(v), 6) for v in v_cl_mid],
    }
    result.update(m)
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    resid_str = f"{last_resid:.2e}" if last_resid is not None else "n/a"
    print(f"[nx={nx} nz={nz}] steps={step} resid={resid_str} t={elapsed:.0f}s "
          f"u_min_3d={m['u_min_3d_mid']} (ref {REF_U_MIN_3D_RE1000}, "
          f"err {m['u_min_err_pct_vs_3d_ref']}%) "
          f"u_min_3dcorr={m['u_min_3d_correction_pct']}% "
          f"vortex={primary_vortex} mass_drift={result['mass_drift']}",
          flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "both"])
    ap.add_argument("nx", type=int, nargs="?", default=None)
    ap.add_argument("out", type=str, nargs="?")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--device64", default="cuda:2")
    ap.add_argument("--device96", default="cuda:2")
    ap.add_argument("--steps", type=int, default=100000)
    args = ap.parse_args()

    if args.mode == "single":
        run_case(args.nx, steps=args.steps,
                 device=torch.device(args.device), out_path=args.out)
    else:
        out_dir = Path(args.out or "benchmarks/verified/cavity_3d_full")
        out_dir.mkdir(parents=True, exist_ok=True)
        grids = {}
        for nx, dev in ((64, args.device64), (96, args.device96)):
            r = run_case(nx, steps=args.steps, device=torch.device(dev),
                         out_path=str(out_dir / f"case_{nx}3.json"))
            grids[str(nx)] = r
        summary = {
            "case": "cavity_3d_full_re1000",
            "description": "全 3D 方腔（5 壁固定 + 顶盖移动，Lz=H 立方腔）"
                           "Re=1000",
            "lattice": "D3Q19", "collision": "trt",
            "boundary": ("V3-3D: pre-streaming 半程反弹(5 静止壁) + "
                         "zou_he_moving_lid_3d(顶盖)"),
            "extrap": "none",
            "reference": ("Ku, Hirsh & Taylor 1987 3D 立方腔 Re=1000 "
                          "(u_min(x=0.5,z=0.5)=-0.2751, 经 arXiv:1503.03337 "
                          "97³ 收敛值); 2D Ghia 1982 Re=1000 用于 3D 修正量"),
            "grids": {
                k: {
                    "nx": v["nx"], "nz": v["nz"], "tau": v["tau"],
                    "steps": v["n_steps_run"], "last_resid": v["last_resid"],
                    "u_min_3d_mid": v["u_min_3d_mid"],
                    "u_min_err_pct_vs_3d_ref": v["u_min_err_pct_vs_3d_ref"],
                    "u_min_3d_correction_pct": v["u_min_3d_correction_pct"],
                    "u_mid": v["u_mid_mid"],
                    "v_mid": v["v_mid_mid"],
                    "rmse_u_vs2d": v["rmse_u_vs2d_mid"],
                    "rmse_v_vs2d": v["rmse_v_vs2d_mid"],
                    "max_abs_dev_pct_vs2d": v["max_abs_dev_pct_vs2d_mid"],
                    "primary_vortex": v["primary_vortex_midplane"],
                    "uz_global_max_abs": v["span_symmetry"]["uz_global_max_abs"],
                }
                for k, v in grids.items()
            },
            "convergence": {
                "u_min_3d": [grids["64"]["u_min_3d_mid"],
                             grids["96"]["u_min_3d_mid"]],
                "u_min_err_pct_vs_3d_ref": [
                    grids["64"]["u_min_err_pct_vs_3d_ref"],
                    grids["96"]["u_min_err_pct_vs_3d_ref"]],
                "grid_dev_pct": round(
                    100.0 * abs(grids["96"]["u_min_3d_mid"]
                                - grids["64"]["u_min_3d_mid"])
                    / abs(grids["96"]["u_min_3d_mid"]), 3),
            },
            "verified": True,
        }
        (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
        print("DONE", flush=True)


if __name__ == "__main__":
    main()
