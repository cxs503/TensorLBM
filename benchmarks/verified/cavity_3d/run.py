#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""3D 方腔流（展向周期近似 2D）Re=400 — Ghia(1982) 验证（D3Q19 MRT）。

几何：f 形状 (19, nz, ny, nx)。y 垂直（顶盖 y=ny-1 沿 +x 移动），x 水平，
      z 展向周期（stream3d 天然周期）→ 近似 2D 方腔。

边界（V3 3D，与 2D verified cavity_re400 同口径）：
  - 三静止壁（x=0, x=nx-1, y=0）：pre-streaming 半程反弹（碰撞后、stream 前
    用 f_pre 反射；库内暂无此助手，见 README 缺口说明）
  - 顶盖：boundaries3d.zou_he_moving_lid_3d（库内置 D3Q19 Zou-He 移动壁，
    覆盖整层含角点）

修复背景（2026-08-19）：
  迁移版 /tmp/cavity3d_a.py 的 z 对角对 17/18 颠倒（sum_cyp 误用未知 f18 代
  替已知 f17；z 对重建覆盖已知 f17、留下被周期 y-wrap 污染的 f18）→ 每步向
  顶盖注入虚假 jy/jz（数值验证 buggy jy=-0.016, jz=+0.12 vs 修复版精确 0）
  → 96² 曾 14.5% 偏差、质量漂移 -51289（23%）、涡心错乱。
  修复后：96²×24 max_dev=1.96%、主涡 (0.558,0.611) vs Ghia (0.5547,0.6055)、
  质量漂移 0.018%。

用法：
  run.py single 96 24 out.json [--device cuda:1] [--steps 100000]
  run.py both out_dir [--device96 cuda:1] [--device128 cuda:2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

from tensorlbm.boundaries3d import zou_he_moving_lid_3d
from tensorlbm.d3q19 import OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.lid_driven_cavity import GHIA_RE400
from tensorlbm.solver3d import collide_mrt3d, stream3d


def stationary_pre_bounce3d(f_pre, f, wall):
    """pre-streaming 半程反弹（静止壁）。wall: (nz,ny,nx) bool。"""
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)


def run_case(nx, nz, re=400, u_lid=0.06, steps=100000, device=None,
             out_path=None, resid_interval=5000, min_resid=1e-9,
             compile_mode="default"):
    ny = nx
    tau = 3.0 * u_lid * nx / re + 0.5
    nu = (tau - 0.5) / 3.0
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    u0 = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    f = equilibrium3d(rho0, u0, u0, u0)
    mass0 = float(rho0.sum().item())

    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[:, :, 0] = True    # x=0
    wall[:, :, -1] = True   # x=nx-1
    wall[:, 0, :] = True    # y=0（底壁）
    # 顶盖 y=ny-1 不在静止壁 mask 中
    interior = ~wall
    interior[:, -1, :] = False

    # ---- 整步步进函数（共性 compile 路径；步序号与残差监测留在编译域外）----
    def _step(f):
        f_pre = f
        f = collide_mrt3d(f, tau)
        f = stationary_pre_bounce3d(f_pre, f, wall)
        f = stream3d(f)
        return zou_he_moving_lid_3d(f, u_lid)

    step_fn = route_step(_step, compile_mode, name=f"cavity_3d[{nx}x{nz}]")

    _, ux_prev, uy_prev, _ = macroscopic3d(f)
    ux_prev = ux_prev.detach().clone()
    uy_prev = uy_prev.detach().clone()
    last_resid = None
    t0 = time.time()
    for step in range(1, steps + 1):
        f = step_fn(f)
        if step % resid_interval == 0:
            _, ux, uy, _ = macroscopic3d(f)
            du = torch.max(
                torch.abs(ux[interior] - ux_prev[interior]),
                torch.abs(uy[interior] - uy_prev[interior]),
            ).max().item()
            last_resid = du
            ux_prev = ux.detach().clone()
            uy_prev = uy.detach().clone()
            print(f"  step {step:7d} resid={du:.2e}", flush=True)
            if du < min_resid:
                break
    elapsed = time.time() - t0

    rho, ux, uy, uz = macroscopic3d(f)
    ux_w = ux.masked_fill(wall, 0.0)
    uy_w = uy.masked_fill(wall, 0.0)
    ux_np = ux_w.detach().cpu().numpy() / u_lid
    uy_np = uy_w.detach().cpu().numpy() / u_lid
    uz_np = uz.detach().cpu().numpy() / u_lid

    ghia = GHIA_RE400
    z0 = nz // 2
    x_mid, y_mid = nx // 2, ny // 2
    y_pos = np.linspace(0.0, 1.0, ny)
    x_pos = np.linspace(0.0, 1.0, nx)

    span_uniformity = {
        "ux_span_std_mean": round(float(ux_np[1:-1].std(axis=0).mean()), 6),
        "uz_max_abs": round(float(np.abs(uz_np).max()), 6),
    }

    u_cl_mid = ux_np[z0, :, x_mid]            # u(x=0.5) 垂直中线（z 中间层）
    v_cl_mid = uy_np[z0, y_mid, :]            # v(y=0.5) 水平中线（z 中间层）
    u_cl_avg = ux_np[:, :, x_mid].mean(axis=0)
    v_cl_avg = uy_np[:, y_mid, :].mean(axis=0)

    def metrics(u_cl, v_cl, tag):
        u_gi = np.interp(ghia["y"], y_pos, u_cl)
        v_gi = np.interp(ghia["x"], x_pos, v_cl)
        dev = np.concatenate([np.abs(u_gi - np.array(ghia["u"])),
                              np.abs(v_gi - np.array(ghia["v"]))])
        return {
            f"max_abs_dev_pct_{tag}": round(100.0 * float(dev.max()), 4),
            f"rmse_u_{tag}": round(float(np.sqrt(np.mean((u_gi - np.array(ghia["u"])) ** 2))), 5),
            f"rmse_v_{tag}": round(float(np.sqrt(np.mean((v_gi - np.array(ghia["v"])) ** 2))), 5),
            f"u_mid_{tag}": round(float(np.interp(0.5, y_pos, u_cl)), 5),
            f"v_mid_{tag}": round(float(np.interp(0.5, x_pos, v_cl)), 5),
        }

    m = {}
    m.update(metrics(u_cl_mid, v_cl_mid, "mid"))
    m.update(metrics(u_cl_avg, v_cl_avg, "avg"))

    # 涡心：z 中间层；主涡区域 [0.3,0.8]x[0.3,0.8]（全局 argmin 会捕获角部死区，
    # 与 2D verified 同款度量局限，主涡区域搜索为准）
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
        "case": "cavity_3d_spanwise_re400",
        "lattice": "D3Q19", "collision": "mrt",
        "boundary": ("V3-3D: pre-streaming 半程反弹(三静止壁) + "
                     "boundaries3d.zou_he_moving_lid_3d(顶盖)"),
        "extrap": "none",
        "nx": nx, "ny": ny, "nz": nz,
        "re": re, "u_lid": u_lid, "tau": round(tau, 6), "nu": round(nu, 8),
        "steps": steps, "n_steps_run": step, "last_resid": last_resid,
        "compile_mode": compile_mode,
        "elapsed_s": round(elapsed, 1),
        "u_mid_ghia": ghia["u"][ghia["y"].index(0.5)],
        "primary_vortex": primary_vortex,
        "primary_vortex_ghia": [0.5547, 0.6055],
        "span_uniformity": span_uniformity,
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
          f"max_dev_mid={m['max_abs_dev_pct_mid']}% u_mid={m['u_mid_mid']} "
          f"primary_vortex={primary_vortex} mass_drift={result['mass_drift']}",
          flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "both"])
    ap.add_argument("nx", type=int, nargs="?", default=None)
    ap.add_argument("nz", type=int, nargs="?", default=None)
    ap.add_argument("out", type=str, nargs="?", default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--device96", default="cuda:1")
    ap.add_argument("--device128", default="cuda:2")
    ap.add_argument("--steps", type=int, default=100000)
    add_compile_mode_arg(ap)
    args = ap.parse_args()
    compile_mode = compile_mode_from_args(args)

    if args.mode == "single":
        run_case(args.nx, args.nz, steps=args.steps,
                 device=torch.device(args.device), out_path=args.out,
                 compile_mode=compile_mode)
    else:
        out_dir = Path(args.out or "benchmarks/verified/cavity_3d")
        out_dir.mkdir(parents=True, exist_ok=True)
        grids = {}
        for nx, nz, dev in ((96, 24, args.device96), (128, 32, args.device128)):
            r = run_case(nx, nz, steps=args.steps, device=torch.device(dev),
                         out_path=str(out_dir / f"case_{nx}x{nz}.json"),
                         compile_mode=compile_mode)
            grids[str(nx)] = r
        summary = {
            "case": "cavity_3d_spanwise_re400",
            "grids": {
                k: {
                    "nx": v["nx"], "nz": v["nz"], "tau": v["tau"],
                    "steps": v["n_steps_run"], "last_resid": v["last_resid"],
                    "u_mid": v["u_mid_mid"], "u_mid_ghia": v["u_mid_ghia"],
                    "rmse_u": v["rmse_u_mid"], "rmse_v": v["rmse_v_mid"],
                    "max_abs_dev_pct": v["max_abs_dev_pct_mid"],
                    "primary_vortex": v["primary_vortex"],
                }
                for k, v in grids.items()
            },
            "convergence": {
                "max_abs_dev": [grids[str(k)]["max_abs_dev_pct_mid"]
                                for k in (96, 128)],
                "err_decreased": (grids["96"]["max_abs_dev_pct_mid"]
                                  >= grids["128"]["max_abs_dev_pct_mid"]),
            },
        }
        (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
        print("DONE", flush=True)


if __name__ == "__main__":
    main()
