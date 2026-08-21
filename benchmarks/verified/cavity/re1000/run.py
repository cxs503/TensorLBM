#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""方腔流 Re=1000 — Ghia(1982) 验证（V3 BC 修复 + RLBM 碰撞，200k 步正式重跑）。

复现：cd /home/wxsc/cxs/TensorLBM && PYTHONPATH=src python benchmarks/verified/cavity_re1000/run.py

关键背景（真实测量，2026-08-19）：
- V3 BC（pre-streaming 半程反弹三静止壁 + zou_he_moving_lid 顶盖）+ collide_mrt
  在 Re=1000 等 Re 配方（u_lid=0.06, tau=3*0.06*nx/1000+0.5 -> 192² tau=0.5346）下
  于 step 80~271 NaN 发散；tau 稳定边界扫描（192²）得 tau>=0.56 才稳，
  即 MRT 稳定上限约 Re~576（u_lid=0.06）。Re=1000 对 MRT 硬性不可达。
- 改用库内 collide_rlbm（Latt & Chopard 2006 正则化 BGK，滤除高阶鬼矩，
  专为 tau->0.5 低粘稳定设计）：256² (tau=0.5461) 稳定，物理结果达标（见 result.json）。

运行：两档 192²/256²，各 200k 步（Re=1000 收敛慢，100k 残差 ~1e-2，200k 才 ~1e-3）。
     默认 CPU（32 线程）；--device cuda:1 可选（2026-08-19 正式重跑即 GPU 版）。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # <repo>/benchmarks

from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic
from tensorlbm.lid_driven_cavity import GHIA_RE1000, zou_he_moving_lid
from tensorlbm.solver import collide_rlbm, stream

torch.set_num_threads(32)
CS2 = 1.0 / 3.0


def stationary_pre_bounce(f_pre, f, wall):
    """V3: pre-streaming 半程反弹（三静止壁，流体侧反射）。"""
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)


def run_case(nx, re=1000, u_lid=0.06, steps=200000, device=None, compile_mode="default"):
    device = device or torch.device("cpu")
    ny = nx
    tau = 3 * u_lid * nx / re + 0.5  # 等 Re 配方
    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, u0, u0)

    wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[:, 0] = True
    wall_mask[:, -1] = True

    interior = ~wall_mask
    resid_mask = interior.clone()
    resid_mask[-1, :] = False

    _, ux_prev, uy_prev = macroscopic(f)
    ux_prev = ux_prev.detach().clone()
    uy_prev = uy_prev.detach().clone()
    last_resid = None

    # ---- 整步步进函数（共性 compile 路径；步序号与监测留在编译域外）----
    def _step(f):
        f_pre = f
        f = collide_rlbm(f, tau=tau)
        f = stationary_pre_bounce(f_pre, f, wall_mask)
        f = stream(f)
        return zou_he_moving_lid(f, u_lid)

    step_fn = route_step(_step, compile_mode, name=f"cavity_re1000[{nx}]")

    t0 = time.time()
    for step in range(1, steps + 1):
        f = step_fn(f)
        if step % 20000 == 0:
            _, ux, uy = macroscopic(f)
            du = (
                torch.max(
                    torch.abs(ux[resid_mask] - ux_prev[resid_mask]),
                    torch.abs(uy[resid_mask] - uy_prev[resid_mask]),
                )
                .max()
                .item()
            )
            last_resid = du
            ux_prev = ux.detach().clone()
            uy_prev = uy.detach().clone()
            print(
                f"  step {step}/{steps} resid={last_resid:.2e} t={time.time() - t0:.0f}s",
                flush=True,
            )
    elapsed = time.time() - t0

    rho, ux, uy = macroscopic(f)
    ux_w = ux.masked_fill(wall_mask, 0.0)
    uy_w = uy.masked_fill(wall_mask, 0.0)
    ux_np = ux_w.detach().cpu().numpy() / u_lid
    uy_np = uy_w.detach().cpu().numpy() / u_lid

    ghia = GHIA_RE1000
    x_mid, y_mid = nx // 2, ny // 2
    y_pos = np.linspace(0.0, 1.0, ny)
    x_pos = np.linspace(0.0, 1.0, nx)
    u_cl = ux_np[:, x_mid]
    v_cl = uy_np[y_mid, :]
    u_gi = np.interp(ghia["y"], y_pos, u_cl)
    v_gi = np.interp(ghia["x"], x_pos, v_cl)
    rmse_u = float(np.sqrt(np.mean((u_gi - np.array(ghia["u"])) ** 2)))
    rmse_v = float(np.sqrt(np.mean((v_gi - np.array(ghia["v"])) ** 2)))
    dev = np.concatenate([np.abs(u_gi - np.array(ghia["u"])), np.abs(v_gi - np.array(ghia["v"]))])
    max_abs_dev_pct = 100.0 * float(dev.max())

    u_mid = float(np.interp(0.5, y_pos, u_cl))
    u_bot = float(np.interp(0.0625, y_pos, u_cl))
    v_mid = float(np.interp(0.5, x_pos, v_cl))

    # 全腔最小速度点（Re=1000 捕获次级涡）与主涡区域最小点（主涡涡心）
    speed2 = ux_np**2 + uy_np**2
    speed2[0, :] = speed2[-1, :] = np.inf
    speed2[:, 0] = speed2[:, -1] = np.inf
    iy0, ix0 = np.unravel_index(np.argmin(speed2), speed2.shape)
    sub_vortex = [round(ix0 / (nx - 1), 4), round(iy0 / (ny - 1), 4)]
    x_lo, x_hi, y_lo, y_hi = 0.2, 0.85, 0.2, 0.9
    in_reg = (
        (x_pos[None, :] >= x_lo)
        & (x_pos[None, :] <= x_hi)
        & (y_pos[:, None] >= y_lo)
        & (y_pos[:, None] <= y_hi)
    )
    iy1, ix1 = np.unravel_index(np.argmin(np.where(in_reg, speed2, np.inf)), speed2.shape)
    primary_vortex = [round(ix1 / (nx - 1), 4), round(iy1 / (ny - 1), 4)]

    resid_str = "n/a" if last_resid is None else f"{last_resid:.2e}"
    print(
        f"[cavity_re1000 nx={nx}] steps={steps} t={elapsed:.0f}s resid={resid_str} "
        f"u(0.5,0.5)={u_mid:+.4f} u_bot={u_bot:+.4f} v(0.5,0.5)={v_mid:+.4f} "
        f"primary_vortex={primary_vortex} sub_vortex={sub_vortex} "
        f"rmse_u={rmse_u:.4f} rmse_v={rmse_v:.4f} max_abs_dev={max_abs_dev_pct:.2f}%",
        flush=True,
    )

    return {
        "nx": nx,
        "re": re,
        "u_lid": u_lid,
        "tau": tau,
        "steps": steps,
        "compile_mode": compile_mode,
        "elapsed_s": round(elapsed, 1),
        "last_resid": last_resid,
        "u_mid": u_mid,
        "u_mid_ghia": -0.0608,
        "u_bot": u_bot,
        "v_mid": v_mid,
        "primary_vortex": primary_vortex,
        "vortex_ghia": [0.5313, 0.5625],
        "sub_vortex": sub_vortex,
        "rmse_u": rmse_u,
        "rmse_v": rmse_v,
        "max_abs_dev_pct": max_abs_dev_pct,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, nargs="+", default=[192, 256])
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--device", default="cpu")
    add_compile_mode_arg(ap)
    args = ap.parse_args()
    device = torch.device(args.device)
    compile_mode = compile_mode_from_args(args)
    results = {}
    for nx in args.nx:
        results[str(nx)] = run_case(nx, steps=args.steps, device=device, compile_mode=compile_mode)
    out = Path(__file__).parent / "result.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("DONE", flush=True)
