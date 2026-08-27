#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""方腔流 Re=100（lid-driven cavity）— Ghia 1982 验证（V3 修复版）。

共性模块路径（库 solver + 库 BC，零手写 collide/stream/equilibrium）：
- tensorlbm.solver.collide_mrt / stream
- tensorlbm.d2q9.equilibrium / macroscopic
- tensorlbm.lid_driven_cavity.zou_he_moving_lid（顶盖动壁 BC）
- 三静止壁：pre-streaming 半程反弹（V3 关键修复，内联 BC，见下）

V3 修复背景（2026-08-19，Re=400 案例经验迁移）：
V0 的 post-streaming bounce_back_cells + 周期 stream 组合使顶盖动量绕入底壁行
（底部回流过量 2.6 倍）→ Re=100 曾 22.5% 偏差。V3 改为三静止壁 pre-streaming
半程反弹（流体侧反射，动量不进壁）+ 顶盖 zou_he_moving_lid，Re=400 从 23.6%
修复到 1.50%→0.83%。Re=100 同根因，本脚本用 V3 正式两档验证。

Re 约定：Re = u_lid*H/ν，H = nx（网格节点数，与任务参数 tau=3*u_lid*nx/re+0.5 一致）。
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
from tensorlbm.lid_driven_cavity import GHIA_RE100, zou_he_moving_lid
from tensorlbm.solver import collide_mrt, stream

CS2 = 1.0 / 3.0


def stationary_pre_bounce(f_pre, f, wall):
    """pre-streaming 半程反弹（三静止壁，V3 关键修复）。

    在 collide 之前用流体侧（碰撞前）分布反射，动量不进入壁面行；
    与 post-streaming bounce_back_cells + 周期 stream 组合（顶盖动量会
    经周期环绕被注入底壁行）形成 A/B 对照。
    """
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)


def run_case(nx, re, u_lid, steps, device, compile_mode="default"):
    ny = nx
    tau = 3.0 * u_lid * nx / re + 0.5
    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, u0, u0)

    # 三静止壁（底/左/右）+ 顶盖动壁（zou_he_moving_lid，整行含角点由库处理）
    wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[:, 0] = True
    wall_mask[:, -1] = True

    interior = ~wall_mask
    resid_mask = interior.clone()
    resid_mask[-1, :] = False  # 顶行由 lid BC 驱动，不计入稳态残差

    # ---- 整步步进函数（共性 compile 路径；步序号与监测留在编译域外）----
    def _step(f):
        f_pre = f
        f = collide_mrt(f, tau=tau)
        f = stationary_pre_bounce(f_pre, f, wall_mask)  # V3：三壁 pre-streaming 半程反弹
        f = stream(f)
        return zou_he_moving_lid(f, u_lid)

    step_fn = route_step(_step, compile_mode, name=f"cavity_re100[{nx}]")

    _, ux_prev, uy_prev = macroscopic(f)
    ux_prev = ux_prev.detach().clone()
    uy_prev = uy_prev.detach().clone()
    last_resid = None

    t0 = time.time()
    for step in range(1, steps + 1):
        f = step_fn(f)
        if step % 10000 == 0 or step == steps:
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
    elapsed = time.time() - t0

    # ---- 度量（与 verified/cavity_re400 同口径）----
    rho, ux, uy = macroscopic(f)
    ux_w = ux.masked_fill(wall_mask, 0.0)
    uy_w = uy.masked_fill(wall_mask, 0.0)
    ux_np = ux_w.detach().cpu().numpy() / u_lid
    uy_np = uy_w.detach().cpu().numpy() / u_lid

    x_mid, y_mid = nx // 2, ny // 2
    y_pos = np.linspace(0.0, 1.0, ny)
    x_pos = np.linspace(0.0, 1.0, nx)
    u_cl = ux_np[:, x_mid]
    v_cl = uy_np[y_mid, :]
    u_gi = np.interp(GHIA_RE100["y"], y_pos, u_cl)  # xp 升序（y_pos），安全
    v_gi = np.interp(GHIA_RE100["x"], x_pos, v_cl)
    rmse_u = float(np.sqrt(np.mean((u_gi - np.array(GHIA_RE100["u"])) ** 2)))
    rmse_v = float(np.sqrt(np.mean((v_gi - np.array(GHIA_RE100["v"])) ** 2)))
    dev = np.concatenate(
        [np.abs(u_gi - np.array(GHIA_RE100["u"])), np.abs(v_gi - np.array(GHIA_RE100["v"]))]
    )
    max_abs_dev_pct = 100.0 * float(dev.max())

    u_mid = float(np.interp(0.5, y_pos, u_cl))  # Ghia u(0.5,0.5) = -0.20581
    u_bot = float(np.interp(0.0625, y_pos, u_cl))  # Ghia u(0.5,0.0625) = -0.04192
    v_mid = float(np.interp(0.5, x_pos, v_cl))

    # 主涡心：内域 argmin(speed²) + 抛物线亚网格细化（主涡物理窗口 x,y∈[0.3,0.9]）
    speed2 = ux_np**2 + uy_np**2
    speed2[: int(0.30 * ny), :] = np.inf
    speed2[int(0.90 * ny) :, :] = np.inf
    speed2[:, : int(0.30 * nx)] = np.inf
    speed2[:, int(0.90 * nx) :] = np.inf
    iy0, ix0 = np.unravel_index(np.argmin(speed2), speed2.shape)

    # 抛物线亚网格：i + 0.5*(a-c)/(a-2b+c)
    def refine(axis, i):
        a = speed2[i - 1, ix0] if axis == 0 else speed2[iy0, i - 1]
        b = speed2[i, ix0] if axis == 0 else speed2[iy0, i]
        c = speed2[i + 1, ix0] if axis == 0 else speed2[iy0, i + 1]
        denom = a - 2 * b + c
        return i + 0.5 * (a - c) / denom if abs(denom) > 1e-30 else i

    vx = refine(1, ix0) / (nx - 1)
    vy = refine(0, iy0) / (ny - 1)

    print(
        f"[cavity_re100 nx={nx}] steps={steps} t={elapsed:.0f}s resid={last_resid:.2e} "
        f"u(0.5,0.5)={u_mid:+.4f} u_bot={u_bot:+.4f} v(0.5,0.5)={v_mid:+.4f} "
        f"vortex=({vx:.3f},{vy:.3f}) rmse_u={rmse_u:.4f} rmse_v={rmse_v:.4f} "
        f"max_abs_dev={max_abs_dev_pct:.2f}%",
        flush=True,
    )

    return {
        "nx": nx,
        "re": re,
        "u_lid": u_lid,
        "tau": round(tau, 4),
        "steps": steps,
        "compile_mode": compile_mode,
        "elapsed_s": round(elapsed, 1),
        "last_resid": last_resid,
        "u_mid": u_mid,
        "u_mid_ghia": -0.20581,
        "u_bot": u_bot,
        "v_mid": v_mid,
        "vortex": [round(vx, 4), round(vy, 4)],
        "vortex_ghia": [0.6172, 0.7344],
        "rmse_u": rmse_u,
        "rmse_v": rmse_v,
        "max_abs_dev_pct": max_abs_dev_pct,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", nargs="+", type=int, default=[128, 192])
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--u-lid", type=float, default=0.06)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="/tmp/cavity_re100_verified.json")
    add_compile_mode_arg(ap)
    args = ap.parse_args()
    compile_mode = compile_mode_from_args(args)

    device = torch.device(args.device)
    torch.set_num_threads(32)

    results = {}
    for nx in args.nx:
        results[str(nx)] = run_case(
            nx, args.re, args.u_lid, args.steps, device, compile_mode=compile_mode
        )

    devs = [results[str(nx)]["max_abs_dev_pct"] for nx in args.nx]
    verdict = {
        "max_abs_dev_pct": devs,
        "err_decreased": devs[-1] < devs[0],
        "pass": all(d <= 3.0 for d in devs) and devs[-1] < devs[0],
    }
    results["convergence"] = verdict
    results["verified"] = bool(verdict["pass"])
    results["verified_date"] = "2026-08-19"

    # tolerate both spellings of --out: a file path (historic default,
    # e.g. /tmp/cavity_re100_verified.json) or a results directory
    out_path = Path(args.out)
    if not out_path.suffix:
        out_path = out_path / "result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"verdict: {json.dumps(verdict)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
