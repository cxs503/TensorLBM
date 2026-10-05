"""W9-A 基准入口：3D 差热立方腔自然对流（D3Q19 BGK + D3Q7 温度，Pr=0.71）。

物理全部来自 tensorlbm 库入口（solver3d / thermal3d / thermal_common /
boundaries3d），本文件零本地物理实现（grep 自检：无手写 collide/stream/
equilibrium/bounce/zou_he/far_field）。配方（判据档冻结）：

- 壁速度：全反弹 bounce-back，库 ``boundaries3d.bounce_back_cells_3d``
  于 streaming 之后施加（bb=post，壁在节点上）——与 #303 温度壁的
  节点等温面同位配对；半程 pre-bounce 变体存在壁位错位（一阶收敛），
  档案留在波次暂存，不入本条目。
- 壁温度：库 ``thermal3d.apply_temperature_boundaries_3d``（#303 修复版）。
- 浮力：碰撞后 ``thermal_common.apply_buoyancy_3d`` 原始增量（无 Guo
  1-1/(2tau) 因子；tau 不变性 2x2 判别见 README），**只加流体节点**
  （force_mask 驱动层一行掩码；不掩码时全反弹壁节点被持续灌注，
  慢线性不稳定 NaN）。
- 温度碰撞前壁节点宏观 u 置零（#300 2D 同款修复）。
- D3Q7 电导率自标定：alpha=(tau_T-1/2)/4（W=[1/4,1/8x6]）；
  nu=(tau_f-1/2)/3；g*beta=Ra*nu*alpha/L^3，L=nx-1，dT=1。
- Nu：整壁 grad1（壁节点-首内邻差分，含角点）面积平均，
  Nu=(Nu_hot+Nu_cold)/2；窗=记录历史末 20% 均值。

用法：
  python run.py --ra 1e4 --n 64 --steps 200000            # 单案
  python run.py --ra 1e3 --n 32 --steps 40000 --device cpu  # CPU 冒烟
判据档完整阶梯步数见 README.md 判据表；out/ 内为正式档机器档案
（probe_fm_*.json，wall_fields_end 原语可独立重算 Nu）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.thermal3d import (
    apply_temperature_boundaries_3d,
    collide_thermal_bgk_3d,
    equilibrium_thermal_3d,
    macroscopic_thermal_3d,
    stream_thermal_3d,
)
from tensorlbm.thermal_common import apply_buoyancy_3d


def main() -> None:
    p = argparse.ArgumentParser(
        description="3D cubic thermal cavity (W9-A): D3Q19+D3Q7, full-way BB, "
        "fluid-masked post-collision buoyancy"
    )
    p.add_argument("--ra", type=float, required=True, help="Rayleigh number")
    p.add_argument("--n", type=int, required=True, help="cubic grid size")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--pr", type=float, default=0.71)
    p.add_argument("--tau_f", type=float, default=0.6)
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--dtype", choices=["fp32", "fp64"], default="fp32")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "out"),
        help="output directory (default: <entry>/out)",
    )
    p.add_argument("--tag", default=None, help="archive tag (default: fm_ra{RA}_n{N})")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device(args.device)
    dt = torch.float64 if args.dtype == "fp64" else torch.float32

    tag = args.tag or f"fm_ra{args.ra:.0e}_n{args.n}".replace("e+0", "e")
    n = args.n
    nz = ny = nx = n
    L = float(nx - 1)
    T_hot, T_cold = 1.0, 0.0

    nu = (args.tau_f - 0.5) / 3.0
    alpha = nu / args.pr
    tau_T = 4.0 * alpha + 0.5
    beta = args.ra * nu * alpha / (L**3)

    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    wall[:, :, 0] = wall[:, :, -1] = True
    wall[:, 0, :] = wall[:, -1, :] = True
    wall[0, :, :] = wall[-1, :, :] = True
    fluid = (~wall).to(dt)

    _, _, xg = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=dt),
        torch.arange(ny, device=dev, dtype=dt),
        torch.arange(nx, device=dev, dtype=dt),
        indexing="ij",
    )
    rho = torch.ones((nz, ny, nx), device=dev, dtype=dt)
    uz = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    ux = torch.zeros_like(rho)
    T = T_hot - (T_hot - T_cold) * xg / L

    f = equilibrium3d(rho, ux, uy, uz).to(dt)
    g = equilibrium_thermal_3d(T, ux, uy, uz).to(dt)
    g = apply_temperature_boundaries_3d(g, T_hot=T_hot, T_cold=T_cold)

    hist: list[dict] = []
    nan_at = None
    sumT0 = None
    t0 = time.time()
    for step in range(args.steps):
        rho, ux, uy, uz = macroscopic3d(f)
        # wall-node macroscopic u is bounce-back garbage; zero it before the
        # temperature collision (#300 2D same-family fix)
        ux = ux.masked_fill(wall, 0.0)
        uy = uy.masked_fill(wall, 0.0)
        uz = uz.masked_fill(wall, 0.0)

        T = macroscopic_thermal_3d(g)
        g = collide_thermal_bgk_3d(g, T, ux, uy, uz, tau_T=tau_T)
        g = stream_thermal_3d(g)
        g = apply_temperature_boundaries_3d(g, T_hot=T_hot, T_cold=T_cold)
        T = macroscopic_thermal_3d(g)

        f = collide_bgk3d(f, args.tau_f)
        f_buoy = apply_buoyancy_3d(f, T, T_ref=0.5, beta=beta, g_y=-1.0, lattice="D3Q19")
        incr = (f_buoy - f) * fluid  # raw increment, fluid nodes only
        f = f + incr
        f = stream3d(f)
        f = bounce_back_cells_3d(f, wall)

        if step % args.sample == 0 or step == args.steps - 1:
            if not torch.isfinite(T).all() or not torch.isfinite(f).all():
                nan_at = step
                break
            nu_hot = float(((T[:, :, 0] - T[:, :, 1]) * L).mean().item())
            nu_cold = float(((T[:, :, -2] - T[:, :, -1]) * L).mean().item())
            umag = torch.sqrt(ux * ux + uy * uy + uz * uz)[~wall]
            umax = float(umag.max().item())
            sumT = float(T.sum().item())
            if sumT0 is None:
                sumT0 = sumT
            hist.append(
                {
                    "step": step,
                    "nu_hot": nu_hot,
                    "nu_cold": nu_cold,
                    "nu": 0.5 * (nu_hot + nu_cold),
                    "u_max": umax,
                    "sumT": sumT,
                    "T_min": float(T.min().item()),
                    "T_max": float(T.max().item()),
                }
            )

    elapsed = time.time() - t0
    # end-state wall fields: raw primitive for independent Nu recomputation
    # (exact decimal print of the stored binary values)
    wall_fields_end = {
        "T_x0_hot_wall": T[:, :, 0].detach().cpu().tolist(),
        "T_x1_hot_adj": T[:, :, 1].detach().cpu().tolist(),
        "T_xm2_cold_adj": T[:, :, -2].detach().cpu().tolist(),
        "T_xm1_cold_wall": T[:, :, -1].detach().cpu().tolist(),
    }
    out = {
        "tag": tag,
        "nu_convention": {
            "definition": "Nu_face = L * mean_fullface(T_wall - T_adj); Nu = (Nu_hot + Nu_cold)/2",
            "gradient": "one-sided first difference across the wall-adjacent link (grad1)",
            "face_mean": "full y-z face INCLUDING corner nodes "
            "(cluster-comparable; verified/cavity convention)",
            "L": L,
            "L_basis": "nx-1 (isothermal surface on the node: #303 T-slab + full-way BB)",
            "sample_every_steps": args.sample,
            "steady_window": "last 20% of recorded history samples",
            "recompute_hint": "nu_hot = mean(T_x0_hot_wall - T_x1_hot_adj) "
            "* L; nu_cold = mean(T_xm2_cold_adj - "
            "T_xm1_cold_wall) * L (must reproduce "
            "history[-1].nu_hot/nu_cold bitwise)",
        },
        "config": {
            "pr": args.pr,
            "ra": args.ra,
            "tau_f": args.tau_f,
            "n": args.n,
            "steps": args.steps,
            "dtype": args.dtype,
            "bb": "post",
            "force": "post_raw",
            "force_mask": True,
            "nu": nu,
            "alpha": alpha,
            "tau_T": tau_T,
            "g_beta": beta,
            "L": L,
        },
        "nan_at": nan_at,
        "elapsed_s": elapsed,
        "steps_per_s": args.steps / max(elapsed, 1e-9),
        "sumT0": sumT0,
        "history": hist,
        "wall_fields_end": wall_fields_end,
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"probe_{tag}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    last = hist[-1] if hist else {}
    print(
        f"[{tag}] Pr={args.pr} tau_f={args.tau_f} {args.dtype} bb=post "
        f"force=post_raw+force_mask tau_T-0.5={tau_T - 0.5:.3e} "
        f"nan_at={nan_at} Nu_last={last.get('nu', float('nan')):.5f} "
        f"u_max={last.get('u_max', float('nan')):.4e} {elapsed:.0f}s -> {path}"
    )


if __name__ == "__main__":
    main()
