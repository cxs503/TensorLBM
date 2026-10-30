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
  python run.py --ra 1e4 --n 64 --auto-steady               # 自动稳态检测
  python run.py --ra 1e4 --n 64 --config thermal_cavity_3d_config.json  # 加载配置
判据档完整阶梯步数见 README.md 判据表；out/ 内为正式档机器档案
（probe_fm_*.json，wall_fields_end 原语可独立重算 Nu）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

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


def _compute_window_drift(hist: list[dict], window_fraction: float = 0.2) -> float:
    """Compute Nu window drift percentage from history.

    Window drift = (mean(last 10% of window) - mean(first 10% of window)) / mean(window) * 100
    Positive means Nu is still increasing (not yet steady).
    """
    if len(hist) < 10:
        return float("nan")
    n_window = max(int(len(hist) * window_fraction), 4)
    window = hist[-n_window:]
    n_sub = max(len(window) // 10, 1)
    first_sub = window[:n_sub]
    last_sub = window[-n_sub:]
    nu_first = sum(r["nu"] for r in first_sub) / len(first_sub)
    nu_last = sum(r["nu"] for r in last_sub) / len(last_sub)
    nu_mean = sum(r["nu"] for r in window) / len(window)
    if nu_mean == 0:
        return float("nan")
    return (nu_last - nu_first) / nu_mean * 100.0


def _check_steady_state(
    hist: list[dict],
    window_drift_threshold_pct: float = 0.01,
    hot_cold_imbalance_threshold_pct: float = 0.1,
    window_fraction: float = 0.2,
) -> dict:
    """Check if simulation has reached steady state.

    Returns dict with convergence metrics and boolean flag.
    """
    if len(hist) < 10:
        return {"converged": False, "reason": "insufficient_samples", "metrics": {}}

    window_drift = _compute_window_drift(hist, window_fraction)
    last = hist[-1]
    nu_hot = last["nu_hot"]
    nu_cold = last["nu_cold"]
    if nu_hot + nu_cold == 0:
        imbalance = float("nan")
    else:
        imbalance = abs(nu_hot - nu_cold) / ((nu_hot + nu_cold) / 2) * 100.0

    converged = (
        abs(window_drift) < window_drift_threshold_pct
        and imbalance < hot_cold_imbalance_threshold_pct
    )

    return {
        "converged": converged,
        "window_drift_pct": window_drift,
        "hot_cold_imbalance_pct": imbalance,
        "n_samples": len(hist),
        "reason": "converged" if converged else "not_converged",
    }


def _load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def _get_recommended_steps(config: dict, ra: float, n: int) -> Optional[int]:
    """Get recommended steps from config for given Ra and N."""
    ra_key = f"ra{ra:.0e}".replace("e+0", "e")
    if ra_key in config.get("recommended_steps", {}):
        ra_steps = config["recommended_steps"][ra_key]
        if str(n) in ra_steps:
            return ra_steps[str(n)]
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="3D cubic thermal cavity (W9-A): D3Q19+D3Q7, full-way BB, "
        "fluid-masked post-collision buoyancy"
    )
    p.add_argument("--ra", type=float, required=True, help="Rayleigh number")
    p.add_argument("--n", type=int, required=True, help="cubic grid size")
    p.add_argument("--steps", type=int, default=None, help="total steps (optional if --auto-steady or --config)")
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
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to JSON config file with recommended steps and steady-state criteria",
    )
    p.add_argument(
        "--auto-steady",
        action="store_true",
        help="automatically extend simulation until steady-state convergence",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=2000000,
        help="maximum steps for auto-steady mode (default: 2M)",
    )
    p.add_argument(
        "--steady-drift-threshold",
        type=float,
        default=None,
        help="window drift threshold %% for steady-state (overrides config)",
    )
    args = p.parse_args()

    # Load config if provided
    config = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path(__file__).resolve().parent / config_path
        config = _load_config(config_path)
        print(f"[config] loaded {config_path}")

    # Determine steps
    steps = args.steps
    if steps is None and config is not None:
        rec = _get_recommended_steps(config, args.ra, args.n)
        if rec is not None:
            steps = rec
            print(f"[config] recommended steps for ra={args.ra:.0e} n={args.n}: {steps}")

    if steps is None and not args.auto_steady:
        p.error("--steps is required unless --auto-steady or --config provides recommended steps")

    # Get steady-state criteria
    steady_criteria = {}
    if config and "steady_state_criteria" in config:
        steady_criteria = config["steady_state_criteria"]
    if args.steady_drift_threshold is not None:
        steady_criteria["window_drift_threshold_pct"] = args.steady_drift_threshold

    window_drift_threshold = steady_criteria.get("window_drift_threshold_pct", 0.01)
    imbalance_threshold = steady_criteria.get("hot_cold_imbalance_threshold_pct", 0.1)
    window_fraction = steady_criteria.get("window_fraction", 0.2)

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

    # Determine total steps for the loop
    if args.auto_steady:
        max_steps = args.max_steps
        print(f"[auto-steady] mode enabled, max_steps={max_steps}, drift_threshold={window_drift_threshold}%")
    else:
        max_steps = steps

    actual_steps = 0
    converged_at = None
    for step in range(max_steps):
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

        if step % args.sample == 0 or step == max_steps - 1:
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

            # Check steady-state convergence in auto mode
            if args.auto_steady and len(hist) >= 20 and converged_at is None:
                ss = _check_steady_state(hist, window_drift_threshold, imbalance_threshold, window_fraction)
                if ss["converged"]:
                    converged_at = step
                    print(
                        f"[auto-steady] CONVERGED at step {step}: "
                        f"drift={ss['window_drift_pct']:.4f}%, "
                        f"imbalance={ss['hot_cold_imbalance_pct']:.4f}%, "
                        f"samples={ss['n_samples']}"
                    )
                    actual_steps = step + 1
                    break

            # Progress logging every 10k steps
            if step > 0 and step % 10000 == 0:
                nu_val = hist[-1]["nu"] if hist else float("nan")
                print(f"[{tag}] step={step} Nu={nu_val:.5f} u_max={hist[-1]['u_max']:.4e}")

        actual_steps = step + 1

    elapsed = time.time() - t0

    # Compute final steady-state metrics
    ss_final = _check_steady_state(hist, window_drift_threshold, imbalance_threshold, window_fraction)

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
            "steps": actual_steps,
            "steps_requested": steps,
            "auto_steady": args.auto_steady,
            "converged_at": converged_at,
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
        "steady_state": {
            "converged": ss_final["converged"],
            "window_drift_pct": ss_final.get("window_drift_pct", float("nan")),
            "hot_cold_imbalance_pct": ss_final.get("hot_cold_imbalance_pct", float("nan")),
            "n_samples": ss_final.get("n_samples", 0),
            "convergence_criteria": {
                "window_drift_threshold_pct": window_drift_threshold,
                "hot_cold_imbalance_threshold_pct": imbalance_threshold,
                "window_fraction": window_fraction,
            },
        },
        "nan_at": nan_at,
        "elapsed_s": elapsed,
        "steps_per_s": actual_steps / max(elapsed, 1e-9),
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
        f"u_max={last.get('u_max', float('nan')):.4e} "
        f"steps={actual_steps} converged={ss_final['converged']} "
        f"drift={ss_final.get('window_drift_pct', float('nan')):.4f}% "
        f"{elapsed:.0f}s -> {path}"
    )


if __name__ == "__main__":
    main()
