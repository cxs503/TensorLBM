#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""Rayleigh-Bénard 对流（水平板间温差驱动）— 热 LBM 扩展验证。

模型：双分布函数热 LBM（src/tensorlbm/thermal.py 的底层原语）
- 速度场 D2Q9 BGK + Guo 力格式注入 Boussinesq 浮力 F_y = ρ·g·β·(T − T_ref)，
  T_ref = (T_hot+T_cold)/2（冷流体下沉、热流体上升的对称参考态）
- 温度场 D2Q5 BGK advection-diffusion，α = (τ_T − 1/2)/3
- 水平向（x）：周期流（solver.stream / temperature_stream 均按模取索引）
- 上下壁（y）：no-slip 用 pre-streaming 半程反弹；温度底部 T=T_hot=1、
  顶部 T=T_cold=0 用 anti-bounce-back（half-way 二阶 Dirichlet，壁面位于
  y=∓0.5；离散稳态解为节点端点固定 + 内部线性——见 README 的 Nu 口径讨论）

无量纲：Ra = g·β·ΔT·H³/(ν·α)，Pr = ν/α；板间距 H = ny（格子单位，
half-way 壁面 y=−0.5 至 y=ny−0.5）。取 ΔT = 1（T_hot=1, T_cold=0），
τ = 0.6（ν = 1/30），Pr = 0.71 → α = ν/Pr，τ_T = 3α + 1/2，
g·β = Ra·ν·α/H³。

参考（刚-刚边界、Pr=0.71、2D roll 稳态）：
- 临界 Rayleigh 数 Ra_c ≈ 1707.76（Chandrasekhar 1961）——Ra=1500 亚临界
  无流动（u_max→0，Nu→1 纯导热），Ra=2000 超临界出现对流
- Nu(Ra)：Clever & Busse (1974) JFM 65:625 — Ra=1e4: Nu≈2.16；
  Ra=1e5: Nu≈4.22。另有常见 2D 数值报告 Ra=1e4 时 Nu≈2.2–2.3
  （与 de Vahl Davis 方腔 Nu=2.243 同量级），本 benchmark 以
  Nu_ref(1e4)=2.24 为判定基准（容差 3% 覆盖 2.16–2.30 区间）。

Nu 口径：壁面平均 Nu = −∂T/∂y·H/ΔT（底部/顶部热通量）——
- grad1/grad2：节点一阶/二阶单侧差分（D2Q5 ABB 的离散稳态解将端点节点
  固定为壁面温度、内部线性，故节点差分与离散解自洽）
- halfway：一阶差分取在 half-way 壁面位置（T[0]≈壁面值时不适用，仅报告）

判定：真实模拟（无外推），Ra=1e4 的 Nu 相对 2.24 误差 ≤3%，
≥2 档网格收敛（Nu 相对差 <5%），且亚临界 Ra=1500 的 u_max 比超临界
低 3 个数量级以上（验证 Ra_c≈1708 附近的转变）。
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402
from tensorlbm.thermal import (  # noqa: E402
    W5, buoyancy_force, collide_bgk_force, pre_streaming_bounce_back,
    temperature_collision, temperature_equilibrium, temperature_stream,
)

NU_REF_1E4 = 2.24   # Ra=1e4, Pr=0.71 刚-刚 2D 稳态（Clever-Busse 2.16 与常见 2D 数值 2.2-2.3 的中值）
NU_REF_1E5 = 4.22   # Clever & Busse (1974)，Ra=1e5
RA_CRIT = 1707.76   # Chandrasekhar (1961)，刚-刚边界


def rb_params(nx, ny, ra, pr, tau):
    """RB 版格子参数：H = ny（板间距），其余同 thermal.thermal_params。"""
    nu = (tau - 0.5) / 3.0
    alpha = nu / pr
    tau_T = 3.0 * alpha + 0.5
    H = float(ny)
    g_beta = ra * nu * alpha / (H ** 3)  # ΔT = 1
    return {"nu": nu, "alpha": alpha, "tau_T": tau_T, "H": H, "g_beta": g_beta}


def apply_rb_temperature_boundaries(g, t_hot, t_cold):
    """RB 温度边界：底部 y=0 热 T_hot（壁面 y=−0.5，缺失方向 3 +y）、
    顶部 y=ny−1 冷 T_cold（缺失方向 4 −y），anti-bounce-back；水平向周期
    由 temperature_stream 处理（% nx 取模），无需额外边界。

    与 thermal.apply_temperature_boundaries 的左右壁 ABB 同构，仅旋转 90°。
    """
    g = g.clone()
    w = W5.to(g.device)
    g[3, 0, :] = -g[4, 0, :] + 2.0 * w[3] * t_hot    # 底部（热）
    g[4, -1, :] = -g[3, -1, :] + 2.0 * w[4] * t_cold  # 顶部（冷）
    return g


def nusselt_number_rb(T, H, dT, t_hot, t_cold, mode="grad2"):
    """上下壁平均 Nusselt 数：Nu = −∂T/∂y·H/ΔT（底部热通量向上为正）。

    mode：
      'halfway' : 一阶差分取在 half-way 壁面 y=∓0.5：
                  Nu_bottom = 2·(T[0] − T_hot)·H/ΔT（符号约定见下）
      'grad1'   : 节点一阶单侧差分
      'grad2'   : 节点二阶单侧差分（默认，与方腔 benchmark 口径一致）
    注意 D2Q5 ABB 离散稳态解将端点节点固定为壁面温度，'halfway' 口径
    在端点节点上会退化（纯导热给 Nu≈0），故主判定用 grad1/grad2。
    """
    if mode == "halfway":
        # ∂T/∂y|_wall ≈ (T[0] − T_hot)/0.5（底部），Nu = −∂T/∂y·H/ΔT = 2(T_hot−T[0])·H/ΔT
        nu_bottom = 2.0 * (t_hot - T[0, :]) * H / dT
        nu_top = 2.0 * (T[-1, :] - t_cold) * H / dT
    elif mode == "grad1":
        nu_bottom = (T[0, :] - T[1, :]) * H / dT
        nu_top = (T[-1, :] - T[-2, :]) * H / dT
    elif mode == "grad2":
        nu_bottom = (1.5 * T[0, :] - 2.0 * T[1, :] + 0.5 * T[2, :]) * H / dT
        nu_top = (1.5 * T[-1, :] - 2.0 * T[-2, :] + 0.5 * T[-3, :]) * H / dT
    else:
        raise ValueError(f"unknown mode: {mode}")
    nb = float(nu_bottom.mean().item())
    nt = float(nu_top.mean().item())
    return {"nu_bottom": nb, "nu_top": nt, "nu": 0.5 * (nb + nt)}


def simulate_rb(nx, ny, ra, pr=0.71, tau=0.6, t_hot=1.0, t_cold=0.0,
                steps=300000, device="cpu", amp=0.01, report_every=20000,
                nu_mode="grad2"):
    """RB 双分布 LBM 主循环。初始场：线性温度 + 单 roll 扰动（波长=域宽，
    垂直半波——线性稳定性最不稳定模，加速收敛）。"""
    device = torch.device(device)
    p = rb_params(nx, ny, ra, pr, tau)
    nu_lat, alpha_lat, tau_T, H, g_beta = p["nu"], p["alpha"], p["tau_T"], p["H"], p["g_beta"]
    dT = t_hot - t_cold
    t_ref = 0.5 * (t_hot + t_cold)

    y = torch.arange(ny, device=device).float()
    x = torch.arange(nx, device=device).float()
    T_lin = t_hot - dT * (y + 0.5) / H
    T0 = T_lin.view(ny, 1).expand(ny, nx).clone()
    pert = amp * torch.sin(2.0 * math.pi * x / nx).view(1, nx) * \
        torch.sin(math.pi * (y + 0.5) / H).view(ny, 1)
    T0 = T0 + pert

    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, u0, u0)
    g = temperature_equilibrium(T0, u0, u0)

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    interior = ~wall

    def _macros():
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)
        return rho, ux, uy, T

    rho, ux, uy, T = _macros()
    ux_prev, uy_prev, T_prev = ux.clone(), uy.clone(), T.clone()
    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        rho, ux, uy, T = _macros()
        F = buoyancy_force(rho, T, g_beta, t_ref=t_ref)
        # 温度：碰撞（用当前 u）→ 周期流 → ABB 上下边界
        g = temperature_collision(g, tau_T, ux, uy)
        g = temperature_stream(g)
        g = apply_rb_temperature_boundaries(g, t_hot, t_cold)
        # 速度：碰撞（Guo 力）→ pre-streaming 半程反弹（上下壁）→ 周期流
        f_pre = f
        f = collide_bgk_force(f, tau, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)

        if step % report_every == 0 or step == steps:
            rho, ux, uy, T = _macros()
            du = max(torch.abs(ux[interior] - ux_prev[interior]).max().item(),
                     torch.abs(uy[interior] - uy_prev[interior]).max().item())
            dTres = torch.abs(T[interior] - T_prev[interior]).max().item()
            um = float((ux ** 2 + uy ** 2).sqrt().max().item())
            nu_cur = nusselt_number_rb(T, H, dT, t_hot, t_cold, mode=nu_mode)
            hist.append({"step": step, "resid_u": du, "resid_T": dTres,
                         "u_max": um, **nu_cur})
            ux_prev, uy_prev, T_prev = ux.clone(), uy.clone(), T.clone()

    elapsed = time.time() - t0
    rho, ux, uy, T = _macros()
    T_cpu = T.detach().cpu()
    u_mag = (ux.detach().cpu() ** 2 + uy.detach().cpu() ** 2).sqrt()

    nu_g2 = nusselt_number_rb(T_cpu, H, dT, t_hot, t_cold, mode="grad2")
    nu_g1 = nusselt_number_rb(T_cpu, H, dT, t_hot, t_cold, mode="grad1")
    nu_hw = nusselt_number_rb(T_cpu, H, dT, t_hot, t_cold, mode="halfway")

    return {
        "nx": nx, "ny": ny, "ra": ra, "pr": pr, "tau": tau, "tau_T": tau_T,
        "nu_lat": nu_lat, "alpha": alpha_lat, "g_beta": g_beta, "H": H,
        "steps": steps, "elapsed_s": round(elapsed, 1),
        "last_resid_u": hist[-1]["resid_u"] if hist else None,
        "last_resid_T": hist[-1]["resid_T"] if hist else None,
        "nu_grad2": nu_g2["nu"], "nu_bottom_grad2": nu_g2["nu_bottom"],
        "nu_top_grad2": nu_g2["nu_top"],
        "nu_grad1": nu_g1["nu"], "nu_halfway": nu_hw["nu"],
        "u_max": float(u_mag.max().item()),
        "T_min": float(T_cpu.min().item()), "T_max": float(T_cpu.max().item()),
        "nu_history": hist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=0, help="0=auto (nx//2)")
    ap.add_argument("--ra", type=float, default=1e4)
    ap.add_argument("--pr", type=float, default=0.71)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--amp", type=float, default=0.01)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="/tmp/rayleigh_benard_scan.json")
    args = ap.parse_args()

    ny = args.ny if args.ny > 0 else args.nx // 2
    device = torch.device(args.device)
    torch.set_num_threads(32)

    res = simulate_rb(args.nx, ny, args.ra, pr=args.pr, tau=args.tau,
                      steps=args.steps, device=device, amp=args.amp)
    key = {k: res[k] for k in
           ["nx", "ny", "ra", "pr", "tau", "tau_T", "steps", "elapsed_s",
            "last_resid_u", "last_resid_T", "u_max", "T_min", "T_max",
            "nu_grad2", "nu_bottom_grad2", "nu_top_grad2", "nu_grad1", "nu_halfway"]}
    key["nu_ref_1e4"] = NU_REF_1E4
    key["nu_ref_1e5"] = NU_REF_1E5
    if abs(args.ra - 1e4) < 1:
        key["err_pct"] = round(100.0 * abs(key["nu_grad2"] - NU_REF_1E4) / NU_REF_1E4, 3)
    elif abs(args.ra - 1e5) < 1e3:
        key["err_pct"] = round(100.0 * abs(key["nu_grad2"] - NU_REF_1E5) / NU_REF_1E5, 3)
    else:
        key["err_pct"] = None
    key["nu_history"] = res["nu_history"]

    Path(args.out).write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in key.items() if k != "nu_history"}))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
