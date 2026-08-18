#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B26: 后向台阶 Re=100 再附着长度验证（Armaly 1984 参考 X_r/h≈3）。

几何: 台阶高 step_h 格, 膨胀比 ER=(ny-2)/(ny-1-step_h)≈1.94 (Armaly 实验 1.94)。
入口: 充分发展抛物线剖面 u(y)=4·U_max·y'·(H-y')/H^2, y'=y-(step_h-0.5), H=ny-1-step_h,
      经 Zou/He 速度 BC 施加 (zou_he_inlet_velocity 支持张量剖面)。
Re = U_max·step_h/ν = 100 (Armaly 定义: 基于最大入口速度与台阶高)。
再附着点: 台阶后壁面剪切 τ_w∝ux(y=1) 的过零点 (线性亚格插值), 距台阶下游立面
          x = x_step-0.5 的距离归一化: X_r/h。

实现说明（真实模拟, 无外推）:
- run_backward_facing_step 为共性模块入口, 本脚本以 monkey-patch 注入两点,
  不修改库文件 (缺口记录见 /tmp/backward_gap.md):
  1) bfs._apply_bfs_inlet   -> 抛物线入口 (库内硬编码均匀入口)
  2) bfs.measure_reattachment_length -> 亚格插值测量 + 捕获速度场快照
- 输出: run_dir/run_metadata.json (模块), benchmarks/verified/backward_step/result.json (本脚本)
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

import numpy as np
import torch

import tensorlbm.backward_facing_step as bfs
from tensorlbm.backward_facing_step import (
    BackwardFacingStepConfig,
    run_backward_facing_step,
)
from tensorlbm.boundaries import zou_he_inlet_velocity, zou_he_outlet_pressure
from tensorlbm.solver import collide_bgk

# 诊断: 强制 BGK 碰撞 (对照 MRT; τ=0.5585 时模块自动选 MRT)
if os.environ.get("BFS_FORCE_BGK", "0") == "1":
    bfs.collide_mrt = collide_bgk  # noqa: SLF001 (模块内 _collide_base 引用模块全局名)
    print(">>> forced BGK collision (BFS_FORCE_BGK=1)", flush=True)

# ---------------------------------------------------------------------------
# 参数 (环境变量可覆盖)
# ---------------------------------------------------------------------------
NX = int(os.environ.get("BFS_NX", "400"))
NY = int(os.environ.get("BFS_NY", "80"))
STEP_H = int(os.environ.get("BFS_STEP_H", "39"))
X_STEP = int(os.environ.get("BFS_X_STEP", "80"))
U_MAX = float(os.environ.get("BFS_U_MAX", "0.05"))          # 抛物线最大入口速度
RE = float(os.environ.get("BFS_RE", "100.0"))               # Re = U_max*h/nu
N_STEPS = int(os.environ.get("BFS_N_STEPS", "250000"))
OUT_INTERVAL = int(os.environ.get("BFS_OUT_INTERVAL", "10000"))
DEVICE = os.environ.get("BFS_DEVICE", "cpu")
OUT_ROOT = Path(
    os.environ.get(
        "BFS_OUT_ROOT",
        "/home/wxsc/cxs/TensorLBM/results_bench_b26_bfs_re100",
    )
)
VERIFIED_DIR = Path(
    os.environ.get("BFS_VERIFIED_DIR", "/home/wxsc/cxs/TensorLBM/benchmarks/verified/backward_step")
)

ER = (NY - 2) / (NY - 1 - STEP_H)   # 下游/上游通道高度比 (格)
NU = U_MAX * STEP_H / RE
TAU = 3.0 * NU + 0.5


# ---------------------------------------------------------------------------
# 1) 抛物线入口 BC (monkey-patch 注入; 库内 _apply_bfs_inlet 硬编码均匀入口)
# ---------------------------------------------------------------------------
def make_parabolic_profile(ny: int, step_h: int, u_max: float) -> np.ndarray:
    """充分发展抛物线剖面 (通道壁面在 y=step_h-0.5 与 y=ny-0.5, 高 H=ny-1-step_h)。"""
    H = float(ny - 1 - step_h)
    yp = np.arange(ny, dtype=np.float64) - (step_h - 0.5)
    u = np.zeros(ny, dtype=np.float64)
    inside = (yp >= 0.0) & (yp <= H)
    u[inside] = 4.0 * u_max * yp[inside] * (H - yp[inside]) / (H * H)
    return u


def _parabolic_inlet(f: torch.Tensor, u_in: float, step_h: int) -> torch.Tensor:
    """Zou/He 入口 BC, 施加抛物线速度剖面 (固体行随后被 bounce-back 覆盖)。"""
    ny = f.shape[1]
    u_profile = torch.tensor(
        make_parabolic_profile(ny, step_h, float(u_in)),
        dtype=f.dtype,
        device=f.device,
    )
    return zou_he_inlet_velocity(f, u_profile, 0.0)


bfs._apply_bfs_inlet = _parabolic_inlet  # noqa: SLF001 (benchmark 脚本注入, 不改库)

# 诊断: 均匀入口 A/B 对照 (BFS_UNIFORM_INLET=1 -> 覆盖为模块默认均匀入口)
if os.environ.get("BFS_UNIFORM_INLET", "0") == "1":
    bfs._apply_bfs_inlet = lambda f, u_in, step_h: zou_he_inlet_velocity(f, float(u_in))  # noqa: SLF001
    print(">>> uniform inlet A/B (BFS_UNIFORM_INLET=1)", flush=True)


# ---------------------------------------------------------------------------
# 1b) 出口: Zou/He 压力出口 (密度锚定) 替代零梯度 copy
#     原因: 出口流动未充分发展时 copy BC 持续质量漂移 (实测 ~0.018/步,
#     40k 步 +1.7% 且线性增长); 压力出口 rho=1 锚定, 漂移有界。
# ---------------------------------------------------------------------------
def _pressure_outlet(f: torch.Tensor) -> torch.Tensor:
    return zou_he_outlet_pressure(f, 1.0)


bfs._apply_bfs_outlet = _pressure_outlet  # noqa: SLF001 (同上)


# ---------------------------------------------------------------------------
# 2) 亚格插值再附着长度测量 + 速度场快照捕获
# ---------------------------------------------------------------------------
_captured: dict[str, list] = {"ux": []}   # 每次诊断点捕获 ux (按序 -> step=(i+1)*OUT_INTERVAL)


def _zero_crossing_col(row: np.ndarray, x_step: int) -> float | None:
    """row = ux[1, x_step:], 返回再附着点的连续列坐标 (从负到正过零点)。"""
    pos = int(np.argmax(row > 0.0))
    if pos == 0 or row[pos] <= 0.0:
        return None
    a, b = row[pos - 1], row[pos]
    if b - a == 0.0:
        return None
    return x_step + (pos - 1) + (0.0 - a) / (b - a)


def measure_reattach_subcell(ux: torch.Tensor, x_step: int, step_h: int) -> float:
    """X_r/h: 距台阶下游立面 x=x_step-0.5 的归一化再附着长度。

    壁面剪切 τ_w ∝ ux(y=1) (bounce-back 壁面位于 y=0.5, 线性近似),
    τ_w=0 即 ux(y=1)=0 的过零点; 线性亚格插值消除整格量化误差。
    """
    row = ux[1, x_step:].detach().cpu().numpy()
    zc = _zero_crossing_col(row, x_step)
    if zc is None:
        return 0.0
    return float((zc - (x_step - 0.5)) / step_h)


def _measure_capture(ux: torch.Tensor, x_step: int, step_h: int) -> float:
    xr_h = measure_reattach_subcell(ux, x_step, step_h)
    _captured["ux"].append(ux.detach().cpu().numpy().copy())
    return xr_h


bfs.measure_reattachment_length = _measure_capture  # noqa: SLF001 (同上)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    config = BackwardFacingStepConfig(
        nx=NX,
        ny=NY,
        step_h=STEP_H,
        x_step=X_STEP,
        u_in=U_MAX,
        re=RE,
        n_steps=N_STEPS,
        output_interval=OUT_INTERVAL,
        output_root=OUT_ROOT,
        run_name=f"armaly_re{int(RE)}_er{ER:.3f}_h{STEP_H}",
        seed=0,
        device=DEVICE,
        overwrite=True,
        use_compile=False,
    )
    config.validate()
    print(
        f"=== B26 BFS Re={RE} (U_max*h/nu) nx={NX} ny={NY} step_h={STEP_H} "
        f"ER={ER:.4f} tau={TAU:.4f} nu={NU:.5f} n_steps={N_STEPS} device={DEVICE} ===",
        flush=True,
    )

    run_dir = run_backward_facing_step(config)
    wall_t = time.time() - t_start

    # ---- 后处理: 收敛性与最终 X_r/h ----
    ux_snaps = _captured["ux"]
    n_snap = len(ux_snaps)
    steps_arr = [(i + 1) * OUT_INTERVAL for i in range(n_snap)]
    xr_series = [
        measure_reattach_subcell(torch.from_numpy(u), X_STEP, STEP_H) for u in ux_snaps
    ]

    final_ux = torch.from_numpy(ux_snaps[-1])
    xr_h = measure_reattach_subcell(final_ux, X_STEP, STEP_H)          # 物理 (距台阶立面)
    xr_h_mod = xr_h - 0.5 / STEP_H                                      # 模块约定 (距 x_step 列)
    err_pct = (xr_h - 3.0) / 3.0 * 100.0

    # 收敛判据: 最后 3 个快照 X_r/h 极差 ≤ 0.02 且最后两步变化 ≤ 0.01
    last3 = xr_series[-3:]
    converged = len(last3) >= 3 and (max(last3) - min(last3)) <= 0.02 and abs(last3[-1] - last3[-2]) <= 0.01
    # 末两步速度场残差 (L∞, 归一化 U_max)
    if n_snap >= 2:
        resid = float(
            np.abs(ux_snaps[-1] - ux_snaps[-2]).max() / U_MAX
        )
    else:
        resid = float("nan")

    # 导出最终速度场 (npz), 供离线复核/壁面剪切分析
    try:
        np.savez(
            run_dir / "final_ux.npz",
            ux=ux_snaps[-1],
            solid=np.asarray(bfs.make_bfs_solid_mask(NY, NX, STEP_H, X_STEP, torch.device("cpu")).numpy()),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"npz dump failed: {exc}", flush=True)

    result = {
        "case": "B26",
        "name": "backward_facing_step_re100",
        "reference": {
            "source": "Armaly, Durst, Pereira & Schoenung, JFM 127:473-496 (1983), "
                      "commonly cited as 'Armaly 1984'",
            "re_definition": "Re = U_max * step_h / nu (最大入口速度)",
            "xr_h_ref": 3.0,
            "er_ref": 1.94,
        },
        "geometry": {
            "nx": NX, "ny": NY, "step_h_cells": STEP_H, "x_step": X_STEP,
            "expansion_ratio": ER,
            "er_dev_pct": (ER - 1.94) / 1.94 * 100.0,
            "inlet": "fully_developed_parabolic (Zou/He), U_max=%.4f" % U_MAX,
        },
        "physics": {"re": RE, "nu": NU, "tau": TAU, "collision": "MRT (module auto, tau<0.60)"},
        "result": {
            "xr_h": xr_h,
            "xr_h_module_convention": xr_h_mod,
            "err_pct": err_pct,
            "xr_series": xr_series,
            "steps": steps_arr,
            "converged": bool(converged),
            "final_Linf_residual_over_Umax": resid,
            "n_steps_run": N_STEPS,
            "wall_time_s": round(wall_t, 1),
            "device": str(DEVICE),
        },
        "verified": bool(converged and abs(err_pct) <= 1.0),
        "notes": (
            "X_r/h 从台阶下游立面 (x=x_step-0.5) 起算; 壁面剪切 τ_w∝ux(y=1), "
            "过零点线性亚格插值。模块自带 measure_reattachment_length 为整格量化 "
            "(偏差 0.5 格 + 无插值), 本脚本测量为物理约定。"
        ),
    }

    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    with (VERIFIED_DIR / "result.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False, default=float)

    print(f"run_dir: {run_dir}", flush=True)
    print(f"X_r/h = {xr_h:.4f}  (module conv {xr_h_mod:.4f})  ref 3.0  err {err_pct:+.2f}%", flush=True)
    print(f"converged={converged}  resid={resid:.2e}  snapshots={n_snap}", flush=True)
    print(f"X_r series: {[round(v, 3) for v in xr_series]}", flush=True)
    print(f"wall time: {wall_t:.0f}s  -> result.json at {VERIFIED_DIR / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
