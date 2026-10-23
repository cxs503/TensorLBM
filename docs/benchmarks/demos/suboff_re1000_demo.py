#!/usr/bin/env python
"""DARPA SUBOFF 裸艇体 Re=1000 总阻力 — 场量可视化演示脚本。

与正式验证档 benchmarks/verified/suboff_re1000/run.py 使用同一组库入口：
  - tensorlbm.general_sim.GeneralSimEngine（PARAMETRIC_SUBOFF 共性情形：
    几何/单位换算/掩码/域/远场 BC 一站式构建），步进链
    lbm_step_correct（collide → NoDynamics → half-way BB → stream →
    far_field_bc_3d）+ 200 步质量修正（Re<10000 自动 BB 壁）
  - tensorlbm.d3q19.equilibrium3d / macroscopic3d
  - 测力同链：drag_pressure_integration + drag_friction_integration
    （formula='mix50'），引擎逐 20 步采样

演示档降分辨率：L_cells=48（=入库粗网格档；主档 L=80 480×169×169）、
12000 步（主档 L=80 12000 步；L=48 档 20000 步），力采样间隔放宽到 20 步。
输出对称面（z=nz/2，含艇轴）切片 |u|/p′、尾流剖面与 Cd_tot 迹线
（frontal→wetted 重标定，同正式档）；Cd 定量判据一律取正式档
result.json（Blasius 0.041995 口径）。

用法：
    python docs/benchmarks/demos/suboff_re1000_demo.py \
        --resolution 48 --steps 12000 --out demo.npz
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_SRC = os.environ.get("TENSORLBM_SRC") or str(
    Path(__file__).resolve().parents[3] / "src"
)
sys.path.insert(0, REPO_SRC)

from tensorlbm.d3q19 import macroscopic3d  # noqa: E402
from tensorlbm.general_sim import (  # noqa: E402
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)

SUBOFF_LENGTH_M = 4.356  # DARPA SUBOFF 裸艇体长 [m]（与正式档一致）
SUBOFF_RADIUS_M = 0.254  # 最大半径 [m]（L/D = 8.57）
U_PHYS = 1.0e-3  # m/s（只定 dt；Re = u·L/ν = 1000）


def wetted_dpS(u_lb, radius_lb, length_lb):
    """湿面积参考 dpS = 0.5·u²·π·D·L（Re=1000 族口径，同正式档）。"""
    return 0.5 * u_lb**2 * math.pi * (2.0 * radius_lb) * length_lb


def frontal_dpS(u_lb, radius_lb):
    return 0.5 * u_lb**2 * math.pi * radius_lb**2


def main():
    ap = argparse.ArgumentParser(description="SUBOFF Re=1000 场量演示")
    ap.add_argument("--resolution", type=int, default=48,
                    help="每艇长格数 L（正式档主档 80 / 粗档 48）")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--sample", type=int, default=20, help="力采样间隔（步）")
    ap.add_argument("--friction", default="mix50",
                    choices=["standard", "faces", "mix50"])
    ap.add_argument("--p0", default="near_wall",
                    choices=["near_wall", "far_field", "domain_avg", "inlet"])
    ap.add_argument("--collision", default="mrt", choices=["mrt", "smagorinsky"])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="suboff_re1000_demo.npz")
    ap.add_argument("--workdir", default="/tmp/suboff_re1000_demo",
                    help="引擎输出目录（演示档不落盘大文件）")
    args = ap.parse_args()

    L = args.resolution
    viscosity = U_PHYS * SUBOFF_LENGTH_M / 1000.0  # Re = 1000
    collision = (CollisionModel.SMAGORINSKY_MRT
                 if args.collision == "smagorinsky" else CollisionModel.MRT)

    config = GeneralSimConfig(
        name=f"suboff_re1000_demo_L{L}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SUBOFF,
            suboff_length=SUBOFF_LENGTH_M,
            suboff_radius=SUBOFF_RADIUS_M,
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=viscosity,
            inlet_velocity=U_PHYS,
            reference_length=SUBOFF_LENGTH_M,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=collision,
            resolution=L,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),  # 流向 6L，侧向 2L+D
            max_steps=args.steps,
            warmup_steps=None,
            snapshot_interval=10_000_000,
            force_sample_interval=args.sample,
            device=args.device,
            wall_treatment=WallTreatment.AUTO,   # Re<10000 → half-way BB
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none",
            p0_method=args.p0,
            friction_formula=args.friction,
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=0.05,
        ),
        output=OutputConfig(
            directory=args.workdir,
            formats=[],
            save_macroscopic=False,
            save_forces=True,
        ),
    )

    print(
        f"=== SUBOFF Re=1000 demo L={L} collision={args.collision} "
        f"friction={args.friction} steps={args.steps} device={args.device} ===",
        flush=True,
    )
    engine = GeneralSimEngine(config)
    setup_info = engine.setup()
    print(
        "setup: " + str({k: setup_info.get(k) for k in (
            "Re", "tau", "u_lb", "nu_lb", "domain_lu", "obstacle_cells",
            "near_wall_cells", "total_cells", "device", "auto_wall_treatment")}),
        flush=True,
    )

    t0 = time.time()
    info = engine.run()
    elapsed = time.time() - t0
    print(f"run: {info.get('status')} steps={engine.step_count} "
          f"t={elapsed:.0f}s ({elapsed / max(engine.step_count, 1) * 1000:.1f} ms/step)",
          flush=True)

    # ---- 场量提取：对称面 z=nz/2（含艇轴）切片 ----
    rho, ux, uy, uz = macroscopic3d(engine.f)
    solid = engine.solid
    nz_g, ny_g, nx_g = solid.shape
    zk = nz_g // 2
    ux_z, uy_z, uz_z, rho_z = ux[zk], uy[zk], uz[zk], rho[zk]
    solid_z = solid[zk]

    # ---- 力迹线：frontal→wetted 重标定（同正式档 run.py） ----
    u_lb = engine.uc.u_lb
    R_lb = SUBOFF_RADIUS_M / (SUBOFF_LENGTH_M / L)
    rescale = frontal_dpS(u_lb, R_lb) / wetted_dpS(u_lb, R_lb, float(L))
    log = engine.forces_log
    steps_log = np.asarray([e["step"] for e in log], dtype=np.float64)
    cd_tot_hist = np.asarray([e["cd_total"] for e in log]) * rescale
    cd_p_hist = np.asarray([e["cd_pressure"] for e in log]) * rescale
    cd_f_hist = np.asarray([e["cd_friction"] for e in log]) * rescale
    n_win = min(100, len(log))
    cd_demo = float(cd_tot_hist[-n_win:].mean()) if len(log) else float("nan")

    print(
        f"saved {args.out}: Cd_tot_demo(末{n_win}样本)={cd_demo:.5f} "
        f"t={elapsed:.0f}s（演示档，定量判据见入库 result.json）",
        flush=True,
    )
    np.savez(
        args.out,
        ux=ux_z.cpu().numpy(),
        uy=uy_z.cpu().numpy(),
        uz=uz_z.cpu().numpy(),
        rho=rho_z.cpu().numpy(),
        mask=solid_z.cpu().numpy(),
        steps_log=steps_log,
        cd_tot_hist=cd_tot_hist,
        cd_p_hist=cd_p_hist,
        cd_f_hist=cd_f_hist,
        nx=nx_g,
        ny=ny_g,
        nz=nz_g,
        L_cells=L,
        u_lb=float(u_lb),
        nu_lb=float(engine.uc.nu_lb),
        tau=float(engine.uc.tau),
        domain_lu=np.asarray([nx_g, ny_g, nz_g]),
        elapsed_s=elapsed,
        ms_per_step=elapsed / max(engine.step_count, 1) * 1000.0,
        cd_demo=cd_demo,
    )


if __name__ == "__main__":
    main()
