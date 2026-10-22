# W11-A2 预注册（prereg）— 2026-10-21 冻结

## 背景
W11-A formal 六档（Re={20,40} × D={40,80,160}，up=10D）全部 FAIL（err 5.43-6.06% 全超 3% 门）。
根因诊断（floor_20261008）：上游域截断（up=10D → 入口 Dirichlet 自由流钳制）。
单变量阶梯证明 up 扩到 40D 即入门（E-D: up=40D, lx=70D, h=20D → err +2.236%）。

## 晋级动作（W11-A2）
扩上游 up≥40D 六档重跑，冻结以下协议：

### §2 求解器（与 W11-A 完全相同）
- collide_mrt(tau, tau_field=sponge) → stream → forces (post-stream, pre-BB) → far_field_bc_2d
- tau = 0.5 + 3*nu
- sponge: W_sp = 10D, power 2, tau_eff = tau*(1 + 10*sigma)
- 自由流初始化（u=U_LB, v=0, rho=1）

### §3 域（唯一变更）
- **LX = 70D**（W11-A: 40D）
- **UP = 40D**（W11-A: 10D）← 关键变更
- **H = 20D**（不变）
- 圆柱中心：cx = UP*D = 40D, cy = H*D

### §4 几何（与 W11-A 完全相同）
- area-equivalent mask（bisection on r_mask so R_area = sqrt(N_solid/pi) closest to D/2）
- nu = u_lb*D_eff/Re
- Cd 分母 0.5*u_lb^2*D_eff

### §5 梯子（与 W11-A 完全相同）
- u_lb = 0.05 固定
- tau 由 nu 推出
- 六档：Re={20,40} × D={40,80,160}

### §6 定常协议（与 W11-A 完全相同）
- t_c = D_eff/u_lb steps
- 相邻 10-t_c 窗口，|dm|/m < 0.15%，从 60 t_c 开始检查
- cap 150 t_c
- Cd = mean last 20 t_c
- 全轨迹存档（npz）

### §7 止损（与 W11-A 完全相同）
- D160 单跑 GPU wall > 17 h → stop, report
- 任何 NaN → stop
- 无任何修正

## 参考（与 W11-A 完全相同）
- Re=20: Cd_ref = 2.0001 (Fornberg 1980 JFM 98:819)
- Re=40: Cd_ref = 1.498 (Fornberg 1985 JCP 61:297)

## 判决标准
- G1 精度门：全部六档 |err| ≤ 3%
- G2 单调收敛：≥2 档网格严格单调收敛（|err| 随 D 增加单调下降）
- G3 重锁链：tau 相对偏差全档 0.0
- G4 参考先锁：prereg 冻结日期 ≤ 运行日期

## 冻结日期
2026-10-21

## 预算
≈25-35 GPU·h（六档，GPU5）

## 运行命令
```bash
cd /nfs/wangxi/tensor_eco/agent_work/m120/TensorLBM
CUDA_VISIBLE_DEVICES=5 /nfs/wangxi/venvs/tensorlbm/bin/python \
  benchmarks/pending/cylinder_re40_st/w11/src/run_formal_w11a2.py \
  --re 40 --D 40
```

六档全跑：
```bash
for re in 20 40; do
  for D in 40 80 160; do
    CUDA_VISIBLE_DEVICES=5 /nfs/wangxi/venvs/tensorlbm/bin/python \
      benchmarks/pending/cylinder_re40_st/w11/src/run_formal_w11a2.py \
      --re $re --D $D
  done
done
```
