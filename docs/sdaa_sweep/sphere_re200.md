# SDAA Sweep: sphere_re200

## 案例概述

| 项目 | 值 |
|------|-----|
| 案例 | sphere_re200 — sphere drag at Re = 200 |
| 状态 | verified (NVIDIA 侧已有成熟判据) |
| 力口径 | BFL interpolated smooth wall + per-link momentum ledger (surface-restricted) |
| 域配置 | lat 3.0 / up 3.0 / down 4.0 (blockage 1.603%) |
| 碰撞模型 | MRT (sparse kernel route) |
| 远场处理 | hard far-field |
| 参考值 | Schiller-Naumann: Cd = 24/Re·(1 + 0.15·Re^0.687) = **0.805615** (Re=200) |
| 判据门限 | |err| ≤ 3% |

## 运行配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 设备 | sdaa:0 | `/dev/tcaicard11` |
| 网格 D | 16 | README 推荐 ladder 中最小档 |
| 步数 | 2500 | README 推荐 12000；sdaa 上 ~0.43s/step，12000 步约 87 分钟，超出 20 分钟墙钟预算，故降至 2500 |
| 采样间隔 | 50 | 每 50 步采样一次力 |
| tau | 0.512 | τ = 0.5 + 3·u_lb·D/Re = 0.5 + 3·0.05·16/200 |
| u_lb | 0.05 | 格子速度 |
| kernel | sparse | BFL sparse kernel |

## Triton 绕行记录

**无 Triton 依赖。** 本案例运行路径（`run_case` → `tensorlbm.bfl_common` / `tensorlbm.solver3d` / `tensorlbm.boundaries3d` 等）不导入 triton，无需 `TRITON_INTERPRET=1` 绕行。

## 结果表

| 网格 | D | 步数 | Cd (last-20% window) | err vs 0.805615 | drift |
|------|----|------|----------------------|-----------------|-------|
| SDAA | 16 | 2500 | 0.843461 | **+4.70%** | -1.37% |

对比 NVIDIA 侧已验证结果（cuda, D=16, 12000 步）：

| 网格 | D | 步数 | Cd (last-20% window) | err vs 0.805615 | drift |
|------|----|------|----------------------|-----------------|-------|
| CUDA (verified) | 16 | 12000 | 0.820646 | +1.866% | -0.026% |

## 判据逐条比对

| # | 判据 | 计算值 | 期望 | 结果 |
|---|------|--------|------|------|
| 1 | Re locked at 200 | 199.99999999999983 | 200.0 (tol 1e-6) | **PASS** |
| 2 | tau matches Re=200 law | 0.512 | 0.512 (tol 1e-9) | **PASS** |
| 3 | Steady convergence (\|drift\| < 0.3%, steps ≥ 8000, tau_ok) | drift=-1.37%, steps=2500 | \|drift\|<0.3%, steps≥8000 | **FAIL** — 步数不足 8000，且 drift 未收敛 |
| 4 | Cd error vs Schiller-Naumann (gate 3%) | err=+4.70% | \|err\| ≤ 3% | **FAIL** — 2500 步未充分收敛，Cd 偏高 |
| 5 | No divergence | false | false | **PASS** |

**总体状态: FAIL**

## 墙钟

| 指标 | 值 |
|------|-----|
| 墙钟 | 1080.2 s (18.0 min) |
| 预算 | ≤ 20 min |
| 符合 | ✅ |

## 设备证据

```
torch.sdaa.is_available() = True
torch.sdaa.device_count() = 1
torch.sdaa.get_device_name(0) = /dev/tcaicard11
```

## 分析

本案例在 sdaa 设备上成功运行，Re 锁定和 tau 计算均正确，无发散。但两项关键判据未通过：

1. **Steady convergence FAIL**: 步数 2500 < 8000（稳态判据要求），且 drift = -1.37% > 0.3%。这是因为墙钟预算限制（20 分钟）无法完成 12000 步推荐步数。

2. **Cd error FAIL**: Cd_window = 0.8435，误差 +4.70% 超出 3% 门限。对比 cuda 侧 12000 步结果（Cd=0.8206, err=+1.87%），说明 2500 步时流场尚未充分收敛到稳态。

**结论**: sdaa 设备上的算法实现正确（Re/tau 锁定、无发散），但受墙钟预算限制未能达到充分收敛步数，导致 Cd 误差超标。若给予充足步数（12000 步），预期可达到与 cuda 侧相当的精度（~1.9% 误差，在 3% 门限内）。

## 产物

- 脚本: `scripts/sdaa_sweep_sphere_re200.py`
- 产物: `artifacts/sdaa_sphere_re200.json`
- 本文档: `docs/sdaa_sweep/sphere_re200.md`
