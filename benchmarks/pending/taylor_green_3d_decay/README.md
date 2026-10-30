# B17-3D-decay: 3D Taylor–Green 涡衰减（SDAA 验证扫荡）

**状态：🟡 SDAA 扫荡完成（CPU eager fallback，三档 ≤3%，单调收敛）**

## 物理问题

周期域 [0,N)³（晶格单位 N，Δx=Δt=1）上的 3D Taylor–Green 涡，初始场
（OpenLB tgv3d 同款）：

```
u = ( U0·sin(kx)·cos(ky)·cos(kz),  -U0·cos(kx)·sin(ky)·cos(kz),  0 )
ν = (τ − 1/2)/3,   k = 2π/N,   Re = U0·N/ν
```

动能衰减率理论值：γ_E_theory = 6νk²（波矢 (±k,±k,±k)，|κ|²=3k²）

## SDAA 扫荡结果（2026-01，CPU eager fallback）

| N | τ | steps | γ_E_sim | γ_E_theory | err_E | R² | ≤3% |
|---|-----|-------|-----------|------------------|-------|-----|-----|
| 32 | 0.700 | 500 | 1.5615e-02 | 1.5421e-02 | **+1.258%** ✅ | 0.999995 | ✅ |
| 48 | 0.800 | 1000 | 1.0341e-02 | 1.0281e-02 | **+0.582%** ✅ | 0.999996 | ✅ |
| 64 | 0.900 | 1000 | 7.7550e-03 | 7.7106e-03 | **+0.576%** ✅ | 0.999995 | ✅ |

- **单调收敛**：1.258% → 0.582% → 0.576% ✅
- **指数性**：R² ≥ 0.999995 ✅
- **质量守恒**：质量漂移 ~1e-5 量级

## 与 GPU 参考对比

| N | GPU err_E (verified) | CPU err_E (SDAA sweep) | 差异 |
|---|---------------------|----------------------|------|
| 64 | +0.309% (2000 steps) | +0.576% (1000 steps) | CPU 步数减半，e-folds 不足 |

CPU 误差略高是因为步数受限（auto_steps 对 N=64 给 2000 步，CPU 跑需 ~180s，
受超时限制只跑了 1000 步）。物理上完全一致（同一代码路径，eager fallback）。

## SDAA 编译状态

- `torch_sdaa` 已安装，但无 SDAA 加速器设备（`SDAA_VISIBLE_DEVICES=8` 但无硬件）
- `compile_route` 自动回退到 eager 模式（`teco_inductor` 对 LBM 整步链编译不完整）
- 数值结果与纯 CPU eager 一致（bit-exact）

## 晋级路径

- 本案例与 `verified/taylor_green_3d` 物理相同，区别在于 SDAA 设备验证
- 已有 verified 案例（GPU 三档全过 ≤3%，单调收敛）
- 本扫荡确认 SDAA 路径（eager fallback）可运行且精度达标
- 待 GPU/SDAA 硬件可用后补跑完整步数（2000+）以对齐 verified 标准

## 复现命令

```bash
cd benchmarks/verified/taylor_green_3d
python run.py --n 64 --re 24 --steps 1000 --device cpu --threads 16 --compile-mode eager
```

## 参考

- 完整 verified 结果：`benchmarks/verified/taylor_green_3d/result.json`
- SDAA 扫荡产物：`artifacts/sdaa_taylor_green_3d_decay_sweep.json`
- 物理推导：`benchmarks/verified/taylor_green_3d/README.md`

<!-- PROVENANCE
[
 {"v": 1.2576, "f": "artifacts/sdaa_taylor_green_3d_decay_sweep.json", "k": "sweep[0].err_e_pct"},
 {"v": 0.5816, "f": "artifacts/sdaa_taylor_green_3d_decay_sweep.json", "k": "sweep[1].err_e_pct"},
 {"v": 0.5755, "f": "artifacts/sdaa_taylor_green_3d_decay_sweep.json", "k": "sweep[2].err_e_pct"}
]
-->
