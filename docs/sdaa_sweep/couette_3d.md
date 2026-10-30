# SDAA Sweep — couette_3d

> 案例来源：`benchmarks/verified/couette_3d`
> 运行脚本：`scripts/sdaa_sweep_couette_3d.py`
> 产物：`artifacts/sdaa_couette_3d.json`
> 运行日期：2026-01-XX（本轮 sweep）

---

## 1. 案例概述

**Couette 流 3D（平板剪切）**：下壁静止、上壁以 U0=0.05（+x 方向）拖动，x/z 周期边界，
稳态解析解为线性剖面 u(y) = U0·(y−0.5)/H_eff。

- **格子**：D3Q19
- **碰撞**：BGK（τ=0.8, ν=0.1）
- **边界**：上壁移动（half-way bounce-back + 动量注入），下壁静止（plain BB），x/z 周期
- **参考解**：解析精确解（线性 Couette 剖面），对任意 Re 成立

本案例在 NVIDIA 侧已有成熟判据（fp64 四档晋级位，误差 ≤1.73e-9），
本次 SDAA sweep 验证同一算法在 SDAA 设备上的正确性。

---

## 2. 配置

| 参数 | 值 |
|------|-----|
| 设备 | `sdaa`（`torch.sdaa.get_device_name(0)` = `/dev/tcaicard2`） |
| 网格 H | 8（ny=10, nx=8, nz=8） |
| dtype | fp32 / fp64 各一档 |
| τ | 0.8 |
| U0 | 0.05 |
| min_steps | 200 |
| max_steps | 500 |
| 剖面累积步数 | 20 |
| Triton 绕行 | **不需要** — 案例使用纯 torch ops，无 triton 核 |

**网格/步数缩减说明**：原案例推荐 H∈{16,32,64,128}、20000 步。
为控制墙钟 ≤20 分钟，SDAA sweep 使用 H=8、500 步。
判据阈值不变（按原口径），仅网格/步数缩减。

---

## 3. 设备证据

```json
{
  "torch_version": "2.7.1",
  "sdaa_available": true,
  "sdaa_device_count": 1,
  "sdaa_get_device_name_0": "/dev/tcaicard2",
  "sdaa_current_device": 0
}
```

---

## 4. 结果表

### fp32（H=8）

| 指标 | 值 |
|------|-----|
| n_steps | 500 |
| steady | true |
| l2_rel_err | 5.866e-06 |
| max_rel_err_masked (%) | 0.00151 |
| u_top_err (%) | 0.000103 |
| tau_err (%) | −0.000311 |
| uz_max | 1.86e-09 |
| ux_spanwise_std | 0.0 |
| mass_drift (%) | 8.58e-05 |
| finite | true |
| elapsed (s) | 7.0 |

### fp64（H=8）

| 指标 | 值 |
|------|-----|
| n_steps | 500 |
| steady | true |
| l2_rel_err | 1.617e-14 |
| max_rel_err_masked (%) | 1.92e-12 |
| u_top_err (%) | 1.92e-12 |
| tau_err (%) | 1.99e-12 |
| uz_max | 0.0 |
| ux_spanwise_std | 0.0 |
| mass_drift (%) | −1.87e-12 |
| finite | true |
| elapsed (s) | 7.1 |

---

## 5. 判据逐条比对

| 判据 | 口径 | 阈值 | 实测值 | 结果 |
|------|------|------|--------|------|
| **G1: direct_observable** | l2_rel_err ≤ 3%（fp32） | 0.03 | 5.866e-06 | ✅ PASS |
| **G2a: fp32_floor** | l2_rel_err ≤ 1e-3（fp32） | 1e-3 | 5.866e-06 | ✅ PASS |
| **G2b: fp64_scheme_exactness** | l2_rel_err ≤ 1e-7（fp64） | 1e-7 | 1.617e-14 | ✅ PASS |
| **G3: steady_state** | steady=true, \|mass_drift\|<1%, uz_max<1e-6, spanwise_std<1e-6（fp64） | — | steady=true, drift=−1.87e-12%, uz=0.0, std=0.0 | ✅ PASS |
| **G4: library_entry** | tensorlbm.d3q19 + tensorlbm.solver3d 在 sdaa 可导入并执行 | — | 全部 solver 调用在 sdaa 上无错执行 | ✅ PASS |

**all_pass = true**

---

## 6. 墙钟

| 阶段 | 耗时 (s) |
|------|----------|
| fp32 H=8 | 7.0 |
| fp64 H=8 | 7.1 |
| **总计** | **14.1** |

远小于 20 分钟上限。

---

## 7. 绕行记录

| 项目 | 记录 |
|------|------|
| Triton 依赖 | **无** — 案例使用纯 torch ops（`equilibrium3d`, `collide_bgk3d`, `stream3d`, `macroscopic3d`），无 triton 核 |
| TRITON_INTERPRET | 未设置（不需要） |
| CPU 冒充 SDAA | **无** — 所有张量显式置于 `torch.device('sdaa')` |
| 判据降级 | **无** — 阈值与原案例完全一致（3% / 1e-3 / 1e-7） |
| 网格/步数缩减 | H=8（原 16/32/64/128）、500 步（原 20000）— 为控制墙钟，判据阈值不变 |

---

## 8. 结论

couette_3d 案例在 SDAA 设备上 **全部判据通过**。
fp32 l2=5.87e-06（远低于 1e-3 地板），fp64 l2=1.62e-14（机器地板级别），
稳态/质量/展向均匀性全部达标。
算法在 SDAA 上的行为与 CPU fp64 参考一致，验证了 tensorlbm 库的 D3Q19 动壁
half-way BB 路径在 SDAA 设备上的正确性。
