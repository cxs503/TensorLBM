# Mission 12 补报：SDAA 验证案例扫荡 — acoustics (D2Q9 平面声波)

> 本报告为 mission 12 交付后补报（scenario_key: report-backfill）。
> 所有数字均从 `artifacts/sdaa_acoustics.json` 程序化提取，零手抄。

## 1 目的

在 SDAA 异构加速卡（`/dev/tcaicard12`）上复现 TensorLBM `benchmarks/verified/acoustics` 案例，
验证 D2Q9 平面声波的线性色散关系与衰减率是否与连续理论值一致，
并确认 SDAA float64 计算精度与原始 CPU 结果等价。

## 2 方法

| 项目 | 值 |
|---|---|
| 案例路径 | `benchmarks/verified/acoustics` |
| 物理问题 | D2Q9 平面声波：线性色散与衰减 |
| 库内核 | `collide_bgk` / `stream` / `d2q9.equilibrium` / `d2q9.macroscopic` |
| SDAA 设备 | `sdaa:0` → `/dev/tcaicard12` |
| 数据类型 | float64 |
| TRITON_INTERPRET 绕行 | 是（import 路径间接依赖 triton） |
| 总墙钟 | 15.3386 s |
| 业务代码修改 | **无** — 仅写独立运行脚本 `scripts/sdaa_sweep_acoustics.py` |

### 预注册判据

| 判据 | 描述 | 阈值 | 单调性要求 |
|---|---|---|---|
| 判据 A | \|c_meas/c_s − 1\| ≤ 3% | ≤ 3% | 随 N 单调下降 |
| 判据 B | \|δ_meas/(νk²) − 1\| ≤ 3% | ≤ 3% | 随 N 单调下降 |

其中 c_s = 1/√3 ≈ 0.577350269，ν = c_s²(τ−½) = (τ−½)/3。

## 3 关键结果

### 3.1 收敛表（τ=1.0, ε=1e-3）

| N | c_meas | c_err vs 连续 (%) | δ_meas | δ_err vs 连续 (%) | 墙钟 (s) | R²_amp | R²_phase |
|---|---|---|---|---|---|---|---|
| 16 | 0.5786145708 | +0.2190 | 2.603994e-02 | +1.3145 | 1.99 | 0.9999999999 | 0.9999978 |
| 32 | 0.5776616356 | +0.0539 | 6.446282e-03 | +0.3231 | 4.18 | 0.9999999999 | 0.9999995 |
| 64 | 0.5774279884 | +0.0135 | 1.607686e-03 | +0.0812 | 8.62 | 0.9999999996 | 0.9999999 |

### 3.2 判据逐条比对

| 判据 | 最大误差 (%) | 阈值 (%) | 单调 | 结果 |
|---|---|---|---|---|
| A: 波速误差 | 0.2190 | 3.0 | ✓ | **PASS** |
| B: 衰减误差 | 1.3145 | 3.0 | ✓ | **PASS** |
| **总体** | — | — | — | **PASS** |

### 3.3 SDAA vs CPU 一致性

| N | c_err CPU (%) | c_err SDAA (%) | δ_err CPU (%) | δ_err SDAA (%) |
|---|---|---|---|---|
| 16 | +0.2190 | +0.2190 | +1.3145 | +1.3145 |
| 32 | +0.0539 | +0.0539 | +0.3231 | +0.3231 |
| 64 | +0.0135 | +0.0135 | +0.0812 | +0.0812 |

SDAA 结果与原始 CPU 结果在有效数字内完全一致（差异 < 1e-10 相对），
证明 SDAA float64 计算精度与 CPU 等价。

### 3.4 辅助指标

| N | counterwave ratio η/2 | max\|ρ−1\| at end | finite check |
|---|---|---|---|
| 16 | 5.74% | 3.33e-06 | ✓ |
| 32 | 2.84% | 9.91e-06 | ✓ |
| 64 | 1.42% | 7.10e-05 | ✓ |

## 4 审计与局限

### 4.1 绕行措施

- `tensorlbm._voxel_kernels` 和 `tensorlbm.triton_fused` 在 import 时触发 triton 加载；
  本案例内核（collide_bgk/stream/equilibrium/macroscopic）本身不使用 triton，但 import 路径间接依赖。
  设置 `TRITON_INTERPRET=1` 后正常加载。
- **无业务代码修改** — 仅写独立运行脚本 `scripts/sdaa_sweep_acoustics.py`。

### 4.2 局限性

1. 仅测试了 D2Q9 单一方程组；D3Q19/D3Q27 声传播尚未在 SDAA 上验证。
2. 网格范围 N∈{16,32,64}，未测试更大网格（N=128+）以确认渐近收敛阶。
3. 仅 τ=1.0 单点；不同弛豫时间下的色散/衰减精度未扫荡。
4. 反向反射波（counterwave ratio η/2 在 N=16 时为 5.74%，N=64 时为 1.42%），
   表明边界反射随网格细化而减小，但未做正式边界条件精度审计。
5. 墙钟总计 15.3 s，性能基准未与 CPU 做正式加速比对比。

### 4.3 工件清单

| 工件 | 路径 |
|---|---|
| 运行脚本 | `scripts/sdaa_sweep_acoustics.py` |
| JSON 输出 | `artifacts/sdaa_acoustics.json` |
| 扫荡文档 | `docs/sdaa_sweep/acoustics.md` |
