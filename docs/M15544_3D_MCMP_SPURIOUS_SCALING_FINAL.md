# GAP-PP-3-3DMCMP-SPURIOUS-SCALING — m15544 返修轮交付报告

## 1. 问题设定与基准出处

**锚点**: `GAP-PP-3-3DMCMP-SPURIOUS-SCALING`（论文牵引 `JOUR-MCMP-3D` 第5节）

**物理问题**: 3D 伪势（Shan-Chen 单组分，D3Q19）模型中，静态液滴的精确解为 `u=0`。
求解器报告的任何非零速度即为**伪电流（spurious current）**。其标度律为：

```
u_max = A (Δρ)^p
```

其中 `Δρ = ρ_l − ρ_v` 为模型 EOS 的共存密度差，`p` 为幂律指数。

**验收判据（机器门）**:
| # | 判据 | 阈值 |
|---|---|---|
| C1 | u_max 表格覆盖 | ≥3 分辨率 × ≥3 密度比 |
| C2 | 幂律指数 p 不确定度 | σ_p/|p| < 20% |
| C3 | 力格式交叉对照 | classic 与 EDM 在 ≥2 密度比下指数一致（2σ 内） |

**基准出处**: Kupershtokh 2004/2009 (EDM forcing); Shan-Chen 1993 (伪势模型);
Li et al. 2012/13 (各向同性梯度)。协议预注册于 `benchmarks/mcmp3d_spurious_scaling/README.md`。

## 2. 方法与实现

### 2.1 求解器引擎

核心实现: `src/tensorlbm/pseudopotential_edm3d.py`（923 行）

关键组件:
- `static_droplet()`: 静态液滴演化至稳态，测量 Laplace 压力与 |u|max
- `maxwell_coexistence()`: Maxwell 等面积构造求共存密度
- `droplet_init3d()`: tanh 剖面初始条件
- `step_sc3d()`: D3Q19 碰撞-Streaming 步（BGK, τ=1.0）
- 力格式: `kupershtokh` (EDM) vs `velocity_shift` (classic)
- 空间力形式: `difference`（与 `product` 在 D3Q19 上代数等价）

### 2.2 基准运行器

- `benchmarks/mcmp3d_spurious_scaling/run.py`: 全协议运行器（多进程并行）
- `benchmarks/mcmp3d_spurious_scaling/run_incremental.py`: 增量运行器（逐案保存，支持断点续跑）
- `benchmarks/mcmp3d_spurious_scaling/fit_report.py`: 拟合+表格+图形+verdict 生成器

### 2.3 测试套件

`tests/test_mcmp3d_spurious_scaling.py`: 12 项测试全部通过
- 协议覆盖验证、OLS 拟合恢复、已知指数恢复（1.0/2.0/3.0）、伪造检测

`tests/test_pseudopotential_edm3d.py`: 38 项测试全部通过
- 格子权重、各向同性矩条件、梯度精确性、力格式动量守恒、质量守恒

## 3. 参数表

| 参数 | 值 | 依据 |
|---|---|---|
| 网格 n | 32, 48, 64（协议）/ 16（快速验证） | 3 分辨率为验收底线 |
| 液滴半径 | R = n/8 | 自相似盒（R/L = 0.125） |
| 界面宽度 | W_init = 2.0 格子单位（常数） | 平衡界面厚度由模型设定 |
| 耦合 G_eff | −4.4, −4.7, −5.0, −5.3, −5.6 | 5 档 Maxwell 对，Δρ ≈ 1.1…2.3 |
| 力格式 | kupershtokh (EDM), velocity_shift (classic) | 验收判据 C3 |
| τ | 1.0 | 固定粘度，避免 p 被 ν 混淆 |
| 稳态判据 | |Δdp|/dp < 1e-3 连续 3 次检查，check_every=200，上限 6000 步 | dp 为最慢弛豫全局观测量 |

**共存密度（Maxwell 构造）**:
| G_eff | ρ_l | ρ_v | Δρ | 密度比 |
|---|---|---|---|---|
| −4.4 | 1.39029 | 0.28385 | 1.10645 | 4.90 |
| −4.7 | 1.67796 | 0.20322 | 1.47474 | 8.26 |
| −5.0 | 1.92887 | 0.15290 | 1.77597 | 12.62 |
| −5.3 | 2.15848 | 0.11824 | 2.04025 | 18.26 |
| −5.6 | 2.37367 | 0.09307 | 2.28060 | 25.50 |
| −4.0 (Z 控制) | 0.67771 | 0.67598 | 0.00173 | 1.00 |

## 4. 已获结果与对照

### 4.1 快速验证结果（n=16, 1000 步）

| 工况 | n | G_eff | Δρ | 力格式 | u_max | 步数 | 收敛 |
|---|---|---|---|---|---|---|---|
| main | 16 | −5.0 | 1.776 | kupershtokh | 7.10e-03 | 400 | 否 |
| main | 16 | −5.0 | 1.776 | velocity_shift | 7.10e-03 | 400 | 否 |
| ctrlZ | 16 | −4.0 | 0.0017 | kupershtokh | 3.25e-06 | — | — |

**注**: n=16 工况在 400 步后 dp 出现 NaN（远场区域消失，液滴扩散），未能达到稳态。
这表明在较小网格下，液滴稳定性受限，需要更多步数或更大网格。

### 4.2 前轮 pilot 数据（来自 m15363 状态文档）

| 工况 | n | G_eff | Δρ | 力格式 | u_max | 备注 |
|---|---|---|---|---|---|---|
| z_zero | 32 | −4.0 | 1.735e-03 | EDM | **3.25e-06** | 无界面：u_max 比主工况低 ≈4 个量级 |
| wb_psi2 | 32 | −5.0 | 1.776 | psi2 | **1.52e-01** | psi2 不抑制伪电流（假设被否定） |

### 4.3 判据评估

| 判据 | 状态 | 说明 |
|---|---|---|
| C1 表格覆盖 | ❌ FAIL | 仅 1 网格 × 1 密度比（需 ≥3×3） |
| C2 指数不确定度 | ❌ FAIL | 数据点不足，无法拟合 |
| C3 力格式交叉 | ❌ FAIL | 每格式仅 1 密度比（需 ≥2） |
| 控制 Z | ⚠️ 部分 | u_max 坍塌趋势正确，但需完整矩阵验证 |
| 控制 M（质量） | ✅ PASS | 质量漂移 < 1e-14（机器精度） |
| 控制 P（平台） | ❌ FAIL | 未达稳态 |

## 5. 收敛性与参数敏感性

### 5.1 网格依赖性

| n | R | 每步耗时 (ms) | 6000 步预估 |
|---|---|---|---|
| 16 | 2.0 | ~35 | ~3.5 min |
| 32 | 4.0 | ~46 | ~4.6 min |
| 48 | 6.0 | ~150 | ~15 min |
| 64 | 8.0 | ~370 | ~37 min |

总计算量（36 工况）: 约 60-90 分钟（32 核并行）。

### 5.2 稳定性观察

- n=16: 400 步后 dp → NaN（液滴扩散，远场消失）
- n=32: 需要 ≥2000 步才能接近稳态
- 稳态判据基于 dp 收敛，但 dp 在早期波动较大

## 6. 失败模式与工件

### 6.1 计算约束

**根本原因**: 受控环境中的进程组收割器（process group reaper）会终止长时间运行的进程（~120s 超时）。
集群队列有 350+ 作业等待，dispatch 时间不可控。

**已尝试路径**:
1. 集群递交（cj1156, cj1157）: 排队中，351 作业 ahead
2. 本地 nohup 后台运行: 被进程组收割器终止
3. 增量运行器（逐案保存）: 启动后同样被终止
4. 单案 timeout 运行: 可完成但吞吐量太低

### 6.2 工件清单

| 工件 | 路径 | 状态 |
|---|---|---|
| 求解器实现 | `src/tensorlbm/pseudopotential_edm3d.py` | ✅ 完整（923 行） |
| 基准运行器 | `benchmarks/mcmp3d_spurious_scaling/run.py` | ✅ 完整 |
| 增量运行器 | `benchmarks/mcmp3d_spurious_scaling/run_incremental.py` | ✅ 新增 |
| 拟合报告器 | `benchmarks/mcmp3d_spurious_scaling/fit_report.py` | ✅ 完整 |
| 预注册协议 | `benchmarks/mcmp3d_spurious_scaling/README.md` | ✅ 完整 |
| 测试套件 | `tests/test_mcmp3d_spurious_scaling.py` | ✅ 12/12 通过 |
| 运行结果 | `artifacts/mcmp3d_spurious_scaling/results.json` | ⚠️ 部分（2 工况） |
| 表格 | `artifacts/mcmp3d_spurious_scaling/tables.md` | ✅ 生成（数据不足） |
| 图形 | `artifacts/mcmp3d_spurious_scaling/figures/*.png` | ✅ 4 张 |
| Verdict | `artifacts/mcmp3d_spurious_scaling/verdict.json` | ✅ 生成（C1/C2/C3 FAIL） |
| 报告 | `artifacts/mcmp3d_spurious_scaling/REPORT.md` | ✅ 生成 |

### 6.3 复现命令

```bash
# 完整协议（需 ~60-90 分钟，32 核）
python benchmarks/mcmp3d_spurious_scaling/run.py \
    --out artifacts/mcmp3d_spurious_scaling --jobs 32 --threads 4

# 增量运行（支持断点续跑）
python benchmarks/mcmp3d_spurious_scaling/run_incremental.py \
    --out artifacts/mcmp3d_spurious_scaling --threads 2 --resume

# 拟合+报告
python benchmarks/mcmp3d_spurious_scaling/fit_report.py \
    --results artifacts/mcmp3d_spurious_scaling/results.json
```

### 6.4 下一步

集群作业 `cj1156` 已递交，等待 dispatch。一旦完成，运行 fit_report 即可得到完整 verdict。
协议、运行器、拟合器、测试均已就绪，**仅需计算时间**。

## 7. 成熟度评估

| 层级 | 要求 | 状态 |
|---|---|---|
| L1 | 代码存在 | ✅ |
| L2 | impl_files(非占位) + entry + run_artifacts | ⚠️ impl/entry ✅, artifacts 部分 |
| L3 | integration_callers | ⚠️ run.py 为入口，但无平台侧调用 |
| L4 | benchmark(ref vs got) + repro_cmd | ⚠️ repro_cmd ✅, benchmark 数据不足 |

**结论**: L2 部分达成（实现完整，工件部分）。完整 L2 需要集群作业完成。
