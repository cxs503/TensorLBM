# XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO

**锚点**：XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO
**任务**：m18366（返修轮1，原 m18225 仅 L0）
**日期**：2026-11-13

## 1. 问题设定

单组分 Shan–Chen 伪势（SCMP）+ 状态方程型伪势（Peng-Robinson / Carnahan-Starling）的
Maxwell 等面积构造在高密度比（10:1 / 100:1 / 1000:1）下的数值可达性验证。

**基准出处**：
- Maxwell 等面积构造：`src/tensorlbm/sc_eos_coexistence.py`（`maxwell_coexistence`）
- EOS 伪势工厂：`src/tensorlbm/multiphase.py`（`make_psi_eos`）
- Yuan & Schaefer (2006) Phys. Fluids 18 042101 — PR-EOS 参数

## 2. 方法与实现

### 2.1 Maxwell 等面积构造

在比容空间（v = 1/ρ）求解：
```
∫_{v_l}^{v_v} [p(v) − p_sat] dv = 0
```

实现于 `sc_eos_coexistence.py`：
- 对数密度网格 + 向量化二分 + 梯形积分
- 无 SciPy 依赖，纯 torch/numpy
- 返回 `MaxwellResult(found, p_sat, rho_vapour, rho_liquid, density_ratio, area_residual)`

### 2.2 稳定性探针

`benchmarks/eos_maxwell/probe_lib.py`：
- 对目标密度比二分 T_r，调用 Maxwell 构造反解 EOS 点
- 静态液滴初始化（tanh 界面），零初速
- EDM / Guo forcing + standard / pressure_tensor 离散
- 测量：密度比维持误差、伪流 Ma、质量漂移、塌缩步

### 2.3 入口

`benchmarks/eos_maxwell/run.py`：
- `--mode pilot`：快速冒烟（1 档，短步数，小盒子）
- `--mode tiers`：三档目标密度比 × forcing
- `--mode controls`：平面界面对照
- `--mode all`：全量

## 3. 参数表

| 参数 | 值 |
|---|---|
| EOS | Peng-Robinson (Yuan & Schaefer 2006) |
| G | 1.0（库约定，吸引） |
| 目标密度比 | 10, 100, 1000 |
| 盒子尺寸 | L=96（tiers），L=64（pilot） |
| 液滴半径 | R=20 |
| 步数 | 20000（tiers），40000（longhold） |
| 浮点精度 | float32 |
| 边界 | 周期 |

## 4. 对照表

### 4.1 Maxwell 构造（解析）

| 目标密度比 | T_r | ρ_l | ρ_v | 面积残差（相对） | 命中目标 |
|---|---|---|---|---|---|
| 10 | 0.8266 | 5.8871 | 0.5880 | ~1e-13 | ✓ |
| 100 | 0.5684 | 7.8854 | 0.0789 | ~1e-12 | ✓ |
| 1000 | 0.4109 | 8.7654 | 0.0088 | ~1e-11 | ✓ |

### 4.2 试点运行（pilot，ratio=10，2000 步）

| 量 | 值 | 判据 | 结果 |
|---|---|---|---|
| 稳定 | True | ≥2000 步 | ✓ |
| 密度比误差 | ~2% | ≤15% | ✓ |
| 伪流 Ma | 0.114 | ≤0.05 PASS / ≤0.10 MARGINAL | ✗ FAIL |
| 质量漂移率 | 8.87e-9/步 | ≤5e-8/步 | ✓ |

### 4.3 全量战役（已递交集群 cj1906）

全量结果见 `benchmarks/eos_maxwell/RESULTS.md`（历史运行）：
- ratio=10：稳定保持，但伪流 Ma=0.115 > 0.10 → FAIL
- ratio=100：EDM 稳定但 ratio_error=1.93 > 0.15 → FAIL；Guo 53 步塌缩
- ratio=1000：全部塌缩（EDM 100 步，Guo 30 步）

## 5. 收敛性/参数敏感性

- **界面宽度臂**（w=4/6/8）：ratio=10 对宽度不敏感（ratio_error ~4.4-4.6%）；ratio=100 在 w=8 时 500 步塌缩
- **τ 杠杆臂**：τ 从 0.6 到 2.0 扫描，ratio=10 的伪流 Ma 最低 0.115（τ=0.6），仍高于 PASS 门
- **线程数臂**（1/2/4/8）：数值可重复，ratio 和 Ma 完全一致
- **扰动臂**（噪声 1e-7）：结果与无扰动一致，表明"稳定 20000 步"是构造性质而非单轨迹偶然

## 6. 失败模式与工件

### 6.1 失败模式

| 密度比 | 机制 | 起始步 |
|---|---|---|
| 10 | 伪流幅值过高（Ma > 0.10），非塌缩 | — |
| 100 (Guo) | 负密度 | 53 |
| 1000 (EDM) | 伪流爆炸（Ma > 0.5） | 100 |
| 1000 (Guo) | 负密度 | 30 |

### 6.2 工件路径

- 实现：`src/tensorlbm/sc_eos_coexistence.py`（502 行）
- 探针：`benchmarks/eos_maxwell/probe_lib.py`（754 行）
- 入口：`benchmarks/eos_maxwell/run.py`
- 测试：`tests/test_sc_eos_coexistence.py`（13 passed），`tests/test_eos_maxwell_stability.py`（17 passed）
- 判据：`benchmarks/eos_maxwell/CRITERION.md`
- 结果：`benchmarks/eos_maxwell/RESULTS.md`，`benchmarks/eos_maxwell/out/results.json`
- 复现命令：`python benchmarks/eos_maxwell/run.py --mode pilot --out benchmarks/eos_maxwell/out --device cpu`

### 6.3 成熟度

| 级别 | 要求 | 状态 |
|---|---|---|
| L2 | impl_files + entry + run_artifacts | ✓ 达成 |
| L3 | integration_callers | ✓ 达成（cavitation.py, microbubble_dr/eos.py, sc_phasechange/highdensity.py） |
| L4 | benchmark (ref vs got) + repro_cmd | ✓ 达成（Maxwell 构造 ref vs 实测对照，repro_cmd 见上） |

## 7. 注意点

1. **Maxwell 构造 ≠ 离散不动点**：连续 EOS 的等面积解在离散 SC 格式下不是精确不动点，
   密度比 10 档的实测比与 Maxwell 锁值偏差 ~2-4%，密度比 100 档偏差 ~193%。
2. **伪流地板**：即使 ratio=10 档能稳定保持 20000+ 步，伪流 Ma 仍在 0.11-0.12 量级，
   高于预注册 PASS 门（0.05）。这是 SCMP 格式的结构性限制，非调参可解。
3. **高密度比不可达**：ratio=1000 档在百步内塌缩，表明 PR-EOS SCMP 在 1000:1 密度比下
   不可用（至少在当前 forcing/离散组合下）。
4. **平面界面对照**：ratio=10 平面臂实测比 11.0（vs Maxwell 10.0），ratio=100 平面臂
   实测比 456（vs Maxwell 100），表明离散格式本身的共存对偏移是主要误差源。
