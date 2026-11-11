# P106W3A2-N01 · TensorLBM 异构加速后端等价性阶梯（mature 单）

- 仓库 / 模块: `TensorLBM` / `src/tensorlbm/hetero_equivalence.py`
- 产物 JSON（判据数字唯一来源）: `artifacts/p106w3a2_n01/l0.json`, `artifacts/p106w3a2_n01/l1`, `artifacts/p106w3a2_n01/inventory.json`
- 生成时间: 2026-11-11T00:59:35Z / schema `tensorlbm-hetero-equivalence-v1`

## 1. 案例介绍（物理背景 / 几何 / 参数）

「换后端不改物理」是可加速计算的前置条件：同一套 lattice-Boltzmann 离散必须在
不同执行层（NumPy / PyTorch-CPU / SDAA 加速器 / 精度档 fp32·fp64）上给出同一物理。
本节点把该要求做成三级等价阶梯 + 成本-精度账本 + fail-closed 语义。

**L0 台阶（逐位一致 / ULP 界）**：三维周期 D3Q19，Taylor-Green 涡初值
（`u0=0.05`, `nu=0.003`, `tau=0.509`），网格 `[64, 64, 64]`，`200` 步；
二阶平衡态 + BGK + 周期迁移，无外力，故质量/动量必须守恒（L2 台阶）。

**L1 台阶（长时统计等价）**：壁面解析平面槽道，D3Q19 + TRT（magic=3/16）+ Guo 力，
Bouzidi 插值反弹壁，Smagorinsky SGS；驱动压强梯度以体力表示，名义摩擦雷诺数
Re_τ = 180.0。参数表：

| 参数 | 值 | 说明 |
|---|---|---|
| nx × nz × ny | 32 × 32 × 64 | 槽高 H=62，半高 h=31.0 |
| Re_τ / u_τ | 180.0 / 0.01 | u_τ = sqrt(a·h) |
| ν / τ+ / τ- | 1.722e-03 / 0.50517 / 36.8 | 分子黏性；TRT 反对称松弛 |
| Δy+ | 5.81 | 首层壁面单位分辨率（粗） |
| Lx/h × Lz/h | 1.03 × 1.03 | 计算域长宽比 |
| 总步数 / 预热 | 10000 / 3000 | 统计窗口 = 预热后每 sample_interval 取样 |
| 样点数 | 700 | 平均 2.26 个涡翻转时间 |
| SGS Cs | 0.1 | 与仓库 Re_τ=180 通道战役同参 |
| 精度档 | float64 / float32 | golden = CPU fp64；候选 = SDAA fp32 |

## 2. 方法（golden 冻结 + 三级阶梯 + fail-closed）

**(a) golden 冻结**：CPU fp64 参考串行实现（`cpu_fp64_serial`，严格按方向顺序归约）
每步导出宏观量与守恒量（质量、三分量动量、动能、ρ_min/ρ_max、u_max），
每步再对 (ρ,u) 全场求 SHA-256，逐步摘要串成 `digest_chain`；
阶梯结束后 `freeze_golden()` 写出 `trace npz + manifest(sha256)`，
审计链 = 逐步摘要链 + npz 文件哈希 + 场哈希。

**(b) L0 判据**：候选与 golden 的逐步摘要链逐位相同（`bitwise_identical`），
**或**全场最大相对偏差落在 a-priori 舍入预算内：
`budget = A_N · N_ops · ε_dtype`，其中 `A_N` 由 golden 轨迹上注入 1-ULP 扰动的孪生实验实测放大率
（`N_ops=400` 由源码常量给出）。预算的作用是把「舍入差异」与「算法/方案差异」分开：
方案级差异会高出预算若干数量级（下文负对照给证）。

**(c) L1 判据**（配对比较：同初值、同种子、同网格）：
`U+ / u_rms+ / v_rms+ / w_rms+` 剖面对 golden 的归一化 L2 偏差 ≤ 2 %，
并且对 MKM-1999 DNS 的偏差不得比 golden 更差（容差 2 %）。

**(d) L2 判据**：无外力周期盒中质量与动量每步每格点漂移 ≤ 1e-12。

**(e) fail-closed**：`resolve_profile()` 对不可用后端/设备抛 `BackendUnavailable` 并登记拒绝；
`require_precision()` 对精度不足抛 `PrecisionInsufficient`；`require_certified()` 对超等级使用抛
`CertificationError`。任何路径都不回退 CPU、不静默降精度（`tests/test_hetero_equivalence.py` 逐条覆盖）。

**(f) 工程实现**：单条命令有墙钟上限，故 L1 用 `ChunkedChannelRun` 分块积分并 checkpoint
（群体场 + 累加统计），`scripts/l1_advance.sh` 在预算内循环推进；分块与一次性积分结果等价性有单测
（`test_chunked_channel_matches_monolithic`，剖面差 < 1e-12）。

## 3. 计算结果

### 3.1 L0 台阶（64³, 200 步）
| 后端档 | 设备 | 精度 | 逐位一致 | 最大相对偏差 | 舍入预算 | ULP 距离 | Mcell·step/s | 判据 |
|---|---|---|---|---|---|---|---|---|
| `cpu_fp64_serial` | cpu | float64 | 是 | 0 | 3.908e-11 | 0 | 0.649 | **PASS** |
| `cpu_fp64_numpy_tree` | cpu | float64 | 是 | 0 | 3.908e-11 | 0 | 0.486 | **PASS** |
| `cpu_fp64_torch_gemm` | cpu | float64 | 是 | 0 | 3.908e-11 | 0 | 0.572 | **PASS** |
| `cpu_fp32_torch` | cpu | float32 | 否 | 5.084e-05 | 1.812e-02 | 4.56e+16 | 0.676 | **PASS** |
| `sdaa_fp64_torch` | sdaa:0 | float64 | 是 | 0 | 3.908e-11 | 0 | 1.811 | **PASS** |
| `sdaa_fp32_torch` | sdaa:0 | float32 | 否 | 3.621e-05 | 9.537e-03 | 3.45e+16 | 2.290 | **PASS** |
| `cpu_fp64_shipped_w` | cpu | float64 | 否 | 5.749e-06 | 3.908e-11 | 2.6e+15 | 0.669 | **FAIL** |

golden 摘要链: `d4154eb5a86781b0e9c8b1ff91cb3bdb354afc6b53b04ed48b8c390424a4d27a`；
golden 场哈希: `4fe26428c9b720b53953468dd57c0c2415d371552616514b204490d997c1f8df`；
golden 轨迹 npz: `artifacts/p106w3a2_n01/l0/golden_cpu_fp64_serial.npz`

### 3.2 L2 台阶（守恒不变量）
| 后端档 | 质量漂移/步/格点 | 动量漂移/步/格点 | 判据 ≤1e-12 |
|---|---|---|---|
| `cpu_fp64_serial` | 1.088e-16 | 3.795e-21 | **PASS** |
| `cpu_fp64_numpy_tree` | 1.088e-16 | 3.795e-21 | **PASS** |
| `cpu_fp64_torch_gemm` | 1.088e-16 | 3.795e-21 | **PASS** |
| `cpu_fp32_torch` | 2.910e-08 | 6.884e-12 | **FAIL** |
| `sdaa_fp64_torch` | 1.088e-16 | 3.795e-21 | **PASS** |
| `sdaa_fp32_torch` | 2.911e-08 | 4.623e-12 | **FAIL** |
| `cpu_fp64_shipped_w` | 2.912e-08 | 4.491e-21 | **FAIL** |

fp64 四档（含 SDAA 加速器）逐位一致且守恒；fp32 两档的守恒漂移 ~2.9e-8，**比判据高 4 个数量级**，
因此 fp32 档在台账里被登记为「L0 certified / L2 failed」——不允许在不变量敏感的算例上被误用。

### 3.3 L1 台阶（Re_τ=180 槽道，10⁴ 步）
| 候选档 | 设备 | 精度 | 最差剖面偏差 | U+ | u_rms+ | v_rms+ | w_rms+ | 判据 ≤2% |
|---|---|---|---|---|---|---|---|---|
| `sdaa_fp32_torch` | sdaa:0 | float32 | 0.214 % | 0.027 % | 0.214 % | 0.003 % | 0.005 % | **PASS** |

MKM-1999 锚定列：golden 对 MKM 的归一化偏差 31.098 %，
候选 31.111 %（`not_worse=True`）。

**物理代表性（诚实项）**：golden 槽道的雷诺切应力峰值
`-uv+ = 3.372e-04`，MKM-1999 为 `0.723`，
比值 `4.66e-04` → **turbulent_state = False**。
即在 Δy+=5.81、Lx/h=1.03、平均 2.3 个涡翻转的
可负担配置下，流场**层流化**了：后端等价性比较仍然有效（配对、同状态），但它证明的是「层流/衰减态的等价」，
**不构成湍流统计等价的认证**。此条按 honest-FAIL 记录，不予粉饰。

### 3.4 负对照（证明判据不是空判据）
| 负对照 | 相对 golden 偏差 | 判据 |
|---|---|---|
| L1 · `sdaa_fp32_torch:interpolated`（SGS Cs=0.0，其余同 golden） | 1277.189 % | **FAIL（符合预期）** |
| L0 · `cpu_fp64_shipped_w`（仓库 float32 权表，其余同 golden） | 5.749e-06 | **FAIL（符合预期）** |

L0 负对照的偏差 `5.749e-06` 是舍入预算 `3.908e-11` 的
约 147108 倍 —— 判据确实把「方案差异」从「舍入差异」里分了出来；
L1 负对照（关掉 SGS 闭合，物理方案变更）同样被 2 % 判据拒绝。

### 3.5 吞吐-精度账本（成本-精度 Pareto）
| 台阶 | 后端档 | 精度 | 误差(相对 golden) | 单位成本 | Mcell·step/s | 加速比 |
|---|---|---|---|---|---|---|
| L0 | `cpu_fp64_serial` | float64 | 0 | 0.4041 s/step | 0.649 | — |
| L0 | `cpu_fp64_numpy_tree` | float64 | 0 | 0.5393 s/step | 0.486 | — |
| L0 | `cpu_fp64_torch_gemm` | float64 | 0 | 0.4583 s/step | 0.572 | — |
| L0 | `cpu_fp32_torch` | float32 | 5.084e-05 | 0.3876 s/step | 0.676 | — |
| L0 | `sdaa_fp64_torch` | float64 | 0 | 0.1448 s/step | 1.811 | — |
| L0 | `sdaa_fp32_torch` | float32 | 3.621e-05 | 0.1145 s/step | 2.290 | — |
| L0 | `cpu_fp64_shipped_w` | float64 | 5.749e-06 | 0.3921 s/step | 0.669 | — |
| L1 | `cpu_fp64_torch` | float64 | 0 | 379.7 s/h/u_τ | 0.535 | 1.00× |
| L1 | `sdaa_fp32_torch` | float32 | 2.140e-03 | 72.59 s/h/u_τ | 2.799 | 5.23× |

- L0 Pareto 前沿: ['sdaa_fp32_torch', 'sdaa_fp64_torch']
- L1 Pareto 前沿: ['sdaa_fp32_torch', 'cpu_fp64_torch']

**读法**：L0 上 SDAA fp64 与 CPU fp64 逐位一致但快 2.8×（0.145 vs 0.404 s/step），
说明「加速不牺牲可复现性」在 fp64 档可以做到；fp32 档再快 ~25 %（SDAA），代价是守恒不变量失守与
约 3.6e-5 的场相对偏差。L1 上 SDAA fp32 把单位对流时间成本从 379.7 s 压到 72.6 s（5.2×），
剖面偏差 0.21 %，是「误差-成本」上可接受的一点，但其 L2 认证为 failed。

### 3.6 精度档认证矩阵与 fail-closed 拒绝清单

| 后端档 | L0 | L1 | L2 |
|---|---|---|---|
| `cpu_fp64_serial` | certified | not_evaluated | certified |
| `cpu_fp64_numpy_tree` | certified | not_evaluated | certified |
| `cpu_fp64_torch_gemm` | certified | not_evaluated | certified |
| `cpu_fp32_torch` | certified | not_evaluated | failed |
| `sdaa_fp64_torch` | certified | not_evaluated | certified |
| `sdaa_fp32_torch` | certified | certified | failed |
| `cpu_fp64_shipped_w` | failed | not_evaluated | failed |
| `cpu_fp64_torch` | not_evaluated | certified | not_evaluated |

拒绝登记（`artifacts/p106w3a2_n01/inventory.json`）：共 5 条，
被拒后端: `cuda_fp32_torch`, `cuda_fp64_torch`, `mindspore_fp32`, `paddle_fp32`, `triton_fused_fp32`。
拒绝原因逐条落盘（模块缺失 / `torch.cuda.is_available()` 为 False / 设备不支持 fp64），
调用方拿到的是异常而不是一个悄悄降级的 CPU 结果。

## 4. 结果图

- `figures/p106w3a2_n01/fig1_l0_ladder.png`：(a) 各档相对 golden 的逐步偏差与 fp64/fp32 舍入预算线；
  (b) 各档 L2 守恒漂移与 1e-12 判据线。
- `figures/p106w3a2_n01/fig2_l1_profiles.png`：(a) U+ 剖面 —— golden / SDAA-fp32 / MKM-1999 DNS / SGS 关闭负对照；
  (b) 三个方向 rms 剖面。
- `figures/p106w3a2_n01/fig3_pareto.png`：L0 与 L1 的误差-成本散点与 Pareto 前沿。

图与表同源（脚本 `scripts/make_p106w3a2_n01_figures.py` 直接读 `artifacts/p106w3a2_n01/*.json`）。

## 5. 注意点（踩坑 / 局限 / 适用范围）

1. **生产求解器无法跑加速器（已修）**：`turbulent_channel3d.py` 里 `torch.stack([torch.ones(...), c[:,0], ...])`
   在 CPU 张量与设备张量间混用，SDAA 上直接 `different device is not supported`；`_spectral_perturbation`
   返回 CPU 张量、`_fold` 后处理也在设备侧混用。已做最小修复（host 侧构建后搬运 / 后处理统一回 CPU float64），
   修复后 CPU 侧 21 条既有通道测试全绿，且 CPU/SDAA 结果一致。
2. **权重表是 float32（缺陷，未改）**：`tensorlbm/d3q19.py` 的 `W` 是 float32，19 个权重之和 = 1 + 1.49e-8，
   等价于每步注入 1.5e-8 相对质量源。对 1e-2 量级的速度场无影响，但让任何 1e-12 守恒判据不可达。
   阶梯内核改用精确有理数权重（1/3, 1/18, 1/36），**生产表保持不动**（改它会平移所有已验证结果），
   并把该表当作负对照档固化下来（`cpu_fp64_shipped_w`）。
3. **halfway 负对照是空操作**：`wall_origin=0.5` 时 Bouzidi 插值参数 `q=1-0.5=0.5`，与 halfway 完全等价，
   所以「换成 halfway 壁」跑出来的结果与 golden 一致（偏差 0.21 %，与正常候选同值）。该负对照作废，
   改用「关闭 SGS 闭合（Cs=0.1→0）」这一真实的物理方案变更加以替代。此坑已在代码注释与本节点记录，避免复用。
4. **L1 物理代表性不足**（见 §3.3）：细网格+长时间才是真正的湍流统计等价认证；本节点在墙钟/单命令预算内
   只能给出层流化状态下的配对等价证据。要拿到湍流态需 ~4h×2h 域、Δy+≤2、≥10 涡翻转平均，
   比现配置贵 ~40×（约 14 CPU·小时）。
5. **fp32 档的 L0 判别力弱于 fp64 档**：fp32 的 a-priori 预算为 `1.812e-02`（放大率 × N_ops × ε_fp32），
   与 L1 的 2 % 判据同量级，因此「落在预算内」对 fp32 只说明没有超预算；实测偏差 `5.084e-05` 仅占预算
   `0.28 %`，一致性本身良好，但认证强度不如 fp64 档的逐位一致。
6. **L0 预算不是形式化证明**：`A_N` 是实测放大率，`N_ops` 是源码常量 400；这是工程界，
   用「方案差异远高于预算」来做判别（负对照实证），不是混沌非线性迭代的前向误差严格证明。
7. **单命令墙钟上限**：L1 用可断点续跑的分块驱动（每块 checkpoint），否则 10⁴ 步无法在一次命令内完成；
   分块一致性有单测锁定。
8. **`health()` 每步调用代价**：初版分块驱动每步做一次全场健康检查，约占 40 % 集成时间，已改为每 250 步一次。

## 6. 误差判定（逐项对基准，显式回答「是否 < 3%」）

| 关键量 | 偏差 | 是否 < 3% |
|---|---|---|
| L0 `cpu_fp64_serial` 场相对偏差 | 0 | <3% |
| L0 `cpu_fp64_numpy_tree` 场相对偏差 | 0 | <3% |
| L0 `cpu_fp64_torch_gemm` 场相对偏差 | 0 | <3% |
| L0 `cpu_fp32_torch` 场相对偏差 | 5.084e-05 | <3% |
| L0 `sdaa_fp64_torch` 场相对偏差 | 0 | <3% |
| L0 `sdaa_fp32_torch` 场相对偏差 | 3.621e-05 | <3% |
| L0 `cpu_fp64_shipped_w` 场相对偏差 | 5.749e-06 | <3% |
| L2 golden 质量漂移/步/格点 | 1.088e-16 | <3% |
| L1 `sdaa_fp32_torch` vs golden（最差剖面） | 0.214 % | <3% |
| L1 golden vs MKM-1999 DNS（U+ 归一化 L2） | 31.098 % | **否** |
| L1 负对照 vs golden（SGS 关闭） | 1277.189 % | **否**（预期：判据必须拒绝） |
| L0 负对照 `cpu_fp64_shipped_w`（float32 权表） | 5.749e-06（数值 <3%，但超舍入预算 3.908e-11 → 判据 FAIL） | **否**（预期：判据必须拒绝） |

**结论**：
- 后端等价性（本节点的物理目标）：fp64 四档逐位一致（偏差 = 0），fp32 两档 3.6e-5 / 5.1e-5，
  L1 加速器档对 golden 2.1e-3（0.21 %）—— **全部 < 3%**。
- 对实验/DNS 基准的绝对精度：golden 槽道对 MKM-1999 的偏差 **31 %，不满足 < 3%**。原因不是后端也不是
  等价性，而是可负担网格下流场层流化（§3.3）——按 honest-FAIL 记录，本节点不宣称湍流精度达标。

## 7. 复现命令

```bash
export PYTHONPATH=src
python scripts/run_p106w3a2_n01_equivalence.py --stage inventory
for t in cpu_fp64_serial cpu_fp64_numpy_tree cpu_fp64_torch_gemm cpu_fp32_torch \
         sdaa_fp64_torch sdaa_fp32_torch cpu_fp64_shipped_w; do
  python scripts/run_p106w3a2_n01_equivalence.py --stage l0 --tier $t
done
python scripts/run_p106w3a2_n01_equivalence.py --stage l0-amp --tier cpu_fp64_serial
python scripts/run_p106w3a2_n01_equivalence.py --stage l0-check
bash scripts/l1_advance.sh cpu_fp64_torch 200 95 --steps 10000 --warmup 3000 \
     --sample-interval 10 --nx 32 --nz 32 --ny 64 --smagorinsky-cs 0.1
python scripts/run_p106w3a2_n01_equivalence.py --stage l1-finalize --profile cpu_fp64_torch \
     --steps 10000 --warmup 3000 --sample-interval 10 --nx 32 --nz 32 --ny 64 --smagorinsky-cs 0.1 --tag full
python scripts/run_p106w3a2_n01_equivalence.py --stage aggregate
python scripts/make_p106w3a2_n01_figures.py
python scripts/render_p106w3a2_n01_report.py
pytest -q tests/test_hetero_equivalence.py
```

## 8. 审计链

- golden 逐步摘要链: `d4154eb5a86781b0e9c8b1ff91cb3bdb354afc6b53b04ed48b8c390424a4d27a`
- golden 轨迹 npz sha256: `b81d1f13e6e720cd45400b02f1a665ead2a40156591c7695c6f95ef20dea3869`
- 权重表缺陷量化: `{"shipped_sum": 1.0000000149011612, "exact_sum": 1.0, "excess_sum": 1.4901161193847656e-08, "max_abs_weight_error": 9.934107481068821e-09, "relative_mass_source_per_step": 1.4901161193847656e-08, "shipped_dtype": "float32", "note": "tensorlbm/d3q19.py W is a float32 weight table whose sum is 1 + 1.49e-8, i.e. ~1.5e-8 relative mass source per step; that blocks any 1e-12 conservation gate.  The equivalence kernel uses exact rationals; the production table is left untouched (changing it would shift every validated result) but flagged in the report."}`
- 认证矩阵: `{"cpu_fp64_serial": {"L0": "certified", "L2": "certified"}, "cpu_fp64_numpy_tree": {"L0": "certified", "L2": "certified"}, "cpu_fp64_torch_gemm": {"L0": "certified", "L2": "certified"}, "cpu_fp32_torch": {"L0": "certified", "L2": "failed"}, "sdaa_fp64_torch": {"L0": "certified", "L2": "certified"}, "sdaa_fp32_torch": {"L0": "certified", "L2": "failed", "L1": "certified"}, "cpu_fp64_shipped_w": {"L0": "failed", "L2": "failed"}, "cpu_fp64_torch": {"L1": "certified"}}`
- 可再生物不入库：`artifacts/p106w3a2_n01/fields/*.npz`（各档终态场，~41 MB）与
  `artifacts/p106w3a2_n01/l1/state_*/state.pt`（续跑 checkpoint）是中间产物，
  其 sha256 已固化在 `l0/*_manifest.json` 与该 JSON 的 `fields_npz_sha256` 字段里；
  按 §7 命令重跑可逐位复现（逐步摘要链相同）。

