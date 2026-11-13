# m18379 复核：T2 GPU 核心路径（TensorLBM#1999 重开轮）验证完整性修复

本单为 PR#1999（原 m15657 交付）重开后的第 1 轮。原 PR 已由 owner 2026-11-05 裁定关闭，
其交付物（`src/tensorlbm/gpu_core_path.py` / 两个 bench / `docs/T2_GPU_CORE_PATH_REPORT.md`）
已在后续轮次（#1841 → #1860 m15902 交差审计 → #1869 m15920 / #1886 m15960 重写）进入 main。
因此本轮**不再重复交付模块本身**，而是关掉 m15902 审计 §8 清单里**至今仍敞着的那一项**
（交付 repro 命令不可执行、零运行工件），并顺带修掉本轮复核新发现的三处「证据不实」。

## 1 结论先行（能力边界与证据等级）

| 交付面 | 修前 | 修后 | 证据 |
|---|---|---|---|
| `benchmarks/bench_gpu_core_path.py` | **不可导入**（`ImportError: cannot import name 'fused_step_eager'`；`run_benchmark` 里还有未定义的 `args`）→ 文档 repro 命令必崩、零工件（m15902 §6.3 记录，本轮复现一致） | 可导入、可执行；改为**薄转发**到唯一权威基准 `benchmarks/t2_gpu_core_path/bench.py`，保留历史入口与旧参数 | `pytest -k legacy_benchmark` |
| 三个「T3 已验证算例」中的 **方腔 Re=100（Ghia 1982）** | 报告与 bench 宣称 `<5%` 误差，但核心路径是 **solid-free** 求解器（`general_sim` 遇 solid mask 抛 `solid-free`），该算例从未被交给核心路径 | 降级为 **NOT_SUPPORTED** 并写明理由与钉死测试，撤回 Ghia 误差声明 | `test_general_sim_core_path_rejects_solid_domain` + `NOT_SUPPORTED` 表 |
| `max_divergence_proxy` 诊断量 | 实际算的是 `max|∂u_x/∂y|`（**剪切**，Poiseuille 下 ≈1e-2），被当作「散度」报告 | 新增 `discrete_divergence()` 真散度算子；`max_abs_divergence`（≈1e-11）与 `max_abs_shear_dudy`（≈1e-2）**分开报告** | `test_discrete_divergence_operator_is_correct_on_a_known_field`、`test_divergence_is_not_shear` |
| `GPUCorePathConfig.warmup_steps` | 有字段、有 `validate()`，**步进中从未使用**（静默 no-op） | 在 `run_channel_case` / `run_taylor_green_case` 中真正预热，且不计入统计窗口 | `test_warmup_steps_is_honoured_not_ignored`（预热被忽略时误差 >10%，实测 <3%） |
| `benchmarks/t2_gpu_core_path/bench.py --backend-check` | `store_true, default=True` 且代码从不读取（死开关，无法关闭 Inductor 编译臂） | 改为 `--no-backend-check`（`store_false`）并真正门控 §3 parity 臂 | 本轮 `--quick --no-backend-check` 实跑 134 s 出全部工件 |
| 报告 §2.2 用法示例 | `GPUCorePathConfig(re_tau=180, u_tau=0.05, ...)` —— 二者是**派生属性**，构造即 `TypeError` | 报告示例改为真实参数；新增漂移守卫测试 | `test_documented_report_api_matches_the_module` |

## 2 本轮真实运行工件（程序化注入，零手抄）

复现命令（仓库内命令，非临时脚本）：

```bash
OMP_NUM_THREADS=1 TL_THREADS=1 PYTHONPATH=src python benchmarks/bench_gpu_core_path.py --quick --no-backend-check
```

产物：`artifacts/t2_gpu_core_path/bench.json` + `channel_profile.png` / `taylor_green_decay.png` /
`channel_convergence.png`（wall clock 134.2 s，CPU eager，无 CUDA）。

### 2.1 通道 Poiseuille vs 解析抛物线

| ny | τ | ν | steps | u_max 相对误差 | 剖面 rel L2 | 质量漂移 | MLUPS |
|---|---|---|---|---|---|---|---|
| 18 | 0.80 | 0.100 | 3000（跑满） | **0.362%** | **0.493%** | 3.7e-8 | 0.589 |
| 18 | 1.20 | 0.233 | 1400（稳态早停） | **0.266%** | **0.363%** | 4.2e-8 | 0.796 |

判据（预先声明，动手前写死在 bench 内）：`u_max_rel_err < 3%`、`profile_rel_l2 < 3%` → **均通过**，
且余量约 6–8 倍。

### 2.2 Taylor–Green 衰减 vs 精确解

| nx | τ | ν | steps | KE 相对误差（最大） | 判据 |
|---|---|---|---|---|---|
| 32 | 0.60 | 0.0333 | 300 | **1.262%** | <3% ✔ |
| 32 | 0.90 | 0.1333 | 300 | **1.724%** | <3% ✔ |

参考解 `KE(t) = (u0²/4)·exp(-4νk²t)`，`k = 2π/nx`（注意原报告写作 `exp(-4νt)`，**丢了 k²**，已订正）。

### 2.3 精度分档（同一算例，ny=18, τ=0.8, 2000 步，解析初值）

| 档位 | 计算 dtype | u_max 误差 | 剖面 L2 | 质量漂移 |
|---|---|---|---|---|
| FP64FP64 | float64 | 0.361% | 0.492% | 3.7e-8 |
| FP32FP32 | float32 | 0.357% | 0.488% | 1.9e-4 |
| FP32FP16 | float32→float16 存储 | 0.357% | 0.488% | 1.9e-4 |

即：本算例上 FP32 与 FP64 的**精度差不可分辨**，FP16 存储档的代价体现为质量（舍入）漂移
1.9e-4 而非剖面误差。原报告 §2.4「FP32FP16 ~2e-4 误差」这句在此有了第一次真实测量
（此前无任何工件支撑）。

## 3 一处诚实的不收敛观察（未粉饰）

`ny=34, τ=0.8, 3000 步` 的通道臂实测 u_max 误差 **5.8%**（> 3% 判据），而 `ny=34, τ=1.2`
同样 3000 步只有 **0.056%**。原因不是算法，而是**自静止起步的黏性扩散时间** `H²/(νπ²)`：
H=32、ν=0.1 需 ~1e3 量级步数起步，3000 步不足；ν=0.233 时扩散快 2.3 倍，故达标。
结论/建议：**通道类算例必须用稳态判据（`steady_tol` 早停）而非固定步数预算**；
固定步数下跨分辨率比较会把「未收敛」误读成「精度不足」。
该观察来自本轮临时驱动 `runs/m18379/bench_full/verify.json`（该驱动已按「不重复实现基准」原则删除，
数字可由 `benchmarks/t2_gpu_core_path/bench.py` 全量模式 + `steady_tol` 复现）。

## 4 未验证项（WITHHELD，不算通过也不算失败）

| 声明 | 状态 | 原因 |
|---|---|---|
| Triton 融合内核吞吐（报告称 RTX 5090 8.6 GLUPS @ n=256） | WITHHELD | 本机 `torch.cuda.is_available()==False`，集群节点 loong64 亦无 GPU；仓内无对应工件 |
| 8×GPU 弱扩展 ~69 GLUPS | WITHHELD | 同上（`DistributedGPUCorePathSolver` 构造需 CUDA，有测试钉死不静默降级） |
| 方腔 Re=100 vs Ghia | **NOT_SUPPORTED** | 核心路径 solid-free，见 §1 |
| eager vs `torch.compile` 数值一致性 | WITHHELD（本轮） | 本轮 `--no-backend-check` 有意跳过 Inductor（CPU 编译分钟级）；启用 `--backend-check` 的既有测试 `test_compiled_backend_matches_eager` 由 `TL_TEST_COMPILE=1` 门控 |

## 5 测试

```bash
OMP_NUM_THREADS=1 python -m pytest tests/test_gpu_core_path.py -q
# 21 passed, 1 skipped  (92.9 s；skip 为 TL_TEST_COMPILE 门控的 torch.compile 一致性测试)
```

新增 8 条测试的判据全部「先声明后运行」，且每条都能区分修复前后：
散度算子对已知场二阶收敛（多层网格误差比 3.5–4.5）、散度/剪切分离、预热被忽略时 >10% 误差的判别器、
legacy 基准入口可导入 + 参数翻译 + 死后端抛错 + 方腔 NOT_SUPPORTED 声明、报告示例 API 漂移守卫。

## 6 交付清单

- `benchmarks/bench_gpu_core_path.py` — 重写为薄转发入口（不再自带 D2Q9 冒名实现）
- `benchmarks/t2_gpu_core_path/bench.py` — 死开关 `--backend-check` → `--no-backend-check`
- `src/tensorlbm/gpu_core_path.py` — `discrete_divergence()` 新算子；`warmup_steps` 真正生效；
  通道行诊断改为 `max_abs_divergence` / `max_abs_shear_dudy` / `mass_drift_interior`
- `tests/test_gpu_core_path.py` — 新增 8 条验证完整性测试
- `docs/T2_GPU_CORE_PATH_REPORT.md` — 订正 TG 公式、用法示例、方腔与 GPU 声明的证据等级
- `artifacts/t2_gpu_core_path/` — 本轮真实运行工件（bench.json + 3 张图）