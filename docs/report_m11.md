# Mission 11 补报：TensorLBM SDAA 适配性普查

> **mission_id**: 11  
> **repo**: TensorLBM  
> **scenario_key**: report-backfill  
> **日期**: 2026-07-26  
> **环境**: LoongArch + Tecorigin SDAA, torch 2.7.1 + torch_sdaa, triton 3.4.0  
> **原单结论**: honest-FAIL（高 Re RANS 基准误差 >15%）

---

## 1. 目的

对 TensorLBM 仓库 `src/tensorlbm` 全部 Python 模块进行 SDAA（Shengshan Deep Learning Accelerator Architecture）适配性普查，明确：
1. 哪些模块可直接在 SDAA 设备上运行（纯 PyTorch）；
2. 哪些模块依赖 Triton 内核编译（需 SDAA 后端或解释模式）；
3. 在 SDAA 真机上运行 SUBOFF 阻力基准测试，验证数值精度。

原单交付物为 `docs/sdaa_adaptability.md`（671 行，30KB），本单为补报——补充论文级进展报告。

---

## 2. 方法

### 2.1 模块分类方法

对 `src/tensorlbm` 下 512 个 `.py` 文件进行静态扫描，按以下规则分类：

| 类别 | 判定标准 | 预期行为 |
|------|----------|----------|
| **A（纯 torch）** | 无 `import triton` / `@triton.jit` / `tl.*` | 可直接在 `device='sdaa'` 运行 |
| **B（triton 依赖）** | 有 triton 引用 | 仅可 `TRITON_INTERPRET=1` CPU 解释执行 |
| **C（混合）** | 同时有 triton + torch | 部分可上 sdaa |

### 2.2 验证方法

- **A 类**：等间距抽样 15 个模块，在 SDAA 真机上验证 import + 张量运算；
- **B 类**：全量 5 个模块，`TRITON_INTERPRET=1` 下验证 import + 最小内核执行；
- **SUBOFF 基准**：3 组 benchmark 分别在 sdaa:20/21/22 上运行 5000 步，对比参考公式。

### 2.3 基准配置

| Benchmark | 碰撞模型 | Re | 网格 | 参考公式 |
|-----------|----------|-----|------|----------|
| bench1 | Cumulant D3Q19 | 1,000 | 320×120×120 | Blasius Cf=1.328/√Re |
| bench2 | RANS k-ε (MRT) | 100,000 | 200×80×80 | ITTC-1957 Cf=0.075/(log₁₀Re−2)² |
| bench3 | RANS k-ε (MRT) | 10,000 | 200×80×80 | ITTC-1957 Cf=0.075/(log₁₀Re−2)² |

---

## 3. 关键结果

### 3.1 模块分类统计

| 类别 | 数量 | 占比 | 验证结果 |
|------|------|------|----------|
| A（纯 torch） | 507 | 99.0% | 抽样 15/15 PASS |
| B（triton 依赖） | 5 | 1.0% | 全量 5/5 PASS（解释模式） |
| C（混合） | 0 | 0.0% | — |
| **合计** | **512** | **100%** | **20/20 PASS** |

### 3.2 SUBOFF 阻力基准结果

| Benchmark | Cf_num | Cf_ref | 误差 (%) | 判定 | 耗时 (s) |
|-----------|--------|--------|----------|------|----------|
| bench1 (Cumulant Re=1e3) | 0.04179 | 0.04200 | **0.50** | ✅ PASS | 1983 |
| bench2 (RANS Re=1e5) | 0.00662 | 0.00833 | **20.53** | ❌ FAIL | 1038 |
| bench3 (RANS Re=1e4) | 0.01065 | 0.01875 | **43.21** | ❌ FAIL | 1028 |

**判定标准**：误差 >15% 不可接受。

### 3.3 KBC 诊断（16³ 立方体，20 步）

| 指标 | 值 |
|------|-----|
| BGK 最终 Cd | −3.034 |
| KBC 最终 Cd | 22.062 |
| 参考 Cd | 1.09 |
| KBC/BGK 比值 | −7.27 |
| H 定理违反单元数 | 105（max excess=5.96e-08） |
| 发现 | H-THEOREM VIOLATION; Cd RATIO 异常; H-MODE RETAINED |

### 3.4 B 类模块清单（triton 依赖）

| 模块 | 证据 |
|------|------|
| `_voxel_kernels.py` | @triton.jit; tl.* |
| `backends/triton_backend.py` | tl.* |
| `triton_fused.py` | @triton.jit; tl.* |
| `triton_fused_obstacle.py` | @triton.jit; tl.* |
| `triton_suboff_step_distributed.py` | tl.* |

---

## 4. 审计

### 4.1 数据溯源

所有数字均从工件 JSON 程序化提取，零手抄：
- `artifacts/suboff_bench1_cumulant_sdaa20.json` → bench1 结果
- `artifacts/suboff_bench2_rans_re1e5_sdaa21.json` → bench2 结果
- `artifacts/suboff_bench3_rans_re1e4_sdaa22.json` → bench3 结果
- `artifacts/suboff_best_rans_summary.json` → 汇总对比
- `artifacts/kbc_diagnostic_16cube_20steps.json` → KBC 诊断
- `docs/sdaa_adaptability.md` → 671 行分类报告

### 4.2 机器判据验证

| 判据 | 值 | 状态 |
|------|-----|------|
| 模块总数 | 512 | ✓ |
| A+B+C = 总数 | 507+5+0 = 512 | ✓ |
| A 类抽样验证 | 15/15 PASS | ✓ |
| B 类验证 | 5/5 PASS | ✓ |
| 全量分类清单 | 512 条 | ✓ |

### 4.3 原单结论复核

- **SDAA 适配性**：99.0% 模块可直接迁移 → **结论成立**
- **Bench1 (Cumulant Re=1e3)**：0.5% 误差 → **优秀**
- **Bench2/3 (RANS)**：20.5%/43.2% 误差 → **FAIL**，超过 15% 阈值
- **KBC 诊断**：H 定理违反 + Cd 异常 → **需关注**

---

## 5. 局限与后续

### 5.1 已知局限

1. **RANS 高 Re 精度不足**：bench2 (Re=1e5) 误差 20.5%，bench3 (Re=1e4) 误差 43.2%，均超过 15% 可接受阈值。可能原因：
   - 网格分辨率不足（200×80×80 对高 Re 边界层解析不够）；
   - RANS k-ε 模型在 LBM 框架下的壁面处理需改进；
   - 5000 步可能未充分收敛。

2. **Triton SDAA 后端缺失**：5 个 B 类模块无法在 SDAA 上编译执行 triton 内核，仅能 CPU 解释模式运行，性能远低于预期。

3. **KBC 异常**：16³ 小网格下 KBC 出现 H 定理违反和 Cd 异常，需进一步排查是否为数值不稳定或实现 bug。

4. **抽样验证覆盖**：A 类 507 个模块仅抽样 15 个验证（2.96%），未覆盖全部模块。

### 5.2 后续建议

| 优先级 | 行动 | 预期效果 |
|--------|------|----------|
| P0 | 细化 RANS 高 Re 网格（壁面 y+≈1） | 降低 bench2/3 误差至 <15% |
| P0 | 排查 KBC H 定理违反根因 | 确保熵稳定 |
| P1 | 将 B 类 triton 内核改写为纯 PyTorch | 消除 SDAA 适配障碍 |
| P2 | 推动 Tecorigin 开发 triton SDAA 后端 | 长期生态兼容 |
| P2 | 扩大 A 类验证覆盖率至 >10% | 增强信心 |

---

## 附录：工件清单

| 文件 | 大小/行数 | 说明 |
|------|-----------|------|
| `docs/sdaa_adaptability.md` | 671 行 / 30KB | 主交付物：全量分类报告 |
| `artifacts/kbc_diagnostic_16cube_20steps.json` | KBC vs BGK 诊断 |
| `artifacts/suboff_bench1_cumulant_sdaa20.json` | Bench1 结果 |
| `artifacts/suboff_bench2_rans_re1e5_sdaa21.json` | Bench2 结果 |
| `artifacts/suboff_bench3_rans_re1e4_sdaa22.json` | Bench3 结果 |
| `artifacts/suboff_best_rans_summary.json` | 汇总对比 |

---

*本报告为 mission 11 补报，所有数字从工件程序化提取，零手抄。原单 honest-FAIL 结论如实保留。*
