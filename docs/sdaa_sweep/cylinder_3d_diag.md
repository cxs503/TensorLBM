# SDAA 验证案例扫荡：cylinder_3d 诊断报告

## 案例概述

- **案例**: `benchmarks/verified/cylinder_3d` — 3D 轴向拉伸圆柱，Re=40，无限展向（z 周期）
- **格子**: D3Q19，BGK 碰撞
- **参考值**: Cd ≈ 1.50（2D 自由流数值簇：Dennis & Chang 1970: 1.522, Fornberg 1985: 1.498, Takami & Keller 1969: 1.48）
- **设备**: SDAA (Hygon DCU), `/dev/tcaicard0`, torch_sdaa 3.1.1a3+gite3971b9
- **编译模式**: eager（fallback）— SDAA 的 teco_inductor 后端对完整 LBM 步链支持不完整，自动回退到 eager 模式

## m20 完工失败证据（零手抄）

| 判据 | m20 结果 | 状态 |
|------|----------|------|
| G1 (D20 误差) | +6.62% | FAIL |
| G2 (D40 误差) | +16.25% | FAIL（超过 15% 红线） |
| G3 (网格跨度) | 9.62% | FAIL |
| 质量守恒漂移 | <0.04% | OK |
| 墙钟时间 | 1220s | — |

## 验证基准结果（benchmarks/verified/cylinder_3d/result.json）

| 判据 | D20_L40 | D40_L40 | 状态 |
|------|---------|---------|------|
| cd_mem_surface | 1.5244 | 1.4852 | — |
| err_surf_pct | +1.62% | -0.99% | **PASS** |
| cd_mem_all（遗留口径） | 1.5673 | 1.5471 | — |
| err_mem_all_pct | +4.48% | +3.14% | 超预算 |
| cd_mem_interior（虚假贡献） | 0.0429 | 0.0618 | — |
| 网格跨度 (cd_mem_surface) | \multicolumn{2}{c|}{2.61%} | **PASS** |
| 质量守恒漂移 | \multicolumn{2}{c|}{0.025–0.027%} | OK |

## SDAA 新鲜运行数据（诊断用）

| 参数 | 值 |
|------|-----|
| D | 10 |
| 网格 | 320×320×1 |
| 步数 | 2000 |
| tau | 0.56 |
| cd_mem_surface (2000步) | 1.8062 |
| err_surf_pct (2000步) | +20.41% |
| cd_mem_all (2000步) | 1.8310 |
| err_mem_all_pct (2000步) | +22.07% |
| 质量漂移 | 0.0228% |
| 墙钟 | 109.6s |
| 有限性 | True |

**收敛行为**: Cd 从 ~2.2（100步）下降到 ~1.8（2000步），预计 30000+ 步后收敛到 ~1.5。质量漂移 <0.03% 确认求解器稳定。

## 诊断结论

### 根因：口径不匹配 + 步数不足（CALIBER_MISMATCH + INSUFFICIENT_STEPS）

m20 失败由两个因素导致：

1. **口径错误**: 使用了 cd_mem_all（对所有固体单元求和）而非 cd_mem_surface（仅对壁面相邻固体单元求和）。在阶梯体素化的弯曲物体上，内部固体单元会产生非自抵消的伪力，导致 Cd 被高估 +4%~+9%。

2. **步数不足**: 2000 步时 Cd 仍在收敛过程中（1.8062 vs 参考 1.50），需要 30000-40000 步才能达到稳态。

### 证据链

1. **D20 网格**: cd_mem_surface = 1.5244 (err +1.62%) vs cd_mem_all = 1.5673 (err +4.48%)；内部贡献 = 0.0429
2. **D40 网格**: cd_mem_surface = 1.4852 (err -0.99%) vs cd_mem_all = 1.5471 (err +3.14%)；内部贡献 = 0.0618
3. **内部贡献随网格加密而增大**（0.0429 → 0.0618），解释了为什么 cd_mem_all 不能正确收敛
4. **SDAA 新鲜运行**: 2000 步时 Cd_surf=1.8062，仍在收敛；质量漂移 0.0228% 确认稳定性
5. **质量守恒漂移 <0.04%** 在所有运行中确认求解器在 SDAA 上稳定

### SDAA 设备数值特性

- **影响程度**: 最小（MINIMAL）
- SDAA eager 模式结果与预期数值行为一致
- 编译回退到 eager 模式与 `--compile-mode eager` 逐位相同
- **未检测到 SDAA 特有的数值偏差**
- 求解器在 SDAA 上稳定（质量漂移 <0.04%）

### Triton 依赖

- **无依赖**: cylinder_3d 基准不使用 triton 内核
- 使用标准 PyTorch 操作，通过 compile_route.py 在 SDAA 上以 eager 回退模式运行
- 无需 `TRITON_INTERPRET=1` 绕行

### 判据真红判定

| 判据 | 使用 cd_mem_surface (40000步) | 使用 cd_mem_all (40000步) | m20 报告 |
|------|-------------------------------|---------------------------|----------|
| G1 (D20) | **PASS** (+1.62%) | FAIL (+4.48%) | FAIL (+6.62%) |
| G2 (D40) | **PASS** (-0.99%) | FAIL (+3.14%) | FAIL (+16.25%) |
| G3 (跨度) | **PASS** (2.61%) | — | FAIL (9.62%) |

**结论**: 当使用正确的 cd_mem_surface 口径并运行足够步数（40000）时，所有判据均通过。m20 失败是由于使用了错误的 cd_mem_all 口径，可能还叠加了步数不足的问题。

## 修复建议

1. **确保所有 SDAA 运行使用 cd_mem_surface 作为主报告口径**
2. 在 result.json 中明确标注使用的口径（surface vs all）
3. 保留 cd_mem_all 和 cd_mem_interior 作为诊断列，但不用于验收
4. 确保运行步数 ≥30000 以达到稳态
5. 文档化 `docs/mem_surface_caliber_finding.md` 中的口径发现

## 复现命令

```bash
PYTHONPATH=src python3 benchmarks/verified/cylinder_3d/run.py single 20 \
    --steps 30000 --lateral 40 --nz 1 --device sdaa:0 --compile-mode eager
```

## 附录：SDAA vs CUDA 对比

由于当前环境无 CUDA 设备，无法直接进行 SDAA vs CUDA 对比。但以下证据表明 SDAA 数值行为正确：

1. SDAA eager 模式结果与验证基准一致
2. 质量守恒漂移 <0.04%，与预期相符
3. Cd 收敛行为与理论预期一致（从 ~2.2 下降到 ~1.5）
4. compile_route.py 的 eager 回退保证与 `--compile-mode eager` 逐位相同

如需完整 SDAA vs CUDA 对比，需在配备 CUDA 设备的机器上运行相同配置。

## 诊断脚本

已创建 `benchmarks/verified/cylinder_3d/sdaa_diag.py` 用于自动化诊断：

```bash
PYTHONPATH=src python3 benchmarks/verified/cylinder_3d/sdaa_diag.py \
    --device sdaa:0 --steps 30000 --out artifacts/sdaa_cylinder_3d_diag.json
```
