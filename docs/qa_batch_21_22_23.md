# 质检报告：批次 21, 22, 23

> **质检批次**: batch_21_22_23  
> **质检员**: TensorPlatform QA Agent  
> **日期**: 2026-07-26  
> **对应工作单**: mission_id 11, 14, 8  

---

## 判定汇总表

| 报告 ID | mission_id | 标题 | 判定 | 主要理由 |
|---------|-----------|------|------|----------|
| 21 | 11 | TensorLBM SDAA 适配性普查 | **PASS** | 五节完整，3/3 数字命中工件，honest-FAIL 如实保留 |
| 22 | 14 | SDAA 验证扫荡——cavity 案例 | **PASS** | 五节完整，3/3 数字命中工件，结论与证据一致 |
| 23 | 8 | SDAA 加速卡点亮 e2e 验证 | **MINOR** | 五节完整，3/3 数字可溯源但**工件缺失**（无 artifacts/ 目录，仅未跟踪文件 sdaa_probe.txt） |

---

## 逐篇质检详情

### 报告 21（mission_id=11）→ PASS

**标题**: Mission 11 补报：TensorLBM SDAA 适配性普查

#### ① 结构五节检查
| 节 | 状态 | 说明 |
|----|------|------|
| 目的 | ✅ | Section 1 明确列出 3 个验证目标 |
| 方法 | ✅ | Section 2 含模块分类方法、验证方法、基准配置表 |
| 关键结果（含数字表） | ✅ | Section 3 含 4 个子表：模块分类统计、SUBOFF 阻力基准、KBC 诊断、B 类模块清单 |
| 审计 | ✅ | Section 4 含数据溯源、机器判据验证、原单结论复核 |
| 局限 | ✅ | Section 5 含 4 条已知局限 + 后续建议表 |

#### ② 数字溯源抽查（3/3 命中）

| # | 报告数字 | 工件路径 | 工件值 | 命中 |
|---|----------|----------|--------|------|
| 1 | bench1 Cf_num=0.04179, Cf_ref=0.04200, error=0.50% | `m11/artifacts/suboff_bench1_cumulant_sdaa20.json` | Cf_numerical=0.041785936, Cf_ref=0.041995047, error_pct=0.498 | ✅ |
| 2 | bench2 Cf_num=0.00662, Cf_ref=0.00833, error=20.53% | `m11/artifacts/suboff_bench2_rans_re1e5_sdaa21.json` | Cf_numerical=0.0066225, Cf_ref=0.0083333, error_pct=20.53 | ✅ |
| 3 | KBC BGK Cd=−3.034, KBC Cd=22.062, ref=1.09 | `m11/artifacts/kbc_diagnostic_16cube_20steps.json` | bgk_final_Cd=−3.034, kbc_final_Cd=22.062, reference_Cd=1.092 | ✅ |

#### ③ honest-FAIL 保留
- 原单标签含 `honest-FAIL`，原单结论为"高 Re RANS 基准误差 >15%"
- 报告 Section 3.2 明确标注 bench2 (20.53%) 和 bench3 (43.21%) 为 ❌ FAIL
- Section 4.3 原单结论复核如实保留 FAIL 判定
- ✅ **未粉饰，honest-FAIL 完整保留**

#### ④ 图
- `figures: []`（空数组）
- 报告未明确写明"无图"理由 → **MINOR 备注**（不影响判定）

#### ⑤ 结论不超证据
- 报告结论严格限定在工件数据范围内
- 局限 Section 5.1 明确列出 RANS 精度不足、Triton 后端缺失、KBC 异常、抽样覆盖率低等 4 条局限
- ✅ **无超出数据支撑的断言**

---

### 报告 22（mission_id=14）→ PASS

**标题**: Mission 14 补报：SDAA 验证扫荡——cavity 案例（TensorLBM）

#### ① 结构五节检查
| 节 | 状态 | 说明 |
|----|------|------|
| 目的 | ✅ | Section 1 明确 4 个验证目标 |
| 方法 | ✅ | Section 2 含物理设置、运行脚本、Triton 绕行、判据表 |
| 关键结果（含数字表） | ✅ | Section 3 含设备表、网格收敛数据表、判据汇总表、收敛趋势 |
| 审计 | ✅ | Section 4.1 审计清单 5 项全通过 |
| 局限 | ✅ | Section 4.2 含 4 条局限 + 4.3 与其他工况关系表 |

#### ② 数字溯源抽查（3/3 命中）

| # | 报告数字 | 工件路径 | 工件值 | 命中 |
|---|----------|----------|--------|------|
| 1 | 32² max_abs_dev_pct=2.248% | `m14/artifacts/sdaa_cavity.json` → grids.32.max_abs_dev_pct | 2.2479876071810687 | ✅ |
| 2 | 48² max_abs_dev_pct=1.903% | `m14/artifacts/sdaa_cavity.json` → grids.48.max_abs_dev_pct | 1.9034586855173146 | ✅ |
| 3 | 总墙钟 103.7s | `m14/artifacts/sdaa_cavity.json` → total_elapsed_s | 103.7 | ✅ |

#### ③ honest-FAIL 保留
- 原单标签不含 `honest-FAIL`，原结论为 PASS
- 报告结论 PASS，与原文一致
- ✅ **无粉饰问题**

#### ④ 图
- `figures: []`（空数组）
- 报告未明确写明"无图"理由 → **MINOR 备注**（不影响判定）

#### ⑤ 结论不超证据
- 报告明确标注"粗网格快速验证"级别
- Section 4.2 局限指出仅 Re=100、单设备、Triton 解释模式等限制
- Section 4.3 明确列出其他工况"待扫荡"
- ✅ **无超出数据支撑的断言**

---

### 报告 23（mission_id=8）→ MINOR

**标题**: Mission 8 补报：SDAA 加速卡点亮 e2e 验证（honest-FAIL）

#### ① 结构五节检查
| 节 | 状态 | 说明 |
|----|------|------|
| 目的 | ✅ | Section 1 明确 3 个验证目标 |
| 方法 | ✅ | Section 2 含探针命令、验证维度表、证据链说明 |
| 关键结果（含数字表） | ✅ | Section 3 含探针输出表、原始输出、判定表 |
| 审计 | ✅ | Section 4.1 本验证证明了什么 |
| 局限 | ✅ | Section 4.2 含 7 条 honest-FAIL 披露 + 4.3 后续建议表 |

#### ② 数字溯源抽查（3/3 命中，但工件缺失）

| # | 报告数字 | 工件路径 | 工件值 | 命中 |
|---|----------|----------|--------|------|
| 1 | 加速卡数量=1 | `m8/sdaa_probe.txt` → "card 1" | card 1 /dev/tcaicard5 | ✅ |
| 2 | 设备名称=/dev/tcaicard5 | `m8/sdaa_probe.txt` | /dev/tcaicard5 | ✅ |
| 3 | 矩阵乘法耗时=12.9 ms | `m8/sdaa_probe.txt` → "matmul_ms 12.9" | 12.9 | ✅ |

**⚠️ 工件缺失**: m8 工作目录**无 `artifacts/` 目录**，证据仅存于未跟踪文件 `sdaa_probe.txt`（未入 git，无 checksum 验证）。报告 Section 2.3 和 4.2 第 5 条已自述此问题，但作为质检发现仍需标注。

#### ③ honest-FAIL 保留
- 原单标签含 `honest-FAIL`
- 报告 Section 4.2 明确披露 7 条"本验证未证明什么"
- 报告结论为硬件层面 PASS，但明确声明"不构成科学计算精度或业务集成的证据"
- ✅ **honest-FAIL 如实保留，未粉饰**

#### ④ 图
- `figures: []`（空数组）
- 报告未明确写明"无图"理由 → **MINOR 备注**（不影响判定）

#### ⑤ 结论不超证据
- 报告极为谨慎，Section 4.2 列出 7 条未证明事项
- 结论明确限定为"硬件烟雾测试"级别
- ✅ **无超出数据支撑的断言**

---

## 缺陷清单

| 报告 ID | 缺陷类型 | 严重度 | 描述 | 证据 |
|---------|----------|--------|------|------|
| 21 | 图缺失说明 | MINOR | `figures=[]` 但未写明无图理由 | 报告 API 返回 figures=[]，正文无"无图"声明 |
| 22 | 图缺失说明 | MINOR | `figures=[]` 但未写明无图理由 | 报告 API 返回 figures=[]，正文无"无图"声明 |
| 23 | 工件缺失 | MINOR | m8 无 `artifacts/` 目录，证据仅存未跟踪文件 `sdaa_probe.txt` | `ls /root/tp-agent/work/m8/artifacts/` → No such file or directory |
| 23 | 图缺失说明 | MINOR | `figures=[]` 但未写明无图理由 | 报告 API 返回 figures=[]，正文无"无图"声明 |

---

## 处置建议

| 报告 ID | 判定 | 处置 |
|---------|------|------|
| 21 | PASS | 无需返工 |
| 22 | PASS | 无需返工 |
| 23 | MINOR | 仅记录，不派返工单（工件缺失为已知问题，报告已自述） |

---

*质检完成。所有数字均从工件程序化复核，零手抄。*
