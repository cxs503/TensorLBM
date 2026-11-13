# m17719 交差审计总结

**审计任务**: m17739  
**被审任务**: m17719 XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER  
**审计日期**: 2026-11-13  
**判级**: **L4** (基准验证)

---

## 验收三件套核验结果

### ✓ 1. run_artifacts 文件真实存在

**9 个工件文件**，全部在 `artifacts/xval_lbt_converter/` 目录下：

| 文件 | 大小 | 状态 |
|---|---|---|
| xval_report.json | 18674 bytes | ✓ 存在 |
| case_poiseuille_water.json | 684 bytes | ✓ 存在 |
| case_cavity_air.json | 691 bytes | ✓ 存在 |
| case_taylor_green_oil.json | 682 bytes | ✓ 存在 |
| case_channel_low_re.json | 687 bytes | ✓ 存在 |
| report_payload.json | 4614 bytes | ✓ 存在 |
| fig_anchor_xval.png | 48316 bytes | ✓ 存在 |
| fig_derived_comparison.png | 52877 bytes | ✓ 存在 |
| fig_grid_convergence.png | 55888 bytes | ✓ 存在 |

### ✓ 2. benchmark ref vs got 数值在容差内

**交叉验证结果**（TensorLBM LatticeBridge vs 2b-t/LB-t Converter）：

| 量 | 最大相对差异 | < 15%? | < 1e-12? |
|---|---|---|---|
| dx | 0.00e+00 | ✓ | ✓ |
| dt | 0.00e+00 | ✓ | ✓ |
| nu_lb | 2.17e-16 | ✓ | ✓ |
| tau | 1.75e-16 | ✓ | ✓ |
| Ma | 2.00e-16 | ✓ | ✓ |
| Re | 1.78e-16 | ✓ | ✓ |

**锚点模拟误差**：

| Anchor | Sim Error | Converter Diff | < 15%? |
|---|---|---|---|
| poiseuille | 2.17e-03 | 1.74e-16 | ✓ |
| couette | 1.83e-04 | 0.00e+00 | ✓ |
| stokes | 2.22e-16 | 0.00e+00 | ✓ |
| taylor_green | 3.24e-04 | 0.00e+00 | ✓ |

**结论**: 所有转换器差异在机器精度（<1e-12），远低于 15% 阈值。

### ✓ 3. repro_cmd 可执行

```bash
cd /root/tp-agent/work/m17719 && \
  python3 -m tensorlbm.xval --out artifacts/xval_lbt_converter && \
  python3 -m pytest tests/test_xval_lbt_converter.py -v
```

**验证结果**:
- ✓ 30/30 pytest 测试通过
- ✓ 生成的工件与原始工件数值一致
- ✓ 运行时间 ~22 秒

---

## 判级理由

**L4 = 基准验证（完整算例 + 误差分析 + 可复现）**

### 证据清单

1. **2602 行真实实现代码**
   - converter.py: 192 行
   - case_schema.py: 286 行
   - xval_benchmark.py: 651 行
   - __main__.py: 6 行
   - cli.py: 79 行
   - 其他支撑模块: ~1388 行

2. **CLI 入口点**: `python -m tensorlbm.xval --out <dir>`

3. **9 个运行工件**: JSON 报告 + PNG 图表

4. **30 个 pytest 测试全部通过**:
   - TestLbtStyleConverter: 13 tests
   - TestCaseSchema: 5 tests
   - TestAnchorXval: 8 tests
   - TestFullXval: 4 tests

5. **4 个集成调用方**（测试套件之外）:
   - app/backend/routers/xval_taylor_green.py
   - benchmarks/xval_aa_pattern_even_odd.py
   - benchmarks/xval_taylor_green/run_xval.py
   - scripts/xval_figures.py

6. **交叉验证基准**:
   - 4 个解析锚点（Poiseuille, Couette, Stokes, Taylor-Green）
   - 网格收敛研究（16/32/64）
   - u_lb 参数敏感性分析

7. **所有转换器差异在机器精度**（<1e-12），远低于 15% 阈值

8. **repro_cmd 验证通过**，产生相同工件

### 非 L5 原因

- 无外部实验验证（仅与解析解和参考实现比较）
- 无出版级不确定度量化（超越解析锚点）

**级别 = 功能实质**；100+ 行报告和 3 张图表不提升级别。

---

## 落表

**SQL 文件**: `verdict_m17719.sql` (77 行)

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) 
VALUES ('TensorLBM', 'm17719 XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER', 17739, 4, '<JSON evidence>', 'board-audit');
```

---

## 报告

**报告文件**:
- `docs/reports/m17719_audit_report.json` (11K)
- `docs/reports/m17719_audit_report.md` (11K)

**报告六要素**:
1. ✓ 案例介绍（物理背景/几何/参数表）
2. ✓ 怎么做的（方法/流程/关键设置/工具链）
3. ✓ 计算结果（数字表，程序化注入零手抄）
4. ✓ 结果图（3 张 PNG 图表）
5. ✓ 注意点（踩过的坑/局限/适用范围）
6. ✓ 误差判定（关键量 vs 基准误差逐项列出）

---

## 结论

**m17719 交差通过**，判级 **L4**。

所有验收三件套真实存在，基准数值在容差内，repro_cmd 可执行。

建议 maturity_gate 裁定 **confirmed**（消账）。

---

## 产物清单

| 产物 | 路径 | 状态 |
|---|---|---|
| 判级 SQL | verdict_m17719.sql | ✓ 已生成 |
| 审计报告 (JSON) | docs/reports/m17719_audit_report.json | ✓ 已生成 |
| 审计报告 (MD) | docs/reports/m17719_audit_report.md | ✓ 已生成 |
| 审计总结 | AUDIT_SUMMARY.md | ✓ 已生成 |

---

**审计完成时间**: 2026-11-13 08:28  
**审计员**: board-audit (m17739)
