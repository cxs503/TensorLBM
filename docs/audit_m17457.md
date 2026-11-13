# 交差审计报告：m17457 XVAL-LBM-AVERAGE-SURROUND-INIT

**审计任务**：m17587（board-audit 交差核验）  
**审计对象**：m17457（XVAL-LBM-AVERAGE-SURROUND-INIT，succeeded）  
**审计日期**：2026-11-13  

---

## ① 取证核验

### run_artifacts ✅ 存在
| 文件 | 大小 | SHA256 | 状态 |
|------|------|--------|------|
| `experiments/xval_average_surround_init/run_xval.py` | 14743 B | ded93b6f | ✅ 存在 |
| `experiments/xval_average_surround_init/artifacts/xval_results.json` | 1772 B | 9726127b | ✅ 存在 |
| `experiments/xval_average_surround_init/artifacts/xval_init_comparison.png` | 54642 B | fd96ad4c | ✅ 存在 |
| `docs/XVAL_AVERAGE_SURROUND_INIT.md` | — | 19b63605 | ✅ 存在 |

### repro_cmd ✅ 可执行
```
cd /root/tp-agent/work/m17457 && python3 experiments/xval_average_surround_init/run_xval.py
```
复跑结果与 xval_results.json 逐字节一致。

### benchmark ref vs got 数值核验
| 工况 | 指标 | 值 | 阈值 | 判定 |
|------|------|-----|------|------|
| 均匀密度 | ρ_diff_rel | 0.0 | <15% | ✅ PASS |
| 均匀密度 | f_diff | 0.0 | <15% | ✅ PASS |
| 密度梯度 | ρ_diff_rel | 0.1999998 (~20%) | <15% | ❌ FAIL |
| 密度梯度 | f_diff | 0.2000001 | <15% | ❌ FAIL |

**总体**：overall_pass=false

### 报告六段 ✅ 齐全
- ① 案例介绍：doc §1 ✅
- ② 方法：doc §2 ✅
- ③ 参数表：doc §3 ✅
- ④ 对照表：doc §4 ✅
- ⑤ 收敛性/敏感性：doc §5 ✅
- ⑥ 失败模式与工件：doc §6 ✅

### 基准裁判
tensor_benchmark_cases 中无 XVAL-LBM-AVERAGE-SURROUND-INIT 锚点行。以任务自身 <15% 判据为准。

---

## ② 判级（L0-L5）

| 级别 | 判定 | 理由 |
|------|------|------|
| L0 | ❌ | 有可执行代码，非纯文档 |
| L1 | ✅ 实质 | 单元级数值对标，可复现 |
| L1 | ❌ 机器 | tests/ 下零 pytest 覆盖 |
| L2 | ❌ | 未添加任何库源码/能力 |
| L3+ | ❌ | 无系统集成 |

**最终级别**：L1（实质）/ L0（机器门）

---

## ③ 落表

```sql
-- tensor_maturity_verdicts id=586
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by, gate_state, gate_level, gate_note)
VALUES ('TensorLBM', 'm17457 XVAL-LBM-AVERAGE-SURROUND-INIT', 17587, 1, '<JSON>', 'm17587-audit', 'pending', 0, '<JSON>');
```

**gate_state**: pending  
**gate_level**: 0（L1 缺 tests/覆盖）  

---

## ④ 发现的问题

1. **零测试覆盖**：tests/ 下无任何 pytest 测试
2. **未入库**：所有工件为 untracked 文件，无 commit
3. **无 tensor_reports**：平台报告纪律被跳过
4. **无 benchmark_cases**：锚点未注册
5. **路径脆弱性**：run_xval.py 的 SRC_DIR 定位依赖 PYTHONPATH 覆盖
6. **文档细节错误**：§5.2 称"6 个液体邻居"，实际为 5 个
7. **过度声明**：§6 称 OSS 可减少 spurious currents，纯推测无证据

---

## ⑤ 处置建议

1. 修复 run_xval.py 路径逻辑（parents[1]→parents[2]）
2. 添加 tests/test_free_surface_init_xval.py
3. 在 tensor_benchmark_cases 注册锚点或退役为"非基准"
4. 提交工件到 git

---

**审计结论**：m17457 交付了实质性的 L1 级单元对标工作，数值结果可复现，但缺乏测试覆盖和平台入库，机器门判为 L0。建议 reopened/rework。
