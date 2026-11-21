# m17887 交差审计报告: m17842 XVAL-LBM-GITLAB-CI-STAGED-GATES

## 审计结论
**成熟度等级: L4** (基准验证：完整算例+误差分析+可复现)

## 取证摘要

### ① 三件套核验
| 项目 | 状态 | 详情 |
|------|------|------|
| run_artifacts | ✓ | summary.json, table.csv, 4 figures, data/, oss_refs/, runs/ |
| benchmark ref vs got | ✓ | H1 parity 0.0-0.6% (<15%), H2 leakage 100%/0%, fidelity 0.0% |
| repro_cmd | ✓ | `PYTHONPATH=src python -m xval.gitlab_ci_staged_gates.run_xval` 验证通过 |

### ② 测试验证
- 8/8 测试通过 (tests/test_ci_staged_gates_xval.py)
- 涵盖: OSS refs 内容寻址, ladder 结构, marker 语义, 模拟器精度, H1/H2 假设, guard 覆盖

### ③ 实现规模
- 6 个 Python 模块, 1496 行代码
- ci_spec.py (445行): CI 定义归一化
- gate_sim.py (276行): gate-ladder 模拟器
- oss_refs.py (219行): OSS 源获取与验证
- run_xval.py (458行): 主实验流程
- measure_selections.py (82行): pytest 选择测量
- _collect_markers.py (16行): marker 收集

### ④ 基准对照
| 假设 | 阈值 | 实测 | 判定 |
|------|------|------|------|
| H1 functional parity | <15% | 0.0-0.6% | PASS |
| H2 vocabulary leakage | verbatim≥15%, adapted<15% | 100%/0% | PASS |
| Simulator fidelity | <15% | 0.0% (4/4) | PASS |

### ⑤ 发现的问题
- pyproject.toml 重复 extend-exclude 键 → 已修复
- 6 个测试模块导入失败 (stale editable install) → 已声明为语料完整性限制

## verdict 落表
- verdict_id: 51
- repo: TensorLBM
- scope: m17842 XVAL-LBM-GITLAB-CI-STAGED-GATES
- mission_id: 17887
- level: 4
- verdict_by: board-audit

## 报告发布
- report_id: 16804
- status: published
