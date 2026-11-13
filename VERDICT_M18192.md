# Board Audit Verdict: m18192 XVAL-LBM-POINTWISE-REFERENCE-REGRESSION

## 取证核验（三件套）

### 1. run_artifacts ✅ 存在
| 工件 | 路径 | 状态 |
|---|---|---|
| summary.json | `xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/summary.json` | ✅ 存在，含 aggregate/provenance/controls/physics_shear_wave/rlbm_from_definition |
| table.csv | `xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/table.csv` | ✅ 581 行（580 数据点 + 表头） |
| raw_points.json | `xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/runs/raw_points.json` | ✅ 321KB 逐点原始记录 |
| figures | `figures/fig1_eps_vs_tau.png, fig2_eps_vs_amplitude.png, fig3_d3q19_and_physics.png` | ✅ 3 张 PNG |
| predeclared.json | `xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/predeclared.json` | ✅ 预登记判据/停时规则 |

### 2. benchmark ref vs got 数值 ✅ 在容差内（除已知约定差异）
| 臂 | 核 | τ | eps_scaled_inf_max | gate | 判定 |
|---|---|---|---|---|---|
| A1_srt | collide_bgk | 0.51-3.0 | ~1e-15 | 1e-10 | ✅ PASS |
| A2_matmul | collide_bgk_matmul | 0.51-3.0 | ~1e-15 | 1e-10 | ✅ PASS |
| A3_trt | collide_trt | 0.51-3.0 | ~1e-15 | 1e-10 | ✅ PASS |
| A4_mrt_deg | collide_mrt (退化) | 0.51-3.0 | ~1e-15 | 1e-10 | ✅ PASS |
| A5_mrt_nat | collide_mrt (原生) | 0.51-3.0 | ~1e-15 | 1e-10 | ✅ PASS |
| A6_d3q19_cont | collide_bgk_d3q19 vs 连续平衡态 | 0.7/1.0 | 0.217/0.152 | 1e-6 | ❌ FAIL（约定差异，非 bug） |
| A6b_d3q19_disc | collide_bgk_d3q19 vs 离散平衡态 | 0.7/1.0 | ~1e-15 | 1e-6 | ✅ PASS |
| A7_rlbm | collide_rlbm vs 定义式 | 0.51-3.0 | ~1e-10 | 1e-10 | ✅ PASS |
| MRT基 | 重建基 vs 仓内M | — | 0.0 | 1e-12 | ✅ |
| 物理臂 | 剪切波粘度 | 0.55/0.7/1.0 | 2.56%/0.59%/0.0003% | 15% | ✅ PASS |

### 3. repro_cmd ✅ 可执行
```
PYTHONPATH=src python xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/run_xval.py
```
验证：run_xval.py 存在且为完整驱动脚本（19KB），调用 lbmpy_pointwise_ref.py 参考算子与 TensorLBM solver 碰撞核逐点比对。

### 4. 测试 ✅ 17 例全绿
```
tests/test_xval_pointwise_reference_regression.py → 17 passed, 2 warnings
```

### 5. 报告六段 ✅ 齐全
`docs/REPORT_XVAL_LBM_POINTWISE_REFERENCE_REGRESSION.md` 含：
① 案例介绍（物理背景/几何/参数表）✅
② 方法与实现 ✅
③ 计算结果（数字表）✅
④ 结果图（3 张 PNG）✅
⑤ 注意点（失败模式/生态摩擦）✅
⑥ 误差判定（逐项列出，显式回答 <3%）✅

## 判级：L4（基准验证）

理由：
- L0 文档成果 ✅
- L1 数学/单元验证 ✅（17 个 pytest 用例）
- L2 独立功能实现 ✅（lbmpy_pointwise_ref.py 真实代码：输入→计算→输出）
- L3 系统集成 ✅（接入 TensorLBM solver 碰撞核，tests/ 回归门）
- L4 基准验证 ✅（580 点完整算例 + 误差分析 + 可复现 repro_cmd + 预登记判据）
- L5 科研工程成果 ❌（未达：此为对标验证研究，非完整科研工程产品）

##  adversarial review 发现

### Blocking Issues: 无
所有核心臂（D2Q9 四核 + D3Q19 约定对齐后 + RLBM + 物理臂）均通过预登记判据。

### Non-blocking Findings:
1. **A6_d3q19_srt FAIL（21.7%/15.2%）**：非 bug，是 lbmpy 连续平衡态与 TensorLBM 离散平衡态的约定差异。报告已诚实标注并给出 A6b 对照臂（离散对齐后 1e-15 PASS）。
2. **C2/C3@rel=1e-6 未检出**：控制灵敏度不足（增益 ~1.6e-2），但报告已诚实申报并追加 rel=1e-3 对照臂证明灵敏度。未事后下调判据。
3. **lssfau/lbmpy fork 特有改动未核对**：登记为 WITHHELD，因 GitHub raw 被安全策略拦截。

### Overstated Claims: 无
报告中的数字均与 summary.json/table.csv 一致，零手抄。

## Verdict 落表

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18192 XVAL-LBM-POINTWISE-REFERENCE-REGRESSION', 18310, 4,
'{"impl_files":["xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/lbmpy_pointwise_ref.py","xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/run_xval.py","tests/test_xval_pointwise_reference_regression.py"],"entry":"xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/run_xval.py","run_artifacts":["summary.json","table.csv (580 rows)","runs/raw_points.json","figures/*.png"],"tests":"17 passed","benchmark":"580 points, D2Q9 PASS eps~1e-15, D3Q19 discrete PASS 1e-15, A6 continuous FAIL 15-22% convention mismatch, physics 0.59-2.56% < 3%","repro_cmd":"PYTHONPATH=src python xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/run_xval.py","external_anchor":"lbmpy 2.0 PyPI"}',
'board-audit');
```

**Disposition**: APPROVE WITH NOTES — 核心验证完整，A6 约定差异和 C2/C3 灵敏度问题已诚实记录，不影响 L4 判级。
