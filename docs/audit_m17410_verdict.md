# 交差审计：m17410 XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN

**审计人**：board-audit（本任务 m17445）｜**日期**：2026-11-13
**对象**：m17410（XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN·lssfau/lbmpy 的 allen-cahn-csf·TensorLBM，已 succeeded）
**工作树**：/root/tp-agent/work/m17410（分支 agent/m17410，commit e9ce2ffb）

---

## ① 取证（三件套逐项核验）

### run_artifacts — ✅ 真实存在
- `artifacts/xval_m17410/xval_allen_cahn_force_chain.json`（75KB，含 lbmpy_provenance sha256、operator_weights、10 算例指标、decision）
- `artifacts/xval_m17410/xval_allen_cahn_force_chain.csv`（15 行 × 7 列，逐算例 rel_l2_force / sigma_err / tangential_fraction / net_force）
- 3 张图：`fig_force_chain_droplet.png`（111KB）、`fig_force_chain_refinement.png`（41KB）、`fig_force_error_slice.png`（46KB）

### benchmark ref vs got — ✅ 数值齐全且可复现
| 算例 | 变体 | ref (lbmpy) | got (TensorLBM) | rel_l2 | 判据(<15%) |
|---|---|---|---|---|---|
| C2-droplet-R16 | isotropic_d3q19 | 1.17e-14 | 1.17e-14 | **1.17e-14** | ✅ 机器精度 |
| C2-droplet-R16 | central7 | 1.12e-01 | 1.12e-01 | **11.2%** | ✅ |
| C3-W3 | central7 | 2.06e-01 | 2.06e-01 | **20.6%** | ❌ 超判据 |

σ 反演：isotropic −6.47% 与 lbmpy 自身值 −6.47% **逐位相同**（共享解析剖面≠离散平衡的偏差）。

### repro_cmd — ✅ 可执行
`python3 scripts/xval_m17410_allen_cahn_csf_force_chain.py --quick` 审计重跑成功，
C2_droplet_rel_l2_isotropic=1.17e-14、central7=1.12e-01，与 JSON/CSV 完全一致。

### 报告六段 — ✅ 齐全
tensor_reports id=16405（published，3 figures）：①案例介绍 ②方法 ③计算结果 ④收敛性 ⑤差异清单 ⑥失败模式/局限。

---

## ② 判级（L0-L5）

| 维度 | 判定 |
|---|---|
| L2 独立功能实现 | ✅ `src/tensorlbm/phasefield/isotropic_fd.py`（D3Q19 各向同性算子）+ `free_energy.py`（operator 选择）——真代码，输入→计算→输出 |
| L3 系统集成 | ✅ 从 `src/tensorlbm/__init__.py:664-665` 导出；`allen_cahn_lbm.py:373` 与 `multiphase3d.py:852` 生产调用 |
| L4 基准验证 | ⚠️ 有完整交叉对标+误差分析+可复现，但 **tensor_benchmark_cases 本锚 0 行**、**tensorbench_results 0 行**——无机器裁判，基准为自声明 |

**verdict = L3**（功能实质：真代码+系统集成；L4 因基准未入平台台账而不可授）。

---

## ③ 落表

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17410 XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN', 17410, 3, '<JSON>', 'board-audit')
-- id=567, gate_state=pending（maturity_gate 自动裁定）
```

---

## ④ 差异与局限（如实声明）

1. **tensor_benchmark_cases 0 行**：本锚基准未注册平台台账，L4 不可机器裁定——这是平台侧缺口，非案例缺陷。
2. **只对标力链，未对标流体步**：`allen_cahn_step` 变密度 BGK 仍 WITHHELD（静态液滴 5 步发散），F_p/F_v 力项缺口需与流体步联合修复（下一轮）。
3. **只做 D3Q19**：未覆盖 2D 对标（lbmpy 支持 D2Q9/D3Q15/D3Q27）。
4. **默认值未改**：`central7` 仍是默认（向后兼容），isotropic 为显式 opt-in。
