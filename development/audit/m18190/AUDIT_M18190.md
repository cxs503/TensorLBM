# Board Audit: m18190 XVAL-LBM-PHASEFIELD-MCMP-POISEUILLE 交差核验

## 审计结论

**判级: L4（完整基准验证）**

m18190 交付物经独立核验，三件套齐全且可复现：
- ✅ run_artifacts: `xval/artifacts/phasefield_mcmp_poiseuille/` 含 results.json + 4 张对比图
- ✅ benchmark 数值: 基准 L2=5.46% < 15% PASS，H=64 收敛 L2=3.72% < 15% PASS
- ✅ repro_cmd: `python3 xval/xval_phasefield_mcmp_poiseuille.py` 在 ~25s 内复现相同结果

## 核验详情

### 数值复现（审计者独立运行）
| 案例 | H | 步数 | u_max (sim) | u_max (ana) | L2 误差 | 通过 |
|---|---|---|---|---|---|---|
| 基准复现 | 46 | 5000 | 1.6176e-2 | 1.5870e-2 | 5.46% | ✓ |
| 原报告基准 | 46 | 5000 | 1.6176e-2 | 1.5870e-2 | 5.46% | ✓ |

### 代码审查
- 860 行 Python/PyTorch 实现
- SOY 自由能 D2Q9 LBM 忠实复现 ebinan92/Fingering_dynamics 算法
- 含解析解对照、收敛性研究、参数敏感性扫描
- 代码位于 `xval/` 目录，未集成入 `src/tensorlbm/` 主求解器

### 未达 L5 原因
1. 代码未集成入主求解器（在 xval/ 目录）
2. 仅与解析解比对，未与 OSS 原始数值输出直接对照
3. 无 pytest 单元测试覆盖

## 产物
- `verdict_m18190.sql`: 成熟度判定 SQL
- Report ID: 17137
