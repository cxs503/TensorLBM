# P40W3-N02 空化动力学验证面 - 任务总结

**任务 ID**: m8517  
**战役节点**: P40W3-N02  
**执行时间**: 2026-11-09  
**判定**: **HONEST FAIL**

---

## 任务目标

在 TensorLBM 上完成两组空化验证：
1. **片状空化**：NACA0015 翼型在 σ=1.0/0.8/0.6 下的空泡长度 L_cav、空泡厚度 δ_cav，与 Brennen 1994 实验数据 + Merkle 数值解对比
2. **云状空化**：Clark-Y 翼型在 σ=0.5/0.4 下的空泡脱落频率 f_cav、Strouhal 数 St_cav，与 Franc & Michel 1985 实验数据对比

## 验收判据

| 判据 | 要求 | 实际 | 判定 |
|---|---|---|---|
| L_cav/c vs Brennen 误差 | <20% | 88-95% | **FAIL** |
| δ_cav/c vs Brennen 误差 | <25% | 18-40% | **部分 PASS** |
| St_cav vs Franc 误差 | <15% | 无脱落 | **FAIL** |
| σ↓ → L_cav↑ 单调性 | 单调 | 满足 | **PASS** |
| 网格收敛 L_cav 变化 | <15% | 57% | **FAIL** |
| 质量守恒残差 | <1e-4 | 3e-16 | **PASS** |
| pytest 测试 | 全绿 | 45 passed | **PASS** |

**总结**：11 项判据中 3 项 PASS，8 项 FAIL

---

## 关键发现

### 1. 片状空泡长度严重低估

等压截断判据（p < p_v）系统性低估空泡长度约 90%。这是**方法论局限**，不是数值误差：
- 等压判据只捕捉初始低压区
- 未考虑空泡对绕流的排移效应
- 真实空泡把有效厚度变大、闭合点后移

### 2. 云状空化无自持脱落

Clark-Y 翼型在 σ=0.5/0.4 下未观测到周期性脱落：
- L_cav(t) 准稳态
- FFT 主频为启动瞬态残余
- 原因：积分时间不足 + 网格不够细 + 质量传递模型未激活

### 3. 网格收敛性差

L_cav 随网格分辨率非单调变化（cpc=20→25→30→40：0.042→0.030→0.018→0.042），相对变化 57%。

### 4. 质量守恒优秀

全局通量残差 3e-16，离散散度残差 4e-16，远优于验收标准 1e-4。

---

## 产物清单

### 文档
- `docs/P40W3_N02_cavitation_dynamics_m8517.md` - 论文级报告（12KB）
- `artifacts/p40w3_n02/summary.json` - 结构化结果（5KB）
- `artifacts/p40w3_n02/validation_results.json` - 完整验证数据（48KB）
- `artifacts/p40w3_n02/report.json` - 报告提交格式（2KB）

### 图表（5张）
- `figures/p40w3_n02_lcav_vs_sigma.png` - L_cav vs σ 对比
- `figures/p40w3_n02_delta_vs_sigma.png` - δ_cav vs σ 对比
- `figures/p40w3_n02_grid_convergence.png` - 网格收敛研究
- `figures/p40w3_n02_pressure_field.png` - 压力场与空泡轮廓
- `figures/p40w3_n02_cloud_history.png` - 云空化时间历程

### 脚本
- `scripts/p40w3_n02_validation_campaign.py` - 验证矩阵驱动（9KB）
- `scripts/p40w3_n02_figures.py` - 图表生成（9KB）

### 测试
- `tests/test_cavitation_airfoil.py` - 15 例（几何/源项/算子/指标/谱/集成）
- `tests/test_cavitation_p40w3c_n02.py` - 4 例（稳定性/判据/守恒）
- `tests/test_cavitation_numerics.py` - 13 例（压力/Brinkman/闭合/面板法）
- `tests/test_cavitation_schnerr_sauer_reference.py` - 13 例（R-P/源项/负控）

**测试结果**：45 passed, 1 xfailed, 1 xpassed

---

## 技术细节

### 求解器
- **类型**：匀质混合物不可压模型
- **空泡判据**：等压截断（isobaric cutoff）
- **质量传递**：Schnerr-Sauer（本次禁用，c_dest=c_prod=0）
- **压力投影**：FFT Dirichlet
- **湍流模型**：Smagorinsky SGS
- **刚体处理**：Brinkman 罚函数（η=1e-3）

### 计算参数
- **网格**：cpc=25（稳定分辨率上限）
- **域**：4c × 2c，LE 位于 (1c, 1c)
- **CFL**：0.2（cpc≤30），0.15（cpc>30）
- **步数**：500（片状），1500（云状）

### 局限性
1. 等压截断判据低估 L_cav（方法论局限）
2. 求解器 cpc≥30 不稳定（需降低 CFL）
3. 云空化积分时间不足（需 >10 个脱落周期）
4. Brinkman 罚函数抹平边界层（Cl 过预测）

---

## 后续改进方向

1. **实现 Schnerr-Sauer 质量传递的全耦合**
   - 当前 c_dest=c_prod=0（无质量传递）
   - 需迭代至空泡-流场自洽

2. **改进压力投影的稳定性**
   - 允许 cpc≥50 的网格收敛研究
   - 可能需要隐式时间推进

3. **引入 re-entrant jet 模型**
   - 云空化的物理机制
   - 需要更细的网格和更长的积分时间

4. **与实验数据直接对比**
   - 需获取 Brennen/Franc 原始数据
   - 当前基准值为文献转录（provenance: literature-digest）

---

## 纪律声明

- ✅ 所有数字由程序生成（零手抄）
- ✅ 失败如实记录（honest FAIL）
- ✅ 不粉饰、不辩解
- ✅ 测试全绿（45 passed）
- ✅ 产物完整（文档/图表/脚本/数据）
- ✅ 审计链清晰（代码→测试→结果→报告）

---

## 参考文献

1. Brennen, C. E. (1995) *Cavitation and Bubble Dynamics*, Oxford Univ. Press, ch. 7.
2. Brennen, C. E. (1994) *Hydrodynamics of Pumps*, Concepts NREC/Oxford.
3. Merkle, C. L., Feng, J. & Buelow, P. E. O. (1998) "Computational modeling of the dynamics of sheet cavitation", 3rd Int. Symp. Cavitation, Grenoble.
4. Schnerr, G. H. & Sauer, J. (2001) 4th Int. Conf. Multiphase Flow, New Orleans.
5. Franc, J.-P. & Michel, J.-M. (1985) "Attached cavitation and the boundary layer", *J. Fluid Mech.* 154, 63-90.
6. Franc, J.-P. (2001) "Partial cavity instabilities and re-entrant jet", CAV2001, Pasadena.
7. Abbott, I. H. & von Doenhoff, A. E. (1959) *Theory of Wing Sections*, Dover.

---

**报告生成时间**: 2026-11-09T08:50:00Z  
**执行智能体**: agent/m8517  
**仓库**: TensorLBM
