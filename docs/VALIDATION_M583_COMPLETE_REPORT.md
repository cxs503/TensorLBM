# TensorLBM 验证框架完整报告 (Mission m583/m606)

## 1. 背景与动机

### 1.1 OpenFOAM tutorials 组织结构对比

OpenFOAM 作为工业级 CFD 求解器，其验证测试按以下层级组织：

| 层级 | OpenFOAM 结构 | TensorLBM 对应 | 差异说明 |
|------|--------------|---------------|---------|
| **基础验证** | `tutorials/basic/` | `validation/m583_cases/` | OpenFOAM 按物理场分类（incompressible/compressible），TensorLBM 按案例类型分类 |
| **案例组织** | 每个 tutorial 含 `0/`, `constant/`, `system/` 三目录 | 单 Python 脚本 + JSON 结果 | OpenFOAM 需要完整的 case 目录树，TensorLBM 采用脚本化验证 |
| **边界条件** | `0/U`, `0/p` 等字段文件 | 脚本内直接定义 BC 函数 | OpenFOAM 声明式，TensorLBM 编程式 |
| **网格定义** | `constant/polyMesh/` | 脚本内生成 | OpenFOAM 需要 blockMesh/snappyHexMesh，TensorLBM 规则网格直接构造 |
| **求解器配置** | `system/fvSchemes`, `fvSolution` | 脚本内参数 | OpenFOAM 分离配置，TensorLBM 内嵌参数 |
| **后处理** | `foamToVTK`, `paraFoam` | matplotlib 直接绘图 | OpenFOAM 依赖 ParaView，TensorLBM 自包含可视化 |

### 1.2 关键差异总结

1. **配置复杂度**：OpenFOAM 需要 3+ 配置文件，TensorLBM 单脚本可执行
2. **网格生成**：OpenFOAM 需要独立网格工具，TensorLBM 规则网格内嵌生成
3. **依赖链**：OpenFOAM 依赖完整工具链（blockMesh → solver → postprocess），TensorLBM 端到端自包含
4. **验证速度**：OpenFOAM 案例通常需要分钟级，TensorLBM 秒级（torch.compile 加速）
5. **可复现性**：OpenFOAM 需要完整 case 目录，TensorLBM 单脚本即可复现

### 1.3 验证目标

本验证框架对标 OpenFOAM 的 `tutorials/incompressible/icoFoam/` 系列，选取三个经典案例：

1. **Poiseuille 流**：压力驱动平行板层流，验证稳态精度
2. **Couette 流**：剪切驱动平行板层流，验证边界条件精度
3. **Taylor-Green 涡**：衰减涡旋，验证瞬态精度与能量守恒

---

## 2. 方法与流程

### 2.1 数值方法

**格子 Boltzmann 方法 (LBM)**：
- 格子模型：D2Q9（二维九速度）
- 碰撞算子：BGK 单松弛时间
- 无量纲松弛时间：τ = 0.8（对应运动粘度 ν = 0.1）
- 边界条件：
  - Poiseuille：Zou-He 压力入口/出口 + 反弹壁面
  - Couette：移动壁面（反弹 + 速度修正）+ 周期性边界
  - Taylor-Green：全周期性边界

### 2.2 工具链

```
TensorLBM 原语 (src/tensorlbm/)
  ├─ d2q9.py: 平衡态分布、离散速度、权重
  ├─ solver.py: BGK 碰撞、流步骤
  ├─ boundaries.py: Zou-He 压力边界
  └─ torch.compile: 自动微分加速（~30x 提速）

验证脚本 (validation/m583_cases/run_all_cases.py)
  ├─ 案例定义与参数设置
  ├─ 时间推进循环
  ├─ 解析解对比
  ├─ 误差计算（L2 相对误差、最大相对误差）
  ├─ 门禁评估（7 项门禁）
  ├─ 红队自检（注入坏值验证门禁必红）
  └─ 可视化（matplotlib）
```

### 2.3 门禁体系

每个案例通过 7 项门禁：

| 门禁 | 描述 | 阈值 |
|------|------|------|
| G1_library_entry | 使用 TensorLBM 原语 | 必须调用 tensorlbm.* |
| G2_direct_observable | L2 相对误差 | < 3% |
| G3_monotone_convergence | 最大相对误差 | < 3% |
| G4_reference_locked | 解析解/参考值已锁定 | 必须有解析解或文献值 |
| G5_steady_state / G5_fit_quality | 稳态/拟合质量 | 稳态判定或 R² > 0.99 |
| G6_mass_conservation | 质量守恒 | 质量漂移 < 0.01% |
| G7_finite_values / G7_consistency | 有限值/一致性 | 无 NaN/Inf，半段一致性 |

---

## 3. 计算结果

### 3.1 Poiseuille 2D

**物理参数**：
- 通道高度：H = 40 格子
- 通道长度：L = 120 格子
- 松弛时间：τ = 0.8
- 运动粘度：ν = 0.1
- 雷诺数：Re = 10.67（基于平均速度）
- 最大解析速度：u_max = 0.04

**数值结果**：
- 时间步数：14200 步
- 达到稳态：是
- 最大数值速度：u_max_num = 0.040148
- **L2 相对误差**：0.004427 (0.44%)
- **最大相对误差**：0.4607%
- **u_max 误差**：0.4329%
- 质量漂移：0.000000%
- 计算耗时：14.9 秒

**误差判定**：
- ✅ L2 相对误差 0.44% < 3%
- ✅ 最大相对误差 0.46% < 3%
- ✅ u_max 误差 0.43% < 3%
- ✅ 所有误差均 < 3%

### 3.2 Couette 2D

**物理参数**：
- 通道高度：H = 40 格子
- 通道长度：L = 40 格子
- 松弛时间：τ = 0.8
- 运动粘度：ν = 0.1
- 雷诺数：Re = 20.0（基于顶板速度）
- 顶板速度：U0 = 0.05

**数值结果**：
- 时间步数：3000 步
- 达到稳态：是
- **L2 相对误差**：1.21e-04 (0.012%)
- **最大相对误差**：0.0182%
- **顶板行误差**：0.000123%
- 质量漂移：0.000879%
- 计算耗时：1.1 秒

**误差判定**：
- ✅ L2 相对误差 0.012% < 3%
- ✅ 最大相对误差 0.018% < 3%
- ✅ 顶板行误差 0.00012% < 3%
- ✅ 所有误差均 < 3%

### 3.3 Taylor-Green 2D

**物理参数**：
- 网格尺寸：N = 64×64
- 松弛时间：τ = 0.596
- 运动粘度：ν = 0.032
- 雷诺数：Re = 100（基于初始涡旋尺度）
- 初始速度幅值：u0 = 0.05
- 波数：k = 2π/N = 0.0982

**解析衰减率**：
- 能量衰减率：γ_E_theory = 4νk² = 1.2337e-03
- 速度衰减率：γ_vel_theory = 2νk² = 6.1685e-04

**数值结果**：
- 时间步数：5000 步（每 50 步记录一次）
- **能量衰减率误差**：-0.0100%
- **速度衰减率误差**：-0.0182%
- **拟合 R²**：0.99999995
- 半段一致性：γ₁ = 1.2333e-03, γ₂ = 1.2337e-03
- 质量漂移：0.00627%
- 计算耗时：6.9 秒

**误差判定**：
- ✅ 能量衰减率误差 0.01% < 3%
- ✅ 速度衰减率误差 0.018% < 3%
- ✅ R² = 1.0000 > 0.99
- ✅ 半段一致性验证通过
- ✅ 所有误差均 < 3%

### 3.4 误差汇总表

| 案例 | L2 相对误差 | 最大相对误差 | 关键量误差 | 是否 < 3% |
|------|------------|-------------|-----------|----------|
| Poiseuille | 0.44% | 0.46% | u_max: 0.43% | ✅ 是 |
| Couette | 0.012% | 0.018% | U_top: 0.00012% | ✅ 是 |
| Taylor-Green | N/A | N/A | γ_E: 0.01%, γ_vel: 0.018% | ✅ 是 |

**总体判定**：✅ 所有案例所有误差均 < 3%，验证通过

---

## 4. 结果图示

### 4.1 Poiseuille 流

**图 1：速度剖面对比**
- 蓝色实线：数值解
- 红色虚线：解析解 u(y) = 4·u_max·(y/H)·(1-y/H)
- 观察：数值解与解析解高度吻合

**图 2：相对误差分布**
- 柱状图展示 L2 误差和最大相对误差
- 绿色虚线：3% 阈值
- 观察：所有误差远低于阈值

### 4.2 Couette 流

**图 3：速度剖面对比**
- 蓝色实线：数值解
- 红色虚线：解析解 u(y) = U0·y/H
- 观察：线性剖面完美复现

**图 4：相对误差分布**
- 柱状图展示 L2 误差和最大相对误差
- 绿色虚线：3% 阈值
- 观察：误差在 0.02% 量级，远低于阈值

### 4.3 Taylor-Green 涡

**图 5：能量衰减曲线**
- 蓝色实线：数值模拟能量 E(t)
- 红色虚线：指数拟合 E0·exp(-γ_E·t)
- 观察：能量指数衰减，拟合 R² ≈ 1.0

**图 6：衰减率对比**
- 柱状图展示能量衰减率和速度衰减率
- 对比解析值与数值值
- 观察：误差在 0.02% 量级

**（注：实际图片文件位于 validation/m583_cases/figures/）**

---

## 5. 红队自检

### 5.1 测试目的

验证门禁体系的鲁棒性：当注入坏值时，门禁必须变红（FAIL）。

### 5.2 测试方案

对每个案例注入超出阈值的误差：

| 案例 | 注入值 | 预期 | 实际 | 门禁变红 |
|------|--------|------|------|---------|
| Poiseuille | L2 误差 = 15% | FAIL | FAIL | ✅ 是 |
| Couette | L2 误差 = 8% | FAIL | FAIL | ✅ 是 |
| Taylor-Green | 能量误差 = 8.5% | FAIL | FAIL | ✅ 是 |

### 5.3 结论

✅ 红队测试通过：所有坏值注入均被门禁正确识别，验证了门禁体系的可靠性。

---

## 6. 注意点与局限

### 6.1 踩过的坑

1. **torch.compile 兼容性**：
   - 问题：早期版本 torch.compile 对动态形状支持不佳
   - 解决：固定网格尺寸，避免动态形状

2. **稳态判定**：
   - 问题：Poiseuille 流需要 14000+ 步才能收敛
   - 解决：设置最小步数 3000，最大步数 15000，监控速度变化

3. **边界条件精度**：
   - 问题：Zou-He 压力边界在角点处需要特殊处理
   - 解决：入口/出口边界分开实现，避免角点冲突

### 6.2 局限性

1. **仅验证层流**：
   - 当前仅验证低雷诺数层流（Re < 100）
   - 未验证湍流模型（需要 Smagorinsky 等亚格子模型）

2. **仅验证二维**：
   - 当前仅验证 D2Q9 二维模型
   - 未验证 D3Q19 三维模型

3. **仅验证规则网格**：
   - 当前仅验证规则矩形网格
   - 未验证复杂几何（需要 immersed boundary 或 body-fitted 网格）

4. **仅验证单松弛时间**：
   - 当前仅验证 BGK 单松弛时间
   - 未验证 MRT（多松弛时间）或 TRT（双松弛时间）

### 6.3 适用范围

本验证框架适用于：
- ✅ D2Q9 格子模型
- ✅ BGK 碰撞算子
- ✅ 低雷诺数层流（Re < 100）
- ✅ 规则矩形网格
- ✅ 简单几何（平行板、周期域）

不适用于：
- ❌ 高雷诺数湍流
- ❌ 复杂几何
- ❌ 多相流
- ❌ 可压缩流（Ma > 0.3）

---

## 7. 展望

### 7.1 短期改进

1. **扩展到三维**：
   - 实现 D3Q19 验证案例
   - 验证 3D Poiseuille、3D Taylor-Green

2. **湍流验证**：
   - 实现 Smagorinsky 亚格子模型
   - 验证高雷诺数 Taylor-Green（Re = 1000+）

3. **复杂几何**：
   - 实现 immersed boundary method
   - 验证圆柱绕流（对标 OpenFOAM pitzDaily）

### 7.2 长期目标

1. **对标 OpenFOAM 完整教程**：
   - 覆盖 `tutorials/incompressible/` 全部案例
   - 建立自动化回归测试套件

2. **性能基准**：
   - 与 OpenFOAM、Palabos、LBM 商业软件对比
   - 建立性能基准（速度、内存、精度）

3. **不确定性量化**：
   - 引入网格收敛性分析（GCI）
   - 建立不确定性量化框架

---

## 8. 结论

本验证框架成功完成了三个经典案例的验证：

1. ✅ **Poiseuille 流**：L2 误差 0.44%，最大误差 0.46%，稳态精度优秀
2. ✅ **Couette 流**：L2 误差 0.012%，最大误差 0.018%，边界条件精度极高
3. ✅ **Taylor-Green 涡**：衰减率误差 0.01%，R² = 1.0，瞬态精度优秀

**总体判定**：✅ PASS

**红队测试**：✅ PASS（坏值注入全部被正确识别）

**可复现性**：✅ 单脚本即可复现全部结果（validation/m583_cases/run_all_cases.py）

**计算效率**：三案例总耗时 22.9 秒（torch.compile 加速）

本验证框架为 TensorLBM 的可靠性提供了坚实基础，后续将扩展到三维、湍流和复杂几何。

---

## 附录：文件清单

```
validation/m583_cases/
├─ run_all_cases.py              # 主脚本（754 行）
├─ run.log                       # 运行日志
├─ summary.json                  # 汇总结果
├─ gates.json                    # 门禁评估
├─ poiseuille_2d/
│  └─ result.json                # Poiseuille 结果
├─ couette_2d/
│  └─ result.json                # Couette 结果
├─ taylor_green_2d/
│  ├─ result.json                # Taylor-Green 结果
│  └─ energy_history.csv         # 能量历史
├─ figures/
│  ├─ poiseuille_profile.png     # Poiseuille 速度剖面
│  ├─ poiseuille_error.png       # Poiseuille 误差
│  ├─ couette_profile.png        # Couette 速度剖面
│  ├─ couette_error.png          # Couette 误差
│  ├─ tg_energy_decay.png        # Taylor-Green 能量衰减
│  └─ tg_decay_rates.png         # Taylor-Green 衰减率
└─ redteam/
   ├─ redteam_result.json        # 红队测试结果
   └─ redteam_evidence.png       # 红队证据图
```

---

**报告版本**：v1.0  
**完成日期**：2026-10-30  
**Mission ID**：m583/m606  
**验证状态**：✅ PASS
