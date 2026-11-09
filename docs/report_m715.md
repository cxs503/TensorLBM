# Mission 715 补报：冰区操纵性数学模型 (MMG-type in ice)

> **补报说明**：本单为 mission 715 的补报（原单已交付但缺论文级报告）。原单工件目录未留档，
> 以下数字均从现存代码与可复现的 pytest 运行中程序化提取，零手抄。
> 无实验基准数据——参考值为模型校准值，已在文中显式声明。

---

## 1. 案例介绍

### 1.1 物理背景

极地船舶在冰区航行时，冰对船体的附加阻力与对舵/螺旋桨的干扰显著改变操纵性特征。
传统 MMG（Maneuvering Modeling Group）框架仅涵盖开阔水域，需扩展以包含：
- **冰阻力附加项**：冰对船体水线附近的挤压与摩擦
- **舵力冰修正**：冰浓度对舵处来流与舵效的影响

本节点实现基于 MMG 框架的简化工程模型，为冰区航行战役提供操纵性预测能力。

### 1.2 船舶参数（典型极地运输船）

| 参数 | 符号 | 数值 | 单位 |
|------|------|------|------|
| 船长 | L | 130.0 | m |
| 船宽 | B | 24.0 | m |
| 吃水 | T | 8.0 | m |
| 型深 | D | 12.0 | m |
| 排水量 | Δ | 15,000 | t |
| 方形系数 | Cb | 0.72 | – |
| 重心高度 | KG | 5.5 | m |
| 重心纵向位置 | xG | 2.0 | m |
| 螺旋桨直径 | Dp | 5.5 | m |
| 转速 | n | 100 | rpm |
| 舵面积 | Ar | 18.0 | m² |
| 舵高 | Hr | 6.0 | m |
| 初始航速 | V | 5.0 | m/s (≈9.7 kn) |

### 1.3 冰环境参数

| 参数 | 数值 | 单位 |
|------|------|------|
| 冰厚 | 1.0 | m |
| 冰弯曲强度 | 500×10³ | Pa |
| 冰密度 | 900 | kg/m³ |
| 冰浓度 | 0.8 | – |

---

## 2. 方法与工具链

### 2.1 模型架构

实现文件：`backend/api/maneuvering_ice.py`（ship-performance-platform 仓）

**IceMMGModel 类**包含三个核心子模型：

#### (a) 冰阻力模型
```
R_ice = k_ice × B × h_i × u^1.5 × 1000  [N]
k_ice = 5.0 × c_ice
```
简化经验公式，冰阻力与速度的 1.5 次方成正比，与冰厚和冰浓度线性相关。

#### (b) 回转圈计算（Clarke 经验公式 + 冰区修正）
```
TD_open = L × (4.0 − 0.02δ + 0.5/(Ar_ratio × 10))
k_ice_turning = 1.0 + 0.8 × c_ice × h_i
TD_ice = TD_open × k_ice_turning
```
开阔水域回转直径基于 Clarke 回归公式，冰区修正因子使回转直径增大 50–100%。

#### (c) Z 形试验计算
```
overshoot_open = 6.0 + 0.2δ − 0.15ψ + 1.0/(Ar_ratio × 10)
k_ice_overshoot = 1.0 + 0.3 × c_ice × h_i
overshoot_ice = overshoot_open × k_ice_overshoot
T_period = 2.5 × L/V + 0.3δ
```
超越角经验公式含舵角/首向角/舵面积比修正，冰区使超越角增大 20–40%。

### 2.2 试验模拟

- **回转试验**：匀速回转假设，轨迹为圆弧，角速度 ω = V/(TD/2)
- **Z 形试验**：指数趋近 + 正弦振荡叠加，模拟首向角响应
- **数值方法**：解析经验公式（非数值积分），计算代价极低

### 2.3 测试框架

测试文件：`backend/tests/test_ice_maneuvering.py`

12 个 pytest 用例覆盖：
- 模型初始化与基本计算
- 冰阻力速度依赖性
- 回转试验战术直径验收
- Z 形试验超越角验收
- 冰区/开阔水域差异
- 冰厚/舵角参数影响趋势

---

## 3. 计算结果

### 3.1 验收判据达成（pytest 运行，2026-11-09 复现）

| 判据 | 计算值 | 参考值 | 相对误差 | 阈值 | 结果 |
|------|--------|--------|----------|------|------|
| zigzag_10_10_overshoot_error | 15.20° | 15.20° | 0.0000 | < 0.2 | ✅ PASS |
| turning_circle_tactical_diameter_error | 1319.5 m | 1320.0 m | 0.0004 | < 0.15 | ✅ PASS |
| pytest_test_ice_maneuvering | 12/12 pass | – | – | all pass | ✅ PASS |

> **注**：参考值为模型校准值（非实验数据），故误差极小属自洽性验证而非独立验证。

### 3.2 冰区效应量化

| 试验 | 开阔水域 | 冰区 (h=1.0m, c=0.8) | 差异 |
|------|----------|----------------------|------|
| Z 形超越角 (10°/10°) | 12.30° | 15.20° | +2.90° (+23.6%) |
| 回转战术直径 | 未留档（需单独运行） | 1319.5 m | 冰区增大约 64% |

### 3.3 参数趋势验证

| 测试项 | 预期趋势 | 实际结果 | 判定 |
|--------|----------|----------|------|
| 冰厚增加 → 回转直径增大 | TD(1.5m) > TD(1.0m) > TD(0.5m) | ✅ 满足 | PASS |
| 舵角增大 → 回转直径减小 | TD(15°) > TD(25°) > TD(35°) | ✅ 满足 | PASS |
| 冰区超越角 ≠ 开阔水域 | diff > 0 | diff = 2.90° | PASS |

### 3.4 K-T 指数

| 参数 | 数值 |
|------|------|
| K (增益指数) | 由超越角反算 |
| T (时间常数) | T_period / (2π) |

> 具体数值未在原单工件中留档，可从 API 响应中获取。

---

## 4. 结果图

原单生成了 6 张 PNG 图（位于 `artifacts/podded_ice_maneuvering/`），但属于后续吊舱推进冰区操纵性成熟化节点（podded_ice_maneuvering），非本节点直接产物。

本节点（基础 MMG 冰区操纵模型）未生成独立可视化图，原因：
1. 模型为经验公式直接计算，无场量分布
2. 轨迹为简单圆弧/正弦曲线，信息量低
3. 验收以数值判据为主，非视觉比对

如需可视化，可基于 API 返回的 trajectory 字段绘制回转轨迹与 Z 形首向响应曲线。

---

## 5. 注意点与局限

### 5.1 模型层级

本模型为 **L2 经验公式级**（非 L3 数值积分级）：
- 未求解 3-DOF 操纵运动方程（surge/sway/yaw 耦合）
- 回转试验为匀速圆弧假设，未考虑速度衰减动态过程
- Z 形试验为解析近似，非时间步进仿真

### 5.2 冰阻力模型

- 采用简化幂律公式 R_ice ∝ u^1.5，未区分破冰/子压/摩擦分量
- 未考虑冰脊、冰坎等离散特征
- 冰-船摩擦系数取隐含常数，未随温度/盐度变化

### 5.3 舵力冰修正

- 仅通过冰浓度影响来流速度，未建模冰对舵的直接冲击
- 未考虑舵叶结冰导致的剖面变化
- 高冰浓度下可能出现舵卡死，模型未覆盖

### 5.4 参考值来源

**关键声明**：验收参考值为模型校准值（self-consistency check），非独立实验或高保真 CFD 基准。
因此：
- 误差 < 0.2 仅证明代码实现与公式一致
- 不能证明模型预测精度
- 绝对精度标记为 **WITHHELD**（无实验对标）

### 5.5 适用范围

- 冰厚：0.5–1.5 m（公式外推未验证）
- 冰浓度：0–1（连续假设）
- 航速：2–8 m/s（低速冰区航行典型范围）
- 船型：排水型船，Cb = 0.6–0.8

---

## 6. 误差判定

### 6.1 关键量 vs 基准

| 关键量 | 计算值 | 基准值 | 基准来源 | 相对误差 | < 3%? |
|--------|--------|--------|----------|----------|-------|
| Z 形超越角 | 15.20° | 15.20° | 模型校准 | 0.00% | ✅ 是 |
| 战术直径 | 1319.5 m | 1320.0 m | 模型校准 | 0.04% | ✅ 是 |

### 6.2 显式回答

**是否 < 3%？** 是——但这是自洽性验证，非独立精度验证。

**无独立基准的原因声明**：
1. 冰区操纵性实验数据极度稀缺（仅少数冰池试验）
2. 本模型为简化工程模型，定位为保障代码正确性的快速评估工具
3. 绝对精度需待后续节点（如吊舱推进冰区操纵性成熟化）与冰池实验或高保真 CFD 对标

### 6.3 后续改进路径

1. 与 Lindqvist/IceRes 等冰阻力模型交叉验证
2. 引入 3-DOF 时间步进仿真（已在 podded_ice_maneuvering 节点实现）
3. 收集冰池实验数据建立独立基准
4. 与模型试验对标（如 KV Svalbard 实船试验数据）

---

## 附录 A：pytest 运行日志（节选）

```
backend/tests/test_ice_maneuvering.py::TestIceMMGModel::test_model_initialization PASSED
backend/tests/test_ice_maneuvering.py::TestIceMMGModel::test_ice_resistance_calculation PASSED
backend/tests/test_ice_maneuvering.py::TestIceMMGModel::test_ice_resistance_speed_dependence PASSED
backend/tests/test_ice_maneuvering.py::TestIceMMGModel::test_simulation_runs PASSED
backend/tests/test_ice_maneuvering.py::TestIceTurningCircle::test_turning_circle_simulation PASSED
backend/tests/test_ice_maneuvering.py::TestIceTurningCircle::test_turning_circle_tactical_diameter_error PASSED
backend/tests/test_ice_maneuvering.py::TestIceTurningCircle::test_ice_increases_turning_diameter PASSED
backend/tests/test_ice_maneuvering.py::TestIceZigzag::test_zigzag_simulation PASSED
backend/tests/test_ice_maneuvering.py::TestIceZigzag::test_zigzag_10_10_overshoot_error PASSED
backend/tests/test_ice_maneuvering.py::TestIceZigzag::test_ice_affects_overshoot PASSED
backend/tests/test_ice_maneuvering.py::TestIceModelPhysics::test_ice_thickness_effect PASSED
backend/tests/test_ice_maneuvering.py::TestIceModelPhysics::test_rudder_angle_effect PASSED

======================== 12 passed, 1 warning in 1.06s =========================
```

## 附录 B：工件路径

| 工件 | 路径 |
|------|------|
| 模型代码 | `backend/api/maneuvering_ice.py` |
| 测试代码 | `backend/tests/test_ice_maneuvering.py` |
| 测试结果 | pytest stdout（本补报复现） |
| 相关图表 | `artifacts/podded_ice_maneuvering/fig*.png`（后续节点产物） |

---

**报告状态**：补报完成（honest meta-report）  
**原单结论**：PASS（12/12 测试通过，验收判据满足）  
**精度声明**：自洽性验证通过，绝对精度 WITHHELD（无独立基准）  
**补报日期**：2026-11-09
