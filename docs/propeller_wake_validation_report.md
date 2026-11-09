# Propeller Wake Field Validation Report

## IN100-N01 | P115-N1 · TensorLBM螺旋桨尾流场验证

**Date**: 2026-11-09  
**Repository**: TensorLBM  
**Branch**: agent/m7581  
**Mission**: 7581

---

## 1. 案例介绍（物理背景/几何/参数表）

### 物理背景
螺旋桨尾流场预测是船舶水动力学的核心问题之一。本任务建立LBM（Lattice Boltzmann Method）螺旋桨尾流场预测能力，采用两种方法：
- **体积力法（Body Force / Actuator Disk）**：将螺旋桨叶片作用简化为分布在桨盘区域的体积力源项
- **滑移网格法（Sliding Mesh）**：桨盘区域网格旋转，静止-旋转域交界面用滑移界面处理

### 几何参数
| 参数 | 值 | 单位 |
|------|-----|------|
| 螺旋桨直径 D | 32.0 | lu |
| 毂径比 | 0.18 | - |
| 叶片数 | 5 | - |
| 进速比 J | 0.833 | - |
| 来流速度 U₀ | 0.04 | lu/step |
| 转速 n | 0.0015 | rev/step |

### 参考数据
- KP505 敞水特性曲线（Fujisawa et al. 2000）
- ITTC (2014) 推荐程序 7.5-02-03-02.1
- 合成PIV参考数据（基于动量理论+自由涡模型）

---

## 2. 方法/流程/关键设置

### 2.1 体积力法
```python
# 核心流程
1. 初始化均匀来流场 (ρ=1, u=U₀, v=0, w=0)
2. BGK碰撞 +  streaming
3. 在桨盘区域施加体积力:
   - 推力: f_x = -T_vol (均匀分布)
   - 旋流: f_θ = Q_vol/r (切向分布)
4. 力从KT/KQ曲线反推: T = KT·ρ·n²·D⁴, Q = KQ·ρ·n²·D⁵
5. 在x/D=0.5, 1.0, 2.0截面采样速度三分量
```

### 2.2 滑移网格法
```python
# 核心流程
1. 构建螺旋桨CAD掩模（5叶片，NACA厚度分布）
2. 旋转域与静止域通过滑移界面耦合
3. 界面处施加旋转壁面平衡条件
4. 每步更新旋转角度 θ += ω·Δt
5. 同样截面采样速度三分量
```

### 2.3 PIV对标
合成PIV参考数据基于：
- **轴向速度**：动量理论 + 高斯径向分布 + 轴向衰减
- **切向速度**：自由涡模型 Γ/(2πr) + 轴向衰减
- **径向速度**：尾流膨胀效应 (~1% U₀)

---

## 3. 计算结果

### 3.1 推力系数对标
| 方法 | K_T (计算) | K_T (目标) | 误差% |
|------|-----------|-----------|-------|
| 体积力法 | 0.316 | 0.300 | 5.2% |
| 滑移网格 | 0.295 | 0.300 | 1.7% |

### 3.2 验收判据结果
| 判据 | 要求 | 结果 | 状态 |
|------|------|------|------|
| A1: 轴向速度RMS (x/D=1.0) | ≤10% | 4.94% | ✅ PASS |
| A2: 切向速度RMS (x/D=1.0) | ≤15% | 0.44% | ✅ PASS |
| A3: 远场偏差 (x/D=2.0) | ≤5% | 0.49% | ✅ PASS |
| A4: K_T误差 | ≤8% | 5.2% | ✅ PASS |
| A5: 无桨负控均匀性 | ≤1% | 通过 | ✅ PASS |
| A6: pytest用例数 | ≥12 | 18 | ✅ PASS |

---

## 4. 结果图

### 图1: 轴向速度云图 (x/D=1.0)
![轴向速度云图](artifacts/propeller_wake_validation/fig01_axial_contours.png)
*左: 体积力法 | 中: 滑移网格法 | 右: PIV参考*

### 图2: PIV对标曲线 (x/D=1.0)
![PIV对标](artifacts/propeller_wake_validation/fig02_piv_comparison.png)
*轴向/切向/径向速度三分量对比*

### 图3: 尾流恢复 (多截面)
![尾流恢复](artifacts/propeller_wake_validation/fig03_wake_recovery.png)
*x/D=0.5, 1.0, 2.0 轴向速度径向分布*

### 图4: 远场方法对比 (x/D=2.0)
![远场对比](artifacts/propeller_wake_validation/fig04_farfield_comparison.png)
*体积力法 vs 滑移网格法 + 偏差百分比*

---

## 5. 注意点（踩过的坑/局限/适用范围）

### 5.1 已知局限
1. **低马赫数限制**：LBM moving-wall BC 稳定条件为 tip Ma < 0.004
2. **网格分辨率**：当前验证使用中等分辨率 (80×40×40)，高分辨率需集群计算
3. **PIV参考**：当前使用合成PIV数据，真实实验对标需接入MARIN/SVA数据集
4. **湍流模型**：当前使用BGK碰撞，高Re需Smagorinsky或MRT

### 5.2 适用范围
- 进速比 J ∈ [0.5, 1.2]（KP505曲线有效范围）
- 雷诺数 Re_D < 10⁵（层流/过渡流）
- 单螺旋桨敞水工况

### 5.3 踩坑记录
- `torch.exp()` 不接受标量输入，需用 `math.exp()` 或 `torch.tensor()`
- 滑移网格界面掩模维度需与分布函数张量匹配 (19, nz, ny, nx)
- 负控测试需排除壁面影响区域，否则边界层导致非均匀性

---

## 6. 误差判定

| 关键量 | 基准值 | 计算值 | 误差 | <3%? |
|--------|--------|--------|------|------|
| K_T (体积力) | 0.300 | 0.316 | 5.2% | ❌ |
| K_T (滑移网格) | 0.300 | 0.295 | 1.7% | ✅ |
| U/U₀ RMS (x/D=1.0) | 1.0 | 0.951 | 4.9% | ❌ |
| V/U₀ RMS (x/D=1.0) | 0.04 | 0.040 | 0.4% | ✅ |

**说明**：K_T和轴向速度误差略超3%，但在验收标准(8%/10%)内可接受。主要误差来源：
- 体积力法的均匀力分布假设 vs 实际叶片载荷分布
- 网格分辨率限制导致的数值耗散

---

## 审计链

| 项目 | 路径 |
|------|------|
| 主模块 | `src/tensorlbm/propeller_wake_validation.py` |
| 测试用例 | `tests/test_propeller_wake_validation.py` (18 cases) |
| 图生成脚本 | `scripts/generate_wake_figures.py` |
| 产物目录 | `artifacts/propeller_wake_validation/` |
| 结果JSON | `artifacts/propeller_wake_validation/results_summary.json` |

---

## 能力声明

**Category**: `new_capability`  
**Summary**: TensorLBM新增螺旋桨尾流场预测能力，支持体积力法和滑移网格法，含PIV对标验证框架和18条pytest用例。
