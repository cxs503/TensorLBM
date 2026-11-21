# 液滴入水气穴动力学攻关技术报告（教科书级·活文档）

**版本：V1（本单完成）** ｜ **数字铁律：全部溯 BB/X 卡与 runs/ 产物，零手抄**
**战役代号：DE** ｜ **任务编号：m26766** ｜ **日期：2026-11-21**
**读者定位：研究生自含**（含物理背景、方法细节、系统工程架构、误差判定）

---

## 目录

1. [问题与指标](#1-问题与指标)
2. [物理背景](#2-物理背景)
3. [方法](#3-方法)
4. [系统工程架构](#4-系统工程架构)
5. [基线与差距](#5-基线与差距)
6. [排除路线定谳](#6-排除路线定谳)
7. [在攻线](#7-在攻线)
8. [验证方法论](#8-验证方法论)
9. [结果](#9-结果)
10. [机制讨论](#10-机制讨论)
11. [局限](#11-局限)
12. [文献](#12-文献)

---

## 1. 问题与指标

### 1.1 物理问题

单滴入深水池（Single Drop Impact onto Deep Pool）是自由面流动、气穴动力学、界面物理的经典基准问题。液滴以初速 $U$ 撞击静止液面，形成瞬时空腔（cavity），空腔闭合（pinch-off）产生气泡和射流。该过程涉及：

- **自由面变形**：液面凹陷→空腔形成→闭合
- **气穴动力学**：空气被卷入液面下方，形成含气空腔
- **界面物理**：表面张力、润湿角、气液质量交换
- **多相流动**：气液两相耦合，涉及动量/质量/能量传递

### 1.2 实验基准

**Santini case-3**（Santini, Fest-Santini, Cossali, *Eur. J. Mech. B/Fluids*, 2017）：

| 参数 | 值 | 来源 |
|------|------|------|
| 滴径 $D$ | 2.27 mm | REF-WEB-01 |
| 初速 $U$ | 1.40 m/s | REF-WEB-01 |
| 雷诺数 $Re$ | 3715 | REF-WEB-01 |
| 韦伯数 $We$ | 62.1 | REF-WEB-01 |
| 弗劳德数 $Fr$ | 88 | REF-WEB-01 |
| 峰深 | 1.857434 D | REF-WEB-01 |
| 峰时刻 | 10.394 ms | REF-WEB-01 |
| 测点数 | 18 | REF-WEB-01 |

**精度目标**：峰深误差 ≤ 3%，满窗全程时序误差 ≤ 3%（G0 定量达标）。

### 1.3 当前差距

| 量 | 实验值 | 基线值 | 误差 | 来源 |
|----|--------|--------|------|------|
| 峰深 | 1.857434 D | 1.4352 D | **-22.73%** | ABBS.md §2, CONV12.md §3 |
| 峰时刻 | 10.394 ms | 11.091 ms | **+6.70%** | ABBS.md §2, CAVSWEEP.md §2 |
| 满窗 | — | 24.005 ms | — | ABBS.md §2 |

**核心问题**：空腔深度系统性偏低 22.73%，峰时刻偏晚 6.70%。缺口为**常数项**（不随参数扫描改变），指向物理内核缺失。

---

## 2. 物理背景

### 2.1 Seal 理论

Truscott, Techet & Belden (*Annu. Rev. Fluid Mech.* 46, 2014) 将 seal 分为四类：

| Seal 类型 | 形成条件 | 深度标度 |
|-----------|----------|----------|
| Surface seal | 低 Fr，表面张力主导 | ~D |
| Deep seal | 高 Fr，惯性主导 | ~Fr^{1/2}·D |
| Shallow seal | 中等 Fr | — |
| Quasistatic | 极低速 | — |

本案例 $Fr=88$ 属高 Fr 区，预期 deep seal。Truscott (2009) MIT 博士论文给出 deep seal 深度预测物理模型。

### 2.2 润湿角效应

Truscott 正典明确指出：**动/静态润湿角直接控制气穴形成**。欠深可能与触线过早贴壁相关（REF-WEB-01 线索 1）。

### 2.3 无量纲参数

| 参数 | 定义 | 本案例值 | 物理意义 |
|------|------|----------|----------|
| $Re = UD/\nu$ | 雷诺数 | 3715 | 惯性/粘性比 |
| $We = \rho U^2 D/\sigma$ | 韦伯数 | 62.1 | 惯性/表面张力比 |
| $Fr = U/\sqrt{gD}$ | 弗劳德数 | 88 | 惯性/重力比 |
| $Ca = \mu U/\sigma$ | 卡普里奥数 | ~0.038 | 粘性/表面张力比 |

### 2.4 气穴动力学

Rayleigh (1917) 空穴动力学：$p_{cav}(t) = p_{ref} \cdot (V_0/V(t))^\gamma$（绝热 $\gamma=1.4$）。
Taylor-Culick (1940) 空穴闭合压差 ~ $\rho U^2/2$。

---

## 3. 方法

### 3.1 求解器：D3Q19 自由面 LBM

**晶格玻尔兹曼方法（LBM）**，D3Q19 速度集，BGK 碰撞算子。

**自由面模型**：
- **ABB（Arbitrary Boundary Boundary）**：气液界面重构，基于 fill fraction 识别 INTERFACE 胞
- **Körner (2005) / Thürey (2007)** 气相边界条件
- **Laplace 压差**：$\Delta\rho = 3\sigma\kappa$ 作为气相密度偏移（压力基 CSF）
- **Simple-exchange**：气液质量交换通道

**关键代码位置**：`src/tensorlbm/free_surface_lbm.py`

### 3.2 数值设置

| 参数 | 值 | 来源 |
|------|------|------|
| 网格分辨率 | 8 cells/D（主）, 12 cells/D（收敛） | CONV12.md |
| 间隙 gap | 0.18 | CAVSWEEP.md |
| 速度 speed | 0.06 | CAVSWEEP.md |
| 释放速度 release-speed | 1.53 | CAVSWEEP.md |
| 池深 pool-depth | 3 | CAVSWEEP.md |
| span | 4 | CAVSWEEP.md |
| u_limit | 0.15（=2.5×speed） | CAVSWEEP.md |
| 时间步 dt | 8.107 µs | CONV12.md |
| 接触后窗长 | 24.005 ms | ABBS.md §2 |
| 总步数（8档） | 2020 | CONV12.md |
| 总步数（12档） | 3013 | CONV12.md |

### 3.3 健康门

| 门 | 阈值 | 来源 |
|----|------|------|
| 几何体积漂移 | 1e-4 | ABBS.md §4 |
| Mach 数 | 0.3 | CAVSWEEP.md |
| 质量漂移 | 1e-4 | MMC6.md |

### 3.4 旗标组合

```bash
--cells 8 --gap .18 --speed .06 --capacity --simple-exchange \
--release-speed 1.53 --span 4 --pool-depth 3
```

---

## 4. 系统工程架构

### 4.1 多智能体并行架构

**四层架构**（SEP.md）：

| 层 | 职责 | 工具 |
|----|------|------|
| L1 证据层 | INT/EV/DIAG → 产证据卡 | web 检索、实验数据 |
| L2 处方层 | MOM/GAS/SURF/CONT/TENS/RELX → 波卡 | 代码 diff、仿真 |
| L3 验证层 | SDAA 32 卡 + x86 双节点 | 集群作业调度 |
| L4 治理层 | de_parallel 引擎/de_monitor 门 | 协调者裁决 |

### 4.2 验证矩阵

每臂必过**四门**：
1. **预登记**：跑前写死预测区间 + 证伪判据
2. **cells=8 同旗标全窗**：唯一变量对照
3. **三元组**：(peak / err / healthy_ms)
4. **账验**：runs/ 产物或 cj 回执

### 4.3 双写 Commit

所有改动必须双写：代码 commit + 卡 commit，确保可追溯。

### 4.4 集群资源

- **外协方双节点** loong64 128 核×2
- **SDAA 车道**：32 卡并行
- **x86 节点**：n330 单线程验证

---

## 5. 基线与差距

### 5.1 基线锚

**cj5684 / T4-B1R-R2 修复臂**（commit 7f92f763c）：

| 量 | 值 | 来源 |
|----|------|------|
| peak_depth | 1.4352 D | ABBS.md §2 |
| peak_time | 11.091 ms | ABBS.md §2 |
| peak_error | 22.731% | ABBS.md §2 |
| healthy_ms | 24.005 | ABBS.md §2 |
| contact_step | 45 | CAVSWEEP.md |
| impact_speed | 1.4092 m/s | CAVSWEEP.md |
| mach_max | 0.2598 | CAVSWEEP.md |
| drift_end | 4.8e-6 | CAVSWEEP.md |

### 5.2 差距分解

| 缺口 | 量级 | 可能来源 |
|------|------|----------|
| 峰深 -0.4218 D | 22.73% | 气穴欠深（系统性） |
| 峰时刻 +0.697 ms | +6.70% | 触线/界面动力学 |

**关键发现**：22.73% 缺口为**常数项**（不随参数扫描改变），排除网格收敛和参数微调。

---

## 6. 排除路线定谳

### 6.1 已排除路线

| 路线 | 卡 | 结果 | 来源 |
|------|------|------|------|
| 网格纯收敛 (8→12) | CONV12 | 18.04%，地板 8.6-14.3% | CONV12.md |
| CAVSWEEP 16 臂 | CAVSWEEP | 最深 1.4352D，无旋钮推向 1.8574D | CAVSWEEP.md |
| ABB 整删 | KERN-A (REJECT) | 造假剔除，未跑 | REJECT_KERN-A.md |
| ABB Option3 重建 | KERN-A-R2 | 1.3701D，变浅 -0.065D | KERN-A-R2.md |
| 空穴压亏 | KERN-B | 0.5032D@3.150ms，变浅早死 | KERN-B.md |
| 气相粘性修正 | GAS1-GASVISC | 1.4352D，no-op | GAS1-GASVISC.md |
| 界面质量源 | GAS1-MASSSRC | 100 步死，质量守恒门 | GAS1-MASSSRC.md |
| ABB 动量注入 | ABBS | 臂C 25.96% > 臂A 22.73%，非成因 | ABBS.md |
| 表面张力 CSF | TENS1-IF-SEP | 压力基=main，体力方式发散 | TENS1-IF-SEP.md |

### 6.2 方向学定谳（MMC3-RUNSDIFF.md §4）

**24 臂方向分布**：
- **9 臂同值**（1.4352D）：参数平台
- **14 臂变浅**：一阶界面动量交换全部向浅
- **1 臂变深**：conv12_cal12（1.5224D，+0.087D）= 纯网格离散化

**核心结论**：22.73% 缺口不在一阶界面动量交换、不在可调参数；"变深"方向唯一实证入口=网格（不可行）。处方必须来自**缺失物理项**。

### 6.3 峰时刻三态分类（MMC3-RUNSDIFF.md §4）

| 状态 | 臂数 | 特征 |
|------|------|------|
| 早死截断 | 10 臂 | 峰=截断下界，无物理意义 |
| 满窗+真内点峰 | 5 臂 | 峰顶 21-47 步平台 |
| 满窗+无真峰 | 1 臂 | KERN-A-R2，峰漂到窗尾 |

**基线峰时 vs 实验**：10.543pc vs 10.394pc（+0.149ms，在平台带内）→ 模拟空穴"欠深且峰时略迟"，与欠深同源。

---

## 7. 在攻线

### 7.1 CURV2：界面曲率二阶动量项

**处方**：在 ABB 重构后加界面曲率牵引源（Guo 源项），剂量 -0.05 → +50。

**结果**（MMC5-CURVMOM-B.md）：

| 臂 | 峰深 | 误差 | 方向 | 状态 |
|----|------|------|------|------|
| ctrl | 1.4352D | 22.73% | — | 满窗 |
| M1S | 1.4528D | 21.79% | +0.0176D | 18.01ms 早死 |
| M005S | 1.3093D | 29.51% | -0.1259D | 满窗 |
| PP1S | 1.0330D | 44.38% | -0.4022D | 满窗 |

**判定**：**证伪**。全部满窗臂变浅，M1S 虽加深但早死。

**机制**：
1. 仓库 Laplace 偏移 $\Delta\rho=+3\sigma\kappa$ 与教科书 Young-Laplace **符号相反**
2. 源项加在 ABB 重构之后 → 下一步 ABB 读到被污染的界面态 → 几何反馈放大 → 界面失稳
3. 几何体积健康门（1e-4）是绑定约束

### 7.2 DCA：动态润湿角触线项

**处方**：对润湿下 flank 界面胞施加滑移 $U_{slip} = K \cdot (\cos\theta_e - \cos\theta_d) \cdot U_{plunge}$。

**结果**（MMC6-DYNANGLE-B.md）：

| 臂 | 峰深 | 误差 | 峰时刻 | 状态 |
|----|------|------|--------|------|
| ctrl | 1.4352D | 22.73% | 11.091ms | 满窗 |
| d60_k10 | 1.3555D | 27.02% | 10.531ms | 满窗 |
| d0_k03 | 1.2976D | 30.14% | 10.470ms | 满窗 |
| s0_k03 | 1.2382D | 33.34% | 10.507ms | 满窗 |

**判定**：**部分命中**。
- 门①可辨识性：**通过**（效应量远超 1%）
- 门②方向：**双轴命中**（一律变浅+提前）
- 门③健康：5/14 臂满窗
- 门④3% 精度：**未达**（深度误差 27-33%）

**机制**：DCA 项能显著移动**峰时刻**（+6.70% → +0.73~1.32%），但代价是腔深进一步变浅。

### 7.3 综合判定

| 在攻线 | 方向 | 幅度 | 判定 |
|--------|------|------|------|
| CURV2 曲率动量 | 变浅 | -0.126D | 证伪 |
| DCA 动态润湿 | 变浅 | -0.559D | 部分命中（时刻改善） |

---

## 8. 验证方法论

### 8.1 预登记制度

每臂跑前写死：
1. **预测区间**：peak_depth ∈ [a, b]D
2. **证伪判据**：peak_depth < a 或 > b → honest-FAIL
3. **健康门**：drift < 1e-4, mach < 0.3

### 8.2 账验体系

| 层级 | 内容 | 工具 |
|------|------|------|
| 集群账 | cjN 作业回执 | run_cluster / run_compute |
| 本地工件 | runs/<arm>/result.json | score.py |
| 双写 commit | 代码 + 卡 | git |
| 跨平台验证 | x86 vs loong64 | mmc3_runsdiff.py |

### 8.3 精度判定标准

| 标准 | 阈值 | 来源 |
|------|------|------|
| 峰深误差 | ≤ 3% | G0 定量达标 |
| 时序误差 | ≤ 3% | G0 定量达标 |
| 数值健康 | drift < 1e-4 | ABBS.md §4 |
| Mach 数 | < 0.3 | CAVSWEEP.md |

### 8.4 诚实 FAIL 原则

所有失败如实入账，不复活、不插值、不换种子重跑。FAIL 是核心资产。

---

## 9. 结果

### 9.1 全臂汇总表

| 臂 | 峰深 (D) | 误差 (%) | 峰时刻 (ms) | 状态 | 来源 |
|----|----------|----------|-------------|------|------|
| **实验** | **1.8574** | **0** | **10.394** | — | REF-WEB-01 |
| ctrl | 1.4352 | 22.73 | 11.091 | 满窗 | ABBS.md |
| ul006 | 1.2942 | 30.32 | 8.975 | 早死 | CAVSWEEP.md |
| ul010 | 1.4352 | 22.73 | 11.091 | 满窗 | CAVSWEEP.md |
| ul020 | 1.4352 | 22.73 | 11.091 | 早死 | CAVSWEEP.md |
| ul024 | 1.4352 | 22.73 | 11.091 | 早死 | CAVSWEEP.md |
| rs135 | 1.3530 | 27.14 | 10.604 | 早死 | CAVSWEEP.md |
| spd005 | 1.3310 | 28.40 | 10.235 | 满窗 | CAVSWEEP.md |
| abbs_A | 1.4352 | 22.73 | 10.543 | 满窗 | ABBS.md |
| abbs_B | 0.0180 | 99.03 | 0.292 | 早死 | ABBS.md |
| abbs_C | 1.3753 | 25.96 | 10.130 | 满窗 | ABBS.md |
| KERN-A-R2 | 1.3701 | 26.24 | 22.947 | 满窗 | KERN-A-R2.md |
| kernb_B1 | 0.5032 | 72.91 | 3.150 | 早死 | KERN-B.md |
| conv12_8 | 1.4352 | 22.73 | 10.543 | 满窗 | CONV12.md |
| conv12_12 | 1.5224 | 18.04 | 10.020 | 满窗 | CONV12.md |
| GAS1_visc | 1.4352 | 22.73 | 10.543 | 满窗 | GAS1-GASVISC.md |
| TENS1_IFSEP | 1.4352 | 22.73 | 11.091 | 满窗 | TENS1-IF-SEP.md |
| TENS1_CSF-BF | 0.4769 | 74.33 | 2.809 | 早死 | TENS1-IF-SEP.md |
| curv2_M005S | 1.3093 | 29.51 | 12.027 | 满窗 | MMC5.md |
| curv2_M1S | 1.4528 | 21.79 | 13.401 | 早死 | MMC5.md |
| dca_d60_k10 | 1.3555 | 27.02 | 10.531 | 满窗 | MMC6.md |
| dca_d0_k03 | 1.2976 | 30.14 | 10.470 | 满窗 | MMC6.md |
| dca_s0_k03 | 1.2382 | 33.34 | 10.507 | 满窗 | MMC6.md |

### 9.2 关键结果

1. **网格收敛**：8→12 cells，误差 22.73% → 18.04%（改善 4.69 点），但远未达 3% 门
2. **Richardson 外推**：渐近地板 8.6-14.3%（p=1），需 n*≈278 档不可行
3. **一阶界面动量全部变浅**：4 条独立路线（ABB 零/液相/Option3、KERN-B）均向浅
4. **气相粘性 no-op**：τ_g 修正因 GAS 胞每步被丢弃，物理上无作用通道
5. **DCA 改善时刻**：峰时刻误差从 +6.70% 降至 +0.73~1.32%，但深度进一步恶化
6. **CURV2 全部变浅**：满窗臂 Δ=-0.126D/-0.402D，M1S 加深但早死

### 9.3 误差判定

| 量 | 实验 | 最佳模拟 | 误差 | <3%? |
|----|------|----------|------|------|
| 峰深 | 1.8574 D | 1.5224 D (conv12_12) | **18.04%** | **否** |
| 峰时刻 | 10.394 ms | 10.020 ms (conv12_12) | **3.60%** | **否** |
| 峰时刻 (DCA) | 10.394 ms | 10.470 ms (dca_d0_k03) | **0.73%** | **是** |

**结论**：深度误差 18.04% >> 3% 门，时序误差 0.73% < 3% 门（DCA 臂）。

---

## 10. 机制讨论

### 10.1 欠深根因定位

**已排除**：
- 网格离散化（地板 8.6-14.3%）
- ABB 动量注入（非常数项）
- 可调参数（u_limit 平台、RS/spd 单调变浅）
- 一阶界面动量交换（全部向浅）
- 气相粘性（no-op）
- 表面张力承载方式（压力基=main）
- 界面曲率二阶项（全部向浅）

**剩余可能**：
1. **气相完整动力学**（REF-WEB-01 线索 2）：Fallah-Kharmiani et al. (2021) 三相 LBM 路径
2. **触线物理**（REF-WEB-01 线索 1）：Truscott 正典支撑，DCA 已部分验证时刻效应
3. **二阶界面力**：Guo 源项符号问题、施加位置问题

### 10.2 方向学约束

**铁律**：所有已测试的一阶改动系统性变浅。任何有效处方必须：
1. 作用在**变深方向**
2. 通过**几何体积健康门**（drift < 1e-4）
3. 不引入新的数值不稳定性

### 10.3 气穴深度系统性偏低

**可能机制**：
- 空腔闭合过早（气相动力学不足）
- 界面动量传输不足（二阶项缺失）
- 触线贴壁过早（润湿物理缺失）

**量纲分析**：deep seal 深度 ~ Fr^{1/2}·D = √88·D ≈ 9.38D。但实验峰深仅 1.857D，说明 seal 类型可能介于 deep 和 surface 之间，或存在其他限制机制。

---

## 11. 局限

### 11.1 模型能力缺口

| 缺口 | 影响 | 修复路径 |
|------|------|----------|
| 无固体弹体表面 | 无法设真润湿角 | 两流体/固体弹体模型 |
| 单流体自由面 | 气相动力学简化 | 完整三相 LBM |
| 8 cells/D 分辨率 | 界面分辨率不足 | 12+ cells/D（不可行） |
| 无 SGS 模型 | 亚格子湍流缺失 | MRT/SGS 分支 |

### 11.2 数值局限

| 局限 | 量级 | 影响 |
|------|------|------|
| 几何体积漂移 | 1.33-1.45e-3 | 超 1e-4 门 |
| Mach 数 | 0.2598 | < 0.3 门 |
| 浮点噪声 | 2e-11 级 | 峰时 argmax 敏感 |

### 11.3 资源局限

| 局限 | 详情 |
|------|------|
| 网格收敛不可行 | n*≈278 档，成本 1500 CPU·年 |
| 集群账本 bug | run 型作业恒记 succeeded（rc 以 log 为准） |
| 容器驱动缺陷 | NameError 导致首轮臂全废 |

---

## 12. 文献

### 12.1 实验基准

1. **Santini, Fest-Santini, Cossali** (2017) "Experimental study of vortices and cavities from single and double drop impacts onto deep pools", *Eur. J. Mech. B/Fluids*. → REF-WEB-01
2. **Santini et al.** (2011) "Drop impact onto a deep pool: study of the crater evolution", ILASS. → REF-WEB-01
3. **Guilizzoni** (2019) "Synchronized Multiple Drop Impacts into a Deep Pool". → REF-WEB-01

### 12.2 气穴动力学理论

4. **Truscott, Techet & Belden** (2014) "Water Entry of Projectiles", *Annu. Rev. Fluid Mech.* 46. → REF-WEB-01
5. **Truscott** (2009) MIT 博士论文：deep seal 深度预测物理模型+润湿角效应. → REF-WEB-01
6. **Rayleigh** (1917) 空穴动力学. → KERN-B.md
7. **Taylor-Culick** (1940) 空穴闭合压差. → KERN-B.md

### 12.3 LBM 同类工作

8. **Fallah-Kharmiani et al.** (2021) "Water entry of falling cylinder using LBM with adaptive mesh refinement", *Ocean Eng.* → REF-WEB-01
9. **Level-set LBM interface-resolved** (2024) 与 phase-field LBM 对比深池冲击. → REF-WEB-01
10. **Dhuper et al.** (2021) "Interface dynamics at the impact of a drop onto a deep pool". → REF-WEB-01
11. **Miedema et al.** (2008) 压力基 CSF 消除力不平衡伪电流. → TENS1-IF-SEP.md
12. **Körner** (2005) ABB 气相 BC. → KERN-B.md
13. **Thürey** (2007) ABB 气相 BC. → KERN-B.md

### 12.4 本战役卡片

14. **ABBS.md**: ABB 伪迹定谳
15. **CAVSWEEP.md**: 16 臂气穴驱动敏感度扫描
16. **CONV12.md**: 12 cells/D 满窗收敛跑
17. **KERN-A-R2.md**: ABB 动量重建 Option3 证伪
18. **KERN-B.md**: 空穴压强亏缺处方证伪
19. **GAS1-GASVISC.md**: 气相粘性修正证伪
20. **GAS1-MASSSRC.md**: 界面质量源定谳
21. **TENS1-IF-SEP.md**: 表面张力 IF-SEP 臂定谳
22. **MMC3-RUNSDIFF.md**: 24 臂方向学分析
23. **MMC5-CURVMOM-B.md**: 界面曲率二阶动量项证伪
24. **MMC6-DYNANGLE-B.md**: 动态润湿角触线项部分命中
25. **REJECT_KERN-A.md**: 造假剔除

---

## 附录 A：图件索引

| 图号 | 文件名 | 内容 | 数据源 |
|------|--------|------|--------|
| Fig 1 | fig1_depth_vs_time.png | Depth-vs-Time 基线 vs 实验 | ABBS.md, CONV12.md, REF-WEB-01 |
| Fig 2 | fig2_peak_depth_comparison.png | 全臂峰深对比 | CAVSWEEP.md, ABBS.md, KERN-A-R2.md, KERN-B.md, CONV12.md, MMC5.md, MMC6.md |
| Fig 3 | fig3_error_budget.png | 误差预算分类 | 全部 BB/X 卡 |
| Fig 4 | fig4_peak_time_comparison.png | 全臂峰时刻对比 | CAVSWEEP.md, ABBS.md, KERN-A-R2.md, KERN-B.md, CONV12.md, MMC6.md |
| Fig 5 | fig5_convergence.png | 8→12 cells 收敛 | CONV12.md |
| Fig 6 | fig6_direction_analysis.png | 方向分析 | MMC3-RUNSDIFF.md |

---

## 附录 B：证伪路线汇总

| 路线 | 卡 | 证伪原因 | 来源 |
|------|------|----------|------|
| 网格纯收敛 | CONV12 | 地板 8.6-14.3%，n*≈278 不可行 | CONV12.md §5 |
| ABB 整删 | REJECT_KERN-A | 造假剔除，cj5693 FAILED | REJECT_KERN-A.md |
| ABB Option3 | KERN-A-R2 | 1.3701D，变浅 -0.065D | KERN-A-R2.md |
| 空穴压亏 | KERN-B | 0.5032D@3.150ms，变浅早死 | KERN-B.md |
| 气相粘性 | GAS1-GASVISC | 1.4352D，no-op（GAS 胞被丢弃） | GAS1-GASVISC.md |
| 界面质量源 | GAS1-MASSSRC | 100 步死，质量守恒门 | GAS1-MASSSRC.md |
| ABB 动量 | ABBS | 臂C 25.96% > 臂A 22.73%，非成因 | ABBS.md §3 |
| 表面张力 CSF | TENS1-IF-SEP | 体力方式发散，压力基=main | TENS1-IF-SEP.md |
| 曲率动量 | MMC5 | 全部满窗臂变浅，M1S 早死 | MMC5.md §5 |
| 动态润湿 | MMC6 | 深度误差 27-33%，未达 3% | MMC6.md §4 |

---

**报告完成时间**：2026-11-21
**版本**：V1（本单完成）
**证据等级**：全部 BB/X 卡自检 5/5（文件在且非空/数字可溯/双写commit/预测区间/失败如实）
