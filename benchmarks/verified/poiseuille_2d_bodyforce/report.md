# 【锚点 CASE-POISEUILLE-2D】二维平面泊肃叶层流（TensorLBM D2Q9 BGK）：解析抛物线 <2 % 验证 + Guo 强迫源项质量守恒缺陷修复

## ① 案例介绍（物理背景 / 几何 / 参数表）

两块无限大平行平板间、沿流向受恒定体积力 `g`（等效常压力梯度 `dp/dx = -ρg`）驱动的**充分发展二维层流**（plane Poiseuille / Hagen–Poiseuille 槽流）。
几何：流向 `x` 周期（`nx=4`），壁法向 `y ∈ [0, H]`，`H=32` 流体行；无滑移壁面。
控制方程与参考解（不可压 N–S 稳态精确解）：

```
ν u''(y) + g = 0 ,  u(0)=u(H)=0
u(y) = g/(2ν)·y(H−y) ,  u_max = gH²/(8ν) ,  ū = gH²/(12ν)
```

**参考解出处**：本报告不手抄任何数字——`verify_reference()` 在仓库内现场复算闭式解并做三重校验：ODE 内部残差 `3.18e-15`、无滑移边界 `u(0)=0.0e+00` / `u(H)=0.0e+00`、闭式不变量与剖面一致性（`u_max` 0.012800 vs 0.012800；`ū` 0.0085333 vs 0.0085333）。教科书页级引用（Hagen 1839 / Poiseuille 1841）**未能在沙箱内机器核对**（无外网），故证据等级按"闭式 + 仓库内三重校验"计，已在文档 §6 局限中如实标注。

| 参数 | 取值 | 说明 |
|---|---|---|
| 格子/碰撞 | D2Q9 / BGK | 单弛豫 ω=1/τ |
| 网格 | ny=36（H=32）× nx=4 | 流向周期、壁法向有界 |
| τ | 0.8 | 敏感性另测 0.6/0.8/1.0/1.2 |
| ν=(τ−1/2)/3 | 0.100000 | 格子单位 |
| g | 1e-05 | 单位质量体积力 |
| u_max / ū | 0.012788 / 0.008533 | 解析 |
| **Re = ūH/ν** | **2.73** | 层流 |
| **Ma = u_max/c_s** | **0.0221** | O(Ma²)≈0.05 % |
| 步数 | 11000 | 自适应稳态门 1e-6 |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **复用现有能力，不另起炉灶**：`tensorlbm.ice_hydro.lbm_channel.D2Q9Channel`（仓库自带 D2Q9 BGK numpy 内核）+ 无滑移半程反弹（壁位 `y=0.5` / `y=H+0.5`）+ 二阶 Guo (2002) 体积力强迫。参考剖面在同一壁位取样，**无修正因子、无拟合、无外推**。
2. **预登记判据**（动手前写入 `run.py: THRESHOLDS`，不因结果调整）：`eps_inf≤2 %`、`eps_2≤2 %`、`|amp_err|≤2 %`、`curv_res≤2 %`、`|mass_drift|≤1e-6`、`steady_rel_change≤1e-6`；未达稳态直接判 FAIL（不平均掩盖）。
3. **仪器化**：每个算例跑到机器稳态后，用最后两个 chunk 的平均剖面计分，同时输出质量漂移与稳态判据（失败模式可诊断）。
4. **发现并修复真实缺陷**：原 `_collide` 只保留 Guo 源项第一项 `3(c_ix−u_x)`，漏掉零和伙伴 `9(c_i·u)c_ix` → `Σ_i S_i = −3u_x g ≠ 0`，每步固定质量汇。修复为完整零和源项，并把一阶 `simple` 路径改为不做半步速度移位。
5. **工具链**：numpy（求解）+ matplotlib（图）+ pytest（回归）；`run.py res|sens|merge` 分片执行→`result.json`→`make_figures.py`→`make_doc.py`→本报告，全链路零手抄。

## ③ 计算结果（数字表）

参考工况（H=32, τ=0.8, g=1e-5）：`u_max` 数值 0.012776 vs 解析 0.012788。

| 量 | ref（解析） | got（LBM 稳态） | 相对误差 | 判据 | 达标 |
|---|---|---|---|---|---|
| u_max | 0.012788 | 0.012776 | -0.09 % | ≤2 % | ✅ |
| 剖面 eps_inf | — | — | 8.993e-04 | ≤2 % | ✅ |
| 剖面 eps_2 | — | — | 1.230e-03 | ≤2 % | ✅ |
| 动量平衡 curv_res | 0 | 3.39e-08 | — | ≤2 % | ✅ |
| 质量漂移 | 0 | -6.9e-13 | — | ≤1e-6 | ✅ |
| 稳态判据 | — | 2.5e-08 | — | ≤1e-6 | ✅ |

网格收敛（H=8→64）：

| H | ny | steps | eps_inf | eps_2 | amp_err | curv_res | mass_drift | 达标 |
|---|---|---|---|---|---|---|---|---|
| 8 | 12 | 2000 | 1.460e-02 | 1.968e-02 | -1.46 % | 1.05e-09 | -8.2e-14 | ✅ |
| 16 | 20 | 5000 | 3.608e-03 | 4.921e-03 | -0.36 % | 2.98e-10 | -3.1e-13 | ✅ |
| 32 | 36 | 11000 | 8.993e-04 | 1.230e-03 | -0.09 % | 3.39e-08 | -6.9e-13 | ✅ |
| 64 | 68 | 27000 | 2.247e-04 | 3.073e-04 | -0.02 % | 3.21e-07 | -1.7e-12 | ✅ |

拟合收敛阶 **2.00**（理论 2）。

## ④ 结果图（场量云图 / 与解析解比较曲线）

本算例的"场"是壁法向一维剖面（流向周期且充分发展，场量与 x 无关，故无二维云图可言——这是该 canonical 算例的物理属性，不是缺图）。四张图全部由 `result.json` 程序化生成。

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

- **踩坑 1（真缺陷）**：Guo 源项截断导致质量汇。修复前 H=64、30k 步质量漂移 `-1.14e-02`（-1.14 %），剖面振幅随步数单调漂移（H=32：10k→60k 步 amp 由 `-0.047 %` 变 `+0.134 %`），**不存在真稳态**；库内原有"5 % 容差"测试只是恰好在该步数通过。修复后质量漂移 `-1.9e-12`（float64 舍入），三个步数给出同一 `eps_inf`。
- **踩坑 2（壁位）**：半程反弹的壁位 O(1) 常数使 H=8 时 `eps_2` 已到 0.1 %（低分辨率不可用），必须走网格收敛判断。
- **局限**：nx=4 只验证一维剖面，不验证流向输运/入口发展段；Re 必须保持层流充分发展；Ma 需 <0.1，否则 O(Ma²) 压缩性误差顶破 2 % 门；教科书页级引用未能机器核对。
- **适用范围**：内流壁面摩擦/剖面类验证的"基准尺"；不适用于压差边界（见 poiseuille_2d）、瞬态启动（startup_poiseuille）、三维效应。

## ⑥ 误差判定（逐项 vs 基准）

| 关键量 | 误差 | 判据 | 结论 |
|---|---|---|---|
| u_max | -0.09 % | ≤2 % | ✅ |
| 剖面 max | 8.993e-04 | ≤2 % | ✅ |
| 剖面 L2 | 1.230e-03 | ≤2 % | ✅ |
| 动量平衡残差 | 3.39e-08 | ≤2 % | ✅ |
| 质量守恒 | -6.9e-13 | ≤1e-6 | ✅ |

**是否 <3 %？是（并进一步满足 <2 % 的锚点判据）：最大误差 +0.090 %，远低于 2 % 门。**

**复现命令**：
```bash
export PYTHONPATH=TensorLBM/src
python benchmarks/verified/poiseuille_2d_bodyforce/run.py res
python benchmarks/verified/poiseuille_2d_bodyforce/run.py sens --only tau,forcing
python benchmarks/verified/poiseuille_2d_bodyforce/run.py sens --only steps
python benchmarks/verified/poiseuille_2d_bodyforce/run.py sens --only legacy
python benchmarks/verified/poiseuille_2d_bodyforce/run.py merge
python benchmarks/verified/poiseuille_2d_bodyforce/make_figures.py
python benchmarks/verified/poiseuille_2d_bodyforce/make_doc.py
pytest TensorLBM/tests/test_poiseuille_2d_bodyforce.py -q
```
