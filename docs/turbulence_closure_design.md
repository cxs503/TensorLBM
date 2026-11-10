# TensorLBM 湍流闭合验证面（turbulence_closure）设计说明

> 节点 P152W1-N02。目标：把 TensorLBM 的湍流预报能力从"层流可算"推到"湍流可算且可验证"，
> 并提供可审计的 DNS/实验闭合链。所有数字由 `python -m tensorlbm.turbulence_closure.campaign`
> 程序化生成，报告里不出现手抄数值。

## 1. 三个物理件

| 件 | 文件 | 作用 |
|---|---|---|
| MRT 碰撞器 | `mrt.py` / `mrt_d2q9.c` / `mrt_channel3d.c` | D2Q9 + D3Q19 多松弛碰撞，含 Smagorinsky 亚格子闭合 |
| 非平衡壁模型 | `mrt_d2q9.c`（NEEM 出入口）、`mrt_channel3d.c`（Reichardt 壁面应力） | 开放边界速度/压力 Dirichlet + 壁面剪应力注入 |
| 验证面 | `cases.py` / `metrics.py` / `reference.py` / `campaign.py` | 三案例 × 三档 Re，逐项误差与判据 |

### 1.1 为什么矩矩阵是"生成"的
经典 MRT 实现把 d'Humières 的矩表逐项抄进代码，一旦速度排序不一致就静默出错。这里改为：

1. 用格点离散速度生成按总阶数分级的单项式候选（1、c_a、c_a c_b、c_a c_b c_c …）；
2. 按序做 Gram-Schmidt 正交归一，丢弃该格点不能表示的退化方向；
3. 取前 `n_q` 个即得矩矩阵 `M`，因此 `M·Mᵀ = I`、`M⁻¹ = Mᵀ`，子空间严格嵌套：
   0 密度、1..d 动量、d+1 能量（Pi 迹）、随后 d(d+1)/2−1 个偏应力、其余鬼模。

偏应力行以 `s_ν = 1/τ_eff` 松弛，故 `ν = c_s²(1/s_ν − 1/2)`；鬼模取 `s=1`（即"正则化"到平衡），
能量行取 `s=1.2`。同一矩阵由 Python 写入文本文件、C 内核读入（`native.matrix_file`），
两套实现因此不可能漂移。

`M` 一旦正交归一，碰撞可以写成极省的形式（本实现即如此）：

```
f* = f + Σ_{k ∈ 需松弛的矩} (s_k − 1) · (M_k · (f − f_eq)) · M_k
```

即"BGK + 少量秩 1 修正"，比逐矩稠密变换省一个量级，且与逐矩写法**严格等价**（测试
`test_mrt_with_equal_rates_reproduces_bgk` 逐点比对 BGK）。

### 1.2 Smagorinsky 亚格子闭合
由非平衡应力 `Π^(neq)_ab = Σ c_a c_b (f − f_eq)` 反解应变率幅值：

```
|S| = sqrt(Π:Π / 2) / (c_s² τ ρ),  ν_t = (C_s Δ)² |S|,  τ_eff = 1/2 + 3(ν + ν_t)
```

两次不动点迭代即可（`mrt.py::MRTModel.tau_field`、C 内核同名逻辑）。

### 1.3 非平衡壁模型
* **NEEM（Guo 2002）**用于出入口：`f = f^eq(ρ_nb, u_in) + [f_nb − f^eq(ρ_nb, u_nb)]`，
  出口用 `ρ = 1` 的压力 Dirichlet。速度/压力因此是"二阶外推"而不是简单 zeroth-order 拷贝。
* **Reichardt 壁面应力模型**用于槽道壁面邻格：由第一层速度反解 `u_τ`
  （`U = u_τ · R(y₁u_τ/ν)`，`R` 为 Reichardt 1951 复合律），把 `ν_w = u_τ y₁` 作为该格等效黏性注入，
  即把壁面动量通量 `τ_w = u_τ²` 直接钉住，内部不受约束。
  **适用窗口的诚实说明**：弱可压缩 LBM 里 Ma 与 τ 强耦合，`y⁺ ≈ Re_τ/(2h)`，
  而 `u_center = Re_τ ν u⁺/h` 又限制 ν 不能太小；因此在 Re_τ=395/590 上取 Ma<0.3 时
  y⁺ 必然落在 1–3，壁面是解析的、壁模型"用不上"。这一点写进报告局限，不用壁模型去凑判据。

## 2. 三个案例

| 案例 | 求解器 | 判据 | 锚 |
|---|---|---|---|
| 槽道流 | `mrt_channel3d.c`（D3Q19 MRT + LES，x/z 周期，半程反弹壁） | u⁺(y⁺) 最大相对误差 < 10% | MKM1999 DNS（随仓、SHA-256） |
| 后向台阶 | `mrt_d2q9.c`（抛物线入口 + NEEM 出口 + 台阶块） | x_R/h 误差 < 15% | Armaly 1983 实验（ER=1.9423） |
| 圆柱绕流 | `mrt_d2q9.c`（均匀入口 + NEEM 出口） | St 误差 < 5% | Williamson 实验 + W&B 1998 二维 DNS |

**槽道初值**：Reichardt 平均剖面 + 5% 三分量随机扰动。从静止起步的黏性自旋时间
`h²/ν` 是统计所需步数的两个量级，必须给平均剖面。

**圆柱堵塞比**：判据对应的是无界实验值，而任何有限域都有堵塞比 `B = D/H`。
本战役不掩盖这一点：在 Re=100 做 `B` 扫描（H = 40D/30D/20D/12D），报告工作堵塞比下的原值
与 `B→0` 线性外推值（`campaign.results["cylinder_blockage"]`）。

## 3. 数值验证（判据之外的第一性检查）

| 检查 | 方法 | 通过线 |
|---|---|---|
| 矩基正交性 | `M·Mᵀ = I` | 1e-12 |
| 碰撞守恒 | 质量/动量逐点守恒 | 1e-12 |
| MRT↔BGK 等价 | 全矩同率时逐点等于 BGK | 1e-12 |
| 剪切黏性 | 剪切波衰减率 = νk² | <5% |
| 内核黏性 | 周期槽道 Poiseuille 稳态解反解 ν | <1%（h=11） |
| 层流孪生 | 3D 内核 vs 解析抛物线 | <2% |

## 4. 复现

```bash
PYTHONPATH=src python -m tensorlbm.turbulence_closure.campaign \
    --out runs/turbulence_validation_face --threads 48 \
    --report docs/turbulence_validation_face.md
PYTHONPATH=src python -m pytest tests/test_turbulence_closure.py -q
```

`results.json` 里每个案例都带 `provenance`（二进制 SHA-256、主机、架构、线程数、
cell-steps/s），每条原始记录含 `meta.txt`/`run.json`。

## 5. 已知局限

1. **二维限制**：台阶与圆柱是二维；Re ≥ 200 时三维实验与二维 DNS 的 St 本身相差 3–7%，
   报告同时给出与实验、与二维 DNS 两条误差。
2. **堵塞比**：圆柱以 `B` 扫描 + 外推处理；台阶入口采用充分发展抛物线剖面，未施湍流入口。
3. **最小盒统计噪声**：槽道 `Lx⁺`、`Lz⁺` 取得偏小（最小盒），平均剖面统计噪声约 1–3%。
4. **壁模型窗口**：见 §1.3，Re_τ≥395 时壁模型退化为近乎解析壁。
5. **力矩测量**：`C_d` 用动量交换与近场压力积分两条路线给出；台阶状离散体上二者存在系统性差异，
   故 `C_d` 只作诊断量，不参与判据（判据只用 St 的**频率**，与力的标定无关）。