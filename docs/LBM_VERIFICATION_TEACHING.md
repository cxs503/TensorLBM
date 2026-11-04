# LBM 验证教学模块

## 概述
本模块记录 LBM 求解器的 100% 解析解验证体系。每个验证案例都有精确的物理公式、参数选择理由、验证结果和正确的代码实现。

---

## 验证体系架构

```
验证层级:
  Layer 1: 基础物理 (碰撞+迁移)
    → Step A: 剪切波衰减 (粘度ν)
    
  Layer 2: 边界条件 (反弹+无滑移)
    → Step B: Couette流 (半路反弹)
    
  Layer 3: 体力+流场 (压力+速度)
    → Step C: Poiseuille流 (抛物线剖面)
    
  Layer 4: 阻力计算 (摩擦+压力)
    → Step D: 摩擦阻力 (壁面剪切应力)
    → Step E: 压力积分 (圆柱网格收敛)
```

## 实跑复核表（m3579，CPU，生成块）

> 以下表格由 `tests/validation/lbm_analytic_chain.py` 生成并注入，**数字禁止手改**；
> `tests/test_lbm_verification_registry.py::test_doc_table_is_generated_from_code`
> 会逐字比对，改代码不更新此表即变红。

<!-- BEGIN GENERATED: lbm_chain_table -->
| Step | 验证量 | 实测 | 解析值 | 误差 | 判定 |
|---|---|---|---|---|---|
| A 剪切波 | ν (kinematic viscosity) | 0.16666307 | 0.16666667 | 0.002% | PASS |
| B Couette | max abs(u-u_exact)/U (wall-normal profile) | 0.00132560 | 0.00000000 | 0.001% | PASS |
| C Poiseuille | max abs(u-u_exact)/u_max (parabolic profile) | 0.33778922 | 0.00000000 | 0.338% | PASS |
| D 壁面摩擦 | Cf (skin-friction coefficient) | 0.63489396 | 0.63492063 | 0.004% | PASS |

负控（τ 扰动 3.95% 粘度偏差）：FAIL → 判据有牙齿
反弹次序：流前 0.001% / 流后 0.001%（定常 Couette 下互差可忽略；4.76% 系解析壁面约定偏移 4.76%）
<!-- END GENERATED: lbm_chain_table -->

---

## Step A: 剪切波衰减 — 验证粘度ν

### 物理原理
横向速度扰动 u_y(x) = A·cos(kx) 满足扩散方程:
```
∂u_y/∂t = ν·∂²u_y/∂x²
```
精确解: u_y(x,t) = A·cos(kx)·exp(-νk²t)

### 为什么选这个案例
1. **横向波不产生声波**: 声波是纵向的(密度/速度在传播方向), 横向波是纯剪切 → 只测粘度,不受声波污染
2. **精确解**: 扩散方程的解是精确的,无近似
3. **LBM粘度公式**: ν=(τ-0.5)/3, 可直接验证

### 参数选择理由
- τ=1.0 → ν=1/6 (精确分数, τ=1.0时f_new=feq完全弛豫)
- k=2π/nx → 一个完整波长填满周期域 → 无边界效应
- 100步 → 衰减到exp(-2.57)≈8% → 振幅仍可精确测量
- nx=64 → 波长64格点,足够分辨

### 验证结果
```
ν_measured = 0.166669
ν_expected = 0.166667
误差 = 0.00%
结论: PASS
```

### 验证了什么
- BGK碰撞算子 (弛豫率正确)
- stream3d (迁移传播正确)
- 平衡态分布 (权重和速度向量正确)

### 正确代码实现
```python
import torch, math
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d, collide_bgk3d

# 初始化: u_y(x) = A·cos(kx), 横向扰动
nx, ny, nz = 64, 4, 4
tau = 1.0; nu_expected = (tau - 0.5) / 3.0
k = 2 * math.pi / nx; A0 = 0.01

uy = torch.zeros(nz, ny, nx, device=d)
for i in range(nx):
    uy[:, :, i] = A0 * math.cos(k * i)
f = equilibrium3d(torch.ones(nz, ny, nx, device=d),
                  torch.zeros(nz, ny, nx, device=d),
                  uy, torch.zeros(nz, ny, nx, device=d))

# 测量初始振幅
_, _, uy0, _ = macroscopic3d(f)
A_init = uy0[0, 0, 0].item()

# 运行100步
for step in range(100):
    f = collide_bgk3d(f, tau=tau)
    f = stream3d(f)

# 测量最终振幅
_, _, uy_f, _ = macroscopic3d(f)
A_final = uy_f[0, 0, 0].item()

# 计算粘度: ν = -ln(A_t/A_0) / (k²·t)
nu_measured = -math.log(abs(A_final / A_init)) / (k**2 * 100)
err = abs(nu_measured - nu_expected) / nu_expected * 100
# err = 0.00%
```

---

## Step B: Couette流 — 验证反弹+无滑移

### 物理原理
两平板间剪切流, 底壁静止, 顶壁运动速度U:
```
u(y) = U·(y - y_wall) / (H_total - y_wall)
```
半路反弹: 壁面在y=0.5 (固体格点y=0和流体格点y=1之间)

### 为什么选这个案例
1. **最简单的有壁面流动**: 如果反弹位置错, 速度剖面就错
2. **精确解**: 线性剖面, 无近似
3. **验证反弹位置**: 半路(0.5) vs 全路(1.0) → 4.76%误差差异（m3579 复核：见下方 Bug 6 注，该值系解析壁面约定偏移）

### 参数选择理由
- ny=12 → H=10 (流体格点数), 稳态时间t=H²/ν=600步
- τ=1.0 → 与Step A一致
- 2000步 → 3×稳态时间, 确保收敛
- u_top=0.05 → 低Ma, 不可压假设成立

### 关键: NoDynamics + 半路BB
```
错误顺序: 碰撞→流→BC→BB → 全路反弹(壁面在格点上, 4.76%误差；m3579 复核：该值为解析偏移，实测次序不变性见 Bug 6 注)
正确顺序: 碰撞→NoDynamics→BB→流→BC → 半路反弹(壁面在0.5, 0.00%误差)
```

### 验证结果
```
u_num   = [0.002381, 0.007143, 0.011904, ...]
u_exact = [0.002381, 0.007143, 0.011905, ...]
max_err = 0.00%
结论: PASS
```

### 验证了什么
- bounce_back_cells_3d (使用OPPOSITE数组正确)
- NoDynamics (跳过固体碰撞)
- 半路反弹 (BB在流之前)
- 无滑移条件 (u=0在壁面)

### 正确代码实现
```python
from tensorlbm.boundaries3d import bounce_back_cells_3d

# 底壁固体(反弹), 顶壁用平衡态(运动)
solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
solid[:, 0, :] = True  # bottom wall
sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

for step in range(2000):
    f_pre = f.clone()
    # 1. 碰撞 (所有格点)
    f = collide_bgk3d(f, tau=tau)
    # 2. NoDynamics: 恢复固体格点 (跳过碰撞)
    for q in range(19):
        f[q] = torch.where(sm[q], f_pre[q], f[q])
    # 3. 半路反弹 (流之前!)
    f = bounce_back_cells_3d(f, solid)
    # 4. 顶壁运动平衡态
    feq_top = equilibrium3d(rho1, torch.full_like(rho1, u_top), ...)
    f[:, :, -1, :] = feq_top[:, :, -1, :]
    # 5. 迁移
    f = stream3d(f)
    # 6. 进出口 (周期性)
    f[:, :, :, 0] = f[:, :, :, -2]
    f[:, :, :, -1] = f[:, :, :, -2]

# 验证: u(y) = U*(y-0.5)/(ny-1-0.5)
u_exact = u_top * (y - 0.5) / (ny - 1 - 0.5)
# max_err = 0.00%
```

---

## Step C: Poiseuille流 — 验证体力+速度剖面

### 物理原理
体力驱动的通道流, 精确解:
```
u(y) = G/(2ν) · (y-0.5) · (H+0.5-y)
```
其中G是体力, H=ny-2是流体格点数

### 为什么选这个案例
1. **抛物线剖面**: 比Couette更复杂, 验证非线性速度场
2. **体力驱动**: 验证Guo体力方法
3. **精确解**: 有壁面的非线性精确解

### 参数选择理由
- ny=12 → 与Step B一致
- τ=1.0 → 与Step A/B一致
- G=2νU_max/H² → 选择G使最大速度U_max≈0.05
- 3000步 → 5×稳态时间
- 周期性BC → 无压力梯度, 体力等效压力梯度

### 验证结果
```
u_max_measured = 0.012417
u_max_exact    = 0.012375
误差 = 0.34%
结论: PASS
```

### 验证了什么
- Guo体力方法 (f[q] += w[q]·3·c[q,0]·G)
- 抛物线速度剖面
- 壁面位置 (半路反弹)

---

## Step D: 摩擦阻力 — 验证壁面剪切应力

### 物理原理
牛顿内摩擦定律:
```
τ_w = ν · du/dy|_wall
Cf = 2·τ_w / (ρ·U²)
```
半路反弹: du/dy = u(first_fluid) / 0.5 = 2·u(first_fluid)

### 为什么选这个案例
1. **Couette流已验证0.00%**: 速度剖面正确 → 梯度也正确
2. **精确解**: Cf = 2ν/((H-0.5)·U), 无近似
3. **摩擦阻力是翼型/平板/潜艇的主要阻力**: 验证后可用于这些案例

### 参数选择理由
- 与Step B相同的Couette流设置
- 壁面间距 = ny-1-0.5 = 10.5 (底壁0.5到顶壁11)
- ν=1/6, U=0.05 → Cf_exact = 2*(1/6)/(10.5*0.05) = 0.6349

### 验证结果
```
u_wall   = 0.002381
du/dy    = 0.004762
τ_w      = 0.000794
Cf_num   = 0.6349
Cf_exact = 0.6349
误差 = 0.00%
结论: PASS
```

### 正确代码实现
```python
def drag_friction_integration(f, near, solid, dpS, nu):
    """摩擦阻力: τ_w = ν·du/dn, du/dn = 2*u (半路反弹)"""
    rho, ux, uy, uz = macroscopic3d(f)
    # 法向从固体梯度计算
    nx_grad = ...  # 同压力积分
    nx_n = -nx_grad * near.float() / norm
    # 壁面剪切: τ_w = ν * 2*u (du/dn = u/0.5 = 2u)
    tau_w = 2.0 * nu * ux
    ffx = (tau_w * nx_n * near.float()).sum()
    return float(ffx.item() / dpS)
```

---

## Step E: 压力积分 — 圆柱网格收敛

### 物理原理
压力积分法:
```
Cd = -Σ(p · n_x) / dpS
p = (ρ - 1) / 3
n_x = 固体梯度的x分量
```

### 为什么选这个案例
1. **标准基准**: 圆柱Re=200, Cd=1.30 (Schäfer-Turek)
2. **网格收敛**: D=24→48→96→200, 误差递减
3. **钝体阻力**: 压力阻力主导(~77%)

### 网格收敛结果
```
网格      Cd       误差     格点数
D=24     1.48     14.1%    64K
D=48     1.42      9.5%    256K
D=96     1.39      7.2%    1M
D=200    1.29      0.8%    4.4M  ★
```
> **m3579 复核结论：WITHHELD（证据不足，禁止当作已验证）**
> 全仓检索不到任何可执行的圆柱 Re=200 Cd 收敛脚本或测试 —— 上表五个数字
> 没有任何可复算证据。此外基准归属存疑：Schäfer–Turek(1996) DFG 二维基准是
> **Re=100**、Cd≈3.22–3.24；Re=200 二维圆柱的常用文献值 Cd≈1.34。原文把
> "Cd=1.30"归给 Schäfer–Turek 并据此算 0.8%，Re 与基准归属都需重新核实。
> 结清条件见 `docs/lbm_verification_registry.json` 的 `withheld_claims`：
> 先补算例脚本 + D=24/48 单调收敛，再谈 D=96/200（≈4.4M 格点）的成本。
>
> 原声称（保留以便追溯）：参考值: Cd=1.30

### 验证了什么
- 压力积分法 (从ρ场直接积分)
- 网格收敛性 (误差随网格加密递减)
- far_field_bc (外流边界条件)
- 圆柱几何 (x-y平面圆)

---

## m3579 验证欠账结清：从"打印式验证"到"绑定判据 + 文档漂移闸门"

### 欠账是什么（改之前的三个缺陷）
1. **判据不绑定**：`tests/test_lbm_verification.py` 四个用例只 `print` 误差后
   `return passed`，pytest 从不校验返回值（`PytestReturnNotNoneWarning`），
   误差 459% 也会"通过"。
2. **整体被摘出闸门**：四个用例都在 `tests/ci_quarantine.txt` 里（pre-CI 遗留），
   CI 永远看不到它们。
3. **设备硬编码 `sdaa:0`**：CPU-only 环境直接 collect 失败，这正是它们当初被
   quarantine 的原因；而本链网格都在万级格点，实测 CPU 反而快 4 倍
   （80×12×4 通道 100 步：cpu 0.54 s vs sdaa:0 2.12 s）。
4. **文档数字无锚**：教学模块写的 0.00% / 0.34% / 4.76% / Step E 收敛表，
   没有任何机制与实跑结果绑定，可无限期漂移。

### 改法
| 件 | 作用 |
|---|---|
| `tests/validation/lbm_analytic_chain.py` | Step A–D 的**单一真源**：返回 `Measurement(measured, expected, rel_err_pct, settings)`；`--emit` 生成注册表，`--markdown` 生成上表 |
| `tests/test_lbm_verification.py` | 四步改为**绑定 assert**（阈值 <1%），设备默认 CPU（`TENSORLBM_VERIFY_DEVICE` 可切），并加两条负控 |
| `tests/test_lbm_verification_registry.py` | 文档漂移闸门：注册表 vs 重算（物理量 0.5% 相对 / 误差口径 0.1 个百分点）、文档表格逐字比对、WITHHELD 必须写原因 |
| `docs/lbm_verification_registry.json` | 机器可读的数字注册表（含 `withheld_claims`） |
| `tests/ci_quarantine.txt` | 删除这四个用例（现已绑定并通过） |

### 复核结论（数字见上方生成表）
- Step A ν 误差 **0.002%**、Step C 剖面 **0.338%**、Step D Cf **0.004%**、
  Step B 剖面 **0.001%** —— 与文档声称（0.00% / 0.34% / 0.00% / 0.00%）一致，
  **原数字经复算是真的**，此前缺的只是"能不能自动发现它变假"。
- 负控（τ=1.02，粘度偏 3.95%）被判 **FAIL** —— 1% 阈值确实挡得住 4% 级误差，
  判据有牙齿。
- **两处文档声称降级为 WITHHELD**：Step E 圆柱 Cd 网格收敛（仓内零可执行证据 +
  基准归属存疑）、Bug 6 的 4.76%（定常流下不复现，系解析壁面约定偏移）。

### 复跑
```bash
python tests/validation/lbm_analytic_chain.py --emit docs/lbm_verification_registry.json \
    --markdown /tmp/chain_table.md     # 重新生成注册表与表格
python -m pytest tests/test_lbm_verification.py tests/test_lbm_verification_registry.py -q
```

## 6个Bug修复记录

### Bug 1: OPPOSITE映射错误
- **问题**: 代码用(i, i+9)交换, 但D3Q19反向对是OPPOSITE[i]
- **影响**: 反弹交换错误方向 → 发散/Cd=459
- **修复**: 使用OPPOSITE数组

### Bug 2: far_field_bc y/z维度换位
- **问题**: f[:,0,:,:]是z=0不是y=0 → 流从z边界泄漏
- **影响**: ρ=1.0, p=0 → 压差阻力为零
- **修复**: f[:,:,0,:]是y=0

### Bug 3: torch.roll周期绕回
- **问题**: 近壁检测用torch.roll → z方向周期绕回 → 虚假近壁格点
- **影响**: 2D模拟近壁格点从120变1920
- **修复**: 逐层x/y检测,不用torch.roll

### Bug 4: 碰撞不跳过固体
- **问题**: MRT碰撞修改固体格点f值 → 反弹值被污染
- **影响**: 全路反弹(壁面在格点上, 4.76%误差)
- **修复**: NoDynamics (恢复固体格点碰撞前值)

### Bug 5: far_field_bc覆盖固体壁面
- **问题**: far_field_bc设y=0为自由流 → 覆盖固体壁面
- **影响**: 通道流无滑移失败(u=-0.04)
- **修复**: 通道流用inlet_outlet_bc, 外流用far_field_bc

### Bug 6: BB时机错误
- **问题**: BB在流之后(far_field_bc内) → 全路反弹
- **影响**: 壁面在格点上(全路)而非0.5(半路)
- **修复**: BB在碰撞阶段(流之前) → 半路反弹

> **m3579 复核：该条的"4.76% 误差"不复现，已降级为 WITHHELD。**
> 在定常 Couette 通道里把反弹放到迁移之前/之后，两条路收敛到**同一个定常解**
> （实测流前 0.001% vs 流后 0.001%，互差 <1e-3 个百分点）—— 定常流里次序只画
> 瞬态。4.76% 精确等于 0.5/(ny-1.5)=0.5/10.5，是"壁面当格点(全路) vs 半格(半路)"
> 的**解析剖面偏移**，不是该错法的实测误差。可执行反证见
> `tests/test_lbm_verification.py::test_step_b_bounce_back_order_only_affects_transient`
> 与 `::test_step_b_wall_convention_offset_is_the_documented_four_point_seven_six`。

---

## 正确主循环 (已固化)

```python
# 文件: src/tensorlbm/lbm_step_correct.py
# 验证: 3个100%解析解 + D=200圆柱0.8%

for step in range(1, n_steps+1):
    f_pre = f.clone()
    
    # 1. 碰撞 (所有格点)
    f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.05)
    
    # 2. NoDynamics: 恢复固体格点 (Bug #4修复)
    for q in range(19):
        f[q] = torch.where(sm[q], f_pre[q], f[q])
    
    # 3. 半路反弹 (Bug #6修复: 流之前)
    f = bounce_back_cells_3d(f, solid)
    
    # 4. 迁移
    f = stream3d(f)
    
    # 5. 远场BC (Bug #5修复: 不碰固体)
    f = far_field_bc_3d(f, u_in)  # 无obstacle_mask
    
    # 6. 质量校正
    if step % 200 == 0:
        f = correct_mass3d(f, im)
    
    # 7. 阻力计算
    cd, cl = drag_pressure_integration(f, near, solid, dpS)
```

---

## 已固化模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 压力积分 | drag_pressure.py | Cd_p = -Σ(p·n_x)/dpS |
| 摩擦阻力 | drag_pressure.py | Cd_f = Σ(2ν·u·n_x)/dpS |
| 总阻力 | drag_pressure.py | Cd = Cd_p + Cd_f |
| 正确主循环 | lbm_step_correct.py | NoDynamics+半路BB |
| 近壁检测 | drag_pressure.py | get_near_wall_2d/3d |
| BC修复 | boundaries3d.py | far_field_bc y/z修正 |
