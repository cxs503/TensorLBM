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
3. **验证反弹位置**: 半路(0.5) vs 全路(1.0) → 4.76%误差差异

### 参数选择理由
- ny=12 → H=10 (流体格点数), 稳态时间t=H²/ν=600步
- τ=1.0 → 与Step A一致
- 2000步 → 3×稳态时间, 确保收敛
- u_top=0.05 → 低Ma, 不可压假设成立

### 关键: NoDynamics + 半路BB
```
错误顺序: 碰撞→流→BC→BB → 全路反弹(壁面在格点上, 4.76%误差)
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
参考值: Cd=1.30

### 验证了什么
- 压力积分法 (从ρ场直接积分)
- 网格收敛性 (误差随网格加密递减)
- far_field_bc (外流边界条件)
- 圆柱几何 (x-y平面圆)

---

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
