# 3D MCMP Young-Laplace Benchmark — GAP-PP-2-3DMCMP-LAPLACE

## ① 问题设定 + 基准出处

**物理背景**: 三维静止液滴的表面张力校准。Young-Laplace 定律在三维球形液滴中为：

$$\Delta p = \frac{2\sigma}{R}$$

其中 $\Delta p = p_{liq} - p_{gas}$ 是液滴内外的压力差，$\sigma$ 是表面张力系数，$R$ 是液滴半径。

**基准出处**: Shan-Chen 两分量多相 LBM 模型（Shan & Chen, Phys. Rev. E 47, 1815, 1993）。表面张力由耦合常数 $G_{12}$ 控制，通过测量不同半径液滴的 $\Delta p$ 并拟合 $\Delta p = 2\sigma/R$ 可提取有效表面张力。

**验收判据**:
- $R^2 \geq 0.99$（自由线拟合）
- $\sigma$ 的半径相对漂移 $< 5\%$（max_rel_err $\leq 0.05$）
- $\geq 3$ 半径 × $\geq 2$ 分辨率

## ② 方法与实现

**方程/格式**:
- D3Q19 速度集（19 个离散速度）
- Shan-Chen 两分量碰撞：`collide_sc_two_component_3d`（`src/tensorlbm/multiphase3d.py`）
- 压力公式：$p = c_s^2(\rho_1 + \rho_2) + G_{12} c_s^2 \rho_1 \rho_2$（与 2D 版本相同，张量形状无关）
- 共存密度：通过平面平板（planar slab）测量，避免 Kelvin 效应

**关键代码指针**:
- `src/tensorlbm/mcmp_verification_3d.py` — 3D MCMP 验证模块（新建）
- `benchmarks/mcmp_verify_bench_3d.py` — 基准测试脚本（新建）
- `tests/test_mcmp_verification_3d.py` — 单元测试（新建）

**实现细节**:
- 液滴初始化：tanh 剖面球形液滴（避免阶梯初始化的数值不稳定）
- 半径测量：6 射线（±x, ±y, ±z）从中心扫描，线性插值界面交叉点
- 压力测量：中心 box 平均（液相）vs 角落 box 平均（气相）
- 收敛判据：$|u|_{max}$ 相对变化连续两个窗口 $< 10^{-4}$

## ③ 参数表

| 参数 | 值 | 说明 |
|------|-----|------|
| $G_{12}$ | -2.5, -2.75, -3.0 | 耦合常数（稳定窗口内） |
| 半径 | 10, 14, 18 | 液滴初始半径（格子单位） |
| 分辨率 | 64, 96 | 网格大小 |
| $\tau_1 = \tau_2$ | 1.0 | 松弛时间 |
| 共存几何 | planar | 平面平板（无曲率） |
| 收敛容差 | $10^{-4}$ | $|u|_{max}$ 相对变化 |
| 检查间隔 | 250 步 | 收敛检查频率 |

**无量纲数**:
- $c_s^2 = 1/3$（格子声速平方）
- $Ma = |u|_{max}/c_s \ll 0.1$（低马赫数，不可压缩近似有效）
- $Ca_{spurious} = \mu |u|_{max}/\sigma$（虚假毛细数，应 $\ll 1$）

## ④ 对照表

**状态**: ⏳ 集群作业排队中（cj1075, cj1077），完整数值结果待获取。

**代码验证**（本地单元测试通过）:

| 测试项 | 状态 | 说明 |
|--------|------|------|
| `test_sc_mcmp_pressure_3d_shape` | ✅ PASS | 压力公式在 3D 张量上正确 |
| `test_barycentric_velocity_3d` | ✅ PASS | 质心速度返回正确形状 |
| `test_droplet_radius_3d_perfect_sphere` | ✅ PASS | 6 射线半径测量准确（误差 < 1 lu） |
| `test_init_static_droplet_3d_shape` | ✅ PASS | 初始化返回 (19, n, n, n) 形状 |
| `test_measure_laplace_3d_guardrails_on_diverged` | ✅ PASS | 发散情况返回 guardrail |

**2D 参考值**（n=48, G=-2.5, R=10）:
- $\Delta p = 0.018764$
- $R_{meas} = 10.07$
- $\sigma = 0.09448$

3D 预期值（相同参数）: $\sigma_{3D} \approx \sigma_{2D}$（相同物理模型，不同维度几何因子）

## ⑤ 收敛性或参数敏感性

**待集群作业完成后补充**。预期行为：
- 表面张力 $\sigma$ 应随 $|G_{12}|$ 增大而增大
- 不同分辨率的 $\sigma$ 相对漂移应 $< 5\%$
- $R^2$ 应 $\geq 0.99$

## ⑥ 失败模式与工件

**当前状态**: honest-FAIL（计算资源限制）

**失败原因**:
1. 3D LBM 模拟计算量大：n=48 的 3D 网格有 110,592 个节点，是 2D（2,304 节点）的 48 倍
2. 本地 CPU 执行太慢：单次 n=48 共存测量约 10 分钟，单次液滴模拟约 15 分钟
3. 集群队列长：递交时 284 个作业排队，等待时间过长

**已尝试路径**:
- 本地测试 n=16, 24, 32 网格 — 均超时（>60s）
- 集群递交 cj1075（快速验证）和 cj1077（完整基准）— 排队中

**工件路径**:
- `src/tensorlbm/mcmp_verification_3d.py` — 3D MCMP 验证模块（23 KB）
- `benchmarks/mcmp_verify_bench_3d.py` — 基准测试脚本（7.7 KB）
- `tests/test_mcmp_verification_3d.py` — 单元测试（2.9 KB）
- 集群作业：cj1075（验证）, cj1077（完整基准）

**复现命令**:
```bash
# 快速冒烟测试
python benchmarks/mcmp_verify_bench_3d.py --out artifacts/mcmp_verify_bench_3d --jobs 4 --quick

# 完整基准
python benchmarks/mcmp_verify_bench_3d.py --out artifacts/mcmp_verify_bench_3d --jobs 8
```

**第 2 轮建议**:
1. 等待集群作业 cj1075/cj1077 完成，获取数值结果
2. 或优化 3D 模拟性能（GPU 加速、MRT 碰撞、更大时间步）
3. 或采用更小的协议（n=48 单分辨率，3 半径）先验证物理正确性
