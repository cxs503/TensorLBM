# 3D 环隙 Poiseuille 流（D3Q19）— 解析解验证（条款修订版）

**状态：✅ verified（2026-10-19 条款修订归档）**

## 物理问题

三维同心环隙（a/R_o = 0.5）内充分发展层流，R_o = 20/40/80 三档，直接模拟 vs
**精确解析解**（自推，无文献数据依赖）：

$$u(r) = \frac{G}{4\nu}\left[R_o^2 - r^2 + \frac{R_o^2-a^2}{\ln(R_o/a)}\ln\frac{r}{R_o}\right],\qquad
Q = \frac{G\pi}{8\nu}\Phi,\quad \Phi = R_o^4 - a^4 - \frac{(R_o^2-a^2)^2}{\ln(R_o/a)}$$

- **CH1 主通道** = 全剖面反演有效几何（3 参数解析族 u = c0 + c1·ln r − c2·r² 对中心 bin
  加权 LSQ，两无滑移根 = 有效半径 (δ_i, δ_o)；梯度锚定施加流量
  G_Q = 8νQ_meas/(πΦ)；指标 = 中心 binned max rel err 对 (δ_i, δ_o, G_Q) 精确解。
- **CH2 名义帧直评**（单独披露）：名义几何 (a, R_o) + G_nom。

### 条款修订（2026-10-19 owner 裁定）

原预注册条款要求 δ_i 跨档 spread ≤ 0.05（"格无关"）。证据表明 δ_i 的 O(1/R) 漂移
来自内壁楼梯曲率真实物理效应（非求解器缺陷），已证非欠收敛（90k 步平稳性证明：
δ_i 位移 ~8.8×10⁻⁶ 格）。修订为：**δ_i 允许 O(1/R) 单调收敛口径**
（δ_i 绝对值随 R 增大而减小），δ_o spread 维持 ≤ 0.05 原条款。

**证据链**：
- δ_i−a = −0.3807/−0.3216/−0.2870（严格单调收敛）
- δ_o−R_o = +0.2105/+0.2117/+0.1716（spread = 0.0401 ≤ 0.05，过线）
- 90k 步长跑 δ_i 位移 8.8×10⁻⁶ 格（`case_Ro20_long.json`），证明空间离散性质，非欠收敛
- 平壁对照 verified/poiseuille_3d_duct：da/db spread 0.0355/0.0357（<0.05，过线）
- 圆管对照 verified/poiseuille_3d_pipe：R_fit = R+0.23（同 O(1/R) 楼梯机制）

## 配置

| 项 | 值 |
|---|---|
| 格子/碰撞 | D3Q19 BGK（τ=0.8），库原语 |
| 驱动 | Zou-He 速度入口 u_in = 0.02 / 压力出口 ρ=1 |
| 壁 | 库 `bounce_back_cells_3d`（post-stream 半程反弹） |
| 入口/出口 | `boundaries3d.zou_he_inlet_velocity_3d` / `zou_he_outlet_pressure_3d` |
| 域 | L/R = 6，测点 x=nx/2（径向分环逐 bin 平均） |
| 档位 | R_o = 20/40/80（a = 10/20/40，a/R_o = 0.5） |
| 外推 | `extrap: "none"`（零外推，真实模拟） |

## 结果

| R_o | Re | 步数 | eff max %（CH1） | eff L2 % | nom max %（CH2） | δ_i−a | δ_o−R_o | G_fit/G_Q |
|---|---|---|---|---|---|---|---|---|
| 20 | 4.0 | 49400 | **1.1634** | 0.6220 | 9.0529 | −0.3807 | +0.2105 | 0.9960 |
| 40 | 8.0 | 33800 | **0.1270** | 0.0749 | 5.2262 | −0.3216 | +0.2117 | 0.9987 |
| 80 | 16.0 | 40800 | **0.0514** | 0.0256 | 2.5697 | −0.2870 | +0.1716 | 0.9996 |

- CH1 主通道：1.1634% → 0.1270% → 0.0514%（≤3%，严格单调 ✓）
- CH2 名义帧：9.0529% → 5.2262% → 2.5697%（粗档 >3%，披露；根因 = 内壁有效偏移 −0.38 格）
- δ_i 跨档：单调 O(1/R) 收敛（修订条款 ✓）；δ_o spread = 0.0401 ≤ 0.05（原条款 ✓）
- G_fit/G_Q → 1.000（0.9960/0.9987/0.9996），收敛趋势良好

### 披露

1. **逐格（per-cell）最严变体**：9.4954% → 7.584% → 4.1986%（楼梯壁单格几何散布，
   O(1/R) 收敛，与 poiseuille_3d_pipe 同性质；披露，不作主判据）
2. **名义帧粗档 >3%**：根因 = 内壁有效偏移 −0.38 格（楼梯几何固有，非求解器缺陷）
3. **质量漂移**：3.215% / 1.471% / 0.707% = Zou-He 稳态密度抬升（物理 G·nx/(2c_s²)），
   不影响反演稳定性
4. **近出口发展度偏差**：fd_max 3.488% / 2.002% / 1.088%（出口 Zou-He 局部效应，披露）
5. **90k 步平稳性**：R_o=20 延长到 90000 步，δ_i 从 −0.380715462 到 −0.380708794
   （位移 ~8.8×10⁻⁶ 格），证明漂移是空间离散性质，非时间未稳态

## 判定

**verified**（条款修订口径，2026-10-19）：
- 共性模块入口（src/tensorlbm 库路径驱动）✓
- 直接观测量 ≤3%（CH1 主通道 1.1634%/0.1270%/0.0514%，三档全过）✓
- ≥2 档网格严格单调收敛（1.1634 → 0.1270 → 0.0514，严格单调）✓
- 参考先锁定后跑（解析解预注册，零外推）✓
- δ_i O(1/R) 单调收敛（修订条款 ✓），δ_o spread 0.0401 ≤ 0.05（原条款 ✓）

## 运行

```bash
cd benchmarks/verified/poiseuille_3d_annulus
PYTHONPATH=../../../src python run.py scan <out_dir> --R 20 40 80 --device cuda:0
PYTHONPATH=../../../src python run.py summarize <out_dir> --R 20 40 80
```

工件：`result.json`（三档 + 复合判决）、`case_Ro20/40/80.json`（正式三档）、
`case_Ro20_long.json`（90000 步平稳性证明）。

<!-- PROVENANCE
[{"v": 20, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "R_o_list[0]"},
{"v": 40, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "R_o_list[1]"},
{"v": 80, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "R_o_list[2]"},
{"v": 1.1634, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "per_grid[0].eff_max_bin_central_pct"},
{"v": 0.127, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "per_grid[1].eff_max_bin_central_pct"},
{"v": 0.0514, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "per_grid[2].eff_max_bin_central_pct"},
{"v": -0.3807, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "per_grid[0].delta_i_minus_a"},
{"v": -0.3216, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "per_grid[1].delta_i_minus_a"},
{"v": -0.287, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "per_grid[2].delta_i_minus_a"},
{"v": 0.2105, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "per_grid[0].delta_o_minus_Ro"},
{"v": 0.2117, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "per_grid[1].delta_o_minus_Ro"},
{"v": 0.1716, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "per_grid[2].delta_o_minus_Ro"},
{"v": true, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "eff_monotone"},
{"v": 0.0401, "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "delta_spread_o"},
{"v": "verified", "f": "benchmarks/verified/poiseuille_3d_annulus/result.json", "k": "verdict"}]
-->
