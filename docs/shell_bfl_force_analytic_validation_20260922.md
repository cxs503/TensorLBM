# 壳层 BFL 力链的解析基准验证 — 调研 + 可复用性 + CPU 实测

日期 2026-09-22 · 工作区 `/root/TensorLBM_feat2` @ `0b06731` (feat2-octree-integrated,
已合并 origin/main 94 提交) · 目标：验证集成壳层 BFL 力链的**时变精度**，
定位"集成力偏高 2.2x"（集成 R6 d_max=2 Cd 2.77 vs 单卡收敛 1.24）。

---

## 1. 基准调研

### 1.1 `benchmarks/verified/taylor_couette`（8 文件，D2Q9，2D r-θ 平面）

| 项 | 内容 |
|---|---|
| 驱动 | `run.py` 三子命令 `single --ri N --out x.json` / `scan --out-dir D --ri 24 48 96` / `aggregate`；`sys.path` 内置，`--device cuda/cpu`，`--re-e-i/--nu/--eta` |
| 解析解 | 稳态轴对称环隙 Couette `u_θ(r)=A r + B/r`，`A=(Ω_o r_o²−Ω_i r_i²)/(r_o²−r_i²)`，`B=r_i²r_o²(Ω_i−Ω_o)/(r_o²−r_i²)`，`Ω_o=0`（README L8-12） |
| 观测量 | ① 方位角分 bin 剖面；② **壁面力矩/单位深度** `M=4πμL·B`，`L=1` |
| 误差算法 | `analyse_profile()`: 对精确解族 `A r+B/r` 在中心 bin 做加权（格子数）二参数最小二乘 → 由**无滑移条件反演有效半径** `r_o,eff=√(−B/A)`、`r_i,eff=√(B/(Ω_i−A))`；主指标 = 逐 bin 剖面对**有效几何**解析解的最大/加权 L2 相对误差，绝对归一 `U_max^ref=Ω_i r_i,eff`，中心区 `u_ref>0.2U_max^ref` |
| 力矩通道 | `interface_wall_torque()`（**约定无关**：逐边界链接取真实离场种群 `g[q,x_s]c_q` 与真实回场种群 `g[q̄,x_f]c_q̄`，链接中点力臂求和）为**主通道**；`compute_obstacle_forces` 全湿格 plain MEM 为次通道；`d3q27.moving_wall_linkwise_me_force_torque`（理想半程反射）只作披露 |
| 数值 | Re_i=20 ≪ Re_crit(η=0.5)≈68.2；τ=0.65；n=99/195/387；40k/161k/645k 步；末 1000 步时间平均 |
| 结果 | 剖面 max **0.746/0.363/0.152%**（单调收敛）；力矩接口通量 **≤0.73%**、两壁反号平衡 ≤0.23%；δ_i=−0.129/−0.112/−0.102（格无关常数） |
| ⚠️关键披露 5 | `d3q27.moving_wall_linkwise_me_force_torque` 在本方案上**系统性高估力矩 ~2.06/2.14/2.14×**；根因＝反弹在 post-stream、collide 在全部格子（含固体格）上运行 → 回场种群是**被 BGK 弛豫过的镜像**，理想半程反射模型不再成立；倍数随 τ 漂移（0.65→0.74 时 2.06→1.69） |

**这是本任务最重要的线索**：`result.json` 里 `T_lw_inner_ratio_disclosed` = **2.061 / 2.139 / 2.136**
与"集成力偏高 2.2x"同一量级，且机理（post-stream 反弹 + 固体格参与碰撞）与壳层 BFL 完全同构。

### 1.2 `benchmarks/verified/womersley`（5 文件，D2Q9，时变解析）

| 项 | 内容 |
|---|---|
| 解析解 | 平面通道谐波驱动 `a(t)=a0 cos(ωt)`，相量 `U(y)=(a0/iω)[1−cosh(λy′)/cosh(λ)]`，`λ=√i·α`，`α=(H/2)√(ω/ν)`（numpy 复数实现，无 scipy；PDE 残差 ~1e-9，ω→0 退回抛物线核对到 1e-12） |
| 驱动/壁面 | x 周期；**pre-stream 半程 BB**（`f_pre[OPPOSITE]` 写在壁行，碰撞前）→ `H_eff=ny−2`，无滑移面在 y=0.5/ny−1.5；体力用库 `turbulent_channel._apply_body_force_2d(f,a_t)`，步序 collide→半程BB→force→stream |
| 误差算法 | ① 8 相位剖面 L2 ≤3%（**固定尺度**归一 `max_t'‖u_ref(t')‖`，因换向相位瞬时相对 L2 病态）；② 逐点 max/u_peak ≤3%；③ **逐行复振幅** `|U_num/U_ana−1|` ≤3%（`|U_ana|≥0.1u_peak`）；④ H→2H 单调 |
| 结果 | 4/4 通过：L2 0.110/0.046%（α4）、0.389/0.101%（α8）单调；复振幅 0.467/0.144%、2.900/0.450% |
| 复振幅提取法 | `S_c=Σ ux·cos(ωt)`、`S_s=Σ ux·sin(ωt)`，`U=(2/n_meas)(S_c − i S_s)` —— **时变 BFL 验证可直接移植的"单频锁相"手法** |

（同族还有 `stokes_second_problem`（振荡平板 + Zou-He 移动盖 + specular 远场，复振幅 max 0.106%→0.023%）与
`startup_poiseuille`（起动流级数解），都属于"解析解 + 时间历程/相位"范式。）

### 1.3 这些基准的角色

- 三者都在 `benchmarks/verified/`，判定 `VERIFIED`，且**完全不含手写内核**（公共入口，grep 自检零匹配）。
- 三者都是 **单网格/直线驱动**（D2Q9 均匀网格，壁面是行/列），**都不触碰 octree 壳层 BFL**。
  → 因此它们能验证"壁面反弹 + 力记账"的**范式**，但**不能直接覆盖壳层 BFL 力链**。

---

## 2. 可复用性评估：能否用相同机制（BFL 壳层）跑 TC？

**结论：不能"直接照搬脚本"，但"方法论"可复用，且 TC 的披露 5 是直击要害的先例。**

| 维度 | TC 基准 | 集成壳层 BFL | 能否复用 |
|---|---|---|---|
| 格子 | D2Q9（`solver.collide_bgk/stream`） | D3Q27（`collide_cumulant_d3q27` + octree 壳） | ✗ 需换格子 |
| 几何 | 方域数字环隙，`cylinder_mask` 楼梯壁 | octree 壳 + BFL `q_field` 贴体插值 | ✗ 机制不同（但 `build_octree_shell(..., inside_fn=)` 支持**任意体内判据**，可造圆柱/环隙） |
| 运动壁 | `rotating_cylinder.moving_wall_bounce_back`（Ladd 半程，整障碍掩码） | `bfl_apply_gather(wall_velocity=...)`（Bouzidi 运动壁项 `−6wρ(c·u_w)` 线性 / `−(3/q)wρ(c·u_w)` 二次） | ✓ **同一物理分支**：旋转壁 |
| 力矩记账 | **接口通量**（真实离场+真实回场 × 中点力臂），约定无关 | `bfl_apply_gather(return_force=True)`：`Σ_links c_d(f_prev[d]+f_bc)` | ✓ **力链公式同构**，可直接对拍 |
| 解析解 | `A r+B/r` 精确解族 | 需换到壳层几何（球/圆柱） | 部分：球壳用"刚性旋转 + 共转壁 ⇒ 解析力矩恒 0"作**零检验**；圆柱壳可用 `inside_fn` 造环隙后套 `A r+B/r` |
| 误差算法 | R_eff 剖面反演 + 力矩交叉验证 | — | ✓ **R_eff 反演思路**、**力矩独立通道**、**单调收敛判定**可直接移植 |
| 运行成本 | GPU 6，40k–645k 步 | — | 壳层全解析场探针 **~60s CPU**（本次实测） |

**可复用清单（推荐）**
1. **解析零解/精确解族**思路：把 Couette `A r+B/r` 换成壳层几何的解析场（刚性旋转 / 球-圆柱 Couette）。
2. **约定无关的接口通量记账**：TC 用它把"库理想反射原语高估 2×"钉死；壳层 BFL 应照做（见 §3）。
3. **复振幅/锁相提取（womersley）**：用于验证**时变** BFL 力（壳层运动壁 ramp、`_ramp_activation`）——
   直接给壳层壁速一个 `u_w(t)=u_w0 cos(ωt)`，用 `Σ F(t)e^{iωt}` 取复力幅值，与解析 `4πμB` 复幅对比。
4. **单调收敛判定**（mesh refine 误差必须下降）：本次 CPU 验证正是靠它抓到了 d_max=1→2 的**反收敛**。

---

## 3. CPU 小验证（已实现并运行）

新增脚本：`scripts/validate_shell_bfl_force_analytic_cpu.py`
（输出工件：`shell_bfl_force_analytic_report.json`；运行 `python scripts/validate_shell_bfl_force_analytic_cpu.py --dmax 1 2 --omega 0.002 0.004`，CPU ~60s）

在**真实 octree 壳层**（`build_octree_shell`, R=12, bl=6, D3Q27）上，只做**单次力链求值**，
输入**解析写死的冻结流场**，从而把力链与耦合动力学解耦：

| 用例 | 解析场 | 解析答案 |
|---|---|---|
| **A** | 均匀平衡来流 `u=U x̂`，静止壁 | 纯 bounce-back 链接和 `Σ c_d(f_d+f_opp)` **恒等于 0**（每叶 `Σ_d 2wρc_d=0`，对任意闭合面成立） |
| **B** | 绕球轴**刚性旋转**平衡场，**共转运动壁** `u_w=Ω ẑ×(r−r_c)` | 应变率恒 0 ⇒ 粘性力矩 **T_z ≡ 0**，净力 ≡ 0（任意 Ω、任意分辨率）——即 Taylor-Couette 的旋转壁类 |
| **C** | 同一链接集/流场上，壳层 BFL vs 库理想半程反射 MEM `f_r=f_o−6wρ(c·u_w)` | 量出约定间距（复现 TC 披露 5 的 ~2x 现象） |

### 实测结果（R=12, U=0.06, Ω=0.002/0.004, D3Q27）

```
[A] 均匀来流 u=0.06, 静止壁, d_max=1   n_leaf=200184  n_links=66416
    shell BFL          Fx = +7.037005e+01
    plain-BB (解析 0)  Fx = +0.000000e+00   <-- 链接集/权重记账干净
    level 1: n_links=66416  Fx=+7.0370e+01  per-link=+1.0595e-03
[A] 均匀来流 u=0.06, 静止壁, d_max=2   n_leaf=279488  n_links=266312
    shell BFL          Fx = +2.613118e+02     <-- 4.01x 链接 → 3.71x 力
    plain-BB (解析 0)  Fx = +0.000000e+00
    level 2: n_links=266312 Fx=+2.6131e+02  per-link=+9.8122e-04  (与 d1 每链几乎相同!)
    level-rescaled Fx = +6.532796e+01   <-- 折入 2^-2(L-1) 后 → 6.53 vs 7.04 (相差 7.5%)

[B] 刚性旋转 Ω=0.002 共转壁 (u_wall=0.024, Ma=0.042), d_max=1
    shell BFL   Fx≈-7e-15(过)   Tz = -3.909046e+02      <-- 解析: 0  ⇒ 不过!
    plain-BB    Fx≈0            Tz = -1.865e-14 (过)
    ideal-ref.  Fx≈0            Tz = -1.231e-01  (≈过)
[B] Ω=0.004:  shell Tz = -7.818093e+02  ⇒ 严格 ∝ Ω (比值 2.0000)
[B] d_max=2, Ω=0.002: shell Tz = -1.609216e+03  (4.01x 链接 → 4.12x 力矩)
    plain-BB  Tz ≈ -8e-13 (过)
```

### 实测结论（三条）

1. **链接集/权重记账本身是干净的**：均匀解析场上 plain-BB 力矩/力**精确为 0**（float64 舍入级），
   跨 rank 分片求和也精确（`_bfl_force_chain_audit.py`：Fx ratio integrated/single = **1.0000**）。
   → 2.2x **不是**分片/邻居表/账本求和的问题。

2. **d_max=1→2 时壳层力不满足细化不变性（反收敛）**：链接数 ×4.01，力 ×3.71，**每链贡献几乎不变**
   （1.06e-3 → 9.81e-4）。折入每层对流尺度 `2^-2(L-1) = (dx_L/dx_1)²` 后，d2 值 65.33 与 d1 值 70.37
   只差 **7.5%**。
   → **壳层账本把 depth-2 叶（dx=0.25）的链接冲量按 depth-1（dx=0.5）同一权重累加**，
   `ShellForceLedger` / `leaf_force_weights` 只带**时间**权重（子步数 2^-(d_max−d_leaf)、除以 2^d_max），
   **没有携带空间因子 dx⁴/dt²（= dx²）**；而 `force.convert_leaf_force_to_l1()` 恰恰假定存在**单一** `dx_leaf`，
   混合层级时无法成立。
   → 这是与"集成 R6 **d2** 2.77 vs 单卡 1.24"（≈2.2x）**量级与方向一致**的**具体机制**。

3. **共转旋转壁的解析零力矩被破坏，且残差线性于 Ω、随细化增长**：
   壳层 `Tz = −390.9 (Ω=0.002) → −781.8 (Ω=0.004)`，严格 ∝ Ω（非 O(u³) 压缩性伪影），
   d2 原始值 −1609（×4.12，≈链接数比）；而 plain-BB 与库理想反射 MEM 都≈0。
   把运动壁修正项 `Σ_links r × [−6wρ(c·u_w)c]` 单独算出 = **Tz = −656.03**（力=1e-16）——
   即**运动壁项在旋转曲面上不是"无力矩"的**（连续极限 `n̂·u_w=0` 才使其为 0；27 向 `c_d` 不严格沿法向 ⇒ 残差不随细化消失、甚至随链接数增长）。
   → **旋转壁（正是 TC 的物理类）分支未被任何现有验证覆盖**：单网格球基准
   （`sphere_bfl_control_volume`）是**静止球绕流**，压根不激励 `wall_velocity` 分支。
   另外 `bfl.py` 线性分支的修正系数写作 `−6 w ρ(c·u_w)`（**无 q**）、二次分支 `−(3/q)wρ(c·u_w)`，
   与仓库自带 `wall_surface_bfl.py` docstring 的标准 Bouzidi 公式（线性项含 q）不一致 —— 待独立复核。

---

## 4. 能否用于定位"集成力偏高"？—— 是，给出具体做法

**能定位，但要点是"复用范式"而非"复用脚本"**（TC 是 D2Q9 直壁，壳层是 D3Q27 贴体）。三条可执行路径：

### 路径 1（已做，成本最低）：解析场冻结探针 + 细化不变性 —— **直接命中 d_max 权重问题**
- 保留 `scripts/validate_shell_bfl_force_analytic_cpu.py`，把 `d_max` 当作判定轴：
  **任一 d_max 下，解析场的力/力矩必须细化不变**（plain-BB 已证明记账干净，故偏差只能来自 BFL 项）。
  当前 d1→d2 力 ×3.71、力矩 ×4.12 ⇒ **不合格**。
- 修复方向（按嫌疑排序）：
  1. `ShellForceLedger.mem_force` / `leaf_force_weights` 增加**每层空间因子** `2^-2(L-1)`（或统一把各层冲量换算到根格单位再累加）；
  2. 或在 `bfl.py` 内把 `link` 乘以 `(dx_leaf/dx_ref)²`；
  3. 检查 `convert_leaf_force_to_l1(dx_leaf, dt_leaf)` 在混合层级下的调用点（是否被误用单值 `dx_leaf`）。
- 快速判定：在集成脚本里把 **d_max 从 2 降到 1** 跑同配置，若 Cd 从 2.77 掉到 ~1.2-1.5 区间，
  即证实"2.2x 是 d2 壳层权重造成的"。

### 路径 2：把 TC 的**力矩记账**搬到壳层（约定无关对拍）
- 照 `interface_wall_torque()` 写一个壳层版：对每条 `bfl_mask` 链接取
  `真实离场 f_prev[d](leaf)` 与 `真实回场 f_bc`（即 `f_out[opp[d]]`，可由
  `bfl_apply_gather` 返回的 `f_out` 直接读），力 `= c_opp[d]·f_bc − c_d·f_prev[d]`，力臂取叶心/链接中点。
- 与 `return_force` 的 `c_d(f_prev+f_bc)` **逐链接对拍**（符号 + 幅度）。若两者差常数倍/符号，
  则是约定错误；若两者一致但都与解析不符，则回到路径 1 的权重问题。
- 这正对应 TC 披露 5 的判据：**"理想反射模型 vs 真实回场"**。

### 路径 3：时变精度（BFL 力链的时变部分）—— 用 womersley 的锁相法
- 壳层运动壁已有启动 ramp (`bfl_ramp_wall_velocity`, `_ramp_activation`)；给壁速
  `u_w(t)=u_w0 cos(ωt)`，对壳层 MEM 力求 `F̂ = (2/n)Σ F(t)e^{-iωt}`，与解析复幅（旋转壁的
  `M=4πμB`、或球壳的解析复力矩）对比 —— 这正是"验证 BFL 力链**时变**精度"的直接手段，
  且 `womersley/run.py` 的 `S_c/S_s` 累加代码可原样移植（含"换向相位病态 ⇒ 用固定尺度归一"的陷阱记录）。
- 注意同时记录**收敛轴**：α（Womersley 数）与 d_max 双轴，只有当误差随两者都下降才算过关；
  本次结果说明 **d_max 轴现在是反的**。

### 现有基准的直接价值
- `taylor_couette`：给出"旋转壁 + 独立力矩通道 + 约定无关记账"的**判定模板**，且**已预先证明**
  库的理想反射链接 MEM 原语在同构方案上高估 ~2.1x（披露 5）——为"2.2x"提供了**同量级先例与机理**。
- `womersley`/`stokes_second_problem`：给出**时变**解析验证的完整工程范式（截断瞬态跳过、整周期测量、
  复振幅提取、固定尺度 L2 归一）。
- **但三条基准都跑在均匀网格直壁 D2Q9 上，均不覆盖 octree 壳层 BFL** ⇒ 需要一个**新的壳层解析验证**
  （本次 CPU 脚本即最小可行版），并把它加入 `benchmarks/verified/` 的判定族（含 d_max 收敛轴）。

---

## 工件

| 文件 | 说明 |
|---|---|
| `scripts/validate_shell_bfl_force_analytic_cpu.py` | 新增：壳层 BFL 力链解析场 CPU 验证（A 均匀场 plain-BB 零基准 + d_max 细化不变性；B 刚性旋转共转壁解析零力矩；C 约定间距；含每层力分解与 `2^-2(L-1)` 重标定） |
| `shell_bfl_force_analytic_report.json` | 上述脚本输出（逐用例 F/T、plain-BB 参照、理想反射参照、每层 n_links/F、level-rescaled 值） |
| `_bfl_force_chain_audit.py`（既有） | 复核用：均匀解析场上 single-card vs integrated vs 16-rank 分片，Fx ratio 全 1.0000 |

## 复现

```bash
cd /root/TensorLBM_feat2
python scripts/validate_shell_bfl_force_analytic_cpu.py --dmax 1 2 --omega 0.002 0.004   # CPU ~60s
python _bfl_force_chain_audit.py                                                          # CPU ~1min
```