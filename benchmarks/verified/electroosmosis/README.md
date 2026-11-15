# 电渗流（EOF）狭缝微通道塞流 — D2Q9 LBM vs Poisson–Boltzmann 精确解

**判定: 达标 (VERIFIED)** — 6/6 配置 LBM 误差 ≤ 0.42%（阈值 3%），两独立离散（LBM / FD）互证，
3/3 负控被拒（19.7% / 200.4% / 100.0%）。最细档：κH=20, ny=129 → **0.103%**。

> 本用例新增的能力：**电渗流动的流场求解**（EDL 电荷密度驱动的体力 → Navier–Stokes），
> 即电动力学中「输运侧」之外一直缺失的「流动侧」。此前 `run_electroosmosis` 并未求解任何流动方程。

---

## 0. 缺陷记录（本用例的前置发现，honest）

审计 `src/tensorlbm/verification/cdr_pnp/pnp_solver.py::run_electroosmosis`（gate C3 的唯一被调方）
发现两处「永不失败」的验证：

1. **循环验证（MOCK）**：函数直接把解析恒等式赋给"数值解"——
   `u_num = u_hs * (1 - psi/zeta)`，随后把 `u_num` 的均值与 `u_hs` 比较，把差值报成
   `u_error`（"数值误差"）。即：拿解析式与解析式比，任何时刻都会"通过"，不可能因物理原因失败。
2. **空腔度量（vacuous）**：默认配置 H=1 μm、c_bulk=1e-3 mol/m³ → κH = 3.29，
   通道内**不存在**塞流核心区（判据要求 `y > 5λ_D`，而 5λ_D = 1.52H），
   代码 `bulk_start = min(int(5*kappa_inv/dy), ny-1)` 落到 `ny-1`，
   `flatness` 走 fallback 分支返回 **0.0** — 阈值 2% 恒过。

修复方式（本 PR）：把「解析式自比」换成真实数值解，gate 重新给出判别力；缺陷由
`tests/test_eof_lbm_verification.py` 锁死（含"非循环"判据 `lbm_vs_fd > 0`：
循环实现的两个离散必然逐位相同）。

---

## 1. 物理背景与解析参考

对称狭缝，两壁同 ζ 电位。Debye–Hückel（小 ζ）极限下 EDL 电位

    ψ(y) = ζ·cosh((y − H/2)/λ_D) / cosh(H/(2λ_D)),   ψ'' = ψ/λ_D²

电荷密度由 Poisson 方程给出 `ρ_e = −ε ψ''`。无外加压差、仅有电渗体力的稳态 NS：

    μ u'' = −ρ_e E = ε E ψ''

右端是全微分，因此对**任意**（不必是 D-H 线性化）EDL 结构精确可积：

    u(y) = (ε E / μ)(ψ(y) − ζ) = u_hs·[1 − ψ(y)/ζ],   u_hs = −ε ζ E / μ

两壁 `u(0)=u(H)=0` 自动满足（等 ζ 壁）。`u_hs` 即 Helmholtz–Smoluchowski 速度。
流场解 u **完全独立于**该式（LBM / FD 各自求解），故 `u_hs[1−ψ/ζ]` 是合法的精确参考。

无量纲控制量 `κH = H/λ_D`：κH 越大，核心区越接近塞流、`u_core/u_hs → 1`。

## 2. 数值方法

| 项 | 设置 |
|---|---|
| 参考解 | 精确 PB 一致剖面 `u_hs[1−ψ/ζ]`（ζ 归一化） |
| 求解器 A | D2Q9 BGK，τ=1.0 → ν=1/6，壁法向体力（cosh 型，幅值 `a0 = u_hs·ν/λ_D²`） |
| 壁面 | 半程反弹（half-way bounce-back），壁位于首/末流体节点外 0.5 格 → `H_eff = ny−2` |
| 求解器 B | 直接 FD（Thomas 三对角）解同一离散稳态 Stokes 问题，独立实现 |
| 力项 | 一阶 Guo 型源项 `S_i = 3 w_i ρ c_ix a(y)`（与 `tensorlbm.turbulent_channel` 同约定，Σ c_ix S_i = ρa） |
| 时间推进 | 至稳态：每 500 步 max|Δu| < 1e-12（最大 3e5 步） |
| 格点 | ny ∈ {65,129}，κH ∈ {5,10,20}；λ_D 分辨率 3.15–25.4 节点/λ_D |
| 尺度 | 全晶格单位，`u_hs_lat = 0.02` → Ma ≈ 0.035 |

### 踩过的坑（已锁测试）

1. **"镜面"壁而非无滑移壁**：最初把 wall-normal 分量翻转（`(c_x,c_y)→(c_x,−c_y)`）当作反弹——
   那其实是 **specular（自由滑移）** 壁。均布体力下整个剖面匀速加速 `u = a·t`，
   永不产生壁面剪切（实测 u 精确等于 `a·step`，80000 步达 8.0）。改为**全分量反转**
   `f_ī = f_i`（half-way bounce-back）后即得抛物线/塞流稳态。
   锁：`test_wall_is_noslip_not_specular`。
2. **Guo 源项 + `+F/2` 速度修正混用**：两套约定混搭会通过 `f_eq` 形成正反馈，
   动量逐步线性增长（τ=1 时每步附加 ≈0.5a）。最终采用仓库既有的一阶源项 + 无 `F/2` 修正。
3. **负控与参考解同尺度**：负控（0.8× 体力）时若把参考解也乘 0.8，线性响应会"恰好匹配"，
   负控失效。参考解必须保持**物理真值**，不得继承 force_scale/force_sign。

## 3. 计算结果（程序化注入，`result.json`）

| κH | ny | 节点/λ_D | LBM max err | LBM L2 | FD max err | LBM vs FD | 核心平坦度 | 步数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5  | 65  | 12.60 | **0.024%** | 0.021% | 0.077% | 0.101% | 0.309% | 32500 |
| 5  | 129 | 25.40 | **0.006%** | 0.005% | 0.019% | 0.025% | 0.335% | 103500 |
| 10 | 65  | 6.30  | **0.102%** | 0.088% | 0.298% | 0.388% | 0.088% | 37500 |
| 10 | 129 | 12.70 | **0.025%** | 0.021% | 0.075% | 0.099% | 0.095% | 123500 |
| 20 | 65  | 3.15  | **0.416%** | 0.374% | 1.129% | 1.433% | 0.407% | 41000 |
| 20 | 129 | 6.35  | **0.103%** | 0.092% | 0.294% | 0.382% | 0.469% | 138500 |

- 每档 LBM 与 FD 误差随 λ_D 分辨率单调下降（图 b，O(dy²) 趋势）；
- 同分辨率下 LBM 误差恒小于 FD 误差（LBM 稳态解对格的采样更优），两者互证 < 1.5%；
- 核心平坦度 0.09–0.47% ≪ 2%（薄 EDL 塞流特征），且 `u_core/u_hs = 0.9997`（<0.03% 偏离 HS 极限）。

### 负控（ny=65, κH=20）

| 控制 | max err | 是否误通过 3% |
|---|---:|:--:|
| physical（正例） | 0.42% | 通过（应通过） |
| force 0.8× | 19.67% | 否（应拒） |
| field reversed | 200.40% | 否（应拒） |
| no field | 99.99% | 否（应拒） |

## 4. 图

- `figures/eof_profiles_convergence.png`：(a) κH=5/10/20 的 u/u_hs 剖面 vs 精确解；
  (b) 误差 vs 节点/λ_D（LBM 实线、FD 虚线、3% 红线）。
- `figures/eof_edl_and_controls.png`：(c) EDL 电位 ψ/ζ 与流速剖面叠加；
  (d) 负控误差柱（log 轴）。

## 5. 误差判定

| 关键量 | 基准 | 误差 | <3% ? |
|---|---|---:|:--:|
| 剖面 max err（最细 κH=20/ny=129） | 精确 PB 剖面 | 0.103% | ✅ |
| 剖面 max err（最粗 κH=20/ny=65, 3.15 节点/λ_D） | 同上 | 0.416% | ✅ |
| 核心速度 u_core/u_hs | HS 极限 | <0.03% | ✅ |
| 核心平坦度 | <2% 塞流判据 | 0.09–0.47% | ✅ |
| LBM vs FD 独立互证 | 彼此 | 0.025–1.43% | ✅ |

**是否 <3%：是（全部 6 档 + 全部关键量）。**

## 6. 局限与适用范围（如实披露）

1. **参考解是解析恒等式**，所以本用例度量的是**离散误差**（+ 求解器正确性），
   不是与实验/外源数据对标的物理不确定度。它的判别力来自负控与两独立离散互证。
2. EDL 用 **Debye–Hückel（线性化 PB）** 剖面；高 ζ（|ζ| ≳ 50 mV，本用例恰为 −50 mV 上界）
   的 Gouy–Chapman 非线性效应未覆盖。注意：解析恒等式 `u ∝ (ψ−ζ)` 本身对非线性 ψ 也成立，
   缺的只是一个被验证过的非线性 ψ 求解链（PNP 侧已有 `run_pnp_edl`，可后续接 LBM 流场）。
3. 一阶力项（无 `F/2` 修正）在 τ 偏离 1 时有 O(τ−1/2) 偏差，本用例固定 τ=1.0。
4. 定常、单相、牛顿流体、无外加压差、稀溶液（无离子体积排斥）、壁面无滑移；
   未覆盖 AC 电场/频率响应、变 ζ、表面电导、流动电位（streaming potential）反问题。
5. `PNPConfig.c_bulk` 注释标 "mol/m³ = 1 mM" 但默认 1e-3 实为 1 μM（κ⁻¹=304 nm），
   本用例在 gate 内显式取 `c_bulk=1.0`（1 mM，κ⁻¹=9.6 nm）避开标注歧义；注释待域内确认后修正。

## 7. 复现

```bash
PYTHONPATH=src python benchmarks/verified/electroosmosis/run.py     # 全量（≈3 min, 单核）
PYTHONPATH=src python benchmarks/verified/electroosmosis/run.py --quick
PYTHONPATH=src python -m pytest tests/test_eof_lbm_verification.py -q   # 11 passed
PYTHONPATH=src python -c "from tensorlbm.verification.cdr_pnp.gates import check_c3_electroosmosis as g; print(g()['passed'])"
```

> 注意：本仓的 `tensorlbm` 可能以 dev 路径装到**其他 worktree**，复现必须显式 `PYTHONPATH=src`。

## 8. 文献/参考

- Helmholtz–Smoluchowski 电渗速度 `u = −εζE/μ`（本用例解析参考的极限形式）。
- Poisson–Boltzmann / Debye–Hückel EDL 电位（Russel, Saville & Schowalter, *Colloidal Dispersions*, 1989）。
- Guo, Zheng & Shi (2002), *Phys. Rev. E* 65, 046308 — LBM 体力项（本实现采用其一阶形式）。
- Ladd / Aidun–Lu 关于 LBM 半程反弹壁面定位（`H_eff = ny−2`，壁在 ±0.5 格）。
- 已有电动力学交付：`src/tensorlbm/verification/cdr_pnp/`（CDR + PNP 输运侧，gate C1/C2/C4/C5/C6）。