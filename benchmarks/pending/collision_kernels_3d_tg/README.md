# W5-C 碰撞核覆盖基准 — 3D Taylor-Green 衰减

对库碰撞核矩阵中此前零定量基准的 6 个高性能核逐个建立 3D Taylor-Green 涡衰减定量基准，
协议与已验证先例 `benchmarks/verified/taylor_green_3d/`（D3Q19 BGK）一致，仅替换碰撞核与
对应格子；另附 collide_bgk27 参照对照（诊断用）。源钉定只读 worktree
`/nfs/wangxi/worktrees/bm_w5`（main @ cf5709db3c）。

## 判定总表

| 核 | 格子 | err_E 链 N=64/96/128 (%) | 单调↓ | R²≥0.999 | |err|≤3% | 判定 |
|---|---|---|---|---|---|---|
| collide_cumulant_d3q27 | D3Q27 | +0.1825 / +0.3738 / +0.4122 | 否 | 是 | 是 | FAIL（单调条款） |
| collide_cascaded_d3q27 | D3Q27 | +0.1826 / +0.3738 / +0.4122 | 否 | 是 | 是 | FAIL（单调条款） |
| collide_mrt27 | D3Q27 | +0.2146 / +0.4065 / +0.4232 | 否 | 是 | 是 | FAIL（单调条款） |
| collide_kbc_d3q19 | D3Q19 | +8.5904 / −16.1851 / −36.5120 | 否 | 否(N=64) | 否 | **FAIL（实质）** |
| collide_trt27 | D3Q27 | +0.3777 / +0.4328 / +0.4713 | 否 | 是 | 是 | FAIL（单调条款） |
| collide_rlbm27 | D3Q27 | +0.3118 / +0.3688 / +0.4030 | 否 | 是 | 是 | FAIL（单调条款） |
| collide_bgk27（对照） | D3Q27 | +0.3137 / +0.3347 / +0.3263 | 否 | 是 | 是 | （对照同样不单调） |

判定条款（先例 judge() 语义，预注册于 NOTES.md）：全部档 |err_E|≤3% 且 |err_E| 随 N 严格
单调下降 且 R²≥0.999 且 err_vel≤3% 且 ≥2 档。五核全部满足容差与指数性（每档 |err_E|≤0.48%，
R²≥0.999995，e0 实测=U0²/8 到 6 位），仅不满足单调下降条款；同款协议下 collide_bgk27
对照链 +0.3137→+0.3347→+0.3263 亦不单调（D3Q19 先例 BGK 为 +0.309→+0.279→+0.250 单调），
即单调条款在 D3Q27 格子 + 固定 Re 的 τ 阶梯（τ=0.9/1.1/1.3 随 N 增）下连裸 BGK 都不满足，
不区分核间优劣；机械判定如实记录为 FAIL。collide_kbc_d3q19 为幅度意义上的实质 FAIL（见该节；该核缺陷已于库 PR #312 修复，表内数字为旧核历史档）。

## 设置（六核共用）

- 周期域 N³，k=2π/N，ν=U0·N/Re，τ=0.5+3ν；Re=24，U0=0.05，阶梯 N=64/96/128
  （τ=0.9/1.1/1.3）。
- 初场 u=(U0·sin kx·cos ky·cos kz, −U0·cos kx·sin ky·cos kz, 0)，ρ=1，f₀=f_eq（库
  equilibrium3d/equilibrium27）；步序 collide(stream(f))（D3Q19: solver3d.stream3d，
  D3Q27: d3q27.stream27_roll）；步数 auto：E/E0→e⁻¹⁰、record_every=50、下限 2000。
- fp32（RTX 5090 cuda:5，TF32 关闭），观测每 50 步由库 macroscopic 实测 E、E 分量、u_max、w_max。
- 判据：γ_E_sim=−d(lnE)/dt（LSQ，t≥50 全部采样点）对解析 γ_E=6νk²；无修正、无重标定。

## 参考

初始场为波矢 (±k,±k,±k) 的 8 支平面波叠加（Fourier 几何事实，与格子无关），每支
|κ|²=3k² ⇒ γ_vel=3νk²、γ_E=6νk²。D3Q27 与 D3Q19 的 cs² 同为 1/3（Qian 权重
8/27、2/27、1/54、1/216），六核剪切模式弛豫率均为 1/τ（源码核对），故
ν=cs²(τ−½)=(τ−½)/3 两格子一致，D3Q27 沿用同一解析式。3D TG 非 NS 精确解：涡拉伸使
γ_sim 略高于 6νk²（先例 BGK +0.25~0.31% @Re=24），如实测量不修正。

## 各核小节

### 1. collide_cumulant_d3q27（tensorlbm.cumulant）

- **设置**：tau；omega_b=1.0、omega_odd=1.0、omega_even=1.0（库默认）。
- **参考**：γ_E=6νk²。
- **判定表**：

| N | τ | γ_sim (1/步) | γ_theory | err_E | R² |
|---|---|---|---|---|---|
| 64 | 0.9 | 7.724703e-03 | 7.710628e-03 | +0.1825% | 0.999998 |
| 96 | 1.1 | 5.159635e-03 | 5.140419e-03 | +0.3738% | 0.999996 |
| 128 | 1.3 | 3.871205e-03 | 3.855314e-03 | +0.4122% | 0.999996 |

- **数值注记**：质量漂移 ≤1.2e-7；E0 实测 3.125001e-04=U0²/8；前后半窗 γ 一致（差 <0.5%）。
  γ_sim 与 collide_cascaded_d3q27 在 5–6 位有效数字上一致（7.724703e-03 vs 7.724706e-03
  等）——与库文档自述一致：legacy "cumulant" 在此 Re=24、Ma=0.087 下其 O(Kn²) 非线性步
  不可分辨，数值上即中心矩（cascaded）碰撞。
- **复现**：`python run.py --kernel cumulant_d3q27 --device cuda:5`；`python verify.py`。

### 2. collide_cascaded_d3q27（tensorlbm.cascaded_collision）

- **设置**：tau；s_bulk=1.0、s_3=1.0、s_4=1.0（库默认；纯中心矩，2 阶 trace/偏应力分离）。
- **参考**：γ_E=6νk²。
- **判定表**：

| N | τ | γ_sim | γ_theory | err_E | R² |
|---|---|---|---|---|---|
| 64 | 0.9 | 7.724706e-03 | 7.710628e-03 | +0.1826% | 0.999998 |
| 96 | 1.1 | 5.159636e-03 | 5.140419e-03 | +0.3738% | 0.999996 |
| 128 | 1.3 | 3.871205e-03 | 3.855314e-03 | +0.4122% | 0.999996 |

- **数值注记**：质量漂移 ≤1.2e-7；E0=3.125001e-04。与 cumulant_d3q27 一致性见上节。
- **复现**：`python run.py --kernel cascaded_d3q27 --device cuda:5`。

### 3. collide_mrt27（tensorlbm.d3q27）

- **设置**：tau；s_e=1.19、s_eps=1.4、s_q=1.2（库默认，d'Humières/Lallemand–Luo 谱系；
  应力行 5–9=1/τ 定 ν，能量/ghost 行独立弛豫）。
- **参考**：γ_E=6νk²。
- **判定表**：

| N | τ | γ_sim | γ_theory | err_E | R² |
|---|---|---|---|---|---|
| 64 | 0.9 | 7.727177e-03 | 7.710628e-03 | +0.2146% | 0.999997 |
| 96 | 1.1 | 5.161316e-03 | 5.140419e-03 | +0.4065% | 0.999996 |
| 128 | 1.3 | 3.871629e-03 | 3.855314e-03 | +0.4232% | 0.999995 |

- **数值注记**：质量漂移 1.6e-5~2.4e-5（27×27 稠密矩阵变换的 fp32 舍入，比逐点核高 ~3 个
  量级，仍远低于判定相关量级）；E0 精确。
- **复现**：`python run.py --kernel mrt27 --device cuda:5`。

### 4. collide_kbc_d3q19（tensorlbm.entropic_kbc）

- **设置**：tau；max_iter=28、tol=1e-8（库默认）。
- **参考**：γ_E=6νk²（若 τ 设定粘度成立——实测不成立，见下）。
- **判定表**：

| N | τ | γ_sim | γ_theory | err_E | R² |
|---|---|---|---|---|---|
| 64 | 0.9 | 8.372999e-03 | 7.710628e-03 | +8.5904% | 0.978800 |
| 96 | 1.1 | 4.308438e-03 | 5.140419e-03 | −16.1851% | 0.999990 |
| 128 | 1.3 | 2.447663e-03 | 3.855314e-03 | −36.5120% | 0.999971 |

- **数值注记（核缺陷）**：τ 不设定该核粘度。预审计横波（fp64，L=32/64）测得
  ν_eff≈0.166≈cs²/2=1/6，与 τ 无关（τ=0.9/1.1/1.3 → ν 偏差 +24.5%/−16.3%/−38.1%）；
  库自带 collision_viscosity_audit 模块独立复核同结论（τ=0.8：recovered 0.1758 vs
  target 0.1，+75.8%）。机理：collide_kbc_d3q19 对 H(f_eq+γs+h*) 做**无条件**逐胞极小化，
  小非平衡极限下解得 γ≈0（剪切模全弛豫、率 1 → ν=cs²·(1−½)=1/6），初值 γ₀=1−1/τ 被丢弃。
  N=64 档前后半窗 γ_sim=9.668e-3/5.741e-3，有效粘度随衰减漂移 → E(t) 非纯指数
  （R²=0.9788）；fp64 复核（预注册 fp32 敏感条款触发）：err +6.77%、R²=0.9766，结论
  对精度稳健（判定以 fp32 阶梯为准）。任何 τ 取值都无法改变 γ≈0，不做参数重调。
- **复现**：`python run.py --kernel kbc_d3q19 --device cuda:5`；fp64 复核
  `python run.py --kernel kbc_d3q19 --n 64 --dtype float64 --out diagnostics_fp64_kbc`。

**缺陷已修复（库 PR #312，2026-09-22 合并）**：构造替换为 Karlin–Bösch–
Chikatamarla 形 f* = feq + (1−1/τ)s + βh（β 逐胞在正定性域 ∩[0,1] 上熵解，
D3Q19/D3Q27 双核）。修复后核在本同款协议下：单步剪切弛豫比 == 1−1/τ
（84 组合最差差 1.1e-14）；TG err_E **−0.675 → −0.358 → −0.161%**（N=64/96/128，
严格单调）；collision_viscosity_audit 从 withheld（>20%）转 admitted 0.09%。
本节判定表与数字为旧核（修复前）历史档。

### 5. collide_trt27（tensorlbm.d3q27）

- **设置**：tau_plus=τ；λ=3/16（库默认；Ginzburg 2008 魔法参数。周期体衰减无壁，Λ 只影响
  ghost/瞬态模，取库默认即测现役配置）。
- **参考**：γ_E=6νk²（ν 由 τ₊ 设定：ν=(τ₊−½)/3）。
- **判定表**：

| N | τ₊ | γ_sim | γ_theory | err_E | R² |
|---|---|---|---|---|---|
| 64 | 0.9 | 7.739752e-03 | 7.710628e-03 | +0.3777% | 0.999996 |
| 96 | 1.1 | 5.162669e-03 | 5.140419e-03 | +0.4328% | 0.999996 |
| 128 | 1.3 | 3.873484e-03 | 3.855314e-03 | +0.4713% | 0.999995 |

- **数值注记**：N=64 前后半窗 γ=7.752e-3/7.760e-3（微升，Λ 相关 ghost 瞬态贡献），其余档
  同级；质量漂移 ≤1.3e-5。六核中 err 地板最高（+0.38~0.47%），仍 ≤3% 的 1/6。
- **复现**：`python run.py --kernel trt27 --device cuda:5`。

### 6. collide_rlbm27（tensorlbm.d3q27）

- **设置**：tau（正则化 BGK：非平衡分布投影到二阶 Hermite 子空间后按 (1−1/τ) 弛豫）。
- **参考**：γ_E=6νk²。
- **判定表**：

| N | τ | γ_sim | γ_theory | err_E | R² |
|---|---|---|---|---|---|
| 64 | 0.9 | 7.734667e-03 | 7.710628e-03 | +0.3118% | 0.999997 |
| 96 | 1.1 | 5.159375e-03 | 5.140419e-03 | +0.3688% | 0.999996 |
| 128 | 1.3 | 3.870851e-03 | 3.855314e-03 | +0.4030% | 0.999995 |

- **数值注记**：N=64 err +0.3118% 与 bgk27 对照 +0.3137% 一致（Hermite 投影在此弱非平衡
  区不改变流体力学模）；质量漂移 ≤1.6e-5。
- **复现**：`python run.py --kernel rlbm27 --device cuda:5`。

### 对照：collide_bgk27（诊断参照，非预注册核）

与六核同协议的 D3Q27 裸 BGK 阶梯：err +0.3137/+0.3347/+0.3263（非单调），R²≥0.999995。
作用：五核的 err 趋势（随 N 微升）在裸 BGK 上同样出现 ⇒ 该趋势由 D3Q27 格子离散项与
固定 Re 阶梯的 τ 增长（0.9→1.3）耦合主导（预审计横波已测各核格子项 ∝k²·f(τ)，f(τ)
随 τ 增长快于 k² 随 N 缩小），非各核缺陷；单调下降条款在 D3Q19 先例通过、在 D3Q27 此
协议下对包括 BGK 在内的全部核不成立。

## 核缺陷与基础设施清单

1. **collide_kbc_d3q19 粘度不由 τ 设定**（实质缺陷）：熵求解无条件极小化 → γ≈0 →
   ν_eff≈1/6 恒定；TG 三档 +8.59/−16.19/−36.51%，N=64 非纯指数。两套独立粘度审计
   （本工作 audit_viscosity.py + 库 collision_viscosity_audit）互证。修复方向（供参考，
   未实施）：γ 求解应约束于 H(f*)≤H(f) 与 γ₀ 邻域，或按 Karlin–Bösch–Chikatamarla
   原式以 β 显式设定剪切弛豫。
2. **collide_cumulant_d3q27 与 collide_cascaded_d3q27 在层流弱压缩区数值恒等**（γ_sim
   5–6 位一致）：与库文档 O(Kn²) 警示一致，两核基准互为旁证；若需区分两者须高 Ma/高 Re
   工况。
3. **协议项**：先例单调下降条款不可迁移至 D3Q27+固定 Re τ 阶梯（bgk27 对照亦不单调）；
   五核 err 全部 ≤0.48%（≥6× 裕度）且指数性完美。
4. **基础设施**：无缺口——D3Q27 stream27_roll（周期）、equilibrium27、macroscopic27 及
   六核碰撞函数齐备，本基准全程只调库函数。

## 复现

```bash
cd /nfs/wangxi/runs/bm_widen_w5_20260921/kernels
/nfs/wangxi/venvs/tensorlbm/bin/python run.py --kernel all --device cuda:5   # 六核阶梯
/nfs/wangxi/venvs/tensorlbm/bin/python run.py --kernel bgk27 --device cuda:5 # 对照
/nfs/wangxi/venvs/tensorlbm/bin/python verify.py --include-diagnostics      # 独立复算
```

工件：result.json（含 env/protocol/每核 cases 与判定）、case_<kernel>_N<n>.json、
energy_history_<kernel>_N<n>.csv（E(t) 序列与拟合窗原始数据）、diagnostics_fp64_kbc/
（KBC fp64 复核）、audit_viscosity.json / audit_extra.json（预注册粘度审计）、audit_viscosity.py / audit_extra.py。完整过程工件（预注册
NOTES、时间戳链、诊断拉取脚本）留服务器暂存目录。
