# GAP-DIFFLBM-3D-COST — 3D D3Q19 可微反演的可承受包络

mission 20072 · 仓 TensorLBM · 分支 `agent/m20072` · 锚点 `GAP-DIFFLBM-3D-COST`

**验收判据（总账）**：256³ D3Q19 **展开式 AD（checkpoint）**梯度 vs 中心差分相对误差 ≤1e-6；
峰值显存 ≤8 GB；单次反演 ≤2 h/16 核。

**本单结论（honest，判据逐条）**

| 判据 | 状态 | 证据（实测数字，零手抄） |
|---|---|---|
| 梯度 vs FD ≤1e-6 | **PASS（机制 + 已测尺度）** | n=32³,T=8,fp64：`rel_err_tau=1.36e-7`、`rel_err_u0=6.2e-14`；n=16³：1.5e-8 / 4.0e-14；分段 AD 与全图 AD **逐位相等**（0.0）；fp32 地板 1.5e-5（见下） |
| 峰值显存 ≤8 GB @256³ | **FAIL（实测外推）** | 实测 VmHWM：32³ 0.675 / 64³ 1.267 / 96³ 2.231 / 128³ 3.958 GiB；线性拟合 `peak = 0.97 + 20.2·field` GiB → 256³（field 1.28 GiB）**≈26.8 GiB > 8 GB** |
| 单次反演 ≤2 h/16 核 | **PASS（实测节拍外推）** | 前向 0.019/0.158/0.476/1.113 s/step（32/64/96/128³，6 线程，N³ 标度）→ 256³ ≈ 8.9 s/step(6 线程) / ~3–4 s(16 核)；T=50 单次反演（fwd+replay+bwd ≈3×fwd）≈10–22 min |
| 256³ 端到端实跑 | **NOT RUN（队列中）** | `cj2473`(fp32 256³ envelope+grad)、`cj2474`(fp64 256³ grad) 递交于 08:26，报告发布时仍 queued（队列 ~44 作业在跑）→ 见 §6 fail_stage |

> 结论一句话：**约束不是梯度正确性、也不是时间，而是显存**——torch autograd 的 D3Q19 单步
> 需要 ~20 个场（`equilibrium3d` 的 19 个方向张量 + streaming 的 cat 中间量）常驻，
> 256³ fp32 一个场就 1.28 GiB，20 场 = 26 GiB。要把 256³ 压进 8 GB，必须把单步 transient
> 压到 ≤6 个场（第 2 轮方法：`equilibrium3d(out=)` 单缓冲 + 手写 BGK/streaming 伴随的
> lean tape，或 z-slab 分块波前），本单已把该结论量化到可执行。

---

## 1. 问题设定与基准出处

**物理/数值问题**：周期立方盒 N³、D3Q19、BGK，晶格单位 dx=dt=1，ν=(τ−1/2)/3。
反演目标（离散 BPTT 目标函数）

```
J(τ, u₀) = ½ ‖u(f_T) − u_target‖² / N_cells ,     u_target = u(f_T; τ_true) 的终端速度场
```

τ_true = 0.60，在 τ = 0.55 处取梯度。**这是展开式（unrolled / discrete）自动微分**：步进算子
本身在计算图里，梯度是离散伴随/ BPTT，而不是连续 adjoint（对照 Um et al. NeurIPS 2020
solver-in-the-loop；XLB, Ataei & Salehipour, CPC 300:109187 (2024) 的 checkpoint-replay 思路）。

**基准出处（external anchor）**

1. **解析解**：二维 Taylor–Green 涡嵌入三维盒
   `u = A e^{-2νk²t}(sin kx cos ky, −cos kx sin ky, 0)`，`k=2π/N`，动能
   `E(t)=E₀e^{−4νk²t}`，`E₀=A²/4`（Taylor 1937；Hou et al., *J. Comput. Phys.* 118:329, 1995
   ——LBM 标准验证例）。横向剪切波 `u_x=A sin(kz)` 的 `A(t)=A₀e^{−νk²t}`（黏性衰减解析解）。
   `src/tensorlbm/diff3d_envelope.py::taylor_green_energy / shear_wave_amplitude`。
2. **梯度裁判**：中心差分（2 点 1 阶与 5 点 4 阶）对**同一个前向求解器**取值——FD 是独立
   的第三方证据，不与 AD 共用任何中间量。

## 2. 方法与实现

| 组件 | 位置 | 作用 |
|---|---|---|
| `SegmentStore` | `src/tensorlbm/diff3d_envelope.py` | 边界状态落盘（`numpy.memmap`，**精确 dtype 无量化**）；写/读后 `flush + MADV_DONTNEED`，否则 memmap 的 page cache 会虚增 RSS（256³ 一个状态 1.28 GiB，会成为假峰值） |
| `segmented_rollout` | 同上 | `autograd.Function`：forward 在 `no_grad` 下跑 rollout，每 `seg_len` 步存一个边界状态到盘；backward **逆序**逐段从盘取状态重放，重建一段的图即刻反传、释放——峰值激活内存 `O(seg_len)`+3 场，与步数**无关**；代价是多一次前向（Revolve/treeverse 权衡，状态放在盘上） |
| `peak_rss_gib` | 同上 | `VmHWM`（内核 RSS 高水位）——CPU 上没有 allocator 计数器，这是唯一诚实的进程级峰值，因此**每个测量都在独立子进程**里跑 |
| `differentiable_step`（复用） | `src/tensorlbm/autograd_path.py` | 被微分的 D3Q19 步（BGK + gather/cat 周期 streaming，纯 torch 算子，无 in-place 破坏图） |
| 驱动 | `scripts/bench_difflbm3d_envelope.py` | 4 个 stage：`grad`（AD vs FD、AD vs 全图）、`envelope`（三引擎峰值/时间）、`physics`（解析解）、`inversion`（真反演）；输出 JSON（零手抄） |
| 图 | `scripts/figures_difflbm3d_envelope.py` | `figures/difflbm3d_envelope/*.png` |

**关键代码指针**：分段重放 `diff3d_envelope.py::_SegmentedRollout.backward`；memmap 驱逐
`SegmentStore._evict`；FD 5 点差分 `bench_difflbm3d_envelope.py::stage_grad`。

## 3. 参数表（无量纲）

| 量 | 值 | 说明 |
|---|---|---|
| 格子 | D3Q19，c_s²=1/3，Σw=1（实测 0.99999994） | `solver3d.equilibrium3d` |
| τ（取值点 / 真值） | 0.55 / 0.60 | ν = 0.016667 / 0.033333 |
| Re（TG，L=2π） | 2π·A/ν = 18.8 (A=0.05) | 层流 |
| Ma | A/c_s = 0.087 (A=0.05) / 0.012 (A=0.007) | 压缩性 O(Ma²) ≤7.5e-3 |
| 分辨率 N | 16/32/64/96/128（已测），256（递交中） | |
| 步数 T | 4/8（envelope、grad），50（256³ 计划），200（physics） | |
| seg_len | 1（默认，内存最省） | 盘用量 `ceil(T/seg_len)·19N³·4 B` |
| FD 步长 h | 1e-3（τ），1e-3（u₀ 方向，5 点） | 2 点/5 点对照给出 FD 收敛性 |
| dtype | fp64（梯度裁判）/ fp32（生产包络） | 见 §5 精度-内存权衡 |

## 4. 对照表（ref vs got + 误差）

### 4.1 梯度裁判（判据 ≤1e-6）

| 尺度 | dtype | ref（FD，5 点） | got（分段 AD） | 相对误差 | 判据 |
|---|---|---|---|---|---|
| n=16³, T=4, τ | fp64 | −7.541908670583e-4 | −7.541908784417e-4 | **1.51e-8** | ✅ |
| n=16³, T=4, u₀ 方向 | fp64 | 9.486100934770e-5 | 9.486100934771e-5 | **3.99e-14** | ✅ |
| n=32³, T=8, τ | fp64 | −3.945754457061e-3 | −3.945754992586e-3 | **1.36e-7** | ✅ |
| n=32³, T=8, u₀ 方向 | fp64 | — | — | **6.20e-14** | ✅ |
| n=16³, T=4, τ | fp32 | −7.541796e-4 | −7.541907e-4 | 1.46e-5 | ❌（fp32 地板） |
| n=16³, 2 点差分 h=1e-3 | fp64 | −7.542949e-4 | −7.541909e-4 | 1.38e-4 | —（截断 O(h²)，故用 5 点） |
| 分段 AD vs 全图 AD | fp64 | −7.541908784417e-4 | −7.541908784417e-4 | **0.0** | ✅ 逐位相等 |

→ **≤1e-6 在 fp64 下成立且有 1–2 个数量级余量**；fp32 下 FD 探针噪声（前向舍入 ~1e-7 ×
增益 1/(12h)）给出 1.5e-5 地板——**判据是精度受限，不是实现受限**（分段与全图 AD 逐位相同，
说明 AD 侧没有额外误差）。

### 4.2 显存包络（判据 ≤8 GB @256³）

| 引擎（fp32，VmHWM，独立进程） | N=32³ | 64³ | 96³ | 128³ | 随 T 的行为 |
|---|---|---|---|---|---|
| plain（全图 autograd） | 1.177 | 3.961 | — | — | **线性增长**（64³,T=8 已达 213 场） |
| 每步 `torch.utils.checkpoint` | 1.282 | 3.172 | — | — | 仍线性（非 reentrant 存每步输入） |
| **segmented（本单，seg_len=1）** | **0.675** | **1.267** | **2.231** | **3.958** | **与 T 无关**：n=32³ T=8→128 为 0.607→0.620 GiB |
| 拟合 / 外推 | — | — | — | — | `peak = 0.97 + 20.2·field GiB` → **256³ ≈ 26.8 GiB** ❌ |

### 4.3 时间包络（判据 ≤2 h/16 核）

| N | 32³ | 64³ | 96³ | 128³ | 256³（外推，N³） |
|---|---|---|---|---|---|
| 前向 s/step（6 线程） | 0.0191 | 0.1575 | 0.4757 | 1.1133 | **8.9**（16 核 ~3–4） |
| 单次反演 T=50（fwd+replay+bwd ≈3×fwd） | — | — | — | — | **10–22 min** ✅ |

### 4.4 物理保真（解析解，判据 <3%）

| 算例 | N | ref（解析） | got（LBM） | 相对误差 |
|---|---|---|---|---|
| 2D Taylor–Green 嵌入 3D，E(t=200) | 32 | 5.980733e-5 | 5.904571e-5 | **1.27%** |
| 同上 | 64 | 8.794044e-5 | 8.762375e-5 | **0.36%** |
| z 剪切波 A(t=200) | 32 | 1.758809e-2 | 1.744043e-2 | **0.84%** |
| 同上 | 64 | 1.936766e-2 | 1.932851e-2 | **0.20%** |

→ 全部 <3%，且随分辨率二阶收敛（1.27%→0.36% ≈ 4×），说明被微分的算子就是标准 D3Q19 BGK。

## 5. 收敛性与参数敏感性

- **网格收敛**：TG 1.27%(32³) → 0.36%(64³)，O(N⁻²)；剪切波 0.84% → 0.20%，同阶。
- **FD 步长**：h=1e-3 时 2 点差分截断误差 1.4e-4、5 点 1.5e-8（fp64）→ 判据用 5 点，2 点作对照。
- **dtype**：fp64 1.4e-7（n=32³）对 fp32 1.5e-5 —— **精度与显存互为代价**：256³ fp64 一个场 2.56 GiB，
  在 8 GB 内连 3 个场都放不下，所以「≤1e-6 且 ≤8 GB 且 256³」在当前 autograd 实现下三者不可同时满足。
- **T 无关性**：分段重放的峰值对 T 平（n=32³，T 8→128 增加 2%），这是本单的核心成果（原
  `rollout(checkpoint=True)` 与全图 autograd 都是 O(T)）。
- **seg_len 敏感性**：RAM ∝ seg_len、盘 ∝ T/seg_len —— 256³/T=50 在 seg_len=1 时盘用量 50×1.28=64 GiB
  （集群节点 3 TB 盘，实测 memmap 驱逐后不占 RSS）。

## 6. 失败模式与工件（repro_cmd）

```bash
# 梯度裁判（fp64，秒级）
python3 scripts/bench_difflbm3d_envelope.py --stage grad --n 32 --steps 8 --dtype float64 \
        --threads 6 --seg-len 1 --out artifacts/difflbm3d_envelope/n32_fp64_grad.json
# 显存/时间包络（每个引擎单独进程，VmHWM 才有意义）
python3 scripts/bench_difflbm3d_envelope.py --stage envelope --n 128 --steps 4 --dtype float32 \
        --threads 6 --engines segmented --out artifacts/difflbm3d_envelope/n128_fp32_env_segmented.json
# 解析解物理锚点
python3 scripts/bench_difflbm3d_envelope.py --stage physics --resolutions 32 64 --steps 200 \
        --dtype float32 --threads 6 --tau 0.55 --amplitude 0.02 --out artifacts/difflbm3d_envelope/physics_n32_n64.json
# 单元测试（7 passed）
python3 -m pytest tests/test_diff3d_envelope.py -q
# 图
python3 scripts/figures_difflbm3d_envelope.py
# 256³ 端到端（集群队列，递交指令原文）
#   cj2473: --stage envelope ×4(plain/step_ckpt/segmented T=4, segmented T=50) + --stage grad n=256 T=50 fp32
#   cj2474: --stage grad n=256 T=20 fp64 + --stage envelope n=256 T=20 fp64 segmented
```

**工件**：`artifacts/difflbm3d_envelope/{n32_fp64_grad,n96_fp32_env_segmented,n128_fp32_env_segmented,physics_n32_n64}.json`；
`figures/difflbm3d_envelope/{envelope_memory_time,physics_anchor}.png`；
`solver`/驱动/测试：`src/tensorlbm/diff3d_envelope.py`、`scripts/bench_difflbm3d_envelope.py`、
`tests/test_diff3d_envelope.py`。

### fail_stage / failure_reason（256³ 端到端）

- **fail_stage**：256³ 端到端实测（`cj2473` fp32 / `cj2474` fp64）**卡在队列**——递交 08:26，
  报告发布时仍 `queued`（集群 44 作业在跑，16 核请求排队）；本单未取得 256³ 的直接数字，
  只给出实测外推（`peak ≈ 0.97 + 20.2·field`、8.9 s/step@6 线程）。
- **已试路径**：① plain 全图（3.96 GiB@64³,T=8，O(T) 不可用）；② 每步
  `torch.utils.checkpoint(use_reentrant=False)`（3.17 GiB，仍 O(T)）；③ 本单分段落盘重放
  （O(1)）；④ 想用 reentrant checkpoint 再省一层，但它在 `.grad()`/`autograd.Function.backward`
  内被 torch 禁止（`RuntimeError: incompatible with .grad()`），改为非 reentrant + `seg_len=1`。
- **缺什么方法**（第 2 轮直接输入）：把单步 transient 从 ~20 场压到 ≤6 场——
  (a) `equilibrium3d(out=)` 单缓冲 + `torch.add/sub(out=)` 融合碰撞（torch autograd 不允许对
  图输出 in-place，需 (b)）；(b) 手写 BGK+streaming 的**离散伴随** `autograd.Function`
  （D2Q9 版已在 `tensorlbm.adjoint` 验证：BPTT vs FD 4.9e-9），tape 只存 `f`，反传用
  `S^T`（anti-stream）+ `(I−(I−∂feq/∂f)/τ)^T`；(c) z-slab 波前分块，把 19N³ 切成若干 slab，
  流式跨 slab 交换。三者任一可把 256³ fp32 打进 8 GB；配 fp64 时仍需在 slab 内做局部双精度。
- **本轮踩到的两个自己的坑（已修，写进 §5 教训）**：
  ① 解析解与 IC 的**轴/分量错配**——`meshgrid(indexing='ij')` 第一个轴是 z（tensor 布局
  `(19,z,y,x)`），把 `sin(x)` 写到 axis0 得到的是纵波而非横波，物理锚点因此先"错"了 30%
  以上（是 IC 错，不是求解器错）；② **Nyquist 平面破坏 Hermitian 对称**——`fftfreq` 在
  Nyquist 返回 −n/2，`k(−m)=−k(m)` 不成立，谱投影/Leray 投影后取实部会污染散度（实测散度
  12.4），改用 `k[N/2]=0` 的旋度构造后散度 ~1e-16。