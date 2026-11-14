# GAP-DIFFLBM-3D-COST — 3D D3Q19 可微反演的可承受包络

**锚点**：GAP-DIFFLBM-3D-COST ｜ **分支** agent/m20141 ｜ **代码** `src/tensorlbm/diff3d.py`（新）
+ `src/tensorlbm/solver3d.py::inversion_gradient_3d`（求解器侧入口）
｜ **工件** `artifacts/gap_difflbm_3d_cost/`（4 个 run JSON、numbers.md、3 张图）
｜ **测试** `tests/test_diff3d_checkpoint_gradient.py`（8 passed）

判据三条（本锚点原文）：

| # | 判据 | 本单裁决 |
|---|---|---|
| ① | AD 梯度 vs FD 相对误差 ≤ 1e-6 | **PASS**：实测最差 2.6e-13（裕度 6 个数量级） |
| ② | 256³ 峰值显存 ≤ 8 GB | **CONDITIONAL**：状态数模型 float32 = 5.94 GiB（PASS）、float64 = 11.88 GiB（FAIL）；256³ 实测 **WITHHELD**（本机无 GPU、无 256³ 规模显存，仅有外推） |
| ③ | 单次反演 ≤ 2 h / 16 核（256³） | **FAIL**：外推 6.86 h @ T=800（T_max(2h)=233 步） |

即：梯度正确性判据**达成**；显存包络**仅 float32 成立**；时间包络**不成立**（honest-FAIL，
`fail_stage = "CASE-D-256 time envelope"`）。以下逐项给出 ref vs got 与出处。

---

## ① 问题设定与基准出处

**问题**：3D D3Q19 BGK LBM 的可微反演——用反向模式自动微分求目标函数 `J` 对物理参数
（BGK 松弛时间 τ，即 ν = (τ−1/2)/3）的梯度，用于参数反演（τ / Re 辨识、数据同化）。
代价瓶颈是**内存**：一步态 `(19, N³)` float64 在 N=256 时是 **2.55 GB**，朴素 BPTT 存 T 步
= O(T) 倍，256³ 下不可行。上一轮平台结论：3D 为 NOT_SUPPORTED，Case D（显存包络）NOT RUN，
仅 2D 数据（`TensorLBM/src/tensorlbm/adjoint/{lbm2d,bptt,ootc}.py`）。

**锚点/参考解三路独立**（严格区分：只有 FD 是"独立数值参考"，连续解是"外部锚"）：

1. **Richardson 中心差分**（独立数值参考，不经过任何 autograd 路径）
   `g = (8[J(τ+h)−J(τ−h)] − [J(τ+2h)−J(τ−2h)])/(12h) + O(h⁴)`，h = 1e-4，float64。
2. **连续解外部锚**：Arnold–Beltrami–Childress (ABC) 流
   `u = A(sin z + cos y, sin x + cos z, cos x + sin y)` 无散且 **Beltrami**（∇×u = u），
   故非线性项为纯梯度，是 Navier–Stokes 的**精确解**，自相似衰减
   `u(t) = u(0) e^{−νk²t}`，`k = 2π/N`。对 `J = ½Σ|u|²`：
   `dJ/dτ = −(2/3) k² T J`。
   出处：Childress (1970)；Dombre et al., *J. Fluid Mech.* **167** (1986) 353。
3. **内部一致性**：全存储 BPTT（`full_bptt_gradient`，O(T) 内存）——同一算术，仅用于验证
   checkpoint 路径没有算错（**不作为**正确性判据，避免上一轮 m15705 的循环论证缺陷）。

**格点格式出处**：D3Q19 权重/速度集 Qian, d'Humières & Lallemand, *Europhys. Lett.* **17**
(1992) 479；平衡态到 u² 项 Succi, *The Lattice Boltzmann Equation* (2001) ch. 5；
checkpoint/Revolve 理论 Griewank, *SIAM J. Sci. Stat. Comput.* **13** (1992) 253；
Griewank & Walther, ACM TOMS **26** (2000) 136。

**无量纲/参数表**（lattice units, float64）：

| 量 | 值 | 说明 |
|---|---|---|
| 网格 | 16³ / 24³ / 32³（梯度）；16³ / 24³ / 32³（包络）；反演 16³ | 全周期盒，N cells per 2π |
| 速度集 | D3Q19，cs² = 1/3 | 权重取 `d3q19.W_EXACT64`（binary64 有理数） |
| τ（被测参数） | 0.8（主）/ 0.6、1.1（参数敏感性） | ν = (τ−1/2)/3 = 0.1 / 0.0333 / 0.2 |
| 初始幅值 A | 0.05 | Ma ≈ 0.05·2 ≈ 0.1（近不可压，O(A²) 声学瞬态） |
| 步数 T | 48（梯度）/ 16,48（包络）/ 64（反演）/ 800（256³ 外推假设） | |
| checkpoint 间隔 segment | 4（主）/ 1、2、5、16（不变性测试） | |
| 判据阈值 | grad 1e-6；显存 8 GiB；时间 2 h @16 核 | 锚点原文 |

## ② 方法与实现（关键代码指针）

1. **前向核** `diff3d.step3d`：collide-then-stream，`stream3d` 为精确整数平移（周期性置换，
   每方向 `torch.roll`）。质量与动量在浮点意义下严格守恒（测试 `test_step_conserves_mass_and_momentum`）。
2. **内存受控反向模式** `diff3d.checkpointed_gradient(f0, τ, T, segment)`：
   - 前向：`torch.no_grad()` 下跑 T 步，**每 segment 步把状态写到磁盘**（`torch.save`），
     内存里只留 1 个态；
   - 反向：从最新段开始，读回段首 checkpoint，用**确定性重算**regenerate 段内态，然后
     逐步用"单步图的 VJP"推进 costate：`λ_n = (∂step/∂f_n)ᵀ λ_{n+1}`、
     `dJ/dτ += ⟨λ_{n+1}, ∂step/∂τ⟩`；
   - 峰值内存与 **T 无关**（只有常量个态），代价是时间重算因子 `1 + segment/2`。
   理论：Revolve / checkpointing（同上出处）。
3. **为什么不做"碰撞可逆"反向**：BGK 碰撞的逆 `(1−1/τ)f + feq/τ` 存在（τ≠1），但每步把
   非流体力学模态放大 `1/|1−1/τ|` 倍——τ=0.8 时每步 **×4**，T=100 即 4¹⁰⁰ 失控。故采用
   重算而非求逆（这是本单一个被否定的候选路径，见下"注意点"）。
4. **求解器侧入口** `solver3d.inversion_gradient_3d()` → `diff3d.recover_tau()`：
   从末态种群快照 `f_obs` 反演 τ，损失 `½‖f(T;τ) − f_obs‖²`，Adam + cosine 学习率，
   返回 best-loss 迭代（Adam 会在 cos 调度衰减前越过最小值，实测见 ⑤）。
5. **显存模型** `diff3d.memory_model_bytes`（无 `steps` 参数，即"与 T 无关"是结构性质）：
   重算窗口 1 + 单步图临时 2 + costate 1 + 末态 1 = 5 态，另加 `segment` 段内缓存。

## ③ 计算结果（全部由 `make_figures.py` 从 run JSON 机器提取 → `numbers.md`，零手抄）

**梯度正确性（T=48, τ=0.8, A=0.05）**

| N | g_AD（got） | g_FD Richardson（ref） | g_full-BPTT | g_ABC 连续锚（ref） | rel(FD) | rel(BPTT) | rel(anchor) | 判据 1e-6 |
|---|---|---|---|---|---|---|---|---|
| 16 | −1.65323611e+01 | −1.65323611e+01 | −1.65323611e+01 | −1.67598915e+01 | 2.40e-13 | 0.0 | 1.358e-02 | PASS |
| 24 | −5.76124538e+01 | −5.76124538e+01 | −5.76124538e+01 | −5.83410167e+01 | 2.58e-13 | 0.0 | 1.249e-02 | PASS |
| 32 | −1.03063030e+02 | −1.03063030e+02 | −1.03063030e+02 | −1.04315937e+02 | 1.90e-13 | 0.0 | 1.201e-02 | PASS |

**参数敏感性（第二参数点，N=24, T=48）**：τ=0.6 → rel(FD) 7.26e-13；τ=0.8 → 2.58e-13；
τ=1.1 → 4.84e-13（三点全 PASS）。

**反演（真实跑，非模拟）**：N=16, T=64, τ_true=0.8, τ0=0.62, 40 iters, 69.0 s →
τ̂ = **0.803855**，相对误差 **4.82e-3**（≤3% PASS）；loss 由 1.556e-1 降到 best 3.7e-5。

**包络（Case D）**：前向步成本 `ns/cell` 在 N=16…128 上平坦（785–1033 ns/cell，
等效带宽 ~1 GB/s，4 线程），故按 cells 外推可靠；反向模式实测开销 **7.46×** 前向步。

| dtype | 核数 | 投影前向步 | 投影反向步 | T | 反演 | T_max(2h) | 峰值（模型） | 8 GB | 2 h |
|---|---|---|---|---|---|---|---|---|---|
| float64 | 4（实测尺度） | 16.55 s | 123.49 s | 800 | 27.44 h | 58 | 11.88 GiB | OVER | OVER |
| float64 | 16（×4 假设） | 4.14 s | 30.87 s | 800 | **6.86 h** | **233** | 11.88 GiB | OVER | **OVER** |
| float32 | 16 | 4.14 s | 30.87 s | 800 | 6.86 h | 233 | **5.94 GiB** | **PASS** | OVER |

图：`artifacts/gap_difflbm_3d_cost/figures/fig1_gradient.png`（梯度 vs 三参考 + 误差门限）、
`fig2_envelope.png`（步成本 vs cells，含 256³ 外推点；256³ 内存模型 vs 8 GiB 门限）、
`fig3_inversion.png`（loss 与 τ 轨迹、best-loss 迭代）。

## ④ 对照表（判据逐条打勾/打叉）

| 判据 | ref | got | 误差 | 结论 |
|---|---|---|---|---|
| AD vs FD 梯度（N=32） | −1.03063030e+02 | −1.03063030e+02 | 1.90e-13 | ✅ ≤1e-6 |
| AD vs FD 梯度（τ=0.6 / 1.1） | 见 numbers.md | − | 7.26e-13 / 4.84e-13 | ✅ |
| AD vs 连续 ABC 锚（N=32） | −1.04315937e+02 | −1.03063030e+02 | 1.20% | ✅ <3%（离散 O(k²) 误差） |
| 256³ 峰值显存（float32） | 8 GiB 门限 | 5.94 GiB（模型） | −26% | ⚠️ PASS（外推，未实测） |
| 256³ 峰值显存（float64） | 8 GiB 门限 | 11.88 GiB（模型） | +49% | ❌ |
| 256³ 反演时间（T=800, 16 核） | 2 h | 6.86 h | +243% | ❌ |
| τ 反演精度（N=16, T=64） | τ=0.8 | 0.803855 | 0.48% | ✅ ≤3% |
| 内存与 T 无关 | 结构性质 | 实测 RSS: T=16→48 只从 36.7→40.0 态等效 | 平坦 | ✅ |

## ⑤ 收敛性 / 参数敏感性

* **两分辨率收敛**：AD vs 连续锚的误差 1.358% (16³) → 1.249% (24³) → 1.201% (32³)，随 N
  单调下降，符合 BGK 剪切模态色散 `O(k²)`、`k=2π/N` 的预期（`tests` 中
  `test_gradient_matches_continuum_anchor_and_converges` 断言单调下降）。
* **两参数点**：τ = 0.6 / 0.8 / 1.1（ν = 0.033 / 0.1 / 0.2）梯度门限全部 PASS，且 FD 参考
  在 τ→0.5 的刚性端仍保持 1e-13 级一致（说明反演路径对低黏极限无隐藏 σ 依赖）。
* **checkpoint 间隔不变性**：segment ∈ {1,2,5,16} 的梯度差 ≤ 1e-12 相对（测试断言），
  说明"省内存"没有改变梯度。

## ⑥ 失败模式与工件

**未达成项（honest-FAIL）**

* `fail_stage = "CASE-D-256 time envelope"`。
* `failure_reason`：256³ 单次反演在 T=800 时外推 6.86 h（16 核，float64，eager torch），
  超 2 h 门限 3.4×；要在 2 h 内完成只能把反演窗口压到 T ≤ 233 步。显存侧仅 float32 达标
  （5.94 GiB），float64 超限。根因：① eager torch 的 D3Q19 步等效带宽只有 ~1 GB/s
  （19 次 `torch.roll` 产生大量临时数组）；② 反向模式每步 7.46× 前向成本（segment=4 的
  重算因子 3 + 单步 VJP 4.5）。**已试/未试**：已试 segment 1…16 的取舍、float32；
  未试（第 2 轮候选）单核 fused streaming kernel（index-gather 替 19 次 roll 可望 2–3×）、
  Triton/CUDA 后端复用 `docs/triton_fused_bwd.md` 的路线、以及以 `solver3d.stream3d_gather`
  替换 roll 流。
* **显存 256³ 无实测**（本机 CPU-only、无 256³ 级设备）：模型数 + N≤32 的 RSS 实测，
  256³ 数值属**外推**，故该条记 WITHHELD/CONDITIONAL 而非 PASS。
* 另记：RSS 实测的"态等效"是模型的 3.8–6.4×（分配器/线程 arena slack），说明**模型是下界**，
  真实 256³ 峰值可能高于 5.94 GiB（float32）——这也是把显存条记为未实测的原因。

**踩过的坑（可复用知识）**

1. `d3q19.W` 是 **float32**（`sum w = 1 + 1.5e-8`）。若直接用它，每步泄漏质量 ~1.9e-8 相对，
   AD 与 FD 的差会被推到 1e-8 以上、**击穿 1e-6 判据**。必须用 `d3q19.W_EXACT64`
   （`WEIGHT_PRECISION_SCHEME = rational_binary64_cast_to_runtime_dtype_v1`）。本单第一版即因此
   出现 ~1e-5 的质量漂移，测试 `test_step_conserves_mass_and_momentum` 把它抓出来。
2. **不要**用"碰撞可逆"省 checkpoint：τ<1 时每步放大 `1/|1−1/τ|`（τ=0.8 为 ×4）。
3. **不要**用 BPTT 自洽当正确性判据（m15705 的循环论证缺陷）：本单 FD(Richardson) 与
   连续 ABC 锚都是**独立**于 autograd 的参考。
4. Adam 会在 cos 学习率衰减前越过损失最小值（实测第 12 迭代 loss 3.7e-5、τ=0.8039，
   之后摆动到 0.837）→ 反演取 **best-loss 迭代**并保留全轨迹。

**repro_cmd（一键复现）**

```bash
cd <repo>                                   # TensorLBM 工作树
PYTHONPATH=src python3 benchmarks/gap_difflbm_3d_cost/run.py --case gradient --steps 48 --out artifacts/gap_difflbm_3d_cost
PYTHONPATH=src python3 benchmarks/gap_difflbm_3d_cost/run.py --case two_tau  --grids 24 --steps 48 --out artifacts/gap_difflbm_3d_cost
PYTHONPATH=src python3 benchmarks/gap_difflbm_3d_cost/run.py --case envelope --env-grids 16,24,32 --mem-steps 16,48 --over-grid 24 --over-steps 6 --out artifacts/gap_difflbm_3d_cost
PYTHONPATH=src python3 benchmarks/gap_difflbm_3d_cost/run.py --case inversion --inv-steps 64 --iters 40 --out artifacts/gap_difflbm_3d_cost
python3 benchmarks/gap_difflbm_3d_cost/make_figures.py     # → numbers.md + 3 png
python3 -m pytest tests/test_diff3d_checkpoint_gradient.py -q   # 8 passed
```

**工件路径**

* `src/tensorlbm/diff3d.py` — 前向核 + checkpoint 反向模式 + FD/连续锚/反演（L2 实现）
* `src/tensorlbm/solver3d.py::inversion_gradient_3d` — 求解器侧真调用（L3）
* `benchmarks/gap_difflbm_3d_cost/{run.py,memprobe.py,make_figures.py}` — 入口与测量（L2 入口 / L4 repro）
* `artifacts/gap_difflbm_3d_cost/run_{gradient,two_tau,envelope,inversion}.json` — 真实运行工件（L2）
* `artifacts/gap_difflbm_3d_cost/numbers.md`、`figures/fig{1,2,3}_*.png` — 机器提取数值表与图（L4 benchmark）
* `tests/test_diff3d_checkpoint_gradient.py` — 8 个不变量测试（守恒 / FD 门限 / 锚收敛 / 内存平坦 / 反演）