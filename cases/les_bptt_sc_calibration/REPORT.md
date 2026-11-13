# XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER — 可微 LBM 图展开 / BPTT 辨识 LES 参数

**任务**：与成熟开源实现交叉对标（XVAL）。对标对象 `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow`
的 *graph-unroll-BPTT-LES-parameter* 做法。**精度判据**：owner 标准，相对误差 >15% 不可接受；
本轮所有判据阈值先于运行写入 `xval/.../predeclared.json`，机器复核见 `artifacts/results.json:verdict`。
**返修说明**：原单 m17843 仅 L1（缺 impl_files/entry/run_artifacts），本轮补齐 L2–L4 全部工件（见 §0 台账）。

## §0 成熟度台账（L2 / L3 / L4）

| 层级 | 要求 | 本单工件（全部在 worktree 内，非占位） |
|---|---|---|
| L2 impl_files | 非占位实现 | `src/tensorlbm/les_bptt.py`（可微 D2Q9 + LES 闭合 + 图展开 + BPTT 反演，~400 行，无 stub/TODO/pass 占位）；`xval/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER/oss_arm.py`（上游参考臂） |
| L2 entry | 可运行入口 | `python3 xval/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER/run_xval.py`（写 `artifacts/results.json`）；`python3 xval/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER/make_figures.py`（写 4 张 png） |
| L2 run_artifacts | 真实运行工件 | `artifacts/results.json`（A1–A6 全量数值 + 环境探针 + verdict）、`figures/fig1..fig4.png`、`artifacts/table.csv` |
| L3 integration_callers | 主求解器/平台侧真调用 | `src/tensorlbm/solver.py::collide_bgk_les`（主求解器碰撞入口，Sc=0 时与既有 `collide_bgk` 逐位相同）、`src/tensorlbm/solver.py::calibrate_les_sc`（反演入口）、`src/tensorlbm/__init__.py` 惰性导出 `collide_bgk_les` / `calibrate_les_sc`；`tests/test_les_bptt.py::test_les_zero_recovers_bgk` 真调用这两个入口 |
| L4 benchmark | ref vs got 数值对照 | §4 表（9 行，含误差与 ✓/✗）+ `artifacts/results.json`（机器可读） |
| L4 repro_cmd | 复现命令 | `cd <repo> && python3 xval/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER/run_xval.py && python3 xval/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER/make_figures.py && python3 -m pytest tests/test_les_bptt.py -q` |

## §1 问题与参考出处

LBM 正向求解器把碰撞/迁移写回就地张量，梯度无法从后期场回传到模型常数，因此**数据驱动 LES 闭合参数无法辨识**。
上游实现给出了关键做法：让 collide/stream 在图内返回张量（`graph_unroll=True`），Python 侧串 N 步（`Unroll`），
LES 松弛时间 `tau` 由场量与标量 `Sc` 在图内算出 —— 于是 N 步后的场损失可回传到 `Sc`。

参考出处（零手抄：源码已逐字 vendored 到 `xval/.../oss_src/`，sha256 见 `oss_src/PROVENANCE.md`）：

- `LatFlow/Domain.py:57` `self.Sc = 0.17`
- `LatFlow/Domain.py:90-96` `EEk = Σ_{n,m} |C_n C_m|`
- `LatFlow/Domain.py:115-121` `NonEq=f-Feq; Q=sqrt(2Σ NonEq²·EEk); tau=½(tau0+sqrt(tau0²+6·Q·Sc/rho))`
- `LatFlow/Domain.py:134-140` `CollideSC(graph_unroll=True)`（图内回写替代 assign）
- `LatFlow/Domain.py:178-196` `StreamSC`（mobius 填充 + 常量核卷积）
- `LatFlow/Domain.py:244-253` `Unroll`（Python 侧串 N 步，返回状态链）
- `les_train.py`（训练循环**上游未完成**：`lr` / `total_loss` 未定义 → 无上游可跑数字，见 §6）
- 平台侧索引：`oss_solutions` id=37 `graph-unroll-bptt-les-parameter`（本单据此定位，未重复造轮）

## §2 方法与实现

1. **可微前向算子**（`les_bptt.py`）：`collide_les` = BGK + Smagorinsky 型 `tau` + 掩码 bounce-back 合成；
   `stream` = 环绕填充 + 3×3 常量核卷积（conv-as-streaming），核由速度表推出并与 roll-gather 逐位互验；
   `unroll` = 串联 N 步并保留 autograd 图。
2. **BPTT 辨识**：目标为参考速度史 `u_ref^t`，损失 `L = mean_t mean((u^t-u_ref^t)²)`；
   - `Adam`（log Sc）给出可发表的收敛轨迹（A2b）；
   - `gradient_root`：一维参数，用 Brent 法对 **BPTT 梯度 dL/dSc** 求根，超线性收敛并带显式驻点残差（A2/A3/A6）。
3. **上游参考臂**（`oss_arm.py`）：用 3 行 TF shim **直接执行** vendored `D2Q9.py`（表不是手抄），
   再把 collide/stream/unroll 逐算子转写为 float32 NumPy，每个算子标注 `Domain.py` 行号。
4. **验收臂**：A1 前向对拍、A2 双胞胎反演、A3 跨实现反演（用 OSS 参考臂数据当"实验数据"）、
   A4 梯度 vs 中心差分、A5 线性 TG 衰减 vs 解析、A6 敏感性/收敛（N、sc_init、tau0 三组共 10 例）。

## §3 参数表（无量纲数完整）

| 量 | 值 | 说明 |
|---|---|---|
| 网格 `n×n` | 48×48 | 周期边界；A1 附加 64² 含圆盘半径 8 的算例 |
| 初场 | Taylor-Green `u_x=u0 sin(kx)cos(ky)`, `u_y=-u0 cos(kx)sin(ky)` | 全域有应变 → LES 项可辨识 |
| `u0` | 0.2 | 格子单位 |
| `k_mode` / `k_phys` | 2 / 0.261799 | `k=2π·k_mode/n` |
| `tau0` | 0.56 | `nu0=(tau0-1/2)/3` |
| `nu0` | 0.020000 | 分子粘度 |
| `Sc` (待辨识) | 0.17 | 上游默认值 (Domain.py:57) |
| `Ma = u0/c_s` | 0.346 | 线性响应区；A5 实测衰减偏差 2.11% 佐证有限 Ma 影响 <3% |
| `Re = u0·n/nu0` | 480 | 格子雷诺数 |
| 展开步数 N | 40（A1/A2/A3）；敏感性另测 10/20/40 | |
| 损失 | `mean_t mean((u-u_ref)²)` | 速度场 MSE，等权 |

## §4 对照表（ref vs got，逐项判定）

| 项目 | 基准/参考 (ref) | 本仓实测 (got) | 误差 | 判定 | 判据 |
|---|---|---|---|---|---|
| A1 前向一致性 (Taylor-Green, 40 步, 无固壁) | OSS 参考臂速度场 | 8.442e-07 | rel-L2 | ✓ | 1e-4 |
| A1 前向一致性 (含圆盘 bounce-back, 40 步) | OSS 参考臂速度场 | 8.188e-07 | rel-L2 | ✓ | 1e-4 |
| A2 双胞胎参数反演 (TensorLBM 数据) | Sc_true = 0.17 | 0.170000 | 1.288e-06 | ✓ | 15% |
| A2b Adam/SGD 轨迹 (40 iter) | Sc_true = 0.17 | 0.166147 | 2.27e-02 | ✓ | 15% |
| A3 跨实现参数反演 (OSS 参考臂数据) | Sc_true = 0.17 | 0.170000 | 2.025e-06 | ✓ | 15% |
| A4 BPTT 梯度 vs 中心差分 (eps=1e-2) | FD = -1.709194e-05 | AD = -1.709106e-05 | 5.09e-05 | ✓ | 15% |
| A4 BPTT 梯度 vs 中心差分 (eps=1e-3) | FD = -1.709107e-05 | AD = -1.709106e-05 | 5.09e-07 | ✓ | 15% |
| A5 线性 TG 衰减 (LES off) vs 解析 | 0.115585 | 0.113149 | 2.11% | ✓ | 3% |
| A6 敏感性最差项 (N / sc_init / tau0 共 10 例) | Sc_true = 0.17 | 见 §5 | 1.288e-06 | ✓ | 15% |

（A5 的 `les=True` 臂**不打闸**：解析基准是层流，不含涡粘；实测衰减更快且 `nu_eff=0.022964` vs `nu0=0.020000`（+14.8% 涡粘增量），方向与幅度均合理，作为物理解读报告。）

## §5 收敛性与参数敏感性（≥2 点）

BPTT 反演在 10 组独立设置下的收敛结果（阈值 15%，全部通过）：

| 设置 | 反演 Sc | 相对误差 | 判定 |
|---|---|---|---|
| N = 10 | 0.170000 | 8.33e-12 | ✓ |
| N = 20 | 0.170000 | 2.81e-10 | ✓ |
| N = 40 | 0.170000 | 1.29e-06 | ✓ |
| sc_init = 0.02 | 0.170000 | 1.29e-06 | ✓ |
| sc_init = 0.05 | 0.170000 | 1.29e-06 | ✓ |
| sc_init = 0.1 | 0.170000 | 1.29e-06 | ✓ |
| sc_init = 0.3 | 0.170000 | 1.29e-06 | ✓ |
| tau0 = 0.51 | 0.170000 | 4.86e-10 | ✓ |
| tau0 = 0.56 | 0.170000 | 1.29e-06 | ✓ |
| tau0 = 0.71 | 0.170000 | 1.03e-07 | ✓ |

- **展开长度 N**：N=10 仍可辨识（信息量随 N 增加而增大），误差量级 ~1e-6；
- **初值 sc_init**：0.02→0.30（含真值两侧）均收敛到同一驻点 → 目标函数单峰、无局部极小陷阱；
- **tau0**：0.51（近稳定极限）/0.56/0.71 均通过 → 辨识不依赖特定粘度；
- **BPTT 梯度正确性**：`dL/dSc` 与中心差分吻合到 5.1e-07（eps=1e-3），说明整条图展开链的伴随是精确的。

## §6 失败模式、局限与工件

**失败模式与已登记风险**

1. `sqrt(tau0²+6·Q·Sc/rho)` 在 `rho→0`（强稀疏/空节点）时发散 —— 上游同样无保护；本仓在 `Q<0` 处 clamp，未改 `rho` 保护（保持与上游一致），适用域=连续相、`rho≈1`。
2. `gradient_root` 依赖 `dL/dSc` 在 `[1e-3,5]` 内变号；不变号时退化为网格采样取最小 `|dL/dSc|`（已实现，未触发）。
3. 双胞胎反演（A2）与跨实现反演（A3）都是"同算子自洽"的反问题，**证明的是 BPTT 机器与实现一致性**，不等于"从真实 DNS 数据辨识出物理 Cs"。真实 DNS/实验数据标定属后续（需外部数据源）。
4. **未跑项（NOT_RUN，如实登记）**：上游 TensorFlow 代码本体**未执行** —— 评测机为 loongarch64，无 TF 轮子（`pip download tensorflow-cpu` 无匹配版本，见 `results.json:env`）。替代证据 = 直接执行 vendored 表 + 逐算子转写 + 行号对照 + 前向对拍 1e-7 级。真正的 TF-on-x86 对拍仍待 x86 CI 补跑。
5. 上游 `les_train.py` 本身不可运行（`lr`/`total_loss` 未定义），故 **不存在上游可跑的 BPTT 基线数字**；本单的"ref"是其算法（表/闭合/图展开）与我们自己的参考臂，而非上游训练结果。此为对标证据强度的真实上限。

**生态摩擦（如实记）**：① 本仓 D2Q9 有**两份表**（`src/tensorlbm/d2q9.py` 与 `TensorLBM/src/tensorlbm/adjoint/` 内的历史副本），新代码只认 `turbulence_closure.lattice.D2Q9` 顺序并在测试里锁死顺序；② 128 线程下 torch 小张量线程抖动使同样 rollout 慢 100 倍（需 `torch.set_num_threads(1)`），已在入口显式设置；③ 无 TF 轮子导致跨栈对标只能"表执行 + 转写"（上文 4）。

**产物**：`artifacts/results.json`（机器判据）、`artifacts/table.csv`、`figures/fig1_bptt_convergence.png`（反演收敛）、
`figures/fig2_tg_decay_analytic.png`（vs 解析）、`figures/fig3_forward_parity.png`（vs OSS 参考臂场图+剖面+误差曲线）、
`figures/fig4_sensitivity.png`（N/sc_init/tau0）。

## §7 本仓 vs OSS 差异清单与移植取舍

| # | 维度 | 上游 OSS 做法 (文件:行) | TensorLBM 本仓做法 | 取舍理由 |
|---|---|---|---|---|
| 1 | 图展开载体 | TF1 graph mode：`graph_unroll=True` 分支把 `self.F[0]` 从 Variable 换成图内张量，`Unroll()` 在 Python 侧串 N 步 (Domain.py:134-140, 244-253) | PyTorch 张量由 Python 携带，`les_bptt.unroll()` 返回状态链 (张量对象即图) | 同一"图展开/无就地写"契约；PyTorch 是平台已在用的栈（上游 TF1 graph 在 loongarch64 无轮子） |
| 2 | LES 闭合 | `Q=sqrt(2Σ(f-feq)²·EEk)`，`EEk=Σ_{n,m}|c_n c_m|`，`tau=½(tau0+sqrt(tau0²+6·Q·Sc/rho))`，`Sc=0.17` (Domain.py:57, 90-96, 115-121) | **逐式照搬**（同一 Q、同一 EEk、同一 tau 公式、同一默认 Sc） | 对标锚点即此闭合；不自创公式，保证参数可互译 |
| 3 | 待辨识参数 | `Sc` 为 TF 变量，交训练循环 (les_train.py) | `Sc` 为 torch 标量，在 log 空间优化（Adam 或对 dL/dSc 做 Brent 求根） | log 空间保证正定；一维参数用求根器比 SGD 收敛快且带显式驻点残差 |
| 4 | streaming | `pad_mobius` + `tf.nn.conv2d` 与常量 STREAM 表 (utils.py:16-22, Domain.py:178-185) | 同为零填充环绕 + 3×3 常量核卷积；核由速度表**推出**（`kernel[q,q,1-c_y,1-c_x]=1`）并与 OSS STREAM 表逐项对齐、与 roll-gather 逐位对拍 | 保留 conv-as-streaming 的融合优势，同时去掉对上游表索引约定的隐式依赖（并用测试锁死） |
| 5 | 边界 | 单体张量掩码合成 `f=f*(1-b)+f_bc*b`，bounce 用 BOUNCE 反相卷积 (Domain.py:98-101, 131-132) | 同一掩码合成式，bounce 用 `f[:,opp]` 索引（等价于 BOUNCE 1×1 卷积，测试覆盖） | 语义等价、去掉 1×1 卷积开销 |
| 6 | 精度 | 全程 float32 | 前向对拍用 float32 对齐上游；参数辨识用 float64 | 辨识是对微小差异求导（ΔSc 效应 ~1%），float32 噪声会污染梯度；前向仍报 float32 对拍结果 |
| 7 | 进出口 | 上游 `les_train.py` 用 Zou-He 进出口回调 (les_train.py:55-90)，入口显式写死 shape=256×512 | 未移植：本模块只做周期边界 | 该回调依赖具体几何与 shape 硬编码；锚点判据是"图展开+BPTT+LES 参数"，不需进出口；未移植项已如实登记 |
| 8 | 训练循环 | 上游 `les_train.py` **未完成**（`lr`/`total_loss` 未定义，无法运行） | 本仓给出可运行的反演器 + 双胞胎/跨实现验收 | 上游没有可跑的 BPTT 基线数字，故对照只能到"实现级"（场/梯度/参数），已在 §6 诚实登记 |

## §8 结论

在被预声明并机器复核的 9 项判据上全部通过：前向与 OSS 参考实现一致到 8.4e-07（判定阈值 1e-4），
BPTT 反演 `Sc` 相对误差 1.3e-06（双胞胎）/ 2.0e-06（跨实现），
梯度与有限差分一致到 5.1e-07，层流解析锚 2.11%（<3%）。
图展开 / BPTT / LES 参数辨识能力由此从"缺"变为可运行、可复核、已被主求解器调用的能力；
未完成的部分已逐条登记（上游 TF 执行、真实 DNS 数据标定、Zou-He 进出口回调）。
