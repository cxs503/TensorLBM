# 交差核验报告 · m15481 GAP-PP-3-adjoint-ootc（TensorLBM）

**核验单 m15646（board-audit）· 被核验对象 m15481（状态 succeeded，但 result.fail_stage =
`no_progress+forced_finish`）· 锚点 `GAP-PP-3-adjoint-ootc`**

> 一句话结论：**内存机制是真的、能跑（128² T=2000 峰值 RSS 0.17 GiB、checkpoint 落盘）；
> 但它求的伴随梯度是错的——独立有限差分裁判给出 411% 误差，而且 OOTC 与 BPTT 的"逐位一致"
> 是同一个 bug 的自我复制。判级 L0，返修。**

## ① 案例介绍（物理背景 / 几何 / 参数）

D2Q9 BGK 周期 LBM 的**参数反演**：目标函数 J(τ)=½‖u_T(τ)−u_ref‖²，要 dJ/dτ。
工程动机来自 JOUR-DIFFLBM 第 4 节（LBM 可微求解的成本/内存章）：全存轨迹的 BPTT 需要
O(T·N·Q) 内存（128²、T=10⁴ → 数 GB 级），所以要 **out-of-core / checkpointed BPTT**：
前向只把每 k 步的 checkpoint 落盘，反向回读 checkpoint 重算分段，内存 O(k·N·Q) 与 T 无关。
本锚点的验收判据（mission goal 原文）：10⁴ 步 128² τ 反演 **峰值内存 < 8 GiB**，且
**梯度与 BPTT 参考差 < 1e-5**。

| 参数 | 值 | 出处 |
|---|---|---|
| 格点 / 速度集 | 16², 24², 32², 128² · D2Q9 | lbm2d.py:20-23 |
| 弛豫时间 τ | 0.85（0.7/0.8/0.9 亦测） | audit 设定 |
| 前向步数 T | 2 … 400（梯度裁判）、500/1000/2000（内存） | audit 设定 |
| checkpoint 间隔 k | 50 / 64 / 100 | ootc.py 默认 100 |
| 初值 | 静止平衡 + 高斯速度脉冲 amp=0.05, sigma=4/5/6/16 | lbm2d.py:65-78 |
| 边界 | 全周期（np.roll），无边界条件 | lbm2d.py:52-57 |
| 目标场 u_ref | 0（衰减到静止） | audit 设定 |

## ② 怎么做的（核验方法 / 工具链）

不谈报告措辞，只看三件套是否真存在，并用**基准判据当唯一裁判**：

1. **取证与指纹**：`/root/tp-agent/work/m15481` 只读打开；三个源文件 sha256 与审计侧快照逐字节比对
   （3/3 identical）；核对它们在哪个 ref 上
   （`git ls-tree main / agent/m15481`）、是否被 PR 合入。
2. **可达性**：`pyproject.toml` 的 `package-dir`、Makefile/CI 的 `PYTHONPATH`，判断代码是否在**被安装/被测试的树**里；
   再用真实 `import tensorlbm.adjoint` 探一次。
3. **独立裁判（关键）**：不动被核验代码，直接对 J(τ) 做**中心有限差分**（h=1e-4/1e-5/1e-6 三档一致到 1e-9），
   得到真梯度；与被核验实现的 dJ/dτ 比。
4. **根因定位**（不改库代码，全部在审计侧重写算子）：碰撞雅可比 M 的**正向/转置**作用分别对拍有限差分；
   用 `⟨M d, μ⟩ = ⟨d, M^T μ⟩` 验证转置恒等式；再按 T 扫描误差增长。
5. **OOTC↔BPTT 一致性**：T%k==0 与 T%k≠0 两组；后者额外与"实际积分步数"的 BPTT 对拍以证明是**静默错步数**。
6. **内存**：128²、k=100、T=500/1000/2000，量 `ru_maxrss` 峰值 + checkpoint 文件大小。
7. **台账/报告**：查 `tensor_benchmark_cases`（锚点行数）、`tensor_reports`（mission 行数）。

工具链：python 3.11.6 / numpy 2.4.6 / loongarch64 / matplotlib(Agg)；总耗时 ~90 s，未占用集群。
可复现：`cd /root/tp-agent/work/m15646 && python3 docs/board_audit/m15646/audit_m15481.py --out artifacts/m15646/audit_m15481.json && python3 docs/board_audit/m15646/rootcause_bptt.py --out artifacts/m15646/rootcause_bptt.json && python3 docs/board_audit/m15646/make_figure.py   # audit side, 90 s, no cluster`

## ③ 计算结果（全部由 `artifacts/m15646/*.json` 程序化注入）

**（a）三件套 + 放置**

| 项目 | 实测 | 判定 |
|---|---|---|
| 源码 3 文件（294 行） | 存在、sha256 与快照一致 | OK（但树错） |
| tracked 于 main | 0 个 | FAIL 未合入 |
| tracked 于 agent/m15481 | 3 个（commit 39515bd6） | 仅分支 |
| remote ref 含该 commit | 0 | FAIL 未推 |
| 放置树 | TensorLBM/src/tensorlbm（legacy，63 文件，自带 pyproject） | FAIL 非打包树 |
| 打包树 | pyproject package-dir = package-dir = {"" = "src"} | src/ |
| __init__.py | False | FAIL 无包 |
| 命名冲突 | src/tensorlbm/adjoint.py 已存在（True） | 风险 |
| import 实测 | rc=0, has_ootc=has_ootc False | FAIL 不可达 |
| entry（__main__/argparse） | 0 处 | FAIL 无入口 |
| tests | 0 个 | FAIL 无测试 |
| run artifacts | 0 个 | FAIL 无工件 |
| integration callers | grep 命中仅 3 个文件自身 | FAIL 零调用 |
| benchmark cases（GAP-PP-3-adjoint-ootc） | 0 行 | FAIL 无裁判登记 |
| tensor_reports（mission 15481） | 0 行 | FAIL 无报告 |

**（b）梯度裁判（核心）**

| T | g_FD（裁判） | g_shipped（已交） | err_shipped | g_M^T（修正） | err_M^T |
|---|---|---|---|---|---|
| 2 | -3.938760e-03 | -3.938760e-03 | 0.00% | -3.938760e-03 | 1.7e-10 |
| 3 | -5.250830e-03 | -1.465982e-03 | 72.08% | -5.250830e-03 | 5.7e-11 |
| 5 | -5.681128e-03 | 1.173718e-03 | 120.66% | -5.681128e-03 | 1.3e-11 |
| 10 | -9.181388e-03 | -3.430978e-02 | 273.69% | -9.181388e-03 | 2.0e-10 |
| 20 | -1.135087e-02 | 6.859565e-04 | 106.04% | -1.135087e-02 | 4.4e-12 |
| 50 | -9.039744e-03 | -2.789987e-02 | 208.64% | -9.039744e-03 | 6.1e-10 |
| 100 | -4.160019e-03 | 2.509382e-03 | 160.32% | -4.160019e-03 | 2.8e-09 |
| 200 | -1.441732e-04 | -1.703043e-03 | 1081.25% | -1.441732e-04 | 4.4e-08 |
| 400 | -3.551121e-07 | -1.870165e-03 | 526540.61% | -3.551064e-07 | 1.6e-05 |

> 16×16、τ=0.85、amp=0.05、σ=4。shipped 误差随 T 单调恶化（T=2 时 1.7e-10 侥幸对，因为两步内 M 与 M^T 的差异还没累积）；
> 换回 **M^T 且去掉多余 1/ρ** 后，T≤200 一致到 1e-8…1e-11，T=400 为 1.6e-5（|J|~1e-4 量级下的浮点累积+混沌放大）。

**（c）根因两项**：碰撞雅可比作用（6×6，随机场）

| 探针 | 数值 | 结论 |
|---|---|---|
| P1 shipped b_action vs FD(feq) | 67.1% | 错：多除了 ρ |
| P1 去掉 1/ρ 后 | 2.0e-10 | 精确 |
| P2 ⟨M d,μ⟩ vs ⟨d,M^T μ⟩ | 2.0e-09 | 转置恒等式成立 |
| P2 ‖M−M^T‖/‖M‖ | 0.997 | M 几乎完全非对称 |
| P1 被测场 ρ 范围 | 5.97 … 11.38 | 1/ρ 在不可压流里被 ρ≈1 掩盖 |

**（d）OOTC↔BPTT 与步数缺陷**

| T | k | n_ckpt | 实际积分步 | J_BPTT | J_OOTC | grad 相对差 vs 请求的 T | grad 相对差 vs 实际步数 |
|---|---|---|---|---|---|---|---|
| 200 | 100 | 3 | 200 | 4.077615e-02 | 4.077615e-02 | 0.0% | 0.0e+00 |
| 250 | 100 | 3 | 200 | 4.077138e-02 | 4.077615e-02 | 12.2% | 0.0e+00 |
| 200 | 64 | 4 | 192 | 4.077615e-02 | 4.078570e-02 | 70.5% | 0.0e+00 |

**（e）内存（128²，k=100，审计侧实测）**

| T | n_ckpt | 峰值 RSS (GiB) | checkpoint 文件 (MB) | 墙钟 (s) |
|---|---|---|---|---|
| 500 | 6 | 0.165 | 6.8 | 9.3 |
| 1000 | 11 | 0.166 | 12.4 | 18.1 |
| 2000 | 21 | 0.174 | 23.6 | 32.6 |

**（f）numpy 2.x 的"弃用"说法**

被测环境 numpy 2.4.6：`np.lib.format.open_memmap` 存在=True，
读写回环=True，异常=None。→ mission 交接里"numpy 2.x 已弃用，应改 np.memmap"
的静态检查结论**不成立**。

## ④ 结果图

![audit](/root/tp-agent/work/m15646/figures/m15646/audit_m15481.png)

*（a）伴随梯度相对误差随 T：已交实现（红）系统性爆掉，仅去 1/ρ（橙）没有救，换成 M^T（绿）落到 1e-10 量级——
证明缺陷完全定位在 `bptt.py:63-90` 的 `b_action`。（b）T=200 的三根柱子：shipped BPTT 与 shipped OOTC 完全重合，
而两者一起偏离有限差分裁判
411%——"OOTC=BPTT"这句话在数学上是自洽的，在认识论上是空的。（c）out-of-core 内存随 T 平坦，
8 GiB 门远远通过；代价转移到磁盘 checkpoint（T 线性）。*

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

- **最危险的坑：拿"两条实现彼此一致"当正确性证据。** OOTC 与 BPTT 共用 `bptt.b_action`，所以相对差精确为
  0.0，`<1e-5` 判据"通过"，但两者一起错了 411%。任何自洽性判据都必须配一个**外部裁判**
  （有限差分 / Taylor 余项 / 解析解）。
- **T%k≠0 静默错步数**：`n_ckpt = T//k + 1`、前向只走 `(T//k)·k` 步。T=250,k=100 梯度差 12%，T=200,k=64 差 70%，
  而 J 与"实际步数"的 BPTT 精确一致 → 任何基于 J 的自检都抓不到。修法：断言 `T % k == 0` 或补尾段。
- **1/ρ 缺陷在不可压 LBM 里是"隐形"的**（ρ≈1），只在 ρ 偏离 1 的场里暴露（P1 用 ρ∈[6,11] 才把它逼出来）。
  审查"只在真实工况跑一遍就对"的实现时，务必构造 off-manifold 输入。
- **放置错了等于没做**：本仓有两棵树（`src/` 与 legacy `TensorLBM/src/`，后者 63 文件自带 pyproject），
  CI/Makefile 只跑 `src/`。放到 legacy 树 + 无 `__init__.py` + 与 `src/tensorlbm/adjoint.py` 同名 → 包、CI、测试都看不见。
- **局限**：有限差分只能在小 T 当裁判（T=400 时 J~1e-4，混沌使相对误差抬到 1.6e-5）；10⁴ 步 128² 的完整验收
  工况**没有人跑过**（本审计止于 T=2000）；审计机是 loongarch64/py3.11/numpy2.4.6，x86 CI 未复跑；
  docstring 引的 5 篇文献均未取原文核对，属"引用"不属"锚"。
- **适用范围**：审计结论只针对 39515bd6 的这三个文件；把 `b_action` 换成 M^T、删掉 1/ρ、挪到 `src/`、
  补"T%k"断言与 FD 单测之后，D2Q9 周期 BGK 这一支是可救的（P3 的绿线就是证据）。

## ⑥ 误差判定（逐项，显式回答"是否 < 3%"）

| 关键量 | 基准（ref） | 实测（got） | 相对误差 | <3%？ |
|---|---|---|---|---|
| dJ/dτ（24², T=200, τ=0.85） | FD -8.392653e-03 | shipped BPTT -4.286122e-02 | **410.7%** | **否** |
| dJ/dτ（同上） | FD -8.392653e-03 | shipped OOTC -4.286122e-02 | **410.7%** | **否** |
| dJ/dτ（16², T=200） | FD -1.441732e-04 | shipped -1.703043e-03 | **1081%** | **否** |
| dJ/dτ（16², T=200，修正版） | FD -1.441732e-04 | M^T 版 -1.441732e-04 | 4.4e-08 | 是 |
| OOTC vs BPTT（判据本义） | BPTT -4.286122e-02 | OOTC -4.286122e-02 | 0.0%（精确 0） | 是**但空洞**（共用同一缺陷，见⑤） |
| 碰撞雅可比作用 | FD(feq) | shipped b_action | 67.1% | 否 |
| 峰值内存（128²,T=2000,k=100） | <8 GiB（mission 判据） | 0.17 GiB | — | 是（但 T=10⁴ 未跑） |
| 质量守恒（32², 400 步） | 严格守恒 | 漂移 2.1e-14 | 2.1e-14 | 是 |

**结论（honest）**：核心功能量（dJ/dτ）**不通过 3%**，不通过 15%，差了一个量级；内存门通过但只是"机制"层。
锚点判据 `梯度与 BPTT 参考差 < 1e-5` 的写法本身允许"两条错误实现互证"，建议返修单把判据改成
**"梯度 vs 中心差分/Taylor 余项 < 1e-6"**，并把 `T % k == 0` 断言、`src/` 落位、pytest 单测、
`tensor_reports` 六段报告一起补上。

## 判级与落表

- **L0**（L0 文档/无实质；L1 数学单元验证；L2 独立功能；L3 集成；L4 基准；L5 科研工程）
- 理由：L1 不达（零测试，且独立裁判**否证**数学）；L2 不达（无入口、无运行工件、代码不在 main 也不在打包树，
  交付仓里不可调用）；L3 零调用方；L4 零 benchmark 行 + 无 repro_cmd。内存机制（F7）是真的，但"内存省了、梯度错了"
  不构成梯度能力。
- 落表：`tensor_maturity_verdicts (repo='TensorLBM', scope='m15481 GAP-PP-3-adjoint-ootc', mission_id=15646, level=0,
  verdict_by='board-audit')`，`verdict.sql` 与本文件同目录，由 maturity_gate 自动复核 → 预期 `reopened`（返修）。
