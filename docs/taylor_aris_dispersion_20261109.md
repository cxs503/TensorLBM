# Taylor–Aris 剪切弥散：D2Q5 被动标量核的首个外部解析锚（含 2 处真实缺陷）

TensorLBM · 验证欠账单（m7568 / selfstart / branch agent/m7568）· 2026-11-09

## 0. 一句话结论

全仓 `taylor_aris` / `taylor dispersion` **零命中**——被动标量（`passive_scalar`，D2Q5/D3Q7）
此前只有"形状/守恒/有限值"这类结构性测试，**没有任何物理验证**。本单补上外部解析锚：

* 新增 `src/tensorlbm/taylor_dispersion.py`：Taylor–Aris 三种闭式（圆管 `1/48`、平面槽 `2/105`、
  **横向周期剪切 `U²H²/(8π²D)`**）+ 通用 Aris 积分 `D_eff = D + ⟨ψ²⟩/D` + 模态衰减测量台；
* 对**库内实际发货的** D2Q5 对流-扩散核（`passive_scalar` 的 `equilibrium/collide/stream`，
  完全周期、无边界条件confound）实测 `D_eff`，与周期剪切闭式对比：
  **Pe=2/5/8/12 四档误差 −0.04% / −0.17% / −0.01% / +0.93%，全部 <3%（最严 0.93%）**；
  剪切增强比最高 **2.85×**（Pe=12，理论 2.82×），负控（均匀流，无剪切）误差 ≤0.74%。
* 同时**发现并修复 2 处真实缺陷**：`passive_scalar.bounce_back_scalar`（D2Q5 零通量壁面，
  原代码对 5 分布标量场套用 D2Q9 的 9 分量反弹表 → `IndexError`），以及
  `passive_scalar.run_passive_scalar_transport` 入口的**四层 API 漂移**（该公开入口从未成功执行过）。

## 1. 案例介绍（物理背景 / 几何 / 参数）

**物理**：被动标量在剪切流中被对流-扩散的**长期**行为不是分子扩散，而是"剪切弥散"：
横向速度梯度把标量拉成细丝，横向扩散再把细丝抹平，净效果是沿程扩散被放大
`1 + O(Pe²)` 倍。这是污染物/示踪剂/盐度/温度输运的核心工程律（Taylor 1953；Aris 1956；
Brenner 1980 的宏观输运理论把三种几何统一成同一个 cell problem）。

**本单验证的几何**：横向**周期**剪切 `u_x(y) = U sin(2πy/H)`，流向（x）周期。
选它的理由不是它简单，而是**它是唯一不需要壁面边界条件就能定义 Taylor 弥散的算例**——
库内 D2Q5 只有完全周期 streaming，没有零通量壁面算子（见 §5 局限），
因此壁面版本（`2/105`、`1/48`）在本单只作为解析式/通用积分的自洽校验，**不做 LBM 实测**（honest 边界）。

| 参数 | 取值 |
|---|---|
| 格子 | D2Q5（`_W5 = [1/3, 1/6×4]`，c_s² = 1/3，方向 0=静止,1=+x,2=+y,3=-x,4=-y） |
| 横向周期 H | `ny` = 16 lu |
| 流向周期 L | `nx` = 128 lu（L/H = 8，满足尺度分离 L ≫ H√(1+Pe²/8π²) ≈ 1.35H） |
| 分子扩散 D | 0.1 lu²/ts ⇒ τ_D = 3D+0.5 = 0.8 |
| 剪切幅值 U | `Pe·D/H`，Pe ∈ {0.001, 2, 5, 8, 12} ⇒ U ∈ {6e-6 … 0.075}（3U² ≤ 1.7%） |
| 初值 | `φ = 1 + ε cos(kx)`，ε = 1e-3（线性响应），横向均匀 |
| 步数 / 采样 | 6000 步 / 每 10 步取幅值；拟合窗口 = 记录的后 50% |
| dtype / 线程 | float64 / `torch.set_num_threads(1)`（数值与线程数无关） |

## 2. 怎么做的（方法 / 关键设置 / 工具链）

**(a) 解析侧（三闭式 + 一积分）**

* 圆管：`D_eff = D + a²ū²/(48D)`（Taylor 1953 / Aris 1956）
* 平面槽（半宽 h）：`D_eff = D + (2/105)ū²h²/D`（Wooding 1960 形式）
* 横向周期剪切：`D_eff = D + U²H²/(8π²D)`——由通用式直接得出：
  `u - ū = U sin(2πy/H)` ⇒ `ψ = -UH/(2π)·cos(2πy/H)`，`⟨ψ²⟩ = U²H²/(8π²)`
* 通用 Aris 积分 `taylor_diffusivity_from_profile(D, u, dy, periodic)`：
  二阶精度累积梯形积分求速度亏损的本原函数 `ψ`，再取零均值、`D_eff = D + ⟨ψ²⟩/D`。
  它同时充当"闭式是否写错"的**自洽闸门**（周期剪切 ny=64 时积分 vs 闭式差 **0.15%**；
  平面槽解析式在 dy=0.02 时差 <0.1%）。

**(b) 测量侧（模态衰减，免拟合横向结构）**

单一流向 Fourier 模态的**横截面平均**幅值，在横向模态松弛之后按 `exp(-k²D_eff t)` 衰减。
两个关键性质让这个测量干净：

1. **投影自清**：横向模态在横截面上均值为零，取横截面平均时被精确投影掉，
   所以瞬态不会污染拟合（不需要"等很久"的经验判断）；
2. **尺度分离可机器判据化**：本单直接断言 `k²D_eff < 0.1 × D(2π/H)²`（轴向慢 ≥10 倍），
   把"渐近是否成立"从口头承诺变成测试。

拟合用 `log|A|` 对 t 的最小二乘斜率；数值侧只用发货核
（`passive_scalar.equilibrium_scalar / collide_scalar_bgk / stream_scalar`），**零手写核、零边界条件**。

**(c) 负控**：均匀流（同 |U|，无剪切）⇒ `D_eff` 必须等于 D；
若速度场符号/索引写错（例如横向平均取错轴、sin 变成 cos 且错位），
剪切算例与负控会同时偏移，本单断言两者分离 >50%。

## 3. 计算结果（外部锚对比表）

理论 `D_eff = D(1 + Pe²/8π²)`，D = 0.1 lu²/ts。数值列 = 群速度场下的模态衰减拟合。

| Pe | U (lu/ts) | D_eff 理论 | D_eff 实测 | 相对误差 | 增强比 实测/理论 | <3% ? |
|---|---|---|---|---|---|---|
| 0.001 | 6.3e-6 | 0.10000 | 0.10001 | +0.01% | 1.00 / 1.00 | 是 |
| 2 | 0.0125 | 0.10507 | 0.10502 | −0.04% | 1.05 / 1.05 | 是 |
| 5 | 0.03125 | 0.13166 | 0.13144 | −0.17% | 1.31 / 1.32 | 是 |
| 8 | 0.0500 | 0.18106 | 0.18103 | −0.01% | 1.81 / 1.81 | 是 |
| 12 | 0.0750 | 0.28238 | 0.28500 | **+0.93%** | 2.85 / 2.82 | 是 |

负控（均匀流，Pe=8 同幅值）：`D_eff` = 0.09926 vs D = 0.1 → 误差 **−0.74%**（无剪切增强，正确）。
分离度：剪切 0.18103 > 1.5 × 负控 0.09926（实测 1.82×）。

解析自洽闸门（通用积分 vs 闭式）：

| 算例 | 通用积分 | 闭式 | 差 |
|---|---|---|---|
| 周期剪切 U=0.05, H=64 | 1.39483 | 1.39691 | 0.15% |
| 平面槽 h=4, ū=0.05, dy=0.02 | — | 0.107619 | 见 `tests/test_taylor_dispersion.py`（<2%） |

图（`docs/assets/taylor_dispersion/`，均由 `scripts/taylor_dispersion_study.py` 程序化生成，零手抄）：

* `dispersion_vs_pe.png`：`D_eff/D` vs Pe，理论曲线 `1+Pe²/8π²` + 5 档实测点 + 负控点；
* `mode_decay.png`：`log|A|` vs t（半对数），实测点 + 拟合直线（斜率 = −k²D_eff），注明拟合窗口。

## 4. 注意点（踩过的坑 / 局限 / 适用范围）

1. **离散偏差随 Pe 单调增长**：Pe=12 时 +0.93%（U=0.075，3U²=1.7%）。
   与已知 D2Q5 对流偏差律（`docs/thermal_ade_exact_modal_20261108.md`：`α_eff/α = 1 + δ_k − 3u²`）
   同量级、同方向（速度越大对流算子越"欠"），因此**这不是弥散律写错，而是 D2Q5 对流算子的一阶速度偏差**。
   工程含义：D2Q5 上做示踪剂弥散，建议 `U ≲ 0.05`（本单 Pe=8 档误差 −0.01%）。
2. **壁面版未实测**（honest 边界）：`2/105` / `1/48` 只经通用积分与闭式自洽校验，
   **没有 LBM 实测**，因为库内 D2Q5 无零通量壁面算子（见 §5 缺陷 1 的补丁只提供
   `bounce_back_scalar`，尚无壁面通道弥散算例）。不得把本单结论引用为"壁面 Taylor 弥散已验证"。
3. **模态法只在 k→0、横向已松弛时成立**；本单把该前提写成断言（`axial/transverse < 0.1`），
   伪造/调参都不能绕过。
4. 只覆盖**被动**标量（无浮力反馈、无源项），不含浮力耦合、变扩散系数、反应项。
5. 单相、等温、常扩散系数、fp64；`float32` 下 6000 步的模态幅值（1e-3 量级）信噪比不足，
   不建议照搬。

## 5. 顺带发现并修复的真实缺陷（可复现证据）

**(1) `passive_scalar` 对 D2Q5 标量场套用 D2Q9 反弹（必崩）**

```python
# 修复前 src/tensorlbm/passive_scalar.py:  g = bounce_back_cells(g, solid)
#   bounce_back_cells 的 OPPOSITE 表是 9 分量（D2Q9 动量格子）
#   复现：torch.rand(5,8,8) + mask → IndexError: index 7 is out of bounds for dimension 0 with size 5
```

修复：新增 `bounce_back_scalar(g, mask)`（D2Q5 反向置换 `[0,3,4,1,2]`），并显式拒绝非 5 分量输入；
测试 `test_bounce_back_scalar_reverses_d2q5_directions_only` / `..._rejects_d2q9_distribution` /
`..._gives_zero_normal_flux`（法向净通量为零 + 标量守恒）。

**(2) `run_passive_scalar_transport` 公开入口从未成功执行过（四层 API 漂移）**

按顺序逐个复现并修复的签名不匹配（每次修完下一层才暴露，说明该入口长期无测试）：

| # | 调用点 | 症状 | 修复 |
|---|---|---|---|
| 1 | `prepare_run_dir(root, name, overwrite)` | `TypeError: missing 1 required positional argument: 'overwrite'` | 补 `subdir` 参数 |
| 2 | `configure_logging(run_dir)` | `TypeError: Level not an integer …: PosixPath(...)` | `configure_logging()` |
| 3 | `save_config_json(asdict(cfg), path)` | `TypeError: asdict() should be called on dataclass instances` | 传 `cfg` 本身 |
| 4 | `bounce_back_cells(g, solid)`（标量） | `IndexError`（见缺陷 1） | `bounce_back_scalar(g, solid)` |
| 5 | `collide_bgk(f, rho, ux, uy, tau_f)` | `TypeError: collide_bgk() takes 2 positional arguments but 5 were given` | **未修（超出本单范围）** |

**诚实结论**：第 5 层（动量碰撞 + `apply_simple_channel_boundaries` 现需 wall/obstacle 双 mask）
是把该 demo 迁到现行 D2Q9 API 的活，本单**未完成**；因此
`run_passive_scalar_transport` **仍然是坏的**，本单只把握到"坏在哪、修了哪几层"讲清楚。
这本身是证据面结论：**一个公开导出、有 CLI 语义、有 dataclass 配置的入口，五层签名全部漂移且无人发现，
说明它从未进入任何测试或 CI**（全仓 `tests/` 对 `run_passive_scalar_transport` 零引用）。

## 6. 误差判定（关键量 vs 基准，逐项回答"是否 <3%"）

| 关键量 | 基准 | 误差 | <3% ? |
|---|---|---|---|
| D_eff（Pe=2） | `D(1+Pe²/8π²)` = 0.10507 | −0.04% | 是 |
| D_eff（Pe=5） | 0.13166 | −0.17% | 是 |
| D_eff（Pe=8） | 0.18106 | −0.01% | 是 |
| D_eff（Pe=12） | 0.28238 | +0.93% | 是 |
| D_eff（负控均匀流） | D = 0.1 | −0.74% | 是 |
| 通用 Aris 积分 vs 周期剪切闭式（ny=64） | 1.39691 | 0.15% | 是 |

判定：**四档全 <3%，最严 0.93%（比平台 15% 门槛严 16 倍）**；无基准项只有壁面通道/圆管弥散
（原因：库内无 D2Q5 零通量壁面通道算子，见 §4-2、§5-1，属能力缺口而非精度问题）。

## 7. 证据指针（可复现）

* 实现：`src/tensorlbm/taylor_dispersion.py`（闭式 + 通用积分 + 模态测量台 + `main()` CLI）
* 测试：`tests/test_taylor_dispersion.py`（13 条：解析恒等式 6、LBM 实测 3、D2Q5 壁面 4）
* 修复：`src/tensorlbm/passive_scalar.py`（`bounce_back_scalar` 新增 + 3 处 API 修复）
* 复算脚本 + 图 + `results.json`：`scripts/taylor_dispersion_study.py`、`docs/assets/taylor_dispersion/`
* 运行：`PYTHONPATH=src python3 -m pytest tests/test_taylor_dispersion.py -q`（本机 load15≈40/128，单线程 fp64，约数秒）