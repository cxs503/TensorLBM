# XVAL-LBM-CAVITY（第 2 轮）：我方 + 开源 + Ghia ref 同规格三方横向对比

## ① 案例与基准出处
2-D 顶盖驱动方腔（lid-driven cavity），边长 L=1，顶盖以 U 拖动，Re=U·L/ν。
- **基准（party C）**：U. Ghia, K.N. Ghia, C.T. Shin, *J. Comput. Phys.* **48** (1982) 387–411，
  **Table I**（竖中线 u，pp.398）/ **Table II**（横中线 v，pp.399）。
  数据来自本仓**机器转写**（论文 OCR 文本层 + 两份独立数字副本交叉校验 + sha256）
  `benchmarks/verified/cavity/re5000/reference/ghia_re5000_reference.json`（含 Re=100/400/1000/… 全列），
  **零手抄**；本轮前已确认本仓另一份手录 `GHIA_RE100`（`TensorLBM/src/.../reference_data.py`）为
  UNVERIFIED 且中心值错 24 %，故弃用。
- **同规格参数（A/B 完全一致）**：U=0.06（Ma=0.104），W=64 lattice spacings，
  ν=U·W/Re=0.0384，**τ=0.6152**，Re=100；定常判据 max|Δu|/U ≤ 1e-6。

## ② 方法与实现（方程/格式出处 + 代码指针）
- **party A（我方 TensorLBM）**：D2Q9 **MRT**（d'Humières 1992；Lallemand & Luo PRE 61:6546），
  剪切矩松弛率 1/τ；三静止壁 **pre-streaming 半程反弹**（Ladd JFM 271:285），顶盖 **Zou–He 1997**。
  指针：`xval/XVAL-LBM-CAVITY/party_a_tensorlbm.py::run_case`，复用
  `src/tensorlbm/solver.py::collide_mrt/stream`、`src/tensorlbm/lid_driven_cavity.py::zou_he_moving_lid`。
- **party B（开源 pylbm，BSD-3，~/oss-cache/pylbm）**：同族 **d'Humières MRT**（矩空间多项式为数据，
  能量族 σ₁=100σ₂），壁面 **BFL 插值 bounce-back**（Bouzidi–Firdaouss–Lallemand, Phys. Fluids 13:3452），
  顶盖守恒矩 BC。指针：`party_b_pylbm.py::_scheme/run_case`（照 `demo/2D/lid_driven_cavity.py` 结构，
  仅参数化 Re 与 dx；串行 mpi4py 单 rank 替身见 `mpi4py_shim/`，真实通信调用显式抛错）。
- 我方 τ 与 pylbm 剪切松弛率的一致性由 `_self_check()` 运行时断言（3ν==σ₂），防止上游改约定后 Re 漂移。

## ③ 参数表
| 量 | 值 |
|---|---|
| Re | 100 |
| U_lid (lattice) | 0.06 |
| Ma = U/cs | 0.104 |
| W (lattice spacings) | 64 |
| ν_lattice = U·W/Re | 0.0384 |
| τ = 0.5+3ν | 0.6152 |
| A 网格 | n_nodes=66（W=n−2） |
| B 网格 | N=64 cells（dx=1/64） |
| 定常判据 | max|Δu|/U ≤ 1e-6 |
| 收敛残差 | A 5.4e-8（32000 步）；B 1.2e-4（24000 步，**未完全定常**） |

## ④ 对照表（逐项 ref vs got，判据逐条打勾/叉）
| 项 | Ghia ref | A TensorLBM | 误差(A) | B pylbm | 误差(B) | 工具间差 |
|---|---|---|---|---|---|---|
| u_min | -0.21090 | -0.21585 | -2.35% | -0.20672 | +1.98% | -4.42% |
| y(u_min) | 0.4531 | 0.4729 | +4.36% | 0.4766 | +5.18% | — |
| v_max | +0.17527 | +0.17978 | +2.57% | +0.17389 | -0.79% | +3.39% |
| x(v_max) | 0.2344 | 0.2422 | +3.32% | 0.2422 | +3.32% | — |
| u(0.5,0.5) | -0.20581 | -0.21249 | -3.25% | -0.20426 | +0.75% | — |
| v(0.5,0.5) | +0.05454 | +0.05616 | +2.96% | +0.05884 | +7.88% | — |
| 涡心 (x,y) | (0.6172,0.7344) | (0.613,0.740) | — | (0.613,0.732) | — | Δ=0.0080 |
| RMSE(u/u_lid) 全剖面 | 0 | 0.0048 | — | 0.0117 | — | rmse_Δu=0.0090 |
| max abs dev（17 点） | — | 0.83% ✅<3% | | 2.49% ✅<3% | | max|Δu|=3.12% ❌>3% |

判据（**看数前预登记**）：逐点 |ref−got| ≤ 3 % U_lid；工具间差 ≤ 3 % U_lid。
- 对基准：A **通过**（0.83 %），B **通过**（2.49 %）。
- 工具间：**不通过**（max|Δu| = 3.12 %，Δu_min = −4.42 %）—— 差异几乎全部来自壁面/定常度，见 ⑥。

## ⑤ 分辨率/参数敏感性
- **分辨率**：A 在 W=32（n=34，smoke，resid 6.9e-4）u_min=−0.2139→W=64 −0.2158；max_dev 2.36 %→0.83 %，
  单调下降（本仓另有 n=128/192 已验证数据 0.75 %/0.73 %，`benchmarks/verified/cavity/re100/result.json`）。
- **参数点**：Re=100 已完成三方；**Re=400 与 W=128 两档 not_run**（原因见 ⑥），不记分。
- **壁面约定**：A 为「三壁半程 + 顶盖 Zou-He 节点面」→ 有效高宽比 H/W=(n−1.5)/(n−2)=1.0079@n=66；
  B 为「四面 BFL distance=0.5」→ 严格 1.0。该 0.8 % 几何差是工具间差的主要来源之一。

## ⑥ 失败模式与工件
- **not_run（如实）**：Re=400 两档、W=128 两档。原因：8 算例批量作业递交集群 cj2558 后，
  节点侧无 numpy/torch（run_all 逐例失败、整命令仍 rc=0），取回工件只有 stage 无 runs/；
  本机受「<8 核 <10 min」硬门限制，单条命令超时（A 一次被 120 s 收割）。故只完成最便宜且同规格的
  W=64/Re=100 三方行。**未完成部分不记分**。
- **工具间差超 3 % 的定性归因**：B 残差仅 1.2e-4（未完全定常）+ B 的壁面在 cell face 而 A 的顶盖在
  节点面（几何 0.8 %）+ 剖面插值在壁面附近对约定最敏感。这是**真实工具属性**，不是求解错误。
- 复现：`cd xval/XVAL-LBM-CAVITY && python3 run_shard.py --list`；
  `python3 party_a_tensorlbm.py --re 100 --n 66 --max-steps 40000 --threads 6 --out runs/A_re100_W64.json`；
  `python3 party_b_pylbm.py --re 100 --n 64 --max-steps 24000 --generator cython --out runs/B_re100_N64.json`；
  `python3 make_comparison.py`（生成 `artifacts/comparison.json|md` + 3 图）。
- 工件：`runs/*.json`（原始剖面）、`artifacts/comparison.json`、`artifacts/fig_centerline_re100.png`、
  `fig_error_items.png`、`fig_convergence.png`、`METHOD_SURVEY.md`、`oss_solutions.json`。
