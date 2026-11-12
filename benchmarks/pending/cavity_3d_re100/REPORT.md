# CASE-LBM-CAVITY-3D-RE100 — 三维顶盖方腔 Re=100（33³）算例池报告

**锚点**：CASE-LBM-CAVITY-3D-RE100（owner 2026-11-12 算例池令，第 1 轮任务榜）
**仓库**：TensorLBM ｜ **分支**：agent/m16078 ｜ **日期**：2026-11-12
**判据（任务原样）**：<5%，指标 = 中截面中心线 u/v 极值与位置 vs Ku1987 表（33³ 网格）
**结论（先说）**：求解链跑通、收敛、量纲与对称性检查通过（REAL 管线）；但 **Ku1987 数值表未能取得**（付费墙 + 全部 OA 镜像在本工作树网络不可达），故「vs Ku 表误差」判为 **WITHHELD**，不填任何猜测数字；以库内 2D Ghia(1982) Re=100 数据为次级锚点给出 3D 侧壁修正量对照。**本单既非 PASS 也非 FAIL 于主判据，而是判据输入缺失（honest WITHHELD）。**

---

## ① 问题与 ref 出处

三维立方腔 `[0,1]³`，5 面静止墙 + 顶盖 `y=1` 沿 +x 匀速 `U=1`，Re = U·L/ν = 100。
ref（任务指定）：H. C. Ku, R. S. Hirsh, T. D. Taylor, *A pseudospectral method for solution of the
three-dimensional incompressible Navier–Stokes equations*, **J. Comput. Phys. 70 (1987) 439–462**,
DOI **10.1016/0021-9991(87)90190-2**（Chebyshev 谱方法，33³ 网格，Re=100/400/1000）。

**取数尝试与结果（零手抄纪律的代价，如实记）**：
| 途径 | 结果 |
|---|---|
| Semantic Scholar Graph API (`DOI:10.1016/0021-9991(87)90190-2`) | `openAccessPdf.status = CLOSED`；仅拿到题录/摘要 |
| OpenAlex `W1992002280` | `is_oa: true` 但 `pdf_url: null`，`source.is_oa=false`（Elsevier 付费） |
| fatcat API（找 web.archive 存档 PDF） | 连接超时 |
| scholar.archive.org / DuckDuckGo HTML | `Cannot assign requested address`（工作树出网白名单外） |
| 引文替代源：arXiv 1503.03337 (iD3Q14 MRT)、1602.05806、2407.19100；J-STAGE jfst 6(5)780；MDPI Fluids 6(9)326 | 均只**图 11 式图形对比**或仅列参考文献，无 Ku 数字表；MDPI 403 |
⇒ **Ku 表原页未能读到，故不转录、不引用任何"记忆中的" Ku 数值。** 生态摩擦登记：`F-LBM-KU1987-PAYWALL`（文献锚点不可达 = 算例池基准供给的结构性瓶颈，建议池侧提供 ref 表快照工件，见 §⑥）。

次级锚点（**库内真实数据，可溯源**）：`src/tensorlbm/lid_driven_cavity.py::GHIA_RE100`
（Ghia, Ghia & Shin 1982, J. Comput. Phys. 48:387，2D Re=100 中心线）：u_min=-0.2109@y=0.4531，v_min=-0.24533@x=0.8047。

## ② 方法与实现（复用现有能力，不另起炉灶）

`benchmarks/pending/cavity_3d_re100/run.py` 直接复用已验证的 3D 求解栈（与 `benchmarks/verified/cavity_3d_full/` 同款，那边 Re=1000 对 Ku 锚 -0.2751 误差 0.26%）：

- 格子 **D3Q19**，碰撞 **TRT（Λ=3/16）** ← `tensorlbm.solver3d.collide_trt3d`
- 5 静止墙：**pre-streaming 半程反弹**（碰撞后、stream 前用 `f_pre` 反射，V3-3D 口径）
- 顶盖：**D3Q19 Zou/He 移动壁** ← `tensorlbm.boundary.boundaries3d.zou_he_moving_lid_3d`
- 迁移：`tensorlbm.solver3d.stream3d`（`TL_STREAM_MODE=gather`）
- 收敛判据：每 1000 步 `max|Δu|`（内部流体节点，全分量）< 1e-9；输出 float64、float32 皆可（本单 float64，`mass_drift` 报出）
- 后处理：`analyze.py`（剖面抽取、三点抛物线极值定位、Richardson p=2 网格外推、图件）

**长度口径（显式声明）**：墙在节点 0 与 n-1，半程反弹把壁面放在节点外半格 ⇒ 腔边长 `L = n-1` 格，
`ν = U_lid·(n-1)/Re`。仓库既有 `cavity_3d_full/run.py` 用 `L=n`（差 3%），本单按 n-1 口径并保留
`--length-convention n` 开关以便对照。

## ③ 参数表（无量纲数完整）

| 量 | 值 | 说明 |
|---|---|---|
| Re | **100** | = U·L/ν |
| L（腔边长） | 32 格（n=33） | L=n−1 |
| U_lid（格子） | 0.1 | Ma = 0.1√3 ≈ 0.17，压缩性误差 O(Ma²)≈3% 量级 |
| ν_lattice | 0.032 | = (τ−1/2)/3 |
| τ (τ₊) | 0.596 | TRT，τ₋ = ½+Λ/(τ₊−½)，Λ=3/16 |
| 网格 | 33³（主）+ 25³（收敛对照） | Ku 亦为 33³ |
| 时长 | 9000 步（33³）/ 7000 步（25³） | `max|Δu|` < 1e-9 停机 |
| 边界 | 5×half-way bounce-back + Zou-He moving lid | 无滑移 |
| 归一化 | u/U_lid，x/L、y/L | 与 Ku 同 |

## ④ 对照表（ref vs got）

**主判据（vs Ku1987 33³ 表）——WITHHELD**：

| 指标 | Ku1987 | got (33³) | 误差 | 判定 |
|---|---|---|---|---|
| u_min (x=0.5,z=0.5) | 未能取得（付费墙） | **−0.21491** | — | ⛔ WITHHELD |
| y(u_min) | 未能取得 | **0.4911** | — | ⛔ WITHHELD |
| v_min (y=0.5,z=0.5) | 未能取得 | **−0.23306** | — | ⛔ WITHHELD |
| x(v_min) | 未能取得 | **0.7909** | — | ⛔ WITHHELD |

**次级对照（vs 库内 2D Ghia1982 Re=100；检验 3D 侧壁修正方向与量级）**：

| 指标 | 2D Ghia Re=100 | TensorLBM 3D 25³ | TensorLBM 3D 33³ | Richardson(25,33) | 3D 修正（33³ vs 2D） | 判定 |
|---|---|---|---|---|---|---|
| u_min | −0.21090 @0.4531 | −0.21548 @0.4975 | **−0.21491 @0.4911** | −0.21414 | **−1.90%** | ✅ 同量级（<5%） |
| v_min | −0.24533 @0.8047 | −0.22915 @0.7849 | **−0.23306 @0.7909** | −0.23834 | **+5.00%** | ⚠️ 边界（恰 5%） |
| v_max | — | 0.14556 | 0.14673 | — | — | — |
| u_max（近顶盖首层） | — | 0.70019 | 0.77179 | — | — | 网格依赖的边界层值，非基准量 |

**物理自检（软判据）**：展向对称性 `uz_midplane_max_abs = 1.9e-15`（float64 机器零级，中心面 uz≡0 成立）；
`u_midplane_asym_adjacent = 3.4e-15`；`uz_global_max_abs = 0.0854`（展向速度在侧壁附近非零，符合 3D 侧壁驱动涡）；
质量漂移 33³ = 0.51（初始总质量 35937，相对 1.4e-5，TRT+反弹壁的已知量级）；全程有限值。

## ⑤ 收敛性 / 参数敏感性

1. **网格收敛（25³→33³，同为 u_lid=0.1, L=n−1）**：u_min −0.21548→−0.21491（变化 0.27%），
   v_min −0.22915→−0.23306（变化 1.7%）；Richardson(p=2) 外推 u_min=−0.2141、v_min=−0.2383。
   ⇒ u_min 已基本网格无关（<0.5%），v_min 单调向更负漂移，33³ 时仍未渐近（Ku 的 33³ 谱解在 v_min 上同样是最粗的一档）。
2. **时间收敛（残差）**：25³ 7000 步 `max|Δu|`=1.18e-10；33³ 9000 步达到 0（64 位下相邻样本位级一致），
   残差历史见 `run_33_ulid010.json:resid_hist`。
3. **Ma 敏感性（u_lid=0.05 档）**：该档在算力竞争（同机另有 18+ 并发作业）下被主动放弃，**未完成 —— 如实登记为未做项**，
   脚本已支持 `--u-lid 0.05` 一键复跑。故压缩性误差对极值的影响在本单**未被量化**（局限，§⑤ 未闭合）。
4. **长度口径敏感性**：`L=n` vs `L=n−1` 使 Re 相差 3%（33³），量级小于 v_min 的 5% 判据但不可忽略；
   本单主口径 L=n−1，并保留开关。

## ⑥ 失败模式与工件 / 复现

**失败模式**：
- (a) **文献不可达**（主判据输入缺失）：见 §①表格，`F-LBM-KU1987-PAYWALL`。
- (b) **算力竞争**：同机其它 agent 作业（18 jobs × 3 threads）使 33³ 单跑从 ~4 min 拉长到 566 s；`u_lid=0.05` 档因此未跑完。
- (c) **无 GPU**：`torch.cuda.is_available()=False`，本单 CPU float64。**求解栈无缺陷**（无 friction 记入求解器）；
  生态病仅在「文献取数」与「算力排队」两处。

**一键复现**（worktree 根目录执行，全部 <10 min/档，>10 min 应走 cluster_queue）：

```bash
# 主算例 33³ Re=100 u_lid=0.1（约 4-9 min，CPU 8 线程）
TL_STREAM_MODE=gather python benchmarks/pending/cavity_3d_re100/run.py \
  --n 33 --re 100 --u-lid 0.1 --threads 8 --resid-tol 1e-9 --resid-interval 1000 \
  --dump artifacts/cavity_3d_re100/field_33.npz \
  --out  artifacts/cavity_3d_re100/run_33_ulid010.json
# 网格对照 25³
TL_STREAM_MODE=gather python benchmarks/pending/cavity_3d_re100/run.py \
  --n 25 --re 100 --u-lid 0.1 --threads 8 --dump artifacts/cavity_3d_re100/field_25.npz \
  --out  artifacts/cavity_3d_re100/run_25_ulid010.json
# 汇总 + 图件 + 对照表
python benchmarks/pending/cavity_3d_re100/analyze.py
```

**工件**：`artifacts/cavity_3d_re100/{run_33_ulid010.json, run_25_ulid010.json, summary.json,
field_33.npz, field_25.npz, log_run_*.txt, figures/{centerlines,grid_convergence,field_midplane}.png}`

**接棒建议（第二轮）**：① 由池侧预置 Ku1987 表快照工件（CSV+原页页号），本算例即可闭合主判据；
② 在集群节点上跑 33³/41³ 双档 + u_lid=0.05 档，量化 Ma 与网格误差；③ 若能取到 Ku 表，直接接入
`analyze.py` 的 `reference` 字段（脚本已留 JSON 接口，`status: WITHHELD → VERIFIED` 一行切换）。