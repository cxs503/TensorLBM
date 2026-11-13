# XVAL-LBM-IB3D-DIFFUSE-INTERFACE-MDF — TensorLBM 3D 漫散界面多重直接力 IBM vs haimingz/vivsim

mission 17846 · 仓 TensorLBM · 分支 `agent/m17846` · 证据分级：算子层 REAL / 物理层 honest-FAIL（未时间收敛）

## ① 问题与参考出处

**问题**：TensorLBM 的 3D 浸入边界只有"单次直接力 + 无面元权重"一条路径
（`ibm.py::ibm_direct_forcing_3d`、`ibm_vec.py::ibm_direct_forcing_3d_vec`），
缺三维多重直接力（MDF）与漫散界面面元积分，无法由标记反力直接得到物理阻力。

**对标对象（真实执行，非自写再实现）**：`haimingz/vivsim` MIT，
commit `fb84b12f02e18f3b9dfdf738d6a1a318d81539f9`（2026-06-28）。
被对标文件逐字节入库：`benchmarks/xval_ib3d_diffuse_interface_mdf/vendor/vivsim/`
（`ib/kernels.py`、`ib/mdf.py`、`ib/stencil.py`、`ib/geometry.py`、`ib3d/stencil.py`、`ib3d/geometry.py`），
sha256 见 `vendor/PROVENANCE.json`；上游方法参照 `examples/3d/flow_past_sphere.py`（`IB_ITER=3`、`IB_PAD=4`、`get_ds` 面积权重、peskin_4pt 核、`forcing_edm`）。

**为什么需要 shim**：JAX 在 loongarch64 无可用轮子（镜像 `lpypi.loongnix.cn` 对 jaxlib 返回 HTTP 504）。
`jaxshim/` 只替换数组层（`jnp` → NumPy、`lax.fori_loop` → 顺序循环、`.at[].add` → 无缓冲 scatter-add、
gather 越界 clamp），**上游源码不改一行**即被执行。

## ② 方法与实现

新增 `src/tensorlbm/ibm3d_mdf.py`（移植 + 张量化，MIT 声明在文件头）：
`build_stencil` / `interpolate` / `spread` / `surface_weights` / `multi_direct_forcing` / `drag_coefficient`。

MDF 递推（与 OSS 一致）：
`g_j = 2·ds_j·(u_t − u_m)`；`u_m ← u_m + interp(0.5·spread(g))`；迭代 `n_iter` 次；
末次只扩散累计力 `G = spread(Σg)`，反力 `F_body = −Σg`。
面元权重 `ds` = 三角面面积按 1/3 分摊到顶点（Σds = 多面体表面积）。

耦合：`force = G + drive`，用 Guo（`1−1/2τ` + 半步速度，每步注入动量严格 = F）注入，
碰撞 BGK，流场 `(nz,ny,nx)`、标记 `(x,y,z)`。
本仓历史路径作为对照基线（单次、无 ds、无因子 2）。

## ③ 参数表

| 量 | 值 |
|---|---|
| 算子层网格 | 24×28×32（nz×ny×nx），球心 (16,14,12)，r=4，icosphere subdiv=3 → 642 标记，64 点模板 |
| 物理层网格 | 48×48×64 = 147 456 cells，D=8，16 层icosphere subdiv=3 → 642 标记 |
| Re = UD/ν | 100（ν=0.008，τ=0.524） |
| U（格气） | 0.1 → Ma = 0.173 |
| 阻塞比 D/L | 1/6 = 0.167（周期性侧向） |
| 步数 / 采样 | 3000 步；每 50 步采样；统计取后 50% |
| 参考值 | Schiller–Naumann `Cd=24/Re(1+0.15Re^0.687)` = **1.0906** |
| 运行 | 单进程 8 线程 CPU（loongarch64），0.11–0.15 s/step |

## ④ 对照表（ref vs got，打勾/叉）

**A. 算子层（TensorLBM `ibm3d_mdf` vs 执行的 vivsim 源码，float64，同一标记/场）**

| 指标 | got | 判据 | 结论 |
|---|---|---|---|
| 模板权重相对差（peskin4pt / cosine4pt / peskin3pt） | 0 / 0 / 1.9e-16 | <15% | ✅（<3%） |
| 模板单元索引命中率 | 1.000 | =1 | ✅ |
| 网格力场相对差（n_iter=1,2,3,5,8 全部） | ≤5.1e-16 | <15% | ✅（<3%） |
| 标记反力相对差 | ≤1.3e-15 | <15% | ✅（<3%） |
| Σds 相对差（vs OSS `get_ds`） | 3.1e-16 | <15% | ✅ |
| Σds vs 4πR² | 200.104 / 201.062 = **−0.48%** | <15% | ✅（内接多面体理论偏小） |
| 动量闭合 ‖ΣG + ΣF_body‖/‖F_body‖ | ≤2.4e-16 | <15% | ✅ |

**B. 物理层（球 Re=100，Cd vs Schiller–Naumann 1.0906）**

| 实现 | Cd（驱动闭合） | 误差 | Cd（标记反力） | 误差 | <15%？ |
|---|---|---|---|---|---|
| MDF n_iter=1, peskin4pt | 1.4246 | **+30.6%** | 0.9140 | **−16.2%** | ❌ |
| MDF n_iter=5, peskin4pt | 1.4098 | **+29.3%** | 0.9052 | **−17.0%** | ❌ |
| MDF n_iter=8, peskin4pt | 1.4092 | **+29.2%** | 0.9048 | **−17.0%** | ❌ |
| MDF n_iter=5, cosine4pt | 1.4266 | **+30.8%** | 0.9123 | **−16.4%** | ❌ |
| 历史单次直接力（基线） | 未完成（报告时仍在跑，日志 step 2300/3000 时 Cd_drive=1.4455） | — | −2.25（口径无效） | — | ⚠️ 未定 |

两个独立估计（驱动闭合 / 标记反力）互相差 ~55%，且**把参考值夹在中间**（0.905 … 1.41 vs 1.091）——
说明该算例**尚未时间收敛**、且驱动-IBM 的动量分摊未闭合，物理层判据**不通过（honest-FAIL）**。

## ⑤ 收敛性 / 参数敏感性

1. **MDF 迭代收敛（算子层，固定流场与目标）**：滑移残差 `‖u_t−u_m‖∞/‖u_m(0)‖∞`
   逐次 = 0.635, 0.405, 0.260, 0.168, 0.110, 0.0726（iter 1–6），单调降、无振荡；`n_iter=5` 时残差 ~0.11，
   即约 89% 的初始滑移被消除（图 1）。TensorLBM 与 OSS 的残差序列逐位一致。
2. **标记密度敏感性**：162 / 642 / 2562 标记（面密度 0.81 / 3.21 / 12.8 标记/格面）下，
   `n_iter=1` 残差 0.641/0.635/0.634，`n_iter=5` 残差 0.117/0.110/0.109 —— **几乎与标记密度无关**（图 2），
   说明本配置的收敛瓶颈是离散核 `I−ds·KKᵀ` 的谱半径而非标记过稀。
3. **n_iter 对解的敏感性（物理层）**：iter1→iter8 的 Cd_drive 只从 1.4246 变到 1.4092（**1.1%**），
   Cd_marker 1.1%；核 peskin4pt vs cosine4pt 差 1.2%。→ 在本算例里 MDF 迭代数**不是**主误差源。
4. **物理层时间收敛**：Cd_drive 从 step50 的 1.102 单调升到 step3000 的 1.41，**仍在爬升**；
   若按尾段斜率外推还需 ≥10⁴ 步才能判定平台。这是判据不通过的**首要**原因（步数预算，非算法）。

## ⑥ 失败模式与工件

- **FAIL 点**：Re=100 球 Cd 未落入 ±15%（驱动闭合 +29%，标记反力 −17%）。
  **失败阶段 = 物理层时间收敛**（`fail_stage=physics_time_convergence`），算子层全部通过。
- 失败模式 1：**驱动-IBM 动量分摊**。驱动体力的空间掩膜用几何球 `r>D/2`，而漫散 IBM 在 ±2 格壳层注力，
  壳层同时承受驱动 → 稳态前两个 Cd 口径不等价；建议后续把驱动掩膜改为 `r > D/2 + 2` 或改用进出口 BC。
- 失败模式 2：**瞬态未走完**（周期侧向域 + 体力驱动，粘性扩散时间 ~L²/ν ≈ 2.9×10⁵ 步）。
- 失败模式 3（生态摩擦）：jaxlib 无 loongarch64 轮子（镜像 504），只能以 NumPy shim 执行上游；
  已用语义测试与逐位 parity 控制该风险，但**"运行 JAX 原版"仍未做到**——记录为 friction。
- 工件：`benchmarks/xval_ib3d_diffuse_interface_mdf/{run_xval.py, make_figures.py, artifacts/*.json, figures/*.png, vendor/, jaxshim/}`，
  `tests/test_ibm3d_mdf.py`（10 项，全绿），差异清单 `docs/ibm3d_mdf/DIFFERENCES_VS_VIVSIM.md`。
- 复现：`python run_xval.py --stage operator`；`python run_xval.py --stage physics --tag t --impl mdf --n-iter 5 --max-steps 3000`；`python make_figures.py`。

图：`fig1_mdf_convergence.png`（滑移残差 vs 迭代）、`fig2_marker_density.png`（标记密度）、
`fig3_cd_history.png`（Cd 历史）、`fig4_cd_bars.png`（各实现 Cd vs 容差带）。
