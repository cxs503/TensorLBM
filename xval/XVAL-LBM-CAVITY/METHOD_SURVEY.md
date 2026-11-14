# XVAL-LBM-CAVITY · 调研与方法选型（第 2 轮）

**锚点**：XVAL-LBM-CAVITY　**仓库**：TensorLBM　**轮次**：第 2 轮（第 1 轮 m20130 honest-FAIL：无产物、无 fail_stage）
**验收判据（总账）**：【横向对比｜XVAL-CAVITY｜我方 + 开源 + ref 三方｜metrics 逐项（工具间差 + vs ref）】

本文件是「先调研、再结合」的调研产物：①开源横向源码清点 ②方法谱系与标准算例出处
③选型与迁移方案（机器可读版见 `oss_solutions.json`）。实测数字全部由
`party_a_tensorlbm.py` / `party_b_pylbm.py` 程序化产出，本文件不手抄任何结果。

---

## 1. 第 1 轮失败复盘（作为排序信号）

`TensorLBM/docs/cavity_re5000_report_m17533.md`（m17533）与
`TensorLBM/cases/cavity_re5000/REPORT.md` 留存的证据显示，前一轮的死法是**算例选点**而非代码：

| 症状 | 数值 | 结论 |
|---|---|---|
| Re=5000 时 τ = 3·U·L/Re + 0.5 ≈ 0.502 | τ−0.5 = 0.0019 | BGK 处于稳定性悬崖，任何边界耦合误差都被放大 |
| n=33 / 8000 步 BGK | max_abs_err = 10.85 % | 粗网格 + 未定常，必然 FAIL |
| n=128/160 RLBM | 10000 步发散 | 自研 RLBM 与 BB/Zou-He 耦合不稳 |
| n=256 RLBM 30000 步 | resid = 0.032（未定常）| 时间步不足；LBM 定常时间 ~ L²/ν |

**教训（迁移到本轮）**：
1. **先选能赢的点**：Ghia(1982) Table I/II 在 Re=100/400/1000 均有逐点数据，且 τ = 0.73/0.56/0.54
   远离悬崖；Re=5000 属 Table III/IV 的极限算例，不适合作「方法首次横向对比」的载体。
2. **定常判据必须先行**：用残差 `max|u(t)-u(t-Δt)|/U ≤ 1e-6` 作为停机条件，而不是固定步数。
3. **坐标约定必须声明**（见 `docs/GHIA_METRIC_CONVENTION_AUDIT.md` m16339）：同一份流场在
   `wall_offset=0` 与 `+0.5` 两种约定下 RMSE 相差 0.37–1.11 %（n=128），与 3 % 门限同量级；
   不声明约定的误差数字不可比。
4. **参考表不可手抄**：本仓 `TensorLBM/src/tensorlbm/verification/lid_driven_cavity/reference_data.py`
   的 `GHIA_RE100` 是 `UNVERIFIED_OFFLINE` 且中心值错 24 %（−0.15612 vs −0.20581，audit F-A）。
   本轮只用**机器转写 + sha256** 的那一份（见 §3）。

---

## 2. 开源横向清点（`~/oss-cache`）

`oss_inventory` 在本工作树不可达（离线）；改为直接清点缓存源码，按 feature 归并：

| 仓库 | 语言/依赖 | 方腔相关资产 | 碰撞/边界结构 | 可用性 |
|---|---|---|---|---|
| `pylbm`（polytechnique.edu，BSD-3） | Python + sympy/numpy/cython，导入时 `import mpi4py.MPI` | `demo/2D/lid_driven_cavity.py`、`lid_driven_cavity_2.py`、`notebooks/06_Lid_driven_cavity.ipynb` | **符号构造** moment 集（`polynomials`/`equilibrium`/`relaxation_parameters`），d'Humières/Lallemand/Geier 三套 MRT 可切换；壁面 BFL bounce-back（Bouzidi–Firdaouss–Lallemand，`distance=0.5`），顶盖用**守恒矩指定** BC | **可跑**（需单进程 mpi4py 替身，见 §5）；cython generator 128² 实测 3.5 ms/step |
| `Maarten-vd-Sande/lbm` | Python + numba，纯 numpy 数组 | 无方腔算例（只有 cylinder_flow / dambreak） | BGK 单松弛 + cell_type 位掩码（`lbm/update_types.py` CT.FLUID/OBSTACLE/INLET/OUTLET），`lbm/streamcollide.py` 逐点循环 | 结构可读，但作为「方腔同规格第三方」需自写驱动 + 自写边界，引入我方假设，**不作对比方** |
| `hschwane/GraSPH` | Python | SPH，非 LBM | — | 不适用 |

**学到的结构（直接迁移）**：pylbm 把「矩空间多项式 + 平衡态 + 逐矩松弛率」当作**数据**而不是
代码分支，于是同规格切换碰撞模型（BGK↔MRT↔TRT）只改三个 list。我方 `collide_mrt` 目前把
`s_e/s_eps/s_q` 作为标量参数（等价于常松弛率的 9 矩 MRT），方向一致，不必重构。

---

## 3. 方法谱系与标准算例出处（external_anchor）

| 环节 | 采用的方法 | 出处（原始文献/教科书） |
|---|---|---|
| 算例与参考数据 | 2-D 顶盖驱动方腔，Table I（竖中线 u）/ Table II（横中线 v） | U. Ghia, K.N. Ghia, C.T. Shin, *J. Comput. Phys.* **48** (1982) 387–411, pp. 398–399 |
| 参考数据获取链 | 论文 OCR 文本层机械转写 + 两份独立数字副本交叉校验 + sha256 | 本仓 `benchmarks/verified/cavity/re5000/build_reference.py` → `reference/ghia_re5000_reference.json`（含 `u_columns`/`v_columns`，Re=100/400/1000/3200/5000/7500/10000 全列） |
| 碰撞（我方） | D2Q9 MRT，9 矩，剪切矩松弛率 1/τ，ν=cs²(τ−½) | d'Humières 1992；Lallemand & Luo, *Phys. Rev. E* **61** (2000) 6546 |
| 碰撞（开源方） | 同族 d'Humières MRT（能量族 σ₁=100σ₂，剪切族 σ₂=3ν） | pylbm `demo/2D/lid_driven_cavity.py` |
| 静止壁 | 半程反弹（half-way bounce-back），二阶壁面位置 | Ziegler 1993；Ladd, *J. Fluid Mech.* **271** (1994) 285 |
| 动壁（我方顶盖） | Zou & He 1997 速度边界（uy=0 解析式） | Zou & He, *Phys. Fluids* **9** (1997) 1591 |
| 壁面（开源方） | BFL 插值 bounce-back，`distance=0.5` | Bouzidi, Firdaouss & Lallemand, *Phys. Fluids* **13** (2001) 3452 |
| 定常判据 | 全场速度增量 L∞ 归一化残差 | 与本仓 `benchmarks/verified/cavity/re100/run.py` 同口径 |
| 度量约定 | 节点→物理坐标映射显式声明（`wall_convention`），壁面行按 BC 值补齐 | 本仓 `docs/GHIA_METRIC_CONVENTION_AUDIT.md`（m16339） |

**标准算例锚点（ref 原表值，程序化从 JSON 读取，不手抄）**：Re=100 竖中线
u_min = −0.21090 @ y=0.4531、u(0.5,0.5) = −0.20581；横中线 v_max = +0.17527 @ x=0.2344。

---

## 4. 选型（为什么是 Re=100/400 + MRT/BFL + 双分辨率）

1. **Re 选点**：Re=100（τ=0.730，扩散主导，任何 solver 都应过）与 Re=400（τ=0.558，主涡
   重心下移、出现下游二次涡，对边界与分辨率敏感）。二者都在 Ghia Table I/II 有逐点数据，
   且 τ−0.5 ≥ 0.058，离稳定性悬崖 30 倍远于 Re=5000。**Re=5000 不在本轮判据内**（第 1 轮
   已证其为本仓当前实现能力之外，见 §1）。
2. **碰撞选型**：两侧都用 MRT（我方 `collide_mrt`，开源方 d'Humières 矩集）。理由：同族方法
   才能把差异归因到边界/几何实现而不是碰撞模型代差；BGK 在 Re=400 已出现 τ→0.5 的敏感度。
3. **壁面选型**：保留双方各自的成熟壁面（我方半程反弹 + Zou-He 动壁；开源方 BFL 0.5 +
   守恒矩动壁），**并把几何约定写进每个 metric 块**——这本身就是横向对比要暴露的东西。
4. **分辨率**：W = 64 与 128 lattice spacings 两档，同 Re 同 U（改变的是 ν，因此 τ 随分辨率变，
   这是 LBM 定 Re 的固有性质，参数表逐档列出）。
5. **同规格实现方式**：我方网格用 `n_nodes = W + 2`，开源方用 `N = W` 个 cell（dx=1/N），两
   侧 U=0.06、ν=U·W/Re、τ 完全相同；因此**同规格**是精确的，差异只来自壁面/几何约定。

---

## 5. 迁移方案（开源侧可跑性的工程障碍及处置）

* **障碍**：pylbm 在 `pylbm/__init__.py:13` 直接 `import mpi4py.MPI`，而本机/节点无 MPI 工具链
  （`pip install mpi4py` 因缺少 `mpi.h` 构建失败）。
* **处置**：`mpi4py_shim/mpi4py/` 提供单 rank 子集（`COMM_WORLD`、`allreduce`、`Wtime`、
  `Create_cart`/`Get_cart_rank`/`Get_topo`、`Irecv`/`Isend` + `Request.Waitall` 的**自投递拷贝**
  ——单 rank 下 peer 恒为 0，故 Irecv/Isend 对等价于把 send 窗口拷到 recv 窗口，用于填周期 halo）。
  任何需要真实通信的调用**显式抛 NotImplementedError**，不会静默给错结果。
* **局限（如实声明）**：该替身只支持 `size=1`。因此开源侧是**单进程同求解器**，不是 MPI 规模
  运行；性能数字（steps/s）不可与 MPI 版比较，只用于本对比。
* 我方侧不需要迁移：`party_a_tensorlbm.py` 只调用本仓 `tensorlbm.{solver,d2q9,lid_driven_cavity}`
  的既有函数，步进组合与 `benchmarks/verified/cavity/re100/run.py`（已验证 V3 路径）一致。

---

## 6. 复现

```bash
cd xval/XVAL-LBM-CAVITY
python3 run_shard.py --list                  # 8 个算例（2 Re × 2 分辨率 × 2 方）
python3 run_all.py                           # 全部跑完（集群递交入口）
python3 make_comparison.py                   # 逐项度量 + 图 + 报告表
```

集群递交（重计算一律走队列）：

```bash
run_compute submit repo=TensorLBM stage_dir=<stage> cores=4 timeout=7200 \
  cmd="cd xval/XVAL-LBM-CAVITY && python3 run_all.py"
```