# 交差审计 · m17637 XVAL-LBM-CARTESIAN-MPI（TensorLBM √pylbm cartesian-mpi）

- 审计单：m17666（本任务）｜被审单：m17637（status=succeeded，claimed_by tp-ds-6，worktree `/root/tp-agent/work/m17637`，head_commit=null）
- 审计分支：`agent/m17666`（base `9bcc6db6` = main）
- 基准裁判：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CARTESIAN-MPI'` → **0 rows**（表内共 55 行，TensorLBM 只有 `GAP-LBM-MCMP-POISEUILLE` / `GAP-LBM-SPHERE` / `H2-FRAZIL-RHEOLOGY` / `CASE-LBM-SQUARE-CYL-NORBERG`）
- 审计脚本：`docs/board_audit/m17666/verify_m17637.py`（C1–C6）→ `runs/m17666_audit/verify_m17637.json`；图 `runs/m17666_audit/m17666_audit_figure.png`；复跑日志 `runs/m17666_audit/repro_run.log`、`runs/m17666_audit/rerun_table_equivalence.csv`

## 1 案例介绍（被审对象是什么）

被审交付 = 一次**交叉对标（XVAL）**，不是新求解器：用 TensorLBM 既有模块 `src/tensorlbm/domain_decomposition.py`（`DomainDecomposer` / `HaloExchange` / `SimulatedMultiDevice`）跑三维 Taylor–Green 涡，比较 **slab 分解（P=1/2/4）vs 单域串行 vs 解析衰减**，并盘点 TensorLBM 与 pylbm 0.12.0 在「笛卡尔分解」做法上的差异。

| 参数 | 值（`artifacts/xval_cartesian_mpi/summary.json["case"]`） |
|---|---|
| 网格 | 24³ 全周期 |
| U0 / Ma / ν / τ | 0.05 / 0.0866 / 0.02 / 0.56（D3Q19 BGK-SRT） |
| k / Re | 2π/24 = 0.26180 / 9.549 |
| 步数 | 100 |
| 分解 | z 向 slab，P = 1 / 2 / 4（`mode="simulated"`，进程内多虚拟卡） |

交付三件套（审计实测存在，C1）：

| 文件 | 字节 | sha256[:16] |
|---|---|---|
| `tools/xval_cartesian_mpi/run_xval_cartesian_mpi.py` | 17832 | c9fe49cd520c8472 |
| `tools/xval_cartesian_mpi/pylbm_tgv_run.py` | 7358 | b81cac76d892d1af |
| `tools/xval_cartesian_mpi/shims/mpi4py/MPI.py` | 13582 | 221db62acf7ddb32 |
| `docs/XVAL-LBM-CARTESIAN-MPI.md` | 12353 | 3f5021b45b62d8b2 |
| `artifacts/xval_cartesian_mpi/summary.json` | 15632 | 20e956fcea3e3882 |
| `artifacts/xval_cartesian_mpi/table_decomposition.csv` | 2848 | 1a34ef5a9b9686db |
| `artifacts/xval_cartesian_mpi/table_equivalence.csv` | 1026 | fbf981423b533bdc |

## 2 怎么审的（独立复现路径）

1. **复跑**（C2）：按 mission result 里的 repro 命令在一次性输出目录重跑，不改被审工作树：
   `cd /root/tp-agent/work/m17637 && PYTHONPATH=tools/xval_cartesian_mpi/shims:/tmp/pylbm_src python tools/xval_cartesian_mpi/run_xval_cartesian_mpi.py --n 24 --steps 100 --out /tmp/m17666_audit/artifacts` → exit 0；`table_equivalence.csv` 去掉 `wall_time_s` 后与交付件**逐字段相同**（`identical_payload: true`）。
2. **工件算术复核**（C2）：从重跑表重算 6 行 rel-L2 / γ_K 误差，与报告数字对照。
3. **对标臂覆盖检查**（C3）：逐行统计 `table_decomposition.csv` 两侧是否有数。
4. **根因复现**（C4）：用被审件自带的 `shims/mpi4py` + `/tmp/pylbm_src` 复现 pylbm 侧报错，并按其**本意**正确索引，看 pylbm 侧尺寸是否能算出。
5. **变异探针**（C5，判据有没有鉴别力）：把 halo 交换换成 no-op，看等价性指标是否变化。
6. **接线检查**（C6）：`grep -rn domain_decomposition src/tensorlbm/*.py` 找生产调用方。
7. **报告六段**（C7）：`tensor_reports id=16602`（7871 chars）抓取核对。

## 3 计算结果（审计实测数字）

### 3.1 复现性：通过

| 项 | 交付件 | 审计重跑 |
|---|---|---|
| `float64` P=1/2/4 rel-L2 vs 串行 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| `float32` P=1/2/4 rel-L2 vs 串行 | 8.7406e-4 % 恒定 | 同 |
| γ_K 误差（f64 / f32） | 4.2099 % / 4.2124 % | 同 |
| K_end 相对解析 | 2.2819 % | 同 |

repro_cmd 可执行、位级确定：**是**。

### 3.2 判据的鉴别力：通过（这是本单最有力的一条）

| 变体 | 终场 rel-L2 vs 串行 |
|---|---|
| 正常 halo（f32，P=2） | 8.7406e-4 % |
| **halo 交换替换为 no-op**（变异探针） | **11.4148 %** |

即「rel-L2 = 0.0」是**测出来的**，不是旁路跳过交换导致的假零（差 4 个数量级）。

### 3.3 对标臂（A/B 臂）覆盖：**不通过**

`table_decomposition.csv` 共 48 行（两侧各 24）：

| 实现 | 行数 | 带 local 尺寸 | 带字节数 | accepted=True |
|---|---|---|---|---|
| tensorlbm | 24 | 8（仅 1 轴可整除的 8 组） | 8 | 8 |
| **pylbm** | **24** | **0** | **0** | **0** |

`summary.json` 里 pylbm 24 行全部是 `{"error": "index 2 is out of bounds for axis 0 with size 2"}`（或 `size 1`）。

根因（C4，可复现，**是被审工具的编码 bug，不是 pylbm 的限制**）：
`run_xval_cartesian_mpi.py:98` 把 `topo.get_region_indices(...)` 返回的**分区边界数组**做 `np.diff`（长度 = 分区数，不是 3），第 106 行又按 `for i in range(3)` 取 `s[i]` → IndexError。
按其本意正确取，同一套 pylbm 代码立刻出数：n=16、split=(2,) 时 pylbm 自身给出的 rank0 局部尺寸 = **(8, 16, 16)**，分区边界 `[[0,8,16],[0,16],[0,16]]`。
⇒ 报告 §3.2「pylbm 在 N 不能被 P 整除时自行分配余数（例如 N=16,P=3 → 6/5/5 → imbalance=20%）」与「pylbm 3 轴（2×2/2×2×2/3×2）全部接受」**均无工件支撑**，是从源码推的，不是跑出来的；两处却写着「数值见 CSV，全部由两侧代码算出」「零手抄：所有 pylbm 侧的分解尺寸/邻居表/消息数都由 pylbm 自己的函数算」。

### 3.4 与基准判据对照

- `tensor_benchmark_cases` 本锚点 **0 行** ⇒ 无注册 ref/tolerance ⇒ **benchmark 判定 WITHHELD**（唯一裁判无判据，不能自评 L4）。
- 平台精度标准：γ_K 4.21 % **< 15 %**（可接受档），但 **> 3 %**（ECOSYSTEM 报告纪律的「是否<3%」必须显式回答：**否**）。该 4.21 % 是 24³ 粗网格 + BGK 的离散误差，与分解无关（P=1/2/4 完全相同）。

## 4 结果图

`runs/m17666_audit/m17666_audit_figure.png`（三面板，由 `verify_m17637.json` 程序化注入）：
左=rel-L2 按 dtype×P（0 画成 1e-12）叠 15 % 判据线 + no-op 变异探针；中=γ_K 误差 vs 3 %/15 % 双判据线；右=arm-A 两侧「有数的行数」覆盖（TensorLBM 24/24 有效 vs pylbm 0/24）。

## 5 注意点（坑 / 局限 / 适用面）

- **F1（阻塞性，对"XVAL"本身）**：锚点叫「cartesian-mpi 交叉对标」，但 pylbm 侧 24/24 无数、0 个邻居/尺寸/字节数可比对象；真正成立的只有 TensorLBM **自洽**（分解 vs 串行）+ 解析臂。§3.2 的定性结论（pylbm 3 轴、余数分配、2·dim 消息、MPI_DOUBLE）全部是**读源码**得出，其中「全部接受/20 % imbalance」被自身 CSV 反证。修法很小（改 `np.diff` 索引 + 一行分层循环），返修成本低。
- **F2（表述）**：mission result JSON「pylbm 邻居表/区域尺寸由其 MpiTopology 实算」不实：只有 `pylbm_neighbours_32_split_2x2`（1 张表、且是 2×2 而 TensorLBM 不支持 2×2，无法同比）真正由 pylbm 代码产出；区域尺寸 0 条。
- **F3（可复现性/资产）**：交付件 **untracked**（`git status` 三个 `??`，mission `head_commit=null`），main(`9bcc6db6`) 无 `tools/xval_cartesian_mpi`。工作树一旦被回收，本锚点证据即消失；gate 只能读工作树。
- **F4（测试）**：本单 **未新增任何测试**。「P=1 必须与单域逐比特一致」这条它自己在 §6 认定为关键判据的自检，**没有固化**成回归断言（现存 `tests/test_domain_decomposition.py` 12 passed 是既有文件，非本单产出）。
- **F5（范围，已诚实披露）**：`n_ranks=2/4` 是进程内 `SimulatedMultiDevice` + 自写 mpi4py shim，**无真实 MPI/NCCL 多进程**；报告 §6「未做：真实多卡 NCCL/多进程 MPI 实测」如实写了，不算粉饰，但标题里的 "MPI" 需按"模拟"读。
- **F6（口径，微小）**：`summary.json` 同一个文件里 `case.gamma_K_exact=0.0054831`（=4νk²）与 `equivalence[].gamma_K_exact=0.0027416`（=2νk²，因为 `gam` 取 `ln(K0/K_end)/(2·steps)`）并存；两侧同除 2 所以 4.21 % 有效，但同一键名两个值易误读。
- **F7（探针弱，结论仍成立）**：`axis_param_probe` 只量 `nz_local`（恒 = nz/P），对 x/y 轴给同样的 slab 形状 ⇒ 探针本身不能证明/否证 `decomposition_axis`；但静态检查确认该字段在模块内**只赋值从不参与切分**（`domain_decomposition.py:117`），所以报告"装饰性字段"的结论正确。

## 6 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | got | 基准 | 误差 | <3 %？ |
|---|---|---|---|---|
| γ_K（f64，P=1/2/4） | 2.8570e-3 | 4νk²/2 = 2.7416e-3 | **4.21 %** | **否**（>3 %，<15 %） |
| K_end（f64 串行） | 3.5296e-4 | 解析 3.6120e-4 | 2.28 % | 是 |
| 分解等价性 rel-L2（f64，P=1/2/4） | 0.0 | 0（逐比特） | 0 % | 是 |
| 分解等价性 rel-L2（f32） | 8.74e-4 % | 0 | 8.74e-6 相对 | 是 |
| pylbm 侧分解几何/字节数 | 无 | pylbm 自身函数 | **不可比（0/24）** | **无基准**：`tensor_benchmark_cases` 0 行 + pylbm 臂全灭 |

## 7 判级与处置

| 级别 | 判据 | 本单 |
|---|---|---|
| L0 文档 | 有报告/文档 | ✅ `docs/XVAL-LBM-CARTESIAN-MPI.md`（141 行）+ report 16602 六段齐 |
| L1 数学/单元验证 | 解析解/单元级校验 | ✅ TGV 衰减 vs 4νk²（4.21 %）、P=1 逐比特 |
| L2 独立功能实现（真代码 输入→计算→输出） | 真跑出数、可复现 | ✅ 17832 B 工具 + 三件套工件，审计复跑 payload 逐字段一致，变异探针证明有鉴别力 |
| L3 系统集成（接入主求解器/平台） | 生产调用方 | ❌ `src/tensorlbm/` 内无任何模块 import `domain_decomposition`；调用方只有 `tests/`、`scripts/m4_dd_validation.py`、本单工具 |
| L4 基准验证（注册算例 + 误差分析 + 可复现） | `tensor_benchmark_cases` 有注册 ref/tolerance | ❌ 0 行；且对标臂 pylbm 0/24 有效行、γ_K 4.21 %>3 % |

**判定：L2**（复现性/鉴别力/liveness 都真，但①不接入主求解器、②无注册基准、③锚点承诺的"对 pylbm 交叉"实际 0/24 有效 → 不能给 L3/L4）。
**处置：REQUEST CHANGES（返修，非重做）** —— 最小返修集：F1 修 `run_xval_cartesian_mpi.py:98/106` 的 `np.diff` 索引并重出 A/B 臂（pylbm 侧应有 24 行数）；F3 把三件套 commit 进分支（否则证据不落地）；F4 把 P=1 逐比特等价性写成 `pytest` 回归。F2 改 mission result/报告 §3.2 的"全部由两侧代码算出/全部接受"表述。F5/F6/F7 记档即可。

**独立性限制**：审计复用被审件自带的 mpi4py shim 与 `/tmp/pylbm_src`（同一 ref 源），未自建 pylbm 环境；未在真实 MPI/NCCL 多卡上验证（环境无 mpi4py，与 m17637 §6 披露一致，本审计未推翻也未补上这一缺口）；`tensor_benchmark_cases` 的 55 行读取用 `psql -U postgres`（`root` 角色无表权限，属读取路径差异，不影响内容）。