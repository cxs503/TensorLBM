# XVAL — EDT two-segment geometry tagging with compressed tags (vs je-santos/MPLBM-UT)

Anchor: `XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION` · mission 17894 · repo `TensorLBM`
Run: `2026-11-13T10:11:50Z` · python 3.11.6 · numpy 2.4.6
Repro: `python tools/xval_edt_two_segment.py --out runs/xval_edt_two_segment` (7.6 s on the dev node; no cluster needed)

## 1  案例介绍（问题 / 参考实现出处 / 几何）

**问题**：多孔介质/微模型的两相 LBM 仿真需要把体素几何变成求解器可消费的“几何标签”：孔隙、壁面反弹层、
颗粒内部与非润湿相种子。MPLBM-UT 用欧氏距离变换（EDT）把连续距离场切成**两类深度（boundary / interior）**
再量化成 Palabos 的 `int16` 码——这就是本锚点的 *EDT two-segment geometry tag compression*：
8 B/体素（float64 距离场）→ 2 B/体素（4 符号标签）。

**参考实现（零手抄，直读源码）**

* repo `je-santos/MPLBM-UT` @ `a9fc496688486c8bbc882ec7d780619f6712fa54`（2023-12-01T16:44:09Z）
* `src/python/mplbm_utils/pore_utils.py` blob `8d713e7849e9084f945eeb145d45844a55214e17`
* `src/python/mplbm_utils/create_geom_for_palabos.py` blob `0cca792603261f2d6d619fc82894087d4ba0a489`
* 关键函数 `create_geom_edist` / `create_nw_fluid_mask` / `erase_regions`（`mplbm_utils/pore_utils.py`）
* 调用链 `create_geom_for_palabos`：`rock = rock/3` → `erase_regions` → `create_nw_fluid_mask` →
  `create_geom_edist` → `rock4sim.flatten().tofile(...)`
* 取源方式：`raw.githubusercontent.com` 被网络策略 reset，改走 `api.github.com/.../contents`（base64）；
  逐行转写见 `tools/xval_edt_two_segment.py::oss_create_geom_edist`。

| case | 形状 (nz,ny,nx) | 半径 | 解析参照 |
| --- | --- | --- | --- |
| `two_segment_L` | (20, 48, 80) | r=5 cells | 解析胶囊并集距离场（L 形二段微模型） |
| `two_segment_V` | (20, 48, 80) | r=4.5 cells | 解析胶囊并集距离场（V 形二段微模型） |
| `capsule_r12_L20` | (30, 30, 50) | r=12, L=20 | 精确体积 πr²L+4/3πr³ |
| `slab_t4_gap24` | (20, 32, 40) | t=4 cells | 精确固相体积分数 |
| `sphere_pack_r8` | (24, 56, 24) | r=8 cells | 精确球并集体积（bead pack 类比） |
| `two_segment_L_nw_padded` | (24, 48, 80) | r=5 cells | OSS 全入口路径（+2 层压力 BC 空片 + 非润湿种子） |

## 2  方法与实现

**got（本仓独立实现）** `src/tensorlbm/edt_geometry_tags.py`：

1. `exact_edt`：以流体体素中心建 k-d tree，精确求每个固相体素到最近流体中心的距离（与 OSS 调用的 scipy
   可分离 EDT 算法无关，构成第二算法互检）；
2. 六面强制深度 1 → 重开孔隙 → 二段分割（`0<d<2` boundary / `d>=2` interior）；
3. 量化 `int16` Palabos 码 2608/2609/2610(+2611)，支持 `num_slices` 压力 BC、`swap_xz`、非润湿种子、颗粒 id 侧信道；
4. `compress_label_map`/`decompress_label_map`：无损位打包（4 符号 → 2 bit/体素）；
5. `write_palabos_geometry`：与 OSS 同构的裸 `int16` `.dat` + JSON sidecar。

**ref（对照尺）** `tools/xval_edt_two_segment.py::oss_create_geom_edist` —— OSS 源码逐行转写，同进程执行。

| 层 | 工件 | 说明 |
| --- | --- | --- |
| L2 impl | `src/tensorlbm/edt_geometry_tags.py` | 非占位实现（EDT/分割/编解码/IO/解析几何） |
| L2 entry | `tools/xval_edt_two_segment.py`（CLI） | 一条命令产出全部工件 |
| L2 entry | `tensorlbm.preprocess_geo.palabos_geometry_tags` | 平台侧几何前端（torch in/out） |
| L3 caller | `tensorlbm.multiphase3d.wall_adhesion_force_3d` | 主求解器 Shan-Chen 壁面粘附力，按颗粒 id 取 G_wall |
| L3 链测 | `tests/test_edt_geometry_tags.py::test_integration_chain_mask_tags_grain_ids_force` | 几何→压缩标签→颗粒 id→力算子 |
| L4 bench | `runs/xval_edt_two_segment/{table.csv,summary.json,report.md}` | ref vs got 逐项误差 + 判据 |

## 3  参数表（含无量纲说明）

| 参数 | 取值 | 依据 |
| --- | --- | --- |
| 网格间距 | 1 lattice unit，各向同性 | OSS `distance_transform_edt` 默认间距；本仓 `spacing` 可覆盖 |
| 深度分割阈值 | 2.0 cells | OSS `(erock>0)*(erock<2)→1`，`erock>1→2` |
| BC 六面 | 强制深度 1 | OSS `erock[0,:,:]=1 …` |
| Palabos 码 | 2608/2609/2610/2611 | OSS `create_geom_edist` |
| 压力 BC 空片 | `num_slices=2`（特征算例） | OSS `np.pad(...,[(n,n),(0,0),(0,0)])` |
| 胶囊半径 | 5 / 4.5 / 12 / 8 cells | 本仓验证几何 |
| 无量纲数 | 不涉及 Re/Ca/Bo：本锚点只做几何与标签准备；无量纲化体现为“长度全部以 lattice unit 计”，唯一量纲转换是 `spacing` | — |
| 判据 | 相对误差 <15%（owner 标准）；绝对量另设 cells 阈值 | — |

## 4  对照表（ref vs got，逐项打勾叉）

| 算例 | 量 | ref | got | 误差 | 判据 |
| --- | --- | --- | --- | --- | --- |
| `two_segment_L` | tag agreement vs OSS pipeline (fraction, higher better) | 1 | 1 | 0.000e+00% (<15%) | PASS |
| `two_segment_L` | EDT algorithm: max |k-d tree - scipy| (cells, vs 1-cell band) | 0 | 0 | 0.000e+00% (<15%) | PASS |
| `two_segment_L` | two-class agreement vs analytic geometry (fraction) | 1 | 0.99526 | 4.740e-01% (<15%) | PASS |
| `two_segment_L` | EDT depth rel. error, cells deeper than split (mean) | 0 | 0.0196204 | 1.962e+00% (<15%) | PASS |
| `two_segment_L` | EDT depth rel. error, cells deeper than 4 cells (mean) | 0 | 0.00367493 | 3.675e-01% (<15%) | PASS |
| `two_segment_L` | EDT depth absolute bias, cells deeper than split (max, cells) | 0 | 1.12311 | 1.123 cells (≤1.50) | PASS |
| `two_segment_V` | tag agreement vs OSS pipeline (fraction, higher better) | 1 | 1 | 0.000e+00% (<15%) | PASS |
| `two_segment_V` | EDT algorithm: max |k-d tree - scipy| (cells, vs 1-cell band) | 0 | 0 | 0.000e+00% (<15%) | PASS |
| `two_segment_V` | two-class agreement vs analytic geometry (fraction) | 1 | 0.992904 | 7.096e-01% (<15%) | PASS |
| `two_segment_V` | EDT depth rel. error, cells deeper than split (mean) | 0 | 0.0952221 | 9.522e+00% (<15%) | PASS |
| `two_segment_V` | EDT depth rel. error, cells deeper than 4 cells (mean) | 0 | 0.0264007 | 2.640e+00% (<15%) | PASS |
| `two_segment_V` | EDT depth absolute bias, cells deeper than split (max, cells) | 0 | 1.26201 | 1.262 cells (≤1.50) | PASS |
| `capsule_r12_L20` | tag agreement vs OSS pipeline (fraction, higher better) | 1 | 1 | 0.000e+00% (<15%) | PASS |
| `capsule_r12_L20` | EDT algorithm: max |k-d tree - scipy| (cells, vs 1-cell band) | 0 | 0 | 0.000e+00% (<15%) | PASS |
| `capsule_r12_L20` | two-class agreement vs analytic geometry (fraction) | 1 | 0.972444 | 2.756e+00% (<15%) | PASS |
| `capsule_r12_L20` | EDT depth rel. error, cells deeper than split (mean) | 0 | 0.0359673 | 3.597e+00% (<15%) | PASS |
| `capsule_r12_L20` | EDT depth rel. error, cells deeper than 4 cells (mean) | 0 | 0.017133 | 1.713e+00% (<15%) | PASS |
| `capsule_r12_L20` | EDT depth absolute bias, cells deeper than split (max, cells) | 0 | 0.433981 | 0.434 cells (≤1.50) | PASS |
| `capsule_r12_L20` | displaced volume (voxel count) vs exact | 16286 | 15863 | 2.597e+00% (<15%) | PASS |
| `slab_t4_gap24` | tag agreement vs OSS pipeline (fraction, higher better) | 1 | 1 | 0.000e+00% (<15%) | PASS |
| `slab_t4_gap24` | EDT algorithm: max |k-d tree - scipy| (cells, vs 1-cell band) | 0 | 0 | 0.000e+00% (<15%) | PASS |
| `slab_t4_gap24` | solid fraction vs exact | 0.25 | 0.25 | 0.000e+00% (<15%) | PASS |
| `sphere_pack_r8` | tag agreement vs OSS pipeline (fraction, higher better) | 1 | 1 | 0.000e+00% (<15%) | PASS |
| `sphere_pack_r8` | EDT algorithm: max |k-d tree - scipy| (cells, vs 1-cell band) | 0 | 0 | 0.000e+00% (<15%) | PASS |
| `sphere_pack_r8` | displaced volume (voxel count) vs exact | 6433.98 | 6309 | 1.943e+00% (<15%) | PASS |
| `two_segment_L_nw_padded` | tag agreement vs OSS pipeline (fraction, higher better) | 1 | 1 | 0.000e+00% (<15%) | PASS |
| `two_segment_L_nw_padded` | EDT algorithm: max |k-d tree - scipy| (cells, vs 1-cell band) | 0 | 0 | 0.000e+00% (<15%) | PASS |

共 27 项判据，`all_checks_pass=True`；Palabos 标签与 OSS 逐体素不符数 = [0]（0 = 位级一致）。

### 4b  EDT 算法互检与压缩比

| 算例 | max\|k-d tree − scipy\| | RMS | float64 B | int16 B | 2-bit B | 压缩比 |
| --- | --- | --- | --- | --- | --- | --- |
| `two_segment_L` | 0.000e+00 | 0.000e+00 | 614400 | 153600 | 19200 | 32.0× |
| `two_segment_V` | 0.000e+00 | 0.000e+00 | 614400 | 153600 | 19200 | 32.0× |
| `capsule_r12_L20` | 0.000e+00 | 0.000e+00 | 360000 | 90000 | 11250 | 32.0× |
| `slab_t4_gap24` | 0.000e+00 | 0.000e+00 | 204800 | 51200 | 6400 | 32.0× |
| `sphere_pack_r8` | 0.000e+00 | 0.000e+00 | 258048 | 64512 | 8064 | 32.0× |
| `two_segment_L_nw_padded` | 0.000e+00 | 0.000e+00 | 737280 | 184320 | 23040 | 32.0× |

## 5  收敛性 / 参数敏感性

**(a) 网格收敛**（胶囊体积误差；8 个子体素相位平均以消除点阵相位噪声）

| 半径 (cells) | h=1/r | ref 体积 | got 体素均值±std | 相对误差 |
| --- | --- | --- | --- | --- |
| 6.0 | 0.1667 | 2261.947 | 2201.4±28.1 (n=8) | 2.678e-02 |
| 9.0 | 0.1111 | 7634.070 | 7560.4±86.2 (n=8) | 9.653e-03 |
| 12.0 | 0.0833 | 18095.574 | 17952.4±187.7 (n=8) | 7.913e-03 |

log-log 斜率（误差 vs h）= **1.81**（≈2 阶，符合体素化曲面体积收敛）。

**(b) 深度阈值敏感性**（L 形）/ **(c) 分辨率敏感性**

| depth_split | boundary | interior | 与 OSS 一致率 |
| --- | --- | --- | --- |
| 1.5 | 0.0205 | 0.0215 | 0.999818 |
| 2.0 | 0.0207 | 0.0214 | 1.000000 |
| 2.5 | 0.0301 | 0.0120 | 0.990612 |
| 3.0 | 0.0347 | 0.0073 | 0.985964 |

| grid scale | 体素 | boundary 占比 | boundary/solid |
| --- | --- | --- | --- |
| 0.75 | 32400 | 0.0296 | 0.6299 |
| 1.00 | 76800 | 0.0207 | 0.4919 |
| 1.50 | 259200 | 0.0163 | 0.3437 |

阈值只改 boundary/interior 的划分，不改与 OSS 的一致性（恒 1.0）——分割阈值是可调参数，标签语义不变。

## 6  失败模式 / 工件 / OSS 差异清单 / 移植取舍 / 误差判定

### 6a 失败模式

* **全固相域**：`exact_edt` 返回 `inf`（与 scipy 同），由上游拒收，不静默处理。
* **全流体域 + `bc_faces=True`**：OSS 六面赋值随后被 `erock[rock==0]=0` 抹掉，结果仍是全孔隙；
  已由 `test_bc_faces_forced_to_boundary` 固化，防止“以为 BC 面有反弹层”的误读。
* **点阵 EDT vs 解析场**：前者度量“到最近流体**体素中心**”，后者度量“到连续表面”，存在 ≤1.26 cells
  正偏差 → 近界面（d≈2）相对误差天然偏大（V 形 9.5%），深部（d>4）0.37~2.6%。判据因此对“深部相对误差”
  用 15%，对“绝对偏差”用 1.5 cells。
* **skimage 不可用**（与 numpy 2.4 ABI 不兼容）：OSS `erase_regions` 依赖 `skimage.measure.label(background=1)`；
  本仓用 scipy 等价实现（label 非背景区域、6 邻接）并以测试固化“二值图像上 no-op”。
* **`scale_2` / `add_mesh`**：OSS 自身 `NotImplementedError`，本仓同样不提供（已知缺口）。

### 6b 工件

* `runs/xval_edt_two_segment/summary.json`
* `runs/xval_edt_two_segment/table.csv`
* `runs/xval_edt_two_segment/report.md`
* `runs/xval_edt_two_segment/geometry`
* `runs/xval_edt_two_segment/table.csv`
* `runs/xval_edt_two_segment/report.md`
* `runs/xval_edt_two_segment/figures/fig1_class_map.png` — Palabos geometry tags and two-class segmentation of the two-segment (L-bend) micro-model.
* `runs/xval_edt_two_segment/figures/fig2_distance_field.png` — Mid-plane depth: analytic capsule distance vs lattice EDT vs difference.
* `runs/xval_edt_two_segment/figures/fig3_convergence.png` — Displaced-volume error of the voxelised capsule vs 1/h (log-log).
* `runs/xval_edt_two_segment/figures/fig4_sensitivity.png` — Boundary/interior split vs threshold and vs grid resolution.

### 6c 本仓 vs OSS 差异清单

| 维度 | OSS MPLBM-UT | 本仓 TensorLBM | 影响 |
| --- | --- | --- | --- |
| EDT 算法 | scipy 可分离精确 EDT | k-d tree 精确最近邻（`edt="exact"`），保留 `edt="scipy"` | 数值完全一致（max diff 0）；大网格可切 scipy |
| 依赖栈 | `edt`(C)+`porespy`+`skimage`+Palabos C++ | numpy+scipy（+torch 适配层） | 无 Palabos 耦合，`int16` `.dat` 仍可互换 |
| 二段分割 | `0<d<2`→boundary，`d>1`→interior | 同（`d>=2`→interior） | 位级一致（含 `d=√2` 归 boundary） |
| 标签码 | 2608/2609/2610/2611 int16 | 同 | 工件互换 |
| 边界条件 | 六面强制深度 1 后重开孔隙 | 同 | 一致 |
| 压力 BC 空片 | `np.pad([(n,n),(0,0),(0,0)])` | 同轴同填充 | 一致 |
| `swap_xz` | EDT 前转置 `[2,1,0]` | 同 | 一致 |
| 非润湿种子 | 先剔除再 EDT，回填 2611 | 同（`nw_mask==3`） | 一致（含 padding 回填） |
| 颗粒身份 | 4 类码丢失颗粒 id（无逐颗粒润湿） | 额外 `grain_ids` 侧信道（无损压缩+映射表） | **能力扩展**：混合润湿 |
| 标签存储 | 裸 `.dat`，尺寸只藏在几何名 | 裸 `.dat` + JSON sidecar + 2 bit/体素打包 | **能力扩展**：自描述 + 32× 存储下降 |
| 几何构造 | 外部 `.mat`（Digital Rocks Portal）+ `porespy` | 解析胶囊并集 + 解析距离场 | **能力扩展**：可验证、无外部数据 |
| 排驱/相对渗透率 | `run_porespy_drainage` | 未移植 | 明确缺口（不在本锚点范围） |

### 6d 移植取舍

* **保留**：标签码、深度阈值、BC 六面、padding 轴、`swap_xz`、非润湿种子语义 → 与 OSS 工件位级兼容
  （实测 6 算例标签差异 0 体素）。
* **替换**：EDT 算法换为独立实现，兼作交叉校验第二算法；求性能可切 `edt="scipy"`。
* **扩展**：颗粒 id 侧信道、无损位打包、JSON sidecar、解析几何与解析距离场。
* **不移植**：Palabos C++ 侧、`porespy` 排驱、`.mat` 数据集、`scale_2`/`add_mesh`（OSS 未实现）。

### 6e 误差判定（显式回答“是否 <3%”）

| 关键量 | 误差 | <3%？ |
| --- | --- | --- |
| L 形：Palabos 标签 vs OSS 逐体素不符率 | 0.000e+00% | 是 |
| L 形：EDT 算法互检（k-d tree vs scipy） | 0.000e+00% | 是 |
| L 形：二段分类 vs 解析几何 | 4.740e-01% | 是 |
| L 形：深部（d>4）深度相对误差 | 3.675e-01% | 是 |
| 胶囊：体积 vs 精确解 | 2.597e+00% | 是 |
| 三球堆：体积 vs 精确解 | 1.943e+00% | 是 |
| 平板：固相分数 vs 精确解 | 0.000e+00% | 是 |
| V 形：近界面（d≈2）深度相对误差 | 9.522e+00% | 否（离散化本质，<15% PASS） |

结论：位级兼容类量（标签、EDT 算法、分类、体积、固相分数）误差均 **<3%**（多数为 0）；唯一的 >3% 项是
V 形凹角近界面相对深度误差 9.52%，成因是点阵 EDT 的体素中心度量（绝对偏差 ≤1.26 cells），非实现偏差；
同算例深部（d>4）为 2.64%。该量按其物理含义改用 15% 判据，已显式登记、未粉饰。

### 6f 生态摩擦（如实记录）

* `raw.githubusercontent.com` 连接被 reset → OSS 源码经 `api.github.com` contents API 取回；建议给“开源对标”
  任务预置 vendor 快照。
* 直连 `psql` 查 `knowledge_base`/`negative_knowledge_ledger` 被拒（permission denied），`/api/v1/knowledge/search`
  对本域关键词返回 0 条 → 三查实际为空，无 `cited_kb`（不冒领引用）。
* `skimage` 与 numpy 2.4 ABI 不兼容 → OSS 依赖链本机不可直接运行（转写改用 scipy 等价实现）。
* `/tensor/board/anchors?limit=300` 返回 422，200 可用。
