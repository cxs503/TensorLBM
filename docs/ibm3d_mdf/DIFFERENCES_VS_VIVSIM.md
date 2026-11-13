# TensorLBM vs haimingz/vivsim — ib3d diffuse-interface MDF: 差异清单与移植取舍

对标对象：`haimingz/vivsim` @ `fb84b12f`（MIT），文件级 sha256 见
`benchmarks/xval_ib3d_diffuse_interface_mdf/vendor/PROVENANCE.json`。
被对标方法：`vivsim/ib3d/{stencil,geometry}.py` + `vivsim/ib/{kernels,mdf}.py`
（漫散界面 4 点核 + 多重直接力 MDF），示例 `examples/3d/flow_past_sphere.py`。

## 1. 移植了什么（保留）

| 要素 | vivsim | TensorLBM `src/tensorlbm/ibm3d_mdf.py` | 一致性证据 |
|---|---|---|---|
| 核函数 | `kernel_peskin_4pt/3pt`, `kernel_cosine_4pt` | 同名三核 | 权重逐点相对差 0（peskin4pt/cosine4pt），3pt 1.9e-16 |
| 模板 | 张量积 4×4×4=64 点，`radius=2`，`floor(x)+[-1,0,1,2]` | 同 | 模板索引命中率 1.000 |
| 面元权重 | 三角面面积 1/3 分摊到顶点 (`get_ds`) | `surface_weights` | Σds 相对差 3.1e-16 |
| MDF 迭代 | `g=2·ds·(u_t−u_m)`；`u_m += interp(0.5·spread(g))`；末次只扩散累计力 | 同（`accumulate=False` 默认） | 网格力/标记反力相对差 ≤5.1e-16 |
| 反力口径 | `marker_reaction = −Σg` | 同 | 同 |

## 2. 刻意不同（取舍）

| 项 | vivsim | TensorLBM | 取舍理由 / 影响 |
|---|---|---|---|
| 数组后端 | JAX（jit/scan/`at[].add`） | PyTorch `index_add_` | JAX 无 loongarch64 wheel；两者都是无缓冲 scatter-add，语义等价（parity 0） |
| 网格布局 | `(C, nx, ny, nz)`，flat=ix·ny·nz+iy·nz+iz | `(C, nz, ny, nx)`，flat=iz·ny·nx+iy·nx+ix | 跟随本仓 solver3d 约定；坐标仍是物理 (x,y,z) |
| 越界模板 | JAX gather 自动 clamp / scatter 丢点 | `wrap=False` 显式丢点（权重置 0），`wrap=True` 可周期 | 保留 OSS 非周期语义 + 周期选项；有单测 |
| 局部子区 | 只在 `IB_PAD=4` 立方内算 IBM | 全域散播（模板静态缓存） | 省略切片管理，小网格代价可忽略 |
| 力注入 | EDM（碰撞后、用 u*） | Guo（`1−1/2τ` + 半步速度） | 两者每步注入动量都严格等于 F（momentum-consistent），差 O(F²) |
| 碰撞 | 示例用 KBC | BGK（TRT 需 guo=False） | 与本仓 3D 求解器一致；MDF 与碰撞解耦 |
| 默认迭代数 | 示例 `IB_ITER=3`，函数默认 5 | 5 | 与函数默认对齐；敏感性见报告 |
| 额外能力 | — | `accumulate=True` 变体 | 诊断用（每轮从累计力重估 u_m），非 OSS 做法 |

## 3. 本仓历史做法 vs OSS（"新增能力"的实质）

历史实现：`src/tensorlbm/ibm.py::ibm_direct_forcing_3d`、`src/tensorlbm/ibm_vec.py::ibm_direct_forcing_3d_vec`

| 项 | 历史实现 | OSS / 新实现 | 后果 |
|---|---|---|---|
| 迭代 | 1 次（Uhlmann 单次直接力） | `n_iter` 次 MDF | 单次留残差 `I−ds·KKᵀ`，壁面滑移 |
| 力幅值 | `g = u_t − u_m`（无因子 2） | `g = 2·ds·(u_t−u_m)` | 历史实现注入动量约为所需的 1/(2ds)，量纲上不是力 |
| 面积权重 | 无 | 三角面元面积权重 ds | 无法由标记反力积分出物理阻力 |
| 核 | hat / 4pt | 同 + cosine4pt / peskin3pt | 选择面扩大 |
| 反力输出 | 无 | `marker_reaction`（力，ρ=1,Δt=1） | 可直接算 Cd |

## 4. 许可

vivsim 为 MIT（(c) 2024 Haiming Zhu）。`src/tensorlbm/ibm3d_mdf.py` 文件头保留出处声明；
`vendor/vivsim/LICENSE` 随源码入库。MIT 无 copyleft 义务，仅需保留版权与许可文本。
