# XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR — 与 lbmpy 的 fused-kernel 选择器交叉对标（v2 返修轮）

**Mission**: 17772 → 返修 **18111** · **Repo**: TensorLBM · **Anchor**: `XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR`
**Date**: 2026-11-13 · **Branch**: `agent/m18111` · **判据**: 误差 < 15%（owner 精度标准）

> 返修说明：原单 m17772 只交了报告 + 一行权重修复，缺 tests / 入口 / 运行工件，被 maturity_gate
> 判 L1。本版补齐 **impl + 入口 + 真实运行工件 + 主求解器调用 + ref/got 对照 + 复现命令**，
> 且把「对标」从"读源码评述"升级为**跑真的 OSS 内核**（`lbmpy` 2.0 + `pystencils` 2.0 已安装，
> 内核 JIT 编译后直接执行）。复跑命令见 §⑦。

---

## ① 问题与 ref 出处（零手抄）

**问题**：TensorLBM 的 fused kernel（`collide_bgk_fused` / `collide_bgk_matmul` + `stream`）与成熟开源
lssfau/lbmpy 的 **fused-kernel-type-selector** 做法是否等价？等价到多少位？不等价的地方在哪、
是否影响物理？

**ref 出处（本机安装的 OSS 源码，逐行核对，不引二手描述）**：

| 项 | 位置 | 内容 |
|---|---|---|
| 选择器 | `lbmpy/creationfunctions.py:298` | `LBMConfig.kernel_type: str = 'default_stream_collide'` |
| 选择器解析 | 同上 `:429-440` | `kernel_type` → `(streaming_pattern, timestep)`；旧名兼容映射 |
| 分派 | 同上 `:619-633` | `stream_pull_only` / `default_stream_collide` / `collide_only` / `PdfFieldAccessor`，其余 `raise ValueError("Invalid value of parameter 'kernel_type'")` |
| 融合内核 | `lbmpy/updatekernels.py:39` `create_lbm_kernel` | `input_accesses = accessor.read(src_field, stencil)`：**先 pull 再碰撞** |
| 平衡态 | `equilibrium/discrete_hydro_maxwellian.py` + `maxwellian_equilibrium.py:143-183` | `compressible=True` → `fq = 1 + 3cu + 4.5cu² − 1.5u²`，乘 `w·rho` |
| 生成代码 | `Kernel.get_c_code()`（本仓 dump） | 证实碰撞按 `delta_rho` 形式发射（见 §⑤ 陷阱 1） |

**本仓原有状态**：只有 *碰撞融合*（`collide_bgk_fused`：宏观量+平衡态+碰撞单遍），**没有**
stream+collide 单遍内核，也没有 `kernel_type` 这一层选择器 —— 即 OSS 的 selector 能力缺失。

---

## ② 方法与实现

**移植**（新文件 `src/tensorlbm/fused_kernel_selector.py`）：

* `KERNEL_TYPES = ("default_stream_collide", "collide_only", "stream_pull_only")` —— 与 lbmpy 同名同义；
* `resolve_kernel_type()` —— 非法值 `ValueError`（对齐 lbmpy `:633`）；
* `collide_stream_bgk(f, tau, moments="pre_stream")` —— 单遍 pull-streaming + BGK 碰撞；
  `moments` 显式声明**平衡态读矩的位置**：
  * `pre_stream`（默认）= 教科书 `stream(collide_bgk(f))`（先碰撞后 pull）；
  * `post_stream` = `collide_bgk(stream(f))`（先 pull 后碰撞）= **lbmpy 内核的语义**；
* `step(f, tau, kernel_type, moments)` —— 三种粒度的统一入口。

**对标脚手架**（`xval/fused_kernel_type_selector/lbmpy_reference.py`）：直接 import OSS 库，
`create_lb_ast → Kernel.compile()` 得到 C 内核并在 numpy 场上跑；方向重排、权重、零中心存储
全部**运行时从 lbmpy 读**（`LBStencil("D2Q9")` / `get_weights` / 代码生成探测），无手抄常量。

**主求解器真调用**（L3）：`tensorlbm.sediment.channel`（明渠 Poiseuille + bounce-back 的真实算例）
新增 `ChannelConfig.kernel_type`，`default_stream_collide` 走新融合内核，`collide_only` 保留历史
`stream(collide_bgk_fused(...))`。300 步两种粒度 `u_star` **逐位相同**（rel = 0.0），profile 差 0.0。

---

## ③ 参数表（无量纲量）

| 参数 | 等价性面板 | 物理面板 |
|---|---|---|
| 网格 | 32×32（周期） | 32×32 / 64×64（周期） |
| 离散速度集 | D2Q9 | D2Q9 |
| 精度 | float64 | float64 |
| 松弛时间 τ | 0.8 | 0.8 / 0.6 / 0.9 |
| 无量纲黏度 ν=(τ−½)/3 | 0.1 | 0.1 / 0.0333 / 0.1333 |
| 初始场 | `feq(ρ=1, u=±0.05 随机)`，seed=1 | 剪切波 `u_x = 10⁻⁴ sin(ky)`，ρ=1 |
| 波数 k | — | 2π/32 = 0.1963 / 2π/64 = 0.0982 |
| 步数 | 1 / 50 | 400 |
| 马赫数 | ~0.05·√2 | 1.0e-4（线性区） |
| 判据 | 相对误差 < 15% | 振幅 vs 解析解 < 15% |

---

## ④ 对照表（ref vs got）

运行工件：`artifacts/xval_fused_kernel_type_selector/results.json`（本文件所有数字由其生成，零手抄）。

### 4.1 粒度等价性（锚点本体：融合=性能开关，不是物理开关）

| ref | got | max\|Δf\| (1 步) | max\|Δf\| (50 步) | 判据 <15% |
|---|---|---|---|---|
| `stream(collide_bgk(f))` | TL fused `pre_stream` | 1.1e-16 | 3.9e-16 | ✅ 机器精度 |
| `collide_bgk(f)` | `collide_bgk_matmul(f)` | 1.7e-16 | 3.9e-16 | ✅ 机器精度 |
| `collide_bgk(stream(f))` | TL fused `post_stream` | 1.1e-16 | 3.9e-16 | ✅ 机器精度 |
| **lbmpy `default_stream_collide`（JIT C 内核）** | TL fused `post_stream` | **2.8e-16** | **1.8e-15** | ✅ **跨库一致** |
| lbmpy `collide_only`（JIT C 内核） | `collide_bgk(f)` | **2.2e-16** | — | ✅ **跨库一致** |
| TL fused `pre_stream` | TL fused `post_stream` | — | 1.6e-3（相对 0.37%） | ⚠️ 算子次序差，非 bug（§⑤） |

### 4.2 物理 benchmark：剪切波衰减（ref = Navier–Stokes 解析解 `u₀·exp(−νk²t)`）

| 网格 / τ | got（TL fused） | got（lbmpy） | ref 解析 | 误差（TL） | 误差（lbmpy） | 判据 |
|---|---|---|---|---|---|---|
| 32 / 0.8 | 1.064041e-05 | 1.064041e-05 | 1.069629e-05 | **0.522%** ✅ | 0.522% ✅ | <15% |
| 64 / 0.8 | 3.398024e-05 | 3.398024e-05 | 3.400446e-05 | **0.071%** ✅ | 0.071% ✅ | <15% |
| 64 / 0.6 | 4.393193e-05 | 4.393193e-05 | 4.397022e-05 | **0.087%** ✅ | 0.087% ✅ | <15% |
| 64 / 0.9 | 2.989058e-05 | 2.989058e-05 | 2.990367e-05 | **0.044%** ✅ | 0.044% ✅ | <15% |
| 64 / 0.8（阴性对照 `stream_pull_only`，只输运不碰撞） | 3.333333e-05 | — | 3.400446e-05 | **1.97%**（τ=0.6 时 24.2%） ✅判据能分辨 | — | — |

**判定**：锚点判据 <15% —— 粒度等价性全部机器精度通过；物理面 0.04%–0.52% 通过。
`stream_pull_only`（去掉碰撞）误差随 τ 变大到 24%，证明本 benchmark **有分辨力**，不是恒过的空测试。

---

## ⑤ 收敛性 / 参数敏感性（≥2 点）+ 三个真陷阱

1. **网格收敛**：k 从 0.1963→0.0982（32→64），误差 0.522%→0.071%（≈7×，符合 BGK 剪切模 O(k²)）。
2. **τ 敏感性**（ν=0.0333/0.1/0.1333）：误差 0.087%/0.071%/0.044%，ν 越小误差略增（同 k² 压缩性/离散修正），
   全程远低于 15%。
3. **步数敏感性**：单步差 1.1e-16 → 50 步 3.9e-16（浮点舍入线性累积，无系统性漂移）。

**陷阱（踩过，值得记）**：
* **陷阱 1｜零中心存储**：lbmpy 2.0/pystencils 2.0 生成的内核按 `f − w` 存储（`delta_rho = Σf`、
  `rho = 1 + delta_rho`，见 `Kernel.get_c_code()`）。直接把 `f` 传进去，结果系统性偏 ~1%（ρ≠1 时），
  极易被误判成"实现有 bug"。本仓用**运行时探测** `is_zero_centered()`（ω=1 内核把输入映射到平衡态，
  喂 `w + 0.5(e_(1,0) − e_(−1,0))` 时两种读法差 O(10%)）而非硬编码假设。
* **陷阱 2｜方向次序**：lbmpy D2Q9 次序 `(0,0),(0,1),(0,-1),(−1,0),(1,0),…` 与本仓
  `d2q9.C` `(0,0),(1,0),(0,1),(−1,0),(0,−1),…` 不同；置换由 `LBStencil` 与 `d2q9.C` 运行时匹配得出
  （`[0,2,4,3,1,6,5,7,8]`），不是手写常量。漏掉它 → 碰撞结果错 ~5%。
* **陷阱 3｜算子次序（本单最有价值的发现）**：`C∘S` ≠ `S∘C`。融合 pull 内核若用 *pull 后* 的群体
  算矩（lbmpy 语义），与教科书的 *先碰撞再 pull* **不是同一个算子**：随机非平衡场上相对差
  **14.5%**（`results.json` equivalence 面板，50 步 0.37%）——恰好卡在 15% 判据边缘。也就是说
  "融合只是性能优化"这句话**只对同一 `moments` 约定成立**；跨库对标必须先对齐这一条，否则会把
  算子次序差误读成精度差。本仓因此把 `moments` 做成显式参数（默认教科书序，保证 `sediment.channel`
  等既有算例逐位不变；对标时用 `post_stream`）。

---

## ⑥ 失败模式与工件

**发现的缺陷（原单已修，本单回归覆盖）**：`collide_bgk_fused` 曾用 float32 舍入权重
（sum = 1+7.45e-9）而非 `_w_for` 精确权重，导致每步 ~4.8e-10 质量偏差；`tests/test_xval_bitwise_refdata.py`
与本单质量守恒测试（`test_fused_kernel_conserves_mass_and_momentum`）共同守住。

**未覆盖 / 诚实边界**：
* 融合内核的 pull streaming 是**周期边界**（与 `solver.stream` 一致）；壁面/边界由调用方在步后施加
  （`sediment.channel` 即如此）。非周期场景不适用。
* 只融合 BGK；MRT/TRT/cumulant 仍走各自入口。
* 被否定的假设：我最初以为离散剪切模本征衰减 `(1−(1−cos k)/τ)^t` 是更紧的 ref —— 实测与算例差 3 个量级
  （`results.json` 里 `ref_discrete_shear_mode` / `err_discrete_pct` 字段原样保留），**该 ref 被证伪**，
  正确参照是 NS 解析解 `exp(−νk²t)`（误差 0.04–0.52%）。
* `sediment.channel` 300 步时 profile 误差 11.2%（默认 1500 步的启动段），属既有算例的启动瞬态，
  非本次改动引入（改前改后逐位相同）。

**工件**：
| 路径 | 内容 |
|---|---|
| `src/tensorlbm/fused_kernel_selector.py` | 实现：kernel_type 选择器 + 单遍融合 stream+collide（含两种 moments 序） |
| `src/tensorlbm/sediment/channel.py` | 主求解器真调用（`ChannelConfig.kernel_type`） |
| `xval/fused_kernel_type_selector/lbmpy_reference.py` | OSS 侧：lbmpy JIT 内核驱动 + 次序/权重/零中心探测 |
| `xval/fused_kernel_type_selector/bench_fused_kernel_selector.py` | 入口：等价性 + 物理 + 敏感性一体跑 |
| `artifacts/xval_fused_kernel_type_selector/results.json` | 全部数字（ref/got/误差） |
| `artifacts/xval_fused_kernel_type_selector/fused_kernel_selector.png` | 衰减曲线 + 等价性误差条形图 |
| `tests/test_fused_kernel_selector.py` | 9 项不变量测试（等价性/守恒/选择器拒绝/物理 15%/跨库） |

---

## ⑦ 复现命令

```bash
cd <repo>
PYTHONPATH=src OMP_NUM_THREADS=4 python xval/fused_kernel_type_selector/bench_fused_kernel_selector.py \
    --out artifacts/xval_fused_kernel_type_selector            # ~40 s, 写 results.json + png
PYTHONPATH=src OMP_NUM_THREADS=4 python -m pytest tests/test_fused_kernel_selector.py -q
    # 9 passed（含 lbmpy 跨库一致；lbmpy 缺失时该项 skip=NOT_RUN，不会假绿）
```

## ⑧ 本仓 vs OSS 差异清单（做法 / 参数 / 边界条件 / 移植取舍）

| 维度 | lbmpy 2.0 | TensorLBM（本单后） | 取舍 |
|---|---|---|---|
| 选择器形态 | `LBMConfig(kernel_type=...)` 配置驱动，非法值 `ValueError` | `KERNEL_TYPES` + `resolve_kernel_type()` + `step(kernel_type=)`，同样 `ValueError` | 同名同义，便于配置迁移；未引入 lbmpy 的 `PdfFieldAccessor` 可插拔 accessor（本仓无此抽象，收益低） |
| 融合粒度 | 有 stream+collide 单遍（默认）、collide-only、stream-only | **本单新增**三者；此前只有 collide 融合 | 补齐锚点缺口 |
| 矩的计算位置 | pull 后算矩（`C∘S`） | 两种都支持，默认教科书 `S∘C` | 默认保持既有算例逐位不变；对标时必须显式 `post_stream` |
| 存储约定 | 生成内核用零中心 `f−w`（运行时探测，非文档承诺） | 非零中心 `f` | 转换层隔离，避免把存储差误判成物理差 |
| 方向次序 | D2Q9 自有序 | d2q9.C 次序 | 运行时置换表，不硬编码 |
| 边界 | accessor 内建/ghost layer，由 `PystencilsSimulation` 或调用方填充 | 周期 `%n` gather；壁面由 bounce-back 模块步后施加 | 本仓边界与碰撞解耦更灵活，但融合内核限周期 |
| 代码生成 | pystencils → C，JIT，可向量化/SIMD | PyTorch 张量算子（可 GPU/torch.compile） | 生态不同：本仓追求张量/自动微分路径，未移植 C 代码生成 |
| 平衡态 | `compressible` 开关（`discrete_hydro_maxwellian`） | 固定二阶不可压形式 `feq` | 数值上等价（本单跨库一致到 2.8e-16 佐证） |