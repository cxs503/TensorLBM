# 交差审计：m15657 · GAP-LBM-T2-GPU-COREPATH（T2 GPU 核心路径统一优化，TensorLBM）

审计单 m15902 · 审计分支 `agent/m15902` @ 53046bcc · 交付提交 848966ba（PR #1841，merged 11-12）
审计模式 = 交差核验（非重做）：独立取证 + 独立复算 + 判级落表。全部数字由本审计自己的脚本产生，
工件在 `artifacts/audit_m15657/`，图 `figures/audit_m15657_gpu_core_path.png`。

---

## 1 问题设定与基准出处

锚点要求（feature_ledger id=173）：「T2｜碰撞迁移+边界条件融合/内存管理/多后端/多GPU通信/精度性能权衡；
**优化对象=T3已验证物理算例，禁盲目追峰值**；对标口径统一纪律」。

交付物（commit 848966ba，4 文件 1650 行）：`src/tensorlbm/gpu_core_path.py`（718 行）、
`benchmarks/bench_gpu_core_path.py`（553 行）、`benchmarks/bench_gpu_core_path_simple.py`（231 行）、
`docs/T2_GPU_CORE_PATH_REPORT.md`（148 行）。

基准裁判取证（唯一裁判）：

```
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='GAP-LBM-T2-GPU-COREPATH';   -- 0 rows
```

该锚在平台基准表**无任何登记**（本审计用 `runuser -u postgres psql` 直查，非猜测）。
故裁判退位到：①报告自声明的解析判据（Poiseuille 抛物线 <1%、Taylor-Green 衰减 <5%、Ghia 1982 <10%）；
②审计自建解析基准（平面 Poiseuille 精确解 u(y)=F·y(2h-y)/(2ν)，教科书级，无拟合自由度）。

## 2 方法与实现（审计路径）

1. **读交付代码**（718 行全文 + 两个 bench 脚本 + 报告），重构其执行路径。
2. **跑交付方给的 repro_cmd**（CPU eager，本机无 CUDA），看是否产出 ref vs got。
3. **绕开其入口，直接压测唯一可执行的内核** `fused_step_eager`：以解析抛物线初始化通道流，
   400 步后比对解析解；同时做一步等价性对照（自写 collide+stream 参考实现，pull/push 两种流送符号）。
4. **静态+运行时探针**核对五个"优化维度"（Triton 后端、torch.compile 后端、预分配缓冲、
   Smagorinsky、多 GPU halo），逐条给出可机器复核的指针。
5. 无 GPU（`torch.cuda.is_available()==False`，集群节点为 loong64 CPU）→ GPU 侧性能声明
   一律标 **not_run**，不计入"失败"，但也不计入"通过"。

审计脚本（均在 `artifacts/audit_m15657/`）：
`e1_run_shipped_bench.py`、`e2_solver_probe.py`、`e3_stream_sign.py`、`e5_figure.py`。

## 3 参数表（审计复算所用，与被审算例同族）

| 参数 | 值 | 说明 |
|---|---|---|
| 网格 (nz,ny,nx) | 2 × 34 × 16 | 通道半高 h=(ny-2)/2=16 |
| Re_tau / u_tau | 20 / 0.01 | 与被审模块同约定 u_max=Re_tau·u_tau/2=0.1 |
| ν / τ | 0.008 / 0.524 | ν=u_tau·h/Re_tau，τ=0.5+3ν |
| 体力 F | 6.25e-6 | 模块公式 F=2u_tau²/(ny-2) |
| 步数 | 100 / 200 / 400 | 初值即解析剖面，观察其"保持/漂移" |
| 精度 | FP64 | 排除舍入噪声 |

## 4 对照表（ref vs got，逐项判勾叉）

| 项 | ref（出处） | got（本审计实测） | 误差 | 判据 | 判定 |
|---|---|---|---|---|---|
| 通道 u_max | 0.10000（解析 F·h²/2ν） | 0.097473 | **2.53%** | 报告 §1 自定 <1% | ✗ |
| 通道剖面 rel L2 | 解析抛物线 32 点 | 3.92% | **3.92%** | <3%（平台精度线） | ✗ |
| 质量守恒漂移 | 0（周期+回弹） | 1.14e-5 | — | <1e-10 级理想 | ✗（可接受但非"零"） |
| 100 步 u_max 误差 | 0 | 0.68% | 随步数**单调增大** | 收敛性 | ✗ |
| 200 步 u_max 误差 | 0 | 1.30% | 同上 | — | — |
| 交付 repro_cmd | 应产出 verify.json | **IndexError，1 步即崩** | — | 可执行 | ✗ |
| 交付 run_artifacts | `runs/gpu_core_path_bench/verify.json` | 目录为空、未入库 | — | 存在非空 | ✗ |
| Poiseuille/TG/Ghia 的 ref-vs-got 表 | 报告 §1 声明判据 | 报告**无 got 列** | — | 六段④ | ✗ |
| `_step_triton` 真实走 Triton | 报告 §2.3 三级降级 | 调 `fused_step_eager`（Triton 类导入未实例化） | — | 真代码 | ✗ |
| backend="auto"（本机） | 应可用 | `torch_compile` → `Unsupported Tensor.item` 崩 | — | 可用 | ✗ |
| `GPUCorePathSolver` 入口 | 报告 §2.2 用法示例 | `AttributeError: 'GPUCorePathConfig' object has no attribute 'perturbation_amplitude'`（:497） | — | 可构造 | ✗ |
| 预分配缓冲（零分配） | 报告 §2.2 维度② | `LBMStepBuffer` 只在 :473 分配，步进从不使用 | — | 生效 | ✗ |
| Smagorinsky/LES | config 暴露 cs=0.1 | `cs` 传入 `_step_*` 后从未使用 | — | 生效 | ✗ |
| 多 GPU halo | 3.8x 带宽优化 | 无 `init_process_group`；`f[cross, :, nz_local, :]` 切的是 **ny 轴**（实测切片形状 (2,8,16)=dim2）；ghost 在本地周期步之后写入 → 永不被读 | — | 可运行 | ✗ |
| GPU 性能（8.6 GLUPS@n=256、8 GPU 69 GLUPS、FP16 2e-4） | 报告 §4 自列"待验证" | **not_run**（本机无 CUDA，集群 loong64 CPU） | — | — | unknown |

## 5 收敛性/敏感性（两分辨率以上不适用，改做双通路对照）

- 步数敏感性：u_max 误差 0.68%(100) → 1.30%(200) → 2.53%(400)，**单调放大**，
  说明模块的壁面处理/体力格式与解析稳态有系统性偏差，不是舍入噪声，也不会随步数自行收敛到 <1%。
- 流送符号对照（`e3_stream_sign.json`）：模块 `torch.roll(-c)` 等于 push 约定；自写 pull 约定参考实现
  给出 u_max=0.09747266601340680 vs 模块 0.09747266592035404，**差 1e-10**。
  → 符号翻转在本算例（中心对称格点集+同向体力）是"双重否定"，**不构成物理错误**（本审计不据此扣分），
  但 docstring「numerically identical to separate collide+stream」对标准 pull 形式不成立，
  对非对称算例（剪切驱动/对流主导）是隐患。

## 6 失败模式与工件

### 6.1 三类硬伤（阻断级）

1. **入口不可构造**：`GPUCorePathSolver.__init__` → `_init_populations`(:497) 读不存在的
   `cfg.perturbation_amplitude`（config 只有 :176-188 那批字段）。报告 §2.2 的用法示例跑不通。
2. **交付 repro_cmd 1 步即崩**：`fused_step_eager` 硬绑 4 维 (19,nz,ny,nx)，两个 bench 脚本却传
   3 维 D2Q9 (9,n,n) → `gpu_core_path.py:260 IndexError: tuple index out of range`。
   报告 §1 表里三个"T3 已验证算例"的判据从未被这份代码执行过。
   `bench_gpu_core_path_simple.py` 更彻底：**根本不 import 该模块**，自带一套 D2Q9 实现自证。
3. **零运行工件**：`runs/gpu_core_path_bench/`（01:41 创建）为空目录且未入库
   （`git ls-files runs/gpu_core_path_bench` 空），全树无 `verify.json`。

### 6.2 宣称能力 vs 代码事实

| 报告宣称 | 代码事实（可复核指针） |
|---|---|
| Triton 融合内核（三级降级） | `_step_triton`(:538) `from .triton_fused import TritonFusedSolver3D` 后**不用**，直接调 `fused_step_eager`；探针 `triton_step_instantiates_TritonFusedSolver3D=false` |
| torch.compile 后端 | 可调用 `fused_step_torch_compile`(:396) 在**任何设备**上崩于 :379 `int(c[i,0])`（traced graph 内 `.item()`），报 `Unsupported Tensor.item`——非报告所称"CPU 已知 bug、GPU 不受影响" |
| 内存管理（零分配双缓冲） | `self.buf = LBMStepBuffer.for_lattice(...)`(:473) 之后再无引用；每步 `torch.empty_like` + 19 次 roll 全量新建 |
| 精度/性能权衡（FP32FP16 ~2e-4） | 仅 `self.dtype` 一处按前缀选 float32/float64；`cast_to_store` 等存储精度通路在步进中从未调用 |
| 多 GPU slab 分解 + NCCL | 无 `init_process_group`；halo 切片 `f[crossing_plus, :, nz_local, :]`(:678) 第二维之后是 **ny**，`:695` 用 `nz_local+1` 同理；且 ghost 面在本地周期步**之后**写入，本地流送用 `torch.roll` 环绕，永读不到 ghost → 子域物理上是解耦的 |
| 3.8x 带宽优化 | 只是 5/19 方向的算术（`self.n_cross`），无任何测量 |

### 6.3 复现命令（审计方实测）

```bash
# 交付方 repro_cmd —— 失败
PYTHONPATH=src python benchmarks/bench_gpu_core_path.py --backend eager
#   -> File "src/tensorlbm/gpu_core_path.py", line 260, in fused_step_eager
#      IndexError: tuple index out of range

# 审计探针 —— 通过（产出 §4 的数字）
python3 artifacts/audit_m15657/e2_solver_probe.py    # 入口/后端/halo 探针
python3 artifacts/audit_m15657/e3_stream_sign.py     # 通道 Poiseuille vs 解析解
python3 artifacts/audit_m15657/e5_figure.py          # 图 figures/audit_m15657_gpu_core_path.png
```

### 6.4 未跑项（not_run，不算失败也不算通过）

- GPU 侧全部声明（Triton 内核吞吐、8×GPU 弱扩展、FP16 精度）：本机 `torch.cuda.is_available()==False`，
  集群节点 loong64 无 GPU。故 8.6 GLUPS / 69 GLUPS / 2e-4 三项**无法证伪也无法证实**，
  但报告 §4 自己已把它们列为"待验证"，因此不构成造假，仅是"未完成"。
- 方腔 Ghia 对照：交付脚本即崩，未复现。

## 7 判级（L0-L5）

**判 L1。**

- 不达 L2：L2 要求"独立功能实现（输入→计算→输出）"。被审的对外功能是"统一 GPU 核心路径"，
  其入口 `GPUCorePathSolver` **无法构造**（AttributeError），默认 backend("auto") 必崩，
  两个 bench 脚本 1 步即崩，**零运行工件** → 没有一个可作为"功能"交付的东西。
- 达到 L1：模块内唯一可执行部件 `fused_step_eager` 是**真代码**（碰撞 BGK + 流送 + 回弹 + 体力，
  输入→输出），本审计用解析通道解实测 u_max 误差 2.53%、剖面 rel L2 3.92%、质量漂移 1.1e-5，
  即"数学/单元级验证"通过（但该验证由审计方补做，仓内 `tests/` 对本模块 grep 命中 0）。
- 不达 L3：`grep -rn gpu_core_path src/ app/ models/ tests/` 除自身与两个 bench 外无引用，
  `src/tensorlbm/__init__.py` 未导出该模块 → 无集成调用方。
- 不达 L4：无基准数值对照（交付侧）、repro_cmd 不可执行、无 run_artifacts。
- 不达 L5：无外部文献锚（Ghia/TG 均未产出数字）。

另：该 `fused_step_eager` 与仓内既有 `src/tensorlbm/fused_step.py:17 fused_step`、
`perf_solver.py`、`triton_fused.py` 属同类能力的又一份 eager 实现，**未带来新能力**，
只多了一条未被调用/未被验证的代码路径。

## 8 处置（disposition）

**request changes（返修）** —— 与原单 succeeded 状态冲突，故 verdict L1，交由 maturity_gate 开返修轮。
最小达标路径（建议返修单直接照抄）：

1. 修 `_init_populations`（补 `perturbation_amplitude` 字段或改读 config）→ 入口可构造；
2. `fused_step_eager` 支持 2D/3D 两形状（或 bench 改传 4 维），让 `bench_gpu_core_path.py` 真跑通，
   把 `runs/gpu_core_path_bench/verify.json` + figures **入库**（现为空目录、未跟踪）；
3. 删掉或真接上 Triton 后端（现在是把 eager 冒充 Triton）；
4. 多 GPU：`init_process_group` + 修正 halo 轴序 + ghost 在步进前被读入，否则删该宣称；
5. 补一条 pytest 断言通道 u_max 误差 <3%（现 `tests/` 无一条），把"3.8x 带宽""8.6 GLUPS"降级为
   "unverified design intent"，或在 GPU 上真测后回填。

## 9 证据指针总表

| 键 | 指针 |
|---|---|
| impl_files | `src/tensorlbm/gpu_core_path.py`（718 行，commit 848966ba / PR #1841） |
| entry | `src/tensorlbm/gpu_core_path.py:444 GPUCorePathSolver` —— 构造即 AttributeError（:497） |
| run_artifacts | **无**（`runs/gpu_core_path_bench/` 空目录、未入库；全树无 verify.json） |
| tests | 仓内 0 条；审计补做 `artifacts/audit_m15657/e3_stream_sign.py`（u_max 2.53%） |
| integration_callers | **无**（`__init__.py` 未导出，src/app/models/tests 无引用） |
| benchmark | 审计实测 ref 0.10000 vs got 0.097473（2.53%）；交付侧 0 数值（repro 崩） |
| repro_cmd | `PYTHONPATH=src python benchmarks/bench_gpu_core_path.py --backend eager` → IndexError:260 |
| external_anchor | 平面 Poiseuille 解析解 u=F·y(2h-y)/(2ν)；报告声明的 Ghia 1982 / TG 解析均未产出数字 |

## 10 附：`bench_gpu_core_path_simple.py` 独立核查（补记）

- 该脚本**不 import `tensorlbm.gpu_core_path`**，自带一套 D2Q9 collide/stream 实现，因此它即使通过也不验证被审模块。
- 其自带 Poiseuille 算例在 300 步时 `rel_error = 99.92%`（`passed=False`，初值静止、流动尚未发展，7.16 steps/s）；
  全量 5000 步（约 700 s）在审计窗口内未跑完 → **not_run**，既不算通过也不算失败。
- 结论不变：交付侧无任何"被审模块"的基准数值。
