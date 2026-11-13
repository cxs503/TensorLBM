# 交差审计 m17763 — XVAL-LBM-FREE-ENERGY-PRESSURE-TENSOR-BODY-FORCE

审计单 m17881 · 审计者 board-audit · 被审 m17763（TensorLBM，succeeded，commit `a53d734c`）
裁定：**L2**（独立功能实现）· 处置：**approve with notes**（L2 无需返修；L3/L4 缺口已在交付物内明示）

审计立场：独立复核（adversarial-review），不采信作者自述数字，逐项从原始工件重算。
本单**不重做**任务，只核验 + 判级 + 落表。

---

## ① 取证：三件套是否真存在

| 项 | 声明 | 审计核验 | 结论 |
|---|---|---|---|
| 实现代码 | `src/tensorlbm/free_energy_stress.py`（217 行） | 文件存在；含 `FreeEnergyParams/grad/laplacian/chemical_potential/bulk_pressure/pressure_tensor/stress_divergence_force/grad_mu_force/surface_tension_analytic/planar_profile/droplet_profile`，无 `NotImplementedError`/空壳 | ✓ 真代码 |
| 入口 | `experiments/xval_ludwig_fe_stress/run_xval.py` | 存在，91 行，`sys.path` 注入 `src` 后 import 模块；审计在 m17763 工作树内**实跑成功**（exit 0） | ✓ 可执行 |
| 运行工件 | `results.json` + `figures/xval_fe_stress.png` | results.json 134 行含逐例 metric；png 存在（105 KB） | ✓ 非空 |
| 测试 | `tests/test_free_energy_stress.py`（3 例） | 审计实跑 `pytest -q` → **3 passed, 5.28 s**；断言为不变量型（解析 σ、平衡态体力→0、曲面 σ），阈值 <15% | ✓ 真断言，非「执行即通过」 |
| 复现命令 | `python experiments/xval_ludwig_fe_stress/run_xval.py` | 审计复跑输出与 results.json 全部位数一致，复跑后 `git status` 干净（results.json 字节不变）→ **确定性可复现** | ✓ |
| 外部锚 `ludwig-cf/ludwig` | 从 `.o` DWARF/反汇编取证（快照无源码文本） | 快照 `/tmp/ludwig_tar/ludwig-master` 内 `src/*.c` = **0 个**（「源码不在快照内」属实）；`symmetric.o` nm 得 `fe_symm_str`/`fe_symm_str_v`/`fe_symm_interfacial_tension`；decodedline 在 400–410 区间含 404/405/410；`.rodata.cst8` 中 IEEE-754 双精度实存 0.5、0.75、−8.0、9.0、−2.0、π；`phi_force_stress.o` 得 `pth_kernel`/`pth_stress_set`/`pth_stress`；`fe_force_method.o` 字符串含 `FE_FORCE_METHOD_*` 与 `stress_divergence` | ✓ 符号/行号/常量级佐证成立（**局限**：未做指令级反汇编复核常量↔行号绑定） |

## ② 基准裁判（唯一裁判）核验

```sql
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-LBM-FREE-ENERGY-PRESSURE-TENSOR-BODY-FORCE';
-- 0 rows
SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug ILIKE '%FREE-ENERGY%';  -- 0
-- 表内共 65 行，无本锚任何一行
```

**结论：本锚的声明唯一裁判在库内不存在（0 行）→ L4 无法按基准表确认。**
这与 m17713（XVAL-LBM-BENCHMARK-METHOD-SELECTION）审计发现的缺口同源，属**平台侧登记缺失**，
不计入本单作者责任；但按判据纪律，裁判缺席即 L4 不可确认（不因「报告好看」补位）。

替代性数值核验（仅供判级参考，**不**充当基准表）：

| 量 | ref（解析） | got（交付物） | 审计独立重算 | 误差 | <15% | <3% |
|---|---|---|---|---|---|---|
| σ, ξ=2 (κ=2) | 1.33333 | 1.25214 | 1.252140 | 6.089% | ✓ | ✗ |
| σ, ξ=4 (κ=8) | 2.66667 | 2.62325 | 2.623255 | 1.628% | ✓ | ✓ |
| σ, ξ=8 (κ=32) | 5.33333 | 5.31124 | 5.311243 | 0.414% | ✓ | ✓ |
| σ 曲面 R=32, ξ=4 | 2.66667 | 2.62282 | — (复跑一致) | 1.644% | ✓ | ✓ |
| σ 曲面 R=64, ξ=8 | 5.33333 | 5.31105 | — (复跑一致) | 0.418% | ✓ | ✓ |
| 曲率平衡 ∫(P_NN−P_TT)/r dr vs σ_flat/R (R=32) | 0.08200 | 0.08244 | — | 0.534% | ✓ | ✓ |
| 同上 (R=64) | 0.08300 | 0.08343 | — | 0.518% | ✓ | ✓ |
| 平衡态体力残差 max\|F_stress\|/(σ/ξ²) | 0 | 5.30e-2 / 2.09e-2 / 5.79e-3 | — | 5.3% / 2.1% / 0.58% | ✓ | 部分 |
| 两式力场逐点相对差（ξ=4） | 0（连续恒等） | 1.042 | — | **104%** | ✗ | ✗ |

- **独立重算方法**：审计侧 numpy 在**同一解析 tanh 场**上用中心差分 + ±8ξ 窗口求和，不 import tensorlbm；
  结果与交付物 `results.json`（6.0895e-2 / 1.62795e-2 / 4.14202e-3）**逐位一致** → 数字非手抄。
- **周期性伪界面控制复核**：全周期域梯形积分（含接缝）得 3.25 / 10.62 / 37.31（相对误差 144%–600%），
  远超解析值 → 证明 ±8ξ 开窗是**必要**控制（作者「失败模式1」属实，不是粉饰）。
- **解析恒等式审计独立推导**：由 p0 = φf′−f = a/2 φ² + 3b/4 φ⁴ 与 d = p0 − κφ∇²φ − ½κ|∇φ|²，
  可推得 ∂_b P_ab = φ ∂_a μ **恒等**（p0′ 中的 3bφ³ 与 φ·3bφ² 精确相消），故 −∇·P ≡ −φ∇μ 在连续极限为恒等式
  → 104% 相对差是离散残差被小分母放大的人工指标，交付物的 honest-FAIL 标注**正确**，非造假亦非物理不符。

## ③ 判级（L0–L5）

| 级 | 判据 | 本单 | 证据 |
|---|---|---|---|
| L0 文档 | — | 过 | 交付物远超文档 |
| L1 数学/单元验证 | tests 非空 | **过** | 3 passed；解析 σ 公式审计侧独立推导正确 |
| L2 独立功能实现（输入→计算→输出） | impl_files 非占位 + entry + run_artifacts | **过** | `src/tensorlbm/free_energy_stress.py:1-217`；`run_xval.py` 实跑 exit 0；`results.json` + png |
| L3 系统集成 | integration_callers 真调用 | **未过** | `grep -rn 'free_energy_stress\|stress_divergence_force' src/`（除模块自身）**0 命中**；`src/tensorlbm/__init__.py` 未导出；`multiphase.py:1223 free_energy_step` 体内部（第 66–69 行）**仍为** `Fx=-phi*grad_mu_x` → 新体力无任何产品代码路径消费 |
| L4 基准验证 | L3 + benchmark(ref vs got) + 复现 | **未过** | ①L3 未过；②基准表 0 行（无机器裁判）；③σ 测自**给定解析场**而非求解器输出（`run_xval.py` 不积分时间、不 import 求解器）→ 静态离散化校验，不构成完整算例 |
| L5 科研工程 | L4 + 外部锚可裁决 | **未过** | 外部锚为「实现取证」，无可用 OSS 算例跑分（作者已声明） |

**判级 = L2**。附百页报告不加级；级别取功能实质。

## ④ 严重度排序发现（评审结论）

| ID | 严重度 | 发现 | 作者自知 | 处置建议 |
|---|---|---|---|---|
| F1 | 阻断 L3 | 体力未接入求解器 | 是（文档 ⑤/差异清单明示「未改求解器…列入后续」） | 后续单：`free_energy_step(force_scheme="stress_divergence")` 可选臂 + 与 `-φ∇μ` 的算例级对照 |
| F2 | 阻断 L4 | 锚无 `tensor_benchmark_cases` 行 | 否（作者按解析基准自裁） | 平台侧为该类 XVAL 锚登记基准行，否则所有 XVAL 单永远卡 L4 |
| F3 | 次要（指标） | `force_diff_rel` 分母是 O(h²) 残差 → 104% 无物理含义 | 是（已标 honest FAIL） | 报告改以绝对值/σ ξ⁻² 归一为准，撤回该列或标 ill-conditioned |
| F4 | 次要（覆盖） | 平面 1D 剖面下 5pt≡iso9（4/12+1/12+1/12=1/2 代数恒等），两模板对照在该场空心；曲面仅用 iso9 | 是（⑤ 已声明局限） | 曲面/多维补 5pt vs iso9 |
| F5 | 次要（范围） | σ 取自解析场，非求解器输出场 | 是（③ 「静态场对照，不积分时间」） | 接 L3 后用真实 free_energy_step 场测 Laplace 压差 |
| F6 | 佐证成立 | 解析 σ 公式、div-P↔−φ∇μ 恒等式正确；OSS 常量/符号实证 | — | — |
| F7 | 提示 | 以 15% 带裁「通过」；若按 <3% 线，ξ=2 的 6.09% 不达标 | 否（未显式回答 <3%） | 报告须显式区分两档判据 |

**虚标检查**：未发现虚标。头条数字（6.09% / 1.64% / 0.58–5.3% / 0.41%）逐项复算为真；
104% 未被作者当作「通过」使用。唯一可指摘处是 L3/L4 语义边界 —— 作者文档与 commit 均未宣称
「已接入求解器」或「基准跑分」，故不构成夸大。

## ⑤ 独立性限制 / 未跑的检查（unknown, not passed）

- 未做 ludwig `.o` 指令级反汇编，把 `.LC` 常量逐条绑到源行（仅到 符号/行号/常量存在 级）。
- 未跑任何以 `stress_divergence_force` 驱动的 LBM 时间推进算例（无该入口，属 F1/F5）。
- 未验 3D（D3Q27）压强张量（交付物声明仅 2D）。
- 环境：审计宿主 128 核共享，纯静态计算无资源瓶颈；审计复跑均在 m17763 工作树内、未改动被审代码。

## ⑥ 处置

approve with notes（L2）。**L3/L4 缺口无需返修即可消账到 L2**（gate ≥2 即 confirmed）；
真正缺失的一环是平台侧基准登记（F2）与后续集成单（F1），已在发现表给出修复路径。

> 工件：`docs/board_audit/m17881/audit_numbers.json`（本次审计全部重算数字与命令）
> `docs/board_audit/m17881/figures/m17881_audit_verification.png`
> verdict 落表：`tensor_maturity_verdicts`(repo=TensorLBM, scope='m17763 XVAL-LBM-FREE-ENERGY-PRESSURE-TENSOR-BODY-FORCE', level=2, verdict_by='board-audit')