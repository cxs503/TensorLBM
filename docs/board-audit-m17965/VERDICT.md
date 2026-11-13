# 交差核验 · m17772 XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR（TensorLBM）

**审计单**: m17965 · **对象**: mission 17772（`succeeded`, finished 2026-11-13 09:38Z）
**裁判**: `tensor_benchmark_cases` / 三件套（run_artifacts · ref-vs-got · repro_cmd）
**判定**: **L1**（数学/单元验证）· gate 建议 `reopened`（三件套缺 3/3，外部锚点未执行）
**独立复算**: `docs/board-audit-m17965/verify_m17772.py` → `verify_m17772.json`（本仓 sha 8cb29b59，loongarch64）

## 1. 三件套核验（mission 侧）

| 项 | 要求 | 实况 | 结论 |
|---|---|---|---|
| run_artifacts | 跑出的数据/日志/工件文件 | 工作树 `/root/tp-agent/work/m17772` 仅 `xval/fused_kernel_type_selector/REPORT.md`（**untracked**），无 json/log/数据集；`result.evidence.changed_files` 只有该 md | ✗ 缺 |
| ref vs got 数值 | 外部参考 vs 本仓结果 | 报告 ④ 的 "ref" 是 **TensorLBM 自己的 `collide_bgk`**，三项都是自比自（bgk↔matmul、bgk↔fused、fused↔matmul）；声明的外部锚点 lbmpy **未运行**（无 lbmpy 进程/工件/版本记录） | ✗ 无外部 ref |
| repro_cmd | 可执行复现命令 | 报告只有散文，无脚本/命令；`result.repro_cmd` 不存在 | ✗ 缺 |
| 提交状态 | 成果入仓 | `head_commit: null`，`git_status: M src/tensorlbm/solver.py / ?? xval/...` → **一行代码+报告均未提交**；`server_quality_audit: {gate_valid:false, key_present:false}` | ✗ |
| 基准裁判 | `tensor_benchmark_cases` 行 | `SELECT ... WHERE anchor_slug='XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR'` → **0 rows**（全库 75 例，仅同族 `...FUSED-PUSH-PULL-AA-PATTERN` 有例） | ✗ 无法判 |

结论：mission 自称 "已修复缺陷 + 通过 <15% 判据"，但**没有任何一项落在可裁判的工件上**。

## 2. 独立复算（审计侧，全部程序化）

复算 `verify_m17772.json`（float64, 9×32×32, tau=0.8, seed=42, 1/50 步）：

| 量 | mission 报告 | 审计复算（main 8cb29b59） | 判读 |
|---|---|---|---|
| bgk↔matmul 1 步 max\|Δ\| | 2.66e-15 | 2.66e-15 | 一致 |
| bgk↔matmul 50 步 max\|Δ\| | 4.44e-15 | 5.33e-15 | 同量级 |
| bgk↔fused 1 步 max\|Δ\| | **4.81e-08** | **1.78e-15** | **不复现（差 7 个量级）** |
| bgk↔fused 50 步 max\|Δ\| | **1.90e-06** | **4.44e-15** | **不复现** |
| float32 差异 | ~9.5e-07 | 7.15e-07(1步)/2.38e-06(50步) | 同量级 |
| 质量漂移（200 步 rel） | 报告只给预测 5e-5（1e5 步） | 三条路径均 −1.6e-15 | 现已干净 |

原因：报告度量的是 **pre-fix 状态**（`_w_on` float32 权重表，`sum(w)=1+7.45e-9`）。该缺陷的修复在 **main 已由 m17718（merge 9c6fd0c7, 2026-11-13）落地**：
`src/tensorlbm/solver.py:114-120` 现在用 `_w_for(dtype, device)` 并附注释。m17772 的改动 `_w_on → _w_for` 与 main 已有内容**逐字重复**，且从未提交 → 对仓库零净贡献。

**因此 m17772 的数字是"已经不存在的状态"**：既不是复现失败（旧代码确实那样），也不是有效证据（现网不成立）。

## 3. 外部锚点：lbmpy —— 审计代为执行（mission 未做）

`lbmpy 2.0` / `pystencils 2.0` 本机可用。审计构建 lbmpy D2Q9 SRT **纯碰撞算子**（`kernel_type='collide_only'`，ω=1/τ=1.25），与 `collide_bgk` / `_fused` / `_matmul` 在**同一物理场**（feq(ρ=1, TGV u₀=0.05)+1e-3 噪声, Ma=0.086）上对比：

- 通道映射按速度集显式构造并**受矩校验**（ρ/u 回读误差 4.4e-16 / 2.1e-17）→ 映射正确，比较不是错位产物；
- `lbmpy vs collide_bgk`：max|Δ|=1.164e-03，**rel 0.261% < 15% → PASS**；
- `lbmpy vs collide_bgk_fused`：**rel 0.261%**（与参考路径同值 → 融合路径无额外偏差）；
- 残差 0.26% 归因于双方平衡态约定差（lbmpy compressible 三阶项 vs 本仓二阶标准式），非碰撞逻辑差异。

**这是本单新增的真实证据**：m17772 声称的"与 lbmpy 数值一致"首次被实际执行，结论 **PASS**；但它属于审计侧成果，不能追溯计给 m17772。

## 4. 级别裁定（L0–L5）

| 级 | 判据 | 是否 | 依据 |
|---|---|---|---|
| L0 | 文档 | ✓ | `xval/.../REPORT.md`（未提交） |
| **L1** | **数学/单元验证** | **✓（审计复算成立）** | 三路径单步/50 步等价性可独立复算；main 上 ≤5.3e-15 |
| L2 | 独立功能实现（自己的真代码 输入→计算→输出） | ✗ | `collide_bgk_fused/matmul` 早已在 main（非本单产物）；本单净代码=1 行，且与 m17718 重复、未提交 |
| L3 | 系统集成 | ✗ | 无新增调用方；`benchmarks/xval_bitwise_refdata/` 与本单无关 |
| L4 | 基准验证（完整算例+误差分析+可复现） | ✗ | 锚点无 benchmark case（0 rows）、无 run_artifacts、无 repro_cmd、无外部锚点执行 |
| L5 | 科研工程成果 | ✗ | — |

> 报告 ⑤ 的"float32/步数敏感性"属分析叙述，不改变级别；百页报告不改级别，级别=功能实质。

## 5. 过度声明（逐条）

1. "与 lssfau/lbmpy 的 fused stream-collide kernel 数值一致" —— lbmpy 从未运行；报告 ④ 是**自比自**。属越级声明（现已由审计侧实际补做，结论仍 PASS，但那是 m17965 的证据）。
2. "发现的缺陷（已修复）" —— 修复在 main 上由 m17718 落地；m17772 的补丁未提交且重复。归因不实（重复发现可如实记为"rediscovery"，不可记为"已修复"）。
3. "相对误差 ~1e-8" —— 该值是 float32 权重缺陷的产物，不是融合算子的固有误差；main 上该误差为 1e-15 量级。数字方向相反（把缺陷当误差预算）。
4. `succeeded` 状态 + `[auto-approve]` 自动批准 vs `server_quality_audit.gate_valid=false` —— 门禁无效仍记成功，属流程漏洞（建议：三件套缺项不得自动批准）。

## 6. 未跑/未覆盖（诚实边界）

- 未执行 stream+collide **融合**核的 lbmpy 对照（只对照碰撞算子；融合是运算重排，本仓三路径已证到机器精度）。
- 未做网格收敛/时间步收敛（无物理量基准，本项目无可收敛量）。
- 未核 lbmpy 的 stream_pull 语义（超出本锚点主张）。
- 未在 float32 上重复 lbmpy 对照（lbmpy 侧为 double 编译）。

## 7. 处理意见

**request changes / gate REOPENED**：m17772 的交付物不可裁判（三件套 3/3 缺、锚点无 benchmark case、无提交）。若需保留该锚点价值，返修最小路径为：
1. 把 `tensor_benchmark_cases` 登记该锚点用例（metric = 三路径等价 max\|Δ\| + lbmpy 对照 rel，tolerance 15%）；
2. 采用本单 `verify_m17772.py` 作 repro_cmd，`verify_m17772.json` 作 run_artifacts；
3. 报告改写：区分"main 上现状（1e-15）"与"历史缺陷（1e-8）"，归因 m17718。

**L1 confirmed 的部分**：单元级等价性结论在 main 上成立且已被独立复算 —— 这部分价值真实，只是证据链不在 m17772 手里。
