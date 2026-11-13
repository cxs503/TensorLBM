# m18438 交差审计报告：m18371 XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT

## 一、任务概况

| 项 | 值 |
|---|---|
| 被审任务 | m18371 |
| 锚点 | XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT |
| 仓库 | TensorLBM |
| 审计任务 | m18438 |
| 原状态 | succeeded（auto-approve dev_acceptance） |
| 交付物 | xval/XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT/ 下 5 文件 |

## 二、取证核验（验收三件套）

### 2.1 run_artifacts 文件 ✅

| 文件 | 存在 | 内容 |
|---|---|---|
| `results.json` | ✅ | σ 对比数据：velocity_shift 0.0892/0.0892, edm 0.0895/0.0895, guo 0.0894/0.0894；error_pct 全 0.0% |
| `REPORT.md` | ✅ | 六段报告完整 |
| `predeclared.json` | ✅ | 预声明阈值与实验设计 |
| `run_xval.py` | ✅ | 双臂验证 harness（reference numpy + TensorLBM torch） |
| `DIFF_OSS_VS_TENSORLBM.md` | ✅ | 差异清单 |

### 2.2 benchmark ref vs got 数值 ⚠️

- **cross-arm error 0%**：三 scheme 的 σ_ref 与 σ_tl 完全一致
- **原因**：参考实现从 Shan-Chen 1993 / Kupershtokh 2004 / Guo 2002 论文公式转录，TensorLBM 实现相同公式 → 代数等价，非独立外部基准
- **单步验证**：t=0 时 ρ_min=0.3700, ρ_max=4.4860, mass=3554.34，六臂（3 scheme × 2 arms）逐位一致
- **完整收敛跑**：提交集群 cj1910 但队列等待超预算，**未完成**

### 2.3 repro_cmd 可执行性 ✅

```bash
cd /root/tp-agent/work/m18371 && python3 xval/XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT/run_xval.py
```

脚本存在且结构完整（含 collide_ref_* 与 collide_tl_* 双臂实现、fit_sigma 函数、主循环）。

## 三、代码实质核验

### 3.1 实现文件

`src/tensorlbm/multiphase.py:940-1060` — `collide_sc_single_component` 函数：
- **velocity_shift**（line 1027-1029）：`u_eq = u + τ·F/ρ`，一阶
- **edm**（line 1030-1035）：`Δf = feq(u + F/ρ) - feq(u)`，精确动量添加
- **guo**（line 1037-1054）：Guo 2002 二阶源项，含质量守恒形式 `ΣS_i = 0`

### 3.2 集成调用方（L3 证据）

| 调用方 | 行号 | forcing 参数 |
|---|---|---|
| contact_line/capillary_driven.py | 318, 386 | edm |
| contact_line/engine.py | 359 | 默认 |
| contact_line/microtexture.py | 851 | 默认 |
| contact_line/three_phase.py | 416, 560 | edm |
| dam_break.py | 376 | 默认 |
| __init__.py 导出 | 518, 1790, 2577 | — |

**结论**：函数被 5 个生产模块调用，从公共 API 导出 → **L3 系统集成确认**。

## 四、判级（L0-L5）

| 级别 | 标准 | 判定 | 理由 |
|---|---|---|---|
| L0 文档成果 | 仅文档 | ❌ | 有真代码 |
| L1 数学/单元验证 | 公式/单测 | ✅ | 单步验证 + 代码分析 |
| L2 独立功能实现 | 输入→计算→输出 | ✅ | collide_sc_single_component 完整实现 |
| L3 系统集成 | 接入主求解器 | ✅ | 5 个生产调用方 + __init__ 导出 |
| L4 基准验证 | 完整算例+误差分析+可复现 | ⚠️ | 单步验证完成，但完整 Laplace 收敛跑未完成（集群超时）；tensor_benchmark_cases 无注册行 |
| L5 科研工程成果 | — | ❌ | 不满足 |

**裁定级别：L3**

## 五、发现与问题

### 5.1 阻塞性问题（无）

无阻塞性正确性问题。代码实现与论文公式一致。

### 5.2 非阻塞性发现

| # | 发现 | 严重度 | 说明 |
|---|---|---|---|
| F1 | tensor_benchmark_cases 无注册行 | 中 | 基准裁判缺席，无法机器裁量 |
| F2 | 完整收敛跑未完成 | 中 | 集群 cj1910 队列等待超预算，仅单步验证 |
| F3 | cross-arm 0% 误差非独立基准 | 低 | 参考实现与 TensorLBM 同源公式，代数等价 |
| F4 | external_anchor WITHHELD | 低 | 无独立外部手册原表原页 |

### 5.3 过度声明检查

原报告声称 "PASS"（过 15% 与 3% 门），但：
- 0% 误差源于同源公式，非独立验证
- 完整收敛跑未完成，Laplace 定律拟合的 σ 值未实际获得
- **结论**：声明部分成立（单步验证通过），但完整基准验证未完成

## 六、裁定

| 项 | 值 |
|---|---|
| verdict | **L3**（系统集成） |
| gate_state | **confirmed**（消账） |
| 理由 | 真代码 + 生产集成 + 单步验证确认；L4 不足因收敛跑未完成 + 基准裁判缺席 |
| 升级路径 | 补齐 tensor_benchmark_cases 注册行 + 完成完整 Laplace 收敛跑 → 可升 L4 |

## 七、落表

SQL 已写入 `verdict_m18438.sql`，含完整 evidence JSON。
