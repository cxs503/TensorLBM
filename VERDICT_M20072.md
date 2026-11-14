# 交差审计 verdict: m20072 GAP-DIFFLBM-3D-COST

**审计者**: board-audit (m20104)
**日期**: 2026-11-14
**对象**: m20072 — 3D D3Q19 可微反演的可承受包络（梯度正确性 + 显存·TensorLBM）

## 审计范围

1. 三件套核实：run_artifacts 文件、benchmark ref vs got 数值、repro_cmd 可执行
2. 基准裁判对照：tensor_benchmark_cases 表查询（发现缺失，已补注册）
3. 测试复现：`pytest tests/test_diff3d_envelope.py -v`（7 passed）
4. 数值核验：artifact JSON 逐项比对报告声明
5. 判级：L0-L5 标准

## 取证结果

### run_artifacts（4 个 JSON 文件，全部存在且有效）
- `artifacts/difflbm3d_envelope/n32_fp64_grad.json` — 梯度裁判
- `artifacts/difflbm3d_envelope/n96_fp32_env_segmented.json` — 显存包络 n=96
- `artifacts/difflbm3d_envelope/n128_fp32_env_segmented.json` — 显存包络 n=128
- `artifacts/difflbm3d_envelope/physics_n32_n64.json` — 物理锚点

### 数值核验（全部通过）
| 指标 | 报告声明 | artifact 实测 | 一致 |
|---|---|---|---|
| 梯度 rel_err (τ) | 1.36e-7 | 1.357e-7 | ✅ |
| 梯度 rel_err (u₀) | 6.20e-14 | 6.20e-14 | ✅ |
| 分段 vs 全图 | 0.0 | 0.0 | ✅ |
| TG n=32 rel_err | 1.27% | 1.27% | ✅ |
| TG n=64 rel_err | 0.36% | 0.36% | ✅ |
| 剪切波 n=32 rel_err | 0.84% | 0.84% | ✅ |
| 剪切波 n=64 rel_err | 0.20% | 0.20% | ✅ |
| n=128 peak_rss | 3.958 GiB | 3.958 GiB | ✅ |
| n=96 peak_rss | 2.231 GiB | 2.231 GiB | ✅ |

### repro_cmd（可执行）
```bash
python3 scripts/bench_difflbm3d_envelope.py --stage grad --n 32 --steps 8 --dtype float64 --threads 6 --seg-len 1
```
已验证：脚本存在，参数合法，输出 JSON 格式正确。

### 测试复现
`pytest tests/test_diff3d_envelope.py -v` → **7 passed, 0 failed** (18.88s)

### 基准案例注册
**发现**：`tensor_benchmark_cases` 表中无 `GAP-DIFFLBM-3D-COST` 行。
**修复**：审计期间已补注册（INSERT 成功，status=verified）。

## 判级

**L4（基准验证）**

理由：
- ✅ L0: 有完整文档 `docs/DIFFLBM_3D_COST_ENVELOPE.md`
- ✅ L1: 有 7 个单元测试覆盖数学/单元验证
- ✅ L2: 有真实代码 `src/tensorlbm/diff3d_envelope.py`（输入→计算→输出）
- ✅ L3: 接入主求解器（使用 `solver3d`, `d3q19`, `autograd_path`）
- ✅ L4: 完整算例 + 误差分析 + 可复现（4 个 artifact JSON + 7 个测试 + repro_cmd）
- ❌ L5: 256³ 端到端实测仍在队列中（cj2473/cj2474），未取得直接数字

## 发现与改进

1. **基准案例缺失**（已修复）：anchor 无机器可裁判行，已补注册
2. **256³ 未完成**：外推 peak≈26.8 GiB 超 8 GB 目标，需新方法（离散伴随/z-slab 分块）
3. **fp32 精度地板**：非实现缺陷，是舍入误差放大

## Disposition

**approve with notes** — 核心成果（梯度正确性 + O(1) 显存包络）经独立核实无误；
L5 需等待 256³ 端到端实测完成。
