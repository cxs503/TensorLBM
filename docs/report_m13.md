# Mission 13 补报：SDAA 扫荡 — aircraft_icing_naca0012_rime

> **补报单 (mission 91)**：原 mission 13 已成功交付（honest-FAIL, 6/11 PASS），但缺论文级进展报告。
> 本报告从工件 `artifacts/sdaa_aircraft_icing_naca0012_rime.json` 程序化提取全部数字，零手抄。

---

## 1. 目的

对 TensorLBM `benchmarks/verified/aircraft_icing_naca0012_rime` 案例在 **SDAA 异构加速卡**
（`/dev/tcaicard24`, torch 2.7.1）上进行独立判据扫荡（SDAA sweep），验证结冰求解链
（Lagrangian 液滴 + Eulerian 沉积水膜 + Macklin 密度 + 冰形生长）在 SDAA 设备上的
**内部一致性**。判据族来自案例 README 的 gate family，覆盖：

- **G1** Lagrangian 质量闭合（最差案 ≤1e-12）
- **G2** Eulerian 闭合（相对 ≤1e-6）
- **G7** IC-A 逐位确定性（两次运行 15 数组 sha256 全等）
- **G9a–d** LWC 线性族（log-log 斜率 / R² / 比例偏差 / 单调性）
- **G10a–d** 网格阶梯 + 三点 Richardson 外推 vs IC-A 先验

## 2. 方法

### 2.1 独立运行脚本

编写 `scripts/sdaa_sweep_aircraft_icing_naca0012_rime.py`（不改业务代码），以
`device="sdaa"` 计算所有张量。入口走 `compile_mode=None`（纯 torch 算子），
`TRITON_INTERPRET=1` 作为 belt-and-braces 环境保护。

### 2.2 运行矩阵与预算

墙钟预算 1150 s（≈20 min），按门覆盖最大化排程：

| 运行 | 格子 | steps+warmup | LWC (g/m³) | 实测墙钟 (s) | 覆盖门 |
|---|---|---|---|---|---|
| `b1_prod` | 320×160 | 3000/6000 | 0.50 | 356.9 | G1, G2, G9(arm) |
| `m20_lwc025` | 320×160 | 3000/6000 | 0.25 | 352.3 | G1, G2, G9(arm) |
| `m20_lwc10` | 320×160 | 3000/6000 | 1.00 | 333.3 | G1, G2, G9(arm) |
| `ladder_L1` | 320×160 | 3000/6000 | 0.50 | — (复用 b1_prod) | G1, G2 |
| `det_run1/2` | — | — | — | SKIP (预算) | G7 |
| `ladder_L2/L3/L4` | — | — | — | SKIP (预算) | G10a–d |

**总实测墙钟**：1047.7 s（预算内）。`ladder_L1` 与 `b1_prod` 配置逐字段相同，
按 `reused_from="b1_prod"` 复用。

### 2.3 SDAA 设备环境

| 参数 | 值 |
|---|---|
| 设备 | `/dev/tcaicard24` |
| torch 版本 | 2.7.1 |
| `torch_sdaa` 模块 | ✓ |
| SDAA 设备数 | 1 |
| smoke matmul | ✓ |
| 显存占用 (smoke 后) | 1,064,960 B |
| TRITON_INTERPRET | 1 |
| compile_mode | None (纯 torch ops) |

## 3. 关键结果

### 3.1 三案实跑结果

| run | 格子 | LWC (g/m³) | β_e,peak | Euler 沉积 (kg) | t_max (mm) | 上/下角峰 (mm) | t_stag (mm) | n_ice | 冰面积 (% c²) |
|---|---|---|---|---|---|---|---|---|---|
| `b1_prod` | 320×160 | 0.50 | 0.920717 | 1.753473e-03 | 8.334375 | 4.167 / 8.334 | 12.5016 | 15 | 0.09155 |
| `m20_lwc025` | 320×160 | 0.25 | 0.919848 | 8.843855e-04 | 4.167188 | 0.000 / 4.167 | 8.3344 | 5 | 0.03052 |
| `m20_lwc10` | 320×160 | 1.00 | 0.928712 | 3.474131e-03 | 20.835938 | 12.502 / 20.836 | 25.0031 | 37 | 0.22583 |

### 3.2 判据逐条比对（6 PASS / 5 FAIL）

| 门 | 判据 | SDAA 测得 | 结果 | 备注 |
|---|---|---|---|---|
| G1 Lagrangian 闭合 | ≤1e-12 | 0.00e+00 | **PASS** | 3 案 + 复用 L1 全为 0 |
| G2 Eulerian 闭合 | ≤1e-6 | 1.769e-08 | **PASS** | 最差案 b1_prod |
| G7 逐位确定性 | 全等 | — | **FAIL** | NOT_RUN：det 对未跑到 |
| G9a log-log 斜率 | [0.97, 1.03] | 0.9870 | **PASS** | (NVIDIA prior: 0.9857) |
| G9b R² | ≥0.999 | 0.9999 | **PASS** | k=3.483 kg/(kg/m³) |
| G9c 比例偏差 | ≤3% | 1.554% | **PASS** | 逐点 1.554/0.676/0.266% |
| G9d 单调性 | 严格递增 | true | **PASS** | 8.844e-04 < 1.753e-03 < 3.474e-03 |
| G10a 阶梯单调 | L1<L2<L3 | — | **FAIL** | NOT_RUN：ladder 不完整 |
| G10b Richardson L3 | \|err\|≤3% | — | **FAIL** | NOT_RUN |
| G10c Richardson L4 | \|err\|≤3% | — | **FAIL** | NOT_RUN |
| G10d Richardson 元判据 | 相对 ≤1e-9 | — | **FAIL** | NOT_RUN |

**汇总**：`n_pass=6, n_total=11, all_pass=False` → **verdict: honest-FAIL**

### 3.3 与 NVIDIA 先验对照（SDAA vs RTX 5090）

| 量 | SDAA | NVIDIA 先验 | 相对偏差 |
|---|---|---|---|
| t_max (mm) | 8.334375 | 8.334375 | **0.000%** |
| t_stag (mm) | 12.501563 | 12.5015625 | **0.000%** |
| n_ice / 冰面积 | 15 / 0.09155% | 15 / 0.09155% | **0.000%** |
| Euler 沉积 (kg) | 1.7535e-03 | 1.7577e-03 | −0.242% |
| β_e,peak | 0.920717 | 0.911886 | +0.968% |
| cd0 | 0.060514 | 0.058272 | +3.85% |
| cd_end | 0.097459 | 0.095279 | +2.29% |
| **cl0** | 0.199712 | 0.154396 | **+29.35%** |
| **cl_end** | 0.073852 | 0.132712 | **−44.35%** |

**关键发现**：冰形积分量（t_max / t_stag / 冰胞数 / 冰面积）在 SDAA 与 NVIDIA **逐位相同**，
沉积质量差 0.24%，β_pk 差 0.97%——结冰链在 SDAA 上等价。但流场气动量 cl0/cl_end
偏差 +29%/−44%，虽不在本案例门族内，属设备敏感性，需后续定位。

### 3.4 墙钟预算分析

SDAA 单步成本：9000 步 / 356.9 s = **39.65 ms/step**；NVIDIA RTX 5090 为 3.20 ms/step，
SDAA 慢 **≈12.4×**。

| 未跑运行 | 格子 | steps | 外推墙钟 |
|---|---|---|---|
| ladder_L2 | 640×320 | 18000 | ≈2855 s |
| ladder_L3 | 890×445 | 24934 | ≈7648 s |
| ladder_L4 | 1280×640 | 39940 | ≈25340 s |
| **阶梯小计** | — | — | **≈35844 s (≈10 h)** |

## 4. 审计与设备链路

### 4.1 设备证据链

- `device_evidence_line` 含 `/dev/tcaicard24`
- `sdaa_smoke_matmul_ok = true`
- 所有张量在 `device="sdaa"` 上计算
- 产物 npz 含 sha256 校验（`runs[*].array_sha256`）
- Triton 路径：`compile_mode=None`，未进入 triton kernel；`TRITON_INTERPRET=1` 为保护性设置

### 4.2 零手抄复核

从 JSON 中 3 案存储值重算 G9a/G9b/G9c：
- G9a = 0.9869524395629171
- G9b = 0.9998826455789217
- G9c = 1.554259549798597

与 JSON 记录值差 0（逐位相等），判据数值确由脚本现算。

### 4.3 产物清单

- `artifacts/sdaa_aircraft_icing_naca0012_rime.json` — 主证据 JSON
- `artifacts/sdaa_runs/aircraft_icing_naca0012_rime/{b1_prod,m20_lwc025,m20_lwc10}/result.npz` — 设备侧产物
- `docs/sdaa_sweep/aircraft_icing_naca0012_rime.md` — 原始扫荡记录

## 5. 局限与未完成

1. **G7 (确定性) 未跑**：需 2 次完整运行做逐位比对，墙钟预算不足。
2. **G10a–d (收敛/外推) 未跑**：需 L2/L3/L4 三级网格阶梯，外推 ≈10 h。
3. **cl0/cl_end 设备差异未定位**：+29%/−44% 偏差超出案例门族范围，需逐步场对比。
4. **未使用 compile_mode 提速**：会改变数值口径，本次不做。
5. **ladder_L1 复用 b1_prod**：非独立运行，覆盖面等价但非严格独立。

### 结论

- **设备链路**：SDAA 上结冰求解链可跑通，6/11 内部一致性判据 PASS，冰形与 NVIDIA 逐位一致。
- **整体判定**：**honest-FAIL** — G7 与 G10a–d 在 20 min 墙钟内无法取得机器值。
- **后续建议**：放宽墙钟至 ≈10.5 h 或启用 compile_mode 提速以关门 G7/G10；
  排查 cl0/cl_end 设备敏感性根因。

---

*生成时间: 2026-10-29T12:55:07Z (原始 JSON) | 补报: mission 91*
*数据来源: artifacts/sdaa_aircraft_icing_naca0012_rime.json (程序化提取)*
