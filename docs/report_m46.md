# Mission 46 补报：SDAA 验证案例扫荡 — sphere_re100_mem

> **状态：honest-FAIL**（SDAA 设备 OOM，两档运行均未完成；GPU 基线验证已通过）  
> **案例**：`benchmarks/verified/sphere_re100_mem`（TensorLBM）  
> **SDAA 扫荡脚本**：`scripts/sdaa_sweep_sphere_re100_mem.py`  
> **工件**：`artifacts/sdaa_sphere_re100_mem.json`

---

## 1. 案例介绍（Purpose）

球体绕流 Re=100 是 LBM 验证的经典基准问题。该雷诺数低于涡脱落下临界（Re≈280），流动为定常轴对称尾流，阻力系数 Cd 有成熟的经验关联式可供对标。

**物理问题**：均匀来流绕球（直径 D），Re = U·D/ν = 100。  
**观测量**：阻力系数 Cd = Fx / (½ρU²·πR²)，采用 BFL 逐链动量账本（interpolated bouzidi 边界，实验室系），末 20% 采样窗均值。  
**参考值**：Schiller–Naumann 关联式 Cd_ref = 24/Re × (1 + 0.15·Re^0.687) = **1.0917310910948732**。  
**判定门**：两档网格 |err| ≤ 3% 且随分辨率严格单调下降。

本案例在 GPU（RTX 5090）上已完成验证（VERIFIED），本次任务为 SDAA 设备（/dev/tcaicard3，14.94 GiB）上的独立复现扫荡。

---

## 2. 方法（How）

### 2.1 GPU 基线验证（已完成）

- **入口**：`benchmarks/verified/sphere_re100/w8a/run.py`，使用 `tensorlbm.bfl_d3q19`（BFL 插值边界）+ `tensorlbm.solver3d`（MRT 低内存碰撞核 `collide_mrt3d_low_memory`）
- **两档网格**：
  - T1：D=40，域 280×280×320，28,722 BFL 链，12,000 步，τ=0.56
  - T2：D=60，域 420×420×480，64,794 BFL 链，24,000 步，τ=0.59
- **参考锁定**：预注册冻结 Cd_ref 后运行，满足 reference_locked_first 门

### 2.2 SDAA 扫荡（本次任务）

- **脚本**：`scripts/sdaa_sweep_sphere_re100_mem.py`（不改业务代码）
- **设备**：SDAA（/dev/tcaicard3），14.94 GiB 总容量
- **运行参数**：与 GPU 基线完全一致（T1: D=40/12k步/τ=0.56；T2: D=60/24k步/τ=0.59）
- **判据口径**：沿用原案例四门标准（library_entry / direct_observable / monotone_convergence / reference_locked_first）
- **Triton 处理**：脚本设置 `TRITON_INTERPRET=1` 绕过编译路径

---

## 3. 关键结果（Key Results）

### 3.1 GPU 基线验证结果（VERIFIED ✅）

| 档位 | D | 域 (nz×ny×nx) | BFL 链数 | 步数 | Cd（末 20% 窗） | err (%) | 稳态漂移 (%) | CV 闭合 (%) | 质量漂移 (ppm) |
|------|---|---------------|----------|------|-----------------|---------|-------------|------------|---------------|
| T1 | 40 | 280×280×320 | 28,722 | 12,000 | 1.1219804015117896 | +2.7708 | −0.1448 | 0.015 | −0.0797 |
| T2 | 60 | 420×420×480 | 64,794 | 24,000 | 1.119462001144486 | +2.5401 | −0.0420 | 0.008 | 0.0 |

**四门判定**：
| 门 | 要求 | 实际 | 结果 |
|---|---|---|---|
| 共性模块入口 | `src/tensorlbm` 库路径驱动 | D3Q19 + MRT 低内存核 + BFL 插值边界 | ✅ PASS |
| 直接观测量 ≤3% | \|err\| ≤ 3% | T1 +2.7708%，T2 +2.5401% | ✅ PASS |
| ≥2 档单调收敛 | \|err\| 随 D 严格单调降 | 2.7708% → 2.5401% | ✅ PASS |
| 参考先锁定后跑 | 预注册冻结 | Cd_ref 先锁，W8-A 后跑 | ✅ PASS |

**结论**：GPU 基线全部四门通过，|err| < 3% 且单调下降，VERIFIED。

### 3.2 SDAA 扫荡结果（FAIL ❌）

| 档位 | D | 步数 | τ | 成功 | 错误 |
|------|---|------|---|------|------|
| T1 | 40 | 12,000 | 0.56 | ❌ | OOM: tried 990 MiB, 1.38 GiB free (14.94 GiB total, 2.23 GiB allocated, 2.29 GiB reserved) |
| T2 | 60 | 24,000 | 0.59 | ❌ | OOM: tried 1.72 GiB, 287.97 MiB free (14.94 GiB total, 2.56 GiB allocated, 3.39 GiB reserved) |

**SDAA 四门判定**：
| 门 | 结果 | 说明 |
|---|---|---|
| library_entry | ✅ PASS | 使用 tensorlbm.bfl_d3q19, tensorlbm.solver3d 等库入口 |
| direct_observable | ❌ FAIL | 运行 OOM，无 Cd 数值结果 |
| monotone_convergence | ❌ FAIL | 运行 OOM，无法判定单调性 |
| reference_locked_first | ✅ PASS | Cd_ref = 1.0917310910948732（Schiller-Naumann）已锁定 |

**结论**：SDAA 设备因显存不足（OOM），两档运行均失败，无法获得数值结果。**honest-FAIL**。

### 3.3 误差判定

**关键量误差是否 < 3%？**

- **GPU 基线**：✅ 是。T1 err = +2.7708%，T2 err = +2.5401%，均 < 3% 且单调下降。
- **SDAA 扫荡**：❌ 无法判定。运行 OOM 失败，无 Cd 数值输出。

**无基准声明原因**：SDAA 设备显存管理策略与 CUDA 不同（Torch-SDAA 预留 2.29–3.39 GiB），导致可用显存不足以容纳 D=40/60 的稀疏核 LBM 域。GPU 基线在 RTX 5090（32 GiB）上成功运行，但 SDAA 卡（14.94 GiB 总容量，实际可用 ~1.38–1.72 GiB）受限于显存碎片化与预留策略。

---

## 4. 结果图（Figures）

GPU 基线验证生成 4 幅图件（位于 `benchmarks/verified/sphere_re100_mem/figures/`）：

1. **curve_cd_convergence.png**（73,351 B）：Cd(t) 收敛史曲线（T1 D=40 / T2 D=60），展示阻力系数从初始瞬态到稳态的收敛过程，红色虚线为 Schiller-Naumann 参考值。
2. **curve_error_tiers.png**（61,621 B）：档位误差曲线（T1 +2.7708% / T2 +2.5401%），展示网格细化后误差严格单调下降。
3. **field_midplane_speed.png**（291,385 B）：球中剖面速度幅值云图（y=0 对称面，Re=100），物理示意（基于势流模型）。
4. **field_wake_structure.png**（115,794 B）：尾流结构云图（y=0 中剖面，x-z 平面），展示 Re=100 轴对称尾流速度亏损，物理示意。

**SDAA 扫荡无图件**：因运行 OOM 失败，无 Cd(t) 历史数据，无法生成收敛曲线或场量云图。

**图件审计问题**：`gates_verified.json` 显示 figures 门 5/9 三元组复算不过——图件 source 字段引用的数据文件（t1_d40big_12k.json / t2_d60big_24k.json / result.json / run.py）位于兄弟目录 `../sphere_re100/w8a/`，而非 `sphere_re100_mem/` 本目录。图件本身存在且可复现，但溯源路径断裂。

---

## 5. 注意点（Caveats & Audit）

### 5.1 SDAA OOM 根因分析

- **显存碎片化**：SDAA 设备总容量 14.94 GiB，但 T1 运行时仅 1.38 GiB free（2.23 GiB allocated + 2.29 GiB reserved by Torch-SDAA）；T2 运行时仅 287.97 MiB free（2.56 GiB allocated + 3.39 GiB reserved）。
- **预留策略差异**：Torch-SDAA 预留 2.29–3.39 GiB（可能用于通信缓冲/内核缓存），显著压缩可用显存。
- **稀疏核未解锁**：GPU 基线使用稀疏核（sparse kernel）解锁显存，但 SDAA 侧可能未启用或实现不完整。

### 5.2 图件溯源断裂

- `figures.json` 中 4 幅图件的 source 字段引用了 `t1_d40big_12k.json` / `t2_d60big_24k.json` / `result.json` / `run.py`，但这些文件位于 `benchmarks/verified/sphere_re100/w8a/`，而非 `benchmarks/verified/sphere_re100_mem/`。
- **影响**：figures 门复算失败（5/9 三元组不过），但图件本身存在且可复现（由 `plot_figs.py` 从兄弟目录读取数据生成）。
- **建议**：修正 `figures.json` source 路径为相对路径 `../sphere_re100/w8a/...`，或在 `sphere_re100_mem/` 目录建立符号链接。

### 5.3 参考值锁定

- Schiller-Naumann 关联式 Cd_ref = 1.0917310910948732 为经验公式，非实验数据。
- 交叉验证：S-N(0.681) = 1.0685，Clift-Gauvin = 1.1092，T1 对 Clift-Gauvin 差约 +1.1pp。
- 参考值预注册锁定后未重开，满足 reference_locked_first 门。

### 5.4 网格档位限制

- 仅两档网格（D=40 / D=60），受 GPU 32G 显存约束。
- D=80 需 64G 卡或多卡分域，未在本次扫荡中尝试。

---

## 6. 误差判定总结（Verdict）

| 维度 | GPU 基线 | SDAA 扫荡 |
|------|---------|----------|
| 四门通过 | ✅ 4/4 PASS | ❌ 2/4 PASS（OOM） |
| \|err\| < 3% | ✅ T1 +2.77%, T2 +2.54% | ❌ 无结果 |
| 单调收敛 | ✅ 2.77% → 2.54% | ❌ 无结果 |
| 参考锁定 | ✅ Cd_ref 预注册 | ✅ Cd_ref 预注册 |
| 图件 | ✅ 4 幅（溯源路径断裂） | ❌ 无图件 |

**最终判定**：
- **GPU 基线**：VERIFIED ✅（四门全过，|err| < 3% 且单调下降）
- **SDAA 扫荡**：honest-FAIL ❌（OOM，无数值结果）

**改进建议**：
1. **SDAA 显存优化**：调查 Torch-SDAA 预留策略，尝试 `PYTORCH_SDAA_RESERVE_MEM=0` 或类似环境变量释放预留显存。
2. **稀疏核适配**：确认 SDAA 侧稀疏核（sparse kernel）实现状态，若未实现则优先移植。
3. **小域验证**：在 SDAA 上尝试 D=20（域 140×140×160）作为 smoke test，验证算法正确性后再扩域。
4. **图件溯源修复**：修正 `figures.json` source 路径，或在 `sphere_re100_mem/` 建立数据文件符号链接。

---

## 附录：工件清单

- `artifacts/sdaa_sphere_re100_mem.json`：SDAA 扫荡完整工件（含设备信息、运行参数、错误信息、四门判定）
- `benchmarks/verified/sphere_re100_mem/README.md`：案例说明（5,691 B）
- `benchmarks/verified/sphere_re100_mem/gates.json`：四门判定（GPU 基线）
- `benchmarks/verified/sphere_re100_mem/gates_verified.json`：四门复算审计（含 figures 门问题）
- `benchmarks/verified/sphere_re100_mem/validation_report.json`：验证报告（GPU 基线）
- `benchmarks/verified/sphere_re100_mem/figures/`：4 幅图件（PNG）
- `scripts/sdaa_sweep_sphere_re100_mem.py`：SDAA 扫荡脚本

---

*报告生成时间：2026-01-23*  
*数据来源：工件程序化提取，零手抄*  
*纪律：honest-FAIL 如实记录，不改业务代码*
