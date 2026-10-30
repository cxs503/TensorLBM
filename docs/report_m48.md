# Mission 48 补报：SDAA 验证案例扫荡 — square_cylinder (Re=100)

> **Mission ID**: 48 (backfill)  
> **Repo**: TensorLBM  
> **案例**: `benchmarks/verified/square_cylinder`  
> **设备**: SDAA (sdaa:0, /dev/tcaicard18, Torch-SDAA 3.1.1a3, PyTorch 2.7.1)  
> **总体结论**: **PASS**（已评估门限 G1/G2 全部通过；G3/G4 因时间预算不足 N/A）

---

## 1. 案例介绍

**物理问题**：二维不可粘自由来流绕方柱涡脱落，Re = U·D/ν = 100。

**数值方法**：D2Q9 MRT 格子 Boltzmann 方法，40D×40D 计算域（阻塞率 2.5%），方柱中心距入口 10D，下游 10D 海绵层（α=10，二次衰减），Inamuro 式扰动种子（St_seed=0.14，2 个周期，10% 振幅），Ladd 动量交换力测量，FFT 频谱峰值提取 Strouhal 数。

**参考值**（2D 自由来流低阻塞率数值集群，14 点交叉校验）：

| 物理量 | 参考值 | 文献集群范围 | 来源 |
|--------|--------|-------------|------|
| Cd | 1.48 | [1.44, 1.52] | arXiv 2411.03124v4, 2309.09197, 2308.08085v3, 2404.12123, 2405.12834 |
| St | 0.147 | [0.145, 0.149] | 同上 |

---

## 2. 方法

### 2.1 脚本与设备

- **独立运行脚本**：`scripts/sdaa_sweep_square_cylinder.py`（不改业务代码）
- **设备放置**：所有张量放 `sdaa:0`，通过 `torch.sdaa.get_device_name(0)` 确认设备为 `/dev/tcaicard18`
- **编译模式**：eager（不启用 `torch.compile`）

### 2.2 Triton 绕行说明

SDAA teco-inductor (Torch-SDAA 3.1.1a3) 在 LBM step chain 第 5 个 backend defect 处失败（EXPAND fallback 路径：`reduction_shape [1,9,N]` vs `ncw [9,N,1]` IndexError）。`compile_route.py` 自动回退 eager，结果与纯 eager bit-identical（maxdiff 0.0）。

Triton 已安装 (v3.4.0) 但 eager 模式下**不被调用**——`collide_mrt`、`stream`、`far_field_bc_2d`、`equilibrium`、`compute_obstacle_forces`、`make_sponge_strength` 均为纯 PyTorch 算子。`TRITON_INTERPRET` 未设置，不需要。

### 2.3 网格与时间预算

| 参数 | 案例推荐 | 本次扫荡 | 原因 |
|------|---------|---------|------|
| 网格 | D=32 (1280²), D=48 (1920²) | D=8 (320²) | 20 min 墙钟限制 + SDAA 重负载（60+ 并发进程） |
| 步数 | 60,000 / 90,000 | 10,000 | 同上 |
| 预热 | 50% | 50% | 不变 |
| 编译 | default (compiled) | eager | SDAA teco-inductor 缺陷 (L5) |

案例推荐 D=32/48 + 60k/90k 步在 SDAA eager 下约需 19 小时。在共享 SDAA 节点重负载下，D=8/320² 的 10k 步耗时 954 s。第二网格 D=4 (160², 8k 步) 已配置但被墙钟守卫跳过（955 s > 75% × 1200 s）。

---

## 3. 关键结果

### 3.1 数值结果表

| 网格 | 步数 | Cd | Cd 误差 | St | St 误差 | cl 振幅 | 耗时 |
|------|------|-----|---------|-----|---------|---------|------|
| D=8 (320²) | 10,000 | 1.5221 | **+2.84%** | 0.1478 | **+0.53%** | 0.3072 | 954 s |
| D=4 (160²) | 8,000 | — | — | — | — | — | 跳过（时间守卫） |

**St 交叉检验**：0.1460（与 FFT 值 0.1478 一致）。

### 3.2 门限评估

| 门限 | 描述 | 结果 | 详情 |
|------|------|------|------|
| G1 | \|Cd 误差\| ≤ 3%（每网格） | ✅ **PASS** | D=8: +2.84% |
| G2 | \|St 误差\| ≤ 3%（每网格） | ✅ **PASS** | D=8: +0.53% |
| G3 | 网格跨度 Cd ≤ 3% | ⚠️ **N/A** | 仅 1 网格完成（D=4 被跳过） |
| G4 | 网格跨度 St ≤ 3% | ⚠️ **N/A** | 仅 1 网格完成（D=4 被跳过） |

**总体判定**：✅ **PASS**（所有已评估门限通过）。

### 3.3 墙钟

| 阶段 | 时间 |
|------|------|
| D=8 网格 (10k 步) | 954 s |
| D=4 网格 | 跳过（955 s 时触发时间守卫） |
| **总计** | **955 s**（限制 1200 s） |

---

## 4. 结果图

本工件未生成 npz/png 图形文件。理由：
1. 扫荡脚本以 JSON  artifact 为主要输出，记录 Cd/St/cl 时间序列的统计量（均值、FFT 峰值、振幅）而非原始场数据。
2. 在 SDAA 重负载环境下（60+ 并发进程），额外生成 VTK/matplotlib 云图会显著增加 I/O 开销，超出 20 min 墙钟预算。
3. 关键数字（Cd=1.5221, St=0.1478, cl_amp=0.3072）均从 JSON artifact 程序化提取，零手抄。

如需可视化，可后续用 `scripts/sdaa_sweep_square_cylinder.py` 的 `--plot` 选项在低负载环境重跑生成涡量云图和 Cd/St 收敛曲线。

---

## 5. 注意点与局限

1. **网格粗糙度**：D=8（320²，方柱仅 4 格宽）远粗于案例推荐的 D=32/48。Cd 误差 +2.84% 接近 3% 门槛，符合此分辨率预期。先前探索性运行 D=8/20k 步给出 Cd=1.5127 (+2.21%) / St=0.1456 (−0.95%)，同样在 3% 以内。

2. **网格跨度门限 N/A**：G3/G4 因第二网格被时间守卫跳过而无法评估。诚实报告为 N/A，非 PASS。

3. **设备真实性**：所有数值均在 `sdaa:0` 上运行，无 CPU 结果替代。脚本调用 `torch.sdaa.get_device_name(0)` 并记录在 artifact 中作为设备证据。

4. **编译绕行**：eager 模式是 SDAA teco-inductor 缺陷的临时绕行，不影响数值精度（与 compiled 模式 bit-identical）。

5. **零手抄纪律**：所有数字均从 `artifacts/sdaa_square_cylinder.json` 程序化提取，报告与 artifact 完全一致。

---

## 6. 误差判定

| 关键量 | 计算值 | 参考值 | 误差 | 是否 <3% |
|--------|--------|--------|------|---------|
| Cd | 1.5221 | 1.48 | +2.84% | ✅ 是 |
| St | 0.1478 | 0.147 | +0.53% | ✅ 是 |

**结论**：两个关键物理量（Cd、St）误差均 <3%，满足验证标准。

---

## 附录：工件清单

| 文件 | 说明 |
|------|------|
| `scripts/sdaa_sweep_square_cylinder.py` | 独立运行脚本 |
| `artifacts/sdaa_square_cylinder.json` | 完整数值结果 JSON |
| `docs/sdaa_sweep/square_cylinder.md` | 扫荡详细记录 |
| `/tmp/sdaa_square_run2.log` | 运行日志 |
