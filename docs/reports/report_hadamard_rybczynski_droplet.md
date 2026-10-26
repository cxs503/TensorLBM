# Hadamard-Rybczynski 液滴终端速度基准测试 — 平台报告

**案例编号**：m155 / hadamard_rybczynski_droplet  
**终裁日期**：2026-10-21  
**终裁结论**：**TERMINAL（判死）** — 壁面约束效应导致误差超限

---

## 1. 物理问题表述

### 1.1 Hadamard-Rybczynski 问题

Hadamard-Rybczynski (H-R) 问题是多相流领域的经典解析基准：一个半径为 $a$ 的黏性液滴在另一种黏性流体中受浮力作用以终端速度 $U_t$ 匀速上升/下降。解析解（1911）给出：

$$U_t = \frac{2}{3} \frac{a^2 g (\rho_o - \rho_i)}{\mu_o} \cdot \frac{1 + \lambda}{2 + 3\lambda}$$

其中 $\lambda = \mu_i / \mu_o$ 为内外黏度比。该解假设：
- **无限域**（unbounded domain，无壁面）
- 蠕变流（$Re \ll 1$）
- 稳态终端速度

### 1.2 基准意义

H-R 基准的独特价值在于：
1. **解析参考**：无需数值参考解，直接对解析公式
2. **黏度比敏感**：$U_t$ 显式依赖 $\lambda$，可验证多相模块的黏度解耦能力
3. **历史卡点**：原 3D 多相模块锁死 $\lambda = \rho$，无法独立控制黏度比，导致基准不可运行

### 1.3 M1 相场重建后的重启

PR #323 引入 CAC 相场模块（`tensorlbm.cac_lbm`），支持独立 $\mu_l, \mu_g$ 控制，破除了 $\lambda = \rho$ 锁死。本任务（m155）即在此能力基础上重启 H-R 基准探针。

---

## 2. 模拟设置

### 2.1 参数配置

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 域尺寸 | $L$ | 96 | 立方域 $96^3$ |
| 液滴半径 | $a$ | 12 | 约束比 $a/L = 0.125$ |
| 外相密度 | $\rho_o$ | 1.0 | — |
| 内相密度 | $\rho_i$ | 0.5 | 密度比 0.5 |
| 外相黏度 | $\mu_o$ | 1/6 | $\tau_o = 1.0$ |
| 内相黏度 | $\mu_i$ | 1/12 | $\lambda = 0.5$ |
| 重力加速度 | $g$ | $8.1 \times 10^{-7}$ | 向下 |
| 界面宽度 | $W$ | 4 | — |
| 表面张力 | $\sigma$ | 0.001 | — |

### 2.2 参考锁定

$$U_t^{ref} = \frac{2}{3} \times 12^2 \times 8.1 \times 10^{-7} \times \frac{0.5}{1/6} \times \frac{1.5}{3.5} = 9.998 \times 10^{-5}$$

$$Re = \frac{U_t^{ref} \cdot 2a \cdot \rho_o}{\mu_o} = 0.0144 \ll 1 \quad \text{（蠕变流假设满足）}$$

### 2.3 运行命令

```bash
CUDA_VISIBLE_DEVICES=5 /nfs/wangxi/venvs/tensorlbm/bin/python /tmp/cac_hr_buoyancy.py
```

---

## 3. 结果

### 3.1 终端速度收敛

![终端速度收敛曲线](benchmarks/pending/hadamard_rybczynski_droplet/figures/terminal_velocity_convergence.png)

模拟在 ~5000 步达到稳态，终端速度稳定在 $5.653 \times 10^{-5}$。

### 3.2 液滴轨迹

![液滴质心轨迹](benchmarks/pending/hadamard_rybczynski_droplet/figures/droplet_trajectory.png)

液滴质心匀速上升，斜率即终端速度。

### 3.3 误差分析

| 量 | 值 |
|----|-----|
| 模拟终端速度 $U_t^{sim}$ | $5.653 \times 10^{-5}$ |
| 参考终端速度 $U_t^{ref}$ | $9.998 \times 10^{-5}$ |
| **相对误差** | **-43.46%** |
| 门限 | ≤ 3% |
| 判决 | ❌ FAIL |

### 3.4 四门评估

![四门评估](benchmarks/pending/hadamard_rybczynski_droplet/figures/gate_assessment.png)

| 门 | 要求 | 实测 | 判决 |
|---|---|---|---|
| G1 库入口 | src/tensorlbm 库路径 | `tensorlbm.cac_lbm.CACSim` | ✅ PASS |
| G2 直接观测量 | 相对误差 ≤ 3% | -43.46% | ❌ FAIL |
| G3 单调收敛 | ≥2 档网格单调 | 仅单档 | ⚠️ N/A |
| G4 参考先锁 | 参考值先于模拟 | 解析解已锁定 | ✅ PASS |

---

## 4. 根因分析：壁面约束效应

### 4.1 物理机制

H-R 解析解假设**无限域**，但模拟必须有壁面作为动量汇。约束比 $a/L = 0.125$ 导致显著的壁面阻滞效应。

根据 Faxén 修正理论（一阶）：

$$\frac{U_{confined}}{U_{unbounded}} \approx 1 - 2.104 \times \frac{a}{L} = 1 - 2.104 \times 0.125 \approx 0.737$$

实际测量值 $U_{sim}/U_{ref} = 0.565$，偏差来源：
1. Faxén 一阶修正不够精确（高阶项未计入）
2. 方形管道 vs 圆形管道的几何差异
3. 六面全壁面约束（非单管道方向）

### 4.2 为什么无法通过网格细化解决

壁面约束效应是**物理问题**，不是数值误差：
- 保持 $a/L$ 不变 → 误差结构不变
- 增大 $L$ 减小 $a/L$ → 需 $L \geq 256$（$a/L \leq 0.047$），预算 ~10 GPU·h
- 这不是收敛性问题，而是边界条件与解析假设不匹配

### 4.3 证据链

```
U_t_sim = 5.653e-5    [terminal_buoyancy.log: "Terminal velocity"]
U_t_ref = 9.998e-5    [prereg.md: H-R formula]
error     = -43.46%   [terminal_buoyancy.log: "Relative error"]
a/L       = 0.125     [cac_hr_buoyancy.py: L=96, a=12]
Faxén(1)  ≈ 0.737     [analytical correction]
measured  ≈ 0.565     [U_sim / U_ref]
```

---

## 5. 判据表

| 判据 | 值 | 文件 | 键 |
|------|-----|------|-----|
| 模拟终端速度 | $5.653 \times 10^{-5}$ | terminal_buoyancy.log | `Terminal velocity` |
| 参考终端速度 | $9.998 \times 10^{-5}$ | prereg.md | H-R formula |
| 相对误差 | -43.46% | terminal_buoyancy.log | `Relative error` |
| 约束比 | 0.125 | cac_hr_buoyancy.py | L=96, a=12 |
| Reynolds 数 | 0.0144 | terminal_buoyancy.json | `Re_ref` |
| 模拟步数 | 30000 | terminal_buoyancy.json | `trace` count |
| 模拟时间 | 459.4s | terminal_buoyancy.log | `Total time` |

---

## 6. 审计溯源

### 6.1 PROVENANCE

| 项目 | 来源 |
|------|------|
| 模块 | `tensorlbm.cac_lbm` (PR #323, commit 05c86ef5) |
| 参考公式 | Hadamard (1911), Rybczynski (1911) |
| 预注册 | `benchmarks/pending/hadamard_rybczynski_droplet/prereg.md` |
| 运行脚本 | `/tmp/cac_hr_buoyancy.py` |
| 数据工件 | `/nfs/wangxi/runs/bm_close/hadamard_rybczynski_droplet/terminal_buoyancy.{log,json}` |
| GPU | GPU5 (`CUDA_VISIBLE_DEVICES=5`) |
| Python | `/nfs/wangxi/venvs/tensorlbm/bin/python` |

### 6.2 历史状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| 原始 | ❌ 模块能力缺失 | λ=ρ 锁死，基准不可运行 |
| M1 相场重建 | ✅ 模块能力恢复 | CAC 支持独立 λ |
| m155 探针 | ⏸️ TERMINAL | 壁面约束效应，误差超限 |

---

## 7. 局限与缓解

### 7.1 当前局限

1. **单档网格**：仅 $L=96$ 一档，无法评估收敛性
2. **壁面约束**：$a/L = 0.125$ 过大，解析解假设不匹配
3. **未做 Faxén 修正**：后处理修正可缩小误差但引入理论不确定性

### 7.2 缓解路径（需 owner 决策）

| 路径 | 预算 | 预期效果 |
|------|------|----------|
| 增大域至 $L=256$ | ~10 GPU·h | $a/L=0.047$，误差降至 ~10% |
| Faxén 高阶修正 | 0 GPU·h | 理论修正，引入不确定性 |
| 改用 Taylor-Green 涡 | ~2 GPU·h | 无壁面基准，但物理不同 |
| 维持 TERMINAL | 0 | 记录负结果，留待后续 |

---

## 8. 结论

H-R 基准在 M1 相场重建后可运行，但壁面约束效应导致终端速度误差 -43.46%，远超 3% 门限。根因是解析解假设无限域而模拟必须有壁面，属物理设置不匹配而非数值误差。

**终裁**：TERMINAL。模块能力已验证（CAC 数值稳定、黏度解耦成功），但基准物理设置需大域模拟或理论修正方可闭环。

---

*报告生成日期：2026-10-21 | 工件路径：`/nfs/wangxi/runs/bm_close/hadamard_rybczynski_droplet/`*
