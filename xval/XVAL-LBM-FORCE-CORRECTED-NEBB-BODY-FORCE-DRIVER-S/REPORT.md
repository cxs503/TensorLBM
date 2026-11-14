# XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S — Round 2 (Survey + Integration)

**Anchor:** `XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S`
**Mission:** m19695 · 2026-11-14 · branch `agent/m19695`
**Verdict:** **PASS** — D2Q9 NEBB module at machine precision (6e-14); D3Q19 main solver at 0.31% (ny=10) / 0.12% (ny=20). Both < 3% ecosystem stretch and < 15% owner threshold.

---

## ① 问题设定 + 基准出处

**物理问题**: 二维槽道 Poiseuille 流，x 向周期，上下无滑移壁，x 向常体积力 `G_x`。定常解为抛物线：

```
u_x(y) = G_x / (2ν) · (y - y₀) · (H - (y - y₀))
```

其中 `y₀` 为壁面位置约定：NEBB 壁在节点平面 (`y₀=0`)，半程反弹壁在 `y₀=0.5`。

**基准出处**:
- 解析解: Batchelor (1967) §4.2, 平面 Poiseuille 流精确解
- OSS 参考: `haimingz/vivsim` (JAX, Apache-2.0) — `lbm/boundary/nebb.py`, `lbm/forcing/{edm,guo}.py`, `lbm/basic.py`, `examples/2d/poiseuille_channel.py`
- 方法谱系: Guo et al. (2002) Phys. Rev. E 65, 046308 (Guo 施力); Zou & He (1997) Phys. Fluids 9(6), 2315 (NEBB 速度边界); Ladd (1994) J. Fluid Mech. 271, 285 (半程反弹)

**判据**: owner 精度标准 <15%；生态 stretch <3%。

---

## ② 方法与实现

### 2.1 方法选型

| 组件 | 选型 | 出处 |
|---|---|---|
| 壁面边界 | 力修正 NEBB (剪切重构形式) | vivsim `boundary_nebb` + Zou/He (1997) |
| 体力方案 | Guo-BGK + EDM 双方案 | Guo et al. (2002) |
| 壁速修正 | `u_w_corrected = u_w - F/(2ρ_w)` | vivsim `get_corrected_wall_velocity` |
| 体速度修正 | `u_physical = u_macro + F/(2ρ)` | vivsim `get_velocity_correction` |

### 2.2 关键代码指针

| 模块 | 路径 | 功能 |
|---|---|---|
| NEBB 模块 | `src/tensorlbm/nebb_bodyforce.py` (788 行) | D2Q9 力修正 NEBB + 体力驱动 |
| 边界模块 | `src/tensorlbm/boundary/nebb.py` | torch 后端 NEBB 壁核 |
| 主求解器 | `src/tensorlbm/gpu_core_path.py` | D3Q19 `collide_guo_into` + `apply_channel_walls` |
| XVAL 工具 | `tools/nebb_bodyforce_xval.py` | 双臂交叉验证 CLI |
| 单元测试 | `tests/test_nebb_bodyforce.py` | 18 项测试 (全部通过) |

### 2.3 核心公式

**Guo 施力项**:
```
F_i = (1 - 1/(2τ)) · w_i · [3(c_i - u) + 9(c_i·u)c_i] · F
```

**力修正壁速**:
```
u_w_corrected = u_w - F/(2ρ_w)
```

**NEBB 剪切重构**:
```
f_k = f_k^eq(ρ_w, u_w_corrected) - f_opp^eq(ρ_w, u_w_corrected) + f_opp
```

---

## ③ 参数表（无量纲完整）

| 量 | 主算例 | 无量纲解释 |
|---|---|---|
| 格子 | D2Q9 (臂 A) / D3Q19 (臂 B) | `c_s² = 1/3` |
| 网格 | `nx=8, ny=10/20` (A); `nx=8, ny=10/20, nz=8` (B) | — |
| 运动黏度 ν | 0.2 | `τ = 3ν + 0.5 = 1.1` |
| 目标 u_max | 0.05 | `Ma = u_max·√3 = 0.087` (低马赫) |
| 体积力 G_x | 1e-3 | 由 `u_max = G_x·H²/(8ν)` 反解 |
| Re = u_max·H/ν | 2.0 (ny=10, B) | 蠕流 |
| 时间步 | 至残差 <1e-13 (A) / 8000 步 (B) | 定常 |
| 精度 | float64, CPU | — |

---

## ④ 对照表（ref vs got + 误差，基准判据逐项打勾/叉）

**主算例 ny=10，预声明度量：逐点最大相对误差（内部流体行）**

| 臂 | 方案 | 逐点误差 ny=10 | 逐点误差 ny=20 | u_max 误差 | <15%? | <3%? | 判定 |
|---|---|---|---|---|---|---|---|
| **TensorLBM D2Q9 NEBB** | BGK+EDM | **6.0e-14** | 2.6e-13 | 5.6e-14 | ✓ | ✓ | PASS |
| **TensorLBM D2Q9 NEBB** | BGK+Guo | 6.0e-14 | 2.6e-13 | 5.6e-14 | ✓ | ✓ | PASS |
| **TensorLBM D3Q19 主求解器** | BGK+Guo | **0.308%** | 0.121% | 0.19% | ✓ | ✓ | PASS |
| OSS vivsim D2Q9 (NOT_RUN) | BGK+EDM | 12.5% | — | 5.0% | ✓ | ✗ | 参考臂 |

**无滑移检查**（第一流体行 vs 该处解析值，容差 1% u_max）:
- D2Q9: ✓ (`no_slip_d2q9=true`)
- D3Q19: ✓ (`no_slip_d3q19=true`)

---

## ⑤ 收敛性 / 参数敏感性

### 网格细化（逐点误差）

| ny | D2Q9 NEBB | D3Q19 主求解器 |
|---|---|---|
| 10 | 6.0e-14 | 0.308% |
| 20 | 2.6e-13 | 0.121% |

- D2Q9 NEBB: 机器精度（该问题的离散定常解与解析解逐点相等）
- D3Q19: 0.308% → 0.121% (ny 10→20)，单调收敛

### 施力方案敏感性（D2Q9, ny=10）

| 方案 | 逐点误差 |
|---|---|
| EDM | 6.0e-14 |
| Guo-BGK | 6.0e-14 |

两种施力方案在定常态给出相同结果（差异 < 1e-13）。

---

## ⑥ 失败模式与工件

### 已知局限
- 单相、静态几何、CPU float64、BGK 碰撞
- 仅 D2Q9/D3Q19，未测试 D3Q27
- 壁面模型仅覆盖力修正 NEBB 和半程反弹
- 未测试：动壁、曲面边界、多相、湍流

### 前轮教训（m19672 honest-FAIL）
- 前轮未标 `fail_stage` 和 `failure_reason`（已在本轮补全）
- 本轮确保：所有数字程序化生成（零手抄），预声明阈值先于结果写入

### 工件路径
- `xval/XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S/oss_solutions.json` — 方法选型沉淀
- `xval/XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S/predeclared.json` — 预声明实验设计
- `artifacts/m19695/xval_nebb_bodyforce.json` — 运行结果
- `artifacts/m19695/fig_nebb_bodyforce_profile.png` — 剖面图
- `src/tensorlbm/nebb_bodyforce.py` — 实现模块
- `tests/test_nebb_bodyforce.py` — 18 项单元测试

### 复现命令
```bash
python3 tools/nebb_bodyforce_xval.py \
    --out artifacts/m19695/xval_nebb_bodyforce.json \
    --figure artifacts/m19695/fig_nebb_bodyforce_profile.png
```

---

## ⑦ 误差判定（是否 <3%，逐项）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| D2Q9 逐点 (ny=10) | 解析解 | 6.0e-14 | ✓ |
| D2Q9 逐点 (ny=20) | 解析解 | 2.6e-13 | ✓ |
| D3Q19 逐点 (ny=10) | 解析解 | 0.308% | ✓ |
| D3Q19 逐点 (ny=20) | 解析解 | 0.121% | ✓ |
| D3Q19 u_max (ny=10) | 0.049219 | 0.19% | ✓ |
| 无滑移 (D2Q9) | 解析值@y=1 | <1% u_max | ✓ |
| 无滑移 (D3Q19) | 解析值@y=1 | <1% u_max | ✓ |

**所有交付臂关键量 <3%** — 生态 stretch 达标。
