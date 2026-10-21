# collision_kernels_3d_tg 终裁报告

**案例状态**: ⏸️ TERMINAL  
**终裁日期**: 2026-10-20  
**终裁结论**: KBC 修复核 PASS，五核 D3Q27 单调性门 FAIL，整体案例 TERMINAL

---

## 1. 物理问题

3D Taylor-Green 涡衰减基准测试，评估六个高性能碰撞核的数值精度与收敛性：
- **cumulant_d3q27** (D3Q27)
- **cascaded_d3q27** (D3Q27)
- **mrt27** (D3Q27)
- **trt27** (D3Q27)
- **rlbm27** (D3Q27)
- **kbc_d3q19** (D3Q19)

**测试协议**:
- 周期域 N³，k=2π/N
- Re=24，U0=0.05
- 网格阶梯 N=64/96/128（对应 τ=0.9/1.1/1.3）
- 观测动能衰减率 γ_E 与解析值 6νk² 的偏差

**判定四门**:
1. **共性模块入口**: 库 only，零手写核
2. **直接观测量 ≤3%**: 所有网格 |err_E| ≤ 3%
3. **单调收敛**: ≥2 档网格严格单调递减
4. **参考先锁定后跑**: 预注册 NOTES_sha256 链先于运行

---

## 2. 结果汇总

### 2.1 五核 D3Q27（cumulant/cascaded/mrt/trt/rlbm）

| 核 | N=64 err% | N=96 err% | N=128 err% | 单调性 | 容差 | R² | 判定 |
|---|---|---|---|---|---|---|---|
| cumulant_d3q27 | +0.1825 | +0.3738 | +0.4122 | ✗ | ✓ | ✓ | FAIL |
| cascaded_d3q27 | +0.1826 | +0.3738 | +0.4122 | ✗ | ✓ | ✓ | FAIL |
| mrt27 | +0.2146 | +0.4065 | +0.4232 | ✗ | ✓ | ✓ | FAIL |
| trt27 | +0.3777 | +0.4328 | +0.4713 | ✗ | ✓ | ✓ | FAIL |
| rlbm27 | +0.3118 | +0.3688 | +0.4030 | ✗ | ✓ | ✓ | FAIL |
| **bgk27 对照** | +0.3137 | +0.3347 | +0.3263 | ✗ | ✓ | ✓ | (对照) |

**关键发现**:
- ✅ 容差门全过：最大偏差 0.4713%（trt27 N=128），远小于 3% 阈值
- ✅ 指数性全过：R² ≥ 0.999995
- ❌ 单调性全败：五核 + bgk27 对照均不单调

**根因分析**:
单调性失败是 **D3Q27 + 固定 Re τ 阶梯的结构性问题**，非核特异：
- bgk27 对照同样失败，证明不是碰撞核缺陷
- 格子离散项 ∝ k²·f(τ)，f(τ) 随 τ 增长快于 k² 随 N 缩小
- 固定 Re 协议下，τ 阶梯（0.9→1.1→1.3）导致离散误差非单调

### 2.2 KBC 修复核（D3Q19）

| N | τ | err_E% | R² | 判定 |
|---|---|---|---|---|
| 64 | 0.9 | -0.6748 | 0.999996 | ✓ |
| 96 | 1.1 | -0.3576 | 0.999994 | ✓ |
| 128 | 1.3 | -0.1608 | 0.999995 | ✓ |

**关键发现**:
- ✅ 容差门全过：最大偏差 0.6748%，远小于 3% 阈值
- ✅ 指数性全过：R² ≥ 0.999994
- ✅ 单调性全过：-0.6748% → -0.3576% → -0.1608% 严格单调递减
- ✅ **四门全过，PASS**

**修复背景**:
- 旧核 KBC 粘度不由 τ 设定（ν_eff ≈ 1/6 恒定），实质缺陷
- 库 PR #312（2026-09-22）修复：f* = feq + (1−1/τ)s + βh
- 修复后剪切粘度由 τ 精确控制，熵条件仅固定非流体动力学 ghost 弛豫

---

## 3. 终裁结论

### 3.1 案例整体判定：TERMINAL

**理由**:
- 案例结构要求 **所有核四门全过** 才能晋级
- 五核 D3Q27 单调性门未过（结构性问题）
- 尽管 KBC 修复核 PASS，但整体案例未达四门齐全

### 3.2 各核判定

| 核 | 格子 | 四门 | 判定 |
|---|---|---|---|
| cumulant_d3q27 | D3Q27 | G3 ✗ | TERMINAL FAIL |
| cascaded_d3q27 | D3Q27 | G3 ✗ | TERMINAL FAIL |
| mrt27 | D3Q27 | G3 ✗ | TERMINAL FAIL |
| trt27 | D3Q27 | G3 ✗ | TERMINAL FAIL |
| rlbm27 | D3Q27 | G3 ✗ | TERMINAL FAIL |
| **kbc_d3q19 (patched)** | **D3Q19** | **G1-G4 ✓** | **PASS** |

### 3.3 晋级路径

**KBC 修复核**:
- 证据充分，四门全过
- 可单独归档为 `verified/collision_kbc_d3q19_tg/`
- 本任务不做拆分，保持案例完整性

**五核 D3Q27**:
- 单调性失败是协议结构性问题，非核缺陷
- 两条可能路径（需 owner 决策）：
  1. 接受「D3Q27+固定 Re τ 阶梯下单调条款不可判」的披露，以容差+指数性放行
  2. 改锁固定 τ、变 Re 协议重预注册重跑（成本：六核×3 档，GPU 分钟级）

---

## 4. 图件可视化

### 4.1 碰撞核弛豫谱 γ_E(τ)

![碰撞核弛豫谱](benchmarks/pending/collision_kernels_3d_tg/figures/curve_gamma_tau_spectrum.png)

**图注**：碰撞核弛豫谱 γ_E(τ)：KBC D3Q19（PASS）与五 D3Q27 核（FAIL）在 τ=0.9/1.1/1.3 三档的模拟衰减率对比，虚线为解析值 6νk²。KBC 修复核的 γ_sim 与解析值高度吻合，五 D3Q27 核系统性偏高但偏差均 <0.5%。

### 4.2 误差收敛曲线

![误差收敛](benchmarks/pending/collision_kernels_3d_tg/figures/curve_error_convergence.png)

**图注**：碰撞核误差收敛曲线 |err_E|(%) vs 网格 N：KBC 单调下降（0.67%→0.36%→0.16%），五 D3Q27 核非单调上升。红色虚线为 3% 容差阈值，所有核均满足容差要求，但仅 KBC 满足单调性要求。

### 4.3 动能衰减史

![动能衰减史](benchmarks/pending/collision_kernels_3d_tg/figures/curve_energy_decay_history.png)

**图注**：动能衰减史 E(t)/E₀ 与瞬时衰减率 γ_E(t)：N=96 档位 KBC（PASS）与三 D3Q27 核（FAIL）对比，虚线为 LSQ 拟合值。左图半对数坐标显示指数衰减特征，右图瞬时衰减率在 t>200 后趋于稳定，验证 LSQ 拟合有效性。

---

## 5. 判据表

### 4.1 四门检查

| 门 | 要求 | 五核 D3Q27 | KBC patched |
|---|---|---|---|
| G1 共性模块入口 | 库 only | ✓ | ✓ |
| G2 容差 ≤3% | 所有网格 \|err\| ≤ 3% | ✓ (max 0.47%) | ✓ (max 0.67%) |
| G3 单调收敛 | ≥2 档严格单调 | ✗ (结构性) | ✓ |
| G4 参考锁定 | 预注册先于运行 | ✓ | ✓ |

### 4.2 机器门禁 gates.json

位置: `benchmarks/pending/collision_kernels_3d_tg/gates.json`

```json
{
  "case": "collision_kernels_3d_tg",
  "verdict": "terminal",
  "reason": "五核单调收敛门未过：D3Q27 + 固定 Re τ 阶梯下结构性不可满足",
  "gates": [
    {"gate": "library_entry", "pass": true, ...},
    {"gate": "direct_observable", "pass": true, ...},
    {"gate": "monotone_convergence", "pass": false, ...},
    {"gate": "reference_locked_first", "pass": true, ...}
  ]
}
```

复算验证: `gates_check.py` 通过（rc=0）

---

## 5. 审计溯源

### 5.1 工件清单

| 工件 | 路径 | 说明 |
|---|---|---|
| 原五核结果 | `benchmarks/pending/collision_kernels_3d_tg/result.json` | 六核+对照原始数据 |
| KBC 修复核结果 | `/nfs/wangxi/runs/bm_widen_w7_20260922/kbc_fix/tg/result.json` | PR #312 修复后复跑 |
| 机器门禁 | `benchmarks/pending/collision_kernels_3d_tg/gates.json` | 终裁判据三元组 |
| 复算验证 | `benchmarks/pending/collision_kernels_3d_tg/gates_verified.json` | gates_check.py 输出 |

### 5.2 PROVENANCE 链

所有判决数字走值-文件-键三元组直读工件，零手抄：
- 五核 err_e_pct: `result.json` → `kernels.<name>.cases[2].err_e_pct`
- KBC err_e_pct: `kbc_fix/tg/result.json` → `cases[i].err_e_pct`
- 单调性标志: `result.json` → `kernels.<name>.convergence.converged_monotone`

### 5.3 版本钉定

- 原案例: worktree bm_w5 (commit cf5709db3c)
- KBC 修复: PR #312 (2026-09-22 合入)
- 终裁日期: 2026-10-20

---

## 6. 局限与缓解

### 6.1 局限

1. **案例整体 TERMINAL**: 五核单调性门未过，无法整体晋级
2. **协议结构性问题**: D3Q27 + 固定 Re τ 阶梯下单调条款不可满足
3. **KBC 未拆分归档**: 保持案例完整性，未单独提取 KBC 到 verified/

### 6.2 缓解

1. **证据完整**: KBC 修复核四门全过，证据充分
2. **根因清晰**: 单调性失败为协议结构性问题，非核缺陷
3. **路径明确**: 提供两条晋级路径供 owner 决策

### 6.3 建议

1. **短期**: 接受终裁，保持案例在 pending/
2. **中期**: owner 决策五核晋级路径（条款修订 or 协议重跑）
3. **长期**: 考虑拆分 KBC 为独立案例 `verified/collision_kbc_d3q19_tg/`

---

## 7. 结论

collision_kernels_3d_tg 案例终裁 TERMINAL：
- **KBC 修复核 PASS**：四门全过，证据充分
- **五核 D3Q27 FAIL**：单调性门结构性失败
- **整体案例 TERMINAL**：未达四门齐全

终裁符合 benchmark 闭环标准：失败也是合法产出，根因证据链完整，判决数字零手抄。
