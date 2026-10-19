# couette_3d 闭环报告：稳态口径裁定与 fp64 晋级

**Mission**: m62 · **案例**: couette_3d · **日期**: 2026-10-19
**终态**: ✅ 晋级 verified/couette_3d（fp64 四档）

## 1. 物理问题表述

三维 Couette 流：两平行平板间剪切流，下壁静止、上壁以 U₀=0.05（+x 方向）拖动，
x/z 双向周期。稳态解析解为线性速度剖面：

$$u(y) = U_0 \cdot \frac{y - 0.5}{H_{\text{eff}}}$$

其中半程反弹约定壁位于 y=0.5 与 y=ny−1.5，有效间隙 H_eff = ny−2 = H。
D3Q19 BGK 碰撞 τ=0.8（ν=0.1），Ma=0.0866，Re = U₀·H/ν。

**验证目的**：verified/couette_2d 的 3D 孪生，验证库 3D 动壁半程反弹
（pre-streaming 动壁 BB + 动量注入）路径的正确性。

## 2. 卡点与根因

### 原 FAIL 判决

| 门 | 条款 | fp32 实测 | 判 |
|---|---|---|---|
| G1 精度 | l2≤3e-2, max_rel≤3.0% | 最差 l2=7.247e-05 | ✓ |
| G2a fp32 地板 | l2≤1e-3 | 最差 7.247e-05 | ✓ |
| G2b fp64 精确 | l2≤1e-7 | 最差 1.734e-09 | ✓ |
| **G3 稳态** | steady==True | **H128 fp32 steady=false** | ✗ |

唯一败因：fp32 H128 在 24000 步触顶未过漂移阈 1e-5。

### 根因链（机器证据）

1. **fp32 残差 = 时间线性舍入累积**（diag_dtype.json）：
   - H64 l2 从 1.43e-05@4k 步线性增至 1.16e-04@24k 步
   - 质量漂移 0.0049%→0.0245% 线性增长
   - 局域动壁（dev_top 线性增长，mid/bottom 恒定）

2. **fp64 同核零漂**（diag_dtype.json）：
   - 同核 fp64 在 60k 步 l2 仍在 1e-13 量级（机器地板）
   - 质量漂移 −1e-10% 量级至 60k 步
   - 证明离散半程 BB BGK 的吸引子即解析剖面，方案误差为零

3. **结论**：G3 漂移阈 1e-5 恰落在 fp32 噪声带内，属判稳口径问题而非物理未收敛。

## 3. 稳态口径裁定

### 裁定动作

以 fp64 四档为晋级位，重锁稳态判据：

| H | dtype | l2_rel_err | steady | mass_drift | uz_max | spanwise_std | n_steps |
|---|---|---|---|---|---|---|---|
| 16 | fp64 | 9.596e-14 | ✅ | −1.10e-11% | 1.04e-17 | 0.0 | 20000 |
| 32 | fp64 | 6.700e-13 | ✅ | −1.06e-10% | 7.01e-16 | 0.0 | 20000 |
| 64 | fp64 | 7.232e-13 | ✅ | −1.28e-10% | 1.31e-15 | 0.0 | 20000 |
| 128 | fp64 | 1.734e-09 | ✅ | −1.37e-10% | 4.67e-15 | 0.0 | 20000 |

**判定**：fp64 四档全过 G3（steady=true，质量漂移 <1e-9%，uz_max <1e-14，
spanwise_std=0），漂移阈在机器地板以下。

### 单调条款披露

标准四门之「≥2档网格严格单调收敛」条款在本案替换为 G2a+G2b：
- **根因**：Couette 流解析解为线性剖面，LBM 离散误差为零或极小；
  观测到的 l2 主要为舍入噪声，非空间离散误差，排单调序不可证伪
- **替代条款**：
  - G2a（fp32 地板不敏感）：l2 ≤ 1e-3 → 最差 7.247e-05 ✓
  - G2b（fp64 方案精确）：l2 ≤ 1e-7 → 最差 1.734e-09 ✓
- **披露记录**：prereg.md（FROZEN 2026-10-06，owner 签核项）

## 4. 判据表（四门总览）

| 门 | 条款 | 阈值 | fp64 最差 | 判 |
|---|---|---|---|---|
| G1 精度 | l2≤3e-2, max_rel≤3.0% | 3% | l2=1.734e-09, max=6.59e-07% | ✅ |
| G2a fp32 地板 | l2≤1e-3 | 1e-3 | 7.247e-05 | ✅ |
| G2b fp64 精确 | l2≤1e-7 | 1e-7 | 1.734e-09 | ✅ |
| G3 稳态（fp64） | steady, drift, uz, std | 见上表 | 全过 | ✅ |
| 单调条款 | ≥2档严格单调 | — | **REPLACED_BY_G2a_G2b** | 披露 |

## 5. 审计溯源

### PROVENANCE 机器溯源（零手抄）

| 值 | 文件 | 键 |
|---|---|---|
| 9.596e-14 | gate/fp64/case_H16_fp64.json | l2_rel_err |
| 6.700e-13 | gate/fp64/case_H32_fp64.json | l2_rel_err |
| 7.232e-13 | gate/fp64/case_H64_fp64.json | l2_rel_err |
| 1.734e-09 | gate/fp64/case_H128_fp64.json | l2_rel_err |
| 7.247e-05 | gate/verdict_v2.json | per_tier_fp32.128.l2_rel_err |
| REPLACED_BY_G2a_G2b | w11/prereg.md | monotone_clause_disclosure |

完整路径前缀：`/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/`

### 工件清单

- fp64 四档 case JSON：`gate/fp64/case_H{16,32,64,128}_fp64.json`
- fp32 三档 case JSON：`gate/fp32/case_H{32,64,128}_fp32.json`
- 判决 v2：`gate/verdict_v2.json`（md5: e14f11ddd58e715e4c8d647f04250cf3）
- dtype 诊断：`diag/diag_dtype.json`（fp32 时间线性累积证据）
- 预注册：`w11/prereg.md`（FROZEN 2026-10-06，md5: c6ad6e2ecb9e85a1e51adfa76472ad6a）

## 6. 局限与缓解

| 局限 | 缓解 |
|---|---|
| 单调条款偏差 | prereg 披露 + G2a+G2b 替代（owner 签核项） |
| fp32 残差为舍入累积 | 以 fp64 为晋级位，fp32 仅作地板证据 |
| fp64 H128 l2 升至 1.73e-09 | 仍远小于 3% 门限（约 6 个数量级余量） |
| 仅 20000 步 | fp64 在 60k 步仍零漂（diag_dtype.json），20k 足够 |

## 7. 归档动作

- pending/couette_3d/ → 标记 superseded，指向 verified/couette_3d/
- verified/couette_3d/：README.md + run.py + result.json
- pending/README.md 索引行更新（❌ → ✅→📦 归档完成）

## 8. 预算

- GPU·h：0.0（fp64 四档已在 W11 战役完成，本 mission 仅判据重锁与归档）
- 墙钟：~15 分钟

---

**终裁**：couette_3d 晋级 verified/，fp64 四档为晋级位，稳态判据以 fp64
踪迹重锁，单调条款偏差已披露并用 G2a+G2b 替代。
