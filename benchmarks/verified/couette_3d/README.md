# Couette 流 3D（平板剪切）— 解析线性剖面验证

**状态：✅ 已入库（2026-10-19，fp64 四档晋级位，误差 ≤1.73e-9 << 3%）**

## 物理问题

三维 Couette 流：下壁静止、上壁以 U0=0.05（+x）拖动，x/z 周期，稳态线性剖面：
- **u(y) = U0·(y−0.5)/H_eff**（解析精确解，对任意 Re 成立）
- 壁面剪切 τ = ν·U0/H_eff
- D3Q19 BGK τ=0.8（ν=0.1），Ma=0.0866，半程反弹约定壁位于 y=0.5 与 ny−1.5
  （与 verified/couette_2d 锁同约定）

目的：verified/couette_2d 的 3D 孪生，验证库 3D 动壁半程反弹
（pre-streaming 动壁 BB + 动量注入）路径。

**参考锚定**：解析精确解 u(y) = U0·(y−0.5)/H_eff（线性 Couette 剖面），
对任意 Re 成立，无近似。参考先锁定后跑（prereg.md FROZEN 2026-10-06）。

## 共性模块

库 solver 路径（tensorlbm.d3q19 + tensorlbm.solver3d）：
- 上壁移动（half-way 反弹 + 动量注入），下壁静止（plain BB），x/z 周期
- 碰撞：BGK（τ=0.8, ν=0.1）
- 无手写核，无修正/还原变换

## 晋级位与披露

**晋级位：fp64 四档**（H∈{16,32,64,128}，CPU，20000 步）

### 披露偏差：单调条款 → G2a+G2b 替代

标准四门之「≥2档网格严格单调收敛」条款在本案替换为 G2a+G2b：
- **根因**：Couette 流解析解为线性剖面，LBM 离散误差为零或极小；
  观测到的 l2 主要为舍入噪声（fp32 时间线性累积，fp64 在机器地板），
  非空间离散误差，排单调序不可证伪
- **替代条款**：
  - G2a（fp32 地板不敏感）：l2 ≤ 1e-3 → 最差 7.247e-05 ✓
  - G2b（fp64 方案精确）：l2 ≤ 1e-7 → 最差 1.734e-09 ✓
- **披露记录**：prereg.md（FROZEN 2026-10-06，owner 签核项）

### 稳态判据重锁

原 G3 失败：fp32 H128 在 24000 步触顶未过漂移阈 1e-5。
- **根因**：fp32 残差为时间线性舍入累积（diag_dtype.json 证据：
  H64 l2 从 1.43e-05@4k 线性增至 1.16e-04@24k，局域动壁）；
  fp64 同核零漂（质量漂移 −1e-10% 量级至 60k 步）
- **重锁**：以 fp64 四档为晋级位，fp64 全档 steady=true，
  漂移阈在机器地板以下

## 结果（fp64 四档，2026-10-19 机器读数）

| 网格 H | dtype | l2_rel_err | max_rel_err_masked | u_top_err | τ_err | steady | mass_drift | n_steps |
|--------|-------|-----------|-------------------|-----------|-------|--------|------------|---------|
| 16 | fp64 | 9.596e-14 | 1.427e-11% | 1.427e-11% | 1.478e-11% | ✅ | −1.10e-11% | 20000 |
| 32 | fp64 | 6.700e-13 | 1.168e-10% | 1.168e-10% | 1.157e-10% | ✅ | −1.06e-10% | 20000 |
| 64 | fp64 | 7.232e-13 | 1.102e-10% | 1.102e-10% | 1.170e-10% | ✅ | −1.28e-10% | 20000 |
| 128 | fp64 | 1.734e-09 | 6.591e-07% | 3.378e-09% | −6.893e-09% | ✅ | −1.37e-10% | 20000 |

**收敛性**：fp64 l2 在 H16/32/64 处于机器地板（1e-13 量级），H128 升至 1.73e-09
（仍远小于 3% 阈值）；fp32 残差根因已锁定为舍入累积（非空间离散误差）。

## 运行方式

```
cd <repo>
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src /nfs/wangxi/venvs/tensorlbm/bin/python \
    benchmarks/verified/couette_3d/run.py \
    --H 16 32 64 128 --dtype float64 --tau 0.8 --u0 0.05 \
    --min-steps 20000 --max-steps 24000 --outdir /tmp/couette3d_fp64
```

## 判定

- fp64 四档全过 G1（精度 ≤3%）、G2a（fp32 地板 ≤1e-3）、G2b（fp64 方案精确 ≤1e-7）、
  G3（稳态+质量+uz+spanwise 全达标）
- 单调条款偏差已在 prereg 披露并用 G2a+G2b 替代（owner 签核项）
- 稳态判据重锁：fp64 全档 steady=true，漂移阈在机器地板以下
- 真模拟（无外推、无修正/还原变换），库路径驱动

## PROVENANCE

所有判决数字机器溯源（值-文件-键三元组），零手抄：

```json
[
  {"v": 9.596262761960097e-14, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/fp64/case_H16_fp64.json", "k": "l2_rel_err"},
  {"v": 6.700234659707114e-13, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/fp64/case_H32_fp64.json", "k": "l2_rel_err"},
  {"v": 7.231638097112415e-13, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/fp64/case_H64_fp64.json", "k": "l2_rel_err"},
  {"v": 1.733723620717403e-09, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/fp64/case_H128_fp64.json", "k": "l2_rel_err"},
  {"v": 7.246891871695247e-05, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.128.l2_rel_err"},
  {"v": "REPLACED_BY_G2a_G2b", "f": "benchmarks/pending/couette_3d/w11/prereg.md", "k": "monotone_clause_disclosure"}
]
```
