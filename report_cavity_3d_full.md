# 平台报告：cavity_3d_full 归档清理（BENCH-CLOSE m60）

## 1. 物理问题表述

全 3D 立方方腔流（Re=1000）：5 个静止壁面 + 1 个移动顶盖（沿 +x 方向匀速），
腔体三个方向均为有限长度（Lx=Ly=Lz=H，无展向周期性）。顶盖运动驱动腔内
三维涡旋结构，侧壁边界层使中心平面主涡强度弱于 2D 情形约 28%。

**参考基准**：Ku, Hirsh & Taylor (1987) 谱方法，中心线 x=0.5, z=0.5 处
u_min = −0.2751（三维立方腔，非 2D Ghia 值 −0.3829）。

## 2. 验证结果（来自 verified/cavity_3d_full/result.json）

### 网格收敛序列

| 档位 | u_min | err vs Ku | 3D 修正量 | 残差 |
|------|-------|-----------|-----------|------|
| 64³  | −0.27142 | 1.338% | 29.11% | 9.75e−07 |
| 96³  | −0.27581 | 0.259% | 27.97% | 3.82e−06 |

**单调收敛**：1.338% → 0.259% 严格下降（monotone_decrease: true）。

### 四门判据

| 判据 | 状态 | 证据 |
|------|------|------|
| 共性模块入口 | ✅ PASS | src/tensorlbm 库（solver3d + boundaries3d）驱动 |
| 直接观测量 ≤3% | ✅ PASS | 96³ 档 0.259% ≤ 3%；64³ 档 1.338% ≤ 3% |
| ≥2 档单调收敛 | ✅ PASS | 1.338% → 0.259% 严格单调 |
| 参考先锁定后跑 | ✅ PASS | Ku 1987 参考预注册，verified_date=2026-08-20 |

## 3. 本次归档清理动作

`benchmarks/pending/cavity_3d_full/` 原为 2026-08-20 的 MRT/RLBM 碰撞格式
诊断稿（MRT 173 步发散，RLBM 200 步发散），不含独立达标数据。同日切换 TRT
碰撞后在 `verified/cavity_3d_full/` 完成正式验证。

本次操作：
1. 删除 pending 目录下 run.py（诊断脚本）与 result.json（发散结果）
2. 简化 README.md 为 superseded 标记，指向 verified/cavity_3d_full
3. 更新 pending/README.md 索引行：🟡 → ⏸️ superseded，晋级动作→"归档完成"

## 4. 审计溯源

- PROVENANCE 值直读自 `benchmarks/verified/cavity_3d_full/result.json`
- 本目录 PROVENANCE 块已更新，仅保留指向 verified 的引用键
- verified/ 目录未做任何修改
- pending/README.md 统计行更新（状态、卡点、晋级动作）

## 5. 局限与缓解

- **局限**：pending 目录仅保留 superseded 标记，MRT/RLBM 发散历史诊断细节
  已并入 verified 侧 result.json 的 description 字段
- **缓解**：TRT 碰撞在 Re=1000 下稳定，MRT/RLBM 发散归因记录保留在 verified
  侧，供后续碰撞核对比研究使用

## 6. 判据表（机器可读）

见 `benchmarks/pending/cavity_3d_full/validation_report.json`。
