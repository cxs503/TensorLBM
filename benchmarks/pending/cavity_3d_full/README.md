# 全 3D 方腔流 cavity_3d_full（5 壁 + 移动顶盖，Re=1000）（pending）

**状态：🟡 达标待晋级——实为已晋级：TRT 解已于 2026-08-20 入库 `benchmarks/verified/cavity_3d_full/`；本目录是同日的 MRT/RLBM 发散诊断历史稿，无独立达标数据，待归档 PR 清理（删除或标注 superseded）。**

## 物理问题

全 3D 立方方腔（5 静止壁 + 顶盖移动，Lz=H 有限非展向周期），Re=1000：
- 顶盖 y=ny−1 以 u_lid 沿 +x 移动，驱动腔内 3D 涡结构；侧壁边界层使中心平面
  主涡弱于 2D（约 −28% 量级），必须用 3D 参考而非 2D Ghia 判据。
- 参考：**Ku, Hirsh & Taylor (1987) 谱方法**，u_min(x=0.5, z=0.5)=−0.2751
  （arXiv:1503.03337 D3Q14 MRT 97³ 收敛序列末端同值）。
- 与 verified/cavity/3d（展向周期）不同：本案例 z 向也是固壁（真立方腔）。

## 计算结果现状

本目录（pending）记录的是**碰撞格式选型诊断**（result.json，2026-08-20）：

| 记录 | 内容 | 结论 |
|------|------|------|
| Re400 组合扫描 | MRT/RLBM × 展向周期/全壁 4 组合 | 全部稳定 |
| Re1000 稳定性 | MRT 173 步 / RLBM 200 步 NaN（tau=0.5096 过低，发散起于顶盖层） | **MRT/RLBM 均发散** |

同日切换 **TRT 碰撞**后完成正式两档验证，数据在 `benchmarks/verified/cavity_3d_full/`
（result.json，D3Q19 + V3-3D 边界，64³/96³ 各 100000 步）：

| 档位 | u_min（中心线） | err vs Ku(−0.2751) | 3D 修正量 vs 2D |
|------|----------------|--------------------|-----------------|
| 64³  | −0.27142 | 1.338% | 29.11% |
| 96³  | −0.27581 | 0.259% | 27.97% |

误差单调下降（`convergence.monotone_decrease: true`），`verified: true`（2026-08-20）。
Wave-2 严格标准重审（PR #299，2026-09-19）澄清本案例为**非修正口径**
（直接观测量、无外推/人工因子），维持 verified 身份。

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

- 共性模块入口：本目录 run.py 用 `boundaries3d.zou_he_moving_lid_3d` +
  `solver3d` 碰撞/流播 + V3 pre-streaming 半程反弹（与 verified/cavity 家族同款）。
- 直接观测量 ≤3%：本目录自身无测量数据（发散诊断档）；达标数据在 verified/
  （0.259–1.338%）。
- ≥2 档网格单调收敛：同上，verified/ 已有 64³→96³ 单调。
- 结论：**本目录不是一份独立判决，是 verified 案例的前史**。

## 根因/诊断

- MRT 在 Re≥576（tau→0.5）失稳是库内已知行为（verified/cavity/re1000 记录）；
  本案例 Re=1000 等_Re 配方下 tau≈0.51，MRT 与正则化 RLBM 都在顶盖层发散。
- **TRT（魔法参数对称化）是稳定解**——verified/cavity_3d_full/result.json 的
  `collision` 字段即 "TRT（MRT/RLBM 在 Re=1000 发散——TRT 是稳定解）"。

## 晋级路径

无需任何重跑：案例已达标且在 `verified/cavity_3d_full/`。处置 =
下次归档 PR 将本 pending 目录删除（或改为指向 verified 的 superseded 占位）。
诊断价值（MRT/RLBM 发散步数、TRT 选型过程）已并入 verified 侧 result.json 的
描述字段。

<!-- PROVENANCE
[{"v": 1.338, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/cavity_3d_full/result.json", "k": "grids.64.err_pct_vs_ku"},
 {"v": 0.259, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/cavity_3d_full/result.json", "k": "grids.96.err_pct_vs_ku"},
 {"v": -0.27142, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/cavity_3d_full/result.json", "k": "grids.64.u_min"},
 {"v": -0.27581, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/cavity_3d_full/result.json", "k": "grids.96.u_min"},
 {"v": 29.11, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/cavity_3d_full/result.json", "k": "grids.64.3d_correction_pct"},
 {"v": 27.97, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/cavity_3d_full/result.json", "k": "grids.96.3d_correction_pct"},
 {"v": 100000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/cavity_3d_full/result.json", "k": "grids.64.steps"},
 {"v": 100000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/cavity_3d_full/result.json", "k": "grids.96.steps"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_3d_full/result.json", "k": "verified"}]
-->
