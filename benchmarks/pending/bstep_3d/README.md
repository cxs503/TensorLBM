# 三维背向台阶（展向周期，bstep_3d）（pending）

**状态：⚠️ 未按战役标准验证(遗留档)——Re=100 单档 +13.18% > 3%，Re=200/400 档记录损坏（X_r=0），从未按战役预注册协议（τ-matched 多档阶梯 + 参考锁）重跑**

## 物理问题

展向周期三维背向台阶：充分发展抛物线入口（3D Zou/He 剖面）→ 台阶扩张 → Zou/He 压力出口，展向周期 Lz=2H；D3Q19，观测量为底壁再附着长度 X_r/h。

参考（`result_H40.json` reference 节）：Armaly, Durst, Pereira & Schoenung, JFM 127:473-496 (1983)，Re = U_max·h/ν，Re=100 时 X_r/h = 3.0，ER 参考 1.94（本地文件记载；未做战役级参考锁——2D 同族 backward_step 案例的参考审计见其 `REFERENCE_AUDIT.md`，两案例参考族同源）。

## 计算结果现状

| 档 | Re | 网格 | ER（实测） | X_r/h 实测 | 参考 | 误差 | 步数/状态 |
|----|----|------|------------|-----------|------|------|-----------|
| H40 | 100 | 400×82×80, h=40 | 1.951219512195122 | 3.3955459594726562 | 3.0 | +13.184865315755209% | 20000 步，未达收敛判据（converged=false） |
| re200_H49 | 200 | 490×101×98, h=49 | 1.9411764705882353 | 0.0 | 5.5 | −100% | 仅 500 步（early abort，记录不可用） |
| re400_H49 | 400 | 490×101×98, h=49 | 1.9411764705882353 | 0.0 | 8.0 | −100% | 30000 步，X_r 序列杂乱（0/4.48/0 交替），再附着检测失效 |

- H40 档展向均匀性完美（xr_planes_std=0.0——3D 周期等价 2D 确认）；入口剖面校验 max 偏差 6.45e-7、flux_ratio 1.0003。
- re200/re400 的 X_r=0 是测量/运行损坏，不是物理结果（re200 只跑了 500 步；re400 的 xr_series 间歇出现非零值后归零）。

## 不达标清单

（战役严格标准：共性模块入口 + 直接观测量对参考 ≤3% + ≥2 档网格单调收敛；修正/还原变换不算直接模拟）

1. **精度门 ✗**：唯一完整档（Re=100/H40）X_r/h 误差 +13.184865315755209% > 3%。
2. **网格阶梯缺失 ✗**：无任何 τ-matched/网格收敛阶梯（单档）；Re=200/400 档记录损坏不可用。
3. **协议缺失 ✗**：无预注册（NOTES）、无参考锁（ref_lock）、无停损/稳态判据——全部为 2026-08 前后遗留档格式。

## 根因/诊断

**未定源**（遗留档无 campaign 诊断）。可参考的旁证：2D 同族案例 backward_step（W2-C）已将同类 +9~15% 量级的低 Re 分离泡偏长完整定源为与 Re 无关的格点伪差（bounce-back 滑移 ∝(τ−0.5)·Δx + 台阶角一阶分辨率误差，见该目录 `result.json` artifact_decomposition）——本案例 ER/口径同族，同一机制是首要嫌疑，但未经本案例独立验证。

## 晋级路径（转 verified 的条件）

1. 按 backward_step（W2-C）的冻结协议重制：τ-matched 多档阶梯（τ 固定、U_max ∝ 1/h）、预注册 NOTES + 参考锁（Erturk 2008 / Armaly 实验带）、ψ 基再附着主判据。
2. 或改用更具区分度的工况（更高 Re 或 3D 有区分度的展向结构）重定案例目标——遗留 re200/re400 档的方向（大 Re）与此一致，但需全新预注册。
3. 计算预算参考：H40 档（400×82×80×20000 步）机器记录 wall_time_s=250.2——多档阶梯全套约为其 10–20×（估算，按 backward_step 三档+敏感性矩阵的规模折算）。
4. 达标后由归档 PR 将本目录 `pending/bstep_3d` → `verified/bstep_3d`。

<!-- PROVENANCE
[{"v": 3.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "reference.xr_h_ref"},
 {"v": 100.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "physics.re"},
 {"v": 1.951219512195122, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "geometry.expansion_ratio"},
 {"v": 3.3955459594726562, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "result.xr_h"},
 {"v": 13.184865315755209, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "result.xr_h_err_pct"},
 {"v": 20000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "result.n_steps_run"},
 {"v": 0.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "result.xr_planes_std"},
 {"v": 6.454392317833957e-07, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "result.inlet_profile_check.max_abs_dev_over_umax"},
 {"v": 1.0002975347565441, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "result.inlet_profile_check.flux_ratio"},
 {"v": 250.2, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "result.wall_time_s"},
 {"v": 400, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "geometry.nx"},
 {"v": 82, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "geometry.ny"},
 {"v": 80, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "geometry.nz"},
 {"v": 40, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_H40.json", "k": "geometry.step_h"},
 {"v": 5.5, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_re200_H49.json", "k": "reference.xr_h_ref"},
 {"v": 200.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_re200_H49.json", "k": "physics.re"},
 {"v": 1.9411764705882353, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_re200_H49.json", "k": "geometry.expansion_ratio"},
 {"v": 0.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_re200_H49.json", "k": "result.xr_h"},
 {"v": -100.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_re200_H49.json", "k": "result.xr_h_err_pct"},
 {"v": 500, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_re200_H49.json", "k": "result.n_steps_run"},
 {"v": 8.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_re400_H49.json", "k": "reference.xr_h_ref"},
 {"v": 400.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_re400_H49.json", "k": "physics.re"},
 {"v": 0.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_re400_H49.json", "k": "result.xr_h"},
 {"v": -100.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_re400_H49.json", "k": "result.xr_h_err_pct"},
 {"v": 30000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_re400_H49.json", "k": "result.n_steps_run"},
 {"v": 4.4838433265686035, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/bstep_3d/result_re400_H49.json", "k": "result.xr_series[5]"}]
-->
