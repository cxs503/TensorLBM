# CASE-COUETTE-2D 第2轮·调研结合 — 交付总结（m18637）

> 锚点=CASE-COUETTE-2D；第2轮（调研结合）；前轮 m18528 honest-FAIL（fail_stage 未标、failure_reason 未留）。
> 本单先调研成熟方法（开源横向 + 方法谱系），再结合推进；选型沉淀落 oss_solutions（feature_slug='CASE-COUETTE-2D'）。

## 交付物

| 交付 | 位置 | 状态 |
|---|---|---|
| 选型沉淀（oss_solutions） | `oss_solutions` 表 id=435（feature_slug='CASE-COUETTE-2D'） | ✅ 已注册 |
| 注册脚本 | `scripts/register_oss_solution_couette_2d.py` | ✅ 已入库 |
| 报告发布脚本 | `scripts/publish_couette_2d_r2_report.py` | ✅ 已入库 |
| 论文级报告 | `tensor_reports` id=17407 | ✅ 已发布 |
| 算例池验证 | `benchmarks/case_pool/couette_2d/`（既有，main 已含） | ✅ 4/4 测试通过 |
| 图 | `benchmarks/case_pool/couette_2d/artifacts/fig1_couette_profile.png`、`fig2_couette_errors.png` | ✅ 已生成 |

## 调研结论（第2轮）

### ① 开源横向（~/oss-cache）
- **pylbm**：`Bounce_back / LinearBounceBack` 边界族（4/(1+2d)、2/(1+d) 变体）+ `demo/2D/Poiseuille.py` 演示——同一壁面处理谱系，可搬其边界类结构。
- **Maarten-vd-Sande_lbm**：自由表面 LBM（`update_types.py` 界面质量再分配）——不适用本无自由面问题。
- 无直接 Couette 实现；Couette 属动壁/剪切型验证，与 Poiseuille 同族。

### ② 方法谱系
- **Zou & He 1997**（Phys. Fluids 9, 1591）：LBM BGK 压力/速度边界条件。
- **Ladd 1994**（J. Fluid Mech. 271, 285）：移动壁 bounce-back 动量修正。
- **Bouzidi 2001**（Phys. Fluids 13, 3452）：插值 bounce-back 壁面亚格定位。
- **Kruger 2017**（Springer）：Couette/Poiseuille 规范验证。

### ③ 选型（落 oss_solutions）
- **方法**：D2Q9 BGK + pre-streaming 半步反弹动壁（Ladd/Zou-He 动量修正 `f_new[opp]=f_pre[i]+2w rho (c.u_wall)/cs^2`），壁面在 y=0.5 与 y=ny-1.5，有效缝宽 H_eff=ny-2。
- **理由**：半程 BB 对线性剖面是精确贴合壁面处理（无几何台阶误差），与 pylbm/lbmpy 谱系一致。
- **迁移**：7 条可复用约定（壁面 0.5 偏移、闭区间含壁节点积分、无量纲系数跨缝高比较、u_ref 掩膜、稳态门、末段时间平均、低黏性/MRT 漂移护栏）。

### ④ 结合实现（复用仓内既有能力）
- **Route A**（主）：`benchmarks/verified/couette_2d/run.py::run_case` — D2Q9 + 库原语。
- **Route B**（独立验证）：`tensorlbm.curved_boundary_calibration.run_plane_couette` — D3Q19 插值反弹两力路。
- **验证**：`tests/test_case_pool_couette_2d.py` 4/4 通过（闭式自洽 + 实跑容差 + 已提交工件验收 + Route B 力路）。

## 验收判据（总账 CASE-COUETTE-2D）

【算例池|COUETTE-2D｜ref: 线性解析解｜容差: <2%】

| 量 | 相对误差 | 判据 | 判定 |
|---|---|---|---|
| 全剖面 L2 | 1.471e-02% | <2% | ✓ |
| 逐点最大 | 2.434e-02% | <2% | ✓ |
| 近壁行 | 1.570e-03% | <2% | ✓ |
| 壁面剪应力 τ_w | 1.489e-03% | <2% | ✓ |
| 动量积分 | 1.476e-02% | <2% | ✓ |
| Route B 力（ME/CV） | 3.241e-05%~1.418e-03% | <2% | ✓ |
| 两路一致 | 1.231e-02% | <2% | ✓ |

**容差族 = PASS**（全部 <2%，比门低 2~4 个数量级）。

**honest 部分分**：预注册护栏（质量漂移 <0.001%、收敛阶 ≥ −0.25）有 3 项未满足（τ=0.6/MRT 漂移、收敛阶 -0.77），如实列入判定表不粉饰。

## 环境约束（如实记录）
- 服务器磁盘满（No space left on device）导致报告图上传失败 → 报告以无图形式发布（id=17407），图已生成于 artifacts/ 随分支入库。
- 前轮 m18528 失败原因（fail_stage 未标、failure_reason 未留）已在本次报告中补全。
