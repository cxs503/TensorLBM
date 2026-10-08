# 球 Re=100 直接力法基准（sphere_re100_mem）（pending）

**状态：🟡 达标待晋级——W8-A 大域阶梯（BFL 稀疏核解锁）T1 D40 +2.771% / T2 D60 +2.540% 双过 3% 门且 |err| 随 D 单调降（judgment.pass=true），晋级 verified/ 由库主裁定**

三代判档链（同一物理问题、逐代收窄）：W5-B 力法修复后 +11% 地板（FAIL）→ W7-B 域形修复定源入口钳制（FAIL 维持，压掉 ~7.2pp）→ W8-A 稀疏核解锁大域（**PASS 候选**）。三代机器档案全在本目录（`result.json` / `w7/` / `w8a/`）。

## 物理问题

均匀来流绕球 Re=100，观测量 = 阻力系数 Cd（直接测力：BFL 逐链动量账本，实验室系，末 20% 采样窗均值）。参考（三代 result.json 同锁）：

- **主锁 Schiller-Naumann(0.687)：Cd_ref = 1.0917310910948732**（门 ±3%、两档 |err| 单调降）；
- 交叉：S-N(0.681) = 1.0685190542670633、Clift-Gauvin = 1.1092345787735252（锁定参考位于交叉族低端，不重开）。

库共性模块：D3Q19 + `collide_mrt3d_low_memory` + BFL 插值边界（`bouzidi_bounce_back_d3q19`，PR #316 稀疏链核）或 BB 楼梯副观测 + hard far field + 每 200 步质量校正；铁律自查零手写核。

## 计算结果现状

**W5-B（2026-09-21，up=1.25D/lat2.0，12000 步）**——力法修复（PR #309 wet-node）后首个正式对：

| 运行 | 路线 | D | Cd | err |
|------|------|---|----|-----|
| formal_bfl_D40 | BFL（判定） | 40 | 1.214927032369151 | +11.28445844211758% |
| formal_bfl_D60 | BFL（判定） | 60 | 1.2141675265765555 | +11.214889498007558% |
| formal_bb_D40 | 楼梯副观测 | 40 | 1.2922772551177482 | +18.369556904507647% |
| formal_bb_D60 | 楼梯副观测 | 60 | 1.2831206191931062 | +17.5308305918349% |

**W7-B（域形修复，lat2.0/up2.75/down2.25，BFL 24000 步）**：

| 运行 | D | Cd | err |
|------|---|----|-----|
| formal_bfl_D40 | 40 | 1.1376133384841147 | +4.2027059377073535% |
| formal_bfl_D60 | 60 | 1.1355162174389517 | +4.010614582769403% |
| formal_bb_D40（12k 副观测） | 40 | 1.150965599401582 | +5.425741630872099% |
| formal_bb_D60（12k 副观测） | 60 | 1.1462960211174618 | +4.99801924371931% |

**W8-A（大域 lat3.0/up3.0/down4.0，BFL 稀疏核；w8a/result.json，verify.py 独立重算）**：

| 档 | D | Cd（末 20% 窗） | err | 稳态漂移 | CV 对账（窗均） | BFL 链数 | 步数 |
|----|---|----------------|-----|---------|----------------|---------|------|
| T1 | 40 | 1.1219804015117896 | **+2.7707656824703957%** | −0.14475665095754808% | 0.0001496104414234331 | 28722 | 12000 |
| T2 | 60 | 1.119462001144486 | **+2.5400861325477218%** | −0.0420075943755402% | 7.813769713472235e-05 | 64794 | 24000 |

- **阶梯判决 PASS**：all_within_gate_and_steady=true、monotone_error_decrease=true（2.7707656824703957% → 2.5400861325477218%）；质量漂移 −0.07971931784817472 ppm / 0.0 ppm。
- **位一致链**：T1（稀疏核）对 W7 密集核存档 `w7/diag/h2_big.json` 240 采样 × cd/cl/cs/mass 逐位相同（t1_reproduction.exact=true）；W7 驱动 vs W8-A 驱动 D20 链亦逐位（bitwise_chain.exact=true）。
- **解锁证据**（PR #316 前该域形不可跑）：D60 大域稀疏路线 600 步峰值 26.672595977783203 GiB alloc（reserved 30.201171875 GiB）；同域密集核第 1 步即 OOM（torch 峰值已分配 30.045341968536377 GiB，`mem_dense_D60big.json.oom.json`）；核速度 sparse/dense 中位比 0.40701854646534846（1.2M 胞）/0.1436306232413634（9.6M 胞）——更快。

## 不达标清单

（战役严格标准：共性模块入口 + 直接观测量对参考 ≤3% + ≥2 档网格单调收敛；修正/还原变换不算直接模拟）

1. **W5-B ✗**：双档 +11.28445844211758%/+11.214889498007558% > 3%（judgment.pass=false；单调条款 ✓）。
2. **W7-B ✗**：双档 +4.2027059377073535%/+4.010614582769403% > 3%（pass=false；单调条款 ✓）。
3. **W8-A ✓ 全过**：两档 ≤3%、稳态、单调、τ 校验、CV 对账、位一致链——**唯一未闭环项是目录仍在 pending/**（晋级为库主行政决定）。

## 根因/诊断

+11% 地板已完整定源（w7/ 诊断矩阵，19 份 diag 档案）：

- **主因 = 入口平面 Dirichlet 钳制**（`far_field_bc_3d` 在 x=0 强制自由流平衡态，压制球前轴向上游势流减速）：单变量 up 阶梯（D40/lat2.0）err = +11.28445844211758%（up1.25）→ +5.825977524186044%（up2.0）→ +4.233570314640698%（up2.75@12k）→ +3.9367347183442902%（up3.0）→ +3.309186859778074%（up4.0）→ +2.7707656824703957%（lat3/up3/down4 大域）；五点最小二乘签名 err(up) ≈ E0 + C·(a/x)³（w7/README 记载，独立复算吻合）。
- 排除项（均 w7/diag 机器档案）：碰撞算子四列（MRT/TRT/MRT 魔术 s_q/BGK）差在亚 pp 带内（墙位 τ 耦合假设不成立）；非平衡远场反射 ≈ hard 远场；出流截断平坦；横向为次级贡献（大域联合另压 ~1pp）。
- W5-B 横向"堵塞饱和"全程固定 up=1.25D——实为入口项主导下的饱和（与 up 假设相容）。

## 晋级路径（转 verified 的条件）

1. **库主裁定晋级**（数据侧已闭环，零计算成本）：归档 PR 将 `pending/sphere_re100_mem` → `verified/sphere_re100_mem`（建议同批处理同问题的 `pending/sphere_re100`，见其 README 晋级路径）。
2. （可选加固）第三档 D80 孪生（胞数约 2.4× T2、步数按 t_c 放大 → 显存超单卡 32G，需 64G 卡或多卡分域；估算个位数 GPU·h/档，由 w8a 每步实测成本外推）进一步压窄单调带。
3. 随迁披露：S-N 主锁位于交叉族低端（T1 大域点对 Clift-Gauvin 差约 +1.1pp，锁定参考不重开）；W5-B/W7-B 世代记录保留为本目录内定源档案。

<!-- PROVENANCE
[{"v": 1.0917310910948732, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "reference.cd_ref"},
 {"v": 1.0685190542670633, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "reference.cross_references.schiller_naumann_0681"},
 {"v": 1.1092345787735252, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "reference.cross_references.clift_gauvin"},
 {"v": 3.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "reference.gate_pct"},
 {"v": 1.214927032369151, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "runs[2].cd_judged"},
 {"v": 11.28445844211758, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "runs[2].err_pct"},
 {"v": 1.2141675265765555, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "runs[3].cd_judged"},
 {"v": 11.214889498007558, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "runs[3].err_pct"},
 {"v": 1.2922772551177482, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "runs[0].cd_judged"},
 {"v": 18.369556904507647, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "runs[0].err_pct"},
 {"v": 1.2831206191931062, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "runs[1].cd_judged"},
 {"v": 17.5308305918349, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "runs[1].err_pct"},
 {"v": 1.1376133384841147, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json", "k": "runs[0].cd_window"},
 {"v": 4.2027059377073535, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json", "k": "runs[0].err_pct"},
 {"v": 1.1355162174389517, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json", "k": "runs[1].cd_window"},
 {"v": 4.010614582769403, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json", "k": "runs[1].err_pct"},
 {"v": 1.150965599401582, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json", "k": "runs[2].cd_window"},
 {"v": 5.425741630872099, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json", "k": "runs[2].err_pct"},
 {"v": 1.1462960211174618, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json", "k": "runs[3].cd_window"},
 {"v": 4.99801924371931, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json", "k": "runs[3].err_pct"},
 {"v": 5.825977524186044, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json", "k": "domain_convergence_signature[1].err_pct"},
 {"v": 4.233570314640698, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json", "k": "domain_convergence_signature[2].err_pct"},
 {"v": 3.9367347183442902, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json", "k": "domain_convergence_signature[3].err_pct"},
 {"v": 3.309186859778074, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json", "k": "domain_convergence_signature[4].err_pct"},
 {"v": 2.7707656824703957, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w7/result.json", "k": "domain_convergence_signature[5].err_pct"},
 {"v": 1.1219804015117896, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[0].cd_window"},
 {"v": 2.7707656824703957, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[0].err_pct"},
 {"v": -0.14475665095754808, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[0].drift_pct"},
 {"v": 0.0001496104414234331, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[0].cv_closure_rel_window"},
 {"v": 28722, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[0].bfl_links"},
 {"v": 12000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[0].steps"},
 {"v": -0.07971931784817472, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[0].mass_drift_ppm"},
 {"v": 1.119462001144486, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[1].cd_window"},
 {"v": 2.5400861325477218, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[1].err_pct"},
 {"v": -0.0420075943755402, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[1].drift_pct"},
 {"v": 7.813769713472235e-05, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[1].cv_closure_rel_window"},
 {"v": 64794, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[1].bfl_links"},
 {"v": 24000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[1].steps"},
 {"v": 26.672595977783203, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/mem_sparse_D60big.json", "k": "meta.mem_peak_loop_gib"},
 {"v": 30.201171875, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/mem_sparse_D60big.json", "k": "meta.mem_peak_reserved_gib"},
 {"v": 600, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/mem_sparse_D60big.json", "k": "meta.steps"},
 {"v": 30.045341968536377, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/mem_dense_D60big.json.oom.json", "k": "torch_peak_allocated_gib"},
 {"v": 0.40701854646534846, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/speed_probe.json", "k": "cases[0].ratio_sparse_over_dense"},
 {"v": 0.1436306232413634, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/speed_probe.json", "k": "cases[1].ratio_sparse_over_dense"}]
-->
