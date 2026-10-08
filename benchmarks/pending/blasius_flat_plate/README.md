# 平板边界层 Blasius（中置薄板，D2Q9）（pending）

**状态：❌ 未达标(根因类)——测量公式修正后主门最好 +27.34% ≫ 3%；旧档正误差 +75~102%（buggy 权重口径）的首因是测量公式 bug（已定位并修正），残余为有限 Re_x 物理效应主导**

（Wave-7 W7-D 判决记录；本 README 为案例导引，完整判决见 `result.json` 与 `verify.py`。）

## 物理问题

均匀流 U 掠过半无限薄平板的层流定常边界层，Blasius 相似解 f'''+f·f''/2=0：η=(y−y_w)·√(U/(ν(x−x0)))，u/U=f'(η)，f''(0)=0.3320573362151963，C_f = 0.6641146724/√Re_x。参考为本目录 `run.py` 内 RK4 打靶自解（h=0.005，`result.json` reference 节：probe x_eff=180、Re_x=900、C_f_ref=0.022137155747679754）。

设置：中置单行薄板（plate_y=ny/2，双侧流体，上下镜面对称远离平板）；plate200（nx=240,ny=1400,plate_len=200）/plate400（nx=440,ny=1400,plate_len=400）/plate400_y1600（ny=1600）；U=0.05、ν=0.01（τ=0.53）、30000 步；BGK 与 TRT（Λ=3/16）双碰撞；half-way BB 双侧表面 + Zou-He 压力出口；碰撞/迁移/边界全走库公共入口。

## 计算结果现状

判定口径 = 修正权重 FD（首 3 流体行二阶单侧差分，权重 (−2,3,−1)；无任何修正/还原/外推），probe x_eff=180（Re_x=900）：

| run | 网格 | 碰撞 | C_f 实测 | C_f 误差（修正 FD） | 旧 buggy 权重 | ME 动量交换交叉 |
|-----|------|------|---------|--------------------|---------------|----------------|
| R1p | plate200 | BGK | 0.032614540308713906 | +47.33% | +101.91% | +47.28% |
| R2p | plate200 | TRT | 0.03256269544363021 | +47.10% | +101.90% | +47.70% |
| R3p | plate400 | TRT | 0.028198640793561932 | +27.38% | +75.04% | +27.38% |
| R4p | plate400 | BGK | 0.028356648981571194 | +28.10% | +75.96% | +28.11% |
| R5p | plate400_y1600 | BGK | 0.02835182845592498 | +28.07% | +75.92% | +28.05% |
| R6p | plate400_y1600 | TRT | 0.028190111741423603 | **+27.34%** | +74.99% | +27.35% |

- 域高无关（y1600 ≡ y1400）；稳态/守恒良好（umax 漂移 ≤1.1e-5、质量漂移 ≤0.004%、全 finite）。
- η 剖面二级证据：L2(0.05<η<5)=0.19899796529597488（plate400 系）/0.27881049235399086（plate200 系）。

## 不达标清单

（战役严格标准：共性模块入口 + 直接观测量对参考 ≤3% + ≥2 档网格单调收敛；修正/还原变换不算直接模拟）

1. **主门 ✗**：`result.json` judgment.main_gate：grids=[plate200, plate400]，best_err_pct=27.342970627011443，all_within_3pct=false，pass=false。
2. **单调条款 ✓**：plate400 系（27.34–28.10%）≤ plate200 系（47.10–47.33%），monotone_error_decrease_plate200_to_plate400=true。
3. **基线复现 ✓**：R1（旧 buggy 口径，与 2026-08-19 旧档同法）+101.91%，within_5pp=true（对旧档记录复现）。

## 根因/诊断

已定源（`result.json` attribution 节 + `verify.py` 独立审计）：

1. **旧档测量公式 bug**：旧 FD 权重解错方程（solve(A,e2) 应为 solve(Aᵀ,e2)），buggy dudy = 正确 + 0.75·u(0.5)——旧档 C_f +76~102% 的首因（合成二次曲线检验：buggy 偏高、修正逐位精确，verify.py V2 第一性原理重推）；旧"低 τ 近壁格子误差"归因及其 τ 扫描结论作废。
2. **有限 Re_x=900 边缘加速**（物理真实，非数值）：归因扫描（attr/ τ 阶梯）显示 cf_ue 误差随 Re_x 降而单调增长、碰撞无关、域高无关——有限 Re_x 位移自感应物理，非低 τ 墙数值效应。
3. **TRT Λ=3/16 ≡ BGK**：两测量层级差在亚 pp 量级（见上表成对数值），"BB 墙位 τ 耦合是地板来源"的假设在这类问题中被否定（与球系记录互为独立复核）。
4. 板尾效应：板尾附近 C_f 爬升（数值注记），plate200 probe（x_eff=180）距板尾比 plate400 近得多——plate200/plate400 差异部分来自出口距离。

## 晋级路径（转 verified 的条件）

1. **加大有效 Re_x**：延长板长并把 probe 移向板中段远端（le、L 与域同步放大），压低有限 Re_x 物理位移项——C_f 误差随 Re_x 单调下降的归因链已建立；预算：plate400 系 30k 步 CPU 32 线程可日内完成，加倍板长约 2–4× CPU 时（估算）。
2. 保持修正 FD 直接口径不变（修正/还原不进判定），若 Re_x 推大后最好档仍 >3%，如实维持 FAIL。
3. 达标后由归档 PR 将本目录 `pending/blasius_flat_plate` → `verified/blasius_flat_plate`（旧档测量 bug 披露条款随迁）。

<!-- PROVENANCE
[{"v": 0.3320573362151963, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/results/R1p_plate200_bgk_tau053.json", "k": "blasius_fpp0_ref"},
 {"v": 0.6641146724303926, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/results/R1p_plate200_bgk_tau053.json", "k": "cf_coeff_locked"},
 {"v": 180, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "reference.probe_x_eff"},
 {"v": 900, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "reference.Re_x"},
 {"v": 0.022137155747679754, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "reference.cf_ref"},
 {"v": 3.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "reference.gate_pct"},
 {"v": 0.032614540308713906, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[0].cf_sim"},
 {"v": 47.32940708578749, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[0].cf_err_pct_corrected_fd"},
 {"v": 101.91288120159699, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[0].fd_err_old_buggy_weights"},
 {"v": 47.280385415642705, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[0].me_err"},
 {"v": 0.03256269544363021, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[1].cf_sim"},
 {"v": 47.0952086834515, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[1].cf_err_pct_corrected_fd"},
 {"v": 101.89792511694684, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[1].fd_err_old_buggy_weights"},
 {"v": 47.704864235174455, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[1].me_err"},
 {"v": 0.028198640793561932, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[2].cf_sim"},
 {"v": 27.381498847328196, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[2].cf_err_pct_corrected_fd"},
 {"v": 75.03615971627323, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[2].fd_err_old_buggy_weights"},
 {"v": 27.3795636014944, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[2].me_err"},
 {"v": 0.028356648981571194, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[3].cf_sim"},
 {"v": 28.095267995498112, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[3].cf_err_pct_corrected_fd"},
 {"v": 75.95526265446277, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[3].fd_err_old_buggy_weights"},
 {"v": 28.105964435785857, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[3].me_err"},
 {"v": 0.02835182845592498, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[4].cf_sim"},
 {"v": 28.073492272811972, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[4].cf_err_pct_corrected_fd"},
 {"v": 75.92350358750808, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[4].fd_err_old_buggy_weights"},
 {"v": 28.04949785388327, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[4].me_err"},
 {"v": 0.028190111741423603, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[5].cf_sim"},
 {"v": 27.342970627011443, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[5].cf_err_pct_corrected_fd"},
 {"v": 74.98687510511843, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[5].fd_err_old_buggy_weights"},
 {"v": 27.352401795158954, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[5].me_err"},
 {"v": 27.342970627011443, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "judgment.main_gate.best_err_pct"},
 {"v": 101.91288120159699, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "judgment.baseline_reproduction.R1_buggy_weights_pct"},
 {"v": 0.19899796529597488, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[2].l2_rel_err_profile"},
 {"v": 0.27881049235399086, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[0].l2_rel_err_profile"},
 {"v": 30000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[0].n_steps"},
 {"v": 1.149152771998413e-05, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[0].umax_drift_last_2000"},
 {"v": 0.004393263878503222, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/result.json", "k": "runs[2].mass_drift_pct"},
 {"v": 240, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/results/R1p_plate200_bgk_tau053.json", "k": "nx"},
 {"v": 1400, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/results/R1p_plate200_bgk_tau053.json", "k": "ny"},
 {"v": 200, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/results/R1p_plate200_bgk_tau053.json", "k": "plate_len"},
 {"v": 0.05, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/results/R1p_plate200_bgk_tau053.json", "k": "U"},
 {"v": 0.01, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/results/R1p_plate200_bgk_tau053.json", "k": "nu"},
 {"v": 0.53, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/results/R1p_plate200_bgk_tau053.json", "k": "tau"},
 {"v": 20, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/results/R1p_plate200_bgk_tau053.json", "k": "x0_leading_edge"},
 {"v": 440, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/results/R4p_plate400_bgk_tau053.json", "k": "nx"},
 {"v": 1400, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/results/R4p_plate400_bgk_tau053.json", "k": "ny"},
 {"v": 400, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/results/R4p_plate400_bgk_tau053.json", "k": "plate_len"},
 {"v": 1600, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/blasius_flat_plate/results/R6p_plate400_y1600_trt.json", "k": "ny"}]
-->
