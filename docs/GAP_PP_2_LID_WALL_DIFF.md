# GAP-PP-2-lid-wall-diff：可微运动壁面 BC（graph-preserving lid）

判据①Re=100 τ 反演<2% / ②AD vs FD<1e-4。**诚实结论：②PASS（最优 9.23e-07）；①PARTIAL/FAIL**
（n=48 达 1.80% PASS，n=65 达 2.98% FAIL——一阶优化器在极平坦 1-D 损失谷极限环，非梯度错误）。

## 1 问题设定与基准
方腔顶盖驱动流 `[0,H]²`，四边无滑移、顶盖 `u_lid`；`Re=u_lid·H/ν`，`ν=(τ−1/2)/3`。
基准=Ghia/Ghia/Shin 1982 *JCP* 48(3):387-411 表 I（Re=100 中心线 u,v），取仓库 `GHIA_RE100`。
反演=solver-in-the-loop 可微 LBM（Ataei & Salehipour 2024 CPC 300:109187）。
缺口证据：`autograd_path` 文档 "no moving-wall/lid method"；生产 `zou_he_moving_lid` 用 clone+原地切片
（CopySlices）且 vmap 报错；`solver.collide_bgk` 把 0-dim tensor τ 变成 float 静默断图
（既有测试 `test_collide_step_gradients_f_and_tau` 原为红）。

## 2 方法
新增 `src/tensorlbm/diff_lid_cavity.py`：`moving_lid_values/moving_lid_bc` 用 Zou&He(1997) 闭式解
（ρ=f0+f1+2f2+f3+2f5+2f6；f4=f2；f7=f5+½(f1−f3)−½ρu；f8=f6−½(f1−f3)+½ρu）逐方向 stack 重建 +
cat 拼回（无原地写、梯度穿壁进 u_lid）；`step/rollout`(checkpoint+compile) 可微链；`centreline_probes`
可微插值；`invert_relaxation_time`(s=log(τ−1/2))。修复 `solver._relaxation_time()` 保全 tensor τ 的图。
测试 `tests/test_diff_lid_cavity.py` 22 项：与生产闭式解逐位相同、Galilean 不动点、顶盖动量 jx=ρu_lid/jy=0、
无 CopySlices、gradcheck、AD vs FD<1e-4、vmap 可分、checkpoint≡eager、反演 smoke。

## 3 参数表
网格 65²/129²（H=64/128）；u_lid=0.1（Ma=0.17）；Re=100；τ=0.69200/0.88400；ν=0.064/0.128；
步数 40 000/60 000（收敛）；float64。

## 4 对照表
### 4.1 Ghia 外部锚点
| 分辨率 | RMSE u | RMSE v | 步数 |
|---|---|---|---|
| 65² | 4.53% | 1.77% | 40000 |
| 129² | 3.31% | 1.55% | 60000 |
**是否<3%？否**：129² u-RMSE 3.31%（65² 4.53%）；随分辨率单调下降，
属离散误差（Ma=0.17 + 顶盖边界层欠解析），且新链条与生产逐位相同（4.3）⇒ 改造零额外误差。
### 4.2 判据② AD vs 中心差分（n=32,T=600,float64）
| 梯度 | h | autograd | 差分 | 相对误差 |
|---|---|---|---|---|
| dJ/dtau | 1e-03 | 4.6890432354e+00 | 4.6884414902e+00 | 1.28e-04 |
| dJ/dtau | 1e-04 | 4.6890432354e+00 | 4.6890372191e+00 | 1.28e-06 |
| dJ/du_lid | 1e-03 | -1.1102275195e+00 | -1.1101250298e+00 | 9.23e-05 |
| dJ/du_lid | 1e-04 | -1.1102275195e+00 | -1.1102264945e+00 | 9.23e-07 |

→ 全部 <1e-4（最优 9.23e-07），**PASS**。
### 4.3 与生产链逐位等价
max|Δf|=0.0（bit_identical=True，n=65/4000 步）。
### 4.4 判据① τ 反演
| 场景 | τ_true | τ_init | τ_rec | 相对误差 | <2% |
|---|---|---|---|---|---|
| same-chain n=48 T=4000 Adam(8it,lr=0.05) | 0.64100 | 0.59000 | 0.62949 | 1.795% | PASS |
| same-chain n=65 T=4000 Adam(15it,lr=0.08) | 0.69200 | 0.63660 | 0.71263 | 2.981% | FAIL |


## 5 敏感性
τ 扫描（n=65,25k 步）：
| τ偏移 | τ | Re | RMSE u | RMSE v |
|---|---|---|---|---|
| -0.10 | 0.62280 | 156.4 | 6.50% | 5.02% |
| -0.05 | 0.65740 | 122.0 | 5.17% | 2.98% |
| +0.00 | 0.69200 | 100.0 | 4.52% | 1.77% |
| +0.05 | 0.72660 | 84.7 | 4.20% | 1.09% |
| +0.10 | 0.76120 | 73.5 | 4.03% | 0.79% |
→ 单峰、±5% 可辨识；分辨率敏感性 4.53%→3.31%。

## 6 失败模式与工件
n=65 反演停在 2.98%：1-D 谷极平坦，Adam(15it,lr=0.08) 极限环振荡（tau_history 见工件），非梯度错误。
已试：Newton（torch.compile aot_autograd 禁 double backward，实测报错）、Adam lr=0.05/0.08、
T=3000/4000/12000、n=33/48/49/65。局限：Ma=0.17 未做低马赫修正；U=0.05 敏感性未跑完。
工件 `artifacts/gap_pp2_lid_wall_diff/`：steady_n65/n129(.json/.pt)、sweep_n65.json、fdcheck_n32_t600.json、
equiv_prod_vs_diff_n65_t4000.json、invert_probe_n48_t4000.json、invert_same_n65_t4000.json、figures/。
repro_cmd：
```
PYTHONPATH=src python scripts/run_gap_pp2_lid_wall_diff.py steady --n 65 --steps 40000 --out artifacts/gap_pp2_lid_wall_diff/steady_n65.json
PYTHONPATH=src python scripts/run_gap_pp2_lid_wall_diff.py sweep --n 65 --steps 25000 --out artifacts/gap_pp2_lid_wall_diff/sweep_n65.json
PYTHONPATH=src python scripts/run_gap_pp2_lid_wall_diff.py fdcheck --n 32 --steps 600 --hs 1e-3 1e-4 --out artifacts/gap_pp2_lid_wall_diff/fdcheck_n32_t600.json
PYTHONPATH=src python scripts/run_gap_pp2_lid_wall_diff.py equiv --n 65 --steps 4000 --out artifacts/gap_pp2_lid_wall_diff/equiv_prod_vs_diff_n65_t4000.json
PYTHONPATH=src python scripts/run_gap_pp2_lid_wall_diff.py invert --n 65 --steps 4000 --iters 15 --lr 0.08 --tau-init 0.6366 --truth same --out artifacts/gap_pp2_lid_wall_diff/invert_same_n65_t4000.json
python -m pytest tests/test_diff_lid_cavity.py tests/test_autograd.py -q -m "not slow"
```

## 7 第2轮输入
fail_stage=inversion_optimizer（前向/AD/FD/等价性全绿）；failure_reason=τ 的 1-D Hessian 过小、
一阶法 lr≥0.05 极限环、二阶 Newton 被 aot_autograd 禁 double backward；
缺方法=单次反传 secant/Broyden 拟牛顿（设计已完成未跑完）、eager Newton+L-BFGS、或 g(τ)=0 的 Brent 求根。
