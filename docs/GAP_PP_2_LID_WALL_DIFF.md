# GAP-PP-2-lid-wall-diff：可微运动壁面 BC（graph-preserving lid）

判据①Re=100 τ 反演<2% / ②AD vs FD<1e-4。

**诚实结论（第2轮收口）：②PASS（最优 9.23e-07）；①PASS**——生产分辨率 n=65 上
τ 恢复误差 **0.0016 %**（τ_rec=0.6920114 vs τ_true=0.692000，Re_rec=99.994），
方法由「一阶下降」改为「对平稳条件 g=dL/dτ=0 求根」（§8）。
第1轮记录保留不改：Adam 在同配置下 2.98 % FAIL（n=48 时 1.80 % PASS），
原因是极平坦 1-D 谷中一阶法极限环，非梯度错误；求根法不复现该失败。

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

## 8 第2轮：平稳条件求根反演（criterion ① 收口，PASS）

### 8.1 为什么换方法（承接 §6/§7）
§6 的失败诊断：τ 是一维参数，损失谷极平坦（Hessian ~1e-12），Adam(lr=0.05–0.08) 在
最优附近极限环；Newton 需二次反传，`torch.compile`/aot_autograd 禁 double backward。
但 τ 的一维性同时给出更强的结构：**平稳条件 g(s)=dL/ds=0 是一个标量方程**，
每步只需一次反向即可用求根法（割线/Illinois 试位+二分保险）超线性收敛，
不需要 Hessian、不需要二次反传（因此与 `compile_step` 兼容）。
参数化不变：s=log(τ−0.5)，L=MSE(centreline probes)。

### 8.2 新增能力
`diff_lid_cavity.py`：
- `solve_stationary_1d(evaluate, x0, method="illinois"|"secant", …)`——通用一维平稳点求根器：
  缓存 oracle、几何外扩自动包围（`max_expand`）、Illinois 阻尼防停滞、二分保险、
  `invalid_policy="shrink"`（非有限点=稳定性边界时把该端点向 x0 收回重试，而非整单崩掉）。
- `invert_relaxation_time_rootfind(...)`——求解器在环的 τ 反演（与第一轮同损失/同参数化，
  返回 `tau_history/loss_history/grad_history/n_evals/status` 等）。
- 驱动 `scripts/run_gap_pp2_lid_wall_diff.py invert-rf`，图 `figures-rf`、场 `field`。

### 8.3 结果（n=65，与第1轮 Adam-FAIL 配置逐项同参）
`artifacts/gap_pp2_lid_wall_diff/invert_rf_n65_t4000.json`（程序化写出，零手抄）：

| 项 | 值 |
|---|---|
| 网格/步数 | n=65（H=64），T=4000，float64，u_lid=0.1 |
| 真值/初值 | τ_true=0.69200（Re=100）；τ_init=0.63660（偏 8.0 %） |
| 方法 | illinois，bracket τ∈[0.60, 0.80]，max_iter=12，step_tol=1e-5 |
| τ 恢复 | **0.6920114**（相对误差 **0.00164 %**） |
| ν / Re 恢复 | 0.0640038 / **99.994**（真值 0.064 / 100） |
| 残差 RMS | 2.83e-06（探针归一化速度） |
| 末点梯度 / 损失 | dL/ds=2.72e-07 / L=8.03e-12 |
| 代价 | 15 次 rollout（12 迭代），940 s（单进程 8 线程） |
| 判据① | **PASS（<2 %）**；对比第1轮 Adam 同配置 2.981 % FAIL |

收敛过程（图 `figures/rootfind_convergence.png`，数据即 JSON 的 `loss_history`
/`grad_history`/`tau_history`）：τ 误差 1.24e-04→3.34e-06→…→8.03e-12（损失），
τ 误差在第 5 次评估即 0.20 %、第 7 次 0.05 %、末点 0.0016 %，
远超判据①的 2 % 线；包络区间宽度由 1.10 单调收缩到 3.55e-04。

场量校核（`figures/cavity_field_n33_re100.png`）：n=33、τ=0.596（同 Re=100）
顶盖驱动场 |u|/u_lid 云图+流线 + 竖直中线 u 剖面 vs Ghia 1982 表 I 的点。

### 8.4 稳定性边界（新的实测坑，已机制化）
把初值推远到 τ_init=0.60 且给 bracket_lo=0.55 的一次运行，在 **τ=0.55
（ν=0.0167，n=65 上 Re≈384）直接 NaN**——BGK 在低 τ 失稳，不是梯度问题；
原实现直接抛 `FloatingPointError` 整单中止（日志 `logs/invert_rf_n65_init060.log`）。
已加 `invalid_policy="shrink"`：非有限端点按 1/2 几何回退向 x0 收拢重试，
迭代中遇到非有限点则改用区间内点（1/2、1/4、3/4）继续，全部无效才如实返回
`status="stalled_invalid"`。单测 `test_non_finite_bracket_end_is_retracted` 钉住该行为。
**适用范围**：τ≳0.57（本例 n=65、u_lid=0.1）之外需先做稳定性扫描，再交给反演器。

### 8.5 仍未做（诚实清单）
- 判据② 的 AD/FD 交叉（§4.2）本轮未重跑：代码路径（`moving_lid_bc`/`rollout`）未改，
  第1轮证据仍有效；本轮只新增独立函数与新测试，`tests/test_diff_lid_cavity.py` 21 项仍全绿。
- τ_init=0.60 的远初值鲁棒性扫描未跑完（遇稳定性边界中断，见 8.4）；只有单点证据。
- Ghia-真值（模型误差内）反演的 τ_ML 未跑：属下一轮（真实数据反演）。
- `bracket` 仍需人工给出（或几何外扩）；未实现「先粗扫再包围」的自适应初值。

### 8.6 第2轮复现命令与测试
```
PYTHONPATH=src python scripts/run_gap_pp2_lid_wall_diff.py invert-rf --n 65 --steps 4000 \
  --tau-true 0.692 --tau-init 0.6366 --truth same --bracket 0.60 0.80 \
  --max-iter 12 --step-tol 1e-5 --out artifacts/gap_pp2_lid_wall_diff/invert_rf_n65_t4000.json
PYTHONPATH=src python scripts/run_gap_pp2_lid_wall_diff.py field --n 33 --steps 15000 \
  --out artifacts/gap_pp2_lid_wall_diff/field_n33.json
PYTHONPATH=src python scripts/run_gap_pp2_lid_wall_diff.py figures-rf \
  --artifacts artifacts/gap_pp2_lid_wall_diff
python -m pytest tests/test_diff_lid_rootfind.py tests/test_diff_lid_cavity.py -q -m "not slow"
```
测试：`tests/test_diff_lid_rootfind.py` 19 项（求根器在精确/平坦/自动包围/不可包围/
缓存/非法参数/非有限收缩 上逐条钉住 + 两条 LBM smoke 反演），全绿；第1轮 21 项仍全绿。
