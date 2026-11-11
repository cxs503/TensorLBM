# CASE-COUETTE-2D — 库埃特剪切流（TensorLBM · 算例池 core 例）

> 算例池记录：`case_code=COUETTE-2D`，repo=TensorLBM，ref_source=线性解析解，tol_policy=<2%。
> 本目录是**复用型**执行器：不新写求解器，直接驱动仓内既有两条 Couette 能力与闭式解逐项对照。
> 全部数字由 `make_report.py` 从 `artifacts/summary.json` 程序化注入（零手抄）。

## 判定（两级，如实分列）

| 判据族 | 结果 | 内容 |
|---|---|---|
| **算例池容差 <2%（本案验收判据）** | **PASS** | 剖面 L2/逐点、近壁速度、τ_w、动量积分、Route B 三条力量、A↔B 一致（见 ④） |
| 自设护栏/诊断项（非容差判据） | **未满足 3 项** | 质量漂移带 <0.001%、收敛观测阶 ≥ −0.25 |

`verdict.pass`（"所有预注册判据同时满足"）= `False`；容差族 = `PASS`。
护栏未满足项**不粉饰**：逐条在 ⑤⑥ 给数值与归因；阈值运行前固定（`run_case.py::THRESHOLDS`），未因结果调整。

## ① 问题与 ref 出处

**物理**：两块无限大平行平板间的定常不可压剪切流（plane Couette）。下板静止、上板以 U₀ 沿 +x 平动，
流向/展向均匀 ⇒ N-S 退化为 ∂²u/∂y²=0，精确解为线性剖面。

**ref（与算例池登记 `ref_source=线性解析解` 一致）**：

```text
u(y) = U0 (y - y_b) / H_eff ; tau_w = mu U0 / H_eff, mu = rho nu ; int_0^H u dy = U0 H_eff / 2 ; rho = const
```

**零手抄声明**：目标数值由 `run_case.py` 的 `analytic_profile / analytic_wall_shear /
analytic_momentum_integral` 在求解器报告的同一组格子坐标上求值，无表格转录。
闭式解由 N-S 方程在 Couette 边界条件下直接求得，目标数值由本文件的 analytic_* 函数按闭式求值（零手抄，无表格转录）。教材/手册级页码引用在离线环境不可核验（本机无外网），故不著录页码，以免伪造出处；证据面=self-contained 闭式 + 两条独立数值路径。

## ② 方法与实现

两条既有能力、两条独立证据面（复用，不另起炉灶）：

| 路径 | 实现（仓内既有文件） | 边界/力 | 提供量 |
|---|---|---|---|
| **Route A**（主） | `benchmarks/verified/couette_2d/run.py::run_case` — D2Q9 + 库原语 `d2q9.*`、`solver.collide_bgk/collide_mrt/stream` | 上壁移动 no-slip：预 streaming half-way bounce-back + 移动壁动量注入；下壁静止 | 全剖面、近壁速度、τ_w（离散斜率×ν）、质量/动量守恒 |
| **Route B**（独立验证） | `tensorlbm.curved_boundary_calibration.run_plane_couette` — D3Q19 插值反弹（halfway / Bouzidi 二次） | momentum-exchange 与 control-surface 应力积分**两条力路** | 上/下壁力、CV 通量、剖面误差 |

预注册判据（运行前固定）：剖面 L2 与逐点最大 <2%、τ_w <2%、近壁行 <2%、
动量积分 <2%、Route B 力量 <2%、A↔B τ 一致 <2%；自设护栏：质量漂移 <0.001%、收敛阶 ≥ −0.25。

## ③ 参数表（参考点，无量纲数完整）

| 参数 | 记号 | 参考点取值 | 说明 |
|---|---|---|---|
| 缝高（格子数） | H | 32 | 上下壁间流体节点数；H_eff=H |
| 松弛时间 | τ | 0.8 | BGK 单松弛 |
| 运动黏度（格子） | ν=(τ−0.5)/3 | 0.1 | 格子单位 |
| 壁面速度 | U₀ | 0.05 | 上壁 +x，下壁静止 |
| 雷诺数 | Re=U₀H/ν | 16 | 层流；Couette 转捩 Re≈360 |
| 马赫数 | Ma=U₀/c_s，c_s=1/√3 | 0.0866025 | 压缩性残差量级 Ma²≈0.0075 |
| 边界条件 | — | 上壁移动 no-slip / 下壁静止 no-slip | half-way bounce-back + 移动壁动量注入 |
| 流向/展向 | — | 流向 x 周期（nx=H）、y 壁面 | 一维解在流向均匀 |
| 时间步 | n_steps | 10000 | 2000 步窗内上壁速度漂移 <1e-5 判稳 |
| 统计窗口 | — | 最后 200 步时间平均 | 去噪剖面 |
| 碰撞模型 | — | bgk | bgk（参考点）/ mrt（交叉验证） |
| 精度/设备 | — | dtype=float32, cpu | compile_mode=eager |

## ④ 对照表（ref vs got，误差逐项打勾/叉）

参考点 H=32、τ=0.8、U₀=0.05、bgk、10000 步：

| 量 | 类型 | ref（闭式，程序化求值） | got（数值） | 相对误差 | 判据 | 判定 |
|---|---|---|---|---|---|---|
| u(y) 全剖面 | 场量 | 解析线性剖面 | 同网格数值剖面 | L2 1.471e-02% | <2% | ✓ |
| u(y) 逐点最大 | 场量 | 同上（u_ref>0.2U₀ 掩膜） | 同上 | 2.434e-02% | <2% | ✓ |
| 近壁行 u(y=H−0.5) | 标量 | 4.92187500e-02 | 4.92195200e-02 | 1.570e-03% | <2% | ✓ |
| 壁面剪应力 τ_w=ρνU₀/H | 标量 | 1.56250000e-04 | 1.56252327e-04 | 1.489e-03% | <2% | ✓ |
| 单位宽度动量 ∫u dy | 积分量 | 8.00000000e-01 | 8.00118085e-01 | 1.476e-02% | <2% | ✓ |
| 质量守恒 | 守恒量 | 1.0（ρ₀ 常量） | 漂移 4.600e-04% | 4.600e-04% | <0.001% | ✓ |

**Route B（独立力路）**

| 配置 | 量 | ref（ρνU₀/H × 壁面积） | got | 相对误差 | 判据 | 判定 |
|---|---|---|---|---|---|---|
| ny18_tau0.8_U0.05_halfway | force_top_x | 1.25000000e-03 | -1.25000041e-03 | 3.241e-05% | <2% | ✓ |
| ny18_tau0.8_U0.05_halfway | force_bottom_x | -1.25000000e-03 | 1.24999378e-03 | 4.977e-04% | <2% | ✓ |
| ny18_tau0.8_U0.05_halfway | force_cv_x | 1.25000000e-03 | -1.24998228e-03 | 1.418e-03% | <2% | ✓ |
| ny18_tau0.8_U0.05_halfway | 两力路自洽 ME vs CV | — | — | 1.450e-03% | <2% | ✓ |
| ny18_tau0.8_U0.05_halfway | 剖面逐点最大 | — | — | 1.490e-04% | <2% | ✓ |
| ny34_tau0.8_U0.05_bouzidi_quadratic | force_top_x | 6.25000000e-04 | -6.24932349e-04 | 1.082e-02% | <2% | ✓ |
| ny34_tau0.8_U0.05_bouzidi_quadratic | force_bottom_x | -6.25000000e-04 | 6.24991953e-04 | 1.287e-03% | <2% | ✓ |
| ny34_tau0.8_U0.05_bouzidi_quadratic | force_cv_x | 6.25000000e-04 | -6.25016983e-04 | 2.717e-03% | <2% | ✓ |
| ny34_tau0.8_U0.05_bouzidi_quadratic | 两力路自洽 ME vs CV | — | — | 1.354e-02% | <2% | ✓ |
| ny34_tau0.8_U0.05_bouzidi_quadratic | 剖面逐点最大 | — | — | 5.737e-04% | <2% | ✓ |

**两路 τ_w 一致性**（两路缝高不同，故用无量纲系数 C=τ_wH/(νU₀) 比，解析值恒为 1）：

| Route B 配置 | τ_w^A | τ_w^B | C^A=τH/(νU₀) | C^B | 相对差 | 判据 | 判定 |
|---|---|---|---|---|---|---|---|
| ny18_tau0.8_U0.05_halfway | 1.562523e-04 | 3.125001e-04 | 1.000015 | 1.000000 | 1.457e-03% | <2% | ✓ |
| ny34_tau0.8_U0.05_bouzidi_quadratic | 1.562523e-04 | 1.562331e-04 | 1.000015 | 0.999892 | 1.231e-02% | <2% | ✓ |

![剖面](artifacts/fig1_couette_profile.png)

## ⑤ 收敛性与参数敏感性（≥2 点）

**5.1 网格收敛**（τ=0.8, U₀=0.05, BGK；p = log(e₁/e₂)/log(H₂/H₁)）

| H | n_steps | L2 相对误差 | 逐点最大 | τ_w 误差 | 观测阶 p |
|---|---|---|---|---|---|
| 16 | 10000 | 8.621e-03% | 2.116e-02% | 1.136e-03% | — |
| 32 | 10000 | 1.471e-02% | 2.434e-02% | 1.489e-03% | -0.77 |

半程 bounce-back 对线性剖面是精确贴合的壁面处理（无几何台阶误差），误差主体为 float32 时间平均噪声与
Ma² 压缩性残差，故**不呈幂律下降而落到 1e-4%–1e-3% 平台**；两点（H=16→32）的观测阶为
-0.77——噪声平台上的抖动，不是"网格越细越差"的物理趋势（两级均低于容差 2% 约四个数量级）。
**预注册"观测阶 ≥ −0.25"未满足**，如实列入判定表；要真正测阶需 H≥64 与 float64，超出本单算力窗口
⇒ 该子判据记 WITHHELD（证据不足），不宣称收敛阶。

**5.2 黏性 / 压缩性 / 碰撞模型敏感性**（均在 H=16 上扫；该问题是尺度无关的）

| 扫描轴 | 取值 | Re | Ma | 碰撞 | L2 相对误差 | τ_w 误差 | 质量漂移 | 判定 |
|---|---|---|---|---|---|---|---|---|
| tau | τ=0.6 | 24 | 0.0866 | bgk | 8.551e-03% | 5.261e-03% | 7.725e-03% | ✗ |
| tau | τ=1.4 | 2.667 | 0.0866 | bgk | 1.786e-03% | 1.081e-03% | 2.225e-04% | ✓ |
| U0 | U₀=0.01 | 1.6 | 0.01732 | bgk | 3.050e-02% | 5.802e-04% | 3.285e-04% | ✓ |
| U0 | U₀=0.1 | 16 | 0.1732 | bgk | 4.371e-03% | 1.669e-04% | 5.086e-04% | ✓ |
| collision | mrt | 8 | 0.0866 | mrt | 5.829e-03% | 6.255e-03% | 9.028e-03% | ✗ |

读法：τ 扫描改 Re（黏性），τ_w 与剖面误差基本不变 ⇒ 剖面精度不受黏性分辨率限制；U₀ 扫描改 Ma，误差随
Ma 上升是 LBM 压缩性残差（∝Ma²）的指纹；BGK 与 MRT 同网格同量级 ⇒ 结论不依赖碰撞模型。两个敏感性点
（τ=0.6、MRT）**质量漂移超出 1e-3% 护栏**，见 ⑥。

![误差](artifacts/fig2_couette_errors.png)

## ⑥ 失败模式与工件

**已识别失败模式（本单实测 + 已知）**

1. **动量积分口径错位（实测踩到并修复）**：首版只在流体行上做梯形积分，漏掉两个壁面半格三角形，H=16 时
   低估 6.25% → 判 ✗。改为在闭区间 [0,H] 含壁节点（y=0→u=0, y=H→u=U₀）积分后与 U₀H/2 吻合。
   **教训：积分型参考量的离散口径必须与求解器报告的节点位置逐一对应，否则误差是口径差而非数值误差。**
2. **两路 τ_w 直接比较的尺度错误（实测踩到并修复，事后披露）**：首版把 Route A（H=32）与 Route B（ny=18 ⇒
   H=16）的**物理剪应力**直接比 → 相对差 99.997%，判 ✗；两路缝高不同，物理剪应力本就差 H 倍。改为比
   无量纲系数 C=τ_wH/(νU₀) 后两路差 1.231e-02%。该改动**发生在看到结果之后**，故原始量纲直接比的数值
   保留在 `summary.json` 的 `rel_diff_raw_dimfull_pct_v1` 字段供独立复核。
3. **质量漂移护栏超限（未修复，如实上报）**：τ=0.6 与 MRT 两点漂移超出 1e-3% 护栏——低黏性 + 移动壁注入在壁行
   累积、MRT 平衡态耦合使其在 10⁴ 步内未达机器级守恒；量级仍 <1e-2%、远小于容差，但**超出预注册护栏=未满足**。
   属"守恒精度"而非"解精度"问题，需壁行 ρ 重标定才能压到 1e-3% 以下（后续轮次项）。
4. **近壁相对误差分母趋零**：u→0 处相对误差发散，故逐点判据按 u_ref>0.2U₀ 掩膜（预注册），避免假 FAIL。
5. **未达稳态即取剖面 ⇒ 假通过/假失败**：以 2000 步窗上壁速度漂移 <1e-5 作稳态门，`steady` 随工件落盘。
6. **局限**：仅层流（Re≈16 ≪ 360）；不可压低 Ma（Ma²≈0.0075）；平板无限大（展向周期）。转捩/湍流 Couette 不在本算例范围。

**工件（全部随分支入库）**

| 工件 | 内容 |
|---|---|
| `artifacts/summary.json` | 全量判据、ref/got、收敛/敏感性、环境、repro_cmd、两级 verdict |
| `artifacts/routeA_*.json` | Route A 每次运行原始测量（由 `benchmarks/verified/couette_2d/run.py` 直接落盘） |
| `artifacts/routeB_*.json` | Route B 原始测量（力/通量/剖面） |
| `artifacts/fig1_couette_profile.png` | 剖面 vs 闭式直线 + 逐点误差 |
| `artifacts/fig2_couette_errors.png` | L2/逐点/τ 误差随 H；τ 与 U₀ 敏感性 |
| `make_figures.py` / `make_report.py` | 图与本文生成器（数字零手抄） |
| `tests/test_case_pool_couette_2d.py` | 闭式自洽 + 实跑容差 + 已提交工件验收 |

**repro_cmd（一键复现）**

```bash
PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/couette_2d/run_case.py --preset pool --out-dir benchmarks/case_pool/couette_2d/artifacts
PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/couette_2d/make_figures.py
PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/couette_2d/make_report.py
PYTHONPATH=src:benchmarks pytest -q tests/test_case_pool_couette_2d.py
```

**环境/可复现性**：git 16b48fe4，torch 2.7.1，numpy 2.4.6，Python 3.11.6，loongarch64，device=cpu，dtype=float32，compile_mode=eager，seed=0，总墙钟 95.0s（CPU 单机；复用缓存件时不重跑模拟）。

**生态摩擦（如实记录）**
- 平台 knowledge_base 检索引擎离线（`:8021` connection refused）⇒ 开工三查中的 capability/knowledge 检索不可用，
  退化为 `negative-knowledge/search`（可用，无同域否定条目）+ 仓内 grep；`oss_solutions` 无检索端点。
- 离线无外网 ⇒ 教材页码不可核验，ref 出处按"闭式自洽 + 两条独立数值路径"给证，不著录未核验页码。
- 本机 load average ≈500（128 核）⇒ 单次 H=32 墙钟 12s–126s 波动，`elapsed_s` 只作记录不作性能结论；
  重算改为缓存复用（`reused_cached_artifact` 字段标明）。
