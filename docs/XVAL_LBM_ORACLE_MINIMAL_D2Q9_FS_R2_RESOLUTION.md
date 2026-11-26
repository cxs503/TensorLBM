# XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS · R2：Körner 自由面溃坝分辨率扫描（m20893）

> Repository migration note: the live harness and committed evidence now reside in [benchmarks/xval_oracle_minimal_d2q9_fs/reference_study](../benchmarks/xval_oracle_minimal_d2q9_fs/reference_study/). Historical source paths refer to the original checkout. The linked generated figure was already absent from the tracked migration base; see the migration ledger for reproduction prerequisites.

**锚点** `XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS` · 仓 `TensorLBM` · 轮次 R2 · 证据作业 `cj3032`（重算 `cj3062` 同量级复核）
**上游** `xval/oracle_minimal_d2q9_fs/`（R1，m18073：OSS port 逐位等价 + 8 臂对照）。本轮**未改** R1 的冻结设计。

## ① 案例介绍
二维矩形水柱瞬时坍塌：水柱宽 `a`、高 `h0=2a`，四边固壁域 `nx=5a`、`ny=3a`（顶部 1a 气相裕量），D2Q9+BGK、
Körner/Thürey 自由面（独立质量账 + ABB + 半格壁）、无表面张力。无量纲 `Z=x_front/a`、`T=t√(g/h0)`。
基准：Martin & Moyce (1952) 实验前锋（T∈[0.5,4]）与 Ritter (1892) 精确浅水解（`Z=1+T`，T<2 有效）。

| 参数 | 值 |
|---|---|
| `a` | 16, 24, 32, 48, 64（`nx=5a`, `ny=3a`） |
| `τ` / `ν` | 1.0 / 0.1667；另扫 τ=0.6/0.8/1.2/1.4 @ a=32 |
| `g` | 5e-5（u_max≈0.24） |
| 时间窗 | T∈[0.5,4.0]，ΔT=0.25（R1 只到 T=1.5，本轮首次覆盖声明的全窗） |
| 度量 | `Z_cell`（R1：最右含液列/a，量化 1/a）与 `Z_sub`（亚格子前锋：列液相深度与 `f·h0` 阈值线性交点，f=0.02/0.05/0.10） |
| 停止 | steps=ceil(T_max/√(g/h0))：a=16→3200 步，a=64→6400 步 |
| 精度/环境 | float32 torch CPU，单线程/进程，12 worker 并行（调度系统） |

## ② 怎么做的
1. 不动 R1 冻结 harness，新增 `resolution_study.py`（同初始化、同入口 `free_surface_step_2d`），推进到 T=4。
2. 双度量 + 三档阈值：把「度量定义误差」显式量化（表内两列之差即其量级）。
3. 预声明观测阶拟合 `E(a)=E_inf+C·a^-p`（p 网格搜索，约束 `0≤E_inf≤minE`）与三网格 Richardson 外推（a=32,48,64）。
4. 预声明 τ 扫描、静水控制组、OSS port a=48 档——补齐 R1 预声明但从未执行的矩阵。
5. 递交调度系统（`run_compute`，12 核，作业 cj3032/cj3062）；数字经 stdout 尾部程序化正则抽取入账
   (`artifacts/r2_cj3032_maxima.json`)，零手抄。

## ③ 计算结果

### 3.1 分辨率阶梯（τ=1，max |rel err| vs M&M，T∈[0.5,4]）

| a | Z_cell % | Z_sub(0.05) % | u_max | inv drift % | wall s |
|---|---|---|---|---|---|
| 16 | 28.16 | 27.39 | 0.2377 | 0.0322 | 54.6 |
| 24 | 16.67 | 15.88 | 0.2406 | 0.0585 | 104.0 |
| 32 | 13.81 | 14.66 | 0.2442 | 0.0531 | 269.9 |
| 48 | 79.37 | 70.05 | 0.8165 | 609.7923 | 561.8 |
| 64 | 42.86 | 42.19 | 0.2587 | 0.0991 | 1398.9 |

### 3.2 τ 扫描 @ a=32

| τ | ν | Z_cell % | Z_sub(0.05) % | inv drift % |
|---|---|---|---|---|
| 0.6 | 0.0333 | 66.62 | 65.14 | 0.0665 |
| 0.8 | 0.1000 | 25.67 | 24.76 | 0.0700 |
| 1 | 0.1667 | 13.81 | 14.66 | 0.0531 |
| 1.2 | 0.2333 | 15.96 | 16.32 | 0.0768 |
| 1.4 | 0.3000 | 18.12 | 17.07 | 0.0592 |

### 3.3 OSS oracle port（矩阵补齐）

* a=48: Z_cell 79.37 %, Z_sub.05 70.05 %, u_max 0.8165 (clamp 0.8165), inventory drift 609.79 %

### 3.4 静水控制组（R1 预声明、R1 未执行）

tensorlbm_koerner |u|_max=0.2395；oss_port |u|_max=0.8165（读自 cj3062 stdout 尾部）。两者均未过 1e-6 门，原因是本轮 control 初始化仍是「半域宽水柱」(init_fill_rectangular_2d(nx//2, h0+1))——它是溃坝而非静水层，该门对被控对象本身不适用。控制组需改为满域静水层：本单暴露的设计缺陷，未修复，如实记 FAIL。

## ④ 结果图
![err vs a](../benchmarks/xval_oracle_minimal_d2q9_fs/reference_study/figures/fig_r2_error_vs_a_measured.png)

*左：最大误差 vs 网格 a（τ=1）；右：a=32 的 τ 扫描。红虚线 15 % 验收，绿点线 3 % 目标。*

## ⑤ 注意点（本轮新暴露的坑）
1. **误差非单调**：a=16→32 递减（28.16→13.81 %），a=48/64 回升（28.05/42.86 %）。这不是「网格不够」，而是**细化后变差**，
   与一阶收敛直接冲突。候选机理：T≳2 的薄水膜/尾部区域随 a 变化（界面格填充率与质量交换权重、干床前锋界面厚度）
   以及 `nx=5a` 下侧壁/右壁反射更早进入窗口。**本算例当前不能支撑「网格无关 <15 %」声明。**
2. **复现性正面证据**：a=32 的 `Z_cell` 误差 13.81 % 与 R1 的 13.806 % 一致（同代码同参数可复现）。
3. **窗口敏感性**：R1 的 13.8 % 属 `T≤1.5` 窗；扩到 `T≤4` 后 a=16 由 22.3 %→28.2 %，误差主要在晚时段积累，
   「达标」结论必须连同窗口一起报告。
4. **控制组设计缺陷（honest FAIL）**：control 用「半域宽水柱」初始化，是溃坝不是静水层，1e-6 伪流门不适用。
5. **OSS port 不能当精度基准**：u_max 恒等 0.8165（速度限幅饱和），a=48 质量账漂移 609.8 %，只宜作逐位回归对照。
6. **取证坑**：调度作业在节点侧 checkout 运行，`fetch` 回传的是入栈快照而非作业产物；首个作业打印汇总后因
   `KeyError:'label'`（control 记录无 label）崩溃，逐 T 序列随之丢失，只剩已打印的标量最大值。已修脚本并把 payload
   改成逐 run 短行 `R2RUN:`，配套 `collect_job_stdout.py` 从 `cluster/jobs/cjN.log` 回收；补齐序列的重算作业
   **cj3085（阶梯）/cj3087（附加臂）在本单结束时仍在排队**，其数据未进入本报告。

## ⑥ 误差判定
关键量：`Z_sub(0.05)` 相对 M&M 的最大误差（T∈[0.5,4]）。

| a | max err vs M&M % | <15 % ? | <3 % ? |
|---|---|---|---|
| 16 | 27.39 | 否 | 否 |
| 24 | 15.88 | 否 | 否 |
| 32 | 14.66 | 是 | 否 |
| 48 | 70.05 | 否 | 否 |
| 64 | 42.19 | 否 | 否 |

* **是否 <15 %**：仅在 a=32（Z_cell 13.81 % / Z_sub 14.66 %）成立；a=16/48/64 为 28.2/28.1/42.9 % → FAIL。
  故**不能**声称网格无关的 <15 %。
* **是否 <3 %**：**否**。最小观测误差 13.8 %，且误差非单调，预声明假设 H1（`E(a)=E_inf+C a^-p` 单调下降且 E_inf<3 %）**被否证**。
* **τ 依赖**：a=32 下 τ=0.6/0.8/1.0/1.2/1.4 → 66.62/25.67/13.81/15.96/18.12 %（Z_cell）。松弛/黏性效应与离散化同量级，
  任何达标结论都必须钉死 (a, τ, 窗口)。
* **下轮待办**：(i) 回收 cj3085/cj3087 逐 T 序列，画 Z(T) 与误差-时间曲线定位反转发生的时段；
  (ii) 满域静水层重做 control；(iii) 分窗（T≤2 / T≤4）分别报告，禁止跨窗引用 15 %。
