# hadamard_rybczynski_droplet（pending）

**状态：❌ 未达标（模块能力）——基准在当前库上不可运行：任何 3D 多相配置的黏度比 λ 都锁死等于密度比（单 τ），预注册 λ 阶梯退化（λ=1 ⇔ 零密度对比 ⇔ 零浮力 ⇔ U_t=0 不可测），形式化 Phase-2 阶梯从未启动；判死证据 = 动态排除 E1–E6 + 15 案包络 0 案存活 + 静态分支归档。修复路径已由后续相场重建弧落地（Wave-10 M1，PR #323）。**

## 物理问题

Hadamard–Rybczynski 液滴：蠕变流（Re≪1）下浮力驱动黏性液滴的终端速度解析解 U_t = (2/3)·a²·g·(ρ_o−ρ_i)·(1+λ)/(μ_o·(2+3λ))，λ=μ_i/μ_o。极限检验：λ→∞ 回到 Stokes 球（系数 2/9）、λ→0 为清洁气泡（系数 1/3，即 1.5× 刚性球），drag 形式（Pigeonneau 1998）与 terminal 形式（Clift, Grace & Weber 2005）由力平衡闭合。该案例对库的价值：给 3D 多相模块一个带内部黏性耗散、可变 λ 的直接可观测量（终端速度 vs 解析）。库内 λ=ρ 锁的格上形式：U_t = (2/3)·a²·g·(1−r²)/(ν·(2+3r))，r=ρ_i/ρ_o。设计约束（等效原理，预注册）：周期域 + 逐质量重力 = 自由落体、无相对运动，浮力基准必须壁面（动量汇）——W4-A 验证的 tank 组合（channel∪tank 全六面 + full-way bounce-back）被探针判无辜。

## 计算结果现状

基准本体未运行（模块能力 FAIL，Phase-1 可行性探针判死，Phase-2 形式化阶梯与门从未启动）。以下为判决证据的机器数字。

### 参考锁定（Phase 0 手算例，机器复算）

| 量 | 值 |
|---|---|
| r = λ | 0.5 |
| a | 12.0 |
| ν | 0.16666666666666666 |
| g | 8.1e-06 |
| U_t（prereg 手算锁定） | 0.0009997714285714286 |
| U_t（机器复算） | 0.0009997714285714288 |
| 相对偏差 | 2.220446049250313e-16（门 1e-12 PASS） |
| Re（prereg 印出 / 复算） | 0.14396685714285715 / 0.14396708571428576（末位印刷差，不影响 U_t 检验） |

### 模块排除动态证据（E1–E6）

| 证据 | 模块 | 机器数字 | 判 |
|---|---|---|---|
| E1 | Color-Gradient 3D（r=0.5 初始化） | 总密度比 1.028876543045044 → 1.0024418830871582（步 100→2000） | 声学坍缩 |
| E2 | SC-MCMP 3D D3Q19，G=+2.5 | NaN@步 50（G=1.0 档同样 NaN@步 50，guo 开关无关） | +G 吸引（混合）符号 bug |
| E3 | SC-MCMP 3D D3Q19，G=−2.5 | 压碎反转：环境组分灌入核内、r1_ctr 反转跌破 1（result.json E3 切片；其嵌套键含点号，数字以同病 D3Q27 复查档为准） | 压碎反转 |
| E4 | Allen-Cahn 3D | 少数相液滴在步 25→1000 完全溶解（W∈{2,4,8}×σ∈{0,0.01} 全部档） | 相场不守恒 |
| E5 | MCMP D3Q27，G=+2.5 | NaN@步 250（尽责复查档） | 同 E2 |
| E6 | MCMP D3Q27，G=-2.5 | 压碎反转：步 250 时 r1_ctr 0.5874204635620117、核内组分 2 占 0.3385912775993347 | 压碎反转 |

### 15 案包络（对比度 1.05–2.1 × G∈[−2.5,+1.5]，probe/p0b_envelope.json）

| 统计 | 值 |
|---|---|
| NaN 案数 | 1 |
| 混合/反转案数 | 14 |
| 持续保持对比度的液滴案数 | 0 |

代表行：对比度 0.9/1.1、G=-1.0 → mixed/inverted（末点 max|u| 0.027280136942863464）；对比度 0.9/1.1、G=−2.5 → NaN@步 250。G 窗口（probe/p0b_gsweep）：非对称初始化在 G≥0.9 即 NaN@步 25。

### 静态排除分支（既有归档复核，未重跑）

| 分支 | 机器数字 | 来源 |
|---|---|---|
| SCMP 3D（单组分伪势） | λ 由共存态锁死：ρ_l=1.957、ρ_v=0.1596（λ=ρ_l/ρ_v≈12.3）；伪电流 max\|u\| 0.13663936654726663–0.1394344319899877 ≫ U_t~1e-3（手算例 0.0009997714285714286） | verified/laplace_droplet |
| SC-MCMP 3D 不等 τ（SC 族唯一 λ≠ρ 路径） | u_eq=self 反收敛 max_rel_err 0.26821538643505377（H64）/ 0.33173655541170183（H128） | pending/two_phase_poiseuille（W5-A） |
| phasefield/ 自由能包 | λ≡1（Boussinesq 单密度），无浮力对比度 | 源码 |
| 自由面家族（dam_break 等）/ multiphase.py | 单流体+flag 无黏性液滴内部；2D only | 源码 |

### 设计约束探针（壁面无辜，probe/p0b_diag.json）

G=5.0 时：周期域 NaN@步 20、加壁 NaN@步 5（NaN 与壁无关，G 符号才是根因）；G=0.0 + 壁稳定，伪速度 0.078182 → 0.000569（160 采样点衰减）。

## 不达标清单

| 门 | 冻结条款 | 实测 | 判 |
|---|---|---|---|
| 共性模块入口 | 库 only 入口可表达工况 | 0 个 3D 模块能表示 λ≠1 的浮力黏性液滴（E1–E6 + 15 案包络 0 存活 + 静态分支） | ✗ |
| 直接观测量 ≤3% | U_t 对解析（锁定 Re_t=0.05） | Phase-2 阶梯从未启动（可行性探针判死） | ✗（未达） |
| ≥2 档网格单调 | D 阶梯 | 未运行 | ✗（未达） |
| 总判 | — | FAIL——模块能力（基准不可运行，非数值未达标） | 机器判定 |

## 根因/诊断

1. **headline：λ 与 ρ 在库内不可分离**。每个可用 3D 配置都锁 λ=ρ（单 τ ⇒ 均匀 ν ⇒ μ=ρ_tot·ν）；预注册 λ 阶梯 {0.1,1,10} 退化为 λ=1 ⇔ 零密度对比 ⇔ 零浮力 ⇔ U_t=0 不可测，已按预注册条款替换为 r∈{0.5,2}（主）——但连这个退化阶梯也跑不起来（下条）。
2. **无 3D 两相模块能维持总密度对比液滴（G3）**：Color-Gradient 声学坍缩（单组分型 EOS p=ρcs² 撑不住界面压差，E1 比值 100→2000 步内坍到 ~1）；SC-MCMP 在所有已探 G/对比度下被压力压碎反转（E3/E6 + 15 案包络 0 存活）。这杀死的不只是 HR，而是整个 3D 浮力液滴/气泡基准类。
3. **multiphase3d G_12 符号 bug（G1）**：docstring 与 StaticDroplet3DConfig 校验要求 G_12>0 相分离，实现却使 +G 吸引（混合、+2.5 NaN），分离需 G<0；已验证默认 G=0.9 产出混合态使库自带 sigma_eff 测量空洞；D3Q19 与 D3Q27 两处同款文案 bug。Wave-10 M2（MCMP3D G1）已修复，归档待办。
4. **allen_cahn_lbm 是未验证孤儿模块（G4）**：唯一带独立 ρ_h/ρ_l 与 ν_h/ν_l（λ 解耦）的模块，但相场动力学不成立——少数相溶解（E4，W×σ 敏感性全灭）、σ 力无曲率项、碰撞不跳过固体、相方程不守恒、M/τ_phi 硬编码；修复路径 = Fakhari/Geier 全方案。
5. **组合陷阱（G5）**：bounce_back_cells_3d 硬编码 19 方向 OPPOSITE 表（27-q 场会错；boundaries_d3q27.bounce_back_cells_27 存在）。
6. **等效原理（设计约束，非缺陷）**：浮力基准必须壁面；W4-A 验证的 tank 组合被探针判无辜（上表），NaN 根因在 G 符号不在壁。

## 晋级路径

1. **已落地**：相场重建弧（R2 → Wave-10 M1）以 PR #323 合入（verified/laplace_cac_phasefield）——Fakhari/Geier 型 CAC 方案，λ 与 ρ 解耦是其设计目标之一。HR 复活 = 在 CAC 模块上重预注册（r∈{0.5,2} 主、{0.25,4} 副，锁定 Re_t=0.05，g∝1/D³·(2+3r)/(1−r²) 逐档计算），壁面 tank 组合 + 有限域 Ladenburg–Faxén 归因链（诊断证据，不作门）。
2. G1（G_12 符号）已由 Wave-10 M2 修复，待其归档后本案例 E2/E5 类 NaN 证据对应更新。
3. 归档 PR 将 pending/hadamard_rybczynski_droplet 移 verified/ 仅在新模块跑出机器判定后发生；本档机器判定已如实写 FAIL（模块能力）。

## 运行方式

基准本体不可运行（无模块入口）。判决证据可再生的生成器（fp32，GPU 或 CPU）：

```bash
cd /nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet
V=/nfs/wangxi/venvs/tensorlbm/bin/python
$V probe/p0_static.py          # CG 坍缩 + 首轮 MCMP NaN 扫描
$V probe/p0b_scan.py           # G>=1 非对称 NaN 矩阵
$V probe/p0b_diag.py           # 壁面无辜诊断
$V probe/p0b_gsweep.py         # G 窗口扫描
$V probe/p0b_envelope.py       # 15 案对比度×G 包络
$V probe/p_final_diligence.py  # AC W/σ 敏感性 + D27 反转
$V out/evidence.py             # 规范证据切片 → out/evidence.json → result.json
```

工件：result.json（机器切片，含参考锁定/手算例复算/模块排除/包络/库缺陷 G1–G5/结局）、prereg.md（冻结预注册 + md5 边车）、NOTES.md（判定记录）、out/evidence.json、probe/*.json。完整性注记：probe/p1_ac.{log,json} 被一次意外的并发双启损坏；AC 结论由 out/evidence.json E4 与 p_final_diligence.json 的干净单进程运行独立重建。

<!-- PROVENANCE [{"v":0.5,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"reference_lock.hand_example.r"},{"v":12.0,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"reference_lock.hand_example.a"},{"v":0.16666666666666666,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"reference_lock.hand_example.nu"},{"v":8.1e-06,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"reference_lock.hand_example.g"},{"v":0.0009997714285714286,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"reference_lock.hand_example.ut"},{"v":0.14396685714285715,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"reference_lock.hand_example.re"},{"v":0.0009997714285714288,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"reference_lock.hand_example.U_t_recomputed"},{"v":0.14396708571428576,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"reference_lock.hand_example.Re_recomputed"},{"v":2.220446049250313e-16,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"reference_lock.hand_example_recomputed_rel_dev"},{"v":1.028876543045044,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"module_exclusion.E1_color_gradient_3d_contrast_collapse.init_ratio_total"},{"v":1.0024418830871582,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"module_exclusion.E1_color_gradient_3d_contrast_collapse.final_ratio_total"},{"v":2.5,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p0b_scan.json","k":"[2].G"},{"v":50,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p0b_scan.json","k":"[2].nan_step"},{"v":1.0,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p0b_scan.json","k":"[0].G"},{"v":50,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p0b_scan.json","k":"[0].nan_step"},{"v":2.5,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p_final_diligence.json","k":"d27[0].G"},{"v":250,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p_final_diligence.json","k":"d27[0].nan"},{"v":-2.5,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p_final_diligence.json","k":"d27[1].G"},{"v":250,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p_final_diligence.json","k":"d27[1].trace[0].step"},{"v":0.5874204635620117,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p_final_diligence.json","k":"d27[1].trace[0].r1_ctr"},{"v":0.3385912775993347,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p_final_diligence.json","k":"d27[1].trace[0].r2_ctr"},{"v":0.9,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p0b_gsweep.json","k":"[7].G"},{"v":25,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p0b_gsweep.json","k":"[7].nan_step"},{"v":1,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"module_exclusion.envelope_15_cases.nan_cases"},{"v":14,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"module_exclusion.envelope_15_cases.mixed_or_inverted"},{"v":0,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"module_exclusion.envelope_15_cases.sustained_contrast_droplet_cases"},{"v":0.027280136942863464,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"module_exclusion.envelope_15_cases.rows[0].end_maxu"},{"v":250,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"module_exclusion.envelope_15_cases.rows[2].nan_step"},{"v":-1.0,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/result.json","k":"module_exclusion.envelope_15_cases.rows[0].G"},{"v":25,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/out/evidence.json","k":"E4_ac3d_dissolution[0].step"},{"v":1000,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/out/evidence.json","k":"E4_ac3d_dissolution[39].step"},{"v":20,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p0b_diag.json","k":"[1].nan_step"},{"v":5,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p0b_diag.json","k":"[0].nan_step"},{"v":5.0,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p0b_diag.json","k":"[1].G"},{"v":0.0,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p0b_diag.json","k":"[4].G"},{"v":0.078182,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p0b_diag.json","k":"[4].max_u[0]"},{"v":0.000569,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/hadamard_rybczynski_droplet/probe/p0b_diag.json","k":"[4].max_u[159]"},{"v":1.957,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/laplace_droplet/result.json","k":"eos.discrete_coexistence.rho_l"},{"v":0.1596,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/laplace_droplet/result.json","k":"eos.discrete_coexistence.rho_v"},{"v":0.13663936654726663,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/laplace_droplet/result.json","k":"dims.3d.per_radius[0].max_u"},{"v":0.1394344319899877,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/laplace_droplet/result.json","k":"dims.3d.per_radius[2].max_u"},{"v":0.26821538643505377,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json","k":"part1.H64.max_rel_err"},{"v":0.33173655541170183,"f":"/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/two_phase_poiseuille/result.json","k":"part1.H128.max_rel_err"}] -->
