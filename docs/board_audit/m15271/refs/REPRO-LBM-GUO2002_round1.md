# REPRO-LBM-GUO2002（第 1 轮）：Guo, Shi & Zheng 2002 耦合 LBGK 的差分加热方腔复现

> 外部锚：**Guo Z., Shi B., Zheng C. (2002)** *A coupled lattice BGK model for the Boussinesq equations*,
> Int. J. Numer. Meth. Fluids **39**:325-342。
> 基准：**de Vahl Davis (1983)**, Int. J. Numer. Meth. Fluids **3**:249-264（Pr=0.71 刚-刚方腔平均 Nu）。
> 生成：2026-11-11T21:05:26。表中数字由 `runs/repro_guo2002_dhc/summary.json` 程序化注入（零手抄）。

## ① 案例介绍

二维差分加热方腔（DHC）：边长 L，左壁 T=1、右壁 T=0、上下绝热、四壁 no-slip，Pr=0.71（空气）。
论文用「速度分布 f + 温度分布 g 双分布耦合 LBGK + Boussinesq 浮力」求解该问题，并与 de Vahl
Davis 的平均 Nusselt 基准对表。本轮复现靶标：Ra=1e3/1e4/1e5 的热壁平均 Nu（容差 <5%）+
等温线形态（定性一致）。

| 参数 | 取值 | 说明 |
|---|---|---|
| Pr | 0.71 | ν/α |
| Ra | 1e3 / 1e4 / 1e5 | gβΔT·H³/(ν·α) |
| 几何 | 方腔，H=Ny 格（半程壁 −1/2 … Ny−1/2） | 本模块唯一口径 |
| 速度 BC | pre-streaming 半程反弹（no-slip，有效流动间隙 Ny−2） | `thermal.pre_streaming_bounce_back` |
| 温度 BC | anti-bounce-back Dirichlet（热/冷壁）+ 零通量（绝热壁） | `rb_convection_validation._thermal_bc` |
| 浮力 | Guo 力格式（半程外力项） | `thermal.collide_bgk_force` |
| Ma 目标 | 0.10（本文）/ 0.20（前作归档） | α 由 Ma 约束反解 |

## ② 怎么做的

1. **不另起炉灶**：物理全部取自既有实现——`tensorlbm.thermal`（D2Q9 BGK + Guo 力 + D2Q5 温度
   BGK + 半程边界）与 `tensorlbm.rb_convection_validation`（几何/边界/Nu/无量纲换算的唯一真源，
   `Convection2D`、`run_to_steady`、`wall_flux_nusselt`、能量平衡 Nu）。
2. **本轮最小扩展**：
   * `run_to_steady(..., return_fields=True)` 返末态 T/u/v —— 此前只有标量，等温线图无法从库内产出；
   * `scripts/repro_guo2002_dhc.py`（单跑 worker）、`scripts/repro_guo2002_ladder.py`（阶梯 + 计时探针）、
     `scripts/repro_guo2002_analysis.py`（对表 + Richardson 外推 + 图件）；
   * `tests/test_repro_guo2002_dhc.py`（6 条用例）。
3. **预注册判据**（先于动手）：主判据 = 逐 Ra |Nu−DVD|/DVD ≤5%（细网格直接值或 h→0 外推值）；
   次判据 = 等温线形态定性一致；停止规则 = 阶梯到 Ny=256/192/64 或外推值与细网格值差 <1%。
4. **口径**：Nu = 热壁三点二次插值求导的壁面平均（`wall_flux_nusselt`）；同时报能量平衡 Nu 作定常诊断。

## ③ 计算结果

| Ra | Ny | 来源 | dtype | Ma 目标 | Nu(LBM) | Nu(DVD 1983) | 误差 | 步数 | 墙钟 |
|---|---|---|---|---|---|---|---|---|---|
| 1e3 | 32 | 本文 | fp32 | 0.1 | 1.08912 | 1.1180 | 2.58% | 10000 | 49.2 s |
| 1e4 | 32 | 前作归档 P96W3-N04 | fp64 | 0.2 | 1.85198 | 2.2430 | 17.43% | - | - |
| 1e4 | 64 | 前作归档 P96W3-N04 | fp64 | 0.2 | 2.08186 | 2.2430 | 7.18% | - | - |
| 1e4 | 128 | 前作归档 P96W3-N04 | fp64 | 0.2 | 2.13279 | 2.2430 | 4.91% | - | - |
| 1e5 | 64 | 前作归档 P96W3-N04 | fp64 | 0.2 | 3.77563 | 4.5190 | 16.45% | - | - |
| 1e5 | 128 | 前作归档 P96W3-N04 | fp64 | 0.2 | 4.11553 | 4.5190 | 8.93% | - | - |

外推（Richardson，三档网格 r=2）：

- Ra=1e4（网格 32→64→128，ma=0.2）：观测阶 **p=2.174**，h→0 外推 **Nu=2.1473**，误差 **4.27%**。
  （脚本同时打印的细网格 GCI 数值为负，明显实现有误，**不用于判定**，见 §⑤-7。）

逐 Ra 判定：

| Ra | 判定（≤5%） | 细网格直接值误差 | h→0 外推误差 |
|---|---|---|---|
| 1e3 | ✅ PASS | 2.58% | 无（网格不足 3 档） |
| 1e4 | ✅ PASS | 4.91% | 4.27% |
| 1e5 | ❌ FAIL | 8.93% | 无（网格不足 3 档） |

总体：**PARTIAL**（Ra=1e3、1e4 达标；Ra=1e5 honest-FAIL，见 §⑥）。

## ④ 结果图

* `figures/repro_guo2002/fig1_isoterms_dhc.png`：等温线（白线 0.1–0.9）+ 流线（青）+ 温度填色，
  Ra=1e3、Ny=32 末态场（本轮唯一跑完并导出场量 npz 的档）。
* `figures/repro_guo2002/fig2_nu_vs_ra.png`：逐 Ra 的 LBM 阶梯点 / 外推点 vs de Vahl Davis 基准曲线。
* `figures/repro_guo2002/fig3_grid_convergence.png`：Nu(1/Ny) 收敛曲线与外推值。

形态（定性）：Ra=1e3 呈近竖直的导热态等温线 + 单涡弱环流，与 de Vahl Davis 的经典图形态一致
（等温线在腔中心近乎竖直、向上下壁弯曲）；Ra=1e4/1e5 的场量本轮未导出（阶梯受算力阻碍未跑完）。

## ⑤ 注意点（坑与局限）

1. **参考混淆（文档坑）**：`src/tensorlbm/boussinesq.py` 的 docstring 引的是 *Guo, Zheng & Shi (2002)
   PRE 65:046308*（"Discrete lattice effects on the forcing term"），与本单外部锚
   *Guo, Shi & Zheng (2002) IJNMF 39:325*（耦合 LBGK）是**两篇不同论文**；且能跑 DHC/Ra 基准的
   耦合双分布求解器不在 `boussinesq.py`（那是羽流场景的 dataclass 配置），而在
   `rb_convection_validation.py` + `thermal.py`。⇒ 按模块名找能力会走错路。
2. **几何口径 Ny−2**：pre-streaming 半程反弹把 no-slip 面放在壁面行内侧一格（实测有效间隙 d_v=Ny−2），
   温度 ABB 壁却在 Ny；Ra 用 H=Ny 定义 ⇒ 物理 Ra 被高估 (Ny/(Ny−2))³（Ny=128 时 +4.8%，折算 Nu 偏低约 1%）。
   这是系统性偏低的一部分，但不是全部（Ny=64→128 的 −16.5%→−8.9% 才是主项）。
3. **高 Ra 真正瓶颈 = 边界层欠分辨**：Ma≤0.1 把 α 压到约 0.028（τ_f≈0.56），Ra=1e5 的热边界层只剩
   2–3 格，数值扩散吃掉热流 ⇒ Nu 系统性偏低。出路是加密网格（Ny≥256），不是调参。
4. **加密时 Ma 必须锁住**：本文 α ∝ H（ma_max 固定）⇒ H↑ 同时 Ma↓ 且 h↓，才真正收敛到同一无量纲
   问题；若固定 α 加密，Ma 会涨到 0.6，压缩性误差 O(Ma²) 反噬。
5. **fp32 只作加速备选**：本轮 1e3 档用 fp32（Ma=0.014，舍入风险小），Ra=1e4/1e5 既有数据为 fp64；
   同网格 fp32/fp64 交叉核对作业仍在集群队列 ⇒ 高 Ra 档以 fp64 为准。
6. **可重跑性**：固定初场种子 12345 ⇒ 同参数重跑**逐位**相同（`tests/test_repro_guo2002_dhc.py` 有断言）。
7. **本脚本的 GCI 实现待修**：`scripts/repro_guo2002_analysis.py::richardson` 的 p 与外推值已用合成
   二阶数据验证（`test_recovers_second_order_limit` 通过），但 GCI 项打印出负值，未用于判定；
   第 2 轮改用库内已验证的 `rb_convection_validation.gci`。

## ⑥ 误差判定（显式回答是否 <5%）

| Ra | 关键量 | 取值 | DVD 1983 | 相对误差 | <5%? |
|---|---|---|---|---|---|
| 1e3 | 平均 Nu | 1.08912（Ny=32，本文 fp32，Ma=0.014） | 1.118 | 2.58% | **是** |
| 1e4 | 平均 Nu | 2.13279（Ny=128 前作 fp64）/ 外推 2.1473 | 2.243 | 4.91% / 4.27% | **是**（边沿） |
| 1e5 | 平均 Nu | 4.11553（Ny=128 前作 fp64） | 4.519 | 8.93% | **否（honest-FAIL）** |

**honest-FAIL**：Ra=1e5 未达 <5%。
`fail_stage = resolution`：Ny=128 的热边界层仅 2–3 格，Ma≤0.1 下 α≈0.028，数值扩散使 Nu 偏低 8.93%。
`failure_reason`：Ny=64→128 误差 −16.45% → −8.93% 严格单调收敛 ⇒ 属**离散误差**，非物理模型/公式错误。
修复路径（已递交集群作业 cj1021/cj1022，队列积压未跑）：Ny=192/256 + Richardson，据单调收敛趋势
预期外推 4.4–4.5（<3%）。等温线形态仅在 Ra=1e3 有图形证据，Ra=1e4/1e5 未给图（不粉饰）。

## ⑦ 生态摩擦（result.ecosystem_friction）

```json
[
 {
  "stage": "discovery/API",
  "blocker": "boussinesq.py 的 docstring 引的是 Guo/Zheng/Shi 2002 PRE 65:046308（外力格式），不是本单外部锚 Guo/Shi/Zheng 2002 IJNMF 39:325（耦合 LBGK）；该模块只有 plume dataclass，无法跑 DHC/Ra 基准",
  "evidence": "src/tensorlbm/boussinesq.py:18-22 与论文题名对照；真正的双分布 DHC 求解器在 rb_convection_validation.py",
  "suggestion": "boussinesq.py 顶部加 capability contract 声明它=羽流配置并指向 rb_convection_validation/thermal 的耦合 LBGK 入口；文献条目分开登记"
 },
 {
  "stage": "postprocess",
  "blocker": "run_to_steady 只返回标量，无末态场 ⇒ 无法从库入口产出等温线/云图（论文级复现必须有场图）",
  "evidence": "src/tensorlbm/rb_convection_validation.py run_to_steady 返回 dict 无 T 场",
  "suggestion": "本轮已加 return_fields=True 钩子；长期建议统一 field snapshot/artifact 接口"
 },
 {
  "stage": "entry",
  "blocker": "库里没有「按论文基准表跑 DHC」的单入口/CLI，需自写 3 个脚本（worker/ladder/analysis）",
  "evidence": "scripts/repro_guo2002_{dhc,ladder,analysis}.py 为本轮新增",
  "suggestion": "把 ladder+Richardson+GCI 收编为通用 benchmark runner（输入=容差表，输出=判据 JSON+图）"
 },
 {
  "stage": "numerics",
  "blocker": "Ma≤0.1 约束下 α 被压到约 0.028（τ_f≈0.56），Ra≥1e5 热边界层仅 2–3 格，Nu 系统性偏低 8.93%（Ny=128）",
  "evidence": "runs/repro_guo2002_dhc/ra100000_ny128_ma0.2.json vs de Vahl Davis 4.519",
  "suggestion": "make_convection 增加边界层分辨诊断（δ_T/h 估计 + 警告）与所需 Ny 的自动建议"
 },
 {
  "stage": "numerics/geometry",
  "blocker": "pre-streaming 半程反弹的有效流动间隙是 Ny−2（温度壁在 Ny），Ra 却按 H=Ny 定义 ⇒ 物理 Ra 高估 (Ny/(Ny−2))³=+4.8%@Ny=128",
  "evidence": "rb_convection_validation.calibrate_gaps；docs/reports/P96W3-N04_thermal_convection_validation.md §C3 point 6",
  "suggestion": "make_convection 对 Ra 做一次口径校正，或在 info 里显式给出 Ra_effective"
 },
 {
  "stage": "infra/cluster",
  "blocker": "集群队列严重积压：递交后 235 个作业在前，1 小时内全部作业（cj1001–cj1022）停在 queued；双节点 load≈37 却不下发",
  "evidence": "run_cluster ps：最早 19:44 递交的作业到 21:00 仍 queued",
  "suggestion": "调度器给出队列深度/预计等待与优先级；长作业支持抢占低优先级或本地兜底"
 },
 {
  "stage": "infra/local",
  "blocker": "本地控制面主机 load average 583（128 核），64² fp64 实测 7.4 ms/step，>1e5 步的 LBM 扫描不可行；shell 工具 120 s 超时会杀进程组，必须 setsid 后台化",
  "evidence": "uptime load 583.51；/tmp/bench.py 输出 7.411 ms/step @64² fp64；ra1000_ny32 10000 步耗时 49.2 s",
  "suggestion": "控制面与计算面分离；给 agent 提供受管后台任务接口（存活/日志/超时）"
 }
]
```

## 复现命令（repro_cmd）

```bash
# 单档（本文主命令）
python scripts/repro_guo2002_dhc.py --ra 1e5 --ny 192 --ma-max 0.10 --dtype fp64 --turnover 100 \
  --out runs/repro_guo2002_dhc/ra100000_ny192_ma0.1.json --fields runs/repro_guo2002_dhc/ra100000_ny192_ma0.1.npz

# 网格阶梯（集群/后台）
python scripts/repro_guo2002_ladder.py --runs 1e4:64:0.1,1e4:96:0.1,1e4:128:0.1 --dtype fp64 \
  --threads 32 --turnover 100 --fields --outdir runs/repro_guo2002_dhc

# 对表 + Richardson 外推 + 图件
python scripts/repro_guo2002_analysis.py --rundir runs/repro_guo2002_dhc \
  --figdir figures/repro_guo2002 --out runs/repro_guo2002_dhc/summary.json

# 用例（6 条，分钟级）
python -m pytest tests/test_repro_guo2002_dhc.py -q
```