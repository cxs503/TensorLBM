# m18207 交差核验 · m18114 XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE

- 被审单：m18114（TensorLBM，succeeded，原单 m17879 的返修轮 1），head `c6ed86d2`
- 被审工作树：`/root/tp-agent/work/m18114`；快照：`docs/board_audit/m18207/evidence/`
- 审计单：m18207（board-audit 席位）；实测事实：`verify_output.json`（本单脚本 `run_audit.py` 程序化写入）
- 落表：`tensor_maturity_verdicts` id=715，level=2，gate_state=pending
- 机器裁定干跑（`/root/tp-agent/maturity_gate.py::judge`）：claim 2 → gate_level 2（无降级原因）；claim 3 → 2（`integration_callers(grep 无引用)`）；claim 4 → 2（同上）

## ① 案例介绍（物理背景 / 几何 / 参数）

对称双阱自由能二组分（binary-fluid）LBM 的**平衡压力闭环**：自由能
`f = (a/2)φ² + (b/4)φ⁴ + (κ/2)|∇φ|²`（`a=-A, b=B`），压力张量
`P_ab = d δ_ab + κ ∂_aφ ∂_bφ`，`d = p0 - κφ∇²φ - (κ/2)|∇φ|²`。解析锚：

- 共存压力 `p_coex = A²/(4B)`
- 界面张力 `σ = (2√2/3)√(κA³)/B`（等价于 `(2/3)A²ξ/B`，其中 `ξ=√(2κ/A)`）
- 形态：周期域内两条**平直**界面（`product-of-tanh`，`nx=128, ny=16`），
  `A=B=1`，`interface_cells = ξ/h ∈ {1.5, 2, 3}`（`κ = Aξ²/2 = 1.125/2/4.5`），
  压力张量诊断在解析平衡场上做；LBM 臂 `steps=2000, tau_f=1.0, tau_g=0.7, Γ=0.5`。
- 判据出处：交付自带 `predeclared.json`（事前声明，未事后调参）＋平台登记基准行
  `tensor_benchmark_cases` id=85/86/87；外部锚：Pooley, Furtado & Carr, PRE 78, 056709 (2008)。

## ② 怎么做的（核验方法/流程）

本单是**交差核验，非重做**，全部动作只读或复算，不改被审工作树：

1. `run_audit.py` 取 9 个交付物的存在性/字节数/sha256 并做快照（A1）。
2. 判交付是否已进生态 main（A2：`git cat-file -e HEAD:...` + 远端分支）。
3. **实跑**测试：`python -m pytest tests/test_binary_equilibrium_pressure_xval.py -q`（A3）。
4. **实跑** `repro_cmd`：`run_xval.py --out <tmp>`，与交付工件逐键比对（A4）。
5. 基准裁判：从 `tensor_benchmark_cases WHERE anchor_slug=...` 取三条登记行，逐行核覆盖与容差（A5）。
6. 用**交付模块本体**独立复算登记行 85 的配置（`A=B=0.1, κ=0.45 → ξ=3`）（A6）。
7. 集成调用方 grep（主求解器/套件/平台侧）（A7）；自述 vs 工件对照（A8）。
8. 落表前用平台守护 `judge()` **干跑**（只读），把机器裁定级别先算出来（见文件头）。

工具链：python3 + torch(CPU, float64) + pytest + psql；图件 `make_figs.py`（matplotlib）。

## ③ 计算结果（数字表，程序化注入）

测试与复现：

| 项 | 结果 |
|---|---|
| pytest `tests/test_binary_equilibrium_pressure_xval.py` | **7 passed**，rc=0 |
| `repro_cmd` 复跑 | rc=0，46.26 s（本单实测），`VERDICT: PASS` 18/18 |
| 复跑工件 vs 交付工件 | `checks` 逐键相等、`rows` 逐键相等；复跑 JSON sha256 `bb130d75…`（与交付一致） |

交付判据表（`A=B=1, nx=128, ny=16`，来自交付 JSON，本单只读注入）：

| ξ/h | σ 解析 | σ 积分 | σ 相对误差 | P_nn 均匀性(max\|P_nn-mean\|/p_coex) | 共存压力相对误差 | LBM 总压 P_nn 偏差 | LBM \|u\|max |
|---|---|---|---|---|---|---|---|
| 1.5 | 1.0 | 0.898486 | **−10.15%** | 23.09%（阈值 25%） | 0.0 | 14.99%（阈值 25%） | 1.29e−2 |
| 2.0 | 1.333333 | 1.252140 | **−6.09%** | 14.19% | 4.05e−13 | 8.30% | 8.16e−3 |
| 3.0 | 2.0 | 1.943142 | **−2.84%** | 6.70% | 1.74e−8 | 3.67% | 2.97e−3 |

基准裁判（`tensor_benchmark_cases`，anchor=`XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE`）：

| 登记行 | 用例 | 登记状态 | 交付覆盖 | 本单独立复算 |
|---|---|---|---|---|
| id=85 | FE-FLAT-BAND-SURFACE-TENSION（ξ=3, A=0.1, κ=0.45, 64×8） | passed（σ 2.01e−2, Pnn 1.4799e−5） | **部分**：σ 家族覆盖，但交付 σ 取自解析场压力张量，非 population 二阶矩 | 用交付模块复算：σ=0.194314，**−2.84%**（带内）；Pnn 均匀性 6.43e−2（静态张量）/3.59e−2（LBM 总压），与登记 1.48e−5 差 3 个量级（口径不同） |
| id=86 | FE-STATIC-DROPLET-LAPLACE（R16/R24） | passed（快照 0.18%/13.48%） | **未覆盖**（模块无 droplet/Laplace 臂） | 未做 |
| id=87 | FE-DECLARED-CURVATURE-ARM-STABILITY（dropR36） | **failed**（NaN from step 500） | **未覆盖** | 未做 |

## ④ 结果图

- `figs/fig1_sigma_convergence.png`：σ 相对误差 vs ξ/h，含 owner <15% 带与过 ξ/h=1.5 的 O(h²) 参考线；实测 10.15%→6.09%→2.84%，二阶收敛趋势成立。
- `figs/fig2_pnn_profiles.png`：三分辨率下 `P_nn(x)/p_coex` 与 `P_tt(x)/p_coex` 剖面；`P_tt` 在界面处下陷而 `P_nn` 近似均匀，界面处残差随 ξ/h 单调变小（23.09%→14.19%→6.70%）。

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

1. **双梯度格式在平界面配置下恒等**（F3）：`iso9` 的 Tomita 权重 `4/12+1/12+1/12=1/2`，
   对 y 不变的 `φ(x)` 退化为 5 点中心差分，实测 `grad_x` 最大差 5.6e−17，σ 差 0.0。
   交付 JSON 里 `iso9`/`5pt` 两行是同一批数字——文档所称「离散敏感性可测而非假设」在该配置下
   没有被真正行使，是重复计数而非两项独立证据。
2. **σ 口径**（F4）：交付 σ 来自**解析平衡场**的自由能压力张量（离散化误差度量），
   登记行 85 的 got 来自 **population 二阶矩**（LBM 态度量）。两者不可逐位互证，报告与基准行应注明口径。
3. **LBM 臂是「相场冻结」的静态两相测试**：每步用 `init_free_energy_g` 重设 `g`，只推进动量群 `f`；
   交付在 `predeclared.json` 已前置声明生产 `g` 方程缺 `-M∇μ` 通量（实测 400 步峰值衰减比 −0.049）。
   因此 LBM 臂量的是**伪流/机械平衡**，不是自洽两相演化——这是该锚点的标准静态测试，但适用边界须写明。
4. **无下游调用方**（F1）：`run_binary_equilibrium_pressure_benchmark` 只在 `multiphase_benchmarks.py:713`
   定义，`run_multiphase_benchmark_suite` 未注册、`benchmarks/bench_multiphase.py` 未导入、平台侧无引用。
5. **交付未进生态 main**（F5）：`git cat-file -e HEAD:src/tensorlbm/phasefield/binary_equilibrium_pressure.py`
   → ABSENT，远端亦无 `m18114` 分支；合并前下游不可消费。
6. **自述瑕疵**（F6）：`run_xval.py` 文档串声明两张 png，实际只 `savefig` 一张；m18114 的
   `result.checks` 把 PNG 的 105928 B 写成 JSON 的大小（JSON 实为 13868 B）。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 实测误差 | <3%？ | 结论 |
|---|---|---|---|---|
| σ（ξ/h=1.5） | (2√2/3)√(κA³)/B=1.0 | 10.15% | **否** | 在 owner 15% 带内，但**不满足 3% 级** |
| σ（ξ/h=2.0） | 1.333333 | 6.09% | **否** | 同上 |
| σ（ξ/h=3.0） | 2.0 | 2.84% | **是（2.84%<3%）** | 刚入 3% 带 |
| p_coex（Maxwell 数值构造） | A²/(4B)=0.25 | ≤1.74e−8（相对） | **是** | 机器精度级 |
| 共存压力（本体 p0，bulk） | 0.25 | 0（ξ=1.5）/4.05e−13 | **是** | 闭式一致 |
| P_nn 均匀性（ξ/h=1.5） | 0（连续极限） | 23.09%（阈值 25%） | **否**（≠0 即离散误差，按预声明带判） | 带内但贴边 |
| LBM 伪流 \|u\|max/c_s | 0 | 1.29e−2/0.5774 = 2.2% | — | 低马赫/弱伪流成立 |
| 登记行 85 σ 家族（本单复算 ξ=3, A=B=0.1） | 0.2 | 2.84%（登记 got 2.01%） | **是** | 同带内，但不可逐位复现 |

**诚实结论**：交付在**平面双界面族**上真实达标（owner <15% 判据逐项过，18/18 本地复现，
工件可逐位重放），代码/测试/入口/工件齐备 → **L2 成立**；但**无下游调用方**（L3 不达）、
**基准覆盖只到平面族且登记行 87 仍 failed**（L4 不达）。故判 **level=2**，与机器干跑一致；
报告中的 σ 数字与 3% 级精度**有 2/3 项不满足**，不得宣称「全部 <3%」。

## 处置（disposition）

- `approve with notes`（按 L2 消账，不返修）：主体交付真实、可复现、判据在 owner 带内；
  不因「无调用方/覆盖不全」把已成立的功能打成 L1。
- 建议后续（不由本单返修）：① 把 `run_binary_equilibrium_pressure_benchmark` 注册进
  `run_multiphase_benchmark_suite` 或 CI 入口以获得真调用方（→L3/L4 路径）；
  ② 补 droplet Laplace（R16/R24）与 dropR36 稳定性臂以覆盖登记行 86/87 并消掉锚点级 failed；
  ③ 在平界面之外（如液滴/曲界面）行使 `iso9` vs `5pt` 才有多分辨别力；
  ④ 交付合入生态 main。