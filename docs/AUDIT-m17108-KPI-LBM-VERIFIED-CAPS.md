# 交差审计 · m17108 / KPI-LBM-VERIFIED-CAPS（TensorLBM）

- 审计单：m17173（board-audit，独立于产出方 tp-ds-6 / deepseek-v4-flash）
- 对象：m17108「返修 m17013：二阶 Stokes 波浪锚 → 独立物理验证 + 求解器级 ref/got 基准」，status=succeeded
- 审计日期：2026-11-13
- 结论：**L2**（独立功能实现；L3/L4 不成立，理由见 §2.4 与 §6）
- 证据包：`artifacts/m17108-audit/`（原样副本 + sha256 清单 + 可复跑复核脚本）

---

## ① 案例介绍（被审对象是什么）

m17013 的交付被判「公式自证」（解析系数对着自己的闭式解核，属恒等式而非验证）。m17108 是返修单，目标是把 TensorLBM 的二阶 Stokes 造波锚（`tensorlbm.nwt.stokes2`）补成
「与闭式系数无关的物理验证 + 求解器级基准」，并如实记录未通过项。

**被测物理对象**：二维数值波浪水槽（NWT）的二阶 Stokes 入射波锚。

| 参数 | 值 | 来源 |
|---|---|---|
| 重力 g | 9.81 m/s² | `analytic_gates.py:G=9.81` |
| 无量纲水深 kh | 0.7 / 1.5 / 3.0 | `acceptance_result.json` 10 算例矩阵 |
| 波陡 ε=ka | 0.02 / 0.08 | 同上 |
| 锚点闭式解 | 二阶 Stokes（Stokes 1847；Dean & Dalrymple 1991 §4.3） | `result.json:reference` |
| 解析闸门工况 | kh=1.5,3.0 × ε=0.005/0.01/0.02/0.04（残差阶）；kh=1.5,3.0 × ε=0.02/0.08（a2 反解） | `analytic_gates.py:run_gates` |
| 求解器算例 | nx=621, ny=21, 13041 cells, 6455 steps, τ=0.505（kh0.7_eps020 为例） | `case.json:case` |

**本次交付的实质 delta**（相对 main 3514c90e，`git diff --stat` 11 文件 / +3163 行）：

- 新增 `benchmarks/verified/nwt_stokes2/analytic_gates.py`（278 行，20 项物理闸门）+ `result_analytic_gates.json`
- 新增 `tests/test_nwt_stokes2_2d_verified.py`（142 行，6 个测试）
- 更新 `benchmarks/pending/nwt_stokes2_2d/result.json`（10 算例 ref/got 矩阵）
- 文档 `docs/CASE-LBM-NWT-STOKES2-INDEPENDENT-VERIFICATION.md` + 6 张图
- **未改求解器源码**：`src/tensorlbm/nwt/stokes2.py` 最后一次改动是 main 同步提交 3514c90e（非本单）

驱动/度量脚本（`run_case.py`/`run_acceptance.py`/`nwt_metrics.py`）**在 main 已存在**（m17013 遗产），本单未新增。

## ② 怎么做的（审计方法与流程）

四步，全部机器可核，不采信报告自述：

1. **取原始账**：`tensor_dev_missions` 取 m17108 的 `result`（checks/on_disk_changes/changed_files）+ `git log/diff` 定位 delta 文件。
2. **三件套落地核验**（文件是否存在、是否真跑过、数字是否可回溯）：
   - `run_artifacts`：`runs/nwt_acceptance/kh*/{case.json,probes.npz}`（10 例，`missing_cases=[]`）
   - `benchmark ref vs got`：`acceptance_result.json`（10 例逐项误差）+ `result_analytic_gates.json`
   - `repro_cmd`：`python benchmarks/verified/nwt_stokes2/analytic_gates.py`
3. **独立复算**（本审计自己写的 `artifacts/m17108-audit/verify_evidence.py`，不调被审脚本的判定）：
   - A：重放闸门套件 → 池 20/20；**静态检查其 import**（判是否走 LBM 路径）
   - B：从 `probes.npz` 的求解器探针时序**重算**谐波均值，与 `case.json` 比对
   - C：由 `acceptance_result.json` 的逐项数字 + 阈值**重算** `all_pass`，与存储旗标比对
4. **基准裁判**：任务指定的 `tensor_benchmark_cases WHERE anchor_slug='KPI-LBM-VERIFIED-CAPS'` 查询结果为 **0 行**（该锚未登记任何基准用例，见 §5 坑 1），故改以仓库内声明的阈值（`thresholds`）与闭式解 ref 为判据，并在 §6 逐项回答 <3%。

复跑命令：

```bash
python benchmarks/verified/nwt_stokes2/analytic_gates.py          # 6 s，20/20 PASS
python -m pytest tests/test_nwt_stokes2_2d_verified.py -q          # 6 passed in 6.51 s
python artifacts/m17108-audit/verify_evidence.py                   # 审计方独立复算，EXIT=0
```

## ③ 计算结果（审计自测数字表，零手抄，全部由脚本注入）

**A. 三件套 & 复现核验**

| 检查 | 结果 | 数值 |
|---|---|---|
| 闸门重放 | 20/20 PASS | `pool={"passed":20,"total":20}` |
| 闸门是否触达 LBM 求解器 | **否** | imports 仅 `tensorlbm.nwt.stokes2.{Stokes2Wave,group_speed,phase_speed}`+torch，无 tank/lattice/wavemaker |
| 工件可回溯性 | 通过 | probes.npz 重算 A1=0.3852498009122123 vs 存储同值（|Δ|=0）；A2 |Δ|=6.9e-18 |
| 判定自洽性 | 通过 | 存储 all_pass=false，按阈值重算 all_pass=false（一致）；C1/C2/C4 逐项与阈值判定一致 |
| 测试 | 6 passed / 6.51 s | `test_analytic_gate_suite_passes` 等 6 项 |
| 算例完整性 | 10 例，缺 0 | `n_cases=10, missing_cases=[]` |

**B. 求解器级 ref vs got（10 例，摘自 `acceptance_result.json`，容差见括号）**

| case | kh | ε | C1 相速误差 (<1%) | A1 误差 (<3%) | A2 误差 (<10%) | 例判 |
|---|---|---|---|---|---|---|
| kh0.7_eps020 | 0.7 | 0.02 | 0.0304 ✗ | 0.0068 ✓ | 0.263 ✗ | FAIL |
| kh0.7_eps080 | 0.7 | 0.08 | 0.0129 ✗ | 0.0298 ✓ | 0.079 ✓ | PASS* |
| kh1.5_eps020 | 1.5 | 0.02 | 0.0439 ✗ | 0.0266 ✓ | 0.414 ✗ | FAIL |
| kh1.5_eps080 | 1.5 | 0.08 | 0.0366 ✗ | 0.0221 ✓ | 0.424 ✗ | FAIL |
| kh3.0_eps020 | 3.0 | 0.02 | 0.0381 ✗ | 0.1242 ✗ | 2.023 ✗ | FAIL |
| kh3.0_eps080 | 3.0 | 0.08 | 0.0509 ✗ | 0.1066 ✗ | 0.104 ✗ | FAIL |

（*该例 C1 仍超 1% 容差，C1 判据整组 FAIL）

| 判据 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| C1 色散 | 1% | 最差 5.09%（kh3.0_eps080）；全 6 例均 >1% | FAIL |
| C2 波形 A1 | 3% | 最差 12.4%（kh3.0_eps020） | FAIL |
| C2 波形 A2 | 10% | 最差 202%（kh3.0_eps020，A2 量级 1.9e-3 已近估计器分辨极限） | FAIL |
| C3 吸收判别力 | KR<5% 且负对照可辨 | KR 0.19%–4.3%，但关吸收负对照 KR 仅 2.3%→5.2% | FAIL（无判别力）|
| C4 质量守恒 | 0.5%/周期 | 通过 | PASS |
| C5 网格收敛 | 相邻波长误差收敛 | λ=40: 19.5%，λ=80: 10.1%，相对变化 1.23 | FAIL |
| C7 运行时限 | 20 min/例 | 通过 | PASS |

**C. 解析闸门（`result_analytic_gates.json`）**

- 残差阶：d log R_kin/d log ε = 2.001 / 2.000；dyn 谐波 2.001 / 2.001（期望 2）
- 由运动学自由面条件反解的二阶谐波：相对闭式解 3.50e-4 / 5.64e-3 / 2.72e-4 / 4.37e-3，全部 < 声明界 2ε²

## ④ 结果图（`artifacts/m17108-audit/figures/`）

- `fig_audit_ladder.png`：上=10 例 C1 相速误差 vs 1% 容差（对数轴）；下=A1/A2 误差 vs 3%/10% 容差。**全 6 个 kh-ε 组合的 C1 柱越过容差线**，A2 在 kh=3.0/ε=0.02 越线 20 倍。
- `fig_audit_gates.png`：左=I2 由边界条件反解的 a2 相对误差 vs 声明界 2ε²（全在界内）；右=I1 自由面运动学残差随 ε 的斜率=2（log-log 与 slope-2 参考线平行）。
  两图数据均由脚本从 JSON 读取（`make_audit_figures.py`），无手输数字。

## ⑤ 注意点（坑与局限）

1. **基准裁判缺位**：`tensor_benchmark_cases` 中 `anchor_slug='KPI-LBM-VERIFIED-CAPS'` **0 行**——该 KPI 锚未登记任何可机器裁的基准用例。本次审计只能用仓库内自声明阈值当判据；这是机制缺口（KPI 无法被自己的裁判机核），建议把 6 个 kh-ε 组合 + 阈值登记为 benchmark case。
2. **「独立物理验证」的适用范围**：20 项闸门确实不再是公式自证（有残差阶、古典极限、由边界条件反解 a2、负对照），但**只调用解析模块**，`exercises_lbm_solver_path=false`。所以它验证的是「闭式锚满足自由面运动学/动力学条件」，**不是**「LBM 求解器能复现该锚」。m17108 的 headline「独立物理验证 + 求解器级基准」前半句成立（限于解析锚）、后半句成立但不通过。
3. **kh=3.0/ε=0.02 的 A2 误差 202%**：解析 A2=1.948e-3，测量值 5.89e-3（超 ref 三倍），量级不可信。可能是估计器分辨极限（探针窗 6 周期）而非物理失效；m17108 自己也写了「A2 不可分辨」。别把该数当物理结论。
4. **工件持久性风险**：m17108 分支 `agent/m17108` **未推送 GitHub**，`/root/tensor-ecosystem/TensorLBM/benchmarks/verified/` 下无 `nwt_stokes2`（只有 main 的 `run.py`+旧 `result.json`）。若不合并/不推送，20 闸门套件随工作树回收即失。本审计已把关键证据（含 sha256）固化在 `artifacts/m17108-audit/evidence/`。
5. **未做全量复跑**：10 例求解器矩阵（每例 ~400 s）未重跑，只做了「工件→数字」回溯（1e-18 级）与闸门/测试复跑。因此「repro_cmd 可执行」成立，「完全独立重跑产出同数字」**未验证**（记为 not_run，不计通过）。
6. 审计独立性限制：本审计单与 m17108 同属平台，作者不同（本单 board-audit，产出方 tp-ds-6）；据 m17108 自述其 C3/C5 为实测不收敛，本审计只核了记录自洽性，未独立判定物理原因。

## ⑥ 误差判定（逐项显式回答「是否 <3%」）

| 关键量 | ref | got | 相对误差 | <3%？ |
|---|---|---|---|---|
| C1 相速（kh0.7/ε0.02） | 0.09295 m/s（闭式@k_meas） | 0.08969 | 3.04% | **否**（3.04%>3%；且超 1% 专用容差） |
| C1 相速（kh0.7/ε0.08） | — | — | 1.29% | 是（但超 1% 容差） |
| C1 相速（kh1.5/ε0.02） | — | — | 4.39% | **否** |
| C1 相速（kh1.5/ε0.08） | — | — | 3.66% | **否** |
| C1 相速（kh3.0/ε0.02） | — | — | 3.81% | **否** |
| C1 相速（kh3.0/ε0.08） | — | — | 5.09% | **否** |
| C2 A1（kh0.7/ε0.02） | 0.38197 | 0.38456 | 0.68% | 是 |
| C2 A1（kh3.0/ε0.02） | 0.19099 | 0.16727 | 12.4% | **否** |
| C2 A2（kh0.7/ε0.02） | 0.02279 | 0.01680 | 26.3% | **否** |
| C2 A2（kh1.5/ε0.08） | 0.074882 | 0.043127 | 42.4% | **否** |
| C2 A2（kh3.0/ε0.02） | 1.948e-3 | 5.890e-3（超 ref 三倍，疑估计器分辨极限） | 202% | **否** |
| 解析闸门 a2(BC) vs 闭式（4 工况） | 闭式 | 反解 | ≤5.64e-3 | 是（≤0.56%） |
| 解析闸门残差阶 | 2 | 2.000–2.001 | — | 是 |

**结论**：解析锚层（L1）全部 <3%（≤0.56%）；**求解器层未达 3%，也未达仓库自定的 1%/3%/10% 容差**（6/6 例 C1 超、4/6 例 A2 超）。平台精度标准「对标参考值误差 >15% 不可接受」下，kh3.0/ε0.02 的 A1(12.4%) 尚可、A2(202%) 不可接受，其余量级在 1–5% 区间。

## 判级与理由（写入 `tensor_maturity_verdicts` 的 evidence 摘要）

- **L1 达成**：20 项解析物理闸门 + 6 项测试全绿，且经审计方独立重放（非自报）。
- **L2 达成**：`analytic_gates.py` 是真实现（非 placeholder），有入口 `main()`，有真实运行工件（`result_analytic_gates.json`、10 例 `case.json`/`probes.npz`），数字可从工件回溯到 1e-18。
- **L3 不成立**：闸门套件在主求解器/平台侧**无调用方**（仅测试 import）；且真正被集成的 `tensorlbm.nwt`（`tank.py:42,305`）是 main 既有遗产，不算本单增量。
- **L4 不成立**：完整算例（10）+ 误差分析 + 可复现都有，但基准**不通过**——C1 6/6 例超容差（1.3–5.1% vs 1%）、C2 A2 202%、C5 网格不收敛（19.5%→10.1%，相对变化 1.23）。「基准验证」不能靠一份公开失败的基准来挣；m17108 自己也如实记了 `all_pass=false`，这一点**没有虚标**。
- **L5 不成立**：无外部实验/文献实测锚（`external_anchor` 仅闭式解，属 L1 级锚），求解器级也无与实验/文献实测的对照。

**消账建议**：本单按 L2 确认（独立功能实现，验证资产真实、可复现、诚实 FAIL）；KPI 锚若要求「求解器级通过独立物理验证」，则 m17108 **不构成**该 KPI 的达成，返修方向=C1 色散误差（3–5%，疑数值粘性/造波-吸收耦合）、C3 负对照判别力、C5 网格收敛。