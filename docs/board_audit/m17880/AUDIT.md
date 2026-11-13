# 交差审计 m17880 — m17762 `XVAL-LBM-FREE-ENERGY-CAHN-HILLIARD-CORE`（TensorLBM）

**对象**：mission m17762（repo=TensorLBM，status=succeeded，finished 2026-11-13 09:48Z，model deepseek-v4-flash，auto-approve）
**审计者**：board-audit @ m17880（独立席位，未修改被审文件；只读 m17762 工作树 + 自带复算）
**裁定**：**L3**（L2/L3 机器可核；L4/L5 拒绝，理由见 §6）
**结论一句话**：`geq_convention` 移植在数值上是真的（TL vs OSS ≤0.052%）并且解释了 swift 约定的虚假扩散；但"复现解析 CH 色散"只在 k≤0.785 成立（5 模中 2 模超 15%，m=11 达 23.69%），且 0 个交付件落在 git（整单只在工作树里）、报告文档缺失、渲染脚本语法错。

---

## 1 案例介绍（物理背景 / 几何 / 参数）

自由能（Cahn–Hilliard）双分布 LBM：`f` 走 NS，`g` 走序参量 ψ，化学势 μ 由自由能泛函变分给出，τ_g 控制迁移率 `M = Γ(τ_g−1/2)`。被审单声称：TensorLBM 历史上的 `swift` 序参量平衡态（ψ 铺满全方向 `w_i ψ(1+adv_i)`）里含有与 Γ 无关的虚假扩散 `D = c_s²(τ_g−1/2)`，因此它解的**不是** CH 方程；把参考实现（Lee & Lin 结构：ψ 只由静止分量承载 + 迁移率核 `λ_i=(3/2)(3|c|²−3)`）移植成 `geq_convention="lee_lin"` 后应无此伪项。

| 项 | 值 |
|---|---|
| 外部锚 | `ebinan92/Fingering_dynamics` @ `150a5739e320498c6ecf22ba31ab9ebc57130462`（审计独立复核：`git rev-parse HEAD` 一致，`lattice_boltzmann/fingering_periodic.py` sha256 `d3789463…` 一致） |
| 几何 | z 不变的 `(nz,ny,nx)=(1,64,64)`，全周期，动量冻结 u=0（只跑序参量块） |
| σ / ξ / W | 0.05 / 2.0（B 段）；C 段 ξ=1.0 |
| A = −a | 0.075 |
| κ | 0.0375（`k_c=√(A/κ)=1.4142`） |
| Γ（迁移率因子） | 1.0（另扫 0.5 / 2.0） |
| τ_g | `1/(3−√3)=0.788675`，τ_g−1/2 = 0.288675 |
| ε（初始幅值） | 0.01（另扫 0.005 / 0.02） |
| 模数 | m = 3, 5, 8, 11, 14 → k = 2πm/N = 0.2945…1.3744 |
| 判据 | owner 标准 <15%（`tensor_benchmark_cases` 原为 **0 行**，本审计新登记 2 行见 §6） |

## 2 怎么做的（方法与工具链）

被审单交付（全部**未入库**，`git status` 全部 `??` / ` M`）：

1. `src/tensorlbm/multiphase3d.py` `free_energy_step_3d(..., geq_convention="swift"|"lee_lin")`（+41/−2 行，默认 swift 保持不变）→ 快照 `refs/src/tensorlbm/multiphase3d.py` + `refs/src/tensorlbm/multiphase3d.patch`。
2. `scripts/xval_m17762_free_energy_ch_core.py`（32 KB，A 算子 / B 参数反演 / C 色散 / D 力形式差 / E swift 衰减扫 / F 作图），OSS 臂用 `importlib` **真跑** 上游 `fingering_periodic.py` 的算子，非重写。
3. `tests/test_xval_m17762_free_energy_ch.py`（6 测试）。
4. `artifacts/xval_m17762/{xval_free_energy_ch_core.json, fig1..3.png, run.log, run_ef.log}`。

本审计的独立动作（全部工件在 `docs/board_audit/m17880/artifacts/`，由 `build_evidence.py` 程序化生成，零手抄）：

- `manifest.json`：快照 9 文件 sha256 与 m17762 原件逐一对齐（identical=true）。
- `repro_compare.json`：**重跑 repro_cmd**（`--parts ABCDEF --out /tmp/m17880_repro`）→ JSON payload 逐键**完全相同**（仅 timestamp/torch 无关项除外），3 张 PNG sha256 **逐位相同**（`a494ff2d…`/`ae733aee…`/`9a2c9396…`）。
- `ref_vs_got.json`：把交付 JSON 的每行 ref/got/误差重算成裁判表（含绝对偏差与 k/k_c）。
- `landing_checks.json`：`git cat-file -e HEAD:<path>` 逐个探针 → 4/4 交付件 **ABSENT**，`HEAD:src/tensorlbm/multiphase3d.py` 中 `geq_convention` 出现 **0** 次。
- `pytest_rerun.json` / `regression_rerun.json`：审计席重跑。
- `render_script_syntax_check.json`：`py_compile` 失败（见 §5 F3）。

## 3 计算结果（裁判表：ref vs got，全是程序化取的）

C 段色散（N=64，Γ=1，ε=0.01，u 冻结）：

| m | k | ref 解析 ω | got OSS ω | got TL(lee_lin) ω | TL vs 解析 | OSS vs 解析 | 绝对偏差 | <15%? |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.2945 | 1.7966e-3 | 1.7137e-3 | 1.7146e-3 | −4.56% | −4.61% | 8.2e-5 | ✅ |
| 5 | 0.4909 | 4.5884e-3 | 4.6801e-3 | 4.6801e-3 | +2.00% | +2.00% | 9.2e-5 | ✅ |
| 8 | 0.7854 | 9.2361e-3 | 9.9072e-3 | 9.9071e-3 | +7.26% | +7.27% | 6.7e-4 | ✅ |
| 11 | 1.0799 | 1.0526e-2 | 1.3019e-2 | 1.30199e-2 | **+23.69%** | +23.68% | 2.5e-3 | ❌ |
| 14 | 1.3744 | 2.2678e-3 | 9.1115e-3 | 9.1109e-3 | **+301.75%** | +301.77% | 6.8e-3 | ❌ |

移植等价性（TL vs OSS，逐模）：0.0522% / 0.0007% / 0.0009% / 0.0083% / 0.0062% → **全部 ≪15%**。

其余裁判项：B 参数反演 σ 复原相对误差 `1.4e-16`（σ=0.05→0.05），β/|a|=1.000，κ/κ_OSS=1.000，μ 算子 TL-vs-OSS 相对 L2 `1.9e-13`；剖面法测 σ = **−10.40%**（TL 与 OSS 同值，故属离散梯度能积分误差而非移植差异）；迁移率标定 `M_eff` 相对预测值 1.94%/2.00%/2.12%（Γ=0.5/1/2）；swift 伪扩散：实测衰减 vs 模型 `−c_s²(τ_g−1/2)k²` 相对差 0.51%/1.41%/3.54%/6.16%/9.76%；算子收敛阶 1.985–1.997（二阶）；D 段 `F_TL−F_OSS = −∇(φμ)` 相对 L2 7.16e-2。

## 4 图

`artifacts/fig_audit_dispersion.png`（由 `make_figures.py` 只读 `ref_vs_got.json` 生成）：
(a) 解析 CH 色散曲线 + TL/OSS 实测点 + swift 臂模型（负值，全 k 衰减）+ k_c 位置；
(b) TL/OSS vs 解析的相对误差棒与 ±15% 带，m=11（24%）与 m=14（302%）越界被标注。
被审单自带 3 图（`refs/artifacts/fig1..3.png`）已随快照入库，与重跑逐位一致。

## 5 注意点（坑 / 局限 / 适用范围）

- **F1（阻断，落地缺陷）**：分支 `agent/m17762` HEAD **不含任何交付件**（4/4 ABSENT，`geq_convention` 计数 0，`git status` 全为 `??/M`）。能力只存在于工作树，任何 fresh checkout 即丢失。本审计已把字节 + patch 快照进 `docs/board_audit/m17880/refs/`（sha256 对齐），补录方式：`git apply refs/src/tensorlbm/multiphase3d.patch`。
- **F2（阻断，裁判缺失）**：m17762 从未向 `tensor_benchmark_cases` 登记 case（审计前该 anchor 0 行）；无 `predeclared.json`／预声明容差 → 15% 只能按 owner 通标事后套用。本审计已补登 2 行（§6）。
- **F3（阻断，报告缺陷）**：代码 docstring 与测试 docstring 引用的 `docs/XVAL_M17762_LBM_FREE_ENERGY_CAHN_HILLIARD_CORE.md` **不存在**；渲染脚本 `scripts/render_xval_m17762_report.py` `py_compile` **失败**（30 处 `%%` 应为 `%`，首错 line 48）→ 六段报告从未生成；`tensor_reports` 中 mission_id=17762 **0 行**（报告欠账）。
- **F4（方法论，非阻断但重要）**：测试只断言 **m=6/N=48（k=0.785, 7.26%）** 这一个最有利模；交付 JSON 自报的 m=11（23.69%）、m=14（301.75%）从未进入任何断言。单模通过 ≠ 色散关系通过。
- **F5（读数陷阱）**：m=14 的 k/k_c = 0.972，解析 ω 近零（2.3e-3），相对误差天然病态；该点应看绝对偏差（6.8e-3，约为峰值 ω 的 52%），不该拿 302% 当"实现坏了"的证据。**m=11 是干净的健康模，23.69% 才是真判据失败点**。
- **F6**：`tl_swift_omega` 全 NaN 是"该约定下线性模不增长"的正确表现（由 E 段负衰减扫证实），不是缺数据。
- 适用范围：仅 z 不变单模小扰动（线性区），未做非线性/多模/3D 斑点（`fingering.py` 的 Saffman–Taylor 分支）验证；因而不能外推为"CH 求解器已通"。

## 6 误差判定（逐项，显式回答是否 <3%）

| 关键量 | ref | got | 误差 | <3%? |
|---|---|---|---|---|
| σ 参数反演（B） | 0.05 | 0.05 | 1.4e-16 | ✅ |
| μ 算子 TL vs OSS（B） | OSS 字面 | TL central7 | 1.9e-13 | ✅ |
| 色散 ω, m=3 | 1.7966e-3 | 1.7146e-3 | 4.56% | ❌ |
| 色散 ω, m=5 | 4.5884e-3 | 4.6801e-3 | 2.00% | ✅ |
| 色散 ω, m=8 | 9.2361e-3 | 9.9071e-3 | 7.26% | ❌ |
| 色散 ω, m=11 | 1.0526e-2 | 1.30199e-2 | 23.69% | ❌ |
| 色散 ω, m=14（近中性模） | 2.2678e-3 | 9.1109e-3 | 301.75%（绝对偏差 6.8e-3） | ❌ |
| 移植等价 TL vs OSS（最差模） | OSS 实测 | TL 实测 | 0.052% | ✅ |
| 迁移率 M_eff vs 预测（Γ=1） | 0.28868 | 0.29445 | 2.00% | ✅ |
| 迁移率 M_eff vs 预测（Γ=0.5/2） | 0.14434/0.57735 | 0.14713/0.58960 | 1.94%/2.12% | ✅ |
| swift 伪扩散衰减 vs 模型（最差模 m=14） | −0.180267 | −0.197859 | 9.76% | ❌ |
| 剖面法测 σ（xi=2） | 0.05 | 0.04480 | −10.40% | ❌ |

**结论（不许含糊）**：参数反演/算子/迁移率三项 **<3% 成立**；**色散关系对解析解 5 模中 3 模超 3%、2 模超 15% → 不成立**；移植等价性 <0.06% 成立。据此 L4 不成立。

## 7 判级（L0–L5）

| 级别 | 判定 | 依据 |
|---|---|---|
| L0 文档 | 部分 | 审计快照在，但被审单声称的 doc 不存在（F3） |
| L1 数学/单元 | ✅ | 6 测试重跑 `6 passed in 9.37s`；算子阶 1.985–1.997；`Σwλcc=δ` `1e-12` |
| **L2 独立功能** | ✅ | `free_energy_step_3d(geq_convention=…)` 真输入→计算→输出；重跑逐位可复现 |
| **L3 系统集成** | ✅ | 被改函数在主求解器路径上且**活的**：`src/tensorlbm/dam_break_3d.py:862`、`scripts/bench_fe_water_entry_3d.py:83` 调用它（默认 swift，未受影响：既有 2 个测试文件重跑 `35 passed, 1 xfailed in 22.25s`），`src/tensorlbm/__init__.py:1868` 再导出；切约定即改变度量（swift 衰减 / lee_lin 增长）。**限定**：`lee_lin` 本身 0 个采纳调用点（repo 内 `geq_convention` 仅出现在 def + 本单脚本/测试） |
| L4 基准验证 | ❌ | (a) 审计前 `tensor_benchmark_cases` 该 anchor **0 行**、无预声明容差；(b) 解析 CH 色散裁判 5 模中 2 模 >15%（m=11 23.69% 是健康模）；(c) 断言只覆盖最有利单模（F4）；(d) 交付件 0 落地（F1）→ 无 provenance |
| L5 科研工程 | ❌ | 外部锚是同级实现（code-to-code），无实验/文献物理真值；线性单模范围，未到工程结论 |

## 8 处置（disposition）

**approve with required remediation**：等级记 L3（maturity_gate 机器可核：impl_files/entry/run_artifacts/integration_callers 全部真实存在），但以下三项必须在后续单闭环，不得以 L3 消账掩盖：
1. **落地**：`git apply docs/board_audit/m17880/refs/src/tensorlbm/multiphase3d.patch` + 三个新文件入库（F1）。
2. **裁判**：已补登的 2 行 benchmark case（1 passed 等价性 / 1 failed 解析色散）作为该 anchor 的机器判据；解析色散 KPI 需要要么扩大 k 范围到 k>1.0 的诚实标注（离散色散对连续解析式的固有偏差），要么把判据改为"TL vs OSS 等价 + 长波极限 vs 解析"并预声明（F2/F5）。
3. **报告**：修 `render_xval_m17762_report.py` 的 `%%`（30 处）→ 生成被引用的 doc → 补发 `tensor_reports`（F3）。

## 9 复算命令（审计席实际执行的）

```bash
# 1) 复现交付工件（逐位）
cd /root/tp-agent/work/m17762 && PYTHONPATH=src python3 scripts/xval_m17762_free_energy_ch_core.py \
  --parts ABCDEF --oss-root /tmp/Fingering_dynamics --out /tmp/m17880_repro
# 2) 单测 + 回归
python3 -m pytest tests/test_xval_m17762_free_energy_ch.py -q          # 6 passed in 9.37s
python3 -m pytest tests/test_multiphase3d_d3q27_free_energy.py tests/test_multiphase_extensions.py -q  # 35 passed, 1 xfailed
# 3) 证据重建 + 图
python3 docs/board_audit/m17880/build_evidence.py && python3 docs/board_audit/m17880/make_figures.py
```

## 10 独立性声明与未做项

- 独立性：审计席不改被审文件、不采信其自述 checks；上述数字全部来自重跑/`git` 探针/交付 JSON 复算。局限：`/tmp/Fingering_dynamics` 为审计环境中已有克隆（HEAD 与 sha256 已双向核对，但克隆来源未审计）；m=14 近中性模的相对误差判读依赖 `k_c` 解析值。
- **未做**（不冒充已核）：① 3D 多模/非线性斑点（`fingering.py` 分支）未验；② Γ 扫只核了交付 JSON 的 3 个点，未独立重扫；③ `xi=2/4` 的 σ 剖面误差来源（作者归因于离散梯度能积分）未独立定源；④ 未核对 deepseek-v4-flash 的 80 次工具调用轨迹（只看终态工件）。