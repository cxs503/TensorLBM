# m18289 交差审计：XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR-XVAL-SYMBOL-SP（判 L1）

## ① 案例介绍
对象=返修轮 **m18289**（原单 m18126 被 maturity_gate 复核为 L0 后开的返修轮，purpose=补齐 L2 三件套）。
学科对象：D2Q9 双组分 Shan–Chen（MCMP）交叉力算符，与开源 **lbmpy 2.0** 交叉对标。
约定：`G_12`（本仓，<0 斥力）↔ `g_ab`（OSS，>0 斥力），桥 `G_12=-g_ab`；D2Q9 对称模板 ⇒
`Σw·c·ψ(x+c) = -Σ w·c·ψ(x−c)`（前向/后向 gather 等价）。
m18289 交付物只有 2 个文件（`xval/mcmp_cross_force/{__init__,operators}.py`，共 166 行），且**未提交**
（`git status` = `?? xval/mcmp_cross_force/`）。本审计把两文件按 sha256 冻结在
`audit/xval_mcmp_cross_force_m18289/refs/`（MANIFEST.sha256, operators.py=bcb03ce1…）。

## ② 怎么做的
1) 读 mission 记录（`/api/v1/tensor/missions/18289`）与工作树；2) 三件套核验（impl 有 / entry 无 / run_artifacts 无）；
3) 基准裁判：`tensor_benchmark_cases` 查该 anchor → **0 行**；4) 执行探针（fp64, torch 2.7.1，本工作树 src/tensorlbm）：
包装器 vs 库核身份、OSS 算符 vs 库核、模板系数 vs cs²、均匀场零力、exp-ψ 组合、周期流；
5) 继承数值复核：用交付代码自身度量 `||got-ref||/||ref-mean(ref)||` 从 6ad01d71 的 npz 重算物理臂；
6) 外部锚复核：lbmpy 2.0 sdist sha256 与上游测试文件 sha256/行号逐条对齐。
（未做：不重跑 OSS 代码生成臂、不重跑两相模拟——独立性限制如实声明。）

## ③ 计算结果（程序化写入，零手抄）
| 探针 | 量 | 值 |
|---|---|---|
| P2 | 包装器 vs 库核 `sc_two_component_force` | bitwise 相等，max_abs_diff=0.0 |
| P3 | OSS 算符 vs 库核（linear ψ） | 最差相对差 **3.41e-16** |
| P4 | 前/后向 gather 系数 / k | +0.33333334 / −0.33333334（反对称残差 2.2e-16，理论 1/3）|
| P5 | 均匀场交叉力 | 0.0（两算符）|
| P6 | exp-ψ 组合 vs OSS | 最差 2.72e-16 |
| P7 | stream_periodic | 质量相对漂移 0.0，与手写 pull 完全一致 |
| 继承物理臂 | N64 g=5 exp | **200.3399%**（交付度量；朴素归一 120.3451%）；其余 3/4 臂 64/64/128 个 NaN |
| 基准裁判 | `tensor_benchmark_cases` 该 anchor | **0 行** ⇒ ref-vs-got 无从裁 |

## ④ 结果图
图1=算符恒等残差+模板系数；图2=继承物理臂 vs 15%/3% 门；图3=本 verdict 的 evidence 键对照门判据方案。

## ⑤ 注意点
- 交付代码是**真代码**（非占位），缺级别的原因是缺运行链（entry/工件/提交），不是数学错：判 L0 与判 L2 都失真。
- `__init__.py:9-18` 宣称 7 文件布局，**5 个不存在**（含 `run.py` 的 `python -m xval.mcmp_cross_force.run` 与 `tests/`）= 文档虚标。
- m18289 产物在任何分支里都不存在；审计按 sha256 冻结副本，否则证据随工作树消失。
- `tensor_reports` 该 mission **0 行**；`/api/v1/reports?mission_id=18289` 的 6 条是平台 equipment_status 占位（size=0, file_url=null），非论文报告。
- 上游 lbmpy 测试还含 `g_aa/g_bb`，本交付置 0（纯交叉力）= 范围限制，非完整模型等价。

## ⑥ 误差判定（逐项显式回答）
| 关键量 | 基准 | 测得 | 误差 | <3%? |
|---|---|---|---|---|
| 算符恒等（linear ψ） | 库核 | 交付 OSS 算符 | 3.4e-16 | **是** |
| exp-ψ 组合 | 同上 | 同上 | 2.7e-16 | **是** |
| 模板系数 | 解析 cs²=1/3 | 0.33333334 | 3e-9 | **是** |
| 物理臂剖面对比 | OSS 臂 | 本仓臂 | **200.34%** | **否（超 15% 门 13 倍）** |
| 基准容差判定 | `tensor_benchmark_cases` | — | **无裁判行 ⇒ 未跑** | 不适用 |

**裁定：L1**（数学/单元验证，审计者执行）。L2 否决（无 entry/无运行工件/未提交）；L3 否决（无求解器侧调用方）；
L4 否决（0 裁判行 + 无 repro_cmd）；L5 否决（无外部基准，仅核了 OSS pin 与转写行）。
落表：`tensor_maturity_verdicts` id=740（repo=TensorLBM, mission_id=18348, level=1, verdict_by=board-audit）；
门判定 dry-run：`judge(1)= (1, [])` ⇒ 待 m18348 终态后由 maturity_gate 自动裁定为 **reopened→返修轮2**。
返修最小面：run.py 出 .npz/.json 工件 + pytest 套件 + 为该 anchor 注册裁判行 + 文档只描述真实存在的文件。
