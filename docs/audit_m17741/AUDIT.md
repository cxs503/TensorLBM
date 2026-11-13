# 交差审计：XVAL-LBM-CHAPMAN-ENSKOG-AUTO（m17672）— m17741 board-audit

- **被审对象**：mission 17672，`TensorLBM`，anchor `XVAL-LBM-CHAPMAN-ENSKOG-AUTO`，status=succeeded
- **被审工作树**：`/root/tp-agent/work/m17672`，HEAD `15a9b393`（"full sweep 35/35 pass; gate no longer divides by a zero reference"）
- **审计人**：m17741（board-audit），2026-11-13
- **独立性声明**：审计人不在被审工作树内改动任何文件；所有数字取自被审工件 + 我自己在 `/tmp/xval_fresh` 的重跑与 `docs/audit_m17741/probe_m17741.py` 三个探针。审计工作树 `m17741` 基于 `9bcc6db6`（m17672 的两个提交尚未进入我的树），因此复跑用 `TARGET_REPO=/root/tp-agent/work/m17672` 显式指认。
- **裁定**：**L2 confirmed**（L3 / L4 rejected，理由见 §4）；disposition = `approve with notes`。

---

## 1. 案例介绍（物理背景）

线性化 Chapman–Enskog 展开给出格子 Boltzmann 输运系数：剪切黏度 `nu = cs2 (tau − 1/2)`、体积黏度 `zeta`、以及阻尼声波的 `nu_L = 2(D−1)/D nu + zeta`。仓库此前把这些公式**手抄**在十余处（`chapman_enskog.py:5-8` 自述），每处只针对一种格子/一种碰撞算子。m17672 的交付物是把这条推导**自动化**：输入 = (格子速度集 c_q、权重 w_q、碰撞算子的特征基 B 与松弛速率 s_i)，输出 = 精确符号 nu / zeta / cs2；再用仓库自身求解器**实测**同一批系数做交叉验证。

参数（格子单位，`result.json` + `table.csv`）：D2Q9 / D3Q19 / D3Q27；BGK `tau ∈ {0.6, 0.8, 1.2}`；MRT `s_e ∈ {1.2, 1.64, 2.0}`；网格 n32 / n64；剪切波 n32×32（1200 步）、声波 n32/n64（400 步）；振幅 1e-3（剪切）/1e-4（声）。

## 2. 怎么做的（方法与工具链）

1. **取证**：读 `benchmarks/xval_chapman_enskog_auto/{result.json, table.csv, convergence.csv, chapman_enskog_xval.png}`、`src/tensorlbm/chapman_enskog.py`、`tests/test_chapman_enskog.py`、`docs/XVAL-LBM-CHAPMAN-ENSKOG-AUTO.md`、`tensor_reports` id 16658/16649。
2. **基准裁判**：`SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CHAPMAN-ENSKOG-AUTO'` → **0 行**（`ILIKE '%CHAPMAN%'` 亦 0 行）。→ 机器裁判对该锚点**无登记判据**，无法用登记容差裁；只能用"工件自带 15% 门 + 我的重跑"作为替代裁判，并如实记录该缺口。
3. **复现**：`PYTHONPATH=src python3 benchmarks/xval_chapman_enskog_auto/run_xval.py --outdir /tmp/xval_fresh` → 35/35 PASS、worst `E1-sound-atten-3d-n64` = −1.155%、与随件 `result.json` 逐检查 `got` 差 **0**（阈值 1e-12，35 项全等）。
4. **单测**：`pytest tests/test_chapman_enskog.py -q` → **14 passed / 45.01 s**。
5. **三个独立探针**（`docs/audit_m17741/probe_m17741.py`，结果 `probe_m17741.json`）：
   - **P1 门有鉴别力**：同一次实测（`tau=0.8, nu_meas=0.1002063, r²=1.0`）对匹配预测 0.1 → 0.206% PASS；对错配预测（按 `tau=1.2` 推 0.23333）→ **57.05% FAIL**。门不是装饰。
   - **P2 非硬编码**：把 D2Q9 的东向权重 ×1.10 后重归一，推导直接 `raise "isolation of zeta failed: diagonal coefficients differ"` —— nu/zeta 走的是矩空间代数而非重抄公式（另：符号值随碰撞算子变，BGK `zeta=1/30` vs MRT `zeta=3/82`，后者不是能"抄"出来的）。
   - **P3 溯源**：用 HEAD 的模块重算 5 条 derivation 记录，`nu_rel_diff` 全 0，`zeta` 全等；`:571` 的 `chapman_enskog_sha256=b9a42507…` 与 HEAD 文件一致。
6. **集成检索**：`grep -rn 'chapman_enskog' src/` → 模块外 0 处导入；`two_level_amr2d.py:107/111/239` 的 `"chapman_enskog"` 是**无关**的 neq_scale 字符串模式；`src/tensorlbm/__init__.py` 未导出该模块。

## 3. 计算结果（数字表，程序化注入；来源 `docs/audit_m17741/audit_stats.json`）

| 项 | 值 |
|---|---|
| 判据数（checks） | 35（`A 剪切2D 6 / B 剪切2D-MRT 9 / C 剪切3D 6 / D 声2D 10 / E 声3D 4`） |
| 随件通过 / 我复跑通过 | **35/35 / 35/35** |
| 门限 | 15%（owner 精度标准） |
| 最差 | `E1-sound-atten-3d-n64`：ref 0.2 vs got 0.197690 → **−1.155%** |
| <3% 的检查数（论文级线） | **35 / 35** |
| <0.4% 的检查数 | 31 / 35 |
| 误差中位数 | 0.1001% |
| 复跑与随件数值漂移 | **0 / 35**（>1e-12 的项为 0） |
| 不同 got 值个数 | 31（35 个 check ≠ 35 个独立测量；B 臂 9 个 check 只含 3 个 distinct 值） |
| 机器裁判登记行 | **0**（`tensor_benchmark_cases` 无该锚点） |
| 集成调用点 | **0** |

典型逐项（取自 `table.csv`）：

| check | ref | got | err |
|---|---|---|---|
| A-shear2d-tau0.6-n64（nu D2Q9 BGK） | 0.0333333 | 0.0333590 | +0.077% |
| B-shear2d-mrt-tau0.8-se1.64（nu） | 0.1 | 0.100370 | +0.370% |
| C-shear3d-tau1.2-n64（nu D3Q19） | 0.233333 | 0.233153 | −0.077% |
| D2-sound-atten-bgk-n64（nu_L） | 0.2 | 0.200156 | +0.078% |
| D4-zeta-mrt-se2（zeta，ref=0） | 0 | −3.584e−05 | −0.036%（按 nu=0.1 尺度，非相对误差） |
| E1-sound-atten-3d-n64（nu_L D3Q19） | 0.2 | 0.197690 | **−1.155%** |

图：`docs/audit_m17741/audit_m17741.png`（三个面板：(a) 35 项逐项 |误差| 对 15% 门与 3% 线的 log 柱图；(b) ref–got parity；(c) 真正跑过的两网格序列，并标出 3D 声学臂 n32→n64 误差**增大**的非单调点）。图由 `make_audit_figure.py` 从 `result.json` 程序化生成，无手抄。

## 4. 判级（L0–L5）

| 级 | 判据 | 结论 | 依据 |
|---|---|---|---|
| L0 | 文档成果 | 超 | 有真实代码与实测 |
| L1 | 数学/单元验证 | **满足** | 14 单测通过；符号/解析恒等式；复跑逐位一致 |
| L2 | 独立功能实现（输入→计算→输出） | **满足** | `chapman_enskog.py:423` 真做符号 CE：f1 = −B diag(σ) B⁻¹ (∂_t1 + c·∇1) f_eq → 应力矩 → nu(:519)/zeta(:537)，带守恒与可解性断言（:483-500, :527-536），断言在扰动权重下真的开火（P2） |
| L3 | 系统集成（接入主求解器/平台） | **不满足** | `grep` 0 处生产调用；未从 `__init__.py` 导出；无任何求解器/流水线引用 |
| L4 | 基准验证（完整算例+误差分析+可复现） | **不满足（按认证口径）** | ①`tensor_benchmark_cases` 该锚点 0 行——无登记判据可裁；②"基准"是交付物自带 harness（ref/got 确为两条独立代码路径，比"同一闭式抄两遍"强），但算例是仓库自身求解器的周期剪切/声衰减，无外部参考；③35/35 依赖**事后**改门（见坑 1，改前 34/35），且最差点落在非收敛的 n32→n64 序列上 |
| L5 | 科研工程成果 | 不满足 | 无实验/外场数据锚 |

> 级别按"功能实质"判：本件证据质量明显高于 L2 门槛（35 项可复现、最差 1.155%、全项 <3%），但能力形态是**独立验证工具**而非接入求解器的新能力，且无裁判登记；因此 L3/L4 记 withheld，不随报告篇幅升级。

## 5. 注意点（踩到的坑 / 局限 / 适用范围）

1. **坑：门是事后改的。** 交付 HEAD `15a9b393` 的 commit message 自述 "Previously 34/35 with the only failure being nan/nan on the zero-reference row" ——即"35/35"是在第一次失败**之后**重新定义误差（ref 恰为 0 时改用 nu 尺度）才得到的。改动本身站得住（对精确 0 参考算相对误差无定义，且实测 −3.58e−05 对 0.1 尺度是 0.036%），但口径变更只写在 commit message 里、没写进工件字段，**外表看是 35/35，实际受影响的是那 1 行**。审计据此把 35/35 读作"35 项在声明门内通过，其中 1 项采用事后声明尺度"。
2. **坑：check 数 ≠ 测量数 ≠ 能力数。** 35 个 check 只有 31 个不同 got；B 臂 9 项仅 3 个独立实测（nu 与 s_e 无关本是正确的物理结论，但被计成 9 项）；D 臂 6 项由同样 3 组声波+剪切实测重组得出（`zeta = nu_L − nu`，一次测量被复用、且相减放大不确定度）。KPI 记账请用"臂族/独立测量"而非 check 数。
3. **坑：3D 声学臂没收敛。** `E1` n32 误差 0.010% → n64 1.155%，**误差随网格变粗反而变小**的假象说明该序列不单调，最差数字落在未收敛臂上。`convergence.csv` 只给了两格，够不上"网格收敛"论证。
4. **局限：线性响应范围。** nu/zeta 定义在 u→0、∇ρ=0；O(Ma·Kn) 余项只作了符号报告，未数值验证。
5. **局限：各向异性格子。** 扰动权重会让推导直接报错（P2）——对 D2Q5 类各向异性四阶矩的格子，模块选择拒绝而非给错数（`fourth_order_isotropic=False`），范围清晰但覆盖不广。
6. **局限：**未跑 `pylbm` 本体做第二独立推导（缺 mpi4py）；D3Q27 只有符号结果无数值实测（报告 §6 已如实登记）。
7. **溯源瑕疵：** `result.json` 里 `provenance.git_head=e53d994d`，而工件随 `15a9b393` 落库（晚一个提交）。模块 sha256 与数值都与 HEAD 一致（P3 复算 0 漂移），属记账瑕疵而非数值问题，但只读审计者无法从工件本身判断是哪个提交跑出来的。

## 6. 误差判定（逐项 vs 基准，显式回答 <3%）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| nu（D2Q9/D3Q19 BGK，6 组 tau×网格） | 符号 CE `cs2(tau−1/2)` | 0.077% – 0.313% | **是** |
| nu（D2Q9 MRT，9 项） | 符号 CE（MRT 基） | 0.312% – 0.804% | **是** |
| c_s（D2Q9/D3Q19） | `sqrt(Σ w c_x²)` | 0.0003% – 0.059% | **是** |
| nu_L（2D BGK/MRT 声衰减，10 项） | `2(D−1)/D nu + zeta` | 0.019% – 0.190% | **是** |
| zeta（MRT，s_e=1.2/1.64/2） | 符号 CE zeta | 0.036% – 0.190%（ref=0 项按尺度） | **是** |
| nu_L（3D D3Q19 声衰减 n32/n64） | `(4/3)nu + zeta` | 0.010% / **1.155%** | **是**（但 n64 未收敛） |
| 整体 | 15% owner 门 | 35/35 通过，最差 1.155%；中位 0.100% | **是；且全 35 项 <3%** |

**与 owner 精度标准对照：** 全部 35 项 <3%（远超 15% 门），无一项需要用聚合分数掩盖。唯一需要打星号的是 `E1-sound-atten-3d-n64` —— 数值 1.155% 合格，但该臂网格序列非单调，"合格"不等于"已收敛"。

**基准裁判缺口（must-fix）：** `tensor_benchmark_cases` 对该锚点为 0 行，本次判级的"基准"实质由交付物自带 harness + 我的独立复跑构成；建议按 `verdict` evidence 里的 remediation 登记裁判行，否则后续 maturity_gate 只能信任作者的 `gate_pct`。

## 7. 处置

- **disposition**：`approve with notes`；**level = 2**（confirmed 建议），L3/L4 promotion rejected。
- **台账纪律**：不得把本件记账为"接入求解器的能力"；KPI 记账用臂族/独立测量数（5 臂 / 31 个不同实测）。
- **返修清单**（非阻塞，orchestrator 可直接派生）：① 登记 `tensor_benchmark_cases` 裁判行；② `git_head` 用实际运行的树并重生成工件；③ 把 zero-ref 误差规则写进工件字段；④ 3D 声学臂补 n48/n96；⑤ 要上 L3 就让求解器/审计路径调用 `derive_equivalent_equation` 并从 `__init__.py` 导出。
---

## 8. 机器复核（maturity_gate）预校验

用 gate 守护**自己的判据函数**（`/root/tp-agent/maturity_gate.py:judge`，只读调用，不改 gate 状态）对本 verdict 的 evidence 预跑：

```
judge(2) -> (2, [])      # L2 所需 impl_files/entry/run_artifacts 全部机器可解析 -> 裁定 2
judge(4) -> (2, ['L4 缺: integration_callers(grep 无引用)',
                 'L3 缺: integration_callers(grep 无引用)'])
```

即：若把这件声明为 L4，gate 会以 `integration_callers(grep 无引用)` 降到 2 —— 与 §4 的判级独立一致。故本 verdict 声明 level=2，落地即为 confirmed（gate 待本审计单终态后由 30min 周期守护裁定，当前 `gate_state=pending` 是"审计单在飞"的正常等待，不是卡住）。

evidence 路径策略（`evidence.path_note`）：m17672 的提交尚未进 `/root/tensor-ecosystem/TensorLBM`（ECO HEAD `c05d7af3` 早于 m17672），所以 `impl_files/entry/run_artifacts` 用**绝对路径**指向被审工作树 `/root/tp-agent/work/m17672/...`，本审计分支自带的工件用 `docs/audit_m17741/*` 相对路径；push-merge 后同义相对路径 `src/tensorlbm/chapman_enskog.py`、`benchmarks/xval_chapman_enskog_auto/*` 才成立。
