# AUDIT m17627 — 交差核验 m17499 `XVAL-LBM-BENCHMARK-METHOD-SELECTION`（TensorLBM）

> 角色：独立审计席位（board-audit）。对象=mission 17499（状态 succeeded，commit `dcae25e8`，PR #2013 前一代链，
> 报告 tensor_reports id=16517）。本单**只核验、不重做**；唯一新增计算是「复现审计」与「TRT 判别探针」两件取证动作。
> 全部数字由 `artifacts/m17627_audit/audit_compute.py`（读工件）+ `trt_probe.py`（独立探针）程序化产出，
> 落盘 `artifacts/m17627_audit/audit_result.json` / `trt_probe.json`，本文零手抄。
> 被审工作树：`/root/tp-agent/work/m17499`（只读访问，未做任何写入）。

## 0. 判定结论（先给裁断）

**级别 = L3**（真代码 + 驱动主求解器内核 + 报告入平台），**L4 不成立**，L5 不适用。
不成立 L4 的决定性理由不是"报告写得不够"，而是三条可机器复核的事实：

| # | 事实 | 证据 |
|---|---|---|
| F1 | `tensor_benchmark_cases` 对该 anchor **0 行**；仓内已登记的同算例容差策略（`tensor_case_pool` id=116，`<8% St / <10% Cd`）**从未被评估**；`reference.json` 的 Re=200 两行是 `WITHHELD`（`ref: null`） | `psql -U foundation -c "SELECT ... WHERE anchor_slug='XVAL-LBM-BENCHMARK-METHOD-SELECTION'"` → 0 rows；`benchmarks/verified/square_cylinder_re200/reference.json` |
| F2 | 预注册 4 臂 × 分辨率阶梯 {10,14}，**实交 ok 臂 = 2/8**（A1 D10；A2 D10 在 commit 之后才跑完）；A4 从未运行，D=12/14 一次都没跑 | `runs/` 实际内容 vs `predeclared.json:variables_changed/stability_fallback` |
| F3 | 报告唯一的那组对比 **A1 vs A2 不是单变量对比**：A1 `u_in=0.10`（Ma=0.1732），A2 `u_in=0.05`（Ma=0.0866），而 `predeclared.json` 把 `U=0.05 (Ma=0.087)` 写进 *controlled* | `runs/A1_D10_U0.100/result.json`、`runs/A2_D10/result.json`、`predeclared.json:variables_controlled` |

## 1. 三件套取证（run_artifacts / ref-vs-got / repro_cmd）

**(1) run_artifacts —— 存在但与交付件脱钩。**
`runs/A1_D10_U0.100/{result.json,trace.npz}`、`runs/A2_D10/{result.json,trace.npz}`、
`runs/A3_D10*/fail.json`、`runs/A5_D10*/fail.json` 均在磁盘上；但 `runs/` 被 `.gitignore:108` 忽略
（`git check-ignore -v` 命中），**commit `dcae25e8` 里没有任何一份原始结果**，交付件只有 `summary.json`/`table.csv`/doc。
且 `summary.json` 的 mtime 早于 A2 的 `result.json`（06:55:30 commit vs 06:57 落盘）⇒ **A2 的实测证据完全没进交付件**：

```
result_json_on_disk  = [A1_D10_U0.100, A2_D10_U0.050]
in_committed_summary = [A1_D10]
dropped              = [A2_D10]
```

**(2) ref vs got —— 交付件里没有这一行。** 报告 ④ 表只有 A1 一行；⑦「误差判定」表的表头下面 **零行**
（`make_report.py:182-196` 只对 `A2_D10`/`A4_D10` 且仅在 `tbl` 命中时出行；stale summary 里没有 A2 ⇒ 表空）。
审计把 A2 灌回同一 `analyze.py` 重算（`artifacts/m17627_audit/xval_copy/`，未改被审树）：

| 量 | A1 (U=0.10) | A2 (U=0.05) | Δ vs A1 | <15%? | 归因 |
|---|---|---|---|---|---|
| Cd 均值 | 1.5768 | 1.4630 | **−7.22 %** | ✅ | 不可单因子归因（见 F3） |
| Cl 幅值 | 0.5638 | 0.5050 | **−10.44 %** | ✅ | 同上 |
| St (FFT) | 0.16942 | 0.17758 | **+4.81 %** | ✅ | 同上 |
| St 双法自洽 | 0.172 % | 1.239 % | — | — | ✅（阈值 5 %） |

即 H1 若按 15 % 线读是"过"，但**过的这组差值同时夹杂 Ma 0.173→0.087 的变化**，不能支撑报告
「同一内核、同一 BC、只切换选型变量」的因果表述。

**(3) repro_cmd —— 审计独立重跑，可执行且确定性可复现。**
`python xval/benchmark_method_selection/run_arm.py --arm A1 --D 10 --u-in 0.1 --steps 1500 --threads 8`
在本审计容器 76.0 s 跑完 1500 步（50.678 ms/步 vs 交付件 52.087 ms/步），与交付 `trace.npz` 前 1500 样本逐点比对：
`max|ΔFx| = 1.42e-14`、`max|ΔFy| = 1.33e-14`，`cl_max_abs` 交付 1.3573298048516944 / 复现 1.35732980485173
⇒ **差分在 float64 舍入级，solver 确定性成立**；另一臂 `--arm A2 --D 6 --steps 400` 5.2 s 正常出结果（入口对非 A1 臂同样可执行）。
日志：`artifacts/m17627_audit/repro_A1_D10_1500.log`、`artifacts/m17627_audit/repro_A2_D6/result.json`。

## 2. 缺陷清单（按严重度；全部附指针）

### BLOCKING —— 直接决定 L4 不成立
1. **预注册控制变量被违反且未申报**：`predeclared.json` 的 `variables_controlled` 写明 `U=0.05 (Ma=0.087)`，
   而主臂 A1 实测 `u_in=0.10`（`runs/A1_D10_U0.100/result.json`），doc ③ 参数表把 `U=0.1 / Ma=0.1732` 写成算例定义，
   两处互斥且报告没有一行「偏差与理由」。`analyze.py:33-36` 用 `_U{u:.3f}` 后缀把该偏差编码进 key，说明代码知道、报告没说。
2. **主对比被 Ma 混杂**（F3）：A1 与 A2 的 Δ 同时含碰撞算子/出口与 Ma 变化，报告 §②「单差异对比」不成立。
3. **交付报告的核心判据表为空**：§④ 仅 1 行、§⑦ 零行；根因是 `summary.json`/`table.csv` 在 A2 落盘前生成（stale），
   且 `runs/` 被 gitignore ⇒ 交付件里没有任何 A2 证据（`analyze.py:33` 的 load 路径可复算 A2，审计已证）。
4. **可执行指针造假/失效 2 处**：doc §⑥/§⑨ 断言存在结构测试 `tests/test_xval_method_selection.py`（**全树不存在**，
   `git ls-files` 只有 `tests/test_xval_taylor_green.py`）；doc §⑦ 引 `refs/norberg1993.json`（**该路径不存在**，
   真实文件在 `cases/square_cylinder_norberg/refs/norberg1993.json`）。
5. **无裁判**：见 F1。仓内明明有容差策略（case pool id=116，`<8 % St / <10 % Cd`）和 graded reference 表，
   报告却只做"选型互换性"、不做绝对量判定，并把 `WITHHELD` 讲成"本单不可得"——**可做的绝对判定被跳过**。

### MAJOR —— 交付件自身不一致（不影响 imple 真实，但影响可取证性）
6. **两张交付图是空图**：`figures/cl_traces_D10.png` 的全图调色板只有白/灰/黑（top colors 全为 255/236/52/0/127 灰阶，
   无任何曲线色像素）⇒ 该图**没有任何数据曲线**；根因 `analyze.py:150` 去找 `runs/A1_D10/trace.npz`，
   而 A1 实际写到 `runs/A1_D10_U0.100/`，A2 当时也尚未落盘 ⇒ 两条曲线都被 `continue` 跳过。
   `kpi_bars_D10.png` 只有 3 个柱簇（三个子图各 1 根柱，= A1 单臂），`resolution_ladder.png` 只有 382 像素的标记
   ⇒ **阶梯图只有一个 D=10 点，没有任何阶梯**。
7. **聚合器静默丢数据**：`analyze.py:36` 的 fail key 为 `{arm}_D{D}_FAIL`，A3/A5 各有 U=0.05 与 U=0.10 两份 `fail.json`
   ⇒ **key 冲突、每个臂丢一条记录**（现存 A3@U0.05 step 587、A5@U0.05 step 5770 从未进入 summary）。
8. **失败臂标签错配**：`analyze.py:63-65` 的 blowup 行不带 `u_in`，`make_report.py:146` 回退成 `U={f.get('u_in','0.05')}`
   ⇒ 交付 doc §④ 写「A3_D10（U=0.05）第 326 步」/「A5_D10（U=0.05）第 624 步」，而 artifact 里 326/624 属于
   **U=0.10** 的两次运行（U=0.05 的是 587/5770）⇒ 失败臂数字与工件**不可对齐**。
9. **预注册阶梯与实现不一致**：`predeclared.json` 阶梯 {10,14,20}、收敛判据 `|KPI(D=14)-KPI(D=10)|`，
   而 `analyze.py`/logs/图全部按 **D=12** 组织（`logs/A1_D12.log` 等），且 D=12 **零次运行**。
   另：**A5 是预注册之外的 post-hoc 臂**（`predeclared.json` 只有 4 臂），报告未声明其入表规则。

### MINOR
10. **重复实现**：`run_arm.py:71-89,92-119` 自建 `square_mask` / `strouhal_fft` / `strouhal_zero_crossing` / `analyze`，
    而仓内已有 `src/tensorlbm/square_cylinder.py`（兄弟算例 `benchmarks/verified/square_cylinder_re200/` 正是用它）；
    同一算例两套后处理 ⇒ 口径漂移风险（本次未发现数值分歧，但属结构性隐患）。
11. **自比作基线**：ref 用的是本仓 A1 自身，无外部/独立实现裁判（OSS 侧"方法移植"是同一批内核，
    报告 §⑥ 已诚实声明这一点——此项不是造假，而是能力边界）。

## 3. 独立佐证（审计新增证据，非重做）

**(a) 与仓内 graded reference 的 TREND 行对照（明确：TREND 级，不可作 PASS/FAIL 门）**
`benchmarks/verified/square_cylinder_re200/reference.json` 的 Re=200 是 `WITHHELD` + `trend_expectation_re200`
（Cd 1.37 带 [1.30,1.45]；St 0.171 带 [0.16,0.182]）：

| 臂 | Cd got | Δ vs TREND | 在带内? | St got | Δ vs TREND | 在带内? |
|---|---|---|---|---|---|---|
| A1 (U=0.10) | 1.5768 | **+15.10 %** | ❌ | 0.16942 | −0.92 % | ✅ |
| A2 (U=0.05) | 1.4630 | +6.79 % | ❌（带上沿 +0.9 %） | 0.17758 | +3.85 % | ✅ |

含义：被当作"本仓记录选型基线"的 A1 恰是偏离仓内自身趋势最远的一臂，且它的偏差方向与其**未申报的 U=0.10（更高 Ma、更强可压缩性）**一致。
这条不能判 A1 错（TREND 非裁判），但它把"用 A1 当 ref"的做法钉成需要复核的风险点。

**(b) TRT 判别探针（回答 m17499 §⑥ 自己留的开放问题）**
A3/A5 在两档 U 下全发散（A3: U0.05→587 步 / U0.10→326 步；A5: U0.05→5770 步 / U0.10→624 步）。
周期盒 + 1e-3 随机扰动、无边界、无 BC、2000 步、N=16（`artifacts/m17627_audit/trt_probe.py`）：
**8/8 组配置全部稳定**（bgk/mrt/trt(Λ=3/16)/trt(A5 钳位) × tau∈{0.5075,0.515}，`rel_growth ≤ 0.9950`，dev 单调衰减）
⇒ **"collide_trt 内核在 bulk 线性不稳定"这一假设被否定**；A3/A5 的发散只能落在「壁面半程反弹 + `far_field_bc_2d` + tau−0.5≈0.008–0.015 的近不可压极限」
这条耦合路径上（tau_minus=2.0 钳位仍发散也说明不是单纯 Λ 取值的锅）。m17499 把这条列为"未分辨、留给第 2 轮"，
本审计给出**否定性证据**缩小了它的范围（kernel bulk ≠ 失效点）。

## 4. 已查 / 未查（诚实边界）

已查：工件存在性与 git 追踪状态；doc→script→summary 的程序化闭环（**重生成 doc 与交付 doc 逐字节相同，diff=0 行**，
`artifacts/m17627_audit/doc_regen.diff` ⇒ 报告零手抄这一点是**真的**）；三张图的像素级内容；
入口命令可执行性 + 确定性（1500 步逐点比对）；A2 回灌后的 Δ；TREND 对照；TRT bulk 稳定性；
`tensor_benchmark_cases` / `tensor_case_pool` / `reference.json` 三方裁判状态。

未查（不得当作通过）：A1/A2 的**分辨率无关性**（无 D=12/14 数据，拒判）；A4（sponge 效应）——**零数据，无法判**；
对外部实验/文献（Norberg 1993 / Sohankar 1998 原表）的绝对误差——**WITHHELD，拒判**；
OSS 侧（Maarten-vd-Sande / pylbm）真正的代码级独立运行——本审计容器同样无法运行（未尝试装 numba/mpi4py），
即"方法差异 vs 实现差异"仍未分离。

## 5. 处置（disposition）

**request changes（返修）+ 降级至 L3**。返修清单按代价排序：
1. 补跑 A1 于 U=0.05（与 A2 同 Ma），把 A1 vs A2 变成真正的单变量对比；同时把 predeclared 的 U 与 doc ③ 对齐（谁改谁、写偏差与理由）。
2. 补 A4（sponge 关）= 预注册 4 臂里的唯一零数据臂；补 D=12 或按预注册改回 D=14。
3. 修 `analyze.py:36` 的 fail key 冲突、blowup 行带 `u_in`、`analyze.py:150` 的 A1 路径；然后重生三张图（当前两张空图/单点图不可作证据）。
4. 交付件纳入原始 `result.json`+`trace.npz`（或至少把 A2 的 result 嵌进 summary），删掉 doc 里不存在的
   `tests/test_xval_method_selection.py` 与 `refs/norberg1993.json` 两个指针（后者改真路径）。
5. 若要冲 L4：注册 `tensor_benchmark_cases` 行，并按 case pool id=116 的 `<8 % St / <10 % Cd` 对它做判定（缺原表就保持 WITHHELD 并说明，别用 TREND 冒充裁判）。

** overstated claims（报告里不成立的表述）**：
"同一内核、同一 BC、只切换选型变量"（A1/A2 的 Ma 不同）；"分辨率阶梯 D=10→D=12"（无 D=12 数据、图为单点）；
"结构测试 tests/test_xval_method_selection.py"（文件不存在）；"§⑦ 上表逐项给出"（表零行）；
"与成熟开源 LBM 实现的选型交叉对标"（OSS 未被运行，是方法移植，报告 §⑥ 已自陈，但标题层仍过强）。

## 6. 独立性声明（independence limitations）

审计与被审同处一台控制容器、同一 torch 2.7.1/float64 栈；"复现"只能证明**确定性可重跑**，不能证明数值正确性；
本审计没有独立求解器（第三方 LBM 未运行），因此**无法给出绝对精度裁决**——这也是 L4 被拒的另一面。
级别判定基于"功能实质"，与报告篇幅无关；被审 99 行 doc 与三张图未改变级别。
## 附录 — 4000 步延展复现（同一入口的第二次独立重跑）

`artifacts/m17627_audit/repro_A1_D10_4000/`（`--arm A1 --D 10 --u-in 0.1 --steps 4000`，占总长 14871 步的 26.9%）：
与交付 `trace.npz` 前 4000 样本比对 max|ΔFx| = 1.78e-14、max|ΔFy| = 1.42e-14（按 |Fy| 峰值归一 = 2.09e-13），
cl_max_abs 交付 1.35732980485169 / 复现 1.35732980485173，wall_s = 234.8，58.701 ms/步。
结论不变：入口可执行、求解确定性成立（差分在 float64 舍入级），可复现性缺陷不在 solver，而在交付/聚合链。
