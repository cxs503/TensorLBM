# 交差审计 · m18096 `XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY`（TensorLBM）

审计单：m18149（`board-audit`）｜被审单：m18096（`xval-lbm-les-scalar-q-abs-cc-proxy-r2`，succeeded）
判定：**L2**（已落表 `tensor_maturity_verdicts` id=691；gate 机器复核模拟 `judge(2, evidence, …) -> (2, [])`，实机 gate 每 30 min 轮询 pending 行裁定）

---

## ① 案例介绍（被审对象）

对象是 OSS 仓 `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow`（LatFlow，MIT）里一个
**非标准 Smagorinsky 闭包**：亚格子黏性不由精确非平衡应力 `Π^neq:S` 驱动，而由标量代理

```
EEk_i = (|c_ix|+|c_iy|+|c_iz|)^2      (Domain.py:90-93 的双重循环)
Q     = sqrt(2 Σ_i (f_i - f_i^eq)^2 EEk_i)                    (Domain.py:121-124)
τ_eff = ½(τ0 + sqrt(τ0² + 6 Q Sc/ρ)),  Sc = 0.17              (Domain.py:57, 125)
```

m18096 的交付=把该闭包移植进 `tensorlbm.turbulence`（D2Q9/D3Q19 两套 + 包级导出），并用
「OSS 逐行转写臂 vs 移植臂」+「闭包代数」两条腿验证。

| 量 | 值 | 出处 |
|---|---|---|
| 格子 | D2Q9 / D3Q19 | `src/tensorlbm/turbulence/turbulence.py:2020,2039` |
| `Sc` | 0.17（OSS 默认，未标定） | `Domain.py:57` |
| `τ0`（E1） | 0.505（近极限低黏） | `run_xval.py` E1 |
| 应变幅值（E3） | \|S\|=0.02，24 个方向 | `predeclared.json` |
| 数据类型 | CPU float64，torch 2.7.1 | `out/kpi_prim.json` |
| 物理锚点 | Ghia et al. 1982 Re=1000 方腔 | **not_run** |

## ② 怎么做的（审计方法，非重做）

1. **三件套取证**：`impl_files` 逐符号定位到行号；`run_artifacts` 逐文件核大小/sha256；`repro_cmd` 实跑。
2. **隔离复跑**：把 `tests/test_xval_les_qabs_cc_proxy.py` + `xval/XVAL-…/{run_xval.py,reference.py}` 复制到
   `/tmp/m18149_audit`，`PYTHONPATH=<被审工作树>/src` 执行，**不写被审工作树**。
3. **外部锚点逐行校源**：直接读 `/tmp/lbtf/LatFlow/Domain.py`、`D2Q9.py`（非二手转述）核对 `Sc`/`EEk`/`Q`/`τ` 与权重速度表。
4. **独立重算**：`docs/audit_m18096/independent_check.py` 不 import tensorlbm、不复用 `reference.py`，
   在纯 numpy 里自建 D2Q9 + CE 扰动，2000 个随机不可压应变上重算代数恒等式。
5. **机器裁判查询**：`SELECT case_name,metric,tolerance … WHERE anchor_slug='XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY'` → **0 行**。
6. **判级后落表 + gate 复核模拟**（`maturity_gate.judge`）。

## ③ 计算结果（数字表，全部由本审计程序化产出/复核）

| 项 | ref | got | 误差 | 结论 |
|---|---|---|---|---|
| K1 `Q` 场，7 个状态（复跑） | OSS 转写 | 移植 | `0.000e+00` | 逐位相等（`torch.equal`） |
| K2 单步碰撞后 `f`（复跑） | 同上 | 同上 | `0.000e+00` | 逐位相等 |
| K2b 出厂限幅臂 vs ref | 同上 | 同上 | `0.000e+00` | 限幅未触发，`τ_eff∈[0.5050,0.6619]` |
| E3 `a = Q̄/(3ρτ\|S\|)`（24 向） | 闭式 `√2/9 = 0.1571348402636772` | `0.15713484026367674 … 0.15713484026367788` | rel 3.5e-15，展宽 **7.24e-13 %** | 通过 |
| **E3 独立重算（本审计）** | `S2 = \|S\|²/81`，`Q/(ρτ\|S\|)=√2/3` | 2000 个随机应变：`max\|S2/(\|S\|²/81)-1\| = 8.88e-16`；`max rel dev(a vs √2/9) = 5.30e-16` | — | **恒等成立，非分辨率效应** |
| `C_s^impl` | 常数 | `0.23804761428476…`（max/min = 1.0000000000000002） | — | 通过 |
| 场量同一性（fig3，64² TGV 全场） | `√2/9` | **5.8295512079842e+13**（max rel dev） | — | **❌ 报告标 ✅，见 ⑤** |
| E4 Ghia Re=1000 | Ghia 1982 | **not_run** | — | **未跑（唯一物理精度腿）** |
| 复跑一致性 | 被审 `out/kpi_prim.json` | 本审计重跑 | 叶子键 **241/241 全等，0 差异** | 可复现 |
| 测试 | — | `9 passed in 6.31 s` | — | 可复现 |

## ④ 结果图

`docs/audit_m18096/fig_audit_m18096.png`（`make_figure.py` 生成）

- **A 面板**：24 个应变方向上实测 `a(φ)` 与闭式 `√2/9` 及 ±15% 判据带的比较——24 个点全部落在带内且与闭式线重合（展宽 1.1e-15）。
- **B 面板**：本审计独立的 numpy 程序在不同 `|S|` 下给出 `Q/(ρτ|S|) = 0.47140452079103173 = √2/3`，偏差 ~1e-16，
  说明 `a=√2/9` 是**代数恒等**而非网格/分辨率效应。

（被审单的 fig1/fig2/fig3 只能看；fig3 的生成脚本在工作树里不存在，见 ⑤。）

## ⑤ 注意点（踩到的坑 / 局限）

1. **交付物从未提交**：m18096 结果 JSON `head_commit=null`，Git 状态为 ` M src/tensorlbm/turbulence/turbulence.py`、` M src/tensorlbm/__init__.py`、`?? xval/…`。
   在全新检出 `/root/tp-agent/work/m18149` 里 `git log --all -S les_scalar_q_abs_cc_proxy` **0 提交**、`grep src/` **0 命中**
   ⇒ **能力不在仓库里，只存在于临时工作树**。本 verdict 的 evidence 因此用绝对路径指向被审工作树。
   *返修建议*：把该移植 commit 进 TensorLBM（若要计入生态能力账，这步是硬门槛）。
2. **报告 ④ 表格把 5.8e+13 的相对偏差标成 ✅**（"场量同一性 fig3"）：在 `|S|→0` 的格点上是比值放大，
   不是通过任何容差；且 `kpi_final.json` 不由 `run_xval.py` 产出（只写 kpi_prim/flow/cavity/kpi），
   `grep -rn field_identity --include=*.py` 全树 0 命中 ⇒ 该数字**无生成脚本可溯**。
3. **K1/K2 = 0.0 是"保真"不是"精度"**：参考臂是同一公式的第二次逐行转写（`torch.equal`），
   0 误差由构造决定；报告 ⑥(iii) 已声明参考臂非 OSS 二进制黑箱比对，但表格 ④ 的 ✅ 会被读成"对标通过"。
4. **唯一物理精度腿（E4 Ghia）not_run**，报告 ⑥ 如实申报（会话墙钟截断，`/tmp/xval_flow.log` 未留存）。
5. 无 `tensor_reports` 行（`report_id IS NULL`）：六要素报告只以工作树内 markdown 形式存在，未进报告库。
6. D3Q19 只做了有限性/形状测试，3-D 各向同性代数未验（报告自陈）。
7. 适用范围：代数恒等只对不可压应变（`tr S=0`）成立。

## ⑥ 误差判定（逐项，并显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| `Q` 场 / 碰撞后 `f`（移植保真） | OSS 转写臂 | 0.0（逐位） | **是**（但为构造性 0，非物理精度） |
| `a = Q̄/(3ρτ\|S\|)` 各向同性 | 闭式 √2/9 | 3.5e-15（实测）/ 5.3e-16（独立重算） | **是** |
| `Q/(ρτ\|S\|)` 线性系数 | 闭式 √2/3 | ~1e-16 | **是** |
| `C_s^impl` 方向无关性 | 常数 0.238048 | max/min-1 = 2e-16 | **是** |
| 场量同一性（fig3） | √2/9 | 5.8e+13（比值放大，无脚本可溯） | **否**（报告标 ✅ 属overstated） |
| **Ghia Re=1000 中心线** | Ghia 1982 表值 | **not_run** | **无法回答——未跑即未通过** |

**总结**：实现保真（0 误差）与闭包代数（√2/9，本审计独立复算到 5e-16）两项**扎实达标**；
但**物理精度（Ghia <15%）未取得任何证据**，且交付物未入库 ⇒ 实质=**L2 独立功能实现**，
不足以判 L3（无集成调用方）/ L4（无注册基准、无完整物理算例误差分析）。

## 判级与整改清单（返修优先级）

| 级别 | 判据 | 本单 |
|---|---|---|
| L1 数学/单元验证 | 测试 | ✅ 9/9（独立复跑） |
| L2 独立功能实现 | 真代码 输入→计算→输出 | ✅ 5 函数 + 包导出（工作树内，未提交） |
| L3 系统集成 | 主求解器/平台调用方 | ❌ `grep` 无 solver/pipeline 调用方 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | ❌ 注册基准 0 行；Ghia not_run；0 误差是自比 |
| L5 科研工程成果 | 新物理 | ❌ 已知启发式闭包的忠实移植 |

整改：①`git commit` 入库；②跑 E4（Ghia Re=1000，65²/97²，上集群）并给中心线 RMS/最大偏差；
③把 `fig3`/`kpi_final.json` 换成有脚本可溯的点wise 比较（对 `|S|>阈值` 的格点统计，避免比值放大）；
④把 E3 闭式写进 docstring/回归测试（已部分）。