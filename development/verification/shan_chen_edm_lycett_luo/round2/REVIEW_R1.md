# 对抗性复核 — `XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO` 第 1 轮（m18323）

**复核者** m21378（verifier 席）｜**被复核产物**
`TensorLBM/xval/XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO/{REPORT.md, run_xval.py, artifacts/xval_results.json}`
（m18323，判据 <15%，自判 **PASS**, σ 误差 3.9%–7.1%）

**独立性限制**：本轮无法访问 m18323 的运行环境，只能静态审阅其源码 + 用自己的复算对齐数值。
复核所用数值全部由本单 `run_xval.py`（库路径）+ `tests/test_xval_sc_edm_laplace_r2.py` 重新产生。

---

## 结论：**Request changes**（原 PASS 不可引用）

R1 的结论"TensorLBM 的 Shan-Chen 实现与 TCLB 参考一致（3.9%–7.1%）"**没有测到 TensorLBM**，
且其绝对表面张力数值存在 ~2.7×10² 的量纲/约定错误。定性认识（力项选择在 τ=1 下不改结果）
方向正确但归因错误。

---

## 分级发现

### S1（阻断）R1 没有运行 TensorLBM 的代码 —— 结论对象错位

| 证据 | 位置 |
|---|---|
| runner 的 import 只有 `json/math/time/dataclasses/pathlib/numpy/torch/matplotlib`，**无任何 `tensorlbm` 符号** | `run_xval.py:1-11` |
| runner 自建 `_C/_W/_CS2/eq(...)` 等 D2Q9 常量与平衡态，自带 collide/stream | `run_xval.py:12-40` |
| REPORT.md 声称的"TensorLBM 现有实现"路径 `TensorLBM/src/tensorlbm/multiphase/__init__.py + sc_d2q9_interface/` 在仓根包中不存在；实际随仓发布的 SCMP 实现是模块 `src/tensorlbm/multiphase.py`（`collide_sc_single_component` / `sc_single_component_force`） | `REPORT.md` 表格 vs `ls src/tensorlbm/multiphase*` |

**影响**：R1 的五个臂全部是同一个自研 NumPy/PyTorch 模型的参数变体，"TensorLBM vs TCLB" 标称不成立；
其 PASS 只能理解为"两种自研配方互相接近"，属于自证。

### S2（阻断）压力约定与库约定相反 —— 绝对 σ 放大 ~2.7×10²

R1 用 `dp = (rc·cs² + (G/2)cs²ψ²) − (rv·cs² + (G/2)cs²ψ²)`（`run_xval.py:118`），即教科书
`G_eff<0` 约定的 **加号** 形式，且把输入密度当作 ψ 之外还乘 `G`，与库的后向 gather 约定

```
p_mech(ρ) = cs²ρ − (G/2)cs²ψ²        (库内 G = +5 为吸引)
```

**符号相反**。用 R1 自己报告的密度 (ρ_l=1.9725, ρ_v=0.1618) 复算：

| 约定 | p_in | p_out | Δp | σ = Δp·R_eq |
|---|---|---|---|---|
| R1 用的（错号） | 1.2752 | 0.0725 | **1.2026** | **≈15.96**（R1 表值） |
| 库约定（正确） | 0.0398 | 0.0354 | **0.00444** | **≈0.055** |

库文档/`benchmarks/verified` 记录的该模型表面张力为 **0.0561**；本轮独立复算 velocity_shift@τ=1
得 **0.05519**（偏差 1.6%）。即 R1 的 σ≈16 与真实值相差 **≈271 倍**，其"σ 误差 3.9%"
是两臂同用同一错号压力后误差相互抵消，属**共模抵消**，不能作为精度证据。

### S3（阻断）判别设计在 τ=1 下数学退化

velocity-shift 松弛到 `feq(u + τF/ρ)`；EDM 松弛到 `feq(u) + [feq(u + F/ρ) − feq(u)]`。
**τ=1 时两者恒等**——不是"psi_exp 力幅小把差异掩盖了"（R1 的归因），而是恒等式。
本轮数值验证（库路径，`τ=1`，float64）：

| τ | `max|f_vs − f_edm|` | `max|f_vs − f_guo|` |
|---|---|---|
| 1.0 | **4.44e-16** | 9.36e-05 |
| 0.8 | 7.48e-05 | 4.21e-05 |

R1 全部算例 `tau=1.0`（`REPORT.md` 参数表），因此 A/B 两臂**按构造不可能不同**，
其表里 `TL_vs_R12` 与 `TL_edm_R12` 数值逐位相同正是这个恒等式的影子，却被写成"力项影响被掩盖"的物理结论。

### S4（非阻断，但使验收不可判定）"TCLB 参考臂"不是 TCLB

所谓 `TCLB_edm/TCLB_guo` 臂的伪势是自拟的 `p = ρcs² + ρ² − 0.5ρ³`、`ψ = √(2(p−ρcs²)/(G cs²))`
（`run_xval.py:34-35`），并非 TCLB `d2q9_scmp` 的 Carnahan–Starling 伪势（`ψ=√(−p_cs+ρ/3)`），
也未复现其 `gs` 力 stencil。因此"与 TCLB 对齐"无实现依据。

### S5（非阻断）雅可比/拟合统计量不成立，作者已自认

2 个半径拟合 σ(R) 的 R²（记 −63643 等）无统计意义；作者已在 §5.1 记录。
本轮改为 5 半径 + 过原点单参数拟合 + 逐半径 σ 离散度。

---

## 未执行/未能核实的检查

* 未复现 m18323 的原始运行环境（无法确认其 `artifacts/xval_results.json` 与源码同步）；
* 未做 TCLB 本体安装运行（仓内无 TCLB 二进制；仅按源码语义比对）——任何"与 TCLB 数值一致"的说法
  在本仓范围内都只能是**配方级**对标，不是**代码级**对标；
* R1 报告的图（fig1/fig2）未逐像素复核。

## 处置

原 R1 判 **PASS 撤回**，替换为本轮 `REPORT.md`（第 2 轮）的判定：
σ 与解析/库内标定值的一致性 **<3% 达标**；共存密度 vs Maxwell **不达标（诚 FAIL）**，原因归到
SC 伪势的离散化而非本仓实现缺陷（见第 2 轮报告 §⑥）。
建议：把 R1 目录在 `feature_ledger` 中标注为 superseded，避免后续席位再次引用其 σ 数值。