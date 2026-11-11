# G21 孤儿模块文献级复核（ddes / prism_solver / hybrid_solver / d3q27_collide）

`selfstart-G21 · mission 14724 · repo TensorLBM · 2026-11-11`

全部数字由 `tools/g21_orphan_verification.py` 程序化注入 `artifacts/g21_orphan_verification/audit.json`，本文件由同一脚本从该 json 渲染；回归由 `tests/test_g21_orphan_verification.py` 钉住。

## 0. 结论（证据分级）

| 模块 | 复核依据 | 数字 | 判定 |
|---|---|---|---|
| `ddes` 屏蔽函数 | Spalart 2006 f_d，240 点 | 最大偏差 1.1e-16；半切换 r_d* 相对误差 8.8e+00 | REAL |
| `ddes` 长度尺度 | Spalart eq.(5) 恒等式 | 最大偏差 0.0e+00（旧头部公式偏差 inf） | 实现 REAL / 文档已修 |
| `ddes` 支路诊断 | 真实长度尺度统计 | 旧口径高报 LES 占比 100.0 pp | 缺陷已修 |
| `ddes` 默认常数 | l_LES vs max(l_RANS) | 0.650 vs 0.0995；LES 可达比例 0.00 | 退化 RANS，已登记 |
| `prism_solver` 壁律 | Reichardt 1951 连续性 | 旧律 y⁺=5 跳变 3.925（79%）→ 复合律连续 | 缺陷已修 |
| `hybrid_solver` | FDM 稳态 / 稳定数 / 占位 | 线性剖面漂移 5.8e-05；λ=3.0e-03；两处占位显式化 | STUB（已显式） |
| `d3q27_collide` | 算术自洽 + 声明溯源 | Re 偏差 0.0e+00；原校验把 Ct 与 ITTC-1957 Cf 对比、工件数 1 | 声明已订正 |

## 1. DDES：与 Spalart (2006) 公式级对拍

- f_d 逐点最大偏差 **1.110e-16**（10⁻⁴…10² 共 240 点）；f_d(10⁻⁴)=1.00e+00、f_d(10²)=0.000000。
- 半切换点实测 r_d*=1.0000000000，解析 (atanh 0.5)^(1/3)/8=0.1023720732，相对偏差 8.8e+00。

| l_RANS | l_LES | f_d | 代码 l_DDES | Spalart eq.(5) | 偏差 | 旧头部 max(l_RANS, l_LES/f_d) | 偏差 |
|---|---|---|---|---|---|---|---|
| 1.00 | 0.65 | 0.00 | 1.0000 | 1.0000 | 0.0e+00 | inf | inf |
| 1.00 | 0.65 | 0.50 | 0.8250 | 0.8250 | 0.0e+00 | 1.3000 | 0.4750 |
| 1.00 | 0.65 | 1.00 | 0.6500 | 0.6500 | 0.0e+00 | 1.0000 | 0.3500 |
| 0.30 | 0.65 | 1.00 | 0.3000 | 0.3000 | 0.0e+00 | 0.6500 | 0.3500 |
| 0.30 | 0.65 | 0.25 | 0.3000 | 0.3000 | 0.0e+00 | 2.6000 | 2.3000 |
| 2.00 | 0.65 | 0.75 | 0.9875 | 0.9875 | 0.0e+00 | 2.0000 | 1.0125 |

实现与 Spalart eq.(5) 一致，但模块头部原公式（max(RANS, C_DESΔ/f_shield)）与实现最大差 **inf**——按头部实现会得到「分离区用 RANS 长度、贴壁区用 LES」的反向行为（若 f_shield 很小则 C_DESΔ/f_shield 更大，max 取到它）。已订正头部。

- 界与单调性：越界 1.1e-16，f_d 单调性违反 0.0e+00，DES 极限 l(f_d=1)=min(l_RANS,C_DESΔ)：成立。

### 1.1 近壁 RANS 长度律不是 van Driest（原注释不实）

| d⁺ | 代码 C_S d/(d+1) | van Driest C_S(1−e^(−d⁺/25)) | 相对偏差 |
|---|---|---|---|
| 1 | 0.05000 | 0.00392 | 1175.2% |
| 2 | 0.06667 | 0.00769 | 767.1% |
| 5 | 0.08333 | 0.01813 | 359.7% |
| 10 | 0.09091 | 0.03297 | 175.7% |
| 25 | 0.09615 | 0.06321 | 52.1% |
| 50 | 0.09804 | 0.08647 | 13.4% |
| 100 | 0.09901 | 0.09817 | 0.9% |
| 200 | 0.09950 | 0.09997 | 0.5% |

仅在 d→∞ 同趋 C_S Δ（d⁺=200 偏差 0.5%），近壁最大差 1175%。属建模选择，未擅改物理，仅把注释改为如实描述。

### 1.2 支路诊断口径（缺陷已修）

旧口径 `les_fraction = 1 − mean(f_d<0.5)` 在合成边界层场上报 1.000；按真实长度尺度（l_DDES==l_LES）统计的 LES 占比 0.000，高报 **100.0 个百分点**（`les_limiter_active`=False）。修复后 `run_ddes_diagnostics` 用 `ddes_branch_fractions` 按长度尺度统计，并新增 `blend_fraction` / `les_limiter_active`。

### 1.3 默认常数下 DES 限幅器永不激活（关键）

- l_LES=C_DES Δ=**0.650**；l_RANS 在 d∈[0.5,200] 上最大值 **0.0995**（上限 = C_S Δ = 0.1）。
- ⇒ max(0, l_RANS−l_LES)≡0，l_DDES≡l_RANS，LES 支路可达比例 **0.00**：默认参数下「DDES」等价于纯 RANS。
- 对照 C_DES=0.05：LES 可达比例 1.00、l_DDES==l_LES 占比 0.00 ⇒ 限幅器按设计工作；根因是常数混用（SA-DES 的 C_DES=0.65 配 Smagorinsky 的 C_S=0.1），不是实现错误。

## 2. prism_solver：壁律连续性（缺陷已修）

- 注入点 y_iface=0.13485，u_τ=0.02，ν=0.01 ⇒ y⁺=0.270；注入 u=0.005280，复合律 0.005280（偏差 2.4e-09）；旧律 0.005394（偏差 2.1e-02）。壁法向分量清零：True。
- 旧两段式：u⁺(5⁻)=5.0000、u⁺(5⁺)=8.9255 ⇒ 跳变 **3.925**（79%），直接注入 LBM 造成近壁速度阶跃。
- 复合律（Reichardt 1951）20001 点采样最大相邻步长 0.0031（量程 0.01%）；粘性极限误差 y⁺=1 0.96%、y⁺=2 0.11%；对数区斜率 2.4159 vs 1/κ=2.4390（0.95%）；u⁺(0)=0.0。

## 3. hybrid_solver：脚手架而非求解器

- FDM 显式步对线性（Couette）剖面：内部层最大变化 8.127e-05（相对 5.8e-05）；全层 2.859e-03（边界两行按设计被 Dirichlet 覆盖）。内部层用统一 h_avg 近似几何增长网格，非均匀间距误差未计入。
- 显式扩散数 λ=νΔt/h_min²=3.00e-03 ≪ 0.5（余量 167×）。
- 占位实现（已改显式）：
  - `FDMBoundaryStep.body_force_from_lbm`：NotImplementedError (explicit)
  - `HybridSolver.drag_coefficient`：NotImplementedError (explicit)
- 结论：模块**不可用**（无 LBM→棱柱体力传递、阻力曾返回 0.0）。登记 STUB，不进可用清单。

## 4. d3q27_collide：配置算术 vs 校验声明

- 算术自洽：Re 重构偏差 0.0e+00；τ 偏差 0.0e+00；ny/nx=0.40、nz/nx=0.40；lattice=D3Q27、C_s=0.0。
- 原声明「SUBOFF Ct=0.00393，2.9% vs ref 0.00405」：0.00405 = ITTC-1957 平板摩擦线 Cf(2×10⁶)=0.004054，与 Ct（总阻力系数）不是同一量；仓内提及 0.00393 的工件数 **1** ⇒ 不可溯源。已改为诚实状态（drag prediction UNVERIFIED），模块定位为参数构造器。

## 5. 改动清单

| 文件 | 改动 | 类型 |
|---|---|---|
| `src/tensorlbm/ddes.py` | 头部公式订正为 Spalart eq.(5)；删除死代码；新增 `ddes_length_fields` / `ddes_branch_fractions`；`DDESResult` 增 `blend_fraction`、`les_limiter_active`；支路分数改为按长度尺度统计 | 缺陷修复 |
| `src/tensorlbm/prism_solver.py` | 两段式壁律 → Reichardt 复合律 `uplus_reichardt`（连续） | 缺陷修复 |
| `src/tensorlbm/hybrid_solver.py` | 两处静默占位（体力返零、阻力返 0.0）→ `NotImplementedError` | 缺陷修复 |
| `src/tensorlbm/d3q27_collide.py` | 删除不可溯源校验声明，标注 UNVERIFIED / 参数构造器定位 | 文档订正 |

## 6. 局限

- 全部为符号/解析级复核（函数级），**不含端到端模拟**：DDES 的 ν_t 未与任何湍流算例对标；首行「DES 限幅器退化」是常数标定问题，改成可用模型需要重新标定 C_DES（未做，属建模决策）。
- `hybrid_solver` 的 FDM 步只验证了稳态保持与稳定数，未验证与 LBM 的耦合精度（耦合并未实现）。
- `d3q27_collide` 只验证参数构造，不构成对 D3Q27 阻力能力的任何验证。
- 机器：本机 128 核，load15 ≥ 100 时按容量纪律只做轻量活，本复核全程 CPU 单线程、秒级。
