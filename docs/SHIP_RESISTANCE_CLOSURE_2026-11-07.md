# 船型模型尺度阻力闭合（KCS / DTMB 5415 / Series 60）— 项目级结论 2026-11-07

> 案例目录：`benchmarks/pending/ship_resistance_closure/`（⛔ 未晋升，诚实 FAIL）
> 代码：`src/tensorlbm/ship_resistance_closure.py`（新增库模块）
> 测试：`tests/test_ship_resistance_closure.py`（24 passed）
> 一键复现：`PYTHONPATH=src python benchmarks/pending/ship_resistance_closure/run.py`（~100 s CPU）

## 1. 为什么做

战役要求「KCS/DTMB 5415/Series 60 × Fn=0.2/0.3/0.4 的 C_T、C_F、C_W 与实验/ITTC 比较」。
仓库已知状态（`benchmarks/STATUS.md`）是：低 Re 层流摩擦口径已验证到 Re=10³，
模型尺度（Re~10⁷、含兴波）**结构性不可达**，两者之间缺一层**可审计的记账**。
本任务补的正是这层记账：把 `C_T = (1+k)C_F + C_W` 拆成三级证据，
每一级各自可以/不可以支撑什么结论，由几何、数值与门限共同判定，而不是靠叙述。

## 2. 做了什么

新增 `tensorlbm.ship_resistance_closure`：

* **GEOMETRY**：半宽图 `b(x,z)`（解析参数化或 STL 三角面采样）→ `nabla / Cb / Cm / Cp / Cwp / S_wet`；
  对锁定 Cb 失败即 `geometry_rejected`（fail-closed）。Series 60 满度族 `p` 由二分标定到
  目标 Cb（`|ΔCb| < 1e-7`），`S_wet` 在 nx=801×nz=641 的细网格上积分并作为 override 固化，
  避免归一化面积随地图分辨率漂移。
* **CORRELATION**：ITTC-1957 `C_F`（复用 `hydrodynamics.ittc57_friction_coefficient`）
  + Holtrop-Mennen (1984) 形状因子（含 lcb/Cstern 显式假设与敏感性区间）。**不冒充模拟**。
* **THEORY**：Michell thin-ship 三重积分（Havelock 形式，对任意半宽图），
  4×/2×/1× 加密 + Roache GCI + **轴向相位分辨率诊断**（`k0 sec(θ) dx > π/8` 时拒绝出数，
  防止低 Fn 混叠），以及**薄船适用性门**（L/B、Cb、B/L、T/L，按实测几何判定）。
* **判据**：分解恒等式 `|C_T − [(1+k)C_F + C_W]|/C_T < 1e-6`、`band_verdict` 失败即关闭、
  `monotone_increasing` 严格单调、参考先锁后跑（`refs_lock.json` 的 sha256 写入结果）。

## 3. 结果（全部机器注入 `results.json`）

* 几何审计：KCS Cb 0.6455（−0.78%）、5415 Cb 0.4993（−1.51%）、Series 60 三个满度成员
  Cb 误差 <1e-7、Wigley Cb 0.4444（闭式 4/9）。**几何门全部按 ±2% 判定**。
* 波阻腿：5415 `C_W=1.836e-3`（GCI 4.24%）、Series 60 Cb=0.60 `C_W=3.202e-3`（GCI 1.45%）、
  Wigley `C_W=1.641e-3`（GCI 3.31%）。
* 分解自洽残差（8 样本）：**0.0**。
* C_T(Fn)：完整阶梯 2/2 严格单调（Series 60 Cb=0.6：0.00459→0.00658→0.01219；
  Wigley：0.00449→0.00549→0.00593）。DTMB 5415 的 Fn=0.30 因最粗加密级非单调而**拒发**。

## 4. 误差判定（逐项显式回答「是否 <3%」）

| 关键量 | 基准 | 偏差 | <3%？ |
|---|---|---|---|
| C_T(DTMB 5415, Fn=0.28) | INSEAN 模型试验 4.2e-3（同面积口径） | **+26.5%** | 否 |
| C_T(DTMB 5415, Fn=0.28) | 同上，换算到参考 S 口径 | **+44.2%** | 否 |
| C_T(Series 60 Cb=0.60, Fn=0.30) | Todd 表 7.6e-3 | **−13.4%** | 否 |
| C_W 波阻（5415/S60/Wigley） | 无实验基准（库内无实测 C_W） | GCI 1.45–4.24% | 数值收敛 <3% ✓，物理对标无基准 |
| Cb（五型） | 锁定首手/二手主尺度 | 0.00–1.51% | 是（几何门 ±2%） |
| KCS C_W（诊断，门外） | 实测残差 C_R≈0.3e-3 | 6.86e-3 → **约 20×** | 否（几何不可用的量化证据） |

结论：**数值层（几何、收敛、恒等式）达到 <3% 级；物理层（与实验的 C_T 对比）不达 3%，
也不达战役给的 8%/10% 带**。不粉饰。

## 5. 局限与归因

1. **几何保真度是主因**：参数化 KCS 的 Cwp=0.791 vs 实船 0.667（+19%），thin-ship 源强被系统性放大。
2. **一阶理论系统偏差**：5415 +26.5%（Fn=0.28 波峰区高估），S60 Cb=0.6 −13.4%（低估），
   误差变号 → 不能用一个校正因子吸收。
3. **LBM 腿缺失**：可达网格的壁链只交付**分辨率 Re** 的摩擦（`suboff_highre` 的
   `Ct_fric/ITTC≈2.5`），自由面+壁函数的准入矩阵在冷路径 fail-closed；
  因此本模块不把相关式结果记在任何求解器名下。
4. **参考口径**：5415 参考值用 IIHR `S/L^2=0.1302`，本仓 STL 三角面积给 0.1483（含声纳导罩、面板化），
   两种口径都列出，不做有利挑选。

## 6. 后续（晋级路径）

首手型值（Tokyo-2015 offsets / Todd 型值表）→ Cwp<2% ⇒ 重跑；波阻换 Rankine 面元或
ITTC-1978 三因次外推并在 Series 60 上对标实测 C_W；壁链交付模型 Re 摩擦后，才允许把
LBM 放进 C_T。在此之前案例停留在 `pending/`，不得晋级。