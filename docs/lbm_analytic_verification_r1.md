# TensorLBM 基础解析验证套件：从 CI 隔离区回收 + 四个案例的误差账 (r1, m3643)

日期：2026-11-03 · 分支 `agent/m3643` · 数据源 `artifacts/lbm_analytic_verification_r1.json`
（由 `scripts/lbm_analytic_verification_report.py` 程序化生成，本文所有数字均抄自该 JSON、
未手工修改；图 `docs/assets/lbm_analytic_verification_r1.png` 同源生成）

## 1. 案例介绍（物理背景与设置）

D3Q19 BGK 内核（`tensorlbm.solver3d.collide_bgk3d` / `stream3d`、`boundaries3d.bounce_back_cells_3d`）
的四条「有闭式解」基础链路，共同构成「碰撞 → 迁移 → 边界 → 体力」的最小完备验证面：

| 案例 | 物理 | 解析解 | 被验证的代码路径 |
|---|---|---|---|
| A 剪切波 | 周期性剪切波衰减 | $u_y=A_0\cos(kx)e^{-\nu k^2 t}$，$\nu=(\tau-\tfrac12)/3$ | 碰撞黏性 $\nu$（BGK 松弛率） |
| B Couette | 底板静止、顶板匀速拖曳 | $u(y)=U(y-0.5)/(n_y-1.5)$ | 半程反弹 + 无滑移 + 运动壁面平衡态入口 |
| C Poiseuille | 通道压力驱动（等效体力 $G$） | $u(y)=\dfrac{G}{2\nu}(y-0.5)(H+0.5-y)$ | 体力源项 + 双壁反弹 + 剖面形状 |
| D 摩擦阻力 | 与 B 同一流场，取壁面剪切 | $\tau_w=\nu\,du/dy$，$C_f=2\nu/((n_y-1.5)U)$ | 壁面切应力 / $C_f$ 后处理链 |

参数表（全部来自代码默认值，未为达标调参）：

| 量 | A | B | C | D |
|---|---|---|---|---|
| 网格 $(n_z,n_y,n_x)$ | $4\times4\times64$ | $4\times12\times80$ | $4\times12\times80$ | $4\times12\times80$ |
| 步数 | 100 | 2000 | 3000 | 3000 |
| $\tau$ | 1.0 | 1.0 | 1.0 | 1.0 |
| $\nu$ | 1/6 | 1/6 | 1/6 | 1/6 |
| 速度尺度 | $A_0=0.01$ | $U=0.05$ | $u_{max}=0.05$ | $U=0.05$ |
| 验收阈值 | 1% | 1% | 1% | 1% |

## 2. 怎么做的

1. **定位欠账**：`tests/ci_quarantine.txt` 把
   `tests/test_lbm_verification.py::test_step_a..d` 四条列为"stock main 上已验证相同的失败"，
   `tests/conftest.py` 的 `pytest_collection_modifyitems` 把它们转成 **non-strict xfail**。
   即：这四条解析解验证**在 CI 里从来没真正判过一次红/绿**——xfail 不 strict，失败也不计数。
2. **查因（两条独立缺陷，均非数值错误）**：
   - 四个函数签名写死 `device="sdaa:0"`，CPU-only 主机上 `torch.device("sdaa:0")` 直接抛错；
   - pytest 把形参 `device` 当作 fixture 请求，**收集期**就报 `fixture 'device' not found`，
     与数值对不对无关。
3. **修法（最小改动，阈值/网格/步数一律不放宽）**：
   - 设备默认值改用仓库已有的 `tensorlbm.utils.default_device_name()`（探测 SDAA→CUDA→
     插件加速器→CPU），显式传入仍可；
   - 数值内核搬进返回度量字典的 `run_step_*` 纯函数；pytest 侧只留**零形参**薄壳
     `test_step_a..d`，断言 `err_pct < 1%` 并在失败时打印实际误差；
   - 删除 `tests/ci_quarantine.txt` 中对应的 4 行（隔离区从 120 条降到 116 条）。
4. **可复算**：`PYTHONPATH=src python scripts/lbm_analytic_verification_report.py`
   跑四个案例、写 JSON 与三联图（衰减曲线 / 剖面 / 误差条形 vs 1% 门槛）。

## 3. 计算结果（程序化注入，见 JSON）

| 案例 | 关键量 | 数值解 | 解析/基准 | 相对误差 | <1% 判定 |
|---|---|---|---|---|---|
| A 剪切波 | $\nu$ | 0.166663 | 0.1666667 | 2.16e-03 % | 是 |
| B Couette | 剖面 $u(y)$ 最大相对误差 | — | 精确线性剖面 | 1.33e-03 % | 是 |
| C Poiseuille | 剖面 $u(y)$ 最大相对误差 | — | 离散一致抛物线 | 3.38e-01 % | 是 |
| D 摩擦 | $C_f$ | 由 $\tau_w/U^2$ 反算 | $2\nu/(10.5U)$ | 4.20e-03 % | 是 |

单机 CPU 墙钟：A 0.5 s、B 36.0 s、C 35.3 s、D 34.3 s（合计 106.1 s；主机 load15≈120+，
故单位耗时偏高）。四例全过，`all_passed=true`。

**pytest 门禁证据**：`python -m pytest tests/test_lbm_verification.py -q` →
`4 passed, 2 warnings in 197.89s`（无 xfail、无 skip；隔离区已删除对应 4 行，
故这四条从「永不判红」变成硬门禁）。

图（`docs/assets/lbm_analytic_verification_r1.png`）：
(a) 剪切波振幅采样点 vs 解析衰减指数，10 个采样点全落在曲线上；
(b) Couette/Poiseuille 剖面 vs 解析（Poiseuille 按 $u_{top}$ 缩放便于同轴比较）；
(c) 四例误差与 1% 门槛（对数轴）。

## 4. 注意点 / 误差归因

- **C 是唯一有可见误差的一例（0.34%）**，量级比其余三例大 2 个数量级。做了 Ma 扫描排除
  可压缩性：$u_{max}=0.05$ 时 0.3378%，$u_{max}=0.025$ 时 0.3325%——误差**基本不随 Ma 变化**，
  故不是 $O(Ma^2)$ 可压缩项，而是 $n_y=12$ 的壁面离散误差（半程反弹壁面位置误差 +
  抛物线剖面在粗网格上的二阶截断）。要压到 0.1% 以下需 $n_y\gtrsim 24$，本套件未改网格
  （改网格=改判据，需另立单并给出网格收敛序列）。
- **B/D 的 1e-3 % 级误差是"半自洽"结果**：解析式与离散半程反弹壁面位置（$y=0.5$、
  $y=n_y-0.5$）按同一约定书写，因此测量的是"实现是否复现该约定"，而非连续极限。
  这类高精度只说明实现与约定一致，**不能**当作数值精度指标。
- **C 的体力项是简化源项**：代码用 $f_q\,{+}{=}\,w_q\,3c_{qx}G$（等价于 $\mathbf u=0$ 的
  Guo 项，缺 $1-1/(2\tau)$ 前置因子与 $\mathbf u\cdot F$ 项）。解析式按同一约定书写
  （$G/(2\nu)$ 而非 $G/(8\nu)$ 的连续式），所以 0.34% 的残差不能再叠加"体力源项误差"的
  解释；若换成连续 Poiseuille 基准，这个源项约定会引入 2 倍的系统偏差——**引用本套件
  结果做基准时必须连带引用源的约定**。
- 本套件只覆盖 D3Q19 + BGK。MRT/Cumulant、D3Q27、壁面函数、多相的自由面链路不在其覆盖内。

## 5. 误差判定（对照 3% / 1% 门槛逐项回答）

| 关键量 | 误差 | 是否 <3% | 是否 <1% |
|---|---|---|---|
| 黏性 $\nu$（A） | 2.16e-03 % | 是 | 是 |
| Couette 剖面（B） | 1.33e-03 % | 是 | 是 |
| Poiseuille 剖面（C） | 3.38e-01 % | 是 | 是 |
| 壁面 $C_f$（D） | 4.20e-03 % | 是 | 是 |

四个案例对解析解的相对误差均 <3%（亦均 <1%），本套件在**其自述约定内**通过。
需要强调的诚实边界：B/D 的高精度来自"解析式与离散边界约定同源"，A/D 的阈值 1% 对
BGK 黏性验证偏松（真实能力在 1e-3 % 量级），因此这套门槛**偏保守而不偏激进**，
不存在"放宽门槛凑绿"的问题。

## 6. 遗留 / 后续

1. Poiseuille 的 $n_y$ 收敛序列（$n_y=12/18/24/36$）以把 C 的 0.34% 变成带观测阶的证据 ——
   需新算力预算，本单在 load15 > 100 下未起重算。
2. 同一套壳可推广到 MRT / Cumulant 与 D3Q27（各自有闭式黏性/Poiseuille 解），
   目前 quarantined 列表里仍有 `test_multi_gpu_d3q19_equivalence`、`test_multi_device_sdaa_common`
   等多卡等价性欠账，属"多卡 CI 环境缺失"而非数值问题，另立单处理。