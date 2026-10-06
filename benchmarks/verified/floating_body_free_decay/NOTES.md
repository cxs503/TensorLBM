# NOTES — floating_body_free_decay 立项与判定记录

## 背景

任务：验证 6DOF / 操纵类 benchmark，达标 `<3%` 入库 verified，否则诚实阻塞。
`docs/irregular_wave_6dof_validation.md` 已明确：`tensorlbm.rigid_body_6dof` /
`tensorlbm.rao_analysis` 是**彼此独立、由调用者提供水动力输入**的线性工具，
已有测试只检查**形状 / 有限性 / 可复现性 / 有界性**，**未**做物理验收
（不提取 LBM 载荷、不耦合 hull / 自由面、无实验数据）。

## 目标选择

三选一：a) 刚体自由落体/振荡（解析）；b) 浮体静水自由衰减（`ω_n=√(C/(m+a))`）；
c) 斜航阻力（实验参考）。

- **(c) 否决**：斜航阻力需要 LBM 模拟船体（SUOFF/Wigley/DTMB），依赖自由面/多相
  架构；`benchmarks/STATUS.md` 已记录自由面/多相为**结构性**失败（非调参），
  且环境为无 CUDA 的 SDAA 宿主。不能诚实达标。
- **(a)/(b) 合并采纳**：两者都是**解析**闭包。选 (b) 浮体自由衰减——它直接落在
  6DOF 栈的主线（Cummins + 固有频率），且公式 `ω_n=√(C/(M+A))` 可作强解析参考。

## 已有测试的边界（关键）

`tests/test_irregular_wave_6dof.py::test_timeseries_rao_and_prescribed_cummins_response_are_bounded`
仅断言 Cummins 响应 `isfinite`。**从未**检验时域振荡的**频率正确性**。本基准补上
这一缺口。

## 预注册判据（在任何数字之前锁定）

1. 每档 `|ω_meas-ω_n|/ω_n ≤ 3%`；
2. 同自由度 coarse/fine 两档跨度 ≤3%；
3. `compute_natural_frequencies` 与解析 (A) 相对差 `<1e-3%`；
4. 实测频率与辛-Euler 离散色散 `ω_h` 相对差 `<0.5%`（证明积分器精确、误差是
   可解释的 O((ω_n dt)²) 离散色散）。

自由度=2（heave）、3（roll）；档位 `ω_n·dt = 0.5`（coarse）/`0.25`（fine）。

## 发现

1. **固有频率链路完全正确**：三方（解析式 / 特征值 / 时域实测）一致；时域实测
   与辛-Euler 理论吻合到 `<0.001%`（证明积分器更新格式确为半隐式/辛 Euler，
   且其频率偏差是纯数值色散）。相对解析 (A) 的 ~1% 误差在 dt 减半时按 4× 收敛
   （+1.07%→+0.26%，+1.21%→+0.30%）。
2. **辐射记忆路径按设计但实现脆弱**：无阻尼时 `K≡0`，卷积不激活，故 (A) 精确。
   一旦启用 `B(ω)=B0`，仓库的矩形法 + 500 步历史截断卷积对有限带宽
   `K(t)`（δ(t) 的 sinc 近似）在细步长下**混叠并注入能量**（见 README 表：
   dt=0.02、ζ=0.0125 即幅值增长，ζ=0.05 误差 +3.19%）。粗步长 dt=0.05/0.10
   看似吻合（±0.02%）但属**设置依赖的巧合**，不可作验收。
   ⇒ 阻尼/记忆效应**仍未验证**，与 `docs/irregular_wave_6dof_validation.md`
   声明的证据边界一致。

## 结论

- **入库 verified 的部分**：静水自由衰减**固有频率**（无阻尼解析闭包），
  见 `result.json` `verdict.pass = true`。
- **诚实未达标/未覆盖**：6DOF 耦合、自由面、辐射阻尼/**流体记忆效应**；
  端到端 LBM 耐波/操纵验证不存在于本闭包。

## 复算

`run.py` 自包含（仅依赖 `tensorlbm` + numpy/torch），~1 min CPU。默认 CPU、
float32（json 指标 float64）。确定性：固定 `torch.manual_seed(0)`，无随机性。