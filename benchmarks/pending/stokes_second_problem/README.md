# stokes_second_problem（pending）

**状态：⚠️ 未按战役标准验证（遗留档）——目录内只有 run.py 骨架（B25 批次设计），无任何机器结果档（result/verdict/probe/evidence 均不存在），战役三条判据从未在该案例上运行过。**

## 物理问题

Stokes 第二问题（振荡平板）：无限平板在自身平面内以 U(t)=U0·cos(ωt) 振荡，一侧半无限静止流体经充分长时间后趋于周期稳态 Stokes 层，满足 ∂u/∂t = ν∂²u/∂y²，解析解 u(y,t) = U0·e^(−y/δ)·cos(ωt − y/δ)，其中 δ = √(2ν/ω) 为 Stokes 层厚度（黏性扩散穿透深度）。骨架设计取 δ = H/4（H 为域高），即 ω = 32ν/H²。这是振荡边界层（Stokes 层）的经典直接可观测基准：全场剖面 u(y) 对解析解逐点比较，且自带周期稳态自检（同相位早/晚周期剖面差）。对库的价值：验证 D2Q9 上 pre-streaming half-way 移动壁反弹在时变壁速 U(t) 下的动量注入正确性——与 verified/couette_2d、pending/couette_3d（恒定壁速）同一动壁路径的时间非定常扩展。

## 计算结果现状

**无。** 目录仅含 run.py（B25 设计骨架，含完整判定标准文本与用法）；不存在任何机器测量档，故本 README 不载任何测量数字，PROVENANCE 块为空。

骨架自带的预注册设计（引自 run.py 文本，均为未执行的设计值，非测量结果）：

| 项 | 设计值 |
|---|---|
| 格子/碰撞 | D2Q9 / BGK（库原语 d2q9.equilibrium/macroscopic + solver.collide_bgk/stream） |
| 边界 | 顶壁 pre-streaming half-way 移动壁反弹（壁速每步 u_w=U0·cos(ωt)）；底壁静止 half-way 反弹；x 向周期 |
| 网格档 | H∈{40,80}，均 δ=H/4（骨架默认 --H 40 80） |
| 采样 | 第 k 个周期（k≥3，瞬态理论已衰减）相位 ωt∈{0,π/2,π}×早/晚周期 |
| 自带门文本 | 解析解在 |u_ana|>5%·U0 掩码 max 相对误差 ≤3%；H=80 不劣于 max(H=40 值, 3%)；同相位周期剖面差 ≤0.5% |

## 不达标清单

| 战役判据 | 现状 |
|---|---|
| 共性模块入口（库原语、无修正调参） | 设计满足（run.py 用库 d2q9/solver 原语，无修正因子），但未经运行证实 |
| 直接观测量 ≤3% | 未测（无任何结果档） |
| ≥2 档网格单调收敛 | 未测（H=40/80 阶梯仅存在于设计文本） |
| 预注册 + 机器判决 | 无（无 prereg.md、无 verdict JSON） |

## 根因/诊断

遗留骨架档：B25 批次写下 run.py 后从未进入战役协议（同批遗留的 couette_3d 旧档当时的 README 数字全部为 `...` 占位符，机器诊断判 skeleton-only）。本案例与其不同点在于连占位 README 也未留下，属纯骨架。需要提示的家族风险：该动壁路径（pre-stream half-way 移动壁）在 couette_3d 的 Phase-0 机器诊断中暴露过"骨架索引习惯与布局不符"类缺陷，本案例为 2D（D2Q9），布局语义不同，但按战役 Phase-0 停批协议，任何复跑前必须先做骨架逐位诊断（pilot + A/B 臂），不允许直接进 gate。

## 晋级路径

按战役 Phase-0 停批协议走完整弧，晋级 = 归档 PR 将 pending/stokes_second_problem → verified/stokes_second_problem：

1. Phase-0 机器诊断：run.py 原样 pilot（CPU 即可，域小）+ 骨架缺陷 A/B 臂（重点核对动壁方向集、壁行索引、相位采样舍入）；产出 diag JSON（机器写）。
2. 冻结 prereg：参考锁 = Stokes 层解析式本身（u_ana 在驱动内机器生成、逐 case JSON 存档，零手抄）；门沿用骨架文本（3% 掩码 max-rel、两档收敛、周期稳态差）或按诊断结果修订并说明；md5 边车。
3. owner 批准后 gate 跑 H∈{40,80}（两档、分钟级，CPU 量级即可，0–0.1 GPU·h）；verify 脚本机器判 G1–G3。
4. 判决 PASS → 归档 PR 挪目录；FAIL → 按 Phase-0 诊断定向修复后重走（库侧修复走独立库 PR）。

## 运行方式

骨架自带 CLI（设计用法，从未在战役下执行）：

```bash
# 单档
python run.py single H out.json [--tau 0.65] [--U 0.05] [--cycles 6 8] [--nx 8]
# 两档扫描
python run.py scan out_dir --H 40 80 ...
```

案例目录：`benchmarks/pending/stokes_second_problem/`。

<!-- 说明：本案例无机器测量档（目录仅 run.py 骨架，无 result/verdict/diag/probe JSON）；表中设计值为 run.py 文本引用，非测量数字。 -->
<!-- PROVENANCE [] -->
