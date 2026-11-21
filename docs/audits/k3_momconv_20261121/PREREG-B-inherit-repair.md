# PREREG-B · K3-MOMCONV-B：`inherit` 臂的数值稳健性修正与修复臂（跑前预登记）

> 本文件在**修复臂派车之前**写入；PREREG.md（三臂主赛）**一字未改**。
> 背景：主赛 `inherit` 臂（commit 9b81d8dbf）在 1.362 ms 因受追踪质量漂移越过健康门
> （1e-4）而早死。诊断（CPU 与 SDAA 一致复现）：第 65 步起单步 `conv_gain` 尖峰至
> 3.113e+14（受追踪质量瞬时爆炸），根因是**继承气相速度未限幅**：近空界面胞的宏观速度
> u = p/rho 在 rho→0 时病态（可达 1e5 量级），该速度进入 ABB 重建的 f_eq_gas 后把
> populations 炸掉，再经 Körner 质量交换把受追踪质量炸飞。

## 1. 修正内容（最小、仅作用于 inherit 分支）

1. 继承速度只取自**有受追踪质量**的转换胞（m > 0.05 格点质量单位）；
2. 继承速度限幅到 ±0.5（与求解器输入保护同一稳定区间）；
3. 其余路径（`off` / `redistribute`）**一字不改** ⇒ ctrl 与 redistribute 臂的代码路径不变，
   主赛两臂仍与本修复臂可比。

CPU 本地验证（cells=8，同工况）：修正后 115 步无 conv_gain 尖峰（max 3.09e-3），
接触步回到 45（与 ctrl 一致），漂移 3.39e-5 < 1e-4 健康门。

## 2. 修复臂定义

| 臂 | momconv | commit | 用途 |
|---|---|---|---|
| k3_inherit_fix | `inherit`（含限幅修正） | 见 runs/k3_inherit_fix/run.log | 修复臂 |

参数与主赛完全相同：cells = 8，gap = 0.18 D，lattice speed = 0.06，release = 1.53 m/s，
pool depth = 3.0 D，u_limit = 0.15，capacity + simple，SDAA 满窗，`--diag-budget`。

## 3. 判决门（与主赛同一套门，不放宽）

- **命中**：峰深相对实验 1.857434 D 的 |err| < 12% 且存活 > 10.4 ms；
- **证伪**：|err| > 15%，或峰深 ≤ ctrl（更浅），或早死（存活 < 10.4 ms / 非 window_complete）；
- 判决点：实验 1.857434 D @ 10.394 ms。

## 4. 声明

- 主赛判决（PREREG.md 三臂）**不因**本修复臂而改写；修复臂结果单列，并注明其 commit 与
  主赛不同（仅 inherit 分支有差异）。
- 若修复臂早死或仍无命中，则「动量携带转换」处方在本单流体自由面模型上判 **honest FAIL**，
  并如实报告早死机制。
