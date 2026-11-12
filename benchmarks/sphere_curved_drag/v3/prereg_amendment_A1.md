# 协议修订 A1（写于 v3 生产跑之前）

对 `../prereg.md` 的两处修订。原文件不修改（判据不得事后重写），修订在此登记。

## A1.1 λ 的定义：`λ = R/L`，不是 `2R/L`

`prereg.md` §3 S1 写 `λ = 2R/L`，但 Hasimoto (1959) 的阵列修正以**球半径**为特征尺度：
`λ = a/L_cell = R/L`。`driver.cd_stokes_confined` 一直用 `R/L`（正确）。若按字面实现 S1，
`λ=1/6` 的目标 `C_D` 从 441.96 抬到 2483.5（5.62×），判据不可通过。
本修订统一为 `λ = R/L`；数值对照见 `render.py` 生成的文档 §3.1 表。

## A1.2 控制器与力仪器解耦（裁决的前提）

原协议控制器 `g ← g·sqrt(u_target/u_meas)` 的**前馈**取自 `F_bfl`，而被裁决的对象正是
`F_bfl` 的绝对精度：仪器进回路 ⇒ 算例无法裁决该仪器。新增三种模式（`driver.run_case` 的
`controller`）：

| 模式 | 前馈来源 | 用途 |
|---|---|---|
| `bfl`（默认，向后兼容） | 实测 `F_bfl` 的 EMA | 复现 m16085/上一轮记录 |
| `momentum` | `F_analytic(u_filt)`（Stokes×Hasimoto / Schiller–Naumann） | **v3 生产跑**：回路里没有被测仪器 |
| `fixed` | 无（`--g` 预设体力密度） | 最干净的收敛算例 / 负控 |

三种模式的仪器读数（`F_mb`、`F_bfl`、`F_analytic`）都写进同一条 case record，
新增字段 `F_analytic`、`analytic_vs_mb_pct`、`analytic_vs_bfl_pct`、`controller`。

## A1.3 判据不变

S1（`< 3 %` vs Hasimoto）、S2（`|F_mb−F_bfl|/F_mb < 2 %`）、稳态门（尾巴半窗漂移 `< 1 %`）
全部按 `prereg.md` 原文执行，不放宽。未稳态记录不进判据。
