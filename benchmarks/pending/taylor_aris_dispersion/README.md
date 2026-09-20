# Taylor–Aris 剪切分散 benchmark（W3-A，Wave-3 新问题类：被动标量分散）

平面缝隙稳态 Poiseuille 流 + 被动标量纵向剪切分散。**理论：Taylor (1953) / Aris
(1956) 缝隙精确解 `Deff/D = 1 + Pe²/210`，`Pe = u_mean·H/D`。**

真实模拟，无外推、无修正因子、无重标定。全部配置/判据/窗口规则在运行前预注册
（`NOTES.md` A 节 + 只读快照 `NOTES_prereg_snapshot.txt`，mtime 2026-09-20T08:09:50Z，
早于一切 case 输出）。

## 判定

**容差判据达标：6/6 档 |Deff_sim/Deff_theory − 1| ≤ 3%，实测 0.050%–0.262%
（低于容差 11–60×）。**
**单调收敛判据未达标：3/3 个 Pe 档 err(H=128) ≥ err(H=64) → 预注册
verdict_pass=FALSE（result.json）。** 根因经诊断链定位为**误差变号 + 正地板**
结构（见"误差结构诊断"），非解发散、非测量偏差、非 fp32 舍入；H=64/128 全部
在 0.3% 内。

## 配置

| 项 | 值 |
|---|---|
| 格子 | 速度 D2Q9 BGK（τ=0.9，ν=0.1333）；标量 D2Q5 BGK（τ_T=0.8，D=α=0.1），无浮力（纯被动） |
| 域 | ny=H+2，nx=m·H；m=ceil(1.15·4·√(6·(1+Pe²/210))) → Pe10:14 / Pe20:20 / Pe30:26；x 全周期 |
| 壁 | 速度：库 `pre_streaming_bounce_back`（半程，no-slip 面 y=0.5/ny−1.5，H_eff=ny−2，W1 已验证模式）；标量：内联解缠零通量反射（同位 y=0.5/ny−1.5，NOTES A.8，死行恒零，逐位守恒） |
| 驱动 | 周期域常体力 `a=12νu_mean/H²`（库 `_apply_body_force_2d`，W1 先例）；无 Zou-He |
| 网格/Pe | H∈{64,128}；Pe_target∈{10,20,30}；u_mean_target=Pe·α/H（Ma 0.041–0.122） |
| 初值 | 流场解析抛物线 + 0.5·H²/ν 纯流场沉降；标量全缝横向均匀高斯线团（A=0.05，σ0=3 节点） |
| 测量 | C̄(x,t)（流体行截面平均）wrapped 矩 σ²=−2lnR·(L/2π)²；窗 [2,3]·H²/α；采样 0.01·H²/α；Deff=斜率/2；Pe_sim 用窗口实测 u_mean |

库入口：`solver.collide_bgk/stream`、`thermal.pre_streaming_bounce_back`、
`thermal.temperature_equilibrium/collision/stream`、`turbulent_channel._apply_body_force_2d`、
`d2q9.equilibrium/macroscopic`。铁律自检
`grep -nE "def (collide|stream|equilibrium|bounce|zou_he|far_field)" run.py` 零命中。

## 管道验证（正式扫描前，全部通过/落实）

| 门 | 结果 |
|---|---|
| V1 稳态剖面 vs 解析抛物线 | Pe30/H64：L2 0.005%、pointwise 0.005%、u_mean 偏 0.007%；Pe10/H64：0.007%/0.008%/0.006%（正式 6 档 V1 全部 L2 ≤0.03%） |
| V2 标量守恒 | 阶段审计（val_mass_probe.txt）：stream 与壁反射对 Σg 贡献**逐位 0.0**、死行恒 0（无壁通量/汇）；全局原始漂移 = 库 W5 常数栅齿 ≤1.8e-2（修订门 2.5e-2，扫描前修订留痕） |
| V3 合成估计器 | wrapped 矩 bias ≤1.7e-15（σ/L=0.05–0.25、非高斯两团、全部生产 nx∈{896..3328}）；naive 中央矩同域 −2e-6～−13.6% → wrapped 选择必要 |
| V4 纯扩散（u≡0） | Deff_sim=0.100005 vs α=0.1（0.0049%），max\|u\|=0 严格静止 |

## 结果（正式扫描，fp32，result.json）

| Pe | H | nx | Deff_sim | Deff_theory(pe_sim) | err |
|---|---|---|---|---|---|
| 10 | 64 | 896 | 0.147507 | 0.147624 | 0.079% |
| 10 | 128 | 1792 | 0.147478 | 0.147616 | 0.094% |
| 20 | 64 | 1280 | 0.290630 | 0.290486 | 0.050% |
| 20 | 128 | 2560 | 0.291130 | 0.290368 | 0.262% |
| 30 | 64 | 1664 | 0.529219 | 0.528633 | 0.111% |
| 30 | 128 | 3328 | 0.529748 | 0.528553 | 0.226% |

独立佐证：⟨u'²⟩/u_mean² = 0.1997–0.2000（理论精确值 1/5）；x_c(t) 漂移速度与实测
u_mean 一致（≤1.4e-4 相对）；pe_sim 偏 target ≤0.03%；子窗敏感性 [1.5,3]/[2,2.5]/
[2.5,3]·H²/α 全部稳定（±0.05pp，无瞬态污染）；动量场质量漂移 ≤2.4e-9。

## 误差结构诊断（post-scan，不替换判据；diag_*.json）

| Pe=20 | H=32 | H=64 | H=128 |
|---|---|---|---|
| fp32 err | −0.625% | +0.050% | +0.262% |
| fp64 err | — | +0.025% | +0.189% |

1. **fp64 对照**：误差不降为 0、单调性不恢复 → 非 fp32 舍入。fp64 各子窗斜率逐位
   相同 → 系统误差恒定于时间（非逐步累积，t=1H²/α 前即建立）。
2. **H=32 探针**：Pe20 误差 −0.625%→+0.05%→+0.26% 随 H **变号**（H≈50 过零，
   |err| 最小在 H=64）→ 总误差 = 负号 O(1/H²) 离散项 + 正号缓变地板（~0.2%）。
   单调收敛判据失败源于此结构，而非解发散。
3. **排除项**：估计器（机器精度，全部生产 nx）；场非高斯（naive/wrapped@t2≈0.94
   与纯 wrapped 高斯预言一致）；流场剖面（⟨u'²⟩/ū² 偏差 ≤0.05%）；瞬态污染
   （子窗稳定）；壁泄漏（阶段审计逐位 0）；栅齿漂移（均匀重标，归一化矩精确相消，
   fp64 恒定斜率双重实证）。正地板 ~0.2% 的确切格点项未完全定位（候选：D2Q5 线性
   平衡剪切输运截断与标量半程壁的耦合）；如实报数，不修正。
4. **库常数栅齿**：`thermal.W5` 为 fp32 常量（Σw=1+2.98e-8）→ 每步全场 +3.7e-8·T
   均匀乘性重标；fp64 运行不消除（常数舍入非算术舍入）。实测漂移 H64 4.2e-3 /
   H128 1.8e-2 与该常数逐位吻合。归一化测量免疫（V4 0.0049% 实证）。

## 工件

`run.py`（入口：synthetic/profile/diffusion/case/scan）、`NOTES.md`（预注册 A +
时间线 B + 披露 C）、`NOTES_prereg_snapshot.txt`（只读快照）、`result.json`（正式
判定）、`case_pe*_H*.json`（6 档全量）、`val_synthetic.json`、`val_profile_*.json`、
`val_diffusion_H64.json`、`val_mass_probe.txt`、`diag_fp64_pe20_H*.json`、
`diag_probe_pe20_H32.json`、`diag_probe_pe40_H64.json`、`scan.log`、`log_*.txt`。
