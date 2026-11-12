# taylor_aris_dispersion — independent caliber audit（判据口径审计）

**审计对象**：`benchmarks/pending/taylor_aris_dispersion/`（Wave-3 A，平面缝隙 Taylor–Aris 剪切分散）
**审计类型**：只读复核（adversarial-review 口径）——不做新 LBM 仿真，只对已归档工件 + 独立一维推导重算
**审计工件**：`caliber_audit.py`（可复跑）、`out/caliber_audit.json`、`out/fig_*.png`、`tests/test_taylor_aris_caliber_audit.py`
**一句话结论**：容差门 6/6 通过（最大残差 0.2624%，比 3% 门松 11.4×，独立重算与归档逐位一致）；**阻断该案的"网格单调收敛门"是不可满足的错口径**（残差集是零均值散点，量级等于本案自带的窗口敏感度带，且不是 README 所称的"0.2% 地板"——本案自己的 Pe=0 纯扩散对照给出的地板是 0.0049%，相差 41×）；README 的根因表述**被本案自带对照推翻**，机制状态应为 WITHHELD。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

平面缝隙（两平行板，板距 H）内充分发展 Poiseuille 流 `u(y)=6·u_mean·ŷ(1−ŷ)`，ŷ=y/H，叠加被动标量（浓度）输运。长时极限下纵向剪切把分子扩散放大为 Taylor–Aris 有效扩散：

```
Deff / D = 1 + Pe²/210,      Pe = u_mean · H / D        (Taylor 1953 / Aris 1956, 平面缝隙)
```

| 参数 | 值 | 出处 |
|---|---|---|
| 动量松弛 τ | 0.9（ν = 0.13333） | `run.py` TAU（预注册） |
| 标量松弛 τ_T | 0.8（D = α = 0.1） | `run.py` TAU_T / ALPHA |
| Pe 档 | 10 / 20 / 30 | `run.py` PE_LIST |
| 网格档 H | 64 / 128（流体行数；nx = m·H，1280…3328） | `run.py` H_LIST |
| 测量窗口 | σ_x²(t) 线性拟合 t∈[2,3]·H²/D，Deff = slope/2 | `run.py` K1/K2 |
| 预注册容差 | 3% | `run.py` TOL_PCT |
| 预注册总判 | `all_within_tol AND all_monotone` | `run.py` `scan()` |

本案的"数据"全部来自归档 `result.json`（6 个配置）与 README 内嵌的 machine-sourced 证据块（`<!-- ... -->` 内的 JSON provenance 行，每行带 `f`(工件路径) / `k`(字段) / `v`(值)，即"零手抄"证据链）。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **独立重推理论常数（新算，非引用）**：不引用本案 README 的 1/210，而是数值求解缝隙 cell problem
   `g'' = φ−1, φ=6ŷ(1−ŷ), g'(0)=g'(1)=0`（Neumann 相容性因 `<φ>=1` 精确成立），再取
   `K = −<(φ−1)g>`，即 `Deff/D = 1+Pe²·K`。积分用两重梯形累加，节点数 2×10³/2×10⁴/2×10⁵ 三级加密。
2. **独立重算 6 个残差**：不信 `result.json` 的 `err_pct` 列，直接从 `deff_sim`/`deff_theory` 重算并逐条比对（脚本在失配时 exit 2，不留"大约一致"的余地）。
3. **残差集统计**：带符号残差的均值/标准差/单样本 t（5 dof，t_crit=2.5706）与逐网格、逐 Pe 分解，用来判定"该找机制（系统性偏置）"还是"口径在要求低于分辨率的伪结构（零均值散点）"。
4. **根因证伪测试**：从 README 的 provenance 块里取本案自己预注册的三类对照——V4 纯扩散（Pe=0，同一 wrapped-moment 估计器）、fp64 诊断（H=64/128 同 Pe=20）、H=32 加密点——与"~0.2% 地板"的说法对撞。
5. **工具链**：`numpy` + `matplotlib(Agg)`；**不跑 LBM**（本机 load15≈110/128，按容量纪律只做轻量活；且本案需要的一切判定量都已归档）。
6. 全程零手抄：本文件 ③ 的表由 `caliber_audit.py --write` 生成并注入到下方标记区。

## ③ 计算结果（程序化注入，零手抄）

<!-- CALIBER-AUDIT:TABLE:START -->
### Independent T1 theory check (1-D cell problem, this audit)

| n nodes | K = −<(φ−1)g> | 1/210 | rel. dev | Neumann compat <φ−1> |
|---|---|---|---|---|
| 2001 | 4.761900593139e-03 | 4.761904761905e-03 | -8.75e-07 | -2.50e-07 |
| 20001 | 4.761904720236e-03 | 4.761904761905e-03 | -8.75e-09 | -2.50e-09 |
| 200001 | 4.761904761488e-03 | 4.761904761905e-03 | -8.75e-11 | -2.50e-11 |

### Recomputed residuals (from `result.json`, not the archived err column)

| Pe | H | deff_sim | deff_theory | signed resid [%] | archived \|err\| [%] | match | sens. band [pp] | mass drift | Ma |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 64 | 0.147507 | 0.147624 | -0.0792 | 0.0792 | yes | 0.0104 | 0.0045 | 0.0406 |
| 10 | 128 | 0.147478 | 0.147616 | -0.0937 | 0.0937 | yes | 0.0468 | 0.0173 | 0.0203 |
| 20 | 64 | 0.290630 | 0.290486 | +0.0496 | 0.0496 | yes | 0.0108 | 0.0042 | 0.0812 |
| 20 | 128 | 0.291130 | 0.290368 | +0.2624 | 0.2624 | yes | 0.0545 | 0.0180 | 0.0406 |
| 30 | 64 | 0.529219 | 0.528633 | +0.1108 | 0.1108 | yes | 0.0078 | 0.0042 | 0.1218 |
| 30 | 128 | 0.529748 | 0.528553 | +0.2261 | 0.2261 | yes | 0.0613 | 0.0181 | 0.0609 |

### Residual statistics

- n = 6, mean = +0.0793 %, sd = 0.1497 %, sem = 0.0611 %
- one-sample t = 1.30 (t_crit 95 %, 5 dof = 2.57) → systematic bias: False, all same sign: False
- max |resid| = 0.2624 % → 11.4× inside the 3 % gate
- mean window-sensitivity band = 0.0319 pp
- grid effect (H=64→128) per Pe: Pe=10.0 -0.0145 pp; Pe=20.0 +0.2128 pp; Pe=30.0 +0.1153 pp

### Archived controls used in the root-cause test

| control | value |
|---|---|
| V4 pure-diffusion H=64 err | +0.0049 % |
| V4 pure-diffusion H=64 f-mass drift | 0.0 |
| synthetic moment-bias V3 | 3.77e-15 |
| Pe=20 H=32 fp32 err | +0.6247 % |
| Pe=20 H=64 fp64 err | +0.0251 % |
| Pe=40 H=64 fp32 err | +0.1874 % |
| Pe=20 H=128 fp64 err | +0.1886 % |

### Rulings

| gate | rule | value | verdict |
|---|---|---|---|
| `G1_tolerance` | |err| <= 3% for all 6 configs | max |err| = 0.2624% | **PASS** |
| `G2_prereg_monotone` | err(H=128) < err(H=64) for every Pe | Pe=10.0: |-0.0792|% -> |-0.0937|% (delta_abs +0.0145 pp, monotone=False); Pe=20.0: |+0.0496|% -> |+0.2624|% (delta_abs +0.2128 pp, monotone=False); Pe=30.0: |+0.1108|% -> |+0.2261|% (delta_abs +0.1153 pp, monotone=False) | **FAIL** |
| `G2p_amended_saturation` | |err| <= 1.0% for all configs AND |mean bias| <= 1.0% AND max level-to-level |delta| <= 1.0 pp (a saturation caliber that does not demand structure below the measurement resolution) | max |err| 0.2624%, mean bias +0.0793%, max delta 0.2128 pp | **PASS** |
| `G3_mechanism` | stated root cause ('positive ~0.2% error floor') must be supported by the case's own controls | V4 pure-diffusion control err = +0.0049% (floor claim 0.2%), fp64 diagnostics +0.0251% / +0.1886% | **REFUTED** |
| `G4_systematic_bias` | the residual set must be consistent with a systematic bias (a mechanism to find) or with zero-mean scatter (no mechanism to find) | mean +0.0793%, sd 0.1497%, t = 1.30 vs t_crit(0.95,5) = 2.57, all-same-sign = False | **SCATTER** |

Notes (one per gate):

- `G1_tolerance`: 11.4x inside the tolerance; 6/6 configs pass individually
- `G2_prereg_monotone`: 0/3 Pe levels decrease with refinement.  The grid effect (<= 0.2128 pp) is 14x smaller than the tolerance and the same order as the case's own window-sensitivity band (0.0319 pp)
- `G2p_amended_saturation`: passes with >=3.8x margin on every clause; the thresholds are fixed a priori as 1/3 of the pre-registered 3% tolerance, not tuned to the data (the data would pass even at 0.3%)
- `G3_mechanism`: the measurement path has no 0.2% floor: with the same wrapped-moment estimator at Pe=0 the residual is 0.0049%, i.e. 41x below the claimed floor; the H-dependence survives in fp64, so it is neither rounding nor measurement bias -> the mechanism stays WITHHELD
- `G4_systematic_bias`: the residual set is a zero-mean scatter (mixed signs, |t| below the 95% critical value), so there is no evidence for a coherent bias term to attribute; the observed spread is 10-60x inside the tolerance, hence harmless for the physics claim

<!-- CALIBER-AUDIT:TABLE:END -->

## ④ 结果图

| 图 | 说明 |
|---|---|
| `out/fig_residual_vs_pe.png` | 带符号残差 vs Pe（H=64/128 两条线）+ ±3% 容差带：全部残差落在 0.05–0.26% 量级，离容差 11–60× |
| `out/fig_residual_vs_refinement.png` | 残差 vs 1/H²（含归档 fp32 H=32、fp64 H=64/128、Pe=40 探针）：**非单调**，Pe=10 与 Pe=20/30 符号相反，H=64→128 在 Pe=20/30 处 \|err\| 变大 |
| `out/fig_floor_refutation.png` | 地板证伪：V4 纯扩散对照 0.0049% vs 声称的 0.2% 地板 |

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **单位陷阱（实测踩到）**：归档的 `err_pct` 字段是**百分数**（0.00492 表示 0.0049%），不是小数绝对值。审计脚本第一版按小数乘 100，把 V4 对照读成 0.49%、fp64 诊断读成 18.9%——三个结论全错方向。修正方式：改为"用同文件里的 `deff_sim/deff_theory` 反算校验单位"，并在 `tests/` 里钉死这条一致性断言。任何只读归档数字的下游复算都需先过这一步。
2. **README 的根因表述被本案自己的对照推翻**：README 状态行写"正号 ~0.2% 地板"。但同一案的预注册 V4 门（Pe=0，D_eff 应精确等于 D=0.1，用**同一个** wrapped-moment 估计器与同一拟合窗口规则）实测 0.0049%，比声称的地板小 41×；同时 H 依赖在 fp64 下依然存在（0.0251%→0.1886%），所以既不是舍入也不是测量偏差。**结论：机制 WITHHELD，不得声称已定位根因。**
3. **单调门为何不可满足（本审计的核心裁决）**：6 个残差是零均值散点（混合符号，t=1.30 < 2.57），逐 Pe 的 H 效应为 −0.0145 / +0.2128 / +0.1153 pp（符号都不一致），而同案自带的窗口敏感度带 0.0078–0.0613 pp 与网格效应同量级；再叠加一个"加密到 H=32 反而 0.6247%"的非单调点。在一个残差仅 0.05–0.26% 的问题上要求"加倍分辨率必须减小误差"，等于要求低于测量分辨率的结构。**这不是解的错误，是判据的错误。**
4. **职责边界（不改预注册判据、不自我晋级）**：修订口径（G2′）只能由评审席裁定，本审计只提供证据与建议；另需注意 G2′（门=容差的 1/3=1%）是**事后**提出的，虽然数据在 0.3% 也过（对阈值不敏感），仍属事后口径，须在晋级记录里显式披露为"amended caliber, post-hoc"。
5. **未做（如实申报）**：
   - 第三档网格 H=96（Pe=20）——可把"非单调"从 3 点变 4 点，但需 0.28 M 步 × 16 万格，本机负载下未提交；
   - 全 6 配置 fp64 重跑（仅 Pe=20 有归档 fp64 点）——这是确认"非舍入"的旁证已足，但不成族；
   - 漂移修正后的斜率重算——归档只留了聚合量，未留逐样本 σ_x²(t)/mass(t) 序列（`run.py` 只导出每 10 个样本），故**无法**做；建议后续 `run.py` 增一行 dump `samples_every10` 到 `.npz`。
6. **适用范围**：本审计只针对 `Deff/D = 1+Pe²/210` 的容差与收敛口径；不覆盖 Pe≳40 的 Taylor–Aris 高阶项、不覆盖反应/多组分扩展。

## ⑥ 误差判定（关键量 vs 基准，逐项）

<!-- CALIBER-AUDIT:ERRTABLE:START -->
| key quantity | baseline | measured | error | <3% ? |
|---|---|---|---|---|
| Deff/D, Pe=10, H=64 | 1.47624389 | 1.47507436 | **-0.0792%** | YES |
| Deff/D, Pe=10, H=128 | 1.47616166 | 1.47477796 | **-0.0937%** | YES |
| Deff/D, Pe=20, H=64 | 2.90485954 | 2.90629931 | **+0.0496%** | YES |
| Deff/D, Pe=20, H=128 | 2.90368361 | 2.91130175 | **+0.2624%** | YES |
| Deff/D, Pe=30, H=64 | 5.28633488 | 5.29218980 | **+0.1108%** | YES |
| Deff/D, Pe=30, H=128 | 5.28553241 | 5.29748246 | **+0.2261%** | YES |
| Taylor-Aris constant K (theory 1/210) | 4.761904761905e-03 | 4.761904761488e-03 (n=200001) | **-8.75e-11** (relative) | YES |

**Explicit answer**: worst key-quantity error = **0.2624%** (Pe=20, H=128) vs the analytical Taylor-Aris baseline → inside the 3% tolerance by 11.4x. The case is *not* blocked by accuracy; it is blocked by the monotonicity caliber.

<!-- CALIBER-AUDIT:ERRTABLE:END -->

---

## 建议（交评审席裁定，不自行执行）

1. **判据**：把 G2（单调）替换为 G2′（饱和口径：`|err| ≤ 1%`（=容差 1/3）且 `|均值偏置| ≤ 1%` 且 `max|Δ_level| ≤ 1 pp`），并在晋级记录标注"post-hoc amended caliber + 理由 + 原口径 FAIL 保留"。数据对 G2′ 三款各自有 ≥3.8× 余量。
2. **文档**：删除/改写 README 状态行"正号 ~0.2% 地板"这一被证伪的根因表述，改为"残差为零均值散点（t=1.30<2.57），机制 WITHHELD；本案自带的 V4 纯扩散对照给出估计器地板 0.0049%"。
3. **补强（若要把机制说清）**：追加 H=96（Pe=20）+ 全配置 fp64 + 逐样本 σ_x²(t)/mass(t) 归档；三者合计可把"非单调"从观测变成可归因。
4. **入库**：在满足 1–2 后，本案的物理主张（Taylor–Aris 平面缝隙分散律被复现，最大残差 0.26%）即可入 `verified/`；在此之前应保持 `pending/`（本审计不移动目录）。

## 复跑

```bash
cd benchmarks/pending/taylor_aris_dispersion
python caliber_audit.py            # 表到 stdout，审计失败(重算失配)则 exit 2
python caliber_audit.py --write    # + out/caliber_audit.json + 3 图 + 注入本文件 ③ 的表
python -m pytest tests/test_taylor_aris_caliber_audit.py   # 回归护栏（仓库根运行）
```