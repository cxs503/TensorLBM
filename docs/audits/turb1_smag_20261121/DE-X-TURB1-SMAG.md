# DE-X-TURB1-SMAG · 液滴入水组合扫描·湍流轴·Smagorinsky LES（HYDRO 基线敏感性）

> 仓库 TensorLBM · 分支 agent/m26960 · 锚 = NEW:DE-X-TURB1-SMAG
> 战役 = DE·组合比较矩阵波 W1（owner 令 2026-11-21：相场 CH/多相族/碰撞/湍流模型铺开组合比较）。
> 本单问：**在 SURF2-HYDRO-R1 命中的 HYDRO 基线（静水压密度初始化）上叠 Smagorinsky SGS，
> 深度/腔演化/触水后衰减对湍流闭合是否敏感？** 若敏感→湍流轴是可行道；若不敏感→湍流轴证伪（honest FAIL 有效）。

## 判决（预登记，跑前写死）

- 判决点 = 实验 1.857434 D @ 10.394 ms（pc 口径，Santini case-3）
- 基线 = hydro 臂（TL_FS_HYDRO_Z_REF=24.5，C_s=0），预期复现 SURF2-HYDRO-R1：
  峰 1.6366277885 D @ 14.082107 ms (abs) / 13.534875 ms (pc)
- **命中（HIT，敏感）** = 任一 Cs 臂：|峰深 − hydro 峰深| > 0.125 D（1 胞量化）或 |峰时刻 − hydro 峰时刻| > 1.0 ms
- **证伪（FALSIFY，不敏感）** = 所有 Cs 臂均在 hydro 基线 1 胞量化内（|Δ峰深| ≤ 0.125 D 且 |Δ峰时刻| ≤ 1.0 ms）
  → 湍流闭合不是晚窗欠深的杠杆
- 稳定性门：任一 Cs 臂在判决点（10.394 ms pc ≈ step 919）前死亡（health_limit/nonfinite）
  本身即为发现（SGS 破坏自由面格式稳定性），单独记录

## 双臂设置（满窗，cells=8 标准锚）

| 项 | smag_hydro（基线） | smag01 | smag02 |
|---|---|---|---|
| cells / gap / speed / rs / pd / ul | 8 / 0.18 / 0.06 / 1.53 / 3.0 / 0.15 | 同左 | 同左 |
| steps 上限 | 10000（window_complete ≈ 2019 步 ≈ 24 ms pc 停） | 同左 | 同左 |
| TL_FS_HYDRO_Z_REF | 24.5 | 24.5 | 24.5 |
| C_s（BGK SGS，sgs_model=smagorinsky） | 0.0（off） | 0.1 | 0.2 |
| 机制 | 静水压补偿，无 SGS | 静水压补偿 + ν_t=(Cs·Δ)²·2·S̄（τ_eff 逐胞） | 同左 |

粗带先行：三臂各 1200 步（≈14.3 ms abs，覆盖判决点与 hydro 峰）判方向，健康才升满窗。

## 实现（最小、OPT-IN、默认位级不变）

- `src/tensorlbm/free_surface_lbm.py`：+14 行静水压补偿块（TL_FS_HYDRO_Z_REF 门控，
  空串/未设=off；与 SURF2-HYDRO-R1 抢救 patch 逐字一致 + 空串门控）。
- `benchmarks/water_entry/single_drop/free_surface_probe.py`：probe() 增 C_s 参数
  （None→读 TL_FS_CS，空/缺省=0.0=legacy 位级路径），透传 free_surface_step；result 记 C_s 溯源。
- `tools/sdaa_fs_probe_split.py`：+`--cs`/`--hydro-z-ref` 旗标（env 透传，state 溯源）。
- 本地冒烟（loong64 CPU，n=8，60 步）：三路径均跑通，contact=45 复现，默认路径重跑位级一致。

## 数据溯源

| 数据源 | 路径 | 内容 |
|---|---|---|
| 粗带结果（1200 步） | runs/{smag_hydro,smag01,smag02}/result.json + run.log | 三臂 history（depth_over_D 时序） |
| 满窗结果 | runs/{smag_hydro_full,smag01_full}/result.json + run.log | 满窗 history（smag02 粗带即终态，见下） |
| 实验基准 | benchmarks/water_entry/single_drop/fig5_experiment.csv | 18 测点（pc 口径） |
| 场量转储 | runs/smag01_full_planes/planes.npz + result.json | smag01 中面 fill/uz @ 4 注册时刻（--planes） |
| 判决输出 | docs/audits/turb1_smag_20261121/verdict.json + fig_{depth,decay,residual,cavity_field}.png | 程序化注入（analyze.py），零手抄 |

## 粗带结果（1200 步 ≈ 14.59 ms abs，判方向用）

| 量 | smag_hydro (C_s=0) | smag01 (C_s=0.1) | smag02 (C_s=0.2) |
|---|---|---|---|
| 峰深 (D) | 1.636628 | 1.464786 | 1.460729（censored，死亡时仍在加深） |
| 峰时刻 (ms, pc) | 13.534875 | 11.722929 | 9.071893（censored） |
| stop_reason | step_limit | step_limit | **health_limit（drift 1.03e-4 > 1e-4 @ 9.63 ms abs）** |
| vs hydro 峰深 | — | **−0.171842 D（1.37 胞）** | −0.175899 D（censored，1.41 胞） |
| vs hydro 峰时刻 | — | **−1.81 ms** | −4.46 ms（censored） |

- **基线位级复现 PASS**：smag_hydro 1.636628 D @ 14.082107 ms abs / 13.534875 ms pc
  = SURF2-HYDRO-R1 hydro 锚 1.6366277885 D @ 14.082107 / 13.534875（Δ=0，patch 逐字生效）。
- **smag02 稳定性发现（预登记稳定性门触发）**：C_s=0.2 使液相体积漂移门提前 ~13 ms 触发
  （9.07 ms pc 死 vs hydro 22.42 ms pc 死）——SGS 耗散加速质量交换误差累积，
  自由面格式在 Cs=0.2 下无法存活到判决点。
- 方向判定：两 Cs 臂均超 1 胞量化门 → **升满窗**（smag_hydro_full + smag01_full 已派）。
  smag02 不升满窗：死亡步 792 < 粗带上限 1200，满窗不产生新信息（确定性格式，同参同死）。

## 满窗结果（程序化注入零手抄）

| 量 | smag_hydro (C_s=0) | smag01 (C_s=0.1) | smag02 (C_s=0.2) | 实验 |
|---|---|---|---|---|
| 峰深 (D) | 1.6366277885 | 1.4647864405 | 1.4607285197（censored） | 1.857434 |
| 峰时刻 (ms, pc) | 13.534875 | 11.722929 | 9.071893（censored） | 10.394 |
| stop_reason | health_limit（drift 1.15e-4 @ 22.96 ms） | **window_complete（24.005 ms pc 满窗）** | health_limit（drift 1.03e-4 @ 9.62 ms） | — |
| 末时刻 (ms, pc) | 22.412196 | 24.005250 | 9.071893 | — |
| Δ峰深 vs hydro (D) | — | **−0.171842（1.37 胞）** | −0.175899（censored，1.41 胞） | — |
| Δ峰时刻 vs hydro (ms, pc) | — | **−1.811946** | −4.462982（censored） | — |
| 判决点 10.394 ms pc 深度 (D) | 1.577067 | 1.417272 | —（判决点前已死） | 1.857434 |
| 晚窗 7–13 ms pc 平均残差 | −11.27% | −20.41% | −20.98%（7–9.07 段） | — |
| 峰后衰减 depth@15/18/21 ms pc | 1.5977 / 1.2517 / 0.9102 | 1.2658 / 0.9563 / 0.6931 | —（无数据） | 1.0821（内插）/ — / —（实验止于 17.40 ms） |

- **smag01 满窗存活**：C_s=0.1 的 SGS 耗散反而改善液相体积漂移行为
  （hydro 22.41 ms pc 死于漂移门，smag01 跑满 24.005 ms pc 窗）——湍流闭合同时是"稳定性杠杆"。
- **方向**：SGS 是"减深"负道——峰浅 0.172 D、峰提前 1.81 ms（峰时刻向实验 10.394 ms 靠拢：
  hydro 偏差 +3.14 ms → smag01 +1.33 ms），但深度缺口扩大（−11.89% → −21.11%）。
  湍流轴敏感（HIT），但 Smagorinsky 方向与实验相反；补深正道应朝保能/减耗散
  （与 K2 主导丢失通道 7_mass_exchange_conversion 一致）。

## 判决

**判决：HIT（敏感）** —— 预登记判据：任一 Cs 臂 |Δ峰深| > 0.125 D 或 |Δ峰时刻| > 1.0 ms。
- smag01：|Δ峰深| = 0.171842 D > 0.125 D 且 |Δ峰时刻| = 1.81 ms > 1.0 ms → **命中**
- smag02：|Δ峰深|（censored）= 0.175899 D > 0.125 D → **命中**（另触发预登记稳定性门：
  判决点前 1.32 ms 死于漂移门，C_s=0.2 破坏自由面格式稳定性）

**诚实注记（不粉饰）**：
1. smag02 为 censored 峰（死亡时仍在加深），Δ 为下界；满窗不产生新信息
  （确定性格式，同参同死于 step 792 < 粗带上限 1200，故未升满窗）。
2. smag02 的早死是**漂移门**（质量守恒 1e-4）而非 mach/rho 门——
  SGS 耗散与 Körner 质量交换的耦合是下一步诊断方向（与 K2 主导通道同族）。
3. smag01 contact=44 vs hydro 45（1 步 = 0.012 ms，量化噪声级，不影响判决）。
4. 本单只扫 C_s∈{0.1,0.2}；WALE 对照在兄弟单 TURB2-WALE（互禁看工作树）。

## 误差判定（逐项，显式回答 <3%）

| 关键量 | 值 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| smag_hydro 峰深 | 1.6366277885 D | SURF2-HYDRO-R1 锚 1.6366277885 D | 0（位级） | **是** |
| smag_hydro 峰时刻 | 13.534875 ms pc | 锚 13.534875 ms pc | 0（位级） | **是** |
| smag01 峰深 | 1.4647864405 D | 实验 1.857434 D | −21.11% | **否** |
| smag01 峰时刻 | 11.722929 ms pc | 实验 10.394 ms pc | +1.33 ms | **否**（无 3% 时刻门，如实报） |
| smag02 峰深 | 1.4607285197 D（censored） | 实验 1.857434 D | −21.36%（下界） | **否** |
| smag01 判决点深度 | 1.417272 D | 实验 1.857434 D | −23.70% | **否** |
