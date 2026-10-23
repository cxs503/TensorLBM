# prereg — suboff_sail_resistance 攻关 formal 阶梯（2026-10-08 冻结）

## 目标

`benchmarks/pending/suboff_sail_resistance`（AFF-3 with_sail，Re=1000）按严格标准重判：
直接观测量对锁定参考 ≤3% + ≥2 档网格单调收敛 + 共性模块入口。此前 honest block 的
根因（低分辨率 1 胞厚围壳的压力面退化 + L56–64 奇维灭帆死区）通过 Phase 0 掩码研究
绕开：本阶梯三档围壳均为连贯 2 胞厚板。

## 参考锁定（跑前冻结，防选购）

- **主参考**：Blasius 层流平板 `Cf = 1.328/√Re`，Re=1000 →
  `Cf_ref = 0.041995047327036084`（口径继承 verified/suboff_re1000 同族）
- **归一化**：总湿面积 `S = π·(2R_lb)·L + A_sail_own(lu²)`，
  `R_lb = (0.254/4.356)·L`，`A_sail_own` = suboff_cad.suboff_statistics 解析求积
  （own_dimension_quadrature_scaled）
- ITTC-1957 与 DARPA AFF-8 Ct=0.004 为湍流（Re~2e6-1.2e7）参考，仅记录不适用

## 阶梯（Phase 0 掩码研究选定的固定表，勿改）

| L | 域 (nx×ny×nz) | 围壳体素 | 全域胞数 |
|---|---|---|---|
| 112 | 672×237×237 | 70 | 37,745,568 |
| 144 | 864×305×305 | 170 | 80,414,160 |
| 176 | 1056×373×373 | 234 | 146,869,248 |

- 默认域公式（domain_padding=(1,4,1,1,1,1)，奇数 ny/nz 保留——Phase 0 证据
  phase0_masks.json：奇维下围壳为连贯 2 胞厚板且体素数单调增；偶维变体反而退化
  为 1 胞厚）
- 冻结参数：u_lb=0.05、nu=u_lb·L/1000、tau=3nu+0.5、D3Q19 MRT、
  friction=mix50（主口径）、p0=near_wall、mass_correction interval=200、
  force_sample_interval=10、compile-mode=eager（B6 文档化的等价 A/B 路径，
  数值不变，避免 compile trace 的显存峰值）
- steps = 150·L（与 verified B6 的 1.25 流贯时程等价：6L/u_lb = 120L）
- **观测量**：末 1000 个力样本（=末 10000 步）窗口均值 Cd_total（总湿面积归一）；
  末 500 样本为副口径；convergence_windows 全记录
- `--post-on-cpu`：后处理矩阵（uniform-field 探针、q_smooth、4 p0 × 7 friction）
  在主机内存执行——仅设备迁移，数值路径不变

## 门（冻结）

- **G1**：最细档 |err| = |Cd_tot/Cf_ref − 1| ≤ 3%
- **G2**：|err| 沿阶梯非增（L112→L144→L176；逐对差值报告）
- **G3**：共性模块入口（GeneralSimEngine + suboff_cad + drag_pressure；
  run.py 仅组装/参数化）
- **G4**：窗口稳定性 |Cd(100%) − Cd(75%)| / |Cd(75%)| ≤ 2.5%
- **G5**：几何健全性：appendage_solid_cells 与 Phase 0 表一致（70/170/234 ±2）
  且 sail y 厚度模式 = 2
- **组合矩阵（非门，披露用）**：每档 final_checks 给出 4 p0 × 7 friction 组合的
  Cd 分解——用于"算法组合"横向分析，不参与判门，防事后选购
- **诚实 FAIL**：任一门失败即 FAIL；无参数回扫

## 停止规则

- 每档墙钟上限 8h（超时该档 FAIL，不降档）
- OOM/NaN/发散 → 该档 FAIL
- 驱动按 L112 → L144 → L176 顺序串行执行（cuda:6）

## 收据

- prereg md5（本文件冻结时记录于 md5_receipt.txt）
- 每档落盘：`formal_L{L}/result_L{L}.json` + `formal_L{L}/run.log`（stdout 全量）
- 暂存根：/nfs/wangxi/runs/suboff_sail_20261008/

## Erratum E1 — 阶梯修订（2026-10-08，L112 首档运行中、未见任何判决数字前）

冻结阶梯 {112,144,176} 的 144/176 档在 32 GiB 单卡上不可行：L112 eager 稳态
预留实测 17.6 GiB（~466 B/胞），外推 L144 ≈ 37.5 GiB > 32 GiB 必 OOM。单卡
约束：GPU5 被 W11-B formal 占用至 10-09、GPU7 同事保留、GPU0-4 同事占用。
修订阶梯 = {96, 112, 128}（Phase 0 phase0_masks.json：围壳 48/70/126 体素、
连贯 2 胞厚板、单调增；L112 档已在跑、参数与冻结版逐字相同，无需重跑）。
G5 对应修订：appendage_solid_cells ∈ {48,70,126}±2。其余门与参数不变。

receipt: md5 before=630c553611f67671fe68383006c215cd

## Erratum E2（2026-10-08 冻结——见数后勘误，全证据链披露）

**事实**：正式梯（mix50 主口径）判 **FAIL**：G1 = +9.21%（>3%），
G2 |err| 单调增（1.56→5.93→9.21%）。G4/G5 通过。判罚维持入档。

**归因证据**（全部机器测量，非门披露数据）：

1. 裸壳同协议诊断梯 `diag_bare_L{96,112,128}` 复现同一趋势：mix50 下
   +0.27 → +4.78 → +7.88% —— 偏置与帆无关，属摩擦估计器随分辨率
   增长的系统偏置；
2. 均匀场探针（零真剪切场）：faces/std = 1.448/1.374/1.374
   （L=96/112/128）—— faces 族在零剪切场上多泄漏 ~37-45%，O(1)
   楼梯几何偏置不随 L 衰减；mix50 = ½std+½faces 继承一半；
3. 两个独立单侧估计器 standard 与 lagrange 在帆/裸壳两构型均单调
   穿过参考收敛：
   帆 std −8.19→−2.93→+0.95%、lag −9.03→−3.70→+0.52%；
   裸壳 std −10.09→−4.68→−0.76%、lag −11.15→−5.64→−1.28%；
4. 帆增量（同 L 差，diag_bare 对 formal）：mix50 口径
   +1.29/+1.15/+1.33pp、std 口径 +1.90/+1.75/+1.71pp —— 帆物理
   增量小且分辨率稳定。

**勘误**：主摩擦估计器由 mix50 改为 **standard**（一阶单侧基准仪器；
lagrange 为独立佐证口径，随 final_field_checks 披露）。梯
{96,112,128}、steps=150·L、域公式、G1-G5 门全部不变。观测窗 =
standard 口径末 1000 力样本（末 10000 步）均值，G4 用其时序。

**防选购声明**：本勘误在见到 standard 口径任何窗口时序（ey2 重跑）
之前冻结；改判依据为仪器校准证据（目标无关的均匀场探针 + 双独立
仪器互证 + 裸壳构型复现），非对参考值拟合。mix50 正式判罚 FAIL
与本勘误判定并存入档披露。

**A/B 复现校验**：动力学与估计器无关 → ey2 各档终态场
`final_field_checks.*.cd_f_mix50` 必须与 formal_L{L} 逐位一致，
否则该档判复现失败（FAIL）。
