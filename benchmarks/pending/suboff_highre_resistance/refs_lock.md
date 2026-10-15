# 高 Re SUBOFF benchmark 参考锁（refs_lock）

战役：suboff_highre_20261010（owner 指令 2026-10-10"直接做高Re的benchmark，对标文献"）
锁定时间：2026-10-10（Phase 0，预注册冻结前）
机器表：`refs_lock.json`（本文件所有数字由 `refs_lock.py` 生成，零手抄）

## 目标构型

DARPA SUBOFF Model 4740，AFF-8（轴对称壳体 + 围壳舵 + 尾十字舵），模型尺度
L_max = 4.356 m，拖曳水池深潜阻力。对应本仓 `SuboffHullType.FULL`。

## 主锚 S1（逐字转录，引用原报告）

Le, Duy Anh & Ngoc Hoa, "Numerical Investigate the Effect of Turbulence
Models on the CFD Computation of Submarine Resistance", CFD Letters 16(10)
2024, 126-139（开放获取，PDF 在 refs/cfdletters2024_suboff.pdf，md5 见
SNAPSHOT）。

逐字（Table 1，"Principal particulars of DARPA SUBOFF submarine in model
scale"）：

| 参数 | 值 |
|---|---|
| Length overall Lmax [m] | 4.356 |
| Length between perpendiculars LPP [m] | 4.261 |
| Maximum hull diameter Dmax [m] | 0.508 |
| Displacement volume [m3] | 0.708 |
| Wetted surface area WSA [m2] | 5.998 |

逐字（Table 3 EFD 列，"The effect of turbulence models on resistance in
comparison with EFD"）：

| V [m/s] | RT_EFD [N] |
|---|---|
| 3.050 | 102.30 |
| 5.144 | 283.80 |
| 6.100 | 389.20 |
| 7.160 | 526.60 |
| 8.230 | 675.60 |

S1 的数据出处标注为其参考文献 [24]："Summary of DARPA Suboff Experimental
Program Data. Naval Surface Warfare Center, Carderock Division (NSWCCD)"
即 Liu & Huang 1998（CRDKNSWC/HD-1298-11，DTIC ADA359226）。

## 交叉验证 S2（独立文档，逐字）

FlexCompute Flow360 验证页 "Underwater Flow Around the DARPA Model 4740
Submarine"（refs/flow360_darpa.html，抓取 2026-10-10）。其 "Resistance
Prediction Errors using DDES" 表逐字：

| Velocity (knots) | Error (N) | Error (%) |
|---|---|---|
| 5.93 | -4.82 | 4.70 |
| 10.0 | 1.03 | -0.36 |
| 11.85 | 1.35 | -0.34 |
| 13.92 | 2.84 | -0.53 |
| 16.0 | -3.74 | 0.55 |
| 17.79 | -7.17 | 0.87 |
| Absolute Error Mean | 3.5 | 1.23 |

符号约定核验（5.93 kn 行）：err_N = CFD − EFD、err_% = (EFD−CFD)/EFD×100
（|−4.82|/4.70×100 = 102.55 ≈ S1 的 102.30）。

由 S2 反推 R_exp = |err_N|/|err_%|×100，与 S1 逐点对比（refs_lock.json
anchors.S2_rel_pct）：**幅值差 ≤ 2.02%，5 点中 2 点落入 err_% 两位小数舍入
带内**，其余 ≤ 2.0%（两位小数舍入带外但同量级）——判为同源（NSWCCD 水池
战役）双文档一致。S2 独有第 6 速 17.79 kn → R_exp ≈ 824.1 N，标记为
**derived**（不入主锚表，仅作趋势旁证）。

## 派生锁定量表（机器生成）

约定（全部为公开声明假设，见 refs_lock.json assumptions）：
- CT = RT / (0.5 · ρ · V² · S)，ρ=1000 kg/m³（淡水，密度带 997–1000 →
  CT ±0.3%）；
- S = 5.998 m²（S1 Table 1）——几何自洽性：本仓 CAD 解析裸壳湿面积
  0.31576·L² ⇒ 物理等效 5.9933 m²，与 5.998 差 **0.08%**（即实验口径的
  参考面积为裸壳解析面积，不含附体 own 面积；sim 侧用同口径）；
- Re = Lmax · V / ν，ν=1.139e-6 m²/s（淡水 15 °C 声明假设）。ν 假设对 CT
  的影响经拟合斜率 m=−0.0935 传递：ν ±4% → CT ±0.4%。

| V [m/s] | Re (Lmax) | CT_exp | CT_ITTC1957(Re) | CT/ITTC |
|---|---|---|---|---|
| 3.050 | 1.166e7 | 0.003667 | 0.002921 | 1.2552 |
| 5.144 | 1.967e7 | 0.003576 | 0.002676 | 1.3363 |
| 6.100 | 2.333e7 | 0.003488 | 0.002603 | 1.3399 |
| 7.160 | 2.738e7 | 0.003425 | 0.002537 | 1.3502 |
| 8.230 | 3.147e7 | 0.003326 | 0.002481 | 1.3405 |

五点对数最小二乘：**CT(Re) = 0.0169331 · Re^(−0.093465)**，残差
−1.38%…+1.49%（refs_lock.json powerlaw_fit）。
ITTC-1957：CT = 0.075/(log10 Re − 2)²（自推公式，与本仓 runner 口径一致）。

## 披露

1. ADA359226 原始报告全文在两台机器（5090/本地）均不可达（archive.org 连接
   超时、apps.dtic.mil DNS 解析失败、mdpi/diva/ntrl/fit 域拒绝），主锚因此
   为**二手逐字转录 + 独立第二文档交叉验证**，非原报告直接逐字。这是本次
   锁定的最强可达证据，如实披露。
2. S1/S2 速度点对应（3.050 m/s = 5.93 kn … 8.230 m/s ≈ 16.00 kn）由
   1 kn = 0.514444 m/s 换算吻合。
3. S1 的 EFD 列保留 2 位小数（力），S2 的 err% 保留 2 位小数——交叉验证的
   舍入带 ±0.005 已计入。
4. 湍流模拟参考水平（非锁）：S1 SSTKO RANS 偏差 0.09–2.80%；S2 DDES 平均
   1.23%。本战役目标为壁函数 LBM 诊断级对标，门限另见预注册。

## 锁定文件

- refs_lock.json（机器生成，含 assumptions/anchors/fit）
- refs_lock.py（生成脚本，可复算）
- refs/cfdletters2024_suboff.pdf、refs/flow360_darpa.html（源文件）
- md5_receipt 链与本文件同步冻结（见 prereg.md 冻结记录）
