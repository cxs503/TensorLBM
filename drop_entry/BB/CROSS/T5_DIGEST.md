# T5_DIGEST · 单打对照组（单滴入深水池 Santini case-3）· 压缩官蒸馏
> 源卡：BB/teams/T5/SOLO.md（DE-T5-S1，qw38，m26274）· 2026-11-20 · 用途=只供新派队只读
> 自检：5/5（①文件在且非空 ②数字可溯 ③双写commit ④预测区间 ⑤失败如实）
> CROSS已读：无（CROSS 目录尚无 *_DIGEST.md）
> 上游验收：PASS（SOLO.md 数字全带出处，无缺失/空心）
## 核心结论
- 早死（0.7~1.8 ms `health_limit`）因果链 = L/I 质量交换单向记账（interface 记贷、liquid 不记借）+ LIQUID 胞 LBM 密度偏离 1 → 液体库存 ~1e-4/ms 单调漂移 → 首触后 1~2 ms 越过 1e-4 门（SOLO#1）。
- 组合配置（simple-exchange+capacity+release 1.53 m/s）可复现「活过 10.4 ms 峰值时刻」：postcontact_healthy_ms=13.851（health_limit@14.410 ms，Ma 0.3006）（SOLO#3）。
- 但 3% 定量验收仍 FAIL——honest-FAIL（SOLO#4）。
## 关键数字（全部带原卡出处）
| 量 | 值 | 出处 |
|---|---|---|
| baseline postcontact_healthy_ms | 0.9121（health_limit@1.520 ms，contact_step=49，能量释放 1.397 m/s） | SOLO#3 |
| 组合 postcontact_healthy_ms | 13.851（health_limit@14.410 ms，contact_step=45） | SOLO#3 |
| 预测区间（组合） | [13,14.5]（归档 cal8=13.851, Ma 0.3006） | SOLO#1 |
| 预测区间（baseline） | [0.8,1.1]（复现归档 d8=1.046） | SOLO#1 |
| 深度对照 1.396ms | 实验0.512/模拟0.250/相对误差51.1% | SOLO#3 |
| 深度对照 6.400ms | 1.647/1.217/26.1% | SOLO#3 |
| 深度对照 10.394ms | 1.857/1.434/22.8% | SOLO#3 |
| 深度对照 13.391ms | 1.304/1.260/3.4% | SOLO#3 |
| 历史峰值 | 1.435 D@11.091 ms（对 1.8574 误差 22.7%） | SOLO#3 |
| 最佳峰值误差 | 13.2%（matched8，13 档全 numeric_pass=False） | SOLO#4 |
| 气穴深度偏低 | 13~23% | SOLO#4 |
| 晚期 Ma 死 | step1183 rho_min=0.724 → Ma 0.3006（风险：Ma>0.3） | SOLO#4 |
| 首点 | 0.0389 D（低于实验 3% 分辨率 0.0132 D） | SOLO#4 |
| 组合与归档逐步一致 | step1185 死亡、drift 1.059e-5、impact 1.40918 m/s 全同，深度 15 位一致 | SOLO#3 |
| 补丁验证 | ‖MM⁻¹−I‖=3.6e-15；--ledger 120 步残差 2.5e-11 | SOLO#2 |
## 可复用方法/仪器
- 补丁：`free_surface_probe.py` +`--ledger`（诊断旗标，默认关，零行为改变）；`solver3d.py` 秩检查 np→torch（1 行，与 `linalg_safe` 同后端；loong64+numpy2.0.1 的 LAPACK SVD 坏）（SOLO#2）。
- 成本命令：`python3 benchmarks/water_entry/single_drop/free_surface_probe.py --cells 8 --speed 0.06 --gap 0.18 --release-speed 1.53 --simple-exchange --capacity --out runs/t5_solo/t5_combined8.json`（集群 cj5662，8 核，~13 min）（SOLO#1）。
- 环境验证：n330 `IMPORT_OK`、`TORCH_RANK 19`（SOLO#1）。
## 诚实披露的失败与局限
- 3% 精度门未过：13 档归档 run 全 `numeric_pass=False`，最佳峰值误差 13.2%；气穴深度系统性偏低 13~23%，未解决（SOLO#4）。
- 晚期 Ma 死未修：step1183 `rho_min` 掉到 0.724（局部低密度袋）→速度尖峰 Ma 0.3006；未动求解器守恒结构（SOLO#4）。
- 库存漂移根因（单向 L/I 交换）是参考模型（waLBerla 不碰 liquid fill）的设计后果，单打席位不敢改（SOLO#4）。
- 环境坑：本机工具 120 s 超时 + 集群首次导入失败（MRT 秩断言）→ 全窗只能修复后上集群（SOLO#4）。
- run 死于 13.851 ms，未覆盖末两点（15.4/17.4 ms）（SOLO#3）。
