# 预注册 — suboff_highre_20261010 正式五锚档

**状态：DRAFT — 未冻结。** 定稿时计算 md5 入 md5_receipt.txt 链，此后 append-only。
判据数字全部由机器 JSON 生成（零手抄纪律）。

## 0. 探针声明（防选购）

p0–p38（含 p35r）全部为标定/诊断探针，**不参与本档判决**。其结论仅用于：
(a) 选定本档基线配置；(b) 库缺陷文档。本档判决只读 r1–r5 五个正式 run 的
result JSON。

**基线选定依据（括号臂，2026-10-11，锚1、bgk_smag、resolved 1e5、16000 步）**：
- `wall_model_slip+reichardt`（p25）：Ct_cv 0.065612（+1689.2%），律牵引
  7.270e-3，模式残留 rel 0.09（干净）→ **本档基线（库生产路径）**。
- `stationary`（无壁模型，p36）：Ct_cv 0.033065（+801.7%）但周期-2 模被无滑移
  halfway-BB 壁面泵回（幅值 0.07→0.17 增长，mean|Δ1|=0.274 vs |Δ2|=0.0063，
  奇偶不对称），观察者受污染 → 只做探针，不入正式档。
- `spalding_exchange`（p35/p35r）：CV 力持续为负（−0.06..−0.11）= 同化注入净
  动量，非物理 → 只做记录。
- 壁律阶梯（p32 log/p33 musker）：Ct_fric 只 ±3% → 水平误差结构性。

## 1. 参考锁（已冻结，不重复）

见 `refs_lock.md` / `refs_lock.json` / `refs_lock.py` + `md5_receipt.txt`。
五锚 = refs_lock.json `anchors[]`（V/RT_N/CT/Re_Lmax_nu1139/CT_ITTC1957 全部
机器读取，本文件表格仅示意）：

| 锚 | V (m/s) | Re | CT_EFD | ITTC-1957 |
|---|---|---|---|---|
| 1 | 3.050 | 1.166e7 | 0.003667 | 0.002922 |
| 2 | 5.144 | 1.967e7 | 0.003576 | 0.002676 |
| 3 | 6.100 | 2.333e7 | 0.003488 | 0.002603 |
| 4 | 7.160 | 2.738e7 | 0.003425 | 0.002537 |
| 5 | 8.230 | 3.147e7 | 0.003326 | 0.002481 |

## 2. 判据

- **主门 M1**：每锚 `|CT_sim − CT_EFD| / CT_EFD ≤ 5%`；5 锚中 ≥4 锚通过才判 PASS。
- **M2 单调性**：CT_sim(Re) 沿五锚非增（EFD CT 严格降）。
- CT_sim 定义 = CV 控制体观察者 `Ct_cv`，post-warmup 全窗均值（与 p 系列同口径）。
- **模式卫生（披露非门）**：每锚报告 ct_cv 奇偶均值差 rel；基线预期 ~0.1
  （p25 先例），若 >0.5 在判决表标注模式污染。
- **诊断量（非门）**：Ct_fric 对 ITTC-1957 比值；形状份额 CT_sim−Ct_fric 对实验
  形状份额；y+ 带内性（[30,5000]）；positivity limiter 上限分数；观察者差。
- **停止条款**：任一 run `finite=False` 或 limiter 分数 >0.1 → 该锚 honest-FAIL
  并原样记录，不重跑不换参。

## 3. 正式配置（= p25 基线逐键复制）

- 碰撞：`collision_kernel=bgk_smag`（collide_smagorinsky_bgk3d），C_s=0.05，
  positivity limiter 开
- 双黏度：collision_nu=1.2e-4（resolved_Re=1e5，τ=0.50036）恒定；
  wall_nu = u_in·L/Re_target 逐锚（refs_lock.json 机器读取 Re）
- 来流：u_in=0.06 恒定（Ma~0.1），ramp 1000 步，far-field BC ×2
- 壁：BFL halfway 链 + `wall_model_slip` + `reichardt`
  （stress_exchange_distance 默认），y+ band [30,5000]
- 网格：400×200×200，L=200，S_ref=12630.59 lu²（解析裸壳口径）
- 数值：steps=16000，warmup=8000，sponge width 24 / strength 0.2，
  solid_handling=marine_bounce，guo chunk 19
- 五锚 r1–r5（GPU5/GPU6 分担，各锚独立全新跑，无 checkpoint 续跑）；
  锚参数 `--re/--ct-ref` 由启动器**直接读 refs_lock.json** 生成（零手抄）
- 诊断加项（不改物理）：每锚 `--dump-fields`（runner 91dc6e3），末步宏观场
  {step, rho, u(3,nz,ny,nx), solid} 存 `{tag}_fields.pt`（CPU，~550MB/锚），
  供 p39 壁法向剖面/律反演取证。JSON 工件只记路径。

## 4. 预测（诚实预注册）

锚 1 探针（p25）：CT_sim 0.0656（+1689%），Ct_fric 7.27e-3 = 0.87 × ITTC(Re=1e5)
（壁链交付**分辨率 Re** 摩擦；sqrt( ITTC(1e5)/ITTC(锚1) )=1.689 ≈ 观测 u_tau 超估
1.6×）。**预期五锚全 FAIL M1**；M2 可能通过（若误差近似 Re-不变）。本档意义：
(a) 结构性水平误差的 Re 依赖性档案；(b) 库缺陷文档的正式载体（三联：周期-2 模
含壁面泵浦新章 / 壁链水平误差 / cumulant 全域标量 Smag 语义）。

## 5. 交付

- r1–r5 result JSON + `_fields.pt`（staging 正式区，append-only）
- 机器判决表（脚本生成：每锚 CT_sim/CT_EFD/err/Ct_fric/ITTC 比/形状份额/
  y+/limiter/奇偶 rel）
- p39 剖面取证（近壁亏损对比 EFD 隐含律剖面）

## 6. 变更日志

- 2026-10-11 00:55 起草（括号臂跑动中）。
- 2026-10-11 02:3x 括号臂判决后定稿：基线钉 wall_model_slip+reichardt；
  stationary（模污染）/spalding（负推力）留探针层；加模式卫生披露条款。
- 02:25 冻结后勘误：上行变更日志“02:3x”为定稿前占位笔误，实际冻结时刻以 md5_receipt.txt 为准（2026-10-11T02:21:00+08:00）。
