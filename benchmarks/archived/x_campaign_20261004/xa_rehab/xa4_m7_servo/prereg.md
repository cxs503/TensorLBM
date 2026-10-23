# XA-4 M7 U∞ 闭环伺服驱动 — Phase 1 单臂验证 prereg（冻结版 v1）

轨道: X 突破计划 XA-4。上游档案只读:
`/nfs/wangxi/runs/bm_mp_20261001/m7_taylor/`（prereg v1 + Amendments A1–A3、
NOTES CLOSURE REGISTER、scripts/{formal_run.py, formal_arms.json, m7lib.py,
ref_lock.py}、out/{p0_sigma.json, p1_poiseuille.json, formal_ca010_R67.json,
ref_curve.json}）。

## 0. 协议地位（明写, 控制器指令要求）

**本轨是全新预注册协议（servo 轨）, 不是对 M7 冻结臂的重判。**
M7 prereg/A2 的"formal 单发无重驱"条款继续管辖其 6 个冻结臂, 它们的
判决与档案原样不动。伺服轨在独立目录、独立判决 JSON、独立门下运行;
其结论只用于决定"是否开设全阶梯伺服协议"（控制器裁量）。冻结臂的
Ca_ach 数字不因本轨改变。

## 1. 背景与假设链

- M7（平面 nz=1 Taylor 液滴成膜, CAC W=4, CPU fp64, σ=0.1）6 臂阶梯:
  粗臂 Ca_ach 偏移 +0.08/+5.06/+5.17%, 细臂（R67/R78/R91）
  **+13.33/+15.89/+14.0%（跨 Ca 均值 ~+14.4%）**——A2 网格传递
  （P3 两相图, R=44 上标定）在细格系统性失效; ca010_R67:
  Ca_ach=0.113328、err_h **+5.306%**（门 3%, FAIL）vs ca010_R44
  +2.300%（PASS）; Ud/U∞ 实测 1.809 vs A2 预测 1.708、U∞ +7.0%。
- **H1**: 细臂 err_h 超带主要由 Ca_ach 偏移介导（判决在连续 Eq.30
  曲线上按 Ca_ach 计, Ca +13% 沿曲线族斜率放大 err_h）。若把 Ca_ach
  伺服到目标 ±2%, err_h 应落回粗臂带（~+2.3%）。
- **H2（备择）**: 细格偏差结构性（成膜分辨率/壁效应）, 与驱动无关——
  伺服到 Ca 目标后 err_h 仍 >3%。
- 两假设都合法结果; Phase 1 单臂定胜负方向, 全阶梯开否报控制器。

## 2. 参考锁（先锁后跑; 判据脚本机器重提并断言一致）

| 量 | 冻结值 | 机器源 |
|---|---|---|
| R67 臂参数 | R=67, ny=138, nx=871, nsteps=233129, gx=2.9531274180874463e-06, ud_target=0.009340303531852497, µ_eff=0.99866883, H_eff=134.175259 | scripts/formal_arms.json tag=ca010_R67 + out/p1_poiseuille.json |
| σ_eff | 0.932787×0.10（P0 因子 @σ=0.1） | scripts/formal_run.py:39 + out/p0_sigma.json |
| Ca_ach/Ca 目标比 | +13.33%（0.113328 vs 0.100） | out/formal_ca010_R67.json verdict + NOTES 2026-10-02 条目 |
| 幂律指数 | Ca_ach ∝ gx^1.41（A2 交叉核对用测量律） | prereg A2 段 + NOTES |
| 判决曲线 | Eq.(30) 连续曲线 h∞_model(Ca, λ=0.1, planar) | scripts/ref_lock.py + out/ref_curve.json |
| 引擎 | m7lib.bootstrap → CAC_PATH（mp_w10 @ 1eae1c3b 基底）; MODULE_SHA 运行时断言与 M7 正式臂同 | scripts/m7lib.py + formal_run.py |
| 冻结估计器 | plateau 窗=泡内去两端 2.5R; Ud=周期质心 LSQ; tail-40 样本; sample_every=500; A1 膜 init 8 胞; travel 2.5L | prereg §4 + formal_run.py（estimators verbatim） |

## 3. 伺服臂（Phase 1 恰一臂）

| 臂 | R | Ca_target | 域 | 初始 gx | 步数预算 |
|---|---|---|---|---|---|
| servo_ca010_R67 | 67 | 0.100 | 138×871（同冻结 R67 形） | = A2 冻结 gx | 基线 233129, 上限 1.3×（303068, 需披露才准用） |

配置 = 冻结 R67 臂逐字（σ=0.1, W=4, mobility=0.05, p0, µ_law="step",
axes="pwp", wall_mask, fp64, CPU, A1 stadium 膜 init, sample 500）,
**唯一新自由度 = gx 由伺服在跑中调整**。

## 4. 伺服协议（冻结）

- 测量: 泡质心周期 LSQ（formal_run.py centroid_circular 逐字）滚动窗;
  从 x_c 首过 **1.1L** 起每 **0.2L** 行程评估一次
  ud_meas = 该 0.2L 窗内质心 LSQ 斜率; Ca_meas = µ_eff·ud_meas/σ_eff。
- 更新律: `gx ← gx·(Ca_target/Ca_meas)^(1/1.41)`;
  触发条件 |Ca_meas/Ca_target−1| > 2%; **至多 3 次更新, 最后一次不晚于
  x_c=1.4L**; 每次更新记 (step, x_c, Ca_meas, gx_old, gx_new) 入 JSON。
- 锁定相: 末次更新后 gx 恒定, 锁定行程 ≥ **1.1L**（> 0.8L 平台形成
  先验 + 裕度）; 总行程 2.5L 与冻结臂同。
- 判决估计器（本协议自有, 预注册）: Ud = 末次更新 +0.3L 起至 2.5L 的
  质心 LSQ（跳过更新后瞬态; 驱动在窗内恒定）; Ca_ach_final =
  µ_eff·Ud/σ_eff; h∞ = tail-40 样本平台均值（窗/去 2.5R 逐字）;
  err_h 按 Ca_ach_final 对 Eq.(30) 连续曲线。
- 健康门: 无 NaN/φ∈[-1e-6,1+1e-6]/φ_int 守恒（LSQ 斜率披露）;
  平台 flatness ≤5%; 泡未触壁。

## 5. 预注册判决规则（见数字后禁改）

- **V1 收敛**: |Ca_ach_final/0.100 − 1| ≤ 2%（且更新次数 ≤3）。
- **V2 占空**: 伺服相步数（末次更新前）/ 总步数 ≤ 60%。
- **V3 健康**: §4 健康门全过。
- **V4 预算**: wall ≤ 1.5× 冻结 R67 wall（同 16 线程口径折算, 见 §6）。
- **结果分支**（V1–V4 全过后才判）:
  - **A（Ca 介导坐实）**: err_h ≤ 3% → 伺服全阶梯提案报控制器
    （6 臂伺服版重跑的新 prereg 另立）。
  - **B（结构性）**: err_h > 3% → 细格偏差与 Ca 偏移无关, 伺服死,
    M7 维持 FAIL, 归因转向膜分辨率/壁效应候选, 如实报。
- V1–V4 任一不过 → 伺服协议本身未验证（协议性 FAIL, 非物理结论）,
  原始 JSON 报控制器, 不开全阶梯。

## 6. 预算与止损

- CPU 轨（不占 GPU6; M7 先例=CPU 192 核盒 nice 分时）。线程 =
  **16/进程**（X 战役铁律 OMP≤16; M7 正式臂为 32 线程先例, 本轨降为
  16 并在 wall 折算里披露: 冻结 3.82h@32thr → 同吞吐折算 ≈7.6h@16thr
  → V4 上限 **11.4h**; 绝对上限 12h）。
- 预算口径: CPU·h ≤ 192（16 线程 ×12h）, 折 GPU·h = 0。
- 止损: NaN/爆掉 → 停, 报 JSON; 3 次更新后仍 |ΔCa|>2% → 停在协议内
  判 V1 FAIL; 超 12h → kill 并报（步数 1.3× 上限内未完成=协议性 FAIL）。
- 启动前查 192 核盒负载（uptime/who）, nice 10, 与 M7 遗留进程错峰。

## 7. 输出与纪律

- out/servo_ca010_R67.json（全 series + 伺服日志 + 双估计器结果 +
  MODULE_SHA + 健康门）; out/xa4_judgment.json（V1–V4 + 分支 A/B 判定;
  判据数字全部机器提取, 对 §2 参考锁断言后才跑）。
- prereg/NOTES append-only; 作废档 superseded_runs/; 不改 M7 档案与
  mp_w10/mp_w11; 不 push/PR; FluidX3D 禁看。

## Amendment 1（2026-10-04, 跑前, 伺服臂零步数烧出前追加; append-only）

- §4 健康门的 φ∈[-1e-6, 1+1e-6] 字面界不适配本引擎族基线: 冻结臂
  out/formal_ca010_R67.json result.phi_max_last = 1.0023946145142026
  （机器档案, 同引擎同配置同估计器）, 即 φ 过冲 ~+0.0024 属引擎 trait
  （液相 φ 微过冲）, 不是失稳信号; 字面界会把与冻结臂同健康的伺服臂
  判成协议性 FAIL。
- 修正: φ 上界改为 phi_max ≤ **1.01**（= 基线 1.0024 + ~4× 裕度）;
  NaN/爆掉 abort（phi_max>1.5, formal_run 同式）维持; φ 下界维持
  ≥ −1e-6; φ_int 守恒 LSQ 斜率披露维持。
- 依据仅含冻结臂档案数字, 不含任何伺服臂结果（本修正案先于伺服臂
  首步落盘）, 无事后门/选购风险。其余 §4/§5 条款原文不动。
