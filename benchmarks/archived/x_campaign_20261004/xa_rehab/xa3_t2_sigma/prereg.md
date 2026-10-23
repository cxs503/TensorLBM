# XA-3 T2 每臂 σ 自测（basis-C）+ G1a 双读法重判 prereg（冻结版 v1）

轨道: X 突破计划 XA-3。上游档案只读:
`/nfs/wangxi/runs/bm_mp_20261003/t2_taylor/`（prereg v1.0+Amendment 1、
CLOSURE、NOTES N10、out/verdict.json、out/arm_*.json、out/calib_*.json、
ref/taylor_ref.py）与 `/nfs/wangxi/runs/bm_mp_20261003/t3_coalescence/`
（NOTES σ 双 µ 标定、CLOSURE §4 σ_eff(µ) 三点）。

## 1. 背景与假设链

- T2（Taylor 剪切液滴, CAC 相场, 引擎 mp_w11 cac_lbm md5
  ccc671ee4738d9266967e74f1fd8924b）收案 OVERALL FAIL: G1a 主门
  D_I/D_SH(Ca_ach)∈[0.96,1.04] 过 6/11（出带 C24c/C32a/C32b/C32c/Re2）;
  G2 单调 0/3; G3 截距 −0.00217/−0.00188/−0.00180（门 |b|≤0.002）。
- N10 机器配方: ratio_sh(R,Ca) ≈ 截距(Ca) − ~25/r_eq0², 截距
  1.0662/1.0721/1.0832（Ca 0.05/0.10/0.15）→ 无限分辨共模 +6.6~+8.3%
  + Ca 项 +1.70pp/0.1Ca。动力学侧全对（τ_rel、θ、G6 相对壁响应）。
- 候选源之一（CLOSURE 未判）: "动态 σ vs 静态 σ_rec"——判决链消费的
  σ_rec 来自 **带壁剪切盒内**静态滴（t2_run.py --mode calib, box=6R 域,
  bands φ>0.9/φ<0.1, σ_rec=dp·R_nom/2, 窗=末 4 采样）。
- **T3 先例（控制器亲验）**: basis-C/T3 制式 = **周期域**（axes="ppp"）
  静态滴, M1 协议逐字, σ_eff(µ) 三点反转 −7.64%@µ3 → −3.48%@µ0.05 →
  +7.55%@µ0.02——W=4 CAC 的 σ_eff 对条件敏感, ±8pp 量级恰与 T2 共模同阶。
- **H1**: 判决链 σ 的测量条件（带壁盒）≠ 无界 S-H 参考假设的体相 σ;
  周期 basis-C 自测 σ_eff^arm 与 σ_rec 的差 (σ_eff/σ_rec−1) 若为负
  ~−7%，则 Ca_ach 系统性低估、D_SH 低估、ratio_sh 偏高——方向与幅度
  可解释 G1a 出带图案。
- **H2（次级, 归因）**: 差异主要来自标定域的壁受限（周期 vs 带壁同尺寸
  对照隔离此维）, 而非 µ 或 (L,R) 族本身。
- H1 用自测 σ 重判后 G1a 带内显著上升且 G2/G3 不恶化 → σ-失配假设
  成立, 报控制器裁量（是否以修正案升级 T2 判决——控制器权, 本轨无自改权）;
  否则 σ-失配假设被杀, T2 判决原样（诚实 FAIL）。

## 2. 参考锁（先锁后跑; 判据脚本从机器源重提并断言一致后才跑）

| 量 | 冻结值 | 机器源 |
|---|---|---|
| σ_rec R16/R24/R32/µ045 | 0.006955418942809962 / 0.0069130817299307085 / 0.0069008720049158635 / 0.006913281043059091 | t2 out/calib_*.json summary.sigma_rec |
| r_eq0 族 | 16.203029548649447 / 24.136302276492046 / 32.102479837175827 | t2 out/calib_*.json r_eq0 |
| 11 臂判决量 | D_I, γ̇_fit, r_eq, ratio_sh, err_sh, Ca_ach | t2 out/verdict.json + out/arm_*.json |
| G1a/G2/G3 冻结门 | [0.96,1.04]; 单调不增(容差0.2pp); 截距≤0.002 | t2 prereg §6（原文, 不改） |
| D_SH 公式因子 | (35/32)·Ca·(1+9.9743·(R/H)³), H_eff=6R/8R | t2 ref/taylor_ref.py + taylor_ref_table.json |
| 引擎 | mp_w11 cac_lbm md5 ccc671ee4738d9266967e74f1fd8924b（运行时断言） | t2 prereg §8 |
| µ 依赖证据 | σ_eff(µ3)=−7.64% / (µ0.05)=−3.48% / (µ0.02)=+7.55% | t3 CLOSURE §4 表 |

## 3. 自测臂表（静态 ⇒ Ca 不进入; 同族臂共享条件; 共 5 条件 × 2 域变体 = 10 run）

初始化 = 复刻臂的 init（名义 R, W=4, 同 init 函数）→ 实现半径≈族 r_eq0,
W/R 与臂逐位同族。配置: fp64, ρ=1/1, µ=族值, σ_in=0.007, mobility=0.05,
p0=0.01, 12000 步, 采样 500, 窗=末 4 采样; 引擎 mp_w11 只读 import。

| tag | R_nom | box | µ | serves | 周期域尺寸 | 带壁复刻 |
|---|---|---|---|---|---|---|
| stp_R16 | 16 | 6 | 0.85 | C16a/b/c | 100³ (axes ppp) | t2 build_sim 复刻 |
| stp_R16_b8 | 16 | 8 | 0.85 | Wlad | 132³ | 同 |
| stp_R24 | 24 | 6 | 0.85 | C24a/b/c | 148³ | 同 |
| stp_R32 | 32 | 6 | 0.85 | C32a/b/c | 196³ | 同 |
| stp_R24_mu045 | 24 | 6 | 0.45 | Re2 | 148³ | 同 |

- **变体 P（主）**: 周期域 basis-C（T3/M1 协议逐字: bands φ>0.9/φ<0.1,
  dp=⟨p⟩in−⟨p⟩out, σ_eff = 窗均 dp·r_eq/2, r_eq=窗均实现半径;
  同时记录 dp·R_nom/2 原约定值）。
- **变体 W（对照）**: 原样复刻 t2 calib 配置（同 build_sim 带壁域+同采样器,
  只读 import t2_run.py）→ σ_conf; 兼作原标定链的确定性回执
  （窗均 dp 应与 calib 档案一致到 fp64 位级; 不一致 → 停报环境漂移）。
- 健康门（每 run 全过才计入）: 无 NaN; 窗内 u_max ≤ 1e-4; φ 积分漂移
  ≤1e-6; 尾段 σ 样本平台（末样-首样)/均值 ≤0.5%。

## 4. 重判协议（纯后处理, 零新剪切臂; D_I/γ̇_fit/r_eq 全部取自档案臂 JSON）

- **R-A 回执（先于一切）**: 用档案 σ_rec 与 ref/taylor_ref.py 因子重算 11 臂
  ratio_sh, 与 verdict.json 逐位一致（≤1e-12）。
- **读法 A（原口径, 复核）**: σ = 档案 σ_rec → G1a 6/11（应逐位复现）。
- **读法 B（主读法）**: σ = σ_eff^arm（变体 P）;
  Ca_ach^B = µ·γ̇_fit·r_eq/σ_eff^arm; D_SH^B = (35/32)·Ca_ach^B·(1+9.9743(R/H)³);
  ratio_sh^B = D_I/D_SH^B → 按冻结带 [0.96,1.04] 重判 G1a。
- **读法 C（对照）**: σ = σ_conf^arm（变体 W）→ 同式重判（隔离壁受限维）。
- G2 重判（读法 B/C）: 每 Ca 族 |err_sh(R)| 随 R 非增（容差 0.2pp）;
- G3 重判: 每 R 对 Ca 的 D_I/Ca_ach LSQ 截距。
- 诊断表: 每族 σ_eff/σ_rec−1、σ_conf/σ_rec−1、隐含 Ca 移动
  ΔCa/Ca = −(σ_eff/σ_rec−1), 对照 N10 共模 +6.6~+8.3% 与 Ca 项。

## 5. 预注册判决规则（见数字后禁改）

- **成功 S1**: 读法 B 的 G1a 带内数 ≥ 9/11 且 > 读法 A 的 6/11。
- **成功 S2（G2 不恶化）**: 每 Ca 族 max_R|err_sh^B| ≤ max_R|err_sh^A| + 0.2pp。
- **成功 S3（G3 不恶化）**: ∀R: |b^B(R)| ≤ max(|b^A(R)|, 0.002)。
- **总成功 = S1∧S2∧S3** → σ-失配假设成立, 停批报控制器裁量
  （升级路径=控制器修正案; T2 冻结判决档不动）。
- 任一不满足 → σ-失配假设被杀: T2 G1a FAIL 维持原判, 共模超额
  归因转入剩余候选（CAC 迁移率项/λ_eff≠1）, 本轨不再重开 T2 σ 线。
- 读法 C 仅归因披露（壁受限维贡献）, 不单独构成升级依据。
- 禁止: 改冻结门/带/公式因子; 对 5 条件之外的 (µ,R) 组合补测后重判
  （防选购）; 用读法 B 结果反向调 σ 再跑（无闭环重驱——那是 M7 伺服轨的
  独立协议）。

## 6. 预算与止损

- 预算: 10 run ≈ 2×(3.5+8+12+33+12) min ≈ 2.3 GPU·h, 上限 **3 GPU·h**
  （X 轨 48 GPU·h 池内）。GPU6 分时, 启动前 nvidia-smi 快照入 ctrl/,
  窗口记 NOTES.md; GPU5/T1 与 GPU0-4/7 禁碰。
- 止损: 任一 run 健康门不过 → 停, 原始 JSON 报控制器, 无判决;
  R32 周期域 196³ OOM（~25 GiB 级, T3 V-R128 先例）→ 预授权回退:
  报控制器后可降 box 域到 R24 尺寸并披露（此时 stp_R32 仅带壁变体有效,
  判决降级为"部分条件自测"如实报）。

## 7. 输出与纪律

- out/xa3_selftest_{tag}_{P,W}.json（全 series + 双 σ 约定值 + 健康门）;
  out/xa3_judgment.json（R-A 回执 + 读法 A/B/C 全表 + S1/S2/S3 判定;
  判据数字全部机器提取）。
- prereg/NOTES append-only; 作废档 superseded_runs/; 零手抄; 不改
  mp_w11; 不 push/PR; FluidX3D 禁看。
