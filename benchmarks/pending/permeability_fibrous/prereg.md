# W9-C prereg — Gebart 横向渗透率基准（方阵 + 六角周期圆柱阵列）

Staging: `/nfs/wangxi/runs/bm_widen_w9_20260929/fibrous_perm/`
Baseline library: `/nfs/wangxi/worktrees/bm_w9` @ e717b464 (=main), imports from its `src/` only.
Status: **PREREG v1.1** — written and frozen BEFORE any formal (non-smoke)
run. v1 → v1.1: 判据档加第 4 级 gap≈72 阶梯（见 §4 修订注），判据本身不变。
**控制器核对记录 2026-09-29: prereg v1.1 逐字批准、36 案正式扫描放行**;
参考表独立全验（f 表/K·R² 换算/Gebart 7 档/偏差链逐点同）、探针判决数
独立复算逐位吻合、铁律 grep 零手写核、W64/W128 探针晋升正式案例。
冻结生效: 此后公式/档位/判据/协议常数不可改; S1/S2 超差判 FAIL 不重跑。
Phase-1 probe (controller-approved, "放行 Phase 1") ALREADY RUN 2026-09-29:
sq vf=0.30 W=64 → steady 54.4k 步, err(exact) +4.78%, gebart −11.25%,
12.9 s; W=128 → steady 25k 步, err +2.18%, gebart −13.45%, 9.1 s
（完美求解器 gebart 通道应为 −15.30%, 残差恰等于主通道 err, 自洽）。
W64/W128 与 sq0.30 阶梯第 2/3 级协议全同; 若控制器批准 prereg v1.1 逐字,
二者直接晋升为正式案例, 否则弃用重跑。

## 0. 案例定位

周期圆柱阵列横向（垂直于圆柱轴）Darcy 渗透率 —— 玻纤毡/回热器 checker brick
的标准代理。新增内容 = **六角堆叠**（W4-A 方阵之外的第二个格点）+ **近密堆区间**
（Vf 0.70/0.75，W4-A 未触及）。方阵主档用 Sangani-Acrivos 1982 精确数值门控
（W4-A 已锁转录），Gebart 公式在所有档作为一致性第二通道被同时验证。

## 1. 公式锁定记录（Phase 0，已裁定）

Locked（4+1 独立通道，逐位一致）:

    K_perp = C1 * R^2 * (sqrt(vf_max/vf) - 1)^(5/2)      [Darcy/表观速度约定]
    square: C1 = 16/(9*pi*sqrt(2)) = 0.4001405849587144, vf_max = pi/4
    hex   : C1 = 16/(9*pi*sqrt(6)) = 0.2310212744396075, vf_max = pi/(2*sqrt(3))

通道: (1) Yazdchi/Srivastava/Luding 2011 IJMF Table 1 逐字转录（K/d² 形式,
C=4/(9π√2)/4/(9π√6)，×4 换 R² 形式）; (2) Yazdchi 博士论文 Table 2.1（Gebart 行
逐字，C_c 与 ε_c 同值，本次会话核验）; (3) 引用文献 R² 形式常数表（Toulouse
thesis、Semantic Scholar PDF、Jia et al.）; (4) 本人润滑推导双方阵（方阵单喉道、
六角蜂窝网络 Kirchhoff，k_hex = k_sq/sqrt(3) 与常数比 1/√3 一致）; (5) Sangani
精确数值锚点（方阵 0.75: +0.268%）。

证伪记录: 控制器 v1 常数 4/(9π√2)·r²（d² 形式系数误配 r²）与 16/(9π√3)（√2×
错误）均被独立推导+多点锚定否决。

名义半径定义（阶梯设计用）:
    sq : 单圆柱居中, R = W*sqrt(vf/pi)            （胞 W×W）
    hex: 超胞 W×H, H = round(sqrt(3)*W)，四折叠角(1 支)+中心(1 支) = 三角格点,
         近邻距 W; R = sqrt(vf*W*H/(2*pi))

K/R² 锁定值（名义 vf）:

| vf | sq exact (Sangani) | sq Gebart | hex Gebart |
|----|--------------------|-----------|------------|
| 0.30 | 1.017676e-1 | 1.201494e-1 | 1.083400e-1 |
| 0.40 | 3.604563e-2  | 4.080793e-2 | — |
| 0.45 | — | 2.337998e-2* | 2.635136e-2* |
| 0.50 | 1.179829e-2  | 1.292291e-2 | — |
| 0.60 | 2.969930e-3  | 3.154847e-3 | 5.824787e-3 |
| 0.70 | 3.319516e-4  | 3.418373e-4 | 1.641231e-3 |
| 0.75 | 3.316537e-5  | 3.325418e-5 | 7.239255e-4 |

(*sq 0.45 档已弃用，见 §2；hex 0.45 为披露档。)

## 2. 参考结构（控制器裁定 a/b/c/6 落实 + 六角检索结果）

**Sangani-Acrivos 1982 精确 f 表**（压力形阻力 f = G·L²/(μU_D)），方阵 9 点:
{0.05:15.56, 0.10:24.83, 0.20:51.53, 0.30:102.90, 0.40:217.89, 0.50:532.55,
0.60:1.763e3, 0.70:1.352e4, 0.75:1.263e5}。provenance = W4-A 双转录
（Basilisk cylinders.c sangani[9][2] + Sharaborin/Rogozin/Kasimov, Fluids
2021 6(9):334 Table 1，逐位一致）；稀/密两端渐近式交叉验证 0.040%/0.27%
（W4-A NOTES §1.3）。换算 K/R² = π/(vf·f)，k_ref = π/(vf·f)·R_real²。

**六角 S&A 精确表检索结论: NOT OBTAINABLE（8 渠道穷尽）**:
1. S&A 1982 原文 (ScienceDirect) — 付费墙;
2. Wang & Sangani 1997 (syr.edu) — Cloudflare 拦截直下, webReader 截断于表格前;
3. Koch & Ladd 1997 (Cambridge) — 付费墙, 无开放 PDF;
4. Gebart 1992 原文 (SAGE) — 付费墙（其 Table 2 有六角对比，取不到）;
5. Yazdchi 2011 IJMF 全文 — 仅稀释拟合式, 无表;
6. Yazdchi 博士论文全文（本次, 195 页逐页提取）— Table 2.1 仅关联式汇总,
   FE 数据仅以图形呈现;
7. Monash FM010225 — 图形, 无表值;
8. basilisk.fr / MDPI 直抓 — 前者两机均超时（W4-A 曾成功, 现不可达）,
   后者本地 399B 拦截页（webReader 已证全文仅方阵表）。
按控制器预批准回退: **六角主门 = Gebart 近密堆 0.70/0.75**（Gebart 在此
按构造精确 + 双润滑推导锚定）; 六角稀档如实披露为参考受限。

**最终档位结构**:
- sq 主门: vf {0.30, 0.40, 0.50, 0.60} — **全部精确表值**，对 Sangani exact
  （0.45 弃用: 表无值, log-log 弦插值两端弦差 8.5% 不可靠, 控制器裁定 (6)）;
- sq 披露: vf 0.70（精确门控、非判据; 0.75 弃用: gap48 需 W=2107=4.4M 胞
  超共享 GPU 预算, 近密堆轴已由 sq0.70+hex0.75 覆盖）;
- hex 主门: vf {0.70, 0.75} — 对 Gebart;
- hex 披露: vf {0.30, 0.45, 0.60} — 参考受限（预期系统性偏差: Gebart 稀档
  高估 K → K_sim/K_ref 符号偏负, 不判 pass/fail, 只披露）。
- 全档双通道: err(主参考) 判据 + err(Gebart) 一致性披露（方阵上应复现锁定
  偏差 — 同时验证求解器与 Gebart 公式）。

**Gebart vs 精确 偏差披露（方阵, 锁定值, %）**:
+49.793 / +37.951 / +25.331 / +18.062 / +13.212 / +9.532 / +6.226 / +2.978 /
+0.268 @ Vf = 0.05/0.10/0.20/0.30/0.40/0.50/0.60/0.70/0.75。
方阵档已精确门控 → 该偏差不进判据; 每案例 gebart 通道就地复现之。

## 3. 测量约定（W4-A 继承 + 修正）

- Darcy/表观约定: 体积力 a_body 只注入流体节点（未掩蔽注入被 bounce-back
  部分反转, W4-A NOTES §2.2）; 宏观梯度 G ≡ a_body;
  U_D = (1-vf_actual)·<u_x>_fluid; **K_sim = (1-vf_actual)·ν·<u_x>_fluid/a_body**。
- 驱动强度: a_body = Re_TARGET·ν²/(k_ref_primary·R_real)·force_scale,
  Re_TARGET = 0.05（Stokes 域）, force_scale = 10。
- fp32 注入余量预检: 3·a_body/2^-24 ≥ 14（全部 30 档最差 sq0.70 W864 = 31）。
- 名义几何判据（主通道）: K_ref 用名义 vf 与名义 R（通道 A）; 有效半径诊断
  （通道 B: vf_actual, r_eff = sqrt(vf_actual·W·H/(n_cyl·π))）仅披露不判。
- 步序: collide_bgk → stream → 流体掩蔽体积力 → bounce_back（库
  turbulent_channel 循环序）。

## 4. 网格阶梯与预算（gap_lu ≈ {12, 24, 48} 全档统一）

| 档 | W 阶梯 | gap | 最细胞数 | max_steps 规则 |
|----|--------|-----|----------|----------------|
| sq 0.30 (main) | 32/64/128/192 | 12.2/24.4/48.9/73.3 | 36 864 | max(6e4, min(40·W·H, 1.6e6)) |
| sq 0.40 (main) | 42/84/168/252 | 12.0/24.0/48.1/72.1 | 63 504 | 同上 |
| sq 0.50 (main) | 60/120/238/357 | 12.1/24.3/48.1/72.2 | 127 449 | 同上 |
| sq 0.60 (main) | 96/192/384/573 | 12.1/24.2/48.4/72.2 | 328 329 | 同上 |
| sq 0.70 (disc) | 216/432/864 | 12.1/24.2/48.3 | 746 496 | 同上 (→1.6e6 cap) |
| hex 0.30 (disc) | 28/56/113 | 12.0/23.8/48.0 | 22 148 | 同上 |
| hex 0.45 (disc) | 40/81/162 | 11.9/24.0/47.8 | 45 522 | 同上 |
| hex 0.60 (disc) | 64/128/258 | 11.9/23.8/48.1 | 115 326 | 同上 |
| hex 0.70 (main) | 100/198/396/594 | 12.2/24.0/48.1/72.1 | 611 226 | 同上 (→1.6e6 cap) |
| hex 0.75 (main) | 132/265/530/795 | 11.9/24.0/48.0/72.1 | 1 094 715 | 同上 (→1.6e6 cap) |

共 36 案例。S1 力线性: sq0.30 W64 与 hex0.75 W265 各加一次 fs=20 复跑
（K_sim 不变性 tol 0.1%）; S2 测量窗: 每 case 内建（末 2000 vs 8000 步,
tol 0.05%）。

**v1.1 修订（2026-09-29, 正式扫描开始前, 依据 Phase-1 探针）**: 判据档
（sq 0.30/0.40/0.50/0.60, hex 0.70/0.75）增加第 4 级 gap≈72 阶梯。依据:
探针 sq0.30 收敛比 2.19≈1/gap（一阶楼梯误差, W64→W128: +4.78%→+2.18%）;
几何量子化带 δR/R≈0.5/R 经敏感度 |dlnK/dlnR|=2+2.5(x+1)/x（x=√(φmax/φ)−1,
φ=0.3/0.5/0.7 → 4.5/10.4/42.4）在 gap 48 处各档最坏 ~±5%（W4-A φ=0.5
"d≳320" 同机制）; gap 72 将带压至 ~±3.5%。披露档维持 3 级。判据不变。

## 5. 协议常数（冻结）

τ = 1.0（ν = 1/6）, D2Q9 BGK, 全周期; SAMPLE_EVERY = 200;
稳态判据: |Δuxf|/|uxf| < 1e-5 跨 2000 步, 连续 3 次 + POST_STEADY 8000;
MEAS = 末 20 样本（4000 步）均值; init = equilibrium(rho=1, u=0);
smoke 模式仅管线检查、永非正式结果。

## 6. 判据（冻结）

- 每档: err(主参考) 随细化严格单调降 AND 最细档 ≤ 3% AND 全网格稳态检出。
- 方阵总判: 4 主档全过（vs Sangani exact）。
- 六角总判: 2 近密堆主档全过（vs Gebart）。
- 基准总判: 方阵 AND 六角。
- 披露档不判 pass/fail（hex 稀档预期参考性负偏, 如实记录）。
- 无外推、无重校准; 诊断通道 B 与 gebart 一致性通道仅披露。
- S1/S2 超差 → 判 FAIL（测量不可信）而非重跑取证。

## 7. 执行环境

python = /nfs/wangxi/venvs/tensorlbm/bin/python; CUDA_VISIBLE_DEVICES=6
（与 W9-B 共享, 仅小域探针并行, 大案例顺序）; TMPDIR=/nfs/wangxi/tmp;
全部写入 /nfs; 不 push/不 PR/不 git 操作; 库入口 only（本文件无手写物理核,
grep 自检）; 最终输出 = 控制器数据报告。

## 8. 诚实性条款

prereg 在首个正式（非 smoke）run 前冻结; 冻结后公式/档位/判据不可改;
Phase-1 探针（§开首）是唯一例外且已声明。诚实 FAIL + 定源（数值分辨率 /
参考偏差 / 库缺陷）优先于任何形式 PASS。散文数字一律对机器 JSON 复核。
