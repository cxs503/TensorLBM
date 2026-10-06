# XA-5 CG σ_eff(L,R) 迁移的 β* 扫描标定 prereg（冻结版 v1）

轨道: X 突破计划 XA-5。上游档案只读:
`/nfs/wangxi/runs/bm_mp_20261004/c1_color_gradient/`（prereg+Amendment 1/2、
CLOSURE、out/calib_sigma.json、out/g1_laplace_*.json、scripts/run_calib.py、
scripts/run_g1_laplace.py、src_patched/tensorlbm/color_gradient2d.py）。

## 1. 背景与假设链

- C1-G1（双 τ CG Laplace 闭合）honest FAIL 定案: 4 个 (A,β) 阶梯臂
  rel err 2.45~32.67% 全不过 3% 门; CLOSURE 定根因 =
  **σ_eff 的 (L,R) 迁移性**——CALIB (L=128, R=25) 的带测 dp·R 不可迁移到
  L=200 阶梯（R=40 在 4/4 臂低于两个更小 R = 结构性离群; 限制探针
  (R=40@L=280) 只回收 +3.9% → 限制非主因）。CALIB 15 点网格上
  σ_eff 跨度 0.0025237→0.0439752（机器表 out/calib_sigma.json）。
- 同表可见 β 维弱依赖（A=0.04: β0.5/0.7/0.9 → 0.010212/0.010619/
  0.010833, 跨 β ~6%）——β 调 interface 锐度（半高宽随 β 变）。
- **H1**: σ_eff 对 (L,R) 的依赖部分由 recolor 分离参数 β 控制（有效
  界面宽度机制）; 存在 β* 使 σ_eff 在 G1 消费网格上平坦
  （跨配置散布 ≤1%）。
- H1 成立 → G1 复活候选（新 prereg 的复活阶梯: 在 β* 下 CALIB 一次、
  G1 R 阶梯重跑, 门=冻结 G1 门原文; 不重判旧档）; H1 不成立 →
  σ_eff(L,R) 迁移非 β 可治, G1 维持 FAIL 关案（CG Laplace 门须
  matched-(L,R) 标定或 dp(r) 拟合的 CLOSURE 债注记维持）。
- A 维冻结: 扫描固定 A=0.04（G1 主臂族）; β* 结论显式限定 A=0.04
  （复活阶梯同 A; 跨 A 外推不主张）。

## 2. 参考锁（先锁后跑; 判据脚本机器重提并断言一致）

| 量 | 冻结值 | 机器源 |
|---|---|---|
| σ_eff 15 点表 | A0.04_β0.5=0.010212267240284484 / β0.7=0.010619020900319402 / β0.9=0.010833453085584525（全 15 点在档） | out/calib_sigma.json |
| G1 四臂 rel err | 11.80/2.45/18.46%（A0.04_β0.7 R=20/30/40）等全表 | out/g1_laplace_*.json + judgments.json |
| 引擎 | src_patched/tensorlbm/color_gradient2d.py md5 81203d6d6f00104f4853a6db81815fe0（运行时断言） | CLOSURE §4 |
| 估计器 | 静滴（2D 圆, ρ 0.65/0.05 = 13:1 约定）, τ_r=τ_b=1, fp64, 30000 步; σ_eff = dp·R; dp=带测压跳（cg_pressure_jump）; 尾段 = 末 4 个 dp 样本均值（Amendment 1） | scripts/run_calib.py（逐字复用） |
| CALIB 几何 | L=128, R=25; G1 几何 L=200, R∈{20,30,40} | run_calib.py / run_g1_laplace.py |
| G1 冻结门（复活阶梯用, 原文） | |σ_rec(R)−σ_eff|/σ_eff ≤ 3% ∀R∈{20,30,40} + 阶梯散布披露 + 健康门（逐色质量漂移 ≤1e-10/步/色, NaN-free, 半高宽末 20% 稳定） | run_g1_laplace.py docstring + C1 prereg |

## 3. 扫描臂表（A=0.04; 5β × 8(L,R) = 40 run）

β 网格: {0.5, 0.6, 0.7, 0.8, 0.9}。
(L,R) 网格 = G1 消费网格（主） + 归因扩展（副）:

| 组 | (L,R) 点 | 用途 |
|---|---|---|
| 主（复活判据网格） | (128,25), (200,20), (200,30), (200,40) | F_primary(β) = 4 点 σ_eff (max−min)/mean |
| 副（R/L 分离） | (128,20), (128,30), (128,40), (200,25) | F_L(β)@固定 R 与 F_R(β)@固定 L 的分解 |

每 run: 与 run_calib.py 逐字同协议同代码路径（唯一改动 = L/R/β 参数化,
engine import 指向 C1 src_patched 只读树）; 30000 步; 输出含全 dp series、
tail-4 σ_eff、单末样值（披露字段, Amendment 1 制式）、半高宽 series、
逐色质量漂移、u_max。

健康门（每 run 全过才计入）: 无 NaN; 逐色质量漂移 ≤1e-10/步/色;
半高宽末 20% 稳定（|Δ|/均值 ≤1%）; u_max ≤1e-3（CG 静滴伪速度量级）。

## 4. 预注册判决规则（见数字后禁改）

- **F_primary(β)** = 主网格 4 点 σ_eff 的 (max−min)/mean;
  **F_secondary(β)** = 全 8 点同式。
- **D1（β* 存在）**: min over β∈网格 F_primary(β) ≤ 1% → 取
  β* = argmin（并列取离 0.7 近者, 减少与旧档参数距离）; 若网格最小值
  在相邻 β 对之间且 >1% 但两侧趋势支持 <1% 的内部极小, 允许**恰一次**
  预注册细化 β*±0.05（在 (128,25)+(200,30) 两点先验, 再全主网格确认;
  细化次数与网格在判 JSON 里全记录, 禁二次细化）。
- **D2（β* 不存在）**: 细化后仍 min F_primary > 1% → H1 被杀, G1 维持
  FAIL, σ_eff(L,R) 迁移登记为非 β 可治（机制归因看 §5 诊断）。
- D1 成立 → 停批报控制器, 附**复活阶梯 prereg 草案**（新档
  xa5_cg_beta/revival_prereg.md: CALIB@(128,25,β*,A=0.04) 一次 →
  G1 R 阶梯 L=200 三 R 重跑, 门=G1 冻结门原文 + 复活臂健康门同 §3;
  预算另计）——复活阶梯的执行是控制器批准后的独立停批点, 本 prereg
  不授权直接跑。
- 诊断（不载判决）: 半高宽(β) 曲线（界面锐度机制）; F_R vs F_L 分解
  （迁移主维）; σ_eff(β) 单调性; (200,25) vs (128,25) 纯 L 效应。

## 5. 预算与止损

- 40 run × 30000 步: 128² ~16k 胞 / 200² ~40k 胞; C1 全战役
  （28 个同类静态 run）实测 0.349 GPU·h → 本扫描估 ≈0.5–0.7 GPU·h;
  上限 **1.5 GPU·h**（含一次细化 ~5 run）。
- GPU6 分时: 启动前 nvidia-smi 快照入 ctrl/, 窗口记 NOTES.md;
  GPU5/T1 与 GPU0-4/7 禁碰。
- 止损: 任一 run 健康门不过 → 该 (β,L,R) 点记 N/A 并继续其余
  （若主网格某 β 缺点 → 该 β 的 F_primary 记不可算, 如实报）;
  NaN 扩散（同 β ≥3 点 N/A）→ 停, 报控制器。

## 6. 输出与纪律

- out/scan_{L}_{R}_beta{b}.json（40 档, 全 series + 双 σ 约定值 +
  健康门）; out/xa5_judgment.json（F_primary/F_secondary 全表 + D1/D2
  判定 + 细化记录 + 诊断; 判据数字全部机器提取, 对 §2 断言后才跑）。
- prereg/NOTES append-only; 作废档 superseded_runs/; 不重判旧档
  （g1_laplace_*.json 原样）; 不改 C1 档案与引擎; 不 push/PR;
  FluidX3D 禁看。
