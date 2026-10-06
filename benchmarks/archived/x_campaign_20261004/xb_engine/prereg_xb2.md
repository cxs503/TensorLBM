# XB-2 prereg — 多程伪势（two-range ψ）opt-in 引擎扩展，默认位同

冻结时间: 2026-10-04 (md5 见停批回报)。批准前不跑正式电池（探针/冒烟除外）。

## 0. 引文勘误（诚实披露，不静默改）
任务书引 "Li et al. 2013 PRE 88, 013303" — 经核（2026-10-04 检索）该编号
实为 Safari & Rahimian "Extended lattice Boltzmann method for numerical
simulation of thermal phase change in two-phase fluid flows"（热相变文，
非多程）。正确锁定文献族:
- M. Sbragaglia, R. Benzi, L. Biferale, S. Succi, K. Sugiyama, F. Toschi,
  PRE 75, 026702 (2007) — 多程伪势原始两带构造。
- Q. Li & K.H. Luo, PRE 88, 053307 (2013)（arXiv:1306.6445，全文已取核）
  — 式(9) 多程力 + 式(13)-(16) 离散压强张量/机械稳定条件分析。
- H. Huang, M. Krafczyk, X. Lu, PRE 84, 046710 (2011) — 多程调 sigma 改
  变密度比的实证（Li-Luo 引言采信）。
- 背景: Li, Luo & Li PRE 86, 016709 (2012) / PRE 87, 053301 (2013)
  （强迫方案语境；XB-2 不改强迫方案，sc 为主）。

## 1. 制式锁定（Li & Luo 2013 式 9 逐字转录）
F(x) = -psi(x) * Sum_{a=1..8} w_a * [ G1*psi(x+e_a) + G2*psi(x+2 e_a) ] e_a
（标准 D2Q9 权；belt-2 = 沿同方向距离 2 的邻居 ≡ Shan D2Q17 的
e_9..e_16，见该文 §III.A 说明）。G2=0 严格退化为库内 standard 方案。
已核结论（同文）: EOS -> p = rho*cs2 + (G1+2 G2) cs2^2 psi^2 /2（其归一）；
离散 sigma 项 ∝ (G1+8 G2) 且伴随 -(2 G2)|∇psi|^2 反向项 => sigma 压降有
下限（其 Table 1: ×0.42 @ (G1+8G2)/G=0.01）；机械稳定条件依赖
(G1,G2) => 共存密度随 G2 漂移（其 Fig.1: rho_v ~10% @ 同点）。
库/本地符号映射: 库 F=-G*psi*backward-gather, G>0=吸引（scmp_lib 头注
与 #322 落地 docstring 一致）；本地 harness 约定 G1+2G2=1（EOS 保形，
belt-2 排斥 = G2<0, G1=1-2G2>1）。Li 案例族换算: B/C/D = G2 ∈
{-0.075,-0.15,-0.165}（其 G=-40 族）。
M4-(c) 基线（冻结，refs/c_spurious_currents.json; L=256,R=20,tau=1,G=+1,
psi_correct, 尾 5 采样 max, sc==edm@tau=1）:
  cs@0.9 ratio 5.48 u_max 5.2635e-2 | cs@0.7 56.63 1.7756e-1
  cs@0.6 442.38 2.5663e-1 | pr@0.9 5.28 3.7141e-2 | pr@0.7 37.09 1.2263e-1
  pr@0.6 156.9 1.7428e-1 | pr@0.55 480.43 2.0414e-1
（ratio>=100 档 = cs@0.6, pr@0.6, pr@0.55。）

## 2. 假设链
H1: belt-2 排斥加宽界面 + 压低 sigma 项 => 静态液滴伪流 u_max 显著降
（Sbragaglia 2007 原始动机即伪流抑制）。
H2: ratio>=100 档 u_max <= 基线/5 可达（目标带 1e-2~5e-2）。
H3（风险，文献已证）: 共存密度漂移先于 u_max 收益绑定 => 双守恒通道
诚实裁决，若绑定则如实 FAIL 并给出 (u_max, 漂移) Pareto 表。
H0: G2 全程无收益或立即不稳 => FAIL。

## 3. 实现设计（加性，默认位同）
- 库 multiphase.py: sc_single_component_force(..., G2=0.0)（新参数）与
  collide_sc_single_component(..., G2=0.0)：standard 分支加 belt-2 项，
  gather 方向与 belt-1 相同（backward 同向，G2 与 G 同号族）；
  _sc_neighbor_weighted_sum(..., stride=2) 扩展（加性重载，默认 stride=1
  原路径零改动）。scheme!=standard 时 G2!=0 报 ValueError（本轨只开
  standard）。
- harness: refs/scmp_lib.py 副本挂本 src_patched（sys.path 改 mp_w11 +
  xb_engine/src_patched），本地加 make_step 的 multirange 分支
  （collide 复用库函数带 G2）。电池直接锻炼 PR 候选代码。
- 3D 镜像为 stretch（不设门）；本轨 2D D2Q9。

## 4. 判别测试（tests/test_multirange.py）
T1 位同: G2=0.0 时力场与库 sc_single_component_force 在随机 rho 场上
  torch.equal（含/不含 solid_mask、wall_psi 两种）+ collide+stream 5000
  步轨迹 torch.equal（tau=1 与 tau=0.8 各一）。
T2 均匀场 F=0 恒等（belt-2 亦然）。
T3 总动量: 周期域 Sum_cells F = 0 精确（belt-2 链两端等大反消，逐格验证）。
T4 解析: 线性 psi 场上 belt-2 项 = -G2*psi*2*cs2*∇psi（Taylor 首阶）
  相对误差 < 1e-6（fp64 光滑场）。

## 5. 验收臂表（正式电池，批准后跑；全部 refs/run_formal_c.py 协议逐字：
  收敛判据 1e-5、尾 5 采样 u_max=max、质量/稳定同门）
B1 基线复核: sc forcing 重跑 cs@0.9 与 pr@0.55 两档，对存档
  c_spurious_currents.json 行位比（u_max/ratio/rho_l/rho_v 及 hist 序列）。
  不全等（mp_w10->mp_w11 栈差）=> 披露并重跑全 13 档基线（预声明），
  后续比较基=comparator 重跑版。
B2 G2 响应曲线: G2∈{-0.02,-0.05,-0.08,-0.10,-0.15,-0.20}
  × tiers{cs@0.9,cs@0.7,cs@0.6,pr@0.9,pr@0.7,pr@0.6,pr@0.55} = 42 跑
  （G1=1-2*G2），同表记录 u_max、ratio、逐相漂移（对 refs/cs|pr_maxwell.json
  锁定值）。
B3 最优 G2 选取规则（预声明）: 在全部 7 档过漂移门（B4-i）的 G2 中取
  cs@0.6 的 u_max 最小者；并列取 |G2| 较小者。
B4 双守恒通道 @ 最优 G2:
  (i) 漂移门: 每 tier 每相 |err_mr| <= |err_base| + 1.0pp
      （err=(rho_sim-rho_mw)/rho_mw, rho_mw=锁定 Maxwell 表; base=B1），
      且无稳定性退化。绝对 <=1% 在基线自身 <=1% 的档上另报（非门）。
  (ii) Laplace 通道: refs/b_laplace.json 协议（pr@0.55, R/L 保持主读法,
      L∈{128,192,256}）于最优 G2 与 G2=0 重跑: 门 = 每 L 残差
      <= 0.03*max(dP)（原门）且单调条款不劣化; sigma 值变化作为机制
      披露（sigma 下降=预期，不判 FAIL）。
B5 成功门（任务书）: ratio>=100 三档（cs@0.6, pr@0.6, pr@0.55）
  u_max(最优G2) <= u_max(base)/5 三档全过 且 B4 两通道过；否则诚实 FAIL
  + Pareto 表。
B6 edm 抽查: 最优 G2 × {cs@0.6, pr@0.55} × edm 2 跑（披露，tau=1 等价
  预期）。
预算: B1 2-15 跑 ×<=30s; B2 42×<=30s; B4 24×<=2min; B6 2×<=30s;
  判别测试 ~10min。合计 < 1.5 GPU·h。

## 6. 判据纪律
直接观测量 <=3% + >=2 档单调（B2 的 u_max(G2) 响应本身即机制单调证据，
  但成功门以 B5 为准）；判据数字零手抄；诚实 FAIL；append-only；
  FluidX3D 禁用; lettuce(MIT)/XLB(Apache-2.0) 可借鉴。
