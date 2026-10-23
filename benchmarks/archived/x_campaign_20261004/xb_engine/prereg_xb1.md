# XB-1 prereg — CAC 各向同性四阶梯度/拉普拉斯算子（opt-in，默认位同）

冻结时间: 2026-10-04 (md5 见停批回报)。批准前不跑正式电池（探针/冒烟除外）。

## 0. 目标与证据基线
把 M1 CAC 的 sigma_eff 系统性赤字（标定绕过 Route-B 的根因）从标定层
升级为引擎层消除。冻结证据（M1 存档，路径=基线，不重锁）：
- P1 律 1-sigma_eff(W) ≈ 1.1/W^2: sigma_effbar(W)=0.8780/0.92390/0.9480/
  0.9647/0.98132 @ W=3/4/5/6/8（128^3 fp64, 周期, rho_l=1/rho_g=0.01,
  nu=0.1, sigma_in=0.01, M=0.05, p0=0.01, 12000 步, 窗=末 2000 每 500 采样,
  sigma_rec=dP*R/2, phi>0.9/phi<0.1 体积均值; refs/calib_sigma.py 原文）。
- W=2 全 R 档 NaN; rho-ratio 无关; B5 高密度比臂 a40/k0.5 = -72.23%
  （壁钉扎主导）、a20/k0.5 = +4.80%（sigma 敏感）。

## 1. 假设链
H1（机理）: 赤字主因 = cac_lbm.py Eq.(27)/(28) 算子（iso_gradient_3d =
3*Sum_{i!=0} w c_a phi(x+c_i)；iso_laplacian_3d = 6[Sum w phi(x+c)-2/3 phi]）
的 O(Delta^2) 截断。Taylor（D3Q19 权 1/18 轴、1/36 对角，四阶矩各向同性
=(1/9)(dd+dd+dd)）:
  G_iso2 = grad(phi) + (1/6) d_a(lap phi) + O(Delta^4)
  L_iso2 = lap phi + (1/12) lap^2 phi + O(Delta^4)
误差 ~ (Delta/W)^2，与 P1 律 1.1/W^2 同标度、同量级（W=4: (1/4)^2*1.22
≈7.6% 即实测 7.61%）。
H2（修复）: 两带四阶各向同性模板把截断推到 O(Delta^4)，W=4 处预测
1-sigma_eff ≈ 1.22*(1/4)^4 ≈ 0.5% <= 1% 达标（系数本身也会变，电池裁决）。
H0（对照，判 FAIL 用）: 算子升级后 sigma_eff(W) 无实质改善 => 根因不在
∇phi/∇²phi 算子（转向 LBE p 校准或 sigma 测量口径），如实 FAIL。

## 2. 模板选型（锁定，含推导）
自有 Taylor 推导（两带组合，D3Q19 同 18 方向、距离 2 邻居）：
记 S(a,lambda)_a = lambda*Sum_{i!=0} w_i c_ia phi(x+a*c_i)
  = lambda[(a/3) d_a phi + (a^3/18) d_a lap phi] + O(Delta^5)
  G4 := 4*Sum w c phi(x+c) - (1/2)*Sum w c phi(x+2c)
     = grad + [4/18-4/18] d(lap) + O(Delta^5)   （线性系数 4/3-1/3=1）
记 T(a) = Sum_{i!=0} w phi(x+a c) - (2/3)phi = (a^2/6) lap + (a^4/72) lap^2 +...
  L4 := 8*T(1) - (1/2)*T(2) = lap + [8/72-8/72] lap^2 + O(Delta^6)
1-D 归约校验（Phase 0 代数已核，判别测试再机器核）:
  G4|_1D = [-phi2+ +8phi+ -8phi- +phi2-]/12 （经典四阶中心差分）
  L4|_1D = [-phi2+ +16phi+ -30phi+16phi- -phi2-]/12 （同上）
文献锚: X. Shan, PRE 73, 047701 (2006) 与 PRE 77, 066702 (2008)（高阶
各向同性算子降低界面伪流/各向异性；belt-2 扩展模板 = 其 D2Q17 思想的
D3Q19 实现）；M. Sbragaglia et al., PRE 75, 026702 (2007)（两带/多程
模板先例）；母本方案 = Liang et al., arXiv:1710.09541 (PRE 97, 033309)
Eq.(27)(28)（被升级对象，#323 已入库）。注: 未在 lettuce(MIT)/XLB
(Apache-2.0) 中找到 CAC 四阶化学势算子先例，本模板为按上述文献思想的
自行推导（推导如上，判别测试机器验收）。

## 3. 实现设计（加性，默认位同）
- 新函数: shifted_neighbors2(phi, axes)（同 18 方向、步长 2，轴策略与
  shifted_neighbors 完全一致——墙轴沿用同一 shift 机制执行两次）;
  iso_gradient_3d_iso4(nb, nb2); iso_laplacian_3d_iso4(phi, nb, nb2)。
- CACSim(..., grad_scheme="iso2") 新参数，默认 "iso2" = 现路径逐位不动。
  "iso4" 时所有 iso_gradient_3d/iso_laplacian_3d 调用点统一换用四阶版
  （chemical_potential 的 ∇²phi、normal_from_gradient、f_source、g_force、
  _macro_fp64 诊断——单一 choke point 辅助函数，一致性由构造保证）。
- 不改任何现有函数体；库 PR 材料含 patch + 判别测试 + README（数字机器提取）。

## 4. 判别测试（tests/test_iso4.py）
T1 线性场 phi=a·x: |G4 - a|inf <= 1e-13 (fp64)；T2 二次场: |L4 - lap|inf
<= 1e-12；T3 三次场收敛率: G4/L4 误差 ~ h^4（h=1/2,1/4 网格）;
T4 球界面法向 90° 旋转对称: |∇phi| 场旋转 90° 后 max 不对称 <= 1e-12;
T5 默认位同: grad_scheme="iso2" 时 G4/L4 代码路径零改动（模块 diff 断言）
+ CACSim 轨迹级 torch.equal（2D 64^2 300 步、3D 64^3 300 步, fp32/fp64,
  含 axes="ppp" 与一墙轴两种）对未打补丁库模块。

## 5. 验收臂表（正式电池，批准后跑）
A1 sigma_eff 阶梯（主门）: iso4, W∈{3,4,5,6,8}×R∈{20,28,40}, 15 跑,
  协议=refs/calib_sigma.py 逐字（sys.path 改 mp_w11 + 本 src_patched）。
  聚合=same as P1: sigma_effbar(W) = mean_R sigma_rec(W,R)。
  门 G1: 1-sigma_effbar(W=4) <= 1%（或等价 W*<=4）且
  1-sigma_effbar(W) 对 W∈{3,4,5,6,8} 单调不增。
A2 归因臂: W=4,R=28 各 1 跑 仅 G4（L 用 iso2）、仅 L4（G 用 iso2）。
A3 附加证据: W=2×R∈{20,28,40} iso4 3 跑，门=无 NaN（基线全 NaN）+
  sigma_eff 披露。
A4 B5 双通道（对照=复用存档，不重锁）:
  通道1 = refs/b5_a40_k0.5.json（Route-B sigma_model=0.10823730488377745）。
  通道2 = refs/b5_hr_droplet.py 逐字配置, sigma_model=0.1（真值）,
  引擎=iso4, fp32, CAP 100k+plateau。门 G2: |err(ch2, a40)| <= 3% =
  引擎消除成立。预声明解读: a40 为壁钉扎主导档（ch1=-72.23%），若 ch2
  仍被壁钉扎支配则 a40 臂如实 FAIL；a20/k0.5 同款 ch2 加跑（预声明
  bonus，sigma 敏感判别器，ch1=+4.80%）作机制证据。
A5 位级复核臂: 默认路径 W=4,R=28,sigma=0.01（12k 步）对
  refs/calib_W4_R28.json：sigma_rec 位比 + 采样序列摘要比。若不全等
  （mp_w10+M1 覆盖 vs mp_w11+副本的栈差）: 披露差异，A1 的跨 W 单调
  条款不受影响（iso4 臂内自洽），P1 对照表标注跨栈。
预算: A1 15×7min=1.75h; A2 0.25h; A3 0.35h; A4 ch2 两臂 <=2h;
  A5 7min; 判别测试 ~10min。合计 ≈ 5.2 GPU·h（GPU6 窗口内分时）。

## 6. 判据纪律
直接观测量 <=3% + >=2 档单调；判据数字零手抄（全部 runner JSON 输出，
报告给机器路径+md5）；诚实 FAIL；append-only；FluidX3D 禁用。
