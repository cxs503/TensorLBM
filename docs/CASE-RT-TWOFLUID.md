# CASE-RT-INSTABILITY-ATWOOD — finite-density-ratio (two-fluid) arm

## 1. 为什么需要这一支

`benchmarks/case_pool/rayleigh_taylor_atwood05` 的 Boussinesq 支（`run_case.py` / `run_late.py`）在 `(phi,y)->(-phi,-y)` 下不变，
实测尖峰/气泡速度比 `1.0004`，而两流体势流理论给出 `sqrt((1+A)/(1-A))=1.732`（A=1/2）。锚点问的是**尖峰**增长率，
对称模型无法回答。本支新增 `tensorlbm.rayleigh_taylor_twofluid`：单组分 Shan-Chen（SCMP）两相 + 重力作用于质量（`F_g=rho g`），
动量方程保留变惯性 `rho du/dt`，尖峰/气泡不对称由模型自身产生。

## 2. 方法

D2Q9、BGK、EDM 力项（Kupershtokh 2004）、`psi=1-exp(-rho)`、x 周期（一个波长）、上下壁半程反弹（固体行不储质量，
壁面 `psi=0` 即中性润湿）。参考值闭式求值、零手抄：
`u_b=sqrt(2Ag/((1+A)Ck))`、`u_s=sqrt(2Ag/((1-A)Ck))`（Layzer 1955；Goncharov 2002；2D 常数 `C=1`），
`A` 取共沸标定实测值而非名义 0.5。拟合窗按**前沿位置**预声明（`|h_spike| in [0.15,0.45] lam`），不看结果选窗。

## 3. 共沸标定（A→1/2）

| sc_g | rho_high | rho_low | ratio | A |
|---|---|---|---|---|
| 2.8 | 1.172 | 0.386 | 3.037 | 0.505 |
| 3.0 | 1.494 | 0.191 | 7.821 | 0.773 |
| 3.2 | 1.767 | 0.145 | 12.187 | 0.848 |

**`sc_g=2.8` 给出 A=0.505（密度比 3.04），即锚点 Atwood=1/2 在该模型内可达**（标定与算例解耦：算例用实测 `rho_high/rho_low` 初始化并据此定 `A`）。

## 4. 结果（实测，程序化注入）

| 案例 | A_meas | 终止原因 | Ma_peak | 拟合窗 τ | u_spike | u_s ref | err | 门 | asym sim / ref | Re |
|---|---|---|---|---|---|---|---|---|---|---|
| twofluid_l64_g7.5e-5 | 0.505 | mach | 0.264 | [] | nan | 0.0394 | nan | NOT_REACHED | nan / 1.743 | nan |
| twofluid_l64_g7.5e-5_tau070 | 0.505 | mach | 0.296 | [] | nan | 0.0394 | nan | NOT_REACHED | nan / 1.743 | nan |

**判定：本支未取得尖峰增长率的可判定值（honest INCOMPLETE，不粉饰）**：两次算例都在进入预声明位置窗之前被格马赫数护栏终止
（实测 `Ma_peak` 见上表，护栏 0.20），前沿位移窗 `[0.15, 0.45] λ` 未到达，
故 `u_spike`/`u_bubble` 为 NaN、门判 NOT_REACHED。原因定位为 SCMP 界面在 A≈0.505（密度比 3）下的寄生电流：初始 tanh 剖面宽度 2 与模型
平衡界面厚度不匹配，弛豫瞬间把 Ma 抬到 0.26（tau=0.8）/0.30（tau=0.7），相位工具 `run_late.py` 的 Ma 护栏先于位移窗触发。

## 5. 本轮踩到的坑（均已复现并修复/记录）

1. **壁面封闭漏质量**：最初用「周期 roll 后覆写壁行」的反弹实现，实测 -0.05%/步质量损失；改为「出域总体按反方向回到源节点」的
   置换式半程反弹后，相对质量漂移 -1.4e-14（`tests/...::test_gravity_step_conserves_mass_...`）。
2. **壁面 psi 镜像 → 润湿膜**：把流体 psi 镜像进壁行会让壁面强吸引，蒸汽相被抽干（标定里出现 `rho_low=46..1e11` 的非物理值）；
   改为壁行 `psi=0`（中性）后标定回到正常范围。
3. **零水平穿越检测方向写反**：本算例重相在上，`phi=0` 是**上升**穿越；原实现检下降穿越，恒返回空（测试捕获）。
4. **插值分母 clamp 取值错**：上升穿越的 `above-below` 为负，`clamp(min=1e-12)` 把分数变成 -1e12（前沿位置 -3e11 行）；改用带符号分母。
5. **MCMP 交叉排斥路线被否定**：两分组（MCMP）变密度路线在 `tau<=1` 的稳定区内只能达到 A≈0.33（`sc_g` 再强即 NaN），
   达不到锚点 A=1/2；因此本支改用 SCMP（结论记入负面知识，代码未保留）。

## 6. 下一步（可单干的补齐项）

1. 用共沸态场直接初始化界面（而非 tanh 拼凑）+ 前 500 步以 `g=0` 预弛豫，消掉寄生电流后再开重力；
2. 把 Ma 护栏改为「界面带内 Ma」与「体相 Ma」两个量，避免寄生电流一票否决位移窗；
3. 若仍不达标，改用双分布（压力演化）变密度 LBM 或 Allen–Cahn 变密度模型（Fakhari 2017）作对照。

## 7. 产物

| 路径 | 内容 |
|---|---|
| `benchmarks/case_pool/rayleigh_taylor_atwood05/figures/twofluid_fronts_twofluid_l64_g7.5e-5.png` | 前沿高度/寄生电流 vs τ；共沸标定曲线 |
| `benchmarks/case_pool/rayleigh_taylor_atwood05/figures/twofluid_fronts_twofluid_l64_g7.5e-5_tau070.png` | 前沿高度/寄生电流 vs τ；共沸标定曲线 |
| `benchmarks/case_pool/rayleigh_taylor_atwood05/figures/twofluid_calibration.png` | 前沿高度/寄生电流 vs τ；共沸标定曲线 |
| `benchmarks/case_pool/rayleigh_taylor_atwood05/artifacts_twofluid/calibration.json` | sc_g 扫描与实测 A |
| `benchmarks/case_pool/rayleigh_taylor_atwood05/artifacts_twofluid/summary.json`, `benchmarks/case_pool/rayleigh_taylor_atwood05/artifacts_twofluid/acceptance_table.md` | 逐例判据（程序化生成）|
| `benchmarks/case_pool/rayleigh_taylor_atwood05/artifacts_twofluid/runs/<case>/history.csv`, `case.json` | 逐步前沿/Ma/质量漂移 |
| `src/tensorlbm/rayleigh_taylor_twofluid.py` | 能力模块（SCMP + 重力 + 前沿追踪 + Layzer 闭式）|
| `tests/test_rayleigh_taylor_twofluid.py` | 不变量测试（质量守恒/力符号/壁面/几何/闭式参考/制品一致性）|
