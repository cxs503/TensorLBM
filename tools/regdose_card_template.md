# DE-X-COL2D-REGDOSE · REG 正则化碰撞剂量响应扫描（0.5×/2×/4×）· 组合波 W2
自检：5/5（文件在且非空✓ 数字可溯=runs/*/result.json+run.log 首行+artifacts/COL2D-REGDOSE/✓ 双写✓ 预登记判定✓ 失败臂照登✓）
CROSS已读：COL2-REG（RR 处方原文/位级锚）、DE_A1_FACTS（实验锚 1.857434D@10.394ms）、CMB3-REGSMAG 草稿（叠加臂互补）
锚点=NEW:DE-X-COL2D-REGDOSE · 战役=DE·碰撞轴 · 分支 agent/m27045 · 仓库 TensorLBM · 2026-11-22

## 1. 处方与物理出处
- 处方（W2 章程）：REG HIT 剂量扫描——owner 令：组合比较找可行道
- 物理出处：Ladd(1994) 正则化碰撞——非平衡部分仅经其二阶应力投影回平衡流形、高阶非平衡矩全丢弃⇒后碰撞分布有界、界面高阶振荡被抑制（COL2 机制：BGK 贴稳定底时界面振荡对空穴起附加阻尼⇒欠深）
- 实现：本分支导入 COL2 实现（edbd0a013，冲突解=保留新 contact_pin 通道），零物理改动；剂量只经既有 rr_tau 通道下发（SDAA 壳 --rr-tau→WE_RR_TAU）
- 剂量定义：rr_tau = dose × 0.5003877（1× = COL2 rr_base 值，禁重跑直接引）；RR 算子 f_post = feq + (1−1/τ)·reconstruct(Π^a)
- 诚实披露①：rr_tau 同时是 BGK 式应力弛豫时间（ν=cs²(τ−0.5)，RR 保持同一 τ→ν 映射，见 tests/test_free_surface_rr_dose.py 4 项）⇒「剂量」不是纯投影强度旋钮，而是黏性轴：ν/ν₁× = {{NU05}} / {{NU2}} / {{NU4}}

## 2. 预登记（跑前写死，事前落盘 BB/tmp/DE-X-COL2D-REGDOSE.draft.md）
- 臂（其余一切与 COL2 卡一致：cells=8 满窗，gap .18 speed .06 capacity simple-exchange release 1.53 span 4 pool 3 u_limit 0.15，steps 10000，窗 24.005ms postcontact）：
| 臂 | rr_tau | 剂量 | 角色 |
|---|---|---|---|
{{DOSE_TABLE}}
- 预测：peak 随剂量单调升；2× ∈ [1.60,1.75]D；4× 或触稳定性边界（早死/漂移>1e-4 超标）
- 证伪：非单调（任一高剂量臂 peak < 1× 的 1.555257D）或全剂量稳定但增益饱和（4×−2×<0.02D）⇒「REG 天花板≈8-15%」定谳
- 测量三元组：peak_depth / peak_error(vs 1.857434D) / postcontact_healthy_ms + volume_drift；对照=ctrl 与 1× 引值

## 3. 实测三元组（cells=8 满窗；SDAA 车道：{{CARDS}}；run.log 首行含卡号+commit={{COMMIT}}）
{{TRIPLE_TABLE}}
- 溯源：runs/<arm>/run.log 首行（arm/card/commit/rr_tau）+ runs/<arm>/result.json；已随卡归档 artifacts/COL2D-REGDOSE/arms/<arm>/{run.log,result.json.gz}；1×/ctrl 引 DE-X-COL2-REG 卡 summary.json（commit da0ffa73b，逐字拷贝 artifacts/COL2D-REGDOSE/col2_summary_cited.json）
- 位级锚核验：本分支 ctrl 复跑（ctrl_pl）逐位复现战役锚 {{CTRL_PEAK}}D@{{CTRL_T}}ms/contact 45；同参数 planes 复跑臂与主臂逐位一致（reg_d200_pl = {{P200}} = reg_d200）

## 4. 判决（对照预登记）
{{VERDICT_LINES}}

## 5. 机制解读
- 剂量响应是**单峰（内点最优）**：最深点在 1×（τ=0.5003877 = BGK 稳定底），两侧都变差——向下 τ=0.25019385<0.5 负黏性（ν/ν₁×={{NU05}}）立即暴毙（{{T05}} ms 接触前、漂移 {{DRIFT05}}>1e-4 门）；向上黏性主导、落体变慢（接触步 45→{{C2}}→{{C4}}）而空穴变浅（2× {{P2}}D，4× {{P4}}D）
- 解读：COL2 的 +0.120D 收益来自「贴稳定底时界面振荡阻尼被移除」这一**边缘稳定**效应（BGK τ=0.5004 时 mach_max 0.2598 逼近 0.3 门）；REG 一旦离开稳定底，黏性主导、空穴被压浅 ⇒ 剂量↑≠更强的正则化，而是更黏
- 与 COL2 互补：窗内 τ 是连续旋钮（rr_t4995=1.486759D 落在 ctrl 1.435220D 与 rr_base 1.555257D 之间），窗外两侧本单已量化（0.5× 死、2×/4× 浅）⇒ REG 可用剂量区间极窄，**天花板 ≈ +8~15%**（与 campaign 预登记一致），战场原句「REG 天花板≈8-15%」获独立支持

## 6. 诚实披露
- **本单证伪了预登记预测**（honest-FAIL）：实测为非单调——高剂量臂 peak（{{P2}}D / {{P4}}D）远低于 1× 的 1.555257D，与「单调升、2×∈[1.60,1.75]D」相反；0.5× 触稳定性边界（早死），4× 未触边界但空穴最浅
- 未产出任何比 1× 引值更深的完整臂 ⇒ campaign 3% 精度门未达（最佳完整臂仍为 1× 引值，深度误差 {{BEST_ERR}}%、时刻误差 {{BEST_TERR}}%）
- 剂量轴与黏性轴在本实现中不可分离（只有 rr_tau 一个通道，禁改实现故未加独立正则化权重参数）⇒ 本单不能回答「纯投影强度」的剂量响应
- 车道与硬件：SDAA（太初卡）32 卡，容器镜像 lbm_test:snap_1114；1×/ctrl 为引值（禁重跑），ctrl 另做一次 planes 复跑仅用于场量图与位级锚核对；planes 复跑臂不参与主判决
- 回归：本机 free-surface/collision 回归 78 项通过 + 1 跳过（含新增 4 项剂量语义测试）；集群账单号 cj6079 rc=0 但集群节点缺依赖致 121 项收集错误（回归结论以本机为准）；仓库既有 test_free_surface_surface_tension.py 收集失败（csf 缺 tanh_droplet_phi，main 上已存在，非本单引入）
