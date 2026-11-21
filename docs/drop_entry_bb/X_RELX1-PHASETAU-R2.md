# DE-X-RELX1-PHASETAU-R2 · 界面胞 fill 加权弛豫率 τ_iface=τ·g(fill) · 剂量矩阵
自检：5/5（文件在且非空✓ 数字可溯=runs/*/result.json+run.log 首行✓ 双写✓ 预登记对照✓ 失败臂照登✓）
CROSS已读：TRACK_RELX_W1（PHASE-TAU 处方+预登记）、GAS1-GASVISC（分相τ先例+锚核验口径）、MMC6-DYNANGLE-B（SDAA 车道本地副本先例+位级锚核验口径）、CONV12/KERN-B（锚三元组）
锚点=NEW:RELX1-PHASETAU-R2-界面胞修正弛豫率τ_iface=τ·g(fill)（续做草稿禁重做） · 战役=DE·碰撞与弛豫格式专线 · 分支 agent/m26765 · 仓库 TensorLBM · 2026-11-21

## 1. 处方与物理出处
- 处方（草稿定谳，禁改）：界面胞(fill<1)用 τ_iface=τ·g(fill)（结构空间分离，非全局τ_s微调），液相胞保持物理τ；默认关=零行为改变
- 物理出处：液气界面有效动量输运异于体相（部分填充胞+气相边界）；KERN-B 证伪 p_cav↓变浅提示界面闭包对弛豫敏感；T4-RED 定谳界面 ABB 动量注入为奇数(反对称)模现象
- 剂量参数化（本单实现）：g(fill)=1+dose·(1−fill)，仅对 flags==INTERFACE 胞施加；LIQUID/GAS/SOLID 保持物理τ；dose=0 走原标量路径（位级不变）；负剂量加 BGK 稳定底 τ_iface≥0.5+1e-4
- 剂量通道：probe 增 iface_tau_dose 参数（默认 None→env WE_IFACE_DOSE→phase_tau_dose.json→0.0）；SDAA 共享包装器不传新 env，剂量经树内 phase_tau_dose.json 随 tar 送达（MMC6 dca_config.json 同型）
- 附带缺陷修复：m26728 合并把 `args.gas_nu_ratio` 引入 probe() 函数体（args 仅存在于 __main__）⇒ SDAA 车道 NameError 秒退；本单将 gas_nu_ratio 提为 probe() 形参（默认 1.0=legacy），恢复车道且默认零行为改变

## 2. 预登记（跑前写死）
- 臂（cells=8 满窗，同旗标：gap .18 speed .06 capacity simple-exchange release 1.53 span 4 pool 3 u_limit 0.15，steps 10000）：
  | 臂 | dose | g(fill) 范围(fill 0→1) |
  |---|---|---|
  | pt_ctrl | 0.0 | 1（=ctrl，位级复现锚） |
  | pt_p05 | +0.5 | 1.5→1 |
  | pt_p10 | +1.0 | 2.0→1 |
  | pt_p20 | +2.0 | 3.0→1 |
  | pt_m05 | −0.5 | 0.5→1（反向符号；τ=0.500388 近 0.5 ⇒ 低 fill 胞触稳定底，预期近 no-op，如实记） |
- 预测（TRACK 卡预登记）：有效剂量臂峰深∈[1.50,1.60]D（对实验 1.857434D 误差 13–19%）
- 证伪判据：所有满窗臂峰深<1.49D，或界面振荡发散（mach>0.3 早死）
- 位级锚门：pt_ctrl 必须复现 1.4352D@11.091ms（±0），窗口 24.005ms（postcontact healthy）
- 三元组口径（全卡一致）：峰值D / 峰时刻(绝对ms) / 窗口(postcontact healthy ms)

## 3. 实测三元组（cells=8 同旗标，SDAA 车道，32 卡自动挑闲）
| 臂 | dose | stop | 峰值D | 峰时刻(绝对ms) | 窗口(ms) | 峰深误差(vs 1.857434D) | Δpeak vs ctrl (D) | mach_max | 体积漂移(end) |
|---|---|---|---|---|---|---|---|---|---|
| pt_ctrl | 0.0 | window_complete | 1.435220 | 11.091 | 24.005 | −22.731% | +0.00000 | 0.2598 | 4.806e-06 |
| pt_p05 | +0.5 | window_complete | 1.230877 | 8.914 | 24.005 | −33.732% | −0.20434 | 0.1250 | 1.107e-05 |
| pt_p10 | +1.0 | drop identity lost（早死@2.262ms，接触前） | 0.340797（接触前，非满窗） | 2.262 | — | — | — | 0.1249 | 2.443e-05 |
| pt_p20 | +2.0 | window_complete | 1.671619 | 12.635 | 24.005 | −10.004% | +0.23640 | 0.1246 | 4.137e-05 |
| pt_m05 | −0.5 | window_complete | 1.356833 | 10.726 | 24.005 | −26.951% | −0.07839 | 0.2598 | 3.332e-05 |

**位级锚核验（pt_ctrl vs 战役锚）**：peak=1.435220D@11.090571ms，窗口=24.005ms，mach_max=0.2598，end 质量漂移=4.806e-06，contact=45（战役锚：1.43522D@11.091ms / 24.005ms / 0.2598 / 4.8e-6 / contact 45）→ **逐项吻合（位级锚成立，±0）**
- 溯源：runs/pt_ctrl/run.log 首行 `arm=pt_ctrl card=5 commit=cd4e0f320 steps=10000`；其余臂 run.log 首行 commit=471772ba9（card 16/17/18/20）；全部数字溯 runs/<arm>/result.json（iface_tau_dose 字段在案）

## 4. 判决（对照预登记）
- **位级锚门：通过**（pt_ctrl 逐项 ±0 复现锚，见 §3）
- **预登记区间 [1.50,1.60]D：未严格命中**——唯一加深臂 pt_p20=1.6716D **超出区间上界**（方向命中、幅度超调）；其余满窗臂（pt_p05=1.2309、pt_m05=1.3568）均<1.49D
- **证伪判据：未触发**（存在满窗臂 1.6716D>1.49D；无 mach>0.3 早死；pt_p10 早死为 tracker 身份丢失非 mach 发散，照登不复活）
- **campaign 3% 精度门：未达**（最佳臂 pt_p20 峰深误差 −10.004%；峰时刻误差 +21.6% 反而恶化，ctrl 为 +6.7%）
- 判定：**方向性支持、幅度超调、精度门未过**——界面弛豫结构分离是真实有效旋钮（剂量-效应非单调、可辨识），但本剂量网格内无臂同时满足深度与时刻的 3% 门

## 5. 机制解读
- **剂量-效应非单调**：+0.5→变浅（1.2309D，峰提前 2.18ms）、+1.0→接触前早死、+2.0→加深（1.6716D，峰推迟 1.54ms）、−0.5→略浅（1.3568D）。非单调提示界面弛豫同时调制两个竞争通道（界面动量输运 vs 界面几何/接触时序：contact 45→39/110/45），+2.0 臂的加深以峰时刻推迟为代价
- **pt_p20 机制**：界面胞弛豫最慢至 3τ ⇒ 界面奇数模（ABB 重构依赖的 f_post 结构）保留更久 ⇒ 空穴加深（误差 22.73%→10.00%，收回缺口 56%）；代价是界面阻尼降低 ⇒ 峰时刻推迟（12.635ms）
- **pt_m05 的极端敏感性（本单新信息）**：名义 −0.5 剂量下 99.9% 界面胞被稳定底钳制（τ_iface=0.5001，仅比物理 τ=0.500388 低 0.057%），峰深仍变浅 5.5% ⇒ **本战役 τ=0.500388 距 BGK 稳定底仅 3.9e-4，界面弛豫是腔峰的高增益旋钮**，反向（加速弛豫）方向在稳定底约束下剂量空间几乎为零
- **pt_p10 早死机制**：下落滴表面全为界面胞，2τ 弛豫使滴面欠平滑、形变剧烈 ⇒ DropTracker 相邻步重叠判据失败（mach 仅 0.1249、漂移 2.4e-5，非数值崩溃）
- 对战役问题的增量：欠深 −22.73% 中约 56% 可被界面弛豫结构分离收回（+2.0 臂），但峰时刻轴反向恶化 ⇒ 单旋钮不能同时修两轴，需与峰时刻修正（DCA 线）联用

## 6. 诚实披露
- 单网格（cells=8）无收敛对；剂量-效应关系为单分辨率观测
- 负剂量受 BGK 稳定底限制（τ=0.500388 距 0.5 仅 3.9e-4）：−0.5 臂实际剂量≈界面 τ 降 0.057%（fill<0.9988 胞全钳制），名义剂量与有效剂量不符，如实标注
- pt_ctrl 的 run.log 记 commit=cd4e0f320（launch 时 HEAD；补丁当时未提交），实际执行代码=cd4e0f320+本单补丁（=471772ba9，dose=0 路径位级等价由锚复现证明）；剂量臂 run.log 记 471772ba9
- 共享 SDAA 包装器全程阻塞（>120s 工具上限）且本机禁后台 ⇒ 相位拆分本地副本 tools/sdaa_fs_probe_pt.py（MMC5/MMC6 先例）；卡锁/驱动/产出布局与共享版一致，未改共享文件
- 场量云图（报告图）取自集群 x86 短跑（SDAA 驱动不落场量；TENS1 已验证 SDAA 与 x86 位级一致）；剂量矩阵数字全部来自 SDAA 车道
- 未做：12 cells 收敛对、+1.5 剂量补点（+1.0 早死与 +2.0 满窗之间的剂量前沿）、两流体/固体弹体模型（超本单范围）

## 7. 复现
- 模型：`src/tensorlbm/free_surface_lbm.py::free_surface_step(iface_tau_dose=...)` + `benchmarks/water_entry/single_drop/free_surface_probe.py`（--iface-tau-dose / phase_tau_dose.json，默认关）
- 单测：`PYTHONPATH=src:. python3 -m pytest tests/test_iface_tau_dose.py -q` → 4 passed（位级不变/掩膜局部性/稳定底/剂量通道）
- 车道：`python3 tools/sdaa_fs_probe_pt.py launch|poll|collect --arm <臂名> --tree <臂树> --steps 10000`（臂树=src+benchmarks 拷贝+phase_tau_dose.json；共享包装器 /root/tp-agent/cluster/sdaa_fs_probe.py 协议不变）
- 工件：`runs/<臂名>/{result.json,run.log}`、`runs/phasetau_summary.json`、`runs/phasetau_depth_time.png`
- 分析/出图：`python3 benchmarks/water_entry/single_drop/phasetau_analyze.py runs`、`phasetau_figures.py`、`phasetau_report.py`（数字程序化注入，零手抄）
