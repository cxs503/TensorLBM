# DE-X-CMB11-RES20 — 分辨率轴外推收官 cells=20 双臂（CMB10 判据③触发单）
锚点=NEW:DE-X-CMB11-RES20 · 战役=DE·多线并行攻关 · 分支 agent/m27315 · 仓库 TensorLBM · 2026-11-25/26
CROSS已读：CMB10-RES16（R16-ctrl 1.633265D@12.6228ms 满窗 / R16-tuned 1.747348D@11.6196ms health_limit@3341；判据③=ctrl 增量 C8→C12 +0.0872 → C12→C16 +0.1108 递增，分辨率轴未饱和）· CMB9-IFACETUNE · CMB8-IFACECOMBO · CLUSTER-CANON
自检：6/6（文件在且非空✓ 数字可溯=runs/cmb11_r20_*/result.json✓ 双写commit✓ 预登记区间✓ 失败臂如实✓ 账号合规✓）

## 1. 案例介绍
液滴入水（Santini 2021, fluids7020050）：球形液滴 1.53 m/s 释放、0.18D 间隙入静水池，考察空穴生长-峰值-回弹。几何 span 4D×池深 3D，D3Q19 Körner 自由面 LBM（capacity+simple-exchange，u_limit 0.15）。实验锚=1.857434D@10.39446ms pc。
- 双臂 cells=20 同物理旗标（--gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3 --u-limit 0.15 --steps 10000 满窗）：R20-ctrl=BGK；R20-tuned=rr_iface+SMAG.075+HYDRO24.5（--rr-tau 0.5003877 --rr-iface --cs 0.075 + TL_FS_HYDRO_Z_REF=24.5）。
- dt=16/20×0.006080ms=0.0048643ms/步；24ms 窗≈4934 步（contact 后）；实测 contact=78(ctrl)/79(tuned)，ctrl 满窗 5012 步/24.3798ms。
- 代码=main（m27225 端口三提交已并 da7f3cf6e，未重复 cherry-pick）；Gate0=单测 14/14（9 rr_collision+5 rr_iface_combo）+60 步冒烟双臂（step_limit 设计退出码 1，result.json 落盘、contact=45、drift 4.5e-05、mach 0.127 有限，双臂配置均正常启动）。

## 2. 怎么做的
- 车道=集群正典（run_compute，stage_dir=工作树，8 核/臂，cmd 无 cd，PYTHONPATH=.:src；递交前 du -sh 194M<200M 无需瘦身，跑完 du 196M 核对）。
- 关键：cells=20 单臂预计 ~700-730min，超 run 型 7200s 上限 ⇒ **shard 型 detached**（脱离会话，超 120min 不杀）；单作业单臂分开递交（账 6674 ctrl / 6675 tuned，双节点轮转均落 n315）。
- 预登记（写死后跑，development/audit/drop_entry/BB/tmp/CMB11-RES20.prereg.md，跑完零改动）：(a) R20-ctrl 峰深 [1.70,1.78]D；(b) R20-tuned 峰深 [1.74,1.79]D；(c) 外推门：ctrl C16→C20 增量 ≤0.08 ⇒ 分辨率轴封顶定谳 / >0.08 ⇒ 报网格无关性缺失；(d) 双门不变：峰深>1.8017D 且峰时刻<11.4334ms(pc)；(e) 关注 ctrl 时刻税是否续推晚（vs C16 的 12.6228ms）。
- 前锚溯源：m27225/m27256 工作树已退役，六份前序 result.json 从集群取回件（/tmp 取回目录）逐字拷入 runs/cmb11_prior/，全部数字程序化提取零手抄。
- 基础设施披露：shard 型 detached 作业的 pgrep 终态探测存在自匹配缺陷（`pgrep -af 'jobs/cjN'` 匹配到自身 bash 包装行，procs 永非空），调度器 reaper 永不判终（CMB10 的 6270/6271 同病，DB 挂 running 3 天）。本单两臂进程实际已退出（节点 pgrep 无 free_surface_probe 进程、result.json 已落盘取回），以 kill 清账（state=killed, note=user-kill，纯簿记清理，非杀计算）；cj 在飞计数=0。

## 3. 计算结果（程序化提取 runs/cmb11_r20_*/result.json，零手抄）
| 臂 | 配置 | cells | stop | 峰深(D) | 深误差 | 峰时刻(ms pc) | 时刻误差 | 末端漂移 | mach | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| **R20-ctrl** | BGK | 20 | window_complete@5012 | **1.542977** | -16.93% | 9.3054 | -10.48% | 9.646e-05 | 0.2641 | 满窗（峰 step 1991） |
| **R20-tuned** | rr_iface+SMAG.075+HYDRO24.5 | 20 | **health_limit@1572** | **1.463567**（死点值，非真峰） | -21.20% | 7.2575（死点值） | -30.18% | 1.003e-04 | 0.1784 | **早死（峰无效）** |

## 4. C8/C12/C16/C20 四点分辨率趋势（收官核心）
| 配置 | C8 | C12 | C16 | C20 | C8→C12 | C12→C16 | C16→C20 | 趋势 |
|---|---|---|---|---|---|---|---|---|
| ctrl 峰深(D) | 1.435220 | 1.522445 | 1.633265 | **1.542977** | +0.087224 | +0.110820 | **-0.090288** | **反转（C16=局部极大）** |
| tuned 峰深(D) | 1.732906 | 1.739759 | 1.747348 | 1.463567* | +0.006853 | +0.007589 | -0.283781* | *死点值，不可比 |
| ctrl 峰时刻(ms pc) | 10.5433 | 10.0204 | 12.6228 | **9.3054** | -0.5229 | +2.6024 | **-3.3174** | **反转推早（C16=局部极大）** |

**外推门判定**：ctrl C16→C20 增量 = **-0.090288 ≤ 0.08** ⇒ 按预登记字面走"封顶"分支；但增量为**负**（深度回落 0.0903D=5.5%），预登记的二分假设（增量恒正、递减=收敛）不成立——**"网格无关性达成"标签不适用**：C16→C20 解仍变化 0.0903D，且方向反转。物理结论=**分辨率轴非单调，C16 点（深 1.6333D/晚 12.62ms）是分辨率伪影局部极大**，C20 回落至 1.5430D/9.31ms（时刻反而更接近实验 10.39ms）。CMB10 的"增量递增⇒未饱和⇒须 cells=20+"判读被本单部分推翻：cells=20 未续推，而是反转。

## 5. 判决（对照预登记双门 DUAL_PASS=峰深>1.8017D 且 峰时刻<11.4334ms）
- **无 DUAL_PASS 臂**（与 CMB 系列一致）。R20-ctrl：深 1.5430<1.8017 ✗、时 9.3054<11.4334 ✓（时刻门首过，深度门未过）；R20-tuned：深 1.4636<1.8017 ✗、时 7.2575<11.4334 ✓（且峰无效=死点值）。
- **预登记逐条**：(a) R20-ctrl 1.5430 ∉ [1.70,1.78]，**未命中**（低于下界 0.157D）——预登记外推（C16 1.6333 续增）被证伪。(b) R20-tuned 1.4636 ∉ [1.74,1.79]，**未命中**（低于下界 0.276D，且为死点值非真峰）。(c) 增量 -0.0903≤0.08 ⇒ 字面"封顶"分支，但负增量=趋势反转，"网格无关性达成"不成立（见 §4）。(d) 双门无臂过。(e) ctrl 时刻税**反转推早**（12.6228→9.3054，-3.32ms），非续推晚——C16 的晚峰是分辨率伪影。

## 6. 误差判定（逐项，显式回答 <3%）
| 关键量 | 值 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| R20-ctrl 峰深 | 1.542977 D | 实验 1.857434 D | -16.93% | **否** |
| R20-ctrl 峰时刻 | 9.3054 ms pc | 实验 10.39446 ms pc | -10.48% | **否** |
| R20-tuned 峰深 | 1.463567 D（死点值） | 实验 1.857434 D | -21.20% | **否**（且无效） |
| R20-tuned 峰时刻 | 7.2575 ms pc（死点值） | 实验 10.39446 ms pc | -30.18% | **否**（且无效） |
结论：**战役 3% 双门（深度<3% 且时刻<3%）无臂达标**（honest-FAIL，与 CMB 系列一致）。

## 7. 诚实披露
1. **分辨率轴非单调（本单核心发现，推翻 CMB10 外推）**：ctrl 峰深 C8→C12→C16→C20 = 1.4352→1.5224→1.6333→**1.5430**（增量 +0.0872/+0.1108/**-0.0903**）；峰时刻 10.54→10.02→12.62→**9.31**ms。C16 是深度与时刻的**双局部极大**（分辨率伪影）；C20 回落且时刻更接近实验（-10.48% vs C16 +21.44%）。预登记 (a) 区间 [1.70,1.78] 建立在"C16 续增"假设上，被证伪。
2. **tuned 臂早死且峰无效（honest-FAIL）**：cells=20 调优态 volume_drift 于 step 1572（t=7.6467ms 绝对=7.2624ms pc）越 1e-04 门（1.003e-04），早于实验峰时刻（10.39ms pc）；死前 30 步深度单调上升（1.4531→1.4636），**报告"峰"1.463567D@7.2575ms pc 是死点值非物理峰**，真峰未知（≥1.4636D、时刻更晚）。死点较 C16（step 3341/20.31ms）大幅提前。
3. **tuned 配置非分辨率稳健（新发现）**：同物理时刻漂移对比——7ms 处 C16-tuned 5.157e-05 vs C20-tuned 9.108e-05（**1.77×**）；漂移累积速率随分辨率上升，health 门在更细网格上更早触发。ctrl（BGK）无此问题：C16/C20 末端漂移 9.657e-05/9.646e-05 持平且均满窗。
4. **时刻税反转**：C12→C16 曾 +2.60ms（推晚），C16→C20 **-3.32ms（推早）**；ctrl 峰时刻在 C8/C12/C20 三点（10.54/10.02/9.31）围绕实验 10.39ms 摆动，C16 为离群晚峰。
5. **基础设施缺陷（非本单引入）**：shard 型 detached 作业 pgrep 终态探测自匹配（`pgrep -af 'jobs/cjN'` 命中自身 bash 包装），reaper 永不判终——CMB10 账 6270/6271 与本单 6674/6675 在 DB 均挂 running；本单以 kill 清账（进程已退、工件已取回，纯簿记）。建议修 cluster_exec.py c_ps 的 pgrep 模式（如 `pgrep -af "[j]obs/cjN"` 或排除自身 pid）。
6. **局限**：cells=20 单分辨率双臂；tuned 早死非满窗且峰无效；无 --planes 场量（云图缺，深度-时刻曲线为主诊断，与 CMB8-10 一致）；节点负载高（双节点 load~39/128 且他户在飞 ~300 作业），ctrl 臂实测 ~16.3h（超预登记 700-730min 估计，排队+负载所致，非算法变慢）；结论不外推其它几何。无 Infrastructure-FAIL（两臂 rc 以集群日志为准：ctrl window_complete、tuned health_limit，均正常设计退出）。

## 8. 数据溯源
| 数据源 | 路径 | 内容 |
|---|---|---|
| R20-ctrl 满窗 | runs/cmb11_r20_ctrl/result.json | 三元组+history 5013 行（window_complete@5012） |
| R20-tuned 早死 | runs/cmb11_r20_tuned/result.json | 三元组+history 1573 行（health_limit@1572） |
| 前锚（逐字拷贝） | runs/cmb11_prior/{C8,C12,C16}_{ctrl,tuned}_result.json | 六份前序 result.json（集群取回件） |
| 图 | runs/cmb11_r20_ctrl/fig1_depth_time.png、fig2_resolution_trend.png | 深度-时刻 vs 实验 + 四点分辨率趋势 |
| 汇总 | runs/cmb11_r20_ctrl/summary.json | 程序化提取全表（cmb11_res20_analyze.py） |
| 实验基准 | benchmarks/water_entry/single_drop/fig5_experiment.csv | 1.857434D@10.394ms（Santini 2021） |
| 集群账 | 6674(ctrl,n315,window_complete)/6675(tuned,n315,health_limit) | 2 作业，shard detached，kill 清账后在飞=0 |
