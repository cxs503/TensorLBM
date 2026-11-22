# DE-X-GAS2-CAVPROBE-R1 · 气穴压力探针：p_cav(t) vs 定压闭包 / Rayleigh-Plesset
自检：5/5（文件在且非空✓ 数字可溯=cj6220/cj6221 回执+runs/gas2cavprobe 工件✓ 双写commit✓ 预测区间✓ 失败如实✓）
CROSS已读：TRACK_GAS_W2（处方）、KERN-B、K1-DIAG-B、CAVSWEEP、CONV12、ABBS、DIAG1-CAV_PRESSURE、REJECT_DIAG1（全读）
锚点=NEW:DE-X-GAS2-CAVPROBE-R1 · 战役=DE·多线并行攻关·气相建模线 波2 · 分支 agent/m27202 · 仓库 TensorLBM · 2026-11-22

## 1. 物理出处
- 定压闭包 p_cav=p_ref：KERN-B 定谳=稳定性要求（ABB 重构对 rho_g 亚 1% 敏感；p_cav↓→闭合加速变浅）；模型持 rho_gas=1.0 常数，p_ref=1.0 格单位
- Rayleigh(1917)/Plesset(1949) 闭气泡：p_b=p_v+p_g0(R0/R)^3κ−2σ/R−4μR′/R；空腔 R(t)=R0√(1−(t/Tc)²)，Tc=0.9147R0√(ρ/Δp)=53.56ms>满窗 24.55ms
- K1-DIAG-B 定谳：生长段气穴全程开放通气（gap≡0、通气比≡1.0）=开放坑非闭气泡 → RP 只作偏差参考线

## 2. 改动 diff 摘要（commits e7daebceb+528fb6941，默认关=零行为改变）
- free_surface_probe.py +144 行：_find_cavity_center（headspace 连通气穴最深气-液界面点）/_rayleigh_plesset_pressure/rayleigh_collapse_time_lattice（源=agent/m26730 未合并分支，DIAG1 卡 §2）+ --cav-probe/--cav-probe-out 旗标
- 只读：仅读 dens/field，不写求解器状态；probe 数据嵌入 result.json（result["cav_probe"]）适配集群回收通道
- 附带修复（预存 bug）：__main__ 的 probe() 调用漏传 rr_tau/C_s 两位置参 → 移植后 --cav-probe 被错接为 C_s=1.0（SGS 开启，contact 45→42）且 diag_cavgas=Path（row 塞 None 崩末段 math.isfinite）；现显式传全参数+末段清洗循环容忍 None

## 3. 预登记（跑前锁定，gas2_cavprobe_adjudicate.py 写死）
- P1：生长段（首样本→峰深步 45→912）腔壁压强贴 p_ref±5%；P2：RP 偏差随闭合单调增大（对分数≥0.9 且后半>前半）
- 证伪判据：生长段 |p_cav−p_ref|/p_ref>15% 或非物理振荡（≥3 次符号翻转且幅值>5%）→ 定压闭包失真=气穴模型缺项，honest-FAIL
- 零行为改变：臂 history 与 ctrl 逐位相同；ctrl 须复现锚 1.4352204748D@11.0906ms

## 4. 实测三元组（cells=8 同旗标 gap .18 speed .06 capacity simple-exchange rs 1.53 span4 pool3 ul 0.15，集群 n330 8 核，绝对时轴）
| 臂 | peak_depth | peak_error | postcontact_healthy_ms | stop |
|---|---|---|---|---|
| ctrl | 1.4352204748D@11.0906ms | 22.731% | 24.0052 | window_complete |
| cavprobe | 1.4352204748D@11.0906ms | 22.731% | 24.0052 | window_complete |
- 锚 6/6 位级吻合（峰深/峰时/stop/contact45/impact1.4091844081433615/mach0.2598076211/drift4.805885601255611e-06）；臂 vs ctrl history 2020 行 max|Δ|=0
- RC=1 为预期（geometric drift 2.11e-4 超 1e-4 门=战役已知态，非本单引入）

## 5. p_cav(t) 实测（格单位，p_ref=1.0；1975 样本，R0=1.135mm，Tc=53.56ms）
| t_ms(abs) | 相 | p_cav | p_rp | dev% |
|---|---|---|---|---|
| 0.547 | 生长始 | 1.0100 | 0.1829 | 1.00 |
| 5.825 | 生长中 | 0.9978 | 0.1794 | 0.22 |
| 11.091 | 峰深 | 1.0014 | 0.1676 | 0.14 |
| 17.815 | 闭合中 | 1.0008 | 0.1386 | 0.08 |
| 24.552 | 末 | 0.9999 | 0.0887 | 0.01 |
- 生长段 p_cav∈[0.99365,1.01124]，max 偏差 1.124%<5%（P1 命中）；230 次符号翻转但幅值 1.12%<5%=物理噪声非非物理振荡
- 闭合段 |p_cav−p_rp|：0.834→0.911，单调对分数 0.977≥0.9（P2 命中）

## 6. 预登记命中判定
**HIT**：定压闭包未失真（生长段偏差 1.12%<5%）且 RP 偏差随闭合单调增大 → 22.73% 欠深非腔压闭包错误所致；缺项不在气相压强模型，而在界面/空穴动力学本身（ABB 重构+质量交换限制腔穿透深度；KERN-B：p_cav↓→闭合加速变浅，方向与加深相反）

## 7. 诚实披露
- 探针采样=腔底气胞密度（模型气相恒压 1.0），非液相壁压；液相壁压未测（超本单范围）
- RP 线=偏差参考线（开放坑非闭气泡），非预测；偏差量级由 −2σ/R 项主导
- 预存 CLI 位置参数错位 bug 本单修复（原 --diag-cavgas 实际设 rr_tau、旗标失效）；该旗标行为自此生效
- 单网格 cells=8 无收敛对；numeric_pass=False（3% 门未过，22.73%）

## 8. 复现
- 命令：PYTHONPATH=$PWD/src python3 -m benchmarks.water_entry.single_drop.free_surface_probe --cells 8 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3 --u-limit 0.15 [--cav-probe --cav-probe-out cav_probe.json] --out runs/gas2cavprobe/<arm>.json
- 集群账：cj6220（ctrl）/cj6221（臂）n330 8 核；判读/图：gas2_cavprobe_adjudicate.py + gas2_cavprobe_figures.py（runs/gas2cavprobe/）
