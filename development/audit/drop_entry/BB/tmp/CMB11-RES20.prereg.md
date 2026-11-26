# CMB11-RES20 预登记（写死后跑，跑完零改动）
锚点=NEW:DE-X-CMB11-RES20 · 分支 agent/m27315 · 仓库 TensorLBM · 2026-11-25
触发=CMB10-RES16 判据③（ctrl 增量 C8→C12 +0.0872 → C12→C16 +0.1108 递增，分辨率轴未饱和）

## 双臂（cells=20，同物理旗标，满窗）
旗标：--gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --span 4 --pool-depth 3 --u-limit 0.15 --steps 10000
dt = 16/20 × 0.006080ms = 0.0048643ms/步；24ms 窗 ≈ 4934 步（contact 后）
- R20-ctrl = BGK
- R20-tuned = rr_iface+SMAG.075+HYDRO24.5（--rr-tau 0.5003877 --rr-iface --cs 0.075 + TL_FS_HYDRO_Z_REF=24.5）

## 预登记判据
(a) R20-ctrl 峰深 ∈ [1.70, 1.78] D
(b) R20-tuned 峰深 ∈ [1.74, 1.79] D
(c) 外推门：ctrl C16→C20 增量 ≤0.08 ⇒ 分辨率轴封顶定谳（网格无关性达成）；>0.08 ⇒ 报网格无关性缺失（cells=24+ 建议或定谳分辨率主导缺口）
(d) 双门不变：峰深 >1.8017D 且 峰时刻 <11.4334ms(pc)
(e) 关注 ctrl 时刻税是否续推晚（vs C16 的 12.6228ms）

## 诚实披露规则
- tuned 臂 C16 有 health_limit 早死前科（step 3341/20.31ms，drift 1.002e-04 越 1e-4 门）；若 C20 早死按 honest-FAIL 如实披露：峰前有效 + 死点 step/时刻 + drift 值。
- 全部数字程序化提取 runs/cmb11_r20_*/result.json，零手抄。

## 前锚（程序化复核值，来自前序工作树 result.json）
C8-ctrl 1.435220D@10.5433ms · C8-tuned 1.732906D@11.9053ms
C12-ctrl 1.522445D@10.0204ms · C12-tuned 1.739759D@12.7201ms
C16-ctrl 1.633265D@12.6228ms · C16-tuned 1.747348D@11.6196ms(health_limit)
实验锚=1.857434D@10.39446ms pc（Santini 2021, fluids7020050）
