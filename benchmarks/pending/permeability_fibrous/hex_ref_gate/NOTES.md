# hex_ref NOTES (append-only)

## 2026-09-29 R3 开工
- prereg v1.0 冻结（sha256 两端一致，见上/下）：方法 M0 MAC-stair + M1 备变体、门=方阵 5 点外推 ≤0.5%、hex=97/56 纵横比锁定阶梯+终端校正、防选购条款 §7。
- 环境就绪：/nfs/wangxi/venvs/hexref（numpy 2.5.3/scipy 1.18.1，pip 新建）；本地机 3GB 仅冒烟。
- 已读输入（开工前，全程仅此）：fibrous_perm/prereg.md §0-§6（几何/口径/判据锁定）+ 本地 /root/w4a_ctrl/fd_stokes_ctrl.py（思路借鉴）。未读：fibrous_perm/{out,out_diag,DIAGNOSTICS.md,NOTES.md}、任何 K_sim 数值。
85cd2936498b35e37000d6152cab10df0a0821e2cb95ba94b8d76b88e0ee0bac  prereg.md @5090

## 2026-09-29 R3 解算器开发记录（正式档开跑前）
- 装配三 bug 全部在**正式档开跑前**本地抓住并修复（行对角 diff harness + N=104 与 ctrl 交叉验证）：lap1d 对角 −1/h²→−2/h²、lil.setdiag 序列语义（只设首元素）、Gx/Gy 块维度。修复后与 W4-A ctrl 实现在 N=96 逐位一致（见下条 A/B）。
- 直接鞍点系统 SuperLU 分解病态慢（N=96 需 103s，COLAMD/主元阈值均无效）→ 启用 prereg §5 预登记回退：**Schur-CG**（SPD 速度块一次分解 + FFT 周期 Poisson 预条件 CG 解压力）。
- **A/B 验证**（回退条款要求）：N=96 直接法 f=100.34767235435905 vs Schur f=100.34767235435888，相对差 1.7e-13（LU 舍入级）✓ 采纳。直接法 N=96 档案保留 out/cases/（MMD_AT_PLUS_A 标记）作锚。
- 方阵三路队列 a/b/c 已 nohup（logs: out/logs/queue_sq_*.log），幂等可断点续跑。

## 2026-09-29 watcher 确认 + 方阵队列进度快照

- gate_watch.sh 确认已运行（PID 2421478, 14:25 启动）：轮询 3 个 QUEUE_SQ_*_ALL_DONE 标记，
  齐后自动跑 src/extrapolate.py sq -> out/sq_gate.json 并在 out/logs/gate_extrapolate.log
  追加 GATE_READY。
- 队列进度（14:4x 快照，单例耗时见 out/logs/queue_sq_*.log）：
  - a: vf0.3 done N<=512 (f=102.05400793 @512, cg_t=187.1s)；剩 768/1024/1280 + vf0.4 全梯
  - b: vf0.5 done N<=768 (f=526.26065721 @768, cg_t=487.9s)；剩 1024/1280 + vf0.6 全梯
  - c: vf0.7 done N<=768 (f=12957.99052910 @768, cg_t=188.1s)；剩 1024/1280/1536 + vf0.75 尝试梯
- 观察：cg_t 随 vf 升高而降（固体面被消元、未知量变少），内存峰值 ~0.001G 量级 RSS 报告
  偏低（maxrss 采样时机），机器 1007G 内存无压力。
- 估算：长杆 = 队列 a（~3-4h）；全部完成后 watcher 自动出门檄件。

## 2026-09-29 hex 几何前置自测（无流场求解，不产生任何 K/f 值，anti-shopping 不触碰）

- src/geom_selftest.py 本地+5090 双跑通过（out/logs/geom_selftest.log）：
  1. dist2_to_cylinders vs 暴力 7x7 镜像：sq 逐位、hexA 2.5e-15、hexB 1.1e-14
  2. 掩码 phi_actual 收敛 vf：hexA 0.70 全档 <=0.03%；0.75 最粗 m=2 +0.147% -> m>=4 <=0.025%；
     hexB m=20/28 <=0.06%
  3. 格子度量：hexA d_nn=1.000040, gap/r=0.2765/0.1993（任务书 0.276/0.199 吻合）；
     hexB 0.2769/0.1997
- 确认 R^2 用名义 vf（R2=vf*W*H/(n_cyl*pi)），楼梯误差只进 f 由 Richardson 处理，不进 R 归一化。
- 队列仍在滚（见 poll_status，本地 b18npo3vk 每 5min 一查）。

## 2026-09-30 控制器门判 v1 + 阶梯延长指令（执行记录）

【判决（控制器独立核验，本轨确认）】当前阶梯下冻结门不成立：
- 裸最细档 err：vf0.3 -0.34% / 0.4 -0.40% / 0.5 -0.58% / 0.6 -1.14% / 0.7 -1.97% / 0.75 -3.49%（121903.84@N1792）
- v1 外推器自由 alpha 撞 0.30 扫描下界 5/6 点（vf0.7 q_inf=14564.8=+7.73%、0.75 +20.8% 边界伪值），
  best-SSE 自由 alpha 在慢收敛尾上不可辩护 -> 估计器协议修正
- 根因 = 收敛尾慢（vf0.7 尾段有效阶 p~1.3-1.4，N=1536 不够），非求解器错

【延长阶梯（门 0.5% 一字不动）】src/queue_sq_ext.sh 09-30 07:41 全并行 nohup：
- vf0.7 += 2048/3072/4096；vf0.6 += 1792/2304；vf0.4 += 1536/1792；vf0.5 += 1536/1792（9 rung）
- 同一 Schur 求解器同容差链；cg_t~N^3（实测 1636s@1536 vf0.7），长杆 4096 ~8.6h
- watcher2（PID 2465480）候 QUEUE_SQ_EXT_ALL_DONE -> src/extrapolate.py sq -> out/sq_gate2.json

【估计器协议 v2（estimator bug fix，非门改动）】src/extrapolate.py 重写：
- 三估计器并列：fixed-alpha LSQ（alpha=1.0 与 1.5 两档，窗口=最细 4/5/6，
  点值=离窗口中位数最远的窗口值即 worst-window）+ Aitken3（末三档经典 delta^2）
- 判过门 = 三者全 |err|<=0.5%（固定 alpha 不落扫描边界由构造保证）
- 自由 alpha 降级为纯诊断（alpha_at_scan_boundary 旗标 + SSE 翻倍带照报）
- 带照报不进门控：fixed=窗口极差与次阶 25%，aitken=delta^2 修正量
- 协议同一性验证：本轨 aitken3 五点与控制器 Aitken3 逐位一致
  （-0.073/-0.888/-0.341/-0.897/-1.392% vs -0.07/-0.89/-0.34/-0.90/-1.39%）

【v2 当前基线（延长档未入，/tmp/sq_gate_v2_preview.json）】
- vf0.3 PASS（-0.169/-0.228/-0.073%）；vf0.4 FAIL（aitken -0.888）；vf0.5 FAIL（fixed15 -0.526 压线）；
  vf0.6 FAIL（-0.473/-0.905/-0.897）；vf0.7 FAIL（-0.758/-0.983/-1.392）；0.75 尝试档散开
- 待延长档齐后 out/sq_gate2.json 出正式门报

【防购物条款】不变：未读任何 W9-C 结果文件；延长与协议修正只向 S&A 方阵表收敛；
hex 仍冻结待门。若延长后 vf0.7 仍 >0.5%：停 + 保档 + 报 4096 尾段 alpha 与三估计器，
门控更换归 owner 裁决。

## 2026-09-30 hex_finalize.py 估计器同步升级 v2（门后、hex 启动前完成）

- 与 extrapolate.py v2 同病同修：原 blind_extrap 用自由 alpha best-SSE 扫描（控制器已判死）。
- 现复用 extrapolate.py 原语（fit_fixed/aitken3/fit_free import，单一实现不漂移）：
  三估计器 = fixed-alpha 1.0/1.5（窗口最细 4/5/6，点值=窗口中位数）+ Aitken3；
  终值 = 三估计器点值中位数（prereg §11.4）。
- 带：band（sum 口径：三估计器极差 + 最大分量带 + 斜率不确定度）与 band_max
  （max 口径，方阵 §6 约定）双报；自由 alpha 留诊断字段。
- 合成阶梯端到端烟测（本地，未触碰真实 hex case，anti-shopping 无涉）：
  真值 K_inf=1.60e-3、alpha=1.35、斜率 2.0 -> 三估计器 1.600015/1.599887/1.599940e-3，
  自由 alpha 诊断=1.35（命中真值），斜率恢复 2.0018，终值 -0.017%。
- 延长队列滚动中（9 档并行），watcher2 候 QUEUE_SQ_EXT_ALL_DONE。

## 2026-10-01 门判 v2 终判（延长档齐，prereg §11.5 停止条款执行）

【终判（out/sq_gate2.json，51 case 池）】
- vf0.3  -0.1691/-0.2281/-0.0728%  PASS
- vf0.4  -0.0852/-0.2504/-0.3393%  PASS
- vf0.5  +0.1990/-0.2448/-0.3901%  PASS（最差 -0.390%）
- vf0.6  -0.1880/-0.5521/-0.2829%  FAIL（fixed15 worst-window 超 0.5% 门 0.052pp）
- vf0.7  +0.1559/-0.4320/-0.7967%  FAIL（aitken3；fixed10/fixed15 在门内）
- vf0.75 +1.7616/-0.9416/-7.6708%  FAIL（尝试档，不阻塞，如实披露）

【独立复核】控制器用自有拟合器对 18 个估计器值 + 6 档 gate_pass 重算，与本轨
out/sq_gate2.json 逐位一致（三估计器协议同一性在 Aitken3 五点已预先验证）。

【尾段有效阶】控制器算：vf0.6 p 1.34->1.39（恢复）；vf0.7 1.34->0.96->0.90->0.58（塌陷）。
本轨独立佐证（滑动三点自由 alpha，不同估计器口径）：vf0.6 尾三元组 0.05/1.23、
vf0.7 尾三元组 1.11/2.01——无稳定单一阶，尾段 ragged 结论一致。

【free-alpha 诊断（协议 v2 降级，不进门控）】vf0.7 alpha=1.40（不撞界）
q_inf err=-0.4992%。

【脚注（如实披露）】sq_vf0.7 N=4096 Schur-CG 在 10000 迭代帽处
cg_res=1.0878374333392318e-10，超 prereg §5 的 1e-10 线 8.8%；对 f 的影响
~1e-8 量级，不改变任何判决（f=13400.80456415629）。

【停止条款】prereg §11.5：不产 hex 值、不启动 M1、不动门；防购物条款 §7
继续有效（未读 fibrous_perm/ 任何结果文件）。owner 报告 = out/gate2_owner_report.md。
