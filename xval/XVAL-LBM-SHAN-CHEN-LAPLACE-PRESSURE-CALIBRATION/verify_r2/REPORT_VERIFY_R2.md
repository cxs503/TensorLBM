# R2 复核报告 — XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION

锚点 XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION（lbm-principles-practice/code 的
shan-chen-laplace-pressure-calibration 做法）｜仓库 TensorLBM｜分支 agent/m21403。
上游：首轮 m20921（2D 四半径 Laplace 标定 σ=0.0555）、R1 m20984（平面界面 Maxwell 腿 + τ/力格式敏感性）。
本轮只补 R1 未做项中与本锚点直接相关的两条：3D 腿与格点截断判据。判据：owner <15%，并逐项回答 <3%。

## ① 案例介绍
SC94 单组分伪势：psi=1-exp(-rho)、物理 G_eff=-5（库调用 G_lib=+5）、tau=1、cs^2=1/3、D2Q9/D3Q19。
R1 遗留：(a) 3D 球形液滴 dp=2σ/R 从未独立复核；(b) 模板四阶矩 κ3 对 σ 的支配性只作讨论。
本轮把 (b) 升级为可否证的预言：若 σ 由 Korteweg 截断应力支配
（sc_surface_tension 式(6) σ_K=|G_eff κ3|∫(∂nψ)^2 dn），则 κ3(D2Q9)=κ3(D3Q19) ⇒ 预言 σ_3D=σ_2D。

## ② 怎么做的
五条腿，判据在 predeclared.json 预登记（P1..P5），运行前固定：
T 模板算术（库自带速度集/权重算 a1,a3,κ3）；M SC EOS 上的 Maxwell 等面积构造（对 pc 二分，∫dp/ρ=0，
按 spinodal 分段找三根）；K σ_K 的 1D 实现 vs 解析闭式 2|G_eff κ3|Δψ²/(3w) 自检；
F 2D/3D 平面液膜（力学平衡 p_l=p_v 为格式自洽闸门 + σ_K）；D 2D/3D×R=10/14/18 液滴扫描
（σ_i=Δp·R_eq/(d-1)，逐半径一致性 + 跨维比值）。求解器零手写，只用库 collide+stream；
数字经 artifacts/r2_summary.json 落盘后由脚本渲染。执行器 verify_r2/run_r2.py。

## ③ 计算结果
3.1 模板四阶矩比（T 腿，已跑）
| 维度 | 速度数 | a1 | a3 | κ3 |
| 2D D2Q9 | 9 | 0.3333333 | 0.0555556 | 0.1666667 |
| 3D D3Q19 | 19 | 0.3333333 | 0.0555556 | 0.1666667 |
比值 1.000000 ⇒ 预言 σ_3D/σ_2D=1。

3.2 Maxwell 构造（M 腿）vs R1 参考：rho_v 0.152696 vs 0.152895（0.13%）、
rho_l 1.928661 vs 1.928866（0.01%）、p_c 0.034188 vs 0.034214（0.08%）；
等化学势残差 1.56e-4；spinodal 压力窗 [-0.007695, 0.044175]。

3.3 Korteweg 锚自检（K 腿）：w=3 时数值 vs 闭式 <3%（回归测试锁定）；w=1.5 处最大偏差 10.2%
（中心差分在陡界面上的截断误差）⇒ 该锚仅 w≳2 可用，薄界面须用拟合剖面 sigma_K_tanh。

## ④ 结果图
artifacts/figures/fig_r2_theory.png：左=SC EOS 曲线 + spinodal 带 + Maxwell 得到的 pc/rho_v/rho_l；
右=Korteweg 锚数值 vs 闭式随界面宽 w（w=1.5 偏差最大，w≥3 时 <3%）。
本轮无场量云图：F/D 数值腿未跑，不用上游图冒充本轮结果。

## ⑤ 注意点
1. F/D 两腿 NOT_RUN 而非 PASS：两次集群递交（cj3411 96 核、cj3442 24 核）30 min 仍排队，
   本单时限内未取回终态；按纪律不计为通过。复算：
   python xval/.../verify_r2/run_r2.py --stage all --jobs 6 --threads 4（3D R=18/L=72 约 20 min/任务）。
2. κ3 相等只是预言：σ_3D=σ_2D 必须由 D 腿给出，本轮不能声称跨维一致性成立。
3. σ_K 公式为启发式（不含 cs^2 因子），2D 上与 Δp·R 的经验差约 7%，只能作 15% 量级粗锚。
4. σ 适用域极窄（R1）：仅 (tau=1, velocity-shift/EDM)；tau=0.7/1.5 与 Guo 偏差 -33%…+83%。
5. cj3411 与 cj3442 同命令重复递交，前者为降资源重投前遗留，建议 kill cj3411。

## ⑥ 误差判定（逐项回答 <3%）
| 量 | 基准 | 误差 | <3% | <15% |
| κ3_3D/κ3_2D | 1 | 0.00% | 是 | 是 |
| rho_v（本轮 Maxwell） | R1 0.152895 | 0.13% | 是 | 是 |
| rho_l（同上） | R1 1.928866 | 0.01% | 是 | 是 |
| p_c（同上） | R1 0.034214 | 0.08% | 是 | 是 |
| σ_K 数值 vs 闭式（w=3） | 解析 | <3% | 是 | 是 |
| σ_K 数值 vs 闭式（w=1.5） | 解析 | 10.2% | 否 | 是 |
| σ_3D vs σ_2D（P1 主判据） | 预言 1.0 | - | NOT_RUN | NOT_RUN |
| 3D 平面 p_l=p_v（P2）/ P3 / P4 / P5 | - | - | NOT_RUN | NOT_RUN |
诚实结论：本轮不主张锚点 <15% 已达成。已交付：可复现执行器 + 预登记判据 + 两条通过的解析/算术腿
+ 一条可否证的跨维预言；数值证据待集群续跑。

## 复算
run_r2.py --stage theory（T+M，秒级）；make_figures_r2_theory.py；run_r2.py --stage all（集群）；
pytest tests/test_xval_sc_laplace_r2.py -q → 5 passed。
