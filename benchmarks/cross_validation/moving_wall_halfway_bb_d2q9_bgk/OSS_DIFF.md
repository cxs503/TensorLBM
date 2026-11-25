# OSS_DIFF — 动壁半程反弹闭包 vs 开源实现

横向对照集群 `~/oss-cache` 内同类实现源码（结构化盘点见 `oss_inventory`）：

| 源 | 位置 | 壁面处理 | 与 TensorLBM 的关系 |
|---|---|---|---|
| pylbm/pylbm (`f9f5d16`) | `pylbm/boundary.py:532 BounceBack`, `:920 LinearBounceBack` | `BounceBack` 参数化为 `alpha=2, delta=1/2`；速度边界经**反对称平衡态** `e^-_q = (f^eq_q − f^eq_opp)/2` 进入 | **数学同构**（推导见下）：TensorLBM 的 Ladd 项即 `−alpha e^-_q` 在 `alpha=2` 的展开 |
| lbm-principles-practice/code (`oss_inventory#4`) | 教科书码 | half-way BB + moving wall with Ladd momentum correction | 与本仓闭包同式（`f_opp = f_q − 2 w_q rho (c_q·u_w)/cs²`） |
| Maarten-vd-Sande/lbm (`oss_inventory#6`) | `lbm/streamcollide.py:59-66,92-94` | 节点式反弹，无动壁项（该仓为自由表面 LBM） | 不适用本无自由面问题；也说明动壁项须自行加入 |
| lssfau/lbmpy (`oss_inventory#9`) | `NoSlipLinearBouzidi` / `QuadraticBounceBack` | 亚格壁位插值弹回 | 平面壁不需要；弯曲/斜壁时才应迁移的下一步 |

## pylbm 参数化 → Ladd 项的推导（源码级对照）

`LinearBounceBack.generate()` 的入口分布为

    f_qbar(x_b, t+1) = a_hat f_hat_q + b_hat f_qbar + beta f_q(t+1) − alpha e^-_q(x_w, t) + K (g^- + S^-)

其中 `a_hat = −1 + alpha(1/2 + delta)`, `beta = 1 − alpha delta`, `b_hat = 1 − alpha/2`。
取 `alpha = 2, delta = 1/2`（即 `BounceBack`）：`a_hat = 1, beta = 0, b_hat = 0`，于是

    f_qbar(x_b, t+1) = f_hat_q(x_b, t) − 2 e^-_q(x_w, t) = f_hat_q − 2 w_q rho (c_q · u_w)/cs²

即 TensorLBM 的 `f_new[q] = f_pre[opp(q)] + 2 w_q rho_w (c_q·u_w)/cs²`（索引重标号后逐项相等）。
**结论**：壁面速度只经反对称平衡态进入 ⇒ 不产生质量源（实测质量漂移 ≤ 3.33e-10 %）；
系数 `alpha=2` 由"壁面位于半个链长处"唯一确定 ⇒ 无可调参数。

另给出一条**顺序约束**（pylbm 的 `f_hat` 是碰撞后分布）：动量项须加在反射后的碰撞后分布上；
TensorLBM 侧实测把该项加在碰撞前会被壁行 `f_pre[opp]` 覆盖而完全失效（误差 93.94 %，见 REPORT §⑥）。
