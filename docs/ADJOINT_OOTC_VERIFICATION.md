# Adjoint / OOTC 梯度验证与修复（GAP-PP-3-adjoint-ootc）

mission: m17532 · 仓: TensorLBM · 分支: agent/m17532 · 日期: 2026-11-13

## 1. 背景

`tensorlbm.adjoint`（D2Q9 BGK 前向 + BPTT 参考梯度 + out-of-core 检查点
反演）已合入 main（m15481/m15705），但 feature_ledger 锚点
`GAP-PP-3-adjoint-ootc` 仍为 open：验收要求 **10^4 步 128^2 τ 反演峰值内存
< 8 GiB，且梯度与 BPTT 参考差 < 1e-5**（JOUR-DIFFLBM 第 4 节成本/内存章
所需）。本单做「验证欠账」：用独立方法（中心差分 + 复步导数）复核已合并
的 adjoint 推导与实现。

## 2. 发现的缺陷（均已修复，commit a3bcfdae）

| # | 缺陷 | 影响 | 证据 |
|---|------|------|------|
| D1 | 反向递推用了 `B mu` 而非 `B^T mu`（B = dfeq/df 非对称） | 梯度错 O(1) | 单步 adjoint 恒等式（复步精确 Jacobian）：B 版残差 0.6，B^T 版 1e-16 |
| D2 | `b_action`/`b_action_t` 多除一个 `rho`：`du_a/df_j = (c_ja-u_a)/rho` 与 `dfeq_i/du_a = W_i·rho·A_ia` 的 rho 应相消 | 梯度错 ~2.8–5 倍 | 复步 ∂feq/∂f：带 /rho 公式 max err 6.9e-4，去掉后 5.6e-17 |
| D3 | `ootc_gradient` 在 `T % k != 0` 时前向只跑 `(T//k)*k` 步，尾段被静默丢弃 | 梯度对应错误时域（J 也错） | 回归测试 T=100,k=30：修复前 J 不匹配 |

注：D1/D2 使 BPTT 与 OOTC **一致地错**（OOTC 逐位复算 BPTT 公式），
所以旧测试（OOTC vs BPTT）全绿——这正是「验证欠账」：模块内自洽 ≠ 正确。

## 3. 验证（独立基准，非模块内自洽）

小尺度 N=16, T=50, tau=1.0, u_ref = 0.9·u_T：

| 检验 | 方法 | 结果 |
|------|------|------|
| BPTT dJ/dtau vs 中心差分 (h=1e-5) | 独立前向 x2 | rel = **4.9e-9** |
| 终端种子 lambda^T | 复步 dJ/df^T (eps=1e-20) | max err 2.5e-7（FD 探针精度） |
| 单步 Jacobian 全 9x9 列 | 复步 (eps=1e-20) | max err 1e-16 |
| anti_stream 方向 | 与 forward roll 逐位对比 | 0.0 |
| b_action_t 数值 | 与 B^T 矩阵乘对比 | 2.2e-16 |
| OOTC vs BPTT（k=10/25/50/100/1/T） | 单元测试 | rel < 1e-15 |
| 尾段 T%k!=0（k=30/7/13/10） | OOTC vs BPTT（J + 梯度） | 18/18 通过 |

## 4. 全尺度验收（10^4 步 128^2）

脚本 `scripts/verify_ootc_acceptance.py`：4 个变体（BPTT 全存储 / OOTC
k=100 / OOTC k=1000 / 中心差分 h=1e-4）各在独立子进程运行，VmHWM
(ru_maxrss) 干净隔离；结果 JSON 零手抄。

- 集群作业 **cj1625**（stage_dir=本工作树，loong64，4 核）：
  `python3 scripts/verify_ootc_acceptance.py --workdir /tmp/ootc_acc_m17532
  --out /tmp/ootc_acc_m17532/results.json`
- 状态（2026-11-13 07:48 递交）：queued（递交时队列 374 个作业在前）。
- 判据：`results.json` 的 `gates.ootc_vs_bptt_pass`（<1e-5）与
  `gates.mem_gate_pass`（OOTC k=100 VmHWM < 8 GiB）；另附 BPTT vs FD
  交叉验证（验证 adjoint 推导本身）。

## 5. 局限

- 前向模型的 streaming 把 EX 滚在 y 轴、EY 滚在 x 轴（轴标签互换）；
  D2Q9 方向集在 x↔y 下对称，物理不变，adjoint 与前向一致即可，未改。
- 目标函数 J = 1/2||u_T - u_ref||^2（终端速度失配），u_ref = 0.9·u_T
  （与单元测试同约定）；tau 为唯一反演参数。
- 全尺度数字以 cj1625 的 results.json 为准（本文件发布时作业在排队）。
