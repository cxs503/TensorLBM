# 解析解验证欠账清偿：test_lbm_verification 四案例回流 CI（2026-11-03）

## 0. 结论（一句话）

`tests/test_lbm_verification.py` 的四个解析解案例（剪切波 / Couette / Poiseuille / 壁面摩擦）
此前以 **非严格 xfail** 形态长期躺在 `tests/ci_quarantine.txt` 里、从未在任何 runner 上被裁决过；
本次把它们改成设备无关、真断言、31 s 跑完的 pytest 用例并**移出隔离名单**，
四案例全部通过（最大相对误差 0.3378% < 1%），
同时补一条负控用例证明判据对该类缺陷确有鉴别力。

## 1. 为什么它们以前不可能通过

1. **硬件钉扎**：`def test_step_x(device="sdaa:0")` —— pytest 收集时按默认参数构造
   `torch.device("sdaa:0")`，CPU-only runner 直接报错（本机是 SDAA 机器，所以这个缺陷
   在本地看不出来，只在 CI 暴露）。
2. **返回值非法**：四个函数 `return passed`（布尔）。pytest 不接受测试函数返回非 `None`，
   即使数值通过也会被判红/告警。
3. **逐分量 Python 循环**：NoDynamics 与 Guo 体力都用 `for q in range(19)` 逐分量写入，
   在 10³–10⁴ 单元的小场上被 Python 循环和线程调度支配——实测不钉线程时三个案例
   115 s 跑不完（user 11m29s），钉 `torch.set_num_threads(1)` 后 79 s 全部跑完。

## 2. 改了什么（最小改动）

* 设备默认改为 CPU，`TENSORLBM_TEST_DEVICE=cuda:0/sdaa:0` 可覆盖（解析解与设备无关）。
* 四个测试改为 **断言**；测量逻辑抽成 `*_case()` 纯函数返回 dict，
  命令行 `python tests/test_lbm_verification.py` 仍打印全部数值并返回退出码。
* 逐分量循环 → 整张量布尔掩码 + 广播（数学逐点等价），并保留半路反弹与进口/出口复制的原顺序。
* 模块级 autouse fixture 把线程数钉到 1（用后恢复）。
* **发现并固化一个真实陷阱**：NoDynamics 必须回填**碰撞前**快照。重构时若误在 `collide()`
  之后取快照，固体格点每步带着碰撞结果再反弹（等效往壁面注入伪动量）：
  200 步后场量最大偏差 4e-3，Couette 剖面误差 0.0013% → 4.8%，Poiseuille → 17%。
  该陷阱由新增负控 `test_no_dynamics_restore_must_use_pre_collision_snapshot` 守住。
* 步数按收敛性实测收敛到 1000（见 §4），单文件运行时间 3000→1000 步/案例量级，
  全文件 31 s，可直接留在 PR 快车道（无需 `slow` 标记）。

## 3. 计算结果（由 `artifacts/analytic_verification_20261103/run_verification.py` 重算并写入 results.json）

| 案例 | 测量量 | 数值 | 解析值 | 相对误差 | <1% ? |
|---|---|---|---|---|---|
| A 剪切波衰减 | ν | 0.166662 | 0.166667 | 0.0027% | 是 |
| B Couette | max|u−u_exact|/U | — | — | 0.0013% | 是 |
| C Poiseuille | max|u−u_exact|/u_max | 0.012417 | 0.012375 | 0.3378% | 是 |
| D 壁面摩擦 | Cf | 0.634895 | 0.634921 | 0.0040% | 是 |

剪切波 200 步历史末点（`|u_y(x=0)|` vs `A_0·exp(-νk²t)`）相对误差 0.0005%。

## 4. 收敛性扫描（判据是否被步数掩盖）

| 步数 | Poiseuille 剖面 | Couette 剖面 | Cf 剪应力 |
|---|---|---|---|
| 100 | 19.5604% | 14.1795% | 43.6829% |
| 250 | 1.3387% | 1.5091% | 4.7024% |
| 500 | 0.3335% | 0.0362% | 0.1136% |
| 800 | 0.3378% | 0.0013% | 0.0041% |
| 1000 | 0.3378% | 0.0013% | 0.0040% |

* Couette 剖面 / Cf 在 800 步后不再变化（0.0013% / 0.004%），500 步已 <1%；
* Poiseuille 在 500/800/1000 步稳定在 0.3335–0.3378%——这是 **ny=12 下半路反弹 + Guo 离散的
  稳态空间离散误差**，不随步数下降，因此不是「跑得不够久」，而是分辨率受限。
  若要压到 <0.1% 需提高 ny（本任务遵循「小幅最小改动」未改网格，如实记录）。

  > **补记（2026-11-07，m6295）**：该开口已由网格加密裁决清偿——0.3378% 的主体是与 y 无关的
  > 均匀滑移（`offset ≈ G/4`），形状误差仅 5.7e-6，总量误差二阶收敛
  > （p = 2.03/2.06/1.87，h=6→22 误差 0.9534%→0.0725%）。结论：真离散滑移、非代码缺陷；
  > 「不随步数下降」不足以证明离散误差，需密网格证据。详见
  > `docs/poiseuille_resolution_order_20261107.md`，判据已锁进
  > `test_step_c_poiseuille_residual_is_second_order_slip`。

## 5. 注意点 / 局限

* Step D 的 Cf 定义是 `Cf = 2τ_w/(ρU²)` 且 `τ_w = νU/L`，因此 `Cf ≡ 2ν/(L·U)` 是**摩擦系数**，
  不是工程阻力系数 `C_D`（后者需按湿面积归一）。判据仅在该定义下成立，文档中已显式声明，
  避免被误当作阻力精度证据。
* 判据是**相对误差 < 1%**（比平台 15% 门槛严格得多），阈值集中在模块常量 `REL_TOL`。
* 四个案例只覆盖单相、静止等温、无外力/常体力 + 半路反弹的 D3Q19 BGK 核；
  多相、壁面函数、移动边界、LES 不在本验证面内。
* 本机 load15 > 100 且 nproc=128，所有测量均为「CPU + 单线程」；数值与线程数无关。

## 6. 证据指针

* 代码：`tests/test_lbm_verification.py`（5 个用例：4 解析解 + 1 负控）
* 隔离名单：`tests/ci_quarantine.txt` 删除该文件 4 行
* 工件：`artifacts/analytic_verification_20261103/{run_verification.py, results.json, fig1_verification.png, fig2_convergence.png}`
* 复核命令：`PYTHONPATH=src python3 -m pytest tests/test_lbm_verification.py -q`

## 7. 报告发布与一处平台缺陷留证（TensorPlatform 冻结，只报告不改码）

本单报告（含两张图）发布在 `tensor_reports` **id 5237**（mission 4765）；
此前同一负载的发布过程暴露一条可复现的入库缺陷，按 2026-11-02 控仓令仅留证不修复：

| 尝试 | 负载 | 结果 |
|---|---|---|
| 1 | markdown+2 图，97.6 KB | HTTP 500 `uq_reports_mission_md5` 重复键，且报错里的 mission_id 是 4747（不是请求体里的 4765） |
| 2 | 同上 + 尾部加 nonce | HTTP 200，但回写的是**别人**的报告（id 5231，TensorDEM Hertz 内容） |
| 3 | 同上，mission_id=4765 | HTTP 200，`mission_id` 记成 4765 但 title/markdown 仍是 Hertz 报告内容（id 5234，脏数据） |
| 4 | 去掉 figures（~30 KB） | HTTP 200，内容正确（id 5235） |
| 5 | 只带 fig1（42.9 KB） | HTTP 200，内容正确 + 1 图（id 5236） |
| 6 | fig1 + 缩到 36 KB 的 fig2（91.1 KB） | HTTP 200，内容正确 + 2 图（id 5237，正式报告） |

结论：`POST /api/v1/tensor/reports` 在大负载下会**静默回显他人报告并落成脏行**，
比已知的「>250 KB 才失败」阈值低得多（97.6 KB 已复现），建议平台侧按 mission_id 校验
写入内容一致性并对负载设硬上限（本次仅留证）。
