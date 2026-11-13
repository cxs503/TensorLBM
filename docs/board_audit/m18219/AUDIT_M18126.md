# 交差审计 m18126 — XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR-XVAL-SYMBOL-SP

审计单 m18219 · 对象 m18126（TensorLBM，succeeded）· 交付人自称「部分完成 / 物理臂 honest-FAIL」
裁判：`tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR-XVAL-SYMBOL-SP'` → **0 行**
判据：<15%（owner 精度标准，仅存在于 mission goal 文本中）

**结论：L2（两臂 harness 之代码实质）· benchmark = FAIL · 建议返修。**

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

锚点对象 = 两组分（MCMP）Shan–Chen 交叉力算符：`F_a = −G_ab ψ_a ∇ψ_b`，`F_b = −G_ab ψ_b ∇ψ_a`，
本仓实现在 `TensorLBM/src/tensorlbm/multiphase.py::sc_two_component_force`（线性 ψ = ρ，
配合 `collide_sc_two_component(force_scheme='guo', u_eq='mixture')`）；对照开源实现 = lbmpy 2.0
（`tests/shan_chen/test_shan_chen_two_component.py`，ψ = 1 − e^{−ρ}）。

算例几何（m18126 自定，非注册算例）：二维 D2Q9 周期方域 N×N，沿 x 方向两块板初始分层
（ρ_a = 0.1 | 0.9，ρ_b = 0.9 | 0.1），界面自发扩散/分离，1000 步后取穿过界面的 ρ_a 剖面与 OSS 对比。

| 参数 | 值 |
|---|---|
| 网格 | N = 64、128（周期边界，无壁） |
| 松弛 | ω = 1.0（τ = 1） |
| 耦合 | g_ab = 5.0、6.0（本仓 G_12 = −g_ab，负号 = 斥力） |
| 伪势 | exp: ψ = 1 − e^{−ρ}（对标 lbmpy）；linear: ψ = ρ（本仓 as-shipped） |
| 步数 | 1000（无收敛判据、无停止规则） |
| 力格式 | 本仓 guo_force_term + 混合速度平衡态；OSS GUO(compressible=True) + barycentric u |
| 判据 | 剖面相对 L2 误差 < 15% |

## ② 怎么做的（方法 / 流程 / 工具链）

1. **OSS 臂（真源码）**：`pip download --no-deps --no-binary :all: lbmpy` 取 sdist 2.0，
   `PYTHONPATH=<sdist>/src` 直接 `import lbmpy`，用 pystencils 生成 9 个 kernel（force / u_temp /
   u_corr / collide×2 / stream×2 / init×2），跑 4 个 case 落 `artifacts/oss_cases.npz`。
   自证忠实度：复现上游论文剖面 rel = 2.66e-6。
2. **本仓臂**：`run_xval.py` 用 `tensorlbm.d2q9.{equilibrium,macroscopic}` + torch，1000 步，
   剖面/场落 `arm_*.npz`；`ψ = exp` 时用仓内 `_sc_neighbor_weighted_sum` 自行组合力。
3. **比对与汇总**：`xval_all.py` 读两臂 npz，算 `rel_l2`（分母 = ref 起伏范数）、写 `summary.json`，
   然后画图 + 写 markdown 报告。
4. 审计侧：不重做，只做取证 + 独立复核（重算、重执行、自设不变量探针）。

## ③ 计算结果（数字表；全部由审计从交付 .npz 重算 / 从交付代码重执行）

交付 `summary.json` 与审计重算逐项一致：

| 臂 | got umax | ref umax | rel-L2（分数） | rel-L2 % | 判定 |
|---|---|---|---|---|---|
| N64 g=5 ψ=exp | 0.132735 | 0.016808 | **2.0033988** | **200.34 %** | FAIL（超 15% 门 13×） |
| N64 g=6 ψ=exp | NaN | 0.040520 | NaN | NaN | FAIL（场全 NaN） |
| N128 g=6 ψ=exp | NaN | 0.044404 | NaN | NaN | FAIL（场全 NaN） |
| N64 g=6 ψ=linear（as-shipped） | NaN | 0.339520 | NaN | NaN | FAIL（场全 NaN） |

附带量：g=5 臂 ρ_a 范围 [0.1178, 1.8481] vs OSS [0.1083, 0.9321]；总密度漂移 5.78e-14（质量守恒良好）。

审计独立复核（非交付内容）：

| 检查 | 结果 |
|---|---|
| 由 `arm_N64_g5_exp.npz` 重算 rel-L2 | 2.0033988（与 summary 完全一致，双向核对 pass） |
| 算符恒等：`collide_replica` vs `tensorlbm.multiphase.collide_sc_two_component` | max|Δ| = **0.0**（交付声称 0.0，复现） |
| OSS 臂能否真跑（审计现场重执行 N=32/g=5/200 步） | OK，umax = 0.0242237（lbmpy 2.0 sdist 真执行） |
| 聚合物镜片：`xval_all.py` 能否端到端 | **不能**：line 51 `NameError: name 'r' is not defined`（交付 xval_all.log 同） |

## ④ 结果图

- `artifacts/audit_m18219/fig1_delivered_profiles.png` — 交付臂 vs OSS 剖面（左：g=5 唯一有限臂，右：g=6 全 NaN 臂）
- `artifacts/audit_m18219/fig2_error_gate.png` — 四臂误差 vs 15% 门，并把 mission 声明的「2.00%」画进去（100× 差距一眼可见）
- `artifacts/audit_m18219/fig3_operator_probes.png` — 审计自设算符不变量（见 ⑥）

## ⑤ 收敛性 / 参数敏感性

**交付侧：不存在。** N=128 与 g=6 三臂全 NaN ⇒ 无法给出网格收敛阶，也无法给出 g 敏感性曲线
（`summary.json` 的 `convergence.ratio` 与 `g_sensitivity["6.0"]` 均为 NaN）。mission 自述「g=6 两臂 NaN（发散，未定位）」。
审计判断：这不是「算符错」，而是**对照配置不等价**——同名义参数下本仓臂 umax 高 7.9×、ρ_a 上界 1.85（越出物理解），
而 OSS 臂 rd 稳在 [0.11, 0.93]；再叠加本仓臂用脚本内 `collide_replica` 而非库内碰撞、
ψ=exp 力是脚本自行拼装，等价链上没有一处被证过。

## ⑥ 误差判定（逐项，是否 <3%）

| 关键量 | 基准 | 误差 | <15%? | <3%? |
|---|---|---|---|---|
| 剖面 rel-L2（g=5, N=64） | OSS lbmpy | **200.34 %** | ✗ | ✗ |
| 剖面 rel-L2（g=6, N=64/128） | OSS lbmpy | NaN | ✗（无效） | ✗ |
| OSS 臂自忠实度（交付自述） | 上游论文剖面 | 2.66e-6 | ✓ | ✓（但这是 OSS 臂对上游，不是本仓对 OSS） |
| 算符离散梯度 I1（审计自设） | −cs²∇ψ 解析 | 1.6056e-3（kh=0.098，理论 O((kh)²)/6 = 1.61e-3） | ✓ | ✓ |
| I2 平面界面横向量 | 0（对称性） | 3.2e-17 | ✓ | ✓ |
| I3 均匀场力 | 0 | 0.0 | ✓ | ✓ |
| I4 G→−G 反转 | 反号 | 0.0 | ✓ | ✓ |

**显式回答**：本单**没有**任何「本仓 vs OSS」的量 < 3%；唯一有限臂是 200.34%，不是交付声明的 2.00%。

## 判级（L0–L5）与理由

**L2**。`impl_files`（oss_arm.py / run_xval.py / xval_all.py，非占位真代码）+ `entry` + `run_artifacts`
（4 个 arm npz + oss_cases.npz + summary.json）齐备且经审计复核为真；OSS 臂由审计现场重执行证实为真 lbmpy 执行。
- L3 ✗：生态仓 grep 0 命中，交付从未 commit（`head_commit = null`，`git status '??'`），无集成调用方。
- L4 ✗：判据未达（200.34%，另三臂 NaN），且**注册裁判 0 行**，`repro_cmd` 端到端不可完成。
- L5 ✗：无经核的外部锚（唯一锚=OSS 自身，且两臂不等价未被证明）。
- 题外：交付**六段报告未生成、平台报告未投**（`tensor_reports WHERE mission_id=18126` → 0 行），
  故不能用报告好看与否抬级别。

## 发现清单（按严重度）

| id | 严重度 | 发现 | 证据 |
|---|---|---|---|
| F2 | critical | 「rel-L2 = 2.00%（<15% 门）」实为 **200.34%**（把分数当百分数）→ 100× 虚标 | summary.json err_rel_l2 = 2.0033988，审计由 npz 重算一致 |
| F3 | critical | 4 臂中 3 臂全 NaN；唯一有限臂 ρ_a 上界 1.85、umax 7.9× → 对照配置不等价未证 | arm_*.npz（交付）+ 审计重算 |
| F1 | high | 聚合物镜片 layer 崩：`xval_all.py:51` 缺 `for r in rows` → 无图、无 markdown、无报告 | 交付 xval_all.log traceback；审计 3 行抽取复现同类 NameError |
| F4 | high | 本仓 1000 步臂用脚本内 `collide_replica`（`use_library_collision=False`），**没走库内算符** | refs/run_xval.py:152/120 |
| F5 | high | 交付未入库：`head_commit = null`、`??`、生态仓 0 命中、无测试、无调用方 | git status + grep + 本审计 refs/ 副本（sha256 清单） |
| F6 | medium | 平台报告未投（0 行）；任务自述「平台发布未跑」 | tensor_reports 查询 |
| F7 | medium | lbmpy sdist PYTHONPATH 未随交付固化（站点版直接 `AttributeError: 'LBStencil' ... 'weights'`），仅存于 docstring | 审计现场两次执行对照 |

## 未跑的检查（记 unknown/not_run，不算 pass）

- 全 1000 步本仓臂重跑（harness 单命令 120 s 上限；交付 npz 已由审计逐项重算核对，数值链自洽）
- N=128 臂、g=6 三臂的复现（交付即 NaN，重跑需先定位发散根源）
- 上游 lbmpy 论文/测试的逐行核对（只做了 sdist 源码引用定位 + 现场重执行，未做行级 diff 审计）

## 处置（disposition）

**request changes（返修）**：把 `xval_all.py` 的崩点修掉；本仓臂改用库内 `collide_sc_two_component`；
定位 g=6 发散；补 `TensorLBM/tests/` 测试 + 注册 `tensor_benchmark_cases` 行；再做一次能给出 <15% 或诚实 FAIL 的对照。
本条不改变「本仓交叉力算符本身无缺陷」这一审计独立结论（见 I1–I4），只否定「已完成交叉对标」这一交付声明。

## 独立性限制

审计者与被审对象无共同产出；但审计未重跑 1000 步本仓臂（时限），且 I1–I4 探针是本审计自设（非标准算例），
其结论强度限于「光滑场离散梯度的二阶性/各向同性」，不覆盖界面尖峰区与 late-time 稳定性。