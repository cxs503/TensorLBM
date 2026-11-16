# CAP-REG-TLBM：TensorLBM 能力注册表试点（m23186）

## ① 案例介绍
TensorLBM 把多年积累的能力写在散文（README / docs）里，无法机械校验：算子被 mock、被改名、或从未接线进求解器，文档仍宣称可用。本单把家底变成**代码自注册**的可校验清单，并向平台 8010 的 /api/v1/tensor/capabilities（含 /summary、/callable、/sync）供数。物理侧锚点样本：D3Q19 平面 Poiseuille（Guo 体力、半路反弹壁面、τ=1.0、u_max=0.05），给四个碰撞族一条共同解析基准轴。

## ② 怎么做的
src/tensorlbm/registry.py 提供 @capability(...) 装饰器（id/kind/maturity/module_path/entry/params/wired/tested/verified/test_path/evidence/endpoint），import 时自注册进 REGISTRY 单例（微服务式，如 FastAPI 路由）。dump CLI 双模式：import 模式（需环境）与 --static AST 扫描（**不 import 包、免 torch**，用「假 torch 包 raise ImportError」的单测证明）。静态模式把 module_path / entry / test_path 三个指针逐一对工作树求解，指不到即 error。纪律：mature ⇒ wired ∧ verified（与平台 422 门同构，本地注册即拦）；verified ⇒ grade∈ANALYTIC|XVALUE 且带 max_error_pct；planned ⇒ 必须有 evidence.feature_ledger；endpoint 只给 wired 且限 cli:/route:/api:。审计脚本列出接线/测试/证据/文档四类虚标并逐条给处置（改码/改文/在案）。误差带由 scripts/m23186_collision_family_verification.py 复现，含 MRT 反 mock 探针。

## ③ 计算结果（程序化注入）
- 登记能力 **25** 项：mature **4** / developing **21** / planned **0**；wired 14、verified 4。
- 静态扫描 error **0**；静态与 import 模式逐字段 **完全一致**（--parity ok=True）。
- 平台同步回执：upserted=25、tombstoned=1、source_commit=21ff1b7d1e7e。
- 审计发现 **4** 条、**open=0**（4 条全裁定：2 条在案=扫描器按行误判，README 生产级措辞改文，2 条本单自查改码）。
- 碰撞族 Poiseuille 误差（协议 {'nx': 24, 'nz': 2, 'ny': 10, 'tau': 1.0, 'u_max': 0.05, 'steps': 200}）：

| capability | max_error | 判定 |
|---|---|---|
| `lbm.collision.bgk.d3q19` | 0.4122 % | <3% PASS |
| `lbm.collision.mrt.d3q19` | 0.4124 % | <3% PASS |
| `lbm.collision.rlbm.d3q19` | 0.4122 % | <3% PASS |
| `lbm.collision.trt.d3q19` | 0.5709 % | <3% PASS |

反 mock 探针：MRT vs BGK 相对差 **0.1542**、s_e 灵敏度 **0.3318**（BGK 别名应为 0）⇒ **not an alias of BGK**。

## ④ 结果图
图1：能力按成熟度/学科族分布。图2：四族相对解析 Poiseuille 剖面的最大误差（对数轴，红虚线 15% 验收线）。无场量云图的理由：交付物是注册表与清单本身，物理场由被登记的求解器在同算例产生，误差曲线即其可判定量化证据。

## ⑤ 注意点
1. python -m tensorlbm.registry 必然先执行包 __init__（import torch）；torch-free 主机必须按**文件路径**调用，99 上 capability_sync 守护按此契约实现。
2. 多 worktree 机器上 editable install 会把 tensorlbm 指向别的检出；import_registry() 前置本树 src/ 并在 imports.shadowed_package 记录被遮罩路径。
3. wired 的机械判据是「src/ 内存在非测试且非契约模块引用该 entry」：KBC 与 CM 因此 wired=False（可 import 但无 stepper 分派）。
4. TensorLBM 无 console script，endpoint 只能用 api:<可导入调用>；cli: 形态暂无真对象，是 TensorPhysics 可调用清单的现存缺口。
5. 误差带取粗网格（nx=24/ny=10/nz=2/200 步，受单机 10 分钟门限）；细网格（ny=12/1000 步）为 0.338/0.0012/0.338/0.343 %，两口径都写进注册表证据。
6. 仅登记 25 项（每域至少一项 + 碰撞全族），非全仓穷举。

## ⑥ 误差判定
关键量 = Poiseuille 剖面相对误差，基准 = 解析抛物线解。4 个 mature 能力**均 < 3%**（最大 0.5709%，3 个约 0.41%，TRT 0.57%）。逐项回答「是否 <3%」：**是**。其余 21 项标 developing（无外部/解析误差带），不参与精度判定。

