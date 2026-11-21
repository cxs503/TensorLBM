# m23154 / anchor LBM-P0-KBC — KBC 与 Cumulant：实现核验 + 「文档认领↔实现注册表」一致性闸

**一句话结论**：KBC（entropic）与 Cumulant 在 TensorLBM 里**都已实现并注册**（`src/tensorlbm/entropic_kbc.py` 640 行、
`src/tensorlbm/cumulant.py` 1615 行，`advanced_collision_contract` 14/14 单元 AVAILABLE），m23039「KBC/Cumulant 未实现」的结论
**只对 legacy 的 2D `solver.py` 代码路径成立**，对整个仓库是错的（falsified）。同时本次审计发现并修掉了一个真实的文档漂移：
`docs/advanced_collision_capability_matrix.md` 仍把 CM/KBC 写成 `WITHHELD_*`、并声称派发器会抛 `CollisionKernelWithheldError`，
而注册表与实际派发早已 AVAILABLE。KBC 的**衰减率精度结论仍在集群排队**（cj4646–cj4653，fp64 Taylor-Green），
故 KBC 家族在证据账本里保持 `WITHHELD`，不做任何粉饰。

## ① 案例介绍

物理背景：碰撞算子是 LBM 的黏性来源。工艺上「支持 Cumulant/KBC」必须同时满足两件事——
(a) 有可执行内核并进入派发注册表；(b) 该内核的剪切黏性确实由 τ 决定（`nu = c_s²(τ−½)`），且在标准算例上
衰减率与解析值一致。判据采用 3D Taylor-Green 涡：`u=(U0 sin kx cos ky cos kz, −U0 cos kx sin ky cos kz, 0)`，
8 支平面波 `|κ|²=3k²` ⇒ `γ_E = 6νk²`，D3Q19/D3Q27 同为 `c_s²=1/3`，解析式一致。

参数表（预注册，见 `docs/m23154/PREREG.json`）：U0=0.05；τ 扫描 (0.6, 0.8, 1.0, 1.5)（tier A，N=32，float64）；
tier B N=48, τ=0.7（Re=24）；tier C 复跑 `benchmarks/pending/collision_kernels_3d_tg` 的 fp32 协议用于对账。
门限 G1 |γ_sim/γ_th−1|≤3%、G2 R²≥0.999、G3 |Δρ/ρ|≤1e-12、G4 min f≥−1e-14。

## ② 怎么做的

1. **先读代码再判**：定位 `entropic_kbc`（逐格熵条件 β 解 + 剪切投影按 BGK 因子 (1−1/τ) 松弛）与 `cumulant`
   （Geier 2015 分块；另含 legacy `collide_cumulant_d3q27` 与 `collide_cumulant_geier_d3q27` 两个条目）。
2. **碰撞级证据**：`tools/collision_claim_registry.py` 对注册表里每个 AVAILABLE 单元做一次 float64 碰撞冒烟——
   守恒（质量/动量）、剪切因子 Π_dev(post)/Π_dev(pre) 必须等于 1−1/τ；并把结果与解析式一起写进证据账本。
3. **衰减率证据**：`development/experiments/m23154_kbc_cumulant_tg.py`（单点，可复现）+ `development/experiments/m23154_sweep.py`（41 个 run 分片），
   递交集群（`stage_dir`=本工作树），结果落 `docs/m23154/runs/*.json`，由 `docs/m23154/build_artifacts.py` 汇总成表与图（零手抄）。
   同时 `tg_evidence()` 会吞入仓库内既有 benchmark 产物（先验证据），避免把「别的任务测过的」当作没测过。
4. **一致性闸**：`tools/collision_claim_registry.py check` 对 README/README.zh-CN/两份能力文档做声明抽取，
   与注册表 + 证据账本对撞：R1（声称 AVAILABLE 但证据不足）、R2（文档仍写 WITHHELD 而证据已 REAL＝陈旧的除名）、
   R3（注册单元必须可导入且守恒）、R4（账本覆盖）。任一 FAIL 即退出码 1，并写成 pytest 常驻。

## ③ 计算结果（程序化注入自 `docs/m23154/COLLISION_CLAIM_REGISTRY.json`）

注册表可执行单元：**14/14** 全部导入成功；单次碰撞最大相对守恒误差 质量 2.49e-14、动量 2.53e-14（门限 1e-12）；
剪切因子与 1−1/τ 的最大偏差 8.28e-10（门限 1e-8）——即**黏性由 τ 精确控制**，KBC 的熵条件只作用于高阶 ghost 分量。

| 家族 | 证据等级 | 最差 |γ_sim/γ_th−1| | 理由 |
|---|---|---|---|---|
| BGK | **REAL** | 0.3347% | conservation OK and Taylor-Green |err| <= 3.0% (0.335%) |
| CM | **REAL** | 0.4122% | conservation OK and Taylor-Green |err| <= 3.0% (0.412%) |
| CUMULANT | **REAL** | 0.4122% | conservation OK and Taylor-Green |err| <= 3.0% (0.412%) |
| KBC | **WITHHELD** | 38.0875% | Taylor-Green decay deviates from gamma_E = 6 nu k^2 by up to 38.09% (tolerance 3.0%), R^2 min 0.97880 |
| MRT | **REAL** | 0.4644% | conservation OK and Taylor-Green |err| <= 3.0% (0.464%) |
| RLBM | **REAL** | 0.4030% | conservation OK and Taylor-Green |err| <= 3.0% (0.403%) |
| TRT | **REAL** | 0.6725% | conservation OK and Taylor-Green |err| <= 3.0% (0.673%) |

注：CUMULANT 的 REAL 来自仓库既有 TG 产物（`benchmarks/pending/collision_kernels_3d_tg/` 的 cumulant D3Q27：
+0.1825/+0.3738/+0.4122%，见 `docs/CUMULANT_D3Q27_VERIFICATION.md`）。KBC 家族之所以是 WITHHELD：
同一份 fp32 产物里 `collide_kbc_d3q19` 的 err_E 链为 **+8.59 / −16.19 / −36.51%**（远超 3%），
本次 float64 复测（cj4646–cj4653）尚未出队，故**不下结论、不抬等级**。

## ④ 结果图

`docs/m23154/figures/` 由 `build_artifacts.py` 生成：fp64 Taylor-Green 能量衰减曲线与「实测 ν(τ) vs 解析 ν=(τ−½)/3」
在集群 run 取回后自动出现。目前尚无 run 文件落地（作业仍在队列），**因此本单没有可嵌入的衰减率图**——
这是排队状态，不是省略；证据账本图（下图）由 `tools/collision_claim_registry.py` 的冒烟数据直接生成。

## ⑤ 注意点 / 局限

* 「AVAILABLE」≠「精度已验证」。注册表只说内核可执行、守恒、黏性可控；衰减率精度是另一条证据链（G1）。
* `cumulant.collide_cumulant_d3q27`（legacy）在低 Kn 下与中心矩碰撞在 6 位有效数字上一致（其 O(Kn²) 非线性项不可分辨），
  要 Geier 形式请用 `collide_cumulant_geier_d3q27`；这一点写进了模块 docstring，本次未改代码。
* KBC D3Q19 的 fp32 FAIL 是否属实需 fp64 复测：fp32 熵条件二分在 tol=1e-8 下实际无法收敛到该判据，且噪声地板会污染
  晚期拟合窗；tier C 专门复跑同一协议用于归因（若 fp64 通过而 fp32 失败，则 FAIL 属测量面缺陷）。
* 本机（<8 核、<10 min）无法承担 KBC 熵解 28 次二分 × 上千步的开销，全部重计算走集群门。

## ⑥ 误差判定（逐项）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 注册内核质量/动量守恒 | 0（精确） | ≤ 2.49e-14 / 2.53e-14 | 是（≪1e-12 量级门限） |
| 剪切应力松弛因子 | 1−1/τ | 偏差 ≤ 8.28e-10 | 是 |
| Cumulant D3Q27 能量衰减率 | 6νk² | ≤ 0.4122%（先验产物） | 是 |
| BGK/MRT/TRT/RLBM/CM 衰减率 | 6νk² | ≤ 0.47%（先验产物） | 是 |
| **KBC D3Q19/D3Q27 衰减率** | 6νk² | **未判定**（fp32 先验 −16.19%/−36.51%；fp64 复测 cj4646–cj4653 排队中） | **未判定，不作「是」的声明** |

honest-FAIL 项：本单**未能**给出 KBC 的 fp64 衰减率判定，原因是集群队列（递交时前方 ~108 个作业）；交付物中所有
已被验证的结论都有 JSON/测试背书，未验证的部分明确标为未判定。
