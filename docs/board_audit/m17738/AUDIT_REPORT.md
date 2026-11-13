## ① 案例介绍（审计对象）
对象 = m17650（XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT，TensorLBM，succeeded）：把上游 haimingz/vivsim `vivsim/multigrid.py` @ fb84b12（MIT，oss_inventory id=11）的 2:1 buffer-layer 加密算子 vendor 进仓并逐字执行（numpy jax.numpy shim，sha256 锁定），移植 `get_omega`/`fine_to_coarse`/`coarse_to_fine` 为 `src/tensorlbm/block_refinement_2to1.py`，并在主 AMR 腔体求解器 `amr_cavity2d.py` 增加 `interface_scheme∈{fh, vivsim_copy}` 双臂对标。物理算例：2-D D2Q9 lid-driven cavity，coarse 64² + 上下 16 粗格加密带，Re=100，U_lid=0.05，12000 粗步。本单为交差核验（非重做），基准裁判 = tensor_benchmark_cases id=61。

## ② 怎么做的（审计流程）
1. 取证：读 m17650 工作树 /root/tp-agent/work/m17650（commit daafea2a）、平台报告 id=16617、tensor_benchmark_cases id=61、mission result。
2. 三件套核验：run_artifacts 文件存在性（JSON 131015 B + 3 PNG + REPORT.md）；benchmark ref vs got 逐项对照 JSON 工件（零手抄，全部程序化读取）；repro_cmd 实跑。
3. 独立复跑：`pytest tests/test_xval_buffer_layer_2to1.py -q` → **48 passed (13.64s)**；`PYTHONPATH=src python3 benchmarks/xval_buffer_layer_2to1/run_xval.py --quick` → **exit 0, 26.9s**，verdict 与交付工件一致（cross_arm_pass=true, identity_pass=true, 两 Ghia 臂 false）。
4. 溯源核验：vendor 文件 sha256 与 PROVENANCE.json 双 pin 全部匹配（upstream d104683c…、shimmed 7985f6c5…）。
5. 落表：tensor_maturity_verdicts id=634（L4，verdict_by=board-audit），evidence 全键真实指针。

## ③ 计算结果（审计复核值，程序化自工件 JSON 注入）
| 量 | 报告值 | 工件值 | 审计判定 |
|---|---|---|---|
| 跨臂中线 max\|Δu\|/U_lid（64²） | 0.951% | 0.9505216323304919% | 一致，<15% 门 PASS |
| 跨臂中线 max\|Δv\|/U_lid（64²） | 1.304% | 1.3043012050911784% | 一致 |
| 48² 敏感性跨臂 | 1.135% | 1.1351944878697402% | 一致 |
| get_omega 恒等最大相对误差 | 2.89e-16 | ≤2.89e-16（逐点） | 一致 |
| 换面算子对拍 | 0.0e+00 | 0.0（12 组全 0） | bit-identical |
| 非平衡缩放比 实测/预测 | 1.72247/1.72254 | 1.7224660557993892/1.722543352601156 | 一致 |
| 时间收敛 9000→12000 | 0.025%/0.059% | 0.024579…/0.059198…% | 一致 |
| 无回流质量漂移 | 4.85e-05/4.81e-05 | 4.845857…e-05/4.813075…e-05 | 一致 |
| FH 臂 vs Ghia | 16.973% | 16.972945160388946% | 一致，honest FAIL |
| OSS 臂 vs Ghia | 16.377% | 16.37739561271667% | 一致，honest FAIL |
| 全量运行耗时 | 343.4s | elapsed_s=343.4100065231323 | 一致 |

## ④ 结果图
三张图均为 m17650 交付的 matplotlib 图（审计未重绘，原样内嵌）：跨臂剖面 vs Ghia、跨臂门限、回流失败模式。

## ⑤ 注意点
- **生态仓滞后**：agent/m17650（daafea2a）审计时尚未并入 TensorLBM main（push-merge 队列在途），/root/tensor-ecosystem 检出无新代码；maturity_gate 的 integration_callers grep 代理在合并前可能判 L3 缺项 → 若被降级为 L2 属机制时差非交付缺陷，合并后应回升 L4。审计工作树已留字节一致快照（docs/board_audit/m17738/refs/，sha256 见 AUDIT_REPORT.md 附录）。
- 上游 multigrid 为无算例无测试原型；buffer 层几何约定未文档化，只移植松弛律与换面索引代数（交付方已披露，审计认可该边界）。
- sha256↔commit 绑定依赖 oss_inventory.cache_commit=fb84b12（上游 .git 对象库损坏，本地不可再验）——交付方已如实披露。
- 无回流臂漂移仅 5e-5：回流在本算例非 load-bearing，交付方未粉饰，审计认可。

## ⑥ 误差判定（逐项回答是否 <3%）
- 基准裁判（tensor_benchmark_cases id=61，唯一裁判）：metric=跨臂中线 max|Δu|/U_lid，tolerance <15% → **got 0.9505% → <3% 是，<15% 是，PASS**。
- 松弛律恒等 2.89e-16 → <3% 是。
- 换面算子对拍 0 → <3% 是。
- FH 臂 vs Ghia 16.973% → <3% **否**（>15% 亦否，honest FAIL，非本单声明判据）。
- OSS 臂 vs Ghia 16.377% → <3% **否**（同上）。
- 结论：声明判据通过，二级锚如实 FAIL；**判级 L4**（完整算例+误差分析+可复现+真集成），不虚标 L5（无独立科研工程增量）。

| `artifacts/xval_buffer_layer_2to1/xval_buffer_layer_2to1.json` | `db6a38415ab7ad927cc957d915de8926a1ebfa63efec6941c1c9891ba13e2c7a` |
| `benchmarks/xval_buffer_layer_2to1/REPORT.md` | `55de8566148ed6704660709dab58a7a1dc2f9769ee9141fc5a9ba4c09a4c15f7` |
| `benchmarks/xval_buffer_layer_2to1/oss_ref/PROVENANCE.json` | `9e75fb620640398759c238e035140128a5f6eb14c4c5e2aca06ba975dd19ce8c` |
| `benchmarks/xval_buffer_layer_2to1/oss_ref/__init__.py` | `15513832fcf0c1916cf9583f77269f714f8732863655f4eaaf127175307e9ab9` |
| `benchmarks/xval_buffer_layer_2to1/oss_ref/lattice_dirs.py` | `728680e945c90e8dce25811063b315cc374880ec1804402ceb8c8a9b8626355d` |
| `benchmarks/xval_buffer_layer_2to1/oss_ref/np_jax_shim.py` | `4fc6c1df6026647f170c18009abb5d045156e264350cff7dbd70ce6cb8056018` |
| `benchmarks/xval_buffer_layer_2to1/oss_ref/vivsim_multigrid.py` | `7985f6c5c498039e12c79a7bd9ead348f1a094729758f0a57405dc89bbd372d5` |
| `benchmarks/xval_buffer_layer_2to1/oss_ref/vivsim_multigrid_upstream.py` | `d104683cfad3c3114f74e42aebf2678cec1de4d63917cb1d5c13c97a72af6515` |
| `benchmarks/xval_buffer_layer_2to1/run_xval.py` | `f6deb7eb0fbfdf72088cd96f69f5214d384efd025906b161a0211cc48ecafc14` |
| `figures/xval_buffer_layer_2to1/cross_arm_gate.png` | `b6afe020c41cc6af05fc8ce55b201d5304141e1e8998ba0daf62c72f086417c0` |
| `figures/xval_buffer_layer_2to1/profiles_vs_ghia.png` | `6dd948e58c4d964bd9da6a462e4f21145688b6d24c5b644fee192cb69ef7170d` |
| `figures/xval_buffer_layer_2to1/reflux_failure_mode.png` | `db95d7df00195e942b012386d32fcbd40a5e8c39c94f20d5df3b29383dd5d2d9` |
| `src/tensorlbm/amr_cavity2d.py` | `3247e8991223b7baf1cd849a03298435ffa2adac6d9f2331ae8636db38e720b0` |
| `src/tensorlbm/block_refinement_2to1.py` | `1dd6f329e4e6e4517a5f5240dc5c0f9353aded08db4002acba16103ceeee0d22` |
| `tests/test_xval_buffer_layer_2to1.py` | `581274381da5c9c5f743302f91202b887ce536de9b88826e0ea452077366be97` |
