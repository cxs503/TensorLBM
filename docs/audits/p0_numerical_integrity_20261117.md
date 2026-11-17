# TensorLBM P0 数值完整性修复验收（2026-11-17）

基线：`2d21ee08d95600d61d3b77f5e0e9b7b088d14f6d`。本轮在 CPU 实际执行，联合回归 **287 passed / 1 skipped / 0 failed**。这修复了审计中复现的 6 个失败项，但不代表全部多相模型或工程算例通过验证。

原始指标、配置、环境、代码文件 SHA-256 见 [JSON 证据](p0_numerical_integrity_20261117.json)；测试摘要见 [运行日志](p0_numerical_integrity_20261117_pytest.txt)。源码哈希绑定测试内容，避免用生成报告所在提交的 SHA 冒充运行时源码。

## 修复和直接证据

| 项目 | 改动 | 验证 |
|---|---|---|
| D3Q27 Guo 外力 | 平衡态和源项统一使用半步力速度，去除 `tau*F/rho` 与显式源项重复计力；源项使用工作精度权重 | 32 个 D3Q27 组合覆盖 FP32/64、4 个 tau、无 SGS/Smagorinsky/WALE/Vreman |
| 三种格子的力矩约定 | D3Q19 源项使用相同物理速度；D2Q9/D3Q19 修复双精度源项权重 | D2Q9/19/27 外力一、二阶矩；不等 tau 下内部力总动量守恒，24 项通过 |
| SC 单组分相分离 | `G>0` 才是本库 `x-c` 邻居约定的吸引；验证器与默认 `G=+5` 一致；参考计算默认 FP64，仍可显式选择 FP32 | 2D：64²/1200 步，密度比 12.4364，质量漂移 6.87e-14；3D：24³/500 步，12.2390，2.80e-14。原守恒门槛未放宽 |
| 液滴槽体边界 | 迁移前使用流体链接反弹，不再周期迁移后补写；化学势及梯度使用同一无通量策略；`rho_top` 真正控制 Zou/He 密度 | 非均匀分布守恒、上下标记脉冲不穿越、两种目标密度和零切向速度检查通过 |
| 相场静水平衡 | 相场对流使用物理半步力速度；FP64 初始化及自由能权重不再复用已舍入 FP32 表 | 单相重力静水柱 32×12/1000 步：密度最大相对误差 2.47e-13、物理速度 5.57e-16、相质量漂移 5.98e-14 |
| 液滴健康门禁 | 每步检查有限性、正密度、速度、相场范围和相质量；失败先写严格 JSON/CSV，再以非零状态退出 | 120×90/120 步：最大归一化相质量漂移 3.60e-7；5 种失败注入均被拒绝并保留记录 |
| 能力声明 | CM/KBC 内核组件改为已有实现；完整组合仍 withheld，KBC 衰减率验证仍 withheld | 与实际碰撞注册表交叉核对；未扩大已批准完整组合 |
| 证据计数 | 分开记录求解方法、来源及声明范围；合成数据拒绝进入物理验收及验证图生成；FV 与相关式对照不计为 LBM benchmark PASS | 数值判定仍为 8 passed / 3 failed / 1 pending；按方法和范围可计的 LBM benchmark PASS 为 5。这里仅重判冻结记录，不是重跑 12 个求解器 |
| 台账 | 刷新目录清单及计数；修正圆柱 `err_decreased` 错误；明确目录数量不是科学通过数 | hard 12→0，soft 17 保留；参考锚点 17/17；独立判定审计 0 阻塞 |

## 兼容性与适用范围

- `examples.droplet_impact._apply_tank_boundaries` 保留导入名，但现在是 `stream_tank` 的别名：**输入必须是碰撞后、迁移前的分布**，不能再先调用周期 `stream`。
- 液滴默认顶边界改为封闭反弹。`--top-bc pressure --rho-top ...` 只为动量分布设置定密度边界，相场仍不可穿透；这不是已验证的两相开放出口。
- 相质量只统计流体单元，归一化分母为初始 `sum(abs(phi))`，避免正负相场抵消导致指标奇异。与旧全域有符号相场归一化的 36.6% 数字不能直接作为同口径倍数比较。
- `--density-ratio` 只控制 Boussinesq 重力系数，不切换 cAC，不证明真实水气密度比。
- 自由能相场的受力对流结果有意改变；这是静水平衡修复。周期边界仍是求解器 API 默认值。
- SC 密度比超过 2 是分相形态门槛；没有据此宣称 Maxwell 共存曲线、表面张力或 EOS 精度已验证。
- 历史湍流报告缺乏可追溯原始证据，已标为来源待核验，未将其直接断言为合成数据。

## 复现与质量门

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src
python -m pytest -q tests/test_constants.py tests/test_d3q27_guo_moments.py tests/test_multiphase_force_moments.py tests/test_multiphase.py tests/test_multiphase_benchmarks.py tests/test_multiphase_g4_default_coupling.py tests/test_multiphase_physical_benchmarks.py tests/test_advanced_collision_contract.py tests/test_general_capability_matrix.py tests/test_gh408_droplet_impact.py tests/test_droplet_impact_initialization.py tests/test_mcmp_mixture.py tests/test_mcmp_mixture_3d.py tests/test_phasefield_common_operators.py tests/test_validation_evidence_admission.py tests/test_validation_matrix.py tests/test_validation_matrix_gated_display.py tests/test_benchmarks_status_ledger.py tests/test_autograd.py
python examples/droplet_impact.py --fast --output-root /tmp/tensorlbm-p0 --overwrite
python benchmarks/status_ledger.py --check
python benchmarks/validation_matrix/runner.py audit
python benchmarks/validation_matrix/independent_audit.py
python scripts/check_platform_docs_consistency.py
python tools/lint_ratchet.py --check  # ruff==0.15.0，与既有 baseline 一致
```

本地均通过。11 个相邻科学测试文件仅做格式整理，逐文件 AST 前后相同，用于清偿已存在的格式债务；没有提高 lint baseline。唯一 skip 来自旧的空参数化测试；另有 21 个含 Triton 的仓库测试文件因环境不具备依赖而未收集，不计为通过。

新增 `numerical-integrity.yml` 使用独立 CPU runner，执行实算、失败负例、台账和参考审计，上传测试 XML、源码 SHA、环境与液滴结果。工作流已写入不等于远端已成功执行；远端状态以对应 PR 检查为准。

## 继续保留的验收缺口

本轮没有 GPU/多卡结果，也未完成两相静滴/毛细波的系统精度识别、黏度全组合鉴定、Maxwell/EOS 对照、网格与时间步敏感性、入水飞溅/载荷参考比较及船海冰耦合。17 个 soft 台账问题、矩阵中的失败/待定项保持可见；未把 quarantine、缺测或参考计算提升为通过。下一阶段应在这些基线之上推进三条标杆算例，再进入工程耦合。

方法参考：[lbmpy 外力模型说明](https://pycodegen.walberla.net/docs/lbmpy/2.1.1/sphinx/forcemodels.html)、[Zou–He 边界原论文](https://arxiv.org/abs/comp-gas/9508001)。
