# M24569 — TLBM-HOTPATCH-ABSORB 独立验证与闭环（adversarial review）

## ① 案例介绍

锚点 `TLBM-HOTPATCH-ABSORB`（TLBM 镜像未入库热补丁吸收：torch.Tensor 防 float
强转保护可微路径）在本单开工前状态为 `in_progress`。本单为该锚点的**独立复核单**：
不重复实现，而是以对抗性审查方式验证已吸收补丁的忠实性、有效性与覆盖完整性，
补齐验证欠账并闭环。

被验证对象：

| 项 | 内容 |
|---|---|
| 补丁语义 | `input_contract._as_float` 对 `torch.Tensor`（含 0-dim）返回 `None` 而非 `float()` 强转，防止静默剥离 autograd 图 |
| 吸收载体 | Merge #2662（m23777），commit `fc2029663` |
| 镜像热补丁 | `/root/tensor-ecosystem/TensorLBM` 工作树未提交 diff（2 文件，+9/-1） |
| 验收规则 | 补丁以 PR 回流，禁改镜像树 |

## ② 怎么做的（方法）

1. **吸收忠实性核对**：镜像工作树未提交 diff 与 main 已吸收版本逐字节
   `diff` 比对（只读，未动镜像树一个字节）。
2. **回归证据复现**：在本工作树（`agent/m24569`，基线 `9b10af0e7`）运行
   `tests/test_input_contract_guards.py` + `tests/test_diff_lid_cavity.py`
   （`-m "not slow"`），核对回归测试 `test_input_contract_keeps_0dim_tensor_in_graph`
   在位且通过。
3. **残余强转扫描**：grep 可微模块（`diff_lid_cavity.py`、`autograd_path.py`）
   中绕过 input_contract 的裸 `float()` 强转点，逐处判定是否在可微路径上。

## ③ 计算结果（核对表）

| 检查点 | 结果 |
|---|---|
| `input_contract.py` 镜像 vs main | **逐字节一致**（仅镜像侧缺文件尾换行符，语义零差异） |
| `tests/test_diff_lid_cavity.py` 镜像 vs main | main **多出**回归测试 `test_input_contract_keeps_0dim_tensor_in_graph`（镜像热补丁无测试）→ main 是超集，吸收完整且更优 |
| 镜像未提交 diff 性质 | **陈旧冗余**：其内容已全部经 #2662 入 main，镜像 HEAD（`927a650f3`=#2660）落后一拍未 ff |
| 回归测试（tensor tau 图保持） | **PASS**（含于下方 51 passed） |
| 本轮测试 | `test_input_contract_guards.py` + `test_diff_lid_cavity.py`（not slow）：**51 passed, 1 deselected, 18.59s** |
| 残余强转扫描 `diff_lid_cavity.py:297` | `tau if isinstance(tau, torch.Tensor) else torch.as_tensor(float(tau))` — 有显式 tensor 分支，**受保护** |
| 残余强转扫描 `diff_lid_cavity.py:479-481,885-887` | `float(tau_init)` 等位于反演**结果报告字典**，发生在优化结束后，仅影响展示不影响图，**可接受** |
| 残余强转扫描 `autograd_path.py:500` | 显式 `value.item()` tensor 分支，**受保护** |

## ④ 结果图

本单为代码吸收验证/对抗性审查任务，无场量数据与基准曲线，故无图。
证据形态为 diff 比对 + pytest 计数（见③表），已在平台报告 markdown 中原样引用。

## ⑤ 注意点

- **镜像树未动**：按验收规则禁改镜像树。镜像工作树仍携带陈旧未提交补丁，
  需镜像侧 ff 到 ≥#2662 后自然清洁（本单无权处理，留待镜像管理员）。
- `_as_float` 每次调用执行 `try: import torch`（已导入时为 sys.modules 查找，
  亚微秒级）；如未来 profiling 证明进入瓶颈可改模块级惰性缓存，本轮**不做**
  （改动最小纪律，无实测瓶颈不动刀）。
- slow 冒烟 `test_invert_recovers_tau_smoke` 本轮**未重跑**（deselected；
  m23777 报告记录 61.8s 通过），本轮结论不依赖它。

## ⑥ 误差判定

本单为**验证/闭环任务**，无物理量-基准数值比较，不适用 <3% 判据；
判据为「吸收忠实 + 回归测试通过 + 无残余可微路径强转漏洞」，三项均满足
（见③表，逐项显式给出）。无含糊。

## 结论

`TLBM-HOTPATCH-ABSORB` 的吸收工作已由 #2662 完成且质量合格（含镜像补丁没有的
回归测试），本单独立复核通过，建议将该锚点置为 done/verified。
