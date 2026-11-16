# M23777 — TLBM 镜像未入库热补丁吸收（可微路径保护）

## 锚点
`TLBM-HOTPATCH-ABSORB`

## 背景与意图
`/srv/tensor/workspace/TensorLBM` 生产镜像工作树有他人未提交改动（只读参考，禁改镜像）：

1. **`src/tensorlbm/input_contract.py`** — `_as_float` 数值强转处加 `torch.Tensor` 保护：
   含 0-dim Tensor 不 coerce float，防 `float(tau)` 静默 detach autograd 图、破坏
   solver-in-the-loop 可微优化路径（tau 反演）。
2. **`tests/test_diff_lid_cavity.py`** — 关联改动：`test_invert_recovers_tau_smoke`
   初始猜测从 `tau_true * 0.92`（低于真值）改为 `tau_true * 1.08`（高于真值）。
3. **未跟踪 `artifacts/test_debt/`** — 跨仓 JUnit 测试报告（运行产物）。

**意图理解**：可微路径保护（不改物理）。`_as_float` 是 `check_relaxation_time` /
`check_positive_scalar` 共享的数值强转入口；0-dim `torch.Tensor` 对 `float()` 是
"plain scalar"，无保护时被 coerce 成 python float，静默剥离 autograd 图。

## 方法
1. 在自有工作树（`agent/m23777`）复现补丁语义：逐字节应用镜像热补丁。
2. 补回归测试 `test_input_contract_keeps_0dim_tensor_in_graph`：
   - **无保护基线**：`check_relaxation_time(torch.tensor(0.66, requires_grad=True))`
     返回 `0.66`（float），`requires_grad` 丢失 → 测试 **FAIL**。
   - **有保护**：返回 `tensor(0.66, requires_grad=True)`，图保持 → 测试 **PASS**。
3. 跑相关套件证明无回归。
4. 正式 PR 入 main（提交说明注明来源=生产镜像热补丁追认）。
5. `artifacts/test_debt/` 判定：加入 `.gitignore`（运行产物，不提交）。

## 计算结果（before/after 回归证据）

| 检查点 | 无保护基线 | 有保护（本 PR） |
|---|---|---|
| `_as_float(0-dim tensor)` | `0.66` (float) | `None` |
| `check_relaxation_time` 返回 | `0.66` (float, 图剥离) | `tensor(0.66, requires_grad=True)` |
| `check_positive_scalar` 返回 | `0.66` (float, 图剥离) | `tensor(1.0, requires_grad=True)` |
| 新回归测试 | **FAIL** (`AssertionError: tensor tau must not be coerced to float`) | **PASS** |
| 普通标量范围检查 | 保留 | 保留（无行为变化） |

## 测试结果
- `tests/test_diff_lid_cavity.py`（非 slow）：**22 passed**，1 deselected。
- `tests/test_diff_lid_cavity.py::test_invert_recovers_tau_smoke`（slow，新初始猜测）：**1 passed**（61.8s）。
- `tests/test_input_contract_guards.py`：**29 passed**。
- `tests/test_autograd.py` tensor-tau 路径（`test_collide_step_gradients_f_and_tau` /
  `test_mrt_float_tau_unchanged_by_graph_fix` / `test_3d_mrt_tau_gradient_flows`）：**7 passed**。

## 注意点
- 镜像树一个字节都不动——补丁以 PR 回流，镜像 ff 后自然清洁。
- `input_contract.py` 的 `_as_float` 用 `try: import torch` 惰性导入，避免在
  无 torch 环境（纯 numpy 路径）引入硬依赖；`ImportError` 时回退到原 `float()` 逻辑。
- 改动最小：仅 `_as_float` 加 8 行保护 + 1 个回归测试 + 1 行 `.gitignore`。

## 误差判定
本单为**缺陷修复/热补丁吸收**，非物理量计算任务，无基准数值可比。
判据为"保护必要且无回归"：无保护时回归测试 FAIL（图剥离），有保护时 PASS（图保持），
且普通标量范围检查行为不变——已显式验证，无含糊。

## artifacts/test_debt/ 去向
**判定：加入 `.gitignore`（`artifacts/test_debt/`）**。
内容为跨仓 JUnit 测试报告（`cross_repo/junit/TensorLBM/TensorLBM.xml`），
属运行产物（test debt），非项目源码；与既有 `artifacts/runs/` 忽略模式一致，
保留在磁盘但不入库。`git check-ignore -v` 已确认生效。
