# tensorlbm.inverse — 反问题优化基础设施

## 概述

`tensorlbm.inverse` 是 TensorLBM 的可微反问题优化基础设施包，提供：

| 模块 | 功能 |
|---|---|
| `optimizer.py` | 通用 Adam + L-BFGS 抛光优化循环，带收敛检测与盒约束 |
| `losses.py` | 可组合损失函数（相对 MSE、加权 MSE、L1） |
| `convergence.py` | 机读收敛日志 JSON schema 与记录器 |
| `seed.py` | 确定性种子控制与扰动初始化 |

## 验收标准（GAP-PP-4-opt-loop）

- [x] τ/Re 恢复 demo 端到端跑通并产出机读收敛 JSON
      （`--convergence-dir` 每个 (track, seed) 一份 `tensorlbm/inverse/convergence/v1`，
      由 `inverse.validate_convergence_payload` 机检）
- [x] TG 衰减 Re 恢复 <3%（n=16 实测：twin −2.6e−6%，analytic +0.180%；
      n=32 与 6-track × 3-seed 全套见 `cj4540` 工件）
- [x] 3 种子可复现（analytic16 seed 0/1/2 收敛到同一最优点，log ν 极差 ≤3e−9）

> 复核记录：`docs/adversarial_review_inverse_infra.md`（m23029，共 9 条发现，
> F1–F6 已修并加回归测试，F7–F9 为已文档化局限）。

## 快速开始

```python
import torch.nn as nn
from tensorlbm.inverse import (
    optimise_loop, LossComposition, default_relative_mse,
    ConvergenceLog, seed_everything, jittered_init,
)

seed_everything(42)

# 你的可微前向模型
log_nu = nn.Parameter(torch.tensor(-2.0, requires_grad=True))
amplitude = nn.Parameter(torch.tensor(1.0, requires_grad=True))

def forward_fn():
    tau = 0.5 + 3.0 * torch.exp(log_nu)
    sim = my_solver(tau, amplitude)
    return sim, observations

log = ConvergenceLog(problem="my_inverse", seed=42)
result = optimise_loop(
    params=[log_nu, amplitude],
    param_names=["log_nu", "amplitude"],
    loss_fn=default_relative_mse(),
    forward_fn=forward_fn,
    iters=20, lr=0.05, polish=10,
    box_constraints={"log_nu": (-10.0, 0.0)},
    convergence_log=log,
)
log.write("artifacts/my_inverse_convergence.json")
```

## 损失组合

```python
from tensorlbm.inverse import LossComposition, LossComponent, relative_mse_loss, l1_loss

# 多目标损失：MSE + 0.1 * L1
comp = LossComposition((
    LossComponent("mse", relative_mse_loss, 1.0),
    LossComponent("l1", l1_loss, 0.1),
))
# comp(sim, obs) → 标量 tensor
# comp.component_losses(sim, obs) → {"mse": ..., "l1": ...}
```

## 机读收敛日志

输出 JSON 遵循 `tensorlbm/inverse/convergence/v1` schema：

```json
{
  "schema": "tensorlbm/inverse/convergence/v1",
  "problem": "tg_tau_recovery",
  "seed": 0,
  "history": [
    {"iter": 0, "phase": "adam", "loss": 1.0, "params": {"log_nu": -2.0}, ...},
    ...
  ],
  "summary": {
    "converged": true,
    "wall_s": 1.23,
    "final_loss": 1e-8,
    ...
  }
}
```

## 与 opt_loop_tau_recovery 的关系

`tensorlbm.opt_loop_tau_recovery` 是本基础设施的参考消费者——Taylor-Green
涡衰减 τ/Re 恢复 demo。**它确实调用本包**（m23029 之前它自带一份私有
Adam+L-BFGS 循环，本包对它来说是死代码；现在 `recover()` 委托给
`inverse.optimise_loop`，损失经 `LossComposition` 组装，种子经
`seed_everything`/`jittered_init`，收敛日志可写 `ConvergenceLog` schema）。
该模块展示了：

1. **振幅 nuisance 参数的必要性**：固定振幅会导致 +3.9% 偏差（n=16），
   拟合振幅后偏差降至 -0.10%
2. **优化 log(ν) 而非 τ**：保证 τ > 0.5 的物理稳定性约束
3. **Adam + L-BFGS 抛光**：Adam 快速下降，L-BFGS 闭合最后差距

## 测试

```bash
# 基础设施包测试（29 项，含校验器/停止判据/stdlib RNG 种子回归）
PYTHONPATH=src pytest tests/test_inverse_infra.py -q

# Taylor-Green demo 测试（10 项，含"demo 必须委托本包"的 wiring 测试）
PYTHONPATH=src pytest tests/test_opt_loop_tau_recovery.py -q
```

## 产出工件

- `artifacts/opt_loop_verify_n16_seed0.json` — n=16 twin/analytic 臂机读结果（本次复核）
- `artifacts/opt_loop_verify_summary.json` — 上表数字 + 4 份收敛日志的机检结果
- `artifacts/inverse_convergence/{analytic16,twin16}_seed*.json` — 本包 schema 的收敛日志
- `artifacts/fig_inverse_convergence.png` — loss / τ 轨迹图
- `artifacts/opt_loop_tau_recovery.json` — 6 track × 3 seed 生产工件（集群 `cj4540`，
  队列拥堵，交付时未回；回来后按本文件路径落盘即可）
