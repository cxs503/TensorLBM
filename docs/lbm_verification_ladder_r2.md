# LBM 解析解验证梯 Step A–D 复核（v2, 2026-11-06）

**一句话**：TensorLBM 的四个解析解基准（剪切波/Couette/Poiseuille/壁面摩擦）在 CI 里被
quarantine 多年，实探发现它们的**失败模式不是物理错，而是验证框架本身是假的**——pytest
收集到的是 `device="sdaa:0"` 默认值（无卡机器直接挂死），而且四个"测试"只 `return bool`
从不 `assert`，因此物理再错也是绿的。本单把它重建成真门：CPU 默认设备 + 向量化内核 +
真断言，四个基准全部通过，关键量误差 **1.7e-5% / 6.6e-12% / 0.337% / 8.7e-12%**，均 <1%
（远严于 3% 判据）。

产物：
- `tests/test_lbm_verification.py`（重写：真断言 + 设备/dtype/线程可配置 + CLI 分块合并）
- `scripts/lbm_verification_report.py`（程序化出表出图，零手抄）
- `artifacts/lbm_verification_ladder/{metrics.json, f32/metrics.json, tables.md, fig_ladder.png, fig_dtype_error.png}`
- `tests/ci_quarantine.txt` 删除 4 行（burn-down）

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

四个案例组成 LBM 的"最小完备"验证梯：碰撞算子（ν）、边界条件（无滑移）、外力项（体力）、
壁面通量（剪切应力）——正好对应求解器出船舶阻力前必须可信的四件事。

| Step | 物理 | 解析解 | 几何 / 参数 | 时间步 |
|---|---|---|---|---|
| A | 剪切波衰减（验证碰撞与 ν） | u_y(x,t)=A₀cos(kx)e^(−νk²t)，ν 由振幅比反演 | 64×4×4 全周期，τ=1 → ν=1/6，k=2π/64，A₀=0.01 | 100 |
| B | Couette 流（验证半路反弹无滑移） | u(y)=u_top·(y−0.5)/(ny−1−0.5) | 80×12×4，底壁固体、顶壁运动平衡态 u_top=0.05，流向周期，τ=1 | 2000 |
| C | Poiseuille 流（验证 Guo 体力） | u(y)=G/(2ν)(y−0.5)(H+0.5−y), H=10 | 80×12×4，双壁固体，G=2νu_max/H², u_max=0.05 | 3000 |
| D | 壁面摩擦（验证壁面剪切应力） | Cf=2ν/((H−0.5)U)，壁面在 y=0.5 | 80×12×4，同 B，U=0.05，间隙 10.5 | 3000 |

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

**数值路径**：`tensorlbm.solver3d.collide_bgk3d`（BGK）→ NoDynamics 固体回滚 →
`tensorlbm.boundaries3d.bounce_back_cells_3d`（半路反弹）→ 顶壁运动平衡态 / Guo 强迫 →
`stream3d` → 流向周期。dt=dx=1 格子单位。

**v2 修的三处欠账**（这是本单的实质增量）：

1. **设备**：v1 的四个测试函数签名默认 `device="sdaa:0"`，pytest 收集到的就是该默认值；
   无 SDAA 的机器上整个文件直接挂死（本机实测 pytest 115 s 无输出被强杀）。v2 用
   `resolve_device()`：`TENSORLBM_TEST_DEVICE` > 缺省 `cpu`，请求 sdaa 而硬件不可用时
   `pytest.skip` 而不是死等。
2. **断言**：v1 是 `return passed`，pytest 眼里永远通过 → 验证梯同虚设。v2 每个 Step 都
   `assert err_pct < 1%`，断言消息里带实测数字。
3. **步耗时**：v1 每步 `for q in range(19): f[q]=torch.where(...)` 加 19 次逐方向加力；
   v2 改成单次广播 `torch.where(mask, f_pre, f)` 与一次向量化 Guo 强迫（逐位等价），
   并加 autouse fixture 把线程限到 1（`TENSORLBM_TEST_THREADS` 可覆盖，退出恢复原值）。
   本机 128 核且 load15≈120 时实测 **12 ms/步 → 0.4 ms/步（29×）**。

**可复现命令**（本机 load 高，整梯一次跑不完就分块，再合并）：

```bash
PYTHONPATH=src python tests/test_lbm_verification.py --only AB --json artifacts/lbm_verification_ladder/metrics_ab.json
PYTHONPATH=src python tests/test_lbm_verification.py --only C  --json artifacts/lbm_verification_ladder/metrics_c.json
PYTHONPATH=src python tests/test_lbm_verification.py --only D  --json artifacts/lbm_verification_ladder/metrics_d.json
PYTHONPATH=src python tests/test_lbm_verification.py --merge artifacts/lbm_verification_ladder \
        --json artifacts/lbm_verification_ladder/metrics.json
PYTHONPATH=src python scripts/lbm_verification_report.py   # 出 tables.md + 两张图
# float32 对照：加 TENSORLBM_TEST_DTYPE=float32，落 artifacts/lbm_verification_ladder/f32/
```

**工具链**：Python 3.11 / torch 2.7.1 / numpy；CPU 单线程；float64 主判据（验证的是数值方法
本身，不是 float32 舍入）。

## 3. 计算结果（数值表，程序化生成）

数字来自 `artifacts/lbm_verification_ladder/tables.md`（由脚本从 metrics JSON 生成，无手抄）。

| Step | 案例 | 关键量 | 数值解 | 解析解 | 误差 % | 判据 % | 判定 |
|---|---|---|---|---|---|---|---|
| A | shear_wave | 运动粘度 ν | 0.166667 | 0.166667 | 1.7218e-05 | 1 | PASS |
| B | couette | 剖面误差 max Δu/u_top | 6.6405e-12 % | 0（解析剖面） | 6.6405e-12 | 1 | PASS |
| C | poiseuille | 峰值 u_max | 0.0124167 | 0.012375 | 0.336701 | 1 | PASS |
| D | wall_friction | 壁面摩擦系数 Cf | 0.634921 | 0.634921 | 8.6556e-12 | 1 | PASS |

dtype 敏感度（解释 C 的 0.337% 从哪来）：

| Step | float64 误差 % | float32 误差 % | 倍数 | 结论 |
|---|---|---|---|---|
| A | 1.7218e-05 | 0.00270013 | 157 | float32 舍入地板主导 |
| B | 6.6405e-12 | 0.0013256 | 2e8 | float32 舍入地板主导 |
| C | 0.336701 | 0.337789 | 1.00 | 物理/离散误差主导（dtype 不敏感） |
| D | 8.6556e-12 | 0.0042011 | 4.9e8 | float32 舍入地板主导 |

结论：float64 下 A/B/D 只是舍入噪声，真正的物理误差只有 C 的 0.337%——那是
Ma=u_max√3≈0.021 的压缩性与有限差分剖面近似的固有偏差，与 dtype 无关。

## 4. 结果图

- `artifacts/lbm_verification_ladder/fig_ladder.png`：Step A 振幅衰减（对数坐标）与
  exp(−νk²t) 解析曲线重合；Step B/C 剖面 vs 解析解（点=LBM，线=解析，肉眼不可分）。
- `artifacts/lbm_verification_ladder/fig_dtype_error.png`：float64 vs float32 误差柱状图，
  1% 阈值线一目了然。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

- **坑 1（本单根因）**：`return bool` 的"测试"是假绿。审计同类文件时先 `grep -n "return passed"`
  与 `assert` 计数，别信文件名里的 "verification"。
- **坑 2**：pytest 路径与 CLI 路径必须共享同一套环境处理（第一版只给 CLI 加了线程限制，
  pytest 仍然 12 ms/步、115 s 跑不完两个用例）。现用 autouse fixture 统一。
- **坑 3**：`torch.device("sdaa:0")` 在无卡机器上会挂死而非快速报错；任何测试默认值都不该钉在
  加速器上。`grep 'device="sdaa:0"'` 仍命中 `test_lbm_step.py`、`test_multi_device_sdaa_common.py`、
  `test_suboff_*_runner.py` 等多个文件，属同类风险，建议后续逐一对齐 `resolve_device()` 模式。
- **局限 1**：仅 BGK 单松弛碰撞；MRT / cascaded / KBC 的粘性与稳定性不在本梯覆盖内。
- **局限 2**：壁面为平直壁 + 半路反弹；曲面壁（插值反弹/BFL）、壁函数、移动壁、MRT+MRT
  组合的无滑移精度未验证。
- **局限 3**：全部为低速小域（Re 极低、Ma≈0.02），不构成湍流/高 Re 的精度证据；船舶阻力
  类结论仍须走 SUBOFF 等实验基准链（见 `docs/suboff_*`）。
- **局限 4**：整梯一次跑约 138 s（A+B 45.8 s，C+D 92.3 s，128 核但被其它负载挤压）。CI 单测
  超时 600 s 内安全；但若要更省，应把 B/C/D 的步数改成按残差收敛判停，而不是固定 3000 步。

## 6. 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%? | <1%? |
|---|---|---|---|---|
| ν（Step A） | (τ−0.5)/3 = 1/6 | 1.72e-05 % | 是 | 是 |
| Couette 剖面（Step B） | 解析线性剖面 | 6.64e-12 % | 是 | 是 |
| u_max（Step C） | G/(2ν)·(H/2−0.5)² | 0.3367 % | 是 | 是 |
| Cf（Step D） | 2ν/((H−0.5)U) = 0.634921 | 8.66e-12 % | 是 | 是 |

**四项全部 <1%，故 <3% 判据满足。** 无 WITHHELD 项。诚实声明：C 的 0.337% 是真实物理/离散
误差而非噪声（float32 同值），不应被"全绿"掩盖；A/B/D 的 1e-12 量级说明公式与实现完全自洽，
但也意味着这三个基准对实现细节不敏感（例如粘性公式若整体错一个常数仍会暴露，但若 τ 与 ν 的
关系被同时改掉则不会——真正的独立校验必须靠 float32/float64 + 不同 τ 的扫描，未在本单展开）。