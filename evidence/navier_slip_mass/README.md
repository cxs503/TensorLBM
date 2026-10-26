# Navier slip 质量守恒失败：本地复现与候选测试修正

## 状态与范围

- 日期：2026-10-25；基线提交：`ad3d19ae6e5f55f02ac691f6719d4a5ae68f4cdc`。
- 工作分支：`arena/d059dc71-tensorlbm`。
- **这是开发侧证据，不是独立验收通过，也不是求解器物理正确性的证明。**
- 本次只改测试及添加可复现证据；没有修改求解器、边界算法、CI、隔离名单或依赖声明。
- 原始 float32 算例仍不能满足绝对 `1e-6` 要求。下面的候选修正改变了整步测试的计算精度，并非修复了 float32 漂移；需要审查者明确接受这一测试契约调整。

## 环境

初始工作区可读写且有终端，但没有安装 PyTorch。已在忽略的 `.venv/` 中安装测试依赖。

- Python 3.11.2，PyTorch 2.2.2+cu121，CPU，单线程。
- `torch.cuda.is_available()` 为 false；wheel 含 CUDA 支持不代表有可用 GPU。
- 精确包版本见 `environment.txt`。该环境不是当前 GitHub CI 环境的精确副本。
- 未连接 SSH、HPC 或其他 Agent；未触发远程 CI，未提交、推送或合并。

## 复现与算子归因

原始测试模块结果：**3 passed, 1 failed**，见 `baseline.log`。

诊断完全保留原测试的尺寸 `(4,21,16)`、静止平衡初态、100 步、tau=1、上下壁面及 Ls=2；分别关闭边界和改用 float64 对照。每个算子前后的分布先转 float64，再逐元素相减并归约，以区分实际存储值的变化与最终 float32 总和的量化。

| 演化精度 | 边界 | 原始总和绝对差 | 分布差的 float64 累加（带符号） |
|---|---|---:|---:|
| float32 | 开 | 1.220703125e-4 | 4.6634674072265625e-4 |
| float32 | 关 | 1.220703125e-4 | 4.6062469482421875e-4 |
| float64 | 开 | 0 | 2.984279490192421e-13 |
| float64 | 关 | 0 | 2.984279490192421e-13 |

float32 有边界轨迹的累计算子质量变化：

- 碰撞：`-1.0585784912109375e-4`。
- 迁移：`0`。
- 边界：`+5.7220458984375e-4`；单步边界最大变化 `5.7220458984375e-6`。

这些数值是同一轨迹内的分阶段记账；有、无边界两条轨迹不同，不能将不同轨迹的碰撞贡献直接当成同一反事实分解。全部结果见 `operator_audit.json`。

### 可以得出的结论

1. 失败可在本地重复，不是仅根据历史 CI 日志推测。
2. 1344 附近 float32 的 ULP 为 `1.220703125e-4`，原门槛 `1e-6` 小于一个 ULP。并且不仅总和归约存在舍入，存储的分布本身也变化了；**只把最终求和改为 double 不能解决原失败**。
3. 无边界对照也失败，因此不能仅凭该失败归罪于 Navier slip 算法。
4. 对静止壁面，bounce-back 与 specular 都是方向置换，其凸组合在精确算术中逐格守恒。新增测试在非平衡分布上检查该代数性质及独立的动量恒等式。
5. 这些结果符合有限精度舍入解释，但不能外推为长期、有驱动流或所有设备上的漂移均可接受。

## 候选变更

`tests/test_navier_slip_bc.py`：

- `test_navier_slip_mass_conservation` 的演化从 float32 改为 float64；保留 100 步、原边界及绝对 `1e-6` 断言。纠正“施加流动”注释：原用例其实是静止平衡态。
- 新增边界独立测试：float32/float64 × 三轴 × Ls={0,0.25,2,∞}，明确参数化 CPU/CUDA。
- 使用固定种子的非均匀、非平衡正分布，检查逐格密度、总质量、法向动量反向、切向动量缩放、非壁面逐位不变、输入不被修改以及 dtype/device 保持。
- 局部容差为 `8 * eps(dtype) * rho_before`，按插值运算舍入设置，不从失败漂移反推；使用 float64 归约。此容差是新增单步算子测试的误差预算，不是整步物理漂移标准。
- 不删除原测试，不使用 xfail，不添加质量归一化或补偿。

**覆盖变化必须明确：原来的 float32 100 步绝对门槛不再由该 pytest 用例强制执行。** 原始 float32 失败仍由诊断脚本直接重现和记录。若产品要求 float32 100 步也绝对漂移小于 `1e-6`，本次不满足该要求，必须另立跨碰撞/归约/边界的数值精度任务，不能接受此次测试变更作为算法修复。

## 实际执行结果

| 检查 | 结果 | 证据 |
|---|---|---|
| 原始目标模块 | 3 passed, 1 failed | `baseline.log` |
| 修改后目标模块 | 28 passed, 24 skipped（CUDA 不可用） | `target.log` |
| 六模块邻近回归，含目标模块 | 94 passed, 24 skipped（CUDA 不可用） | `regression.log` |
| Ruff（变更的两个 Python 文件） | passed | 本地命令输出 |
| `git diff --check` | passed | 本地命令输出 |

回归范围：Navier slip、D3Q19、低内存 D3Q19、移动盖板边界、边界注册、自由表面应用。回归并非全仓测试。日志里的五个 FastAPI 缺失文件是 conftest 的全局 collection-ignore 报告，不计为已验证。

### 复现命令

在仓库根目录执行；新环境可先执行：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r evidence/navier_slip_mass/environment.txt
```

保留 `pipefail`，避免 tee 掩盖 pytest 的失败退出码：

```bash
set -o pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=src .venv/bin/python evidence/navier_slip_mass/reproduce.py \
  | tee evidence/navier_slip_mass/operator_audit.json
.venv/bin/python -m pytest tests/test_navier_slip_bc.py -q -ra \
  2>&1 | tee evidence/navier_slip_mass/target.log
.venv/bin/python -m pytest \
  tests/test_navier_slip_bc.py tests/test_d3q19.py \
  tests/test_low_memory_d3q19.py tests/test_boundaries3d_moving_lid.py \
  tests/test_boundary_registry.py tests/test_free_surface_application.py \
  -q -ra --timeout=120 2>&1 | tee evidence/navier_slip_mass/regression.log
.venv/bin/ruff check tests/test_navier_slip_bc.py evidence/navier_slip_mass/reproduce.py
git diff --check
```

诊断脚本只报告原门槛是否通过，不因原失败而非零退出；pytest 才是自动断言入口。复现原始 pytest 失败需要使用基线版本的测试文件；这里不通过切换分支或覆盖当前修改来重跑基线。

## 独立审查与后续验收

- [ ] 审查者确认“高精度整步守恒 + 各精度局部算子检查”的测试契约是否符合产品目标；不要将测试通过表述成 float32 漂移已修复。
- [ ] 在实际 CI 支持的 Python/PyTorch 版本重跑上述命令并保存结果。
- [ ] 在实际 GPU 上运行显式 CUDA 参数用例；当前 24 个跳过不算通过。
- [ ] 另设受驱动通道的解析速度剖面/有效滑移长度验证，明确壁面位置、松弛时间、网格收敛、质量统计域及长期漂移预算。当前测试不验证 `beta=Ls/(1+Ls)` 的物理标定。
- [ ] 移动壁面的密度权重、速度方向约定等需独立审查；本次新增边界测试限定 `u_wall=0`，邻近移动盖板回归不等于验证 Navier 移动壁面。
- [ ] 最终由独立审查者或负责人决定接受、拒绝或要求算法层修复；本次没有执行合并。
