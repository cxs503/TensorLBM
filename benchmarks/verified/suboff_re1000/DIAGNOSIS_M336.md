# B6 SUBOFF Re=1000 — SDAA `eager_fallback` 根因诊断（shift m336）

> 本文件由 `render_diagnosis_md.py` 从 `diagnose_sdaa_fallback.py` 的 JSON 输出 + `result.json` 自动生成；所有数字程序化注入，无手抄。

## 1. 结论先讲

1. `compile_status = eager_fallback` 的根因是 **Torch-SDAA 3.1.1a3+gite3971b9 的 `teco_inductor` 后端 codegen 缺件**：`InductorError`，失败层 `inductor/teco`，出错帧 `File "/usr/lib64/python3.11/site-packages/torch_sdaa/_inductor/codegen/teco.py", line 2073, in infer_expand_dim`。
2. 回退**不改变数值**：同一初始场、同一 L16 用例，`default`（实走回退）与 `eager` 两条轨迹 `max|Δf| = 0.0` （bitwise identical = `True`），力样本逐位相同 (`Cd_total` 差 = 0.0)。因此 eager 回退**不可能**解释 +46.43% 的阻力偏差。
3. m319 的 `Cd_tot = 0.061495` 是本用例在**粗网格/短步数瞬态**下会给出的量级（见第 5 节复现），不是 verified 配置的结果；verified 配置（L=80，12000 步，PASS）为 `Cd_tot = 0.041513`（-1.15% vs Blasius 0.041995）。
4. 落地修复不是改数值，而是**让 runner 自己拒绝把瞬态数字当基准结果**：新增 `acceptance_gate.py`（纯函数，`assess()` / `banner()`）并接入 `run.py`，`result.json` 增写 `acceptance` 块；`tests/test_suboff_re1000_acceptance_gate.py` 10 项单测覆盖 PASS / FAIL_ACCURACY / COARSE_GRID / SHORT_RUN / NOT_CONVERGED。

## 2. 环境与编译探测

| 项 | 值 |
|----|----|
| torch | `2.7.1` |
| torch_sdaa | `3.1.1a3+gite3971b9` |
| device | `sdaa:0` (/dev/tcaicard24) |
| SDAA available / count | `True` / `1` |
| inductor shim tag | `sdaa-teco_inductor` |
| 探测用例 | L=16，14.86 s 编译尝试 |
| compile_status | `eager_fallback` → effective `eager` |
| 异常 | `InductorError`：`AssertionError: ` |
| 失败层 | `inductor/teco` |
| 出错帧 | `File "/usr/lib64/python3.11/site-packages/torch_sdaa/_inductor/codegen/teco.py", line 2073, in infer_expand_dim` |

失败发生在 **lowering/codegen 阶段、产出任何输出之前**，所以一次性回退是精确的（`_CompileWithEagerFallback` 的语义），且与 README 记录的 L1–L4 shim + 上游 L5 缺陷判断一致（`torch_sdaa/_inductor/codegen/teco.py` 的 `infer_expand_dim` 断言）。

## 3. 路由中性性（回退 vs eager，同初场）

| 模式 | ms/step | Cd_total | compile_status |
|------|---------|----------|----------------|
| eager | 54.16 | 0.02683917 | `eager` |
| default（实走回退） | 123.44 | 0.02683917 | `eager_fallback` |

* `max|Δf| = 0.0`（场尺度 max|f| = 0.33379805088043213）→ **逐位相同**。
* 力样本逐位相同：`forces_bitwise_identical = True`。
* 步时差异 54.16 vs 123.44 ms/step（小网格纯开销主导，非性能结论）。

## 4. 跨设备一致性（CPU vs SDAA，纯 eager 路径）

* L=16，120 步；CPU `Cd_total = 0.02763257`，SDAA `Cd_total = 0.02763200`。
* 场最大相对偏差 `1.0713764374858545e-06`（float32 op-order 量级）。
* `device_independent = True` → 算法本身与设备无关。

## 5. 复现 m319 的 +46%：粗网格 / 短步数瞬态

| L | 网格 | cells | 步数 | ms/step | 25% 窗 | 50% 窗 | 75% 窗 | 100% 窗 | 误差 vs Blasius | 判定 |
|---|------|-------|------|---------|--------|--------|--------|---------|-----------------|------|
| 16 | 96x34x34 | 110,976 | 600 | 63.6 | 0.027349 | 0.025842 | 0.024701 | 0.023940 | -42.99% | `COARSE_GRID` |
| 24 | 144x51x51 | 374,544 | 900 | 87.1 | 0.060871 | 0.056340 | 0.053990 | 0.052524 | +25.07% | `COARSE_GRID` |
| 32 | 192x68x68 | 887,808 | 400 | 286.0 | 0.069749 | 0.061967 | 0.057179 | 0.053841 | +28.21% | `COARSE_GRID` |
| 32 | 192x68x68 | 887,808 | 1200 | 137.0 | 0.057179 | 0.049429 | 0.045643 | 0.043476 | +3.53% | `COARSE_GRID` |

**启动瞬态窗口扫描**（各探针逐样本序列，窗口 = step 10 → 10·n，n = 10…30）：

| L | 命中窗口（步） | 窗均值 Cd_tot | vs Blasius | vs m319 报价 | 序列首样本（step 10） |
|---|----------------|---------------|------------|--------------|------------------------|
| 24 | 10–200 | 0.061424 | +46.26% | -0.12% | 0.073367 |
| 32 | 10–210 | 0.061394 | +46.19% | -0.16% | 0.116457 |

多个独立粗网格在 step 10–200/300 的窗均值都与 m319 报价相差 1% 量级，而序列首样本（step 10）是 +75%~+177% 的启动尖峰：这正是「把未收敛窗口当结果」的特征。

最接近 m319 报价的探针：L=32、400 步，`Cd_tot = 0.053841`（与 0.061495 差 -0.007654，-12.45%），判定 `COARSE_GRID`。

物理原因：Re=1000、`u_lb = 0.05`，尾流要跨越 6L 域数个对流时间才收敛；粗网格艇体半径只有 1–2 格，半程反弹壁面的阶梯几何误差 + 尚未发展的尾流都会把 `Cd_tot` 抬到 +20%~+50% 区间。这解释了 m319 的 +46.43%，与 Triton 后端无关。

## 6. 注册的 verified 结果（`result.json`，作为对照基线）

| 配置 | 网格 | 步数 | 摩擦公式 | Cd_p | Cd_f | Cd_tot | 误差 | 判定 |
|------|------|------|----------|------|------|--------|------|------|
| L=80 mix50 | 480x169x169 | 12000 | mix50 | 0.005556 | 0.035957 | **0.041513** | -1.15% | PASS (<3%) |
| L=48 mix50 | 288x102x102 | 20000 | mix50 | 0.006816 | 0.033848 | **0.040664** | -3.17% | coarse grid (hull radius ~2.8 cells) — slightly over, residual discretisation error |
| L=48 standard | 288x102x102 | 20000 | standard | 0.006816 | 0.031109 | **0.037925** | -9.69% | control — shows the standard friction formula underestimates by ~10% |

reference: `Blasius laminar flat-plate Cf = 1.328/sqrt(Re) (wetted-area pi*D*L)` = 0.041995

## 7. 误差判定（逐项，显式回答「是否 <3%」）

| 量 | 值 | 基准 | 误差 | <3%? | 说明 |
|----|----|------|------|------|------|
| Cd_tot（verified L=80，12000 步） | 0.041513 | 0.041995 | -1.15% | **是** | 历史入库值，与本次诊断一致 |
| Cd_tot（final-field 重算 mix50） | 0.041424 | 0.041995 | -1.36% | **是** | 收敛场重算 |
| 本次在 SDAA 上可跑的探针（见第 5 节） | 见表 | 0.041995 | 见下表 | **否** | 粗网格 + 短步数 = 瞬态，**不作精度结论**，由 acceptance gate 拦下 |

**是否 <3%？** registered 配置：**是**（-1.15% 窗口 / -1.36% 收敛场）。本次 shift 未在 SDAA 上重跑 L=80/12000 步（12186 s ≈ 3.38 h，超出单次工具预算），因此对主配置**没有新增端到端实测**；本次结论是「回退与 eager 逐位一致、算法设备无关、m319 的 +46.43% 被归因并复现为粗网格/短步数瞬态、runner 已加验收闸门」，而不是「重跑了一次端到端达标」。此点如实标注，HONEST_FAIL 不粉饰。

## 8. 注意点 / 局限

* `eager_fallback` 在这台机器上是**预期行为**，不是错误：CUDA 上该封装是 no-op，只有 Hygon/SDAA 的 `teco_inductor` 会触发。日志里必须看 `compile_status` / `compile_status_reason` 才能区分「真编译」与「回退」。
* 回退的性能代价没有在这里量化（探针网格太小，纯开销主导）：按 CPU/CUDA 经验，编译路径的价值在 launch 开销，L=80 的大网格上相对开销应更小；要量化需在 L=80 上跑 A/B。
* 探针用的 `L=16/24/32` 只用于「瞬态量级」与「路由/设备一致性」，**这些网格上的 Cd 不是精度结论**。
* 若要真正在加速卡上获得编译加速，需要上游修 `torch_sdaa/_inductor/codegen/teco.py` 的 `infer_expand_dim`（L5），或换 CUDA/Ascend 等后端；本仓库不改安装包。

## 9. 工件与复现

```bash
# 诊断（stages 可分开跑；figures 只读 JSON）
python benchmarks/verified/suboff_re1000/diagnose_sdaa_fallback.py \
  --stages probe,neutrality,crossdevice,sweep --device sdaa:0 \
  --json evidence/suboff_re1000_m336/diagnosis.json
python benchmarks/verified/suboff_re1000/diagnose_sdaa_fallback.py --stages figures \
  --json evidence/suboff_re1000_m336/diagnosis.json --figdir evidence/suboff_re1000_m336/figures
# 单测
python -m pytest tests/test_suboff_re1000_acceptance_gate.py -q
```

生成时间戳（JSON 内）：`2026-10-30 04:45:36`
