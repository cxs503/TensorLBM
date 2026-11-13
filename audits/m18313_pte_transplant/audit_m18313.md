# 交差审计 m18313 → m18195 XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT（TensorLBM）

- 审计对象：mission 18195（status=succeeded，branch agent/m18195，head `caa9059520b37caf4532d652148400a40161c02f`）
- 审计方：board-audit（m18313），日期 2026-11-13
- 结论：**L2（独立功能实现）confirmed**；L3/L4 不成立（缺件清单见 §5）

## 1. 验收三件套核验

| 件 | 判定 | 证据 |
|---|---|---|
| run_artifacts 文件 | ✅ 真存在 | `artifacts/xval_pressure_tensor_equilibrium/results.json`（437 行真实数值）；`docs/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT.md`（202 行）；`docs/figures/xval_pte_{convergence,fields}.png`（86050/90786 B）；`xval/.../ref/oss_discrete_pressure_tensor.json`；`predeclared.json` |
| benchmark ref vs got 数值 | ⚠️ 工件内存在、登记基准 0 行 | 工件内 ref/got 齐全（§3）；但 `tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT'` → **0 行**（psql 直查）；平台 L4 门 `CLAIMED_NOT_PROVEN`（rows=0） |
| repro_cmd 可执行 | ✅ 审计方复跑通过 | `run_xval.py` 5.0 s exit=0，verdict PASS，worst_resolved_sigma_error=0.06089506891217726 与提交工件**逐位一致**（diff 仅 wall_seconds 3.8→5.0）；`pytest tests/test_pressure_tensor_equilibrium_xval.py -q` → **12 passed**（6.29 s） |

## 2. 实现与集成面

- 实现：`src/tensorlbm/phasefield/pressure_tensor_equilibrium.py`（614 行）——D2Q9 自由能二元流体平衡态，Korteweg 应力折进平衡态分布，压强直接读分布函数二阶矩；9 点各向同性 grad/lap；flat/droplet 诊断。
- 入口：`xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/run_xval.py`（审计方重跑成功）。
- 符号裁判：`derive_oss_pressure_tensor.py` sympy 推导离散二阶矩 = 连续自由能压强张量，残差 **0**（数值闭合残差 1.7e-16）。
- 集成调用方：**零**。全仓 grep `pressure_tensor_equilibrium` 仅命中模块自身/测试/xval 目录；模块 docstring 明言 “no solver in the repository is modified”。⇒ **L3 不成立**。

## 3. 基准裁判（唯一裁判）

- `tensor_benchmark_cases` 该 anchor **0 行**；`/tensor/bench/results?mission_id=18195` total=0；`/tensor/bench/l4-verdict/18195` = **CLAIMED_NOT_PROVEN**（failed_checks: execution_success / source_cluster_job / cluster_job_bound / ref_got_error_present / within_tolerance）。
- 工件内 ref vs got（程序化抽取自 results.json，零手抄，见 `numbers_from_artifact.json`）：

| 臂 | 点 | σ_ref | σ_got | 误差 | <15% |
|---|---|---|---|---|---|
| 平界面 | ξ=1 | 0.066667 | 0.053629 | -19.56% | ❌（欠分辨，已文档化剔除） |
| 平界面 | ξ=1.5 | 0.100000 | 0.089849 | -10.15% | ✅ |
| 平界面 | ξ=2 | 0.133333 | 0.125214 | -6.09% | ✅ |
| 平界面 | ξ=3 | 0.200000 | 0.194314 | -2.84% | ✅ |
| 平界面 | ξ=4 | 0.266667 | 0.262325 | -1.63% | ✅ |
| 液滴 | ξ=2, R=10/12/16/24 | 0.133333 | 0.124310…0.124858 | -6.36%…-6.77% | ✅ |
| 液滴 | ξ=3, R=10/12/16/24 | 0.200000 | 0.193748…0.194772 | -2.61%…-3.13% | ✅ |
| 力形式对照臂 B | R=16, ξ=2 | 0.133333 | 3.544e-08 | -100.00% | ❌（结构性读不出机械压强，honest 记录） |

- 已分辨子集（ξ≥2）全部 <15%；3% 目标档仅 ξ≥3 达到（G2b=false 如实记录）。
- 外部锚：sailfish-team/sailfish @ `b7140c5`（LGPL3），4 个 vendored 文件 sha256 审计方独立复算**全部匹配**（lb_binary.py `92527092…`、sym.py `77dfe766…`、sym_equilibrium.py `6ef58388…`、mako `f5479378…`）。

## 4. 报告六段核验

①问题+ref 出处 ✅ ②方法与实现 ✅ ③参数表（无量纲）✅ ④对照表（ref vs got 打勾叉）✅ ⑤收敛性/参数敏感性（ξ/R/τ/A 四轴扫描）✅ ⑥失败模式与工件 ✅（+⑦本仓 vs OSS 差异清单）。平台报告 id **17063** published。预声明阈值先于扫描固定（predeclared.json，无事后放宽）；ξ=1 欠分辨点如实记为失败模式；对称双井 Δp_bulk≡0 故不宣称 Laplace 百分比（避免循环论证）——honest 纪律良好，无粉饰。

## 5. 判级与缺件清单

- **L2 成立**：真代码（输入→计算→输出）+ 12 单测 + 机器精度代数闭合 + 可复现扫描 + 外部锚 sha256 钉死。
- **L3 不成立**：零集成调用方（未接入主求解器/平台）。
- **L4 不成立**：锚点基准 0 行 + 平台重算行 0 行（L4 机器门 CLAIMED_NOT_PROVEN）。
- 消账路径（若日后要升 L4）：① 在 `tensor_benchmark_cases` 登记该 anchor（case_name/metric/tolerance/ref_source）；② 把平界面/液滴 σ 算例纳入集群重算产 ref/got/error 行；③ 接入主求解器（phasefield 链）后补 L3。

## 6. 审计方复现记录

- `cd /root/tp-agent/work/m18195 && python3 xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/run_xval.py` → PASS（5.0 s）
- `python3 -m pytest tests/test_pressure_tensor_equilibrium_xval.py -q` → 12 passed（6.29 s）
- `sha256sum ref/sailfish_src/...`（4 文件）→ 与 results.json provenance 全匹配
- `psql -U postgres -d tensorplatform` 直查 `tensor_benchmark_cases` → 0 行
- 平台 API：`/tensor/missions/18195`、`/tensor/reports/17063`、`/tensor/bench/l4-verdict/18195`、`/tensor/bench/results?mission_id=18195`
