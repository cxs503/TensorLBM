# XVAL-LBM-PYTEST-BASELINE-REFERENCE — TensorLBM vs `pylbm/pylbm` pytest baseline-reference practice

Anchor `XVAL-LBM-PYTEST-BASELINE-REFERENCE` · mission 18463（m18235 返修轮 1）· branch `agent/m18463`
OSS 臂：`pylbm/pylbm` checkout `/root/oss-cache/pylbm`（按 file:line 引用源码，零手抄）。

## ① 问题 + ref 出处

对标做法 = pylbm 的 pytest 基线参考机制：demo 测试把守恒矩写成 HDF5，与 `tests/reference/*.h5` 逐数据集比对，
容差 `atol=1e-7, rtol=1e-14`；基线可用 `--h5diff-generate-path` 重新生成；缺基线是硬错误而非静默跳过。

| 机制 | 出处 | 行为 |
|---|---|---|
| 测试 id→基线名 | `tests/conftest.py:318-335` | `item.name`；`[`/`]`→`_`；尾部 `_.h5`→`.h5`；`single_reference` 用函数名 |
| 写基线+比对 | `tests/conftest.py:337-410` | 测试函数返回数据对象；逐 key `pytest.approx(rel, abs)` |
| 容差默认 | `tests/conftest.py:345-346` | `atol=1e-7`, `rtol=1e-14` |
| 缺基线 | `tests/conftest.py:383-391` | 抛 "File not found for comparison test … expected for new tests" |
| 生成模式 | `tests/conftest.py:191-198,412-418` | 写基线后 `pytest.skip` |
| 目录覆盖 | `tests/conftest.py:201-208,347-362` | `--h5diff-reference-path`（本地/URL），默认 `<test>/reference` |
| 用例组织 | `tests/test_demo_2d.py:5-6,42-44` | `slow` + `h5diff(single_reference=True)` + 生成器参数化 |
| 图像基线 | `tests/test_domain.py:132`、`test_geometry.py:71`、`test_stencil.py:31` | `@pytest.mark.mpl_image_compare` → `tests/baseline/*.png` |

**对 `oss_solutions` 清单第 19 行的更正**：清单称 `reference/` 是"解析/文献解"；读源码后实为
`reference/`=自洽数值基线(HDF5)、`baseline/`=PNG 图像基线；本报告按源码口径。

## ② 方法与实现

1. **移植机制**：`src/tensorlbm/baseline_io.py`（格式层，无判据；`.npz` 块=命名数组+JSON 元数据含 payload sha256 与环境指纹；
   NPZ 而非 HDF5 因 `h5py` 在本仓是 optional extra、numpy 是硬依赖）+ `src/tensorlbm/verification/baseline_reference.py`
   （判据层：`baseline_name()`=上游 `generate_test_name` 直译；`compare()` 逐元素实现 `|Δ|≤max(atol, rtol·|ref|)`；
   缺基线抛 `ReferenceNotFound`；生成模式 `TLBM_BASELINE_GENERATE`；目录 `TLBM_BASELINE_REFERENCE_DIR`；
   另加上游没有的逐字段 max_abs/max_rel/n_exceeded 与 sha256 防手改）+ CLI 入口（list/check/register）。
2. **生产集成（真实调用链）**：`run_lid_driven_cavity()` 每次运行都写 `runs/**/reference_block.npz`
   （中心线 u/v、rho 场、质量漂移、max|u|、Ghia 分数 + config/tau/nu/壁面约定）。求解器只写 IO、不 import 判据模块。
3. **数值臂**：A=生产入口跑 Ghia Re=100（nx=33/65）；B=用 OSS 自带 h5 在 pylbm demo 自身参数下逐场对比（数组读自 OSS 文件）；
   C=注册基线并复跑比对（上游语义）。
4. **测试**：`tests/test_baseline_reference.py`（18 项）+ `tests/test_cavity_baseline.py`（2 项），共 20 passed；
   基线 `tests/reference/cavity_re100_nx33_u0.1_s300.npz`。

## ③ 参数表（无量纲数完整）

| 臂 | 网格 | tau | nu | Re | u_lid (Ma) | steps (t*) | 碰撞 |
|---|---|---|---|---|---|---|---|
| A-coarse | 33x33 | 0.59900 | 0.03300 | 100 | 0.1 (0.1) | 4000 (t*≈12.1) | MRT |
| A-fine | 65x65 | 0.69500 | 0.06500 | 100 | 0.1 (0.1) | 8000 (t*≈12.3) | BGK |
| B-pylbm | 64x64 | 0.500960 | 0.000320 | 10000 | 0.05 (0.05) | 32 (t*=0.5) | MRT |
| C-baseline | 33x33 | 0.59900 | 0.03300 | 100 | 0.1 | 300 | MRT |

B 臂 tau/nu 由 pylbm demo `mu=5e-6, dx=1/64` 换算（sigma2=3mu/(la·rho·dx)=9.600e-04，nu=sigma2/3；dt=dx/la ⇒ t=0.5 对应 32 步），
参数由 OSS 代码决定，非本仓挑选。

## ④ 对照表（ref vs got + 勾叉）

| 臂 | steps | RMSE_u | RMSE_v | max|du| | max|dv| | 质量漂移 | <15% | <3% |
|---|---|---|---|---|---|---|---|---|
| nx=33 | 4000 | 0.0672 | 0.0223 | 0.1026 | 0.0463 | -3.20% | OK | FAIL |
| nx=65 | 8000 | 0.0522 | 0.0125 | 0.0969 | 0.0230 | -0.78% | OK | FAIL |

逐点表（nx=65，ref=Ghia 1982）：

| 量 | 坐标 | ref | got | delta | |delta|<=15% |
|---|---|---|---|---|---|
| u | 1.0000 | +1.0000 | +1.0000 | +0.0000 | OK |
| u | 0.9766 | +0.8412 | +0.8941 | +0.0529 | OK |
| u | 0.9688 | +0.7887 | +0.8414 | +0.0527 | OK |
| u | 0.9609 | +0.7372 | +0.7882 | +0.0510 | OK |
| u | 0.9531 | +0.6872 | +0.7376 | +0.0505 | OK |
| u | 0.8516 | +0.2315 | +0.2579 | +0.0264 | OK |
| u | 0.7344 | +0.0033 | +0.0293 | +0.0260 | OK |
| u | 0.6172 | -0.1364 | -0.0980 | +0.0384 | OK |
| u | 0.5000 | -0.2058 | -0.1634 | +0.0424 | OK |
| u | 0.4531 | -0.2109 | -0.1706 | +0.0403 | OK |
| u | 0.2813 | -0.1566 | -0.1412 | +0.0154 | OK |
| u | 0.1719 | -0.1015 | -0.1238 | -0.0223 | OK |
| u | 0.1016 | -0.0643 | -0.1270 | -0.0627 | OK |
| u | 0.0703 | -0.0478 | -0.1315 | -0.0838 | OK |
| u | 0.0625 | -0.0419 | -0.1328 | -0.0908 | OK |
| u | 0.0547 | -0.0372 | -0.1340 | -0.0969 | OK |
| u | 0.0000 | +0.0000 | +0.0000 | +0.0000 | OK |
| v | 0.0000 | +0.0000 | +0.0000 | +0.0000 | OK |
| v | 0.0625 | +0.0923 | +0.0901 | -0.0022 | OK |
| v | 0.0703 | +0.1009 | +0.1017 | +0.0008 | OK |
| v | 0.0781 | +0.1089 | +0.1121 | +0.0032 | OK |
| v | 0.0938 | +0.1232 | +0.1313 | +0.0081 | OK |
| v | 0.1563 | +0.1608 | +0.1814 | +0.0206 | OK |
| v | 0.2266 | +0.1751 | +0.1981 | +0.0230 | OK |
| v | 0.2344 | +0.1753 | +0.1980 | +0.0227 | OK |
| v | 0.5000 | +0.0545 | +0.0571 | +0.0025 | OK |
| v | 0.8047 | -0.2453 | -0.2679 | -0.0225 | OK |
| v | 0.8594 | -0.2245 | -0.2445 | -0.0200 | OK |
| v | 0.9063 | -0.1691 | -0.1818 | -0.0127 | OK |
| v | 0.9453 | -0.1031 | -0.1068 | -0.0037 | OK |
| v | 0.9531 | -0.0886 | -0.0901 | -0.0014 | OK |
| v | 0.9609 | -0.0739 | -0.0732 | +0.0007 | OK |
| v | 0.9688 | -0.0591 | -0.0559 | +0.0032 | OK |
| v | 1.0000 | +0.0000 | +0.0000 | +0.0000 | OK |

误差集中在底壁附近（y≈0.05–0.10，Δu≈−0.09，底角二次涡欠分辨）；内区 y>=0.28 逐点 |Δu|<=0.042。

B 臂（ref = pylbm h5，同参数）：

| 场 | ref 范围 | max|d| | rel L2 | 上游容差 | 15% 判据 |
|---|---|---|---|---|---|
| rho | [+0.99568, +1.00639] | 1.1814e+00 | 0.0342 | FAIL | OK |
| qx | [-0.01422, +0.02573] | 4.6182e-01 | 6.709 | FAIL | FAIL |
| qy | [-0.01364, +0.01427] | 4.8953e-01 | 32.78 | FAIL | FAIL |

**B 臂 = honest FAIL**：同参数 32 步内本仓解已进入不稳定分支（max|dρ|=1.18，max|dqx|=0.46，而 OSS 场幅 0.026），64 步 NaN。
属数值方案稳定性边界差异（pylbm Geier 矩 MRT+正则化，tau=0.50096、格点 Re≈1e4），不作为精度结论。

C 臂（回归基线上游语义，atol=1e-7 / rtol=1e-14）：

| 字段 | max|d| | 判定 |
|---|---|---|
| ux_centerline_normalised | 0.0e+00 | OK |
| uy_centerline_normalised | 0.0e+00 | OK |
| rho_field | 0.0e+00 | OK |
| mass | 0.0e+00 | OK |
| mass_drift_relative | 0.0e+00 | OK |
| max_speed | 0.0e+00 | OK |
| ghia_rmse_u | 0.0e+00 | OK |
| ghia_rmse_v | 0.0e+00 | OK |

整块 max|d|=0（逐位复现）⇒ 该机制在 TensorLBM 上构成可复现回归门。

## ⑤ 收敛性/参数敏感性（>=2 点）

1. 网格：nx=33→65（Re=100, u=0.1）RMSE_u 0.0672→0.0522；质量漂移 -3.20%→-0.78%。
2. 瞬态/稳定性：pylbm 参数下 32 步 rel-L2(qx)=6.71 → 64 步 NaN（失败点落在 32–64 步）。
3. 分量：nx=65 的 RMSE_v(1.25%) 远小于 RMSE_u(5.22%)（涡心横向位置比主涡强度更易收敛）。

## ⑥ 失败模式、工件与生态摩擦

- F1 底壁二次涡欠分辨：Δu≈−0.09；nx 33→65 仅把 RMSE_u 从 6.72% 降到 5.22%，进 3% 需 nx>=129 或二阶壁面格式。
- F2 tau→0.5 的高格点 Re 下本仓 MRT 发散而 OSS 存活：稳定性边界更窄，建议引入正则化/Geier 矩。
- F3 Zou–He 动壁质量泄漏：nx=33/4000 步 -3.20%、nx=65/8000 步 -0.78%；非发散型但需长时监控（基线段含 mass 漂移字段）。
- F4 工具链摩擦：本镜像 torch<->numpy 桥损坏（`Tensor.numpy()` -> `Numpy is not available`），已在 `baseline_io.to_numpy()` 加 `tolist()` 回退；
  生产 `ghia_metric_report` 在 nx=65 上 segfault（core dump），驱动改为自行计分（独立复算），生产钩子被 stub 并计入摩擦。
- F5 返修上下文：m18235 审计的 `TensorLBM/src/tensorlbm/verification/lid_driven_cavity/solver.py` 是**仓内嵌套过期快照**（m17533 误提交），
  其中 Zou–He `f[3]=f4`、`f7/f8` 同号 ±(1/6)ρu 确为缺陷但不在生产路径；本单在 `src/tensorlbm/lid_driven_cavity.py`（`f4=f2`、±0.5ρu_lid）上作业。

工件：`src/tensorlbm/baseline_io.py`、`src/tensorlbm/verification/baseline_reference.py`、`src/tensorlbm/lid_driven_cavity.py`、
`tests/test_baseline_reference.py`、`tests/test_cavity_baseline.py`、`tests/reference/cavity_re100_nx33_u0.1_s300.npz`、
`artifacts/*.json`、`artifacts/figures/fig1..3_*.png`、`predeclared.json`。
复现：`python xval/XVAL-LBM-PYTEST-BASELINE-REFERENCE/run_xval.py --register-baseline`；
`python -m pytest tests/test_baseline_reference.py tests/test_cavity_baseline.py`。

## ⑦ 误差判定（是否 <3%？）

| 关键量 | 基准 | 误差 | <3% | <15% |
|---|---|---|---|---|
| u 中心线 RMSE（nx=65） | Ghia 1982 | 5.22% | FAIL | OK |
| v 中心线 RMSE（nx=65） | Ghia 1982 | 1.25% | OK | OK |
| 逐点 max|du|（nx=65） | Ghia 1982 | 9.69% | FAIL | OK |
| qx/qy 场（pylbm 参数 32 步） | pylbm h5 | rel-L2 6.71 / 32.8 | FAIL | FAIL |
| 回归基线复现（C 臂） | 签入基线 | 0（逐位） | OK | OK |

**明确回答：否，未全部 <3%。** u 剖面 5.22% 与近壁逐点最大 9.69% > 3%（均在 15% 带内）；v 剖面 1.25% < 3%；
B 臂不达标且超 15%（发散），如实判 FAIL，未粉饰。
