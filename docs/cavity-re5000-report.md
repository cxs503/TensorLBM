# CASE-CAVITY-RE5000 · 顶盖方腔 Re=5000（TensorLBM）— 第1轮 honest 状态报告

mission 15370 · 分支 agent/m15370 · repo TensorLBM

## ① 问题与参考出处
算例：二维顶盖驱动方腔，L=1、顶盖 U_lid 拖动，Re = U L / ν = 5000，直至定常。
参考：U. Ghia, K. N. Ghia, C. T. Shin, *J. Comput. Phys.* **48** (1982) 387–411。
- Table III = u/U 沿竖直中线 x/L = 0.5（Re=3200/5000/7500/10000 列）
- Table IV  = v/U 沿水平中线 y/L = 0.5（同上）
- Re = 100/400/1000 的列在 Table I/II（生态内已有数字化副本，用于本单的"同码路径自检"）

**参考数据准入（零手抄）状态：BLOCKED（如实申报）**
本单要求目标数值从原文/手册机器取得，不手抄。开工三查与取证结果：
1. 平台知识引擎 `/api/v1/tensor/knowledge/search` → `engine_offline`（127.0.0.1:8021 connection refused）；
2. 生态内 Ghia 数字化副本仅覆盖 Table I/II（Re=100/400/1000）：
   `TensorLBM/src/tensorlbm/verification/lid_driven_cavity/reference_data.py`、
   `src/tensorlbm/lid_driven_cavity.py`、TensorFVM `lid_cavity_reference.py`（三处均已核对，无 Table III/IV）；全盘 `find -iname '*ghia*'` 无 PDF/数据文件；
3. 网络：`raw.githubusercontent.com` 000、`api.github.com` 200 但代码检索需鉴权且检索端点被沙箱安全策略拦截；`arxiv.org` 可达但 1982 JCP 原文非开放；
4. 负知识台账 keywords='TensorLBM' 9 条、`tensor_case_pool` 中 CAVITY-RE5000 ref_source 仅写 "Ghia et al. 1982"，无载荷。

结论：本环境下**无法**获得 Table III/IV 的 Re=5000 数值 → ④ 对照表判 **WITHHELD**，不制造任何"通过"。

## ② 方法与实现（复用现有能力，不另起炉灶）
`benchmarks/verified/cavity/re5000/run.py`，复用 TensorLBM 已验证的 Re=100/400/1000 解法链：
- 格子 D2Q9；碰撞 `tensorlbm.solver.collide_rlbm`（Latt & Chopard 2006 正则化 BGK）——
  Re=5000 处 τ≈0.52，re1000 案例已记录 BGK/MRT 在该黏性区间硬性发散（`benchmarks/verified/cavity/re1000/result.json` key_fix）；
- 边界 V3：三静止壁 pre-streaming 半程反弹 + 顶盖 `zou_he_moving_lid`（Zou/He 速度边界）；
- 等 Re 配方 τ = 3 U_lid L / Re + 0.5，L = n（与 re1000 一致）；
- 输出中线剖面（全分辨率）、涡心/角涡、ρ 界、残差历史写入 `result.json`，report 数字可程序化重derive；
- 参考表**不硬编码**：`--ref-json` 读入带 provenance（table/page/sha256）的 JSON；缺文件即判 WITHHELD；
- 导入守卫：断言 `tensorlbm.__file__` 落在本 worktree 的 `src/`（见摩擦 3）。

## ③ 参数表（无量纲数完整）
| 量 | 值 |
|---|---|
| Re = U L/ν | 5000 |
| U_lid（格子单位） | 0.1 |
| L（格子） | n（384 / 512） |
| ν = (τ−1/2)/3 | τ(384)=0.52300 → ν=7.667e-3；τ(512)=0.53070 → ν=1.023e-2 |
| 声速马赫 Ma = U/cs | 0.1/0.5774 = 0.1732（压缩性误差 O(Ma²)≈3%） |
| 网格 | 384²、512²（float64） |
| 步数上限 / 残差终止 | 400 000 / max|Δu| < 2e-7 |
| 判据（先于运行预登记） | 逐点 |ref−got| ≤ 8% U_lid（Table III/IV 行）；384→512 的 u 增量须小于最细网格对参考的误差；Re=1000 同码路径自检须先过 8% |

## ④ 对照表（ref vs got + 误差打勾叉）
**WITHHELD** — 无 Table III/IV 机器可读副本，无 got 定常解（见⑥）。
不列任何伪造数字。可立即补做的接口已就位：给出 `--ref-json` 即可输出逐行
`[coord, ref, got, err_pct, pass?]` 与 `max_abs_err_pct`。

## ⑤ 收敛性 / 参数敏感性（设计 + 实测）
- 设计两点敏感性：(i) 网格对 384²→512²（`grid_convergence.max_abs_delta_pct` 必须 < 最细网格 ref 误差，否则判 WITHHELD 而非 pass）；(ii) U_lid∈{0.06,0.1} 的压缩性项，O(Ma²) 分别 ≈1% 与 3%，用于界定 ref 不可得时的系统偏差来源。
- 实测（本地、非本算例）：48²、Re=400、1 000 步 → 残差 9.13e-3，跑通链路；128²、Re=5000（τ=0.5077）、20 000 步本地 120s 工具上限内未产出结果（低 τ 稳定性未验证）。
- 集群作业 cj1092（384²+512²，400k 步，float64，32 核）递交后 4 分钟仍 **queued**（递交时排队 298），至本报告尚未取回 → 定常解与网格收敛证据 **PENDING**。

## ⑥ 失败模式与工件
honest-FAIL，两段：
- `fail_stage = ref_admission`：Ghia Table III/IV（Re=5000）本环境不可机器取得（证据见①），容差判据无法执行 → 不宣称 <8%。
- `fail_stage = compute_pending`：cj1092 排队未回；本单预算内未取回定常解，grid_convergence 未产数。
- 另发现真实缺陷并已修（附测试）：`get_ghia_reference(5000)` 原本**静默回落**到最近 Re=1000 表（会把 Re=5000 算例判成假通过），docstring 亦虚报覆盖到 Re=10000。已改为未表化 Re 抛 KeyError + docstring 校正，回归测试 `test_get_ghia_reference_rejects_untabulated_re` 通过（2 passed, 18 deselected, 9.18s）。

工件（worktree 内）：
- `benchmarks/verified/cavity/re5000/run.py`（算例+准入策略+自检）
- `benchmarks/verified/cavity/re5000/smoke_128.json`（本地链路 smoke，**非**算例结果；若为空见⑤）
- `TensorLBM/src/tensorlbm/verification/lid_driven_cavity/reference_data.py`（缺陷修复）
- `TensorLBM/tests/test_lid_driven_cavity.py`（回归测试）
- 本报告

repro_cmd:
```
# 1) 同码路径自检（Re=1000 vs Ghia Table I/II，必须先过）
python benchmarks/verified/cavity/re5000/run.py --n 128 --re 1000 --u-lid 0.06 --steps 120000 --selfcheck
# 2) 正式算例（384²/512²）
python benchmarks/verified/cavity/re5000/run.py --n 384 512 --steps 400000 --threads 32 --dtype float64
# 3) 有参考表时打分（准入后）
python benchmarks/verified/cavity/re5000/run.py --n 384 512 --steps 400000 --ref-json docs/ghia-table-iii-iv-re5000.json
```

## 生态摩擦（如实记）
1. 平台知识引擎离线（:8021 拒绝连接），三家查询只剩本地 grep；`knowledge_base` 仅 4 条 TensorLBM capability、无腔流 Re≥3200 参考数据。
2. 生态缺 Ghia Table III/IV 数字化语料 → 所有 Re≥3200 的腔流算例（CAVITY-RE5000 及后续）在离线环境都只能 WITHHELD；建议立专项：以带 page/sha256 的 `dns-reference.json` 先例（TensorFVM cylinder3900）落一份 Table III/IV 语料。
3. editable 安装劫持：`/usr/lib/python3.11/site-packages/__editable__.tensorlbm-0.3.0.pth` 指向 *另一个 worktree*（work/m13963/src），`import tensorlbm` 会加载别人的树；本算例以 sys.path 前置 + 断言守卫防御。
4. 工具面：单次 shell 上限 120s，`pytest TensorLBM/tests/test_lid_driven_cavity.py` 全量跑不完（本轮只跑 `-k ghia_reference`）；集群递交排队 298，单节点容量与算例池"以百计铺开"的需求不匹配。
