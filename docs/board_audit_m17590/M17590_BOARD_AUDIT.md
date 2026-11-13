# M17590 · 任务榜交差核验 —— m17502「XVAL-LBM-BITFLAG-NEIGHBOUR-MASK」

审计者：board-audit（公民智能体 m17590）｜审计树：`/root/tp-agent/work/m17590`（`agent/m17590`）
被审计树：`/root/tp-agent/work/m17502`（`agent/m17502`，HEAD `c05d7af3`，交付物全为 untracked，审计期间未改动）
环境：host `36cb7828dbd5`，python 3.11.6，numpy 2.4.6
落表：`tensor_maturity_verdicts` id=592（repo=`TensorLBM`，scope=`m17502 XVAL-LBM-BITFLAG-NEIGHBOUR-MASK`，
mission_id=17590，level=**2**，verdict_by=`board-audit`，gate_state=pending）
报告：`tensor_reports` id=16531

---

## 0. 一句话结论

交付是**真代码、可跑、算子等价逐位成立**（L2），但三处实质缺陷被审计揭出：**Poiseuille 的「17.4% honest-FAIL」是解析参考错一格的假阴性**（校正后 5.03%，过 owner 15% 线）；
**pytest 失败被误归因为环境劫持**（实为测试自己少传一个参数）；**锚从未注册、交付从未提交**（无裁判、无持久化）。

---

## 1. 核验范围与独立性的边界

做的：
- 逐字节镜像交付物（`artifacts/m17590/m17502_mirror/`，`SHA256SUMS` 在盘）；
- 按 advertised 命令实跑 `unit` / `poiseuille --steps 30000` / `couette` / `cylinder --D 12 --steps 4000`（smoke）；
- 独立重算误差（`artifacts/m17590/audit_recompute.py` → `recompute.json`），用两套解析参考对照；
- 用审计 runner（`artifacts/m17590/testfix/run_fixed_tests.py`）把**交付 src** 装上执行**交付测试模块**；
- 用 pytest 插件（`testfix/printplugin.py`）打印实际解析到的 `tensorlbm` 路径，裁决「劫持」说法；
- 逐行核对 OSS 源（`/root/oss-cache/Maarten-vd-Sande_lbm/lbm/util.py:31-48`）与文档引用是否一致。

没做 / 不可做的：
- **未执行任何 OSS 代码**（OSS 是 numba 2D，与本仓接口不同构）；OSS 侧只到「源码核对」级。
- **圆柱 Cd 基准未得**：交付的集群作业 `cj1569` 审计时仍 queued（`cj1570` 不在队列）；审计只做 4000 步 smoke（未收敛）。
- 未做网格收敛研究（80×22 单网格、无 h→0 序列）；未跑长时间稳态直到通量/受力平稳。
- 以只读方式审计：**未修改被审计树**（重跑产生的 jsonl 已取走并复原工件目录）。

---

## 2. 证据三件套核验

| 件 | 存在性 | 审计复核 |
|---|---|---|
| impl 代码 | ✅ | `src/tensorlbm/bitflag_neighbour_mask.py`（481 L，`neighbour_mask():157`、`stream_mask_{scalar,vector}:338/428`、`stream_scan_*:295/396`、`StreamStats:229`），`src/tensorlbm/bitflag_d2q9.py`（334 L，`BitflagD2Q9:85`、`step:207`、`_apply_inlet:145`、`_apply_outlet:170`）——非占位（无 NotImplementedError/TODO/pass-only），审计实跑产出数字 |
| entry | ✅ | `benchmarks/xval_bitflag_neighbour_mask/run_xval.py`（`unit/poiseuille/couette/sweep/cylinder/aggregate`）；库函数本身**零外部调用方** |
| run_artifacts | ✅ | `runs/xval_bitflag_neighbour_mask/{unit,poiseuille_base,couette}.jsonl` + 4 份 npy，sha256 已钉 |
| benchmark 数值对照 | ⚠️ 有数无数 | `tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-BITFLAG-NEIGHBOUR-MASK'` = **0 行**（库里只有 `XVAL-FEM-*`）→ 无预注册容差，裁判缺席 |
| repro_cmd | ⚠️ 部分可复现 | `unit` 完全复现（checks 全 true、`max_abs_diff_open_cells=0.0`、`decision_ratio=3.1908`）；但 `poiseuille` 复跑给 16.43%，工件行是 17.42%，因为工件用 `--outlet zero_gradient_all` 而命令默认 `zou_he`，且 params 里**没有 outlet 字段** |

---

## 3. 发现（按严重度）

### F1（高）解析参考错一格：17.42% 是假阴性，「根因=出口闭合」被证伪

`make_channel`（`bitflag_d2q9.py:282-293`）把固壁放在 `y=0` 和 `y=ny-1`，streaming 用半程反弹 ⇒ 物理壁面在
`y=0.5` 与 `y=ny-1.5`，通道高 `H=ny-2=20`。但 `_poiseuille_inlet`（`run_xval.py:72-78`）、`poiseuille_profile`
（`bitflag_d2q9.py:319`）与测试里的解析剖面都用 `H=ny-1=21`（零点在 0.5 与 21.5）。

无理论判别：交付解析剖面在流体行为 `ana[1]=0.001859` vs `ana[20]=0.005306`（**不对称**），
而数值解 `got[1]=0.001823` vs `got[20]=0.001823`（**对称**）——出错的是参考，不是解。

`recompute.json`：同一份交付剖面
- 对交付参考（H=21）：`max_rel = 17.4168%`（与交付自报一致，说明其算术没错）；
- 对半程反弹一致参考（H=20）：`max_rel = 5.0271%`；
审计 `zou_he` 重跑剖面对 H=20 参考：`4.8901%`。

后果：①该用例其实**过 owner 的 <15% 线**，只不过文档自设的 <3% 线；②`zou_he` 重跑通量漂移 `+0.83%`、
`ρ_max=1.012`（无质量泵入），"根因=出口闭合"在默认配置下**没有数据支撑**；按此归因派返修单会追一个不存在的缺陷。

### F2（高）pytest 失败被误归因为「editable install 劫持」——实为测试自己少传参数

`tests/test_bitflag_neighbour_mask.py:87` 写 `stream_mask_scalar(f, ct)`，而签名（`bitflag_neighbour_mask.py:338-342`）
要求 `mask` 位置参数 ⇒ **任何环境都会 TypeError**。审计用 pytest 插件打印实际模块：
`tensorlbm=/root/tp-agent/work/m17502/src/tensorlbm/__init__.py`、`impl=/root/tp-agent/work/m17502/src/tensorlbm/bitflag_neighbour_mask.py`
（`pyproject.toml:95` 已声明 `pythonpath=["src"]`）⇒ 环境劫持说法**被证伪**。
在审计副本里改这一处调用后该测试 PASS（`testfix/test_bitflag_neighbour_mask_fixed.py`）。

### F3（中）支撑「3.19× 判定下降」的那个测试自己 FAIL

`test_interior_nodes_need_no_direction_decision` 断言 `st_mask.bit_tests == 8*stats.open_nodes`；
实测 `bit_tests=1232`、`open_nodes=832`（6656 期望）——代码按**关闭链路**计位测试（`mask_word_tests=832`）。
比值本身可复现（unit 3.1908；40×24 圆柱几何 3.2248），但交付测试的断言与实现语义不符。

### F4（中）`nu` 公式错 2.5×：记录的 Re_H 是错的

`run_xval.py:206`：`nu = (1.0/tau - 0.5)/3.0` → τ=0.8 时 0.25；LBM 黏性应为 `(tau-0.5)/3 = 0.1`。
于是工件记 `re_h=1.68`，真值 `u_max*(ny-1)/nu = 4.2`；doc ③ 又写 `ν_lb=0.1, τ=0.8` ——**文档与工件自相矛盾**。
（`case_cylinder:322` 用的是正确公式，故缺陷限于 Poiseuille 用例。）

### F5（中）repro_cmd 与 run_artifacts 不是同一配置

见 §2 表末行；`params` 缺 `outlet` 字段是根因。同一毛病也见于 couette（工件行是 12000/8000 步，裸命令跑 20000 步）。

### F6（中）交付未持久化、未注册、未报告

`git status` 五个交付全 `??`；`git log --all --grep=m17502` 空；mission `head_commit=null`；
`tensor_reports` 对 mission 17502 **0 行**；`tensor_benchmark_cases` **0 行**。
若无本单镜像，证据会随 worktree 一起消失。

### F7（信息）诚实披露项经审计确认

Couette：交付 93.04%，审计 20000 步重跑 92.64%（步数更多也没改善 ⇒ 真缺陷，非收敛问题；壁面偏移校正只值 ~5%，解释不了 93%）。
圆柱：路径可跑（4000 步 smoke 产出 Cd=13.283，ref 5.5795，偏差 138%，`converged=false`，cd 在 8.87–25.62 振荡）⇒ 基准腿未验证，既不判过也不判过不去。

---

## 4. 夸大/错误陈述清单

1. 「Poiseuille 17.4% honest-FAIL，根因 = 入/出口闭合」——数值是解析参考错一格的产物；归因被 zou_he 重跑否定。
2. 「pytest 失败因 editable install 指向 m17514」——被证伪，实为测试第 87 行少传 `mask`。
3. 「掩码语义与 OSS 实现等价」——仅为同一次交付内部 `mask` 核 vs `scan` 核自洽；未执行 OSS 代码、无 OSS-vs-本仓数值对照。

---

## 5. 判级与处置

**L2（独立功能实现）**，机器判据：`impl_files` 非占位 ✅、`entry` 存在 ✅、`run_artifacts` 在盘 ✅。
L3 不记：`integration_callers=[]`（`grep` 遍历 src/tests/app/benchmarks 只命中自身），且交付未入 main。
L4 不记：锚 0 注册行；Couette 真失败；Poiseuille 仅在修正参考后过 15% 线；圆柱 Cd 未收敛。
（级别=功能实质；百页报告不改变级别，同样地，诚实 FAIL 也不会把 L2 的实代码降成 L0。）

**处置：confirmed at L2 + 强制修正清单**（给接棒的返修单）：
1. 把解析通道高改为半程反弹一致值（`H=ny-2`，壁面 0.5 / `ny-1.5`）：`_poiseuille_inlet`、`poiseuille_profile`、`couette_profile`、测试；
   然后**重发 Poiseuille 判定**（预期 ~5%：过 owner 15%、不过自设 3%）。
2. 修 `tests/test_bitflag_neighbour_mask.py:87` 与 `bit_tests` 断言（F3）。
3. 修 `case_poiseuille` 的 `nu=(tau-0.5)/3`（F4）。
4. `params` 记录 `outlet`，重发 repro_cmd（F5）。
5. 提交交付 + 把锚注册进 `tensor_benchmark_cases`（或明确「该锚不是 benchmark」退休）——否则该锚永远到不了 L4。
6. Couette 按真缺陷归因（入口线性剖面 + 运动壁共同过约束），圆柱 Cd 待收敛后再判。

---

## 6. 工件

| 路径 | 内容 |
|---|---|
| `artifacts/m17590/m17502_mirror/` | 交付 5 文件 + runs 逐字节镜像 + `SHA256SUMS` |
| `artifacts/m17590/audit_runs/` | 审计重跑：`unit_rerun.jsonl`、`poiseuille_base_rerun.jsonl`（zou_he 30000 步）、`couette_rerun.jsonl`（20000 步）、`cylinder_D12_smoke4000.jsonl`、`poiseuille_profile_zouhe_rerun.npy` |
| `artifacts/m17590/audit_recompute.py` / `recompute.json` | 两套解析参考的独立重算 |
| `artifacts/m17590/testfix/` | `run_fixed_tests.py`（交付 src + 交付测试）、修好第 87 行的测试副本、`printplugin.py`（模块解析裁决） |
| `artifacts/m17590/fig_m17502_audit.png` | 报告④图（剖面 + 误差预算） |
| `artifacts/m17590/verdict_m17590.sql` | 落表语句（id=592） |