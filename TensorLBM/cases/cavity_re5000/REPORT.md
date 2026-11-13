# CASE-CAVITY-RE5000 · 顶盖方腔 Re = 5000 (D2Q9)

- 案例：`TensorLBM/cases/cavity_re5000/`，入口 `run.py`，图 `figs/`，实跑工件 `result.json` + `runs/*.json`
- 上游：mission 15543 交付曾被判「参考数据造假 + 工件非入口产出」（`docs/audit_m15543.md`），PR#1985 被 owner 关闭；本单在其之上**重做**并修掉两处阻塞性缺陷。
- 结论速览：**求解器边界条件回归已修复并通过回归测试（34 passed）**；**伪造的 `GHIA_RE5000` 已删除**，改为带 provenance 的外部参考注入；**Re=5000 / n=129 的定常解未取得 —— honest FAIL**（详见 ④⑥）。

## ① 案例介绍

方腔 `L×L`，下/左/右壁无滑移，顶盖以 `U_lid = 0.1`（格子单位）沿 +x 拖动，`Re = U_lid·L/ν = 5000`，`L = n-1`。

| 参数 | 值 | 说明 |
|---|---|---|
| 格子 | D2Q9 | `n = 129` → `τ = 0.50768` |
| Re | 5000 | `ν = 0.1·128/5000 = 2.56e-3` |
| U_lid | 0.1 | Ma ≈ 0.17 |
| 碰撞 | TRT（λ=3/16，`τ⁻ = 24.91`）/ RLBM / BGK 可选 | 见 ⑤ |
| 边界 | 半程链接反弹 + Ladd 顶盖动量项 | 本单重写 |
| 参考 | Ghia et al. (1982) JCP 48:387-411 Table III/IV | **本环境离线不可得** → WITHHELD |

## ② 怎么做的

1. **定位回归**：main 上 `tests/test_lid_driven_cavity.py::test_solver_step_preserves_finite` 红。实测 `CavitySolver` 在 `Re=100, n=33` 第 38 步出现 NaN（`Re=1000/5000` 第 27/26 步），而 m17533 之前的版本 200 步稳定 → 确认 m17533 重写 `step()` 时引入回归。
2. **重写边界条件**（`verification/lid_driven_cavity/solver.py`）：
   - 壁面半程**链接**反弹：对每个流体节点、每个"邻居是壁"的方向 `i`，把碰撞后 `f_i` 反射为该节点的 `f_opp(i)`（旧实现用的是壁节点上一时步的旧值，信息落后一步）；
   - 顶盖动量修正 `f_opp = f_i + 6·w_opp·ρ·(c_opp·u_w)`（Ladd 1994）；
   - 壁节点置为壁面平衡态（ρ=1，u=u_w），使输出剖面的壁面值是物理值而非周期回卷垃圾。
3. **修复正则化碰撞内核**：`f ← f − ω·f^neq,reg` 等价于保留全部非流体力学（ghost）模，等于没做正则化；改为标准形式 `f ← f^eq + (1−ω)·f^neq,reg`（Latt & Chopard 2006）。
4. **新增 TRT 碰撞与顶盖速度爬升**：`magic = 3/16`（`τ⁻=0.5+λ/(τ−0.5)`）；`ramp_steps` 线性爬升（脉冲启动在 Re=5000 会在 ~1e3 步内把 ρ 打到 [0.64,1.20]）。
5. **删除伪造参考**：移除 `GHIA_RE5000`、`AVAILABLE_RE` 回退为 `(100,400,1000)`，`get_ghia_reference(5000)` 抛 `KeyError`；新增 `load_reference_json()`（校验无量纲、坐标单调、长度一致，sha256 由 runner 记入工件）+ `LITERATURE_ANCHORS` 点锚。runner 无 `--ref-json` 时 `run_status="withheld"`（exit 3），**永不假装通过**。
6. 工具链：PyTorch 2.7.1 CPU float64，`run.py --collision trt --ramp-steps 2000`，`make_figures.py` 作图（数字全部来自工件，无手抄）。

## ③ 计算结果（程序化注入，全部来自工件）

| 用例 | n | 步数 | 最终残差 max\|Δu\| | u_min (y) | v_max (x) | 状态 |
|---|---|---|---|---|---|---|
| Re=100 | 65 | 60 000 | 2.47e-07 | **−0.20655 (0.469)** | 0.24439 (0.188) | ok（已定常） |
| Re=1000 | 65 | 100 000 | 5.39e-04 | −0.09336 (0.484) | 0.14723 (0.016) | ok（**物理不成立**，见 ⑥） |
| Re=5000 | 129 | ≤10 000 | 3.08e-02 → NaN | — | — | **diverged**（`result.json` `run_status=failed`） |

Re=5000 诊断（`runs/re5000_n129_trt.log` 与步进探针）：`k=500` 时 `ρ∈[0.9988,1.0126]`，`k=1000` 时 `ρ∈[0.641,1.204]`、`max|u|=0.60`，爆点位于顶盖下方 1–2 格（(126,65)–(127,97)）。BGK（第 500 步）、RLBM（≈5 000 步）、TRT（≈10 000 步）、加/不加爬升均发散。

## ④ 结果图

- `figs/centerline_profiles.png`：Re=100 / Re=1000, n=65 的中心线剖面 u(y)、v(x)（`runs/validation_payload.json` 程序化读取）。
- `figs/convergence_history.png`：收敛史（每监控段 max|Δu|，对数轴）。
- **未给场量云图**：`result.json` 只存中心线剖面，未保存全场快照（避免工件膨胀）；云图需 `run.py` 加 `--save-fields` 后重跑，本单未做，如实声明。
- **未给 Ghia 曲线叠加**：Table III/IV（Re=5000）离线不可得，伪造表已删除；`make_figures.py --ref-json` 在拿到带 sha256 的转录后可叠加。

## ⑤ 注意点（踩坑与局限）

1. 壁节点"信息落后一步"的写法在 `Re=100` 也会在 ~40 步内爆掉，症状是角落 f 值指数增长 —— 修复后 Re=100 跑 6 万步残差 2.5e-7。
2. 脉冲启动（无爬升）在 Re=5000 直接把 ρ 打出 [0.6,1.4]；爬升 2000 步后 5 000 步内稳定，但 10 000 步附近仍失稳 → 爬升必要但**不充分**。
3. `τ = 0.5077`（n=129）已贴近 BGK 极限；n=65 的 Re=5000（`τ=0.5038`）无论 BGK/RLBM/TRT 均在 ≤1 000 步内发散 —— 细网格是硬约束。
4. 旧 `run.py` 有两处硬缺陷：模块级缺 `import torch`（`run_single` 里 `torch.max` → NameError）、`solver.solid` 属性在 m17533 后已不存在 → 入口脚本当时**不可运行**。
5. 数据纪律：`GHIA_RE100/400/1000` 仍是 `UNVERIFIED_OFFLINE`（audit F-A：Re=100 u(0.5,0.5) 本仓 −0.15612 vs 文献 −0.20581，差 24%），本单只在结构/插值测试中使用，**不作验收闸门**。

## ⑥ 误差判定（关键量 vs 基准，逐项回答"是否 <3%"）

| 量 | 本算例 | 基准 | 来源 | 误差 | <3%? |
|---|---|---|---|---|---|
| Re=100 `u_min` (y) | −0.20655 (0.469) | −0.21090 (0.4531) | Ghia 1982 Table I（本仓文档引用值） | **2.06%** | ✅ 是 |
| Re=1000 `u_min` (y) | −0.09336 (0.484) | −0.38289 (0.1719) | 同上 | 75.6% | ❌ 否（该工件无效） |
| Re=5000 `u_min` | 无（发散，`result.json` 无数值） | — | Table III/IV 离线不可得 | — | 无法判定（WITHHELD） |

- Re=100 的 2.06% 是本单**唯一可宣称的精度证据**，且只覆盖 n=65、Re=100。
- Re=1000 的 100 000 步工件残差仅 5.4e-4 但剖面与文献锚相差 75.6%，且 `v_max` 落在 x≈0.016（贴壁）→ 判该工件无效（可能受修复前内核与贴壁污染影响），**不作为通过证据**，列入待修项。
- Re=5000：无任何可判定误差；`result.json` 记 `run_status=failed`，平台锚点应为 WITHHELD/FAIL，不得记 pass。

## 待修项（如实申报，未完成）

1. Re=5000 / n=129 定常解：需更强稳定化（Bouzidi 二阶曲面壁、entropic/MRT 限幅、或 n≥257 `τ≈0.515`）后重跑 ≥1e5 步；本单只做了 BGK/RLBM/TRT + 爬升三种尝试。
2. Re=1000 剖面异常需复跑（当前内核 + TRT + 爬升，n=129）。
3. Ghia Table III/IV 的 provenance 转录（页码 + sha256）仍未入库；拿到即可用 `--ref-json` 打开点对点闸门。
4. `make_figures.py` 末尾 `relative_to` 在相对 `--out` 下会抛 ValueError（图已正常落盘），未修。