# Board audit m17878 — m17757 `XVAL-LBM-FILL-GRADIENT-NORMAL` (TensorLBM)

审计对象：mission **m17757**（TensorLBM，branch `agent/m17757`，commit `8f8e48ff`，状态 succeeded）
锚点：`XVAL-LBM-FILL-GRADIENT-NORMAL` — TensorLBM 自由面 fill-gradient 界面法向 vs 开源 Maarten-vd-Sande/lbm
本单：**m17878**（交差核验单，非重做）· 判定级别 **L2** · `verdict_by=board-audit`

---

## 0. 结论先行

| 项 | 结果 |
|---|---|
| 三件套（run_artifacts / ref vs got 数值 / repro_cmd） | **齐备**：`xval/fill_gradient_normal/{results.json, REPORT.md, figures/}` 存在；`python3 xval/fill_gradient_normal/run_xval.py` 审计方重跑 **bit-identical**（10/10 案例、0 个 KPI 偏差 >1e-12） |
| 声明的主判据（A1 vs A2 = 0.00%） | **退化**：A2 ≡ A1·\|∇fill\| 逐格成立（残差 5.6e-17），任意场必为 0，**不能**作为实现一致性证据 |
| 有信息量的判据（A1 vs 解析法向） | 10/10 在 15% 内，最差 **10.24%**（sphere N=32），cone 型锐界面轴偏置 5.10%/7.83%/8.40%，抹平 w=8 时 0.71% |
| tensor_benchmark_cases | 审计时 **0 行**（表内 65 行）→ 机器裁判缺位；本审计**补登记**一行并写明退化告警 |
| 集成 | **0 调用方**：被审计函数是死代码；可达路径 `free_surface_common.interface_normal_3d` 符号相反 |
| 判级 | **L2**（真代码输入→计算→输出、可复现、有解析真值）；**L3/L4 拒判** |

处置（adversarial-review disposition）：**approve with notes → confirmed-L2，不返修**；L4 若要成立需 R1–R4（见 §6）。

---

## 1. 取证范围与方法（可独立复现）

审计脚本：`docs/board_audit/m17878/verify_m17757.py`（六项检查，全部读运行工件，不手抄）

```bash
python3 docs/board_audit/m17878/verify_m17757.py      # -> artifacts/verify_m17757.json
python3 docs/board_audit/m17878/insert_verdict.py     # 幂等写 verdict + benchmark 行
python3 docs/board_audit/m17878/make_figures.py       # 报告图
```

| 检查 | 问题 | 方法 | 结果 |
|---|---|---|---|
| C1 REPRO | 交付入口能否重建 results.json | 重跑 `run_xval.main()`，逐 KPI 比对 | 10/10 案例 **0 偏差**，`all_pass=true` 复现 |
| C2 EQUIVALENCE | 交付的 A1 臂是否真是生产函数 | 导入 `src/tensorlbm/free_surface_lbm.py:642`，同输入比数值 | 3D tube / 2D slice 最大差 **0.0**；符号 liquid→gas 正确 |
| C3 DEGENERACY | 主判据是否可判别 | 3 个任意场（均匀随机、0/1 随机、平滑正弦）跑 A1 vs A2 | dir_err = −1.6e-17 / 3.8e-17 / −1.7e-17，**恒 0** |
| C4 INTEGRATION | 被审计函数是否被调用 | `grep -rn _compute_interface_normal(` in `src/ tests/` | **0 调用点**（仅定义）；`interface_normal_3d` 2 个消费者 |
| C5 BENCHMARK | 锚点是否登记 | `SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug=...` | **0 行**（表 65 行，TensorLBM 其他锚点 61–64/69/70 已登记） |
| C6 LIVE PATH | 死函数 vs 活函数关系 | 同 φ 场对比 | 符号**相反**（−∇fill vs +∇φ）；KPI 用 \|n·n\|，**符号不可见** |

独立性声明（limitations）：
- 审计方无网络：未运行真实 Maarten-vd-Sande/lbm 源码，故「外部锚」的可执行侧未复核（与交付同一限制）。
- 我的重跑在同一树内、同一 torch 2.7.1 / float64 / CPU，冷启动一次；未换机器复跑，未做逐位跨平台验证。
- C2 的等价性是逐位相等（0.0），依赖同一 torch 版本；跨 torch 版本不保证。

---

## 2. 基准裁判逐项（`tensor_benchmark_cases` 为唯一裁判）

审计时该锚点**无登记行**，故「报告自称通过」当时无机器裁判可对。审计方按实测定值补登记
（`insert_verdict.py::upsert_benchmark`），metric 字段显式写明退化告警，**不得**被读作 L4 背书：

| KPI | ref 基准 | got | 误差 | 15% 线 |
|---|---|---|---|---|
| circle-sharp N=64/128/256 A1 vs 解析 | 径向解析法向 | 4.48 / 5.34 / 5.10 % | 同左 | ✅ |
| plane-sharp θ=30° N=256 A1 vs 解析 | 常向量 | 7.83 % | 同左 | ✅ |
| sphere-sharp N=32/64/128 A1 vs 解析 | 径向 | 10.24 / 8.55 / 8.40 % | 同左 | ✅（最差 10.24%） |
| circle-smeared w=2/4/8 N=256 A1 vs 解析 | 径向 | 1.75 / 1.18 / 0.71 % | 同左 | ✅ |
| 全部 7 案例 A1 vs A2（交付 headline） | OSS 公式移植 | 0.00 % | — | **不可计分（恒等）** |
| A3/A2 幅值比（GAS=0.5 bug 复现） | 0.5 | 0.500 | 0.000 | ✅ |

结论：**有信息量的判据全部 <15%，交付的「通过」在实质误差意义上成立；但其 headline 数字（0.00%）不承载证据。**

---

## 3. 分级判定（L0–L5）

| 级 | 判据 | 本单 |
|---|---|---|
| L1 数学/单元 | tests 非空 | ✅ 有解析真值对照（A4）+ 审计方独立重跑 |
| **L2 独立功能** | impl_files 非占位 + entry + run_artifacts | ✅ **达到**：真代码（构造 fill 场→四臂法向→误差统计），工件 results.json 可位级复现 |
| L3 系统集成 | L2 + integration_callers 非空 | ❌ 被审计函数 **0 调用方**；活路径符号相反且未被本 XVAL 覆盖 |
| L4 基准验证 | L3 + benchmark 数值对照 + repro_cmd | ❌ L3 不满足；且主判据退化、外部锚为文档移植 |
| L5 科研工程 | L4 + external_anchor | ❌ 同上 |

**判定：L2。** 级别=功能实质，非报告篇幅；交付报告 121 行、图 3 张、差异清单 7 项齐备，不改变级别。

---

## 4. 严重度排序的发现（含复现）

**F1 · blocking-for-L4 · 主判据是代数恒等式**
`A2 = −0.5·central_diff(fill)`，`A1 = −∇fill/|∇fill|` ⇒ `A2 ≡ A1·|∇fill|` 逐格成立，共模板共场 ⇒ 方向差恒 0。
复现：`verify_m17757.py::c3_degeneracy`（任意场 |dir_err| ≤ 3.8e-17，恒等残差 5.6e-17）。
含义：「与 OSS 方向 100% 一致」不构成证据；真正被测的是 A1 vs 解析（5–10%）。

**F2 · blocking-for-L3 · 被审计函数是死代码，活路径符号相反**
`grep -rn "_compute_interface_normal(" src tests` → 仅 `free_surface_lbm.py:642` 定义，**0 调用点**。
活路径：`free_surface_common.interface_normal_3d`（+∇φ/|∇φ|，返回幅值），消费者 `interface_compression_3d:182`、`mean_curvature_3d:255`。
复现：C4/C6；step 界面上 `n_x(x=7) = +1.0`（死函数）vs `−1.0`（活函数）。
附加：KPI 用 `1−|n·n|`，**对符号不敏感**，因此交付差异清单 #4「符号约定一致」这一轴并未被测量。

**F3 · blocking-for-L4 · 锚点未登记**
审计时 `tensor_benchmark_cases` 0 行；已由本审计补登记（含退化告警），`status=passed` 仅针对解析锚定 KPI。

**F4 · medium · 外部锚是文档移植，非 OSS 运行**
交付自述「无网络 checkout」，A2 公式逐字取自 `oss_solutions id=85` method 文本。属「文档→代码」，不是「代码→代码」交叉实现；跨实现一致性在严格意义上 **WITHHELD**。

**F5 · low · 等价性声明未随交付留证**
REPORT ② 声称「与真实函数 max diff = 0.0」，交付内无该脚本/测试。审计方独立复核为 **真**（0.0），但交付当时属未附证据的声明。

**F6 · low · 无形状守卫**
`_compute_interface_normal` 假定 `(nz,ny,nx)`；传入 `(1,nz,ny,nx)` 时 dims 偏移，**静默返回全零法向**（C2 `batched_input_returns_all_zero=true`）。`flags` 形参未使用。

**F7 · low · 无回归测试**
`grep tests/ fill_gradient|interface_normal` → 0 命中；唯一自检是脚本内 `all_pass` 标志。

---

## 5. 未执行的检查（unknown / not_run，不得读作通过）

1. 真实 OSS 仓库（Maarten-vd-Sande/lbm）源码运行 —— 无网络，未跑。
2. 生产自由面算例（Körner 质量再分配 / dam-break）中法向误差的端到端影响 —— 未跑（该函数无调用方，端到端无对象）。
3. 跨平台/跨 torch 版本位级复现 —— 未跑。
4. 抹平场的 A2 臂（交付仅对锐场算 A1 vs A2） —— 交付未提供，审计未补。
5. 真实 3D 自由面场（fill 来自 RUN 的 mass/rho）而非解析构造场 —— 未跑。

---

## 6. 返修建议（不阻断 confirmed-L2）

- **R1** 删掉 0.00% headline，主判据改为 A1 vs A4（解析）+ A2 幅值比；如需交叉实现，用可判别的量（角度分布、幅值、符号三轴分开报）。
- **R2** 要么把法向接进活 VOF 路径（并与 `interface_normal_3d` 统一符号），要么在报告中明确标注「死代码模板验证」。
- **R3** 执行真实 OSS 内核（或签核的参考实现），把外部锚升为 code-to-code。
- **R4** 把 C2 等价性检查落成 pytest；先登记 benchmark 行再声称通过。

---

## 7. 判决

**confirmed-L2（消账，不返修）**。交付物真实、可跑、可位级复现，解析锚定误差全部 <15%（最差 10.24%）；
但 L3（零积分、死代码、活路径符号相反）与 L4（锚点未登记 + 主判据退化 + 外部锚为文档移植）明确拒判。
机器门（`maturity_gate.judge`）对本次 evidence 的本地复算：`(2, [])` —— 与声明级别一致。

产物清单：
`docs/board_audit/m17878/{verify_m17757.py, insert_verdict.py, make_figures.py, AUDIT_REPORT.md,
artifacts/verify_m17757.json, figures/fig1_kpi_informative_vs_degenerate.png,
figures/fig2_sign_flip_and_degeneracy.png, refs/…(受审计原件快照+sha256)}`