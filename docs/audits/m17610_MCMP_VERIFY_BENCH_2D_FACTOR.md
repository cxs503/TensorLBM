# m17610 · MCMP verify-bench：2D Laplace 因子 2× 缺陷 + 证据链断链修复

**仓库** TensorLBM ｜ **分支** `agent/m17610` ｜ **基线** `c05d7af3`（Merge #2000）
**上游线索** PR #1971（m15467，已裁定关闭）→ 实际内容随 merge #1822 `cf959e0d` 入 main；
本单是 m15891 审计发现 **F5「docs/MCMP_VERIFY_BENCH.md 的绝对 σ 列仍是旧 2D 因子产物」** 的收口。
**判级意图** L2 独立功能实现（真代码、真数据、可复现重算）；**不评** L4/L5（本轮无新物理，只是把既存测量的派生量按正确约定重算 + 修复断掉的证据链）。

---

## ① 案例介绍

被修对象是库级 MCMP 标定基准 `benchmarks/mcmp_verify_bench.py` 及其报告 `docs/MCMP_VERIFY_BENCH.md`（FEAT-LBM-VERIFY-BENCH，上游 mission 8636）。

物理：周期域内两组分 Shan-Chen 静止圆滴。**二维**圆滴（单位深度圆柱面）满足

```
Delta p = sigma / R          (dims = 2)
Delta p = 2 sigma / R        (dims = 3, 球滴)
```

即 `sigma = Delta p R / (dims - 1)`。测量协议（2026-11-09 预注册，未改）：平板共存密度 → 半径 [14, 19, 24]、n=96、12000 步；`Delta p` = 中心 4×4 箱 − 角点 4×4 箱；`R` = 四射线 `rho_1 = rho_av` 线性插值交点。判据：过原点单未知量拟合的相对残差 max < 5%。

原始测量数据：`artifacts/mcmp_verify_bench/results.json`（2026-11-09T15:21:16，TensorLBM 0.3.0 / torch 2.7.1，full 协议）。

## ② 怎么做的

1. **读代码定位**：`src/tensorlbm/mcmp_verification.py::measure_laplace` 与 `_fit_laplace_level` 在 2026-11-12（`cf959e0d`）才引入 `dims` 形参（`sigma = dp R/(dims-1)`，`x = (dims-1)/R`），默认 2；此前恒为 3D 球因子 `x = 2/R`、`sigma = dp R/2`。**库已修，但报告与 runner 的元数据未跟上**。
2. **查已提交工件**：`git ls-files artifacts/mcmp_verify_bench/` → **空**。该目录整批在合并时丢失（原提交 `9125f8a0`，分支 `agent/m8636`，never merged）。
3. **跑测试取证**：`pytest tests/benchmarks/test_mcmp_verify_bench.py -m "not slow"` → **1 failed, 21 passed**。失败项 `test_report_is_rendered_from_json` 断言（非 skip）`artifacts/mcmp_verify_bench/results.json` 存在 ⇒ **main 上「零手抄数字」的机器保证是坏的**（报告声称「重新渲染会失败」，但该测试在 main 上本来就红）。
4. **控对实验（不重跑求解器）**：σ 是 `dp·R` 的确定后处理，故用原 `results.json` 里逐例记录的 `dp/R/|u|max/steps/guardrails` 做因子 A/B：同一批原始量分别按 `x=1/R`（dims=2）与 `x=2/R`（旧/3D）重拟合，比较 σ 与残差。这是**配对同数据对照**，唯一变化量 = 因子，排除求解器与网格差异。
5. **重算与重渲染**：逐例 `sigma = dp·R`、逐层 `_fit_laplace_level(..., dims=2)` 重拟合、`Ca_sp = mu|u|max/sigma` 重算；`hero_fields.npz`、`oss_consumption_chain.json` 从 `9125f8a0` 原样恢复（与因子无关的原始场量/清单）；`render_report` 重渲染文档与 5 张图。
6. **回归锁定**：新增 3 个测试（见 ⑥）把因子钉死在测试里。

全量重跑（`--jobs 24`，集群 `cj1607`）已递交但因 395 个前置作业未派发；本轮**不依赖**它——σ 只是 `dp·R` 的后处理，原始量同一，重跑只会在 float 末位产生差异。

## ③ 计算结果（程序化，零手抄）

### 表 A｜配对同数据 A/B：因子只缩放 σ，残差逐位不变

| G₁₂ | 旧报告 σ₀（x=2/R） | 正确 σ₀（x=1/R） | 比值 | 旧残差 max | 新残差 max |
|---|---|---|---|---|---|
| −2.500 | 0.094689 | **0.189379** | 2.0000 | 1.6654 % | 1.6654 % |
| −2.750 | 0.102854 | **0.205707** | 2.0000 | 1.0790 % | 1.0790 % |
| −3.000 | 0.106412 | **0.212824** | 2.0000 | 1.3082 % | 1.3082 % |

**这就是缺陷活下来的原因**：`sigma_0 ∝ 因子` 且 `x ∝ 因子`，故 `sigma_0·x` 不变 ⇒ 残差判据（<5%，本算例唯一的 PASS 依据）**在结构上无法发现因子错误**。只有绝对 σ 及其派生量会错。

### 表 B｜逐例 σ 列（G₁₂ = −2.5）

| R 实测 | Δp | 旧「sigma = dp R/2」 | 正确 σ = Δp·R |
|---|---|---|---|
| 13.602 | 0.013790 | 0.093788 | **0.187575** |
| 18.667 | 0.010221 | 0.095402 | **0.190803** |
| 23.606 | 0.008156 | 0.096266 | **0.192533** |

### 表 C｜派生量 Ca_sp（n=48, G₁₂=−2.5）

| 量 | 旧 | 新 |
|---|---|---|
| σ | 0.088519 | 0.177038 |
| Ca_sp = μ|u|max/σ | 0.019399 | **0.009699** |

即旧报告把伪毛细数**高估 2×**，把绝对表面张力**低估 2×**（量级 1e-1 而非 5e-2）。

### 表 D｜缺陷 B：证据链断链

| 件 | 修复前（main @ c05d7af3） | 修复后 |
|---|---|---|
| `artifacts/mcmp_verify_bench/results.json` | 缺失 | 入库（含 `laplace_dims=2`、`derivation` 溯源） |
| `oss_consumption_chain.json` | 缺失 | 从 `9125f8a0` 恢复（9 条，id 1/5/20/21/44/281） |
| `hero_fields.npz` | 缺失 | 从 `9125f8a0` 恢复（原始场量，与因子无关） |
| 图 5 张 | 缺失，文档链接 `figures/figN.png` 相对 docs/ **死链** | 重生成 + 链接改 `../artifacts/mcmp_verify_bench/figures/...`（活链） |
| `test_report_is_rendered_from_json` | **FAILED** | PASS |

## ④ 结果图

![fig1](../artifacts/mcmp_verify_bench/figures/fig1_laplace.png)

*图 1｜Δp vs 1/R，三点连线为过原点拟合 `Delta p = sigma/R`（2D）。修正后斜率即 σ≈0.189/0.206/0.213；旧版同一图用的是 2/R 横轴，斜率减半。*

![fig2](../artifacts/mcmp_verify_bench/figures/fig2_spurious.png)

*图 2｜伪电流 |u|max 与 Ca_sp；Ca_sp 依赖 σ，修正后整体减半（表 C）。*

## ⑤ 注意点与局限

- **未重跑求解器**：本轮 σ 由 2026-11-09 记录的原始 `dp/R` 重算（配对同数据对照）。集群全量重跑 `cj1607` 未派发（前置 395 作业），故**未**取得独立重跑确认；若重跑完成，预期 σ₀ 与表 A 一致到 float 末位（同一代码路径、CPU 确定性）。
- **不是新物理**：修正的是报告约定错误（2D 数据套 3D 因子），物理内核（力格式/压力关系/半径测量）未动。
- **判据不变**：PASS/FAIL 结论未变（残差在因子变换下不变，表 A），所以「<5% PASS」的既往结论仍然成立；变的只是绝对量值。
- **σ 随曲率仍有 ~5% 趋势**（自由线诊断，见报告 §1），本轮不涉及，仍是既存限制；绝对 σ 仍无独立外部锚（m15891 F7）。
- 报告正文的因子文字由 `protocol.laplace_dims` 渲染，因此老 json（无该键）会回落到 dims=2 而把旧（减半）数字标成 `sigma/R` —— 这正是必须同时修数据与 runner 元数据、不能只改文档的原因。

## ⑥ 误差判定（显式回答）

| 关键量 | 参考 | 修复前误差 | 修复后误差 | <3 % ? |
|---|---|---|---|---|
| σ₀（G=−2.5/2.75/3.0） | 2D 解析约定 `Delta p = sigma/R`（单未知量过原点拟合同源模型） | **−100 %**（恰为真值一半） | 0（定义一致，比值 2.0000 精确） | **是**（结构性精确，非拟合） |
| Δp 半径无关性残差 max | σ₀/R | 1.6654 % | 1.6654 % | 是 |
| Ca_sp | μ|u|max/σ，σ=Δp·R | **+100 %** | 0 | 是 |
| 绝对 σ vs 实验/文献 | 无独立外部锚 | 不可评 | 不可评（如实声明） | **不作声明** |

结论：因子类量为**精确修正**（不是精度改善）；残差判定不变；绝对 σ 的**外部**正确性本轮仍未证实（无基准），不夸大为「σ 已验证」。

## ⑦ 本轮改动清单

| 文件 | 改动 |
|---|---|
| `benchmarks/mcmp_verify_bench.py` | 拟合显式 `dims=2`（此前依赖默认值）、`measure_laplace(run, dims=2)`、协议元数据 `laplace_dims/law/metric` 由 3D 表述改为 2D 表述 |
| `src/tensorlbm/mcmp_verify_report.py` | σ 列头 `sigma = dp R`（dims=2 时不再渲染成 `R/1`）；因子=1 时律表达式渲染为 `Delta p = sigma/R`；图链接改为 doc 相对常量路径（修死链，且保证从 tmp 目录重渲染文本逐字一致） |
| `artifacts/mcmp_verify_bench/results.json` | 按 2D 因子重算 σ/σ₀/σ_slope/Ca_sp，含 `laplace_dims=2` 与 `derivation` 溯源 |
| `artifacts/mcmp_verify_bench/{hero_fields.npz,oss_consumption_chain.json}` | 从 `9125f8a0` 恢复 |
| `artifacts/mcmp_verify_bench/figures/fig1..5.png` | 由 `render_report` 确定性重生成（图 1 横轴/σ 已用 2D 因子） |
| `docs/MCMP_VERIFY_BENCH.md` | 重渲染（σ 列翻倍、律表述修正、图链活链） |
| `tests/benchmarks/test_mcmp_verify_bench.py` | 新增 3 测试：`test_level_fit_uses_the_2d_laplace_factor`（独立 `x=1/R` 重拟合）、`test_factor_scaling_leaves_the_acceptance_metric_invariant`（残差不变/σ 折半，锁住「判据看不见因子」这一事实）、`test_committed_sigmas_carry_the_2d_factor`（逐例 σ==Δp·R、Ca_sp 同因子） |
| 本文件 | 审计/验证报告 |

复现（不重跑求解器）：

```bash
git show 9125f8a0:artifacts/mcmp_verify_bench/results.json > /tmp/raw_results.json
python3 - <<'EOF'
import json, sys; sys.path.insert(0, "src")
import tensorlbm.mcmp_verification as mv
raw = json.load(open("/tmp/raw_results.json"))
for G in (-2.5, -2.75, -3.0):
    cases = [c for c in raw["laplace"]["cases"] if c["G_12"] == G]
    old = [l for l in raw["laplace"]["levels"] if l["G_12"] == G][0]
    new = mv._fit_laplace_level(G, cases, 0.05, dims=2)
    print(G, old["sigma_origin"], new["sigma_origin"], new["sigma_origin"]/old["sigma_origin"],
          old["max_rel_err"], new["max_rel_err"])
EOF
python3 benchmarks/mcmp_verify_bench.py --report   # (需重跑；本轮的渲染见 ⑦)
pytest tests/benchmarks/test_mcmp_verify_bench.py -m "not slow" -q
```

**验证记录**：`pytest tests/benchmarks/test_mcmp_verify_bench.py -m "not slow"` → **27 passed, 3 deselected**（修复前 main 同一命令：1 failed / 21 passed）。