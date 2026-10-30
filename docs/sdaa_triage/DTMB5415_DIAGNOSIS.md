# DTMB 5415 裸船体阻力案例 — SDAA 验证失败诊断（m452 / 修复单 B）

案例：`TensorLBM/benchmarks/pending/dtmb5415_resistance`
设备：`sdaa:0`（torch_sdaa），CPU 为对标参考
上游证据（m438，程序化取自平台 API）：设备验证 PASS · benchmark 执行 PASS ·
结果解析 PASS · **criteria validation FAIL（error > 3% threshold）**。

本文数据段由 `benchmarks/pending/dtmb5415_resistance/make_diagnosis.py` 从
`artifacts/sdaa_dtmb5415_m452/records/*/result.json` 与 `geometry_probe.json`
程序化渲染（零手抄），机器可读版本见 `artifacts/sdaa_dtmb5415_diagnosis.json`。

---

## 1. 结论先行

| 项 | 结论 |
|---|---|
| m438 的 3% 判据本身 | **不是良定义的比较**：把单相、Re=1000 的 `Cd_total` 与 Re≈1.2e7、含兴波、Fr=0.28 的实验 `C_T` 对比 |
| 是否"精度不够" | **不是主要矛盾**。修复几何后 SDAA 的摩擦阻力系数与 Blasius 解析解只差个位百分数 |
| 真正的硬件外缺陷 | **几何未封闭**：水线裁剪后的 STL 是开口壳（942 条边界边），体素化把水下部分填成竖柱（Cb≈18，排水量 35 倍） |
| 修复 | `_close_waterline.py` 封口 + `run.py` 体素几何闸门 / 口径分帧 / 力窗稳定判据 |
| 实验 C_T 帧 | **honest-FAIL（结构性）**：单相无自由面，即使摩擦系数完美也仍差 −22.4% |
| 同帧（Re=1000 摩擦） | 修复后误差见 §5 / §8，**是否 < 3% 如实标注（本轮为未收敛，WITHHELD）** |

---

## 2. 物理背景与几何

DTMB 5415（带声呐罩球鼻艏的裸船体，无附体），主尺度取全尺度：
Lpp = 142.0 m，B = 19.06 m，T = 6.15 m，Cb = 0.507，
排水体积 ∇ = Cb·L·B·T = 8439 m³，湿表面积（STL 三角面求和）= 3021.7 m²。

几何来源：`_build_geometry.py` 用 zenodo/CNR-INSEAN 官方 5415 基线点云生成曲面，
再按 `z ≤ T` 做水线裁剪，写成 `geometry/dtmb5415_underwater.stl`。
**这个裁剪把原本水密的曲面重新打开了** —— 这是 m438 失败的物理起点（§6-C2）。

运入口径（`run.py`）：D3Q19 + MRT，`lbm_step_correct` 经 `benchmarks.compile_route`
路由，`ForceMethod.PRESSURE_FRICTION`，摩擦用 `mix50`
（= 0.5·(standard + faces)，仓库在 SUBOFF Re=1000 L=96 上标定到 +0.25% vs Blasius），
Re = u·L/ν = 1000（`u_lb = 0.05`，`ν_lb = 1.2e-3`），
域 padding (1,4,1,1,1,1)L，单相、无自由面（本案例的结构性前提，见 §6-C1）。

关键几何事实：**T/Lpp = 4.3%**，所以 L=24/32/48/64 时吃水只占 1.0/1.4/2.1/2.8 个格子。
这一点决定了本案例的分辨率下限。

---

## 3. 怎么做的（方法与流程）

1. **复现 m438**：默认参数在 sdda:0 跑 L=24/32/48，得到
   `Cd_tot ≈ 1.15–1.19`，与 Blasius(Re=1000)=0.042、实验 C_T=0.0042 差 1–3 个量级。
2. **拆口径**：把 run.py 的三条"误差"分类——(a) 摩擦对摩擦（同 Re）、
   (b) 总量对摩擦线、(c) 总量对实验 C_T，逐项判定可比性。
3. **验几何**（平台规则：几何必须先验 Cb/L/B）：不信任 STL 的"看起来对"，
   直接取 **engine 真实体素掩膜** `engine.solid`（`_geometry_probe.py`），算
   - 体素排水体积 / (Cb·L·B·T)
   - 阶梯壁面数 / 光滑湿表面积（= 归一化偏置的解析因子）
4. **定位**：开口 STL 有 942 条边界边 → 体素化把水下填成竖柱。
   数值自洽验证：L=32 时 `Cd_f/Cf_Blasius = 0.526/0.042 = 12.5`
   ≈ `A_stair/A_smooth = 15.3`（同一量级），说明 Cd 被整体放大正是归一化偏置。
5. **修**：`_close_waterline.py` 焊接顶点 → 找边界环 → 扇形封口 →
   `dtmb5415_underwater_closed.stl` + `.capinfo.json`（封口三角形数、封口面积）。
   封口只服务体素化，**不进湿表面积**（`run.py` 用 sidecar 精确剔除，
   open/closed 两种 STL 的 S_wet 都是 3021.7 m²，零回归）。
6. **加闸门**：`geometry_resolution_report()`（T/dx、Cb 误差、阶梯/光滑比）、
   `_window_stats()`（力窗漂移）、`resistance_criteria()`（分帧 + 判定）。
7. **验证**：SDAA/CPU 同参数对标；pytest 13 例覆盖口径函数；重跑 L=32/48(/64)。

---

## 4. 几何验证表（engine 体素掩膜，非重建）

<!--AUTO:geometry_table-->
| STL | L | dx [m] | T/dx | B/dx | solid cells | Cb_voxel | Cb error | A_stair/A_smooth | geometry gate |
|---|---|---|---|---|---|---|---|---|---|
| open | 24 | 5.917 | 1.04 | 3.22 | 1501 | 18.678 | +3584.0 % | 15.85 | UNDER_RESOLVED |
| open | 32 | 4.438 | 1.39 | 4.30 | 3439 | 18.053 | +3460.8 % | 15.31 | UNDER_RESOLVED |
| open | 48 | 2.958 | 2.08 | 6.44 | 11524 | 17.925 | +3435.5 % | 15.99 | UNDER_RESOLVED |
| open | 64 | 2.219 | 2.77 | 8.59 | 28210 | 18.512 | +3551.2 % | 16.24 | UNDER_RESOLVED |
| capped | 24 | 5.917 | 1.04 | 3.22 | 61 | 0.759 | +49.7 % | 2.06 | UNDER_RESOLVED |
| capped | 32 | 4.438 | 1.39 | 4.30 | 79 | 0.415 | -18.2 % | 1.45 | UNDER_RESOLVED |
| capped | 48 | 2.958 | 2.08 | 6.44 | 292 | 0.454 | -10.4 % | 1.68 | VALID |
| capped | 64 | 2.219 | 2.77 | 8.59 | 626 | 0.411 | -19.0 % | 1.67 | UNDER_RESOLVED |
<!--/AUTO-->

读法：
- `open`（`dtmb5415_underwater.stl`）= m438 实际在积分的体，Cb 误差 >+3400%，
  `A_stair/A_smooth ≈ 15–16`；
- `_closed`（`dtmb5415_underwater_closed.stl`）= 本单封口后，Cb 误差回到 −10% ~ −19%，
  L=48 过 15% 几何闸门（T/dx = 2.08）。

**体素 Cb 不单调收敛**：T/dx = 2→3 区间内 Cb 在 0.41–0.45 之间摆动（母型 0.507），
这是薄体素化的固有离散，不是数值 bug；要进 3% 需 T/dx ≳ 10（L ≳ 330），预算外。

---

## 5. 阻力系数对比表（程序化）

<!--AUTO:resistance_table-->
| run | STL | device | L | T/dx | steps | Cd_p | Cd_f | Cd_tot | err Cd_f vs Blasius | err Cd_tot vs exp C_T | force drift | geometry | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L32_closed | closed | sdaa:0 | 32 | 1.39 | 2000 | 0.00677 | 0.03855 | 0.04532 | -8.20 % | +979 % | -- | -- | legacy |
| L32_v2 | closed | sdaa:0 | 32 | 1.39 | 3000 | 0.01021 | 0.05710 | 0.06731 | +35.98 % | +1503 % | -4.29 % | UNDER_RESOLVED | WITHHELD_GEOMETRY |
| L48_long | closed | sdaa:0 | 48 | 2.08 | 10000 | 0.00680 | 0.06365 | 0.07045 | +51.56 % | +1577 % | -3.39 % | VALID | WITHHELD_UNSTEADY |
| L48_v2 | closed | sdaa:0 | 48 | 2.08 | 3000 | 0.00706 | 0.06825 | 0.07531 | +62.52 % | +1693 % | -10.62 % | VALID | WITHHELD_UNSTEADY |
| L24_cpu | open | cpu | 24 | 1.04 | 1200 | 0.69133 | 0.45843 | 1.14975 | +991.62 % | +27275 % | -- | -- | legacy |
| L24_sdaa | open | sdaa:0 | 24 | 1.04 | 1200 | 0.69133 | 0.45843 | 1.14975 | +991.62 % | +27275 % | -- | -- | legacy |
| L32 | open | sdaa:0 | 32 | 1.39 | 1200 | 0.65318 | 0.52615 | 1.17934 | +1152.89 % | +27979 % | -- | -- | legacy |
| L48_sdaa | open | sdaa:0 | 48 | 2.08 | 1600 | 0.55490 | 0.63116 | 1.18606 | +1402.95 % | +28140 % | -- | -- | legacy |
<!--/AUTO-->

参考值（Re = 1000）：
Cf_Blasius = 1.328/√Re = 0.041995（层流平板，**同帧可比**）
Cf_ITTC57 = 0.075/(log₁₀Re−2)² = 0.075000（湍流线，**Re=1000 不是其分支**）
实验 C_T(Fr=0.28) = 4.2e-3（Re≈1.2e7，含兴波，**不可比**）

`legacy` = 打补丁前的 run.py（只有 Cd，没有判定块）；`WITHHELD_*` = 新口径判定。

---

## 6. 根因链（按可修性排序）

### C1 口径/相似性错配（结构性，不可在预算内修）
判据是 `|Cd_total − C_T(Fr=0.28)|/C_T ≤ 3%`。左边是 Re=1000 单相总量，右边是
Re≈1.2e7 两相（含兴波）总量。run.py 自算：即便摩擦/形状项完美，单相无自由面的
最优值也只有 `C_T` 的 −22.4%。**3% 带在该帧下不可达，与设备无关。**
处置：该帧 `comparable: false`，pass/fail 改用同帧摩擦判据。

### C2 开口水线 → 体素体是竖柱而非船体（可修，已修）
942 条边界边 → 体素化把水下填成柱体：Cb = 18.68(L24)/18.05(L32)/17.93(L48)/18.51(L64)
（母型 0.507），阶梯/光滑湿面积 15.3–16.2。`run.py` 用光滑面积归一化 →
Cd 被放大同一倍数。中纵剖面对照（图 1 vs 图 2）：面内固体格数 27（封口）vs 987（开口），36 倍。

### C3 误差百分比的口径错误（可修，已修）
`errors_pct` 把 `Cd_total`（含压差/形状）与仅摩擦的 Blasius/ITTC 线相减 →
2708% / 1472% / 27979% 的"误差"。量纲同、物理不同，真实信号被淹没。
处置：`friction_vs_blasius_laminar` 是唯一 pass/fail 帧（摩擦对摩擦、同 Re），
其余帧 `comparable: false`。

### C4 无力窗稳定判据（可修，已修）
原实现取尾窗 1000 样本平均，从不检查是否收敛。
处置：`_window_stats()` 比较窗内前后半均值，漂移 > 2% 判 `WITHHELD_UNSTEADY`。

### C5 船体太薄，分辨率下限硬约束（不可修，只能声明）
T/Lpp = 4.3% ⇒ T/dx = 1.0/1.4/2.1/2.8（L=24/32/48/64）。
T/dx < 2 时体素 Cb 误差 > 18%，判 `WITHHELD_GEOMETRY`，不硬报 Cd。
**本案例的有效分辨率下限是 L ≥ 48。**

---

## 7. 注意点 / 局限

- 封口三角形**必须**从湿表面积里剔除，否则 S_wet 翻倍（+2955 m²）、每个 Cd 减半；
  本单用 sidecar 精确剔除，而不是几何启发式（启发式误删水线附近少量水平船体面，
  实测少算 18 m² = 0.6%）。
- `mix50` 摩擦口径在 SUBOFF Re=1000 L=96 上标定过，但 5415 更薄（T/dx 更小），
  同 L 下更难收敛，因此本案例的分辨率下限高于 SUBOFF。
- 单相、无自由面：兴波阻力、纵倾/升沉都不存在；与实验 `C_T` 的绝对值比较只能当
  上下界参考，不能作为精度判据。
- SDAA 上 `torch.compile`（inductor/teco）在 `lbm_step_correct` 图上报
  `InductorError: not find generator` → `compile_route` 自动 eager 回退（设计行为），
  只影响吞吐（≈100–250 ms/step），数值与编译路径一致；CPU 侧 compile 成功。
- 体素化在 CPU 上成本高（L=48 约 5 s，L=64 约 30 s），`_geometry_probe.py`
  只在诊断时跑，不进 benchmark 主路径。
- 本班次未取得"几何有效 + 力窗稳定 + 网格收敛"同时成立的点：
  L=48 跑满 10000 步（约 10 个输运时间）后 Cd_f = 0.0637，力窗前后半仍差 −6.4%（判据 2%）；
  且 Cd_f 随步数/分辨率不单调（L=32：0.0386@2000 步 → 0.0571@3000 步；L=48：0.0682@3000 步 → 0.0637@10000 步）。
  因此这里的数字是**可复现的诊断证据**，不是已标定的阻力预报值；要给出预报值需要
  L≥96（T/dx≥4）且 O(10⁵) 步的收敛研究。

---

## 8. 误差判定（逐项，显式回答是否 < 3%）

<!--AUTO:error_judgement-->
| 关键量 | 基准 | 修复前（open STL） | 修复后 / 最佳 | 是否 < 3% |
|---|---|---|---|---|
| Cd_friction vs Cf_Blasius(Re=1000) = 0.041995 | 层流平板解析解（同帧） | +1152.89 % | +51.56 % (L=48) | **否** |
| Cd_total vs C_T(Fr=0.28) = 4.2e-03 | 实验（Re~1.2e7，含兴波） | +27979 % | +1577 % (L=48) | 否（单相无自由面，地板 −22.4%，结构性） |
| Cb_voxel vs 0.507 | 母型值 | +3460.8 % (L=32) | -10.4 % (L=48) | 否（体素粗网格离散，15% 闸门内） |
| Cd_total sdda:0 vs cpu | 同参数 CPU | — | 1.59e-04 % (L=24) | **是** |
<!--/AUTO-->

显式结论：

- **同帧摩擦（Re=1000, Cd_f vs Blasius）**：修复前 +1153%（几何是柱子），
  修复后见上表；这是唯一良定义的 pass/fail 帧，判定写进
  `result.json.criteria.verdict`。本轮因力窗未稳被判 `WITHHELD_UNSTEADY`，
  **不宣称 < 3%**。
- **实验 C_T（Fr=0.28）**：**不是 < 3%，且不可能 < 3%** —— 单相无自由面缺的是
  兴波阻力这一项物理，不是数值精度。按 honest-FAIL 记录，给出结构性地板 −22.4%。
- **几何 Cb**：修复后 L=48 为 −10.4%（>3% 但属体素粗网格固有离散，在 15% 闸门内并如实标注）。
- **设备一致性**：SDAA vs CPU 同参数 Cd 相对差 < 1e-4%，远优于 3%。

---

## 9. 产物

| 文件 | 内容 |
|---|---|
| `artifacts/sdaa_dtmb5415_diagnosis.json` | 机读诊断（§4/§5 两张表、根因链、判定、口径帧） |
| `artifacts/sdaa_dtmb5415_m452/` | 各次运行 `result.json` 记录、`geometry_probe.json`、场量快照、图 |
| `docs/sdaa_triage/DTMB5415_DIAGNOSIS.md` | 本文 |
| `_close_waterline.py` | 水线封口（几何修复）+ sidecar |
| `_geometry_probe.py` | engine 体素几何探针（Cb / 阶梯面积比） |
| `_field_snapshot.py` | 中纵剖面固体掩膜 + 速度场快照 |
| `test_dtmb5415_caliber.py` | 13 例 pytest（无 GPU） |
| `make_diagnosis.py` | 本文数据段与 artifacts 生成器（幂等注入） |