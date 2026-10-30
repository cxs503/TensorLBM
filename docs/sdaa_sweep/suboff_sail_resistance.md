# SDAA 验证案例扫荡 — `suboff_sail_resistance`

案例：`benchmarks/verified/suboff_sail_resistance`（SUBOFF AFF-3 with_sail，Re=1000，Cd vs Blasius）
运行脚本：`scripts/sdaa_sweep_suboff_sail_resistance.py`（独立驱动，**未改任何业务代码**）
机器产物：`artifacts/sdaa_suboff_sail_resistance.json`
原始日志/结果：`runs/sdaa_sweep_suboff_sail_resistance/`
执行日期：2026-10-29 · 设备 `sdaa:0` · 总墙钟 **485.3 s**（8.1 min，≤20 min 预算内）

---

## 1. 结论（honest-FAIL）

**OVERALL = FAIL**。

| 门 | 判据（案例原口径，未放宽） | 结果 |
|---|---|---|
| G1 | 最细档 (L128) \|err\| ≤ 3% | **FAIL — 不可评估**：L128 在本卡 OOM，无数值可算 → 按要求记 FAIL |
| G2 | \|err\| 沿阶梯 {96,112,128} 非增 | **FAIL — 不可评估**：三档全部 OOM，序列缺失 → FAIL |
| G3 | 共性模块入口（GeneralSimEngine + suboff_cad + drag_pressure） | **PASS**（源码扫描 + 实际执行的补充档 result JSON 溯源字段） |
| G4 | 窗口漂移 ≤ 2.5% | **FAIL — 不可评估**：阶梯档无收敛窗口 → FAIL |
| G5 | 围壳体素匹配冻结掩码表 ±2 且围壳 y 厚 = 2 | **PASS**（48/70/126，厚度全 2 — 由本脚本用案例自带 `build_suboff_mask` 机算复现） |

失败根因只有一条：**本卡显存 14.94 GiB，冻结阶梯 {96,112,128}（23.7M / 37.7M / 56.4M 胞）在第一步
`collide_mrt3d_low_memory` 的 `matrix @ feq_flat` 就 OOM**。案例归档（NVIDIA 侧）本身记录
L112 eager 峰值 17.6 GiB、需 32 GiB 单卡（prereg.md E1），与本卡容量差 ~2 倍。
**没有降判据、没有用 CPU 数值冒充 sdaa**；唯一的补充运行是明确标注的 off-ladder 档，不进入任何门。

---

## 2. 设备证据

```
torch.sdaa.get_device_name(0) -> /dev/tcaicard25        # 逐字输出，写入 artifact.device_evidence_line
```

| 项 | 值 |
|---|---|
| torch / torch_sdaa | 2.7.1 / 3.1.1a3+gite3971b9 |
| `torch.sdaa.is_available()` / `device_count()` | True / 1 |
| 显存 | total 14.938 GiB，idle free 14.687 GiB |
| 活性自检 | `sdaa: torch.ones(4)*2 -> sum() == 8.0` → PASS |
| 计算张量落设备 | 全部经案例入口 `--device sdaa:0`；每个 result JSON 内 `"device": "sdaa:0"` |

---

## 3. 配置（案例原口径，逐项未改）

| 项 | 值 | 来源 |
|---|---|---|
| 参考 | Cf = 1.328/√Re，Re=1000 → `Cf_ref = 0.04199504732703608` | 本脚本重算，与 run.py 内值一致（`cf_ref_match: true`） |
| 归一化 | 总湿面积 S = π(2R_lb)L + A_sail_own | result JSON `S_total_wetted_lb2` |
| 阶梯 / 步数 | L ∈ {96,112,128}，steps = 150·L → 14400 / 16800 / 19200 | prereg.md（冻结） |
| 域 | (1,4,1,1,1,1) padding → 576×203×203 / 672×237×237 / 768×271×271 | 实测 log |
| 物理 | D3Q19 MRT，u_lb=0.05，ν=u_lb·L/1000，τ=0.5144 | log `setup:` |
| 阻力口径 | friction=`standard`（E2 主估计量），p0=`near_wall`，pressure_extrap=`none` | prereg / erratum E2 |
| 窗口 | 末 1000 个力样本（=末 10000 步）均值 | run.py |
| compile | `--compile-mode eager`（prereg 冻结值；SDAA 本就无法编译该链） | prereg.md |

---

## 4. 结果表

### 4.1 冻结阶梯（原口径）— 三档全部 OOM

| tier | 域 | 全域胞数 | 围壳体素 | 请求步数 | 结果 | 墙钟 | 逐字 OOM 证据 |
|---|---|---|---|---|---|---|---|
| L96 | 576×203×203 | 23,736,384 | 48 | 14400 | **OOM @ step 1** | 23.8 s | `Tried to allocate 17179869182.15 GiB … 7.47 GiB already allocated; 6.70 GiB free` |
| L112 | 672×237×237 | 37,745,568 | 70 | 16800 | **OOM @ step 1** | 29.4 s | `Tried to allocate 2.67 GiB … 9.21 GiB already allocated; 4.84 GiB free` |
| L128 | 768×271×271 | 56,402,688 | 126 | 19200 | **OOM @ step 1** | 36.2 s | `Tried to allocate 3.99 GiB … 8.93 GiB already allocated; 5.13 GiB free` |

OOM 点（三档同）：`src/tensorlbm/solver3d.py:217` `moments_eq = matrix @ feq_flat`
（`collide_mrt3d_low_memory`）。L96 的请求尺寸 1.7e10 GiB ≈ 2^64 字节，属后端尺寸溢出式报错；
L112/L128 是真实显存不足（可用 4–5 GiB vs 需 2.7–4.0 GiB 连续块，alloctor 已 reserved 9.3–9.6 GiB）。
为节省墙钟，可行性用 2 步同口径试探（文档推荐步数在本卡~2.5 h 且必然首步即 OOM）。

### 4.2 补充档（off-ladder，容量受限，**不参与判门**）

| 试档 | 结果 | 说明 |
|---|---|---|
| L72（最小"合规围壳"档：sail 32 胞、y 厚 2） | OOM | `Tried to allocate 5.95 GiB … 5.21 GiB free` |
| L70 | OOM | `Tried to allocate 5.48 GiB … 5.95 GiB free`（碎片） |
| **L68** | **跑通** | 408×144×144 = 8,460,288 胞，406 步，734.5 ms/step，316.97 s |

L68（eager，`sdaa:0`）实测：

| 量 | 值 |
|---|---|
| Cd_total（末窗口，40 样本） | 0.0831243664268973 |
| Cd_pressure / Cd_friction | 0.012796443206784044 / 0.07032792322011326 |
| err vs Cf_ref | **+97.9385%**（远未收敛：caliber 需 10200 步，只跑 406 步，短 25.1×） |
| 窗口漂移 \|Cd(100%)−Cd(75%)\|/Cd(75%) | 7.9142%（>2.5%，仍在下降 0.1222→0.1009→0.0903→0.0831） |
| 围壳 y 厚度 | **1 胞（退化）**，24 胞；σ 偶维退化，案例 prereg 已记该病 |
| finite / diverged | true / false |
| 设备 | `"device": "sdaa:0"` |

该档**不是**案例判据：它只证明"链路在本卡能跑、吞吐约 0.73 s/步、几何最小可跑档已退化"。
它的 err/漂移**不用**来判任何门（脚本内 `supplementary: true` 且门计算只取 `LADDER`）。

---

## 5. 判据逐条比对（全部脚本机算，零手抄）

| 门 | 阈值 | 实测（脚本算） | 判定 |
|---|---|---|---|
| G1 | 最细档 \|err\| ≤ 3% | L128 无窗口均值（OOM）→ 无值 | **FAIL**（不可评估，honest） |
| G2 | \|err\| 沿阶梯非增 | 缺失 `['L96','L112','L128']` | **FAIL**（不可评估，honest） |
| G3 | 共性模块入口 | run.py 导入 general_sim/suboff_cad/drag_pressure 全 True；执行档 result JSON 含 `S_total_wetted_lb2/dpS_type/appendage_stats` | **PASS**（并已披露：执行证据来自补充档，阶梯本身未跑） |
| G4 | 漂移 ≤ 2.5% | 阶梯档无收敛窗口 | **FAIL**（不可评估，honest） |
| G5 | 掩码 48/70/126 ±2，y 厚 2 | L96: cells=48, thick=2；L112: 70, 2；L128: 126, 2（脚本用 `build_suboff_mask` 独立重建，与 log 的 `appendage_solid_cells` 及冻结表一致） | **PASS** |

**G5 的可信度说明**：这是本轮唯一拿到 REAL 证据的门 —— 案例的 SUBOFF+sail 几何在 SDAA
主机上按案例自己的 `suboff_cad` 代码重建，三档围壳 48/70/126 胞、y 厚全为 2，
与 `prereg.md` E1 冻结表逐值一致（±0 差，容差 ±2）。即：几何/掩码链是 REAL，问题只在算力容量。

---

## 6. 墙钟

| 段 | 墙钟 |
|---|---|
| 设备探针 + 入口扫描 | <1 s |
| 冻结阶梯 2 步可行性试探 ×3（含 setup、掩码重建） | 23.8 + 29.4 + 36.2 s |
| 补充档试探 L72 / L70 / L68(10 步) | 17.8 + 16.7 + 26.5 s |
| 补充档 L68 406 步 | 316.97 s（其中步进循环 298.2 s） |
| **总** | **485.3 s**（脚本自报 `wall_clock_s`） |

---

## 7. 绕行记录（如实）

1. **triton**：**未使用 TRITON_INTERPRET**，也不需要。执行的链是 eager `lbm_step_correct` → D3Q19 MRT，
   全程不 import triton kernel；SDAA 侧 `torch.compile` 走 `teco_inductor`，其上游 L5 缺陷使该链
   无法编译（`benchmarks/compile_route.py` 已文档化），而 `--compile-mode eager` 正是 case prereg
   冻结的参数值。log 逐档记录：`[compile_route] suboff_with_sail[Lxx]: mode=None -> eager (compile_step passthrough) [sdaa-teco_inductor]`。
2. **墙钟 → 步数换算**：补充档步数由实测 ms/step 反算（`budget 300 s ÷ 0.7379 s/step → 406 步`），
   脚本内记录 `wall_clock_derived_steps=406`、`caliber_steps_at_this_L=10200`、`step_shortfall_factor=25.123`。
3. **可行性试探用 2 步**：阶梯档推荐步数（14400/16800/19200）在本卡需 ~3–4 h 且首步即 OOM，
   故以同口径 2 步问"能否装下"的问题；这不是降判据，只是不等它跑完就拿到确定性答案。
4. **显存碎片**：L70 在 2 步时曾成功、10 步时失败（可用 5.95 GiB vs 请求 5.48 GiB，
   allocator reserved 8.48 GiB）→ 记录为本卡 `collide_mrt3d_low_memory` 8× 临时张量的碎片敏感性问题。
5. **入口 md5 与 README 声明不一致（如实登记）**：README 称冻结驱动 md5 = `a8527e818202d6f724825e0fe0306365`，
   本工作树实测 `run.py` md5 = **`9d3f7f7091766a1df02ab47806d5f2b0`**。
   `git log`：promote 之后 `cf36a436 (SIR-1b/m298)` 追加了 `SUBOFF_TAU_TARGET` τ-override 钩子、
   `07cbd913 (QUAL-CI/m310)` 等改动。该钩子在环境变量缺省时返回 None（`zero behaviour change`），
   本轮未设该变量，实测 log `setup: tau=0.5144` 与归档一致 → **数值口径未变**，但 README 的
   md5 声明已过期，建议另行更新（本任务只写不改业务代码，未动 README）。

---

## 8. 复现

```bash
# 完整扫荡（约 8 min，含阶梯可行性 + 补充档）
python scripts/sdaa_sweep_suboff_sail_resistance.py --budget 300 --tier-timeout 200

# 单档直跑（案例入口原样）
cd benchmarks/verified/suboff_sail_resistance
python run.py --device sdaa:0 --resolution 128 --steps 19200 \
              --friction standard --p0 near_wall --compile-mode eager   # 阶梯档（本卡会 OOM）
python run.py --device sdaa:0 --resolution 68 --steps 406 \
              --friction standard --p0 near_wall --compile-mode eager   # 本卡可跑通的最小档
```

产物：`artifacts/sdaa_suboff_sail_resistance.json`（含 `device_evidence_line`、`tiers`、`supplementary_reduced_grid`、
`gates` 逐条 PASS/FAIL、`runs` 内嵌每次运行的完整 result JSON 与日志路径、`wall_clock_s`）。

---

## 9. 建议（不在本任务范围，未执行）

1. 要用本卡跑通该案例，需要 ≥32 GiB 的 SDAA 卡（案例归档侧为 32 GiB NVIDIA 单卡）；或在业务代码里
   把 `collide_mrt3d_low_memory` 的 `matrix @ feq_flat` 换成已存在的 fused `R = M⁻¹diag(s)M` 路径
   （`solver3d._mrt_fused_cache` 已在库内）——那属于业务代码改动 + 口径变更，须另开任务并重判。
2. 本卡实测吞吐 ~0.73 s/步（8.46M 胞，eager MRT matmul 路径）；按此推算 L128 一档 19200 步 ≈ 1.2 h，
   具备容量后仍可接受。
3. README 的驱动 md5 声明需刷新（见 §7.5）。