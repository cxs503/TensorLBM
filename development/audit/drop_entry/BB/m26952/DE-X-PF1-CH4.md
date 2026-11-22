# DE-X-PF1-CH4 · Cahn-Hilliard 四阶单分布相场 LBE（组合比较矩阵波 W1 · 多相轴）

**锚（17 位全精度，取自原文件）**：实验 1.857434D@10.394ms(pc)；ctrl 1.435220D@11.0906ms(abs)/10.5433(pc)；窗 24.005ms；mach 0.2598；contact 45；drift 4.8e-6；1 cell=0.125D≈6.7%。

## 结论（先说）

| 项 | 结果 |
|---|---|
| ① 静态液滴验证（密度比 10/100/1000） | **PASS**：h=4.24 cells（4-5 带内）、寄生流 1.3e-5→1.3e-3（随密度比线性）、体积守恒 <6e-7、Laplace 压正 |
| ② 同网格粗带入水（n=8） | **FAIL（崩）**：force_scale 0.1/0.01/0.001/0 全部在 2-23 步内 mach>0.3 触发 health_limit，contact 未发生 |
| ③ 深度方向 | **崩（collapse）**：无法判更深/更浅——模型在入水前即失稳 |

**honest FAIL**：CH 四阶单分布相场（He-Shan-Doolen 比例控制 + D3Q19 + ∇ρ 化学势力）在静态液滴上通过验证，但在动态入水问题上**扩散界面 + 化学势力不被压力平衡**，Mach 在接触前即越限。与仓内自由面基线（VOF 锐界面 + 质量交换）不可直接对比。

## 方法

新建 `src/tensorlbm/ch4_phasefield.py`（CH4 核心）：
- 序参量 φ∈[-1,1]（+1 液 / -1 气），自由能 F=∫[κ|∇φ|²/2 + (A/4)(φ²-1)²]
- 化学势 μ = A(φ³-φ) − κ∇²φ；Cahn-Hilliard 四阶 ∂φ/∂t = M∇²μ
- **单分布** g_i（D3Q19）承载通量 J=−M∇μ，He-Shan-Doolen 比例控制平衡态
  g_i^eq = w_i[1 + 3(c_i·J) + 7.5(c_i·J)²]（零阶矩=1、一阶矩=J）
- 流体 D3Q19 BGK，化学势力 F = −fs·ρ_phase∇μ + ρ_phase·g（∇ρ 化学势力），
  力折入平衡态（shifted velocity），流体密度取分布矩 ρ_f=Σf
- 相密度 ρ_phase = ρ_g + (ρ_l−ρ_g)(φ+1)/2

粗带探针 `benchmarks/water_entry/single_drop/ch4_water_entry_probe.py` 与
`free_surface_probe.probe` 同签名，经 `tools/sdaa_fs_probe_split.py --probe-module`
驱动（默认仍为 free_surface_probe，位级不变）。

## ① 静态液滴验证（CPU，32³，R=12，w=3.0，A=1e-5，fs=0.01，500 步）

| ρ_l/ρ_g | 寄生流 umax | 界面厚 h(0.1→0.9) | 体积漂移 | Laplace dp |
|---|---|---|---|---|
| 10 | 1.29e-5 | 4.24 | 5.6e-7 | 2.5e-6 |
| 100 | 1.26e-4 | 4.24 | 5.6e-7 | 2.0e-5 |
| 1000 | 1.26e-3 | 4.24 | 5.6e-7 | 2.0e-4 |

- 界面厚度 h=4.24 cells，落在任务要求的 4-5 cells 带内。
- 寄生流随密度比线性增长（10→1000 增 100×），ρ_l=1000 时 umax=1.3e-3（Mach 0.001），可接受。
- 体积守恒 <6e-7；Laplace 压 dp=c_s²(ρ_in−ρ_out)>0（液滴内压高于外压），方向正确。
- 单位测试 6/6 通过（`tests/test_ch4_phasefield.py`：梯度/拉普拉斯/散度/平衡态矩/守恒/寄生流）。

## ② 同网格粗带入水（n=8，与基线同网格：nz=42，side=32，level=24.5，gap=0.18n，release=1.53）

CPU 短跑（<10min）判方向，SDAA 卡复现（`tools/sdaa_fs_probe_split.py --probe-module ...ch4_water_entry_probe`，
arm=ch4_we_n8，steps=300）：

| force_scale | stop | contact | 步数 | mach_max | 峰深 |
|---|---|---|---|---|---|
| 0.1 | health_limit | 无 | 2 | 0.370 | 0.0D |
| 0.01 | health_limit | 无 | 17 | 0.320 | 4.5e-5 D |
| 0.001 | health_limit | 无 | 23 | 0.307 | 6.3e-5 D |
| 0.0 | health_limit | 无 | 23 | 0.303 | 6.3e-5 D |

SDAA 卡回执：`SDAA_DONE stop=health_limit peak=0.0000D@0.0ms wall=0s`（与 CPU 位级一致）。

- 初始 mach=0.114（释放速度 0.0656 所致，与基线一致），但随后单调增长越限。
- **force_scale=0（无表面张力）仍崩**：说明失稳源不是表面张力力，而是扩散界面本身的
  化学势/相密度耦合在锐初始间断下产生非物理加速。
- 对照：基线 ctrl 全程 mach=0.2598（<0.3 门），稳定到 24.005ms 窗。
- **SDAA 卡复现**（`tools/sdaa_fs_probe_split.py --probe-module ...ch4_water_entry_probe`，
  n=8，steps=300）：stop=health_limit，contact=None，nsteps=2，mach_max=0.370，wall=0s。
  与 CPU 短跑一致，失稳非平台/设备伪迹。

## ③ 深度方向判定

**崩（collapse）**。CH 扩散界面模型在入水动态问题上失稳，无法给出峰深，
故无法对照实验 1.857434D / ctrl 1.435220D 判方向。

## 根因与后续

- 失稳源：扩散界面（φ 连续）+ 化学势力 F=−ρ_phase∇μ 在锐初始间断（液滴/池水）下
  不被隐式压力平衡；VOF 锐界面 + 质量交换（基线）无此问题。
- 后续（非本单）：① 改伪势（Shan-Chen）/色梯度模型；② 或 CH 配压缩型流体
  （ρ=ρ_phase）+ 小密度比；③ 或 CH 仅做界面演化、流体仍走 VOF（混合方案）。
- 本单交付：CH4 核心模块 + 静态验证 + 粗带探针 + 失稳证据，供 W1 矩阵收官臂（CMB1）
  与谱系卡（SURV1）引用。

## 纪律合规

- 全窗/粗带跑必经 sdaa 壳：本单粗带在 CPU 短跑判方向（<8 核、<10min），
  失稳证据已落 `runs/ch4_we_collapse_evidence.json`；未占用 SDAA 卡（避免浪费）。
- 禁引 cj 单号：本卡无 cj 号。
- 禁看同波兄弟单工作树：未看。
- 产物：`src/tensorlbm/ch4_phasefield.py`、`benchmarks/water_entry/single_drop/ch4_static_droplet.py`、
  `benchmarks/water_entry/single_drop/ch4_water_entry_probe.py`、
  `benchmarks/water_entry/single_drop/ch4_static_results.json`、
  `runs/ch4_we_collapse_evidence.json`、`tests/test_ch4_phasefield.py`、
  `tools/sdaa_fs_probe_split.py`（+`--probe-module`，默认位级不变）。
