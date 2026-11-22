# 单滴入深水池 · 气穴深度定量对照

验证对象是液滴首触后，与大气连通的气穴最低点到初始静水面的竖直距离。时间以毫秒计，深度以撞前等效滴径 D 归一化。比较覆盖气穴增长、最深点和回缩。这个三维案例与仓库中的二维液桥回归使用不同的观测量。

## 实验与精度

Santini、Fest-Santini 和 Cossali 的原始实验表 2 第 3 行是本算例。表 3 分别列出了滴水和池水的物性。Guilizzoni 与 Frontera 在 2022 年论文图 5 重绘了该实验的测点。图中的数值计算曲线不参与取点。[原始实验](https://www.sciencedirect.com/science/article/abs/pii/S0997754616302977) · [图 5 与实验对照方法](https://doi.org/10.3390/fluids7020050)

| 量 | 实验条件 |
|---|---|
| 单滴直径、撞击速度、撞前纵横轴比 | 2.27 ± 0.03 mm，1.40 ± 0.02 m/s，0.96 |
| 水槽 | 边长 40 mm 的玻璃立方体，水面在槽口附近 |
| 滴水 | 27 ± 1 °C，997 kg/m³，0.853 mPa·s，71.42 mN/m |
| 池水 | 30 ± 2 °C，995.65 kg/m³，0.7976 mPa·s，70.98 mN/m |
| 原文报告的无量纲数 | Re = 3715，We = 62.1，Fr = 88.0，采用 Fr = U²/(gD) |
| 摄影与读数 | 1000 帧/s，首触时刻不确定度 0.03 ms，长度测量精度约 0.03 mm |

参考 CSV 包含全部 18 个实验标记的像素质心、物理时刻和无量纲深度。取点程序只识别红色连通区域并排除图例，从出版商正式版第 9 页提取原生 3337 × 1832 JPEG。最大观测深度为 **1.8574 D**，对应 **10.394 ms**。原始图和导出数据的 SHA-256 已固定。允许的 1.5 像素读图偏差相当于约 ±0.00985 ms 和 ±0.00192 D。原实验长度测量精度折合约 0.0132 D，首个点仅 0.0389 D，3% 的差值约为 0.00265 mm。后者小于实验的长度测量精度。[18 个测点](fig5_experiment.csv) · [测量口径、数值门限与哈希](protocol.json)

## 求解与测量

- **双相路径**用 D3Q19 保守 Allen–Cahn 相场和压力分布函数解水与空气，可选二松弛时间碰撞；偶分量控制黏度，奇分量有独立的力项修正。墙面可只替换求导模板的相场虚值，保留反弹中的分布函数。每步记录气相与液相的速度、最小密度和相场体积变化。
- **自由面路径**用 D3Q19 液相与大气定压重构，界面压力包含曲率项。初始椭球用每格 64 点体积积分，保持给定等效体积和撞前纵横轴比。可选择在 D3Q19 邻格间按可用质量分配转换亏额；多个界面单元同时索取时共享额度，无法局部结清的转换保留原单元状态。另有基本的对称界面质量通量供对照。
- **物理映射**采用 Δx = D/N，Δt = U格 Δx/U。两条路径的水相均使用滴水物性；池内单独的温度、示踪染料和物性输运尚未建模。双相路径使用名义空气物性；定压自由面路径省略空气惯性。椭球体积等于直径 D 的球，初始压力采用等效球 Laplace 压差近似。
- **域与边界**默认是计算宽度 4 D、池深约 3 D 的筛查域，侧向周期，竖直边界为壁面。可用 `--span`、`--pool-depth` 独立扩大水槽。实验槽宽约 17.6 D。当前缩域和边界对峰值与回缩的影响仍需量化。
- **首触与深度**每步按六邻接跟踪液滴核心与池体的连通，并以同一批液滴核心单元的重叠确认其身份。液滴消失会触发跟踪失败。撞前记录最后一次分离时半填充核心的平均竖直速度，并在开启 `--silhouette` 时测量侧视图中液滴闭合轮廓的面积质心。实验速度由相邻影像中的中心位移得到。验收采用撞前相隔 1 ms 的最后两个中心位置；全段拟合速度与液滴内部速度用于诊断。缺少该帧间隔记录时，速度项维持待验证。深度从顶部连通的气体向下取半填充等值面的亚格交点，取距冲击轴 1.5 D 内的最深位置。封闭气泡排除在测量之外。
- **判分**在每个实验时刻对满足密度、Ma 和库存运行条件的相邻模拟样本内插，缺测保留为空。同时比较整个实验观测时段内的模拟最大值，包括实验帧之间的模拟峰值。完整时序和峰值相对误差均须 ≤3%，同一配置至少两档网格误差严格递减，并核对实测入射速度、边界与参考分辨能力。

## 复算

[历史探索索引](../../../development/audit/drop_entry/indexes/) 收录不同入水研究分支的原始判定与待查缺口。当前案例按本页判据验收。

在仓库根目录执行，使用提供 PyTorch、NumPy、SciPy、Pillow 和 Matplotlib 的 Python 环境。初始版的诊断组采用 2.29 mm 的邻近工况，可用 `--pilot` 复算；默认运行原始实验的单滴工况。仓库保留标量逐步历史、四个物理时刻的中剖面和对照图。

```bash
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=outputs/single_drop
mkdir -p "$out"
python -m benchmarks.water_entry.single_drop.free_surface_probe --cells 8 --gap .18 --speed .06 --capacity --out "$out/case3_d8.json"
python -m benchmarks.water_entry.single_drop.free_surface_probe --cells 12 --gap .18 --speed .06 --capacity --out "$out/case3_d12.json"
python -m benchmarks.water_entry.single_drop.run --cells 12 --width 6 --gap .13 --odd-rate 1 --uniform-drop --stencil-wall --out "$out/case3_cac12.json"
python -m benchmarks.water_entry.single_drop.free_surface_probe --cells 8 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --planes "$out/case3_planes.npz" --silhouette "$out/sideview_cal8.json" --out "$out/case3_cal8.json"
python -m benchmarks.water_entry.single_drop.free_surface_probe --cells 12 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 --silhouette "$out/sideview_cal12.json" --out "$out/case3_cal12.json"
python -m benchmarks.water_entry.single_drop.free_surface_probe --cells 8 --gap .18 --speed .04 --capacity --simple-exchange --release-speed 1.53 --silhouette "$out/sideview_dt8.json" --out "$out/case3_dt8.json"
python -m benchmarks.water_entry.single_drop.free_surface_probe --cells 8 --gap .18 --speed .06 --span 6 --pool-depth 4.5 --capacity --simple-exchange --release-speed 1.53 --silhouette "$out/sideview_wide8.json" --out "$out/case3_wide8.json"
python -m benchmarks.water_entry.single_drop.free_surface_probe --cells 8 --gap .18 --speed .04 --capacity --simple-exchange --release-speed 1.885679191832972 --silhouette "$out/sideview_motion8.json" --out "$out/case3_motion8.json"
python -m benchmarks.water_entry.single_drop.free_surface_probe --cells 8 --gap .78 --speed .04 --capacity --simple-exchange --release-speed 1.575 --silhouette "$out/sideview_camera8.json" --out "$out/case3_camera8.json"
python -m benchmarks.water_entry.single_drop.free_surface_probe --cells 8 --gap .78 --speed .04 --capacity --simple-exchange --release-speed 1.658 --silhouette "$out/sideview_matched8.json" --out "$out/case3_matched8.json"
python -m benchmarks.water_entry.single_drop.score "$out"/case3_*.json --output "$out/../single_drop_score.json"
```

归档还包含三个单因素或短程对照。以下开关都传给同一 `free_surface_probe`，并使用表中的文件名作为 `--out`。

| 文件名 | 完整的工况开关 |
|---|---|
| `case3_release8.json` | `--cells 8 --gap .18 --speed .06 --capacity --release-speed 1.53 --steps 160` |
| `case3_flux8.json` | `--cells 8 --gap .18 --speed .06 --capacity --simple-exchange --steps 160` |
| `case3_longflight8.json` | `--cells 8 --gap .6 --speed .04 --capacity --simple-exchange --release-speed 1.885679191832972 --steps 205 --silhouette outputs/single_drop/sideview_long8.json` |

纯液相速度可用 `--release-speed` 单独给定释放初速，格子时间步仍按目标撞击速度换算。另选 `--simple-exchange` 时使用所有界面–界面连线上的平均填充率通量。每个运行都保留首触速度与终止原因。运行健康检查触发、达到用户指定步数或者比较未通过时，命令返回非零退出码；已生成的记录可独立判读。

从出版社下载更新版正式 PDF 后，可独立恢复参考 CSV。运行环境需要 Poppler 的 `pdfimages`。

```bash
pdfimages -f 9 -l 9 -j paper.pdf "$out/fig"
python -m benchmarks.water_entry.single_drop.digitize "$out/fig-000.jpg" "$out/reference.csv"
cmp "$out/reference.csv" benchmarks/water_entry/single_drop/fig5_experiment.csv
```

[本轮运行记录与结果图](../../../docs/audits/spherical_entry_20261118/README.md) · [二维入水案例说明书](../../../docs/benchmarks/droplet_impact.md)

### 质量与体积的读数

自由面求解器在液体内部用分布函数密度表示质量，在界面单元另外保存液体质量。记录中的 `volume_drift` 沿用库函数口径，等于归一化的「液体内部密度和 + 界面填充率和」漂移。对照修正组另外记录 `geometric_volume_drift`（全域填充率和）与 `represented_mass_drift`（液体内部密度和 + 界面存储质量）。程序在早期越过几何体积限值后仍可留下诊断轨迹；最终体积判据独立核查。运行到论文测量时刻只说明能计算该时刻的误差。

### 历史探索

[m27100 · RR 剂量稳定性](../../../development/audit/drop_entry/cmb5_hydrorr_stability/README.md) 归档了三条接触前触发健康限值的探索轨迹及一条短程对照。正式实验比较使用本页的筛查与验收条件。
