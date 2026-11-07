# sphere_settling_gcs — G15 第三法：动能通量控制面（pending）

**状态：见 `result.json` 的 `verdict`（本页只写口径与文件索引，数字一律不抄写，
以 `result.json` / `docs/sphere_settling_gcs_20261107.md` 为准）。**

## 这是什么

G 账本两条已失效力路线（G12 壁面压力/摩擦积分、G15 逐链 MEM）之外的**第三类
方法**：在**解析球面**上积分**原始二阶矩动能通量** `Π_ij = Σ_q f_q c_qi c_qj`，

```
F_body = f·V_f^in − ∮_S Π_ij n_j dA          （定常；f = 体力密度）
```

仪器：`src/tensorlbm/force_control_surface.py`；验收测试：`tests/test_force_control_surface.py`
（10 项解析测试，无时间步进）。

## 与前两法的关系（方法学差异，不是参数微调）

| | G12 压力/摩擦 | G15 MEM | **本方法** |
|---|---|---|---|
| 被积量 | `p n` + `τ_w·n` | 逐链 `2c_q f_q` | 完整动能通量 `Π·n` |
| 采样位置 | 体素壁面 | 壁面链（或湿节点） | **离体素面 R+2…R+3 的解析球面** |
| 自由流背景 | 需 `p0` 处理（实测为 no-op） | 加性背景未抵消（+264.6%） | **精确零空间**（`∮n dA=0`），恒等为 0 |
| 阶梯几何误差 | 直接继承 | 直接继承 | 不接触壁面（仅经流场间接） |
| 自检能力 | 无 | 无 | 控制面半径无关性 + 精确动量平衡锚点 `F=ρgV_f` |

## 物理算例

周期立方槽 + 单个体素球固定，流体受均匀体力 `ρg`（Guo 格式）。稳态时流体
漂移速度**就是**同重力下该球的沉降终速（球静止参考系下的沉降问题），唯一被测量
是力。参考值：无界 Stokes `F = 6πμaU` / `U_s = 2a²Δρg/(9μ)`。

三项必答验证面：

1. **终速 vs 解析 Stokes**：`λ=R/L → 0` 外推（约束效应）+ `1/R → 0` 外推（体素表示）；
2. **瞬时力振荡幅度**：尾窗 `ptp/mean`、`rms/mean` 定量报告；
3. **方法边界**：Re 区间扫描（0.02–0.20）、最小网格比 `R≥4`、控制面间隙、以及
   求解器 `τ` 上限（本库反弹边界在 `τ≳5` 出现偏高的迁移率，见报告 §5.1）。

## 文件

| 文件 | 作用 |
|---|---|
| `prereg.md` | 冻结的判据/参数/判定规则（先于正式算例） |
| `run.py` | 生产驱动：`domain` / `refine` / `reynolds` / `anchor` |
| `probe.py` | 前置探针（τ 依赖、锚点标定、`τ=14.9` 反例） |
| `merge_results.py` | 汇总 `out_*.json` → `result.json` + 判据 PASS/FAIL |
| `make_figures.py` | 报告图（全部从 `result.json` 读数生成） |
| `make_report.py` | 生成 `docs/sphere_settling_gcs_20261107.md`（数字程序化注入） |
| `plot_field.py` | 稳态中截面场快照（`field_snapshot.json`） |
| `out_*.json` / `result.json` | 机器输出（唯一数字来源） |
| `log_*.txt` | 运行日志（含每千步收敛轨迹） |

## 复现

```bash
cd benchmarks/pending/sphere_settling_gcs
python run.py domain   --out out_domain.json
python run.py refine   --out out_refine.json
python run.py reynolds --out out_reynolds.json
python run.py anchor   --out out_anchor.json
python merge_results.py && python make_figures.py && python make_report.py
python -m pytest ../../tests/test_force_control_surface.py -q
```

单核级算力（128 核共享机，每算例 2–20 分钟；`ν=1`、`τ=3.5`、`Re≤0.2`、
`Ma≤1.1e-2`）。

## 诚实声明

未做移动边界（真沉降）、未做 `Re≳1`、未修库内反弹边界的大 `τ` 缺陷（后者是本
轮**量化发现**，但属另一单）。任何未通过判据按 `prereg.md` §5 原样记录，不
改判据、不覆盖。