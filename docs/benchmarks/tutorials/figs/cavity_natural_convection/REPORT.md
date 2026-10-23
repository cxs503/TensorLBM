# m219 交付报告 — cavity_natural_convection 论文级教程（重算）

## 1. 交付物清单

| 类别 | 路径 | 说明 |
|---|---|---|
| 教程正文 | `docs/benchmarks/tutorials/cavity_natural_convection.md` | 220 行，八节结构，中文正文 + 英文副题，5 图 5 表 |
| 图件 | `docs/benchmarks/tutorials/figs/cavity_natural_convection/fig{1..5}_*.png` | 全部由本 mission 自跑 `data/` 生成 |
| 图件生成脚本 | `.../plot_figures.py`、`.../derived_stats.py` | 从 `data/*.npz`、`data/scan_result.json` 读数据出图 |
| 重算脚本 | `.../rerun_scan.py` | 复用入库 `benchmarks/verified/cavity_natural_convection/driver.py`，物理全在库入口 |
| 自跑产物 | `.../data/` | `scan_result.json`、`result_n{64,128,256}.json`、`fields_n*.npz`、`derived.json`、`run.log`、`DONE.json` |
| 判据工件 | `.../data/validation_report.json` | 8 项检查（C1–C8）全 pass，机器可读，验收器直读 |
| 正文渲染脚本 | `.../gen_tutorial.py` | 教程全文由该脚本从数据渲染（零手抄） |
| 出版件 | `docs/benchmarks/tutorials/files/cavity_natural_convection.{docx,pdf}` | 由 `.../gen_publish.py` 从 md 渲染，5 图内嵌、6 表 |
| 索引 | `docs/benchmarks/tutorials/README.md` | 新增一行，案数 32 → 33 |

## 2. 计算事实（全部来自本 mission 自跑产物）

- 环境：NVIDIA GeForce RTX 5090，torch 2.11.0+cu128，fp32；thermal 库指纹 `sha256[:12] = 598889356088`（三档同）。
- 参数：Ra=1e4、Pr=0.71、τ=0.6（ν=1/30、α=0.046948），H = n−1，Guo 浮力 + D2Q5 温度场，四壁半程反弹、上下绝热，`mask_wall_u=True`。
- 三档网格 n = 64 / 128 / 256；步数 32000 / 115500 / 457600；耗时 30.5 / 110.3 / 437.7 s（合计 578 s）。
- 主判据：Nu = 2.109792 → 2.177843 → 2.212536，对 de Vahl Davis (1983) 参考 2.243 的误差 −5.939% → −2.905% → −1.358%（严格单调下降，过 3% 门档数 = 2/3；n=64 如实不过门）。
- 次判据：u_max 15.8574/16.0204/16.0821（参考 16.178），v_max 18.9629/19.2896/19.4836（参考 19.617），n≥128 全部 ≤3%。
- 诊断：三点 Richardson 观测阶 p = 0.972（≈一阶，与 grad1 壁面差分口径一致），外推 Nu∞ = 2.2486（对参考 +0.250%）。
- 与入库档案 `benchmarks/verified/cavity_natural_convection/result_rerun.json` 逐位一致：ΔNu、Δu_max、Δv_max 全为 0.0e+00，步数与稳态触发点全同（C2 pass）——**本次是真重跑，不是引用旧档案**。

## 3. 口径与局限（教程第 6 节已逐条披露，此处仅索引）

1. H = n−1 约定（等温壁落在节点上），driver 覆写库内 H=n 的 gβ 反解；
2. Nu 差分口径 grad1（壁节点−首内邻一阶），是 Nu 误差全为负的机制，非判据调参；
3. `mask_wall_u=True` 为非物理自由度（壁环宏观 u 置零），去掉则稳态不达成；
4. 参考值是基准数值解非解析解，自身带 0.1–0.3% 量级不确定度；
5. 仅单一 (Ra, Pr) 点；fp32，n=256 档 gβ=9.44e-07 接近 ULP 量级（未做 fp64 A/B）；
6. n=64 档 |Nu 误差| 5.939%、|v_max 误差| 3.334% 超门，如实报告为粗网格离散误差。

## 4. 复现命令

```bash
cd docs/benchmarks/tutorials/figs/cavity_natural_convection
python rerun_scan.py --out-dir data --grids 64 128 256 --device cuda:0
python derived_stats.py && python plot_figures.py && python gen_tutorial.py && python gen_publish.py
```

## 5. 验证记录

- `python -m pytest -q tests -k "docs or tutorial"` → 7 passed, 1 skipped。
- `python scripts/check_platform_docs_consistency.py` → `[docs-consistency] OK`。
- 图件可重生成性：重跑 `plot_figures.py` 后五张 PNG 的 md5 与入库版本逐位一致。

## [待办]（留给后续班次）

1. **正式档未降级**：本教程三档均为正式档参数（与入库档案同协议），不存在降档冒充问题；但耗时（578 s 单卡）依赖 GPU，CPU 环境需按 README 备注放大约一个量级。
2. **参考值第二来源未交叉**：Barakos et al. (1994) 的有限体积解已在参考文献列出但未取数值逐点比对，可作后续增强。
3. **docx/pdf 渲染器为本案自带**（`gen_publish.py`），仓库内其余 32 案出版件为历史产物、无统一生成脚本；若后续要统一出版链，可把本脚本提升为公共工具。
4. **PR 未创建**：平台禁用 `git push`，独立分支 `agent/m219` 已本地提交完整交付，merge 留 owner。