# `examples/guide_gaps` — 三个 guide 缺口（m21096）的可复算证据包

覆盖 `dam_break_sc`（斜置坝轴 / Stoker 间断）、`forchheimer`（多孔介质惯性修正）、
`kelvin_helmholtz`（无粘线性稳定性）三个长期 `escalated` 的 guide 缺口。
每个缺口 = 一个真求解器 + 一个可机检参考 + 至少两条断言（解析比对 + 退化/收敛保护）。

| 文件 | 作用 |
|---|---|
| `swe2d.py` | 二维浅水有限体积求解器（MUSCL+minmod / HLL / SSP-RK2）+ Stoker (1957) 闭式解 + 斜向平面平均 |
| `dam_break_sc.py` | 缺口 1 驱动：θ 扫描、前缘/涌高/伪切向速度、质量收支，写 `artifacts/guide_gaps/dam_break_sc.json` |
| `brinkman_forchheimer.py` | Darcy-Forchheimer-Brinkman 求解器族（解析 bulk 根、细网格 Newton 参考、粗网格 2D FV、Ergun 关联式） |
| `forchheimer.py` | 缺口 2 驱动：压降—流速曲线拟合、K/β 反演、Darcy 退化，写 `artifacts/guide_gaps/forchheimer.json` |
| `kelvin_helmholtz_solver.py` | 两层线性稳定性：Chebyshev 配点 + 界面条件 → ω 的二次特征值问题 |
| `kelvin_helmholtz.py` | 缺口 3 驱动：ω_i(k)、谱收敛、表面张力截止、退化支，写 `artifacts/guide_gaps/kelvin_helmholtz.json` |
| `make_summary.py` | 汇总三缺口 → `artifacts/guide_gaps/summary.json`（含 `<15%? <3%?` 布尔与参考来源串） |
| `make_report.py` | 生成图（`artifacts/guide_gaps/figures/*.png`）与 `docs/guide_gaps/REPORT.md` |

```bash
for c in dam_break_sc forchheimer kelvin_helmholtz; do python3 examples/guide_gaps/$c.py --quick; done
python3 examples/guide_gaps/make_summary.py
python3 examples/guide_gaps/make_report.py
python3 -m pytest tests/test_guide_gaps.py -q
```

依赖 numpy / scipy / matplotlib（无 GPU、无 torch）。`--full` 加密网格，门禁用 `--quick`。
