# W26 · Ghia (1982) 锚核销 + 解析四例（D3Q19-BGK 消费面）

任务：`paper-exp-d3q19-bgk-analytic-fourcase-1`（TensorLBM 仓，runner=native）

## 交付物

| 文件 | 内容 |
|---|---|
| `anchor_register.json` | 主交付：Ghia 表 I/II 三 Re 逐项数值 + 逐项溯源（PDF sha256 / 页 / 行带 / 单元格 bbox / 逐字 OCR token / 墨水证据）+ 判据 C1–C4 逐条裁决 + 解析四例残差 + 内部校验 + 图 sha256 |
| `figures/fig1..fig6` | 锚剖面、符号见证聚类、守恒不变量、Poiseuille/Couette 与二阶斜率、Taylor-Green 场与残差、方腔 Stokes 极限模态 |

## 方法（零手抄）

1. 取原始论文 PDF（哈希 `1d76e463…19f2d93`，与仓内
   `benchmarks/verified/cavity/re5000/reference/ghia1982_tables_I_II.ocr.txt` 头部记录的 PDF sha256 一致）。
2. **数字**来自 PDF 文本层逐词包围盒（PyMuPDF）；**符号**不信任 OCR（它把印刷负号读成
   `a/AI/Al/xJ/X)/Jx/4`），改由页面栅格（8 px/pt）逐格量墨水宽度判定：值列右对齐，
   负号多出一个减号字宽 ⇒ 每列裂成 27.0 pt / 31.5 pt 两簇，阈值 29.3 pt，6 列全部干净可分。
3. 解析四例用闭式解回代算子算残差（mpmath 高精度求导 / FFT 谱导数），每例配"故意写错的解"作对照，
   证明残差检查有判别力。

## 复现

```bash
curl -s -H 'Accept: application/vnd.github.raw' -o /tmp/ghia1982.pdf \
  https://api.github.com/repos/nilot-pal/Lid-driven-cavity/contents/ghia1982.pdf
sha256sum /tmp/ghia1982.pdf        # 1d76e4636d24d0fad41c7bdb5ed383e9b224b0bfe3aa0317ce7dfc50619f2d93
GHIA_PDF=/tmp/ghia1982.pdf python3 scripts/w26_ghia/extract_ghia_tables.py --out /tmp/cells_raw.json
python3 scripts/w26_ghia/analytic_fourcase.py --out /tmp/analytic.json
GHIA_PDF=/tmp/ghia1982.pdf python3 scripts/w26_ghia/build_anchor_register.py
python3 -m pytest tests/test_w26_ghia_anchor_register.py -q
```

PDF 受版权保护，不入库；建库脚本缺 PDF 时明确报错，测试自动 skip。

## 结论速览

* C1 逐字锚位：**PASS**（表 I/II × Re=100/400/1000 × 17 行 = 102 个值，全部双见证）
* C2 解析四例残差 ≤1e-12：**PASS**（最差 1.84e-14；对照项 0.495 / 2.0 / 0.030 / 25.1）
* C3 网格分辨与 Re 口径：**PASS**（129×129 线网格、Re=U·L/ν、半程反弹 N_eff=N−1 的 1/N 偏置已登记）
* C4 内部一致性：**PASS**（1/128 点阵、右对齐、符号双簇、双见证 84/102 一致 0 冲突）
* 附带缺陷登记：仓内既有 `lid_driven_cavity/reference_data.py` 与论文表不符（u(y=1)=0 违反盖板边界条件，
  逐点最大差 1.0，v 差 0.26–0.46），消费者应改用本锚。