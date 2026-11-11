# suboff_anchor_w22 — W22 DTRC SUBOFF AFF-8 外部锚抄录 + ITTC-1957 闭合检验

任务：`paper-exp-lbm-suboff-tau-w22-anchor8`（mission 14504，TensorLBM，学科域：阻力）
前身：`paper-exp-lbm-suboff-tau-4`（零报告欠账，本单显式承接其判据）

## 总判决：HONEST-FAIL

| 判据 | 阈值 | 量值 | 裁决 |
|---|---|---|---|
| C1 锚件抄录完整率 | 100% | 100.0000%（2342 叶 / 8 源 / 回读失配 0） | PASS |
| C2 ITTC-57 C_F 复算 vs 锚件记录（逐档） | <1% | max 5.412e-03%（记录值舍入量级） | PASS |
| C2'（披露行，非判据）案例 Re 口径冲突 | — | +78.592% | 披露 |
| C3 三档 $C_T$ 闭合残差 | <3% | −144.08% / −130.91% / −122.05%（主口径 A） | **FAIL** |
| C4 k 双口径差异 | <5% | $k$ 基 +17.595%（$(1+k)$ 基 +3.577%，同时披露） | **FAIL** |

归因（完整见报告 §5/§8）：ITTC-1957 是湍流相关线（$Re=10^2$ 有极点，$Re=10^3$ 处值 0.075），
本案是 $Re=10^3$ 的层流算例（参考为 Blasius），AFF-8 的形式因子记录于
$Re\approx1.17\times10^7$；两个因子来自不同 Re 区间，**跨 Re 的相似性换算式闭合不可能成立**。
把 $C_W$ 放开也无解：闭合所需的 $C_W$ 三档全为负（−0.0556 / −0.0534 / −0.0517），
物理上非负的兴波阻力无法闭合该恒等式。

## 工件

| 文件 | 内容 |
|---|---|
| `anchor_transcription.json` | 6+2 个源工件的全标量叶逐字段抄录：RFC-6901 指针 / 文档序指针 + 逐字符定位 + 源 sha256 + 独立回读记录 + 完整率普查 |
| `ittc57_closure.json` | AN-ITTC57 三路复算与逐档对照、案例 Re 口径冲突、三个闭合口径 + 一个定义性口径、k 双口径与 Prohaska 拟合/敏感性、判据逐条裁决、归因 |
| `figures/` | 4 张图 + `manifest.json`（每图 sha256 与数据指针清单） |
| `build_anchor_transcription.py` | 抄录/回读生成器（`W22_TRANSCRIPTION_OUT` 可改输出路径） |
| `build_ittc57_closure.py` | 复算/闭合/Prohaska 生成器（阈值预声明于 `CRITERIA_PREDECLARED`） |
| `make_figures.py` | 出图 + manifest |
| `build_report.py` | 生成 `papers/reports/w22_lbm_suboff_anchor8.md`（数字全由工件注入） |

上游锚源：`benchmarks/verified/suboff_anchor/anchor.json`、
`benchmarks/pending/suboff_highre_resistance/refs_lock.json`；
被检案例：`benchmarks/verified/suboff_sail_resistance/`。

## 复现

```bash
python3 benchmarks/verified/suboff_anchor_w22/build_anchor_transcription.py
python3 benchmarks/verified/suboff_anchor_w22/build_ittc57_closure.py
python3 benchmarks/verified/suboff_anchor_w22/make_figures.py
python3 benchmarks/verified/suboff_anchor_w22/build_report.py
python3 -m pytest tests/test_suboff_anchor_w22.py -q     # 13 项不变量
```

## 边界

只产实验工件与结论，不写论文正文；不修改平台仓；所有数字由工件注入，禁手抄。
外部锚全部为仓内已归档文本的二手转录（离线环境无法读原报告），该限制由上游
`suboff_anchor/anchor.json :: external_evidence.offline_disclosure` 承载。