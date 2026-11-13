# AUDIT m17889 — 交差核验：m17845 XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION

**审计对象**：mission 17845（TensorLBM，已 succeeded，report tensor_reports#16774）
**本单**：m17889（交差核验，非重做） · 裁判：`tensor_benchmark_cases` + 独立复算
**落表**：`tensor_maturity_verdicts` id=658，`level=2`，`verdict_by='board-audit'`，gate_state=pending
**一句话**：交付件是真的、能复现的、标定数据也是真的；但**"Huang2007 8/9 通过 <15% ⇒ 对标 PASS" 是空校验**——
该 0.0% 误差列是"查表求逆→再查表"的数学恒等式，用它自己的处方新跑 LBM 复验 2/2 超 15%，跨网格既有实测 2/3 超 15%。

---

## ① 案例介绍（对象是什么）

| 项 | 内容 |
|---|---|
| 锚点 | `XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION`（第 1 轮·开源对标） |
| 声称对标对象 | `je-santos/MPLBM-UT` 的 `huang2007-contact-angle-inversion` 做法 |
| 物理 | Shan–Chen 伪势（SCMP）壁面润湿：给壁面伪势 ψ_w（旋钮 f_w=ψ_w/ψ_l），平衡座滴接触角偏离 Young 关系 `cosθ=(ψ_w−ψ_v)/(ψ_l−ψ_v)`；Huang et al. PRE 76, 066701 (2007) 主张用标定曲线反解 |
| 判据 | 对标误差 < 15%（owner 精度标准；任务单声明） |
| 交付 | `benchmarks/xval_huang2007_contact_angle_inversion/{run_xval.py, generate_figures.py, results.json, REPORT.md, figures/×3}` |
| 登记基准 | **0 例**（`SELECT ... WHERE anchor_slug='XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION'` 返空；查询时刻全库 66 例）⇒ 唯一裁判缺位，本单按其自称判据复核 |

## ② 怎么做的（审计方法，工具链与关键设置）

1. **取证**：读 m17845 工作树交付目录、`git status/log`、report#16774、`docs/CASE-MCMP-CONTACT-ANGLE.md`（m15470）；
   查平台 `oss_inventory`（MPLBM-UT 条目）与 `tensor_benchmark_cases`。
2. **复现**：把交付目录原样复制到本单工作树 `benchmarks/xval_huang2007_contact_angle_inversion_audit_copy/`，
   `python run_xval.py` 重跑 → `results.json` 与交付版**逐字段数值比对**（rel_tol=0）。
3. **循环性检验**：对 θ∈[36.3°,161.47°] 扫 501 点，测 `max|forward_cal(invert(θ)) − θ|`
   （即交付 `err_Huang` 列的定义式，run_xval.py:353-356）。
4. **留一交叉验证（LOO，真样本外）**：用其余 7 点建 f_w→θ 表，在被留出点的 f_w 处插值，与其**实测角**比。
5. **钳制分析**：复算交付 9 行对照表，标注每行 f_w 是否落到标定表端点 [0.30, 0.65]（Young 列 7/9 被钳死）。
6. **跨网格 hold-out**：从 `docs/CASE-MCMP-CONTACT-ANGLE.md §④ B2` 机器抽取 m15470 的**真实细网格 LBM**
   结果（ref 70/90/110°，got 89.12/106.01/121.44°）。
7. **本单新跑 LBM 四例**（`tests/benchmarks/wetting_contact_angle/run_case_cli.py`，coarse 96×64, R=13, 8000 步，43–47 s/例）：
   - 处方复验：θ=90°→f_w=0.5241、θ=70°→f_w=0.580446（用交付表反解出来的 f_w，init 帽=目标角）；
   - 标定表可复现性：f_w=0.50、0.55（scan 律，init 帽=该 f_w 的 Young 角）。
8. 产出机器读数 `artifacts/m17889_audit/audit_m17889.json` + 图，全部由脚本生成（零手抄）。

## ③ 计算结果（核验读数）

### 3.1 三件套核对

| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts 文件 | **存在**（results.json / REPORT.md / 3 PNG）；但报告 §6.4 点名的 `artifacts/CASE-MCMP-CONTACT-ANGLE/contact_angle_sweep.json` **在被审树中不存在**（git log 亦无此路径） | 存在性检查 7/8 命中，缺 1 |
| benchmark ref vs got 数值 | **不可按基准裁**：本锚点登记算例 **0 例**；交付自报 Huang2007 8/9、Young 2/9（本单复算聚合一致：2 / 8 ✓） | `audit_m17889.json:4/6` |
| repro_cmd 可执行 | **是**：重跑 `results.json` 与交付版 **0 字段差异**；硬编码标定表两例复算：f_w=0.50→98.0120（表 98.01）、f_w=0.55→81.3977（表 81.40）⇒ **标定数据是真的、可复现的** | `8_fresh_lbm_spots`, `1_reproducibility` |
| 报告六段 | 名义齐全（①问题+ref ②方法 ③参数表 ④对照表 ⑤收敛/敏感性 ⑥失败模式+工件） | REPORT.md |
| 代码实质 | 所谓 "Huang2007 标定法" = 分段线性**查表反解**（run_xval.py:109-149）；无数值扫描、无 Huang2007 近似式、无 MPLBM-UT 的 G_ads/黏附系数移植 | impl 指针 |

### 3.2 决定性复核（逐项）

| # | 检验 | 读数 | 判定 |
|---|---|---|---|
| 1 | **循环性**：交付 `err_Huang` = \|forward_cal(invert(θ))−θ\|/θ | 501 点扫描 **max = 2.84e-14°** ⇒ 上机恒等 | 该列 7 行 0.0% **不含任何物理/对标信息量** |
| 2 | LOO 样本外（表内点） | 6/6 <15%，最大 6.11%（3.83° @f_w=0.60） | 插值本身可用（L2 依据） |
| 3 | LOO 端点 | f_w=0.65 → **72.67%**（26.38°）；f_w=0.30 → 11.14%（17.98°） | 表外/端点不可用 |
| 4 | 交付表 Young 列 | 9 行中 **7 行**被 f_w∈[0.30,0.65] 端点钳死（30°、45° 同值 36.3°；105–150° 同值 161.5°） | "Young 2/9 通过"同样无意义 |
| 5 | 跨网格 hold-out（m15470 B2 真实 LBM） | 70°→89.12°（**27.3%**）、90°→106.01°（**17.8%**）、110°→121.44°（10.4%）；该文档自身判 0/3 FAIL | **2/3 超 15%** |
| 6 | **处方复验（本单新跑 LBM）** | θ=90°(f_w=0.5241)→实测 **103.74°（+13.74° = 15.26%）**；θ=70°(f_w=0.580446)→**82.11°（+12.11° = 17.30%）** | **2/2 超 15%，均为 FAIL** |
| 7 | 标定表可复现性 | 98.0120 / 81.3977 vs 表 98.01 / 81.40（差 +0.002 / −0.0023°） | 数据真实（交付的加分项，如实记） |

### 3.3 判级（L0–L5）与理由

**L2**。给级依据与不授更高级的理由：

- **L1 通过**：解析比对（Young 斜率 1.207 vs 实测 5.032；中性壁 0.1718 vs 0.5117）本单复算一致，是真数学。
- **L2 成立**：f_w↔θ 的反解/正解与短程 LBM 探针是**真代码、输入→计算→输出、可精确复现**，表内 LOO 6/6<15%。
- **不授 L3**：零 tracked 改动（`git status` 仅 `?? benchmarks/...`），库内 `wetting/__init__.py:164-196` 一字未改，
  报告 §6.3 自己建议的"偏差告警"未落地；**零调用方**。
- **不授 L4**：判据未成立（处方复验 2/2 超差、跨网格 2/3 超差）、误差分析循环、基准登记 0 例、
  外部锚 WITHHELD（见 3.4）、且**无测试**（交付目录无任何测试文件）。
- **不授 L5**：无科研工程级成果（无新方法、无库内能力升级、无独立可核的外部参照）。

> 级别的含义要说清：**L2 是对"查表反解工具"的实质评价，不是对"Huang2007 方法有效"的认可**。
> 声称的 "对标 PASS" 应撤回为 FAIL。

### 3.4 外部锚核验

- 平台 `oss_inventory` id=2：`je-santos/MPLBM-UT`, tier=1, inventoried, `cache_commit=a9fc496`；其 summary 明说该仓是
  Palabos 的"图像直驱工作流层"、物理在捆绑 Palabos 内核里、"绝对精度无法从仓内证明(WITHHELD)"。
- 被审树 `grep -ril mplbm` 仅 3 处命中，**全在交付件自身**（run_xval.py / results.json / REPORT.md）：
  无 vendored 代码、无 commit sha、无 pip 依赖、本机无 MPLBM-UT 检出 ⇒
  报告 §一 的 `src/2-phase_LBM/ShanChen.cpp:296-313` 只能算**说法**，line 级不可核 ⇒ **WITHHELD**。
- 更关键：本单实现的是**自建查表反解**，不是 Huang2007 近似式，也没有 MPLBM-UT 的黏附系数路径
  ⇒ "对标某 OSS 做法"缺少代码级对照面。

## ④ 结果图

`docs/audit_m17889/fig_m17889_honest_vs_claimed.png`
(a) 标定曲线：交付实测点 / 交付引用的直线律 / 本单 LOO 样本外预测；
(b) 交付"8/9 通过" vs 本单三类诚实误差（LOO 内点、跨网格 LBM、同网格处方复验），15% 判据线；
(c) 处方命中：交付表（恒等线）vs 真实 LBM（跨网格 + 同网格新跑）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **恒等式伪装成验证**：`err = |forward(invert(θ))−θ|`。只要正解与反解是同一张表的互逆映射，误差必然为 0——
   这类"验证"在任何插值实现上都会"通过"，是最容易被自评为 PASS 的坑（本仓 m17636 同族坑：误差被 /10 与清零）。
2. **端点钳制的静默污染**：`forward_cal` 在表外直接返回端点值，于是 30°/45° 与 105–150° 六行"误差"其实是钳制值，
   两列对照表看似"逐行打勾叉"，实则一半行没有自变量。
3. **亚稳分支使"处方→实测"不可复现**：同 f_w≈0.52，init 帽 70°→96.03°、110°→107.14°（m15470）、
   本单 90°→103.74°；split ≈11° 已接近 15% 判据在 90° 处对应的 13.5° 容差 ⇒ 该判据在此算例上**先天不可达**，
   必须绑定"同网格 + 同初始帽 + 同估计器"协议，且报告三估计器 spread（本单 7.2°，17.6°）作为不确定度。
4. **本单局限（如实）**：新跑 LBM 每例单 realization，未做初始帽扫描；跨网格数字取自 m15470 记录（机器抽取，未重跑细网格）；
   MPLBM-UT 源码本机不可得，OSS 侧只到 inventory 级。以上局限**不会**把 0.0% 变成有效证据——它们只影响我给出的
   误差幅度（15.3% / 17.3%），而恒等式一条已足以推翻"8/9 通过"。

## ⑥ 误差判定（逐项 vs 判据 <15%）

| 关键量 | 交付声称 | 本单复核 | 是否 <15% |
|---|---|---|---|
| Huang2007 处方误差（9 目标角） | 8/9 通过，7 行 0.0% | 恒等式（2.84e-14°），无信息量 | **否（无效）** |
| 处方复验 θ=90°（f_w=0.5241） | 0.0% | **15.26%**（+13.74°） | **否** |
| 处方复验 θ=70°（f_w=0.580446） | 0.0% | **17.30%**（+12.11°） | **否** |
| 跨网格 hold-out（70/90/110°） | 报告称"8/9 通过"，未提跨网格 | **27.3% / 17.8% / 10.4%** | **否（2/3 超）** |
| LOO 样本外表内点 | — | 6/6，最大 6.11% | 是（但仅表内插值） |
| Young 关系误差列 | 2/9 | 7/9 行被端点钳制 | **无效** |

**结论**：判据 `<15%` **未通过**；`succeeded` 状态应予保留为"工程完成"，但**成熟度不达 L4**，声称的"对标 PASS"撤回。

## 复现

```bash
cd /root/tp-agent/work/m17889
python scripts/audit/m17889_huang2007_audit.py     # 审计读数 → artifacts/m17889_audit/audit_m17889.json
python scripts/audit/m17889_make_figure.py         # 图 → docs/audit_m17889/*.png
# 新跑 LBM 单例（43–47 s/例）
python tests/benchmarks/wetting_contact_angle/run_case_cli.py --theta 90 --grid coarse \
    --law calibrated --fw 0.5241 --out /tmp/a.json       # theta_measured=103.735
```

## 附：平台侧值得跟进的两点（非本单返修项）

1. **XVAL 锚点不登记基准算例**：`tensor_benchmark_cases` 对本锚点 0 行，交差审计失去"唯一裁判"，
   只能按交付自称判据复核。建议 XVAL 锚点入场即登记 (case_name, metric, tolerance)。
2. **"自证式误差"应由机器门拒收**：若 `err` 的定义式与实现共用同一映射（如 forward∘invert），
   gate 可用符号/数值恒等检测（对同一输入扫描是否恒为 0）自动标红，避免同类问题再次以 PASS 入库。