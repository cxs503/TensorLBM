# 参考提取记录 — NACA0012 α=5° Re=1000 平均升力系数 Cl

Agent W3-D, 2026-09-20。任务: 从 Kurtulus (2015) 提取精确 mean Cl(α=5°, Re=1000)，
至少一个独立交叉源，两源差异如实报告（>3% → 预判「参考受限」）。

## 1. 主源

Kurtulus, B. (2015), "On the Unsteady Behavior of the Flow Around NACA 0012
Airfoil with Steady External Conditions at Re=1000", *International Journal of
Mathematics and Computers*?? — 实际期刊: DOI 10.1260/1756-8293.7.3.301 →
*International Journal of Aerodynamics*?? 权威著录: **Int. J. of Micro Air
Vehicles* 7(3):301-326** (Multi-Science). 全文 PDF: open.metu.edu.tr
(open access), 本地 `/root/w3d_ref/kurtulus2015_local.pdf` (2,858,846 B, %%EOF 完整;
URL 中冒号须 %3A 编码; 服务器端 wget 会截断, 一律本地下载后校验)。

**论文无任何 Cl 数值表** (全文 pdftotext -layout 核查: 唯一数值表 = Table 1 网格数).
唯一平均-Cl-vs-α 数据呈现 = **Figure 4** (PDF 第 7 页, 双 panel:
(a) mean Cl ± min/max 误差棒, (b) mean Cl + 文献对比)。

原文关键句 (逐字):
- "results are obtained with 1° increments from α=0° to α=41°"
- "mean values are calculated in the interval of t*∈[73 146]"
- "Figure 4 presents the mean lift coefficient (Cl) distribution versus angle
  of attack for NACA 0012 airfoil"
- "Figure 4a shows the minimum and maximum amplitude values ... as error bars"
- "amplitude of Cl is zero until 8°"  (→ α=5° 为稳态, 无脱落, 与本基准选型一致)
- "the unsteady vortex pattern ... starts at 8°" / "Below 7°, no alternating
  vortices are observed"

## 2. 数字化方法 (Figure 4a, fig4-000.png 960×446 RGB 原生位图)

- 提取: `pdfimages -f 7 -l 7 -png kurtulus2015_local.pdf` (页面定位先由
  pdftotext 分页 grep "Figure 4" 标题句确定)。
- 轴定标 (机器验证, 非目测):
  - x 轴基线 y=376.5; y 轴 x=80.5; x 刻度 80.5..445.5, **61.0 px = 10°**;
  - y 刻度 376.5..10.5, **36.6 px = 0.2**; 刻度标签经 8× LANCZOS 放大逐字核对
    (0.0@轴, 0.2, 0.4, 0.6, ..., 2.0/2.2@顶; 各核对 ≥2 次一致)。
  - 换算: **Cl = (376.5 − y_px) / 183.0**; panel (b) 独立定标 Cl=(375.5−y)/183.0,
    α=(x−533.5)/6.05, 两 panel 结果一致。
- 读数: 每整度列的暗游程 (dark-run) 分析 + 方形标记 (≈11 px) 中心;
  误差棒只用于包络, 不入读数。

### 数值 (Figure 4a 直读, panel a/b 双确认)

| α° | Cl (直读) | 备注 |
|----|-----------|------|
| 0  | ≈0.007-0.01 | 外推 (对称翼, 应为 0) |
| 2  | 0.055      | |
| 4  | 0.109      | |
| 5  | **0.131**  | 连接线直读; 4/6° 插值 0.134 |
| 6  | 0.158-0.161| |
| 8  | 0.202      | (此起误差棒开始张开) |
| 10 | 0.243-0.245| |
| 12 | 0.279      | |
| 15 | 0.306      | |
| 20 | 0.407      | |
| 30 | 0.699      | |
| 40 | ≈0.91      | 误差棒 0.68-1.18 |
| 50 | ≈1.13      | |
| 高处散点簇 cy 43-46/71-75 | 1.65-1.83 | Clmax 包络帽, 非平均线 |

**锁定参考值: Cl_ref(5°) = 0.132 ± 0.015** (直读 0.131 / 插值 0.134 的中心;
不确定度来自 36.6px=0.2 → 1px≈0.0055, 标记半径 ~5px → ±0.015 合理)。

## 3. 独立交叉源 1: Di Ilio et al. (2020), arXiv:2006.10487

HLBM 模拟 NACA0012 Re=1000, α=0-30°, 其 **Figure 10** (PDF 第 10 页) 绘
present study + Liu et al. + **Kurtulus** (方点) + Khalid & Akhtar。

- 定标 (矢量级, pdftotext -bbox 给出标签字形精确坐标, 200dpi 渲染 2339×1654):
  y 标签 1.4@475.5px, 1.2@552.0, 0.8@704.9, 0.6@781.4, 0.4@857.9, 0.2@934.4,
  0@1010.9 → **76.5px = 0.2 → Cl = (1010.9 − y)/382.5**;
  x 标签 0@468.5, 5@602.4, 10@736.2, 30@1271.7 → α = (x−468.5)/26.77。
- **结果: Di Ilio 图中转绘的 Kurtulus 方点逐点等于本文 Figure 4a 数字化值**
  (其图 "2°"=0.108 ≡ K(4°)=0.109; "4°"=0.202 ≡ K(8°); "6°"=0.277 ≡ K(12°)=0.279;
  "8°"≈0.32 ≡ K(16°)≈0.325; "10°"=0.413 ≡ K(20°)=0.407; "12°"=0.497 ≡ K(24°)≈0.49;
  "15°"≈0.57-0.68 ≡ K(30°)=0.699; "20°"≈0.88-0.96 ≡ K(40°)≈0.91)。
  注意: Di Ilio 把 Kurtulus 的稀疏 α 网格点摆到了自家 α 轴位置上 (横轴压缩 2×),
  属转绘摆放问题; 纵向数值吻合本身**证实了本文对 Figure 4a 的提取正确**。
- Di Ilio 自家 HLBM (present study): Cl(4°)≈0.22-0.23, Cl(6°)≈0.26-0.28
  → **Cl(5°) ≈ 0.24** (5° 刻度列游程 0.227-0.264 之中点)。

## 4. 交叉源 2: Kurtulus 论文自身文字锚 (内部一致性审计)

- "the lift curve slope Clα is equal to π at α=0° which is half of the
  linearized lift curve slope of 2π" → 若按 π/rad 线性到 5°: Cl(5°)≈0.27,
  是其自身 Figure 4 值的 **2.07×**。
- "mean (Cl/Cd) ... maximum value of 2.55 at 11°" + "Drag coefficient and
  zero angle of attack is found to be about 0.12":
  Cd(11°)≈0.13-0.15 (由 Cd(0°)=0.12 缓升) → Cl(11°)=2.55·Cd≈0.33-0.38,
  为 Fig4 值 0.262 的 1.3-1.45×。
- Figure 7 (Cd 面板) 数字化: TL/TR 面板 Cd(0°) 位置与文字 0.12 一致 (量级),
  Cd(90°)≈2.8 与文字完全一致 → 论文 Cd 为常规约定。**图文自不一致集中在 Cl 的
  y 轴**: Figure 4 的 Cl 轴疑似少标 2× (图 0.131 vs 文字隐含 0.26-0.27)。
  (Figure 7 各子图布局为 2×2 嵌入位图, Cl 面板识别不完全确定, 此条为辅助证据。)

## 5. 结论与预注册判定

- **锁定 (判据用): Cl_ref = 0.132 ± 0.015** — 来自任务指定主源 Kurtulus 2015
  Figure 4a 的数字化 (论文无表值)。
- **披露列: 0.26** — Kurtulus 文字斜率锚 (π/rad) 与 Di Ilio 自家 HLBM (0.24)
  所隐含的常规约定值 (两者一致, 也与 Liu/Khalid 量级一致)。
- **两源差: 0.132 vs 0.24-0.27 → 散布 ~85% >> 3% → 预判「参考受限」**。
  形式判据 (Cl ≤3% + 单调) 以 0.132 计算; 旧档 c60/c90 (0.2326/0.2488) 若复现,
  对锁定参考误差 +76~89%, 对披露参考 -4~-10% — 两列都报, 不选购。

## 6. 工件

- /root/w3d_ref/kurtulus2015_local.pdf, k2015.txt (pdftotext -layout 全文)
- /root/w3d_ref/fig4-000.png (Figure 4 原生位图) + 标签放大图 glab_*/lab_*/xlab_*
- /root/w3d_ref/diilio2020.pdf, dpg-10.png (第 10 页 200dpi), diilio.txt
- 服务器 staging: /nfs/wangxi/runs/bm_widen_w3_20260920/naca_cl/
  (staging/ref_pdf/ 内早前传输截断的 PDF 已重传完整版, 以字节数+%%EOF 为准)
