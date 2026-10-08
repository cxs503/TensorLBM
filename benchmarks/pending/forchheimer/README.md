# Forchheimer：周期方阵圆柱有限 Re 曳力（pending）

**状态：❌ 未达标（根因类：参考受限）——三档 err_A 1.86–3.06% 平坦系统性偏低、对 d 无收敛趋势，严格单调条款三档全破；偏移约半可由参考侧双源带宽吸收（K&L97 转绘系 vs FDM 系互差 1.0–1.6%），总判 FAIL。**

## 物理问题

周期方阵圆柱单胞（固相分数 φ_s=0.40，掩码实测 0.400208）在有限 Reynolds 数
（弱惯性 / Forchheimer 区）的曳力增量 R(Re) = y(Re)/y_Stokes——
verified/permeability（斯托克斯极限，2D 周期方阵）同族几何的有限 Re 延伸。

- 参考（图级数字化，降级披露）：**Koch & Ladd 1997**（JFM 349:31–66，原文闭源），
  经 **Forslund et al. 2023**（Transp. Porous Med. 148:545–569，开放获取）
  Fig. 6 (top) 矢量提取（`extract_ref_fig6.py`，× 字形两对角线交点=包围盒中心逐点几何验证）
  ——系列 A（主判据，K&L97 转绘）与系列 B（副列，作者自研 FDM）。
  主判据用 R 比值形式，对任何仿射归一化免疫。

### 共性模块

- D2Q9 `collide_bgk` → `stream` → 体力（库 `_apply_body_force_2d`，仅流体节点）→
  `bounce_back_cells`；run.py 零手写物理核；源钉定只读 worktree main @ cf5709db3c。
- 每 (d,τ) 先跑同参数斯托克斯锚、再割线调体力 a 落 Re_target（容差 0.2%）；
  R_sim = (a/u)_Re ÷ (a/u)_锚——τ 依赖的 BB 墙位偏置在比值中一阶相消。
- τ=0.55（Ma 上限所迫，≠ verified/permeability 的 τ=1.0，逐档披露）；
  Re 档取锁定参考系列 A 格点 25.20 / 46.88 / 103.64（名义 {1,10,30} 偏离披露：参考锁定优先）。
- 全役 fp64：fp32 预检（d=104 锚）fs∈{1,10} 散布 +9.23%，注入地板污染，
  d≥104 的 fp32 斯托克斯窗口为空（`stage0.json`）。

## 计算结果现状（`result.json`，主判据 = 系列 A，err_A = |R_sim/R_A − 1|，R_A 按 PCHIP(log Re) 插值）

| 档 Re | d | Re_ach | R_sim | R_ref_A | err_A | err_B（副列） | err_SA（斯托克斯自检） | Ma(u_max) |
|---|---|---|---|---|---|---|---|---|
| A 25.20 | 52 | 25.209 | 1.147751 | 1.182270 | **2.920%** | 1.354% | 2.765% | 0.0933 |
| A 25.20 | 104 | 25.205 | 1.155275 | 1.182242 | **2.281%** | 0.705% | 0.950% | 0.0476 |
| A 25.20 | 208 | 25.213 | 1.153491 | 1.182291 | **2.436%** | 0.863% | 0.522% | 0.0237 |
| B 46.88 | 104 | 46.911 | 1.260668 | 1.284528 | **1.858%** | 0.834% | 0.950% | 0.0863 |
| B 46.88 | 156 | 46.917 | 1.250823 | 1.284550 | **2.626%** | 1.610% | 0.491% | 0.0570 |
| B 46.88 | 208 | 46.923 | 1.256604 | 1.284571 | **2.177%** | 1.157% | 0.522% | 0.0430 |
| C 103.64 | 208 | 103.809 | 1.397085 | 1.435800 | **2.696%** | 1.422% | 0.522% | 0.0947 |
| C 103.64 | 312 | 103.844 | 1.393335 | 1.435863 | **2.962%** | 1.690% | 0.205% | 0.0631 |
| C 103.64 | 416 | 103.635 | 1.391576 | 1.435479 | **3.058%** | 1.789% | 0.247% | 0.0472 |

- 档 A：最细 2.436% ≤ 3% ✓，严格单调 ✗（2.920→2.281→2.436）→ FAIL
- 档 B：最细 2.177% ≤ 3% ✓，严格单调 ✗（1.858→2.626→2.177）→ FAIL
- 档 C：最细 3.058% > 3% ✗，严格单调 ✗（2.696→2.962→3.058）→ FAIL
- **总判 FAIL**（无后验放宽；`verify.py` 独立复算——PCHIP / 锁定节点 / 独立 log-log
  三种插值互检——全部通过，插值项远小于判据带）。

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

1. **共性模块入口**：✓——物理核全部走库共性路径（D2Q9 collide/stream/体力/反弹），
   零手写核。
2. **直接观测量 ≤3%**：✗——档 C 最细 err_A 3.058% > 3%（档 A/B 最细 2.436%/2.177%
   数值过线）。
3. **≥2 档网格单调收敛**：✗——三档 d 阶梯全部不严格单调。

## 根因/诊断

1. **偏移为主、离散为辅**：err_A 九点全部 1.858–3.058% 平坦系统性偏低，对 d 无收敛
   趋势；同 (d,τ) 的斯托克斯自检 err_SA 在 d≥104 全部 ≤1%（最小 0.205% @ d=312）——
   斯托克斯极限我方复现无虞，偏差专门落在弱惯性增量上（R−1：sim 0.148/0.261/0.397
   vs 参考系 A 0.182/0.284/0.435）。
2. **参考侧带宽 ~1.0–1.6% 吸收偏移约半**：对系列 B（FDM）err 0.705–1.789%，全部 ≤3%
   无一破带；其余为方法侧残余（τ=0.55 BB 墙位、Ma² 可压缩性、Re 轴约定差，方向一致
   无一单独主导）。
3. **单调条款无判别力（判据设计发现）**：各 d 档掩码实现使斯托克斯 y0（f 形）在
   217.051–219.774 间摆动（~±0.6% 实现噪声地板），高于阶梯间误差变化——
   严格单调要求在该偏移+噪声水平下不具判别力。
4. **档 C 反向漂移与 Ma² 定量一致**：ma_umax 0.0947→0.0631→0.0472 随细化下降而
   err_A 抬升（2.696→2.962→3.058%），可压缩性偏高曳力随 Ma→0 消失的方向一致。

## 晋级路径（转 verified 条件；达标即归档 PR 移 pending/forchheimer → verified/forchheimer）

1. **参考侧（主导）**：取得 K&L97 表级数据（原文闭源，检索档案见服务器暂存 NOTES §3），
   或 owner 裁定以双源中点/带宽口径重锁参考；
2. **方法侧（次要）**：降 Ma（u_max 减半或 τ 回 1.0 + 域加倍）压缩可压缩性偏移与
   BB 墙位 τ 残余；
3. 重跑三档 d 阶梯：主判据最细 ≤3% 且严格单调 → 达标即归档 PR 移
   `pending/forchheimer` → `verified/forchheimer`。

## 运行方式

```bash
cd benchmarks/pending/forchheimer
PYTHONPATH=../../../src python run.py stage0                          # 几何裁定 + fp32 预检
PYTHONPATH=../../../src python run.py stokes --d 104                  # 单 (d,τ) 斯托克斯锚
PYTHONPATH=../../../src python run.py finite --d 104 --re-target 46.88
PYTHONPATH=../../../src python run.py report                          # 聚合 result.json
PYTHONPATH=../../../src python verify.py                              # 独立复算
```

工件：`result.json`（overall=FAIL + 9 行判决）、`case_anchor_*.json` ×6、
`case_re*.json` ×9、`stage0.json`、`forchheimer_ref_fig6.json` + `extract_ref_fig6.py`
（锁定参考）。完整过程工件（NOTES、修订链、检索档案、队列脚本）留服务器暂存
`/nfs/wangxi/runs/bm_widen_w6_20260921/forchheimer/`。

<!-- PROVENANCE
[{"v": 25.2, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[0].re_target"},
{"v": 46.88, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[3].re_target"},
{"v": 103.64, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[6].re_target"},
{"v": 52, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[0].d"},
{"v": 104, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[1].d"},
{"v": 208, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[2].d"},
{"v": 156, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[4].d"},
{"v": 312, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[7].d"},
{"v": 416, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[8].d"},
{"v": 25.209382888907204, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[0].re_ach"},
{"v": 25.20510563245374, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[1].re_ach"},
{"v": 25.212510204522996, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[2].re_ach"},
{"v": 46.911459834560866, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[3].re_ach"},
{"v": 46.91707776859517, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[4].re_ach"},
{"v": 46.922783309632884, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[5].re_ach"},
{"v": 103.80944678216218, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[6].re_ach"},
{"v": 103.84391722384565, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[7].re_ach"},
{"v": 103.6350046459312, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[8].re_ach"},
{"v": 1.1477506903002888, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[0].R_sim"},
{"v": 1.1552752546566418, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[1].R_sim"},
{"v": 1.1534914996042447, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[2].R_sim"},
{"v": 1.2606677574373577, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[3].R_sim"},
{"v": 1.2508234809582952, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[4].R_sim"},
{"v": 1.256604176478648, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[5].R_sim"},
{"v": 1.3970846591497847, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[6].R_sim"},
{"v": 1.393335289922649, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[7].R_sim"},
{"v": 1.3915756660085783, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[8].R_sim"},
{"v": 1.182270158068317, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[0].R_ref_A"},
{"v": 1.182242222145988, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[1].R_ref_A"},
{"v": 1.1822905796793186, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[2].R_ref_A"},
{"v": 1.2845284569590711, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[3].R_ref_A"},
{"v": 1.2845496861271368, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[4].R_ref_A"},
{"v": 1.284571245443951, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[5].R_ref_A"},
{"v": 1.435799678817205, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[6].R_ref_A"},
{"v": 1.4358627372376338, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[7].R_ref_A"},
{"v": 1.435478508158458, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[8].R_ref_A"},
{"v": 0.029197614041471587, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[0].err_A"},
{"v": 0.022810018948905553, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[1].err_A"},
{"v": 0.024358715674521636, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[2].err_A"},
{"v": 0.018575454200680075, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[3].err_A"},
{"v": 0.026255274928698724, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[4].err_A"},
{"v": 0.021771520314264547, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[5].err_A"},
{"v": 0.026964081576695387, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[6].err_A"},
{"v": 0.029618045104228163, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[7].err_A"},
{"v": 0.03058411665542915, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[8].err_A"},
{"v": 0.013543075881592759, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[0].err_B"},
{"v": 0.007051239035417378, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[1].err_B"},
{"v": 0.008627054863289185, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[2].err_B"},
{"v": 0.008338246240240155, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[3].err_B"},
{"v": 0.01609848809074288, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[4].err_B"},
{"v": 0.01156828330956372, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[5].err_B"},
{"v": 0.014215399299561948, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[6].err_B"},
{"v": 0.01690276053976436, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[7].err_B"},
{"v": 0.017889874241803883, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[8].err_B"},
{"v": 0.02764522139180614, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[0].err_SA_stokes"},
{"v": 0.009497738416852863, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[1].err_SA_stokes"},
{"v": 0.005218828237460782, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[2].err_SA_stokes"},
{"v": 0.004914889873687267, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[4].err_SA_stokes"},
{"v": 0.0020528069236838142, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[7].err_SA_stokes"},
{"v": 0.002468950720603269, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[8].err_SA_stokes"},
{"v": 0.09326579545086405, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[0].ma_umax"},
{"v": 0.047567658683067074, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[1].ma_umax"},
{"v": 0.02373222340630078, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[2].ma_umax"},
{"v": 0.08631621009142489, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[3].ma_umax"},
{"v": 0.05701483384272297, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[4].ma_umax"},
{"v": 0.043048771923894336, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[5].ma_umax"},
{"v": 0.09470183255650116, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[6].ma_umax"},
{"v": 0.06310168638126303, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[7].ma_umax"},
{"v": 0.04718910801323347, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[8].ma_umax"},
{"v": 217.051120722085, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[4].y0_f_form"},
{"v": 219.7738894659736, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[1].y0_f_form"},
{"v": 0.024358715674521636, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "tier_verdicts.A.finest_err"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "tier_verdicts.A.strictly_monotone"},
{"v": 0.021771520314264547, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "tier_verdicts.B.finest_err"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "tier_verdicts.B.strictly_monotone"},
{"v": 0.03058411665542915, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "tier_verdicts.C.finest_err"},
{"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "tier_verdicts.C.strictly_monotone"},
{"v": "FAIL", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "overall"},
{"v": 0.09227611820485015, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/stage0.json", "k": "fp32_precheck.d104.fs_spread"},
{"v": 0.40020751953125, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/stage0.json", "k": "geometry_adjudication.phi040.phi_s_mask"},
{"v": "cf5709db3c2bba273c64807e721071de110627d1", "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/stage0.json", "k": "provenance.tensorlbm_commit"},
{"v": 2.9197614041471587, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[0].err_A", "k2": "rows[0].err_A", "op": "scale100"},
{"v": 1.8575454200680075, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[3].err_A", "k2": "rows[3].err_A", "op": "scale100"},
{"v": 3.058411665542915, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[8].err_A", "k2": "rows[8].err_A", "op": "scale100"},
{"v": 2.6255274928698724, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[4].err_A", "k2": "rows[4].err_A", "op": "scale100"},
{"v": 1.609848809074288, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/result.json", "k": "rows[4].err_B", "k2": "rows[4].err_B", "op": "scale100"},
{"v": 9.227611820485016, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/forchheimer/stage0.json", "k": "fp32_precheck.d104.fs_spread", "k2": "fp32_precheck.d104.fs_spread", "op": "scale100"}]
-->
