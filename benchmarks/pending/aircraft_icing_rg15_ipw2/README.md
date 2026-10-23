# aircraft_icing_rg15_ipw2 — RG-15 翼型 IPW2 官方三案（glaze/mixed/rime）  ⏳ PENDING

状态：**pending**（2026-10-06，honest-FAIL 定级）。RG-15 翼型在 IPW2 官方三工况下的
结冰模拟：判据族含逐 bin 质量闭合、Messinger 闭合、渲染质量守恒、不变性、逐位复现，
以及 MCCS 实测冰形锚。**定级 pending 的理由（全部机器值）**：

- **D1/D2 MCCS 冰形锚的模型类差距**：注册后模拟/实测冰面积比 0.138–0.318×、
  厚度比 0.255–0.455×——模拟冰显著小于实测。
  模型类 = 等温空气动力学 + 一维表面 Messinger 热力学 + 面板 β 驱动增长；门不放宽。
- **G3 FAIL**：per-bin Eulerian 闭合 max = 3.028e-02（门 <=1e-6）。根因 =
  ledger-free droplet-warmup 窗口的净通量不平衡（warmup = 6000 时随窗长
  线性增长，库文档界 <5e-2；生产窗总账闭合 2.70e-17 干净）。
- **G11a FAIL**：案 3.1 n_f 模块 vs 独立 0-D 相对差 10.05%
  （门 0.5%）——差距 100% 来自输入重构（panels 导出量不自洽面板 n_f）；两求解器同输入
  逐位一致（`G11_solver_crosscheck` 6 格 abs_diff 全 0.0）。重校准留 owner。
- **per-bin β 曲线空**：库不对称（per-bin 路径无 IC-D3 空窗回退；`run_glaze_icing` 的
  trailing 窗为空）→ bins7 β 档案中 bin 曲线为 0 长度、β_pk_per_bin 全空。如实披露。
- **温度场模型边界**：无体温度场，以 T_s(s/c) 面板曲线替代 2-D 温度云图（不虚构渲染）。

大判据 19/21（FAIL = G3 + G11a；verdict md5 128625e189512600b54ce5af3dec385f）。

## 工况（IPW2 官方条件，ast 提取自冻结脚本）

| 案 | T∞ (°C) | Re |
|---|---|---|
| 3.1 glaze | -2.0 | 5.7e+05 |
| 3.2 mixed | -4.0 | 5.8e+05 |
| 3.3 rime | -10.0 | 6.0e+05 |

公共：弦长 0.3 m、V∞ = 25.0 m/s、AoA 4.0°、
LWC 0.44 g/m³、MVD 24 µm、曝光
1200 s、p = 101300.0 Pa、展长 0.58 m、
RH = 0.975。

格子：320×160（chord_frac = 0.4）、u_in = 0.05、
cumulant、c_s = 0.1、Eulerian 液滴相 donor2、
macklin rime 密度、β 窗 = clean、
warmup 6000 + steps 3000、glaze 多段 5 shots。

## 官方 7-bin DSD（B4 变体用；ast 提取 + 参考锁交叉核）

| bin | D (µm) | LWC 份额 | per-bin 闭合误差 |
|---|---|---|---|
| 1 | 7.3 | 5% | 5.6297e-03 |
| 2 | 12.5 | 10% | 1.0999e-02 |
| 3 | 18.7 | 20% | 1.6952e-02 |
| 4 | 26.8 | 30% | 2.3231e-02 |
| 5 | 36.8 | 20% | 2.8109e-02 |
| 6 | 51.2 | 10% | 3.0284e-02 |
| 7 | 81.3 | 5% | 2.9987e-02 |

## B3 三案结果（`evidence/b3/`）

| 案 | T∞ (°C) | Re | β_pk | 冻结 (kg) | 闭合 | n_f 驻点 | 驻点冰厚 (mm) | 终态冰胞 |
|---|---|---|---|---|---|---|---|---|
| 3.1 glaze | -2.0 | 5.7e+05 | 0.781526 | 4.1807e-04 | 3.1e-17 | 0.6486 | 0.90 | 36 |
| 3.2 mixed | -4.0 | 5.8e+05 | 0.786555 | 4.1387e-04 | 1.2e-16 | 1.0000 | 1.27 | 35 |
| 3.3 rime | -10.0 | 6.0e+05 | 0.788040 | 4.2884e-04 | 7.6e-18 | 1.0000 | 1.01 | 36 |

## B4 官方 7-bin DSD（案 3.3）

wall = 1072.3 s；β_pk = 0.770270（mono 案 3.3 为
0.788040；D5：impacted +12.784% /
frozen +12.498% / β_pk -2.255%）；
冻结 4.8244e-04 kg、冲击 5.0284e-04 kg、
蒸发 2.0398e-05 kg、runback = 0.0、
总账闭合 2.70e-17；n_f 驻点 = 1.0。

## B5 shot/accel 不变性（`evidence/b5/`）

门族（steps = 3000，d ∈ {120, 300, 600, 1200}）max = 0.595% ≤ 1% **PASS**
（生产格 s3000_d240 恒等参照，不入证据）；披露族（变 shot 长度）：
s800/s1500/s6000 = 2314.4%/
56.79%/4241.96%
（破坏 accel 不变性的已知性质，如实披露非门）。G8 估计器对 exp11 先验 8/8 复现。

## B6 β_stag 分辨率阶梯（`evidence/b6/`，D3 披露非门）

| nx | dx (mm) | β_stag |
|---|---|---|
| 320 | 2.3438 | 0.7880 |
| 480 | 1.5625 | 0.8047 |
| 640 | 1.1719 | 0.8179 |

三档对 exp12 先验逐位一致；分辨率受限（弹道期望簇 0.9–1.0，R7）定量披露。

## MCCS 实测冰形对比（D1/D2；注册纪律 D0 全 0 漂移）

| 案 | 切割 | 面积 sim/meas (mm²) | 面积比 | 厚度 sim/meas (mm) | 厚度比 |
|---|---|---|---|---|---|
| 3.1 glaze | v2 | 197.8/641.2 | 0.308 | 7.03/25.79 | 0.273 |
| 3.1 glaze | v3 | 197.8/681.4 | 0.290 | 7.03/26.38 | 0.267 |
| 3.2 mixed | v2 | 192.3/1395.6 | 0.138 | 11.72/46.02 | 0.255 |
| 3.2 mixed | v3 | 192.3/670.3 | 0.287 | 11.72/27.88 | 0.420 |
| 3.3 rime | v2 | 197.8/682.5 | 0.290 | 11.72/26.02 | 0.450 |
| 3.3 rime | v3 | 197.8/621.9 | 0.318 | 11.72/25.75 | 0.455 |

注册变换 = verdict D1 存档拟合（θ/t），本档重拟合漂移全 0（D0 门，PASS）。

## 其余门

| 门 | 判据 | 测得 | 结果 |
|---|---|---|---|
| G4 Messinger 闭合（B3×3 + B4） | <=1e-12 | 1.2e-16 | PASS |
| G5 渲染质量守恒（4 案） | <=1e-9 and pending==0 | max \|r/f−1\| = 1.1e-16，pending = [0.0, 0.0, 0.0, 0.0] | PASS |
| G6 rime 极限 | 3.3: |n_f-1|<=1e-12, runback==0; 3.2: |n_f-1|<=1e-12 | n_f_stag(3.3) = 1.0，runback(3.3) = 0.0，n_f_stag(3.2) = 1.0 | PASS |
| G8 估计器复现 exp11 | 8/8 \|diff\|≤1e-12 | 8/8 tags |diff|<=1e-12 | PASS |
| G12 regime 排序 | 见测得 | n_f 驻点(−2°C) = 0.6486 < (−10°C) = 1.0；T_s 驻点(−2°C) = 0.0 °C | PASS |
| G13 β_pk 逐位复现 accept | ours == accept | 0.7815262675285339 == 0.7815262675285339（全曲线 md5 cfafc939d75f5a0a52a8a77151e3d909） | PASS |
| G11a 求解器一致性 | rel ≤ 0.5% | 案 3.1 10.05%（3.2/3.3 严格 0） | FAIL |

regime 计数（`G12_regime_counts`）：
- 3.1 glaze: dry=80, evap=101, glaze=21, rime=7
- 3.2 mixed: dry=90, evap=108, glaze=2, rime=10
- 3.3 rime: dry=99, evap=98, rime=11

## Reproduce

```bash
# 从仓库根目录；run.py 自定位 <repo>/src（无需 PYTHONPATH）
CUDA_VISIBLE_DEVICES=0 python benchmarks/pending/aircraft_icing_rg15_ipw2/run.py b3   # 3 案 multishot, ~642 s
CUDA_VISIBLE_DEVICES=0 python benchmarks/pending/aircraft_icing_rg15_ipw2/run.py b4   # bins7, ~1072 s（需 IC-E-D1 修复）
CUDA_VISIBLE_DEVICES=0 python benchmarks/pending/aircraft_icing_rg15_ipw2/run.py b5   # 8 格, ~319 s
CUDA_VISIBLE_DEVICES=0 python benchmarks/pending/aircraft_icing_rg15_ipw2/run.py b6   # 3 档, ~127 s
python benchmarks/pending/aircraft_icing_rg15_ipw2/run.py b7                          # CPU 流场渲染, ~193 s
# outputs to <case>/output/{b3,b4,b5,b6,b7}/; reference geometry + MCCS cuts in <case>/reference/
```

## 目录

- `run.py` — 分发器；`_b_common.py`（RG-15 几何/工况/配置）+ `_b3_b4_glaze.py` +
  `_b5_invariance.py` + `_b6_beta_res.py` + `_b7_render.py`（战役脚本路径适配版）
- `evidence/` — 机器档案：b3（3×result + β/masks/panels npz）、b4（result + β/masks/
  panels npz）、b5（json + 8×β npz）、b6（json + 3×β npz）、b7（render json + 3×流场 npz）
- `reference/` — RG-15 dat、官方条件锁（txt）+ IPW 案描述摘录 + 6 份 MCCS 实测切割 CSV
- `figs/` — 27 图（三案 ×7：flow/alpha/beta/ice_mccs/thermo/regime/history + B4 ×4 +
  B5 + B6）
- `result.json` — 机器组装汇总

## Notes

- 判据溯源：`/nfs/wangxi/runs/icing_bm_20261006/verdict.json`（md5 128625e189512600b54ce5af3dec385f，
  由冻结 verify.py 在 B4 复跑后原样重生成，零手改）；prereg md5 65b3565e6c0bbf450ef50b9ab8660a65。
- 证据树：B3/B5/B6/B7 = pristine Phase-1 树（35856caa (pristine Phase-1 tree)）；B4 = 修复树
  （exp/icing-warmup-bins @ 100f78d6 (IC-E-D1 fix)）。A/B 证明修复对 mono 案逐位一致（40/40 数组）。
- B4 存档由战役 rerun 脚本产生（含 per-bin β 记录器扩展，捕获到 0 长度曲线）；
  `run.py b4` 臂与之处判据字段一致。
