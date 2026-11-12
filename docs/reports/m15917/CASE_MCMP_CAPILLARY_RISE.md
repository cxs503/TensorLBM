# CASE-MCMP-CAPILLARY-RISE — Jurin 毛细上升（TensorLBM / CAC 相场），m15917 第二轮

**算子**: TensorLBM（保守 Allen-Cahn 相场 LBM + Young 润湿壁条件）
**锚点**: CASE-MCMP-CAPILLARY-RISE　**判据**: |h_sim − h_Jurin| < 10 %
**结论**: **honest FAIL（主判据未过）** —— 但本轮把"为什么过不了"从"未知"推进到"定位到具体代码缺陷 + 数值失稳机理"，并修掉两个真实缺陷（有 A/B 对照数字）。

| 级别 | 证据 | 状态 |
|---|---|---|
| L1 | `tests/test_case_pool_capillary_rise_cac.py`（6 tests，13 s，全绿） | 过 |
| L2 | impl: `src/tensorlbm/cac_lbm2d.py`（壁面 ghost + `_shift` 修复）、`benchmarks/case_pool/capillary_rise/{jurin_cac,ab_wall_fix,chunk_run}.py`；entry: 三个 CLI；run_artifacts: `artifacts/m15917/*.json` + 本目录 figs | 过 |
| L3 | 调用方: `benchmarks/in32_l01_wetting_ladder/run_case.py:28`（`from tensorlbm.wetting import ladder as L` → `L.jurin` → `cac.CAC2D`）、`src/tensorlbm/wetting/ladder.py`、`benchmarks/case_pool/capillary_rise/jurin_cac.py` | 过 |
| L4 | ref vs got 数值对照见 §4（含 10 % 门判定）+ repro_cmd 见 §7 | 有对照，**主判据 FAIL** |
| L5 | 文献锚: Jurin (1718) Phil. Trans. 30, 739–747；二维平行板 r→w 与曲率恒等式 2cosθ/w（测试里独立核过） | — |

---

## 1. 问题 + ref 出处

毛细上升：竖直毛细管（此算例为二维平板缝，缝宽 w）插入液池，弯月面在毛细压与静水压平衡时上升

```
h = 2 σ cosθ / (Δρ g w)          Jurin (1718) Phil. Trans. R. Soc. 30, 739–747
```

二维平板缝的弯月面是半径 R_c = (w/2)/cosθ 的圆弧，平均曲率 2cosθ/w，因此管径 r → 缝宽 w（de Gennes,
Brochard-Wyart & Quéré, *Capillarity and Wetting Phenomena*, ch. 2 的同一表示）。**ref 数值不手抄**：
σ 由本次运行自测（平面膜张力积分，`ladder.sigma_flat`），θ 是施加给壁面边界条件的值，Δρ = ρ_l − ρ_g、
g、w 全部写进工件 JSON，h_ref 由 `jurin_ref()` 从这些输入算得（`jurin_cac.py:70`）。

**为什么换求解器**：本算例池原有的 A/B 两条路线（`run_case.py` / `sc_closed_box.py`，Shan-Chen 伪势）的壁面润湿由
伪密度 `psi_w` 控制，实测有效接触角 88.1°（申报 35°），毛细驱动根本不可控 —— 上一轮交付 h = 2.84 vs ref 33.25
（−91 %）。本轮改用库里已有的 **Young 接触角边界条件**（`cac_lbm2d`，θ 是输入量），才谈得上对标。

## 2. 方法 / 关键设置

1. **标定腿（与上升腿独立）**：σ ← 平面膜积分 `sigma_meas = 0.5∫(f + k/2|∇φ|²)dy`；实测 0.02918 vs 输入 0.03（**−2.7 %**）。
2. **上升腿**：`CAC2D(shape=(120,64), axes='wp', wall_phi_policy='contact', contact_angle_deg=θ,
   gravity=(0,−1e−4), gravity_mode='rho')`；几何 = 底板 2 行 + 顶板 2 行 + 一对竖直挡板（厚 2，缝宽 20）立于
   y=20 的液池中；气体可从挡板顶部溢出（非封闭管）。测点：通道带 x∈[x_left+3, x_right−1) 与储液带 x∈[4, x_left−2)
   的 φ=0.5 亚格点穿越高度之差 h = y_channel − y_reservoir（`wetting.metrics.meniscus_height`）。
3. **工具链**：python 3.11 / torch 2.7.1 / CPU 4 线程；整段运行 15–90 s，可 `chunk_run.py` 断点续算（每块落盘 `f,g`）。
4. **预声明**：门 |err| < 10 %；停止判据 = 末段 20 % 样本 h 极差 < 2 %。判据先于结果写死在本文件与脚本里。

## 3. 参数表（全部无量纲格点单位）

| 参数 | 值 | 说明 |
|---|---|---|
| ρ_l / ρ_g | 1.0 / 0.1 | 密度比 10 |
| Δρ | 0.9 | |
| σ（输入 / 实测） | 0.03 / 0.02918 | 实测 −2.7 % |
| 界面宽度 W | 4.0 | tanh 型弥散界面 |
| mobility / τ_f | 0.05 / 0.65 | τ_f = 0.5 + 3M |
| μ_l / μ_g | 0.05 / 0.005 | harmonic 平均 |
| 缝宽 w / 挡板厚 / 池深 | 20 / 2 / 20 | 格点 |
| 域 | 120 × 64（x 周期，y 壁面） | |
| g | 1.0e−4 | 体积力，gravity_mode='rho' |
| Bond 数 Bo = Δρ g w²/σ | 1.33 | |
| 毛细压 2σcosθ/w（θ=60°） | 1.50e−3 | = Δρ g h → h=16.67 |
| θ（主算例 / 负对照） | 60° / 90° | 施加于壁面 BC |

## 4. 对照表（ref vs got + 误差打勾/叉）

### 4.1 主判据（θ = 60°，门 10 %）→ **FAIL（未收敛即失稳）**

| 量 | ref | got | 误差 | 判定 |
|---|---|---|---|---|
| h（step 375，最后一个有限样本） | 16.667 | 4.180 | −74.9 % | ✗（且未收敛，不能当平衡值） |
| h（step 125 / 250） | 16.667 | 0.37 / 1.78 | −97.8 % / −89.3 % | ✗ 上升过程 |
| 液质量 ∫φ | 守恒 | step 375 时 −14.09 % | — | ✗ |
| φ_min（流体） | ≥ 0（双阱） | −0.0986 | — | ✗ 越界 |
| 数值稳定性 | 跑到稳态 | step ≈ 440 出现 NaN | — | ✗ 失稳 |

### 4.2 负对照（θ = 90°，解析驱动恒为 0）→ **PASS**

| 量 | ref | got | 判定 |
|---|---|---|---|
| h | 0（Jurin, cos90°=0） | **0.865** @3000 步（末段极差 3.19 %） | ✓（= θ=60° Jurin 尺度的 5.2 %，< 10 %） |
| ∫φ 漂移 | 0 | +1.06 % / 1250 步 | ✓（< 5 %） |
| u_max（全节点） | — | 0.0024 | ✓ |

### 4.3 两个缺陷修复的 A/B（θ = 90°，1250 步，同种子同参数；`artifacts/m15917/ab_wall_fix_theta90.json`）

| 变体 | h | 液质量漂移 | u_max |
|---|---|---|---|
| `old_both`（任务前 main） | **−19.399** | **−50.26 %** | 0.0569 |
| `axes_fix`（只修 `_shift`） | **+0.559** | +1.06 % | 0.0042 |
| `ghost_fix`（只修壁面法向 ghost） | −21.245 | −49.51 % | 0.0354 |
| `fixed`（两者都修 = 本分支） | **+0.549** | +1.06 % | **0.0024** |

读法：决定性是 **`_shift` 的 axes 策略缺陷**（修前把 `axes='wp'` 当 `'ww'`，x 周期缝变成零填充质量汇，
1250 步漏掉一半液体，中性壁"假上升" −19.4 格点），修复后中性壁回到 h≈0、质量守恒到 1 %；
壁面法向 ghost 修复把速度底噪再降 1.75×（0.0042→0.0024），并为竖直挡板给出正确的 Young 通量（单元测试核）。

## 5. 收敛性 / 参数敏感性（至少两点）

1. **收敛性**：θ=90° 支路 h 末段极差 3.19 %（3000 步，采样 250 步）< 2 % 门未达但已趋平（h: 0.607@4000 步 vs 0.865@3000 步 不同域长，见工件）；
   质量漂移随步数线性 ~+8.5e−4 %/步 → 零净通量约束并非严格守恒（已知局限）。
2. **网格/缝宽**：(120,64) 与 (160,88) 两套域上 θ=90° 都收敛到 |h| ≲ 1.5 格点，结论与域尺寸无关（故障模式也不随域变）。
3. **θ 敏感性（关键）**：θ=90° 稳定；θ=60° 失稳。即缺陷在 **cosθ ≠ 0 的壁面通量**上，不是几何或测量链。

## 6. 失败模式（measured，非猜测）+ 工件

* **F1（本轮定位，已修）**：`cac_lbm2d._shift` 只判 `axes[0]`，`axes='wp'` 实际为 `'ww'` → 侧边 x 缝变质量汇。
  证据：`ab_wall_fix_theta90.json`，−50.26 % 液体 1250 步。
* **F2（本轮定位，已修）**：润湿 ghost 用 8 邻居各向同性平均，把"水平底板"推导套到竖直挡板上 → 竖直壁通量方向错。
  证据：`ghost_fix` 变体 u_max 0.0354 vs `fixed` 0.0024；单元测试 `test_ghost_uses_the_normal_neighbour_not_the_isotropic_average`。
* **F3（未修，本算例的拦路虎）**：`wall_phi_policy='contact'` 且 cosθ≠0 时，壁面吸收相场：∫φ 以 −0.5/步量级下降、
  φ_min 越界（−0.099）、约 440 步发散到 NaN。**与 F2 无关**（`ghost_fix` 变体与 `fixed` 变体同样失稳，`jurin_theta60_wetting.json`）。
  推断（未验证）：`zero_net` 修正以 `base`（邻点值）而非壁面旧值计算通量预算，cosθ≠0 时每步净注入非零；
  下一步应改为按壁面旧值算 Δφ 并在液相内部回灌。这是两轮法的接棒点。
* **F4（生态摩擦）**：`cluster_queue` 提交的 cj1203/cj1204 在 2.5 h 内始终 queued（两节点 load≈35–38 饱和），
  本轮 15–90 s 的分块运行全部在本地用 `chunk_run.py` 断点续算完成；集群调度饱和是可复现的生态病。

## 7. 复现命令

```bash
# 主判据（θ=60°，~20 s 内看到失稳）
PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/jurin_cac.py --theta 60 --steps 1000 \
    --sample-every 125 --no-calibration --out artifacts/m15917/jurin_theta60_repro.json

# 负对照（θ=90°，~60 s）
PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/jurin_cac.py --theta 90 --steps 3000 \
    --sample-every 250 --no-calibration --threads 4 --out artifacts/m15917/jurin_neutral_theta90.json

# 两个缺陷的 A/B（~70 s）
PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/ab_wall_fix.py --steps 1250 \
    --out artifacts/m15917/ab_wall_fix_theta90.json

# 单元/回归
python3 -m pytest tests/test_case_pool_capillary_rise_cac.py -q          # 6 passed, 13 s
python3 -m pytest tests/test_wetting_ladder.py -q                        # 回归（30 tests）
```

## 8. 误差判定（显式回答）

| 关键量 | vs 基准误差 | 是否 < 3 % |
|---|---|---|
| σ 标定 | −2.7 % vs 输入 | **是** |
| 中性壁 h（θ=90°） | +0.865 格点（ref 0）= 5.2 % of 16.667 尺度 | 否 |
| 主判据 h（θ=60°） | −74.9 %（未收敛） | **否 —— 本算例 honest FAIL** |

不粉饰：毛细上升 Jurin 判据在 CAC-Young 壁面上**本轮仍未通过**，卡在 F3（壁面通量不守恒导致失稳）；
本轮的真实增量是两个已修缺陷 + 把失败从"pseudo-density 润湿不可控"推进到"壁面通量守恒实现缺陷（F3）"这一可接棒的具体位置。