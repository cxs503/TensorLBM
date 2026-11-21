# K3-MOMCONV · 动量携带型界面转换（I→G 转换不销毁被转质量的动量）

> 仓库 TensorLBM · 分支 agent/m26869 · 锚 = NEW:K3-MOMCONV（K2-BUDGET-B 处方直系，minimax 赛马道）
> 判决：**证伪（honest FAIL）**：两臂均未落入命中门（误差 > 15 % 或更浅/早死）
> 自检：数字可溯 = `runs/k3_ctrl|result.json`、`runs/k3_inherit/result.json`、`runs/k3_redistribute/result.json`、
> `runs/k3_momconv_verdict/verdict.json` ✓ | 预登记先跑（`docs/audits/k3_momconv_20261121/PREREG.md`）✓ |
> ctrl 锚复现 Δ = 2.318e-11 D / 0.000e+00 ms ✓ | 失败如实 ✓
> commit = 14b174847

> 前情（定谳勿重验）：K2 五项收支不闭合（|R|/E_scale = 1781.68 %），主导通道 = ⑦ 质量交换转换（248.98 %）。
> 机制：I→G 转换时被转液相质量的速度/动量在模型里直接销毁（气相速度为零），峰时刻轴向液柱动能只剩 11.1 %；
> 实验坑深只需 0.55 % 撞击能量作重力位能 ⇒ 缺口在惯性/动量分配侧（"能量送不进去"），不在静水侧。

## 1. 判决

**证伪（honest FAIL）**：两臂均未落入命中门（误差 > 15 % 或更浅/早死）。

- 最深臂：**redistribute**（峰深 1.4352204748 D @ 11.090571 ms，
  相对实验 -22.7310 %）。
- 命中臂：无；证伪臂：inherit, redistribute。
- 判决点（实验）：坑深 1.857434 D @ 10.394 ms。
- **补充臂 inherit_fix**（`inherit`，commit 81d4d8ad9，预登记 `PREREG-B-inherit-repair.md` / `PREREG-C-redistribute-liq.md`）：峰深 0.6429826376 D @ 3.161786 ms，存活 3.1739 ms，err -65.3833 %，判定 **证伪**（stop_reason = health_limit）。
- **补充臂 redistribute_liq**（`redistribute_liq`，commit cd3a9da5c，预登记 `PREREG-B-inherit-repair.md` / `PREREG-C-redistribute-liq.md`）：峰深 0.2748627020 D @ 1.629536 ms，存活 1.6417 ms，err -85.2020 %，判定 **证伪**（stop_reason = health_limit）。

## 2. 关键量

| 量 | 模拟 | 基准/阈值 | 判定 |
|---|---|---|---|
| ctrl 锚复现（峰深） | 1.4352204748 D | 战役锚 1.4352204748 D | Δ = 2.318e-11（位级） |
| ctrl 锚复现（峰时刻） | 11.090571 ms | 战役锚 11.090571 ms | Δ = 0.000e+00 ms |
| 最深臂峰深 | 1.4352204748 D | 实验 1.857434 D | -22.7310 %（命中门 ±12 %） |
| 最深臂存活 | 24.5525 ms | > 10.4 ms | 是 |
| 转换通道净损耗 / E_scale（ctrl → 最深臂） | +143.7351 % → +143.7351 % | 方向性下降为佳 | 未下降 |

> 注：**最深臂 = redistribute 是位级 no-op 臂**（与 ctrl 逐字段零差异，见 §5），故「最深臂峰深 = ctrl 峰深」；
> 真正执行了动量交付的三臂全部早死，其峰深不参与「最深臂」比较。

## 3. 工况与方法

- 工况：Santini case-3 单滴入水，cells = 8，gap = 0.18 D，lattice speed = 0.06，
  prescribed release = 1.53 m/s，pool depth = 3.0 D，u_limit = 0.15，capacity + simple 质量交换，
  SDAA 满窗（三臂并行、同参数）。
- 实现（最小改动、默认关）：`src/tensorlbm/free_surface_lbm.py` 的 `free_surface_step(..., momconv="off", momconv_state=None)`。
  - `inherit`：转换胞把**受追踪质量动量** m·u 交给它触及的 INTERFACE 胞（等分，界面注入），
    同时气相胞继承液相速度（ABB 重建以之为气相速度，跨步持久于 `momconv_state`）。
  - `redistribute`：同一动量按邻接 LIQUID 胞的受追踪质量份额分配。
  - 交付方式为保密度动量踢 Δf_q = 3 w_q (ΔP·c_q)：精确加上 ΔP 动量、不改密度。
- 旗标：`benchmarks/water_entry/single_drop/free_surface_probe.py --momconv {inherit,redistribute}`；
  SDAA 包装器（共享 `sdaa_fs_probe.py` 与树内 `tools/sdaa_fs_probe_split.py`）透传 `WE_MOMCONV`。
- 判决脚本：`tools/k3_momconv_verdict.py`（数字全部程序化注入，零手抄）；
  门写死在预登记 `docs/audits/k3_momconv_20261121/PREREG.md`，跑后一字未改。

## 4. 三臂逐项结果

| 臂 | momconv | commit | 峰深 (D) | 峰时刻 (ms) | 存活 (ms) | err vs 实验 | conv_loss/E_scale | conv_gain/E_scale | 净转换损耗/E_scale | 峰时 KE_axial | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ctrl | `off` | 9b81d8dbf | 1.4352204748 | 11.090571 | 24.5525 | -22.7310% | +248.9829% | +105.2477% | +143.7351% | 5.610230e-02 | 基线（锚） |
| inherit | `inherit` | 9b81d8dbf | 0.1480812002 | 1.349839 | 1.3620 | -92.0276% | +105.7392% | 质量爆炸 | 质量爆炸 | 3.193960e-01 | 证伪 |
| redistribute | `redistribute` | 9b81d8dbf | 1.4352204748 | 11.090571 | 24.5525 | -22.7310% | +248.9829% | +105.2477% | +143.7351% | 5.610230e-02 | 证伪 |
| inherit_fix | `inherit` | 81d4d8ad9 | 0.6429826376 | 3.161786 | 3.1739 | -65.3833% | +203.1942% | 质量爆炸 | 质量爆炸 | 2.066002e-01 | 证伪 |
| redistribute_liq | `redistribute_liq` | cd3a9da5c | 0.2748627020 | 1.629536 | 1.6417 | -85.2020% | +122.3911% | 质量爆炸 | 质量爆炸 | 2.571555e-01 | 证伪 |

（`inherit_fix` 为修复臂：仅 inherit 分支加继承速度限幅，预登记 `PREREG-B-inherit-repair.md`；
`redistribute_liq` 为解释臂：接收胞集合扩到整个液相（bulk + 界面），预登记 `PREREG-C-redistribute-liq.md`。）

（conv_* 为接触→窗口末的累计转换通道量，E_scale = 接触时液相动能；KE_axial 为深度峰时刻的轴向液柱动能。）

## 5. 机制解读

- **inherit**：转换通道净损耗 +143.74 % → **质量爆炸**（单步 conv_gain 尖峰，账目无意义）；峰时轴向动能 5.6102e-02 → 3.1940e-01；坑深 1.435220 → 0.148081 D；stop_reason = health_limit。
- **redistribute**：转换通道净损耗 +143.74 % → +143.74 % E_scale（Δ +0.00 %），conv_loss 份额 Δ +0.00 %；峰时轴向动能 5.6102e-02 → 5.6102e-02；坑深 1.435220 → 1.435220 D；stop_reason = window_complete。
- **inherit_fix**：转换通道净损耗 +143.74 % → **质量爆炸**（单步 conv_gain 尖峰，账目无意义）；峰时轴向动能 5.6102e-02 → 2.0660e-01；坑深 1.435220 → 0.642983 D；stop_reason = health_limit。
- **redistribute_liq**：转换通道净损耗 +143.74 % → **质量爆炸**（单步 conv_gain 尖峰，账目无意义）；峰时轴向动能 5.6102e-02 → 2.5716e-01；坑深 1.435220 → 0.274863 D；stop_reason = health_limit。

**机制定位（本单最重要的产出）**：
1. `redistribute` 臂与 ctrl **逐字段位级相同**（2020 行 history × 9 字段，差异 0，最大绝对差 0.0；峰深同为 1.435220474823 D @ 11.090571429 ms）⇒ 该臂**根本没有触发过动量交付**。代码级原因：该臂的接收胞集合是纯 bulk LIQUID 胞，而本工况里每一次 I→G 转换都发生在界面层，转换胞的邻居全是 INTERFACE 胞，**bulk LIQUID 接收胞集合恒为空** ⇒ 转换销毁的动量无处可交，处方在这一臂上是位级 no-op（不是「无效」，是「没被执行」）。
2. `inherit` 臂把动量交给界面胞（真正执行了交付），但**破坏了 Körner 质量记账**：动量踢改写界面胞 populations ⇒ 改变质量交换的 population 差分 ⇒ 受追踪质量与 population 密度失配；表现是单步 `conv_gain` 尖峰（主赛臂第 65 步 3.1e14，修复臂 26 次尖峰、最大 6.1e16）与受追踪质量漂移越过 1e-4 健康门（主赛 1.362 ms，修复臂 3.174 ms）⇒ 早死。
3. 因此本单把处方**证伪**为一个模型级结论：在本单流体自由面模型（mass/population 双表示）里，I→G 转换的动量**要么交不出去**（bulk 接收集为空），**要么一交就破坏记账而早死**（界面接收集）。这与 K2 的「⑦ 是模型级能量汇、不是物理耗散」是同一层结论的加强版：被转质量的动量在这个模型里**无处安放**，不是「传错了地方」。

## 6. 诚实披露

- 本单是**处方验证**，不是精度修复：命中门 12 % 比战役总门 3 % 宽，不宣称通过验收。
- **honest FAIL**：没有任何一臂落入命中门；`redistribute` 臂是位级 no-op（交付从未触发），`inherit`（及其修复臂）早死。结论是「该处方在本模型上不可用」，不是「物理上无效」。
- 动量交付作用在**所有** I→G 转换上，包括撞击前池面的数值转换：动量踢会轻微扰动撞击前的池面，接触步因此可能与 ctrl 不同（见各臂 stop_reason/接触步），这是诚实的实现后果，不是隐藏的调参。
- 动量踢改变 populations，进而改变 Körner 质量交换的输运，受追踪质量漂移（健康门 1e-4）对两臂都比 ctrl 更敏感；早死即由此而来（如实报告）。
- `inherit` 臂的 ABB 气相速度效应在速度上是**二阶**的（f_eq(q)+f_eq(opp q) 只含 (u·c)² 与 |u|²），因此该臂的动量交付主体是显式的界面注入，而不是 ABB 项本身；两者都实现，量级差异见上节。
- 转换通道量（conv_loss/conv_gain）由 K2 的 budget_diag 只读诊断给出，口径与 K2 完全一致。
- 适用范围：cells=8 单网格、Santini case-3 单工况、单流体自由面模型（常数大气压 + Laplace）。
- **inherit 臂早死**（stop_reason = health_limit，存活 1.3620 ms < 10.4 ms）：按预登记判证伪；早死机制为动量踢扰动质量交换后触发健康门。
- **inherit_fix 臂早死**（stop_reason = health_limit，存活 3.1739 ms < 10.4 ms）：按预登记判证伪；早死机制为动量踢扰动质量交换后触发健康门。

## 7. 复现

```bash
# 1) SDAA 满窗三臂（分相位，避免单次调用阻塞）
python3 tools/sdaa_fs_probe_split.py launch  --arm k3_ctrl         --tree . --steps 10000 --cells 8 \
        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --diag-budget
python3 tools/sdaa_fs_probe_split.py launch  --arm k3_inherit      --tree . --steps 10000 --cells 8 \
        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --diag-budget --momconv inherit
python3 tools/sdaa_fs_probe_split.py launch  --arm k3_redistribute --tree . --steps 10000 --cells 8 \
        --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --diag-budget --momconv redistribute
# 补充臂（各自跑前预登记：PREREG-B / PREREG-C）
python3 tools/sdaa_fs_probe_split.py launch  --arm k3_inherit_fix      --tree . --steps 10000 --cells 8         --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --diag-budget --momconv inherit
python3 tools/sdaa_fs_probe_split.py launch  --arm k3_redistribute_liq --tree . --steps 1200 --cells 8         --gap .18 --speed .06 --rs 1.53 --pd 3.0 --ul .15 --diag-budget --momconv redistribute_liq
python3 tools/sdaa_fs_probe_split.py poll    --arm <臂名> --tree .
python3 tools/sdaa_fs_probe_split.py collect --arm <臂名> --tree .
# 2) 判决（数字全部程序化注入）
python3 tools/k3_momconv_verdict.py --tree .
# 3) 图 + 卡 + 报告
python3 tools/k3_momconv_figures.py --tree .
python3 tools/k3_momconv_card.py --tree . --card docs/drop_entry_bb/X/K3-MOMCONV.md
```

- 工件：`runs/k3_*/{result.json,run.log}`（含 `budget_diag.rows` 逐项时序）、`runs/k3_momconv_verdict/{verdict.json,verdict.md}`
- git 可溯副本：`artifacts/k3_momconv/{k3_*_summary.json,verdict.json}`
- 平台报告发布：id = 22819（摘要修订版；首版 id = 22817，内容等价）
- 单元测试：`tests/test_free_surface_momconv.py`（动量交付守恒、保密度、非负 populations）
- 图：`docs/audits/k3_momconv_20261121/figs/*.png`
- 预登记：`docs/audits/k3_momconv_20261121/PREREG.md`（主赛三臂）、`PREREG-B-inherit-repair.md`（修复臂）、
  `PREREG-C-redistribute-liq.md`（解释臂）—— 全部在各自派车前写入，跑后一字未改
