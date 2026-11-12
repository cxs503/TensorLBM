# CASE-LBM-SQUARE-CYL-NORBERG / Re=200：参考表首次机器审计——三处无源数字被查出并修复

锚点 `CASE-LBM-SQUARE-CYL-NORBERG` · mission **m16678**（verifier 席）· repo TensorLBM
分支 `agent/m16678`。本单不给新流场，给的是**可机械复核的裁判口径**。

## ① 案例介绍（物理背景 / 几何 / 参数）

二维方柱绕流（法向入射 α=0°），锚点要求 `Re = U·D/ν = 200`、阻塞比 `B = D/H = 1/20`，
两个目标量 `St = f·D/U` 与时间平均 `Cd`，锚点容差 `St < 8%`、`Cd < 10%`。
裁判面 `benchmarks/verified/square_cylinder_re200/reference.json` 分级承载：
`VERIFIED_TABLE`（真从全文读出的数）/ `WITHHELD`（拿不到原文，故意不写数）/ `TREND`（插值，不作裁判）。

| 项 | 值 | 来源 |
|---|---|---|
| 锚点参考（原文） | Norberg 1993 JWEIA 49:187；Sohankar–Norberg–Davidson 1998 IJNMF 26:39 | 锚点文本 |
| 可得的受钉全文 | arXiv 2411.03124v4 Table 1（α=0° 块） | ar5iv，sha256 钉死 |
| 已分级行 | Re=100/150 的 Cd、St（VERIFIED_TABLE）；Re=200 的 Cd、St（WITHHELD） | reference.json |

## ② 怎么做的（方法 / 工具链）

1. 新增 `benchmarks/verified/square_cylinder_re200/audit_reference.py`：把「引文信任」换成
   「引文复核」。它先校验被审计全文的 **sha256 是否等于钉死值**（不同副本直接拒绝审计），
   再把 ar5iv 的 LaTeX-in-HTML 表格重新解析成数字，逐格与 `reference.json` 的 `ref`/`band` 比对；
   `WITHHELD` 行必须保持 `ref=null` **且**写出 `withheld_reason`；`TREND` 行必须自证「非裁判」。
2. 无 `--source` 时只报 **SKIP**（不是 PASS）：没审过的东西不许记成通过。
3. `--selftest` 用合成文档证明审计器**有判别力**：忠实文档 11/11 OK，篡改一个 Cd 格 → 2 FAIL，
   给 `WITHHELD` 行偷填一个「记忆值」→ 1 FAIL。
4. 对被钉全文实跑一次，产出 `reference_audit.json`（含 source sha256 与逐项 OK/FAIL）。

## ③ 计算结果（首次审计发现，程序化输出零手抄）

对**原样提交**的 `reference.json` 首次实跑：`counts: OK 8 / FAIL 3`。三处 FAIL 都是真缺陷：

| 行 | 原值 | 全文实际列（Present / Sohankar / Yoon） | 判定 |
|---|---|---|---|
| Re=100 `cd` | `ref=1.47`、band [1.44,1.48] | 1.48 / 1.46 / 1.44 | **1.47 不在任何来源列**（是聚合值），且与姊妹算例 `square_cylinder/refspec.py` 的 `REF_CD=1.48` **不一致**——正是 audit m16183 指控的「同一测量两个常量」 |
| Re=100 `st` | band 下边缘 0.145 | 0.147 / 0.147 | **0.145 全文不存在**（来自未钉源的簇） |
| Re=150 `st` | `ref=0.160`、band [0.159,0.162] | 0.159 / 0.162 | 0.160 是两列中点（聚合，非引文） |

修复（旧值留在 `ref_old_unaudited` / `band_old_unaudited`，不静默改）：
Re=100 Cd→**1.48**（Present，与姊妹算例同一常量）、Re=100 St→**0.147**（band 收窄为 [0.147,0.147]）、
Re=150 Cd→**1.41**、Re=150 St→**0.159**（band 保留 [0.159,0.162]）。
`TREND` 行据此重推（Re=200 非裁判外推）：Cd 1.375（±0.05）、St 0.171（±0.011）。

修复后实跑：**`counts: OK 11 / FAIL 0`**，`reference_audit.json` 固定 sha256
`f95490265a063…397d7a`；`--selftest` 仍对篡改/伪造 FAIL（判别力未被放宽换绿）。

## ④ 结果图

本单为参考口径审计（表格 + 机器门），无场量云图、无解析解曲线可比：审计对象是**数字的出处**
而不是流场，因此不出图。可视化判据在算例自身（`make_figures.py`：尾迹涡量、力历史、分辨率研究），
本单未触碰其物理内容。

## ⑤ 注意点 / 局限

* **Re=200 的实测仍未产出**：本单提交的集群作业 `cj1340`（D=24/32, Re=200, B=1/20,
  `--periods 24`）在窗口内一直排队（`state=queued`），本地 D≥16 单点预估 40 min–2 h（load15≈90），
  故本单**不宣布任何 Re=200 的 Cd/St**。审计器与参考表已就绪，作业落地后 `run.py` + `verify.py`
  可直接给数——这是留给下一单的明确接口，不是含糊待办。
* 锚点名义参考（Norberg 1993 实验 / Sohankar 1998）仍 **WITHHELD**：本机只有 arXiv 可达，
  出版商 403，且 2D 深度平均求解器本不该直接对标 3D 风洞值；本单**没有**用记忆值或替身值掩盖这一缺口
  ——反而把这一状态写成了被检查的不变量。
* 全文 sha256 钉的是 `ar5iv` 渲染版；若上游重渲染，审计会以 MISMATCH 拒绝比对（这是设计意图，
  不是故障）：换版本须显式更新 `SOURCE_SHA256` 并重跑。
* `--selftest` 是结构性判别力检查，不替代真实全文审计；两者都跑才记 PASS。

## ⑥ 误差判定（关键量 vs 基准）

| 关键量 | 基准 | 本单结果 | 是否 <3% |
|---|---|---|---|
| Re=100 `Cd` 常量 | 全文 Present 列 1.48 | 修复后 1.48，偏差 **0.00%** | 是 |
| Re=100 `St` 常量 | 全文 0.147（两列一致） | 0.147，偏差 **0.00%** | 是 |
| Re=150 `Cd` 常量 | 全文 Present 1.41 | 1.41，偏差 **0.00%** | 是 |
| Re=150 `St` 常量 | 全文 Present 0.159 | 0.159，偏差 **0.00%** | 是 |
| Re=200 `Cd` / `St`（实测） | 锚点原文 | **未产出**（集群 cj1340 排队） | 不适用，无基准亦无实测 |
| 审计器判别力 | 篡改/伪造样本 | 篡改 2 FAIL、伪造 1 FAIL | 是（反向判别成立） |
