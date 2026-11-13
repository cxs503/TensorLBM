# AUDIT m18753 — XVAL-LBM-TWO-CONSERVATION-ZOUHE-WITH-CHEMICAL-POT

审计任务 m18785（board-audit）· 被审 `/root/tp-agent/work/m18753`（mission 18753，state=succeeded, accepted=true）
审计者不重做科研，只做交差核验：三件套是否真存在、基准判据是否成立、级别是否与功能实质相符。被审树只读；本审计的一切重算写在
`artifacts/audit_m18753/repro/`（mission 源码逐字副本，sha256 见 `repro/mission_src/SHA256SUMS.txt`）。

## 0. 结论（一句话）

**L2（独立功能实现）成立；L3/L4 有反证。三件套里"基准"这一件是空的：该 anchor 在 `tensor_benchmark_cases`
里 0 行（全表 113 行）——基准裁判无数据，报告好看不算数。另有一处硬伤：全部交付文件在
`/root/tp-agent/work/m18753` 是 **untracked**，`git log --all` 与 main@HEAD 的树里都不存在 → 仓库侧没有任何能力增量。**

## 1. 取证清单（三件套逐项核）

| 项 | 声称 | 核验结果 | 判据 |
|---|---|---|---|
| run_artifacts | `artifacts/xval_two_conservation/{results.json,fig_conservation.png}` | **存在**：results.json 5609 B / fig 86 507 B；数值与报告 §4 逐行一致 | REAL |
| run_artifacts（报告 §6 证据） | 墙壁-only 600 步稳定 [-1.000,1.000]；周期盒 1500 步 | **无工件落盘**（交付内不存在这两个 run 的任何 json/log） | WITHHELD |
| impl_files | `src/tensorlbm/phasefield/zouhe_two_conservation.py`（312 行）、`free_energy_channel.py`（386 行） | **文件存在且是真代码**（无 `NotImplementedError`/`pass` 占位）；但两文件在 git 里 **untracked**，main 树 `git ls-tree -r HEAD` 命中 0 | 真实但未入库 |
| tests | `tests/test_zouhe_two_conservation_chemical_potential.py`（4 gate 全绿） | **本次复跑 4 passed in 4.63 s** | REAL |
| entry | `reconstruct_inlet` / `reconstruct_outlet` / `FreeEnergyChannel2D.step` | 真入口，`run_xval.py` 调用链完整 | REAL |
| repro_cmd | `PYTHONPATH=src python3 xval/<anchor>/run_xval.py` | **可执行**；本次复跑得到同样的定性结论，但 B1 数字不逐位复现（见 §3 F6） | 部分可复现 |
| benchmark | 报告 §4 的 ref vs got 判据 | **`tensor_benchmark_cases` 该 anchor = 3 行**（表内共 113 行）→ 唯一裁判无数据 | 缺失 |
| external_anchor | `ebinan92/Fingering_dynamics` `fingering.py:303,304,322,369` | **源码级核对成立**（`/tmp/Fingering_dynamics` HEAD 150a5739，2019-02-28）：:303-304 `rho_inlet = 1/(1-ux)(f0+f2+f4+2(f3+f6+f7) - psi_x*mu/2)`；:305 `psi_in = 1-(...)`；:312/322/332 `g[i] = w[i]*psi_in/(w1+w5+w8)`（= 交付的 `impose` 变体）；:369 `def zou_he_boundary_outlet`。交付对上游的公式转写**准确** | 源码级 REAL / **数值级未做** |

## 2. 基准裁判（`tensor_benchmark_cases`）核对

```sql
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-TWO-CONSERVATION-ZOUHE-WITH-CHEMICAL-POT';
-- 3 rows (table total = 113)
```

结论：**该 anchor 没有任何登记行**，所以「got 是否在容差内」在机器判据上不可判定；交付里所有的"判据/打勾"都是
自声明（报告 §1 的 predeclared 文本 + §4 的自评）。本审计已按 m18649 的先例补登 3 行可机核的 case（标记
`ROW REGISTERED BY audit m18785`），其中通道稳定性一行**登记为 failed**（honest-FAIL 与实测一致）。

## 3. 独立复算与反例（审计自己的臂，先声明后跑）

复算脚本 `artifacts/audit_m18753/repro/audit_probe.py` / `audit_probe2.py`，读数
`repro/probe_out/audit_probe*.json`，图 `figures/fig_m18753_audit.png`。
**P0 保真门**：审计自己写的 `step` 副本在 mission 设置下复现了交付的发散步数 **83**（交付 83），
故后续各臂可采信。

| 臂 | 设置 | 结果（600 步上限） | 读数 |
|---|---|---|---|
| A | mission 默认 `flux/corrected` | 第 **83** 步发散 | 复现交付 83 |
| E | mission `classic` | 第 **86** 步发散 | 复现交付 86 |
| D | mission `impose`（交付唯一"稳定"臂） | 600 步不发散，但 `max|psi|`=1.3472（**越出物理界 1**）、中截面剖面误差 **19.19 %**（交付在 200 步时是 9.60 %） | 稳定≠合格；门槛随 horizon 漂 |
| C | `flux` 入口 + 出口 psi 取上游列（零梯度） | 第 **245** 步发散 | 见 F5 |
| G | 关掉开边界（x 周期 + 半程反弹墙） | 600 步有界，`max|psi|`=1.00225 | 支撑"失效在开边界闭包" |

### 发现（按严重度）

**F1 [blocking] 交付未入库。** `git -C /root/tp-agent/work/m18753 status --porcelain` → 7 条 `??`
（4 个交付文件 + 图 + docs + xval 目录）；`git log --all --oneline -- '*zouhe_two_conservation*'` 为空；
main@d2cf04a 树内 `git ls-tree -r HEAD | grep -c` = 0，全仓 `grep -rl` 调用方 = 0。
⇒ 平台侧与第三方都拿不到能力增量，L3 反证成立（不是"缺证"而是"不在仓里"）。

**F2 [blocking] 基准缺失。** 见 §2，anchor 0 行。

**F3 [blocking] 核心声称（含化学势的双守恒 Zou-He 已验证）在求解器层面不成立。** 交付自己的 B4：
`corrected` 83 步、`classic` 86 步发散；唯一存活的 `impose` 在 200 步已达 9.60 %，
600 步升到 19.19 %（>15 % 判据），且 `max|psi|` 从 1.124 涨到
1.347（越出双阱物理界）。交付报告已如实写 honest-FAIL，这点**没有粉饰**；但"验证通过"的
整体位置只能是算子级，不是能力级。

**F4 [major，夸大] B1/B2 是代数恒等式，不构成物理验证，更不构成 XVAL。**
B1 检验的重建式就是"令 `sum f = rho`、`sum c f + F/2 = rho u`、`sum g = psi`、`sum c_x g = psi u_x` 反解出的闭式"，
满足到 1e-15 是构造必然（审计 P1 用独立 numpy 重推同一套公式，mass-flux 通道残差
6.25e-17）。B2 的"缺陷律"同理：
用 `force_correction=False` 重建后改用 Guo 半力度量，必然差 `|F|/(2 rho)`——这是两种约定的定义差，
不是与任何外部/解析基准的比较。**全交付没有任何一处把 LBM 数值与上游 `Fingering_dynamics` 的输出、
或解析流场做过数值对比**；`XVAL` 名号只有源码级公式转写支撑。

**F5 [major] 报告 §6 的根因推断被审计反例否证（部分）。** 报告称"增长源定位在最后一列（出口）——即 psi 闭包本身"。
审计臂 C 把出口 psi 换成上游列（零梯度，仍是同一 `reconstruct_outlet`）后，发散只是从 83 步**推迟**到
245 步，且 `max|psi|` 的极值点从 col_of_max=47（出口）搬到 **col 0（入口）**
（见 `audit_probe.json` C 臂 trace 与图 (a)）。即：入口的 flux-consistent psi 闭包同样不稳，
两条边界的 psi 闭包都在墙邻行（row 1）长大。⇒ 报告的"下一轮从出口回压/零梯度入手"的整改方向**不足以**解决问题；
ψ 开边界闭包 + 角点/墙邻行耦合才是正确靶点。

**F6 [minor] B1 不可逐位复现（跑分本身不确定）。** `run_xval.py:b1_invariants` 用
`random_column(ny, seed=ny + hash(mode) % 97)`，`hash(str)` 每进程加盐。审计同种子（同 PYTHONHASHSEED）两次逐位相同，
换 seed 后同一 ny 的 `corrected mass_flux_rel` 从 1.866e-15 变到
4.491e-15（最大比 1.212）；本次整体复跑与交付工件的
`corrected_mass_flux_rel_max` 5.0944e-15 vs 6.5060e-15
不一致。**结论不翻**（都是 ≤1e-13），但"记录数字"不可复现，若登记成基准 case 就会变成一张漂移的基准。

**F7 [minor] 文档公式与代码不一致：入口横向目标速度被硬钉为 0。** 文档 §2 写 `D = rho*u_y - Fy/2 - f2 - f6 + f4 + f7`，
代码 `zouhe_two_conservation.py:168` 是 `d58 = -half*fy - f2 - f6 + f4 + f7`（缺 `rho*u_y`），且
`reconstruct_inlet` 签名根本没有 `uy` 形参。审计 P1 用独立实现取 `u_y,target=0.01`：两版 f 种群差
**5.3675e-03**，而 mission 代码重建后的 `u_y` 仍是 **7.1e-17**（=0）。
对本题（`u_y=0` 入流）无害 → 不改变级别，但文档声称的通式代码没实现。

**F8 [minor] 报告缺 `<3 %` 的显式回答。** 报告只答 15 % owner 带（且 B4 老实说"不能声称"），
owner 2026-10-29 的 ⑥ 要求逐项显式回答是否 <3 %。就本单实质：B4 唯一存活臂 200 步 9.60 %、
600 步 19.19 %，**均不满足 3 %**；B1/B2 的 1e-15 不适用（是恒等式，不是误差）。

## 4. 判级

**level = 2（独立功能实现）**，理由与反证：

* L1 成立：4 个单元 gate 真跑真绿（本次复跑 4 passed）。
* **L2 成立**：`zouhe_two_conservation.py` 312 行真算子（输入 f/g/u/F → 输出 f/g + 不变量），
  `free_energy_channel.py` 386 行真求解器驱动，审计用独立 numpy 重推同一闭式并把不变量复算到
  6.2e-17。
* L3 不成立（反证）：文件未入库（F1）+ 全仓调用方 0 + 无平台/主求解器接线。
* L4 不成立（反证）：`tensor_benchmark_cases` 该 anchor 0 行（F2）+ 唯一 solver 级算例发散（F3）
  + B1/B2 非外部基准（F4）+ 跑分不确定/无收敛研究（F6）。
* L5 不成立：无实验/文献数据对标；上游仓库只有源码级转写，未跑其数值。

## 5. 未跑的检查与局限

* 未编译/未运行上游 `ebinan92/Fingering_dynamics`（MATLAB/CuPy 依赖未知），故"与上游数值一致"仍 **WITHHELD**；
  source-level 核对只见公式级转写正确。
* 未做网格收敛研究（`ny=34/nx=48` 单一算例），未做界面分辨率（`H/xi`）扫描。
* 审计的反例臂 C/D/G 用的是交付 `step` 的**逐字副本 + 两个开关**（保真门 P0 已对齐发散步），
  不等价于对交付代码的全覆盖改造测试。
* 审计与被审单元同体制（同一 harness），只对机器可核事实负责；本报告没有对被审者的诚实性做判断，
  只记录其报告与实测的差异。

## 6. 整改建议（reopened 的最小闭环）

1. **入库**：把 4 个交付文件 + 工件 commit 到 `agent/m18753`（F1）。
2. **基准**：本审计已补登 3 行 case（含 1 行 failed），通道稳定性要真正 pass 必须过 600 步 + 15 %。
3. **ψ 开边界**：按 F5 的靶点改——两侧边界 ψ 闭包都换掉（回压/零梯度 + 角点补齐），
   或在 ψ 上做 ghost/缓冲层而不是覆盖式重建；验收判据：600 步 `max|psi| ≤ 1.0 + 1e-2`、剖面 <3 %。
4. **可复现**：`random_column` 的种子改成字面常量（去掉 `hash(mode)`），并把 `uy` 目标显式入参（F6/F7）。
5. **对标**：真正跑一次上游算例（或其等价 MATLAB 参数）并把输出落成 `ref_vs_got` 表，才能自称 XVAL。

---
机读证据：`artifacts/audit_m18753/audit_m18753.json`；复算：`repro/audit_probe.py`、`repro/audit_probe2.py`；
图：`figures/fig_m18753_audit.png`。
