# NACA0012 Re=1000 α=0° 阻力系数 Cd 参考口径多源核对

本文件裁定本 benchmark **唯一正确的对标口径**（Cd at α=0°），列可追溯来源与排除项。
裁定结论先行：**CD_REF = 0.12**（脚本原写死的 0.105 无任何来源支撑，予以否决）。

## 0. 问题口径 ↔ 参考口径

物理问题：均匀来流绕 **NACA 0012 对称翼型**，基于弦长的 Reynolds 数
`Re = U∞·C/ν = 1000`，攻角 `α = 0°`。α=0 时对称翼型无升力（Cl≡0），drag 是
唯一可观测量，且**摩擦（黏性）阻力主导**（Kurtulus 2015 正文："until 8° angle of
attack viscous drag is the dominant term for the drag coefficient"）。

量纲口径：`Cd = Fx / (½ ρ U∞² · C)`（二维、单位展长），与 wind-tunnel/CFD 文献约定一致。

## 1. 采纳口径（主锚）：Kurtulus 2015 正文

Kurtulus, D. F. (2015), *On the Unsteady Behavior of the Flow Around NACA 0012
Airfoil with Steady External Conditions at Re=1000*, **Int. J. Micro Air Vehicles
7(3):301–326** (DOI 10.1260/1756-8293.7.3.301)。全文 PDF 本地
`/tmp/kurt.pdf`（2,858,846 B，与 W3-D 记录字节数一致），文本抽取 `/tmp/kurt.txt`。

**正文逐字（`/tmp/kurt.txt` 第 173 行）：**

> "Drag coefficient and zero angle of attack is found to be about 0.12 for NACA 0012
> airfoil at Re=1000."

同一段落给出的自洽标定（证实该数值是完整 Cd–α 曲线的一部分，非孤点）：
- 平均 Cd 在 α≈26° 达拐点，Cd ≈ **0.69**；
- α=90° 时 Cd ≈ **2.8**（std 0.3）；
- 平均 (Cl/Cd) 在 α=11° 达峰值 **2.55**。

→ **锁定 CD_REF = 0.12**（论文文字，Figure 7 Cd 面板同源）。

## 2. 独立交叉源 1：Di Ilio et al. (2020) 混合格子 Boltzmann (HLBM)

Di Ilio, G., et al. (2020), *Fluid flow around NACA 0012 airfoil at low-Reynolds
numbers with the hybrid lattice Boltzmann method*, **arXiv:2006.10487**（本地
`/tmp/diilio2020.pdf`，2,646,016 B；文本 `/tmp/diilio.txt`）。

**Table 1（α=0°，NACA0012，HLBM vs XFOIL）：**

| Re | Cd (HLBM) | Cd (XFOIL) |
|----|-----------|-----------|
| **1000** | **0.119** | **0.119** |
| 2000 | 0.084 | 0.084 |
| 5000 | 0.052 | 0.054 |
| 10000 | 0.037 | 0.040 |

该 HLBM 弦长 512–1024 lattice units（远细于本基准 C=64/128），二阶空间收敛，
是**独立的贴体曲面数值解**（非阶梯），与 Kurtulus 的 0.12 相差 **0.8%**。

## 3. 交叉源 2：XFOIL (Drela)

Di Ilio 同表给出 XFOIL（Drela, MIT 面板法）在 Re=1000 亦为 **0.119**，
与 HLBM 逐位吻合。XFOIL 是低 Re 翼型设计的工程标准工具，构成第三条链路。

## 4. 被否决的旧值 0.105

脚本原 `CD_REF = 0.105`：**在 Kurtulus 2015 全文（无表值，仅 Figure 4/7 图线与文字）
及 Di Ilio/XFOIL 中均无支撑**，与上述三源（0.12 / 0.119 / 0.119）相差 −12.5%，
属无来源的历史占位值。任务书已裁定为 0.12。**予以否决，改判 CD_REF = 0.12。**

## 5. 明确排除的口径

- **高 Re 风洞实验**（Ladson/NASA TM-4074, Re ~3–6×10⁶）：Re 区间与 2-D 层流
  物理均不符，不适用。
- **三维/有限展长翼**：本基准是二维单位展长；三维翼有诱导阻力，口径不同。
- **本次求解器自身的历史值**（W3-D α=5° 阶梯 Cd≈0.134）：非参考，仅作交叉观测量级。

## 6. 判定门

`|Cd_sim − 0.12| / 0.12 ≤ 3%` ⇔ **Cd ∈ [0.1164, 0.1236]**。

多源散布：0.119（Di Ilio HLBM）—0.12（Kurtulus 文字），跨度 **0.8% ≪ 3%**，
故 3% 门比参考自身不确定性更宽，判定是稳健的。任务书披露的保守簇 (0.09, 0.12)
下边界 0.09（粗网格/不同黏性口径）不改变主锚 0.12 的采用。

## 7. 与力法口径的关系（本 case 的关键）

α=0° 时阻力由**摩擦主导**，故对**壁面摩擦的分辨率**极敏感：
- Re=1000 下边界层厚度 δ ~ C/√Re，**C=64 仅约 2 格、C=128 约 4 格**跨边界层；
  本库用**半程反弹 + 阶梯边界**（无曲面贴体处理），摩擦有欠解析风险（如实披露）。
- 力口径 = **Ladd (1994) 动量交换，仅对 surface（壁邻）solid 单元求和**，后处理场
  （post-step, post-BB）取负，见 `docs/mem_surface_caliber_finding.md`；all-solid 与
  interior-only 求和作为诊断一并报告（内部伪力）。

## 8. 工件与可追溯性

- `/tmp/kurt.pdf`（Kurtulus 2015 全文）+ `/tmp/kurt.txt`（文本抽取，第 173 行引文）
- `/tmp/diilio2020.pdf`（arXiv:2006.10487）+ `/tmp/diilio.txt`（Table 1 引文）
- 结果见同目录 `README.md` 与 `result.json`（2 档网格 C=64/128，各 |err|≤3% 且跨度≤3% 方入 verified）。