# 球体 / SUBOFF 高 Re 阻力参考数据（中文版）

> 用途：为高 Re 验证（LBM/CFD 求解器）提供权威阻力参考值。
> 编译日期：2026-08-13
> 置信度标记：
> - **[A]** 教科书/原始论文级，高置信（数值可复核）
> - **[B]** 文献综述/工程惯例，中置信（建议核对原文后再用于验收门槛）
> - **[C]** 任务规格/本地仓库引用，未经原始文献核验（慎用于正式验收）

> ⚠️ **方法声明**：本次编译未执行实时联网检索（本会话无 web 搜索工具）。
> 数据来自本地仓库文档（`TensorLBM_feat2/docs/suboff_reference_data_r1.md` 等）
> 与经典文献知识；所有 [B] 级精确数字在用于验收门槛前应核对原始报告
> （DTRC/SHD-1298-07 等）。**禁止将本文 [B]/[C] 级数值当作已核验的
> 一级来源引用。**

---

## 表 1. Schiller–Naumann 阻力公式及其适用区间

公式（Schiller & Naumann, 1933, VDI Zeitschrift 77:318–320）[A]：

```
CD = (24/Re)·(1 + 0.15·Re^0.687)
```

| Re | SN 公式 CD | 球体实验 CD | 偏差 | 判定 |
|----|-----------|------------|------|------|
| 100 | 1.0919 | ≈1.09 [A] | <1% | ✅ 适用（严格区间内） |
| 500 | 0.562 | ≈0.55 [A] | ~2% | ✅ 适用 |
| 800 | 0.474 | ≈0.47 [A] | ~1% | ✅ 适用（严格上界） |
| 1000 | 0.439 | ≈0.47 [A] | ~−7% | ⚠️ 边缘（部分文献扩展到 Re≤1000） |
| 1×10⁴ | 0.204 | ≈0.40 [A] | −49% | ❌ 不适用 |
| 1×10⁵ | 0.098 | ≈0.40 [A] | −75% | ❌ 不适用 |
| 1×10⁶ | 0.048 | ≈0.1–0.2 [A] | −50%~−75% | ❌ 不适用 |

**结论**：
- **严格适用范围：0.1 < Re ≤ 800**（原始论文）；工程上可放宽至 Re≤1000（误差约 7%）。
- **Re > 1000 后 SN 公式系统性低估 CD**（在亚临界平台期低估约 50%），**高 Re 验证不得使用 SN 公式**。
- 高 Re 分段处理 [A]（标准球体阻力曲线）：
  - 800 < Re < 2×10⁵：亚临界平台，CD ≈ 0.44（0.4–0.5）
  - Re ≈ 2×10⁵–4×10⁵：**阻力危机**，CD 从 0.4 骤降至 ~0.1（转捩敏感区）
  - Re > 4×10⁵：超临界分支，CD ≈ 0.09–0.2（对湍流度/粗糙度高度敏感）
- 参考：Clift, Grace & Weber (1978) *Bubbles, Drops, and Particles*（全 Re 关联式综述）；
  Brown & Lawler (2003), *Powder Technology* 129:38–47（现代全 Re 拟合式）。

---

## 表 2. 球体高 Re 实验阻力系数（标准阻力曲线）

来源：Wieselsberger (1922) 数据（收录于 Schlichting, *Boundary-Layer Theory*）；
Achenbach (1972), *J. Fluid Mech.* 54(3):565–575（Re 至 5×10⁶，光滑球、低湍流度）[A]

条件：光滑球、自由来流湍流度低、无阻塞修正。**CD 基准面 = 迎风截面积 πD²/4**。

| Re | CD 实验值 | 流态 | 备注 |
|----|----------|------|------|
| 10 | ≈4.1 | 层流分离尾迹 | [A] |
| 100 | ≈1.09（1.087） | 层流分离 | [A]（与本地记忆 Re=100 基准一致） |
| 1×10³ | ≈0.47 | 亚临界 | [A] |
| 1×10⁴ | ≈0.40 | 亚临界平台 | [A]（LES 基准见表 5） |
| 1×10⁵ | ≈0.40（0.39–0.46） | 亚临界平台末端 | [A] |
| 2×10⁵ | ≈0.15–0.35 | **阻力危机 onset** | [A] 强烈依赖 Tu/粗糙度 |
| 3×10⁵ | ≈0.07–0.12 | 危机后极小值 | [A] Achenbach 光滑球 ~0.09 |
| **1×10⁶** | **≈0.1–0.2** | **超临界** | [A] 见下方要点 |
| 5×10⁶ | ≈0.18 | 超临界回升 | [A] Achenbach：CD 随 Re 缓慢回升 |

**Re=1×10⁶ 要点（回答"~0.1–0.2?"）**：
- ✅ **是，光滑球超临界分支 CD ≈ 0.1–0.2**（Achenbach 光滑球 ~0.09–0.13；
  Wieselsberger 曲线 ~0.18；工程常用 0.15–0.2）。
- ⚠️ **关键陷阱**：Re=1×10⁶ 可能处于超临界分支（CD≈0.1–0.2）**或**亚临界分支
  （CD≈0.4，若边界层未转捩——极低湍流度 + 极光滑表面时可能发生，且存在双稳态/迟滞）。
  因此高 Re 球体验证**必须同时报告 Tu、粗糙度、阻塞比**，否则 CD 无唯一参考值。
- 粗糙/转捩触发表面（如高尔夫球效应）可将危机提前，CD 落在 0.2–0.4 之间。

---

## 表 3. SUBOFF 总阻力系数实验值（AFF-1 裸艇体 / AFF-8 全附体）

模型参数 [B]：L = 4.356 m，D = 0.508 m（L/D ≈ 8.57）；湿表面积 S(AFF-1) ≈ 6.0 m²，
S(AFF-8) ≈ 6.35 m²（精确值以 Groves et al. 1989 几何报告为准）。
**CT 基准面 = 湿表面积，CT = R / (½ρU²S)**。

| 配置 | Re（基于 L） | CT 参考值 | 来源 | 置信度 |
|------|-------------|----------|------|--------|
| AFF-1 裸艇体 | 1.2×10⁷ | ≈0.0030–0.0032（常用 0.0031） | Roddy (1990) DTRC/SHD-1298-07 水筒实验；SIMMAN 2008 基准 [B] | [B] 精确值需核对原文 |
| AFF-8 全附体 | 1.2×10⁷ | ≈0.0038–0.0042（常用 ~0.004） | 同上 [B] | [B] |
| AFF-8 全附体 | 1.44×10⁷ | ≈0.0037–0.0039（随 Re 略降） | Chase & Carrica (2013), *Ocean Engineering* 60:68–80 所用实验数据 [B] | [B] |
| AFF-8 全附体 | 2.0×10⁶ | 0.0040 ±10% | 本地库 `suboff_reference_data_r1.md`（任务规格引用，未核验）[C] | [C] ⚠️ Re 条件与文献常用 1.2×10⁷ 不同，勿混用 |
| ITTC-1957 Cf（仅摩擦，供对照） | 1.2×10⁷ | 0.00291 ±5% | ITTC 1957 模型-实船相关线 [A]（公式 Cf=0.075/(lgRe−2)²） | [A] |
| ITTC-1957 Cf（对照） | 2.0×10⁶ | 0.00405 ±5% | 同上 [A] | [A] |

**要点**：
- 自洽性检查 [B]：Cf_ITTC(1.2×10⁷)=0.00291，裸艇体 CT≈0.0031 ⇒ 形状因子 (1+k)≈1.06，
  对 L/D≈8.6 轴对称体合理；AFF-8 相对裸体 +25~30% 附体阻力，合理。
- ⚠️ 本地库 AFF-8@Re=2.0e6 的 0.0040 与文献主流的 Re=1.2×10⁷ 条件不一致；
  若 LBM 网格只能达到 Re≈10⁶–10⁷ 量级，建议以 **Re=1.2×10⁷、CT≈0.004（AFF-8）** 为
  文献级锚点，Re=2.0e6 值仅作内部任务参照。
- 原始报告：Roddy, R.F. (1990), *Investigation of the Stability and Control Characteristics
  of Several Configurations of the DARPA SUBOFF Model (DTRC Model 5470)*, DTRC/SHD-1298-07,
  David Taylor Research Center；几何见 Groves, Huang & Chang (1989), DTRC/SHD-1298-01。
- 流场补充实验：Huang et al. (1992), *Measurements of Flows over an Axisymmetric Body with
  Various Appendages (DARPA SUBOFF Experiments)*, Proc. 19th Symp. Naval Hydrodynamics
  （速度型/湍流量剖面，用于验证不止阻力）。

---

## 表 4. SUBOFF CFD 验证基准实践（湍流模型 / 网格 / 阻力误差）

来源：SUBOFF RANS/DES 文献综述 + SIMMAN 2008 Workshop 惯例 [B]

| 项目 | 通行做法 | 典型阻力误差 |
|------|---------|-------------|
| 湍流模型（阻力） | RANS：k-ω SST、Spalart–Allmaras 最常用；realizable k-ε 次之 | CT 相对实验 ±2–5%（细网格）；k-ε 类略大（2–8%） |
| 高保真（流场/分离） | DES、IDDES、LES（AFF-1 常用） | CT ±2–6%（需 20–100M 网格） |
| 近壁处理 | 低 Re 模型 y⁺≈1；壁面函数 y⁺≈30–300 | y⁺ 不当可致 CT 偏差 >10% |
| 网格量 | RANS 阻力 1–10M 单元；DES/LES 50–100M；3 套网格（细化比 ~1.3–1.5）做无关性 | 最细两套网格间 CT 变化 <1–2% 视为网格收敛 |
| 远场边界 | 距艇体 ≥5–10 L | 不足时 CT 系统性偏差 |
| 基准组织 | SIMMAN 2008（哥本哈根）含 SUBOFF 阻力/操纵基准 | 各参与方 CT 散布 ±5–10% |

**对 LBM 求解器的启示** [B]：
- LBM 粗网格 + 壁面函数（本仓库 wall-function 路线）的 CT 误差预算建议取 **±10%**
  （参考本地库对 AFF-8 的不确定度设定），优于 ±10% 才有资格声称"高 Re 物理验证"；
- 球体 Re=1×10⁶ 因双稳态问题不适合做"唯一值"验收，建议验收锚点 = **球体 Re≤1000
  （SN/实验曲线，唯一值）+ SUBOFF AFF-8（Re≈10⁷ 量级，±10% 容差）**。

---

## 表 5. 球体高 Re 的 CFD 基准（补充）

| Re | 基准数据 | 来源 | 备注 |
|----|---------|------|------|
| 3700 | DNS 数据（CD、St、分离角） | Rodríguez et al. (2011), *J. Fluid Mech.* 679:263–287（球体 DNS 里程碑）[A] | 中 Re 验证首选 |
| 1×10⁴ | LES/DES 基准（CD≈0.39–0.42） | Constantinescu & Squires (2003), *Flow Turb. Combust.* 70:267–298 [A] | 亚临界 LES 经典 |
| 1×10⁶ | 实验：CD≈0.1–0.2（表 2） | Achenbach (1972) JFM [A] | 超临界 LES/DNS 文献稀少，验证以实验曲线为主 |

---

## 结论 / 给父任务的建议

1. **Schiller–Naumann 仅适用于 Re≤800（放宽至 1000，误差 7%）**；高 Re 验证用
   分段阻力曲线：亚临界 CD≈0.44（8×10²–2×10⁵）→ 危机区 → 超临界 CD≈0.1–0.2（>4×10⁵）。
2. **球体 Re=1×10⁶：CD≈0.1–0.2 确认成立（光滑球、低 Tu）**，但存在亚临界/超临界双稳态，
   必须报告 Tu、粗糙度、阻塞比；不宜作为唯一值验收锚点。
3. **SUBOFF AFF-8 实验锚点：Re=1.2×10⁷，CT≈0.004（0.0038–0.0042）**；裸艇体 AFF-1
   CT≈0.0031。与本地库 Ct≈0.004（AFF-8）记忆一致；但本地库标注的 Re=2.0e6 条件与
   文献常用 Re=1.2×10⁷ 不同，建议标注区分。
4. **CFD 误差参照**：RANS（SST/SA）±2–5%；粗网格 LBM+壁面函数建议容差 ±10%。
5. **待核验项（[B] 级精确数字）**：Roddy (1990) 报告中的 CT 精确值、S 湿面积精确值、
   Chase & Carrica (2013) 引用的实验 CT——需拿到 DTRC/SHD-1298-07 原文后升级为 [A]。

---

## 参考文献

1. Schiller, L., Naumann, A. (1933). Über die grundlegenden Berechnungen bei der
   Schwerkraftaufbereitung. *Zeitschrift des Vereines Deutscher Ingenieure*, 77, 318–320.
2. Wieselsberger, C. (1922). Weitere Feststellungen über die Gesetze des Flüssigkeits- und
   Luftwiderstandes. *Physikalische Zeitschrift*, 23, 219–224.
3. Schlichting, H. (1979). *Boundary-Layer Theory*, 7th ed., McGraw-Hill.（球体阻力曲线）
4. Achenbach, E. (1972). Experiments on the flow past spheres at very high Reynolds numbers.
   *Journal of Fluid Mechanics*, 54(3), 565–575.
5. Clift, R., Grace, J.R., Weber, M.E. (1978). *Bubbles, Drops, and Particles*. Academic Press.
6. Brown, P.P., Lawler, D.F. (2003). Sphere drag in the motion of settling particles.
   *Powder Technology*, 129, 38–47.
7. Groves, N.C., Huang, T.T., Chang, M.S. (1989). Geometric Characteristics of DARPA SUBOFF
   Models (DTRC Model Numbers 5470 and 5471). DTRC/SHD-1298-01, David Taylor Research Center.
8. Roddy, R.F. (1990). Investigation of the Stability and Control Characteristics of Several
   Configurations of the DARPA SUBOFF Model (DTRC Model 5470). DTRC/SHD-1298-07.
9. Huang, T.T., Liu, H.-L., Groves, N.C., Forlini, T.J., Blanton, J.N., Gowing, S. (1992).
   Measurements of Flows over an Axisymmetric Body with Various Appendages (DARPA SUBOFF
   Experiments). *Proc. 19th Symposium on Naval Hydrodynamics*, Seoul.
10. Chase, N., Carrica, P.M. (2013). Submarine propeller computations and application to
    self-propulsion of DARPA SUBOFF. *Ocean Engineering*, 60, 68–80.
11. SIMMAN 2008. *Workshop on Verification and Validation of Ship Manoeuvring Simulation
    Methods*, Copenhagen（SUBOFF 基准案例）。
12. Rodríguez, I., Borell, R., Lehmkuhl, O., Pérez-Segarra, C.D., Oliva, A. (2011). Direct
    numerical simulation of the flow over a sphere at Re=3700. *J. Fluid Mech.* 679, 263–287.
13. Constantinescu, G.S., Squires, K.D. (2003). LES and DES investigations of turbulent flow
    over a sphere at Re=10,000. *Flow, Turbulence and Combustion*, 70, 267–298.
14. 本地仓库：`TensorLBM_feat2/docs/suboff_reference_data_r1.md`（AFF-8 Ct=0.0040@Re=2e6，
   标注"primary-source verification pending"，本表标记为 [C]）。
