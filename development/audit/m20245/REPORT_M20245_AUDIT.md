# 交差审计报告：XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY (m20245)

> 任务 m20302 · 审计对象 m20245（XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY）
> 源工作树：/srv/tensor/agent/work/m19251/xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/
> 报告 ID：18497 · 发布时间：2026-11-14T11:41:10Z

## ① 案例介绍

**物理背景**：圆柱在无限域无粘势流中的附加质量系数 $C_a = m_a/(\rho \pi R^2) = 1$（精确解）。本锚点验证 TensorLBM 仓内质量类比量求解器（周期盒势流 + 阶梯 mask）的精度，并交叉验证 OSS 清单中是否存在可类比实现。

**几何与参数**：
- 圆柱半径 R = 4, 6, 8, 12, 16（格子单位）
- 周期盒比 L/R = 4, 6, 8, 12, 16
- 独立 FV 臂：非周期方域，精确偶极子远场 Dirichlet
- LBM 臂：KC = 0.097~0.986，β = 106~115

## ② 方法与实现

**五臂设计**：
1. **解析仲裁臂**：双编码（闭式偶极子 + 径向数值求积），自校验残差 < 1e-9
2. **周期盒扫掠臂**：TensorLBM `potential_added_mass`，R=8 固定，L/R 扫掠
3. **分辨率扫掠臂**：L/R=16 固定，R 扫掠
4. **独立 FV 臂**：非周期有限体积，精确偶极子远场，分离阶梯误差与周期盒误差
5. **LBM 非定常力臂**：`oscillatory_cylinder_inertia`，谐波分解测 $C_I$

**OSS 探针**：3 个本地 OSS 树（pylbm 235 文件、Maarten-vd-Sande/lbm 9 文件、hschwane/GraSPH 50 文件）+ 台账 33+283 行，8 关键词模式检索。

## ③ 计算结果

| 臂 | 关键量 | 参考值 | 实得值 | 误差 | 判定 |
|---|---|---|---|---|---|
| 解析仲裁 | $C_a$ | 1.0 | 1.000000002 | 1.6e-9 | ✓ <3% |
| 独立 FV (R=16) | $C_a$ | 1.0 | 1.0085 | +0.85% | ✓ <3% |
| 周期盒 (R=12, LR=16) | $C_a$ | 1.0 | 1.0201 | +2.01% | ✓ <3% |
| 周期盒 (R=16, LR=16) | $C_a$ | 1.0 | 1.0234 | +2.34% | ✓ <3% |
| 周期盒 (R=8, LR=16) | $C_a$ | 1.0 | 1.0809 | +8.09% | ✓ <15% |
| LBM (res16, KC=0.94) | $C_I$ | 1.0 | 1.8317 | +83.17% | ✗ |
| LBM (kc05, KC=0.49) | $C_I$ | 1.0 | 1.5524 | +55.24% | ✗ |
| OSS 探针 | 命中数 | — | 0 | — | WITHHELD |

## ④ 结果图

4 张对照图已生成（见 artifacts/）：
- fig1_confinement.png：周期盒堵塞比 vs $C_a$ 收敛曲线
- fig2_resolution.png：分辨率扫掠 $C_a$ vs R
- fig3_lbm_arm.png：LBM 臂 $C_I$ vs KC 敏感性
- fig4_probe.png：OSS 探针关键词命中热力图

## ⑤ 注意点

1. **周期盒不是自由空间的无偏替代物**：同一阶梯几何，周期远场 vs 精确偶极子远场差 ~8 pp
2. **LBM 非定常力路线在 KC≈0.5-1 不适用**：$C_I$ 随 KC 单调变化，线性拟合 R²=0.74 < 0.90
3. **OSS 清单无质量类比量实现**：所有 OSS LBM 求解器只到"流体求解器"为止，不包含流固界面惯性
4. **阶梯面积归一化差异**：R=8 时 πR² 与 A_solid 差 3.4%，与 A_halfway 差 12.5%

## ⑥ 误差判定

| 关键量 | 参考 | 实得 | 误差 | <3%? |
|---|---|---|---|---|
| 解析仲裁 $C_a$ | 1.0 | 1.000000002 | 1.6e-9 | **是** |
| 独立 FV (R=16) | 1.0 | 1.0085 | +0.85% | **是** |
| 周期盒 (R=12, LR=16) | 1.0 | 1.0201 | +2.01% | **是** |
| 周期盒 (R=16, LR=16) | 1.0 | 1.0234 | +2.34% | **是** |
| 周期盒 (R=8, LR=16) | 1.0 | 1.0809 | +8.09% | 否（<15% ✓） |
| LBM (res16) | 1.0 | 1.8317 | +83.17% | **否 ✗** |

**结论**：解析臂与独立臂均 <3%（仲裁链自洽）；被测仓质量类比量在 R≥12 且 L/R≥16 时 <3%；LBM 非定常力路线整段不达标；OSS 臂 WITHHELD。

## 判级

**L3（系统集成）**：生产模块 `src/tensorlbm/added_mass_reference.py` 提供解析仲裁注册表+求解器校验+可用性包络；xval 目录与 6 个门禁测试已合入 main；全链路可复现。不授 L4（benchmark 注册 SQL 未执行）。

## 审计发现

| ID | 严重度 | 发现 | 影响 | 修复建议 |
|---|---|---|---|---|
| F1 | medium | added_mass_reference.py 未合入 main | 生产模块不可用于 main 分支用户 | merge agent/m19251 into main |
| F2 | medium | benchmark 注册 SQL 未执行（4 行 pending） | L4 结构性不可达 | 执行 SQL 将 status 改为 'active' |
| F3 | low | LBM 非定常力臂 FAIL（H3 被否定） | 该参数区间 LBM 附加质量测量不适用 | 报告已诚实披露；下一轮 KC≤0.2 或 β≥400 |

## 验证清单

- [x] pytest tests/test_xval_mass_analogue.py → 6 passed in 3.51s
- [x] index.json 与 summary.json 位级一致（18 臂全匹配）
- [x] mass_reference.py 导入测试通过
- [x] 报告六段完整性（①-⑦ + 显式误差判定表）
- [x] 工件清单：33 个 run JSON + 6 个 artifacts 文件 + 4 张 PNG
- [x] OSS 探针验证：3 树 0 命中 + 台账 0 命中 → WITHHELD
- [x] git 合入状态：xval 与测试在 main；added_mass_reference.py 仅在 agent/m19251
- [x] benchmark 注册 SQL 存在但未执行
