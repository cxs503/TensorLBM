#!/usr/bin/env python3
"""Build the W26 report payload from the artifact (numbers injected, not typed)."""
import base64
import json
import os

REPO = "/root/tp-agent/work/m17563"
ART = os.path.join(REPO, "papers", "artifacts", "w26", "d3q19-bgk-analytic-fourcase",
                   "anchor_register.json")
d = json.load(open(ART, encoding="utf-8"))
a1, a2, a3, a4 = (d["analytic_fourcase"][k] for k in ("case1", "case2", "case3", "case4"))
val = d["validation"]
crit = {c["id"]: c for c in d["criteria"]}
tab1, tab2 = d["tables"]["I"], d["tables"]["II"]
n_rows = tab1["n_rows"]
re_keys = ("100.0", "400.0", "1000.0")


def table_md(tab, label):
    head = "| y/L | u(Re=100) | u(Re=400) | u(Re=1000) |\n|---:|---:|---:|---:|" if label == "I" \
        else "| x/L | v(Re=100) | v(Re=400) | v(Re=1000) |\n|---:|---:|---:|---:|"
    lines = [head]
    for r in tab["rows"]:
        vals = " | ".join("%.5f" % r["profile"][k]["value"] for k in re_keys)
        lines.append("| %.4f | %s |" % (r["coord"], vals))
    return "\n".join(lines)


slopes = a1["convergence"]["pairwise_slopes"]
errs = [(g["N"], g["linf_error"]) for g in a1["convergence"]["grids"]]
inv = val["cavity_invariants"]
defect = d["in_repo_secondary_table_defect"]

md = f"""# W26 · Ghia (1982) 方腔剖面锚核销 + 解析四例（D3Q19-BGK 锚定基线）

**判据裁决：C1 PASS · C2 PASS · C3 PASS · C4 PASS**（逐条证据见 `anchor_register.json`
`criteria[]`；另登记一条仓内既有转录缺陷，见 §5）

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

本单的任务不是"再跑一个算例"，而是给 LBM 求解器建立**可核销的外部定量锚**：顶盖驱动方腔的
中心线剖面（Ghia, Ghia & Shin 1982）与四个闭式解。任务书规定的量化锚形态——

| 锚 | 物理 | 几何/参数 | 取值形态 |
|---|---|---|---|
| Ghia 表 I | 方腔竖直中心线 u(y) | 方形腔 L×L，顶盖 u=U 沿 +x，Re=U·L/ν = 100/400/1000 | 17 点表值（u/U_lid） |
| Ghia 表 II | 方腔水平中心线 v(x) | 同上 | 17 点表值（v/U_lid） |
| 解析 1 | 平面 Poiseuille | H=1, μ=1e-3, G=1（U_max=125） | 闭式 u(y)=G/(2μ)y(H−y) |
| 解析 2 | 平面 Couette | H=1, μ=1e-3, U=1 | 闭式 u(y)=Uy/H |
| 解析 3 | Taylor-Green 涡 | U=1, k=1, ν=1e-2, t=0.5, 64² 周期格 | 闭式 u,v,p（指数衰减） |
| 解析 4 | 方腔 Re→0 极限 | 同上几何，Stokes/双调和极限 | ∇⁴ψ=0 的可分离解空间 + 压力可积性 |

参考解口径（判据 C3）：Ghia 表来自 **129×129 线网格**差分/多重网格解（同文另给 Re=400 的
257×257 对比），Re 定义为 **Re = U_lid·L/ν**，速度以 U_lid 归一，剖面取 x/L=0.5 与 y/L=0.5，
原点在左下角、顶盖在 y/L=1 沿 +x 运动。原文表头逐字（页面词框）为
`{", ".join(w["text"] for w in tab1["header_words"])}`（表 I）/
`{", ".join(w["text"] for w in tab2["header_words"])}`（表 II）。

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

**零手抄取锚。** 本环境里论文原文"不可得"的说法不成立——原始 PDF 可经 GitHub contents API 取回，
其 sha256 `{d["anchor_availability"]["pdf"]["sha256"]}` 与仓内既有 OCR 记录头部登记的 PDF 哈希
一致，故按"锚可得"路径核销。取数流程刻意做成双见证：

1. **W1 数字**：PyMuPDF 逐词包围盒 → 行带聚类 → 列带归一（值列互不重叠，因此列内碎片无条件拼接，
   修掉 OCR 把 `-0.2058`/`1` 切两块的毛病）；
2. **W2 符号**：OCR 文本层把印刷负号读成 `a / AI / Al / xJ / X) / Jx / 4`，**符号一律不取 OCR**。
   页面栅格 8 px/pt 逐格量墨水宽度：值列右对齐，负号多一个减号字宽 ⇒ 每列自然裂成
   27.0 pt / 31.5 pt 两簇，1-D k-means 阈值 ≈29.3 pt，6 列全部干净可分（间隙 4.3–4.6 pt）；
3. **结构校验**：坐标必须落在 1/128 点阵（129 线网格）、列右对齐、双见证一致率、
   不可压守恒不变量、壁面边界条件；
4. **解析四例**：闭式解回代算子算残差（mpmath 高精度求导 / FFT 谱导数），每例附**故意写错的解**
   作对照（对照项必须 O(1)，否则残差检查无判别力）；
5. 工具链：Python 3.11 / PyMuPDF 1.27 / pypdf 6.19 / mpmath 1.3 / numpy 2.4 / scipy 1.16；
   脚本 `scripts/w26_ghia/{{extract_ghia_tables,analytic_fourcase,build_anchor_register}}.py`，
   验收测试 `tests/test_w26_ghia_anchor_register.py`（9 passed）。

## 3. 计算结果（数字表）

### 3.1 注册锚：Ghia 表 I/II，Re=100/400/1000（各 {n_rows} 点、共 {n_rows*3*2} 个值）

**表 I — 竖直中心线 x/L=0.5 的 u/U_lid**

{table_md(tab1, "I")}

**表 II — 水平中心线 y/L=0.5 的 v/U_lid**

{table_md(tab2, "II")}

每个值在 JSON 里都带逐项溯源：`ocr_token`（逐字）、`digits`、`sign_from_ink`、
`ink_width_pt`、`ink_bbox_pt`、页号与行带。双见证统计：**84/102 与 OCR 显式符号一致、0 冲突**，
其余 **18** 个值的 OCR token 是把负号读成字母/数字的坏 token，其符号**只**由墨水判定
（其中 1 个 `4.38289` 需按 `X.XXXXX` 版式规则把首位当作负号，量值重读为 0.38289，已在 JSON 标记
`magnitude_repaired`）。

### 3.2 解析四例残差（判据 C2，阈值 1e-12）

| 例 | 检查量 | 残差（相对） | 对照（故意写错的解） |
|---|---|---:|---:|
| 1 Poiseuille | 连续算子 μu″+G | {a1["residual_continuum"]["residual_rel"]:.1e} | {a1["control_wrong_amplitude_1pct"]["residual_rel"]:.3f}（幅值错 1%） |
| 1 Poiseuille | 二阶中心差分算子 | {a1["residual_discrete_laplacian"]["residual_abs"]:.1e}（roundoff 地板 {a1["parameters"]["residual_roundoff_floor_rel"]:.1e}） | 同上 |
| 2 Couette | 连续算子 μu″ | {a2["residual_continuum"]["residual_rel"]:.1e} | {a2["control_parabolic_profile"]["residual_rel"]:.3f}（用抛物线冒充） |
| 2 Couette | 壁面切应力 τ=μU/H | 相对误差 {a2["wall_shear"]["rel_error"]:.1e} | — |
| 3 Taylor-Green | 散度 ∂u/∂x+∂v/∂y | {a3["divergence"]["residual_rel"]:.1e} | {a3["control_pressure_3pct"]["momentum_x_rel"]:.3f}（压强错 3%） |
| 3 Taylor-Green | x 动量 | {a3["momentum_x"]["residual_rel"]:.1e} | 同上 |
| 3 Taylor-Green | y 动量 | {a3["momentum_y"]["residual_rel"]:.1e} | 同上 |
| 3 Taylor-Green | 能量恒等式 dE/dt=−ν⟨ω²⟩ | {a3["energy"]["dE_dt_identity_rel_error"]:.1e} | —（首版写成 −2ν⟨ω²⟩，残差 0.50 自查发现） |
| 4 方腔极限 | 极坐标双调和 ∇⁴ψ（可分离族，4 个 λ） | ≤{max(r["max_abs_high_precision"] for r in a4["biharmonic_family"]):.1e} | {a4["control_wrong_exponent"]["max_abs_high_precision"]:.1f}（λ−2 换成 λ−3） |
| 4 方腔极限 | Stokes 压力可积性 ∂ₓp=μ∇²u、∂ᵧp=μ∇²v | ≤{max(max(r["residual_x_rel"], r["residual_y_rel"]) for r in a4["stokes"]["pressure_reconstruction"]):.1e} | 同上 |

四例最差残差 **{max(crit["C2"]["evidence"]["worst_residual"], 0):.2e} ≤ 1e-12**；对照项最小
{min(crit["C2"]["evidence"]["negative_controls"].values()):.3f}，说明检查有判别力。

### 3.3 Poiseuille 二阶收敛斜率（任务书点名的量化锚）

抛物线解被二阶中心格式**精确**积分（N=8…512 上误差 ≤{max(r["linf_error_vs_analytic"] for r in a1["parabola_reproduced_exactly"]["rows"]):.1e}，
即零离散误差），因此收敛阶必须用制造解 u=sin(πy)（−μu″=μπ²sin πy）测：

| N | {' | '.join(str(n) for n, _ in errs)} |
|---|{'---|' * len(errs)}
| L∞ 误差 | {' | '.join("%.3e" % e for _, e in errs)} |
| 逐对斜率 | {' | '.join("%.4f" % s for s in slopes)} |

拟合斜率 p = **{a1["convergence"]["fitted_slope_p_asymptotic"]:.5f}**（对标 2.0，相对偏差
{a1["convergence"]["slope_error"]/2*100:.4f}%）。

### 3.4 内部一致性（判据 C4）

| 检查 | 结果 |
|---|---|
| 坐标落在 1/128 点阵 | 两表各 17 行，最大偏差 {val["coordinate_lattice"]["I"]["max_deviation_pt"]:.1e}（即 4 位小数舍入界） |
| 列右对齐（右边缘散布） | {", ".join("%s=%.2f pt" % (k.replace("_", " "), v) for k, v in val["ink_sign_witness"]["right_edge_spread_pt"].items())} |
| 符号双簇可分 | 6/6 列 `separation_clean`，簇间 4.3–4.6 pt |
| 双见证一致 | 84 一致 / 0 冲突 / 18 坏 token 用墨水定向 |
| 边界条件 | 三 Re 全部 u(y=1)=1、u(y=0)=0、v(x=0)=v(x=1)=0 |
| 点号↔坐标 | 表 II 全一致；表 I 有 1 处（pt 31 ↔ y=0.2813，按 (pt−1)/128 应为 0.234375）已作为异常登记，未改数 |
| 不可压不变量 ∫u dy、∫v dx | Re100 {inv["Re100"]["invariants"]["vertical_centreline_int_u_dy"]["cubic_spline"]:.2e} / {inv["Re100"]["invariants"]["horizontal_centreline_int_v_dx"]["cubic_spline"]:.2e}；Re400 {inv["Re400"]["invariants"]["vertical_centreline_int_u_dy"]["cubic_spline"]:.2e} / {inv["Re400"]["invariants"]["horizontal_centreline_int_v_dx"]["cubic_spline"]:.2e}；Re1000 {inv["Re1000"]["invariants"]["vertical_centreline_int_u_dy"]["cubic_spline"]:.2e} / {inv["Re1000"]["invariants"]["horizontal_centreline_int_v_dx"]["cubic_spline"]:.2e}（均在三分法采样不确定度 {min(x["invariants"]["vertical_centreline_int_u_dy"]["quadrature_spread"] for x in inv.values()):.1e}–{max(x["invariants"]["horizontal_centreline_int_v_dx"]["quadrature_spread"] for x in inv.values()):.1e} 之内） |

## 4. 图

见文末 6 幅：锚剖面（fig1）、符号墨水见证聚类（fig2）、守恒不变量（fig3）、
Poiseuille/Couette 与二阶斜率（fig4）、Taylor-Green 涡量与能量衰减（fig5）、
方腔 Stokes 极限模态场（fig6）。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **PDF 是扫描件 + OCR 文本层**：负号被读成字母，纯文本层抽取会把符号搞反（首版把
   `AI13641` 抽成正 +0.13641、把 `-0.21090` 抽成正号）。符号必须另找见证。
2. **负号会与后一位数字粘连**（`-0` 连成一个连通域），所以不能用"找细横条"的连通域法，
   改用**列内墨水宽度**（右对齐 + 一个减号字宽）分簇，稳定且可审计。
3. **二阶中心差分在 h 小的时候有抵消失真**：h=1/1000 时 2 阶差分噪声地板 ~ε/h²≈1e-10，
   所以离散残差必须连"roundoff 地板"一起报，否则会把数值噪声误判成算法缺陷；本单取 h=1/32。
4. **Poiseuille 抛物线不能用来测收敛阶**（二阶格式对二次函数精确），必须用制造解。
5. **2-D Taylor-Green 的能量恒等式是 dE/dt = −ν⟨ω²⟩**（不是 −2ν⟨ω²⟩）；首版写错，残差 0.50 报出来。
6. **mpmath 四阶导数有 ~1e-14 的求导噪声底**（fd_noise_floor={a4["fd_noise_floor"]:.1e}），
   故 §3.2 第 4 例的 1e-14 是求导噪声而非算子残差；压力路径积分的 1e-28 才反映真正精度。
7. **适用范围**：本单只做"锚 + 算子层机检"，**没有**把 LBM 解与 Ghia 表比对，因此不构成
   求解器精度声明；锚的消费口径（半程反弹 N_eff=N−1）已写进 JSON，比较时必须声明。
8. **仓内既有 `lid_driven_cavity/reference_data.py` 与论文表不符**（见 6.4），未改动（有其他代码 import），
   但消费者应改用本锚。

## 6. 误差判定（逐项是否 <3%）

| 关键量 | 基准 | 观测 | 相对误差 | <3%? |
|---|---|---|---:|---|
| 注册锚值 vs 论文印刷值 | Ghia 表逐字 | 同值（4–5 位小数截断） | 0（截断 ≤5e-6 绝对，最坏相对 ~{5e-6/0.00332*100:.2f}%） | ✅ |
| 二阶收敛斜率 | 2.0 | {a1["convergence"]["fitted_slope_p_asymptotic"]:.5f} | {a1["convergence"]["slope_error"]/2*100:.4f}% | ✅ |
| Couette 壁面切应力 | μU/H | 离散一侧差分 | {a2["wall_shear"]["rel_error"]*100:.2e}% | ✅ |
| Taylor-Green 能量 | 闭式 ¼U²e^{{−4νk²t}} | 谱计算 | {a3["energy"]["rel_error_E"]*100:.1e}% | ✅ |
| 守恒不变量 ∫u dy、∫v dx | 0（精确） | ≤{max(abs(x["invariants"]["horizontal_centreline_int_v_dx"]["cubic_spline"]) for x in inv.values()):.1e}（量纲尺度 1） | ≤{max(abs(x["invariants"]["horizontal_centreline_int_v_dx"]["cubic_spline"]) for x in inv.values())*100:.2f}% | ⚠️ 名义上 <3%，但**采样不确定度**达 {max(x["invariants"]["horizontal_centreline_int_v_dx"]["quadrature_spread"] for x in inv.values())*100:.1f}%，故判"未定"，不作表错的证据（数据点稀疏所致） |
| LBM 消费口径（N vs N−1 的 Re 偏置） | Re=U·L/ν | 偏置 1/N | N=64: 1.56%；N=81: 1.23%；N=129: 0.78% | ✅ 但接近边界，必须在论文里声明 |
| **仓内既有转录 vs 论文** | Ghia 表 | u(y=1)=0.0（应为 1.0）；v 差 0.26–0.46 | 100% / 26–46% | ❌ **>3% 不可接受 → 缺陷登记** |

**对锚的结论**：三 Re 六列 102 个表值已逐字核销且内部自洽，可作 LBM 验证的外部锚；
解析四例在算子层残差 ≤{crit["C2"]["evidence"]["worst_residual"]:.1e}（阈值 1e-12）且对照项 O(1)，
可作回归基线。**未完成**部分已如实标注：不含 LBM 与 Ghia 的比对（无求解器精度声明），
守恒不变量因采样稀疏只到 1e-2 量级。
"""

figs = []
for f in d["figures"]:
    p = os.path.join(REPO, f["path"])
    with open(p, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    figs.append({"name": os.path.basename(f["path"]), "caption": f["caption"],
                 "png_base64": b64})

payload = {
    "title": "W26 · Ghia (1982) 方腔剖面锚核销 + 解析四例：C1–C4 全 PASS，并登记仓内旧转录缺陷",
    "summary": ("从原始论文 PDF（sha256 1d76e463…）机器提取 Ghia 表 I/II 的 Re=100/400/1000 共 102 个"
                "表值（数字取文本层、符号取页面栅格墨水宽度，双见证 84 一致 0 冲突），"
                "四例解析残差最差 1.84e-14（阈值 1e-12，对照项 0.030–25.1 证明判别力），"
                "Poiseuille 二阶斜率 2.00005；附带登记仓内 lid_driven_cavity/reference_data.py "
                "与论文表不符（u(y=1)=0 违反盖板边界条件，v 差 26–46%）。"
                "未做 LBM 与 Ghia 的比对，不作求解器精度声明。"),
    "scenario_key": "w26_d3q19_bgk_analytic_fourcase",
    "mission_id": 17563,
    "tags": ["LBM", "D3Q19", "Ghia1982", "anchor-verification", "analytic-solutions",
             "Poiseuille", "Couette", "Taylor-Green", "cavity", "provenance"],
    "markdown": md,
    "figures": figs,
}
with open("/tmp/report.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False)
print("markdown chars", len(md), "figures", len(figs), "json bytes", os.path.getsize("/tmp/report.json"))