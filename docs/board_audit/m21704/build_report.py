"""Build /tmp/report.json for mission 21704 (board audit of m21640), numbers injected from artifacts."""
import base64
import json
import os

ART = "/srv/tensor/agent/work/m21704/TensorLBM/artifacts/audit_m21704"
art = json.load(open(os.path.join(ART, "m21640_results.json")))
rer = json.load(open(os.path.join(ART, "results_rerun_all6.json")))
m = {r["case"]: r for r in art["records"]}
mm = {r["case"]: r for r in rer["records"]}


def pct(rec, k):
    v = rec["metrics"].get(k)
    return "nan" if v != v else f"{100*v:.2f}%"


r167, r233, h48 = mm["layered_r167"], mm["layered_r233"], mm["layered_r167_h48"]
cs, cm = mm["control_single"], mm["control_mixture"]
st = mm["layered_static"]
a167 = m["layered_r167"]

rows = []
for tag, r in (("control_single (G=0, 1 comp)", cs), ("control_mixture (G=0, 2 comp)", cm),
               ("layered_r167 H=32 (registered KPI)", r167), ("layered_r233 H=32", r233),
               ("layered_r167_h48 H=48", h48), ("layered_static (g=0)", st)):
    rows.append(f"| {tag} | {pct(r,'u_peak_err')} | {pct(r,'eps_inf')} | {pct(r,'eps_l2_rel')} | "
                f"{pct(r,'Q_err')} | {r['metrics'].get('y_iface'):.2f} | {r['gates']} |")
table = "\n".join(rows)

figs = []
for name, cap in (("fig1_layered_profile_vs_exact.png",
                   "Fig.1 layered Poiseuille: MCMP LBM 速度剖面 vs 精确分段二次解（nominal H/2 与实测界面 19.21），右图 φ_A(y) 分层剖面"),
                  ("fig2_kpi_errors.png",
                   "Fig.2 各算例 KPI 误差 vs 15% owner 门 / 3% 自定目标（前三例为 m21640 工件，后三例为本审计补跑）")):
    with open(os.path.join(ART, name), "rb") as fh:
        figs.append({"name": name, "caption": cap, "png_base64": base64.b64encode(fh.read()).decode()})

md = f"""# 交差审计：XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM（m21640）

**判定：L4（基准验证）/ approve with notes。** 基准主判据成立且逐位可复现；但入库台账 `status='passed'`
漏记了本单自己预注册的 `interface_position<=5%H` 子门 **FAIL（10.02%H）**，且「静态界面伪流」KPI 在本配置下
**结构性恒 0、不可失败**。

## 1. 案例介绍（物理背景 / 几何 / 参数）

两组分 Shan-Chen（MCMP）D2Q9 在平板 Poiseuille 槽道中的**分层共流**：下半层黏度 ν=0.100、
上半层 ν=0.1667（τ=0.8 / 1.0），`ψ=ρ₀(1-e^(-ρ/ρ₀))`，`G=2.5>0` 排斥使两组分分层；
EDM（Kupershtokh 2004）力注入，平衡速度取 Martys-Chen 黏度加权公共速度；壁面半程反弹 +
`wall_mode="mirror"`（中性润湿）；`G` 在 2000 步内斜坡加载（否则 ρ≈2 的骤分层瞬态 200 步内发散）。
几何：H=32（ny=34，含两行固体），nx=1 周期，g_x=5e-5，12000 步。注册例另含两个 G=0 控制例与一个
g=0 静态界面例。

## 2. 怎么做的（方法 / 流程 / 关键设置）

1. **取证**：读 m21640 工件（`results.json` 28 303 B，sha256 与 `sha256.txt` 一致）、入口、tests、STATUS 文档；
   对照 `tensor_benchmark_cases` id=149 的 ref/metric/tolerance 逐项核 got。
2. **独立复核参考解**（不碰求解器）：检查 `solve_layered` 的边界、逐层 `νu''=-g`、界面 u 与 νu′ 连续、峰值下界。
3. **逐位复现**：在合并后 main（`15ef2a4c0`）上按文档 repro_cmd 重跑 4 个交付例 + 补跑 2 个「未跑」例（r233、h48），
   与工件逐字段比对。
4. **反证**：针对「静态伪流 0.0%」构造对称性分析 + 数值探针（nx=1/8、噪声 1e-3…0.2）。
5. **落表**：`tensor_maturity_verdicts` (id=889, level=4, evidence 12 键全为真实指针)。

## 3. 计算结果（数字表，程序化注入）

| case | u_peak_err | eps_inf | eps_l2_rel | Q_err | y_iface | gates |
|---|---|---|---|---|---|---|
{table}

逐位复现对照（工件 vs 重跑）：`layered_r167` 的 u_peak_err={a167['metrics']['u_peak_err']!r}、
eps_inf={a167['metrics']['eps_inf']!r}、eps_l2_rel={a167['metrics']['eps_l2_rel']!r}、
Q_err={a167['metrics']['Q_err']!r}、y_iface={a167['metrics']['y_iface']!r} **完全相同**，且全场 `ux_total`
数组 `array_equal=True`。

## 4. 结果图

见图 1（剖面 vs 精确解 + 分层剖面）与图 2（KPI vs 门限）。

## 5. 注意点（坑 / 局限）

- **F1（major）台账与自身判据冲突**：`summary.all_gates_pass=false`、`interface_position<=5%H=false`、
  入口退出码 1，但 DB 行记为 `passed`，`tolerance` 文本未记该 FAIL（作者在 STATUS/mission result 里**如实披露**了）。
- **F2（major）伪流 KPI 空转**：nx=1 且初值 x 均匀 ⇒ SC 力 x 分量按 ±x 邻居成对抵消 ⇒ `u_x≡0`（逐位 0.0）。
  探针：nx=8 均匀 0.0；噪声 1e-3 → 9.1e-14；0.05 → 6.2e-11；0.2 → 5.6e-10。该门永远不会失败。
- **F3**：文档/台账/commit message 的 control 0.25% 不可复现，工件与重跑均为 0.3675%（单）/0.3673%（混合）。
- **F4**：工件 `environment.git_commit=1e5e0891b` 是父提交（运行时工作树脏），argv 只记最后一次 `--case`；
  逐位复现已补上代码身份这一环。
- **F5**：`results.json` 有 3 个裸 `NaN` → 非 RFC-8259 JSON，`jsonb`/严格解析器拒收。
- **F8**：交付时 2/6 例未跑（如实声明）；本审计补跑后：H=48 全门通过（界面 0.52%H、eps_l2 3.82%），
  r233 eps_inf 13.80%（距 15% 门仅 1.2 pp）。**H=32 的 8.8%/10%H 主要是 1 格宽扩散界面在粗网格上的定位误差**。
- 未做：外部锚（无第二套代码/实验）、正式网格收敛（仅两点）、3D/曲界面/大密度比、GPU 路径。

## 6. 误差判定（逐项 vs 基准，显式回答「是否 <3%」）

| 关键量（layered_r167, H=32, 注册 KPI 例） | got | 精确分层解 | 误差 | <3%? |
|---|---|---|---|---|
| 峰值速度 u_peak | {r167['metrics']['u_peak_num']:.6f} | {r167['metrics']['u_peak_ref']:.6f} | {pct(r167,'u_peak_err')} | **否** |
| 场误差 ε∞ | — | — | {pct(r167,'eps_inf')} | **否** |
| 场误差 ε_L2,rel | — | — | {pct(r167,'eps_l2_rel')} | **否** |
| 层流量 Q | — | — | {pct(r167,'Q_err')} | **否** |
| 界面位置 y_iface | {r167['metrics']['y_iface']:.2f} | 16.00（nominal H/2） | 10.02%H（预注册子门 5%H **FAIL**） | **否** |
| 层内黏度拟合 ν_lower/upper | 0.1088 / 0.1757 | 0.1000 / 0.1667 | +8.8% / +5.4% | **否** |
| 控制例 ε∞（单/混合，G=0） | {pct(cs,'eps_inf')} / {pct(cm,'eps_inf')} | 精确单流体 Poiseuille | 同左 | **是**（0.37% < 3%） |
| 网格加密 H=48：ε_L2 / Q / u_peak | — | — | {pct(h48,'eps_l2_rel')} / {pct(h48,'Q_err')} / {pct(h48,'u_peak_err')} | **否 / 否 / 是** |

**结论（honest）**：注册配置 H=32 下 4 项 15% owner KPI 全部通过（7.44/8.84/8.23/8.16%），但**均未达 3% 自定目标**；
预注册的界面位置子门 FAIL（10.02%H vs 5%H）。H=48 加密后 eps_L2 降到 3.82%、界面 0.52%H、u_peak 0.07%，
说明误差来源是界面分辨不足而非模型错误。控制例 0.37% 达 3% 目标。伪流 0.0% **不计为达标**（结构性空转）。
"""

payload = {
    "title": "交差审计 m21640：XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM 判 L4（基准主判据成立、界面 5%H 子门 FAIL、伪流 KPI 空转）",
    "summary": "m21640 两组分 SC+EDM 分层 Poiseuille：4 项 15% KPI 通过（7.44/8.84/8.23/8.16%）且逐位可复现；但台账 status=passed 漏记自身预注册 interface_position<=5%H 的 FAIL(10.02%H)，静态伪流 KPI 结构性恒 0；补跑 H=48 全门通过、r233 eps_inf 13.8%。判 L4 / approve with notes。",
    "scenario_key": "board_audit",
    "mission_id": 21704,
    "tags": ["board-audit", "XVAL", "TensorLBM", "two-component-SC", "layered-Poiseuille", "m21640", "L4"],
    "markdown": md,
    "figures": figs,
}
with open("/tmp/report.json", "w") as fh:
    json.dump(payload, fh, ensure_ascii=False)
print("written", os.path.getsize("/tmp/report.json"), "bytes, figures:", len(figs))