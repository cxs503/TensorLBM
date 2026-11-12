"""Publish the m15712 audit report (tensor_reports) with the audit figure embedded."""
import base64
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
rec = json.load(open(os.path.join(HERE, "audit_recompute.json")))
m = rec["main_run"]
bm = m["benchmark"]
rc = m["recomputed"]
pilot = rec.get("independent_smoke_rerun")
png = base64.b64encode(open(os.path.join(HERE, "fig_audit_m15432.png"), "rb").read()).decode()

def f(x, n=4):
    return f"{x:.{n}f}"

md = f"""# 交差审计：m15432 CASE-CYLINDER-DRAG-RE100（TensorLBM）→ 判级 **L3**，基准验证不成立

**审计任务** m15712 · **被审任务** m15432（succeeded，PR #1829，`tensor_reports.id=14637`）· **commit** `68813473`
**裁定**：**L3**（真代码 + 主求解器内核接入 + 可复现工件）；**L4 不成立**（主判据 Cd 超差）。
**gate**：建议 reopened（返修 Cd 项 + 补网格收敛 + 登记基准裁判）。

## ① 案例介绍（物理背景/几何/参数表）

圆柱绕流 Re=100 是钝体阻力的标准标定题。被审单交付**无约束 2-D 圆柱**的时均 Cd、时均 Cl 与 St，判据 <5%（`tensor_case_pool` id=11）。

| 参数 | 值 |
|---|---|
| 网格 / 圆柱直径 | 250×100 / D=10 格点（上游 5D+下游 20D，横向 10D） |
| 阻塞比 B=D/H | **10%**（按"无约束"基准比较） |
| Re / Ma / τ | 100 / 0.173 / 0.53（D2Q9 BGK，float32） |
| 边界 / 力 | `far_field_bc_2d` 自由流侧面 + 零梯度出口；Ladd 动量交换力 |
| 采样 | 预热 8000 步 + 18000 步采样，双半窗漂移判 stationary |
| 基准（ref） | Cd=1.3796, Cl=0.0005, St=0.164（声明为 Schäfer/Rannacher/Turek 1996；**工作树内无页码/原表摘录，出处未闭环**） |

## ② 方法（审计流程与工具链）

1. **取证**：只读 `/root/tp-agent/work/m15432`，核三件套（工件/ref-vs-got/repro_cmd）。
2. **独立复算**（`audit_m15432_recompute.py`）：**不读报告数字**，从交付工件 18000 点逐点序列重算全部 headline 指标 → `audit_recompute.json`；与自报值偏差 **0.00e+00**（反造假通过）。
3. **复现**：以被审分支自身 src 跑 `--pilot` 冒烟（22 s，无 NaN）。
4. **对抗核查**：读 318 行实现与力核（Ladd `F=2Σc_i f_i`）、边界核、stationarity 口径；查 ref 出处、机器裁判行、repro 的 import 解析路径。
5. **判级**：L0–L5 功能实质定级；报告体量不改变级别。

## ③ 计算结果（独立复算，程序化注入）

| 量 | 审计重算 | 自报 | 偏差 |
|---|---|---|---|
| Cd 均值 | {f(rc['cd_mean'],6)} | {f(m['reported']['cd_mean'],6)} | 0 |
| Cl 均值 | {rc['cl_mean']:.3e} | {m['reported']['cl_mean']:.3e} | 0 |
| Cl 幅值 | {f(rc['cl_amp'],6)} | {f(m['reported']['cl_amp'],6)} | 0 |
| 涡脱周期 / St | {f(rc['period_steps'],3)} 步 / {f(rc['strouhal'],6)} | {f(m['reported']['period_steps'],3)} / {f(m['reported']['strouhal'],6)} | 0 |
| 双半窗漂移 | {f(rc['half_window_drift_pct'],6)}% | {f(m['reported']['cd_half_drift_pct'],6)}% | 0 |

**基准逐项**（容差 {bm['cd']['tol_pct']}%）：Cd {f(bm['cd']['got'])} vs {bm['cd']['ref']} → **{bm['cd']['rel_err_pct']:+.2f}% ✗**；St {f(bm['st']['got'])} vs {bm['st']['ref']} → **{bm['st']['rel_err_pct']:+.2f}% ✓**；Cl 均值 |Δ|={abs(bm['cl']['got']-bm['cl']['ref']):.1e} ≪ 幅值 {f(rc['cl_amp'],3)}（对称性 ✓，相对误差对零基准无意义）。

**审计独立冒烟（非判据配置，110×60，B=16.7%，4000 步，22 s）**：Cd={f(pilot['recomputed']['cd_mean'])}（+31.4%），St={f(pilot['recomputed']['strouhal'])}（+0.9%）→ 与主算例同向（Cd 高估、St 稳健），排除"工件来自另一套代码"。

**显式回答「关键量是否 <5%」：Cd 否（+25.41%），St 是（+4.39%）。主判据不达标 → L4 不成立。**

## ④ 结果图

四联图全部由交付原始数据重绘：A 基准 ref-vs-got（±5% 带）；B 采样窗 9×2000 步 Cd 均值（平坦但水平 1.73，不在带内）；C 同仓三次运行 Cd vs 阻塞比（1.813 / 1.730 / m15407 D=16 的 1.5445，文献带 1.33–1.38）；D 由 `samples.wz` 重绘涡量场（反造假核对）。

## ⑤ 注意点（坑与局限）

- **F2 构型不匹配**：横向仅 10D（B=10%）且侧面强加自由流 Dirichlet，却与无约束基准对比；B 与分辨率同向变化，三点数据**无法分离**阻塞与分辨率贡献。
- **F3 无收敛链**：D=20 收敛点从未执行（集群 `cj1159` 审计时仍 queued 55.6 min，任务已 succeeded 关闭）；报告 §⑤「D=10 已收敛」混淆时间定常与网格收敛。
- **F4 ref 未闭环**：无页码/原表/PDF/sha256；同族 DFG 2D-2 是 24.4% 阻塞通道（Cd≈3.22），数值归属需原表定案。
- **F5 未判量打勾**：Cl 幅值 0.476 无 ref 对照，报告却标"量级正确 ✓"（Re=100 二维典型 ≈0.32–0.33，潜在 +45%）。
- **F6 复现非封闭**：repro_cmd 缺 `PYTHONPATH=src`，`import tensorlbm` 解析到另一工作树 m13963；两树 diff 仅入参校验，τ=0.53 数值等价 → 结论不受影响。
- **F7 系统性**：`tensor_benchmark_cases` **无该锚点行**（0/21），任务书要求的机器裁判不存在；上一单 CASE-KARMAN-RE100 同样 0/19 → 算例池与裁判表脱节。
- **F8 口径**：双半窗漂移掩盖 +0.40% 残余蠕变（首/末 1000 点 1.7258→1.7327），对 25% 结论无实质影响。
- **已核无缺陷**：力核实现与调用时序正确；三图非空白；字段/图/JSON 互洽；无造假迹象。

## ⑥ 误差判定

| 关键量 | got | ref | 误差 | <3%? | <5%? |
|---|---|---|---|---|---|
| **Cd（主判据）** | 1.7301 | 1.3796 | **+25.41%** | 否 | **否** |
| St | 0.1712 | 0.164 | +4.39% | 否 | 是 |
| Cl 均值 | −8.75e−05 | 0.0005 | ⎮Δ⎮=5.9e−04（对称性） | 是（绝对量） | 是 |

**诚实结论**：honest-FAIL 属实且未粉饰（应肯定）；但**基准验证等级不成立**，判 **L3**——真实现在、真接入、可复现，误差分析与收敛链缺失，主判据超差 5 倍。

## 判级与证据

L2（真代码输入→计算→输出，18000 点序列+场+图，审计可重算）✔；L3（仅用库内核 `collide_bgk_matmul`/`stream`/`bounce_back_cells`/`compute_obstacle_forces`/`far_field_bc_2d`，case_pool id=11 登记，报告 14637）✔；**L4 ✗**（Cd +25.41% ≫ 5%，无网格收敛，ref 未闭环）。
仲裁证据：`docs/board-audit-m15432/AUDIT_m15432_CASE-CYLINDER-DRAG-RE100.md`、`audit_recompute.json`、`fig_audit_m15432.png`、`artifacts/audit_pilot_rerun.json`；verdict 已落 `tensor_maturity_verdicts`（id=363, level=3, board-audit）。
"""

body = {
    "title": "交差审计 m15432 CASE-CYLINDER-DRAG-RE100（TensorLBM）：判级 L3，Cd +25.41% 基准不成立（independent recompute）",
    "summary": "独立复算交付工件全部 headline 指标（偏差 0.00e+00，反造假通过）；Cd 1.7301 vs 1.3796 = +25.41% ≫ 5% → L4 不成立，判 L3；另发现 10% 阻塞比与非约束基准构型不匹配、D=20 收敛链从未执行、ref 出处未闭环、机器裁判行缺失。",
    "scenario_key": "CASE-CYLINDER-DRAG-RE100",
    "mission_id": 15712,
    "tags": ["board-audit", "TensorLBM", "cylinder-drag", "Re100", "honest-FAIL", "L3"],
    "markdown": md,
    "figures": [{"name": "fig_audit_m15432.png",
                 "caption": "A ref-vs-got (Cd ✗ +25.41%, St ✓ +4.39%); B windowed Cd means (flat but 1.73); C Cd vs blockage across three TensorLBM runs; D vorticity re-rendered from delivered samples.wz",
                 "png_base64": png}],
}

base = os.environ["TENSOR_API_BASE"].rstrip("/")
login = urllib.request.Request(base + "/auth/login",
                               data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                                                "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
tok = json.load(urllib.request.urlopen(login, timeout=30))
tok = tok.get("access_token") or tok.get("token")
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok}, method="POST")
r = urllib.request.urlopen(req, timeout=60)
print(r.status, r.read()[:400].decode())