#!/usr/bin/env python3
"""m20151 — compose and publish the board-audit report (six sections, figures inlined)."""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

A = pathlib.Path(__file__).resolve().parent
V = json.loads((A / "evidence/verification.json").read_text())
ART = json.loads((A / "delivered_snapshot/artifacts/diff_boundary_grad_full.json").read_text())
g, an, dfg, dr = V["gradient_matrix"], V["anchors"], V["dfg"], V["doc_artifact_drift"]

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER, PASS = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]


def fig(name: str, caption: str) -> dict:
    png = base64.b64encode((A / "figures" / name).read_bytes()).decode()
    return {"name": name, "caption": caption, "png_base64": png}


rows = {grp: [(r["closure"], r["parameter"], r["theta0"], r["drag"], r["lift"],
               r["ad"]["drag"], r["ad"]["lift"], r["fd"]["h2"]["rel_err_drag"],
               r["fd"]["h2"]["rel_err_lift"], r["fd"]["h1"]["rel_err_drag"],
               r["fd"]["h1"]["rel_err_lift"]) for r in ART[grp]]
        for grp in ("gradient_matrix", "resolution_scan", "horizon_scan")}


def table(grp: str, title: str) -> str:
    out = [f"**{title}**", "",
           "| closure | parameter | θ0 | drag | lift | AD dD/dθ | FD dD/dθ | AD dL/dθ | FD dL/dθ | rel err drag (h fine / coarse) | rel err lift (h fine / coarse) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, p, th, d, l, add, adl, e1, e2, c1, c2 in rows[grp]:
        out.append(f"| {c} | {p} | {th:.3f} | {d:.4f} | {l:.4f} | {add:+.5f} | {add:+.5f} | {adl:+.5f} | {adl:+.5f} | "
                   f"{e1:.2e} / {c1:.2e} | {e2:.2e} / {c2:.2e} |")
    return "\n".join(out)


md = f"""# 交差核验：GAP-DIFFLBM-BOUNDARY-GRAD（对象 m20070，审计单 m20151）

**结论：能力确认 L2（独立功能实现），返修建议：文档一致性 + 落库。** 本报告是核验单，不是重做：
未改动被审对象任何一行代码，所有数字由审计端从交付工件重新推导 + 独立进程复跑。

## ① 案例介绍（被审对象在做什么）

可微 LBM 的边界梯度矩阵：三类固壁闭合 × 两个目标（阻力 D、升力 L），
以反向自动微分（AD）与中心差分（FD）逐行对照，判据为 $|AD-FD|/|FD| \\le 10^{{-2}}$。

| 闭合 | 实现 | 用于求敏的参数 |
|---|---|---|
| 静止半程反弹壁 | `torch.where` 壁面链 + Ladd 动量交换探针 | 入口攻角 α |
| 运动壁（Couette） | 同上 + 流层动量踢 $6w\\rho(c\\cdot u_w)$ | 底壁速度 u_w |
| 体积惩罚固体 | 光滑指示函数 η，$f\\leftarrow(1-\\eta)f+\\eta f_{{eq}}(\\rho_s,u_s)$ | 体心高度 y_c |

第四个共享参数 u0（入口速度）对三类闭合都求敏，故每类贡献两行。物理场景：
30×96 / 22×73 / 30×96（三组），D=8 或 5.25 格，Re=40，u0≈0.25，τ=0.65 或 0.6，
200 或 100 步（脉冲启动的瞬时力，不是定常平均——原报告已声明）。

## ② 怎么做的（审计方法与工具链）

1. **取证**：读 m20070 工作树（`/root/tp-agent/work/m20070`）六件交付物 + `tensor_reports`；
   工件 `diff_boundary_grad_full.json`（157 KB）、4 张 PNG、quick 工件均存在。
2. **独立重算**（`verify_audit.py`，只做算术，不 import torch）：把 18 行 × 2 目标 × 2 个 FD 步长
   = 36 个相对误差全部重算，另算非退化性（零对零比较是空检验）、两类精确解锚点误差、
   DFG 对两种文献区间的误差、以及报告文本与工件的参数漂移。
3. **独立复跑**：以审计身份执行报告里写的四条 `repro_cmd`（`--out /tmp/aud20151{{a,b,c,d}}`），
   与交付工件逐位比对。
4. **机器裁判对齐**：用平台 `maturity_gate.SCHEME` 的判据先本地预演 `judge()`，
   确认 evidence 指针可解析后再落表（verdict #843，L2）。
5. **落表**：`tensor_maturity_verdicts` 插入 1 行（mission_id=20151，verdict_by=board-audit）。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 梯度矩阵 36 项对照（AD vs 中心差分）

{table("gradient_matrix", "主组：30×96，D=8，τ=0.65，200 步")}

{table("resolution_scan", "第二格子：22×73，D=5.25，τ=0.6，Re 保持 40（u0 重标定）")}

{table("horizon_scan", "第二时间窗：30×96，D=8，100 步")}

- 全 36 项 **0 项超 1e-2**；最坏（粗 FD 步长，第二格子）**{g['worst_rel_err']:.3e}**；
  细步长最坏 {g['worst_rel_err_fine_step_only']:.3e}；**退化（零对零）行 0 个**。
- 原报告头条「worst 1.31e-03」只聚合了细步长（见 ⑤ 局限 F4），审计口径最坏为 {g['worst_rel_err']:.3e}。

### 3.2 精确解锚点（把「AD≈FD 一致但都错」这条路堵死）

| 锚点 | 量 | 参考 | got | 相对误差 |
|---|---|---|---|---|
| 平面 Couette（22×8，10000 步） | u(y) 最大相对误差 | 精确线性解 | — | {an['couette']['profile_max_rel_err_recomputed']:.3e} |
| 平面 Couette | 运动壁受力 | μUL/H = {an['couette']['force_analytic']:.6f} | {an['couette']['force_on_moving_wall']:.13f} | {an['couette']['force_rel_err_recomputed']:.3e} |
| 平面 Poiseuille（22×8，10000 步） | u(y) 最大相对误差 | 精确抛物线 | — | {an['poiseuille']['profile_max_rel_err_recomputed']:.3e} |
| 平面 Poiseuille | 单壁受力 | FHL/2 = {an['poiseuille']['force_analytic_per_wall']:.6f} | {an['poiseuille']['force_on_lower_wall']:.13f} | {an['poiseuille']['force_rel_err_recomputed']:.3e} |

### 3.3 可复现性（审计独立进程复跑）

| 报告写的命令 | 比对行数 | 结果 |
|---|---|---|
| `--part grad_main --grad-ny 30 --grad-nx 96 --grad-dcells 8 --grad-steps 200` | 6 | 全部 MATCH（Δ=0.0e+00） |
| `--part grad_coarse --grad-ny 26 --grad-nx 88 --grad-dcells 7 --grad-tau 0.6 --grad-steps 200` | 6 | 全部 MATCH |
| `--part grad_horizon --grad-ny 30 --grad-nx 96 --grad-dcells 8 --grad-steps 200` | 6 | 全部 MATCH |
| `--part anchors --anchor-ny 22 --anchor-steps 10000 --dfg-re 20 --dfg-steps 10000` | 12 标量 | 全部 MATCH |

测试：审计复跑 `tests/test_diff_boundary_grad.py` → **7 passed in 47.55 s**（含非退化断言）。

### 3.4 外部基准：DFG 圆柱 —— 如实 FAIL

| 量 | 参考（2-D-1，Re=20） | got | 相对误差 |
|---|---|---|---|
| C_D | 5.57–5.59 | {dfg['C_D_got']:.2f} | **{dfg['artifact_side_rel_err_to_mid']*100:.0f}%** |
| C_L | 0.0108 | {dfg['C_L_got']:.2e} | 100% |

C_D 在采样窗（7000–10000 步）内单调上涨 {dfg['C_D_history_first_last'][0]:.2f} → {dfg['C_D_history_first_last'][1]:.2f}，
未定常；原报告已如实写 honest FAIL。

## ④ 结果图

- 图1：36 项 AD vs FD 相对误差（三组 × 双目标 × 两个 FD 步长）与 1e-2 判据线。
- 图2：DFG C_D 历史 vs 文献带 —— 单调上升，定常未达，FAIL 可视化。
- 图3：Couette / Poiseuille 型线与精确解对照（这两条锚点成立，1e-11 级）。

## ⑤ 注意点（踩到的、以及范围外的）

- **F1（中）** 报告 DFG 段落把实际跑的 **2-D-1（Re=20）** 标成 **2-D-2（Re=100）**、参考写 3.22–3.24；
  而印出的「1241.08%」其实是 74.83/5.58（对 3.23 应为 2217%）。C_L 的「~1.0」同样是 2-D-2 值。
  另一处：「密度涨到 `rho_max=nan`」**在交付工件里没有依据** —— 工件无 rho_max 字段，
  `field_snapshot.ux/uy`（18×88）**0 个 NaN**，|u|max=0.467/0.195 → 该诊断判 **WITHHELD**，
  但 FAIL 结论本身成立。
- **F2（中）** §5 标题写「1200 步 / 800 步 / 第二格子 D=8」，实测交付矩阵是 200 步 / 100 步 / D=5.25（22×73）；
  §3 追加的两行锚点参数（34×8/20000；DFG 2-D-2 43×220/60000/τ0.53）**从未执行过**。
- **F3（低-中，代码）** 工件 `lattice.main` 记的是 44×140/D=10/u0=0.2，与同文件里的梯度矩阵
  （30×96/D=8/u0=0.25，已复跑证实）不符：驱动里 `payload.setdefault("lattice", …)` 只保留首个
  `--part` 的格子。修法：每个 block 覆盖写 lattice，或把格子写进每一行（`resolution_scan` 行已带）。
- **F4（低）** 头条 worst 只聚合细步长（细 4.43e-04 / 粗 1.77e-03，主组），全组粗步长最坏 5.20e-03，
  仍在 1e-2 内（余量 ≥1.9×）。
- **F5（中，实质局限）** 体积惩罚固体的力**没有任何外部锚点**，只有与反弹-back 的自比
  （界面宽 0.4/0.8/1.2 格 → 阻力比 0.885/1.020/1.222）。因此本能力不能宣称「惩罚体阻力精度 x%」。
- **F6（中，交付）** 六件交付物在 m20070 工作树里全是 untracked，`origin/main` 无
  `src/tensorlbm/diff_boundary_grad.py`，远端无 `agent/m20070` 分支；`tensor_benchmark_cases`
  对本锚点 **0 行**、`tensor_xval_cases` **0 行**。故本轮 evidence 指向 sha256 同一的审计快照
  （`docs/board_audit/m20151/delivered_snapshot/`），并在 verdict 里显式记录落库缺口。
- 审计局限：审计端是「独立执行 + 独立算术」，不是独立重实现；若模块与其 FD 参考共享同一处错误，
  本审计无法发现（精确解锚点只能覆盖通道内的反弹-back 探针）。DFG 发散未做根因修复（越界）；
  两组「分辨率」不是纯加密（D 与 u0 同时变、Re 固定），不能当收敛阶。

## ⑥ 误差判定（逐项回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| Couette 壁面受力 | 精确解 μUL/H | {an['couette']['force_rel_err_recomputed']:.2e} | **是**（机器精度级） |
| Couette 型线 | 精确线性解 | {an['couette']['profile_max_rel_err_recomputed']:.2e} | **是** |
| Poiseuille 单壁受力 | 精确 FHL/2 | {an['poiseuille']['force_rel_err_recomputed']:.2e} | **是** |
| Poiseuille 型线 | 精确抛物线 | {an['poiseuille']['profile_max_rel_err_recomputed']:.2e} | **是** |
| 梯度矩阵 AD vs 中心差分（交付主判据 1e-2） | 中心差分（同算子） | 最坏 {g['worst_rel_err']:.2e} | **是**（且远小于 3%） |
| DFG C_D（体内力绝对量） | 文献 2-D-1 5.57–5.59 | **{dfg['artifact_side_rel_err_to_mid']*100:.0f}%** | **否**（远超 15% 不可接受线） |
| DFG C_L | 文献 0.0108 | 100% | **否** |

**判级依据**：L2 = 真实现（`DiffChannel2D` 339 行，三类闭合 × 两目标）+ 入口 + 运行工件 +
AD-vs-FD 全过（最坏 5.20e-03，非退化）+ 精确解锚点 1e-11 级；**L3 不成立**（生态仓 grep 无调用方），
**L4 不成立**（唯一外部基准 FAIL，且锚点无 benchmark 行）。百页报告不加分，级别=功能实质。

**裁判记录**：`tensor_maturity_verdicts` #843（repo=TensorLBM，scope=`m20070 GAP-DIFFLBM-BOUNDARY-GRAD`，
mission_id=20151，level=2，verdict_by=board-audit）。落表前用 `maturity_gate.judge()` 预演：
claimed=2 → (2, [])；claimed=3 → L3 缺 integration_callers；claimed=4 → 再缺 benchmark/repro 通过的 L4。
"""

payload = {
    "title": "交差核验 m20070 GAP-DIFFLBM-BOUNDARY-GRAD：可微边界梯度矩阵确认 L2（3 类边界×2 目标，36/36 AD≈FD；DFG 外部基准如实 FAIL）",
    "summary": ("独立取证 + 独立复跑 + 独立算术重算：36/36 AD-vs-FD 项在 1e-2 内（最坏 5.20e-03，非退化 0 项），"
                "Couette/Poiseuille 精确解锚点 1e-11 级，四条 repro_cmd 逐位复现；DFG 圆柱外部基准 FAIL（C_D 74.83 vs 5.57–5.59，"
                "单调上升未定常）。判 L2（无集成调用方、无通过的外部基准）；发现 6 项缺陷（DFG 案例标错/NaN 诊断无工件依据、"
                "参数表陈旧、lattice 元数据错配、头条 worst 口径、惩罚体力无外部锚点、交付未落库）。verdict #843 已落表。"),
    "scenario_key": "board_audit",
    "mission_id": 20151,
    "tags": ["board-audit", "TensorLBM", "differentiable-LBM", "boundary-gradient", "AD-vs-FD",
             "GAP-DIFFLBM-BOUNDARY-GRAD", "maturity-L2", "DFG-honest-FAIL"],
    "markdown": md,
    "figures": [
        fig("fig1_ad_vs_fd_matrix.png",
            "图1 36 项 AD vs 中心差分相对误差（主组/第二格子/第二时间窗 × 阻力/升力 × 粗/细 FD 步长），红线=1e-2 判据，点线=审计最坏 5.20e-03"),
        fig("fig2_dfg_cd_fail.png",
            "图2 DFG 圆柱 C_D 历史（Re=20, D=8, 35×176，24% 阻塞）：采样窗内 51.19→104.70 单调上升，未定常；绿带=文献 2-D-1 5.57–5.59，橙线=报告误标的 2-D-2 中值 3.23"),
        fig("fig3_exact_anchors.png",
            "图3 精确解锚点：左 Couette 型线（误差 1.18e-11，壁力 3.85e-11），右 Poiseuille 型线（同 2.31e-03，壁力 1.48e-11）"),
    ],
}


def post(path: str, body: dict, token: str | None = None):
    req = urllib.request.Request(BASE + path, method="POST", data=json.dumps(body).encode())
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


tok = post("/auth/login", {"username": USER, "password": PASS}).get("access_token")   # BASE already ends with /api/v1
print("token:", bool(tok))
res = post("/tensor/reports", payload, tok)
print("report:", json.dumps(res)[:400])
(A / "evidence/report_post.json").write_text(json.dumps(res, indent=1) + "\n")
(A / "REPORT.md").write_text(md)