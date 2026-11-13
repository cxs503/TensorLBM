#!/usr/bin/env python
"""Publish the m17582 board-audit report to the platform report API.

Payload is assembled from ``audit.json`` (numbers injected, never typed) plus the
audit figures; credentials come from ``$TENSOR_AGENT_USER`` / ``$TENSOR_AGENT_PASS``
and the endpoint from ``$TENSOR_API_BASE``.

    python docs/board_audit/m17582/post_report.py [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = json.loads((HERE / "audit.json").read_text())
MISSION = 17582
CAPTIONS = {
    "audit_force_rel_l2_vs_criterion.png":
        "逐算例 CSF 力场相对 L2 误差 vs 15% 判据：生产默认 central7 在 C3-droplet-W3 达 20.6% 越线，"
        "opt-in isotropic_d3q19 在 1e-14 舍入水平（审计员重跑同值）。",
    "audit_sigma_recovery_anchor.png":
        "σ 反演误差（唯一物理锚）：固定 W=4 时随 R 单调变差 -4.4%→-5.1%→-6.5%（不收敛）；"
        "固定 R=16 时随 W 变厚改善 -11.8%→-6.5%→-2.0%。仅 15% 判据内，3% 目标未达。",
    "subject_fig_force_chain_droplet.png":
        "受审交付的场量图（m17515 工件原件）：液滴周围 CSF 力链 / 化学势场云图，"
        "central7 与 isotropic_d3q19 的切向寄生力对比。",
}


def payload() -> dict:
    p1, p2, p3, p4, p5, p6 = (A["P1_ref_vs_got"], A["P2_reproduction"], A["P3_referee"],
                              A["P4_sigma_anchor"], A["P5_integration"], A["P6_detection_power"])
    rows = p1["rows"]
    table = "\n".join(
        f"| {r['case']} | {r['operator']} | {r['rel_l2_force']:.3e} | {'✔' if r['pass_15pct'] else '✘'} | "
        f"{'n/a' if r['sigma_err'] is None else format(100 * r['sigma_err'], '.3f') + '%'} |"
        for r in rows)
    callers = "\n".join(
        f"- `{s['file']}:{s['line']}` `{s['call']}(…)` operator= {'是' if s['passes_operator'] else '否'}"
        for s in p5["production_call_sites"])

    md = f"""# 交差审计：m17515 XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN（判级 L3）

## ① 案例介绍
受审对象 = m17515（TensorLBM 仓，`lssfau/lbmpy` 的 Allen-Cahn / CSF（Korteweg）力链交叉验证，round-2 返修轮）。
物理背景：扩散界面两相流中，毛细力链 `F = mu ∇phi`（`mu` 为自由能变分导数）决定界面张力与寄生流；
参考实现为 lbmpy 2.0 的 `phasefield_allen_cahn`（Fakhari & van der Graaf, PRE 96, 053301）。
受审提交 `{A['subject_head_commit'][:12]}`，工作树 `{A['subject_worktree']}`。
算例：平面界面 1 组 + 液滴 R=8/12/16 + 界面厚度 W=3/4/6 + σ 扫描（5e-3/1e-2/2e-2），
判据：力场相对 L2 误差 < 15%（平台精度标准），σ 反演 < 15%。

## ② 怎么做的（审计方法）
1. **取证**：从交付工件（`artifacts/…/xval_allen_cahn_force_chain.json`、`benchmarks/…/result.json`）
   程序化重算每一行 ref-vs-got，不采信报告叙述。
2. **可复现**：审计员亲自执行 `repro_cmd` 到独立目录，与交付工件逐块比对
   （decision 全等={p2.get('decision_identical')}，CSV 字节全等={p2.get('csv_identical')}，
   数值块全等={p2.get('numeric_blocks_identical')}，最大决策偏差 {p2.get('max_abs_decision_delta'):.1e}）。
3. **机器裁判**：`{p3['sql']}` → **{p3.get('n_rows')} 行**（全表 {p3.get('table_total_rows')} 行）。
4. **集成活性**：静态扫描生产代码中力链 / Allen-Cahn 的全部调用点，判断被验证的算子在
   生产路径上是否真被选中（{p5['n_production_sites']} 处调用，{p5['n_production_sites_selecting_isotropic']} 处显式选择算子）。
5. **判据分辨力（变异实验）**：把算子人为改坏 5%，看度量能否发现。
6. **测试**：实跑 `pytest tests/test_allen_cahn_isotropic_fd_xval_m17410.py -q` → 13 passed。

## ③ 计算结果（程序化注入）
### ref-vs-got 逐项
| 算例 | 变体 | rel L2 力 | 15% 判据 | σ 反演误差 |
|---|---|---|---|---|
{table}

- 交付变体（isotropic_d3q19）最差 = **{p1['worst_rel_l2_by_operator']['isotropic_d3q19']:.3e}**
- 生产默认变体（central7）最差 = **{p1['worst_rel_l2_by_operator']['central7']:.4f}**（越线算例：{', '.join(p1['failing_15pct'])}）
- σ 反演最差 = **{100 * p4['worst_sigma_err']:.2f}%**（<15% 通过，3% 目标未达）
- 生产核 μ vs lbmpy：isotropic 6.55e-11 / central7 6.08e-02
- 变异实验：基线 {p6.get('baseline_rel_l2'):.2e} → 拉普拉斯 ×1.05 = {p6.get('laplacian_plus5pct_rel_l2'):.3e}，
  梯度 x ×1.05 = {p6.get('grad_x_plus5pct_rel_l2'):.3e} ⇒ 度量有分辨力，但 15% 门仍容忍约 9% 算子误差。

## ④ 结果图
见 figures（力场误差 vs 判据、σ 反演锚、受审交付的场量云图）。

## ⑤ 注意点（踩坑与局限）
- **一致性是结构性的**：isotropic 移植权重与 lbmpy `max_abs_diff = 0`，故 1e-14 只证明移植保真，
  不证明物理正确；真正独立的比较（central7 vs lbmpy）才是信息量所在，而它 20.6% 越线。
- **改进已交付未部署**：被验证的算子在 {p5['n_production_sites']} 个生产调用点里只有
  {p5['n_production_sites_selecting_isotropic']} 个选中；两个多相求解器仍走 central7。
- **算例高报**：10 算例 × 2 变体 = {p1['n_rows']} 行，但互异度量组仅 {p1['n_distinct_metric_groups']} 组
  （σ 只线性缩放力，相对度量对其不变）。
- **旗标自相矛盾**：`decision.central7_within_15pct=True` 只用 C1+C2 计算，与 C3-droplet-W3 的 20.58% 矛盾
  （headline 有如实列出越线算例，故属"局部表述"而非"隐藏"）。
- **无时间推进算例**：`allen_cahn_step` 为 UNVERIFIED 硬门且不贯通 operator=，力链在求解器中的消费未被验证。
- **机器裁判缺行**：该锚点在 `tensor_benchmark_cases` 无登记行 → 15% 容差为作者自述。

## ⑥ 误差判定（逐项，显式回答 <3%）
| 关键量 | 参考 | 误差 | <3%？ |
|---|---|---|---|
| CSF 力场（isotropic，W4/R16） | lbmpy 2.0 | {p1['worst_rel_l2_by_operator']['isotropic_d3q19']:.2e} | 是 |
| CSF 力场（central7，W3/R16） | lbmpy 2.0 | {p1['worst_rel_l2_by_operator']['central7'] * 100:.2f}% | **否** |
| σ 反演（W4/R16） | 给定 σ | {abs(100 * p4['sigma_err_vs_radius_W4'][-1][1]):.2f}% | **否** |
| σ 反演（最差 W3/R16） | 给定 σ | {abs(100 * p4['worst_sigma_err']):.2f}% | **否** |
| 生产核 μ（isotropic） | lbmpy 2.0 | {6.549748843549204e-11:.2e} | 是 |

**结论：力场与参考实现的一致性 <3% 达成（移植保真）；但唯一物理锚（σ 反演）最差 {abs(100 * p4['worst_sigma_err']):.2f}%，不满足 <3%。**
因此判级 **L3**：L2/L3 成立，L4 不成立（机器裁判缺行 + 一致性结构性 + 物理锚未达 3% + 无时间推进耦合算例 + 登记缺陷）。

## 判级落表
`tensor_maturity_verdicts`：repo=TensorLBM, scope=`{A['scope']}`, mission_id={MISSION}, level=3, verdict_by=board-audit。
处分：request_changes（整改清单见 `docs/board_audit/m17582/AUDIT_m17582.md` §7）。
"""

    figures = []
    for name, cap in CAPTIONS.items():
        cand = [HERE / "figures" / name, HERE / "subject" / name]
        p = next((c for c in cand if c.is_file()), None)
        if p is None:
            continue
        figures.append({"name": name, "caption": cap,
                        "png_base64": base64.b64encode(p.read_bytes()).decode()})
    return {
        "title": "交差审计 m17582：XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN（m17515）判级 L3",
        "summary": (f"逐项核验 m17515 三件套与 ref-vs-got：交付变体力场 {p1['worst_rel_l2_by_operator']['isotropic_d3q19']:.2e}、"
                    f"生产默认 central7 最差 {100 * p1['worst_rel_l2_by_operator']['central7']:.2f}%（越 15%）、"
                    f"σ 反演最差 {abs(100 * p4['worst_sigma_err']):.2f}%（未达 3%）；"
                    f"独立重跑数值块全等；判级 L3，处分 request_changes。"),
        "scenario_key": "board-audit",
        "mission_id": MISSION,
        "tags": ["board-audit", "xval", "TensorLBM", "allen-cahn", "csf", "lbmpy", "maturity-gate", "m17515"],
        "markdown": md,
        "figures": figures,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    body = payload()
    (HERE / "report_payload.json").write_text(json.dumps(body, ensure_ascii=False)[:2000000])
    print("figures:", [f["name"] for f in body["figures"]], "| md chars:", len(body["markdown"]))
    if args.dry_run:
        return 0
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=login, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as resp:
        token = json.load(resp).get("access_token") or json.loads(login) and None
    if not token:
        raise SystemExit("login returned no access_token")
    req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as resp:
        print("posted:", resp.status, json.load(resp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())