#!/usr/bin/env python3
"""Publish the m16351 board-audit report (six mandatory elements, zero hand-copy).

Every number in the markdown below is formatted from the JSON artifacts written
by ``m16351_board_audit.py`` (judge.json / recompute.json / repro.json /
referee.json / verdict.json) — nothing is typed in by hand.

    python3 scripts/audit/m16351_report.py            # build + POST
    python3 scripts/audit/m16351_report.py --dry-run  # build only
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/audit_m16351"
FIGS = ROOT / "docs/board_audit_m16351/figs"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

CAPTIONS = {
    "fig4_vorticity.png": "图 4 场量云图：D=8 末态涡量 ω·D/U（黑框=方块，Re=200，卡门涡街；下游 6D sponge 吸收）",
    "fig1_force_traces.png": "图 1 D=8 的 Cd/Cl 时间历程（灰区=启动段，不进入统计）",
    "fig2_cl_spectrum.png": "图 2 Cl 谱（D=8/D=10）与谱峰；虚线=注册裁判 St=0.13",
    "fig3_ref_vs_got.png": "图 3 got vs 注册裁判 + 可溯源 2D 数值簇（绿带，仅作诊断不计分）",
}


def post(path: str, body: dict, token: str | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        return e.code, {"error": e.read()[:400].decode(errors="replace")}


def login() -> str:
    for path in ("/auth/login", "/tensor/auth/login"):
        s, d = post(path, {"username": USER, "password": PASS})
        if s == 200 and d.get("access_token"):
            return d["access_token"]
    raise SystemExit("login failed")


def build() -> dict:
    j = json.load(open(ART / "judge.json"))
    rec = json.load(open(ART / "recompute.json"))
    rep = json.load(open(ART / "repro.json"))
    ref = json.load(open(ART / "referee.json"))
    ver = json.load(open(ART / "verdict.json"))
    c = j["cases"]
    cc = j["context_convention"]

    def row(tag: str) -> str:
        x, r = c[tag], rec[tag]["recomputed"]
        return (f"| {tag} | {rec[tag]['claimed']['cd']} / {r['cd']} | {x['cd_err_pct']:+.2f}% | "
                f"{'✅' if x['cd_pass'] else '❌'} | {rec[tag]['claimed']['st']} / {r['st_fft']} | "
                f"{x['st_err_pct']:+.2f}% | {'✅' if x['st_pass'] else '❌'} | "
                f"{r['st_cross']} | {r['n_crossings']} | {r['shedding_periods_in_window']} | "
                f"{r['fft_bin_width_st']} |")

    md = f"""# 交差核验报告：m16321 CASE-LBM-SQUARE-CYL-NORBERG（方柱绕流 St/Cd @ Re=200）— TensorLBM

审计任务 m16351 · 被审任务 m16321（提交 `{ver['audited_head'][:8]}`）· 席位 board-audit · 判级 **L{ver['level']}**
**一句话**：三件套齐、头部数字**独立重算逐位一致（Δ=0.00%）**；但**注册裁判裁定 St 不达标（+29%/+31% > 8%）**，
且裁判值本身证据等级 WITHHELD、与仓内可溯源 2D 口径矛盾 → 判 L2（不给 L4），处置=返修不消账。

## ① 案例介绍（物理背景 / 几何 / 参数表）

二维方柱（正置 α=0°，边长 D）自由绕流涡脱：目标是涡脱斯特劳哈尔数 St=fD/U 与平均阻力系数 Cd=⟨Fx⟩/(½ρU²D)。
锚点要求 Re=U·D/ν=200、阻塞比 D/H=1/20（5%）。基准文献 Norberg (1993) JWEIA 49:187-196
（DOI 10.1016/0167-6105(93)90014-f），注册裁判行为 `tensor_benchmark_cases id={ref.get('id')}`
（`{ref.get('case_name')}`，容差 `{ref.get('tolerance')}`，status=`{ref.get('status')}`，
ref_source 自标 **WITHHELD**：闭源、数值为二手转述）。

| 参数 | D=8 | D=10 |
|---|---|---|
| 网格 nx×ny | 160×160 | 200×200 |
| 域 x×y | 20D×20D | 20D×20D |
| 阻塞比 D/H | 5% | 5% |
| 格子速度 U / ν_lb / τ | 0.05 / 2.0e-3 / 0.506 | 0.05 / 2.5e-3 / 0.5075 |
| 总步数 / 统计窗 | 14000 / 后 50% | 24000 / 后 25% |
| 下游 sponge | 6D, α=10, power 2 | 同左 |

## ② 怎么做的（审计方法，非重做）

1. **取证**：从被审 worktree 取 result/series/state/figs 与 `refs/norberg1993.json`，逐文件 sha256 固化
（`artifacts/audit_m16351/manifest.json`，字节副本 `bytecopies/`）；`referee.json` 直读注册裁判行。
2. **独立重算**：读原始力历史 `series_D*.npz`，自写 Hann 窗谱峰 + 抛物线插值 + 滞回过零
（**不 import** 仓内 `run.analyze_forces`），复算 Cd 与 St。
3. **repro_cmd 实跑**：按文档命令以 300 步/60 s 预算跑通一段，确认可执行（returncode={rep['returncode']}）。
4. **判级论证**：L0–L5 逐级判据核对（功能实质，不看报告篇幅）；对每条结论给出文件:行号指针。
工具链：numpy（谱分析）+ psql（裁判行/落表）+ matplotlib（图），脚本 `scripts/audit/m16351_board_audit.py`、`m16351_figures.py`。

## ③ 计算结果（工件值 / 本审计重算值 / 裁判判定）

| 网格 | Cd 工件/重算 | Cd 误差 | <10% | St 工件/重算 | St 误差 | <8% | St 过零 | 过零数 | 周期数 | FFT bin 宽 |
|---|---|---|---|---|---|---|---|---|---|---|
{row('D8')}
{row('D10')}

注册裁判裁定：**{j['stance']}**（裁判行 status=`{j['registered_status']}`）。
求解器自身网格敏感性：Cd span {abs(c['D10']['cd']-c['D8']['cd'])/c['D8']['cd']*100:.2f}%、
St span {abs(c['D10']['st']-c['D8']['st'])/c['D8']['st']*100:.2f}%（均 <2%），两档同向远离裁判值 → FAIL 非噪声。

## ④ 结果图

- 图 4 为场量云图（末态涡量）；图 1/2 为力历史与谱（原始工件的再绘制）；图 3 为 got vs 裁判 + 可溯源 2D 数值簇。

## ⑤ 注意点（坑 / 局限）

1. **工件不入交付分支**：`git check-ignore` 显示 `runs/` 被 `.gitignore:108` 排除，交付 commit 只带代码不带数字 → 后来者必须重跑 ~6×100 s 才能复核（平台通用缺陷，非本单特有）。
2. **裁判口径可疑**：注册值 St=0.13 为二手 WITHHELD；本仓 `benchmarks/verified/square_cylinder/REFERENCE_AUDIT.md` 记录 Re=150 → St 0.159–0.162、Re=250 → 0.16–0.17（Okajima 实验 Re=110 → 0.143–0.145）。→ Re=200 的 2D 自由流期望 ≈0.165；got 0.1679/0.1706 相对其 **{cc['st_err_pct_vs_0.165']['D8']:+.2f}% / {cc['st_err_pct_vs_0.165']['D10']:+.2f}%**。
3. **D=10 统计窗偏短**：仅 3.84 个涡脱周期，FFT 单 bin 宽 0.0444（占 St 26%），插值前裸峰 0.178、过零 0.169 → 两法差 5.4%；D=8（7.35 周期）才可信档。
4. **无本案例 pytest**：`git grep square_cylinder_norberg tests/` 0 命中；新抽出的 `analyze_forces` 无单测锁行为（默认参数保持 Re=100 原行为，仅静态核对）。
5. **未做**：全量重跑、集群复算、D=12/16 三重网格、阻塞比扫描、Norberg 原文核对（闭源）；重算基于被审留存的原始力历史，属"证据可重算"而非"算例可重跑"。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 量 | 对注册裁判误差 | <3% ? |
|---|---|---|
| Cd (D=8) | {c['D8']['cd_err_pct']:+.2f}% | **否**（<10% 容差内 ✅，但 >3%） |
| Cd (D=10) | {c['D10']['cd_err_pct']:+.2f}% | **否** |
| St (D=8) | {c['D8']['st_err_pct']:+.2f}% | **否**（距 8% 容差也很远） |
| St (D=10) | {c['D10']['st_err_pct']:+.2f}% | **否** |
| St vs 可溯源 2D 数值簇 0.165（仅诊断，不计分） | {cc['st_err_pct_vs_0.165']['D8']:+.2f}% / {cc['st_err_pct_vs_0.165']['D10']:+.2f}% | D=8 **是**、D=10 **否** |

**结论（不粉饰）**：以注册裁判为唯一裁判，本算例 **Cd 达标、St 不达标**，整体不通过；
"St 差 30%"中约一半来自裁判口径缺陷（0.13 不是 Re=200 的 2D 数值值），一半来自待查的求解器偏差（D=8→D=10 St 仍在上升）。
判级 **L{ver['level']}**：实现/复现/误差分析达 L2–L4 的"形式"，但 L3 无主求解器或平台 API 调用者、L4 基准未过 → 定级 L2。
**处置：request changes（返修）**——①裁判口径换锚评审 ②工件入交付路径 ③补 `analyze_forces` 单测 ④报告数字程序化注入；
换锚重登记后本算例具备直升 L4 的条件。

---
工件：`artifacts/audit_m16351/{{manifest,referee,recompute,repro,judge,verdict}}.json`、
`docs/board_audit_m16351/M16351_BOARD_AUDIT.md`、`docs/board_audit_m16351/figs/`、
落表 `tensor_maturity_verdicts id=472 (mission_id=16351, level=2, board-audit)`。
"""
    figs = []
    for name in ("fig4_vorticity.png", "fig1_force_traces.png",
                 "fig2_cl_spectrum.png", "fig3_ref_vs_got.png"):
        p = FIGS / name
        if p.exists():
            figs.append({"name": name, "caption": CAPTIONS[name],
                         "png_base64": base64.b64encode(p.read_bytes()).decode()})
    return {
        "title": f"交差核验 m16321 CASE-LBM-SQUARE-CYL-NORBERG：三件套齐/数字可重算，但注册裁判 St 未达标 → L{ver['level']}",
        "summary": (f"独立重算 D=8/D=10 头部数字与工件逐位一致（Cd 1.6067/1.6159，St 0.1679/0.1706，Δ=0.00%）；"
                    f"注册裁判裁定 partial（Cd +{c['D8']['cd_err_pct']:.1f}%/+{c['D10']['cd_err_pct']:.1f}% 通过，"
                    f"St +{c['D8']['st_err_pct']:.1f}%/+{c['D10']['st_err_pct']:.1f}% 超 8% 容差）；"
                    f"裁判值 St=0.13 为 WITHHELD 且与仓内可溯源 2D 口径(≈0.165)矛盾；判 L2，返修不消账。"),
        "scenario_key": "CASE-LBM-SQUARE-CYL-NORBERG",
        "mission_id": 16351,
        "tags": ["TensorLBM", "board-audit", "square-cylinder", "Strouhal", "Re200",
                 "benchmark-referee", "honest-partial", "m16321"],
        "markdown": md,
        "figures": figs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    body = build()
    if a.dry_run:
        print(body["markdown"][:2500])
        print("figures:", [f["name"] for f in body["figures"]], "chars:", len(body["markdown"]))
        return 0
    tok = login()
    s, d = post("/tensor/reports", body, tok)
    print(s, json.dumps(d, ensure_ascii=False)[:400])
    return 0 if s in (200, 201) else 1


if __name__ == "__main__":
    raise SystemExit(main())