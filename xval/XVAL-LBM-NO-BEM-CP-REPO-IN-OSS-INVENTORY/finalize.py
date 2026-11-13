#!/usr/bin/env python3
"""Wait for the XVAL runs, then aggregate + render doc + publish the platform report.

Runs detached (setsid) so it survives the driving shell. Deadline-bounded: it
finalizes with whatever run JSONs exist at the deadline and records the gap
honestly in artifacts/finalize_status.json.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUNS = HERE / "runs"
ART = HERE / "artifacts"
DEADLINE_S = float(os.environ.get("FINALIZE_DEADLINE_S", "2700"))
MISSION = 18106


def wait_for_runs() -> dict:
    want = ["n64_ra1e5.json", "n64_ra1e5_tau0p8.json", "n96_ra1e5.json", "n128_ra1e5.json"]
    t0 = time.time()
    while time.time() - t0 < DEADLINE_S:
        have = [w for w in want if (RUNS / w).exists()]
        proc = subprocess.run(["bash", "-lc", "pgrep -f 'run_xval.py --nx' | wc -l"],
                              capture_output=True, text=True)
        alive = int(proc.stdout.strip() or 0)
        if len(have) == len(want) or alive == 0:
            break
        time.sleep(20)
    have = [w for w in want if (RUNS / w).exists()]
    return {"present": have, "missing": [w for w in want if w not in have],
            "waited_s": round(time.time() - t0, 1)}


def png_b64(path: Path, max_px: int = 680) -> str:
    from PIL import Image
    im = Image.open(path)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if max(im.size) > max_px:
        r = max_px / max(im.size)
        im = im.resize((int(im.width * r), int(im.height * r)))
    q = im.quantize(colors=64) if im.mode == "RGB" else im
    buf = io.BytesIO()
    q.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    status = wait_for_runs()

    for script in ("analyze.py", "make_doc.py"):
        r = subprocess.run([sys.executable, str(HERE / script)], capture_output=True, text=True)
        status[script] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]}

    S = json.loads((ART / "summary.json").read_text())
    doc = (REPO / "docs" / "REPORT_XVAL_LBM_NO_BEM_CP_REPO_IN_OSS_INVENTORY.md").read_text()

    figs = []
    for name, caption in (
        ("fig1_nu_history.png", "Nu 随步数逼近 de Vahl Davis 1983 参考线（多网格 + τ 参数点）"),
        ("fig2_ref_vs_got.png", "Nu ref vs got（τ=0.6 全网格），柱顶标注相对误差"),
        ("fig3_gate_matrix.png", "预注册闸门矩阵（由原始 run JSON 程序化复算）"),
    ):
        p = ART / name
        if p.exists():
            figs.append({"name": name, "caption": caption, "png_base64": png_b64(p)})

    rows = sorted(S["runs"], key=lambda r: (r["tau"], r["nx"]))
    tbl = "\n".join(
        f"| {r['nx']} | {r['tau']:g} | {r['steps_done']} | {r['stopped_early']} | {r['nu']:.4f} | {r['err_nu_pct']:+.2f}% | {r['u_max_nd']:.2f} ({r['err_u_pct']:+.2f}%) | {r['v_max_nd']:.2f} ({r['err_v_pct']:+.2f}%) | {r['last_resid_u']:.1e} |"
        for r in rows)
    gates = "\n".join(f"- **{k}**: {'PASS' if v.get('pass') else 'FAIL'} — {json.dumps(v.get('detail', v.get('span_pct')), ensure_ascii=False)[:160]}"
                      for k, v in S["gates"].items())

    md = f"""## ① 案例介绍
差热方腔自然对流（DHC）：左壁 T=1、右壁 T=0、上下绝热、四壁无滑移；空气 Pr=0.71，Ra=1e5。
几何=单位方腔（库 half-way 壁约定 H=nx）。基准出处：**de Vahl Davis (1983) Int. J. Numer. Methods Fluids 3(3):249-264, Table 1 p.257**：
Nu_mean=4.519、u_max=34.72、v_max=68.60（扩散尺度 u_nd=u_lu·H/α）。
锚点 `XVAL-LBM-NO-BEM-CP-REPO-IN-OSS-INVENTORY`（第 2 轮；前轮 m17997 因 llm_error 零工具调用 honest-FAIL，本单重开）。

## ② 怎么做的
- 先用库入口 `tensorlbm.thermal.simulate_natural_convection`（D2Q9 速度 + D2Q5 温度双分布 BGK、Guo 浮力、ABB 等温壁、解缠绝热壁），驱动只做参数换算与观测量提取（零本地物理）。
- 开源横向调研（机器产生 `artifacts/oss_survey.json`）：oss_inventory {S['n_runs'] and json.loads((ART/'oss_survey.json').read_text())['oss_inventory_rows']} 行 + `~/oss-cache` 三仓逐文件扫描 → 判定 **{S['oss_survey_verdict']}**：本地无「双分布热 LBM + 自然对流」可解析实现 → 外部锚只能走文献谱系（与 negative_knowledge_ledger m17996 的 thermal_energy_transport_lbm 无 OSS 参考一致）。
- 预注册 `predeclared.json`（判据/网格集/稳态停止规则/混淆因子）先于任何产出运行写定；`analyze.py` 从原始 run JSON 程序化复算每个误差与闸门，`make_doc.py` 渲染文档，零手抄。
- 计算走本地后台（集群 cj1782/cj1783 排队 300+ 未派发，记录为生态摩擦）。

## ③ 参数表
τ=0.6 → ν=0.03333、α=0.04695、τ_T=0.6408；τ=0.8 → ν=0.1、α=0.14085、τ_T=0.9225；H=nx；ΔT=1；Ma_lattice ≲0.07；网格 N={', '.join(str(r['nx']) for r in rows)}。

## ④ 计算结果（程序化注入，零手抄）
| N | τ | steps | 稳态触发 | Nu | err(Nu) | u_max_nd (err) | v_max_nd (err) | resid_u |
|---|---|---|---|---|---|---|---|---|
{tbl}

闸门（由 `artifacts/summary.json` 复算，逐条）：
{gates}

## ⑤ 注意点 / 局限
- D2Q5 温度方程截断误差使粗网格 Nu 系统性偏低（负偏差）；半程反弹壁使有效腔长 H=nx；float32 长时累积。
- 未触发稳态检测的网格不得计入判据（G4 逐网格核）；本轮若有网格未收敛，如实标 FAIL，不改口径。
- 集群队列积压（排队 300+/未派发）→ 生产算例退回本地后台，吞吐受限。

## ⑥ 误差判定（是否 <3% / <15%）
见 ④ 表逐项 err%；`artifacts/summary.json → gates.G2_owner_15pct / G3_caliber_3pct` 给出机器判定，
文档 `docs/REPORT_XVAL_LBM_NO_BEM_CP_REPO_IN_OSS_INVENTORY.md` 的 ④ 表同步标注。**本轮不粉饰：未过即写 FAIL。**
"""
    payload = {
        "title": "XVAL · TensorLBM 热 LBM vs de Vahl Davis 1983 (Ra=1e5) — 第2轮（文献锚 + OSS 横向调研）",
        "summary": "thermal LBM DHC Ra=1e5 XVAL：OSS 清单/缓存无参考实现（机器调研落 artifacts/oss_survey.json）→ 走文献锚；多网格 + τ 敏感性；闸门由原始 JSON 复算",
        "scenario_key": "xval-lbm-thermal-dhc-ra1e5",
        "mission_id": MISSION,
        "tags": ["TensorLBM", "XVAL", "thermal", "natural-convection", "de-Vahl-Davis", "Ra1e5"],
        "markdown": doc + "\n\n---\n\n# 附录：报告六段（程序化注入）\n" + md,
        "figures": figs,
    }
    rp = Path(f"/tmp/report_{MISSION}.json")
    rp.write_text(json.dumps(payload))
    status["report_json_bytes"] = rp.stat().st_size
    status["figures_bytes"] = sum(len(f["png_base64"]) for f in figs)

    # ---- publish
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    user = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
    pw = os.environ.get("TENSOR_AGENT_PASS", "")
    try:
        req = urllib.request.Request(
            base + "/auth/login",
            data=json.dumps({"username": user, "password": pw}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        tok = json.loads(urllib.request.urlopen(req, timeout=30).read())
        token = tok.get("access_token") or tok.get("token")
        status["login"] = "ok" if token else str(tok)[:200]
        req2 = urllib.request.Request(
            base + "/tensor/reports", data=rp.read_bytes(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST")
        body = urllib.request.urlopen(req2, timeout=120).read().decode()
        status["report_post"] = body[:500]
    except Exception as exc:  # noqa: BLE001
        status["report_post_error"] = repr(exc)

    (ART / "finalize_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False))
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())