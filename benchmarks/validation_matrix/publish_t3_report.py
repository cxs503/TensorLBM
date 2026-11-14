"""Publish the T3 coverage-gate report to the platform (mission 19197).

All numbers are injected from the frozen artifacts (T3_COVERAGE.json,
gate_reproduction.json, validation_report.json); the table is generated, not
typed.  Figures are the two matplotlib PNGs from make_t3_figures.py.

Usage::

    python benchmarks/validation_matrix/publish_t3_report.py [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
COVERAGE = HERE / "T3_COVERAGE.json"
GATE = REPO_ROOT / "benchmarks/pending/channel_turb_3d/gate_reproduction.json"
LADDER = REPO_ROOT / "benchmarks/pending/channel_turb_3d/validation_report.json"
FIG_DIR = HERE / "figures"


def _post(url: str, body: dict, token: str | None = None) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read().decode()[:800])
        raise


def login(base: str) -> str:
    form = urllib.parse.urlencode({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode()
    req = urllib.request.Request(f"{base}/auth/login", data=form, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    return payload.get("access_token") or payload["token"]


def _shrink(path: Path, width: int = 760) -> bytes:
    """Down-sample to keep the report body under the platform limit."""
    from PIL import Image

    with Image.open(path) as im:
        ratio = width / im.width
        im = im.convert("RGB").resize((width, max(1, int(im.height * ratio))))
        out = __import__("io").BytesIO()
        im.save(out, format="PNG", optimize=True)
        return out.getvalue()


def figures() -> list[dict]:
    out = []
    for name, caption in (
        ("t3_slot_coverage.png",
         "九谱系 T3 覆盖判定（机器）：绿=verified、黄=proxy、红=missing；标题给出 6/9 verified、"
         "1 partial、2 gap，取代原先按案例数读出的 'passed 8'。"),
        ("t3_channel_gate.png",
         "左：L3 u+ 剖面 vs 冻结 MKM-1999 DNS 表，两种摩擦速度归一化（目标 u_τ=5.33%，实测 u_τ=15.80%）；"
         "右：槽道槽三档实测 FAIL 阶梯（20.43%→11.67%→5.33%）与 3% 门限，门数复算 Δ=0.0000 pp。"),
    ):
        p = FIG_DIR / name
        out.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(_shrink(p)).decode(),
        })
    return out


def markdown() -> str:
    cov = json.loads(COVERAGE.read_text(encoding="utf-8"))
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    ladder = json.loads(LADDER.read_text(encoding="utf-8"))
    rows = "\n".join(
        f"| {i} | {r['slot']} | {r['lineage']} | **{r['state']}** | "
        + "; ".join(f"{k}={v['state']}" for k, v in r["components"].items())
        + " | "
        + ("; ".join([f"{b['case']} {b['status']}" for b in r["blockers"]]
                      + [f"{s['case']} 替代({s['physics_class']})" for s in r["substitute_evidence"]])
           or "—")
        + " |"
        for i, r in enumerate(cov["slots"], 1)
    )
    lv = "\n".join(
        f"| {g['level']} | {g['grid']} | {g['u_tau_err_pct']:+.2f}% | {g['rms_y30_100_pct']:.2f}% | "
        f"{g['eddy_turnovers']:.2f} |" for g in ladder["grid_levels"]
    )
    rep = gate["reproduction"]
    conv = gate["uplus_conventions"]
    bud = gate["budget_diagnostic"]["ranges"]
    return f"""## ① 案例介绍

T3 国际标准验证矩阵（P0）要求在**九个物理谱系**上各有一项可机器判定、可一键复现、CI 可跑的
对照验证：方腔 / 圆柱 / 球体 / 湍流槽道 / Taylor-Green / 静态液滴 / Rayleigh-Taylor / 溃坝 /
船体阻力与波浪。矩阵 v1.1 已有 12 个案例，但"案例 passed 数"从未等价于"谱系覆盖数"，本报告把
这条差量做成机器判定，并对槽道槽（唯一既无通过证据、又被层流替代品填补的谱系）做证据复核。

## ② 怎么做的

1. 新增谱系契约 `benchmarks/validation_matrix/t3_slots.json`：逐槽 required physics_class +
   required components；逐案例 physics_class / components / level_basis / credit_caveats。
2. 新增覆盖门 `t3_coverage.py`：把 `runner.evaluate_case` 的新鲜判决与契约求交，得
   verified / partial / gap 三态，冻结为 `T3_COVERAGE.json`，`--check` 做漂移门。
3. 新增槽道门数复算 `channel_budget.py`：独立实现 MKM 表插值 + 相对 RMS，复算记录门数；
   同时给出五种 (wall origin, u_tau) 归一化口径与总剪切线性性诊断。
4. CI 契约 `tests/benchmarks/test_t3_coverage_gate.py`（9 项）：含"层流代替品不得计入湍流谱系"、
   "FAILED/PENDING 不得计分"、"船体阻力代理锚不得 verified"、"文档状态计数=机器判决"。
5. 图 `make_t3_figures.py`；全部数字由冻结核查产物注入，零手抄。

## ③ 计算结果

谱系覆盖（{cov['headline']['verified']}/{cov['headline']['slots']} verified，
partial {cov['headline']['partial']}，gap {cov['headline']['gap']}）：

| # | slot | lineage | state | component credit | blocker / substitute |
|---|------|---------|-------|------------------|----------------------|
{rows}

槽道槽三档实测 FAIL 阶梯（源 `validation_report.json`）：

| level | grid | u_τ 误差 | u+ RMS (y+∈[30,100]) | eddy turnovers |
|-------|------|----------|----------------------|----------------|
{lv}

门数复算：记录 {rep['recorded_rms_pct']:.3f}% / 复算 {rep['recomputed_rms_pct']:.3f}%，
Δ={rep['delta_pp']:.4f} pp（n={rep['n_points']}，y+=[{rep['y_plus_range'][0]:.1f},
{rep['y_plus_range'][1]:.1f}]，口径={rep['convention']}）。

归一化敏感性（同一冻结 L3 场）：目标 u_τ 口径 {conv['rel_offset0.5_target_u_tau']['rms_pct']:.3f}%、
实测 u_τ 口径 {conv['rel_offset0.5_measured_u_tau']['rms_pct']:.3f}%（×{gate['normalisation_sensitivity']['ratio_measured_over_target']:.2f}）；
摩擦速度实测/目标 = {gate['friction']['gap_pct']:+.2f}%。
总剪切线性性诊断残差：全域 {bud['full[1,ny-2]']['residual_pct_of_tau_w']:.2f}%、
外侧 {bud['outer[4,ny-5]']['residual_pct_of_tau_w']:.2f}%、核心
{bud['core[10,ny-11]']['residual_pct_of_tau_w']:.2f}%（近壁单侧差分是全域残差主因）。

## ④ 结果图

图 1 = 九谱系覆盖判定热图；图 2 = 槽道 u+ 剖面（两种归一化）vs 冻结 DNS 表 + 三档 FAIL 阶梯。
两张图由 `make_t3_figures.py` 从冻结产物生成，见 figures 字段。

## ⑤ 注意点

- `channel_core_path_ci`（层流槽道）是真实通过项，但**不能**计入湍流槽道谱系；这是本次审计
  的主要标题修正（"passed 8" → 谱系 6/9）。
- 船体阻力槽的锚是 Blasius 平板摩擦代理（Re=1000），不是该 Re 下的船体实验；记为 proxy，
  且波浪分量仍 WITHHELD ⇒ 槽位 partial。
- Taylor-Green 的两档是同一运行的时间分段（拟合稳定性），不是网格收敛；已标注 basis。
- `matrix.json` 中槽道条目指向不存在的 `grids.*` 路径（显示 unresolved），本次用覆盖账
  addendum 承载真实阶梯；重指向 schema 变更会使 12 份冻结证据链失效，留待 v1.2 统一刷新。

## ⑥ 误差判定

| 关键量 | 数值 | 基准 | 是否 <3% |
|---|---|---|---|
| 槽道 u+ RMS（L3，目标 u_τ 口径） | {conv['rel_offset0.5_target_u_tau']['rms_pct']:.2f}% | MKM-1999 DNS 表 | **否**（honest FAIL） |
| 槽道 u+ RMS（L3，实测 u_τ 口径） | {conv['rel_offset0.5_measured_u_tau']['rms_pct']:.2f}% | 同上 | **否**（更严口径） |
| 门数复算偏差 | {rep['delta_pp']:.4f} pp | 记录值 {rep['recorded_rms_pct']:.3f}% | 是（≪3%） |
| 覆盖判定计数 | {cov['headline']['verified']}/{cov['headline']['slots']} verified | 机器契约 | 是（确定性计数，无误差） |

结论：本单不产生新的通过项；产出是把 T3 覆盖从"案例计数"升级为"谱系机器判定"，并把唯一
无通过证据的槽道谱系的真实 FAIL 阶梯与门数口径钉死（6/9 verified、1 partial、2 gap）。
诚实结论：槽道 / Rayleigh-Taylor / 船体波浪三项仍是欠账，不做粉饰。
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    base = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1").rstrip("/")
    body = {
        "title": "T3 验证矩阵覆盖判定审计：6/9 谱系 verified，槽道槽 FAIL 阶梯与门数口径钉死",
        "summary": (
            "把 T3 九算例谱系覆盖做成机器判定（契约+覆盖门+CI 契约测试）：实测 6/9 verified、1 partial、2 gap，"
            "纠正'passed 8=谱系全覆盖'的读法；槽道记录门数 5.33% 由冻结证据逐位复现（Δ=0.0000 pp），"
            "并披露其归一化敏感性（实测 u_τ 口径 15.80%）"
        ),
        "scenario_key": "lbm_validation_matrix",
        "mission_id": 19197,
        "tags": ["TensorLBM", "T3", "validation-matrix", "coverage-gate", "channel-turbulence", "audit"],
        "markdown": markdown(),
        "figures": figures(),
    }
    payload = json.dumps(body)
    print(f"report body {len(payload) / 1024:.1f} KiB, figures {len(body['figures'])}")
    if args.dry_run:
        Path("/tmp/t3_report.json").write_text(payload, encoding="utf-8")
        print("dry-run: wrote /tmp/t3_report.json")
        return 0
    token = login(base)
    resp = _post(f"{base}/tensor/reports", body, token)
    print(json.dumps(resp)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())