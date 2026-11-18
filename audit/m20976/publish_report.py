"""m20976: figures + platform report for the board-audit of m20893 (R2).

Figures
  fig1  left : fill/|u| cloud at T=0.5, a=16 (fresh 400-step run of the delivered
               arm, snapshots=True) -> real field, not a mock
        right : Z(T) of the delivered a=32 tau=1.2 arm (per-T series recovered
               from cluster/jobs/cj3062.log) vs Martin & Moyce 1952
  fig2  error vs grid a (5 rungs) with the 15 % acceptance / 3 % target bands,
        and the R2 report's (wrong) a=48 point overlaid to expose F1
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parent
sys.path.insert(0, str(HERE / "repro"))
sys.path.insert(0, str(WT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ART = json.loads((HERE / "repro" / "r2_cj3032_maxima.json").read_text())
MM = json.loads((WT / "src" / "artifacts" / "martin_moyce_1952_dam_break.json").read_text())

# ---- run the delivered arm with snapshots (cheap: 400 steps, ~12 s) ----------
import resolution_study as R  # noqa: E402

res = R.run_tensorlbm(16, 1.0, 5e-5, 0.5, 0.25, snapshots=True)
umag = np.array(res["final_umag"])
fill = np.array(res["final_fill"])
at50 = R.eval_run(res, "mm")
(HERE / "evidence").mkdir(parents=True, exist_ok=True)
(HERE / "evidence" / "snapshot_a16_T0.5.json").write_text(json.dumps({
    "a": 16, "T": res["series"][-1]["T"], "steps": res["steps"],
    "Z_cell": res["series"][-1]["Z_cell"], "Z_sub0.05": res["series"][-1]["Z_sub0.05"],
    "Z_cell_err_pct": at50["vs_mm"]["Z_cell"]["max_abs_rel_err_pct"],
    "Z_sub05_err_pct": at50["vs_mm"]["Z_sub0.05"]["max_abs_rel_err_pct"],
    "shape": list(umag.shape)}))

# ---- per-T series of the delivered a=32 tau=1.2 arm from cj3062.log ---------
tail = json.loads(Path("/srv/tensor/agent/cluster/jobs/cj3062.log").read_text())["output_tail"]
t12 = None
i = tail.find('"tau":1.2')
if i >= 0:
    s = tail.rfind('{"arm"', 0, i)
    dec = json.JSONDecoder()
    try:
        t12, _ = dec.raw_decode(tail[s:])
    except Exception:  # truncated head
        t12 = None
zser = None
if t12 and t12.get("series"):
    zser = [(p["T"], p["Z_sub0.05"]) for p in t12["series"]]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
im = ax[0].imshow(np.where(fill > 1e-6, umag, np.nan), origin="lower", cmap="viridis")
ax[0].set_title(f"a=16, T={res['series'][-1]['T']:.3f}: |u| (liquid/interface cells)")
ax[0].set_xlabel("x [lattice]"); ax[0].set_ylabel("y [lattice]")
plt.colorbar(im, ax=ax[0], label="|u| [lattice]")

mmt = [d["t_star"] for d in MM["data"]]
mmz = [d["X_star"] for d in MM["data"]]
ax[1].plot(mmt, mmz, "k^--", ms=3, label="Martin & Moyce 1952")
if zser:
    ax[1].plot([p[0] for p in zser], [p[1] for p in zser], "o-", ms=2.5,
               label="TensorLBM a=32, tau=1.2 (cj3062 series)")
else:
    ax[1].text(0.5, 0.5, "cj3062 per-T series truncated", transform=ax[1].transAxes)
ax[1].set_xlabel("T = t sqrt(g/h0)"); ax[1].set_ylabel("Z = x_front / a")
ax[1].set_title("front position vs experiment"); ax[1].legend(fontsize=7)
fig.tight_layout()
f1 = WT / "figures"; f1.mkdir(exist_ok=True)
p1 = f1 / "fig_m20976_field_and_front.png"
fig.savefig(p1, dpi=150)

fig2, ax2 = plt.subplots(figsize=(6.4, 4.2))
ladder = sorted([r for r in ART["rows"] if r["label"].startswith("tensorlbm")
                 and r["tau"] == 1.0], key=lambda r: r["a"])
aa = [r["a"] for r in ladder]
ax2.plot(aa, [r["Z_sub05_err_pct"] for r in ladder], "o-", label="Z_sub(0.05) vs M&M")
ax2.plot(aa, [r["Z_cell_err_pct"] for r in ladder], "s--", label="Z_cell vs M&M")
ax2.plot([48], [79.37], "rx", ms=11, mew=2,
         label="a=48 as printed in report §3.1/§6 (OSS-port row, F1)")
ax2.axhline(15, color="r", ls=":", label="15 % acceptance")
ax2.axhline(3, color="g", ls=":", label="3 % target")
ax2.set_xlabel("a [lattice, h0=2a, nx=5a]"); ax2.set_ylabel("max |rel err| [%], T in [0.5,4]")
ax2.set_title("R2 ladder: non-monotone, only a=32 < 15 %")
ax2.legend(fontsize=7); fig2.tight_layout()
p2 = f1 / "fig_m20976_error_vs_a.png"
fig2.savefig(p2, dpi=150)


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


ladder_md = "\n".join(
    f"| {r['a']} | {r['Z_cell_err_pct']} | {r['Z_sub05_err_pct']} | {r['u_max']} | "
    f"{r['inventory_drift_pct']} | {'是' if r['Z_sub05_err_pct'] <= 15 else '否'} | 否 |"
    for r in ladder)

md = f"""# m20976 交差核验：XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS（对象 m20893/R2）判级 L2，gate=reopened

## ① 案例介绍
二维水柱瞬时坍塌（Körner/Thürey D2Q9 自由面，无表面张力）：水柱宽 `a`、高 `h0=2a`，域 `nx=5a`、`ny=3a`，
四边固壁，`g=5e-5`，`τ=1.0`（ν=1/6）。无量纲 `Z=x_front/a`、`T=t√(g/h0)`。基准 = Martin & Moyce (1952)
（DOI 10.1098/rsta.1952.0006, Fig. 4，17 个数字化点，树内 `src/artifacts/martin_moyce_1952_dam_break.json`）。
对象 m20893 是 R2：把 R1(m18073) 预声明却从未执行的 `a∈{{16,24,32,48,64}}` 分辨率阶梯推进到全窗 `T∈[0.5,4]`，并加 τ 扫描
`0.6–1.4`（a=32）、静水控制组、OSS port a=48 档。参数：float32 torch CPU，12 worker 调度作业 cj3032/cj3062。

## ② 怎么做的（审计流程）
1. 对象工作树 `/srv/tensor/agent/work/m20893` 已不可访问 ⇒ 以**交付 commit `578410df6`**（仅在 `refs/heads/agent/m20893`）、
   `tensor_dev_missions.id=20893.result`、`tensor_reports.id=18989`、`tensor_benchmark_cases` 为证。
2. 三件套核验：工件文件存在性、**工件 vs 作业回执逐字段对账**（`audit_m20976/parse_job_logs.py` 解析
   `cluster/jobs/cj3032.log|cj3062.log`）、repro_cmd 实跑（`run_tensorlbm(16, T≤0.5)` 400 步 12.21 s）。
3. 基准裁判：`SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS'` → **0 行**
   （`tensor_xval_cases`/`repro_pool` 同锚亦 0）⇒ 机器裁判缺席，只能以预声明 15 % 门 + M&M 实验点作替代判据。
4. 预声明分析复算：用交付的 `fit_order` 对 5 档最大误差做 `E(a)=E_inf+C·a^-p`；`richardson()` 试算。
5. 仓内红测：`pytest tests/test_xval_oracle_minimal_d2q9_fs.py -q`。

## ③ 计算结果（裁判 = 预声明 15 % 门；数字取自交付工件，已与作业回执逐字段对上）

| a | Z_cell % | Z_sub(0.05) % | u_max | inv drift % | <15 % | <3 % |
|---|---|---|---|---|---|---|
{ladder_md}

* a=32 的 13.81 % 与 R1 冻结值 13.806 % 同量级 ⇒ 同代码同参可复现。
* τ 扫描 @a=32（Z_cell %）：τ=0.6→66.62, 0.8→25.67, 1.0→13.81, 1.2→15.96, 1.4→18.12。
* OSS port a=48：79.37 %、`u_max` 恒 0.8165（速度限幅饱和）、液量漂移 609.79 % ⇒ **不能作裁判**。
* 拟合：`Z_cell` 与 `Z_sub.05` 均 `ok=false, "no admissible (E_inf, p) in grid"`；Richardson `NOT_COMPUTABLE_FROM_ARTEFACT`
  （`KeyError:'points'`，工件只存标量最大值）。

## ④ 结果图
左：交付臂 a=16、T=0.5 的 |u| 云图（本审计现跑 400 步 `snapshots=True`，真场量）；右：交付的 a=32 τ=1.2 臂的 Z(T)
（逐 T 序列从 `cj3062.log` 尾部回收）vs Martin & Moyce 1952；第二图：5 档网格误差 vs 15 %/3 % 门，红叉为 R2 报告
§3.1/§6 印出的错误 a=48 点（实为 OSS port 臂，F1）。

## ⑤ 注意点
1. **报告 a=48 行错位（F1）**：§3.1 与 §⑥ 印的是 OSS port 数字 79.37/70.05/0.8165/609.79，梯级真值 28.05/27.37/0.2517/0.0783
   （`rj_cj3032_maxima.json` 与 `cj3032.log` 汇总行皆如此，报告 §⑤ 正文也用 28.05）⇒ 验收表与自身正文矛盾。结论不变（都远超 15 %）。
2. **预声明分析未交付（F2）**：拟合无可行解、Richardson 不可算，报告未给 p/E_inf；H1 否证仅以「误差非单调」定性论证（该论证成立）。
3. **裁判表 0 行（F3）**、**红测未修（F4）**：`test_tensorlbm_arm_smoke` 仍 1 failed/4 passed（`run_xval.py:137 ModuleNotFoundError`）。
4. **未合入 main（F5）**：`578410df6` 只在 `agent/m20893`；而 `result.pr_gate` 记 PR 2418 merged —— 与树证据矛盾，需平台侧核对。
5. **取证口径（F6）**：`cj3032` rc=1 崩在 `KeyError:'label'`，保留 tail 1 561 B 无 payload ⇒ 梯级逐 T 序列确实丢失；但 `cj3062`
   tail 仍存 τ 臂逐 T 序列 ⇒「序列全部丢失」应限定范围；声明的补算 `cj3085/cj3087` 在盘上**无日志**。
6. **参数表错（F7）**：§① 称 a=64→6400 步，按 `steps=ceil(T/√(g/h0))`, h0=2a 应为 203 步（差 32×；本地 a=16 T≤0.5→400 步自洽）。
7. **外部锚摘要过期（F8）**：`DESIGN` 用文件自带 `sha256=e6300b9a…`，实算 `6f766bfa…`。
8. **控制组设计缺陷（F9）**：静水控制仍用半域宽水柱初始化（溃坝），1e-6 伪流门不适用，报告如实记 FAIL。

## ⑥ 误差判定
| a | max err vs M&M % (Z_sub.05) | <15 % | <3 % |
|---|---|---|---|
| 16 | 27.39 | 否 | 否 |
| 24 | 15.88 | 否 | 否 |
| 32 | 14.66（Z_cell 13.81） | 是（仅此档） | 否 |
| 48 | 27.37 | 否 | 否 |
| 64 | 42.19 | 否 | 否 |

**是否 <15 %**：仅 a=32 成立 ⇒ 不能声称网格无关达标。**是否 <3 %**：**否**，最小观测误差 13.81 %，且误差非单调
（28.16→16.67→13.81→28.05→42.86）⇒ 预声明 H1（E_inf<3 %）**否证**，与报告一致（honest FAIL，未粉饰）。
**判级 L2**（真代码 690 行、输入→计算→输出闭环、KPI 可回溯到作业回执）；**L3 不成立**（harness 无反向调用点、未入 main）；
**L4 不成立**（裁判 0 行 + 精度 1/5 档 + 拟合/外推不可算 + 红测）。gate = **reopened**。
审计未跑：全窗四档独立重算作业 cj3101 递交后仍在排队（如实记 not_run，不按 pass 计）。
"""

payload = {
    "title": "m20976 交差核验：XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS（对象 m20893）判级 L2 · gate=reopened",
    "summary": "核验 m20893/R2：三件套=工件✅/裁判0行❌/repro✅；全窗 5 档仅 a=32 过 15 %，最小 13.81 % ≫3 %，H1 否证（honest FAIL 成立）；"
               "判级 L2，gate=reopened（报告 a=48 行错位、拟合/Richardson 未交付、红测未修、未合入 main）",
    "scenario_key": "XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS",
    "mission_id": 20976,
    "tags": ["board-audit", "XVAL", "LBM", "free-surface", "dam-break", "TensorLBM", "m20893", "L2", "reopened"],
    "markdown": md,
    "figures": [
        {"name": "fig_m20976_field_and_front.png",
         "caption": "左：交付臂 a=16、T=0.5 的 |u| 场云图（审计现跑 400 步，液体/界面格）；右：a=32 τ=1.2 的 Z(T) vs Martin & Moyce 1952（序列取自 cj3062.log）",
         "png_base64": b64(p1)},
        {"name": "fig_m20976_error_vs_a.png",
         "caption": "R2 分级误差 vs 网格 a（T∈[0.5,4]），15 %/3 % 门；红叉=报告 §3.1/§6 印出的 a=48 点（实为 OSS port 臂，F1）",
         "png_base64": b64(p2)},
    ],
}
Path("/tmp/report.json").write_text(json.dumps(payload))

base = os.environ["TENSOR_API_BASE"].rstrip("/")
body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                   "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
req = urllib.request.Request(base + "/auth/login", data=body,
                             headers={"Content-Type": "application/json"})
tok = json.loads(urllib.request.urlopen(req, timeout=30).read())
token = tok.get("access_token") or tok.get("token")
req2 = urllib.request.Request(base + "/tensor/reports", data=Path("/tmp/report.json").read_bytes(),
                              headers={"Content-Type": "application/json",
                                       "Authorization": f"Bearer {token}"})
resp = json.loads(urllib.request.urlopen(req2, timeout=60).read())
print("REPORT:", json.dumps(resp)[:300])
print("FIG:", p1, p2)