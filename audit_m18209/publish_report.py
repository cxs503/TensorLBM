"""m18209 board audit of m18141: evidence figure + paper-grade report publication."""
import base64
import io
import json
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
XT = "/root/tp-agent/work/m18141/xval/XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION"
API = "http://127.0.0.1:8010/api/v1"
USER, PASS = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]

bench = json.load(open(f"{XT}/benchmark.json"))
vfy = json.load(open(f"{HERE}/verify_m18141.json"))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(13.6, 5.0))

# (a) key benchmark rows: got vs tolerance, log scale
rows = [r for r in bench["rows"]]
sel = []
for r in rows:
    k = r["kpi"]
    if r["tol"] and r["tol"] > 0 and (r["rel_err"] > 0 or k.endswith("_l2")):
        sel.append(r)
sel = sel[:14]
names = [r["kpi"].replace("_", " ")[:34] for r in sel]
errs = [max(r["rel_err"], 1e-18) for r in sel]
tols = [r["tol"] for r in sel]
cols = ["#2a9d3f" if r["pass"] else "#c1121f" for r in sel]
y = range(len(sel))
ax[0].barh(list(y), errs, color=cols, height=0.6, label="measured rel. error (got vs ref)")
ax[0].plot(tols, list(y), "k|", ms=16, mew=2, label="predeclared tolerance")
ax[0].set_yticks(list(y))
ax[0].set_yticklabels(names, fontsize=7)
ax[0].set_xscale("log")
ax[0].set_xlabel("relative error (log)")
ax[0].set_title("(a) benchmark.json row-by-row: measured vs tolerance\n"
                "green = pass, red = fail (control arm / known cost / physics)")
ax[0].legend(fontsize=7, loc="lower right")
ax[0].grid(alpha=0.3, axis="x")

# (b) predeclared H1 thresholds vs measured -> the falsifier that fired
labels = ["roundtrip\nA1 N16", "roundtrip\nA1 N32", "total-force cons.\nA1 N16", "total-force cons.\nA1 N32"]
vals = [vfy["rows"]["indep_roundtrip_A1_N16_const_l2"],
        vfy["rows"]["indep_roundtrip_A1_N32_const_l2"],
        vfy["rows"]["indep_deposit_dev_A1_N16_const"],
        vfy["rows"]["indep_deposit_dev_A1_N32_const"]]
bcols = ["#2a9d3f", "#2a9d3f", "#e08c00", "#c1121f"]
ax[1].bar(labels, [max(v, 1e-18) for v in vals], color=bcols, width=0.55)
ax[1].axhline(1e-6, color="k", ls="--", lw=1.4, label="H1 required < 1e-6 (both KPIs)")
ax[1].axhline(0.15, color="#c1121f", ls=":", lw=1.6, label="predeclared H0 falsifier >= 15 %")
ax[1].set_yscale("log")
ax[1].set_ylabel("relative deviation (log)")
ax[1].set_ylim(1e-17, 3)
for i, v in enumerate(vals):
    ax[1].annotate(f"{v:.2e}", (i, max(v, 1e-18)), textcoords="offset points", xytext=(0, 6),
                   ha="center", fontsize=8)
ax[1].set_title("(b) predeclared hypothesis H1 (roundtrip AND conservation < 1e-6)\n"
                "auditor recompute: conservation 2.5e-2 / 8.5e-1 -> H1 falsified, H0 falsifier fired")
ax[1].legend(fontsize=7, loc="upper left")
ax[1].grid(alpha=0.3, axis="y")

fig.suptitle("m18141 XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION — board audit (m18209): "
             "machine-precision PASS is by construction, the predeclared conjunction is not", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
png = f"{HERE}/fig_audit_m18141.png"
fig.savefig(png, dpi=150)
plt.close(fig)
print("wrote", png, os.path.getsize(png), "bytes")

# ---------------------------------------------------------------- report
md = f"""# 交差审计 m18141 · XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION（TensorLBM）→ L3，L4 扣留

**审计单**：m18209（任务榜令交差核验单，非重做） · **被审**：worktree `m18141`，HEAD `e2743f0e` ·
**OSS 锚**：joconnor22/LIFE @ `2af2428`（`inc/Utils.h:220-231` 三点 δ、`src/IBMNode.cpp:73-74` spread、`src/Objects.cpp:235-320` computeEpsilon）
**裁定**：L3（L1/L2/L3 实质具备，L4 扣留），disposition = *approve with notes*。

## 1. 案例与取证

被审交付=把 LIFE 的 **ε-Gram 隐式修正**（3 点正则 δ + 每体小 Gram 系统 `Aε=1` + `f·ε·ds·δ` 沉积）逐行语义移植进 TensorLBM 的 IBM 力传递面，并给出力传递往返/沉积记账（Part A）与 Re=100 固定圆柱阻力（Part B）两条验证面。

| 三件套 | 核验结果 |
|---|---|
| `run_artifacts` | `results.json`(1154 行)、`benchmark.json`(25 行)、`blockage_sweep.json`、`run.log`、`sweep.log`、`manifest.json`(逐文件 sha256_16)、`fig1–fig4.png` 均实存（非空壳） |
| `benchmark` ref vs got | 25 行 / 20 pass / 5 fail；审计用**自写独立实现**（自建支撑+spread/interp，不调 `run_xval.py`）复算 4 行，**逐位一致**（A2 N16 2.454839e-02、A2 N32 4.600482e-01、径向 2.188069e-03、沉积代价 8.521660e-01；ε 1.811561–1.879587） |
| `repro_cmd` | pytest **12/12 passed**（26.18 s）；`python -m tensorlbm.ibm_epsilon_gram` 6/6 自检；`run_xval.py` Part A **8/8 臂与 run.log 逐位一致**、Part B A1 `Cd=1.6156 / tail=1.6431` 复现；`sweep_blockage.py 192x96` `Cd=1.621828698136388` 完全复现 |
| 平台裁判 | ⚠️ `SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION'` → **0 行**（全表 94 行）：无注册算例/容差，基准裁判在平台侧不存在，只能靠交付自带 `predeclared.json`+`benchmark.json` |

## 2. 判级（级别=功能实质，不由报告长度决定）

- **L1**：解析判据到机器精度（1-D 分割单位 1.11e-16；2-D 支撑和 0.0），12 测试断言不变量。
- **L2**：`src/tensorlbm/ibm_epsilon_gram.py` 506 行真实现（`:62 δ`/`:82 支撑`/`:144 ds`/`:167 Gram 解`/`:262 spread`/`:289 interpolate`/`:360-370 对外 API`），输入→计算→输出齐备，三条路径（loop/向量化/IBMStencil）互证 0.0–2.2e-16。
- **L3**：修正**不是并行代码路径**——`:370 spread_with_correction` 走仓内 `IBMStencil.spread`（`ibm_force_transfer.py:126`），`tests/test_ibm_epsilon_gram_solver_smoke.py:47` 经 `collide_bgk/stream/body_source_2d/disc_mask_2d` 组真实时间推进并让修正进入力路径。**如实标注**：`grep` 显示 **0 个生产调用者**（`__init__.py` 未 re-export），修正为 opt-in。
- **L4 扣留**（4 条硬理由）：① 裁判表 0 行；② 唯一独立物理 KPI 不达标——`Cd=1.6156` vs 无界文献带中心 1.35 = **+19.7% > 15%**（交付自报 honest FAIL）；③ 机器精度的 PASS 行**按构造为恒等**（`Aε=1` ⇒ 常量场往返精确，审计探针 `A·ε−1=3.3e-16`），它认证接线正确性而非保真度，唯一非平凡恒等行（N=32 径向 2.19e-3）无网格/时间收敛研究；④ 物理均值窗口仅 ~0.4 个脱落周期（std 1.39）。
- **L5** 不主张：无实验锚、无新物理能力（本质是归一化修正）。

## 3. 关键发现（图 a/b 为证）

**图 (a)** 把 `benchmark.json` 逐行 measured-relative-error 与预声明容差并列：绿=pass、红=fail。**图 (b)** 把预声明 H1 的门槛（往返 **且** 总力守恒 < 1e-6）与审计重算的实测值并列——往返 1.3e-16 过线，但**总力守恒 2.517e-02（N=16）/ 8.522e-01（N=32）越过 1e-6 六个数量级以上，并触发 H0 自己的 15% 证伪器**。

- **F1（moderate）预声明 H1 被自己的证伪器打掉、报告未裁决**：`predeclared.json` 写死 H1 = 往返<1e-6 **AND** 守恒<1e-6；H0 证伪器 = 任一 ≥15%。实测守恒 85%（N=32）→ H1 合取不成立、H0 触发。报告 §4.2 两个数字都如实写了、也写了"两条性质不可兼得"，但随后把验收依据换成 §4.1 往返 KPI（"方法验收依据仍是 §4.1"）——**看到结果后换判据**（post-hoc KPI switch）。非造假，属判据裁决缺失，必须入账 open item。
- **F2（moderate）裁判缺失**：平台 `tensor_benchmark_cases` 该 anchor 0 行。
- **F3（moderate）claim 定性**："改善 210×–1e14" 的头部行是构造性恒等；真正非平凡的是 **A2 控制组（2.5e-2 / 4.6e-1）** 与 **沉积代价（8.52e-1）**。报告 §4.2 披露机制，但 §4.1/§8 呈现读起来像实证验证。
- **F4（moderate）阻塞比证伪被平均口径混淆**：主算例 `700/500/200`、补充点 `700/300/400`（`blockage_sweep.json` 明载），报告称"唯一变量是盒子"但时间平均窗也变了；`n_avg=200` ≈0.4 个脱落周期。故"阻塞比不解释 +20%"是弱证据（两点拟合 k=−0.09 报告自己判无意义）。honest FAIL 判定成立，**归因**不成立。
- **F5/F6/F7（minor）**："≤3.9%" 以最大值归一（真值 4.09%）；`fig4` 图注对"收敛"表述偏强（窗口 0.4 周期）；`predeclared.json` mission_id=17862 且产物 `git_sha` 是交付 commit 的父提交 + `git_dirty=true`（审计核对无源漂移）。
- **F8（positive）纪律守住**：物理臂如实判 FAIL 不调参；同核控制组隔离唯一变量 ε；阈值预声明并发布；逐文件 sha256；测试断言不变量而非仅执行。

## 4. 注意点与局限（适用范围）

- OSS LIFE 二进制**从未编译/运行**（需 Boost+LAPACK+完整 FSI 算例）→ 无程序对程序对跑；锚是逐行语义移植，交付已在 `predeclared.json.baseline_control.oss_note`、REPORT §7/§8 披露。
- 3-D 路径、柔体每步 ε 重解、MPI 分解：声明未实现，未验证。
- 恒等臂为 single-shot（预声明），无网格/时间步收敛序列。
- 全部数字本地同一 loongarch64 机复现，未上集群。
- **独立性限制**：审计与被审共享主机/torch 构建/OSS 阅读；独立性体现在代码路径（自写 support/spread/interp）与裁判重建。第三方可执行参考不存在 ⇒ **被确立的是可复现性，不是保真度**。

## 5. 误差判定（关键量 vs 基准，逐项）

| 关键量 | ref | got | 相对误差 | <15%？ | 判 |
|---|---|---|---|---|---|
| δ 1-D 分割单位（解析） | 1 | 1（±1.11e-16） | 1.1e-16 | ✅ | 解析恒等 |
| spread→interp 往返 A1 N16 常量 | 0 | 1.47e-16 | — | ✅（且按构造） | 构造性恒等（F3） |
| spread→interp 往返 A1 N32 径向 | 0 | 2.19e-3 | 2.19e-3 | ✅ | 唯一非平凡恒等行 |
| 往返 A2（控制，同核 ε=1）N16 / N32 | 0 | 2.455e-2 / 4.600e-1 | 2.5% / 46% | ❌（预期失败，暴露缺陷） | 控制组 |
| 沉积总力守恒 A1 N16 / N32 | 0 | 2.517e-2 / 8.522e-1 | 2.5% / **85%** | ❌ / ❌ | 修正代价（F1 触发 H0） |
| Cd(Re=100) A1 / A2 / A3 | 1.35（带 1.30–1.40） | 1.6156 / 1.6303 / 1.6817 | **+19.7% / +20.8% / +24.6%** | ❌ | **honest FAIL**（>15%，归因未定，F4） |

**显式回答：是否 <3%？** 恒等/解析类 KPI 全部 ≪3%（机器精度至 0.22%）；**物理 KPI `Cd` 不满足**（+19.7%，>3% 且 >15%），交付已如实判 FAIL——审计确认该 FAIL 成立，并把其**归因**（阻塞比假设）标记为被平均口径混淆的弱证据。

## 6. 处置与落表

**approve with notes**：核心能力真实、可复现、接入仓内 IBM 力传递机制 ⇒ **L3**。L4 扣留理由如上；**F1、F4 作为 open item 入账**，物理臂保持 honest-FAIL 不得洗成 pass。

落表：`tensor_maturity_verdicts` id=**722**，`repo='TensorLBM'`，`scope='m18141 XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION'`，`mission_id=18209`，`level=3`，`verdict_by='board-audit'`，`gate_state='pending'`（等 maturity_gate 自裁：confirmed=消账 / reopened=返修）。

**审计工件**：`audit_m18209/(verify_m18141.py, verify_m18141.json, build_verdict.py, AUDIT_M18141.md, fig_audit_m18141.png)`、`verdict_m18209.sql`。
"""

body = {
    "title": "交差审计 m18141 XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION（TensorLBM）：L3 确认，L4 扣留（预声明 H1 被自家证伪器打掉 + 物理 Cd +19.7%）",
    "summary": "board-audit 复核 m18141：三件套齐全、benchmark 25 行经审计独立实现逐位复现（4/4）、Part A 8/8 与 sweep 点完全可复现、pytest 12/12；但平台裁判表该 anchor 0 行、预声明 H1（往返且守恒<1e-6）因沉积守恒 2.5e-2/8.5e-1 不成立且触发 H0 的 15% 证伪器而报告未裁决（事后换判据），物理 Cd 1.6156 vs 1.35 = +19.7% honest-FAIL（阻塞比归因被平均口径混淆）。裁定 L3，L4 扣留，已落表 id=722。",
    "scenario_key": "lbm",
    "mission_id": 18209,
    "tags": ["board-audit", "XVAL", "LBM", "IBM", "epsilon-gram", "maturity-verdict", "L3", "honest-FAIL"],
    "markdown": md,
    "figures": [{
        "name": "fig_audit_m18141.png",
        "caption": "(a) benchmark.json 逐行 measured rel-error vs 预声明容差（绿 pass / 红 fail）；(b) 预声明 H1 门槛（往返且守恒 <1e-6）与审计重算：往返 1.3e-16 过线，总力守恒 2.5e-2（N16）/8.5e-1（N32）越线并触发 H0 的 15% 证伪器 ⇒ H1 不成立。",
        "png_base64": base64.b64encode(open(png, "rb").read()).decode(),
    }],
}


def post(path, obj, token=None):
    hdr = {"Content-Type": "application/json"}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(API + path, data=json.dumps(obj).encode(), headers=hdr, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read().decode())


tok = None
for p in ("/auth/login", "/tensor/auth/login"):
    for payload in ({"username": USER, "password": PASS}, {"user": USER, "password": PASS},
                    {"agent_user": USER, "agent_pass": PASS}):
        try:
            s, d = post(p, payload)
            tok = d.get("access_token") or d.get("token") or (d.get("data") or {}).get("access_token")
            if tok:
                print("login ok via", p)
                break
        except Exception as e:
            print("login fail", p, e)
    if tok:
        break
if not tok:
    raise SystemExit("login failed")
s, d = post("/tensor/reports", body, tok)
print("report post:", s, json.dumps(d)[:400])