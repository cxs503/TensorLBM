#!/usr/bin/env python3
"""Publish the DE-X-CAVSWEEP paper-level progress report (owner 2026-10-29 spec)."""
import json, os, base64, urllib.request

API = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
DATA = "/root/tp-agent/work/m26392/runs/cavsweep"

def post(path, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(API + path, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read().decode()

st, body = post("/auth/login", {"username": USER, "password": PASS})
print("login:", st, body[:200])
j = json.loads(body)
tok = j.get("access_token") or j.get("token")

def arm(a):
    d = json.load(open(os.path.join(DATA, a + ".json")))
    h = [x for x in d["history"] if x.get("depth_over_D") is not None]
    return d, max(h, key=lambda x: x["depth_over_D"])

EXP = 1.857434
order = ["ctrl","ul006","ul010","ul020","ul024","rs135","rs175","gap12","pd20",
         "caponly","simonly","nearest","spd005","spd007","ul012pd20","rs135ul020"]
rows = []
for a in order:
    d, pk = arm(a)
    rows.append((a, d["stop_reason"], d["history"][-1]["time_ms"],
                 pk["depth_over_D"], pk["time_ms"], d["contact_step"],
                 max(x["mach_max"] for x in d["history"]),
                 d["history"][-1]["volume_drift"]))

tbl = ["| 臂 | stop | t_end/ms | peak/D | peak_t/ms | contact | mach_max | drift_end |",
       "|---|---|---|---|---|---|---|---|"]
for a, stop, te, pk, pt, cs, mm, de in rows:
    tbl.append("| %s | %s | %.3f | %.4f | %.3f | %s | %.4f | %.2e |"
               % (a, stop, te, pk, pt, "—" if cs is None else cs, mm, de))
table = "\n".join(tbl)

def fig(name, cap):
    b = base64.b64encode(open(os.path.join(DATA, name), "rb").read()).decode()
    return {"name": name, "caption": cap, "png_base64": b}

figures = [
    fig("fig1_ulimit.png", "u_limit 敏感度：峰深 0.10–0.24 平坦 1.4352D（纯稳定器绿灯）；ul006 死于 drift 门（下界）；ul020/ul024 死于 Ma 门（复现原始 cal8）"),
    fig("fig2_depth_time.png", "腔深-时间曲线 vs Santini fig.5 实验点：全臂 ≤1.4352D，实验峰 1.8574D 未触及"),
    fig("fig3_drift.png", "volume_drift 轨迹：9 臂死于 1e-4 门（gap12 初始化 2.8e-3；caponly/simonly 证交换模式必要性）"),
]

e = {a: (pk - EXP) / EXP * 100 for a, _, _, pk, _, _, _, _ in rows}

markdown = f"""# DE-X-CAVSWEEP · 16 臂气穴驱动敏感度扫描 · 论文级进展报告

## ① 案例介绍
单滴入深水池（Santini case-3）：直径 D 液滴自 gap=0.18D 以 release-speed 1.53 m/s 释放，
入深水池（pool 3D，span 4D）。LBM 自由面（D3Q19，cells=8/D，lattice speed 0.06，
capacity+simple-exchange 质量守恒组合）。实验基准：Santini fig.5 侧视轮廓，峰深 1.8574D @10.394ms。
精度缺口定谳（上游）：ctrl 峰深 1.4352D，误差 -22.73%（恒定项=气穴欠深）；ABB 动量注入（DE-X-ABBS）
与网格 8→12（仅 0.6pp）已排除。本单补 T4 自认缺口：u_limit=2.5·speed 经验值未扫描。

## ② 怎么做的
- 16 臂预注册（2026-11-20）：base=--gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53；
  变量=u_limit{0.06,0.10,0.20,0.24}、release-speed{1.35,1.75}、gap 0.12、pool-depth 2.0、
  关 simple-exchange、关 capacity、nearest-surface、lattice speed{0.05,0.07}、UL0.12+PD2.0、UL0.20 重复臂。
- 协调者 node099 太初卡 0-15 全窗并行（cells=8 span4）；本单回收 16/16 原始 JSON（md5 对账过）+ 判读 + 写卡。
- 工具链：free_surface_probe.py（健康门：Ma>0.3 / rho≤0 / volume_drift>1e-4）；cavsweep_extract.py 程序化提取；
  cavsweep_figures.py 绘图。判读判据预注册：u_limit 单调补偿=红灯 / 平坦=纯稳定器绿灯。

## ③ 计算结果（程序化注入，零手抄）
{table}

## ④ 结果图
见 figures（u_limit 敏感度 / 腔深-时间 vs 实验 / drift 轨迹）。

## ⑤ 注意点
- 12/16 臂死健康门（9 臂 drift>1e-4、3 臂 Ma>0.3）：gap12 初始化即死（drift 2.8e-3，28×门）；
  pd20/rs175/spd007/ul012pd20 死于 step3（~1e-4）；caponly 1.301ms、simonly 3.174ms（交换模式=质量守恒必要条件，实证）；
  ul006 9.315ms（早于实验峰时 10.394ms，其峰 1.2942D 为下界）；ul020/ul024/rs135ul020 死 Ma 门（0.3006，精确复现原始 cal8）。
- 峰为平台（9.95–11.70ms 内 145 步在峰 0.02D 内）：argmax 时刻数值脆弱（T4 修复臂报 10.543ms，本表 11.091ms，同平台）。
- 确定性：rs135ul020≡ul020、ul012pd20≡pd20、nearest≡ctrl（逐位同）。
- 适用范围：cells=8 单网格、Santini case-3 单工况；非收敛研究。

## ⑥ 误差判定（关键量 vs 基准，逐项）
| 关键量 | 模拟 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| 峰深（最深 6 臂 ctrl/ul010/ul020/ul024/nearest/rs135ul020） | 1.4352D | 1.8574D | -22.73% | **否** |
| 峰深（rs135） | 1.3530D | 1.8574D | -27.16% | **否** |
| 峰深（spd005） | 1.3310D | 1.8574D | -28.34% | **否** |
| 峰深（ul006，下界） | 1.2942D | 1.8574D | ≤-30.33% | **否** |
| 重复性（rs135ul020 vs ul020） | 逐位同 | — | 0 | 是（确定性） |

**honest-FAIL**：0/16 臂过 3% 门；numeric_pass=False。
**定谳**：u_limit=纯稳定器（绿灯）；**无旋钮把峰深推向 1.8574D**——22.7% 缺口=系统性气穴欠深
（T5 13–23% 偏差同源），非参数可调；方向应转内核气穴深度物理（ABB 重构/界面动力学），参数细扫预期增益 <1pp。
"""

payload = {
    "title": "DE-X-CAVSWEEP: 16-arm cavitation-driven sensitivity sweep — u_limit is a pure stabilizer; no knob closes the 22.7% cavity-depth gap",
    "summary": "16/16 臂回收定谳：u_limit 0.10–0.24 峰深平坦 1.4352D（纯稳定器绿灯）；无旋钮推向实验 1.8574D；22.73% 缺口=系统性气穴欠深（与 ABBS 裁决一致）；12/16 臂死健康门如实入账，0/16 过 3% 门。",
    "scenario_key": "DE-X-CAVSWEEP",
    "mission_id": 26392,
    "tags": ["DE", "water-entry", "cavitation", "sensitivity-sweep", "honest-fail", "TensorLBM"],
    "markdown": markdown,
    "figures": figures,
}
st, body = post("/tensor/reports", payload, tok)
print("report:", st, body[:300])
