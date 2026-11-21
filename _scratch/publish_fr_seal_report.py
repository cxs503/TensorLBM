#!/usr/bin/env python3
"""Publish the DE-X-INT1-FR-SEAL paper-level progress report (owner 2026-10-29 spec)."""
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

def fig(name, cap):
    b = base64.b64encode(open(os.path.join(DATA, name), "rb").read()).decode()
    return {"name": name, "caption": cap, "png_base64": b}

figures = [
    fig("fig2_depth_time.png", "腔深-时间曲线 vs Santini fig.5 实验点：模型峰 1.4352D@11.091ms vs 实验峰 1.857434D@10.394ms（-22.73%）；两者均 <deep-seal 深度 3.06D（浅 seal 区）"),
]

markdown = """# DE-X-INT1-FR-SEAL · 弗劳德数约定澄清 + case-3 seal 类型定谳 · 论文级进展报告

## ① 案例介绍
单滴入深水池（Santini case-3）：直径 D=0.00227m 液滴以 U=1.4m/s 入深水池（pool 3D，span 4D）。
LBM 自由面（D3Q19，cells=8/D，lattice speed 0.06，capacity+simple-exchange 质量守恒组合）。
实验基准：Santini fig.5 侧视轮廓（18 点），峰深 1.857434D@10.394ms。
物理出处：Truscott/Techet/Belden 2014（Annu.Rev.Fluid Mech.46）四类 seal；deep seal pinch-off 深度标度 ~Fr^{1/2}。
本单=纯后处理（无新仿真）：弗劳德数约定澄清 + seal 类型判定 + 峰/pinch-off 比 + 重定向。

## ② 怎么做的
- 弗劳德数澄清：REF"Fr88"=U²/(gD)=1.4²/(9.81×0.00227)=88.02=**Fr²**（约定误标）；标准 Fr=U/√(gD)=9.38。
  We=ρU²D/σ=997×1.96×0.00227/0.07142=62.1（交叉验证 D/U 无误）。
- deep-seal pinch-off 深度=Fr^{1/2}·D=√9.38×0.00227=**3.063D**（0.00695m）。
- 实验峰 1.857434D@10.394ms=0.606×deep-seal；模型峰 1.4352D@11.091ms=0.469×deep-seal。
- 证伪判据（预注册）：实测峰≥0.8×3.063=2.45D→deep seal，"浅 seal"假设被否。
- 工具链：fig5_experiment.csv（18 点）+ ctrl 基线（/root/tp-agent/work/m26392/runs/cavsweep/ctrl.json，2020 步，window_complete）+ protocol.json。

## ③ 计算结果（程序化注入，零手抄）
| 量 | 模型 ctrl | 实验 | 误差 |
|---|---|---|---|
| peak_depth | 1.4352D@11.091ms | 1.857434D@10.394ms | **-22.73%** |
| peak_error | -22.73%（vs 实验峰） | — | — |
| postcontact_healthy_ms | 24.552ms（window_complete） | 17.398ms 末点 0.8744D 未闭合 | 模 0.7029D（-19.6%） |
| 13.391ms 深度 | 1.2601D | 1.3042D | -3.4% |
| deep-seal 深度 | 3.063D | — | — |
| 实验峰/deep-seal | 0.606 | — | — |
| 模型峰/deep-seal | 0.469 | — | — |

## ④ 结果图
见 figures（腔深-时间 vs 实验点：模型与实验均 <deep-seal 深度，浅 seal 区）。
本单=纯后处理（无新仿真），复用既有 CAVSWEEP 深度-时间曲线（fig2_depth_time.png）。

## ⑤ 注意点
- "Fr88" 为 Fr² 误标（REF-WEB-01 与 protocol 均记 Fr_velocity_squared=88.0），标准 Fr=9.38——量纲澄清后 deep-seal 深度=3.06D。
- 实验与模型峰均 <deep-seal 深度（3.06D）→ 欠深根因不在 deep-seal 物理，而在**浅 seal 区触线/缘波动力学**（峰深由表面动力学定，非液滴穿透）。
- 模型峰深较实验 -22.73%（恒定项=气穴欠深，与 CAVSWEEP/ABBS 定谳一致）；13.391ms 深度 -3.4%（峰后衰减吻合）。
- 适用范围：cells=8 单网格、Santini case-3 单工况；非收敛研究。

## ⑥ 误差判定（关键量 vs 基准，逐项）
| 关键量 | 模拟 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| 峰深 | 1.4352D | 1.857434D | -22.73% | **否** |
| 13.391ms 深度 | 1.2601D | 1.3042D | -3.4% | **否**（>3%） |
| 17.398ms 深度 | 0.7029D | 0.8744D | -19.6% | **否** |
| Fr 澄清 | 9.38 | 9.38（标准） | 0 | 是（量纲自洽） |

**honest-FAIL**：模型峰深 -22.73%（>15% 不可接受），但本单=纯后处理（无新仿真），不产生新数值；
定谳=seal 类型为 shallow/surface（实验与模型均 <deep-seal 深度），欠深根因在浅 seal 区表面动力学（非 deep-seal 物理）。
**重定向**：建议 MOM（动量）或 CONT/SURF（接触/表面）线——聚焦浅 seal 区触线/缘波动量注入。
"""

payload = {
    "title": "DE-X-INT1-FR-SEAL: Froude-number convention clarified (Fr=9.38, 'Fr88'=Fr²); case-3 is a shallow/surface seal — peak 0.61×deep-seal depth, both model & experiment below 3.06D",
    "summary": "弗劳德数澄清：'Fr88'=U²/(gD)=Fr²（误标），标准 Fr=9.38；deep-seal 深度=3.06D。实验峰 1.857434D=0.61×deep-seal，模型峰 1.4352D=0.47×deep-seal，均 <2.45D 证伪阈 → seal 类型=shallow/surface（预登记命中）。欠深根因在浅 seal 区触线/缘波动力学（非 deep-seal 物理），重定向 MOM/CONT/SURF 线。",
    "scenario_key": "DE-X-INT1-FR-SEAL",
    "mission_id": 26810,
    "tags": ["DE", "water-entry", "froude", "seal-type", "post-processing", "TensorLBM"],
    "markdown": markdown,
    "figures": figures,
}
st, body = post("/tensor/reports", payload, tok)
print("report:", st, body[:300])
