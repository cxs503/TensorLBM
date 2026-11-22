#!/usr/bin/env python3
"""Publish K1-CAVCLOS-R2 report (login /auth/login -> POST /tensor/reports)."""
import base64
import json
import os
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
D = "/root/tp-agent/work/m27075/docs/audits/k1_cavclos_r2_20261122"


def call(method, path, payload=None, token=None):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


md = open(D + "/report.md").read()
report = {
    "title": "K1 空穴过闭合抑制处方满窗定谳：方向相反（减深 0.0705D）+ 非单调稳定前沿（honest-FAIL，K1 轴被否定）",
    "summary": "K1 空穴过闭合臂重开（前作 m26737 只交 step-100 烟窗即报成功，预登记从未检验）。本单 --steps 10000 满窗三档剂量：主臂 α=0.5（cj6093）峰 1.364683D@10.348768ms pc，-26.53% vs 实验 1.857434D，health_limit 16.052ms pc——比 ctrl 1.435220D 还浅 0.070537D，预登记带 [1.50,1.70]D 未命中，方向与处方预期相反；阶梯 α=0.15（cj6095）满窗 1.374095D、α=0.3（cj6094）1.094ms pc 早死（censored）。稳定前沿非单调（0.15 活/0.3 死/0.5 半窗死）。K1 轴独立价值=负，与 CONT2-CONT-RIM 动量瓶颈判别一致，后续组合波不应再叠加 K1 处方。",
    "scenario_key": "DE-X-K1-CAVCLOS-R2",
    "mission_id": 27075,
    "tags": ["TensorLBM", "water-entry", "cavity-closure", "free-surface-LBM", "honest-FAIL", "falsified-hypothesis", "DE-campaign"],
    "markdown": md,
    "figures": [
        {"name": "fig_depth_curve.png",
         "caption": "深度-时刻曲线（pc）：模型 α=0.5 vs 实验 18 测点 + ctrl 锚 + 预登记带 + 判决点线",
         "png_base64": b64(D + "/fig_depth_curve.png")},
        {"name": "fig_field_planes.png",
         "caption": "中面 fill 云图 @ 4 注册 post-contact 时刻（1.396/6.4/10.394/13.4 ms pc）",
         "png_base64": b64(D + "/fig_field_planes.png")},
        {"name": "fig_dose_response.png",
         "caption": "剂量-响应（峰深 vs α）与稳定前沿（存活 vs α，非单调）",
         "png_base64": b64(D + "/fig_dose_response.png")},
    ],
}

st, body = call("POST", "/auth/login",
                {"username": os.environ["TENSOR_AGENT_USER"],
                 "password": os.environ["TENSOR_AGENT_PASS"]})
print("login:", st, body[:120])
if st != 200:
    raise SystemExit(1)
tok = json.loads(body).get("access_token", "")
st2, body2 = call("POST", "/tensor/reports", report, tok)
print("report:", st2, body2[:300])
raise SystemExit(0 if 200 <= st2 < 300 else 1)
