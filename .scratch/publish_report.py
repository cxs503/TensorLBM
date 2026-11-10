"""Publish the P218-N06 report to the TensorPlatform report API.

Reads credentials from the environment (never embeds them), builds the payload
from the generated markdown + PNG figures, and POSTs it.

    python .scratch/publish_report.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
MD = "docs/P218-N06_rarefied_slip_ladder.md"
OUTDIR = "runs/P218-N06_ladder"


def post(path, payload, token=None, timeout=60):
    url = f"{BASE}/{path.lstrip('/')}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return fh.status, fh.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:2000]
    except Exception as e:  # pragma: no cover
        return -1, repr(e)


def login():
    creds = {"username": USER, "password": PASS, "user": USER, "pass": PASS}
    for path in ("auth/login", "tensor/auth/login", "login"):
        st, body = post(path, creds, timeout=20)
        if st == 200:
            try:
                j = json.loads(body)
            except Exception:
                continue
            for k in ("access_token", "token", "data"):
                v = j.get(k) if isinstance(j, dict) else None
                if isinstance(v, str):
                    print("login ok via", path)
                    return v
                if isinstance(v, dict):
                    for kk in ("access_token", "token"):
                        if isinstance(v.get(kk), str):
                            print("login ok via", path)
                            return v[kk]
        print(f"  login try {path}: {st} {body[:120]}")
    return None


def main():
    token = None
    for attempt in range(12):
        token = login()
        if token:
            break
        print(f"retry {attempt + 1}/12 in 10s (API may be restarting)")
        time.sleep(10)
    if not token:
        print("no token; aborting")
        return 1
    with open(MD, encoding="utf-8") as fh:
        md = fh.read()
    figs = []
    for name, cap in (
        ("fig1_couette_profiles.png", "图1 不同 Kn 下 Couette 速度剖面（DVM, σ=1）"),
        ("fig2_slip_length.png", "图2 滑移长度 vs TMAC 与 Maxwell 定律"),
        ("fig3_flow_rate.png", "图3 归一化质量流率 G=Q/Q_FM 与 Knudsen 极小值"),
        ("fig4_knudsen_layer.png", "图4 Kn=0.05 泊肃叶剖面 vs NS+滑移与 Knudsen 层偏差"),
        ("fig5_fm_degeneration.png", "图5 Couette 逼近自由分子塞流的偏差 vs Kn"),
    ):
        p = os.path.join(OUTDIR, "figures", name)
        with open(p, "rb") as fh:
            figs.append({"name": name, "caption": cap,
                         "png_base64": base64.b64encode(fh.read()).decode()})
    payload = {
        "title": "P218-N06 TensorLBM 稀薄气体/滑移流验证阶梯：Maxwell 滑移壁 · Knudsen 层 · 跨流态 Q(Kn)（C2/C3/C4 PASS，C1/C5 honest-FAIL）",
        "summary": ("建立线性化 BGK 离散速度求解器 + 独立 DSMC 双求解器稀薄气体阶梯："
                    "C2 Knudsen 层 δ/λ=0.935（-6.5%）、C3 六档 Q(Kn) 与 DSMC 差 ≤7.4% 且 Kn_min=1.05∈[0.5,2]、"
                    "C4 Kn=0.001 剖面与连续流差 0.4%；C1（Maxwell 一阶滑移公式对漫反射壁低 ~8%，实测 l_s/λ=1.084）"
                    "与 C5（Kn=1 属过渡流，剖面偏离 FM 塞流 47%，不可达）如实报 FAIL。"),
        "scenario_key": "P218-N06",
        "mission_id": 9041,
        "tags": ["TensorLBM", "rarefied-gas", "slip-flow", "Knudsen-layer", "DSMC",
                 "BGK", "P218-N06", "resistance"],
        "markdown": md,
        "figures": figs,
    }
    st, body = -1, ""
    for attempt in range(6):
        st, body = post("tensor/reports", payload, token=token, timeout=180)
        print(f"  post attempt {attempt + 1}: {st}")
        if st in (200, 201):
            break
        time.sleep(10)
    print("report POST ->", st, body[:600])
    with open("/tmp/report_publish_result.json", "w") as fh:
        json.dump({"status": st, "body": body, "ts": time.time()}, fh)
    return 0 if st in (200, 201) else 1


if __name__ == "__main__":
    sys.exit(main())