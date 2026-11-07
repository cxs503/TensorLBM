#!/usr/bin/env python3
"""Post the P16-N7 report (markdown + figures) to the platform report API.

Payload is kept well below the 250 KB limit that silently fails: figures are
downscaled and JPEG-compressed before base64 encoding.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request

from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(HERE, "docs", "P16_N7_LNG_WIND_LOAD.md")
FIGDIR = os.path.join(HERE, "figures", "p16n7")
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")
MISSION = int(os.environ.get("REPORT_MISSION_ID", "5190"))

FIGURES = [
    ("fig1_CX_vs_angle.png", "C_X(theta): LBM vs platform empirical model and "
                             "engineering drag build-up"),
    ("fig2_CY_CN_CK_vs_angle.png", "Transverse force, yaw and heel moment "
                                   "coefficients vs wind angle"),
    ("fig3_force_spectrum.png", "Beam-wind lateral force time history and its "
                                "spectrum with the detected Strouhal number"),
    ("fig4_grid_convergence.png", "Grid convergence of C_X(0 deg) with Roache "
                                  "GCI band"),
]


def post(url: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    hdr = {"Content-Type": "application/json"}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=hdr, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
            try:
                return {"status": resp.status, "body": json.loads(raw)}
            except json.JSONDecodeError:
                return {"status": resp.status, "body": raw[:600]}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "body": exc.read().decode()[:600]}
    except Exception as exc:                                  # noqa: BLE001
        return {"status": None, "error": str(exc)}


def encode_figure(path: str, max_width: int = 720) -> str:
    """Downscale and quantise to a palette PNG (the API wants real PNG bytes)."""
    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)),
                         Image.LANCZOS)
    img = img.convert("P", palette=Image.ADAPTIVE, colors=48)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    md = open(DOC).read()
    login = post(f"{BASE}/login", {"username": USER, "password": PASS})
    if login.get("status") != 200:
        print("login failed", login)
        return 1
    body = login["body"] if isinstance(login["body"], dict) else {}
    token = body.get("access_token") or body.get("token")
    if not token:
        print("no token in login response", login)
        return 1

    figs = []
    budget = 185_000          # keep the whole payload well under 250 KB
    used = len(md)
    for name, caption in FIGURES:
        p = os.path.join(FIGDIR, name)
        if not os.path.exists(p):
            continue
        b64 = encode_figure(p)
        if used + len(b64) > budget:
            continue
        used += len(b64)
        figs.append({"name": name, "caption": caption, "png_base64": b64})
    payload = {
        "title": "P16-N7 LNG 船 Moss 型上层建筑风阻力 LBM 预报与 ship-platform 集成",
        "summary": ("TensorLBM D3Q19+Smagorinsky+壁面函数对 LNG Moss 上层建筑 7 个风向角"
                    "做外流预报，产出 C_X/C_Y/C_N/C_K 系数表（schema v1）并与平台经验模型、"
                    "工程叠加基准、立方体标定交叉验证；GCI/质量收支/Strouhal/pytest 判据逐项"
                    "给出 honest 判定，风洞对标因离线无数据未完成。"),
        "scenario_key": "lng_moss_superstructure_wind_load",
        "mission_id": MISSION,
        "stage": "S3",
        "tags": ["tensorlbm", "wind-load", "lng", "lbm-les", "ship-platform", "p16-n7"],
        "markdown": md,
        "figures": figs,
    }
    size = len(json.dumps(payload))
    res = post(f"{BASE}/tensor/reports", payload, token)
    print(json.dumps({"payload_bytes": size, "figures": len(figs),
                      "status": res.get("status"),
                      "response": str(res.get("body"))[:400]}, ensure_ascii=False)[:1200])
    return 0 if res.get("status") in (200, 201) else 1


if __name__ == "__main__":
    raise SystemExit(main())