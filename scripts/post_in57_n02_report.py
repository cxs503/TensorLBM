"""Post the IN57-N02 progress report to the platform (figures downscaled < 60 kB)."""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def login() -> str:
    req = urllib.request.Request(
        BASE + "/auth/login",
        data=json.dumps({"username": USER, "password": PASS}).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=20))["access_token"]


def shrink(path: Path, max_w: int = 900, budget: int = 55000) -> str | None:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)), Image.LANCZOS)
    for _ in range(8):
        for quant in (256, 128, 64):
            buf = io.BytesIO()
            img.convert("P", palette=Image.ADAPTIVE, colors=min(quant, 256)).save(
                buf, format="PNG", optimize=True
            )
            data = buf.getvalue()
            if len(data) <= budget:
                return base64.b64encode(data).decode()
        img = img.resize((int(img.width * 0.75), int(img.height * 0.75)), Image.LANCZOS)
    return None


def main() -> None:
    doc = Path("docs/IN57_N02_BUBBLE_DR.md")
    markdown = doc.read_text()
    figdir = Path("artifacts/bubble_dr_in57_n02/figures")
    campaign = json.loads(Path("artifacts/bubble_dr_in57_n02/campaign.json").read_text())
    gates = campaign["gates"]
    npass = sum(1 for v in gates.values() if v["pass"])
    figs = []
    if figdir.exists():
        for p in sorted(figdir.glob("*.png")):
            b64 = shrink(p)
            if b64:
                figs.append({"name": p.name, "caption": p.stem.replace("_", " "), "png_base64": b64})
    payload = {
        "title": "IN57-N02 船底气层/微气泡减阻 LBM 两相仿真：覆盖率–减阻率–净功率闭包",
        "summary": (
            f"TensorLBM 新增 tensorlbm.bubble_dr（3D 单气泡 HR 对照 + 2D 顶壁气层通道 + 压缩机账本）；"
            f"判据 {npass}/{len(gates)} 通过；DR 最优 {max(r['DR'] for r in campaign['curtain_records']):.3f}，"
            f"覆盖率最高 {max(r['coverage'] for r in campaign['curtain_records']):.3f}；"
            "单气泡 3D 判据 honest FAIL（SC 模型气泡在液相中溶解，见报告）。"
        ),
        "scenario_key": "ship_performance",
        "mission_id": 6231,
        "tags": ["IN57-N02", "TensorLBM", "bubble-drag-reduction", "air-layer", "Shan-Chen", "two-phase"],
        "markdown": markdown,
        "figures": figs,
    }
    tok = login()
    req = urllib.request.Request(
        BASE + "/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok},
    )
    try:
        r = urllib.request.urlopen(req, timeout=120)
        print(r.status, r.read()[:300])
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read()[:500])


if __name__ == "__main__":
    main()