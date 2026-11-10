#!/usr/bin/env python3
"""Publish the p339w2-n06 report to the TensorPlatform reports API.

Reads ``result.json`` + the generated markdown + the four PNG figures and posts
a single paper-grade report (six required elements, figures embedded as
base64).  Credentials come from ``TENSOR_AGENT_USER`` / ``TENSOR_AGENT_PASS``;
the base URL from ``TENSOR_API_BASE``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

CAPTIONS = {
    "p339w2_n06_fig1_kozeny_carman.png":
        "图 1 G1：K/d_p^2–孔隙率，解析几何 LBM 点 vs Kozeny-Carman(C=180) 与 150 常数版本",
    "p339w2_n06_fig2_ergun.png":
        "图 2 G2：两档孔隙 Re 下压降的 LBM 动量平衡值与 Ergun 粘性/惯性分解",
    "p339w2_n06_fig3_interface.png":
        "图 3 G3/G4/G5：压力/剪切驱动口剖面、口径 B 盲预测、床内 Brinkman 衰减拟合",
    "p339w2_n06_fig4_geometry.png":
        "图 4 G1 几何审计：数字化随机球堆积三正交切片",
}


def login(base: str) -> str:
    user = os.environ["TENSOR_AGENT_USER"]
    pwd = os.environ["TENSOR_AGENT_PASS"]
    for path, payload in (
        ("/auth/login", {"username": user, "password": pwd}),
        ("/auth/token", {"username": user, "password": pwd}),
        ("/login", {"username": user, "password": pwd}),
    ):
        try:
            r = requests.post(base + path, json=payload, timeout=20)
        except Exception as exc:  # pragma: no cover
            print(f"login {path}: {exc}", file=sys.stderr)
            continue
        if r.ok:
            data = r.json()
            for key in ("access_token", "token", "jwt"):
                if isinstance(data, dict) and key in data:
                    print(f"login via {path}", file=sys.stderr)
                    return data[key]
            if isinstance(data, dict) and "data" in data:
                for key in ("access_token", "token"):
                    if key in data["data"]:
                        print(f"login via {path}", file=sys.stderr)
                        return data["data"][key]
        print(f"login {path}: {r.status_code} {r.text[:120]}", file=sys.stderr)
    raise SystemExit("could not obtain a token")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(HERE / "result.json"))
    ap.add_argument("--markdown", default=str(REPO / "docs" / "p339w2_n06_porous_ladder.md"))
    ap.add_argument("--figdir", default=str(REPO / "docs" / "figures"))
    ap.add_argument("--mission-id", type=int, default=10909)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    res = json.loads(Path(args.result).read_text())
    md = Path(args.markdown).read_text()
    figs = []
    for name, cap in CAPTIONS.items():
        p = Path(args.figdir) / name
        if p.exists():
            figs.append({
                "name": name,
                "caption": cap,
                "png_base64": base64.b64encode(p.read_bytes()).decode(),
            })
    gates = res["gates"]
    summary = (f"TensorLBM 中低孔隙率多孔介质渗流验证阶梯：{res['n_pass']}/5 门通过 "
               + "; ".join(f"{g['gate'].split()[0]}={100*g['error']:+.1f}%"
                           f"({'PASS' if g['pass'] else 'FAIL'})" for g in gates))
    payload = {
        "title": "P339W2-N06 TensorLBM 中低孔隙率多孔介质渗流验证阶梯"
                 "（Kozeny-Carman / Ergun / Beavers-Joseph / Brinkman）",
        "summary": summary,
        "scenario_key": "p339w2-n06-lbm-porous-media",
        "mission_id": args.mission_id,
        "tags": ["TensorLBM", "porous-media", "Kozeny-Carman", "Ergun",
                 "Beavers-Joseph", "Brinkman", "verification-ladder", "p339w2"],
        "markdown": md,
        "figures": figs,
    }
    if args.dry_run:
        print(json.dumps({k: (v if k != "markdown" else v[:200] + "...")
                          for k, v in payload.items() if k != "figures"},
                         ensure_ascii=False, indent=1)[:1500])
        print(f"figures: {[f['name'] for f in figs]}")
        return
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    token = login(base)
    r = requests.post(f"{base}/tensor/reports", json=payload, timeout=120,
                      headers={"Authorization": f"Bearer {token}"})
    print(r.status_code, r.text[:400])
    r.raise_for_status()


if __name__ == "__main__":
    main()