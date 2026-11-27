#!/usr/bin/env python3
"""Publish the p83w2-n01 report to the platform report API.

Builds the payload from the campaign artefacts (``verification.json``,
``figures.json``, the generated markdown) and posts it; no number is typed in
by hand and no secret is written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path


def _login(base: str, user: str, password: str) -> str:
    req = urllib.request.Request(
        f"{base}/auth/login",
        data=json.dumps({"username": user, "password": password}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    return d.get("access_token") or d.get("token")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/p83w2_n01")
    ap.add_argument("--markdown", default="docs/P83W2_N01_MA_ERROR_LAW.md")
    ap.add_argument("--mission-id", type=int, default=9255)
    ap.add_argument("--scenario-key", default="lbm_ma_error_law_cylinder")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    ver = json.loads((root / "verification.json").read_text())
    md = Path(args.markdown).read_text()
    figs = []
    fjson = root / "figures.json"
    if fjson.exists():
        names = json.loads(fjson.read_text())
        for f in names:
            png = root / f["name"]
            if png.exists():
                import base64
                figs.append({
                    "name": f["name"],
                    "caption": f["caption"],
                    "png_base64": base64.b64encode(png.read_bytes()).decode(),
                })
    crit = ver["criteria"]
    passed = [k for k in ("a", "b", "c", "d", "e") if crit[k]["pass"]]
    failed = crit["f"]["failing"]
    summary = (
        "圆柱绕流强制对流传热（Pr=0.7）LBM 可压缩性误差律 |Nu(Ma)/Nu0-1|=C·Ma^n 与 Ma 运行包络："
        f"判据通过 {','.join(passed) or '无'}"
        + (f"；未通过 {','.join(failed)}（honest FAIL，不粉饰）" if failed else "")
    )
    payload = {
        "title": "p83w2-n01 LBM 壁面传热可压缩性误差律与 Ma 运行包络（TensorLBM）",
        "summary": summary,
        "scenario_key": args.scenario_key,
        "mission_id": args.mission_id,
        "tags": ["TensorLBM", "compressibility", "Mach-envelope", "heat-transfer",
                 "cylinder", "error-law", "p83w2-n01"],
        "markdown": md,
        "figures": figs,
    }
    if args.dry_run:
        Path("/tmp/report_p83w2_n01.json").write_text(json.dumps(payload))
        print(f"dry run: {len(figs)} figures, markdown {len(md)} chars -> "
              f"/tmp/report_p83w2_n01.json")
        return
    base = os.environ["TENSOR_API_BASE"]
    token = _login(base, os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        path = fh.name
    out = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{base}/tensor/reports",
         "-H", f"Authorization: Bearer {token}",
         "-H", "Content-Type: application/json", "-d", f"@{path}"],
        capture_output=True, text=True, timeout=180,
    )
    print(out.stdout[:1000] or out.stderr[:1000])
    assert out.returncode == 0, out.returncode
    Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
