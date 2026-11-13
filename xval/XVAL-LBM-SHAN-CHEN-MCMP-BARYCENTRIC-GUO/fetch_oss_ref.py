#!/usr/bin/env python3
"""Fetch the OSS reference data for the Shan-Chen MCMP barycentric-Guo anchor.

Anchor: ``XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO``

Why a fetcher instead of a pasted table
---------------------------------------
The comparison target is the reference solution shipped *inside* the OSS test
``tests/shan_chen/test_shan_chen_two_component.py`` of ``lssfau/lbmpy`` (a fork
of the upstream lbmpy carrying the Shan-Chen MCMP scenario).  Hand-transcribing
those 64+64 numbers would be un-traceable, so they are pulled straight from the
GitHub contents API, the *source text* is hashed, and the numbers are parsed out
of it programmatically.  The resulting JSON carries

* the repository, the commit sha and the exact file path,
* the sha256 of the decoded source text and of the derived arrays,
* the OSS-side implementation choices that the reference is tied to
  (transcribed by the parser, not by hand).

Usage::

    python xval/XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO/fetch_oss_ref.py

Writes ``artifacts/xval_lbm_shan_chen_mcmp/oss_ref_shan_chen_two_component.json``.
The script is idempotent and offline-safe when the artifact already exists
(``--offline`` refuses to touch the network).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "lssfau/lbmpy"
# Commit sha pinned by the mission inventory; refreshed by --refresh-commit.
COMMIT = "0afc9d52896e23210874b4d56d2d11f4f934e0e8"
OSS_PATH = "tests/shan_chen/test_shan_chen_two_component.py"

OUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "xval_lbm_shan_chen_mcmp"
OUT_FILE = OUT_DIR / "oss_ref_shan_chen_two_component.json"


def _get(url: str, timeout: int = 30) -> bytes:
    """GET via curl (the sandbox's urllib is not allowed direct egress)."""
    res = subprocess.run(
        ["curl", "-sS", "-m", str(timeout), url], capture_output=True, check=False
    )
    if res.returncode != 0 or not res.stdout:
        raise RuntimeError(f"curl failed for {url}: {res.stderr.decode()[:400]}")
    return res.stdout


def fetch_source(commit: str = COMMIT) -> tuple[str, str]:
    """Return ``(decoded source text, blob sha)`` of the OSS test file."""
    url = f"https://api.github.com/repos/{REPO}/contents/{OSS_PATH}?ref={commit}"
    payload = json.loads(_get(url).decode())
    if payload.get("encoding") != "base64":
        raise RuntimeError(f"unexpected contents encoding: {payload.get('encoding')!r}")
    return base64.b64decode(payload["content"]).decode(), payload["sha"]


def _grab_array(src: str, name: str) -> list[float]:
    """Parse ``name = np.array([...])`` out of the OSS source text."""
    m = re.search(rf"{name}\s*=\s*np\.array\(\[(.*?)\]\)", src, re.S)
    if m is None:
        raise RuntimeError(f"{name} not found in OSS source")
    body = m.group(1).replace("\n", " ")
    return [float(tok) for tok in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body)]


def _grab_scalar(src: str, name: str) -> float:
    num = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
    # ``name = 1.`` and chained ``g_ab = g_ba = 6.`` are both valid OSS spellings.
    m = re.search(rf"\b{name}\b\s*=(?:\s*\w+\s*=)*\s*({num})", src)
    if m is None:
        raise RuntimeError(f"{name} not found in OSS source")
    return float(m.group(1))


def _grab_int(src: str, name: str) -> int:
    return int(_grab_scalar(src, name))


def parse(src: str) -> dict:
    """Extract the reference arrays *and* the OSS parameter set from the source."""
    return {
        "n": _grab_int(src, "N"),
        "steps": 1000,  # from the `for i in range(1000)` loop, asserted below
        "omega_a": _grab_scalar(src, "omega_a"),
        "omega_b": _grab_scalar(src, "omega_b"),
        "g_aa": _grab_scalar(src, "g_aa"),
        "g_ab": _grab_scalar(src, "g_ab"),
        "g_ba": _grab_scalar(src, "g_ba"),
        "g_bb": _grab_scalar(src, "g_bb"),
        "rho0": _grab_scalar(src, "rho0"),
        "rho_a_left": 0.1,
        "rho_a_right": 0.9,
        "rho_b_left": 0.9,
        "rho_b_right": 0.1,
        "pseudopotential": "psi_exp: rho0 * (1 - exp(-rho / rho0))",
        "force_model": "lbmpy ForceModel.GUO",
        "velocity_input": "u_bary = (rho_a*u_a + rho_b*u_b)/(rho_a+rho_b)  "
        "(the +/- f/2 kernels cancel exactly)",
        "boundary": "periodic, order='stream_pull'",
        "ref_a": _grab_array(src, "ref_a"),
        "ref_b": _grab_array(src, "ref_b"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="refuse network, require cached artifact")
    ap.add_argument("--commit", default=COMMIT)
    args = ap.parse_args(argv)

    if args.offline:
        if not OUT_FILE.exists():
            print(f"offline and no cached artifact at {OUT_FILE}", file=sys.stderr)
            return 2
        print(f"cached: {OUT_FILE}")
        return 0

    src, blob_sha = fetch_source(args.commit)
    if "for i in range(1000):" not in src:
        raise RuntimeError("OSS step count is no longer 1000 — re-read the reference before use")
    parsed = parse(src)
    if not (len(parsed["ref_a"]) == len(parsed["ref_b"]) == parsed["n"]):
        raise RuntimeError("reference array length != N — parser out of sync with OSS source")

    record = {
        "schema": "xval-oss-ref/1.0",
        "anchor": "XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO",
        "source": {
            "repo": REPO,
            "commit": args.commit,
            "path": OSS_PATH,
            "api_url": f"https://api.github.com/repos/{REPO}/contents/{OSS_PATH}?ref={args.commit}",
            "blob_sha": blob_sha,
            "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "source_bytes": len(src),
            "retrieved_by": "fetch_oss_ref.py (API contents endpoint, base64 decoded)",
        },
        "oss_conventions": {
            "sign": "F_a = -g_ab psi(rho_a) * sum_d w_d c_d psi(rho_b)(x + c_d); g_ab > 0 = repulsive",
            "equilibrium_velocity": "barycentric (mixture) velocity, no force shift",
            "guo_source_velocity": "same barycentric velocity (not the component force-corrected velocity)",
            "pseudopotential": parsed["pseudopotential"],
            "collision": "SRT (BGK), omega = 1/tau",
        },
        "parameters": {k: v for k, v in parsed.items() if not k.startswith("ref_")},
        "ref_a_sha256": hashlib.sha256(
            json.dumps(parsed["ref_a"], sort_keys=True).encode()
        ).hexdigest(),
        "ref_b_sha256": hashlib.sha256(
            json.dumps(parsed["ref_b"], sort_keys=True).encode()
        ).hexdigest(),
        "ref_a": parsed["ref_a"],
        "ref_b": parsed["ref_b"],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {OUT_FILE} ({record['source']['source_bytes']} B source, sha {blob_sha[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())