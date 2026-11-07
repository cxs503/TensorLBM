#!/usr/bin/env python3
"""Integrate the LBM wind-load coefficient table with the ship platform.

What this script does
---------------------
1. Loads ``artifacts/p16n7/wind_load_table.json`` (schema v1).
2. Probes the ship-performance-platform for a dedicated wind-load endpoint
   (``/api/v1/calculations/wind_resistance`` — required by the campaign brief).
   When that endpoint does not exist, it falls back to the live coupled
   environmental-force endpoint ``/api/v1/calculations/environmental-forces``,
   which is where the platform actually computes the wind force, and records
   the finding instead of silently pretending the injection succeeded.
3. Builds the platform request for every table angle with the LBM
   coefficients attached (``wind.coefficients`` + pre-computed ``wind.forces``)
   and compares the platform's empirical wind force with the LBM-based force.
4. Writes ``artifacts/p16n7/platform_integration.json`` with the interface
   contract, the probe result, the comparison table and the honest status.

Usage
-----
    python scripts/inject_wind_table.py [--base http://127.0.0.1:8002/api/v1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

from tensorlbm.wind_load_table import interpolate, loads, to_platform_payload  # noqa: E402

TABLE = os.path.join(HERE, "artifacts", "p16n7", "wind_load_table.json")
OUT = os.path.join(HERE, "artifacts", "p16n7", "platform_integration.json")

SHIP = {
    "L": 290.0, "B": 46.0, "T": 11.5, "D": 26.0, "displacement": 120000.0,
    "Cb": 0.82, "KG": 12.0, "Dp": 8.0, "n_rpm": 80.0, "Ar": 60.0, "Hr": 0.5,
    "V0": 0.0, "A_fw_wet": 2000.0, "A_lw_wet": 6000.0,
}


def post_json(url: str, payload: dict, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": resp.status, "body": json.loads(resp.read() or b"{}")}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "body": exc.read().decode()[:400]}
    except Exception as exc:                                   # noqa: BLE001
        return {"status": None, "error": str(exc)}


def get_json(url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": resp.status, "body": json.loads(resp.read() or b"{}")}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "body": exc.read().decode()[:400]}
    except Exception as exc:                                   # noqa: BLE001
        return {"status": None, "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8002/api/v1")
    ap.add_argument("--wind-speed", type=float, default=15.0)
    args = ap.parse_args()

    table = loads(TABLE)
    ref = table["reference"]
    ship = dict(SHIP, A_fw=ref["A_F_m2"], A_lw=ref["A_L_m2"])

    # --- 1. probe the endpoint named by the campaign brief -----------------
    #  (a) direct POST; the sandbox may block outbound POSTs, in which case
    #  (b) the OpenAPI route table is used to prove the endpoint's absence
    probe = post_json(
        f"{args.base}/calculations/wind_resistance",
        {"loa": ship["L"], "beam": ship["B"], "draft": ship["T"],
         "wind_speed": args.wind_speed, "wind_angle_deg": 30.0},
    )
    route_probe = None
    if probe.get("status") in (None, 405):
        spec = get_json(args.base.replace("/api/v1", "") + "/openapi.json")
        paths = list((spec.get("body") or {}).get("paths", {})) if spec.get("status") == 200 else []
        route_probe = {
            "openapi_status": spec.get("status"),
            "n_paths": len(paths),
            "has_wind_resistance": any("wind_resistance" in p for p in paths),
            "has_wind_any": [p for p in paths if "wind" in p.lower()],
            "has_env_forces": "/api/v1/calculations/environmental-forces" in paths,
        }
    #  (c) the documented route table captured from the live platform
    audit = None
    if os.path.exists(os.path.join(HERE, "artifacts", "p16n7",
                                   "platform_baseline.json")):
        with open(os.path.join(HERE, "artifacts", "p16n7",
                               "platform_baseline.json")) as fh:
            audit = json.load(fh).get("endpoint_audit")
    dedicated_available = probe["status"] == 200 or bool(
        route_probe and route_probe["has_wind_resistance"]) or bool(
        audit and audit.get("has_wind_resistance_route"))

    # --- 2. environmental-force endpoint (the one that really exists) ------
    cached_responses = {}
    base_json = os.path.join(HERE, "artifacts", "p16n7", "platform_baseline.json")
    if os.path.exists(base_json):
        with open(base_json) as fh:
            pb = json.load(fh)
        cached_responses = {round(r["angle_deg"], 3): r["response"]
                            for r in pb.get("responses", [])}
    rows = []
    for angle in table["angles_deg"]:
        payload = to_platform_payload(table, angle_deg=angle,
                                      wind_speed=args.wind_speed, ship=ship)
        # strip the LBM extras for the platform call, keep them for the record
        plain = {"ship": ship,
                 "wind": {"V_wind": payload["wind"]["V_wind"],
                          "psi_wind": payload["wind"]["psi_wind"]}}
        resp = post_json(f"{args.base}/calculations/environmental-forces", plain)
        body = resp.get("body") if isinstance(resp.get("body"), dict) else {}
        cached = cached_responses.get(round(angle, 3))
        if not body and cached:
            body = cached
            resp = {"status": "cached"}
        lbm = payload["wind"]["forces"]
        rows.append({
            "angle_deg": angle,
            "lbm_coefficients": payload["wind"]["coefficients"],
            "lbm_forces": lbm,
            "platform_status": resp.get("status"),
            "platform_forces": {k: body.get(k) for k in
                                ("F_wind_x", "F_wind_y", "N_wind")} if body else None,
            "ratio_platform_over_lbm_Fx": (
                abs(body.get("F_wind_x", 0.0)) / abs(lbm["F_x_N"])
                if body and lbm["F_x_N"] else None),
            "ratio_platform_over_lbm_Fy": (
                abs(body.get("F_wind_y", 0.0)) / abs(lbm["F_y_N"])
                if body and lbm["F_y_N"] else None),
        })

    out = {
        "schema": "tensorlbm.wind_load_platform_integration/v1",
        "table": os.path.relpath(TABLE, HERE),
        "table_schema": table["schema"],
        "base_url": args.base,
        "wind_speed_ms": args.wind_speed,
        "wind_heading_convention": {
            "platform": "psi_wind [deg], 0 = head wind, 90 = starboard beam",
            "lbm": "same; LBM table theta = wind angle relative to the ship "
                   "centreline, geometry rotated about the planform centroid",
        },
        "reference_areas": {
            "A_F_m2": ref["A_F_m2"], "A_L_m2": ref["A_L_m2"],
            "B_m": ref["B_m"], "H_m": ref["H_m"],
        },
        "endpoint_probe": {
            "requested": "POST /calculations/wind_resistance",
            "status": probe.get("status"),
            "direct_post_error": probe.get("error"),
            "openapi_route_probe": route_probe,
            "endpoint_audit": audit,
            "available": dedicated_available,
            "conclusion": (
                "dedicated endpoint present" if dedicated_available else
                "the campaign-designated endpoint /api/v1/calculations/"
                "wind_resistance does not exist on the live platform; the wind "
                "force is computed inside /api/v1/calculations/"
                "environmental-forces, so the LBM table is injected through "
                "that contract (wind.coefficients / wind.forces in the request "
                "payload, documented in wind_load_table.to_platform_payload)"),
        },
        "comparison": rows,
        "status": "table_delivered; runtime injection blocked upstream"
                  if not dedicated_available else "table_delivered; endpoint verified",
        "next_step_for_platform": (
            "add a coefficient-table lookup (or accept wind.coefficients) in the "
            "wind-force routine so the LBM table replaces the empirical "
            "build-up; identical JSON schema, no change to the ship/current/ice "
            "coupling"),
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"dedicated_endpoint": dedicated_available,
                      "status": out["status"],
                      "angles": [r["angle_deg"] for r in rows]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())