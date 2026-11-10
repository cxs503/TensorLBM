"""C6 engineering hook: sediment bed elevation consumed by a UKC/siltation sink.

The solver's job is to produce the *number* (per-column bed elevation); the ship
platform owns the *decision* (UKC / siltation).  This integration test

1. runs the sediment chain end to end with a cheap configuration,
2. hands the ``bed_elevation`` block to the UKC consumer
   (:func:`tensorlbm.sediment.ukc_hook.bed_elevation_to_ukc`),
3. checks the UKC arithmetic against an independent hand computation,
4. attempts the real HTTP transport to the ship platform and *records* the
   outcome.  As of 2026-11-09 the ship platform exposes no UKC/siltation route
   (its router table has resistance / propulsion / seakeeping / maneuvering /
   stability / marine-icing / lbm / ...), so a 404 is the expected, recorded
   result; the test never fabricates a successful call.  If
   ``TENSOR_SHIP_API_BASE`` and ``TENSOR_SHIP_UKC_PATH`` are set (i.e. the ship
   side has landed the endpoint) the test requires a 2xx instead.
"""

from __future__ import annotations

import json
import os

import pytest

from tensorlbm.sediment.scour import PileConfig
from tensorlbm.sediment.ukc_hook import (
    CONTRACT,
    bed_elevation_to_ukc,
    build_payload,
    post_bed_elevation,
    probe_endpoint,
)
from tensorlbm.sediment.verification import (
    SedimentScenario,
    build_results,
    c2_rouse_profiles,
    c3_mass_budget,
    c4_negative_controls,
    c5_pile_scour,
)

SCEN = SedimentScenario()

#: the ship-platform base URL used by the audit probe (backend port, see
#: /root/tp-agent/ship_backend_start.sh -> uvicorn on 8002)
SHIP_BASE = os.environ.get("TENSOR_SHIP_API_BASE", "http://127.0.0.1:8001")
#: route the ship platform would expose for this payload (not present yet)
SHIP_UKC_PATH = os.environ.get("TENSOR_SHIP_UKC_PATH", "/api/v1/calculations/ukc")


@pytest.fixture(scope="module")
def sediment_results() -> dict:
    """End-to-end sediment run (cheap anchors; one scour relaxation)."""
    from tensorlbm.sediment.shields import critical_shields

    c1 = {
        "points": [
            {
                "d_star": 1.0,
                "theta_ref": critical_shields(1.0),
                "theta_num": critical_shields(1.0),
                "rel_err": 0.0,
                "converged": True,
                "bisection_iters": 0,
                "re_star_cr": 0.0,
                "d_star_old_formula": 0.0,
                "old_formula_rel_err": 0.0,
            }
        ],
        "max_rel_err": 0.0,
        "tol": 0.15,
        "pass": True,
    }
    c2 = c2_rouse_profiles(SCEN, z_targets=(0.6,))
    c3 = c3_mass_budget(SCEN, n_steps=400)
    c4 = c4_negative_controls(SCEN)
    c5 = c5_pile_scour(
        PileConfig(nx=48, ny=72, diameter=6.0, n_steps=4000, n_avg=800),
        trend=False,
    )
    return build_results(scen=SCEN, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5)


def test_ukc_consumer_arithmetic(sediment_results: dict) -> None:
    """The UKC consumer turns bed elevation into clearance consistently."""
    be = sediment_results["bed_elevation"]
    z_b = be["bed_elevation_m"]
    assert len(z_b) == len(be["x_over_D"]) > 0
    assert all(v <= 0.0 for v in z_b), "a scour hole must be a bed lowering"

    charted, draft, squat, allow = 6.0, 5.0, 0.1, 0.2
    ukc = bed_elevation_to_ukc(
        be, charted_depth_m=charted, draft_m=draft, squat_m=squat, allowances_m=allow
    )
    assert ukc["contract"] == CONTRACT
    # independent hand computation of the governing clearance
    expected = [charted + z - (draft + squat + allow) for z in z_b]
    assert ukc["ukc_m"] == pytest.approx(expected, rel=0.0, abs=1e-12)
    assert ukc["min_ukc_m"] == pytest.approx(min(expected), abs=1e-12)
    i = int(min(range(len(expected)), key=lambda k: expected[k]))
    assert ukc["min_ukc_column"]["index"] == i
    assert ukc["max_siltation_m"] == 0.0  # pure scour case: no siltation
    # the scour depth must be recovered from the depth changes (the section is
    # subsampled to 25 stations, so the discrete maximum can fall between them:
    # allow 10 %)
    assert max(-v for v in z_b) == pytest.approx(
        sediment_results["scour_depth"]["S_max_m"], rel=0.1
    )


def test_ukc_payload_is_versioned_and_serialisable(sediment_results: dict) -> None:
    payload = build_payload(
        sediment_results, charted_depth_m=6.0, draft_m=5.0, squat_m=0.1,
        allowances_m=0.2,
    )
    assert payload["contract"] == CONTRACT
    assert payload["source"]["solver"] == "TensorLBM"
    assert payload["source"]["verification_schema"] == sediment_results["schema"]
    text = json.dumps(payload)
    assert len(text) > 200
    assert json.loads(text)["ukc"]["contract"] == CONTRACT


def _fetch_json(url: str, timeout: float = 5.0):
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def test_ship_platform_ukc_endpoint_audit(tmp_path) -> None:
    """Transport audit: post the payload to the ship platform and record reality.

    * platform reachable, route absent (today) -> the 404 and the route census
      are recorded; the endpoint is a *ship-repo* deliverable, so the test
      documents the gap instead of fabricating a call;
    * ``TENSOR_SHIP_UKC_PATH`` set (i.e. the ship side landed the route) ->
      a 2xx response is required.
    """
    probe = probe_endpoint(SHIP_BASE, "/docs")
    record = {"ship_base": SHIP_BASE, "path": SHIP_UKC_PATH, "probe": probe,
              "contract": CONTRACT}
    if not probe["reachable"]:
        (tmp_path / "ukc_endpoint_audit.json").write_text(json.dumps(record, indent=2))
        pytest.skip(f"ship platform unreachable at {SHIP_BASE}: {probe.get('error')}")

    # route census: evidence that no UKC/siltation route exists on the platform
    try:
        spec = _fetch_json(SHIP_BASE.rstrip("/") + "/openapi.json")
        paths = sorted(spec.get("paths", {}))
    except Exception as exc:  # pragma: no cover - platform without openapi
        spec, paths = {}, []
        record["openapi_error"] = f"{type(exc).__name__}: {exc}"
    import re

    ukc_routes = [p for p in paths if re.search("ukc|silt|clearance|under.?keel|bed", p, re.I)]
    record.update({"n_routes": len(paths), "ukc_routes": ukc_routes})

    payload = build_payload(
        {
            "schema": "tensorlbm.sediment.verification/v1",
            "bed_elevation": {
                "x_over_D": [-1.0, 0.0, 1.0, 2.0],
                "bed_elevation_m": [-0.002, -0.065, -0.031, -0.001],
                "scour_depth_m": [0.002, 0.065, 0.031, 0.001],
            },
        },
        charted_depth_m=6.0,
        draft_m=5.0,
    )
    resp = post_bed_elevation(payload, SHIP_BASE, SHIP_UKC_PATH)
    record.update({"post_status": resp["status"], "post_body": resp["body"][:200]})
    (tmp_path / "ukc_endpoint_audit.json").write_text(json.dumps(record, indent=2))

    if os.environ.get("TENSOR_SHIP_UKC_PATH"):
        assert resp["status"] is not None and 200 <= int(resp["status"]) < 300, record
    else:
        # recorded gap: the platform answers, but has no route for this payload
        assert resp["status"] is not None, record
        assert int(resp["status"]) in (404, 405, 422), record
        assert ukc_routes == [], (
            "a UKC-like route appeared on the ship platform: re-point "
            f"TENSOR_SHIP_UKC_PATH at it (found {ukc_routes})"
        )
