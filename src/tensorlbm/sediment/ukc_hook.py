"""Engineering hook: sediment bed elevation -> ship UKC / siltation consumer.

Contract
--------
``bed_elevation_to_ukc`` turns the ``bed_elevation`` block of the sediment
verification JSON (per-column bed elevation change, m, positive up) into the
under-keel-clearance quantities a ship platform needs::

    available_depth(x) = h0 + z_b(x)                     (h0 = charted depth)
    ukc(x)             = available_depth(x) - draft - squat - allowances
    siltation(x)       = max(z_b(x), 0)                  (bed rise only)

and the *governing* value is ``min_x ukc(x)`` together with the column where it
occurs.  Scour (``z_b < 0``) is reported but does not reduce ``ukc`` -- it is
recorded because it changes the seabed for subsequent passes.

Why this lives here
-------------------
TensorPlatform's rule is that the solver repository owns the *number* and the
consumer owns the *decision*; the ship platform (``ship-performance-platform``)
is where the UKC/siltation endpoint belongs.  This module ships the transport
format plus a client (:func:`post_bed_elevation`) so the consumer side only has
to accept the payload.  As of 2026-11-09 the ship platform exposes no UKC or
siltation route (checked against its router table: resistance / propulsion /
seakeeping / maneuvering / stability / marine-icing / lbm / ...), so the pytest
integration test validates the contract locally and *records* the absence of the
endpoint instead of pretending to call it;
:func:`probe_endpoint` performs that check when a base URL is reachable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

__all__ = [
    "CONTRACT",
    "bed_elevation_to_ukc",
    "build_payload",
    "post_bed_elevation",
    "probe_endpoint",
]

#: versioned transport contract between the solver and the ship platform.
CONTRACT = "tensorlbm.sediment.bed_elevation/v1"


def bed_elevation_to_ukc(
    bed_elevation: dict,
    charted_depth_m: float,
    draft_m: float,
    squat_m: float = 0.0,
    allowances_m: float = 0.0,
) -> dict:
    """Consume the ``bed_elevation`` block and return the UKC deliverables.

    Args:
        bed_elevation: dict with ``x_over_D``/``bed_elevation_m`` (equal
            lengths) as emitted by
            :func:`tensorlbm.sediment.verification.build_results`.
        charted_depth_m: undisturbed water depth (m, positive down from the
            surface to the original bed).
        draft_m: ship draft (m).
        squat_m: dynamic squat allowance (m).
        allowances_m: additional allowance (m, e.g. wave/heel/trim margin).

    Returns:
        dict with the per-column arrays, ``min_ukc_m``, the governing column,
        ``max_siltation_m`` and the input parameters (audit trail).
    """
    z_b = [float(v) for v in bed_elevation["bed_elevation_m"]]
    x = [float(v) for v in bed_elevation["x_over_D"]]
    if len(z_b) != len(x):
        raise ValueError("x and bed elevation must have the same length")
    if not z_b:
        raise ValueError("empty bed elevation profile")
    subtract = draft_m + squat_m + allowances_m
    available = [charted_depth_m + v for v in z_b]
    ukc = [a - subtract for a in available]
    silt = [max(v, 0.0) for v in z_b]
    i_min = int(min(range(len(ukc)), key=lambda k: ukc[k]))
    return {
        "contract": CONTRACT,
        "x_over_D": x,
        "bed_elevation_m": z_b,
        "charted_depth_m": charted_depth_m,
        "available_depth_m": available,
        "ukc_m": ukc,
        "siltation_m": silt,
        "min_ukc_m": ukc[i_min],
        "min_ukc_column": {"index": i_min, "x_over_D": x[i_min], "bed_elevation_m": z_b[i_min]},
        "max_siltation_m": max(silt),
        "draft_m": draft_m,
        "squat_m": squat_m,
        "allowances_m": allowances_m,
    }


def build_payload(
    results: dict,
    charted_depth_m: float,
    draft_m: float,
    squat_m: float = 0.0,
    allowances_m: float = 0.0,
) -> dict:
    """Assemble the versioned payload the ship platform would consume."""
    return {
        "contract": CONTRACT,
        "source": {
            "solver": "TensorLBM",
            "module": "tensorlbm.sediment",
            "verification_schema": results.get("schema"),
        },
        "ukc": bed_elevation_to_ukc(
            results["bed_elevation"],
            charted_depth_m=charted_depth_m,
            draft_m=draft_m,
            squat_m=squat_m,
            allowances_m=allowances_m,
        ),
    }


def post_bed_elevation(
    payload: dict, base_url: str, path: str, timeout: float = 5.0
) -> dict:
    """POST the payload to a ship-platform route; return ``{status, body}``.

    Never raises on an HTTP error: the status code is returned so callers can
    distinguish "endpoint missing" (404) from "payload rejected" (4xx/5xx with
    a body) and record it in the audit trail.
    """
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return {"status": resp.status, "body": resp.read().decode("utf-8", "replace")}
    except urllib.error.HTTPError as exc:  # 404/422/... -> recorded, not raised
        return {"status": exc.code, "body": exc.read().decode("utf-8", "replace")}
    except Exception as exc:  # connection refused, DNS, timeout
        return {"status": None, "body": f"{type(exc).__name__}: {exc}"}


def probe_endpoint(base_url: str, path: str, timeout: float = 3.0) -> dict:
    """Probe whether ``base_url + path`` exists (used to record endpoint gaps)."""
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return {"reachable": True, "status": resp.status}
    except urllib.error.HTTPError as exc:
        return {"reachable": True, "status": exc.code}
    except Exception as exc:
        return {"reachable": False, "status": None, "error": f"{type(exc).__name__}: {exc}"}