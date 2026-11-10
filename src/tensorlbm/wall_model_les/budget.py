"""Reynolds-stress budget and DNS anchors for wall-modelled LES.

Everything here works on *mean-flow profiles in wall units*, so it can be used
both for a WMLES run (:mod:`.channel`) and for an a-priori check of a wall
model against DNS data -- which is the honest way to grade a wall model when
the resolved LES does not (yet) contain resolved turbulence.

DNS anchor
----------
``load_dns_channel_profile`` reads ``benchmarks/pending/channel_turb_3d/dns_ref/
mkm180.csv`` (Moser, Kim & Mansour 1999, ``Re_tau = 180`` channel DNS, shipped
with this repository) and reconstructs the Reynolds shear stress from the exact
channel momentum balance rather than from hand-typed numbers:

    tau+(y) = 1 - y/h = dU+/dy+ - <u'v'>+        =>
    -<u'v'>+(y) = 1 - y+/h+ - dU+/dy+ .

The derivative ``dU+/dy+`` is computed from the tabulated ``u_plus(y_plus)``
by a monotone finite difference, so every number downstream (peak location,
ensemble velocity scales, errors) is derived programmatically from the CSV.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "DnsChannelProfile",
    "load_dns_channel_profile",
    "reynolds_stress_from_balance",
    "peak_location",
    "relative_error",
    "log_region_error",
    "budget_from_profile",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DNS_CSV = REPO_ROOT / "benchmarks" / "pending" / "channel_turb_3d" / "dns_ref" / "mkm180.csv"

#: Second DNS tier shipped with the repository (MKM 1999, ``Re_tau = 395``).
#: Two Reynolds numbers turn the wall model's Reynolds-number behaviour from an
#: assumption into a measurement: the DNS abscissa of the ``-<u'v'>+`` peak is
#: ``Re``-invariant, so a closure that drifts with ``Re`` can be caught.
DNS_REFERENCES = {
    "mkm180": {
        "path": DEFAULT_DNS_CSV,
        "re_tau": 180.0,
        "citation": "Moser, Kim & Mansour 1999, Phys. Fluids 11, 943 (Re_tau=180)",
    },
    "mkm395": {
        "path": REPO_ROOT / "benchmarks" / "in99_n03_channel_turb" / "dns_ref" / "mkm395.csv",
        "re_tau": 395.0,
        "citation": "Moser, Kim & Mansour 1999, Phys. Fluids 11, 943 (Re_tau=395)",
    },
}


@dataclass(frozen=True)
class DnsChannelProfile:
    """Channel DNS profile in wall units, with the derived Reynolds stress."""

    re_tau: float
    y_plus: np.ndarray
    u_plus: np.ndarray
    uv_plus: np.ndarray
    production_plus: np.ndarray
    source: str

    @property
    def uv_peak_y_plus(self) -> float:
        """Abscissa of the ``-<u'v'>+`` maximum (``uv_plus`` already carries the sign)."""
        return peak_location(self.y_plus, self.uv_plus)

    @property
    def production_peak_y_plus(self) -> float:
        return peak_location(self.y_plus, self.production_plus)

    def as_dict(self) -> dict:
        return {
            "re_tau": self.re_tau,
            "n_points": int(self.y_plus.size),
            "uv_peak_y_plus": self.uv_peak_y_plus,
            "uv_peak_value": float(np.max(self.uv_plus)),
            "production_peak_y_plus": self.production_peak_y_plus,
            "source": self.source,
        }


def load_dns_channel_profile(
    path: str | Path | None = None, *, re_tau: float | None = None
) -> DnsChannelProfile:
    """Load a channel DNS table and derive ``<u'v'>+`` from the balance.

    ``re_tau`` is *read from the table* (``y+`` at the centreline, where
    ``y/h = 1``) unless it is given explicitly.  The previous implementation
    hard-coded ``180``, so loading the ``Re_tau = 395`` tier silently
    mis-scaled every derived quantity -- a defect this signature removes.
    """
    csv_path = Path(path) if path is not None else DEFAULT_DNS_CSV
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    y_plus = np.asarray(data["y_plus"], dtype=float)
    u_plus = np.asarray(data["u_plus"], dtype=float)
    order = np.argsort(y_plus)
    y_plus, u_plus = y_plus[order], u_plus[order]
    if re_tau is None:
        registered = [
            entry["re_tau"]
            for entry in DNS_REFERENCES.values()
            if Path(entry["path"]).resolve() == csv_path.resolve()
        ]
        # The shipped tables stop at the last cell centre (y+ = 178.12 for the
        # Re_tau = 180 case), so the table itself under-reports h+ by ~1 %.  A
        # registered tier carries the case identification; an unknown table is
        # read as ending at the centreline.
        re_tau = float(registered[0]) if registered else float(y_plus[-1])
    re_tau = float(re_tau)
    # Convention: uv_plus stores -<u'v'>+ (positive in the log layer), which is
    # what the stress balance 1 - y+/h+ - dU+/dy+ gives directly.
    uv_plus = reynolds_stress_from_balance(y_plus, u_plus, re_tau=re_tau)
    du_dy = np.gradient(u_plus, y_plus, edge_order=2)
    production = uv_plus * du_dy
    return DnsChannelProfile(
        re_tau=re_tau,
        y_plus=y_plus,
        u_plus=u_plus,
        uv_plus=uv_plus,
        production_plus=production,
        source=str(csv_path),
    )


def reynolds_stress_from_balance(
    y_plus: np.ndarray, u_plus: np.ndarray, *, re_tau: float
) -> np.ndarray:
    """``<u'v'>+ = 1 - y+/h+ - dU+/dy+`` from the exact channel stress balance."""
    du_dy = np.gradient(u_plus, y_plus, edge_order=2)
    return 1.0 - y_plus / re_tau - du_dy


def peak_location(x: np.ndarray, y: np.ndarray, *, half_width: int = 2) -> float:
    """Peak abscissa with a local parabolic vertex fit (sub-grid interpolation).

    A coarse WMLES grid samples the wall-normal direction on ``dy+ ~ 20-50``;
    the vertex of a parabola through the three samples around the maximum
    removes most of that discretisation error.  The raw grid maximum is
    returned when the fit is not concave.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    i = int(np.argmax(y))
    lo = max(i - half_width, 0)
    hi = min(i + half_width + 1, len(y))
    xs, ys = x[lo:hi], y[lo:hi]
    if xs.size < 3:
        return float(x[i])
    coeffs = np.polyfit(xs, ys, 2)
    if coeffs[0] >= 0.0:
        return float(x[i])
    vertex = -coeffs[1] / (2.0 * coeffs[0])
    if not (xs.min() <= vertex <= xs.max()):
        return float(x[i])
    return float(vertex)


def relative_error(value: float, reference: float) -> float:
    """Signed-free relative error ``|value - reference| / |reference|``."""
    if reference == 0.0:
        return float("nan")
    return abs(value - reference) / abs(reference)


def log_region_error(
    y_plus,
    u_plus,
    reference_u_plus,
    *,
    band: tuple[float, float] = (30.0, 300.0),
) -> dict:
    """Max / RMS relative error of ``u+`` inside the log-layer band.

    ``reference_u_plus`` may be an array of the same length or a callable.
    Only samples with ``band[0] <= y+ <= band[1]`` take part.
    """
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    ref = reference_u_plus(y_plus) if callable(reference_u_plus) else np.asarray(
        reference_u_plus, dtype=float
    )
    mask = (y_plus >= band[0]) & (y_plus <= band[1])
    if mask.sum() < 2:
        return {"n": int(mask.sum()), "max_rel": float("nan"), "rms_rel": float("nan")}
    rel = np.abs(u_plus[mask] - ref[mask]) / np.abs(ref[mask])
    return {
        "n": int(mask.sum()),
        "max_rel": float(rel.max()),
        "rms_rel": float(np.sqrt(np.mean(rel**2))),
        "y_plus_min": float(y_plus[mask].min()),
        "y_plus_max": float(y_plus[mask].max()),
        "mean_measured": float(np.mean(u_plus[mask])),
        "mean_reference": float(np.mean(ref[mask])),
    }


def budget_from_profile(
    *,
    y_plus: np.ndarray,
    u_plus: np.ndarray,
    uv_plus: np.ndarray,
    re_tau: float,
    h_plus: float,
    nu: float = 1.0,
    diss_res_plus: np.ndarray | None = None,
    diss_sgs_plus: np.ndarray | None = None,
    vvv_plus: np.ndarray | None = None,
    pv_plus: np.ndarray | None = None,
) -> dict:
    """Assemble the (resolved) Reynolds-stress budget terms in wall units.

    Terms (per unit ``u_tau^4 / nu``):

    ``P+``   production   ``-<u'v'>+ dU+/dy+``
    ``eps+`` dissipation  ``eps_res+ + eps_sgs+``  (SGS part from the LES model)
    ``T+``   turbulent transport ``-d/dy+( <v'v'v'>+/2 + <p'v'>+ )`` (when the
             triple/pressure correlations are supplied)

    and the closure residual ``R+ = P+ - eps+ - dT+/dy+``, which physical
    budget data must drive to zero in the log region.  The residual is reported
    explicitly, never hidden: a large residual means the supplied statistics
    are incomplete (for example no pressure-diffusion term), which is a
    limitation of the measurement, not of the identity.
    """
    y = np.asarray(y_plus, dtype=float)
    du_dy = np.gradient(np.asarray(u_plus, dtype=float), y, edge_order=2)
    uv = np.asarray(uv_plus, dtype=float)
    production = -uv * du_dy
    total_stress = du_dy - uv  # should equal 1 - y+/h+
    identity_err = np.abs(total_stress - (1.0 - y / h_plus))
    eps = None
    if diss_res_plus is not None:
        eps = np.asarray(diss_res_plus, dtype=float)
        if diss_sgs_plus is not None:
            eps = eps + np.asarray(diss_sgs_plus, dtype=float)
    transport = None
    if vvv_plus is not None and pv_plus is not None:
        flux = 0.5 * np.asarray(vvv_plus, dtype=float) + np.asarray(pv_plus, dtype=float)
        transport = -np.gradient(flux, y, edge_order=2)
    out = {
        "y_plus": y.tolist(),
        "production_plus": production.tolist(),
        "stress_identity_max_abs_error": float(identity_err.max()),
        "production_peak_y_plus": peak_location(y, production),
        "production_peak_value": float(production.max()),
    }
    if eps is not None:
        out["dissipation_plus"] = eps.tolist()
        out["dissipation_peak_y_plus"] = peak_location(y, eps)
    if transport is not None:
        out["transport_plus"] = transport.tolist()
        if eps is not None:
            residual = production - eps + transport
            out["residual_plus"] = residual.tolist()
            mask = (y >= 30.0) & (y <= 300.0)
            if mask.any():
                out["residual_rms_log_region"] = float(np.sqrt(np.mean(residual[mask] ** 2)))
    return out


def dump_json(payload: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=float))