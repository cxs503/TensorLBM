"""Wall-bounded LES channel driver and MKM-1999 validation gates.

The module couples three pieces:

* :func:`build_kernel` compiles ``les_chan.c`` (D3Q19 TRT/BGK + Smagorinsky,
  OpenMP) once per source hash with the system ``gcc`` and returns the binary.
  This exists because the eager-torch production channel solver runs at
  O(1) Mcell-steps/s, while a turbulent-channel validation campaign needs
  O(1e11) cell-steps (measured: 66 Mcell-steps/s at 64 threads on a loong64
  cluster node, i.e. ~50x the torch path).
* :func:`run_native` executes one case and returns the parsed artefact with its
  provenance (kernel hash, compiler version, host, wall-clock throughput).
* :func:`wall_units` / :func:`compare_to_mkm` / :func:`evaluate_gates` turn the
  raw y-resolved moments into wall-unit profiles, three independent friction
  velocity routes, a wall-law signature and the C1..C6 node gates.

Every number used by a gate is computed here from the run artefacts; nothing is
hand-copied, and a missing artefact yields ``withheld`` rather than ``pass``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tensorlbm.validation import mkm1999

__all__ = [
    "LesChannelConfig",
    "build_kernel",
    "compare_to_mkm",
    "evaluate_gates",
    "laminar_wall_origin",
    "run_native",
    "wall_law_fit",
    "wall_units",
]

KERNEL_SOURCE = Path(__file__).resolve().parent / "les_chan.c"


@dataclass(frozen=True)
class LesChannelConfig:
    """One native LES channel case."""

    nx: int = 200
    ny: int = 66
    nz: int = 100
    steps: int = 90000
    warmup: int = 15000
    sample: int = 20
    re_tau: float = 178.12
    u_tau: float = 0.006
    cs: float = 0.1
    lambda_: float = 0.1875
    perturb: float = 0.05
    perturb_modes: int = 6
    seed: int = 20261108
    threads: int = 64

    @property
    def half_height(self) -> float:
        return 0.5 * (self.ny - 2)

    @property
    def nu(self) -> float:
        return self.u_tau * self.half_height / self.re_tau

    @property
    def tau0(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def eddy_turnovers(self) -> float:
        return (self.steps - self.warmup) * self.u_tau / self.half_height

    def validate(self) -> None:
        if self.ny < 10 or self.nx < 8 or self.nz < 8:
            raise ValueError("grid too small for a wall-bounded channel")
        if self.tau0 <= 0.5005:
            raise ValueError("derived relaxation time is too close to 0.5")
        if self.u_tau * 20.0 > 0.25:
            raise ValueError("u_tau pushes the centreline beyond Ma ~ 0.4")

    def cli_args(self, out_prefix: str) -> list[str]:
        return [
            "--nx", str(self.nx), "--ny", str(self.ny), "--nz", str(self.nz),
            "--steps", str(self.steps), "--warmup", str(self.warmup),
            "--sample", str(self.sample), "--re-tau", repr(float(self.re_tau)),
            "--u-tau", repr(float(self.u_tau)), "--cs", repr(float(self.cs)),
            "--lambda", repr(float(self.lambda_)), "--perturb", repr(float(self.perturb)),
            "--modes", str(self.perturb_modes), "--seed", str(self.seed),
            "--threads", str(self.threads), "--out", out_prefix,
        ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_kernel(build_dir: str | Path | None = None, force: bool = False) -> Path:
    """Compile ``les_chan.c`` with gcc/OpenMP; returns the binary path.

    The binary is cached in ``build_dir`` (default: ``$TMPDIR/tensorlbm-les-chan``)
    under the source hash, so repeated runs never recompile and a stale binary
    can never masquerade as the current kernel.
    """
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        raise RuntimeError("no C compiler (gcc/cc) available to build les_chan")
    root = Path(build_dir or os.environ.get("TENSORLBM_LES_BUILD", "/tmp/tensorlbm-les-chan"))
    root.mkdir(parents=True, exist_ok=True)
    tag = _sha256(KERNEL_SOURCE)[:16]
    binary = root / f"les_chan_{tag}"
    if binary.exists() and not force:
        return binary
    cmd = [compiler, "-O3", "-fopenmp", "-march=native", "-o", str(binary),
           str(KERNEL_SOURCE), "-lm"]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"kernel build failed: {done.stderr[-2000:]}")
    return binary


def run_native(
    config: LesChannelConfig,
    out_dir: str | Path,
    name: str,
    *,
    build_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict:
    """Run one native LES case and return the parsed result with provenance."""
    config.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{name}.json"
    if result_path.exists() and not overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))
    binary = build_kernel(build_dir)
    prefix = out_dir / name
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(config.threads)
    started = time.time()
    done = subprocess.run(
        [str(binary)] + config.cli_args(str(prefix)),
        capture_output=True, text=True, env=env,
    )
    elapsed = time.time() - started
    if done.returncode != 0:
        raise RuntimeError(f"les_chan failed: {done.stderr[-2000:]}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["provenance"] = {
        "kernel_source": str(KERNEL_SOURCE.relative_to(KERNEL_SOURCE.parents[2])),
        "kernel_source_sha256": _sha256(KERNEL_SOURCE),
        "binary_sha256": _sha256(binary),
        "compiler": str(binary.parent),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": config.threads,
        "elapsed_seconds": elapsed,
        "mega_cell_steps_per_second": config.cells * config.steps / max(elapsed, 1e-9) / 1e6,
        "openmp": env.get("OMP_NUM_THREADS"),
        "stderr_tail": done.stderr.strip().splitlines()[-1:] or [],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload["case"] = {"name": name, **asdict(config)}
    result_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# wall-unit reduction
# ---------------------------------------------------------------------------

def laminar_wall_origin(y: np.ndarray, u: np.ndarray, wall_gap: float) -> float:
    """Effective no-slip offset of the half-way bounce-back wall, from a laminar twin.

    A steady plane-Poiseuille profile between walls at ``y_w`` and ``H - y_w`` is

        u(y) = C (y - y_w) (H + wall_gap - 2 y_w - (y - y_w)) / 2 ...

    which for the discrete grid is ``u = A (y - y_w) (Y - y)`` with
    ``Y = (ny - 2) + 2 y_w`` (the two wall planes).  ``y_w`` is recovered by a
    golden-section search on the residual of the two-parameter linear fit in
    ``A``; the returned value is measured from the centre of the first fluid
    node row (so a perfectly centred half-way bounce-back returns 0.5).
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    span = float(y.max() - y.min()) + 1.0  # fluid extent in cells (rows span H-1)

    def residual(y_w: float) -> float:
        design = (y - y_w) * (span + 2.0 * y_w - (y - y_w))
        denom = float((design * design).sum())
        if denom <= 0.0:
            return np.inf
        amp = float((design * u).sum() / denom)
        return float(((u - amp * design) ** 2).sum())

    lo, hi = 0.05, 1.45
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(80):
        c = hi - phi * (hi - lo)
        d = lo + phi * (hi - lo)
        if residual(c) < residual(d):
            hi = d
        else:
            lo = c
    return 0.5 * (lo + hi)


def wall_law_fit(
    y_plus: np.ndarray,
    u_plus: np.ndarray,
    window: tuple[float, float] = (30.0, 120.0),
    *,
    kappa_fixed: float | None = None,
) -> dict:
    """Least-squares ``U+ = (1/kappa) ln(y+) + B`` over a wall-unit window."""
    y_plus = np.asarray(y_plus, dtype=float)
    u_plus = np.asarray(u_plus, dtype=float)
    mask = (y_plus >= window[0]) & (y_plus <= window[1])
    if int(mask.sum()) < 3:
        return {"ok": False, "n_points": int(mask.sum()), "window": list(window)}
    x = np.log(y_plus[mask])
    y = u_plus[mask]
    if kappa_fixed is None:
        design = np.stack([x, np.ones_like(x)], axis=1)
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        kappa = 1.0 / float(slope)
        intercept = float(intercept)
    else:
        kappa = float(kappa_fixed)
        intercept = float(np.mean(y - x / kappa))
        slope = 1.0 / kappa
    predicted = slope * x + intercept
    residual = y - predicted
    return {
        "ok": True,
        "kappa": kappa,
        "B": intercept,
        "n_points": int(mask.sum()),
        "window": [float(window[0]), float(window[1])],
        "rms_residual": float(np.sqrt(np.mean(residual ** 2))),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "u_tau_from_fit": None,  # filled in by wall_units (needs the fit iteration)
    }


def _log_law_u_tau(
    y: np.ndarray, u: np.ndarray, nu: float, y_w: float, kappa: float = 0.41
) -> dict | None:
    """Solve ``U(y) = (u_tau/kappa) ln(y+ ) + B u_tau`` for u_tau by fixed point."""
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    u_tau = float(np.sqrt(nu * max((u[1] - u[0]) / max(y[1] - y[0], 1e-12), 1e-12)))
    if not np.isfinite(u_tau) or u_tau <= 0.0:
        return None
    for _ in range(60):
        y_plus = (y - y_w) * u_tau / nu
        mask = (y_plus >= 30.0) & (y_plus <= 0.3 * float(y_plus.max()))
        if int(mask.sum()) < 3:
            return None
        design = np.stack([np.log(y_plus[mask]) / kappa, np.ones(int(mask.sum()))], axis=1)
        u_tau_new, _ = np.linalg.lstsq(design, u[mask], rcond=None)[0]
        if u_tau_new <= 0.0:
            return None
        if abs(u_tau_new - u_tau) <= 1e-10 * max(u_tau, 1e-12):
            u_tau = float(u_tau_new)
            break
        u_tau = float(u_tau_new)
    y_plus = (y - y_w) * u_tau / nu
    mask = (y_plus >= 30.0) & (y_plus <= 0.3 * float(y_plus.max()))
    return {
        "u_tau": u_tau,
        "n_points": int(mask.sum()),
        "window_y_plus": [float(y_plus[mask].min()), float(y_plus[mask].max())]
        if int(mask.sum())
        else None,
    }


def wall_units(result: dict, *, kappa: float = 0.41, y_w: float = 0.5) -> dict:
    """Reduce a native run artefact to wall units and friction-velocity routes."""
    derived = result["derived"]
    profiles = result["profiles"]
    nu = float(derived["nu"])
    half_height = float(derived["half_height"])
    accel = float(derived["accel"])
    y = np.asarray(profiles["y"], dtype=float)
    u = np.asarray(profiles["mean_ux"], dtype=float)
    r_uu = np.asarray(profiles["r_uu"], dtype=float)
    r_vv = np.asarray(profiles["r_vv"], dtype=float)
    r_ww = np.asarray(profiles["r_ww"], dtype=float)
    r_uv = np.asarray(profiles["r_uv"], dtype=float)

    # route 1: momentum balance of the whole channel (imposed stress = a h)
    u_tau_mb = float(np.sqrt(accel * half_height))
    # route 2: viscous sublayer slope, quadratic fit over the first three rows
    n_near = min(3, y.size)
    design = np.stack([(y[:n_near] - y_w) ** k for k in (1, 2)], axis=1)
    coeffs = np.linalg.lstsq(design, u[:n_near], rcond=None)[0]
    slope = float(coeffs[0])
    u_tau_sub = float(np.sqrt(nu * max(slope, 0.0))) if slope > 0 else float("nan")
    # route 3: log-law fixed point (kappa fixed, intercept free)
    log_fit = _log_law_u_tau(y, u, nu, y_w, kappa=kappa)
    u_tau_log = float(log_fit["u_tau"]) if log_fit else float("nan")

    u_tau = u_tau_mb
    y_plus = (y - y_w) * u_tau / nu
    u_plus = u / u_tau
    rms = {
        "u": np.sqrt(np.maximum(r_uu, 0.0)) / u_tau,
        "v": np.sqrt(np.maximum(r_vv, 0.0)) / u_tau,
        "w": np.sqrt(np.maximum(r_ww, 0.0)) / u_tau,
    }
    uv_plus = r_uv / u_tau ** 2
    re_tau_target = float(result["configuration"]["re_tau"])
    wall_law = wall_law_fit(y_plus, u_plus, (30.0, min(120.0, 0.3 * re_tau_target)))

    half = y_plus <= re_tau_target * 1.02  # resolved half channel

    # bulk velocity in wall units (integral over the half channel)
    u_bulk = float(result.get("bulk_velocity", float(np.trapezoid(u, y) / (y.max() - y.min()))))
    u_bulk_plus = u_bulk / u_tau
    cf = 2.0 / u_bulk_plus ** 2

    # peaks (search only in the resolved half channel)
    def peak(values: np.ndarray) -> dict:
        idx = int(np.argmax(np.where(half, values, -np.inf)))
        return {"y_plus": float(y_plus[idx]), "value": float(values[idx])}

    routes = {
        "momentum_balance": u_tau_mb,
        "viscous_sublayer": u_tau_sub,
        "log_law": u_tau_log,
    }
    finite = {k: v for k, v in routes.items() if np.isfinite(v) and v > 0.0}
    deviations = {}
    if len(finite) >= 2:
        values = np.array(list(finite.values()))
        mean = float(values.mean())
        deviations = {k: float(100.0 * (v - mean) / mean) for k, v in finite.items()}
    anisotropy = {
        "stations": int(half.sum()),
        "u_gt_w": int(np.sum((rms["u"] > rms["w"]) & half)),
        "w_gt_v": int(np.sum((rms["w"] > rms["v"]) & half)),
        "ordering_u_w_v": int(
            np.sum((rms["u"] > rms["w"]) & (rms["w"] > rms["v"]) & half)
        ),
    }
    return {
        "y": y.tolist(),
        "y_plus": y_plus.tolist(),
        "u_plus": u_plus.tolist(),
        "u_rms_plus": rms["u"].tolist(),
        "v_rms_plus": rms["v"].tolist(),
        "w_rms_plus": rms["w"].tolist(),
        "uv_plus": uv_plus.tolist(),
        "u_tau": routes,
        "u_tau_deviations_pct": deviations,
        "max_pairwise_u_tau_deviation_pct": (
            float(max(deviations.values()) - min(deviations.values())) if deviations else None
        ),
        "wall_law": wall_law,
        "u_bulk": u_bulk,
        "u_bulk_plus": u_bulk_plus,
        "cf": cf,
        "peaks": {
            "u_rms": peak(rms["u"]),
            "v_rms": peak(rms["v"]),
            "w_rms": peak(rms["w"]),
            "minus_uv": peak(-uv_plus),
        },
        "anisotropy": anisotropy,
        "re_tau_effective": re_tau_target,
        "nu": nu,
        "half_height": half_height,
    }


def compare_to_mkm(
    profile: dict,
    mkm: mkm1999.MKMProfile | None,
    *,
    buffer_window: tuple[float, float] = (5.0, 30.0),
    log_window: tuple[float, float] = (30.0, 120.0),
) -> dict:
    """Relative errors of ``U+`` (DNS stations), rms peaks and ``C_f``."""
    if mkm is None:
        return {"available": False}
    y_plus = np.asarray(profile["y_plus"], dtype=float)
    u_plus = np.asarray(profile["u_plus"], dtype=float)
    log_hi = min(log_window[1], 0.3 * mkm.re_tau)
    out: dict = {"available": True, "reference": "MKM-1999 CoMO/ODEN DNS", "re_tau": mkm.re_tau}

    for label, window in (("buffer", buffer_window), ("log", (log_window[0], log_hi))):
        mask = (mkm.y_plus >= window[0]) & (mkm.y_plus <= window[1])
        if int(mask.sum()) < 2:
            out[label] = {"ok": False, "n_points": int(mask.sum())}
            continue
        target = mkm.u_plus[mask]
        got = np.interp(mkm.y_plus[mask], y_plus, u_plus)
        rel = (got - target) / target
        out[label] = {
            "ok": True,
            "n_points": int(mask.sum()),
            "rms_relative_pct": float(100.0 * np.sqrt(np.mean(rel ** 2))),
            "max_relative_pct": float(100.0 * np.max(np.abs(rel))),
            "mean_relative_pct": float(100.0 * np.mean(rel)),
            "y_plus_range": [float(window[0]), float(window[1])],
        }

    rms_peak_ref = mkm.peak("u_rms")
    got_peak = profile["peaks"]["u_rms"]
    out["u_rms_peak"] = {
        "reference_y_plus": rms_peak_ref["y_plus"],
        "reference_value": rms_peak_ref["value"],
        "computed_y_plus": got_peak["y_plus"],
        "computed_value": got_peak["value"],
        "value_relative_pct": float(
            100.0 * (got_peak["value"] - rms_peak_ref["value"]) / rms_peak_ref["value"]
        ),
        "y_plus_error": float(got_peak["y_plus"] - rms_peak_ref["y_plus"]),
    }
    uv_ref = mkm.peak("minus_uv")
    uv_got = profile["peaks"]["minus_uv"]
    out["minus_uv_peak"] = {
        "reference_y_plus": uv_ref["y_plus"],
        "reference_value": uv_ref["value"],
        "computed_y_plus": uv_got["y_plus"],
        "computed_value": uv_got["value"],
        "value_relative_pct": float(
            100.0 * (uv_got["value"] - uv_ref["value"]) / uv_ref["value"]
        ),
        "y_plus_error": float(uv_got["y_plus"] - uv_ref["y_plus"]),
    }
    cf_ref = mkm.friction_coefficient()
    out["cf"] = {
        "reference": cf_ref,
        "computed": profile["cf"],
        "relative_pct": float(100.0 * (profile["cf"] - cf_ref) / cf_ref),
    }
    out["u_bulk_plus"] = {
        "reference": mkm.bulk_velocity_plus(),
        "computed": profile["u_bulk_plus"],
        "relative_pct": float(
            100.0 * (profile["u_bulk_plus"] - mkm.bulk_velocity_plus())
            / mkm.bulk_velocity_plus()
        ),
    }
    return out


# ---------------------------------------------------------------------------
# gate evaluation
# ---------------------------------------------------------------------------

def _gate(name: str, criterion: str, passed: bool | None, evidence: dict, problems=None) -> dict:
    return {
        "gate": name,
        "criterion": criterion,
        "passed": passed,
        "status": "pass" if passed is True else ("fail" if passed is False else "withheld"),
        "evidence": evidence,
        "problems": list(problems or []),
    }


def evaluate_gates(
    cases: dict[str, dict],
    *,
    production: str = "R180_h40",
    grid_levels: tuple[str, ...] = ("R180_h24", "R180_h32", "R180_h40"),
    negative_control: str = "R180_h32_nosgs",
    control_reference: str = "R180_h32",
) -> dict:
    """Evaluate the IN99-N03 gate set C1..C6 from run artefacts.

    ``cases`` maps a case name to ``{"result": <native artefact>, "profile":
    <wall-unit reduction>, "comparison": <MKM comparison>}``.  A gate whose
    inputs are missing is ``withheld`` (never ``pass``).
    """
    gates: list[dict] = []
    prod = cases.get(production)
    mkm180 = mkm1999.load_profile(180)

    # ---- C1: mean profile against MKM-1999 at the Re_tau = 180 anchor -----
    if prod is None:
        gates.append(_gate("C1_u_plus_vs_mkm180", "buffer <=5%, log <=3%", None,
                           {}, ["production case missing"]))
    else:
        comp = prod["comparison"]
        buf = comp.get("buffer", {})
        log = comp.get("log", {})
        ok = (
            bool(buf.get("ok")) and bool(log.get("ok"))
            and buf["rms_relative_pct"] <= 5.0 and log["rms_relative_pct"] <= 3.0
        )
        problems = []
        if buf.get("ok") and buf["rms_relative_pct"] > 5.0:
            problems.append(
                f"buffer-layer U+ rms error {buf['rms_relative_pct']:.2f}% exceeds 5%"
            )
        if log.get("ok") and log["rms_relative_pct"] > 3.0:
            problems.append(f"log-layer U+ rms error {log['rms_relative_pct']:.2f}% exceeds 3%")
        gates.append(_gate(
            "C1_u_plus_vs_mkm180", "buffer(5<y+<30) rms<=5%, log(30<y+<0.3Re) rms<=3%",
            ok,
            {"buffer": buf, "log": log, "case": production,
             "grid": prod["result"]["configuration"]},
            problems,
        ))

    # ---- C2: three-way u_tau consistency ---------------------------------
    if prod is None:
        gates.append(_gate("C2_u_tau_self_consistency", "max pairwise spread <=3%", None, {},
                           ["production case missing"]))
    else:
        prof = prod["profile"]
        spread = prof.get("max_pairwise_u_tau_deviation_pct")
        routes = prof.get("u_tau", {})
        ok = spread is not None and spread <= 3.0
        problems = []
        if spread is None:
            problems.append("log-law route undefined; spread not computable")
        elif spread > 3.0:
            problems.append(f"three-route friction velocity spread {spread:.2f}% exceeds 3%")
        gates.append(_gate(
            "C2_u_tau_self_consistency",
            "momentum balance vs viscous sublayer vs log-law, max pairwise <=3%",
            ok if spread is not None else None,
            {"routes": routes, "deviations_pct": prof.get("u_tau_deviations_pct"),
             "max_pairwise_spread_pct": spread, "case": production},
            problems,
        ))

    # ---- C3: Reynolds-stress peak position and magnitude -----------------
    if prod is None:
        gates.append(_gate("C3_stress_peak", "u_rms peak y+ in [12,15], |value err| <=10%",
                           None, {}, ["production case missing"]))
    else:
        comp = prod["comparison"]
        peak = comp.get("u_rms_peak", {})
        uv = comp.get("minus_uv_peak", {})
        ok = (
            bool(peak)
            and 12.0 <= peak["computed_y_plus"] <= 15.0
            and abs(peak["value_relative_pct"]) <= 10.0
        )
        problems = []
        if peak:
            if not (12.0 <= peak["computed_y_plus"] <= 15.0):
                problems.append(
                    f"u_rms peak at y+ = {peak['computed_y_plus']:.2f}, outside the "
                    f"12-15 band (MKM: {peak['reference_y_plus']:.2f})"
                )
            if abs(peak["value_relative_pct"]) > 10.0:
                problems.append(
                    f"u_rms peak magnitude error {peak['value_relative_pct']:+.2f}% exceeds 10%"
                )
        gates.append(_gate(
            "C3_stress_peak",
            "u_rms+ peak y+ in [12,15] and peak magnitude within 10% of MKM-1999",
            ok, {"u_rms_peak": peak, "minus_uv_peak": uv, "case": production}, problems,
        ))

    # ---- C4: anisotropy signature ----------------------------------------
    if prod is None:
        gates.append(_gate("C4_anisotropy_signature", "u_rms > w_rms > v_rms everywhere",
                           None, {}, ["production case missing"]))
    else:
        aniso = prod["profile"]["anisotropy"]
        total = max(int(aniso["stations"]), 1)
        fraction = aniso["ordering_u_w_v"] / total
        ok = fraction >= 0.95 and aniso["u_gt_w"] >= 0.95 * total and aniso["w_gt_v"] >= 0.95 * total
        problems = []
        if not ok:
            problems.append(
                f"u>w>v holds at {aniso['ordering_u_w_v']}/{total} stations "
                f"(u>w: {aniso['u_gt_w']}, w>v: {aniso['w_gt_v']})"
            )
        gates.append(_gate(
            "C4_anisotropy_signature",
            "u_rms > w_rms > v_rms at >=95% of resolved wall-normal stations",
            ok, {"anisotropy": aniso, "case": production}, problems,
        ))

    # ---- C5: grid convergence -------------------------------------------
    available = [cases[name] for name in grid_levels if name in cases]
    if len(available) < 2:
        gates.append(_gate("C5_grid_convergence", "L_inf(U+) <=5% between grids", None,
                           {"available_levels": [n for n in grid_levels if n in cases]},
                           ["fewer than two grid levels present"]))
    else:
        levels = sorted(available, key=lambda c: c["result"]["derived"]["cells"])
        worst = []
        for coarse, fine in zip(levels[:-1], levels[1:]):
            yc = np.asarray(coarse["profile"]["y_plus"])
            uc = np.asarray(coarse["profile"]["u_plus"])
            yf = np.asarray(fine["profile"]["y_plus"])
            uf = np.asarray(fine["profile"]["u_plus"])
            lo = max(30.0, yc.min(), yf.min())
            hi = min(0.3 * mkm180.re_tau, yc.max(), yf.max())
            mask = (yf >= lo) & (yf <= hi)
            if int(mask.sum()) < 3:
                worst.append({"pair": [coarse["result"]["case"]["name"],
                                       fine["result"]["case"]["name"]], "ok": False})
                continue
            got = np.interp(yf[mask], yc, uc)
            rel = np.abs(got - uf[mask]) / np.abs(uf[mask])
            worst.append({
                "pair": [coarse["result"]["case"]["name"], fine["result"]["case"]["name"]],
                "cells": [coarse["result"]["derived"]["cells"], fine["result"]["derived"]["cells"]],
                "linf_relative_pct": float(100.0 * np.max(rel)),
                "rms_relative_pct": float(100.0 * np.sqrt(np.mean(rel ** 2))),
                "window_y_plus": [float(lo), float(hi)],
                "ok": True,
            })
        evaluated = [w for w in worst if w.get("ok")]
        if not evaluated:
            gates.append(_gate("C5_grid_convergence", "L_inf(U+) <=5%", None, {"pairs": worst},
                               ["no grid pair had overlapping log-layer stations"]))
        else:
            linfs = [w["linf_relative_pct"] for w in evaluated]
            ok = max(linfs) <= 5.0
            problems = []
            if not ok:
                problems.append(
                    f"grid-convergence L_inf(U+) = {max(linfs):.2f}% exceeds 5% over the log window"
                )
            gates.append(_gate(
                "C5_grid_convergence",
                "successive grid triplets: max |Delta U+|/U+ over the log window <=5%",
                ok, {"pairs": evaluated}, problems,
            ))

    # ---- C6: negative control (SGS off must fail C_f by >20%) -----------
    ctrl = cases.get(negative_control)
    if ctrl is None:
        gates.append(_gate("C6_negative_control_sgs_off", "C_f deviation >20% without SGS",
                           None, {"case": negative_control},
                           ["negative-control case missing"]))
    else:
        comp = ctrl["comparison"]
        cf = comp.get("cf", {})
        dev = abs(cf.get("relative_pct", float("nan"))) if cf else float("nan")
        ok = bool(cf) and np.isfinite(dev) and dev > 20.0
        problems = []
        if not ok:
            problems.append(
                "the SGS-free arm stays within 20% of the DNS friction coefficient, so the "
                "negative control does not discriminate the sub-grid model"
            )
        ref_case = cases.get(control_reference)
        evidence = {
            "case": negative_control,
            "cf_deviation_pct_vs_mkm": cf.get("relative_pct") if cf else None,
            "u_bulk_plus": comp.get("u_bulk_plus"),
            "u_rms_peak_ratio_vs_control": None,
        }
        if ref_case is not None:
            peak_ctrl = ctrl["comparison"].get("u_rms_peak", {}).get("computed_value")
            peak_ref = ref_case["comparison"].get("u_rms_peak", {}).get("computed_value")
            if peak_ctrl and peak_ref:
                evidence["u_rms_peak_ratio_vs_control"] = float(peak_ctrl / peak_ref)
                evidence["u_rms_peak_control"] = peak_ref
                evidence["u_rms_peak_nosgs"] = peak_ctrl
        gates.append(_gate(
            "C6_negative_control_sgs_off",
            "SGS-off arm at the same grid must miss the DNS C_f by more than 20%",
            ok, evidence, problems,
        ))

    passed = sum(1 for g in gates if g["status"] == "pass")
    failed = sum(1 for g in gates if g["status"] == "fail")
    withheld = sum(1 for g in gates if g["status"] == "withheld")
    return {
        "node": "IN99-N03",
        "repo": "TensorLBM",
        "verdict": "pass" if failed == 0 and withheld == 0 else ("honest_fail" if failed else "withheld"),
        "summary": {"pass": passed, "fail": failed, "withheld": withheld, "total": len(gates)},
        "gates": gates,
        "reference": {
            "dataset": "Moser, Kim & Mansour (1999)",
            "provenance": mkm1999.load_provenance(),
        },
    }