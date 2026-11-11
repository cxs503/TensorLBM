"""Contamination audit of the TRT "F/2" forcing-scale defect (mission m7767).

Background
----------
The forced D3Q19 TRT channel of :mod:`tensorlbm.rough_channel3d` (and its C
back-end ``native/rough_channel3d.c``) subtracted the equilibrium in *both* TRT
modes and therefore had to weight the Guo force with ``(1 - s_minus/2)``.  From
the introduction of the native back-end (m5362) until the fix in m7767 the
weight was ``(1 - s_plus/2)``, which drives the channel with

    D(tau) = 1 + (s_minus - s_plus)/2          (closed form, ``drive_factor``)

instead of ``F``.  At the laminar anchor ``D`` is well below 1 and at the
campaign Reynolds numbers (``tau -> 0.5``) it collapses to ~0.4 %, i.e. the
defect is *not* a harmless factor of two but a tau-dependent rescaling.

What this module answers
------------------------
Every coefficient that the ice-roughness programme publishes -- the three-zone
onset parameter ``k0`` of ``C_w(k_s^+)``, the drag ratios ``Cf_rough/Cf_smooth``
(``cf_ratio``), the ``C_w`` amplification ``A`` and the downstream ice/water
drag and free-drift response -- is obtained from that channel.  Coefficients
calibrated *inside* the defect window would be contaminated.  The audit is a
paired experiment: the same cases are run with the fixed engine and with a
defect-window engine, and the relative deltas of the calibrated quantities are
reported with a confidence interval.  ``DELTA_ACCEPT`` (1 %) is the acceptance
line: below it the turbulent calibration is declared uncontaminated (an
acceptable negative result), above it a re-calibration table is mandatory.

Engine variants
---------------
``fixed``
    ``HEAD`` source.
``defect_single_factor``
    ``HEAD`` source with *only* the forcing weight reverted (isolates the
    single factor; used for the cheap anchors).
``defect_window``
    the source as it was at the defect-window commit (``git show``), i.e. the
    engine that actually produced the archived runs.  Used for the negative
    control: an audit that cannot see a defect that is demonstrably there has
    no discriminating power.

Nothing here hard-codes a physics number: the deltas come from the engine
records, the confidence interval from the replicate seeds.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .. import rough_channel3d as rc3
from ..rough_channel3d_native import (
    NativeBackendUnavailable,
    run_rough_channel_native,
)
from . import freedrift as fd
from . import propagation as pr
from . import regime_law as rl
from . import sweep as sw

__all__ = [
    "BLOCKING_EXPR_FIXED",
    "BLOCKING_EXPR_DEFECT",
    "DEFECT_INTRO_COMMIT",
    "FIX_COMMIT",
    "DELTA_ACCEPT",
    "VARIANTS",
    "build_engine",
    "contamination",
    "drive_factor",
    "engine_sha256",
    "laminar_anchor",
    "native_source",
    "quantities",
    "tau_for_drive_factor",
]

#: The two textual forms of the Guo-force weight in ``native/rough_channel3d.c``.
BLOCKING_EXPR_FIXED = "force * (1.0 - 0.5 * s_minus)"
BLOCKING_EXPR_DEFECT = "force * (1.0 - 0.5 * s_plus)"

#: Git provenance of the defect window (filled by ``git log -S``; see the
#: ``provenance`` block of the audit JSON for the command output).
DEFECT_INTRO_COMMIT = "d5d4a421"
FIX_COMMIT = "ec046ace"

#: Acceptance line of the audit: |delta| <= 1 % means "not contaminated".
DELTA_ACCEPT = 0.01

VARIANTS = ("fixed", "defect_single_factor", "defect_window")

#: Ice-keel geometric roughness the downstream drag is evaluated at (the
#: reference condition of the drift back-fill, ``docs/ICE_ROUGH_CW_CLOSURE.md``
#: §4 C4: ``k_s = 0.35 m``, ``u_rel = 0.15 m/s``).
K_S_REFERENCE_M: float = 0.35

_SRC = Path(__file__).resolve().parent.parent / "native" / "rough_channel3d.c"
_REPO = _SRC.parent.parent.parent


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def _git_show(rev: str, path: str) -> str:
    out = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=str(_REPO), capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git show {rev}:{path} failed: {out.stderr[-400:]}")
    return out.stdout


def native_source(variant: str) -> str:
    """C source of ``variant`` (see module docstring)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown engine variant {variant!r}")
    if variant == "defect_window":
        return _git_show(DEFECT_INTRO_COMMIT, "src/tensorlbm/native/rough_channel3d.c")
    text = _SRC.read_text()
    if variant == "fixed":
        if text.count(BLOCKING_EXPR_FIXED) != 1:
            raise RuntimeError(
                "the fixed forcing weight is not present exactly once in "
                f"{_SRC} -- the audit is pinned to a form that no longer exists",
            )
        return text
    # defect_single_factor: revert exactly one token, everything else HEAD.
    if text.count(BLOCKING_EXPR_FIXED) != 1:
        raise RuntimeError("cannot revert a forcing weight that is not unique")
    return text.replace(BLOCKING_EXPR_FIXED, BLOCKING_EXPR_DEFECT)


def engine_sha256(variant: str) -> str:
    return hashlib.sha256(native_source(variant).encode()).hexdigest()


def _compile_flags() -> list[str]:
    env = os.environ.get("TENSORLBM_NATIVE_CFLAGS")
    if env is not None and env.strip():
        return env.split()
    return ["-O3", "-funroll-loops", "-fopenmp"]


def build_engine(variant: str, *, cache_dir: str | Path | None = None) -> Path:
    """Compile ``variant`` (cached on source + flag hash) and return the binary."""
    import shutil

    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        raise NativeBackendUnavailable("no C compiler (gcc/cc) on PATH")
    root = Path(cache_dir or os.environ.get("TENSORLBM_NATIVE_CACHE")
                or os.path.join(tempfile.gettempdir(), "tensorlbm_native"))
    root.mkdir(parents=True, exist_ok=True)
    text = native_source(variant)
    tag = hashlib.sha1((" ".join(_compile_flags()) + text).encode()).hexdigest()[:10]
    out = root / f"rough_channel3d_{variant}_{tag}"
    if out.is_file():
        return out
    src = root / f"rough_channel3d_{variant}_{tag}.c"
    src.write_text(text)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    proc = subprocess.run(
        [compiler, *_compile_flags(), "-o", str(tmp), str(src), "-lm"],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise NativeBackendUnavailable(f"compile failed: {proc.stderr[-800:]}")
    os.replace(tmp, out)
    return out


# ---------------------------------------------------------------------------
# Closed-form defect signature
# ---------------------------------------------------------------------------

def _s_minus(tau: float) -> float:
    return 1.0 / (rc3.TRT_LAMBDA / (float(tau) - 0.5) + 0.5)


def drive_factor(tau: float) -> float:
    """Momentum actually supplied per step, in units of the body force ``F``.

    The collision subtracts ``feq`` in both TRT modes, so with the *wrong*
    weight ``(1 - s_plus/2)`` the antisymmetric mode carries an extra
    ``-(s_minus - s_plus) F/2``: the channel is driven by
    ``F [1 + (s_minus - s_plus)/2]``.  The correctly weighted engine supplies
    ``F`` exactly (``drive_factor`` is then 1 by construction), which is what
    the laminar anchor measures.
    """
    tau = float(tau)
    if not tau > 0.5:
        raise ValueError("drive_factor needs tau > 0.5")
    s_plus = 1.0 / tau
    return 1.0 + 0.5 * (_s_minus(tau) - s_plus)


def tau_for_drive_factor(target: float, *, lo: float = 0.5001, hi: float = 50.0) -> float:
    """Relaxation time at which the defective engine supplies exactly ``target*F``.

    The seeding specification quotes the defect signature as ``F/2``; that is
    a property of one particular ``tau``, not of the defect (which is a
    tau-dependent rescaling, see :func:`drive_factor`).  This helper locates
    ``tau`` so the claim can be tested at machine precision.
    """
    from scipy.optimize import brentq  # local import: only needed by the audit

    return float(brentq(lambda t: drive_factor(t) - float(target), lo, hi, xtol=1e-14))


# ---------------------------------------------------------------------------
# Laminar anchor (defect signature at engine level)
# ---------------------------------------------------------------------------

def laminar_anchor(
    variant: str,
    *,
    binary: Path | str | None = None,
    re_tau: float = 10.0,
    h: int = 10,
    u_tau: float = 0.02,
    steps: int = 17000,
    threads: int = 2,
    init_scale: float = 0.5,
) -> dict[str, Any]:
    """Forced-channel laminar Poiseuille anchor: ``U_b^+ -> Re_tau/3``.

    Returns ``u_plus_bulk``, the ratio to the analytic value and the closed-form
    :func:`drive_factor` of the same case, so the measured and the analytic
    signature can be compared instead of conflated.
    """
    exe = Path(binary) if binary is not None else build_engine(variant)
    cfg = rc3.RoughChannelConfig(
        nx=h, ny=h, nz=2 * h + 2, re_tau=float(re_tau), u_tau=float(u_tau),
        steps=int(steps), warmup=int(steps) // 3, sample_interval=10,
        threads=int(threads), use_compile=False, init_scale=float(init_scale),
        noise_amplitude=0.0, smagorinsky_cs=0.0,
    )
    solid, _kind, _ks, _geom = rc3.geometry_and_mask(cfg, None)
    const = rc3.case_constants(cfg, solid)
    rec = run_rough_channel_native(cfg, None, smooth_b_reference=5.2, binary=exe)
    analytic = float(re_tau) / 3.0
    return {
        "engine": variant,
        "engine_sha256": engine_sha256(variant),
        "re_tau": float(re_tau),
        "h": int(h),
        "u_tau": float(u_tau),
        "steps": int(steps),
        "tau": float(const["tau"]),
        "u_plus_bulk": float(rec["u_plus_bulk"]),
        "u_plus_bulk_analytic": analytic,
        "ratio_to_analytic": float(rec["u_plus_bulk"]) / analytic,
        "drive_factor_closed_form": drive_factor(float(const["tau"])),
        "cf": float(rec["cf"]),
    }


# ---------------------------------------------------------------------------
# Turbulent campaign cases (paired fixed / defect-window recompute)
# ---------------------------------------------------------------------------

def case_record_id(case: sw.Case, variant: str) -> str:
    return f"{case.case_id}__{variant}__s{case.seed}"


def campaign_case_set(
    *,
    ks_targets: Sequence[float] = (20.0, 45.0, 90.0),
    lambda_over_ks: float = 5.0,
    seeds: Sequence[int] = (20261109, 20261110),
    profile: str = "compact",
) -> list[sw.Case]:
    """Rough cases plus the smooth baselines at their ``Re_tau`` (as in ``sweep``).

    The smooth baseline is a *per-seed* control: the pairing in
    :func:`quantities` is per seed, so a seed without its own baseline would
    silently borrow another seed's one (fixed in m11525 -- with a single shared
    ``seen_re`` set the second and later seeds got no baseline at all).
    """
    cases: list[sw.Case] = []
    for seed in seeds:
        seen_re: dict[float, bool] = {}
        for ks in ks_targets:
            cases.append(sw.Case(case_id=sw._case_id(ks, lambda_over_ks),
                                 ks_plus_target=float(ks),
                                 lambda_over_ks=float(lambda_over_ks),
                                 profile=profile, seed=int(seed)))
            rough = sw.Case(case_id=sw._case_id(ks, lambda_over_ks),
                            ks_plus_target=float(ks), lambda_over_ks=float(lambda_over_ks),
                            profile=profile, seed=int(seed))
            rt = round(rough.re_tau, 3)
            if rt not in seen_re:
                seen_re[rt] = True
                cases.append(sw.Case(case_id=f"smooth_re{rt:g}", ks_plus_target=rt,
                                     lambda_over_ks=1.0, profile=profile,
                                     kind="smooth", seed=int(seed)))
    return cases


def run_one(case: sw.Case, variant: str, out_dir: str | Path, *,
            threads: int | None = None) -> dict[str, Any]:
    """Run one case with ``variant`` and write ``<out_dir>/<case>__<variant>.json``."""
    exe = build_engine(variant)
    cfg = case.config(threads=threads)
    elements = case.elements()
    rec = run_rough_channel_native(cfg, elements, binary=exe)
    rec["case"] = case.to_json()
    rec["engine"] = variant
    rec["engine_sha256"] = engine_sha256(variant)
    rec["ks_nodes"] = float(case.ks_nodes)
    rec["dy_plus"] = float(rec["re_tau"]) / float(rec["h_eff"])
    rec["wall_resolved"] = bool(rec["dy_plus"] <= 1.5)
    path = Path(out_dir) / f"{case_record_id(case, variant)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_plain(rec), allow_nan=True, indent=1))
    os.replace(tmp, path)
    return rec


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_records(run_dir: str | Path, variant: str | None = None) -> dict[str, dict[str, Any]]:
    """Load ``<case>__<variant>__s<seed>.json`` records, keyed by that stem."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(run_dir).glob("*.json")):
        stem = path.stem
        if variant is not None and f"__{variant}__" not in stem:
            continue
        out[stem] = json.loads(path.read_text())
    return out


def quantities(records: Mapping[str, Mapping[str, Any]], *,
               lambda_over_ks: float,
               ks_targets: Sequence[float],
               u_rel: float = 0.15,
               drift_env: fd.DriftEnvironment | None = None) -> dict[str, Any]:
    """Calibrated quantities of one engine: ``k0``, ``C_w``, ``cf_ratio``, ``A``, drag.

    The per-seed values are kept so the caller can form the paired delta and its
    confidence interval; ``k0`` is fitted exactly as the campaign does
    (``regime_law.fit_k0`` on the log-intercepts, smooth baselines matched by
    ``Re_tau``).
    """
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    smooth_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for stem, rec in records.items():
        seed = int(rec.get("case", {}).get("seed", 0))
        kind = rec.get("case", {}).get("kind", "rough")
        (smooth_by_seed if kind == "smooth" else by_seed).setdefault(seed, {})[stem] = rec

    per_seed: list[dict[str, Any]] = []
    for seed in sorted(by_seed):
        rseed = by_seed[seed]
        sseed = smooth_by_seed.get(seed) or (next(iter(smooth_by_seed.values()), {}) if smooth_by_seed else {})
        rows: dict[str, Any] = {}
        ks_used: list[float] = []
        du_used: list[float] = []
        for ks in ks_targets:
            matches = [r for r in rseed.values()
                       if abs(float(r["case"]["ks_plus_target"]) - float(ks)) < 1e-9
                       and abs(float(r["case"]["lambda_over_ks"]) - float(lambda_over_ks)) < 1e-9]
            if not matches:
                continue
            rec = matches[0]
            smooth = sw._smooth_reference(sseed, float(rec["re_tau"]))
            du = sw._delta_u_plus(rec, smooth)
            rows[f"{ks:g}"] = {
                "ks_plus_target": float(ks),
                "re_tau": float(rec["re_tau"]),
                "u_plus_bulk": float(rec["u_plus_bulk"]),
                "cf": float(rec["cf"]),
                "cf_smooth": float(smooth.get("cf", float("nan"))),
                "dU_plus": float(du["dU_plus_used"]),
                "dU_plus_measure": du["dU_plus_measure"],
                "log_window_valid": bool(du["log_window_valid"]),
            }
            ks_used.append(float(ks))
            du_used.append(float(du["dU_plus_used"]))
        fit = rl.fit_k0(ks_used, du_used) if len(ks_used) >= 2 else {"k0": float("nan")}
        k0 = float(fit["k0"])
        if not np.isfinite(k0) and ks_used:
            # A single usable point cannot be fitted, but it still pins the law
            # through the k0 inversion (regime_law.k0_from_delta_u_plus).  This
            # keeps the downstream drag of a partly-diverged engine measurable
            # instead of dropping it; the point count is carried in the record.
            try:
                k0 = float(rl.k0_from_delta_u_plus(ks_used[0], du_used[0]))
                fit = {"k0": k0, "k0_source": "single-point inversion",
                       "n_points": 1, "single_point_ks_plus": float(ks_used[0])}
            except ValueError as exc:
                # a non-finite / sign-flipped dU^+ is a *result* (the defective
                # engine can report a negative roughness function); it must be
                # recorded as an unusable point, not crash the aggregate.
                k0 = float("nan")
                fit = {"k0": k0, "k0_source": f"not invertible: {exc}",
                       "n_points": 0, "single_point_ks_plus": float(ks_used[0])}
        law = rl.ThreeZoneLaw(lambda_over_ks=float(lambda_over_ks), k0=k0,
                              k0_source="audit-refit", n_points=int(fit.get("n_points", 0)))
        for key, row in rows.items():
            ks = row["ks_plus_target"]
            try:
                cf_ratio = law.cf_ratio(ks, row["cf_smooth"])
            except ValueError:
                cf_ratio = float("nan")
            row["cf_ratio"] = float(cf_ratio)
            row["cf_ratio_lbm"] = float(row["cf"] / row["cf_smooth"]) if row["cf_smooth"] else float("nan")
            # Clauser-matched amplification of the *drag coefficient* that the
            # drift model consumes: A = Cf_rough/Cf_smooth at the same u_tau,
            # and the ice/water drag coefficient C_w = C_w,smooth * A.
            row["cw_amplification"] = float(row["cf_ratio"])
            row["C_w"] = float(fd.CW_SMOOTH_REFERENCE) * float(row["cf_ratio"])
            row["C_w_lbm"] = (float(fd.CW_SMOOTH_REFERENCE) * float(row["cf_ratio_lbm"])
                              if np.isfinite(row["cf_ratio_lbm"]) else float("nan"))
        per_seed.append({"seed": seed, "k0": k0, "k0_fit": fit, "points": rows})

    # downstream application: ice/water drag coefficient at the reference
    # keel roughness and the free-drift response it produces (this is the
    # quantity an ice-drift model actually consumes).
    env = drift_env or fd.DriftEnvironment()
    downstream: list[dict[str, Any]] = []
    for entry in per_seed:
        law = rl.ThreeZoneLaw(lambda_over_ks=float(lambda_over_ks), k0=entry["k0"],
                              k0_source="audit-refit")
        row: dict[str, Any] = {"seed": entry["seed"], "k_s_m": K_S_REFERENCE_M,
                               "u_rel_m_s": float(u_rel)}
        try:
            table = pr.cw_table({float(lambda_over_ks): law}, ks_m=(K_S_REFERENCE_M,),
                                lambda_over_ks=(float(lambda_over_ks),),
                                u_rel=u_rel, z_ref_m=env.z_ref, nu=env.nu_water)
            row.update({k: float(v) for k, v in table[0].items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)})
            row["C_w"] = float(table[0]["C_w"])
            row["C_w_amplification"] = float(table[0]["amplification_over_smooth"])
            row["ice_water_drag_Pa"] = float(env.rho_water * row["C_w"] * u_rel**2)
            drift = fd.drift_with_roughness_amplification(env, K_S_REFERENCE_M, law)
            row["drift_speed_m_s"] = float(drift.get("drift_speed_m_s", float("nan")))
            row["drift_angle_deg"] = float(drift.get("drift_angle_deg", float("nan")))
            row["drift_C_w"] = float(drift.get("C_w", float("nan")))
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            row["error"] = str(exc)
        downstream.append(row)
    return {"lambda_over_ks": float(lambda_over_ks), "per_seed": per_seed,
            "downstream": downstream}


# ---------------------------------------------------------------------------
# Paired comparison / verdict
# ---------------------------------------------------------------------------

#: Two-sided 95 % Student-t quantiles, ``t_{0.975, dof}``.  A paired delta with
#: ``n`` replicate seeds has ``dof = n - 1``; with the small ``n`` an audit can
#: afford the normal quantile 1.96 is *not* a 95 % interval, so the quantile is
#: taken from the table (mission m11525: the previous pass quoted 1.96 with
#: n = 2, which is a 1.6-sigma interval in disguise).
STUDENT_T_975: dict[int, float] = {
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469,
    7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788,
    13: 2.1604, 14: 2.1448, 15: 2.1314, 16: 2.1199, 17: 2.1098, 18: 2.1009,
    19: 2.0930, 20: 2.0860, 21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639,
    25: 2.0595, 26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452, 30: 2.0423,
}


def t_critical_975(dof: int) -> float:
    """Two-sided 95 % Student-t quantile (exact table for ``dof <= 30``)."""
    dof = int(dof)
    if dof < 1:
        raise ValueError("t_critical_975 needs dof >= 1")
    if dof in STUDENT_T_975:
        return STUDENT_T_975[dof]
    # Cornish-Fisher expansion about the normal quantile for large dof
    z = 1.959963985
    return z + (z**3 + z) / (4.0 * dof) + (5 * z**5 + 16 * z**3 + 3 * z) / (96.0 * dof**2)


def _summarise(values: Sequence[float]) -> dict[str, float]:
    """Mean, sd and the *t*-based 95 % CI half-width of a sample of deltas."""
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "ci95_half_width": float("nan"),
                "n": 0, "dof": 0, "t_critical_975": float("nan"),
                "ci_method": "student-t", "ci95": [float("nan"), float("nan")]}
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    dof = int(vals.size - 1)
    tcrit = t_critical_975(dof) if dof >= 1 else float("nan")
    half = float(tcrit * sd / math.sqrt(vals.size)) if dof >= 1 else float("nan")
    return {"mean": mean, "sd": sd, "dof": dof, "t_critical_975": float(tcrit),
            "ci95_half_width": half, "n": int(vals.size), "ci_method": "student-t",
            "ci95": [mean - half, mean + half] if half == half else [float("nan")] * 2}


def contamination(fixed: Mapping[str, Any], defect: Mapping[str, Any], *,
                  accept: float = DELTA_ACCEPT) -> dict[str, Any]:
    """Relative delta (defect vs fixed) of every calibrated quantity, with 95 % CI.

    Deltas are formed *pairwise per seed* (the seeds are the only stochastic
    degree of freedom), so the CI is the standard error of the paired
    differences, not a difference of two independent sample means.
    """
    fx = {e["seed"]: e for e in fixed["per_seed"]}
    df = {e["seed"]: e for e in defect["per_seed"]}
    seeds = sorted(set(fx) & set(df))

    quantities_out: dict[str, Any] = {}

    def _add(name: str, getter) -> None:
        deltas = []
        pairs = []
        for seed in seeds:
            a, b = getter(df[seed]), getter(fx[seed])
            if a is None or b is None:
                continue
            if not (np.isfinite(a) and np.isfinite(b)) or b == 0.0:
                continue
            d = (a - b) / abs(b)
            deltas.append(float(d))
            pairs.append({"seed": seed, "defect": float(a), "fixed": float(b), "rel_delta": float(d)})
        stats = _summarise(deltas)
        stats["pairs"] = pairs
        stats["evaluable"] = bool(stats["n"] > 0)
        stats["abs_mean"] = abs(stats["mean"]) if stats["evaluable"] else float("nan")
        stats["exceeds_accept"] = bool(stats["evaluable"] and stats["abs_mean"] > accept)
        quantities_out[name] = stats

    _add("k0", lambda e: float(e.get("k0", float("nan"))))
    # NOTE: the amplification ``A = Cf_rough/Cf_smooth`` is reported per ks as
    # ``ks<k>.cf_ratio`` (law-based) and ``cf_ratio_lbm`` (direct ratio).
    ks_keys = sorted({k for e in fixed["per_seed"] for k in e["points"]},
                     key=lambda s: float(s))
    for key in ks_keys:
        for field in ("dU_plus", "cf", "cf_ratio", "cf_ratio_lbm", "C_w", "C_w_lbm"):
            _add(f"ks{key}.{field}",
                 lambda e, k=key, f=field: float(e["points"].get(k, {}).get(f, float("nan"))))
    dmap_f = {d["seed"]: d for d in fixed.get("downstream", [])}
    dmap_d = {d["seed"]: d for d in defect.get("downstream", [])}
    for field in ("C_w", "C_w_amplification", "ice_water_drag_Pa",
                  "drift_speed_m_s", "drift_angle_deg"):
        def _get(e, f=field):
            if f not in e:
                return None
            return float(e[f])
        deltas, pairs = [], []
        for seed in sorted(set(dmap_f) & set(dmap_d)):
            a, b = _get(dmap_d[seed]), _get(dmap_f[seed])
            if a is None or b is None or b == 0.0:
                continue
            d = (a - b) / abs(b)
            deltas.append(float(d))
            pairs.append({"seed": seed, "defect": a, "fixed": b, "rel_delta": float(d)})
        stats = _summarise(deltas)
        stats["pairs"] = pairs
        stats["evaluable"] = bool(stats["n"] > 0)
        stats["abs_mean"] = abs(stats["mean"]) if stats["evaluable"] else float("nan")
        stats["exceeds_accept"] = bool(stats["evaluable"] and stats["abs_mean"] > accept)
        quantities_out[f"downstream.{field}"] = stats

    k0_delta = quantities_out["k0"]["mean"] if np.isfinite(quantities_out["k0"]["mean"]) else float("nan")
    exceed = [k for k, v in quantities_out.items() if v["exceeds_accept"]]
    evaluable = [k for k, v in quantities_out.items() if v["evaluable"]]
    turbine = [k for k in evaluable if k.startswith("ks")]
    # "clean" needs evidence: an empty comparison is never reported as clean.
    turbulent_clean = bool(evaluable) and not exceed and all(
        not quantities_out[k]["exceeds_accept"] for k in turbine
        if k.endswith(("cf_ratio", "cf_ratio_lbm", "dU_plus")))
    return {
        "accept_threshold": float(accept),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "quantities": quantities_out,
        "k0_rel_delta": k0_delta,
        "exceeding_quantities": exceed,
        "turbulent_calibration_clean": bool(turbulent_clean),
        "contaminated": bool(len(exceed) > 0 or not evaluable),
        "evaluable_quantities": len(evaluable),
    }

# ---------------------------------------------------------------------------
# Recalibration / state-shift accounting (mission m11525)
# ---------------------------------------------------------------------------

def recalibration_table(verdict: Mapping[str, Any], *,
                        accept: float = DELTA_ACCEPT) -> dict[str, Any]:
    """Map every *contaminated* calibrated quantity onto its corrected value.

    ``verdict`` is the output of :func:`contamination`, whose ``pairs`` carry
    the raw (defect, fixed) values per seed.  The table answers the audit
    question directly: for a coefficient reported by the defect-window engine,
    which multiplicative correction recovers the fixed-engine value, and is
    that correction larger than the acceptance line?  A quantity without
    paired evidence is listed as ``NOT_EVALUABLE`` -- never as "no correction
    needed".
    """
    rows: list[dict[str, Any]] = []
    for name, stats in sorted(verdict.get("quantities", {}).items()):
        pairs = stats.get("pairs") or []
        if not stats.get("evaluable") or not pairs:
            rows.append({"quantity": name, "status": "NOT_EVALUABLE",
                         "n_pairs": 0, "recalibration_required": None})
            continue
        defect = float(np.mean([p["defect"] for p in pairs]))
        fixed = float(np.mean([p["fixed"] for p in pairs]))
        factor = (fixed / defect) if defect not in (0.0,) else float("nan")
        rel = float(stats["mean"])
        rows.append({
            "quantity": name,
            "status": "EVALUATED",
            "n_pairs": int(stats["n"]),
            "defect_mean": defect,
            "fixed_mean": fixed,
            "rel_delta": rel,
            "ci95": stats.get("ci95"),
            "correction_factor_fixed_over_defect": float(factor),
            "recalibration_required": bool(abs(rel) > accept or
                                           (abs(rel) - float(stats.get("ci95_half_width") or 0.0)) > accept),
        })
    required = [r["quantity"] for r in rows if r.get("recalibration_required")]
    unknown = [r["quantity"] for r in rows if r["status"] != "EVALUATED"]
    return {
        "accept_threshold": float(accept),
        "rows": rows,
        "recalibration_required": required,
        "not_evaluable": unknown,
        "verdict": ("RECALIBRATE" if required else
                    ("INCONCLUSIVE" if unknown else "NO_RECALIBRATION")),
    }


def engine_state_shift(records_fixed: Mapping[str, Mapping[str, Any]],
                       records_defect: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """How far the *reported* state of a case moves between the two engines.

    The forcing weight does not only rescale ``C_f``: with the wrong weight the
    momentum supplied is ``D(tau) F``, so the reported friction velocity
    ``sqrt(g h_eff)`` is inflated, the reported ``U^+`` deflated and the
    measured ``k_s^+ = k_s u_tau/nu`` shifted.  This function makes that shift
    explicit per case (it is the reason a nominal-case pairing is *not* a
    matched-physics pairing and must be read with the shift at hand).
    """
    def _key(rec: Mapping[str, Any]) -> tuple[str, int]:
        case = rec.get("case", {}) if isinstance(rec, Mapping) else {}
        return (str(case.get("case_id")), int(case.get("seed", 0)))

    fx = {_key(r): r for r in records_fixed.values()}
    df = {_key(r): r for r in records_defect.values()}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(fx) & set(df)):
        a, b = df[key], fx[key]
        if a.get("case", {}).get("kind") == "smooth":
            continue
        ks_target = float(a["case"]["ks_plus_target"])

        def _ratio(x: float, y: float) -> float:
            return float(x) / float(y) if y else float("nan")

        rows.append({
            "case_id": key[0], "seed": key[1], "ks_plus_target": ks_target,
            "ks_plus_measured_fixed": float(b.get("ks_plus", float("nan"))),
            "ks_plus_measured_defect": float(a.get("ks_plus", float("nan"))),
            "ks_plus_ratio_defect_over_fixed": _ratio(float(a.get("ks_plus", float("nan"))),
                                                      float(b.get("ks_plus", float("nan")))),
            "u_plus_ratio_defect_over_fixed": _ratio(float(a["u_plus_bulk"]), float(b["u_plus_bulk"])),
            "cf_ratio_defect_over_fixed": _ratio(float(a["cf"]), float(b["cf"])),
            "tau_defect": float(a.get("tau", float("nan"))),
            "drive_factor_closed_form_defect": drive_factor(float(a["tau"])) if a.get("tau", 0.0) > 0.5 else float("nan"),
            "wall_resolved_fixed": bool(b.get("wall_resolved", False)),
            "wall_resolved_defect": bool(a.get("wall_resolved", False)),
        })
    by_ks: dict[str, Any] = {}
    for row in rows:
        by_ks.setdefault(f"{row['ks_plus_target']:g}", []).append(row)
    summary = {}
    for ks, group in by_ks.items():
        def _m(field: str) -> float:
            vals = [g[field] for g in group if np.isfinite(g[field])]
            return float(np.mean(vals)) if vals else float("nan")
        summary[ks] = {
            "n": len(group),
            "ks_plus_ratio_mean": _m("ks_plus_ratio_defect_over_fixed"),
            "u_plus_ratio_mean": _m("u_plus_ratio_defect_over_fixed"),
            "cf_ratio_mean": _m("cf_ratio_defect_over_fixed"),
            "implied_drive_factor_from_cf": (1.0 / _m("cf_ratio_defect_over_fixed")
                                            if _m("cf_ratio_defect_over_fixed") else float("nan")),
            # the *true* friction velocity of the defect run is smaller by
            # sqrt(D); the reported k_s^+ label is unchanged by construction,
            # so this is the only honest "state shift" scalar
            "true_u_tau_ratio_defect_over_fixed": (float(np.sqrt(1.0 / _m("cf_ratio_defect_over_fixed")))
                                                   if _m("cf_ratio_defect_over_fixed") else float("nan")),
            "closed_form_drive_factor_defect": _m("drive_factor_closed_form_defect"),
        }
    return {"rows": rows, "by_ks": summary,
            "note": ("a nominal-case pairing is not a matched-physics pairing: the "
                     "defective engine reports a shifted friction velocity, hence a "
                     "shifted U^+ / k_s^+ for the same input k_s^+; the shift is "
                     "reported per k_s^+ so the paired delta can be read correctly")}
