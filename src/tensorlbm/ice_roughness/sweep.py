"""Ice-keel roughness campaign: case table, runner and criteria adjudication.

The campaign is a Nikuradse-style experiment in the lattice: **one fixed
geometry per spacing ratio is swept in Reynolds number**, so that
``dU^+(k_s^+)`` is a clean function of ``k_s^+`` alone.  Concretely, every case
of a ``lambda/k_s`` group uses the same rib height and pitch in lattice units
(``k_s = 6`` nodes, ``pitch = 6 lambda/k_s`` nodes) and reaches the target
``k_s^+`` by setting ``Re_tau = k_s^+ h_eff / k_s``.  That is the only design
in which the fitted onset parameter ``k0`` is a property of the morphology
rather than of the grid.

Cases
-----
``k_s^+ = 20, 45, 90, 200, 500``  x  ``lambda/k_s = 2, 5, 10``  -> 15 rough cases
plus a smooth-wall baseline at every ``Re_tau`` used (5 cases).  The smooth
baselines serve two purposes: criterion C1 (solver accuracy against the
DNS-backed channel correlation) and the ``dU^+`` reference level (a matched
smooth run removes the grid-dependent bias of the fitted log-law intercept).

Resolution bookkeeping
----------------------
Each record carries ``dy_plus`` and ``ks_nodes``.  ``k_s^+/dy^+ = ks_nodes``,
so a resolved geometry needs ``ks_nodes >= ~4`` and the *crest* boundary layer
needs ``dy^+ << k_s^+``; the high-``k_s^+`` end of the sweep is therefore
geometry-resolved but wall-under-resolved, which is reported per case
(``log_window_valid``, ``wall_resolved``) instead of being hidden.

Honest scoping note
-------------------
``k_s^+ = 500`` in a channel with ``k_s/h <= 0.45`` forces ``Re_tau >= 1100``
and then ``3 k_s^+ > 0.35 Re_tau``: *no log layer can exist*.  Those cases are
still measured (``Cf`` from the momentum balance is exact) but their ``dU^+``
falls back to the bulk (Hama/Clauser) definition and they are excluded from the
log-layer fit; the report says so explicitly.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .. import rough_channel3d as rc3
from .. import roughness_elements as re
from ..rough_channel3d_native import run_rough_channel_native
from . import geometry as ice_geom
from . import regime_law as rl

__all__ = [
    "PROFILES",
    "KS_PLUS_TARGETS",
    "Case",
    "campaign_cases",
    "run_case",
    "load_records",
    "aggregate",
]

#: ``k_s^+`` targets of the sweep (mission: 5 ... 500).
KS_PLUS_TARGETS: tuple[float, ...] = (20.0, 45.0, 90.0, 200.0, 500.0)

#: Numerical profiles.  ``campaign`` is the reported sweep; ``smoke`` is the
#: reduced-size twin used by the test-suite and by the pipeline check.
PROFILES: dict[str, dict[str, Any]] = {
    "campaign": {
        "h": 48,             # half channel height in lattice units
        # 5 nodes per roughness height keeps k_s/h = 5/47 = 0.106, i.e.
        # 3 k_s^+ = 0.318 Re_tau < 0.35 Re_tau: the log layer the roughness
        # function is extracted from exists for every case.  (With 6 nodes the
        # window closes for the whole sweep -- measured, 2026-11-09.)
        "ks_nodes": 5,
        "u_tau": 0.012,      # bulk Mach ~0.14 -> compressibility error ~2%
        "turnovers": 4.0,    # total = warmup + sampling, in units of 2h/u_tau
        "warmup_fraction": 0.25,
        "init_scale": 1.0,   # start AT the log-law equilibrium (see below)
        "sample_interval": 10,
        "smagorinsky_cs": 0.10,
        "threads": 16,
    },
    #: Reduced-cost twin of ``campaign`` for shared / heavily loaded hosts:
    #: same grid and geometry, three eddy turnovers instead of four.
    "compact": {
        "h": 48,
        "ks_nodes": 5,
        "u_tau": 0.012,
        "turnovers": 3.0,
        "warmup_fraction": 0.334,
        "init_scale": 1.0,
        "sample_interval": 10,
        "smagorinsky_cs": 0.10,
        "threads": 18,
    },
    "smoke": {
        "h": 24,
        "ks_nodes": 3,
        "u_tau": 0.02,
        "turnovers": 0.5,
        "warmup_fraction": 0.3,
        "init_scale": 1.0,
        "sample_interval": 20,
        "smagorinsky_cs": 0.10,
        "threads": 4,
    },
}


@dataclass(frozen=True)
class Case:
    """One lattice experiment: fixed geometry, target ``k_s^+`` via ``Re_tau``."""

    case_id: str
    ks_plus_target: float
    lambda_over_ks: float
    profile: str = "campaign"
    kind: str = "rough"            # "rough" | "smooth"
    morphology: str = "ribs"        # "ribs" (ice keels) | "sand" (Nikuradse)
    seed: int = 20261109
    phase: int = 0

    @property
    def params(self) -> Mapping[str, Any]:
        return PROFILES[self.profile]

    @property
    def h(self) -> int:
        return int(self.params["h"])

    @property
    def nz(self) -> int:
        return 2 * self.h + 2

    @property
    def h_eff_estimate(self) -> float:
        return float(self.h - 1)

    @property
    def ks_nodes(self) -> float:
        return float(self.params["ks_nodes"])

    @property
    def re_tau(self) -> float:
        if self.kind == "smooth":
            return float(self.ks_plus_target)
        return float(self.ks_plus_target) * self.h_eff_estimate / self.ks_nodes

    @property
    def u_tau(self) -> float:
        return float(self.params["u_tau"])

    # NOTE on ``init_scale``: a forced channel starting below its equilibrium
    # bulk velocity accelerates at du_b/dt = u_tau^2/h_eff per step, i.e. it
    # needs ~10 eddy turnovers to climb from 0.5 U_b^+ to the equilibrium
    # U_b^+ ~ 18.  Sweeps that sample after 1-2 turnovers therefore report the
    # *initial* profile, not a steady state (measured: U_b^+ = 9.3 at 1.6
    # turnovers vs 18.3 at equilibrium, Cf inflated 4x).  Starting on the
    # log-law equilibrium (init_scale = 1) plus >= 4 turnovers for the
    # statistics removes that bias.

    @property
    def steps(self) -> int:
        total = float(self.params["turnovers"]) * 2.0 * self.h / self.u_tau
        return max(200, int(round(total)))

    @property
    def warmup(self) -> int:
        return max(100, int(round(float(self.params["warmup_fraction"]) * self.steps)))

    def config(self, *, threads: int | None = None) -> rc3.RoughChannelConfig:
        p = self.params
        return rc3.RoughChannelConfig(
            nx=self.h, ny=self.h, nz=self.nz,
            re_tau=self.re_tau, u_tau=self.u_tau,
            steps=self.steps, warmup=self.warmup,
            sample_interval=int(p["sample_interval"]),
            seed=self.seed, threads=int(threads or p["threads"]),
            use_compile=False, init_scale=float(p["init_scale"]), noise_amplitude=0.03,
            smagorinsky_cs=float(p["smagorinsky_cs"]),
            note=self.case_id,
        )

    def elements(self) -> re.RoughnessElementSet | None:
        if self.kind == "smooth":
            return None
        if self.morphology == "sand":
            return re.sand_grain_elements(
                self.h, self.h, self.nz, self.ks_nodes, seed=self.seed
            )
        return ice_geom.rib_elements_for_case(
            self.h, self.h, self.nz, self.ks_nodes, self.lambda_over_ks, phase=self.phase
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "morphology": self.morphology,
            "profile": self.profile,
            "ks_plus_target": float(self.ks_plus_target),
            "lambda_over_ks": float(self.lambda_over_ks),
            "re_tau": float(self.re_tau),
            "h": self.h, "nz": self.nz,
            "ks_nodes": float(self.ks_nodes),
            "u_tau": float(self.u_tau),
            "steps": int(self.steps),
            "warmup": int(self.warmup),
            "seed": int(self.seed),
        }


def _case_id(ks_plus: float, lam: float) -> str:
    return f"ribs_ks{ks_plus:g}_lam{lam:g}"


def campaign_cases(profile: str = "campaign", *, ratios: Sequence[float] | None = None,
                   ks_plus_targets: Sequence[float] | None = None) -> list[Case]:
    """Case table.  ``compact`` runs the same design with a reduced target list."""
    """The full case table (rough cases first, smooth baselines appended)."""
    ratios = tuple(ice_geom.LAMBDA_OVER_KS_LEVELS if ratios is None else ratios)
    targets = tuple(KS_PLUS_TARGETS if ks_plus_targets is None else ks_plus_targets)
    cases: list[Case] = []
    for lam in ratios:
        for ks in targets:
            cases.append(Case(case_id=_case_id(ks, lam), ks_plus_target=ks,
                              lambda_over_ks=float(lam), profile=profile))
    # Nikuradse sand-grain control group: same k_s^+ targets and the same
    # k_s/h as the rib groups, so its roughness function can be compared with
    # the sand-grain law that *defines* criterion C2 without assuming that an
    # ice keel behaves like a sand grain.
    for ks in targets:
        cases.append(Case(case_id=f"sand_ks{ks:g}", ks_plus_target=ks,
                          lambda_over_ks=float("nan"), profile=profile, morphology="sand"))
    seen: set[float] = set()
    for c in cases:
        rt = round(c.re_tau, 3)
        if rt in seen:
            continue
        seen.add(rt)
        cases.append(Case(case_id=f"smooth_re{rt:g}", ks_plus_target=rt,
                          lambda_over_ks=1.0, profile=profile, kind="smooth"))
    return cases


# ---------------------------------------------------------------------------
# Running one case
# ---------------------------------------------------------------------------

def run_case(case: Case, out_dir: str | Path, *, threads: int | None = None,
             rebuild: bool = False) -> dict[str, Any]:
    """Run one case, write ``<out_dir>/<case_id>.json`` and return the record."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cfg = case.config(threads=threads)
    elements = case.elements()
    record = run_rough_channel_native(cfg, elements, rebuild=rebuild)
    record["case"] = case.to_json()
    record["ice_geometry"] = ice_geom.geometry_summary(elements)
    record["ks_nodes"] = float(case.ks_nodes)
    record["dy_plus"] = float(record["re_tau"]) / float(record["h_eff"])
    record["wall_resolved"] = bool(record["dy_plus"] <= 1.5)
    _atomic_json(out_path / f"{case.case_id}.json", record)
    return record


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_plain(payload), indent=2))
    os.replace(tmp, path)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def load_records(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(Path(run_dir).glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except Exception:  # pragma: no cover - partial file from a killed run
            continue
        out[rec.get("case", {}).get("case_id", p.stem)] = rec
    return out


# ---------------------------------------------------------------------------
# Aggregation and acceptance criteria
# ---------------------------------------------------------------------------

def _smooth_reference(records: Mapping[str, Mapping[str, Any]], re_tau: float) -> dict[str, Any]:
    """Nearest available smooth baseline (exact match preferred)."""
    best_id, best_key = None, None
    for cid, rec in records.items():
        if rec.get("case", {}).get("kind") != "smooth":
            continue
        key = abs(math.log(float(rec["re_tau"]) / re_tau))
        if best_key is None or key < best_key:
            best_id, best_key = cid, key
    if best_id is None:
        return {}
    rec = records[best_id]
    return {
        "case_id": best_id,
        "re_tau": float(rec["re_tau"]),
        "cf": float(rec["cf"]),
        "u_plus_bulk": float(rec["u_plus_bulk"]),
        "b": float((rec.get("log_fit") or {}).get("b", rl.B_SMOOTH)),
        "log_re_tau_mismatch": float(abs(math.log(float(rec["re_tau"]) / re_tau))),
    }


def _delta_u_plus(rec: Mapping[str, Any], smooth: Mapping[str, Any]) -> dict[str, Any]:
    """Bulk (Hama/Clauser) and log-intercept roughness functions of one case."""
    u_plus_bulk = float(rec["u_plus_bulk"])
    u_bulk_smooth = float(smooth.get("u_plus_bulk", 0.0)) if smooth else 0.0
    d_bulk = (u_bulk_smooth - u_plus_bulk) if u_bulk_smooth else float("nan")
    fit = rec.get("log_fit") or {}
    b_smooth = float(smooth.get("b", rl.B_SMOOTH)) if smooth else rl.B_SMOOTH
    d_profile = (b_smooth - float(fit["b"])) if fit.get("b") is not None else float("nan")
    window = rec.get("log_window") or {}
    n_points = int(window.get("n_points", 0))
    log_valid = bool(fit) and n_points >= 3
    return {
        "dU_plus_profile": float(d_profile),
        "dU_plus_bulk": float(d_bulk),
        "log_window_valid": bool(log_valid),
        "log_window_points": n_points,
        "b_smooth_reference": float(b_smooth),
        "dU_plus_used": float(d_profile) if log_valid else float(d_bulk),
        "dU_plus_measure": "log_intercept" if log_valid else "bulk_hama_clauser",
    }


def _cf_nikuradse_prediction(re_tau: float, ks_plus: float) -> float:
    """Cf that the Nikuradse sand-grain law predicts for this ``ks+`` and ``Re_tau``."""
    u_b_plus_smooth = math.sqrt(2.0 / rl.cf_channel_smooth_reference(re_tau))
    u_b_plus_rough = u_b_plus_smooth - float(rl.nikuradse_delta_u_plus(ks_plus))
    return 2.0 / u_b_plus_rough**2


def _group_key(lam: Any) -> str:
    try:
        v = float(lam)
    except (TypeError, ValueError):
        return "sand"
    return "sand" if not math.isfinite(v) else f"{v:g}"


def _cf_nikuradse_prediction(re_tau: float, ks_plus: float) -> float:
    """Cf that the Nikuradse sand-grain law predicts for this ``ks+`` and ``Re_tau``."""
    u_b_plus_smooth = math.sqrt(2.0 / rl.cf_channel_smooth_reference(re_tau))
    u_b_plus_rough = u_b_plus_smooth - float(rl.nikuradse_delta_u_plus(ks_plus))
    return 2.0 / u_b_plus_rough**2


def aggregate(run_dir: str | Path, *, criteria: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Collect case records, fit the three-zone laws and adjudicate C1-C4."""
    crit = {
        "c1_smooth_cf_rel_error": 0.03,
        "c2_fully_rough_cf_rel_error": 0.05,
        "c3_ks_plus_grid": [1.0, 2.0, 5.0, 10.0, 20.0, 45.0, 70.0, 100.0, 200.0, 500.0, 1000.0],
        "c4_max_angle_error_deg": 10.0,
        "c4_reference_lambda_over_ks": 5.0,
    }
    if criteria:
        crit.update(dict(criteria))

    records = load_records(run_dir)
    if not records:
        raise FileNotFoundError(f"no case records in {run_dir}")

    smooth_rows: list[dict[str, Any]] = []
    rough_rows: list[dict[str, Any]] = []
    for cid, rec in sorted(records.items()):
        case = rec.get("case", {})
        if case.get("kind") == "smooth":
            ref = rl.cf_channel_smooth_reference(float(rec["re_tau"]))
            smooth_rows.append({
                "case_id": cid, "re_tau": float(rec["re_tau"]), "cf": float(rec["cf"]),
                "cf_reference": float(ref),
                "rel_error": float((rec["cf"] - ref) / ref),
                "u_plus_bulk": float(rec["u_plus_bulk"]),
                "dy_plus": float(rec.get("dy_plus", float("nan"))),
                "max_mach": float(rec.get("max_mach", float("nan"))),
                "log_fit_b": float((rec.get("log_fit") or {}).get("b", float("nan"))),
                "wall_seconds": float(rec.get("wall_seconds", 0.0)),
                "backend": rec.get("backend"),
            })
            continue
        smooth = _smooth_reference(records, float(rec["re_tau"]))
        du = _delta_u_plus(rec, smooth)
        ks_plus = float(rec["ks_plus"])
        cf_nik = _cf_nikuradse_prediction(float(rec["re_tau"]), ks_plus)
        rough_rows.append({
            "case_id": cid,
            "group": _group_key(case.get("lambda_over_ks")),
            "morphology": case.get("morphology", "ribs"),
            "lambda_over_ks": float(case.get("lambda_over_ks", float("nan"))),
            "ks_plus_target": float(case.get("ks_plus_target", float("nan"))),
            "ks_plus": ks_plus,
            "ks_nodes": float(rec.get("ks_nodes", float("nan"))),
            "re_tau": float(rec["re_tau"]),
            "dy_plus": float(rec.get("dy_plus", float("nan"))),
            "cf": float(rec["cf"]),
            "u_plus_bulk": float(rec["u_plus_bulk"]),
            "u_tau": float(rec["u_tau"]),
            "max_mach": float(rec.get("max_mach", float("nan"))),
            "dU_plus_profile": du["dU_plus_profile"],
            "dU_plus_bulk": du["dU_plus_bulk"],
            "dU_plus_used": du["dU_plus_used"],
            "dU_plus_measure": du["dU_plus_measure"],
            "log_window_valid": du["log_window_valid"],
            "log_window_points": du["log_window_points"],
            "smooth_reference": smooth.get("case_id"),
            "cf_nikuradse": float(cf_nik),
            "cf_rel_error_vs_nikuradse": float((rec["cf"] - cf_nik) / cf_nik),
            "wall_seconds": float(rec.get("wall_seconds", 0.0)),
        })

    # ---- group-wise fit of the onset parameter k0 -----------------------
    groups: list[dict[str, Any]] = []
    laws: dict[str, rl.ThreeZoneLaw] = {}
    for key in sorted({r["group"] for r in rough_rows}, key=lambda g: (g == "sand", g)):
        rows = [r for r in rough_rows if r["group"] == key]
        lam = rows[0]["lambda_over_ks"]
        rows_log = [r for r in rows if r["log_window_valid"] and here_positive(r["dU_plus_profile"])]
        use = rows_log if len(rows_log) >= 2 else [r for r in rows if here_positive(r["dU_plus_used"])]
        fit = rl.fit_k0([r["ks_plus"] for r in use], [r["dU_plus_used"] for r in use])
        fit["n_cases"] = len(rows)
        fit["n_cases_log_window"] = len(rows_log)
        fit["measure"] = "log_intercept" if len(rows_log) >= 2 else "bulk_hama_clauser"
        grp = {
            "group": key,
            "morphology": rows[0]["morphology"],
            "lambda_over_ks": lam,
            "cases": [r["case_id"] for r in rows],
            "fit": fit,
            "equivalent_sand_grain_ratio": rl.equivalent_sand_grain_ratio(fit["k0"])
            if math.isfinite(fit["k0"]) else float("nan"),
            "regimes": rl.ks_plus_boundaries(fit["k0"]) if math.isfinite(fit["k0"]) else {},
        }
        groups.append(grp)
        if math.isfinite(fit["k0"]) and fit["k0"] > 0:
            laws[key] = rl.ThreeZoneLaw(
                lambda_over_ks=float(lam), k0=float(fit["k0"]), k0_source="lbm-fit",
                k0_band=(fit["k0_low"], fit["k0_high"]), n_points=int(fit["n_points"]),
                fit_residual=float(fit.get("rms_log_residual", float("nan"))),
            )

    # ---- C1: smooth-wall accuracy --------------------------------------
    c1_rows = [s for s in smooth_rows if math.isfinite(s["rel_error"])]
    c1 = {
        "threshold_rel_error": crit["c1_smooth_cf_rel_error"],
        "reference": "U_b^+ = 2.439 ln(Re_tau) + 5.65 (Moser 1999 / Lee-Moser 2015 channel DNS fit)",
        "max_abs_rel_error": max((abs(s["rel_error"]) for s in c1_rows), default=float("nan")),
        "n_cases": len(c1_rows),
        "passed": bool(c1_rows) and all(abs(s["rel_error"]) <= crit["c1_smooth_cf_rel_error"] for s in c1_rows),
        "rows": c1_rows,
    }

    # ---- C2: fully rough regime vs Nikuradse ---------------------------
    fully_rough_rows = []
    for r in rough_rows:
        law = laws.get(r["group"])
        if law is None:
            continue
        if law.regime_of(r["ks_plus"]) == "fully_rough":
            fully_rough_rows.append(r)
    sand_fr = [r for r in fully_rough_rows if r["group"] == "sand" and r["log_window_valid"]]
    c2 = {
        "threshold_rel_error": crit["c2_fully_rough_cf_rel_error"],
        "scope": "Nikuradse sand-grain control group, fully-rough cases with a valid log layer",
        "n_cases": len(sand_fr),
        "max_abs_rel_error": max((abs(r["cf_rel_error_vs_nikuradse"]) for r in sand_fr), default=float("nan")),
        "passed": bool(sand_fr) and all(abs(r["cf_rel_error_vs_nikuradse"]) <= crit["c2_fully_rough_cf_rel_error"]
                                        for r in sand_fr),
        "rows": [{k: r[k] for k in ("case_id", "ks_plus", "cf", "cf_nikuradse",
                                    "cf_rel_error_vs_nikuradse", "log_window_valid")} for r in sand_fr],
        "all_fully_rough": [{k: r[k] for k in ("case_id", "group", "ks_plus", "cf", "cf_nikuradse",
                                               "cf_rel_error_vs_nikuradse", "log_window_valid")}
                            for r in fully_rough_rows],
    }

    # ---- C3: monotone transitional interpolation -----------------------
    c3_rows = []
    cf_smooth = rl.cf_channel_smooth_reference(500.0)
    grid = crit["c3_ks_plus_grid"]
    for grp in groups:
        law = laws.get(grp["group"])
        if law is None:
            grp["monotone"] = False
            continue
        prev_du, prev_ratio, ok = -1.0, 0.0, True
        for ks in grid:
            du = law.delta_u_plus(ks)
            b = law.regimes()
            if b["ks_plus_smooth_upper"] < ks < b["ks_plus_fully_rough_lower"]:
                ratio = law.cf_ratio(ks, cf_smooth)
                c3_rows.append({"group": grp["group"], "ks_plus": ks, "dU_plus": du,
                                "cf_ratio": ratio, "regime": law.regime_of(ks)})
                if du <= prev_du or ratio <= prev_ratio:
                    ok = False
                prev_du, prev_ratio = du, ratio
            else:
                prev_du = max(prev_du, du)
        top = law.regimes()["ks_plus_fully_rough_lower"] * 10.0
        slope = (law.delta_u_plus(top * 1.1) - law.delta_u_plus(top)) / math.log(1.1)
        c3_rows.append({"group": grp["group"], "ks_plus": top, "dU_plus": law.delta_u_plus(top),
                        "cf_ratio": law.cf_ratio(top, cf_smooth), "regime": "fully_rough",
                        "dDU_plus_dlnks_plus": slope, "one_over_kappa": 1.0 / rl.KAPPA})
        # the finite ln(1 + ks+/k0) form approaches the 1/kappa slope from
        # below: at ks+ = 10 x (fully-rough onset) the deficit is 0.5% and it
        # vanishes like k0/ks+.  2% is the honest tolerance for a *finite*
        # difference; the exact asymptote is 1/kappa by construction.
        if abs(slope - 1.0 / rl.KAPPA) > 2e-2:
            ok = False
        grp["monotone"] = ok
    c3 = {
        "passed": bool(c3_rows) and all(g.get("monotone", False) for g in groups),
        "rows": c3_rows,
        "n_points_checked": len(c3_rows),
    }

    # ---- C4: Nansen drift angle after the back-fill --------------------
    from . import freedrift as fd
    ref_lam = float(crit["c4_reference_lambda_over_ks"])
    keel = ice_geom.default_ice_keel(lambda_over_ks=ref_lam)
    env = fd.DriftEnvironment()
    c4: dict[str, Any] = {"threshold_deg": crit["c4_max_angle_error_deg"], "passed": False,
                          "ice_keel": keel.as_dict()}
    law_ref = laws.get(f"{ref_lam:g}") or (laws.get("sand") if laws else None)
    if law_ref is not None:
        backfill = fd.drift_with_roughness_amplification(env, keel.ks_m, law_ref,
                                                         cw_smooth=fd.CW_SMOOTH_REFERENCE)
        const = fd.drift_angle(env, fd.CW_SMOOTH_REFERENCE)
        sweep_rows = []
        for ua in (2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0):
            e = fd.DriftEnvironment(**{**env.__dict__, "wind_speed": ua})
            cl = fd.drift_with_roughness_amplification(e, keel.ks_m, law_ref)
            cst = fd.drift_angle(e, fd.CW_SMOOTH_REFERENCE)
            sweep_rows.append({"wind_speed_m_s": ua, "closure_C_w": cl["C_w"],
                               "closure_angle_deg": cl["drift_angle_deg"],
                               "closure_ks_plus": cl["ks_plus"], "closure_regime": cl["regime"],
                               "constant_C_w": fd.CW_SMOOTH_REFERENCE,
                               "constant_angle_deg": cst["drift_angle_deg"],
                               "angle_delta_deg": cl["drift_angle_deg"] - cst["drift_angle_deg"]})
        err = float(abs(backfill["drift_angle_deg"] - fd.NANSEN_DRIFT_ANGLE_DEG))
        c4.update({
            "passed": bool(err <= crit["c4_max_angle_error_deg"]),
            "angle_error_deg": err,
            "law_used": law_ref.as_dict(),
            "backfill": {k: v for k, v in backfill.items() if k != "iteration_history"},
            "constant_reference": {**const, "angle_error_deg": abs(const["drift_angle_deg"] - fd.NANSEN_DRIFT_ANGLE_DEG)},
            "angle_improvement_deg": abs(const["drift_angle_deg"] - fd.NANSEN_DRIFT_ANGLE_DEG) - err,
            "sweep": sweep_rows,
            "form": "C_w = C_w,smooth * Cf_rough/Cf_smooth(k_s^+, lambda/k_s) (LBM amplification)",
        })
        # secondary: absolute log-law closure (depends on the OBL reference height)
        try:
            alt = fd.free_drift_solution(env, keel.ks_m, ref_lam,
                                         laws={law_ref.lambda_over_ks: law_ref})
            c4["log_law_closure_absolute"] = {k: v for k, v in alt.items() if k != "iteration_history"}
        except Exception as exc:  # pragma: no cover
            c4["log_law_closure_absolute_error"] = str(exc)

    summary = {
        "generated_by": "tensorlbm.ice_roughness.sweep.aggregate",
        "run_dir": str(run_dir),
        "n_cases": len(records),
        "n_rough": len(rough_rows),
        "n_smooth": len(smooth_rows),
        "criteria": crit,
        "smooth_baselines": smooth_rows,
        "rough_cases": rough_rows,
        "groups": groups,
        "laws": {k: v.as_dict() for k, v in laws.items()},
        "C1_smooth_wall_cf": c1,
        "C2_fully_rough_vs_nikuradse": c2,
        "C3_transition_monotone": c3,
        "C4_nansen_drift_angle": c4,
        "all_passed": bool(c1["passed"] and c2["passed"] and c3["passed"] and c4["passed"]),
    }
    return _plain(summary)


def here_positive(value: Any) -> bool:
    """``True`` for a finite, positive float (``dU^+`` must be an *increase*)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0
