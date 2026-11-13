"""negative-no-oss-reference discipline for XVAL (cross-validation) claims.

Adopted practice
----------------
TensorKnowledge's ``negative-no-oss-reference`` practice: a solver claim may only
be judged against a reference that is **resolvable in this environment**, and a
capability that has no such reference must be recorded as an explicit NEGATIVE
result ("no OSS reference available", with the probe evidence attached) instead
of being compared against a remembered, copied or invented reference column.

Why this lives in TensorLBM
---------------------------
TensorLBM owns lattice-Boltzmann numerical methods and solver-local verification;
what may count as a reference for a solver claim is part of that verification
surface.  Nothing here knows about ship physics or platform orchestration.

Three pieces
------------
``probe_oss_references``
    Executable inventory of the OSS lattice-Boltzmann references visible to this
    interpreter: ``python_module`` candidates are imported and fingerprinted
    (version, file, sha256), ``binary`` candidates are looked up on PATH / in the
    declared candidate list and fingerprinted the same way.  The record carries a
    ``record_sha256`` so a claim can cite exactly one probe snapshot.

``audit_capability_coverage``
    Maps TensorLBM capability families onto the probed references.  A family is
    ``OSS_REFERENCE_AVAILABLE`` only when one probed reference is importable *and*
    the declared evidence symbol of that reference resolves; look-alike modules
    that do not implement the family are listed as ``reject_evidence`` and do not
    count.  Everything else is ``NO_OSS_REFERENCE``.

``XvalReferenceGate``
    Verdict machine for claims.  ``PASS`` / ``FAIL`` require a resolvable
    reference id; an unresolvable one returns ``WITHHELD`` and carries the
    negative-knowledge payload that must be registered.  There is deliberately no
    code path that turns an unreferenced number into a ``PASS``.

CLI
---
``python -m tensorlbm.xval.oss_reference --out artifacts/xval_oss_probe``
writes ``oss_reference_probe.json``, ``capability_coverage.json`` and
``negative_knowledge_payloads.json``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import importlib.util
import json
import os
import platform as _platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "PROBE_SCHEMA",
    "OSS_CANDIDATES",
    "CAPABILITY_REQUIREMENTS",
    "ANALYTIC_REFERENCES",
    "FAILURE_KIND",
    "XvalReferenceGate",
    "Claim",
    "UnreferencedClaimError",
    "probe_oss_references",
    "audit_capability_coverage",
    "negative_knowledge_payloads",
    "write_audit_bundle",
]

PROBE_SCHEMA = "xval_oss_probe/v1"
COVERAGE_SCHEMA = "xval_capability_coverage/v1"
TOLERANCE_PCT = 15.0
#: ledger enum value used for "this capability has no resolvable OSS reference"
FAILURE_KIND = "hypothesis_falsified"

_REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# reference candidate registry
# --------------------------------------------------------------------------
OSS_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "oss_id": "lbmpy",
        "kind": "python_module",
        "target": "lbmpy",
        "upstream": "lbmpy 2.x (FAU Erlangen / pycodegen), Apache-2.0",
        "note": "generated (pystencils) LBM kernels, independent code path from the in-house torch solver",
    },
    {
        "oss_id": "pylbm",
        "kind": "python_module",
        "target": "pylbm",
        "upstream": "pylbm (LBM scheme builder on sympy), BSD-3",
        "note": "reference for scheme specification; no compiled kernel",
    },
    {
        "oss_id": "lettuce",
        "kind": "python_module",
        "target": "lettuce",
        "upstream": "lettuce (GPU LBM on torch), MIT",
        "note": "torch-based GPU LBM, would allow a same-backend comparison",
    },
    {
        "oss_id": "xlb",
        "kind": "python_module",
        "target": "xlb",
        "upstream": "XLB (jax-based multi-GPU LBM), MIT",
        "note": "needs jax; used as a free-surface / thermal reference when present",
    },
    {
        "oss_id": "walberla",
        "kind": "python_module",
        "target": "waLBerla",
        "upstream": "waLBerla (FAU Erlangen), free-surface lattice gas",
        "note": "Python bindings are rarely packaged; presence is checked, not assumed",
    },
    {
        "oss_id": "palabos",
        "kind": "binary",
        "env_var": "XVAL_PALABOS_BIN",
        "candidates": ("benchmarks/xval_taylor_green/palabos/bin/tg2d_palabos",),
        "path_names": ("tg2d_palabos",),
        "upstream": "Palabos 2.x (gitlab/unigespc), AGPL-3.0",
        "note": "in-tree XVAL harness binary; must be built by hand from the Palabos release",
    },
    {
        "oss_id": "openfoam_icofoam",
        "kind": "binary",
        "env_var": "XVAL_ICOFoam_BIN",
        "candidates": (),
        "path_names": ("icoFoam",),
        "upstream": "OpenFOAM (ESI/Foundation), GPL-3.0",
        "note": "finite-volume (not LBM) reference for the same incompressible case",
    },
)


# --------------------------------------------------------------------------
# capability families and the OSS evidence that would actually cover them
# --------------------------------------------------------------------------
# ``evidence`` maps oss_id -> sequence of importable module paths that must
# resolve for the reference to count as covering the family.  ``reject_evidence``
# records look-alike modules that were *checked and rejected* (honest record of
# why an available package still does not provide the capability).
CAPABILITY_REQUIREMENTS: tuple[dict[str, Any], ...] = (
    {
        "capability": "core_d2q9_srt_bgk",
        "evidence": {"lbmpy": ("lbmpy.creationfunctions", "lbmpy.methods"), "pylbm": ("pylbm.scheme",), "lettuce": ("lettuce",)},
        "note": "periodic single-relaxation-time D2Q9 step",
    },
    {
        "capability": "trt_mrt_cumulant_collision",
        "evidence": {"lbmpy": ("lbmpy.methods", "lbmpy.cumulants"), "pylbm": ("pylbm.scheme",)},
        "note": "multi-relaxation / cumulant collision operators",
    },
    {
        "capability": "guo_body_force",
        "evidence": {"lbmpy": ("lbmpy.forcemodels",)},
        "note": "Guo/Shan-Chen style body-force source terms",
    },
    {
        "capability": "bounce_back_walls_and_velocity_inlet",
        "evidence": {"lbmpy": ("lbmpy.boundaries",), "palabos": ()},
        "note": "half-way bounce-back, UBB/NoSlip boundary conditions",
    },
    {
        "capability": "smagorinsky_les",
        "evidence": {"lbmpy": ("lbmpy.turbulence_models",)},
        "note": "eddy-viscosity subgrid closures",
    },
    {
        "capability": "non_newtonian_viscosity",
        "evidence": {"lbmpy": ("lbmpy.non_newtonian_models",)},
        "note": "shear-rate dependent viscosity (Bingham / power law)",
    },
    {
        "capability": "phasefield_allen_cahn_cahn_hilliard",
        "evidence": {"lbmpy": ("lbmpy.phasefield_allen_cahn", "lbmpy.phasefield")},
        "note": "phase-field two-phase coupling",
    },
    {
        "capability": "shan_chen_pseudopotential_multiphase",
        "evidence": {},
        "reject_evidence": {"lbmpy": ("lbmpy.phasefield", "lbmpy.forcemodels")},
        "note": "pseudo-potential Shan-Chen; the phase-field modules are not a substitute for the pseudo-potential force",
    },
    {
        "capability": "free_surface_volume_tracking",
        "evidence": {},
        "reject_evidence": {"lbmpy": ("lbmpy.partially_saturated_cells", "lbmpy.geometry")},
        "note": "free-surface fill level / mass tracking; partially saturated cells are a boundary treatment, not a volume-tracking free surface",
    },
    {
        "capability": "immersed_boundary_ibm",
        "evidence": {},
        "note": "Lagrangian IBM force spreading; no probed reference exposes it",
    },
    {
        "capability": "adaptive_mesh_refinement",
        "evidence": {},
        "reject_evidence": {"lbmpy": ("lbmpy.sparse",)},
        "note": "block-structured AMR; lbmpy.sparse is sparse storage, not refinement",
    },
    {
        "capability": "thermal_energy_transport_lbm",
        "evidence": {},
        "note": "double-distribution thermal LBM; not exposed by any probed reference",
    },
    {
        "capability": "acoustic_pulse_absorbing_boundary",
        "evidence": {},
        "note": "acoustic damping / absorbing boundary treatment",
    },
)


ANALYTIC_REFERENCES: dict[str, dict[str, str]] = {
    "taylor_green_2d_exact": {
        "kind": "analytic",
        "formula": "u=-U0 cos(kx) sin(ky) e^{-2 nu k^2 t}, v=+U0 sin(kx) cos(ky) e^{-2 nu k^2 t}; K=0.5<u^2+v^2>=K0 e^{-4 nu k^2 t}",
        "source": "exact solution of incompressible Navier-Stokes (convection is a pure gradient); identical to the TG2DSpec docstring in src/tensorlbm/xval/case_spec.py",
    },
    "poiseuille_plane_channel": {
        "kind": "analytic",
        "formula": "u(y)=g/(2 nu) y (H-y), u_max = g H^2/(8 nu)",
        "source": "steady incompressible plane-Poiseuille solution; already used as the analytic anchor of the compressed-node-neighbour-table XVAL",
    },
}


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------
def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _module_sha256(module_path: str | None) -> str | None:
    """sha256 of a module source file, or ``None`` for frozen/builtin modules."""
    if not module_path:
        return None
    p = Path(module_path)
    if not p.is_file():
        return None
    return _sha256_file(p)


def _probe_module(spec: Mapping[str, Any]) -> dict[str, Any]:
    target = str(spec["target"])
    rec: dict[str, Any] = {
        "oss_id": spec["oss_id"],
        "kind": "python_module",
        "target": target,
        "upstream": spec.get("upstream", ""),
        "note": spec.get("note", ""),
        "available": False,
        "version": None,
        "path": None,
        "sha256": None,
        "error": None,
    }
    try:
        found = importlib.util.find_spec(target) is not None
    except Exception as exc:  # pragma: no cover - defensive
        rec["error"] = f"find_spec failed: {exc!r}"
        return rec
    if not found:
        rec["error"] = "module not importable in this interpreter"
        return rec
    try:
        mod = importlib.import_module(target)
    except Exception as exc:
        rec["error"] = f"import failed: {exc!r}"
        return rec
    rec["available"] = True
    rec["version"] = str(getattr(mod, "__version__", "") or "") or None
    mod_file = getattr(mod, "__file__", None)
    rec["path"] = str(mod_file) if mod_file else None
    rec["sha256"] = _module_sha256(mod_file)
    return rec


def _probe_binary(spec: Mapping[str, Any]) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "oss_id": spec["oss_id"],
        "kind": "binary",
        "target": spec.get("path_names", ("",))[0],
        "upstream": spec.get("upstream", ""),
        "note": spec.get("note", ""),
        "available": False,
        "version": None,
        "path": None,
        "sha256": None,
        "error": None,
    }
    tried: list[str] = []
    cands: list[Path] = []
    env_var = spec.get("env_var")
    if env_var and os.environ.get(env_var):
        cands.append(Path(os.environ[env_var]))
    for rel in spec.get("candidates", ()):  # type: ignore[arg-type]
        cands.append(_REPO_ROOT / rel)
    for name in spec.get("path_names", ()):  # type: ignore[arg-type]
        which = shutil.which(name)
        if which:
            cands.append(Path(which))
    for cand in cands:
        tried.append(str(cand))
        if cand.is_file() and os.access(cand, os.X_OK):
            rec.update(available=True, path=str(cand), sha256=_sha256_file(cand))
            return rec
    rec["error"] = "not found; tried " + (", ".join(tried) if tried else "(no candidate paths)")
    return rec


def probe_oss_references(
    candidates: Sequence[Mapping[str, Any]] | None = None,
    *,
    generated_at: str | None = None,
    extra_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Probe every OSS reference candidate and return one fingerprinted record."""
    specs = tuple(candidates) if candidates is not None else OSS_CANDIDATES
    refs: list[dict[str, Any]] = []
    for spec in specs:
        if spec.get("kind") == "python_module":
            refs.append(_probe_module(spec))
        elif spec.get("kind") == "binary":
            refs.append(_probe_binary(spec))
        else:  # pragma: no cover - registry is closed
            raise ValueError(f"unknown probe kind {spec.get('kind')!r} for {spec.get('oss_id')!r}")
    env = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": _platform.platform(),
        "machine": _platform.machine(),
        "env_flags": {
            k: bool(os.environ.get(k))
            for k in ("XVAL_PALABOS_BIN", "XVAL_ICOFoam_BIN")
        },
    }
    if extra_environment:
        env.update(extra_environment)
    payload = {
        "schema": PROBE_SCHEMA,
        "generated_at": generated_at or _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "environment": env,
        "references": refs,
        "available_ids": [r["oss_id"] for r in refs if r["available"]],
    }
    payload["record_sha256"] = hashlib.sha256(
        json.dumps(
            {"schema": payload["schema"], "environment": payload["environment"], "references": refs, "available_ids": payload["available_ids"]},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return payload


def _module_resolves(dotted: str) -> bool:
    if not dotted:
        return True
    try:
        return importlib.util.find_spec(dotted) is not None
    except Exception:
        return False


# --------------------------------------------------------------------------
# capability coverage
# --------------------------------------------------------------------------
def audit_capability_coverage(
    probe: Mapping[str, Any],
    requirements: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify each capability family as OSS_REFERENCE_AVAILABLE / NO_OSS_REFERENCE."""
    reqs = tuple(requirements) if requirements is not None else CAPABILITY_REQUIREMENTS
    by_id = {r["oss_id"]: r for r in probe.get("references", [])}
    rows: list[dict[str, Any]] = []
    for req in reqs:
        cap = str(req["capability"])
        supporting: list[str] = []
        rejected: list[dict[str, Any]] = []
        checked: list[dict[str, Any]] = []
        for oss_id, symbols in (req.get("evidence") or {}).items():
            rec = by_id.get(oss_id)
            if rec is None:
                checked.append({"oss_id": oss_id, "probed": False, "reason": "id not in probe record"})
                continue
            if not rec.get("available"):
                checked.append({"oss_id": oss_id, "probed": True, "available": False, "reason": rec.get("error")})
                continue
            missing = [s for s in symbols if not _module_resolves(s)]
            checked.append(
                {
                    "oss_id": oss_id,
                    "probed": True,
                    "available": True,
                    "version": rec.get("version"),
                    "symbols": list(symbols),
                    "missing_symbols": missing,
                }
            )
            if not missing:
                supporting.append(oss_id)
        for oss_id, symbols in (req.get("reject_evidence") or {}).items():
            rec = by_id.get(oss_id)
            if rec and rec.get("available"):
                checked.append(
                    {
                        "oss_id": oss_id,
                        "probed": True,
                        "available": True,
                        "version": rec.get("version"),
                        "symbols": list(symbols),
                        "verdict": "look_alike_rejected: implements a different capability",
                    }
                )
                rejected.append(
                    {
                        "oss_id": oss_id,
                        "symbols": list(symbols),
                        "why_not": "module exists but does not implement this capability (see note)",
                    }
                )
        rows.append(
            {
                "capability": cap,
                "status": "OSS_REFERENCE_AVAILABLE" if supporting else "NO_OSS_REFERENCE",
                "supporting_references": supporting,
                "references_checked": checked,
                "look_alikes_rejected": rejected,
                "note": req.get("note", ""),
            }
        )
    n_avail = sum(1 for r in rows if r["status"] == "OSS_REFERENCE_AVAILABLE")
    return {
        "schema": COVERAGE_SCHEMA,
        "probe_sha256": probe.get("record_sha256"),
        "probe_generated_at": probe.get("generated_at"),
        "capabilities_total": len(rows),
        "capabilities_with_oss_reference": n_avail,
        "capabilities_without_oss_reference": len(rows) - n_avail,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------
class UnreferencedClaimError(RuntimeError):
    """Raised when a code path tries to judge a claim without a reference id."""


@dataclass(frozen=True)
class Claim:
    """One cross-validation claim, i.e. one measured KPI set plus its reference."""

    claim_id: str
    capability: str
    reference_kind: str  # "oss" | "analytic" | "literature"
    reference_id: str
    errors_pct: Mapping[str, float]
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    @property
    def max_error_pct(self) -> float:
        if not self.errors_pct:
            raise UnreferencedClaimError(f"claim {self.claim_id!r} carries no KPI at all")
        return max(abs(float(v)) for v in self.errors_pct.values())


class XvalReferenceGate:
    """PASS / FAIL / WITHHELD verdict machine for XVAL claims.

    Rules (predeclared in ``xval/XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE/predeclared.json``):

    * ``oss`` reference -> must appear in the probe record with ``available`` true,
      otherwise the claim is ``WITHHELD`` and a negative-knowledge payload is
      returned with it;
    * ``analytic`` / ``literature`` reference -> must be a declared id in
      ``ANALYTIC_REFERENCES`` (traceable formula in-repo), otherwise ``WITHHELD``;
    * with a resolvable reference: ``PASS`` iff every KPI < tolerance, else ``FAIL``.
    """

    def __init__(
        self,
        probe: Mapping[str, Any],
        *,
        tolerance_pct: float = TOLERANCE_PCT,
        analytic_registry: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self.probe = probe
        self.tolerance_pct = float(tolerance_pct)
        self.analytic_registry = dict(analytic_registry if analytic_registry is not None else ANALYTIC_REFERENCES)
        self._available = {
            r["oss_id"]: r for r in probe.get("references", []) if r.get("available")
        }

    # -- reference resolution -------------------------------------------------
    def resolve(self, kind: str, ref_id: str) -> dict[str, Any]:
        if kind == "oss":
            rec = self._available.get(ref_id)
            if rec is None:
                return {
                    "resolved": False,
                    "detail": f"OSS id {ref_id!r} is not available in probe {self.probe.get('record_sha256')}",
                }
            return {
                "resolved": True,
                "detail": f"{ref_id} {rec.get('version')} @ {rec.get('path')}",
                "sha256": rec.get("sha256"),
                "version": rec.get("version"),
            }
        if kind in ("analytic", "literature"):
            rec = self.analytic_registry.get(ref_id)
            if rec is None:
                return {
                    "resolved": False,
                    "detail": f"{kind} id {ref_id!r} is not in the declared analytic registry",
                }
            return {"resolved": True, "detail": rec.get("source", ""), "formula": rec.get("formula", "")}
        raise UnreferencedClaimError(f"unknown reference kind {kind!r} for claim reference {ref_id!r}")

    # -- verdict --------------------------------------------------------------
    def decide(self, claim: Claim) -> dict[str, Any]:
        if not claim.reference_id:
            raise UnreferencedClaimError(f"claim {claim.claim_id!r} has an empty reference id")
        res = self.resolve(claim.reference_kind, claim.reference_id)
        base = {
            "claim_id": claim.claim_id,
            "capability": claim.capability,
            "reference": {"kind": claim.reference_kind, "id": claim.reference_id, **res},
            "errors_pct": {k: float(v) for k, v in claim.errors_pct.items()},
            "max_error_pct": claim.max_error_pct,
            "tolerance_pct": self.tolerance_pct,
            "probe_sha256": self.probe.get("record_sha256"),
            "artifacts": list(claim.artifacts),
        }
        if not res["resolved"]:
            base.update(
                status="WITHHELD",
                reason=(
                    f"reference {claim.reference_kind}:{claim.reference_id} cannot be resolved in this environment, "
                    f"so the measured KPIs cannot be judged; register a negative-no-oss-reference entry instead "
                    f"of quoting them as validated. ({res['detail']})"
                ),
                negative_knowledge_required=True,
            )
            return base
        worst = base["max_error_pct"]
        if worst < self.tolerance_pct:
            base.update(
                status="PASS",
                reason=f"max KPI deviation {worst:.3g}% < {self.tolerance_pct:g}% against {claim.reference_kind}:{claim.reference_id}",
                negative_knowledge_required=False,
            )
        else:
            base.update(
                status="FAIL",
                reason=f"max KPI deviation {worst:.3g}% >= {self.tolerance_pct:g}% against {claim.reference_kind}:{claim.reference_id}",
                negative_knowledge_required=False,
                culprits=[k for k, v in claim.errors_pct.items() if abs(float(v)) >= self.tolerance_pct],
            )
        return base


# --------------------------------------------------------------------------
# negative-knowledge payloads
# --------------------------------------------------------------------------
def negative_knowledge_payloads(
    coverage: Mapping[str, Any],
    *,
    mission_id: int | None = None,
    domain_tags: Sequence[str] = ("TensorLBM",),
    source_artifacts: Sequence[str] = (),
    registered_by: str = "tensorlbm.xval.oss_reference",
    confidence: float = 0.9,
) -> list[dict[str, Any]]:
    """One ``RegisterRequest``-shaped payload per capability with no OSS reference."""
    out: list[dict[str, Any]] = []
    for row in coverage.get("rows", []):
        if row.get("status") != "NO_OSS_REFERENCE":
            continue
        cap = row["capability"]
        evidence = {
            "probe_schema": coverage.get("schema"),
            "probe_sha256": coverage.get("probe_sha256"),
            "probe_generated_at": coverage.get("probe_generated_at"),
            "references_checked": row.get("references_checked", []),
            "look_alikes_rejected": row.get("look_alikes_rejected", []),
            "capabilities_total": coverage.get("capabilities_total"),
            "capabilities_without_oss_reference": coverage.get("capabilities_without_oss_reference"),
            "kind": "negative_no_oss_reference",
            "mission_id": mission_id,
        }
        out.append(
            {
                "title": f"TensorLBM {cap} 无本地可解析 OSS 参考（negative-no-oss-reference）",
                "hypothesis": f"TensorLBM 的 {cap} 可以直接与某个 OSS 实现做 XVAL 对照并给出通过判据",
                "failure_kind": FAILURE_KIND,
                "falsification_evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                "failure_reason": (
                    f"probe {coverage.get('probe_sha256', '')[:16]} 在 .py-importable/可执行候选 {len(coverage.get('rows', []))} 族检查中"
                    f"找不到覆盖 {cap} 的参考实现，因此该能力的 XVAL 判据只能是 WITHHELD；"
                    f"已在 {', '.join(source_artifacts) or 'artifacts'} 留证据"
                ),
                "applicable_conditions": {
                    "environment": "offline work tree; github.com unreachable, only installed packages / in-tree binaries count",
                    "probe_scope": "see references_checked in falsification_evidence",
                },
                "keywords": [cap, "XVAL", "no-oss-reference", "TensorLBM", "WITHHELD"],
                "domain_tags": list(domain_tags),
                "differentiation_required": ["提供可解析 OSS 参考后才能把该能力从 WITHHELD 升级"],
                "source_artifacts": list(source_artifacts),
                "confidence": float(confidence),
                "registered_by": registered_by,
                "_mission_id": mission_id,
            }
        )
    return out


# --------------------------------------------------------------------------
# bundle writer
# --------------------------------------------------------------------------
def write_audit_bundle(
    out_dir: str | Path,
    *,
    probe: Mapping[str, Any],
    coverage: Mapping[str, Any],
    payloads: Sequence[Mapping[str, Any]],
    mission_id: int | None = None,
) -> dict[str, Any]:
    """Write the three JSON artefacts plus a sha256 manifest, return the manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, obj in (
        ("oss_reference_probe.json", probe),
        ("capability_coverage.json", coverage),
        ("negative_knowledge_payloads.json", {"mission_id": mission_id, "payloads": list(payloads)}),
    ):
        p = out / name
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False))
        written[name] = _sha256_file(p)
    manifest = {
        "out_dir": str(out),
        "probe_sha256": probe.get("record_sha256"),
        "files": written,
        "generated_at": probe.get("generated_at"),
        "mission_id": mission_id,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    ap = argparse.ArgumentParser(description="Probe OSS LBM references and audit capability coverage")
    ap.add_argument("--out", default="artifacts/xval_negative_no_oss_reference")
    ap.add_argument("--mission-id", type=int, default=None)
    args = ap.parse_args(argv)

    probe = probe_oss_references()
    coverage = audit_capability_coverage(probe)
    payloads = negative_knowledge_payloads(coverage, mission_id=args.mission_id)
    manifest = write_audit_bundle(Path(args.out), probe=probe, coverage=coverage, payloads=payloads, mission_id=args.mission_id)
    print(json.dumps({"available_ids": probe["available_ids"], "tables": {
        "with_reference": coverage["capabilities_with_oss_reference"],
        "without_reference": coverage["capabilities_without_oss_reference"],
    }, "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())