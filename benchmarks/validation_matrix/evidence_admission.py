"""Separate numerical verdicts from the scientific claim they may support.

Declarations describe the audited producer, not an automatic authenticity proof.
The runner hashes the producer, matrix and artifact for traceability. Unknown
origins, fixtures and non-LBM methods cannot count as LBM validation passes.
"""


def _synthetic(payload):
    if not isinstance(payload, dict):
        return False
    if payload.get("synthetic") is True or payload.get("usable_for_physical_validation") is False:
        return True
    if payload.get("evidence_type") in ("synthetic_fixture", "synthetic", "demo", "mock"):
        return True
    return any(_synthetic(payload.get(key)) for key in ("metadata", "provenance"))


def admit_evidence(case, payload=None):
    evidence = case.get("evidence") or {}
    origin = evidence.get("origin", "unclassified")
    method = evidence.get("method", "unclassified")
    scope = evidence.get("claim_scope", "unclassified")
    synthetic = origin == "synthetic_fixture" or _synthetic(payload)
    reasons = []
    if synthetic:
        reasons.append("synthetic_or_demo_artifact")
    if origin != "recorded_solver_run":
        reasons.append("no_recorded_solver_origin")
    if method != "lbm":
        reasons.append("not_lbm_solver_evidence")
    if scope != "benchmark_validation":
        reasons.append("reference_or_correlation_only")
    return {
        "origin": origin,
        "method": method,
        "claim_scope": scope,
        "synthetic": synthetic,
        "lbm_validation_eligible": not reasons,
        "exclusion_reasons": reasons,
        "engineering_validated": False,
    }
