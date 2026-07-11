"""Fail-closed aggregation for benchmark artifacts, numerics, and physics KPIs.

This module does not run solvers.  It turns outputs from an already completed
benchmark run into a durable regression-gate manifest, so CI never mistakes a
submitted, partial, or numerically invalid run for a benchmark success.
"""
from __future__ import annotations

import json
import math
import os
import csv
import re
import tempfile
from pathlib import Path
from typing import Any

_TERMINAL_SUCCESS = {"PASSED", "COMPLETED"}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _required_regexes(value: Any, field: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        errors.append(f"{field} must be a non-empty list of regular expressions")
        return []
    return value


def evaluate_acoustic_campaign_gate(artifacts_root: str | Path, specification: dict[str, Any]) -> dict[str, Any]:
    """Validate selected acoustic campaign logs without treating process exit as physics success.

    The specification names every accepted case and its expected contained log,
    metric evidence, and terminal physics-PASS line.  This is deliberately
    separate from the generic per-run JSON gate because the historical campaign
    predates ``run_status.json`` and only produced immutable CSV/log artifacts.
    """
    root = Path(artifacts_root).resolve()
    errors: list[str] = []
    if not isinstance(specification, dict):
        return {"schema_version": 1, "artifacts_root": str(root), "cases": [], "pass": False,
                "errors": ["specification must be an object"]}
    status_name = specification.get("status_file")
    cases = specification.get("cases")
    if not isinstance(status_name, str) or not isinstance(cases, dict) or not cases:
        return {"schema_version": 1, "artifacts_root": str(root), "cases": [], "pass": False,
                "errors": ["specification requires status_file and a non-empty cases object"]}
    try:
        status_path = _within_root(root, status_name)
        status_rows = _read_csv_rows(status_path)
    except (OSError, ValueError, csv.Error) as exc:
        return {"schema_version": 1, "artifacts_root": str(root), "cases": [], "pass": False,
                "errors": [f"status unavailable: {exc}"]}

    rows: list[dict[str, Any]] = []
    for name, raw_case in cases.items():
        case_errors: list[str] = []
        row: dict[str, Any] = {"case": name, "errors": case_errors}
        spec = raw_case if isinstance(raw_case, dict) else {}
        matching_status = [item for item in status_rows if item.get("case") == name]
        if len(matching_status) != 1:
            case_errors.append("status.csv must contain exactly one row for case")
            row.update({section: {"pass": False} for section in ("completion", "artifacts", "metrics", "physics")})
            row["pass"] = False
            rows.append(row)
            continue
        status = matching_status[0]
        exit_code = status.get("exit_code")
        completion_ok = exit_code == "0"
        row["completion"] = {"pass": completion_ok, "exit_code": exit_code}
        expected_log = spec.get("log")
        status_log = status.get("log")
        if not isinstance(expected_log, str) or not isinstance(status_log, str):
            case_errors.append("case log and status log must be relative paths")
            log_path = None
        else:
            try:
                log_path = _within_root(root, expected_log)
                status_candidate = _within_root(root, status_log)
                if status_candidate != log_path:
                    raise ValueError("status log does not resolve to specified contained log")
            except ValueError as exc:
                case_errors.append(str(exc))
                log_path = None
        if log_path is None or not log_path.is_file() or log_path.stat().st_size == 0:
            row["artifacts"] = {"pass": False}
            text = ""
        else:
            row["artifacts"] = {"pass": True, "log": str(log_path.relative_to(root))}
            text = log_path.read_text(encoding="utf-8", errors="replace")
        metric_patterns = _required_regexes(spec.get("required_metrics"), "required_metrics", case_errors)
        matched_metrics = [pattern for pattern in metric_patterns if re.search(pattern, text, re.MULTILINE)]
        row["metrics"] = {"pass": bool(metric_patterns) and len(matched_metrics) == len(metric_patterns),
                          "matched": matched_metrics}
        physics_pattern = spec.get("physics_pass")
        if not isinstance(physics_pattern, str):
            case_errors.append("physics_pass must be a regular expression")
            physics_ok = False
        else:
            physics_ok = re.search(physics_pattern, text, re.MULTILINE) is not None
        row["physics"] = {"pass": physics_ok}
        row["pass"] = all(row[section]["pass"] for section in ("completion", "artifacts", "metrics", "physics"))
        rows.append(row)

    observed_summary: dict[str, dict[str, int]] = {}
    for prefix, label in (("rossiter_", "rossiter"), ("te_", "tail_edge")):
        population = [item for item in status_rows if item.get("case", "").startswith(prefix)]
        passed = 0
        for item in population:
            try:
                candidate = _within_root(root, item.get("log", ""))
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                continue
            # A zero process exit is not enough: count only a terminal physics
            # PASS produced by the solver log itself.
            physics_pass = bool(re.search(r"(?m)^  PASS — .*基准测试$", text)) or bool(
                re.search(r"(?m)^  状态            : ✓ PASS$", text)
            )
            if item.get("exit_code") == "0" and physics_pass:
                passed += 1
        if population:
            observed_summary[label] = {"passed": passed, "total": len(population)}
    expected_summary = specification.get("expected_summary")
    summary_ok = expected_summary is None or observed_summary == expected_summary
    if not summary_ok:
        errors.append("observed artifact summary does not match expected_summary")
    return {"schema_version": 1, "artifacts_root": str(root), "cases": rows, "summary": observed_summary,
            "recommended_tail_edge_default": specification.get("recommended_tail_edge_default"),
            "pass": bool(rows) and all(row["pass"] for row in rows) and summary_ok, "errors": errors}


def _is_finite_json(value: Any) -> bool:
    """Return false for null/non-finite values anywhere in a KPI payload."""
    if value is None:
        return False
    if isinstance(value, bool) or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_is_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(_is_finite_json(item) for item in value.values())
    return False


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _within_root(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes artifacts root: {relative_path}") from exc
    return candidate


def _physics_result(
    spec: dict[str, Any], root: Path, run_status: dict[str, Any], errors: list[str],
) -> dict[str, Any]:
    """Resolve an explicit KPI result from a report or completed-run status.

    ``status_metrics`` intentionally requires an opt-in in the gate config.
    A run's own metrics are a valid physics result only when that completed
    run reported ``metrics.pass: true``; this never upgrades legacy failures.
    """
    physics = spec.get("physics")
    if not isinstance(physics, dict):
        return {"pass": False, "reason": "missing explicit physics KPI result"}
    status_metrics = physics.get("status_metrics")
    if status_metrics is not None:
        if status_metrics is not True:
            return {"pass": False, "reason": "physics.status_metrics must be true"}
        if physics.get("report") is not None:
            return {"pass": False, "reason": "physics cannot select both report and status_metrics"}
        result = run_status.get("metrics")
        if not isinstance(result, dict):
            return {"pass": False, "reason": "run status has no metrics object"}
        passed = result.get("pass") is True
        finite = _is_finite_json(result)
        return {
            "pass": passed and finite,
            "source": "status_metrics",
            "reported_pass": passed,
            "finite_metrics": finite,
            "result": result,
        }

    result = dict(physics)
    report_path = physics.get("report")
    if report_path is not None:
        if not isinstance(report_path, str):
            return {"pass": False, "reason": "physics.report must be a relative path"}
        try:
            external = _read_json(_within_root(root, report_path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"physics report unavailable: {exc}")
            return {"pass": False, "reason": "physics report unavailable"}
        result = external
        case_name = physics.get("case")
        if case_name is not None:
            cases = external.get("cases")
            if not isinstance(cases, list):
                return {"pass": False, "reason": "physics report has no cases"}
            match = next((row for row in cases if isinstance(row, dict) and row.get("case") == case_name), None)
            if match is None:
                return {"pass": False, "reason": f"physics case not found: {case_name}"}
            result = match
    passed = result.get("pass") is True
    finite = _is_finite_json(result.get("metrics", {}))
    return {"pass": passed and finite, "reported_pass": passed, "finite_metrics": finite, "result": result}


def evaluate_regression_gate(artifacts_root: str | Path, cases: dict[str, Any]) -> dict[str, Any]:
    """Evaluate declarative cases into a machine-readable, fail-closed manifest."""
    root = Path(artifacts_root).resolve()
    report_cases: list[dict[str, Any]] = []
    for name, raw_spec in cases.items():
        errors: list[str] = []
        spec = raw_spec if isinstance(raw_spec, dict) else {}
        row: dict[str, Any] = {"case": name, "errors": errors}
        run_dir_name = spec.get("run_dir", name)
        if not isinstance(run_dir_name, str):
            errors.append("run_dir must be a relative path")
            row.update({"completion": {"pass": False}, "artifacts": {"pass": False}, "numerics": {"pass": False}, "physics": {"pass": False}, "pass": False})
            report_cases.append(row)
            continue
        try:
            run_dir = _within_root(root, run_dir_name)
            status_path = _within_root(run_dir, "run_status.json")
            status = _read_json(status_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"status unavailable: {exc}")
            row.update({"completion": {"pass": False}, "artifacts": {"pass": False}, "numerics": {"pass": False}, "physics": {"pass": False}, "pass": False})
            report_cases.append(row)
            continue

        requested = status.get("requested_steps")
        completed = status.get("completed_steps")
        completion_ok = (
            status.get("state") in _TERMINAL_SUCCESS
            and isinstance(requested, int) and requested >= 0
            and completed == requested
            and status.get("numerical_failure") is None
        )
        row["completion"] = {"pass": completion_ok, "state": status.get("state"), "requested_steps": requested, "completed_steps": completed}

        required = spec.get("required_artifacts", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            errors.append("required_artifacts must be a list of relative paths")
            missing = ["<invalid specification>"]
        else:
            missing = []
            for relative in required:
                try:
                    item = _within_root(run_dir, relative)
                    if not item.is_file() or item.stat().st_size == 0:
                        missing.append(relative)
                except (OSError, ValueError):
                    missing.append(relative)
        row["artifacts"] = {"pass": not missing, "missing": missing}

        metrics = status.get("metrics")
        numerics_ok = (
            isinstance(metrics, dict)
            and status.get("numerical_failure") is None
            and _is_finite_json(metrics)
        )
        row["numerics"] = {"pass": numerics_ok, "metrics_present": isinstance(metrics, dict)}
        row["physics"] = _physics_result(spec, root, status, errors)
        row["pass"] = all(bool(row[section]["pass"]) for section in ("completion", "artifacts", "numerics", "physics"))
        report_cases.append(row)
    return {"schema_version": 1, "artifacts_root": str(root), "cases": report_cases, "pass": bool(report_cases) and all(row["pass"] for row in report_cases)}


def write_regression_manifest(path: str | Path, report: dict[str, Any]) -> None:
    """Atomically persist a strict JSON gate result."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
