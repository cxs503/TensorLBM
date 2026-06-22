"""Report generation endpoints for the TensorLBM platform.

Generates structured HTML engineering reports from completed simulation jobs.
The report bundles job metadata, configuration, convergence history,
force coefficients, and result images in a self-contained HTML file.
"""
from __future__ import annotations

import base64
import json
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from .. import job_manager

if TYPE_CHECKING:
    from pathlib import Path

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _img_to_data_uri(path: Path) -> str:
    """Encode a PNG/JPG file as a data URI."""
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def _load_forces_csv(job: job_manager.Job) -> list[dict[str, Any]]:
    """Read forces.csv from the job output directory if present."""
    csv_path = job.output_dir / "forces.csv"
    if not csv_path.exists():
        # Scan sub-directories
        candidates = list(job.output_dir.rglob("forces.csv"))
        if not candidates:
            return []
        csv_path = candidates[0]
    import csv
    rows: list[dict[str, Any]] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: _try_float(v) for k, v in row.items()})
    return rows


def _try_float(val: str | None) -> float | str | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return val


def _load_metadata(job: job_manager.Job) -> dict[str, Any]:
    """Try to load run_metadata.json from the job output."""
    candidates = list(job.output_dir.rglob("run_metadata.json"))
    if not candidates:
        return {}
    try:
        return json.loads(candidates[0].read_text())
    except Exception:
        return {}


def _list_images(job: job_manager.Job) -> list[Path]:
    return sorted(job.output_dir.rglob("*.png"))[:20]  # cap at 20 images


def _numeric_series(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[float]:
    vals: list[float] = []
    for row in rows:
        for key in keys:
            val = row.get(key)
            if isinstance(val, (int, float)):
                vals.append(float(val))
                break
    return vals


def _tail_stats(values: list[float], window: int = 20) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    tail = values[-min(window, len(values)) :]
    mean = sum(tail) / len(tail)
    if len(tail) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in tail) / len(tail)
    return mean, math.sqrt(var)


def _steady_state_score(*series: list[float]) -> float | None:
    rel_stds: list[float] = []
    for values in series:
        mean, std = _tail_stats(values)
        if mean is None or std is None or abs(mean) < 1e-12:
            continue
        rel_stds.append(std / abs(mean))
    if not rel_stds:
        return None
    worst = max(rel_stds)
    return max(0.0, 1.0 - min(1.0, worst * 10.0))


def compute_engineering_kpis(job: job_manager.Job) -> dict[str, Any]:
    """Compute reusable engineering KPIs from diagnostics and force history."""
    forces = _load_forces_csv(job)
    diag = job.diagnostics

    cd_values = _numeric_series(forces, ("Cd", "cd", "drag"))
    cl_values = _numeric_series(forces, ("Cl", "cl", "lift"))
    if not cd_values:
        cd_values = _numeric_series(diag, ("Cd", "cd", "drag"))
    if not cl_values:
        cl_values = _numeric_series(diag, ("Cl", "cl", "lift"))

    mean_cd, std_cd = _tail_stats(cd_values)
    mean_cl, std_cl = _tail_stats(cl_values)
    steady_score = _steady_state_score(cd_values, cl_values)

    latest_step = None
    for row in reversed(diag):
        step = row.get("step") or row.get("t") or row.get("iter")
        if isinstance(step, (int, float)):
            latest_step = int(step)
            break

    return {
        "diagnostic_snapshots": len(diag),
        "force_rows": len(forces),
        "image_count": len(_list_images(job)),
        "runtime_seconds": job.run_duration_seconds,
        "latest_step": latest_step,
        "mean_cd_last": mean_cd,
        "std_cd_last": std_cd,
        "mean_cl_last": mean_cl,
        "std_cl_last": std_cl,
        "steady_state_score": steady_score,
        "steady_state_detected": (
            steady_score is not None and steady_score >= 0.7
        ),
    }


def _report_summary_payload(job: job_manager.Job) -> dict[str, Any]:
    """Return a structured summary shared by single-job and compare endpoints."""
    forces = _load_forces_csv(job)
    meta = _load_metadata(job)
    image_count = len(_list_images(job))
    diag = job.diagnostics
    kpis = compute_engineering_kpis(job)
    return {
        "job_id": job.job_id,
        "name": job.name,
        "job_type": job.job_type,
        "status": job.status.value,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error": job.error,
        "diagnostic_steps": len(diag),
        "force_rows": len(forces),
        "image_count": image_count,
        "run_metadata_available": bool(meta),
        "report_url": f"/api/reports/{job.job_id}",
        "engineering_kpis": kpis,
    }


def _flatten_compare_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """Flatten numeric report-summary metrics for comparison tables."""
    out: dict[str, float] = {}
    for key in ("diagnostic_steps", "force_rows", "image_count"):
        value = summary.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = float(value)
    for key, value in (summary.get("engineering_kpis") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = float(value)
    return out


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def _build_html_report(job: job_manager.Job) -> str:
    """Build a self-contained HTML engineering report for a job."""
    forces = _load_forces_csv(job)
    images = _list_images(job)
    kpis = compute_engineering_kpis(job)

    # ---- convergence table data ----------------------------------------
    diag = job.diagnostics  # list of dicts with step, Cd, Cl, etc.

    # ---- mini chart (ASCII-style data URIs not needed; use inline SVG) ---
    # Build a simple SVG line chart from diagnostics if available
    def _svg_line_chart(
        data: list[float], label: str, color: str = "#0d6efd", width: int = 500, height: int = 120
    ) -> str:
        if not data:
            return ""
        mn, mx = min(data), max(data)
        rng = mx - mn or 1.0
        n = len(data)
        pts = []
        for i, v in enumerate(data):
            x = int(i / max(n - 1, 1) * (width - 40)) + 20
            y = height - 20 - int((v - mn) / rng * (height - 40))
            pts.append(f"{x},{y}")
        polyline = " ".join(pts)
        return (
            f'<svg width="{width}" height="{height}" style="display:block">'
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<text x="20" y="{height-4}" font-size="10" fill="#666">'
            f"{label} min={mn:.4g} max={mx:.4g}</text>"
            f"</svg>"
        )

    # Build drag chart
    cd_vals = [d.get("Cd") or d.get("cd") or d.get("drag") for d in diag if d]
    cd_vals = [v for v in cd_vals if v is not None]
    drag_svg = _svg_line_chart(cd_vals, "Drag coefficient Cd", "#0d6efd")

    # ---- images section --------------------------------------------------
    img_html_parts: list[str] = []
    for img_path in images:
        try:
            uri = _img_to_data_uri(img_path)
            img_html_parts.append(
                f'<div style="display:inline-block;margin:4px">'
                f'<img src="{uri}" '
                f'style="max-width:300px;max-height:220px;'
                f'border-radius:4px;border:1px solid #dee2e6" '
                f'title="{img_path.name}"/>'
                f'<div style="font-size:11px;color:#666;text-align:center">'
                f"{img_path.name}</div>"
                f"</div>"
            )
        except Exception:
            pass
    images_html = (
        "\n".join(img_html_parts)
        if img_html_parts
        else "<p><em>No images available.</em></p>"
    )

    # ---- forces table ---------------------------------------------------
    if forces:
        keys = list(forces[0].keys())
        header = "".join(f"<th>{k}</th>" for k in keys)
        rows_html = ""
        for row in forces[-20:]:  # last 20 rows
            rows_html += "<tr>" + "".join(
                f"<td>{_format_cell(row[k])}</td>"
                for k in keys
            ) + "</tr>"
        forces_table = (
            f'<table class="table table-sm table-striped"><thead><tr>{header}</tr></thead>'
            f"<tbody>{rows_html}</tbody></table>"
        )
    else:
        forces_table = "<p><em>No forces.csv available.</em></p>"

    # ---- metadata section -----------------------------------------------
    config_json = json.dumps(job.config, indent=2)
    result_json = json.dumps(job.result, indent=2)
    kpi_rows = [
        ("Latest step", kpis["latest_step"]),
        ("Runtime (s)", kpis["runtime_seconds"]),
        ("Mean Cd (tail)", kpis["mean_cd_last"]),
        ("Std Cd (tail)", kpis["std_cd_last"]),
        ("Mean Cl (tail)", kpis["mean_cl_last"]),
        ("Std Cl (tail)", kpis["std_cl_last"]),
        ("Steady-state score", kpis["steady_state_score"]),
        ("Steady-state detected", kpis["steady_state_detected"]),
    ]
    kpi_table = "".join(
        "<tr>"
        f"<th>{label}</th>"
        f"<td>{'—' if value is None else _format_cell(value)}</td>"
        "</tr>"
        for label, value in kpi_rows
    )
    error_html = (
        '<div class="alert alert-danger mt-2"><strong>Error:</strong> '
        + job.error
        + "</div>"
        if job.error
        else ""
    )

    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    status_color = {
        "completed": "#198754", "failed": "#dc3545",
        "running": "#ffc107", "queued": "#6c757d",
    }.get(job.status.value, "#0d6efd")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>TensorLBM – Report: {job.name}</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"/>
<style>
  body {{ font-family: system-ui, sans-serif; background:#f8f9fa; }}
  .report-header {{
    background: linear-gradient(135deg,#1a3a5c 0%,#0d6efd 100%);
    color:#fff; padding:2rem 2rem 1.5rem;
  }}
  .report-header h1 {{ font-size:1.6rem; font-weight:700; }}
  .report-header .sub {{ opacity:.8; font-size:.9rem; }}
  .section {{ background:#fff; border-radius:.5rem;
               box-shadow:0 1px 4px rgba(0,0,0,.08);
               padding:1.25rem; margin-bottom:1.25rem; }}
  .section h5 {{ font-weight:700; border-bottom:1px solid #e9ecef;
                  padding-bottom:.5rem; margin-bottom:1rem; }}
  pre {{ background:#1e2a3a; color:#a8d8a8; border-radius:.35rem;
         padding:.75rem; font-size:.8rem; overflow-x:auto; }}
  .status-pill {{
    display:inline-block; padding:.25rem .75rem;
    border-radius:999px; font-size:.85rem; font-weight:600;
    background:{status_color}; color:#fff;
  }}
</style>
</head>
<body>
<div class="report-header">
  <h1>&#x1F30A; TensorLBM &mdash; Engineering Report</h1>
  <div class="sub">
    <strong>Job:</strong> {job.name} &nbsp;|&nbsp;
    <strong>ID:</strong> <code style="color:#aef">{job.job_id}</code> &nbsp;|&nbsp;
    <strong>Type:</strong> {job.job_type} &nbsp;|&nbsp;
    <span class="status-pill">{job.status.value.upper()}</span>
  </div>
  <div class="sub mt-1">Generated: {now_str}</div>
</div>

<div class="container-fluid py-3">

  <!-- Summary -->
  <div class="section">
    <h5>&#128203; Summary</h5>
    <div class="row">
      <div class="col-md-4"><strong>Created:</strong> {job.created_at}</div>
      <div class="col-md-4"><strong>Started:</strong> {job.started_at or '—'}</div>
      <div class="col-md-4"><strong>Completed:</strong> {job.completed_at or '—'}</div>
    </div>
    {error_html}
  </div>

  <!-- Engineering KPIs -->
  <div class="section">
    <h5>&#129516; Engineering KPIs</h5>
    <div class="table-responsive">
      <table class="table table-sm table-bordered mb-0">
        <tbody>{kpi_table}</tbody>
      </table>
    </div>
  </div>

  <!-- Convergence -->
  <div class="section">
    <h5>&#128200; Convergence History</h5>
    {drag_svg if drag_svg else '<p><em>No convergence data.</em></p>'}
    <p class="text-muted small mt-2">
      Total diagnostics snapshots: {len(diag)}. Last 5 entries shown below.
    </p>
    <pre>{json.dumps(diag[-5:], indent=2) if diag else '[]'}</pre>
  </div>

  <!-- Force coefficients -->
  <div class="section">
    <h5>&#128207; Force Coefficients (last 20 rows of forces.csv)</h5>
    {forces_table}
  </div>

  <!-- Result images -->
  <div class="section">
    <h5>&#128247; Result Images</h5>
    <div style="display:flex;flex-wrap:wrap;gap:8px">
      {images_html}
    </div>
  </div>

  <!-- Run configuration -->
  <div class="section">
    <h5>&#9881; Run Configuration</h5>
    <pre>{config_json}</pre>
  </div>

  <!-- Result metadata -->
  <div class="section">
    <h5>&#128196; Result Metadata</h5>
    <pre>{result_json}</pre>
  </div>

  <!-- Log (last 50 lines) -->
  <div class="section">
    <h5>&#128220; Solver Log (last 50 lines)</h5>
    <pre>{chr(10).join(job.logs[-50:])}</pre>
  </div>

</div>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/{job_id}", response_class=HTMLResponse)
async def get_report(job_id: str) -> HTMLResponse:
    """Generate and return an HTML engineering report for a completed job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    html = _build_html_report(job)
    return HTMLResponse(content=html)


@router.get("/{job_id}/summary")
async def get_report_summary(job_id: str) -> dict:
    """Return a JSON summary of key report metrics without the full HTML."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _report_summary_payload(job)


@router.get("/compare/kpis")
async def compare_reports(
    ids: Annotated[list[str], Query(min_length=2, max_length=10)],
) -> dict[str, Any]:
    """Compare report summaries and engineering KPIs for multiple jobs."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for job_id in ids:
        job = job_manager.get_job(job_id)
        if job is None:
            missing.append(job_id)
            continue
        summary = _report_summary_payload(job)
        rows.append({
            **summary,
            "compare_metrics": _flatten_compare_metrics(summary),
        })

    if not rows:
        raise HTTPException(status_code=404, detail="No matching jobs found")

    metric_summary: dict[str, dict[str, float | str]] = {}
    metric_keys = sorted({k for row in rows for k in row["compare_metrics"]})
    for key in metric_keys:
        vals = [
            (row["job_id"], float(row["compare_metrics"][key]))
            for row in rows
            if key in row["compare_metrics"]
        ]
        if not vals:
            continue
        best_job_id, best_value = min(vals, key=lambda item: item[1])
        only_vals = [value for _, value in vals]
        metric_summary[key] = {
            "min": min(only_vals),
            "max": max(only_vals),
            "mean": sum(only_vals) / len(only_vals),
            "best_job_id": best_job_id,
            "best_value": best_value,
        }

    return {
        "count": len(rows),
        "rows": rows,
        "missing": missing,
        "metric_summary": metric_summary,
    }
