"""Webhook / email notification system for TensorLBM platform.

Sends configurable push notifications when simulation jobs complete (or fail),
so users are alerted without having to poll the platform.  Analogous to the
notification hooks in PowerFlow Enterprise and XFlow Cloud.

Configuration
-------------
Webhooks are registered per-job at submission time via the
``notification_webhook`` field in any solver request.  The platform calls the
webhook URL with a POST request containing the job summary JSON.

Alternatively, a global default webhook can be set via the environment
variable ``TENSORLBM_NOTIFY_WEBHOOK``.

API
---
* ``POST /api/notifications/webhook-test``   – test a webhook URL.
* ``GET  /api/notifications/settings``       – current global settings.
* ``POST /api/notifications/settings``       – update global settings.

Programmatic use (from job callbacks)
--------------------------------------
::

    from app.backend.routers.notifications import notify_job_event

    await notify_job_event(job, event="completed")
"""
from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("tensorlbm.notifications")
router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory global settings (updated at runtime)
# ---------------------------------------------------------------------------

_global_settings: dict[str, Any] = {
    "webhook_url": os.environ.get("TENSORLBM_NOTIFY_WEBHOOK", ""),
    "notify_on_complete": True,
    "notify_on_failure": True,
    "notify_on_cancel": False,
    "timeout_s": 10,
}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class WebhookTestRequest(BaseModel):
    url: str = Field(..., description="Webhook URL to test (HTTP POST).")
    payload: dict[str, Any] | None = Field(
        default=None,
        description="Custom JSON payload.  Defaults to a standard test event.",
    )


class NotificationSettings(BaseModel):
    webhook_url: str = Field(default="", description="Global default webhook URL.")
    notify_on_complete: bool = True
    notify_on_failure: bool = True
    notify_on_cancel: bool = False
    timeout_s: int = Field(default=10, ge=1, le=120)


# ---------------------------------------------------------------------------
# Core delivery function
# ---------------------------------------------------------------------------

async def _post_webhook(url: str, payload: dict[str, Any], timeout_s: int = 10) -> dict[str, Any]:
    """POST *payload* to *url* asynchronously.

    Returns a result dict with ``status``, ``http_status``, ``elapsed_ms``.
    Does **not** raise on HTTP errors – all errors are returned in the dict
    so callers can log without crashing the server.
    """
    start = datetime.now(UTC)
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, json=payload)
        elapsed_ms = int((datetime.now(UTC) - start).total_seconds() * 1000)
        ok = resp.is_success
        logger.info("Webhook %s → HTTP %d (%d ms)", url, resp.status_code, elapsed_ms)
        return {
            "status": "ok" if ok else "http_error",
            "http_status": resp.status_code,
            "elapsed_ms": elapsed_ms,
            "url": url,
        }
    except httpx.TimeoutException:
        elapsed_ms = int((datetime.now(UTC) - start).total_seconds() * 1000)
        logger.warning("Webhook timeout: %s (%d ms)", url, elapsed_ms)
        return {"status": "timeout", "elapsed_ms": elapsed_ms, "url": url}
    except Exception as exc:
        elapsed_ms = int((datetime.now(UTC) - start).total_seconds() * 1000)
        logger.error("Webhook error for %s: %s", url, exc)
        return {"status": "error", "error": str(exc), "elapsed_ms": elapsed_ms, "url": url}


async def notify_job_event(
    job: object,
    event: str = "completed",
    *,
    webhook_url: str | None = None,
) -> dict[str, Any] | None:
    """Send a job-lifecycle notification to the configured webhook.

    This function is called by the job-manager completion hooks and by
    individual router endpoints after key lifecycle events.

    Args:
        job:         Platform ``Job`` object.
        event:       Event name: ``'completed'``, ``'failed'``, ``'cancelled'``.
        webhook_url: Per-job override URL.  Falls back to global setting.

    Returns:
        Delivery result dict, or ``None`` if notifications are disabled.
    """
    settings = _global_settings

    # Determine target URL
    url = webhook_url or settings.get("webhook_url", "")
    if not url:
        return None

    # Check event filter
    should_send = (
        (event == "completed" and settings.get("notify_on_complete", True))
        or (event == "failed" and settings.get("notify_on_failure", True))
        or (event == "cancelled" and settings.get("notify_on_cancel", False))
    )
    if not should_send:
        return None

    payload = {
        "event": event,
        "job_id": job.job_id,
        "job_name": job.name,
        "job_type": job.job_type,
        "status": job.status.value if hasattr(job.status, "value") else str(job.status),
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "run_duration_s": job.run_duration_seconds,
        "error": job.error,
        "platform": "TensorLBM",
        "timestamp": datetime.now(UTC).isoformat(),
    }

    timeout = int(settings.get("timeout_s", 10))
    return await _post_webhook(url, payload, timeout_s=timeout)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.post("/webhook-test")
async def test_webhook(req: WebhookTestRequest) -> dict:
    """Send a test POST request to a webhook URL.

    Useful for verifying connectivity before associating a webhook with a
    live simulation job.

    The default test payload contains a ``test=true`` flag so the receiver
    can distinguish test events from real job notifications.
    """
    url = req.url
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="URL must start with http:// or https://")

    payload = req.payload or {
        "event": "test",
        "platform": "TensorLBM",
        "message": "Webhook connectivity test from TensorLBM platform.",
        "timestamp": datetime.now(UTC).isoformat(),
        "test": True,
    }

    result = await _post_webhook(url, payload, timeout_s=_global_settings.get("timeout_s", 10))
    return result


@router.get("/settings")
async def get_notification_settings() -> dict:
    """Return the current global notification settings.

    The ``webhook_url`` field is partially masked for security.
    """
    settings = dict(_global_settings)
    url = settings.get("webhook_url", "")
    if url and len(url) > 12:
        settings["webhook_url"] = url[:8] + "****" + url[-4:]
    return settings


@router.post("/settings")
async def update_notification_settings(settings: NotificationSettings) -> dict:
    """Update global notification settings.

    Changes take effect immediately for all future job notifications.
    The webhook URL is stored in memory; to persist across restarts set
    ``TENSORLBM_NOTIFY_WEBHOOK`` in the environment.
    """
    global _global_settings
    _global_settings["webhook_url"] = settings.webhook_url
    _global_settings["notify_on_complete"] = settings.notify_on_complete
    _global_settings["notify_on_failure"] = settings.notify_on_failure
    _global_settings["notify_on_cancel"] = settings.notify_on_cancel
    _global_settings["timeout_s"] = settings.timeout_s
    logger.info("Notification settings updated: notify_on_complete=%s notify_on_failure=%s",
                settings.notify_on_complete, settings.notify_on_failure)
    return {"status": "updated", **_global_settings}
