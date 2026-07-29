"""XFlow-style streaming endpoints: SSE and field-specific WebSocket.

Provides three ways to follow a running simulation live:
1. GET /api/stream/{job_id}/events    – Server-Sent Events (one-way, easy)
2. WS  /ws/field/{job_id}             – Field-dedicated WebSocket
3. GET /api/stream/{job_id}/snapshot  – On-demand downsampled snapshot
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from ..services.xflow_streaming import hub, StreamSubscriber
from .. import job_manager

router = APIRouter(prefix="/api/stream", tags=["Streaming"])
logger = logging.getLogger("tensorlbm.streaming")


@router.get("/{job_id}/events")
async def stream_events(
    job_id: str,
    request: Request,
    channel: str = Query("field", pattern="^(field|progress|all)$"),
    fps: float = Query(2.0, ge=0.1, le=30.0),
):
    """Server-Sent Events stream of live simulation diagnostics.

    Each event is a JSON-encoded frame with the current step, downsampled
    velocity/pressure fields, and force coefficients (Cd, Cl, etc.).
    """
    sub_id = f"sse-{uuid.uuid4().hex[:8]}"
    sub = StreamSubscriber(
        subscriber_id=sub_id, job_id=job_id, fps=fps, channel=channel
    )
    hub.subscribe(sub)

    async def event_gen():
        try:
            # Send initial state immediately
            yield f"event: hello\ndata: {json.dumps({'subscriber_id': sub_id, 'job_id': job_id, 'channel': channel, 'fps': fps})}\n\n"
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            # Bridge: hook into the job_manager notification queue
            loop = asyncio.get_running_loop()
            from ..main import _notify_queue  # type: ignore[attr-defined]
            # Inject a listener task
            async def relay() -> None:
                while True:
                    try:
                        msg = await _notify_queue.get()  # type: ignore[union-attr]
                    except asyncio.CancelledError:
                        break
                    if msg.get("job_id") != job_id:
                        continue
                    diags = msg.get("diagnostics", [])
                    for d in diags:
                        if d.get("kind") != "stream_frame":
                            continue
                        if channel not in ("all", d.get("channel", "field")):
                            continue
                        await queue.put(d.get("frame", {}))
            relay_task = asyncio.create_task(relay())
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        frame = await asyncio.wait_for(queue.get(), timeout=5.0)
                        yield f"event: frame\ndata: {json.dumps(frame, default=str)}\n\n"
                    except asyncio.TimeoutError:
                        # Keep-alive ping
                        yield ": keepalive\n\n"
            finally:
                relay_task.cancel()
                hub.unsubscribe(sub_id)
        except asyncio.CancelledError:
            hub.unsubscribe(sub_id)
            raise

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.websocket("/ws/field/{job_id}")
async def ws_field(websocket: WebSocket, job_id: str, channel: str = "field", fps: float = 2.0) -> None:
    """Dedicated WebSocket for live field streaming on a single job."""
    await websocket.accept()
    sub_id = f"ws-{uuid.uuid4().hex[:8]}"
    sub = StreamSubscriber(subscriber_id=sub_id, job_id=job_id, fps=fps, channel=channel)
    hub.subscribe(sub)
    try:
        await websocket.send_json({"type": "hello", "subscriber_id": sub_id, "channel": channel, "fps": fps})
        # Pull frames from the global notification queue
        from ..main import _notify_queue  # type: ignore[attr-defined]
        loop = asyncio.get_running_loop()
        async def relay() -> None:
            while True:
                try:
                    msg = await _notify_queue.get()  # type: ignore[union-attr]
                except asyncio.CancelledError:
                    break
                if msg.get("job_id") != job_id:
                    continue
                for d in msg.get("diagnostics", []):
                    if d.get("kind") != "stream_frame":
                        continue
                    if channel not in ("all", d.get("channel", "field")):
                        continue
                    try:
                        await websocket.send_json({"type": "frame", "frame": d.get("frame", {})})
                    except Exception:  # noqa: BLE001
                        return
        relay_task = asyncio.create_task(relay())
        try:
            while True:
                # Allow client to change channel/fps
                raw = await websocket.receive_text()
                try:
                    cmd = json.loads(raw)
                    if cmd.get("op") == "set_fps":
                        sub.fps = float(cmd.get("fps", 2.0))
                    elif cmd.get("op") == "set_channel":
                        sub.channel = cmd.get("channel", "field")
                except Exception:  # noqa: BLE001
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            relay_task.cancel()
    finally:
        hub.unsubscribe(sub_id)


@router.get("/{job_id}/snapshot")
async def get_snapshot(job_id: str):
    """Return the latest downsampled snapshot (one-shot, no subscription)."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    # Use the streaming hub's extraction (one-shot, no subscription)
    sub = StreamSubscriber(subscriber_id=f"oneshot-{uuid.uuid4().hex[:6]}", job_id=job_id, fps=0.1, channel="field")
    frame = hub._extract_frame(sub)
    return {"job_id": job_id, "frame": frame}


@router.get("/_stats")
async def stats():
    """Diagnostics: active subscribers and per-job counts."""
    return {
        "active_subscribers": hub.list_subscribers(),
        "active_jobs": len(job_manager.list_jobs()),
    }