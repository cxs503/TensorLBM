"""XFlow-style real-time field streaming service.

Polls active simulation engines, downsamples velocity/pressure fields,
and pushes them to subscribed clients via the job_manager diagnostic
channel (which the WebSocket broadcaster forwards to all connected clients).

Design:
- Lightweight: no torch operations on the streaming thread
- Adaptive: downsampling respects a target byte budget per frame
- Backpressure-safe: dropped frames are logged but never block the solver
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..routers.simulations import _jobs  # type: ignore[attr-defined]

logger = logging.getLogger("tensorlbm.streaming")

# Target bandwidth: ~256 KB per frame, 2 FPS → ~512 KB/s per stream
TARGET_FRAME_BYTES = 256 * 1024
DEFAULT_FPS = 2.0


@dataclass(slots=True)
class StreamSubscriber:
    """A client subscribed to a job's live field updates."""
    subscriber_id: str
    job_id: str
    fps: float = DEFAULT_FPS
    max_bytes: int = TARGET_FRAME_BYTES
    channel: str = "field"  # 'field' (velocity/pressure) | 'progress' | 'all'
    last_sent_step: int = 0
    last_sent_ts: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class StreamingHub:
    """Singleton hub coordinating active streams and the polling thread."""

    _subscribers: dict[str, StreamSubscriber] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _poll_thread: threading.Thread | None = None
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _jobs_ref: Any = None  # Will be set to routers.simulations._jobs

    def attach(self, jobs_ref: Any) -> None:
        """Bind the simulations job registry. Call once at startup."""
        self._jobs_ref = jobs_ref
        if self._poll_thread is None or not self._poll_thread.is_alive():
            self._stop_event.clear()
            self._poll_thread = threading.Thread(
                target=self._poll_loop, name="streaming-poll", daemon=True
            )
            self._poll_thread.start()
            logger.info("StreamingHub polling thread started")

    def subscribe(self, sub: StreamSubscriber) -> None:
        with self._lock:
            self._subscribers[sub.subscriber_id] = sub
        logger.info("Subscriber %s attached to job=%s channel=%s", sub.subscriber_id, sub.job_id, sub.channel)

    def unsubscribe(self, subscriber_id: str) -> bool:
        with self._lock:
            return self._subscribers.pop(subscriber_id, None) is not None

    def list_subscribers(self, job_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            subs = self._subscribers.values()
        items = [
            {
                "subscriber_id": s.subscriber_id,
                "job_id": s.job_id,
                "fps": s.fps,
                "channel": s.channel,
                "last_sent_step": s.last_sent_step,
            }
            for s in subs
            if job_id is None or s.job_id == job_id
        ]
        return items

    def _poll_loop(self) -> None:
        """Periodically check active jobs and push field frames."""
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                logger.exception("Streaming poll tick failed")
            self._stop_event.wait(0.25)  # 4 Hz poll, sub-sample per subscriber FPS

    def _tick(self) -> None:
        if self._jobs_ref is None:
            return
        now = time.monotonic()
        with self._lock:
            subs = list(self._subscribers.values())
        for sub in subs:
            interval = 1.0 / max(sub.fps, 0.1)
            if now - sub.last_sent_ts < interval:
                continue
            frame = self._extract_frame(sub)
            if frame is None:
                continue
            sub.last_sent_ts = now
            sub.last_sent_step = frame.get("step", 0)
            # Push via job_manager diagnostic → WebSocket broadcaster
            try:
                from .. import job_manager  # local import to avoid cycles
                job_manager.push_diagnostic(sub.job_id, {
                    "kind": "stream_frame",
                    "channel": sub.channel,
                    "frame": frame,
                })
            except Exception:  # noqa: BLE001
                logger.exception("Failed to push diagnostic for sub=%s", sub.subscriber_id)

    def _extract_frame(self, sub: StreamSubscriber) -> dict[str, Any] | None:
        """Read current state of the job's engine and produce a downsampled frame."""
        with self._jobs_ref[1]:  # _job_lock
            job = self._jobs_ref[0].get(sub.job_id)
        if not job or sub.job_id not in self._jobs_ref[0]:
            return None
        engine = job.get("engine")
        if engine is None or getattr(engine, "step_count", 0) <= sub.last_sent_step:
            return None  # no new data
        snap = self._latest_snapshot(engine)
        step = engine.step_count
        forces = self._latest_forces(engine)
        return {
            "step": step,
            "ts": time.time(),
            "status": job.get("status", "running"),
            "snapshot": snap,
            "forces": forces,
        }

    @staticmethod
    def _latest_snapshot(engine: Any) -> dict[str, Any] | None:
        snaps = getattr(engine, "snapshots", None)
        if not snaps:
            return None
        snap = snaps[-1]
        # Convert tensors to numpy, downsample to ~64³ for streaming
        out: dict[str, Any] = {}
        for key in ("ux", "uy", "uz", "rho", "p"):
            arr = snap.get(key)
            if arr is None:
                continue
            try:
                if hasattr(arr, "detach"):
                    arr = arr.detach().cpu().numpy()
                arr = np.asarray(arr)
                # Downsample to ~32 along largest dim
                target = 32
                if arr.size > target ** 3:
                    factor = max(1, int(np.cbrt(arr.size / (target ** 3))))
                    arr = arr[::factor, ::factor, ::factor]
                # Clamp size to budget
                if arr.nbytes > TARGET_FRAME_BYTES // 4:
                    arr = arr[:32, :32, :32]
                out[key] = arr.astype(np.float32).tolist() if arr.size < 4096 else None
            except Exception:  # noqa: BLE001
                continue
        return out if out else None

    @staticmethod
    def _latest_forces(engine: Any) -> dict[str, float] | None:
        log = getattr(engine, "forces_log", None)
        if not log:
            return None
        last = log[-1]
        try:
            return {k: float(v) for k, v in last.items() if isinstance(v, (int, float))}
        except Exception:  # noqa: BLE001
            return None

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)


# Singleton
hub = StreamingHub()