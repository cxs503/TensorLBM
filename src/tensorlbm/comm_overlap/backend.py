"""Communication backends for the TCLB-style wait-any overlap port.

Two interchangeable backends share one interface so the *same* schedule code can
be exercised (a) against a real process group and (b) against a deterministic
virtual clock used for the latency/bandwidth regime study:

``TorchDistBackend``
    Real point-to-point traffic on a ``torch.distributed`` gloo group.  ``waitany``
    is the polling analogue of ``MPI_Waitany``: it blocks until *any* outstanding
    receive completes and returns that one handle.

``VirtualClockBackend``
    No traffic at all: charges a configurable latency/bandwidth/pack/compute cost
    model to a virtual clock, and fills each receive buffer from a ground-truth
    supplier when the modelled transfer "completes".  Deterministic (no wall
    clock), which is what makes the overlap ratio a reproducible measurement
    rather than a noisy one.

Reference design: ``CFD-GO/TCLB`` ``src/Lattice.cu.Rt:327-386`` (``MPIStream_A`` /
``MPIStream_B`` with ``CROSS_MPI_WAITANY``) and ``src/Lattice.h.Rt:70-72``
(separate ``inStream``/``outStream`` for the H2D/D2H legs).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class BackendStats:
    """Measured (or modelled) cost split of one exchange schedule."""

    posts_recv: int = 0
    posts_send: int = 0
    waitany_calls: int = 0
    waitall_calls: int = 0
    stall_us: float = 0.0
    compute_us: dict[str, float] = field(default_factory=dict)
    events: list[tuple[float, str, str]] = field(default_factory=list)
    clock_us: float = 0.0

    def add_compute(self, kind: str, amount: float) -> None:
        self.compute_us[kind] = self.compute_us.get(kind, 0.0) + float(amount)

    @property
    def total_compute_us(self) -> float:
        return float(sum(self.compute_us.values()))

    @property
    def step_us(self) -> float:
        return self.clock_us

    @property
    def stall_fraction(self) -> float:
        return self.stall_us / self.clock_us if self.clock_us > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "posts_recv": self.posts_recv,
            "posts_send": self.posts_send,
            "waitany_calls": self.waitany_calls,
            "waitall_calls": self.waitall_calls,
            "stall_us": round(self.stall_us, 6),
            "compute_us": {k: round(v, 6) for k, v in sorted(self.compute_us.items())},
            "total_compute_us": round(self.total_compute_us, 6),
            "clock_us": round(self.clock_us, 6),
            "stall_fraction": round(self.stall_fraction, 6),
            "events": [[round(t, 6), k, n] for t, k, n in self.events],
        }


class TorchDistBackend:
    """Point-to-point backend on an initialised ``torch.distributed`` group."""

    def __init__(
        self,
        send_plan: dict[str, tuple[int, int]],
        recv_plan: dict[str, tuple[int, int]],
        poll_sleep: float = 2e-5,
    ) -> None:
        """``poll_sleep`` is kept for API compatibility with poll-based backends."""
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("TorchDistBackend requires an initialized process group")
        self._dist = dist
        self.send_plan = dict(send_plan)
        self.recv_plan = dict(recv_plan)
        self.poll_sleep = float(poll_sleep)
        self.stats = BackendStats()
        self._t0 = time.perf_counter()
        self._recvs: dict[str, Any] = {}
        self._sends: dict[str, Any] = {}

    # -- clock ---------------------------------------------------------
    def _now_us(self) -> float:
        return (time.perf_counter() - self._t0) * 1e6

    def reset(self) -> None:
        self.stats = BackendStats()
        self._t0 = time.perf_counter()

    def tick(self) -> None:
        self.stats.clock_us = self._now_us()

    def close(self) -> None:
        self.tick()

    # -- schedule primitives -------------------------------------------
    def post_recv(self, name: str, buffer: Any) -> str:
        src, tag = self.recv_plan[name]
        self.stats.posts_recv += 1
        self.stats.events.append((self._now_us(), "post_recv", name))
        work = self._dist.irecv(buffer, src=src, tag=tag)
        return (name, work)

    def post_send(self, name: str, payload: Any) -> str:
        dst, tag = self.send_plan[name]
        self.stats.posts_send += 1
        self.stats.events.append((self._now_us(), "post_send", name))
        work = self._dist.isend(payload, dst=dst, tag=tag)
        return (name, work)

    def waitany(self, handles: Sequence[Any]) -> int:
        """Block until some receive completes; return its index (MPI_Waitany)."""
        if not handles:
            raise ValueError("waitany requires at least one handle")
        self.stats.waitany_calls += 1
        t_start = self._now_us()
        # gloo in this build exposes no progress engine: a receive only completes
        # inside its own wait(), so completion order degenerates to declaration
        # order.  The *schedule* still consumes exactly one buffer per return,
        # which is what the callers depend on; overlap is quantified on the
        # deterministic backend instead (documented limitation).
        name, work = handles[0]
        work.wait()
        self.stats.stall_us += self._now_us() - t_start
        self.stats.events.append((self._now_us(), "waitany_done", name))
        return 0

    def waitall(self, handles: Sequence[Any]) -> None:
        self.stats.waitall_calls += 1
        t_start = self._now_us()
        for _name, work in handles:
            work.wait()
        self.stats.stall_us += self._now_us() - t_start

    def deliver(self, name: str) -> None:
        """No-op: gloo already wrote the received payload in place."""
        return None

    def run_compute(self, kind: str, units: float, fn: Callable[[], Any]) -> Any:
        t_start = self._now_us()
        out = fn()
        self.stats.add_compute(kind, self._now_us() - t_start)
        return out


class VirtualClockBackend:
    """Deterministic latency/bandwidth model of the same schedule."""

    def __init__(
        self,
        supplier: Callable[[str], Any],
        latency_us: float = 5.0,
        bandwidth_bytes_per_us: float = 32.0,
        compute_us: dict[str, float] | None = None,
        compute_unit_scale: float = 1.0,
        jitter_us: dict[str, float] | None = None,
    ) -> None:
        self.supplier = supplier
        self.jitter_us = dict(jitter_us or {})
        self.latency_us = float(latency_us)
        self.bandwidth_bytes_per_us = float(bandwidth_bytes_per_us)
        self.compute_cost = dict(
            compute_us or {"collide": 2.0, "stream": 2.0, "pack": 4.0}
        )
        self.compute_unit_scale = float(compute_unit_scale)
        self.stats = BackendStats()
        self._now = 0.0
        self._recv_arrival: dict[str, float] = {}
        self._send_done: dict[str, float] = {}
        self._buffers: dict[str, Any] = {}

    # -- clock ---------------------------------------------------------
    def reset(self) -> None:
        self.stats = BackendStats()
        self._now = 0.0
        self._recv_arrival.clear()
        self._send_done.clear()

    def tick(self) -> None:
        self.stats.clock_us = self._now

    def close(self) -> None:
        self.tick()

    def _payload_us(self, payload: Any) -> float:
        numel = int(getattr(payload, "numel", lambda: 1)())
        itemsize = int(getattr(payload, "element_size", lambda: 8)())
        return numel * itemsize / max(self.bandwidth_bytes_per_us, 1e-12)

    # -- schedule primitives -------------------------------------------
    def post_recv(self, name: str, buffer: Any) -> Any:
        self.stats.posts_recv += 1
        self._buffers[name] = buffer
        self._recv_arrival[name] = (
            self._now + self.latency_us + self._payload_us(buffer) + self.jitter_us.get(name, 0.0)
        )
        self.stats.events.append((self._now, "post_recv", name))
        return name

    def post_send(self, name: str, payload: Any) -> Any:
        self.stats.posts_send += 1
        self._send_done[name] = self._now + self.latency_us + self._payload_us(payload)
        self.stats.events.append((self._now, "post_send", name))
        return name

    def waitany(self, handles: Sequence[Any]) -> int:
        if not handles:
            raise ValueError("waitany requires at least one handle")
        self.stats.waitany_calls += 1
        arrivals = [self._recv_arrival[name] for name in handles]
        index = int(min(range(len(arrivals)), key=lambda i: arrivals[i]))
        target = arrivals[index]
        if target > self._now:
            self.stats.stall_us += target - self._now
            self._now = target
        self.stats.events.append((self._now, "waitany_done", handles[index]))
        return index

    def waitall(self, handles: Sequence[Any]) -> None:
        self.stats.waitall_calls += 1
        if not handles:
            return
        first = handles[0]
        done = (
            [self._recv_arrival[name] for name in handles]
            if first in self._recv_arrival
            else [self._send_done[name] for name in handles]
        )
        target = max(done)
        if target > self._now:
            self.stats.stall_us += target - self._now
            self._now = target

    def deliver(self, name: str) -> None:
        """Deliver the (ground-truth) payload of a completed receive."""
        if name not in self._buffers:
            raise KeyError(f"no receive buffer registered for {name!r}")
        self.supplier(name, self._buffers[name])

    def run_compute(self, kind: str, units: float, fn: Callable[[], Any]) -> Any:
        out = fn()
        cost = self.compute_cost.get(kind, 0.0) * float(units) * self.compute_unit_scale
        self.stats.add_compute(kind, cost)
        self._now += cost
        self.stats.events.append((self._now, "compute", kind))
        return out