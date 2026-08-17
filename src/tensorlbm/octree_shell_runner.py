"""Octree shell drag chain — one-stop runner (octree build / BFL fn / Cd stats).

De-duplicates the 200-400 lines of common scaffolding repeated across the
four octree validate scripts (``examples/octree_{sphere,integrated,
distributed,multigpu}_validate.py``):

  * geometry construction  -> :func:`build_shell_octree` (wraps
    ``octree_boundary.geometry.build_octree_shell`` parameters, incl. the
    validated ``bl = max(2, R/2)`` default and the ``inside_fn`` hook for
    arbitrary geometry like SUBOFF);
  * the per-substep boundary callback -> :func:`make_shell_bfl_fn`
    (wraps ``bfl_apply_gather`` + ``leaf_force_weights``, with the
    ``bfl_ramp_wall_velocity`` moving-wall startup ramp, optional ``q_min``
    clamp and the shard-facade ``force_weights`` fallback);
  * Cd statistics -> :class:`CdStatTracker` (warmup-gated accumulation,
    cumulative mean ``Cd = acc/n_samples/dynamic_area``, rolling-window
    reports, JSON-able finalize).

Normalisation helpers live in the sibling modules built in parallel:

  * ``tensorlbm.lbm_re_tau.tau_from_re(u, L_ref=2R, Re)`` — BGK tau;
  * ``tensorlbm.drag_normalize.compute_wall_link_dx(octree, l1_block)`` /
    ``drag_normalize.dynamic_area(u, radius_leaf)`` — wall-link-resolved
    leaf dx and the ``0.5 u^2 pi r^2`` reference area.

Smoke test (CPU, R6 d1, 5 steps):

  PYTHONPATH=src python -m tensorlbm.octree_shell_runner
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from .drag_normalize import compute_wall_link_dx, dynamic_area
from .lbm_re_tau import tau_from_re
from .octree_boundary.bfl import (
    bfl_apply_gather,
    bfl_ramp_wall_velocity,
    leaf_force_weights,
)
from .octree_boundary.geometry import build_octree_shell
from .octree_boundary.stepping import build_ghost_plan

__all__ = [
    "ShellBflFn",
    "CdStatTracker",
    "build_shell_octree",
    "default_bl_thickness",
    "make_shell_bfl_fn",
    "reynolds_tau",
    "schiller_naumann",
    "smoke_test",
    "sphere_radius_leaf",
]


def default_bl_thickness(radius_l1: float, bl: Optional[float] = None) -> float:
    """Shell band thickness in host cells.

    ``None`` -> the validated default ``max(2, round(R/2))`` (R6 -> 3,
    R8 -> 4, the accuracy-validated choices of the octree validate scripts).
    """
    if bl is not None:
        return float(bl)
    return max(2.0, round(float(radius_l1) / 2.0))


def build_shell_octree(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius_l1: float,
    bl: Optional[float] = None,
    d_max: int = 1,
    transition: int = 1,
    lattice: str = "D3Q19",
    device: Optional[torch.device] = None,
    inside_fn: Optional[Callable] = None,
):
    """One-stop octree shell construction.

    Wraps ``octree_boundary.geometry.build_octree_shell`` so every validate
    script shares the same parameter resolution:

    Args:
        shape: host grid shape ``(nz, ny, nx)`` (L1 physical cells, or the
            coarse grid on the legacy two-level path).
        center: body centre in host physical cell coordinates.
        radius_l1: body radius in host cell units (sphere only; ignored when
            ``inside_fn`` is given).
        bl: shell band thickness in host cells; ``None`` = ``max(2, R/2)``.
        d_max: maximum leaf depth (1 or 2).
        transition: extra cell band appended to the shell mask.
        lattice: "D3Q19" or "D3Q27".
        device: torch device (default CPU).
        inside_fn: ``inside_fn(centers) -> bool tensor`` for arbitrary
            geometry (SUBOFF solid masks etc.).
    """
    bl_cells = default_bl_thickness(radius_l1, bl)
    shape3 = (int(shape[0]), int(shape[1]), int(shape[2]))
    center3 = (float(center[0]), float(center[1]), float(center[2]))
    octree = build_octree_shell(
        shape3,
        center=center3,
        radius=float(radius_l1),
        bl_thickness_cells=bl_cells,
        d_max=int(d_max),
        transition=int(transition),
        lattice=lattice,
        device=device if device is not None else torch.device("cpu"),
        inside_fn=inside_fn,
    )
    return octree


def sphere_radius_leaf(radius_l1: float, dx_leaf: float) -> float:
    """Body radius in wall (leaf) lattice units: ``radius_l1 / dx_leaf``.

    ``dx_leaf`` is the wall-link leaf dx in host cells (use
    ``drag_normalize.compute_wall_link_dx`` for the area-correct value).
    """
    return float(radius_l1) / float(dx_leaf)


def reynolds_tau(u_in: float, reynolds: float, diameter: float) -> float:
    """BGK relaxation time for a target Reynolds number (delegates to the
    sibling ``tensorlbm.lbm_re_tau.tau_from_re``; ``diameter = 2 * radius``)."""
    return float(tau_from_re(u_in, diameter, reynolds))


def schiller_naumann(reynolds: float) -> float:
    """Schiller-Naumann sphere-drag reference: 24/Re (1 + 0.15 Re^0.687)."""
    return 24.0 / reynolds * (1.0 + 0.15 * reynolds ** 0.687)


class ShellBflFn:
    """Callable per-substep BFL boundary fn for the shell steppers.

    Matches the signature the steppers invoke (identical to the validate
    scripts' ``bfl_fn`` / ``bfl_callback``):

    ``bfl_fn(octree_, out, post, ghost_plan, ghost_vals, *, substep)
       -> (f_out, force)``

    where ``force`` is the ``(3,)`` float64 link momentum exchange in leaf
    lattice units (already per-leaf-weighted).

    The moving-wall startup ramp (``bfl_ramp_wall_velocity``) reads
    :attr:`step`; advance it once per root step via :meth:`set_step` (or the
    ``step_holder`` binding, see :func:`make_shell_bfl_fn`).
    """

    def __init__(
        self,
        octree,
        lidx: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        *,
        leaf_weights: Optional[torch.Tensor] = None,
        ramp_steps: int = 0,
        q_min: Optional[float] = None,
        moving_wall: bool = True,
        step_holder: Optional[list] = None,
    ):
        dev = device if device is not None else torch.device("cpu")
        if leaf_weights is None:
            leaf_weights = leaf_force_weights(octree)
        leaf_weights = leaf_weights.to(dev)
        if lidx is not None:
            # distributed path: local leaf shard weights (validated pattern
            # ``leaf_force_weights(octree).to(dev)[lidx]``)
            leaf_weights = leaf_weights[lidx]
        self.leaf_weights = leaf_weights
        self.ramp_steps = int(ramp_steps)
        self.q_min = q_min
        self.moving_wall = bool(moving_wall)
        self.step = 0
        # optional mutable 1-list rebound by the caller each root step
        # (``step_holder[0] = step``) — an alternative to set_step()
        self._step_holder = step_holder

    def set_step(self, step: int) -> None:
        """Bind the current root step (drives the moving-wall ramp)."""
        self.step = int(step)
        if self._step_holder is not None:
            self._step_holder[0] = self.step

    def __call__(self, octree_, out, post, gplan, ghost_vals, *, substep):
        if self._step_holder is not None:
            self.step = int(self._step_holder[0])
        if self.moving_wall and self.ramp_steps > 0:
            rho_w, uwx, uwy, uwz = bfl_ramp_wall_velocity(
                octree_, post, self.step, self.ramp_steps,
            )
            wall_velocity = (uwx, uwy, uwz)
            wall_density = rho_w
        else:
            wall_velocity = None
            wall_density = None
        # Shard facades expose shard-local weights on the facade itself
        # (multigpu pattern); a plain OctreeGrid falls back to ours.
        fw = getattr(octree_, "force_weights", None)
        if fw is None:
            fw = self.leaf_weights
        return bfl_apply_gather(
            octree_, out, post,
            ghost_plan=gplan, ghost_vals=ghost_vals,
            wall_velocity=wall_velocity, wall_density=wall_density,
            force_weights=fw, return_force=True,
            q_min=self.q_min,
        )


def make_shell_bfl_fn(
    octree,
    lidx: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    *,
    ramp_steps: int = 0,
    q_min: Optional[float] = None,
    moving_wall: bool = True,
    step_holder: Optional[list] = None,
) -> ShellBflFn:
    """Factory for the shell BFL boundary fn (see :class:`ShellBflFn`).

    Args:
        octree: the built shell (``OctreeGrid`` or a distributed facade).
        lidx: local leaf indices of this rank/shard; ``None`` = full shell.
        device: device for the force weights (default CPU).
        ramp_steps: moving-wall ramp length (0 = no ramp / fixed wall).
        q_min: BFL q clamp (high-Re safeguard); ``None`` = disabled.
        moving_wall: apply the ``bfl_ramp_wall_velocity`` ramp when
            ``ramp_steps > 0`` (disable at very high Re where the
            ``(3/q)*moving_base`` term diverges as tau -> 0.5).
        step_holder: optional mutable 1-list rebound by the caller each root
            step (``step_holder[0] = step``); the fn also accepts
            :meth:`ShellBflFn.set_step`.
    """
    bfl_fn = ShellBflFn(
        octree, lidx=lidx, device=device,
        ramp_steps=ramp_steps, q_min=q_min, moving_wall=moving_wall,
        step_holder=step_holder,
    )
    return bfl_fn


class CdStatTracker:
    """Warmup-gated Cd accumulation with cumulative/rolling means + reports.

    Conventions extracted from the four octree validate scripts:

      * samples are accepted only when ``step > warmup_steps`` (the startup
        transient — initial impact force — must not pollute Cd);
      * ``Cd = acc[component] / n_samples / dynamic_area`` (distributed
        scripts: ``mem_acc[0] / max(step - warmup, 1) / dynamic_area``);
      * ``ramp_steps`` drives the wall-velocity startup ramp — handled by
        :class:`ShellBflFn`; kept here for reporting/JSON parity only.
    """

    def __init__(
        self,
        dynamic_area: float,
        *,
        warmup_steps: int = 0,
        ramp_steps: int = 0,
        label: str = "",
        report_interval: Optional[int] = None,
        dim: int = 3,
        device: Optional[torch.device] = None,
    ):
        self.dynamic_area = float(dynamic_area)
        self.warmup_steps = int(warmup_steps)
        self.ramp_steps = int(ramp_steps)
        self.label = label
        self.report_interval = report_interval
        self.device = device if device is not None else torch.device("cpu")
        self.acc = torch.zeros(int(dim), dtype=torch.float64, device=self.device)
        self.last_step = 0
        self.n_samples = 0
        # accepted per-step x-force samples (leaf lattice units), for the
        # rolling-window report style of the single-GPU scripts
        self.samples: list[float] = []

    # -- sampling ---------------------------------------------------------
    def begin_step(self, step: int) -> None:
        """Record the current root step (call once per root step)."""
        self.last_step = int(step)

    def update(self, force, step: Optional[int] = None) -> bool:
        """Accumulate one per-root-step force sample.

        ``force`` is the ``(3,)`` float64 MEM force (leaf lattice units) or
        a bare float x-component.  Returns True when the sample was accepted
        (i.e. past the warmup gate).
        """
        if step is not None:
            self.last_step = int(step)
        if self.last_step <= self.warmup_steps:
            return False
        if isinstance(force, (int, float)):
            f = torch.zeros(self.acc.shape[0], dtype=torch.float64,
                            device=self.device)
            f[0] = float(force)
        else:
            f = force.detach().to(torch.float64).reshape(-1)
            if f.shape[0] != self.acc.shape[0]:
                raise ValueError(
                    f"force has {f.shape[0]} components, tracker expects "
                    f"{self.acc.shape[0]}",
                )
        self.acc += f
        self.samples.append(float(f[0].item()))
        self.n_samples += 1
        return True

    # -- statistics -------------------------------------------------------
    def cd(self, component: int = 0) -> float:
        """Cumulative-mean Cd (``acc/n_samples/dynamic_area``)."""
        if self.n_samples <= 0:
            return float("nan")
        return float(self.acc[component].item()) / self.n_samples / self.dynamic_area

    def rolling_cd(self, window: Optional[int] = None, component: int = 0) -> float:
        """Mean Cd over the last ``window`` accepted samples (all if None)."""
        if not self.samples:
            return float("nan")
        tail = self.samples[-window:] if window else self.samples
        return sum(tail) / len(tail) / self.dynamic_area

    def mean_force(self, component: int = 0) -> float:
        """Mean accepted x-force in leaf lattice units."""
        if self.n_samples <= 0:
            return float("nan")
        return float(self.acc[component].item()) / self.n_samples

    # -- reporting --------------------------------------------------------
    def maybe_report(self, step: Optional[int] = None, extra: str = "") -> Optional[str]:
        """Print a periodic ``Cd`` line when ``step % report_interval == 0``."""
        if self.report_interval is None or not self.samples:
            return None
        if step is None:
            step = self.last_step
        if step % self.report_interval != 0:
            return None
        line = (
            f"[{self.label}] step={step} n={self.n_samples} "
            f"Cd={self.cd():.6f} "
            f"Cd_win={self.rolling_cd(self.report_interval):.6f}"
            + (f" {extra}" if extra else "")
        )
        print(line, flush=True)
        return line

    def finalize(self, reference_cd: Optional[float] = None,
                 window: Optional[int] = None) -> dict:
        """JSON-able summary dict (mirrors the validate scripts' run dicts)."""
        cd = self.cd(0)
        out = {
            "label": self.label,
            "cd": cd,
            "cd_window": self.rolling_cd(window) if window else cd,
            "n_samples": self.n_samples,
            "warmup_steps": self.warmup_steps,
            "ramp_steps": self.ramp_steps,
            "mean_force_x_leaf_lu": self.mean_force(0),
            "dynamic_area": self.dynamic_area,
        }
        if reference_cd is not None and math.isfinite(reference_cd) \
                and math.isfinite(cd) and reference_cd != 0.0:
            out["reference_cd"] = float(reference_cd)
            out["reference_error_pct"] = abs(cd - reference_cd) / reference_cd * 100.0
        return out


# ---------------------------------------------------------------------------
# CPU smoke test (R6 d1 octree + bfl fn + tracker, 5 steps)
# ---------------------------------------------------------------------------


def smoke_test(
    shape: tuple[int, int, int] = (32, 24, 24),
    radius: float = 6.0,
    d_max: int = 1,
    u_in: float = 0.06,
    steps: int = 5,
    warmup_steps: int = 2,
    ramp_steps: int = 2,
) -> dict:
    """CPU smoke: build an R6 d1 octree, wire the BFL fn, run 5 steps."""
    from .d3q19 import equilibrium3d

    device = torch.device("cpu")
    nz, ny, nx = shape
    center = (nz * 0.5, ny * 0.5, nx * 0.5)

    octree = build_shell_octree(
        shape, center, radius, bl=None, d_max=d_max, device=device,
    )
    # initialise the leaf populations from the uniform inflow equilibrium
    # (host-cell gather, the validated pattern of the integrated script)
    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, u_in)
    zero = torch.zeros_like(rho)
    eq = equilibrium3d(rho, ux, zero, zero, device=device)
    host = octree.leaf_host_cell
    octree.f_leaf = eq[:, host[:, 0], host[:, 1], host[:, 2]].clone()

    # ghost plan: supplies SHELL_OUTSIDE upstream cells (if any)
    ghost_plan = build_ghost_plan(octree, shape, solid_fallback=True)
    n_slots = int(ghost_plan.slot.max().item()) + 1 \
        if ghost_plan.slot.numel() else 0
    ghost_vals = torch.zeros((octree.Q, n_slots), device=device)

    bfl_fn = make_shell_bfl_fn(octree, None, device, ramp_steps=ramp_steps)
    dx_leaf = compute_wall_link_dx(octree, l1_block=False)
    radius_leaf = sphere_radius_leaf(radius, dx_leaf)
    area = dynamic_area(u_in, radius_leaf)
    tracker = CdStatTracker(
        area, warmup_steps=warmup_steps, ramp_steps=ramp_steps,
        label="smoke", report_interval=1,
    )

    print(
        f"[smoke] n_leaf={octree.n_leaf} Q={octree.Q} "
        f"bfl_links={int(octree.bfl_mask.sum().item())} "
        f"bl={default_bl_thickness(radius)} d_max={d_max} "
        f"dx_leaf={dx_leaf:.4f} radius_leaf={radius_leaf:.4f} "
        f"area={area:.6e}",
        flush=True,
    )
    for step in range(1, steps + 1):
        tracker.begin_step(step)
        bfl_fn.set_step(step)
        f_out, force = bfl_fn(
            octree, octree.f_leaf.clone(), octree.f_leaf,
            ghost_plan, ghost_vals, substep=0,
        )
        octree.f_leaf = f_out  # advance the static smoke field
        accepted = tracker.update(force, step)
        tracker.maybe_report()
        print(
            f"[smoke] step={step} Fx={float(force[0].item()):.6e} "
            f"accepted={accepted} Cd_acc={tracker.cd():.6f}",
            flush=True,
        )

    summary = tracker.finalize()
    assert summary["n_samples"] == max(steps - warmup_steps, 0), summary
    assert math.isfinite(summary["cd"]), summary
    print(
        f"[smoke] final: Cd={summary['cd']:.6f} n_samples={summary['n_samples']} "
        f"mean_Fx={summary['mean_force_x_leaf_lu']:.6e}",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    summary = smoke_test()
    print(f"SMOKE OK: {summary}", flush=True)
