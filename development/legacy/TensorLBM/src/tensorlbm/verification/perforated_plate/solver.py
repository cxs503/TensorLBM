"""D2Q9 LBM solver for oscillatory flow through a sharp-edged perforated plate.

Pure-NumPy implementation (float64, deterministic).  The previous torch based
stub is replaced by a vectorised NumPy kernel so that whole KC/beta/resolution
matrices can be run on the cluster.

Numerical ingredients
---------------------
* D2Q9, BGK collision, ``tau`` from ``nu = (tau - 1/2)/3``.
* Guo (1993) forcing for the sinusoidal body force ``F_x(t) = F0 sin(omega t)``
  including the ``u = (sum c f + F/2)/rho`` velocity correction.
* Half-way bounce-back on every solid link (channel walls *and* plate), so that
  the effective (hydrodynamic) plate thickness equals the geometric one.
* Momentum-exchange (Ladd/Schaefer) force measurement: for every link that ends
  inside a solid cell the momentum ``2 c_i f_i`` is handed to the solid.  This
  gives the plate force and the wall force without requiring a pressure gradient.
* Pressure probing on planes at fixed streamwise stations (``dp_stations``) with
  the slab (window) momentum budget made explicit, see :meth:`step`.

Geometry
--------
Channel periodic in x, solid rows ``j = 0`` and ``j = ny-1`` (walls), plate of
``plate_thickness`` columns at ``x = plate_col`` with ``n_holes`` rectangular
holes of height ``hole_size`` evenly distributed over the channel height.
``beta`` is the *actual* open-area ratio of the plate, measured from the mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# D2Q9 lattice
# ---------------------------------------------------------------------------

C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
CS2 = 1.0 / 3.0
CS4 = CS2 * CS2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PerforatedPlateConfig:
    """Configuration of the perforated-plate oscillatory-flow case.

    Lengths are in lattice units, time in lattice steps (dt = 1).
    """

    nx: int = 288
    ny: int = 96
    plate_col: int = -1          # auto: nx // 2
    plate_thickness: int = 2
    hole_pattern: str = "array"  # 'array' | 'single'
    n_holes: int = 3
    hole_size: int = 6
    beta: float = -1.0           # > 0 : derive hole layout from this target
    tau: float = 0.53
    force_amplitude: float = 1e-4
    omega: float = 2 * math.pi / 1200.0
    steps: int = 7200
    warmup_cycles: int = 3
    n_cycles_measure: int = 3
    samples_per_cycle: int = 48
    probe_wall_shear: bool = False   # extra full-length lower-wall force probe
    init_profile: str = "womersley"  # 'rest' | 'womersley' (DC-free AC start)
    device: str = "cpu"          # kept for API compatibility (NumPy only)
    dtype: str = "float64"       # kept for API compatibility
    label: str = "perforated_plate"

    # ---- derived quantities ------------------------------------------------
    @property
    def plate_col_auto(self) -> int:
        return self.plate_col if self.plate_col > 0 else self.nx // 2

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    @property
    def period_steps(self) -> int:
        return max(1, int(round(self.period)))

    @property
    def total_cycles(self) -> int:
        return self.warmup_cycles + self.n_cycles_measure

    def validate(self) -> None:
        if self.nx < 32 or self.ny < 16:
            raise ValueError("grid too small: nx>=32 and ny>=16 required")
        if self.tau <= 0.5:
            raise ValueError("tau must be > 0.5 (positive viscosity)")
        if self.force_amplitude < 0:
            raise ValueError("force_amplitude must be >= 0")
        if self.omega <= 0:
            raise ValueError("omega must be > 0")
        if self.plate_thickness < 1:
            raise ValueError("plate_thickness must be >= 1")
        if self.plate_col_auto - 1 < 1 or self.plate_col_auto + self.plate_thickness >= self.nx:
            raise ValueError("plate does not fit inside the domain")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _hole_intervals(ny: int, n_holes: int, hole_size: int) -> list[tuple[int, int]]:
    """Evenly spaced holes over the interior rows ``1..ny-2``.

    Returns list of half-open intervals ``[start, stop)`` in the y index.
    """
    if n_holes < 1:
        raise ValueError("n_holes must be >= 1")
    ny_fluid = ny - 2
    hole_size = int(max(1, min(hole_size, ny_fluid)))
    spacing = ny_fluid / float(n_holes)
    intervals = []
    for h in range(n_holes):
        centre = 1.0 + spacing * (h + 0.5)
        start = int(round(centre - hole_size / 2.0))
        start = max(1, min(start, ny - 1 - hole_size))
        intervals.append((start, start + hole_size))
    return intervals


def _layout_from_beta(ny: int, n_holes: int, beta: float) -> int:
    """Hole height (lattice units) that realises ``beta`` for given hole count."""
    ny_fluid = ny - 2
    total_open = beta * ny_fluid
    hole_size = int(round(total_open / n_holes))
    return int(max(1, min(hole_size, ny_fluid // n_holes)))


def make_perforated_plate_mask(
    nx: int,
    ny: int,
    plate_col: int,
    hole_pattern: str = "array",
    n_holes: int = 1,
    hole_size: int = 10,
    plate_thickness: int = 2,
    beta: float = -1.0,
    device=None,
) -> np.ndarray:
    """Boolean mask of the solid region, shape ``(nx, ny)``, True = solid.

    ``device`` is accepted for backward compatibility and ignored (NumPy only).
    """
    mask = np.zeros((nx, ny), dtype=bool)
    mask[:, 0] = True
    mask[:, ny - 1] = True

    if beta is not None and beta >= 1.0:
        return mask  # no plate at all

    if beta is not None and beta > 0.0:
        hole_size = _layout_from_beta(ny, n_holes, beta)

    if hole_pattern == "single":
        n_holes = 1

    cols = [plate_col + k for k in range(plate_thickness)]
    intervals = _hole_intervals(ny, n_holes, hole_size)
    for col in cols:
        if col < 0 or col >= nx:
            continue
        mask[col, :] = True
        for (a, b) in intervals:
            mask[col, a:b] = False
    return mask


def compute_open_area_ratio(mask: np.ndarray, plate_col: int, ny: int) -> float:
    """Open-area ratio of the plate measured from the mask (2-D, i.e. length ratio)."""
    if plate_col < 0 or plate_col >= mask.shape[0]:
        return 1.0
    ny_fluid = ny - 2
    open_cells = int((~mask[plate_col, 1:ny - 1]).sum())
    if open_cells >= ny_fluid:
        return 1.0
    return max(0.0, min(1.0, open_cells / float(ny_fluid)))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class PerforatedPlateSolver:
    """D2Q9 BGK solver, oscillatory channel flow through a perforated plate."""

    def __init__(self, cfg: PerforatedPlateConfig):
        cfg.validate()
        self.cfg = cfg
        self.nx = int(cfg.nx)
        self.ny = int(cfg.ny)
        self.tau = float(cfg.tau)
        self.nu = cfg.nu
        self.omega = float(cfg.omega)

        self.pc = cfg.plate_col_auto
        self.plate_thickness = int(cfg.plate_thickness)

        # --- masks ---------------------------------------------------------
        self.plate_mask = np.zeros((self.nx, self.ny), dtype=bool)
        if not (cfg.beta is not None and cfg.beta >= 1.0):
            pm = make_perforated_plate_mask(
                self.nx, self.ny, self.pc,
                hole_pattern=cfg.hole_pattern,
                n_holes=cfg.n_holes,
                hole_size=cfg.hole_size,
                plate_thickness=self.plate_thickness,
                beta=cfg.beta,
            )
            # keep only the plate columns (walls are handled separately)
            self.plate_mask[self.pc:self.pc + self.plate_thickness, :] = \
                pm[self.pc:self.pc + self.plate_thickness, :]
        self.wall_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.wall_mask[:, 0] = True
        self.wall_mask[:, self.ny - 1] = True
        self.solid = self.plate_mask | self.wall_mask

        self.beta = compute_open_area_ratio(
            self.solid, self.pc if self.plate_mask.any() else -1, self.ny
        )
        self.ny_fluid = self.ny - 2

        # --- probe stations ------------------------------------------------
        self.i_q = max(2, self.pc - 6)                       # flow-rate probe
        self.i_q_down = min(self.nx - 3, self.pc + self.plate_thickness + 5)
        gap = 5
        self.i_up = max(1, self.pc - gap)
        self.i_down = min(self.nx - 2, self.pc + self.plate_thickness - 1 + gap)
        self.window_width = float(self.i_down - self.i_up)

        # --- fields --------------------------------------------------------
        self.rho0 = 1.0
        self.f = np.zeros((9, self.nx, self.ny), dtype=np.float64)
        self._set_initial_field()

        self.step_count = 0
        self._build_tables()
        self._q_hist = []
        self._cx3 = C[:, 0].reshape(9, 1, 1).astype(np.float64)
        self._cy3 = C[:, 1].reshape(9, 1, 1).astype(np.float64)
        self._w3 = W.reshape(9, 1, 1)

        # window wall mask for wall-force bookkeeping
        self._win = np.zeros(self.nx, dtype=bool)
        self._win[self.i_up:self.i_down + 1] = True
        self._wall_win_mask = self.wall_mask & self._win[:, None]
        self._wall_lower_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self._wall_lower_mask[:, 0] = True

    # -- initial condition ---------------------------------------------------
    def _set_initial_field(self) -> None:
        """Set f to a local equilibrium with the requested initial velocity.

        ``init_profile = 'womersley'`` starts from the *zero-mean* oscillatory
        solution of the unobstructed channel at t = 0.  This matters: starting
        from rest with ``F = F0 sin(omega t)`` leaves a spurious mean flow of
        order ``F0/omega`` which decays only on the slow wall-diffusion time
        ``h^2/(pi^2 nu)`` and, through the quadratic plate drag, suppresses the
        oscillatory part by two orders of magnitude.
        """
        ux0 = np.zeros((self.nx, self.ny), dtype=np.float64)
        if getattr(self.cfg, "init_profile", "rest") == "womersley" and self.cfg.force_amplitude > 0:
            try:
                from .reference import womersley_channel_u
                y = np.arange(self.ny, dtype=np.float64) - 0.5 * (self.ny - 1)
                u_prof = np.asarray(
                    womersley_channel_u(y, 0.0, float(self.ny - 2), self.nu,
                                        float(self.cfg.force_amplitude), self.omega),
                    dtype=np.float64,
                )
                ux0[:] = u_prof[None, :]
            except Exception:                        # pragma: no cover
                ux0[:] = 0.0
        uy0 = np.zeros_like(ux0)
        usq = ux0 * ux0
        for i in range(9):
            cu = C[i, 0] * ux0 + C[i, 1] * uy0
            self.f[i] = W[i] * self.rho0 * (
                1.0 + cu / CS2 + 0.5 * cu * cu / CS4 - 0.5 * usq / CS2)

    # -- forcing ------------------------------------------------------------
    def force_density(self, t: float) -> float:
        """Body-force density F_x(t) = F0 sin(omega t)  [lattice units]."""
        return float(self.cfg.force_amplitude) * math.sin(self.omega * t)

    # -- macroscopic --------------------------------------------------------
    def macroscopic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = self.f.sum(axis=0)
        ux = (self._cx3 * self.f).sum(axis=0) / rho
        uy = (self._cy3 * self.f).sum(axis=0) / rho
        return rho, ux, uy

    # -- collision ----------------------------------------------------------
    def _collide(self, fx_body: float) -> None:
        f = self.f
        rho = f.sum(axis=0)
        ux = (self._cx3 * f).sum(axis=0) / rho
        uy = (self._cy3 * f).sum(axis=0) / rho
        # Guo velocity correction
        ux = ux + 0.5 * fx_body / rho
        uy = uy
        usq = ux * ux + uy * uy
        cu = self._cx3 * ux + self._cy3 * uy
        feq = self._w3 * rho * (1.0 + cu / CS2 + 0.5 * cu * cu / CS4 - 0.5 * usq / CS2)
        self.f = f - (f - feq) / self.tau + self._guo_source(fx_body, ux, uy, rho)

    def _guo_source(self, fx_body: float, ux: np.ndarray, uy: np.ndarray,
                    rho: np.ndarray) -> np.ndarray:
        if fx_body == 0.0:
            return np.zeros_like(self.f)
        pref = (1.0 - 0.5 / self.tau)
        cx = self._cx3
        cu = cx * ux + self._cy3 * uy
        src = self._w3 * pref * ((cx - ux) / CS2 + cu * cx / CS4) * fx_body
        return src

    # -- gather tables (built once) ----------------------------------------
    def _build_tables(self) -> None:
        """Pre-compute streaming gather indices and bounce-back link tables."""
        nx, ny = self.nx, self.ny
        n = nx * ny
        src_idx = np.zeros((9, n), dtype=np.int64)
        src_solid = np.zeros((9, n), dtype=bool)
        flat_solid = self.solid.reshape(n)
        xs = np.arange(nx)[:, None]
        ys = np.arange(ny)[None, :]
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            sx = np.mod(xs - cx, nx)
            sy = np.mod(ys - cy, ny)
            src_idx[i] = (sx * ny + sy).reshape(n)
            src_solid[i] = flat_solid[src_idx[i]]
        self._src_idx = src_idx
        self._src_solid = src_solid
        self._link_cache: dict[int, np.ndarray] = {}

    def _link_masks(self, mask: np.ndarray) -> np.ndarray:
        """Boolean table ``(9, N)``: f_i leaves a fluid cell towards ``mask``."""
        key = id(mask)
        cached = self._link_cache.get(key)
        if cached is not None and cached.shape[1] == mask.size:
            return cached
        n = self.nx * self.ny
        flat = mask.reshape(n)
        out = np.zeros((9, n), dtype=bool)
        xs = np.arange(self.nx)[:, None]
        ys = np.arange(self.ny)[None, :]
        for i in range(1, 9):
            tx = np.mod(xs + int(C[i, 0]), self.nx)
            ty = np.mod(ys + int(C[i, 1]), self.ny)
            tgt = flat[(tx * self.ny + ty).reshape(n)]
            out[i] = tgt & ~self.solid.reshape(n)
        self._link_cache[key] = out
        return out

    # -- streaming with half-way bounce-back -------------------------------
    def _stream(self) -> None:
        f = self.f.reshape(9, -1)
        gathered = np.take_along_axis(f, self._src_idx, axis=1)
        reflected = f[OPPOSITE]
        self.f = np.where(self._src_solid, reflected, gathered).reshape(9, self.nx, self.ny)

    # -- force measurement -------------------------------------------------
    def _force_on(self, mask: np.ndarray) -> np.ndarray:
        """Momentum-exchange force on the cells of ``mask`` (post-collision f)."""
        links = self._link_masks(mask)
        f = self.f.reshape(9, -1)
        mom = 2.0 * np.einsum('ij,ij->i', f, links)   # momentum per direction
        return C.astype(np.float64).T @ mom

    # -- probes -------------------------------------------------------------
    def _probe(self) -> dict:
        rho, ux, uy = self.macroscopic()
        q = float(ux[self.i_q, 1:self.ny - 1].sum())
        q_down = float(ux[self.i_q_down, 1:self.ny - 1].sum())
        rho_up = float(rho[self.i_up, 1:self.ny - 1].mean())
        rho_down = float(rho[self.i_down, 1:self.ny - 1].mean())
        dp_stations = CS2 * (rho_up - rho_down)
        flux_up = float((rho[self.i_up, 1:self.ny - 1]
                         * ux[self.i_up, 1:self.ny - 1] ** 2).sum())
        flux_down = float((rho[self.i_down, 1:self.ny - 1]
                           * ux[self.i_down, 1:self.ny - 1] ** 2).sum())
        return {
            "flux_up": flux_up,
            "flux_down": flux_down,
            "flow_rate": q,
            "flow_rate_down": q_down,
            "rho_up": rho_up,
            "rho_down": rho_down,
            "dp_stations": dp_stations,
            "u_max_probe": float(np.abs(ux[self.i_q, 1:self.ny - 1]).max()),
        }

    # -- one time step ------------------------------------------------------
    def step(self, t: float) -> dict:
        """Advance one lattice step and return instantaneous diagnostics."""
        fx_body = self.force_density(t)

        # 1) collide + body force  -> post-collision populations
        self._collide(fx_body)

        # 2) momentum-exchange forces (must use post-collision f)
        f_plate = self._force_on(self.plate_mask) if self.plate_mask.any() else np.zeros(2)
        f_wall_win = self._force_on(self._wall_win_mask)
        f_wall_lower = (self._force_on(self._wall_lower_mask)
                        if self.cfg.probe_wall_shear else None)

        # 3) stream (half-way bounce-back inside _stream)
        self._stream()

        # 4) probes
        probe = self._probe()
        self.step_count += 1

        q = probe["flow_rate"]
        # window slab momentum budget:
        #   dp_stations = rho*W*du_bar/dt - rho*F*W + (F_wall + F_plate)/A
        du_bar_dt = self._du_dt(q)
        area = float(self.ny_fluid)
        # slab momentum budget of the pressure probe window
        #   p_up - p_down = rho W du_bar/dt - rho F W + (F_wall + F_plate)/A
        # and p_up - p_down = F_plate/A for a thin plate (steady or not, the
        # plate force is the only streamwise sink inside the window).
        dp_plate_from_force = f_plate[0] / area
        dp_plate_from_stations = (
            probe["dp_stations"]
            - 1.0 * self.window_width * du_bar_dt
            + 1.0 * fx_body * self.window_width
            - f_wall_win[0] / area
            + (probe["flux_up"] - probe["flux_down"]) / area
        )

        if f_wall_lower is not None:
            probe["force_lower_wall_x"] = float(f_wall_lower[0])
        probe.update({
            "force_plate_x": float(f_plate[0]),
            "force_plate_y": float(f_plate[1]),
            "force_wall_window_x": float(f_wall_win[0]),
            "dp_plate": dp_plate_from_force,
            "dp_plate_stations": dp_plate_from_stations,
            "force_body": fx_body,
        })
        self._q_hist.append(q)
        return probe

    def _du_dt(self, q_now: float) -> float:
        """Backward-difference estimate of d(u_bar)/dt = d(Q/A)/dt."""
        if len(self._q_hist) == 0:
            return 0.0
        return (q_now - self._q_hist[-1]) / (float(self.ny_fluid))

    # -- full run -----------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        self._q_hist = []
        warmup_steps = cfg.warmup_cycles * cfg.period_steps
        # fix the number of steps so that the measurement window is an integer
        # number of periods
        n_steps = warmup_steps + cfg.n_cycles_measure * cfg.period_steps

        keys = ["time", "flow_rate", "flow_rate_down", "dp_plate",
                "dp_plate_stations", "dp_stations", "force_plate_x",
                "force_body", "u_max_probe", "force_lower_wall_x",
                "flux_up", "flux_down"]
        series = {k: [] for k in keys}
        snapshots = []

        # snapshot schedule inside the last oscillation period
        snap_times = set()
        t_start = n_steps - cfg.period_steps + 1
        for k in range(cfg.samples_per_cycle):
            snap_times.add(t_start + int(round(k * cfg.period_steps / cfg.samples_per_cycle)))

        cyc_amp = []
        cyc_len = cfg.period_steps
        for step in range(1, n_steps + 1):
            res = self.step(float(step))
            if step > warmup_steps:
                series["time"].append(float(step))
                for k in keys:
                    if k == "time":
                        continue
                    series[k].append(res[k] if k in res else 0.0)
                if step in snap_times:
                    rho, ux, uy = self.macroscopic()
                    snapshots.append({
                        "t": step,
                        "ux": ux.copy(),
                        "uy": uy.copy(),
                    })
            # per-cycle |Q| amplitude for the stationarity check
            if step % cyc_len == 0 and step > warmup_steps - cyc_len:
                seg = self._q_hist[-cyc_len:]
                cyc_amp.append(max(abs(min(seg)), abs(max(seg))))

        out = {k: np.asarray(v, dtype=np.float64) for k, v in series.items()}
        return {
            "time_series": out,
            "snapshots": snapshots,
            "config": cfg,
            "beta": self.beta,
            "plate_col": self.pc,
            "nu": self.nu,
            "tau": self.tau,
            "ny_fluid": self.ny_fluid,
            "warmup_steps": warmup_steps,
            "cycle_amplitudes": cyc_amp,
            "plate_thickness": self.plate_thickness,
        }


def run_perforated_plate(cfg: PerforatedPlateConfig) -> dict:
    """Convenience wrapper: build the solver, run it, return result + solver."""
    solver = PerforatedPlateSolver(cfg)
    result = solver.run()
    result["solver"] = solver
    return result