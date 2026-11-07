"""2-D double-distribution-function (DDF) thermal lattice-Boltzmann core.

Scheme
------
* **Momentum** — D2Q9 BGK collision, Guo (2002) forcing, Boussinesq buoyancy
  ``F_y = +rho * g * beta * (T - T_ref)`` (upward for hot fluid; the gravity
  magnitude ``g`` points along ``-y`` and the hydrostatic part is absorbed in
  the pressure).
* **Energy** — D2Q5 BGK advection-diffusion (passive scalar),
  ``alpha = cs^2 (tau_T - 1/2) = (tau_T - 1/2)/3``, equilibrium
  ``g_i^eq = w_i T (1 + 3 c_i.u)``.
* **Momentum walls** — half-way bounce-back: solid nodes are mirrored before
  streaming, placing the no-slip plane half-way between the solid node and the
  adjacent fluid node.
* **Thermal walls** — half-way closure applied at the *first fluid node* of
  every wall link (the solid node is never used as a data carrier, which avoids
  the spurious rest-population flux of full-way bounce-back):

  .. code-block:: text

      adiabatic :  g_opp(i)(x_f) = + g_i^+(x_f)
      isothermal:  g_opp(i)(x_f) = - g_i^+(x_f) + 2 w_opp(i) T_w

  with ``i`` pointing into the wall, ``g^+`` the post-collision value and the
  wall located half a cell from ``x_f`` — on the same plane as the no-slip
  wall, so momentum and thermal geometry agree.  For a steady linear profile
  this closure is exact, which is verified by the pure-conduction test
  (``Nu = 1`` to round-off).
* **Open boundaries** — equilibrium inflow with non-equilibrium extrapolation
  from the first interior column, zero-gradient outflow.

Streaming is one precomputed-index gather and the macroscopic moments are one
matrix product, keeping Python overhead of the 2-D kernel low enough for
10^4-10^5-step validation runs.  float64-capable so that discrete conservation
can be audited down to round-off.

References
----------
Guo, Z., Zheng, C., & Shi, B. (2002). Discrete lattice effects on the forcing
    term in the lattice Boltzmann method. *Phys. Rev. E* 65, 046308.
He, X., Chen, S., & Doolen, G. D. (1998). A novel thermal model for the LBM in
    the incompressible limit. *J. Comput. Phys.* 146, 282-300.
Yoshida, H., & Nagaoka, M. (2010). Lattice Boltzmann model for the convection
    and anisotropic diffusion equation. *J. Comput. Phys.* 229, 7774-7795.
"""

from __future__ import annotations

import torch

__all__ = [
    "C9",
    "W9",
    "OPP9",
    "C5",
    "W5",
    "OPP5",
    "DDF2D",
    "Conduction2D",
    "lagrange_derivative_weights",
    "bilinear_sample",
    "shifted",
]

_DT = torch.float64

# ---------------------------------------------------------------------------
# Lattices (index order identical to tensorlbm.d2q9)
# ---------------------------------------------------------------------------
C9 = torch.tensor(
    [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]],
    dtype=torch.int64,
)
W9 = torch.tensor(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=_DT,
)
OPP9 = torch.tensor([0, 2, 1, 4, 3, 6, 5, 8, 7], dtype=torch.int64)

C5 = torch.tensor([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.int64)
W5 = torch.tensor([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6], dtype=_DT)
OPP5 = torch.tensor([0, 2, 1, 4, 3], dtype=torch.int64)

CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def lagrange_derivative_weights(x_nodes, x_eval) -> list[float]:
    """Derivative weights of the Lagrange polynomial through ``x_nodes``.

    Returns ``d`` with ``p'(x_eval) = sum_j d[j] * y_j``.  For three equally
    spaced nodes ``(1, 2, 3)`` evaluated at ``0.5`` this gives ``(-2, 3, -1)``;
    for nodes ``(0.5, 1.5, 2.5)`` evaluated at ``0.5`` it gives
    ``(-1.5, 2.0, -0.5)``.
    """
    n = len(x_nodes)
    out: list[float] = []
    for j in range(n):
        acc = 0.0
        for m in range(n):
            if m == j:
                continue
            term = 1.0
            for k in range(n):
                if k == j or k == m:
                    continue
                term *= (x_eval - x_nodes[k]) / (x_nodes[j] - x_nodes[k])
            acc += term / (x_nodes[j] - x_nodes[m])
        out.append(acc)
    return out


def _expand(vec: torch.Tensor, ndim: int) -> torch.Tensor:
    return vec.reshape((-1,) + (1,) * (ndim - 1))


def shifted(a: torch.Tensor, cy: int, cx: int, fill=0):
    """``out[y, x] = a[y + cy, x + cx]`` with ``fill`` outside the array."""
    ny, nx = a.shape
    out = torch.full_like(a, fill)
    ys_src = slice(max(0, cy), ny + min(0, cy))
    ys_dst = slice(max(0, -cy), ny + min(0, -cy))
    xs_src = slice(max(0, cx), nx + min(0, cx))
    xs_dst = slice(max(0, -cx), nx + min(0, -cx))
    out[ys_dst, xs_dst] = a[ys_src, xs_src]
    return out


def bilinear_sample(field: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor):
    """Bilinear interpolation of ``field[ny, nx]`` at fractional index points.

    ``xs``/``ys`` are tensors of equal shape in *index* coordinates
    (``x`` = column, ``y`` = row); out-of-range samples clamp to the edge.
    """
    ny, nx = field.shape
    x0 = torch.floor(xs).long().clamp(0, nx - 1)
    y0 = torch.floor(ys).long().clamp(0, ny - 1)
    x1 = (x0 + 1).clamp(0, nx - 1)
    y1 = (y0 + 1).clamp(0, ny - 1)
    fx = (xs - x0.to(xs.dtype)).clamp(0.0, 1.0)
    fy = (ys - y0.to(ys.dtype)).clamp(0.0, 1.0)
    return (
        field[y0, x0] * (1 - fx) * (1 - fy)
        + field[y0, x1] * fx * (1 - fy)
        + field[y1, x0] * (1 - fx) * fy
        + field[y1, x1] * fx * fy
    )


def _stream_indices(q: int, c: torch.Tensor, ny: int, nx: int, device) -> torch.Tensor:
    """Precomputed flat gather indices implementing toroidal streaming."""
    yy = torch.arange(ny).reshape(ny, 1)
    xx = torch.arange(nx).reshape(1, nx)
    rows = []
    for i in range(q):
        cy, cx = int(c[i, 1]), int(c[i, 0])
        rows.append((((yy - cy) % ny) * nx + ((xx - cx) % nx)).reshape(-1))
    return torch.stack(rows).to(device)


class _ThermalLinks:
    """Half-way thermal closures on wall-adjacent fluid nodes.

    Each link is ``(i, nodes, tw, dirichlet)``: ``i`` is the lattice direction
    pointing into the wall, ``nodes`` the flat indices of the first fluid nodes,
    ``tw`` a tensor of wall temperatures aligned with ``nodes`` (``None`` for
    adiabatic links) and ``dirichlet`` selects the closure sign.
    """

    def __init__(self) -> None:
        self.dirs: list[int] = []
        self.nodes: list[torch.Tensor] = []
        self.tw: list[torch.Tensor | None] = []
        self.dirichlet: list[bool] = []

    def add(self, i: int, nodes: torch.Tensor, tw=None) -> None:
        if nodes.numel() == 0:
            return
        self.dirs.append(int(i))
        self.nodes.append(nodes)
        self.tw.append(tw)
        self.dirichlet.append(tw is not None)

    def __len__(self) -> int:
        return len(self.dirs)


class DDF2D:
    """Coupled D2Q9 (flow) + D2Q5 (temperature) lattice on a rectangular grid."""

    ADIABATIC = 0
    DIRICHLET = 1

    def __init__(
        self,
        ny: int,
        nx: int,
        nu: float = 0.02,
        alpha: float = 0.02,
        gravity: float = 0.0,
        beta: float = 1.0,
        T_ref: float = 0.0,
        dtype: torch.dtype = _DT,
        device: str | torch.device = "cpu",
    ) -> None:
        self.ny, self.nx = int(ny), int(nx)
        self.n = self.ny * self.nx
        self.nu = float(nu)
        self.alpha = float(alpha)
        self.tau = 0.5 + 3.0 * self.nu
        self.tau_T = 0.5 + 3.0 * self.alpha
        if not (0.5 < self.tau < 2.5):
            raise ValueError(f"D2Q9 relaxation out of stable range: tau={self.tau}")
        if not (0.5 < self.tau_T < 3.0):
            raise ValueError(f"D2Q5 relaxation out of stable range: tau_T={self.tau_T}")
        self.gravity = float(gravity)
        self.beta = float(beta)
        self.T_ref = float(T_ref)
        #: uniform body-force density along +x (drives Poiseuille flow)
        self.body_fx = 0.0
        self.dtype = dtype
        self.device = torch.device(device)

        dev, dt = self.device, dtype
        self.f = torch.zeros(9, self.ny, self.nx, dtype=dt, device=dev)
        self.g = torch.zeros(5, self.ny, self.nx, dtype=dt, device=dev)
        self.solid = torch.zeros(self.ny, self.nx, dtype=torch.bool, device=dev)
        self.tbc = torch.full(
            (self.ny, self.nx), self.ADIABATIC, dtype=torch.uint8, device=dev
        )
        self.T_wall = torch.zeros(self.ny, self.nx, dtype=dt, device=dev)
        self.inflow: dict | None = None
        self.outflow = False

        self.rho = torch.ones(self.ny, self.nx, dtype=dt, device=dev)
        self.ux = torch.zeros(self.ny, self.nx, dtype=dt, device=dev)
        self.uy = torch.zeros(self.ny, self.nx, dtype=dt, device=dev)
        self.T = torch.zeros(self.ny, self.nx, dtype=dt, device=dev)

        self._cx = _expand(C9[:, 0].to(dt), 3)
        self._cy = _expand(C9[:, 1].to(dt), 3)
        self._w9 = _expand(W9.to(dt), 3)
        self._c5x = _expand(C5[:, 0].to(dt), 3)
        self._c5y = _expand(C5[:, 1].to(dt), 3)
        self._w5 = _expand(W5.to(dt), 3)
        self._opp9 = OPP9.to(dev)
        self._opp5 = OPP5.to(dev)
        self._w51 = W5.to(dt).reshape(-1, 1)
        self._w91 = W9.to(dt).reshape(-1, 1)
        self._cx1 = C9[:, 0].to(dt).reshape(-1, 1)
        self._cy1 = C9[:, 1].to(dt).reshape(-1, 1)
        self._c5x1 = C5[:, 0].to(dt).reshape(-1, 1)
        self._c5y1 = C5[:, 1].to(dt).reshape(-1, 1)
        self._mom = torch.stack(
            [torch.ones(9, dtype=dt), C9[:, 0].to(dt), C9[:, 1].to(dt)], dim=1
        ).to(dev)
        self._sidx9 = _stream_indices(9, C9, self.ny, self.nx, dev)
        self._sidx5 = _stream_indices(5, C5, self.ny, self.nx, dev)
        self._fluid = ~self.solid
        self._links = _ThermalLinks()
        self.steps = 0

    # -- geometry / boundary setup ---------------------------------------
    def set_box_walls(self, thickness: int = 1, walls: str = "all") -> None:
        """Mark the outermost ``thickness`` rows/columns as solid (no-slip).

        ``walls`` selects ``"all"``, ``"y"`` (top/bottom rows only) or ``"x"``.
        """
        t = int(thickness)
        if walls in ("all", "y"):
            self.solid[:t, :] = True
            self.solid[-t:, :] = True
        if walls in ("all", "x"):
            self.solid[:, :t] = True
            self.solid[:, -t:] = True
        self._fluid = ~self.solid

    def set_solid(self, mask) -> None:
        self.solid |= mask.to(self.solid.device)
        self._fluid = ~self.solid

    def set_temperature_bc(self, mask, kind: int, value=0.0) -> None:
        if isinstance(value, torch.Tensor):
            self.T_wall[mask] = value[mask].to(self.T_wall.dtype)
        else:
            self.T_wall[mask] = float(value)
        self.tbc[mask] = kind

    def build_thermal_links(self) -> None:
        """Collect the half-way thermal closures on all wall-adjacent fluid nodes."""
        links = _ThermalLinks()
        fluid = ~self.solid
        flat2d = torch.arange(self.n, device=self.solid.device).reshape(self.ny, self.nx)
        wall_dir = self.solid & (self.tbc == self.DIRICHLET)
        for i in range(1, 5):
            cy, cx = int(C5[i, 1]), int(C5[i, 0])
            nb_solid = shifted(self.solid, cy, cx, fill=False)
            mask = fluid & nb_solid
            if not bool(mask.any()):
                continue
            nb_dir = shifted(wall_dir, cy, cx, fill=False)
            m_adi = mask & (~nb_dir)
            if bool(m_adi.any()):
                links.add(i, flat2d[m_adi])
            m_dir = mask & nb_dir
            if bool(m_dir.any()):
                nb_idx = shifted(flat2d, cy, cx, fill=0)[m_dir]
                links.add(i, flat2d[m_dir], nb_idx)
        self._links = links

    # -- initialisation ---------------------------------------------------
    def init_uniform(self, ux: float = 0.0, uy: float = 0.0, T: float = 0.0) -> None:
        rho = torch.ones(self.ny, self.nx, dtype=self.dtype, device=self.device)
        ux_t = torch.full_like(rho, float(ux))
        uy_t = torch.full_like(rho, float(uy))
        self.f = self.equilibrium_flow(rho, ux_t, uy_t)
        self.g = self.equilibrium_thermal(torch.full_like(rho, float(T)), ux_t, uy_t)

    def init_fields(self, T=None, ux=0.0, uy=0.0) -> None:
        rho = torch.ones(self.ny, self.nx, dtype=self.dtype, device=self.device)
        ux_t = ux.to(rho.dtype) if torch.is_tensor(ux) else torch.full_like(rho, float(ux))
        uy_t = uy.to(rho.dtype) if torch.is_tensor(uy) else torch.full_like(rho, float(uy))
        self.f = self.equilibrium_flow(rho, ux_t, uy_t)
        T_t = torch.zeros_like(rho) if T is None else T.to(rho.dtype)
        self.g = self.equilibrium_thermal(T_t, ux_t, uy_t)

    # -- equilibria ------------------------------------------------------
    def equilibrium_flow(self, rho, ux, uy):
        cu = self._cx * ux + self._cy * uy
        usq = ux * ux + uy * uy
        return self._w9 * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)

    def equilibrium_thermal(self, T, ux, uy):
        return self._w5 * T * (1.0 + 3.0 * (self._c5x * ux + self._c5y * uy))

    def _eq_flow_1d(self, rho, ux, uy):
        cu = self._cx1 * ux + self._cy1 * uy
        usq = ux * ux + uy * uy
        return self._w91 * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)

    def _eq_thermal_1d(self, T, ux, uy):
        cu = self._c5x1 * ux + self._c5y1 * uy
        return self._w51 * T * (1.0 + 3.0 * cu)

    # -- diagnostics -----------------------------------------------------
    def macroscopic(self):
        m = self.f.reshape(9, self.n).t() @ self._mom  # (n, 3) = (rho, rho ux, rho uy)
        rho = m[:, 0]
        safe = rho.clamp_min(1e-12)
        shape = (self.ny, self.nx)
        self.rho = rho.reshape(shape)
        self.ux = m[:, 1].reshape(shape) / safe.reshape(shape)
        self.uy = m[:, 2].reshape(shape) / safe.reshape(shape)
        self.T = self.g.reshape(5, self.n).sum(0).reshape(shape)
        return self.rho, self.ux, self.uy, self.T

    def total_energy(self) -> float:
        """Discrete thermal energy ``sum_i sum_nodes g_i == sum_nodes T``."""
        return float(self.g.sum())

    def wall_flux_normal(self, axis: str, index: int, k: float | None = None):
        """Wall heat flux ``q = -k dT/dn`` per wall node, positive into fluid.

        Three-point one-sided quadratic fit through the first three fluid nodes
        evaluated at the half-cell wall location, i.e. weight vector
        ``(-2, 3, -1)`` (see :func:`lagrange_derivative_weights`).
        """
        k = self.alpha if k is None else float(k)
        T = self.T
        if axis == "x":
            if index == 0:
                d = -2.0 * T[:, 1] + 3.0 * T[:, 2] - T[:, 3]
            else:
                d = -(-2.0 * T[:, -2] + 3.0 * T[:, -3] - T[:, -4])
        elif axis == "y":
            if index == 0:
                d = -2.0 * T[1, :] + 3.0 * T[2, :] - T[3, :]
            else:
                d = -(-2.0 * T[-2, :] + 3.0 * T[-3, :] - T[-4, :])
        else:  # pragma: no cover
            raise ValueError(axis)
        return -k * d

    # -- time stepping ---------------------------------------------------
    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            self._step_once()

    def _step_once(self) -> None:
        m = self.f.reshape(9, self.n).t() @ self._mom
        rho = m[:, 0]
        safe = rho.clamp_min(1e-12)
        shape = (self.ny, self.nx)
        rhov = rho.reshape(shape)
        uxv = m[:, 1].reshape(shape) / safe.reshape(shape)
        uyv = m[:, 2].reshape(shape) / safe.reshape(shape)
        Tv = self.g.reshape(5, self.n).sum(0).reshape(shape)
        self.rho, self.ux, self.uy, self.T = rhov, uxv, uyv, Tv

        fxv = fyv = None
        if self.gravity != 0.0:
            fyv = (self.gravity * self.beta * rhov * (Tv - self.T_ref)) * self._fluid
        if self.body_fx != 0.0:
            fxv = torch.full_like(rhov, self.body_fx) * self._fluid
        if fyv is not None or fxv is not None:
            inv_rho = 1.0 / rhov.clamp_min(1e-12)
            ux_s = uxv if fxv is None else uxv + 0.5 * fxv * inv_rho
            uy_s = uyv if fyv is None else uyv + 0.5 * fyv * inv_rho
            feq = self.equilibrium_flow(rhov, ux_s, uy_s)
            cu = self._cx * ux_s + self._cy * uy_s
            fx_t = fxv if fxv is not None else 0.0
            fy_t = fyv if fyv is not None else 0.0
            force = (1.0 - 0.5 / self.tau) * self._w9 * (
                3.0 * (self._cx - ux_s) * fx_t
                + 3.0 * (self._cy - uy_s) * fy_t
                + 9.0 * cu * (self._cx * fx_t + self._cy * fy_t)
            )
            f_post = self.f - (self.f - feq) / self.tau + force
        else:
            ux_s, uy_s = uxv, uyv
            feq = self.equilibrium_flow(rhov, uxv, uyv)
            f_post = self.f - (self.f - feq) / self.tau

        geq = self.equilibrium_thermal(Tv, uxv, uy_s)
        g_post = self.g - (self.g - geq) / self.tau_T

        # momentum walls: full-way mirror of the solid nodes
        solid = self.solid
        if bool(solid.any()):
            f_post[:, solid] = f_post[self._opp9][:, solid]

        f_new = torch.gather(f_post.reshape(9, self.n), 1, self._sidx9)
        g_new = torch.gather(g_post.reshape(5, self.n), 1, self._sidx5)
        gp = g_post.reshape(5, self.n)

        # thermal walls: half-way closure on the fluid side
        tw_flat = self.T_wall.reshape(-1)
        for i, nodes, nb, is_dir in zip(
            self._links.dirs, self._links.nodes, self._links.tw, self._links.dirichlet
        ):
            j = int(OPP5[i])
            if is_dir:
                g_new[j, nodes] = -gp[i, nodes] + 2.0 * float(W5[j]) * tw_flat[nb]
            else:
                g_new[j, nodes] = gp[i, nodes]

        self.f = f_new.reshape(9, self.ny, self.nx)
        self.g = g_new.reshape(5, self.ny, self.nx)
        self.steps += 1

        if self.inflow is not None:
            self._apply_inflow()
        if self.outflow:
            self.f[:, :, -1] = self.f[:, :, -2]
            self.g[:, :, -1] = self.g[:, :, -2]

    def _column_moments(self, col: int):
        f = self.f[:, :, col]
        rho = f.sum(0)
        ux = (f * self._cx1).sum(0) / rho.clamp_min(1e-12)
        uy = (f * self._cy1).sum(0) / rho.clamp_min(1e-12)
        T = self.g[:, :, col].sum(0)
        return rho, ux, uy, T

    def _apply_inflow(self) -> None:
        spec = self.inflow
        rho1, ux1, uy1, T1 = self._column_moments(1)
        neq_f = self.f[:, :, 1] - self._eq_flow_1d(rho1, ux1, uy1)
        neq_g = self.g[:, :, 1] - self._eq_thermal_1d(T1, ux1, uy1)
        ny = self.ny
        rho_in = spec.get("rho")
        rho_t = rho1 if rho_in is None else torch.full_like(rho1, float(rho_in))
        T_in = spec.get("T")
        if T_in is None:
            T_t = T1
        elif torch.is_tensor(T_in):
            T_t = T_in.to(self.dtype)
        else:
            T_t = torch.full((ny,), float(T_in), dtype=self.dtype, device=self.device)
        ux_in = spec["ux"]
        if torch.is_tensor(ux_in):
            ux_t = ux_in.to(self.dtype).reshape(-1)
        else:
            ux_t = torch.full((ny,), float(ux_in), dtype=self.dtype, device=self.device)
        uy_in = spec.get("uy", 0.0)
        if torch.is_tensor(uy_in):
            uy_t = uy_in.to(self.dtype).reshape(-1)
        else:
            uy_t = torch.full((ny,), float(uy_in), dtype=self.dtype, device=self.device)
        self.f[:, :, 0] = self._eq_flow_1d(rho_t, ux_t, uy_t) + neq_f
        self.g[:, :, 0] = self._eq_thermal_1d(T_t, ux_t, uy_t) + neq_g


class Conduction2D:
    """D2Q5 pure-diffusion lattice for solid sub-domains (periodic along ``x``).

    Rows ``0 .. ny-1`` are conduction nodes.  Isothermal surfaces are imposed
    with the same half-way closure as :class:`DDF2D`, i.e. on the node itself
    for an interface link (temperature supplied by the caller each step) or on
    the first interior node for a ghost-surface link.  ``add_dirichlet_link``
    returns a live ``(nx,)`` vector that the caller may update in place.
    """

    def __init__(self, ny: int, nx: int, alpha: float = 0.02,
                 dtype: torch.dtype = _DT, device="cpu") -> None:
        self.ny, self.nx = int(ny), int(nx)
        self.n = self.ny * self.nx
        self.alpha = float(alpha)
        self.tau_T = 0.5 + 3.0 * self.alpha
        if not (0.5 < self.tau_T < 3.0):
            raise ValueError(f"D2Q5 relaxation out of stable range: tau_T={self.tau_T}")
        self.dtype = dtype
        self.device = torch.device(device)
        dt, dev = dtype, self.device
        self.g = torch.zeros(5, self.ny, self.nx, dtype=dt, device=dev)
        self._w5 = _expand(W5.to(dt), 3)
        self._sidx5 = _stream_indices(5, C5, self.ny, self.nx, dev)
        self.steps = 0
        self.T = torch.zeros(self.ny, self.nx, dtype=dt, device=dev)
        self._links: list[dict] = []

    def init_uniform(self, T: float = 0.0) -> None:
        T_t = torch.full((self.ny, self.nx), float(T), dtype=self.dtype, device=self.device)
        self.g = self._w5 * T_t

    def add_link(self, i: int, row: int | None = None, T: float | None = 0.0,
                 nodes=None):
        """Register a half-way closure in lattice direction ``i``.

        Acts on ``nodes`` (flat indices) or, if omitted, on ``row``.  ``T=None``
        registers an adiabatic (zero-flux) reflection; otherwise a Dirichlet
        link is created and its live temperature vector is returned so that the
        caller can update it in place between steps.
        """
        if nodes is None:
            flat = torch.arange(self.n, device=self.device).reshape(self.ny, self.nx)
            nodes = flat[row, :]
        tw = None
        if T is not None:
            tw = torch.full(nodes.shape, float(T), dtype=self.dtype, device=self.device)
        self._links.append({"i": int(i), "nodes": nodes, "tw": tw})
        return tw

    def add_dirichlet_link(self, i: int, row: int, T: float = 0.0):
        """Backward-compatible alias for a Dirichlet link on ``row``."""
        return self.add_link(i, row=row, T=T)

    def macroscopic(self):
        self.T = self.g.sum(0)
        return self.T

    def total_energy(self) -> float:
        return float(self.g.sum())

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            self._step_once()

    def _step_once(self) -> None:
        T = self.g.sum(0)
        geq = self._w5 * T
        g_post = self.g - (self.g - geq) / self.tau_T
        gp = g_post.reshape(5, self.n)
        gn = torch.gather(gp, 1, self._sidx5)
        for lnk in self._links:
            i = lnk["i"]
            j = int(OPP5[i])
            if lnk["tw"] is None:
                gn[j, lnk["nodes"]] = gp[i, lnk["nodes"]]
            else:
                gn[j, lnk["nodes"]] = (
                    -gp[i, lnk["nodes"]] + 2.0 * float(W5[j]) * lnk["tw"]
                )
        self.g = gn.reshape(5, self.ny, self.nx)
        self.steps += 1