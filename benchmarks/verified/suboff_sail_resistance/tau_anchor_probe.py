#!/usr/bin/env python3
"""FA anchor-family probe for the uniform-grid tau ladder (SIR-1b / m298).

Read-only instrument: it never writes ``engine.f``, so enabling it cannot
change the dynamics, and the frozen E2 numbers of ``suboff_sail_resistance``
are untouched when the probe is off (env var unset = zero behaviour change).

Anchor family carried over from the AMR physics chain (#257/#265/#275),
translated to this single-level uniform grid:

===================  ===========================================================
anchor               definition here
===================  ===========================================================
``mem_fx``           Ladd link sum  F_x = Σ_{fluid→solid links i}
                     c_ix·(f_i(x_f) + f_opp_i(x_s)) — the momentum-exchange
                     impulse on the wall, an estimator fully independent of the
                     pressure/friction surface integration behind Cd_total.
``mem_bgsub_fx``     same sum with the uniform free-stream equilibrium
                     background removed (closed form of
                     ``momentum_exchange_background_subtracted``).
``ct``               ``mem_fx / dpS`` (link-caliber drag coefficient).
``E0_x/E1_x/E2_x``   #257/#265 algebra on the same link set: the equilibrium
                     pair splits into E0 = 2·w_i·rho_link (strictly
                     density-linear), E1 = 2·w_i·rho·3(c·u) (thermal) and
                     E2 = 2·w_i·rho·(4.5(c·u)²−1.5u²) (second order).
``NEQ_x``            link sum minus (E0+E1+E2); identity residual recorded.
``e0``               ``E0_x / dpS``.
``rho_fwd/mid/aft``  mean density of fluid cells in the fwd/mid/aft third of
                     the hull span.
``D_QW``             ``rho_fwd − rho_aft`` — the bow/stern dipole, sign
                     convention as in the AMR chain (negative = stern-heavy).
===================  ===========================================================

Every reduction runs on gathered 1-D index vectors over the crossing (fluid→
solid) link set — O(near-wall cells), not O(domain) — so a sample costs
milliseconds and can be taken on a fine cadence.
"""

from __future__ import annotations

import json

import torch

from tensorlbm.d3q19 import OPPOSITE, C, W, macroscopic3d

__all__ = ["AnchorProbe", "build_axial_thirds"]


def build_axial_thirds(solid: torch.Tensor, cx: float, length: float) -> list[torch.Tensor]:
    """Split the *fluid* domain into three slabs along the hull axis.

    Boundaries are thirds of the hull span centred at ``cx``.  Only fluid cells
    participate (solid cells hold stale populations).  Cells upstream /
    downstream of the hull span (domain padding) are folded into the nearest
    slab, so the three masks partition the whole fluid domain.
    """
    nz, ny, nx = solid.shape
    dev = solid.device
    xx = torch.arange(nx, device=dev, dtype=torch.float32).view(1, 1, nx)
    fluid = ~solid
    lo = cx - 0.5 * length
    e0, e1, e2 = lo, lo + length / 3.0, lo + 2.0 * length / 3.0
    m_fwd = (xx < e1) & fluid
    m_mid = ((xx >= e1) & (xx < e2)) & fluid
    m_aft = ((xx >= e2) & (xx <= e0 + length)) & fluid
    # padding cells (x < lo or x > lo+length) go to the nearest slab
    m_aft = m_aft | ((xx > e0 + length) & fluid)
    covered = m_fwd | m_mid | m_aft
    if bool((fluid & ~covered).any()):  # defensive: nothing may be dropped
        m_fwd = m_fwd | (fluid & ~covered & (xx < e1))
        m_aft = m_aft | (fluid & ~covered & ~(xx < e1))
    return [m_fwd, m_mid, m_aft]


class AnchorProbe:
    """Read-only per-cadence anchor sampler; one JSON line per sample."""

    def __init__(
        self,
        solid: torch.Tensor,
        near: torch.Tensor,
        cx: float,
        length: float,
        dpS: float,
        u_lb: float,
        path: str,
        cadence: int = 200,
    ) -> None:
        self.dpS = float(dpS)
        self.u_lb = float(u_lb)
        self.cadence = max(1, int(cadence))
        self.path = path
        self.shape = tuple(int(v) for v in solid.shape)
        self.thirds = build_axial_thirds(solid, cx, length)
        self.n_thirds = [int(m.sum().item()) for m in self.thirds]

        device = solid.device
        c = C.to(device).float()
        nnz, nny, nnx = self.shape
        plane = nny * nnx
        cells = int(nnz * plane)

        # Pre-compute, per lattice direction, the linear indices of the
        # fluid→solid crossing cells and of their solid neighbours.  Nothing
        # here depends on time, so it is built once.
        self.links = []
        for i in range(1, 19):
            ci = c[i]
            di, dj, dk = int(ci[0].item()), int(ci[1].item()), int(ci[2].item())
            solid_shifted = torch.roll(solid, (-dk, -dj, -di), dims=(0, 1, 2))
            crossing = near & solid_shifted
            n_cross = int(crossing.sum().item())
            if n_cross == 0:
                continue
            idx_f = crossing.reshape(-1).nonzero(as_tuple=True)[0]
            # neighbour (solid) cell of each crossing cell
            kz = torch.div(idx_f, plane, rounding_mode="floor")
            rem = idx_f - kz * plane
            ky = torch.div(rem, nnx, rounding_mode="floor")
            kx = rem - ky * nnx
            idx_s = (kz + dk) * plane + (ky + dj) * nnx + (kx + di)
            bad = (idx_s < 0) | (idx_s >= cells)
            if bool(bad.any()):
                keep = ~bad
                idx_f, idx_s = idx_f[keep], idx_s[keep]
                n_cross = int(idx_f.numel())
            self.links.append(
                {
                    "i": i,
                    "opp": int(OPPOSITE[i].item()),
                    "w": float(W[i].item()),
                    "cx": float(ci[0].item()),
                    "d": (di, dj, dk),
                    "idx_f": idx_f,
                    "idx_s": idx_s,
                    "n": n_cross,
                    "plane": plane,
                    "nnx": nnx,
                }
            )
        self.n_links = len(self.links)
        self.n_crossing = int(sum(lk["n"] for lk in self.links))

        self._fh = open(path, "w")
        self.n_samples = 0
        self.max_identity_residual = 0.0

    def _bgsub_inline(self, f: torch.Tensor) -> float:
        """Closed-form background-subtracted impulse (library algebra, gathered).

        ``F = Σ_q c_qx [Σ_cross (f_q + f_opp) − n_cross·2ρ₀w_q(1 + 4.5(c_q·U₀)²
        − 1.5|U₀|²)]`` — identical to
        :func:`tensorlbm.momentum_exchange.momentum_exchange_background_subtracted`
        but restricted to the crossing set by index gather.
        """
        u0 = self.u_lb
        u0sq = u0 * u0
        total = 0.0
        for lk in self.links:
            dsum = (
                f[lk["i"]].reshape(-1)[lk["idx_f"]].sum()
                + f[lk["opp"]].reshape(-1)[lk["idx_s"]].sum()
            )
            cdotu = lk["d"][0] * u0
            bg = 2.0 * 1.0 * lk["w"] * (1.0 + 4.5 * cdotu * cdotu - 1.5 * u0sq)
            total += lk["cx"] * (float(dsum.item()) - lk["n"] * bg)
        return total

    def sample(self, f: torch.Tensor, step: int, tau: float, nu_lb: float) -> dict:
        import time

        t0 = time.time()
        rho, ux, uy, uz = macroscopic3d(f)
        rho_d = rho.double().reshape(-1)
        ux_d = ux.double().reshape(-1)
        uy_d = uy.double().reshape(-1)
        uz_d = uz.double().reshape(-1)

        mem_x = 0.0
        e0_x = e1_x = e2_x = 0.0
        for lk in self.links:
            i_f, i_s = lk["idx_f"], lk["idx_s"]
            w_i, cx_i = lk["w"], lk["cx"]
            di, dj, dk = lk["d"]
            pair = f[lk["i"]].reshape(-1)[i_f].double() + f[lk["opp"]].reshape(-1)[i_s].double()
            mem_x += cx_i * float(pair.sum().item())
            # equilibrium split on the donor (fluid-side) cell
            rl = rho_d[i_f]
            cu = di * ux_d[i_f] + dj * uy_d[i_f] + dk * uz_d[i_f]
            uu = ux_d[i_f] ** 2 + uy_d[i_f] ** 2 + uz_d[i_f] ** 2
            e0 = (2.0 * w_i) * rl
            e1 = (2.0 * w_i) * rl * (3.0 * cu)
            e2 = (2.0 * w_i) * rl * (4.5 * cu * cu - 1.5 * uu)
            e0_x += cx_i * float(e0.sum().item())
            e1_x += cx_i * float(e1.sum().item())
            e2_x += cx_i * float(e2.sum().item())

        neq_x = mem_x - (e0_x + e1_x + e2_x)
        mem_bgsub = self._bgsub_inline(f)
        means = [float(rho[m].mean().item()) for m in self.thirds]

        row = {
            "step": int(step),
            "tau": float(tau),
            "nu_lb": float(nu_lb),
            "re_lb": float(0.0),  # filled by the driver from the setup receipt
            "mem_fx": mem_x,
            "mem_bgsub_fx": mem_bgsub,
            "E0_x": e0_x,
            "E1_x": e1_x,
            "E2_x": e2_x,
            "NEQ_x": neq_x,
            "identity_residual": mem_x - (e0_x + e1_x + e2_x + neq_x),
            "ct": mem_x / self.dpS,
            "e0": e0_x / self.dpS,
            "ct_bgsub": mem_bgsub / self.dpS,
            "rho_fwd": means[0],
            "rho_mid": means[1],
            "rho_aft": means[2],
            "D_QW": means[0] - means[2],
            "sample_ms": round((time.time() - t0) * 1000.0, 2),
        }
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        self.n_samples += 1
        self.max_identity_residual = max(self.max_identity_residual, abs(row["identity_residual"]))
        return row

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass
