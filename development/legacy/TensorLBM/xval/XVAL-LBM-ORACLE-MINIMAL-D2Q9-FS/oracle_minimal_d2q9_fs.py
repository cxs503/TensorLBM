"""Pure-NumPy reimplementation of the Maarten-vd-Sande/lbm D2Q9 free-surface LBM.

Oracle arm of XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS.

Key fixes vs naive vectorisation:
- Explicit boundary handling (no periodic wrap from np.roll)
- Velocity clamp from original code (stability guard)
- Proper NaN guards in mass redistribution
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------------
# D2Q9 lattice constants (Maarten-vd-Sande ordering)
# ---------------------------------------------------------------------------

V = np.array([
    [ 0,  0], [ 0, -1], [ 0,  1], [-1,  0], [-1, -1],
    [-1,  1], [ 1,  0], [ 1, -1], [ 1,  1],
], dtype=np.int64)

T = np.array([
    4/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/9, 1/36, 1/36,
], dtype=np.float64)

V_INV = np.array([0, 2, 1, 6, 8, 7, 3, 5, 4], dtype=np.int64)

# Cell-type bitflags
CT_FLUID=1; CT_INTERFACE=2; CT_GAS=4; CT_OBSTACLE=8
CT_INLET=16; CT_OUTLET=32
CT_NO_FLUID_NEIGH=64; CT_NO_EMPTY_NEIGH=128; CT_NO_IFACE_NEIGH=256
CT_TO_FLUID=512; CT_TO_GAS=1024
CT_LIQUID = CT_FLUID | CT_INTERFACE


def set_equilibrium(rho, ux, uy):
    """D2Q9 equilibrium (9, nx, ny)."""
    usqr = 1.5 * (ux*ux + uy*uy)
    feq = np.empty((9,) + rho.shape, dtype=np.float64)
    for i in range(9):
        vu = 3.0 * (V[i,0]*ux + V[i,1]*uy)
        feq[i] = rho * T[i] * (1.0 + vu + 0.5*vu*vu - usqr)
    return feq


def collide_bgk(fin, rho, ux, uy, omega, gravity):
    """BGK collision + gravity forcing."""
    feq = set_equilibrium(rho, ux, uy)
    fout = fin + omega * (feq - fin)
    for i in range(9):
        grav = V[i,0]*gravity[0] + V[i,1]*gravity[1]
        fout[i] -= rho * T[i] * grav
    return fout


def update_rho_u(fin):
    """Recover macroscopic fields with velocity clamp (from original code)."""
    rho = fin.sum(axis=0)
    rho_safe = np.where(rho > 1e-12, rho, 1e-12)
    ux = np.einsum('i,ijk->jk', V[:,0].astype(np.float64), fin) / rho_safe
    uy = np.einsum('i,ijk->jk', V[:,1].astype(np.float64), fin) / rho_safe
    # Velocity clamp from original code (stability guard)
    vel = np.sqrt(ux*ux + uy*uy)
    maxvel = np.sqrt(2.0/3.0)
    too_fast = vel > maxvel
    scale = np.where(too_fast, maxvel / np.maximum(vel, 1e-30), 1.0)
    ux *= scale
    uy *= scale
    return rho, ux, uy


def get_epsilon(cell_type, rho, mass):
    """Fill fraction."""
    with np.errstate(divide='ignore', invalid='ignore'):
        eps = np.where(rho > 0, mass / rho, 0.5)
    eps = np.clip(eps, 0.0, 1.0)
    eps = np.where((cell_type & CT_FLUID) | (cell_type & CT_OBSTACLE), 1.0, eps)
    eps = np.where(cell_type & CT_GAS, 0.0, eps)
    return eps


def get_normal(cell_type, rho, mass):
    """Surface normal from fill-fraction gradient."""
    eps = get_epsilon(cell_type, rho, mass)
    nx_s, ny_s = eps.shape
    normal = np.zeros((nx_s, ny_s, 2), dtype=np.float64)
    normal[1:-1, :, 0] = 0.5 * (eps[:-2, :] - eps[2:, :])
    normal[:, 1:-1, 1] = 0.5 * (eps[:, :-2] - eps[:, 2:])
    return normal


def _pull(arr, vx, vy):
    """Pull: arr[ix,iy] <- arr[ix-vx, iy-vy].  Boundary: replicate edge."""
    result = np.empty_like(arr)
    nx, ny = arr.shape
    for ix in range(nx):
        for iy in range(ny):
            sx = ix - vx
            sy = iy - vy
            if 0 <= sx < nx and 0 <= sy < ny:
                result[ix, iy] = arr[sx, sy]
            else:
                result[ix, iy] = arr[max(0,min(nx-1,sx)), max(0,min(ny-1,sy))]
    return result


def _pull_fast(arr, vx, vy):
    """Fast pull using slicing (no per-cell loop)."""
    nx, ny = arr.shape
    result = np.empty_like(arr)
    # Source ranges for each destination cell
    if vx >= 0:
        src_x_start, src_x_end = 0, nx - vx
        dst_x_start, dst_x_end = vx, nx
    else:
        src_x_start, src_x_end = -vx, nx
        dst_x_start, dst_x_end = 0, nx + vx
    if vy >= 0:
        src_y_start, src_y_end = 0, ny - vy
        dst_y_start, dst_y_end = vy, ny
    else:
        src_y_start, src_y_end = -vy, ny
        dst_y_start, dst_y_end = 0, ny + vy

    # Interior: proper pull
    result[dst_x_start:dst_x_end, dst_y_start:dst_y_end] = \
        arr[src_x_start:src_x_end, src_y_start:src_y_end]

    # Boundary: replicate nearest edge value
    # Top rows (ix < vx if vx > 0)
    if vx > 0:
        result[:vx, :] = arr[0, :]
    elif vx < 0:
        result[nx+vx:, :] = arr[-1, :]
    # Bottom rows
    if vy > 0:
        result[:, :vy] = arr[:, 0:1]
    elif vy < 0:
        result[:, ny+vy:] = arr[:, -1:]

    return result


def lbm_streaming(fin, fout, cell_type, mass, rho_prev, mass_prev, gravity):
    """Free-surface streaming step."""
    nx, ny = fin.shape[1], fin.shape[2]
    fin_new = fin.copy()
    mass_new = mass.copy()

    rho_tmp, ux_tmp, uy_tmp = update_rho_u(fin)
    feq = set_equilibrium(rho_tmp, ux_tmp, uy_tmp)

    is_fluid = (cell_type & CT_FLUID) != 0
    is_interface = (cell_type & CT_INTERFACE) != 0
    is_obstacle = (cell_type & CT_OBSTACLE) != 0
    is_gas = (cell_type & CT_GAS) != 0

    eps = get_epsilon(cell_type, rho_prev, mass_prev)
    normal_full = get_normal(cell_type, rho_prev, mass_prev)

    for n in range(1, 9):
        vx, vy = int(V[n,0]), int(V[n,1])
        n_opp = int(V_INV[n])

        # Pull-streamed values
        pulled = _pull_fast(fout[n], vx, vy)
        ct_nei = _pull_fast(cell_type.astype(np.float64), vx, vy).astype(np.int64)
        eps_nei = _pull_fast(eps, vx, vy)

        is_obstacle_nei = (ct_nei & CT_OBSTACLE) != 0
        is_fluid_nei = (ct_nei & CT_FLUID) != 0
        is_interface_nei = (ct_nei & CT_INTERFACE) != 0
        is_gas_nei = (ct_nei & CT_GAS) != 0

        # FLUID cells: standard pull, bounce-back at obstacle neighbours
        fluid_val = np.where(is_obstacle_nei, fout[n_opp], pulled)
        fin_new[n] = np.where(is_fluid, fluid_val, fin_new[n])

        # INTERFACE cells
        iface_val = pulled.copy()
        iface_val = np.where(is_interface & is_obstacle_nei, fout[n_opp], iface_val)
        iface_val = np.where(
            is_interface & is_gas_nei,
            feq[n] + feq[n_opp] - fout[n_opp],
            iface_val
        )
        # Surface normal correction
        v_inv_dot_n = V[n_opp,0]*normal_full[:,:,0] + V[n_opp,1]*normal_full[:,:,1]
        outgoing = v_inv_dot_n > 0
        iface_val = np.where(
            is_interface & outgoing & ~is_obstacle_nei & ~is_gas_nei,
            feq[n] + feq[n_opp] - fout[n_opp],
            iface_val
        )
        fin_new[n] = np.where(is_interface, iface_val, fin_new[n])

        # Mass exchange for interface cells
        delta_mass = np.where(
            is_interface & is_fluid_nei,
            pulled - fout[n_opp],
            np.where(
                is_interface & is_interface_nei,
                (pulled - fout[n_opp]) * 0.5 * (eps + eps_nei),
                0.0
            )
        )
        mass_new += delta_mass

    # Obstacle cells: zero
    for n in range(9):
        fin_new[n] = np.where(is_obstacle, 0.0, fin_new[n])

    return fin_new, mass_new


def _shift_fast(arr, vx, vy):
    """Shift: result[ix,iy] = arr[ix+vx, iy+vy].  Boundary: replicate edge."""
    nx, ny = arr.shape
    result = np.empty_like(arr)
    if vx >= 0:
        src_x_start, src_x_end = vx, nx
        dst_x_start, dst_x_end = 0, nx - vx
    else:
        src_x_start, src_x_end = 0, nx + vx
        dst_x_start, dst_x_end = -vx, nx
    if vy >= 0:
        src_y_start, src_y_end = vy, ny
        dst_y_start, dst_y_end = 0, ny - vy
    else:
        src_y_start, src_y_end = 0, ny + vy
        dst_y_start, dst_y_end = -vy, ny

    result[dst_x_start:dst_x_end, dst_y_start:dst_y_end] = \
        arr[src_x_start:src_x_end, src_y_start:src_y_end]

    if vx > 0:
        result[nx-vx:, :] = arr[-1, :]
    elif vx < 0:
        result[:-vx, :] = arr[0, :]
    if vy > 0:
        result[:, ny-vy:] = arr[:, -1:]
    elif vy < 0:
        result[:, :-vy] = arr[:, 0:1]

    return result


def update_types(rho, cell_type, mass, rho_prev, mass_prev, ux_prev, uy_prev, cell_type_prev):
    """Update cell-type flags."""
    nx, ny = rho.shape
    fill_offset = 0.003
    lonely_tresh = 0.1

    cell_type = cell_type.copy()
    mass = mass.copy()
    rho = rho.copy()

    is_iface = (cell_type & CT_INTERFACE) != 0
    has_empty = (cell_type & CT_NO_EMPTY_NEIGH) == 0
    has_fluid = (cell_type & CT_NO_FLUID_NEIGH) == 0
    has_iface = (cell_type & CT_NO_IFACE_NEIGH) == 0

    to_fluid = is_iface & (
        (mass > (1 + fill_offset) * rho) |
        ((mass >= (1 - lonely_tresh) * rho) & has_empty)
    )
    to_gas = is_iface & (
        (mass < -fill_offset * rho) |
        ((mass <= lonely_tresh * rho) & has_fluid) |
        (has_iface & has_fluid)
    )

    cell_type[to_fluid] |= CT_TO_FLUID
    cell_type[to_gas] |= CT_TO_GAS
    cell_type &= ~(CT_NO_FLUID_NEIGH + CT_NO_EMPTY_NEIGH + CT_NO_IFACE_NEIGH)

    # Interface → fluid: convert adjacent gas cells to interface
    to_fluid_mask = (cell_type & CT_TO_FLUID) != 0
    for n in range(1, 9):
        vx, vy = int(V[n,0]), int(V[n,1])
        shifted = _shift_fast(to_fluid_mask.astype(np.float64), vx, vy).astype(bool)
        gas_to_iface = shifted & ((cell_type & CT_GAS) != 0)
        if gas_to_iface.any():
            _average_surround(gas_to_iface, rho, rho_prev, mass, mass_prev,
                              ux_prev, uy_prev, cell_type_prev)
            cell_type[gas_to_iface] = CT_INTERFACE

    # Interface → gas: convert adjacent fluid cells to interface
    to_gas_mask = (cell_type & CT_TO_GAS) != 0
    for n in range(1, 9):
        vx, vy = int(V[n,0]), int(V[n,1])
        shifted = _shift_fast(to_gas_mask.astype(np.float64), vx, vy).astype(bool)
        fluid_to_iface = shifted & ((cell_type & CT_FLUID) != 0)
        cell_type[fluid_to_iface] = CT_INTERFACE

    # Distribute excess mass
    fdist = np.zeros((9, nx, ny), dtype=np.float64)
    normal_full = get_normal(cell_type_prev, rho_prev, mass_prev)

    is_to_fluid = (cell_type & CT_TO_FLUID) != 0
    is_to_gas = (cell_type & CT_TO_GAS) != 0
    is_not_obs = (cell_type & CT_OBSTACLE) == 0

    mex = np.zeros((nx, ny), dtype=np.float64)
    normal_dir = np.zeros((nx, ny, 2), dtype=np.float64)

    mex[is_to_fluid] = mass[is_to_fluid] - rho[is_to_fluid]
    mass[is_to_fluid] = rho[is_to_fluid]
    normal_dir[is_to_fluid] = normal_full[is_to_fluid]

    mex[is_to_gas] = mass[is_to_gas]
    mass[is_to_gas] = 0.0
    normal_dir[is_to_gas] = -normal_full[is_to_gas]

    needs_dist = is_not_obs & (is_to_fluid | is_to_gas)

    eta_arr = np.zeros((9, nx, ny), dtype=np.float64)
    is_iface_nei_arr = np.zeros((9, nx, ny), dtype=bool)

    for n in range(1, 9):
        vx, vy = int(V[n,0]), int(V[n,1])
        eta = vx * normal_dir[:,:,0] + vy * normal_dir[:,:,1]
        eta = np.maximum(eta, 0.0)
        ct_nei = _shift_fast(cell_type.astype(np.float64), vx, vy).astype(np.int64)
        is_iface_nei = (ct_nei & CT_INTERFACE) != 0
        eta_arr[n] = eta
        is_iface_nei_arr[n] = is_iface_nei

    eta_total = eta_arr[1:].sum(axis=0)
    IF_total = is_iface_nei_arr[1:].sum(axis=0).astype(np.float64)

    for n in range(1, 9):
        has_eta = eta_total > 0
        no_eta_has_IF = (~has_eta) & (IF_total > 0) & needs_dist
        fdist[n] = np.where(
            needs_dist & has_eta & is_iface_nei_arr[n],
            mex * eta_arr[n] / np.maximum(eta_total, 1e-30),
            np.where(
                needs_dist & no_eta_has_IF & is_iface_nei_arr[n],
                mex / np.maximum(IF_total, 1e-30),
                0.0
            )
        )

    # Collect distributed mass
    is_iface_now = (cell_type & CT_INTERFACE) != 0
    for n in range(1, 9):
        vx, vy = int(V[n,0]), int(V[n,1])
        fdist_from_nei = _shift_fast(fdist[n], vx, vy)
        mass[is_iface_now] += fdist_from_nei[is_iface_now]

    cell_type[is_to_fluid] = CT_FLUID
    cell_type[is_to_gas] = CT_GAS

    # Recompute neighbourhood flags
    for ix in range(nx):
        for iy in range(ny):
            if cell_type[ix, iy] & CT_OBSTACLE:
                continue
            cell_type[ix, iy] |= (CT_NO_FLUID_NEIGH | CT_NO_EMPTY_NEIGH | CT_NO_IFACE_NEIGH)
            for n in range(1, 9):
                ix_next = ix - V[n,0]
                iy_next = iy - V[n,1]
                if 0 <= ix_next < nx and 0 <= iy_next < ny:
                    if cell_type[ix_next, iy_next] & CT_FLUID:
                        cell_type[ix, iy] &= ~CT_NO_FLUID_NEIGH
                    elif cell_type[ix_next, iy_next] & CT_GAS:
                        cell_type[ix, iy] &= ~CT_NO_EMPTY_NEIGH
                    elif cell_type[ix_next, iy_next] & CT_INTERFACE:
                        cell_type[ix, iy] &= ~CT_NO_IFACE_NEIGH

    return cell_type, mass, rho


def _average_surround(mask, rho, rho_prev, mass, mass_prev, ux_prev, uy_prev, cell_type_prev):
    """Initialize newly created interface cells with averaged neighbour values."""
    nx, ny = rho.shape
    is_fi = (cell_type_prev & (CT_FLUID | CT_INTERFACE)) != 0

    rho_sum = np.zeros((nx, ny), dtype=np.float64)
    ux_sum = np.zeros((nx, ny), dtype=np.float64)
    uy_sum = np.zeros((nx, ny), dtype=np.float64)
    count = np.zeros((nx, ny), dtype=np.float64)

    for n in range(1, 9):
        vx, vy = int(V[n,0]), int(V[n,1])
        rho_nei = _shift_fast(rho_prev, vx, vy)
        ux_nei = _shift_fast(ux_prev, vx, vy)
        uy_nei = _shift_fast(uy_prev, vx, vy)
        valid_nei = _shift_fast(is_fi.astype(np.float64), vx, vy) > 0.5

        rho_sum += np.where(valid_nei, rho_nei, 0.0)
        ux_sum += np.where(valid_nei, ux_nei, 0.0)
        uy_sum += np.where(valid_nei, uy_nei, 0.0)
        count += valid_nei.astype(np.float64)

    has_count = count > 0
    rho[mask & has_count] = (rho_sum / np.maximum(count, 1))[mask & has_count]
    mass[mask] = 0.0


@dataclass
class DamBreakResult:
    front_positions: np.ndarray
    front_times: np.ndarray
    liquid_mask: np.ndarray
    mass_history: np.ndarray
    rho_final: np.ndarray
    ux_final: np.ndarray
    uy_final: np.ndarray
    cell_type_final: np.ndarray


def run_dam_break(nx=80, ny=80, dam_width=30, dam_height=40,
                  omega=1.0, gravity=(0.0, -0.1), n_steps=300, snapshot_interval=10):
    """Run dam-break using the Maarten-vd-Sande/lbm free-surface D2Q9 algorithm."""
    fin = np.zeros((9, nx, ny), dtype=np.float64)
    rho = np.zeros((nx, ny), dtype=np.float64)
    ux = np.zeros((nx, ny), dtype=np.float64)
    uy = np.zeros((nx, ny), dtype=np.float64)
    mass = np.zeros((nx, ny), dtype=np.float64)
    cell_type = np.full((nx, ny), CT_GAS, dtype=np.int64)

    dam_y_offset = ny - dam_height
    cell_type[1:dam_width, dam_y_offset+1:ny-1] = CT_FLUID
    cell_type[1:dam_width, dam_y_offset] = CT_INTERFACE
    cell_type[dam_width, dam_y_offset:ny-1] = CT_INTERFACE

    cell_type[:, 0] = CT_OBSTACLE
    cell_type[:, -1] = CT_OBSTACLE
    cell_type[0, :] = CT_OBSTACLE
    cell_type[-1, :] = CT_OBSTACLE

    is_fluid = (cell_type & CT_FLUID) != 0
    is_iface = (cell_type & CT_INTERFACE) != 0
    mass[is_fluid] = 1.0
    mass[is_iface] = 0.5
    rho[:] = mass.copy()

    feq = set_equilibrium(rho, ux, uy)
    fin[:] = feq.copy()

    # Set initial neighbourhood flags
    for ix in range(nx):
        for iy in range(ny):
            if cell_type[ix, iy] & CT_OBSTACLE:
                continue
            cell_type[ix, iy] |= (CT_NO_FLUID_NEIGH | CT_NO_EMPTY_NEIGH | CT_NO_IFACE_NEIGH)
            for n in range(1, 9):
                ix_next = ix - V[n,0]
                iy_next = iy - V[n,1]
                if 0 <= ix_next < nx and 0 <= iy_next < ny:
                    if cell_type[ix_next, iy_next] & CT_FLUID:
                        cell_type[ix, iy] &= ~CT_NO_FLUID_NEIGH
                    elif cell_type[ix_next, iy_next] & CT_GAS:
                        cell_type[ix, iy] &= ~CT_NO_EMPTY_NEIGH
                    elif cell_type[ix_next, iy_next] & CT_INTERFACE:
                        cell_type[ix, iy] &= ~CT_NO_IFACE_NEIGH

    front_positions = []
    front_times = []
    liquid_masks = []
    mass_history = []

    for step in range(n_steps):
        fout = collide_bgk(fin, rho, ux, uy, omega, gravity)
        mass_prev = mass.copy()
        rho_prev = rho.copy()

        fin, mass = lbm_streaming(fin, fout, cell_type, mass, rho_prev, mass_prev, gravity)

        rho_prev2 = rho.copy()
        mass_prev2 = mass.copy()
        ux_prev = ux.copy()
        uy_prev = uy.copy()
        cell_type_prev = cell_type.copy()

        rho, ux, uy = update_rho_u(fin)
        is_fluid_now = (cell_type & CT_FLUID) != 0
        rho[is_fluid_now] = mass[is_fluid_now]

        cell_type, mass, rho = update_types(
            rho, cell_type, mass, rho_prev2, mass_prev2, ux_prev, uy_prev, cell_type_prev
        )

        if step % snapshot_interval == 0:
            liquid = ((cell_type & CT_LIQUID) != 0)
            liquid_masks.append(liquid.copy())
            mass_history.append(float(np.nansum(mass[liquid])) if liquid.any() else 0.0)
            col_has_liquid = liquid.any(axis=1)
            front_x = float(np.max(np.where(col_has_liquid)[0])) if col_has_liquid.any() else 0.0
            front_positions.append(front_x)
            front_times.append(step)

            if step % (snapshot_interval * 10) == 0:
                n_nan = int(np.isnan(fin).sum())
                print(f"  step {step}/{n_steps}  front_x={front_x:.1f}  "
                      f"mass={mass_history[-1]:.4f}  n_liquid={int(liquid.sum())}  "
                      f"nan={n_nan}", flush=True)

    return DamBreakResult(
        front_positions=np.array(front_positions),
        front_times=np.array(front_times),
        liquid_mask=np.array(liquid_masks),
        mass_history=np.array(mass_history),
        rho_final=rho.copy(),
        ux_final=ux.copy(),
        uy_final=uy.copy(),
        cell_type_final=cell_type.copy(),
    )
