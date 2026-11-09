import re  # noqa: F401

P = "src/tensorlbm/boundary/partial_slip_wall3d.py"
src = open(P, encoding="utf-8").read()
orig = src


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:80])
    src = src.replace(old, new)


# ---------------------------------------------------------------- imports
rep(
    "from tensorlbm.d3q19 import C as _C_TORCH\n"
    "from tensorlbm.d3q19 import OPPOSITE as _OPP_TORCH\n"
    "from tensorlbm.d3q19 import W_EXACT64 as _W_TORCH\n",
    "from tensorlbm.d3q19 import C as _C_TORCH\n"
    "from tensorlbm.d3q19 import OPPOSITE as _OPP_TORCH\n"
    "from tensorlbm.d3q19 import W_EXACT64 as _W_TORCH\n"
    "from tensorlbm.d3q27 import C as _C27_TORCH\n"
    "from tensorlbm.d3q27 import OPPOSITE as _OPP27_TORCH\n"
    "from tensorlbm.d3q27 import W_EXACT64 as _W27_TORCH\n",
)

rep(
    '    "C19",\n    "W19",\n    "OPP19",\n    "MIRROR_Y",\n    "mirror_table_y",\n    "mixed_wall_row_3d",\n',
    '    "C19",\n    "W19",\n    "OPP19",\n    "MIRROR_Y",\n    "C27",\n    "W27",\n    "OPP27",\n    "MIRROR_Y_27",\n'
    '    "mirror_table_y",\n    "mirror_table",\n    "Lattice",\n    "LATTICE_19",\n    "LATTICE_27",\n    "LATTICES",\n'
    '    "mixed_wall_row_lattice",\n    "mixed_wall_row_3d",\n    "channel_poiseuille_lattice",\n    "channel_poiseuille_3d",\n',
)

# ---------------------------------------------------------------- tables
OLD_TABLES = '''# ---------------------------------------------------------------------------
# Lattice tables -- taken from the repository's canonical D3Q19 module so the
# mirror table cannot drift away from the lattice constants.
# ---------------------------------------------------------------------------
C19 = np.asarray(_C_TORCH.cpu().numpy(), dtype=np.int64)
W19 = np.asarray(_W_TORCH.cpu().numpy(), dtype=np.float64)
OPP19 = np.asarray(_OPP_TORCH.cpu().numpy(), dtype=np.int64)
_Q = int(C19.shape[0])


def mirror_table_y() -> np.ndarray:
    """Direction with ``c_y`` flipped and ``c_x, c_z`` kept (wall normal ``e_y``).

    Derived from the lattice constants instead of being hand-transcribed, so a
    change of velocity set cannot silently break the specular half of the wall.
    """
    lookup = {tuple(int(v) for v in C19[q]): q for q in range(_Q)}
    return np.array([lookup[(int(c[0]), -int(c[1]), int(c[2]))] for c in C19], dtype=np.int64)


MIRROR_Y = mirror_table_y()

_CX = C19[:, 0].reshape(_Q, 1, 1, 1)
_CY = C19[:, 1].reshape(_Q, 1, 1, 1)
_CZ = C19[:, 2].reshape(_Q, 1, 1, 1)
_W = W19.reshape(_Q, 1, 1, 1)
_INWARD_LOW = (C19[:, 1] > 0).reshape(_Q, 1, 1)     # stream wall -> fluid at j=1
_INWARD_HIGH = (C19[:, 1] < 0).reshape(_Q, 1, 1)    # stream wall -> fluid at j=ny-2
'''

NEW_TABLES = '''# ---------------------------------------------------------------------------
# Lattice tables -- taken from the repository's canonical lattice modules
# (``d3q19`` / ``d3q27``) so neither the mirror table nor the weights can drift
# away from the velocity sets they are supposed to describe.
# ---------------------------------------------------------------------------
C19 = np.asarray(_C_TORCH.cpu().numpy(), dtype=np.int64)
W19 = np.asarray(_W_TORCH.cpu().numpy(), dtype=np.float64)
OPP19 = np.asarray(_OPP_TORCH.cpu().numpy(), dtype=np.int64)
C27 = np.asarray(_C27_TORCH.cpu().numpy(), dtype=np.int64)
W27 = np.asarray(_W27_TORCH.cpu().numpy(), dtype=np.float64)
OPP27 = np.asarray(_OPP27_TORCH.cpu().numpy(), dtype=np.int64)


def mirror_table(C: np.ndarray) -> np.ndarray:
    """Direction with ``c_y`` flipped and ``c_x, c_z`` kept (wall normal ``e_y``).

    Derived from the lattice constants instead of being hand-transcribed, so a
    change of velocity set cannot silently break the specular half of the wall.
    """
    C = np.asarray(C, dtype=np.int64)
    lookup = {tuple(int(v) for v in C[q]): q for q in range(C.shape[0])}
    return np.array(
        [lookup[(int(c[0]), -int(c[1]), int(c[2]))] for c in C], dtype=np.int64
    )


def mirror_table_y() -> np.ndarray:
    """D3Q19 specialisation of :func:`mirror_table` (kept for compatibility)."""
    return mirror_table(C19)


MIRROR_Y = mirror_table_y()
MIRROR_Y_27 = mirror_table(C27)


class Lattice:
    """A velocity set plus the broadcast-shaped tables the wall and solver need.

    Everything that used to be a module-level constant of the D3Q19
    implementation (reshaped lattice vectors, weights, the specular mirror
    table and the two wall-inward direction masks) lives here, so the same
    kernels run on any velocity set without a second copy of the numerics.
    """

    __slots__ = (
        "name", "C", "W", "OPP", "MIRROR", "Q",
        "cx", "cy", "cz", "w", "inward_low", "inward_high",
    )

    def __init__(self, name: str, C: np.ndarray, W: np.ndarray, OPP: np.ndarray):
        self.name = str(name)
        self.C = np.asarray(C, dtype=np.int64)
        self.W = np.asarray(W, dtype=np.float64)
        self.OPP = np.asarray(OPP, dtype=np.int64)
        self.MIRROR = mirror_table(self.C)
        self.Q = int(self.C.shape[0])
        q = self.Q
        self.cx = self.C[:, 0].reshape(q, 1, 1, 1)
        self.cy = self.C[:, 1].reshape(q, 1, 1, 1)
        self.cz = self.C[:, 2].reshape(q, 1, 1, 1)
        self.w = self.W.reshape(q, 1, 1, 1)
        # stream wall -> fluid at j=1 (low wall) / j=ny-2 (high wall)
        self.inward_low = (self.C[:, 1] > 0).reshape(q, 1, 1)
        self.inward_high = (self.C[:, 1] < 0).reshape(q, 1, 1)

    def __repr__(self) -> str:                      # pragma: no cover - cosmetic
        return f"Lattice({self.name!r}, Q={self.Q})"


LATTICE_19 = Lattice("D3Q19", C19, W19, OPP19)
LATTICE_27 = Lattice("D3Q27", C27, W27, OPP27)
LATTICES = {"D3Q19": LATTICE_19, "D3Q27": LATTICE_27}

# Backwards-compatible aliases for the D3Q19-shaped constants.
_Q = LATTICE_19.Q
_CX = LATTICE_19.cx
_CY = LATTICE_19.cy
_CZ = LATTICE_19.cz
_W = LATTICE_19.w
_INWARD_LOW = LATTICE_19.inward_low
_INWARD_HIGH = LATTICE_19.inward_high
'''
rep(OLD_TABLES, NEW_TABLES)

# ---------------------------------------------------------------- wall row
OLD_WALL = '''def mixed_wall_row_3d(f_post_adjacent: np.ndarray, sigma: float, top: bool):
    """Mixed bounce-back / specular populations for one y-wall plane.
'''
NEW_WALL = '''def mixed_wall_row_lattice(
    lattice: Lattice, f_post_adjacent: np.ndarray, sigma: float, top: bool
):
    """Mixed bounce-back / specular populations for one y-wall plane.

    ``lattice`` selects the velocity set (``LATTICE_19`` / ``LATTICE_27``); the
    operator is identical in both, it is the mirror/opposite tables that carry
    the velocity set.  See :func:`mixed_wall_row_3d` for the D3Q19 signature.
'''
rep(OLD_WALL, NEW_WALL)

OLD_WALL_BODY = '''    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")
    mix = (1.0 - sigma) * f_post_adjacent[OPP19] + sigma * f_post_adjacent[MIRROR_Y]
    inward = _INWARD_HIGH if top else _INWARD_LOW
    return np.where(inward, mix, f_post_adjacent)
'''
NEW_WALL_BODY = '''    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")
    if f_post_adjacent.shape[0] != lattice.Q:
        raise ValueError(
            f"{lattice.name} expects {lattice.Q} populations per cell, "
            f"got {f_post_adjacent.shape[0]}"
        )
    mix = (1.0 - sigma) * f_post_adjacent[lattice.OPP] + sigma * f_post_adjacent[lattice.MIRROR]
    inward = lattice.inward_high if top else lattice.inward_low
    return np.where(inward, mix, f_post_adjacent)


def mixed_wall_row_3d(f_post_adjacent: np.ndarray, sigma: float, top: bool):
    """D3Q19 mixed bounce-back / specular wall row (compatibility entry point)."""
    return mixed_wall_row_lattice(LATTICE_19, f_post_adjacent, sigma, top)
'''
rep(OLD_WALL_BODY, NEW_WALL_BODY)

# ---------------------------------------------------------------- kernels
OLD_KERN = '''def _equilibrium(rho, ux, uy, uz):
    cu = 3.0 * (_CX * ux + _CY * uy + _CZ * uz)
    usq = ux * ux + uy * uy + uz * uz
    return _W * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)


def _macroscopic(f, fx=0.0, fy=0.0, fz=0.0):
    rho = f.sum(0)
    safe = np.where(rho == 0.0, 1.0, rho)       # wall rows carry no mass
    ux = ((_CX * f).sum(0) + 0.5 * fx) / safe
    uy = ((_CY * f).sum(0) + 0.5 * fy) / safe
    uz = ((_CZ * f).sum(0) + 0.5 * fz) / safe
    return rho, ux, uy, uz


def _collide_bgk_guo(f, tau, fx=0.0, fy=0.0, fz=0.0):
    """BGK collision with Guo forcing (second-order in the force)."""
    rho, ux, uy, uz = _macroscopic(f, fx, fy, fz)
    feq = _equilibrium(rho, ux, uy, uz)
    f_post = f - (f - feq) / tau
    cu = _CX * ux + _CY * uy + _CZ * uz
    src = (
        3.0 * ((_CX - ux) * fx + (_CY - uy) * fy + (_CZ - uz) * fz)
        + 9.0 * cu * (_CX * fx + _CY * fy + _CZ * fz)
    )
    return f_post + (1.0 - 0.5 / tau) * _W * src
'''
NEW_KERN = '''def _equilibrium(lattice: Lattice, rho, ux, uy, uz):
    cu = 3.0 * (lattice.cx * ux + lattice.cy * uy + lattice.cz * uz)
    usq = ux * ux + uy * uy + uz * uz
    return lattice.w * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)


def _macroscopic(lattice: Lattice, f, fx=0.0, fy=0.0, fz=0.0):
    rho = f.sum(0)
    safe = np.where(rho == 0.0, 1.0, rho)       # wall rows carry no mass
    ux = ((lattice.cx * f).sum(0) + 0.5 * fx) / safe
    uy = ((lattice.cy * f).sum(0) + 0.5 * fy) / safe
    uz = ((lattice.cz * f).sum(0) + 0.5 * fz) / safe
    return rho, ux, uy, uz


def _collide_bgk_guo(lattice: Lattice, f, tau, fx=0.0, fy=0.0, fz=0.0):
    """BGK collision with Guo forcing (second-order in the force)."""
    rho, ux, uy, uz = _macroscopic(lattice, f, fx, fy, fz)
    feq = _equilibrium(lattice, rho, ux, uy, uz)
    f_post = f - (f - feq) / tau
    cu = lattice.cx * ux + lattice.cy * uy + lattice.cz * uz
    src = (
        3.0 * ((lattice.cx - ux) * fx + (lattice.cy - uy) * fy + (lattice.cz - uz) * fz)
        + 9.0 * cu * (lattice.cx * fx + lattice.cy * fy + lattice.cz * fz)
    )
    return f_post + (1.0 - 0.5 / tau) * lattice.w * src
'''
rep(OLD_KERN, NEW_KERN)

# ---------------------------------------------------------------- result
rep(
    '    momentum_growth: float | None = None\n    geometry: str = "plane3d"\n',
    '    momentum_growth: float | None = None\n    geometry: str = "plane3d"\n'
    '    lattice: str = "D3Q19"\n',
)

# ---------------------------------------------------------------- solver
rep(
    'def channel_poiseuille_3d(\n'
    '    sigma: float,\n'
    '    ny: int = 33,\n',
    'def channel_poiseuille_lattice(\n'
    '    sigma: float,\n'
    '    lattice: Lattice = LATTICE_19,\n'
    '    ny: int = 33,\n',
)

rep(
    '''    zero = np.zeros((nx, ny, nz))
    f = _equilibrium(np.ones_like(zero), zero, zero, zero)
''',
    '''    zero = np.zeros((nx, ny, nz))
    f = _equilibrium(lattice, np.ones_like(zero), zero, zero, zero)
''',
)

rep(
    '''        fx = g * f.sum(0)
        f_post = _collide_bgk_guo(f, tau, fx=fx)
        f_post[:, :, 0, :] = mixed_wall_row_3d(f_post[:, :, 1, :], sigma, top=False)
        f_post[:, :, ny - 1, :] = mixed_wall_row_3d(
            f_post[:, :, ny - 2, :], sigma, top=True
        )
        new_f = np.zeros_like(f)
        for q in range(_Q):
            cx, cyq, cz = (int(v) for v in C19[q])
''',
    '''        fx = g * f.sum(0)
        f_post = _collide_bgk_guo(lattice, f, tau, fx=fx)
        f_post[:, :, 0, :] = mixed_wall_row_lattice(
            lattice, f_post[:, :, 1, :], sigma, top=False
        )
        f_post[:, :, ny - 1, :] = mixed_wall_row_lattice(
            lattice, f_post[:, :, ny - 2, :], sigma, top=True
        )
        new_f = np.zeros_like(f)
        for q in range(lattice.Q):
            cx, cyq, cz = (int(v) for v in lattice.C[q])
''',
)

rep(
    '''        f = new_f
        _rho, ux, _uy, _uz = _macroscopic(f, g * f.sum(0))
''',
    '''        f = new_f
        _rho, ux, _uy, _uz = _macroscopic(lattice, f, g * f.sum(0))
''',
)

rep(
    '''    return Channel3DResult(
        sigma=sigma, nx=nx, ny=ny, nz=nz, tau=tau, g=g,
        u_profile=plane_mean, u_slice=u[:, :, 0], plane_spread=spread,
        length=(ny - 2) / 2.0, steps=step, converged=converged,
        residual=residual, momentum_growth=mom_growth,
    )
''',
    '''    return Channel3DResult(
        sigma=sigma, nx=nx, ny=ny, nz=nz, tau=tau, g=g,
        u_profile=plane_mean, u_slice=u[:, :, 0], plane_spread=spread,
        length=(ny - 2) / 2.0, steps=step, converged=converged,
        residual=residual, momentum_growth=mom_growth, lattice=lattice.name,
    )


def channel_poiseuille_3d(sigma: float, **kwargs) -> Channel3DResult:
    """D3Q19 entry point (compatibility wrapper around the generic solver)."""
    return channel_poiseuille_lattice(sigma, LATTICE_19, **kwargs)
''',
)

open(P, "w", encoding="utf-8").write(src)
print("patched", len(orig.splitlines()), "->", len(src.splitlines()))