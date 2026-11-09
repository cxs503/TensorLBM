P = "src/tensorlbm/boundary/partial_slip_wall3d.py"
src = open(P, encoding="utf-8").read()


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:80])
    src = src.replace(old, new)


rep(
    "    steps: int = 60000,\n"
    "    tol: float = 1e-10,\n"
    "    track_momentum: bool = False,\n"
    ") -> Channel3DResult:",
    "    steps: int = 60000,\n"
    "    tol: float = 1e-10,\n"
    "    track_momentum: bool = False,\n"
    "    init_wave: float = 0.0,\n"
    ") -> Channel3DResult:",
)

rep(
    '''    zero = np.zeros((nx, ny, nz))
    f = _equilibrium(lattice, np.ones_like(zero), zero, zero, zero)
''',
    '''    zero = np.zeros((nx, ny, nz))
    ux0 = zero
    if init_wave:
        # Non-uniform start: a single x/z wave of relative amplitude
        # ``init_wave`` added to the streamwise velocity.  It cannot survive a
        # steady state that is x/z-uniform, so the run doubles as a collapse
        # test: uniformity must be *reached*, not merely preserved from a
        # symmetric initial condition.
        if not 0.0 <= init_wave < 1.0:
            raise ValueError(f"init_wave must be in [0, 1), got {init_wave}")
        nu = (tau - 0.5) / 3.0
        if nu <= 0.0:
            raise ValueError("tau must be > 0.5")
        amp = float(init_wave) * g * ((ny - 2) / 2.0) ** 2 / (3.0 * nu)
        ix = np.arange(nx).reshape(nx, 1, 1)
        iz = np.arange(nz).reshape(1, 1, nz)
        ux0 = amp * np.sin(2.0 * np.pi * ix / nx) * np.cos(2.0 * np.pi * iz / nz) * np.ones(
            (1, ny, 1)
        )
    f = _equilibrium(lattice, np.ones_like(zero), ux0, zero, zero)
''',
)

open(P, "w", encoding="utf-8").write(src)
print("ok")