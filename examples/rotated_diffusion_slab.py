"""Harmonic diffusion on a unit slab with periodic y and rotated material axes."""

import numpy as np

from tensorlbm.scalar_harmonic import solve_harmonic_scalar_slab


def errors(n, kxy, insulated=False):
    # Unit heat capacity, physical K, and diffusive scaling dt = dx²/12.
    dx = 1 / n
    dt = dx**2 / 12
    kxx, kyy = 2.0, 1.0
    omega = b = 2 * np.pi
    x = (np.arange(n) + 0.5) * dx
    faces = np.arange(n + 1) * dx
    phase = np.exp(1j * b * x)[:, None]
    sol = solve_harmonic_scalar_slab(
        n,
        n,
        0.5 + kxx / 4,
        omega * dt,
        phase[:, 0],
        None if insulated else 0.0,
        tau_y=0.5 + kyy / 4,
        tau_xy=kxy / 4,
    )
    # T = exp(i b y) f(x), f(0)=1, q = -K grad(T). At x=1 impose f=0 or qx=0.
    beta = b * kxy / kxx
    lam = np.sqrt((b * b * (kyy - kxy * kxy / kxx) + 1j * omega) / kxx)

    def profile(x):
        shape, gradient = (np.cosh, np.sinh) if insulated else (np.sinh, np.cosh)
        scale = np.exp(-1j * beta * x) / shape(lam)
        f = scale * shape(lam * (1 - x))
        derivative = -1j * beta * f - lam * scale * gradient(lam * (1 - x))
        return f, -kxx * derivative - 1j * b * kxy * f, -kxy * derivative - 1j * b * kyy * f

    # Compare face-mean flux and cell-mean temperature.
    nodes, weights = np.polynomial.legendre.leggauss(8)
    f, _, qy = profile(x[:, None] + 0.5 * dx * nodes)
    mean_phase = phase * np.sinc(b * dx / (2 * np.pi))
    refs = (
        mean_phase * (f @ weights / 2),
        mean_phase * profile(faces)[1],
        np.exp(1j * b * faces[1:])[:, None] * (qy @ weights / 2),
    )
    results = (
        sol.scalar.numpy(),
        sol.x_transfer.numpy() * dx / dt,
        sol.y_transfer.numpy() * dx / dt,
    )
    if insulated:
        np.testing.assert_allclose(results[1][:, -1], 0.0, atol=2e-14)
    return np.array(
        [np.linalg.norm(value - ref) / np.linalg.norm(ref) for value, ref in zip(results, refs)]
    )


if __name__ == "__main__":
    for cross, insulated in ((-0.65, False), (0.65, False), (-0.65, True), (0.65, True)):
        previous = None
        for n in (16, 32, 64, 128):
            err = errors(n, cross, insulated)
            order = np.log2(previous / err) if previous is not None else []
            print(
                f"Kxy={cross:+.2f} insulated={insulated} n={n} rel_L2(T,qx,qy)={err} order={order}"
            )
            if previous is not None and not np.all(order > 1.8):
                raise SystemExit("observed order below 1.8")
            previous = err
