import pathlib

p = pathlib.Path("/srv/tensor/agent/work/m20669/src/tensorlbm/boundary/interpolated_bounce_back.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)


# ---------------------------------------------------------------- imports -----
rep(
    "from tensorlbm.d2q9 import C as _C2\nfrom tensorlbm.d2q9 import OPPOSITE as _OPP2\n"
    "from tensorlbm.d3q19 import C as _C3\nfrom tensorlbm.d3q19 import OPPOSITE as _OPP3\n",
    "from tensorlbm.d2q9 import C as _C2\nfrom tensorlbm.d2q9 import OPPOSITE as _OPP2\n"
    "from tensorlbm.d2q9 import W as _W2\n"
    "from tensorlbm.d3q19 import C as _C3\nfrom tensorlbm.d3q19 import OPPOSITE as _OPP3\n"
    "from tensorlbm.d3q19 import W as _W3\n",
)

# ------------------------------------------------------------- module doc ----
rep(
    """Degenerate links (``q < 1/2`` where the node behind the fluid node is itself
solid, i.e. a fluid layer thinner than one cell) fall back to ``q = 1/2``; the
number of such links is returned for disclosure.
\"\"\"""",
    """Degenerate links (``q < 1/2`` where the node behind the fluid node is itself
solid, i.e. a fluid layer thinner than one cell) fall back to ``q = 1/2``; the
number of such links is returned for disclosure.

Moving / non-zero-velocity walls
--------------------------------
The general ``alpha`` form of the closure carries a Dirichlet term
``-alpha e^-_q(x_w)`` with ``e^-_q = (f^eq_q - f^eq_opp(q)) / 2`` evaluated with
the *wall* density and velocity.  For unit density the quadratic terms cancel
between opposite directions and the term is exactly

    e^-_q(x_w) = 3 w_q rho_w (c_q . u_w)

Pass ``wall_velocity`` (and optionally ``wall_density`` / ``weights``) to include
it.  Note that this term is **not** a wall-normal-only correction: it is
``3 w_q rho_w c_qx u_wx`` for a wall moving in x, which is non-zero on the
*diagonal* cut links just as much as on the axis link, so a wall-aligned channel
with a sliding wall needs it too (measured: leaving it out puts an O(1) slip
error on the wall value; see ``xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME/
REPORT_V3.md``).  It is zero only for a wall at rest, which is why the historical
two-branch operator could ignore it.

``wall_velocity`` is given in the *same* velocity convention as the equilibrium
used for the fluid nodes (i.e. the stored momentum).  With a body force added
after the collision with isotropic weights and a reported velocity of
``sum f c / rho - F/2``, the stored wall velocity of a no-slip wall is
``F/2``, not zero; pass that half-step contribution explicitly.
\"\"\"""",
)

# --------------------------------------------------------------- signature ---
rep(
    """    alpha: float | Callable[[torch.Tensor], torch.Tensor] | None = None,
    post_collision_correction: bool = False,
    f_old: torch.Tensor | None = None,
    return_report: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, InterpolatedBounceBackReport]:""",
    """    alpha: float | Callable[[torch.Tensor], torch.Tensor] | None = None,
    post_collision_correction: bool = False,
    f_old: torch.Tensor | None = None,
    wall_velocity: torch.Tensor | tuple[float, ...] | None = None,
    wall_density: float | torch.Tensor = 1.0,
    weights: torch.Tensor | None = None,
    return_report: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, InterpolatedBounceBackReport]:""",
)

# ----------------------------------------------------------- args docstring --
rep(
    """            The reflected value is

                ``alpha_hat f_pre(i) + beta_hat f_pre(opp i) + beta f_new(i)``

            with ``alpha_hat = -1 + alpha (1/2 + q)``, ``beta = 1 - alpha q``,
            ``beta_hat = 1 - alpha/2`` (zero-velocity wall, so the antisymmetric
            equilibrium term ``-alpha e^-_q(x_w)`` of the pylbm form vanishes).""",
    """            The reflected value is

                ``alpha_hat f_pre(i) + beta_hat f_pre(opp i) + beta f_new(i)
                - alpha e^-_q(x_w)``

            with ``alpha_hat = -1 + alpha (1/2 + q)``, ``beta = 1 - alpha q``,
            ``beta_hat = 1 - alpha/2``; the Dirichlet term is present only when
            ``wall_velocity`` is given (it vanishes identically for a wall at
            rest).""",
)

rep(
    """        return_report: also return an :class:`InterpolatedBounceBackReport`.""",
    """        wall_velocity: wall velocity for the cut links, in the equilibrium
            (stored-momentum) convention, broadcastable to ``(*spatial, dim)``
            and evaluated at the *fluid* end of each link.  ``None`` keeps the
            historical zero-velocity closure; a zero tensor is equivalent but
            exercises the Dirichlet code path (bit-identical result).
        wall_density: wall density of the Dirichlet term, scalar or array
            broadcastable to ``spatial`` (default ``1.0``).
        weights: lattice weights ``(Q,)`` matching ``c`` -- required when
            ``wall_velocity`` is given, since the term is
            ``3 w_q rho_w (c_q . u_w)``.  The 2D/3D wrappers supply them.
        return_report: also return an :class:`InterpolatedBounceBackReport`.""",
)

# ------------------------------------------------------------- validation ---
rep(
    """    dim = len(spatial)
    device = f_new.device
    c = c.to(device=device)""",
    """    dim = len(spatial)
    device = f_new.device
    c = c.to(device=device)
    if wall_velocity is not None:
        if int(c.shape[1]) != dim:
            raise ValueError(f"c must have shape (Q, {dim}), got {tuple(c.shape)}")
        if weights is None:
            raise ValueError("wall_velocity requires the lattice weights (pass weights=...)")
        wall_velocity = torch.as_tensor(wall_velocity, device=device, dtype=f_new.dtype)
        try:
            wall_velocity = torch.broadcast_to(wall_velocity, spatial + (dim,))
        except RuntimeError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"wall_velocity must broadcast to {spatial + (dim,)}, got "
                f"{tuple(wall_velocity.shape)}"
            ) from exc
        if not bool(torch.isfinite(wall_velocity).all()):
            raise ValueError("wall_velocity contains non-finite values")
        w_row = torch.as_tensor(weights, device=device, dtype=f_new.dtype)
        if tuple(w_row.shape) != (int(c.shape[0]),):
            raise ValueError(
                f"weights must have shape ({int(c.shape[0])},), got {tuple(w_row.shape)}"
            )
        rho_w = torch.as_tensor(wall_density, device=device, dtype=f_new.dtype)
        rho_w = torch.broadcast_to(rho_w, spatial) if tuple(rho_w.shape) != spatial else rho_w
    else:
        w_row = None
        rho_w = None""",
)

# --------------------------------------------------------- the value term ---
rep(
    """        val = alpha_hat * f_i_pre + beta_hat * f_d_pre + beta * f_i_new
        if post_collision_correction:""",
    """        val = alpha_hat * f_i_pre + beta_hat * f_d_pre + beta * f_i_new
        if wall_velocity is not None:
            # Dirichlet term -alpha e^-_q(x_w), e^-_q = 3 w_q rho_w (c_q . u_w)
            # (the quadratic equilibrium terms cancel between opposite directions).
            assert w_row is not None and rho_w is not None
            ci = c[i].to(dtype=f_new.dtype)
            e_minus = 3.0 * w_row[i] * rho_w * (wall_velocity * ci).sum(dim=-1)
            val = val - alpha_q * e_minus
        if post_collision_correction:""",
)

# --------------------------------------------------------------- wrappers ---
rep(
    """    return interpolated_bounce_back(f_new, f_pre, sdf, solid, _C2, _OPP2, **kwargs)  # type: ignore[arg-type]""",
    """    kwargs.setdefault("weights", _W2)
    return interpolated_bounce_back(f_new, f_pre, sdf, solid, _C2, _OPP2, **kwargs)  # type: ignore[arg-type]""",
)
rep(
    """    return interpolated_bounce_back(f_new, f_pre, sdf, solid, _C3, _OPP3, **kwargs)  # type: ignore[arg-type]""",
    """    kwargs.setdefault("weights", _W3)
    return interpolated_bounce_back(f_new, f_pre, sdf, solid, _C3, _OPP3, **kwargs)  # type: ignore[arg-type]""",
)

p.write_text(s)
print("patched", len(s.splitlines()), "lines")