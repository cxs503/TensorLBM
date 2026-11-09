import re
from pathlib import Path

p = Path("src/tensorlbm/passive_scalar.py")
src = p.read_text()

# 1) D2Q5 opposite table + scalar no-flux bounce-back helper
anchor = '_CS2_D2Q5 = 1.0 / 3.0\n'
assert anchor in src
helper = anchor + '''
#: Opposite-direction permutation for D2Q5 (0=rest, 1=+x, 2=+y, 3=-x, 4=-y).
_OPP5 = torch.tensor([0, 3, 4, 1, 2], dtype=torch.long)


def bounce_back_scalar(g: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Bounce-back (zero-flux) for a **D2Q5 scalar** distribution on solid cells.

    ``tensorlbm.boundary.boundaries.bounce_back_cells`` is defined for the
    D2Q9 *momentum* lattice (its opposite table has 9 entries) and therefore
    raises ``IndexError`` when handed a 5-population scalar field; the scalar
    analogue must reverse the D2Q5 directions itself.

    Args:
        g:    Scalar DF, shape ``(5, ny, nx)``.
        mask: Boolean solid mask, shape ``(ny, nx)``.

    Returns:
        Distribution with the populations of masked cells reversed.
    """
    if g.shape[0] != 5:
        msg = f"bounce_back_scalar expects a D2Q5 distribution (5, ny, nx), got {tuple(g.shape)}"
        raise ValueError(msg)
    opp = _OPP5.to(g.device)
    return torch.where(mask.unsqueeze(0), g[opp], g)
'''
src = src.replace(anchor, helper, 1)

# 2) fix prepare_run_dir call (signature: output_root, subdir, run_name, overwrite)
old = 'run_dir = prepare_run_dir(cfg.output_root, cfg.run_name or "passive_scalar", cfg.overwrite)'
assert old in src, "prepare_run_dir call not found"
new = ('run_dir = prepare_run_dir(\n'
       '        cfg.output_root, "passive_scalar", cfg.run_name or "passive_scalar", cfg.overwrite\n'
       '    )')
src = src.replace(old, new, 1)

# 3) scalar solid bounce-back must use the D2Q5 helper, not the D2Q9 one
old_bb = '        # Zero flux at solid (bounce-back)\n        g = bounce_back_cells(g, solid)'
assert old_bb in src, "scalar bounce_back_cells call not found"
new_bb = '        # Zero flux at solid (D2Q5 bounce-back; NOT the D2Q9 momentum helper)\n        g = bounce_back_scalar(g, solid)'
src = src.replace(old_bb, new_bb, 1)

# 4) export the helper
if '"bounce_back_scalar"' not in src:
    src = src.replace('__all__ = [', '__all__ = [\n    "bounce_back_scalar",', 1) if '__all__ = [' in src else src

p.write_text(src)
print("patched", p)
print("has __all__:", '__all__' in src)
print("occurrences bounce_back_cells:", src.count("bounce_back_cells("))
print("occurrences bounce_back_scalar:", src.count("bounce_back_scalar("))