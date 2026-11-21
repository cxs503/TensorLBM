"""R3 metric-definition work (m21017).

Finding (measured, not assumed): the frozen R1 metric ``front_cells()/a`` counts
the *empty* one-cell INTERFACE envelope that ``init_flags_from_fill_2d`` puts in
front of the liquid block, so ``Z(t=0) = (a+1)/a`` instead of 1.0 -- a +1 cell
(+1/a in Z, i.e. +6.25 % at a=16) high bias.  ``resolution_study.py``'s
``Z_cell`` (fill > 1e-6) is the physical front and gives exactly 1.0.

Both definitions are now recorded side by side instead of silently picking one:

* ``run_xval.py``      keeps ``Z`` (legacy, R1-comparable) and adds ``Z_fill``;
* ``resolution_study.py``  replaces the (wrong) invented ``Z_cell0`` with
  ``Z_cell_flag`` = the legacy flag-inclusive column, so the R2/R3 ladder error
  can be re-read under both conventions and the metric-definition error is
  quantified rather than assumed.
"""
from pathlib import Path

# ---------------------------------------------------------------- run_xval.py
p = Path("xval/oracle_minimal_d2q9_fs/run_xval.py")
s = p.read_text()
old = '''            fluid = (flags == LIQUID) | (flags == INTERFACE)
            cols = torch.nonzero(fluid.any(dim=0)).flatten()
            front = int(cols.max().item()) if cols.numel() else 0  # 0-based, wall at 0
            series.append({"step": step, "T": step * dtT, "Z": front / a})'''
new = '''            fluid = (flags == LIQUID) | (flags == INTERFACE)
            cols = torch.nonzero(fluid.any(dim=0)).flatten()
            front = int(cols.max().item()) if cols.numel() else 0  # 0-based, wall at 0
            # physical front: rightmost column holding actual liquid mass.  The
            # flag-inclusive ``Z`` above also counts the *empty* one-cell
            # INTERFACE envelope placed in front of the block, i.e. it reads
            # (a+1)/a at t = 0 (+1 cell bias, +1/a in Z).  Both are recorded:
            # ``Z`` stays the R1-comparable legacy value, ``Z_fill`` is the
            # corrected one (Z_fill(t=0) = 1 exactly).
            cols_fill = torch.nonzero((fill > 1e-6).any(dim=0)).flatten()
            front_fill = int(cols_fill.max().item()) if cols_fill.numel() else 0
            series.append({"step": step, "T": step * dtT, "Z": front / a,
                           "Z_fill": front_fill / a})'''
assert old in s
p.write_text(s.replace(old, new))
print("patched", p)

# ------------------------------------------------------- resolution_study.py
p = Path("xval/oracle_minimal_d2q9_fs/resolution_study.py")
s = p.read_text()

s = s.replace('''                     "Z_cell0": "Z_cell - 1/a: same column index with the leading "
                                "INTERFACE layer removed (x_front = x_rightmost - 1); "
                                "= 1 exactly at t = 0 by construction",''',
              '''                     "Z_cell_flag": "rightmost LIQUID|INTERFACE column / a (legacy "
                                    "R1 run_xval convention). It also counts the empty "
                                    "one-cell INTERFACE envelope in front of the block, "
                                    "so Z_cell_flag(t=0) = (a+1)/a: a +1 cell (+1/a) high "
                                    "bias, recorded to quantify the metric-definition "
                                    "error instead of assuming it",''')

s = s.replace('''def snapshot_metrics(fill: np.ndarray, a: float, h0: float) -> dict:
    depth = depth_profile(fill)
    xr = front_cells(fill)
    out = {"Z_cell": xr / a,
           # the initial block spans x in [0, a+1): the leading cell is the
           # INTERFACE layer, so the liquid column ends at x_rightmost - 1 and
           # Z_cell0 = 1 at t = 0 (documented convention, not a fit).
           "Z_cell0": max(xr - DESIGN["interface_offset_cells"], 0.0) / a}''',
              '''def snapshot_metrics(fill: np.ndarray, a: float, h0: float,
                     fluid_mask: np.ndarray | None = None) -> dict:
    """Front metrics for one snapshot.

    ``Z_cell``  = rightmost column with liquid content (fill > 1e-6), the
                  physical column front: exactly 1 at t = 0.
    ``Z_cell_flag`` = rightmost LIQUID|INTERFACE column (legacy R1 convention,
                  computed only when ``fluid_mask`` is supplied); it counts the
                  empty interface envelope and reads (a+1)/a at t = 0.
    ``Z_sub*``  = sub-cell toe at depth threshold f*h0.
    """
    depth = depth_profile(fill)
    out = {"Z_cell": front_cells(fill) / a}
    if fluid_mask is not None:
        cols = np.flatnonzero(np.asarray(fluid_mask).any(axis=0))
        out["Z_cell_flag"] = (float(cols[-1]) if cols.size else 0.0) / a''')

# thread the fluid mask through the two arms
s = s.replace('''            m = snapshot_metrics(fill_now.cpu().numpy(), float(a), float(h0))
            m.update({"step": step, "T": step * dtT})
            series.append(m)
            inv_tr.append({"step": step,
                           "drift_pct": 100.0 * (total_liquid_inventory_2d(f, mass, flags)
                                                 - inv0) / inv0})''',
              '''            m = snapshot_metrics(fill_now.cpu().numpy(), float(a), float(h0),
                                 fluid_mask=((flags == LIQUID) | (flags == INTERFACE)
                                             ).cpu().numpy())
            m.update({"step": step, "T": step * dtT})
            series.append(m)
            inv_tr.append({"step": step,
                           "drift_pct": 100.0 * (total_liquid_inventory_2d(f, mass, flags)
                                                 - inv0) / inv0})''')
s = s.replace('''            m = snapshot_metrics(fill, float(a), float(h0))
            m.update({"step": step, "T": step * dtT})''',
              '''            m = snapshot_metrics(fill, float(a), float(h0),
                                 fluid_mask=(st.flags() != P.GAS))
            m.update({"step": step, "T": step * dtT})''')

s = s.replace('SER_KEYS = ("Z_cell", "Z_cell0", "Z_sub0.02", "Z_sub0.05", "Z_sub0.1")',
              'SER_KEYS = ("Z_cell", "Z_cell_flag", "Z_sub0.02", "Z_sub0.05", "Z_sub0.1")')
s = s.replace('''    """One [T, Z_cell, Z_cell0, Z_sub.02, Z_sub.05, Z_sub.10] row (log-safe)."""''',
              '''    """One [T, Z_cell, Z_cell_flag, Z_sub.02, Z_sub.05, Z_sub.10] row (log-safe)."""''')
s = s.replace('keys = ["Z_cell", "Z_cell0"] + [f"Z_sub{f:g}" for f in THRESH_FRACS]',
              'keys = ["Z_cell", "Z_cell_flag"] + [f"Z_sub{f:g}" for f in THRESH_FRACS]')
p.write_text(s)
print("patched", p)
print("Z_cell0 occurrences left:", s.count("Z_cell0"))