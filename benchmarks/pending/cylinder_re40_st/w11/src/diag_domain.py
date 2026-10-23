"""Domain-design diagnostics: decompose the +15% Cd puzzle into
(family-domain vs compact) x (sponge on/off). D=20, area convention."""

import sys

sys.path.insert(0, "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/src")
from pilot_lb_cpu import run

which = sys.argv[1]
if which == "fam":
    run(
        "fam_h20_sp10",
        20,
        "area",
        40.0,
        0.05,
        40.0,
        10.0,
        20.0,
        sponge_d=10.0,
        sponge_alpha=10.0,
        cap_tc=120.0,
        plateau_rel=0.0005,
        trace_path="trace_fam_h20_sp10.npz",
    )
    run(
        "fam_h20_sp0",
        20,
        "area",
        40.0,
        0.05,
        40.0,
        10.0,
        20.0,
        sponge_d=10.0,
        sponge_alpha=0.0,
        cap_tc=120.0,
        plateau_rel=0.0005,
        trace_path="trace_fam_h20_sp0.npz",
    )
else:
    run(
        "cmp_h8_sp0",
        20,
        "area",
        40.0,
        0.05,
        20.0,
        5.0,
        8.0,
        sponge_d=6.0,
        sponge_alpha=0.0,
        cap_tc=120.0,
        plateau_rel=0.0005,
        trace_path="trace_cmp_h8_sp0.npz",
    )
