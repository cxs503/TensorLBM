import json

d = json.load(open("/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/geom.json"))
print("== OLD channel (Schaefer-Turek, old run.py geometry) ==")
for k, v in sorted(d["old_channel"].items()):
    print(
        k,
        "R_nom=%.1f" % v["mask_R"],
        "R_row=%.2f" % v["R_row"],
        "row_drift=%+.4f%%" % v["drift_R_row_pct"],
        "area_drift=%+.4f%%" % v["drift_R_area_pct"],
        "N_solid=%d" % v["n_solid"],
        "Re_eff/Re_nom(row)=%.5f" % v["Re_eff_over_Re_nominal_row"],
        "Re_eff/Re_nom(area)=%.5f" % v["Re_eff_over_Re_nominal_area"],
    )
print("== NEW free-stream ladder ==")
for k, v in sorted(d["new_freestream"].items()):
    print(
        k,
        "R=%.1f" % v["R_nominal"],
        "R_row=%.2f" % v["mask_R_equals_nominal"]["R_row"],
        "row_drift=%+.4f%%" % v["drift_R_row_pct"],
        "area_drift=%+.4f%%" % v["drift_R_area_pct"],
        "arealock_res=%+.5f%%" % v["area_lock"]["residual_pct"],
        "rowlock_R_row=%.2f" % v["row_lock"]["R_row"],
        "rowlock_area_drift=%+.4f%%" % ((v["row_lock"]["R_area"] / v["R_nominal"] - 1) * 100),
    )
