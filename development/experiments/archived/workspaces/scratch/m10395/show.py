import json, sys, glob
cid = sys.argv[1] if len(sys.argv) > 1 else "sweep_r1.00"
d = json.load(open(f"artifacts/p88w1/{cid}.json"))
for g, r in zip(d["pressure"]["gauges"], d["pressure"]["raw_gauges"]):
    print("row %2d y %+0.4f amp %7.2f lin %7.2f err %+6.1f%% rawmax %7.1f broadband %6.2f snr %5.1f" % (
        g["row"], g["y_from_swl_m"], g["p_measured_amp_Pa"], g["p_linear_RAO_amp_Pa"],
        g["error_vs_linear_pct"], r["raw_max_Pa"],
        r.get("broadband_rms_Pa", float("nan")), r.get("snr", float("nan"))))
print("runup", d["pressure"]["runup"]["amplitude"], "rawmax", d["pressure"]["runup"]["raw_max_m"],
      "vimp", d["pressure"]["impact_velocity_m_s"])
print("bounds", d["pressure"]["bounds"])
print("p_amp", d["pressure"]["p_amp_Pa"], "a1", d["a1"]["amplitude"], "a2", d["a2"]["amplitude"],
      "ref", d["reference"]["a_peak_half"])
print("mass drift", d["max_mass_drift"], "3gh", d["weak_compressibility_ratio"])