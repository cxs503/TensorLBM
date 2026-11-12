import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S="/root/tp-agent/work/m15399/artifacts/opt_loop_tau_recovery.json"
d=json.load(open(S)); a=json.load(open("docs/board_audit/m15449/audit_evidence.json"))
tracks=[t["track"] for t in d["tracks"]]
err=[max(abs(r["rel_err_pct"]) for r in t["runs"]) for t in d["tracks"]]
delta={c["track"]:c["err_max_abs_delta_pct"] for c in a["C3_report_table_vs_json"]["rows"] if "err_max_abs_delta_pct" in c}
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
c=[("#c0392b" if t in d["summary"]["control_arms"] else "#2471a3") for t in tracks]
ax[0].bar(tracks,err,color=c); ax[0].set_yscale("log")
ax[0].axhline(3.0,ls="--",color="k",label="mission gate 3%")
for i,v in enumerate(err): ax[0].text(i,v*1.15,f"{v:.3g}",ha="center",fontsize=8)
ax[0].set_ylabel("max |rel err| of Re [%] (log)"); ax[0].set_title("A. artifact error vs pre-declared 3% gate (m15399)")
ax[0].tick_params(axis="x",rotation=35); ax[0].legend()
ax[1].bar(list(delta),list(delta.values()),color="#27ae60"); ax[1].set_yscale("log")
ax[1].set_ylabel("|report table - JSON| [pp]"); ax[1].set_title("B. zero-hand-copy probe: report Delta re-derived from JSON")
ax[1].tick_params(axis="x",rotation=35)
for i,(k,v) in enumerate(delta.items()): ax[1].text(i,max(v,1e-9)*1.4,f"{v:.2g}",ha="center",fontsize=8)
plt.tight_layout(); plt.savefig("docs/board_audit/m15449/fig_audit_m15449.png",dpi=130)
print("ok")
