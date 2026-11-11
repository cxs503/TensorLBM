import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
s=json.load(open("corrected_summary/summary_recomputed.json")); d=json.load(open("refs/artifacts/runs/summary.json"))
fig,ax=plt.subplots(1,2,figsize=(13,5))
# left: err% per Ra by source
labs=[];vals=[];cols=[]
for ra,e in s["by_ra"].items():
    for row in e["direct"]:
        labs.append(f"Ra={ra}\nNy={row['ny']}"); vals.append(row["err_pct"])
        cols.append("tab:blue" if row["source"]=="this_work" else "tab:gray")
ex=[("Ra=1e4\nextrap",s["by_ra"]["1e4"]["extrapolation"]["ma0.2"]["err_pct"])]
for k,v in ex: labs.append(k); vals.append(v); cols.append("tab:green")
ax[0].bar(range(len(vals)),vals,color=cols,edgecolor="k",lw=.5)
ax[0].axhline(5,color="tab:red",ls="--",lw=1.6,label="predeclared tol 5%")
ax[0].axhline(-5,color="tab:red",ls="--",lw=1.6)
ax[0].set_xticks(range(len(labs))); ax[0].set_xticklabels(labs,fontsize=7.5)
ax[0].set_ylabel("|Nu_LBM - Nu_DVD1983| / Nu_DVD  [%]")
ax[0].set_title("A. Bias vs de Vahl Davis 1983 (negative = under-prediction)\nblue=this mission re-run, gray=prior artifact P96W3-N04 quoted")
ax[0].bar(range(len(vals)),[-v for v in vals],color=cols,edgecolor="k",lw=.5,alpha=.35)
ax[0].legend(fontsize=8)
# right: Nu vs Ra
nus=[1.0,2.0,3.0,4.0]; nus=[]
xs=[]; ys=[]; cs=[]
for ra,e in s["by_ra"].items():
    for row in e["direct"]:
        xs.append(float(ra)); ys.append(row["nu"]); cs.append("tab:blue" if row["source"]=="this_work" else "tab:gray")
ax[1].scatter(xs,ys,c=cs,s=55,label="LBM ladder points")
ax[1].scatter([float(k) for k in s["by_ra"]],[e["nu_ref"] for e in s["by_ra"].values()],marker="*",s=260,c="tab:red",label="de Vahl Davis 1983")
ax[1].set_xscale("log"); ax[1].set_xlabel("Ra"); ax[1].set_ylabel("mean Nu (hot wall)")
ax[1].set_title("B. Nusselt vs Rayleigh: ladder points vs reference table")
ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_audit_m15212_nu_error.png",dpi=140)
print("saved")
