import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
B=json.load(open("runs/m18650/audit_bundle/benchmark.json"))
well=[r for r in B["rows"] if r["id"] not in ("bptt-gradient-vs-fd","holdout-ke-selected-vs-filtered-ref")]
ids=[r["id"] for r in well]; errs=[100*r["rel_err"] for r in well]
fig,ax=plt.subplots(1,2,figsize=(12,4.4))
ax[0].barh(ids,errs,color="#3b7dd8"); ax[0].axvline(15,color="r",ls="--",label="owner bar 15%")
ax[0].set_xlabel("relative error vs ref [%]"); ax[0].set_title("m18462 benchmark rows (8 well-formed, auditor recomputed)")
for i,e in enumerate(errs): ax[0].text(e+0.02,i,f"{e:.3f}%",va="center",fontsize=7)
ax[0].legend(fontsize=8)
lab=["exact","128^2 BGK","256^2 BGK","32^2 BGK","LES sel.","no-model"]
val=[well[0]["ref"],2.1253025573097425e-3,2.131038483842912e-3,2.1006477005317116e-3,2.1006429539644116e-3,2.1006477005317114e-3]
ax[1].bar(lab,val,color=["k","#3b7dd8","#3b7dd8","#8ab4f8","#e07b39","#7f7f7f"])
ax[1].set_ylabel("kinetic energy K(t=200)"); ax[1].set_title("K: exact TGV anchor vs grids / selected LES")
ax[1].tick_params(axis="x",labelsize=8)
for i,v in enumerate(val): ax[1].text(i,v,f"{v:.6e}",ha="center",va="bottom",fontsize=6,rotation=90)
plt.tight_layout(); plt.savefig("docs/board_audit/m18650/fig_audit_m18650.png",dpi=140)
print("fig ok", len(well))
