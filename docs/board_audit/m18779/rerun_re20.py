import json,time,torch
torch.set_num_threads(8)
from tensorlbm.verification.cylinder_flow import run_re20_validation
t=time.time()
r=run_re20_validation()
r['wall_sec']=time.time()-t
json.dump(r,open('/root/tp-agent/work/m18779/xval_audit_re20.json','w'),indent=1)
print(json.dumps(r['results'],indent=1))
