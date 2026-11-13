import subprocess
PGENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
def q(sql, db="tensorplatform"):
    r = subprocess.run(["runuser","-u","postgres","--","psql","-d",db,"-tA","-F","|","-c",sql],
                       capture_output=True, text=True, env=PGENV)
    return r.stdout if r.returncode == 0 else "ERR:" + r.stderr.strip()[:300]

print("== plan node")
print(q("""select id||' | '||left(coalesce(name,''),80)||' | '||coalesce(status,'')||' || '||left(coalesce(goal,''),1500)
from campaign_plan_nodes where name ilike '%R4-COEXISTENCE%'"""))
print("== oss_inventory multiphase entries")
print(q("""select id||' :: '||repo||' | '||tier||' | '||left(coalesce(focus,''),90)||' | methods='||left(coalesce(methods,''),90)||' | multiphase='||left(coalesce(multiphase,''),120)
from oss_inventory where repo ilike '%lbm%' order by id desc limit 40"""))