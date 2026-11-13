import subprocess
PGENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
def q(sql, db="tensorplatform"):
    r = subprocess.run(["runuser","-u","postgres","--","psql","-d",db,"-tA","-F","|","-c",sql],
                       capture_output=True, text=True, env=PGENV)
    return r.stdout if r.returncode == 0 else "ERR:" + r.stderr.strip()[:300]

print("== anchor node XVAL-LBM-R4-COEXISTENCE")
print(q("""select id||' | '||left(coalesce(name,''),90)||' | '||coalesce(status,'')||' | '||left(coalesce(goal,''),900)
from nodes where name ilike '%R4-COEXISTENCE%'"""))
print("== oss_inventory liblld / coexistence sources")
print(q("""select id||' | '||left(coalesce(repo,''),40)||' | '||left(coalesce(feature_slug,''),60)||' | '||left(coalesce(note,''),200)
from oss_inventory where feature_slug ilike '%coexist%' or note ilike '%coexistence%' or feature_slug ilike '%calib%' limit 20"""))
print("== kb coexistence/free-energy lessons")
print(q("""select id||' | '||ktype||' | '||left(coalesce(title,''),80)
from knowledge_base where status='active' and (title ilike '%coexist%' or title ilike '%free-energy%' or title ilike '%free energy%' or title ilike '%calib%') order by id desc limit 20"""))
print("== negative ledger cols")
print(q("select column_name from information_schema.columns where table_name='negative_knowledge_ledger';"))
print("== negative rows LBM")
print(q("""select id||' | '||left(coalesce(title,''),110) from negative_knowledge_ledger
where title ilike '%lbm%' or coalesce(keywords,'') ilike '%lbm%' order by id desc limit 15"""))