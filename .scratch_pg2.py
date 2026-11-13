import subprocess
PGENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
def q(sql, db="tensorplatform"):
    r = subprocess.run(["runuser","-u","postgres","--","psql","-d",db,"-tA","-F","|","-c",sql],
                       capture_output=True, text=True, env=PGENV)
    if r.returncode != 0:
        return "ERR:" + r.stderr.strip()[:300]
    return r.stdout

print("== kb same-domain (LBM / coexistence / calibration)")
print(q("""select id||' | '||ktype||' | '||coalesce(domain,'')||' | '||title
from knowledge_base where status='active'
and (domain ilike '%lbm%' or title ilike '%lbm%' or title ilike '%coexist%'
     or title ilike '%calibrat%' or title ilike '%free%energy%')
order by id desc limit 25"""))
print("== negative ledger")
print(q("""select id||' | '||coalesce(domain,'')||' | '||left(coalesce(title,''),80)
from negative_knowledge_ledger order by id desc limit 20"""))
print("== oss_solutions LBM")
print(q("""select id||' | '||coalesce(repo,'')||' | '||coalesce(feature_slug,'')||' | '||coalesce(transfer,'')
from oss_solutions where repo ilike '%lbm%' or feature_slug ilike '%coexist%' or feature_slug ilike '%calib%'
order by id desc limit 30"""))
print("== tables like oss")
print(q("select table_name from information_schema.tables where table_name ilike '%oss%' or table_name ilike '%negative%' or table_name ilike '%knowledge%';"))