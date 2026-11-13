import subprocess
PGENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
def q(sql, db="tensorplatform"):
    r = subprocess.run(["runuser","-u","postgres","--","psql","-d",db,"-tA","-F","|","-c",sql],
                       capture_output=True, text=True, env=PGENV)
    return r.stdout if r.returncode == 0 else "ERR:" + r.stderr.strip()[:300]

print("== oss_inventory coexistence hit")
print(q("""select id||' :: '||repo from oss_inventory
where (methods||coalesce(multiphase,'')||coalesce(boundary,'')||coalesce(summary,'')) ilike '%coexist%'"""))
print("== oss_solutions coexistence hit")
print(q("""select id||' | '||repo||' | '||feature_slug from oss_solutions
where (coalesce(transfer,'')||' '||coalesce(feature_slug,'')) ilike '%coexist%'"""))
print("== campaign_plan_nodes cols")
print(q("select column_name from information_schema.columns where table_name='campaign_plan_nodes';"))