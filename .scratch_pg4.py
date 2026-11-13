import subprocess
PGENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
def q(sql, db="tensorplatform"):
    r = subprocess.run(["runuser","-u","postgres","--","psql","-d",db,"-tA","-F","|","-c",sql],
                       capture_output=True, text=True, env=PGENV)
    return r.stdout if r.returncode == 0 else "ERR:" + r.stderr.strip()[:300]

print(q("select table_name from information_schema.tables where table_schema='public' and (table_name ilike '%node%' or table_name ilike '%plan%' or table_name ilike '%mission%') order by 1;"))
print("== oss_inventory cols")
print(q("select column_name from information_schema.columns where table_name='oss_inventory';"))